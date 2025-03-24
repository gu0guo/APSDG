import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl.function as fn
from dgl.nn.pytorch import GraphConv
from geoopt.manifolds.stereographic import PoincareBall
from geoopt.manifolds.sphere import Sphere


class APSDG(nn.Module):
    """
    自适应积空间离散动态图链接预测模型（APSDG）

    根据论文：自适应积空间离散动态图链接预测模型
    该模型通过结合欧几里得空间、双曲空间和超球面空间，构建积空间作为嵌入空间，
    以更好地适应数据的复杂结构特征，并使用强化学习机制动态调整各空间的维度比例和曲率参数。
    """

    def __init__(self, snapshot_graphs, config, device):
        super(APSDG, self).__init__()

        # 基本参数设置
        self.num_nodes = snapshot_graphs[0].num_nodes()
        self.hidden_dim = config.get('hidden_dim', 64)
        self.embedding_dim = config.get('embedding_dim', 64)
        self.history_window = config.get('history_window', 5)
        self.device = device
        self.num_layers = config.get('num_layers', 2)

        # 初始维度比例（可学习）
        e_ratio = config.get('e_ratio', 1 / 3)
        b_ratio = config.get('b_ratio', 1 / 3)
        s_ratio = config.get('s_ratio', 1 / 3)

        # 初始化比例参数（使用log空间确保正值）
        self.log_e_ratio = nn.Parameter(torch.tensor(np.log(e_ratio), device=device))
        self.log_b_ratio = nn.Parameter(torch.tensor(np.log(b_ratio), device=device))
        self.log_s_ratio = nn.Parameter(torch.tensor(np.log(s_ratio), device=device))

        # 更新维度比例
        self._update_dimension_ratios()

        # 曲率参数（可学习）
        self.b_curvature = nn.Parameter(torch.tensor([config.get('b_curvature', -1.0)], device=device))
        self.s_curvature = nn.Parameter(torch.tensor([config.get('s_curvature', 1.0)], device=device))

        # 初始化各空间流形
        self.update_manifolds()

        # 节点嵌入
        self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, self.embedding_dim, device=device))

        # 消息传递层
        self.message_passing_layers = nn.ModuleList([
            ProductSpaceLayer(self.embedding_dim) for _ in range(self.num_layers)
        ])

        # 历史信息融合模块
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=config.get('num_heads', 1),
            batch_first=True,
            device=device
        )

        # HyperGRU模块
        self.gru_cell = nn.GRUCell(
            input_size=self.embedding_dim,
            hidden_size=self.embedding_dim,
            device=device
        )

        # 强化学习参数
        self.rl_alpha = config.get('rl_alpha', 0.01)

        # 历史存储
        self.history_embeddings = []

    def _update_dimension_ratios(self):
        """更新各空间的维度比例"""
        # 计算归一化的比例
        e_ratio = torch.exp(self.log_e_ratio)
        b_ratio = torch.exp(self.log_b_ratio)
        s_ratio = torch.exp(self.log_s_ratio)
        total = e_ratio + b_ratio + s_ratio

        e_ratio = e_ratio / total
        b_ratio = b_ratio / total
        s_ratio = s_ratio / total

        # 计算各维度
        self.e_dim = max(1, int(self.embedding_dim * e_ratio.item()))
        self.b_dim = max(1, int(self.embedding_dim * b_ratio.item()))
        self.s_dim = max(1, int(self.embedding_dim * s_ratio.item()))

        # 调整，确保总和为embedding_dim
        total_dim = self.e_dim + self.b_dim + self.s_dim
        if total_dim != self.embedding_dim:
            dim_diff = self.embedding_dim - total_dim
            # 将差异添加到最大的维度
            max_dim_idx = torch.argmax(torch.tensor([e_ratio, b_ratio, s_ratio])).item()
            if max_dim_idx == 0:
                self.e_dim += dim_diff
            elif max_dim_idx == 1:
                self.b_dim += dim_diff
            else:
                self.s_dim += dim_diff

    def update_manifolds(self):
        """更新几何流形"""
        self.euclidean = torch.nn.Identity()  # 欧几里得空间直接使用恒等变换
        self.poincare = PoincareBall(c=self.b_curvature.abs())
        self.sphere = Sphere(c=self.s_curvature)

    def update_rl_params(self, reward):
        """基于强化学习更新维度比例"""
        with torch.no_grad():
            # 基于梯度和奖励更新参数
            if self.log_e_ratio.grad is not None:
                self.log_e_ratio.data += self.rl_alpha * reward * self.log_e_ratio.grad
            if self.log_b_ratio.grad is not None:
                self.log_b_ratio.data += self.rl_alpha * reward * self.log_b_ratio.grad
            if self.log_s_ratio.grad is not None:
                self.log_s_ratio.data += self.rl_alpha * reward * self.log_s_ratio.grad

            # 更新维度比例
            self._update_dimension_ratios()

            # 更新流形
            self.update_manifolds()

    def forward(self, graph, is_training=True):
        """前向传播"""
        # 分割节点嵌入到不同空间
        e_emb = self.node_embeddings[:, :self.e_dim]
        b_emb = self.node_embeddings[:, self.e_dim:self.e_dim + self.b_dim]
        s_emb = self.node_embeddings[:, self.e_dim + self.b_dim:]

        # 消息传递
        for layer in self.message_passing_layers:
            e_emb, b_emb, s_emb = layer(graph, e_emb, b_emb, s_emb, self.poincare, self.sphere)

        # 拼接嵌入
        current_emb = torch.cat([e_emb, b_emb, s_emb], dim=1)

        # 历史信息融合
        if len(self.history_embeddings) >= self.history_window:
            # 堆叠历史嵌入
            history_tensor = torch.stack(self.history_embeddings[-self.history_window:], dim=0).transpose(0,
                                                                                                          1)  # [nodes, window, dim]

            # 注意力机制
            query = current_emb.unsqueeze(1)  # [nodes, 1, dim]
            context, _ = self.attention(query, history_tensor, history_tensor)
            context = context.squeeze(1)  # [nodes, dim]

            # HyperGRU更新
            updated_emb = self.gru_cell(current_emb, context)
        else:
            updated_emb = current_emb

        # 更新历史嵌入
        if is_training:
            self.history_embeddings.append(updated_emb.detach())
            if len(self.history_embeddings) > self.history_window:
                self.history_embeddings.pop(0)

        return updated_emb

    def predict_link(self, src_nodes, dst_nodes):
        """链接预测"""
        # 获取节点嵌入
        src_emb = self.node_embeddings[src_nodes]
        dst_emb = self.node_embeddings[dst_nodes]

        # 分割嵌入
        src_e = src_emb[:, :self.e_dim]
        src_b = src_emb[:, self.e_dim:self.e_dim + self.b_dim]
        src_s = src_emb[:, self.e_dim + self.b_dim:]

        dst_e = dst_emb[:, :self.e_dim]
        dst_b = dst_emb[:, self.e_dim:self.e_dim + self.b_dim]
        dst_s = dst_emb[:, self.e_dim + self.b_dim:]

        # 计算各空间距离
        e_dist = torch.norm(src_e - dst_e, dim=1)
        b_dist = self.poincare.dist(src_b, dst_b)
        s_dist = self.sphere.dist(src_s, dst_s)

        # 按维度比例加权
        e_weight = self.e_dim / self.embedding_dim
        b_weight = self.b_dim / self.embedding_dim
        s_weight = self.s_dim / self.embedding_dim

        # 费米-狄拉克解码器
        r = 2.0  # 默认超参数
        s = 1.0  # 默认超参数

        combined_dist = e_weight * e_dist + b_weight * b_dist + s_weight * s_dist
        scores = torch.exp(-combined_dist / s) ** r

        return scores


class ProductSpaceLayer(nn.Module):
    """积空间消息传递层"""

    def __init__(self, embedding_dim):
        super(ProductSpaceLayer, self).__init__()
        # 在每个空间内进行特征转换的线性层
        self.linear_e = nn.Linear(embedding_dim, embedding_dim)
        self.linear_b = nn.Linear(embedding_dim, embedding_dim)
        self.linear_s = nn.Linear(embedding_dim, embedding_dim)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, graph, e_emb, b_emb, s_emb, poincare, sphere):
        """前向传播"""
        # 欧几里得空间消息传递
        with graph.local_scope():
            graph.ndata['h_e'] = e_emb
            graph.update_all(fn.copy_u('h_e', 'm'), fn.mean('m', 'neigh_e'))
            e_neigh = graph.ndata['neigh_e']
            e_out = self.linear_e(e_neigh)
            e_out = self.activation(e_out)

        # 双曲空间消息传递
        with graph.local_scope():
            # 映射到切空间
            origin = torch.zeros_like(b_emb[0]).unsqueeze(0).to(b_emb.device)
            b_tangent = poincare.logmap(origin, b_emb)

            # 线性变换
            b_transformed = self.linear_b(b_tangent)

            # 消息传递
            graph.ndata['h_b'] = b_transformed
            graph.update_all(fn.copy_u('h_b', 'm'), fn.mean('m', 'neigh_b'))
            b_neigh = graph.ndata['neigh_b']

            # 映射回双曲空间
            b_out = poincare.expmap(origin, b_neigh)

        # 超球面空间消息传递
        with graph.local_scope():
            # 确保在球面上
            s_emb = F.normalize(s_emb, p=2, dim=1)

            # 变换
            s_transformed = self.linear_s(s_emb)

            # 消息传递
            graph.ndata['h_s'] = s_transformed
            graph.update_all(fn.copy_u('h_s', 'm'), fn.mean('m', 'neigh_s'))
            s_neigh = graph.ndata['neigh_s']

            # 再次归一化确保在球面上
            s_out = F.normalize(s_neigh, p=2, dim=1)

        return e_out, b_out, s_out


# 基准模型的简化实现
class HAT(nn.Module):
    """Hyperbolic Attention Network"""

    def __init__(self, snapshot_graphs, config, device):
        super(HAT, self).__init__()
        self.num_nodes = snapshot_graphs[0].num_nodes()
        self.embedding_dim = config.get('embedding_dim', 64)
        self.device = device

        # 节点嵌入
        self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, self.embedding_dim, device=device))

        # 双曲空间
        self.curvature = nn.Parameter(torch.tensor([config.get('curvature', -1.0)], device=device))
        self.poincare = PoincareBall(c=self.curvature.abs())

        # 注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=1,
            batch_first=True,
            device=device
        )

        # 图卷积层
        self.gc = GraphConv(self.embedding_dim, self.embedding_dim, activation=F.relu)

    def forward(self, graph, *args, **kwargs):
        with graph.local_scope():
            # 更新流形
            self.poincare = PoincareBall(c=self.curvature.abs())

            # 图卷积
            h = self.gc(graph, self.node_embeddings)

            # 使用注意力机制
            query = h.unsqueeze(1)
            key = h.unsqueeze(1)
            value = h.unsqueeze(1)

            h, _ = self.attention(query, key, value)
            h = h.squeeze(1)

            return h

    def predict_link(self, src_nodes, dst_nodes):
        src_emb = self.node_embeddings[src_nodes]
        dst_emb = self.node_embeddings[dst_nodes]
        dist = self.poincare.dist(src_emb, dst_emb)
        return torch.exp(-dist)


class HGCN(nn.Module):
    """Hyperbolic Graph Convolutional Network"""

    def __init__(self, snapshot_graphs, config, device):
        super(HGCN, self).__init__()
        self.num_nodes = snapshot_graphs[0].num_nodes()
        self.embedding_dim = config.get('embedding_dim', 64)
        self.device = device

        # 节点嵌入
        self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, self.embedding_dim, device=device))

        # 双曲空间
        self.curvature = nn.Parameter(torch.tensor([config.get('curvature', -1.0)], device=device))
        self.poincare = PoincareBall(c=self.curvature.abs())

        # 图卷积层
        self.gc1 = GraphConv(self.embedding_dim, self.embedding_dim, activation=F.relu)
        self.gc2 = GraphConv(self.embedding_dim, self.embedding_dim)

    def forward(self, graph, *args, **kwargs):
        # 更新流形
        self.poincare = PoincareBall(c=self.curvature.abs())

        # 图卷积
        h = self.gc1(graph, self.node_embeddings)
        h = self.gc2(graph, h)

        return h

    def predict_link(self, src_nodes, dst_nodes):
        src_emb = self.node_embeddings[src_nodes]
        dst_emb = self.node_embeddings[dst_nodes]
        dist = self.poincare.dist(src_emb, dst_emb)
        return torch.exp(-dist)


class EvolveGCN(nn.Module):
    """EvolveGCN模型"""

    def __init__(self, snapshot_graphs, config, device):
        super(EvolveGCN, self).__init__()
        self.num_nodes = snapshot_graphs[0].num_nodes()
        self.embedding_dim = config.get('embedding_dim', 64)
        self.device = device

        # 节点嵌入
        self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, self.embedding_dim, device=device))

        # 图卷积层
        self.gc1 = GraphConv(self.embedding_dim, self.embedding_dim, activation=F.relu)
        self.gc2 = GraphConv(self.embedding_dim, self.embedding_dim)

        # 权重更新GRU
        self.weight_rnn = nn.GRUCell(
            input_size=self.embedding_dim * self.embedding_dim,
            hidden_size=self.embedding_dim * self.embedding_dim,
            device=device
        )

        # 历史权重
        self.gc1_weights = None
        self.gc2_weights = None

    def forward(self, graph, is_training=True):
        # 更新权重
        if is_training and self.gc1_weights is not None:
            # 展平权重
            gc1_flat = self.gc1.weight.view(-1)
            gc2_flat = self.gc2.weight.view(-1)

            # 使用GRU更新
            gc1_flat = self.weight_rnn(self.gc1_weights, gc1_flat)
            gc2_flat = self.weight_rnn(self.gc2_weights, gc2_flat)

            # 重塑权重
            self.gc1.weight.data = gc1_flat.view_as(self.gc1.weight.data)
            self.gc2.weight.data = gc2_flat.view_as(self.gc2.weight.data)

        # 图卷积
        h = self.gc1(graph, self.node_embeddings)
        h = self.gc2(graph, h)

        # 保存当前权重
        if is_training:
            self.gc1_weights = self.gc1.weight.detach().view(-1)
            self.gc2_weights = self.gc2.weight.detach().view(-1)

        return h

    def predict_link(self, src_nodes, dst_nodes):
        src_emb = self.node_embeddings[src_nodes]
        dst_emb = self.node_embeddings[dst_nodes]
        return torch.sigmoid(torch.sum(src_emb * dst_emb, dim=1))


class GRUGCN(nn.Module):
    """GRU-GCN模型"""

    def __init__(self, snapshot_graphs, config, device):
        super(GRUGCN, self).__init__()
        self.num_nodes = snapshot_graphs[0].num_nodes()
        self.embedding_dim = config.get('embedding_dim', 64)
        self.device = device

        # 节点嵌入
        self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, self.embedding_dim, device=device))

        # 图卷积层
        self.gc = GraphConv(self.embedding_dim, self.embedding_dim, activation=F.relu)

        # GRU层
        self.gru = nn.GRUCell(
            input_size=self.embedding_dim,
            hidden_size=self.embedding_dim,
            device=device
        )

        # 隐藏状态
        self.hidden = None

    def forward(self, graph, is_training=True):
        # 图卷积
        h = self.gc(graph, self.node_embeddings)

        # GRU更新
        if self.hidden is None:
            self.hidden = torch.zeros_like(h)

        h = self.gru(h, self.hidden)

        # 更新隐藏状态
        if is_training:
            self.hidden = h.detach()

        return h

    def predict_link(self, src_nodes, dst_nodes):
        src_emb = self.node_embeddings[src_nodes]
        dst_emb = self.node_embeddings[dst_nodes]
        return torch.sigmoid(torch.sum(src_emb * dst_emb, dim=1))


class HTGN(nn.Module):
    """Hyperbolic Temporal Graph Networks"""

    def __init__(self, snapshot_graphs, config, device):
        super(HTGN, self).__init__()
        self.num_nodes = snapshot_graphs[0].num_nodes()
        self.embedding_dim = config.get('embedding_dim', 64)
        self.device = device

        # 节点嵌入
        self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, self.embedding_dim, device=device))

        # 双曲空间
        self.curvature = nn.Parameter(torch.tensor([config.get('curvature', -1.0)], device=device))
        self.poincare = PoincareBall(c=self.curvature.abs())

        # 图卷积层
        self.gc = GraphConv(self.embedding_dim, self.embedding_dim)

        # GRU层
        self.gru = nn.GRUCell(
            input_size=self.embedding_dim,
            hidden_size=self.embedding_dim,
            device=device
        )

        # 注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=1,
            batch_first=True,
            device=device
        )

        # 历史嵌入
        self.history_embs = []

    def forward(self, graph, is_training=True):
        # 更新流形
        self.poincare = PoincareBall(c=self.curvature.abs())

        # 图卷积获取当前快照嵌入
        h = self.gc(graph, self.node_embeddings)

        # 如果有历史嵌入，使用注意力和GRU
        if len(self.history_embs) > 0:
            # 堆叠历史嵌入
            history = torch.stack(self.history_embs, dim=0).transpose(0, 1)

            # 注意力
            query = h.unsqueeze(1)
            context, _ = self.attention(query, history, history)
            context = context.squeeze(1)

            # GRU更新
            h = self.gru(h, context)

        # 更新历史
        if is_training:
            self.history_embs.append(h.detach())
            if len(self.history_embs) > 10:
                self.history_embs.pop(0)

        return h

    def predict_link(self, src_nodes, dst_nodes):
        src_emb = self.node_embeddings[src_nodes]
        dst_emb = self.node_embeddings[dst_nodes]
        dist = self.poincare.dist(src_emb, dst_emb)
        return torch.exp(-dist)


class HGWaveNet(nn.Module):
    """Hyperbolic Graph WaveNet"""

    def __init__(self, snapshot_graphs, config, device):
        super(HGWaveNet, self).__init__()
        self.num_nodes = snapshot_graphs[0].num_nodes()
        self.embedding_dim = config.get('embedding_dim', 64)
        self.device = device

        # 节点嵌入
        self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, self.embedding_dim, device=device))

        # 双曲空间
        self.curvature = nn.Parameter(torch.tensor([config.get('curvature', -1.0)], device=device))
        self.poincare = PoincareBall(c=self.curvature.abs())

        # 图卷积层
        self.gc = GraphConv(self.embedding_dim, self.embedding_dim)

        # 时间卷积
        self.tcn = nn.Conv1d(
            in_channels=self.embedding_dim,
            out_channels=self.embedding_dim,
            kernel_size=3,
            padding=1,
            device=device
        )

        # 历史嵌入
        self.history_embs = []

    def forward(self, graph, is_training=True):
        # 更新流形
        self.poincare = PoincareBall(c=self.curvature.abs())

        # 图卷积
        h = self.gc(graph, self.node_embeddings)

        # 如果有足够历史，使用时间卷积
        if len(self.history_embs) >= 3:
            # 准备时间卷积输入
            recent = self.history_embs[-3:] + [h]
            time_seq = torch.stack(recent, dim=2)  # [nodes, dim, time]

            # 时间卷积
            h_tcn = self.tcn(time_seq)
            h = h_tcn[:, :, -1]  # 取最后一个时间点

        # 更新历史
        if is_training:
            self.history_embs.append(h.detach())
            if len(self.history_embs) > 10:
                self.history_embs.pop(0)

        return h

    def predict_link(self, src_nodes, dst_nodes):
        src_emb = self.node_embeddings[src_nodes]
        dst_emb = self.node_embeddings[dst_nodes]
        dist = self.poincare.dist(src_emb, dst_emb)
        return torch.exp(-dist)


def load_model(model_name, snapshot_graphs, config, device):
    """加载指定模型"""
    models = {
        'apsdg': APSDG,
        'hat': HAT,
        'hgcn': HGCN,
        'evolvegcn': EvolveGCN,
        'grugcn': GRUGCN,
        'htgn': HTGN,
        'hgwavenet': HGWaveNet
    }

    if model_name.lower() in models:
        return models[model_name.lower()](snapshot_graphs, config, device).to(device)
    else:
        raise ValueError(f"未知模型: {model_name}")