import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl.function as fn
from dgl.nn.pytorch import GraphConv
import geoopt
from geoopt.manifolds.stereographic import PoincareBall
from geoopt.manifolds.sphere import Sphere


###################### APSDG模型 #######################
class APSDG(nn.Module):
    def __init__(self, snapshot_graphs, config, device):
        """
        自适应积空间离散动态图链接预测模型

        参数:
            snapshot_graphs: 离散动态图快照列表
            config: 模型配置参数
            device: 设备
        """
        super(APSDG, self).__init__()

        # 基本参数设置
        self.num_nodes = snapshot_graphs[0].num_nodes()
        self.hidden_dim = config['hidden_dim']
        self.embedding_dim = config['embedding_dim']
        self.history_window = config['history_window']
        self.device = device

        # 初始维度比例
        e_ratio = config.get('e_ratio', 1 / 3)
        b_ratio = config.get('b_ratio', 1 / 3)
        s_ratio = config.get('s_ratio', 1 / 3)

        # 计算各空间的嵌入维度
        self.e_dim = max(1, int(self.embedding_dim * e_ratio))
        self.b_dim = max(1, int(self.embedding_dim * b_ratio))
        self.s_dim = max(1, int(self.embedding_dim * s_ratio))

        # 调整维度以确保总和等于embedding_dim
        total = self.e_dim + self.b_dim + self.s_dim
        if total != self.embedding_dim:
            # 调整最大的维度
            if self.e_dim >= self.b_dim and self.e_dim >= self.s_dim:
                self.e_dim += (self.embedding_dim - total)
            elif self.b_dim >= self.e_dim and self.b_dim >= self.s_dim:
                self.b_dim += (self.embedding_dim - total)
            else:
                self.s_dim += (self.embedding_dim - total)

        # 曲率参数初始化
        self.b_curvature = nn.Parameter(torch.tensor([config.get('b_curvature', -1.0)]))
        self.s_curvature = nn.Parameter(torch.tensor([config.get('s_curvature', 1.0)]))

        # 初始化几何流形
        self.update_manifolds()

        # 记录维度比例为可训练参数
        self.log_e_ratio = nn.Parameter(torch.tensor(np.log(e_ratio)))
        self.log_b_ratio = nn.Parameter(torch.tensor(np.log(b_ratio)))
        self.log_s_ratio = nn.Parameter(torch.tensor(np.log(s_ratio)))

        # 初始化节点嵌入
        self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, self.embedding_dim))

        # 消息传递层
        self.product_space_message_passing = ProductSpaceMessagePassing(
            self.embedding_dim, self.e_dim, self.b_dim, self.s_dim, config.get('num_layers', 2)
        )

        # 历史信息融合模块
        self.history_fusion = HistoryFusionModule(
            self.embedding_dim, self.history_window, config.get('attention_dim', 64)
        )

        # HyperGRU模块
        self.hyper_gru = HyperGRU(
            self.embedding_dim, self.embedding_dim
        )

        # 强化学习参数
        self.rl_alpha = config.get('rl_alpha', 0.01)

        # 存储最近的历史快照嵌入
        self.history_embeddings = []

    def update_manifolds(self):
        """更新几何流形"""
        # 使用geoopt创建流形
        self.euclidean = geoopt.manifolds.Euclidean()
        self.poincare = PoincareBall(c=self.b_curvature.abs())
        self.sphere = Sphere(c=self.s_curvature)

    def update_dimension_ratios(self, reward):
        """
        基于强化学习更新维度比例

        参数:
            reward: 奖励信号
        """
        with torch.no_grad():
            # 计算维度比例的梯度，基于奖励进行更新
            e_grad = reward * self.log_e_ratio.grad if self.log_e_ratio.grad is not None else 0
            b_grad = reward * self.log_b_ratio.grad if self.log_b_ratio.grad is not None else 0
            s_grad = reward * self.log_s_ratio.grad if self.log_s_ratio.grad is not None else 0

            # 更新参数
            self.log_e_ratio.data += self.rl_alpha * e_grad
            self.log_b_ratio.data += self.rl_alpha * b_grad
            self.log_s_ratio.data += self.rl_alpha * s_grad

            # 重新计算比例
            e_ratio = torch.exp(self.log_e_ratio)
            b_ratio = torch.exp(self.log_b_ratio)
            s_ratio = torch.exp(self.log_s_ratio)

            # 归一化比例
            total = e_ratio + b_ratio + s_ratio
            e_ratio = e_ratio / total
            b_ratio = b_ratio / total
            s_ratio = s_ratio / total

            # 计算新的维度
            self.e_dim = max(1, int(self.embedding_dim * e_ratio.item()))
            self.b_dim = max(1, int(self.embedding_dim * b_ratio.item()))
            self.s_dim = max(1, int(self.embedding_dim * s_ratio.item()))

            # 调整维度以确保总和等于embedding_dim
            total = self.e_dim + self.b_dim + self.s_dim
            if total != self.embedding_dim:
                # 调整最大的维度
                if self.e_dim >= self.b_dim and self.e_dim >= self.s_dim:
                    self.e_dim += (self.embedding_dim - total)
                elif self.b_dim >= self.e_dim and self.b_dim >= self.s_dim:
                    self.b_dim += (self.embedding_dim - total)
                else:
                    self.s_dim += (self.embedding_dim - total)

            # 更新流形
            self.update_manifolds()

    def forward(self, graph, is_training=True):
        """
        前向传播

        参数:
            graph: 当前快照图
            is_training: 是否处于训练模式

        返回:
            node_embeddings: 更新后的节点嵌入
        """
        # 分割嵌入以匹配各空间维度
        e_emb = self.node_embeddings[:, :self.e_dim]
        b_emb = self.node_embeddings[:, self.e_dim:self.e_dim + self.b_dim]
        s_emb = self.node_embeddings[:, self.e_dim + self.b_dim:]

        # 进行积空间消息传递
        e_new, b_new, s_new = self.product_space_message_passing(
            graph, e_emb, b_emb, s_emb, self.poincare, self.sphere
        )

        # 拼接嵌入作为当前快照的表示
        current_emb = torch.cat([e_new, b_new, s_new], dim=1)

        # 如果有足够的历史嵌入，使用历史融合模块
        if len(self.history_embeddings) >= self.history_window:
            # 使用注意力机制融合历史信息
            history_context = self.history_fusion(current_emb, self.history_embeddings)

            # 使用HyperGRU更新嵌入
            updated_emb = self.hyper_gru(current_emb, history_context)
        else:
            # 如果历史不足，直接使用当前嵌入
            updated_emb = current_emb

        # 在训练模式下更新历史嵌入
        if is_training:
            # 更新历史嵌入队列
            self.history_embeddings.append(updated_emb.detach())
            if len(self.history_embeddings) > self.history_window:
                self.history_embeddings.pop(0)

        return updated_emb

    def predict_link(self, src_nodes, dst_nodes):
        """
        链接预测

        参数:
            src_nodes: 源节点列表
            dst_nodes: 目标节点列表

        返回:
            scores: 预测分数
        """
        # 获取节点嵌入
        src_emb = self.node_embeddings[src_nodes]
        dst_emb = self.node_embeddings[dst_nodes]

        # 分割嵌入以匹配各空间维度
        src_e = src_emb[:, :self.e_dim]
        src_b = src_emb[:, self.e_dim:self.e_dim + self.b_dim]
        src_s = src_emb[:, self.e_dim + self.b_dim:]

        dst_e = dst_emb[:, :self.e_dim]
        dst_b = dst_emb[:, self.e_dim:self.e_dim + self.b_dim]
        dst_s = dst_emb[:, self.e_dim + self.b_dim:]

        # 计算各空间的距离
        e_dist = self.euclidean.dist(src_e, dst_e)
        b_dist = self.poincare.dist(src_b, dst_b)
        s_dist = self.sphere.dist(src_s, dst_s)

        # 根据各自空间的维度比例进行加权
        e_weight = self.e_dim / self.embedding_dim
        b_weight = self.b_dim / self.embedding_dim
        s_weight = self.s_dim / self.embedding_dim

        # 计算总相似度（使用负距离转换为相似度）
        scores = -(e_weight * e_dist + b_weight * b_dist + s_weight * s_dist)

        return torch.sigmoid(scores)


class ProductSpaceMessagePassing(nn.Module):
    """积空间消息传递模块"""

    def __init__(self, embedding_dim, e_dim, b_dim, s_dim, num_layers):
        super(ProductSpaceMessagePassing, self).__init__()

        # 维度参数
        self.embedding_dim = embedding_dim
        self.e_dim = e_dim
        self.b_dim = b_dim
        self.s_dim = s_dim
        self.num_layers = num_layers

        # 使用DGL的GraphConv替代自定义欧几里得消息传递
        self.e_convs = nn.ModuleList([
            GraphConv(e_dim, e_dim, norm='both', bias=True, activation=None)
            for _ in range(num_layers)
        ])

        # 双曲和球面空间的变换层
        self.b_transforms = nn.ModuleList([
            nn.Linear(b_dim, b_dim) for _ in range(num_layers)
        ])
        self.s_transforms = nn.ModuleList([
            nn.Linear(s_dim, s_dim) for _ in range(num_layers)
        ])

        # 激活函数
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, graph, e_emb, b_emb, s_emb, poincare, sphere):
        """
        前向传播

        参数:
            graph: 图结构
            e_emb: 欧几里得空间嵌入
            b_emb: 双曲空间嵌入
            s_emb: 超球面空间嵌入
            poincare: 双曲空间流形
            sphere: 超球面空间流形

        返回:
            updated_e_emb: 更新后的欧几里得空间嵌入
            updated_b_emb: 更新后的双曲空间嵌入
            updated_s_emb: 更新后的超球面空间嵌入
        """
        for layer_idx in range(self.num_layers):
            # 欧几里得空间消息传递（使用DGL的GraphConv）
            e_emb = self.e_convs[layer_idx](graph, e_emb)
            e_emb = self.activation(e_emb)

            # 双曲空间消息传递
            # 将双曲嵌入映射到切空间
            origin = torch.zeros_like(b_emb[0]).unsqueeze(0).to(b_emb.device)
            tangent_emb = poincare.logmap(origin, b_emb)

            # 在切空间进行变换
            transformed_emb = self.b_transforms[layer_idx](tangent_emb)

            # 消息传递
            with graph.local_scope():
                graph.ndata['h'] = transformed_emb
                graph.update_all(fn.copy_u('h', 'm'), fn.ReduceFunction.mean('m', 'neigh'))
                neigh_emb = graph.ndata['neigh']

            # 映射回双曲空间
            b_emb = poincare.expmap(origin, neigh_emb)

            # 超球面空间消息传递
            # 确保嵌入在球面上（归一化）
            s_emb = F.normalize(s_emb, p=2, dim=1)

            # 在切空间进行变换
            tangent_emb = self.s_transforms[layer_idx](s_emb)

            # 消息传递
            with graph.local_scope():
                graph.ndata['h'] = tangent_emb
                graph.update_all(fn.copy_u('h', 'm'), fn.ReduceFunction.mean('m', 'neigh'))
                neigh_emb = graph.ndata['neigh']

            # 确保结果仍在球面上
            s_emb = F.normalize(neigh_emb, p=2, dim=1)

        return e_emb, b_emb, s_emb


class HistoryFusionModule(nn.Module):
    """历史信息融合模块"""

    def __init__(self, embedding_dim, history_window, attention_dim):
        super(HistoryFusionModule, self).__init__()

        self.embedding_dim = embedding_dim
        self.history_window = history_window
        self.attention_dim = attention_dim

        # 注意力机制使用PyTorch的MultiheadAttention
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=1,
            batch_first=True
        )

        # 线性变换层，用于计算注意力查询
        self.query_transform = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, current_emb, history_embs):
        """
        使用注意力机制融合历史信息

        参数:
            current_emb: 当前快照嵌入
            history_embs: 历史快照嵌入列表

        返回:
            history_context: 融合后的历史上下文嵌入
        """
        # 将历史嵌入堆叠为一个张量 [history_window, num_nodes, embedding_dim]
        history_tensor = torch.stack(history_embs, dim=0).transpose(0, 1)  # [num_nodes, history_window, embedding_dim]

        # 生成查询向量
        query = self.query_transform(current_emb).unsqueeze(1)  # [num_nodes, 1, embedding_dim]

        # 使用注意力机制
        context, _ = self.attention(query, history_tensor, history_tensor)

        # 移除维度 [num_nodes, 1, embedding_dim] -> [num_nodes, embedding_dim]
        context = context.squeeze(1)

        return context


class HyperGRU(nn.Module):
    """HyperGRU模块，基于双曲空间的GRU变体"""

    def __init__(self, input_dim, hidden_dim):
        super(HyperGRU, self).__init__()

        # 维度参数
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 使用PyTorch的GRUCell作为基础组件
        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)

    def forward(self, x, h):
        """
        HyperGRU前向传播

        参数:
            x: 输入嵌入
            h: 隐藏状态

        返回:
            new_h: 新的隐藏状态
        """
        # 直接使用PyTorch的GRUCell
        new_h = self.gru_cell(x, h)
        return new_h


################## HAT模型 - Hyperbolic Attention Networks ######################
class HAT(nn.Module):
    """HAT模型实现（用于基准比较）"""

    def __init__(self, snapshot_graphs, config, device):
        super(HAT, self).__init__()

        self.num_nodes = snapshot_graphs[0].num_nodes()
        self.embedding_dim = config['embedding_dim']
        self.device = device

        # 初始化节点嵌入
        self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, self.embedding_dim))

        # 双曲空间注意力层
        self.num_layers = config.get('num_layers', 2)
        self.attention_layers = nn.ModuleList([
            HyperbolicAttentionLayer(self.embedding_dim)
            for _ in range(self.num_layers)
        ])

        # 双曲空间曲率参数
        self.curvature = nn.Parameter(torch.tensor([config.get('curvature', -1.0)]))
        self.poincare = PoincareBall(c=self.curvature.abs())

    def forward(self, graph, *args, **kwargs):
        """前向传播"""
        x = self.node_embeddings

        # 更新流形
        self.poincare = PoincareBall(c=self.curvature.abs())

        # 通过多层注意力层
        for layer in self.attention_layers:
            x = layer(graph, x, self.poincare)

        return x

    def predict_link(self, src_nodes, dst_nodes):
        """链接预测"""
        # 获取节点嵌入
        src_emb = self.node_embeddings[src_nodes]
        dst_emb = self.node_embeddings[dst_nodes]

        # 计算双曲空间中的距离
        dist = self.poincare.dist(src_emb, dst_emb)

        # 转换为相似度分数
        scores = torch.exp(-dist)

        return scores


class HyperbolicAttentionLayer(nn.Module):
    """双曲空间注意力层"""

    def __init__(self, embedding_dim):
        super(HyperbolicAttentionLayer, self).__init__()

        self.embedding_dim = embedding_dim

        # 注意力参数
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=1,
            batch_first=True
        )

    def forward(self, graph, x, poincare):
        """前向传播"""
        with graph.local_scope():
            # 将嵌入从双曲空间映射到切空间
            origin = torch.zeros_like(x[0]).unsqueeze(0).to(x.device)
            tangent_x = poincare.logmap(origin, x)

            # 构建注意力图
            graph.ndata['h'] = tangent_x

            # 准备节点嵌入矩阵
            node_matrix = tangent_x.clone()

            # 为每个节点获取邻居索引
            adj_lists = []
            for i in range(graph.num_nodes()):
                neighbors = graph.predecessors(i).tolist()
                if not neighbors:
                    neighbors = [i]  # 如果没有邻居，使用自身
                adj_lists.append(neighbors)

            # 对每个节点计算注意力
            new_h = torch.zeros_like(tangent_x)
            for i in range(graph.num_nodes()):
                # 获取当前节点的邻居嵌入
                neighbors = adj_lists[i]
                neighbor_h = tangent_x[neighbors]

                # 生成查询、键和值
                query = tangent_x[i:i + 1].unsqueeze(0)  # [1, 1, dim]
                key = neighbor_h.unsqueeze(0)  # [1, num_neighbors, dim]
                value = neighbor_h.unsqueeze(0)  # [1, num_neighbors, dim]

                # 计算注意力
                attn_output, _ = self.attention(query, key, value)
                new_h[i] = attn_output.squeeze()

            # 将结果映射回双曲空间
            new_x = poincare.expmap(origin, new_h)

            return new_x


##################### HGCN模型 - Hyperbolic Graph Convolutional Networks ##########################
class HGCN(nn.Module):
    """HGCN模型实现（用于基准比较）"""

    def __init__(self, snapshot_graphs, config, device):
        super(HGCN, self).__init__()

        self.num_nodes = snapshot_graphs[0].num_nodes()
        self.embedding_dim = config['embedding_dim']
        self.device = device

        # 初始化节点嵌入
        self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, self.embedding_dim))

        # 双曲图卷积层
        self.num_layers = config.get('num_layers', 2)
        self.conv_layers = nn.ModuleList([
            HyperbolicGraphConv(self.embedding_dim, self.embedding_dim)
            for _ in range(self.num_layers)
        ])

        # 双曲空间曲率参数
        self.curvature = nn.Parameter(torch.tensor([config.get('curvature', -1.0)]))
        self.poincare = PoincareBall(c=self.curvature.abs())

    def forward(self, graph, *args, **kwargs):
        """前向传播"""
        x = self.node_embeddings

        # 更新流形
        self.poincare = PoincareBall(c=self.curvature.abs())

        # 通过多层图卷积
        for layer in self.conv_layers:
            x = layer(graph, x, self.poincare)

        return x

    def predict_link(self, src_nodes, dst_nodes):
        """链接预测"""
        # 获取节点嵌入
        src_emb = self.node_embeddings[src_nodes]
        dst_emb = self.node_embeddings[dst_nodes]

        # 计算双曲空间中的距离
        dist = self.poincare.dist(src_emb, dst_emb)

        # 转换为相似度分数
        scores = torch.exp(-dist)

        return scores


class HyperbolicGraphConv(nn.Module):
    """双曲图卷积层"""

    def __init__(self, in_dim, out_dim):
        super(HyperbolicGraphConv, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        # 线性变换
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, graph, x, poincare):
        """前向传播"""
        with graph.local_scope():
            # 将嵌入从双曲空间映射到切空间
            origin = torch.zeros_like(x[0]).unsqueeze(0).to(x.device)
            tangent_x = poincare.logmap(origin, x)

            # 在切空间进行线性变换
            transformed_x = self.linear(tangent_x)

            # 消息传递
            graph.ndata['h'] = transformed_x
            graph.update_all(fn.copy_u('h', 'm'), fn.ReduceFunction.mean('m', 'neigh'))
            neigh_x = graph.ndata['neigh']

            # 将结果映射回双曲空间
            new_x = poincare.expmap(origin, neigh_x)

            return new_x


############################ EvolveGCN模型 ###############################
class EvolveGCN(nn.Module):
    """EvolveGCN模型实现（用于基准比较）"""

    def __init__(self, snapshot_graphs, config, device):
        super(EvolveGCN, self).__init__()

        self.num_nodes = snapshot_graphs[0].num_nodes()
        self.embedding_dim = config['embedding_dim']
        self.device = device

        # 初始化节点嵌入
        self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, self.embedding_dim))

        # 使用DGL的GraphConv
        self.gc1 = GraphConv(self.embedding_dim, self.embedding_dim, norm='both', bias=True)
        self.gc2 = GraphConv(self.embedding_dim, self.embedding_dim, norm='both', bias=True)

        # 权重更新RNN
        self.weight_rnn = nn.GRUCell(self.embedding_dim ** 2, self.embedding_dim ** 2)

        # 存储历史权重
        self.gc1_weights_history = []
        self.gc2_weights_history = []

    def forward(self, graph, is_training=True):
        """前向传播"""
        # 更新权重
        if is_training and len(self.gc1_weights_history) > 0:
            # 使用RNN更新权重
            gc1_weight = self.gc1.weight.view(-1)
            gc1_history = self.gc1_weights_history[-1].view(-1)
            new_gc1_weight = self.weight_rnn(gc1_history, gc1_weight)
            self.gc1.weight.data = new_gc1_weight.view_as(self.gc1.weight.data)

            gc2_weight = self.gc2.weight.view(-1)
            gc2_history = self.gc2_weights_history[-1].view(-1)
            new_gc2_weight = self.weight_rnn(gc2_history, gc2_weight)
            self.gc2.weight.data = new_gc2_weight.view_as(self.gc2.weight.data)

        # 两层GCN
        x = self.node_embeddings
        x = F.relu(self.gc1(graph, x))
        x = self.gc2(graph, x)

        # 存储权重历史
        if is_training:
            self.gc1_weights_history.append(self.gc1.weight.detach().clone())
            self.gc2_weights_history.append(self.gc2.weight.detach().clone())

            # 限制历史长度
            if len(self.gc1_weights_history) > 10:
                self.gc1_weights_history.pop(0)
                self.gc2_weights_history.pop(0)

        return x

    def predict_link(self, src_nodes, dst_nodes):
        """链接预测"""
        # 获取节点嵌入
        src_emb = self.node_embeddings[src_nodes]
        dst_emb = self.node_embeddings[dst_nodes]

        # 计算内积
        scores = torch.sum(src_emb * dst_emb, dim=1)

        return torch.sigmoid(scores)


################################# GRUGCN模型 ###############################
class GRUGCN(nn.Module):
    """GRUGCN模型实现（用于基准比较）"""

    def __init__(self, snapshot_graphs, config, device):
        super(GRUGCN, self).__init__()

        self.num_nodes = snapshot_graphs[0].num_nodes()
        self.embedding_dim = config['embedding_dim']
        self.device = device

        # 初始化节点嵌入
        self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, self.embedding_dim))

        # GCN层
        self.gc = GraphConv(self.embedding_dim, self.embedding_dim, norm='both', bias=True, activation=F.relu)

        # GRU层
        self.gru = nn.GRUCell(self.embedding_dim, self.embedding_dim)

        # 历史嵌入
        self.hidden_state = None

    def forward(self, graph, is_training=True):
        """前向传播"""
        # 图卷积
        x = self.node_embeddings
        x = self.gc(graph, x)

        # GRU更新
        if self.hidden_state is None:
            self.hidden_state = torch.zeros_like(x)

        h = self.gru(x, self.hidden_state)

        # 更新隐藏状态
        if is_training:
            self.hidden_state = h.detach()

        return h

    def predict_link(self, src_nodes, dst_nodes):
        """链接预测"""
        # 获取节点嵌入
        src_emb = self.node_embeddings[src_nodes]
        dst_emb = self.node_embeddings[dst_nodes]

        # 计算内积
        scores = torch.sum(src_emb * dst_emb, dim=1)

        return torch.sigmoid(scores)


######################## HTGN模型 - Hyperbolic Temporal Graph Networks ######################
class HTGN(nn.Module):
    """HTGN模型实现（用于基准比较）"""

    def __init__(self, snapshot_graphs, config, device):
        super(HTGN, self).__init__()

        self.num_nodes = snapshot_graphs[0].num_nodes()
        self.embedding_dim = config['embedding_dim']
        self.device = device

        # 初始化节点嵌入
        self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, self.embedding_dim))

        # 双曲空间曲率参数
        self.curvature = nn.Parameter(torch.tensor([config.get('curvature', -1.0)]))
        self.poincare = PoincareBall(c=self.curvature.abs())

        # 双曲图神经网络
        self.hgnn = HyperbolicGraphNN(self.embedding_dim)

        # 双曲GRU
        self.hgru = HyperGRU(self.embedding_dim, self.embedding_dim)

        # 时间注意力
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=1,
            batch_first=True
        )

        # 历史嵌入列表
        self.history_embeddings = []

    def forward(self, graph, is_training=True):
        """前向传播"""
        # 更新流形
        self.poincare = PoincareBall(c=self.curvature.abs())

        # 获取当前快照的嵌入
        current_emb = self.hgnn(graph, self.node_embeddings, self.poincare)

        # 如果有历史嵌入，使用时间注意力和GRU更新
        if len(self.history_embeddings) > 0:
            # 时间注意力
            history_tensor = torch.stack(self.history_embeddings, dim=0).transpose(0, 1)
            query = current_emb.unsqueeze(1)
            context, _ = self.attention(query, history_tensor, history_tensor)
            context = context.squeeze(1)

            # 使用GRU更新
            updated_emb = self.hgru(current_emb, context)
        else:
            updated_emb = current_emb

        # 更新历史嵌入
        if is_training:
            self.history_embeddings.append(updated_emb.detach())
            if len(self.history_embeddings) > 10:  # 保存最近的10个快照
                self.history_embeddings.pop(0)

        return updated_emb

    def predict_link(self, src_nodes, dst_nodes):
        """链接预测"""
        # 获取节点嵌入
        src_emb = self.node_embeddings[src_nodes]
        dst_emb = self.node_embeddings[dst_nodes]

        # 计算双曲空间中的距离
        dist = self.poincare.dist(src_emb, dst_emb)

        # 转换为相似度分数
        scores = torch.exp(-dist)

        return scores


class HyperbolicGraphNN(nn.Module):
    """双曲图神经网络模块"""

    def __init__(self, embedding_dim):
        super(HyperbolicGraphNN, self).__init__()

        self.embedding_dim = embedding_dim

        # 线性变换
        self.linear = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, graph, x, poincare):
        """前向传播"""
        with graph.local_scope():
            # 将嵌入从双曲空间映射到切空间
            origin = torch.zeros_like(x[0]).unsqueeze(0).to(x.device)
            tangent_x = poincare.logmap(origin, x)

            # 在切空间进行线性变换
            transformed_x = self.linear(tangent_x)

            # 消息传递
            graph.ndata['h'] = transformed_x
            graph.update_all(fn.copy_u('h', 'm'), fn.ReduceFunction.mean('m', 'neigh'))
            neigh_x = graph.ndata['neigh']

            # 将结果映射回双曲空间
            new_x = poincare.expmap(origin, neigh_x)

            return new_x


# HGWaveNet模型
class HGWaveNet(nn.Module):
    """HGWaveNet模型实现（用于基准比较）"""

    def __init__(self, snapshot_graphs, config, device):
        super(HGWaveNet, self).__init__()

        self.num_nodes = snapshot_graphs[0].num_nodes()
        self.embedding_dim = config['embedding_dim']
        self.device = device

        # 初始化节点嵌入
        self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, self.embedding_dim))

        # 双曲空间曲率参数
        self.curvature = nn.Parameter(torch.tensor([config.get('curvature', -1.0)]))
        self.poincare = PoincareBall(c=self.curvature.abs())

        # 双曲扩散图卷积
        self.hdgc = HyperbolicDiffusionGraphConv(self.embedding_dim)

        # 双曲扩展因果卷积
        self.hdcc = HyperbolicDilatedCausalConv(self.embedding_dim)

        # 历史嵌入列表
        self.history_embeddings = []

    def forward(self, graph, is_training=True):
        """前向传播"""
        # 更新流形
        self.poincare = PoincareBall(c=self.curvature.abs())

        # 图卷积获取当前快照的嵌入
        current_emb = self.hdgc(graph, self.node_embeddings, self.poincare)

        # 如果有历史嵌入，使用因果卷积
        if len(self.history_embeddings) >= 3:  # 需要至少3个历史快照
            # 选择最近的3个历史快照
            recent_history = self.history_embeddings[-3:]

            # 因果卷积
            updated_emb = self.hdcc(current_emb, recent_history, self.poincare)
        else:
            updated_emb = current_emb

        # 更新历史嵌入
        if is_training:
            self.history_embeddings.append(updated_emb.detach())
            if len(self.history_embeddings) > 20:  # 保存最近的20个快照
                self.history_embeddings.pop(0)

        return updated_emb

    def predict_link(self, src_nodes, dst_nodes):
        """链接预测"""
        # 获取节点嵌入
        src_emb = self.node_embeddings[src_nodes]
        dst_emb = self.node_embeddings[dst_nodes]

        # 计算双曲空间中的距离
        dist = self.poincare.dist(src_emb, dst_emb)

        # 转换为相似度分数
        scores = torch.exp(-dist)

        return scores


class HyperbolicDiffusionGraphConv(nn.Module):
    """双曲扩散图卷积模块"""

    def __init__(self, embedding_dim):
        super(HyperbolicDiffusionGraphConv, self).__init__()

        self.embedding_dim = embedding_dim

        # 线性变换
        self.linear = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, graph, x, poincare):
        """前向传播"""
        with graph.local_scope():
            # 将嵌入从双曲空间映射到切空间
            origin = torch.zeros_like(x[0]).unsqueeze(0).to(x.device)
            tangent_x = poincare.logmap(origin, x)

            # 在切空间进行线性变换
            transformed_x = self.linear(tangent_x)

            # 消息传递
            graph.ndata['h'] = transformed_x
            graph.update_all(fn.copy_u('h', 'm'), fn.ReduceFunction.mean('m', 'neigh'))
            neigh_x = graph.ndata['neigh']

            # 将结果映射回双曲空间
            new_x = poincare.expmap(origin, neigh_x)

            return new_x


class HyperbolicDilatedCausalConv(nn.Module):
    """双曲扩展因果卷积模块"""

    def __init__(self, embedding_dim):
        super(HyperbolicDilatedCausalConv, self).__init__()

        self.embedding_dim = embedding_dim

        # 使用1D卷积替代自定义权重
        self.conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=embedding_dim,
            kernel_size=3,
            padding=0,
            dilation=1
        )

    def forward(self, current_emb, history_embs, poincare):
        """前向传播"""
        # 将嵌入从双曲空间映射到切空间
        origin = torch.zeros_like(current_emb[0]).unsqueeze(0).to(current_emb.device)
        tangent_current = poincare.logmap(origin, current_emb)
        tangent_history = [poincare.logmap(origin, h) for h in history_embs]

        # 拼接当前和历史嵌入
        concat = torch.stack([tangent_history[0], tangent_history[1], tangent_history[2]], dim=2)

        # 转换为卷积输入格式 [batch, channels, length]
        conv_input = concat.transpose(1, 2)

        # 应用卷积
        conv_output = self.conv(conv_input)

        # 转换回原始格式并加上当前嵌入
        result = conv_output.squeeze(2) + tangent_current

        # 将结果映射回双曲空间
        return poincare.expmap(origin, result)

########################### 加载模型函数 ################################
def load_model(model_name, snapshot_graphs, config, device):
    """
    加载指定的模型

    参数:
        model_name: 模型名称
        snapshot_graphs: 离散动态图快照列表
        config: 模型配置
        device: 设备

    返回:
        model: 初始化的模型实例
    """
    if model_name == 'apsdg':
        return APSDG(snapshot_graphs, config, device).to(device)
    elif model_name == 'hat':
        return HAT(snapshot_graphs, config, device).to(device)
    elif model_name == 'hgcn':
        return HGCN(snapshot_graphs, config, device).to(device)
    elif model_name == 'evolvegcn':
        return EvolveGCN(snapshot_graphs, config, device).to(device)
    elif model_name == 'grugcn':
        return GRUGCN(snapshot_graphs, config, device).to(device)
    elif model_name == 'htgn':
        return HTGN(snapshot_graphs, config, device).to(device)
    elif model_name == 'hgwavenet':
        return HGWaveNet(snapshot_graphs, config, device).to(device)
    else:
        raise ValueError(f"未知模型: {model_name}")
