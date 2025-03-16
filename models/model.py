import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl
import dgl.function as fn
from dgl.nn.pytorch import GATConv, GraphConv, GINConv, SAGEConv


# 积空间操作函数
def euclidean_distance(x, y):
    """欧几里得空间中的距离计算"""
    return torch.sum((x - y) ** 2, dim=-1)


def hyperbolic_distance(x, y, c):
    """双曲空间中的距离计算"""
    norm_x = torch.sum(x ** 2, dim=-1, keepdim=True)
    norm_y = torch.sum(y ** 2, dim=-1, keepdim=True)
    xy_dot = torch.sum(x * y, dim=-1, keepdim=True)

    # 双曲公式: d_H(x, y) = arccosh(1 + 2 * ||x-y||^2 / ((1-c*||x||^2)*(1-c*||y||^2)))
    num = torch.maximum(torch.tensor(1e-10).to(x.device), 2 * torch.sum((x - y) ** 2, dim=-1))
    denom = torch.maximum(torch.tensor(1e-10).to(x.device), (1 - c * norm_x) * (1 - c * norm_y))

    return torch.acosh(1 + num / denom)


def spherical_distance(x, y, c):
    """超球面空间中的距离计算"""
    # 球面公式: d_S(x, y) = arccos(xy_dot / (||x|| * ||y||))
    norm_x = torch.sqrt(torch.sum(x ** 2, dim=-1, keepdim=True))
    norm_y = torch.sqrt(torch.sum(y ** 2, dim=-1, keepdim=True))
    xy_dot = torch.sum(x * y, dim=-1, keepdim=True)

    # 归一化点积以避免数值问题
    cos_val = xy_dot / (norm_x * norm_y)
    cos_val = torch.clamp(cos_val, -1 + 1e-7, 1 - 1e-7)

    return torch.acos(cos_val) / torch.sqrt(c)


def mobius_addition(x, y, c):
    """莫比乌斯加法运算"""
    xy_dot = torch.sum(x * y, dim=-1, keepdim=True)
    x_norm_sq = torch.sum(x ** 2, dim=-1, keepdim=True)
    y_norm_sq = torch.sum(y ** 2, dim=-1, keepdim=True)

    # 莫比乌斯加法公式
    numerator = (1 + 2 * c * xy_dot + c * y_norm_sq) * x + (1 - c * x_norm_sq) * y
    denominator = 1 + 2 * c * xy_dot + c * c * x_norm_sq * y_norm_sq

    return numerator / denominator


def exp_map(x, v, c):
    """指数映射函数，从切空间映射到双曲空间"""
    v_norm = torch.norm(v, dim=-1, keepdim=True)
    second_term = torch.tanh(torch.sqrt(c) * v_norm / 2) * v / (torch.sqrt(c) * v_norm)
    return mobius_addition(x, second_term, c)


def log_map(x, y, c):
    """对数映射函数，从双曲空间映射到切空间"""
    addition = mobius_addition(-x, y, c)
    addition_norm = torch.norm(addition, dim=-1, keepdim=True)
    return 2 / torch.sqrt(c) * torch.atanh(torch.sqrt(c) * addition_norm) * addition / addition_norm


# APSDG模型 - 自适应积空间离散动态图链接预测模型
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
            graph, e_emb, b_emb, s_emb, self.b_curvature, self.s_curvature
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

        # 计算各空间的相似度
        e_sim = -euclidean_distance(src_e, dst_e)
        b_sim = -hyperbolic_distance(src_b, dst_b, self.b_curvature)
        s_sim = -spherical_distance(src_s, dst_s, self.s_curvature)

        # 根据各自空间的维度比例进行加权
        e_weight = self.e_dim / self.embedding_dim
        b_weight = self.b_dim / self.embedding_dim
        s_weight = self.s_dim / self.embedding_dim

        # 计算总相似度
        scores = e_weight * e_sim + b_weight * b_sim + s_weight * s_sim

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

        # 线性变换层
        self.e_transform = nn.ModuleList([
            nn.Linear(e_dim, e_dim) for _ in range(num_layers)
        ])
        self.b_transform = nn.ModuleList([
            nn.Linear(b_dim, b_dim) for _ in range(num_layers)
        ])
        self.s_transform = nn.ModuleList([
            nn.Linear(s_dim, s_dim) for _ in range(num_layers)
        ])

        # 激活函数
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, graph, e_emb, b_emb, s_emb, b_curvature, s_curvature):
        """
        前向传播

        参数:
            graph: 图结构
            e_emb: 欧几里得空间嵌入
            b_emb: 双曲空间嵌入
            s_emb: 超球面空间嵌入
            b_curvature: 双曲空间曲率
            s_curvature: 超球面空间曲率

        返回:
            updated_e_emb: 更新后的欧几里得空间嵌入
            updated_b_emb: 更新后的双曲空间嵌入
            updated_s_emb: 更新后的超球面空间嵌入
        """
        for layer_idx in range(self.num_layers):
            # 欧几里得空间消息传递
            updated_e = self._euclidean_message_passing(graph, e_emb, self.e_transform[layer_idx])

            # 双曲空间消息传递
            updated_b = self._hyperbolic_message_passing(graph, b_emb, self.b_transform[layer_idx], b_curvature)

            # 超球面空间消息传递
            updated_s = self._spherical_message_passing(graph, s_emb, self.s_transform[layer_idx], s_curvature)

            # 更新嵌入
            e_emb = self.activation(updated_e)
            b_emb = updated_b  # 双曲空间不使用激活函数，以保持几何性质
            s_emb = updated_s  # 超球面空间不使用激活函数，以保持几何性质

        return e_emb, b_emb, s_emb

    def _euclidean_message_passing(self, graph, emb, transform):
        """欧几里得空间的消息传递"""
        with graph.local_scope():
            # 线性变换
            transformed_emb = transform(emb)

            # 消息传递
            graph.ndata['h'] = transformed_emb
            graph.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'neigh'))

            # 聚合
            neigh_emb = graph.ndata['neigh']

            return neigh_emb

    def _hyperbolic_message_passing(self, graph, emb, transform, curvature):
        """双曲空间的消息传递"""
        with graph.local_scope():
            # 将嵌入从双曲空间映射到切空间
            origin = torch.zeros_like(emb[0]).unsqueeze(0)
            tangent_emb = log_map(origin, emb, curvature)

            # 在切空间进行线性变换
            transformed_emb = transform(tangent_emb)

            # 消息传递
            graph.ndata['h'] = transformed_emb
            graph.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'neigh'))

            # 聚合
            neigh_emb = graph.ndata['neigh']

            # 将结果映射回双曲空间
            return exp_map(origin, neigh_emb, curvature)

    def _spherical_message_passing(self, graph, emb, transform, curvature):
        """超球面空间的消息传递"""
        with graph.local_scope():
            # 对球面嵌入进行归一化，确保在球面上
            norm_emb = F.normalize(emb, p=2, dim=1)

            # 球面旋转变换（通过线性变换和归一化近似）
            transformed_emb = transform(norm_emb)
            transformed_emb = F.normalize(transformed_emb, p=2, dim=1)

            # 消息传递
            graph.ndata['h'] = transformed_emb
            graph.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'neigh'))

            # 聚合
            neigh_emb = graph.ndata['neigh']

            # 再次归一化，确保结果仍在球面上
            return F.normalize(neigh_emb, p=2, dim=1)


class HistoryFusionModule(nn.Module):
    """历史信息融合模块"""

    def __init__(self, embedding_dim, history_window, attention_dim):
        super(HistoryFusionModule, self).__init__()

        self.embedding_dim = embedding_dim
        self.history_window = history_window
        self.attention_dim = attention_dim

        # 注意力机制参数
        self.query_transform = nn.Linear(embedding_dim, attention_dim)
        self.key_transform = nn.Linear(embedding_dim, attention_dim)
        self.attention_vector = nn.Parameter(torch.randn(attention_dim, 1))

    def forward(self, current_emb, history_embs):
        """
        使用注意力机制融合历史信息

        参数:
            current_emb: 当前快照嵌入
            history_embs: 历史快照嵌入列表

        返回:
            history_context: 融合后的历史上下文嵌入
        """
        # 将历史嵌入堆叠为一个张量
        history_tensor = torch.stack(history_embs, dim=0)  # [history_window, num_nodes, embedding_dim]

        # 计算查询向量
        query = self.query_transform(current_emb).unsqueeze(0)  # [1, num_nodes, attention_dim]

        # 计算键向量
        keys = self.key_transform(history_tensor)  # [history_window, num_nodes, attention_dim]

        # 计算注意力分数
        scores = torch.tanh(query + keys)  # [history_window, num_nodes, attention_dim]

        # 计算注意力权重
        weights = torch.matmul(scores, self.attention_vector)  # [history_window, num_nodes, 1]
        weights = F.softmax(weights, dim=0)  # 在历史维度上进行softmax

        # 加权求和
        context = torch.sum(weights * history_tensor, dim=0)  # [num_nodes, embedding_dim]

        return context


class HyperGRU(nn.Module):
    """HyperGRU模块，基于双曲空间的GRU变体"""

    def __init__(self, input_dim, hidden_dim):
        super(HyperGRU, self).__init__()

        # 维度参数
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # GRU门控参数
        self.w_ir = nn.Linear(input_dim, hidden_dim, bias=False)
        self.w_hr = nn.Linear(hidden_dim, hidden_dim)
        self.w_iz = nn.Linear(input_dim, hidden_dim, bias=False)
        self.w_hz = nn.Linear(hidden_dim, hidden_dim)
        self.w_in = nn.Linear(input_dim, hidden_dim, bias=False)
        self.w_hn = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, h):
        """
        HyperGRU前向传播

        参数:
            x: 输入嵌入
            h: 隐藏状态

        返回:
            new_h: 新的隐藏状态
        """
        # 重置门
        r = torch.sigmoid(self.w_ir(x) + self.w_hr(h))

        # 更新门
        z = torch.sigmoid(self.w_iz(x) + self.w_hz(h))

        # 候选隐藏状态
        n = torch.tanh(self.w_in(x) + r * self.w_hn(h))

        # 更新隐藏状态
        new_h = (1 - z) * h + z * n

        return new_h


# HAT模型 - Hyperbolic Attention Networks
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
        self.attention_layers = nn.ModuleList([
            HyperbolicAttentionLayer(self.embedding_dim) for _ in range(config.get('num_layers', 2))
        ])

        # 双曲空间曲率参数
        self.curvature = nn.Parameter(torch.tensor([config.get('curvature', -1.0)]))

    def forward(self, graph, *args, **kwargs):
        """前向传播"""
        x = self.node_embeddings

        # 通过多层注意力层
        for layer in self.attention_layers:
            x = layer(graph, x, self.curvature)

        return x

    def predict_link(self, src_nodes, dst_nodes):
        """链接预测"""
        # 获取节点嵌入
        src_emb = self.node_embeddings[src_nodes]
        dst_emb = self.node_embeddings[dst_nodes]

        # 计算双曲空间中的距离
        dist = hyperbolic_distance(src_emb, dst_emb, self.curvature)

        # 转换为相似度分数
        scores = torch.exp(-dist)

        return scores


class HyperbolicAttentionLayer(nn.Module):
    """双曲空间注意力层"""

    def __init__(self, embedding_dim):
        super(HyperbolicAttentionLayer, self).__init__()

        self.embedding_dim = embedding_dim

        # 注意力参数
        self.query_transform = nn.Linear(embedding_dim, embedding_dim)
        self.key_transform = nn.Linear(embedding_dim, embedding_dim)
        self.value_transform = nn.Linear(embedding_dim, embedding_dim)

        self.attention_scale = embedding_dim ** 0.5

    def forward(self, graph, x, curvature):
        """前向传播"""
        with graph.local_scope():
            # 将嵌入从双曲空间映射到切空间
            origin = torch.zeros_like(x[0]).unsqueeze(0)
            tangent_x = log_map(origin, x, curvature)

            # 计算查询、键和值
            query = self.query_transform(tangent_x)
            key = self.key_transform(tangent_x)
            value = self.value_transform(tangent_x)

            # 在图上进行注意力计算
            graph.ndata['q'] = query
            graph.ndata['k'] = key
            graph.ndata['v'] = value

            # 消息函数：计算注意力分数
            def message_func(edges):
                # 计算点积注意力
                attention = torch.sum(edges.src['k'] * edges.dst['q'], dim=1) / self.attention_scale
                return {'a': attention, 'v': edges.src['v']}

            # 归约函数：聚合邻居信息
            def reduce_func(nodes):
                # 对注意力权重应用softmax
                alpha = F.softmax(nodes.mailbox['a'], dim=1)
                # 加权聚合
                h = torch.sum(alpha.unsqueeze(-1) * nodes.mailbox['v'], dim=1)
                return {'h': h}

            # 执行消息传递
            graph.update_all(message_func, reduce_func)

            # 获取新的嵌入
            new_tangent_x = graph.ndata['h']

            # 将结果映射回双曲空间
            new_x = exp_map(origin, new_tangent_x, curvature)

            return new_x


# HGCN模型 - Hyperbolic Graph Convolutional Networks
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
        self.conv_layers = nn.ModuleList([
            HyperbolicGraphConv(self.embedding_dim, self.embedding_dim)
            for _ in range(config.get('num_layers', 2))
        ])

        # 双曲空间曲率参数
        self.curvature = nn.Parameter(torch.tensor([config.get('curvature', -1.0)]))

    def forward(self, graph, *args, **kwargs):
        """前向传播"""
        x = self.node_embeddings

        # 通过多层图卷积
        for layer in self.conv_layers:
            x = layer(graph, x, self.curvature)

        return x

    def predict_link(self, src_nodes, dst_nodes):
        """链接预测"""
        # 获取节点嵌入
        src_emb = self.node_embeddings[src_nodes]
        dst_emb = self.node_embeddings[dst_nodes]

        # 计算双曲空间中的距离
        dist = hyperbolic_distance(src_emb, dst_emb, self.curvature)

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

    def forward(self, graph, x, curvature):
        """前向传播"""
        with graph.local_scope():
            # 将嵌入从双曲空间映射到切空间
            origin = torch.zeros_like(x[0]).unsqueeze(0)
            tangent_x = log_map(origin, x, curvature)

            # 在切空间进行线性变换
            transformed_x = self.linear(tangent_x)

            # 消息传递
            graph.ndata['h'] = transformed_x
            graph.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'neigh'))

            # 聚合
            neigh_x = graph.ndata['neigh']

            # 将结果映射回双曲空间
            new_x = exp_map(origin, neigh_x, curvature)

            return new_x


# EvolveGCN模型
class EvolveGCN(nn.Module):
    """EvolveGCN模型实现（用于基准比较）"""

    def __init__(self, snapshot_graphs, config, device):
        super(EvolveGCN, self).__init__()

        self.num_nodes = snapshot_graphs[0].num_nodes()
        self.embedding_dim = config['embedding_dim']
        self.device = device

        # 初始化节点嵌入
        self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, self.embedding_dim))

        # GCN层
        self.gc1_weights = nn.Parameter(torch.randn(self.embedding_dim, self.embedding_dim))
        self.gc2_weights = nn.Parameter(torch.randn(self.embedding_dim, self.embedding_dim))

        # 权重更新RNN
        self.weight_rnn = nn.GRUCell(self.embedding_dim ** 2, self.embedding_dim ** 2)

        # 存储历史权重
        self.gc1_weights_history = []
        self.gc2_weights_history = []

    def _gcn_layer(self, graph, features, weights):
        """图卷积层"""
        with graph.local_scope():
            # 归一化邻接矩阵
            degs = graph.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            norm = norm.to(features.device)

            graph.ndata['norm'] = norm.unsqueeze(1)

            # 线性变换
            features = torch.matmul(features, weights)

            # 消息传递
            graph.ndata['h'] = features
            graph.update_all(
                message_func=fn.u_mul_e('h', 'norm', 'm'),
                reduce_func=fn.sum(msg='m', out='h')
            )

            # 获取新特征
            return graph.ndata['h']

    def forward(self, graph, is_training=True):
        """前向传播"""
        # 更新权重
        if is_training and len(self.gc1_weights_history) > 0:
            # 使用RNN更新权重
            flattened_weights1 = self.gc1_weights.view(-1)
            flattened_history1 = self.gc1_weights_history[-1].view(-1)
            new_flattened_weights1 = self.weight_rnn(flattened_history1, flattened_weights1)
            self.gc1_weights.data = new_flattened_weights1.view(self.embedding_dim, self.embedding_dim)

            flattened_weights2 = self.gc2_weights.view(-1)
            flattened_history2 = self.gc2_weights_history[-1].view(-1)
            new_flattened_weights2 = self.weight_rnn(flattened_history2, flattened_weights2)
            self.gc2_weights.data = new_flattened_weights2.view(self.embedding_dim, self.embedding_dim)

        # 两层GCN
        x = self.node_embeddings
        x = F.relu(self._gcn_layer(graph, x, self.gc1_weights))
        x = self._gcn_layer(graph, x, self.gc2_weights)

        # 存储权重历史
        if is_training:
            self.gc1_weights_history.append(self.gc1_weights.detach())
            self.gc2_weights_history.append(self.gc2_weights.detach())

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


# GRUGCN模型
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


# HTGN模型 - Hyperbolic Temporal Graph Networks
class HTGN(nn.Module):
    """HTGN模型实现（用于基准比较）"""

    def __init__(self, snapshot_graphs, config, device):
        super(HTGN, self).__init__()

        self.num_nodes = snapshot_graphs[0].num_nodes()
        self.embedding_dim = config['embedding_dim']
        self.device = device

        # 初始化节点嵌入
        self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, self.embedding_dim))

        # 双曲图神经网络
        self.hgnn = HyperbolicGraphNN(self.embedding_dim)

        # 双曲GRU
        self.hgru = HyperGRU(self.embedding_dim, self.embedding_dim)

        # 双曲时间注意力
        self.hta = HyperbolicTemporalAttention(self.embedding_dim)

        # 双曲空间曲率参数
        self.curvature = nn.Parameter(torch.tensor([config.get('curvature', -1.0)]))

        # 历史嵌入列表
        self.history_embeddings = []

    def forward(self, graph, is_training=True):
        """前向传播"""
        # 获取当前快照的嵌入
        current_emb = self.hgnn(graph, self.node_embeddings, self.curvature)

        # 如果有历史嵌入，使用时间注意力和GRU更新
        if len(self.history_embeddings) > 0:
            # 时间注意力
            history_context = self.hta(current_emb, self.history_embeddings, self.curvature)

            # 使用双曲GRU更新
            updated_emb = self.hgru(current_emb, history_context)
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
        dist = hyperbolic_distance(src_emb, dst_emb, self.curvature)

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

    def forward(self, graph, x, curvature):
        """前向传播"""
        with graph.local_scope():
            # 将嵌入从双曲空间映射到切空间
            origin = torch.zeros_like(x[0]).unsqueeze(0)
            tangent_x = log_map(origin, x, curvature)

            # 在切空间进行线性变换
            transformed_x = self.linear(tangent_x)

            # 消息传递
            graph.ndata['h'] = transformed_x
            graph.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'neigh'))

            # 聚合
            neigh_x = graph.ndata['neigh']

            # 将结果映射回双曲空间
            new_x = exp_map(origin, neigh_x, curvature)

            return new_x


class HyperbolicTemporalAttention(nn.Module):
    """双曲时间注意力模块"""

    def __init__(self, embedding_dim):
        super(HyperbolicTemporalAttention, self).__init__()

        self.embedding_dim = embedding_dim

        # 注意力参数
        self.query_transform = nn.Linear(embedding_dim, embedding_dim)
        self.key_transform = nn.Linear(embedding_dim, embedding_dim)
        self.attention_vector = nn.Parameter(torch.randn(embedding_dim, 1))

    def forward(self, current_emb, history_embs, curvature):
        """前向传播"""
        # 如果没有历史，返回当前嵌入
        if len(history_embs) == 0:
            return current_emb

        # 将历史嵌入堆叠
        history_tensor = torch.stack(history_embs, dim=0)

        # 将嵌入从双曲空间映射到切空间
        origin = torch.zeros_like(current_emb[0]).unsqueeze(0)
        tangent_current = log_map(origin, current_emb, curvature)

        # 堆叠的批处理log_map
        tangent_history = torch.zeros_like(history_tensor)
        for i in range(len(history_embs)):
            tangent_history[i] = log_map(origin, history_embs[i], curvature)

        # 计算查询和键
        query = self.query_transform(tangent_current).unsqueeze(0)  # [1, num_nodes, dim]
        keys = self.key_transform(tangent_history)  # [history_len, num_nodes, dim]

        # 计算注意力分数
        scores = torch.tanh(query + keys)  # [history_len, num_nodes, dim]

        # 计算注意力权重
        weights = torch.matmul(scores, self.attention_vector)  # [history_len, num_nodes, 1]
        weights = F.softmax(weights, dim=0)  # 在历史维度上进行softmax

        # 加权求和
        weighted_sum = torch.sum(weights * tangent_history, dim=0)  # [num_nodes, dim]

        # 将结果映射回双曲空间
        context = exp_map(origin, weighted_sum, curvature)

        return context


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

        # 双曲扩散图卷积
        self.hdgc = HyperbolicDiffusionGraphConv(self.embedding_dim)

        # 双曲扩展因果卷积
        self.hdcc = HyperbolicDilatedCausalConv(self.embedding_dim)

        # 双曲空间曲率参数
        self.curvature = nn.Parameter(torch.tensor([config.get('curvature', -1.0)]))

        # 历史嵌入列表
        self.history_embeddings = []

    def forward(self, graph, is_training=True):
        """前向传播"""
        # 图卷积获取当前快照的嵌入
        current_emb = self.hdgc(graph, self.node_embeddings, self.curvature)

        # 如果有历史嵌入，使用因果卷积
        if len(self.history_embeddings) >= 3:  # 需要至少3个历史快照
            # 选择最近的3个历史快照
            recent_history = self.history_embeddings[-3:]

            # 因果卷积
            updated_emb = self.hdcc(current_emb, recent_history, self.curvature)
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
        dist = hyperbolic_distance(src_emb, dst_emb, self.curvature)

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

    def forward(self, graph, x, curvature):
        """前向传播"""
        with graph.local_scope():
            # 将嵌入从双曲空间映射到切空间
            origin = torch.zeros_like(x[0]).unsqueeze(0)
            tangent_x = log_map(origin, x, curvature)

            # 在切空间进行线性变换
            transformed_x = self.linear(tangent_x)

            # 消息传递
            graph.ndata['h'] = transformed_x
            graph.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'neigh'))

            # 聚合
            neigh_x = graph.ndata['neigh']

            # 将结果映射回双曲空间
            new_x = exp_map(origin, neigh_x, curvature)

            return new_x


class HyperbolicDilatedCausalConv(nn.Module):
    """双曲扩展因果卷积模块"""

    def __init__(self, embedding_dim):
        super(HyperbolicDilatedCausalConv, self).__init__()

        self.embedding_dim = embedding_dim

        # 卷积参数
        self.weight1 = nn.Parameter(torch.randn(embedding_dim, embedding_dim))
        self.weight2 = nn.Parameter(torch.randn(embedding_dim, embedding_dim))
        self.weight3 = nn.Parameter(torch.randn(embedding_dim, embedding_dim))

    def forward(self, current_emb, history_embs, curvature):
        """前向传播"""
        # 将嵌入从双曲空间映射到切空间
        origin = torch.zeros_like(current_emb[0]).unsqueeze(0)
        tangent_current = log_map(origin, current_emb, curvature)

        tangent_history = [log_map(origin, h, curvature) for h in history_embs]

        # 一维卷积操作（简化为加权求和）
        transformed_current = torch.matmul(tangent_current, self.weight1)
        transformed_history1 = torch.matmul(tangent_history[0], self.weight1)
        transformed_history2 = torch.matmul(tangent_history[1], self.weight2)
        transformed_history3 = torch.matmul(tangent_history[2], self.weight3)

        # 加权求和
        result = transformed_current + transformed_history1 + transformed_history2 + transformed_history3

        # 将结果映射回双曲空间
        return exp_map(origin, result, curvature)


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

