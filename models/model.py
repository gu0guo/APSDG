import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import numpy as np


# 几何空间操作定义
class GeometricOperations:
    """几何空间基本操作集合类"""

    @staticmethod
    def euclidean_distance(x, y):
        """欧几里得空间距离计算"""
        return torch.sum((x - y) ** 2, dim=-1)

    @staticmethod
    def hyperbolic_distance(x, y, c):
        """双曲空间距离计算"""
        x_norm_sq = torch.sum(x ** 2, dim=-1, keepdim=True)
        y_norm_sq = torch.sum(y ** 2, dim=-1, keepdim=True)
        xy_dot_prod = torch.sum(x * y, dim=-1, keepdim=True)

        num = 2 * torch.sum((x - y) ** 2, dim=-1)
        denom = (1 - c * x_norm_sq) * (1 - c * y_norm_sq)

        # 避免数值问题
        num = torch.clamp(num, min=1e-10)
        denom = torch.clamp(denom, min=1e-10)

        return torch.acosh(1 + num / denom)

    @staticmethod
    def spherical_distance(x, y, c):
        """超球面空间距离计算"""
        # 归一化输入向量
        x_norm = torch.sqrt(torch.sum(x ** 2, dim=-1, keepdim=True))
        y_norm = torch.sqrt(torch.sum(y ** 2, dim=-1, keepdim=True))

        # 确保不为零
        x_norm = torch.clamp(x_norm, min=1e-10)
        y_norm = torch.clamp(y_norm, min=1e-10)

        # 点积并归一化
        dot_prod = torch.sum(x * y, dim=-1)
        cos_val = dot_prod / (x_norm.squeeze(-1) * y_norm.squeeze(-1))

        # 数值稳定性
        cos_val = torch.clamp(cos_val, min=-1 + 1e-7, max=1 - 1e-7)

        return torch.acos(cos_val) / torch.sqrt(c)

    @staticmethod
    def mobius_addition(x, y, c):
        """莫比乌斯加法"""
        xy_dot = torch.sum(x * y, dim=-1, keepdim=True)
        x_norm_sq = torch.sum(x ** 2, dim=-1, keepdim=True)
        y_norm_sq = torch.sum(y ** 2, dim=-1, keepdim=True)

        numerator = (1 + 2 * c * xy_dot + c * y_norm_sq) * x + (1 - c * x_norm_sq) * y
        denominator = 1 + 2 * c * xy_dot + (c ** 2) * x_norm_sq * y_norm_sq

        return numerator / denominator

    @staticmethod
    def exp_map(x, v, c):
        """指数映射函数，从切空间映射到双曲空间"""
        v_norm = torch.norm(v, dim=-1, keepdim=True)
        # 避免除零
        v_norm = torch.clamp(v_norm, min=1e-10)

        second_term = torch.tanh(torch.sqrt(c) * v_norm / 2) * v / (torch.sqrt(c) * v_norm)
        return GeometricOperations.mobius_addition(x, second_term, c)

    @staticmethod
    def log_map(x, y, c):
        """对数映射函数，从双曲空间映射到切空间"""
        # 移动到原点
        y_minus_x = GeometricOperations.mobius_addition(-x, y, c)
        y_minus_x_norm = torch.norm(y_minus_x, dim=-1, keepdim=True)
        # 避免除零
        y_minus_x_norm = torch.clamp(y_minus_x_norm, min=1e-10)

        return 2 / torch.sqrt(c) * torch.atanh(torch.sqrt(c) * y_minus_x_norm) * y_minus_x / y_minus_x_norm


# 积空间消息传递模块
class ProductSpaceMessagePassing(nn.Module):
    """实现积空间消息传递机制"""

    def __init__(self, embedding_dim, e_dim, b_dim, s_dim, num_layers):
        super(ProductSpaceMessagePassing, self).__init__()

        # 维度参数
        self.embedding_dim = embedding_dim
        self.e_dim = e_dim
        self.b_dim = b_dim
        self.s_dim = s_dim
        self.num_layers = num_layers

        # 线性变换层
        self.e_transforms = nn.ModuleList([nn.Linear(e_dim, e_dim) for _ in range(num_layers)])
        self.b_transforms = nn.ModuleList([nn.Linear(b_dim, b_dim) for _ in range(num_layers)])
        self.s_transforms = nn.ModuleList([nn.Linear(s_dim, s_dim) for _ in range(num_layers)])

        # 激活函数
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, graph, e_emb, b_emb, s_emb, b_curvature, s_curvature):
        """
        前向传播

        Args:
            graph: DGL图
            e_emb: 欧几里得空间嵌入
            b_emb: 双曲空间嵌入
            s_emb: 超球面空间嵌入
            b_curvature: 双曲空间曲率
            s_curvature: 超球面空间曲率

        Returns:
            更新后的三种空间嵌入
        """
        for layer_idx in range(self.num_layers):
            # 欧几里得空间消息传递
            e_emb = self._euclidean_message_passing(
                graph, e_emb, self.e_transforms[layer_idx]
            )
            e_emb = self.activation(e_emb)  # 欧式空间可以直接应用激活函数

            # 双曲空间消息传递
            b_emb = self._hyperbolic_message_passing(
                graph, b_emb, self.b_transforms[layer_idx], b_curvature
            )
            # 双曲空间不直接应用激活函数，以保持几何性质

            # 超球面空间消息传递
            s_emb = self._spherical_message_passing(
                graph, s_emb, self.s_transforms[layer_idx], s_curvature
            )
            # 超球面空间不直接应用激活函数，以保持几何性质

        return e_emb, b_emb, s_emb

    def _euclidean_message_passing(self, graph, emb, transform):
        """欧几里得空间消息传递"""
        with graph.local_scope():
            # 线性变换
            h = transform(emb)

            # 消息传递
            graph.ndata['h'] = h
            graph.update_all(fn.copy_u('h', 'm'), fn.ReduceFunction.mean('m', 'neigh'))

            # 获取聚合结果
            return graph.ndata['neigh']

    def _hyperbolic_message_passing(self, graph, emb, transform, curvature):
        """双曲空间消息传递"""
        with graph.local_scope():
            # 将嵌入从双曲空间映射到切空间
            origin = torch.zeros_like(emb[0]).unsqueeze(0).to(emb.device)
            tangent_emb = GeometricOperations.log_map(origin, emb, curvature)

            # 在切空间进行线性变换
            h = transform(tangent_emb)

            # 消息传递
            graph.ndata['h'] = h
            graph.update_all(fn.copy_u('h', 'm'), fn.ReduceFunction.mean('m', 'neigh'))

            # 获取聚合结果
            neigh = graph.ndata['neigh']

            # 将结果映射回双曲空间
            return GeometricOperations.exp_map(origin, neigh, curvature)

    def _spherical_message_passing(self, graph, emb, transform, curvature):
        """超球面空间消息传递"""
        with graph.local_scope():
            # 对球面嵌入进行归一化
            norm_emb = F.normalize(emb, p=2, dim=1)

            # 变换（通过线性变换和归一化近似）
            h = transform(norm_emb)
            h = F.normalize(h, p=2, dim=1)

            # 消息传递
            graph.ndata['h'] = h
            graph.update_all(fn.copy_u('h', 'm'), fn.ReduceFunction.mean('m', 'neigh'))

            # 获取聚合结果
            neigh = graph.ndata['neigh']

            # 再次归一化
            return F.normalize(neigh, p=2, dim=1)


# 历史信息融合模块
class HistoryFusionModule(nn.Module):
    """实现历史信息融合"""

    def __init__(self, embedding_dim, history_window, attention_dim):
        super(HistoryFusionModule, self).__init__()

        self.embedding_dim = embedding_dim
        self.history_window = history_window
        self.attention_dim = attention_dim

        # 注意力机制参数，论文公式 (11)-(14)
        self.query_transform = nn.Linear(embedding_dim, attention_dim)
        self.key_transform = nn.Linear(embedding_dim, attention_dim)
        self.attention_vector = nn.Parameter(torch.randn(attention_dim, 1))

    def forward(self, current_emb, history_embs):
        """
        使用注意力机制融合历史信息

        Args:
            current_emb: 当前快照嵌入
            history_embs: 历史快照嵌入列表

        Returns:
            融合后的历史上下文嵌入
        """
        # 将历史嵌入堆叠
        history_tensor = torch.stack(history_embs, dim=0)  # [history_window, num_nodes, embedding_dim]

        # 构建上下文矩阵M，论文公式 (11)
        M = history_tensor

        # 计算注意力分数，论文公式 (12)
        Q = self.key_transform.weight
        transformed_M = torch.matmul(M, Q.t())  # [history_window, num_nodes, attention_dim]
        r = self.attention_vector
        e = torch.tanh(transformed_M) @ r  # [history_window, num_nodes, 1]

        # 归一化获取注意力权重，论文公式 (13)
        a = F.softmax(e, dim=0)  # 在历史维度上做softmax

        # 加权求和得到综合历史嵌入，论文公式 (14)
        context = torch.sum(a * history_tensor, dim=0)  # [num_nodes, embedding_dim]

        return context


# HyperGRU模块
class HyperGRU(nn.Module):
    """双曲空间GRU实现，用于处理时间信息"""

    def __init__(self, input_dim, hidden_dim):
        super(HyperGRU, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # GRU门控参数，论文公式 (15)-(19)
        self.W_r = nn.Linear(input_dim, hidden_dim, bias=False)
        self.U_r = nn.Linear(hidden_dim, hidden_dim)
        self.b_r = nn.Parameter(torch.zeros(hidden_dim))

        self.W_u = nn.Linear(input_dim, hidden_dim, bias=False)
        self.U_u = nn.Linear(hidden_dim, hidden_dim)
        self.b_u = nn.Parameter(torch.zeros(hidden_dim))

        self.W_m = nn.Linear(input_dim, hidden_dim, bias=False)
        self.U_m = nn.Linear(hidden_dim, hidden_dim)
        self.b_m = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, H_t, Z_t_minus_1):
        """
        HyperGRU前向传播

        Args:
            H_t: 当前时刻通过图神经网络得到的隐藏状态
            Z_t_minus_1: 上一时刻的综合嵌入

        Returns:
            Z_t: 当前时刻的综合嵌入
        """
        # 重置门，论文公式 (15)
        r_t = torch.sigmoid(self.W_r(H_t) + self.U_r(Z_t_minus_1) + self.b_r)

        # 更新门，论文公式 (16)
        u_t = torch.sigmoid(self.W_u(H_t) + self.U_u(Z_t_minus_1) + self.b_u)

        # 候选隐藏状态，论文公式 (17)
        m_t = torch.tanh(self.W_m(H_t) + r_t * (self.U_m(Z_t_minus_1) + self.b_m))

        # 更新隐藏状态，论文公式 (18)(19)
        Z_t = (1 - u_t) * Z_t_minus_1 + u_t * m_t

        return Z_t


# APSDG模型实现
class APSDG(nn.Module):
    """自适应积空间离散动态图链接预测模型"""

    def __init__(self, snapshot_graphs, config):
        super(APSDG, self).__init__()

        # 基本参数设置
        self.num_nodes = snapshot_graphs[0].num_nodes()
        self.embedding_dim = config['embedding_dim']
        self.hidden_dim = config.get('hidden_dim', self.embedding_dim)
        self.history_window = config.get('history_window', 5)

        # 初始维度比例
        e_ratio = config.get('e_ratio', 1 / 3)
        b_ratio = config.get('b_ratio', 1 / 3)
        s_ratio = config.get('s_ratio', 1 / 3)

        # 计算各空间的嵌入维度
        self.e_dim = max(1, int(self.embedding_dim * e_ratio))
        self.b_dim = max(1, int(self.embedding_dim * b_ratio))
        self.s_dim = self.embedding_dim - self.e_dim - self.b_dim  # 确保维度之和等于总维度

        # 维度比例的可学习参数
        self.log_e_ratio = nn.Parameter(torch.tensor(np.log(e_ratio)))
        self.log_b_ratio = nn.Parameter(torch.tensor(np.log(b_ratio)))
        self.log_s_ratio = nn.Parameter(torch.tensor(np.log(s_ratio)))

        # 曲率参数初始化
        self.b_curvature = nn.Parameter(torch.tensor([config.get('b_curvature', -1.0)]))
        self.s_curvature = nn.Parameter(torch.tensor([config.get('s_curvature', 1.0)]))

        # 节点嵌入初始化
        self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, self.embedding_dim))

        # 积空间消息传递模块
        self.product_space_message_passing = ProductSpaceMessagePassing(
            self.embedding_dim, self.e_dim, self.b_dim, self.s_dim, config.get('num_layers', 2)
        )

        # 历史信息融合模块
        self.history_fusion = HistoryFusionModule(
            self.embedding_dim, self.history_window, config.get('attention_dim', 64)
        )

        # HyperGRU模块
        self.hyper_gru = HyperGRU(self.embedding_dim, self.embedding_dim)

        # 强化学习参数
        self.rl_alpha = config.get('rl_alpha', 0.01)
        self.beta1 = config.get('beta1', 0.5)
        self.beta2 = config.get('beta2', 0.5)

        # 历史嵌入列表
        self.history_embeddings = []

    def update_dimension_ratios(self, reward):
        """
        基于强化学习更新维度比例

        Args:
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

            # 更新维度
            self.e_dim = max(1, int(self.embedding_dim * e_ratio.item()))
            self.b_dim = max(1, int(self.embedding_dim * b_ratio.item()))
            self.s_dim = self.embedding_dim - self.e_dim - self.b_dim  # 确保维度之和等于总维度

    def forward(self, graph, is_training=True):
        """
        前向传播

        Args:
            graph: 当前快照图
            is_training: 是否处于训练模式

        Returns:
            updated_emb: 更新后的节点嵌入
        """
        # 分割初始嵌入为三个空间的嵌入
        e_emb = self.node_embeddings[:, :self.e_dim]
        b_emb = self.node_embeddings[:, self.e_dim:self.e_dim + self.b_dim]
        s_emb = self.node_embeddings[:, self.e_dim + self.b_dim:]

        # 积空间消息传递
        e_new, b_new, s_new = self.product_space_message_passing(
            graph, e_emb, b_emb, s_emb, self.b_curvature, self.s_curvature
        )

        # 拼接为当前快照的隐藏表示
        H_t = torch.cat([e_new, b_new, s_new], dim=1)

        # 如果有足够的历史嵌入，使用历史融合模块
        if len(self.history_embeddings) >= self.history_window:
            # 使用注意力机制融合历史信息
            Z_t_minus_1 = self.history_fusion(H_t, self.history_embeddings)

            # 使用HyperGRU更新嵌入
            Z_t = self.hyper_gru(H_t, Z_t_minus_1)
        else:
            # 如果历史不足，直接使用当前隐藏表示
            Z_t = H_t

        # 在训练模式下更新历史嵌入
        if is_training and Z_t is not None:
            # 更新历史嵌入队列
            self.history_embeddings.append(Z_t.detach())
            if len(self.history_embeddings) > self.history_window:
                self.history_embeddings.pop(0)

        return Z_t

    def predict_link(self, src_nodes, dst_nodes):
        """
        链接预测

        Args:
            src_nodes: 源节点列表
            dst_nodes: 目标节点列表

        Returns:
            scores: 预测分数
        """
        # 获取节点嵌入
        src_emb = self.node_embeddings[src_nodes]
        dst_emb = self.node_embeddings[dst_nodes]

        # 分割嵌入到不同的空间
        src_e = src_emb[:, :self.e_dim]
        src_b = src_emb[:, self.e_dim:self.e_dim + self.b_dim]
        src_s = src_emb[:, self.e_dim + self.b_dim:]

        dst_e = dst_emb[:, :self.e_dim]
        dst_b = dst_emb[:, self.e_dim:self.e_dim + self.b_dim]
        dst_s = dst_emb[:, self.e_dim + self.b_dim:]

        # 计算各空间的距离
        e_dist = GeometricOperations.euclidean_distance(src_e, dst_e)
        b_dist = GeometricOperations.hyperbolic_distance(src_b, dst_b, self.b_curvature)
        s_dist = GeometricOperations.spherical_distance(src_s, dst_s, self.s_curvature)

        # 使用费米-狄拉克解码器计算链接概率，论文公式 (19)
        r = 2.0  # 超参数r
        s = 1.0  # 超参数s

        # 组合各空间距离，基于维度比例加权
        e_weight = self.e_dim / self.embedding_dim
        b_weight = self.b_dim / self.embedding_dim
        s_weight = self.s_dim / self.embedding_dim

        combined_dist = e_weight * e_dist + b_weight * b_dist + s_weight * s_dist

        # 应用费米-狄拉克解码器
        scores = torch.sigmoid(r * (s - combined_dist))

        return scores


# HAT模型 - 简化实现
class HAT(nn.Module):
    """双曲注意力网络模型"""

    def __init__(self, snapshot_graphs, config):
        super(HAT, self).__init__()

        self.num_nodes = snapshot_graphs[0].num_nodes()
        self.embedding_dim = config['embedding_dim']

        # 初始化节点嵌入
        self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, self.embedding_dim))

        # 双曲空间曲率参数
        self.curvature = nn.Parameter(torch.tensor([config.get('curvature', -1.0)]))

        # 图注意力层
        self.attention = nn.Linear(self.embedding_dim, self.embedding_dim)

    def forward(self, graph, *args, **kwargs):
        """前向传播"""
        # 这是一个简化的实现
        with graph.local_scope():
            graph.ndata['h'] = self.node_embeddings
            graph.update_all(fn.copy_u('h', 'm'), fn.ReduceFunction.mean('m', 'neigh'))
            return graph.ndata['neigh']

    def predict_link(self, src_nodes, dst_nodes):
        """链接预测"""
        src_emb = self.node_embeddings[src_nodes]
        dst_emb = self.node_embeddings[dst_nodes]

        # 计算双曲距离并转为相似度分数
        dist = GeometricOperations.hyperbolic_distance(src_emb, dst_emb, self.curvature)
        return torch.exp(-dist)


# HGCN模型 - 简化实现
class HGCN(nn.Module):
    """双曲图卷积网络模型"""

    def __init__(self, snapshot_graphs, config):
        super(HGCN, self).__init__()

        self.num_nodes = snapshot_graphs[0].num_nodes()
        self.embedding_dim = config['embedding_dim']

        # 初始化节点嵌入
        self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, self.embedding_dim))

        # 双曲空间曲率参数
        self.curvature = nn.Parameter(torch.tensor([config.get('curvature', -1.0)]))

        # 图卷积层
        self.weight = nn.Parameter(torch.randn(self.embedding_dim, self.embedding_dim))

    def forward(self, graph, *args, **kwargs):
        """前向传播"""
        # 简化实现
        with graph.local_scope():
            graph.ndata['h'] = torch.matmul(self.node_embeddings, self.weight)
            graph.update_all(fn.copy_u('h', 'm'), fn.ReduceFunction.mean('m', 'neigh'))
            return graph.ndata['neigh']

    def predict_link(self, src_nodes, dst_nodes):
        """链接预测"""
        src_emb = self.node_embeddings[src_nodes]
        dst_emb = self.node_embeddings[dst_nodes]

        # 计算双曲距离并转为相似度分数
        dist = GeometricOperations.hyperbolic_distance(src_emb, dst_emb, self.curvature)
        return torch.exp(-dist)


# EvolveGCN模型 - 简化实现
class EvolveGCN(nn.Module):
    """EvolveGCN模型实现"""

    def __init__(self, snapshot_graphs, config):
        super(EvolveGCN, self).__init__()

        self.num_nodes = snapshot_graphs[0].num_nodes()
        self.embedding_dim = config['embedding_dim']

        # 初始化节点嵌入
        self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, self.embedding_dim))

        # GCN层权重
        self.gc_weights = nn.Parameter(torch.randn(self.embedding_dim, self.embedding_dim))

        # 权重更新RNN
        self.weight_rnn = nn.GRUCell(self.embedding_dim ** 2, self.embedding_dim ** 2)

        # 存储历史权重
        self.weights_history = []

    def forward(self, graph, is_training=True):
        """前向传播"""
        # 更新权重
        if is_training and len(self.weights_history) > 0:
            flattened_weights = self.gc_weights.view(-1)
            flattened_history = self.weights_history[-1].view(-1)
            new_flattened_weights = self.weight_rnn(flattened_history, flattened_weights)
            self.gc_weights.data = new_flattened_weights.view(self.embedding_dim, self.embedding_dim)

        # 图卷积操作
        with graph.local_scope():
            # 线性变换
            h = torch.matmul(self.node_embeddings, self.gc_weights)

            # 消息传递
            graph.ndata['h'] = h
            graph.update_all(fn.copy_u('h', 'm'), fn.ReduceFunction.mean('m', 'neigh'))

            result = graph.ndata['neigh']

        # 存储权重历史
        if is_training:
            self.weights_history.append(self.gc_weights.detach())
            if len(self.weights_history) > 10:  # 限制历史长度
                self.weights_history.pop(0)

        return result

    def predict_link(self, src_nodes, dst_nodes):
        """链接预测"""
        src_emb = self.node_embeddings[src_nodes]
        dst_emb = self.node_embeddings[dst_nodes]

        # 计算内积并应用sigmoid激活
        scores = torch.sum(src_emb * dst_emb, dim=1)
        return torch.sigmoid(scores)


# GRUGCN模型 - 简化实现
class GRUGCN(nn.Module):
    """GRUGCN模型实现"""

    def __init__(self, snapshot_graphs, config):
        super(GRUGCN, self).__init__()

        self.num_nodes = snapshot_graphs[0].num_nodes()
        self.embedding_dim = config['embedding_dim']

        # 初始化节点嵌入
        self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, self.embedding_dim))

        # GCN层
        self.gc_weight = nn.Parameter(torch.randn(self.embedding_dim, self.embedding_dim))

        # GRU层
        self.gru = nn.GRUCell(self.embedding_dim, self.embedding_dim)

        # 历史隐藏状态
        self.hidden_state = None

    def forward(self, graph, is_training=True):
        """前向传播"""
        # GCN操作
        with graph.local_scope():
            h = torch.matmul(self.node_embeddings, self.gc_weight)

            graph.ndata['h'] = h
            graph.update_all(fn.copy_u('h', 'm'), fn.ReduceFunction.mean('m', 'neigh'))

            gcn_output = graph.ndata['neigh']

        # GRU更新
        if self.hidden_state is None:
            self.hidden_state = torch.zeros_like(gcn_output)

        new_hidden = self.gru(gcn_output, self.hidden_state)

        # 更新隐藏状态
        if is_training:
            self.hidden_state = new_hidden.detach()

        return new_hidden

    def predict_link(self, src_nodes, dst_nodes):
        """链接预测"""
        src_emb = self.node_embeddings[src_nodes]
        dst_emb = self.node_embeddings[dst_nodes]

        # 计算内积并应用sigmoid激活
        scores = torch.sum(src_emb * dst_emb, dim=1)
        return torch.sigmoid(scores)


# HTGN模型 - 简化实现
class HTGN(nn.Module):
    """双曲时序图网络模型"""

    def __init__(self, snapshot_graphs, config):
        super(HTGN, self).__init__()

        self.num_nodes = snapshot_graphs[0].num_nodes()
        self.embedding_dim = config['embedding_dim']

        # 初始化节点嵌入
        self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, self.embedding_dim))

        # 双曲图神经网络
        self.gcn_weight = nn.Parameter(torch.randn(self.embedding_dim, self.embedding_dim))

        # 双曲GRU
        self.gru = HyperGRU(self.embedding_dim, self.embedding_dim)

        # 双曲空间曲率参数
        self.curvature = nn.Parameter(torch.tensor([config.get('curvature', -1.0)]))

        # 历史嵌入列表
        self.history_embeddings = []

    def forward(self, graph, is_training=True):
        """前向传播"""
        # 双曲图神经网络
        with graph.local_scope():
            # 将嵌入从双曲空间映射到切空间
            origin = torch.zeros_like(self.node_embeddings[0]).unsqueeze(0).to(self.node_embeddings.device)
            tangent_emb = GeometricOperations.log_map(origin, self.node_embeddings, self.curvature)

            # 线性变换
            h = torch.matmul(tangent_emb, self.gcn_weight)

            # 消息传递
            graph.ndata['h'] = h
            graph.update_all(fn.copy_u('h', 'm'), fn.ReduceFunction.mean('m', 'neigh'))

            neigh = graph.ndata['neigh']

            # 映射回双曲空间
            current_emb = GeometricOperations.exp_map(origin, neigh, self.curvature)

        # 时序更新
        if len(self.history_embeddings) > 0:
            # 使用最近的历史嵌入
            prev_emb = self.history_embeddings[-1]

            # 使用双曲GRU更新
            updated_emb = self.gru(current_emb, prev_emb)
        else:
            updated_emb = current_emb

        # 更新历史嵌入
        if is_training:
            self.history_embeddings.append(updated_emb.detach())
            if len(self.history_embeddings) > 10:  # 限制历史长度
                self.history_embeddings.pop(0)

        return updated_emb

    def predict_link(self, src_nodes, dst_nodes):
        """链接预测"""
        src_emb = self.node_embeddings[src_nodes]
        dst_emb = self.node_embeddings[dst_nodes]

        # 计算双曲距离并转为相似度分数
        dist = GeometricOperations.hyperbolic_distance(src_emb, dst_emb, self.curvature)
        return torch.exp(-dist)


# HGWaveNet模型 - 简化实现
class HGWaveNet(nn.Module):
    """双曲图波网络模型"""

    def __init__(self, snapshot_graphs, config):
        super(HGWaveNet, self).__init__()

        self.num_nodes = snapshot_graphs[0].num_nodes()
        self.embedding_dim = config['embedding_dim']

        # 初始化节点嵌入
        self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, self.embedding_dim))

        # 双曲图卷积权重
        self.gc_weight = nn.Parameter(torch.randn(self.embedding_dim, self.embedding_dim))

        # 扩张卷积权重
        self.dilated_conv_weights = nn.ParameterList([
            nn.Parameter(torch.randn(self.embedding_dim, self.embedding_dim)) for _ in range(3)
        ])

        # 双曲空间曲率参数
        self.curvature = nn.Parameter(torch.tensor([config.get('curvature', -1.0)]))

        # 历史嵌入列表
        self.history_embeddings = []

    def forward(self, graph, is_training=True):
        """前向传播"""
        # 双曲图卷积
        with graph.local_scope():
            # 将嵌入从双曲空间映射到切空间
            origin = torch.zeros_like(self.node_embeddings[0]).unsqueeze(0).to(self.node_embeddings.device)
            tangent_emb = GeometricOperations.log_map(origin, self.node_embeddings, self.curvature)

            # 线性变换
            h = torch.matmul(tangent_emb, self.gc_weight)

            # 消息传递
            graph.ndata['h'] = h
            graph.update_all(fn.copy_u('h', 'm'), fn.ReduceFunction.mean('m', 'neigh'))

            neigh = graph.ndata['neigh']

            # 映射回双曲空间
            current_emb = GeometricOperations.exp_map(origin, neigh, self.curvature)

        # 如果有历史嵌入，使用扩张卷积
        if len(self.history_embeddings) >= 3:  # 需要至少3个历史快照
            # 选择最近的3个历史快照
            recent_history = self.history_embeddings[-3:]

            # 将嵌入映射到切空间
            tangent_history = [
                GeometricOperations.log_map(origin, h, self.curvature) for h in recent_history
            ]
            tangent_current = GeometricOperations.log_map(origin, current_emb, self.curvature)

            # 扩张卷积操作
            result = tangent_current
            for i, hist in enumerate(tangent_history):
                result = result + torch.matmul(hist, self.dilated_conv_weights[i])

            # 映射回双曲空间
            updated_emb = GeometricOperations.exp_map(origin, result, self.curvature)
        else:
            updated_emb = current_emb

        # 更新历史嵌入
        if is_training:
            self.history_embeddings.append(updated_emb.detach())
            if len(self.history_embeddings) > 20:  # 限制历史长度
                self.history_embeddings.pop(0)

        return updated_emb

    def predict_link(self, src_nodes, dst_nodes):
        """链接预测"""
        src_emb = self.node_embeddings[src_nodes]
        dst_emb = self.node_embeddings[dst_nodes]

        # 计算双曲距离并转为相似度分数
        dist = GeometricOperations.hyperbolic_distance(src_emb, dst_emb, self.curvature)
        return torch.exp(-dist)


# 模型加载函数
def load_model(model_name, snapshot_graphs, config, device):
    """
    加载指定的模型

    Args:
        model_name: 模型名称
        snapshot_graphs: 离散动态图快照列表
        config: 模型配置
        device: 计算设备

    Returns:
        model: 初始化的模型实例
    """
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
        model_class = models[model_name.lower()]
        model = model_class(snapshot_graphs, config).to(device)
        return model
    else:
        raise ValueError(f"未知模型: {model_name}")
