import os
import dgl
import torch
import numpy as np
from tqdm import tqdm
from utils import save_pickle, load_pickle


def create_snapshots(data_path, dataset_name, window_size, num_snapshots):
    """
    从原始数据创建离散动态图快照

    参数:
        data_path: 数据文件路径
        dataset_name: 数据集名称
        window_size: 时间窗口大小
        num_snapshots: 快照数量，如果为None则自动计算

    返回:
        snapshots: 离散动态图快照列表
    """
    print(f"正在为{dataset_name}创建离散动态图快照...")

    # 创建输出目录
    preprocessed_dir = os.path.join('dataset', 'preprocessed', dataset_name)
    os.makedirs(preprocessed_dir, exist_ok=True)

    # 检查是否已存在处理好的快照
    snapshots_file = os.path.join(preprocessed_dir, 'snapshots.pkl')
    if os.path.exists(snapshots_file):
        print("发现预处理的快照文件，直接加载...")
        return load_pickle(snapshots_file)

    # 加载原始数据（假设格式为源节点，目标节点，时间戳）
    raw_data_file = os.path.join(data_path, f'{dataset_name}.txt')
    data = []
    node_set = set()

    print("加载原始数据...")
    with open(raw_data_file, 'r') as f:
        for line in f:
            src, dst, timestamp = line.strip().split()
            src, dst, timestamp = int(src), int(dst), float(timestamp)
            data.append((src, dst, timestamp))
            node_set.add(src)
            node_set.add(dst)

    # 按时间戳排序
    data.sort(key=lambda x: x[2])

    # 确定时间范围
    min_time = data[0][2]
    max_time = data[-1][2]
    total_time = max_time - min_time

    # 如果未指定快照数，则使用自动计算的窗口数
    #if num_snapshots is None:
        #num_snapshots = int(total_time / window_size)

    # 将时间窗口大小调整为覆盖整个时间范围
    window_size = total_time / num_snapshots

    # 创建节点映射，将节点ID映射到连续整数
    node_map = {node: idx for idx, node in enumerate(sorted(node_set))}
    num_nodes = len(node_map)

    # 创建快照
    snapshots = []
    edges_per_snapshot = [[] for _ in range(num_snapshots)]

    print("分配边到快照...")
    for src, dst, timestamp in tqdm(data):
        snapshot_idx = min(int((timestamp - min_time) / window_size), num_snapshots - 1)
        edges_per_snapshot[snapshot_idx].append((node_map[src], node_map[dst]))

    # 构建DGL图
    print("构建DGL图...")
    for i in tqdm(range(num_snapshots)):
        edges = edges_per_snapshot[i]
        if not edges:
            # 如果没有边，创建一个空图
            g = dgl.graph(([], []), num_nodes=num_nodes)
        else:
            src, dst = zip(*edges)
            g = dgl.graph((src, dst), num_nodes=num_nodes)
        snapshots.append(g)

    # 保存快照
    save_pickle(snapshots_file, snapshots)

    # 保存节点映射以便后续使用
    save_pickle(os.path.join(preprocessed_dir, 'node_map.pkl'), node_map)

    print(f"成功创建{num_snapshots}个离散动态图快照")
    return snapshots


def create_train_val_test_split(snapshots, train_ratio=0.7, val_ratio=0.1):
    """
    将快照分割为训练、验证和测试集

    参数:
        snapshots: 离散动态图快照列表
        train_ratio: 训练集比例
        val_ratio: 验证集比例

    返回:
        split_dict: 包含分割信息的字典
    """
    num_snapshots = len(snapshots)
    train_size = int(num_snapshots * train_ratio)
    val_size = int(num_snapshots * val_ratio)
    test_size = num_snapshots - train_size - val_size

    # 确保至少有一个快照用于验证和测试
    if val_size == 0:
        val_size = 1
        train_size = max(1, train_size - 1)
    if test_size == 0:
        test_size = 1
        train_size = max(1, train_size - 1)

    # 分割索引
    train_snapshots = list(range(train_size))
    val_snapshots = list(range(train_size, train_size + val_size))
    test_snapshots = list(range(train_size + val_size, num_snapshots))

    # 创建分割字典
    split_dict = {
        'train': train_snapshots,
        'valid': val_snapshots,
        'test': test_snapshots
    }

    return split_dict


def prepare_link_prediction_data(snapshots, split_dict):
    """
    准备链接预测任务的数据

    参数:
        snapshots: 离散动态图快照列表
        split_dict: 包含分割信息的字典

    返回:
        split_edge: 包含训练、验证和测试边的字典
    """
    # 初始化分割边字典
    split_edge = {
        'train': {'edge': []},
        'valid': {'edge': [], 'edge_neg': []},
        'test': {'edge': [], 'edge_neg': []}
    }

    num_nodes = snapshots[0].num_nodes()

    # 收集训练集边
    train_edges = set()
    for i in split_dict['train']:
        src, dst = snapshots[i].edges()
        for s, d in zip(src.tolist(), dst.tolist()):
            train_edges.add((s, d))

    # 收集验证集边
    valid_edges = set()
    for i in split_dict['valid']:
        src, dst = snapshots[i].edges()
        for s, d in zip(src.tolist(), dst.tolist()):
            valid_edges.add((s, d))

    # 移除已在训练集中的边
    valid_edges = valid_edges - train_edges

    # 收集测试集边
    test_edges = set()
    for i in split_dict['test']:
        src, dst = snapshots[i].edges()
        for s, d in zip(src.tolist(), dst.tolist()):
            test_edges.add((s, d))

    # 移除已在训练集或验证集中的边
    test_edges = test_edges - train_edges - valid_edges

    # 转换为张量
    split_edge['train']['edge'] = torch.tensor(list(train_edges))
    split_edge['valid']['edge'] = torch.tensor(list(valid_edges))
    split_edge['test']['edge'] = torch.tensor(list(test_edges))

    # 生成负样本边
    # 为验证集生成负样本
    valid_neg_edges = generate_negative_edges(snapshots, train_edges, valid_edges, num_nodes, len(valid_edges))
    split_edge['valid']['edge_neg'] = torch.tensor(valid_neg_edges)

    # 为测试集生成负样本
    test_neg_edges = generate_negative_edges(snapshots, train_edges.union(valid_edges), test_edges, num_nodes,
                                             len(test_edges))
    split_edge['test']['edge_neg'] = torch.tensor(test_neg_edges)

    return split_edge


def generate_negative_edges(snapshots, existing_edges, positive_edges, num_nodes, num_samples):
    """
    生成负样本边

    参数:
        snapshots: 离散动态图快照列表
        existing_edges: 现有边的集合
        positive_edges: 正样本边的集合
        num_nodes: 节点数量
        num_samples: 需要生成的负样本数量

    返回:
        negative_edges: 负样本边列表
    """
    all_edges = existing_edges.union(positive_edges)
    negative_edges = []

    # 为效率考虑，预先计算每个节点的邻居，避免重复计算
    neighbors = {}
    for s, d in all_edges:
        if s not in neighbors:
            neighbors[s] = set()
        neighbors[s].add(d)

    # 生成负样本边
    pbar = tqdm(total=num_samples, desc="生成负样本边")
    while len(negative_edges) < num_samples:
        # 随机选择源节点和目标节点
        src = np.random.randint(0, num_nodes)
        dst = np.random.randint(0, num_nodes)

        # 确保不是自环且不在现有边中
        if src != dst and (src not in neighbors or dst not in neighbors[src]):
            negative_edges.append([src, dst])
            pbar.update(1)

    pbar.close()
    return negative_edges


def prepare_new_link_prediction_data(snapshots, split_dict):
    """
    准备新链接预测任务的数据

    参数:
        snapshots: 离散动态图快照列表
        split_dict: 包含分割信息的字典

    返回:
        split_edge: 包含训练、验证和测试边的字典，其中验证和测试边是"新"边
    """
    # 初始化分割边字典
    split_edge = {
        'train': {'edge': []},
        'valid': {'edge': [], 'edge_neg': []},
        'test': {'edge': [], 'edge_neg': []}
    }

    num_nodes = snapshots[0].num_nodes()

    # 收集训练集边
    train_edges = set()
    for i in split_dict['train']:
        src, dst = snapshots[i].edges()
        for s, d in zip(src.tolist(), dst.tolist()):
            train_edges.add((s, d))

    # 收集所有历史边（用于确定"新"边）
    historical_edges = train_edges.copy()

    # 收集验证集边（只保留新边）
    valid_edges = set()
    for i in split_dict['valid']:
        src, dst = snapshots[i].edges()
        for s, d in zip(src.tolist(), dst.tolist()):
            if (s, d) not in historical_edges:
                valid_edges.add((s, d))
                historical_edges.add((s, d))  # 更新历史边

    # 收集测试集边（只保留新边）
    test_edges = set()
    for i in split_dict['test']:
        src, dst = snapshots[i].edges()
        for s, d in zip(src.tolist(), dst.tolist()):
            if (s, d) not in historical_edges:
                test_edges.add((s, d))
                historical_edges.add((s, d))  # 更新历史边

    # 转换为张量
    split_edge['train']['edge'] = torch.tensor(list(train_edges))
    split_edge['valid']['edge'] = torch.tensor(list(valid_edges))
    split_edge['test']['edge'] = torch.tensor(list(test_edges))

    # 生成负样本边
    # 为验证集生成负样本
    valid_neg_edges = generate_negative_edges(snapshots, historical_edges - valid_edges, valid_edges, num_nodes,
                                              len(valid_edges))
    split_edge['valid']['edge_neg'] = torch.tensor(valid_neg_edges)

    # 为测试集生成负样本
    test_neg_edges = generate_negative_edges(snapshots, historical_edges - test_edges, test_edges, num_nodes,
                                             len(test_edges))
    split_edge['test']['edge_neg'] = torch.tensor(test_neg_edges)

    return split_edge


def load_data(args):
    """
    加载数据集

    参数:
        args: 参数对象，包含dataset属性

    返回:
        snapshots: 离散动态图快照列表
        split_edge: 包含训练、验证和测试边的字典
    """
    dataset_name = args.dataset
    num_snapshots = args.num_snapshots

    # 检查预处理的数据是否存在
    preprocessed_dir = os.path.join('dataset', 'preprocessed', dataset_name)
    snapshots_file = os.path.join(preprocessed_dir, 'snapshots.pkl')

    if os.path.exists(snapshots_file):
        print(f"加载预处理的{dataset_name}数据...")
        snapshots = load_pickle(snapshots_file)
    else:
        # 创建快照
        data_path = os.path.join('dataset', 'raw')
        snapshots = create_snapshots(data_path, dataset_name, num_snapshots)

    # 创建数据分割
    split_dict = create_train_val_test_split(snapshots, train_ratio=0.8, val_ratio=0.1)

    # 准备链接预测数据
    if args.task == 'link_prediction':
        split_edge = prepare_link_prediction_data(snapshots, split_dict)
    elif args.task == 'new_link_prediction':
        split_edge = prepare_new_link_prediction_data(snapshots, split_dict)
    else:
        raise ValueError(f"未知任务: {args.task}")

    return snapshots, split_edge
