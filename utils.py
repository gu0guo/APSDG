import pickle as pkl
import yaml
import json
import torch
import numpy as np
import gc
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score


############### utils ################

def print_dict(d):
    """打印字典内容的辅助函数"""
    for key in d:
        print(key, ":", d[key])


################# io #################

def save_pickle(filename, obj):
    """将对象保存为pickle文件"""
    with open(filename, "wb") as f:
        pkl.dump(obj, f)


def load_pickle(filename):
    """从pickle文件加载对象"""
    with open(filename, "rb") as f:
        gc.disable()
        obj = pkl.load(f)
        gc.enable()
    return obj


def save_json(filename, obj):
    """将对象保存为json文件"""
    with open(filename, 'w') as f:
        json.dump(obj, f, indent=4)


def load_json(filename):
    """从json文件加载对象"""
    with open(filename, 'r') as f:
        obj = json.load(f)
    return obj


def load_yaml(filename):
    """从yaml文件加载配置"""
    with open(filename, 'r') as f:
        obj = yaml.load(f, Loader=yaml.FullLoader)
    return obj


def save_yaml(filename, obj):
    """将配置保存为yaml文件"""
    with open(filename, 'w') as f:
        yaml.dump(obj, f, indent=4, sort_keys=False)


############## metric ##############

def evaluate_auc_ap(pos_pred, neg_pred):
    """
    计算AUC和AP评估指标

    参数:
        pos_pred: 正样本的预测分数
        neg_pred: 负样本的预测分数

    返回:
        auc: ROC曲线下面积
        ap: 平均精度
    """
    # 准备标签和预测值
    y_true = np.concatenate([np.ones(len(pos_pred)), np.zeros(len(neg_pred))])
    y_pred = np.concatenate([pos_pred, neg_pred])

    # 计算AUC和AP
    auc = roc_auc_score(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)

    return auc, ap


def dot_product(src_emb, dst_emb):
    """计算点积，用于链接预测"""
    if src_emb.shape != dst_emb.shape:
        return (src_emb.unsqueeze(-2) * dst_emb).sum(dim=-1)
    else:
        return (src_emb * dst_emb).sum(dim=-1)


class EdgeDataloader:
    """边数据加载器，用于批处理"""

    def __init__(self, edge_set, batch_size):
        self.edge_set = edge_set
        self.batch_size = batch_size
        self.num_edges = edge_set.size(0)
        self.indices = torch.randperm(self.num_edges)
        self.current_pos = 0

    def __iter__(self):
        self.current_pos = 0
        self.indices = torch.randperm(self.num_edges)
        return self

    def __next__(self):
        if self.current_pos >= self.num_edges:
            raise StopIteration
        else:
            batch_indices = self.indices[self.current_pos:min(self.current_pos + self.batch_size, self.num_edges)]
            self.current_pos += self.batch_size
            return self.edge_set[batch_indices]

    def __len__(self):
        return (self.num_edges + self.batch_size - 1) // self.batch_size


class SnapshotBatchSampler:
    """快照批量采样器，用于离散动态图数据"""

    def __init__(self, snapshots, batch_size):
        self.snapshots = snapshots
        self.batch_size = batch_size
        self.num_snapshots = len(snapshots)

    def __iter__(self):
        for i in range(self.num_snapshots):
            edges = self.snapshots[i].edges()
            num_edges = edges[0].size(0)
            for j in range(0, num_edges, self.batch_size):
                end_idx = min(j + self.batch_size, num_edges)
                batch_edges = (edges[0][j:end_idx], edges[1][j:end_idx])
                yield i, batch_edges
