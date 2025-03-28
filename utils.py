import pickle as pkl
import yaml
import json
import numpy as np
import gc
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




