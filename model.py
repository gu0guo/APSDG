#6个基准模型和本文提出的APSDG模型的定义
import torch
import torch.nn as nn
from utils import *
###############EvolveGCN模型###############
class EvolveGCN(nn.Module):
    def __init__(self, snapshots, emb_dim, num_layers, device):

    def forward(self, ):

###############GRUGCN模型###############
class GRUGCN(nn.Module):
    def __init__(self, snapshots, emb_dim, num_layers, device):

    def forward(self, ):

###############HAT模型###############
class HAT(nn.Module):
    def __init__(self,snapshots, emb_dim, num_layers, device):

    def forward(self,):

###############HGCN模型###############
class HGCN(nn.Module):
    def __init__(self, snapshots, emb_dim, num_layers, device):

    def forward(self, ):

###############HGWaveNet模型###############
class HGWaveNet(nn.Module):
    def __init__(self, snapshots ,emb_dim, num_layers, device):

    def forward(self, ):

###############HTGN模型###############
class HTGN(nn.Module):
    def __init__(self, snapshots ,emb_dim, num_layers, device):

    def forward(self, ):

###############APSDG模型###############
class APSDG(nn.Module):
    def __init__(self, snapshots ,emb_dim, num_layers, device):
        super(APSDG, self).__init__()
        #初始化模型参数和子模块
        #3.4.1维度比例初始化
        #3.4.3曲率初始化
    def forward(self, ):
        # 3.1 积空间消息传递#卷积操作
            #3.1.1积空间特征初始化
            #3.1.2积空间线性变换
            #3.1.3积空间领域聚合
            #3.1.4积空间非线性激活
        # 3.2 历史信息融合
            #3.2.1) 历史信息存储单元
            #3.2.2) 注意力聚合机制
            #3.2.3HyperGRU单元
        #得到新的节点嵌入

#积空间图神经网络，负责在积空间中进行消息传递
class ProductSpaceGNN(nn.Module):
    def __init__(self,)
        super(ProductSpaceGNN, self).__init__()

    def foward(self,):


#历史信息存储单元
class HistoryStorage(nn.Module):
    def __init__(self,):
        super(HistoryStorage, self).__init__()

    def forward(self,):

#注意力聚合机制
class AttentionAggregation(nn.Module):
    def __init__(self,):
        super(AttentionAggregation, self).__init__()

    def forward(self,):


#HyperGRU单元
class HyperGRU(nn.Module):
    def __init__(self,):
        super(HyperGRU, self).__init__()

    def forward(self,):



##############################加载模型##############################
def load_model(snapshots, model, dataset, device):
    if model == 'evolvegcn':
        config_flie='config/evolvegcn-config.yaml'
        config = defaultdict(int)
        config.update(load_yaml(config_file))
        model = EvolveGCN(snapshots, config['model']['emb_dim'], config['model']['num_layers'], device)

    elif model == 'grugcn':
        config_flie='config/grugcn-config.yaml'
        config = defaultdict(int)
        config.update(load_yaml(config_file))
        model = GRUGCN(snapshots, config['model']['emb_dim'], config['model']['num_layers'], device)

    elif model == 'hat':
        config_flie='config/hat-config.yaml'
        config = defaultdict(int)
        config.update(load_yaml(config_file))
        model = HAT(snapshots, config['model']['emb_dim'], config['model']['num_layers'], device)

    elif model == 'hgcn':
        config_flie='config/hgcn-config.yaml'
        config = defaultdict(int)
        config.update(load_yaml(config_file))
        model = HGCN(snapshots, config['model']['emb_dim'], config['model']['num_layers'], device)

    elif model == 'hgwavenet':
        config_flie='config/hgwavenet-config.yaml'
        config = defaultdict(int)
        config.update(load_yaml(config_file))
        model = HGWaveNet(snapshots, config['model']['emb_dim'], config['model']['num_layers'], device)

    elif model == 'htgn':
        config_flie='config/htgn-config.yaml'
        config = defaultdict(int)
        config.update(load_yaml(config_file))
        model = HTGN(snapshots, config['model']['emb_dim'], config['model']['num_layers'], device)

    return model.to(device)