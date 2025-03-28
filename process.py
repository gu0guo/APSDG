import torch
import time
import numpy as np
from tqdm import tqdm
from utils import evaluate_auc_ap


def train_epoch(model, snapshot_graphs, split_edge, optimizer, device, args):
    """
    训练一个周期

    参数:
        model: 模型实例
        snapshot_graphs: 离散动态图快照列表
        split_edge: 训练/验证/测试的边集合
        optimizer: 优化器
        device: 设备
        args: 参数对象

    返回:
        loss: 平均损失
    """
    model.train()

    # 获取训练边
    train_edge = split_edge['train']['edge'].to(device)

    # 计算总损失
    total_loss = 0
    num_batches = 0

    # 遍历所有训练快照
    train_snapshots = train_edge.shape[0]
    batch_size = args.batch_size

    # 分批次处理
    pbar = tqdm(range(0, train_snapshots, batch_size), desc="训练批次")
    for start_idx in pbar:
        end_idx = min(start_idx + batch_size, train_snapshots)
        batch_edge = train_edge[start_idx:end_idx]

        # 获取当前快照的源节点和目标节点
        src_nodes = batch_edge[:, 0]
        dst_nodes = batch_edge[:, 1]

        # 更新模型嵌入
        for i in range(len(snapshot_graphs)):
            # 使用历史快照更新模型
            model.forward(snapshot_graphs[i], is_training=True)

        # 预测链接概率
        pos_score = model.predict_link(src_nodes, dst_nodes)

        # 负采样
        neg_src_nodes = src_nodes
        neg_dst_nodes = torch.randint(0, model.num_nodes, (len(src_nodes),), device=device)

        # 预测负样本的概率
        neg_score = model.predict_link(neg_src_nodes, neg_dst_nodes)

        # 计算损失
        pos_loss = -torch.log(pos_score + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_score + 1e-15).mean()
        loss = pos_loss + neg_loss

        # 更新模型参数
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
        optimizer.step()

        # 更新总损失
        total_loss += loss.item()
        num_batches += 1

        # 更新进度条
        pbar.set_postfix({'loss': loss.item()})

    # 计算平均损失
    avg_loss = total_loss / num_batches

    return avg_loss


def validate(model, snapshot_graphs, split_edge, device):
    """
    验证模型性能

    参数:
        model: 模型实例
        snapshot_graphs: 离散动态图快照列表
        split_edge: 训练/验证/测试的边集合
        device: 设备

    返回:
        auc: ROC曲线下面积
        ap: 平均精度
    """
    model.eval()

    # 获取验证边和负样本边
    valid_edge = split_edge['valid']['edge'].to(device)
    valid_edge_neg = split_edge['valid']['edge_neg'].to(device)

    with torch.no_grad():
        # 更新模型嵌入
        for i in range(len(snapshot_graphs)):
            # 使用历史快照更新模型
            model.forward(snapshot_graphs[i], is_training=False)

        # 预测正样本分数
        pos_score_list = []
        for src, dst in valid_edge:
            pos_score_list.append(model.predict_link(src.unsqueeze(0), dst.unsqueeze(0)).cpu().numpy()[0])

        # 预测负样本分数
        neg_score_list = []
        for src, dst in valid_edge_neg:
            neg_score_list.append(model.predict_link(src.unsqueeze(0), dst.unsqueeze(0)).cpu().numpy()[0])

        # 计算AUC和AP
        auc, ap = evaluate_auc_ap(np.array(pos_score_list), np.array(neg_score_list))

    return auc, ap


def test(model, snapshot_graphs, split_edge, device):
    """
    测试模型性能

    参数:
        model: 模型实例
        snapshot_graphs: 离散动态图快照列表
        split_edge: 训练/验证/测试的边集合
        device: 设备

    返回:
        results: 包含测试性能指标的字典
    """
    model.eval()

    # 获取测试边和负样本边
    test_edge = split_edge['test']['edge'].to(device)
    test_edge_neg = split_edge['test']['edge_neg'].to(device)

    with torch.no_grad():
        # 更新模型嵌入
        for i in range(len(snapshot_graphs)):
            # 使用历史快照更新模型
            model.forward(snapshot_graphs[i], is_training=False)

        # 预测正样本分数
        pos_score_list = []
        for src, dst in test_edge:
            pos_score_list.append(model.predict_link(src.unsqueeze(0), dst.unsqueeze(0)).cpu().numpy()[0])

        # 预测负样本分数
        neg_score_list = []
        for src, dst in test_edge_neg:
            neg_score_list.append(model.predict_link(src.unsqueeze(0), dst.unsqueeze(0)).cpu().numpy()[0])

        # 计算AUC和AP
        auc, ap = evaluate_auc_ap(np.array(pos_score_list), np.array(neg_score_list))

    # 构建结果字典
    results = {
        'test_auc': auc,
        'test_ap': ap,
    }

    return results


def train(model, snapshot_graphs, split_edge, args, device):
    """
    训练模型

    参数:
        model: 模型实例
        snapshot_graphs: 离散动态图快照列表
        split_edge: 训练/验证/测试的边集合
        args: 参数对象
        device: 设备

    返回:
        best_model: 性能最佳的模型
    """
    # 初始化优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 用于早停的变量
    best_valid_auc = 0
    best_valid_ap = 0
    best_model = None
    patience_counter = 0

    # 训练循环
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        # 训练一个周期
        loss = train_epoch(model, snapshot_graphs, split_edge, optimizer, device, args)

        # 验证
        valid_auc, valid_ap = validate(model, snapshot_graphs, split_edge, device)

        # 更新最佳模型
        if valid_auc + valid_ap > best_valid_auc + best_valid_ap:
            best_valid_auc = valid_auc
            best_valid_ap = valid_ap
            best_model = model
            patience_counter = 0
        else:
            patience_counter += 1

        # 早停判断
        if patience_counter >= args.patience:
            print(f"早停 - 本轮训练结束")
            break

        # 打印本轮训练信息
        end_time = time.time()
        epoch_time = end_time - start_time
        print(
            f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Valid AUC: {valid_auc:.4f}, Valid AP: {valid_ap:.4f}, Time: {epoch_time:.2f}s')

    return best_model or model  # 如果没有找到最佳模型，则返回最后一个模型


def evaluate_link_prediction(model, snapshot_graphs, split_edge, device):
    """
    评估链接预测任务性能

    参数:
        model: 模型实例
        snapshot_graphs: 离散动态图快照列表
        split_edge: 训练/验证/测试的边集合
        device: 设备

    返回:
        results: 包含性能指标的字典
    """
    results = test(model, snapshot_graphs, split_edge, device)
    print(f"链接预测任务结果: AUC = {results['test_auc']:.4f}, AP = {results['test_ap']:.4f}")
    return results


def evaluate_new_link_prediction(model, snapshot_graphs, split_edge, device):
    """
    评估新链接预测任务性能

    参数:
        model: 模型实例
        snapshot_graphs: 离散动态图快照列表
        split_edge: 训练/验证/测试的边集合
        device: 设备

    返回:
        results: 包含性能指标的字典
    """
    results = test(model, snapshot_graphs, split_edge, device)
    print(f"新链接预测任务结果: AUC = {results['test_auc']:.4f}, AP = {results['test_ap']:.4f}")
    return results
