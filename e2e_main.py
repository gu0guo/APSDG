import os
import argparse
import torch
import yaml
import time
import numpy as np
from copy import deepcopy
from models.model import load_model
from preprocess import load_data
from process import train, evaluate_link_prediction, evaluate_new_link_prediction
from utils import load_yaml, save_yaml, save_pickle


def main():
    """主函数，执行端到端训练和评估流程"""
    parser = argparse.ArgumentParser(description='自适应积空间离散动态图链接预测模型')

    # 数据集和模型参数
    parser.add_argument('--dataset', type=str, required=True, help='数据集名称')
    parser.add_argument('--model', type=str, required=True,
                        help='模型名称: apsdg, hat, hgcn, evolvegcn, grugcn, htgn, hgwavenet')
    parser.add_argument('--task', type=str, default='link_prediction',
                        help='任务: link_prediction, new_link_prediction')

    # 训练参数
    parser.add_argument('--device', type=int, default=0, help='GPU设备ID')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=1024, help='批量大小')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='权重衰减')
    parser.add_argument('--clip_norm', type=float, default=1.0, help='梯度裁剪范数')
    parser.add_argument('--patience', type=int, default=10, help='早停耐心值')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--window_size', type=float, default=1.0, help='时间窗口大小')

    # 解析参数
    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # 设置设备
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # 创建必要的目录
    os.makedirs('config', exist_ok=True)
    os.makedirs('trained_model', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # 加载模型配置
    config_file = f'config/{args.model}-config.yaml'
    if os.path.exists(config_file):
        config = load_yaml(config_file)
    else:
        # 如果配置文件不存在，创建默认配置
        config = {
            'embedding_dim': 64,
            'hidden_dim': 32,
            'history_window': 5,
            'num_layers': 2
        }
        # 为APSDG模型添加特定配置
        if args.model == 'apsdg':
            config.update({
                'e_ratio': 1 / 3,
                'b_ratio': 1 / 3,
                's_ratio': 1 / 3,
                'b_curvature': -1.0,
                's_curvature': 1.0,
                'rl_alpha': 0.01
            })
        # 保存配置
        save_yaml(config_file, config)

    # 加载数据
    print(f"加载数据集: {args.dataset}")
    snapshot_graphs, split_edge = load_data(args)
    print(f"数据集加载完成 - {len(snapshot_graphs)}个快照, {snapshot_graphs[0].num_nodes()}个节点")

    # 加载模型
    print(f"加载{args.model}模型")
    model = load_model(args.model, snapshot_graphs, config, device)
    print("模型加载完成")

    # 训练模型
    print("开始训练模型...")
    start_time = time.time()
    best_model = train(model, snapshot_graphs, split_edge, args, device)
    train_time = time.time() - start_time
    print(f"训练完成，耗时: {train_time:.2f}秒")

    # 保存模型
    model_save_path = f'trained_model/{args.model}_{args.dataset}_{args.task}.pt'
    torch.save(best_model.state_dict(), model_save_path)
    print(f"模型已保存至: {model_save_path}")

    # 评估模型
    print("评估模型性能...")
    if args.task == 'link_prediction':
        results = evaluate_link_prediction(best_model, snapshot_graphs, split_edge, device)
    else:  # new_link_prediction
        results = evaluate_new_link_prediction(best_model, snapshot_graphs, split_edge, device)

    # 保存结果
    results['train_time'] = train_time
    results_save_path = f'results/{args.model}_{args.dataset}_{args.task}_results.pkl'
    save_pickle(results_save_path, results)
    print(f"结果已保存至: {results_save_path}")


if __name__ == '__main__':
    main()
