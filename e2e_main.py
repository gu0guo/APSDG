# 定义命令行参数解析器
from model import load_model

parser = argparse.ArgumentParser(description='APSDG')
parser.add_argument('--dataset', type=str)  # 数据集名称
parser.add_argument('--device', type=int)  # 计算设备，如GPU或CPU
parser.add_argument('--model', type=str)  # 模型名称
parser.add_argument('--batch_size', type=int)  # 批处理大小，每个批次的样本数量
parser.add_argument('lr', type=float)  # 模型的学习率
parser.add_argument('--epochs', type=int)  # 训练周期数，模型将在数据上迭代的次数

args = parser.parse_args()  # 解析命令行参数
print(args)  # 打印解析后的参数
# 设置计算设备
device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'  # 检查是否有可用的GPU
device = torch.device(device)  # 将设备设置为GPU或CPU
# 加载快照数据
snapshots = load_data(args)
# 加载模型
model = load_model(snapshots, args.model, args.dataset, device)
# 训练模型

for epoch in range(1, 1 + args.epochs):  # 遍历训练周期
    # 创建数据加载器
    # 根据模型类型选择训练函数
    if args.model == 'APSDG':
        e2e_apsdg_train(snapshots, args.epochs, args, device)
    else:
        e2e_train(snapshots, args.epochs, args, device)
    # 定期评估模型
            # 打印评估结果
            # 保存性能最佳的模型
            # 早停策略
# 加载最佳模型

# 最终测试
results = e2e_test()
# 打印测试结果