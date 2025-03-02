#训练APSDG模型的函数
#3.3费米狄拉克解码器计算链接预测概率
#3.4实现积空间的自适应调整
    #损失函数
    #3.4.2基于强化学习的优化
    #3.4.3可学习曲率
def e2e_apsdg_train(snapshots,EPOCH,args,device):
    snapshots = snapshots.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # 定义优化器
    #遍历数据加载器



#训练基准模型的函数
def e2e_train(snapshots,EPOCH,args,device):
    snapshots = snapshots.to(device)
    model.train()
    #遍历数据加载器







#测试模型的函数
def e2e_test(snapshots,EPOCH,args,device):
    model.eval()
    # 获取嵌入特征

    #初始化测试结果
    results = {}
    # 获取负样本边
    # 计算验证集的命中率
    # 计算测试集的命中率
    # 计算整体验证集和测试集的命中率