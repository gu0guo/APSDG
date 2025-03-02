    config文件夹里是模型的配置信息
    dataset是数据集
    script是训练测试的自动化脚本
    e2e_main.py作为项目的主入口文件，负责解析命令行参数，调用 process.py 中的函数，以及管理训练和测试流程。
    preprocess.py是原始数据集的预处理
    process.py包含模型训练和测试的核心逻辑，如 e2e_train 和 e2e_test 函数。
    utils.py是代码里常用的工具函数
