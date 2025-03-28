#!/bin/bash

# GRUGCN模型运行脚本

# 设置数据集
DATASETS=("enron" "dblp" "hepph" "as733" "fbw")

# 链接预测任务
for DATASET in "${DATASETS[@]}"
do
    echo "运行 GRUGCN 在 $DATASET 上的链接预测任务"

    python e2e_main.py \
        --dataset $DATASET \
        --model grugcn \
        --task link_prediction \
        --device 0 \
        --epochs 100 \
        --batch_size 1024 \
        --lr 0.01 \
        --weight_decay 0.0001 \
        --clip_norm 1.0 \
        --patience 10 \
        --num_snapshots 30
done

# 新链接预测任务
for DATASET in "${DATASETS[@]}"
do
    echo "运行 GRUGCN 在 $DATASET 上的新链接预测任务"

    python e2e_main.py \
        --dataset $DATASET \
        --model grugcn \
        --task new_link_prediction \
        --device 0 \
        --epochs 100 \
        --batch_size 1024 \
        --lr 0.01 \
        --weight_decay 0.0001 \
        --clip_norm 1.0 \
        --patience 10 \
        --num_snapshots 30
done
