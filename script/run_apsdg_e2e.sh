#!/bin/bash

# APSDG模型运行脚本

# 设置数据集
#DATASETS=("enron" "dblp" "hepph" "as733" "fbw")
DATASETS=( "hepph" )
# 链接预测任务
for DATASET in "${DATASETS[@]}"
do
    echo "运行 APSDG 在 $DATASET 上的链接预测任务"

    python e2e_main.py \
        --dataset $DATASET \
        --model apsdg \
        --task link_prediction \
        --device 0 \
        --epochs 100 \
        --batch_size 512 \
        --lr 0.001 \
        --weight_decay 0.0005 \
        --clip_norm 1.0 \
        --patience 50 \
        --num_snapshots 72
done

# 新链接预测任务
for DATASET in "${DATASETS[@]}"
do
    echo "运行 APSDG 在 $DATASET 上的新链接预测任务"

    python e2e_main.py \
        --dataset $DATASET \
        --model apsdg \
        --task new_link_prediction \
        --device 0 \
        --epochs 100 \
        --batch_size 512 \
        --lr 0.001 \
        --weight_decay 0.0005 \
        --clip_norm 1.0 \
        --patience 50 \
        --num_snapshots 72
done
