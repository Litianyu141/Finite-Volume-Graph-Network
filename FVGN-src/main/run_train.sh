#!/bin/bash
BASE_TRAIN_DIR="/home/doomduke/GEP-FVGN/repos-py/FVM/my_FVNN"
LOG_FILE="/home/doomduke/GEP-FVGN/Logger/net GN-Cell; hs 128; mu 0.001; rho 1; dt 0.01;/Hybrid_loss_mom_normalize_grid_feature=10.txt"
exec > $BASE_TRAIN_DIR/train_handler.txt 2>&1

# training set
batch_size=16
lr=0.001
loss_cont=1
loss_mom=10
dataset_type="h5"
dataset_dir_h5="/mnt/h/Hybrid-dataset-Re200-1500/converted_dataset_1st/h5"
loss="direct_mean_loss"
nn_act="SiLU"
Noise_injection_factor=0.02

load_date=""
load_index=""

while true; do

    echo "Starting training..."

    # 构造基本命令
    CMD="python -u $BASE_TRAIN_DIR/train.py --batch_size=$batch_size --lr=$lr --dataset_type=$dataset_type --dataset_dir_h5=$dataset_dir_h5 --loss=$loss --nn_act=$nn_act --Noise_injection_factor=$Noise_injection_factor --loss_cont=$loss_cont --loss_mom=$loss_mom"
    
    # 根据load_date和load_index的值添加对应的参数
    if [ -n "$load_date" ]; then
        CMD="$CMD --load_date=$load_date"
    fi

    if [ -n "$load_index" ]; then
        CMD="$CMD --load_index=$load_index"
    fi

    # 运行CMD并将输出保存到 LOG_FILE
    $CMD > $LOG_FILE 2>&1

    echo "Starting detecting OOM error..."

    # 检查是否有 OOM 错误
    if grep -q "CUDA out of memory." $LOG_FILE; then
        echo "Detected OOM error, preparing to restart..."

        # 从日志中提取路径
        EXTRACTED_PATH=$(grep -o "saved tecplot file at .*" $LOG_FILE | tail -1 | sed -n 's|saved tecplot file at \(.*\)|\1|p')
        
        # 提取日期
        load_date=$(echo "$EXTRACTED_PATH" | awk -F'/' '{print $(NF-2)}')
        
        # 构建STATE_DIR路径
        STATE_DIR=$(echo "$EXTRACTED_PATH" | awk -F'/' 'BEGIN{OFS=FS} {$NF=""; $(NF-1)="states"; print $0}')
        
        # 检索最大的epoch文件
        MAX_EPOCH_FILE=$(ls -v "$STATE_DIR"*.state | tail -1)
        load_index=$(basename "$MAX_EPOCH_FILE" .state)

        echo "Loading from date: $load_date and epoch: $load_index"
        
        continue
    else
        echo "Not detected Training OOM error"
        exit 0
    fi

done
