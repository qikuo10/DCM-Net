#!/usr/bin/env bash
# 五折交叉验证 - DCM-Net with Swin-Base (1024-dim)

GPU=0
SEED=20000912
DIM=1024
DATA_DIR="/path/to/cholect45"  # TODO: change to your dataset path

echo "=========================================="
echo "DCM-Net 5-Fold Training (Swin-Base)"
echo "=========================================="
echo "GPU: ${GPU}"
echo "Seed: ${SEED}"
echo "Input Dim: ${DIM}"
echo "Data Dir: ${DATA_DIR}"
echo "=========================================="

cd DCM_network

for K in 0 1 2 3 4
do
    VERSION='swinB_1024_DCM_k'${K}'_seed'${SEED}
    
    echo ""
    echo "=========================================="
    echo "Fold ${K} / 5"
    echo "Version: ${VERSION}"
    echo "=========================================="
    
    python run.py -t -e \
        --seed ${SEED} \
        --input_dim ${DIM} \
        --data_dir ${DATA_DIR} \
        --mask \
        --pos_w 'pre_w' \
        --loss_type all \
        --fpn \
        --dataset_variant=cholect45-crossval \
        --kfold=${K} \
        --epochs=1000 \
        --batch=31 \
        -l 1e-2 5e-3 1e-2 \
        -w 9 18 200 \
        --version=${VERSION} \
        --version1=swinB_features \
        --gpu ${GPU} \
        --val_interval 20
    
    if [ $? -eq 0 ]; then
        echo "Fold ${K} completed successfully"
    else
        echo "Fold ${K} failed"
        exit 1
    fi
done

echo ""
echo "=========================================="
echo "All 5 folds completed!"
echo "=========================================="
