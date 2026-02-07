#!/bin/bash

DSET=$1
EPOCHS=$2
DATA=data

if [ -z "$EPOCHS" ]; then
    EPOCHS=10
fi

if [ -z "$DSET" ]; then
    echo "usage: train_vit.sh <dataset_name> [epochs]"
    echo "example: train_vit.sh Emotion6 15"
    read -p "Press Enter to continue..."
    exit 1
fi

echo "========================================"
echo "dataset: $DSET"
echo "epochs: $EPOCHS"
echo "========================================"

python vit.py \
    --dataset "$DSET" \
    --epochs "$EPOCHS" \
    --batch_size 10 \
    --lr 3e-5 \
    --weight_decay 1e-5 \
    --device cuda 
read -p "Press Enter to continue..."
