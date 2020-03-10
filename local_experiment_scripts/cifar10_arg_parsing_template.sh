#!/bin/sh

cd ..
export DATASET_DIR="data/"
# Activate the relevant virtual environment:

python train_evaluate_emnist_classification_system.py --batch_size 64 --continue_from_epoch -1 --seed 1 \
                                                      --num_epochs 3 --experiment_name 'cifar10_mixmatch_test_exp' \
                                                      --use_gpu "True" --val-iteration 100 --dataset_name "cifar10" \
                                                      --arc "vqvae" --beta 1.0 --lr_decay "no" --vqorder "before"

