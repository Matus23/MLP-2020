#!/bin/sh

cd ..
export DATASET_DIR="data/"
# Activate the relevant virtual environment:

python train_evaluate_emnist_classification_system.py --batch_size 64 --continue_from_epoch -1 --seed 0 \
                                                      --num_epochs 1 --experiment_name 'cifar10_mixmatch_test_exp' \
                                                      --use_gpu "True" --val-iteration 200 --dataset_name "cifar10"

