#!/bin/sh


cd cluster_experiment_scripts

sbatch mixmatch_mnist_0.sh
sbatch mixmatch_mnist_1.sh
sbatch mixmatch_mnist_2.sh

sbatch mixmatch_cifar10_0.sh
