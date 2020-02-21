#!/bin/sh


cd cluster_experiment_scripts

sbatch ae_mixmatch_cifar10_0.sh
sbatch ae_mixmatch_cifar10_1.sh
sbatch ae_mixmatch_cifar10_2.sh

sbatch ae_mixmatch_180_cifar10_0.sh
sbatch ae_mixmatch_180_cifar10_1.sh
sbatch ae_mixmatch_180_cifar10_2.sh

sbatch ae_mixmatch_320_cifar10_0.sh
sbatch ae_mixmatch_320_cifar10_1.sh
sbatch ae_mixmatch_320_cifar10_2.sh


