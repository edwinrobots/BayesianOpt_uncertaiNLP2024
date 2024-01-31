#!/bin/bash
#SBATCH -J bdl-swag-travel
#SBATCH -o ./log/warm-start/travel/final/swag/swag_bert_6epoch_1e-4-16-0.01.output
#SBATCH -e ./log/warm-start/travel/final/swag/swag_bert_6epoch_1e-4-16-0.01.err
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --cpus-per-gpu 3


echo Running on host "$(hostname)"
echo Time is "$(date)"
echo Directory is "$(pwd)"
echo SBATCH job ID is "${SBATCH_JOBID}"


# module add lang/python/anaconda/pytorch


# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sw/lang/anaconda.3.7-2019.03-pytorch.1.2.0/lib/


# time python3 main.py --epochs 3 --topic cooking --batch_size 32 --do_train True --model_name swag_bert_3epoch --lr_init 1e-4 --swag_lr 5e-5 --swag_start 2 --do_test True
time python3 main.py --topic travel --do_train True --save_dir ./model/travel/hyperparam --model_name swag_bert_6epoch_1e-4-16-0.01 --lr_init 5e-5 --swag_lr 5e-5 --swag_start 3 --stop_epochs 3 --epochs 6 --batch_size 16 --wd 0.001



