#!/bin/bash
#SBATCH -J bdl-cooking
#SBATCH -o ./log/warm-start/cooking/final/5e-5-16-0.01-3epochs.output
#SBATCH -e ./log/warm-start/cooking/final/5e-5-16-0.01-3epochs.err
#SBATCH --gres=gpu:4
#SBATCH --mem=30G
#SBATCH --cpus-per-gpu 3

echo Running on host "$(hostname)"
echo Time is "$(date)"
echo Directory is "$(pwd)"
echo PBS job ID is "${SLURM_JOBID}"



source /ukp-storage-1/fang/miniconda3/bin/activate /ukp-storage-1/fang/miniconda3


time python3 main.py --epochs 3 \
 --topic cooking \
 --batch_size 16 \
 --do_train True \
 --save_dir ./model/cooking/ \
 --model_name vanilla_bert_3epoch_5e-5-16-0.01 \
 --lr_init 5e-5 \
 --stop_epochs 2 \
 --wd 0.01 \