#!/bin/bash
#SBATCH -J bdl-travel
#SBATCH -o ./log/warm-start/travel/final/2e-5-16-0.001-3epochs.output
#SBATCH -e ./log/warm-start/travel/final/2e-5-16-0.001-3epochs.err
#SBATCH --gres=gpu:4
#SBATCH --mem=30G
#SBATCH --cpus-per-gpu 3


echo Running on host "$(hostname)"
echo Time is "$(date)"
echo Directory is "$(pwd)"
echo PBS job ID is "${SLURM_JOBID}"



source /ukp-storage-1/fang/miniconda3/bin/activate /ukp-storage-1/fang/miniconda3

time python3 main.py --epochs 3 \
--topic travel --batch_size 16 \
--do_train True \
--save_dir ./model/travel/ \
--model_name vanilla_bert_3epoch_2e-5-16-0.001 \
--lr_init 2e-5 \
--stop_epochs 2 \
--wd 0.001


