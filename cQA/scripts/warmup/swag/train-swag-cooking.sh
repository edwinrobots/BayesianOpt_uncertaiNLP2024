#!/bin/bash
#SBATCH -J bdl-swag-cooking
#SBATCH -o ./log/warm-start/cooking/final/swag/swag_bert_6epoch_1e-4-32-0.001.output
#SBATCH -e ./log/warm-start/cooking/final/swag/swag_bert_6epoch_1e-4-32-0.001.err
#SBATCH --gres=gpu:2
#SBATCH --mem=30G
#SBATCH --cpus-per-gpu 3

cd "${SBATCH_O_WORKDIR}"

echo Running on host "$(hostname)"
echo Time is "$(date)"
echo Directory is "$(pwd)"
echo SBATCH job ID is "${SBATCH_JOBID}"





time python3 main.py --topic cooking --do_train True --save_dir ./model/cooking/hyperparam --model_name swag_bert_6epoch_1e-4-32-0.001 --lr_init 1e-4 --swag_lr 5e-5 --swag_start 3 --stop_epochs 3 --epochs 6 --batch_size 32 --wd 0.001