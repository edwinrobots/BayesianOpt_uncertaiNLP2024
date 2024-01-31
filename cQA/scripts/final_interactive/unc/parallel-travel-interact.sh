#!/bin/bash
#SBATCH -J travel-bdl
#SBATCH -o ./log/parallel/travel/interaction_res/bert_unc/%x.output
#SBATCH -e ./log/parallel/travel/interaction_res/bert_unc/%x.err
#SBATCH --array=0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu 3
#SBATCH --mem 60G


echo Running on host "$(hostname)"
echo Time is "$(date)"
echo Directory is "$(pwd)"
echo JOB ID: "${SLURM_JOBID}"
echo SBATCH ARRAY ID: ${SLURM_ARRAY_TASK_ID}
num_q=100
start_q=`expr $SLURM_ARRAY_TASK_ID \* $num_q`
# start_q=0
echo start question id is $start_q

fake=`time python3 main.py --start_q $start_q --num_q $num_q --epochs 4 --querier_type unc --sample_nums 20 --wd 0.01 --ilr 1e-4 --topic travel --n_iter_rounds 4 --batch_size 1 --save_dir ./model/ --model_name vanilla_bert_3epoch_2e-5-16-0.001 --interactive True `
echo "$fake" > ./log/parallel/travel/4rounds-vanilla/interaction_res/bert_unc/start-${start_q}.output
# time python3 main.py --epochs 2 --topic cooking --n_iter_rounds 8 --batch_size 10 --do_test True --model_name 8rounds-1 --interactive True --pretrained_model roberta-base
