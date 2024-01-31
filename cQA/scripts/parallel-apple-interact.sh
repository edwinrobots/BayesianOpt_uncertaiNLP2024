#!/bin/bash
#SBATCH -J apple-bdl
#SBATCH -o ./log/parallel/apple/interaction_res/bert_dropout/%x.output
#SBATCH -e ./log/parallel/apple/interaction_res/bert_dropout/%x.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu 3
#SBATCH --array=0
#SBATCH --mem 120G


echo Running on host "$(hostname)"
echo Time is "$(date)"
echo Directory is "$(pwd)"
echo JOB ID: "${SLURM_JOBID}"
echo SBATCH ARRAY ID: ${SLURM_ARRAY_TASK_ID}
num_q=100
start_q=`expr $SLURM_ARRAY_TASK_ID \* $num_q`
# start_q=0
echo start question id is $start_q


fake=`time python3 main.py --start_q $start_q --num_q $num_q --epochs 4 --sample_nums 20 --wd 0.01 --ilr 1e-4 --topic apple --n_iter_rounds 3 --batch_size 1 --save_dir ./model/ --model_name vanilla_bert_3epoch_32-0.001 --interactive True `
echo "$fake" > ./log/parallel/apple/interaction_res/bert_dropout_3rounds/start-${start_q}.output
