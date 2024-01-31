#!/bin/bash
#SBATCH -J travel-swag-final
#SBATCH -o ./log/parallel/travel/interaction_res/bert_swag/4rounds-5epochs-mp.output
#SBATCH -e ./log/parallel/travel/interaction_res/bert_swag/4rounds-5epochs-mp.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu 3
#SBATCH --array=0-7
#SBATCH --mem 120G


echo Running on host "$(hostname)"
echo Time is "$(date)"
echo Directory is "$(pwd)"
echo JOB ID: "${SLURM_JOBID}"
echo SBATCH ARRAY ID: ${SLURM_ARRAY_TASK_ID}
num_q=100
start_q=`expr $SLURM_ARRAY_TASK_ID \* $num_q`
echo start question id is $start_q


# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sw/lang/anaconda.3.7-2019.03-pytorch.1.2.0/lib/

fake=`time python3 main.py --start_q $start_q --num_q $num_q --epochs 4 --sample_nums 20 --wd 0.001 --lr_init 5e-5 --topic travel --n_iter_rounds 4 --batch_size 1 --save_dir ./model/ --model_name swag_bert_6epoch_1e-4-16-0.01 --interactive True`
echo "$fake" > ./log/parallel/travel/interaction_res/swag/start-${start_q}.output
