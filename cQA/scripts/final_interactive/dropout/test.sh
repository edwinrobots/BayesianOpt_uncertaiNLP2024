#!/bin/bash
#SBATCH -J test
#SBATCH -o ./log/parallel/apple/interaction_res/test/%x.output
#SBATCH -e ./log/parallel/apple/interaction_res/test/%x.err
#SBATCH --gres=gpu:1


echo Running on host "$(hostname)"
echo Time is "$(date)"
echo Directory is "$(pwd)"
echo SBATCH job ID is "${SLURM_JOBID}"
echo SBATCH ARRAY ID: ${SLURM_ARRAY_TASK_ID}
num_q=100
start_q=`expr ${SLURM_ARRAY_TASK_ID} \* ${num_q}`
echo start question id is $start_q

source /ukp-storage-1/fang/miniconda3/bin/activate
# 
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sw/lang/anaconda.3.7-2019.03-pytorch.1.2.0/lib/


fake=`time python3 main.py --start_q $start_q --num_q $num_q --epochs 4 --sample_nums 20 --wd 0.01 --ilr 1e-4 --topic apple --n_iter_rounds 4 --batch_size 1 --save_dir /ukp-storage-1/fang/BDL/model/ --model_name vanilla_bert_3epoch_2e-5-32-0.001 --interactive True `
echo "$fake" > ./log/parallel/apple/interaction_res/test/start-${start_q}.output