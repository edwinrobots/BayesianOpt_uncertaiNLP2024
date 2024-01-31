#!/bin/bash
#SBATCH -J cooking-bdl
#SBATCH -o ./log/parallel/cooking/interaction_res/bert_dropout/%x.output
#SBATCH -e ./log/parallel/cooking/interaction_res/bert_dropout/%x.err
#SBATCH --array=1-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu 3
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

source /storage/ukp/work/fang/miniconda3/bin/activate /storage/ukp/work/fang/miniconda3/envs/bdl
module load cuda/11.1
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sw/lang/anaconda.3.7-2019.03-pytorch.1.2.0/lib/



fake=`time python3 main.py --start_q $start_q --num_q $num_q --epochs 4 --sample_nums 20 --wd 0.01 --ilr 1e-4 --topic cooking --n_iter_rounds 3 --batch_size 1 --save_dir ./model/ --model_name vanilla_bert_3epoch_5e-5-16-0.01 --interactive True `
echo "$fake" > ./log/parallel/cooking/interaction_res/bert_dropout_3rounds/start-${start_q}.output
