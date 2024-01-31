#!/bin/bash
#SBATCH -J cooking-bdl
#SBATCH -o ./log/parallel/cooking/noise_exp/1/%x.output
#SBATCH -e ./log/parallel/cooking/noise_exp/1/%x.err
#SBATCH --array=0-7
#SBATCH --cpus-per-gpu 2 
#SBATCH --gres=gpu:1
#SBATCH --mem 50G

echo Running on host "$(hostname)"
echo Time is "$(date)"
echo Directory is "$(pwd)"
echo JOB ID: "${SLURM_JOBID}"
echo SBATCH ARRAY ID: ${SLURM_ARRAY_TASK_ID}
num_q=100
start_q=`expr $SLURM_ARRAY_TASK_ID \* $num_q`
echo start question id is $start_q

source /storage/ukp/work/fang/miniconda3/bin/activate /storage/ukp/work/fang/miniconda3
module load cuda/11.1


fake=`time python3 main.py --start_q $start_q --num_q $num_q --epochs 4 --sample_nums 20 --wd 0.01 --ilr 1e-4 --topic cooking --n_iter_rounds 4 --batch_size 1 --noise 1 --save_dir ./model/ --model_name vanilla_bert_3epoch_5e-5-16-0.01 --interactive True `
echo "$fake" > ./log/parallel/cooking/noise_exp/1/start-${start_q}.output
