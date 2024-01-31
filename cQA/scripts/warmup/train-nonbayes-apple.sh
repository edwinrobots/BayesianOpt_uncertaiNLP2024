#!/bin/bash
#SBATCH -J bdl-apple
#SBATCH -o ./log/warm-start/apple/final/2e-5-32-0.001-3epochs.output
#SBATCH -e ./log/warm-start/apple/final/2e-5-32-0.001-3epochs.err
#SBATCH --gres=gpu:4
#SBATCH --mem=30G


echo Running on host "$(hostname)"
echo Time is "$(date)"
echo Directory is "$(pwd)"
echo PBS job ID is "${SLURM_JOBID}"



# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sw/lang/anaconda.3.7-2019.03-pytorch.1.2.0/lib/



time python3 main.py \
--epochs 3 \
--topic apple \
--batch_size 32 \
--do_train True \
--save_dir ./model/apple/hyperparam \
--model_name ddd \
--lr_init 2e-5 \
--stop_epochs 2 \
--wd 0.001 \
--model_name vanilla_bert_3epoch_32-0.001 \

# time python3 main.py --epochs 3 --topic cooking --batch_size 16 --do_train True --save_dir /work/zu20361/BDL/model/cooking/hyperparam --model_name vanilla_bert_3epoch_16-0.001 --lr_init 5e-5 --stop_epochs 2 --wd 0.001
# vanilla_bert_3epoch_2e-5-32-0.001 \

