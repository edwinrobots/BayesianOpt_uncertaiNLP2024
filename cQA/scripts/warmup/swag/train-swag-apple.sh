#!/bin/bash
#SBATCH -J bdl-swag-apple
#SBATCH -o ./log/warm-start/apple/final/swag/swag_bert_6epoch_1e-4-16-0.001.output
#SBATCH -e ./log/warm-start/apple/final/swag/swag_bert_6epoch_1e-4-16-0.001.err
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --cpus-per-gpu 3

cd "${SBATCH_O_WORKDIR}"

echo Running on host "$(hostname)"
echo Time is "$(date)"
echo Directory is "$(pwd)"
echo SBATCH job ID is "${SBATCH_JOBID}"



module load cuda/11.1
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sw/lang/anaconda.3.7-2019.03-pytorch.1.2.0/lib/
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sw/lang/anaconda.3.7-2019.03/lib/
# module add lang/python/anaconda/3.7.3-2019.03-tensorflow-2.0
# module add lang/python/anaconda/3.7-2019.10

# time python3 main.py --epochs 3 --topic cooking --batch_size 32 --do_train True --model_name swag_bert_3epoch --lr_init 1e-4 --swag_lr 5e-5 --swag_start 2 --do_test True
time python3 main.py --topic apple --do_train True --save_dir ./model/apple/hyperparam --model_name swag_bert_6epoch_1e-4-16-0.001 --lr_init 1e-4 --swag_lr 5e-5 --swag_start 3 --stop_epochs 3 --epochs 6 --batch_size 16 --wd 0.001



