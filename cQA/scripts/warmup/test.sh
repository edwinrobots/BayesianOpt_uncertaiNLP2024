#!/bin/bash
#PBS -N bdl-apple
#PBS -o ./log/exact_match/apple/hyperparam/2e-5-32-0.001-3epochs.output
#PBS -e ./log/exact_match/apple/hyperparam/2e-5-32-0.001-3epochs.err
#PBS -l select=1:ncpus=1:ngpus=4:mem=30G
#PBS -l walltime=2:40:00
#所有16都完了， 32的batch一个没做
cd "${PBS_O_WORKDIR}"

echo Running on host "$(hostname)"
echo Time is "$(date)"
echo Directory is "$(pwd)"
echo PBS job ID is "${PBS_JOBID}"
echo This jobs runs on the following machines:
echo $(cat "${PBS_NODEFILE}" | uniq)


module add lang/python/anaconda/pytorch
# module add lang/python/anaconda/3.7-2019.03
# conda create -n env python=3.7 
# source activate env 
# pip install pandas numpy torch==1.4.0 tqdm transformers --user

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sw/lang/anaconda.3.7-2019.03-pytorch.1.2.0/lib/
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sw/lang/anaconda.3.7-2019.03/lib/
# module add lang/python/anaconda/3.7.3-2019.03-tensorflow-2.0
# module add lang/python/anaconda/3.7-2019.10

# time python3 main.py --epochs 3 --topic cooking --batch_size 32 --do_train True --model_name swag_bert_3epoch --lr_init 1e-4 --swag_lr 5e-5 --swag_start 2 --do_test True
time python3 main.py --epochs 3 --topic apple --batch_size 32 --do_train True --save_dir /work/zu20361/BDL/model/cooking/hyperparam --model_name vanilla_bert_3epoch_2e-5-32-0.001 --lr_init 2e-5 --stop_epochs 2 --wd 0.001
# time python3 main.py --epochs 3 --topic cooking --batch_size 16 --do_train True --save_dir /work/zu20361/BDL/model/cooking/hyperparam --model_name vanilla_bert_3epoch_16-0.001 --lr_init 5e-5 --stop_epochs 2 --wd 0.001


