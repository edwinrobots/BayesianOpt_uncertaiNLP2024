#!/bin/bash
#PBS -N prop-0.9
#PBS -o ./log/exact_match/travel/proportion_exp/2e-5-16-0.001-3eopchs-0.9.output
#PBS -e ./log/exact_match/travel/proportion_exp/2e-5-16-0.001-3epochs-0.9.err
#PBS -l select=2:ncpus=1:ngpus=3:mem=20G:gputype=RTX2080Ti
#PBS -l walltime=3:55:00
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

time python3 main.py --epochs 3 --topic travel --batch_size 16 --do_train True --save_dir /work/zu20361/BDL/model/travel/proportion_exp --proportion 0.9 --model_name vanilla_bert_3epoch_2e-5-16-0.001-0.9 --lr_init 2e-5 --stop_epochs 2 --wd 0.001


