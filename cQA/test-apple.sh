#!/bin/bash
#PBS -N bdl-apple
#PBS -o ./log/parallel/apple/interaction_res/baseline.output
#PBS -e ./log/parallel/apple/interaction_res/baseline.err
#PBS -l select=1:ncpus=1:ngpus=3:mem=30G
#PBS -l walltime=3:20:00

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



time python3 main.py --topic apple --save_dir /work/zu20361/BDL/model/ --model_name vanilla_bert_3epoch_2e-5-32-0.001 --test_mode test --do_test True

