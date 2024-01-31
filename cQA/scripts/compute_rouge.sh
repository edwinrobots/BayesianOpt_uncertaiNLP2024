#!/bin/bash
#SBATCH -J bdl
#SBATCH -o ./log/compute_rouge.output
#SBATCH -e ./log/compute_rouge.err
#SBATCH --cpus-per-task=10
#SBATCH --mem=5G



echo Running on host "$(hostname)"
echo Time is "$(date)"
echo Directory is "$(pwd)"
echo PBS job ID is "${SLURM_JOBID}"


source /ukp-storage-1/fang/miniconda3/bin/activate /ukp-storage-1/fang/miniconda3
# conda create -n env python=3.7 
# source activate env 
# pip install pandas numpy torch==1.6.0 tqdm transformers --user
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sw/lang/anaconda.3.7-2019.03/lib/
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sw/lang/anaconda.3.7-2019.03-pytorch.1.2.0/lib/
# module add lang/python/anaconda/3.7.3-2019.03-tensorflow-2.0
# module add lang/python/anaconda/3.7-2019.10

time python3 main.py --start_q 250 --num_q 10 --epochs 4 --sample_nums 20 --topic apple --n_iter_rounds 4 --batch_size 1 --save_dir /work/zu20361/BDL/model/ --model_name all-data-cooking --interactive True
# time python3 main.py --start_q 750 --num_q 41 --epochs 4 --sample_nums 20 --topic cooking --n_iter_rounds 4 --batch_size 1 --save_dir /work/zu20361/BDL/model/ --model_name all-data-cooking --interactive True
# time python3 main.py --epochs 2 --topic cooking --n_iter_rounds 8 --batch_size 10 --do_test True --model_name 8rounds-1 --interactive True --pretrained_model roberta-base