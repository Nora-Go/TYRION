#!/bin/bash -l
#SBATCH --job-name=OptTranslator_pretraining
#SBATCH --gres=gpu:a100:4 --partition=a100
#SBATCH --mail-type=ALL
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=4

export NCCL_DEBUG=INFO
module load gcc/12.1.0
module load python/3.9-anaconda
module load cuda/12.1.1
#conda env create -n ldm --file environment.yaml
conda activate mocca2

echo "Copying and extracting tar to /scratch/data"
mkdir /scratch/dataCaFFeBig
mkdir /scratch/data_unlabeled
mkdir /scratch/data_optical

tar -xvf $WORK/data_raw.tar  -C /scratch/dataCaFFeBig
tar -xvf $WORK/unlabeled_data_raw_train.tar  -C /scratch/data_unlabeled
tar -xvf $WORK/unlabeled_data_raw_val.tar  -C /scratch/data_unlabeled
tar -xvf $WORK/data_optical.tar  -C /scratch/data_optical

echo "Started main script"

srun python pretraining_OptTranslator.py 1
