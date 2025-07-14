#!/bin/bash -l
#SBATCH --job-name=train_wo_sentinel1_pretraining
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

tar -xvf $WORK/data_raw_with_val_idx.tar  -C /scratch/dataCaFFeBig

echo "Started main script"

srun python train_Tyrion.py 1