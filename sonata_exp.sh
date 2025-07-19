#!/bin/bash

#SBATCH --account=advdls25
#SBATCH --job-name=sonata_exps
#SBATCH --partition=mb-h100
#SBATCH --nodes=1
#SBATCH --tasks-per-node=32  # CPUs
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:h100:8 # GPUs
#SBATCH --mem=128G
#SBATCH --time=1-10:00:00 # 7 days
#SBATCH --output=sonata_rope_%j.out
#SBATCH --error=sonata_rope_%j.err



echo "CUDA_VISIBLE_DEVICES:" $CUDA_VISIBLE_DEVICES
nvidia-smi -L

# Load modules
module load miniconda3/24.3.0
module load arcc/1.0 gcc/13.2.0 cuda-toolkit/12.4.1


conda activate /cluster/medbow/project/advdls25/jowusu1/pt_transformers/points
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


# Define variables for the interpreter path and number of GPUs
INTERPRETER_PATH=$(which python)
NUM_GPU=8


echo " pretraining on SONATA for rope..."
sh scripts/train.sh -m 1 -g 8 -d sonata -c pretrain-sonata-v1m1-0-base -n pretrain-sonata-v1m1-0-base-rope 



echo "All experiments completed."

