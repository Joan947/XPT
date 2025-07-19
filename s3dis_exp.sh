#!/bin/bash

#SBATCH --account=advdls25
#SBATCH --job-name=s3dis_exps
#SBATCH --partition=mb-l40s
#SBATCH --nodes=1
#SBATCH --tasks-per-node=32  # CPUs
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:l40s:8 # GPUs
#SBATCH --mem=128G
#SBATCH --time=1-08:00:00 # 8 hours
#SBATCH --output=ptrope_training_%j.out
#SBATCH --error=ptrope_training_%j.err



echo "CUDA_VISIBLE_DEVICES:" $CUDA_VISIBLE_DEVICES
nvidia-smi -L

# Load modules
module load miniconda3/24.3.0
module load arcc/1.0
module load gcc/13.2.0
module load cuda-toolkit/12.4.1

conda activate /cluster/medbow/project/advdls25/jowusu1/pt_transformers/points
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


# Define variables for the interpreter path and number of GPUs
INTERPRETER_PATH=$(which python)
NUM_GPU=8

sh scripts/train.sh -p  ${INTERPRETER_PATH} -g ${NUM_GPU} -d s3dis -c semseg-pt-v3m1-0-base -n semseg-pt-v3m1-rope 

# echo " testing..."
# sh scripts/test.sh -p ${INTERPRETER_PATH} -g ${NUM_GPU} -d s3dis -n semseg-pt-v3m1-rope  -w model_best   

echo "All experiments completed."