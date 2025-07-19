#!/bin/bash

#SBATCH --account=advdls25
#SBATCH --job-name=sonata_ft
#SBATCH --partition=mb-h100
#SBATCH --nodes=1
#SBATCH --tasks-per-node=32  # CPUs
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:h100:8 # GPUs
#SBATCH --mem=128G
#SBATCH --time=15:00:00 # 3.15 days
#SBATCH --output=sonatascannet_ft_%j.out
#SBATCH --error=sonatascannet_ft_%j.err



echo "CUDA_VISIBLE_DEVICES:" $CUDA_VISIBLE_DEVICES
nvidia-smi -L

(
  while true; do
    echo "=== GPU Usage ===" >> gpu_usage.log
    nvidia-smi >> gpu_usage.log
    echo "=== CPU Usage ===" >> cpu_usage.log
    top -b -n 1 | head -20 >> cpu_usage.log
    sleep 3000
  done
) &

# Load modules
module load miniconda3/24.3.0
module load arcc/1.0 gcc/13.2.0 cuda-toolkit/12.4.1


conda activate /cluster/medbow/project/advdls25/jowusu1/pt_transformers/points
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# the pre-trained experiment is record in:
# exp/sonata/pretrain-sonata-v1m1-0-base


# ScanNet 

# S3DIS Area 5
# echo "S3DIS Area 5 full tuning resuming.."
# sh scripts/train.sh -m 1 -g 8 -d sonata -c semseg-sonata-v1m1-3c-s3dis-ft -n semseg-sonata-v1m1-0-base-3c-s3dis-ft-5 -w exp/sonata/pretrain-sonata-v1m1-0_qkv/model/model_last.pth 
# echo "S3DIS Area 5 full tuning done"


# # S3DIS Area 3
# echo "S3DIS Area 3 full tuning resuming.."
# sh scripts/train.sh -m 1 -g 8 -d sonata -c semseg-sonata-v1m1-3c-s3dis-ft-3 -n semseg-sonata-v1m1-0-base-3c-s3dis-ft-3 -w exp/sonata/pretrain-sonata-v1m1-0-base/model/model_last.pth -r true
# echo "S3DIS Area 3 full tuning done"



# # S3DIS Area 6
# echo "S3DIS Area 6 full tuning resuming.."
# sh scripts/train.sh -m 1 -g 8 -d sonata -c semseg-sonata-v1m1-3c-s3dis-ft-6 -n semseg-sonata-v1m1-0-base-3c-s3dis-ft-6 -w exp/sonata/pretrain-sonata-v1m1-0-base/model/model_last.pth -r true
# echo "S3DIS Area 6 full tuning done"

# # S3DIS Area 2
# echo "S3DIS Area 2 full tuning resuming.."
# sh scripts/train.sh -m 1 -g 8 -d sonata -c semseg-sonata-v1m1-3c-s3dis-ft-2 -n semseg-sonata-v1m1-0-base-3c-s3dis-ft-2 -w exp/sonata/pretrain-sonata-v1m1-0-base/model/model_last.pth -r true
# echo "S3DIS Area 2 full tuning done"

echo "ScanNet test full tuning starting.."
sh scripts/train.sh -m 1 -g 1 -d sonata -c semseg-sonata-v1m1-0c-scannet-ft -n semseg-sonata-v1m1-0-base-0c-scannet-ft -w exp/sonata/pretrain-sonata-v1m1-0_qkv/model/model_last.pth 
echo "ScanNet full tuning test done"

# # ScanNetpp
# echo "ScanNetpp test full tuning starting.."
# sh scripts/test.sh -m 1 -g 8 -d sonata -c semseg-sonata-v1m1-2c-scannetpp-ft -n semseg-sonata-v1m1-0-base-2c-scannetpp-ft -w exp/sonata/semseg-sonata-v1m1-0-base-2c-scannetpp-ft/model/model_last.pth
# echo "ScanNetpp full tuning test done"
# echo "All experiments completed."

# # S3DIS Area 4
# echo "S3DIS Area 4 test full tuning starting.."
# sh scripts/test.sh -m 1 -g 8 -d sonata -c semseg-sonata-v1m1-3c-s3dis-ft-4 -n semseg-sonata-v1m1-0-base-3c-s3dis-ft-4 -w exp/sonata/semseg-sonata-v1m1-0-base-3c-s3dis-ft-4/model/model_last.pth
# echo "S3DIS Area 4 full tuning test done"

# # S3DIS Area 1
# echo "S3DIS Area 1 test full tuning starting.."
# sh scripts/train.sh -m 1 -g 8 -d sonata -c semseg-sonata-v1m1-3c-s3dis-ft-1 -n semseg-sonata-v1m1-0-base-3c-s3dis-ft-1 -w exp/sonata/semseg-sonata-v1m1-0-base-3c-s3dis-ft-1/model/model_last.pth
# echo "S3DIS Area 1 full tuning test done"
