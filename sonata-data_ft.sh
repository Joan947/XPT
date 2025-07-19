#!/bin/bash

#SBATCH --account=advdls25
#SBATCH --job-name=sonata-data_ft
#SBATCH --partition=mb-h100
#SBATCH --nodes=1
#SBATCH --tasks-per-node=32  # CPUs
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:h100:8 # GPUs
#SBATCH --mem=128G
#SBATCH --time=2-15:00:00 # 3.15 days
#SBATCH --output=dec-lin_ft_%j.out
#SBATCH --error=dec-lin_ft_%j.err



echo "CUDA_VISIBLE_DEVICES:" $CUDA_VISIBLE_DEVICES
nvidia-smi -L

(
  while true; do
    echo "=== GPU Usage ===" >> gpu_usage-dt.log
    nvidia-smi >> gpu_usage-dt.log
    echo "=== CPU Usage ===" >> cpu_usage-dt.log
    top -b -n 1 | head -20 >> cpu_usage-dt.log
    sleep 2500
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
# linear probing
echo "ScanNet linear probing starting.."
sh scripts/train.sh -m 1 -g 8 -d sonata -c semseg-sonata-v1m1-0a-scannet-lin -n semseg-sonata-v1m1-0-base-0a-scannet-lin-1 -w exp/sonata/pretrain-sonata-v1m1-0-base/model/pretrain-sonata-v1m1-0-base.pth 
echo "ScanNet linear probing done"
# decoder probing
echo "ScanNet decoder probing starting.."
sh scripts/train.sh -m 1 -g 8 -d sonata -c semseg-sonata-v1m1-0b-scannet-dec -n semseg-sonata-v1m1-0-base-0b-scannet-dec-1 -w exp/sonata/pretrain-sonata-v1m1-0-base/model/pretrain-sonata-v1m1-0-base.pth 
echo "ScanNet decoder probing done"


# ScanNetpp
#echo "ScanNetpp linear probing starting.."
#sh scripts/train.sh -m 1 -g 8 -d sonata -c semseg-sonata-v1m1-2a-scannetpp-lin -n semseg-sonata-v1m1-0-base-2a-scannetpp-lin-1 -w exp/sonata/pretrain-sonata-v1m1-0-base/model/pretrain-sonata-v1m1-0-base.pth 
#echo "ScanNetpp linear probing done"
#echo "ScanNetpp decoder probing starting.."
#sh scripts/train.sh -m 1 -g 8 -d sonata -c semseg-sonata-v1m1-2b-scannetpp-dec -n semseg-sonata-v1m1-0-base-2b-scannetpp-dec-1 -w exp/sonata/pretrain-sonata-v1m1-0-base/model/pretrain-sonata-v1m1-0-base.pth 
#echo "ScanNetpp decoder probing done"


# S3DIS Area 5
echo "S3DIS Area 5 linear probing starting.."
sh scripts/train.sh -m 1 -g 8 -d sonata -c semseg-sonata-v1m1-3a-s3dis-lin -n semseg-sonata-v1m1-0-base-3a-s3dis-lin-1 -w exp/sonata/pretrain-sonata-v1m1-0-base/model/pretrain-sonata-v1m1-0-base.pth 
echo "S3DIS Area 5 linear probing done"
echo "S3DIS Area 5 decoder probing starting.."
sh scripts/train.sh -m 1 -g 8 -d sonata -c semseg-sonata-v1m1-3b-s3dis-dec -n semseg-sonata-v1m1-0-base-3b-s3dis-dec-1 -w exp/sonata/pretrain-sonata-v1m1-0-base/model/pretrain-sonata-v1m1-0-baset.pth 
echo "S3DIS Area 5 decoder probing done"





echo "All experiments completed."
