#!/bin/bash

#SBATCH --account=advdls25
#SBATCH --job-name=sonata
#SBATCH --partition=mb-h100
#SBATCH --nodes=1
#SBATCH --tasks-per-node=32  # CPUs
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:h100:8 # GPUs
#SBATCH --mem=128G
#SBATCH --time=2-10:00:00 # 2.10 days
#SBATCH --output=ft_%j.out
#SBATCH --error=ft_%j.err



echo "CUDA_VISIBLE_DEVICES:" $CUDA_VISIBLE_DEVICES
nvidia-smi -L

(
  while true; do
    echo "=== GPU Usage ===" >> ft_gpu_usage.log
    nvidia-smi >> ft_gpu_usage.log
    echo "=== CPU Usage ===" >> ft_cpu_usage.log
    top -b -n 1 | head -20 >> ft_cpu_usage.log
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

INTERPRETER_PATH=$(which python)

# ScanNet 

# S3DIS Area 5
echo "S3DIS Area 5 full tuning test starting.."
#sh scripts/train.sh -m 1 -g 8 -d sonata -c semseg-sonata-v1m1-3c-s3dis-ft -n semseg-sonata-v1m1-0-base-3c-s3dis-ft-1-5 -w exp/sonata/pretrain-sonata-v1m1-0-base/model/pretrain-sonata-v1m1-0-base.pth -r true
sh scripts/test.sh -p ${INTERPRETER_PATH} -g 8 -d sonata -n semseg-sonata-v1m1-0-base-3c-s3dis-ft-1-5 -w model_best

echo "S3DIS Area 5 full tuning test done"


# S3DIS Area 3
echo "S3DIS Area 3 full tuning test starting.."
#sh scripts/train.sh -m 1 -g 8 -d sonata -c semseg-sonata-v1m1-3c-s3dis-ft-3 -n semseg-sonata-v1m1-0-base-3c-s3dis-ft-1-3 -w exp/sonata/pretrain-sonata-v1m1-0-base/model/pretrain-sonata-v1m1-0-base.pth -r true
sh scripts/test.sh -p ${INTERPRETER_PATH} -g 8 -d sonata -n semseg-sonata-v1m1-0-base-3c-s3dis-ft-1-3 -w model_best 
echo "S3DIS Area 3 full tuning test done"



# S3DIS Area 6
echo "S3DIS Area 6 full tuning test starting.."
#sh scripts/train.sh -m 1 -g 8 -d sonata -c semseg-sonata-v1m1-3c-s3dis-ft-6 -n semseg-sonata-v1m1-0-base-3c-s3dis-ft-1-6 -w exp/sonata/pretrain-sonata-v1m1-0-base/model/pretrain-sonata-v1m1-0-base.pth -r true
sh scripts/test.sh -p ${INTERPRETER_PATH} -g 8 -d sonata -n semseg-sonata-v1m1-0-base-3c-s3dis-ft-1-6 -w model_best 
echo "S3DIS Area 6 full tuning test done"

# S3DIS Area 2
echo "S3DIS Area 2 full tuning test starting.."
#sh scripts/train.sh -m 1 -g 8 -d sonata -c semseg-sonata-v1m1-3c-s3dis-ft-2 -n semseg-sonata-v1m1-0-base-3c-s3dis-ft-1-2 -w exp/sonata/pretrain-sonata-v1m1-0-base/model/pretrain-sonata-v1m1-0-base.pth -r true
sh scripts/test.sh -p ${INTERPRETER_PATH} -g 8 -d sonata -n semseg-sonata-v1m1-0-base-3c-s3dis-ft-1-2 -w model_best 
echo "S3DIS Area 2 full tuning test done"

#echo "ScanNet full tuning starting.."
#sh scripts/train.sh -m 1 -g 8 -d sonata -c semseg-sonata-v1m1-0c-scannet-ft -n semseg-sonata-v1m1-0-base-0c-scannet-ft-1 -w exp/sonata/semseg-sonata-v1m1-0-base-0c-scannet-ft/model/pretrain-sonata-v1m1-0-base.pth -r true

#echo "ScanNet full tuning done"

# ScanNetpp
#sh scripts/train.sh -m 1 -g 8 -d sonata -c semseg-sonata-v1m1-2c-scannetpp-ft -n semseg-sonata-v1m1-0-base-2c-scannetpp-ft -w exp/sonata/semseg-sonata-v1m1-0-base-2c-scannetpp-ft/model/pretrain-sonata-v1m1-0-base.pth -r true
sh scripts/test.sh -p ${INTERPRETER_PATH} -g 8 -d sonata -n semseg-sonata-v1m1-0-base-2c-scannetpp-ft -w model_best
echo "ScanNetpp full tuning test done"

# S3DIS Area 4
echo "S3DIS Area 4 full tuning test starting.."
#sh scripts/train.sh -m 1 -g 8 -d sonata -c semseg-sonata-v1m1-3c-s3dis-ft-4 -n semseg-sonata-v1m1-0-base-3c-s3dis-ft-1-4 -w exp/sonata/pretrain-sonata-v1m1-0-base/model/pretrain-sonata-v1m1-0-base.pth -r true
sh scripts/test.sh -p ${INTERPRETER_PATH} -g 8 -d sonata -n semseg-sonata-v1m1-0-base-3c-s3dis-ft-1-4 -w model_best
echo "S3DIS Area 4 full tuning test done"

# S3DIS Area 1
echo "S3DIS Area 1 full tuning test starting.."
#sh scripts/train.sh -m 1 -g 8 -d sonata -c semseg-sonata-v1m1-3c-s3dis-ft-1 -n semseg-sonata-v1m1-0-base-3c-s3dis-ft-1-1 -w exp/sonata/pretrain-sonata-v1m1-0-base/model/pretrain-sonata-v1m1-0-base.pth -r true
sh scripts/test.sh -p ${INTERPRETER_PATH} -g 8 -d sonata -n semseg-sonata-v1m1-0-base-3c-s3dis-ft-1-1 -w model_best
echo "S3DIS Area 1 full tuning test done"




# testing others

echo "ScanNet linear probing test starting.."
#sh scripts/train.sh -m 1 -g 8 -d sonata -c semseg-sonata-v1m1-0a-scannet-lin -n semseg-sonata-v1m1-0-base-0a-scannet-lin-1 -w exp/sonata/pretrain-sonata-v1m1-0-base/model/pretrain-sonata-v1m1-0-base.pth -r true
sh scripts/test.sh -p ${INTERPRETER_PATH} -g 8 -d sonata -n semseg-sonata-v1m1-0-base-0a-scannet-lin-1 -w model_best
echo "ScanNet linear probing test done"

echo "ScanNet decoder probing test starting.."
#sh scripts/train.sh -m 1 -g 8 -d sonata -c semseg-sonata-v1m1-0b-scannet-dec -n semseg-sonata-v1m1-0-base-0b-scannet-dec-1 -w exp/sonata/pretrain-sonata-v1m1-0-base/model/pretrain-sonata-v1m1-0-base.pth -r true
sh scripts/test.sh -p ${INTERPRETER_PATH} -g 8 -d sonata -n semseg-sonata-v1m1-0-base-0b-scannet-dec-1 -w model_best
echo "ScanNet decoder probing test done"

#echo "ScanNetpp linear probing test starting.."
#sh scripts/train.sh -m 1 -g 8 -d sonata -c semseg-sonata-v1m1-2a-scannetpp-lin -n semseg-sonata-v1m1-0-base-2a-scannetpp-lin-1 -w exp/sonata/pretrain-sonata-v1m1-0-base/model/pretrain-sonata-v1m1-0-base.pth  -r true
#sh scripts/test.sh -p ${INTERPRETER_PATH} -g 8 -d sonata -n semseg-sonata-v1m1-0-base-2a-scannetpp-lin-1 -w model_best
#echo "ScanNetpp linear probing test done"

#echo "ScanNetpp decoder probing test starting.."
#sh scripts/train.sh -m 1 -g 8 -d sonata -c semseg-sonata-v1m1-2b-scannetpp-dec -n semseg-sonata-v1m1-0-base-2b-scannetpp-dec-1 -w exp/sonata/pretrain-sonata-v1m1-0-base/model/pretrain-sonata-v1m1-0-base.pth -r true
#sh scripts/test.sh -p ${INTERPRETER_PATH} -g 8 -d sonata -n semseg-sonata-v1m1-0-base-2b-scannetpp-dec-1 -w model_best
#echo "ScanNetpp decoder probing test done"



echo "All experiments completed."
