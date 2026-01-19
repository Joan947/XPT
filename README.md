# XPT: Enhanced Point Transformer
 
<img width="960" height="540" alt="xpt" src="https://github.com/user-attachments/assets/c29fe949-faf3-4946-b4cf-5fc9defdf5d6" />


**XPT** (Enhanced Point Transformer) is an efficient and stable transformer architecture for 3D point cloud understanding. Built upon Point Transformer v3 (PTv3), XPT introduces three key architectural improvements that reduce computational cost and training time while maintaining or improving model accuracy.

## Key Features

- **Normalization-Free Training**: Dynamic Tanh (DyT) replaces LayerNorm for faster, more memory-efficient training 
- **Enhanced Stability**: QKV normalization improves training robustness without standard pre-normalization layers 
- **Efficient Positional Encoding**: 3D Rotary Positional Embedding (RoPE) encodes spatial relationships with minimal overhead 
- **Superior Performance**: Achieves up to 60% reduction in latency while improving accuracy on benchmark datasets 

XPT addresses computational inefficiencies in point cloud transformers by introducing three architectural modifications: Dynamic Tanh for normalization-free training, QKV normalization for training stability, and 3D RoPE for efficient positional encoding. These improvements collectively reduce latency by up to 60% while achieving superior accuracy on standard benchmarks. 

## Installation

XPT is built on Point Transformer v3. Follow the PTv3 installation instructions:

**PTv3 Repository**: [https://github.com/Pointcept](https://github.com/Pointcept/Pointcept)

```bash
# Clone the repository
git clone https://github.com/Joan947/XPT.git
cd XPT

# Install dependencies (follow PTv3 requirements)
```

## Datasets

Download the required datasets for evaluation following PTV3 and follow its commands for training and inference:

- **ScanNet v2**
- **S3DIS**
- **ScanNet++**


## Performance

### Supervised Learning Results

| Method | Dataset | allAcc | mAcc | mIoU |
|--------|---------|--------|------|------|
| PTv3 | ScanNet v2 | 91.56% | 83.87% | 76.60% |
| **XPT** | ScanNet v2 | **91.89%** | **84.47%** | **76.83%** |
| PTv3 | S3DIS Area 5 | 90.78% | 75.89% | 70.51% |
| **XPT** | S3DIS Area 5 | **91.12%** | **77.92%** | **71.89%** |


### Efficiency Comparison (ScanNet v2)

| Model | Params | Training Latency | Inference Latency |
|-------|--------|------------------|-------------------|
| PTv3 | 46M | 1204ms | 148ms |
| XPT (DyT) | 46M | 446ms | 132ms |
| XPT (QKVn) | 46M | 570ms | 73ms |
| **XPT (Full)** | 58M | **269ms** | **58ms** |

XPT achieves 60% reduction in inference latency compared to PTv3. 

### Self-Supervised Learning with SONATA

| Method | Dataset | allAcc | mAcc | mIoU |
|--------|---------|--------|------|------|
| PTv3 | ScanNet v2 | 91.72% | 83.24% | 75.77% |
| DyT | ScanNet v2 | 90.88% | 82.70% | 75.25% |
| PTv3 | S3DIS Area 5 | 91.38% | 78.97% | 72.40% |
| **DyT** | S3DIS Area 5 | **92.19%** | 78.92% | **73.32%** |

## Model Variants

XPT offers multiple configurations:

- **XPT (DyT)**: PTv3 with Dynamic Tanh replacing LayerNorm
- **XPT (QKVn)**: PTv3 with DyT applied to QKV projections
- **XPT (Full)**: Complete architecture with DyT, QKV normalization, and 3D RoPE


## Hardware Requirements

- **Training**: 8× NVIDIA H100 GPUs (or equivalent)
- **Inference**: 1× NVIDIA H100 GPU (or equivalent)
- **Memory**: 16GB+ GPU memory per device


## Acknowledgments

This work builds upon:
- [Point Transformer v3](https://github.com/Pointcept/Pointcept) for the base architecture 
- [SONATA](https://arxiv.org/abs/2503.16429) for self-supervised learning framework 
- [RoFormer](https://arxiv.org/abs/2104.09864) for rotary positional embeddings 
