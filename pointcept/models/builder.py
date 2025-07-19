"""
Model Builder

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from pointcept.utils.registry import Registry

from pointcept.models.dynamic_tanh import DynamicTanh
import torch.nn as nn


MODELS = Registry("models")
MODULES = Registry("modules")


def build_model(cfg):
    """Build models."""
    return MODELS.build(cfg)
    

