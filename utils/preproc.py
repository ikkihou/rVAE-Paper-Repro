#!/usr/bin/env python3
"""
##
##       filename: preproc.py
##        created: 2024/05/16
##         author: Paul_Bao
##            IDE: Neovim
##       Version : 1.0
##       Contact : paulbao@mail.ecust.edu.cn
"""

# here put the import lib

"""
The code below are ported from https://github.com/pycroscopy/atomai
"""
from typing import Tuple, Optional, Union, List, Type
import numpy as np
import torch
import platform


def get_device():
    if platform.system() == "Darwin":
        # macOS 操作系统
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS device on macOS.")
        else:
            device = torch.device("cpu")
            print("MPS device not available on macOS. Falling back to CPU.")
    else:
        # 其他操作系统
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using CUDA device.")
        else:
            device = torch.device("cpu")
            print("CUDA device not available. Falling back to CPU.")
    return device


DEVICE = get_device()


def to_onehot(idx: torch.Tensor, n: int) -> torch.Tensor:
    """
    One-hot encoding of label
    """
    if torch.max(idx).item() >= n:
        raise AssertionError(
            "Labelling must start from 0 and "
            "maximum label value must be less than total number of classes"
        )
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    device_ = DEVICE
    onehot = torch.zeros(idx.size(0), n, device=device_)
    onehot.scatter_(1, idx, 1)
    return onehot
