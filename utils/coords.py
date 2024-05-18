#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   coords.py
@Time    :   2024/05/16 19:41:14
@Author  :   Paul_Bao
@Version :   1.0
@Contact :   paulbao@mail.ecust.edu.cn
"""

# here put the import lib
from typing import Tuple, Optional, Union, List, Dict
import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial, ndimage, optimize
from sklearn import cluster
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


def grid2xy(X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
    """
    Maps (M, N) grid to (M*N, 2) xy coordinates
    """
    X = torch.cat((X1[None], X2[None]), 0)
    d0, d1 = X.shape[0], X.shape[1] * X.shape[2]
    X = X.reshape(d0, d1).T
    return X


def imcoordgrid(im_dim: Tuple) -> torch.Tensor:
    """
    Returns a grid with pixel coordinates (used e.g. in rVAE)
    """
    xx = torch.linspace(-1, 1, im_dim[0])
    yy = torch.linspace(1, -1, im_dim[1])
    x0, x1 = torch.meshgrid(xx, yy, indexing="xy")
    return grid2xy(x0, x1)


def transform_coordinates(
    coord: Union[np.ndarray, torch.Tensor],
    phi: float,
    coord_dx: Union[np.ndarray, torch.Tensor, int] = 0,
) -> torch.Tensor:
    """
    Pytorch-based 2D rotation of coordinates followed by translation.
    Operates on batches.

    Args:
        coord (numpy array or torch tensor): batch with initial coordinates
        phi (float): rotational angle in rad
        coord_dx (numpy array or torch tensor): translation vector

    Returns:
        Transformed coordinates batch
    """

    if isinstance(coord, np.ndarray):
        coord = torch.from_numpy(coord).float()
    if isinstance(coord_dx, np.ndarray):
        coord_dx = torch.from_numpy(coord_dx).float()
    rotmat_r1 = torch.stack([torch.cos(phi), torch.sin(phi)], 1)
    rotmat_r2 = torch.stack([-torch.sin(phi), torch.cos(phi)], 1)
    rotmat = torch.stack([rotmat_r1, rotmat_r2], axis=1)
    coord = torch.bmm(coord, rotmat)

    return coord + coord_dx
