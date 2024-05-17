#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   vaes.py
@Time    :   2024/05/17 10:27:35
@Author  :   Paul_Bao
@Version :   1.0
@Contact :   paulbao@mail.ecust.edu.cn
"""

# here put the import lib
import torch
from models.rvae import rVAE


def test_rVAE():
    model = rVAE(
        in_dim=(28, 28),
        latent_dim=2,
        hidden_dims=[512, 256],
        softplus_out=1,
    )
    out = model(torch.rand(100, 1, 28, 28))
