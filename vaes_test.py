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


from typing import Dict, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn import cluster, mixture, decomposition

import cv2

import os
from pathlib import Path

from models.vanilla_vae import VanillaVAE
from models import fcVAE, rVAE
from experiment import VAEXperiment

import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from lightning_fabric.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


class imlocal:
    """
    STEM image local crystallography class
    highly ported from atomai https://github.com/pycroscopy/atomai/tree/master
    """

    def __init__(
        self,
        network_output: np.ndarray,
        coord_class_dict_all: Dict[int, np.ndarray],
        window_size: int = None,
        coord_class: int = 0,
    ) -> None:
        self.network_output = network_output
        self.nb_classes = network_output.shape[-1]
        self.coord_all = coord_class_dict_all
        self.coord_class = float(coord_class)
        self.r = window_size
        (self.imgstack, self.imgstack_com, self.imgstack_frames) = (
            self.extract_subimages_()
        )
        self.d0, self.d1, self.d2, self.d3 = self.imgstack.shape

    def get_imgstack(
        self,
        imgdata: np.ndarray,
        coord: np.ndarray,
        r: int,
    ) -> Tuple[np.ndarray]:

        img_cr_all = []
        com = []
        for c in coord:
            cx = int(np.around(c[0]))
            cy = int(np.around(c[1]))
            if r % 2 != 0:
                img_cr = np.copy(
                    imgdata[
                        cx - r // 2 : cx + r // 2 + 1, cy - r // 2 : cy + r // 2 + 1
                    ]
                )
            else:
                img_cr = np.copy(
                    imgdata[cx - r // 2 : cx + r // 2, cy - r // 2 : cy + r // 2]
                )
            if img_cr.shape[0:2] == (int(r), int(r)) and not np.isnan(img_cr).any():
                img_cr_all.append(img_cr[None, ...])
                com.append(c[None, ...])
        if len(img_cr_all) == 0:
            return None, None
        img_cr_all = np.concatenate(img_cr_all, axis=0)
        com = np.concatenate(com, axis=0)
        return img_cr_all, com

    def extract_subimages(
        self,
        imgdata: np.ndarray,
        coordinates: Union[Dict[int, np.ndarray], np.ndarray],
        window_size: int,
        coord_class: int = 0,
    ) -> Tuple[np.ndarray]:

        if isinstance(coordinates, np.ndarray):
            coordinates = np.concatenate(
                (coordinates, np.zeros((coordinates.shape[0], 1))), axis=-1
            )
            coordinates = {0: coordinates}
        if np.ndim(imgdata) == 2:
            imgdata = imgdata[None, ..., None]
        subimages_all, com_all, frames_all = [], [], []
        for i, (img, coord) in enumerate(zip(imgdata, coordinates.values())):
            coord_i = coord[np.where(coord[:, 2] == coord_class)][:, :2]
            stack_i, com_i = self.get_imgstack(img, coord_i, window_size)
            if stack_i is None:
                continue
            subimages_all.append(stack_i)
            com_all.append(com_i)
            frames_all.append(np.ones(len(com_i), int) * i)
        if len(subimages_all) > 0:
            subimages_all = np.concatenate(subimages_all, axis=0)
            com_all = np.concatenate(com_all, axis=0)
            frames_all = np.concatenate(frames_all, axis=0)
        return subimages_all, com_all, frames_all

    def extract_subimages_(self) -> Tuple[np.ndarray]:

        imgstack, imgstack_com, imgstack_frames = self.extract_subimages(
            self.network_output, self.coord_all, self.r, self.coord_class
        )
        return imgstack, imgstack_com, imgstack_frames


def test_rVAE():

    STEM_real = np.load("./assets/3DStack13-1-exp.npy")
    decoded_imgs = np.load("./assets/3DStack13-1-dec.npy")
    lattice_coord = np.load("./assets/3DStack13-1-coord.npy", allow_pickle=True)[()]

    s = imlocal(np.sum(decoded_imgs[..., :-1], -1)[..., None], lattice_coord, 24, 0)
    imgstack, imgstack_com, imgstack_frm = s.imgstack, s.imgstack_com, s.imgstack_frames

    imstack_train, imstack_test = train_test_split(
        imgstack,
        test_size=0.15,
        shuffle=True,
        random_state=42,
    )
    print(
        f"train stack shape: {imstack_train.shape},\ntest stack shape: {imstack_test.shape}"
    )

    train_dataset, val_dataset = TensorDataset(
        torch.from_numpy(imstack_train.transpose(0, 3, 1, 2))
    ), TensorDataset(
        torch.from_numpy(imstack_test.transpose(0, 3, 1, 2)),
    )  ## both are [24 24 1]

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=100,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=100,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    )

    RESULT_DIR = r"./"

    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(RESULT_DIR, "tb_logs"), name="rVAE"
    )

    seed_everything(1265, True)

    # model = VanillaVAE(in_channels=1, latent_dim=200, hidden_dims=[16, 32, 64], img_size = 24)
    model = rVAE(
        in_dim=(24, 24),
        latent_dim=2,
        hidden_dims=[512, 512],
        softplus_out=1,
    )

    exp_params = {
        "LR": 0.005,
        "weight_decay": 0.0,
        "scheduler_gamma": 0.95,
        "kld_weight": 0.00025,
    }

    experiment = VAEXperiment(model, exp_params)

    runner = Trainer(
        logger=tb_logger,
        callbacks=[
            LearningRateMonitor(),
            ModelCheckpoint(
                save_top_k=2,
                dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                monitor="val_loss",
                save_last=True,
            ),
        ],
        # strategy=DDPStrategy(accelerator="cpu"),
        max_epochs=5,
    )

    Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)

    runner.fit(experiment, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    test_rVAE()
