#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   rvae.py
@Time    :   2024/05/16 01:13:19
@Author  :   Paul_Bao
@Version :   1.0
@Contact :   paulbao@mail.ecust.edu.cn
"""

# here put the import lib

import logging

from copy import deepcopy

import os
import math
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.nn import functional as F

from models import BaseVAE
from models.types_ import Any, List
from utils.coords import imcoordgrid, transform_coordinates
from .types_ import *

import platform


def get_device():
    if platform.system() == "Darwin":
        # macOS 操作系统
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            logging.debug("Using MPS device on macOS.")
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


def KLd_rot(phi_prior: torch.Tensor, phi_logsd: torch.Tensor) -> torch.Tensor:
    """
    Kullback–Leibler (KL) divergence for rotation latent variable
    """
    phi_sd = torch.exp(phi_logsd)
    kl_rot = -phi_logsd + np.log(phi_prior) + phi_sd**2 / (2 * phi_prior**2) - 0.5
    return kl_rot


class rEncoder(nn.Module):
    """
    Encoder network for rotationally-invariant VAE
    """

    def __init__(
        self,
        in_feature: int = None,
        hidden_dims: List[int] = None,
        latent_dim: int = None,
        *args,
        **kwargs,
    ) -> None:
        super(rEncoder, self).__init__()

        activation = kwargs.get("activation", "relu")
        ## encoder
        modules = []
        in_dim = in_feature
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_features=in_dim, out_features=h_dim),
                    nn.LeakyReLU(0.2) if activation == "relu" else nn.Tanh(),
                )
            )
            in_dim = h_dim

        self.encoder = nn.Sequential(*modules)

        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logstd = nn.Linear(hidden_dims[-1], latent_dim)
        self._out = nn.Softplus() if kwargs.get("softplus_out") else lambda x: x

    def forward(self, x: Tensor):
        x = x.reshape(x.size(0), -1)
        logging.debug(f"encoder input tensor shape: {x.shape}")
        result = self.encoder(x)
        mu, log_std = self.fc_mu(result), self._out(self.fc_logstd(result))
        return [mu, log_std]


class rDecoder(nn.Module):
    """
    Decoder network for rotationally-invariant VAE
    """

    def __init__(
        self,
        out_dim: Tuple[int] = None,
        latent_dim: int = None,
        hidden_dims: List[int] = None,
        skip: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super(rDecoder, self).__init__(
            *args,
            **kwargs,
        )

        if len(out_dim) == 2:
            c = 1
            self.reshape_ = (1, out_dim[0], out_dim[1])

        self.skip = skip

        self.coord_latent = coord_latent(latent_dim, hidden_dims[0], not skip)

        fc_decoder = []
        in_dim = hidden_dims[0]
        for h_dim in hidden_dims:
            fc_decoder.append(
                nn.Sequential(
                    nn.Linear(in_features=in_dim, out_features=h_dim),
                    # nn.LeakyReLU(0.2),
                    nn.Tanh(),
                )
            )
            in_dim = h_dim

        self.fc_decoder = nn.Sequential(*fc_decoder)
        self.out = nn.Sequential(
            nn.Linear(hidden_dims[-1], c),
            nn.Sigmoid(),
        )

    def forward(self, x_coord: Tensor, z: Tensor):
        """
        Forward pass
        """
        batch_dim = x_coord.size()[0]
        h = self.coord_latent(x_coord, z)
        if self.skip:
            residual = h
            for i, fc_block in enumerate(self.fc_decoder):
                h = fc_block(h)
                if (i + 1) % 2 == 0:
                    h = h.add(residual)
        else:
            h = self.fc_decoder(h)
        h = self.out(h)
        h = h.reshape(batch_dim, *self.reshape_)
        logging.debug(f"model output shape: {h.shape}")
        return h


class coord_latent(nn.Module):
    """
    The "spatial" part of the rVAE's decoder that allows for translational
    and rotational invariance (based on https://arxiv.org/abs/1909.11663)

    Args:
        latent_dim:
            number of latent dimensions associated with images content
        out_dim:
            number of output dimensions
            (usually equal to number of hidden units
             in the first layer of the corresponding VAE's decoder)
        activation:
            Applies tanh activation to the output (Default: False)
    """

    def __init__(self, latent_dim: int, out_dim: int, activation: bool = False) -> None:
        """
        Initiate parameters
        """
        super(coord_latent, self).__init__()
        self.fc_coord = nn.Linear(2, out_dim)
        self.fc_latent = nn.Linear(latent_dim, out_dim, bias=False)
        self.activation = nn.Tanh() if activation else None

    def forward(self, x_coord: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        x_coord [B, H*W, 2]
        z       [B, z_dim]

        """
        batch_dim, n = x_coord.size()[:2]
        x_coord = x_coord.reshape(batch_dim * n, -1)  ## [B*H*W, 2]
        logging.debug(f"x_coord shape: {x_coord.shape}")
        h_x = self.fc_coord(x_coord)
        h_x = h_x.reshape(batch_dim, n, -1)
        h_z = self.fc_latent(z)
        h = h_x.add(h_z.unsqueeze(1))
        h = h.reshape(batch_dim * n, -1)
        if self.activation is not None:
            h = self.activation(h)

        return h


class rVAE(BaseVAE):
    """rotationally-invariant VAE
    Variational Autoencoder (VAE) based on the idea of "spatial decoder"
    by Bepler et al. in arXiv:1909.11663. In addition, this class allows
    implementating the class-conditioned VAE and skip-VAE (arXiv:1807.04863)
    with rotational and translational variance.


    """

    def __init__(
        self,
        in_dim: Union[Tuple, int] = None,
        latent_dim: int = 2,
        nb_classes: int = 0,
        translation: bool = True,
        hidden_dims: List[int] = None,
        **kwargs: Union[int, bool, str],
    ) -> None:
        """Initializes rVAE's modules and parameters

        Args:
            in_dim (Union[Tuple, int], optional):
                Input dimensions for image data passed as tuple (H, W) or vectorized image data passed as int H*W. Defaults to None.

            latent_dim (int, optional):
                Number of VAE latent dimensions associated with image content. Defaults to 2.

            nb_classes (int, optional):
                Number of classes for class-conditioned rVAE

            translation (bool, optional):
                account for xy shifts of image content. Defaults to True.

            hidden_dims (list[int], optional):
            Sequence of hidden layer dimensions. Defaults to None.

            activation (str, optional):
                Activation function used in networks. Defaults to "tanh".
        """
        super(rVAE, self).__init__()
        coord = 3 if translation else 1  # xy translation and/or rotation
        self.translation = translation
        self.content_latent_dim = latent_dim
        self.dx_prior = kwargs.get("translation_prior", 0.1)
        self.phi_prior = kwargs.get("rotation_prior", 0.1)
        self.kdict = deepcopy(kwargs)
        self.in_dim = in_dim
        # self.kdict_["num_iter"] = 0
        self.x_coord = imcoordgrid(in_dim)

        if hidden_dims == None:
            hidden_dims = [1024, 512, 256]

        self.hidden_dims = deepcopy(hidden_dims)
        self.in_img_dim = (
            in_dim if isinstance(in_dim, int) else int(in_dim[0] * in_dim[1])
        )
        logging.debug(f"Encoder input : {self.in_img_dim}")

        self.z_dim = latent_dim + coord + nb_classes

        self.encoder = rEncoder(
            in_feature=self.in_img_dim,
            hidden_dims=self.hidden_dims,
            latent_dim=self.z_dim,
            **self.kdict,
        )
        ## make sure that the hidden_dims is reversed before pass into decoder
        hidden_dims.reverse()
        self.decoder = rDecoder(
            out_dim=in_dim,
            latent_dim=self.content_latent_dim,  ## Zx
            hidden_dims=hidden_dims,
        )

    def encode(self, input: torch.tensor) -> List[torch.tensor]:
        result = self.encoder(input)
        return result  ## [mu, log_std]

    def decode(self, x_coord: torch.tensor, z: Tensor) -> Any:
        result = self.decoder(x_coord, z)
        return result

    def reparameterize(self, mu: Tensor, logstd: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:

        x_coord_ = self.x_coord.expand(input.size(0), *self.x_coord.size()).to(DEVICE)
        logging.debug(f"x_coord_ shape: {x_coord_.shape}")

        mu, log_std = self.encode(input)
        z = self.reparameterize(mu, log_std)
        logging.debug(f"latent_code shape: {z.shape}")

        phi = z[:, 0]
        if self.translation:
            dx = z[:, 1:3]
            dx = (dx * torch.tensor(self.dx_prior).to(DEVICE)).unsqueeze(1)
            z = z[:, 3:]
        else:
            dx = 0
            z = z[:, 1:]

        x_coord_ = transform_coordinates(x_coord_, phi, dx)
        logging.debug(f"transformed x_coord_ shape: {x_coord_.shape}")
        logging.debug(f"latent image content z shape: {z.shape}")

        return [self.decode(x_coord_, z), input, mu, log_std]

    def loss_function(self, *inputs: torch.Any, **kwargs) -> torch.tensor:
        """Computes the rVAE loss function"""
        reconstr = inputs[0]
        input = inputs[1]
        mu = inputs[2]
        log_std = inputs[3]

        phi_log_std = log_std[:, 0]
        z_mu, z_log_std = mu[:, 1:], log_std[:, 1:]

        ## reconstruction loss
        reconstr_loss = F.mse_loss(reconstr, input)

        ## klds
        kld_rot = KLd_rot(
            phi_prior=self.phi_prior,
            phi_logsd=phi_log_std,
        ).mean()

        kld_z = torch.mean(
            -0.5 * torch.sum(1 + 2 * z_log_std - z_mu**2 - z_log_std.exp() ** 2, dim=1),
            dim=0,
        )

        loss = kld_rot + kld_z + reconstr_loss

        return {
            "loss": loss,
            "Reconstruction_Loss": reconstr_loss.detach(),
            "KLD_rotation": (kld_rot).detach(),
            "KLD_z": (kld_z).detach(),
        }

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """

        n_rows, n_cols = int(math.sqrt(num_samples)), int(math.sqrt(num_samples))

        grid_x = norm.ppf(np.linspace(0.95, 0.05, n_rows))
        grid_y = norm.ppf(np.linspace(0.05, 0.95, n_cols))

        z_sample = np.empty((num_samples, 2))
        index = 0
        for xi in grid_x:
            for yi in grid_y:
                z_sample[index] = np.array([xi, yi])
                index += 1

        # z = torch.randn(num_samples, self.content_latent_dim)
        # logging.debug(f"sample z shape: {z.shape}")

        z = torch.tensor(z_sample.astype(np.float32)).to(current_device)

        x_coord_ = self.x_coord.expand(num_samples, *self.x_coord.size()).to(DEVICE)

        samples = self.decode(x_coord_, z)
        # return samples.reshape(
        #     num_samples, 1, int(math.sqrt(self.in_dim)), int(math.sqrt(self.in_dim))
        # )
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:  ## x 是验证集中的一个batch
        """
        Given an input vectorized image x, returns the reconstructed image
        :param x: (Tensor) [B x (C x H x W)]
        :return: (Tensor) [B x (C x H x W)]
        """
        # print(x.shape)
        # B, C, H, W = x.shape
        # H, W = int(math.sqrt(V)), int(math.sqrt(V))

        recons = self.forward(x)[0]
        # recons = torch.reshape(recons, shape=(B, 1, H, W))
        return recons

    def manifold2d(self, **kwargs: Union[int, List, str, bool]) -> None:
        d = kwargs.get("d", 9)
        cmap = kwargs.get("cmap", "gnuplot")
        in_dim = self.in_dim
        if len(in_dim) == 2:
            figure = np.zeros((in_dim[0] * d, in_dim[1] * d))
        elif len(in_dim) == 3:
            figure = np.zeros((self.in_dim[0] * d, self.in_dim[1] * d, self.in_dim[-1]))

        grid_x = norm.ppf(np.linspace(0.95, 0.05, d))
        grid_y = norm.ppf(np.linspace(0.05, 0.95, d))

        for i, xi in enumerate(grid_x):
            for j, yi in enumerate(grid_y):
                z_sample = np.array([xi, yi])
                # print(f"z_sample shape: {z_sample.shape}")
                x_coord = self.x_coord.expand(1, *self.x_coord.size())
                # with torch.no_grad():
                imdec = self.decode(
                    x_coord,
                    torch.tensor(z_sample.astype(np.float32)).unsqueeze(0),
                )
                figure[
                    i * in_dim[0] : (i + 1) * in_dim[0],
                    j * in_dim[1] : (j + 1) * in_dim[1],
                ] = (
                    imdec.detach().cpu().numpy()[0, 0]
                )
        if figure.min() < 0:
            figure = (figure - figure.min()) / figure.ptp()

        ## plot
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(
            figure,
            cmap=cmap,
            origin=kwargs.get("origin", "lower"),
            extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()],
        )
        ax.set_xlabel("$z_1$")
        ax.set_ylabel("$z_2$")

        draw_grid = kwargs.get("draw_grid")
        if draw_grid:
            major_ticks_x = np.arange(0, d * in_dim[0], in_dim[0])
            major_ticks_y = np.arange(0, d * in_dim[1], in_dim[1])
            ax.set_xticks(major_ticks_x)
            ax.set_yticks(major_ticks_y)
            ax.grid(which="major", alpha=0.6)
        for item in (
            [ax.xaxis.label, ax.yaxis.label]
            + ax.get_xticklabels()
            + ax.get_yticklabels()
        ):
            item.set_fontsize(18)
        if not kwargs.get("savefig"):
            plt.show()
        else:
            savedir = kwargs.get("savedir", "./vae_learning/")
            fname = kwargs.get("filename", "manifold_2d")
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            fig.savefig(os.path.join(savedir, "{}.png".format(fname)))
            plt.close(fig)
        return figure
