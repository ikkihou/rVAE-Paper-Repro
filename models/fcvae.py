#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fcvae.py
@Time    :   2024/05/14 19:57:45
@Author  :   Paul_Bao
@Version :   1.0
@Contact :   paulbao@mail.ecust.edu.cn
"""

# here put the import lib
import math
import copy
import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *


class fcVAE(BaseVAE):
    """Fully connected VAE that takes vectorized small-size image as input"""

    def __init__(
        self,
        in_dim: int,
        latent_dim: int = 2,
        hidden_dims: List = None,
        **kwargs,
    ) -> None:
        super(fcVAE, self).__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        self.in_dim = in_dim
        self.hidden_dims = copy.deepcopy(hidden_dims)

        self.latent_dim = latent_dim

        self.encoder = None
        self.decoder = None

        ## encoder
        modules = []
        in_dim_ = in_dim
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_features=in_dim_, out_features=h_dim),
                    nn.LeakyReLU(0.2),
                )
            )
            in_dim_ = h_dim

        # modules.append(nn.Linear(in_dim, out_features=hidden_dims[0]))
        # for i in range(len(hidden_dims) - 1):
        #     modules.append(
        #         nn.Sequential(
        #             nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
        #             nn.LeakyReLU(0.2),
        #         )
        #     )

        self.encoder = nn.Sequential(*modules)

        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        ## decoder
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[-1]),
            nn.LeakyReLU(0.2),
        )

        hidden_dims.reverse()
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.LeakyReLU(0.2),
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], in_dim),
            nn.Sigmoid(),
        )

    def encode(self, input: torch.tensor) -> List[torch.tensor]:
        result = self.encoder(input)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)

        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self, *args, **kwargs) -> torch.tensor:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """

        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs["M_N"]  # Account for the minibatch samples from the dataset
        # recons_loss = F.mse_loss(recons, input)
        recons_loss = 0.5 * torch.sum(
            (recons.reshape(recons.size(0), -1) - input.reshape(recons.size(0), -1))
            ** 2,
            1,
        ).mean()

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )

        loss = recons_loss + kld_weight * kld_loss

        return {
            "loss": loss,
            "Reconstruction_Loss": recons_loss.detach(),
            "KLD": kld_loss.detach(),  ## NOTE: issue in the PyTorch-VAE repo https://github.com/AntixK/PyTorch-VAE/issues/68
        }

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples.reshape(
            num_samples, 1, int(math.sqrt(self.in_dim)), int(math.sqrt(self.in_dim))
        )

    def generate(self, x: Tensor, **kwargs) -> Tensor:  ## x 是验证集中的一个batch
        """
        Given an input vectorized image x, returns the reconstructed image
        :param x: (Tensor) [B x (C x H x W)]
        :return: (Tensor) [B x (C x H x W)]
        """
        # print(x.shape[1])
        B, V = x.shape
        H, W = int(math.sqrt(V)), int(math.sqrt(V))

        recons = self.forward(x)[0]
        recons = torch.reshape(recons, shape=(B, 1, H, W))

        return recons

    ##TODO: Complete code here
    def manifold2d(self):
        pass
