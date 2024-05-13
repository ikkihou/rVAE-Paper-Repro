#!/usr/bin/env python3
"""
##
##       filename: blocks.py
##        created: 2024/05/13
##         author: Paul_Bao
##            IDE: Neovim
##       Version : 1.0
##       Contact : paulbao@mail.ecust.edu.cn
"""

# here put the import lib


import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod


class BaseVAE(nn.Module, ABC):
    def __init__(self):
        super(BaseVAE, self).__init__()

    @abstractmethod
    def encode(self, x):
        pass

    @abstractmethod
    def reparameterize(self, mu, log_var):
        pass

    @abstractmethod
    def decode(self, z):
        pass

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var


class ConcreteVAE(BaseVAE):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(ConcreteVAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)

        # Decoder layers
        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        mean = self.fc_mean(h)
        log_var = self.fc_log_var(h)
        return mean, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc2(z))
        recon = torch.sigmoid(self.fc3(h))
        return recon
