import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical
import math
import time
import os

ONEOVERSQRT2PI = 1.0 / math.sqrt(2 * math.pi)

color1 = 'red'
color2 = 'black'
color3 = 'royalblue'
font2 = 23
font3 = 18
plt.rc('font',family='Times New Roman', size=15)
plt.rcParams['savefig.dpi'] = 300

device = 'cuda'
p1 = 250
p2 = 255


def to_numpy(x):
    return x.cpu().data.numpy() if device == 'cuda' else x.detach().numpy()


class Forward_Log_Net(nn.Module):
    def __init__(self):
        super(Forward_Log_Net, self).__init__()
        self.layer1 = nn.Linear(5, 3200)
        self.layer2 = nn.Linear(3200, 1600)
        self.layer3 = nn.Linear(1600, 800)
        self.layer4 = nn.Linear(800, 400)
        self.layer5 = nn.Linear(400, p1)
        self.activation = nn.LeakyReLU()

    def forward(self, inputs):
        x1 = self.activation(self.layer1(inputs))
        x2 = self.activation(self.layer2(x1))
        x3 = self.activation(self.layer3(x2))
        x4 = self.activation(self.layer4(x3))
        x5 = self.layer5(x4)
        return x5


class Forward_Net(nn.Module):
    def __init__(self):
        super(Forward_Net, self).__init__()
        self.layer1 = nn.Linear(5, 3200)
        self.layer2 = nn.Linear(3200, 1600)
        self.layer3 = nn.Linear(1600, 800)
        self.layer4 = nn.Linear(800, 400)
        self.layer5 = nn.Linear(400, p1)
        self.activation = nn.ReLU6()

    def forward(self, inputs):
        x1 = self.activation(self.layer1(inputs))
        x2 = self.activation(self.layer2(x1))
        x3 = self.activation(self.layer3(x2))
        x4 = self.activation(self.layer4(x3))
        x5 = self.layer5(x4)
        return x5


class Inverse_Net(nn.Module):
    def __init__(self):
        super(Inverse_Net, self).__init__()
        self.layer1 = nn.Linear(p1, 400)
        self.layer3 = nn.Linear(400, 800)
        self.layer4 = nn.Linear(800, 1600)
        self.layer5 = nn.Linear(1600, 3200)
        self.last = nn.Linear(3200, 5)
        self.activation = nn.ReLU6(True)

    def forward(self, inputs):
        x1 = self.activation(self.layer1(inputs))
        x3 = self.activation(self.layer3(x1))
        x4 = self.activation(self.layer4(x3))
        x5 = self.activation(self.layer5(x4))
        x6 = self.last(x5)
        return x6


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.layer_0_input = nn.Linear(10, 100)
        self.layer_0_label = nn.Linear(p1, 100)
        self.layer1 = nn.Linear(200, 400)
        self.bn1 = nn.BatchNorm1d(400)
        self.layer2 = nn.Linear(400, 800)
        self.bn2 = nn.BatchNorm1d(800)
        self.layer3 = nn.Linear(800, 1600)
        self.bn3 = nn.BatchNorm1d(1600)
        self.layer4 = nn.Linear(1600, 3200)
        self.bn4 = nn.BatchNorm1d(3200)
        self.last = nn.Linear(3200, 5)
        self.activation = nn.ReLU6(True)

    def forward(self, z, y):
        z = self.activation(self.layer_0_input(z))
        y = self.activation(self.layer_0_label(y))
        x0 = torch.cat((z, y), 1)
        x1 = self.activation(self.layer1(x0))
        x2 = self.activation(self.layer2(x1))
        x3 = self.activation(self.layer3(x2))
        x4 = self.activation(self.layer4(x3))
        x5 = self.last(x4)
        return x5


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer_0_input = nn.Linear(5, 100)
        self.layer_0_label = nn.Linear(p1, 100)
        self.layer1 = nn.Linear(200, 3200)
        self.bn1 = nn.BatchNorm1d(3200)
        self.layer2 = nn.Linear(3200, 1600)
        self.bn2 = nn.BatchNorm1d(1600)
        self.layer3 = nn.Linear(1600, 800)
        self.bn3 = nn.BatchNorm1d(800)
        self.layer4 = nn.Linear(800, 400)
        self.bn4 = nn.BatchNorm1d(400)
        self.last = nn.Linear(400, 1)
        self.activation = nn.ReLU6(True)

    def forward(self, z, y):
        z = self.activation(self.layer_0_input(z))
        y = self.activation(self.layer_0_label(y))
        x0 = torch.cat((z, y), 1)
        x1 = self.activation(self.layer1(x0))
        x2 = self.activation(self.layer2(x1))
        x3 = self.activation(self.layer3(x2))
        x4 = self.activation(self.layer4(x3))
        x5 = self.last(x4)
        return x5


class Encoder(nn.Module):
    def __init__(self, d):
        super(Encoder, self).__init__()
        self.layer_0_data = nn.Linear(5, d)
        self.layer_0_label = nn.Linear(p1, 200-d)
        self.layer1 = nn.Linear(200, 3200)
        self.bn1 = nn.BatchNorm1d(3200)
        self.layer2 = nn.Linear(3200, 1600)
        self.bn2 = nn.BatchNorm1d(1600)
        self.layer3 = nn.Linear(1600, 800)
        self.bn3 = nn.BatchNorm1d(800)
        self.layer4 = nn.Linear(800, 400)
        self.bn4 = nn.BatchNorm1d(400)
        self.layer5 = nn.Linear(400, 100)
        self.activation = nn.ReLU6(True)

    def forward(self, x, y):
        x = self.activation(self.layer_0_data(x))
        y = self.activation(self.layer_0_label(y))
        x0 = torch.cat((x, y), 1)
        x1 = self.activation(self.bn1(self.layer1(x0)))
        x2 = self.activation(self.bn2(self.layer2(x1)))
        x3 = self.activation(self.bn3(self.layer3(x2)))
        x4 = self.activation(self.bn4(self.layer4(x3)))
        x5 = self.layer5(x4)
        return x5


class Decoder(nn.Module):
    def __init__(self, ld, d):
        super(Decoder, self).__init__()
        self.layer_0_latent = nn.Linear(ld, d)
        self.layer_0_label = nn.Linear(p1, 200-d)
        self.layer1 = nn.Linear(200, 3200)
        self.bn1 = nn.BatchNorm1d(3200)
        self.layer2 = nn.Linear(3200, 1600)
        self.bn2 = nn.BatchNorm1d(1600)
        self.layer3 = nn.Linear(1600, 800)
        self.bn3 = nn.BatchNorm1d(800)
        self.layer4 = nn.Linear(800, 400)
        self.bn4 = nn.BatchNorm1d(400)
        self.layer5 = nn.Linear(400, 5)
        self.activation = nn.ReLU6(True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z, y):
        z = self.activation(self.layer_0_latent(z))
        y = self.activation(self.layer_0_label(y))
        x0 = torch.cat((z, y), 1)
        x1 = self.activation(self.bn1(self.layer1(x0)))
        x2 = self.activation(self.bn2(self.layer2(x1)))
        x3 = self.activation(self.bn3(self.layer3(x2)))
        x4 = self.activation(self.bn4(self.layer4(x3)))
        x5 = self.sigmoid(self.layer5(x4))
        return x5


class VAE(nn.Module):
    def __init__(self, encoder, decoder, latent_dim):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._enc_mu = nn.Linear(100, latent_dim)
        self._enc_log_sigma = nn.Linear(100, latent_dim)

    def re_param(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self._enc_mu(h_enc)
        log_sigma = self._enc_log_sigma(h_enc)
        sigma = log_sigma.exp_()
        return mu, sigma

    def sample_z(self, mu, sigma):
        std = Variable(torch.randn(*mu.size()), requires_grad=False).to(device)
        z = mu + sigma * std
        return z

    def latent_loss(self, z_mean, z_std):
        mean_sq = z_mean * z_mean
        std_sq = z_std * z_std
        return 0.5 * torch.mean(mean_sq + std_sq - torch.log(std_sq) - 1)

    def forward(self, x, y):
        latent = self.encoder(x, y)
        mu, sigma = self.re_param(latent)
        z = self.sample_z(mu, sigma)
        dec = self.decoder(z, y)
        ll = self.latent_loss(mu, sigma)
        return dec, ll


class PDN(nn.Module):
    def __init__(self, in_features=p1, out_features=5, num_gaussians=50):
        super(PDN, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(in_features, in_features * 2),
            nn.ReLU6(),
            nn.Linear(in_features * 2, in_features * 4),
            nn.ReLU6(),
            nn.Linear(in_features * 4, in_features * 8),
            nn.ReLU6(),
            nn.Linear(in_features * 8, in_features * 16),
            nn.ReLU6(),
        )

        self.out = out_features
        self.num = num_gaussians

        self.pi = nn.Linear(in_features * 16, num_gaussians)
        self.sigma = nn.Linear(in_features * 16, out_features * num_gaussians)
        self.mu = nn.Linear(in_features * 16, out_features * num_gaussians)

    def forward(self, x):
        x = self.hidden(x)
        pi = F.softmax(self.pi(x), dim=1)
        sigma = torch.exp(self.sigma(x))
        sigma = sigma.view(-1, self.num, self.out)
        sigma = F.threshold(sigma, 1e-3, 1e-3)
        mu = self.mu(x)
        # mu = torch.exp(mu)
        mu = mu.view(-1, self.num, self.out)
        return pi, sigma, mu


def gaussian_probability(sigma, mu, target):
    target = target.unsqueeze(1).expand_as(mu)
    ret = ONEOVERSQRT2PI * torch.exp(-0.5 * ((target - mu) / sigma) ** 2) / sigma
    ret = torch.prod(ret, 2)
    return ret


def mdn_loss(pi, sigma, mu, target):
    prob = pi * gaussian_probability(sigma, mu, target)
    nll = -torch.log(torch.sum(prob, dim=1) + 1)
    loss = torch.mean(nll)
    return loss


def sample(pi, sigma, mu):
    categorical = Categorical(pi)
    pis = list(categorical.sample().data)
    samples = Variable(sigma.data.new(sigma.size(0), sigma.size(2)))
    for j, idx in enumerate(pis):
        samples[j] = mu[j, idx]
    return samples


# find multiple structures that fit
def multi_sample(pi, sigma, mu):
    val_pis = find_structure(to_numpy(pi[0]), 1e-2)
    print("Validate probabilities and indexes:")
    print(val_pis)
    samples = Variable(sigma.data.new(len(val_pis), sigma.size(0), sigma.size(2)))
    for i in range(len(val_pis)):
        for j, idx in enumerate([val_pis[i][0]]):
            samples[i, j] = mu[j, idx]
    return samples


def find_structure(pis, threshold):
    structures = []
    for i in range(len(pis)):
        if pis[i] > threshold:
            structures.append([i, pis[i]])
    return structures



