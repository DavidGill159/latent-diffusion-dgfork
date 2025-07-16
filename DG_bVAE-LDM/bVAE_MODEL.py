import torch
import torch.nn as nn
import torch.nn.functional as F
import piqa
import cv2
import numpy as np


class Encoder(nn.Module):
    def __init__(self, channels=1, latent_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, latent_dim)
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 8 * 8),
            nn.Unflatten(dim=-1, unflattened_size=(256, 8, 8)),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.decoder(x)


def apply_gabor_bank(image_tensor, device):
    ks = 31
    sigmas = [4.0]
    thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    lambdas = [10.0]
    gammas = [0.5]

    filters = [
        torch.tensor(cv2.getGaborKernel((ks, ks), sigma, theta, lam, gamma, 0, ktype=cv2.CV_32F),
                     dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        for sigma in sigmas for theta in thetas for lam in lambdas for gamma in gammas
    ]

    responses = [
        F.conv2d(image_tensor, filt.expand(image_tensor.shape[1], -1, -1, -1), padding=ks//2, groups=1)
        for filt in filters
    ]
    return torch.cat(responses, dim=1)


def apply_single_gabor(image_tensor, device):
    ks = 31
    sigma, theta, lam, gamma = 4.0, 0, 10.0, 0.5
    kernel = cv2.getGaborKernel((ks, ks), sigma, theta, lam, gamma, 0, ktype=cv2.CV_32F)
    kernel_tensor = torch.tensor(kernel, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    kernel_tensor = kernel_tensor.expand(image_tensor.shape[1], -1, -1, -1)
    return F.conv2d(image_tensor, kernel_tensor, padding=ks // 2, groups=1)


class VAE(nn.Module):
    def __init__(self, latent_dim=512, device='cpu'):
        super().__init__()
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim)
        self.mu = nn.Linear(latent_dim, latent_dim)
        self.log_var = nn.Linear(latent_dim, latent_dim)
        self.latent_dim = latent_dim
        self.device = device
        self.haarpsi = piqa.HaarPSI().to(device)

    def reparametrise(self, mu, log_var):
        return mu + log_var.mul(0.5).exp() * torch.randn_like(mu).to(self.device)

    def forward(self, x):
        z = self.encoder(x)
        mu, log_var = self.mu(z), self.log_var(z)
        z_sampled = self.reparametrise(mu, log_var)
        x_recon = self.decoder(z_sampled)
        return x_recon, mu, log_var, z_sampled

    def loss_fn(self, x, beta=0.0, loss_type='pixel_MSE'):
        x_recon, mu, log_var, _ = self.forward(x)
        recon_loss = F.mse_loss(x_recon, x, reduction='sum') / x.shape[0]
        kl_loss = -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp()) / x.shape[0]
        pixel_loss = F.mse_loss(x_recon, x, reduction='sum') / x.shape[0]
        haar_loss = torch.tensor(0.0, device=self.device)

        if loss_type == 'haarpsi_combined':
            x_rescaled = x * 0.5 + 0.5
            x_recon_rescaled = x_recon * 0.5 + 0.5
            x_rgb = x_rescaled.repeat(1, 3, 1, 1)
            x_recon_rgb = x_recon_rescaled.repeat(1, 3, 1, 1)
            haar_loss = (1.0 - self.haarpsi(x_rgb, x_recon_rgb)).mean()
            recon_loss = 0.5 * pixel_loss + 0.5 * haar_loss
        elif loss_type == 'single_gabor':
            fx, fx_recon = apply_single_gabor(x, self.device), apply_single_gabor(x_recon, self.device)
            recon_loss = F.mse_loss(fx_recon, fx, reduction='sum') / x.shape[0]
        elif loss_type == 'gabor_bank':
            fx, fx_recon = apply_gabor_bank(x, self.device), apply_gabor_bank(x_recon, self.device)
            recon_loss = F.mse_loss(fx_recon, fx, reduction='sum') / x.shape[0]
        elif loss_type != 'pixel_MSE':
            raise ValueError(f"Unsupported loss_type: {loss_type}")

        return recon_loss, beta * kl_loss, pixel_loss, haar_loss
