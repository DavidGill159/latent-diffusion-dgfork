"""
- wrapper.encode() returns a spatial latent map that the LDM can accept
- wrapper.decode() lets you plug any sampled/denoised latent back into your original bVAE decoder
- the original latent traversal methods (on flat vectors) still work - you can:
            - pick a vector z, tweak z[i] += α
            - reshape to a map and decode through the wrapper
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

class VAEWrapperForLDM(nn.Module):
    """
    Wraps a flat-vector VAE to expose encode/decode functions
    that produce and accept latent *maps* for LDM compatibility.
    """
    def __init__(self, vae_model, latent_map_shape=(4, 16, 8)):
        super().__init__()
        self.vae = vae_model.eval()  # Don't train it within the LDM
        self.latent_map_shape = latent_map_shape
        self.latent_dim = int(torch.prod(torch.tensor(latent_map_shape)))
        assert self.vae.latent_dim == self.latent_dim, \
            f"Latent dim mismatch: VAE has {self.vae.latent_dim}, but reshaping to {self.latent_dim}"

    def encode(self, x):
        """
        Input: image tensor (B, 1, H, W)
        Output: latent map (B, C, H', W') e.g. (B, 4, 16, 8)
        """
        with torch.no_grad():
            z_flat = self.vae.encoder(x)
            mu, log_var = self.vae.mu(z_flat), self.vae.log_var(z_flat)
            z_sampled = self.vae.reparametrise(mu, log_var)
        return z_sampled.view(x.size(0), *self.latent_map_shape)

    def decode(self, z_map):
        """
        Input: latent map (B, C, H', W')
        Output: reconstructed image (B, 1, H, W)
        """
        z_flat = z_map.view(z_map.size(0), -1)
        with torch.no_grad():
            recon = self.vae.decoder(z_flat)
        return recon

    def forward(self, x):
        """
        For compatibility; same as encode-decode
        """
        return self.decode(self.encode(x))
