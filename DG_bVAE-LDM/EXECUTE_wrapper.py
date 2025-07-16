

from vae_wrapper import VAEWrapperForLDM
from bVAE_MODEL import VAE

# Load your trained VAE
vae = VAE(latent_dim=512, device=device).to(device)
vae.load_state_dict(torch.load("your_path/vae_model.pt", map_location=device))

# Wrap for LDM
wrapper = VAEWrapperForLDM(vae_model=vae, latent_map_shape=(4, 16, 8))

# Use in LDM-style pipeline
z_map = wrapper.encode(image_batch)        # (B, 4, 16, 8)
image_recon = wrapper.decode(z_map)        # (B, 1, 512, 512)
