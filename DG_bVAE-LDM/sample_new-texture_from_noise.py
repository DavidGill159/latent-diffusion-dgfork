from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
import torch

# === 1. Load Config and Model Checkpoint ===
config_path = "configs/latent-diffusion/DG_bVAE_ldm_config.yaml"
ckpt_path = "logs/2025-07-14T21-38-54_DG_bVAE_ldm_config/checkpoints/last.ckpt"



config = OmegaConf.load(config_path)
model = instantiate_from_config(config.model)
pl_sd = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(pl_sd["state_dict"], strict=False)
model.cuda()
model.eval()

# === 2. Initialize Sampler ===
sampler = DDIMSampler(model)

# === 3. Run Sampling ===
batch_size = 8
shape = (1, 64, 64)  # Your latent z shape (z_channels, H, W)
samples, _ = sampler.sample(S=50, batch_size=batch_size, shape=shape, eta=1.0, unconditional_guidance_scale=1.0)

# === 4. Decode Latents to Images ===
decoded_imgs = model.decode_first_stage(samples)

# === 5. Save or visualize ===
from torchvision.utils import save_image
save_image(decoded_imgs, "generated.png", nrow=4, normalize=True)
