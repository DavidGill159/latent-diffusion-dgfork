import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import os

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

# === Step 1: Load model from checkpoint ===
config_path = "configs/latent-diffusion/DG_bVAE_ldm_config.yaml"
ckpt_path = "logs/2025-07-14T21-38-54_DG_bVAE_ldm_config/checkpoints/last.ckpt" 


config = OmegaConf.load(config_path)
model = instantiate_from_config(config.model)
pl_sd = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(pl_sd["state_dict"], strict=False)
model.cuda()
model.eval()

# === Step 2: Load and preprocess image ===
image_path = "C:/code_DG/Image_dataset/All_natural_images/all_textures/braided_0107.png"  # <- replace with a real path to your test image
img = Image.open(image_path).convert("L").resize((512, 512))

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Map [0,1] → [-1,1]
])

img_tensor = transform(img).unsqueeze(0).to(model.device)  # [1,1,512,512]

# === Step 3: Encode → Decode ===
with torch.no_grad():
    z = model.first_stage_model.encode(img_tensor).mode()  # shape: [1,1,32,32] or [1,1,64,64] depending on bVAE
    recon = model.first_stage_model.decode(z)

# === Step 4: Save output ===
output_dir = "outputs/reconstructions"
os.makedirs(output_dir, exist_ok=True)

save_image((recon + 1) / 2, os.path.join(output_dir, "reconstructed.png"))

# Optional: Compare with original
save_image(torch.cat([(img_tensor + 1) / 2, (recon + 1) / 2], dim=0),
           os.path.join(output_dir, "comparison.png"), nrow=2)

print(f"Saved reconstruction to {output_dir}")
