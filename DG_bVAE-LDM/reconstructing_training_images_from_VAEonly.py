import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from ldm.models.autoencoder import AutoencoderKL  # or your bVAE class
from ldm.util import instantiate_from_config

# === Load model from config & checkpoint ===
config_path = "configs/latent-diffusion/DG_bVAE_ldm_config.yaml"
config = OmegaConf.load(config_path)
vae_config = config.model.params.first_stage_config
model = instantiate_from_config(vae_config)

ckpt = torch.load(vae_config.params.ckpt_path, map_location="cpu")
model.load_state_dict(ckpt["state_dict"], strict=False)
model.eval().cuda()  # or .to('cuda')

# === Load a grayscale image ===
img_path = "C:/code_DG/Image_dataset/All_natural_images/all_textures/braided_0107.png"
img = Image.open(img_path).convert("L")  # force grayscale
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])
img_tensor = transform(img).unsqueeze(0).to(model.device)  # [1,1,512,512]

# === Run through β-VAE ===
with torch.no_grad():
    z = model.encode(img_tensor)
    recon = model.decode(z.sample()).cpu()  # use z.mode() if no noise wanted

# === Show side-by-side ===
orig_img = img_tensor.squeeze().cpu().numpy()
recon_img = recon.squeeze().numpy()

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(orig_img, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("β-VAE Reconstruction")
plt.imshow(recon_img, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
