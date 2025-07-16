
import torch

# === Step 1: Load original .pt state dict ===
pt_path = "C:/code_DG/latent-texture-image-space/Convolutional_bVAE_Pipeline/training_logs/beta_4/vae_model_epoch199.pt"
state_dict = torch.load(pt_path)

# === Step 2: Wrap and save as .ckpt ===
ckpt_path = "C:/code_DG/latent-texture-image-space/Convolutional_bVAE_Pipeline/training_logs/beta_4/wrapped_for_LDM/beta4_vae_epoch199.ckpt"
torch.save({"state_dict": state_dict}, ckpt_path)
print(f"✅ Saved wrapped checkpoint to: {ckpt_path}")

# === Step 3: Verify contents ===
print(f"\n🔍 Verifying contents of: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location="cpu")

if "state_dict" not in ckpt:
    print("❌ ERROR: 'state_dict' not found in checkpoint!")
else:
    print("✅ Checkpoint contains 'state_dict' with the following parameters:")
    for k, v in ckpt["state_dict"].items():
        print(f" - {k}: shape {tuple(v.shape)}")
