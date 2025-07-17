# File: ldm/data/DG_custom.py

import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CustomTrain(Dataset):
    def __init__(self, data_root, size=512):
        self.root = data_root
        self.image_paths = [os.path.join(data_root, fname) for fname in os.listdir(data_root) if fname.endswith(".png")]

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # ✅ Ensure single-channel grayscale
            transforms.Resize((size, size)),  # ✅ Use the passed-in value
            transforms.ToTensor(),  # ✅ Ensures shape is (C, H, W)
            transforms.Normalize(mean=[0.5], std=[0.5])

        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("L")  # Load in grayscale mode
        image = self.transform(image)
        return {"image": image}


class CustomValidation(CustomTrain):
    pass

