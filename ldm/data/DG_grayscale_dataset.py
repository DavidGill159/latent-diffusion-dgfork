import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class GrayscaleImageDataset(Dataset):
    def __init__(self, root, image_size=256):
        self.root = root
        self.image_paths = [os.path.join(root, fname) for fname in os.listdir(root) if fname.endswith(".png")]

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # ✅ Ensure single-channel grayscale
            transforms.Resize((image_size, image_size)),  # ✅ Use the passed-in value
            transforms.ToTensor(),  # ✅ Ensures shape is (C, H, W)
            transforms.Normalize(mean=[0.5], std=[0.5])

        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("L")  # Load in grayscale mode
        image = self.transform(image)
        return {"image": image}









#######################################################################################################

# # OLD VERSION - hardcodes image res leading to misaligned dims that need correcting = slows down processing.

# import os
# from torch.utils.data import Dataset
# from PIL import Image
# import torchvision.transforms as transforms

# class GrayscaleImageDataset(Dataset):
#     def __init__(self, root, image_size=256):
#         self.root = root
#         self.image_paths = [os.path.join(root, fname) for fname in os.listdir(root) if fname.endswith(".png")]

#         self.transform = transforms.Compose([
#             transforms.Grayscale(num_output_channels=1),  # ✅ Ensure single-channel grayscale
#             transforms.Resize((256, 256)),
#             transforms.ToTensor(),  # ✅ Ensures shape is (C, H, W)
#         ])

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         image = Image.open(self.image_paths[idx]).convert("L")  # Load in grayscale mode
#         image = self.transform(image)
#         return {"image": image}
