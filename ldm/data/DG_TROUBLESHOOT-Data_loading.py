

# from ldm.data.DG_grayscale_dataset import GrayscaleImageDataset
# from torch.utils.data import DataLoader

# # Load dataset
# dataset = GrayscaleImageDataset(root="C:/code_DG/Image_dataset/All_natural_images/all_textures", image_size=512)
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# # Iterate over one batch and check shape
# for batch in dataloader:
#     print(batch["image"].shape)  # Expected output: (4, 1, 256, 256)
#                                  #                   4 - batch size
#                                  #                      1 - images are grayscale (single channeled)
#                                  #                         256 - images are resized correctly
#     break



from ldm.data.DG_custom import CustomTrain
from torch.utils.data import DataLoader

# Load dataset
dataset = CustomTrain(root="C:/code_DG/Image_dataset/All_natural_images/all_textures", image_size=512)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Iterate over one batch and check shape
for batch in dataloader:
    print(batch["image"].shape)  # Expected output: (4, 1, 256, 256)
                                 #                   4 - batch size
                                 #                      1 - images are grayscale (single channeled)
                                 #                         256 - images are resized correctly
    break