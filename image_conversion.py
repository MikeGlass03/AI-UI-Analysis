from PIL import Image
from torchvision.transforms import v2
import torch
from torch.utils.data import Dataset
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent

# Transform the image then convert to a tensor and normalize it to be used with Pytorch
image_transforms = v2.Compose([
    v2.Resize((384, 216), antialias=True),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=1.0),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard Values for normalization in PyTorch stuff
])

# Loads images and targets, transforming both to be used with Pytorch 
class UICritImageDataset(Dataset):
    def __init__(self, dataframe, transform=image_transforms):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image_path = PROJECT_DIR / row["image_path"]

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        targets = torch.tensor(
            [
                float(row['learnability']),
                float(row['efficiency']),
                float(row['design_quality_rating'])
            ],
            dtype=torch.float32,
        )

        return image, targets
        