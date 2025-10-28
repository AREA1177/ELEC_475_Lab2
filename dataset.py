"""
dataset.py
ELEC 475 Lab 2 – Pet Nose Localization (Step 2)
Custom PyTorch Dataset for oxford-iiit-pet-noses dataset.
Supports data augmentation and image resizing to (3 × 227 × 227).
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


class PetNoseDataset(Dataset):
    """
    Custom dataset for loading pet images and (x, y) nose coordinates.
    """

    def __init__(self, root_dir, annotations_file, transform=None):
        """
        Args:
            root_dir (str): Path to 'images-original/images/'.
            annotations_file (str): Path to 'train-noses.txt' or 'test-noses.txt'.
            transform (callable, optional): torchvision transforms to apply to images.
        """
        self.root_dir = root_dir
        self.annotations_file = annotations_file
        self.transform = transform

        # Parse annotation file: each line → (filename, (x, y))
        self.samples = []
        with open(annotations_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue  # skip empty lines

                # Expected format: filename,"(x, y)"
                if "," not in line:
                    continue

                filename_part, coords_part = line.split(",", 1)
                filename = filename_part.strip()
                coords_str = coords_part.strip().replace('"', "").replace("(", "").replace(")", "")
                coords_str = coords_str.replace(",", " ").strip()
                parts = coords_str.split()

                if len(parts) != 2:
                    print(f"⚠️  Skipping malformed line: {line}")
                    continue  # skip bad entries gracefully

                try:
                    x, y = map(float, parts)
                    self.samples.append((filename, (x, y)))
                except ValueError:
                    print(f"⚠️  Could not parse coordinates in line: {line}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, (x, y) = self.samples[idx]
        img_path = os.path.join(self.root_dir, filename)
        image = Image.open(img_path).convert("RGB")

        # Get original image size BEFORE transform
        orig_w, orig_h = image.size  # (width, height)

        # Apply transforms (resize → tensor)
        if self.transform:
            image = self.transform(image)

        # Scale coordinates to match resized (227x227)
        scaled_x = x * (227.0 / orig_w)
        scaled_y = y * (227.0 / orig_h)
        target = torch.tensor([scaled_x, scaled_y], dtype=torch.float32)

        return image, target


def get_transforms(augment=False):
    """
    Returns torchvision transform pipeline.
    Args:
        augment (bool): If True, apply random augmentations.
    """
    if augment:
        transform = T.Compose([
            T.Resize((227, 227)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.RandomRotation(degrees=10),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = T.Compose([
            T.Resize((227, 227)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
    return transform


# ========== Reality check test ==========
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Example paths (adjust to your setup)
    IMG_DIR = r"C:\Users\Aidan\PycharmProjects\475 Lab 2\oxford-iiit-pet-noses\images-original\images"
    TRAIN_ANN = r"C:\Users\Aidan\PycharmProjects\475 Lab 2\oxford-iiit-pet-noses\train_noses.txt"

    dataset = PetNoseDataset(
        root_dir=IMG_DIR,
        annotations_file=TRAIN_ANN,
        transform=get_transforms(augment=True)
    )

    print(f"Dataset size: {len(dataset)} images")

    # Pick one random sample
    img, coords = dataset[0]
    print("Sample image shape:", img.shape)
    print("Ground truth coordinates:", coords.tolist())

    # Visualize (convert tensor → image)
    img_vis = img.permute(1, 2, 0).numpy()
    img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min())
    plt.imshow(img_vis)
    plt.title(f"Nose coords: {coords.tolist()}")
    plt.show()
