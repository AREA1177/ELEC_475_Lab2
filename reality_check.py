"""
reality_check.py
ELEC 475 Lab 2 – Pet Nose Localization
Verifies Dataset + DataLoader correctness by printing and visualizing samples.
"""

import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import PetNoseDataset, get_transforms


# === CONFIGURE PATHS HERE ===
IMG_DIR = r"C:\Users\Aidan\PycharmProjects\475 Lab 2\oxford-iiit-pet-noses\images-original\images"
TRAIN_ANN = r"C:\Users\Aidan\PycharmProjects\475 Lab 2\oxford-iiit-pet-noses\train_noses.txt"


def visualize_sample(image_tensor, coords, idx):
    """Visualize one sample with GT nose coordinates overlayed."""
    img = image_tensor.permute(1, 2, 0).numpy()  # CxHxW → HxWxC
    img = (img - img.min()) / (img.max() - img.min())

    plt.imshow(img)
    plt.scatter(coords[0], coords[1], color='red', s=20, label="GT Nose")
    plt.title(f"Sample {idx} | Nose coords: ({coords[0]:.1f}, {coords[1]:.1f})")
    plt.legend()
    plt.show()


def reality_check():
    """Iterate through the dataset and visualize some samples."""
    # Load dataset
    dataset = PetNoseDataset(
        root_dir=IMG_DIR,
        annotations_file=TRAIN_ANN,
        transform=get_transforms(augment=False)
    )

    # Create DataLoader for iteration
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    print(f"✅ Dataset loaded: {len(dataset)} samples")
    print("Iterating through 3 batches to verify correctness...\n")

    # Iterate through a few batches
    for batch_idx, (images, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}:")
        for i in range(len(images)):
            print(f"  → Sample {i + 1} | GT Nose: {labels[i].tolist()}")
        if batch_idx == 0:  # visualize only first batch
            visualize_sample(images[0], labels[0], 0)
        if batch_idx >= 2:
            break

    print("\n✅ Reality check completed successfully!")


if __name__ == "__main__":
    if not os.path.exists(TRAIN_ANN):
        print(f"❌ Annotations file not found: {TRAIN_ANN}")
    elif not os.path.exists(IMG_DIR):
        print(f"❌ Image directory not found: {IMG_DIR}")
    else:
        reality_check()
