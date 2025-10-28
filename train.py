"""
train.py
ELEC 475 Lab 2 – Step 3 Training SnoutNet Model (without augmentation)
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from model import SnoutNet
from dataset import PetNoseDataset, get_transforms


# ====================== CONFIG ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 25
LR = 1e-4

IMG_DIR = r"C:\Users\Aidan\PycharmProjects\475 Lab 2\oxford-iiit-pet-noses\images-original\images"
TRAIN_ANN = r"C:\Users\Aidan\PycharmProjects\475 Lab 2\oxford-iiit-pet-noses\train_noses.txt"
TEST_ANN  = r"C:\Users\Aidan\PycharmProjects\475 Lab 2\oxford-iiit-pet-noses\test_noses.txt"

MODEL_SAVE_PATH = "snoutnet_model.pth"
# ====================================================


def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for images, targets in dataloader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(dataloader.dataset)


def validate_epoch(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * images.size(0)
    return running_loss / len(dataloader.dataset)


def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("SnoutNet Training Loss Curve (No Augmentations)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_curve_no_aug.png")
    plt.show()


def main():
    print(f"Using device: {DEVICE}")

    # ---------- Dataset & Loaders ----------
    train_dataset = PetNoseDataset(
        root_dir=IMG_DIR,
        annotations_file=TRAIN_ANN,
        transform=get_transforms(augment=False)
    )
    test_dataset = PetNoseDataset(
        root_dir=IMG_DIR,
        annotations_file=TEST_ANN,
        transform=get_transforms(augment=False)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    # ---------- Model & Training Setup ----------
    model = SnoutNet().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_losses, val_losses = [], []

    print("Starting training loop ...\n")
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_loss   = validate_epoch(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch+1}/{EPOCHS}]  "
              f"Train Loss: {train_loss:.6f}  |  Val Loss: {val_loss:.6f}")

    # ---------- Save model ----------
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n✅ Training complete! Model saved to {MODEL_SAVE_PATH}")

    # ---------- Plot loss curves ----------
    plot_loss(train_losses, val_losses)


if __name__ == "__main__":
    main()
