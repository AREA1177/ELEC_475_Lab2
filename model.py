"""
model.py
ELEC 475 Lab 2 – Pet Nose Localization (Step 1)
Implements SnoutNet: a CNN regression model for pet nose coordinate prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SnoutNet(nn.Module):
    """
    SnoutNet — convolutional neural network for pet nose localization.

    Input shape : (batch_size, 3, 227, 227)
    Output shape: (batch_size, 2)  → predicted (x, y) coordinates
    """

    def __init__(self):
        super(SnoutNet, self).__init__()

        # ======== Convolutional feature extractor ========
        # Uses Conv2d and MaxPool2d as documented:
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        # https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=96,
            kernel_size=11, stride=4, padding=2
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(
            in_channels=96, out_channels=256,
            kernel_size=5, stride=1, padding=2
        )
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(
            in_channels=256, out_channels=384,
            kernel_size=3, stride=1, padding=1
        )
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(
            in_channels=384, out_channels=384,
            kernel_size=3, stride=1, padding=1
        )
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(
            in_channels=384, out_channels=256,
            kernel_size=3, stride=1, padding=1
        )
        self.relu5 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)

        # ======== Fully connected regression head ========
        # After the last pooling layer, the feature map is approximately (256 × 6 × 6).
        # Hence, the flattened size is 256 * 6 * 6 = 9216.

        self.drop1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(in_features=256 * 6 * 6, out_features=4096)
        self.relu6 = nn.ReLU(inplace=True)

        self.drop2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=4096, out_features=1024)
        self.relu7 = nn.ReLU(inplace=True)

        # Output: 2 neurons for continuous regression values (x, y)
        self.fc3 = nn.Linear(in_features=1024, out_features=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward propagation through SnoutNet."""

        # Convolutional + pooling blocks
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.pool5(x)

        # Flatten using Tensor.view (as per PyTorch docs)
        # https://pytorch.org/docs/stable/generated/torch.Tensor.view.html
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # reshape to [batch_size, num_features]

        # Fully connected regression head
        x = self.drop1(x)
        x = self.fc1(x)
        x = self.relu6(x)

        x = self.drop2(x)
        x = self.fc2(x)
        x = self.relu7(x)

        x = self.fc3(x)  # output (x, y)
        return x


# ========== Self-test ==========
if __name__ == "__main__":
    print("Testing SnoutNet model...")
    model = SnoutNet()
    dummy_input = torch.randn(1, 3, 227, 227)
    output = model(dummy_input)
    print("Output tensor shape:", output.shape)
    assert output.shape == (1, 2), "❌ Output shape should be [1, 2]"
    print("✅ SnoutNet model passes basic shape test.")
