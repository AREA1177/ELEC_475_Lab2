"""
test_model_shapes.py
Utility script to verify SnoutNet architecture and correctness.

Run:
    python test_model_shapes.py
"""

import torch
from torchsummary import summary
from model import SnoutNet


def inspect_forward_pass():
    print("=== SnoutNet Forward Pass Inspection ===")
    model = SnoutNet()

    # Create dummy input tensor (Batch=1, Channels=3, Width=227, Height=227)
    x = torch.randn(1, 3, 227, 227)
    print(f"Input shape: {x.shape}")

    # Pass through each layer manually for inspection
    with torch.no_grad():
        x = model.conv1(x); print(f"After conv1: {x.shape}")
        x = model.relu1(x)
        x = model.pool1(x); print(f"After pool1: {x.shape}")

        x = model.conv2(x); print(f"After conv2: {x.shape}")
        x = model.relu2(x)
        x = model.pool2(x); print(f"After pool2: {x.shape}")

        x = model.conv3(x); print(f"After conv3: {x.shape}")
        x = model.relu3(x)

        x = model.conv4(x); print(f"After conv4: {x.shape}")
        x = model.relu4(x)

        x = model.conv5(x); print(f"After conv5: {x.shape}")
        x = model.relu5(x)
        x = model.pool5(x); print(f"After pool5: {x.shape}")

        # Flatten check using view
        x = x.view(x.size(0), -1)
        print(f"After flatten (view): {x.shape}")

        x = model.fc1(x); print(f"After fc1: {x.shape}")
        x = model.fc2(x); print(f"After fc2: {x.shape}")
        x = model.fc3(x); print(f"After fc3 (output): {x.shape}")

    print("✅ Forward pass completed successfully.\n")


def model_summary():
    print("=== SnoutNet Architecture Summary ===")
    model = SnoutNet()
    summary(model, (3, 227, 227), device="cpu")
    print("✅ Summary completed.\n")


def gradient_test():
    print("=== SnoutNet Gradient Test ===")
    model = SnoutNet()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Dummy batch of 4 images and random regression targets
    x = torch.randn(4, 3, 227, 227)
    target = torch.randn(4, 2)

    output = model(x)
    loss = criterion(output, target)
    print(f"Loss: {loss.item():.6f}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("✅ Backward pass + optimization successful.\n")


if __name__ == "__main__":
    inspect_forward_pass()
    model_summary()
    gradient_test()
    print("All verification tests passed ✅")