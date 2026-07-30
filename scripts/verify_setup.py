# File Name: verify_setup.py
# Last Updated: July 29, 2026
# Description:
#   A quick sanity-check script to confirm the local environment is set up
#   correctly before running any real training or evaluation. It checks
#   that the required packages are installed, prints out which device
#   (GPU or CPU) is available, and does a small test run of the model to
#   make sure it builds and produces output of the right shape.

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torchvision
import numpy as np
import matplotlib

print("Verification: Robust Overfitting")
print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print(f"Torchvision Version: {torchvision.__version__}")
print(f"Numpy Version: {np.__version__}")
print(f"Matplotlib Version: {matplotlib.__version__}")

# Prefer GPU acceleration when available, falling back to CPU.
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Device: CUDA (GPU accelerator) is available.")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Device: MPS (Metal Performance Shaders on Apple Silicon) is available.")
else:
    device = torch.device("cpu")
    print("Device: CPU (no GPU accelerator found).")

try:
    from Models.preact_resnet import PreActResNet18
    model = PreActResNet18(num_classes=10).to(device)
    print("PreActResNet-18 model loaded successfully.")
    
    # Run a small batch of 2 CIFAR-10-sized images through the model to
    # confirm it actually works end-to-end, not just that it builds.
    mock_input = torch.randn(2, 3, 32, 32).to(device)
    mock_output = model(mock_input)
    print(f"Mock Input Shape: {mock_input.shape}")
    print(f"Mock Output Shape: {mock_output.shape}")
    
    assert mock_output.shape == (2, 10), "Output shape mismatch!"
    print("Sanity-check forward pass: SUCCESS")
    print("Environment setup is verified and working!")
except Exception as e:
    print(f"Verification Failed: {e}")
    sys.exit(1)