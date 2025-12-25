# Created: 2025-12-26 00:59:00
"""
Export PyTorch MNIST model to ONNX format
"""

import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    import sys
    import os

    # Fix encoding issue on Windows
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

    print("Loading PyTorch model...")
    model = CNN()
    model.load_state_dict(torch.load('mnist_cnn_model.pth', map_location='cpu', weights_only=True))
    model.eval()

    print("Exporting to ONNX...")
    dummy_input = torch.randn(1, 1, 28, 28)

    # Use legacy export for compatibility
    torch.onnx.export(
        model,
        dummy_input,
        'docs/mnist_model.onnx',
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamo=False
    )
    print("ONNX model saved to docs/mnist_model.onnx")
