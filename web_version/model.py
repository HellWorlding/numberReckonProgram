# Created: 2025-12-26 00:36:40
"""
CNN Model for MNIST digit recognition
Shared model architecture for web version
"""

import torch
import torch.nn as nn
import os

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CNN(nn.Module):
    """CNN model for MNIST digit recognition"""

    def __init__(self):
        super(CNN, self).__init__()

        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Second convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Third convolutional block
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Fully connected layers
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


class MNISTModel:
    """Wrapper class for model inference"""

    def __init__(self, model_path="mnist_cnn_model.pth"):
        self.model = CNN().to(device)
        self.model_path = model_path
        self.load()

    def load(self):
        """Load a pre-trained model"""
        if os.path.exists(self.model_path):
            self.model.load_state_dict(
                torch.load(self.model_path, map_location=device, weights_only=True)
            )
            self.model.eval()
            print(f"Model loaded from {self.model_path}")
            return True
        else:
            print(f"Model not found at {self.model_path}")
            return False

    def predict(self, image):
        """Predict digit from preprocessed image array"""
        self.model.eval()

        # Normalize the image (MNIST mean and std)
        image = (image - 0.1307) / 0.3081

        # Convert to tensor
        tensor = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            output = self.model(tensor)
            probabilities = torch.softmax(output, dim=1)[0]
            predicted_digit = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_digit].item()

        return {
            'digit': predicted_digit,
            'confidence': float(confidence),
            'probabilities': probabilities.cpu().numpy().tolist()
        }
