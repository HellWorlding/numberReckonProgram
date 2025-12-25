# Created: 2025-12-26 00:55:14
"""
Convert PyTorch MNIST model to TensorFlow.js format
"""

import torch
import torch.nn as nn
import numpy as np
import os

# Define the same CNN architecture as PyTorch version
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


def convert_to_onnx():
    """Convert PyTorch model to ONNX format"""
    print("Loading PyTorch model...")
    model = CNN()
    model.load_state_dict(torch.load('mnist_cnn_model.pth', map_location='cpu', weights_only=True))
    model.eval()

    print("Converting to ONNX...")
    dummy_input = torch.randn(1, 1, 28, 28)

    torch.onnx.export(
        model,
        dummy_input,
        'mnist_model.onnx',
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print("ONNX model saved to mnist_model.onnx")


def train_keras_model():
    """Train a Keras model with same architecture and convert to TensorFlow.js"""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    print("Building Keras model...")
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),

        # First conv block
        layers.Conv2D(32, 3, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(2, 2),

        # Second conv block
        layers.Conv2D(64, 3, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(2, 2),

        # Third conv block
        layers.Conv2D(64, 3, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),

        # FC layers
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize and reshape
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    print("Training model...")
    model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=5,
        validation_split=0.1,
        verbose=1
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f}")

    # Save Keras model in H5 format
    model.save('mnist_keras_model.h5')
    print("Keras model saved to mnist_keras_model.h5")

    # Also save as SavedModel format
    model.export('mnist_saved_model')
    print("SavedModel saved to mnist_saved_model/")

    return model


if __name__ == '__main__':
    print("=" * 50)
    print("Model Conversion for TensorFlow.js")
    print("=" * 50)

    # Train Keras model and convert to TensorFlow.js
    train_keras_model()

    print("\nConversion complete!")
    print("TensorFlow.js model files are in: tfjs_model/")
