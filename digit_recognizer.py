"""
Handwritten Digit Recognition System
Uses MNIST dataset to train a CNN model and provides a GUI for drawing digits.
Uses PyTorch for better cross-platform compatibility.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw
import os

# Constants
MODEL_PATH = "mnist_cnn_model.pth"
CANVAS_SIZE = 280
IMAGE_SIZE = 28

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
    """Wrapper class for model training and prediction"""

    def __init__(self):
        self.model = CNN().to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, epochs=5, batch_size=128):
        """Train the model on MNIST dataset"""
        print("Loading MNIST dataset...")

        # Data transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Load MNIST dataset
        train_dataset = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )

        test_dataset = datasets.MNIST(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        print(f"Training on {device}...")
        print("-" * 50)

        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)

                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

                if (batch_idx + 1) % 100 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}], "
                          f"Batch [{batch_idx+1}/{len(train_loader)}], "
                          f"Loss: {loss.item():.4f}")

            train_acc = 100 * correct / total
            print(f"Epoch [{epoch+1}/{epochs}] - "
                  f"Loss: {running_loss/len(train_loader):.4f}, "
                  f"Accuracy: {train_acc:.2f}%")

        # Evaluate on test set
        test_acc = self.evaluate(test_loader)
        print(f"\nTest Accuracy: {test_acc:.2f}%")

    def evaluate(self, test_loader):
        """Evaluate model on test data"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        return 100 * correct / total

    def save(self, path=MODEL_PATH):
        """Save the trained model"""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path=MODEL_PATH):
        """Load a pre-trained model"""
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            self.model.eval()
            print(f"Model loaded from {path}")
            return True
        return False

    def predict(self, image):
        """Predict digit from image"""
        self.model.eval()

        # Normalize the image
        image = (image - 0.1307) / 0.3081

        # Convert to tensor
        tensor = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            output = self.model(tensor)
            probabilities = torch.softmax(output, dim=1)[0]
            predicted_digit = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_digit].item()

        return predicted_digit, confidence, probabilities.cpu().numpy()


class DigitRecognizerApp:
    """GUI application for handwritten digit recognition"""

    def __init__(self, model):
        self.model = model
        self.root = tk.Tk()
        self.root.title("Handwritten Digit Recognizer")
        self.root.resizable(False, False)

        # Drawing state
        self.last_x = None
        self.last_y = None

        # Create PIL image for drawing
        self.image = Image.new('L', (CANVAS_SIZE, CANVAS_SIZE), 'black')
        self.draw = ImageDraw.Draw(self.image)

        self.setup_ui()

    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")

        # Title
        title_label = ttk.Label(
            main_frame,
            text="Draw a digit (0-9)",
            font=('Helvetica', 16, 'bold')
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))

        # Canvas for drawing
        self.canvas = tk.Canvas(
            main_frame,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg='black',
            cursor='cross'
        )
        self.canvas.grid(row=1, column=0, padx=(0, 10))

        # Bind mouse events
        self.canvas.bind('<Button-1>', self.start_draw)
        self.canvas.bind('<B1-Motion>', self.draw_line)
        self.canvas.bind('<ButtonRelease-1>', self.stop_draw)

        # Right panel for results
        result_frame = ttk.Frame(main_frame)
        result_frame.grid(row=1, column=1, sticky="n")

        # Prediction display
        pred_label = ttk.Label(
            result_frame,
            text="Prediction:",
            font=('Helvetica', 12)
        )
        pred_label.grid(row=0, column=0, sticky="w")

        self.prediction_var = tk.StringVar(value="-")
        self.prediction_label = ttk.Label(
            result_frame,
            textvariable=self.prediction_var,
            font=('Helvetica', 48, 'bold'),
            foreground='blue'
        )
        self.prediction_label.grid(row=1, column=0, pady=10)

        # Confidence display
        conf_label = ttk.Label(
            result_frame,
            text="Confidence:",
            font=('Helvetica', 12)
        )
        conf_label.grid(row=2, column=0, sticky="w")

        self.confidence_var = tk.StringVar(value="-")
        self.confidence_label = ttk.Label(
            result_frame,
            textvariable=self.confidence_var,
            font=('Helvetica', 14)
        )
        self.confidence_label.grid(row=3, column=0, pady=(0, 20))

        # Probability bars
        prob_label = ttk.Label(
            result_frame,
            text="Probabilities:",
            font=('Helvetica', 12)
        )
        prob_label.grid(row=4, column=0, sticky="w")

        self.prob_frame = ttk.Frame(result_frame)
        self.prob_frame.grid(row=5, column=0, pady=5)

        self.prob_bars = []
        self.prob_labels = []
        for i in range(10):
            frame = ttk.Frame(self.prob_frame)
            frame.grid(row=i, column=0, sticky="w", pady=1)

            label = ttk.Label(frame, text=f"{i}:", width=2)
            label.grid(row=0, column=0)

            bar = ttk.Progressbar(frame, length=100, mode='determinate')
            bar.grid(row=0, column=1, padx=5)

            prob_text = ttk.Label(frame, text="0%", width=6)
            prob_text.grid(row=0, column=2)

            self.prob_bars.append(bar)
            self.prob_labels.append(prob_text)

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=10)

        self.clear_btn = ttk.Button(
            button_frame,
            text="Clear",
            command=self.clear_canvas
        )
        self.clear_btn.grid(row=0, column=0, padx=5)

        self.predict_btn = ttk.Button(
            button_frame,
            text="Predict",
            command=self.predict_digit
        )
        self.predict_btn.grid(row=0, column=1, padx=5)

        # Instructions
        instructions = ttk.Label(
            main_frame,
            text="Draw a digit on the black canvas, then click 'Predict'",
            font=('Helvetica', 10),
            foreground='gray'
        )
        instructions.grid(row=3, column=0, columnspan=2, pady=(10, 0))

    def start_draw(self, event):
        """Start drawing on mouse click"""
        self.last_x = event.x
        self.last_y = event.y

    def draw_line(self, event):
        """Draw line while mouse is dragging"""
        if self.last_x and self.last_y:
            # Draw on canvas
            self.canvas.create_line(
                self.last_x, self.last_y, event.x, event.y,
                fill='white', width=15, capstyle=tk.ROUND, smooth=True
            )
            # Draw on PIL image
            self.draw.line(
                [self.last_x, self.last_y, event.x, event.y],
                fill='white', width=15
            )
        self.last_x = event.x
        self.last_y = event.y

    def stop_draw(self, event):
        """Stop drawing on mouse release"""
        self.last_x = None
        self.last_y = None

    def clear_canvas(self):
        """Clear the canvas"""
        self.canvas.delete('all')
        self.image = Image.new('L', (CANVAS_SIZE, CANVAS_SIZE), 'black')
        self.draw = ImageDraw.Draw(self.image)
        self.prediction_var.set("-")
        self.confidence_var.set("-")
        for i in range(10):
            self.prob_bars[i]['value'] = 0
            self.prob_labels[i]['text'] = "0%"

    def preprocess_image(self):
        """Preprocess the drawn image for prediction"""
        # Resize to 28x28
        img = self.image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)

        # Convert to numpy array and normalize
        img_array = np.array(img).astype('float32') / 255.0

        return img_array

    def predict_digit(self):
        """Predict the drawn digit"""
        # Preprocess image
        img_array = self.preprocess_image()

        # Get prediction
        digit, confidence, probabilities = self.model.predict(img_array)

        if digit is not None:
            # Update prediction display
            self.prediction_var.set(str(digit))
            self.confidence_var.set(f"{confidence*100:.1f}%")

            # Update probability bars
            for i in range(10):
                prob = probabilities[i] * 100
                self.prob_bars[i]['value'] = prob
                self.prob_labels[i]['text'] = f"{prob:.1f}%"

    def run(self):
        """Start the application"""
        self.root.mainloop()


def main():
    """Main function to run the digit recognizer"""
    print("=" * 50)
    print("Handwritten Digit Recognition System")
    print("=" * 50)
    print(f"Using device: {device}")

    # Initialize model
    mnist_model = MNISTModel()

    # Try to load existing model, otherwise train new one
    if not mnist_model.load():
        print("\nNo pre-trained model found. Training new model...")
        mnist_model.train(epochs=5)
        mnist_model.save()

    print("\nStarting GUI application...")
    print("Draw a digit on the canvas and click 'Predict'")
    print("-" * 50)

    # Start GUI
    app = DigitRecognizerApp(mnist_model)
    app.run()


if __name__ == "__main__":
    main()
