# Created: 2025-12-26 00:33:57

# CLAUDE.md - Desktop Version

This file provides guidance to Claude Code (claude.ai/code) when working with the desktop version.

# numberReckonProgram - Desktop Version
Native desktop application for handwritten digit recognition using Tkinter GUI.

## Project Overview
Standalone desktop application with Tkinter GUI. Users draw digits on a canvas and get real-time predictions with confidence scores and probability bars.

## Commands

**Install dependencies:**
```bash
pip install torch torchvision numpy pillow
```

**Run the application:**
```bash
python digit_recognizer.py
```

**Run on Windows (double-click):**
```
numberReckonProgram.bat
```

## Tech Stack
- Python 3.12
- PyTorch 2.9+ (CNN model, training, inference)
- torchvision (MNIST dataset loading)
- NumPy (array operations)
- Pillow (image processing)
- Tkinter (GUI, built-in with Python)

## Architecture

### Project Structure
```
desktop_version/
├── digit_recognizer.py    # Main application (CNN model + GUI)
├── numberReckonProgram.bat # Windows launcher script
├── mnist_cnn_model.pth    # Trained model weights
├── dist/
│   └── numberReckonProgram.exe  # Standalone executable (~150MB)
├── data/
│   └── MNIST/raw/         # MNIST dataset (auto-downloaded)
└── CLAUDE.md              # This file
```

### Core Classes
| Class | Description |
|-------|-------------|
| `CNN` | PyTorch neural network: 3 conv blocks (Conv2d → BatchNorm → ReLU → MaxPool) + FC layers with dropout |
| `MNISTModel` | Model wrapper for training, evaluation, save/load, and prediction |
| `DigitRecognizerApp` | Tkinter GUI with 280x280 canvas, prediction display, and probability bars |

## Code Style
- **File header**: Include creation date/time comment
- **Language**: All code and comments in English
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Constants**: UPPER_CASE at module level
- **Docstrings**: Triple-quoted strings for class and method documentation

## Development Notes

### Current State
- Fully implemented and functional
- Model achieves ~98.8% accuracy on MNIST test set
- Training runs for 5 epochs by default
- GPU support via CUDA (falls back to CPU automatically)

### Limitations
- Canvas input may differ from MNIST data distribution
- Model expects white digits on black background
- Image normalization hardcoded: mean=0.1307, std=0.3081
