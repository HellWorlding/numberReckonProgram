# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# numberReckonProgram
MNIST based handwriting number recognition program with GUI interface.

## Project Overview
A handwritten digit recognition application that uses a CNN model trained on MNIST dataset. Users can draw digits (0-9) on a canvas and get real-time predictions with confidence scores.

### Versions
| Version | Description | Status |
|---------|-------------|--------|
| Desktop | Tkinter GUI native application + EXE | Implemented |
| Web | Flask + HTML5 Canvas browser app | Implemented |

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

### Core Classes
| Class | Description |
|-------|-------------|
| `CNN` | PyTorch neural network: 3 conv blocks (Conv2d → BatchNorm → ReLU → MaxPool) + FC layers with dropout |
| `MNISTModel` | Model wrapper for training, evaluation, save/load, and prediction |
| `DigitRecognizerApp` | Tkinter GUI with 280x280 canvas, prediction display, and probability bars |

### Project Structure
```
mnistPrac/
├── desktop_version/           # Desktop application (Tkinter GUI)
│   ├── digit_recognizer.py    # Main desktop app
│   ├── numberReckonProgram.bat # Windows launcher
│   ├── mnist_cnn_model.pth    # Trained model weights
│   ├── data/MNIST/raw/        # MNIST dataset
│   └── CLAUDE.md              # Desktop version docs
├── web_version/               # Web application (Flask + HTML5 Canvas)
│   ├── app.py                 # Flask server (planned)
│   ├── model.py               # CNN model class (planned)
│   ├── static/                # CSS, JS files (planned)
│   ├── templates/             # HTML templates (planned)
│   └── CLAUDE.md              # Web version docs
├── digit_recognizer.py        # Original standalone version
├── numberReckonProgram.bat    # Original Windows launcher
├── CLAUDE.md                  # Root project documentation
└── README.md                  # Project readme
```

## Code Style

- **File header**: All new files must include creation date and time as a comment at the top
  ```python
  # Created: 2025-12-26 14:30:00
  ```
- **Language**: All code and comments in English
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Constants**: UPPER_CASE at module level (e.g., `MODEL_PATH`, `CANVAS_SIZE`)
- **Docstrings**: Triple-quoted strings for class and method documentation
- **Error handling**: Use `if os.path.exists()` for file checks, `try/except` for critical operations
- **Type hints**: Not currently used, but recommended for new code

## Development Notes

### Current State
- Model achieves ~98.8% accuracy on MNIST test set
- Training runs for 5 epochs by default
- GPU support via CUDA (falls back to CPU automatically)

### Limitations
- Canvas input may differ from MNIST data distribution (hand-drawn vs scanned)
- Model expects white digits on black background
- Image normalization hardcoded: mean=0.1307, std=0.3081
