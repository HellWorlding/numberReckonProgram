# Created: 2025-12-26 00:33:57

# CLAUDE.md - Web Version

This file provides guidance to Claude Code (claude.ai/code) when working with the web version.

# numberReckonProgram - Web Version
Browser-based handwritten digit recognition using MNIST trained model.

## Project Overview
Web application version of the digit recognizer. Users can draw digits in a browser canvas and get predictions via a web API.

## Commands

**Install dependencies:**
```bash
pip install flask torch torchvision numpy pillow
```

**Run the server:**
```bash
python app.py
```

**Access:**
```
http://localhost:5000
```

## Tech Stack
- Python 3.12
- Flask (web framework)
- PyTorch (model inference)
- HTML5 Canvas (drawing interface)
- JavaScript (frontend interaction)

## Architecture

### Planned Structure
```
web_version/
├── app.py              # Flask server and API endpoints
├── model.py            # CNN model class (shared from desktop)
├── static/
│   ├── css/
│   │   └── style.css   # Styling
│   └── js/
│       └── canvas.js   # Canvas drawing and prediction logic
├── templates/
│   └── index.html      # Main page template
├── mnist_cnn_model.pth # Trained model weights
└── CLAUDE.md           # This file
```

### API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main page with drawing canvas |
| `/predict` | POST | Receive image data, return prediction |

## Code Style
- **File header**: Include creation date/time comment
- **Language**: All code and comments in English
- **Backend**: Follow Flask conventions, use blueprints for scaling
- **Frontend**: Vanilla JS, no frameworks required

## Development Notes

### Status
- Fully implemented and functional
- Shares model weights with root project (mnist_cnn_model.pth)

### Requirements
- CORS handling for API
- Base64 image encoding for canvas data transfer
- JSON response format for predictions
