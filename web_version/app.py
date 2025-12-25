# Created: 2025-12-26 00:36:40
"""
Flask Web Application for Handwritten Digit Recognition
Provides API endpoints for digit prediction via HTML5 Canvas
"""

from flask import Flask, render_template, request, jsonify
from model import MNISTModel
import numpy as np
from PIL import Image
import base64
import io
import os

app = Flask(__name__)

# Initialize model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'mnist_cnn_model.pth')
model = None


def get_model():
    """Lazy load the model"""
    global model
    if model is None:
        model = MNISTModel(MODEL_PATH)
    return model


@app.route('/')
def index():
    """Main page with drawing canvas"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Receive canvas image and return prediction"""
    try:
        data = request.get_json()

        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        # Decode base64 image
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)

        # Open image and convert to grayscale
        image = Image.open(io.BytesIO(image_bytes)).convert('L')

        # Resize to 28x28
        image = image.resize((28, 28), Image.Resampling.LANCZOS)

        # Convert to numpy array and normalize to 0-1
        img_array = np.array(image).astype('float32') / 255.0

        # Get prediction
        mnist_model = get_model()
        result = mnist_model.predict(img_array)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    print("=" * 50)
    print("Handwritten Digit Recognition - Web Version")
    print("=" * 50)
    print("Starting server at http://localhost:5000")
    print("-" * 50)

    # Preload model
    get_model()

    app.run(debug=True, host='0.0.0.0', port=5000)
