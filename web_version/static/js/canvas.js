// Created: 2025-12-26 00:36:40
// Canvas drawing and prediction logic for digit recognizer

document.addEventListener('DOMContentLoaded', function() {
    const canvas = document.getElementById('drawingCanvas');
    const ctx = canvas.getContext('2d');
    const clearBtn = document.getElementById('clearBtn');
    const predictBtn = document.getElementById('predictBtn');
    const predictionResult = document.getElementById('predictionResult');
    const confidenceResult = document.getElementById('confidenceResult');
    const probBarsContainer = document.querySelector('.prob-bars');

    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;

    // Initialize canvas
    function initCanvas() {
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 15;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
    }

    // Initialize probability bars
    function initProbBars() {
        probBarsContainer.innerHTML = '';
        for (let i = 0; i < 10; i++) {
            const row = document.createElement('div');
            row.className = 'prob-row';
            row.innerHTML = `
                <span class="prob-label">${i}:</span>
                <div class="prob-bar-container">
                    <div class="prob-bar" id="prob-bar-${i}"></div>
                </div>
                <span class="prob-value" id="prob-value-${i}">0%</span>
            `;
            probBarsContainer.appendChild(row);
        }
    }

    // Get mouse/touch position
    function getPosition(e) {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;

        if (e.touches) {
            return {
                x: (e.touches[0].clientX - rect.left) * scaleX,
                y: (e.touches[0].clientY - rect.top) * scaleY
            };
        }
        return {
            x: (e.clientX - rect.left) * scaleX,
            y: (e.clientY - rect.top) * scaleY
        };
    }

    // Start drawing
    function startDrawing(e) {
        isDrawing = true;
        const pos = getPosition(e);
        lastX = pos.x;
        lastY = pos.y;
        e.preventDefault();
    }

    // Draw
    function draw(e) {
        if (!isDrawing) return;
        e.preventDefault();

        const pos = getPosition(e);
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();

        lastX = pos.x;
        lastY = pos.y;
    }

    // Stop drawing
    function stopDrawing() {
        isDrawing = false;
    }

    // Clear canvas
    function clearCanvas() {
        initCanvas();
        predictionResult.textContent = '-';
        confidenceResult.textContent = '-';
        for (let i = 0; i < 10; i++) {
            document.getElementById(`prob-bar-${i}`).style.width = '0%';
            document.getElementById(`prob-value-${i}`).textContent = '0%';
        }
    }

    // Predict digit
    async function predict() {
        const imageData = canvas.toDataURL('image/png');

        try {
            predictBtn.disabled = true;
            predictBtn.textContent = 'Predicting...';

            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ image: imageData })
            });

            const result = await response.json();

            if (result.error) {
                alert('Error: ' + result.error);
                return;
            }

            // Update prediction display
            predictionResult.textContent = result.digit;
            confidenceResult.textContent = (result.confidence * 100).toFixed(1) + '%';

            // Update probability bars
            result.probabilities.forEach((prob, i) => {
                const percentage = (prob * 100).toFixed(1);
                document.getElementById(`prob-bar-${i}`).style.width = percentage + '%';
                document.getElementById(`prob-value-${i}`).textContent = percentage + '%';
            });

        } catch (error) {
            alert('Error connecting to server: ' + error.message);
        } finally {
            predictBtn.disabled = false;
            predictBtn.textContent = 'Predict';
        }
    }

    // Event listeners - Mouse
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);

    // Event listeners - Touch
    canvas.addEventListener('touchstart', startDrawing);
    canvas.addEventListener('touchmove', draw);
    canvas.addEventListener('touchend', stopDrawing);

    // Button listeners
    clearBtn.addEventListener('click', clearCanvas);
    predictBtn.addEventListener('click', predict);

    // Initialize
    initCanvas();
    initProbBars();
});
