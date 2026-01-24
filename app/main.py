from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
import tempfile
from pathlib import Path

from src.serving.predictor import Predictor

app = FastAPI(title="Chicken Feces Classifier API")

predictor = Predictor()

@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Chicken Feces Classifier</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                background-color: #f0f2f5;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
                color: #333;
            }
            .container {
                text-align: center;
                background: white;
                padding: 3rem;
                border-radius: 12px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                max-width: 500px;
                width: 90%;
            }
            h1 {
                margin-top: 0;
                color: #2c3e50;
                font-weight: 700;
                margin-bottom: 2rem;
            }
            p {
                margin-bottom: 2rem;
                line-height: 1.6;
                color: #666;
            }
            .button {
                display: inline-block;
                background-color: #0070f3;
                color: white;
                padding: 1rem 2rem;
                text-decoration: none;
                border-radius: 6px;
                font-weight: 600;
                transition: background-color 0.2s;
            }
            .button:hover {
                background-color: #0056b3;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Chicken Feces Classifier API</h1>
            <p>Welcome to the API. Use the interactive documentation to test the classifier.</p>
            <a href="/docs" class="button">Go to API Docs</a>
        </div>
    </body>
    </html>
    """

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix or ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = predictor.predict(tmp_path)
        return JSONResponse(content=result)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
