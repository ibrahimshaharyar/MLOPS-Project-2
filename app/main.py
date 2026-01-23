from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tempfile
from pathlib import Path

from src.serving.predictor import Predictor

app = FastAPI(title="Chicken Feces Classifier API")

predictor = Predictor()

@app.get("/")
def root():
    return {"status": "ok", "message": "API is running"}

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
