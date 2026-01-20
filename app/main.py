from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tempfile
from pathlib import Path

import src.serving.predictor as predictor_module
predictor = predictor_module.Predictor()

app = FastAPI(title="Chicken Feces Classifier API")


@app.get("/")
def root():
    return {"status": "ok", "message": "Chicken feces classifier running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    suffix = Path(file.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        result = predictor.predict(tmp_path)
        return JSONResponse(content=result)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
