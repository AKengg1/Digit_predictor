"""
FastAPI server for the NumPy digit recognizer.

Run from the web/ folder:
    uvicorn main:app --reload
Then open http://localhost:8000/docs to test it.
"""
import os
import sys

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Make model/network.py importable (model/ is a sibling of web/)
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model")
sys.path.append(os.path.abspath(MODEL_DIR))

from network import predict  # noqa: E402

app = FastAPI(title="Digit Recognizer API")

# Allow the React dev server to call this API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "https://frontend-jade-six-96.vercel.app",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    pixels: list[float]  # 784 values between 0 and 1


@app.post("/predict")
def predict_digit(body: PredictRequest):
    if len(body.pixels) != 784:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 784 pixel values, got {len(body.pixels)}",
        )
    if min(body.pixels) < 0 or max(body.pixels) > 1:
        raise HTTPException(
            status_code=400,
            detail="Pixel values must be between 0 and 1",
        )
    return predict(body.pixels)