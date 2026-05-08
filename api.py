import sys
sys.stdout.reconfigure(encoding='utf-8')

import io, os, json, logging
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
import cv2

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO, format="%(levelname)s:  %(message)s")
logger = logging.getLogger(__name__)

SESSION = None
TORCH_MODEL = None
USE_ONNX = False
MODEL_ACCURACY = 0.0
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


def load_onnx():
    global SESSION, USE_ONNX
    try:
        import onnxruntime as ort
        if Path("digit_model.onnx").exists():
            SESSION = ort.InferenceSession("digit_model.onnx", providers=["CPUExecutionProvider"])
            USE_ONNX = True
            logger.info("ONNX model loaded")
            return True
    except ImportError:
        pass
    return False


def load_torch():
    global TORCH_MODEL, MODEL_ACCURACY
    try:
        import torch
        import torch.nn as nn

        class DigitCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                )
                self.classifier = nn.Sequential(
                    nn.Linear(64 * 7 * 7, 128), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(128, 10)
                )

            def forward(self, x):
                return self.classifier(self.features(x).view(x.size(0), -1))

        if Path("digit_model.pth").exists():
            ck = torch.load("digit_model.pth", map_location="cpu", weights_only=False)
            m = DigitCNN()
            m.load_state_dict(ck["model_state_dict"])
            m.eval()
            TORCH_MODEL = m
            MODEL_ACCURACY = ck.get("accuracy", 0.0)
            logger.info("PyTorch model loaded (fallback)")
            return True
    except Exception as e:
        logger.error(f"PyTorch load failed: {e}")
    return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL_ACCURACY
    if not load_onnx():
        load_torch()
    h = Path("static/history.json")
    if h.exists():
        MODEL_ACCURACY = json.loads(h.read_text()).get("final_accuracy", 0) / 100
    yield


app = FastAPI(title="DigitAI", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
Path("static").mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


def preprocess(image_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Cannot decode image")
    # user draws on white canvas, MNIST expects white digit on black
    if np.mean(img) > 128:
        img = 255 - img
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = (img - MNIST_MEAN) / MNIST_STD
    return img.reshape(1, 1, 28, 28).astype(np.float32)


@app.get("/", response_class=HTMLResponse)
async def root():
    p = Path("static/index.html")
    return HTMLResponse(p.read_text(encoding="utf-8")) if p.exists() else HTMLResponse("<h1>index.html not found</h1>", 404)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": USE_ONNX or TORCH_MODEL is not None,
        "mode": "onnx" if USE_ONNX else "pytorch",
        "model_accuracy": round(MODEL_ACCURACY * 100, 2),
        "device": "cpu"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not (USE_ONNX or TORCH_MODEL):
        raise HTTPException(503, "No model loaded. Run train_model.py then export_onnx.py")
    try:
        tensor = preprocess(await file.read())
        if USE_ONNX:
            from scipy.special import softmax
            logits = SESSION.run(["output"], {"input": tensor})[0][0]
            probs = softmax(logits)
        else:
            import torch
            with torch.no_grad():
                logits = TORCH_MODEL(torch.tensor(tensor))
                probs = torch.softmax(logits, dim=1).numpy()[0]

        digit = int(np.argmax(probs))
        return {
            "digit": digit,
            "confidence": round(float(probs[digit]) * 100, 2),
            "probabilities": [round(float(p) * 100, 2) for p in probs]
        }
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(e)
        raise HTTPException(500, "Prediction error")


@app.get("/metrics")
async def metrics():
    p = Path("static/history.json")
    if p.exists():
        return json.loads(p.read_text())
    return {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": [], "final_accuracy": 0, "epochs": 0}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"\nDigitAI running at http://localhost:{port}\n")
    uvicorn.run("api:app", host="0.0.0.0", port=port, log_level="info")
