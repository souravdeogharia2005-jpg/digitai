"""
export_onnx.py
==============
Converts digit_model.pth (PyTorch) -> digit_model.onnx
ONNX model is much lighter for production deployment:
  - PyTorch inference: ~700MB RAM
  - ONNX Runtime inference: ~50MB RAM
Run: python export_onnx.py
"""
import torch
import torch.nn as nn

class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64*7*7, 128), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(128, 10),
        )
    def forward(self, x):
        return self.classifier(self.features(x).view(x.size(0), -1))

print("[*] Loading PyTorch model...")
checkpoint = torch.load("digit_model.pth", map_location="cpu", weights_only=False)
model = DigitCNN()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("[*] Exporting to ONNX...")
dummy = torch.randn(1, 1, 28, 28)
torch.onnx.export(
    model, dummy, "digit_model.onnx",
    export_params=True,
    opset_version=13,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
)
print("[OK] Exported -> digit_model.onnx")

# Verify
import onnxruntime as ort
import numpy as np
sess = ort.InferenceSession("digit_model.onnx", providers=["CPUExecutionProvider"])
out = sess.run(["output"], {"input": np.random.randn(1,1,28,28).astype(np.float32)})[0]
print(f"[OK] ONNX verified. Output shape: {out.shape}  (should be (1,10))")
print("[DONE] digit_model.onnx is ready for production!")
