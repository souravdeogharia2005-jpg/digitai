import os, sys
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
sys.stdout.reconfigure(encoding='utf-8')
"""
DigitAI -- CNN Training Script
================================
Trains a Convolutional Neural Network on the MNIST dataset (70,000 images).
Architecture:
  Conv2D(32) → ReLU → MaxPool →
  Conv2D(64) → ReLU → MaxPool →
  Flatten → Dense(128) → Dropout(0.3) → Dense(10, Softmax)

Target accuracy: 99%+
Usage: python train_model.py
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BATCH_SIZE   = 64
EPOCHS       = 10
LEARNING_RATE = 0.001
MODEL_PATH   = "digit_model.pth"
STATIC_DIR   = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[*] Device: {device}")

# ─────────────────────────────────────────────
# PHASE 2 — DATA PREPROCESSING PIPELINE
# ─────────────────────────────────────────────
print("\n[+] Loading MNIST dataset (70,000 images)...")

transform = transforms.Compose([
    transforms.ToTensor(),          # 0–255 → 0.0–1.0  (Normalization)
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean & std
])

# Download if not present
train_dataset = datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"   [ok] Training samples : {len(train_dataset):,}")
print(f"   [ok] Test samples     : {len(test_dataset):,}")
print(f"   [ok] Image shape      : 1x28x28 (grayscale)")
print(f"   [ok] Classes          : 0-9 digits")

# ─────────────────────────────────────────────
# PHASE 3 — CNN MODEL ARCHITECTURE
# ─────────────────────────────────────────────
class DigitCNN(nn.Module):
    """
    CNN Architecture:
      Input (1×28×28)
        → Conv2D(32, 3×3) + ReLU          # Feature extraction layer 1
        → MaxPool(2×2)                     # Spatial reduction
        → Conv2D(64, 3×3) + ReLU          # Feature extraction layer 2
        → MaxPool(2×2)                     # Spatial reduction
        → Flatten                          # 1D representation
        → Dense(128) + ReLU               # Pattern learning
        → Dropout(0.3)                     # Regularization
        → Dense(10) + Softmax             # Output: probabilities for 0–9
    """
    def __init__(self):
        super(DigitCNN, self).__init__()
        # Feature extraction block
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # → 32×28×28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # → 32×14×14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # → 64×14×14
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # → 64×7×7
        )
        # Classification block
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)   # Flatten
        x = self.classifier(x)
        return x  # Raw logits (CrossEntropyLoss handles softmax)

    def predict_proba(self, x):
        """Returns softmax probabilities (for inference)."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)


model = DigitCNN().to(device)
print("\n[*] CNN Architecture:")
print(model)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n   Total trainable parameters: {total_params:,}")

# ─────────────────────────────────────────────
# PHASE 4 — COMPILE + TRAINING SYSTEM
# ─────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += data.size(0)
        if (batch_idx + 1) % 200 == 0:
            print(f"      Batch {batch_idx+1}/{len(loader)} | Loss: {loss.item():.4f}")
    return total_loss / total, correct / total

def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_targets

print(f"\n[>>] Training CNN for {EPOCHS} epochs...\n" + '-'*50)

for epoch in range(1, EPOCHS + 1):
    print(f"\n  Epoch {epoch}/{EPOCHS}")
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc, preds, targets = eval_epoch(model, test_loader, criterion)
    scheduler.step()

    history['train_acc'].append(train_acc * 100)
    history['val_acc'].append(val_acc * 100)
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)

    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
    status = '[OK]' if val_acc > 0.97 else '[..]'
    print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc*100:.2f}%  {status}")

# ─────────────────────────────────────────────
# PHASE 5 — EVALUATION SYSTEM
# ─────────────────────────────────────────────
final_loss, final_acc, final_preds, final_targets = eval_epoch(model, test_loader, criterion)
print(f"\n[RESULTS] FINAL TEST RESULTS")
print('-'*40)
print(f"   Test Loss     : {final_loss:.4f}")
status2 = '[TARGET MET]' if final_acc > 0.97 else '[Below 97%]'
print(f"   Test Accuracy : {final_acc*100:.2f}%  {status2}")
print(f"\n{classification_report(final_targets, final_preds, target_names=[str(i) for i in range(10)])}")

# ─────────────────────────────────────────────
# PHASE 8 — SAVE MODEL
# ─────────────────────────────────────────────
torch.save({
    'model_state_dict': model.state_dict(),
    'accuracy': final_acc,
    'history': history,
}, MODEL_PATH)
print(f"\n[SAVED] Model saved -> {MODEL_PATH}")

# Save history for API
with open(os.path.join(STATIC_DIR, 'history.json'), 'w') as f:
    json.dump({
        'accuracy': history['train_acc'],
        'val_accuracy': history['val_acc'],
        'loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'final_accuracy': round(final_acc * 100, 2),
        'epochs': EPOCHS,
    }, f)

# ─────────────────────────────────────────────
# PHASE 9 — VISUALIZATION SYSTEM
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#0a0a0f')

epochs_range = range(1, EPOCHS + 1)

# Accuracy Plot
ax1 = axes[0]
ax1.set_facecolor('#0d0d1a')
ax1.plot(epochs_range, history['train_acc'], 'o-', color='#7c3aed', lw=2.5, label='Train Accuracy', markersize=5)
ax1.plot(epochs_range, history['val_acc'],   's-', color='#06b6d4', lw=2.5, label='Val Accuracy',   markersize=5)
ax1.fill_between(epochs_range, history['train_acc'], alpha=0.1, color='#7c3aed')
ax1.fill_between(epochs_range, history['val_acc'],   alpha=0.1, color='#06b6d4')
ax1.set_title('Model Accuracy', color='white', fontsize=14, fontweight='bold', pad=12)
ax1.set_xlabel('Epoch', color='#aaa', fontsize=11)
ax1.set_ylabel('Accuracy (%)', color='#aaa', fontsize=11)
ax1.tick_params(colors='#aaa')
ax1.spines[:].set_color('#333')
ax1.legend(facecolor='#1a1a2e', labelcolor='white')
ax1.grid(True, color='#222', linestyle='--', alpha=0.5)

# Loss Plot
ax2 = axes[1]
ax2.set_facecolor('#0d0d1a')
ax2.plot(epochs_range, history['train_loss'], 'o-', color='#f59e0b', lw=2.5, label='Train Loss', markersize=5)
ax2.plot(epochs_range, history['val_loss'],   's-', color='#ef4444', lw=2.5, label='Val Loss',   markersize=5)
ax2.fill_between(epochs_range, history['train_loss'], alpha=0.1, color='#f59e0b')
ax2.fill_between(epochs_range, history['val_loss'],   alpha=0.1, color='#ef4444')
ax2.set_title('Model Loss', color='white', fontsize=14, fontweight='bold', pad=12)
ax2.set_xlabel('Epoch', color='#aaa', fontsize=11)
ax2.set_ylabel('Loss', color='#aaa', fontsize=11)
ax2.tick_params(colors='#aaa')
ax2.spines[:].set_color('#333')
ax2.legend(facecolor='#1a1a2e', labelcolor='white')
ax2.grid(True, color='#222', linestyle='--', alpha=0.5)

plt.tight_layout(pad=2)
plt.savefig(os.path.join(STATIC_DIR, 'training_graph.png'), dpi=150, bbox_inches='tight', facecolor='#0a0a0f')
print(f"   [chart] Training graphs saved -> static/training_graph.png")

# Sample predictions grid
fig2, axes2 = plt.subplots(2, 5, figsize=(12, 5))
fig2.patch.set_facecolor('#0a0a0f')
fig2.suptitle('Sample Predictions', color='white', fontsize=14, fontweight='bold', y=1.02)
model.eval()
test_images, test_labels = next(iter(test_loader))
with torch.no_grad():
    probs = model.predict_proba(test_images[:10].to(device)).cpu().numpy()
for i, ax in enumerate(axes2.flat):
    ax.set_facecolor('#0d0d1a')
    ax.imshow(test_images[i].squeeze(), cmap='inferno')
    pred = np.argmax(probs[i])
    true = test_labels[i].item()
    color = '#06b6d4' if pred == true else '#ef4444'
    ax.set_title(f"Pred: {pred} | True: {true}", color=color, fontsize=9, fontweight='bold')
    ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(STATIC_DIR, 'sample_predictions.png'), dpi=120, bbox_inches='tight', facecolor='#0a0a0f')
print(f"   [img] Sample predictions saved -> static/sample_predictions.png")

print('='*50)
print('  [DONE] TRAINING COMPLETE!')
print(f"  Final Accuracy: {final_acc*100:.2f}%")
print(f"  Model: {MODEL_PATH}")
print('  Now run: python api.py')
print('='*50)
