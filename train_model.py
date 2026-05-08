import os
import sys
import json
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report

os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
sys.stdout.reconfigure(encoding='utf-8')
matplotlib.use('Agg')
warnings.filterwarnings('ignore')

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
MODEL_PATH = "digit_model.pth"
STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading MNIST dataset...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")


class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def predict_proba(self, x):
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)


model = DigitCNN().to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params:,}")

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
            print(f"Batch {batch_idx+1}/{len(loader)} | Loss: {loss.item():.4f}")
            
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


print(f"Training CNN for {EPOCHS} epochs...")

for epoch in range(1, EPOCHS + 1):
    print(f"Epoch {epoch}/{EPOCHS}")
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc, preds, targets = eval_epoch(model, test_loader, criterion)
    scheduler.step()

    history['train_acc'].append(train_acc * 100)
    history['val_acc'].append(val_acc * 100)
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")


final_loss, final_acc, final_preds, final_targets = eval_epoch(model, test_loader, criterion)
print(f"Final Test Accuracy: {final_acc*100:.2f}%")
print(classification_report(final_targets, final_preds, target_names=[str(i) for i in range(10)]))


torch.save({
    'model_state_dict': model.state_dict(),
    'accuracy': final_acc,
    'history': history,
}, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

with open(os.path.join(STATIC_DIR, 'history.json'), 'w') as f:
    json.dump({
        'accuracy': history['train_acc'],
        'val_accuracy': history['val_acc'],
        'loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'final_accuracy': round(final_acc * 100, 2),
        'epochs': EPOCHS,
    }, f)


fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#0a0a0f')

epochs_range = range(1, EPOCHS + 1)

ax1 = axes[0]
ax1.set_facecolor('#0d0d1a')
ax1.plot(epochs_range, history['train_acc'], 'o-', color='#7c3aed', lw=2.5, label='Train Accuracy', markersize=5)
ax1.plot(epochs_range, history['val_acc'], 's-', color='#06b6d4', lw=2.5, label='Val Accuracy', markersize=5)
ax1.fill_between(epochs_range, history['train_acc'], alpha=0.1, color='#7c3aed')
ax1.fill_between(epochs_range, history['val_acc'], alpha=0.1, color='#06b6d4')
ax1.set_title('Model Accuracy', color='white', fontsize=14, fontweight='bold', pad=12)
ax1.set_xlabel('Epoch', color='#aaa', fontsize=11)
ax1.set_ylabel('Accuracy (%)', color='#aaa', fontsize=11)
ax1.tick_params(colors='#aaa')
ax1.spines[:].set_color('#333')
ax1.legend(facecolor='#1a1a2e', labelcolor='white')
ax1.grid(True, color='#222', linestyle='--', alpha=0.5)

ax2 = axes[1]
ax2.set_facecolor('#0d0d1a')
ax2.plot(epochs_range, history['train_loss'], 'o-', color='#f59e0b', lw=2.5, label='Train Loss', markersize=5)
ax2.plot(epochs_range, history['val_loss'], 's-', color='#ef4444', lw=2.5, label='Val Loss', markersize=5)
ax2.fill_between(epochs_range, history['train_loss'], alpha=0.1, color='#f59e0b')
ax2.fill_between(epochs_range, history['val_loss'], alpha=0.1, color='#ef4444')
ax2.set_title('Model Loss', color='white', fontsize=14, fontweight='bold', pad=12)
ax2.set_xlabel('Epoch', color='#aaa', fontsize=11)
ax2.set_ylabel('Loss', color='#aaa', fontsize=11)
ax2.tick_params(colors='#aaa')
ax2.spines[:].set_color('#333')
ax2.legend(facecolor='#1a1a2e', labelcolor='white')
ax2.grid(True, color='#222', linestyle='--', alpha=0.5)

plt.tight_layout(pad=2)
plt.savefig(os.path.join(STATIC_DIR, 'training_graph.png'), dpi=150, bbox_inches='tight', facecolor='#0a0a0f')

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

print("Training complete!")
