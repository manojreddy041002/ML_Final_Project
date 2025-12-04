'''
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import json

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tfm = transforms.ToTensor()
train = datasets.MNIST(DATA_DIR, train=True, download=True, transform=tfm)
test  = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tfm)

train_loader = DataLoader(train, batch_size=128, shuffle=True)
test_loader  = DataLoader(test, batch_size=512)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.p  = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64*7*7, 10)
    def forward(self, x):
        x = self.p(F.relu(self.c1(x)))
        x = self.p(F.relu(self.c2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = CNN().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(3):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward(); opt.step()
    print("Epoch", epoch+1, "done")

# Evaluate
model.eval()
preds, gts = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        out = model(xb.to(device)).argmax(1).cpu()
        preds += out.tolist(); gts += yb.tolist()
acc = accuracy_score(gts, preds)
print(f"CNN accuracy: {acc:.4f}")
json.dump({"model": "CNN", "accuracy": acc}, open("mnist_cnn.json", "w"))
'''
# mnist_cnn.py
# Train a small CNN on MNIST and save results for comparison.
# Outputs:
#   - mnist_cnn.json  ({"model":"CNN","accuracy":...,"latency_sec":...,"params":...})
#   - mnist_cnn.pth   (PyTorch weights)

import json, os, time, random
import torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score

# ----- config -----
SEED       = 42
BATCH_TR   = 128
BATCH_TE   = 512
EPOCHS     = 3          # bump to 5–10 for max accuracy
LR         = 1e-3
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_JSON   = ROOT / "mnist_cnn.json"
OUT_WEIGHTS= ROOT / "mnist_cnn.pth"

# ----- setup -----
random.seed(SEED)
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

tfm = transforms.ToTensor()  # MNIST already normalized to [0,1]
train_ds = datasets.MNIST(DATA_DIR, train=True,  download=True, transform=tfm)
test_ds  = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tfm)

train_loader = DataLoader(train_ds, batch_size=BATCH_TR, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_TE, shuffle=False, num_workers=0)

# ----- model -----
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3, padding=1)   # 28x28 -> 28x28
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)  # 28x28 -> 28x28
        self.p  = nn.MaxPool2d(2,2)                # 28x28 -> 14x14 -> 7x7
        self.fc = nn.Linear(64*7*7, 10)
    def forward(self, x):
        x = self.p(F.relu(self.c1(x)))
        x = self.p(F.relu(self.c2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = CNN().to(device)
opt   = torch.optim.AdamW(model.parameters(), lr=LR)
lossf = nn.CrossEntropyLoss()

# ----- train -----
t0 = time.time()
for ep in range(1, EPOCHS+1):
    model.train()
    running = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        logits = model(xb)
        loss   = lossf(logits, yb)
        loss.backward()
        opt.step()
        running += loss.item() * xb.size(0)
    avg = running / len(train_loader.dataset)
    print(f"Epoch {ep}/{EPOCHS} - train_loss: {avg:.4f}")

# ----- eval -----
model.eval()
preds, gts = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        out = model(xb.to(device)).argmax(1).cpu()
        preds += out.tolist()
        gts   += yb.tolist()

acc = accuracy_score(gts, preds)
latency = time.time() - t0
params = sum(p.numel() for p in model.parameters())

print(f"Test accuracy: {acc:.4f}")
print(f"Total time (s): {latency:.2f}")
print(f"Params: {params}")

# ----- save artifacts -----
torch.save(model.state_dict(), OUT_WEIGHTS)
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(
        {"model":"CNN","accuracy":float(acc),"latency_sec":latency,"params":int(params)},
        f, indent=2
    )
print(f"Saved -> {OUT_JSON}, {OUT_WEIGHTS}")
