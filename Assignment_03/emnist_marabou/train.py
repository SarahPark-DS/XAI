# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.onnx
import os

# Hyperparameters setting
BATCH_SIZE = 256
EPOCHS = 10
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/emnist_fc.onnx"

# Data Load
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1736, ), (0.3317, )) # EMNIST Letters mean / std
])

train_dataset = datasets.EMNIST(
    root = "data", split = "letters", train = True, download = True, transform = transform
)
test_dataset = datasets.EMNIST(
    root = "data", split = "letters", train = False, download = True, transform = transform
)

# EMNIST Letters: labels are 1-26, shift to 0.25
train_dataset.targets -= 1
test_dataset.targets -= 1

train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False)


# Model
class EMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 26)
        )

    def forward(self, x):
        return self.net(x)
    
model = EMNIST().to(DEVICE)

# Train
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = LR)

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE),labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # validation
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            correct += (model(images).argmax(1) == labels).sum().item()
    acc = correct / len(test_dataset) * 100
    print(f"Epoch {epoch:02d} | loss: {total_loss/len(train_loader):.4f} | test acc: {acc:.2f}%")


# Export
os.makedirs("models", exist_ok = True)
model.eval()
dummy = torch.randn(1, 1, 28, 28).to(DEVICE)

torch.onnx.export(
    model, dummy, MODEL_PATH, 
    input_names = ["input"],
    output_names = ["output"],
    opset_version = 11
)

print(f"\nModel exported -> {MODEL_PATH}")
