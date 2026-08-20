import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from model import ARPSpoofNN


# === 1. Load data ===
train_df = pd.read_csv("arp_train_preprocessed.csv")
test_df  = pd.read_csv("arp_test_preprocessed.csv")

X_train = train_df.drop(columns=["label"]).values.astype(np.float32)
y_train = train_df["label"].values.astype(np.float32).reshape(-1, 1)

X_test  = test_df.drop(columns=["label"]).values.astype(np.float32)
y_test  = test_df["label"].values.astype(np.float32).reshape(-1, 1)

train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
    batch_size=1024, shuffle=True
)

test_loader = DataLoader(
    TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
    batch_size=1024, shuffle=False
)

# === 2. Setup model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = ARPSpoofNN(input_dim=X_train.shape[1]).to(device)

criterion = nn.BCELoss()
optimizer = optim.NAdam(model.parameters(), lr=0.001)
epochs = 25

train_losses = []
test_accuracies = []


# === 3. Training Loop ===
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()

        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)

    avg_loss = total_loss / len(train_loader.dataset)
    train_losses.append(avg_loss)

    # === Test Accuracy ===
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = (model(xb) > 0.5).float()

            correct += (preds == yb).sum().item()
            total += yb.size(0)

    acc = correct / total
    test_accuracies.append(acc)

    print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f} | Test Acc: {acc:.6f}")


torch.save(model.state_dict(), "arp_spoofing_final_model.pt")
