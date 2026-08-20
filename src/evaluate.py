import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import ARPSpoofNN
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,recall_score, f1_score, ConfusionMatrixDisplay)

# === 1. Load data ===
test_df = pd.read_csv("arp_test_preprocessed.csv")
X_test = test_df.drop(columns=["label"]).values.astype(np.float32)
y_test = test_df["label"].values.astype(np.float32).reshape(-1)

# === 2. Load model ===
model = ARPSpoofNN(input_dim=X_test.shape[1])
model.load_state_dict(torch.load("arp_spoofing_final_model.pt"))
model.eval()

# === 3. Predict ===
with torch.no_grad():
    preds = model(torch.tensor(X_test))
    y_pred = (preds.numpy() > 0.5).astype(int).reshape(-1)

# === 4. Metrics ===
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)


print(f"Accuracy : {accuracy:.6f}")
print(f"Precision: {precision:.6f}")
print(f"Recall   : {recall:.6f}")
print(f"F1 Score : {f1:.6f}")

# === 5. Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred, normalize="true")
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Attack"])
disp.plot(cmap="Blues", values_format=".2f")
plt.title("Confusion Matrix (%) - DNN Classifier")
plt.tight_layout()
plt.show()
