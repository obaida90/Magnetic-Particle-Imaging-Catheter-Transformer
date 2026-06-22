
# This Python script performs inference using a trained CatheterTransformer model to predict 3D catheter tip positions from input sensor signals. 
#It first loads and reshapes preprocessed signal data from an unseen dataset (X_unseen_scaled.npy),
#organizing it into batches with 2 channels and a feature length determined from the checkpoint. 
#The model, a transformer-based architecture with embedding dimensions of 128, 8 attention heads, and 4 layers (changed according to the setting), 
#is loaded with pretrained weights from Bestweight_model.pth and set to evaluation mode. 

import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from BAmodeModel2 import CatheterTransformer  

# === Configs ===
MODEL_PATH = "/Bestweight_model.pth"
SIGNAL_PATH = "8000BA/X_unseen_scaled.npy"
OUTPUT_FOLDER = "UnseenBApositions 2"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === Device Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load and reshape signals ===
X_unseen = np.load(SIGNAL_PATH)
CHANNELS = 2
features_per_channel = X_unseen.shape[1] // CHANNELS
X_unseen_reshaped = X_unseen.reshape(-1, CHANNELS, features_per_channel)
print(f"Input shape for inference: {X_unseen_reshaped.shape}")

# === Load model ===
checkpoint = torch.load(MODEL_PATH, map_location=device)
signal_len = checkpoint.get("signal_length", features_per_channel)

model = CatheterTransformer(
    in_channels=CHANNELS,
    signal_len=signal_len,
    embed_dim=128,
    num_heads=8,
    num_layers=4,
    dim_feedforward=512,
    dropout=0.2,
    patch_size=32
).to(device)

state_dict = checkpoint["model_state_dict"]
model.load_state_dict(state_dict, strict=False)
model.eval()

# === Prepare DataLoader ===
X_tensor = torch.tensor(X_unseen_reshaped, dtype=torch.float32)
loader = DataLoader(TensorDataset(X_tensor), batch_size=32, shuffle=False)

# === Perform inference ===
all_preds = []
with torch.no_grad():
    for batch in loader:
        x = batch[0].to(device)
        preds = model(x)
        all_preds.append(preds.cpu().numpy())

# === Save predictions ===
predictions = np.concatenate(all_preds, axis=0)  # Shape: (8000, 3)
np.save(os.path.join(OUTPUT_FOLDER, "catheter_positions_xyz.npy"), predictions)
np.save(os.path.join(OUTPUT_FOLDER, "x_pred.npy"), predictions[:, 0])
np.save(os.path.join(OUTPUT_FOLDER, "y_pred.npy"), predictions[:, 1])
np.save(os.path.join(OUTPUT_FOLDER, "z_pred.npy"), predictions[:, 2])
