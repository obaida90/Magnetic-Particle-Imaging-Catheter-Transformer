##################################################################################
# Signal extraction from MPI measurement, Std. mode, BA mode and HBA mode 
##################################################################################

import numpy as np
import h5py
import os
from numpy.fft import rfft

# File path
measurement_path = "/Path to the mdf file"

# Output directory
# output_dir = "Std. ModeSignals"
# output_dir = "HBA ModeSignals"
output_dir = "/Std,BA, HBA MODE SIGNALS"
os.makedirs(output_dir, exist_ok=True)

# Open measurement file
with h5py.File(measurement_path, 'r') as f:
    # Load raw signal data (shape: T, R, F_time)
    u_raw = f['/measurement/data'][:].squeeze()  # shape: (T, R, F_time)

    print("Raw signal shape:", u_raw.shape)  # e.g. (400, 2, 2048)

    # Apply FFT to convert to frequency domain
    u_fft = rfft(u_raw, axis=-1)  # shape: (T, R, F_freq)

    print("FFT signal shape:", u_fft.shape)  # e.g. (400, 2, 1025)

    # Save each frame
    for t in range(u_fft.shape[0]):
        frame = u_fft[t]  # shape: (2, F_freq)
        frame_path = os.path.join(output_dir, f"frame_{t:03d}.npy")
        np.save(frame_path, frame)

    print(f"Saved {u_fft.shape[0]} frames to {output_dir}")








--------------------------------------
############################################################################################
# Extract Ground Truth Trajectories from all modes " FOR THREE CHANNEL INPUT"
############################################################################################
--------------------------------------
import numpy as np
import h5py
from scipy.ndimage import center_of_mass
import matplotlib.pyplot as plt
import os

# # === Paths ===
system_matrix_file = "system matrix file.mdf"
# signal_dir = "/Signals"  # Update to correct path
signal_dir = "Std,BA, HBA ModeSignals"
output_dir = "/positions"
os.makedirs(output_dir, exist_ok=True)

# === Load system matrix with all 3 channels ===
with h5py.File(system_matrix_file, 'r') as fSM:
    S = fSM['/measurement/data'][:].squeeze()  # shape: (3, F, N)
    isBG = fSM['/measurement/isBackgroundFrame'][:].view(bool)
    S = S[:, :, isBG == False]                 # Remove background frames

    if S.shape[0] != 3:
        raise ValueError("System matrix does not contain 3 channels. Cannot proceed.")

    Nx, Ny, Nz = [int(i) for i in fSM['/calibration/size'][:]]
    N = S.shape[2]
    F_system = S.shape[1]
    S = S.reshape(3 * F_system, N)

print(f" System matrix loaded: shape = {S.shape} (3 channels, {F_system} frequencies)")

# === Kaczmarz solver ===
def kaczmarzReg(A, b, iterations=3, lambd=1e-3):
    M, N = A.shape
    x = np.zeros(N, dtype=np.complex128)
    residual = np.zeros(M, dtype=np.complex128)
    energy = np.linalg.norm(A, axis=1)
    for _ in range(iterations):
        for k in range(M):
            if energy[k] > 0:
                beta = (b[k] - A[k, :].dot(x) - np.sqrt(lambd) * residual[k]) / (energy[k]**2 + lambd)
                x += beta * A[k, :].conj()
                residual[k] += np.sqrt(lambd) * beta
    return x


# === Process signal frames ===
positions = []
files = sorted([f for f in os.listdir(signal_dir) if f.endswith('.npy')])
frames_saved = 0

for i, fname in enumerate(files):
    path = os.path.join(signal_dir, fname)
    u = np.load(path)  # shape: (3, F)
    if u.shape[0] != 3 or u.shape[1] < F_system:
        print(f" Frame {i:03d} skipped: signal shape = {u.shape}, expected (3, ≥{F_system})")
        continue

    u = u[:, :F_system]         # shape: (3, F_system)
    u = u.reshape(-1)           # shape: (3 * F_system)

    if S.shape[0] != u.shape[0]:
        print(f"Shape mismatch on frame {i:03d}: S={S.shape}, u={u.shape}")
        continue

    # --- Reconstruct ---
    c = kaczmarzReg(S, u, iterations=3, lambd=1e-3)
    c_3d = c.reshape(Nx, Ny, Nz).real

    # # --- Save volume ---
    # np.save(os.path.join(output_dir, f"frame_{i:03d}_recon.npy"), c_3d)

    # --- Extract & save catheter tip position ---
    pos = center_of_mass(c_3d)
    positions.append(pos)
    np.save(os.path.join(output_dir, f"frame_{i:03d}_tip.npy"), np.array(pos))

    


# === Save full trajectory ===
if positions:
    positions = np.array(positions)
    np.save(os.path.join(output_dir, "trajectory_gt.npy"), positions)
    print(f"\n Done: {frames_saved} frames processed and saved in '{output_dir}'")
else:
    print("No frames processed successfully.")
