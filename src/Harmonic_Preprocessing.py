import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

class HarmonicMPIPreprocessor:
    """
    Algorithm 1: Harmonic-Based MPI Signal Preprocessing
    """
    def __init__(self, 
                 fs: float,
                 fx: float, 
                 fy: float,
                 fz: float,
                 harmonic_range: Tuple[int, int],
                 bandwidth: float = 10.0):
        """
        Initialize the preprocessor.
        
        Args:
            fs: Sampling frequency [Hz]
            fx, fy, fz: Fundamental frequencies for each axis [Hz]
            harmonic_range: [h1, h2] range of harmonics to keep
            bandwidth: Delta f for frequency selection [Hz]
        """
        self.fs = fs
        self.fundamentals = np.array([fx, fy, fz])
        self.h1, self.h2 = harmonic_range
        self.bandwidth = bandwidth
        
        # Initialize Min-Max scaler parameters
        self.X_min = None
        self.X_max = None
        self.P_min = None
        self.P_max = None
        
    def _compute_fft(self, signal: np.ndarray) -> np.ndarray:
        """Compute FFT of the signal."""
        return np.fft.fft(signal, axis=-1)
    
    def _create_harmonic_mask(self, 
                             n_samples: int, 
                             fundamental: float) -> np.ndarray:
        """
        Create frequency mask to keep only harmonic components.
        keep |f - h*fk| <= Δf, ∀h ∈ [h1, h2]
        
        Args:
            n_samples: Number of frequency bins
            fundamental: Fundamental frequency for this axis
            
        Returns:
            Binary mask of shape (n_samples,)
        """
        # Compute frequency bins
        freqs = np.fft.fftfreq(n_samples, d=1/self.fs)
        mask = np.zeros(n_samples, dtype=bool)
        
        # Keep frequencies within bandwidth of each harmonic
        for h in range(self.h1, self.h2 + 1):
            center_freq = h * fundamental
            # |f - h*fk| <= Δf
            idx = np.abs(freqs - center_freq) <= self.bandwidth
            mask = mask | idx
            
        return mask
    
    def preprocess_sample(self, 
                         S: np.ndarray, 
                         P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess a single MPI signal sample.
        
        Args:
            S: Complex MPI signals of shape (3, F) for x,y,z axes
            P: True spatial positions of shape (3,)
            
        Returns:
            X: Filtered signal features (real and imaginary parts)
            P: Position
        """
        # Step 1: Compute frequency-domain signal
        # Fi ← F{Si}
        F_signal = self._compute_fft(S)
        
        X_filtered = []
        
        # Step 2: For each axis k ∈ {x,y,z}
        for axis_idx, fundamental in enumerate(self.fundamentals):
            # Step 3: Create harmonic mask
            # keep |f - h*fk| ≤ Δf, ∀h ∈ [h1, h2]
            mask = self._create_harmonic_mask(S.shape[-1], fundamental)
            
            # Step 4: Apply mask to frequency domain signal
            # Ffiltered_i[k] ← Fi[k] ◦ maskk
            filtered_fft = F_signal[axis_idx] * mask
            
            # Step 5: Extract real and imaginary parts
            # Xi ← [Re(Ffiltered_i), Im(Ffiltered_i)]
            real_part = np.real(filtered_fft)
            imag_part = np.imag(filtered_fft)
            
            # Stack real and imaginary features
            X_filtered.extend([real_part, imag_part])
        
        X = np.array(X_filtered)
        
        # Step 6: Store (Xi, Pi)
        return X, P
    
    def fit_scalers(self, 
                   X_data: List[np.ndarray], 
                   P_data: List[np.ndarray]):
        """
        Fit Min-Max scalers on the dataset.
        """
        # Flatten X_data for scaling
        X_flat = np.concatenate([x.flatten() for x in X_data]).reshape(-1, 1)
        
        # Fit scaler for X
        scaler_X = MinMaxScaler()
        scaler_X.fit(X_flat)
        self.X_min = scaler_X.data_min_[0]
        self.X_max = scaler_X.data_max_[0]
        
        # Fit scaler for P
        P_array = np.array(P_data)
        self.P_min = P_array.min(axis=0)
        self.P_max = P_array.max(axis=0)
        
        # Avoid division by zero
        self.P_range = self.P_max - self.P_min
        self.P_range[self.P_range == 0] = 1.0
        
        self.X_range = self.X_max - self.X_min
        if self.X_range == 0:
            self.X_range = 1.0
    
    def normalize(self, 
                  X: np.ndarray, 
                  P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Min-Max normalization.
        
        Returns:
            X_normalized: Normalized signal features
            P_normalized: Normalized positions
        """
        # Step 7: Normalize X
        # Ŝ ← (X - Xmin) / (Xmax - Xmin)
        X_normalized = (X - self.X_min) / self.X_range
        
        # Step 8: Normalize P
        # P˙ ← (P - Pmin) / (Pmax - Pmin)
        P_normalized = (P - self.P_min) / self.P_range
        
        return X_normalized, P_normalized
    
    def preprocess_dataset(self, 
                          S_data: List[np.ndarray], 
                          P_data: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Preprocess entire dataset.
        
        Args:
            S_data: List of complex MPI signals (3, F)
            P_data: List of positions (3,)
            
        Returns:
            X_normalized: Normalized signal features
            P_normalized: Normalized positions
            scaler_params: Dictionary of {Xmin, Xmax, Pmin, Pmax}
        """
        X_list = []
        P_list = []
        
        # Process each sample
        print("Processing samples...")
        for idx, (S, P) in enumerate(zip(S_data, P_data)):
            X, P_proc = self.preprocess_sample(S, P)
            X_list.append(X)
            P_list.append(P_proc)
            
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(S_data)} samples")
        
        # Fit Min-Max scalers
        print("Fitting Min-Max scalers...")
        self.fit_scalers(X_list, P_list)
        
        # Normalize all samples
        print("Normalizing data...")
        X_normalized = []
        P_normalized = []
        
        for idx, (X, P) in enumerate(zip(X_list, P_list)):
            X_norm, P_norm = self.normalize(X, P)
            X_normalized.append(X_norm)
            P_normalized.append(P_norm)
            
            if (idx + 1) % 100 == 0:
                print(f"Normalized {idx + 1}/{len(S_data)} samples")
        
        # Stack into arrays
        X_stack = np.stack(X_normalized, axis=0)
        P_stack = np.stack(P_normalized, axis=0)
        
        # Save scaler parameters
        scaler_params = {
            'X_min': self.X_min,
            'X_max': self.X_max,
            'P_min': self.P_min,
            'P_max': self.P_max
        }
        
        print(f"\nPreprocessing complete!")
        print(f"Normalized signal shape: {X_stack.shape}")
        print(f"Normalized positions shape: {P_stack.shape}")
        print(f"Scaler parameters: Xmin={self.X_min:.4f}, Xmax={self.X_max:.4f}")
        print(f"                  Pmin={self.P_min}, Pmax={self.P_max}")
        
        return X_stack, P_stack, scaler_params


def main():
    """
    Example usage of the Harmonic-Based MPI Signal Preprocessing algorithm.
    """
    # Parameters
    fs = 1000.0          # Sampling frequency [Hz]
    fx, fy, fz = 50.0, 60.0, 70.0  # Fundamental frequencies [Hz]
    harmonic_range = (1, 5)  # [h1, h2] harmonic range
    bandwidth = 10.0     # Δf [Hz]
    
    # Generate synthetic dataset for demonstration
    print("Generating synthetic MPI data...")
    np.random.seed(42)
    n_samples = 50
    signal_length = 1024
    n_axes = 3
    
    S_data = []
    P_data = []
    
    for sample_idx in range(n_samples):
        # Time vector
        t = np.linspace(0, signal_length/fs, signal_length, endpoint=False)
        signal = np.zeros((n_axes, signal_length), dtype=complex)
        
        # Random position (true spatial position)
        P = np.random.uniform(-1, 1, 3)
        
        # Generate harmonic signals for each axis
        for axis_idx, f in enumerate([fx, fy, fz]):
            # Generate signal with harmonics
            harmonic_signal = np.zeros(signal_length, dtype=complex)
            
            for h in range(harmonic_range[0], harmonic_range[1] + 1):
                # Amplitude decreases with harmonic order
                amp = 0.5 / h * (1 + 0.1 * np.random.randn())
                harmonic_signal += amp * np.exp(1j * 2 * np.pi * h * f * t)
            
            # Modulate by position
            harmonic_signal *= (1 + 0.3 * P[axis_idx])
            
            # Add noise
            noise = 0.02 * (np.random.randn(signal_length) + 1j * np.random.randn(signal_length))
            signal[axis_idx] = harmonic_signal + noise
        
        S_data.append(signal)
        P_data.append(P)
    
    print(f"Generated {n_samples} samples")
    print(f"Signal shape: {S_data[0].shape}")
    print(f"Position shape: {P_data[0].shape}")
    print()
    
    # Initialize preprocessor
    preprocessor = HarmonicMPIPreprocessor(
        fs=fs,
        fx=fx,
        fy=fy,
        fz=fz,
        harmonic_range=harmonic_range,
        bandwidth=bandwidth
    )
    
    # Preprocess dataset
    X_normalized, P_normalized, scaler_params = preprocessor.preprocess_dataset(
        S_data, P_data
    )
    
    # Display results
    print("\n" + "="*50)
    print("PREPROCESSING RESULTS")
    print("="*50)
    print(f"Normalized signals (Ŝ):")
    print(f"  Shape: {X_normalized.shape}")
    print(f"  Min: {X_normalized.min():.4f}")
    print(f"  Max: {X_normalized.max():.4f}")
    print(f"  Mean: {X_normalized.mean():.4f}")
    print(f"  Std: {X_normalized.std():.4f}")
    print()
    print(f"Normalized positions (P˙):")
    print(f"  Shape: {P_normalized.shape}")
    print(f"  Min: {P_normalized.min(axis=0)}")
    print(f"  Max: {P_normalized.max(axis=0)}")
    print(f"  Mean: {P_normalized.mean(axis=0)}")
    print()
    print(f"Scaler parameters ({Xmin, Xmax, Pmin, Pmax}):")
    for key, value in scaler_params.items():
        print(f"  {key}: {value}")
    
    # Verify normalization
    print("\n" + "="*50)
    print("VERIFICATION")
    print("="*50)
    print(f"X is normalized: {np.allclose(X_normalized.min(), 0.0, atol=1e-6) and np.allclose(X_normalized.max(), 1.0, atol=1e-6)}")
    print(f"P is normalized: {np.allclose(P_normalized.min(axis=0), 0.0, atol=1e-6) and np.allclose(P_normalized.max(axis=0), 1.0, atol=1e-6)}")
    
    # Save normalized dataset
    np.savez('mpi_preprocessed_data.npz',
             X_normalized=X_normalized,
             P_normalized=P_normalized,
             X_min=scaler_params['X_min'],
             X_max=scaler_params['X_max'],
             P_min=scaler_params['P_min'],
             P_max=scaler_params['P_max'])
    
    print("\nSaved normalized dataset to 'mpi_preprocessed_data.npz'")
    print("\nPreprocessing complete!")


if __name__ == "__main__":
    main()
