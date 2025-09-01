"""
EEMD (Ensemble Empirical Mode Decomposition) implementation for signal denoising.
Implements EEMD decomposition and Sample Entropy calculation.
"""
import numpy as np
import pandas as pd
from scipy import interpolate
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class EMD:
    """Empirical Mode Decomposition implementation."""
    
    def __init__(self, max_imfs: int = 10, max_iterations: int = 1000, tolerance: float = 0.2):
        self.max_imfs = max_imfs
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def _find_extrema(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Find local maxima and minima."""
        diff = np.diff(signal)
        diff_sign = np.sign(diff)
        
        # Find points where sign changes
        sign_changes = np.diff(diff_sign)
        
        # Maxima: positive to negative transition
        max_indices = np.where(sign_changes < 0)[0] + 1
        max_values = signal[max_indices]
        
        # Minima: negative to positive transition  
        min_indices = np.where(sign_changes > 0)[0] + 1
        min_values = signal[min_indices]
        
        return max_indices, max_values, min_indices, min_values
    
    def _create_envelope(self, indices: np.ndarray, values: np.ndarray, signal_length: int) -> np.ndarray:
        """Create envelope using cubic spline interpolation."""
        if len(indices) < 2:
            return np.zeros(signal_length)
        
        # Add boundary points
        indices_extended = np.concatenate([[0], indices, [signal_length - 1]])
        values_extended = np.concatenate([[values[0]], values, [values[-1]]])
        
        # Create cubic spline
        try:
            spline = interpolate.CubicSpline(indices_extended, values_extended, bc_type='natural')
            x = np.arange(signal_length)
            envelope = spline(x)
        except:
            # Fallback to linear interpolation
            envelope = np.interp(np.arange(signal_length), indices_extended, values_extended)
        
        return envelope
    
    def _is_imf(self, signal: np.ndarray) -> bool:
        """Check if signal satisfies IMF conditions."""
        max_indices, _, min_indices, _ = self._find_extrema(signal)
        
        # Check number of extrema
        n_max = len(max_indices)
        n_min = len(min_indices)
        n_extrema = n_max + n_min
        
        # Check zero crossings
        zero_crossings = np.sum(np.diff(np.sign(signal)) != 0)
        
        # IMF conditions:
        # 1. Number of extrema and zero crossings differ by at most 1
        # 2. Mean of upper and lower envelopes is close to zero
        condition1 = abs(n_extrema - zero_crossings) <= 1
        
        if n_max < 2 or n_min < 2:
            return condition1
        
        # Calculate mean envelope
        upper_envelope = self._create_envelope(max_indices, signal[max_indices], len(signal))
        lower_envelope = self._create_envelope(min_indices, signal[min_indices], len(signal))
        mean_envelope = (upper_envelope + lower_envelope) / 2
        
        # Check if mean is close to zero
        envelope_criterion = np.mean(np.abs(mean_envelope)) < self.tolerance * np.std(signal)
        
        return condition1 and envelope_criterion
    
    def _extract_imf(self, signal: np.ndarray) -> np.ndarray:
        """Extract a single IMF from the signal."""
        h = signal.copy()
        
        for _ in range(self.max_iterations):
            max_indices, max_values, min_indices, min_values = self._find_extrema(h)
            
            if len(max_indices) < 2 or len(min_indices) < 2:
                break
            
            # Create envelopes
            upper_envelope = self._create_envelope(max_indices, max_values, len(h))
            lower_envelope = self._create_envelope(min_indices, min_values, len(h))
            
            # Calculate mean envelope
            mean_envelope = (upper_envelope + lower_envelope) / 2
            
            # Subtract mean from signal
            h_new = h - mean_envelope
            
            # Check stopping criterion
            if self._is_imf(h_new):
                return h_new
            
            h = h_new
        
        return h
    
    def decompose(self, signal: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """Decompose signal into IMFs and residue."""
        imfs = []
        residue = signal.copy()
        
        for i in range(self.max_imfs):
            # Extract IMF
            imf = self._extract_imf(residue)
            
            # Check if we can extract more IMFs
            max_indices, _, min_indices, _ = self._find_extrema(residue - imf)
            if len(max_indices) < 2 or len(min_indices) < 2:
                break
            
            imfs.append(imf)
            residue = residue - imf
            
            # Stop if residue is monotonic
            if len(self._find_extrema(residue)[0]) < 2:
                break
        
        return imfs, residue


class EEMD:
    """Ensemble Empirical Mode Decomposition implementation."""
    
    def __init__(self, n_ensembles: int = 100, noise_scale: float = 0.2, max_imfs: int = 10):
        self.n_ensembles = n_ensembles
        self.noise_scale = noise_scale
        self.max_imfs = max_imfs
        self.emd = EMD(max_imfs=max_imfs)
    
    def decompose(self, signal: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """Perform EEMD decomposition."""
        signal = np.array(signal, dtype=np.float64)
        
        # Initialize ensemble IMFs
        ensemble_imfs = [[] for _ in range(self.max_imfs)]
        ensemble_residues = []
        
        # Generate noise realizations and decompose
        for i in range(self.n_ensembles):
            # Add white noise
            noise_std = self.noise_scale * np.std(signal)
            noise = np.random.normal(0, noise_std, len(signal))
            noisy_signal = signal + noise
            
            # Decompose noisy signal
            imfs, residue = self.emd.decompose(noisy_signal)
            
            # Store IMFs
            for j, imf in enumerate(imfs):
                if j < len(ensemble_imfs):
                    ensemble_imfs[j].append(imf)
            
            ensemble_residues.append(residue)
        
        # Average ensemble IMFs
        averaged_imfs = []
        for imf_ensemble in ensemble_imfs:
            if imf_ensemble:  # If this IMF level has data
                # Ensure all IMFs have the same length
                min_length = min(len(imf) for imf in imf_ensemble)
                truncated_imfs = [imf[:min_length] for imf in imf_ensemble]
                averaged_imf = np.mean(truncated_imfs, axis=0)
                averaged_imfs.append(averaged_imf)
        
        # Average residue
        min_length = min(len(res) for res in ensemble_residues)
        truncated_residues = [res[:min_length] for res in ensemble_residues]
        averaged_residue = np.mean(truncated_residues, axis=0)
        
        return averaged_imfs, averaged_residue


class SampleEntropy:
    """Sample Entropy calculation for time series complexity measure."""
    
    def __init__(self, m: int = 2, r: float = None):
        self.m = m  # Pattern length
        self.r = r  # Tolerance for matching
    
    def _maxdist(self, xi: np.ndarray, xj: np.ndarray) -> float:
        """Calculate maximum distance between two patterns."""
        return max([abs(ua - va) for ua, va in zip(xi, xj)])
    
    def calculate(self, signal: np.ndarray) -> float:
        """Calculate Sample Entropy of a signal."""
        signal = np.array(signal, dtype=np.float64)
        N = len(signal)
        
        # Set tolerance if not provided
        if self.r is None:
            self.r = 0.2 * np.std(signal)
        
        def _phi(m):
            patterns = []
            for i in range(N - m + 1):
                patterns.append(signal[i:i + m])
            
            C = 0
            for i in range(len(patterns)):
                template = patterns[i]
                for j in range(len(patterns)):
                    if i != j and self._maxdist(template, patterns[j]) <= self.r:
                        C += 1
            
            if len(patterns) == 0:
                return 0
            return C / len(patterns)
        
        phi_m = _phi(self.m)
        phi_m1 = _phi(self.m + 1)
        
        if phi_m == 0 or phi_m1 == 0:
            return 0
        
        return -np.log(phi_m1 / phi_m)


class EEMDDenoiser:
    """EEMD-based signal denoising following the paper's methodology."""
    
    def __init__(self, n_ensembles: int = 20, noise_scale: float = 0.15):
        self.eemd = EEMD(n_ensembles=n_ensembles, noise_scale=noise_scale, max_imfs=8)
        self.sample_entropy = SampleEntropy()
    
    def denoise(self, signal: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Denoise signal using EEMD method from the paper.
        
        Args:
            signal: Input signal to denoise
            
        Returns:
            Tuple of (denoised_signal, metadata)
        """
        # Perform EEMD decomposition
        print("Performing EEMD decomposition...")
        imfs, residue = self.eemd.decompose(signal)
        
        if not imfs:
            print("Warning: No IMFs extracted, returning original signal")
            return signal, {'n_imfs': 0, 'max_entropy_imf': -1}
        
        # Calculate Sample Entropy for each IMF
        print(f"Calculating Sample Entropy for {len(imfs)} IMFs...")
        entropies = []
        for i, imf in enumerate(imfs):
            entropy = self.sample_entropy.calculate(imf)
            entropies.append(entropy)
            print(f"IMF {i+1}: Sample Entropy = {entropy:.4f}")
        
        # Find IMF with maximum Sample Entropy (most complex/noisy)
        max_entropy_idx = np.argmax(entropies)
        max_entropy_imf = imfs[max_entropy_idx]
        
        print(f"IMF {max_entropy_idx + 1} has highest Sample Entropy ({entropies[max_entropy_idx]:.4f})")
        
        # Reconstruct signal without the most complex IMF
        # x_filtered(t) = x(t) - IMF_max_SaEn(t)
        denoised_signal = signal - max_entropy_imf[:len(signal)]
        
        metadata = {
            'n_imfs': len(imfs),
            'imf_entropies': entropies,
            'max_entropy_imf': max_entropy_idx,
            'max_entropy_value': entropies[max_entropy_idx],
            'original_signal_length': len(signal),
            'denoised_signal_length': len(denoised_signal)
        }
        
        print(f"Signal denoised: removed IMF {max_entropy_idx + 1}")
        
        return denoised_signal, metadata
    
    def process_price_series(self, prices: pd.Series) -> Tuple[pd.Series, dict]:
        """Process price series and return denoised prices."""
        # Convert to numpy array
        price_values = prices.values
        
        # Apply EEMD denoising
        denoised_values, metadata = self.denoise(price_values)
        
        # Convert back to pandas Series
        denoised_prices = pd.Series(denoised_values, index=prices.index[:len(denoised_values)])
        
        return denoised_prices, metadata


if __name__ == "__main__":
    # Test EEMD denoising with synthetic data
    print("Testing EEMD denoising...")
    
    # Create a test signal
    n_points = 1000
    t = np.linspace(0, 10, n_points)
    
    # Create a composite signal with noise
    clean_signal = np.sin(2 * np.pi * 0.5 * t) + 0.5 * np.sin(2 * np.pi * 2 * t)
    noise = 0.2 * np.random.randn(n_points)
    noisy_signal = clean_signal + noise
    
    print(f"Test signal length: {len(noisy_signal)}")
    
    # Initialize denoiser with reduced parameters for testing
    denoiser = EEMDDenoiser(n_ensembles=20, noise_scale=0.1)
    
    # Denoise the signal
    try:
        denoised_signal, metadata = denoiser.denoise(noisy_signal)
        
        print(f"Denoised signal length: {len(denoised_signal)}")
        print(f"Number of IMFs: {metadata['n_imfs']}")
        print(f"Max entropy IMF: {metadata['max_entropy_imf'] + 1}")
        
        # Show some statistics
        original_std = np.std(noisy_signal)
        denoised_std = np.std(denoised_signal)
        print(f"Original signal std: {original_std:.4f}")
        print(f"Denoised signal std: {denoised_std:.4f}")
        print(f"Noise reduction: {((original_std - denoised_std) / original_std * 100):.2f}%")
        
        print("EEMD denoising test completed successfully!")
        
    except Exception as e:
        print(f"Error during EEMD testing: {e}")
        import traceback
        traceback.print_exc()