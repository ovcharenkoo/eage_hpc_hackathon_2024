import numpy as np
from scipy.signal import correlate
from scipy.signal import welch, csd

def frequency_weighted_similarity_with_phase(t, p, dt, fmax):
    """
    Compute a score representing the similarity between two seismic shot gathers
    in terms of both amplitude and phase content, with more emphasis on low frequencies.

    Parameters:
    t (np.ndarray): Target shot gather [nrec, nt]
    p (np.ndarray): Predicted shot gather [nrec, nt]
    dt (float): Time sampling interval in seconds
    fmax (float): Maximum frequency (Hz) for comparison

    Returns:
    float: Similarity score between 0 (identical) and 1 (completely different)
    """

    # Perform FFT on both the target and predicted shot gathers along the time axis (axis=1)
    T_f = np.fft.rfft(t, axis=1)
    P_f = np.fft.rfft(p, axis=1)
    
    # Get the corresponding frequency axis
    nt = t.shape[1]
    freqs = np.fft.rfftfreq(nt, d=dt)

    # Filter frequencies up to fmax
    valid_freqs = freqs <= fmax
    T_f = T_f[:, valid_freqs]
    P_f = P_f[:, valid_freqs]
    freqs = freqs[valid_freqs]

    # Compute the amplitude spectra
    T_amp = np.abs(T_f)
    P_amp = np.abs(P_f)

    # Compute the phase spectra
    T_phase = np.angle(T_f)
    P_phase = np.angle(P_f)

    # Compute the relative amplitude difference
    epsilon = 1e-6
    relative_amp_diff = np.abs(T_amp - P_amp) / (T_amp + epsilon)

    # Compute the phase difference
    phase_diff = np.abs(T_phase - P_phase)

    # Apply weighting: inverse of frequency gives more weight to lower frequencies
    weights = 1 / (freqs + epsilon)

    # Compute the weighted sum of relative amplitude and phase differences
    weighted_amp_diff = np.sum(relative_amp_diff * weights, axis=1)
    weighted_phase_diff = np.sum(phase_diff * weights, axis=1)

    # Combine amplitude and phase differences (you can adjust the relative weight of amplitude vs phase if necessary)
    combined_diff = (weighted_amp_diff + weighted_phase_diff) / 2

    # Normalize by the sum of weights to ensure the score is between 0 and 1
    normalization_factor = np.sum(weights)
    normalized_diff = combined_diff / normalization_factor

    # Average over all receivers to get the final score
    score = np.mean(normalized_diff)

    return score


def frequency_weighted_amplitude_difference(t, p, dt, fmax):
    """
    Compute a score representing the amplitude difference between two seismic shot gathers
    with more emphasis on low frequencies.

    Parameters:
    t (np.ndarray): Target shot gather [nrec, nt]
    p (np.ndarray): Predicted shot gather [nrec, nt]
    dt (float): Time sampling interval in seconds
    fmax (float): Maximum frequency (Hz) for comparison

    Returns:
    float: Amplitude difference score between 0 (identical) and 1 (completely different)
    """

    # Perform FFT on both the target and predicted shot gathers along the time axis (axis=1)
    T_f = np.fft.rfft(t, axis=1)
    P_f = np.fft.rfft(p, axis=1)
    
    # Get the corresponding frequency axis
    nt = t.shape[1]
    freqs = np.fft.rfftfreq(nt, d=dt)

    # Filter frequencies up to fmax
    valid_freqs = freqs <= fmax
    T_f = T_f[:, valid_freqs]
    P_f = P_f[:, valid_freqs]
    freqs = freqs[valid_freqs]

    # Compute the amplitude spectra
    T_amp = np.abs(T_f)
    P_amp = np.abs(P_f)

    # Compute the relative amplitude difference
    epsilon = 1e-6
    relative_amp_diff = np.abs(T_amp - P_amp) / (T_amp + epsilon)

    # Apply weighting: inverse of frequency gives more weight to lower frequencies
    weights = 1 / (freqs + epsilon)

    # Compute the weighted sum of relative amplitude differences
    weighted_amp_diff = np.sum(relative_amp_diff * weights, axis=1)

    # Normalize by the sum of weights to ensure the score is between 0 and 1
    normalization_factor = np.sum(weights)
    normalized_amp_diff = weighted_amp_diff / normalization_factor

    # Average over all receivers to get the final amplitude difference score
    score = np.mean(normalized_amp_diff)

    return score

def frequency_weighted_phase_difference(t, p, dt, fmax):
    """
    Compute a score representing the phase difference between two seismic shot gathers
    with more emphasis on low frequencies.

    Parameters:
    t (np.ndarray): Target shot gather [nrec, nt]
    p (np.ndarray): Predicted shot gather [nrec, nt]
    dt (float): Time sampling interval in seconds
    fmax (float): Maximum frequency (Hz) for comparison

    Returns:
    float: Phase difference score between 0 (identical) and 1 (completely different)
    """

    # Perform FFT on both the target and predicted shot gathers along the time axis (axis=1)
    T_f = np.fft.rfft(t, axis=1)
    P_f = np.fft.rfft(p, axis=1)
    
    # Get the corresponding frequency axis
    nt = t.shape[1]
    freqs = np.fft.rfftfreq(nt, d=dt)

    # Filter frequencies up to fmax
    valid_freqs = freqs <= fmax
    T_f = T_f[:, valid_freqs]
    P_f = P_f[:, valid_freqs]
    freqs = freqs[valid_freqs]

    # Compute the phase spectra
    T_phase = np.angle(T_f)
    P_phase = np.angle(P_f)

    # Compute the phase difference
    phase_diff = np.abs(T_phase - P_phase)

    # Apply weighting: inverse of frequency gives more weight to lower frequencies
    weights = 1 / (freqs + 1e-6)

    # Compute the weighted sum of phase differences
    weighted_phase_diff = np.sum(phase_diff * weights, axis=1)

    # Normalize by the sum of weights to ensure the score is between 0 and 1
    normalization_factor = np.sum(weights)
    normalized_phase_diff = weighted_phase_diff / normalization_factor

    # Average over all receivers to get the final phase difference score
    score = np.mean(normalized_phase_diff)

    return score


def cross_correlation_similarity(t, p, dt, fmax, epsilon=1e-10):
    """
    Compute a similarity score between two seismic shot gathers using cross-correlation.
    This method compares the waveforms by calculating their degree of alignment.

    Parameters:
    t (np.ndarray): Target shot gather [nrec, nt]
    p (np.ndarray): Predicted shot gather [nrec, nt]
    dt (float): Time sampling interval in seconds
    fmax (float): Maximum frequency (Hz) for comparison (not used in cross-correlation)
    epsilon (float): Small value to prevent division by zero

    Returns:
    float: Similarity score between -1 (completely different) and 1 (identical)
    """

    nrec, nt = t.shape
    similarity_scores = []

    # Iterate over each receiver
    for i in range(nrec):
        # Compute energy (sum of squares) for both target and predicted signals
        t_energy = np.sum(t[i]**2) + epsilon  # Add epsilon to avoid division by zero
        p_energy = np.sum(p[i]**2) + epsilon  # Add epsilon to avoid division by zero

        # Cross-correlate the target and predicted signals for the i-th receiver
        corr = correlate(t[i], p[i], mode='full', method='auto')

        # Normalize the cross-correlation
        corr /= np.sqrt(t_energy * p_energy)

        # Find the maximum value of the normalized cross-correlation
        max_corr = np.max(np.abs(corr))

        # Append the maximum correlation as the similarity score for this receiver
        similarity_scores.append(max_corr)

    # Average over all receivers to get the final similarity score
    final_score = np.mean(similarity_scores)

    return final_score



def normalized_rmse_similarity(t, p, dt, fmax, epsilon=1e-10):
    """
    Compute a similarity score between two seismic shot gathers using Normalized Root Mean Square Error (NRMSE).
    This method captures both amplitude and phase differences.

    Parameters:
    t (np.ndarray): Target shot gather [nrec, nt]
    p (np.ndarray): Predicted shot gather [nrec, nt]
    dt (float): Time sampling interval in seconds
    fmax (float): Maximum frequency (Hz) for comparison (not used in NRMSE)
    epsilon (float): Small value to prevent division by zero

    Returns:
    float: Similarity score between 0 (identical) and 1 (completely different)
    """

    # Calculate the difference (error) between the target and predicted shots
    error = t - p

    # Compute the Root Mean Square Error (RMSE)
    rmse = np.sqrt(np.mean(error**2, axis=1))

    # Calculate the range of the target shot gather (max - min)
    t_min = np.min(t, axis=1)
    t_max = np.max(t, axis=1)
    t_range = t_max - t_min + epsilon  # Add epsilon to prevent division by zero

    # Normalize RMSE by the range of the target shot gather
    nrmse = rmse / t_range

    # Average over all receivers to get the final NRMSE score
    final_score = np.mean(nrmse)

    return final_score

import numpy as np
from sklearn.metrics import mutual_info_score

def mutual_information_similarity(t, p, dt, fmax, bins=30):
    """
    Compute a similarity score between two seismic shot gathers using Mutual Information (MI).
    This method captures both linear and non-linear relationships between the signals.

    Parameters:
    t (np.ndarray): Target shot gather [nrec, nt]
    p (np.ndarray): Predicted shot gather [nrec, nt]
    dt (float): Time sampling interval in seconds
    fmax (float): Maximum frequency (Hz) for comparison (not used in MI)
    bins (int): Number of bins to use for histogram estimation

    Returns:
    float: Similarity score where higher values indicate stronger statistical dependence
    """

    nrec, nt = t.shape
    mi_scores = []

    # Iterate over each receiver
    for i in range(nrec):
        # Flatten the target and predicted signals to 1D arrays
        t_flat = t[i].flatten()
        p_flat = p[i].flatten()

        # Compute the 2D histogram to estimate the joint probability distribution
        hist_2d, x_edges, y_edges = np.histogram2d(t_flat, p_flat, bins=bins)

        # Normalize the histogram to get the joint probability distribution
        pxy = hist_2d / float(np.sum(hist_2d))

        # Compute the marginal probabilities
        px = np.sum(pxy, axis=1)
        py = np.sum(pxy, axis=0)

        # Calculate the Mutual Information (MI)
        mi = np.nansum(pxy * np.log(pxy / (px[:, None] * py[None, :] + 1e-10) + 1e-10))

        # Append the MI score for this receiver
        mi_scores.append(mi)

    # Average over all receivers to get the final MI score
    final_score = np.mean(mi_scores)

    return final_score


def coherence_similarity(t, p, dt, fmax, nperseg=256):
    """
    Compute a similarity score between two seismic shot gathers using Coherence.
    Coherence measures the linear relationship between two signals in the frequency domain.

    Parameters:
    t (np.ndarray): Target shot gather [nrec, nt]
    p (np.ndarray): Predicted shot gather [nrec, nt]
    dt (float): Time sampling interval in seconds
    fmax (float): Maximum frequency (Hz) for comparison
    nperseg (int): Length of each segment for the FFT calculation (default is 256)

    Returns:
    float: Similarity score between 0 (completely different) and 1 (identical)
    """
    
    nrec, nt = t.shape
    fs = 1.0 / dt  # Sampling frequency
    coherence_scores = []

    # Iterate over each receiver
    for i in range(nrec):
        # Compute power spectral densities and cross power spectral density
        f, Pxx = welch(t[i], fs=fs, nperseg=nperseg)
        _, Pyy = welch(p[i], fs=fs, nperseg=nperseg)
        _, Pxy = csd(t[i], p[i], fs=fs, nperseg=nperseg)

        # Calculate coherence (|Pxy|^2 / (Pxx * Pyy))
        coherence = np.abs(Pxy)**2 / ((Pxx * Pyy) + 1e-10)

        # Limit to frequencies up to fmax
        fmax_idx = np.where(f <= fmax)[0]
        coherence_in_band = coherence[fmax_idx]

        # Average coherence within the frequency band of interest
        avg_coherence = np.mean(coherence_in_band)

        # Append the coherence score for this receiver
        coherence_scores.append(avg_coherence)

    # Average over all receivers to get the final coherence score
    final_score = np.mean(coherence_scores)
    
    # Ensure that the score is normalized between 0 and 1
    final_score = np.clip(final_score, 0, 1)

    return final_score


# score_amp = frequency_weighted_amplitude_difference(shot, 1.1 * shot, dt, fmax=vfmax)
# score_phase = frequency_weighted_phase_difference(shot, 1.1 * shot, dt, fmax=vfmax)
# print(score_amp, score_phase)

# frequency_weighted_similarity_with_phase(shot, 1.1 * shot, dt, fmax=vfmax)