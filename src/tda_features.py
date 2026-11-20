import numpy as np
from scipy.spatial.distance import pdist
from ripser import ripser


def delay_embed(signal, tau, d=2):
    """
    Constructs a d-dimensional delay embedding of a 1D time-series. 
    Parameters: 
    - signal: array-like 
        One-dimensional time series.
    - tau: int
        Delay in samples.
    - d: int, default = 2
        Dimension of the embedding

    Returns:
    ndarray 
        ndarray of shape (n_samples - tau*(d-1), 2), d-dimensional delay embedding corresponding to the time-series
    """
    signal = np.asarray(signal)
    
    if signal.ndim != 1:
        raise ValueError("Expected a 1D signal.")

    if not isinstance(tau, (int, np.integer)) or tau <=0:
        raise ValueError("tau must be a positive integer.")
        
    if not isinstance(d, (int, np.integer)) or d <= 1:
        raise ValueError("d must be an integer > 1.")

    n_points = len(signal) - (d-1)*tau

    if n_points <2:
        raise ValueError("Time-series is too short for the chosen parameters.")

    return np.column_stack([
        signal[i*tau : i*tau + n_points] for i in range(d)
    ])



def extract_H1_features(embedding, threshold = 0.1):
    """
    Extract the maximum H1 persistence, the second-largest H1 persistence, the total H1 persistence, the count of H1 features whose persistence
    is above threshold parameter, and the persistence entropy from an embedding.

    Parameters:

    -embedding: array-like
        n-dimensional point-cloud (delay embedding)
    - threshold: float
        value above which H1 persistence is considered prominent

    Returns:
    Dict
        Dictionary containing the extracted H1 features.
    """

    if not isinstance(threshold, (float, int, np.integer)):
        raise ValueError("Threshold must be a number")
    
    embedding = np.asarray(embedding)
    diameter = float(pdist(embedding).max())
    
    if not np.isfinite(diameter) or diameter == 0:
        raise ValueError("Embedding must have finite, nonzero diameter")
    
    H1 = ripser(embedding, maxdim=1)["dgms"][1]
    
    lengths = H1[:, 1] - H1[:, 0]
    lengths = lengths[np.isfinite(lengths) & (lengths/diameter > 1e-10)]
    lengths = np.sort(lengths/diameter)[::-1] #H1 persistences is normilized by parameter and sorted in descending order
    total_length = float(lengths.sum())
    
    entropy = 0.0

    if total_length > 0:
        weights = lengths/total_length
        entropy = float(-(weights * np.log(weights)).sum()) #Shannon Entropy on the H1 features (Persistence Entropy)

    if len(lengths) == 0:
        max_H1, second_H1 = 0.0, 0.0
    else:
        max_H1 = float(lengths[0])
        if len(lengths) < 2:
            second_H1 = 0.0
        else:
            second_H1 = float(lengths[1])

    

    return{
        "max_H1": max_H1,
        "second_H1": second_H1,
        "total_H1_persistence": float(total_length),
        "prominent_H1_features": int((lengths > threshold).sum()),
        "persistence_entropy": float(entropy)
    }