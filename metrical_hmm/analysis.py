"""
Analysis utilities for Metrical HMM belief trajectories.

This module provides functions for analyzing the geometric and
statistical properties of belief trajectories.
"""

import numpy as np
from typing import Tuple, List, Optional
from numpy.typing import NDArray


def belief_entropy(beliefs: NDArray) -> NDArray:
    """
    Compute Shannon entropy of belief distributions over time.

    Entropy measures uncertainty in the belief state. Low entropy means
    high confidence about the hidden state; high entropy means uncertainty.

    Parameters
    ----------
    beliefs : NDArray of shape (T, n_states)
        Belief distributions over time.

    Returns
    -------
    entropy : NDArray of shape (T,)
        Entropy at each time step (in nats).
    """
    # Avoid log(0) by adding small epsilon
    eps = 1e-12
    beliefs_safe = np.clip(beliefs, eps, 1.0)
    return -np.sum(beliefs_safe * np.log(beliefs_safe), axis=1)


def max_entropy(n_states: int) -> float:
    """Maximum possible entropy for n_states (uniform distribution)."""
    return np.log(n_states)


def normalized_entropy(beliefs: NDArray) -> NDArray:
    """
    Compute entropy normalized to [0, 1] range.

    0 = pure state (no uncertainty)
    1 = uniform distribution (maximum uncertainty)
    """
    H = belief_entropy(beliefs)
    H_max = max_entropy(beliefs.shape[1])
    return H / H_max


def belief_velocity(beliefs: NDArray) -> NDArray:
    """
    Compute the "velocity" of belief change over time.

    This measures how fast the belief distribution is changing,
    using L2 norm of the difference between consecutive beliefs.

    Parameters
    ----------
    beliefs : NDArray of shape (T, n_states)
        Belief distributions over time.

    Returns
    -------
    velocity : NDArray of shape (T-1,)
        L2 distance between consecutive belief states.
    """
    diffs = np.diff(beliefs, axis=0)
    return np.linalg.norm(diffs, axis=1)


def belief_acceleration(beliefs: NDArray) -> NDArray:
    """
    Compute "acceleration" (second derivative) of belief trajectory.

    Parameters
    ----------
    beliefs : NDArray of shape (T, n_states)

    Returns
    -------
    acceleration : NDArray of shape (T-2,)
    """
    velocity = belief_velocity(beliefs)
    return np.diff(velocity)


def detect_resets(beliefs: NDArray, w1_threshold: float = 0.5) -> NDArray:
    """
    Detect line boundary resets where belief concentrates on W1.

    Parameters
    ----------
    beliefs : NDArray of shape (T, 4)
        Belief distributions over time.
    w1_threshold : float
        Threshold for P(W1) to be considered a reset.

    Returns
    -------
    reset_indices : NDArray
        Indices where resets occur.
    """
    w1_prob = beliefs[:, 0]  # W1 is state 0
    peaks = (w1_prob[1:-1] > w1_prob[:-2]) & (w1_prob[1:-1] > w1_prob[2:])
    peak_indices = np.where(peaks)[0] + 1

    # Filter by threshold
    reset_indices = peak_indices[w1_prob[peak_indices] > w1_threshold]
    return reset_indices


def cycle_period(beliefs: NDArray, state_idx: int = 0) -> Tuple[float, NDArray]:
    """
    Estimate the cycle period from belief oscillations.

    Uses autocorrelation of a specific state's probability to find
    the dominant period.

    Parameters
    ----------
    beliefs : NDArray of shape (T, n_states)
        Belief distributions over time.
    state_idx : int
        Which state to use for period estimation.

    Returns
    -------
    period : float
        Estimated cycle period in time steps.
    autocorr : NDArray
        Full autocorrelation function.
    """
    signal = beliefs[:, state_idx]
    signal = signal - signal.mean()

    # Compute autocorrelation
    n = len(signal)
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[n-1:]  # Keep positive lags
    autocorr = autocorr / autocorr[0]  # Normalize

    # Find first peak after lag 0
    peaks = (autocorr[1:-1] > autocorr[:-2]) & (autocorr[1:-1] > autocorr[2:])
    peak_indices = np.where(peaks)[0] + 1

    if len(peak_indices) > 0:
        period = peak_indices[0]
    else:
        period = float('nan')

    return period, autocorr


def trajectory_length(beliefs: NDArray) -> float:
    """
    Compute total arc length of belief trajectory through simplex.

    Parameters
    ----------
    beliefs : NDArray of shape (T, n_states)

    Returns
    -------
    length : float
        Total L2 arc length.
    """
    return belief_velocity(beliefs).sum()


def trajectory_curvature(beliefs: NDArray) -> NDArray:
    """
    Compute local curvature of belief trajectory.

    Higher curvature indicates sharper turns (e.g., at resets).

    Parameters
    ----------
    beliefs : NDArray of shape (T, n_states)

    Returns
    -------
    curvature : NDArray of shape (T-2,)
        Local curvature at each interior point.
    """
    # Use discrete curvature formula
    v1 = np.diff(beliefs[:-1], axis=0)  # T-2 vectors
    v2 = np.diff(beliefs[1:], axis=0)   # T-2 vectors

    # Curvature ~ angle between consecutive velocity vectors
    # cos(theta) = (v1 . v2) / (|v1| |v2|)
    dot = np.sum(v1 * v2, axis=1)
    norm1 = np.linalg.norm(v1, axis=1)
    norm2 = np.linalg.norm(v2, axis=1)

    # Avoid division by zero
    eps = 1e-12
    cos_theta = dot / (norm1 * norm2 + eps)
    cos_theta = np.clip(cos_theta, -1, 1)

    # Curvature approximation (angle change per unit length)
    theta = np.arccos(cos_theta)
    avg_step = (norm1 + norm2) / 2
    curvature = theta / (avg_step + eps)

    return curvature


def state_dwell_times(states: NDArray) -> dict:
    """
    Compute statistics on how long the HMM stays in each state.

    Parameters
    ----------
    states : NDArray of shape (T,)
        Hidden state sequence.

    Returns
    -------
    stats : dict
        Dictionary with 'mean', 'std', and 'counts' for each state.
    """
    from collections import defaultdict

    n_states = int(states.max()) + 1
    dwell_times = defaultdict(list)

    current_state = states[0]
    current_dwell = 1

    for s in states[1:]:
        if s == current_state:
            current_dwell += 1
        else:
            dwell_times[current_state].append(current_dwell)
            current_state = s
            current_dwell = 1

    # Don't forget the last run
    dwell_times[current_state].append(current_dwell)

    stats = {}
    for state in range(n_states):
        times = dwell_times[state]
        if len(times) > 0:
            stats[state] = {
                'mean': np.mean(times),
                'std': np.std(times),
                'count': len(times),
                'times': times
            }
        else:
            stats[state] = {'mean': 0, 'std': 0, 'count': 0, 'times': []}

    return stats


def compare_trajectories(
    beliefs1: NDArray,
    beliefs2: NDArray
) -> dict:
    """
    Compare two belief trajectories using various metrics.

    Parameters
    ----------
    beliefs1, beliefs2 : NDArray of shape (T, n_states)
        Two belief trajectories to compare.

    Returns
    -------
    metrics : dict
        Dictionary of comparison metrics.
    """
    # Ensure same length
    min_len = min(len(beliefs1), len(beliefs2))
    b1 = beliefs1[:min_len]
    b2 = beliefs2[:min_len]

    # Point-wise distances
    distances = np.linalg.norm(b1 - b2, axis=1)

    # KL divergence (symmetrized)
    eps = 1e-12
    kl_12 = np.sum(b1 * np.log((b1 + eps) / (b2 + eps)), axis=1)
    kl_21 = np.sum(b2 * np.log((b2 + eps) / (b1 + eps)), axis=1)
    js_div = (kl_12 + kl_21) / 2  # Jensen-Shannon

    return {
        'l2_distance_mean': distances.mean(),
        'l2_distance_max': distances.max(),
        'js_divergence_mean': js_div.mean(),
        'js_divergence_max': js_div.max(),
        'correlation': np.corrcoef(b1.flatten(), b2.flatten())[0, 1]
    }


def summary_statistics(beliefs: NDArray, states: NDArray = None) -> dict:
    """
    Compute comprehensive summary statistics for a belief trajectory.

    Parameters
    ----------
    beliefs : NDArray of shape (T, 4)
        Belief states over time.
    states : NDArray of shape (T,), optional
        True hidden states (if available).

    Returns
    -------
    stats : dict
        Dictionary of summary statistics.
    """
    stats = {
        'length': len(beliefs),
        'mean_entropy': belief_entropy(beliefs).mean(),
        'std_entropy': belief_entropy(beliefs).std(),
        'trajectory_length': trajectory_length(beliefs),
        'mean_velocity': belief_velocity(beliefs).mean(),
    }

    # Cycle period estimation
    period, _ = cycle_period(beliefs)
    stats['estimated_period'] = period

    # Reset detection
    resets = detect_resets(beliefs)
    stats['num_resets'] = len(resets)
    if len(resets) > 1:
        stats['mean_reset_interval'] = np.diff(resets).mean()

    # State marginal statistics
    for i, name in enumerate(['W1', 'S1', 'W2', 'S2']):
        stats[f'mean_p_{name}'] = beliefs[:, i].mean()

    if states is not None:
        dwell = state_dwell_times(states)
        for i, name in enumerate(['W1', 'S1', 'W2', 'S2']):
            if i in dwell:
                stats[f'{name}_mean_dwell'] = dwell[i]['mean']

    return stats
