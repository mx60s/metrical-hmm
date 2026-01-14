"""
Core Hidden Markov Model implementation for Metrical HMM.

This module implements an HMM modeling iambic dimeter (da-DUM da-DUM per line)
with 4 states representing metrical positions: W1, S1, W2, S2.
"""

import numpy as np
from typing import Tuple, Optional
from numpy.typing import NDArray


# State indices
W1, S1, W2, S2 = 0, 1, 2, 3
STATE_NAMES = ['W1', 'S1', 'W2', 'S2']


class MetricalHMM:
    """
    Hidden Markov Model for iambic dimeter.

    States represent position within the metrical line:
    - W1: Weak position, foot 1 (unstressed expected)
    - S1: Strong position, foot 1 (stressed expected)
    - W2: Weak position, foot 2 (unstressed expected)
    - S2: Strong position, foot 2 (stressed expected, line ends)

    Parameters
    ----------
    emission_model : object
        Pluggable emission model with methods:
        - n_outputs: int property
        - prob(state, observation) -> float
        - sample(state) -> observation
        - emission_matrix() -> NDArray
    strictness : float, default=0.9
        Controls how strongly transitions follow the canonical cycle.
        Higher values = cleaner cycle, lower = more variation.
        Range: [0.5, 1.0]
    allow_skips : bool, default=True
        If True, allows "skip" transitions (e.g., W1->W2).
        If False, only canonical cycle transitions allowed.
    line_reset_prob : float, default=0.85
        Probability of resetting to W1 after S2 (line boundary).
    """

    def __init__(
        self,
        emission_model,
        strictness: float = 0.9,
        allow_skips: bool = True,
        line_reset_prob: float = 0.85,
    ):
        self.emission_model = emission_model
        self.strictness = np.clip(strictness, 0.5, 1.0)
        self.allow_skips = allow_skips
        self.line_reset_prob = line_reset_prob

        self.n_states = 4
        self.state_names = STATE_NAMES

        # Build matrices
        self.A = self._build_transition_matrix()
        self.B = emission_model.emission_matrix()
        self.pi = self._build_initial_distribution()

        # Validate
        self._validate_matrices()

    def _build_transition_matrix(self) -> NDArray:
        """
        Build the transition matrix based on strictness and skip settings.

        Canonical cycle: W1 -> S1 -> W2 -> S2 -> W1
        """
        A = np.zeros((4, 4))

        if self.allow_skips:
            # With skips: distribute probability among all transitions
            # Main transition gets `strictness`, rest distributed
            noise = (1.0 - self.strictness) / 3.0

            # W1 row: mainly to S1
            A[W1] = [noise, self.strictness, noise, noise]

            # S1 row: mainly to W2
            A[S1] = [noise, noise, self.strictness, noise]

            # W2 row: mainly to S2
            A[W2] = [noise, noise, noise, self.strictness]

            # S2 row: mainly reset to W1
            remaining = 1.0 - self.line_reset_prob
            skip_prob = remaining / 3.0
            A[S2] = [self.line_reset_prob, skip_prob, skip_prob, skip_prob]
        else:
            # Pure cycle: only canonical transitions allowed
            # Self-loop with small probability for variation
            self_loop = 1.0 - self.strictness

            A[W1, W1] = self_loop
            A[W1, S1] = self.strictness

            A[S1, S1] = self_loop
            A[S1, W2] = self.strictness

            A[W2, W2] = self_loop
            A[W2, S2] = self.strictness

            A[S2, S2] = 1.0 - self.line_reset_prob
            A[S2, W1] = self.line_reset_prob

        # Ensure rows sum to 1 (handle floating point)
        A = A / A.sum(axis=1, keepdims=True)
        return A

    def _build_initial_distribution(self) -> NDArray:
        """Build initial state distribution (strong preference for W1)."""
        pi = np.array([0.85, 0.05, 0.05, 0.05])
        return pi / pi.sum()

    def _validate_matrices(self):
        """Validate that matrices are proper probability distributions."""
        assert np.allclose(self.A.sum(axis=1), 1.0), "Transition rows must sum to 1"
        assert np.allclose(self.B.sum(axis=1), 1.0), "Emission rows must sum to 1"
        assert np.allclose(self.pi.sum(), 1.0), "Initial distribution must sum to 1"
        assert np.all(self.A >= 0), "Transition probabilities must be non-negative"
        assert np.all(self.B >= 0), "Emission probabilities must be non-negative"

    def sample(self, n_steps: int, seed: Optional[int] = None) -> Tuple[NDArray, NDArray]:
        """
        Generate a sequence of (hidden_states, observations).

        Parameters
        ----------
        n_steps : int
            Length of sequence to generate.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        states : NDArray of shape (n_steps,)
            Hidden state indices at each time step.
        observations : NDArray of shape (n_steps,)
            Observed emissions at each time step.
        """
        rng = np.random.default_rng(seed)

        states = np.zeros(n_steps, dtype=int)
        observations = np.zeros(n_steps, dtype=int)

        # Initial state
        states[0] = rng.choice(self.n_states, p=self.pi)
        observations[0] = self.emission_model.sample(states[0], rng)

        # Generate sequence
        for t in range(1, n_steps):
            states[t] = rng.choice(self.n_states, p=self.A[states[t-1]])
            observations[t] = self.emission_model.sample(states[t], rng)

        return states, observations

    def forward(self, observations: NDArray) -> NDArray:
        """
        Run forward algorithm, returning normalized belief states.

        The belief state at time t is the posterior distribution over
        hidden states given observations up to time t:
            belief[t] = P(S_t | o_1, o_2, ..., o_t)

        Parameters
        ----------
        observations : NDArray of shape (T,)
            Sequence of observation indices.

        Returns
        -------
        beliefs : NDArray of shape (T, 4)
            Posterior distribution over states at each time step.
        """
        T = len(observations)
        beliefs = np.zeros((T, self.n_states))

        # Initialize: belief[0] proportional to pi * B[:, obs[0]]
        alpha = self.pi * self.B[:, observations[0]]
        alpha = alpha / alpha.sum()
        beliefs[0] = alpha

        for t in range(1, T):
            # Predict: propagate through transition
            alpha_pred = self.A.T @ alpha

            # Update: incorporate observation
            alpha = alpha_pred * self.B[:, observations[t]]
            alpha = alpha / alpha.sum()
            beliefs[t] = alpha

        return beliefs

    def forward_log(self, observations: NDArray) -> NDArray:
        """
        Run forward algorithm in log-space for numerical stability.

        Use this for longer sequences (>100 steps) to avoid underflow.

        Parameters
        ----------
        observations : NDArray of shape (T,)
            Sequence of observation indices.

        Returns
        -------
        beliefs : NDArray of shape (T, 4)
            Posterior distribution over states at each time step.
        """
        T = len(observations)
        beliefs = np.zeros((T, self.n_states))

        # Initialize in log space
        log_alpha = np.log(self.pi + 1e-300) + np.log(self.B[:, observations[0]] + 1e-300)
        log_alpha = log_alpha - np.logaddexp.reduce(log_alpha)  # Normalize
        beliefs[0] = np.exp(log_alpha)

        log_A = np.log(self.A + 1e-300)
        log_B = np.log(self.B + 1e-300)

        for t in range(1, T):
            # Predict: log-sum-exp over previous states
            log_alpha_pred = np.zeros(self.n_states)
            for j in range(self.n_states):
                log_alpha_pred[j] = np.logaddexp.reduce(log_alpha + log_A[:, j])

            # Update: add log-likelihood of observation
            log_alpha = log_alpha_pred + log_B[:, observations[t]]
            log_alpha = log_alpha - np.logaddexp.reduce(log_alpha)  # Normalize
            beliefs[t] = np.exp(log_alpha)

        return beliefs

    def likelihood(self, observations: NDArray, log: bool = False) -> float:
        """
        Compute P(observations) using forward algorithm.

        Parameters
        ----------
        observations : NDArray of shape (T,)
            Sequence of observation indices.
        log : bool, default=False
            If True, return log-likelihood instead.

        Returns
        -------
        likelihood : float
            P(observations) or log P(observations).
        """
        T = len(observations)

        if log:
            # Log-space computation
            log_alpha = np.log(self.pi + 1e-300) + np.log(self.B[:, observations[0]] + 1e-300)
            log_A = np.log(self.A + 1e-300)
            log_B = np.log(self.B + 1e-300)

            for t in range(1, T):
                log_alpha_pred = np.zeros(self.n_states)
                for j in range(self.n_states):
                    log_alpha_pred[j] = np.logaddexp.reduce(log_alpha + log_A[:, j])
                log_alpha = log_alpha_pred + log_B[:, observations[t]]

            return np.logaddexp.reduce(log_alpha)
        else:
            # Standard computation
            alpha = self.pi * self.B[:, observations[0]]

            for t in range(1, T):
                alpha = (self.A.T @ alpha) * self.B[:, observations[t]]

            return alpha.sum()

    def viterbi(self, observations: NDArray) -> Tuple[NDArray, float]:
        """
        Find most likely state sequence using Viterbi algorithm.

        Parameters
        ----------
        observations : NDArray of shape (T,)
            Sequence of observation indices.

        Returns
        -------
        path : NDArray of shape (T,)
            Most likely state sequence.
        log_prob : float
            Log probability of the most likely path.
        """
        T = len(observations)

        # Work in log space
        log_A = np.log(self.A + 1e-300)
        log_B = np.log(self.B + 1e-300)
        log_pi = np.log(self.pi + 1e-300)

        # Viterbi variables
        V = np.zeros((T, self.n_states))  # Max log-prob to reach state at time t
        backptr = np.zeros((T, self.n_states), dtype=int)

        # Initialize
        V[0] = log_pi + log_B[:, observations[0]]

        # Forward pass
        for t in range(1, T):
            for j in range(self.n_states):
                scores = V[t-1] + log_A[:, j]
                backptr[t, j] = np.argmax(scores)
                V[t, j] = scores[backptr[t, j]] + log_B[j, observations[t]]

        # Backtrack
        path = np.zeros(T, dtype=int)
        path[-1] = np.argmax(V[-1])
        log_prob = V[-1, path[-1]]

        for t in range(T - 2, -1, -1):
            path[t] = backptr[t + 1, path[t + 1]]

        return path, log_prob

    def is_ergodic(self, n_power: int = 10) -> bool:
        """
        Check if the HMM is ergodic (all states reachable from all others).

        Parameters
        ----------
        n_power : int, default=10
            Power to raise transition matrix to.

        Returns
        -------
        ergodic : bool
            True if A^n has all positive entries.
        """
        A_n = np.linalg.matrix_power(self.A, n_power)
        return np.all(A_n > 1e-10)

    def stationary_distribution(self) -> NDArray:
        """
        Compute the stationary distribution of the Markov chain.

        Returns
        -------
        stationary : NDArray of shape (4,)
            Stationary distribution (left eigenvector for eigenvalue 1).
        """
        eigenvalues, eigenvectors = np.linalg.eig(self.A.T)

        # Find eigenvector for eigenvalue 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        stationary = stationary / stationary.sum()

        return stationary

    def __repr__(self) -> str:
        return (
            f"MetricalHMM(strictness={self.strictness:.2f}, "
            f"allow_skips={self.allow_skips}, "
            f"line_reset_prob={self.line_reset_prob:.2f})"
        )


class GeneralizedMetricalHMM:
    """
    Generalized Hidden Markov Model for iambic meter with variable feet per line.

    Supports dimeter (2 feet, 4 states), trimeter (3 feet, 6 states),
    tetrameter (4 feet, 8 states), pentameter (5 feet, 10 states), etc.

    States represent position within the metrical line:
    - W1, S1: First foot (weak, strong)
    - W2, S2: Second foot
    - ... and so on
    - Final S state triggers line reset to W1

    Parameters
    ----------
    feet_per_line : int, default=4
        Number of iambic feet per line. Tetrameter=4, Pentameter=5, etc.
    emission_model : object, optional
        Pluggable emission model. If None, uses GeneralizedBinaryEmission.
    strictness : float, default=0.9
        Controls how strongly transitions follow the canonical cycle.
    allow_skips : bool, default=True
        If True, allows "skip" transitions.
    line_reset_prob : float, default=0.85
        Probability of resetting to W1 after final S position.
    """

    def __init__(
        self,
        feet_per_line: int = 4,
        emission_model = None,
        strictness: float = 0.9,
        allow_skips: bool = True,
        line_reset_prob: float = 0.85,
    ):
        self.feet_per_line = feet_per_line
        self.n_states = 2 * feet_per_line
        self.strictness = np.clip(strictness, 0.5, 1.0)
        self.allow_skips = allow_skips
        self.line_reset_prob = line_reset_prob

        # Generate state names: W1, S1, W2, S2, ..., Wn, Sn
        self.state_names = []
        for i in range(1, feet_per_line + 1):
            self.state_names.extend([f'W{i}', f'S{i}'])

        # Create emission model if not provided
        if emission_model is None:
            self.emission_model = GeneralizedBinaryEmission(self.n_states)
        else:
            self.emission_model = emission_model

        # Build matrices
        self.A = self._build_transition_matrix()
        self.B = self.emission_model.emission_matrix()
        self.pi = self._build_initial_distribution()

        # Validate
        self._validate_matrices()

    def _build_transition_matrix(self) -> NDArray:
        """
        Build transition matrix for the generalized cycle.

        Canonical cycle: W1 -> S1 -> W2 -> S2 -> ... -> Wn -> Sn -> W1
        """
        n = self.n_states
        A = np.zeros((n, n))

        if self.allow_skips:
            # Distribute noise across all non-canonical transitions
            noise = (1.0 - self.strictness) / (n - 1)

            for i in range(n):
                # Fill with noise
                A[i, :] = noise

                if i == n - 1:
                    # Last state (final S): reset to W1
                    A[i, :] = (1.0 - self.line_reset_prob) / (n - 1)
                    A[i, 0] = self.line_reset_prob
                else:
                    # Normal progression to next state
                    A[i, i + 1] = self.strictness
        else:
            # Pure cycle: only canonical transitions + self-loops
            self_loop = 1.0 - self.strictness

            for i in range(n - 1):
                A[i, i] = self_loop
                A[i, i + 1] = self.strictness

            # Last state: reset or self-loop
            A[n - 1, n - 1] = 1.0 - self.line_reset_prob
            A[n - 1, 0] = self.line_reset_prob

        # Ensure rows sum to 1
        A = A / A.sum(axis=1, keepdims=True)
        return A

    def _build_initial_distribution(self) -> NDArray:
        """Build initial state distribution (strong preference for W1)."""
        pi = np.ones(self.n_states) * 0.01
        pi[0] = 0.85  # Strong preference for W1
        return pi / pi.sum()

    def _validate_matrices(self):
        """Validate probability distributions."""
        assert np.allclose(self.A.sum(axis=1), 1.0), "Transition rows must sum to 1"
        assert np.allclose(self.B.sum(axis=1), 1.0), "Emission rows must sum to 1"
        assert np.allclose(self.pi.sum(), 1.0), "Initial distribution must sum to 1"
        assert np.all(self.A >= 0), "Transition probabilities must be non-negative"
        assert np.all(self.B >= 0), "Emission probabilities must be non-negative"

    def sample(self, n_steps: int, seed: Optional[int] = None) -> Tuple[NDArray, NDArray]:
        """Generate a sequence of (hidden_states, observations)."""
        rng = np.random.default_rng(seed)

        states = np.zeros(n_steps, dtype=int)
        observations = np.zeros(n_steps, dtype=int)

        states[0] = rng.choice(self.n_states, p=self.pi)
        observations[0] = self.emission_model.sample(states[0], rng)

        for t in range(1, n_steps):
            states[t] = rng.choice(self.n_states, p=self.A[states[t-1]])
            observations[t] = self.emission_model.sample(states[t], rng)

        return states, observations

    def forward(self, observations: NDArray) -> NDArray:
        """Run forward algorithm, returning normalized belief states."""
        T = len(observations)
        beliefs = np.zeros((T, self.n_states))

        alpha = self.pi * self.B[:, observations[0]]
        alpha = alpha / alpha.sum()
        beliefs[0] = alpha

        for t in range(1, T):
            alpha_pred = self.A.T @ alpha
            alpha = alpha_pred * self.B[:, observations[t]]
            alpha = alpha / alpha.sum()
            beliefs[t] = alpha

        return beliefs

    def forward_log(self, observations: NDArray) -> NDArray:
        """Run forward algorithm in log-space for numerical stability."""
        T = len(observations)
        beliefs = np.zeros((T, self.n_states))

        log_alpha = np.log(self.pi + 1e-300) + np.log(self.B[:, observations[0]] + 1e-300)
        log_alpha = log_alpha - np.logaddexp.reduce(log_alpha)
        beliefs[0] = np.exp(log_alpha)

        log_A = np.log(self.A + 1e-300)
        log_B = np.log(self.B + 1e-300)

        for t in range(1, T):
            log_alpha_pred = np.zeros(self.n_states)
            for j in range(self.n_states):
                log_alpha_pred[j] = np.logaddexp.reduce(log_alpha + log_A[:, j])
            log_alpha = log_alpha_pred + log_B[:, observations[t]]
            log_alpha = log_alpha - np.logaddexp.reduce(log_alpha)
            beliefs[t] = np.exp(log_alpha)

        return beliefs

    def is_ergodic(self, n_power: int = 10) -> bool:
        """Check if the HMM is ergodic."""
        A_n = np.linalg.matrix_power(self.A, n_power)
        return np.all(A_n > 1e-10)

    def stationary_distribution(self) -> NDArray:
        """Compute the stationary distribution."""
        eigenvalues, eigenvectors = np.linalg.eig(self.A.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        return stationary / stationary.sum()

    def positions_per_line(self) -> int:
        """Return the number of metrical positions per line."""
        return self.n_states

    def __repr__(self) -> str:
        meter_names = {2: 'dimeter', 3: 'trimeter', 4: 'tetrameter',
                       5: 'pentameter', 6: 'hexameter'}
        meter = meter_names.get(self.feet_per_line, f'{self.feet_per_line}-foot')
        return (
            f"GeneralizedMetricalHMM(meter='{meter}', "
            f"n_states={self.n_states}, strictness={self.strictness:.2f})"
        )


class GeneralizedBinaryEmission:
    """
    Binary emission model for generalized metrical HMM.

    Even-indexed states (W positions) favor unstressed.
    Odd-indexed states (S positions) favor stressed.
    """

    def __init__(self, n_states: int, clarity: float = 0.85):
        self.n_states = n_states
        self.clarity = np.clip(clarity, 0.5, 1.0)
        self._names = ['unstressed', 'stressed']

    @property
    def n_outputs(self) -> int:
        return 2

    @property
    def output_names(self):
        return self._names

    def emission_matrix(self) -> NDArray:
        """Build emission matrix based on position type."""
        noise = 1.0 - self.clarity
        B = np.zeros((self.n_states, 2))

        for i in range(self.n_states):
            if i % 2 == 0:  # W position (even index)
                B[i] = [self.clarity, noise]
            else:  # S position (odd index)
                B[i] = [noise, self.clarity]

        return B

    def sample(self, state: int, rng: np.random.Generator) -> int:
        B = self.emission_matrix()
        return rng.choice(self.n_outputs, p=B[state])

    def __repr__(self) -> str:
        return f"GeneralizedBinaryEmission(n_states={self.n_states}, clarity={self.clarity:.2f})"
