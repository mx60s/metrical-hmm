"""
Pluggable emission models for Metrical HMM.

This module provides different emission models that can be used with
the MetricalHMM class. Each model defines how observations are generated
from hidden states.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List
from numpy.typing import NDArray


# Observation indices for binary model
UNSTRESSED, STRESSED = 0, 1


class EmissionModel(ABC):
    """Abstract base class for emission models."""

    @property
    @abstractmethod
    def n_outputs(self) -> int:
        """Number of possible output symbols."""
        pass

    @property
    @abstractmethod
    def output_names(self) -> List[str]:
        """Names of output symbols."""
        pass

    @abstractmethod
    def emission_matrix(self) -> NDArray:
        """
        Return emission probability matrix B.

        Returns
        -------
        B : NDArray of shape (n_states, n_outputs)
            B[i, k] = P(observation = k | state = i)
        """
        pass

    @abstractmethod
    def sample(self, state: int, rng: np.random.Generator) -> int:
        """
        Sample an observation from the given state.

        Parameters
        ----------
        state : int
            Current hidden state index.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        observation : int
            Sampled observation index.
        """
        pass

    def prob(self, state: int, observation: int) -> float:
        """Get emission probability P(observation | state)."""
        return self.emission_matrix()[state, observation]


class BinaryEmission(EmissionModel):
    """
    Binary emission model: {unstressed, stressed}.

    Strong positions (S1, S2) favor stressed emissions.
    Weak positions (W1, W2) favor unstressed emissions.

    Parameters
    ----------
    clarity : float, default=0.85
        Probability of emitting the "expected" symbol.
        Higher values = cleaner signal, beliefs converge faster.
        Lower values = noisier, beliefs spread out more.
        Range: [0.5, 1.0]
    """

    def __init__(self, clarity: float = 0.85):
        self.clarity = np.clip(clarity, 0.5, 1.0)
        self._names = ['unstressed', 'stressed']

    @property
    def n_outputs(self) -> int:
        return 2

    @property
    def output_names(self) -> List[str]:
        return self._names

    def emission_matrix(self) -> NDArray:
        """
        Build emission matrix.

        W1, W2 (states 0, 2): favor unstressed
        S1, S2 (states 1, 3): favor stressed
        """
        noise = 1.0 - self.clarity

        B = np.array([
            [self.clarity, noise],      # W1: unstressed expected
            [noise, self.clarity],      # S1: stressed expected
            [self.clarity, noise],      # W2: unstressed expected
            [noise, self.clarity],      # S2: stressed expected
        ])

        return B

    def sample(self, state: int, rng: np.random.Generator) -> int:
        B = self.emission_matrix()
        return rng.choice(self.n_outputs, p=B[state])

    def __repr__(self) -> str:
        return f"BinaryEmission(clarity={self.clarity:.2f})"


class VocabEmission(EmissionModel):
    """
    Small vocabulary emission model with syllable-like tokens.

    Uses a vocabulary of tokens with known stress patterns,
    providing richer observations for more naturalistic sequences.

    Parameters
    ----------
    clarity : float, default=0.85
        How strongly emissions match expected stress pattern.
    """

    # Default vocabulary: (token, is_stressed)
    DEFAULT_VOCAB = [
        ('da', False),      # Unstressed: like "a-BOUT"
        ('the', False),     # Unstressed: "THE book"
        ('to', False),      # Unstressed: "to GO"
        ('DUM', True),      # Stressed: like "a-BOUT"
        ('BEAT', True),     # Stressed: "heart-BEAT"
        ('STRONG', True),   # Stressed: "be STRONG"
    ]

    def __init__(self, clarity: float = 0.85, vocab: List[tuple] = None):
        self.clarity = np.clip(clarity, 0.5, 1.0)
        self.vocab = vocab if vocab is not None else self.DEFAULT_VOCAB

        # Separate stressed and unstressed tokens
        self.unstressed_indices = [i for i, (_, stressed) in enumerate(self.vocab) if not stressed]
        self.stressed_indices = [i for i, (_, stressed) in enumerate(self.vocab) if stressed]

        self._names = [tok for tok, _ in self.vocab]

    @property
    def n_outputs(self) -> int:
        return len(self.vocab)

    @property
    def output_names(self) -> List[str]:
        return self._names

    def emission_matrix(self) -> NDArray:
        """
        Build emission matrix.

        Weak positions favor unstressed tokens uniformly.
        Strong positions favor stressed tokens uniformly.
        """
        n_unstressed = len(self.unstressed_indices)
        n_stressed = len(self.stressed_indices)
        noise = 1.0 - self.clarity

        B = np.zeros((4, self.n_outputs))

        for state in range(4):
            is_strong = state in [1, 3]  # S1 or S2

            if is_strong:
                # Favor stressed tokens
                for idx in self.stressed_indices:
                    B[state, idx] = self.clarity / n_stressed
                for idx in self.unstressed_indices:
                    B[state, idx] = noise / n_unstressed
            else:
                # Favor unstressed tokens
                for idx in self.unstressed_indices:
                    B[state, idx] = self.clarity / n_unstressed
                for idx in self.stressed_indices:
                    B[state, idx] = noise / n_stressed

        return B

    def sample(self, state: int, rng: np.random.Generator) -> int:
        B = self.emission_matrix()
        return rng.choice(self.n_outputs, p=B[state])

    def decode_sequence(self, observations: NDArray) -> List[str]:
        """Convert observation indices to token strings."""
        return [self._names[obs] for obs in observations]

    def __repr__(self) -> str:
        return f"VocabEmission(clarity={self.clarity:.2f}, vocab_size={self.n_outputs})"


class PoeticVocabEmission(EmissionModel):
    """
    Extended vocabulary with more poetic/natural syllables.

    Designed for sequences that look more like natural language
    while still having clear stress patterns.
    """

    DEFAULT_VOCAB = [
        # Unstressed syllables (function words, weak syllables)
        ('a', False),
        ('the', False),
        ('to', False),
        ('and', False),
        ('in', False),
        ('of', False),
        ('for', False),
        ('with', False),
        # Stressed syllables (content words, strong syllables)
        ('LOVE', True),
        ('HEART', True),
        ('DREAM', True),
        ('LIGHT', True),
        ('NIGHT', True),
        ('SOUL', True),
        ('LIFE', True),
        ('TIME', True),
    ]

    def __init__(self, clarity: float = 0.85, vocab: List[tuple] = None):
        self.clarity = np.clip(clarity, 0.5, 1.0)
        self.vocab = vocab if vocab is not None else self.DEFAULT_VOCAB

        self.unstressed_indices = [i for i, (_, stressed) in enumerate(self.vocab) if not stressed]
        self.stressed_indices = [i for i, (_, stressed) in enumerate(self.vocab) if stressed]

        self._names = [tok for tok, _ in self.vocab]

    @property
    def n_outputs(self) -> int:
        return len(self.vocab)

    @property
    def output_names(self) -> List[str]:
        return self._names

    def emission_matrix(self) -> NDArray:
        n_unstressed = len(self.unstressed_indices)
        n_stressed = len(self.stressed_indices)
        noise = 1.0 - self.clarity

        B = np.zeros((4, self.n_outputs))

        for state in range(4):
            is_strong = state in [1, 3]

            if is_strong:
                for idx in self.stressed_indices:
                    B[state, idx] = self.clarity / n_stressed
                for idx in self.unstressed_indices:
                    B[state, idx] = noise / n_unstressed
            else:
                for idx in self.unstressed_indices:
                    B[state, idx] = self.clarity / n_unstressed
                for idx in self.stressed_indices:
                    B[state, idx] = noise / n_stressed

        return B

    def sample(self, state: int, rng: np.random.Generator) -> int:
        B = self.emission_matrix()
        return rng.choice(self.n_outputs, p=B[state])

    def decode_sequence(self, observations: NDArray) -> List[str]:
        """Convert observation indices to token strings."""
        return [self._names[obs] for obs in observations]

    def format_as_verse(self, observations: NDArray, line_length: int = 4) -> str:
        """Format observation sequence as verse with line breaks."""
        tokens = self.decode_sequence(observations)
        lines = []
        for i in range(0, len(tokens), line_length):
            line_tokens = tokens[i:i + line_length]
            lines.append(' '.join(line_tokens))
        return '\n'.join(lines)

    def __repr__(self) -> str:
        return f"PoeticVocabEmission(clarity={self.clarity:.2f}, vocab_size={self.n_outputs})"
