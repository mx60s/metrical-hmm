"""
Metrical HMM: Hidden Markov Model for Poetic Meter Analysis

This package provides tools for modeling iambic meter as a Hidden Markov Model
and visualizing the resulting belief geometry on the probability simplex.
"""

from .hmm import (
    MetricalHMM,
    GeneralizedMetricalHMM,
    GeneralizedBinaryEmission,
    STATE_NAMES,
    W1, S1, W2, S2,
)
from .emissions import (
    BinaryEmission,
    VocabEmission,
    PoeticVocabEmission,
    EmissionModel,
    UNSTRESSED,
    STRESSED,
)
from .visualization import (
    belief_to_3d,
    plot_belief_trajectory,
    plot_multiple_trajectories,
    plot_belief_trajectory_2d,
    plot_state_marginals,
    plot_reset_dynamics,
    compare_trajectories,
    create_animation,
    TETRAHEDRON_VERTICES,
    # Interactive plotly versions
    plot_belief_trajectory_interactive,
    plot_multiple_trajectories_interactive,
    plot_reset_dynamics_interactive,
    compare_trajectories_interactive,
    # PCA-based visualizations for high-dimensional spaces
    plot_belief_trajectory_pca_3d_interactive,
    plot_multiple_trajectories_pca_3d_interactive,
    plot_state_marginals_grouped,
    create_animation_generalized,
)
from .analysis import (
    belief_entropy,
    normalized_entropy,
    belief_velocity,
    detect_resets,
    cycle_period,
    summary_statistics,
)

__version__ = "0.1.0"
__all__ = [
    # HMM
    "MetricalHMM",
    "GeneralizedMetricalHMM",
    "GeneralizedBinaryEmission",
    "STATE_NAMES",
    "W1", "S1", "W2", "S2",
    # Emissions
    "BinaryEmission",
    "VocabEmission",
    "PoeticVocabEmission",
    "EmissionModel",
    "UNSTRESSED", "STRESSED",
    # Visualization
    "belief_to_3d",
    "plot_belief_trajectory",
    "plot_multiple_trajectories",
    "plot_belief_trajectory_2d",
    "plot_state_marginals",
    "plot_reset_dynamics",
    "compare_trajectories",
    "create_animation",
    "TETRAHEDRON_VERTICES",
    # Interactive plotly versions
    "plot_belief_trajectory_interactive",
    "plot_multiple_trajectories_interactive",
    "plot_reset_dynamics_interactive",
    "compare_trajectories_interactive",
    # PCA-based visualizations
    "plot_belief_trajectory_pca_3d_interactive",
    "plot_multiple_trajectories_pca_3d_interactive",
    "plot_state_marginals_grouped",
    "create_animation_generalized",
    # Analysis
    "belief_entropy",
    "normalized_entropy",
    "belief_velocity",
    "detect_resets",
    "cycle_period",
    "summary_statistics",
]
