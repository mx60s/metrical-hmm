"""
Visualization utilities for belief geometry on the simplex.

This module provides functions to visualize HMM belief trajectories
on the 3-simplex (tetrahedron) and in 2D projections.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from typing import Optional, List, Tuple
from numpy.typing import NDArray


# Tetrahedron vertices (regular tetrahedron centered at origin)
# These vertices are positioned so the tetrahedron is symmetric
TETRAHEDRON_VERTICES = np.array([
    [1, 1, 1],      # W1
    [1, -1, -1],    # S1
    [-1, 1, -1],    # W2
    [-1, -1, 1]     # S2
]) / np.sqrt(3)

STATE_NAMES = ['W1', 'S1', 'W2', 'S2']
STATE_COLORS = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6']  # Green, Red, Blue, Purple


def belief_to_3d(belief: NDArray) -> NDArray:
    """
    Convert a 4D belief vector to a 3D point in the tetrahedron.

    The belief vector represents a point on the 3-simplex, which is
    embedded as a tetrahedron in 3D space.

    Parameters
    ----------
    belief : NDArray of shape (4,) or (T, 4)
        Probability distribution(s) over 4 states.

    Returns
    -------
    point : NDArray of shape (3,) or (T, 3)
        3D coordinates in the tetrahedron.
    """
    return belief @ TETRAHEDRON_VERTICES


def draw_tetrahedron(ax, alpha: float = 0.3, linewidth: float = 1.5):
    """
    Draw the tetrahedron wireframe on a 3D axis.

    Parameters
    ----------
    ax : Axes3D
        Matplotlib 3D axis.
    alpha : float
        Transparency of edges.
    linewidth : float
        Width of edge lines.
    """
    # All edges of tetrahedron
    edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 2), (1, 3), (2, 3)
    ]

    for i, j in edges:
        ax.plot3D(
            [TETRAHEDRON_VERTICES[i, 0], TETRAHEDRON_VERTICES[j, 0]],
            [TETRAHEDRON_VERTICES[i, 1], TETRAHEDRON_VERTICES[j, 1]],
            [TETRAHEDRON_VERTICES[i, 2], TETRAHEDRON_VERTICES[j, 2]],
            'k-', alpha=alpha, linewidth=linewidth
        )

    # Label vertices
    for i, (v, name) in enumerate(zip(TETRAHEDRON_VERTICES, STATE_NAMES)):
        ax.text(
            v[0] * 1.15, v[1] * 1.15, v[2] * 1.15,
            name, fontsize=12, fontweight='bold',
            color=STATE_COLORS[i], ha='center', va='center'
        )


def plot_belief_trajectory(
    beliefs: NDArray,
    title: str = "Belief Trajectory",
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'viridis',
    point_size: int = 20,
    line_alpha: float = 0.7,
    show_colorbar: bool = True,
    ax: Optional[Axes3D] = None
) -> Tuple[plt.Figure, Axes3D]:
    """
    Plot belief trajectory on the 3-simplex (tetrahedron).

    Parameters
    ----------
    beliefs : NDArray of shape (T, 4)
        Belief states over time.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    cmap : str
        Colormap for time progression.
    point_size : int
        Size of scatter points.
    line_alpha : float
        Transparency of trajectory line.
    show_colorbar : bool
        Whether to show time colorbar.
    ax : Axes3D, optional
        Existing axis to plot on.

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    ax : Axes3D
        Matplotlib 3D axis.
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.get_figure()

    # Draw tetrahedron
    draw_tetrahedron(ax)

    # Convert beliefs to 3D
    points_3d = belief_to_3d(beliefs)
    T = len(beliefs)
    time_colors = np.arange(T)

    # Plot trajectory line
    ax.plot(
        points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
        '-', alpha=line_alpha, linewidth=1, color='gray'
    )

    # Scatter with time-based coloring
    scatter = ax.scatter(
        points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
        c=time_colors, cmap=cmap, s=point_size, alpha=0.8
    )

    if show_colorbar:
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label('Time step', fontsize=10)

    # Set axis properties
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title, fontsize=14)

    # Set equal aspect ratio
    max_range = 0.8
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])

    return fig, ax


def plot_multiple_trajectories(
    belief_list: List[NDArray],
    title: str = "Multiple Belief Trajectories",
    figsize: Tuple[int, int] = (10, 8),
    alpha: float = 0.5,
    colors: Optional[List[str]] = None
) -> Tuple[plt.Figure, Axes3D]:
    """
    Overlay multiple belief trajectories to visualize attractor structure.

    Parameters
    ----------
    belief_list : list of NDArray
        List of belief trajectory arrays, each of shape (T, 4).
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    alpha : float
        Transparency of each trajectory.
    colors : list of str, optional
        Colors for each trajectory. If None, uses a colormap.

    Returns
    -------
    fig : Figure
    ax : Axes3D
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    draw_tetrahedron(ax)

    if colors is None:
        cmap = plt.cm.get_cmap('tab10')
        colors = [cmap(i % 10) for i in range(len(belief_list))]

    for i, beliefs in enumerate(belief_list):
        points_3d = belief_to_3d(beliefs)
        ax.plot(
            points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
            '-', alpha=alpha, linewidth=1, color=colors[i]
        )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title, fontsize=14)

    max_range = 0.8
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])

    return fig, ax


def plot_belief_trajectory_2d(
    beliefs: NDArray,
    state_names: Optional[List[str]] = None,
    title: str = "Belief Trajectory (PCA Projection)",
    figsize: Tuple[int, int] = (10, 6),
    cmap: str = 'viridis'
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot belief trajectory in 2D using PCA projection.

    Parameters
    ----------
    beliefs : NDArray of shape (T, n_states)
        Belief states over time.
    state_names : list of str, optional
        Names for each state. If None, uses S0, S1, ...
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    cmap : str
        Colormap.

    Returns
    -------
    fig : Figure
    ax : Axes
    """
    from sklearn.decomposition import PCA

    n_states = beliefs.shape[1]

    # Generate state names if not provided
    if state_names is None:
        if n_states == 4:
            state_names = STATE_NAMES
        else:
            state_names = [f'S{i}' for i in range(n_states)]

    # Generate colors
    if n_states <= len(STATE_COLORS):
        colors = STATE_COLORS[:n_states]
    else:
        colors = [plt.cm.tab10(i % 10) for i in range(n_states)]

    # Project to 2D
    pca = PCA(n_components=2)
    points_2d = pca.fit_transform(beliefs)

    # Also project simplex vertices
    vertices_2d = pca.transform(np.eye(n_states))

    fig, ax = plt.subplots(figsize=figsize)

    # Plot trajectory
    T = len(beliefs)
    scatter = ax.scatter(
        points_2d[:, 0], points_2d[:, 1],
        c=np.arange(T), cmap=cmap, s=30, alpha=0.8
    )
    ax.plot(points_2d[:, 0], points_2d[:, 1], '-', alpha=0.3, color='gray')

    # Mark vertices
    for i, (v, name) in enumerate(zip(vertices_2d, state_names)):
        ax.scatter(v[0], v[1], s=200, c=[colors[i]], marker='s', zorder=10)
        ax.annotate(name, (v[0], v[1]), fontsize=10, fontweight='bold',
                   xytext=(5, 5), textcoords='offset points')

    plt.colorbar(scatter, ax=ax, label='Time step')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
    ax.set_title(title)
    ax.set_aspect('equal')

    return fig, ax


def plot_state_marginals(
    beliefs: NDArray,
    title: str = "State Marginals Over Time",
    figsize: Tuple[int, int] = (12, 4)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot belief probability for each state over time.

    Parameters
    ----------
    beliefs : NDArray of shape (T, 4)
        Belief states over time.
    title : str
        Plot title.
    figsize : tuple
        Figure size.

    Returns
    -------
    fig : Figure
    ax : Axes
    """
    fig, ax = plt.subplots(figsize=figsize)

    T = len(beliefs)
    time = np.arange(T)

    for i, name in enumerate(STATE_NAMES):
        ax.plot(time, beliefs[:, i], label=name, color=STATE_COLORS[i], linewidth=2)

    ax.set_xlabel('Time step', fontsize=12)
    ax.set_ylabel('P(state)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_reset_dynamics(
    beliefs: NDArray,
    observations: NDArray,
    line_length: int = 4,
    title: str = "Reset Dynamics at Line Boundaries",
    figsize: Tuple[int, int] = (10, 8)
) -> Tuple[plt.Figure, Axes3D]:
    """
    Visualize belief trajectory with line boundaries highlighted.

    Parameters
    ----------
    beliefs : NDArray of shape (T, 4)
        Belief states over time.
    observations : NDArray of shape (T,)
        Observations (used to identify line boundaries).
    line_length : int
        Number of positions per line.
    title : str
        Plot title.
    figsize : tuple
        Figure size.

    Returns
    -------
    fig : Figure
    ax : Axes3D
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    draw_tetrahedron(ax)

    points_3d = belief_to_3d(beliefs)
    T = len(beliefs)

    # Plot trajectory
    ax.plot(
        points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
        '-', alpha=0.5, linewidth=1, color='gray'
    )

    # Highlight line boundaries (every line_length steps)
    boundary_indices = list(range(0, T, line_length))

    # Regular points
    regular_mask = np.ones(T, dtype=bool)
    regular_mask[boundary_indices] = False

    ax.scatter(
        points_3d[regular_mask, 0],
        points_3d[regular_mask, 1],
        points_3d[regular_mask, 2],
        c='blue', s=20, alpha=0.5, label='Within line'
    )

    # Boundary points (line starts)
    ax.scatter(
        points_3d[boundary_indices, 0],
        points_3d[boundary_indices, 1],
        points_3d[boundary_indices, 2],
        c='red', s=60, marker='*', label='Line boundary'
    )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title, fontsize=14)
    ax.legend()

    max_range = 0.8
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])

    return fig, ax


def compare_trajectories(
    beliefs_list: List[NDArray],
    labels: List[str],
    title: str = "Trajectory Comparison",
    figsize: Tuple[int, int] = (15, 5)
) -> Tuple[plt.Figure, List[Axes3D]]:
    """
    Side-by-side comparison of multiple belief trajectories.

    Parameters
    ----------
    beliefs_list : list of NDArray
        List of belief arrays to compare.
    labels : list of str
        Labels for each trajectory.
    title : str
        Overall title.
    figsize : tuple
        Figure size.

    Returns
    -------
    fig : Figure
    axes : list of Axes3D
    """
    n = len(beliefs_list)
    fig = plt.figure(figsize=figsize)

    axes = []
    for i, (beliefs, label) in enumerate(zip(beliefs_list, labels)):
        ax = fig.add_subplot(1, n, i + 1, projection='3d')
        draw_tetrahedron(ax)

        points_3d = belief_to_3d(beliefs)
        T = len(beliefs)

        ax.plot(
            points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
            '-', alpha=0.7, linewidth=1
        )
        ax.scatter(
            points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
            c=np.arange(T), cmap='viridis', s=15, alpha=0.8
        )

        ax.set_title(label, fontsize=12)
        max_range = 0.8
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])

        axes.append(ax)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    return fig, axes


def create_animation(
    beliefs: NDArray,
    observations: NDArray = None,
    emission_model = None,
    interval: int = 100,
    trail_length: int = 20,
    figsize: Tuple[int, int] = (12, 6)
):
    """
    Create an animation of belief evolution.

    Parameters
    ----------
    beliefs : NDArray of shape (T, 4)
        Belief states over time.
    observations : NDArray of shape (T,), optional
        Observations for annotation.
    emission_model : EmissionModel, optional
        For decoding observations to text.
    interval : int
        Milliseconds between frames.
    trail_length : int
        Number of past points to show in trail.
    figsize : tuple
        Figure size.

    Returns
    -------
    anim : FuncAnimation
        Matplotlib animation object.
    """
    from matplotlib.animation import FuncAnimation

    fig = plt.figure(figsize=figsize)

    # 3D trajectory panel
    ax1 = fig.add_subplot(121, projection='3d')
    draw_tetrahedron(ax1)
    ax1.set_title('Belief Trajectory')

    # State marginals panel
    ax2 = fig.add_subplot(122)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([-0.5, 3.5])
    ax2.set_xlabel('Probability')
    ax2.set_title('State Probabilities')

    points_3d = belief_to_3d(beliefs)
    T = len(beliefs)

    # Initialize plot elements
    current_point, = ax1.plot([], [], [], 'ro', markersize=10)
    trail_line, = ax1.plot([], [], [], 'b-', alpha=0.5, linewidth=2)

    bars = ax2.barh(range(4), [0, 0, 0, 0], color=STATE_COLORS)
    ax2.set_yticks(range(4))
    ax2.set_yticklabels(STATE_NAMES)

    time_text = ax2.text(0.5, 3.8, '', fontsize=12, ha='center')
    obs_text = ax2.text(0.5, -0.8, '', fontsize=10, ha='center')

    max_range = 0.8
    ax1.set_xlim([-max_range, max_range])
    ax1.set_ylim([-max_range, max_range])
    ax1.set_zlim([-max_range, max_range])

    def init():
        current_point.set_data([], [])
        current_point.set_3d_properties([])
        trail_line.set_data([], [])
        trail_line.set_3d_properties([])
        for bar in bars:
            bar.set_width(0)
        time_text.set_text('')
        obs_text.set_text('')
        return [current_point, trail_line, *bars, time_text, obs_text]

    def update(frame):
        # Update 3D point
        current_point.set_data([points_3d[frame, 0]], [points_3d[frame, 1]])
        current_point.set_3d_properties([points_3d[frame, 2]])

        # Update trail
        start = max(0, frame - trail_length)
        trail_line.set_data(points_3d[start:frame+1, 0], points_3d[start:frame+1, 1])
        trail_line.set_3d_properties(points_3d[start:frame+1, 2])

        # Update bar chart
        for bar, prob in zip(bars, beliefs[frame]):
            bar.set_width(prob)

        # Update text
        time_text.set_text(f't = {frame}')

        if observations is not None and emission_model is not None:
            obs_name = emission_model.output_names[observations[frame]]
            obs_text.set_text(f'Observation: {obs_name}')

        return [current_point, trail_line, *bars, time_text, obs_text]

    anim = FuncAnimation(
        fig, update, init_func=init,
        frames=T, interval=interval, blit=False
    )

    plt.tight_layout()
    return anim


# =============================================================================
# Interactive Plotly Visualizations
# =============================================================================

def _get_tetrahedron_edges():
    """Get edge coordinates for drawing tetrahedron wireframe in plotly."""
    edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 2), (1, 3), (2, 3)
    ]

    x_edges, y_edges, z_edges = [], [], []
    for i, j in edges:
        x_edges.extend([TETRAHEDRON_VERTICES[i, 0], TETRAHEDRON_VERTICES[j, 0], None])
        y_edges.extend([TETRAHEDRON_VERTICES[i, 1], TETRAHEDRON_VERTICES[j, 1], None])
        z_edges.extend([TETRAHEDRON_VERTICES[i, 2], TETRAHEDRON_VERTICES[j, 2], None])

    return x_edges, y_edges, z_edges


def plot_belief_trajectory_interactive(
    beliefs: NDArray,
    title: str = "Belief Trajectory (Interactive)",
    point_size: int = 4,
    line_width: float = 2,
    height: int = 700,
    width: int = 800,
):
    """
    Plot interactive belief trajectory using Plotly.

    Parameters
    ----------
    beliefs : NDArray of shape (T, 4)
        Belief states over time.
    title : str
        Plot title.
    point_size : int
        Size of scatter points.
    line_width : float
        Width of trajectory line.
    height : int
        Figure height in pixels.
    width : int
        Figure width in pixels.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive plotly figure.
    """
    import plotly.graph_objects as go

    points_3d = belief_to_3d(beliefs)
    T = len(beliefs)

    # Tetrahedron wireframe
    x_edges, y_edges, z_edges = _get_tetrahedron_edges()

    fig = go.Figure()

    # Add tetrahedron edges
    fig.add_trace(go.Scatter3d(
        x=x_edges, y=y_edges, z=z_edges,
        mode='lines',
        line=dict(color='black', width=2),
        name='Simplex',
        hoverinfo='skip'
    ))

    # Add vertex labels
    fig.add_trace(go.Scatter3d(
        x=TETRAHEDRON_VERTICES[:, 0] * 1.15,
        y=TETRAHEDRON_VERTICES[:, 1] * 1.15,
        z=TETRAHEDRON_VERTICES[:, 2] * 1.15,
        mode='text',
        text=STATE_NAMES,
        textfont=dict(size=14, color=STATE_COLORS),
        hoverinfo='skip',
        showlegend=False
    ))

    # Add trajectory line
    fig.add_trace(go.Scatter3d(
        x=points_3d[:, 0],
        y=points_3d[:, 1],
        z=points_3d[:, 2],
        mode='lines',
        line=dict(color='rgba(100,100,100,0.5)', width=line_width),
        name='Trajectory',
        hoverinfo='skip'
    ))

    # Add points colored by time
    fig.add_trace(go.Scatter3d(
        x=points_3d[:, 0],
        y=points_3d[:, 1],
        z=points_3d[:, 2],
        mode='markers',
        marker=dict(
            size=point_size,
            color=np.arange(T),
            colorscale='Viridis',
            colorbar=dict(title='Time', thickness=15),
            opacity=0.8
        ),
        name='Beliefs',
        hovertemplate=(
            't=%{marker.color:.0f}<br>'
            'P(W1)=%{customdata[0]:.3f}<br>'
            'P(S1)=%{customdata[1]:.3f}<br>'
            'P(W2)=%{customdata[2]:.3f}<br>'
            'P(S2)=%{customdata[3]:.3f}<extra></extra>'
        ),
        customdata=beliefs
    ))

    # Layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(range=[-0.8, 0.8], title=''),
            yaxis=dict(range=[-0.8, 0.8], title=''),
            zaxis=dict(range=[-0.8, 0.8], title=''),
            aspectmode='cube'
        ),
        height=height,
        width=width,
        showlegend=True
    )

    return fig


def plot_multiple_trajectories_interactive(
    belief_list: List[NDArray],
    title: str = "Multiple Belief Trajectories (Interactive)",
    labels: Optional[List[str]] = None,
    height: int = 700,
    width: int = 800,
):
    """
    Overlay multiple belief trajectories with interactive plotly visualization.

    Parameters
    ----------
    belief_list : list of NDArray
        List of belief trajectory arrays.
    title : str
        Plot title.
    labels : list of str, optional
        Labels for each trajectory.
    height : int
        Figure height.
    width : int
        Figure width.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go
    import plotly.express as px

    fig = go.Figure()

    # Tetrahedron wireframe
    x_edges, y_edges, z_edges = _get_tetrahedron_edges()
    fig.add_trace(go.Scatter3d(
        x=x_edges, y=y_edges, z=z_edges,
        mode='lines',
        line=dict(color='black', width=2),
        name='Simplex',
        hoverinfo='skip'
    ))

    # Vertex labels
    fig.add_trace(go.Scatter3d(
        x=TETRAHEDRON_VERTICES[:, 0] * 1.15,
        y=TETRAHEDRON_VERTICES[:, 1] * 1.15,
        z=TETRAHEDRON_VERTICES[:, 2] * 1.15,
        mode='text',
        text=STATE_NAMES,
        textfont=dict(size=14, color=STATE_COLORS),
        hoverinfo='skip',
        showlegend=False
    ))

    # Color palette
    colors = px.colors.qualitative.Plotly

    if labels is None:
        labels = [f'Trajectory {i+1}' for i in range(len(belief_list))]

    for i, (beliefs, label) in enumerate(zip(belief_list, labels)):
        points_3d = belief_to_3d(beliefs)
        color = colors[i % len(colors)]

        fig.add_trace(go.Scatter3d(
            x=points_3d[:, 0],
            y=points_3d[:, 1],
            z=points_3d[:, 2],
            mode='lines',
            line=dict(color=color, width=2),
            name=label,
            opacity=0.7
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(range=[-0.8, 0.8], title=''),
            yaxis=dict(range=[-0.8, 0.8], title=''),
            zaxis=dict(range=[-0.8, 0.8], title=''),
            aspectmode='cube'
        ),
        height=height,
        width=width
    )

    return fig


def plot_reset_dynamics_interactive(
    beliefs: NDArray,
    observations: NDArray,
    line_length: int = 4,
    title: str = "Reset Dynamics (Interactive)",
    height: int = 700,
    width: int = 800,
):
    """
    Visualize belief trajectory with line boundaries highlighted (interactive).

    Parameters
    ----------
    beliefs : NDArray of shape (T, 4)
        Belief states over time.
    observations : NDArray of shape (T,)
        Observations.
    line_length : int
        Number of positions per line.
    title : str
        Plot title.
    height : int
        Figure height.
    width : int
        Figure width.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    points_3d = belief_to_3d(beliefs)
    T = len(beliefs)

    fig = go.Figure()

    # Tetrahedron wireframe
    x_edges, y_edges, z_edges = _get_tetrahedron_edges()
    fig.add_trace(go.Scatter3d(
        x=x_edges, y=y_edges, z=z_edges,
        mode='lines',
        line=dict(color='black', width=2),
        name='Simplex',
        hoverinfo='skip'
    ))

    # Vertex labels
    fig.add_trace(go.Scatter3d(
        x=TETRAHEDRON_VERTICES[:, 0] * 1.15,
        y=TETRAHEDRON_VERTICES[:, 1] * 1.15,
        z=TETRAHEDRON_VERTICES[:, 2] * 1.15,
        mode='text',
        text=STATE_NAMES,
        textfont=dict(size=14, color=STATE_COLORS),
        hoverinfo='skip',
        showlegend=False
    ))

    # Trajectory line
    fig.add_trace(go.Scatter3d(
        x=points_3d[:, 0],
        y=points_3d[:, 1],
        z=points_3d[:, 2],
        mode='lines',
        line=dict(color='gray', width=1),
        name='Trajectory',
        opacity=0.5
    ))

    # Separate boundary and regular points
    boundary_indices = list(range(0, T, line_length))
    regular_mask = np.ones(T, dtype=bool)
    regular_mask[boundary_indices] = False

    # Regular points
    fig.add_trace(go.Scatter3d(
        x=points_3d[regular_mask, 0],
        y=points_3d[regular_mask, 1],
        z=points_3d[regular_mask, 2],
        mode='markers',
        marker=dict(size=3, color='blue', opacity=0.6),
        name='Within line',
        hovertemplate='t=%{customdata}<extra></extra>',
        customdata=np.arange(T)[regular_mask]
    ))

    # Boundary points
    fig.add_trace(go.Scatter3d(
        x=points_3d[boundary_indices, 0],
        y=points_3d[boundary_indices, 1],
        z=points_3d[boundary_indices, 2],
        mode='markers',
        marker=dict(size=8, color='red', symbol='diamond'),
        name='Line boundary',
        hovertemplate='Line start: t=%{customdata}<extra></extra>',
        customdata=boundary_indices
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(range=[-0.8, 0.8], title=''),
            yaxis=dict(range=[-0.8, 0.8], title=''),
            zaxis=dict(range=[-0.8, 0.8], title=''),
            aspectmode='cube'
        ),
        height=height,
        width=width
    )

    return fig


def compare_trajectories_interactive(
    beliefs_list: List[NDArray],
    labels: List[str],
    title: str = "Trajectory Comparison (Interactive)",
    height: int = 500,
    width: int = 1200,
):
    """
    Side-by-side comparison of trajectories using plotly subplots.

    Parameters
    ----------
    beliefs_list : list of NDArray
        List of belief arrays to compare.
    labels : list of str
        Labels for each trajectory.
    title : str
        Overall title.
    height : int
        Figure height.
    width : int
        Figure width.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    n = len(beliefs_list)

    # Create subplots
    fig = make_subplots(
        rows=1, cols=n,
        specs=[[{'type': 'scatter3d'} for _ in range(n)]],
        subplot_titles=labels,
        horizontal_spacing=0.02
    )

    x_edges, y_edges, z_edges = _get_tetrahedron_edges()

    for i, beliefs in enumerate(beliefs_list):
        col = i + 1
        points_3d = belief_to_3d(beliefs)
        T = len(beliefs)

        # Tetrahedron
        fig.add_trace(go.Scatter3d(
            x=x_edges, y=y_edges, z=z_edges,
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=False,
            hoverinfo='skip'
        ), row=1, col=col)

        # Vertex labels
        fig.add_trace(go.Scatter3d(
            x=TETRAHEDRON_VERTICES[:, 0] * 1.15,
            y=TETRAHEDRON_VERTICES[:, 1] * 1.15,
            z=TETRAHEDRON_VERTICES[:, 2] * 1.15,
            mode='text',
            text=STATE_NAMES,
            textfont=dict(size=12, color=STATE_COLORS),
            showlegend=False,
            hoverinfo='skip'
        ), row=1, col=col)

        # Trajectory
        fig.add_trace(go.Scatter3d(
            x=points_3d[:, 0],
            y=points_3d[:, 1],
            z=points_3d[:, 2],
            mode='lines+markers',
            line=dict(color='rgba(100,100,100,0.5)', width=2),
            marker=dict(
                size=3,
                color=np.arange(T),
                colorscale='Viridis',
                showscale=(i == n - 1),
                colorbar=dict(title='Time', x=1.02) if i == n - 1 else None
            ),
            showlegend=False,
            hovertemplate='t=%{marker.color:.0f}<extra></extra>'
        ), row=1, col=col)

    # Update all scenes
    for i in range(n):
        scene_name = 'scene' if i == 0 else f'scene{i+1}'
        fig.update_layout(**{
            scene_name: dict(
                xaxis=dict(range=[-0.8, 0.8], title='', showticklabels=False),
                yaxis=dict(range=[-0.8, 0.8], title='', showticklabels=False),
                zaxis=dict(range=[-0.8, 0.8], title='', showticklabels=False),
                aspectmode='cube'
            )
        })

    fig.update_layout(
        title=title,
        height=height,
        width=width
    )

    return fig


# =============================================================================
# PCA-based Visualizations for High-Dimensional Belief Spaces
# =============================================================================

def plot_belief_trajectory_pca_3d_interactive(
    beliefs: NDArray,
    state_names: Optional[List[str]] = None,
    title: str = "Belief Trajectory (PCA 3D)",
    point_size: int = 4,
    height: int = 700,
    width: int = 800,
):
    """
    Plot belief trajectory in 3D using PCA projection (interactive).

    Works for any number of states by projecting to top 3 principal components.
    Essential for visualizing high-dimensional belief spaces (e.g., 8 states).

    Parameters
    ----------
    beliefs : NDArray of shape (T, n_states)
        Belief states over time.
    state_names : list of str, optional
        Names for each state.
    title : str
        Plot title.
    point_size : int
        Size of scatter points.
    height : int
        Figure height.
    width : int
        Figure width.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go
    from sklearn.decomposition import PCA

    n_states = beliefs.shape[1]
    T = len(beliefs)

    if state_names is None:
        state_names = [f'S{i}' for i in range(n_states)]

    # PCA projection to 3D
    pca = PCA(n_components=3)
    points_3d = pca.fit_transform(beliefs)

    # Project simplex vertices (unit vectors) for reference
    vertices = np.eye(n_states)
    vertices_3d = pca.transform(vertices)

    fig = go.Figure()

    # Add trajectory line
    fig.add_trace(go.Scatter3d(
        x=points_3d[:, 0],
        y=points_3d[:, 1],
        z=points_3d[:, 2],
        mode='lines',
        line=dict(color='rgba(100,100,100,0.4)', width=2),
        name='Trajectory',
        hoverinfo='skip'
    ))

    # Add points colored by time
    fig.add_trace(go.Scatter3d(
        x=points_3d[:, 0],
        y=points_3d[:, 1],
        z=points_3d[:, 2],
        mode='markers',
        marker=dict(
            size=point_size,
            color=np.arange(T),
            colorscale='Viridis',
            colorbar=dict(title='Time', thickness=15),
            opacity=0.8
        ),
        name='Beliefs',
        customdata=beliefs,
        hovertemplate='t=%{marker.color:.0f}<extra></extra>'
    ))

    # Add state vertices as reference points
    # Use a color cycle for states
    colors = [
        '#2ecc71', '#e74c3c', '#3498db', '#9b59b6',
        '#f39c12', '#1abc9c', '#e91e63', '#00bcd4'
    ]

    for i, (v, name) in enumerate(zip(vertices_3d, state_names)):
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter3d(
            x=[v[0]], y=[v[1]], z=[v[2]],
            mode='markers+text',
            marker=dict(size=8, color=color, symbol='diamond'),
            text=[name],
            textposition='top center',
            textfont=dict(size=12, color=color),
            name=name,
            showlegend=False
        ))

    var_explained = pca.explained_variance_ratio_
    fig.update_layout(
        title=f"{title}<br><sub>Var explained: PC1={var_explained[0]:.1%}, PC2={var_explained[1]:.1%}, PC3={var_explained[2]:.1%}</sub>",
        scene=dict(
            xaxis_title=f'PC1 ({var_explained[0]:.1%})',
            yaxis_title=f'PC2 ({var_explained[1]:.1%})',
            zaxis_title=f'PC3 ({var_explained[2]:.1%})',
            aspectmode='cube'
        ),
        height=height,
        width=width
    )

    return fig


def plot_multiple_trajectories_pca_3d_interactive(
    belief_list: List[NDArray],
    state_names: Optional[List[str]] = None,
    title: str = "Multiple Trajectories (PCA 3D)",
    labels: Optional[List[str]] = None,
    height: int = 700,
    width: int = 800,
):
    """
    Overlay multiple belief trajectories in PCA 3D space (interactive).

    Parameters
    ----------
    belief_list : list of NDArray
        List of belief trajectory arrays.
    state_names : list of str, optional
        Names for each state.
    title : str
        Plot title.
    labels : list of str, optional
        Labels for each trajectory.
    height : int
        Figure height.
    width : int
        Figure width.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go
    import plotly.express as px
    from sklearn.decomposition import PCA

    # Stack all beliefs for fitting PCA
    all_beliefs = np.vstack(belief_list)
    n_states = all_beliefs.shape[1]

    if state_names is None:
        state_names = [f'S{i}' for i in range(n_states)]

    # Fit PCA on combined data
    pca = PCA(n_components=3)
    pca.fit(all_beliefs)

    # Project vertices
    vertices = np.eye(n_states)
    vertices_3d = pca.transform(vertices)

    fig = go.Figure()

    # Color palette
    colors = px.colors.qualitative.Plotly

    if labels is None:
        labels = [f'Trajectory {i+1}' for i in range(len(belief_list))]

    for i, (beliefs, label) in enumerate(zip(belief_list, labels)):
        points_3d = pca.transform(beliefs)
        color = colors[i % len(colors)]

        fig.add_trace(go.Scatter3d(
            x=points_3d[:, 0],
            y=points_3d[:, 1],
            z=points_3d[:, 2],
            mode='lines',
            line=dict(color=color, width=2),
            name=label,
            opacity=0.7
        ))

    # Add state vertices
    state_colors = [
        '#2ecc71', '#e74c3c', '#3498db', '#9b59b6',
        '#f39c12', '#1abc9c', '#e91e63', '#00bcd4'
    ]

    for i, (v, name) in enumerate(zip(vertices_3d, state_names)):
        color = state_colors[i % len(state_colors)]
        fig.add_trace(go.Scatter3d(
            x=[v[0]], y=[v[1]], z=[v[2]],
            mode='markers+text',
            marker=dict(size=8, color=color, symbol='diamond'),
            text=[name],
            textposition='top center',
            textfont=dict(size=10, color=color),
            showlegend=False
        ))

    var_explained = pca.explained_variance_ratio_
    fig.update_layout(
        title=f"{title}<br><sub>Var explained: {sum(var_explained):.1%} total</sub>",
        scene=dict(
            xaxis_title=f'PC1 ({var_explained[0]:.1%})',
            yaxis_title=f'PC2 ({var_explained[1]:.1%})',
            zaxis_title=f'PC3 ({var_explained[2]:.1%})',
            aspectmode='cube'
        ),
        height=height,
        width=width
    )

    return fig


def plot_state_marginals_grouped(
    beliefs: NDArray,
    state_names: List[str],
    group_by: str = 'position',
    title: str = "Grouped State Marginals",
    figsize: Tuple[int, int] = (14, 5)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot belief marginals grouped by position type or foot number.

    Parameters
    ----------
    beliefs : NDArray of shape (T, n_states)
        Belief states over time.
    state_names : list of str
        Names like ['W1', 'S1', 'W2', 'S2', ...].
    group_by : str
        'position' to group W vs S, 'foot' to group by foot number.
    title : str
        Plot title.
    figsize : tuple
        Figure size.

    Returns
    -------
    fig : Figure
    ax : Axes
    """
    T, n_states = beliefs.shape
    time = np.arange(T)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left plot: P(Weak) vs P(Strong)
    weak_mask = np.array([i % 2 == 0 for i in range(n_states)])
    p_weak = beliefs[:, weak_mask].sum(axis=1)
    p_strong = beliefs[:, ~weak_mask].sum(axis=1)

    axes[0].plot(time, p_weak, 'b-', label='P(Weak)', linewidth=2)
    axes[0].plot(time, p_strong, 'r-', label='P(Strong)', linewidth=2)
    axes[0].set_xlabel('Time step')
    axes[0].set_ylabel('Probability')
    axes[0].set_title('Weak vs Strong Position')
    axes[0].legend()
    axes[0].set_ylim([0, 1])
    axes[0].grid(True, alpha=0.3)

    # Right plot: P(each foot)
    n_feet = n_states // 2
    foot_colors = plt.cm.viridis(np.linspace(0, 1, n_feet))

    for foot in range(n_feet):
        w_idx = 2 * foot
        s_idx = 2 * foot + 1
        p_foot = beliefs[:, w_idx] + beliefs[:, s_idx]
        axes[1].plot(time, p_foot, color=foot_colors[foot],
                    label=f'Foot {foot+1}', linewidth=2)

    axes[1].set_xlabel('Time step')
    axes[1].set_ylabel('Probability')
    axes[1].set_title('Probability by Foot')
    axes[1].legend()
    axes[1].set_ylim([0, 1])
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    return fig, axes


def create_animation_generalized(
    beliefs: NDArray,
    state_names: List[str],
    observations: NDArray = None,
    emission_model = None,
    interval: int = 100,
    figsize: Tuple[int, int] = (14, 6)
):
    """
    Create animation for generalized (high-dimensional) belief trajectories.

    Shows PCA 2D projection alongside state probability bar chart.

    Parameters
    ----------
    beliefs : NDArray of shape (T, n_states)
        Belief states over time.
    state_names : list of str
        Names for each state.
    observations : NDArray, optional
        Observations for annotation.
    emission_model : object, optional
        For decoding observations.
    interval : int
        Milliseconds between frames.
    figsize : tuple
        Figure size.

    Returns
    -------
    anim : FuncAnimation
    """
    from matplotlib.animation import FuncAnimation
    from sklearn.decomposition import PCA

    T, n_states = beliefs.shape

    # PCA for 2D projection
    pca = PCA(n_components=2)
    points_2d = pca.fit_transform(beliefs)
    vertices_2d = pca.transform(np.eye(n_states))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Setup left panel (PCA 2D)
    ax1.set_xlim(points_2d[:, 0].min() - 0.1, points_2d[:, 0].max() + 0.1)
    ax1.set_ylim(points_2d[:, 1].min() - 0.1, points_2d[:, 1].max() + 0.1)
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax1.set_title('Belief Trajectory (PCA)')
    ax1.set_aspect('equal')

    # Plot vertices
    colors = plt.cm.tab10(np.linspace(0, 1, n_states))
    for i, (v, name) in enumerate(zip(vertices_2d, state_names)):
        ax1.scatter(v[0], v[1], s=100, c=[colors[i]], marker='s', zorder=5)
        ax1.annotate(name, (v[0], v[1]), fontsize=9, fontweight='bold')

    # Initialize animated elements
    trail_line, = ax1.plot([], [], 'b-', alpha=0.3, linewidth=1)
    current_point, = ax1.plot([], [], 'ro', markersize=10)

    # Setup right panel (bar chart)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([-0.5, n_states - 0.5])
    ax2.set_xlabel('Probability')
    ax2.set_title('State Probabilities')

    bars = ax2.barh(range(n_states), [0] * n_states, color=colors)
    ax2.set_yticks(range(n_states))
    ax2.set_yticklabels(state_names)

    time_text = ax2.text(0.5, n_states - 0.2, '', fontsize=12, ha='center')

    def init():
        trail_line.set_data([], [])
        current_point.set_data([], [])
        for bar in bars:
            bar.set_width(0)
        time_text.set_text('')
        return [trail_line, current_point, *bars, time_text]

    def update(frame):
        # Update trail (last 30 points)
        start = max(0, frame - 30)
        trail_line.set_data(points_2d[start:frame+1, 0], points_2d[start:frame+1, 1])

        # Update current point
        current_point.set_data([points_2d[frame, 0]], [points_2d[frame, 1]])

        # Update bars
        for bar, prob in zip(bars, beliefs[frame]):
            bar.set_width(prob)

        time_text.set_text(f't = {frame}')

        return [trail_line, current_point, *bars, time_text]

    anim = FuncAnimation(
        fig, update, init_func=init,
        frames=T, interval=interval, blit=False
    )

    plt.tight_layout()
    return anim
