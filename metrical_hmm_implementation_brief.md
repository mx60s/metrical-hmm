# Implementation Brief: Metrical HMM with Belief Geometry Visualization

## Project Context

### Research Goal
I'm investigating whether pretrained LLMs develop internal representations that mirror **belief geometry** — the geometric structure of posterior probability distributions that arise from Bayesian inference on Hidden Markov Models.

Previous work has proven that toy transformer models trained to predict the next state of toy HMMs will develop internal representations whose geometry matches the theoretical belief simplex geometry. I'm now testing whether this phenomenon extends to **pretrained LLMs** on more complex, naturalistic HMMs.

### Why Metrical HMMs
I need HMMs satisfying three constraints:
1. **Complex geometry**: Belief states must track structures more elaborate than simple square/ring graphs
2. **Naturalistic task**: LLMs must have seen this pattern extensively during pretraining (so they've developed robust features for it)
3. **Ergodic outputs**: The token sequence must be ergodic (any state reachable from any other)

Metrical HMMs (modeling poetic meter like iambic verse) satisfy all three:
- **Geometry**: Cyclic structure with hierarchical reset at line boundaries creates interesting periodic attractor dynamics
- **Naturalness**: LLMs have seen enormous amounts of metered text (Shakespeare, Milton, song lyrics, nursery rhymes, rap)
- **Ergodicity**: Quasi-periodic — states cycle regularly through metrical positions with good mixing over multiple lines

---

## HMM Specification: Iambic Dimeter

### Overview
Model the simplest interesting metrical pattern: **iambic dimeter** (da-DUM da-DUM per line).

An iamb is a metrical foot consisting of an unstressed syllable followed by a stressed syllable. Dimeter means two feet per line. So each line has 4 syllabic positions.

### State Space (4 states)
States represent **position within the metrical line**:

| State | Meaning | Metrical Role |
|-------|---------|---------------|
| `W1` | Weak position, foot 1 | Unstressed syllable expected |
| `S1` | Strong position, foot 1 | Stressed syllable expected |
| `W2` | Weak position, foot 2 | Unstressed syllable expected |
| `S2` | Strong position, foot 2 | Stressed syllable expected (line ends after this) |

### Transition Matrix (4×4)
The transitions encode:
- **Normal progression**: W1 → S1 → W2 → S2 → W1 (cycling)
- **Line boundary reset**: S2 wraps back to W1 with high probability
- **Metrical variations**: Small probability of "substitution feet" (trochees, spondees) allowing deviation from strict pattern

```
Transition matrix A[i,j] = P(next_state = j | current_state = i)

           W1      S1      W2      S2
    W1 [ 0.05    0.90    0.03    0.02 ]
    S1 [ 0.02    0.05    0.88    0.05 ]
    W2 [ 0.02    0.03    0.05    0.90 ]
    S2 [ 0.85    0.05    0.05    0.05 ]
```

**Interpretation:**
- From W1: 90% chance of proceeding to S1 (normal iamb), 5% staying (extra unstressed), small chances of skipping
- From S1: 88% chance of proceeding to W2 (next foot), allowing some variation
- From W2: 90% chance of proceeding to S2 (completing second iamb)
- From S2: 85% chance of resetting to W1 (new line), 15% for variations/enjambment

### Emission Model
**Option A: Binary emissions (simpler)**

Emit tokens from {`stressed`, `unstressed`} based on metrical position:

```
Emission matrix B[i,k] = P(observation = k | state = i)

              unstressed   stressed
    W1    [     0.85         0.15    ]
    S1    [     0.15         0.85    ]
    W2    [     0.85         0.15    ]
    S2    [     0.15         0.85    ]
```

**Option B: Small vocabulary (richer)**

Use a small vocabulary of syllables with known stress patterns:

| Token | Stress | Example words containing it |
|-------|--------|----------------------------|
| `da` | unstressed | a-BOUT, be-FORE |
| `the` | unstressed | THE book |
| `to` | unstressed | to GO |
| `DUM` | stressed | a-BOUT, HAP-py |
| `BEAT` | stressed | heart-BEAT |
| `STRONG` | stressed | be STRONG |

Emission probabilities would favor unstressed tokens in W positions, stressed in S positions.

**Recommendation:** Start with Option A (binary) for cleaner visualization, extend to Option B later.

### Initial State Distribution
Start at beginning of line:
```
π = [0.85, 0.05, 0.05, 0.05]  # Strong preference for W1
```

---

## Implementation Tasks

### Task 1: Core HMM Implementation

Create a Python class or set of functions implementing:

```python
class MetricalHMM:
    def __init__(self):
        # Define A (4x4 transition matrix)
        # Define B (4x2 emission matrix for binary case)
        # Define π (initial distribution)
    
    def sample(self, n_steps):
        """Generate a sequence of (hidden_states, observations) of length n_steps"""
        pass
    
    def forward(self, observations):
        """
        Run forward algorithm, returning belief states at each time step.
        
        Returns: 
            beliefs: array of shape (T, 4) where beliefs[t] is the 
                     posterior distribution over states at time t
        """
        pass
    
    def likelihood(self, observations):
        """Compute P(observations) using forward algorithm"""
        pass
```

### Task 2: Belief Geometry Computation

The **belief state** at time t is the posterior distribution over hidden states given observations up to time t:

```
belief[t] = P(S_t | o_1, o_2, ..., o_t)
```

This is a point on the 3-simplex (since we have 4 states). The forward algorithm computes this:

```python
def forward(self, observations):
    T = len(observations)
    beliefs = np.zeros((T, 4))
    
    # Initialize: belief[0] ∝ π * B[:, obs[0]]
    alpha = self.pi * self.B[:, observations[0]]
    alpha = alpha / alpha.sum()  # Normalize
    beliefs[0] = alpha
    
    for t in range(1, T):
        # Predict: alpha_pred = A^T @ alpha (sum over previous states)
        alpha_pred = self.A.T @ alpha
        
        # Update: alpha ∝ alpha_pred * B[:, obs[t]]
        alpha = alpha_pred * self.B[:, observations[t]]
        alpha = alpha / alpha.sum()  # Normalize
        beliefs[t] = alpha
    
    return beliefs
```

### Task 3: Visualization

#### 3.1 Simplex Embedding
The 4-state belief lives on a 3-simplex (tetrahedron). To visualize in 3D, embed the simplex vertices:

```python
# Tetrahedron vertices (centered at origin)
vertices = np.array([
    [1, 1, 1],      # W1
    [1, -1, -1],    # S1
    [-1, 1, -1],    # W2
    [-1, -1, 1]     # S2
]) / np.sqrt(3)

def belief_to_3d(belief):
    """Convert 4D belief vector to 3D point in tetrahedron"""
    return belief @ vertices
```

#### 3.2 Static Visualization
Plot the tetrahedron wireframe with belief trajectory overlaid:

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_belief_trajectory(beliefs, title="Belief Trajectory"):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw tetrahedron edges
    # ... (connect vertices)
    
    # Convert beliefs to 3D
    points_3d = np.array([belief_to_3d(b) for b in beliefs])
    
    # Plot trajectory
    ax.plot(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
            'b-', alpha=0.7, linewidth=1)
    
    # Color by time
    scatter = ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                        c=np.arange(len(beliefs)), cmap='viridis', s=20)
    
    # Label vertices
    for i, (v, name) in enumerate(zip(vertices, ['W1', 'S1', 'W2', 'S2'])):
        ax.text(v[0], v[1], v[2], name, fontsize=12, fontweight='bold')
    
    plt.colorbar(scatter, label='Time step')
    ax.set_title(title)
    return fig, ax
```

#### 3.3 Animation (Optional but Recommended)
Animate the belief point moving through the simplex over time:

```python
from matplotlib.animation import FuncAnimation

def animate_beliefs(beliefs, interval=100):
    # ... setup figure ...
    
    def update(frame):
        # Update point position
        # Update trail
        pass
    
    anim = FuncAnimation(fig, update, frames=len(beliefs), interval=interval)
    return anim
```

#### 3.4 Key Visualizations to Produce

1. **Single trajectory**: Generate ~50-100 observations, plot belief trajectory
   - Should see cyclic pattern (loop through W1→S1→W2→S2→W1)
   - Variations cause deviations from perfect cycle

2. **Multiple trajectories**: Overlay several runs to see the "attractor" structure
   - Beliefs should concentrate in a "tube" around the cycle

3. **Reset dynamics**: Highlight line boundaries (S2→W1 transitions)
   - Show how beliefs "snap back" to W1 region at line ends

4. **Effect of metrical violations**: Compare trajectories for:
   - Perfect meter (stressed/unstressed alternates exactly)
   - Violated meter (unexpected stress pattern)
   - Should see different geometric paths

---

## Expected Results

### Geometric Structure
The belief trajectory should exhibit:

1. **Cyclic attractor**: A loop structure W1 → S1 → W2 → S2 → W1 embedded in the tetrahedron

2. **Tube around cycle**: Due to emission noise and transition noise, beliefs don't follow a perfect 1D loop but spread into a "tube" or "ribbon" around it

3. **Reset concentration**: Beliefs concentrate near the W1 vertex at line boundaries, creating periodic "pinch points" in the trajectory

4. **Phase portrait structure**: If you project onto 2D (e.g., first two principal components), should see limit-cycle-like behavior

### What This Tests in LLMs
The hypothesis is that if you:
1. Generate token sequences from this HMM
2. Feed them to a pretrained LLM
3. Extract hidden state activations at each position
4. Apply dimensionality reduction (PCA, etc.)

...the LLM's internal geometry should **mirror the belief simplex geometry** — showing similar cyclic structure with reset dynamics at line boundaries.

---

## Suggested Notebook Structure

```
1. Setup & Imports
   - numpy, matplotlib, etc.

2. HMM Definition
   - Define transition matrix A
   - Define emission matrix B  
   - Define initial distribution π

3. Core Algorithms
   - Sampling function
   - Forward algorithm for belief computation

4. Simplex Visualization Utilities
   - Tetrahedron embedding
   - Plotting functions

5. Experiments
   5.1 Generate sample sequences, verify they "look like" meter
   5.2 Compute belief trajectories
   5.3 Visualize single trajectory
   5.4 Visualize multiple trajectories (attractor structure)
   5.5 Analyze reset dynamics at line boundaries
   5.6 Compare perfect vs. violated meter

6. (Optional) Animation
   - Animate belief evolution

7. (Optional) Extension to Iambic Tetrameter
   - 8 states instead of 4
   - Richer structure
```

---

## Notes & Gotchas

1. **Normalization**: Always normalize belief vectors after each update to avoid numerical underflow and keep them as valid probability distributions

2. **Log-space**: For long sequences, consider implementing forward algorithm in log-space to avoid underflow

3. **Ergodicity check**: Verify that your transition matrix is ergodic by checking that A^n has all positive entries for some n (or compute eigenvalues — should have unique eigenvalue 1)

4. **Emission design**: The emission probabilities (0.85/0.15) control how "noisy" the observations are. Higher values (0.95/0.05) = cleaner signal, beliefs snap quickly to correct state. Lower values (0.7/0.3) = noisier, beliefs spread out more.

5. **Sequence length**: Generate sequences that are multiples of 4 (complete lines) plus some extra to see multiple line-reset cycles. Suggest 40-100 tokens.

---

## References

- Metrical phonology background: Liberman & Prince (1977), Hayes (1995)
- HMM forward algorithm: Rabiner (1989) "A Tutorial on Hidden Markov Models"
- Belief geometry in transformers: [cite the toy transformer paper you're building on]
- Simplex visualization: Any computational geometry reference

---

## Questions to Resolve During Implementation

1. Should emissions be binary {stressed, unstressed} or a small vocabulary?
2. What transition probabilities best balance "clean cycle" vs. "interesting variations"?
3. How long should sequences be for good visualization?
4. Should we implement Viterbi decoding as well (for comparison)?
