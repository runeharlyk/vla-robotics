# SRPO: Self-Referential Policy Optimization — Paper Overview

## Overview

SRPO (Self-Referential Policy Optimization) is a VLA reinforcement learning post-training framework that replaces sparse binary (0/1) task-success rewards with dense, zero-shot transferable progress-wise rewards derived from a pre-trained latent world model. It builds on GRPO-style policy optimization and is designed to fully exploit information from failed trajectories.

---

## Method

### World Progress Reward

1. **Rollout collection**: Both successful and failed trajectories are gathered into a *Rollout Reference Set*.

2. **Encoding**: All trajectories are encoded by a world model encoder $\mathcal{W}$ into latent representations:
$$h_i = \mathcal{W}(o_{0:T}^{(i)})$$

3. **Clustering successful references**: The latent embeddings of *successful* trajectories are clustered with **DBSCAN** to obtain a set of representative cluster centers $C$:
$$C = \text{DBSCAN}(\mathcal{S})$$
where $\mathcal{S} = \{o_{0:T}^{(i)}; R(z_{0:T}^{(i)}, l) = 1, \forall i\}$ is the set of successful trajectory observations.

4. **Distance to success**: For each trajectory, compute the L2 distance to the nearest cluster center:
$$d_i = \min\left(\{ \|h_i - h_j\|_2 \;;\; h_j \in C \}\right)$$

5. **Reward assignment**:
$$g_i = \begin{cases} 1.0 & \text{for successful trajectory} \\ \alpha \cdot \sigma\!\left(\dfrac{d_i - \bar{d}}{\sigma_d}\right) & \text{for failed trajectory} \end{cases}$$
where $\sigma(\cdot)$ is the sigmoid activation (maps output to $(0, 1)$), $\alpha$ is a scaling coefficient controlling the trade-off between progress awareness and outcome correctness (optimal value $\alpha = 0.8$), and $\bar{d}$, $\sigma_d$ are the mean and standard deviation of distances across all failed trajectories.

#### Reward Embedding Computation (Appendix B)

For **success trajectories**, embeddings are computed via a **cumulative sliding window**: starting from frames $0$–$10$, the window is progressively extended ($1$–$11$, $1$–$12$, …, $1$–$(T{-}1)$), producing a sequence of embeddings representing cumulative visual context up to each time step. The L2 distance of each window embedding to the embedding of the entire video is computed, then normalized so the frame with maximum distance = 0 and minimum distance = 1.

For **failure trajectories**, the minimum L2 distance from each trajectory segment to the success cluster centers is computed, then normalized so the segment with maximum distance = 0 and the segment closest to success = 1.

### Policy Optimization (SRPO)

Follows the GRPO framework. Given a group of $M$ rollouts:

- **Probability ratio**: $r_{i,t}(\theta) = \dfrac{\pi_\theta(a_t^{(i)} \mid o_t^{(i)}, l)}{\pi_{\theta_\text{old}}(a_t^{(i)} \mid o_t^{(i)}, l)}$

- **Advantage** (group-normalized world progress rewards): $\hat{A}_i = \dfrac{g_i - \mu_g}{\sigma_g}$

- **KL regularization**: $\omega(\theta) = \beta \, D_\text{KL}(\pi_\theta \| \pi_\text{ref})$

- **Clipped surrogate objective**:
$$\mathcal{L}_{t,i}^\text{CLIP}(\theta) = \min\!\left(r_{i,t}(\theta)\,\hat{A}_i,\; \text{clip}(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon)\,\hat{A}_i\right)$$

- **Overall objective**:
$$\mathcal{L}_\text{SRPO}(\theta) = \mathbb{E}_{t,i}\left[\mathcal{L}_{t,i}^\text{CLIP}(\theta)\right] + \omega(\theta)$$

Group statistics for reward normalization:
$$\mu_{\hat{R}} = \frac{1}{M}\sum_{j=1}^{M} \hat{R}_j, \qquad \sigma_{\hat{R}} = \sqrt{\frac{1}{M}\sum_{j=1}^{M}(\hat{R}_j - \mu_{\hat{R}})^2 + \epsilon}$$

---

## Hyperparameters

### SFT Stage (Appendix F.1)

Training on $8 \times$ A100 GPUs, initialized from the OpenVLA (with action chunking and parallel decoding) checkpoint.

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | $5 \times 10^{-4}$ |
| Batch size | 8 |
| Max training steps | 150,005 |
| LR decay | Applied after 100,000 steps |
| LoRA rank | 32 |
| Image augmentation | Enabled |
| Input modalities | Text instruction + single image observation |
| Action chunking | 8 action chunks |

### SRPO Post-Training Stage (Appendix F.2)

Initialized from the one-shot SFT model.

| Hyperparameter | Value |
|---|---|
| Algorithm | SRPO (GRPO-based advantage estimation) |
| Learning rate | $5 \times 10^{-6}$ |
| Samples per group ($M$) | 8 |
| Batch size | 64 ($\times 8$ training), 496 (validation) |
| Mini-batch size | 128 |
| Progress reward weight | 0.8 |
| Number of trials per task | 50 |
| Action tokens | 7 |
| Action chunks | 8 |
| Context length | 512 tokens (prompt), 128 tokens (response) |
| Trajectory mini-batch size | 16 |
| Log-probability batch size | 8 (rollout and reference model) |
| FP16 inference | Enabled for the video embedding model |
| Model offloading | Enabled |

### Model Architecture Notes (Appendix F.3)

- Retains the **Llama 2** backbone for generating **discrete action tokens** (as opposed to the continuous action heads in OpenVLA-OFT).
- Discrete token output provides direct access to action log-probabilities, which is essential for policy gradient methods.
- This may sacrifice some action precision relative to continuous output methods.

### Progress Reward Weight $\alpha$ (Appendix D)

The activation function $\phi(\cdot)$ in the reward equation is a sigmoid preceded by a scaling coefficient $\alpha$. This controls the balance between progress awareness (failed trajectory credit) and outcome correctness (success = 1.0).

| $\alpha$ | Performance (relative) |
|---|---|
| 0 | Worst — purely binary outcome reward |
| 0.3 | Low |
| 0.5 | Medium |
| **0.8** | **Best — optimal balance** |
| 1.0 | Good but suboptimal — over-emphasis on progress |

The paper sets $\alpha = 0.8$ (listed as "Progress Reward Weight" in F.2). The performance hierarchy is $0 < 0.3 < 0.5 < 1.0 < 0.8$.

### Unreported Hyperparameters

The paper does not specify DBSCAN hyperparameters ($\varepsilon$, `min_samples`), the KL coefficient $\beta$, or the clip ratio $\epsilon$.

---

## World Model

| Property | Detail |
|---|---|
| Model | **V-JEPA 2** (Assran et al., 2025) |
| Pre-training | Large-scale robotics video data |
| Role | Encodes full observation trajectories $o_{0:T}$ into latent representations for reward computation |
| Key property | Task-agnostic; zero-shot transferable — no fine-tuning per task required |

---

## Base Policy Model

| Property | Detail |
|---|---|
| Architecture | **OpenVLA\*** — OpenVLA enhanced with **action chunking** and **parallel decoding** |
| Starting checkpoint | Official OpenVLA pre-trained checkpoint |
| SFT stage | One trajectory per task (one-shot SFT), referred to as **OpenVLA\*-One** |
| Full SFT baseline | Full dataset SFT, referred to as **OpenVLA\*-Full** |
| Policy inputs | Third-person RGB image ($T$) + language goal instruction ($I$) |

---

## Training Pipeline

1. **Supervised Fine-Tuning (SFT)**: Single demonstration per task from the official checkpoint (one-shot SFT → OpenVLA\*-One).
2. **Online RL Post-Training**: SRPO applied on top of the one-shot SFT policy via live environment interaction.
3. **Training framework**: Built on **SiiRL** (Wang et al., 2025c).

An **offline SRPO** variant is also reported, which does not involve online environment interaction.

---

## Simulator / Environment

| Property | Detail |
|---|---|
| Simulator | Physics-based simulator (LIBERO uses MuJoCo-based simulation) |
| State $z_t$ | Object positions, velocities, etc. — not directly observable by the agent |
| Action space | End-effector commands |
| Observation $o_t$ | Third-person RGB image only |
| Goal conditioning | Natural language task description $l$ |

---

## Benchmarks

### LIBERO (Primary)
- **Suites**: Spatial, Object, Goal, Long — 10 tasks each (40 tasks total)
- **Metric**: Task success rate (%)
- **Source**: Liu et al., 2023

### LIBERO-Plus (Generalization)
- **Perturbation dimensions** (7): Camera, Robot-Init, Language, Light, Background, Noise, Layout
- **Source**: Fei et al., 2025a
- Evaluated in both **zero-shot** and **with augmented data** settings

---

## Results

### LIBERO Benchmark (Table 1)

| Model | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|
| OpenVLA\*-One (base) | 63.6 | 54.9 | 59.6 | 17.3 | 48.9 |
| + Offline SRPO | 92.5 | 96.8 | 92.0 | 88.7 | 92.5 |
| + Online SRPO | **98.8** | **100.0** | **99.4** | **98.6** | **99.2** |
| OpenVLA\*-Full | 91.6 | 95.3 | 90.6 | 86.5 | 91.0 |
| SimpleVLA-RL | 98.2 | 98.7 | 98.8 | 91.7 | 96.9 |
| RIPT-VLA | 99.0 | 98.6 | 98.6 | 93.8 | 97.5 |
| RLinf | 99.4 | 99.8 | 98.8 | 94.0 | 98.0 |

### LIBERO-Plus Zero-Shot Generalization (Table 2)

| Model | Total Avg |
|---|---|
| OpenVLA\*-One (base) | 19.4 |
| + Online SRPO | **59.6** (+40.2) |
| OpenVLA\*-Full | 51.1 |
| RIPT-VLA | 68.4 |
| OpenVLA-OFT | 69.6 |

### LIBERO-Plus With Augmented Data

| Model | Total Avg |
|---|---|
| OpenVLA\*-One (base) | 30.7 |
| + Online SRPO | **82.1** (+51.4) |
| OpenVLA\*-Full | 73.0 |
| OpenVLA-OFT+ | 79.5 |

---

## Baseline Comparisons

| Method | Algorithm | Notes |
|---|---|---|
| SimpleVLA-RL (Li et al., 2025) | GRPO | Binary 0/1 reward |
| RIPT-VLA (Tan et al., 2025) | GRPO | Binary 0/1 reward; uses wrist + proprioception |
| RLinf (Zang et al., 2025) | GRPO | Binary 0/1 reward; uses wrist + proprioception |
| VLA-RL (Lu et al., 2025) | PPO | — |
| GRAPE (Zhang et al., 2024) | Trajectory-wise DPO | — |
| TGRPO (Chen et al., 2025b) | GRPO | Task-specific hand-crafted progress reward |
| World-Env (Xiao et al., 2025) | — | World model as simulator |

---

## Reward Comparison (Table 3)

Evaluated on 5 metrics: SC (success correlation), Mono (monotonicity), MMD, JS divergence, SMD.

| Method | SC | Mono | MMD | JS | SMD |
|---|---|---|---|---|---|
| Pixel-level | 0.125 | 0.498 | 0.274 | 0.548 | 2.100 |
| ImageBind | 0.957 | 0.837 | 0.356 | 0.408 | 18.111 |
| **SRPO (V-JEPA 2)** | **0.998** | **0.992** | **0.615** | **0.572** | **188.799** |

V-JEPA 2-based latent rewards produce significantly more monotonic, task-progress-aligned signals compared to pixel-level and general-purpose vision embeddings (ImageBind).

---

## Key Design Choices

- **No task-specific reward engineering** — reward is derived purely from latent trajectory similarity to successful references.
- **No expert demonstrations required** at RL time — successful rollouts from the current policy serve as the self-reference.
- **DBSCAN clustering** of successful trajectory embeddings prevents a single outlier trajectory from dominating the reward signal.
- **Trajectory-level reward** (one scalar $g_i$ per episode) normalized within the group, following GRPO convention.
- **Only third-person RGB + language** as policy input — no wrist camera, proprioception, or depth.

---

## Progress Reward Benchmark (Appendix A)

The benchmark dataset contains **700 successful** and **300 failure** trajectories across diverse task domains, with human-annotated labels. Successful trajectories are selected for approximately linear (non-backtracking) progress. Perturbations include camera viewpoint changes, lighting variations, compound objects, sensor noise, and background distractions.

### Metrics

**Temporal Correlation (SC)** — Spearman's rank correlation between progress values and frame numbers across all $N$ tasks:
$$\rho = \frac{1}{N} \sum_{k=1}^{N} \frac{\sum_{i=1}^{T_k}(x_i^{(k)} - \bar{x}^{(k)})(y_i^{(k)} - \bar{y}^{(k)})}{\sqrt{\sum_{i=1}^{T_k}(x_i^{(k)} - \bar{x}^{(k)})^2 \sum_{i=1}^{T_k}(y_i^{(k)} - \bar{y}^{(k)})^2}}$$
where $x_i^{(k)}$ = frame number, $y_i^{(k)}$ = progress value. Higher absolute value → stronger monotonic relationship.

**Temporal Monotonicity (Mono)** — Average percentage of steps where progress increases:
$$M_\text{mono} = \frac{1}{N} \sum_{k=1}^{N} \frac{1}{T_k - 1} \sum_{t=1}^{T_k - 1} \mathbb{1}(r_{t+1}^{(k)} > r_t^{(k)})$$
Values closer to 1.0 indicate smoother monotonic progression.

**Maximum Mean Discrepancy (MMD)** — Distributional separation between success and failure trajectories in RKHS:
$$\text{MMD} = \frac{1}{N} \sum_{k=1}^{N} \left\| \frac{1}{n_k}\sum_{i=1}^{n_k}\phi(R_{s,k}^{(i)}) - \frac{1}{m_k}\sum_{j=1}^{m_k}\phi(R_{f,k}^{(j)}) \right\|_\mathcal{H}^2$$
Larger values indicate better separation.

**Jensen-Shannon Divergence (JSD)** — Distributional divergence between success and failure:
$$\text{JSD} = \frac{1}{N} \sum_{k=1}^{N} \left[\frac{1}{2}D_\text{KL}(P_\text{success}^{(k)} \| M^{(k)}) + \frac{1}{2}D_\text{KL}(P_\text{failure}^{(k)} \| M^{(k)})\right]$$
where $M^{(k)} = \frac{1}{2}(P_\text{success}^{(k)} + P_\text{failure}^{(k)})$. Values closer to $\ln 2$ indicate maximal discriminability.

**Standardized Mean Difference (SMD)** — Effect size of separation between success and failure means:
$$\text{SMD} = \frac{1}{N} \sum_{k=1}^{N} \frac{\mu_\text{success}^{(k)} - \mu_\text{failure}^{(k)}}{\sigma_\text{pooled}^{(k)}}$$
where $\sigma_\text{pooled}^{(k)} = \sqrt{\dfrac{(n_k - 1)(\sigma_\text{success}^{(k)})^2 + (m_k - 1)(\sigma_\text{failure}^{(k)})^2}{n_k + m_k - 2}}$. Larger absolute values indicate stronger differentiation.

---

## Ablations (Appendix C)

### Self-Referential Mechanism

The self-referential design uses **within-batch successful rollouts** as reference trajectories, updated each iteration as the policy improves.

**Ablation**: Replace within-batch successes with a **fixed set of 50 pre-selected expert trajectories per task**.

Results:
- The fixed-reference variant initially trains **slightly slower** than full SRPO but still significantly outperforms GRPO early on; it then **plateaus**, requiring $\approx 1.4\times$ more training steps and still yielding suboptimal final performance.
- Two causes: (1) progress rewards from fixed trajectories still outperform sparse binary rewards early on; (2) static references fail to provide nuanced progress signals for the diverse rollouts the evolving policy generates, creating a performance ceiling.

### Success Clustering

Two motivations for DBSCAN clustering over nearest-neighbour matching:
1. Tasks can be solved via multiple distinct strategies; a failure should be compared to the **nearest strategy**, not an arbitrary success.
2. Individual success trajectories may contain noisy segments; cluster centroids provide a **more robust, prototypical** distance reference.

**Ablation**: Replace cluster centroid distance with distance to the **single nearest success trajectory**.

Results:
- Comparable efficiency in early training (few successes → clustering advantage is minimal).
- Significant performance gap opens in later stages as the set of successful strategies diversifies and clustering correctly distills prototypical patterns.

---

## Training Efficiency (Section 5.2)

SRPO reaches convergence in very few RL steps per suite: **79** (Spatial), **59** (Object), **103** (Goal), **219** (Long). This is a significant advantage over SFT, which requires tens of thousands of steps.

Compared to GRPO, SRPO achieves a steeper learning curve — especially for long-horizon tasks — because it extracts signal from near-successful failed trajectories rather than discarding them.

---

## Trajectory Exploration (Section 5.3)

A fundamental limitation of imitation learning is confinement to the demonstration data distribution. Analysis of end-effector trajectories on LIBERO-Spatial shows that the SRPO-trained policy:

1. **Explores previously unreachable regions** of the action space not covered by any demonstration.
2. **Generates more dispersed trajectories**, indicating active spatial exploration rather than fitting to specific paths.

Even starting from a single demonstration per task, online RL fine-tuning enables the policy to discover novel grasping positions and approach strategies. This directly explains the generalization gains on LIBERO-Plus over full-shot SFT.

---

## Real-World Experiments (Section 5.4 / Appendix G)

### Setup

Due to safety concerns with online exploration in physical settings, real-world experiments use an **offline RL** paradigm. The approach integrates **Advantage-Weighted Regression (AWR)** with SRPO's progress reward:

- Demonstration data is collected and stored in a trajectory buffer.
- Step-level advantage is computed as incremental progress: $A_{i,t} = D_{i,t} - \mu / \sigma$, where $D_{i,t} = R_{i,t} - R_{i,t-1}$.

Robot: **X-ARM 7**. Policy backbones: **π₀** (diffusion-based) and **π₀-FAST** (autoregressive).

### Tasks

Five manipulation tasks: Put Apple into Plate, Put Pear into Plate, Fold Towels, Clean Whiteboard, Select Poker.

### Results

| Backbone | Avg. improvement over SFT |
|---|---|
| π₀ | +66.8% |
| π₀-FAST | +86.7% |

### Real-World Reward Benchmark (Table 4)

Evaluated on 30 success + 20 failure trajectories per task:

| Task | SC | Mono | MMD | JS | SMD |
|---|---|---|---|---|---|
| Put Apple | 0.987 | 0.975 | 0.589 | 0.562 | 165.3 |
| Put Pear | 0.991 | 0.982 | 0.601 | 0.578 | 172.8 |
| Fold Towel | 0.984 | 0.968 | 0.572 | 0.549 | 158.6 |
| Wipe Board | 0.993 | 0.986 | 0.624 | 0.591 | 181.2 |
| Select Poker | 0.989 | 0.979 | 0.595 | 0.569 | 169.5 |
| **Average** | **0.989** | **0.978** | **0.596** | **0.570** | **169.5** |

Results confirm the progress reward generalizes to real-world manipulation with near-perfect temporal correlation and strong success/failure discriminability, despite the sim-to-real domain shift.

---

## Pixel-Level World Models for Reward Shaping (Appendix E)

An alternative reward shaping strategy uses pixel-level world models (e.g., video generation models like Cosmos-Predict2) to synthesize reference trajectories conditioned on language instructions. The paper evaluated **Cosmos-Predict2-14B** (Ali et al., 2025) zero-shot on LIBERO tasks.

**Findings**:
- Zero-shot generation suffers from poor scene consistency and physically implausible outputs.
- Task-specific SFT on expert demonstrations could mitigate this but is costly and hard to scale.
- This approach is considered **inferior** to SRPO's latent world representation paradigm, which requires no video generation and is more cost-effective and generalizable.

---

## Implementation Analysis: Paper vs siiRL Production Code

We performed a detailed code audit of the siiRL repository (`sii-research/siiRL`) — the official training framework referenced in the paper — and compared the three sources of truth: the paper text, the siiRL production code, and our local reimplementation.
The sections below document every detail needed to reconstruct the method.

### V-JEPA 2 Embedding: Exact Computation

#### What the paper says (Section 3.2, Equation 2)

A world-model encoder $\mathcal{W}$ encodes each trajectory's full observation sequence into a single embedding vector:

$$h_i = \mathcal{W}(o_{0:T}^{(i)})$$

No further details about frame sampling, pooling, or normalization are given.

#### What siiRL actually does (`siirl/utils/embodied/video_emb.py`)

The `VideoEmbeddingModel` class loads V-JEPA 2 as a `vit_giant_xformers_rope` model (ViT-G architecture from the `vjepa2` repository) with `num_frames=64` and `img_size=384`.

**Step 1 — Frame sampling.**
Exactly 64 frames are selected per trajectory:

- If the trajectory has $\geq 64$ frames: `np.linspace(0, total_frames - 1, num=64, dtype=int)` — evenly spaced indices.
- If fewer than 64 frames: `np.resize(indices, 64)` — the index array is cyclically repeated to fill 64 slots.

**Step 2 — Preprocessing.**
Frames are converted from `(T, H, W, C)` uint8 numpy arrays to `(T, C, H, W)` torch tensors, then passed through a `video_transforms` pipeline:

1. `Resize(short_side_size)` where `short_side_size = int(256.0 / 224 * 384) = 438`, using bilinear interpolation.
2. `CenterCrop(384, 384)`.
3. `ClipToTensor()` — converts to float in $[0, 1]$.
4. `Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))` — ImageNet normalization.

**Step 3 — Forward pass.**
The 64-frame clip is passed as a single video input `(1, C, T, H, W)` through the V-JEPA 2 encoder.

**Step 4 — Pooling.**
The output is `embedding.mean(dim=1)` — mean-pooling over all spatial + temporal patch tokens.
This produces one 1536-dimensional vector per trajectory.

**Key detail — the embedding is produced during rollout**, not during reward computation.
The `EmbodiedHFRollout` class calls `self.embedding_model.get_embeddings(batch_names, batch_frames)` at the end of each rollout chunk, storing the result as `vjepa_embedding` in the `TensorDict` batch.
The reward function then reads these pre-computed embeddings — it never touches the V-JEPA model itself.

#### Summary: one trajectory = one 1536-dim vector

Each trajectory is reduced to a single point in $\mathbb{R}^{1536}$ via:
*64 evenly-sampled frames → ImageNet-normalised 384×384 centre crops → V-JEPA 2 ViT-G → mean pool over patch tokens.*

---

### DBSCAN Clustering: Exact Computation

#### What the paper says (Section 3.2, Equation 3)

$$C = \text{DBSCAN}(\mathcal{S})$$

No hyperparameters, preprocessing, or fallback behaviour are specified.

#### What siiRL actually does (`siirl/utils/reward_score/embodied.py`)

The `_compute_cluster_centers` function performs three steps:

**Step 1 — StandardScaler.**
Before DBSCAN, embeddings are standardised to zero mean and unit variance per dimension:

```python
scaler = StandardScaler()
scaled_embeddings = scaler.fit_transform(embeddings)
```

This is critical because V-JEPA 2 ViT-G produces 1536-dim embeddings where different dimensions can have very different magnitudes.
Standardisation ensures DBSCAN's $\varepsilon$ parameter is meaningful across all dimensions.

**Step 2 — DBSCAN.**
DBSCAN runs on the scaled embeddings with fixed hyperparameters:

- $\varepsilon = 0.5$ (on the standardised space)
- `min_samples = 2`
- Metric: Euclidean

```python
clustering = DBSCAN(eps=0.5, min_samples=2).fit(scaled_embeddings)
```

**Step 3 — Cluster centres.**
For each non-noise cluster label, the mean of the scaled points is computed and then **inverse-transformed** back to original embedding space:

```python
for label in set(clustering.labels_) - {-1}:
    cluster_points = scaled_embeddings[clustering.labels_ == label]
    center = scaler.inverse_transform(
        cluster_points.mean(axis=0, keepdims=True)
    ).flatten()
    cluster_centers.append(center)
```

**Fallback — if DBSCAN finds no clusters** (all points classified as noise):

```python
if not cluster_centers:
    cluster_centers = [embeddings.mean(axis=0)]
```

A single centre equal to the mean of all success embeddings is used.

#### Key detail: clustering is per-task

The reward function groups all samples by `task_name` (extracted from `task_file_name`), then runs clustering independently per task.
Only successful trajectories within that task form the reference set.
This means cluster centres are task-specific, not shared across tasks.

---

### Distance Computation and Reward: Exact Steps

#### What the paper says (Equations 4–5)

$$d_i = \min\left(\{ \|h_i - h_j\|_2 \;;\; h_j \in C \}\right)$$

$$g_i = \begin{cases} 1.0 & \text{success} \\ \phi\!\left(\dfrac{d_i - \bar{d}}{\sigma_d}\right) & \text{failure} \end{cases}$$

where $\phi(\cdot)$ is sigmoid, $\bar{d}$ and $\sigma_d$ are the mean/std of failed trajectory distances.

#### What siiRL actually does (`siirl/utils/reward_score/embodied.py`)

The production code uses a substantially different normalisation and activation than described in the paper.

**Step 1 — Distance matrix.**
Euclidean distance from each failed trajectory embedding to every cluster centre (in original, unscaled space):

```python
distance_matrix = cdist(fail_emb, cluster_centers, "euclidean")
min_distances = distance_matrix.min(axis=1)
```

**Step 2 — Min-max normalisation (NOT z-score).**
Distances are normalised to $[0, 1]$ using min-max within the current task's failed batch:

```python
min_dist, max_dist = min_distances.min(), min_distances.max()
dist_range = max_dist - min_dist
if dist_range < 1e-6:
    normalized_dists = np.full_like(min_distances, 0.5)
else:
    normalized_dists = (min_distances - min_dist) / dist_range
```

**Step 3 — Sigmoid mapping with fixed parameters.**

```python
sigmoid_steepness = 10.0
sigmoid_offset = 0.5
sigmoid_inputs = sigmoid_steepness * (sigmoid_offset - normalized_dists)
reward_values = 0.6 * special.expit(sigmoid_inputs)
```

This maps:
- A failed trajectory at the minimum distance (closest to success): `sigmoid(10 * 0.5) ≈ 0.6 * 0.993 ≈ 0.596`
- A failed trajectory at the maximum distance (furthest from success): `sigmoid(10 * -0.5) ≈ 0.6 * 0.007 ≈ 0.004`
- A failed trajectory at the midpoint: `sigmoid(0) = 0.6 * 0.5 = 0.3`

**Step 4 — Final reward assembly.**

```python
final_rewards[success_indices] = 1.0
final_rewards[fail_indices] = reward_values  # in [0, 0.6]
```

Failed trajectory rewards are capped at 0.6 — there is no configurable $\alpha$ parameter in the production code.

---

### Paper vs siiRL Code: Differences Summary

| Aspect | Paper (Section 3.2) | siiRL Production Code |
|---|---|---|
| **Embedding granularity** | One vector per trajectory (implied) | One vector per trajectory (confirmed) |
| **Frame sampling** | Not specified | Exactly 64 frames, evenly spaced (`np.linspace`) |
| **Preprocessing** | Not specified | Resize → CenterCrop(384) → ImageNet normalisation |
| **Pooling** | Not specified | Mean over all patch tokens (`embedding.mean(dim=1)`) |
| **Pre-DBSCAN scaling** | Not mentioned | `StandardScaler` (zero mean, unit variance) — critical |
| **DBSCAN $\varepsilon$** | Not specified | 0.5 (on standardised space) |
| **DBSCAN `min_samples`** | Not specified | 2 |
| **DBSCAN fallback** | Not mentioned | Mean of all success embeddings if no clusters found |
| **Distance normalisation** | Z-score: $(d_i - \bar{d}) / \sigma_d$ | Min-max: $(d_i - d_\text{min}) / (d_\text{max} - d_\text{min})$ |
| **Activation** | $\phi(\cdot)$ — sigmoid | $0.6 \times \text{sigmoid}(10 \times (0.5 - d_\text{norm}))$ |
| **Reward range (failures)** | $(0, \alpha)$ with $\alpha=0.8$ | $(0, 0.6)$ — hardcoded |
| **$\alpha$ parameter** | 0.8, ablated in Appendix D | Not present — the 0.6 cap is baked in |
| **Clustering scope** | "Successful trajectories" (ambiguous) | Per-task: only successes for the same task |
| **Self-referential set** | In-batch successes | In-batch successes (confirmed — no persistent reference set across iterations) |
| **Distributed handling** | Not discussed | All-gather across DP ranks, rank-0 computes, broadcast |

### Key Insight: The StandardScaler Is Essential

The paper describes DBSCAN as if it runs directly on raw embeddings.
The production code reveals that **standardisation before DBSCAN is critical** for V-JEPA 2 embeddings.
Without it, the fixed $\varepsilon = 0.5$ would be meaningless in the raw 1536-dim space where L2 distances can be much larger.

The scaler fits on the current batch's success embeddings, so the $\varepsilon$ threshold adapts implicitly to the embedding magnitude — which varies across tasks and training stages.

### Key Insight: Min-Max vs Z-Score Changes Reward Semantics

The paper's z-score normalisation (`(d - mean) / std`) produces a distribution centred around 0, then sigmoid maps it.
This means roughly half of failed trajectories get rewards above `sigmoid(0) * α = 0.4` and half below.

The production code's min-max normalisation (`(d - min) / (max - min)`) compresses all distances to $[0, 1]$, then the fixed sigmoid curve assigns:
- The closest failure: reward ≈ 0.596
- The furthest failure: reward ≈ 0.004

This gives a wider reward spread and is less sensitive to outlier distances than z-score.

---

## Our Local Reimplementation: Deviations from siiRL

Our codebase (`vla/rl/srpo_reward.py`) reimplements the SRPO reward with several intentional and unintentional differences from the siiRL production code.

### Embedding: Per-Frame vs Per-Trajectory

| | siiRL Production | Our Code (Reward Path) | Our Code (Diagnostics) |
|---|---|---|---|
| **Granularity** | 1 embedding per trajectory | Per-frame embeddings, grouped by trajectory | 1 embedding per trajectory |
| **Frame sampling** | 64 evenly-spaced frames | Every $k$-th frame (`subsample_every`, default 5) | Every $k$-th frame |
| **Encoding** | 64-frame video clip → V-JEPA 2 → mean pool | Each frame independently → `encode_frames` | 64-frame clip (transformers) or mega-batch (timm) |

Our reward code in `_encode_trajectories_per_frame` encodes each subsampled frame **independently** through `encode_frames`, producing a list of per-frame `(D,)` vectors per trajectory.
The siiRL code encodes the full 64-frame clip as a single video, getting temporal context from V-JEPA 2's spatiotemporal attention.

This means our per-frame embeddings lack temporal context that V-JEPA 2 is designed to exploit.

### Distance: Per-Frame Aggregation

Our code computes the distance $d_i$ differently:

```python
frame_stack = torch.stack(frame_embs, dim=0)         # (F, D)
per_frame_dists = torch.cdist(frame_stack, centres)   # (F, K)
min_per_frame = per_frame_dists.min(dim=1).values     # (F,) min over clusters
d_i = min_per_frame.mean()                            # scalar: mean over frames
```

For each frame: find the nearest cluster centre.
Then average those per-frame minimum distances across the trajectory.

The siiRL code simply computes `cdist(trajectory_embedding, cluster_centers).min()` — one distance per trajectory.

### DBSCAN: No StandardScaler

Our code runs DBSCAN directly on raw embeddings without standardisation:

```python
db = DBSCAN(eps=eps, min_samples=k, metric="euclidean").fit(X)
```

This means the `eps` parameter has a completely different meaning than in siiRL.
Our code compensates with an auto-eps feature that picks eps from the k-th nearest neighbour distance percentile:

```python
if self.cfg.dbscan_auto_eps and len(X) > k:
    kth_dists = NearestNeighbors(n_neighbors=k).fit(X).kneighbors()[0][:, -1]
    eps = float(np.percentile(kth_dists, self.cfg.dbscan_percentile))
```

### Reward Normalisation: Z-Score (Matches Paper, Not siiRL)

Our code uses the paper's z-score normalisation:

```python
d_mean = d_all.mean()
d_std = d_all.std(correction=0).clamp(min=eps)
normalised = (d_all - d_mean) / d_std
activated = self._activation(normalised) * self.cfg.alpha
```

With `sigmoid(-z_score) * alpha` where `alpha = 0.8`.

### Reference Set: Persistent with Demo Seeding

Our code maintains a **persistent reference set** across training iterations:

- Demo embeddings are added once and never evicted (`_demo_embeddings`).
- Online success embeddings are added with FIFO eviction (`_online_embeddings`, capped at `max_references`).
- Clusters are re-fitted after each insertion.

The siiRL code uses only **in-batch successes** — no persistence across iterations.
The paper describes self-referential learning as using "successful trajectories generated within the current training batch," matching the siiRL approach.

### Deviations Summary

| Aspect | siiRL (Authoritative) | Our Code | Action Needed |
|---|---|---|---|
| StandardScaler before DBSCAN | Yes | No | Should add |
| Per-trajectory vs per-frame distance | Per-trajectory | Per-frame mean | Design choice — per-frame gives denser signal |
| 64-frame video clip encoding | Yes (temporal context) | Per-frame independently | Consider switching to clip mode |
| Reward normalisation | Min-max → fixed sigmoid | Z-score → sigmoid × α | Both valid, different semantics |
| Reward cap | 0.6 | 0.8 (alpha) | Matches paper, not siiRL |
| Persistent reference set | No (in-batch only) | Yes (demo + FIFO online) | Intentional improvement for low-success-rate scenarios |
| Auto-eps for DBSCAN | No (fixed 0.5 on scaled space) | Yes (k-NN percentile) | Good addition — compensates for lack of StandardScaler |

---

## Reconstructing SRPO: Minimal Recipe

Based on the code analysis above, these are the exact steps to reproduce the siiRL production reward:

### 1. Encode Trajectories

```python
# For each trajectory with frames (T, H, W, C) as uint8 numpy arrays:
total_frames = len(frames)
if total_frames >= 64:
    indices = np.linspace(0, total_frames - 1, num=64, dtype=int)
else:
    indices = np.resize(np.arange(total_frames), 64)
sampled = [frames[i] for i in indices]

# Convert to (T, C, H, W) tensor
video_tensor = torch.from_numpy(np.stack(sampled)).permute(0, 3, 1, 2)

# Preprocess: resize, center crop, normalise
short_side = int(256.0 / 224 * 384)  # = 438
transform = Compose([
    Resize(short_side, interpolation="bilinear"),
    CenterCrop((384, 384)),
    ClipToTensor(),  # -> float [0, 1]
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
x = transform(video_tensor).unsqueeze(0)  # (1, C, 64, 384, 384)

# Forward through V-JEPA 2 ViT-G
with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
    output = vjepa2_model(x)  # (1, num_patches, 1536)

embedding = output.mean(dim=1).float().squeeze(0)  # (1536,)
```

### 2. Cluster Successful Embeddings (Per Task)

```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# success_embeddings: (N_success, 1536) numpy array
scaler = StandardScaler()
scaled = scaler.fit_transform(success_embeddings)

db = DBSCAN(eps=0.5, min_samples=2).fit(scaled)

centers = []
for label in set(db.labels_) - {-1}:
    cluster_scaled = scaled[db.labels_ == label]
    center = scaler.inverse_transform(
        cluster_scaled.mean(axis=0, keepdims=True)
    ).flatten()
    centers.append(center)

if not centers:
    centers = [success_embeddings.mean(axis=0)]
centers = np.array(centers)  # (K, 1536)
```

### 3. Compute Rewards for Failed Trajectories

```python
from scipy.spatial.distance import cdist
from scipy.special import expit  # sigmoid

# fail_embeddings: (N_fail, 1536) numpy array
distances = cdist(fail_embeddings, centers, "euclidean")  # (N_fail, K)
min_dists = distances.min(axis=1)                          # (N_fail,)

# Min-max normalise
d_min, d_max = min_dists.min(), min_dists.max()
d_range = d_max - d_min
if d_range < 1e-6:
    normed = np.full_like(min_dists, 0.5)
else:
    normed = (min_dists - d_min) / d_range

# Sigmoid mapping
rewards = 0.6 * expit(10.0 * (0.5 - normed))

# Final: success = 1.0, failure = rewards ∈ [~0, ~0.6]
```

### 4. GRPO Advantage Estimation

```python
# Group rewards by prompt (task + trial):
# For each group of M samples with rewards [r_1, ..., r_M]:
group_mean = mean(rewards_in_group)
group_std = std(rewards_in_group)
advantages = (rewards - group_mean) / (group_std + 1e-6)
```

### 5. PPO Clipped Policy Update

Standard PPO with asymmetric clipping:
- `clip_ratio_low = 0.2`
- `clip_ratio_high = 0.28`
- No critic network
- `ppo_epochs = 1`
- `learning_rate = 5e-6`
- `temperature = 1.6` for rollout sampling

---

## Additional Findings from siiRL Repository

### Reward Pipeline Details

- The embodied reward manager applies reward at the **terminal valid action token** using `finish_step * action_token_len`.
- A `reward_coef` scaling term is present in the reward manager path (example script value: `5.0`).
- Validation verification uses the environment completion signal (`complete`) as success score and reports aggregate success-style metrics.

### Sampling and Filtering Logic

- Accuracy filtering keeps only prompt groups with mean success in a bounded range: `0.1 <= acc <= 0.9`.
- Optional truncation filtering removes groups that hit `max_steps`, based on `finish_step`.
- An `oversample_factor` parameter allows over-sampling to compensate for filtered-out prompts.

### Policy Optimization Details

- PPO implementation includes **dual-clip logic** with asymmetric clip ranges (`clip_ratio_low`, `clip_ratio_high`) and an extra `clip_ratio_c` term for negative advantages.
- GRPO advantage computation in embodied mode uses a finish-step-derived mask over action tokens.
- The response tensor is 3D `(batch, traj_len, action_token_len)` and is flattened for the loss.

### Script-Level Defaults Observed

- Common SRPO launch settings include: `ppo_epochs=1`, `grad_clip=1.0`, `clip_ratio_low=0.2`, `clip_ratio_high=0.28`, `num_envs=16`, `max_steps=512`, and rollout sampling with `temperature=1.6`.
- Critic is disabled in SRPO runs (`critic.use_critic_model=False`) with `adv_estimator=grpo`.

### Distributed Training

- The reward computation supports multi-GPU data-parallel training.
- All ranks participate in `all_gather_object` to collect embeddings from all DP ranks.
- Only rank 0 computes rewards, then broadcasts the result back to all ranks.
- Data is sorted by `dp_rank` after gathering to ensure deterministic ordering.
