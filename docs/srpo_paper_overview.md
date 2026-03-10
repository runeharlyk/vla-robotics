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

## Additional Findings from SiiRL Repository

The SiiRL training code and launch scripts reveal several practical implementation details that are not explicit in the paper text:

### Reward Pipeline Details

- The embodied reward manager applies reward at the **terminal valid action token** using `finish_step * action_token_len`.
- A `reward_coef` scaling term is present in the reward manager path (example script value: `5.0`).
- Validation verification uses the environment completion signal (`complete`) as success score and reports aggregate success-style metrics.

### Sampling and Filtering Logic

- Embodied training supports prompt-group filtering via `algorithm.filter_groups.enable=True`.
- Accuracy filtering keeps only prompt groups with mean success in a bounded range (example script values: `0.1 <= acc <= 0.9`).
- Optional truncation filtering removes groups that hit `max_steps`, based on `finish_step`.

### Policy Optimization Details

- PPO implementation includes **dual-clip logic** with asymmetric clip ranges (`clip_ratio_low`, `clip_ratio_high`) and an extra `clip_ratio_c` term.
- KL control supports both **fixed** and **adaptive** controllers (`target_kl`, `horizon`).
- GRPO advantage computation in embodied mode uses a finish-step-derived mask over action tokens.

### Script-Level Defaults Observed

- Common SRPO launch settings include: `ppo_epochs=1`, `grad_clip=1.0`, `clip_ratio_low=0.2`, `clip_ratio_high=0.28`, `num_envs=16`, `max_steps=512`, and rollout sampling with `temperature=1.6`.
- Critic is disabled in SRPO runs (`critic.use_critic_model=False`) with `adv_estimator=grpo`.
