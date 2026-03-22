# SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning — Paper Overview

## Overview

SimpleVLA-RL is a VLA reinforcement learning framework that extends the veRL (Volcano Engine Reinforcement Learning) infrastructure to enable end-to-end online RL for embodied manipulation. It applies GRPO-style policy optimization with simple binary outcome rewards (0/1 task success), eliminating the need for hand-crafted process rewards. The framework introduces VLA-specific trajectory sampling, parallelized multi-environment rendering, and three exploration-enhancing strategies — dynamic sampling, clipping asymmetry, and elevated rollout temperature — that together drive consistent 10–15% performance gains over a standard GRPO baseline.

---

## Motivation

Two fundamental challenges limit the SFT-only scaling paradigm for VLA models:

1. **Data Scarcity**: High-quality human-operated robot trajectories are scarce and expensive. Scaling SFT requires large volumes of task-specific demonstrations, which constrains both data scale and diversity.
2. **Poor Generalization**: SFT-trained VLAs rely on limited, scene- and task-specific data and inevitably degrade on unseen tasks, environments, or objects involving distribution shift.

SimpleVLA-RL addresses both challenges by borrowing the RL paradigm from Large Reasoning Models (LRMs) such as DeepSeek-R1: outcome-level rewards with online environment interaction, enabling exploration beyond the demonstration distribution without requiring additional labeled data.

---

## Method

### RL Formulation for VLAs

SimpleVLA-RL formalizes VLA RL as follows:

- **State** $s_t = (o_t^\text{vis}, o_t^\text{prop}, l_\text{task})$: multimodal observations (RGB images, depth, point clouds), proprioceptive information, and language instructions.
- **Action** $a_t \in \mathbb{R}^d$: end-effector delta or joint angle target, decoded from hidden states via an action tokenizer or diffusion expert.
- **Environment**: provides state transitions $s_{t+1} = \text{Env}(s_t, a_t)$ and reward signals. SimpleVLA-RL uses $\alpha = 1$ (pure outcome reward):
$$r(a_{i,t} \mid s_{i,t}) = \begin{cases} 1, & \text{is\_successful}[\text{traj}_i(a_i, s_i)] \\ 0, & \text{otherwise} \end{cases}$$
- **Rollout**: the policy outputs action chunks $(a_t, a_{t+1}, \ldots, a_{t+k-1})$ of length $k$; these are executed in the environment, which returns the next state $s_{t+k}$ for the next inference step. Rollout continues until task completion or maximum episode length.

### Interactive VLA Rollout

Unlike LLM rollout, which is a single forward pass to terminal output, VLA rollout requires continuous closed-loop interaction with the environment: each action alters the physical state, and subsequent action chunks must be conditioned on real-time sensory feedback. SimpleVLA-RL implements parallel environment management — multiple environments are initialized, stepped, and pruned simultaneously, with completed episodes removed from the active pool each step.

The VLA model generates diverse trajectories via temperature sampling on the action token distribution (compatible with PPO-like algorithms because it produces explicit action log-probabilities). Deterministic regression decoders (MLP) and diffusion-based decoders are incompatible with this RL framework.

### Outcome Reward

SimpleVLA-RL uses trajectory-level binary rewards uniformly propagated to all action tokens within a trajectory:

$$R(a_{i,t} \mid s_{i,t}) = \begin{cases} 1, & \text{task successful} \\ 0, & \text{otherwise} \end{cases}$$

This reward is simple, broadly applicable across environments, and free from non-transferable task-specific process reward design.

### Exploration Enhancements

Three modifications are applied to the base GRPO objective to promote exploration:

#### 1. Dynamic Sampling
GRPO computes group-relative advantage estimates. When all trajectories in a group share the same outcome (all success or all failure), advantages collapse to zero and gradients vanish. Dynamic Sampling discards such degenerate groups during rollout, ensuring only groups with **mixed outcomes** are retained for training:
$$0 < \left|\{ \text{traj}_i(a_i, s_i) \mid \text{is\_successful}[\text{traj}_i(a_i, s_i)] \}\right| < G$$
This guarantees non-zero advantage estimates throughout training.

#### 2. Clip Higher (Asymmetric Clipping)
Standard PPO/GRPO clips the importance sampling ratio symmetrically to $[1-\epsilon, 1+\epsilon]$. The upper bound limits the probability increase of low-probability tokens, potentially restricting exploration of novel strategies. Following DAPO, SimpleVLA-RL uses an **asymmetric clipping range** $[1 - \varepsilon_\text{low},\; 1 + \varepsilon_\text{high}]$ with $\varepsilon_\text{low} = 0.2 < \varepsilon_\text{high} = 0.28$, allowing underrepresented tokens to gain probability mass more freely.

#### 3. Higher Rollout Temperature
Increasing the action token sampling temperature from $T = 1.0$ to $T = 1.6$ during rollout increases trajectory diversity and discovery of novel solution strategies — particularly important when the VLA policy's training distribution is homogeneous.

### Training Objective

The full SimpleVLA-RL objective combines modified GRPO clipping with dynamic sampling as a hard constraint, and **removes KL divergence regularization** (following DAPO):

$$\mathcal{J}(\theta) = \mathbb{E}_{s_0 \sim \mathcal{D},\, \{a_t\}_{i=1}^G \sim \pi_{\theta_\text{old}}(\cdot \mid s_t)} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|a_i|} \sum_{t=1}^{|a_i|} \min\!\left( r_{i,t}(\theta)\,\hat{A}_i,\; \text{clip}(r_{i,t}(\theta), 1 - \varepsilon_\text{low}, 1 + \varepsilon_\text{high})\,\hat{A}_i \right) \right]$$

subject to: $0 < \left|\{ \text{traj}_i \mid \text{is\_successful}[\text{traj}_i] \}\right| < G$

where:
$$r_{i,t}(\theta) = \frac{\pi_\theta(a_{i,t} \mid s_{i,t})}{\pi_{\theta_\text{old}}(a_{i,t} \mid s_{i,t})}, \qquad \hat{A}_i = \frac{R_i - \text{mean}(\{R_i\}_{i=1}^G)}{\text{std}(\{R_i\}_{i=1}^G)}$$

Removing KL regularization eliminates the need for a reference model (reducing memory and compute), and removes the constraint that limits exploration of behaviors beyond the reference policy.

---

## Hyperparameters

### RL Training Stage

Training uses $8 \times$ NVIDIA A800 80 GB GPUs with full-parameter training (no LoRA).

| Hyperparameter | Value |
|---|---|
| Algorithm | GRPO (modified: asymmetric clip, no KL, dynamic sampling) |
| Learning rate | $5 \times 10^{-6}$ |
| Training batch size | 64 |
| Sampling count per group ($G$) | 8 |
| Mini-batch size | 128 |
| Clip low ($\varepsilon_\text{low}$) | 0.2 |
| Clip high ($\varepsilon_\text{high}$) | 0.28 |
| Rollout temperature ($T$) | 1.6 |
| Evaluation sampling | Greedy |
| Action chunks | 8 (LIBERO), 25 (RoboTwin 1.0 & 2.0) |
| Total action tokens | 256 |
| Max environment steps | 512 (LIBERO), 200/400/800 (RoboTwin, task-dependent) |
| KL regularization | **Disabled** |

### SFT Stage

SFT is performed before RL with task-specific demonstration data, also using OpenVLA-OFT as the backbone.

| Benchmark | Demonstrations Used |
|---|---|
| LIBERO (full) | 500 per task suite (50 tasks total) |
| LIBERO (one-shot) | 1 per task (10 per suite) |
| RoboTwin 1.0 | 50 per task |
| RoboTwin 2.0 | 1,000 per task |

### Model Architecture Notes

- Uses **OpenVLA-OFT** backbone: LLaMA 2-7B with vision encoders, action chunking, and parallel decoding.
- The **LLaMA 2 output head** generates discrete action tokens using cross-entropy loss; the official MLP continuous action head is replaced. This provides direct access to action log-probabilities required for policy gradient computation, at the cost of some action precision.
- Inputs: single-view RGB image, language instruction, and robot proprioceptive state. **Wrist camera images are excluded** (unlike the official OpenVLA-OFT), reducing input complexity.
- Proprioceptive state is **not used** in LIBERO experiments, but is used in RoboTwin.
- The official OpenVLA-OFT checkpoint cannot be used due to architectural differences; SFT is performed **from scratch** using the same datasets and hyperparameters as the official implementation.

---

## Base Policy Model

| Property | Detail |
|---|---|
| Architecture | **OpenVLA-OFT** — OpenVLA enhanced with action chunking and parallel decoding |
| Backbone | LLaMA 2-7B with SigLIP vision encoder |
| Action decoder | LLaMA 2 output head → discrete action tokens (cross-entropy loss) |
| SFT stage | Full-data SFT or one-shot SFT per task, applied before RL |
| Policy inputs | Single-view RGB image + language instruction (+ proprioception for RoboTwin) |
| Action chunking | 8 (LIBERO), 25 (RoboTwin) |

---

## Training Pipeline

1. **Supervised Fine-Tuning (SFT)**: OpenVLA-OFT is fine-tuned from scratch on task-specific demonstration data (full-trajectory or one-trajectory).
2. **Online RL Post-Training**: SimpleVLA-RL is applied on top of the SFT policy via live simulation interaction, using binary outcome rewards.
3. **Training framework**: Built on **veRL** (Sheng et al., 2024), extended with VLA-specific interactive rollout, parallel multi-environment rendering, and optimized loss computation for action token sequences.

---

## Simulator / Environment

| Property | Detail |
|---|---|
| Simulators | LIBERO (MuJoCo-based), RoboTwin 1.0 & 2.0 |
| Action space | End-effector delta commands (7-DoF: 6-DoF pose + gripper) |
| Observation $o_t$ | RGB image (single third-person view) |
| Proprioception | Joint angles / end-effector pose (used in RoboTwin) |
| Goal conditioning | Natural language task description |
| Reward | Binary task success (1 or 0) — no process rewards |
| Domain randomization | RoboTwin 2.0: clutter, lighting, background, tabletop height, language paraphrasing |

---

## Benchmarks

### LIBERO
- **Suites**: Spatial, Object, Goal, Long — 10 tasks each (40 tasks total)
- **Training data**: 50 expert demonstrations per task (500 per suite) for full SFT; 1 per task for one-shot SFT
- **RL scenarios**: 500 per suite
- **Metric**: Average success rate (SR) across 50 held-out test scenarios per task
- **Source**: Liu et al., 2023

### RoboTwin 1.0
- **Tasks**: 17 bimanual manipulation tasks; 4 evaluated
- **Training data**: 50 demonstrations per task; 100 RL scenarios per task
- **Source**: Mu et al., 2025

### RoboTwin 2.0
- **Tasks**: 50 tasks across multiple robot embodiments; 12 evaluated, spanning 4 horizon levels
- **Robot**: Agilex Piper dual-arm manipulator
- **Training data**: 1,000 demonstrations per task; 1,000 RL scenarios per task
- **Evaluation**: 100 held-out test scenarios per task (3 runs for reproducibility)
- **Domain randomization**: Enabled (clutter, lighting, background, tabletop height, language)
- **Source**: Chen et al., 2025a

#### RoboTwin 2.0 Task Classification

| Horizon Group | Step Range | Avg Steps | Task Count | Example Tasks |
|---|---|---|---|---|
| Short | 112–130 | 121 | 4 | lift_pot, beat_block_hammer, pick_dual_bottles, place_phone_stand |
| Medium | 151–223 | 176 | 4 | move_can_pot, place_a2b_left, place_empty_cup, handover_mic |
| Long | 283–313 | 298 | 2 | handover_block, stack_bowls_two |
| Extra-Long | 466–637 | 552 | 2 | blocks_rank_rgb, put_bottles_dustbin |
| **Overall** | — | **256** | **12** | |

---

## Results

### LIBERO Benchmark (Table 2)

| Model | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|
| Octo | 78.9 | 85.7 | 84.6 | 51.1 | 75.1 |
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| π₀ + FAST | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| π₀ | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| UniVLA | 96.5 | 96.8 | 95.6 | 92.0 | 95.2 |
| OpenVLA-OFT (SFT baseline) | 91.6 | 95.3 | 90.6 | 86.5 | 91.0 |
| **SimpleVLA-RL** | **99.4** | **99.1** | **99.2** | **98.5** | **99.1** |
| Δ over SFT baseline | +7.8 | +3.8 | +8.6 | +12.0 | +8.1 |

### RoboTwin 1.0 (Table 3, 4 tasks)

| Model | Hammer Beat | Block Handover | Blocks Stack | Shoe Place | Avg |
|---|---|---|---|---|---|
| DP | 0.0 | 12.0 | 7.1 | 4.3 | 5.9 |
| DP3 | 64.7 | 84.3 | 24.0 | 59.3 | 58.1 |
| OpenVLA-OFT (SFT) | 67.2 | 61.6 | 7.1 | 23.4 | 39.8 |
| **SimpleVLA-RL** | **92.6** | **89.6** | **40.2** | **59.3** | **70.4** |
| Δ over SFT baseline | +25.4 | +28.0 | +33.1 | +35.9 | +30.6 |

### RoboTwin 2.0 (Table 4, 12 tasks)

| Horizon | π₀ Avg | RDT Avg | OpenVLA-OFT (SFT) Avg | SimpleVLA-RL Avg | Δ |
|---|---|---|---|---|---|
| Short (4 tasks) | 45.5 | 24.5 | 21.3 | **64.9** | +43.6 |
| Medium (4 tasks) | 58.8 | 47.8 | 47.1 | **72.5** | +25.4 |
| Long + Extra-Long (4 tasks) | 43.3 | 27.8 | 46.5 | **69.0** | +22.4 |
| **Overall (12 tasks)** | 49.2 | 33.3 | 38.3 | **68.8** | **+30.5** |

---

## Data Efficiency Analysis (Section 5.1)

SimpleVLA-RL was evaluated under one-shot SFT (1 demonstration per task, 10 per LIBERO suite) to assess RL performance under data scarcity.

### LIBERO: One-Trajectory vs. Full-Trajectory SFT (Table 5)

| Setting | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|
| One-Trajectory SFT | 63.6 | 54.9 | 59.6 | 17.3 | 48.9 |
| One-Trajectory SFT + RL | 98.2 | 98.7 | 98.8 | 91.7 | **96.9** |
| Δ | +34.6 | +43.8 | +39.2 | +74.4 | +48.0 |
| Full-Trajectory SFT | 91.6 | 95.3 | 90.6 | 86.5 | 91.0 |
| Full-Trajectory SFT + RL | **99.4** | **99.1** | **99.2** | **98.5** | **99.1** |
| Δ | +7.8 | +3.8 | +8.6 | +12.0 | +8.1 |

Key finding: **One-Trajectory SFT + RL (96.9%) surpasses Full-Trajectory SFT alone (91.0%)**, closing the gap to Full-Trajectory SFT + RL by 2.2 percentage points. LIBERO-Long improves most dramatically: from 17.3% to 91.7% with only 10 demonstrations total.

---

## Generalization Analysis (Section 5.2)

### Setting

For each of LIBERO-Spatial, LIBERO-Object, and LIBERO-Goal: 9 of 10 tasks are used for training; 1 task is held out as an unseen out-of-distribution evaluation target. Both SFT (trained on 450 seen-task demonstrations) and SimpleVLA-RL (starting from One-Trajectory SFT) are evaluated on unseen tasks as seen-task performance increases.

### Results

- **SFT** achieves >90% on seen tasks but exhibits **severe overfitting** to training task distributions, with catastrophic forgetting on unseen tasks (success rates often collapse to 0%).
- **SimpleVLA-RL** shows **consistent improvement on unseen tasks** across all three dimensions as training progresses.

| Suite | SFT on Unseen | SimpleVLA-RL on Unseen |
|---|---|---|
| LIBERO-Goal (3 tasks) | Drops to 0% immediately | +5% to +15% improvement |
| LIBERO-Object (3 tasks) | Improves on 1 of 3; fails on 2 | Improves on all 3 (+36.5% best) |
| LIBERO-Spatial (3 tasks) | Degrades on all 3 (−10% to −100%) | Improves on all 3 (+7.1% to +28.5%) |

This demonstrates that online RL enables VLA models to **retain previously acquired capabilities** while learning generalizable skills, whereas SFT causes catastrophic forgetting under distribution shift.

---

## Baseline Comparisons

| Method | Algorithm | Notes |
|---|---|---|
| OpenVLA-OFT (Kim et al., 2025) | SFT | Strong full-trajectory SFT baseline |
| π₀ (Black et al., 2024) | Diffusion-based SFT | Flow matching; no RL |
| UniVLA (Bu et al., 2025b) | SFT | Task-centric latent action learning |
| RDT-1B (Liu et al., 2024) | Diffusion SFT | Bimanual manipulation foundation model |
| RIPT-VLA (Tan et al., 2025) | RLOO (RL) | Binary reward; wrist camera + proprioception |
| SRPO (paper concurrent) | GRPO (RL) | Dense progress reward via V-JEPA 2 latent model |
| VLA-RL (Lu et al., 2025) | PPO (RL) | — |
| TGRPO (Chen et al., 2025d) | GRPO (RL) | Trajectory-wise reward via Claude 3.7 evaluation |
| ConRFT (Chen et al., 2025c) | Reinforced Fine-Tuning | Alternating RL + SFT rounds in real world |
| GRAPE (Zhang et al., 2024) | DPO | Preference alignment; no online interaction |

---

## Key Design Choices

- **Binary outcome reward only** — no hand-crafted process rewards or dense reward shaping; scales to any environment with task-success detection.
- **No KL regularization** — eliminates the reference model, reduces memory and compute, and removes the behavioral anchor that would constrain exploration of novel strategies.
- **Asymmetric clip range** ($\varepsilon_\text{low} < \varepsilon_\text{high}$) — allows low-probability tokens (novel actions) to increase in likelihood more freely than high-probability tokens can decrease, biasing the policy toward exploration.
- **Dynamic sampling** — prevents degenerate gradient signals by discarding all-success and all-failure groups; ensures every training batch contains meaningful signal.
- **Token-based action generation** — mandatory for policy gradient computation; diffusion-based and MLP-regression decoders are excluded because they do not provide differentiable action probability distributions.
- **Single third-person RGB + language** — no wrist camera; proprioception added only for RoboTwin.
- **Full-parameter training** — no LoRA; permits complete policy adaptation during RL.

---

## "Pushcut": Emergent Behavior Discovery (Section 6.1)

During RL training on RoboTwin 2.0, the policy spontaneously discovers **novel solution strategies entirely absent from demonstration data**. This phenomenon is named "pushcut" — a push-driven shortcut.

**Examples**:
- **move_can_pot**: All demonstrations use a "grasp–move–place" strategy. After RL training, the policy learns to **push the can directly into the target location** — a simpler and more direct solution.
- **place_a2b_left/right**: Demonstrations grasp Object A and place it beside Object B. The RL policy learns to **push Object A into position** instead.

**Why this emerges**: Sparse binary rewards assign equal credit to grasping and pushing strategies when both achieve task success. This lack of procedural constraint provides the agent a broader exploration space, enabling discovery of unanticipated yet effective solutions.

This is analogous to the "Aha Moment" in DeepSeek-R1 — RL enables qualitative behavioral shifts beyond imitation, whereas SFT is bounded by the strategy space of the demonstrations.

---

## Failure Modes and Limitations (Section 6.2)

### RL Fails Without Initial Task Capability

The base OpenVLA-OFT model (0-trajectory SFT) achieves **0% on all RoboTwin 2.0 tasks** and **remains at 0% after RL**. With no successful trajectories generated during rollout, every trajectory receives zero reward, advantages are identically zero, and gradients vanish. SimpleVLA-RL requires a non-zero initial task success rate to bootstrap.

### Model Prior Determines RL Effectiveness

| SFT Prior | Avg SR Before RL | Avg SR After RL | Gain |
|---|---|---|---|
| 0 trajectories | 0.0% | 0.0% | 0% |
| 100 trajectories | 7.3% | 25.4% | +18.1% |
| 1,000 trajectories | 28.2% | 50.4% | +22.2% |

Stronger initial capabilities provide more effective starting points for exploration. The gains from RL are greater in absolute terms for the better-initialized model.

### Performance Threshold Effect

When initial SR is very low (near 0%), RL yields only marginal gains. In the pick_dual_bottles task: 100-trajectory SFT (1.2% → 4.3%) vs. 1,000-trajectory SFT (29.7% → 68.3%). **A minimum level of task competence is required for effective RL exploration.**

---

## Real-World Experiments (Section 5.3)

### Setup

Sim-to-real transfer experiments with **no real-world training data**. Models are trained purely in simulation using RoboTwin 2.0 scenarios.

- **Robot**: Two AgileX Piper robotic arms (bimanual)
- **Policy backbone**: OpenVLA-OFT (SFT → RL)
- **Evaluation**: 50 real-world trials per task, on clean tabletops with unseen backgrounds

### Tasks

Four RoboTwin 2.0 tasks: Stack Bowls, Handover Block, Pick Bottle, Click Bell.

### Results (Table 6)

| Task | RDT | OpenVLA-OFT (SFT) | SimpleVLA-RL | Δ over SFT |
|---|---|---|---|---|
| Stack Bowls | 60.0 | 38.0 | **70.0** | +32.0 |
| Place Empty Cup | 4.0 | 2.0 | **10.0** | +8.0 |
| Pick Bottle | 10.0 | 0.0 | **14.0** | +14.0 |
| Click Bell | 20.0 | 30.0 | **60.0** | +30.0 |
| **Avg** | 23.5 | 17.5 | **38.5** | **+21.0** |

The Pick Bottle task (requiring high action precision — the bottle falls if not perfectly grasped) shows the most striking result: SFT fails completely while SimpleVLA-RL achieves 14%. This demonstrates that RL training in simulation improves **action precision and robustness**, not merely task-level success rates.
