```markdown
# SAC Flow: Sample-Efficient Reinforcement Learning of Flow-Based Policies via Velocity-Reparameterized Sequential Modeling

**Authors:** Yixian Zhang\*, Shu'ang Yu\*, Tonghe Zhang, Mo Guang, Haojia Hui, Kaiwen Long, Yu Wang, Chao Yu†, Wenbo Ding†
**Affiliations:** Tsinghua University, Carnegie Mellon University, Li Auto, Shanghai AI Laboratory
**Code:** https://github.com/Elessar123/SAC-FLOW

---

## Problem Statement

Flow-based policies are expressive generative models capable of capturing multimodal action distributions, making them attractive for continuous-control and robotic manipulation tasks.
However, training them with off-policy reinforcement learning (e.g., SAC) is notoriously unstable.
Prior workarounds either distill the flow into a simpler one-step actor or use surrogate objectives that avoid differentiating through the full rollout — both of which blunt the expressive benefits of the flow.

---

## Key Insight

The paper identifies that the Euler integration used during flow rollout is **algebraically equivalent to a residual recurrent computation (residual RNN)**:

```
A_{t_{i+1}} = A_{t_i} + f_θ(t_i, A_{t_i}, s),   where f_θ(·) = Δt_i · v_θ(·)
```

This means backpropagating through K flow steps is equivalent to backpropagating through K RNN layers — making it susceptible to the same **vanishing and exploding gradient** pathologies.

---

## Proposed Method

### Velocity Reparameterizations

Two novel velocity network designs are proposed as drop-in replacements for the standard MLP velocity field:

#### Flow-G (GRU-Gated Velocity)
Introduces a GRU-style update gate to regulate gradient flow:

```
g_i = sigmoid(f_z([s; A_{t_i}; t_i]))
v̂_θ = tanh(f_h([s; A_{t_i}; t_i]))
v_θ = g_i ⊙ (v̂_θ − A_{t_i})
A_{t_{i+1}} = A_{t_i} + Δt_i · (g_i ⊙ (v̂_θ − A_{t_i}))
```

The gate `g_i` adaptively interpolates between retaining the current intermediate action and forming a new one.
Gate head is initialized with `W=0, b=5.0` so that `g_i ≈ 1` at the start of training, acting as near-identity.

#### Flow-T (Transformer-Decoder Velocity)
Parameterizes the velocity using a Transformer decoder conditioned on the environment state via cross-attention:

- Forms separate embeddings: `Φ_A^i = E_A(φ_t(t_i), A_{t_i})` and `Φ_S = E_S(φ_s(s))`
- Applies L pre-norm residual decoder blocks, each with state-only cross-attention (no token-to-token mixing across flow timesteps) and a feed-forward network
- Projects the final token to velocity space: `v_θ = W_o(LN(Φ_A^{(L)}))`

A diagonal mask on self-attention ensures the **Markov property** is preserved — each action-time token is refined independently using global state context, not a causal history.

### Noise-Augmented Rollout for Tractable Likelihoods

SAC requires explicit log-likelihoods for entropy regularization, but the deterministic K-step rollout has intractable densities.
The paper converts the rollout to stochastic by adding isotropic Gaussian noise with a compensating drift:

```
A_{t_{i+1}} = A_{t_i} + b_θ(t_i, A_{t_i}, s)Δt_i + σ_θ√Δt_i · ε_i,   ε_i ~ N(0, I_d)
```

The drift `b_θ` inflates the learned velocity to counteract diffusion while keeping the terminal marginal unchanged.
This induces a factored joint path density `p_c(A | s)` over intermediate actions, giving a tractable per-step Gaussian log-likelihood used as the entropy term in SAC.

### Training Objectives

#### From-Scratch Training (SAC Flow)

**Actor loss:**
```
L_actor(θ) = α log p_c(A^θ | s_h) − Q_ψ(s_h, a_h^θ),   a_h^θ = tanh(A_{t_K}^θ)
```

**Critic loss (TD):**
```
L_critic(ψ) = [Q_ψ(s_h, a_h) − (r_h + γ Q_ψ̄(s_{h+1}, a_{h+1}) − α log p_c(A_{h+1} | s_{h+1}))]²
```

#### Offline-to-Online Training

Adds a proximity regularizer to the actor loss:
```
L_actor^o(θ) = α log p_c(A^θ | s_h) − Q_ψ(s_h, a_h^θ) + β ‖a_h^θ − a_h‖²,   (s_h, a_h) ~ B
```

The pipeline begins with flow-matching pretraining on expert data, then transitions to online learning while maintaining proximity to the replay buffer.

#### Fine-Tuning (Adapter Mode)

Flow-G and Flow-T can be applied as lightweight adapters on top of an arbitrary pre-trained flow policy `v_θ^pre`:

- **Flow-G Adapter:** Freezes `v_θ^pre` as the candidate network; only trains a new gate network `z_θ`, initialized so `g_i ≈ 1` at start
- **Flow-T Adapter:** Replaces the final projection `W_o` with `v_θ^pre`; zero-initializes new cross-attention and FFN blocks; preserves original behavior at initialization
- Uses a warm-up phase (`L_warmup` steps) where only a behavioral cloning loss is used before transitioning to full SAC training

---

## Experimental Setup

### Benchmarks

| Benchmark | Tasks | Reward Type | Setting |
|---|---|---|---|
| MuJoCo | Hopper-v4, Walker2d-v4, HalfCheetah-v4, Ant-v4, Humanoid-v4, HumanoidStandup-v4 | Dense | From-scratch |
| OGBench | cube-double, cube-triple, cube-quadruple (UR-5 arm, multi-object placement) | Sparse | Offline-to-online |
| Robomimic | Lift, Can, Square (multi-human demonstrations, 300 trajectories/task) | Sparse | Offline-to-online |

### Baselines

**From-scratch:**
- SAC (Gaussian policy)
- PPO (Gaussian policy)
- QSM — Q-Score Matching; trains a score function to match Q-value gradients
- DIME — diffusion policy trained via KL divergence minimization against exponentiated critic targets
- FlowRL — flow policy with Wasserstein-2 regularized off-policy training (prior SOTA)

**Offline-to-online:**
- ReinFlow — on-policy PPO fine-tuning of flow policies with noise injection for log-prob computation
- FQL — Flow Q-Learning; distills flow into a one-step policy for off-policy updates
- QC-FQL — extends FQL to action chunking in temporally extended action spaces

---

## Hyperparameters

### Common (From-Scratch)

| Parameter | Value |
|---|---|
| Optimizer | Adam (β₁=0.5 for flow-based methods) |
| Batch size | 512 |
| Replay buffer size | 1×10⁶ |
| Discount factor γ | 0.99 |
| Policy learning rate | 3×10⁻⁴ |
| Critic learning rate | 1×10⁻³ |
| Target network update rate τ | 0.005 (SAC), 1.0 (Flow variants) |
| Learning starts | 50,000 |
| Entropy coefficient α (initial) | 0.2 |
| Target entropy | −dim(A) for SAC, 0 for flow variants |
| Automatic entropy tuning | True |
| Online environment steps | 1×10⁶ |
| Flow sampling steps K | 4 |
| Diffusion denoising steps | 16 (baselines) |

### Architecture Details

**Classic MLP (baseline flow):**
- Input: `[s; A_{t_i}; t_i]`
- Backbone: MLP 256→256, ReLU
- Velocity form: `v_θ = μ_θ([s; A_{t_i}; t_i])`

**Flow-G:**
- Gate network: MLP 128→d_a, Swish activation
- Candidate network: MLP 128→d_a, Swish activation
- Velocity form: `v_θ = g_i ⊙ (50·tanh(v̂) − A_{t_i})`
- Gate head init: W=0, b=5.0
- Log-std clamp: tanh to [−5, 2]

**Flow-T:**
- Transformer hidden dim d=64, heads n_H=4, layers n_L=2 (from-scratch)
- Obs encoder: 32→SiLU→64
- Self-attention: diagonal mask (position-wise only)
- Cross-attention: action token queries shared state embedding
- Log-std clamp: tanh to [−5, 2]

### Offline-to-Online Specific

| Parameter | QC-FQL | Flow-G | Flow-T |
|---|---|---|---|
| Actor backbone | MLP 512×4, GELU | MLP 512×4 + gate (h=256, swish) | Decoder n_L=2, d=128, n_H=4 |
| Flow/denoising steps K | 10 | 4 | 4 |
| Sampling noise std | deterministic | 0.10 | 0.10 |
| Action chunking | True | True | True |
| Batch size | 256 | 256 | 256 |
| LR | 3×10⁻⁴ | 3×10⁻⁴ | 3×10⁻⁴ |
| τ | 0.005 | 0.005 | 0.005 |
| SAC target entropy | N/A | 0 | 0 |

### Regularization Parameter β (Offline-to-Online)

Format: `offline / online`

| Environment | FQL | QC-FQL | Flow-G | Flow-T |
|---|---|---|---|---|
| cube-double | 300 | 300 | 300 | 300 |
| cube-triple | 300 | 100 | 100 | 100 |
| cube-quadruple | 300 | 100 | 100 | 100 |
| lift | 10000 | 10000 | 10000/1000 | 10000/1000 |
| can | 10000 | 10000 | 10000/1000 | 10000/1000 |
| square | 10000 | 10000 | 10000/1000 | 10000/1000 |

---

## Results

### From-Scratch (MuJoCo)

- SAC Flow-G and SAC Flow-T achieve superior or comparable performance to all baselines on most tasks
- Up to **130% improvement** over FlowRL on HumanoidStandup
- Humanoid-v4 is the one exception where the methods do not outperform all baselines
- Gradient norm stays within a maximum variation of 0.29 across rollout steps (vs. exploding norms in the naive baseline)
- PPO is included as a sample-efficiency reference, confirming off-policy methods converge faster

### Offline-to-Online (OGBench, Robomimic)

- All methods trained on 1M offline updates + 1M online steps
- SAC Flow-T achieves rapid convergence in cube-triple and cube-quadruple tasks, with up to **60% higher success rate** than baselines
- On Robomimic, SAC Flow-G and SAC Flow-T perform comparably to QC-FQL due to large β values severely constraining the actor's learning capacity
- Both methods outperform the on-policy baseline ReinFlow under 1M online steps

### Ablation Studies

- **Velocity parameterization:** Naive SAC Flow (MLP velocity) exhibits exploding gradients; gradient norm escalates sharply from step k=3 to k=0; SAC Flow-G and Flow-T maintain stable norms throughout
- **Sampling steps robustness:** SAC Flow-T and Flow-G remain robust across K=4, 7, 10 sampling steps; Flow-T is particularly stable

---

## Limitations

- **Humanoid-v4 performance:** The method does not surpass all baselines on the full Humanoid locomotion task from scratch
- **Sparse reward from-scratch:** All from-scratch methods (including SAC Flow) struggle on tasks with large exploration spaces and sparse rewards (Robomimic Can, OGBench Cube-Double), motivating the offline-to-online setting
- **Robomimic offline-to-online:** Performance is constrained by the need for a large β regularization coefficient, which restricts the expressivity advantage of flow-based policies, making them behave similarly to the simpler one-step policy in QC-FQL
- **Real-robot validation:** Experiments are conducted entirely in simulation; real-robot transfer and sim-to-real robustness have not yet been validated
- **Lighter sequential models:** The authors note that lighter sequential parameterizations beyond GRU and Transformer have not yet been explored

---

## Broader Context and Distinctions

- Compared to **FQL / QC-FQL:** SAC Flow directly fine-tunes the flow model end-to-end without distilling into an auxiliary one-step policy
- Compared to **FlowRL:** SAC Flow directly maximizes the SAC objective rather than using a surrogate Wasserstein-2 constrained objective
- Compared to **ReinFlow:** SAC Flow is off-policy and therefore significantly more sample-efficient
- The noise-augmented rollout is theoretically grounded: the compensating drift preserves the terminal marginal while making the per-step transitions Gaussian and tractable

---

## Future Work (Stated by Authors)

- Validate SAC Flow on real robots
- Explore lighter sequential parameterizations for the velocity network
- Study sim-to-real robustness for reliable deployment
```