# Flow Policy Optimization (FPO): Flow Matching Policy Optimization — Paper Overview

## Overview

Flow Policy Optimization (FPO) is an on-policy reinforcement learning algorithm that brings flow matching into the policy gradient framework.
It casts policy optimization as maximizing an advantage-weighted ratio computed from the conditional flow matching (CFM) loss, in a manner compatible with the PPO-clip framework.
FPO sidesteps the need for exact likelihood computation while preserving the generative capabilities of flow-based models.
Unlike prior diffusion-based RL methods that bind training to a specific sampling procedure, FPO is agnostic to the choice of diffusion or flow integration at both training and inference time.
FPO trains diffusion-style policies from scratch in continuous control tasks and enables flow-based models to capture multimodal action distributions, achieving higher performance than Gaussian policies—particularly in under-conditioned settings.
For an overview of FPO's key ideas, see the accompanying blog post: flowreinforce.github.io

---

## Motivation

Flow-based generative models—particularly diffusion models—excel at modeling continuous distributions in high-dimensional spaces across images, videos, speech, audio, robotics, and molecular dynamics.
Reinforcement learning is widely used as a post-training strategy for aligning foundation models with task-specific goals.
Prior approaches for diffusion-based RL reframe the denoising process as an MDP, binding training to specific sampling methods and extending the credit-assignment horizon.
FPO instead treats the sampling procedure as a black box during rollouts, allowing flexible integration with any sampling approach—deterministic or stochastic, first- or higher-order, with any number of integration steps.
FPO also sidesteps the complex likelihood calculations typically associated with flow-based models by using the flow matching loss as a surrogate for log-likelihood in the policy gradient.

---

## Method

### Policy Gradients and PPO

The goal of RL is to learn a policy $\pi_\theta$ that maximizes expected return.
The vanilla policy gradient increases the likelihood of actions that result in higher rewards:

$$\max_\theta \; \mathbb{E}_{a_t \sim \pi_\theta(a_t \mid o_t)} \left[ \log \pi_\theta(a_t \mid o_t)\, \hat{A}_t \right]$$

where $\hat{A}_t$ is an advantage estimated from rollout rewards and a learned value function.
PPO incorporates a trust region by clipping the likelihood ratio:

$$\max_\theta \; \mathbb{E}_{a_t \sim \pi_{\theta_\text{old}}(a_t \mid o_t)} \left[ \min\!\left( r(\theta)\,\hat{A}_t,\; \text{clip}(r(\theta), 1 - \varepsilon_\text{clip}, 1 + \varepsilon_\text{clip})\,\hat{A}_t \right) \right]$$

where $r(\theta) = \frac{\pi_\theta(a_t \mid o_t)}{\pi_{\theta_\text{old}}(a_t \mid o_t)}$.
PPO requires exact likelihoods for sampled actions, which are tractable for Gaussian policies but computationally prohibitive for flow matching and diffusion models.

### Conditional Flow Matching (CFM)

Flow matching learns a vector field that transports samples from a tractable prior (e.g., Gaussian noise) to the target data distribution.
The CFM objective trains the model to denoise data perturbed with Gaussian noise.
Given data $x$ and noise $\epsilon \sim \mathcal{N}(0, I)$:

$$\mathcal{L}_{\text{CFM},\theta} = \mathbb{E}_{\tau, q(x), p_\tau(x_\tau \mid x)} \left\| \hat{v}_\theta(x_\tau, \tau) - u(x_\tau, \tau \mid x) \right\|_2^2$$

where $x_\tau = \alpha_\tau x + \sigma_\tau \epsilon$ is the partially noised sample at flow step $\tau$, $\hat{v}_\theta(x_\tau, \tau)$ is the model's velocity estimate, and $u(x_\tau, \tau \mid x)$ is the conditional flow $x - \epsilon$.
The model can also estimate the denoised sample $x$ or noise $\epsilon$ instead of velocity; these formulations are mathematically equivalent under appropriate reweighting.

### Flow Policy Optimization (FPO)

FPO replaces the Gaussian likelihood ratio in PPO with a proxy $\hat{r}_\text{FPO}$ based on the flow matching loss:

$$\max_\theta \; \mathbb{E}_{a_t \sim \pi_\theta(a_t \mid o_t)} \left[ \min\!\left( \hat{r}_\text{FPO}(\theta)\,\hat{A}_t,\; \text{clip}(\hat{r}_\text{FPO}(\theta), 1 - \varepsilon_\text{clip}, 1 + \varepsilon_\text{clip})\,\hat{A}_t \right) \right]$$

The FPO ratio is:

$$\hat{r}_\text{FPO}(\theta) = \exp\!\left( \hat{\mathcal{L}}_{\text{CFM},\theta_\text{old}}(a_t; o_t) - \hat{\mathcal{L}}_{\text{CFM},\theta}(a_t; o_t) \right)$$

For a given action-observation pair, $\hat{\mathcal{L}}_{\text{CFM},\theta}(a_t; o_t)$ is a Monte Carlo estimate of the per-sample CFM loss:

$$\hat{\mathcal{L}}_{\text{CFM},\theta}(a_t; o_t) = \frac{1}{N_\text{mc}} \sum_{i}^{N_\text{mc}} \ell_\theta(\tau_i, \epsilon_i)$$

$$\ell_\theta(\tau_i, \epsilon_i) = \left\| \hat{v}_\theta(a_t^{\tau_i}, \tau_i; o_t) - (a_t - \epsilon_i) \right\|_2^2$$

$$a_t^{\tau_i} = \alpha_{\tau_i} a_t + \sigma_{\tau_i} \epsilon_i$$

where $\tau_i \in [0,1]$ and $\epsilon_i \sim \mathcal{N}(0, I)$.
The same $(\tau_i, \epsilon_i)$ samples are used for both $\hat{\mathcal{L}}_{\text{CFM},\theta}$ and $\hat{\mathcal{L}}_{\text{CFM},\theta_\text{old}}$.

### Theoretical Foundation

The FPO ratio derives from the relationship between the weighted denoising loss $\mathcal{L}_\theta^w$ and the ELBO (Kingma & Gao, 2023).
For the diffusion schedule with constant weight $w(\lambda_\tau) = 1$:

$$\mathcal{L}_\theta^w(a_t) = -\text{ELBO}_\theta(a_t) + c$$

where $c$ is constant w.r.t. $\theta$.
Thus:

$$r_\theta^\text{FPO} = \frac{\exp(\text{ELBO}_\theta(a_t \mid o_t))}{\exp(\text{ELBO}_{\theta_\text{old}}(a_t \mid o_t))} = \exp\!\left( \mathcal{L}_{\theta_\text{old}}^w(a_t) - \mathcal{L}_\theta^w(a_t) \right)$$

The ratio decomposes into the standard likelihood ratio and an inverse KL-gap correction term.
Maximizing it increases modeled likelihood while reducing the KL gap.
With a single $(\tau, \epsilon)$ pair, the ratio estimate is an upper bound in expectation (Jensen's inequality), but the gradient is directionally unbiased.
FPO can be trained effectively even with $N_\text{mc} = 1$.

### Algorithm

| Step | Description |
| --- | --- |
| 1 | Collect trajectories using any flow model sampler; compute advantages $\hat{A}_t$ |
| 2 | For each action, store $N_\text{mc}$ timestep-noise pairs $\{(\tau_i, \epsilon_i)\}$ and compute $\ell_\theta(\tau_i, \epsilon_i)$ |
| 3 | $\theta_\text{old} \leftarrow \theta$ |
| 4 | For each optimization epoch: sample mini-batch; for each $(o_t, a_t)$ compute $\hat{r}_\theta$ and $L_\text{FPO}(\theta)$; update $\theta$ |
| 5 | Update value function parameters $\phi$ like standard PPO |

The FPO loss for each sample:

$$L_\text{FPO}(\theta) = \min\!\left( \hat{r}_\theta \hat{A}_t,\; \text{clip}(\hat{r}_\theta, 1 \pm \epsilon)\,\hat{A}_t \right)$$

$$\hat{r}_\theta = \exp\!\left( -\frac{1}{N_\text{mc}} \sum_{i=1}^{N_\text{mc}} \left( \ell_\theta(\tau_i, \epsilon_i) - \ell_{\theta_\text{old}}(\tau_i, \epsilon_i) \right) \right)$$

### Denoising MDP Comparison

Existing algorithms (DDPO, DPPO, Flow-GRPO) reformulate the denoising process as an MDP, treating each denoising step as a Gaussian policy step.
Limitations of this approach:

- **Extended horizon**: Credit assignment is multiplied by the number of denoising steps (typically 10–50).
- **Noise as observation**: Initial noise values are treated as environment observations, increasing problem dimensionality.
- **Sampling rigidity**: Limited to stochastic sampling by construction.

FPO instead uses flow matching as the fundamental primitive, inheriting flexibility from standard flow/diffusion models: deterministic samplers, higher-order integration, and any number of sampling steps.
FPO does not require a custom sampler or extra environment steps.

---

## Hyperparameters

### MuJoCo Playground

| Hyperparameter | Value |
| --- | --- |
| Optimizer | Adam |
| Total environment steps | 60M |
| Batch size | 1024 |
| Updates per batch | 16 |
| Learning rate (FPO, DPPO) | $3 \times 10^{-4}$ |
| Sampling steps | 10 |
| Clip epsilon $\varepsilon_\text{clip}$ (FPO) | 0.05 |
| Clip epsilon (DPPO) | 0.2 |
| Denoising noise $\sigma_t$ (DPPO) | 0.05 |
| MC samples $N_\text{mc}$ (FPO) | 8 (default); 1, 4, 8 swept |

### Humanoid Control (PHC)

| Property | Detail |
| --- | --- |
| Simulator | Isaac Gym |
| Agent | SMPL-based humanoid, 24 actuated joints |
| Policy inputs | Proprioceptive observations + goal information (root, hands, or all joints) |
| Goal conditioning | Full (all joints), root+hands, or root only |
| Reward | Per-joint tracking reward (DeepMimic-style) |
| Terrain randomization | Enabled for rough-terrain experiments |

---

## Base Policy Model

| Property | Detail |
| --- | --- |
| Policy type | Flow-based generative model (diffusion-style) |
| Model output | Velocity $\hat{v}_\theta(a_t^\tau, \tau; o_t)$ (or equivalently $\hat{\epsilon}_\theta$ or $\hat{a}_t$) |
| Conditioning | Observation $o_t$ |
| Sampling | Any flow/diffusion sampler (Euler, higher-order, deterministic, stochastic) |
| GridWorld | Two-layer MLP modeling $p(a_t \mid s, a_t^\tau)$; $a_t \in \mathbb{R}^2$, $s \in \mathbb{R}^2$ |

---

## Experiments

### Domains

1. **GridWorld** (Gymnasium): 25×25 grid with two goal regions; sparse reward; saddle points with multiple optimal actions.
2. **MuJoCo Playground** (Zakka et al., 2025): 10 continuous control tasks from DeepMind Control Suite.
3. **Humanoid Control** (Isaac Gym): PHC setting; motion-capture tracking with full or sparse goal conditioning.

### Baselines

| Method | Algorithm | Notes |
| --- | --- | --- |
| Gaussian PPO | PPO | Standard diagonal Gaussian policy |
| DPPO (Ren et al., 2024) | Denoising MDP + PPO | Treats each denoising step as Gaussian policy |
| FPO | Flow matching + PPO | CFM-based ratio; sampler-agnostic |

---

## Results

### GridWorld Results

- FPO consistently maximizes return.
- At saddle-point states, the learned flow evolves from Gaussian to **bimodal** action distributions.
- Multiple trajectories from the same starting state reach different goals, illustrating multimodal behavior.
- Gaussian PPO reaches goals but exhibits more deterministic behavior, favoring the nearest goal with less trajectory variation.

### MuJoCo Playground Results

- FPO outperforms Gaussian PPO and DPPO on 8 of 10 tasks.
- Average evaluation reward: FPO 759.3 ± 45.3 vs. Gaussian PPO 667.8 ± 66.0 vs. DPPO 652.5 ± 83.7.

### FPO Ablations (Table 1)

| Variant | Avg Reward |
| --- | --- |
| FPO (8 $(\tau,\epsilon)$ pairs, $\epsilon$-MSE, $\varepsilon_\text{clip}=0.05$) | 759.3 ± 45.3 |
| FPO, 1 $(\tau,\epsilon)$ | 691.6 ± 50.3 |
| FPO, 4 $(\tau,\epsilon)$ | 731.2 ± 58.2 |
| FPO, $u$-MSE | 664.6 ± 48.5 |
| FPO, $\varepsilon_\text{clip}=0.1$ | 623.3 ± 76.3 |
| FPO, $\varepsilon_\text{clip}=0.2$ | 526.4 ± 76.8 |

Findings: more $(\tau,\epsilon)$ samples improve learning; $\epsilon$-MSE outperforms $u$-MSE (velocity MSE) in Playground; clipping choice significantly impacts performance.

### Humanoid Control Results

| Conditioning | Gaussian PPO Success | FPO Success | Gaussian PPO MPJPE | FPO MPJPE |
| --- | --- | --- | --- | --- |
| All joints | 98.7% | 96.4% | 31.62 | 41.98 |
| Root + Hands | 46.5% | **70.6%** | 97.65 | **62.91** |
| Root only | 29.8% | **54.3%** | 123.70 | **73.55** |

- With full joint conditioning, FPO is close to Gaussian PPO.
- Under sparse conditioning (root or root+hands), FPO substantially outperforms Gaussian PPO.
- FPO enables single-stage training of under-conditioned policies; Gaussian policies struggle to learn viable walking.
- FPO trained with terrain randomization walks stably across procedurally generated rough ground.

---

## Key Design Choices

- **Flow matching loss as likelihood proxy** — avoids exact likelihood computation; uses CFM objective as surrogate for log-likelihood in policy gradient.
- **Sampler-agnostic** — training and inference can use any diffusion/flow integration (deterministic, stochastic, first-order, higher-order, variable steps).
- **PPO-compatible** — drop-in replacement for Gaussian policies; works with GAE, GRPO, and standard actor-critic training.
- **$\epsilon$-MSE vs. $u$-MSE** — $\epsilon$-MSE (predict noise, convert to velocity) preferred over direct velocity MSE in MuJoCo; scale-invariant to action scale.
- **Monte Carlo ratio estimation** — $N_\text{mc}$ pairs of $(\tau, \epsilon)$ per action; gradient unbiased even with $N_\text{mc}=1$; more samples improve efficiency.

---

## Discussion and Limitations

### Strengths

- Simple to implement; integrates with standard PPO infrastructure.
- Flow-based policies capture multimodal action distributions.
- Enables training under sparse goal conditioning where Gaussian policies fail.
- Compatible with flow-based mechanisms: sampling, distillation, fine-tuning.

### Limitations

- **Compute**: Flow-based policies are more computationally intensive than Gaussian policies.
- **Lack of established machinery**: No KL divergence estimation for adaptive learning rates; no entropy regularization.
- **Image diffusion fine-tuning**: Applying FPO to pre-trained image diffusion models was unstable—likely due to fine-tuning on self-generated output (Shumailov et al., 2024, 2023; Alemohammad et al., 2024) and sensitivity to classifier-free guidance.
- Instability appears to be a broader challenge in RL for image generation, not specific to FPO.

### Future Work

FPO may offer practical benefits for fine-tuning behavior-cloned diffusion policies in robotics, where flow-based policies are already pretrained and FPO's compatibility and simplicity could enable task-reward fine-tuning.
