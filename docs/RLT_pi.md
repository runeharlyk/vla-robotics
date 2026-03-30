# RL Token: Bootstrapping Online RL with Vision-Language-Action Models

## Goal

This paper proposes **RLT**: a way to take a pretrained **vision-language-action model (VLA)** and improve it with **sample-efficient online reinforcement learning** on real robots.

The central problem is:

* VLAs can do many tasks zero-shot or few-shot.
* They often fail at the **last, precision-critical part** of manipulation.
* Full RL fine-tuning of a large VLA is too slow, expensive, and data-hungry for real-world online learning.

The solution is to:

1. extract a compact **RL token** from the VLA,
2. freeze the VLA,
3. train a small **actor-critic** online using the RL token plus VLA action proposals,
4. regularize the RL policy to stay near the VLA policy.

This turns RL into **local refinement of a strong pretrained policy**, rather than learning from scratch.

---

# What the method is for

RLT is designed for tasks where:

* a pretrained VLA already provides a decent initial policy,
* the main remaining difficulty is a **short critical phase** requiring high precision,
* robot interaction data is limited to **minutes or a few hours**,
* reward is sparse,
* human supervision is available for:

  * episode success/failure labels,
  * optional teleoperation interventions,
  * optionally choosing when to switch from base VLA to RL policy during training.

Typical examples in the paper:

* screw installation,
* zip tie fastening,
* Ethernet insertion,
* charger insertion.

These are tasks where:

* demonstrations exist, but are not enough to reach the needed precision/speed,
* failures are often due to tiny alignment or contact errors,
* the policy must improve exactly the part of the behavior where imitation learning saturates.

---

# High-level method

RLT has two stages.

## Stage 1: Adapt the VLA to expose an RL token

A pretrained VLA contains useful internal features, but they are:

* high-dimensional,
* hard to choose from,
* not ideal as-is for lightweight online RL.

The paper adds a small **encoder-decoder transformer** on top of the frozen VLA to compress its internal token sequence into a single compact representation: the **RL token**.

This RL token is trained to be a bottleneck that still allows reconstruction of the original VLA embeddings.

Optionally, during this stage, the VLA is also fine-tuned on task-specific demonstrations.

After this stage:

* the VLA is frozen,
* the RL token module is frozen,
* online RL uses the RL token as state input.

## Stage 2: Online RL on top of the RL token

A lightweight actor and critic are trained online.

Inputs:

* RL token,
* proprioceptive state,
* VLA-proposed action chunk.

The actor learns to **refine** the VLA action chunk rather than generate behavior from scratch.

The critic evaluates chunked actions.

The actor is regularized toward the VLA action chunk so exploration stays local and stable.

---

# Base VLA assumptions

The method is built on **π0.6**, a VLA consisting of:

* a pretrained vision-language model backbone,
* a diffusion-based action expert.

Inputs to the VLA:

* up to 4 camera images,
* language instruction,
* proprioceptive state.

Output of the VLA:

* an **action chunk**
  [
  \tilde{a}_{t:t+H-1} = (\tilde{a}*t, \dots, \tilde{a}*{t+H-1}) \in \mathbb{R}^{H \times d}
  ]

Paper settings:

* (H = 50) actions,
* corresponds to **1 second** at **50 Hz**.

In practice, the robot executes only a prefix of the VLA chunk before replanning.

---

# Core idea: the RL token

## Why it is needed

Direct online RL on the full VLA is a poor fit because:

* the representation is too large,
* updating the full model is too expensive,
* online data is limited,
* real-world RL needs small, fast-to-train networks.

But throwing away the VLA representation loses the main advantage of the pretrained model.

The RL token is meant to preserve:

* task-relevant perceptual structure,
* pretrained knowledge from large-scale robot + web training,
* a compact interface for efficient RL.

## Construction

Let:

* (z = f(s,\ell;\theta_{\text{vla}})) be the final-layer token embeddings of the pretrained VLA,
* (z_{1:M} = {z_1, \dots, z_M}) be the token sequence.

Append a learned RL token embedding:

[
e_{\text{rl}} = e_\phi(\langle rl \rangle)
]

Then pass the sequence through a lightweight encoder transformer (g_\phi):

[
z_{\text{rl}} = g_\phi([z_{1:M}, e_{\text{rl}}])_{M+1}
]

The output at the special token position becomes the **RL token**.

## Training objective

A decoder transformer (d_\phi), followed by a linear output head (h_\phi), autoregressively reconstructs the original VLA embeddings from the RL token.

Using stop-gradient targets (\bar{z}_i = sg(z_i)), the reconstruction loss is:

[
L_{\text{ro}}
=============

\mathbb{E}*D
\left[
\sum*{i=1}^{M}
\left|
\left[h_\phi(d_\phi([z_{\text{rl}}, \bar{z}_{1:i-1}]))\right]_i - \bar{z}_i
\right|_2^2
\right]
]

Interpretation:

* the RL token must contain enough information to let the decoder reconstruct the VLA embeddings,
* so it becomes a compact summary of task-relevant VLA features.

## Important implementation note from the paper

They note that in their experiments:

* each task has a fixed language instruction,
* so they **drop language embeddings** in this step.

But the construction can be applied more generally to all VLA embeddings.

---

# RL formulation

The robot task is modeled as an MDP:

[
(\mathcal{S}, \mathcal{A}, p, r, \gamma)
]

with:

* state space (\mathcal{S}),
* continuous action space (\mathcal{A}),
* dynamics (p(s_{t+1}\mid s_t,a_t)),
* reward (r(s_t,a_t)),
* discount (\gamma\in[0,1)).

## Reward

They use **sparse binary rewards**:

* human supervisor labels episode end as success or failure,
* (r_T = 1) for success,
* (r_T = 0) otherwise.

## Chunked action RL

The RL policy operates on chunks:

[
a_{t:t+C-1} \in \mathbb{R}^{C \times d}
]

where:

* (H) = VLA chunk horizon,
* (C) = RL chunk length,
* they choose (C < H) so the RL policy is more reactive than the VLA.

The chunk-level value target is:

[
Q^\pi(s_t, a_{t:t+C-1})
=======================

\sum_{t'=t}^{t+C-1} \gamma^{t'-t} r_{t'}
+
\gamma^C \mathbb{E}*{a' \sim \pi(\cdot \mid s*{t+C})}
\left[
Q^\pi(s_{t+C}, a')
\right]
]

---

# Online RL design

## State to the RL networks

The RL state is:

[
x = (z_{\text{rl}}, s^p)
]

where:

* (z_{\text{rl}}) = RL token,
* (s^p) = proprioceptive state.

Depending on task, proprio can include:

* joint position,
* joint velocity,
* end-effector pose.

## Critic

The critic estimates:

[
Q_\psi(x, a_{1:C}) \in \mathbb{R}
]

It is trained by off-policy TD learning:

[
L_Q
===

\mathbb{E}*{(x,a*{1:C},x') \sim \mathcal{B}}
\left[
(\hat{Q} - Q_\psi(x,a_{1:C}))^2
\right]
]

with target:

[
\hat{Q}
=======

\sum_{t'=1}^{C} \gamma^{t'-1} r_{t'}
+
\gamma^C
\mathbb{E}*{a' \sim \pi*\theta}
\left[
Q_{\psi'}(x', a')
\right]
]

They follow **TD3** style target networks, and in appendix mention:

* critic ensemble of **two Q-functions**,
* target uses the **minimum** of the two.

## Actor

The actor takes:

* state (x),
* reference VLA chunk (\tilde{a}_{1:C}),

and outputs a Gaussian policy over action chunks:

[
\pi_\theta(a_{1:C} \mid x, \tilde{a}_{1:C})
===========================================

\mathcal{N}
\left(
\mu_\theta(x,\tilde{a}_{1:C}),
\sigma^2 I
\right)
]

### Why condition on the VLA action?

Two reasons given:

1. **local refinement**: RL edits a strong proposal instead of learning from scratch,
2. **mode preservation**: the sampled VLA chunk carries mode information from a multimodal VLA distribution, which a unimodal Gaussian actor would otherwise struggle to recover.

## Actor objective

The actor maximizes Q while staying close to the VLA action:

[
L_\pi(\theta)
=============

\mathbb{E}*{s \sim \mathcal{B},\ a*{1:C} \sim \pi_\theta}
\left[

* Q_\psi(x, a_{1:C})

-

\beta |a_{1:C} - \tilde{a}_{1:C}|_2^2
\right]
]

where:

[
\tilde{a}*{1:C} \sim \pi*{\text{vla}}(\cdot \mid s,\ell)
]

and (\beta) is the strength of the policy constraint toward the VLA.

### Interpretation

This is essentially behavior anchoring around the VLA prior.

Without this term:

* the actor must explore the full chunked action space using only critic gradients,
* which the paper finds hurts performance most among ablations.

---

# Reference action dropout

A failure mode is that the actor might simply copy the VLA chunk and never learn meaningful refinement.

To prevent that, during training:

* for a random subset of transitions,
* the reference action chunk is replaced with zeros before being given to the actor.

This forces the actor to retain an independent action generation pathway.

Appendix detail:

* the actor input reference chunk is **masked out 50% of the time during training**,
* and **always provided at inference**.

---

# Full algorithm

## Stage 1: Train RL token and optionally fine-tune VLA

Train:

* RL token encoder-decoder parameters (\phi),
* optionally VLA parameters (\theta_{\text{vla}}),

using:

[
L_{\text{ro}}(\phi) + \alpha L_{\text{vla}}(\theta_{\text{vla}})
]

where:

* (L_{\text{ro}}) is the RL token reconstruction loss,
* (L_{\text{vla}}) is supervised VLA fine-tuning loss,
* (\alpha) is the VLA fine-tuning weight.

After this step:

* freeze (\phi),
* freeze (\theta_{\text{vla}}).

## Stage 2: Online RL

Initialize:

* replay buffer (\mathcal{B}),
* actor (\pi_\theta),
* critic (Q_\psi).

For each RL chunk boundary:

1. sample VLA reference chunk (\tilde{a}_{t:t+C-1}),
2. extract RL state (x_t = (z_{\text{rl}}(s_t), s^p_t)),
3. choose executed action:

   * human action if intervention,
   * VLA action if still in warmup,
   * RL actor sample otherwise,
4. execute chunk,
5. store transition:
   [
   \langle x_t, a_{t:t+C-1}, \tilde{a}*{t:t+C-1}, r_t, x*{t+1} \rangle
   ]
6. perform (G) gradient updates from replay.

---

# Replay buffer contents

The replay buffer aggregates data from:

* VLA warmup rollouts,
* RL policy rollouts,
* human teleoperation interventions.

A key design point:

* if a human intervenes, the **human action replaces the VLA reference** in the replay buffer.

This means the actor learns from:

* autonomous data,
* corrective human behavior.

---

# Warmup

Before online RL starts, they populate the replay buffer by rolling out the VLA for:

* (N_{\text{warm}}) environment steps.

Purpose:

* gives the critic an initial signal,
* starts RL from competent behavior.

They also mention learning starts shortly after warmup.

---

# Subsampling intermediate chunk transitions

Even though the policy outputs chunks of length (C), the robot still receives observations at every intermediate control step.

So they improve data efficiency by saving extra chunk transitions using stride 2.

Example stored transitions:

* (\langle x_0, a_{0:C}\rangle),
* (\langle x_2, a_{2:C+2}\rangle),
* (\langle x_4, a_{4:C+4}\rangle), etc.

Appendix detail:

* because control is 50 Hz and chunks are subsampled every 2 steps,
* **each second of data produces about 25 RL training samples**.

This is an important practical trick.

---

# Critical-phase training

The method is not used to replace the whole long-horizon policy at first.

Instead, it targets only the **critical phase** of a task:

* insertion,
* fastening,
* rotation,
* precise contact-rich subtask.

Training procedure:

* the base VLA handles the easier early stages,
* a human operator decides when to hand off to the RL policy,
* RL then controls the precision-critical phase,
* episode ends when human marks success/failure.

Why this matters:

* reduces data collection burden,
* concentrates exploration on the part that matters most,
* shortens effective credit assignment.

For autonomous test-time deployment, they mention a final short VLA fine-tuning phase to predict **when to hand over** to RL, using human handoff decisions as labels.

---

# When to apply RLT

RLT is a good fit when these conditions hold.

## Good fit

* You already have a reasonably capable pretrained VLA.
* The task has a short precision-critical segment where the VLA is too slow or inconsistent.
* You can collect at least some demonstrations for task adaptation.
* You can run online robot data collection for minutes to hours.
* You can tolerate sparse rewards and provide success/failure labels.
* Human intervention during training is acceptable.
* Control is continuous and chunked actions are natural.

## Especially good for

* contact-rich manipulation,
* insertion,
* threading,
* fastening,
* tasks requiring exploiting compliance,
* tasks where demonstration data provides a good prior but not optimal behavior.

## Less suitable or not addressed

* tasks where the base VLA is very poor and cannot reach the relevant state distribution,
* problems needing large-scale autonomous reward collection without humans,
* settings where chunked action execution is impossible,
* tasks where no good pretrained VLA prior exists,
* cases where the whole long-horizon task must be learned from scratch online.

---

# Why chunked RL is important

This is one of the most important method choices.

The paper argues that single-step RL methods struggle because:

* task horizons become extremely long at 50 Hz,
* rewards are sparse,
* TD credit assignment becomes very hard,
* running the base VLA at every single control step is infeasible.

RLT instead predicts chunks of length (C), shortening the effective RL horizon.

Paper result:

* replacing chunks with single-step actions performs poorly,
* the single-step variant could not reliably match base policy performance.

---

# Hyperparameters and concrete implementation details

This section collects every implementation-relevant value explicitly stated in the paper and appendix.

## Robot/control setup

* **Control frequency**: 50 Hz
* **Per-step action dimension**: 14
* **Base VLA chunk horizon**: (H = 50)
* **Base VLA chunk duration**: 1 second
* **RL chunk length**: (C = 10)
* **RL actor output dimensionality**:
  [
  10 \times 14 = 140
  ]

## Observation inputs

RL token is produced from:

* **2 wrist camera images**
* **1 base camera image**

Additional proprioception:

* screw task: joint position
* zip tie, Ethernet, charger: end-effector pose

Appendix also states actor/critic receive:

* RL token,
* proprioceptive position,
* velocity.

## Demonstration data

Per task:

* **1 to 10 hours** of teleoperated demonstrations

Used for:

* task-specific VLA fine-tuning,
* RL token training.

## RL token + VLA adaptation stage

Train for:

* **2000 to 10000 gradient steps** on single-task data

Objective:

* RL token reconstruction loss,
* optionally VLA supervised fine-tuning loss.

## Actor/critic networks

For zip tie, Ethernet, charger:

* **2-layer MLP**
* **hidden dimension 256**

For screw installation:

* **3-layer MLP**
* **hidden dimension 512**

## Critic details

* ensemble of **2 Q-functions**
* target uses **minimum** of the two Q-values
* follows **TD3**

## Actor details

* Gaussian policy
* **small fixed standard deviation**
* outputs chunk (a_{t:t+C-1}\in\mathbb{R}^{C\times d})

The paper does **not** provide the actual numeric std value.

## Reference action dropout

* reference chunk masked out at **50%** during training
* reference chunk **always provided** at inference

## Replay subsampling

* stride = **2** control steps

## Update schedule

* **2 critic updates per 1 actor update**
* **update-to-data ratio = 5**

Paper says the high UTD ratio is essential in low-data online RL.

## Reward

* sparse terminal success reward:

  * success = +1
  * failure = 0

In appendix they phrase it as:

* “A sparse +1 reward is provided by the operator during training when the RL task has been completed.”

## Online training duration

Depending on task:

* **400 to 1000 episodes**
* approximately **15 minutes to 5 hours of actual robot data** excluding resets/overhead

For harder full-task settings (screw and zip tie), they report performance after gathering about:

* **5 hours of data**

## Full-task episode lengths

Each full task spans:

* **30–120 seconds**
* roughly **1500–6000 control steps**

Critical phases last:

* **5–20 seconds**
* roughly **250–1000 control steps**

## Warmup

The paper defines a warmup period with (N_{\text{warm}}), but does **not** give a numeric value in the provided text.

## Policy regularization coefficient

The actor uses coefficient (\beta), but the numeric value is **not provided** in the provided text.

## VLA fine-tuning weight

They define (\alpha) for weighting VLA fine-tuning loss during stage 1, but the numeric value is **not provided** in the provided text.

## Discount factor

A discount (\gamma) is used, but the numeric value is **not provided** in the provided text.

## Batch size, optimizer, learning rate, target update rate

These are **not specified** in the provided text.

---

# Complete practical recipe

## Inputs required

To implement RLT, you need:

* a pretrained VLA that outputs chunked actions,
* access to some internal token embeddings from the VLA,
* task-specific demonstration data,
* online robot rollouts,
* human success/failure labels,
* optional teleoperation intervention interface.

## Training pipeline

### 1. Collect task-specific demonstrations

Collect 1–10 hours of teleoperated demonstrations for the target task.

### 2. Add RL token encoder-decoder on top of VLA embeddings

* take final-layer VLA token embeddings,
* append learned RL token,
* run lightweight encoder transformer,
* use RL token output as compressed latent,
* train decoder transformer to reconstruct original embeddings.

### 3. Optionally fine-tune the VLA on the task

Train RL token and optionally VLA jointly on demonstration data:

[
L_{\text{ro}} + \alpha L_{\text{vla}}
]

Train for 2000–10000 gradient steps.

### 4. Freeze VLA and RL token module

No online updates to the VLA backbone during RL.

### 5. Initialize actor and critic from scratch

* small MLP actor,
* twin-Q critic,
* actor uses RL token + proprio + VLA action chunk,
* critic uses RL token + proprio + action chunk.

### 6. Warm up replay with VLA rollouts

Collect initial transitions using the frozen VLA before RL policy acts.

### 7. Train online with off-policy actor-critic

At each chunk boundary:

* query frozen VLA for reference action chunk,
* compute RL token,
* sample RL chunk from actor conditioned on RL token and reference chunk,
* optionally allow human intervention,
* store transition in replay,
* run off-policy updates.

### 8. Use subsampled chunk transitions

Store chunk transitions at stride 2 for more training signal.

### 9. Focus training on the critical phase first

Use base VLA for easy parts, RL only for precision-critical phase.

For harder long-horizon tasks:

* first train RL on isolated critical phase,
* then transition to full-task setting.

### 10. Optional final handoff predictor

Fine-tune the VLA to predict when to switch to RL automatically at test time.

---

# Design choices that matter most

These are the main implementation ideas that appear essential.

## 1. Freeze the VLA during online RL

The whole method depends on keeping online adaptation lightweight and sample-efficient.

The VLA is used as:

* perceptual backbone,
* action prior,
* source of RL token features.

## 2. Use a learned RL token instead of raw VLA features or standard vision encoders

Ablation result:

* replacing RL token with ImageNet-pretrained ResNet-10 reduces throughput by about 50%.

Meaning:

* the token captures manipulation-relevant information unavailable in generic image features.

## 3. Use chunked RL actions

Without chunks:

* horizon is too long,
* sparse reward propagation is too hard,
* performance collapses.

## 4. Condition actor on VLA action and regularize toward it

These two pieces work together:

* conditioning gives access to VLA behavioral prior and action mode,
* regularization stabilizes exploration.

## 5. Use reference-action dropout

Without it, actor may over-copy the reference action.

## 6. High update-to-data ratio

They explicitly say UTD=5 is essential in the low-data regime.

## 7. Human interventions are part of the system

This is not a fully autonomous RL pipeline.

Humans provide:

* sparse reward labels,
* optional corrective actions,
* handoff decisions for critical phase training.

---

# Results relevant to implementation

## Main empirical claims

Across four real-robot tasks, RLT:

* improves success rate,
* improves speed,
* gives up to **3× speedup** on the hardest phase,
* can exceed expert teleoperation speed on one task.

Examples:

* screw insertion success improved from **20% to 65%** in one difficult setting,
* Ethernet task critical phase was around **2× faster than base policy**,
* overall throughput improved across tasks.

## Where gains are largest

The largest gains come in:

* the most contact-sensitive,
* most precision-critical,
* most failure-prone subphases.

This matches the intended use case.

## Comparison to baselines

They compare against:

* HIL-SERL,
* PLD,
* DSRL,
* DAgger.

Findings:

* single-step methods such as HIL-SERL and PLD perform poorly on these sparse long-horizon 50 Hz tasks,
* DAgger can match success on simpler tasks but is limited by human demonstration speed,
* DSRL achieves high success but lower throughput,
* RLT gives the best speed improvement while maintaining or improving success.

---

# Ablations and what they imply

## Without RL token

Replace RL token with ImageNet-pretrained ResNet-10.

Effect:

* significant drop in throughput,
* confirms learned token is useful.

Implementation takeaway:

* do not substitute this with a generic vision encoder unless necessary.

## Without chunking

Set (C=1).

Effect:

* poor performance,
* cannot reliably match base policy,
* also makes RL-token-based implementation infeasible at 50 Hz due to repeated VLA queries.

Implementation takeaway:

* chunking is not optional.

## Without BC regularizer

Set (\beta = 0).

Effect:

* biggest performance drop among ablations.

Implementation takeaway:

* anchoring to the VLA prior is crucial.

## Without pass-through reference action input

Remove (\tilde a) from actor input.

Effect:

* slower learning,
* more failures during training,
* can sometimes recover final performance on simpler tasks, but less reliably.

Implementation takeaway:

* reference action conditioning improves sample efficiency and stability.

---

# Emergent behavior

A notable result is that RLT does not merely imitate humans better.

On Ethernet insertion:

* teleop median critical-phase length: **146** timesteps
* base VLA median: **228**
* final RLT median: **66**

The paper reports the final policy often:

* approaches the port more fluidly,
* inserts decisively,
* uses pressure and slight wiggling to exploit compliance.

This behavior was **not present in demonstrations**, suggesting RLT can discover better-than-demonstration strategies.

---

# What to watch out for when implementing

## 1. You need a decent base VLA

RLT is not intended to rescue a bad base policy from scratch.

The VLA should already:

* understand the scene,
* generate relevant action modes,
* bring the robot into the right neighborhood.

## 2. Online RL should target the bottleneck, not the whole task initially

Critical-phase training is a major practical simplification.

Training on the whole long-horizon task from the start would likely waste samples.

## 3. Sparse rewards are workable only because of the rest of the design

Sparse reward works here because of:

* pretrained VLA prior,
* chunked actions,
* off-policy replay,
* human interventions,
* local exploration around reference chunks.

Remove several of these and learning likely becomes too hard.

## 4. Reference conditioning and regularization are complementary

Conditioning alone may lead to copying.

Regularization alone without explicit conditioning would throw away VLA mode information.

Both are important.

## 5. The actor still needs its own action pathway

This is why the 50% masking of the reference chunk matters.

## 6. Real-world throughput depends on asynchronous rollout + learning

The paper mentions rollout and learning are done asynchronously for time efficiency.

## 7. Some important numeric hyperparameters are missing

The provided text does not specify:

* (\beta),
* (\alpha),
* (\gamma),
* optimizer,
* learning rates,
* batch size,
* target update coefficient,
* warmup length.

An implementation will need to choose these.

---

# Minimal mathematical summary

## RL token extraction

[
z_{\text{rl}} = g_\phi([z_{1:M}, e_{\text{rl}}])_{M+1}
]

Train with reconstruction loss:

[
L_{\text{ro}}
=============

\mathbb{E}*D
\left[
\sum*{i=1}^{M}
\left|
\left[h_\phi(d_\phi([z_{\text{rl}}, \bar{z}_{1:i-1}]))\right]_i - \bar{z}_i
\right|_2^2
\right]
]

## Critic

[
L_Q
===

\mathbb{E}*{(x,a*{1:C},x') \sim \mathcal{B}}
\left[
(\hat{Q} - Q_\psi(x,a_{1:C}))^2
\right]
]

[
\hat{Q}
=======

\sum_{t'=1}^{C} \gamma^{t'-1} r_{t'}
+
\gamma^C
\mathbb{E}*{a' \sim \pi*\theta}
\left[
Q_{\psi'}(x', a')
\right]
]

## Actor

[
\pi_\theta(a_{1:C} \mid x,\tilde a_{1:C})
=========================================

\mathcal N(\mu_\theta(x,\tilde a_{1:C}), \sigma^2 I)
]

[
L_\pi(\theta)
=============

\mathbb{E}
\left[

* Q_\psi(x, a_{1:C})

-

\beta |a_{1:C} - \tilde a_{1:C}|_2^2
\right]
]

---

# Implementation-oriented pseudocode

```markdown
Given:
- pretrained VLA π_vla
- task demos D
- online replay buffer B
- RL chunk length C
- VLA chunk horizon H
- optional human interventions

Stage 1: Train RL token
1. Extract final-layer token embeddings z from frozen VLA on demo observations.
2. Append learned RL token embedding.
3. Encode with lightweight transformer encoder.
4. Decode original embeddings autoregressively from RL token.
5. Optimize reconstruction loss.
6. Optionally jointly fine-tune VLA with supervised action loss.
7. Freeze VLA and RL token module.

Stage 2: Online RL
1. Initialize actor π_θ and twin critic Q_ψ from scratch.
2. Warm up replay buffer with VLA rollouts.
3. Repeat:
   a. At chunk boundary, query frozen VLA for reference chunk ã.
   b. Compute RL token z_rl and state x = (z_rl, proprio).
   c. Choose executed chunk:
      - human action if intervention
      - VLA chunk during warmup
      - actor sample π_θ(. | x, ã) otherwise
   d. Execute chunk.
   e. Store transition (x, executed_action, reference_action, reward, x').
   f. Also store stride-2 subsampled chunk transitions.
   g. For G updates:
      - update twin critic using TD target over chunked rewards
      - update actor with Q objective + β ||a - ã||²
      - mask reference chunk 50% of the time during actor training
```

---

# Limitations noted by the paper

The method still depends on humans for:

* reward labels,
* intervention corrections,
* switching between base VLA and RL during training.

The authors suggest future automation via:

* reward models,
* progress prediction.

So this is a practical semi-autonomous method, not a fully autonomous RL improvement loop.

---

# Final takeaway

RLT is best understood as:

* **representation distillation from a VLA into a compact RL state**
* plus **chunked off-policy actor-critic**
* plus **VLA-anchored policy refinement**

The method works because it combines:

* the generalization and priors of a large pretrained VLA,
* the sample efficiency of small-network off-policy RL,
* chunked action learning for sparse high-frequency control,
* local exploration around a strong reference behavior.

For implementation, the most important pieces are:

* learn a compact RL token from VLA embeddings,
* freeze the VLA during online RL,
* predict chunked actions,
* condition actor on VLA chunk,
* regularize toward that chunk,
* use reference dropout,
* train off-policy with high replay reuse,
* focus online RL on the critical precision phase first.
