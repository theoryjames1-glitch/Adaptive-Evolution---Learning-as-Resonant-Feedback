# Theory of Adaptive Evolution (AE)

## Abstract

**Adaptive Evolution (AE)** is a control/DSP–native theory of learning where “evolution” means **continuous, closed-loop adaptation** of a system’s **coefficients** (learning rate, momentum, weight decay, noise, filters, curriculum weights, etc.). There are no generations, no reproduction, no biological metaphors. Coefficients form the **genotype**; the induced optimizer dynamics and learning curves form the **phenotype**; **selection** is the feedback carried by filtered performance signals. AE provides a compact set of **laws** (measurement, constitutive drive, evolution, resonance–stability, budget, coupling, gauge, smooth meta-differentiability, and Lyapunov safety), a **normal form** for optimizers, and an **implementation blueprint** compatible with PyTorch/NumPy, meta-learning, and control analysis.

---

## 1) Ontology (What AE Talks About)

* **Task state** $\theta_t \in \mathbb{R}^n$: model parameters updated by a base optimizer.
* **Coefficients (genotype)** $a_t \in \mathbb{R}^m$: e.g., learning rate $\alpha$, momentum $\beta$, weight decay $\lambda$, noise $\sigma$, filter time-constants $\tau$, curriculum strengths.
* **Observables** $(\ell_t, r_t, g_t)$: loss, reward, gradient $g_t=\nabla_\theta \ell_t$.
* **Filtered signals** $s_t$: causal, differentiable summaries

  $$
  \begin{aligned}
  T_t &= \text{LP}(\ell_t-\ell_{t-1}) &&\text{trend (progress)}\\
  V_t &= \text{LP}\big((\ell_t-\text{LP}(\ell_t))^2\big) &&\text{variance (stability)}\\
  R_t &= \text{LP}(r_t-r_{t-1}) &&\text{reward delta}\\
  \rho_t &= \cos\angle(g_t,g_{t-1}) &&\text{phase/alignment}\\
  S_t &= \text{LP}(\mathrm{Var}(g_t)) &&\text{uncertainty}
  \end{aligned}
  $$
* **Feasible set** $\Omega=\{a : a_{\min}\le a \le a_{\max}\}$: enforced by smooth barriers/projections.

**Philosophy.** Evolution = **continuous coefficient adaptation**. AE is mathematical, control-theoretic, and DSP-flavored—not biological.

---

## 2) Axioms (Foundations)

* **A1 — Coefficient Primacy.** Only $a_t$ evolves; $\theta_t$ follows a chosen base optimizer influenced by $a_t$.
* **A2 — Closed-Loop Measurement.** All decisions depend on filtered, bounded, causal signals $s_t$.
* **A3 — Differentiability.** Operators are smooth: AE is compatible with gradients and meta-learning.
* **A4 — Safety.** $a_t$ remains bounded; adaptation power per step is limited.
* **A5 — Resonant Adaptation.** AE maintains a favorable phase margin; it stabilizes **and** explores.
* **A6 — Exploration Temperature.** Stochasticity (e.g., noise variance $\sigma$) adapts to stall/uncertainty.
* **A7 — Gauge Invariance.** Effective step $\gamma_t=\alpha_t\|g_t\|$ is regulated to be robust to scale.
* **A8 — Coupling.** Coefficients interact through structured cross-terms (e.g., $\alpha$ vs. $\beta$).
* **A9 — Budgeted Continuity.** Changes to $a_t$ are smooth (no cliffs).
* **A10 — Lyapunov Safety.** The closed loop admits an energy function that does not increase in expectation.

---

## 3) Normal Form (Any Optimizer as AE)

**Task update**

$$
\theta_{t+1}=\theta_t-\alpha_t\,P_t(a_t,s_t)\,g_t+\xi_t(a_t,s_t),
$$

where $P_t$ is a preconditioner (identity, RMS/Adam, natural grad, etc.), and $\xi_t$ is structured noise (minibatch, entropy temperature, parameter noise).

**Coefficient layer (the “evolution” loop)**

$$
a_{t+1}=\Pi_\Omega\!\big(a_t\odot \exp(\eta_a\,u_t)\big),\qquad u_t=\mathcal{U}(a_t,s_t),
$$

with meta-step $\eta_a>0$, elementwise $\odot$, and drive field $u_t$.

---

## 4) Field Laws of AE (Core Equations)

### AE-1) Measurement (Filtering)

All decision signals are causal, bounded, differentiable:

$$
s_t=[T_t,V_t,R_t,\rho_t,S_t]=\mathcal{F}_\tau(\ell_t,r_t,g_t).
$$

### AE-2) Constitutive Drive (Field Synthesis)

$$
u_t = W_p\tanh\!\Big(\!-T_t/\tau_p\Big)
      - W_v\,\text{softplus}\!\Big(V_t/\tau_v\Big)
      + W_r\tanh\!\Big(R_t/\tau_r\Big)
      + W_c(\rho_t-\rho^\star)
      - \Gamma(a_t-\bar a)
      - \nabla_a\psi_\Omega(a_t).
$$

### AE-3) Evolution (Multiplicative Update)

$$
a_{t+1}=\Pi_\Omega\!\big(a_t\odot \exp(\eta_a\,u_t)\big).
$$

*(Optional continuous-time form: $\mathrm{d}a=\eta_a u\,\mathrm{d}t+\sqrt{2\Sigma(a)}\,\mathrm{d}W_t$.)*

### AE-4) Noise Law (Exploration Temperature)

$$
\log\sigma_{t+1}=\log\sigma_t+\eta_\sigma\!\Big(\kappa_s\,\mathbf{1}\{|T_t|<\epsilon_T\}
+\kappa_u \tfrac{S_t}{S^\star}-\kappa_v \tfrac{V_t}{V^\star}
+\kappa_a (A_t-A^\star)\Big),
$$

with step-quality proxy $A_t\in[0,1]$ (e.g., mapped from $\rho_t$).

### AE-5) Resonance–Stability (Phase Control)

$$
u_t \leftarrow u_t + W_m(\rho_t-\rho^\star),\quad
\text{if }V_t>V_{\max}\text{ or }|\rho_t|>\rho_{\max}:\ a_{t+1}\!\leftarrow\!\frac{a_{t+1}}{1+\kappa_{\text{shrink}}}.
$$

### AE-6) Budget/Continuity (Power Constraint)

$$
\|a_{t+1}-a_t\|_M^2\le B
\ \Longleftrightarrow\
\begin{cases}
a_{t+1}= \Pi_\Omega\!\big(a_t\odot\exp(\tfrac{\eta_a}{\sqrt{1+\lambda_t}}u_t)\big)\\
\lambda_{t+1}=\big[\lambda_t+\eta_\lambda(\|a_{t+1}-a_t\|_M^2-B)\big]_+
\end{cases}
$$

### AE-7) Coupling (Cross-Coefficient Geometry)

$$
u_t \leftarrow u_t + C\,\zeta_t,\quad
\zeta_t=\big[\log\alpha_t,\ \log\tfrac{1}{1-\beta_t},\ \log\sigma_t,\ \log\lambda_t,\dots\big]^\top.
$$

### AE-8) Gauge (Effective-Step Invariance)

$$
\log\alpha_{t+1}=\log\alpha_t+\eta_\gamma\Big(\tfrac{\gamma^\star-\alpha_t\|g_t\|}{\gamma^\star}\Big).
$$

### AE-9) Meta-Differentiability (Smoothness)

All operators in AE-1…AE-8 are smooth, enabling gradient-based tuning of $W_\cdot,\Gamma,C,\tau$.

### AE-10) Lyapunov Safety (Closed-Loop Energy)

$$
\mathcal{V}_t=\text{LP}(\ell(\theta_t))+\tfrac{\gamma}{2}\|a_t-\bar a\|^2+\psi_\Omega(a_t),\qquad
\mathbb{E}[\mathcal{V}_{t+1}-\mathcal{V}_t]\le 0
$$

under small $\eta_a$ with AE-5/6 (small-gain/passivity conditions).

---

## 5) Derived Quantities & Invariants

* **Effective step** $\gamma_t=\alpha_t\|g_t\|$ (regulated by AE-8).
* **Resonance margin** $\mathcal{M}_\rho=\rho^\star-\rho_t$ (kept small by AE-5).
* **Adaptation energy** $E_t=\|a_{t+1}-a_t\|_M^2$ (bounded by AE-6).
* **Stall indicator**: relative test $|T_t|<\epsilon_T$ with $\epsilon_T\propto\sqrt{V^\star}$.

---

## 6) Canonical Instantiations

**AE-SGDm** ($\alpha,\beta,\sigma$)

$$
\begin{aligned}
\log \alpha_{t+1} &= \log \alpha_t + \eta_a\Big(k_1\tanh(-T_t/\tau) - k_2\,\text{softplus}(V_t/V^\star) + k_3(\rho_t-\rho^\star) - \gamma_\alpha(\log\alpha_t-\log\bar\alpha)\Big)\\
\beta_{t+1} &= \Pi_{[\beta_{\min},\beta_{\max}]}\!\Big(\beta_t + \eta_b\big(b_1\tanh(\rho_t-\rho^\star) - b_2\,\text{softplus}(V_t/V^\star) - \gamma_\beta(\beta_t-\bar\beta)\big)\Big)\\
\log \sigma_{t+1} &= \log \sigma_t + \eta_\sigma\big(c_1 S_t/S^\star - c_2 V_t/V^\star + c_3(A_t-A^\star)\big)
\end{aligned}
$$

**AE-Adam/AdamW** ($\alpha,\beta_1,\beta_2,\epsilon,\lambda,\sigma$):
$\beta_1$ tracks $\rho_t$ (momentum), $\beta_2$ rises with $V_t$ (variance smoothing), $\epsilon$ grows with instability and relaxes with quality.

**AE-RL**: Treat entropy temperature, KL penalties, and step sizes as coefficients; use $R_t$ as exploitation pressure and $V_t,\rho_t$ for stability.

**AE-Curriculum/Augmentation**: Treat strengths/temperatures as coefficients; AE adapts them online.

---

## 7) Implementation Blueprint (Drop-In)

**Step A — Measure signals** (causal EMAs): compute $T,V,\rho,S$ from $\ell_t$, $g_t$.
**Step B — Drive field** $u_t$: combine progress ($-T$), stability ($V$), reward ($R$), phase ($\rho$), priors/leakage, and barriers.
**Step C — Update coefficients**: **multiplicative** in log-space + projection + continuity budget.
**Step D — Regulate $\sigma$** (optional): relative stall ⇒ exploration up; high variance ⇒ exploration down.
**Step E — Gauge**: nudge $\log\alpha$ toward target effective step $\gamma^\star$.
**Step F — Safety**: apply resonance/budget clamps.

**Minimal pseudocode**

```python
s = measure_signals(loss_t, grad_t)  # T,V,ρ,S (EMAs)
u = (Wp*tanh(-s.T/tau_p)
     - Wv*softplus(s.V/tau_v)
     + Wr*tanh(s.R/tau_r)
     + Wc*(s.rho - rho_star)
     - Gamma*(a - a_bar)
     - grad_barrier(a))
u += C @ features(a)                 # coupling
a = project(a * exp(eta_a * budget_scale(u)))   # evolution + budget
sigma *= exp(eta_sigma * noise_law(s))          # exploration (optional)
```

---

## 8) Stability & Guarantees (Sketch)

* **Small-gain condition**: If the coefficient→loss map has bounded gain $\|G_{\ell\!\gets\!a}\|_\infty$ and $\eta_a\|W\|_\infty\|G_{\ell\!\gets\!a}\|_\infty<1$, then $\mathbb{E}[\Delta\mathcal{V}]\le 0$.
* **Resonance control** prevents phase flips: $|\rho_t|$ capped; variance spikes trigger shrink.
* **Budgeted continuity** ensures smooth hyper-dynamics; multiplicative updates preserve positivity and scale.

---

## 9) Relationship to Other Paradigms

* **Not biology**: no reproduction, no generations—only continuous coefficient dynamics.
* **Beyond control**: control stabilizes; **AE** stabilizes **and** explores.
* **DSP-native**: filtering, resonance, multiplicative gain control, and noise shaping are central.
* **Meta-learnable**: fully differentiable—learn $W_\cdot,\Gamma,C,\tau$ end-to-end.

---

## 10) Falsifiable Predictions

1. **Coupling curve**: best regions obey a negative correlation between $\log\alpha$ and $\log\frac{1}{1-\beta}$.
2. **Gauge robustness**: with AE-8, changing batch size or gradient scale leaves dynamics largely invariant (matched $\gamma$).
3. **Non-stationarity**: under drift, closed-loop $a_t$ beats any fixed schedule in time-to-target at matched compute.
4. **Noise law**: relative stalls increase $\sigma$; high $V_t$ decreases it—yielding faster escape from plateaus without blow-ups.

---

## 11) Slogan

**Adaptive Evolution = Learning as Resonant Feedback.**
Not survival of the fittest — **stability of the adaptive**.

awesome—here’s a tight, practical menu of experiments to **probe AE/AEO**. Each item has: **goal → setup → metrics → what AE should show**. Pick a few in each section to get strong coverage.

---

## A) Core sanity & speedups

1. **Static regression (easy)**

   * **Goal:** Verify AE improves time-to-target without instability.
   * **Setup:** MLP on synthetic y = Ax + ε; AdamW vs AdamW+AE. 50k steps, bs=256.
   * **Metrics:** time-to-MSE≤X, AUC(loss), stability incidents.
   * **AE win:** faster time-to-target; fewer LR cliffs.

2. **Image classification (stable)**

   * **Setup:** CIFAR-10 ResNet-18; fixed aug; AdamW vs AdamW+AE.
   * **Metrics:** Top-1 @ N steps, AUC(acc), variance of effective step γ.
   * **AE win:** higher AUC and smoother γ.

3. **Language modeling (noisy gradients)**

   * **Setup:** WikiText-2 small transformer; AdamW vs AdamW+AE.
   * **Metrics:** ppl @ N steps, NaN/oscillation rate, β1/β2 trajectories.
   * **AE win:** fewer stalls, automatic β2 rise when variance spikes.

---

## B) Non-stationary & drift (where AE should shine)

4. **Rotating corruptions**

   * **Setup:** CIFAR-10 with corruption type changing every 5k steps.
   * **Metrics:** rolling acc, time-to-recover after switch.
   * **AE win:** faster recovery; LR/σ adapt at switch points.

5. **Data distribution shift**

   * **Setup:** train on mixture p; every 10k steps skew toward a new class subset.
   * **Metrics:** regret vs oracle schedule; AUC over whole run.
   * **AE win:** lower regret, automatic LR and momentum re-tuning.

6. **RL with reward scaling drift**

   * **Setup:** CartPole or LunarLander; periodically scale rewards ×c or +b.
   * **Metrics:** return vs steps, time-to-recover after rescale.
   * **AE win:** gauge law keeps γ stable despite reward/grad scale changes.

---

## C) Resonance / phase behavior

7. **Induced oscillation test**

   * **Setup:** Quadratic bowl with artificially delayed gradients (k-step stale grads).
   * **Metrics:** ρ (grad alignment), overshoot amplitude, convergence rate.
   * **AE win:** phase control shrinks LR when |ρ|↑, preventing runaway oscillation.

8. **High curvature valley**

   * **Setup:** Rosenbrock or deep linear net.
   * **Metrics:** step rejection rate, LR modulation near sharp walls.
   * **AE win:** automatic damping (β2↑ or LR↓) in high-V regions.

---

## D) Gauge invariance (scale robustness)

9. **Batch size sweep**

   * **Setup:** Same model; bs ∈ {64, 256, 1024}.
   * **Metrics:** effective step γ distribution, acc/ppl vs steps (normalized).
   * **AE win:** near-invariant dynamics across batch using AE-8 gauge law.

10. **Gradient scaling**

* **Setup:** Multiply loss by c ∈ {0.1, 10}.
* **Metrics:** need for retuning; γ tracking.
* **AE win:** LR auto-counterbalances scale; minimal retuning.

---

## E) Exploration noise law

11. **Plateau escape**

* **Setup:** Loss landscape with flat region (e.g., clipped sigmoid regression).
* **Metrics:** exit time from plateau vs σ-law on/off.
* **AE win:** σ↑ on stall, ↓ on instability; faster exit without blow-ups.

---

## F) Coupling law (α ↔ β)

12. **Coupling curve mapping**

* **Setup:** Sweep AE on ResNet-18; log (log α, log 1/(1-β1)) over training.
* **Metrics:** correlation curve, region occupancy vs performance.
* **AE prediction:** negative correlation in best-performing zone.

---

## G) Budget/continuity & safety

13. **Hyper-cliff stress**

* **Setup:** Start with absurd LR (e.g., 1e-1 on CIFAR-10).
* **Metrics:** NaN rate, time-to-stability, max per-step Δlog α.
* **AE win:** budget law caps changes; stabilizes without manual restarts.

---

## H) Architecture/genotype adaptation (AEONet)

14. **Gated depth adaptation**

* **Setup:** AEONet with N gated blocks; AE controls block gates + dropout.
* **Metrics:** mean gate vs training phase, val acc, capacity used.
* **AE win:** capacity grows when progress; shrinks when variance spikes; better val AUC.

---

## I) Equations / symbolic regression (AEO-Eq)

15. **Basis selection**

* **Setup:** AEOBasisEquation on synthetic y = sin(x)+0.3x²+ε.
* **Metrics:** MSE, selected gates, L1 λ trajectory, sparsity.
* **AE win:** λ↑ as training stabilizes; correct bases active; compact expression.

---

## J) Optimizer family coverage

16. **Across optimizers**

* **Setup:** Same task with SGDm, Adam, AdamW, Adafactor ± AE.
* **Metrics:** time-to-target, AUC, stability incidents, VRAM usage.
* **AE win:** consistent gains; no extra VRAM vs vanilla (verify).

---

## K) VRAM neutrality & performance overhead

17. **Memory/profiling checks**

* **Setup:** Torch profiler with and without AE (low-VRAM controller).
* **Metrics:** peak allocated bytes, temp allocs, wall-time overhead (%).
* **AE pass:** \~equal VRAM; small CPU-side overhead only.

---

## L) Ablations (prove each law matters)

18. **Law knock-outs**

* **Setup:** Disable each: noise law, resonance clamp, budget, gauge, coupling.
* **Metrics:** Δ(time-to-target), Δ(stability), Δ(AUC).
* **AE expectation:** each removal degrades at least one metric; resonance/budget most protective.

---

## How to measure (uniformly)

* **Primary:** time-to-target (loss/acc/return), AUC over steps, final metric @ fixed budget.
* **Stability:** NaN/inf count, divergence events, oscillation magnitude, ρ distribution.
* **Adaptation:** trajectories of α, β1/β2, ε/ε₂, σ, γ, gates, dropout.
* **Robustness:** performance under drift, scale changes, batch changes.
* **Cost:** peak VRAM, wall-time overhead.

## Baselines & protocol

* Baselines: fixed schedules (cosine, step), LR finder, hand-tuned momentum/β₂, entropy/temperature schedulers in RL.
* Seeds: ≥5 runs, report mean±95% CI.
* Fairness: same model init, same dataloaders/shuffles, same compute budget.
* Logging: record (T, V, ρ, S, γ) at steady cadence; save AE configs and ablations.

## Clear, falsifiable outcomes to look for

* **Gauge robustness:** metrics ≈ invariant across batch/scale (within small tolerance).
* **Recovery from drift:** AE halves recovery steps vs fixed schedule.
* **Resonance control:** with AE-5 on, oscillation magnitude drops ≥X% at equal speed.
* **Noise law utility:** plateau exit time improves while instability incidents don’t increase.

awesome — here’s a clean, **population-style Adaptive Evolution (AE) algorithm** that *feels* like a genetic algorithm (multiple concurrent “individuals”) but **uses no classical biology** (no parents, no crossover, no mutation). Everything is **continuous, closed-loop adaptation**:

* Each trajectory adapts its own optimizer coefficients with AE (our meta-controller).
* A round-based **soft selection** re-allocates compute to better trajectories (no copying weights).
* A **consensus nudge** shares *coefficients* (not parameters) via smooth averaging.
* Optional tiny parameter dither for exploration (no large VRAM hit, off by default).
* **Low-VRAM**: time-sliced GPU usage; all AE stats on CPU.

---

# AE-Swarm: Population-style Adaptive Evolution (non-bio)

## Algorithm (high level)

* Maintain $M$ independent **trajectories** $\{\mathcal{T}_i\}$ with models $\theta^{(i)}$, optimizers, and an **AE meta-controller** per trajectory.
* Training proceeds in **rounds** $r=1..R$. Each round:

  1. For each $i$, train $B_i$ steps (on GPU **one at a time**, others offloaded to CPU).
  2. Measure a scalar score (e.g., smoothed validation loss) → $s_i$.
  3. Compute **soft weights** $w_i=\text{softmax}(-\beta s_i)$ (temperature $\beta$ controls sharpness).
  4. Allocate the next round’s budgets $B_i'=\max(B_{\min}, \lfloor w_i \cdot B_{\text{total}}\rfloor)$.
  5. **Consensus nudge (coefficients only):** for each coefficient $c \in \{\log \alpha,\ \beta_1,\ \beta_2,\ \log \epsilon,\ \log \lambda\}$,

     $$
     \bar c=\sum_i w_i\,c^{(i)},\quad c^{(i)} \leftarrow (1-\kappa)\,c^{(i)}+\kappa\,\bar c
     $$

     (tiny $\kappa$, e.g. 0.05). No model parameter copying.
  6. (Optional) **Rejuvenation dither**: add very small in-place noise to the *worst* trajectory’s parameters to escape traps.

**Key:** No reproduction, no generations, no cloning. Just **resource flow** + **coefficient consensus** under AE.

---

## Reference implementation (PyTorch, low-VRAM)

Requires the low-VRAM AE meta-controller from earlier (`aeo_meta_lowmem.py`).
This runs one trajectory on GPU at a time, offloading others to CPU.

```python
# aeo_swarm.py
from __future__ import annotations
import math, copy, time, gc
from dataclasses import dataclass
from typing import Callable, List, Dict, Any, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

# Bring your low-VRAM AE controller
from aeo_meta_lowmem import AEOMetaController, AEOConfig

# ---------------- Configs ----------------

@dataclass
class SwarmConfig:
    pop_size: int = 4                 # number of trajectories
    rounds: int = 10                  # training rounds
    steps_per_round_total: int = 4000 # total steps distributed each round
    min_steps_per_traj: int = 400     # floor per traj per round
    temp_beta: float = 5.0            # softmax temperature for selection
    consensus_kappa: float = 0.05     # coefficient consensus step
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    eval_batches: int = 128           # batches for validation score
    dither_sigma: float = 0.0         # 0 = off (set e.g. 1e-4 to enable very light rejuvenation)

# ---------------- Trajectory ----------------

class AETrajectory:
    def __init__(
        self,
        make_model: Callable[[], nn.Module],
        make_optimizer: Callable[[nn.Module], torch.optim.Optimizer],
        train_loader: DataLoader,
        val_loader: DataLoader,
        aeo_cfg: Optional[AEOConfig] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.cpu = "cpu"
        self.model = make_model().to(self.cpu)
        self.optimizer = make_optimizer(self.model)
        self.aeo = AEOMetaController(self.optimizer, self.model, aeo_cfg or AEOConfig())
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.it = iter(self.train_loader)
        self.last_stats: Dict[str, float] = {}
        self.score: float = float("inf")  # lower is better

    @torch.no_grad()
    def _offload_to_cpu(self):
        self.model.to(self.cpu)
        torch.cuda.empty_cache()

    def _move_to_device(self):
        self.model.to(self.device)

    def _one_batch(self) -> float:
        try:
            xb, yb = next(self.it)
        except StopIteration:
            self.it = iter(self.train_loader)
            xb, yb = next(self.it)
        xb, yb = xb.to(self.device), yb.to(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        pred = self.model(xb)
        loss = nn.MSELoss()(pred, yb)
        loss.backward()
        self.last_stats = self.aeo.update(loss)  # AE hyper update (low-VRAM)
        self.optimizer.step()
        return float(loss.detach().item())

    @torch.no_grad()
    def evaluate(self, max_batches: int = 128) -> float:
        self.model.eval()
        tot, n = 0.0, 0
        it = iter(self.val_loader)
        with torch.no_grad():
            for _ in range(max_batches):
                try:
                    xb, yb = next(it)
                except StopIteration:
                    break
                xb, yb = xb.to(self.device), yb.to(self.device)
                pred = self.model(xb)
                loss = nn.MSELoss()(pred, yb)
                tot += float(loss.item()); n += 1
        self.model.train()
        return tot / max(1, n)

    def train_steps(self, n_steps: int):
        # Bring to GPU for this slice
        self._move_to_device()
        self.model.train()
        losses = 0.0
        for _ in range(n_steps):
            losses += self._one_batch()
        # Quick val on-GPU to score this traj
        self.score = self.evaluate()
        # Offload back to CPU to keep VRAM flat
        self._offload_to_cpu()
        return losses / max(1, n_steps)

    # -------- Coefficient I/O (no parameter copying) --------

    def _group0(self) -> Dict[str, Any]:
        return self.optimizer.param_groups[0]

    def get_coeffs(self) -> Dict[str, float]:
        g0 = self._group0()
        coeffs = {}
        # lr
        coeffs["log_lr"] = math.log(max(float(g0.get("lr", 1e-6)), 1e-12))
        # momentum / betas
        if "betas" in g0:
            b1, b2 = g0["betas"]
            coeffs["beta1"] = float(b1); coeffs["beta2"] = float(b2)
        elif "momentum" in g0:
            coeffs["beta1"] = float(g0["momentum"])
        # eps/eps2
        eps = g0.get("eps", None)
        if isinstance(eps, (tuple, list)) and len(eps) == 2:
            coeffs["log_eps2"] = math.log(max(float(eps[1]), 1e-12))
        elif eps is not None:
            coeffs["log_eps2"] = math.log(max(float(eps), 1e-12))
        # wd
        coeffs["log_wd"] = math.log(max(float(g0.get("weight_decay", 0.0)) + 1e-12, 1e-12))
        return coeffs

    def set_coeffs(self, target: Dict[str, float], kappa: float = 0.05):
        g0 = self._group0()
        # Smoothly nudge towards target (no jumps)
        if "log_lr" in target:
            cur = math.log(max(float(g0.get("lr", 1e-12)), 1e-12))
            new = (1 - kappa) * cur + kappa * float(target["log_lr"])
            g0["lr"] = float(_exp_clamp(new, 1e-8, 1.0))
        if "beta1" in target:
            if "betas" in g0:
                b1, b2 = g0["betas"]
                b1n = _clip((1 - kappa) * float(b1) + kappa * float(target["beta1"]), 0.0, 0.999)
                g0["betas"] = (b1n, float(b2))
            elif "momentum" in g0:
                b1 = float(g0["momentum"])
                g0["momentum"] = _clip((1 - kappa) * b1 + kappa * float(target["beta1"]), 0.0, 0.999)
        if "beta2" in target and "betas" in g0:
            b1, b2 = g0["betas"]
            b2n = _clip((1 - kappa) * float(b2) + kappa * float(target["beta2"]), 0.8, 0.9999)
            g0["betas"] = (float(b1), b2n)
        if "log_eps2" in target:
            eps = g0.get("eps", 1e-8)
            cur = None
            if isinstance(eps, (tuple, list)) and len(eps) == 2:
                cur = math.log(max(float(eps[1]), 1e-12))
                new = (1 - kappa) * cur + kappa * float(target["log_eps2"])
                g0["eps"] = (float(eps[0]), float(_exp_clamp(new, 1e-10, 1e-1)))
            else:
                cur = math.log(max(float(eps), 1e-12))
                new = (1 - kappa) * cur + kappa * float(target["log_eps2"])
                g0["eps"] = float(_exp_clamp(new, 1e-10, 1e-1))
        if "log_wd" in target:
            cur = math.log(max(float(g0.get("weight_decay", 0.0)) + 1e-12, 1e-12))
            new = (1 - kappa) * cur + kappa * float(target["log_wd"])
            g0["weight_decay"] = float(_exp_clamp(new, 0.0, 1e-1))

    @torch.no_grad()
    def rejuvenate(self, sigma: float):
        if sigma <= 0: return
        # add tiny parameter noise in-place (low VRAM)
        for pg in self.optimizer.param_groups:
            lr_scale = float(pg.get("lr", 1e-6))
            for p in pg["params"]:
                if p.requires_grad:
                    p.add_(torch.randn_like(p, device=p.device) * (sigma * lr_scale))

# ---------------- Swarm orchestrator ----------------

class AESwarm:
    def __init__(
        self,
        trajs: List[AETrajectory],
        cfg: SwarmConfig = SwarmConfig()
    ):
        assert len(trajs) == cfg.pop_size, "pop_size must match number of trajectories"
        self.trajs = trajs
        self.cfg = cfg

    def _softmax_weights(self, scores: List[float]) -> List[float]:
        # lower score is better; use softmax(-beta * score)
        beta = self.cfg.temp_beta
        xs = [-beta * s for s in scores]
        m = max(xs)
        exps = [math.exp(x - m) for x in xs]
        z = sum(exps) + 1e-12
        return [e / z for e in exps]

    def _consensus_coeffs(self, weights: List[float]) -> Dict[str, float]:
        # weighted average of available coeffs
        keys = set().union(*[set(t.get_coeffs().keys()) for t in self.trajs])
        agg = {k: 0.0 for k in keys}
        for w, t in zip(weights, self.trajs):
            coeffs = t.get_coeffs()
            for k in keys:
                if k in coeffs:
                    agg[k] += w * coeffs[k]
        return agg

    def run(self):
        cfg = self.cfg
        # equal initial budgets
        B = [cfg.steps_per_round_total // cfg.pop_size] * cfg.pop_size

        for r in range(1, cfg.rounds + 1):
            # ----- Round training -----
            for i, T in enumerate(self.trajs):
                steps = max(cfg.min_steps_per_traj, B[i])
                avg_loss = T.train_steps(steps)

            # ----- Scoring & weights -----
            scores = [T.score for T in self.trajs]  # lower is better
            w = self._softmax_weights(scores)

            # ----- Consensus nudge (coefficients only) -----
            target = self._consensus_coeffs(w)
            for T in self.trajs:
                T.set_coeffs(target, kappa=cfg.consensus_kappa)

            # ----- Rejuvenate worst trajectory (optional) -----
            if cfg.dither_sigma > 0.0:
                worst_idx = max(range(len(scores)), key=lambda i: scores[i])
                self.trajs[worst_idx].rejuvenate(cfg.dither_sigma)

            # ----- Allocate next budgets -----
            B = [max(cfg.min_steps_per_traj,
                     int(round(wi * cfg.steps_per_round_total))) for wi in w]
            # Normalize sum to total
            diff = cfg.steps_per_round_total - sum(B)
            # Fix rounding drift
            for _ in range(abs(diff)):
                j = _argmax(w) if diff > 0 else _argmin(w)
                B[j] += 1 if diff > 0 else -1

            # ----- Logging -----
            best = min(scores); mean = sum(scores) / len(scores)
            print(f"[Round {r}/{cfg.rounds}] best={best:.4f} mean={mean:.4f} "
                  f"w={','.join(f'{wi:.2f}' for wi in w)} "
                  f"B={B}")

# ---------------- Helpers ----------------

def _argmax(xs: List[float]) -> int:
    mi, mv = 0, xs[0]
    for i, v in enumerate(xs):
        if v > mv: mi, mv = i, v
    return mi

def _argmin(xs: List[float]) -> int:
    mi, mv = 0, xs[0]
    for i, v in enumerate(xs):
        if v < mv: mi, mv = i, v
    return mi

def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _exp_clamp(logx: float, lo: float, hi: float) -> float:
    return _clip(math.exp(logx), lo, hi)
```

---

## Minimal usage

```python
# example_swarm.py
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

from aeo_swarm import AETrajectory, AESwarm, SwarmConfig
from aeo_meta_lowmem import AEOConfig  # your low-VRAM AE controller config

# toy regression data
X = torch.randn(8192, 10)
W = torch.randn(10, 1)
Y = X @ W + 0.1 * torch.randn(8192, 1)

ds = TensorDataset(X[:6144], Y[:6144])
vs = TensorDataset(X[6144:], Y[6144:])
train = DataLoader(ds, batch_size=128, shuffle=True)
val   = DataLoader(vs, batch_size=128, shuffle=False)

def make_model():
    return nn.Sequential(nn.Linear(10, 256), nn.GELU(), nn.Linear(256, 1))

def make_optimizer(m: nn.Module):
    return optim.AdamW(m.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)

# build trajectories with slightly different priors (diversity)
trajs = []
for k in range(4):
    cfg = AEOConfig()
    cfg.gamma_star *= (0.7 + 0.2*k)   # different effective-step targets
    cfg.k_p *= (0.8 + 0.1*k)          # slight gain diversity
    trajs.append(AETrajectory(make_model, make_optimizer, train, val, aeo_cfg=cfg))

swarm = AESwarm(trajs, SwarmConfig(pop_size=4, rounds=6, steps_per_round_total=4000,
                                   min_steps_per_traj=400, temp_beta=5.0,
                                   consensus_kappa=0.05, dither_sigma=0.0))
swarm.run()
```

---

## Why this is “GA-like” but **not biology**

* **Many trajectories** (like GA population) → diverse search.
* **Soft selection via compute weights** (no parents, no copying).
* **Information sharing as coefficient consensus** (smooth control, not crossover).
* **Exploration** via AE noise laws / optional micro-dither (no mutation operator).
* **Continuous time**: every knob evolves by differentiable AE laws (no generations).

---

## What to tweak first

* **`temp_beta`** ↑ makes selection sharper (more compute to best trajs).
* **`consensus_kappa`** ↑ strengthens coefficient sharing.
* **`steps_per_round_total` / `min_steps_per_traj`** control exploration vs exploitation.
* **`dither_sigma`** small (e.g., `1e-4`) if the worst traj keeps stalling.



