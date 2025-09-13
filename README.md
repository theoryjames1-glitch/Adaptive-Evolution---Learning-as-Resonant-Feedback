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

### AE.py

```python
# -*- coding: utf-8 -*-
"""
AE.py — Adaptive Evolution Optimizer (PyTorch, low-VRAM, numerically safe)
Evolution = continuous coefficient adaptation (non-biological).

Exports:
- AEConfig
- AEAgent
- AEEnsemble (optional)

Guards & safety:
- version-safe noise generation (no randn_like(generator=...))
- gradient sanitization (NaN/Inf -> finite bounds)
- numeric guards on loss stats (clamp before variance)
- emergency brake on extreme loss/grad norms
- NEW: per-step update norm cap (optional) and parameter clamp (abs bound)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Union, List
import math
import torch
from torch import Tensor

LossFn = Callable[[Tensor], Union[Tensor, Tuple[Tensor, Tensor]]]

# ---------- helpers ----------
def _clip(x: float, lo: float, hi: float) -> float: return float(max(lo, min(hi, x)))
def _budget(d: float, mag: Optional[float]) -> float: return float(_clip(d, -mag, mag)) if mag else float(d)
def _softplus(x: float) -> float:
    if x > 20: return x
    if x < -20: return math.exp(x)
    return math.log1p(math.exp(x))

class _EMA:
    """Scalar EMA with bias correction (CPU)."""
    def __init__(self, beta: float):
        self.b = float(beta); self.m: Optional[float] = None; self.t = 0
    def update(self, x: float) -> float:
        x = float(x)
        if self.m is None: self.m = x; self.t = 1; return x
        self.m = self.b*self.m + (1.0-self.b)*x; self.t += 1
        return self.m / max(1.0 - self.b**self.t, 1e-12)

# ---------- config ----------
@dataclass
class AEConfig:
    # Filters
    beta_mu: float = 0.98; beta_T: float = 0.90; beta_V: float = 0.95; beta_S: float = 0.95; beta_star: float = 0.99
    # Gains
    k_p: float = 0.8; k_v: float = 0.8; k_c: float = 0.4
    gamma_lr: float = 0.05; gamma_mu: float = 0.05
    # Noise-law gains
    k_s_stall: float = 0.5; k_s_uncert: float = 0.4; k_s_var: float = 0.4; k_s_quality: float = 0.2
    # Targets / clamps
    rho_star: float = 0.30; rho_max: float = 0.95; V_max_mult: float = 4.0
    gamma_star: float = 1e-2; auto_gamma: bool = True; auto_gamma_warmup: int = 200
    # Meta-steps
    eta_a: float = 5e-4; eta_mu: float = 8e-4; eta_sigma: float = 5e-4; eta_gamma: float = 1e-3
    # Budgets
    max_dlog_alpha: Optional[float] = 0.10; max_d_mu: Optional[float] = 0.02; max_dlog_sigma: Optional[float] = 0.10
    # Bounds
    alpha_min: float = 1e-6; alpha_max: float = 1.0
    mu_min: float = 0.0; mu_max: float = 0.999
    sigma_min: float = 0.0; sigma_max: float = 1.0
    # Anchors
    alpha_bar: float = 3e-4; mu_bar: float = 0.9
    # Stall threshold
    tau_T_stall: float = 0.25
    # Device/dtype
    device: Union[str, torch.device] = "cpu"; dtype: torch.dtype = torch.float64
    # Emergency limits
    loss_emergency: float = 1e8
    grad_emergency: float = 1e6
    # NEW: step & parameter safety
    step_norm_max: Optional[float] = None     # cap on ||Δθ|| per-step (None disables)
    param_abs_max: float = 1e6                 # clamp θ ∈ [-param_abs_max, +param_abs_max]

# ---------- AE Agent ----------
class AEAgent:
    """
    Adaptive Evolution optimizer: continuous coefficient adaptation (α, μ, σ).
    VRAM-neutral: all signals are scalar reductions; no grad clones/flatten.
    """
    def __init__(
        self,
        dim: int,
        cfg: AEConfig = AEConfig(),
        seed: Optional[int] = None,
        init_theta: Optional[Tensor] = None,
        init_alpha: float = 3e-4,
        init_mu: float = 0.9,
        init_sigma: float = 0.0,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        clip_grad_norm: Optional[float] = None,
    ):
        self.cfg = cfg
        self.device = torch.device(device or cfg.device)
        self.dtype = dtype or cfg.dtype
        self.clip_grad_norm = clip_grad_norm

        self.gen = torch.Generator(device=self.device)
        if seed is not None: self.gen.manual_seed(int(seed))

        if init_theta is None:
            self.theta = torch.zeros(dim, dtype=self.dtype, device=self.device, requires_grad=True)
        else:
            self.theta = init_theta.detach().to(self.device, self.dtype).requires_grad_(True)
        self.m = torch.zeros_like(self.theta)

        self.alpha = float(init_alpha); self.mu = float(init_mu); self.sigma = float(init_sigma)

        self.mu_l = _EMA(self.cfg.beta_mu); self.T_ema = _EMA(self.cfg.beta_T)
        self.V_ema = _EMA(self.cfg.beta_V); self.S_ema = _EMA(self.cfg.beta_S)
        self.Vstar = _EMA(self.cfg.beta_star); self.Sstar = _EMA(self.cfg.beta_star)

        self._step = 0; self._gamma_med = _EMA(0.99); self._gamma_star_auto: Optional[float] = None
        self._prev_loss: Optional[float] = None

    def step(self, loss_fn: LossFn) -> Dict[str, float]:
        # forward & gradient
        out = loss_fn(self.theta)
        if isinstance(out, (tuple, list)):
            loss, grad = out
        else:
            loss = out
            grad = torch.autograd.grad(loss, self.theta, retain_graph=False, create_graph=False, allow_unused=False)[0]
        if grad is None:
            raise ValueError("loss_fn must depend on theta; grad is None")

        # Sanitize gradient BEFORE any checks: NaN->0, ±Inf->±grad_emergency
        grad = torch.nan_to_num(grad, 0.0, self.cfg.grad_emergency, -self.cfg.grad_emergency)

        # Optional grad clipping
        gnorm_t = float(torch.linalg.norm(grad).item())
        if self.clip_grad_norm is not None and gnorm_t > self.clip_grad_norm and gnorm_t > 0.0:
            grad = grad * (self.clip_grad_norm / gnorm_t)
            gnorm_t = float(torch.linalg.norm(grad).item())

        # ---------- Measurement with guards ----------
        # Loss: allow non-finite here; clamp to emergency scalar below
        L = float(loss.item()) if torch.isfinite(loss).item() else self.cfg.loss_emergency
        if abs(L) > self.cfg.loss_emergency:
            L = math.copysign(self.cfg.loss_emergency, L)

        mu_l = self.mu_l.update(L)
        dloss = 0.0 if self._prev_loss is None else (L - self._prev_loss)
        self._prev_loss = L
        T = self.T_ema.update(dloss)

        # Variance: bound ld before squaring to avoid overflow in V
        ld = L - mu_l
        if not math.isfinite(ld): ld = 0.0
        ld = max(min(ld, 1e6), -1e6)
        V = self.V_ema.update(ld*ld + 1e-12)

        # Grad stats (scalar reductions only)
        g = grad; m = self.m
        sum_g  = float(g.sum().item())
        sum_g2 = float((g*g).sum().item())
        n_elem = g.numel()
        gmean  = sum_g / max(1, n_elem)
        gvar   = max(sum_g2 / max(1, n_elem) - gmean*gmean, 0.0)
        S      = self.S_ema.update(gvar + 1e-12)
        gnorm  = math.sqrt(sum_g2) + 1e-12
        mnorm  = float(torch.linalg.norm(m).item()) + 1e-12
        dot_gm = float((g*m).sum().item())
        rho    = _clip(dot_gm / (gnorm * mnorm), -1.0, 1.0) if mnorm > 1e-12 else 0.0

        # Emergency brake if things are wild
        if (abs(L) > self.cfg.loss_emergency) or (gnorm > self.cfg.grad_emergency) or (not math.isfinite(gnorm)):
            self.alpha = max(self.cfg.alpha_min, self.alpha / 10.0)
            self.mu = min(self.mu, 0.5)

        Vstar = max(self.Vstar.update(V), 1e-12)
        Sstar = max(self.Sstar.update(S), 1e-12)

        # ---------- Gauge (auto γ*) ----------
        self._step += 1
        gamma_est = abs(self.alpha * gnorm)
        gtrack = self._gamma_med.update(gamma_est)
        if self.cfg.auto_gamma and self._gamma_star_auto is None and self._step > self.cfg.auto_gamma_warmup:
            self._gamma_star_auto = 0.8 * max(gtrack, 1e-12)
        gamma_star = self._gamma_star_auto if self._gamma_star_auto is not None else self.cfg.gamma_star

        # ---------- Drive & evolution ----------
        # α (log-space)
        drive_alpha = ( self.cfg.k_p * math.tanh(-T)
                      - self.cfg.k_v * _softplus(V / Vstar)
                      + self.cfg.k_c * (rho - self.cfg.rho_star)
                      - self.cfg.gamma_lr * (math.log(max(self.alpha,1e-12)) - math.log(self.cfg.alpha_bar)) )
        gauge = self.cfg.eta_gamma * ((gamma_star - self.alpha*gnorm) / (gamma_star + 1e-12))
        dlog_alpha = _budget(self.cfg.eta_a * drive_alpha + gauge, self.cfg.max_dlog_alpha)
        self.alpha = _clip(math.exp(math.log(max(self.alpha, self.cfg.alpha_min)) + dlog_alpha),
                           self.cfg.alpha_min, self.cfg.alpha_max)

        # μ (additive in [0,1])
        drive_mu = ( 0.8 * math.tanh(rho - self.cfg.rho_star)
                   - 0.6 * _softplus(V / Vstar)
                   - self.cfg.gamma_mu * (self.mu - self.cfg.mu_bar) )
        d_mu = _budget(self.cfg.eta_mu * drive_mu, self.cfg.max_d_mu)
        self.mu = _clip(self.mu + d_mu, self.cfg.mu_min, self.cfg.mu_max)

        # σ (log-space, usually disabled in tests)
        stall_thr = self.cfg.tau_T_stall * math.sqrt(Vstar); stall = 1.0 if abs(T) < stall_thr else 0.0
        A = 0.5 * (rho + 1.0)
        drive_sigma = ( self.cfg.k_s_stall * stall
                      + self.cfg.k_s_uncert * (S / Sstar)
                      - self.cfg.k_s_var   * (V / Vstar)
                      + self.cfg.k_s_quality * (A - 0.5) )
        dlog_sigma = _budget(self.cfg.eta_sigma * drive_sigma, self.cfg.max_dlog_sigma)
        self.sigma = _clip(math.exp(math.log(max(self.sigma, self.cfg.sigma_min + 1e-16)) + dlog_sigma),
                           self.cfg.sigma_min, self.cfg.sigma_max)

        # Resonance clamp
        if (V > self.cfg.V_max_mult * Vstar) or (abs(rho) > self.cfg.rho_max):
            self.alpha = max(self.cfg.alpha_min, self.alpha / 1.25)

        # ---------- parameter update ----------
        self.m = self.mu * self.m + (1.0 - self.mu) * grad
        upd = -self.alpha * self.m
        # optional step cap
        if self.cfg.step_norm_max is not None:
            sn = float(torch.linalg.norm(upd).item())
            if sn > self.cfg.step_norm_max and sn > 0.0:
                upd = upd * (self.cfg.step_norm_max / sn)

        # Version-safe Gaussian noise
        if self.sigma > 0.0:
            n = torch.empty_like(self.theta)
            try:
                n.normal_(mean=0.0, std=self.sigma, generator=self.gen)
            except TypeError:
                n.normal_(mean=0.0, std=self.sigma)
            upd = upd + n

        with torch.no_grad():
            self.theta.add_(upd)
            # parameter clamp (prevents runaway states)
            pm = self.cfg.param_abs_max
            if pm and math.isfinite(pm) and pm > 0:
                self.theta.clamp_(-pm, pm)
        self.theta.requires_grad_(True)

        return {"loss": L, "T": T, "V": V, "S": S, "rho": rho,
                "alpha": self.alpha, "mu": self.mu, "sigma": self.sigma,
                "gamma": float(self.alpha * gnorm), "gnorm": gnorm}

    update = step

# ---------- Ensemble (coefficients-only consensus) ----------
class AEEnsemble:
    def __init__(self, agents: List[AEAgent], beta_weight: float = 2.0, kappa: float = 0.25):
        assert len(agents) >= 2
        self.agents = agents; self.beta_weight = float(beta_weight); self.kappa = float(kappa)

    def step(self, loss_fn: LossFn) -> Dict[str, float]:
        logs = [a.step(loss_fn) for a in self.agents]
        losses = torch.tensor([lg["loss"] for lg in logs], dtype=torch.float64)
        w = torch.softmax(-self.beta_weight * (losses - losses.min()), dim=0).cpu().numpy().tolist()
        bar_alpha = sum(wi * a.alpha for wi, a in zip(w, self.agents))
        bar_mu    = sum(wi * a.mu    for wi, a in zip(w, self.agents))
        bar_sigma = sum(wi * a.sigma for wi, a in zip(w, self.agents))
        for a in self.agents:
            a.alpha = _clip((1.0 - self.kappa) * a.alpha + self.kappa * bar_alpha, a.cfg.alpha_min, a.cfg.alpha_max)
            a.mu    = _clip((1.0 - self.kappa) * a.mu    + self.kappa * bar_mu,    a.cfg.mu_min,    a.cfg.mu_max)
            a.sigma = _clip((1.0 - self.kappa) * a.sigma + self.kappa * bar_sigma, a.cfg.sigma_min, a.cfg.sigma_max)
        return {"mean_loss": float(losses.mean().item()),
                "min_loss": float(losses.min().item()),
                "max_loss": float(losses.max().item()),
                "bar_alpha": bar_alpha, "bar_mu": bar_mu, "bar_sigma": bar_sigma, "w": w}
```

