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

