# Signal-Faithful Long-Only Portfolio Construction

> A research project to replace the current "rank-then-size-by-signal/vol" portfolio builder with a **fixed, signal-agnostic construction system** that (1) stays faithful to the input ranking signal(s), (2) controls turnover by design, and (3) is validated on **simulated markets** so that we can prove, before touching production, that *a better signal mechanically yields better financial outcomes*.

---

## 1. Context

### 1.1 Inputs available at each rebalance date `t`

| Symbol | Description |
|---|---|
| `x_t` | Cross-sectional **ranking** of every stock in the universe (one per signal / horizon: 1M, 3M, 6M, …) |
| `S_t` | Covariance matrix of the full universe, EMA-weighted over roughly the last quarter |
| `c_t` | Boolean **compliance** mask. Non-compliant names are excluded, with no exceptions |
| `w_{t-1}` | Current (drifted) portfolio holdings |

### 1.2 Current production

1. Rank stocks by signal and keep the **top K = 20**.
2. Size each holding by `signal / vol^γ` with `γ = 1`.
3. Rebalance monthly and ignore turnover (about 40% per month).

### 1.3 Observed problem

When names are **ranked/selected** by `signal / vol`, the portfolio ends up chosen mostly for low volatility. A handful of stocks with much lower vol than the rest dominate, the signal is distorted, and performance suffers.

### 1.4 Diagnosis, and why the literature already explains it

This is a well-understood issue, and the fix is a matter of *where* volatility enters the pipeline.

* **Grinold's rule** (Grinold 1994; Grinold & Kahn 2000) states that a forecast should be converted into an expected (residual) return as
  $$\alpha_i = \text{IC} \cdot \sigma_i \cdot z_i,$$
  where `z_i` is the standardized score (mean 0, std 1 across the universe), `σ_i` the (residual) volatility and `IC` the information coefficient.
* Plugging this into an unconstrained mean-variance optimizer with a *diagonal* risk model gives
  $$w_i \propto \frac{\alpha_i}{\sigma_i^2} = \frac{\text{IC}\, z_i}{\sigma_i}.$$
  So **sizing** by `z/σ` (γ = 1) is actually MVO-consistent. The bug is not the sizing.
* The bug is **selecting** on `z/σ`. Selection should be driven by the *score* (that is the forecast). Dividing by σ before selection promotes stocks with mediocre scores but tiny vol, and since `1/σ` has a fat right tail across the cross-section, it swamps the signal. Using **total** vol instead of **residual** vol, and having no position caps, makes it worse.
* The proper measure of "faithfulness to the signal" is the **Transfer Coefficient** (Clarke, de Silva & Thorley 2002): the cross-sectional correlation between the risk-adjusted active weights and the forecasts. The Fundamental Law with constraints,
  $$\text{IR} \approx \text{TC} \cdot \text{IC} \cdot \sqrt{\text{BR}},$$
  tells us exactly what we are trying to maximize: given IC (the signal team's job) and breadth, the construction team's job is to maximize **TC** net of costs. Long-only plus K = 20 caps TC well below 1. Our current selection rule lowers it further.

**Project objective in one line:** maximize the transfer coefficient, net of transaction costs and subject to long-only, compliance and concentration constraints, using a construction rule that is fixed and does not depend on which signal is plugged in.

---

## 2. Design principles

1. **Separate concerns.** Signal → alpha → risk → construction → trading. Each step is a module with a defined interface, testable on its own.
2. **Selection by score, sizing by risk.** Volatility never decides *which* names enter; it only influences *how much*.
3. **Stability over optimality.** Raw sample covariance plus an optimizer is the "Markowitz enigma" (Michaud 1989): it maximizes estimation error. Every optimizer-based variant must use a regularized risk model and/or constraints that act as shrinkage (Jagannathan & Ma 2003).
4. **Turnover is controlled by construction, not by after-the-fact clipping.** Small, continuous adjustments toward a target, not periodic rebuilds.
5. **Conviction is an explicit input.** A "trust the signal more right now" dial (new model, regime change, deliberate switch) must be able to authorize high turnover on demand.
6. **Signal-agnostic and frozen.** Once validated, the system is fixed. A better signal is just a better input, and the system must turn it into better outcomes **monotonically**.
7. **No overfitting.** Design decisions are made on simulated data with known ground truth, and validated on held-out simulations before a single look at real history.

---

## 3. Literature map: what we reuse rather than reinvent

| Brief requirement | Established answer | Key references |
|---|---|---|
| Turn rankings into expected returns | Gaussian rank transform → z-score → `α = IC·σ·z` | Grinold (1994); Grinold & Kahn (2000); Qian, Hua & Sorensen (2007) |
| Measure fidelity to the signal | Transfer Coefficient; constrained Fundamental Law | Clarke, de Silva & Thorley (2002, 2006) |
| Unstable covariance in solvers | Shrinkage (linear / nonlinear), factor risk models, constraints as implicit shrinkage, robust optimization | Ledoit & Wolf (2004, 2020); Jagannathan & Ma (2003); Ceria & Stubbs (2006); Michaud (1989) |
| Simple baselines that are hard to beat | 1/N, score-weighted, inverse-vol | DeMiguel, Garlappi & Uppal (2009) |
| Optimizer-free covariance-aware weighting | Hierarchical Risk Parity | López de Prado (2016) |
| Turnover: "trade partially toward an aim portfolio" | Optimal trading rate with predictable returns and costs; aim portfolio blends current and future alpha | Gârleanu & Pedersen (2013) |
| Turnover: "close worst, open new" | Buffer / hysteresis rules (enter at rank ≤ K, exit at rank > K_exit) | Novy-Marx & Velikov (2016) |
| Turnover inside a solver | L1 trading cost penalty / no-trade region; single- and multi-period convex formulations | Boyd et al. (2017) |
| Cardinality (exactly K names) | Mixed-integer QP, or convex relaxations | Bertsimas & Shioda (2009) |
| Multiple horizons (1M / 3M / 6M) | Combine signals by decay rate; slower signals deserve more weight when trading is costly | Gârleanu & Pedersen (2013); Grinold (2006, "a dynamic model of portfolio management") |
| Testing many ideas without overfitting | Synthetic data with known IC; Deflated Sharpe Ratio; Probability of Backtest Overfitting (CSCV) | Bailey & López de Prado (2014); Bailey, Borwein, López de Prado & Zhu (2017); López de Prado (2018) |

---

## 4. System architecture

```
                ┌──────────────────────────────────────────────────────────────┐
 rankings x_t ─►│ 1. Signal layer      rank → z (Gaussian) → horizon blend → α  │
 (1M,3M,6M)     └───────────────────────────────┬──────────────────────────────┘
                ┌───────────────────────────────▼──────────────────────────────┐
 S_t ──────────►│ 2. Risk layer        shrinkage / factor model, residual vols  │
                └───────────────────────────────┬──────────────────────────────┘
                ┌───────────────────────────────▼──────────────────────────────┐
 compliance c_t►│ 3. Construction      target ("aim") portfolio w*_t            │
                └───────────────────────────────┬──────────────────────────────┘
                ┌───────────────────────────────▼──────────────────────────────┐
 w_{t-1},       │ 4. Trading layer     buffer rule / partial trade / L1 penalty │
 conviction κ_t►│                      → executed w_t                           │
                └───────────────────────────────┬──────────────────────────────┘
                ┌───────────────────────────────▼──────────────────────────────┐
                │ 5. Evaluation        simulated + real backtests, TC, metrics  │
                └──────────────────────────────────────────────────────────────┘
```

### 4.1 Signal layer

* **Rank → score:** `z_i = Φ⁻¹((rank_i − 0.5) / N)` on the compliant universe only, then re-standardize. This is robust to outliers and makes signals of different provenance comparable.
* **Score → alpha:** `α_i = IC_h · σ_i^{res} · z_i`. IC per horizon is estimated (or, in simulation, known). Residual vol, not total vol.
* **Multi-horizon blend:** for horizons `h ∈ {1M, 3M, 6M}`, convert each to a *per-period* alpha and combine:
  $$\alpha_i = \sum_h \beta_h \, \text{IC}_h \, \sigma_i \, z_{i,h}, \qquad \beta_h \text{ reflecting decay and cross-correlation of signals}$$
  Following Gârleanu & Pedersen, slow signals get relatively more weight in the aim portfolio because they are cheaper to trade on. Optimal `β` from the signal covariance (a standard alpha-combination problem, Grinold & Kahn ch. 10–11).
* **Asynchronous updates:** filings arrive on different days. The layer must accept partial updates (only some names re-scored) and carry forward stale scores with an optional staleness decay.

### 4.2 Risk layer

Candidates, all producing a positive-definite, well-conditioned `Σ_t`:

* **R0** Current EMA sample covariance (baseline, expected to be unstable in a solver)
* **R1** Ledoit–Wolf linear shrinkage toward constant correlation
* **R2** Nonlinear (analytical) shrinkage, Ledoit & Wolf (2020)
* **R3** Statistical factor model (PCA, k factors + diagonal specific risk)
* **R4** Diagonal only (residual vol) as a robustness reference

Report condition number, out-of-sample realized vs predicted risk (bias statistic) for each.

### 4.3 Construction layer: candidate constructors

Every constructor maps `(α, Σ, c, constraints) → w*`. Common constraints: long-only, fully invested, compliance (`w_i = 0` if not compliant), max weight `w_max`, optional sector/name caps.

| ID | Constructor | Selection | Sizing | Uses Σ? |
|---|---|---|---|---|
| **C0** | **Production baseline** | top-K by `z/σ` | ∝ `z/σ` | no |
| **C1** | Score-select, score-weight | top-K by `z` | equal, or ∝ `z⁺` | no |
| **C2** | Score-select, risk-size (the direct fix) | top-K by `z` | ∝ `z/σ^γ`, capped at `w_max`, γ ∈ {0, 0.5, 1} | diagonal |
| **C3** | Score-select, HRP-size | top-K by `z` | HRP within selected names, tilted by `z` | yes, no inversion |
| **C4** | Regularized mean-variance | via optimizer | $\max_w \ \alpha^\top w - \lambda\, w^\top \Sigma w$ s.t. constraints, optional `‖w‖₀ ≤ K` | yes (R1–R3) |
| **C5** | Signal-tracking (max TC) | via optimizer | $\min_w \ (w - w^{\text{ideal}})^\top \Sigma (w - w^{\text{ideal}})$ s.t. long-only + constraints, where $w^{\text{ideal}} = \Sigma^{-1}\alpha/\lambda$ (unconstrained long-short) | yes |
| **C6** | Robust MVO | via optimizer | C4 with an uncertainty penalty on α (Ceria & Stubbs 2006) | yes |

Notes:
* C2 is the minimal, low-risk change to production and should be shipped as an interim fix if it wins.
* C5 makes "faithfulness" literal: find the feasible long-only portfolio closest, in risk terms, to what the signal would hold without constraints.
* For C4–C6, the `w_max` cap and long-only constraint are themselves a form of shrinkage; we must test whether a cardinality constraint is needed at all or whether K should emerge from α and λ.

### 4.4 Trading layer: turnover by construction

Three mechanisms, which can be combined:

**T1 – Buffer / hysteresis rule (the "close worst, open new" idea).**
Hold the current names unless their rank falls below `K_exit > K`; fill vacancies with the best new names until K positions. This is exactly Novy-Marx & Velikov's buffer and directly implements the brief's intuition. Parameters: `K`, `K_exit`.

**T2 – Partial trading toward the aim (Gârleanu & Pedersen).**
$$w_t = w_{t-1} + \tau_t\,(w^*_t - w_{t-1}), \qquad \tau_t \in (0, 1].$$
With weekly or daily runs, small `τ` gives the "small adjustments every week" behavior while the long-run portfolio converges to the aim.

**T3 – Turnover inside the solver (for C4–C6).**
$$\max_w \ \alpha^\top w - \lambda\, w^\top \Sigma w - \kappa_t \,\| w - w_{t-1} \|_1$$
The L1 term creates a no-trade region: positions whose marginal alpha does not beat the cost are left alone.

**Conviction dial.** A single scalar `ζ_t ∈ [0, 1]` ("how much do we trust the new signal right now") drives all three mechanisms:
* `κ_t = κ_0 (1 − ζ_t)`, `τ_t = τ_0 + (1 − τ_0) ζ_t`, `K_exit` shrinks toward `K` as `ζ_t → 1`.
* `ζ_t = 1` is a full reset to the aim portfolio (new model deployment, intentional switch).
* By default `ζ_t` is set manually. A stretch goal is to derive it from data, e.g. from the dispersion of the new scores or the change in `z` between dates.

### 4.5 Rebalancing frequency

Monthly, weekly and daily runs are just different settings of the same system. With T1–T3 in place, a higher run frequency should *not* mean higher turnover. It means the same total turnover spread more finely and reacting faster to new information. This is a hypothesis to verify in simulation.

---

## 5. Simulation lab

The core of the project. We need a generator in which **signal quality is a dial with known ground truth**, so we can (a) compare constructors fairly, (b) tune hyperparameters without touching real history, and (c) prove monotonicity in IC.

### 5.1 Market generator

* Universe of `N` stocks (match production, e.g. 1,000–3,000).
* Factor structure: $r_t = B f_t + \varepsilon_t$ with market + sector + style factors.
* Factor returns with volatility clustering (GARCH or regime switching) and fat tails (Student-t).
* **Heterogeneous idiosyncratic vols with a fat right tail of `1/σ`.** This is essential to reproduce the pathology we observed in production. Calibrate the cross-sectional vol distribution to our real universe.
* Compliance mask: random with persistence (names flip status rarely).
* Proportional transaction costs, optionally scaled by vol / liquidity.

Two flavors:
1. **Parametric:** fully synthetic, calibrated to real moments.
2. **Semi-parametric:** block bootstrap of real residuals and factor returns, keeping cross-sectional and temporal dependence.

### 5.2 Signal generator (the dial)

For each horizon `h`, generate scores correlated with future `h`-period residual returns:
$$s_{i,t}^{(h)} = \rho_h \cdot \tilde r^{(h)}_{i,t+1:t+h} + \sqrt{1-\rho_h^2}\;\eta_{i,t}^{(h)},$$
where `r̃` is the cross-sectionally standardized future residual return and `η` is persistent noise (AR(1) across time), giving a realistic signal autocorrelation. This makes the realized IC approximately `ρ_h`.

Extras:
* **Asynchronous refresh:** each stock's score updates on its own (random, filing-like) schedule.
* **Decay:** IC as a function of lag, to test the horizon-blending logic.
* **Misspecification:** signals whose stated IC differs from true IC, to test robustness of the `IC·σ·z` scaling.
* **Signal change event:** a date at which a new, better model replaces the old one, to test the conviction dial.

### 5.3 Experimental protocol

1. **Design set:** seeds `1…N_design`. All exploration and hyperparameter choice happens here.
2. **Freeze:** the winning configuration (risk model, constructor, trading rule, parameters) is written to a versioned config file.
3. **Test set:** fresh seeds `N_design+1…N_design+N_test`, never seen during design. Report full distributions, not point estimates.
4. **IC sweep:** for the frozen system, sweep `ρ ∈ {0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10}` and plot every metric against IC.
5. **Real-data check (once):** a single pass on historical data with the production signal, reporting the Deflated Sharpe Ratio given the number of configurations tried during design, and the PBO via CSCV.

### 5.4 Acceptance criteria

The frozen system is accepted if, on the test seeds:

* **Monotonicity:** median Sharpe, Sortino and annualized return increase monotonically with IC, and the system with `IC = 0` has net Sharpe ≈ 0 or below (no spurious edge from construction alone).
* **Fidelity:** realized TC is higher than C0 at every IC level.
* **Efficiency:** net-of-cost IR beats C0 and C1 at every IC level, with a statistically significant gap across seeds.
* **Turnover:** materially below 40%/month at equal or better net IR, with turnover rising as expected when `ζ_t = 1`.
* **Stability:** weights do not jump when `Σ` is perturbed (bootstrap the risk model and measure weight dispersion).
* **Scaling with frequency:** weekly/daily runs with the trading layer are not worse net of costs than monthly.

---

## 6. Metrics

| Category | Metrics |
|---|---|
| Return / risk | Annualized return, vol, Sharpe, Sortino, Calmar, max drawdown, hit rate |
| Relative | Active return, tracking error, IR vs universe equal-weight and cap-weight |
| Fidelity | Transfer Coefficient, realized IC of holdings, rank correlation between `w` and `z` |
| Concentration | Effective number of names `1/Σw²`, max weight, share of weight in bottom vol decile |
| Trading | Turnover (one-way), cost drag, average holding period |
| Robustness | Deflated Sharpe, PBO, dispersion across seeds, sensitivity to Σ perturbation |

---

## 7. Repository layout (proposed)

```
.
├── README.md
├── configs/
│   ├── frozen/                 # versioned, accepted system configs
│   └── experiments/            # design-phase sweeps
├── src/portopt/
│   ├── signals/                # rank→z, IC scaling, horizon blending, staleness
│   ├── risk/                   # shrinkage, factor models, diagnostics
│   ├── construct/              # C0–C6
│   ├── trade/                  # buffer, partial trade, L1, conviction dial
│   ├── sim/                    # market + signal generators
│   ├── backtest/               # event loop, costs, drift
│   └── metrics/                # performance, TC, DSR, PBO
├── notebooks/                  # analysis only, never source of truth
└── tests/
```

Suggested stack: Python, NumPy/pandas, `cvxpy` (with OSQP/Clarabel; a MIQP solver only if cardinality is kept), `scikit-learn` for shrinkage baselines.

---

## 8. Roadmap

| Phase | Deliverable |
|---|---|
| **0. Baseline** | Reproduce C0 on real data and in the simulator; confirm the low-vol pathology appears in simulation |
| **1. Quick win** | C1/C2 (selection by score); measure TC and IR gain; candidate interim production fix |
| **2. Simulator** | Market + signal generator with IC dial, asynchronous refresh, change events |
| **3. Risk models** | R0–R4 with conditioning and bias diagnostics |
| **4. Constructors** | C3–C6 on the simulator |
| **5. Trading layer** | T1–T3 + conviction dial; frequency study (monthly / weekly / daily) |
| **6. Multi-horizon** | 1M / 3M / 6M blending |
| **7. Freeze & test** | Frozen config, test-seed evaluation, IC sweep, single real-data check with DSR/PBO |
| **8. Production** | Interface spec: inputs (`x_t`, `S_t`, `c_t`, `w_{t-1}`, `ζ_t`) → orders |

---

## 9. Open questions for the team

1. Is K = 20 a hard business constraint, or a proxy for "concentrated"? If the latter, let K emerge from the optimizer plus `w_max`.
2. Is the benchmark absolute return or a universe/index? This decides whether we work with total or active risk.
3. Do we have per-stock liquidity data to build a realistic cost model?
4. Is the covariance `S_t` total-return covariance, and can we also obtain residual vols directly?
5. Who sets the conviction dial `ζ_t`, and is there a governance process around a "reset"?
6. Are there sector, country or single-name limits beyond compliance?

---

## 10. References

* Bailey, D. & López de Prado, M. (2014). The Deflated Sharpe Ratio. *Journal of Portfolio Management*.
* Bailey, D., Borwein, J., López de Prado, M. & Zhu, Q. (2017). The Probability of Backtest Overfitting. *Journal of Computational Finance*.
* Bertsimas, D. & Shioda, R. (2009). Algorithm for cardinality-constrained quadratic optimization. *Computational Optimization and Applications*.
* Boyd, S. et al. (2017). Multi-Period Trading via Convex Optimization. *Foundations and Trends in Optimization*.
* Ceria, S. & Stubbs, R. (2006). Incorporating estimation errors into portfolio selection: Robust portfolio construction. *Journal of Asset Management*.
* Clarke, R., de Silva, H. & Thorley, S. (2002). Portfolio Constraints and the Fundamental Law of Active Management. *Financial Analysts Journal*.
* Clarke, R., de Silva, H. & Thorley, S. (2006). The Fundamental Law of Active Portfolio Management. *Journal of Investment Management*.
* DeMiguel, V., Garlappi, L. & Uppal, R. (2009). Optimal Versus Naive Diversification. *Review of Financial Studies*.
* Gârleanu, N. & Pedersen, L. H. (2013). Dynamic Trading with Predictable Returns and Transaction Costs. *Journal of Finance*.
* Grinold, R. (1994). Alpha is Volatility Times IC Times Score. *Journal of Portfolio Management*.
* Grinold, R. & Kahn, R. (2000). *Active Portfolio Management* (2nd ed.). McGraw-Hill.
* Jagannathan, R. & Ma, T. (2003). Risk Reduction in Large Portfolios: Why Imposing the Wrong Constraints Helps. *Journal of Finance*.
* Ledoit, O. & Wolf, M. (2004). Honey, I Shrunk the Sample Covariance Matrix. *Journal of Portfolio Management*.
* Ledoit, O. & Wolf, M. (2020). Analytical Nonlinear Shrinkage of Large-Dimensional Covariance Matrices. *Annals of Statistics*.
* López de Prado, M. (2016). Building Diversified Portfolios that Outperform Out of Sample. *Journal of Portfolio Management*.
* López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
* Michaud, R. (1989). The Markowitz Optimization Enigma: Is "Optimized" Optimal? *Financial Analysts Journal*.
* Novy-Marx, R. & Velikov, M. (2016). A Taxonomy of Anomalies and Their Trading Costs. *Review of Financial Studies*.
* Qian, E., Hua, R. & Sorensen, E. (2007). *Quantitative Equity Portfolio Management*. Chapman & Hall/CRC.
