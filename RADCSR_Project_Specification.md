# PROJECT SPECIFICATION
# A Risk-Aware, Reliability-Adaptive Framework for Early Detection of Medication Shortages Under Changing Demand Conditions

**Working framework name:** RADCSR — Reliability-Aware Drift-Calibrated Shortage-Risk
**Status:** RADCSR is a *candidate* contribution. It must be justified empirically against baselines. Do not assume novelty.

---

## 0. HOW TO USE THIS DOCUMENT

This is a build specification for an AI coding agent. Implement in the order of the phases below. Each phase must produce saved artefacts (CSV/JSON/PNG) before the next phase begins. Never skip Phase 2 (baseline diagnosis) and jump to Phase 7 (RADCSR).

**Non-negotiable rules**

1. All evaluation uses **chronological** splits. No random shuffling of time series. No future leakage.
2. The **forecasting model is held fixed** when comparing reliability/calibration/risk methods. Only the reliability layer changes.
3. Inventory, orders, lead times, stockouts, and VEN labels are **simulated**. They must never be described as observed.
4. Every table/figure must be reproducible from a fixed random seed (`SEED = 42`).
5. Every baseline must be evaluated on **identical test streams** and at **matched false-alarm rates** where detectors are compared.
6. Report failures honestly. If a baseline wins, say so.

---

## 1. RESEARCH PROBLEM AND QUESTIONS

### 1.1 Problem statement

A pharmacy demand forecast can remain numerically "accurate on average" while its **predictive distribution becomes untrustworthy** after a demand regime change. A stockout probability computed from a stale or miscalibrated distribution can therefore be dangerously wrong even when MAE looks acceptable.

The project asks: **when forecast reliability deteriorates, how should shortage-risk estimates and clinical prioritisation adapt — and does a reliability-aware mechanism outperform established alternatives?**

### 1.2 Primary research question

> Under changing medication-demand conditions, do existing forecast-reliability monitoring and uncertainty-calibration methods translate reliably into calibrated, clinically meaningful shortage-risk estimates — and if not, can a reliability-adaptive, drift-calibrated mechanism improve that translation without excessive conservatism?

### 1.3 Sub-questions

| ID | Question |
|---|---|
| RQ1 | Which forecasting baseline (Seasonal Naive, SARIMA, XGBoost, LightGBM, Quantile LightGBM) is most accurate and best calibrated on the target series? |
| RQ2 | Do rolling error monitors (WAPE/MAE), CUSUM, and ADWIN detect forecast degradation early enough, with acceptable false alarms? |
| RQ3 | Do calibration methods (rolling residual quantile, EWMA variance, ACI, DtACI, AdaPI, EnbPI) maintain coverage after abrupt and gradual shifts? |
| RQ4 | Does improved interval coverage actually improve **stockout-probability calibration** (Brier score) and **warning lead time**? |
| RQ5 | Do reliability signals add information beyond coverage feedback alone? |
| RQ6 | Does RADCSR improve shortage-risk calibration and clinically weighted prioritisation versus strong baselines, without inflating inventory? |

### 1.4 Hypotheses (to be tested, not assumed)

- **H1:** Static probabilistic forecasts become miscalibrated after abrupt surges and gradual drift.
- **H2:** Rolling WAPE and CUSUM detect degradation but do not produce calibrated shortage probabilities.
- **H3:** ADWIN detects demand drift that does not necessarily correspond to forecast degradation.
- **H4:** ACI restores interval coverage but does not automatically restore stockout-probability calibration.
- **H5:** Combining forecast-error degradation, directional bias, and coverage failure gives better reliability discrimination than any single signal.
- **H6:** RADCSR improves Brier score and warning lead time versus the strongest baseline without increasing excess inventory beyond a stated tolerance.

---

## 2. PHASE 1 — DATASET CONSTRUCTION

### 2.1 Source

Primary: pharmacy transactional dataset (Kaggle pharmacy transactions / Pharma Sales Data). Each row is a transaction with a date, a medicine/product identifier, and a quantity.

### 2.2 Target series definition

Aggregate to a medicine-level demand series:

```
DATE | MEDICINE_ID | DEMAND_QTY
```

Rules:
- `DEMAND_QTY = sum(quantity)` per `DATE × MEDICINE_ID`.
- Negative quantities = returns/refunds. Compute **net demand**, then floor at zero:
  `MODEL_DEMAND = max(0, net_demand)`.
- Reindex to a complete calendar (daily or weekly) and fill missing periods with **0**, not NaN.
- Keep the raw audit table separately; never overwrite raw data.

### 2.3 Granularity decision

- Daily if the medicine has ≥ 60% non-zero days.
- Weekly otherwise.
- Do **not** use monthly unless ≥ 36 months exist.

### 2.4 Series inclusion filter

Keep a medicine only if:
- ≥ 18 periods of history (weekly) or ≥ 180 days (daily),
- ≥ 6 non-zero demand periods,
- no unresolved unit inconsistency,
- < 5% unexplained extreme values.

### 2.5 Outlier handling

Flag, do not auto-delete:

```
Q1, Q3 per medicine
IQR = Q3 - Q1
EXTREME_HIGH = demand > Q3 + 3*IQR
```

Decision rule:
- Confirmed data error → remove, log it.
- Genuine bulk issue / surge → **retain** (these are the events the project studies).
- Ambiguous → winsorise at the medicine's 99th percentile and record the count.

### 2.6 Required dataset diagnostics (save as `output/phase1_diagnostics.json`)

- number of medicines retained / dropped and why,
- total periods, date range,
- % zero demand,
- negative-value count and net effect,
- extreme-value count before/after treatment,
- per-medicine summary: mean, median, CV, intermittency ratio, ADI (average demand interval), CV².

Classify each series:
- **Smooth:** ADI < 1.32 and CV² < 0.49
- **Intermittent:** ADI ≥ 1.32
- **Lumpy:** ADI ≥ 1.32 and CV² ≥ 0.49

This classification is used later to test whether reliability methods fail on sparse demand.

---

## 3. PHASE 2 — FEATURE ENGINEERING

Create features **only from information available at the forecast origin**.

| Feature | Definition |
|---|---|
| `lag_1 … lag_k` | Demand at t−1 … t−k (k = 7 daily / 4 weekly) |
| `roll_mean_w` | Rolling mean over window w |
| `roll_std_w` | Rolling std over window w |
| `roll_max_w` | Rolling max over window w |
| `zero_ratio_w` | Fraction of zero-demand periods in window w |
| `month`, `quarter`, `dow` | Calendar encodings |
| `sin/cos(2π·doy/365)` | Annual seasonality |
| `trend_index` | Monotone time index |
| `adi`, `cv2` | Intermittency features |
| `days_since_last_issue` | Gap feature for sparse series |

**Leakage guard:** every rolling/lag feature must be shifted so that row *t* contains no value from *t* or later.

---

## 4. PHASE 3 — FORECASTING BASELINES

Train all five. Evaluate on the same chronological test window. Produce `output/phase3_forecast_metrics.csv`.

### 4.1 Seasonal Naive

\[
\hat{D}_{t+1} = D_{t+1-s}
\]

with \(s=7\) (daily) or \(s=52\) (weekly). Mandatory reference: if a complex model cannot beat it, that is a finding.

### 4.2 SARIMA

\[
\phi(B)\Phi(B^s)(1-B)^d(1-B^s)^D y_t = \theta(B)\Theta(B^s)\varepsilon_t
\]

Start with `SARIMA(1,0,1)(1,0,1,s)`, select orders by AIC on the training window only. Fit per medicine (or per top-N medicine). Produce point forecast plus 80%/90% intervals.

### 4.3 XGBoost (point)

```python
XGBRegressor(
    n_estimators=1000, learning_rate=0.05, max_depth=6,
    min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
    objective="reg:squarederror", eval_metric="rmse",
    tree_method="hist", random_state=42, n_jobs=-1,
    early_stopping_rounds=50
)
```

### 4.4 LightGBM (point)

```python
LGBMRegressor(
    n_estimators=2000, learning_rate=0.03, num_leaves=31,
    min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
    objective="regression", random_state=42, n_jobs=-1
)
```

### 4.5 Quantile LightGBM (PRIMARY probabilistic model)

Train three models with `objective="quantile"`, `alpha ∈ {0.10, 0.50, 0.90}`.

\[
\min_{\theta}\ \sum_{t} \rho_{q}\left(y_t - \hat{q}_t(\theta)\right),
\qquad
\rho_q(e)=
\begin{cases}
q\,e, & e \ge 0\\
(q-1)\,e, & e < 0
\end{cases}
\]

Output: \(\hat{q}_{0.10}, \hat{q}_{0.50}, \hat{q}_{0.90}\) per medicine-period.

**This model is frozen for all subsequent reliability experiments.** Record its version hash.

### 4.6 Forecast metrics

| Metric | Formula |
|---|---|
| MAE | \(\frac{1}{n}\sum|y_t-\hat{y}_t|\) |
| RMSE | \(\sqrt{\frac{1}{n}\sum(y_t-\hat{y}_t)^2}\) |
| WAPE | \(\frac{\sum|y_t-\hat{y}_t|}{\sum|y_t|}\times 100\) |
| Pinball loss | \(\frac{1}{n}\sum \rho_q(y_t-\hat{q}_t)\) per quantile |
| PICP | \(\frac{1}{n}\sum \mathbf{1}\{L_t \le y_t \le U_t\}\) |
| MPIW | \(\frac{1}{n}\sum (U_t-L_t)\) |
| Coverage error | \(|\text{PICP} - 0.90|\) |

Do **not** use MAPE (undefined at zero demand).

---

## 5. PHASE 4 — INVENTORY AND SHORTAGE SIMULATION

No public dataset provides real inventory. Simulate transparently and label it.

### 5.1 State variables

\[
I_{t+1} = I_t + R_t - D_t
\]

with \(R_t\) = receipts arriving at \(t\), \(D_t\) = realised demand.

### 5.2 Reorder policy (periodic review, order-up-to)

\[
S_t = \hat{\mu}_{L} + z_{\alpha}\hat{\sigma}_{L}
\]

where \(\hat{\mu}_L,\hat{\sigma}_L\) are forecast mean and std over lead time \(L\), and \(z_\alpha\) is the service-level quantile.

Order when \(I_t \le s\) (reorder point), quantity \(Q_t = S_t - I_t\).

### 5.3 Lead time

\[
L_t \sim \text{DiscreteUniform}(L_{\min}, L_{\max})
\]

Base: \(L \in \{5,6,7\}\) days. Disruption scenario: \(L \in \{14,\dots,21\}\).

### 5.4 Stockout and shortage

\[
\text{STOCKOUT}_t = \mathbf{1}\{D_t > I_t + R_t\}
\]
\[
\text{SHORTAGE\_QTY}_t = \max(0,\ D_t - I_t - R_t)
\]

### 5.5 Baseline parameter grid (sensitivity required)

| Parameter | Values |
|---|---|
| Initial stock (days of demand) | 15, 30, 45 |
| Reorder point (days) | 10, 20, 30 |
| Target cover (days) | 30, 45, 60 |
| Lead time (days) | 5–7, 7–10, 14–21 |
| Service level | 0.90, 0.95, 0.99 |

Run the full pipeline on at least **three** parameter combinations and report sensitivity. Never report a single arbitrary configuration as definitive.

### 5.6 Clinical priority (simulated)

Assign VEN by documented rule, e.g.:

```
V (Vital):        top tercile demand AND no substitute
E (Essential):    mid tercile OR partial substitute
N (Non-essential): low demand AND substitutable
```

Store `VEN_CLASS` and `SUBSTITUTABLE ∈ {0,1}`. State clearly these are rule-based, not clinically validated.

---

## 6. PHASE 5 — DEMAND SCENARIOS

Generate controlled scenarios on the same base series. Save scenario labels.

| Scenario | Construction |
|---|---|
| **S1 Stable** | Unmodified history |
| **S2 Abrupt surge** | Multiply demand by 1.5–2.0 for 2–4 consecutive periods starting at a fixed index |
| **S3 Gradual drift** | Multiply by \((1+g)^t\), \(g \in \{0.01, 0.03\}\) per period |
| **S4 Volatility increase** | Keep mean, multiply residual std by 2–3 |
| **S5 Bias shift** | Add constant offset \(+\delta\) without changing variance |
| **S6 Intermittent** | Restrict to lumpy/sparse series |
| **S7 Lead-time disruption** | Demand unchanged, \(L\) extended to 14–21 days |
| **S8 Correlated surge** | AR(1) positive autocorrelation in the surge |

Each scenario must record `SCENARIO_ID`, `SHIFT_START_INDEX`, `SHIFT_MAGNITUDE`.

---

## 7. PHASE 6 — RELIABILITY AND CALIBRATION BASELINES

All methods operate on the frozen Quantile LightGBM forecasts. Each produces a per-period reliability signal and, where applicable, an adjusted predictive distribution.

### 7.1 Rolling error monitoring

\[
\text{WAPE}_t^{(W)} = \frac{\sum_{i=t-W+1}^{t}|y_i-\hat{y}_i|}{\sum_{i=t-W+1}^{t}|y_i|}\times 100
\]

Alarm when \(\text{WAPE}_t > \mu_{\text{base}} + k\sigma_{\text{base}}\) or relative ratio \(> \tau\).

### 7.2 CUSUM on standardised absolute residuals

\[
z_t = \frac{|e_t|-\mu_0}{\sigma_0},\qquad
S_t = \max\left(0,\ S_{t-1} + z_t - k\right)
\]

Alarm when \(S_t > h\). Calibrate \(h\) to a target in-control ARL by simulation.

### 7.3 ADWIN

Maintain adaptive window; split into \(W_0, W_1\); alarm when

\[
|\hat{\mu}_0 - \hat{\mu}_1| > \epsilon_{\text{cut}}
\]

Apply to (a) raw demand, (b) absolute residuals, (c) signed residuals — three separate experiments.

### 7.4 Rolling residual quantile interval

\[
s_t = |y_t - \hat{y}_t|,\qquad
q_t = Q_{1-\alpha}\left(s_{t-W+1..t}\right)
\]
\[
PI_t = [\hat{y}_t - q_t,\ \hat{y}_t + q_t]
\]

### 7.5 EWMA residual variance

\[
m_t = \lambda m_{t-1} + (1-\lambda)e_t
\]
\[
v_t = \lambda v_{t-1} + (1-\lambda)(e_t - m_{t-1})^2
\]

Interval: \(\hat{y}_t \pm z_{1-\alpha/2}\sqrt{v_t}\).

### 7.6 ACI (Adaptive Conformal Inference)

\[
\alpha_{t+1} = \alpha_t + \gamma\left(\alpha - \text{err}_t\right),
\qquad \text{err}_t = \mathbf{1}\{y_t \notin C_t\}
\]

### 7.7 DtACI (dynamic learning rate)

Replace fixed \(\gamma\) with an online-tuned \(\gamma_t\) that adapts to the observed shift rate. Implement per Gibbs & Candès (2024).

### 7.8 AdaPI (direct interval scaling)

\[
PI_t = [\hat{y}_t - F_t z_t,\ \hat{y}_t + F_t z_t]
\]
\[
F_t = g(C_t, L)
\]

with linear variant:
\[
F_t = 1 - M\frac{L - 100C_t}{L-100}
\]

Implement logarithmic and HLI variants as ablations.

### 7.9 EnbPI

Bootstrap ensemble of \(B\) models; interval from out-of-bag residual quantile:
\[
PI_t = [\hat{y}_t - Q_{1-\alpha}(R_t),\ \hat{y}_t + Q_{1-\alpha}(R_t)]
\]

### 7.10 Reliability signal outputs (required table)

For each method and period, store:

```
DATE | MEDICINE_ID | METHOD | ALARM | ADJ_LOWER | ADJ_UPPER | COVERAGE_W | INTERVAL_WIDTH
```

---

## 8. PHASE 7 — SHORTAGE-RISK ESTIMATION BASELINES

All risk estimators consume a predictive distribution (static or adjusted) and inventory state.

### 8.1 Core definition

Lead-time demand:
\[
D_{t,t+L} = \sum_{k=1}^{L} D_{t+k}
\]

Shortage event:
\[
Y_t = \mathbf{1}\{D_{t,t+L} > I_t + O_t\}
\]

Stockout probability:
\[
R_t = P\left(D_{t,t+L} > I_t + O_t\right)
\]

### 8.2 Scenario / Monte Carlo baseline

\[
\hat{R}_t = \frac{1}{N}\sum_{j=1}^{N}\mathbf{1}\{D^{(j)}_{t,t+L} > I_t + O_t\},\quad N=1000
\]

Use block bootstrap (block length \(b=7\)) to preserve temporal dependence.

### 8.3 Robust optimisation baseline

\[
S^{RO}_t = \max_{D_L \in \mathcal{U}} \left(D_L - I_t - O_t\right)
\]

with box or budgeted uncertainty set.

### 8.4 Wasserstein DRO baseline

\[
\hat{R}^{DRO}_t = \sup_{Q \in \mathcal{P}_t} Q\left(D_{t,t+L} > I_t + O_t\right)
\]
\[
\mathcal{P}_t = \{Q : W_p(Q, \hat{P}_t) \le \varepsilon_t\}
\]

Implement fixed-radius and rolling-window variants.

### 8.5 Severity measures

\[
\text{ESQ}_t = \mathbb{E}\left[\max(0, D_{t,t+L} - I_t - O_t)\right]
\]
\[
\text{CVaR}_{0.95}(L) = \frac{1}{1-\beta}\mathbb{E}\left[L \mid L \ge \text{VaR}_\beta(L)\right]
\]

### 8.6 Newsvendor reference policy

\[
Q^* = F^{-1}\left(\frac{C_u}{C_u + C_o}\right)
\]

Use as the inventory-decision reference, not as a risk estimator.

---

## 9. PHASE 8 — RADCSR: THE PROPOSED FRAMEWORK

RADCSR is a **reliability-adaptive, drift-calibrated shortage-risk layer**. It does not replace the forecaster. It modifies how the predictive distribution is trusted and translated into risk.

### 9.1 Component 1 — Multi-signal reliability state

At each period \(t\), compute four normalised reliability signals:

**(a) Error degradation**
\[
r^{(E)}_t = \frac{\text{WAPE}^{(W)}_t}{\text{WAPE}_{\text{base}}}
\]

**(b) Directional bias**
\[
r^{(B)}_t = \frac{\bar{e}^{(W)}_t}{\hat{\sigma}^{(W)}_t},
\qquad \bar{e}^{(W)}_t = \frac{1}{W}\sum_{i=t-W+1}^{t}(y_i-\hat{y}_i)
\]

Positive \(r^{(B)}\) means systematic under-forecasting — the shortage-relevant direction.

**(c) Coverage failure**
\[
r^{(C)}_t = \frac{0.90 - \text{PICP}^{(W)}_t}{0.90}
\]

**(d) Drift severity**
\[
r^{(D)}_t = \frac{|\hat{\mu}_{0}-\hat{\mu}_{1}|}{\hat{\sigma}_{\text{pooled}}}
\]

### 9.2 Component 2 — Reliability index

Combine into a bounded index:

\[
R^{rel}_t = \sigma\left(w_E \tilde{r}^{(E)}_t + w_B \tilde{r}^{(B)}_t + w_C \tilde{r}^{(C)}_t + w_D \tilde{r}^{(D)}_t\right)
\]

where \(\tilde{r}\) are clipped to \([0,3]\) and scaled, and \(\sigma(\cdot)\) is a logistic squashing function. Weights \(w\) are set by validation, **not** hand-tuned on test data. Report weight sensitivity.

**Critical design requirement:** \(r^{(B)}\) must enter with an asymmetric penalty, because under-forecasting raises shortage risk while over-forecasting raises expiry risk.

### 9.3 Component 3 — Reliability-adaptive predictive distribution

Adjust the predictive spread:

\[
\hat{\sigma}^{adj}_t = \hat{\sigma}_t \cdot \left(1 + \kappa \cdot R^{rel}_t\right)
\]

and shift the location by the estimated bias:

\[
\hat{\mu}^{adj}_t = \hat{\mu}_t + \beta \cdot \bar{e}^{(W)}_t
\]

The adjusted lead-time demand distribution is then used for risk:

\[
\hat{R}^{RADCSR}_t = P_{\text{adj}}\left(D_{t,t+L} > I_t + O_t\right)
\]

**Conservatism control:** cap the inflation factor:
\[
1 + \kappa R^{rel}_t \le \gamma_{\max}
\]
and report results for \(\gamma_{\max} \in \{1.5, 2.0, 3.0\}\).

### 9.4 Component 4 — Clinical weighting

Define clinical consequence weight:
\[
c_j = \alpha_V \cdot \text{VEN}_j + \alpha_S \cdot (1-\text{SUBST}_j)
\]

Risk-adjusted priority score:
\[
\Pi_{j,t} = \hat{R}^{RADCSR}_{j,t} \cdot c_j
\]

**Do not** present \(\Pi\) as a calibrated probability. It is a prioritisation score. Report it separately from \(\hat{R}\).

### 9.5 Component 5 — Alert rule

Alert when:
\[
\hat{R}^{RADCSR}_{j,t} > \tau_R \quad \text{AND} \quad R^{rel}_{j,t} > \tau_{rel}
\]

Two-threshold design prevents alerting on high risk that is itself unreliable. Report the \((\tau_R, \tau_{rel})\) grid and the chosen operating point.

### 9.6 Ablations (mandatory)

To prove each component matters, run:

| Variant | Removed component |
|---|---|
| RADCSR-noBias | drop \(r^{(B)}\) |
| RADCSR-noCoverage | drop \(r^{(C)}\) |
| RADCSR-noDrift | drop \(r^{(D)}\) |
| RADCSR-noClinical | drop \(c_j\) |
| RADCSR-noCap | remove \(\gamma_{\max}\) |
| RADCSR-singleSignal | use only \(r^{(E)}\) |

If an ablation performs equally well, the removed component is not justified.

---

## 10. PHASE 9 — EVALUATION

### 10.1 Forecast quality

MAE, RMSE, WAPE, pinball loss.

### 10.2 Calibration

\[
\text{PICP},\quad \text{MPIW},\quad |\text{PICP}-0.90|,\quad \text{CRPS}
\]

### 10.3 Shortage-probability quality

\[
\text{Brier} = \frac{1}{T}\sum_{t}\left(\hat{R}_t - Y_t\right)^2
\]

Also: log loss, reliability diagram, expected calibration error (ECE).

### 10.4 Detection quality

- Precision, recall, F1 for stockout events.
- **Warning lead time**: periods between first alarm and first actual stockout.
- **False-alarm rate** during stable periods.
- Detection delay after `SHIFT_START_INDEX`.

### 10.5 Inventory consequences

- Expected shortage quantity.
- CVaR₀.₉₅ of shortage.
- Excess inventory (units and days of cover).
- Stockout cost vs holding cost (parameterised).

### 10.6 Clinical prioritisation quality

- Recall of Vital-class shortages.
- Precision@K for top-K prioritised medicines.
- Rank correlation between \(\Pi\) and realised clinical harm proxy.

### 10.7 Statistical testing

- Paired bootstrap over test periods for Brier/MAE differences.
- Report 95% CIs.
- Correct for multiple comparisons (Holm–Bonferroni) when comparing many methods.

---

## 11. PHASE 10 — EXPERIMENT MATRIX

Run every reliability method × every scenario × three inventory configurations.

| | S1 Stable | S2 Surge | S3 Drift | S4 Volatility | S5 Bias | S6 Intermittent | S7 Lead-time | S8 Correlated |
|---|---|---|---|---|---|---|---|---|
| Static | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Rolling quantile | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| EWMA variance | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| ACI | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| DtACI | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| AdaPI | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| EnbPI | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Scenario risk | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| RO | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Wasserstein DRO | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **RADCSR** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

Total runs: 11 methods × 8 scenarios × 3 configs = 264. Log every run with a unique `RUN_ID`.

---

## 12. PHASE 11 — DECISION GATE

Before claiming RADCSR as a contribution, all four must hold:

1. **Statistical improvement** — lower Brier score than the strongest baseline (paired bootstrap, p < 0.05).
2. **Robustness improvement** — lower shortage underestimation during degraded conditions.
3. **Conservatism control** — no more than a stated % increase in excess inventory during stable periods.
4. **Practical improvement** — earlier warning lead time without unacceptable false alarms.

If any fails, report it. A negative result is still a valid dissertation finding: "existing methods already suffice under condition X."

---

## 13. REPOSITORY STRUCTURE

```
project/
├── config/
│   └── config.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── forecasting/
│   │   ├── seasonal_naive.py
│   │   ├── sarima.py
│   │   ├── xgboost_model.py
│   │   ├── lightgbm_model.py
│   │   └── quantile_lgbm.py
│   ├── reliability/
│   │   ├── rolling_error.py
│   │   ├── cusum.py
│   │   ├── adwin.py
│   │   ├── rolling_quantile.py
│   │   ├── ewma_variance.py
│   │   ├── aci.py
│   │   ├── dtaci.py
│   │   ├── adapi.py
│   │   └── enbpi.py
│   ├── inventory/
│   │   ├── simulator.py
│   │   └── scenarios.py
│   ├── risk/
│   │   ├── scenario_risk.py
│   │   ├── robust.py
│   │   ├── dro.py
│   │   └── radcsr.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── statistical_tests.py
│   └── pipeline.py
├── notebooks/
├── output/
│   ├── figures/
│   ├── tables/
│   └── logs/
└── tests/
```

---

## 14. CONFIGURATION FILE

```yaml
seed: 42

data:
  granularity: weekly          # daily | weekly
  min_periods: 18
  min_nonzero: 6
  outlier_iqr_multiplier: 3.0
  winsorize_quantile: 0.99

forecast:
  horizon: 1
  quantiles: [0.10, 0.50, 0.90]
  train_frac: 0.60
  val_frac: 0.20
  test_frac: 0.20

reliability:
  window: 8
  cusum_k: 0.5
  cusum_h: 5.0
  adwin_delta: 0.002
  ewma_lambda: 0.9
  aci_gamma: 0.01
  target_coverage: 0.90

inventory:
  init_stock_days: 30
  reorder_point_days: 20
  target_cover_days: 45
  lead_time_days: [5, 6, 7]
  service_level: 0.95

radcsr:
  weights: {error: 0.30, bias: 0.30, coverage: 0.25, drift: 0.15}
  kappa: 0.5
  beta: 1.0
  gamma_max: 2.0
  tau_risk: 0.30
  tau_reliability: 0.50

scenarios:
  surge_multiplier: [1.5, 2.0]
  surge_duration: [2, 4]
  drift_rate: [0.01, 0.03]
  volatility_multiplier: [2.0, 3.0]
  bias_offset_std: 1.0
```

---

## 15. REQUIRED OUTPUTS

### Tables
1. `phase1_diagnostics.csv` — series inventory and quality.
2. `phase3_forecast_metrics.csv` — model comparison.
3. `phase6_reliability_signals.csv` — per-period signals.
4. `phase7_risk_estimates.csv` — per-period risk per method.
5. `phase9_evaluation.csv` — full metric matrix.
6. `phase10_ablation.csv` — RADCSR component ablations.
7. `phase11_sensitivity.csv` — inventory parameter sensitivity.

### Figures
1. Demand series with scenario shift markers.
2. Forecast vs actual for each baseline (sample medicines).
3. Rolling WAPE / CUSUM / ADWIN alarm timelines.
4. Coverage over time: static vs ACI vs RADCSR.
5. Reliability diagram for stockout probability.
6. Brier score by scenario, all methods.
7. Warning lead time vs false-alarm rate (ROC-style).
8. Shortage severity vs excess inventory trade-off.
9. Ablation bar chart.
10. Sensitivity heatmap.

---

## 16. WRITING CONSTRAINTS FOR THE FINAL REPORT

**Permitted claims:**
- "Real pharmacy transaction data were used to construct demand series."
- "Inventory, lead times, stockouts, and VEN labels were simulated using stated assumptions."
- "Method X reduced Brier score by Y (95% CI ...) under scenario Z."

**Forbidden claims:**
- "RADCSR is novel" — until the literature review and experiments support it.
- "The model predicts real stockouts" — stockouts are simulated.
- "VEN classification reflects clinical priority" — it is rule-based.
- "Daily/real-time detection" if the data are weekly.
- "Existing methods do not work" — write "existing methods do not jointly translate reliability into calibrated shortage risk."

---

## 17. BUILD ORDER FOR THE AGENT

1. `data_loader.py` + `preprocessing.py` → Phase 1 diagnostics.
2. `features.py` → leakage tests.
3. `seasonal_naive.py`, `sarima.py`, `xgboost_model.py`, `lightgbm_model.py`, `quantile_lgbm.py` → Phase 3 metrics.
4. `inventory/simulator.py` + `scenarios.py` → simulated stockouts.
5. `reliability/*` → Phase 6 signals.
6. `risk/scenario_risk.py`, `robust.py`, `dro.py` → Phase 7 baselines.
7. `evaluation/metrics.py` → Phase 9 harness.
8. Run full baseline matrix. **Stop and report.**
9. Only then implement `risk/radcsr.py`.
10. Run ablations and sensitivity.
11. Generate all figures and tables.

**Do not implement RADCSR before step 8 is complete and its results are reviewed.**

---

## 18. MINIMUM VIABLE DELIVERABLE

If time is constrained, the smallest defensible study is:

- 30 medicines, weekly granularity, 2 years.
- Seasonal Naive + Quantile LightGBM.
- Static, rolling quantile, ACI, scenario risk.
- Scenarios S1, S2, S3, S7.
- Metrics: WAPE, PICP, Brier, warning lead time, false-alarm rate.
- One inventory configuration plus a 3-point sensitivity check.

This is enough to demonstrate whether the reliability-to-risk translation problem exists. RADCSR is only justified if it does.
