# Model Card — Credit Default Risk Scorecard

**Version** 1.0.0 · **Type** Binary classifier (XGBoost + isotonic calibration) ·
**Seed** 42 · **License / status** Portfolio project, **not approved for production use**

**Deployed at** [https://credit-risk-api-1dbj.onrender.com/docs](https://credit-risk-api-1dbj.onrender.com/docs) — a public, unauthenticated demo endpoint. It is a
demonstration of the serving architecture, **not a lending service**; see §1 and §6.1 for
why its output must not inform a real credit decision.

---

## 1. Intended use

### What it is for

Estimating the probability that a loan applicant will default, **at the moment of
application**, to support one of three decisions:

1. **Approve / decline**, against a cut-off derived from an explicit cost assumption.
2. **Risk-based pricing** — the output is a calibrated probability, so it can feed an
   expected-loss calculation directly.
3. **Portfolio segmentation** — the decile table supports tiering an existing book.

### Who it is for

Credit risk analysts and underwriting systems. The output is a recommendation and a
probability, **not an automated decision**. Every decline should be reviewable by a human,
and every applicant is entitled to an explanation (see §7).

### What it is explicitly *not* for

| Not for | Why |
|---|---|
| **Any real lending decision** | Trained on synthetic data. See §6.1 — this is disqualifying, not a caveat |
| Collections or recovery scoring | Trained on origination-time features only; says nothing about a loan already in trouble |
| Mid-life re-scoring of an existing loan | It deliberately cannot see repayment behaviour after disbursal |
| Fraud detection | Default and fraud are different targets with different signals |
| Pricing outside the observed range | Income ₹0.1M–₹12.0M, loans ₹0.05M–₹47.8M, ages 18–70. Outside that it extrapolates |
| Populations unlike the training data | Indian retail borrowers, 2019–2024, ten metros |

---

## 2. Training data

### Source

Three CSVs of 50,000 rows each, joined one-to-one on `cust_id`:

| Table | Contents |
|---|---|
| `customers.csv` | Demographics, income, residence, location |
| `loans.csv` | Product, amounts, tenure, dates, and the `default` target |
| `bureau_data.csv` | Account counts, delinquency history, DPD, enquiries, utilisation |

**Provenance is unknown and undocumented.** No data dictionary, no collection methodology,
no definition of `default`. That alone would block production sign-off.

### Composition

| | |
|---|---|
| Rows after cleaning | **49,992** (of 50,000) |
| Target | `default`, boolean |
| **Base default rate** | **8.594%** (4,297 defaults) |
| Date range | 2019-07-24 → 2024-07-22 (1,825 days) |
| Geography | 10 Indian cities across 9 states |
| Gender | 60% M / 40% F |
| Employment | 64% self-employed / 36% salaried |
| Residence | 56% Owned / 24% Mortgage / 20% Rented |

### Cleaning applied

| Issue | Rows | Action |
|---|---|---|
| `loan_purpose` = `Personaal` | 22 | Normalised to `Personal` |
| `residence_type` null | 62 | Explicit `Unknown` category (not mode-imputed) |
| All money columns = 0 | 8 | Dropped |
| `processing_fee` corrupted | 7 | Repaired as `gst / 9`, confirmed by the disbursement identity |

Full reasoning in [`../README.md`](../README.md#data-and-cleaning); every count is asserted
in `tests/test_data.py`.

### Splits

* **Stratified 70/15/15** — 34,994 train / 7,499 validation / 7,499 test, default rate
  held at 8.59% in each.
* **Out-of-time** — trained on loans disbursed before 2023-07-23, tested on the 10,015
  disbursed after. This is the split a risk team actually cares about.

---

## 3. Features

22 features (17 numeric, 5 categorical) from 20 raw application-time inputs. Full list and
credit intuition in [`../README.md`](../README.md#features).

**Top 5 by mean |SHAP|:**

| Rank | Feature | mean abs SHAP |
|---|---|---|
| 1 | `credit_utilization_ratio` | 4.938 |
| 2 | `delinquency_ratio` | 2.305 |
| 3 | `loan_to_income` | 1.431 |
| 4 | `residence_type_Owned` | 0.978 |
| 5 | `residence_type_Rented` | 0.725 |

### Excluded, and why

* **Post-disbursal** (`principal_outstanding`, `processing_fee`, `gst`,
  `net_disbursement`, `installment_start_dt`) — unavailable at decision time. Verified by
  ablation: adding `principal_outstanding` changes AUC by **−0.0001**.
* **Identifiers** (`cust_id`, `loan_id`).
* **Geography** (`city`, `state`, `zipcode`) — flat default rates (8.1–9.1%), and location
  is a well-documented proxy for protected characteristics.
* **`gender`** — protected characteristic. Excluded from the feature set, retained in the
  raw frame solely to audit fairness (§5).

---

## 4. Performance

Test set: 7,499 loans, 8.59% base rate.

| Metric | Logistic baseline | **Final model** | Out-of-time |
|---|---|---|---|
| ROC-AUC | 0.9834 | **0.9885** | 0.9855 |
| PR-AUC | 0.8598 | **0.8862** | 0.8759 |
| Gini | 0.9668 | **0.9771** | 0.9711 |
| KS | 88.14 | **90.30** | 90.40 |
| Brier | 0.0538 | **0.0240** | 0.0243 |

At the cost-optimal threshold **0.080** (FN cost 0.60, FP cost 0.06 of exposure):

| | |
|---|---|
| Recall (defaults caught) | 96.27% |
| Precision | 58.16% |
| F1 | 0.7251 |
| Approval rate | 85.78% |
| Bad rate in approved book | 0.37% |
| Confusion matrix | TN 6,409 · FP 446 · FN 24 · TP 620 |

**Calibration.** Isotonic, selected on a held-out half of validation (never on test).
Brier 0.0381 → **0.0240**; expected calibration error 0.0479 → **0.0063**, a 7.6× reduction.

**Rank ordering** is monotone across all ten deciles; decile 1 captures 560 of 644 test
defaults (74.67% event rate, 8.69× lift).

---

## 5. Performance by segment

Same 0.08 threshold everywhere. Full tables in `reports/tables/slice_*.csv`.

### Gender (protected — not a model input)

| Segment | n | Base rate | AUC | Approval rate | Recall | Precision | Mean PD |
|---|---|---|---|---|---|---|---|
| Male | 4,515 | 8.39% | 0.9902 | 86.45% | 96.57% | 59.80% | 0.0862 |
| Female | 2,984 | 8.88% | 0.9859 | 84.79% | 95.85% | 55.95% | 0.0927 |

**Reading:** a 1.66-point approval gap. It tracks a real 0.49-point difference in observed
default rate, gender is not an input, and AUC differs by only 0.004 — so the model ranks
both groups about equally well. Under the four-fifths rule the ratio is 0.981, far above
the 0.80 threshold. **No disparate impact detected on this axis.**

### Employment status

| Segment | n | Base rate | AUC | Approval rate | Recall |
|---|---|---|---|---|---|
| Self-Employed | 4,847 | 8.38% | 0.9889 | 86.07% | 95.81% |
| Salaried | 2,652 | 8.97% | 0.9881 | 85.26% | 97.06% |

Nearly identical. Notable because self-employed applicants are often penalised by
scorecards built on salaried populations; here they are not.

### Residence type — **the fairness concern**

| Segment | n | Base rate | AUC | Approval rate | Recall |
|---|---|---|---|---|---|
| Owned | 4,269 | 5.74% | 0.9923 | **89.79%** | 98.78% |
| Mortgage | 1,777 | 9.23% | 0.9869 | 84.30% | 95.73% |
| Rented | 1,441 | 16.24% | 0.9798 | **75.71%** | 94.02% |
| Unknown | 12 | 8.33% | 1.0000 | 91.67% | 100% |

**A 14-point approval gap between owners and renters**, and the model is measurably
*worse* at ranking renters (AUC 0.9798 vs 0.9923). See §6.3.

### Loan purpose

| Segment | n | Base rate | AUC | Approval rate |
|---|---|---|---|---|
| Auto | 1,481 | 3.85% | 0.9935 | 92.51% |
| Personal | 2,584 | 4.76% | 0.9887 | 91.10% |
| Education | 1,132 | 9.72% | 0.9859 | 84.10% |
| Home | 2,302 | 15.38% | 0.9843 | 76.32% |

Performance is stable across all slices (AUC 0.980–0.994); the approval-rate spread tracks
genuine differences in default rate.

---

## 6. Limitations

### 6.1 The training data is synthetic — this is the headline limitation

The model achieves 0.9885 AUC. A real consumer scorecard reaches 0.70–0.80. The gap is not
skill; it is that the target was generated as a near-deterministic function of two bureau
fields. Applicants with utilisation below 30% and no delinquency default at a rate of
**exactly zero** — 0 out of 8,150. Real borrowers never behave that cleanly.

**Consequence:** every absolute number in §4 is meaningless as a forecast of real-world
performance. The *pipeline* is production-shaped; the *metrics* are not transferable.

### 6.2 The score is bimodal

94.5% of the test book lands in Excellent or Poor, with 5.5% in Good and Average combined.
Risk-based pricing needs a spectrum, and this does not provide one. It is a data property,
not a calibration artefact — Platt scaling concentrates 97.3% at the extremes, more than
isotonic's 94.5%.

### 6.3 Housing tenure does a lot of the work

`residence_type` is the 4th and 5th most important feature and produces a 14-point approval
gap. The correlation with default is real (5.7% vs 16.2%), but housing tenure correlates
with wealth, age, migration status and, in many markets, caste, religion and ethnicity.
**Before production this requires a proxy-discrimination analysis** — regress
`residence_type` on protected attributes and measure how much of its predictive power
survives once the protected variance is removed. That analysis has not been done here.

### 6.4 `default` is undefined

Nothing in the data says whether it means 90+ days past due, charge-off, or something else.
The definition determines what the model actually predicts. Unresolvable without the data
owner.

### 6.5 The model has never seen a downturn

The default rate is flat across all 21 quarters (7.8–9.6%), including the COVID period —
which is itself implausible and further evidence the data is generated. A model trained
only on benign conditions will be over-optimistic in a recession.

### 6.6 One snapshot per borrower

No repeat applications, no behavioural history, no macroeconomic covariates.

### 6.7 Other

* **Extrapolation** — sensible bounds are validated at the API, but an applicant at the
  edge of the observed range gets a less reliable score.
* **`Unknown` residence type** has 12 test rows. The metrics for that slice are noise.
* **Isotonic step function** ties nearby scores, costing some PR-AUC (0.9009 → 0.8862).
* **Point estimates only** — no uncertainty interval on the PD.

---

## 7. Ethical considerations

### Adverse action

Credit denial carries a legal duty in most jurisdictions (ECOA/Reg B in the US, the RBI
fair practices code in India) to give **specific reasons**. SHAP supports this: for any
applicant, the top contributing features can be turned into plain-language reasons
("credit utilisation of 94% is well above the 43% typical of approved applicants").

`reports/figures/shap_force_*.png` shows worked examples for the highest-risk,
lowest-risk and borderline applicants. **"The model said so" is not a lawful reason.**

### Protected characteristics

`gender` is excluded from the feature set. It is retained in the raw data *only* to audit
fairness — you cannot measure disparate impact on an attribute you have thrown away. This
is the standard practice: exclude from the model, retain for monitoring.

`age` **is** a model input, which needs care. In the US, age is protected under ECOA and
using it directly is generally not permitted for credit decisions. In India it is
routinely used. **Before production in a US context, `age` must be removed and the model
retrained.** Its SHAP importance is 0.223 — 13th of 31 — so the accuracy cost would be
small.

### Feedback loops

Declined applicants never generate an outcome, so retraining on approved loans only
narrows the model's view of risk over time (reject inference). Mitigation: reserve a small
random sample of marginal applicants for approval, or apply a standard reject-inference
method. Not implemented here.

### Automation bias

An 0.9885 AUC invites blind trust. §6.1 explains why that would be a serious error. The
service returns a `decision` field as a *recommendation*, and the threshold is exposed in
every response so the caller can see what it was measured against.

### Transparency

Every feature, exclusion, threshold and cost assumption is documented and versioned.
`GET /model/info` exposes the version, feature list, metrics and leakage ablation at
runtime, so an auditor can query a running instance directly.

---

## 8. Maintenance

### Monitoring

| Signal | Green | Investigate | Act |
|---|---|---|---|
| PSI, any top-5 feature | < 0.10 | 0.10–0.25 | > 0.25 → retrain |
| Approval rate | ±2 pts | ±5 pts | > ±5 pts |
| Observed vs predicted default rate | ±1 pt | ±2 pts | > ±2 pts → recalibrate |
| p99 latency | < 50 ms | < 200 ms | > 200 ms |

Current PSI maxima: 0.0036 (train vs test), 0.0023 (in-time vs out-of-time) — both deeply
stable, as expected when both sides come from one generated population. On real data this
would be the primary early-warning signal.

### Retraining triggers

1. PSI > 0.25 on any top-5 feature.
2. Observed default rate in the approved book drifts more than 2 points from predicted.
3. Scheduled: every 6 months regardless.
4. Any change to the credit policy, the product mix, or the definition of `default`.
5. Macroeconomic regime change — the model has never seen one (§6.5).

### What breaks first

Calibration, before discrimination. The *ranking* of applicants is fairly robust; the
*level* of the probabilities drifts as soon as the population shifts. Recalibrating on
recent data is cheap and does not require a full retrain — which is a good reason to keep
the calibrator as a separable wrapper, as it is here.

---

## 9. Reproducibility

| | |
|---|---|
| Command | `python -m src.train` (~3 min) |
| Seed | 42, everywhere |
| Python | 3.11 |
| Key versions | scikit-learn 1.5.2 · xgboost 2.1.4 · numpy 1.26.4 · pandas 2.2.3 |
| Verified | Two consecutive full runs produced identical metrics to 4 decimal places |
| Artifacts | `models/model.joblib`, `models/metadata.json` |
| Tests | 65, all passing |

`models/metadata.json` records the training timestamp, every hyperparameter, the full
cleaning report, every metric, the fairness slices, and a fixed prediction fixture used to
detect training/serving skew.

---

## 10. Contact and provenance

Portfolio project. The model artifact, the metrics in this card and the figures in
`reports/` are all produced by `src/train.py` at the version recorded in
`models/metadata.json`. If you are reading this card away from that artifact, check the
`model_version` and `trained_at` fields match.
