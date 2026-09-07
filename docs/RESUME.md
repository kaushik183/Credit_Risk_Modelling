# Resume, Portfolio and Pitch Material

Ready to paste. Every number is real and traceable to `models/metadata.json` or
`reports/tables/`.

**Read [§8, the honesty note](#8-the-honesty-note--where-the-line-is), before using any of
this.** It tells you exactly which claims the repo supports and which would be
overclaiming. Getting caught past that line costs more than every bullet below gains.

---

## 1. Project title — three options

| Register | Title |
|---|---|
| **Conservative** | Credit Default Risk Model with REST API Deployment |
| **Recommended** | Application-Time Credit Risk Scorecard with Calibrated Probability Serving |
| **Punchy** | Production-Shaped Credit Scorecard: Leakage-Audited, Calibrated, Deployed |

**Use the middle one.** "Application-time" and "calibrated" are the two words that signal
you understand credit modelling rather than just having fitted a classifier — and both are
genuinely true of this project. The third is fine for a personal site but "production-shaped"
invites the question of whether it *is* production, and it is not.

---

## 2. One-line description (under 20 words)

> End-to-end credit default risk scorecard: XGBoost with calibrated probabilities, served
> via FastAPI, containerised and deployed. *(17 words)*

Alternative, if the audience is more business than engineering:

> Credit risk model that scores loan applications in real time and recommends approve or
> decline on cost-optimised thresholds. *(19 words)*

---

## 3. Three-line version (portfolio / LinkedIn project entry)

> Built an end-to-end credit default risk model on 50,000 loan applications, predicting
> default probability using only features available at the moment of application — a
> deliberate design decision verified with a leakage ablation.
>
> Tuned XGBoost with Optuna and calibrated its output with isotonic regression, cutting
> calibration error 7.6× so the probabilities are usable for risk-based pricing rather than
> ranking alone. Set the approve/decline threshold from an explicit 10:1 cost assumption
> instead of the default 0.5, catching 96.3% of defaults.
>
> Served it through a FastAPI application with Pydantic validation, packaged in Docker, and
> deployed to a live URL with GitHub Actions CI and 65 tests — including one asserting the
> API reproduces the exact probability recorded during training, which makes
> training/serving skew structurally detectable.

---

## 4. Resume bullets

Four to six, each under two lines. Pick the ones matching the role.

**Pipeline and data quality**
> Built an end-to-end credit risk pipeline over three joined tables (50,000 loans),
> resolving six data-quality defects — including seven corrupted fee records reconstructed
> exactly from an accounting identity — and enforced all cleaning counts with automated
> tests so documentation cannot drift from code.

**Leakage and feature design**
> Designed an application-time feature set of 22 engineered features, excluding five
> post-disbursal columns that would be unavailable at decision time; quantified the
> decision with an ablation showing the excluded data changed AUC by −0.0001, and enforced
> the exclusion with a regression test.

**Modelling and tuning**
> Tuned XGBoost with Optuna over 40 trials using 4-fold stratified cross-validation and
> early stopping, benchmarked against a logistic-regression scorecard and three
> class-imbalance strategies at an 8.6% base rate; reached 0.9885 ROC-AUC and 0.8862 PR-AUC
> on a held-out test set.

**Calibration and threshold economics**
> Calibrated model output with isotonic regression selected on held-out validation data
> (never on test), reducing expected calibration error from 0.048 to 0.006; set the
> decision threshold from an explicit false-negative/false-positive cost ratio, raising
> default capture to 96.3% and cutting the bad rate in the approved book from 8.6% to 0.37%.

**API and deployment**
> Deployed the model as a FastAPI service with Pydantic v2 validation and cross-field
> business rules, containerised with a non-root multi-stage Docker build, and shipped to a
> live public URL; measured 3.6 ms p50 in-process latency and 12.6 ms for a 1,000-applicant
> batch.

**Testing and CI**
> Wrote 65 tests covering data cleaning, feature guards, metric correctness and API
> contract, and configured GitHub Actions to lint, train and rebuild the container image on
> every push — including a parity test asserting the served model reproduces the exact
> probability recorded at training time.

**Optional, if the role values judgement over metrics** — this one is unusual and lands well
with senior interviewers:
> Investigated an implausibly high 0.9885 AUC rather than reporting it, established through
> a leakage ablation and an out-of-time holdout that the dataset was synthetic and
> near-deterministic, and documented the realistic 0.70–0.80 production range alongside the
> measured result.

---

## 5. Skills line

Only what is genuinely exercised in the repo:

> **Python, pandas, NumPy, scikit-learn, XGBoost, Optuna, SHAP · FastAPI, Pydantic v2,
> uvicorn · Docker, GitHub Actions, Render · pytest, ruff · gradient boosting, probability
> calibration, class imbalance, feature engineering, data-leakage analysis, model
> evaluation (ROC-AUC, PR-AUC, KS, PSI, Brier), cost-based threshold selection, model
> fairness auditing, credit risk scorecards**

**Do not add to this list** unless you add it to the repo. No MLflow, no Kubernetes, no
Spark, no AWS, no feature store, no A/B testing — none of those are here.

---

## 6. The 60-second spoken pitch

Written to be *said*, not read. Short sentences, one idea each. Practise it out loud —
reading it silently will not surface the sentences you stumble on.

> *"I built a credit default risk model — it looks at a loan application and predicts
> whether the borrower will default.*
>
> *The dataset was 50,000 loans with about an 8.6% default rate. The first real decision
> was which columns I was allowed to use. Several of them — like the outstanding principal
> — only exist after the loan has been paid out, so a lender wouldn't have them when
> deciding. I excluded those and built it as an application-time scorecard. Then I tested
> that decision: I refit the model with the most suspicious column added back, and it
> changed AUC by about a ten-thousandth. So it was excluded on principle and it cost
> nothing.*
>
> *For the model, I started with logistic regression as a baseline, because that's the
> industry-standard scorecard, then tuned XGBoost with Optuna. The important part came
> after: XGBoost is trained to rank, not to output real probabilities, so I calibrated it
> with isotonic regression. That cut calibration error by about seven and a half times.
> It matters because if you price a loan off the number, eight percent has to actually
> mean eight defaults in a hundred.*
>
> *I also didn't use 0.5 as the cut-off. A missed default costs roughly ten times what
> turning away a good customer does, so I set the threshold on expected cost. That gets to
> about 96% of defaults caught, and drops the bad rate in the approved book from 8.6% to
> under half a percent.*
>
> *Then I served it — FastAPI, Docker, live URL, 65 tests, CI on every push.*
>
> *The thing I'd actually want to tell you, though, is that the AUC came out at 0.99, which
> isn't believable for a credit model. Real ones are 0.70 to 0.80. So I investigated before
> writing it down. It's not leakage — I checked three ways. The data is synthetic, and the
> target was generated almost deterministically from two bureau features. There's a group
> of 8,000 borrowers with a default rate of exactly zero, and real people don't behave like
> that. So the method transfers. The numbers don't. I documented it that way."*

**Why it ends there.** Volunteering the weakness is the strongest move available to you.
It pre-empts the question everyone will ask, and it demonstrates the thing that is actually
hard to fake: knowing what a believable number looks like.

---

## 7. The 15-second version

> *"A credit default risk model — takes a loan application, returns a calibrated
> probability of default and an approve/decline recommendation. XGBoost behind a FastAPI
> service, containerised and deployed live. The interesting part was the leakage decision:
> I restricted it to data a lender actually has at application time, and measured what that
> cost — which turned out to be nothing."*

---

## 8. The honesty note — where the line is

The whole project's credibility rests on the metrics being reported honestly. Do not
undermine that in the resume.

### Fully supported — say these freely

| Claim | Evidence |
|---|---|
| End-to-end pipeline, 50,000 rows → live URL | The repo |
| 0.9885 ROC-AUC, 0.8862 PR-AUC, KS 90.3, Brier 0.024 on held-out test | `metadata.json` |
| Beat a logistic baseline (0.9834 AUC) | Both trained in `train.py` |
| Leakage ablation: −0.0001 AUC | `leakage_ablation` in `metadata.json` |
| Calibration cut ECE 0.048 → 0.006 | `calibration_comparison` |
| 96.3% default capture; approved-book bad rate 8.6% → 0.37% | `test_metrics_at_threshold` |
| Out-of-time holdout: 0.9855 AUC | `out_of_time_test` |
| 65 tests, CI green, Docker verified, live URL | Actions tab; the deployment |
| 3.6 ms p50 in-process; 388 ms live | Measured, both recorded |
| Optuna, 40 trials, 4-fold stratified CV | `metadata.json` |

### Overclaiming — do not say these

| Do not say | Why | Say instead |
|---|---|---|
| **"In production"** / "production-grade" | Free tier, synthetic data, no auth, no monitoring, one user | "Deployed to a live endpoint" / "production-shaped" |
| **"0.99 AUC"** without context | Sounds either dishonest or naive | Always pair it with the synthetic-data caveat |
| "Reduced defaults by X%" / any business impact | No business ever used it | "On a held-out test set, captured 96.3% of defaults" |
| "Processed 50,000 records" as if a data-engineering feat | It is three CSVs read by pandas | Emphasise the cleaning decisions, not the volume |
| "Built a scalable microservice architecture" | One container, one service, two workers | "Containerised service with health checks and CI" |
| "Real-time ML at scale" | ~2 requests/second on a free tier | "Sub-4 ms in-process inference" |
| "MLOps pipeline" | No experiment tracking, registry, or automated retraining | "Automated training pipeline with CI" |
| Expertise in FastAPI / Docker / SHAP | You used each once, here | "Used FastAPI and Docker to build and deploy this" |

### The three sentences that keep you safe

If asked anything close to "is this real / is this production", say:

1. *"It's a portfolio project on a synthetic dataset — I'd never present these numbers as a
   forecast of real performance."*
2. *"The AUC is 0.99, which isn't credible for credit risk. I investigated it rather than
   reporting it, and documented why."*
3. *"It's deployed and it works, but it's a free tier with no authentication — it's a
   demonstration of the architecture, not a service anyone should send real applications
   to."*

**None of those weaken you.** Every one signals judgement, which is the scarcer skill.

### What you can honestly claim to have learned

Because this is what the interview is really probing:

- What data leakage is, how to test for it, and why you exclude a feature even when it
  costs nothing
- Why calibration and discrimination are different properties, and when each matters
- How to turn a business cost assumption into a decision threshold
- Why accuracy is meaningless on imbalanced data, and which metrics replace it
- How a model becomes a service, and what training/serving skew is
- What it takes to make a result reproducible — pinned versions, fixed seeds, tests on the
  numbers themselves
- **How to recognise a result that is too good, and investigate rather than celebrate it**

That last one is the rarest and it is genuinely demonstrated here. Lead with it.
