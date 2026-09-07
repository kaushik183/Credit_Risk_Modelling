# Prompt for Claude Code

Paste everything below the line into Claude Code from the repo root (with the notebook and the three CSVs present).

---

You are building a production-grade credit default risk model end to end: data prep → XGBoost model → FastAPI service → deployed URL → documentation deep enough that I can defend every decision in a job interview.

## Current state of the repo

- `Credit_Risk_Model.ipynb` — only 8 cells: imports, three `read_csv` calls, `.head()` / `.shape` checks, and two merges (`customers` + `loans` on `cust_id`, then `+ bureau_data`). Nothing else exists. Treat it as a starting point, not something to preserve.
- `customers.csv` — 50,000 rows: `cust_id, age, gender, marital_status, employment_status, income, number_of_dependants, residence_type, years_at_current_address, city, state, zipcode`
- `loans.csv` — 50,000 rows: `loan_id, cust_id, loan_purpose, loan_type, sanction_amount, loan_amount, processing_fee, gst, net_disbursement, loan_tenure_months, principal_outstanding, bank_balance_at_application, disbursal_date, installment_start_dt, default`
- `bureau_data.csv` — 50,000 rows: `cust_id, number_of_open_accounts, number_of_closed_accounts, total_loan_months, delinquent_months, total_dpd, enquiry_count, credit_utilization_ratio`

Target is `loans.default` (boolean). Base default rate is **8.594%** — imbalanced, so accuracy is a useless metric here.

## Data quality issues I already found — handle these explicitly, don't rediscover them

1. `loan_purpose` contains 22 rows spelled `Personaal` (typo of `Personal`). Normalize.
2. `residence_type` has 62 nulls. Decide and justify: impute with mode, or treat as its own `Unknown` category (XGBoost handles a separate category fine).
3. 8 rows have `sanction_amount = 0`, `loan_amount = 0`, `processing_fee = 0`, and the same 8 customers have `income = 0`. These break every ratio feature. Drop them and log the count.
4. `gst` is inconsistent — it is roughly 9× `processing_fee` in 49,985 of 50,000 rows, which is not a plausible 18% GST on the fee. Flag this in the docs and drop the column rather than pretending it's meaningful.
5. 7 rows fail the identity `loan_amount − processing_fee − gst ≈ net_disbursement`. Same treatment: log, drop or repair, and explain.
6. `processing_fee / loan_amount` is 2% for almost every row but reaches 148% in a handful. Cap or drop the outliers.

## Leakage decision — this is the single most important design call, make it deliberately

Some columns are only knowable **after** the loan is disbursed and running:

- `principal_outstanding` (balance mid-repayment — strongly post-hoc)
- `processing_fee`, `gst`, `net_disbursement` (post-approval)
- `installment_start_dt`

Build the model as an **application-time scorecard**: only features a lender would have at the moment of decision. Drop the post-disbursal columns from the feature set. Then, as evidence, run one quick ablation where `principal_outstanding` is included, and report how much AUC it inflates. That number is my proof in an interview that I checked for leakage instead of chasing a suspiciously high score.

Also drop identifiers (`cust_id`, `loan_id`, `zipcode`) from features. Keep `city` / `state` only if you justify the encoding — high-cardinality categoricals in a 50k-row dataset usually hurt.

## Feature engineering

At minimum:
- `loan_to_income = loan_amount / income`
- `delinquency_ratio = delinquent_months / total_loan_months`
- `avg_dpd_per_delinquency = total_dpd / delinquent_months` (guard the divide-by-zero)
- `loan_to_sanction = loan_amount / sanction_amount`
- `balance_to_loan = bank_balance_at_application / loan_amount`
- `credit_utilization_ratio` used as is
- account mix: `open / (open + closed)`
- optionally month/quarter from `disbursal_date`

Add anything else you can defend. For every engineered feature, write one sentence in the docs on the credit-risk intuition behind it.

## Splitting

Use a **stratified 70/15/15 train/validation/test split** with a fixed seed. Then also check whether `disbursal_date` spans enough time for a time-based out-of-time (OOT) holdout; if it does, produce that as a second evaluation and report both. OOT performance is what a risk team would actually ask about.

## Modeling

1. **Baseline: logistic regression** with WOE/one-hot encoding and scaling. This is the industry-standard scorecard and my comparison point.
2. **XGBoost** (`XGBClassifier`, `objective="binary:logistic"`, `eval_metric="auc"`), handling imbalance via `scale_pos_weight` — compare against undersampling and against doing nothing, and report which won and why.
3. Hyperparameter search with Optuna (or `RandomizedSearchCV` if you prefer fewer dependencies) over `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `reg_lambda`. Use stratified CV on train+val, early stopping on the validation fold.
4. **Calibrate** the final model (isotonic or Platt on the validation set) — raw XGBoost outputs aren't probabilities, and a lender needs real PDs. Show a reliability curve and Brier score before/after.
5. Wrap preprocessing + model in a single **sklearn `Pipeline`** so the API cannot drift from training. Serialize with joblib.

## Evaluation — report real numbers, never placeholders

Actually run everything and paste the true outputs into the docs:

- ROC-AUC, PR-AUC (more honest at 8.6% positives), Gini
- **KS statistic** and the decile rank-ordering table (event rate per decile — the table credit teams always ask for)
- Confusion matrix at a threshold chosen by an explicit cost assumption (state the assumed cost of a false negative vs false positive, e.g. lost principal vs lost interest margin), not at 0.5
- Precision / recall / F1 at that threshold
- **SHAP**: global importance bar plot plus 2–3 individual force plots
- **PSI** between train and test on the top features, as a template for production drift monitoring
- Map calibrated PD to a **300–900 credit score** with a rating band (Poor / Average / Good / Excellent) and show the score distribution

Save all plots to `reports/figures/`.

## Repo structure

Refactor out of the notebook into real modules:

```
src/
  data.py          # load + merge + clean
  features.py      # feature engineering (imported by BOTH training and API)
  train.py         # trains, evaluates, saves artifacts
  evaluate.py      # metrics, plots, decile table, PSI
api/
  main.py          # FastAPI app
  schemas.py       # Pydantic v2 request/response models
models/            # model.joblib, metadata.json (version, date, features, metrics)
tests/             # pytest
reports/
notebooks/         # keep the original notebook, cleaned up, as EDA only
```

`features.py` must be the single source of truth for transformations, imported by both training and serving. Say so in the docs — it's the answer to "how do you prevent training/serving skew?"

## FastAPI service

- `POST /predict` — one applicant → `{probability_of_default, credit_score, rating, model_version}`
- `POST /predict/batch` — list of applicants
- `GET /health` — liveness, plus whether the model loaded
- `GET /model/info` — version, training date, feature list, headline metrics
- Auto docs at `/docs`
- Pydantic v2 validation with realistic bounds (age 18–100, income > 0, enums for categoricals) so bad input returns 422 with a clear message, not a 500
- Load the model **once at startup** via a lifespan handler, not per request
- Structured request/latency logging
- Handle the case where the model file is missing: fail loudly at startup

Write pytest tests: schema validation, a known-good payload, a known-bad payload, `/health`, and a test asserting the pipeline gives the same prediction as `train.py` did for a fixed row.

## Deployment

Dockerize (slim Python base, non-root user, `uvicorn` with workers). Then deploy to a free tier and give me a live URL — **Render** by default, or Fly.io / Railway / Hugging Face Spaces if you hit a blocker. Include:

- `Dockerfile` + `.dockerignore`
- `render.yaml` (or equivalent)
- A GitHub Actions workflow running lint + pytest on push
- Exact step-by-step deploy commands in the README, including anything I must click in a browser

Anything you can't do yourself (signing into Render, connecting a repo), stop and give me numbered instructions instead of guessing.

## Documentation — this is a first-class deliverable, not an afterthought

Write for someone smart who doesn't know this project. Plain language, short sentences, no jargon left undefined.

**`README.md`** — what the project is, the business problem, architecture diagram (mermaid), quickstart, API examples with real `curl` commands and real responses, live URL, results table.

**`docs/MODEL_CARD.md`** — intended use, training data, features, performance overall and sliced by segment (gender, employment type, loan purpose — fairness matters in lending), limitations, ethical considerations.

**`docs/INTERVIEW_GUIDE.md`** — the one I care most about. Cover:
- The problem in one paragraph a non-technical person understands
- The data and how the three tables relate
- Every cleaning decision and *why*
- Every engineered feature and its credit intuition
- Why XGBoost, and honestly when logistic regression would be the better choice (regulation, explainability, adverse-action notices)
- How gradient boosting actually works, in plain words
- What each tuned hyperparameter does and which direction it pushes bias/variance
- Class imbalance: what I tried, what I picked, why
- Every metric defined plainly, with my actual number and what "good" looks like in credit risk
- Why calibration matters when you price loans off the probability
- The leakage story and the ablation number
- Threshold selection and the cost assumption
- Serving architecture, why the pipeline is serialized whole, latency numbers
- Monitoring: PSI, retraining triggers, what breaks first in production
- What I'd do with more time / where the model is weak
- A **Q&A section: ~20 questions an interviewer would ask, with answers in my voice** — including the hostile ones ("your AUC is high, are you sure you don't have leakage?", "why not a neural net?", "how would you explain a rejection to a customer?")

## How to work

Go phase by phase, and pause for my review after each: (1) cleaning + features, (2) modeling + evaluation, (3) API + tests, (4) deployment, (5) docs. Show me real metrics as you go. Pin dependency versions in `requirements.txt` and verify a clean `pip install` + full run works from scratch before you call it done. Fixed random seed everywhere.

If a number looks too good, investigate it before writing it down.
