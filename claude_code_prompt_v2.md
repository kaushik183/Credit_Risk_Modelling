# Prompt for Claude Code (Opus 5) — v2

Paste everything below the line into Claude Code from the repo root.

If phases 1–4 already exist in the repo, say so and jump straight to **Phase 5 (documentation)** — that is the part this version of the prompt is really about.

---

You are building a production-grade credit default risk model end to end: data prep → XGBoost model → FastAPI service → deployed URL → documentation that teaches me the whole thing from scratch so I can defend it in an interview.

Read this whole brief before starting. Then work in the five phases at the bottom, pausing after each.

---

# PART A — WHO YOU ARE WRITING FOR

This matters more than any other instruction in this document. **The documentation is the primary deliverable, not an afterthought.** The code exists so the docs have something true to describe.

**My background:**

- I know machine learning fundamentals. Train/test splits, overfitting, cross-validation, precision and recall, gradient boosting at a conceptual level — assume I have seen these before, but define them anyway in one clean line when they first appear.
- **I have never used SHAP.** I do not know what a Shapley value is, what a force plot shows, or how to read one.
- **I have never used FastAPI.** I do not know what an endpoint is in practice, what Pydantic does, what ASGI or uvicorn are, what a request/response cycle looks like, or why anyone would wrap a model in a web server instead of just calling `model.predict()`.
- **I have never deployed anything.** I do not know what Docker actually does, what an image versus a container is, what Render/Fly/Railway do for me, what a port is in this context, or what happens between `git push` and a live URL.
- **I have never used GitHub Actions or any CI.** I do not know what a workflow, a runner, a job, or a trigger is.
- I have not seen credit-risk domain concepts: KS statistic, PSI, LGD, EMI, adverse-action notices, reject inference, scorecards.

**So: for anything on that list, explain from zero.** Not a reminder — an actual explanation, as if to a smart person who has genuinely never encountered it. Assume nothing about tooling. Assume competence about reasoning.

**Write in plain English.** Short sentences. No sentence that requires a term you have not already defined. Every acronym expanded on first use, every time, in every file (each doc must stand alone). No phrase like "as you know" or "simply" or "just" — if it were simple I would not be asking.

**Every concept gets the same four-part treatment:**

1. **What it is** — one or two plain sentences, no jargon.
2. **Why it exists** — what breaks without it. Make the pain concrete.
3. **How it works** — the mechanism, in words first, then the minimum code.
4. **Where it lives in this project** — the actual file, the actual line, the actual number this repo produced.

Part 4 is non-negotiable. A generic explanation of SHAP is worthless to me. An explanation of SHAP that ends with "and here is the SHAP plot for applicant #4,412 in our test set, and here is how to read it" is what I need.

**Use one worked example throughout.** Pick a single real applicant from the test set early and carry them through the entire documentation — their raw fields, their engineered features, their model score, their SHAP explanation, their API request and response, their decision under the threshold. One person, followed from CSV row to live HTTP response. That thread is what makes a system click.

---

# PART B — THE PROJECT

## B1. Current state

- `Credit_Risk_Model.ipynb` — a stub: imports, three `read_csv` calls, `.head()`/`.shape` checks, two merges. Nothing else. Keep it as EDA only; the real code goes in `src/`.
- `customers.csv` — 50,000 rows: `cust_id, age, gender, marital_status, employment_status, income, number_of_dependants, residence_type, years_at_current_address, city, state, zipcode`
- `loans.csv` — 50,000 rows: `loan_id, cust_id, loan_purpose, loan_type, sanction_amount, loan_amount, processing_fee, gst, net_disbursement, loan_tenure_months, principal_outstanding, bank_balance_at_application, disbursal_date, installment_start_dt, default`
- `bureau_data.csv` — 50,000 rows: `cust_id, number_of_open_accounts, number_of_closed_accounts, total_loan_months, delinquent_months, total_dpd, enquiry_count, credit_utilization_ratio`

Target: `loans.default` (boolean), **8.594% positive**. Join is 1:1 on `cust_id` — assert it with `validate="one_to_one"` so a bad merge cannot silently duplicate rows.

## B2. Data quality — verified facts, not guesses

Do not re-litigate these, but do re-verify each one in code and assert the counts in `tests/test_data.py` so the docs can never drift from reality.

1. `loan_purpose` has 22 rows spelled `Personaal`. Both spellings default at the same rate → it is a typo, not a product. Normalise.
2. `residence_type` has 62 nulls. **Do not impute the mode.** The mode is `Owned`, the *lowest-risk* category (5.6% default vs 16.3% for `Rented`), so mode-imputation would silently label these applicants as the safest kind of borrower on no evidence. Give them an explicit `Unknown` category.
3. Eight rows have `sanction_amount`, `loan_amount`, `processing_fee`, `gst`, `net_disbursement`, `income` and `bank_balance_at_application` all zero. They are placeholders, not loans. Drop and log.
4. `processing_fee` = exactly 2% of `loan_amount`; `gst` = exactly **18% of `loan_amount`** (not of the fee); `net_disbursement` = exactly 80% of `loan_amount`. All three are exact linear functions of `loan_amount` and carry zero independent information. Drop them — and note in the docs that the reason is redundancy, not inconsistency.
5. Seven rows have a corrupted `processing_fee` (up to 148% of the loan). These are the *same* seven rows that fail `loan_amount − processing_fee − gst = net_disbursement`. The true fee is recoverable exactly as `gst / 9`, confirmed by the identity going from 7 violations to 0. Repair rather than drop; nothing is guessed.

Result: 49,992 of 50,000 rows survive.

## B3. Leakage — the most important design call

These columns only exist *after* the loan is disbursed: `principal_outstanding`, `processing_fee`, `gst`, `net_disbursement`, `installment_start_dt`.

Build an **application-time scorecard**: only what a lender knows at the moment of decision. Exclude all of them. Drop identifiers (`cust_id`, `loan_id`, `zipcode`) too.

Then run the ablation: refit the identical model with `principal_outstanding` added back and report the AUC delta. **Report the result honestly even if it is a null result** — "I expected inflation and found none" is a stronger interview answer than a clean story, because it shows the check was real.

Enforce it in code: a test that fails if any post-disbursal column can reach the feature matrix.

## B4. Features

Affordability: `loan_to_income`, `emi_to_monthly_income`, `loan_to_sanction`, `balance_to_loan`, `months_of_income_in_bank`, `income_per_dependant`.
Behaviour: `delinquency_ratio`, `avg_dpd_per_delinquency`, `credit_utilization_ratio`.
File shape: `open_account_ratio`, `total_accounts`, `enquiry_count`.
Pass-through: `loan_tenure_months`, `total_loan_months`, `age`, `years_at_current_address`, `number_of_dependants`. One-hot: `marital_status`, `employment_status`, `residence_type`, `loan_purpose`, `loan_type`.

Every ratio goes through a `_safe_divide` guard — ~40% of borrowers have zero delinquent months, and an unguarded divide produces `inf`, which XGBoost will happily split on and which will return nonsense from the API. Test that no infinities survive.

Each feature needs **one sentence of credit intuition** in the docs. Not "ratio of loan to income" — "the oldest underwriting ratio there is; someone asking for 5× their annual income is stretched in a way someone asking for 0.5× is not."

## B5. Splitting, modelling, evaluation

- Stratified 70/15/15 train/val/test, fixed seed. Plus an out-of-time split on `disbursal_date` if the range allows — OOT performance is what a risk team actually asks about.
- **Baseline: logistic regression.** This is the industry-standard scorecard and the honest comparison point.
- **XGBoost**, imbalance handled via `scale_pos_weight`; measure it against do-nothing and undersampling rather than assuming. Do not use SMOTE — inventing borrowers is hard to defend to a credit committee — and say so in the docs.
- Tune with Optuna, stratified CV, early stopping inside each fold.
- **Calibrate** (isotonic vs Platt). Select the calibrator on a held-out half of validation, never on test. Clamp reported probabilities away from exactly 0 and 1.
- Wrap preprocessing + model + calibrator in one sklearn `Pipeline`, serialized as a single artifact.

Report, with real numbers, never placeholders: ROC-AUC, PR-AUC, Gini, KS, Brier, ECE, precision/recall/F1 at a cost-chosen threshold, the decile rank-ordering table, SHAP global + individual, PSI train-vs-test, and a PD → 300–900 credit score mapping with rating bands. Save plots to `reports/figures/`.

**Threshold by explicit cost, not 0.5.** State the assumption: false negative ≈ 60% of exposure (loss given default), false positive ≈ 6% (foregone interest margin) → roughly 10:1. Put both constants in `config.py` so they are a one-line change, and say in the docs that they are assumptions, not measurements.

**If the metrics come out implausibly high, investigate before writing them down.** Cross-tabulate the target against the top features and check whether the data is synthetic and near-deterministic. If it is, say so loudly in the docs, state what a realistic production number would be, and explain how you ruled out leakage. Do not quietly present an unbelievable AUC as a success.

## B6. Repo structure

```
src/
  data.py          # load + merge + clean
  features.py      # feature engineering — imported by BOTH training and serving
  train.py         # trains, evaluates, saves artifacts
  evaluate.py      # metrics, plots, decile table, PSI
  config.py        # seeds, paths, cost constants, thresholds
api/
  main.py          # FastAPI app
  schemas.py       # Pydantic v2 request/response models
models/            # model.joblib, metadata.json
tests/
reports/figures/, reports/tables/
docs/
notebooks/         # original notebook, cleaned, EDA only
```

`features.py` must be the single source of truth, imported by training and serialized inside the model artifact.

## B7. The API

Endpoints: `POST /predict`, `POST /predict/batch`, `GET /health`, `GET /model/info`, auto docs at `/docs`.

- Pydantic v2 validation with realistic bounds (age 18–100, income > 0, utilisation 0–100, enums for categoricals) plus cross-field rules (`loan_amount ≤ sanction_amount`, `delinquent_months ≤ total_loan_months`). Bad input → 422 with a readable message, never a 500, never a confident score computed from nonsense.
- Model loaded **once at startup** via a lifespan handler. Fail loudly if the artifact is missing — a container that can never serve a valid request must not report itself healthy.
- Structured JSON request logging with a request id and latency; `X-Request-ID` and `X-Response-Time-ms` headers.
- Response includes the probability, the credit score, the rating band, the threshold it was compared against, and the model version. Frame it as a **recommendation**, not an automated lending decision.
- Measure and report real p50/p95/p99 latency and batch throughput.

Tests: schema validation, known-good payload, known-bad payload, `/health`, and the important one — `train.py` saves a fixed row plus the probability it produced, and a test asserts the API returns the identical number after a joblib round-trip in a fresh process. That test is the structural guarantee against training/serving skew.

## B8. Deployment

Docker (slim base, non-root user, uvicorn), then a live URL on **Render** by default (Fly.io / Railway / Hugging Face Spaces as fallbacks). Include `Dockerfile`, `.dockerignore`, `render.yaml`, and a GitHub Actions workflow running lint + tests on push.

Anything you cannot do yourself — signing into Render, connecting the repo — stop and give me numbered browser instructions rather than guessing.

---

# PART C — THE DOCUMENTATION (the main deliverable)

Seven files. Each stands alone: re-define terms in each file rather than cross-referencing a glossary I would have to hold in my head.

## C1. `README.md`

Landing page. What the project is, the business problem in plain language, an architecture diagram (mermaid), the live URL, a results table, quickstart commands verified to work from a clean clone, and real `curl` examples with real responses pasted in.

## C2. `docs/CONCEPTS.md` — the from-zero primer

Everything I need before any other doc makes sense. Each entry gets the four-part treatment from Part A.

**Credit domain:** default, delinquency, days past due (DPD), credit utilisation, EMI, loan-to-income, secured vs unsecured, loss given default (LGD), exposure, expected loss (PD × LGD × exposure), a scorecard, adverse-action notices, the 300–900 score scale and how our probabilities map onto it.

**Modelling terms as used here:** class imbalance and why 91.4% accuracy is worthless at an 8.6% base rate; `scale_pos_weight` and what re-weighting gradients actually does; each metric with a plain definition, our number, and what "good" looks like in real credit risk (**including that a real scorecard lands at 0.70–0.80 AUC and KS 30–40, so ours being far higher is a red flag we investigated, not a triumph**); KS statistic explained by walking through how the two cumulative curves are built; PSI explained by walking through the bin-by-bin calculation on one real feature; the decile table and why credit committees ask for it; calibration versus discrimination as two separate properties, with a concrete example of a model that ranks perfectly and prices catastrophically.

**Data leakage:** what it is, why `principal_outstanding` is the suspect, why the fix is a rule and not a hunch.

## C3. `docs/SHAP_EXPLAINED.md` — assume zero prior knowledge

Do not start with game theory notation.

1. Start with the problem: XGBoost is hundreds of trees; you cannot read it. But a declined applicant is legally owed a reason. That gap is why SHAP exists.
2. The core idea by analogy: a team produces a result together, and you want each member's fair share of the credit. Shapley values answer that by asking what changes when each member is added, averaged over every possible order of adding them. Then map that analogy onto features and a prediction, term by term.
3. The one equation, only after the intuition, with every symbol named in words.
4. Additivity: base value + all SHAP values = this prediction exactly. Show it summing on our worked applicant, with the actual numbers.
5. **How to read each plot we generate** — bar plot, beeswarm, waterfall/force — axis by axis. What is a dot, what is the x-axis, what does colour mean, what does left versus right mean. Include the actual figure and narrate it.
6. TreeSHAP versus KernelSHAP and why the tree version is fast enough to run per-request.
7. The honest limits: SHAP shows the model's reasoning, not the world's causality; correlated features split credit between themselves; and a compliance team will ask you to defend SHAP itself, which is one of the arguments for a linear scorecard.
8. Turn it into an adverse-action sentence: from the applicant's SHAP values to "your credit utilisation is 94%, versus an average of 37% among approved applicants."

## C4. `docs/API_AND_DEPLOYMENT.md` — assume zero prior knowledge

The longest doc. Build it up in layers, each answering "why not just stop here?"

**Layer 0 — why a web service at all.** I have `model.joblib`. Explain what "serving" means: other systems, in other languages, on other machines, need this prediction, and they cannot import my Python. Explain what an API is, what HTTP request/response is, what a JSON body is, what a status code is, what POST versus GET means. Concretely, with our own payload.

**Layer 1 — FastAPI.** What a framework does for you. What `@app.post("/predict")` actually means, decorator and all. What Pydantic is and why a typed schema at the boundary is different from validating inside the function. What automatic `/docs` is and what OpenAPI means. What ASGI and uvicorn are and why they sit under FastAPI. What a lifespan handler is and why loading the model there instead of per request matters — with our measured latency difference. What synchronous versus asynchronous means here, and honestly whether it matters for a CPU-bound `predict_proba`.

Then **a line-by-line walkthrough of our actual `api/main.py` and `api/schemas.py`.** Every import, every decorator, every parameter. Not a summary — a walkthrough.

**Layer 2 — Docker.** The problem it solves ("works on my machine"). Image versus container versus registry, in plain words with an analogy. What each line of *our* Dockerfile does and why it is in that order (layer caching, and why `requirements.txt` is copied before the source). Why a non-root user. What `EXPOSE` and port mapping mean. The exact commands to build, run, and inspect locally, and what each flag does.

**Layer 3 — hosting.** What a platform like Render actually gives you: a machine, a public URL, TLS, restarts, logs. What a health check is and what the orchestrator does with it. Environment variables and why secrets never live in the repo. Cold starts on free tiers and what that means for a demo in front of an interviewer. Then the exact deploy steps, numbered, including every browser click.

**Layer 4 — GitHub Actions.** What continuous integration is and what it prevents. Workflow, job, step, runner, trigger — each defined. A line-by-line walkthrough of *our* YAML file. What "the build is red" means in practice and what I should do when it happens.

**Layer 5 — the request's full journey.** Follow our worked applicant end to end: `curl` command → HTTP over the network → uvicorn → FastAPI routing → Pydantic validation → the loaded pipeline → feature engineering → XGBoost → calibrator → score mapping → JSON response → the log line it produced. One page, one narrative, no gaps.

## C5. `docs/MODEL_CARD.md`

Intended use, training data, features, performance overall and sliced by segment (gender, employment type, loan purpose, residence type — fairness matters in lending), limitations, ethical considerations. Report disparities you find rather than only the ones that look good, and explain the four-fifths rule in one plain line.

## C6. `docs/INTERVIEW_GUIDE.md`

The defence document. Every decision and its reasoning, in the order an interviewer would probe: problem → data → cleaning → features → model choice → boosting explained plainly → hyperparameters and which way each pushes bias/variance → imbalance → metrics → calibration → leakage → threshold economics → serving → monitoring → weaknesses.

Then **~20 questions with answers in my voice**, including the hostile ones: "are you sure you don't have leakage?", "why not a neural net?", "how would you explain a rejection to a customer?", "your precision is only 58%, isn't that bad?", "you dropped gender but men are still approved more — explain", "how do you prevent training/serving skew?", "what breaks first in production?", "is this an automated lending decision?".

Where the honest answer is a weakness, write the honest answer. A candidate who volunteers "this AUC is not credible and here is why" is more convincing than one who defends it.

## C7. `docs/RESUME.md` — new, and I want this done carefully

Ready-to-paste material, no editing needed by me:

1. **Project title** — three options, from conservative to punchy.
2. **One-line description** for a resume project header (under 20 words).
3. **Three-line version** for a portfolio or LinkedIn project entry.
4. **4–6 resume bullets**, each in "action verb → what → how → measurable result" form, quantified with the project's real numbers, each under two lines. Cover: the pipeline and data cleaning, the model and tuning, calibration and threshold economics, the API and deployment, and testing/CI. Write them so they survive a keyword screen without becoming buzzword soup.
5. **A skills line** — the concrete tools and techniques this project actually demonstrates. Nothing that is not genuinely in the repo.
6. **A 60-second spoken pitch** for "walk me through a project" — written to be said out loud, not read.
7. **A 15-second version** for the elevator case.
8. **An honesty note**: which claims are supported by the repo and which would be overclaiming (e.g. do not call it "production" if it is on a free tier with synthetic data). Tell me exactly where the line is so I never get caught past it.

---

# PART D — HOW TO WORK

Five phases. **Stop after each and show me what you have before continuing.**

1. Cleaning + features (+ `tests/test_data.py`, `tests/test_features.py`)
2. Modelling + evaluation, with real numbers printed
3. API + tests
4. Docker + deployment + CI
5. Documentation — expect this to take as long as phases 1–4 combined, and do not compress it

**Rules throughout:**

- Actually run everything. Every number in every doc is a real output, never a placeholder, never "approximately". If you have not run it, do not write it.
- Pin versions in `requirements.txt` and verify a clean install and a full run from scratch before declaring done.
- Fixed random seed everywhere.
- If a number looks too good, investigate before writing it down.
- Prefer the honest finding over the tidy narrative. Negative results get reported.
- When a decision has a real trade-off, document both sides and say which you picked and why. "There is no downside" is almost always a sign you have not looked.

**Final self-check before you tell me phase 5 is done.** Re-read `CONCEPTS.md`, `SHAP_EXPLAINED.md` and `API_AND_DEPLOYMENT.md` as someone who has never used SHAP, never written an API, and never deployed anything. Flag every sentence that assumes knowledge I said I do not have, and fix it. Then list for me, explicitly, the three concepts you think are still hardest to grasp from the docs alone — I will ask you to go deeper on those.
