# Interview Guide — Credit Default Risk Model

Everything needed to defend this project, in plain language. Every number here is real and
comes from `models/metadata.json` or `reports/tables/`.

**Contents**

1. [The problem in one paragraph](#1-the-problem-in-one-paragraph)
2. [The data](#2-the-data)
3. [Cleaning decisions](#3-cleaning-decisions)
4. [Features and their credit intuition](#4-features-and-their-credit-intuition)
5. [Why XGBoost — and when logistic regression is better](#5-why-xgboost--and-when-logistic-regression-is-better)
6. [How gradient boosting actually works](#6-how-gradient-boosting-actually-works)
7. [The hyperparameters](#7-the-hyperparameters)
8. [Class imbalance](#8-class-imbalance)
9. [Every metric, defined](#9-every-metric-defined)
10. [Calibration](#10-calibration)
11. [The leakage story](#11-the-leakage-story)
12. [Why is the AUC so high?](#why-is-the-auc-so-high)
13. [Threshold selection](#13-threshold-selection)
14. [Serving architecture](#14-serving-architecture)
15. [Monitoring](#15-monitoring)
16. [Weaknesses and next steps](#16-weaknesses-and-next-steps)
17. [**Q&A — 20 interview questions**](#17-qa--20-interview-questions)

---

## 1. The problem in one paragraph

**For a non-technical listener:** When someone applies for a loan, the bank has to guess
whether they will pay it back. Guess wrong in one direction and you lose the money you lent.
Guess wrong in the other and you turn away a good customer and lose the interest you would
have earned. Losing the loan hurts about ten times more than losing the customer. This
project looks at 50,000 past loans, learns which kinds of applicants ended up not paying,
and turns that into a score — like a credit score — that a lender can use on the next
applicant who walks in. It also draws the line for where to say no, chosen so the bank's
total expected losses are as small as possible.

**One extra sentence for a technical listener:** it is a binary classifier on an imbalanced
target (8.594% positives) built only from features available at application time, with
calibrated probability output so the number can drive pricing rather than just ranking.

---

## 2. The data

Three tables, 50,000 rows each, joined one-to-one on `cust_id`:

| Table | What it is | Key columns |
|---|---|---|
| `customers.csv` | Who is applying | age, income, employment, residence, dependants, city |
| `loans.csv` | What they are applying for, **and the target** | purpose, type, amounts, tenure, dates, `default` |
| `bureau_data.csv` | Their credit-bureau file | open/closed accounts, delinquent months, DPD, enquiries, utilisation |

**How they relate:** one customer → one loan → one bureau record. The join is 1:1, verified
with `validate="one_to_one"`, so the merge cannot silently fan out. If you have ever
watched an inner join quietly double a dataset and inflate every metric, you know why that
assertion is there.

**Target:** `loans.default`, boolean. **8.594%** positive — one loan in twelve.

**Why the imbalance matters immediately:** a model that predicts "nobody defaults" is 91.4%
accurate and completely worthless. Accuracy is off the table from the first line of code.

---

## 3. Cleaning decisions

Six issues. Each was verified in the data before being acted on, and every count is
asserted in `tests/test_data.py` so the documentation cannot drift.

### 3.1 `loan_purpose` = `Personaal` (22 rows)

Normalised to `Personal`. **Why it is a typo and not a product:** both spellings default at
4.5%. A genuinely different product would behave differently.

### 3.2 `residence_type` null (62 rows) → explicit `Unknown` category

**The decision that matters here is what I did *not* do.** The obvious move is to impute
the mode. The mode is `Owned` — which is the **lowest-risk** category at 5.6% default
versus 16.3% for `Rented`. Mode imputation would therefore quietly tell the model that
these 62 applicants are the safest kind of borrower, on no evidence at all.

Instead they get their own `Unknown` level. One-hot encoding gives it a column, and the
tree can split on it if it is useful or ignore it if it is not. In an application flow,
"the applicant did not answer this question" is itself information.

At 62 rows (0.12%) this cannot move any metric. I chose it for the reasoning, not the score
— which is exactly the honest answer to give if asked.

### 3.3 Eight all-zero rows → dropped

Eight rows have `sanction_amount`, `loan_amount`, `processing_fee`, `gst`,
`net_disbursement`, `income` **and** `bank_balance_at_application` all at zero. These are
not loans, they are placeholders. Every ratio feature would be 0/0. Dropped and logged.

### 3.4 `gst` — the prompt's description was incomplete

The brief said `gst` is "roughly 9× `processing_fee`, which is not a plausible 18% GST on
the fee". True, but the real structure is more damning:

| Column | Exact rule | Holds for |
|---|---|---|
| `processing_fee` | 2% of `loan_amount` | 49,985 / 49,992 |
| `gst` | **18% of `loan_amount`** | 49,992 / 49,992 |
| `net_disbursement` | **80% of `loan_amount`** | 49,992 / 49,992 |

`gst` is 18% of the *loan*, not of the fee. The 9× ratio is just 18/2. All three columns
are exact linear functions of `loan_amount` and carry **zero** independent information.

**Interview value:** "the brief told me to flag it as inconsistent, but when I checked the
arithmetic the columns turned out to be perfectly consistent — just perfectly redundant."
Checking beats accepting.

### 3.5 and 3.6 — two problems that are one problem

The brief listed "7 rows fail the disbursement identity" and "`processing_fee` reaches 148%
of the loan" as separate issues. They are the **same 7 rows**.

Only `processing_fee` is corrupted. `gst` and `net_disbursement` are still correct on those
rows, which means the true fee is recoverable **exactly** as `gst / 9`, and the repair is
confirmed independently by the identity `loan_amount − processing_fee − gst =
net_disbursement`, which goes from 7 violations to **0**.

**Why repair rather than drop:** nothing is guessed — two independent identities force the
value — and 7 otherwise-valid applications are kept. Dropping recoverable data is a real
cost when the positive class only has 4,297 members.

**Result: 49,992 of 50,000 rows survive.**

---

## 4. Features and their credit intuition

22 features from 20 raw inputs. One sentence of credit reasoning for each:

### Affordability

| Feature | Intuition |
|---|---|
| `loan_to_income` | The oldest underwriting ratio there is. Someone asking for 5× their annual income is stretched in a way someone asking for 0.5× is not |
| `emi_to_monthly_income` | A crude fixed-obligation-to-income ratio. **This is the one that matters most conceptually** — borrowers do not default because the loan is large, they default because the monthly payment does not fit the monthly budget. Computed as principal/tenure so it understates the true EMI, but the ranking across applicants is what the model uses |
| `loan_to_sanction` | How much of the approved limit they actually drew. Taking every rupee available signals they needed every rupee available |
| `balance_to_loan` | Liquidity cushion. Savings are the first thing that absorbs a bad month |
| `months_of_income_in_bank` | The same cushion in months of income — how long they could service the loan with no salary. Scale-free, so a high and a low earner compare fairly |
| `income_per_dependant` | Two people on the same salary do not have the same disposable income if one supports four others |

### Repayment behaviour

| Feature | Intuition |
|---|---|
| `delinquency_ratio` | Share of credit history spent delinquent. **Past repayment behaviour is the single best predictor of future repayment behaviour**, and the ratio normalises for how long someone has had credit |
| `avg_dpd_per_delinquency` | Separates "pays a few days late, often" from "goes 90 days dark". The second is far more dangerous. Guarded against divide-by-zero — 40% of borrowers have zero delinquent months, for whom the value is 0 by definition, not infinity |
| `credit_utilization_ratio` | Used as-is. Already leaning on revolving credit to make ends meet |

### Credit file shape

| Feature | Intuition |
|---|---|
| `open_account_ratio` | Mostly-open accounts means accumulating obligations; mostly-closed means a record of finishing what you started |
| `total_accounts` | File depth. A thin file is hard to judge; a very thick one suggests credit hunger |
| `enquiry_count` | Applications elsewhere. Shopping hard for credit is a classic distress signal |

Pass-through: `loan_tenure_months` (exposure duration), `total_loan_months` (history
length), `age`, `years_at_current_address` (residential stability),
`number_of_dependants`. One-hot: `marital_status`, `employment_status`, `residence_type`,
`loan_purpose`, `loan_type`.

### The divide-by-zero guard is not a detail

`avg_dpd_per_delinquency = total_dpd / delinquent_months` is undefined for the 40% of
borrowers with no delinquency. Unguarded, that produces `inf`, XGBoost splits on it happily,
and the API returns nonsense for real applicants. Every ratio goes through `_safe_divide`,
and `tests/test_features.py` asserts no infinities survive on the real data.

---

## 5. Why XGBoost — and when logistic regression is better

### Why XGBoost here

1. **Non-linear thresholds.** Default risk is not linear in utilisation. It sits near 1.1%
   below 30% utilisation, then explodes to 48.6% above 90%. Logistic regression needs you
   to hand-engineer that; trees find it.
2. **Interactions for free.** "High utilisation *and* renting" is worse than either alone.
   A tree encodes that as a path; logistic regression needs an explicit interaction term.
3. **Mixed feature types** with no scaling required.
4. **It won.** 0.9885 vs 0.9834 AUC, and PR-AUC 0.8862 vs 0.8598 — a 3-point PR-AUC gain,
   which matters far more at an 8.6% base rate.

### When logistic regression is genuinely the better choice

**This is the answer that separates a candidate who has thought about lending from one who
has only thought about models.** I would pick logistic regression when:

1. **Regulation demands it.** In regulated consumer lending, model risk management (SR 11-7
   in the US) and fair-lending review are far easier with a linear model. A scorecard where
   each attribute contributes a fixed number of points can be shown to a regulator on one
   page.
2. **Adverse-action notices.** A declined applicant is legally entitled to specific reasons.
   With logistic regression the reasons fall out of the coefficients directly. With XGBoost
   you need SHAP, and you must then defend SHAP itself to a compliance team.
3. **Monotonicity has to be guaranteed.** A lender cannot ship a model where *more* income
   sometimes *lowers* your score — it is indefensible to a customer and to a regulator. A
   logistic model with the right sign is monotone by construction. XGBoost needs explicit
   monotone constraints (which XGBoost does support, and which I would add before
   production).
4. **The gain is small.** Here, +0.005 AUC. If a linear model gets you within a point of the
   tree, the operational simplicity usually wins.

**Honest summary:** for this dataset XGBoost is better. For a real regulated lending
product, I would ship the logistic scorecard as the decisioning model and use XGBoost as a
challenger to find features the linear model is missing.

---

## 6. How gradient boosting actually works

**In plain words.** Imagine a panel of advisors who each guess whether a borrower defaults.
The first is very simple and gets a lot wrong. The second is not asked to guess the answer
— it is asked to predict *how wrong the first one was*, and its correction is added on top.
The third predicts the remaining error, and so on. Each advisor is deliberately weak, and
each only nudges the answer a little, but a few hundred small corrections stacked on top of
each other produce something very accurate.

**Slightly more precisely.**

1. Start with a constant prediction — the base log-odds of default.
2. Compute, for every row, the gradient of the loss with respect to the current prediction.
   For log-loss this is essentially "predicted probability minus actual outcome" — the
   signed error.
3. Fit a small decision tree to predict those gradients.
4. Add that tree's output to the running prediction, **multiplied by the learning rate** so
   each correction is a small step, not a lurch.
5. Repeat a few hundred times.

**What makes it "extreme"** (the X in XGBoost): it uses second-order gradients (the
curvature, not just the slope) to pick better splits, adds L1/L2 penalties on leaf weights
directly into the split criterion so regularisation is part of the objective rather than an
afterthought, and is engineered for speed with histogram-based splitting.

**The intuition for why it beats a single tree:** one deep tree tries to be right about
everything at once and memorises noise. Many shallow trees, each fixing a little of what is
left, spread the fitting across many small, low-variance decisions.

**Boosting vs bagging (Random Forest):** Random Forest builds many deep trees *in parallel*
on bootstrap samples and averages them — that reduces **variance**. Boosting builds shallow
trees *sequentially*, each correcting the last — that reduces **bias**. Which is why
boosting overfits more readily and needs early stopping, and Random Forest mostly does not.

---

## 7. The hyperparameters

Tuned with Optuna, 40 trials, 4-fold stratified CV on train+validation, early stopping
inside each fold. Best CV AUC **0.98900**.

| Parameter | Chosen | What it does | ↑ pushes toward |
|---|---|---|---|
| `max_depth` | **3** | Max levels per tree; controls interaction order. Depth 3 = up to 3-way interactions | **Variance** (overfit) |
| `learning_rate` | **0.061** | How much of each tree's correction is applied | **Variance** (lower = slower, better-generalising) |
| `n_estimators` | 1200 (early-stopped) | Number of boosting rounds | **Variance** |
| `subsample` | **0.705** | Fraction of rows per tree | **Variance** (lower = more randomness = less overfit) |
| `colsample_bytree` | **0.997** | Fraction of columns per tree | **Variance** |
| `min_child_weight` | **13** | Minimum weighted rows in a leaf | **Bias** (higher = more conservative) |
| `gamma` | **3.93** | Minimum loss reduction required to split | **Bias** |
| `reg_lambda` | **0.535** | L2 penalty on leaf weights | **Bias** |
| `reg_alpha` | **0.029** | L1 penalty on leaf weights (can zero them out) | **Bias** |

**How to read the result:** the search landed on shallow trees (depth 3) with strong
pruning (`gamma` 3.93, `min_child_weight` 13) and heavy row subsampling (0.705). That is a
deliberately **high-bias, low-variance** configuration. Given that two features carry most
of the signal, the search correctly concluded there were no deep interactions worth chasing
and that the risk was overfitting, not underfitting.

**Why `n_estimators` is 1200 but not really.** Early stopping on the validation fold halts
training when validation AUC stops improving for 50 rounds. So `n_estimators` is a ceiling,
not a budget — searching it as a hard count while also early-stopping would be tuning the
same thing twice.

**Bias/variance in one line:** bias is being wrong the same way every time (too simple);
variance is being wrong differently every time (memorising noise). Depth, learning rate and
sampling control the first group; the regularisation parameters control the second.

---

## 8. Class imbalance

8.594% positives. Three strategies, measured on validation rather than assumed:

| Strategy | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|
| Do nothing | 0.9889 | **0.8991** | **0.0239** |
| `scale_pos_weight = 10.63` | 0.9887 | 0.8990 | 0.0362 |
| Random undersampling (28,978 good loans discarded) | 0.9875 | 0.8896 | 0.0485 |

**What each does.** `scale_pos_weight` multiplies the gradient contribution of every
positive row by neg/pos (10.63), so the 4,297 defaults collectively push as hard as the
45,703 repayments. Nothing is discarded and nothing duplicated. Undersampling throws away
majority rows until the classes balance.

**What won, and what I picked.** Undersampling clearly lost — it discards 90% of the
majority class and, with it, most of the information about what a good borrower looks like.
"Do nothing" and `scale_pos_weight` tied on ranking.

I kept **`scale_pos_weight`**. Honest reasoning: it makes the minority-class emphasis
explicit and is the standard, defensible choice for imbalanced boosting; its worse raw
Brier (0.0362 vs 0.0239) is a *level* distortion, not a *ranking* one, and calibration
fixes exactly that — final Brier is **0.0240**. If asked "but do-nothing tied you", the
answer is: on this dataset the imbalance is mild enough not to matter, and I would rather
ship the configuration that stays correct when the base rate shifts.

**What I did not do: SMOTE.** Synthesising interpolated borrowers who never existed is hard
to defend to a credit committee, and it tends to blur exactly the decision boundary you
care about. Re-weighting achieves the same emphasis without inventing data.

---

## 9. Every metric, defined

Actual numbers are from the test set (7,499 loans, 8.59% base rate).

| Metric | Plain definition | Mine | "Good" in credit risk |
|---|---|---|---|
| **ROC-AUC** | Take a random defaulter and a random repayer. The chance the model scores the defaulter riskier | **0.9885** | 0.70–0.80 is a solid production scorecard. Mine is unrealistically high — see §12 |
| **PR-AUC** | Area under precision–recall. The honest headline when only 8.6% are positive, because ROC-AUC is flattered by the huge number of easy negatives | **0.8862** | Compare to the base rate: a useless model scores 0.086 |
| **Gini** | `2 × AUC − 1`. Same information, different scale — the number credit teams actually quote | **0.9771** | 0.40–0.60 typical |
| **KS** | Line everyone up by risk. The biggest gap between "% of bad loans caught so far" and "% of good loans caught so far" | **90.30** | **30–40 is healthy.** Above 60 in production almost always means leakage — which is why I investigated (§12) |
| **Brier** | Mean squared error of the probabilities. Unlike AUC it punishes a model that ranks well but is badly calibrated | **0.0240** | Lower is better; compare against the base-rate-only score of 0.0785 |
| **ECE** | Average gap between predicted probability and observed frequency. Isolates calibration from discrimination | **0.0063** | Under 0.01 is well calibrated |
| **Precision** | Of those we declined, what share would actually have defaulted | **58.16%** | Trades off against recall |
| **Recall** | Of all defaulters, what share did we catch | **96.27%** | The one that matters most when FN costs 10× FP |
| **F1** | Harmonic mean of precision and recall | **0.7251** | Rarely the right objective in lending — it implicitly assumes FP and FN cost the same |
| **PSI** | How far a feature's distribution has moved from training | **0.0036** max | <0.10 stable, 0.10–0.25 watch, >0.25 retrain |

**The decile table** is the one a credit committee always asks for. Split the book into ten
equal groups by predicted risk and show the *actual* default rate in each:

| Decile | Loans | Defaults | Rate | Lift |
|---|---|---|---|---|
| 1 (riskiest) | 750 | 560 | 74.67% | 8.69× |
| 2 | 750 | 84 | 11.20% | 1.30× |
| 3–10 | 5,999 | 0 | 0.00% | 0.00× |

It must be **monotone** — the rate has to fall as you move to safer deciles — and mine is.
A model can post a good AUC and still rank-order badly in the middle of the book, which is
precisely where pricing decisions get made.

---

## 10. Calibration

**Why it matters.** Raw XGBoost outputs are not probabilities. The model is trained to
*rank*, and `scale_pos_weight` deliberately distorts the output level on top of that. If
you price a loan off the number, then "PD = 3%" must actually mean three defaults per
hundred such loans — otherwise every expected-loss calculation downstream is wrong.

**What I did.** Fitted both calibrators on the validation fold and compared:

| Method | ROC-AUC | Brier | ECE |
|---|---|---|---|
| None (raw XGBoost) | 0.9888 | 0.0381 | 0.0479 |
| **Isotonic** ✅ | 0.9885 | **0.0240** | **0.0063** |
| Sigmoid (Platt) | 0.9888 | 0.0262 | 0.0159 |

Isotonic cut calibration error **7.6×**. It costs a little PR-AUC (0.9009 → 0.8862) because
it is a step function that ties nearby scores together — an acceptable trade when the number
drives pricing.

**The methodological point worth raising unprompted.** A calibrator cannot be judged on the
fold it was fitted to, and it must not be judged on test — that is selecting a model on the
holdout. So validation is split in half: both calibrators are fitted on one half, scored on
the other, the lower Brier wins, and only then is the winner refitted on all of validation.
The test column above is reporting, computed *after* the choice was made.

**Isotonic vs Platt.** Isotonic fits a free-form monotone step function — flexible but
data-hungry. Platt fits a single sigmoid — two parameters, safe on small folds, but it can
only undo distortions that happen to be sigmoid-shaped. With ~7,500 validation rows,
isotonic had enough data, and it won on the held-out half.

**One more honest detail.** Isotonic's lowest step is exactly 0.0 and its highest exactly
1.0. A reported PD of zero is indefensible — no lender believes an applicant can never
default, and no pricing formula can divide by it. Reported PDs are clamped to
`[0.0001, 0.9999]`. The clamp is monotone, so no ranking metric changes.

---

## 11. The leakage story

**The design call.** Some columns are only knowable *after* the loan is disbursed:

* `principal_outstanding` — the balance part-way through repayment
* `processing_fee`, `gst`, `net_disbursement` — post-approval
* `installment_start_dt` — post-approval

A lender does not have any of these when deciding. So the model is built as an
**application-time scorecard** and they are all excluded.

**The evidence.** I refit the identical model with `principal_outstanding` added back:

| Feature set | Test ROC-AUC |
|---|---|
| Application-time only (production) | 0.98882 |
| + `principal_outstanding` | 0.98868 |
| **Inflation** | **−0.00014** |

**It inflated nothing — the AUC actually went marginally down.** Its raw correlation with
`default` is −0.011.

**Why this is still worth having done, and worth saying so.** The honest framing is: "I
expected this to inflate the score and it did not. That is a negative result, and I am
reporting it rather than quietly dropping the experiment. The feature is still excluded,
because a feature you cannot obtain at scoring time is unusable regardless of what it does
to the metric — if it *had* helped, that would have been a reason to be more careful, not
less."

**Belt and braces.** `tests/test_features.py::test_no_post_disbursal_column_can_reach_the_model`
asserts the exclusion in code, and `tests/test_api.py` asserts the API never advertises
one. A future contributor cannot reintroduce leakage without a test going red.

---

## Why is the AUC so high?

**This is the first question a good interviewer will ask, so raise it first.**

0.9885 AUC and KS 90.3 are not credible for a credit model. A production consumer scorecard
lands at 0.70–0.80 AUC and KS 30–40. So I investigated before writing the number down.

**It is not leakage.** Three independent checks:

1. Every feature is application-time, and tests enforce it.
2. The ablation adds the most suspicious post-disbursal column back: **−0.0001 AUC**.
3. The out-of-time model — trained only on loans disbursed before July 2023, tested on
   those after — still scores **0.9855**. Leakage does not survive a time split.

**The actual reason: the data is synthetic and the target is nearly deterministic.**

| Utilisation | No delinquency | 0–10% | 10–30% | >30% |
|---|---|---|---|---|
| **<30%** | **0.00%** (0 of 8,150) | 0.16% | 0.36% | 11.02% |
| **30–60%** | 0.17% | 0.93% | 2.04% | 20.81% |
| **60–80%** | 1.07% | 6.02% | 11.10% | 44.39% |
| **>80%** | 19.05% | 39.76% | 52.35% | 84.95% |

A cohort of **8,150 real borrowers with a default rate of exactly zero does not exist**.
Real data is noisy: some low-utilisation, never-delinquent borrowers lose a job and default
anyway. This target was generated as a clean function of these features.

`credit_utilization_ratio` alone gives 0.877 AUC. Two features give 0.940.

**What I say in an interview:** "The pipeline is production-shaped and the methodology
transfers. The absolute numbers do not — on a real portfolio I would expect 0.70–0.80 AUC
and a decile-1 rate near 25% rather than 75%. I would rather tell you that up front than
have you find it."

---

## 13. Threshold selection

**Why not 0.5.** 0.5 is correct only if a missed defaulter and a wrongly-declined customer
cost the same. They do not.

**The cost assumption, stated explicitly:**

* **False negative** (approve a loan that defaults) → you lose the unrecovered principal.
  Loss given default on a mixed secured/unsecured book runs around **60% of exposure**.
* **False positive** (decline a customer who would have repaid) → you lose only the interest
  margin you never earned. At a ~3% net interest margin over a ~2-year average life, roughly
  **6% of exposure**.

**The ratio is what matters: a missed default is ~10× worse.** So the cut-off belongs well
below 0.5. Sweeping every candidate and minimising expected cost gives **0.080** — chosen on
validation, then applied unchanged to test.

| Threshold | Recall | Precision | Approval rate | Bad rate approved | Defaults missed |
|---|---|---|---|---|---|
| **0.080** (cost-optimal) | **96.27%** | 58.16% | 85.78% | **0.37%** | **24** |
| 0.500 (naive) | 73.45% | 85.07% | 92.59% | 2.46% | 171 |

The naive threshold approves 6.8 points more of the book but **misses 171 defaults instead
of 24**. Under the stated costs the extra margin on those approvals nowhere near covers 147
additional defaults.

**Be ready to defend the numbers themselves.** 60% and 6% are assumptions, not measurements
— I do not have recovery data or a pricing sheet. The *ratio* is what drives the threshold,
and I would replace both with the lender's actual LGD and margin on day one. The framework
is the deliverable; the constants are placeholders.

---

## 14. Serving architecture

**One artifact.** `models/model.joblib` is a complete sklearn `Pipeline`:

```
raw application fields
  → FunctionTransformer(build_features)    ← src/features.py, the same function training used
  → ColumnTransformer(OneHot + passthrough)
  → XGBClassifier
  → CalibratedClassifierCV (isotonic)
```

**Why serialize the whole pipeline rather than just the booster.** If the API reimplemented
feature engineering, the two implementations would drift the first time someone changed a
ratio. Because `build_features` is *inside* the artifact, the API physically cannot compute
a feature differently. **This is the answer to "how do you prevent training/serving skew?"**
— it is a structural guarantee, not a process.

And it is tested: `train.py` saves one test row plus the probability it produced, and
`tests/test_api.py::test_prediction_matches_what_train_py_recorded` asserts the API returns
the same number after a joblib round-trip, through Pydantic coercion, in a fresh process.
That single test is the most valuable one in the suite.

**Model loaded once, at startup**, in a FastAPI lifespan handler — not per request. Loading
per request would dominate the latency budget entirely.

**Fails loudly.** If the artifact is missing the app raises at startup rather than booting
and serving 500s. A container that can never serve a valid request must not report itself
healthy, or an orchestrator will keep routing traffic to it.

**Validation at the edge.** Pydantic v2 with realistic bounds (age 18–100, income > 0,
utilisation 0–100, enums for every categorical) plus cross-field rules
(`loan_amount ≤ sanction_amount`, `delinquent_months ≤ total_loan_months`,
`total_dpd = 0` when `delinquent_months = 0`). Bad input returns **422 with a readable
message**, never a 500 and never a confident score computed from nonsense.

**Latency**, 300 requests after warm-up:

| | p50 | p95 | p99 |
|---|---|---|---|
| `POST /predict` | 3.62 ms | 3.83 ms | 4.32 ms |

Batch: 1,000 applicants in **12.6 ms** total (0.013 ms each) — one vectorised
`predict_proba` instead of a thousand.

**Observability.** One JSON log line per request with a request id, method, path, status
and latency; `X-Request-ID` and `X-Response-Time-ms` on every response.

---

## 15. Monitoring

**PSI is the primary alarm.** Freeze the training bin edges, then watch new traffic move
through them. `<0.10` stable, `0.10–0.25` investigate, `>0.25` retrain. Current maxima:
0.0036 (train vs test) and 0.0023 (in-time vs OOT).

**What I would alarm on in production:**

| Signal | Threshold |
|---|---|
| PSI on any top-5 feature | > 0.25 |
| Approval rate | moves > 5 points |
| Observed vs predicted default rate | diverges > 2 points |
| p99 latency | > 200 ms |
| Rate of 422s | a spike means an upstream schema change |

**Retraining triggers:** PSI breach; observed-vs-predicted divergence; scheduled every 6
months regardless; any credit-policy or product-mix change; any macroeconomic regime change.

**What breaks first — the answer interviewers like.** **Calibration, before discrimination.**
The *ranking* of applicants is fairly robust to population shift; the *level* of the
probabilities drifts almost immediately. That is a good reason to keep the calibrator as a
separable wrapper — recalibrating on recent data is cheap and does not require a full
retrain.

**The subtler production problem: reject inference.** Declined applicants never generate an
outcome, so the next training set only contains loans you approved. Over time the model's
view of risk narrows and it becomes over-confident about the population it already likes.
Mitigation: approve a small random sample of marginal applicants to keep generating
unbiased labels, or apply a formal reject-inference method. Not implemented here.

---

## 16. Weaknesses and next steps

**Where the model is weak**

1. **Synthetic data** — the metrics do not transfer (§12). This dominates everything else.
2. **Bimodal output** — 94.5% of the book lands at Excellent or Poor, leaving 5.5% in
   between. Risk-based pricing needs a spectrum. It is a data property, not a calibration
   artefact: Platt is *more* concentrated at 97.3%.
3. **Housing tenure does heavy lifting** — a 14-point approval gap between owners and
   renters, and the model ranks renters measurably worse (AUC 0.9798 vs 0.9923).
4. **`default` is undefined** — 90 DPD? Charge-off? The definition determines what is being
   predicted.
5. **No downturn in the data** — the default rate is flat across all 21 quarters, including
   COVID, which is itself implausible.
6. **No uncertainty estimate** — a point PD, with no interval.

**With more time, in priority order**

1. **Monotone constraints** on `credit_utilization_ratio`, `loan_to_income` and
   `delinquency_ratio`. XGBoost supports them natively. A model where more income can lower
   your score is indefensible to a customer and a regulator, and this is cheap insurance.
2. **Proxy-discrimination analysis on `residence_type`** — regress it on protected
   attributes and measure how much predictive power survives once that variance is removed.
3. **Remove `age`** if this were ever deployed in a US context — it is protected under ECOA.
   Its SHAP importance is 13th of 31, so the cost is small.
4. **Reject-inference design**, before the first retraining cycle bakes in the bias.
5. **A proper EMI** using the actual interest rate rather than principal/tenure.
6. **Prediction intervals**, via conformal prediction or a quantile ensemble.
7. **Champion/challenger in production** — ship the logistic scorecard for decisioning and
   run XGBoost alongside to find what the linear model misses.

---

## 17. Q&A — 20 interview questions

### The hostile ones

**Q1. Your AUC is 0.9885. Are you sure you don't have leakage?**

Yes, and I checked three ways rather than assuming. First, every feature is available at
application time, and there is a test that fails if a post-disbursal column ever enters the
feature set. Second, I ran an ablation adding `principal_outstanding` — the most suspicious
column — back in: AUC moved by −0.0001. Third, an out-of-time model trained only on
pre-July-2023 loans still scores 0.9855, and leakage does not survive a time split.

The real reason is that the data is synthetic. 8,150 borrowers with low utilisation and no
delinquency have a default rate of exactly zero. Real portfolios are not that clean. So the
pipeline is sound and the method transfers, but I would expect 0.70–0.80 on a real book and
I would not quote this number without that caveat.

**Q2. Why not a neural network?**

On 50,000 rows of tabular data with 22 features, gradient-boosted trees are the stronger
default and this is well established empirically. A neural net needs more data to beat
trees on tabular problems, needs careful preprocessing that trees do not, and costs a great
deal of explainability — which is the currency of regulated lending. I would also have to
explain backpropagation to a credit committee to get a model that, here, would probably be
slightly worse. If I had a million rows plus unstructured data — transaction narratives,
documents — the calculus changes.

**Q3. How would you explain a rejection to a customer?**

With SHAP, translated into plain language. For any applicant I can rank the features that
pushed their score down and phrase them concretely: "your credit utilisation is 94%,
compared with an average of 37% among the applicants we approve" and "you have been late on payments in 9
of the last 60 months". That is what an adverse-action notice requires — specific,
actionable reasons, not "the model said so", which is not a lawful answer.

I would also flag the tension honestly: a tree ensemble makes this harder than a linear
scorecard, where the reasons fall straight out of the coefficients. That is one of the
reasons I would consider shipping logistic regression as the decisioning model even though
XGBoost scores better.

**Q4. Your recall is 96% but precision is only 58%. Isn't that a bad model?**

It is a deliberate choice, not an accident. The threshold was set by an explicit cost
assumption: a missed default costs about 60% of exposure, a wrongly-declined customer costs
about 6% — roughly 10:1. At that ratio, buying recall with precision is correct. At 0.080 I
miss 24 defaults; at 0.5 I would miss 171 while approving only 6.8 points more of the book.

If you tell me the real LGD and margin, I will re-derive the threshold in one line — the
cost constants live in `config.py` precisely so that is a one-line change.

**Q5. You dropped `gender`, but the model still approves men at a higher rate. Explain.**

Removing a variable does not remove disparate impact, and I would not claim otherwise. The
gap is 1.66 points — 86.45% for men versus 84.79% for women — against a real 0.49-point
difference in observed default rate. The four-fifths ratio is 0.981, well above the 0.80
threshold, and the model's AUC differs by only 0.004 between the groups, so it ranks both
about equally well.

I retained `gender` in the raw data specifically so I could measure this. You cannot audit
a disparity on an attribute you have thrown away. The larger concern in this model is not
gender at all — it is `residence_type`, which produces a 14-point approval gap and needs a
proxy-discrimination review before it goes anywhere near production.

### On the data and cleaning

**Q6. Why treat the 62 missing residence types as a category instead of imputing?**

Because the mode is `Owned`, which is the lowest-risk level at 5.6% default versus 16.3%
for `Rented`. Mode-imputing would silently tell the model these 62 applicants are the safest
kind of borrower, on no evidence. An `Unknown` category lets the tree decide, and "the
applicant did not answer" is genuinely informative in an application flow. At 0.12% of rows
it cannot move the metrics — I chose it for the reasoning, not the score.

**Q7. You repaired 7 rows instead of dropping them. Why is that not making data up?**

Because the value is forced, not guessed. `processing_fee` is corrupted on those rows but
`gst` and `net_disbursement` are not, and in this dataset `gst` is exactly 9× the fee. So
the true fee is `gst / 9` — and the reconstruction is confirmed independently by the
identity `loan_amount − processing_fee − gst = net_disbursement`, which goes from 7
violations to 0. Two independent constraints agree. Dropping recoverable rows is a real
cost when the positive class only has 4,297 members.

**Q8. Why drop `city` and `state`? Surely location matters in credit.**

It usually does, but not here. There are only 10 cities and 9 states, and default rates run
8.1% to 9.1% — a spread well inside one standard error at this sample size. There is
nothing to gain. And location is the textbook proxy for protected characteristics; that is
what redlining *is*. When a feature carries no signal and carries fair-lending risk, both
arguments point the same way.

If the rates had differed materially I would have kept them and done the proxy analysis
properly rather than dropping them reflexively.

**Q9. The brief told you `gst` was inconsistent. You disagreed. Why?**

The brief said `gst` is about 9× `processing_fee`, which is not a plausible 18% GST on the
fee. That observation is correct but the conclusion was incomplete. Checking the arithmetic:
`gst` is exactly 18% of `loan_amount`, `processing_fee` is exactly 2%, and
`net_disbursement` is exactly 80%. The 9× ratio is just 18/2. So the column is not
inconsistent — it is perfectly consistent and perfectly redundant, an exact linear function
of `loan_amount` carrying zero independent information.

Same action, better reason. And the brief listed the 7 identity failures and the fee
outliers as two problems; they are the same 7 rows.

### On modelling

**Q10. Walk me through gradient boosting.**

Start with a constant prediction. Compute how wrong it is on every row — for log-loss that
is essentially predicted probability minus actual outcome. Fit a small tree to predict those
errors, and add its output to the running prediction, scaled down by the learning rate so
each correction is a nudge. Repeat a few hundred times. Each tree is deliberately weak and
only fixes a little of what the previous ones left behind.

The "extreme" part is that XGBoost uses second-order gradients to choose better splits and
folds L1/L2 penalties directly into the split criterion, so regularisation is part of the
objective rather than bolted on.

Contrast with Random Forest: that builds deep trees in parallel and averages them to cut
**variance**. Boosting builds shallow trees sequentially to cut **bias** — which is exactly
why boosting overfits more readily and needs early stopping.

**Q11. Your search chose `max_depth=3` and `gamma=3.93`. What does that tell you?**

That the model needs to be simple. Depth 3 allows at most three-way interactions, and gamma
of 3.93 prunes any split that does not buy a substantial loss reduction. Combined with
`min_child_weight=13` and 70% row subsampling, that is a deliberately high-bias,
low-variance configuration.

It makes sense given the data: `credit_utilization_ratio` alone gives 0.877 AUC and two
features give 0.940. There are no deep interactions to find, so the real risk was
overfitting, and the search correctly regularised hard rather than chasing complexity.

**Q12. How did you handle imbalance, and why that choice?**

I tested three options rather than picking one. Undersampling lost clearly — it discards
28,978 good loans and with them most of the information about what a good borrower looks
like. Doing nothing and `scale_pos_weight` tied on ranking (PR-AUC 0.8991 vs 0.8990).

I shipped `scale_pos_weight` because it makes the minority-class emphasis explicit, is the
standard defensible choice, and its worse raw Brier is a level distortion that calibration
is designed to fix — final Brier is 0.0240. I deliberately avoided SMOTE: synthesising
borrowers who never existed is hard to defend to a credit committee and tends to blur the
decision boundary you care most about.

**Q13. Why does your logistic baseline also score 0.983? Doesn't that make XGBoost pointless?**

On this dataset, nearly. The gain is +0.005 AUC, though PR-AUC improves more meaningfully
(0.8862 vs 0.8598) and that matters more at an 8.6% base rate.

But that is the *right* comparison to have run, because it is the argument for shipping the
simpler model. If a linear scorecard gets within half a point of the tree, the operational
and regulatory simplicity usually wins in real lending. I would ship logistic for
decisioning and keep XGBoost as a challenger.

**Q14. What is calibration and why do you care?**

Ranking and probability are different problems. XGBoost is trained to rank, and
`scale_pos_weight` distorts the output level further. So the raw output tells you who is
riskier but not what anyone's actual chance of default is.

That matters the moment you price off the number: expected loss is PD × LGD × exposure, and
if the PD is wrong by a factor of two every price is wrong. Isotonic calibration cut Brier
from 0.0381 to 0.0240 and calibration error from 0.0479 to 0.0063 — a 7.6× improvement.

I would add one methodological note: I selected the calibrator on a held-out half of the
validation set, not on test, because choosing on test is selecting a model on the holdout.

**Q15. Isotonic or Platt — and how did you decide?**

Isotonic, chosen empirically. It fits a free-form monotone step function, which is flexible
but needs data; Platt fits a single sigmoid, which is safe on small folds but can only undo
sigmoid-shaped distortions. With ~7,500 validation rows isotonic had enough data and won on
the held-out half (Brier 0.0228 vs 0.0252).

The cost is that isotonic ties nearby scores, which is why PR-AUC drops slightly, 0.9009 to
0.8862. Worth it when the number drives pricing. One consequence worth flagging: isotonic's
end steps are exactly 0 and 1, so I clamp reported PDs to [0.0001, 0.9999] — no lender can
price off "will never default".

### On deployment and production

**Q16. How do you prevent training/serving skew?**

Structurally, not procedurally. `src/features.py` is imported by training *and* serialized
inside the model artifact, so the API does not have its own copy of the feature logic to
drift from. There is one implementation.

And it is verified: training saves a fixed row along with the probability it produced, and a
test asserts the API returns the same number after a joblib round-trip, through Pydantic
coercion, in a fresh process. If anyone changes a ratio without retraining, that test goes
red.

**Q17. What breaks first in production?**

Calibration, before discrimination. The ranking is fairly robust to population shift; the
probability *levels* drift almost immediately as the applicant mix changes. That is why the
calibrator is a separable wrapper — recalibrating on recent data is cheap and does not need
a full retrain.

The subtler failure is reject inference. Declined applicants never produce an outcome, so
each retraining cycle sees only loans you approved, and the model's view of risk narrows
over time. I would reserve a small random sample of marginal applicants for approval to keep
generating unbiased labels.

**Q18. How would you monitor this?**

PSI as the primary signal — freeze the training bin edges and watch new traffic move through
them, alarming above 0.25 on any top-5 feature. Alongside that: approval rate moving more
than 5 points, observed default rate in the approved book diverging more than 2 points from
predicted, p99 latency, and the rate of 422s (a spike means an upstream schema change).

Retraining triggers are a PSI breach, an observed-versus-predicted divergence, a scheduled
6-month refresh, or any change to credit policy, product mix, or the definition of default.

**Q19. Your API returns a decision. Is that an automated lending decision?**

No, and the documentation is explicit about it. It is a recommendation against a threshold,
and the threshold is returned in every response so the caller can see exactly what it was
compared against. Every decline should be human-reviewable and every applicant is entitled
to specific reasons.

I would also push back on fully automating this model in particular, given the synthetic
training data — the metrics simply are not a forecast of real performance.

**Q20. What would you do differently with another month?**

Four things, in order. Add monotone constraints on utilisation, loan-to-income and
delinquency ratio — a model where more income can lower your score is indefensible to a
regulator, and XGBoost supports them natively. Do the proxy-discrimination analysis on
`residence_type`, which is currently the biggest fair-lending exposure. Design reject
inference before the first retraining cycle bakes the bias in. And get real data — the
single highest-value change here is not a modelling one, because no amount of tuning fixes a
synthetic target.

I would also compute a proper EMI using the actual interest rate rather than
principal-over-tenure, and add prediction intervals via conformal prediction so the service
can say how confident it is, not just what it thinks.
