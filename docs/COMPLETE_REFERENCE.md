# Complete Reference — Everything About This Project

**One document. Every concept from zero, every file, every function, every number.**

If you read only one thing before an interview, read this. It assumes you know basic
Python and basic machine learning (train/test split, overfitting, what a classifier is) and
**nothing else** — not SHAP, not FastAPI, not Docker, not credit risk.

Other docs exist and go deeper on single topics ([`START_HERE.md`](START_HERE.md) for a
reading order, [`API_AND_DEPLOYMENT.md`](API_AND_DEPLOYMENT.md) for serving,
[`INTERVIEW_GUIDE.md`](INTERVIEW_GUIDE.md) for Q&A, [`MODEL_CARD.md`](MODEL_CARD.md) for
fairness). This one is the single source: it covers all of it, in order, with the code.

---

## Contents

**Part 1 — [Orientation](#part-1--orientation)**
**Part 2 — [Concepts from zero](#part-2--concepts-from-zero)**
**Part 3 — [The data](#part-3--the-data)**
**Part 4 — [The code, file by file](#part-4--the-code-file-by-file)**
**Part 5 — [The artifacts](#part-5--the-artifacts)**
**Part 6 — [Infrastructure](#part-6--infrastructure)**
**Part 7 — [One request, end to end](#part-7--one-request-end-to-end)**
**Part 8 — [Every number and where it comes from](#part-8--every-number-and-where-it-comes-from)**

---

# Part 1 — Orientation

## 1.1 What this is, in one paragraph

A **credit scorecard**. It takes a loan application — age, income, what they want to
borrow, and their credit-bureau history — and returns the probability they will default,
a 300–900 credit score, and an approve/decline recommendation. It is trained on 50,000
historical loans of which 8.594% defaulted. It is served as a web API, packaged in Docker,
and running live at
[credit-risk-api-1dbj.onrender.com](https://credit-risk-api-1dbj.onrender.com/docs).

## 1.2 The mental model

Four ideas drive every decision in the repo. If you remember only these, you can
reconstruct most of the reasoning:

1. **Only use what a lender knows at decision time.** Several columns in the data only
   exist *after* the loan is paid out. Using them would make the model unusable in
   production. This is the "leakage" decision.
2. **Rank, then calibrate.** XGBoost is trained to put risky people above safe people. It
   is *not* trained to output true probabilities. If you price a loan off the number, you
   must fix that separately.
3. **The cut-off comes from money, not from 0.5.** Approving a defaulter costs ~10× what
   declining a good customer costs. So the threshold belongs far below 0.5.
4. **Feature code must be written once.** If training and serving each compute features,
   they will drift. Ours is one function, serialized inside the model file.

## 1.3 How data flows

```
data/raw/*.csv  (3 files × 50,000 rows)
      │
      ▼
src/data.py ──────── merge 1:1 on cust_id, apply 6 cleaning rules → 49,992 rows
      │
      ▼
src/train.py ─────── split 70/15/15 (+ out-of-time)
      │               ├─ logistic baseline
      │               ├─ compare 3 imbalance strategies
      │               ├─ Optuna: 40 trials tuning XGBoost
      │               ├─ calibrate (isotonic vs Platt)
      │               ├─ leakage ablation
      │               └─ evaluate → reports/figures/, reports/tables/
      │
      ▼
models/model.joblib ── ONE sklearn Pipeline:
      │                  raw fields → build_features → one-hot → XGBoost → calibrator
      │
      ▼
api/main.py ──────── loads it once at startup, serves POST /predict
      │
      ▼
Docker → Render → https://credit-risk-api-1dbj.onrender.com
```

`src/features.py` sits in the middle of this: imported by `train.py`, **and** serialized
inside `model.joblib`, so `api/main.py` uses the identical code without importing it
directly.

## 1.4 The headline results

Test set: 7,499 held-out loans, 8.59% default rate.

| Metric | Logistic baseline | **XGBoost (final)** | Out-of-time |
|---|---|---|---|
| ROC-AUC | 0.9834 | **0.9885** | 0.9855 |
| PR-AUC | 0.8598 | **0.8862** | 0.8759 |
| KS | 88.14 | **90.30** | 90.40 |
| Brier | 0.0538 | **0.0240** | 0.0243 |

At the chosen threshold of 0.080: **96.27% of defaults caught**, approval rate 85.78%, and
the bad rate inside the approved book falls from 8.59% to **0.37%**.

## 1.5 The caveat you must always state

**0.9885 AUC is not believable for a credit model.** Real scorecards reach 0.70–0.80.

It is not leakage — that was checked three ways (§4.6.6). The dataset is **synthetic** and
the target was generated almost deterministically from two bureau features: applicants
with low utilisation and no delinquency default at a rate of **exactly zero**, 0 out of
8,150. Real borrowers never behave that cleanly.

**The method transfers. The numbers do not.** Say this before anyone asks.

---

# Part 2 — Concepts from zero

Every term used anywhere in this project, defined once, properly.

## 2.1 Credit domain

**Default** — the borrower stopped paying. In this dataset it is a true/false flag. The
precise definition (90 days late? written off?) is *not documented with the data*, which is
a genuine limitation: the definition determines what you are actually predicting.

**Delinquent** — behind on payments but not yet in default. `delinquent_months` counts how
many months of their history were late.

**DPD (days past due)** — how many days late a payment was. `total_dpd` is the cumulative
sum across their history. The distinction that matters: being 3 days late twelve times is
very different from being 90 days dark once, and `total_dpd / delinquent_months` separates
them.

**Credit utilisation** — of the revolving credit available to you (credit-card limits), what
percentage are you currently using? 85% means nearly maxed out. **This is the single
strongest predictor in our model**, and that is realistic — it signals someone already
leaning on credit to make ends meet.

**Enquiry count** — how many times lenders have checked your credit file. Lots of enquiries
means you are applying everywhere, which is a distress signal.

**EMI (equated monthly instalment)** — the fixed monthly payment. The key underwriting
insight: borrowers do not default because a loan is large, they default because the
*monthly payment* does not fit the monthly budget.

**Secured vs unsecured** — secured loans have collateral the lender can seize (house, car).
Unsecured do not. Counter-intuitively, secured loans default *more* in this data (10.8% vs
4.5%) — because home loans are secured and they are the riskiest category here.

**Sanction vs loan amount** — `sanction_amount` is the limit approved; `loan_amount` is what
was actually drawn. Drawing the full limit signals the borrower needed every rupee.

**Exposure** — how much money the lender has at risk.

**LGD (loss given default)** — if they default, what fraction of the exposure do you
actually lose after recovering collateral? We assume **60%**.

**PD (probability of default)** — the number this model outputs.

**Expected loss = PD × LGD × exposure.** This formula is *why* calibration matters: if PD
is wrong by a factor of two, every price is wrong by a factor of two.

**Scorecard** — the industry term for a credit risk model, from when it was literally a
table of points.

**Adverse-action notice** — the legally required letter telling a declined applicant *why*.
Required under the Equal Credit Opportunity Act (ECOA) in the US and the RBI fair practices
code in India. **"The model said no" is not a lawful reason** — this is the entire practical
motivation for SHAP (§2.4).

**Reject inference** — declined applicants never produce an outcome, so your next training
set only contains people you approved. Over time the model's view of risk narrows. We have
not solved this; it is documented as a limitation.

**Four-fifths rule** — a US fair-lending rule of thumb: one group's approval rate should be
at least 80% of the most-approved group's. Ours is 0.981 for gender, well clear.

## 2.2 The shape of this ML problem

**Binary classification with class imbalance.** 8.594% of rows are positive (defaulted).

**Why accuracy is useless here.** A model that predicts "nobody defaults" is **91.4%
accurate** and catches zero defaults. Accuracy is off the table from the first line of code.
This single fact drives the entire metric choice in §2.3.

**Three ways to handle imbalance**, all three of which we measured:

- **Do nothing** — let the model see the raw 8.6%.
- **`scale_pos_weight`** — multiply the gradient contribution of every positive row by
  neg/pos (= 10.63 here), so the 4,297 defaults collectively push as hard as the 45,703
  repayments. Nothing is discarded, nothing duplicated.
- **Undersampling** — randomly throw away majority rows until balanced. Cheap, but discards
  ~29,000 good loans and the information about what a good borrower looks like.

**What we did not do: SMOTE** (synthesising interpolated fake borrowers). It is hard to
defend to a credit committee and tends to blur exactly the decision boundary you care
about. Re-weighting achieves the same emphasis without inventing people.

**Stratified split** — when splitting train/test, preserve the 8.594% positive rate in each
part. Without stratification, random chance could give the test set 7% or 10% and the
metrics would not be comparable.

**Out-of-time (OOT) split** — train on loans disbursed before a cutoff date, test on loans
after. Answers the question a risk team actually asks: *"if we had trained this a year ago,
how would it have done since?"* A random split cannot answer that.

## 2.3 Every metric, defined

Our numbers are from the test set (7,499 loans, 8.59% base rate).

### ROC-AUC = **0.9885**

**Receiver Operating Characteristic — Area Under Curve.** Take one random defaulter and one
random repayer. ROC-AUC is the probability the model scores the defaulter as riskier. 0.5
is a coin flip, 1.0 is perfect.

*What good looks like:* **0.70–0.80 in real consumer credit.** Ours is far higher because
the data is synthetic (§1.5).

### PR-AUC = **0.8862**

**Precision-Recall Area Under Curve** (also called average precision). More honest than
ROC-AUC when positives are rare: ROC-AUC is flattered by the huge number of easy negatives,
while PR-AUC only looks at how well you find the rare positives.

*Reference point:* a useless model scores ≈ the base rate, 0.086.

### Gini = **0.9771**

Just `2 × AUC − 1`. Identical information, different scale. Credit teams quote this one.
Typical production value: 0.40–0.60.

### KS = **90.30**

**Kolmogorov–Smirnov statistic.** Line everyone up by predicted risk. Walk down the list
tracking two running percentages: "what share of all bad loans have I seen so far" and
"what share of all good loans have I seen so far". KS is the **largest gap** between those
two curves, ×100.

*What good looks like:* **30–40 is a healthy production scorecard. Above 60 almost always
means leakage** — which is exactly why ours (90.3) triggered an investigation.

### Brier score = **0.0240**

Mean squared error of the probabilities: `mean((predicted − actual)²)`. Unlike AUC it
punishes a model that *ranks* well but assigns wrong probability *levels*. Lower is better.
Reference: predicting the base rate for everyone scores 0.0785.

### ECE = **0.0063**

**Expected Calibration Error.** Bin the predictions; in each bin compare "what the model
said on average" against "what actually happened"; average the gaps weighted by bin size.
Brier mixes calibration and discrimination together — ECE isolates calibration. Under 0.01
is well calibrated.

### Precision / Recall / F1 = **58.16% / 96.27% / 0.7251**

- **Recall** — of all actual defaulters, what share did we flag? The one that matters most
  when a false negative costs 10× a false positive.
- **Precision** — of everyone we declined, what share would really have defaulted?
- **F1** — harmonic mean of the two. **Rarely the right objective in lending**, because it
  implicitly assumes false positives and false negatives cost the same. They do not.

### PSI = **0.0036** (max, train vs test)

**Population Stability Index.** Measures how far a feature's distribution has moved from
training. Freeze the training bin edges, push new data through them, and compare the
proportions:

```
PSI = Σ (actual% − expected%) × ln(actual% / expected%)
```

*Industry thresholds:* `< 0.10` stable · `0.10–0.25` investigate · `> 0.25` retrain.

This is the **primary production drift alarm**.

### The decile table

Split the book into ten equal groups by predicted risk, show the *actual* default rate in
each. **Must be monotone** — the rate has to fall as you move to safer deciles. A model can
have a good AUC and still rank-order badly in the middle of the book, which is exactly
where pricing decisions happen.

Ours (test set):

| Decile | Loans | Defaults | Rate | Lift |
|---|---|---|---|---|
| 1 (riskiest) | 750 | 560 | 74.67% | 8.69× |
| 2 | 750 | 84 | 11.20% | 1.30× |
| 3–10 | 5,999 | 0 | 0.00% | 0.00× |

Monotone ✓. **100% of defaults sit in the top two deciles** — extreme concentration, again
the synthetic-data effect.

## 2.4 Engineering and ML machinery

**Gradient boosting** — the algorithm behind XGBoost. In plain words:

> Start with a constant guess. Compute how wrong it is on every row. Fit a *small* decision
> tree to predict those errors. Add that tree's correction to the running prediction,
> scaled down by the learning rate so each step is a nudge. Repeat a few hundred times.

Each tree is deliberately weak and only fixes a little of what the previous ones left
behind. The "extreme" in XGBoost: it uses second-order gradients (curvature, not just
slope) to pick better splits, and folds L1/L2 penalties directly into the split criterion.

*Versus Random Forest:* RF builds deep trees **in parallel** and averages them — that
reduces **variance**. Boosting builds shallow trees **sequentially**, each correcting the
last — that reduces **bias**. Which is why boosting overfits more readily and needs early
stopping.

**Early stopping** — watch performance on a validation fold while training and stop adding
trees when it stops improving. Prevents overfitting without having to guess the right
number of trees.

**Calibration** — correcting probability *levels* without changing the *ranking*. Two
methods:
- **Isotonic** — fits a free-form monotone step function. Flexible but data-hungry.
- **Platt (sigmoid)** — fits a single sigmoid curve. Two parameters, safe on small data, but
  can only undo sigmoid-shaped distortions.

**Log-odds** — probabilities live in [0,1], which is awkward for adding things up. Log-odds
stretch that onto the whole number line: `log(p / (1−p))`. 0 = 50%, negative = less likely.
Gradient boosting and SHAP both work in log-odds.

**Sigmoid** — converts log-odds back to a probability. The inverse of the above.

**One-hot encoding** — turning a categorical column into one 0/1 column per category.
`loan_purpose` becomes four columns. Our 22 features become 31 columns after encoding.

**sklearn Pipeline** — chains transformation steps and a final model into one object, so
`fit` and `predict` run the identical sequence. Serializing the whole pipeline is what
makes training/serving skew structurally impossible here.

**joblib** — the library that serializes (saves) Python objects to disk. Our
`model.joblib` is 145 KB.

**Optuna** — a hyperparameter search library. Instead of trying every combination (grid
search) or random ones, it learns which regions of the search space look promising and
concentrates there.

**SHAP (SHapley Additive exPlanations)** — splits a single prediction into one number per
feature, showing how much each pushed the answer up or down.

The idea comes from cooperative game theory: four people earn a bonus together; each
person's fair share is their marginal contribution, averaged over **every possible order**
of joining the team. Swap "team" for "the applicant's feature values" and "bonus" for "the
gap between the average prediction and this one".

The key property is **additivity**:
```
prediction = base_value + Σ (SHAP value of each feature)
```
The parts are *guaranteed* to sum to the whole. §7 verifies this on real numbers.

**TreeSHAP vs KernelSHAP** — exact Shapley values need every subset of features (2²² here —
impossible). KernelSHAP approximates for any model, slowly. **TreeSHAP** exploits tree
structure to compute exact values in polynomial time. We use TreeSHAP.

**API / HTTP / JSON / status codes** — an API lets another program ask yours a question over
a network. It sends an HTTP **request** and gets one **response**. The data travels as
**JSON** text. The response carries a **status code**: `200` OK, `422` your input was
invalid, `500` we crashed. **The 422-vs-500 distinction matters a lot here**: 422 means the
API correctly rejected bad data; 500 means a bug.

**FastAPI / Pydantic / uvicorn** — uvicorn listens on the network port; FastAPI routes the
request to the right function; Pydantic validates the incoming JSON before your code runs.

**Docker** — packages code *and its entire environment* into one unit so it runs identically
anywhere. An **image** is the frozen recipe; a **container** is a running copy.

**CI (continuous integration)** — a robot that runs your tests automatically on every push,
so a broken change is caught before it reaches production.

---

# Part 3 — The data

## 3.1 The three tables

50,000 rows each, joined **one-to-one** on `cust_id`. One customer → one loan → one bureau
record.

| File | What it is | Columns |
|---|---|---|
| `customers.csv` | Who is applying | `cust_id, age, gender, marital_status, employment_status, income, number_of_dependants, residence_type, years_at_current_address, city, state, zipcode` |
| `loans.csv` | What they want, **and the target** | `loan_id, cust_id, loan_purpose, loan_type, sanction_amount, loan_amount, processing_fee, gst, net_disbursement, loan_tenure_months, principal_outstanding, bank_balance_at_application, disbursal_date, installment_start_dt, default` |
| `bureau_data.csv` | Their credit history | `cust_id, number_of_open_accounts, number_of_closed_accounts, total_loan_months, delinquent_months, total_dpd, enquiry_count, credit_utilization_ratio` |

**The join is asserted, not assumed.** `merge_raw` passes `validate="one_to_one"` and
pre-checks for duplicate `cust_id`s. A silent many-to-one join is a classic way to inflate
a dataset and every metric with it.

**Coverage:** dates span 2019-07-24 → 2024-07-22 (1,825 days). Income ₹0.1M–₹12.0M. Ages
18–70. Ten Indian cities across nine states. 60% M / 40% F. 64% self-employed.

## 3.2 The six data-quality problems

Each was **verified in code**, and every count is asserted in `tests/test_data.py` so the
documentation cannot drift from reality.

### 1. `loan_purpose` typo — 22 rows

22 rows spelled `Personaal`. **Evidence it is a typo, not a product:** both spellings
default at the same ~4.5% rate. Normalised to `Personal`.

### 2. `residence_type` nulls — 62 rows → `Unknown`

**The decision that matters here is what we did *not* do.** The obvious move is to impute
the mode. The mode is `Owned` — which is the **lowest-risk** category at 5.6% default vs
16.3% for `Rented`. Mode imputation would silently tell the model these 62 applicants are
the safest kind of borrower, on no evidence at all.

Instead they get an explicit `Unknown` level. One-hot encoding gives it a column and the
tree can split on it or ignore it. In a real application flow, "the applicant skipped this
field" is itself information.

At 0.12% of rows this cannot move any metric. **It was chosen for the reasoning, not the
score** — which is the honest thing to say if asked.

### 3. Eight all-zero rows → dropped

Eight rows have `sanction_amount`, `loan_amount`, `processing_fee`, `gst`,
`net_disbursement`, `income` **and** `bank_balance_at_application` all zero. These are
placeholders, not loans. Every ratio feature would be 0/0. Dropped and counted.

### 4. `gst` — the brief was incomplete, and checking paid off

The brief said `gst` is "roughly 9× `processing_fee`, not a plausible 18% GST on the fee".
True, but the real structure is more damning. Checking the arithmetic:

| Column | Exact rule | Holds for |
|---|---|---|
| `processing_fee` | **2%** of `loan_amount` | 49,985 / 49,992 |
| `gst` | **18%** of `loan_amount` (not of the fee) | 49,992 / 49,992 |
| `net_disbursement` | **80%** of `loan_amount` | 49,992 / 49,992 |

The 9× ratio is just 18/2. **All three are exact linear functions of `loan_amount` and
carry zero independent information.** Same action (drop), much stronger reason: redundancy,
not inconsistency.

### 5 & 6. Two problems that are actually one

The brief listed "7 rows fail the disbursement identity" and "`processing_fee` reaches 148%
of the loan" as separate issues. **They are the same 7 rows.**

Only `processing_fee` is corrupted. `gst` and `net_disbursement` are still correct on those
rows, so the true fee is recoverable **exactly** as `gst / 9` — and the repair is confirmed
independently by the identity `loan_amount − processing_fee − gst = net_disbursement`
going from **7 violations to 0**.

**Why repair rather than drop:** nothing is guessed (two independent identities force the
value), and 7 valid applications are kept. (`processing_fee` is post-approval and never
reaches the model anyway — this is bookkeeping hygiene so the EDA and identity checks are
honest.)

**Result: 49,992 of 50,000 rows survive.**

## 3.3 Where the signal actually is

**Default rate by credit utilisation** — monotone and very steep:

| Utilisation | Default rate |
|---|---|
| 0–10% | 1.12% |
| 40–50% | 2.06% |
| 70–80% | 11.47% |
| 90–100% | **48.56%** |

**By category:**

| Feature | Spread |
|---|---|
| `residence_type` | Owned 5.6% · Mortgage 9.4% · **Rented 16.3%** |
| `loan_purpose` | Auto 4.3% · Personal 4.5% · Education 9.8% · **Home 15.5%** |
| `loan_type` | Unsecured 4.5% · **Secured 10.8%** |
| `gender` | F 8.79% · M 8.47% — **indistinguishable** |
| `city` | 8.1%–9.1% across all ten — **flat, inside noise** |

Two of these are worth being ready to explain:

- **Secured defaults more than unsecured**, which looks backwards. It is a composition
  effect: home loans are secured and they default at 15.5%, dominating the secured book.
  The model sees `loan_purpose` too, so it can separate them.
- **`city` and `gender` carry nothing**, which is why dropping them costs no accuracy.

---

# Part 4 — The code, file by file

2,231 lines across `src/` and `api/`, plus 574 lines of tests.

## 4.1 `src/config.py` — 93 lines

Every constant another module might want to change. Nothing here is computed; it is all
declarations.

```python
RANDOM_SEED = 42          # used everywhere: splits, XGBoost, Optuna sampler
TEST_SIZE = 0.15
VAL_SIZE  = 0.15
OOT_START_DATE = "2023-07-23"   # reserves the final ~12 months
TARGET = "default"
```

**The two most contestable numbers in the entire project live here:**

```python
COST_FALSE_NEGATIVE = 0.60   # approve a defaulter → lose ~60% of principal (LGD)
COST_FALSE_POSITIVE = 0.06   # decline a good customer → lose ~6% interest margin
```

**Why this matters as an interview point:** when someone says "your 60% LGD is wrong", the
answer is *"probably — I don't have recovery data. It's a one-line change and the threshold
re-derives automatically. The ratio is what drives the result, not the absolute values."*
**Assumptions you cannot find are assumptions you cannot challenge.**

```python
PD_FLOOR   = 1e-4
PD_CEILING = 1.0 - 1e-4
```

Isotonic calibration's lowest step is *exactly* 0.0 and its highest *exactly* 1.0. A
reported PD of zero is indefensible — no lender believes an applicant can never default,
and no pricing formula can divide by it. This clamp is **monotone**, so no ranking metric
changes.

```python
SCORE_MIN, SCORE_MAX = 300, 900
RATING_BANDS = [(300,500,"Poor"), (500,650,"Average"), (650,750,"Good"), (750,901,"Excellent")]
```

## 4.2 `src/data.py` — 209 lines

Loads, merges, cleans. **Its distinguishing feature is that it counts itself.**

### `CleaningReport` — a dataclass that makes the docs testable

```python
@dataclass
class CleaningReport:
    rows_in: int = 0
    rows_out: int = 0
    loan_purpose_typos_fixed: int = 0
    residence_type_nulls_filled: int = 0
    zero_amount_rows_dropped: int = 0
    processing_fee_rows_repaired: int = 0
    identity_violations_before: int = 0
    identity_violations_after: int = 0
    duplicate_rows_dropped: int = 0
    notes: list[str] = field(default_factory=list)
```

Every number the README quotes — 22 typos, 62 nulls, 8 dropped, 7 repaired — comes out of
this object, gets written into `metadata.json`, and is asserted in `tests/test_data.py`.
**Documentation that disagrees with the code makes a test fail.**

`identity_violations_before` / `_after` is the elegant part: the repair **proves itself** in
the report, 7 → 0.

### `merge_raw()`

```python
for name, frame in (("customers", customers), ("loans", loans), ("bureau", bureau)):
    dupes = frame["cust_id"].duplicated().sum()
    if dupes:
        raise ValueError(f"{name} has {dupes} duplicate cust_id values; joins would fan out")

df = customers.merge(loans, on="cust_id", how="inner", validate="one_to_one")
df = df.merge(bureau, on="cust_id", how="inner", validate="one_to_one")
```

Inner joins are deliberate: a loan with no customer record cannot be scored, so carrying it
as a row of NaNs would be worse than dropping it. On this data all three tables cover the
same 50,000 customers, so nothing is lost.

### `clean()` — the six rules, in order

```python
df = df.drop_duplicates()                                    # rule 0
typo_mask = df["loan_purpose"].str.strip().str.lower() == "personaal"
df.loc[typo_mask, "loan_purpose"] = "Personal"               # rule 1
df["residence_type"] = df["residence_type"].fillna("Unknown") # rule 2
zero_mask = (df["loan_amount"] == 0) | (df["income"] == 0)
df = df.loc[~zero_mask].copy()                                # rule 3
```

Then the repair (rules 4–5):

```python
fee_ratio   = df["processing_fee"] / df["loan_amount"]
corrupt_fee = fee_ratio > 0.03                               # normal is exactly 0.02

identity_resid_before = df["loan_amount"] - df["processing_fee"] - df["gst"] - df["net_disbursement"]
report.identity_violations_before = int((identity_resid_before.abs() > 1.0).sum())   # → 7

df.loc[corrupt_fee, "processing_fee"] = df.loc[corrupt_fee, "gst"] / 9.0             # the repair

identity_resid_after = df["loan_amount"] - df["processing_fee"] - df["gst"] - df["net_disbursement"]
report.identity_violations_after = int((identity_resid_after.abs() > 1.0).sum())     # → 0
```

Finally, types are fixed and the boolean target becomes 0/1:

```python
df["disbursal_date"] = pd.to_datetime(df["disbursal_date"])
df[config.TARGET] = df[config.TARGET].astype(bool).astype(int)
```

## 4.3 `src/features.py` — 270 lines — **the most important file**

Three things live here and **nowhere else**.

### (a) The input contract

```python
RAW_NUMERIC_INPUTS     = [15 columns]
RAW_CATEGORICAL_INPUTS = [5 columns]
RAW_INPUTS = RAW_NUMERIC_INPUTS + RAW_CATEGORICAL_INPUTS    # 20 raw fields
```

`api/schemas.py` mirrors this, and `metadata.json` publishes it through `GET /model/info`.

### (b) The exclusions — a dict, not a list, so reasons travel

```python
EXCLUDED_COLUMNS = {
    "cust_id":   "identifier",
    "loan_id":   "identifier",
    "zipcode":   "no signal + fair-lending proxy risk",
    "city":      "no signal + fair-lending proxy risk",
    "state":     "no signal + fair-lending proxy risk",
    "gender":    "protected characteristic - excluded from features, retained for fairness auditing",
    "principal_outstanding": "post-disbursal (mid-repayment balance)",
    "processing_fee":        "post-approval, and exactly 2% of loan_amount",
    "gst":                   "post-approval, and exactly 18% of loan_amount",
    "net_disbursement":      "post-approval, and exactly 80% of loan_amount",
    "installment_start_dt":  "post-approval",
    "disbursal_date":        "used only to build the out-of-time split, not a feature",
}
```

**Why a dict matters:** the *reason* travels with the decision into `metadata.json` and out
through the API. Six months later nobody has to guess why `city` is missing.

Note `disbursal_date`: it is used to build the OOT split but **no month/quarter feature is
derived from it**. The default rate is flat across all 21 quarters (7.8–9.6%), so a date
feature would only encode noise — and it would be the first thing to drift in production.

### (c) `_safe_divide` — not boilerplate

```python
def _safe_divide(numerator, denominator, fill=0.0):
    denominator = denominator.replace(0, np.nan)
    result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan).fillna(fill)
```

**Why it exists:** `avg_dpd_per_delinquency = total_dpd / delinquent_months` is undefined
for the **19,993 borrowers (40%)** who have zero delinquent months. Unguarded that is `inf`,
XGBoost will happily split on it, and the API returns nonsense for real applicants.
`tests/test_features.py` asserts no infinities survive on the real data.

### `build_features()` — 20 raw fields → 22 features

```python
missing = [c for c in RAW_INPUTS if c not in df.columns]
if missing:
    raise ValueError(f"build_features is missing required raw columns: {missing}")
```

Fails loudly rather than producing a half-built matrix.

**Affordability block:**
```python
monthly_income = df["income"] / 12.0
emi = _safe_divide(df["loan_amount"], df["loan_tenure_months"])

out["loan_to_income"]           = _safe_divide(df["loan_amount"], df["income"])
out["emi_to_monthly_income"]    = _safe_divide(emi, monthly_income)
out["loan_to_sanction"]         = _safe_divide(df["loan_amount"], df["sanction_amount"])
out["balance_to_loan"]          = _safe_divide(df["bank_balance_at_application"], df["loan_amount"])
out["months_of_income_in_bank"] = _safe_divide(df["bank_balance_at_application"], monthly_income)
out["income_per_dependant"]     = df["income"] / (1.0 + df["number_of_dependants"].clip(lower=0))
```

**Behaviour block:**
```python
out["delinquency_ratio"]        = _safe_divide(df["delinquent_months"], df["total_loan_months"])
out["avg_dpd_per_delinquency"]  = _safe_divide(df["total_dpd"], df["delinquent_months"])
out["credit_utilization_ratio"] = df["credit_utilization_ratio"].astype(float)
```

**Credit-file-shape block:**
```python
total_accounts = df["number_of_open_accounts"] + df["number_of_closed_accounts"]
out["open_account_ratio"] = _safe_divide(df["number_of_open_accounts"], total_accounts)
out["total_accounts"]     = total_accounts.astype(float)
```

Then six pass-through columns, five categoricals stripped to clean strings, and finally:

```python
return out[FEATURE_COLUMNS]     # explicit column order, every time
```

**That last line is load-bearing.** Silent column reordering between training and serving is
an invisible source of wrong predictions.

### The credit intuition for every feature

| Feature | Why a lender cares |
|---|---|
| `loan_to_income` | The oldest underwriting ratio. 5× your income is stretched; 0.5× is not |
| `emi_to_monthly_income` | A crude FOIR/DTI. **Conceptually the most important** — what breaks a borrower is the monthly payment not fitting the monthly budget |
| `loan_to_sanction` | Drawing the full approved limit signals they needed every rupee |
| `balance_to_loan` | Liquidity cushion — savings absorb the first bad month |
| `months_of_income_in_bank` | The same cushion, scale-free, so high and low earners compare fairly |
| `income_per_dependant` | Same salary, four dependants ≠ same disposable income |
| `delinquency_ratio` | Share of history spent late. **Past repayment is the best predictor of future repayment**, normalised for history length |
| `avg_dpd_per_delinquency` | Separates "a few days late, often" from "90 days dark" |
| `credit_utilization_ratio` | Already leaning on credit to make ends meet |
| `open_account_ratio` | Accumulating obligations vs a record of closing them out |
| `total_accounts` | File depth — thin is hard to judge, very thick suggests credit hunger |
| `enquiry_count` | Shopping hard for credit is a distress signal |

### `build_features_with_leakage()`

Identical, plus `principal_outstanding` and `principal_outstanding_ratio`. **Used only by
the ablation in `train.py`, never by the production model.** Its existence is the evidence
that the leakage decision was tested rather than assumed.

## 4.4 `src/pipeline.py` — 174 lines

Builds the sklearn objects. Kept separate from `train.py` so that loading the model at
serving time does not drag Optuna, SHAP and matplotlib into the API process.

### The pipeline shape

```python
Pipeline([
    ("features", FunctionTransformer(build_features, validate=False, feature_names_out=...)),
    ("encode",   ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ("num", "passthrough" or StandardScaler(), numeric_features),
    ])),
    ("model",    XGBClassifier(...)),
])
```

- `FunctionTransformer` wraps our plain function so it becomes a pipeline step. **This is
  what puts feature engineering inside the serialized artifact.**
- `handle_unknown="ignore"` — an unseen category becomes an all-zero block instead of an
  exception. A second line of defence; the API rejects it with a 422 first.
- `scale_numeric` is **False for XGBoost** (trees do not care about scale) and **True for
  logistic regression** (which does).

### `fit_xgb_with_early_stopping()` — the one awkward piece

sklearn's `Pipeline.fit` cannot pass a *transformed* validation set to the final estimator,
so early stopping inside a pipeline is not directly supported. This works around it:

```python
Xt_train = encoder.fit_transform(feature_step.fit_transform(X_train))
Xt_val   = encoder.transform(feature_step.transform(X_val))

model.set_params(early_stopping_rounds=early_stopping_rounds)
model.fit(Xt_train, y_train, eval_set=[(Xt_val, y_val)], verbose=False)
return pipeline        # same object — its steps are now fitted
```

Fit the preprocessing on train, push validation through the **already-fitted** transformers
(so no validation information leaks into the preprocessing), then fit XGBoost with early
stopping. The steps are fitted in place, so what gets serialized is still one pipeline
taking raw fields.

**Be honest about this if asked:** it is the least elegant code in the repo, and it is the
price of wanting early stopping *and* a single serialized artifact. The parity test
(§4.9) is what makes it safe.

## 4.5 `src/evaluate.py` — 315 lines

**Pure functions, no state.** Every function takes `y_true` and `y_prob` and returns a
number or a DataFrame. That is why the same code scores the logistic baseline, raw XGBoost,
the calibrated model and the OOT holdout without a single branch.

### `ks_statistic()`

```python
fpr, tpr, _ = roc_curve(y_true, y_prob)
return float(np.max(tpr - fpr) * 100.0)
```

The largest gap between the two cumulative curves — three lines, because `roc_curve`
already computes exactly those curves.

### `cost_optimal_threshold()`

```python
for thr in np.unique(np.round(np.linspace(0.01, 0.95, 189), 4)):
    pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    cost = fn * cost_fn + fp * cost_fp        # 0.60 and 0.06 from config
    if cost < best["expected_cost"]:
        best = {"threshold": float(thr), "expected_cost": float(cost)}
```

A brute-force sweep of 189 candidate cut-offs, keeping the one with the lowest expected
cost. Simple and completely transparent — which matters when you have to defend the number.

### `decile_table()` — note the ranking trick

```python
frame["rank"]   = frame["p"].rank(method="first", ascending=False)
frame["decile"] = pd.qcut(frame["rank"], q=n_bins, labels=range(1, n_bins + 1)).astype(int)
```

**Why rank before binning:** tree models produce heavy ties in predicted probability.
`qcut` applied directly to the probabilities collapses bins and yields deciles of wildly
different sizes. Ranking first spreads ties across boundaries, so deciles come out equal
(750, 750, 750…).

### `population_stability_index()`

```python
edges = np.unique(np.percentile(expected, np.linspace(0, 100, n_bins + 1)))
if len(edges) < 3:
    return 0.0                      # near-constant feature: no meaningful bins
edges[0], edges[-1] = -np.inf, np.inf

exp_pct = np.clip(exp_counts / exp_counts.sum(), 1e-6, None)
act_pct = np.clip(act_counts / act_counts.sum(), 1e-6, None)
return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
```

Two details worth knowing:
- **Bin edges come from the *expected* (training) distribution.** That is the whole point —
  in production you freeze the training bins and watch new data move through them.
- **The `1e-6` floor is not cosmetic.** An empty bin sends `log(actual/expected)` to
  infinity and one unlucky bin poisons the entire PSI.

### `clip_pd()` and `pd_to_credit_score()`

```python
def clip_pd(prob):
    return np.clip(np.asarray(prob, dtype=float), config.PD_FLOOR, config.PD_CEILING)

def pd_to_credit_score(prob_default, score_min=300, score_max=900):
    prob = clip_pd(prob_default)
    return np.round(score_min + (1.0 - prob) * (score_max - score_min)).astype(int)
```

A **linear map on `1 − PD`**, chosen deliberately over the industry-standard log-odds
"points to double the odds" scaling: it is trivially explainable to a customer and a
regulator, and anyone can reproduce it by hand. It is strictly decreasing in PD, so the
ordering is identical to the model's — **this is a presentation layer, not extra
modelling**.

### `slice_metrics()`

Per-segment AUC and approval rate — the fairness table in the model card. It reports both
deliberately: **a model can be equally accurate on two groups and still approve them at
very different rates.**

## 4.6 `src/train.py` — 769 lines — the orchestrator

Run with `python -m src.train` (~157 seconds) or `--trials 5 --skip-shap` (~20 seconds).

About 155 lines of it is plotting, which is most of why it looks long.

### 4.6.1 `make_splits()` — two splits, two different questions

```python
X_train, X_temp, y_train, y_temp = train_test_split(
    df, y, test_size=0.30, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
```

→ **34,994 train / 7,499 validation / 7,499 test**, each holding the 8.59% default rate.

```python
cutoff   = pd.Timestamp("2023-07-23")
oot_mask = df["disbursal_date"] >= cutoff
oot_viable = span.days >= 365 and 0.05 <= oot_mask.mean() <= 0.45
```

→ **10,015 out-of-time rows.** The viability check is deliberate: an OOT split is only
meaningful if the data spans enough time and the holdout is a sensible size. The code
*checks* rather than assuming.

**Why two splits:** the random one answers "how well does it rank applicants it has never
seen?" The OOT one answers "if we had trained this a year ago, how would it have done
since?" Only the second is the question a risk team actually asks.

### 4.6.2 `compare_imbalance_strategies()` — measured, not assumed

| Strategy | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|
| Do nothing | 0.9889 | **0.8991** | **0.0239** |
| `scale_pos_weight = 10.63` | 0.9887 | 0.8990 | 0.0362 |
| Random undersampling (28,978 good loans discarded) | 0.9875 | 0.8896 | 0.0485 |

**What we shipped and why — this is a subtle point worth rehearsing.** Undersampling
clearly lost. "Do nothing" and `scale_pos_weight` **tied** on ranking.

We shipped `scale_pos_weight` because it makes the minority-class emphasis explicit, is the
standard defensible choice, and **its worse raw Brier is a *level* distortion, not a
*ranking* one — which calibration is designed to fix.** Final Brier after calibration:
0.0240.

If asked "but do-nothing tied you": *on this dataset the imbalance is mild enough not to
matter, and I'd rather ship the configuration that stays correct when the base rate shifts.*

### 4.6.3 `tune_xgboost()` — Optuna, 40 trials

```python
X = pd.concat([splits["X_train"], splits["X_val"]])     # tune on train+val
y = np.concatenate([splits["y_train"], splits["y_val"]])
```

**The test set is never touched in this function.**

```python
params = {
    "max_depth":        trial.suggest_int("max_depth", 3, 8),
    "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
    "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
    "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
    "reg_lambda":       trial.suggest_float("reg_lambda", 0.1, 20.0, log=True),
    "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
    "n_estimators":     trial.suggest_int("n_estimators", 200, 1200, step=100),
}
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
```

4-fold stratified CV with early stopping **inside each fold**, so the chosen
hyperparameters are not tuned to one lucky validation split.

**Best CV AUC 0.98900.** Winning parameters:

| Parameter | Chosen | What it does | Higher pushes toward |
|---|---|---|---|
| `max_depth` | **3** | Levels per tree = interaction order | Variance (overfit) |
| `learning_rate` | **0.061** | How much of each correction is applied | Variance |
| `n_estimators` | 1200 (early-stopped) | Number of boosting rounds | Variance |
| `subsample` | **0.705** | Fraction of rows per tree | *Lower* = less overfit |
| `colsample_bytree` | **0.997** | Fraction of columns per tree | *Lower* = less overfit |
| `min_child_weight` | **13** | Min weighted rows in a leaf | Bias (conservative) |
| `gamma` | **3.93** | Min loss reduction to make a split | Bias |
| `reg_lambda` | **0.535** | L2 penalty on leaf weights | Bias |
| `reg_alpha` | **0.029** | L1 penalty (can zero weights out) | Bias |

**How to read this result:** shallow trees (depth 3), heavy pruning (gamma 3.93,
min_child_weight 13), strong row subsampling (0.705). That is a deliberately **high-bias,
low-variance** configuration — and it makes sense, because `credit_utilization_ratio` alone
gives 0.877 AUC and two features give 0.940. There are no deep interactions to find, so the
real risk was overfitting and the search correctly regularised hard.

**Why `n_estimators=1200` is not really 1200:** early stopping halts training when
validation AUC stops improving for 50 rounds. It is a ceiling, not a budget — searching it
as a hard count *and* early-stopping would be tuning the same thing twice.

### 4.6.4 `compare_calibration()` — and the methodological trap it avoids

**The problem calibration solves.** XGBoost is trained to rank, and `scale_pos_weight`
deliberately distorts the output *level* on top of that. The raw output tells you who is
riskier but not anyone's actual chance of default. That matters the instant you price off
the number: expected loss = PD × LGD × exposure.

**The trap, and how the code avoids it.** A calibrator cannot be judged on the fold it was
fitted to — and it must not be judged on test, because that is selecting a model on the
holdout. So:

```python
fit_idx, select_idx = train_test_split(np.arange(len(X_val)), test_size=0.5,
                                       stratify=y_val, random_state=42)
# fit BOTH calibrators on one half, score them on the OTHER half
for label, method in (("isotonic", "isotonic"), ("sigmoid (Platt)", "sigmoid")):
    trial = CalibratedClassifierCV(fitted_pipeline, method=method, cv="prefit")
    trial.fit(X_fit, y_fit)
    selection[label] = {"brier_holdout": ..., "ece_holdout": ...}

winner = min(selection, key=lambda k: selection[k]["brier_holdout"])

# only NOW refit the winner on all of validation
final = CalibratedClassifierCV(fitted_pipeline, method=method, cv="prefit")
final.fit(X_val, y_val)
```

Selection on the held-out half: **isotonic 0.02281 vs Platt 0.02516** → isotonic wins.

Reported on test *after* the choice was locked in:

| Method | ROC-AUC | Brier | ECE |
|---|---|---|---|
| None (raw XGBoost) | 0.9888 | 0.0381 | 0.0479 |
| **Isotonic** ✅ | 0.9885 | **0.0240** | **0.0063** |
| Sigmoid (Platt) | 0.9888 | 0.0262 | 0.0159 |

**Isotonic cut calibration error 7.6×.** It costs a little PR-AUC (0.9009 → 0.8862) because
it is a step function that ties nearby scores — a fair trade when the number drives pricing.

### 4.6.5 `expected_calibration_error()`

```python
frame["bin"] = pd.qcut(frame["p"].rank(method="first"), n_bins, labels=False)
grouped = frame.groupby("bin").agg(obs=("y","mean"), pred=("p","mean"), n=("y","size"))
return float((grouped["n"] * (grouped["obs"] - grouped["pred"]).abs()).sum() / grouped["n"].sum())
```

Bin, compare what the model said against what happened, average weighted by bin size.

### 4.6.6 `leakage_ablation()` — the evidence

```python
clean_pipe = make_xgb_pipeline(params, scale_pos_weight=spw)          # 22 features
leaky_pipe = make_xgb_pipeline(params, scale_pos_weight=spw,
                               numeric_features=leaky_numeric,
                               builder=build_features_with_leakage)    # + principal_outstanding
```

Identical hyperparameters, identical splits, one extra feature.

| Feature set | Test ROC-AUC |
|---|---|
| Application-time only (production) | 0.98882 |
| + `principal_outstanding` | 0.98868 |
| **Inflation** | **−0.00014** |

**It inflated nothing — AUC went marginally *down*.** Its raw correlation with `default` is
−0.011.

**How to frame this, and it is a strong frame:** *"I expected inflation and found none.
That's a negative result and I'm reporting it rather than quietly dropping the experiment.
The feature is still excluded, because a feature you can't obtain at scoring time is
unusable regardless of what it does to the metric — if it had helped, that would have been
a reason to be more careful, not less."*

### 4.6.7 The out-of-time evaluation — a subtle correctness detail

```python
oot_fit, oot_val, oot_fit_y, oot_val_y = train_test_split(
    oot_tr, oot_y, test_size=0.15, stratify=oot_y, random_state=42)
```

The OOT model carves its validation fold **out of the in-time data**. Otherwise early
stopping would peek at post-cutoff rows and the OOT result would be contaminated — which
would defeat the entire point of having an OOT split.

Result: **0.9855 AUC**, versus 0.9885 in-time. It holds up, which is the third piece of
leakage evidence.

### 4.6.8 Threshold, chosen on validation

```python
cal_val_p  = calibrated.predict_proba(X_val)[:, 1]
thr_search = ev.cost_optimal_threshold(y_val, cal_val_p)    # → 0.080
threshold  = thr_search["threshold"]
test_at_thr = ev.threshold_metrics(y_test, cal_test_p, threshold)
```

**Chosen on validation, applied unchanged to test.** Choosing it on test would be tuning on
the holdout.

| Threshold | Recall | Precision | Approval rate | Bad rate approved | Defaults missed |
|---|---|---|---|---|---|
| **0.080** (cost-optimal) | **96.27%** | 58.16% | 85.78% | **0.37%** | **24** |
| 0.500 (naive) | 73.45% | 85.07% | 92.59% | 2.46% | 171 |

The naive threshold approves 6.8 points more of the book but **misses 171 defaults instead
of 24**.

### 4.6.9 The prediction fixture — the last thing `train.py` writes

```python
fixture = {
    "payload": {c: ... for c in RAW_INPUTS},
    "expected_probability": float(ev.clip_pd(calibrated.predict_proba(fixture_row)[0, 1])),
    "expected_credit_score": int(...),
}
```

One real test row and the probability the freshly-fitted model gave it. `tests/test_api.py`
asserts the API reproduces that exact number after a joblib round-trip, through Pydantic
coercion, in a fresh process.

**This is the single highest-value line in the file** — it turns "we prevent training/serving
skew" from a claim into a test.

## 4.7 `api/schemas.py` — 161 lines

### Literal types, not strings

```python
MaritalStatus    = Literal["Married", "Single"]
EmploymentStatus = Literal["Salaried", "Self-Employed"]
ResidenceType    = Literal["Owned", "Mortgage", "Rented", "Unknown"]
LoanPurpose      = Literal["Personal", "Home", "Auto", "Education"]
LoanType         = Literal["Secured", "Unsecured"]
```

Three benefits at once: a typo is a 422; `/docs` renders a dropdown; the allowed values
appear in the OpenAPI schema for callers.

**Why this matters more than it looks:** without it, `"self-employed"` (wrong case) would be
one-hot encoded to **all zeros** — an applicant with no employment status at all — and the
model would return a confident, meaningless score. **Silent wrongness is worse than an
error.**

### Realistic bounds

```python
age:    Annotated[int,   Field(ge=18, le=100)]
income: Annotated[float, Field(gt=0, le=1e9)]
credit_utilization_ratio: Annotated[float, Field(ge=0, le=100)]
loan_tenure_months:       Annotated[int,   Field(ge=1, le=480)]
```

`ge` = greater-or-equal, `le` = less-or-equal, `gt` = strictly greater. The bounds are
**deliberately realistic, not merely type-correct**: `age` caps at 100 because 130 is a bug,
and `income` uses `gt=0` rather than `ge=0` because zero income breaks every ratio feature
— exactly the 8 rows dropped in cleaning.

### Cross-field rules

```python
@model_validator(mode="after")
def check_internal_consistency(self):
    if self.loan_amount > self.sanction_amount:
        raise ValueError("loan_amount cannot exceed sanction_amount")
    if self.delinquent_months > self.total_loan_months:
        raise ValueError("delinquent_months cannot exceed total_loan_months")
    if self.number_of_open_accounts + self.number_of_closed_accounts == 0:
        raise ValueError("applicant must have at least one open or closed credit account")
    if self.delinquent_months == 0 and self.total_dpd > 0:
        raise ValueError("total_dpd must be 0 when delinquent_months is 0")
    return self
```

`mode="after"` runs once every individual field has passed. These are rules **no
single-field bound can express** — each involves two fields.

They catch upstream bugs that would otherwise produce a plausible-looking score from
impossible data. Someone late 40 months in a 12-month history is a broken join, and a
scorecard should refuse rather than guess.

**Note what is absent from the whole file:** no `principal_outstanding`, no
`net_disbursement`, no `gst`. **The API physically cannot accept post-disbursal data**, so
the leakage decision is enforced at the network boundary, not merely in training code.

## 4.8 `api/main.py` — 233 lines

### Lifespan — load once, fail loudly

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not config.MODEL_PATH.exists():
        raise RuntimeError(f"Model artifact not found at {config.MODEL_PATH}. "
                           "Run `python -m src.train` before starting the API.")
    ARTIFACTS["model"] = joblib.load(config.MODEL_PATH)
    ARTIFACTS["metadata"] = json.loads(config.METADATA_PATH.read_text())
    ARTIFACTS["threshold"] = float(ARTIFACTS["metadata"]["threshold"]["value"])
    yield                      # ← the app runs here for its whole life
```

**The measured reason for loading here:**

| | Time |
|---|---|
| `joblib.load()`, cold | **534 ms** |
| `joblib.load()`, warm | 2.6 ms |
| A whole `/predict` request | **3.62 ms** |

Loading inside the handler would add 2.6–534 ms to *every call* — best case doubling
latency, worst case 150× slower.

**Failing loudly is deliberate:** a container that can never serve a valid request must
never report itself healthy, or the orchestrator keeps routing traffic to it.
`tests/test_startup.py` asserts this.

### Middleware — observability

```python
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    started = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - started) * 1000
    logger.info(json.dumps({"event": "request", "request_id": request_id,
                            "path": request.url.path, "status": response.status_code,
                            "latency_ms": round(latency_ms, 2)}))
    response.headers["X-Request-ID"] = request_id
    return response
```

Code before `call_next` runs on the way in, after it on the way out. **JSON rather than
prose** so a log aggregator can chart p99 latency and error rate directly.

### `_score()` — four decisions in six lines

```python
def _score(applicants: list[ApplicantRequest]) -> list[PredictionResponse]:
    frame = pd.DataFrame([a.model_dump() for a in applicants])
    frame = frame[RAW_INPUTS]                                    # 2
    probabilities = clip_pd(model.predict_proba(frame)[:, 1])    # 3
    scores  = pd_to_credit_score(probabilities)
    ratings = score_to_rating(scores)
    return [PredictionResponse(...) for p, s, r in zip(probabilities, scores, ratings, strict=True)]
```

1. **It always takes a list.** `/predict` calls it with one applicant. One code path means
   single and batch can never diverge — asserted by a test.
2. **`frame[RAW_INPUTS]`** forces column order explicitly rather than trusting dict ordering.
3. **`clip_pd`** keeps reported PDs off exactly 0 and 1.
4. **One `predict_proba` for the whole batch** — why 1,000 applicants take 12.6 ms total
   rather than 1,000 × 3.6 ms.

**Where is the feature engineering?** Nowhere in this file. It travels *inside*
`model.joblib`, which unpickles `src.features.build_features` at load time. **The API cannot
compute a feature differently from training because it does not compute features at all.**

### The endpoints

| Endpoint | Returns |
|---|---|
| `POST /predict` | PD, credit score, rating, decision, threshold, model version |
| `POST /predict/batch` | Up to 1,000 applicants; one bad applicant rejects the whole batch |
| `GET /health` | `status`, `model_loaded`, `model_version`, `uptime_seconds` |
| `GET /model/info` | Version, training date, feature list, metrics, threshold, ablation result |
| `GET /docs` | Auto-generated interactive documentation |

## 4.9 `tests/` — 65 tests, 574 lines, ~1.7 seconds

Grouped by **what breaks if they fail**:

| File | Protects against |
|---|---|
| `test_data.py` | Cleaning silently changing — every count in the docs is asserted |
| `test_features.py` | Leakage reintroduction, infinities, non-determinism, column reordering |
| `test_evaluate.py` | Metric bugs — checked against known answers (perfect separation → KS 100; PSI against itself → 0) |
| `test_api.py` | Contract drift, bad input returning 500 instead of 422, **and training/serving skew** |
| `test_startup.py` | The app booting without a model and serving errors |

### The three most valuable tests

**1. The parity test** — the most important test in the suite:
```python
def test_prediction_matches_what_train_py_recorded(client, valid_payload, metadata):
    expected = metadata["prediction_fixture"]["expected_probability"]
    body = client.post("/predict", json=valid_payload).json()
    assert body["probability_of_default"] == pytest.approx(expected, abs=1e-5)
```

**2. The leakage guard** — makes the decision permanent:
```python
def test_no_post_disbursal_column_can_reach_the_model():
    post_disbursal = {"principal_outstanding", "processing_fee", "gst",
                      "net_disbursement", "installment_start_dt"}
    assert post_disbursal.isdisjoint(FEATURE_COLUMNS)
    assert post_disbursal <= set(EXCLUDED_COLUMNS)
```
A future contributor cannot reintroduce leakage without a test going red.

**3. Row-independence** — so batching is a performance optimisation, never a semantic change:
```python
def test_row_order_does_not_change_any_row_result():
    batch = build_features(frame)
    for i in range(len(frame)):
        single = build_features(frame.iloc[[i]]).reset_index(drop=True)
        pd.testing.assert_frame_equal(single, batch.iloc[[i]].reset_index(drop=True))
```

**`conftest.py` skips rather than fails** the API tests when no artifact exists, with a
message saying to run `python -m src.train`. A test that fails because you have not trained
yet teaches people to ignore red suites.

---

# Part 5 — The artifacts

## 5.1 `models/model.joblib` — 145 KB

**One object containing four stages:**

```
raw application fields (20)
  → FunctionTransformer(build_features)   → 22 features
  → ColumnTransformer(OneHot + passthrough) → 31 columns
  → XGBClassifier                          → raw probability
  → CalibratedClassifierCV (isotonic)      → calibrated probability
```

**Why serialize the whole pipeline rather than just the booster.** If the API reimplemented
feature engineering, the two implementations would drift the first time someone changed a
ratio. Because `build_features` is *inside* the artifact, the API physically cannot compute
a feature differently.

**This is the answer to "how do you prevent training/serving skew?"** — it is a structural
guarantee, not a process. It is also why `src/` must be copied into the Docker image: the
pickle imports `src.features` when it loads.

## 5.2 `models/metadata.json` — 16 KB

Everything needed to audit a model you did not train. Key fields:

| Field | Contents |
|---|---|
| `model_version`, `trained_at` | 1.0.0, ISO timestamp |
| `algorithm` | "XGBClassifier + isotonic calibration, wrapped in a sklearn Pipeline" |
| `raw_input_columns`, `feature_columns` | The contract, published via `GET /model/info` |
| `excluded_columns` | Every exclusion **with its reason** |
| `best_params`, `best_iteration`, `n_optuna_trials` | Full tuning record |
| `cleaning_report` | All nine counts |
| `metrics` | Baseline, raw XGB, calibrated, out-of-time |
| `threshold` | Value, cost assumptions, metrics at it *and* at 0.5 |
| `leakage_ablation` | Both AUCs and the delta |
| `calibration_comparison` | All three methods with the winner flagged |
| `slice_metrics` | Six fairness slices |
| `prediction_fixture` | The parity-test payload and expected probability |

**Why it matters:** an auditor can query a *running* instance via `GET /model/info` and see
the version, feature list, metrics and leakage result without access to the training code.

## 5.3 `reports/`

**11 figures:** ROC/PR curves, KS curve, reliability (calibration) curve, decile rank
ordering, score distribution, cost-vs-threshold, SHAP global importance, SHAP beeswarm, and
three individual SHAP force plots.

**14 tables (CSV):** decile tables (test and OOT), PSI reports, SHAP importances, the full
Optuna trial history, imbalance and calibration comparisons, six fairness slices.

---

# Part 6 — Infrastructure

## 6.1 `Dockerfile` — two-stage build

**Stage 1 (builder):** installs `build-essential` (a C toolchain some wheels need) and the
Python packages into `/install`.

```dockerfile
COPY requirements-api.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-api.txt
```

**This is the layer-caching decision — the most commonly asked Docker question.** Docker
caches each step and reuses it if that step's inputs have not changed. Requirements change
rarely; source changes constantly. By copying *only* the requirements file and installing
**before** copying source, an edit to `main.py` reuses the cached dependency layer — a
rebuild takes seconds instead of minutes.

**Stage 2 (runtime):** starts from a **clean** `python:3.11-slim`, so the compilers never
ship.

```dockerfile
RUN apt-get install -y --no-install-recommends libgomp1 curl
RUN useradd --create-home --uid 10001 appuser
COPY --from=builder /install /usr/local
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser api/ ./api/
COPY --chown=appuser:appuser models/ ./models/
USER appuser
```

- `libgomp1` is XGBoost's OpenMP runtime. Without it `import xgboost` fails with a `dlopen`
  error that says nothing about a missing package.
- **Non-root** `uid 10001` — if compromised, the process should not own the filesystem.
- **`src/` must be copied** because the pickled pipeline imports `src.features` at load.
  Note what is *absent*: `data/`, `tests/`, `notebooks/`, `reports/`.

```dockerfile
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/health" || exit 1
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --workers ${WEB_CONCURRENCY}
```

- **Shell form** (not the list form) so `${PORT}` expands — Render assigns it at runtime.
- **`--host 0.0.0.0`** means listen on all interfaces. The default `127.0.0.1` accepts only
  connections from *inside* the container — a classic first-deployment bug.

**Verified:** 585 MB, healthy in 10 seconds, `uid=10001(appuser)`, two startup log lines
(one per worker), `/predict` returns **0.357143 — identical to local**.

## 6.2 `requirements-api.txt` vs `requirements.txt`

The serving set excludes Optuna, SHAP, matplotlib and seaborn — roughly half the image and
much less code on a public service. **The modelling versions are pinned identically to
training**, because a joblib artifact is not guaranteed to load across sklearn versions.

Verified: the API boots and serves correctly from a venv containing only
`requirements-api.txt`.

## 6.3 `.github/workflows/ci.yml`

Two jobs. **Job 1** lints with ruff, trains a 5-trial model, runs all 65 tests. **Job 2**
(`needs: test`) builds the Docker image, boots it, and asserts `/health` reports
`"model_loaded": true` and `/predict` returns a probability.

```yaml
runs-on: ubuntu-latest
```

**Quietly valuable:** CI runs Linux/x86 while development was on an arm64 Mac, so a green
build proves the pinned dependencies and the artifact work on the **deployment
architecture**, not just the machine that produced them.

Why train in CI when `models/` is committed? Because it **exercises the whole training path
on every push** — that is how you find out a data or feature change broke training before it
silently ships a stale model.

## 6.4 `render.yaml` and the live service

```yaml
services:
  - type: web
    runtime: docker
    plan: free
    healthCheckPath: /health
    autoDeploy: true
```

`autoDeploy: true` means Render rebuilds on **every push to main** — including docs-only
commits. Two consequences: retraining alone changes nothing in production until you commit
and push the new `model.joblib`; and nothing drifts on its own.

**Live:** https://credit-risk-api-1dbj.onrender.com/docs

| | p50 | p95 |
|---|---|---|
| In-process | 3.62 ms | 3.83 ms |
| Live on Render | 388 ms | 806 ms |

**Be precise about the gap: it is network and free-tier platform, not the model.** TLS
handshake, Render's router, shared CPU in another region. Cold start after 15 min idle:
30–50 s — open the URL a minute before an interview demo.

## 6.5 Reproducibility

Verified: a fresh venv, `pip install -r requirements.txt`, `python -m src.train` →
**all 54 recorded values bit-identical** to the committed artifacts, 65 tests pass.

---

# Part 7 — One request, end to end

Applicant **C26424** from the test set, followed from CSV row to JSON response. They are a
good example because **the model declined them and they did in fact default.**

## 7.1 Their raw record

`cust_id: C26424`, loan disbursed 2022-03-15, Jaipur:

| Field | Value | Field | Value |
|---|---|---|---|
| age | 24 | loan_purpose | Education |
| income | ₹380,000 | loan_type | Secured |
| dependants | 1 | sanction_amount | ₹796,000 |
| marital_status | Single | loan_amount | ₹683,000 |
| employment_status | Salaried | loan_tenure_months | 51 |
| residence_type | Owned | bank_balance | ₹113,045 |
| years_at_address | 17 | open / closed accounts | 1 / 1 |
| **credit_utilization_ratio** | **85%** | total_loan_months | 60 |
| **delinquent_months** | **9** | total_dpd | 40 |
| enquiry_count | 1 | **actual outcome** | **defaulted** |

Not shown to the model: their gender (F), city, and `zipcode` — all deliberately excluded.
Nor `principal_outstanding`, which would not exist at decision time.

## 7.2 Their engineered features

| Feature | Value | What it says |
|---|---|---|
| `loan_to_income` | **1.80** | Borrowing 1.8× annual income |
| `emi_to_monthly_income` | **0.42** | 42% of monthly income to this instalment. Tight |
| `loan_to_sanction` | 0.86 | Drew most of the approved limit |
| `balance_to_loan` | 0.17 | Savings cover a sixth of the loan |
| `months_of_income_in_bank` | 3.57 | ~3.5 months of cushion |
| `income_per_dependant` | 190,000 | ₹380k supporting two people |
| `delinquency_ratio` | **0.15** | Late in 15% of all months of credit history |
| `avg_dpd_per_delinquency` | 4.44 | ~4 days late when late — chronic, not catastrophic |
| `credit_utilization_ratio` | **85** | Nearly maxed out — the dominant signal |
| `open_account_ratio` | 0.50 | One open, one closed |
| `total_accounts` | 2 | A thin credit file |

## 7.3 SHAP — additivity verified

Base value (log-odds) = **0.3838**. Top contributions:

| Feature | Their value | SHAP | Reading |
|---|---|---|---|
| `credit_utilization_ratio` | 85 | **+1.9029** | The single biggest reason they look risky |
| `residence_type_Owned` | 1 | **−1.0528** | They own their home — the strongest point in their favour |
| `enquiry_count` | 1 | **−0.9503** | Not shopping desperately for credit |
| `residence_type_Rented` | 0 | −0.6038 | Not renting, which also helps |
| `loan_tenure_months` | 51 | +0.4995 | Long loan = longer exposure |
| `age` | 24 | +0.3180 | Young, thin track record |
| `income_per_dependant` | 190,000 | +0.2551 | Supporting someone on modest income |
| `loan_purpose_Education` | 1 | +0.2543 | Education loans default more than auto/personal |

Now add them up:

```
base value                    =  0.3838
sum of ALL SHAP values        = +0.7292
                                -------
total (log-odds)              =  1.1130
sigmoid(1.1130)               =  0.752681   ← converts log-odds to probability
what the model actually says  =  0.752682
```

**The parts sum to the whole, to six decimal places.** That is additivity, on our data.

## 7.4 The 13 steps

1. **The command.** `curl -X POST .../predict -H 'Content-Type: application/json' -d '{...}'`
2. **DNS and TLS.** Hostname resolves; encrypted connection established. *(Most of the 388 ms.)*
3. **Render's router** forwards to the container.
4. **uvicorn** — one of two workers — accepts and translates HTTP into an ASGI message.
5. **Middleware** stamps request id `70703ddf`, starts a timer.
6. **Routing.** FastAPI matches `POST /predict` → the `predict` function.
7. **Pydantic validates.** Age 24 ✓, utilisation 85 ✓, income 380000 ✓, purpose allowed ✓.
   Cross-field: 683000 ≤ 796000 ✓, 9 ≤ 60 ✓, 1+1 > 0 ✓, dpd 40 with 9 delinquent months ✓.
   **Any failure stops here with a 422 and the model is never reached.**
8. **`_score()`** builds a one-row DataFrame ordered by `RAW_INPUTS`.
9. **Inside the pipeline:** `build_features` → 22 features → one-hot → 31 columns →
   612 trees vote → **raw output 0.7527** → isotonic calibration → **0.3571**.
10. **Score mapping.** `300 + (1 − 0.3571) × 600` = **686** → "Good" band.
11. **Decision.** 0.3571 ≥ 0.08 → **decline**.
12. **Response:**
    ```json
    {"probability_of_default":0.357143,"credit_score":686,"rating":"Good",
     "decision":"decline","threshold":0.08,"model_version":"1.0.0"}
    ```
13. **Log line:**
    ```json
    {"event":"request","request_id":"70703ddf","method":"POST","path":"/predict","status":200,"latency_ms":0.66}
    ```

## 7.5 The three things this one applicant teaches

**1. Raw 0.75 vs calibrated 0.36 is not a bug — it is the whole reason calibration exists.**
The raw model over-states risk because `scale_pos_weight` deliberately distorts the output
level. Isotonic corrects it.

**2. "Good" and "decline" together is not a contradiction.** The rating places them against
the whole population (686 really is better than most); the decision compares their 35.7% PD
against the 8% cut-off.

**3. At the naive 0.5 threshold they would have been APPROVED** — 0.357 < 0.5 — **and they
defaulted.** One applicant carrying the entire threshold argument.

## 7.6 Turning SHAP into an adverse-action sentence

Their largest positive SHAP is `credit_utilization_ratio` at +1.90. Translated into a
lawful, specific reason:

> *"Your credit utilisation is 85%, compared with an average of 37% among the applicants we
> approve. You have also been late on payments in 9 of the last 60 months."*

Specific, quantified, actionable, and traceable to a number in the model rather than to
someone's judgement.

---

# Part 8 — Every number and where it comes from

## 8.1 Data

| Number | Meaning | Source |
|---|---|---|
| 50,000 × 3 | Raw rows per table | `data/raw/` |
| **49,992** | Rows after cleaning | `cleaning_report.rows_out` |
| **8.594%** | Base default rate | `dataset.base_default_rate` |
| 34,994 / 7,499 / 7,499 | Train / validation / test | `make_splits()` |
| 10,015 | Out-of-time test rows | `dataset.oot_test_rows` |
| 22 → 7 → 8 → 62 | Typos fixed, fees repaired, rows dropped, nulls categorised | `cleaning_report` |
| 7 → 0 | Identity violations before → after repair | `cleaning_report` |

## 8.2 Model performance (test set)

| Number | Meaning |
|---|---|
| **0.9885** | ROC-AUC, calibrated model |
| 0.9834 | ROC-AUC, logistic baseline |
| 0.9855 | ROC-AUC, out-of-time |
| **0.8862** | PR-AUC |
| 0.9771 | Gini |
| **90.30** | KS |
| **0.0240** | Brier (from 0.0381 raw) |
| **0.0063** | ECE (from 0.0479 raw) — **7.6× improvement** |
| **−0.00014** | Leakage ablation AUC delta |
| 0.98900 | Best Optuna CV AUC (40 trials) |

## 8.3 At the threshold

| Number | Meaning |
|---|---|
| **0.080** | Cost-optimal threshold (chosen on validation) |
| 60% / 6% | Assumed FN / FP cost — a **10:1** ratio |
| **96.27%** | Recall (defaults caught) |
| 58.16% | Precision |
| 0.7251 | F1 |
| 85.78% | Approval rate |
| **0.37%** | Bad rate in approved book (from 8.59%) |
| 24 vs 171 | Defaults missed at 0.080 vs 0.500 |

## 8.4 Serving

| Number | Meaning |
|---|---|
| 145 KB | `model.joblib` |
| 534 ms / 2.6 ms | Cold / warm model load |
| **3.62 / 3.83 / 4.32 ms** | p50 / p95 / p99 in-process |
| 388 / 806 ms | p50 / p95 live on Render |
| 12.6 ms | 1,000-applicant batch (0.013 ms each) |
| 585 MB | Docker image |
| 65 | Tests, all passing |

## 8.5 The caveat numbers

| Number | Why it matters |
|---|---|
| **0.70–0.80** | Realistic production AUC. Ours is 0.9885 |
| **30–40** | Realistic production KS. Ours is 90.3 |
| **0 of 8,150** | Low-utilisation, never-delinquent borrowers who defaulted. **Real people do not behave like this** |
| 0.877 | AUC from `credit_utilization_ratio` **alone** |
| 94.5% | Share of test book at Excellent or Poor — the score is effectively bimodal |

---

## Closing: the five things to say unprompted

1. **"The AUC is 0.99, which isn't credible for credit risk."** Then explain how you ruled
   out leakage three ways and why the data is synthetic. **Say this first, always.**
2. **"I excluded post-disbursal columns, then measured what that cost."** −0.0001. A
   negative result, reported.
3. **"Calibration matters because we price off the number."** Raw 0.75 → calibrated 0.36 on
   a real applicant.
4. **"The threshold comes from a cost ratio, not 0.5."** 171 defaults missed versus 24.
5. **"Feature code is written once and serialized inside the model."** A test asserts the
   API reproduces training's exact probability.

Every one of those is true, verifiable in the repo, and demonstrates judgement rather than
just tooling.
