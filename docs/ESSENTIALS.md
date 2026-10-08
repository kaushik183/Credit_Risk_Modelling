# Credit Default Risk Model — Essentials

Everything you need to explain this project, in one short read.

---

## 1. What this is

A model that looks at a loan application and answers: **will this person pay the money
back?**

It outputs three things:

- a **probability of default** (e.g. 0.357 = a 35.7% chance they stop paying)
- a **credit score** from 300 to 900
- an **approve or decline** recommendation

It learned from 50,000 past loans. 8.594% of those loans went bad.

It runs as a live web service that any other system can call:
[credit-risk-api-1dbj.onrender.com](https://credit-risk-api-1dbj.onrender.com/docs)

---

## 2. The four ideas behind every decision

If you remember only this page, you can reconstruct the rest.

**1. Only use what a lender knows when deciding.**
Some columns in the data only exist *after* the loan is paid out — like how much is still
owed halfway through. Using them would make the model useless in real life, because at
decision time that information does not exist yet. This is called **leakage**, and
avoiding it was the most important design choice.

**2. Ranking and probability are two different jobs.**
XGBoost is good at putting risky people above safe people. It is *not* good at saying "this
person has a 12% chance". If you want to price a loan off the number, you have to fix the
number separately. That fix is called **calibration**.

**3. The approve/decline line comes from money, not from 0.5.**
Approving someone who defaults costs about **10 times** more than turning away someone who
would have paid. So the cut-off sits far below the obvious 0.5.

**4. Feature code gets written once.**
If the training code and the live service each calculate features separately, they will
eventually disagree, and the model will quietly return wrong answers. Ours is one function,
saved *inside* the model file.

---

## 3. The words you need

| Word | What it means |
|---|---|
| **Default** | The borrower stopped paying |
| **Delinquent** | Behind on payments, but not yet given up on |
| **DPD (days past due)** | How many days late a payment was |
| **Credit utilisation** | How much of your available credit you are already using. 85% = nearly maxed out |
| **EMI** | The fixed monthly payment on a loan |
| **LGD (loss given default)** | If they default, how much of the money you actually lose. We assume 60% |
| **PD (probability of default)** | The number this model outputs |
| **Scorecard** | The banking word for a credit risk model |
| **Leakage** | Accidentally using information you would not have in real life |
| **Calibration** | Fixing probabilities so "10%" really means 10 out of 100 |
| **Adverse-action notice** | The letter legally required to tell a rejected applicant *why* |

---

## 4. The data

Three spreadsheets, 50,000 rows each, joined on customer ID:

| File | What it holds |
|---|---|
| `customers.csv` | Age, income, job, where they live, dependants |
| `loans.csv` | What they want to borrow, for what, over how long — **and whether they defaulted** |
| `bureau_data.csv` | Their credit history: accounts, late payments, credit utilisation |

### Six problems found in the data

| # | Problem | What we did |
|---|---|---|
| 1 | 22 rows spell "Personal" as **"Personaal"** | Fixed the typo. Both spellings default at the same rate, so it is a spelling mistake, not a different product |
| 2 | 62 rows have **no residence type** | Gave them their own "Unknown" category |
| 3 | 8 rows have **every money column = 0** | Deleted them. They are not real loans |
| 4 | `gst` column looked wrong | Deleted it — see below |
| 5 | 7 rows **fail an accounting check** | Repaired them — see below |
| 6 | 7 rows have a **fee of 148% of the loan** | The same 7 rows as #5 |

**49,992 of 50,000 rows survived.**

### Three of those are worth explaining properly

**Why we did NOT fill in the missing residence types with the most common value.**
The most common value is "Owned". Owners are the *safest* group — they default at 5.6%,
versus 16.3% for renters. So filling in "Owned" would have quietly told the model that
those 62 people were the safest kind of borrower, with no evidence at all. Giving them an
"Unknown" label lets the model decide for itself.

**Why the `gst` column is useless.**
We checked the arithmetic and found it is *exactly* 18% of the loan amount, every single
time. The processing fee is exactly 2%, and the net disbursement exactly 80%. All three are
just the loan amount multiplied by a fixed number — so they contain **no new information at
all**. We removed them for being redundant, not for being wrong.

**Why we repaired 7 rows instead of deleting them.**
Seven rows had a corrupted fee. But the other two related columns were still correct, and
there is a rule connecting all three:

```
loan_amount − processing_fee − gst = net_disbursement
```

That rule let us calculate the true fee exactly. We did not guess it. After the repair, the
rule holds for every single row — **7 broken rows became 0**.

---

## 5. The features

The model sees **22 features**, built from 20 raw fields. The useful ones are ratios, not
raw amounts — because ₹500,000 means something very different to someone earning ₹300,000
than to someone earning ₹5,000,000.

| Feature | Why a lender cares |
|---|---|
| `loan_to_income` | Borrowing 5× your salary is stretched. 0.5× is not |
| `emi_to_monthly_income` | **The key one.** People default because the *monthly payment* doesn't fit their budget, not because the loan is big |
| `balance_to_loan` | Savings are the cushion that absorbs one bad month |
| `delinquency_ratio` | What share of their credit history was spent paying late. Past behaviour predicts future behaviour |
| `avg_dpd_per_delinquency` | Separates "a few days late, often" from "disappears for 90 days" |
| `credit_utilization_ratio` | Already maxing out credit = already struggling |
| `open_account_ratio` | Are they piling up debts, or closing them off? |

Plus age, loan length, number of dependants, credit history length, and five yes/no
categories (marital status, job type, residence, loan purpose, secured or not).

### What we deliberately left out

| Left out | Why |
|---|---|
| `principal_outstanding` and 4 others | **Only exist after the loan is paid out.** This is the leakage decision |
| `cust_id`, `loan_id` | Just ID numbers. A model that learns from an ID has memorised, not learned |
| `city`, `state`, `zipcode` | Default rates are flat across all of them (8.1%–9.1%) — no information. Also, using location can be a backdoor to discrimination |
| `gender` | Legally protected. And the rates are 8.79% vs 8.47% — basically identical, so removing it costs nothing. We kept it in the raw data only to *check* the model for bias |

### One small detail worth knowing

About **40% of borrowers have never been late.** For them,
`total_dpd ÷ delinquent_months` is dividing by zero, which produces infinity. Unguarded,
the model would train on garbage and the live service would return nonsense. Every ratio in
the code passes through a guard that catches this.

---

## 6. The model

**We tried the simple option first.** Logistic regression is the banking industry standard.
It scored 0.9834. XGBoost scored 0.9885 — better, but only slightly.

**Handling the imbalance.** Only 8.6% of loans defaulted, so a lazy model could say
"nobody defaults" and be 91.4% accurate while catching nothing. We tested three fixes and
shipped `scale_pos_weight`, which tells the model to pay 10× more attention to the rare
defaults. We avoided throwing away good loans to balance the classes — that discards
29,000 rows of useful information.

**Tuning.** 40 automated experiments (Optuna) searching for the best settings, using
cross-validation. The test set was never touched during this.

**Calibration — the step most people skip.** The raw model said applicant C26424 had a
**75%** chance of default. After calibration: **36%**. The ranking did not change, but the
*number* became trustworthy. This matters because the business formula is:

```
expected loss = probability × loss-given-default × loan size
```

If the probability is wrong by half, every price is wrong by half.

Calibration cut our error by **7.6×**.

**Choosing the cut-off.** Not 0.5. We assumed:

- Approving a defaulter costs **60%** of the loan
- Rejecting a good customer costs **6%** (the interest we miss out on)

That's a 10:1 ratio, so the cut-off lands at **0.080**. Compare:

| Cut-off | Defaults we catch | Defaults we miss |
|---|---|---|
| **0.080** (ours) | **96.3%** | **24** |
| 0.500 (the lazy default) | 73.5% | 171 |

Using 0.5 would approve slightly more people but let through **seven times more bad loans**.

---

## 7. The results — and the thing you must say first

| Measure | What it means | Our score |
|---|---|---|
| **ROC-AUC** | Pick one defaulter and one good payer at random. How often does the model rank them correctly? | **0.9885** |
| **KS** | How cleanly the model separates good from bad | **90.30** |
| **Recall** | Of all the defaults, how many did we catch? | **96.27%** |
| **Bad rate after filtering** | Of the loans we approved, how many went bad? | **0.37%** (down from 8.59%) |

### ⚠️ Say this before anyone asks

**An AUC of 0.9885 is not believable for a credit model.** Real ones score 0.70–0.80.

A number that good normally means you have made a mistake — so we investigated instead of
celebrating. **It is not leakage.** We checked three ways:

1. We added the most suspicious excluded column back in. The score moved by **0.0001** —
   nothing.
2. There is an automated test that fails if any "after the loan" column sneaks into the
   model.
3. We trained on older loans and tested on newer ones. Still 0.9855. Leakage does not
   survive that test.

**The real reason is that the data is artificial.** It was generated, not collected. In
this dataset, 8,150 people with low credit utilisation and no late payments defaulted at a
rate of **exactly zero**. Real people are never that predictable — some lose a job and
default anyway.

**So: the method is real, the numbers are not transferable.** Saying this yourself turns
your biggest weakness into your strongest signal.

---

## 8. One applicant, start to finish

Meet **C26424** — a real row from the test set. The model declined them, **and they really
did default.**

**Who they are:** 24 years old, earns ₹380,000 a year, wants ₹683,000 for education over 51
months. Owns their home. Has ₹113,045 saved.

**The two numbers that matter:** their credit utilisation is **85%** (nearly maxed out), and
they were **late in 9 of their last 60 months**.

### What the model does with them

| Step | Result |
|---|---|
| Raw fields in | 20 values |
| Turned into features | `loan_to_income` 1.80 · `delinquency_ratio` 0.15 · `utilisation` 85 |
| XGBoost says | 0.7527 |
| After calibration | **0.3571** |
| Credit score | 300 + (1 − 0.3571) × 600 = **686** |
| Rating band | "Good" |
| Decision | **Decline** (0.3571 is above our 0.080 cut-off) |
| What actually happened | **They defaulted.** The model was right |

### Why it decided that — in plain English

The model can explain itself using a tool called **SHAP**, which splits a decision into one
number per feature. For this applicant:

| What the model saw | Effect |
|---|---|
| Credit utilisation 85% | **Pushed strongly toward decline** |
| They own their home | Pushed toward approve |
| Only 1 credit enquiry | Pushed toward approve |
| 51-month loan | Slightly toward decline |
| Only 24 years old | Slightly toward decline |

Turned into the rejection letter a regulator would require:

> *"Your credit utilisation is 85%, compared with an average of 37% among applicants we
> approve. You were also late on payments in 9 of the last 60 months."*

### Three things this one person teaches

1. **75% → 36% is not a bug.** It is exactly why calibration exists.
2. **"Good" rating + "decline" decision is not a contradiction.** The rating compares them
   to everyone else; the decision compares their risk to our cut-off.
3. **At the lazy 0.5 cut-off, this person would have been APPROVED** — and they defaulted.
   One person, demonstrating the whole threshold argument.

---

## 9. How it's served

The trained model is a 145 KB file. But the systems that need it — a bank's loan software,
a mobile app — are not written in Python and are not on your laptop. So we wrap it in a
**web service**: they send a message, they get an answer back.

| What you call | What you get |
|---|---|
| `POST /predict` | One applicant scored |
| `POST /predict/batch` | Up to 1,000 at once |
| `GET /health` | Is it alive, and did the model load? |
| `GET /model/info` | Version, features, scores |
| `/docs` | A clickable page where you can try it in a browser |

**Four things worth pointing at:**

**Bad input gets rejected politely.** Age 15? Income of −50,000? Credit utilisation of 150%?
Each comes back as a clear error message, not a crash and not a confident wrong answer.
There are also cross-checks — you cannot borrow more than you were approved for, and you
cannot have been late more months than you have had credit.

**The model loads once, when the service starts.** Loading it takes 534 milliseconds;
answering a request takes 3.6. Loading it per request would make the service 150× slower.

**If the model file is missing, the service refuses to start.** That sounds unhelpful, but
it is deliberate: a service that can never give a real answer should not pretend to be
healthy, or the system keeps sending it traffic.

**Feature calculation lives inside the model file.** The web service does not calculate
features at all — it cannot, because it has no copy of that code. This makes it *impossible*
for the live service to disagree with training. There is a test that proves it: training
saves one example and its answer, and the test checks the live service gives the identical
number.

**Speed:** 3.6 milliseconds on the machine. 388 milliseconds over the internet — the
difference is network travel and a free hosting tier, not the model thinking.

It is packaged with **Docker** (which bundles the code and everything it needs to run, so it
behaves identically everywhere) and deployed to **Render** (which gives it a public web
address). Every time the code is pushed, an automated robot runs all 65 tests before
anything goes live.

---

## 10. The code, in one table

| File | Lines | What it does |
|---|---|---|
| `src/config.py` | 93 | All the settings in one place — including the cost assumptions, so they are easy to challenge and change |
| `src/data.py` | 209 | Loads the three files, joins them, applies the six cleaning rules, **and counts everything it touched** |
| `src/features.py` | 270 | **The most important file.** Turns raw fields into the 22 features. Used by both training and the live service |
| `src/pipeline.py` | 174 | Assembles the model's processing steps |
| `src/evaluate.py` | 315 | All the scoring measures, the credit-score conversion, and the bias checks |
| `src/train.py` | 769 | Runs everything: baseline → tuning → calibration → leakage check → saves the model |
| `api/schemas.py` | 161 | Defines what a valid application looks like |
| `api/main.py` | 233 | The web service |
| `tests/` | 574 | 65 tests |

### The three tests that matter most

1. **The parity test** — checks the live service returns the exact same number training did.
   This is what makes "training and serving can't drift" a fact instead of a promise.
2. **The leakage test** — fails if anyone ever adds an "after the loan" column back in. The
   decision cannot be undone by accident.
3. **The cleaning tests** — every number quoted in the documentation (22 typos, 62 nulls, 8
   deleted, 7 repaired) is checked by a test. **The documentation cannot drift from reality.**

---

## 11. What to say in an interview

### Five things to volunteer before being asked

1. **"The AUC is 0.99, which isn't believable for credit risk."** Then explain how you ruled
   out leakage and why the data is artificial. **Always lead with this.**
2. **"I excluded data that wouldn't exist at decision time, then measured what it cost."**
   Answer: nothing. A negative result, reported honestly.
3. **"Calibration matters because we price loans off the number."** Raw 75% → calibrated
   36% on a real applicant.
4. **"The cut-off comes from a cost ratio, not 0.5."** 24 defaults missed instead of 171.
5. **"Feature code is written once and saved inside the model."** A test proves it.

### Where the model is weak — know these

- **The data is artificial.** Real performance would be far lower.
- **The score is nearly all-or-nothing** — 94.5% of applicants land at either the top or
  bottom band. Real pricing needs a spread.
- **Home ownership matters a lot** — owners get approved 14 points more often than renters.
  The link to default is real, but this would need a fairness review before real use.
- **Nobody defined "default."** 90 days late? Written off? It changes what is being
  predicted.
- **The model has never seen a recession.** Default rates are flat across all five years of
  data.

### What you can honestly claim

You built a complete machine-learning system: data cleaning, feature design, a tested
decision about leakage, a tuned model, calibrated probabilities, a cost-based cut-off, a
live web service, 65 tests, and automated checks on every change — **and you can explain the
trade-off behind every one of those choices.**

### What not to claim

Do not call it "production". It runs on a free tier, on artificial data, with no password
protection. Call it **deployed**, and say the numbers are not a forecast of real-world
performance.

That honesty is not a weakness. Knowing what a believable number looks like is the rarer
skill.
