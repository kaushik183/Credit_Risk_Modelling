# Start Here — the order to read this project

You have a repo you did not type line by line, and you need to be able to defend every
decision in it. That is a normal situation and it is solvable, because **the reasoning in
this project is real and it is all written down.** The job is to load it into your head in
the right order.

**Prefer one document to six?** [`COMPLETE_REFERENCE.md`](COMPLETE_REFERENCE.md) covers
everything below in a single ~10,000-word pass: concepts from zero, every file and function,
every number traced, and one applicant followed end to end. Use this reading order instead
if you would rather absorb it in sequenced pieces with checkpoints.

Total reading: **~135 minutes.** Add the exercises and it is a focused half-day.

---

## The rule that makes this honest

If asked, say plainly: *"I used AI tooling to build it, the same way I'd use any tool."*
That is unremarkable in 2026 and nobody serious will hold it against you.

**What would be dishonest is claiming to understand something you do not.** So the bar is
not "pretend you typed it". The bar is: **every decision in this repo, you can explain the
reasoning for, and say why the alternative was rejected.** That is a real and defensible
bar, and this reading order gets you there.

If you hit a question you genuinely cannot answer, the correct move in an interview is
"I'd have to check that" — not a guess. One honest "I don't know" costs you far less than
one confident wrong answer.

---

## Phase 1 — The 20-minute version (do this first, always)

If an interview is tomorrow and you read nothing else, read this.

### 1. `README.md` — the top third only, down to "Architecture" · **7 min**

Gets you: the business problem, the headline results table, and — most importantly — the
call-out box explaining why 0.9885 AUC is **not** a brag.

**Be able to say without looking:**
- What the model predicts and for whom
- Base default rate: **8.594%** — one loan in twelve
- Final: **0.9885 AUC**, and *immediately* that this is not credible for a real scorecard
- Why: the data is synthetic; 8,150 low-utilisation borrowers defaulted at exactly 0%
- That you ruled out leakage three ways (ablation, tests, out-of-time split)

> **This is the single most important thing in the project.** An interviewer who sees
> 0.9885 will assume you have leakage or do not understand your own metrics. Getting in
> front of that — "the number is not believable and here is how I checked" — turns your
> biggest liability into your strongest signal.

### 2. `docs/INTERVIEW_GUIDE.md` §0 — glossary + applicant C26424 · **8 min**

Every credit and tooling term in one plain line each, then the one applicant threaded
through the whole project.

**Learn C26424 cold.** They are your worked example for every question:
> 24 years old, ₹380k income, wants ₹683k for education, **85% credit utilisation**,
> late in 9 of 60 months. Model says **PD 0.357 → score 686 → decline**.
> **They actually defaulted.**

One concrete person you can talk about beats any amount of abstract description.

### 3. `docs/INTERVIEW_GUIDE.md` §17 — the 20 Q&As · **5 min skim**

Skim all twenty; read Q1 (leakage), Q4 (precision), Q5 (gender) properly. These are the
hostile ones and they are where interviews are won or lost.

---

## Phase 2 — The core reasoning (~45 min)

Now the decisions, in the order an interviewer probes them.

### 4. `INTERVIEW_GUIDE.md` §1–§4 — problem, data, cleaning, features · **12 min**

The six data problems and what you did about each. **Two are worth memorising**, because
they show you checked the brief instead of trusting it:

- **`gst` is not "inconsistent"** — it is *exactly* 18% of `loan_amount` (fee 2%, net
  disbursement 80%). All three are exact linear functions of one column, so they carry zero
  information. Dropped for redundancy, not inconsistency.
- **The 7 identity failures and the 7 fee outliers are the same 7 rows.** `processing_fee`
  is corrupted; `gst` and `net_disbursement` are fine, so the true fee is recoverable
  exactly as `gst / 9` — confirmed by violations going 7 → 0. Repaired, not dropped.

Also fix in your head: **why `residence_type` nulls became `Unknown` and not the mode** —
the mode is `Owned`, the *lowest-risk* category, so imputing it would silently label 62
applicants as the safest kind of borrower on no evidence.

### 5. `INTERVIEW_GUIDE.md` §5–§8 — model choice, boosting, hyperparameters, imbalance · **12 min**

§6 (how boosting works in plain words) is the one to be able to say out loud
unprompted. Practice it: *"Start with a constant guess. Fit a small tree to predict how
wrong it is. Add that correction, scaled down. Repeat a few hundred times."*

§5 has the answer that separates you from most candidates: **when logistic regression is
the better choice** — regulation, adverse-action notices, guaranteed monotonicity.

### 6. `INTERVIEW_GUIDE.md` §9–§11 — metrics, calibration, leakage · **12 min**

Every metric with our number and what "good" looks like. **Anchor on: real scorecards land
at 0.70–0.80 AUC and KS 30–40.** Knowing the realistic range is what proves you understand
the metric rather than just reporting it.

§10 (calibration) is the most under-appreciated section. The one-liner:
*"XGBoost is trained to rank, not to output probabilities. If you price a loan off the
number, it has to mean what it says."* C26424 makes it concrete — raw model says
**0.75**, calibrated says **0.36**.

### 7. `INTERVIEW_GUIDE.md` §13 — threshold economics · **6 min**

Why not 0.5. Missed default ≈ 60% of exposure, wrongly declined customer ≈ 6% → **10:1**,
so the cut-off sits at **0.080**, far below 0.5.

**The killer detail:** at 0.5 the model would have *approved* C26424 (0.357 < 0.5) and they
defaulted. Across the test set, 0.5 misses **171 defaults instead of 24**.

Also know this is an *assumption*, not a measurement — both constants live in `config.py`
as a one-line change.

### 8. `INTERVIEW_GUIDE.md` §12 — why the AUC is so high · **3 min**

Re-read it. Yes, again. It is the question you are most likely to be asked.

---

## Phase 3 — The parts you have never used (~35 min)

Be honest with yourself about which of these you have genuinely not touched, and spend
your time there.

### 9. `INTERVIEW_GUIDE.md` §12.5 — SHAP from zero · **12 min**

Written assuming you have never heard of a Shapley value. The analogy (four people, a
bonus, fair shares averaged over every joining order) comes before any equation.

**Do the arithmetic yourself once** — it makes the whole idea click:
```
base 0.3838 + SHAP sum 0.7292 = 1.1130 log-odds → sigmoid → 0.752681
the model actually says                                      0.752682
```

**Be able to read a beeswarm plot** — one dot per applicant, x-axis is SHAP value, colour
is the feature's actual value. That is a common live question, and the plot is sitting in
`reports/figures/shap_beeswarm.png`. Open it and narrate it to yourself.

### 10. `INTERVIEW_GUIDE.md` §14 — serving, Docker, CI from zero · **15 min**

§14.0 answers "why not just call `model.predict()`?" — because the caller is a Java system
on another machine that cannot import your Python.

§14.1 traces C26424's request through all 11 layers. **Read it with the live API open in
another tab** and actually send the request. Watching the thing you just read about return
`686` makes it stick in a way reading twice will not.

§14.2–14.3: image vs container, what Render actually provides, CI vocabulary.

### 10b. `docs/API_AND_DEPLOYMENT.md` — the deep version · **25 min**

Guide §14 is the summary; **this is the real teaching document** for the part you have
never done. Five layers, each ending with "why not just stop here?": why a web service
exists at all → FastAPI, Pydantic, uvicorn and a line-by-line walkthrough of our two API
files → Docker line by line, including *why* `requirements.txt` is copied before the source
→ what Render actually gives you → CI vocabulary → C26424's request through all 13 steps.

**Read this if you have time for only one Phase 3 item.** Deployment is where a follow-up
question is most likely to go deeper than the guide covers.

### 11. `INTERVIEW_GUIDE.md` §18 — the code walkthrough · **8 min**

For "walk me through your code". File by file, plus §18.13, an honest list of what you
would refactor. **Having a refactor list ready is a strong signal** — it says you can see
your own work critically.

---

## Phase 4 — Depth, if you have it (~30 min)

### 12. `docs/MODEL_CARD.md` — §5 fairness and §6 limitations · **11 min**

Where the model is weak, sliced by segment. Know that **`residence_type` is the real
fairness exposure** — a 14-point approval gap between owners and renters — not gender.
Volunteering that is a strong move.

### 13. `notebooks/01_exploratory_data_analysis.ipynb` · **6 min**

Shows how each cleaning decision was *reached*, with the outputs already there.

### 14. `INTERVIEW_GUIDE.md` §15–§16 — monitoring and weaknesses · **8 min**

"What breaks first in production?" → **calibration, before discrimination.** The ranking is
robust; the probability levels drift as soon as the population shifts.

### 15. `docs/RESUME.md` — bullets, pitch, and the honesty line · **8 min**

Paste-ready titles, bullets and a 60-second spoken pitch — **practise that out loud**,
reading it silently will not surface the sentences you stumble on.

§8 is the part to actually study: a table of what the repo supports versus what would be
overclaiming, and three sentences that keep you safe if asked "is this real?".

### 16. Open the actual code · **as long as you have**

Read in this order — smallest and most important first:
`config.py` (93) → `features.py` (270) → `api/schemas.py` (161) → `api/main.py` (233) →
`data.py` (209) → `evaluate.py` (315) → `train.py` (769, skim the plotting).

---

## Self-test: can you answer these without looking?

If yes to all fifteen, you can defend this project.

**The dangerous ones**
1. Your AUC is 0.9885. Do you have leakage?
2. Why is 0.5 the wrong threshold here?
3. You excluded gender, but men are approved more. Explain.
4. Precision is 58%. Isn't that a bad model?
5. How do you prevent training/serving skew?

**The fundamentals**
6. Explain gradient boosting in plain words.
7. Why is accuracy useless at an 8.6% base rate?
8. What is calibration, and why does it matter here?
9. What is KS, and what is a good value in credit risk?
10. Why XGBoost over logistic regression — and when would you flip that?

**The specifics**
11. Why did `residence_type` nulls become `Unknown` instead of the mode?
12. Why repair 7 rows rather than drop them?
13. What does SHAP actually compute?
14. What happens between `curl` and a JSON response?
15. What breaks first in production, and how would you know?

Every answer is in the guide. If you can answer all fifteen out loud, unprompted, you
understand this project — regardless of who typed it.

---

## Where the docs are still thin — know this before you are caught

Being able to name your own gaps is worth more than pretending there are none.

| Gap | Where you would feel it |
|---|---|
| **FastAPI internals** | `API_AND_DEPLOYMENT.md` covers routing, Pydantic, lifespan, middleware, and async honestly. It does not make you a FastAPI developer — on dependency injection, background tasks or WebSockets, say you have used it for this project and would need to read up |
| **Docker internals** | Image vs container, our Dockerfile line by line, and layer caching are covered. Networking, volumes and multi-container orchestration are not |
| **SHAP mathematics** | §12.5 gives intuition, the equation, and verified additivity. The formal derivation of Shapley values is not there |
| **The data is synthetic** | Never let this be a discovery. Say it first, every time |

**Do not claim:** that this is "in production" (it is a free tier with synthetic data), that
the metrics predict real-world performance (§12), or deep expertise in FastAPI/Docker/SHAP
beyond using them here.

**Do claim:** you built an end-to-end ML system with a deliberate leakage decision you
tested, calibrated probabilities, threshold economics grounded in cost, a serving layer
with a structural guarantee against training/serving skew, 65 tests, CI, and a live
deployment — and that you can explain the trade-off behind every one of those choices.

That is all true, and it is a strong claim on its own.
