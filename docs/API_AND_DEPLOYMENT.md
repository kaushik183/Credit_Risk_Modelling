# From Model File to Live URL — Serving and Deployment, From Zero

This document assumes you have **never built a web API, never used Docker, never deployed
anything, and never set up continuous integration.** Every term is defined where it first
appears. Nothing is called "simple".

It is built in layers. Each layer ends by answering *"why not just stop here?"* — because
that question is the actual reason each piece of the stack exists.

Throughout, one real applicant is followed: **C26424**, from our test set. 24 years old,
₹380,000 annual income, wants ₹683,000 for education, **85% credit utilisation**, late in 9
of their last 60 months. The model scores them **PD 0.357 → credit score 686 → decline**.
They did in fact default.

**Contents**

- [Layer 0 — Why a web service at all](#layer-0--why-a-web-service-at-all)
- [Layer 1 — FastAPI](#layer-1--fastapi)
- [Layer 2 — Docker](#layer-2--docker)
- [Layer 3 — Hosting on Render](#layer-3--hosting-on-render)
- [Layer 4 — GitHub Actions (CI)](#layer-4--github-actions-continuous-integration)
- [Layer 5 — The full journey of one request](#layer-5--the-full-journey-of-one-request)

---

## Layer 0 — Why a web service at all

### The situation

Training produced one file: `models/model.joblib`, 145 KB. In Python you can do this:

```python
import joblib
model = joblib.load("models/model.joblib")
model.predict_proba(applicant_dataframe)   # → 0.357
```

That works. So why is there a `api/` directory at all?

### What breaks without a service

**The thing that needs the prediction is usually not Python, and usually not on your
machine.**

A loan-origination system written in Java. A mobile app in Swift. A React dashboard. A
colleague's R script. None of them can `import` your pipeline. Even another Python service
would have to install your exact dependency versions and get a copy of the artifact — and
then there would be two copies that could drift apart.

What every one of those *can* do is send a message over a network. That is what an API is
for.

Three more things you get:

1. **One copy of the model.** Update it in one place and every caller gets the new version.
2. **Control.** You can log, rate-limit, authenticate, and monitor at one boundary.
3. **Scale independence.** The scoring service can be scaled without touching the callers.

### The vocabulary

- **API (application programming interface)** — a contract that lets one program ask
  another a question. Ours accepts a loan applicant and returns a risk assessment.
- **HTTP** — the message format of the web. You send a **request**; you get back exactly
  one **response**.
- **Endpoint** — one specific thing the API can do, named by a **path**: `/predict`,
  `/health`.
- **GET vs POST** — `GET` fetches something without changing anything (`GET /health`).
  `POST` sends data to be processed (`POST /predict`). We use POST for scoring because the
  applicant's 20 fields travel in the request **body**, and a GET has no body.
- **JSON (JavaScript Object Notation)** — the text format for that body. Braces, keys and
  values: `{"age": 24, "income": 380000}`. Every language can read and write it.
- **Status code** — a number in the response saying what happened:

| Code | Meaning | When ours returns it |
|---|---|---|
| **200** | OK | A successful prediction |
| **422** | Unprocessable Entity | Your input failed validation — age 15, utilisation 150 |
| **500** | Internal Server Error | We crashed. A bug |
| **503** | Service Unavailable | Up but not ready to serve |

**The 422-versus-500 distinction is the important one.** A 422 means we inspected your data
and it was invalid — the API working exactly as designed, and the caller can read the
message and fix it. A 500 means we fell over. **Most of `schemas.py` exists to turn
would-be 500s into 422s.**

### Concretely, for C26424

The caller sends:

```http
POST /predict HTTP/1.1
Host: credit-risk-api-1dbj.onrender.com
Content-Type: application/json

{"age":24,"income":380000,"number_of_dependants":1,"years_at_current_address":17,
 "marital_status":"Single","employment_status":"Salaried","residence_type":"Owned",
 "loan_purpose":"Education","loan_type":"Secured","sanction_amount":796000,
 "loan_amount":683000,"loan_tenure_months":51,"bank_balance_at_application":113045,
 "number_of_open_accounts":1,"number_of_closed_accounts":1,"total_loan_months":60,
 "delinquent_months":9,"total_dpd":40,"enquiry_count":1,"credit_utilization_ratio":85}
```

And gets back:

```http
HTTP/1.1 200 OK
Content-Type: application/json
X-Request-ID: 70703ddf
X-Response-Time-ms: 0.66

{"probability_of_default":0.357143,"credit_score":686,"rating":"Good",
 "decision":"decline","threshold":0.08,"model_version":"1.0.0"}
```

That is the entire contract. Everything below exists to make that exchange reliable.

> **Why not stop here?** Because writing HTTP handling, JSON parsing, routing and
> validation by hand is hundreds of lines of fiddly, security-sensitive code. That is what
> Layer 1 is for.

---

## Layer 1 — FastAPI

### What a framework does for you

A **web framework** handles the mechanics — parsing HTTP, matching paths to functions,
serialising responses — so you write only the part that is specific to your problem.

**FastAPI** is the one we use. Three pieces to keep straight, because they are often
confused:

| Piece | Job | Analogy |
|---|---|---|
| **uvicorn** | Listens on a network port, accepts connections | The receptionist who answers the phone |
| **FastAPI** | Decides which function handles which path | The switchboard routing the call |
| **Pydantic** | Checks incoming data is valid before your code runs | The form checker who rejects a blank field |

**ASGI (Asynchronous Server Gateway Interface)** is the agreed protocol between uvicorn and
FastAPI. It is the reason you can swap uvicorn for another server without changing app
code. You rarely think about it; you just need to know the word.

### What `@app.post("/predict")` actually means

```python
@app.post("/predict", response_model=PredictionResponse, tags=["scoring"])
async def predict(applicant: ApplicantRequest) -> PredictionResponse:
    return _score([applicant])[0]
```

Line by line:

- `@app.post(...)` is a **decorator** — a line starting with `@` that modifies the function
  beneath it. This one registers `predict` in FastAPI's routing table for POST requests to
  `/predict`. Without it the function is just a function nobody can reach.
- `response_model=PredictionResponse` tells FastAPI the shape of the reply. It validates
  the outgoing data too, and publishes the shape in the docs.
- `tags=["scoring"]` groups it in the auto-generated documentation page.
- `applicant: ApplicantRequest` is the part that does the most work. That **type annotation
  is not a comment.** FastAPI reads it, and automatically parses the JSON body, validates
  it against the schema, and returns a 422 if it fails — all before your function runs. By
  the time the body of `predict` executes, `applicant` is guaranteed valid.

That last point is the thing to understand about FastAPI: **the type hints are the
behaviour.**

### Why a typed schema at the boundary beats validating inside the function

You could write `if age < 18: raise ...` inside `predict`. Declaring it in the schema is
better for four reasons:

1. **It runs before your code**, so invalid data never reaches the model.
2. **It is machine-readable**, so the documentation is generated from it and cannot drift.
3. **It is exhaustive** — every field is checked every time, not just the ones you
   remembered.
4. **The error is precise**: which field, what rule, what value you sent.

### Automatic `/docs` and OpenAPI

**OpenAPI** is a standard for describing an API in JSON — every endpoint, every field,
every type and bound. FastAPI generates it from your schemas at `/openapi.json`, and serves
an interactive page at `/docs` built from it.

That page is not decoration. You can expand `POST /predict`, click **Try it out**, and send
a real request from the browser. **This is the fastest way to demo the project in an
interview** — no curl, no Postman.

Try it: [credit-risk-api-1dbj.onrender.com/docs](https://credit-risk-api-1dbj.onrender.com/docs)

### The lifespan handler — and the measured reason for it

A **lifespan handler** runs code once when the app starts and once when it stops.

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not config.MODEL_PATH.exists():
        raise RuntimeError(f"Model artifact not found at {config.MODEL_PATH}. ...")
    ARTIFACTS["model"] = joblib.load(config.MODEL_PATH)
    ...
    yield          # ← the app runs here, for its whole life
    logger.info(json.dumps({"event": "shutdown"}))
```

Everything before `yield` runs at startup, everything after at shutdown.

**Why this matters, measured on this project:**

| | Time |
|---|---|
| `joblib.load()`, first call | **534 ms** |
| `joblib.load()`, warm | 2.6 ms |
| A whole `/predict` request | **3.62 ms** |

Loading the model inside the request handler would add somewhere between 2.6 ms and 534 ms
to *every single call* — best case doubling latency, worst case making it 150× slower. The
model is read once and reused for the life of the process.

**It also fails loudly.** If the artifact is missing, the `RuntimeError` prevents the app
from starting at all. That is deliberate: a container that can never serve a valid request
must never report itself healthy, or the platform will keep sending it traffic.
`tests/test_startup.py` asserts this behaviour.

### `async` — and an honest answer about it

`async def` marks a function that can pause while waiting for something slow (a database, a
network call) and let the server handle other requests meanwhile.

**Honestly: it does very little for us.** Our work is `predict_proba` — pure CPU. There is
no waiting to overlap with. An `async` function doing CPU work holds the event loop and
blocks other requests just as a synchronous one would.

We use `async def` because it is idiomatic FastAPI and costs nothing here. **Concurrency
actually comes from running two uvicorn worker processes** (`WEB_CONCURRENCY=2`), which are
separate OS processes and genuinely run in parallel.

If an interviewer asks whether async helps here, the correct answer is: *"not meaningfully
— the workload is CPU-bound, so the parallelism comes from worker processes, not the event
loop."* Saying async makes it faster would be wrong.

### Walkthrough: `api/schemas.py`

```python
MaritalStatus    = Literal["Married", "Single"]
EmploymentStatus = Literal["Salaried", "Self-Employed"]
ResidenceType    = Literal["Owned", "Mortgage", "Rented", "Unknown"]
```

`Literal` means *only these exact strings are acceptable*. Three benefits at once: a typo
like `"self-employed"` (wrong case) is a 422; `/docs` renders a dropdown; the allowed
values appear in the OpenAPI schema for callers to read.

This matters more than it looks. Without it, `"self-employed"` would be one-hot encoded to
**all zeros** — an applicant with no employment status at all — and the model would return
a confident, meaningless score. **Silent wrongness is worse than an error.**

```python
age:    Annotated[int,   Field(ge=18, le=100, description="...")]
income: Annotated[float, Field(gt=0, le=1e9, description="...")]
credit_utilization_ratio: Annotated[float, Field(ge=0, le=100, ...)]
```

`Annotated[type, Field(...)]` attaches rules to a type. `ge` = greater or equal, `le` =
less or equal, `gt` = strictly greater. The bounds are **deliberately realistic, not merely
type-correct**: `age` is capped at 100 because 130 is a data-entry bug, and `income` uses
`gt=0` rather than `ge=0` because a zero-income applicant breaks every ratio feature —
exactly the 8 rows dropped during cleaning.

```python
@model_validator(mode="after")
def check_internal_consistency(self) -> "ApplicantRequest":
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

`mode="after"` means this runs once every individual field has passed. These are rules **no
single-field bound can express** — each involves two fields at once.

They catch upstream bugs that would otherwise produce a plausible-looking score from
impossible data. Someone late 40 months in a 12-month credit history is a broken join or a
column mix-up, and a scorecard should refuse rather than guess.

**Note what is absent from the whole file:** no `principal_outstanding`, no
`net_disbursement`, no `gst`. The API physically cannot accept post-disbursal data, so the
leakage decision is enforced at the network boundary, not merely in training code.

### Walkthrough: `api/main.py`

**The artifact store.**

```python
ARTIFACTS: dict[str, Any] = {"model": None, "metadata": {}, "threshold": 0.5, "started_at": time.time()}
```

A module-level dict the lifespan handler fills at startup and the handlers read. Plain and
easy to follow. In a larger service this would be dependency-injected for easier testing.

**The middleware.**

```python
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    started = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - started) * 1000
    logger.info(json.dumps({"event": "request", "request_id": request_id, ...}))
    response.headers["X-Request-ID"] = request_id
    return response
```

**Middleware** wraps every request: code before `call_next` runs on the way in, code after
runs on the way out. This one stamps a unique id, times the request, and logs one JSON
line.

Logging **JSON rather than prose** matters: a log aggregator can chart p99 latency and error
rate directly. `latency_ms=0.66` in the log versus 3.62 ms measured by the client is the
difference between our processing time and the round trip — useful for telling "the model
is slow" apart from "the network is slow".

**Scoring.**

```python
def _score(applicants: list[ApplicantRequest]) -> list[PredictionResponse]:
    frame = pd.DataFrame([a.model_dump() for a in applicants])
    frame = frame[RAW_INPUTS]                                    # explicit column order
    probabilities = clip_pd(model.predict_proba(frame)[:, 1])    # clamp off 0 and 1
    scores = pd_to_credit_score(probabilities)
    ratings = score_to_rating(scores)
    return [PredictionResponse(...) for p, s, r in zip(probabilities, scores, ratings, strict=True)]
```

Four decisions in six lines:

1. **`_score` takes a list**, always. `/predict` calls it with one applicant. One code path
   means single and batch predictions can never diverge — asserted by a test.
2. **`frame[RAW_INPUTS]`** forces column order explicitly rather than trusting dict
   ordering. Silent column reordering is a classic, invisible source of wrong predictions.
3. **`clip_pd`** clamps to `[0.0001, 0.9999]`. Isotonic calibration's end steps are exactly
   0 and 1, and no lender can price off "will never default". Monotone, so no ranking metric
   changes.
4. **One `predict_proba` for the whole batch** — why 1,000 applicants take 12.6 ms total
   rather than 1,000 × 3.6 ms.

**Where the feature engineering is.** Nowhere in this file. It travels *inside*
`model.joblib`, which unpickles `src.features.build_features` at load time. The API cannot
compute a feature differently from training because it does not compute features at all.
That is the structural guarantee against training/serving skew, and
`tests/test_api.py::test_prediction_matches_what_train_py_recorded` asserts it by
comparing against a probability `train.py` recorded during training.

> **Why not stop here?** Because this runs on *your* machine, with your Python and your
> system libraries. Layer 2 is about making that reproducible anywhere.

---

## Layer 2 — Docker

### The problem it solves

Your laptop has Python 3.11, exact library versions, and `libomp` installed via Homebrew
(remember — XGBoost failed to import until that was installed). A server has a different
OS, a different Python, and none of your libraries.

"Works on my machine" is not a joke, it is the default outcome. **Docker packages the code
and its entire environment together** so it runs identically everywhere.

### Image vs container vs registry

| Term | What it is | Analogy |
|---|---|---|
| **Image** | A frozen snapshot: OS, Python, libraries, your code. Built once, read-only | A recipe with all ingredients sealed in |
| **Container** | A running instance of an image | The meal cooked from it. One recipe, many meals |
| **Registry** | A server storing images so others can download them | The cookbook library |

One image can run as many containers, on any machine with Docker, identically.

### Our Dockerfile, line by line

```dockerfile
FROM python:3.11-slim AS builder
```
Every image starts `FROM` another. `python:3.11-slim` is an official image with Python
pre-installed, stripped of documentation and extras. `AS builder` names this stage — we
have two, and this is the **multi-stage build** pattern.

```dockerfile
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
```
`build-essential` is the C compiler toolchain some Python packages need to build. The
`rm -rf` of the package lists in the *same* `RUN` matters: each instruction creates a
**layer**, and a file deleted in a later layer still occupies space in the earlier one.
Cleaning up in the same command keeps it out of the image entirely.

```dockerfile
COPY requirements-api.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-api.txt
```

**This is the layer-caching decision, and the most commonly asked Docker question.**

Docker caches each layer and reuses it if that step's inputs have not changed. Requirements
change rarely; source code changes constantly. By copying **only** the requirements file
and installing *before* copying the source, an edit to `main.py` reuses the cached
dependency layer — a rebuild takes seconds instead of minutes.

Copy the source first and every code change reinstalls every package. Same result, far
slower.

`--prefix=/install` puts packages in one directory the next stage can lift wholesale.
`--no-cache-dir` stops pip keeping downloaded wheels we will never reuse.

```dockerfile
FROM python:3.11-slim AS runtime
```
**The second stage starts from a clean base.** Everything from the builder — compilers,
caches — is discarded unless explicitly copied. This is why the toolchain never ships.

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*
```
`libgomp1` is XGBoost's OpenMP runtime — the Linux equivalent of the `libomp` that had to
be installed on macOS. Without it `import xgboost` fails with a `dlopen` error that says
nothing about a missing package. `curl` exists solely so `HEALTHCHECK` has something to
call.

```dockerfile
RUN useradd --create-home --uid 10001 appuser
```
By default containers run as **root**. If the process is compromised, root inside the
container is a much better starting position for an attacker. This creates an unprivileged
user. Verifiable: `docker exec crapi id` → `uid=10001(appuser)`.

```dockerfile
COPY --from=builder /install /usr/local
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser api/ ./api/
COPY --chown=appuser:appuser models/ ./models/
USER appuser
```
`--from=builder` lifts the installed packages out of the first stage. Then only what is
needed to serve: `api/`, `models/`, and `src/` — **`src/` is required because the pickled
pipeline imports `src.features` when it loads.** That is the training/serving contract
made physical. Note what is *absent*: `data/`, `tests/`, `notebooks/`, `reports/`.

`USER appuser` — everything after this runs unprivileged.

```dockerfile
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PORT=8000 WEB_CONCURRENCY=2
EXPOSE 8000
```
`PYTHONUNBUFFERED=1` is important in containers: without it Python buffers stdout and your
logs vanish if the process dies. `EXPOSE` is **documentation only** — it declares the port
but publishes nothing.

```dockerfile
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/health" || exit 1
```
Docker calls this every 30 seconds. Three consecutive failures mark the container
unhealthy and an orchestrator restarts it. `--start-period=20s` is grace time during boot.
This is why `/health` reports whether the model actually loaded, not merely that the
process is alive.

```dockerfile
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --workers ${WEB_CONCURRENCY}
```
- **Shell form** (not the `["uvicorn", ...]` list form) so `${PORT}` is expanded — Render
  and Fly assign the port at runtime.
- `--host 0.0.0.0` means "listen on all interfaces". The default `127.0.0.1` accepts only
  connections from *inside* the container, so the service would be unreachable. **This is a
  classic first-deployment bug.**
- `--workers 2` starts two processes for genuine parallelism.

### The commands, with every flag explained

```bash
docker build -t credit-risk-api:local .
```
`-t` tags the image with a name. The `.` is the **build context** — the directory sent to
the Docker daemon. `.dockerignore` keeps `data/`, `.venv/` and `.git/` out of it.

```bash
docker run -d --name crapi -p 8099:8000 credit-risk-api:local
```
`-d` detached (background). `--name` names the container. **`-p 8099:8000` maps port 8099
on your machine to 8000 inside** — this is what `EXPOSE` did *not* do. Left of the colon is
your machine, right is the container.

```bash
docker logs crapi                                    # see stdout
docker exec crapi id                                 # run a command inside → uid=10001
docker inspect --format='{{.State.Health.Status}}' crapi   # → healthy
docker rm -f crapi                                   # stop and remove
```

### What we verified

| Check | Result |
|---|---|
| Image size | 585 MB |
| Healthy after | 10 seconds |
| Running as | `uid=10001(appuser)` ✅ |
| Startup log lines | 2 (one per worker) ✅ |
| `/predict` on C26424 | `0.357143` — **identical to local** ✅ |
| Bad input | 422 ✅ |

> **Why not stop here?** Because the container runs on your laptop, which is not on the
> internet and is not always on. Layer 3.

---

## Layer 3 — Hosting on Render

### What a hosting platform actually gives you

Without one you would rent a server, install Docker, configure a firewall, obtain and renew
TLS certificates, set up a reverse proxy, arrange restart-on-crash, and ship logs somewhere.

**Render** does all of that. You give it a repository; it gives you:

1. **A machine** to run the container.
2. **A public URL with HTTPS** — TLS certificate obtained and renewed automatically.
3. **Automatic restarts** when the process dies or the health check fails.
4. **Logs**, collected and searchable.
5. **Rebuild on push** — it watches GitHub.

### `render.yaml`

```yaml
services:
  - type: web              # accepts HTTP traffic (vs a background worker)
    name: credit-risk-api
    runtime: docker        # use our Dockerfile, not Render's Python builder
    plan: free
    dockerfilePath: ./Dockerfile
    healthCheckPath: /health   # Render polls this; failures trigger a restart
    autoDeploy: true           # rebuild on every push to main
    envVars:
      - key: WEB_CONCURRENCY
        value: "2"
```

Committing this file is called **infrastructure as code** — the configuration is versioned
and reviewable, rather than a set of clicks nobody can reconstruct.

### Environment variables and secrets

An **environment variable** is a value passed to the process at startup, so the same image
behaves differently in different places. We use `WEB_CONCURRENCY` and `PORT`.

**The rule: secrets never live in the repository.** API keys and database passwords go in
the platform's environment settings, which are encrypted and not in git. We have no
secrets, which is itself a reason this deployment is simple — and why the endpoint is
currently unauthenticated.

### Cold starts — know this before demoing

On the free tier the instance **sleeps after 15 minutes of inactivity**. The next request
must boot a fresh container: **30–50 seconds.**

**Before an interview demo, open the URL a minute early to wake it.** Watching a page spin
for 45 seconds while you explain your architecture is avoidable.

Measured once warm:

| | p50 | p95 |
|---|---|---|
| In-process (no network) | 3.62 ms | 3.83 ms |
| Live on Render | 388 ms | 806 ms |

**Be precise about that gap in an interview: it is network and platform, not model.** TLS
handshake, Render's router, and a shared free-tier CPU in another region. The model itself
still takes single-digit milliseconds. On a paid instance in the caller's region, expect
tens of milliseconds.

### The deploy steps

1. **Push to GitHub.** `git push origin main`.
2. **Sign in** at [dashboard.render.com](https://dashboard.render.com) using **Sign in with
   GitHub**.
3. **Grant repository access.** Render asks which repositories it may read. Choose *Only
   select repositories* → `Credit_Risk_Modelling`. **The repo is private, so skipping this
   means the next step cannot see it** — the most common place to get stuck.
4. **New → Blueprint**, select the repo, confirm. Render reads `render.yaml`. No build
   command, no environment variables to set.
5. **Wait ~5 minutes.** In the log, look for:
   ```json
   {"event": "startup", "model_version": "1.0.0", "threshold": 0.08, "load_seconds": 0.041}
   ```
   Two of those, one per worker. Their absence plus an exited service means the app refused
   to boot because the artifact was missing — working as designed.
6. **Verify:**
   ```bash
   curl https://credit-risk-api-1dbj.onrender.com/health
   # {"status":"ok","model_loaded":true,"model_version":"1.0.0","uptime_seconds":88.1}
   ```

**On the URL:** the service is `credit-risk-api-1dbj`, not `credit-risk-api`, because that
hostname was already taken by another Render user and the platform appended a
discriminator. Expect a different suffix if you redeploy.

### Checking it is really predicting

`model_loaded: true` only proves a file loaded. The real test is that **output changes
sensibly with input.** Holding one applicant fixed and varying only utilisation, live:

| Utilisation | PD | Score | Decision |
|---|---|---|---|
| 5% | 0.0001 | 900 | approve |
| 40% | 0.0001 | 900 | approve |
| 60% | 0.0755 | 855 | approve |
| **75%** | **0.4419** | **635** | **decline** |
| 85% | 0.9500 | 330 | decline |
| 95% | 0.9999 | 300 | decline |

Monotone, and the decision flips between 60% and 75%. **That is what "the model is
predicting" actually looks like** — not a constant, and not noise.

> **Why not stop here?** Because `autoDeploy: true` means every push goes live. Without a
> gate, a broken commit is a broken production service. Layer 4.

---

## Layer 4 — GitHub Actions (continuous integration)

### What CI is and what it prevents

**Continuous integration (CI)** is a robot that runs your checks automatically on every
push. Without it, "I forgot to run the tests" ships to production. With `autoDeploy: true`,
that lands on a live URL within minutes.

### The vocabulary

| Term | Meaning |
|---|---|
| **Workflow** | The whole automated process. One YAML file in `.github/workflows/` |
| **Trigger** | What starts it. Ours: any push or pull request to `main` |
| **Runner** | The fresh virtual machine GitHub provides |
| **Job** | An independent unit of work. Jobs can run in parallel or depend on each other |
| **Step** | One command or action inside a job |
| **Action** | A reusable step someone else wrote, e.g. `actions/checkout@v4` |

**"Fresh" is the point.** The runner has none of your local setup, so a green build proves
the project works from nothing — not just on a machine where you already installed the
right things three months ago.

### Our workflow, walked through

```yaml
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  workflow_dispatch:
```
Three triggers. `workflow_dispatch` adds a manual "Run workflow" button.

```yaml
jobs:
  test:
    name: Lint and test
    runs-on: ubuntu-latest
```
`runs-on` picks the runner OS. **Ubuntu on x86, while development was on an arm64 Mac** —
so a green build also proves the pinned dependencies and the model artifact work on the
deployment architecture.

```yaml
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
```
Clone the repo, install Python 3.11. `cache: pip` reuses downloaded packages between runs.

```yaml
      - name: Lint
        run: ruff check src api tests
      - name: Format check
        run: ruff format --check src api tests
        continue-on-error: true
```
**Ruff** is a fast Python linter. Lint failures fail the build. Formatting uses
`continue-on-error: true` — reported but not blocking, because failing a build over
whitespace teaches people to ignore red builds.

```yaml
      - name: Train a smoke-test model
        run: python -m src.train --trials 5 --skip-shap
```
5 Optuna trials instead of 40 — seconds, not minutes. `models/` **is** committed, so the
API tests would pass without this. We train anyway because **it exercises the whole
training path on every push**, catching a data or feature change that broke training
before it silently ships a stale model.

```yaml
      - name: Run tests
        run: pytest tests/ -v --tb=short
      - name: Upload training artifacts
        if: always()
```
`if: always()` uploads the artifacts even when tests fail — so you can inspect what the
failing run produced.

```yaml
  docker:
    name: Build and smoke-test the image
    needs: test
```
`needs: test` means this job waits for the first. No point building an image from code that
fails its tests.

```yaml
      - name: Boot it and check /health and /predict
        run: |
          docker run -d --name api -p 8000:8000 credit-risk-api:ci
          for i in $(seq 1 30); do
            if curl -fsS http://localhost:8000/health; then break; fi
            sleep 2
          done
          curl -fsS http://localhost:8000/health | grep '"model_loaded":true'
          curl -fsS -X POST http://localhost:8000/predict ... | grep probability_of_default
```
The retry loop waits up to 60 seconds for startup. `grep` is doing assertion work: if
`"model_loaded":true` is absent, grep exits non-zero and the step fails. **This catches the
case where the image builds fine but cannot actually serve** — which a build-only check
would miss entirely.

### When the build is red

"Red" means a step failed. Click the run in the **Actions** tab, open the failed step, and
**read the log from the bottom** — the real error is usually just above the summary.

Red means do not deploy. With `autoDeploy: true`, Render may already be building the same
commit, so a red CI is your signal to fix forward quickly or revert.

Our first run passed both jobs, every step, with no skips.

---

## Layer 5 — The full journey of one request

Everything above, in one narrative. C26424, from typing a command to reading a log line.

**1. The command.**
```bash
curl -X POST https://credit-risk-api-1dbj.onrender.com/predict \
  -H 'Content-Type: application/json' \
  -d '{"age":24,...,"credit_utilization_ratio":85}'
```
`-X POST` sets the method, `-H` sets a header declaring the body is JSON, `-d` is the body.

**2. DNS and TLS.** The hostname resolves to a Render IP. An encrypted HTTPS connection is
established. *(This is most of the 388 ms.)*

**3. Render's router** matches the hostname to your service and forwards to the container.

**4. uvicorn** — one of two workers — accepts the connection and translates raw HTTP into
an ASGI message for FastAPI.

**5. Middleware, inbound.** `log_requests` generates request id `70703ddf` and starts a
timer.

**6. Routing.** FastAPI matches `POST /predict` → the `predict` function.

**7. Pydantic validation.** Age 24 ✓ (18–100). Utilisation 85 ✓ (0–100). Income 380000 ✓
(>0). `loan_purpose: "Education"` ✓ (allowed). Then cross-field: 683000 ≤ 796000 ✓,
9 ≤ 60 ✓, 1+1 > 0 ✓, dpd 40 with 9 delinquent months ✓. **Any failure stops here with a
422 and the model is never reached.**

**8. `_score()`** builds a one-row DataFrame, ordered by `RAW_INPUTS`.

**9. Inside the pipeline** (all three steps live in `model.joblib`):
- `build_features` → 20 raw fields become 22 features. Utilisation 85, `delinquency_ratio`
  0.15, `loan_to_income` 1.80, `emi_to_monthly_income` 0.42.
- `ColumnTransformer` → one-hot encoding expands 5 categoricals into 31 total columns.
- `XGBClassifier` → 612 trees vote (early stopping halted at iteration 612 of a
  1,200 ceiling). **Raw output: 0.7527.**
- `CalibratedClassifierCV` (isotonic) → **corrected to 0.3571.** The raw model over-states
  risk; calibration is what makes the number mean what it says.

**10. Post-processing.** `clip_pd` leaves 0.3571 unchanged. Score = `300 + (1 − 0.3571) ×
600` = **686**, which falls in the "Good" band. 0.3571 ≥ 0.08 → **decline**.

**11. The response.**
```json
{"probability_of_default":0.357143,"credit_score":686,"rating":"Good",
 "decision":"decline","threshold":0.08,"model_version":"1.0.0"}
```

**12. Middleware, outbound.** Stops the timer, attaches `X-Request-ID: 70703ddf` and
`X-Response-Time-ms`, and writes:
```json
{"event":"request","request_id":"70703ddf","method":"POST","path":"/predict","status":200,"latency_ms":0.66}
```

**13. Back through** Render's router, the TLS tunnel, and into your terminal.

**One last thing worth understanding.** "Good" and "decline" in the same response is not a
contradiction: the *rating* places them against the whole population, while the *decision*
compares their 35.7% PD against the 8% cut-off. And at the naive 0.5 threshold this
applicant would have been **approved** — and they defaulted. One request, carrying the
entire argument of the project.
