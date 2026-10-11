// Credit Risk Scorecard UI.
// Talks to the same API as everyone else: POST /predict, GET /model/info, GET /health.
// No build step and no libraries: one field spec drives the form, validation and the request body.

// ---------------------------------------------------------------------------
// Field spec. Limits mirror api/schemas.py so bad input is caught before the request.
// ---------------------------------------------------------------------------
const GROUPS = [
  { title: 'Applicant', fields: [
    { name: 'age', label: 'Age', kind: 'int', min: 18, max: 100, unit: 'years' },
    { name: 'number_of_dependants', label: 'Dependants', kind: 'int', min: 0, max: 20 },
    { name: 'years_at_current_address', label: 'Years at current address', kind: 'int', min: 0, max: 100 },
    { name: 'income', label: 'Annual income', kind: 'money', min: 1, max: 1e9, wide: true },
    { name: 'residence_type', label: 'Residence', kind: 'select', options: ['Owned', 'Mortgage', 'Rented', 'Unknown'] },
    { name: 'marital_status', label: 'Marital status', kind: 'seg', options: ['Married', 'Single'] },
    { name: 'employment_status', label: 'Employment', kind: 'seg', options: ['Salaried', 'Self-Employed'], wide: true },
  ]},
  { title: 'Loan', fields: [
    { name: 'loan_purpose', label: 'Purpose', kind: 'select', options: ['Personal', 'Home', 'Auto', 'Education'] },
    { name: 'loan_type', label: 'Type', kind: 'seg', options: ['Secured', 'Unsecured'] },
    { name: 'loan_tenure_months', label: 'Tenure', kind: 'int', min: 1, max: 480, unit: 'months', years: true },
    { name: 'sanction_amount', label: 'Sanctioned amount', kind: 'money', min: 1, max: 1e9 },
    { name: 'loan_amount', label: 'Loan amount', kind: 'money', min: 1, max: 1e9 },
    { name: 'bank_balance_at_application', label: 'Bank balance', kind: 'money', min: 0, max: 1e9 },
  ]},
  { title: 'Credit history (bureau)', fields: [
    { name: 'number_of_open_accounts', label: 'Open accounts', kind: 'int', min: 0, max: 100 },
    { name: 'number_of_closed_accounts', label: 'Closed accounts', kind: 'int', min: 0, max: 100 },
    { name: 'total_loan_months', label: 'Credit history', kind: 'int', min: 1, max: 1200, unit: 'months', years: true },
    { name: 'delinquent_months', label: 'Months delinquent', kind: 'int', min: 0, max: 1200 },
    { name: 'total_dpd', label: 'Total days past due', kind: 'int', min: 0, max: 100000 },
    { name: 'enquiry_count', label: 'Credit enquiries', kind: 'int', min: 0, max: 100 },
    { name: 'credit_utilization_ratio', label: 'Credit utilisation', kind: 'range', min: 0, max: 100, unit: '%', full: true },
  ]},
];
const FIELDS = GROUPS.flatMap(g => g.fields);

// Example applicants: the same JSON files the test suite uses (tests/fixtures/).
const PRESETS = {
  low: { age: 42, income: 2400000, number_of_dependants: 1, years_at_current_address: 12, marital_status: 'Married', employment_status: 'Salaried', residence_type: 'Owned', loan_purpose: 'Auto', loan_type: 'Secured', sanction_amount: 900000, loan_amount: 700000, loan_tenure_months: 24, bank_balance_at_application: 650000, number_of_open_accounts: 2, number_of_closed_accounts: 3, total_loan_months: 96, delinquent_months: 0, total_dpd: 0, enquiry_count: 2, credit_utilization_ratio: 12 },
  mid: { age: 24, income: 380000, number_of_dependants: 1, years_at_current_address: 17, marital_status: 'Single', employment_status: 'Salaried', residence_type: 'Owned', loan_purpose: 'Education', loan_type: 'Secured', sanction_amount: 796000, loan_amount: 683000, loan_tenure_months: 51, bank_balance_at_application: 113045, number_of_open_accounts: 1, number_of_closed_accounts: 1, total_loan_months: 60, delinquent_months: 9, total_dpd: 40, enquiry_count: 1, credit_utilization_ratio: 85 },
  high: { age: 27, income: 480000, number_of_dependants: 3, years_at_current_address: 1, marital_status: 'Single', employment_status: 'Self-Employed', residence_type: 'Rented', loan_purpose: 'Home', loan_type: 'Secured', sanction_amount: 2000000, loan_amount: 1950000, loan_tenure_months: 48, bank_balance_at_application: 40000, number_of_open_accounts: 5, number_of_closed_accounts: 1, total_loan_months: 40, delinquent_months: 18, total_dpd: 420, enquiry_count: 8, credit_utilization_ratio: 94 },
};

// Must match src/config.py: SCORE_MIN/MAX and RATING_BANDS.
const SCORE_MIN = 300, SCORE_MAX = 900;
const BANDS = [
  { from: 300, to: 500, label: 'Poor', color: 'var(--band-poor)' },
  { from: 500, to: 650, label: 'Average', color: 'var(--band-average)' },
  { from: 650, to: 750, label: 'Good', color: 'var(--band-good)' },
  { from: 750, to: 900, label: 'Excellent', color: 'var(--band-excellent)' },
];

let threshold = 0.08;     // replaced by /model/info
let live = false;         // after the first score, edits re-score automatically
let requestSeq = 0;       // ignore responses that arrive out of order
let debounceTimer = null;
let hasResult = false;     // drives the mobile mini score bar
let resultVisible = true;

const $ = id => document.getElementById(id);
const inr = new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 0 });
const pct = (p, digits = 2) => `${(p * 100).toFixed(digits)}%`;

// ---------------------------------------------------------------------------
// Build the form
// ---------------------------------------------------------------------------
function fieldHtml(f) {
  const unit = f.unit ? ` <span class="unit">(${f.unit})</span>` : f.kind === 'money' ? ' <span class="unit">(₹)</span>' : '';
  const label = `<label for="f-${f.name}">${f.label}${unit}</label>`;
  const cls = `field${f.full ? ' full' : f.wide ? ' wide' : ''}`;
  if (f.kind === 'seg') {
    const opts = f.options.map((o, i) => `<label><input type="radio" name="${f.name}" value="${o}" ${i === 0 ? 'checked' : ''}><span>${o}</span></label>`).join('');
    return `<div class="${cls}" data-field="${f.name}"><label>${f.label}</label><div class="seg" role="radiogroup" aria-label="${f.label}">${opts}</div><div class="help"></div></div>`;
  }
  if (f.kind === 'select') {
    const opts = f.options.map(o => `<option>${o}</option>`).join('');
    return `<div class="${cls}" data-field="${f.name}">${label}<select id="f-${f.name}" name="${f.name}">${opts}</select><div class="help"></div></div>`;
  }
  if (f.kind === 'range') {
    return `<div class="${cls}" data-field="${f.name}">${label}<div class="range-row">
      <input type="range" id="r-${f.name}" min="${f.min}" max="${f.max}" step="1" aria-label="${f.label} slider">
      <input type="number" id="f-${f.name}" name="${f.name}" min="${f.min}" max="${f.max}" step="1" inputmode="decimal">
      </div><div class="help"></div></div>`;
  }
  const step = f.kind === 'money' ? 1000 : 1;
  return `<div class="${cls}" data-field="${f.name}">${label}<input type="number" id="f-${f.name}" name="${f.name}" min="${f.min}" max="${f.max}" step="${step}" inputmode="numeric"><div class="help"></div></div>`;
}

$('fields').innerHTML = GROUPS.map(g =>
  `<fieldset><legend>${g.title}</legend><div class="grid">${g.fields.map(fieldHtml).join('')}</div></fieldset>`
).join('');

// Keep the utilisation slider and its number box in sync.
const range = $('r-credit_utilization_ratio'), rangeBox = $('f-credit_utilization_ratio');
range.addEventListener('input', () => { rangeBox.value = range.value; onEdit(); });
rangeBox.addEventListener('input', () => { range.value = rangeBox.value || 0; });

// ---------------------------------------------------------------------------
// Read, write and validate values
// ---------------------------------------------------------------------------
function readValues() {
  const out = {};
  for (const f of FIELDS) {
    if (f.kind === 'seg') {
      out[f.name] = document.querySelector(`input[name="${f.name}"]:checked`)?.value;
    } else if (f.kind === 'select') {
      out[f.name] = $(`f-${f.name}`).value;
    } else {
      const raw = $(`f-${f.name}`).value.trim();
      out[f.name] = raw === '' ? null : Number(raw);
    }
  }
  return out;
}

function writeValues(v) {
  for (const f of FIELDS) {
    if (!(f.name in v)) continue;
    if (f.kind === 'seg') {
      const el = document.querySelector(`input[name="${f.name}"][value="${v[f.name]}"]`);
      if (el) el.checked = true;
    } else {
      $(`f-${f.name}`).value = v[f.name];
    }
  }
  range.value = rangeBox.value;
}

// Same rules as the API (single-field bounds + the cross-field checks in schemas.py).
function validate(v) {
  const errors = {};
  for (const f of FIELDS) {
    if (f.kind === 'seg' || f.kind === 'select') continue;
    const x = v[f.name];
    if (x === null || Number.isNaN(x)) errors[f.name] = 'Required';
    else if ((f.kind === 'int') && !Number.isInteger(x)) errors[f.name] = 'Whole number';
    else if (x < f.min) errors[f.name] = `At least ${f.kind === 'money' ? inr.format(f.min) : f.min}`;
    else if (x > f.max) errors[f.name] = `At most ${f.max.toLocaleString('en-IN')}`;
  }
  if (!errors.loan_amount && !errors.sanction_amount && v.loan_amount > v.sanction_amount)
    errors.loan_amount = 'Cannot exceed the sanctioned amount';
  if (!errors.delinquent_months && !errors.total_loan_months && v.delinquent_months > v.total_loan_months)
    errors.delinquent_months = 'Cannot exceed the credit history length';
  if (!errors.number_of_open_accounts && !errors.number_of_closed_accounts && v.number_of_open_accounts + v.number_of_closed_accounts === 0)
    errors.number_of_open_accounts = 'Needs at least one open or closed account';
  if (!errors.total_dpd && v.delinquent_months === 0 && v.total_dpd > 0)
    errors.total_dpd = 'Must be 0 when months delinquent is 0';
  return errors;
}

// Helper line under each input: a readable version of the number, or the error.
function renderHelp(v, errors) {
  for (const f of FIELDS) {
    const wrap = document.querySelector(`[data-field="${f.name}"]`);
    const help = wrap.querySelector('.help');
    const err = errors[f.name];
    wrap.classList.toggle('invalid', Boolean(err));
    if (err) { help.textContent = err; continue; }
    const x = v[f.name];
    if (f.kind === 'money' && x) help.textContent = `${inr.format(x)}${x >= 1e7 ? ` · ${(x / 1e7).toFixed(2)} crore` : x >= 1e5 ? ` · ${(x / 1e5).toFixed(1)} lakh` : ''}`;
    else if (f.years && x >= 12) help.textContent = `≈ ${(x / 12).toFixed(x % 12 ? 1 : 0)} years`;
    else help.textContent = '';
  }
}

// ---------------------------------------------------------------------------
// Gauge (inline SVG, 300 -> 900 left to right over the top)
// ---------------------------------------------------------------------------
const G = { cx: 150, cy: 150, r: 118 };
const angle = s => Math.PI * (1 - (s - SCORE_MIN) / (SCORE_MAX - SCORE_MIN));
const pt = (s, r = G.r) => [G.cx + r * Math.cos(angle(s)), G.cy - r * Math.sin(angle(s))];
const arc = (a, b, r = G.r) => { const [x1, y1] = pt(a, r), [x2, y2] = pt(b, r); return `M${x1.toFixed(1)},${y1.toFixed(1)} A${r},${r} 0 0 1 ${x2.toFixed(1)},${y2.toFixed(1)}`; };

function drawGauge() {
  const cut = cutScore();
  const [tx1, ty1] = pt(cut, G.r - 13), [tx2, ty2] = pt(cut, G.r + 13);
  const labels = [500, 650, 750].map(s => { const [x, y] = pt(s, G.r + 20); return `<text x="${x.toFixed(1)}" y="${(y + 4).toFixed(1)}" text-anchor="middle" class="g-tick">${s}</text>`; }).join('');
  $('gauge').innerHTML = `
    <style>.g-tick { font: 500 10px "IBM Plex Mono", monospace; fill: var(--faint); }</style>
    ${BANDS.map((b, i) => `<path id="band-${i}" d="${arc(b.from + (i ? 2 : 0), b.to - (i < 3 ? 2 : 0))}" style="fill:none;stroke:${b.color};stroke-width:18;transition:opacity .3s"/>`).join('')}
    ${labels}
    <text x="32" y="172" text-anchor="middle" class="g-tick">300</text>
    <text x="268" y="172" text-anchor="middle" class="g-tick">900</text>
    <line x1="${tx1.toFixed(1)}" y1="${ty1.toFixed(1)}" x2="${tx2.toFixed(1)}" y2="${ty2.toFixed(1)}" style="stroke:var(--ink);stroke-width:3;stroke-linecap:round"/>
    <g class="marker" id="marker" style="transform-box:view-box;transform-origin:150px 150px;transform:rotate(0deg)">
      <circle cx="${150 - G.r}" cy="150" r="10" style="fill:var(--surface);stroke:var(--ink);stroke-width:3"/>
    </g>`;
  $('gaugeNote').textContent = `Approval line: score ${Math.round(cut)}+ (under ${pct(threshold, 0)} chance of default)`;
}

// Score is a straight line on (1 - PD), so the PD cut-off maps to one score.
const cutScore = () => SCORE_MIN + (1 - threshold) * (SCORE_MAX - SCORE_MIN);

// ---------------------------------------------------------------------------
// Render a prediction
// ---------------------------------------------------------------------------
function renderResult(r) {
  $('empty').hidden = true;
  $('result').hidden = false;
  $('error').hidden = true;

  $('score').textContent = r.credit_score;
  $('rating').textContent = r.rating;
  const deg = ((r.credit_score - SCORE_MIN) / (SCORE_MAX - SCORE_MIN)) * 180;
  $('marker').style.transform = `rotate(${deg}deg)`;
  BANDS.forEach((b, i) => { $(`band-${i}`).style.opacity = b.label === r.rating ? 1 : 0.28; });

  const approve = r.decision === 'approve';
  const pdText = r.probability_of_default < 0.001 ? '< 0.10%' : pct(r.probability_of_default);
  $('decision').className = `decision ${r.decision}`;
  $('decisionWord').textContent = approve ? 'Approve' : 'Decline';
  $('decisionWhy').textContent = approve
    ? `Chance of default (${pdText}) is below the ${pct(r.threshold, 1)} cut-off.`
    : `Chance of default (${pdText}) is at or above the ${pct(r.threshold, 1)} cut-off. Approval needs a score of about ${Math.round(cutScore())} or more.`;

  $('pdValue').textContent = pdText;
  $('miniScore').textContent = r.credit_score;
  $('miniDecision').textContent = approve ? 'Approve' : 'Decline';
  $('miniDecision').className = `mini-decision ${r.decision}`;
  hasResult = true;
  syncMini();
  const fill = $('pdFill');
  fill.style.width = `${Math.max(r.probability_of_default * 100, 0.6)}%`;
  fill.classList.toggle('over', !approve);
  $('pdCut').style.left = `${r.threshold * 100}%`;
  $('pdCutLabel').textContent = `cut-off ${pct(r.threshold, 0)}`;
}

function renderError(message, list = []) {
  const box = $('error');
  box.hidden = false;
  box.innerHTML = `<strong>${message}</strong>${list.length ? `<ul>${list.map(x => `<li>${x}</li>`).join('')}</ul>` : ''}`;
}

// ---------------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------------
async function score() {
  const values = readValues();
  const errors = validate(values);
  renderHelp(values, errors);
  if (Object.keys(errors).length) {
    renderError('Fix the highlighted fields to score this application.');
    return;
  }

  const seq = ++requestSeq;
  $('scoreBtn').disabled = true;
  $('result').classList.add('stale');
  $('updating').hidden = !live;
  try {
    const res = await fetch('/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(values) });
    const body = await res.json().catch(() => ({}));
    if (seq !== requestSeq) return; // a newer request is on its way
    if (!res.ok) {
      // 422 from Pydantic: detail is a list of {loc, msg}; other errors: a string.
      const list = Array.isArray(body.detail) ? body.detail.map(d => `${(d.loc || []).slice(1).join('.') || 'application'}: ${d.msg}`) : [];
      renderError(typeof body.detail === 'string' ? body.detail : 'The API rejected this application.', list);
      return;
    }
    renderResult(body);
    live = true;
    $('formHint').textContent = 'Live: change any value and the score updates.';
  } catch {
    if (seq === requestSeq) renderError('Could not reach the API. If the service was asleep, try again in a few seconds.');
  } finally {
    if (seq === requestSeq) {
      $('scoreBtn').disabled = false;
      $('result').classList.remove('stale');
      $('updating').hidden = true;
    }
  }
}

function onEdit() {
  document.querySelectorAll('.presets button').forEach(b => b.classList.remove('active'));
  const values = readValues();
  renderHelp(values, validate(values));
  if (!live) return;
  clearTimeout(debounceTimer);
  $('result').classList.add('stale');
  debounceTimer = setTimeout(score, 350);
}

$('form').addEventListener('submit', e => { e.preventDefault(); score(); });
$('form').addEventListener('input', e => { if (e.target !== range) onEdit(); });
$('form').addEventListener('change', e => { if (e.target.type === 'radio' || e.target.tagName === 'SELECT') onEdit(); });

document.querySelectorAll('[data-preset]').forEach(btn => btn.addEventListener('click', () => {
  writeValues(PRESETS[btn.dataset.preset]);
  document.querySelectorAll('.presets button').forEach(b => b.classList.toggle('active', b === btn));
  score();
}));

// ---------------------------------------------------------------------------
// Model card + status (GET /model/info, GET /health)
// ---------------------------------------------------------------------------
async function loadModelInfo() {
  try {
    const m = await (await fetch('/model/info')).json();
    threshold = Number(m.threshold?.value ?? threshold);
    const t = m.threshold?.test_metrics_at_threshold || {};
    const test = m.metrics?.xgboost_calibrated_test || {};
    const d = m.dataset || {};
    const fn = m.threshold?.cost_false_negative, fp = m.threshold?.cost_false_positive;
    const costRatio = fn && fp ? Math.round(fn / fp) : 10;
    const years = d.date_min && d.date_max ? ` (${d.date_min.slice(0, 4)}–${d.date_max.slice(0, 4)})` : '';
    $('modelSummary').textContent =
      `${/xgb/i.test(m.algorithm || '') ? 'XGBoost with calibrated probabilities' : (m.algorithm || 'Gradient-boosted model')}, trained on ${(d.train_rows || 0).toLocaleString('en-IN')} loans${years}. ` +
      `Uses ${m.n_features} features known at application time. Cut-off ${pct(threshold, 1)}, chosen because a missed default is assumed to cost ${costRatio}× a lost customer.`;
    const stats = [
      ['Defaults caught', t.recall != null ? pct(t.recall, 1) : '–'],
      ['Approval rate', t.approval_rate != null ? pct(t.approval_rate, 1) : '–'],
      ['Bad rate after approval', t.bad_rate_in_approved != null ? pct(t.bad_rate_in_approved) : '–'],
      ['Test ROC-AUC*', test.roc_auc != null ? test.roc_auc.toFixed(3) : '–'],
    ];
    $('stats').innerHTML = stats.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join('');
  } catch {
    $('modelSummary').textContent = 'Model details are unavailable right now.';
  }
  drawGauge();
}

async function loadHealth() {
  const el = $('status');
  try {
    const h = await (await fetch('/health')).json();
    el.className = `status ${h.model_loaded ? 'ok' : 'down'}`;
    el.querySelector('span').textContent = h.model_loaded ? `Model v${h.model_version} · online` : 'Model not loaded';
  } catch {
    el.className = 'status down';
    el.querySelector('span').textContent = 'API offline';
  }
}

// Mini score bar: shown (on small screens, via CSS) only while the result card is off-screen.
function syncMini() { $('mini').hidden = !hasResult || resultVisible; }
new IntersectionObserver(([entry]) => { resultVisible = entry.isIntersecting; syncMini(); }, { threshold: 0.25 })
  .observe(document.querySelector('.result-card'));

// Start with the low-risk example already scored, so the page shows a result straight away.
drawGauge();
writeValues(PRESETS.low);
renderHelp(readValues(), {});
loadHealth();
loadModelInfo().then(() => {
  document.querySelector('[data-preset="low"]').classList.add('active');
  score();
});
