# Demo Script — 5 to 7 Minutes

A minute-by-minute plan for demonstrating the project in a lab exam or viva.
Memorise the **bolded** lines; everything else is narration you can improvise.

**Before you start:** `uv run streamlit run app.py`, open
<http://localhost:8501>, leave the browser at the Home page, and keep
`README.md` and `docs/PROJECT_REPORT.md` open in a second window.

---

## 0:00 — 0:40 · Home Dashboard

*(Page is already open.)*

> "This is a consumer complaint classification system built on real data from
> the CFPB — the US Consumer Financial Protection Bureau. The sample contains
> **forty thousand real complaint narratives** from 2015 to 2024, across eleven
> product families."

Point at the KPI row, then the category chart:

> "Credit reporting dominates the raw data at about two thirds of complaints —
> which is why the modelling step caps it, so the model doesn't just learn to
> predict one class."

Scroll to the insight panel and the recent-complaints table.

---

## 0:40 — 2:10 · Complaint Classification (the core demo)

Click **🔎 Complaint Classification** in the sidebar.

> "The model takes free text. Let me use a real example."

Click the **Collector harassment** button, then **⚡ Predict category**.

> "The text is cleaned — lower-cased, URLs and redactions removed, stop-words
> removed — then converted to TF-IDF features, and passed to Logistic
> Regression. The prediction is **Debt Collection** with high confidence."

Point at the confidence bar and the top-alternatives chart.

> "Confidence is a probability from the softmax, not a guarantee — and the app
> shows the top alternatives so the user can see when the text is ambiguous."

Open **"See the cleaned text used by the model"**:

> "This is exactly what the model saw. The same cleaning function is used in
> training and in the app — that prevents a very common silent bug."

Then type or paste a **mortgage** complaint and predict again:

> "Different vocabulary, different category — the model is genuinely
> discriminating, not guessing."

---

## 2:10 — 3:10 · Complaint Analytics

Click **📈 Complaint Analytics**.

> "Every chart here responds to the filters."

Change the date range to 2020–2024 and remove one category from the multiselect;
point at the counter under the filters:

> "The KPI row and all six charts update to the filtered population."

Type `overdraft` in **Search narrative**:

> "Free-text search across narratives — useful to find cohorts for manual
> review."

Clear the search before moving on.

---

## 3:10 — 4:10 · Dataset Explorer + CSV upload

Click **🗃️ Dataset Explorer**.

> "Three tabs: browse, upload, and data quality."

In **Browse**, search for `identity theft`, show the filtered table, then point
at the download button. Switch to **🧪 Data quality**:

> "Missing-value analysis per column, duplicate narrative count, length
> statistics, and the field reference — this is the data audit."

Switch to **⬆️ Upload & classify** and upload a small prepared CSV (keep a
3–5-row file ready; any CSV with a narrative-like column works).

> "Any CSV with a text column can be batch-classified. The app detects the text
> column, classifies up to ten thousand rows, shows the predicted distribution,
> and lets you download the results."

If time is short, skip the upload and just show the control.

---

## 4:10 — 5:20 · Model Performance (the evidence)

Click **🎯 Model Performance**.

> "Everything on this page is from the held-out test set — thirty-three hundred
> complaints the model never saw. TF-IDF was fitted on the training text only."

Point at the KPI row:

> "**Logistic Regression: 79.4% accuracy, 0.795 weighted F1.** Naive Bayes
> reached 75.8% and 0.746 — it's faster, but its independence assumption hurts
> on smaller classes, which the macro-F1 shows: 0.76 versus 0.65."

Scroll to the confusion matrix:

> "The largest confusion is debt collection versus credit reporting — eighty-three
> cases. That's not a bug; those narratives literally mention both, for example a
> collection account appearing on a credit report."

Scroll to per-class F1:

> "Mortgage and student loan are strongest — very distinctive vocabulary.
> Payday loans are weakest at 0.51 because only 294 training rows exist."

---

## 5:20 — 6:00 · About + honesty statement

Click **ℹ️ About Project**.

> "The architecture and the full pipeline are documented here, and in the
> report. One thing I want to state explicitly: the **official CFPB bulk export
> no longer contains narrative text** — I verified it against the downloaded
> file. So the sample comes from a public mirror of the same database, and the
> report documents that. Nothing is synthetic."

> "The app is fully working locally. It has not been deployed to a public URL —
> the deployment guide with exact steps is included."

---

## 6:00 — 6:40 · Code and tests (only if asked)

Show the terminal:

```bash
uv run python -m pytest tests/test_smoke.py -q
# 8 passed
```

> "Eight automated tests: all six pages render without exceptions, a full
> prediction runs, and the batch classifier works."

If asked for the training command:

```bash
uv run python train_model.py
```

> "It rebuilds the split, refits TF-IDF on train only, trains both models, and
> rewrites every artefact — the metrics are reproducible."

---

## If something goes wrong

| Problem | Recovery |
| --- | --- |
| App not running | `uv run streamlit run app.py`, wait for "Local URL" |
| Prediction button does nothing | Check the text has ≥ 4 words; click the example button first |
| "Model artefacts not found" | `uv run python train_model.py`, then sidebar **Reload model & data** |
| Charts blank | Refresh the page (Ctrl+R); Plotly needs the websocket connected |
| No internet | Everything runs locally; only Google Fonts and external links degrade |

---

## Likely follow-up questions (short answers)

- **Why Logistic Regression over Naive Bayes?** Higher weighted and macro F1;
  coefficients are interpretable per class.
- **Why TF-IDF over word counts?** Down-weights ubiquitous words; bigrams capture
  phrases.
- **How do you prevent leakage?** Fit vectorizer on train only; deduplicate
  before splitting.
- **Why cap the dominant class?** Otherwise credit reporting is two thirds of
  training and the model neglects the rest.
- **What is the weakest class?** Payday/personal loan, F1 0.512, small support.
- **Is this production-ready?** No — it is a triage aid; one complaint in five is
  misrouted and confidence must be shown.
