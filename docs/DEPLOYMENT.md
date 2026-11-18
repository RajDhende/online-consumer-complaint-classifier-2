# Deployment Guide — GitHub + Streamlit Community Cloud

This guide is written for a first-time deployer. It assumes no prior cloud
experience. Every command is copy-pasteable.

> **Status:** the project runs locally and has been verified there. It has
> **not** been deployed to a public URL. Follow the steps below to deploy it
> yourself.

---

## 0. What you need

| Requirement | Notes |
| --- | --- |
| A GitHub account | Free — <https://github.com/signup> |
| Git (or GitHub Desktop) | <https://git-scm.com/downloads> — or use GitHub's web uploader |
| A Streamlit Community Cloud account | Free — sign in with GitHub at <https://share.streamlit.io> |
| This project folder | The one containing `app.py` |

The repository is already deployment-ready:

- `requirements.txt` — pinned, pip-installable dependencies
- `data/complaints.csv.gz` — 12 MB dataset (gzip is read natively by pandas)
- `model/` — 7 MB of trained artefacts, so **no retraining is needed**
- `.streamlit/config.toml` — theme, headless server, 200 MB upload limit
- `.gitignore` — excludes the 368 MB ZIP and 788 MB parquet

---

## 1. Create the GitHub repository

1. Go to <https://github.com/new>.
2. **Repository name:** `consumer-complaint-classifier` (any name works).
3. Visibility: **Public** (required for free Streamlit Cloud).
4. Do **not** initialise with a README (you already have one).
5. Click **Create repository**.

---

## 2. Upload the project

### Option A — Git command line (recommended)

From the project folder:

```bash
git init
git add .
git commit -m "Online consumer complaint classification system"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/consumer-complaint-classifier.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username. Git will ask you to sign in;
use a personal access token as the password if prompted.

Verify with:

```bash
git status          # should say "nothing to commit, working tree clean"
git remote -v       # should show your repository
```

### Option B — GitHub web uploader

1. On the new repository page, click **uploading an existing file**.
2. Drag the project files in — **but skip** `complaints.csv.zip`,
   `data/cfpb_full.parquet`, `.venv/`, `__pycache__/`, `.pytest_cache/`.
3. Keep the folder structure: `app.py` must be at the repository root, with
   `data/`, `model/`, `docs/`, `.streamlit/` beside it.
4. Commit.

> **File-size warning:** GitHub rejects files over 100 MB and the web uploader
> stops at 25 MB. The included ZIP/parquet files exceed both limits. They are
> raw download caches only — the app never needs them because
> `data/complaints.csv.gz` and `model/` are committed.

### Check before deploying

```bash
ls app.py requirements.txt data/complaints.csv.gz model/trained_model.joblib
```

All four must exist in the repository.

---

## 3. Deploy on Streamlit Community Cloud

1. Sign in at <https://share.streamlit.io> with your GitHub account and authorise
   access to your repositories.
2. Click **Create app** (or **New app**).
3. Choose **Deploy a public app from GitHub**.
4. Fill in:

   | Field | Value |
   | --- | --- |
   | Repository | `YOUR_USERNAME/consumer-complaint-classifier` |
   | Branch | `main` |
   | Main file path | `app.py` |
   | App URL | your preferred subdomain, e.g. `complaint-classifier` |

5. Open **Advanced settings**:
   - **Python version:** choose the newest available (3.13 or 3.14).
     Python 3.12 and newer are supported by the pinned packages.
   - **Secrets:** none required.
6. Click **Deploy**. The first build installs the packages in
   `requirements.txt` and typically takes **3–8 minutes**.

When the build finishes, the app opens at
`https://YOUR_SUBDOMAIN.streamlit.app`.

---

## 4. Verify the deployment

Check each item once:

- [ ] Home Dashboard shows **40,000 complaints** and **11 categories** in the
      sidebar and KPI cards.
- [ ] Sidebar shows the model name **Logistic Regression**.
- [ ] Complaint Classification → click **Collector harassment** →
      **Predict category** returns Debt Collection.
- [ ] Model Performance shows Accuracy **79.4%** and Weighted F1 **79.5%**.
- [ ] Dataset Explorer → Upload tab accepts a small CSV and returns downloads.

If the sidebar says *"Model artefacts not found"* or *"dataset is missing"*,
the upload step skipped files — return to Section 2.

---

## 5. Updating the deployed app

```bash
git add .
git commit -m "Describe the change"
git push
```

Streamlit Cloud redeploys automatically on every push to `main`. You can also
click **Reboot** in the app's menu (⋮) to restart without a code change.

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| Build fails with `No module named …` | Missing or incomplete `requirements.txt` | Ensure it was committed at the repo root (it pins every dependency). |
| Build fails resolving `requires-python` | Streamlit Cloud's Python is older than 3.12 | Select Python 3.12 or newer in Advanced settings. |
| `FileNotFoundError: data/complaints.csv` | Dataset not committed | Commit `data/complaints.csv.gz`; the loader reads gzip automatically. |
| "Model artefacts not found" | `model/` folder missing | Commit the whole `model/` directory. |
| App is very slow on first load | Cold start: 40k-row CSV + model load | Wait ~10–20 seconds; subsequent loads are cached. |
| `Error: File upload exceeds limit` | Upload > 200 MB | Limit is set in `.streamlit/config.toml`; keep demonstration CSVs small. |
| Charts render but text is plain | Google Fonts blocked on the viewer's network | Cosmetic only; the app falls back to system fonts. |

---

## 7. What deployment does *not* do

- It does not retrain the model. Training happens offline via
  `uv run python train_model.py`.
- It does not download the 788 MB parquet. The committed 12 MB gzip sample is
  the working dataset; `fetch_data.py` is only needed to rebuild it.
- It does not expose any private data. Only public CFPB complaint text is used.

---

## 8. Alternative: run locally for the viva

If the network is unavailable during the examination, run locally — this is the
verified path:

```bash
uv sync
uv run streamlit run app.py
```

Then open <http://localhost:8501>. With `pip` instead:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
