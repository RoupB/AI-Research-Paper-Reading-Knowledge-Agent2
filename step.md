# step.md — How to Run & Test ClaimCheck

Step-by-step guide to set up, run, and test this project.
Run **all commands from the project root**: `C:\Files\IISc\Deep Learning\Project4_2`

> Shell: **PowerShell** (Windows). Python: **3.11** (per `CLAUDE.md`).

---

## Step 0 — Open a terminal at the project root

```powershell
cd "C:\Files\IISc\Deep Learning\Project4_2"
```

---

## Step 1 — Create & activate a virtual environment (recommended)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> If activation is blocked by execution policy, run once:
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`
>
> ⚠️ CLAUDE.md targets **Python 3.11**. If imports fail, use a 3.11 interpreter
> (the cached `.pyc` files show 3.14 was used previously).

---

## Step 2 — Install dependencies

```powershell
pip install -r requirements.txt
```

---

## Step 3 — Verify environment variables

The `.env` file already contains `ANTHROPIC_API_KEY` and `GITHUB_TOKEN`, so no
action is normally needed.

- For **tests only** (Steps 5–6), real keys are not required — a dummy
  `GITHUB_TOKEN=ghp_test` is enough.
- For a **live pipeline run** (Step 7), valid keys are required.

> ⚠️ Security: `.env` holds live secrets. Do not commit or share it publicly;
> rotate the keys if they have been exposed.

---

## Step 4 — Initialize the database

```powershell
python db/init_db.py
```

Creates `./data/audit.db` from `db/schema.sql`.

---

## Step 5 — Run the automated test suite (main testing step)

```powershell
pytest tests/ -v
```

Run a single test file while iterating:

```powershell
pytest tests/test_db.py -v
pytest tests/test_arxiv_tool.py -v
```

---

## Step 6 — Smoke-test the full pipeline (no internal API spend)

```powershell
pytest tests/test_pipeline_smoke.py -v
```

---

## Step 7 — Run the real pipeline (uses live Anthropic + GitHub APIs — costs tokens)

Interactive (prompts for scope via stdin):

```powershell
python main.py run --max 5
```

Non-interactive / resume from last checkpoint:

```powershell
python main.py run --max 5 --resume
```

- Start small (`--max 5`) for testing to limit cost and time.
- Reports are written to `./reports/`.
- Pipeline status advances:
  `discovered -> repo_resolved -> claims_extracted -> code_analyzed -> gaps_analyzed -> done`

---

## Step 8 — Launch the Streamlit UI

```powershell
python main.py ui
```

or directly:

```powershell
streamlit run app/streamlit_app.py
```

---

## Step 9 — (Optional) Debug logging

```powershell
$env:LOG_LEVEL="DEBUG"; python main.py run --max 3
```

---

## Minimal test flow (no API tokens spent)

```
Step 0  ->  Step 1  ->  Step 2  ->  Step 4  ->  Step 5  ->  Step 6
```

This installs deps, builds the DB, and runs the full test suite + smoke test
without spending API tokens. Only run **Step 7** for a live end-to-end run.

---

## Quick reference — all commands

```powershell
cd "C:\Files\IISc\Deep Learning\Project4_2"   # Step 0
python -m venv .venv                           # Step 1
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt                # Step 2
python db/init_db.py                           # Step 4
pytest tests/ -v                               # Step 5
pytest tests/test_pipeline_smoke.py -v         # Step 6
python main.py run --max 5                     # Step 7 (live)
python main.py ui                              # Step 8
```
