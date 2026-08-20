# Dataset Fitness Check

An interactive tool that reads a tabular dataset and reports whether it's fit
for a planned study — automating the early **CRISP-DM** stages (Business
Understanding, Data Understanding, and the start of Data Preparation).

**Try it:** upload a `.csv`, `.tsv`, `.xlsx`, or `.parquet`, state your goal,
and get a goal-aware fitness verdict plus tailored preprocessing code.

## How it works — three layers

The split is deliberate, and it's the core design idea:

- **Layer 1 — profiler (`profiler.py`)** computes every *fact* about the data
  (types, missingness, duplicates, cryptic names, numbers-stored-as-text,
  ID-like columns, redundant pairs). Pure pandas, deterministic, no network.
- **Layer 2 — rules (`rules.py`)** maps a stated goal → the requirements that
  task needs (segmentation, classification, regression, forecasting) and checks
  the profile against them. Same data can be fit for one task and unfit for
  another. Deterministic, no network.
- **Layer 3 — LLM (`llm.py`)** turns the findings into runnable preprocessing
  code for the exact columns. It receives only the schema and findings — never
  the raw data — so it cannot invent values, and uploaded data never leaves the
  app. If no key is set or the free tier is rate-limited, the app falls back to
  the deterministic report: the LLM is an enhancement, never a dependency.

`loaders.py` isolates all file-format handling: add a format there and nothing
downstream changes.

## Enabling the AI code layer (optional)

The app works fully without it. To turn on Layer 3, add a free
[Groq](https://console.groq.com) API key to the app's secrets:

- **Streamlit Community Cloud:** Manage app → Settings → Secrets, then add
  `GROQ_API_KEY = "your_key_here"`.
- **Local:** `export GROQ_API_KEY=your_key_here` before running.

Without a key, the deterministic report is the full output.

The model is set once in `llm.py` (`GROQ_MODEL`), currently `openai/gpt-oss-120b`.
Groq deprecates free-tier models periodically (llama-3.3-70b-versatile was
decommissioned 2026-08-16) — if calls start failing, check the
[Groq models page](https://console.groq.com/docs/models) and update
`GROQ_MODEL`. `GROQ_MODEL_FALLBACK` (`qwen/qwen3.6-27b`) is a tested alternate;
`REASONING_EFFORT_BY_MODEL` in the same file must carry an entry for whichever
model is active, since gpt-oss and qwen use different `reasoning_effort`
values and a missing/wrong one can make the model spend its whole token
budget on hidden reasoning and return empty output.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
