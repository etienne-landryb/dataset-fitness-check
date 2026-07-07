# Dataset Fitness Check

An interactive tool that reads a tabular dataset and reports whether it's fit
for a planned study — automating the early **CRISP-DM** stages (Data
Understanding and the start of Data Preparation).

**Try it:** upload a `.csv`, `.tsv`, `.xlsx`, or `.parquet` and read the report.

## Design note

The system is layered, and the split is deliberate:

- **Layer 1 — profiler (`profiler.py`)** computes every *fact* about the data
  (types, missingness, duplicates, cryptic names, numbers-stored-as-text,
  ID-like columns, redundant pairs). Pure pandas, deterministic, no network.
- **Layer 2 — rules** (next) map a stated goal → the requirements that goal
  needs, and check the profile against them.
- **Layer 3 — LLM** (next) turns the profile into readable advice and tailored
  preprocessing code. It only *narrates* Layer 1's numbers — it never computes
  them, so it can't invent statistics.

`loaders.py` isolates all file-format handling: add a format there and nothing
downstream changes.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
