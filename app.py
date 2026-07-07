"""
app.py -- Phase 1 deploy shell.

Deliberately minimal: upload -> load -> show ingestion warnings -> show the
deterministic profile. No goal input, no LLM yet (those are Phases 2 and 3).
The point of Phase 1 is a real, public, working tool -- deploy it now while
it is still simple, so deployment never becomes the scary final step.
"""

import streamlit as st

from loaders import SUPPORTED_EXTENSIONS, load_dataframe
from profiler import profile, render

st.set_page_config(page_title="Dataset Fitness Check", layout="wide")

st.title("Dataset Fitness Check")
st.caption(
    "Upload a tabular file and get an objective health check before you build a "
    "study on it. Supported: " + ", ".join(sorted(SUPPORTED_EXTENSIONS)) + "."
)
st.caption(
    "Assumptions (v1): Excel files are read from the **first sheet** with headers "
    "in **row 1**. Cleaning messy multi-sheet workbooks is planned for a later version."
)

# The uploader's `type` list both states the supported formats and greys out
# everything else -- the same SUPPORTED_EXTENSIONS that drives ingestion.
upload = st.file_uploader(
    "Choose a dataset",
    type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
)

if upload is None:
    st.info("Waiting for a file. Try a CSV or an .xlsx export.")
    st.stop()

result = load_dataframe(upload, filename=upload.name)

for err in result.errors:
    st.error(err)
if not result.ok:
    st.stop()

for warn in result.warnings:
    st.warning(warn)

df = result.df
st.success(f"Loaded **{upload.name}** as {result.fmt}: {df.shape[0]:,} rows x {df.shape[1]} columns.")

st.subheader("Preview")
st.dataframe(df.head(20), use_container_width=True)

st.subheader("Fitness report")
st.code(render(profile(df)), language="text")

st.caption(
    "Next versions: tell the tool your goal (segmentation, classification, "
    "forecasting...) for goal-aware fitness checks, then get tailored "
    "preprocessing code."
)
