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
from rules import TASKS, classify_goal, evaluate_fitness, render_fitness
from llm import generate_code

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

# --------------------------------------------------------------------------- #
# Layer 2: goal-aware fitness
# --------------------------------------------------------------------------- #
prof = profile(df)

st.subheader("Is it fit for your goal?")
st.caption("Describe what you want to do, or pick a task. The check adapts to it.")

goal_text = st.text_input(
    "Your goal (optional)",
    placeholder="e.g. segment my customers into groups / predict who will churn",
)
suggested, matched = classify_goal(goal_text)

task_labels = {spec.label: tid for tid, spec in TASKS.items()}
default_label = TASKS[suggested].label if suggested else list(task_labels)[0]
if suggested and matched:
    st.caption(f"Detected task from your wording (matched \u201c{matched}\u201d): **{TASKS[suggested].label}**. Change it below if that's wrong.")

chosen_label = st.selectbox(
    "Task", list(task_labels),
    index=list(task_labels).index(default_label),
)
task_id = task_labels[chosen_label]
st.caption(TASKS[task_id].blurb)

target = None
if TASKS[task_id].needs_target:
    target = st.selectbox(
        "Which column is the target (the thing to predict)?",
        list(df.columns),
    )

result = evaluate_fitness(task_id, df, prof, target=target)
headline, sub = result["verdict"]
if headline == "Not ready yet":
    st.error(f"**{headline}** — {sub}")
elif headline == "Usable with preparation":
    st.warning(f"**{headline}** — {sub}")
else:
    st.success(f"**{headline}** — {sub}")

st.code(render_fitness(result), language="text")

# --------------------------------------------------------------------------- #
# Layer 3: LLM-written preprocessing code (optional enhancement, never required)
# --------------------------------------------------------------------------- #
st.subheader("Get tailored preprocessing code")
st.caption(
    "Turns the findings above into runnable code for your exact columns. "
    "Only the column names and findings are sent to the model — never your data."
)

# Gate the API call behind a button so it fires only on click (protects the
# free rate limit), and key the cached result to the current inputs.
gen_key = (upload.name, task_id, str(target), goal_text)
if st.button("Generate preprocessing code"):
    with st.spinner("Writing preprocessing code..."):
        text, status = generate_code(
            goal_text, TASKS[task_id].label, target,
            [(c, str(df[c].dtype)) for c in df.columns],
            result["findings"],
        )
    st.session_state["llm_result"] = {"key": gen_key, "text": text, "status": status}

cached = st.session_state.get("llm_result")
if cached and cached["key"] == gen_key:
    status = cached["status"]
    if status == "ok":
        st.markdown(cached["text"])
    elif status == "no_key":
        st.info(
            "No Groq key is configured, so the deterministic report above is the full "
            "output. (To enable AI-written code, add a GROQ_API_KEY in the app secrets.)"
        )
    elif status == "rate_limited":
        st.warning(
            "The free AI tier is at its limit right now — the report above still covers "
            "everything that needs fixing. Try the code generator again later."
        )
    else:
        st.warning(
            "Couldn't reach the AI service just now; the deterministic report above is "
            "unaffected."
        )
