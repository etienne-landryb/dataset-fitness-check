"""
llm.py -- Layer 3 (LLM) of the data-fitness assistant.

Takes the structured findings from Layer 2 and asks a model on Groq's free
tier to write tailored, runnable preprocessing code. The model receives only
the goal, the column names/dtypes, and the findings -- never the raw data --
so uploaded data never leaves the app and the model cannot invent values.

Design guarantees:
  * "Code computes, model narrates": the model only writes code around facts
    that Layers 1-2 already computed. It never recomputes a statistic.
  * Graceful fallback: no key / rate limit / network error -> returns a status
    the caller uses to fall back to the deterministic report. The LLM is an
    enhancement, never a dependency.
"""

from __future__ import annotations

import os

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# llama-3.3-70b-versatile was decommissioned by Groq on 2026-08-16; every
# call to it now fails and the app falls back to the deterministic report.
GROQ_MODEL = "openai/gpt-oss-120b"   # current default: strong free-tier open model for code

# Documented alternate if gpt-oss-120b has issues (quality, availability, rate
# limits): swap GROQ_MODEL above to this string. No other code changes needed
# -- REASONING_EFFORT_BY_MODEL below carries the per-model reasoning setting
# so the swap actually behaves correctly (see next comment).
GROQ_MODEL_FALLBACK = "qwen/qwen3.6-27b"

# Both models are reasoning models that spend part of `max_tokens` on hidden
# reasoning before writing visible content -- confirmed by testing, at the
# default reasoning setting a model can burn the *entire* budget on reasoning
# and return empty content (finish_reason "length"). Each model also uses
# different valid values for `reasoning_effort` (gpt-oss: low/medium/high,
# qwen: none/default), so the value can't be hardcoded once for both. This
# map picks the lowest-reasoning setting per model, keeping the visible code
# block from being crowded out.
REASONING_EFFORT_BY_MODEL = {
    "openai/gpt-oss-120b": "low",
    "qwen/qwen3.6-27b": "none",
}

SYSTEM_PROMPT = """You are a data-preprocessing assistant embedded in a tool that has ALREADY analysed a dataset.
You are given: the user's goal, the dataframe's columns with dtypes, and a list of concrete findings about the data.
Your only job is to write a short explanation and ONE runnable Python code block that prepares the data for the stated goal.

Hard rules:
- Assume a pandas DataFrame named `df` is already loaded in memory. Never read a file.
- Use ONLY the column names provided. Never invent columns, values, or statistics.
- The findings already contain the facts; write code that acts on them, do not recompute or assert numbers.
- Address findings that carry an action; skip purely informational ones unless code genuinely helps.
- Output: 2-4 sentences of plain explanation, then exactly one ```python fenced code block, and nothing after it.
- Prefer idiomatic pandas and scikit-learn, with short inline comments. Keep it copy-paste runnable."""


def get_api_key() -> str | None:
    """Groq key from Streamlit secrets (deployed) or env var (local dev)."""
    try:
        import streamlit as st
        # accessing st.secrets can raise if no secrets file exists -> guarded
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY")


def build_user_message(goal, task_label, target, columns_info, findings) -> str:
    """Assemble the prompt from schema + findings only (no raw data)."""
    lines = [f"User goal: {goal or task_label}", f"Task type: {task_label}"]
    if target:
        lines.append(f"Target column: {target}")
    lines.append("\nColumns (name: dtype):")
    for name, dtype in columns_info:
        lines.append(f"  - {name}: {dtype}")
    lines.append("\nFindings to address:")
    for f in findings:
        action = f" | suggested action: {f.action}" if getattr(f, "action", None) else ""
        lines.append(f"  - [{f.level}] {f.title}: {f.detail}{action}")
    lines.append("\nWrite the preprocessing code now.")
    return "\n".join(lines)


def generate_code(goal, task_label, target, columns_info, findings, timeout=30):
    """Return (text, status).

    status is one of: 'ok' | 'no_key' | 'rate_limited' | 'error'.
    On any non-ok status, text is None and the caller shows the deterministic
    report as the fallback.
    """
    key = get_api_key()
    if not key:
        return None, "no_key"

    try:
        from openai import OpenAI  # Groq is OpenAI-API compatible
        client = OpenAI(api_key=key, base_url=GROQ_BASE_URL, timeout=timeout)
        # Reasoning models can emit reasoning/think content; keep it out of
        # the response entirely so `message.content` is always clean text.
        # Neither field is on the openai SDK's typed signature -> both go via
        # extra_body into the JSON request body.
        extra_body = {"include_reasoning": False}
        effort = REASONING_EFFORT_BY_MODEL.get(GROQ_MODEL)
        if effort is not None:
            extra_body["reasoning_effort"] = effort

        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_message(
                    goal, task_label, target, columns_info, findings)},
            ],
            temperature=0.2,   # low -> stable, deterministic-ish code
            max_tokens=900,
            extra_body=extra_body,
        )
        return resp.choices[0].message.content, "ok"
    except Exception as e:  # noqa: BLE001 -- classify then fall back
        msg = str(e).lower()
        if any(k in msg for k in ("rate", "429", "quota", "limit")):
            return None, "rate_limited"
        return None, "error"
