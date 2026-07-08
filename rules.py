"""
rules.py -- Layer 2 (deterministic) of the data-fitness assistant.

Turns a stated goal into a task type, then checks the Layer 1 profile against
what that task actually requires. Same data can be fit for one task and unfit
for another -- that is the whole point of asking the goal first.

No LLM, no network. This is CRISP-DM Business Understanding (interpret the
goal) meeting Data Understanding (the profile). The output is a list of
Findings that Layer 3 will later narrate and turn into code.

Supported tasks (v1): segmentation, classification, regression, forecasting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #

LEVELS = ("blocker", "warning", "fix", "info", "pass")


@dataclass
class Finding:
    level: str          # one of LEVELS
    title: str
    detail: str
    action: str | None = None


@dataclass
class TaskSpec:
    id: str
    label: str
    needs_target: bool
    target_kind: str | None       # "categorical" | "numeric" | None
    blurb: str


TASKS: dict[str, TaskSpec] = {
    "segmentation": TaskSpec(
        "segmentation", "Segmentation / clustering", False, None,
        "Group rows into segments from their features (unsupervised)."),
    "classification": TaskSpec(
        "classification", "Classification", True, "categorical",
        "Predict a category/label (e.g. churn yes/no) from features."),
    "regression": TaskSpec(
        "regression", "Regression", True, "numeric",
        "Predict a continuous number (e.g. revenue) from features."),
    "forecasting": TaskSpec(
        "forecasting", "Forecasting / time series", True, "numeric",
        "Predict a value forward in time from a dated history."),
}


# Keyword -> task. First match wins; time-series words beat plain 'predict'.
_GOAL_KEYWORDS = [
    ("forecasting", ["forecast", "time series", "time-series", "over time", "future",
                     "next month", "next week", "demand", "seasonal", "trend"]),
    ("segmentation", ["segment", "cluster", "clustering", "persona", "group",
                      "grouping", "tier", "unsupervised", "customer type"]),
    ("classification", ["classif", "churn", "fraud", "spam", "default", "label",
                        "category", "yes/no", "will they", "detect", "risk of"]),
    ("regression", ["regress", "predict amount", "predict price", "how much",
                    "estimate", "predict revenue", "predict sales", "continuous"]),
]


def classify_goal(text: str) -> tuple[str | None, str | None]:
    """Best-effort map free text -> task id. Returns (task_id, matched_keyword).

    Deterministic keyword match. In Phase 3 the LLM handles messy phrasing;
    for now this is a helpful default the user can override in the UI.
    """
    if not text:
        return None, None
    low = text.lower()
    for task_id, words in _GOAL_KEYWORDS:
        for w in words:
            if w in low:
                return task_id, w
    return None, None


# --------------------------------------------------------------------------- #
# Helpers over the profile + DataFrame
# --------------------------------------------------------------------------- #

def _usable_features(df, profile, target=None):
    ids = {c for c, _ in profile["id_candidates"]}
    consts = set(profile["constant"])
    excluded = ids | consts | ({target} if target else set())
    feats = [c for c in df.columns if c not in excluded]
    numeric = [c for c in feats if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in feats if not pd.api.types.is_numeric_dtype(df[c])]
    return feats, numeric, categorical


def _infer_target_kind(df, target):
    if target is None or target not in df.columns:
        return None
    col = df[target]
    if pd.api.types.is_bool_dtype(col):
        return "categorical"
    if pd.api.types.is_numeric_dtype(col):
        return "discrete-numeric" if col.nunique(dropna=True) <= 10 else "numeric"
    return "categorical"


def _shared_findings(df, profile) -> list[Finding]:
    """Checks that matter regardless of task -- pulled from Layer 1."""
    out: list[Finding] = []

    if profile["numeric_as_text"]:
        cols = ", ".join(c for c, _ in profile["numeric_as_text"])
        out.append(Finding(
            "fix", "Numbers stored as text",
            f"{cols} look numeric but are stored as text, so they are excluded "
            "from every statistic until converted.",
            "Convert with pd.to_numeric after stripping commas/percent signs."))

    if profile["constant"]:
        out.append(Finding(
            "info", "Zero-variance columns",
            f"{', '.join(profile['constant'])} carry no information.",
            "Drop before modeling."))

    if profile["shape"]["duplicate_rows"]:
        out.append(Finding(
            "warning", "Duplicate rows",
            f"{profile['shape']['duplicate_rows']:,} exact duplicate rows can bias "
            "counts and models.",
            "Review and de-duplicate unless duplicates are expected."))

    if profile["cryptic_names"]:
        cols = ", ".join(c for c, _ in profile["cryptic_names"])
        out.append(Finding(
            "info", "Cryptic column names",
            f"{cols} are hard to interpret; document them before sharing results.",
            "Rename or add a data dictionary."))

    return out


def _missing_target_finding(df, profile, target) -> Finding | None:
    m = profile["missing"]["per_column"].get(target)
    if m and m[0] > 0:
        return Finding(
            "blocker", "Missing values in the target",
            f"{m[0]:,} rows ({m[1]}%) have no value for '{target}'. Those rows "
            "cannot be used for supervised learning.",
            "Drop rows with a missing target, or fix the labels.")
    return None


# --------------------------------------------------------------------------- #
# Task-specific evaluators
# --------------------------------------------------------------------------- #

def _eval_segmentation(df, profile, target) -> list[Finding]:
    out = []
    _, numeric, categorical = _usable_features(df, profile)
    usable = len(numeric) + len(categorical)
    if usable < 2:
        out.append(Finding("blocker", "Too few usable features",
            f"Only {usable} feature(s) left after removing IDs and constant columns. "
            "Clustering needs at least two informative features.",
            "Add features or reconsider the goal."))
    else:
        out.append(Finding("pass", "Enough features to cluster",
            f"{len(numeric)} numeric and {len(categorical)} categorical feature(s) available."))

    if len(numeric) >= 2:
        # Compare spread (std), not raw range: k-means distance is dominated by
        # the highest-variance feature regardless of its min-max span. A feature
        # centred at ~1000 swamps one at ~0-1 even if its own range looks modest.
        stds = {c: float(df[c].std()) for c in numeric}
        nonzero = [s for s in stds.values() if s > 0]
        if nonzero and max(nonzero) / min(nonzero) > 15:
            biggest = max(stds, key=stds.get)
            out.append(Finding("warning", "Features on very different scales",
                f"'{biggest}' varies far more than the others, so distance-based "
                "clustering (k-means) will be dominated by it unless you standardize first.",
                "Apply StandardScaler / MinMaxScaler before clustering."))

    if categorical:
        out.append(Finding("info", "Categorical features need encoding",
            f"{', '.join(categorical)} must be encoded before distance-based clustering.",
            "One-hot encode low-cardinality categoricals; consider k-prototypes for mixed data."))

    if profile["missing"]["high_missing"]:
        out.append(Finding("warning", "High-missing columns",
            f"{', '.join(profile['missing']['high_missing'])} are heavily missing; "
            "most clustering algorithms cannot accept NaN.",
            "Impute or drop before clustering."))

    if profile["redundant_pairs"]:
        pairs = "; ".join(f"{a}~{b}" for a, b, _ in profile["redundant_pairs"])
        out.append(Finding("warning", "Redundant feature pairs",
            f"{pairs} are near-identical and will double-weight that dimension.",
            "Drop one of each pair, or reduce with PCA."))
    return out


def _eval_classification(df, profile, target) -> list[Finding]:
    out = []
    if not target or target not in df.columns:
        out.append(Finding("blocker", "No target chosen",
            "Classification needs a target column to predict.",
            "Select the column holding the label."))
        return out

    kind = _infer_target_kind(df, target)
    if kind == "numeric":
        out.append(Finding("warning", "Target looks continuous",
            f"'{target}' has many distinct numeric values -- that is usually a "
            "regression target, not a classification label.",
            "Confirm the task, or bin the target into classes."))

    mt = _missing_target_finding(df, profile, target)
    if mt:
        out.append(mt)

    vc = df[target].value_counts(dropna=True)
    if not vc.empty:
        total = int(vc.sum())
        minority = float(vc.min()) / total
        n_classes = int(vc.shape[0])
        out.append(Finding("pass", "Target is usable",
            f"'{target}' has {n_classes} classes across {total:,} labelled rows."))
        if minority < 0.10:
            out.append(Finding("warning", "Imbalanced classes",
                f"The smallest class is {minority*100:.1f}% of rows. Accuracy will be "
                "misleading and the model may ignore the minority class.",
                "Stratify the split; use class weights or resampling; report F1/recall."))
        if vc.min() < 20:
            out.append(Finding("warning", "Very small classes",
                f"The smallest class has only {int(vc.min())} rows -- too few to learn from.",
                "Collect more, merge classes, or drop the rare class."))

    _, numeric, categorical = _usable_features(df, profile, target)
    if len(numeric) + len(categorical) == 0:
        out.append(Finding("blocker", "No usable features",
            "Every non-target column is an ID or constant.",
            "Add predictive features."))

    ids = [c for c, _ in profile["id_candidates"] if c != target]
    if ids:
        out.append(Finding("warning", "ID-like columns present",
            f"{', '.join(ids)} look like identifiers and can cause leakage or noise.",
            "Exclude them from the feature set."))
    return out


def _eval_regression(df, profile, target) -> list[Finding]:
    out = []
    if not target or target not in df.columns:
        out.append(Finding("blocker", "No target chosen",
            "Regression needs a numeric target to predict.",
            "Select the column holding the value."))
        return out

    if not pd.api.types.is_numeric_dtype(df[target]):
        out.append(Finding("blocker", "Target is not numeric",
            f"'{target}' is not numeric, so it cannot be a regression target as-is.",
            "Pick a numeric target, or if it's numbers-as-text, convert it first."))
    else:
        out.append(Finding("pass", "Target is numeric",
            f"'{target}' is a valid continuous target."))
        if _infer_target_kind(df, target) == "discrete-numeric":
            out.append(Finding("info", "Target has few distinct values",
                f"'{target}' takes very few values -- classification may fit better.",
                "Confirm whether this is really a continuous quantity."))

    mt = _missing_target_finding(df, profile, target)
    if mt:
        out.append(mt)

    for a, b, r in profile["redundant_pairs"]:
        if target in (a, b):
            other = b if a == target else a
            out.append(Finding("warning", "Possible target leakage",
                f"'{other}' is near-perfectly correlated with the target (|r|={r}).",
                "Check that it isn't a proxy or a post-outcome field; drop if so."))

    _, numeric, categorical = _usable_features(df, profile, target)
    if len(numeric) + len(categorical) == 0:
        out.append(Finding("blocker", "No usable features",
            "Every non-target column is an ID or constant.",
            "Add predictive features."))
    return out


def _eval_forecasting(df, profile, target) -> list[Finding]:
    out = []
    has_dt = bool(profile["dtypes"]["datetime"]) or bool(profile["datetime_candidates"])
    if not has_dt:
        out.append(Finding("blocker", "No time column found",
            "Forecasting needs a date/time column and none was detected.",
            "Add or parse a timestamp column."))
    else:
        if profile["datetime_candidates"]:
            cols = ", ".join(c for c, _ in profile["datetime_candidates"])
            out.append(Finding("fix", "Dates stored as text",
                f"{cols} look like dates but are text.",
                "Parse with pd.to_datetime before resampling."))
        else:
            out.append(Finding("pass", "Time column present",
                f"{', '.join(profile['dtypes']['datetime'])} can index the series."))

    numeric = profile["dtypes"]["numeric"]
    if not numeric:
        out.append(Finding("blocker", "No numeric value to forecast",
            "There is no numeric column to project forward.",
            "Ensure the quantity of interest is numeric."))

    if profile["shape"]["rows"] < 30:
        out.append(Finding("warning", "Very short history",
            f"Only {profile['shape']['rows']} rows -- too little history for a reliable forecast.",
            "Gather more periods before modeling."))
    return out


_EVALUATORS = {
    "segmentation": _eval_segmentation,
    "classification": _eval_classification,
    "regression": _eval_regression,
    "forecasting": _eval_forecasting,
}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def evaluate_fitness(task_id: str, df, profile, target: str | None = None) -> dict:
    """Return findings + an overall verdict for a task against a profile."""
    if task_id not in _EVALUATORS:
        raise ValueError(f"Unknown task '{task_id}'. Options: {list(_EVALUATORS)}")

    findings = _shared_findings(df, profile) + _EVALUATORS[task_id](df, profile, target)

    if any(f.level == "blocker" for f in findings):
        verdict = ("Not ready yet", "Blocking issues must be resolved before this study is feasible.")
    elif any(f.level in ("warning", "fix") for f in findings):
        verdict = ("Usable with preparation", "No blockers, but clean the flagged items first.")
    else:
        verdict = ("Looks fit", "No blocking issues found for this task.")

    return {"task": TASKS[task_id].label, "target": target,
            "verdict": verdict, "findings": findings}


def render_fitness(result: dict) -> str:
    order = {"blocker": 0, "fix": 1, "warning": 2, "info": 3, "pass": 4}
    marks = {"blocker": "[BLOCKER]", "fix": "[FIX]", "warning": "[WARN]",
             "info": "[INFO]", "pass": "[OK]"}
    L = [f"GOAL: {result['task']}"
         + (f"   TARGET: {result['target']}" if result["target"] else ""),
         f"VERDICT: {result['verdict'][0]} -- {result['verdict'][1]}", ""]
    for f in sorted(result["findings"], key=lambda x: order.get(x.level, 9)):
        L.append(f"{marks.get(f.level, '[-]')} {f.title}")
        L.append(f"    {f.detail}")
        if f.action:
            L.append(f"    -> {f.action}")
    return "\n".join(L)
