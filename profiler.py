"""
profiler.py -- Layer 1 (deterministic) of the data-fitness assistant.

Computes an OBJECTIVE profile of a DataFrame: structure, types, missingness,
duplicates, plus a set of "insight" checks aimed at the problems that quietly
break real studies:

  * cryptic / placeholder column names        (users can't act on `v1`, `x3`)
  * numbers stored as text ("1,234", "45%")   (silently excluded from stats)
  * mixed-type columns                          (part number, part string)
  * ID columns masquerading as features         (all-unique -> useless signal)
  * zero-variance / near-constant columns
  * redundant pairs (near-perfect correlation)  -- computed with numeric_only
  * inconsistent categories ("Male" vs "male ")

No modeling, no LLM, no network. Every value here is a FACT. Layers 2 (rules)
and 3 (LLM) consume this profile and never recompute statistics themselves --
that division is what stops the model from inventing numbers.
"""

from __future__ import annotations

import re
import warnings

import pandas as pd

# --------------------------------------------------------------------------- #
# Small, single-purpose helpers. Each check is isolated so new ones drop in
# without touching the others.
# --------------------------------------------------------------------------- #

_GENERIC_NAME_PATTERNS = [
    re.compile(r"^unnamed", re.I),
    re.compile(r"^(col|column|var|variable|feature|field|attr|attribute)_?\d+$", re.I),
    re.compile(r"^x\d+$", re.I),
    re.compile(r"^\d+(\.\d+)?$"),   # a purely numeric name (headerless symptom)
    re.compile(r"^[a-z]$", re.I),   # a single letter
]


def _is_cryptic(name) -> str | None:
    """Return a reason string if a column name is likely unusable to a human."""
    n = str(name).strip()
    if not n:
        return "empty name"
    for pat in _GENERIC_NAME_PATTERNS:
        if pat.match(n):
            return "generic / placeholder name"
    # short, vowel-less token -> almost certainly an abbreviation or code
    letters = re.sub(r"[^A-Za-z]", "", n)
    if 0 < len(n) <= 4 and letters and not re.search(r"[aeiou]", n, re.I):
        return "short code / abbreviation"
    return None


def _clean_numeric_str(s: pd.Series) -> pd.Series:
    """Strip thousands separators, percent signs and whitespace before coercion."""
    return (
        s.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
    )


def _frac_numeric(s: pd.Series) -> float:
    non_null = s.dropna()
    if non_null.empty:
        return 0.0
    coerced = pd.to_numeric(_clean_numeric_str(non_null), errors="coerce")
    return float(coerced.notna().mean())


def _frac_datetime(s: pd.Series) -> float:
    non_null = s.dropna()
    if non_null.empty:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            coerced = pd.to_datetime(non_null, errors="coerce")
        except Exception:  # noqa: BLE001
            return 0.0
    return float(coerced.notna().mean())


# --------------------------------------------------------------------------- #
# Individual checks
# --------------------------------------------------------------------------- #

def _dtype_split(df: pd.DataFrame) -> dict[str, list[str]]:
    return {
        "numeric": [str(c) for c in df.select_dtypes(include="number").columns],
        "categorical": [str(c) for c in df.select_dtypes(include=["object", "category"]).columns],
        "datetime": [str(c) for c in df.select_dtypes(include=["datetime", "datetimetz"]).columns],
        "boolean": [str(c) for c in df.select_dtypes(include="bool").columns],
    }


def _missingness(df: pd.DataFrame) -> dict:
    n = len(df)
    miss = df.isna().sum()
    per_col = {
        str(c): (int(miss[c]), round(float(miss[c]) / n * 100, 1) if n else 0.0)
        for c in df.columns if miss[c] > 0
    }
    total = round(float(df.isna().values.mean()) * 100, 2) if n and df.shape[1] else 0.0
    high = [c for c, (_, pct) in per_col.items() if pct >= 40]
    empty = [c for c, (_, pct) in per_col.items() if pct >= 100]
    return {"per_column": per_col, "total_pct": total, "high_missing": high, "fully_empty": empty}


def _constant(df: pd.DataFrame):
    consts, near = [], []
    for c in df.columns:
        if df[c].nunique(dropna=False) <= 1:
            consts.append(str(c))
            continue
        top = df[c].value_counts(dropna=False, normalize=True)
        if not top.empty and top.iloc[0] >= 0.99:
            near.append((str(c), round(float(top.iloc[0]) * 100, 1)))
    return consts, near


def _duplicate_columns(df: pd.DataFrame):
    """Group columns with identical content via a cheap per-column signature.

    A hash collision could in theory pair two different columns; for a profiling
    hint that risk is acceptable and keeps this O(cols) instead of O(cols^2).
    """
    sigs: dict[int, object] = {}
    dups = []
    for c in df.columns:
        try:
            sig = int(pd.util.hash_pandas_object(df[c], index=False).sum())
        except TypeError:
            sig = hash(tuple(df[c].astype(str)))
        if sig in sigs:
            dups.append((str(sigs[sig]), str(c)))
        else:
            sigs[sig] = c
    return dups


def _id_candidates(df: pd.DataFrame):
    n = len(df)
    out = []
    for c in df.columns:
        name = str(c).lower()
        # All-unique only implies an identifier for integer or text columns.
        # A continuous float being all-unique just means it's a measurement
        # (e.g. usage_hours) -- not an ID. This avoids the false positive.
        is_float = pd.api.types.is_float_dtype(df[c])
        all_unique = n > 1 and not is_float and df[c].nunique(dropna=True) == n
        name_id = name in ("id", "index", "uuid") or name.endswith("_id") or name.startswith("id_")
        if all_unique or name_id:
            out.append((str(c), "all values unique" if all_unique else "name suggests an identifier"))
    return out


def _high_cardinality(df: pd.DataFrame, exclude=(), ratio=0.5, floor=20):
    """Text columns with many distinct values -> likely an identifier, not a
    real category. `exclude` skips columns another check already explains
    (numbers-as-text, dates-as-text) so the same column isn't accused twice.
    """
    n = len(df)
    exclude = set(exclude)
    out = []
    for c in df.select_dtypes(include=["object", "category"]).columns:
        if str(c) in exclude:
            continue
        nun = df[c].nunique(dropna=True)
        if n and nun / n > ratio and nun > floor:
            out.append((str(c), int(nun)))
    return out


def _numeric_as_text(df: pd.DataFrame, threshold=0.9):
    out = []
    for c in df.select_dtypes(include=["object"]).columns:
        frac = _frac_numeric(df[c])
        if frac >= threshold:
            out.append((str(c), round(frac * 100, 1)))
    return out


def _datetime_candidates(df: pd.DataFrame, threshold=0.8):
    out = []
    for c in df.select_dtypes(include=["object"]).columns:
        if _frac_numeric(df[c]) >= 0.9:   # a pure-number column is not a date
            continue
        frac = _frac_datetime(df[c])
        if frac >= threshold:
            out.append((str(c), round(frac * 100, 1)))
    return out


def _mixed_type(df: pd.DataFrame):
    """Object columns that are genuinely part-number, part-string."""
    out = []
    for c in df.select_dtypes(include=["object"]).columns:
        non_null = df[c].dropna()
        if non_null.empty:
            continue
        frac_num = pd.to_numeric(_clean_numeric_str(non_null), errors="coerce").notna().mean()
        if 0.1 < frac_num < 0.9:
            out.append((str(c), round(float(frac_num) * 100, 1)))
    return out


def _cryptic_names(df: pd.DataFrame):
    out = []
    for c in df.columns:
        reason = _is_cryptic(c)
        if reason:
            out.append((str(c), reason))
    return out


def _inconsistent_categories(df: pd.DataFrame):
    out = []
    for c in df.select_dtypes(include=["object"]).columns:
        non_null = df[c].dropna().astype(str)
        if non_null.empty:
            continue
        raw = non_null.nunique()
        norm = non_null.str.strip().str.lower().nunique()
        if norm < raw:
            out.append((str(c), int(raw - norm)))
    return out


def _redundant_pairs(df: pd.DataFrame, threshold=0.95):
    """Near-perfectly correlated numeric pairs.

    numeric_only=True is exactly the guard you flagged: with mixed columns a
    plain .corr() would either error or silently drop things depending on the
    pandas version. Making it explicit documents intent and is version-safe.
    """
    num = df.select_dtypes(include="number")
    if num.shape[1] < 2:
        return []
    corr = num.corr(numeric_only=True).abs()
    cols = list(corr.columns)
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if pd.notna(r) and r >= threshold:
                pairs.append((str(cols[i]), str(cols[j]), round(float(r), 3)))
    return pairs


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def profile(df: pd.DataFrame) -> dict:
    """Return the full deterministic profile as a plain dict (JSON-friendly).

    This dict is the hand-off to Layer 2 (rules) and Layer 3 (LLM).
    """
    n_rows, n_cols = df.shape
    consts, near = _constant(df)
    numeric_as_text = _numeric_as_text(df)
    datetime_candidates = _datetime_candidates(df)
    # columns already explained by another check -> don't re-flag as identifiers
    explained = {t[0] for t in numeric_as_text} | {t[0] for t in datetime_candidates}
    return {
        "shape": {
            "rows": int(n_rows),
            "cols": int(n_cols),
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
            "duplicate_rows": int(df.duplicated().sum()),
        },
        "dtypes": _dtype_split(df),
        "missing": _missingness(df),
        "constant": consts,
        "near_constant": near,
        "duplicate_columns": _duplicate_columns(df),
        "id_candidates": _id_candidates(df),
        "high_cardinality": _high_cardinality(df, exclude=explained),
        "numeric_as_text": numeric_as_text,
        "datetime_candidates": datetime_candidates,
        "mixed_type": _mixed_type(df),
        "cryptic_names": _cryptic_names(df),
        "inconsistent_categories": _inconsistent_categories(df),
        "redundant_pairs": _redundant_pairs(df),
    }


def render(p: dict) -> str:
    """Human-readable report. Each flag carries a one-line 'why it matters' so
    the output already reads like advice -- and so Layer 3 has that context to
    expand. Only non-empty sections are shown.
    """
    L: list[str] = []
    s = p["shape"]
    L.append(f"STRUCTURE  {s['rows']:,} rows x {s['cols']} cols  ({s['memory_mb']} MB)")
    if s["duplicate_rows"]:
        L.append(f"  - {s['duplicate_rows']:,} duplicate rows -> may bias counts and models.")

    d = p["dtypes"]
    L.append(
        f"TYPES      numeric={len(d['numeric'])}  categorical={len(d['categorical'])}  "
        f"datetime={len(d['datetime'])}  boolean={len(d['boolean'])}"
    )

    m = p["missing"]
    if m["total_pct"]:
        L.append(f"MISSING    {m['total_pct']}% of all cells are empty.")
        if m["fully_empty"]:
            L.append(f"  - fully empty: {', '.join(m['fully_empty'])} -> drop.")
        if m["high_missing"]:
            L.append(f"  - >=40% missing: {', '.join(m['high_missing'])} -> impute or drop.")

    def section(title, items, fmt):
        if items:
            L.append(title)
            for it in items:
                L.append("  - " + fmt(it))

    section("BLOCKER  constant columns (no variance -> useless as features):",
            p["constant"], lambda c: f"{c}")
    section("WARN     near-constant columns:",
            p["near_constant"], lambda t: f"{t[0]} ({t[1]}% one value)")
    section("WARN     likely ID columns (not real features):",
            p["id_candidates"], lambda t: f"{t[0]} ({t[1]})")
    section("WARN     high-cardinality text (looks like an identifier, not a category):",
            p["high_cardinality"], lambda t: f"{t[0]} ({t[1]} distinct)")
    section("FIX      numbers stored as text (excluded from stats until cleaned):",
            p["numeric_as_text"], lambda t: f"{t[0]} ({t[1]}% numeric)")
    section("FIX      mixed number/text in one column:",
            p["mixed_type"], lambda t: f"{t[0]} ({t[1]}% numeric)")
    section("INFO     dates stored as text (parse for time-aware handling):",
            p["datetime_candidates"], lambda t: f"{t[0]} ({t[1]}% date-like)")
    section("FIX      inconsistent category labels (case/whitespace duplicates):",
            p["inconsistent_categories"], lambda t: f"{t[0]} (~{t[1]} collapsible)")
    section("WARN     redundant near-duplicate numeric pairs (multicollinearity):",
            p["redundant_pairs"], lambda t: f"{t[0]} ~ {t[1]}  (|r|={t[2]})")
    section("WARN     duplicate columns (identical content):",
            p["duplicate_columns"], lambda t: f"{t[0]} == {t[1]}")
    section("READABILITY  cryptic column names (a human can't act on these):",
            p["cryptic_names"], lambda t: f"{t[0]} -> {t[1]}")

    return "\n".join(L)
