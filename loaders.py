"""
loaders.py -- Ingestion layer for the CRISP-DM data-fitness assistant.

One job: turn an uploaded file (any supported tabular format) into a clean
pandas DataFrame, and surface ingestion-level problems *before* profiling --
the kind that silently corrupt every downstream step:

  * unsupported / unreadable formats
  * a file whose header row is actually data (the classic `iris` mistake)
  * a stray row-index saved as an "Unnamed: 0" column

Everything downstream (profiler, rule engine, LLM) receives a DataFrame and
never needs to know the original format. Add a format here and nothing else
in the pipeline changes -- that is why this layer is deliberately small.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import pandas as pd

# Extension -> human-readable format name. This dict is also the single source
# of truth for the UI's "supported formats" list and the uploader's filter.
SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".csv": "CSV",
    ".tsv": "TSV",
    ".txt": "delimited text",
    ".xlsx": "Excel",
    ".xls": "Excel (legacy)",
    ".parquet": "Parquet",
}


@dataclass
class LoadResult:
    """Outcome of an ingestion attempt.

    `df` is None when the file could not be read at all (see `errors`).
    `warnings` are non-fatal: the data loaded, but something looks off and
    the user should be told before they trust the profile.
    """

    df: pd.DataFrame | None
    fmt: str
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.df is not None and not self.errors


def _read_by_extension(file, ext: str, sheet_name=0) -> pd.DataFrame:
    """Dispatch on extension. The ONLY place that knows about file formats."""
    if ext in (".csv", ".txt"):
        # sep=None + python engine sniffs the delimiter (comma / semicolon / etc.)
        return pd.read_csv(file, sep=None, engine="python")
    if ext == ".tsv":
        return pd.read_csv(file, sep="\t")
    if ext in (".xlsx", ".xls"):
        # v1 assumption, stated in the UI: first sheet, header in row 1.
        return pd.read_excel(file, sheet_name=sheet_name)
    if ext == ".parquet":
        return pd.read_parquet(file)
    raise ValueError(f"Unsupported extension: {ext}")


def _looks_like_headerless(columns) -> bool:
    """True when most column *names* parse as numbers.

    A genuine header ("age", "price") is rarely numeric. When the names are
    mostly numbers, the first data row was probably consumed as the header --
    exactly what happens loading a dataset that ships without column names
    (the `iris` case you hit).
    """
    names = [str(c) for c in columns]
    if not names:
        return False
    numeric_like = 0
    for n in names:
        try:
            float(n.replace(",", "").strip())
            numeric_like += 1
        except (ValueError, AttributeError):
            pass
    return numeric_like / len(names) >= 0.5


def _unnamed_columns(columns) -> list[str]:
    return [str(c) for c in columns if str(c).startswith("Unnamed:")]


def load_dataframe(file, filename: str | None = None, sheet_name=0) -> LoadResult:
    """Load `file` into a DataFrame and run ingestion-level sanity checks.

    `file` may be a path or a file-like object (e.g. a Streamlit upload).
    `filename` is used to detect the extension when `file` has no `.name`.
    """
    name = filename or getattr(file, "name", "") or ""
    ext = os.path.splitext(name)[1].lower()

    if ext not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        return LoadResult(
            None, "unknown",
            errors=[f"Unsupported format '{ext or 'unknown'}'. Supported: {supported}."],
        )

    try:
        df = _read_by_extension(file, ext, sheet_name=sheet_name)
    except Exception as e:  # noqa: BLE001 -- surface any reader failure to the user
        return LoadResult(
            None, SUPPORTED_EXTENSIONS[ext],
            errors=[f"Could not read the file as {SUPPORTED_EXTENSIONS[ext]}: {e}"],
        )

    result = LoadResult(df=df, fmt=SUPPORTED_EXTENSIONS[ext])

    if df.shape[1] == 0:
        result.errors.append("No columns detected -- check the delimiter or the sheet.")
        return result
    if df.shape[0] == 0:
        result.warnings.append("The file parsed but contains no data rows.")

    if _looks_like_headerless(df.columns):
        result.warnings.append(
            "Header check: most column names look like numbers, so your first data "
            "row may have been read as the header. This is the classic issue when a "
            "dataset ships without column names (e.g. iris loaded raw). If so, reload "
            "the file telling the reader there is no header, then assign names."
        )

    unnamed = _unnamed_columns(df.columns)
    if unnamed:
        shown = ", ".join(unnamed[:3]) + ("..." if len(unnamed) > 3 else "")
        result.warnings.append(
            f"Found {len(unnamed)} 'Unnamed' column(s) ({shown}). This usually means a "
            "row index was saved into the file -- consider dropping it or setting it as "
            "the index."
        )

    return result
