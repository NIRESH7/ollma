"""
Structured Excel Query Helpers.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from excel_ingestion import load_structured_payloads


def is_numeric_question(question: str) -> bool:
    numeric_keywords = {
        "sum",
        "total",
        "average",
        "mean",
        "max",
        "min",
        "highest",
        "lowest",
        "count",
        "how many",
    }
    q = question.lower()
    return any(k in q for k in numeric_keywords)


def _tokenize(text: str) -> List[str]:
    tokens = re.split(r"[\W_]+", (text or "").lower())
    return [t for t in tokens if len(t) > 1 or t.isdigit()]


def _sheet_to_dataframe(sheet_data: Dict[str, Any]) -> pd.DataFrame:
    records = sheet_data.get("records", [])
    if records:
        rows = [r.get("values", {}) for r in records if isinstance(r.get("values"), dict)]
        return pd.DataFrame(rows)
    raw_rows = sheet_data.get("raw_rows", [])
    if raw_rows:
        rows = [r.get("values", {}) for r in raw_rows if isinstance(r.get("values"), dict)]
        return pd.DataFrame(rows)
    return pd.DataFrame(sheet_data.get("data", []))


def _to_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")


def _choose_numeric_column(question: str, numeric_cols: List[str]) -> Optional[str]:
    if not numeric_cols:
        return None
    q_tokens = set(_tokenize(question))
    best_col = numeric_cols[0]
    best_score = 0
    for col in numeric_cols:
        col_tokens = set(_tokenize(col.replace("_", " ")))
        score = len(q_tokens & col_tokens)
        if score > best_score:
            best_score = score
            best_col = col
    return best_col


def _filter_rows_by_question(df: pd.DataFrame, question: str) -> pd.DataFrame:
    q_tokens = [t for t in _tokenize(question) if len(t) > 2]
    if not q_tokens or df.empty:
        return df

    text_cols = [c for c in df.columns if df[c].dtype == "object"]
    if not text_cols:
        return df

    row_scores = pd.Series(0, index=df.index, dtype="int64")
    for col in text_cols:
        col_series = df[col].astype(str).str.lower()
        for token in q_tokens:
            row_scores += col_series.str.contains(token, regex=False).astype("int64")

    filtered = df[row_scores > 0]
    return filtered if not filtered.empty else df


def run_dataframe_query(question: str, folder_name: str = "All") -> Optional[str]:
    payloads = load_structured_payloads(folder_name=folder_name)
    if not payloads:
        return None

    q = question.lower()
    for payload in payloads:
        for _, sheet_data in payload.get("sheets", {}).items():
            df = _sheet_to_dataframe(sheet_data)
            if df.empty:
                continue

            df = _filter_rows_by_question(df, q)

            numeric_cols: List[str] = []
            for col in df.columns:
                num_col = _to_numeric_series(df[col])
                if not num_col.isna().all():
                    df[col] = num_col
                    numeric_cols.append(col)

            if not numeric_cols:
                continue

            col = _choose_numeric_column(q, numeric_cols)
            if not col:
                continue

            try:
                if "sum" in q or "total" in q:
                    return f"Total {col} = {df[col].sum():,.2f}"
                if "average" in q or "mean" in q:
                    return f"Average {col} = {df[col].mean():,.2f}"
                if "max" in q or "highest" in q:
                    return f"Highest {col} = {df[col].max():,.2f}"
                if "min" in q or "lowest" in q:
                    return f"Lowest {col} = {df[col].min():,.2f}"
                if "count" in q or "how many" in q:
                    return f"Count = {len(df)}"
            except Exception as e:
                print(f"--- [NUMERIC] Error calculating {col}: {e} ---")
                continue

    return None


def get_row_by_code(code: str, folder_name: str = "All") -> Optional[str]:
    code_norm = str(code).strip().lower()
    payloads = load_structured_payloads(folder_name=folder_name)
    for payload in payloads:
        for _, sheet_data in payload.get("sheets", {}).items():
            for record in sheet_data.get("records", []):
                values = record.get("values", {})
                if any(str(v).strip().lower() == code_norm for v in values.values()):
                    return "\n".join([f"{k}: {v}" for k, v in values.items()])
    return None
