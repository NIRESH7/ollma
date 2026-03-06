"""
Excel Hybrid Query Engine - Full Hybrid Flow (v4)
Step 4a: Detect math questions and run pure Pandas computations
         Returns 100% accurate results - no LLM involved for math.
"""

import re
import os
import pandas as pd
from database import get_qdrant_client
import json
import re
from qdrant_client.http import models
from typing import Optional

# Re-use collection name
COLLECTION_NAME = "local_documents"

def get_row_by_code(code: str, folder_name: str = None) -> str:
    """Retrieve exact record from Qdrant JSON metadata using a code."""
    client = get_qdrant_client()
    
    # Search for the code in metadata.json_data or page_content
    # We use a broad text match on page_content but filter for high precision
    scroll_filter_conditions = [
        models.FieldCondition(
            key="page_content",
            match=models.MatchText(text=str(code))
        )
    ]
    if folder_name and folder_name != "All":
        scroll_filter_conditions.append(
            models.FieldCondition(
                key="metadata.folder",
                match=models.MatchValue(value=folder_name)
            )
        )

    search_result, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=scroll_filter_conditions
        ),
        limit=5,
        with_payload=True
    )
    
    if not search_result:
        return f"Code {code} not found."
        
    responses = []
    for p in search_result:
        # Check if the code actually exists as a value in the JSON
        json_str = p.payload.get("metadata", {}).get("json_data", "{}")
        data = json.loads(json_str)
        
        # Exact value check in any field
        row_dict = data.get("data", {})
        if any(str(v).strip() == str(code).strip() for v in row_dict.values()):
            # Format as Key: Value instead of JSON for the user/AI
            lines = [f"{k}: {v}" for k, v in row_dict.items()]
            responses.append("\n".join(lines))
            
    if not responses:
        return f"Code {code} not found in structured data."
        
    return "\n---\n".join(responses)

def list_columns(folder_name: str = None) -> list[str]:
    """Retrieve available column names from the current dataset."""
    client = get_qdrant_client()
    # Fetch a few records to extract keys
    scroll_filter_conditions = []
    if folder_name and folder_name != "All":
        scroll_filter_conditions.append(
            models.FieldCondition(
                key="metadata.folder",
                match=models.MatchValue(value=folder_name)
            )
        )

    records, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=scroll_filter_conditions
        ),
        limit=5,
        with_payload=True
    )
    
    columns = set()
    for r in records:
        json_str = r.payload.get("metadata", {}).get("json_data", "{}")
        data = json.loads(json_str)
        columns.update(data.get("data", {}).keys())
        
    return list(columns)

# Import the shared cache from excel_ingestion
try:
    from excel_ingestion import EXCEL_CACHE
except ImportError:
    EXCEL_CACHE = {}

# ── Keywords that trigger the Pandas engine ────────────────────────
NUMERIC_PATTERNS = re.compile(
    r'\b(sum of|total sum|calculate total|average of|avg of|mean of|median of|'
    r'highest|maximum|\bmax\b|lowest|minimum|\bmin\b|count of|how many|'
    r'percentage of|percent of)\b',
    re.IGNORECASE
)

EXCLUSION_WORDS = {'colour', 'color', 'name', 'date', 'code', 'what is', 'which'}


def is_numeric_question(question: str) -> bool:
    """Return True if the question requires a math computation."""
    q = question.lower()
    if any(ex in q for ex in EXCLUSION_WORDS):
        # Only override if explicit computation words are present
        if not re.search(r'\b(sum of|calculate total|average of|count of)\b', q):
            return False
    return bool(NUMERIC_PATTERNS.search(q))


def _is_numeric(val) -> bool:
    try:
        float(str(val).replace(",", ""))
        return True
    except (ValueError, TypeError):
        return False


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", ""), errors="coerce")


def _filter_rows(df: pd.DataFrame, question: str) -> pd.DataFrame:
    """Filter DataFrame rows by keywords in the question."""
    q_lower = question.lower()
    STOPWORDS = {
        'sum', 'total', 'average', 'count', 'max', 'min', 'highest', 'lowest',
        'how', 'many', 'what', 'give', 'the', 'of', 'for', 'and', 'is', 'are',
        'calculate', 'mean', 'median', 'percentage', 'percent'
    }
    tokens = [t for t in re.findall(r'\b\w{2,}\b', q_lower) if t not in STOPWORDS]

    if not tokens:
        return df

    filtered = df.copy()
    for token in tokens:
        mask = filtered.apply(
            lambda row: any(token in str(v).lower() for v in row.values if pd.notna(v)),
            axis=1
        )
        if mask.any():
            filtered = filtered[mask]
            print(f"  Filter '{token}': {len(filtered)} rows")

    return filtered


def _find_numeric_col(df: pd.DataFrame, question: str) -> Optional[str]:
    """Find the most relevant numeric column for the question."""
    q_lower = question.lower()
    numeric_cols = [
        c for c in df.columns
        if _coerce_numeric(df[c]).notna().sum() > 0
    ]
    # Prefer column whose name appears in the question
    for col in numeric_cols:
        if col.lower() in q_lower or any(w in col.lower() for w in q_lower.split()):
            return col
    return numeric_cols[0] if numeric_cols else None


def run_dataframe_query(question: str, folder_name: str = "All") -> Optional[str]:
    """
    Run a math query on the Pandas cache.
    Returns a descriptive result string or None.
    """
    if not EXCEL_CACHE:
        return None

    q_lower = question.lower()
    results = []

    for file_path, entry in EXCEL_CACHE.items():
        entry_folder = entry.get("folder", "default")
        if folder_name and folder_name != "All" and entry_folder != folder_name:
            continue

        file_name = entry.get("file_name", os.path.basename(file_path))

        for sheet_name, df in entry.get("sheets", {}).items():
            if df.empty:
                continue

            df_filtered = _filter_rows(df, question)
            target_col = _find_numeric_col(df_filtered, question)
            if not target_col:
                continue

            series = _coerce_numeric(df_filtered[target_col]).dropna()
            if series.empty:
                continue

            label = f"{file_name} / {sheet_name}"

            if re.search(r'\b(sum|total)\b', q_lower):
                results.append(f"Total {target_col} = {series.sum():,.2f} (from {len(series)} rows) [{label}]")
            elif re.search(r'\b(average|avg|mean)\b', q_lower):
                results.append(f"Average {target_col} = {series.mean():,.2f} [{label}]")
            elif re.search(r'\b(highest|maximum|max)\b', q_lower):
                results.append(f"Highest {target_col} = {series.max():,.2f} [{label}]")
            elif re.search(r'\b(lowest|minimum|min)\b', q_lower):
                results.append(f"Lowest {target_col} = {series.min():,.2f} [{label}]")
            elif re.search(r'\b(count|how many)\b', q_lower):
                results.append(f"Count of {target_col} = {len(series)} [{label}]")
            elif re.search(r'\b(median)\b', q_lower):
                results.append(f"Median {target_col} = {series.median():,.2f} [{label}]")
            else:
                results.append(
                    f"{target_col} stats [{label}]: "
                    f"Sum={series.sum():,.2f}, Avg={series.mean():,.2f}, "
                    f"Min={series.min():,.2f}, Max={series.max():,.2f}, Count={len(series)}"
                )

    if is_numeric_question(question):
        print("[ROUTE] -- Tool: get_row_by_code")
        # Extract code from question
        codes = re.findall(r'\b\d+\b', question)
        if codes:
            # For simplicity, take the first/largest number as the code
            code = max(codes, key=len) 
            result = get_row_by_code(code, folder_name)
            if "not found" not in result.lower():
                return result

    if not results:
        return None

    return "COMPUTED RESULTS FROM EXCEL DATA:\n" + "\n".join(results)
