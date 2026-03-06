"""
Excel Hybrid Query Engine (v6 - Helper Module)
Provides helper functions for numeric checks and basic row lookups.
The main query logic now resides in rag.py.
"""

import re
import json
import os
import pandas as pd
from typing import Optional

# Configuration
COLLECTION_NAME = "local_documents"

def is_numeric_question(question: str) -> bool:
    """Return True if the question requires a math computation."""
    numeric_keywords = {"sum", "total", "average", "mean", "max", "min", "highest", "lowest", "count", "how many"}
    q = question.lower()
    return any(k in q for k in numeric_keywords)

try:
    from excel_ingestion import EXCEL_CACHE
except ImportError:
    EXCEL_CACHE = {}

def run_dataframe_query(question: str, folder_name: str = "All") -> Optional[str]:
    """
    Simplified Pandas query for flat tables in EXCEL_CACHE.
    Main context-aware queries are handled by rag.py.
    """
    if not EXCEL_CACHE:
        return None

    q = question.lower()
    for file_path, entry in EXCEL_CACHE.items():
        if folder_name != "All" and entry.get("folder") != folder_name:
            continue

        for sheet_name, df in entry.get("sheets", {}).items():
            if df.empty: continue
            
            # Simple numeric column detection
            numeric_cols = df.select_dtypes(include="number").columns
            if len(numeric_cols) == 0: continue
            
            col = numeric_cols[0]
            if "sum" in q or "total" in q:
                return f"Total {col} = {df[col].sum():,.2f}"
            if "average" in q or "mean" in q:
                return f"Average {col} = {df[col].mean():,.2f}"
            if "max" in q or "highest" in q:
                return f"Highest {col} = {df[col].max():,.2f}"
            if "min" in q or "lowest" in q:
                return f"Lowest {col} = {df[col].min():,.2f}"
            if "count" in q or "how many" in q:
                return f"Count = {len(df[col])}"

    return None

def get_row_by_code(code: str, folder_name: str = None) -> Optional[str]:
    """Retrieve exactly matched row from Qdrant metadata."""
    from database import get_qdrant_client
    client = get_qdrant_client()
    
    try:
        # Search for code in raw_row or json_data via scrolling
        records, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=20,
            with_payload=True
        )
        
        for r in records:
            payload = r.payload
            metadata = payload.get("metadata", {})
            if folder_name and folder_name != "All" and metadata.get("folder") != folder_name:
                continue
                
            json_str = metadata.get("json_data")
            if not json_str: continue
            
            data = json.loads(json_str)
            row = data.get("data", data) # handle direct dict or nested 'data' key
            
            if any(str(v).strip() == str(code).strip() for v in row.values() if not isinstance(v, (dict, list))):
                lines = [f"{k}: {v}" for k, v in row.items() if not isinstance(v, (dict, list))]
                return "\n".join(lines)
    except:
        pass
    return None