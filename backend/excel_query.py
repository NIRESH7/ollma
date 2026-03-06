"""
Excel Hybrid Query Engine (v7 - Helper Module)
Provides helper functions for numeric checks and basic row lookups.
Updated to support the new v7 Deterministic Cache format.
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
    Simplified Numeric Engine for EXCEL_CACHE.
    Supports both legacy DataFrames and new v7 Structured JSON.
    """
    if not EXCEL_CACHE:
        return None

    q = question.lower()
    for file_path, entry in EXCEL_CACHE.items():
        if folder_name != "All" and entry.get("folder") != folder_name:
            continue

        for sheet_name, sheet_data in entry.get("sheets", {}).items():
            # Handle new v7 Structured JSON format
            if isinstance(sheet_data, dict):
                data = sheet_data.get("data", [])
                if not data: continue
                # Attempt to convert list of dicts to a numeric-friendly DataFrame
                df = pd.DataFrame(data)
            else:
                # Fallback for legacy DataFrame format
                df = sheet_data
                
            if df.empty: continue
            
            # Simple numeric column detection
            numeric_cols = df.select_dtypes(include="number").columns
            if len(numeric_cols) == 0:
                # Try to convert object columns to numeric if they look like numbers
                for col in df.columns:
                    try:
                        # Convert column to numeric, ignoring commas and symbols
                        temp_col = pd.to_numeric(df[col].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce')
                        if not temp_col.isna().all():
                            df[col] = temp_col
                    except:
                        continue
                numeric_cols = df.select_dtypes(include="number").columns
                
            if len(numeric_cols) == 0: continue
            
            # Use the first numeric column found or one that matches intent
            col = numeric_cols[0]
            
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
            records_list = data.get("data", [])
            
            for row in records_list:
                if any(str(v).strip() == str(code).strip() for v in row.values() if not isinstance(v, (dict, list))):
                    lines = [f"{k}: {v}" for k, v in row.items() if not isinstance(v, (dict, list))]
                    return "\n".join(lines)
    except:
        pass
    return None