"""
Deterministic Spreadsheet Ingestion & Hybrid Intelligence (v7)
Features:
1. Multi-layout detection (tabular, key-value, hierarchical)
2. snake_case JSON normalization
3. Memory-cached structured data for instant lookup
"""

import os
import re
import time
import json
import warnings
import datetime
import pandas as pd
from typing import Optional, List, Dict, Any
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from database import get_qdrant_client
from langchain_community.llms import Ollama

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────
COLLECTION_NAME = "local_documents"
OLLAMA_MODEL = "qwen2:1.5b"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Global Cache for instant structured lookup
EXCEL_CACHE: Dict[str, Any] = {}

# ── Helpers ────────────────────────────────────────────────────────
def _to_snake_case(text: str) -> str:
    """Normalize string to lowercase_snake_case."""
    s = str(text).strip()
    s = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s)
    s = re.sub(r'[^a-zA-Z0-9]', '_', s).lower()
    return re.sub(r'_+', '_', s).strip('_')

def _is_numeric(val) -> bool:
    try:
        if pd.isna(val) or str(val).strip() == "": return False
        float(str(val).replace(",", "").replace("%", ""))
        return True
    except:
        return False

def _looks_like_header(row: List[Any]) -> bool:
    """A row is a header if it's mostly non-numeric strings."""
    non_empty = [v for v in row if str(v).strip()]
    if len(non_empty) < 2: return False
    string_count = sum(1 for v in non_empty if not _is_numeric(v) and len(str(v)) > 1)
    return string_count >= len(non_empty) * 0.7

def _detect_layout(df: pd.DataFrame) -> str:
    """
    Detects if the dataframe follows a 'tabular', 'key_value', or 'hierarchical' layout.
    """
    if df.empty: return "unknown"
    
    # 1. Check for tabular: Mostly columns with a header in top 5 rows
    header_found = False
    for i in range(min(5, len(df))):
        if _looks_like_header(df.iloc[i].tolist()):
            header_found = True
            break
            
    # 2. Check for key-value: Two main columns where first is mostly labels
    non_empty_cols = [c for c in df.columns if not df[c].astype(str).str.strip().eq("").all()]
    if len(non_empty_cols) == 2:
        col1 = df[non_empty_cols[0]].astype(str).str.strip()
        # If column 1 has many unique short strings and is mostly non-blank
        if col1.nunique() > len(df) * 0.5:
            return "key_value"
            
    if header_found: return "tabular"
    
    # Default to hierarchical if mixed or complex
    return "hierarchical"

# ── Ingestion Logic ──────────────────────────────────────────────
def ingest_with_intelligence_engine(file_path: str, folder_name: str = "default"):
    """
    High-Performance Deterministic Ingestion Engine.
    """
    file_name = os.path.basename(file_path)
    print(f"\n🚀 [HYBRID-ENGINE] Starting Deterministic Extraction: {file_name}", flush=True)
    start_time_total = time.time()
    
    try:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in (".xlsx", ".xls"):
            xl = pd.ExcelFile(file_path, engine="openpyxl")
            sheets = xl.sheet_names
        else:
            sheets = ["CSV_DATA"]
            xl = None
    except Exception as e:
        print(f"[ERROR] Load fail: {e}")
        return {"error": str(e)}

    all_docs = []
    file_structured_data = {"folder": folder_name, "sheets": {}}

    for sheet_name in sheets:
    
        print(f"\n  🔍 [FILE] Scanning Sheet: {sheet_name}", flush=True)
        try:
            if xl:
                df = xl.parse(sheet_name, header=None).fillna("")
            else:
                df = pd.read_csv(file_path, header=None).fillna("")
        except: continue

        layout = _detect_layout(df)
        print(f"    LOG: STRUCTURE_DETECTED | Layout: {layout}")

        sheet_json = {"layout": layout, "data": []}
        
        # --- DETERMINISTIC PARSERS ---
        if layout == "tabular":
            header_idx = -1
            for i in range(5):
                if _looks_like_header(df.iloc[i].tolist()):
                    header_idx = i; break
            if header_idx >= 0:
                headers = [_to_snake_case(h) for h in df.iloc[header_idx]]
                data_df = df.iloc[header_idx+1:].copy()
                data_df.columns = headers
                sheet_json["data"] = data_df.to_dict(orient="records")
                
        elif layout == "key_value":
            # Extract Label -> Value pairs
            for idx, row in df.iterrows():
                vals = [str(v).strip() for v in row if str(v).strip()]
                if len(vals) >= 2:
                    sheet_json["data"].append({_to_snake_case(vals[0]): vals[1]})

        else: # Hierarchical fallback
            # Simple hierarchical preservation (Section -> Content)
            current_section = "General"
            for idx, row in df.iterrows():
                vals = [str(v).strip() for v in row if str(v).strip()]
                if len(vals) == 1:
                    current_section = vals[0]
                elif len(vals) > 1:
                    sheet_json["data"].append({
                        "section": _to_snake_case(current_section),
                        "row_data": [str(v) for v in vals]
                    })

        file_structured_data["sheets"][sheet_name] = sheet_json
        
        # --- EMBEDDING PREPARATION ---
        # Flatten to segments: [Sheet > Section > Key: Value]
        segments = _flatten_structured_json(sheet_json, sheet_name)
        for seg in segments:
            all_docs.append(Document(
                page_content=seg,
                metadata={
                    "file_name": file_name,
                    "sheet": sheet_name,
                    "folder": folder_name,
                    "json_data": json.dumps(sheet_json) # Store full structure context
                }
            ))

    # --- MEMORY CACHING for Instant Lookup ---
    EXCEL_CACHE[file_path] = file_structured_data
    
    # --- STORAGE ---
    if all_docs:
        print(f"[QDRANT] Starting Batch Insert | Docs: {len(all_docs)}")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        client = get_qdrant_client()
        vector_store = QdrantVectorStore(client=client, embedding=embeddings, collection_name=COLLECTION_NAME)
        vector_store.add_documents(all_docs, batch_size=64)
    
    elapsed = int(time.time() - start_time_total)
    print(f"🏆 INGESTION COMPLETE | TIME: {elapsed}s")
    return {"status": "completed"}

def _flatten_structured_json(sheet_json: Dict, sheet_name: str) -> List[str]:
    segments = []
    for item in sheet_json.get("data", []):
        if isinstance(item, dict):
            parts = [f"Sheet: {sheet_name}"]
            for k, v in item.items():
                parts.append(f"{k}: {v}")
            segments.append(" | ".join(parts))
    return segments
