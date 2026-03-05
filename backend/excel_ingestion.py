"""
Excel Ingestion Engine - Full Hybrid Flow (v4)
Step 1: Read every sheet → Pandas DataFrame
Step 2: Flatten every row into KEY=VALUE format
Step 3: Store in Qdrant (search) AND Pandas Cache (math)
"""

import os
import re
import warnings
import datetime
import pandas as pd
from typing import Optional, Callable
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from database import get_qdrant_client

warnings.filterwarnings("ignore")

COLLECTION_NAME = "local_documents"
# STEP 3b: Pandas Cache — {file_path: {file_name, folder, sheets: {sheet_name: DataFrame}}}
EXCEL_CACHE: dict = {}


def _get_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def _is_numeric(val) -> bool:
    try:
        if pd.isna(val):
            return False
        float(str(val).replace(",", ""))
        return True
    except (ValueError, TypeError):
        return False


def _looks_like_header(row_vals: list) -> bool:
    """A header row has mostly short non-numeric strings."""
    if not row_vals:
        return False
    non_empty = [v for v in row_vals if str(v).strip() not in ("", "nan", "None")]
    if len(non_empty) < 2:
        return False
    string_count = sum(1 for v in non_empty if not _is_numeric(v) and 1 < len(str(v)) < 60)
    return string_count >= len(non_empty) * 0.6


def _looks_like_section_header(row_vals: list) -> bool:
    """A section header is a single prominent label (e.g., 'DIA-26')."""
    non_empty = [str(v).strip() for v in row_vals if str(v).strip() not in ("", "nan", "None")]
    return len(non_empty) == 1 and len(non_empty[0]) < 40


def ingest_excel_file(
    file_path: str,
    folder_name: str = "default",
    progress_callback: Optional[Callable] = None,
):
    """
    Full Hybrid Excel Ingestion:
    1. Read every sheet
    2. Track section headers (e.g. DIA-26)
    3. Detect column header rows dynamically
    4. Flatten each data row → KEY=VALUE string
    5. Store in Qdrant + Pandas Cache
    """
    file_name = os.path.basename(file_path)
    print(f"\n📂 [EXCEL-INGEST] Processing: {file_name}", flush=True)

    try:
        xl = pd.ExcelFile(file_path, engine="openpyxl")
    except Exception as e:
        print(f"❌ [EXCEL-INGEST] Failed to open file: {e}")
        return {"error": str(e)}

    all_chunks: list[Document] = []
    # sheet_name -> list of dicts (one per data row)
    all_sheet_records: dict[str, list[dict]] = {}

    for s_idx, sheet_name in enumerate(xl.sheet_names):
        print(f"  📄 Sheet [{s_idx+1}/{len(xl.sheet_names)}]: {sheet_name}")
        if progress_callback:
            progress_callback(s_idx + 1, len(xl.sheet_names))

        try:
            raw_df = xl.parse(sheet_name, header=None, dtype=str)
            raw_df = raw_df.fillna("")
        except Exception as e:
            print(f"  ⚠️ [EXCEL-INGEST] Could not parse sheet '{sheet_name}': {e}")
            continue

        rows = raw_df.values.tolist()
        
        current_section = sheet_name      # Will be updated if we find a header like "DIA-26"
        current_columns: list[str] = []
        sheet_records: list[dict] = []

        for row_idx, raw_row in enumerate(rows):
            # Clean the row
            row_vals = [str(v).strip() for v in raw_row]
            row_vals_ne = [v for v in row_vals if v not in ("", "nan", "None")]
            
            if not row_vals_ne:
                continue  # Skip completely empty rows

            # ── Detect Section Header (e.g. "DIA-26") ──
            if _looks_like_section_header(row_vals_ne):
                current_section = row_vals_ne[0]
                current_columns = []  # Reset columns for new section
                print(f"    📌 Section: {current_section}")
                continue

            # ── Detect Column Header Row ──
            if _looks_like_header(row_vals_ne) and not any(_is_numeric(v) for v in row_vals_ne):
                # This row becomes our column names
                current_columns = [v for v in row_vals if v not in ("", "nan", "None")]
                print(f"    📋 Headers detected: {current_columns[:5]}...")
                continue

            # ── Data Row: Flatten to KEY=VALUE ──
            if current_columns:
                # Map column names to values
                record = {
                    "FILE": file_name,
                    "SHEET": sheet_name,
                    "SECTION": current_section,
                }
                has_data = False
                for col_idx, col_name in enumerate(current_columns):
                    if col_idx < len(row_vals):
                        val = row_vals[col_idx]
                        if val not in ("", "nan", "None"):
                            record[col_name] = val
                            has_data = True
                if not has_data:
                    continue
                    
                sheet_records.append(record)

                # ── STEP 2: Create KEY=VALUE string ──
                kv_string = " | ".join(f"{k}={v}" for k, v in record.items())
                
                all_chunks.append(Document(
                    page_content=kv_string,
                    metadata={
                        "source": file_path,
                        "file_name": file_name,
                        "sheet": sheet_name,
                        "section": current_section,
                        "folder": folder_name,
                        "source_type": "excel",
                        "timestamp": datetime.datetime.utcnow().isoformat(),
                    }
                ))
            else:
                # No columns yet — treat as a text chunk
                text = " | ".join(row_vals_ne)
                all_chunks.append(Document(
                    page_content=f"FILE={file_name} | SHEET={sheet_name} | SECTION={current_section} | TEXT={text}",
                    metadata={
                        "source": file_path,
                        "file_name": file_name,
                        "sheet": sheet_name,
                        "section": current_section,
                        "folder": folder_name,
                        "source_type": "excel_text",
                        "timestamp": datetime.datetime.utcnow().isoformat(),
                    }
                ))

        if sheet_records:
            all_sheet_records[sheet_name] = sheet_records

    # ── STEP 3b: Store in Pandas Cache ──
    if all_sheet_records:
        EXCEL_CACHE[file_path] = {
            "file_name": file_name,
            "folder": folder_name,
            "sheets": {
                sn: pd.DataFrame(recs)
                for sn, recs in all_sheet_records.items()
            }
        }
        total_records = sum(len(v) for v in all_sheet_records.values())
        print(f"  ✅ Pandas Cache: {total_records} rows across {len(all_sheet_records)} sheets")

    if not all_chunks:
        return {"error": "No data detected in Excel. Please check the file structure."}

    # ── STEP 3a: Store in Qdrant ──
    try:
        embeddings = _get_embedding_model()
        client = get_qdrant_client()
        vector_store = QdrantVectorStore(
            client=client, embedding=embeddings, collection_name=COLLECTION_NAME
        )
        vector_store.add_documents(all_chunks)
        post_count = client.count(collection_name=COLLECTION_NAME).count
        print(f"  ✅ Qdrant: {len(all_chunks)} chunks stored. Total vectors: {post_count}", flush=True)
    except Exception as e:
        print(f"❌ [EXCEL-INGEST] Qdrant Error: {e}")
        return {"error": str(e)}

    total_rows = sum(
        len(df) for f in EXCEL_CACHE.values() for df in f["sheets"].values()
    )
    return {
        "status": "Success",
        "num_chunks": len(all_chunks),
        "total_rows": total_rows,
        "total_vectors": post_count,
    }
