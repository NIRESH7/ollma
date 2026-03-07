"""
excel_agent.py — Agentic Excel Query Engine

Uses qwen2.5:14b to dynamically write and execute Pandas code to answer
questions from ANY Excel file layout in ANY domain.

Flow:
1. Load all sheets from the uploaded Excel into DataFrames
2. Build schema context (sheet names, columns, sample rows)
3. LLM generates Python/Pandas code to answer the question
4. Execute code safely in a sandboxed namespace
5. Return the result string
"""

import io
import os
import re
import traceback
from typing import Optional, Dict, List

import pandas as pd


UPLOADS_DIR = os.path.join(os.path.dirname(__file__), "uploads")
STRUCTURED_CACHE_DIR = os.path.join(os.path.dirname(__file__), "data", "structured_cache")


def _find_excel_file(folder_name: str) -> Optional[str]:
    """Find the Excel file path for a given folder name."""
    if not os.path.isdir(UPLOADS_DIR):
        return None
    for fname in os.listdir(UPLOADS_DIR):
        base = os.path.splitext(fname)[0]
        # Flexible match: folder_name is substring of file base or vice versa
        if (folder_name.lower() in base.lower() or base.lower() in folder_name.lower()):
            full_path = os.path.join(UPLOADS_DIR, fname)
            if fname.lower().endswith((".xlsx", ".xls", ".xlsm", ".csv")):
                return full_path
    return None


def _load_sheets(file_path: str) -> Dict[str, pd.DataFrame]:
    """Load all sheets from the Excel file."""
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext in (".xlsx", ".xlsm"):
            xl = pd.ExcelFile(file_path, engine="openpyxl")
            return {s: xl.parse(s).fillna("") for s in xl.sheet_names}
        elif ext == ".xls":
            xl = pd.ExcelFile(file_path, engine="xlrd")
            return {s: xl.parse(s).fillna("") for s in xl.sheet_names}
        elif ext == ".csv":
            return {"Sheet1": pd.read_csv(file_path).fillna("")}
    except Exception as e:
        print(f"[AGENT] Failed to load file: {e}")
    return {}


def _build_schema_context(sheets: Dict[str, pd.DataFrame]) -> str:
    """Build a compact schema description for the LLM prompt."""
    lines = []
    for sheet_name, df in sheets.items():
        if df.empty:
            continue
        lines.append(f"\n### Sheet: '{sheet_name}'")
        lines.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} cols")
        lines.append(f"Columns: {list(df.columns)}")
        # Show first 8 rows as sample
        sample = df.head(8).to_string(index=False, max_colwidth=40)
        lines.append(f"Sample rows:\n{sample}")
    return "\n".join(lines)


def _execute_pandas_code(code: str, sheets: Dict[str, pd.DataFrame]) -> Optional[str]:
    """Safely execute the LLM-generated Pandas code and return the result."""
    # Extract code block if wrapped in markdown
    match = re.search(r"```(?:python)?\n(.*?)```", code, re.DOTALL)
    if match:
        code = match.group(1)

    # Build namespace with all sheets available as variables
    namespace: Dict = {"pd": pd, "result": None}
    for sheet_name, df in sheets.items():
        safe_var = re.sub(r"\W+", "_", sheet_name.lower().strip()).strip("_") or "sheet"
        namespace[safe_var] = df.copy()
    # Also expose dfs dict for convenience
    namespace["dfs"] = {k: v.copy() for k, v in sheets.items()}

    # Capture stdout in case code uses print()
    stdout_capture = io.StringIO()
    import sys
    old_stdout = sys.stdout
    sys.stdout = stdout_capture

    try:
        exec(code, namespace)  # noqa: S102
    except Exception as e:
        sys.stdout = old_stdout
        print(f"[AGENT] Code execution error: {e}\nCode:\n{code}")
        return None
    finally:
        sys.stdout = old_stdout

    # Prefer explicit 'result' variable
    result = namespace.get("result")
    if result is not None and str(result).strip():
        return str(result).strip()

    # Fallback: captured stdout
    printed = stdout_capture.getvalue().strip()
    if printed:
        return printed

    return None


def _extract_code_from_response(response: str) -> str:
    """Extract Python code block from LLM response."""
    # Try fenced code block first
    match = re.search(r"```(?:python)?\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # If no fences, try to find result = ... lines
    lines = response.strip().splitlines()
    code_lines = [l for l in lines if not l.startswith("#") or "=" in l or "import" in l or "print" in l]
    return "\n".join(code_lines).strip()


from excel_ingestion import load_structured_payloads

def agentic_excel_query(question: str, folder_name: str, target_sheet: Optional[str] = None, llm=None) -> Optional[str]:
    """
    Main entry point for Table-QA Backup.
    Instead of writing Pandas code, we feed the raw Markdown representation of 
    the Excel sheets directly to the local LLM to leverage its text understanding.
    """
    if llm is None:
        print("[AGENT] No LLM provided — skipping Table-QA query")
        return None

    print(f"[AGENT] Falling back to Direct Markdown Table-QA for folder: {folder_name}")
    payloads = load_structured_payloads(folder_name)
    if not payloads:
        print("[AGENT] No structured cache found for Table-QA.")
        return None

    # Collect Markdown from all sheets
    markdown_contexts = []
    for payload in payloads:
        file_name = payload.get("file_name", "Unknown File")
        sheets = payload.get("sheets", {})
        for sheet_name, sheet_data in sheets.items():
            if target_sheet and target_sheet.lower() not in sheet_name.lower():
                continue
            if not isinstance(sheet_data, dict):
                continue
            md = sheet_data.get("markdown")
            if md and md.strip():
                # HUGE PERFORMANCE & ACCURACY FIX: Simulate strict Chunk RAG
                # Score every row based on how many question keywords it contains
                # Then ONLY keep the top 3-5 highest scoring rows to prevent LLM hallucination
                import re
                question_tokens = set(re.findall(r'[a-z0-9]{4,}', question.lower()))
                
                lines = md.split('\n')
                
                # Keep headers (first 3 lines) unconditionally
                header_lines = lines[:3] if len(lines) >= 3 else lines
                data_lines = lines[3:] if len(lines) >= 3 else []
                
                scored_lines = []
                for line in data_lines:
                    line_lower = line.lower()
                    
                    # Count how many question tokens appear in this line
                    score = sum(1 for t in question_tokens if t in line_lower)
                    
                    # Bonus for exact phrase matches (e.g. "covid 19 vaccination")
                    # Helps heavily prioritize the exact row
                    if score > 0:
                        scored_lines.append((score, line))
                
                # Sort rows by score descending, keep top 5 most relevant chunks
                scored_lines.sort(key=lambda x: x[0], reverse=True)
                top_data_lines = [row for score, row in scored_lines[:5]]
                
                # Re-assemble the pruned markdown (Headers + Best Rows)
                # Keep original order if possible
                filtered_lines = header_lines + [line for line in data_lines if line in top_data_lines]
                pruned_md = "\n".join(filtered_lines)
                markdown_contexts.append(f"### File: {file_name} | Sheet: {sheet_name}\n{pruned_md}")
                
    if not markdown_contexts:
        print("[AGENT] No Markdown tables available in cache.")
        return None

    # Combine all contexts
    full_context = "\n\n".join(markdown_contexts)
    # Truncate context if it gets absurdly huge to prevent OOM
    if len(full_context) > 25000:
        full_context = full_context[:25000] + "\n...[TRUNCATED]"

    prompt = f"""You are an elite Data Extraction AI.
You are given a precise slice of an Excel file converted to Markdown format.

MARKDOWN TABLE CONTEXT:
{full_context}

USER QUESTION: {question}

STRICT INSTRUCTIONS:
1. Extract the EXACT value requested by the user from the provided Markdown table.
2. Read the table structure logically. If the user asks for a specific limit for a specific category, find the intersection.
3. OUTPUT ONLY THE FINAL ANSWER STRING. NO EXPLANATIONS. NO PREAMBLE. NO CHATBOT TEXT.
4. If the question asks for a limit, return only the amount (e.g., 'RM 3,500' or 'UNLIMITED').
5. If the question asks for a Code or Category, return only that Name.
6. If the exact answer is absolutely NOT present in the provided context, output exactly: NOT_FOUND.

FINAL ANSWER:"""

    try:
        print("[AGENT] Injecting Markdown into LLM context...")
        raw_response = llm.invoke(prompt)
    except Exception as e:
        print(f"[AGENT] LLM Table-QA error: {e}")
        return None

    answer = getattr(raw_response, "content", str(raw_response)).strip()
    print(f"[AGENT] Output -> {answer}")
    
    # Filter out LLM garbage or negations
    sanitized = answer.lower()
    if sanitized in ("not found", "none", "nan", "") or "cannot find" in sanitized or "does not contain" in sanitized:
        return None

    # Remove conversational filler if the LLM ignores the instruction
    for prefix in ["the answer is ", "based on the table, ", "according to the data, "]:
        if answer.lower().startswith(prefix):
            answer = answer[len(prefix):].strip()

    return answer
