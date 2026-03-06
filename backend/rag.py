"""
NeuralRAG Query & Retrieval Pipeline (v7 - Deterministic Strict)
1. STRUCTURED_LOOKUP: Direct KEY:VALUE match in memory cache.
2. NUMERIC_ENGINE: Math computation (Pandas).
3. VECTOR_RETRIEVAL: Top 20 semantic segments.
4. LLM_REASONING: Final fallback for synthesis.
"""

import os
import re
import time
import json
from typing import Optional, List, Dict, Any
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http import models
from database import get_qdrant_client
from excel_query import is_numeric_question, run_dataframe_query, get_row_by_code
from excel_ingestion import EXCEL_CACHE, _to_snake_case

# ── Configuration ──────────────────────────────────────────────────
COLLECTION_NAME = "local_documents"
OLLAMA_MODEL = "qwen2:1.5b"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ── Model Singletons ───────────────────────────────────────────────
print(f"--- [RAG] Initializing Embeddings & LLM ({OLLAMA_MODEL}) ---")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = Ollama(
    model=OLLAMA_MODEL,
    temperature=0,
    top_p=0.9,
    repeat_penalty=1.1,
    base_url=OLLAMA_BASE_URL
)
print("--- [RAG] Ready ---")

# ── Step 1: Structured Label Lookup ───────────────────────────────
def _structured_lookup(question: str, folder_name: str) -> Optional[str]:
    """LOG: STRUCTURED_LOOKUP | Scanning in-memory JSON cache."""
    q_norm = _to_snake_case(question)
    
    # Extract keywords for fuzzy matching
    keywords = [k for k in re.split(r'_', q_norm) if len(k) > 3]
    
    for file_path, file_data in EXCEL_CACHE.items():
        if folder_name != "All" and file_data.get("folder") != folder_name:
            continue
            
        for sheet_name, sheet_data in file_data.get("sheets", {}).items():
            for record in sheet_data.get("data", []):
                for key, val in record.items():
                    # 1. Exact match
                    if key == q_norm:
                        return str(val)
                    # 2. Keyword check (e.g., "What is the deductible?" matches "deductible")
                    if any(k in key for k in keywords):
                        return str(val)
    return None

# ── Logic ────────────────────────────────────────────────────────
def query_rag(question: str, folder_name: str = "All") -> str:
    print(f"\nLOG: QUERY_RECEIVED | {question}")
    t_start = time.time()
    
    # ── STEP 1: STRUCTURED LOOKUP ──
    print(f"LOG: STRUCTURED_LOOKUP | Attempting deterministic key match...")
    direct_match = _structured_lookup(question, folder_name)
    if direct_match:
        print(f"LOG: FINAL_ANSWER | (Direct Match) {direct_match}")
        return direct_match

    # ── STEP 2: NUMERIC ENGINE ──
    if is_numeric_question(question):
        print(f"LOG: NUMERIC_ENGINE | Routing to Pandas computation...")
        math_result = run_dataframe_query(question, folder_name)
        if math_result:
            print(f"LOG: FINAL_ANSWER | (Math Engine) {math_result}")
            return math_result

    # ── STEP 3: VECTOR RETRIEVAL ──
    print(f"LOG: VECTOR_RETRIEVAL | Fetching 20 segments from Qdrant...")
    client = get_qdrant_client()
    vector_store = QdrantVectorStore(client=client, embedding=embeddings, collection_name=COLLECTION_NAME)
    
    folder_filter = None
    if folder_name and folder_name != "All":
        folder_filter = models.Filter(
            must=[models.FieldCondition(key="metadata.folder", match=models.MatchValue(value=folder_name))]
        )
    
    docs = vector_store.similarity_search(question, k=20, filter=folder_filter)
    
    if not docs:
        print(f"LOG: FINAL_ANSWER | Data not available.")
        return "Data not available."

    # ── STEP 4: LLM REASONING ──
    print(f"LOG: LLM_REASONING | Fallback to synthesis...")
    context = "\n---\n".join([d.page_content for d in docs])
    
    prompt = f"""Use ONLY the context below to answer strictly.
1. No fluff. 
2. Match labels to question exactly.
3. If not found, return: Data not available

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

    try:
        response = llm.invoke(prompt).strip()
        answer = response.split("\n")[0].strip()
        answer = re.sub(r'^(ANSWER:|FINAL ANSWER:|RESULT:)', '', answer, flags=re.IGNORECASE).strip()
        
        if not answer: answer = "Data not available."
        print(f"LOG: FINAL_ANSWER | (LLM) {answer}")
        return answer
    except:
        return "Data not available."
