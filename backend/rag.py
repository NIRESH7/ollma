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

# ── Step 1: Structured Label/Value Lookup ─────────────────────────
def _structured_lookup(question: str, folder_name: str) -> Optional[str]:
    """LOG: STRUCTURED_LOOKUP | Scanning in-memory JSON cache for Values and Keys."""
    # Normalize question and extract chunks
    q_norm = question.lower()
    q_tokens = [t.strip() for t in re.split(r'[\W_]+', q_norm) if len(t.strip()) > 3]
    
    # Also get snake_case version of chunks
    snake_chunks = [_to_snake_case(t) for t in q_tokens]
    
    for file_path, file_data in EXCEL_CACHE.items():
        if folder_name != "All" and file_data.get("folder") != folder_name:
            continue
            
        for sheet_name, sheet_data in file_data.get("sheets", {}).items():
            for record in sheet_data.get("data", []):
                # 1. Look for a VALUE match (e.g., "JL 12" or "JL12") to identify the row
                row_matches = False
                record_values = [str(v).lower() for v in record.values() if not isinstance(v, (dict, list))]
                
                # Check if any significant token from the question exists in the record values
                for token in q_tokens:
                    if any(token in val for val in record_values if len(val) > 2):
                        row_matches = True
                        break
                
                if row_matches:
                    # 2. Once row is found, find the KEY that matches other question keywords (e.g., "limit")
                    for key, val in record.items():
                        key_norm = key.lower()
                        # If the key contains "limit", "amount", or other cost/benefit keywords
                        target_keywords = ["limit", "amount", "remarks", "spending", "top_up", "med", "point"]
                        
                        # Match if key contains a target keyword AND question mentions it
                        for kw in target_keywords:
                            if kw in key_norm and kw in q_norm:
                                # Final check to ensure we aren't returning the value that matched the row identification
                                if str(val).lower() not in q_norm:
                                    return str(val)
                                    
                    # Fallback: If row matches but specific key doesn't, we'll let Vector/LLM handle it
                    # OR we can return a concise summary of the found row
    return None

# ── Logic ────────────────────────────────────────────────────────
def query_rag(question: str, folder_name: str = "All") -> str:
    print(f"\nLOG: QUERY_RECEIVED | {question}")
    t_start = time.time()
    
    # ── STEP 0: TYPO RECOVERY ──
    # Quick fix for common user typos in this specific domain
    clean_question = question.replace("garde", "grade").replace("JL12", "JL 12")
    
    # ── STEP 1: STRUCTURED LOOKUP ──
    print(f"LOG: STRUCTURED_LOOKUP | Attempting deterministic row-value match...")
    direct_match = _structured_lookup(clean_question, folder_name)
    if direct_match:
        print(f"LOG: FINAL_ANSWER | (Direct Match) {direct_match}")
        return direct_match

    # ── STEP 2: NUMERIC ENGINE ──
    if is_numeric_question(clean_question):
        print(f"LOG: NUMERIC_ENGINE | Routing to Pandas computation...")
        math_result = run_dataframe_query(clean_question, folder_name)
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
    
    docs = vector_store.similarity_search(clean_question, k=20, filter=folder_filter)
    
    if not docs:
        print(f"LOG: FINAL_ANSWER | Data not available.")
        return "Data not available."

    # ── STEP 4: LLM REASONING ──
    print(f"LOG: LLM_REASONING | Fallback to synthesis with fuzzy tolerance...")
    context = "\n---\n".join([d.page_content for d in docs])
    
    # Relaxed prompt for better reasoning with typos
    prompt = f"""Use the context provided to answer the user's question.
1. Be concise. 
2. If the user makes a typo (e.g. 'garde' instead of 'grade'), use the correct term from the context.
3. If the answer is found in a table row matching the code/grade mentioned, return exactly that value.
4. If not found, return: Data not available

CONTEXT:
{context}

QUESTION:
{clean_question}

ANSWER:"""

    try:
        response = llm.invoke(prompt).strip()
        answer = response.split("\n")[0].strip()
        answer = re.sub(r'^(ANSWER:|FINAL ANSWER:|RESULT:)', '', answer, flags=re.IGNORECASE).strip()
        
        if not answer: answer = "Data not available."
        print(f"LOG: FINAL_ANSWER | (LLM) {answer}")
        return answer
    except Exception as e:
        print(f"LOG: ERROR | {e}")
        return "Data not available."
