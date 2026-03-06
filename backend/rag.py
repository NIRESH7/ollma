"""
RAG Pipeline - Full Hybrid Flow (v4)
Step 4: Route question → Pandas (math) or Qdrant (lookup)
Step 5: Post-filter Qdrant results to exact matches only
Step 6: LLM generates 1 clean sentence answer
"""

import os
import re
import time
import json
from difflib import SequenceMatcher
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http import models
from database import get_qdrant_client
from excel_query import is_numeric_question, run_dataframe_query, get_row_by_code
from langchain_core.documents import Document

# ── Configuration ──────────────────────────────────────────────────
COLLECTION_NAME = "local_documents"
# Default to qwen2:1.5b for better performance on Intel i5
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2:1.5b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ── Model Singletons (loaded once at startup) ──────────────────────
print(f"--- [RAG] Initializing embeddings & LLM ({OLLAMA_MODEL}) ---")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = Ollama(
    model=OLLAMA_MODEL,
    temperature=0,
    num_predict=64,
    top_p=0.1,
    repeat_penalty=1.1,
    stop=["USER QUESTION:", "DATABASE RECORDS:", "\n\n"]
)
print("--- [RAG] Ready ---")


# ── Helpers ────────────────────────────────────────────────────────
def _normalize(text: str) -> str:
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text.lower().replace('-', ' ').replace('_', ' ').replace('/', ' ')).strip()


def _analyze_query(question: str) -> dict:
    """Interpret the user's question and locate relevant identifiers or intent."""
    prompt = f"""ROLE: You are a data assistant that interprets user questions based on a dataset with unknown structures.
PRIMARY OBJECTIVE: Determine the intent and identify if the question references a specific ID, code, or unique value.

QUESTION: "{question}"

Instructions:
1. If the question asks for a specific record by an ID or Code (e.g., "What is the status of BUG-001?"), set "action" to "get_row_by_code" and "payload" to that ID.
2. If the question is general or about patterns (e.g., "How many high severity bugs?"), set "action" to "search_dataset".
3. Semantic Similarity: Focus on the meaning of the request to identify identifiers.

Return ONLY a valid JSON object:
{{"action": "...", "payload": "..."}}"""
    try:
        raw = llm.invoke(prompt).strip()
        if "```json" in raw: raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw: raw = raw.split("```")[1].split("```")[0].strip()
        data = json.loads(raw)
        return data
    except Exception as e:
        print(f"  ⚠️ [QUERY-ANALYSIS] Error: {e}. Falling back.")
        return {
            "product": question, 
            "attributes": ["UPC", "GTIN", "ArticleCode"],
            "search_terms": _extract_keywords(question),
            "synonyms": ["GTIN", "ArticleCode", "UPC", "Barcode"]
        }


def _extract_keywords(question: str) -> list[str]:
    """Fallback method to extract search tokens."""
    q_norm = _normalize(question)
    STOPWORDS = {
        'what', 'which', 'where', 'when', 'who', 'give', 'tell', 'find', 'me',
        'show', 'list', 'the', 'is', 'are', 'for', 'and', 'that', 'value',
        'total', 'available', 'all', 'in', 'of', 'on', 'at', 'to', 'from',
        'this', 'item', 'with', 'about', 'it', 'was', 'were', 'be', 'an', 'as',
    }
    tokens = re.findall(r'\b[\w.]{2,}\b', q_norm)
    return [t for t in tokens if t not in STOPWORDS]


# ── STEP 4b: Qdrant retriever ──────────────────────────────────────
def _hybrid_retrieve(question: str, analysis: dict, folder_name: str = None) -> list:
    """Combines Vector search with Exact Keyword filters for high precision."""
    client = get_qdrant_client()
    vector_store = QdrantVectorStore(
        client=client, collection_name=COLLECTION_NAME, embedding=embeddings
    )
    
    # 1. Path A: Vector Search (Semantic)
    docs = vector_store.similarity_search(question, k=2, filter=None)
    
    # 2. Path B: Exact Keyword Filters (Lexical)
    product_name = analysis.get("product", "")
    search_terms = analysis.get("search_terms", [])
    
    # Combine product words, attributes and synonyms
    filter_tokens = set(search_terms)
    if product_name:
        filter_tokens.update(product_name.split()[:3])
    
    # Add synonyms and attributes to filter tokens for better lexical coverage
    def _collect_tokens(items):
        if isinstance(items, str):
            filter_tokens.update(items.split())
        elif isinstance(items, list):
            for item in items:
                _collect_tokens(item)

    _collect_tokens(analysis.get("synonyms", []))
    _collect_tokens(analysis.get("attributes", []))
    
    filter_tokens = [t for t in filter_tokens if len(t) > 2] # ignore tiny words

    if filter_tokens:
        try:
            kw_docs, _ = client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.folder",
                            match=models.MatchValue(value=folder_name)
                        ) if folder_name and folder_name != "All" else None,
                    ],
                    should=[
                        models.FieldCondition(
                            key="page_content",
                            match=models.MatchText(text=t)
                        ) for t in filter_tokens
                    ]
                ),
                limit=5,
                with_payload=True
            )
            for p in kw_docs:
                docs.append(Document(page_content=p.payload.get("page_content", ""), metadata=p.payload.get("metadata", {})))
        except:
            pass

    return docs


def _post_filter(docs: list, keywords: list[str]) -> list:
    """
    Keep only documents that contain ALL numeric keywords closely.
    If multiple docs match, prioritize those with better semantic alignment.
    """
    if not keywords:
        return docs

    numeric_kw = [k for k in keywords if any(c.isdigit() for c in k)]
    # Post-filter: Check for numeric exact matches
    perfect_numeric_matches = []
    other_matches = []
    
    for doc in docs:
        numeric_hit = True
        # Check both the content and the raw_row metadata for numbers
        search_text = (doc.page_content + " " + doc.metadata.get("raw_row", "")).lower()
        
        for nk in numeric_kw:
            target_num = re.sub(r'[^\d.]', '', nk).rstrip('0').rstrip('.')
            if not target_num: continue
            
            # Boundary-aware numeric check in the search text
            content_nums = re.findall(r'\b\d+\.?\d*\b', search_text)
            normalized_content_nums = [n.rstrip('0').rstrip('.') for n in content_nums]
            
            if target_num not in normalized_content_nums:
                numeric_hit = False
                break
        
        if numeric_hit:
            perfect_numeric_matches.append(doc)
        else:
            other_matches.append(doc)

    print(f"[HYBRID-FILTER] Perfect Numeric Matches: {len(perfect_numeric_matches)} | Others: {len(other_matches)}", flush=True)
    
    # Return perfect matches if found, otherwise fallback to others
    return perfect_numeric_matches if perfect_numeric_matches else other_matches[:3]


# ── STEP 6: LLM Answer Generator ──────────────────────────────────
def _generate_answer(question: str, filtered_docs: list) -> str:
    """Generate a direct answer based strictly on the provided dataset records."""
    if not filtered_docs:
        return "No matching data found in the uploaded dataset."

    # Build context: each doc is already a KEY: VALUE string
    context_lines = []
    seen = set()
    for doc in filtered_docs[:2]:  # Top 2 rows for speed/relevance
        # Format for readability
        line = doc.page_content.replace(" | ", "\n").strip()
        if line and line not in seen:
            seen.add(line)
            context_lines.append(line)

    context = "\n---\n".join(context_lines)

    prompt = f"""ROLE: You are a data assistant. Answer strictly based on the DATABASE RECORDS.
OBJECTIVE: Return ONLY the specific information requested.

COLUMN MATCHING:
- Semantically match the question intent to the most relevant fields in the records.
- If multiple fields match, pick the most specific one.

RESPONSE RULES:
- Return ONLY the direct answer.
- No explanations. No hallucinations.
- If not found, say: "No matching data found in the uploaded dataset."

DATABASE RECORDS:
{context}

USER QUESTION: {question}

ANSWER:"""

    print(f"--- [LLM] Sending {len(context_lines)} rows to model ---", flush=True)
    t = time.time()
    try:
        # Use simple string replacement for reliability
        result = llm.invoke(prompt).strip()
        print(f"--- [LLM] Response in {time.time()-t:.2f}s ---", flush=True)
    except Exception as e:
        print(f"--- [LLM] Error: {e} ---", flush=True)
        return "AI module error. Please try again."

    # Clean up: take first non-empty line and strip common prefixes
    for line in result.split('\n'):
        line = line.strip()
        if not line: continue
        # Strip common AI noise
        line = re.sub(r'^(DATA:|ANSWER:|RULES:|QUESTION:|INSTRUCTIONS:|Result:)', '', line, flags=re.IGNORECASE).strip()
        if line: return line

    return "Not found in data."



def query_rag(question: str, folder_name: str = None) -> str:
    print(f"\n[USER QUESTION] {question}")
    start = time.time()

    # 0. Fast Regex Identifier Check (BUG-XXX, ID-XXX, Alphanumeric Codes)
    # This short-circuits to avoid LLM Router call for direct lookups
    id_match = re.search(r'\b(BUG-\d+|[A-Z0-9]{3,}-\d+|\d{4,})\b', question.upper())
    if id_match:
        identifier = id_match.group(1)
        print(f"[ROUTER] Detected Identifier lookup: {identifier}")
        print(f"[TOOL CALL] get_row_by_code({identifier})")
        
        data_match = get_row_by_code(identifier, folder_name)
        if "not found" not in data_match.lower():
            print(f"[INFO] Direct ID record found. Passing to LLM for specific extraction.")
            # Wrap the direct match as a Document and use the generator to extract only the answer
            id_doc = Document(page_content=data_match, metadata={})
            return _generate_answer(question, [id_doc])

    # 1. Tool-Based Query Analysis (Fallback for complex queries)
    analysis = _analyze_query(question)
    action = analysis.get("action", "search_dataset")
    payload = analysis.get("payload", "")
    
    if action == "get_row_by_code":
        print(f"[ROUTER] Detected code lookup via AI")
    else:
        print(f"[ROUTER] Semantic Search")

    # 2. Execute Tool
    if action == "get_row_by_code" and payload:
        print(f"[TOOL CALL] get_row_by_code({payload})")
        data_match = get_row_by_code(payload, folder_name)
        
        if "not found" not in data_match.lower():
            print(f"[TOOL RESULT]\n{data_match}")
            # Generate answer from the exact record
            filtered_docs = [Document(page_content=data_match, metadata={})]
            return _generate_answer(question, filtered_docs)

    # 3. Fallback to Hybrid Search
    print("[ROUTE] -- Semantic Search")
    try:
        # Standard search logic
        docs = _hybrid_retrieve(question, {"search_terms": [question]}, folder_name)
    except Exception as e:
        print(f"[SEARCH] Error: {e}")
        return "Search error."

    if not docs:
        return "No data found."

    # 4. Generate Answer
    answer = _generate_answer(question, docs[:5])
    print(f"[AI RESPONSE] {answer}")
    print(f"--- [INFO] Total response time: {time.time()-start:.2f}s ---")
    return answer
