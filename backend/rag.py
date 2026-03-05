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
from excel_query import is_numeric_question, run_dataframe_query

# ── Configuration ──────────────────────────────────────────────────
COLLECTION_NAME = "local_documents"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ── Model Singletons (loaded once at startup) ──────────────────────
print(f"--- [RAG] Initializing embeddings & LLM ({OLLAMA_MODEL}) ---")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = Ollama(
    model=OLLAMA_MODEL,
    temperature=0,
    num_predict=128,
    top_p=0.5,
    repeat_penalty=1.1,
)
print("--- [RAG] Ready ---")


# ── Helpers ────────────────────────────────────────────────────────
def _normalize(text: str) -> str:
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text.lower().replace('-', ' ').replace('_', ' ').replace('/', ' ')).strip()


def _analyze_query(question: str) -> dict:
    """Use a fast LLM call to extract search entities and mapping needs."""
    prompt = f"""Analyze this question for a database lookup.
    
QUESTION: "{question}"

Instructions:
- "product": Clean product name (e.g. "Glico Pocky").
- "attributes": List of data points requested (e.g. ["UPC", "Description"]).
- "search_terms": 3-5 keywords primarily numbers and unique product words.
- "synonyms": List of equivalent terms. If they ask for "UPC", include ["GTIN", "ArticleCode", "UPC"].

JSON ONLY:"""
    try:
        raw = llm.invoke(prompt).strip()
        if "```json" in raw: raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw: raw = raw.split("```")[1].split("```")[0].strip()
        data = json.loads(raw)
        # Ensure consistency
        if "UPC" in str(data.get("attributes", [])) or "UPC" in str(data.get("synonyms", [])):
            if "synonyms" not in data: data["synonyms"] = []
            data["synonyms"].extend(["GTIN", "ArticleCode", "UPC", "Article Description Code"])
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
    docs = vector_store.similarity_search(question, k=15, filter=None)
    
    # 2. Path B: Exact Keyword Filters (Lexical)
    product_name = analysis.get("product", "")
    search_terms = analysis.get("search_terms", [])
    
    # Combine product words and attributes
    filter_tokens = set(search_terms)
    if product_name:
        filter_tokens.update(product_name.split()[:3])
    
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
                limit=15,
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
    text_kw = [k for k in keywords if not any(c.isdigit() for c in k)]

    filtered = []
    seen_content = set()

    for doc in docs:
        content_norm = _normalize(doc.page_content)
        if content_norm in seen_content: continue
        
        section_norm = _normalize(doc.metadata.get("section", ""))
        combined = content_norm + " " + section_norm

        # Numeric keywords (like 38g) must appear closely (handling units)
        numeric_hit = True
        for nk in numeric_kw:
            base_num = re.sub(r'[^\d]', '', nk)
            if base_num and base_num not in combined:
                numeric_hit = False
                break
        
        if not numeric_hit:
            continue

        # For text keywords, we check if at least some appear
        hit_count = sum(1 for k in text_kw if k in combined)
        if text_kw and hit_count == 0:
            continue

        filtered.append(doc)
        seen_content.add(content_norm)

    print(f"[HYBRID-FILTER] Retained {len(filtered)} high-quality docs.", flush=True)
    return filtered if filtered else docs[:3]


# ── STEP 6: LLM Answer Generator ──────────────────────────────────
def _generate_answer(question: str, filtered_docs: list) -> str:
    """Build context from matched rows and generate a clean, factual answer."""
    if not filtered_docs:
        return "Not found in data."

    # Build context: each doc is already a KEY=VALUE string
    context_lines = []
    seen = set()
    for doc in filtered_docs[:12]:  # Top 12 rows
        line = doc.page_content.strip()
        if line and line not in seen:
            seen.add(line)
            context_lines.append(line)

    context = "\n".join(context_lines)

    prompt = f"""You are a precise DATA-ONLY extraction bot.
    
STRICT RULES:
1. Answer the question using ONLY the provided DATA below.
2. If the answer is not explicitly written in the DATA, say exactly: "Not found in data."
3. DO NOT use your own knowledge. Never assume or guess.
4. If you find multiple matches, list them clearly.
5. If the product name doesn't match EXACTLY as written in data, say: "Not found in data."

DATA SECTIONS:
{context}

USER QUESTION: {question}

FINAL ANSWER (ONE SENTENCE MAX OR "Not found in data."):"""

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



# ── Main Query Entry Point ─────────────────────────────────────────
def query_rag(question: str, folder_name: str = None) -> str:
    print(f"\n[RAG-v2] Question: {question}")
    start = time.time()

    # 1. Query Analysis
    analysis = _analyze_query(question)
    keywords = analysis.get("search_terms", []) # updated key
    print(f"  Analysis: {json.dumps(analysis)}", flush=True)

    # 2. Route Math
    if is_numeric_question(question):
        print("[ROUTE] -- Pandas (math)")
        df_result = run_dataframe_query(question, folder_name)
        if df_result:
            print(f"[PANDAS] Answered in {time.time()-start:.2f}s")
            return df_result.replace("COMPUTED RESULTS FROM EXCEL DATA:\n", "").strip()

    # 3. Hybrid Retrieval
    print("[ROUTE] -- Hybrid Search (Vector + Lexical)")
    try:
        docs = _hybrid_retrieve(question, analysis, folder_name)
    except Exception as e:
        print(f"[HYBRID-SEARCH] Error: {e}")
        return "Search engine error. Please try again."

    if not docs:
        return "No data found. Please upload the relevant file first."

    # 4. Filter & De-duplicate
    filtered_docs = _post_filter(docs, keywords)

    # 5. Answer Generation
    answer = _generate_answer(question, filtered_docs)

    print(f"[RESULT] '{answer}' (total: {time.time()-start:.2f}s)")
    return answer
