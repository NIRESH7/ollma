import os
import time
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http import models
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from database import get_qdrant_client

# Configuration
COLLECTION_NAME = "local_documents"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Global Model Singletons (Initializes only ONCE on startup)
print(f"--- [RAG] Initializing Local Embeddings & LLM (Ollama: {OLLAMA_MODEL}) ---")

# Purely Local HuggingFace Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("--- [RAG] Using Local HuggingFace Embeddings (all-MiniLM-L6-v2) ---")

# Purely Local Ollama
print("--- [RAG] Using Local Ollama Engine ---")
llm = Ollama(
    model=OLLAMA_MODEL,
    temperature=0,      # Strict facts
    num_predict=512,    # Allow longer expert extraction
    top_p=0.5,           # High confidence only
    repeat_penalty=1.1   # Avoid repetitive noise
)
print("--- [RAG] Models Loaded and Ready ---")

def get_rag_chain(folder_name: str = None):
    # 2. Vector Store - Use Singleton
    client = get_qdrant_client()
    vector_store = QdrantVectorStore(
        client=client, 
        collection_name=COLLECTION_NAME, 
        embedding=embeddings
    )
    
    search_kwargs = {"k": 3}
    if folder_name and folder_name != "All":
        print(f"--- [RAG] Filtering by folder: {folder_name} ---")
        # Use Qdrant's Filter model for better compatibility with langchain-qdrant
        search_kwargs["filter"] = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.folder", 
                    match=models.MatchValue(value=folder_name)
                )
            ]
        )
    
    # Increase k for better context
    retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
    
    # 4. Prompt - Full Context Expert Mode
    template = """You are a document data extraction expert.

Below is ALL extracted data from the document. Read EVERY line carefully.
Find the answer ONLY in the data below. Do NOT guess.
Answer with ONLY the exact value - no explanation, no extra words.

DOCUMENT DATA:
{context}

QUESTION: {question}
EXACT ANSWER:"""
    
    prompt = PromptTemplate.from_template(template)
    
    # 5. Chain with LCEL
    rag_chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()} 
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return retriever, rag_chain

import json
import re

def structural_table_analyzer(text: str):
    """Universal Document Parser (Optimized for any format)."""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    facts = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Pattern 1: Explicit Pairs (Label: Value, Label - Value, Label = Value)
        kv = re.search(r'^([^:=]{2,40})[:=-]\s*(.+)$', line)
        if kv:
            facts.append(f"{kv.group(1).strip()} is {kv.group(2).strip()}")
            i += 1
            continue

        # Pattern 2: Multi-line pairing (Label on one line, value on next)
        # Very common in OCR of structured forms/receipts
        if i + 1 < len(lines):
            l0, l1 = lines[i], lines[i+1]
            # Check if l1 looks like a standalone value (Date, Amount, ID)
            if len(l0.split()) < 5 and re.search(r'(\d|/|-|\$|€|£|INR)', l1):
                # Heuristic: If l1 is mostly numeric/identifiers, pair it
                facts.append(f"{l0} -> {l1}")
                i += 2
                continue

        # Pattern 3: Table Row Detection (3+ columns of data)
        # Handles grid layouts where data is separated by spaces
        potential_cols = re.split(r'\s{2,}', line) # Split by double spaces (OCR common)
        if len(potential_cols) >= 3:
            facts.append(f"ROW DATA: {' | '.join(potential_cols)}")
            i += 1
            continue

        # Pattern 4: Significant Standalone Sentences
        # Keep lines that start with keywords or look like important summaries
        if any(k in line.upper() for k in ["TOTAL", "DUE", "TERMS", "PAYMENT", "ACCOUNT", "DATE", "TAX", "FEE", "COST", "VAT", "GST", "NET"]):
            facts.append(line)
            i += 1
            continue
            
        i += 1

    # Fallback: if we found very few facts, return a cleaned version of the original text
    if len(facts) < 3:
        return "\n".join(lines[:30]) # Return first 30 lines as raw context
        
    return "\n".join(facts)

def query_rag(question: str, folder_name: str = None):
    print("\n" + "🚀 " + "="*60)
    print(f"--- [MASTER SYSTEM] 1000% ACCURACY & FAST MODE ---".center(60))
    print(f"--- [QUERY]: {question} ---".center(60))
    print("="*60)
    
    start_time = time.time()
    
    try:
        retriever, chain = get_rag_chain(folder_name)
        # Retrieve top 5 chunks to cover ALL sections of any document
        retriever.search_kwargs["k"] = 5
        
        docs = retriever.invoke(question)
        if not docs: return "No data found."

        print("\n🔍 EXTRACTING FULL DOCUMENT DATA...")
        
        all_facts = []
        seen = set()  # Deduplicate facts across chunks
        for doc in docs:
            reconstructed = structural_table_analyzer(doc.page_content)
            src = os.path.basename(doc.metadata.get('source', 'doc'))
            print(f"✅ SOURCE: {src} ({len(reconstructed.split(chr(10)))} facts)")
            for fact in reconstructed.split('\n')[:8]:
                print(f"   | {fact}")
            print(f"   ... (+ more data sent to model)")
            # Add unique facts only
            for fact in reconstructed.split('\n'):
                if fact.strip() and fact.strip() not in seen:
                    seen.add(fact.strip())
                    all_facts.append(fact)

        print(f"\n📊 TOTAL UNIQUE FACTS: {len(all_facts)} | Sending full context to model...")
        print("="*60)

        # Send FULL deduplicated context to the model
        context_str = "\n".join(all_facts)
        result = chain.invoke({"context": context_str, "question": question})
        
        clean_ans = result.strip().split('\n')[0].replace("Answer:", "").strip()
        elapsed = time.time() - start_time

        # FINAL EXPERT OUTPUT
        print("\n" + "╔" + "═"*58 + "╗")
        print("║" + " EXTRACTION SUCCESS ".center(58) + "║")
        print("╠" + "═"*58 + "╢")
        print(f" SPEED    : {elapsed:.2f} seconds")
        print(f" DATA     : {clean_ans}")
        print("╚" + "═"*58 + "╝\n")

        return clean_ans

    except Exception as e:
        print(f"--- [CRITICAL ERROR]: {e} ---")
        return "Internal Error."
