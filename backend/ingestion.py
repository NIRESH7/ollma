import os
import warnings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from database import get_qdrant_client
from langchain_core.documents import Document
from excel_ingestion import ingest_with_intelligence_engine
from embeddings_provider import get_embeddings

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration for Render Cloud Deployment
COLLECTION_NAME = "local_documents"

from langchain_community.document_loaders import PDFPlumberLoader, TextLoader, Docx2txtLoader

def load_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        print(f"--- [INGEST] Using PDFPlumberLoader for Layout Preservation on {file_path} ---")
        loader = PDFPlumberLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return loader.load()

def extract_text_with_ocr(file_path, progress_callback=None):
    """
    Fallback method to extract text from scanned PDFs using RapidOCR (No external binaries needed).
    """
    import sys
    print(f"--- [OCR] Starting RapidOCR fallback for: {file_path} ---")
    try:
        from rapidocr_onnxruntime import RapidOCR
        from pypdf import PdfReader
        
        ocr = RapidOCR()
        reader = PdfReader(file_path)
        num_pages = len(reader.pages)
        print(f"--- [OCR] PDF has {num_pages} pages. ---")
        documents = []
        
        for i, page in enumerate(reader.pages):
            if progress_callback:
                progress_callback(i + 1, num_pages)
            page_text = ""
            try:
                # Extract images from the page
                images = page.images
                if images:
                    print(f"--- [OCR] Processing page {i+1} ({len(images)} images) ---")
                    for img in images:
                        try:
                            result, _ = ocr(img.data)
                            if result:
                                for line in result:
                                    if line and len(line) >= 2:
                                        text_content = line[1]
                                        # Add a newline after each extracted block to prevent too much squishing
                                        page_text += text_content + "\n"
                        except Exception as img_e:
                            print(f"--- [OCR] Warning: Failed to process an image on page {i+1}: {img_e} ---")
            except Exception as page_e:
                print(f"--- [OCR] Warning: Failed to extract images from page {i+1}: {page_e} ---")

            if page_text.strip():
                documents.append(Document(page_content=page_text, metadata={"source": file_path, "page": i}))
        
        if documents:
            print(f"--- [OCR] TOTAL SUCCESS: Extracted text from {len(documents)} pages. ---")
        else:
            print(f"--- [OCR] WARNING: Failed to extract any text even with OCR. ---")
            
        return documents

    except ImportError as e:
        print(f"--- [OCR] CRITICAL ERROR: Import failed! {e} ---")
        return []
    except Exception as e:
        print(f"--- [OCR] Critical Error during OCR: {e} ---")
        return []

def ingest_file(file_path: str, folder_name: str = "default", progress_callback=None):
    print(f"--- [INGEST] Starting ingestion for: {file_path} in folder: {folder_name} ---", flush=True)
    if not os.path.exists(file_path):
        print(f"--- [INGEST] ERROR: File not found at {file_path} ---")
        return {"error": "File not found"}

    # ── Route spreadsheet data files to the structured intelligence engine ──
    ext = os.path.splitext(file_path)[1].lower()
    if ext in (".xlsx", ".xlsm", ".xls", ".csv"):
        print(f"--- [INGEST] Routing to Universal Spreadsheet Intelligence Engine for: {file_path} ---")
        return ingest_with_intelligence_engine(file_path, folder_name)

    docs = []
    used_ocr = False

    # 1. Try Standard Load
    try:
        docs = load_document(file_path)
        print(f"--- [INGEST] Successfully loaded {len(docs)} segments (Standard) ---")
    except Exception as e:
        print(f"--- [INGEST] Standard load failed: {e}. Trying OCR... ---")

    # Filter empty pages
    docs = [d for d in docs if d.page_content and d.page_content.strip()]

    # 2. If no text, try OCR
    if not docs and file_path.lower().endswith(".pdf"):
        print("--- [INGEST] No text found with standard loader. Attempting OCR... ---")
        docs = extract_text_with_ocr(file_path, progress_callback=progress_callback)
        used_ocr = True
        
        # Filter again after OCR
        docs = [d for d in docs if d.page_content and d.page_content.strip()]

    if not docs:
        if used_ocr:
             msg = "No text extracted even with OCR. The file might be empty, corrupted, or contain unsupported image formats."
        else:
             msg = "No text extracted. This file appears to be empty or an image/scanned PDF."
             
        print(f"--- [INGEST] FAILURE: {msg} ---")
        return {"error": msg}

    # ── Extraction Summary ──
    total_chars = sum(len(d.page_content) for d in docs)
    print(f"\n📄 [EXTRACTED] Summary for: {os.path.basename(file_path)}")
    print(f"   • Total Pages/Segments: {len(docs)}")
    print(f"   • Total Characters: {total_chars:,}")
    
    if docs:
        preview = docs[0].page_content[:200].replace("\n", " ").strip()
        print(f"   • Text Preview: \"{preview}...\"\n")
    
    # 1.5 Add Metadata
    for doc in docs:
        doc.metadata["folder"] = folder_name
        doc.metadata["ocr_processed"] = used_ocr
    
    # 2. Split with semantic separators
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    splits = text_splitter.split_documents(docs)
    print(f"--- [INGEST] Split into {len(splits)} chunks for Vector Search ---", flush=True)
    
    # 3. Embeddings (supports offline fallback)
    embeddings, _, embedding_backend = get_embeddings()
    print(f"--- [INGEST] Embedding backend: {embedding_backend} ---")
    
    # Use Singleton Client
    client = get_qdrant_client()
    
    # Check pre-ingest count
    try:
        pre_count = client.count(collection_name=COLLECTION_NAME).count
    except Exception:
        pre_count = 0

    # Using the modern QdrantVectorStore
    qdrant = QdrantVectorStore(
        client=client,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
    )
    
    print(f"--- [INGEST] Adding {len(splits)} documents to Qdrant ---", flush=True)
    qdrant.add_documents(splits)
    
    # Check post-ingest count
    try:
        post_count = client.count(collection_name=COLLECTION_NAME).count
    except Exception:
        post_count = 0
    
    status_msg = "Ingested (OCR)" if used_ocr else "Ingested"

    print(f"--- [INGEST] Success! Documents added to Qdrant. ---", flush=True)
    
    return {
        "num_chunks": len(splits), 
        "status": status_msg, 
        "total_vectors": post_count,
        "ocr_used": used_ocr
    }
