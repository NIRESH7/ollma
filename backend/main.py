import warnings
# Suppress Pydantic V1 warning on Python 3.14+
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=UserWarning, module="langchain")

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import List
import os
import shutil
import time
import uvicorn
import logging

# ── Silence Polling Logs ───────────────────────────────────────────
class PollingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Stop printing the constant "GET /upload-status" clutter
        return "/upload-status/" not in record.getMessage()

# Apply the filter to uvicorn access logs
logging.getLogger("uvicorn.access").addFilter(PollingFilter())
# ──────────────────────────────────────────────────────────────────

import json
from fastapi import Form
from ingestion import ingest_file
from rag import query_rag
from excel_ingestion import delete_structured_cache

from database import get_qdrant_client, close_qdrant_client
from qdrant_client.http import models
from embeddings_provider import get_embeddings

app = FastAPI(title="Local RAG API")

# Security
API_KEY_NAME = "X-API-KEY"
API_KEY = os.getenv("API_KEY", "your-secret-key-change-me") # Default for local, set in env for cloud
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(header_api_key: str = Depends(api_key_header), request: Request = None):
    # Allow local requests without API key for development convenience
    client_host = request.client.host if request and request.client else ""
    local_hosts = ("127.0.0.1", "localhost", "::1", "::ffff:127.0.0.1")
    
    is_local = client_host in local_hosts
    
    if is_local or header_api_key == API_KEY:
        return header_api_key
    
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Could not validate credentials",
    )

# Global state for tracking upload progress
upload_stats = {}

@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Silence the polling checks to keep the terminal clean for the user
    if "/upload-status/" in request.url.path:
        return await call_next(request)
        
    start_time = time.time()
    print(f"--- [DEBUG] INCOMING REQUEST: {request.method} {request.url.path} ---")
    response = await call_next(request)
    process_time = time.time() - start_time
    print(f"--- [DEBUG] COMPLETED REQUEST: {request.method} {request.url.path} | STATUS: {response.status_code} | TIME: {process_time:.4f}s ---", flush=True)
    return response

# Configuration matching other files
COLLECTION_NAME = "local_documents"
# Dynamic vector size from active embedding backend
_, VECTOR_SIZE, VECTOR_BACKEND = get_embeddings()

@app.on_event("startup")
def startup_event():
    print(f"--- [STARTUP] Checking Qdrant Collection (Size: {VECTOR_SIZE}, Backend: {VECTOR_BACKEND}) ---")
    try:
        client = get_qdrant_client()
        # Simplest way to check for collections
        collections_resp = client.get_collections()
        col_names = [c.name for c in collections_resp.collections]
        
        if COLLECTION_NAME not in col_names:
            print(f"--- [STARTUP] Creating Collection '{COLLECTION_NAME}' with size {VECTOR_SIZE} ---")
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config={
                    "size": VECTOR_SIZE,
                    "distance": "Cosine"
                }
            )
        else:
            print(f"--- [STARTUP] Collection '{COLLECTION_NAME}' already exists. ---")
    except Exception as e:
        # Log and continue - don't crash the server
        print(f"--- [STARTUP] Warning/Error checking collection: {e} ---")

@app.on_event("shutdown")
def shutdown_event():
    print("--- [SHUTDOWN] Releasing Qdrant lock... ---", flush=True)
    close_qdrant_client()


# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Local RAG API is running"}

# ... imports ...
import json
from fastapi import Form

# ... existing code ...

FOLDERS_FILE = os.path.join("data", "folders.json")
os.makedirs("data", exist_ok=True)

def get_folders():
    if not os.path.exists(FOLDERS_FILE):
        return ["default"]
    try:
        with open(FOLDERS_FILE, "r") as f:
            return json.load(f)
    except:
        return ["default"]

def save_folder(folder_name):
    folders = get_folders()
    if folder_name not in folders:
        folders.append(folder_name)
        with open(FOLDERS_FILE, "w") as f:
            json.dump(folders, f)

class CreateFolderRequest(BaseModel):
    folder_name: str

@app.get("/folders", dependencies=[Depends(get_api_key)])
def list_folders():
    return {"folders": get_folders()}

@app.post("/folders", dependencies=[Depends(get_api_key)])
def create_folder(request: CreateFolderRequest):
    save_folder(request.folder_name)
    return {"status": "created", "folder": request.folder_name}

@app.get("/folders/{folder_name}/files", dependencies=[Depends(get_api_key)])
def list_folder_files(folder_name: str):
    print(f"--- [API] Listing files for folder: {folder_name} ---")
    try:
        client = get_qdrant_client()
        # Scroll through points with a filter to find unique filenames
        # Note: metadata schema matches rag.py (metadata.folder)
        limit = 100
        offset = None
        unique_files = set()
        
        # Simple implementation: scroll a reasonable number of points
        scroll_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.folder",
                    match={"value": folder_name}
                )
            ]
        )
        
        points, next_page_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=scroll_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )
        
        for p in points:
            if not p.payload:
                continue

            if "source" in p.payload and p.payload["source"]:
                unique_files.add(os.path.basename(p.payload["source"]))
                continue

            metadata = p.payload.get("metadata", {})
            if isinstance(metadata, dict):
                meta_source = metadata.get("source") or metadata.get("file_name")
                if meta_source:
                    unique_files.add(os.path.basename(meta_source))
                
        return {"files": list(unique_files)}
    except Exception as e:
        print(f"--- [API] Error listing files: {e} ---")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/folders/{folder_name}", dependencies=[Depends(get_api_key)])
def delete_folder(folder_name: str):
    print(f"--- [API] Deleting folder: {folder_name} ---")
    try:
        client = get_qdrant_client()
        # 1. Delete vectors from Qdrant using simple Filter
        # Note: if the folder is "All", we don't delete by metadata, we might want to skip or handle differently
        if folder_name and folder_name != "All":
            client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.folder",
                            match=models.MatchValue(value=folder_name)
                        )
                    ]
                )
            )
        
        # 2. Remove structured cache entries for the folder.
        cache_cleanup = delete_structured_cache(folder_name)

        # 3. Remove from folders.json
        folders = get_folders()
        if folder_name in folders:
            folders.remove(folder_name)
            with open(FOLDERS_FILE, "w") as f:
                json.dump(folders, f)
        
        return {
            "status": "deleted",
            "folder": folder_name,
            "cache_cleanup": cache_cleanup,
        }
    except Exception as e:
        print(f"--- [API] Error deleting folder: {e} ---")
        raise HTTPException(status_code=500, detail=str(e))

class QueryRequest(BaseModel):
    question: str
    folder: str = "All"

@app.post("/upload/", dependencies=[Depends(get_api_key)])
def upload_file(
    files: List[UploadFile] = File(...),
    folder: str = Form("default"),
    job_id: str = Form(None)
):
    print(f"\n[UPLOAD] File upload started", flush=True)
    print(f"--- [API] Received Upload Request for {len(files)} files in folder: {folder} (Job: {job_id}) ---", flush=True)
    
    if job_id:
        upload_stats[job_id] = {"status": "processing", "progress": 0, "total_pages": 0, "current_page": 0}
    
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    
    # Ensure folder exists
    save_folder(folder)
    
    results = []
    
    for file in files:
        try:
            file_path = os.path.join(upload_dir, file.filename)
            # Get file size
            file.file.seek(0, os.SEEK_END)
            file_size = file.file.tell()
            file.file.seek(0)
            
            print(f"[UPLOAD] File received: {file.filename}")
            print(f"[UPLOAD] File size: {file_size} bytes")
            
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            print(f"--- [API] Processing {file.filename} ---")
            
            # Progress callback for ingestion
            def progress_callback(current, total):
                if job_id:
                    upload_stats[job_id]["current_page"] = current
                    upload_stats[job_id]["total_pages"] = total
                    upload_stats[job_id]["progress"] = int((current / total) * 100)
                    print(f"--- [API] Job {job_id} Progress: {current}/{total} ---")

            # Trigger ingestion
            ingest_result = ingest_file(file_path, folder_name=folder, progress_callback=progress_callback)
            
            status = "success"
            error = None
            if "error" in ingest_result:
                status = "failed"
                error = ingest_result["error"]
                print(f"--- [API] Error ingesting {file.filename}: {error} ---")
            else:
                print(f"[UPLOAD] Upload successful")
                print(f"--- [API] Successfully ingested {file.filename} ---", flush=True)
                
            results.append({
                "filename": file.filename,
                "status": status,
                "details": ingest_result,
                "error": error
            })
            
        except Exception as e:
            print(f"--- [API] Exception processing {file.filename}: {e} ---")
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            })

    # Return 200 OK with detailed results for client to parse
    print(f"--- [API] Upload Process Completed. Results: {len(results)} files processed ---")
    for r in results:
        print(f"  - {r['filename']}: {r['status']} {'(' + r['error'] + ')' if r['error'] else ''}")
    
    if job_id:
        upload_stats[job_id]["status"] = "completed"
    
    return {"status": "completed", "results": results}

@app.get("/upload-status/{job_id}", dependencies=[Depends(get_api_key)])
def get_upload_status(job_id: str):
    if job_id not in upload_stats:
        return {"error": "Job ID not found"}
    return upload_stats[job_id]

@app.post("/query/", dependencies=[Depends(get_api_key)])
def query_index(request: QueryRequest):
    print(f"--- [API] Received Query Request: {request.question} (Folder: {request.folder}) ---")
    try:
        # Trigger RAG
        answer = query_rag(request.question, folder_name=request.folder)
        print(f"--- [API] Query Success ---")
        return {"answer": answer}
    except Exception as e:
        print(f"--- [API] Query Error: {e} ---")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/collection/")
def debug_collection():
    print("--- [DEBUG] Inspecting Collection ---")
    try:
        client = get_qdrant_client()
        count = client.count(collection_name=COLLECTION_NAME).count
        collections = client.get_collections()
        try:
            info = client.get_collection(collection_name=COLLECTION_NAME)
            info_dict = info.model_dump()
        except Exception:
            info_dict = "Error getting collection info"
            
        return {
            "collection_name": COLLECTION_NAME,
            "exists": any(c.name == COLLECTION_NAME for c in collections.collections),
            "point_count": count,
            "collection_info": info_dict
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    print("\n🚀 [SERVER] NeuralRAG Backend starting on http://0.0.0.0:8000", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)
