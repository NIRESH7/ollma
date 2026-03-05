import uvicorn
import os

if __name__ == "__main__":
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    print("\n🚀 [SERVER] Starting NeuralRAG Backend...")
    print("--- [CONFIG] Reloading enabled (Excluding data, uploads, qdrant_db) ---")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_excludes=["data/*", "uploads/*", "qdrant_db/*"]
    )
