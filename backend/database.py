from qdrant_client import QdrantClient
import os


# Using local persistent storage for a truly local-first setup
QDRANT_PATH = "qdrant_db"

_client_instance = None

def get_qdrant_client():
    global _client_instance
    if _client_instance is None:
        url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")
        
        if url:
            print(f"--- [DB] Initializing QdrantClient with URL: {url} ---")
            _client_instance = QdrantClient(url=url, api_key=api_key)
        else:
            print(f"--- [DB] Initializing QdrantClient with local path: {QDRANT_PATH} ---")
            _client_instance = QdrantClient(path=QDRANT_PATH)
    return _client_instance
