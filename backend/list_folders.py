from qdrant_client import QdrantClient

def list_folders():
    client = QdrantClient(path="qdrant_db")
    collection_name = "local_documents"
    points, _ = client.scroll(collection_name=collection_name, limit=100, with_payload=True)
    
    folders = set()
    files = set()
    for p in points:
        meta = p.payload.get("metadata", {})
        folders.add(meta.get("folder"))
        files.add(meta.get("file_name") or meta.get("source"))
    
    print(f"Folders found: {folders}")
    print(f"Files found: {files}")

if __name__ == "__main__":
    list_folders()
