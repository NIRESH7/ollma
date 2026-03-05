import sys
import os
from qdrant_client import QdrantClient

def dump_qdrant():
    client = QdrantClient(path="qdrant_db")
    collection_name = "local_documents"
    
    # Scroll through points in the 'qdrant' folder
    from qdrant_client.http import models
    points, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="metadata.folder", match=models.MatchValue(value="qdrant"))]
        ),
        limit=20,
        with_payload=True
    )
    
    print(f"Found {len(points)} points in folder 'qdrant'")
    for p in points:
        print("-" * 50)
        print(f"ID: {p.id}")
        payload = p.payload
        print(f"Content: {payload.get('page_content')}")
        print(f"Metadata: {payload.get('metadata')}")

if __name__ == "__main__":
    dump_qdrant()
