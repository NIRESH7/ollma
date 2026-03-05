import sys
import os
import json

# Add current directory to path
sys.path.append(os.getcwd())

from rag import query_rag

def test_rag_v2():
    question = "for this item GLICO STICK STRAWBERRY 38G what is the upc"
    print(f"Testing Query: {question}")
    
    # We use 'qdrant' folder as seen in user's screenshot
    answer = query_rag(question, folder_name="qdrant")
    
    print("\n" + "="*50)
    print(f"FINAL ANSWER: {answer}")
    print("="*50)

if __name__ == "__main__":
    test_rag_v2()
