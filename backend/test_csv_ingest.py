import pandas as pd
import os
import sys

# Add backend to path
sys.path.append(os.getcwd())

from ingestion import ingest_file

def create_sample_csv():
    data = {
        "Product": ["Pocky Strawberry", "Pocky Chocolate", "Pocky Matcha"],
        "Weight": ["38g", "48g", "35g"],
        "GTIN": ["1234567890123", "9876543210987", "4561237894561"],
        "Price": ["1.99", "2.49", "1.99"]
    }
    df = pd.DataFrame(data)
    csv_path = "uploads/test_sample.csv"
    os.makedirs("uploads", exist_ok=True)
    df.to_csv(csv_path, index=False)
    return csv_path

def test_csv_ingestion():
    csv_path = create_sample_csv()
    print(f"Testing CSV Ingestion for {csv_path}...")
    
    # We will try to ingest it into a test folder
    result = ingest_file(csv_path, folder_name="csv_test")
    
    print("\n" + "="*50)
    print(f"INGESTION RESULT: {result}")
    print("="*50)

if __name__ == "__main__":
    test_csv_ingestion()
