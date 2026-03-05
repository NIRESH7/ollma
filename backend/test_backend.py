import requests
import os

def test_api():
    url = "http://127.0.0.1:8000/upload/"
    test_file = "test_upload.txt"
    
    # Create a dummy test file
    with open(test_file, "w") as f:
        f.write("This is a test file for verifying the ingestion logic and logging.")
    
    try:
        print(f"Sending request to {url}...")
        with open(test_file, "rb") as f:
            files = [("files", (test_file, f, "text/plain"))]
            data = {"folder": "test_verification", "job_id": "test_job_123"}
            response = requests.post(url, files=files, data=data)
            
        print(f"Response Status: {response.status_code}")
        print(f"Response Body: {response.json()}")
        
    except Exception as e:
        print(f"Error during test: {e}")
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)

if __name__ == "__main__":
    test_api()
