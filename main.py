import uvicorn
import sys
import os

if __name__ == "__main__":
    # Ensure root is in path
    sys.path.append(os.getcwd())
    uvicorn.run("src.api.server:app", host="127.0.0.1", port=8000, log_level="info")
