import subprocess
import sys
import os
import time
import requests
from typing import Optional

def check_backend_status(port: int = 5000) -> bool:
    try:
        response = requests.get(f"http://localhost:{port}/status")
        return response.status_code == 200
    except requests.RequestException:
        return False

def start_backend(python_path: Optional[str] = None) -> subprocess.Popen:
    if sys.platform == "win32":
        python_cmd = python_path or "python"
        creationflags = subprocess.CREATE_NEW_CONSOLE
    else:
        python_cmd = python_path or "python3"
        creationflags = 0
    
    return subprocess.Popen(
        [python_cmd, "backend/main.py"],
        creationflags=creationflags
    )

def main():
    # Check if backend is already running
    if check_backend_status():
        print("Backend is already running!")
        return

    # Start the backend
    process = start_backend()
    
    # Wait for backend to start (max 30 seconds)
    start_time = time.time()
    while time.time() - start_time < 30:
        if check_backend_status():
            print("Backend started successfully!")
            return
        time.sleep(1)
        print("Waiting for backend to start...")
    
    print("Failed to start backend!")
    process.terminate()
    sys.exit(1)

if __name__ == "__main__":
    main() 