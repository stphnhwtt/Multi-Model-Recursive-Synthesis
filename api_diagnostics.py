#!/usr/bin/env python3
"""
MMRS API Diagnostics Tool
This script performs detailed diagnostics on the API connections.
"""

import os
import sys
import json
import requests
import socket
import time
import traceback
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ANSI colors
RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
CYAN = '\033[0;36m'
NC = '\033[0m'  # No Color

def print_colored(color, message):
    """Print a colored message."""
    print(f"{color}{message}{NC}")

def print_header(message):
    """Print a header message."""
    print("\n" + "=" * 60)
    print_colored(YELLOW, message)
    print("=" * 60)

def check_internet_connection():
    """Check if there is an internet connection."""
    print_header("INTERNET CONNECTION TEST")
    try:
        # Try to connect to Google's DNS server
        socket.create_connection(("8.8.8.8", 53), timeout=5)
        print_colored(GREEN, "✓ Internet connection is available")
        return True
    except OSError:
        print_colored(RED, "✗ No internet connection available")
        return False

def check_backend():
    """Check if the backend is running."""
    print_header("BACKEND SERVER TEST")
    try:
        response = requests.get("http://localhost:5001/status", timeout=5)
        if response.status_code == 200:
            print_colored(GREEN, "✓ Backend is running")
            try:
                status_data = response.json()
                print(f"  Status: {status_data.get('status', 'unknown')}")
                print(f"  Uptime: {status_data.get('uptime', 'unknown')} seconds")
                print(f"  Version: {status_data.get('version', 'unknown')}")
            except:
                print_colored(YELLOW, "  Could not parse status response")
            return True
        else:
            print_colored(RED, f"✗ Backend returned status code {response.status_code}")
            print(response.text)
            return False
    except requests.exceptions.RequestException as e:
        print_colored(RED, f"✗ Backend is not running: {e}")
        return False

def check_api_keys():
    """Check if API keys are set in environment variables."""
    print_header("API KEY CHECK")
    
    api_keys = {}
    missing_keys = []
    
    # Check for OpenAI API key
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key and openai_key != "your_openai_key_here":
        api_keys["OPENAI_API_KEY"] = openai_key
        print_colored(GREEN, "✓ OPENAI_API_KEY is set")
    else:
        missing_keys.append("OPENAI_API_KEY")
        print_colored(RED, "✗ OPENAI_API_KEY is not set")
    
    # Check for Anthropic API key
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key and anthropic_key != "your_anthropic_key_here":
        api_keys["ANTHROPIC_API_KEY"] = anthropic_key
        print_colored(GREEN, "✓ ANTHROPIC_API_KEY is set")
    else:
        missing_keys.append("ANTHROPIC_API_KEY")
        print_colored(RED, "✗ ANTHROPIC_API_KEY is not set")
    
    # Check for Hugging Face API key
    hf_key = os.getenv("HUGGINGFACE_API_KEY")
    if hf_key and hf_key != "your_huggingface_key_here":
        api_keys["HUGGINGFACE_API_KEY"] = hf_key
        print_colored(GREEN, "✓ HUGGINGFACE_API_KEY is set")
    else:
        missing_keys.append("HUGGINGFACE_API_KEY")
        print_colored(RED, "✗ HUGGINGFACE_API_KEY is not set")
    
    # Check for Grok API key
    grok_key = os.getenv("GROK_API_KEY")
    if grok_key and grok_key != "your_grok_key_here":
        api_keys["GROK_API_KEY"] = grok_key
        print_colored(GREEN, "✓ GROK_API_KEY is set")
    else:
        missing_keys.append("GROK_API_KEY")
        print_colored(RED, "✗ GROK_API_KEY is not set")
    
    if not api_keys:
        print_colored(YELLOW, "\nNo valid API keys found. You can only test the local LLaMA model.")
    else:
        print_colored(GREEN, f"\nFound {len(api_keys)} valid API key(s)")
    
    return api_keys, missing_keys

def test_api_endpoints():
    """Test connectivity to the API endpoints."""
    print_header("API ENDPOINT CONNECTIVITY TEST")
    
    endpoints = [
        {
            "name": "OpenAI API",
            "url": "https://api.openai.com/v1/models",
            "key_name": "OPENAI_API_KEY",
            "header_name": "Authorization",
            "header_format": "Bearer {}"
        },
        {
            "name": "Anthropic API",
            "url": "https://api.anthropic.com/v1/messages",
            "key_name": "ANTHROPIC_API_KEY",
            "header_name": "x-api-key",
            "header_format": "{}"
        },
        {
            "name": "Hugging Face API",
            "url": "https://api-inference.huggingface.co/models",
            "key_name": "HUGGINGFACE_API_KEY",
            "header_name": "Authorization",
            "header_format": "Bearer {}"
        }
        # Grok API endpoint is not publicly documented, so we skip it
    ]
    
    for endpoint in endpoints:
        key = os.getenv(endpoint["key_name"])
        if not key or key == f"your_{endpoint['key_name'].lower()}_here":
            print_colored(YELLOW, f"⚠ Skipping {endpoint['name']} (no API key)")
            continue
        
        print_colored(BLUE, f"\nTesting {endpoint['name']} connectivity...")
        try:
            headers = {
                endpoint["header_name"]: endpoint["header_format"].format(key)
            }
            response = requests.get(
                endpoint["url"],
                headers=headers,
                timeout=10
            )
            
            if response.status_code in (200, 401, 403):  # 401/403 means the key might be wrong but the endpoint is reachable
                print_colored(GREEN, f"✓ {endpoint['name']} is reachable (status code: {response.status_code})")
                if response.status_code in (401, 403):
                    print_colored(YELLOW, "  Note: Authentication failed, but the endpoint is reachable")
            else:
                print_colored(RED, f"✗ {endpoint['name']} returned unexpected status code: {response.status_code}")
                print(f"  Response: {response.text[:200]}...")
        except requests.exceptions.RequestException as e:
            print_colored(RED, f"✗ Could not connect to {endpoint['name']}: {e}")

def test_local_model():
    """Test the local LLaMA model."""
    print_header("LOCAL MODEL TEST")
    
    print_colored(BLUE, "Testing LLaMA (local model)...")
    
    payload = {
        "models": ["llama"],
        "prompt": "This is a test prompt for the local LLaMA model.",
        "iterations": 1,
        "apiKeys": {}
    }
    
    try:
        response = requests.post(
            "http://localhost:5001/synthesize",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            response_data = response.json()
            if "result" in response_data:
                print_colored(GREEN, "✓ Local LLaMA model is working")
                print(f"  Response: {response_data['result']}")
                return True
            else:
                print_colored(RED, "✗ Local LLaMA model returned unexpected response")
                print(json.dumps(response_data, indent=2))
                return False
        else:
            print_colored(RED, f"✗ Request failed with status code: {response.status_code}")
            print(response.text)
            return False
    except requests.exceptions.RequestException as e:
        print_colored(RED, f"✗ Request failed: {e}")
        return False

def test_external_model(model_id, api_keys):
    """Test an external model."""
    print_colored(BLUE, f"\nTesting {model_id}...")
    
    payload = {
        "models": [model_id],
        "prompt": "This is a test prompt. Please respond with a short confirmation.",
        "iterations": 1,
        "apiKeys": api_keys
    }
    
    try:
        response = requests.post(
            "http://localhost:5001/synthesize",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            response_data = response.json()
            if "error" in response_data:
                print_colored(RED, f"✗ Error testing {model_id}:")
                print(json.dumps(response_data, indent=2))
                return False
            else:
                print_colored(GREEN, f"✓ {model_id} is working")
                print(f"  Response: {response_data.get('result', 'No result')}")
                return True
        else:
            print_colored(RED, f"✗ Request failed with status code: {response.status_code}")
            print(response.text)
            return False
    except requests.exceptions.RequestException as e:
        print_colored(RED, f"✗ Request failed: {e}")
        return False

def check_backend_logs():
    """Check the backend logs for errors."""
    print_header("BACKEND LOG CHECK")
    
    log_file = "backend_output.log"
    if not os.path.exists(log_file):
        print_colored(YELLOW, f"⚠ Log file {log_file} not found")
        return
    
    print_colored(BLUE, f"Checking {log_file} for errors...")
    
    try:
        with open(log_file, "r") as f:
            log_content = f.read()
        
        # Look for common error patterns
        error_patterns = [
            "Error:", "ERROR:", "Exception:", "Traceback", "Failed"
        ]
        
        found_errors = False
        for pattern in error_patterns:
            if pattern in log_content:
                found_errors = True
                print_colored(RED, f"✗ Found '{pattern}' in logs")
                
                # Extract the error context (3 lines before and after)
                lines = log_content.split("\n")
                for i, line in enumerate(lines):
                    if pattern in line:
                        start = max(0, i - 3)
                        end = min(len(lines), i + 4)
                        print_colored(YELLOW, "  Error context:")
                        for j in range(start, end):
                            if j == i:
                                print_colored(RED, f"  > {lines[j]}")
                            else:
                                print(f"    {lines[j]}")
        
        if not found_errors:
            print_colored(GREEN, "✓ No obvious errors found in logs")
    except Exception as e:
        print_colored(RED, f"✗ Error reading log file: {e}")

def main():
    """Main function."""
    print_colored(CYAN, """
    ╔═══════════════════════════════════════════════╗
    ║             MMRS API DIAGNOSTICS              ║
    ╚═══════════════════════════════════════════════╝
    """)
    
    # Check internet connection
    internet_available = check_internet_connection()
    if not internet_available:
        print_colored(RED, "\nWarning: No internet connection. External API tests will fail.")
    
    # Check if backend is running
    backend_running = check_backend()
    if not backend_running:
        print_colored(RED, "\nError: Backend is not running. Please start it with ./start_mmrs.sh or ./start_app.sh")
        sys.exit(1)
    
    # Check API keys
    api_keys, missing_keys = check_api_keys()
    
    # Test API endpoints
    if internet_available:
        test_api_endpoints()
    
    # Test models
    print_header("MODEL TESTS")
    
    # Test local model
    local_model_working = test_local_model()
    
    # Test external models based on available API keys
    if "OPENAI_API_KEY" in api_keys and internet_available:
        test_external_model("gpt-4", api_keys)
    
    if "ANTHROPIC_API_KEY" in api_keys and internet_available:
        test_external_model("claude", api_keys)
    
    if "HUGGINGFACE_API_KEY" in api_keys and internet_available:
        test_external_model("huggingface", api_keys)
    
    if "GROK_API_KEY" in api_keys and internet_available:
        test_external_model("grok", api_keys)
    
    # Check backend logs
    check_backend_logs()
    
    # Summary
    print_header("DIAGNOSTIC SUMMARY")
    
    if not internet_available:
        print_colored(RED, "✗ No internet connection")
    else:
        print_colored(GREEN, "✓ Internet connection available")
    
    if backend_running:
        print_colored(GREEN, "✓ Backend server is running")
    else:
        print_colored(RED, "✗ Backend server is not running")
    
    if local_model_working:
        print_colored(GREEN, "✓ Local LLaMA model is working")
    else:
        print_colored(RED, "✗ Local LLaMA model is not working")
    
    if missing_keys:
        print_colored(YELLOW, f"⚠ Missing API keys: {', '.join(missing_keys)}")
    else:
        print_colored(GREEN, "✓ All API keys are set")
    
    print_colored(YELLOW, "\nRecommendations:")
    if not backend_running:
        print("- Start the backend server with ./start_mmrs.sh or ./start_app.sh")
    
    if missing_keys:
        print("- Add missing API keys to your .env file or through the web interface")
    
    if not internet_available:
        print("- Check your internet connection")
    
    print_colored(CYAN, "\nDiagnostics complete!")

if __name__ == "__main__":
    main() 