#!/usr/bin/env python3
"""
MMRS API Test Tool (Python Version)
This script tests the MMRS API with actual API keys from the environment.
"""

import os
import sys
import json
import requests
from dotenv import load_dotenv
import time

# Load environment variables from .env file
load_dotenv()

# ANSI colors
RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'  # No Color

def print_colored(color, message):
    """Print a colored message."""
    print(f"{color}{message}{NC}")

def check_backend():
    """Check if the backend is running."""
    print_colored(BLUE, "\nChecking if backend is running...")
    try:
        response = requests.get("http://localhost:5001/status", timeout=5)
        if response.status_code == 200:
            print_colored(GREEN, "Backend is running!")
            return True
        else:
            print_colored(RED, f"Backend returned status code {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print_colored(RED, f"Backend is not running: {e}")
        print_colored(RED, "Please start it with ./start_mmrs.sh or ./start_app.sh")
        return False

def get_api_keys():
    """Get API keys from environment variables."""
    api_keys = {}
    
    # Check for OpenAI API key
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key and openai_key != "your_openai_key_here":
        api_keys["OPENAI_API_KEY"] = openai_key
    
    # Check for Anthropic API key
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key and anthropic_key != "your_anthropic_key_here":
        api_keys["ANTHROPIC_API_KEY"] = anthropic_key
    
    # Check for Hugging Face API key
    hf_key = os.getenv("HUGGINGFACE_API_KEY")
    if hf_key and hf_key != "your_huggingface_key_here":
        api_keys["HUGGINGFACE_API_KEY"] = hf_key
    
    # Check for Grok API key
    grok_key = os.getenv("GROK_API_KEY")
    if grok_key and grok_key != "your_grok_key_here":
        api_keys["GROK_API_KEY"] = grok_key
    
    return api_keys

def test_model(model, api_keys):
    """Test a specific model."""
    test_prompt = "This is a test prompt to verify the API connection. Please respond with a short confirmation."
    
    print_colored(BLUE, f"\nTesting model: {YELLOW}{model}{NC}")
    
    # Create request payload
    payload = {
        "models": [model],
        "prompt": test_prompt,
        "iterations": 1,
        "apiKeys": api_keys
    }
    
    # Make the request
    print("Sending request to backend...")
    try:
        response = requests.post(
            "http://localhost:5001/synthesize",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30  # Longer timeout for API calls
        )
        
        if response.status_code != 200:
            print_colored(RED, f"Error: HTTP {response.status_code}")
            try:
                print(json.dumps(response.json(), indent=2))
            except:
                print(response.text)
            return False
        
        response_data = response.json()
        if "error" in response_data:
            print_colored(RED, f"Error testing {model}:")
            print(json.dumps(response_data, indent=2))
            return False
        else:
            print_colored(GREEN, "Success! Response:")
            print(json.dumps(response_data, indent=2))
            return True
    except requests.exceptions.RequestException as e:
        print_colored(RED, f"Request failed: {e}")
        return False

def main():
    """Main function."""
    print_colored(YELLOW, "MMRS API Test Tool (Python Version)")
    print_colored(YELLOW, "================================")
    
    # Check if backend is running
    if not check_backend():
        sys.exit(1)
    
    # Get API keys
    api_keys = get_api_keys()
    available_keys = list(api_keys.keys())
    
    if available_keys:
        print_colored(GREEN, f"\nFound {len(available_keys)} API keys in environment:")
        for key in available_keys:
            print(f"- {key}")
    else:
        print_colored(YELLOW, "\nNo API keys found in environment variables.")
        print("You can still test the local LLaMA model.")
    
    # Test LLaMA (local model that doesn't require API key)
    print_colored(YELLOW, "\nTesting LLaMA (local model)")
    test_model("llama", api_keys)
    
    # Test models based on available API keys
    if "OPENAI_API_KEY" in api_keys:
        print_colored(YELLOW, "\nTesting GPT-4 (using OPENAI_API_KEY)")
        test_model("gpt-4", api_keys)
    
    if "ANTHROPIC_API_KEY" in api_keys:
        print_colored(YELLOW, "\nTesting Claude (using ANTHROPIC_API_KEY)")
        test_model("claude", api_keys)
    
    if "HUGGINGFACE_API_KEY" in api_keys:
        print_colored(YELLOW, "\nTesting Hugging Face (using HUGGINGFACE_API_KEY)")
        test_model("huggingface", api_keys)
    
    if "GROK_API_KEY" in api_keys:
        print_colored(YELLOW, "\nTesting Grok (using GROK_API_KEY)")
        test_model("grok", api_keys)
    
    print_colored(GREEN, "\nAPI testing complete!")
    print_colored(YELLOW, "If you encountered any errors, please check:")
    print("1. Your API keys are correctly set in the .env file")
    print("2. The backend is properly connecting to the model APIs")
    print("3. The backend logs for more detailed error information (backend_output.log)")

if __name__ == "__main__":
    main() 