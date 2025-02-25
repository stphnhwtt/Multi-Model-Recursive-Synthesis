#!/bin/bash

# Text colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${YELLOW}MMRS API Test Tool${NC}"
echo -e "${YELLOW}================${NC}"

# Check if backend is running
echo -e "\n${BLUE}Checking if backend is running...${NC}"
if ! curl -s http://localhost:5001/status > /dev/null; then
    echo -e "${RED}Backend is not running. Please start it with ./start_mmrs.sh or ./start_app.sh${NC}"
    exit 1
fi

echo -e "${GREEN}Backend is running!${NC}"

# Function to test a specific model
test_model() {
    local model=$1
    local api_key_name=$2
    local test_prompt="This is a test prompt to verify the API connection. Please respond with a short confirmation."
    
    echo -e "\n${BLUE}Testing model: ${YELLOW}$model${NC}"
    
    # Create API key JSON
    local api_keys="{}"
    if [ ! -z "$api_key_name" ]; then
        echo -e "Using API key: $api_key_name"
        api_keys="{\"$api_key_name\":\"YOUR_API_KEY_HERE\"}"
    fi
    
    # Create request payload
    local payload="{\"models\":[\"$model\"],\"prompt\":\"$test_prompt\",\"iterations\":1,\"apiKeys\":$api_keys}"
    
    # Make the request
    echo -e "Sending request to backend..."
    response=$(curl -s -X POST http://localhost:5001/synthesize \
        -H "Content-Type: application/json" \
        -d "$payload")
    
    # Check for errors
    if echo "$response" | grep -q "error"; then
        echo -e "${RED}Error testing $model:${NC}"
        echo "$response" | python3 -m json.tool
        return 1
    else
        echo -e "${GREEN}Success! Response:${NC}"
        echo "$response" | python3 -m json.tool
        return 0
    fi
}

# Test LLaMA (local model that doesn't require API key)
echo -e "\n${YELLOW}Testing LLaMA (local model)${NC}"
test_model "llama" ""

# Ask user which other models to test
echo -e "\n${YELLOW}Do you want to test other models that require API keys? (y/n)${NC}"
read -r test_others

if [[ "$test_others" == "y" || "$test_others" == "Y" ]]; then
    # Test GPT-4
    echo -e "\n${YELLOW}Do you want to test GPT-4? (y/n)${NC}"
    read -r test_gpt4
    if [[ "$test_gpt4" == "y" || "$test_gpt4" == "Y" ]]; then
        test_model "gpt-4" "OPENAI_API_KEY"
    fi
    
    # Test Claude
    echo -e "\n${YELLOW}Do you want to test Claude? (y/n)${NC}"
    read -r test_claude
    if [[ "$test_claude" == "y" || "$test_claude" == "Y" ]]; then
        test_model "claude" "ANTHROPIC_API_KEY"
    fi
    
    # Test Hugging Face
    echo -e "\n${YELLOW}Do you want to test Hugging Face? (y/n)${NC}"
    read -r test_hf
    if [[ "$test_hf" == "y" || "$test_hf" == "Y" ]]; then
        test_model "huggingface" "HUGGINGFACE_API_KEY"
    fi
    
    # Test Grok
    echo -e "\n${YELLOW}Do you want to test Grok? (y/n)${NC}"
    read -r test_grok
    if [[ "$test_grok" == "y" || "$test_grok" == "Y" ]]; then
        test_model "grok" "GROK_API_KEY"
    fi
fi

echo -e "\n${GREEN}API testing complete!${NC}"
echo -e "${YELLOW}If you encountered any errors, please check:${NC}"
echo -e "1. Your API keys are correctly configured in the web interface"
echo -e "2. The backend is properly connecting to the model APIs"
echo -e "3. The backend logs for more detailed error information (backend_output.log)" 