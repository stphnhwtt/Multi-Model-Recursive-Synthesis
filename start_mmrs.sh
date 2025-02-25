#!/bin/bash

# Text colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting MMRS Backend Setup...${NC}"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to create virtual environment. Please install python3-venv${NC}"
        exit 1
    fi
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to activate virtual environment${NC}"
    exit 1
fi

# Install/upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
python -m pip install --upgrade pip

# Install required packages
echo -e "${YELLOW}Installing required packages...${NC}"
pip install fastapi uvicorn python-dotenv requests aiohttp numpy

# Check if port 5001 is in use
if lsof -Pi :5001 -sTCP:LISTEN -t >/dev/null ; then
    echo -e "${RED}Port 5001 is already in use. Attempting to free it...${NC}"
    lsof -ti :5001 | xargs kill -9
    sleep 2
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Creating .env file...${NC}"
    cat > .env << EOL
# Server Settings
HOST=localhost
PORT=5001

# API Keys (replace with your actual keys)
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
HUGGINGFACE_API_KEY=your_huggingface_key_here
EOL
fi

# Start the backend server
echo -e "${GREEN}Starting MMRS backend server...${NC}"
echo -e "${YELLOW}The server will be available at http://localhost:5001${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
echo ""

# Start uvicorn with error handling
uvicorn backend.main:app --host localhost --port 5001 --reload

# If uvicorn fails, show error
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to start the server. Please check the error messages above.${NC}"
    exit 1
fi 