#!/bin/bash

# Text colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${YELLOW}MMRS Debug Tool${NC}"
echo -e "${YELLOW}================${NC}"

# Check environment
echo -e "\n${BLUE}Checking environment...${NC}"
echo -e "Python version:"
python3 --version
echo -e "\nPython path:"
which python3

# Check virtual environment
echo -e "\n${BLUE}Checking virtual environment...${NC}"
if [ -d "venv" ]; then
    echo -e "${GREEN}Virtual environment exists${NC}"
    
    # Activate virtual environment
    source venv/bin/activate
    
    echo -e "\nInstalled packages:"
    pip list | grep -E "fastapi|uvicorn|python-dotenv|requests|aiohttp|numpy"
else
    echo -e "${RED}Virtual environment not found${NC}"
fi

# Check file structure
echo -e "\n${BLUE}Checking file structure...${NC}"
echo -e "MMRS.py exists: $([ -f "MMRS.py" ] && echo "${GREEN}Yes${NC}" || echo "${RED}No${NC}")"
echo -e "backend/main.py exists: $([ -f "backend/main.py" ] && echo "${GREEN}Yes${NC}" || echo "${RED}No${NC}")"
echo -e "backend/__init__.py exists: $([ -f "backend/__init__.py" ] && echo "${GREEN}Yes${NC}" || echo "${RED}No${NC}")"
echo -e ".env exists: $([ -f ".env" ] && echo "${GREEN}Yes${NC}" || echo "${RED}No${NC}")"

# Check port status
echo -e "\n${BLUE}Checking port status...${NC}"
if lsof -Pi :5001 -sTCP:LISTEN -t >/dev/null ; then
    PID=$(lsof -Pi :5001 -sTCP:LISTEN -t)
    echo -e "${YELLOW}Port 5001 is in use by process ${PID}${NC}"
    echo -e "Process details:"
    ps -p $PID -o command=
else
    echo -e "${GREEN}Port 5001 is available${NC}"
fi

# Test backend connection
echo -e "\n${BLUE}Testing backend connection...${NC}"
curl -s -o /dev/null -w "%{http_code}" http://localhost:5001/status
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}Backend is responding${NC}"
    echo -e "\nStatus response:"
    curl -s http://localhost:5001/status | python3 -m json.tool
else
    echo -e "\n${RED}Backend is not responding${NC}"
fi

# Check logs if available
echo -e "\n${BLUE}Checking for error logs...${NC}"
if [ -f "backend_error.log" ]; then
    echo -e "${YELLOW}Last 10 lines of error log:${NC}"
    tail -n 10 backend_error.log
else
    echo -e "${YELLOW}No error log found${NC}"
fi

echo -e "\n${YELLOW}Debug complete${NC}"
echo -e "${YELLOW}If you're still having issues, try running:${NC}"
echo -e "  ${GREEN}./start_mmrs.sh > backend_output.log 2>&1${NC}"
echo -e "${YELLOW}This will capture all output to a log file for further debugging.${NC}" 