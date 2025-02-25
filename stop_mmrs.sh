#!/bin/bash

# Text colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Stopping MMRS Backend...${NC}"

# Function to check if a port is in use
check_port() {
    lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null
    return $?
}

# Function to get PID of process using a port
get_port_pid() {
    lsof -Pi :$1 -sTCP:LISTEN -t
}

# Try to stop the server gracefully first
if check_port 5001; then
    PID=$(get_port_pid 5001)
    if [ ! -z "$PID" ]; then
        echo -e "${YELLOW}Found backend process (PID: $PID)${NC}"
        
        # Try graceful shutdown first
        echo -e "${YELLOW}Attempting graceful shutdown...${NC}"
        kill $PID
        
        # Wait for up to 5 seconds for graceful shutdown
        COUNTER=0
        while [ $COUNTER -lt 5 ] && check_port 5001; do
            sleep 1
            let COUNTER=COUNTER+1
        done
        
        # If process is still running, force kill it
        if check_port 5001; then
            echo -e "${YELLOW}Graceful shutdown timed out, forcing termination...${NC}"
            kill -9 $PID
        fi
        
        # Final check
        if ! check_port 5001; then
            echo -e "${GREEN}Backend server stopped successfully${NC}"
        else
            echo -e "${RED}Failed to stop backend server${NC}"
            exit 1
        fi
    fi
else
    echo -e "${YELLOW}No backend server found running on port 5001${NC}"
fi

# Clean up any uvicorn temporary files
if [ -d "__pycache__" ]; then
    echo -e "${YELLOW}Cleaning up temporary files...${NC}"
    rm -rf __pycache__
fi

echo -e "${GREEN}MMRS Backend shutdown complete${NC}" 