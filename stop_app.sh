#!/bin/bash

# Text colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Stopping MMRS Application...${NC}"

# Stop the backend
echo -e "${BLUE}Stopping backend server...${NC}"
./stop_mmrs.sh

# Stop the frontend
echo -e "${BLUE}Stopping frontend...${NC}"
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    FRONTEND_PIDS=$(ps aux | grep "react-scripts start" | grep -v grep | awk '{print $2}')
    if [ ! -z "$FRONTEND_PIDS" ]; then
        echo -e "${YELLOW}Found frontend processes: $FRONTEND_PIDS${NC}"
        for PID in $FRONTEND_PIDS; do
            kill $PID 2>/dev/null
            echo -e "${GREEN}Stopped frontend process $PID${NC}"
        done
    else
        echo -e "${YELLOW}No frontend processes found${NC}"
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    FRONTEND_PIDS=$(ps aux | grep "react-scripts start" | grep -v grep | awk '{print $2}')
    if [ ! -z "$FRONTEND_PIDS" ]; then
        echo -e "${YELLOW}Found frontend processes: $FRONTEND_PIDS${NC}"
        for PID in $FRONTEND_PIDS; do
            kill $PID 2>/dev/null
            echo -e "${GREEN}Stopped frontend process $PID${NC}"
        done
    else
        echo -e "${YELLOW}No frontend processes found${NC}"
    fi
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    echo -e "${YELLOW}Please close the frontend terminal window manually${NC}"
else
    echo -e "${YELLOW}Please stop the frontend process manually${NC}"
fi

# Clean up any temporary files
echo -e "${BLUE}Cleaning up temporary files...${NC}"
if [ -f "backend.pid" ]; then
    rm backend.pid
fi

echo -e "${GREEN}MMRS Application shutdown complete!${NC}" 