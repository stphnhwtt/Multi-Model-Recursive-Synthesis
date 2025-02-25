#!/bin/bash

# Text colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting MMRS Application...${NC}"

# Check if PORT is already set in .env
if grep -q "^PORT=" .env; then
    # Update PORT in .env to 5001
    sed -i '' 's/^PORT=.*/PORT=5001/' .env
else
    # Add PORT=5001 to .env
    echo "PORT=5001" >> .env
fi

# Create a temporary file for React environment variables
cat > .env.development.local << EOL
PORT=3000
REACT_APP_API_URL=http://localhost:5001
EOL

# Start the backend in the background
echo -e "${BLUE}Starting backend server on port 5001...${NC}"
./start_mmrs.sh > backend_output.log 2>&1 &
BACKEND_PID=$!
echo $BACKEND_PID > backend.pid

# Wait for backend to start
echo -e "${YELLOW}Waiting for backend to start...${NC}"
MAX_RETRIES=30
COUNT=0
while [ $COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:5001/status > /dev/null; then
        echo -e "${GREEN}Backend started successfully!${NC}"
        break
    fi
    sleep 1
    COUNT=$((COUNT+1))
    echo -n "."
done

if [ $COUNT -eq $MAX_RETRIES ]; then
    echo -e "${RED}Failed to start backend!${NC}"
    echo -e "${YELLOW}Check backend_output.log for details${NC}"
    exit 1
fi

# Start the frontend
echo -e "${BLUE}Starting frontend on port 3000...${NC}"
echo -e "${YELLOW}This will open a new terminal window...${NC}"

if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    osascript -e 'tell app "Terminal" to do script "cd '$PWD' && npm start"'
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    if command -v gnome-terminal &> /dev/null; then
        gnome-terminal -- bash -c "cd '$PWD' && npm start; exec bash"
    elif command -v xterm &> /dev/null; then
        xterm -e "cd '$PWD' && npm start; exec bash" &
    else
        echo -e "${RED}Could not find a suitable terminal emulator.${NC}"
        echo -e "${YELLOW}Please open a new terminal and run: npm start${NC}"
    fi
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    start cmd /k "cd /d %CD% && npm start"
else
    echo -e "${YELLOW}Could not automatically open a terminal for the frontend.${NC}"
    echo -e "${YELLOW}Please open a new terminal and run: npm start${NC}"
fi

echo -e "${GREEN}MMRS Application startup complete!${NC}"
echo -e "${BLUE}Backend URL: ${NC}http://localhost:5001"
echo -e "${BLUE}Frontend URL: ${NC}http://localhost:3000"
echo -e "${YELLOW}To stop the application, run: ./stop_app.sh${NC}" 