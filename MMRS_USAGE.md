# MMRS System Usage Guide

## Overview

The Multi-Model Recursive Synthesis (MMRS) system integrates multiple language models to generate improved responses through recursive refinement. This guide explains how to set up, run, and troubleshoot the system.

## Quick Start

### Option 1: All-in-One Startup (Recommended)

1. **Start both backend and frontend with a single command**:
   ```bash
   ./start_app.sh
   ```
   This will:
   - Start the backend server on port 5001
   - Open a new terminal window and start the frontend on port 3000
   - Configure all necessary environment variables

2. **Stop both backend and frontend**:
   ```bash
   ./stop_app.sh
   ```

### Option 2: Manual Startup

1. **Start the backend server**:
   ```bash
   ./start_mmrs.sh
   ```

2. **Start the frontend** (in a separate terminal):
   ```bash
   npm start
   ```

3. **Access the web interface**:
   Open your browser and navigate to http://localhost:3000

4. **Stop the backend server** when finished:
   ```bash
   ./stop_mmrs.sh
   ```

## Troubleshooting

If you encounter issues, use one of these diagnostic tools:

```bash
# Basic debugging
./debug_mmrs.sh

# Comprehensive API diagnostics
./api_diagnostics.py

# Frontend-to-API connection test
./test_frontend_api.js
```

The API diagnostics tool performs detailed checks including:
- Internet connectivity
- Backend server status
- API key configuration
- External API endpoint connectivity
- Model functionality tests
- Backend log analysis

The frontend-to-API test script verifies:
- Backend connectivity
- API endpoint functionality
- The exact request flow used by the React app

### Common Issues and Solutions

1. **"Failed to load" error**:
   - Ensure the backend server is running (`./start_mmrs.sh`)
   - Check if the correct port is being used (default: 5001)
   - Verify that all required packages are installed
   - Make sure the frontend is configured to use the correct backend URL
   - Run `./api_diagnostics.py` for detailed API connection diagnostics
   - Run `./test_frontend_api.js` to test the frontend-to-API connection

2. **Port conflicts**:
   - The backend should run on port 5001
   - The frontend should run on port 3000
   - If you see port conflicts, use `./stop_app.sh` to clean up all processes
   - Check running processes with `lsof -i :5001` and `lsof -i :3000`

3. **API Key Issues**:
   - Configure API keys through the web interface
   - Ensure API keys are valid for the selected models
   - Use `./api_diagnostics.py` to verify API key configuration and connectivity

4. **Backend Not Starting**:
   - Check if the port is already in use
   - Verify that all dependencies are installed
   - Check the logs for detailed error messages:
     ```bash
     ./start_mmrs.sh > backend_output.log 2>&1
     ```
   - Run the diagnostics tool: `./api_diagnostics.py`

5. **Model Connection Issues**:
   - Run `./api_diagnostics.py` to test connectivity to each model's API
   - Check if your API keys are valid and properly configured
   - Verify internet connectivity for external API services

6. **Frontend-to-Backend Connection Issues**:
   - Run `./test_frontend_api.js` to verify the connection
   - Check that the API URL in the frontend matches the backend port (5001)
   - Ensure CORS is properly configured in the backend

## System Architecture

The MMRS system consists of:

1. **Backend (Python/FastAPI)**:
   - Handles model integration and synthesis logic
   - Manages API requests and responses
   - Implements the recursive synthesis algorithm

2. **Frontend (React)**:
   - Provides a user interface for model selection
   - Handles API key management
   - Displays synthesis results

## Advanced Configuration

### Environment Variables

Edit the `.env` file to configure:
- Server host and port
- Default API keys
- Other system settings

For the frontend, you can create a `.env.development.local` file with:
```
PORT=3000
REACT_APP_API_URL=http://localhost:5001
```

### Adding New Models

To add a new model:
1. Update the `src/config/models.ts` file with the new model configuration
2. Implement the model integration in the backend

## Logs and Debugging

- Backend logs: `backend_output.log` (when using the logging option)
- Basic diagnostics: Run `./debug_mmrs.sh` for system diagnostics
- Comprehensive API diagnostics: Run `./api_diagnostics.py` for detailed API connection testing
- Frontend-to-API testing: Run `./test_frontend_api.js` to verify the frontend connection

## Security Notes

- API keys are stored in memory only and not persisted
- Use HTTPS in production environments
- Consider implementing proper authentication for production use 