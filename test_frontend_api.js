#!/usr/bin/env node

/**
 * MMRS Frontend-to-API Connection Test
 * 
 * This script tests the connection between the frontend and backend API
 * by simulating the exact request flow that happens in the React app.
 */

// Import fetch - CommonJS version
const fetch = (...args) => import('node-fetch').then(({default: fetch}) => fetch(...args));

// Text colors for console output
const RED = '\x1b[31m';
const GREEN = '\x1b[32m';
const YELLOW = '\x1b[33m';
const BLUE = '\x1b[34m';
const RESET = '\x1b[0m';

// Configuration
const API_BASE_URL = 'http://localhost:5001';
const TIMEOUT_MS = 30000;

/**
 * Print a colored message to the console
 */
function printColored(color, message) {
  console.log(`${color}${message}${RESET}`);
}

/**
 * Check if the backend is running
 */
async function checkBackendStatus() {
  printColored(BLUE, '\nChecking if backend is running...');
  
  try {
    const response = await fetch(`${API_BASE_URL}/status`);
    
    if (response.ok) {
      const data = await response.json();
      printColored(GREEN, '✓ Backend is running!');
      console.log(`  Status: ${data.status}`);
      console.log(`  Uptime: ${data.uptime} seconds`);
      console.log(`  Version: ${data.version}`);
      return true;
    } else {
      printColored(RED, `✗ Backend returned status code ${response.status}`);
      return false;
    }
  } catch (error) {
    printColored(RED, `✗ Backend is not running: ${error.message}`);
    printColored(YELLOW, 'Make sure the backend is started with ./start_mmrs.sh or ./start_app.sh');
    return false;
  }
}

/**
 * Test the synthesize endpoint with a simple prompt
 */
async function testSynthesizeEndpoint() {
  printColored(BLUE, '\nTesting synthesize endpoint with LLaMA model...');
  
  const testPayload = {
    models: ['llama'],
    prompt: 'This is a test prompt from the frontend API test script.',
    iterations: 1,
    apiKeys: {}
  };
  
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), TIMEOUT_MS);
    
    const response = await fetch(`${API_BASE_URL}/synthesize`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(testPayload),
      signal: controller.signal
    });
    
    clearTimeout(timeoutId);
    
    if (response.ok) {
      const data = await response.json();
      printColored(GREEN, '✓ Synthesize endpoint is working!');
      printColored(YELLOW, 'Response:');
      console.log(JSON.stringify(data, null, 2));
      return true;
    } else {
      printColored(RED, `✗ Synthesize endpoint returned status code ${response.status}`);
      try {
        const errorData = await response.json();
        console.log(JSON.stringify(errorData, null, 2));
      } catch (e) {
        console.log(await response.text());
      }
      return false;
    }
  } catch (error) {
    printColored(RED, `✗ Error testing synthesize endpoint: ${error.message}`);
    return false;
  }
}

/**
 * Test the exact request flow that happens in the React app
 */
async function testReactAppFlow() {
  printColored(BLUE, '\nSimulating the exact request flow from the React app...');
  
  // This simulates the ApiService.synthesize method in the React app
  async function simulateApiServiceSynthesize(selectedModels, prompt, iterations) {
    // Validation (same as in the React app)
    if (!selectedModels.length) {
      return { result: '', error: 'Please select at least one model' };
    }
    if (!prompt.trim()) {
      return { result: '', error: 'Please enter a prompt' };
    }
    if (iterations < 1 || iterations > 5) {
      return { result: '', error: 'Iterations must be between 1 and 5' };
    }
    
    // We're only testing with LLaMA which doesn't need API keys
    const apiKeys = {};
    
    const request = {
      models: selectedModels,
      prompt,
      iterations,
      apiKeys
    };
    
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), TIMEOUT_MS);
      
      const response = await fetch(`${API_BASE_URL}/synthesize`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(
          errorData.message || 
          `Server error: ${response.status} ${response.statusText}`
        );
      }
      
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Synthesis API error:', error);
      return {
        result: '',
        error: error.message || 'An unexpected error occurred. Please try again.'
      };
    }
  }
  
  // Test parameters (simulating user input in the React app)
  const selectedModels = ['llama'];
  const prompt = 'This is a test prompt simulating user input from the React app.';
  const iterations = 1;
  
  printColored(YELLOW, 'Test parameters:');
  console.log(`  Models: ${selectedModels.join(', ')}`);
  console.log(`  Prompt: ${prompt}`);
  console.log(`  Iterations: ${iterations}`);
  
  try {
    const response = await simulateApiServiceSynthesize(selectedModels, prompt, iterations);
    
    if (response.error) {
      printColored(RED, `✗ Error: ${response.error}`);
      return false;
    }
    
    printColored(GREEN, '✓ React app flow is working correctly!');
    printColored(YELLOW, 'Response:');
    console.log(JSON.stringify(response, null, 2));
    return true;
  } catch (error) {
    printColored(RED, `✗ Error in React app flow: ${error.message}`);
    return false;
  }
}

/**
 * Main function
 */
async function main() {
  printColored(YELLOW, '=================================================');
  printColored(YELLOW, '  MMRS FRONTEND-TO-API CONNECTION TEST');
  printColored(YELLOW, '=================================================');
  
  // Step 1: Check if backend is running
  const backendRunning = await checkBackendStatus();
  if (!backendRunning) {
    printColored(RED, '\nCannot proceed with tests because backend is not running.');
    process.exit(1);
  }
  
  // Step 2: Test the synthesize endpoint directly
  const endpointWorking = await testSynthesizeEndpoint();
  
  // Step 3: Test the React app flow
  const reactFlowWorking = await testReactAppFlow();
  
  // Summary
  printColored(YELLOW, '\n=================================================');
  printColored(YELLOW, '  TEST SUMMARY');
  printColored(YELLOW, '=================================================');
  
  console.log(`Backend Status: ${backendRunning ? GREEN + '✓ Online' : RED + '✗ Offline'}${RESET}`);
  console.log(`Synthesize Endpoint: ${endpointWorking ? GREEN + '✓ Working' : RED + '✗ Failed'}${RESET}`);
  console.log(`React App Flow: ${reactFlowWorking ? GREEN + '✓ Working' : RED + '✗ Failed'}${RESET}`);
  
  if (backendRunning && endpointWorking && reactFlowWorking) {
    printColored(GREEN, '\n✓ All tests passed! The frontend can successfully connect to the API.');
  } else {
    printColored(RED, '\n✗ Some tests failed. Please check the errors above.');
  }
}

// Run the main function
main().catch(error => {
  printColored(RED, `Unhandled error: ${error.message}`);
  process.exit(1);
}); 