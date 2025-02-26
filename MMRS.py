import asyncio
import aiohttp
from typing import List, Callable, Any, Dict, Optional
import logging
from dataclasses import dataclass
import numpy as np
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for each model."""
    name: str
    endpoint: Optional[str] = None  # For API-based models
    local_func: Optional[Callable] = None  # For local models

async def call_model(config: ModelConfig, input_data: str, context: List[str] = None, api_keys=None) -> str:
    """Asynchronously call a model (API or local)."""
    try:
        if config.endpoint:
            # Log which model we're calling and if we have API keys
            logger.info(f"Calling model: {config.name} at {config.endpoint}")
            logger.info(f"API keys provided: {list(api_keys.keys()) if api_keys else 'None'}")
            
            async with aiohttp.ClientSession() as session:
                if "anthropic" in config.endpoint:
                    # Use Claude-specific handler
                    if not api_keys or "ANTHROPIC_API_KEY" not in api_keys:
                        raise ValueError("ANTHROPIC_API_KEY is required but not provided")
                    
                    api_key = api_keys.get("ANTHROPIC_API_KEY", "")
                    if not api_key:
                        raise ValueError("ANTHROPIC_API_KEY is empty")
                    
                    logger.info(f"Using ANTHROPIC_API_KEY: {'[KEY PROVIDED]' if api_key else '[EMPTY]'}")
                    return await call_claude_api(session, config.endpoint, api_key, input_data)
                elif "openai" in config.endpoint:
                    # OpenAI-specific handler would go here
                    if not api_keys or "OPENAI_API_KEY" not in api_keys:
                        raise ValueError("OPENAI_API_KEY is required but not provided")
                    
                    api_key = api_keys.get("OPENAI_API_KEY", "")
                    if not api_key:
                        raise ValueError("OPENAI_API_KEY is empty")
                    
                    logger.info(f"Using OPENAI_API_KEY: {'[KEY PROVIDED]' if api_key else '[EMPTY]'}")
                    return await call_openai_api(session, config.endpoint, api_key, input_data)
                elif "huggingface" in config.endpoint:
                    if not api_keys or "HUGGINGFACE_API_KEY" not in api_keys:
                        raise ValueError("HUGGINGFACE_API_KEY is required but not provided")
                    
                    api_key = api_keys.get("HUGGINGFACE_API_KEY", "")
                    if not api_key:
                        raise ValueError("HUGGINGFACE_API_KEY is empty")
                    
                    logger.info(f"Using HUGGINGFACE_API_KEY: {'[KEY PROVIDED]' if api_key else '[EMPTY]'}")
                    # Use generic handler with proper headers
                    headers = {"Authorization": f"Bearer {api_key}"}
                    
                    # Ensure the input data is properly formatted for HuggingFace
                    payload = {"inputs": input_data}
                    
                    logger.info(f"Sending request to HuggingFace API with prompt: {input_data[:50]}...")
                    async with session.post(config.endpoint, headers=headers, json=payload) as response:
                        if response.status == 200:
                            result = await response.json()
                            logger.info(f"HuggingFace API response structure: {type(result)}")
                            return result[0].get("generated_text", f"{config.name} failed")
                        else:
                            error_text = await response.text()
                            logger.error(f"HuggingFace API error: {response.status}, {error_text}")
                            return f"HuggingFace error: {response.status}"
                elif "grok" in config.endpoint:
                    if not api_keys or "GROK_API_KEY" not in api_keys:
                        raise ValueError("GROK_API_KEY is required but not provided")
                    
                    api_key = api_keys.get("GROK_API_KEY", "")
                    if not api_key:
                        raise ValueError("GROK_API_KEY is empty")
                    
                    logger.info(f"Using GROK_API_KEY: {'[KEY PROVIDED]' if api_key else '[EMPTY]'}")
                    # Assuming Grok API is similar to OpenAI's
                    headers = {"Authorization": f"Bearer {api_key}"}
                    
                    # Ensure the input data is properly formatted for Grok
                    payload = {
                        "messages": [{"role": "user", "content": input_data}],
                        "max_tokens": 1000
                    }
                    
                    logger.info(f"Sending request to Grok API with prompt: {input_data[:50]}...")
                    async with session.post(config.endpoint, headers=headers, json=payload) as response:
                        if response.status == 200:
                            result = await response.json()
                            logger.info(f"Grok API response structure: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
                            if "choices" in result and len(result["choices"]) > 0:
                                return result["choices"][0]["message"]["content"]
                            else:
                                logger.error(f"Unexpected Grok response structure: {result}")
                                return f"Grok error: Unexpected response structure"
                        else:
                            error_text = await response.text()
                            logger.error(f"Grok API error: {response.status}, {error_text}")
                            return f"Grok error: {response.status}"
                else:
                    # Generic handler for other APIs
                    logger.info(f"Using generic handler for {config.name}")
                    payload = {
                        "input": input_data,
                        "prompt": input_data,  # Include both common formats
                        "context": context or []
                    }
                    logger.info(f"Sending request to generic API with prompt: {input_data[:50]}...")
                    async with session.post(config.endpoint, json=payload) as response:
                        if response.status == 200:
                            result = await response.json()
                            logger.info(f"Generic API response structure: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
                            # Try different common response formats
                            if "output" in result:
                                return result["output"]
                            elif "response" in result:
                                return result["response"]
                            elif "result" in result:
                                return result["result"]
                            elif "generated_text" in result:
                                return result["generated_text"]
                            else:
                                logger.warning(f"Unknown response format from {config.name}: {result}")
                                return f"{config.name} response format unknown"
                        else:
                            error_text = await response.text()
                            logger.error(f"Generic API error: {response.status}, {error_text}")
                            return f"{config.name} error: {response.status}"
        elif config.local_func:  # Local model
            return config.local_func(input_data, context)
        else:
            raise ValueError(f"No valid endpoint or function for {config.name}")
    except Exception as e:
        logger.error(f"Error calling {config.name}: {e}")
        return f"{config.name} error: {str(e)}"

def combine_outputs(outputs: List[str], strategy: str = "concat") -> str:
    """Combine model outputs with configurable strategy."""
    if not outputs:
        return "No outputs to combine"
    if strategy == "concat":
        return " | ".join(outputs)
    elif strategy == "majority" and all(isinstance(o, str) for o in outputs):
        from collections import Counter
        return Counter(outputs).most_common(1)[0][0]
    elif strategy == "average" and all(o.replace('.', '').isdigit() for o in outputs):
        return str(np.mean([float(o) for o in outputs]))
    logger.warning(f"Unknown strategy {strategy}, defaulting to first output")
    return outputs[0]

class MultiModelRecursiveSynthesis:
    def __init__(self, models: List[ModelConfig], max_iterations: int = 3, score_threshold: float = 0.9, api_keys: Dict[str, str] = None):
        self.models = models
        self.max_iterations = max_iterations
        self.score_threshold = score_threshold
        self.api_keys = api_keys or {}  # Store API keys in the class

    async def process_initial_input(self, benchmark_input: str) -> List[str]:
        """Step 1: Asynchronously send input to all models."""
        tasks = [call_model(model, benchmark_input, api_keys=self.api_keys) for model in self.models]
        outputs = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Log full outputs
        for i, output in enumerate(outputs):
            model_name = self.models[i].name
            logger.info(f"Full output from {model_name}:")
            logger.info(f"{output}")
            
        return outputs

    async def recursive_refinement(self, initial_outputs: List[str]) -> List[str]:
        """Step 2 & 3: Feed outputs back for optimization."""
        tasks = []
        for i, model in enumerate(self.models):
            input_data = initial_outputs[i]
            context = [o for j, o in enumerate(initial_outputs) if j != i]
            tasks.append(call_model(model, input_data, context, api_keys=self.api_keys))
        
        outputs = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Log full outputs
        for i, output in enumerate(outputs):
            model_name = self.models[i].name
            logger.info(f"Full refined output from {model_name}:")
            logger.info(f"{output}")
            
        return outputs

    def evaluate(self, combined_output: str, benchmark: str) -> float:
        """Evaluate output against benchmark (customize this)."""
        # Example: simple string similarity (Levenshtein distance approximation)
        from difflib import SequenceMatcher
        score = SequenceMatcher(None, combined_output, benchmark).ratio()
        return score

    async def run(self, benchmark_input: str, recurse: bool = True) -> tuple[str, float, List[str], List[Dict]]:
        """Execute the full recursive synthesis process."""
        iteration = 0
        current_input = benchmark_input

        # Store initial outputs to return to the frontend
        initial_outputs = None
        
        # Track all iterations for display
        all_iterations = []

        while iteration < self.max_iterations:
            logger.info(f"Starting iteration {iteration + 1}")
            
            # Step 1: Initial outputs
            current_outputs = await self.process_initial_input(current_input)
            
            # Convert any exceptions to strings for better display
            current_outputs_cleaned = []
            for output in current_outputs:
                if isinstance(output, Exception):
                    current_outputs_cleaned.append(f"Error: {str(output)}")
                else:
                    current_outputs_cleaned.append(output)
            
            logger.info(f"Iteration {iteration + 1} - Initial Outputs (cleaned): {current_outputs_cleaned}")
            
            # Save the very first outputs
            if iteration == 0:
                initial_outputs = current_outputs_cleaned
                
                # For the first iteration, use the benchmark input as input for all models
                first_iteration_inputs = {}
                for i in range(len(self.models)):
                    first_iteration_inputs[i] = benchmark_input
            
            # Step 2 & 3: Recursive refinement
            optimized_outputs = await self.recursive_refinement(current_outputs_cleaned)
            
            # Clean optimized outputs as well
            optimized_outputs_cleaned = []
            for output in optimized_outputs:
                if isinstance(output, Exception):
                    optimized_outputs_cleaned.append(f"Error: {str(output)}")
                else:
                    optimized_outputs_cleaned.append(output)
                    
            logger.info(f"Iteration {iteration + 1} - Optimized Outputs (cleaned): {optimized_outputs_cleaned}")

            # Step 4: Combine outputs
            combined_output = combine_outputs(optimized_outputs_cleaned)
            logger.info(f"Iteration {iteration + 1} - Combined Output: {combined_output}")

            # Step 5: Evaluate
            score = self.evaluate(combined_output, benchmark_input)
            logger.info(f"Iteration {iteration + 1} - Score: {score}")
            
            # Store this iteration's data
            iteration_data = {
                "iteration": iteration + 1,
                "inputs": first_iteration_inputs if iteration == 0 else {i: current_outputs_cleaned[i] for i in range(len(current_outputs_cleaned))},
                "outputs": {i: optimized_outputs_cleaned[i] for i in range(len(optimized_outputs_cleaned))},
                "combined": combined_output,
                "score": score
            }
            all_iterations.append(iteration_data)

            if not recurse or score >= self.score_threshold:
                return combined_output, score, initial_outputs, all_iterations

            current_input = combined_output
            iteration += 1

        logger.warning(f"Max iterations ({self.max_iterations}) reached")
        return combined_output, score, initial_outputs, all_iterations

async def call_claude_api(session, endpoint, api_key, input_data):
    logger.info(f"Calling Claude API with endpoint: {endpoint}")
    logger.info(f"API key present: {bool(api_key)}")  # Don't log the actual key
    
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    # Ensure the input data is properly formatted for Claude
    # Claude expects a specific message format
    payload = {
        "model": "claude-3-opus-20240229",
        "messages": [{"role": "user", "content": input_data}],
        "max_tokens": 1000
    }
    
    logger.info(f"Sending request to Claude API with prompt: {input_data[:50]}...")
    try:
        logger.info(f"Sending request to Claude API with payload structure: {list(payload.keys())}")
        async with session.post(endpoint, headers=headers, json=payload) as response:
            status = response.status
            logger.info(f"Claude API response status: {status}")
            
            response_text = await response.text()
            logger.info(f"Claude API raw response: {response_text[:200]}...")
            
            if status == 200:
                try:
                    result = json.loads(response_text)
                    logger.info(f"Claude API parsed response structure: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
                    
                    # Check if the expected fields exist
                    if "content" in result and len(result["content"]) > 0 and "text" in result["content"][0]:
                        return result["content"][0]["text"]
                    else:
                        logger.error(f"Unexpected response structure: {result}")
                        return f"Claude error: Unexpected response structure"
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON response: {response_text}")
                    return f"Claude error: Invalid JSON response"
            else:
                logger.error(f"Claude API error: {status}, {response_text}")
                raise Exception(f"API error: {status}, {response_text}")
    except Exception as e:
        logger.error(f"Exception in call_claude_api: {str(e)}")
        raise

async def call_openai_api(session, endpoint, api_key, input_data):
    logger.info(f"Calling OpenAI API with endpoint: {endpoint}")
    logger.info(f"API key present: {bool(api_key)}")  # Don't log the actual key
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Ensure the input data is properly formatted for OpenAI
    payload = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": input_data}],
        "max_tokens": 1000
    }
    
    logger.info(f"Sending request to OpenAI API with prompt: {input_data[:50]}...")
    try:
        logger.info(f"Sending request to OpenAI API with payload structure: {list(payload.keys())}")
        async with session.post(endpoint, headers=headers, json=payload) as response:
            status = response.status
            logger.info(f"OpenAI API response status: {status}")
            
            response_text = await response.text()
            logger.info(f"OpenAI API raw response: {response_text[:200]}...")
            
            if status == 200:
                try:
                    result = json.loads(response_text)
                    logger.info(f"OpenAI API parsed response structure: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
                    
                    # Check if the expected fields exist
                    if "choices" in result and len(result["choices"]) > 0 and "message" in result["choices"][0] and "content" in result["choices"][0]["message"]:
                        return result["choices"][0]["message"]["content"]
                    else:
                        logger.error(f"Unexpected response structure: {result}")
                        return f"OpenAI error: Unexpected response structure"
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON response: {response_text}")
                    return f"OpenAI error: Invalid JSON response"
            else:
                logger.error(f"OpenAI API error: {status}, {response_text}")
                raise Exception(f"API error: {status}, {response_text}")
    except Exception as e:
        logger.error(f"Exception in call_openai_api: {str(e)}")
        raise

# Example usage
async def main():
    # Define models (mix of API and local)
    def local_model(input_data: str, context: List[str] = None) -> str:
        return f"LocalModel processed '{input_data}' with context {context or 'none'}"

    models = [
        ModelConfig(name="ModelA", endpoint="https://api.example.com/modelA"),
        ModelConfig(name="LocalModel", local_func=local_model),
        ModelConfig(name="Claude", endpoint="https://api.anthropic.com/v1/messages"),
    ]
    synthesizer = MultiModelRecursiveSynthesis(models=models, max_iterations=3, score_threshold=0.9)

    benchmark_test = "Solve this problem: 2 + 2"
    final_output, final_score, initial_outputs, all_iterations = await synthesizer.run(benchmark_test, recurse=True)

    print(f"\nFinal Output: {final_output}")
    print(f"Final Score: {final_score}")
    print(f"Initial Outputs: {initial_outputs}")
    print(f"All Iterations: {all_iterations}")

if __name__ == "__main__":
    asyncio.run(main())