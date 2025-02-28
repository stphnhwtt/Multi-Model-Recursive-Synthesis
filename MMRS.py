import asyncio
import aiohttp
from typing import List, Callable, Any, Dict, Optional
import logging
from dataclasses import dataclass
import numpy as np
import json
import os
import random
import time

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
    # Set timeout and retry parameters
    timeout = aiohttp.ClientTimeout(total=900)  # 15 minute timeout (increased by 50% from 10 minutes)
    max_retries = 7  # Increase max retries
    base_retry_delay = 2  # seconds
    
    try:
        if config.endpoint:
            # Log which model we're calling and if we have API keys
            logger.info(f"Calling model: {config.name} at {config.endpoint}")
            logger.info(f"API keys provided: {list(api_keys.keys()) if api_keys else 'None'}")
            
            for attempt in range(max_retries):
                try:
                    async with aiohttp.ClientSession(timeout=timeout) as session:
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
                                    raise Exception(f"HuggingFace error: {response.status}")
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
                                        raise Exception(f"Grok error: Unexpected response structure")
                                else:
                                    error_text = await response.text()
                                    logger.error(f"Grok API error: {response.status}, {error_text}")
                                    raise Exception(f"Grok error: {response.status}")
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
                                        raise Exception(f"{config.name} response format unknown")
                                else:
                                    error_text = await response.text()
                                    logger.error(f"Generic API error: {response.status}, {error_text}")
                                    raise Exception(f"{config.name} error: {response.status}")
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    if attempt < max_retries - 1:
                        # More aggressive exponential backoff with jitter
                        retry_wait = base_retry_delay * (2 ** attempt) + (random.random() * base_retry_delay)
                        logger.warning(f"Request to {config.name} failed with error: {str(e)}. Retrying in {retry_wait:.2f} seconds... (Attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(retry_wait)
                    else:
                        logger.error(f"All retry attempts failed for {config.name}: {str(e)}")
                        return f"{config.name} error: Request failed after {max_retries} attempts: {str(e)}"
                except Exception as e:
                    # For non-connection errors, don't retry
                    logger.error(f"Error calling {config.name}: {e}")
                    return f"{config.name} error: {str(e)}"
                    
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
    def __init__(self, models: List[ModelConfig], max_iterations: int = 3, score_threshold: float = 0.9, api_keys: Dict[str, str] = None, distillation_model: Optional[ModelConfig] = None):
        self.models = models
        self.max_iterations = max_iterations
        self.score_threshold = score_threshold
        self.api_keys = api_keys or {}  # Store API keys in the class
        self.distillation_model = distillation_model  # Add distillation model
        self.max_prompt_length = 8000  # Maximum prompt length before chunking
        self.chunk_overlap = 500  # Overlap between chunks to maintain context
        self.operation_start_time = None  # Track operation start time

    async def _monitor_operation_time(self, operation_name, model_name):
        """Monitor long-running operations and log warnings."""
        warning_thresholds = [90, 180, 450, 720]  # Warning thresholds in seconds (increased by 50%)
        warned = set()
        
        while True:
            await asyncio.sleep(30)  # Check every 30 seconds
            if self.operation_start_time is None:
                break
                
            elapsed = time.time() - self.operation_start_time
            for threshold in warning_thresholds:
                if elapsed > threshold and threshold not in warned:
                    logger.warning(f"Operation '{operation_name}' for model '{model_name}' has been running for over {threshold} seconds")
                    warned.add(threshold)
                    
    async def _run_with_monitoring(self, coro, operation_name, model_name):
        """Run a coroutine with time monitoring."""
        self.operation_start_time = time.time()
        monitor_task = asyncio.create_task(self._monitor_operation_time(operation_name, model_name))
        
        try:
            result = await coro
            return result
        finally:
            self.operation_start_time = None
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

    def _chunk_long_prompt(self, prompt: str) -> List[str]:
        """Break a long prompt into manageable chunks."""
        if len(prompt) <= self.max_prompt_length:
            return [prompt]
        
        chunks = []
        # Split by paragraphs to avoid breaking in the middle of sentences
        paragraphs = prompt.split('\n\n')
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed the limit, save current chunk and start a new one
            if len(current_chunk) + len(paragraph) + 2 > self.max_prompt_length:
                if current_chunk:
                    chunks.append(current_chunk)
                
                # If a single paragraph is too long, split it by sentences
                if len(paragraph) > self.max_prompt_length:
                    sentences = paragraph.split('. ')
                    current_chunk = ""
                    
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) + 2 > self.max_prompt_length:
                            if current_chunk:
                                chunks.append(current_chunk)
                            current_chunk = sentence + ". "
                        else:
                            current_chunk += sentence + ". "
                else:
                    current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        # Add overlap between chunks for context
        overlapped_chunks = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                # Get the last part of the previous chunk for context
                prev_chunk = chunks[i-1]
                overlap = prev_chunk[-self.chunk_overlap:] if len(prev_chunk) > self.chunk_overlap else prev_chunk
                overlapped_chunks.append(f"[CONTINUED FROM PREVIOUS] {overlap}\n\n{chunk}")
            else:
                overlapped_chunks.append(chunk)
        
        logger.info(f"Split long prompt into {len(overlapped_chunks)} chunks")
        return overlapped_chunks

    async def process_long_input(self, benchmark_input: str) -> List[str]:
        """Process potentially long input by chunking if necessary."""
        chunks = self._chunk_long_prompt(benchmark_input)
        
        if len(chunks) == 1:
            # If there's only one chunk, process normally
            return await self.process_initial_input(benchmark_input)
        
        # Process each chunk separately
        all_outputs = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            chunk_outputs = await self.process_initial_input(chunk)
            all_outputs.append(chunk_outputs)
        
        # Combine outputs from all chunks for each model
        combined_outputs = []
        for model_idx in range(len(self.models)):
            model_outputs = [outputs[model_idx] for outputs in all_outputs if model_idx < len(outputs)]
            # Filter out error messages
            valid_outputs = [output for output in model_outputs if not isinstance(output, Exception) and not output.startswith("Error:")]
            
            if valid_outputs:
                combined_output = "\n\n".join(valid_outputs)
                combined_outputs.append(combined_output)
            else:
                combined_outputs.append(f"Error: Failed to process chunks for {self.models[model_idx].name}")
        
        return combined_outputs

    async def process_initial_input(self, benchmark_input: str) -> List[str]:
        """Step 1: Asynchronously send input to all models."""
        tasks = []
        for model in self.models:
            # Create a coroutine for the model call
            model_coro = call_model(model, benchmark_input, api_keys=self.api_keys)
            # Wrap it with monitoring
            task = asyncio.create_task(
                self._run_with_monitoring(model_coro, "initial processing", model.name)
            )
            # Add model name as a task attribute for better error reporting
            task.model_name = model.name
            tasks.append(task)
        
        # Use gather with return_exceptions=True to prevent one failure from stopping all tasks
        outputs = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Log full outputs
        for i, output in enumerate(outputs):
            model_name = self.models[i].name
            if isinstance(output, Exception):
                logger.error(f"Error from {model_name}: {str(output)}")
                # Convert exception to string for display
                outputs[i] = f"Error: {str(output)}"
            else:
                logger.info(f"Full output from {model_name}:")
                logger.info(f"{output}")
            
        return outputs

    async def recursive_refinement(self, initial_outputs: List[str]) -> List[str]:
        """Step 2 & 3: Feed outputs back for optimization."""
        # Read the synthesis prompt file
        synthesis_prompt = ""
        try:
            with open("Synthesis_Prompt.txt", "r") as f:
                synthesis_prompt = f.read().strip()
            logger.info(f"Successfully read Synthesis_Prompt.txt")
        except Exception as e:
            logger.error(f"Error reading Synthesis_Prompt.txt: {e}")
            logger.info("Continuing without synthesis prompt")
        
        tasks = []
        for i, model in enumerate(self.models):
            input_data = initial_outputs[i]
            context = [o for j, o in enumerate(initial_outputs) if j != i]
            
            # Prepend the synthesis prompt to the input data if available
            if synthesis_prompt:
                input_data = f"{synthesis_prompt}\n\n{input_data}"
                logger.info(f"Prepended synthesis prompt to input for model {model.name}")
            
            # Create a coroutine for the model call
            model_coro = call_model(model, input_data, context, api_keys=self.api_keys)
            # Wrap it with monitoring
            task = asyncio.create_task(
                self._run_with_monitoring(model_coro, "recursive refinement", model.name)
            )
            # Add model name as a task attribute for better error reporting
            task.model_name = model.name
            tasks.append(task)
        
        # Use gather with return_exceptions=True to prevent one failure from stopping all tasks
        outputs = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Log full outputs
        for i, output in enumerate(outputs):
            model_name = self.models[i].name
            if isinstance(output, Exception):
                logger.error(f"Error from {model_name} during refinement: {str(output)}")
                # Convert exception to string for display
                outputs[i] = f"Error: {str(output)}"
            else:
                logger.info(f"Full refined output from {model_name}:")
                logger.info(f"{output}")
            
        return outputs

    def evaluate(self, combined_output: str, benchmark: str) -> float:
        """Evaluate output against benchmark (customize this)."""
        # Example: simple string similarity (Levenshtein distance approximation)
        from difflib import SequenceMatcher
        score = SequenceMatcher(None, combined_output, benchmark).ratio()
        return score

    async def perform_distillation(self, final_output: str) -> str:
        """Perform the consensus distillation step using the selected distillation model."""
        if not self.distillation_model:
            logger.warning("No distillation model provided, skipping distillation step")
            return final_output
        
        # Read the consensus prompt file
        consensus_prompt = ""
        try:
            with open("Consensus_Prompt.txt", "r") as f:
                consensus_prompt = f.read().strip()
            logger.info(f"Successfully read Consensus_Prompt.txt")
            logger.info(f"Consensus prompt content (first 100 chars): {consensus_prompt[:100]}")
        except Exception as e:
            logger.error(f"Error reading Consensus_Prompt.txt: {e}")
            logger.info("Continuing without consensus prompt")
            return final_output
        
        # Combine the consensus prompt with the final output
        distillation_input = f"{consensus_prompt}\n\n{final_output}"
        logger.info(f"Sending final output to distillation model: {self.distillation_model.name}")
        logger.info(f"Combined input length: {len(distillation_input)}")
        
        # Call the distillation model with monitoring
        try:
            logger.info(f"Calling distillation model: {self.distillation_model.name}")
            
            # Create a coroutine for the model call
            model_coro = call_model(self.distillation_model, distillation_input, api_keys=self.api_keys)
            # Wrap it with monitoring
            distillation_output = await self._run_with_monitoring(
                model_coro, 
                "distillation", 
                self.distillation_model.name
            )
            
            if isinstance(distillation_output, Exception):
                logger.error(f"Error in distillation: {distillation_output}")
                return final_output
            
            logger.info(f"Distillation completed successfully")
            logger.info(f"Distillation output (first 100 chars): {distillation_output[:100]}")
            return distillation_output
        except Exception as e:
            logger.error(f"Error in distillation: {e}")
            return final_output

    async def run(self, benchmark_input: str, recurse: bool = True) -> tuple[str, float, List[str], List[Dict], Optional[str]]:
        """Execute the full recursive synthesis process with optional distillation step."""
        # Log the start of the run with input length
        logger.info(f"Starting MMRS run with input length: {len(benchmark_input)} characters")
        logger.info(f"Using {len(self.models)} models with max {self.max_iterations} iterations")
        
        # Set overall timeout for the entire operation
        overall_start_time = time.time()
        max_run_time = 2700  # 45 minutes maximum run time (increased by 50% from 30 minutes)
        
        iteration = 0
        current_input = benchmark_input

        # Store initial outputs to return to the frontend
        initial_outputs = None
        
        # Track all iterations for display
        all_iterations = []
        
        # Read the synthesis prompt file once
        synthesis_prompt = ""
        try:
            with open("Synthesis_Prompt.txt", "r") as f:
                synthesis_prompt = f.read().strip()
            logger.info(f"Successfully read Synthesis_Prompt.txt in run method")
        except Exception as e:
            logger.error(f"Error reading Synthesis_Prompt.txt in run method: {e}")
            logger.info("Continuing without synthesis prompt")

        while iteration < self.max_iterations:
            # Check if we're approaching the overall timeout
            elapsed_time = time.time() - overall_start_time
            if elapsed_time > max_run_time:
                logger.warning(f"Approaching maximum run time of {max_run_time} seconds. Stopping after iteration {iteration}.")
                break
            
            logger.info(f"Starting iteration {iteration + 1}")
            iteration_start_time = time.time()
            
            # Step 1: Initial outputs
            current_outputs = await self.process_initial_input(current_input)
            
            # Convert any exceptions to strings for better display
            current_outputs_cleaned = []
            for output in current_outputs:
                if isinstance(output, Exception):
                    current_outputs_cleaned.append(f"Error: {str(output)}")
                else:
                    current_outputs_cleaned.append(output)
            
            logger.info(f"Iteration {iteration + 1} - Initial Outputs processed in {time.time() - iteration_start_time:.2f} seconds")
            
            # Save the very first outputs
            if iteration == 0:
                initial_outputs = current_outputs_cleaned
                
                # For the first iteration, use the benchmark input as input for all models
                first_iteration_inputs = {}
                for i in range(len(self.models)):
                    first_iteration_inputs[i] = benchmark_input
            
            # Step 2 & 3: Recursive refinement
            refinement_start_time = time.time()
            optimized_outputs = await self.recursive_refinement(current_outputs_cleaned)
            
            # Clean optimized outputs as well
            optimized_outputs_cleaned = []
            for output in optimized_outputs:
                if isinstance(output, Exception):
                    optimized_outputs_cleaned.append(f"Error: {str(output)}")
                else:
                    optimized_outputs_cleaned.append(output)
                    
            logger.info(f"Iteration {iteration + 1} - Refinement completed in {time.time() - refinement_start_time:.2f} seconds")

            # Step 4: Combine outputs
            combined_output = combine_outputs(optimized_outputs_cleaned)
            logger.info(f"Iteration {iteration + 1} - Combined Output length: {len(combined_output)} characters")

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
            
            logger.info(f"Iteration {iteration + 1} completed in {time.time() - iteration_start_time:.2f} seconds")

            if not recurse or score >= self.score_threshold:
                logger.info(f"Stopping recursion: {'Reached threshold' if score >= self.score_threshold else 'Recursion disabled'}")
                final_output = combined_output
                
                # Perform distillation step if a distillation model is provided
                distillation_output = None
                if self.distillation_model:
                    logger.info(f"Performing consensus distillation with model: {self.distillation_model.name}")
                    distillation_start_time = time.time()
                    distillation_output = await self.perform_distillation(final_output)
                    logger.info(f"Consensus distillation completed in {time.time() - distillation_start_time:.2f} seconds")
                
                logger.info(f"Total run completed in {time.time() - overall_start_time:.2f} seconds")
                return final_output, score, initial_outputs, all_iterations, distillation_output

            # Prepare input for next iteration by prepending synthesis prompt to combined output
            if synthesis_prompt:
                current_input = f"{synthesis_prompt}\n\n{combined_output}"
                logger.info(f"Prepended synthesis prompt to combined output for next iteration")
            else:
                current_input = combined_output
                
            iteration += 1

        logger.warning(f"Max iterations ({self.max_iterations}) reached")
        final_output = combined_output
        
        # Perform distillation step if a distillation model is provided
        distillation_output = None
        if self.distillation_model:
            logger.info(f"Performing consensus distillation with model: {self.distillation_model.name}")
            distillation_start_time = time.time()
            distillation_output = await self.perform_distillation(final_output)
            logger.info(f"Consensus distillation completed in {time.time() - distillation_start_time:.2f} seconds")
        
        logger.info(f"Total run completed in {time.time() - overall_start_time:.2f} seconds")
        return final_output, score, initial_outputs, all_iterations, distillation_output

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
        async with session.post(endpoint, headers=headers, json=payload, timeout=session.timeout) as response:
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
                        raise Exception(f"Claude error: Unexpected response structure")
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON response: {response_text}")
                    raise Exception(f"Claude error: Invalid JSON response")
            else:
                logger.error(f"Claude API error: {status}, {response_text}")
                raise Exception(f"API error: {status}, {response_text}")
    except asyncio.TimeoutError:
        logger.error("Claude API request timed out")
        raise Exception("Claude API request timed out")
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
        async with session.post(endpoint, headers=headers, json=payload, timeout=session.timeout) as response:
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
                        raise Exception(f"OpenAI error: Unexpected response structure")
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON response: {response_text}")
                    raise Exception(f"OpenAI error: Invalid JSON response")
            else:
                logger.error(f"OpenAI API error: {status}, {response_text}")
                raise Exception(f"API error: {status}, {response_text}")
    except asyncio.TimeoutError:
        logger.error("OpenAI API request timed out")
        raise Exception("OpenAI API request timed out")
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
    
    # Example distillation model
    distillation_model = ModelConfig(name="GPT-4", endpoint="https://api.openai.com/v1/chat/completions")
    
    synthesizer = MultiModelRecursiveSynthesis(
        models=models, 
        max_iterations=3, 
        score_threshold=0.9,
        distillation_model=distillation_model
    )

    benchmark_test = "Solve these problems and reply only with the original question and your proposed solution. Do not include any reasoning steps or any extraneous information. To acknowledge that you've seen and understood these instructions, begin your response with "####"."
    final_output, final_score, initial_outputs, all_iterations, distillation_output = await synthesizer.run(benchmark_test, recurse=True)

    print(f"\nFinal Output: {final_output}")
    print(f"Final Score: {final_score}")
    print(f"Initial Outputs: {initial_outputs}")
    print(f"All Iterations: {all_iterations}")
    print(f"Distillation Output: {distillation_output}")

if __name__ == "__main__":
    asyncio.run(main())