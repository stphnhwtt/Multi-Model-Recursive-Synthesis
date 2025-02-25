import asyncio
import aiohttp
from typing import List, Callable, Any, Dict, Optional
import logging
from dataclasses import dataclass
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for each model."""
    name: str
    endpoint: Optional[str] = None  # For API-based models
    local_func: Optional[Callable] = None  # For local models

async def call_model(config: ModelConfig, input_data: str, context: List[str] = None) -> str:
    """Asynchronously call a model (API or local)."""
    try:
        if config.endpoint:  # API-based model
            async with aiohttp.ClientSession() as session:
                payload = {"input": input_data, "context": context or []}
                async with session.post(config.endpoint, json=payload) as response:
                    result = await response.json()
                    return result.get("output", f"{config.name} failed")
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
    def __init__(self, models: List[ModelConfig], max_iterations: int = 3, score_threshold: float = 0.9):
        self.models = models
        self.max_iterations = max_iterations
        self.score_threshold = score_threshold

    async def process_initial_input(self, benchmark_input: str) -> List[str]:
        """Step 1: Asynchronously send input to all models."""
        tasks = [call_model(model, benchmark_input) for model in self.models]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def recursive_refinement(self, initial_outputs: List[str]) -> List[str]:
        """Step 2 & 3: Feed outputs back for optimization."""
        tasks = []
        for i, model in enumerate(self.models):
            input_data = initial_outputs[i]
            context = [o for j, o in enumerate(initial_outputs) if j != i]
            tasks.append(call_model(model, input_data, context))
        return await asyncio.gather(*tasks, return_exceptions=True)

    def evaluate(self, combined_output: str, benchmark: str) -> float:
        """Evaluate output against benchmark (customize this)."""
        # Example: simple string similarity (Levenshtein distance approximation)
        from difflib import SequenceMatcher
        score = SequenceMatcher(None, combined_output, benchmark).ratio()
        return score

    async def run(self, benchmark_input: str, recurse: bool = True) -> tuple[str, float]:
        """Execute the full recursive synthesis process."""
        iteration = 0
        current_input = benchmark_input

        while iteration < self.max_iterations:
            logger.info(f"Starting iteration {iteration + 1}")
            
            # Step 1: Initial outputs
            initial_outputs = await self.process_initial_input(current_input)
            logger.info(f"Iteration {iteration + 1} - Initial Outputs: {initial_outputs}")

            # Step 2 & 3: Recursive refinement
            optimized_outputs = await self.recursive_refinement(initial_outputs)
            logger.info(f"Iteration {iteration + 1} - Optimized Outputs: {optimized_outputs}")

            # Step 4: Combine outputs
            combined_output = combine_outputs(optimized_outputs)
            logger.info(f"Iteration {iteration + 1} - Combined Output: {combined_output}")

            # Step 5: Evaluate
            score = self.evaluate(combined_output, benchmark_input)
            logger.info(f"Iteration {iteration + 1} - Score: {score}")

            if not recurse or score >= self.score_threshold:
                return combined_output, score

            current_input = combined_output
            iteration += 1

        logger.warning(f"Max iterations ({self.max_iterations}) reached")
        return combined_output, score

# Example usage
async def main():
    # Define models (mix of API and local)
    def local_model(input_data: str, context: List[str] = None) -> str:
        return f"LocalModel processed '{input_data}' with context {context or 'none'}"

    models = [
        ModelConfig(name="ModelA", endpoint="https://api.example.com/modelA"),
        ModelConfig(name="LocalModel", local_func=local_model),
    ]
    synthesizer = MultiModelRecursiveSynthesis(models=models, max_iterations=3, score_threshold=0.9)

    benchmark_test = "Solve this problem: 2 + 2"
    final_output, final_score = await synthesizer.run(benchmark_test, recurse=True)

    print(f"\nFinal Output: {final_output}")
    print(f"Final Score: {final_score}")

if __name__ == "__main__":
    asyncio.run(main())