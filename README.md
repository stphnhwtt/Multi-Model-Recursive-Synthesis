# MMRS Backend Setup Instructions

## Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment (recommended)

## Setup Steps

### 1. Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install fastapi uvicorn python-dotenv aiohttp numpy
pip install openai anthropic huggingface_hub
```

### 2. Create Environment Variables
Create a `.env` file in your project root:
```env
# API Keys
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
HUGGINGFACE_API_KEY=your_huggingface_key
GROK_API_KEY=your_grok_key

# Server Settings
HOST=localhost
PORT=5000
```

### 3. Backend Implementation

Create the following file structure:
```
backend/
├── __init__.py
├── main.py
├── models/
│   ├── __init__.py
│   ├── base.py
│   ├── openai_model.py
│   ├── claude_model.py
│   ├── huggingface_model.py
│   ├── llama_model.py
│   └── grok_model.py
└── utils/
    ├── __init__.py
    └── synthesis.py
```

#### 3.1 Base Model Class (models/base.py)
```python
from abc import ABC, abstractmethod
from typing import Optional, List

class BaseModel(ABC):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    @abstractmethod
    async def generate(self, prompt: str, context: List[str] = None) -> str:
        pass

    @abstractmethod
    async def initialize(self):
        pass
```

#### 3.2 Synthesis Logic (utils/synthesis.py)
```python
from typing import List, Dict
import asyncio
from models.base import BaseModel

async def recursive_synthesis(
    models: List[BaseModel],
    prompt: str,
    iterations: int,
    max_concurrent_calls: int = 3
) -> str:
    current_outputs = []
    current_prompt = prompt

    for iteration in range(iterations):
        # Generate responses from all models
        tasks = [model.generate(current_prompt) for model in models]
        responses = await asyncio.gather(*tasks)
        
        # Combine responses for next iteration
        current_outputs = responses
        if iteration < iterations - 1:
            # Use combined output as next prompt
            current_prompt = " | ".join(responses)

    # Return final combined output
    return " | ".join(current_outputs)
```

#### 3.3 Main FastAPI Application (main.py)
```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import model implementations
from models.openai_model import OpenAIModel
from models.claude_model import ClaudeModel
from models.huggingface_model import HuggingFaceModel
from models.llama_model import LlamaModel
from models.grok_model import GrokModel
from utils.synthesis import recursive_synthesis

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SynthesisRequest(BaseModel):
    models: List[str]
    prompt: str
    iterations: int
    apiKeys: Dict[str, str]

@app.post("/synthesize")
async def synthesize(request: SynthesisRequest):
    try:
        # Initialize models based on request
        active_models = []
        for model_id in request.models:
            if model_id == "gpt-4":
                active_models.append(OpenAIModel(request.apiKeys.get("OPENAI_API_KEY")))
            elif model_id == "claude":
                active_models.append(ClaudeModel(request.apiKeys.get("ANTHROPIC_API_KEY")))
            elif model_id == "huggingface":
                active_models.append(HuggingFaceModel(request.apiKeys.get("HUGGINGFACE_API_KEY")))
            elif model_id == "llama":
                active_models.append(LlamaModel())
            elif model_id == "grok":
                active_models.append(GrokModel(request.apiKeys.get("GROK_API_KEY")))

        # Initialize all models
        await asyncio.gather(*[model.initialize() for model in active_models])

        # Perform synthesis
        result = await recursive_synthesis(
            models=active_models,
            prompt=request.prompt,
            iterations=request.iterations
        )

        return {"result": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "localhost"),
        port=int(os.getenv("PORT", 5000)),
        reload=True
    )
```

### 4. Model Implementations

You'll need to implement each model class (OpenAIModel, ClaudeModel, etc.) following the BaseModel interface. Here's an example for OpenAI:

```python
# models/openai_model.py
from .base import BaseModel
import openai
from typing import List, Optional

class OpenAIModel(BaseModel):
    async def initialize(self):
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        openai.api_key = self.api_key

    async def generate(self, prompt: str, context: List[str] = None) -> str:
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
```

### 5. Running the Backend

```bash
# From the project root
python backend/main.py
```

The server will start on http://localhost:5000

### 6. Testing

1. Make sure both frontend and backend are running
2. Configure API keys in the frontend interface
3. Select models and enter a prompt
4. The response will be processed through the recursive synthesis pipeline

## Notes

- Implement proper error handling and logging
- Add request validation
- Consider adding rate limiting
- Implement proper security measures for API key handling
- Add monitoring and metrics collection
- Consider adding response streaming for long-running requests

## Troubleshooting

1. CORS issues: Verify the CORS configuration in main.py matches your frontend URL
2. API key errors: Check that all required API keys are properly set in .env
3. Model errors: Check individual model implementations for proper error handling
4. Connection issues: Verify both frontend and backend ports are correct and available 