import json
import logging
from typing import Any, Optional

try:
    import ollama
except ImportError:
    ollama = None

LOGGER = logging.getLogger(__name__)

class LocalLLM:
    """Wrapper for local LLM inference using Ollama."""
    
    def __init__(self, model: str = "phi3"):
        self.model = model
        self.available = ollama is not None

    def generate(self, prompt: str, system_prompt: Optional[str] = None, response_format: Optional[str] = None) -> str:
        """Generate a response from the local LLM."""
        if not self.available:
            return "Error: 'ollama' library not installed. Run 'pip install ollama'."
            
        try:
            messages = []
            if system_prompt:
                messages.append({'role': 'system', 'content': system_prompt})
            messages.append({'role': 'user', 'content': prompt})
            
            kwargs = {}
            if response_format:
                kwargs['format'] = response_format
                
            response = ollama.chat(model=self.model, messages=messages, **kwargs) # type: ignore
            return response['message']['content']
        except Exception as e:
            LOGGER.error(f"LLM Error: {e}")
            return f"Error connecting to Ollama: {e}. Ensure 'ollama serve' is running."

    def generate_json(self, prompt: str, system_prompt: Optional[str] = None) -> Any:
        """Generate a response and attempt to parse it as JSON."""
        raw = self.generate(prompt, system_prompt, response_format="json")
        # Basic cleanup if model wraps JSON in backticks
        cleaned = raw.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0].strip()
            
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            LOGGER.warning(f"Failed to parse LLM output as JSON: {raw}")
            return {"error": "Failed to parse JSON", "raw_output": raw}
