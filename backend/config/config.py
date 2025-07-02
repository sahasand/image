from dataclasses import dataclass
from typing import Dict

@dataclass
class ModelConfig:
    name: str
    model_id: str
    revision: str = "main"
    torch_dtype: str = "float16"

# Available models (can be extended)
MODELS: Dict[str, ModelConfig] = {
    "sd-1.5": ModelConfig(
        name="Stable Diffusion 1.5",
        model_id="runwayml/stable-diffusion-v1-5",
    ),
}

# Hardware settings
DEVICE = "auto"  # "cuda", "mps", "cpu", or "auto"
USE_HALF_PRECISION = True
ENABLE_MEMORY_EFFICIENT_ATTENTION = True

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000
