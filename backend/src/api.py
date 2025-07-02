from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

from .image_generator import ImageGenerator

app = FastAPI(title="AI Image Generator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

generator = ImageGenerator()


class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    width: int = 512
    height: int = 512
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    batch_size: int = 1


class GenerationResponse(BaseModel):
    success: bool
    images: Optional[List[str]] = None
    image_urls: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "model": generator.current_model or "none"}


@app.post("/generate", response_model=GenerationResponse)
def generate(req: GenerationRequest) -> GenerationResponse:
    try:
        result = generator.generate_image(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            width=req.width,
            height=req.height,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            seed=req.seed,
            batch_size=req.batch_size,
        )
        return GenerationResponse(success=True, images=result["images"], image_urls=result["image_paths"])
    except Exception as e:  # pylint: disable=broad-except
        return GenerationResponse(success=False, error=str(e))


@app.get("/models")
def list_models() -> List[str]:
    from ..config import config

    return list(config.MODELS.keys())


@app.get("/models/current")
def current_model() -> Dict[str, Optional[str]]:
    return {"current_model": generator.current_model}


@app.post("/models/switch/{model_name}")
def switch_model(model_name: str) -> Dict[str, str]:
    generator.load_model(model_name)
    return {"status": "loaded", "model": model_name}
