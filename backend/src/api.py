from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

from .image_generator import ImageGenerator
from ..config import config

app = FastAPI(title="AI Image Generator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

generator = ImageGenerator()

# Authentication dependency
security = HTTPBearer(auto_error=False)


def require_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> None:
    token = config.API_TOKEN
    if token is None:
        return
    if credentials is None or credentials.credentials != token:
        raise HTTPException(status_code=401, detail="Invalid or missing token")


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
def generate(req: GenerationRequest, _: None = Depends(require_token)) -> GenerationResponse:
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
def list_models(_: None = Depends(require_token)) -> List[str]:
    return list(config.MODELS.keys())


@app.get("/models/current")
def current_model(_: None = Depends(require_token)) -> Dict[str, Optional[str]]:
    return {"current_model": generator.current_model}


@app.post("/models/switch/{model_name}")
def switch_model(model_name: str, _: None = Depends(require_token)) -> Dict[str, str]:
    generator.load_model(model_name)
    return {"status": "loaded", "model": model_name}
