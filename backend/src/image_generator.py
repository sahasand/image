from __future__ import annotations

import base64
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
from diffusers import StableDiffusionPipeline
import torch

from ..config import config


class ImageGenerator:
    """Load and run Stable Diffusion pipelines."""

    def __init__(self) -> None:
        self.pipeline: Optional[StableDiffusionPipeline] = None
        self.current_model: Optional[str] = None

    def load_model(self, model_name: str = "sd-1.5") -> None:
        cfg = config.MODELS.get(model_name)
        if cfg is None:
            raise ValueError(f"Unknown model {model_name}")
        torch_dtype = torch.float16 if cfg.torch_dtype == "float16" else torch.float32
        hf_token = os.environ.get("HF_TOKEN")
        kwargs = {
            "revision": cfg.revision,
            "torch_dtype": torch_dtype,
        }
        if hf_token:
            kwargs["token"] = hf_token
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            cfg.model_id,
            **kwargs,
        )
        device = config.DEVICE if config.DEVICE != "auto" else (
            "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        self.pipeline.to(device)
        self.current_model = model_name

    def generate_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        batch_size: int = 1,
    ) -> Dict[str, Any]:
        if self.pipeline is None:
            self.load_model()
        generator = torch.Generator(device=self.pipeline.device)
        if seed is not None:
            generator = generator.manual_seed(seed)
        output = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=batch_size,
            generator=generator,
        )
        images: List[Image.Image] = output.images
        image_paths = []
        base64_images = []
        output_dir = Path("backend/outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        for idx, img in enumerate(images):
            filename = output_dir / f"generated_{idx}.png"
            img.save(filename)
            image_paths.append(str(filename))
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            base64_images.append(base64.b64encode(buffered.getvalue()).decode())
        return {"images": base64_images, "image_paths": image_paths}

