# AI Image Generator - Project Documentation

## 📋 Project Overview

This is a comprehensive, production-ready AI image generation application built using the latest open-source diffusion models, including **Stable Diffusion 3.5 Large** - the most advanced model available. The system provides both a web interface and REST API for generating high-quality images from text prompts.

### 🎯 Key Features
- **Latest AI Model**: **Stable Diffusion 3.5 Large** as default (state-of-the-art)
- **Multi-Model Support**: SD 3.5 Large, SD XL, SD 1.5, and more
- **Hardware Optimization**: Native Apple Silicon (MPS), CUDA, and CPU support
- **Web Interface**: Modern, responsive UI with real-time generation
- **REST API**: Complete API with OpenAPI documentation
- **Advanced Features**: ControlNet, image-to-image, inpainting
- **Performance Optimized**: PyTorch 2.0 optimizations and memory management
- **Authenticated Access**: Hugging Face integration for gated models

## 🏗️ Architecture

```
backend/
├── src/                          # Core application code
│   ├── image_generator.py        # Main generation engine
│   ├── advanced_features.py      # ControlNet, img2img, inpainting
│   ├── performance_optimizer.py  # PyTorch 2.0 optimizations
│   ├── api.py                    # FastAPI REST API
│   ├── web_interface.html        # Web UI
│   └── test_suite.py            # Comprehensive tests
├── config/
│   └── config.py                # Configuration management
├── models/cache/                # Downloaded model weights
├── outputs/                     # Generated images
├── requirements.txt             # Python dependencies
├── run_server.py               # Application launcher
├── Dockerfile                  # Container configuration
└── docker-compose.yml          # Orchestration
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+ (3.10+ recommended)
- **12GB+ RAM (16GB+ recommended for SD 3.5 Large)**
- GPU with 8GB+ VRAM or Apple Silicon (M1/M2/M3)
- **15GB+ free disk space** (SD 3.5 Large requires ~12GB)
- **Hugging Face account** (for SD 3.5 Large access)

### Quick Setup
```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Authenticate with Hugging Face (for SD 3.5 Large)
export HF_TOKEN=your_hugging_face_token_here

# Run the application
python run_server.py
```

### Docker Setup
```bash
# Build and run
docker-compose up -d

# For GPU support (NVIDIA only)
docker-compose --profile gpu up -d

# For development
docker-compose --profile dev up -d
```

## 🔧 Configuration

### Model Configuration
Edit `config/config.py` to modify:

```python
# Available models
MODELS = {
    "sd-1.5": ModelConfig(
        name="Stable Diffusion 1.5",
        model_id="runwayml/stable-diffusion-v1-5",
        # ... other settings
    )
}

# Hardware settings
DEVICE = "auto"  # "cuda", "mps", "cpu", or "auto"
USE_HALF_PRECISION = True
ENABLE_MEMORY_EFFICIENT_ATTENTION = True

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000
```

### Environment Variables
```bash
# Optional environment variables
export HF_HOME=/path/to/model/cache
export TRANSFORMERS_CACHE=/path/to/model/cache
export CUDA_VISIBLE_DEVICES=0
```

## 🎮 Usage

### Web Interface
1. Start the server: `python run_server.py`
2. Open browser to: `http://localhost:8000/web`
3. Enter a text prompt
4. Adjust parameters (size, steps, guidance)
5. Click "Generate Image"

### API Usage

#### Generate Image
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful sunset over mountains, digital art",
    "width": 512,
    "height": 512,
    "num_inference_steps": 20,
    "guidance_scale": 7.5,
    "seed": 42
  }'
```

#### Switch Models
```bash
curl -X POST "http://localhost:8000/models/switch/sd-xl"
```

#### List Models
```bash
curl "http://localhost:8000/models"
```

### Python API
```python
from src.image_generator import ImageGenerator

# Initialize generator
generator = ImageGenerator()
generator.load_model("sd-1.5")

# Generate image
result = generator.generate_image(
    prompt="A cute robot in a garden",
    width=512,
    height=512,
    num_inference_steps=20,
    seed=42
)

# Access generated images
images = result["images"]  # PIL Image objects
paths = result["image_paths"]  # Saved file paths
```

## 🔧 Troubleshooting

### Common Issues

#### Server Won't Start
**Problem**: `ModuleNotFoundError` or import errors
```bash
# Solution: Ensure virtual environment is activated
source venv/bin/activate
pip install -r requirements.txt
```

**Problem**: Port already in use
```bash
# Solution: Kill existing process or use different port
kill $(lsof -ti:8000)
# Or run on different port
uvicorn src.api:app --port 8001
```

#### Model Loading Issues
**Problem**: Models fail to download
```bash
# Solution: Check internet connection and disk space
df -h  # Check disk space
ping huggingface.co  # Check connectivity
```

**Problem**: Out of memory during model loading
```python
# Solution: Enable memory efficient loading in config.py
ENABLE_MEMORY_EFFICIENT_ATTENTION = True
USE_HALF_PRECISION = True
```

#### Generation Issues
**Problem**: Images generate too slowly
```python
# Solutions:
# 1. Reduce image size
width=512, height=512  # Instead of 1024x1024

# 2. Reduce inference steps
num_inference_steps=10  # Instead of 50

# 3. Use faster model
generator.switch_model("sd-1.5")  # Instead of "sd-xl"
```

**Problem**: Out of memory during generation
```python
# Solutions:
# 1. Reduce batch size
batch_size=1

# 2. Enable memory cleanup
from src.performance_optimizer import memory_manager
memory_manager.cleanup_memory(force=True)
```

#### Web Interface Issues
**Problem**: Web interface doesn't load
- Check server is running: `curl http://localhost:8000/health`
- Check browser console for errors
- Try different browser or incognito mode

**Problem**: CORS errors
- The API allows all origins by default
- For production, modify CORS settings in `src/api.py`

### Debug Mode
```bash
# Run with debug logging
PYTHONPATH=/path/to/backend python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from src.api import app
import uvicorn
uvicorn.run(app, host='0.0.0.0', port=8000, log_level='debug')
"
```

### Performance Monitoring
```python
# Enable performance profiling
from src.performance_optimizer import profiler
profiler.enable_profiling()

# ... run generations ...

# View metrics
profiler.print_metrics()
```

## 🧪 Testing

### Run Basic Tests
```bash
# Activate environment
source venv/bin/activate

# Test basic functionality
python src/test_generator.py

# Run comprehensive test suite
python src/test_suite.py

# Run specific tests with pytest
pytest src/test_suite.py::TestImageGenerator::test_basic_generation -v
```

### API Testing
```bash
# Test health endpoint
curl http://localhost:8000/health

# Test generation endpoint
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "test image", "width": 256, "height": 256, "num_inference_steps": 5}'
```

## 🚀 Deployment

### Local Production
```bash
# Run with production settings
uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 1
```

### Docker Production
```bash
# Build production image
docker build -t ai-image-generator .

# Run with resource limits
docker run -p 8000:8000 \
  --memory=8g \
  --cpus=4 \
  -v $(pwd)/models:/app/models \
  ai-image-generator
```

### Cloud Deployment
```yaml
# kubernetes deployment example
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-image-generator
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ai-image-generator
  template:
    metadata:
      labels:
        app: ai-image-generator
    spec:
      containers:
      - name: ai-image-generator
        image: ai-image-generator:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
```

## 🎨 Advanced Features

### ControlNet Usage
```python
from src.advanced_features import AdvancedImageGenerator

# Initialize advanced generator
advanced_gen = AdvancedImageGenerator()

# Generate with Canny edge control
result = advanced_gen.generate_with_controlnet(
    prompt="A beautiful landscape",
    control_image=input_image,  # PIL Image
    controlnet_type="canny",
    controlnet_conditioning_scale=1.0
)
```

### Image-to-Image
```python
# Transform existing image
result = advanced_gen.generate_img2img(
    prompt="Transform this into a painting",
    init_image=source_image,
    strength=0.8  # How much to change (0.0-1.0)
)
```

### Inpainting
```python
# Fill in masked areas
result = advanced_gen.generate_inpainting(
    prompt="A cat sitting in the garden",
    image=base_image,
    mask_image=mask_image  # White areas will be inpainted
)
```

## 📊 Performance Optimization

### Memory Management
```python
from src.performance_optimizer import memory_manager

# Monitor memory usage
memory_info = memory_manager.get_memory_info()
print(f"GPU memory: {memory_info['gpu']['percent']:.1f}%")

# Force cleanup
memory_manager.cleanup_memory(force=True)
```

### PyTorch 2.0 Optimizations
```python
from src.performance_optimizer import pytorch_optimizer

# Compile model for faster inference
compiled_pipeline = pytorch_optimizer.optimize_pipeline(
    pipeline, "my_pipeline"
)
```

### Batch Processing
```python
from src.performance_optimizer import batch_processor

# Start batch processor
batch_processor.start_batch_processor()

# Add requests to queue
request_id = "unique_id"
batch_processor.add_request(
    request_id, 
    generator.generate_image,
    prompt="batch image",
    width=512,
    height=512
)

# Get result
result = batch_processor.get_result(request_id)
```

## 🔒 Security Considerations

### Content Safety
- Built-in safety checker (can be disabled)
- NSFW content filtering
- Prompt filtering capabilities

### API Security
```python
# Add authentication (example)
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.post("/generate")
async def generate_image(
    request: GenerationRequest,
    token: str = Depends(security)
):
    # Validate token
    if not validate_token(token):
        raise HTTPException(401, "Invalid token")
    # ... generation logic
```

### Resource Limits
```python
# Limit concurrent generations
from asyncio import Semaphore
generation_semaphore = Semaphore(2)  # Max 2 concurrent

# Rate limiting
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)
app.add_route("/generate", generate_image, methods=["POST"])
limiter.limit("10/minute")(generate_image)
```

## 📈 Monitoring & Logging

### Application Logs
```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
```

### Metrics Collection
```python
from prometheus_client import Counter, Histogram, generate_latest

# Define metrics
generation_counter = Counter('images_generated_total', 'Total images generated')
generation_duration = Histogram('generation_duration_seconds', 'Time spent generating images')

# Use in code
generation_counter.inc()
with generation_duration.time():
    result = generator.generate_image(...)
```

## 🤝 Contributing

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8 jupyter

# Pre-commit hooks
pip install pre-commit
pre-commit install
```

### Code Style
```bash
# Format code
black src/

# Lint code
flake8 src/

# Run tests
pytest src/test_suite.py
```

### Adding New Models
1. Add model configuration to `config/config.py`
2. Test loading in `src/test_generator.py`
3. Update documentation
4. Submit pull request

## 📚 API Reference

### Core Endpoints

#### Health Check
- **GET** `/health`
- Returns server status and model information

#### Generate Image
- **POST** `/generate`
- Body: `GenerationRequest`
- Returns: `GenerationResponse` with base64 images

#### Model Management
- **GET** `/models` - List available models
- **GET** `/models/current` - Current model info
- **POST** `/models/switch/{model_name}` - Switch model

#### Image Management
- **GET** `/images` - List generated images
- **GET** `/images/{filename}` - Serve image file
- **DELETE** `/images/{filename}` - Delete image

### Request/Response Schemas

```python
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
    images: Optional[List[str]] = None  # Base64 encoded
    image_urls: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
```

## 🔗 Useful Resources

### Documentation
- [Hugging Face Diffusers](https://huggingface.co/docs/diffusers/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)

### Model Resources
- [Stable Diffusion Models](https://huggingface.co/models?pipeline_tag=text-to-image)
- [ControlNet Models](https://huggingface.co/lllyasviel)
- [LoRA Models](https://huggingface.co/models?other=lora)

### Community
- [Stable Diffusion Reddit](https://reddit.com/r/StableDiffusion)
- [Hugging Face Discord](https://discord.gg/hugging-face)
- [PyTorch Forums](https://discuss.pytorch.org/)

## 📄 License

This project uses open-source components:
- **Code**: MIT License
- **Stable Diffusion Models**: CreativeML Open RAIL-M License
- **Other Models**: Check individual model licenses

## 🆘 Support

For issues and questions:
1. Check this documentation
2. Review troubleshooting section
3. Check existing GitHub issues
4. Create new issue with detailed information

### Issue Template
```
**Environment:**
- OS: [macOS/Linux/Windows]
- Python: [version]
- PyTorch: [version]
- GPU: [model/none]

**Issue:**
[Detailed description]

**Steps to Reproduce:**
1. [Step 1]
2. [Step 2]

**Expected vs Actual:**
[What you expected vs what happened]

**Logs:**
[Relevant logs/error messages]
```