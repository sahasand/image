# AI Image Generator

This project provides a simple web interface and REST API for generating images using Stable Diffusion models. The current implementation includes only **Stable Diffusion 1.5**. The repository contains a Next.js frontend and a Python backend implemented with FastAPI.

## Quick Start

### Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python run_server.py
```

The API will be available at `http://localhost:8000`.

### Frontend

```bash
npm install
npm run dev
```

Open `http://localhost:3000` to access the Next.js frontend.

## API Endpoints

- `GET /health` – Server status
- `POST /generate` – Generate image from prompt
- `GET /models` – List available models
- `POST /models/switch/{model_name}` – Switch model

For full documentation see `CLAUDE.md`.

