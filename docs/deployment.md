# Deployment Guide — VeritasCore

> Instructions for local development, Docker, GPU Docker, and cloud deployment.

---

## 1. Local Development

### Prerequisites

- Python 3.10–3.12
- [Ollama](https://ollama.ai) (for the LLM decomposer)
- Git

### Setup

```bash
# Clone
git clone https://github.com/Saqlain-mushtaq-alamin/VeritasCore.git
cd VeritasCore

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux/macOS

# Install with dev dependencies
pip install -e ".[dev]"

# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys (optional)

# Download models
python scripts/download_models.py

# Start Ollama (in a separate terminal)
ollama pull llama3.2
ollama serve

# Run the API server
uvicorn veritascore.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Environment Variables (`.env`)

```env
# LLM Backend (for decomposer)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2

# Optional: Bing Search (for ungrounded mode)
BING_API_KEY=your_bing_key_here

# Optional: OpenAI-compatible endpoint
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama

# Performance
DEVICE=cuda           # or "cpu"
NLI_BATCH_SIZE=8
NLI_FP16=true         # Enable fp16 for GPU (reduces VRAM, ~2x speedup)
```

### Makefile Shortcuts

```bash
make test             # Run unit tests with coverage
make lint             # Ruff + mypy checks
make benchmark        # Run HaluEval + FEVER benchmarks
make demo             # Launch Gradio demo
make docker-build     # Build Docker image
```

---

## 2. Docker (CPU)

### Build and Run

```bash
# Build image
docker build -f docker/Dockerfile -t veritascore:latest .

# Run container
docker run -p 8000:8000 \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  veritascore:latest
```

> **Note:** Ollama must be running on the host machine. The container connects to it via `host.docker.internal`.

### Docker Compose (with Ollama)

```yaml
# docker-compose.yml
version: "3.9"

services:
  veritascore:
    build:
      context: .
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
      - DEVICE=cpu
    depends_on:
      - ollama

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

volumes:
  ollama_data:
```

```bash
docker compose up --build
# Then pull the model inside the ollama container:
docker compose exec ollama ollama pull llama3.2
```

---

## 3. Docker (GPU — NVIDIA)

### Prerequisites

- NVIDIA GPU with ≥ 8 GB VRAM
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### GPU Dockerfile

Create `docker/Dockerfile.gpu`:

```dockerfile
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app
COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir -e "."

ENV DEVICE=cuda
ENV NLI_FP16=true

EXPOSE 8000
CMD ["uvicorn", "veritascore.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build GPU image
docker build -f docker/Dockerfile.gpu -t veritascore:gpu .

# Run with GPU access
docker run --gpus all -p 8000:8000 \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  veritascore:gpu
```

---

## 4. HuggingFace Spaces (Gradio Demo)

### Deploy the Demo

1. **Create a new HuggingFace Space** at [huggingface.co/new-space](https://huggingface.co/new-space)
   - SDK: **Gradio**
   - Hardware: **CPU Basic** (free) or **T4 GPU** for faster inference

2. **Push the demo files:**

```bash
# Clone your Space repo
git clone https://huggingface.co/spaces/<YOUR_HF_USERNAME>/veritascore-demo
cd veritascore-demo

# Copy demo files
cp /path/to/VeritasCore/demo/app.py .
cp /path/to/VeritasCore/demo/requirements.txt .

git add .
git commit -m "Add VeritasCore Gradio demo"
git push
```

3. **`requirements.txt`** for the Space:

```
veritascore>=1.0.0
gradio>=4.0.0
```

4. HuggingFace will automatically build and deploy the Space.

### Run Demo Locally

```bash
pip install gradio
python demo/app.py
# Opens at http://localhost:7860
```

---

## 5. Production Checklist

| Item | Command / Note |
|------|---------------|
| Set `DEVICE=cuda` | Required for < 200ms/claim latency |
| Enable `NLI_FP16=true` | Halves VRAM, ~2× speedup on GPU |
| Pre-load models on startup | Set `PRELOAD_MODELS=true` in `.env` |
| Configure reverse proxy (nginx) | Recommended for HTTPS + rate limiting |
| Set `API_KEY` | Enable bearer token authentication |
| Monitor with `/health` endpoint | Returns model status + uptime |
| Resource limits | RAM: 16 GB+; VRAM: 8 GB for full pipeline |
