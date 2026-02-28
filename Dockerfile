# RabbitLLM — layer-streaming inference for 70B+ LLMs on consumer GPUs
# Build: docker build -t rabbitllm .
# Run (GPU): docker run --gpus all -it rabbitllm python scripts/inference_example.py --model Qwen/Qwen2.5-0.5B-Instruct
# Run (help): docker run --rm rabbitllm

FROM python:3.12-slim

WORKDIR /app

# Install RabbitLLM from the build context with optional GDS (GPU Direct Storage) support.
# For Flash Attention, use a image with CUDA and install rabbitllm[flash] separately.
COPY pyproject.toml README.md ./
COPY src/ src/
COPY scripts/ scripts/
COPY example.py ./

RUN pip install --no-cache-dir -e ".[gds]"

# Default: show inference script help (override with full command)
CMD ["python", "scripts/inference_example.py", "--help"]
