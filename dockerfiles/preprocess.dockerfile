FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libvulkan1 \
    mesa-vulkan-drivers \
    libegl1-mesa \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --no-install-project

COPY src/ src/
COPY scripts/ scripts/

ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["python", "scripts/preprocess_data.py"]
