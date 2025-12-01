FROM ubuntu:22.04

# Non-interactive installs
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    git \
    curl \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libjpeg-dev \
    zlib1g-dev \
    libxml2-dev \
    libxslt1-dev \
    pkg-config \
    ffmpeg \
    # LibreOffice (headless) for converting Office files (.pptx/.docx/.odp) to PDF/images
    libreoffice-core \
    libreoffice-writer \
    libreoffice-impress \
    libreoffice-common \
    libreoffice-java-common \
    default-jre-headless \
    # Fonts commonly used in slides/docs
    fonts-dejavu-core \
    fonts-liberation \
    fonts-noto-core \
    fonts-noto-cjk \
    # Extra runtime deps for some wheels (OpenMP, poppler data, unzip utility)
    libgomp1 \
    poppler-data \
    unzip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

WORKDIR /workspace

# Copy only requirements at build time to leverage caching
COPY requirements.txt /workspace/requirements.txt

# Install Python requirements (paddlepaddle/paddleocr, faiss-cpu, etc.)
RUN python3 -m pip install --no-cache-dir -r /workspace/requirements.txt

# Install CPU PyTorch as a common dependency for sentence-transformers; if user prefers GPU, they should install appropriate CUDA wheels
RUN python3 -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu || true

# Create a non-root user for safer interactive use (optional)
RUN useradd -m appuser || true
USER appuser

ENV PATH="/home/appuser/.local/bin:${PATH}"

CMD ["/bin/bash"]
