ARG PYTORCH="2.2.2"
ARG CUDA="12.1"
ARG CUDNN="8"
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel


ENV CUDA_HOME="/usr/local/cuda" \
    FORCE_CUDA="1" \
    TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6 8.7 8.9+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends \
    curl \
    ffmpeg \
    git \
    ninja-build \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip --no-cache-dir install \
    aenum \
    nptyping \
    rasterio \
    numpy==1.23.5 \
    nvidia-pyindex \
    openmim

RUN python -m pip install pip-tools

WORKDIR /workspace

RUN pip uninstall -y torch torchvision torchaudio && \
    pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --extra-index-url https://download.pytorch.org/whl/cu121
RUN pip uninstall -y transformers && \
    pip install git+https://github.com/huggingface/transformers accelerate>=0.26.0

RUN pip install qwen-vl-utils[decord]
RUN pip install \
    ultralytics==8.0.20 \
    scikit-learn \
    matplotlib \
    spacy==3.5.3 \
    openai \
    Pillow \
    python-dotenv \
    backoff
    # nltk \
    # regex

