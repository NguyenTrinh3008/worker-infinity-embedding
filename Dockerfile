FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base

ENV HF_HOME=/runpod-volume

# install python and other packages
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    wget \
    libgl1 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# install uv
RUN pip install uv

# install python dependencies
COPY requirements.txt /requirements.txt
RUN uv pip install -r /requirements.txt --system

# install torch (using stable version instead of test to avoid timeout)
RUN pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --no-cache-dir

# Add src files (includes updated handler.py with model-specific instruction support)
ADD src .

# Add test input (supports both BAAI/bge-small-en-v1.5 and BAAI/bge-code-v1)
COPY test_input.json /test_input.json

# start the handler
CMD python -u /handler.py
