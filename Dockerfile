# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Environment
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Zurich
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System + Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    build-essential git unzip wget curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Make `python` and `pip` point to Python 3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Project
WORKDIR /app
COPY . /app
RUN chown -R 1000:root /app

# Upgrade pip tooling (optional but recommended)
RUN pip install --upgrade pip setuptools wheel

# Install Python dependencies directly into the system environment
ARG PIP_EXTRAS=dev
RUN pip install ".[${PIP_EXTRAS}]"

# Default shell
CMD ["bash"]
