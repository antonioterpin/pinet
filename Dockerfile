# syntax=docker/dockerfile:1.7

# Set the base image at build time
ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Environment
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Zurich
ENV CONDA_DIR=/opt/conda
ENV CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git python3-pip unzip wget ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Miniconda
RUN wget -qO /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && bash /tmp/miniconda.sh -b -p $CONDA_DIR \
 && rm /tmp/miniconda.sh
ENV PATH="$CONDA_DIR/bin:$PATH"

# Conda env
RUN conda config --remove channels defaults || true \
 && conda config --add channels conda-forge \
 && conda config --set channel_priority strict \
 && conda update -y conda \
 && conda create -y -n pinet python=3.12 pip \
 && conda clean -afy

# Ensure the environment is on PATH
ENV CONDA_DEFAULT_ENV=pinet
ENV PATH="$CONDA_DIR/envs/$CONDA_DEFAULT_ENV/bin:$PATH"

# Project
WORKDIR /app
COPY . /app
RUN chown -R 1000:root /app

# Install Python dependencies
ARG PIP_EXTRAS=dev
RUN pip install ".[${PIP_EXTRAS}]"

# Prepare entrypoint
ENTRYPOINT ["conda", "run", "-n", "pinet", "--no-capture-output"]
CMD ["bash"]
