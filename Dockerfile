# syntax=docker/dockerfile:1.7
FROM ubuntu:24.04

# Noninteractive apt; cleaner pip; unbuffered py
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# OS + Python toolchain
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    build-essential ca-certificates curl git \
 && rm -rf /var/lib/apt/lists/*

# Allow global pip installs in container (remove PEP 668 guard)
RUN set -eux; rm -f /usr/lib/python3*/EXTERNALLY-MANAGED || true

# (Optional) Upgrade pip/wheel/setuptools for better wheels support
#RUN pip3 install --upgrade pip wheel setuptools

# ----- Copy SDK first (better build caching) -----
WORKDIR /opt/taruagent-src
COPY src/ .

# Install the SDK into site-packages
RUN pip3 install -e .

# ----- App & testing trees (mirrored structure) -----
# Place your repo trees under /apps and /testing inside the image
WORKDIR /
COPY apps/ /apps/

# Make the CLI script executable and create a symlink for easy access
RUN chmod +x /apps/TaruAIcli.py
RUN ln -s /apps/TaruAIcli.py /usr/local/bin/tarucli

# Default working directory inside container
WORKDIR /apps

# Default shell
CMD ["/bin/bash"]

