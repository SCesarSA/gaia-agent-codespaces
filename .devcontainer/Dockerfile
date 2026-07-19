FROM mcr.microsoft.com/devcontainers/python:1-3.11-bookworm

COPY requirements.txt /tmp/gaia-requirements.txt

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r /tmp/gaia-requirements.txt \
    && python -m pip check
