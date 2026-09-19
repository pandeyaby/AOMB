# AOMB stranger-verify Codespace — Linux CPU only (no MPS, no GPU spend, no API keys).
FROM mcr.microsoft.com/devcontainers/python:3.12-bookworm

# Ranking-card baselines need pyarrow/numpy; diptych gate is stdlib + fixtures.
# CPU torch is optional (STRANGER_FAST=0). Keep image lean for the cite path.
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir pyarrow numpy rustbpe tiktoken

WORKDIR /workspaces/AOMB
