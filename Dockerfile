FROM python:3.13-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY . .

RUN python -m pip install --upgrade pip \
    && python -m pip install -e .[dev]

EXPOSE 8000 8501