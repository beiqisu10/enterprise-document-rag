FROM python:3.13-slim

WORKDIR /app

# Enable Python unbuffered output and add .venv binaries to PATH
ENV PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# Install uv package manager
RUN pip install --upgrade pip && pip install uv

# Copy dependency manifests first (leverages Docker cache)
COPY pyproject.toml uv.lock README.md ./

# Install dependencies into .venv, skipping the project itself for now
RUN uv sync --frozen --no-install-project

# Copy the rest of the source code
COPY . .

# Install the project as a package (now that source is available)
RUN uv sync --frozen

EXPOSE 8000 8501

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]