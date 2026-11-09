FROM python:3.11-slim
WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set environment variables for uv
ENV UV_CACHE_DIR=/root/.cache/uv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock ./

# Install dependencies with caching
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Copy the rest of the application
COPY src ./src
COPY models ./models
# .env is not copied for security. Pass BOT_TOKEN via -e when running:
# docker run -e BOT_TOKEN="your_token_here" sentiment-bot

CMD ["uv", "run", "python", "-m", "src.bot"]