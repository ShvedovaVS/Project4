install:
	uv sync

lint:
	uv run ruff check src tests
	uv run black --check src tests

test:
	uv run pytest -q

bot:
	uv run python -m src.bot

train:
	uv run python -m src.train