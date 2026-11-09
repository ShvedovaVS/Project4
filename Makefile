install:
	uv sync

lint:
	uv run ruff check src tests
	uv run black --check src tests

test:
	uv run pytest -q

run:
	uv run uvicorn src.api:app --reload

bot:
	uv run python -m src.bot

train:
	uv run python -m src.train