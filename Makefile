lint:
	black .
	ruff check --fix .
	mypy .
