RUFF ?= ruff
BASEDPYRIGHT ?= basedpyright

.PHONY: format format-check lint typecheck check

# default target
check: format-check lint typecheck

format:
	$(RUFF) format .

format-check:
	$(RUFF) format --check .

lint:
	$(RUFF) check .

typecheck:
	$(BASEDPYRIGHT) .

check: format-check lint typecheck
