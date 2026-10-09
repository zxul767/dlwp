UV ?= uv
RUFF ?= ruff
BASEDPYRIGHT ?= basedpyright
UV_RUN = $(UV) run --

.PHONY: format format-check lint typecheck check

# default target
check: format-check lint typecheck

format:
	$(UV_RUN) $(RUFF) format .

format-check:
	$(UV_RUN) $(RUFF) format --check .

lint:
	$(UV_RUN) $(RUFF) check .

typecheck:
	$(UV_RUN) $(BASEDPYRIGHT) .

check: format-check lint typecheck
