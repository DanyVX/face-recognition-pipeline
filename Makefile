# Thin wrapper: all logic lives in scripts/dev.py so Windows (no make) works too:
#   uv run python scripts/dev.py <task>
# Requires uv (https://docs.astral.sh/uv/).
RUN := uv run --no-project python scripts/dev.py
TASKS := setup lint format typecheck test test-fast bench run docker-build clean

.PHONY: $(TASKS) help
help:
	@echo "Targets: $(TASKS)"

$(TASKS):
	@$(RUN) $@
