
SHELL := bash

# Cluster-facing targets forward to this CLI. Run `make <target>` for the
# normal flow, or `KIT_DRY_RUN=1 make <target>` to print the commands.
CLI ?= PYTHONPATH=. python3 openshell/cli.py

.PHONY: help env run-app test refresh-certs build-openshell create-sandbox \
	wait-sandbox setup-egress start-agent expose-agent rotate-tokens deploy-openshell undeploy-openshell

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-22s %s\n", $$1, $$2}'

env: ## Create venv and install dependencies
	@echo "==> Creating virtual environment and installing dependencies..." && \
	  rm -rf .venv && \
	  uv sync --python 3.12 && \
	  echo "" && \
	  echo "Done. Activate with: source .venv/bin/activate" && \
	  echo "Export MAAS_BASE_URL and MAAS_API_KEY, then run: make run-app"

run-app: ## Run agent locally with hot-reload
	@source .venv/bin/activate && set -a && if [ -f .env ]; then . ./.env; fi && set +a && \
	  if lsof -ti:$${PORT:-8000} >/dev/null 2>&1; then \
	    echo "ERROR: Port $${PORT:-8000} is already in use." && \
	    exit 1; \
	  fi && \
	  uv run uvicorn main:app --host 127.0.0.1 --port $${PORT:-8000} --reload --reload-exclude .venv

test: ## Run unit tests
	uv run --extra dev python -m pytest tests/ $(PYTEST_ARGS)

# ── openShell sandbox targets ────────────────────────────────────────────────
# Every target below maps 1:1 to a `openshell/cli.py` subcommand:
#   make build-openshell   -> python3 openshell/cli.py build
# Deployment inputs (app name, namespace, workspace, gateway alias, image,
# sandbox name, secret references) live in values.yaml; see README.md.

refresh-certs: ## Refresh openShell mTLS client certificates from the cluster
	@$(CLI) refresh-certs

build-openshell: ## Build openShell sandbox image via OpenShift BuildConfig
	@$(CLI) build

create-sandbox: ## Delete existing sandbox and create a new one (with 120s timeout)
	@$(CLI) create

wait-sandbox: ## Poll until sandbox phase is Ready (max 300s)
	@$(CLI) wait

setup-egress: ## Add MaaS + vector database egress policies for agent Python binary
	@$(CLI) egress

start-agent: ## Start agent with K8s ServiceAccount auth inside sandbox
	@$(CLI) start

expose-agent: ## Expose agent service and create Route
	@$(CLI) expose

rotate-tokens: ## Rotate ServiceAccount tokens and restart the agent
	@$(CLI) rotate-tokens

deploy-openshell: ## Full sandbox deploy: create, wait, egress, start, expose
	@$(CLI) deploy

undeploy-openshell: ## Remove openShell sandbox and route
	@$(CLI) undeploy
