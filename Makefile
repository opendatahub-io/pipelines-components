MARKDOWNLINT ?= markdownlint
UVRUN ?= uv run
RUFF ?= $(UVRUN) ruff
YAMLLINT ?= $(UVRUN) yamllint
PYTEST ?= $(UVRUN) pytest

.PHONY: format fix lint lint-format lint-python lint-markdown lint-yaml lint-imports test test-coverage component pipeline tests readme sync-packages

format:
	$(RUFF) format components pipelines scripts
	$(RUFF) check --fix components pipelines scripts

fix: format
	@echo "Auto-fixing Python formatting and lint issues..."
	@echo "Note: Markdown and YAML issues may need manual fixes"

lint: lint-format lint-python lint-markdown lint-yaml lint-imports

lint-format:
	$(RUFF) format --check components pipelines scripts

lint-python:
	$(RUFF) check components pipelines scripts

lint-markdown:
	$(MARKDOWNLINT) -c .markdownlint.json .

lint-yaml:
	$(YAMLLINT) -c .yamllint.yml .

lint-imports:
	$(UVRUN) .github/scripts/check_imports/check_imports.py --config .github/scripts/check_imports/import_exceptions.yaml components pipelines

test:
	cd .github/scripts && $(PYTEST) */tests/ -v $(ARGS)

test-coverage:
	cd .github/scripts && $(PYTEST) */tests/ --cov=. --cov-report=term-missing -v $(ARGS)

component:
	@if [ -z "$(CATEGORY)" ]; then echo "Error: CATEGORY is required. Usage: make component CATEGORY=data_processing NAME=my_component [SUBCATEGORY=x] [NO_TESTS=true] [CREATE_SHARED=true]"; exit 1; fi
	@if [ -z "$(NAME)" ]; then echo "Error: NAME is required. Usage: make component CATEGORY=data_processing NAME=my_component [SUBCATEGORY=x] [NO_TESTS=true] [CREATE_SHARED=true]"; exit 1; fi
	@set -e; \
	SUBCATEGORY_ARG=""; \
	if [ -n "$(SUBCATEGORY)" ]; then SUBCATEGORY_ARG="--subcategory=$(SUBCATEGORY)"; fi; \
	NO_TESTS_ARG=""; \
	if [ "$(NO_TESTS)" = "true" ]; then NO_TESTS_ARG="--no-tests"; fi; \
	CREATE_SHARED_ARG=""; \
	if [ "$(CREATE_SHARED)" = "true" ]; then CREATE_SHARED_ARG="--create-shared"; fi; \
	$(UVRUN) scripts/generate_skeleton/generate_skeleton.py --type=component --category="$(CATEGORY)" --name="$(NAME)" $$SUBCATEGORY_ARG $$NO_TESTS_ARG $$CREATE_SHARED_ARG; \
	echo ""; \
	echo "Generating READMEs..."; \
	if [ -n "$(SUBCATEGORY)" ]; then \
		$(UVRUN) -m scripts.generate_readme --component "components/$(CATEGORY)/$(SUBCATEGORY)/$(NAME)" --fix; \
	else \
		$(UVRUN) -m scripts.generate_readme --component "components/$(CATEGORY)/$(NAME)" --fix; \
	fi
	@$(MAKE) --no-print-directory sync-packages

pipeline:
	@if [ -z "$(CATEGORY)" ]; then echo "Error: CATEGORY is required. Usage: make pipeline CATEGORY=training NAME=my_pipeline [SUBCATEGORY=x] [NO_TESTS=true] [CREATE_SHARED=true]"; exit 1; fi
	@if [ -z "$(NAME)" ]; then echo "Error: NAME is required. Usage: make pipeline CATEGORY=training NAME=my_pipeline [SUBCATEGORY=x] [NO_TESTS=true] [CREATE_SHARED=true]"; exit 1; fi
	@set -e; \
	SUBCATEGORY_ARG=""; \
	if [ -n "$(SUBCATEGORY)" ]; then SUBCATEGORY_ARG="--subcategory=$(SUBCATEGORY)"; fi; \
	NO_TESTS_ARG=""; \
	if [ "$(NO_TESTS)" = "true" ]; then NO_TESTS_ARG="--no-tests"; fi; \
	CREATE_SHARED_ARG=""; \
	if [ "$(CREATE_SHARED)" = "true" ]; then CREATE_SHARED_ARG="--create-shared"; fi; \
	$(UVRUN) scripts/generate_skeleton/generate_skeleton.py --type=pipeline --category="$(CATEGORY)" --name="$(NAME)" $$SUBCATEGORY_ARG $$NO_TESTS_ARG $$CREATE_SHARED_ARG; \
	echo ""; \
	echo "Generating READMEs..."; \
	if [ -n "$(SUBCATEGORY)" ]; then \
		$(UVRUN) -m scripts.generate_readme --pipeline "pipelines/$(CATEGORY)/$(SUBCATEGORY)/$(NAME)" --fix; \
	else \
		$(UVRUN) -m scripts.generate_readme --pipeline "pipelines/$(CATEGORY)/$(NAME)" --fix; \
	fi
	@$(MAKE) --no-print-directory sync-packages

tests:
	@if [ -z "$(TYPE)" ]; then echo "Error: TYPE is required. Usage: make tests TYPE=component|pipeline CATEGORY=data_processing NAME=my_component [SUBCATEGORY=x]"; exit 1; fi
	@if [ -z "$(CATEGORY)" ]; then echo "Error: CATEGORY is required. Usage: make tests TYPE=component|pipeline CATEGORY=data_processing NAME=my_component [SUBCATEGORY=x]"; exit 1; fi
	@if [ -z "$(NAME)" ]; then echo "Error: NAME is required. Usage: make tests TYPE=component|pipeline CATEGORY=data_processing NAME=my_component [SUBCATEGORY=x]"; exit 1; fi
	@if [ "$(TYPE)" = "component" ] || [ "$(TYPE)" = "pipeline" ]; then \
		SUBCATEGORY_ARG=""; \
		if [ -n "$(SUBCATEGORY)" ]; then SUBCATEGORY_ARG="--subcategory=$(SUBCATEGORY)"; fi; \
		$(UVRUN) scripts/generate_skeleton/generate_skeleton.py --type="$(TYPE)" --category="$(CATEGORY)" --name="$(NAME)" $$SUBCATEGORY_ARG --tests-only; \
	else \
		echo "Error: TYPE must be either 'component' or 'pipeline'"; exit 1; \
	fi

readme:
	@if [ -z "$(TYPE)" ]; then echo "Error: TYPE is required. Usage: make readme TYPE=component|pipeline CATEGORY=data_processing NAME=my_component [SUBCATEGORY=x]"; exit 1; fi
	@if [ -z "$(CATEGORY)" ]; then echo "Error: CATEGORY is required. Usage: make readme TYPE=component|pipeline CATEGORY=data_processing NAME=my_component [SUBCATEGORY=x]"; exit 1; fi
	@if [ -z "$(NAME)" ]; then echo "Error: NAME is required. Usage: make readme TYPE=component|pipeline CATEGORY=data_processing NAME=my_component [SUBCATEGORY=x]"; exit 1; fi
	@if [ "$(TYPE)" = "component" ] || [ "$(TYPE)" = "pipeline" ]; then \
		if [ -n "$(SUBCATEGORY)" ]; then \
			$(UVRUN) -m scripts.generate_readme "--$(TYPE)" "$(TYPE)s/$(CATEGORY)/$(SUBCATEGORY)/$(NAME)" --fix; \
		else \
			$(UVRUN) -m scripts.generate_readme "--$(TYPE)" "$(TYPE)s/$(CATEGORY)/$(NAME)" --fix; \
		fi; \
	else \
		echo "Error: TYPE must be either 'component' or 'pipeline'"; exit 1; \
	fi

sync-packages:
	@$(UVRUN) python -m scripts.sync_packages.sync_packages

AIPCC_INDEX_URL := https://console.redhat.com/api/pypi/public-rhai/rhoai/3.4/cpu-ubi9/simple

requirements:
	echo "--index-url $(AIPCC_INDEX_URL)" > requirements.txt
	echo "" >> requirements.txt
	uv pip compile pyproject.toml --generate-hashes --no-header --no-annotate \
		--no-emit-package kfp-components \
		--python-version 3.12 \
		--index-url $(AIPCC_INDEX_URL) >> requirements.txt
	echo "--index-url $(AIPCC_INDEX_URL)" > requirements-build.txt
	echo "" >> requirements-build.txt
	printf 'setuptools\nwheel\n' | uv pip compile --generate-hashes --no-header --no-annotate \
		--python-version 3.12 \
		--index-url $(AIPCC_INDEX_URL) - >> requirements-build.txt
