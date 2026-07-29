# ==============================================================================
# 🏥 CLINICAL PIPELINE MANAGER
# ==============================================================================
PYTHON := python3
SCRIPTS_DIR := scripts

# Default command prefix (runs via Docker)
RUN := docker-compose exec pipeline

# If the user appends "local", remove the Docker prefix to run natively
ifeq ($(filter local,$(MAKECMDGOALS)),local)
	RUN :=
endif

# Dummy target to prevent Make from complaining about "make local"
local:
	@:

.PHONY: help all generate features evaluate thresholds validate search test clean local

help:
	@echo "============================================================"
	@echo "             CLINICAL PIPELINE MANAGER                      "
	@echo "============================================================"
	@echo "COMMANDS (Defaults to Docker execution):"
	@echo "  make all        - Run entire core pipeline"
	@echo "  make generate   - Step 1: Generate Synthetic iCARE Data"
	@echo "  make features   - Step 2: Build Phenotypes (SIRS/Pitt)"
	@echo "  make evaluate   - Step 3: Compute Clinical Scores"
	@echo "  make thresholds - Step 4: Evaluate Stewardship Thresholds"
	@echo "  make validate   - Step 5: Run Clinical Audit (cases.csv)"
	@echo "  make search     - Discover Clinical Codes (Keywords)"
	@echo "  make test       - Run Pytest Suite"
	@echo "  make clean      - Remove old reports"
	@echo ""
	@echo "OPTIONS:"
	@echo "  local           - Run natively on your machine instead of Docker"
	@echo "  ARGS=\"...\"      - Pass custom flags to Python (e.g., config files)"
	@echo ""
	@echo "EXAMPLES:"
	@echo "  make generate local"
	@echo "  make evaluate ARGS=\"--eval-config custom_eval.yaml\""
	@echo "============================================================"

all: generate features evaluate thresholds validate

generate:
	@echo "\n--- Step 1: Generating Synthetic iCARE Data ---"
	$(RUN) $(PYTHON) -m $(SCRIPTS_DIR).01_generate_data $(ARGS)

features:
	@echo "\n--- Step 2: Building Clinical Features ---"
	$(RUN) $(PYTHON) -m $(SCRIPTS_DIR).02_build_features_icare $(ARGS)

evaluate:
	@echo "\n--- Step 3: Evaluating Clinical Scores ---"
	$(RUN) $(PYTHON) -m $(SCRIPTS_DIR).03_evaluate_scores $(ARGS)

thresholds:
	@echo "\n--- Step 4: Evaluating Stewardship Thresholds ---"
	$(RUN) $(PYTHON) -m $(SCRIPTS_DIR).04_evaluate_thresholds $(ARGS)

validate:
	@echo "\n--- Step 5: Validating Scores (Clinical Audit) ---"
	$(RUN) $(PYTHON) -m $(SCRIPTS_DIR).05_validate_scores $(ARGS)

search:
	@echo "\n--- Discover Clinical Codes (Keywords) ---"
	$(RUN) $(PYTHON) -m $(SCRIPTS_DIR).06_find_clinical_codes $(ARGS)

test:
	@echo "\n--- Running Unit Tests ---"
	$(RUN) $(PYTHON) -m pytest tests/ $(ARGS)

clean:
	@echo "\n--- 🧹 Cleaning up reports ---"
	rm -f reports/*.log reports/*.txt
	@echo "Done."


.PHONY: publish

publish:
	@echo "Tagging and triggering PyPI release..."
	$(eval TAG := v$(shell date +'%Y.%m.%d.%H'))
	git tag $(TAG)
	git push origin $(TAG)
	@echo "Release $(TAG) pushed!"


.PHONY: build-pkg test-pkg

build-pkg:
	@echo "\n--- 📦 Building PyPI Package (Wheel & Source) ---"
	$(RUN) python -m build

test-pkg: build-pkg
	@echo "\n--- 1. Creating Isolated Venv ---"
	$(RUN) python -m venv /tmp/pkg_test_venv
	$(RUN) /tmp/pkg_test_venv/bin/pip install --upgrade pip --quiet

	@echo "\n--- 2. Installing Built Wheel ---"
	$(RUN) /tmp/pkg_test_venv/bin/pip install dist/*.whl

	@echo "\n--- 3. Running Import Smoke Test ---"
	$(RUN) bash -c "cd /tmp && /tmp/pkg_test_venv/bin/python -c 'import src; import scripts'"
	@echo "✅ Modules imported successfully from wheel!"

	@echo "\n--- 4. Running Pytest Suite ---"
	$(RUN) bash -c "cd /tmp && /tmp/pkg_test_venv/bin/pytest /app/tests/"

	@echo "\n--- 5. Cleaning Up ---"
	$(RUN) rm -rf /tmp/pkg_test_venv
	@echo "✅ Package verification complete!"