# ==============================================================================
# 🏥 CLINICAL PIPELINE MANAGER
# ==============================================================================
PYTHON := python3
SCRIPTS_DIR := icare_risk.scripts

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
	@echo "  make docs-serve - Serve Zensical Docs Locally"
	@echo "  make docs-build - Build Zensical Docs"
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
	$(RUN) $(PYTHON) -m $(SCRIPTS_DIR).a_generate_data $(ARGS)

features:
	@echo "\n--- Step 2: Building Clinical Features ---"
	$(RUN) $(PYTHON) -m $(SCRIPTS_DIR).b_build_features_icare $(ARGS)

evaluate:
	@echo "\n--- Step 3: Evaluating Clinical Scores ---"
	$(RUN) $(PYTHON) -m $(SCRIPTS_DIR).c_evaluate_scores $(ARGS)

thresholds:
	@echo "\n--- Step 4: Evaluating Stewardship Thresholds ---"
	$(RUN) $(PYTHON) -m $(SCRIPTS_DIR).d_evaluate_thresholds $(ARGS)

validate:
	@echo "\n--- Step 5: Validating Scores (Clinical Audit) ---"
	$(RUN) $(PYTHON) -m $(SCRIPTS_DIR).e_validate_scores $(ARGS)

search:
	@echo "\n--- Discover Clinical Codes (Keywords) ---"
	$(RUN) $(PYTHON) -m $(SCRIPTS_DIR).f_find_clinical_codes $(ARGS)

test:
	@echo "\n--- Running Unit Tests ---"
	$(RUN) $(PYTHON) -m pytest tests/ $(ARGS)

clean:
	@echo "\n--- Cleaning up reports ---"
	rm -f reports/*.log reports/*.txt
	@echo "Done."

.PHONY: docs-serve docs-build

docs-serve:
	@echo "\n--- 📖 Serving Documentation Locally ---"
	$(RUN) zensical serve

docs-build:
	@echo "\n--- 📖 Building Documentation ---"
	$(RUN) zensical build --clean



.PHONY: publish

publish:
	@echo "Tagging and triggering PyPI release..."
	$(eval TAG := v$(shell date +'%Y.%m.%d.%H.%M'))
	git tag $(TAG)
	git push origin $(TAG)
	@echo "Release $(TAG) pushed!"


.PHONY: build-pkg test-pkg

build-pkg:
	@echo "\n--- Building PyPI Package (Wheel & Source) ---"
	$(RUN) python -m build

test-pkg: build-pkg
	@echo "\n--- Cleaning old distribution files & build caches ---"
	rm -rf dist/* build/ src/*.egg-info
	@echo "\n--- Building PyPI Package (Wheel & Source) ---"
	$(RUN) python -m build

	@echo "\n--- 1. Creating Isolated Venv ---"
	$(RUN) python -m venv /tmp/pkg_test_venv
	$(RUN) /tmp/pkg_test_venv/bin/pip install --upgrade pip --quiet

	@echo "\n--- 2. Installing Built Wheel & Pytest ---"
	$(RUN) /tmp/pkg_test_venv/bin/pip install dist/*.whl pytest

	@echo "\n--- 3. Running Import Smoke Test ---"
	$(RUN) bash -c "cd /tmp && /tmp/pkg_test_venv/bin/python -c 'import icare_risk; print(\"Smoke Test Passed!\")'"
	@echo "Modules imported successfully from wheel!"

	@echo "\n--- 4. Running CLI Console Script Smoke Tests ---"
	$(RUN) bash -c "/tmp/pkg_test_venv/bin/icare-risk-generate --help"
	$(RUN) bash -c "/tmp/pkg_test_venv/bin/icare-risk-features --help"
	@echo "CLI Entry Points verified successfully!"

	@echo "\n--- 5. Running Pytest Suite ---"
	$(RUN) bash -c "cd /tmp && /tmp/pkg_test_venv/bin/pytest /app/tests/ -v"

	@echo "\n--- 6. Cleaning Up ---"
	$(RUN) rm -rf /tmp/pkg_test_venv
	@echo "Package verification complete!"