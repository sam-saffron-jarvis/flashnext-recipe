# Checks that need no GPU, model or network: unit tests, syntax, JSON, shellcheck.
PYTHON ?= python3
export PYTHONDONTWRITEBYTECODE := 1

SH_SCRIPTS   := flashnext/launch.sh runit/flashnext/run runit/ollama-proxy/run
BASH_SCRIPTS := strata/install.sh strata/launch.sh strata/env.sh strata/prepare-cpu.sh

.PHONY: check
check:
	$(PYTHON) -m unittest flashnext.test_server flashnext.test_ollama_proxy
	cd bench && $(PYTHON) -m unittest test_benchmark
	sh -n $(SH_SCRIPTS)
	bash -n $(BASH_SCRIPTS)
	$(PYTHON) -c 'import ast, sys; [ast.parse(open(f).read(), f) for f in sys.argv[1:]]' \
		strata/download_models.py flashnext/smoke.py bench/bench_driver.py bench/decode_by_content.py bench/legacy-llamacpp/bench.py
	$(PYTHON) -c 'import json, sys; [json.load(open(f)) for f in sys.argv[1:]]' \
		strata/*.json.tmpl provenance/*.json
	@if command -v shellcheck >/dev/null; then \
		shellcheck -s sh $(SH_SCRIPTS) && shellcheck -x -s bash $(BASH_SCRIPTS); \
	else echo "shellcheck not installed; skipped"; fi
