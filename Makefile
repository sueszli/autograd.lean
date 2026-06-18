.PHONY: init
init:
	git config core.hooksPath .githooks

.PHONY: precommit
precommit:
	lake build

.PHONY: run
run:
	mkdir -p data
	curl -fsSL https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt -o data/input.txt
	uv run scripts/dump_parity.py
	lake exe autograd
