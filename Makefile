.PHONY: init
init:
	git config core.hooksPath .githooks

.PHONY: precommit
precommit:
	lake build

.PHONY: data
data:
	mkdir -p data
	curl -fsSL https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt -o data/input.txt

.PHONY: run
run:
	lake exe autograd

# forward-parity check: same init weights + same tokens through Python (original.py
# clone) and Lean. The diff isolates math-kernel correctness from RNG/init drift.
.PHONY: parity
parity:
	uv run scripts/dump_parity.py
	lake exe autograd parity
