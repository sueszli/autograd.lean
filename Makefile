.PHONY: precommit
precommit:
	lake build

.PHONY: run
run:
	# precommit hook setup
	git config core.hooksPath .githooks
	# train data download
	mkdir -p data
	curl -fsSL https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt -o data/input.txt
	# reference for parity check
	uv run Parity/dump_original_weights.py
	# run
	lake exe autograd
