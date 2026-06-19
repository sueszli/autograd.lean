.PHONY: init
init:
	git config core.hooksPath .githooks

.PHONY: precommit
precommit:
	lake build

.PHONY: run
run:
	lake exe parity
