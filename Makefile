.PHONY: init
init:
	git config core.hooksPath .githooks

.PHONY: precommit
precommit:
	lake build
	lake build Tests

.PHONY: run
run:
	lake exe autograd
