.PHONY: init
init:
	printf '#!/bin/sh\nmake precommit\n' > .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit

.PHONY: precommit
precommit:
	lake build

.PHONY: run
run:
	lake exe parity
