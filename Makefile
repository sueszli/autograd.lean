.PHONY: run
run:
	printf '#!/bin/sh\nlake build\n' > .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit
	lake build
	lake exe parity
