.PHONY: precommit
precommit:
	printf '#!/bin/sh\nmake precommit\n' > .git/hooks/pre-push && chmod +x .git/hooks/pre-push
	lake build
	lake exe parity

.PHONY: run
run: precommit
