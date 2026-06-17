# Thinking Guidelines

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

Tradeoff: These guidelines bias toward caution over speed. For trivial tasks, use judgment.

(1) Think Before Coding

Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

(2) Simplicity First

Minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

(3) Surgical Changes

Touch only what you must. Clean up only your own mess.

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

(4) Goal-Driven Execution

Define success criteria. Loop until verified.

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

These guidelines are working if: fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

# Coding Guidelines

Simplicity:

- no code > simple code > clever code
- more dependencies is better than more code
- minimize indentation: guard first programming, early returns, happy path first
- maximize for code locality: a little duplication is okay. inline small functions (<5 LoC). no module-level globals or top-level helpers unless reused in multiple places
- avoid OOP primitives like classes and inheritance where possible
- minimum viable diff. no schedulers, no early stopping, no eval-during-train
- no shared utility modules. each standalone script duplicates its own boilerplate

Avoid LLM smells:

- avoid docstrings where possible. prefer tight technical comments
- never use emdashes

Robustness:

- use `assert` statements frequently as invariants. prefer crashing over failing silently. prefer `assert` over runtime exceptions
- maximize test coverage, without adding noise (except for standalone scripts)
- prefer typed enums over strings for states
- use match statements over multiple `isinstance` branches

Performance:

- have mechanistic sympathy, design with data oriented programming best-practices

# Writing Guidelines

applies to JOURNAL, the Mission/Runbook sections above, README-like prose. tightness over completeness.

- never write a blob of text. blobs do not get read.
- sentences must fit max plaintext line width. if a sentence is too long, split it.
- one statement per line. no wrapping inside a sentence.
- prefer ascii diagrams and tables over many words.
- bullets over paragraphs for lists. period.
- visual structure beats prose. headings, sublists, tables, indentation.
- no semicolons. no emdashes. no emojis.
- enumerate as `(1)`, `(2)`, `(a)`, `(b)`. never `1)` or `a)`.
- each chapter is self-contained. no cross-references like "see ch7" or "per the audit above". restate what the reader needs.
- titles are findings, not process. "v4 frozen, LR is the knob" beats "overnight + morning campaign".
- drop process narration. no "~9 hours", "after 30 configs", dates of when work happened.
- drop redundant text. if the code or the Runbook already says it, don't repeat it.
- drop section banners that paraphrase code or function names.
- no cryptic shorthands without explanation. if a term is non-obvious, expand on first use or list in a legend.
- anchor jargon with a concrete example. show the worksheet for 27 * 64 before claiming "we predict a chargrid".
- diagrams: annotate each block with what it does and why. naked boxes are useless.
- diagrams: single-line annotations to the right of the block or arrow. don't stack multiple `v` arrows just to fit a label. extend the annotation line instead.
- "other ideas:" as bullets, not "deferred alternatives" prose.
- numbers must be backed by a file. cite `runs/.../summary.json` or omit the number.
- when you correct an earlier claim, fix it in place. don't add "actually it was X (was Y before)" parenthetical.

Prose smells to avoid (LLM tells, kill on sight):

- punchline sentences. "The wall was always architectural." "Capacity is never the answer." "Then the breakthrough came." dramatic single-clause sentences that try to land. cut or merge.
- consecutive short sentences used as rhetoric. "It worked. The rollout climbed. Honesty held." reads like a movie trailer. combine into a normal sentence or a bulleted list.
- "X is the Y of Z" formulations. "perprojs is the breakthrough of Stage 1". just describe what it does.
- "not just X, it's Y" intensifiers. "not just a fix, it's a redesign". just say what it is.
- "does not merely X but Y" rhetoric. "does not merely route but separates". same. drop "merely".
- editorial commentary about the work itself. "This is the deliverable." "The implications are clear." just state facts.
- exclamation marks anywhere. ever.
