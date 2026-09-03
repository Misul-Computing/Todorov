# Operating directive

status: current (as of 2026-07-14).

This file defines how the project maintains its wiki and state documentation.
It is binding for every agent and human who touches the project. It supersedes
any previous convention that contradicts it.

This directive is a living document. Its update-history section is append-only.

## Scope and authority

This directive governs:

- Every file under `neuroloc/wiki/`
- `neuroloc/wiki/PROJECT_PLAN.md`, `state/program_status.yaml`, and
  `docs/STATUS_BOARD.md`
- `AGENTS.md` and `neuroloc/HANDOFF.md`
- Run cards and mistake documents

It does not govern:

- Python source under `neuroloc/model/*.py` and
  `neuroloc/simulations/**/*.py`; code rules live in `AGENTS.md`
- External dependencies and packaging
- The gitignored `neuroloc/output/` tree

Markdown files under `neuroloc/model/` remain in scope. The exemption applies
only to source code.

When rules conflict, use this order:

1. The user's explicit written instruction
2. For code questions, `AGENTS.md` over this directive
3. For wiki and state questions, this directive over `AGENTS.md`
4. Individual wiki articles
5. Machine-generated summaries

The user's explicit instruction always overrides both governing documents.

## Source-of-truth hierarchy

Resolve document disagreements in this order:

1. **Code wins over documentation.** If implementation and prose disagree about
   implemented behavior, repair the prose.
2. **Later commit wins by default.** When wiki articles make incompatible
   claims, the later commit takes precedence. Mark the older article as
   superseded or narrow its scope. If the later commit is itself wrong, append
   a correction rather than silently rewriting an append-only record.
3. **State files have named roles.**
   - `PROJECT_PLAN.md` is canonical for project state, the current question,
     decision rules, and update history.
   - `program_status.yaml` is the machine-readable mirror for `latest_run`,
     `latest_run_results`, `next_action`, and `run_history`.
   - `STATUS_BOARD.md` is the human-readable status mirror.
   - `AGENTS.md` is canonical for agent-facing rules and the run-summary trail.
4. **Wiki articles are reasoning records.** Promote a claim to
   `PROJECT_PLAN.md` or `AGENTS.md` before treating it as binding.

## Article lifecycle

Every wiki article has exactly one lifecycle state.

### `current`

The article reflects the present understanding and is the active reference for
its topic. It may be edited as evidence changes.

Banner: `status: current (as of YYYY-MM-DD).`

Eligible directories: `synthesis/`, `tests/`, `knowledge/`, `concepts/`,
`mechanisms/`, `bridge/`, `comparisons/`, and `entities/`.

### `superseded by <link>`

A newer article has overtaken the article. Retain it for evidence continuity
and point readers to the replacement.

Banner: `status: superseded by <path>. retained for evidence continuity.`

The superseding article must carry a reverse link in its final `See also`
section.

### `historical context only`

The article records a completed event, run, or dated hypothesis. It is frozen
evidence rather than a current interpretation.

Banner: `status: historical context only. frozen as of YYYY-MM-DD. do not edit.`

Eligible directories: `mistakes/` and run-specific entries under `tests/`.

Only these edits are permitted:

- Typographic or formatting fixes that do not change meaning
- A final `See also` forward pointer
- An appended factual correction

Append a factual correction under `## Correction (YYYY-MM-DD)` immediately
before the final `## See also` section. Never rewrite or delete the original
claim. If the correction is later falsified, append another dated correction.

### `definitional`

The article defines a term, mechanism, entity, or concept. It changes only when
the underlying definition changes.

Banner: `status: definitional. last fact-checked YYYY-MM-DD.`

Eligible directories: `concepts/`, `entities/`, `mechanisms/`, and selected
`bridge/` articles.

## Banner format

The lifecycle banner must be the first non-heading line. No prose may appear
between the title and banner. The machine-readable banner remains lowercase
even though current prose uses professional sentence case.

## Migration policy

Articles that predate the 2026-04-16 directive and have not been touched remain
in a pre-migration state. A modification, scheduled refactor, or required
reverse link triggers migration: the same change must add the correct banner
and final `See also` section.

The migration order is `synthesis/`, `tests/`, `mistakes/`, `knowledge/`,
`bridge/`, `comparisons/`, `concepts/`, `entities/`, then `mechanisms/`. Once
migration completes, every referenced wiki article must satisfy the lifecycle
format.

## Append-only sections

Never edit prior entries in these surfaces:

- `PROJECT_PLAN.md` under `Update history`
- `AGENTS.md` under `Results summary`, `Bug history`, `Phase sequencing`, and
  `Architecture rules`
- Every article under `mistakes/`

Append corrections or clarifications. Do not launder evidence by silently
rewriting a recorded mistake.

## Cross-references

Every migrated article carries `## See also` as its last section. Use
bidirectional links when one article supersedes another, cites another as
load-bearing evidence, or pairs a paid-run card with a mistake record. Both
sides of such a link belong in the same change.

## Prosecutor protocol

Run the prosecutor on changes to:

- `neuroloc/wiki/synthesis/`, `neuroloc/wiki/mistakes/`, or `neuroloc/wiki/tests/`
- `PROJECT_PLAN.md` or `OPERATING_DIRECTIVE.md`
- `program_status.yaml`, `STATUS_BOARD.md`, or `AGENTS.md`
- Paid-run cards

The prosecutor must reach zero findings:

1. Review the complete changed surface with `feature-dev:code-reviewer`.
2. Fix every finding, regardless of priority.
3. Search for the entire bug class, not only the named instance.
4. Re-run the reviewer.
5. Repeat until it returns zero findings.

An explicit written waiver from Deyan is the only alternative to a fix. Wiki
changes outside the listed directories may skip review only when they do not
alter a factual claim.

## Run cards

Every paid run produces `neuroloc/wiki/tests/<run_name>_results.md` with a historical
banner. The gitignored output tree is ephemeral; preserve the run card in the
wiki when the result matters.

The run card cross-links any synthesis article informed by the result, any
mistake document for that run, and the prior runs it compares against.

## File naming

- Use lowercase words separated by underscores for ordinary article names.
- Keep one article per claim or event.
- Reserve capitalized top-level names for meta-documents such as
  `PROJECT_PLAN.md`, `OPERATING_DIRECTIVE.md`, `INDEX.md`, `HANDOFF.md`, and
  `AGENTS.md`.

## Documentation style

Current prose uses professional sentence case. File names, lifecycle banners,
and machine keys keep their established casing conventions. Lowercase
historical records do not need a style-only rewrite; edit them only when their
status, path, or factual content requires correction.

## Disagreement handling

The user's explicit written instruction wins. If an agent disagrees, it states
the evidence and stops the disputed action until the user decides.

`AGENTS.md` wins on code rules. This directive wins on wiki and state rules.
Use the stricter rule when a question spans both domains.

Later commit wins by default. If the later commit is itself wrong, append a
correction rather than silently rewriting an append-only record.

## Update history

Append every revision with date, author, and a one-line explanation.

- **2026-04-16** — deyan todorov — file created. first draft after the five-paid-runs diagnosis cycle concluded. scope, source-of-truth hierarchy, four-state article lifecycle, banner format, append-only sections, bidirectional cross-reference rule, prosecutor protocol for wiki changes, run-card location rule, file naming, disagreement resolution. supersedes the ad-hoc practice documented in `CLAUDE.md` under "keeping this file current" and the scattered rules across prior mistake docs.
- **2026-04-16** — deyan todorov — first-round prosecutor fixes. C1: directive's own banner was two sentences; trimmed to the single line the format mandates with explanatory prose moved below. C2: added the "migration policy for pre-existing articles" section defining pre-migration state, migration triggers, scheduled completion order, and post-migration enforcement. C3: narrowed the scope exemption from `neuroloc/model/` and `neuroloc/simulations/` to their `*.py` files only, so run cards or other markdown that happen to live under those directories are in scope. I1: authority-order table was unconditional (CLAUDE.md > directive always) while the conflict-resolution section split by domain; reconciled by rewriting the table to carry the domain split itself. I2: added a concrete format for appended corrections on `historical context only` articles (`## correction (YYYY-MM-DD)` heading, append-only, preserving original text). M1: removed the see-also entry pointing at the planned `INDEX.md` (the directive's own rule forbids see-also entries to non-existent articles).
- **2026-04-16** — deyan todorov — second-round prosecutor fix. the first-round C1 fix trimmed the banner to one line but left three lines of introductory prose ABOVE the banner at lines 3-5. the directive's own banner-format rule says the banner must be the first non-heading content after the title, with no other opening content permitted. the directive itself was in violation. fix: moved the three lines of intro prose to below the banner, so the file now reads title, blank, banner, blank, intro prose, blank, living-document note. no content removed; only relocated.
- **2026-07-14** — deyan todorov — replaced the nonexistent `CLAUDE.md` authority with the repository's actual `AGENTS.md` instruction surface in all normative rules while preserving the dated historical entries.
- **2026-07-14** — deyan todorov — replaced the lowercase-only prose convention with professional sentence case for current documentation while preserving historical records unless they need substantive correction.
- **2026-07-14** — Deyan Todorov — Consolidated the current normative rules in professional sentence case after prosecutor review while preserving all prior update-history entries verbatim.

## See also

- `neuroloc/wiki/PROJECT_PLAN.md` — Canonical project state
- `AGENTS.md` — Agent-facing code and workflow rules
- `neuroloc/wiki/INDEX.md` — Wiki navigation map
