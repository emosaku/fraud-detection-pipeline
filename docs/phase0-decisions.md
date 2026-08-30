# Phase 0 — Engineering Decisions

Scope: project scaffolding, tooling, and environment. No pipeline logic.

## Standalone git repo, not nested in the parent workspace

The project was initialized inside a larger, unrelated local workspace directory
(containing other unrelated projects and messy in-progress changes). Rather than
tracking it inside that parent repo, it got its own fresh `git init` and its own
GitHub repo.

**Why:** a portfolio project should be independently cloneable and reviewable without
dragging along unrelated history or files. Nesting it would have coupled its commit
history to unrelated work happening in sibling directories.

## src-layout package with one subpackage per phase

`src/fraud_pipeline/` with `ingestion/`, `features/`, `models/`, `scoring/`,
`storage/`, `monitoring/`, `dashboard/` — created empty in Phase 0, populated as each
phase is built.

**Why:** the phase boundaries in the roadmap are also natural module boundaries. Rather
than inventing a generic structure and figuring out where phase-specific code goes
later, the package layout was decided up front to mirror the roadmap 1:1. `src-layout`
(rather than a flat top-level package) is the standard modern choice for an installable
package — it prevents accidentally importing from the working directory instead of the
installed package.

## `pyproject.toml`, `hatchling`, plain `venv` — no `uv`/`poetry`

Dependency and build management uses `pyproject.toml` with `hatchling` as the build
backend, installed into a plain `python -m venv`.

**Why:** neither `uv` nor `poetry` was already configured on the development machine,
and introducing either would add a tool dependency with no benefit at this project's
scale. `hatchling` is a lightweight, standard backend for a pure-Python src-layout
package — no compiled extensions, no need for anything heavier.

## Minimal upfront dependencies, everything else deferred

Phase 0 installed only `pandas`, `numpy`, `python-dotenv`, `pytest`, `ruff` — every
other dependency (`apache-beam`, `scikit-learn`, `google-cloud-bigquery`, etc.) was
deliberately left out and tracked in `docs/future-dependencies.md` with the phase that
would introduce it.

**Why:** installing a phase's dependencies only when that phase actually starts keeps
the environment honest about what's actually being used, and avoids a wall of unused
packages that make the project harder to reason about. In practice, both
`google-cloud-pubsub` and `apache-beam[gcp]` ended up promoted to core dependencies
earlier than the original plan, in Phase 1 and Phase 2 respectively — the "future"
table was a starting guess, not a contract, and got corrected as real work arrived at
each dependency instead of speculatively front-loading them.

## `ruff` for both lint and format

One tool instead of `black` + `flake8`/`isort` separately.

**Why:** less configuration surface, one command for each concern
(`ruff check` / `ruff format`), and it's fast enough to run on every save without
friction.

## `.env` / `.env.example` split, `data/raw` and `data/processed` git-ignored with `.gitkeep`

`.env.example` is committed with blank placeholders; `.env` itself is git-ignored.
`data/raw/` and `data/processed/` are git-ignored (contents, not the directories
themselves — `.gitkeep` keeps the empty folders present in a fresh clone).

**Why:** standard practice for keeping secrets and generated/downloaded data out of
version control while still documenting what configuration and directory structure a
fresh clone needs. This convention got tested for real in Phase 1 and Phase 2 — a
Kaggle API token and, briefly, real GCP project values ended up in `.env.example` by
mistake at different points and had to be caught and reverted, which is exactly the
failure mode this split is meant to prevent.
