# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1](https://github.com/nayrosk/overbrainer/compare/v0.2.0...v0.2.1) - 2026-09-26

### Documentation

- fix the rustdoc warnings and check the docs in CI ([#30](https://github.com/nayrosk/overbrainer/pull/30))

### Fixed

- *(exec)* clear the cleartext-logging alerts CodeQL raised on tests ([#27](https://github.com/nayrosk/overbrainer/pull/27))

## [0.2.0](https://github.com/nayrosk/overbrainer/compare/v0.1.1...v0.2.0) - 2026-09-25

### Added

- *(skill)* ship an agent skill and install it with overbrainer skill install ([#14](https://github.com/nayrosk/overbrainer/pull/14))
- *(cli)* complete run IDs, topics and targets in bash, zsh and fish ([#13](https://github.com/nayrosk/overbrainer/pull/13))

## [0.1.1](https://github.com/nayrosk/overbrainer/compare/v0.1.0...v0.1.1) - 2026-09-24

### Documentation

- *(readme)* install the published crate and add its badge ([#10](https://github.com/nayrosk/overbrainer/pull/10))

## [0.1.0](https://github.com/nayrosk/overbrainer/releases/tag/v0.1.0) - 2026-09-24

### Added

- *(tui)* crimson look, context footer and subtle motion; abandon a starting run without quitting ([#8](https://github.com/nayrosk/overbrainer/pull/8))
- add a terminal UI ([#7](https://github.com/nayrosk/overbrainer/pull/7))
- train on Runpod GPU pods ([#6](https://github.com/nayrosk/overbrainer/pull/6))
- *(cli)* add train, train attach, train cancel and runs ls
- *(runs)* start, follow and cancel training runs
- *(runs)* add run IDs, run.json records and metric summaries
- *(events)* add training metric and job status events
- *(exec)* run jobs over SSH with tar transfers
- *(exec)* build container and native job commands
- *(exec)* run jobs locally in their own process group
- *(exec)* add the Executor trait, job scripts and line reading
- *(train)* render the Axolotl config, commands and artifacts
- *(train)* add the metrics plugin and parse the lines it writes
- *(train)* write the Axolotl config as YAML that PyYAML reads back
- *(config)* add training hyperparameters and target runtime options
- *(llm)* support budgeted extended thinking for Claude 4.5 models
- *(cli)* add the pipeline commands with progress and cost summary
- *(pipeline)* add the answers and split stages
- *(pipeline)* add the subtopics and questions stages
- *(events)* add the stage event bus
- *(dedup)* add lexical and embedding deduplication
- *(prompts)* add overridable prompt templates and write them on init
- *(pricing)* read model prices from the provider listing
- *(llm)* add the anthropic protocol and build clients from config
- *(llm)* add the openai protocol client with reasoning extraction
- *(llm)* add client trait, completion types, errors and retries
- *(dataset)* add records, stable IDs and crash-safe JSONL files
- *(config)* add per-role request settings and pipeline tunables
- *(cli)* add init and config check commands
- *(secrets)* resolve literal and vault KV v2 secret references
- *(config)* load layered settings and reject env-only keys in file
- *(config)* add typed settings and semantic validation

### Changed

- *(exec)* stop re-exporting CONTAINER_ROOT
- *(pricing)* read model listings in one place

### Documentation

- say where a container engine keeps the Hugging Face token
- *(runs)* say how many failures MAX_FAILURES really allows
- *(exec)* state what environment a job starts with
- correct the cancel and retry wording
- document training, targets and runs
- *(llm)* reword the thinking configuration comment
- *(dataset)* state what the ID separator guarantees
- say that split does not accept --force
- document the data pipeline, configuration and releases
- fix precedence, vault example and secret handling notes
- document setup, configuration and secrets

### Fixed

- *(build)* drop the science::ml category crates.io does not know ([#9](https://github.com/nayrosk/overbrainer/pull/9))
- *(templates)* pin the example docker image by digest
- *(config)* keep an out-of-range integer a string, not a float
- *(exec)* signal a job's process group under every sh
- *(runs)* drain the whole metrics file when the job ends
- *(exec)* validate a job's secrets on every target
- *(exec)* keep an interrupted cancel from mislabelling a finished run
- *(cli)* cancel a run in any state that still has a job
- *(cli)* load the configuration once and explain what cancel needs
- *(runs)* report a cancel that succeeded even when its record cannot be updated
- *(runs)* cancel untracked jobs and keep logs of cancelled runs
- *(runs)* validate run IDs on every path and skip unreadable records
- *(exec)* align missing run dir downloads and harden SSH tests
- *(exec)* explain why a cancel can find the group still present
- *(exec)* cap the tar error text
- *(exec)* drain tar stderr, keep secrets out of local jobs and guard nested copies
- *(exec)* let cancel markers take priority over the exit code in status
- *(exec)* report cancelling jobs as running and never stall on long lines
- *(exec)* make cancel and status safe against finished and recycled jobs
- *(train)* refine the template warning, eval cadence and copy errors
- *(train)* create the metrics directory before the plugin writes to it
- *(train)* escape YAML line-break characters in quoted strings
- *(config)* keep non-canonical integers in axolotl_extra as strings
- *(config)* reject empty target image and venv values
- *(config)* give env values of axolotl_extra their types
- *(config)* restrict target paths and image references to safe characters
- *(dedup)* return ready futures from the lexical deduplicator
- *(dataset)* read JSONL files as bytes
- *(pipeline)* warn when every usable answer is orphaned
- *(pipeline)* spread the eval set over topics, then subtopics
- *(pipeline)* log the provider error when saving after a stop fails
- *(config)* say "at line N, column M" in TOML syntax errors
- *(config)* reject a temperature together with thinking
- *(config)* cap pipeline.concurrency at 1024
- *(pipeline)* leave orphaned answers out of the split
- *(pipeline)* size the eval set from all usable answers
- *(pipeline)* never train reasoning from parents that hide it
- *(dedup)* embed in chunks and skip seeding topics with nothing to fill
- *(dataset)* keep a valid last line that only lost its newline
- *(cli)* resolve secrets lazily and connect each role once
- *(pipeline)* train only raw reasoning and keep finished answers on fatal stop
- *(pipeline)* resume partial topics and isolate per-item dedup failures
- *(pipeline)* report an item's final failure exactly once
- *(pipeline)* match a JSON array past bracketed prose
- *(dedup)* retry embedding requests under the pipeline's retry policy
- *(config)* never echo TOML source in syntax errors
- *(config)* keep offending values out of type error messages
- *(config)* reject NaN max_hours and split validation into focused checks
- *(cli)* honor -C in init and merge an existing .gitignore
- *(logging)* keep dependency error lines out of default output
- *(secrets)* keep vault error sources and harden token lookup
- *(cli)* never print .env content on a parse error
- *(config)* reject log in overbrainer.toml
- *(config)* accept env overrides of numeric target fields
- *(config)* replace load's generic hasher with an EnvSource enum
