# Makefile – one-command workflows for Paraphina

PY = .venv/bin/python

.PHONY: fmt check test research profiles ci

# Format Rust code
fmt:
	cargo fmt

# Lint Rust code strictly
check:
	cargo clippy -- -D warnings

# Run all Rust tests + doc-tests
test:
	cargo test

# Full research / calibration suite (Exp06)
research:
	$(PY) tools/exp06_full_research_suite.py

# Quick 10k-tick smoke runs for each profile
profiles:
	cargo run -- --ticks 10000 --profile conservative
	cargo run -- --ticks 10000 --profile balanced
	cargo run -- --ticks 10000 --profile aggressive

# CI entrypoint: fast checks only
ci: fmt check test
