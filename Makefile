# ny Makefile
# Build, test, benchmark, and profile commands

.PHONY: all build test bench bench-html profile flamegraph clean clean-debug disk-usage test-python test-python-plugin lint lint-tippy lint-clippy fmt fmt-fix deny proofs-check gate gate-fast

# Default target
all: build test

# Build
build:
	cargo build --release

build-debug:
	cargo build

# Testing
test:
	cargo test --release

test-verbose:
	cargo test --release -- --nocapture

# Python tests (ny-python package)
# Requires: pip install maturin pytest (or: pip install ".[dev]" from crates/ny-python)
test-python:
	@command -v pytest >/dev/null 2>&1 || { echo "pytest not found. Install with: pip install pytest"; exit 1; }
	@command -v maturin >/dev/null 2>&1 || { echo "maturin not found. Install with: pip install maturin"; exit 1; }
	@echo "Building ny-python extension with maturin..."
	cd crates/ny-python && maturin develop --release
	@echo "Running Python tests..."
	cd crates/ny-python && python -m pytest tests/ -v

# Python plugin tests only (no Rust build needed, pure Python)
test-python-plugin:
	@command -v pytest >/dev/null 2>&1 || { echo "pytest not found. Install with: pip install pytest"; exit 1; }
	@echo "Running ny pytest plugin tests..."
	cd crates/ny-python && python -m pytest ny_pytest/tests/ -v

# Benchmarking with Criterion
bench:
	cargo bench

bench-propagate:
	cargo bench -p ny-propagate

bench-gpu:
	cargo bench -p ny-gpu

# Open HTML reports
bench-html:
	@echo "Opening benchmark reports..."
	@open target/criterion/report/index.html 2>/dev/null || \
		xdg-open target/criterion/report/index.html 2>/dev/null || \
		echo "Open target/criterion/report/index.html in your browser"

# Profiling with samply (Firefox Profiler compatible)
# Install: cargo install samply
profile-samply:
	@echo "Profiling with samply (opens Firefox Profiler)..."
	cargo build --release --example profile -p ny-propagate
	samply record ./target/release/examples/profile

# Profiling with cargo-flamegraph
# Install: cargo install flamegraph
# macOS: Requires dtrace permissions (SIP may need adjustment)
# Linux: Requires perf
flamegraph:
	@echo "Generating flamegraph..."
	cargo flamegraph --release --example profile -p ny-propagate -o flamegraph.svg
	@echo "Flamegraph saved to flamegraph.svg"

# Profile specific benchmark
flamegraph-bench:
	@echo "Generating flamegraph for benchmarks..."
	cargo flamegraph --release --bench propagation -p ny-propagate -o flamegraph-bench.svg -- --bench
	@echo "Flamegraph saved to flamegraph-bench.svg"

# Memory profiling with DHAT (heap profiler)
# Add dhat = "0.3" to dev-dependencies and use #[global_allocator]
profile-dhat:
	@echo "DHAT profiling requires code instrumentation."
	@echo "See: https://docs.rs/dhat"

# Quick performance check
perf-check:
	cargo build --release --example profile -p ny-propagate
	./target/release/examples/profile

# Lint gate (default: stock clippy).
#   lint-tippy  - tippy, the Trust toolchain's lint tool (newer clippy base than
#                 the pinned stock toolchain, so it sees lints stock clippy doesn't);
#                 optional — run separately, skips when the toolchain is absent
#   lint-clippy - stock cargo clippy on the pinned toolchain (rust-toolchain.toml)
lint: lint-clippy

lint-clippy:
	cargo clippy --all-targets --all-features -- -D warnings

# Fast, incremental lint gate — the compiled ny-lint-guard binary runs clippy
# only on the ny-* crates changed vs origin/main (no CI, so drift is caught
# locally in seconds). `lint-fix` auto-applies the mechanically-safe fixes and
# lists the rest (soundness-sensitive lints are never auto-rewritten).
lint-fast:
	cargo run -q -p ny-lint-guard -- check

lint-fix:
	cargo run -q -p ny-lint-guard -- fix

# Tippy requires the external Trust stage2 toolchain (not included in this
# repo): set TRUST_STAGE2 to an installed stage2 tree (with targo/tippy). The
# stock `lint` target remains public and portable, but an explicit
# `lint-tippy` invocation is a real gate and must never silently skip.
# CAVEAT: tippy carries a toolchain-integrity guard that snapshots EVERY
# ancestor directory up to `/` — a toolchain under $HOME gets rejected
# whenever anything churns a home-dir entry mid-run. Install the stage2 tree
# to an ancestor-quiet path instead (rsync -a the stage2 build tree to a
# shared, quiet location) and do not modify/rebuild the toolchain while
# this target is running.
TRUST_STAGE2 ?= $(HOME)/.rustup/toolchains/trust
lint-tippy:
	@test -n "$(TRUST_STAGE2)" && [ -x "$(TRUST_STAGE2)/bin/targo" ] || { \
		echo "Trust stage2 toolchain not found at $(TRUST_STAGE2)."; \
		echo "Build/link Trust or set TRUST_STAGE2 explicitly; use 'make lint-clippy' for the stock-only gate."; \
		exit 1; }
	PATH="$(TRUST_STAGE2)/bin:$$PATH" "$(TRUST_STAGE2)/bin/targo" tippy --all-targets --all-features -- -D warnings

# First-party drift gate (no CI exists, so every push-storm re-drifts the
# checks the audits keep re-zeroing by hand; this is the one local command
# that runs them all). Per-check PASS/FAIL scoreboard, nonzero exit on any
# FAIL. See scripts/gate.sh --help for the check list and --only <letter>.
#   gate      - full gate: clippy drift + ny-cert/ny-cli/ny-propagate test
#               slices + harness pytest + submission-packaging invariants
#   gate-fast - clippy drift + ny-cert + packer unit tests only
gate:
	scripts/gate.sh

gate-fast:
	scripts/gate.sh --fast

# RUSTSEC advisory + license gate (see deny.toml)
# Install: cargo install cargo-deny
deny:
	@command -v cargo-deny >/dev/null 2>&1 || { echo "cargo-deny not found. Install with: cargo install cargo-deny"; exit 1; }
	cargo deny check advisories
	cargo deny check licenses

# Type-check the Kani proof harness. It is workspace-excluded and carries its own
# manifest, so no workspace command reaches it and it rots silently otherwise.
# Its harnesses are `#[cfg(kani)]`-gated, so this compiles the proof-support code
# without kani installed; `cargo kani` in that directory runs the proofs.
# (crates/ny-cert/proofs/kani is not checkable this way — its harnesses are
# ungated, so it only compiles under `cargo kani`, which injects the kani crate.)
proofs-check:
	cargo check --manifest-path proofs/kani/Cargo.toml --all-targets

# Format check
fmt:
	cargo fmt --all -- --check

fmt-fix:
	cargo fmt --all

# Clean targets (for managing 10-50GB build artifacts)
# Use 'make clean-debug' for incremental cleanup, 'make clean' for full reset

# Remove debug artifacts only (keeps release builds, saves rebuild time)
# Typically removes 10-20GB, keeps 2-3GB release artifacts
clean-debug:
	@echo "Removing debug build artifacts..."
	rm -rf target/debug/
	@echo "Debug artifacts removed. Release builds preserved."
	@du -sh target/ 2>/dev/null || echo "target/ removed"

# Full clean (removes all build artifacts)
clean:
	cargo clean
	rm -f flamegraph*.svg
	rm -f perf.data*

# Show current disk usage by target/ subdirectories
disk-usage:
	@echo "Build artifact disk usage:"
	@du -sh target/ 2>/dev/null || echo "No target/ directory"
	@echo ""
	@echo "Breakdown:"
	@du -sh target/*/ 2>/dev/null | sort -hr || echo "No subdirectories"

# Install profiling tools
install-tools:
	@echo "Installing profiling tools..."
	cargo install samply
	cargo install flamegraph
	cargo install cargo-criterion
	@echo "Done. Note: flamegraph may require additional system setup."
	@echo "  macOS: May need to disable SIP or use dtrace permissions"
	@echo "  Linux: Ensure perf is installed (linux-tools-generic)"

# Help
help:
	@echo "ny Build & Profile Commands"
	@echo ""
	@echo "Building:"
	@echo "  make build        - Release build"
	@echo "  make build-debug  - Debug build"
	@echo ""
	@echo "Testing:"
	@echo "  make test              - Run all Rust tests"
	@echo "  make test-verbose      - Run Rust tests with output"
	@echo "  make test-python       - Build ny-python and run Python tests"
	@echo "  make test-python-plugin - Run pytest plugin tests (no build)"
	@echo ""
	@echo "Benchmarking:"
	@echo "  make bench        - Run all Criterion benchmarks"
	@echo "  make bench-propagate - Benchmark ny-propagate only"
	@echo "  make bench-gpu    - Benchmark ny-gpu only"
	@echo "  make bench-html   - Open benchmark HTML reports"
	@echo ""
	@echo "Profiling:"
	@echo "  make perf-check   - Quick performance check"
	@echo "  make profile-samply - Profile with samply (Firefox Profiler)"
	@echo "  make flamegraph   - Generate flamegraph SVG"
	@echo ""
	@echo "Cleanup (target/ can grow to 10-50GB):"
	@echo "  make disk-usage   - Show current artifact sizes"
	@echo "  make clean-debug  - Remove debug only (keeps release, saves rebuild time)"
	@echo "  make clean        - Full clean (removes all build artifacts)"
	@echo ""
	@echo "Tools:"
	@echo "  make install-tools - Install samply, flamegraph, criterion"
	@echo "  make gate         - First-party drift gate: clippy + test slices + submission invariants (scoreboard)"
	@echo "  make gate-fast    - Fast drift gate: clippy + ny-cert + packer unit tests"
	@echo "  make lint         - Lint gate (stock clippy)"
	@echo "  make lint-tippy   - Trust toolchain tippy gate (requires stage2; set TRUST_STAGE2 if needed)"
	@echo "  make lint-clippy  - Stock cargo clippy only"
	@echo "  make fmt          - Check formatting"
	@echo "  make deny         - RUSTSEC advisory + license gate (cargo-deny)"
	@echo "  make proofs-check - Type-check the Kani proof harness"
