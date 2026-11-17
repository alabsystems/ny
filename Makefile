# ny Makefile
# Build, test, benchmark, and profile commands
#
# Builds, tests and benchmarks use `targo --unverified` without a compiler proof;
# lint and format use `targo tippy` / `targo fmt`. publish/config.sh defines the
# release gate: an unverified workspace build and relaxation-envelope test floor.

PYTHON ?= python3

.PHONY: all build build-debug test test-verbose test-python test-python-plugin test-python-tooling bench bench-propagate bench-gpu bench-html profile-samply flamegraph flamegraph-bench profile-dhat perf-check lint lint-fast lint-fix lint-tippy gate gate-fast deny fmt fmt-fix clean-debug clean disk-usage install-tools help

# Default target
all: build test

# Build
build:
	targo --unverified build --release

build-debug:
	targo --unverified build

# Tests for the workspace's default members; see CONTRIBUTING.md for all crates.
test:
	targo --unverified test --release

test-verbose:
	targo --unverified test --release -- --nocapture

# Python tests (ny-python package)
# Requires: pip install maturin pytest (or: pip install ".[dev]" from crates/ny-python)
test-python:
	@$(PYTHON) -c 'import pytest' >/dev/null 2>&1 || { echo "pytest not importable by $(PYTHON). Install with: $(PYTHON) -m pip install pytest"; exit 1; }
	@command -v maturin >/dev/null 2>&1 || { echo "maturin not found. Install with: pip install maturin"; exit 1; }
	@echo "Building ny-python extension with maturin..."
	cd crates/ny-python && maturin develop --release
	@echo "Running Python tests..."
	cd crates/ny-python && $(PYTHON) -m pytest tests/ -v

# Python plugin tests only (no Rust build needed, pure Python)
test-python-plugin:
	@$(PYTHON) -c 'import pytest' >/dev/null 2>&1 || { echo "pytest not importable by $(PYTHON). Install with: $(PYTHON) -m pip install pytest"; exit 1; }
	@echo "Running ny pytest plugin tests..."
	cd crates/ny-python && $(PYTHON) -m pytest ny_pytest/tests/ -v

# Repository Python/shell tooling tests (no extension build needed).
test-python-tooling:
	@env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 $(PYTHON) -s -c 'import numpy, onnx, onnxruntime, pytest, sys, yaml; __import__("tomllib" if sys.version_info >= (3, 11) else "tomli")' >/dev/null 2>&1 || { echo "Python tooling dependencies are not importable in isolated mode by $(PYTHON). Use the README virtualenv setup, install requirements.txt there, and invoke make with PYTHON set to that interpreter."; exit 1; }
	env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 $(PYTHON) -s -m pytest tests/ -q
	NY_TEST_PYTHON="$(PYTHON)" targo --unverified test --locked -p ny-test-utils --features python-tools --tests

# Benchmarking with Criterion
bench:
	targo --unverified bench

bench-propagate:
	targo --unverified bench -p ny-propagate

bench-gpu:
	targo --unverified bench -p ny-gpu

# Open HTML reports
bench-html:
	@echo "Opening benchmark reports..."
	@open target/criterion/report/index.html 2>/dev/null || \
		xdg-open target/criterion/report/index.html 2>/dev/null || \
		echo "Open target/criterion/report/index.html in your browser"

# Profiling with samply (Firefox Profiler compatible)
# Install: targo --unverified install samply
profile-samply:
	@echo "Profiling with samply (opens Firefox Profiler)..."
	targo --unverified build --release --example profile -p ny-propagate
	samply record ./target/release/examples/profile

# Profiling with the `flamegraph` binary (crate flamegraph)
# Install: targo --unverified install flamegraph
# macOS: Requires dtrace permissions (SIP may need adjustment)
# Linux: Requires perf
# The build and the profile are separate steps: `cargo flamegraph` re-enters
# `$CARGO build` without a lane, which targo refuses ("`targo build` refuses to
# create an implicitly unverified artifact"), so build here and profile the binary.
# The binary's path is read from the build's JSON messages (the `executable`
# of the named target), so it is right under any CARGO_TARGET_DIR.
BUILT_EXECUTABLE = $(PYTHON) -c 'import json, sys; exes = [m["executable"] for m in map(json.loads, sys.stdin) if m.get("reason") == "compiler-artifact" and m["target"]["name"] == sys.argv[1] and m.get("executable")]; print(exes[-1]) if exes else sys.exit("no executable for target " + sys.argv[1] + " in the build output")'
flamegraph:
	@echo "Generating flamegraph..."
	exe="$$(targo --unverified build --release --example profile -p ny-propagate --message-format=json-render-diagnostics | $(BUILT_EXECUTABLE) profile)" \
	  && flamegraph -o flamegraph.svg -- "$$exe"
	@echo "Flamegraph saved to flamegraph.svg"

# Profile specific benchmark
flamegraph-bench:
	@echo "Generating flamegraph for benchmarks..."
	exe="$$(targo --unverified bench --no-run -p ny-propagate --bench propagation --message-format=json-render-diagnostics | $(BUILT_EXECUTABLE) propagation)" \
	  && flamegraph -o flamegraph-bench.svg -- "$$exe" --bench
	@echo "Flamegraph saved to flamegraph-bench.svg"

# Memory profiling with DHAT (heap profiler)
# Add dhat = "0.3" to dev-dependencies and use #[global_allocator]
profile-dhat:
	@echo "DHAT profiling requires code instrumentation."
	@echo "See: https://docs.rs/dhat"

# Quick performance check
perf-check:
	targo --unverified build --release --example profile -p ny-propagate
	./target/release/examples/profile

# Lint gate: formatting + the whole-workspace clippy gate (-D warnings).
# ny-lint-guard runs the build tool's clippy (`targo tippy` under Trust, plain
# clippy if $CARGO is a stock cargo) over --all-targets --all-features.
lint: fmt
	targo --unverified run -q -p ny-lint-guard -- full

# Incremental gate for ny-* crates changed vs origin/main. `lint-fix` applies
# fixes except the four numeric lints listed in ny-lint-guard, then checks
# remaining warnings.
lint-fast:
	targo --unverified run -q -p ny-lint-guard -- check

lint-fix:
	targo --unverified run -q -p ny-lint-guard -- fix

# Override TRUST_STAGE2 to lint with another installed Trust tree. A missing
# frontend fails this target; do not modify the selected toolchain during a run.
TRUST_STAGE2 ?= $(HOME)/.rustup/toolchains/trust
lint-tippy:
	@test -n "$(TRUST_STAGE2)" && [ -x "$(TRUST_STAGE2)/bin/targo" ] || { \
		echo "Trust stage2 toolchain not found at $(TRUST_STAGE2)."; \
		echo "Build/link Trust or set TRUST_STAGE2 explicitly."; \
		exit 1; }
	PATH="$(TRUST_STAGE2)/bin:$$PATH" "$(TRUST_STAGE2)/bin/targo" tippy --locked --workspace --all-targets --all-features -- -D warnings

# Local drift gate. Each lane reports PASS, FAIL or VOID (missing prerequisites).
# gate-fast runs lanes a-h; gate runs a-o, including model loading, soundness
# oracles, API contracts, score-risk checks and advisory Trust verification.
# See scripts/gate.sh --help for prerequisites and --only <letter>.
gate:
	scripts/gate.sh

gate-fast:
	scripts/gate.sh --fast

# RUSTSEC advisory, dependency-ban/source, and license gate (see deny.toml)
# Install: targo --unverified install cargo-deny
# cargo-deny reads Cargo.lock through `$CARGO metadata`; targo dispatches it with
# CARGO=targo, and `targo metadata` needs no lane.
deny:
	@command -v cargo-deny >/dev/null 2>&1 || { echo "cargo-deny not found. Install with: targo --unverified install cargo-deny"; exit 1; }
	targo deny check advisories
	targo deny check bans
	targo deny check licenses
	targo deny check sources

# Format check
fmt:
	targo fmt --all -- --check

fmt-fix:
	targo fmt --all

# Remove debug artifacts while retaining release builds.
clean-debug:
	@echo "Removing debug build artifacts..."
	rm -rf target/debug/
	@echo "Debug artifacts removed. Release builds preserved."
	@du -sh target/ 2>/dev/null || echo "target/ removed"

# Full clean (removes all build artifacts)
clean:
	targo clean
	rm -f flamegraph*.svg
	rm -f perf.data*

# Show current disk usage by target/ subdirectories
disk-usage:
	@echo "Build artifact disk usage:"
	@du -sh target/ 2>/dev/null || echo "No target/ directory"
	@echo ""
	@echo "Breakdown:"
	@du -sh target/*/ 2>/dev/null | sort -hr || echo "No subdirectories"

# Install profiling tools. (cargo-criterion is not among them: it re-enters
# `$CARGO bench` without a lane, which targo refuses.)
install-tools:
	@echo "Installing profiling tools..."
	targo --unverified install samply
	targo --unverified install flamegraph
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
	@echo "  make test              - Test default workspace members (release)"
	@echo "  make test-verbose      - Run Rust tests with output"
	@echo "  make test-python       - Build ny-python and run Python tests"
	@echo "  make test-python-plugin - Run pytest plugin tests (no build)"
	@echo "  make test-python-tooling - Run repository Python/shell tooling tests"
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
	@echo "  make install-tools - Install samply, flamegraph"
	@echo "  make gate         - First-party drift gate: Rust/Python hygiene + test slices + submission invariants"
	@echo "  make gate-fast    - Drift checks a-h: Rust/Python hygiene, focused tests and packaging"
	@echo "  make lint         - Rust formatting + workspace Tippy gate"
	@echo "  make lint-tippy   - Trust toolchain tippy gate (set TRUST_STAGE2 if needed)"
	@echo "  make fmt          - Check formatting"
	@echo "  make deny         - Advisory, dependency/source, and license gate (cargo-deny)"
