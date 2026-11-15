# Benchmarks

Measurement scripts and cross-tool diagnostics for the `ny` CLI. They are not
correctness tests: correctness belongs in the Cargo test suite, and pytest does
not collect these files by default (`testpaths` in `pytest.ini`). Commit
scripts, input manifests, and small fixtures here, never benchmark output.

`compare_crown_simple.py` and the pytest diagnostics run `target/release/ny`.
Build it first:

```bash
targo --unverified build --release -p ny-cli
```

Each diagnostic exits nonzero if anything it needs (the release binary, its
corpus, or its third-party comparator) is missing.

## Cross-tool comparisons

```bash
python benchmarks/compare_crown_simple.py    # ny vs Auto-LiRPA bounds on 2-3 layer networks
python benchmarks/probe_autolirpa_acasxu.py  # Auto-LiRPA only, on tests/models/acasxu_1_1.nnet
python benchmarks/compare_attention.py       # Auto-LiRPA attention IBP timing; does not run ny
```

## VNN-COMP corpora

`benchmarks/download_benchmarks.sh [years...]` (also `ny benchmarks download`)
clones the corpora into the gitignored `benchmarks/vnncompYYYY/`. The default
years are 2021 and 2023-2026.

```bash
benchmarks/download_benchmarks.sh
pytest benchmarks/acasxu_diagnostic.py -v --ny-benchmark-timeout=10
pytest benchmarks/vnncomp_diagnostic.py -v --ny-benchmark-method=beta
```

`benchmarks/conftest.py` defines `--ny-benchmark-timeout` (seconds, default
10), `--ny-benchmark-method` (`ibp`, `crown`, `alpha`, or `beta`; default
`crown`), and `--ny-benchmark-results` (JSON output path).

For full sweeps at the official per-instance budgets, use `ny benchmarks run`
(see the top-level README).

## Experiment planners

These validate their pinned inputs and print an experiment plan as JSON. They
never run a solver or start a GPU job.

- `scripts/plan_biccos_mts_factorial.py` plans the BICCOS multi-tree branching
  factorial against clean alpha-beta-CROWN and auto_LiRPA checkouts at the
  commits pinned in `biccos_mts_factorial_v1.json`.
- `scripts/plan_compact_tail_envelope.py` plans the compact-tail envelope
  experiment for one CIFAR100 property (`compact_tail_envelope_v1.json`). It
  requires exact hashes for the model, the property, and every evidence file.
  Its output is diagnostic only: the compact K16 executor stays off unless
  both `NY_IMB_TAIL_CERT_AY=1` and `NY_COMPACT_TAIL_K16=1` are set, and no
  planner record can authorize a verdict.

`abcrown_transfer_corpus_v1.json` and `abcrown_transfer_dispositions_v1.json`
fix the alpha-beta-CROWN transfer corpus and the recorded disposition of each
transferred technique.
