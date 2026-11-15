# Configs

Checked-in VNN-COMP presets, the waiver, policy, and baseline files that checks
read, and experiment arms.

- `vnncomp24/`, `vnncomp25/`, `vnncomp26/`: per-category preset yamls.
  `ny vnncomp` searches them newest year first and, in each, tries the
  lower-cased category name before the name with its `_20NN` suffix removed
  (`resolve_preset_path` in `crates/ny-cli/src/commands/vnncomp.rs`). A
  `vnncomp26` yaml therefore shadows the same category in `vnncomp25`: run
  `ls configs/vnncomp26/` before editing an older preset. `ny benchmarks run
  --year Y` skips directories newer than Y.
- `backend_capability_waivers.yaml`, `model_load_waivers.yaml`,
  `preset_score_at_risk.yaml`: shipped presets whose backend or model this
  build cannot use, and the banked score each category puts at risk. Read by
  `preset::backend_capability_tests` and `preset::model_load_smoke_tests`
  in `ny-cli`, which `scripts/gate.sh` runs.
- `architecture_layers.toml`: crate layering policy, checked by
  `scripts/check_architecture_layers.py`.
- `benchmark_regressions/gpu_crown_backward.json`: baselines for
  `scripts/check_gpu_crown_backward_regression.py`.
- `experiments/`: experiment arms, not production presets.

Presets must be deterministic and reviewable. Do not commit local machine
paths, report output, credentials, or generated run logs.
