// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! STATIC capability guard: every model a shipped preset points at must still
//! LOAD, and every failure must name the BANKED SCORE it deletes.
//!
//! WHY: EIGHT separate commits have now silently zeroed a banked category by
//! refusing to load its model — 25dee0c5 (cast dtype), 38a2fecf (external
//! tensors), 1ede1d30 (wgpu quarantine), 44d9e46d (witness f32), 819b0554
//! (opset/Dropout), the opset-13 Softmax preflight (vit), the `Equal` dtype-7
//! refusal (cctsdb), and the opset-8 Dropout refusal (vggnet16). Each was
//! individually CORRECT about ONNX or float semantics. None had its CAPABILITY
//! COST measured before it landed. `cctsdb_yolo_2023` had been banked at 100.0
//! normalized four days before the first of them and scored 0 for a week.
//!
//! AND THE GUARD ITSELF GOT ROUTED AROUND. This test was red at plain
//! origin/main and every lane classified it "inherited" and moved on. So three
//! things are true of it now that were not before:
//!
//!   1. A failure states the SCORE AT RISK — the category's banked normalized,
//!      read from `configs/preset_score_at_risk.yaml`. "A test is red" was
//!      ignored eight times; "this deletes 100.0 banked points" reads
//!      differently.
//!   2. ABSENT BENCHMARK DATA IS A HARD FAILURE, not a silent pass. The old
//!      code returned `Ok` when nothing was checked, so a bare worktree — no
//!      `benchmarks/` — reported PASS while enforcing exactly nothing. That is
//!      how a bare-worktree control talked a lane into calling a REAL failure
//!      "inherited". The real-model guard therefore lives in the explicit
//!      `external-vnncomp` lane, where absent data is an actionable failure.
//!   3. A preset whose benchmark root has no recorded stake fails
//!      `every_shipped_preset_declares_its_banked_score_at_risk`, which needs no
//!      benchmark data at all — so a new preset cannot enter the repo without
//!      declaring what it is worth.
//!
//! This is deliberately a LOAD check, not a verification run: no budget, no
//! verdict, no GPU, no ORT. It exercises exactly the admission path a
//! fail-closed loader gate lands on.
//!
//! `NY_PRESET_LOAD_SMOKE=all` loads every distinct model in every preset's
//! `instances.csv` instead of the default one-per-architecture-family sample.
//! Measured 2026-07-29 with the 2024/2025/2026 benchmark repositories checked
//! out: family-sample 93 model files in 25 s, `all` 541 model files in 63 s.
//! The default hermetic suite still checks the preset/stake registries; it does
//! not claim that external models were loaded.

#[cfg(feature = "external-vnncomp")]
use super::build_onnx_load_config;
use super::load_preset;
#[cfg(feature = "external-vnncomp")]
use ny_onnx::OnnxLoadConfig;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    normalize_lexically(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."))
}

/// Resolve `.`/`..` WITHOUT touching the filesystem.
///
/// Needed because a preset's `root_path` is relative to its own directory and
/// climbs out with `../..`; the staleness check below has to compare the
/// resulting location against the repo layout even when the directory does not
/// exist (which is precisely the stale case).
fn normalize_lexically(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::ParentDir => {
                out.pop();
            }
            std::path::Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Collapse digit runs so `patch-1.onnx`/`patch-3.onnx` and
/// `tllBench_n=2_..._instance_1_0.onnx`/`..._1_3.onnx` share one signature.
///
/// Loader admission gates trip on OPS and DTYPES, which are properties of a
/// model's architecture family rather than of its index, so one representative
/// per family is the information-bearing sample. The representative is the
/// SMALLEST file in its family, which keeps a 100-model / 1.8 GB category
/// (metaroom) at two loads.
fn family_signature(model: &str) -> String {
    let mut out = String::with_capacity(model.len());
    let mut in_digits = false;
    for ch in model.chars() {
        if ch.is_ascii_digit() {
            if !in_digits {
                out.push('#');
                in_digits = true;
            }
        } else {
            in_digits = false;
            out.push(ch);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// SCORE AT RISK — configs/preset_score_at_risk.yaml
// ---------------------------------------------------------------------------

/// What one benchmark root is WORTH if its models stop loading.
#[derive(Debug, Clone, serde::Deserialize)]
struct RootStake {
    /// Repo-relative benchmark root, exactly as a preset's `root_path` resolves.
    root: String,
    /// Scorecard row these instances are banked under.
    benchmark: String,
    /// `regular` | `extended` | `unscored`.
    track: String,
    /// Normalized points the repo loses if every instance returns `unknown`.
    banked_normalized: f64,
    #[serde(default)]
    #[allow(dead_code)]
    note: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
struct ScoreAtRiskFile {
    measured_on: String,
    measured_at_sha: String,
    roots: Vec<RootStake>,
}

fn score_at_risk() -> BTreeMap<String, RootStake> {
    let path = repo_root().join("configs/preset_score_at_risk.yaml");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|err| {
        panic!(
            "the score-at-risk registry must exist at {}: {err}\n\nIt is what lets a model-load \
             failure say how many BANKED POINTS it deletes instead of just saying a test is red.",
            path.display()
        )
    });
    let parsed: ScoreAtRiskFile = serde_yaml::from_str(&text)
        .unwrap_or_else(|err| panic!("{} must be valid YAML: {err}", path.display()));
    assert!(
        !parsed.measured_on.trim().is_empty() && !parsed.measured_at_sha.trim().is_empty(),
        "{} must record WHEN and at WHICH SHA the stakes were read; an undated stake is a guess",
        path.display()
    );
    let mut out = BTreeMap::new();
    for stake in parsed.roots {
        assert!(
            matches!(stake.track.as_str(), "regular" | "extended" | "unscored"),
            "{}: track must be regular|extended|unscored, got '{}'",
            stake.root,
            stake.track
        );
        assert!(
            stake.banked_normalized >= 0.0 && stake.banked_normalized <= 100.0,
            "{}: banked_normalized is a per-benchmark normalized score in [0, 100], got {}",
            stake.root,
            stake.banked_normalized
        );
        assert!(
            out.insert(stake.root.clone(), stake.clone()).is_none(),
            "{}: duplicate root entry in the score-at-risk registry",
            stake.root
        );
    }
    out
}

/// Render one root's stake for a failure message.
fn stake_label(root_rel: &str, stakes: &BTreeMap<String, RootStake>) -> String {
    match stakes.get(root_rel) {
        Some(stake) => format!(
            "SCORE AT RISK {:.1} banked normalized ({}, {} track)",
            stake.banked_normalized, stake.benchmark, stake.track
        ),
        None => format!(
            "SCORE AT RISK UNRECORDED (no entry for {root_rel} in \
             configs/preset_score_at_risk.yaml)"
        ),
    }
}

// ---------------------------------------------------------------------------
// Shipped-preset inventory
// ---------------------------------------------------------------------------

/// A shipped preset that pins a benchmark root. Data-free: nothing here needs
/// the benchmark repositories to be checked out.
struct PresetRoot {
    /// Repo-relative preset path, for messages.
    preset: String,
    /// Absolute benchmark root.
    #[cfg(feature = "external-vnncomp")]
    root: PathBuf,
    /// The same root, repo-relative — the key into the score-at-risk registry.
    root_rel: String,
    /// The `benchmarks/vnncompYYYY` data root this preset needs on disk.
    #[cfg(feature = "external-vnncomp")]
    data_root: PathBuf,
    /// Parsed preset (kept so the model inventory can reuse it).
    #[cfg(feature = "external-vnncomp")]
    config: super::PresetConfig,
}

/// Every shipped preset, parsed, WITHOUT touching benchmark data.
///
/// Returns the roots plus the labels of presets that pin no root at all (the
/// `*_local.yaml` solver-only presets), which are legitimately exempt.
fn preset_roots() -> (Vec<PresetRoot>, Vec<String>) {
    let root = repo_root();
    let configs = root.join("configs");
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(&configs)
        .expect("configs/ must exist")
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_dir()
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("vnncomp"))
        })
        .collect();
    dirs.sort();

    let mut out = Vec::new();
    let mut rootless = Vec::new();
    for dir in dirs {
        let mut yamls: Vec<PathBuf> = std::fs::read_dir(&dir)
            .expect("preset dir readable")
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.extension().and_then(|e| e.to_str()) == Some("yaml"))
            .collect();
        yamls.sort();
        for yaml in yamls {
            let label = yaml
                .strip_prefix(&root)
                .unwrap_or(&yaml)
                .to_string_lossy()
                .to_string();
            let preset = load_preset(&yaml)
                .unwrap_or_else(|err| panic!("shipped preset {label} must parse: {err}"));
            let Some(root_path) = preset.general.root_path.as_ref() else {
                rootless.push(format!(
                    "{label}: no general.root_path (solver-only preset)"
                ));
                continue;
            };
            let bench_root = normalize_lexically(&dir.join(root_path));
            // STALENESS, checkable without any data: the repo's benchmark
            // checkouts live under `benchmarks/vnncompYYYY/` — that is what
            // benchmarks/download_benchmarks.sh populates. A root_path resolving
            // anywhere else can never be satisfied by a download, so every
            // instance of that preset reports a missing input forever.
            let benchmarks_dir = root.join("benchmarks");
            assert!(
                bench_root.starts_with(&benchmarks_dir),
                "PRESET ROOT PATH IS STALE — {label} declares general.root_path '{}', which \
                 resolves to {} instead of somewhere under {}. No benchmark download can ever \
                 satisfy that path, so every instance of this preset reports a missing input.",
                root_path.display(),
                bench_root.display(),
                benchmarks_dir.display()
            );
            // `benchmarks/vnncompYYYY/...` -> the data root is the `vnncomp*`
            // component.
            #[cfg(feature = "external-vnncomp")]
            let data_root = bench_root
                .ancestors()
                .find(|ancestor| {
                    ancestor
                        .file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| name.starts_with("vnncomp"))
                })
                .map(Path::to_path_buf)
                .unwrap_or_else(|| benchmarks_dir.clone());
            // Separators normalized to `/`: this is the KEY into
            // configs/preset_score_at_risk.yaml, whose entries are POSIX
            // literals like `benchmarks/vnncomp2025`. The host separator would
            // miss every one of them on Windows, so the guard reported that
            // every shipped preset lacked a recorded stake.
            let root_rel = bench_root
                .strip_prefix(&root)
                .unwrap_or(&bench_root)
                .to_string_lossy()
                .replace('\\', "/");
            out.push(PresetRoot {
                preset: label,
                #[cfg(feature = "external-vnncomp")]
                root: bench_root,
                root_rel,
                #[cfg(feature = "external-vnncomp")]
                data_root,
                #[cfg(feature = "external-vnncomp")]
                config: preset,
            });
        }
    }
    (out, rootless)
}

#[cfg(feature = "external-vnncomp")]
struct PresetInventory {
    preset: String,
    root: PathBuf,
    root_rel: String,
    /// Distinct model paths, relative to `root`.
    models: Vec<String>,
    /// The preset's own ONNX load configuration, so the smoke test exercises the
    /// same admission path a scored run does (a preset's
    /// `model.onnx_optimization_flags` change what the loader admits).
    load_config: OnnxLoadConfig,
    /// Canonical spelling of `model.onnx_optimization_flags`. Two presets that
    /// share a model AND these flags share one load; differing flags are loaded
    /// separately, because that is a different admission path.
    flag_key: String,
}

/// Parse the ONNX column of an `instances.csv` (`onnx,vnnlib,timeout`).
#[cfg(feature = "external-vnncomp")]
fn instance_models(csv_path: &Path) -> Vec<String> {
    let text = std::fs::read_to_string(csv_path)
        .unwrap_or_else(|err| panic!("instances.csv unreadable at {}: {err}", csv_path.display()));
    let mut seen = BTreeSet::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let Some(first) = trimmed.split(',').next() else {
            continue;
        };
        let model = first.trim().trim_start_matches("./");
        if !model.is_empty() {
            seen.insert(model.to_string());
        }
    }
    seen.into_iter().collect()
}

/// Is this `benchmarks/vnncompYYYY` root an ACTUAL checkout of that year's
/// benchmark repository?
///
/// `benchmarks/download_benchmarks.sh` populates a year root by cloning the
/// year's benchmark repository into it, so a real checkout carries that
/// repository's own top-level files: a `.git`, and the `setup.sh` every
/// VNN-COMP benchmark repository ships to fetch its large models.
///
/// "The directory exists" is NOT the same question, and answering it instead is
/// what made this guard silently stop guarding. A box that has hand-staged ONE
/// category (`benchmarks/vnncomp2025/benchmarks/cifar100_2024` symlinked in,
/// nothing else downloaded) HAS a `benchmarks/vnncomp2025/` directory and has
/// no repository. Reading that as a checkout turns "this box has no data" into
/// a false "PRESET POINTS AT A MISSING CATEGORY" panic on the first 2025
/// preset, which aborts the whole guard: the remaining presets are never
/// inventoried, the absent-data summary that prices the WHOLE stake never
/// prints, and not one model is ever loaded.
#[cfg(feature = "external-vnncomp")]
fn is_benchmark_repo_checkout(data_root: &Path) -> bool {
    data_root.join(".git").exists() || data_root.join("setup.sh").is_file()
}

/// One preset the guard could NOT check because its benchmark data is absent.
#[cfg(feature = "external-vnncomp")]
struct AbsentData {
    preset: String,
    root_rel: String,
    data_root: PathBuf,
}

/// Every shipped preset's model inventory, plus the presets whose benchmark data
/// is NOT on this box.
///
/// The second return value is the reason this function no longer returns a
/// silent empty vector: absent data is reported to the caller so it can FAIL,
/// instead of being folded into "nothing to check".
#[cfg(feature = "external-vnncomp")]
fn inventories(stakes: &BTreeMap<String, RootStake>) -> (Vec<PresetInventory>, Vec<AbsentData>) {
    let (roots, _rootless) = preset_roots();
    let mut out = Vec::new();
    let mut absent = Vec::new();
    for entry in roots {
        let PresetRoot {
            preset,
            root,
            root_rel,
            data_root,
            config,
        } = entry;
        if !is_benchmark_repo_checkout(&data_root) {
            absent.push(AbsentData {
                preset,
                root_rel,
                data_root,
            });
            continue;
        }
        assert!(
            root.is_dir(),
            "PRESET POINTS AT A MISSING CATEGORY — {preset} declares general.root_path {} but \
             that directory does not exist while its benchmark repository IS checked out. Every \
             instance of this preset would report a missing input. {}",
            root.display(),
            stake_label(&root_rel, stakes)
        );
        let csv_name = config
            .general
            .csv_name
            .as_deref()
            .unwrap_or("instances.csv");
        let csv_path = root.join(csv_name);
        assert!(
            csv_path.is_file(),
            "PRESET POINTS AT A MISSING INSTANCE LIST — {preset} declares general.csv_name \
             '{csv_name}' under {}, which does not exist. {}",
            root.display(),
            stake_label(&root_rel, stakes)
        );
        let load_config = build_onnx_load_config(&config).unwrap_or_else(|err| {
            panic!("{preset}: model.onnx_optimization_flags must be valid: {err}")
        });
        let mut flags = config.model.onnx_optimization_flags.clone();
        flags.sort();
        out.push(PresetInventory {
            preset,
            models: instance_models(&csv_path),
            root,
            root_rel,
            load_config,
            flag_key: flags.join("+"),
        });
    }
    (out, absent)
}

/// Classify a loader failure so the report names the CAUSE, not just the diff.
#[cfg(feature = "external-vnncomp")]
fn classify_load_failure(message: &str) -> &'static str {
    let lower = message.to_ascii_lowercase();
    if lower.contains("no such file") || lower.contains("not found") {
        "MISSING-MODEL-FILE"
    } else if lower.contains("cast") && lower.contains("dtype") {
        "LOADER-GATE(cast-dtype)"
    } else if lower.contains("dtype") || lower.contains("float32") {
        "LOADER-GATE(dtype)"
    } else if lower.contains("opset") {
        "LOADER-GATE(opset)"
    } else if lower.contains("unsupported") || lower.contains("not supported") {
        "LOADER-GATE(unsupported-op)"
    } else if lower.contains("shape") || lower.contains("dimension") {
        "SHAPE-INFERENCE"
    } else {
        "MODEL-LOAD-FAILURE"
    }
}

/// One recorded, dated model-load debt from `configs/model_load_waivers.yaml`.
#[derive(Debug, serde::Deserialize)]
#[cfg(feature = "external-vnncomp")]
struct LoadWaiver {
    /// Repo-relative model path.
    model: String,
    /// Substring the loader error must contain, so a waiver cannot silently
    /// absorb a DIFFERENT failure of the same model.
    error_contains: String,
    gate: String,
    since: String,
    cost: String,
}

#[derive(Debug, serde::Deserialize)]
#[cfg(feature = "external-vnncomp")]
struct LoadWaiverFile {
    waivers: Vec<LoadWaiver>,
}

#[cfg(feature = "external-vnncomp")]
fn load_waivers() -> Vec<LoadWaiver> {
    let path = repo_root().join("configs/model_load_waivers.yaml");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|err| {
        panic!(
            "the model-load waiver registry must exist at {}: {err}",
            path.display()
        )
    });
    let parsed: LoadWaiverFile = serde_yaml::from_str(&text)
        .unwrap_or_else(|err| panic!("{} must be valid YAML: {err}", path.display()));
    for waiver in &parsed.waivers {
        assert!(
            !waiver.error_contains.trim().is_empty(),
            "{}: a waiver must pin the loader error it covers",
            waiver.model
        );
        assert!(
            !waiver.gate.trim().is_empty() && !waiver.since.trim().is_empty(),
            "{}: a waiver must record the gate commit and date",
            waiver.model
        );
        // An unmeasured cost is the exact defect this registry exists to
        // prevent, so refuse the placeholder outright.
        let cost = waiver.cost.trim();
        assert!(
            !cost.is_empty() && !cost.eq_ignore_ascii_case("unmeasured"),
            "{}: a waiver must state the MEASURED cost in instances and banked rows",
            waiver.model
        );
        // A waiver is a DEBT, and a debt has a size. Naming the banked score is
        // the whole point: eight gates landed because "a category stopped
        // loading" never got quoted as a number.
        assert!(
            cost.contains("normalized") || cost.contains("normalised"),
            "{}: a waiver's `cost` must quote the BANKED NORMALIZED it forfeits (the number from \
             configs/preset_score_at_risk.yaml), not just an instance count — that number is what \
             makes the debt legible. Got: {cost}",
            waiver.model
        );
    }
    parsed.waivers
}

/// Sum the distinct benchmarks named by a set of at-risk roots.
///
/// Distinct BENCHMARK, not distinct root: three roots share `ml4acopf_2024`, and
/// counting its 68.3 three times would inflate the headline number the message
/// exists to make trustworthy.
fn total_at_risk(roots: &BTreeSet<String>, stakes: &BTreeMap<String, RootStake>) -> (f64, usize) {
    let mut by_benchmark: BTreeMap<&str, f64> = BTreeMap::new();
    for root in roots {
        if let Some(stake) = stakes.get(root) {
            by_benchmark.insert(&stake.benchmark, stake.banked_normalized);
        }
    }
    (by_benchmark.values().sum(), by_benchmark.len())
}

// ---------------------------------------------------------------------------
// THE GUARDS
// ---------------------------------------------------------------------------

/// DATA-FREE: every shipped preset's benchmark root declares what it is worth.
///
/// This one needs no benchmark repository, so it cannot be made vacuous by an
/// empty worktree. A preset added without a stake entry fails here, which means
/// the load guard can never again report a failure it has no price for.
#[test]
fn every_shipped_preset_declares_its_banked_score_at_risk() {
    let stakes = score_at_risk();
    let (roots, rootless) = preset_roots();
    assert!(
        !roots.is_empty(),
        "no shipped preset pins a benchmark root — configs/vnncomp*/ is empty or unreadable, and \
         this guard would be enforcing nothing"
    );

    let mut undeclared: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for entry in &roots {
        if !stakes.contains_key(&entry.root_rel) {
            undeclared
                .entry(entry.root_rel.clone())
                .or_default()
                .push(entry.preset.clone());
        }
    }
    assert!(
        undeclared.is_empty(),
        "SHIPPED PRESET WITH NO RECORDED STAKE — {} benchmark root(s) have no entry in \
         configs/preset_score_at_risk.yaml, so a model-load refusal against them could only ever \
         report \"a test is red\". Add each one with its banked normalized (the scorecard's \
         ny_norm / meas_norm column) before shipping the preset:\n  {}",
        undeclared.len(),
        undeclared
            .iter()
            .map(|(root, presets)| format!("{root}\n      referenced by: {}", presets.join(", ")))
            .collect::<Vec<_>>()
            .join("\n  ")
    );

    // The reverse direction: a stake entry whose root no preset pins any more is
    // a stale claim about what this repo is worth.
    let live: BTreeSet<&str> = roots.iter().map(|entry| entry.root_rel.as_str()).collect();
    let orphans: Vec<&str> = stakes
        .keys()
        .map(String::as_str)
        .filter(|root| !live.contains(root))
        .collect();
    assert!(
        orphans.is_empty(),
        "STALE SCORE-AT-RISK ENTRY — no shipped preset pins these roots any more; delete them \
         from configs/preset_score_at_risk.yaml so the registry keeps describing the repo that \
         exists:\n  {}",
        orphans.join("\n  ")
    );

    let (total, benchmarks) = total_at_risk(&live.iter().map(|s| s.to_string()).collect(), &stakes);
    eprintln!(
        "preset stake registry: {} preset(s) pinning {} distinct root(s) across {benchmarks} \
         benchmark(s), {total:.1} banked normalized under guard ({} solver-only preset(s) exempt)",
        roots.len(),
        live.len(),
        rootless.len()
    );
}

/// THE GUARD. Every model a shipped preset references still loads at HEAD.
#[cfg(feature = "external-vnncomp")]
#[test]
fn every_shipped_preset_model_still_loads() {
    let mode = std::env::var("NY_PRESET_LOAD_SMOKE").unwrap_or_default();
    let load_all = mode.eq_ignore_ascii_case("all");
    assert!(
        mode.is_empty() || load_all,
        "invalid NY_PRESET_LOAD_SMOKE={mode:?}; the external-vnncomp conformance lane cannot be \
         disabled at runtime. Leave it unset for one model per architecture family, or set \
         NY_PRESET_LOAD_SMOKE=all for the complete inventory"
    );

    let stakes = score_at_risk();
    let (inventories, absent) = inventories(&stakes);

    // ABSENT DATA IS A HARD FAILURE.
    //
    // The old code printed a note and returned Ok, so a worktree with no
    // `benchmarks/` reported PASS while checking zero models. That vacuous green
    // is how a bare-worktree control convinced a lane that a REAL load failure
    // was "inherited". Absent data now fails, loudly, with the stake it is
    // failing to protect.
    let absent_roots: BTreeSet<String> =
        absent.iter().map(|entry| entry.root_rel.clone()).collect();
    let (absent_total, absent_benchmarks) = total_at_risk(&absent_roots, &stakes);
    let absent_report: Vec<String> = absent
        .iter()
        .map(|entry| {
            format!(
                "{}\n      root:    {}\n      missing: {}\n      {}",
                entry.preset,
                entry.root_rel,
                entry.data_root.display(),
                stake_label(&entry.root_rel, &stakes)
            )
        })
        .collect();
    assert!(
        absent.is_empty(),
        "BENCHMARK DATA IS NOT CHECKED OUT — this guard verified NOTHING for {} preset(s) \
         covering {absent_benchmarks} benchmark(s) worth {absent_total:.1} banked normalized. A \
         pass here would be VACUOUS, and a vacuous pass is exactly how a real load failure gets \
         mistaken for an inherited one.\n\nFix it by making the data reachable — \
         benchmarks/download_benchmarks.sh, or in a worktree symlink the year roots from your \
         main checkout:\n    ln -sfn <main-checkout>/benchmarks/vnncomp2025 \
         <worktree>/benchmarks/vnncomp2025\n\nThe default hermetic suite can run without the \
         external-vnncomp feature, but a selected external conformance lane never turns missing \
         data into a pass.\n\n  {}",
        absent.len(),
        absent_report.join("\n  ")
    );
    assert!(
        !inventories.is_empty(),
        "no shipped preset produced a model inventory — the guard would enforce nothing"
    );

    // Load each distinct (model file, optimization-flag set) ONCE even when
    // several presets share it (canary presets deliberately reuse their
    // category's root).
    // `None` config = load with loader defaults (used for the waived-model debt
    // list, which is not tied to any one preset's optimization flags).
    #[allow(clippy::type_complexity)]
    let mut planned: BTreeMap<
        (PathBuf, String),
        (Option<&OnnxLoadConfig>, Vec<String>, String),
    > = BTreeMap::new();
    let mut checked_presets = 0usize;
    for inventory in &inventories {
        checked_presets += 1;
        let mut representatives: BTreeMap<String, (PathBuf, u64)> = BTreeMap::new();
        for model in &inventory.models {
            let path = inventory.root.join(model);
            let size = path.metadata().map(|meta| meta.len()).unwrap_or(u64::MAX);
            if load_all {
                representatives.insert(model.clone(), (path, size));
                continue;
            }
            let signature = family_signature(model);
            match representatives.get(&signature) {
                Some((_, best)) if *best <= size => {}
                _ => {
                    representatives.insert(signature, (path, size));
                }
            }
        }
        for (path, _) in representatives.into_values() {
            planned
                .entry((path, inventory.flag_key.clone()))
                .or_insert_with(|| {
                    (
                        Some(&inventory.load_config),
                        Vec::new(),
                        inventory.root_rel.clone(),
                    )
                })
                .1
                .push(inventory.preset.clone());
        }
    }

    // Every WAIVED model is loaded unconditionally, even when the family sample
    // would have skipped it: the waiver list is the debt ledger, and a debt that
    // silently disappears from the sample can never be shown to be paid off.
    let waivers = load_waivers();
    let root = repo_root();
    let sampled: BTreeSet<PathBuf> = planned.keys().map(|(path, _)| path.clone()).collect();
    for waiver in &waivers {
        let path = root.join(&waiver.model);
        if sampled.contains(&path) {
            // Already sampled: a preset entry covers it, with that preset's own
            // flags.
            continue;
        }
        assert!(
            path.exists(),
            "WAIVED MODEL IS NOT ON DISK — configs/model_load_waivers.yaml claims a debt against \
             {}, which does not exist. A debt that cannot be re-tested can never be shown to be \
             paid off; fix the path or delete the waiver.",
            path.display()
        );
        planned.insert(
            (path, "<waived>".to_string()),
            (None, vec!["<model-load waiver>".to_string()], String::new()),
        );
    }

    let mut failures = Vec::new();
    let mut failed_roots: BTreeSet<String> = BTreeSet::new();
    let mut recorded = BTreeSet::new();
    let mut stale = BTreeSet::new();
    for ((path, _), (config, presets, root_rel)) in &planned {
        let relative = path.strip_prefix(&root).unwrap_or(path).to_string_lossy();
        let waiver = waivers
            .iter()
            .find(|waiver| waiver.model.as_str() == relative.as_ref());
        let loaded = match *config {
            Some(config) => ny_onnx::load_onnx_with_config(path, config),
            None => ny_onnx::load_onnx(path),
        };
        match loaded {
            Ok(_) => {
                if let Some(waiver) = waiver {
                    stale.insert(format!(
                        "{} (waived for {} since {})",
                        waiver.model, waiver.gate, waiver.since
                    ));
                }
            }
            Err(err) => {
                let message = err.to_string();
                let entry = format!(
                    "{} [{}]\n      model:   {}\n      {}\n      error:   {message}",
                    presets.join(", "),
                    classify_load_failure(&message),
                    path.display(),
                    stake_label(root_rel, &stakes)
                );
                match waiver {
                    Some(waiver) if message.contains(&waiver.error_contains) => {
                        recorded.insert(format!(
                            "{} [gate {} since {}]\n      {}",
                            waiver.model,
                            waiver.gate,
                            waiver.since,
                            waiver.cost.trim()
                        ));
                    }
                    _ => {
                        if !root_rel.is_empty() {
                            failed_roots.insert(root_rel.clone());
                        }
                        failures.push(entry);
                    }
                }
            }
        }
    }

    let (lost, lost_benchmarks) = total_at_risk(&failed_roots, &stakes);
    assert!(
        failures.is_empty(),
        "SHIPPED PRESET MODELS NO LONGER LOAD — THIS PUTS {lost:.1} BANKED NORMALIZED POINTS AT \
         RISK across {lost_benchmarks} benchmark(s) (the FULL banked value of every benchmark \
         with an unloadable model; the realised loss is the subset of rows that use these \
         models, and you owe that number too).\n\n{} of {} sampled model file(s) across \
         {checked_presets} preset(s) fail the loader's admission path and are NOT recorded in \
         configs/model_load_waivers.yaml. A load refusal is a CAPABILITY REGRESSION, not a hard \
         instance: every one of these categories now scores ZERO no matter how good the verifier \
         is.\n  {}\n\nIf a fail-closed loader gate is responsible, the gate may well be RIGHT — \
         but its capability cost must be MEASURED and recorded BEFORE it lands. Eight gates have \
         now shipped without paying that. Either land the narrow admission, or add a dated waiver \
         to configs/model_load_waivers.yaml quoting the banked normalized above as its `cost`.",
        failures.len(),
        planned.len(),
        failures.join("\n  ")
    );
    assert!(
        stale.is_empty(),
        "STALE MODEL-LOAD WAIVER — these models LOAD again, so the recorded debt is paid. Delete \
         the entries from configs/model_load_waivers.yaml and RE-SWEEP the ledger rows each one \
         was covering; they are currently banked as zeros:\n  {}",
        stale.iter().cloned().collect::<Vec<_>>().join("\n  ")
    );

    if !recorded.is_empty() {
        eprintln!(
            "preset load smoke: {} RECORDED model-load debt(s) — these categories score 0 at \
             HEAD:",
            recorded.len()
        );
        for entry in &recorded {
            eprintln!("  {entry}");
        }
    }
    eprintln!(
        "preset load smoke: {} model file(s) checked across {checked_presets} preset(s) (mode={})",
        planned.len(),
        if load_all { "all" } else { "family-sample" }
    );
}

#[test]
fn family_signature_collapses_indices_not_architectures() {
    assert_eq!(family_signature("onnx/patch-1.onnx"), "onnx/patch-#.onnx");
    assert_eq!(
        family_signature("onnx/patch-1.onnx"),
        family_signature("onnx/patch-3.onnx")
    );
    assert_eq!(
        family_signature("onnx/tllBench_n=2_N=M=16_m=1_instance_1_0.onnx"),
        family_signature("onnx/tllBench_n=2_N=M=16_m=1_instance_1_3.onnx")
    );
    // Distinct architectures must NOT collapse.
    assert_ne!(
        family_signature("onnx/cifar_biasfield_vnncomp2022_cifar_bias_field_0.onnx"),
        family_signature(
            "onnx/cifar_biasfield_vnncomp2022_cifar_bias_field_0_RSPLITTER_cifar_biasfield_vnncomp2022_prop_0.onnx"
        )
    );
}

/// A stake with no price is the defect this registry exists to prevent, so the
/// renderer must SAY so rather than printing a blank.
#[test]
fn stake_label_names_the_gap_when_a_root_is_unrecorded() {
    let stakes = score_at_risk();
    let known = stakes
        .keys()
        .next()
        .expect("the stake registry must not be empty")
        .clone();
    let rendered = stake_label(&known, &stakes);
    assert!(
        rendered.contains("SCORE AT RISK") && rendered.contains("banked normalized"),
        "a recorded stake must render its banked normalized: {rendered}"
    );
    let missing = stake_label("benchmarks/vnncomp2099/benchmarks/nope", &stakes);
    assert!(
        missing.contains("UNRECORDED"),
        "an unrecorded stake must say so out loud: {missing}"
    );
}
