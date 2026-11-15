// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `#wave-beta-bab` (ENGINE_PROGRAM M6) at the CLI boundary, with the REAL
//! ny-bbe CPU f32 wave search registered exactly as `main.rs` registers it:
//!
//! * `wave_tree_mode_is_dark_without_the_keys_and_typed_with_them` — the typed
//!   key `margin_row.tree_mode` and its two knobs resolve to their declaration
//!   defaults (`legacy`, 256, 0) without the keys and arrive through the scored
//!   run's own preset projection with `Source::Config`, never
//!   `Source::LegacyEnv`;
//! * `wave_policy_is_resolved_from_the_receipts_lever_set` — ONE authority:
//!   the scored lane's `MarginRowRuntimePolicy::wave` is resolved from the same
//!   projected LeverSet the flight receipt serialises (no governed-cell read);
//! * `registered_cpu_f32_backend_reaches_the_armed_wave_tree` — armed, the
//!   lane runs the wave engine (its own `[bbe] phase=wave ... wave-beta-bab/v1`
//!   receipt, no `fallback=legacy`) and the backend's search engaged;
//! * `real_cpu_f32_wave_proves_robust_tiny_nets` — on synthetic twin-wall nets
//!   the legacy tree proves only through a tree, the real search + the lane's
//!   certify queue prove too (positive control), with a publishable token;
//! * `real_cpu_f32_wave_never_closes_a_sat_instance` — the orientation swap
//!   (violated at the box midpoint) is never `Unsat`;
//! * `real_cpu_f32_wave_is_thread_count_independent` — certify-pool widths 1
//!   and 4 give the same verdict and tree statistics.

use std::sync::Arc;
use std::time::{Duration, Instant};

use ny_propagate::margin_row::bab::wave::{TreeMode, WavePolicy};
use ny_propagate::margin_row::{
    BabConfig, MarginRowBab, MarginRowOutcome, RootGates, RoundMode, TwinNet, TwinOpSpec, TwinSpec,
};

use super::preset_delivered_config_value;

fn projected_levers(preset: &crate::preset::PresetConfig) -> ny_levers::LeverSet {
    let serialized = serde_json::to_value(preset).expect("preset serializes");
    let captured = ny_levers::RawLeverInputs::capture_with(ny_levers::all(), |_| None);
    ny_levers::LeverSet::resolve_layered(ny_levers::all(), &captured, |decl| {
        preset_delivered_config_value(&serialized, decl)
    })
    .expect("the projected preset must match its declarations")
}

#[test]
fn wave_tree_mode_is_dark_without_the_keys_and_typed_with_them() {
    use ny_levers::decls::margin_row::{
        MARGIN_ROW_TREE_MODE as MODE, MARGIN_ROW_TREE_WAVE_BATCH as BATCH,
        MARGIN_ROW_TREE_WAVE_CERT_THREADS as THREADS,
    };
    let dark = projected_levers(&crate::preset::PresetConfig::default());
    let mode = dark.resolved(&MODE).expect("declared");
    assert_eq!(mode.value.as_str(), Some(TreeMode::Legacy.token()));
    assert_eq!(mode.source, ny_levers::Source::Default);
    for decl in [&BATCH, &THREADS] {
        let resolved = dark.resolved(decl).expect("declared");
        assert_eq!(resolved.source, ny_levers::Source::Default, "{}", decl.name);
    }
    assert_eq!(
        dark.resolved(&BATCH).expect("declared").value.as_u64(),
        Some(256)
    );
    assert_eq!(
        dark.resolved(&THREADS).expect("declared").value.as_u64(),
        Some(0)
    );

    let mut preset = crate::preset::PresetConfig::default();
    preset.margin_row.tree_mode = Some(TreeMode::Wave);
    preset.margin_row.tree_wave_batch = Some(512);
    preset.margin_row.tree_wave_cert_threads = Some(6);
    let typed = projected_levers(&preset);
    let mode = typed.resolved(&MODE).expect("declared");
    assert_eq!(mode.value.as_str(), Some(TreeMode::Wave.token()));
    assert_eq!(
        mode.value.as_str().and_then(TreeMode::from_token),
        Some(TreeMode::Wave),
        "the scored value must parse back into the lane's selector"
    );
    let batch = typed.resolved(&BATCH).expect("declared");
    assert_eq!(batch.value.as_u64(), Some(512));
    let threads = typed.resolved(&THREADS).expect("declared");
    assert_eq!(threads.value.as_u64(), Some(6));
    for resolved in [mode, batch, threads] {
        assert_eq!(resolved.source, ny_levers::Source::Config);
        assert_ne!(resolved.source, ny_levers::Source::LegacyEnv);
    }
}

/// `wave_cuda` (the device search, M6) is a value of the SAME typed key: it
/// arrives through the scored run's own projection with `Source::Config`,
/// parses back into the lane's selector, arms the wave engine and asks for the
/// device backend first (then the CPU search).
#[test]
fn wave_cuda_tree_mode_arrives_typed_through_the_scored_projection() {
    use ny_levers::decls::margin_row::MARGIN_ROW_TREE_MODE as MODE;
    use ny_propagate::margin_row::WaveBackendKind;
    let mut preset = crate::preset::PresetConfig::default();
    preset.margin_row.tree_mode = Some(TreeMode::WaveCuda);
    let typed = projected_levers(&preset);
    let mode = typed.resolved(&MODE).expect("declared");
    assert_eq!(mode.value.as_str(), Some("wave_cuda"));
    assert_eq!(mode.source, ny_levers::Source::Config);
    let policy = WavePolicy::from_levers(&typed).expect("the frozen set resolves");
    assert_eq!(policy.mode, TreeMode::WaveCuda);
    assert!(policy.armed());
    assert_eq!(
        policy.mode.backends(),
        [WaveBackendKind::Cuda, WaveBackendKind::CpuF32]
    );
    let yaml: crate::preset::PresetConfig =
        serde_yaml::from_str("margin_row:\n  tree_mode: wave_cuda\n").expect("the token parses");
    assert_eq!(yaml.margin_row.tree_mode, Some(TreeMode::WaveCuda));
}

/// #wave-beta-bab ONE AUTHORITY: the lane's tree-engine policy is resolved from
/// the SAME frozen LeverSet the flight receipt serialises
/// (`FlightRecorder::materialize_levers` -> `run_and_translate` ->
/// `MarginRowRuntimePolicy::from_levers`), so a run whose receipt names
/// `tree_mode: wave` runs the wave tree and a dark receipt runs `legacy`,
/// whether or not the bridge ever wrote a governed cell. Before this the lane
/// read the cells (`WavePolicy::from_gates`) and agreed with the receipt only
/// when `preset_gates::apply` had run.
#[test]
fn wave_policy_is_resolved_from_the_receipts_lever_set() {
    use ny_propagate::margin_row::MarginRowRuntimePolicy;

    let dark = projected_levers(&crate::preset::PresetConfig::default());
    let dark_policy = MarginRowRuntimePolicy::from_levers(&dark).expect("dark policy");
    assert_eq!(dark_policy.wave, WavePolicy::default());
    assert!(!dark_policy.wave.armed());

    let mut preset = crate::preset::PresetConfig::default();
    preset.margin_row.tree_mode = Some(TreeMode::Wave);
    preset.margin_row.tree_wave_batch = Some(512);
    preset.margin_row.tree_wave_cert_threads = Some(6);
    let typed = projected_levers(&preset);
    let policy = MarginRowRuntimePolicy::from_levers(&typed).expect("typed policy");
    assert_eq!(
        policy.wave,
        WavePolicy {
            mode: TreeMode::Wave,
            batch: 512,
            cert_threads: 6,
        }
    );
    // The receipt's own view of the same set names the same engine.
    assert_eq!(
        typed
            .resolved(&ny_levers::decls::margin_row::MARGIN_ROW_TREE_MODE)
            .expect("declared")
            .value
            .as_str()
            .and_then(TreeMode::from_token),
        Some(policy.wave.mode)
    );
}

// ---------------------------------------------------------------------------
//  Synthetic twin-wall nets and the real backend
// ---------------------------------------------------------------------------

/// Register the REAL CPU f32 wave search exactly as `main.rs` does (first
/// install wins; every test registers the same backend).
pub(super) fn register_real_backend() {
    ny_propagate::margin_row::set_wave_search_backend_factory(|| {
        Some(Arc::new(ny_bbe::CpuF32Backend::new()) as Arc<dyn ny_core::search::SearchBackend>)
    });
}

struct Lcg(u64);

impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }

    fn vec(&mut self, n: usize, scale: f64) -> Vec<f64> {
        (0..n).map(|_| scale * (2.0 * self.unit() - 1.0)).collect()
    }
}

/// 2x6x6 input, strided conv, residual block, two trunk ReLUs, a 5-wide head
/// and 4 classes (the shape of ny-propagate's `tiny_spec`).
pub(super) fn tiny_spec(seed: u64, scale: f64) -> TwinSpec {
    let mut rng = Lcg(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xA5A5);
    let conv1 = TwinOpSpec::Conv {
        input: 0,
        weight: rng.vec(4 * 2 * 3 * 3, scale),
        bias: rng.vec(4, scale),
        bias_err: vec![0.0; 4],
        weight_rel_err: 1e-15,
        kernel: (4, 2, 3, 3),
        stride: (2, 2),
        pads: (1, 1, 1, 1),
        ishape: (2, 6, 6),
        oshape: (4, 3, 3),
    };
    let conv2 = TwinOpSpec::Conv {
        input: 2,
        weight: rng.vec(4 * 4 * 3 * 3, scale),
        bias: rng.vec(4, scale),
        bias_err: vec![1e-18; 4],
        weight_rel_err: 1e-15,
        kernel: (4, 4, 3, 3),
        stride: (1, 1),
        pads: (1, 1, 1, 1),
        ishape: (4, 3, 3),
        oshape: (4, 3, 3),
    };
    TwinSpec {
        n_in: 72,
        ops: vec![
            conv1,
            TwinOpSpec::Relu { input: 1 },
            conv2,
            TwinOpSpec::Add { lhs: 3, rhs: 2 },
            TwinOpSpec::Relu { input: 4 },
            TwinOpSpec::Flatten { input: 5 },
            TwinOpSpec::Gemm {
                input: 6,
                weight: rng.vec(5 * 36, scale),
                bias: rng.vec(5, scale),
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
                shape: (5, 36),
            },
            TwinOpSpec::Relu { input: 7 },
            TwinOpSpec::Gemm {
                input: 8,
                weight: rng.vec(4 * 5, scale),
                bias: rng.vec(4, scale),
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
                shape: (4, 5),
            },
        ],
    }
}

/// The midpoint argmax, the runner-up and the adversarial classes.
pub(super) fn targets(net: &TwinNet, lo: &[f64], hi: &[f64]) -> (usize, usize, Vec<usize>) {
    let mid = ndarray::Array2::from_shape_fn((lo.len(), 1), |(i, _)| f64::midpoint(lo[i], hi[i]));
    let (y, _) = net
        .forward_points(&mid, &std::collections::BTreeMap::new())
        .expect("forward");
    let (w2, b2, (n_out, n_y)) = net.gemm2();
    let scores: Vec<f64> = (0..n_out)
        .map(|o| {
            let mut sm = b2[o];
            for k in 0..n_y {
                sm += w2[o * n_y + k] * y[[k, 0]].max(0.0);
            }
            sm
        })
        .collect();
    let t = (0..n_out)
        .max_by(|&a, &b| scores[a].total_cmp(&scores[b]))
        .expect("classes");
    let worst = (0..n_out)
        .filter(|&o| o != t)
        .max_by(|&a, &b| scores[a].total_cmp(&scores[b]))
        .expect("classes");
    (t, worst, (0..n_out).filter(|&o| o != t).collect())
}

pub(super) fn wave_cfg(cert_threads: usize) -> BabConfig {
    BabConfig {
        frontier: 1,
        frontier_contextual: 1,
        lru_cap: 64,
        max_expansions: 4000,
        deadline: Some(Instant::now() + Duration::from_secs(120)),
        wave: WavePolicy {
            mode: TreeMode::Wave,
            batch: 32,
            cert_threads,
        },
        ..BabConfig::default()
    }
}

fn legacy_cfg() -> BabConfig {
    BabConfig {
        frontier: 1,
        frontier_contextual: 1,
        lru_cap: 64,
        max_expansions: 400,
        ..BabConfig::default()
    }
}

/// Instances the legacy lane proves only through a tree.
pub(super) fn tree_instances(limit: usize) -> Vec<(TwinNet, RootGates, usize, Vec<usize>)> {
    let mut out = Vec::new();
    for seed in 0..96u64 {
        let Ok(net) = TwinNet::compile(&tiny_spec(seed, 0.35)) else {
            continue;
        };
        for width in [0.01, 0.02, 0.04] {
            let lo = vec![0.01; 72];
            let hi: Vec<f64> = lo.iter().map(|l| l + width).collect();
            let (t, _, adv) = targets(&net, &lo, &hi);
            let Ok(root) = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None) else {
                continue;
            };
            let MarginRowOutcome::Unsat(stats, _) =
                MarginRowBab::run(&net, &root, t, &adv, legacy_cfg())
            else {
                continue;
            };
            if stats.expansions > 0 {
                out.push((net, root, t, adv));
                break;
            }
        }
        if out.len() >= limit {
            break;
        }
    }
    assert!(
        !out.is_empty(),
        "no synthetic instance needs a tree: the wave tests are VOID"
    );
    out
}

pub(super) fn summary() -> String {
    ny_propagate::margin_row::take_last_wave_summary().unwrap_or_default()
}

#[test]
fn registered_cpu_f32_backend_reaches_the_armed_wave_tree() {
    register_real_backend();
    let (net, root, t, adv) = tree_instances(1).swap_remove(0);
    let _ = summary();
    let _ = MarginRowBab::run(&net, &root, t, &adv, wave_cfg(2));
    let line = summary();
    assert!(
        line.contains("[bbe] phase=wave armed=1 backend=cpu_f32"),
        "the armed wave tree printed no receipt (VOID): {line:?}"
    );
    assert!(line.contains("wave-beta-bab/v1"), "{line}");
    assert!(
        !line.contains("fallback=legacy"),
        "the registered backend refused the wave: {line}"
    );
    assert!(
        !line.contains(" searched=0 "),
        "the backend never searched: {line}"
    );
}

#[test]
fn real_cpu_f32_wave_proves_robust_tiny_nets() {
    register_real_backend();
    let instances = tree_instances(4);
    let mut proved = 0usize;
    for (net, root, t, adv) in &instances {
        let out = MarginRowBab::run(net, root, *t, adv, wave_cfg(2));
        let line = summary();
        assert!(!line.contains("fallback=legacy"), "{line}");
        if let MarginRowOutcome::Unsat(stats, token) = out {
            assert!(token.publishable());
            assert_eq!(stats.ledger_ok, Some(true));
            proved += 1;
        }
    }
    assert!(
        proved > 0,
        "the real wave proved none of {} legacy-provable instances: positive control failed",
        instances.len()
    );
}

#[test]
fn real_cpu_f32_wave_never_closes_a_sat_instance() {
    register_real_backend();
    let mut checked = 0usize;
    let mut engaged = 0usize;
    for seed in [29u64, 31, 37, 41] {
        let Ok(net) = TwinNet::compile(&tiny_spec(seed, 0.35)) else {
            continue;
        };
        let lo = vec![0.01; 72];
        let hi = vec![0.05; 72];
        let (t, worst, _) = targets(&net, &lo, &hi);
        let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
        // Violated at the midpoint: `y_worst - y_t <= 0` there (`t` is the
        // midpoint argmax, `worst` the runner-up).
        let _ = summary();
        let out = MarginRowBab::run(
            &net,
            &root,
            worst,
            &[t],
            BabConfig {
                max_expansions: 400,
                ..wave_cfg(2)
            },
        );
        assert!(
            !matches!(out, MarginRowOutcome::Unsat(..)),
            "MOAT VIOLATION: the wave closed a witnessed SAT instance (seed {seed})"
        );
        let line = summary();
        engaged +=
            usize::from(line.contains("wave-beta-bab/v1") && !line.contains("fallback=legacy"));
        checked += 1;
    }
    assert!(checked >= 2, "only {checked} SAT fixtures compiled: VOID");
    // A run that fell back to the legacy tree never exercised the real
    // search's claims against the certify queue: the moat is VOID unless the
    // wave engine itself ran on a SAT instance.
    assert!(
        engaged > 0,
        "no SAT fixture reached the wave engine (every run fell back): VOID"
    );
}

#[test]
fn real_cpu_f32_wave_is_thread_count_independent() {
    register_real_backend();
    let (net, root, t, adv) = tree_instances(1).swap_remove(0);
    let key = |out: &MarginRowOutcome| match out {
        MarginRowOutcome::Unsat(s, _) => (
            true,
            s.expansions,
            s.domains_created,
            s.closed,
            s.stop.clone(),
        ),
        MarginRowOutcome::Unknown { stats, reason } => {
            stats
                .as_ref()
                .map_or((false, 0, 0, 0, reason.clone()), |s| {
                    (
                        false,
                        s.expansions,
                        s.domains_created,
                        s.closed,
                        s.stop.clone(),
                    )
                })
        }
    };
    let a = MarginRowBab::run(&net, &root, t, &adv, wave_cfg(1));
    let b = MarginRowBab::run(&net, &root, t, &adv, wave_cfg(4));
    assert_eq!(key(&a), key(&b), "the certify-pool width changed the tree");
}
