// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `#frozen-dual-state` (V1 stage A) and `#frozen-dual-bab` (V1 stage B) at
//! the CLI boundary:
//!
//! * the typed key `margin_row.frozen_dual_bab` and its two knobs
//!   (`frozen_dual_bab_node_cap`, `frozen_dual_bab_secs`) resolve DARK without
//!   the keys and arrive through the scored run's own preset projection
//!   (`preset_delivered_config_value`, no environment) with `Source::Config`,
//!   never `Source::LegacyEnv`;
//! * the CIFAR100 and TinyImageNet 2025 presets enable the bounded tier and
//!   four breakpoint steps through typed config; other presets keep it off;
//! * `frozen_dual_bab_agrees_with_the_p1_replica_node_counts` — stage B's
//!   certified BFS on the REAL cora `cifar10-set` network (weights from the
//!   corpus ONNX, the P1 replica's arm-A certified boxes from
//!   `testdata/frozen_dual_p1/`) closes the same objectives in the same
//!   number of nodes as the numpy replica, within a factor of 3, and its state
//!   root matches the replica's to 2e-5;
//! * `frozen_dual_bab_never_closes_a_cora_sat_class` — the same real network
//!   with the runner-up as the "true" class: the image centre is a
//!   counterexample (double-double), the state contains it, and the search
//!   must not close;
//! * `frozen_dual_state_contains_onnx_samples` — the REAL-ONNX containment pin:
//!   the CIFAR100 `resnet_medium` twin net exactly as the scored lane extracts
//!   it, certified interval boxes on a shrunken sub-box of a real VNNLIB box
//!   (small enough that the state build fits a debug test), and every claim of
//!   the frozen dual state checked in double-double at sampled points —
//!   corners included. The corpus is resolved in this checkout or, from a git
//!   worktree (which has no `benchmarks/`), in the main checkout; a missing
//!   corpus is VOID and FAILS the test.
//!
//! `#wide-fd-search` (ENGINE_PROGRAM M4), with the REAL ny-bbe CPU f32 level
//! search:
//!
//! * `frozen_dual_bab_search_is_dark_without_the_key_and_typed_with_it` — the
//!   typed key `margin_row.frozen_dual_bab_search` resolves to `f64` without
//!   the key and arrives with `Source::Config` for every other token;
//! * `real_cpu_f32_search_closes_the_p1_objectives_and_certifies_every_prune`
//!   — the backend searches (probe ok, no fallback), closes every P1 objective
//!   the f64 search closes, and every recorded prune re-certifies bit for bit
//!   in the f64 evaluator;
//! * `real_cpu_f32_search_never_closes_a_cora_sat_class`;
//! * `real_cpu_f32_search_is_thread_count_independent`;
//! * `registered_cpu_f32_backend_reaches_the_armed_tier` — registered as
//!   `main.rs` registers it, the armed tier resolves it and searches with it,
//!   over a compiled nine-objective lane root and over the real Cora P1
//!   replica's certified frame (`run_tier_on_frame`), where it must also
//!   CLAIM; the lane fixture includes a witnessed SAT objective;
//! * `registered_cpu_f32_backend_searches_the_shared_trunk_states` — M3 x M4:
//!   `frozen_dual_bab_state: shared` with `frozen_dual_bab_search: cpu_f32`,
//!   over the real Cora lane-root and P1 frames; the state side does not depend on the search,
//!   the backend searches the shared-trunk states (and claims on the P1
//!   frame), and every class the composed tier closes holds at the image
//!   centre and at sampled points (double-double margins).

use std::path::{Path, PathBuf};

use rayon::prelude::*;

use ny_propagate::margin_row::net::TwinOp;
use ny_propagate::margin_row::zono_state::{
    check_point, FrameInput, FrozenDualFrame, FrozenDualOptions, FrozenDualState, LayerInput,
};
use ny_propagate::margin_row::TwinNet;

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
fn frozen_dual_bab_is_dark_without_the_key() {
    let levers = projected_levers(&crate::preset::PresetConfig::default());
    let resolved = levers
        .resolved(&ny_levers::decls::margin_row::MARGIN_ROW_FROZEN_DUAL_BAB)
        .expect("the lever is declared");
    assert!(!resolved.value.as_bool());
    assert_eq!(resolved.source, ny_levers::Source::Default);
}

#[test]
fn frozen_dual_bab_preset_source_is_typed() {
    let mut preset = crate::preset::PresetConfig::default();
    preset.margin_row.frozen_dual_bab = Some(true);
    let levers = projected_levers(&preset);
    let resolved = levers
        .resolved(&ny_levers::decls::margin_row::MARGIN_ROW_FROZEN_DUAL_BAB)
        .expect("the lever is declared");
    assert!(resolved.value.as_bool());
    assert_eq!(resolved.source, ny_levers::Source::Config);
    assert_ne!(resolved.source, ny_levers::Source::LegacyEnv);
}

#[test]
fn frozen_dual_bab_knobs_are_dark_without_the_keys_and_typed_with_them() {
    use ny_levers::decls::margin_row::{
        MARGIN_ROW_FROZEN_DUAL_BAB_NODE_CAP as CAP, MARGIN_ROW_FROZEN_DUAL_BAB_SECS as SECS,
    };
    let dark = projected_levers(&crate::preset::PresetConfig::default());
    for decl in [&CAP, &SECS] {
        let resolved = dark.resolved(decl).expect("the lever is declared");
        assert_eq!(resolved.source, ny_levers::Source::Default, "{}", decl.name);
    }
    let mut preset = crate::preset::PresetConfig::default();
    preset.margin_row.frozen_dual_bab = Some(true);
    preset.margin_row.frozen_dual_bab_node_cap = Some(250_000);
    preset.margin_row.frozen_dual_bab_secs = Some(20.0);
    let levers = projected_levers(&preset);
    let cap = levers.resolved(&CAP).expect("declared");
    assert_eq!(cap.source, ny_levers::Source::Config);
    assert_eq!(cap.value.as_u64(), Some(250_000));
    let secs = levers.resolved(&SECS).expect("declared");
    assert_eq!(secs.source, ny_levers::Source::Config);
    assert_eq!(secs.value.as_f64(), Some(20.0));
    for resolved in [cap, secs] {
        assert_ne!(resolved.source, ny_levers::Source::LegacyEnv);
    }
}

/// `#fd-shared-state` (ENGINE_PROGRAM M3): `margin_row.frozen_dual_bab_state`
/// resolves DARK (`per_class`, `Source::Default`) without the key and arrives
/// through the scored run's own preset projection with `Source::Config` (never
/// `LegacyEnv`) with it; both tokens project to the `Enum` member the tier's
/// `TierPolicy::from_gates` parses. `no_shipped_preset_arms_frozen_dual_bab`
/// below covers it too (its name contains `frozen_dual_bab`).
#[test]
fn frozen_dual_bab_state_is_dark_without_the_key_and_typed_with_it() {
    use ny_levers::decls::margin_row::MARGIN_ROW_FROZEN_DUAL_BAB_STATE as STATE;
    use ny_propagate::margin_row::zono_bab::FrozenDualStateMode;
    let dark = projected_levers(&crate::preset::PresetConfig::default());
    let resolved = dark.resolved(&STATE).expect("the lever is declared");
    assert_eq!(resolved.source, ny_levers::Source::Default);
    assert_eq!(
        resolved.value.as_str(),
        Some(FrozenDualStateMode::PerClass.token()),
        "the undeclared key must resolve to the bit-identical per-class tier"
    );
    for mode in [FrozenDualStateMode::PerClass, FrozenDualStateMode::Shared] {
        let mut preset = crate::preset::PresetConfig::default();
        preset.margin_row.frozen_dual_bab = Some(true);
        preset.margin_row.frozen_dual_bab_state = Some(mode);
        let levers = projected_levers(&preset);
        let resolved = levers.resolved(&STATE).expect("declared");
        assert_eq!(resolved.source, ny_levers::Source::Config, "{mode:?}");
        assert_ne!(resolved.source, ny_levers::Source::LegacyEnv, "{mode:?}");
        assert_eq!(
            resolved
                .value
                .as_str()
                .and_then(FrozenDualStateMode::from_token),
            Some(mode),
            "the projected token must parse back to the typed value"
        );
    }
}

#[test]
fn frozen_dual_breakpoint_defaults_reach_only_the_intended_shipped_presets() {
    use ny_levers::decls::margin_row::{
        MARGIN_ROW_FROZEN_DUAL_BAB as ENABLED, MARGIN_ROW_FROZEN_DUAL_REFINE_STEPS as STEPS,
    };

    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
    let mut scanned = 0usize;
    let mut armed = Vec::new();
    for year in ["vnncomp24", "vnncomp25", "vnncomp26"] {
        let Ok(entries) = std::fs::read_dir(root.join(year)) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("yaml") {
                continue;
            }
            scanned += 1;
            let preset = crate::preset::load_preset(&path).expect("shipped preset parses");
            let levers = projected_levers(&preset);
            let enabled = levers.resolved(&ENABLED).expect("tier declaration");
            let steps = levers.resolved(&STEPS).expect("refinement declaration");
            let category = path
                .file_stem()
                .and_then(|s| s.to_str())
                .expect("preset category");
            if year == "vnncomp25" && matches!(category, "cifar100_2024" | "tinyimagenet_2024") {
                assert!(
                    enabled.value.as_bool(),
                    "{} must enable the tier",
                    path.display()
                );
                assert_eq!(steps.value.as_u64(), Some(4), "{}", path.display());
                assert_eq!(enabled.source, ny_levers::Source::Config);
                assert_eq!(steps.source, ny_levers::Source::Config);
                armed.push(category.to_string());
            } else {
                assert!(
                    !enabled.value.as_bool(),
                    "{} unexpectedly enables the tier",
                    path.display()
                );
                assert_eq!(
                    steps.value.as_u64(),
                    Some(0),
                    "{} unexpectedly enables refinement",
                    path.display()
                );
            }
        }
    }
    assert!(
        scanned > 0,
        "the scan read no yaml under {}: VOID",
        root.display()
    );
    armed.sort();
    assert_eq!(armed, ["cifar100_2024", "tinyimagenet_2024"]);
}

// ---------------------------------------------------------------------------
// Real-ONNX containment
// ---------------------------------------------------------------------------

const CORPUS: &str = "benchmarks/vnncomp2025/benchmarks/cifar100_2024";

/// Resolve the cifar100 corpus (see the module docs).
fn corpus_dir() -> PathBuf {
    corpus(CORPUS)
}

/// Resolve a corpus directory (workspace-relative) in this checkout or, from a
/// git worktree, in the main checkout. Missing is VOID and panics.
fn corpus(rel: &str) -> PathBuf {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let direct = workspace.join(rel);
    if direct.is_dir() {
        return direct;
    }
    if let Ok(text) = std::fs::read_to_string(workspace.join(".git")) {
        if let Some(gitdir) = text.trim().strip_prefix("gitdir:") {
            let gitdir = PathBuf::from(gitdir.trim());
            if let Some(main_git) = gitdir
                .ancestors()
                .find(|path| path.file_name().is_some_and(|name| name == ".git"))
            {
                if let Some(main) = main_git.parent() {
                    let candidate = main.join(rel);
                    if candidate.is_dir() {
                        return candidate;
                    }
                }
            }
        }
    }
    panic!(
        "VOID: the corpus ({rel}) is missing under {} and under the main checkout; this test \
         cannot pass without it",
        workspace.display()
    );
}

/// Unit roundoff of binary64.
const U: f64 = f64::EPSILON / 2.0;

fn gamma(n: usize) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    let nu = n as f64 * U;
    nu / (1.0 - nu)
}

fn dn(x: f64) -> f64 {
    ny_propagate::zono_core::rounding::next_down(x)
}

fn up(x: f64) -> f64 {
    ny_propagate::zono_core::rounding::next_up(x)
}

/// Certified interval of every tensor over `[lo, hi]` (test-local interval
/// arithmetic with a deliberately generous rounding and fold-error slack; the
/// containment check re-verifies every box it produces).
fn certified_ibp(net: &TwinNet, lo: &[f64], hi: &[f64]) -> Vec<(Vec<f64>, Vec<f64>)> {
    let mut t: Vec<(Vec<f64>, Vec<f64>)> = vec![(lo.to_vec(), hi.to_vec())];
    for op in &net.ops {
        let out = match op {
            TwinOp::Conv(cv) => {
                let (l, u) = &t[cv.input];
                let co = cv.kernel.0;
                let p = cv.oshape.1 * cv.oshape.2;
                let kk = cv.k_fwd;
                let g = 2.0 * gamma(kk + 4);
                let mut ol = vec![0.0; co * p];
                let mut ou = vec![0.0; co * p];
                for oc in 0..co {
                    for sp in 0..p {
                        let b = cv.bias[oc];
                        let (mut sl, mut su, mut mag) = (b, b, b.abs());
                        for tap in 0..kk {
                            let gi = cv.gather[tap * p + sp];
                            if gi == usize::MAX {
                                continue;
                            }
                            let w = cv.wmat[oc * kk + tap];
                            if w >= 0.0 {
                                sl += w * l[gi];
                                su += w * u[gi];
                            } else {
                                sl += w * u[gi];
                                su += w * l[gi];
                            }
                            mag += w.abs() * l[gi].abs().max(u[gi].abs());
                        }
                        let slack =
                            g * mag + 1.01 * cv.weight_rel_err * mag + cv.bias_err[oc] + 1e-300;
                        ol[oc * p + sp] = dn(sl - slack);
                        ou[oc * p + sp] = up(su + slack);
                    }
                }
                (ol, ou)
            }
            TwinOp::ChannelAffine {
                input,
                scale,
                shift,
                scale_rel_err,
                shift_err,
            } => {
                let (l, u) = &t[*input];
                let mut ol = Vec::with_capacity(l.len());
                let mut ou = Vec::with_capacity(l.len());
                for j in 0..l.len() {
                    let s = scale[j];
                    let (a, b) = if s >= 0.0 {
                        (s * l[j], s * u[j])
                    } else {
                        (s * u[j], s * l[j])
                    };
                    let mag = s.abs() * l[j].abs().max(u[j].abs()) + shift[j].abs();
                    let slack = 8.0 * U * mag + 1.01 * scale_rel_err * mag + shift_err[j] + 1e-300;
                    ol.push(dn(a + shift[j] - slack));
                    ou.push(up(b + shift[j] + slack));
                }
                (ol, ou)
            }
            TwinOp::Relu { input, .. } => {
                let (l, u) = &t[*input];
                (
                    l.iter().map(|v| v.max(0.0)).collect(),
                    u.iter().map(|v| v.max(0.0)).collect(),
                )
            }
            TwinOp::Add { lhs, rhs } => {
                let (la, ua) = &t[*lhs];
                let (lb, ub) = &t[*rhs];
                let ol = la
                    .iter()
                    .zip(lb)
                    .map(|(a, b)| dn(a + b - 4.0 * U * (a.abs() + b.abs())))
                    .collect();
                let ou = ua
                    .iter()
                    .zip(ub)
                    .map(|(a, b)| up(a + b + 4.0 * U * (a.abs() + b.abs())))
                    .collect();
                (ol, ou)
            }
            TwinOp::Flatten { input } => t[*input].clone(),
            TwinOp::Gemm {
                input,
                weight,
                bias,
                bias_err,
                weight_rel_err,
                shape,
            } => {
                let (no, ni) = *shape;
                let (l, u) = &t[*input];
                let g = 2.0 * gamma(ni + 4);
                let mut ol = Vec::with_capacity(no);
                let mut ou = Vec::with_capacity(no);
                for o in 0..no {
                    let b = bias[o];
                    let (mut sl, mut su, mut mag) = (b, b, b.abs());
                    for i in 0..ni {
                        let w = weight[o * ni + i];
                        if w >= 0.0 {
                            sl += w * l[i];
                            su += w * u[i];
                        } else {
                            sl += w * u[i];
                            su += w * l[i];
                        }
                        mag += w.abs() * l[i].abs().max(u[i].abs());
                    }
                    let slack = g * mag + 1.01 * weight_rel_err * mag + bias_err[o] + 1e-300;
                    ol.push(dn(sl - slack));
                    ou.push(up(su + slack));
                }
                (ol, ou)
            }
        };
        t.push(out);
    }
    t
}

fn relax(l: &[f64], u: &[f64]) -> LayerInput {
    let n = l.len();
    let mut alpha = vec![0.0; n];
    let mut s = vec![0.0; n];
    for j in 0..n {
        if l[j] >= 0.0 {
            alpha[j] = 1.0;
            s[j] = 1.0;
        } else if u[j] > 0.0 {
            s[j] = (u[j] / (u[j] - l[j])).clamp(0.0, 1.0);
            alpha[j] = if u[j] >= -l[j] { 1.0 } else { 0.0 };
        }
    }
    LayerInput {
        l: l.to_vec(),
        u: u.to_vec(),
        alpha,
        s,
    }
}

/// The frame input over the box `c +- scale * half` and its unstable count.
fn frame_input_at(
    net: &TwinNet,
    lo: &[f64],
    hi: &[f64],
    scale: f64,
) -> (FrameInput, usize, Vec<f64>, Vec<f64>) {
    let (sl, sh): (Vec<f64>, Vec<f64>) = lo
        .iter()
        .zip(hi)
        .map(|(&l, &h)| {
            let c = 0.5 * l + 0.5 * h;
            let half = 0.5 * (h - l) * scale;
            ((c - half).max(l), (c + half).min(h))
        })
        .unzip();
    let tensors = certified_ibp(net, &sl, &sh);
    let layer_of = |op: usize| -> LayerInput {
        match &net.ops[op] {
            TwinOp::Relu { input, .. } => relax(&tensors[*input].0, &tensors[*input].1),
            _ => panic!("op {op} is not a relu"),
        }
    };
    let trunk: Vec<LayerInput> = net.trunk_relus.iter().map(|&op| layer_of(op)).collect();
    let head = layer_of(net.i_gemm1 + 1);
    let unstable = trunk
        .iter()
        .chain(std::iter::once(&head))
        .map(|li| {
            li.l.iter()
                .zip(&li.u)
                .filter(|(l, u)| **l < 0.0 && **u > 0.0)
                .count()
        })
        .sum();
    (
        FrameInput {
            lo: sl.clone(),
            hi: sh.clone(),
            trunk,
            head,
        },
        unstable,
        sl,
        sh,
    )
}

#[test]
fn frozen_dual_state_contains_onnx_samples() {
    const CAP: usize = 128;
    let dir = corpus_dir();
    let onnx = dir.join("onnx/CIFAR100_resnet_medium.onnx");
    assert!(onnx.is_file(), "VOID: {} is missing", onnx.display());
    let instances = std::fs::read_to_string(dir.join("instances.csv")).expect("instances.csv");
    let vnnlib_rel = instances
        .lines()
        .find_map(|line| {
            let mut parts = line.split(',');
            let model = parts.next()?;
            let prop = parts.next()?;
            model
                .contains("resnet_medium")
                .then(|| prop.trim().to_string())
        })
        .expect("VOID: no resnet_medium row in instances.csv");
    let vnnlib = dir.join(vnnlib_rel);
    let spec = crate::commands::margin_row_bab::extract_twin_spec(&onnx)
        .expect("the scored extractor accepts CIFAR100_resnet_medium");
    let (lo, hi, t, adv) = crate::commands::margin_row_bab::parse_vnnlib_robustness(&vnnlib)
        .expect("parse the real vnnlib");
    let net = TwinNet::compile(&spec).expect("compile");

    // Shrink the box geometrically until the certified boxes leave between 1
    // and CAP unstable neurons (a non-vacuous state a debug build can afford).
    let mut chosen = None;
    let mut prev_scale = 1.0f64;
    for k in 0..48 {
        let scale = 0.5f64.powi(k);
        let (input, unstable, sl, sh) = frame_input_at(&net, &lo, &hi, scale);
        if unstable == 0 {
            // Bisect (in log scale) between the last over-full scale and this one.
            let (mut s_lo, mut s_hi) = (scale, prev_scale);
            for _ in 0..16 {
                let mid = (s_lo * s_hi).sqrt();
                let (input, unstable, sl, sh) = frame_input_at(&net, &lo, &hi, mid);
                if unstable == 0 {
                    s_lo = mid;
                } else if unstable > CAP {
                    s_hi = mid;
                } else {
                    chosen = Some((input, unstable, sl, sh, mid));
                    break;
                }
            }
            break;
        }
        if unstable <= CAP {
            chosen = Some((input, unstable, sl, sh, scale));
            break;
        }
        prev_scale = scale;
    }
    let (input, unstable, sl, sh, scale) =
        chosen.expect("VOID: no sub-box with 1..=CAP unstable neurons was found");
    let frame = FrozenDualFrame::new(&net, input, t).expect("frame");
    assert_eq!(frame.n_unstable, unstable);
    assert!(
        frame.n_unstable > 0,
        "VOID: the state has no relaxation symbol"
    );
    eprintln!(
        "[frozen-dual-onnx] scale={scale:.3e} unstable={unstable} n_gens={} layers={}",
        frame.n_gens,
        frame.layers.len()
    );

    let mut checked = 0usize;
    for &j in adv.iter().take(2) {
        let st = FrozenDualState::build(&net, &frame, j, &FrozenDualOptions::default())
            .expect("state build");
        let points: Vec<Vec<f64>> = (0..32u64)
            .map(|s| {
                let mut h = s.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (j as u64);
                sl.iter()
                    .zip(&sh)
                    .map(|(&l, &u)| {
                        h ^= h << 13;
                        h ^= h >> 7;
                        h ^= h << 17;
                        #[allow(clippy::cast_precision_loss)]
                        let r = (h >> 11) as f64 / (1u64 << 53) as f64;
                        if s % 4 == 0 {
                            if r < 0.5 {
                                l
                            } else {
                                u
                            }
                        } else {
                            (l + (u - l) * r).clamp(l, u)
                        }
                    })
                    .collect()
            })
            .collect();
        let reports: Vec<_> = points
            .par_iter()
            .map(|x| check_point(&net, &frame, &st, x).expect("check"))
            .collect();
        for (s, pc) in reports.iter().enumerate() {
            assert!(
                pc.holds(),
                "CIFAR100 resnet_medium class {j} point {s}: the frozen dual state does not \
                 contain the network (a false-unsat generator): box={:e} symbol={:e} row={:e} \
                 obj={:e}",
                pc.box_excess,
                pc.symbol_excess,
                pc.row_excess,
                pc.obj_excess
            );
            checked += 1;
        }
        eprintln!(
            "[frozen-dual-onnx] class={j} root_g={:+.6e} r_obj={:.3e} max_r_k={:.3e} \
             build_ms={:.0} points={}",
            st.root_bound(),
            st.r_obj,
            st.max_row_remainder(),
            st.stats.build_ms,
            reports.len()
        );
    }
    assert!(checked > 0, "nothing checked: VOID");
}

// ---------------------------------------------------------------------------
// #frozen-dual-bab: node-count agreement with the P1 numpy replica (real cora)
// ---------------------------------------------------------------------------

use ny_propagate::margin_row::zono_bab::{
    ledger_closes, search, search_with_refinement_backend, Candidate, SearchOptions, SearchReport,
};
use ny_propagate::margin_row::zono_state::{
    dd_forward, static_order, FrozenDualProblem, NodeScratch,
};
use ny_propagate::margin_row::{TwinOpSpec, TwinSpec};

const CORA: &str = "benchmarks/vnncomp2026/benchmarks/cora_2024/1.0";

/// The replica's `ETA`: a generous per-op absolute underflow allowance.
const ETA: f64 = 8.0 * f64::from_bits(2); // 2^-1070

fn hexf(s: &str) -> f64 {
    f64::from_bits(u64::from_str_radix(s, 16).expect("fixture hex"))
}

/// One P1 objective, as the replica recorded it.
#[derive(Debug, Clone)]
struct P1Obj {
    obj: usize,
    j: usize,
    verdict: String,
    nodes: u64,
    depth: usize,
    crown_root: f64,
    state_root: f64,
    n_gens: usize,
    unstable: usize,
    trace: String,
}

/// A P1 arm-A fixture (`testdata/frozen_dual_p1/*.txt`, exported read-only
/// from the replica).
struct P1Fixture {
    tag: String,
    onnx: String,
    vnnlib: String,
    label: usize,
    others: Vec<usize>,
    pre: (f64, f64),
    w_probes: Vec<(usize, usize, usize, f64)>,
    b_probes: Vec<(usize, usize, f64)>,
    x_probes: Vec<(usize, f64, f64)>,
    l: Vec<Vec<f64>>,
    u: Vec<Vec<f64>>,
    objs: Vec<P1Obj>,
}

fn load_p1(tag: &str) -> P1Fixture {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata/frozen_dual_p1")
        .join(format!("cifar10-set_{tag}_armA.txt"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("VOID: fixture {} unreadable: {e}", path.display()));
    let mut fx = P1Fixture {
        tag: String::new(),
        onnx: String::new(),
        vnnlib: String::new(),
        label: usize::MAX,
        others: Vec::new(),
        pre: (f64::NAN, f64::NAN),
        w_probes: Vec::new(),
        b_probes: Vec::new(),
        x_probes: Vec::new(),
        l: Vec::new(),
        u: Vec::new(),
        objs: Vec::new(),
    };
    let mut traces: Vec<(usize, String)> = Vec::new();
    for line in text.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let tok: Vec<&str> = line.split_whitespace().collect();
        let num = |i: usize| -> usize { tok[i].parse().expect("fixture integer") };
        match tok[0] {
            "tag" => fx.tag = tok[1].to_string(),
            "onnx" => fx.onnx = tok[1].to_string(),
            "vnnlib" => fx.vnnlib = tok[1].to_string(),
            "label" => fx.label = num(1),
            "others" => fx.others = (1..tok.len()).map(num).collect(),
            "pre" => fx.pre = (hexf(tok[1]), hexf(tok[2])),
            "w_probe" => fx.w_probes.push((num(1), num(2), num(3), hexf(tok[4]))),
            "b_probe" => fx.b_probes.push((num(1), num(2), hexf(tok[3]))),
            "x_probe" => fx.x_probes.push((num(1), hexf(tok[2]), hexf(tok[3]))),
            "l" | "u" => {
                let m = num(1);
                let vals: Vec<f64> = tok[2..].iter().map(|h| hexf(h)).collect();
                let dst = if tok[0] == "l" { &mut fx.l } else { &mut fx.u };
                assert_eq!(dst.len(), m, "fixture layers out of order");
                dst.push(vals);
            }
            "obj" => {
                let mut o = P1Obj {
                    obj: num(1),
                    j: usize::MAX,
                    verdict: String::new(),
                    nodes: 0,
                    depth: 0,
                    crown_root: f64::NAN,
                    state_root: f64::NAN,
                    n_gens: 0,
                    unstable: 0,
                    trace: String::new(),
                };
                let mut i = 2;
                while i + 1 < tok.len() {
                    let v = tok[i + 1];
                    match tok[i] {
                        "j" => o.j = v.parse().expect("j"),
                        "verdict" => o.verdict = v.to_string(),
                        "nodes" => o.nodes = v.parse().expect("nodes"),
                        "depth" => o.depth = v.parse().expect("depth"),
                        "crown_root" => o.crown_root = hexf(v),
                        "state_root" => o.state_root = hexf(v),
                        "n_gens" => o.n_gens = v.parse().expect("n_gens"),
                        "unstable" => o.unstable = v.parse().expect("unstable"),
                        _ => {}
                    }
                    i += 2;
                }
                fx.objs.push(o);
            }
            "trace" => traces.push((num(1), tok[2..].join(" "))),
            other => panic!("unknown fixture line {other:?}"),
        }
    }
    for (obj, trace) in traces {
        if let Some(o) = fx.objs.iter_mut().find(|o| o.obj == obj) {
            o.trace = trace;
        }
    }
    assert!(
        fx.label != usize::MAX && fx.l.len() == 7 && fx.u.len() == 7 && !fx.objs.is_empty(),
        "VOID: fixture {tag} is incomplete"
    );
    fx
}

/// A fully-connected ONNX chain: optional scalar `Mul`/`Add` preamble, then
/// `MatMul` + `Add` (+ `Relu`) layers. Weights are the f32 initializers,
/// exactly, as `(out, in)` row-major f64.
struct FcNet {
    pre: (f64, f64),
    w: Vec<Vec<f64>>,
    b: Vec<Vec<f64>>,
    shapes: Vec<(usize, usize)>,
}

fn f32_payload(t: &ny_onnx::onnx_proto::TensorProto) -> Vec<f64> {
    assert_eq!(t.data_type, 1, "{} is not FLOAT", t.name);
    if t.raw_data.is_empty() {
        t.float_data.iter().map(|&v| f64::from(v)).collect()
    } else {
        t.raw_data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f64::from(f32::from_le_bytes(*c)))
            .collect()
    }
}

fn load_fc_onnx(path: &Path) -> FcNet {
    use prost::Message;
    let bytes = std::fs::read(path).expect("read onnx");
    let model = ny_onnx::onnx_proto::ModelProto::decode(bytes.as_slice()).expect("decode onnx");
    let g = model.graph.as_ref().expect("graph");
    let inits: std::collections::HashMap<&str, &ny_onnx::onnx_proto::TensorProto> =
        g.initializer.iter().map(|t| (t.name.as_str(), t)).collect();
    let init_of = |n: &ny_onnx::onnx_proto::NodeProto| {
        n.input
            .iter()
            .find_map(|i| inits.get(i.as_str()).copied())
            .unwrap_or_else(|| panic!("{} has no initializer input", n.op_type))
    };
    let mut fc = FcNet {
        pre: (1.0, 0.0),
        w: Vec::new(),
        b: Vec::new(),
        shapes: Vec::new(),
    };
    let mut pending: Option<(Vec<f64>, (usize, usize))> = None;
    for n in &g.node {
        match n.op_type.as_str() {
            "Mul" if fc.w.is_empty() && pending.is_none() => {
                fc.pre.0 = f32_payload(init_of(n))[0];
            }
            "Add" if fc.w.is_empty() && pending.is_none() => {
                fc.pre.1 = f32_payload(init_of(n))[0];
            }
            "MatMul" => {
                let t = init_of(n);
                let (ni, no) = (
                    usize::try_from(t.dims[0]).expect("dim"),
                    usize::try_from(t.dims[1]).expect("dim"),
                );
                let v = f32_payload(t);
                // (in, out) -> (out, in)
                let mut w = vec![0.0; no * ni];
                for i in 0..ni {
                    for o in 0..no {
                        w[o * ni + i] = v[i * no + o];
                    }
                }
                pending = Some((w, (no, ni)));
            }
            "Add" => {
                let (w, shape) = pending.take().expect("Add without a MatMul");
                fc.w.push(w);
                fc.b.push(f32_payload(init_of(n)));
                fc.shapes.push(shape);
            }
            "Relu" => {}
            other => panic!("unexpected op {other} in the cora FC net"),
        }
    }
    fc
}

/// The FC chain as a twin net: every hidden layer but the last is a 1x1 conv
/// on a `(C, 1, 1)` tensor (the same affine map), the last hidden layer is the
/// head Gemm, the output layer the final Gemm. The scalar preamble is NOT in
/// the net: its input is `x' = s x + t`, over the replica's certified box.
fn cora_twin_spec(fc: &FcNet) -> TwinSpec {
    let layers = fc.w.len();
    assert!(layers >= 3, "need trunk + head + output layers");
    let n_in = fc.shapes[0].1;
    let mut ops = Vec::new();
    let mut cur = 0usize;
    for k in 0..layers - 2 {
        let (no, ni) = fc.shapes[k];
        ops.push(TwinOpSpec::Conv {
            input: cur,
            weight: fc.w[k].clone(),
            bias: fc.b[k].clone(),
            bias_err: vec![0.0; no],
            weight_rel_err: 0.0,
            kernel: (no, ni, 1, 1),
            stride: (1, 1),
            pads: (0, 0, 0, 0),
            ishape: (ni, 1, 1),
            oshape: (no, 1, 1),
        });
        cur = ops.len();
        ops.push(TwinOpSpec::Relu { input: cur });
        cur = ops.len();
    }
    ops.push(TwinOpSpec::Flatten { input: cur });
    cur = ops.len();
    for k in [layers - 2, layers - 1] {
        let (no, ni) = fc.shapes[k];
        ops.push(TwinOpSpec::Gemm {
            input: cur,
            weight: fc.w[k].clone(),
            bias: fc.b[k].clone(),
            bias_err: Vec::new(),
            weight_rel_err: 0.0,
            shape: (no, ni),
        });
        cur = ops.len();
        if k == layers - 2 {
            ops.push(TwinOpSpec::Relu { input: cur });
            cur = ops.len();
        }
    }
    TwinSpec { n_in, ops }
}

/// The replica's input box: the vnnlib box widened one ulp outward, pushed
/// through the scalar preamble with a `2.0001 u` charge, rounded outward
/// (`p1lib.parse_vnnlib` + `input_box`, the same IEEE operations).
fn replica_input_box(vnnlib: &Path, n_in: usize, pre: (f64, f64)) -> (Vec<f64>, Vec<f64>) {
    let text = std::fs::read_to_string(vnnlib).expect("read vnnlib");
    let mut lo = vec![f64::NAN; n_in];
    let mut hi = vec![f64::NAN; n_in];
    for line in text.lines() {
        let Some(rest) = line.trim().strip_prefix("(assert (") else {
            continue;
        };
        let tok: Vec<&str> = rest.trim_end_matches(')').split_whitespace().collect();
        if tok.len() != 3 {
            continue;
        }
        let Some(idx) = tok[1].strip_prefix("X_") else {
            continue;
        };
        let i: usize = idx.parse().expect("input index");
        let v: f64 = tok[2].parse().expect("bound");
        match tok[0] {
            "<=" => hi[i] = v,
            ">=" => lo[i] = v,
            _ => {}
        }
    }
    assert!(
        lo.iter().chain(&hi).all(|v| v.is_finite()),
        "vnnlib box incomplete"
    );
    let (s, t) = pre;
    let unit = f64::EPSILON / 2.0;
    let mut xl = Vec::with_capacity(n_in);
    let mut xu = Vec::with_capacity(n_in);
    for i in 0..n_in {
        let l = dn(lo[i]);
        let h = up(hi[i]);
        let a = s * l + t;
        let b = s * h + t;
        let ea = 2.0001 * unit * ((s * l).abs() + t.abs()) + ETA;
        let eb = 2.0001 * unit * ((s * h).abs() + t.abs()) + ETA;
        xl.push(dn(a - ea));
        xu.push(up(b + eb));
    }
    (xl, xu)
}

/// The replica's arm-A relaxation of one layer: adaptive alpha (`u > -l`) and
/// the chord `u / (u - l)` on unstable neurons.
fn replica_layer(l: &[f64], u: &[f64]) -> LayerInput {
    let n = l.len();
    let mut alpha = vec![0.0; n];
    let mut s = vec![0.0; n];
    for j in 0..n {
        if l[j] >= 0.0 {
            alpha[j] = 1.0;
            s[j] = 1.0;
        } else if u[j] > 0.0 {
            alpha[j] = if u[j] > -l[j] { 1.0 } else { 0.0 };
            s[j] = u[j] / (u[j] - l[j]);
        }
    }
    LayerInput {
        l: l.to_vec(),
        u: u.to_vec(),
        alpha,
        s,
    }
}

/// The real cora net + the fixture's frame input, with every provenance probe
/// checked (weights, preamble, input box): a drifted fixture or corpus fails
/// loudly instead of comparing different problems.
fn p1_setup(fx: &P1Fixture) -> (TwinNet, FrameInput) {
    let dir = corpus(CORA);
    let fc = load_fc_onnx(&dir.join(&fx.onnx));
    assert_eq!(fc.w.len(), 8, "cifar10-set has eight affine layers");
    assert_eq!(
        (fc.pre.0.to_bits(), fc.pre.1.to_bits()),
        (fx.pre.0.to_bits(), fx.pre.1.to_bits()),
        "{}: preamble drift",
        fx.tag
    );
    for &(k, r, c, v) in &fx.w_probes {
        let ni = fc.shapes[k].1;
        assert_eq!(
            fc.w[k][r * ni + c].to_bits(),
            v.to_bits(),
            "{}: weight W{k}[{r},{c}] drift",
            fx.tag
        );
    }
    for &(k, i, v) in &fx.b_probes {
        assert_eq!(
            fc.b[k][i].to_bits(),
            v.to_bits(),
            "{}: bias b{k}[{i}]",
            fx.tag
        );
    }
    let n_in = fc.shapes[0].1;
    let (xl, xu) = replica_input_box(&dir.join(&fx.vnnlib), n_in, fc.pre);
    for &(i, l, u) in &fx.x_probes {
        assert_eq!(
            (xl[i].to_bits(), xu[i].to_bits()),
            (l.to_bits(), u.to_bits()),
            "{}: input box drift at {i}",
            fx.tag
        );
    }
    let net =
        TwinNet::compile(&cora_twin_spec(&fc)).expect("the cora chain compiles as a twin net");
    let trunk: Vec<LayerInput> = (0..6).map(|m| replica_layer(&fx.l[m], &fx.u[m])).collect();
    let head = replica_layer(&fx.l[6], &fx.u[6]);
    (
        net,
        FrameInput {
            lo: xl,
            hi: xu,
            trunk,
            head,
        },
    )
}

/// Run every fixture objective with at most `max_nodes` replica nodes and
/// compare. Returns the number compared.
fn p1_compare(tag: &str, max_nodes: u64) -> usize {
    let fx = load_p1(tag);
    let (net, input) = p1_setup(&fx);
    let frame = FrozenDualFrame::new(&net, input, fx.label).expect("frame");
    let mut compared = 0usize;
    for o in fx.objs.iter().filter(|o| o.nodes <= max_nodes) {
        assert_eq!(o.verdict, "closed", "{tag} obj {}: fixture verdict", o.obj);
        assert!(fx.others.contains(&o.j));
        assert_eq!(
            frame.n_gens, o.n_gens,
            "{tag} obj {}: generator count",
            o.obj
        );
        assert_eq!(
            frame.n_unstable, o.unstable,
            "{tag} obj {}: unstable count",
            o.obj
        );
        let st = FrozenDualState::build(&net, &frame, o.j, &FrozenDualOptions::default())
            .expect("state");
        let root = st.root_bound();
        assert!(
            (root - o.state_root).abs() <= 2e-5 * (1.0 + o.state_root.abs()),
            "{tag} obj {} (class {}): state root {root:+.9} vs the replica's {:+.9} \
             (crown {:+.9}) — not the replica's problem",
            o.obj,
            o.j,
            o.state_root,
            o.crown_root
        );
        // Containment on the real network at the centre and seven seeded
        // points (every other one a box corner), in double-double.
        for k in 0..8u64 {
            let mut h = k.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (o.j as u64);
            let x: Vec<f64> = frame
                .lo
                .iter()
                .zip(&frame.hi)
                .map(|(&l, &u)| {
                    h ^= h << 13;
                    h ^= h >> 7;
                    h ^= h << 17;
                    #[allow(clippy::cast_precision_loss)]
                    let r = (h >> 11) as f64 / (1u64 << 53) as f64;
                    if k == 0 {
                        f64::midpoint(l, u)
                    } else if k % 2 == 0 {
                        if r < 0.5 {
                            l
                        } else {
                            u
                        }
                    } else {
                        (l + (u - l) * r).clamp(l, u)
                    }
                })
                .collect();
            let pc = check_point(&net, &frame, &st, &x).expect("check");
            assert!(
                pc.holds(),
                "{tag} obj {} point {k}: the frozen dual state does not contain the real cora \
                 network (a false-unsat generator): {pc:?}",
                o.obj
            );
        }
        let order = static_order(&st);
        let prob = FrozenDualProblem::new(&st, &order).expect("problem");
        let rep = search(
            &prob,
            &SearchOptions {
                node_cap: (4 * o.nodes).max(5_000),
                ..SearchOptions::default()
            },
        );
        let trace: Vec<String> = rep
            .levels
            .iter()
            .map(|l| format!("{}:{}:{}", l.depth, l.frontier, l.pruned))
            .collect();
        eprintln!(
            "[frozen-dual-p1] {tag} obj={} class={} root={root:+.6} replica_root={:+.6} \
             nodes={} replica_nodes={} depth={} replica_depth={} verdict={:?} \
             cert_min_slack={:.3e} far_used={} secs={:.2}\n  ny      {}\n  replica {}",
            o.obj,
            o.j,
            o.state_root,
            rep.nodes,
            o.nodes,
            rep.depth,
            o.depth,
            rep.verdict,
            rep.cert_min_slack,
            rep.far_used,
            rep.secs,
            trace.join(" "),
            o.trace
        );
        assert!(
            rep.closed(),
            "{tag} obj {} (class {}): the replica closes in {} nodes, ny's search did not \
             ({:?} after {} nodes)",
            o.obj,
            o.j,
            o.nodes,
            rep.verdict,
            rep.nodes
        );
        #[allow(clippy::cast_precision_loss)]
        let ratio = rep.nodes.max(1) as f64 / o.nodes.max(1) as f64;
        assert!(
            (1.0 / 3.0..=3.0).contains(&ratio),
            "{tag} obj {} (class {}): {} nodes vs the replica's {} (ratio {ratio:.3}) is \
             outside the factor-3 tolerance",
            o.obj,
            o.j,
            rep.nodes,
            o.nodes
        );
        compared += 1;
    }
    compared
}

#[test]
fn frozen_dual_bab_agrees_with_the_p1_replica_node_counts() {
    // The debug-affordable objectives: img141 (724 nodes), img321 obj1 (262)
    // and obj3 (70). The heavy ones run in the release-only test below.
    let compared = p1_compare("img141", 1_000) + p1_compare("img321", 1_000);
    assert!(
        compared >= 3,
        "only {compared} P1 objectives compared: VOID"
    );
}

/// img321 obj2 (38,244 nodes) and img168 obj7 (14,994 nodes) on the tracked
/// P1 fixtures: hermetic, but not debug-affordable, so it is compiled only for
/// release test builds (`targo --unverified test --release -p ny-cli --bin ny
/// frozen_dual_bab_agrees`), the precedent of ny-propagate's release-only
/// tests, rather than `#[ignore]`d in the default lane.
///
/// Under the 50,000-node cap the tracked fixtures hold exactly FOUR closed
/// objectives: img321 obj1 (262), obj2 (38,244) and obj3 (70), and img168
/// obj7 (14,994); img321 obj4 (601,266 nodes) is over the cap. The guard
/// read `>= 5` while the test was `#[ignore]`d and never ran; the first
/// release run (2026-10-02) compared all four, each inside the factor-3
/// node tolerance, and failed only on that count.
#[cfg(not(debug_assertions))]
#[test]
fn frozen_dual_bab_agrees_with_the_p1_replica_node_counts_heavy() {
    let compared = p1_compare("img321", 50_000) + p1_compare("img168", 50_000);
    assert!(
        compared >= 4,
        "only {compared} P1 objectives compared: VOID"
    );
}

fn replay_breakpoint_prunes(prob: &FrozenDualProblem, report: &SearchReport) -> usize {
    assert_eq!(report.pruned.len() as u64, report.certified_prunes);
    let mut new_prunes = 0;
    for node in &report.pruned {
        let bound = prob.eval_node(
            &node.sides,
            &node.l0,
            &node.l1,
            &mut NodeScratch::default(),
            None,
        );
        assert!(
            bound.prunes(),
            "real network prune lacks a positive certificate"
        );
        assert_eq!(bound.certified.to_bits(), node.certified.to_bits());
        new_prunes += usize::from(node.candidate == Candidate::Breakpoint);
    }
    assert_eq!(new_prunes as u64, report.refine_prunes);
    new_prunes
}

fn assert_float_bits_equal(expected: &[f64], actual: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&expected, &actual)) in expected.iter().zip(actual).enumerate() {
        assert_eq!(actual.to_bits(), expected.to_bits(), "float {i} differs");
    }
}

/// Compact construction must reproduce the original ranking and certificate
/// data while generating every candidate row before deciding which to retain.
fn compare_real_cora_ranked_build(
    net: &TwinNet,
    frame: &FrozenDualFrame,
    class: usize,
) -> (FrozenDualProblem, FrozenDualProblem) {
    use ny_propagate::margin_row::zono_state::build::estimate_ranked_state_bytes;
    use ny_propagate::margin_row::zono_state::estimate_state_bytes;

    const RETAIN: usize = 32;
    // Keep both builders on the same bounded chunk schedule. The streaming
    // peak estimate also reserves scoring scratch, so use eight columns to
    // compare complete admitted working sets rather than just stored rows.
    let opts = FrozenDualOptions {
        chunk_cols: 8,
        ..FrozenDualOptions::default()
    };
    let legacy = FrozenDualState::build(net, frame, class, &opts).expect("legacy real state");
    let ranked = FrozenDualState::build_ranked(net, frame, class, &opts, RETAIN)
        .expect("streamed real state");
    let order = static_order(&legacy);
    let expected_rows = &order[..RETAIN.min(order.len())];
    let compact = &ranked.state;
    assert!(legacy.num_unstable() > RETAIN, "fixture must discard rows");
    assert_eq!(ranked.total_unstable, frame.n_unstable);
    assert_eq!(ranked.original_rows, expected_rows);
    assert_eq!(compact.num_unstable(), RETAIN);
    assert_eq!((compact.t, compact.class), (legacy.t, legacy.class));
    assert_eq!(compact.n_gens, legacy.n_gens);
    assert_eq!(compact.n_in_gens, legacy.n_in_gens);
    assert_eq!(compact.c_obj.to_bits(), legacy.c_obj.to_bits());
    assert_eq!(compact.r_obj.to_bits(), legacy.r_obj.to_bits());
    assert_float_bits_equal(&legacy.d, &compact.d);
    assert_eq!(
        compact.root_bound().to_bits(),
        legacy.root_bound().to_bits()
    );
    for (slot, &original) in expected_rows.iter().enumerate() {
        assert_float_bits_equal(legacy.row(original), compact.row(slot));
        for (old, new) in [
            (&legacy.c_in, &compact.c_in),
            (&legacy.r, &compact.r),
            (&legacy.lam, &compact.lam),
            (&legacy.mu, &compact.mu),
            (&legacy.lo, &compact.lo),
            (&legacy.hi, &compact.hi),
        ] {
            assert_eq!(new[slot].to_bits(), old[original].to_bits());
        }
        assert_eq!(compact.enc[slot], legacy.enc[original]);
        assert_eq!(compact.layer[slot], legacy.layer[original]);
        assert_eq!(compact.site[slot], legacy.site[original]);
        assert_eq!(compact.neuron[slot], legacy.neuron[original]);
    }
    // Retaining fewer rows must not drop any objective generator or skip the
    // seed columns needed to score the discarded rows.
    assert_eq!(compact.stats.columns, 1 + frame.n_unstable);
    assert_eq!(compact.stats.columns, legacy.stats.columns);
    assert_eq!(compact.stats.passes, legacy.stats.passes);
    let legacy_estimate = estimate_state_bytes(frame, opts.chunk_cols);
    let ranked_estimate = estimate_ranked_state_bytes(frame, opts.chunk_cols, RETAIN);
    assert_eq!(legacy.stats.estimate_bytes, legacy_estimate);
    assert_eq!(compact.stats.estimate_bytes, ranked_estimate);
    assert!(ranked_estimate < legacy_estimate);
    assert!(compact.bytes() < legacy.bytes());
    eprintln!(
        "[frozen-dual-streamed-cora] class={class} generated={} retained={} \
         legacy_estimate={} compact_estimate={} legacy_bytes={} compact_bytes={} \
         passes={} columns={}",
        ranked.total_unstable,
        compact.num_unstable(),
        legacy_estimate,
        ranked_estimate,
        legacy.bytes(),
        compact.bytes(),
        compact.stats.passes,
        compact.stats.columns,
    );
    // The point checker needs the full symbol metadata; compact states must
    // refuse rather than reconstruct missing relaxation symbols as zero.
    assert!(check_point(net, frame, compact, &frame.lo).is_err());
    // Legacy containment transfers to the retained certificate data through
    // the bit checks above; dropped symbols still appear in compact.d.
    for point in [
        frame.lo.clone(),
        frame.hi.clone(),
        frame
            .lo
            .iter()
            .zip(&frame.hi)
            .map(|(&lo, &hi)| f64::midpoint(lo, hi))
            .collect(),
    ] {
        assert!(check_point(net, frame, &legacy, &point)
            .expect("legacy network containment")
            .holds());
    }
    let legacy_problem = FrozenDualProblem::new(&legacy, expected_rows).expect("legacy top rows");
    let compact_order: Vec<_> = (0..compact.num_unstable()).collect();
    let compact_problem =
        FrozenDualProblem::new(compact, &compact_order).expect("compact ranked rows");
    assert_eq!(compact_problem.n(), legacy_problem.n());
    assert_eq!(compact_problem.max_depth(), legacy_problem.max_depth());
    assert_eq!(
        compact_problem.c0().to_bits(),
        legacy_problem.c0().to_bits()
    );
    assert_eq!(
        compact_problem.root_bound().to_bits(),
        legacy_problem.root_bound().to_bits()
    );
    for depth in [1, 2, 4, 8, 16, RETAIN] {
        for mask in [0u64, u64::MAX, 0x5a5a_5a5a, 0xc3c3_c3c3] {
            use ny_propagate::margin_row::zono_state::Side;
            let sides: Vec<_> = (0..depth)
                .map(|i| {
                    if mask & (1 << i) == 0 {
                        Side::Off
                    } else {
                        Side::On
                    }
                })
                .collect();
            let l0: Vec<_> = (0..depth).map(|i| (i % 5) as f64 * 0.0625).collect();
            let l1: Vec<_> = (0..depth).map(|i| (i % 3) as f64 * 0.03125).collect();
            let expected =
                legacy_problem.eval_node(&sides, &l0, &l1, &mut NodeScratch::default(), None);
            let actual =
                compact_problem.eval_node(&sides, &l0, &l1, &mut NodeScratch::default(), None);
            assert_eq!(actual.value.to_bits(), expected.value.to_bits());
            assert_eq!(actual.certified.to_bits(), expected.certified.to_bits());
            assert_eq!(actual.prunes(), expected.prunes());
        }
    }
    (legacy_problem, compact_problem)
}

fn assert_same_search_work(expected: &SearchReport, actual: &SearchReport) {
    assert_eq!(actual.verdict, expected.verdict);
    assert_eq!(actual.nodes, expected.nodes);
    assert_eq!(actual.depth, expected.depth);
    assert_eq!(actual.certified_prunes, expected.certified_prunes);
    assert_eq!(
        actual.root_certified.to_bits(),
        expected.root_certified.to_bits()
    );
    assert_eq!(
        actual.cert_min_slack.to_bits(),
        expected.cert_min_slack.to_bits()
    );
    assert_eq!(actual.worst_open.to_bits(), expected.worst_open.to_bits());
    assert_eq!(actual.far_used, expected.far_used);
    assert_eq!(actual.peak_frontier, expected.peak_frontier);
    assert_eq!(actual.levels, expected.levels);
    assert_eq!(actual.pruned, expected.pruned);
    assert_eq!(actual.slack_hist, expected.slack_hist);
    assert_eq!(actual.crosschecks, expected.crosschecks);
    assert_eq!(actual.crosscheck_failures, expected.crosscheck_failures);
    assert_eq!(actual.evaluator, expected.evaluator);
    assert_eq!(actual.far_first, expected.far_first);
    assert_eq!(actual.refine_calls, expected.refine_calls);
    assert_eq!(actual.refine_steps, expected.refine_steps);
    assert_eq!(actual.refine_accepted, expected.refine_accepted);
    assert_eq!(actual.refine_prunes, expected.refine_prunes);
    assert_eq!(actual.refine_refused, expected.refine_refused);
    assert_eq!(actual.refine_skipped, expected.refine_skipped);
}

#[test]
fn streamed_ranked_states_preserve_real_cora_certificates_and_search_work() {
    for (tag, classes) in [("img321", &[4, 2][..]), ("img141", &[0][..])] {
        let fixture = load_p1(tag);
        let (net, input) = p1_setup(&fixture);
        let frame = FrozenDualFrame::new(&net, input, fixture.label).expect("real Cora frame");
        for &class in classes {
            let (legacy, compact) = compare_real_cora_ranked_build(&net, &frame, class);
            let opts = SearchOptions {
                node_cap: 5_000,
                record_prunes: 5_000,
                crosscheck_every: 1,
                ..SearchOptions::default()
            };
            let baseline = search_with_refinement_backend(&legacy, &opts, None);
            let streamed = search_with_refinement_backend(&compact, &opts, None);
            assert!(
                baseline.closed(),
                "{tag} class {class}: baseline must close"
            );
            assert_same_search_work(&baseline, &streamed);
            assert!(ledger_closes(&streamed.levels));
            replay_breakpoint_prunes(&compact, &streamed);
            eprintln!(
                "[frozen-dual-streamed-cora] tag={tag} class={class} nodes={} verdict={:?}",
                streamed.nodes, streamed.verdict
            );
        }
    }
}

#[test]
fn streamed_ranked_states_preserve_a_witnessed_real_cora_sat_objective() {
    let fixture = load_p1("img141");
    let (net, input) = p1_setup(&fixture);
    let centre: Vec<_> = input
        .lo
        .iter()
        .zip(&input.hi)
        .map(|(&lo, &hi)| f64::midpoint(lo, hi))
        .collect();
    let values = dd_forward(&net, &centre).expect("real-network witness");
    let outputs = &values[net.ops.len()];
    let mut classes: Vec<_> = (0..outputs.len()).collect();
    classes.sort_by(|&a, &b| {
        outputs[b]
            .to_f64()
            .total_cmp(&outputs[a].to_f64())
            .then(a.cmp(&b))
    });
    let (top, runner_up) = (classes[0], classes[1]);
    assert!((outputs[runner_up] - outputs[top]).to_f64() < 0.0);
    let frame = FrozenDualFrame::new(&net, input, runner_up).expect("swapped-target frame");
    let (legacy, compact) = compare_real_cora_ranked_build(&net, &frame, top);
    let backend = ny_bbe::CpuF32Backend::with_threads(1);
    for refine_steps in [0, 4] {
        let opts = SearchOptions {
            node_cap: 256,
            record_prunes: 256,
            refine_steps,
            crosscheck_every: 1,
            ..SearchOptions::default()
        };
        let injected =
            (refine_steps > 0).then_some(&backend as &dyn ny_core::search::SearchBackend);
        let baseline = search_with_refinement_backend(&legacy, &opts, injected);
        let streamed = search_with_refinement_backend(&compact, &opts, injected);
        assert_same_search_work(&baseline, &streamed);
        assert!(
            !streamed.closed(),
            "streaming closed a witnessed unsafe class"
        );
        assert!(!ledger_closes(&streamed.levels));
        if refine_steps > 0 {
            assert!(streamed.refine_calls > 0);
        }
        replay_breakpoint_prunes(&compact, &streamed);
    }
}

#[test]
fn cpu_breakpoint_refinement_on_real_cora_fixed_node_workloads() {
    let backend = ny_bbe::CpuF32Backend::with_threads(1);
    let mut improved = 0;
    for (tag, classes) in [("img321", &[4, 2][..]), ("img141", &[0][..])] {
        let fixture = load_p1(tag);
        let (net, input) = p1_setup(&fixture);
        let frame = FrozenDualFrame::new(&net, input, fixture.label).expect("real Cora frame");
        for &class in classes {
            let (baseline, refined) =
                compare_real_cora_breakpoint(&net, &frame, tag, class, 5_000, &backend);
            let prior_engine_nodes = match (tag, class) {
                ("img321", 4) => 70,
                ("img321", 2) => 262,
                ("img141", 0) => 720,
                _ => unreachable!("the three fixed Cora controls"),
            };
            assert!(
                refined.nodes <= prior_engine_nodes,
                "{tag} class {class}: cumulative search change regressed from the prior \
                 four-step engine's {prior_engine_nodes} nodes to {}",
                refined.nodes,
            );
            if tag == "img141" && class == 0 {
                assert!(
                    refined.nodes < baseline.nodes && refined.refine_prunes > 0,
                    "the measured Cora improvement must persist through actual certified prunes"
                );
            }
            improved += usize::from(refined.nodes < baseline.nodes);
        }
    }
    assert!(
        improved > 0,
        "the real workload set must include an improvement"
    );
    eprintln!("[breakpoint-dual-cora] improved_workloads={improved}/3");
}

/// The larger real Cora objective, a 100,000-node comparison on the tracked
/// P1 fixture: compiled only for release test builds (not debug-affordable)
/// rather than `#[ignore]`d.
#[cfg(not(debug_assertions))]
#[test]
fn cpu_breakpoint_refinement_on_larger_real_cora_fixed_node_workload() {
    let fixture = load_p1("img321");
    let (net, input) = p1_setup(&fixture);
    let frame = FrozenDualFrame::new(&net, input, fixture.label).expect("real Cora frame");
    compare_real_cora_breakpoint(
        &net,
        &frame,
        "img321",
        3,
        100_000,
        &ny_bbe::CpuF32Backend::with_threads(1),
    );
}

fn compare_real_cora_breakpoint(
    net: &TwinNet,
    frame: &FrozenDualFrame,
    tag: &str,
    class: usize,
    node_cap: u64,
    backend: &ny_bbe::CpuF32Backend,
) -> (SearchReport, SearchReport) {
    let state = FrozenDualState::build(net, frame, class, &FrozenDualOptions::default())
        .expect("real Cora state");
    for point in [
        frame.lo.clone(),
        frame.hi.clone(),
        frame
            .lo
            .iter()
            .zip(&frame.hi)
            .map(|(&l, &h)| f64::midpoint(l, h))
            .collect(),
    ] {
        let checked = check_point(net, frame, &state, &point).expect("containment replay");
        assert!(
            checked.holds(),
            "{tag} class {class}: state lost network containment"
        );
    }
    let order = static_order(&state);
    let problem = FrozenDualProblem::new(&state, &order).expect("real Cora dual problem");
    assert!(
        problem.root_bound() < 0.0,
        "{tag} class {class}: root already closes"
    );
    let options = SearchOptions {
        node_cap,
        record_prunes: node_cap as usize,
        // Work is bounded by identical node caps, independent of host load.
        deadline: None,
        ..SearchOptions::default()
    };
    let baseline = search_with_refinement_backend(&problem, &options, None);
    let refined = search_with_refinement_backend(
        &problem,
        &SearchOptions {
            refine_steps: 4,
            ..options
        },
        Some(backend),
    );
    eprintln!(
        "[breakpoint-dual-cora] tag={tag} class={class} baseline_nodes={} \
                 refined_nodes={} baseline={:?} refined={:?} calls={} accepts={} new_prunes={}",
        baseline.nodes,
        refined.nodes,
        baseline.verdict,
        refined.verdict,
        refined.refine_calls,
        refined.refine_accepted,
        refined.refine_prunes,
    );
    assert!(
        baseline.closed(),
        "{tag} class {class}: positive control stopped open"
    );
    assert!(
        refined.closed(),
        "{tag} class {class}: refinement lost a closure"
    );
    assert!(ledger_closes(&refined.levels));
    assert!(
        refined.refine_calls > 0,
        "{tag} class {class}: engine did not run"
    );
    assert_eq!(refined.crosscheck_failures, 0);
    replay_breakpoint_prunes(&problem, &refined);
    (baseline, refined)
}

#[test]
fn frozen_dual_bab_never_closes_a_cora_sat_class() {
    never_closes_a_cora_sat_class(false);
}

#[test]
fn cpu_breakpoint_refinement_never_closes_a_witnessed_cora_sat_class() {
    never_closes_a_cora_sat_class(true);
}

fn never_closes_a_cora_sat_class(refine: bool) {
    let fx = load_p1("img141");
    let (net, input) = p1_setup(&fx);
    let centre: Vec<f64> = input
        .lo
        .iter()
        .zip(&input.hi)
        .map(|(l, h)| f64::midpoint(*l, *h))
        .collect();
    let vals = dd_forward(&net, &centre).expect("dd forward");
    let logits: Vec<f64> = vals[net.ops.len()].iter().map(|v| v.to_f64()).collect();
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
    let (top, second) = (idx[0], idx[1]);
    let margin = (vals[net.ops.len()][second] - vals[net.ops.len()][top]).to_f64();
    assert!(
        margin < 0.0,
        "VOID: the centre is not a counterexample for the swapped target ({margin:e})"
    );
    // The runner-up as the "true" class: `top` is a witnessed sat class.
    let frame = FrozenDualFrame::new(&net, input, second).expect("frame");
    let st =
        FrozenDualState::build(&net, &frame, top, &FrozenDualOptions::default()).expect("state");
    let pc = check_point(&net, &frame, &st, &centre).expect("check");
    assert!(
        pc.holds(),
        "the frozen dual state does not contain the cora centre point: {pc:?}"
    );
    assert!(pc.margin < 0.0);
    let order = static_order(&st);
    let prob = FrozenDualProblem::new(&st, &order).expect("problem");
    let backend = ny_bbe::CpuF32Backend::with_threads(1);
    let rep = search_with_refinement_backend(
        &prob,
        &SearchOptions {
            node_cap: if refine { 256 } else { 3_000 },
            record_prunes: 3_000,
            refine_steps: if refine { 4 } else { 0 },
            ..SearchOptions::default()
        },
        refine.then_some(&backend as &dyn ny_core::search::SearchBackend),
    );
    assert!(
        !rep.closed(),
        "SOUNDNESS EMERGENCY: the search closed a class with a witnessed counterexample \
         (margin {margin:e} at the image centre) after {} nodes",
        rep.nodes
    );
    if refine {
        assert!(
            rep.refine_calls > 0,
            "real negative control did not engage refinement"
        );
        assert!(!ledger_closes(&rep.levels));
        replay_breakpoint_prunes(&prob, &rep);
        eprintln!(
            "[breakpoint-dual-cora-sat] witness_margin={margin:e} nodes={} calls={} accepts={} \
             new_prunes={} verdict={:?}",
            rep.nodes, rep.refine_calls, rep.refine_accepted, rep.refine_prunes, rep.verdict,
        );
    }
}

// ---------------------------------------------------------------------------
// #wide-fd-search (ENGINE_PROGRAM M4): the typed key and the REAL ny-bbe CPU
// f32 level search, certified by the tier, on the real cora network
// ---------------------------------------------------------------------------

use ny_propagate::margin_row::zono_bab::search_with_backend;
use ny_propagate::margin_row::FrozenDualSearch;

#[test]
fn frozen_dual_bab_search_is_dark_without_the_key_and_typed_with_it() {
    let decl = &ny_levers::decls::margin_row::MARGIN_ROW_FROZEN_DUAL_BAB_SEARCH;
    let dark = projected_levers(&crate::preset::PresetConfig::default());
    let resolved = dark.resolved(decl).expect("the lever is declared");
    assert_eq!(resolved.value.as_str(), Some(FrozenDualSearch::F64.token()));
    assert_eq!(resolved.source, ny_levers::Source::Default);
    for search in [
        FrozenDualSearch::CpuF32,
        FrozenDualSearch::Cuda,
        FrozenDualSearch::Auto,
    ] {
        let mut preset = crate::preset::PresetConfig::default();
        preset.margin_row.frozen_dual_bab = Some(true);
        preset.margin_row.frozen_dual_bab_search = Some(search);
        let typed = projected_levers(&preset);
        let resolved = typed.resolved(decl).expect("the lever is declared");
        assert_eq!(resolved.value.as_str(), Some(search.token()));
        assert_eq!(resolved.source, ny_levers::Source::Config);
        assert_ne!(resolved.source, ny_levers::Source::LegacyEnv);
        assert_eq!(
            resolved
                .value
                .as_str()
                .and_then(FrozenDualSearch::from_token),
            Some(search),
            "the scored value must parse back into the tier's selector"
        );
    }
}

/// The P1 objectives the debug build can afford, as `(tag, problem, replica
/// nodes)`.
fn p1_problems(tag: &str, max_nodes: u64) -> Vec<(String, FrozenDualProblem, u64)> {
    let fx = load_p1(tag);
    let (net, input) = p1_setup(&fx);
    let frame = FrozenDualFrame::new(&net, input, fx.label).expect("frame");
    fx.objs
        .iter()
        .filter(|o| o.nodes <= max_nodes)
        .map(|o| {
            let st = FrozenDualState::build(&net, &frame, o.j, &FrozenDualOptions::default())
                .expect("state");
            let order = static_order(&st);
            (
                format!("{tag} obj {} class {}", o.obj, o.j),
                FrozenDualProblem::new(&st, &order).expect("problem"),
                o.nodes,
            )
        })
        .collect()
}

#[test]
fn real_cpu_f32_search_closes_the_p1_objectives_and_certifies_every_prune() {
    let backend = ny_bbe::CpuF32Backend::new();
    let mut problems = p1_problems("img141", 1_000);
    problems.extend(p1_problems("img321", 1_000));
    assert!(
        problems.len() >= 3,
        "only {} P1 objectives: VOID",
        problems.len()
    );
    let mut scratch = NodeScratch::default();
    let (mut claimed, mut refuted, mut prunes) = (0u64, 0u64, 0usize);
    for (name, prob, replica_nodes) in &problems {
        let opts = SearchOptions {
            node_cap: (20 * replica_nodes).max(5_000),
            record_prunes: 512,
            ..SearchOptions::default()
        };
        let f64_path = search(prob, &opts);
        let wide = search_with_backend(prob, &opts, &backend);
        eprintln!(
            "[fd-wide-p1] {name} f64: {:?} nodes={} prunes={} | cpu_f32: {:?} nodes={} \
             prunes={} claimed={} refuted={} probe={} fallback={} secs={:.2}",
            f64_path.verdict,
            f64_path.nodes,
            f64_path.certified_prunes,
            wide.verdict,
            wide.nodes,
            wide.certified_prunes,
            wide.claimed,
            wide.refuted,
            wide.probe,
            wide.fallback_chunks,
            wide.secs
        );
        assert_eq!(
            (wide.search, wide.probe, wide.fallback_chunks),
            ("cpu_f32", "ok", 0),
            "{name}: VOID — the real CPU f32 backend did not search"
        );
        assert!(
            wide.closed(),
            "{name}: the f64 search closes this objective ({} nodes) but the CPU f32 search \
             did not ({:?} after {} nodes)",
            f64_path.nodes,
            wide.verdict,
            wide.nodes
        );
        for p in &wide.pruned {
            let again = prob.eval_node(&p.sides, &p.l0, &p.l1, &mut scratch, None);
            assert!(again.prunes(), "{name}: a recorded prune does not certify");
            assert_eq!(again.certified.to_bits(), p.certified.to_bits(), "{name}");
            prunes += 1;
        }
        claimed += wide.claimed;
        refuted += wide.refuted;
    }
    eprintln!("[fd-wide-p1] claimed={claimed} refuted={refuted} prunes_rechecked={prunes}");
    assert!(claimed > 0 && prunes > 0, "VOID");
}

#[test]
fn real_cpu_f32_search_never_closes_a_cora_sat_class() {
    let fx = load_p1("img141");
    let (net, input) = p1_setup(&fx);
    let centre: Vec<f64> = input
        .lo
        .iter()
        .zip(&input.hi)
        .map(|(l, h)| f64::midpoint(*l, *h))
        .collect();
    let vals = dd_forward(&net, &centre).expect("dd forward");
    let logits: Vec<f64> = vals[net.ops.len()].iter().map(|v| v.to_f64()).collect();
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
    let (top, second) = (idx[0], idx[1]);
    let frame = FrozenDualFrame::new(&net, input, second).expect("frame");
    let st =
        FrozenDualState::build(&net, &frame, top, &FrozenDualOptions::default()).expect("state");
    let order = static_order(&st);
    let prob = FrozenDualProblem::new(&st, &order).expect("problem");
    let rep = search_with_backend(
        &prob,
        &SearchOptions {
            node_cap: 3_000,
            ..SearchOptions::default()
        },
        &ny_bbe::CpuF32Backend::new(),
    );
    assert_eq!(rep.search, "cpu_f32", "VOID: the backend did not search");
    assert!(
        !rep.closed(),
        "SOUNDNESS EMERGENCY: the CPU f32 search closed a class with a witnessed \
         counterexample after {} nodes",
        rep.nodes
    );
}

#[test]
fn real_cpu_f32_search_is_thread_count_independent() {
    let problems = p1_problems("img141", 1_000);
    let (name, prob, _) = problems.first().expect("a P1 objective");
    let opts = SearchOptions {
        node_cap: 20_000,
        record_prunes: 64,
        ..SearchOptions::default()
    };
    let digest = |r: &SearchReport| {
        (
            r.verdict,
            r.nodes,
            r.depth,
            r.certified_prunes,
            r.levels.clone(),
            r.claimed,
            r.refuted,
            r.cert_min_slack.to_bits(),
            r.pruned
                .iter()
                .map(|p| (p.sides.clone(), p.certified.to_bits()))
                .collect::<Vec<_>>(),
        )
    };
    let one = search_with_backend(prob, &opts, &ny_bbe::CpuF32Backend::with_threads(1));
    assert_eq!(one.search, "cpu_f32", "VOID: {name}");
    for threads in [2usize, 4, 7] {
        let many = search_with_backend(prob, &opts, &ny_bbe::CpuF32Backend::with_threads(threads));
        assert_eq!(
            digest(&one),
            digest(&many),
            "{name}: {threads} search threads changed the certified search"
        );
    }
}

/// The replica's open objectives as a tier's inputs: their classes (the
/// fixture lists exactly the objectives the replica's root left open), and a
/// lane root per `others` entry — the replica's CROWN root for an open class,
/// NaN otherwise (the tier reads it for open classes only, to order them).
fn p1_open_objectives(fx: &P1Fixture) -> (Vec<usize>, Vec<f64>) {
    let open: Vec<usize> = fx.objs.iter().map(|o| o.j).collect();
    let dj: Vec<f64> = fx
        .others
        .iter()
        .map(|&c| {
            fx.objs
                .iter()
                .find(|o| o.j == c)
                .map_or(f64::NAN, |o| o.crown_root)
        })
        .collect();
    (open, dj)
}

/// Every class in `closed` holds on the real cora network at the centre of
/// `[lo, hi]` and at 32 deterministic points of it (double-double margins
/// `y_label - y_j`).
fn assert_closed_classes_hold(
    net: &TwinNet,
    lo: &[f64],
    hi: &[f64],
    label: usize,
    closed: &[usize],
    what: &str,
) {
    let mut points: Vec<Vec<f64>> = vec![lo
        .iter()
        .zip(hi)
        .map(|(l, h)| f64::midpoint(*l, *h))
        .collect()];
    let mut state = 0x5EED_F00D_u64;
    for _ in 0..32 {
        points.push(
            lo.iter()
                .zip(hi)
                .map(|(l, h)| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    #[allow(clippy::cast_precision_loss)]
                    let u = (state >> 11) as f64 / (1u64 << 53) as f64;
                    (l + u * (h - l)).clamp(*l, *h)
                })
                .collect(),
        );
    }
    for &j in closed {
        for (k, x) in points.iter().enumerate() {
            let vals = dd_forward(net, x).expect("dd forward");
            let logits = &vals[net.ops.len()];
            let m = (logits[label] - logits[j]).to_f64();
            assert!(
                m > 0.0,
                "SOUNDNESS EMERGENCY: {what} closed class {j}, but point {k} has margin {m:e}"
            );
        }
    }
}

/// A bounded production-route fixture: two inputs, a stable trunk ReLU,
/// three unstable head ReLUs, five generators and nine open objectives.
/// Output 0 is f=.05+(x+y)/4+relu(x)+relu(y)+relu(-x-y), with minimum .05.
/// Output 1 is .5*f+.1 (a violation at the origin); outputs 2..9 are constants.
/// The real Cora P1 leg below retains the large-network claim/control gate.
fn registered_tier_lane_network() -> TwinNet {
    use ny_propagate::margin_row::{TwinOpSpec, TwinSpec};
    let mut output_weight = vec![1.0, 1.0, 1.0, 0.25, 0.25];
    output_weight.extend_from_slice(&[0.5, 0.5, 0.5, 0.125, 0.125]);
    output_weight.resize(10 * 5, 0.0);
    let mut output_bias = vec![-0.95, -0.375, 0.0];
    output_bias.resize(10, -0.05);
    TwinNet::compile(&TwinSpec {
        n_in: 2,
        ops: vec![
            TwinOpSpec::ChannelAffine {
                input: 0,
                scale: vec![1.0, 1.0],
                shift: vec![2.0, 2.0],
                scale_rel_err: 0.0,
                shift_err: vec![0.0, 0.0],
                shape: (2, 1, 1),
            },
            TwinOpSpec::Relu { input: 1 },
            TwinOpSpec::Gemm {
                input: 2,
                weight: vec![1.0, 0.0, 0.0, 1.0, -1.0, -1.0, 1.0, 0.0, 0.0, 1.0],
                bias: vec![-2.0, -2.0, 4.0, 0.0, 0.0],
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
                shape: (5, 2),
            },
            TwinOpSpec::Relu { input: 3 },
            TwinOpSpec::Gemm {
                input: 4,
                weight: output_weight,
                bias: output_bias,
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
                shape: (10, 5),
            },
        ],
    })
    .expect("compiled nine-objective lane fixture")
}

#[test]
fn registered_cpu_f32_backend_reaches_the_armed_tier() {
    // Two frames, one registered backend. (1) The full production route,
    // RootGates -> root_eval -> run_tier, on a bounded compiled topology with
    // nine open objectives and a witnessed SAT objective. Its two inputs and
    // three unstable ReLUs keep state construction below every fair share.
    // The broad Cora lane-root frame previously spent all nine shares building
    // and ranking states before any backend upload; its 120-second debug wall
    // was a workload gate, not a reliable registration/routing fixture.
    // (2) The REAL Cora P1 certified arm-A frame (class 0: 724 replica nodes,
    // first certified prunes at depth 4), unchanged: the registered backend
    // must search it AND claim, and every closed class is checked on-network.
    use ny_propagate::margin_row::engine::BackwardEngine;
    use ny_propagate::margin_row::zono_bab::{run_tier, run_tier_on_frame, TierPolicy, TierReport};
    use ny_propagate::margin_row::{FdBackendKind, RootGates, RoundMode};
    // Registered exactly as `main.rs` registers it (first install wins).
    ny_propagate::margin_row::set_frozen_dual_search_backend_factory(FdBackendKind::CpuF32, || {
        Some(std::sync::Arc::new(ny_bbe::CpuF32Backend::new())
            as std::sync::Arc<dyn ny_core::search::SearchBackend>)
    });
    let fx = load_p1("img141");
    let (net, input) = p1_setup(&fx);
    let (lo, hi) = (input.lo.clone(), input.hi.clone());
    let policy = TierPolicy::clamped(2_000, 30.0);
    // The routing leg has bounded work, rather than a larger timeout.
    let lane_policy = TierPolicy::clamped(2_000, 30.0);
    // The classes the armed tier searched on the backend (probe ok, no chunk
    // fell back to the f64 search).
    let searched = |armed: &TierReport, leg: &str| {
        let mut searched = 0usize;
        for class in &armed.classes {
            let Ok(r) = &class.result else { continue };
            if r.search.search == "cpu_f32" {
                searched += 1;
                assert_eq!(r.search.probe, "ok", "{leg}: class {}", class.class);
                assert_eq!(r.search.fallback_chunks, 0, "{leg}: class {}", class.class);
            }
        }
        searched
    };

    // (1) The lane root, with the production builder and evaluator.
    let lane_net = registered_tier_lane_network();
    let lane_lo = [-1.0, -1.0];
    let lane_hi = [1.0, 1.0];
    let lane_adv: Vec<usize> = (1..10).collect();
    let mut root = RootGates::build_prec(
        &lane_net,
        &lane_lo,
        &lane_hi,
        RoundMode::Outward,
        None,
        false,
    )
    .expect("lane root");
    root.head_alpha = Some(vec![0.5; lane_net.n_y]);
    let eng = BackwardEngine::new(&lane_net, &root);
    let re = ny_propagate::margin_row::bab::root_eval(&eng, &lane_net, 0, &lane_adv)
        .expect("lane root eval");
    assert_eq!(
        re.tree_classes, lane_adv,
        "VOID: every one of the nine objectives must reach the tier"
    );
    let lane = |policy: &TierPolicy| {
        run_tier(
            &lane_net,
            &root,
            0,
            &lane_adv,
            &re.dj,
            &re.tree_classes,
            &re.root_ybox(),
            None,
            policy,
        )
    };
    let lane_base = lane(&lane_policy);
    let lane_armed = lane(&lane_policy.with_search(FrozenDualSearch::CpuF32));
    let lane_searched = searched(&lane_armed, "lane root");
    assert_eq!(
        lane_base.classes_built, 9,
        "every baseline objective builds"
    );
    assert_eq!(lane_armed.classes_built, 9, "every armed objective builds");
    assert_eq!(lane_searched, 9, "every objective reaches CPU level search");
    for class in &lane_armed.classes {
        let run = class.result.as_ref().expect("armed lane class built");
        assert_eq!((run.state_gens, run.unstable, run.depth_cap), (5, 3, 3));
        assert!(run.search.nodes > 0, "class {} did not search", class.class);
    }
    let witness = dd_forward(&lane_net, &[0.0, 0.0]).expect("lane SAT witness");
    let logits = &witness[lane_net.ops.len()];
    assert!((logits[0] - logits[1]).to_f64() < 0.0);
    assert!(
        !lane_base.closed.contains(&1),
        "baseline closed witnessed SAT class"
    );
    assert!(
        !lane_armed.closed.contains(&1),
        "backend closed witnessed SAT class"
    );
    assert_closed_classes_hold(
        &lane_net,
        &lane_lo,
        &lane_hi,
        0,
        &lane_armed.closed,
        "the cpu_f32 tier (lane root)",
    );

    // (2) The P1 replica's frame.
    let frame = FrozenDualFrame::new(&net, input, fx.label).expect("frame");
    let (open, dj) = p1_open_objectives(&fx);
    assert!(
        !open.is_empty(),
        "VOID: the P1 fixture has no open objective"
    );
    let p1 = |policy: &TierPolicy| {
        run_tier_on_frame(&net, frame.clone(), &fx.others, &dj, &open, None, policy)
    };
    let p1_base = p1(&policy);
    let p1_armed = p1(&policy.with_search(FrozenDualSearch::CpuF32));
    let p1_searched = searched(&p1_armed, "P1 frame");
    assert_closed_classes_hold(
        &net,
        &lo,
        &hi,
        fx.label,
        &p1_armed.closed,
        "the cpu_f32 tier (P1 frame)",
    );

    eprintln!(
        "[fd-wide-tier] lane_root: classes={} searched={lane_searched} closed_f64={:?} \
         closed_cpu_f32={:?} claimed={} refuted={} | p1_frame: classes={} \
         searched={p1_searched} closed_f64={:?} closed_cpu_f32={:?} claimed={} refuted={}",
        lane_armed.classes.len(),
        lane_base.closed,
        lane_armed.closed,
        lane_armed.claimed,
        lane_armed.refuted,
        p1_armed.classes.len(),
        p1_base.closed,
        p1_armed.closed,
        p1_armed.claimed,
        p1_armed.refuted
    );
    assert!(
        lane_searched > 0,
        "VOID: the registered CPU f32 backend never searched a class of the armed tier \
         (lane root)"
    );
    assert!(
        p1_searched > 0,
        "VOID: the registered CPU f32 backend never searched a class of the armed tier \
         (P1 frame)"
    );
    assert!(p1_armed.claimed > 0, "VOID: nothing was claimed (P1 frame)");
}

#[test]
fn registered_cpu_f32_backend_searches_the_shared_trunk_states() {
    // M3 x M4 on the real cora net: `shared` builds the open classes' states
    // over ONE trunk before the fair shares, and `cpu_f32` scores their levels
    // on the REAL ny-bbe backend (registered as `main.rs` registers it). The
    // trunk decides each class's state and the backend only proposes, so the
    // state side must be the shared f64 tier's bit for bit; the backend must
    // search; and a class the composed tier closes must hold at the centre and
    // at sampled points. Two frames of img321: (1) the lane root's (nine open
    // classes near -1.3e3 .. -1.8e3 at the ENG3 gate, `certified_prunes=0`,
    // `claimed=0`: the composition, not the claims); (2) the P1 replica's
    // certified arm-A frame over the four objectives the replica leaves open
    // (classes 2..5; the replica's per-class traces first prune at depths 5,
    // 5, 1 and 8, and the trunk's anchor candidate IS the hardest class's own
    // state; img141 leaves one, which shares nothing), where the claim guard
    // lives.
    use ny_propagate::margin_row::engine::BackwardEngine;
    use ny_propagate::margin_row::zono_bab::{
        run_tier, run_tier_on_frame, ClassRun, FrozenDualStateMode, TierPolicy, TierReport,
    };
    use ny_propagate::margin_row::{FdBackendKind, RootGates, RoundMode};
    use ny_propagate::zono_core::ZonoRefusal;
    ny_propagate::margin_row::set_frozen_dual_search_backend_factory(FdBackendKind::CpuF32, || {
        Some(std::sync::Arc::new(ny_bbe::CpuF32Backend::new())
            as std::sync::Arc<dyn ny_core::search::SearchBackend>)
    });
    let fx = load_p1("img321");
    let (net, input) = p1_setup(&fx);
    let (lo, hi) = (input.lo.clone(), input.hi.clone());
    // A long wall so the node cap, not a debug-build share, ends each class.
    let shared = TierPolicy::clamped(2_000, 120.0).with_state(FrozenDualStateMode::Shared);
    // The shared f64 tier `base` and the shared cpu_f32 tier `armed` over one
    // frame: the state side agrees bit for bit; returns the classes the
    // backend searched.
    let compare = |base: &TierReport, armed: &TierReport, leg: &str| {
        assert_eq!(
            (base.state, armed.state),
            (FrozenDualStateMode::Shared, FrozenDualStateMode::Shared),
            "{leg}"
        );
        assert_eq!(
            base.trunk_lambda, armed.trunk_lambda,
            "{leg}: the trunk's slope choice depends on the search"
        );
        assert_eq!(
            base.shared_root_loss_max.map(f64::to_bits),
            armed.shared_root_loss_max.map(f64::to_bits),
            "{leg}"
        );
        assert_eq!(
            base.classes.len(),
            armed.classes.len(),
            "{leg}: both tiers report every open class"
        );
        let mut searched = 0usize;
        for (a, b) in base.classes.iter().zip(&armed.classes) {
            assert_eq!(a.class, b.class, "{leg}: the tier order");
            let (x, y) = match (&a.result, &b.result) {
                (Ok(x), Ok(y)) => (x, y),
                // The state side cannot depend on the search, so a class is
                // built on both paths or refused on both for the same reason;
                // only a share that expired at a different moment on the two
                // runs (a debug build on the real net) may split them. Never
                // skip a divergence silently.
                (p, q) => {
                    let why = |r: &Result<ClassRun, ZonoRefusal>| {
                        r.as_ref().err().map(ZonoRefusal::token)
                    };
                    let (p, q) = (why(p), why(q));
                    assert!(
                        p == q || p == Some("deadline") || q == Some("deadline"),
                        "{leg}: class {}: the shared state was built or refused depending on \
                         the search (f64 {p:?}, cpu_f32 {q:?})",
                        a.class
                    );
                    continue;
                }
            };
            assert_eq!(
                (x.state, y.state),
                (FrozenDualStateMode::Shared, FrozenDualStateMode::Shared),
                "{leg}"
            );
            assert_eq!(
                (
                    x.root_g.to_bits(),
                    x.root_loss.map(f64::to_bits),
                    x.depth_cap
                ),
                (
                    y.root_g.to_bits(),
                    y.root_loss.map(f64::to_bits),
                    y.depth_cap
                ),
                "{leg}: class {}: the class's shared state depends on the search",
                a.class
            );
            if y.search.search == "cpu_f32" {
                searched += 1;
                assert_eq!(y.search.probe, "ok", "{leg}: class {}", a.class);
                assert_eq!(y.search.fallback_chunks, 0, "{leg}: class {}", a.class);
            }
        }
        searched
    };

    // (1) The lane root.
    let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
    let eng = BackwardEngine::new(&net, &root);
    let re = ny_propagate::margin_row::bab::root_eval(&eng, &net, fx.label, &fx.others)
        .expect("root eval");
    assert!(
        re.tree_classes.len() >= 2,
        "VOID: a shared trunk needs several open classes (open: {:?})",
        re.tree_classes
    );
    let lane = |policy: &TierPolicy| {
        run_tier(
            &net,
            &root,
            fx.label,
            &fx.others,
            &re.dj,
            &re.tree_classes,
            &re.root_ybox(),
            None,
            policy,
        )
    };
    let lane_base = lane(&shared);
    let lane_armed = lane(&shared.with_search(FrozenDualSearch::CpuF32));
    let lane_searched = compare(&lane_base, &lane_armed, "lane root");
    assert_closed_classes_hold(
        &net,
        &lo,
        &hi,
        fx.label,
        &lane_armed.closed,
        "the shared-trunk cpu_f32 tier (lane root)",
    );

    // (2) The P1 replica's frame.
    let frame = FrozenDualFrame::new(&net, input, fx.label).expect("frame");
    let (open, dj) = p1_open_objectives(&fx);
    assert!(
        open.len() >= 2,
        "VOID: a shared trunk needs several open classes (P1 open: {open:?})"
    );
    let p1 = |policy: &TierPolicy| {
        run_tier_on_frame(&net, frame.clone(), &fx.others, &dj, &open, None, policy)
    };
    let p1_base = p1(&shared);
    let p1_armed = p1(&shared.with_search(FrozenDualSearch::CpuF32));
    let p1_searched = compare(&p1_base, &p1_armed, "P1 frame");
    assert_closed_classes_hold(
        &net,
        &lo,
        &hi,
        fx.label,
        &p1_armed.closed,
        "the shared-trunk cpu_f32 tier (P1 frame)",
    );

    eprintln!(
        "[fd-wide-shared-tier] lane_root: classes={} searched={lane_searched} lambda={} \
         closed_f64={:?} closed_cpu_f32={:?} claimed={} refuted={} | p1_frame: classes={} \
         searched={p1_searched} lambda={} closed_f64={:?} closed_cpu_f32={:?} claimed={} \
         refuted={}",
        lane_armed.classes.len(),
        lane_armed.trunk_lambda,
        lane_base.closed,
        lane_armed.closed,
        lane_armed.claimed,
        lane_armed.refuted,
        p1_armed.classes.len(),
        p1_armed.trunk_lambda,
        p1_base.closed,
        p1_armed.closed,
        p1_armed.claimed,
        p1_armed.refuted
    );
    assert!(
        lane_searched > 0,
        "VOID: the registered CPU f32 backend never searched a shared-trunk class (lane root)"
    );
    assert!(
        p1_searched > 0,
        "VOID: the registered CPU f32 backend never searched a shared-trunk class (P1 frame)"
    );
    assert!(p1_armed.claimed > 0, "VOID: nothing was claimed (P1 frame)");
}

// ---------------------------------------------------------------------------
//  #wide-fd-search on the DEVICE (ny-cuda): the same lane, the same tree
// ---------------------------------------------------------------------------
//
// The device kernels reproduce the CPU f32 search bit for bit (the KATs in
// `cuda_fd_search_tests`), so the tier searching on `cuda` must build the SAME
// tree as on `cpu_f32` — same nodes, levels, claims, refutations and certified
// prunes — and every prune must re-certify in f64. Gated on an advertised
// device: elsewhere each test prints SKIP=VOID and proves nothing.

#[cfg(feature = "cuda")]
mod cuda_wide {
    use super::super::cuda_fd_search_tests::search_device;
    use super::*;
    use ny_propagate::margin_row::zono_bab::SearchReport;

    type Digest = (
        ny_propagate::margin_row::zono_bab::SearchVerdict,
        u64,
        usize,
        u64,
        Vec<ny_propagate::margin_row::zono_bab::LevelTrace>,
        u64,
        u64,
        u64,
        Vec<(Vec<ny_propagate::margin_row::zono_state::Side>, u64)>,
    );

    fn digest(r: &SearchReport) -> Digest {
        (
            r.verdict,
            r.nodes,
            r.depth,
            r.certified_prunes,
            r.levels.clone(),
            r.claimed,
            r.refuted,
            r.cert_min_slack.to_bits(),
            r.pruned
                .iter()
                .map(|p| (p.sides.clone(), p.certified.to_bits()))
                .collect(),
        )
    }

    #[test]
    fn real_cuda_search_is_the_cpu_f32_search_on_the_p1_objectives() {
        let test = "real_cuda_search_is_the_cpu_f32_search_on_the_p1_objectives";
        let Some(dev) = search_device(test) else {
            return;
        };
        let cpu = ny_bbe::CpuF32Backend::new();
        let mut problems = p1_problems("img141", 1_000);
        problems.extend(p1_problems("img321", 1_000));
        assert!(
            problems.len() >= 3,
            "only {} P1 objectives: VOID",
            problems.len()
        );
        let mut scratch = NodeScratch::default();
        let mut prunes = 0usize;
        for (name, prob, replica_nodes) in &problems {
            let opts = SearchOptions {
                node_cap: (20 * replica_nodes).max(5_000),
                record_prunes: 512,
                ..SearchOptions::default()
            };
            let on_cpu = search_with_backend(prob, &opts, &cpu);
            let on_dev = search_with_backend(prob, &opts, &dev);
            eprintln!(
                "[fd-wide-cuda-p1] {name} cpu_f32: {:?} nodes={} claimed={} refuted={} | cuda: \
                 {:?} nodes={} claimed={} refuted={} probe={} fallback={} secs={:.2}",
                on_cpu.verdict,
                on_cpu.nodes,
                on_cpu.claimed,
                on_cpu.refuted,
                on_dev.verdict,
                on_dev.nodes,
                on_dev.claimed,
                on_dev.refuted,
                on_dev.probe,
                on_dev.fallback_chunks,
                on_dev.secs
            );
            assert_eq!(
                (on_dev.search, on_dev.probe, on_dev.fallback_chunks),
                ("cuda", "ok", 0),
                "{name}: VOID — the device backend did not search"
            );
            assert_eq!(
                digest(&on_cpu),
                digest(&on_dev),
                "{name}: the device search built a different tree than the CPU f32 search"
            );
            for p in &on_dev.pruned {
                let again = prob.eval_node(&p.sides, &p.l0, &p.l1, &mut scratch, None);
                assert!(again.prunes(), "{name}: a recorded prune does not certify");
                assert_eq!(again.certified.to_bits(), p.certified.to_bits(), "{name}");
                prunes += 1;
            }
        }
        assert!(prunes > 0, "VOID: no prune was re-certified");
    }

    #[test]
    fn real_cuda_search_never_closes_a_cora_sat_class() {
        let test = "real_cuda_search_never_closes_a_cora_sat_class";
        let Some(dev) = search_device(test) else {
            return;
        };
        let fx = load_p1("img141");
        let (net, input) = p1_setup(&fx);
        let centre: Vec<f64> = input
            .lo
            .iter()
            .zip(&input.hi)
            .map(|(l, h)| f64::midpoint(*l, *h))
            .collect();
        let vals = dd_forward(&net, &centre).expect("dd forward");
        let logits: Vec<f64> = vals[net.ops.len()].iter().map(|v| v.to_f64()).collect();
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
        let (top, second) = (idx[0], idx[1]);
        let frame = FrozenDualFrame::new(&net, input, second).expect("frame");
        let st = FrozenDualState::build(&net, &frame, top, &FrozenDualOptions::default())
            .expect("state");
        let order = static_order(&st);
        let prob = FrozenDualProblem::new(&st, &order).expect("problem");
        let rep = search_with_backend(
            &prob,
            &SearchOptions {
                node_cap: 3_000,
                ..SearchOptions::default()
            },
            &dev,
        );
        assert_eq!(
            rep.search, "cuda",
            "VOID: the device backend did not search"
        );
        assert!(
            !rep.closed(),
            "SOUNDNESS EMERGENCY: the device search closed a class with a witnessed \
             counterexample after {} nodes",
            rep.nodes
        );
    }

    #[test]
    fn registered_cuda_backend_reaches_the_armed_tier() {
        use ny_propagate::margin_row::engine::BackwardEngine;
        use ny_propagate::margin_row::zono_bab::{run_tier, TierPolicy};
        use ny_propagate::margin_row::{FdBackendKind, RootGates, RoundMode};
        let test = "registered_cuda_backend_reaches_the_armed_tier";
        if search_device(test).is_none() {
            return;
        }
        // Registered exactly as `main.rs` registers them (first install wins).
        ny_propagate::margin_row::set_frozen_dual_search_backend_factory(
            FdBackendKind::Cuda,
            || {
                Some(
                    std::sync::Arc::new(ny_cuda::search::CudaSearchBackend::new())
                        as std::sync::Arc<dyn ny_core::search::SearchBackend>,
                )
            },
        );
        ny_propagate::margin_row::set_frozen_dual_search_backend_factory(
            FdBackendKind::CpuF32,
            || {
                Some(std::sync::Arc::new(ny_bbe::CpuF32Backend::new())
                    as std::sync::Arc<dyn ny_core::search::SearchBackend>)
            },
        );
        let fx = load_p1("img141");
        let (net, input) = p1_setup(&fx);
        let root =
            RootGates::build(&net, &input.lo, &input.hi, RoundMode::Outward, None).expect("root");
        let eng = BackwardEngine::new(&net, &root);
        let re = ny_propagate::margin_row::bab::root_eval(&eng, &net, fx.label, &fx.others)
            .expect("root eval");
        assert!(
            !re.tree_classes.is_empty(),
            "VOID: the root closes every class, so no tier runs"
        );
        let run = |policy: &TierPolicy| {
            run_tier(
                &net,
                &root,
                fx.label,
                &fx.others,
                &re.dj,
                &re.tree_classes,
                &re.root_ybox(),
                None,
                policy,
            )
        };
        for search in [FrozenDualSearch::Cuda, FrozenDualSearch::Auto] {
            let armed = run(&TierPolicy::clamped(2_000, 30.0).with_search(search));
            let mut searched = 0usize;
            for class in &armed.classes {
                let Ok(r) = &class.result else { continue };
                if r.search.search == "cuda" {
                    searched += 1;
                    assert_eq!(r.search.probe, "ok", "{search:?} class {}", class.class);
                    assert_eq!(
                        r.search.fallback_chunks, 0,
                        "{search:?} class {}",
                        class.class
                    );
                }
            }
            eprintln!(
                "[fd-wide-cuda-tier] search={} classes={} searched={searched} closed={:?} \
                 claimed={} refuted={}",
                search.token(),
                armed.classes.len(),
                armed.closed,
                armed.claimed,
                armed.refuted
            );
            assert!(
                searched > 0,
                "VOID: the registered device backend never searched a class under {search:?}"
            );
        }
    }
}

/// Ordinary per-class states, and singleton M3 joint/anchor states, retain
/// the original class chart bit for bit. This isolates relation proposals
/// without replacing the immutable slopes
/// or comparing different relaxations. The production caller records no
/// traces: crosscheck_every=1 instead independently replays EVERY prune from
/// the original FrozenDualProblem. Exact-rational recorded Relation replay
/// is covered in ny-propagate's fd_relations_ tests.
fn compare_real_cora_relation_chart(
    net: &TwinNet,
    frame: FrozenDualFrame,
    tag: &str,
    class: usize,
    node_cap: u64,
    expected_closed: bool,
    mode: ny_propagate::margin_row::zono_bab::FrozenDualStateMode,
) {
    use ny_propagate::margin_row::zono_bab::{run_tier_on_frame, TierPolicy};
    assert!(
        frame.n_unstable <= ny_propagate::margin_row::zono_bab::FROZEN_DUAL_MAX_DEPTH,
        "pinned Cora chart must meet the full-state default frontier"
    );
    let unstable = frame.n_unstable;
    let state = FrozenDualState::build(net, &frame, class, &FrozenDualOptions::default())
        .expect("original real Cora class chart");
    let center: Vec<_> = frame
        .lo
        .iter()
        .zip(&frame.hi)
        .map(|(&l, &h)| f64::midpoint(l, h))
        .collect();
    assert!(check_point(net, &frame, &state, &center)
        .expect("original network containment")
        .holds());
    let root = state.root_bound();
    let order = static_order(&state);
    let depth = order
        .len()
        .min(ny_propagate::margin_row::zono_bab::FROZEN_DUAL_MAX_DEPTH);
    let problem = FrozenDualProblem::new(&state, &order[..depth]).expect("original problem");
    let baseline = search_with_refinement_backend(
        &problem,
        &SearchOptions {
            node_cap,
            record_prunes: node_cap as usize,
            crosscheck_every: 1,
            ..SearchOptions::default()
        },
        None,
    );
    replay_breakpoint_prunes(&problem, &baseline);
    assert_eq!(
        baseline.closed(),
        expected_closed,
        "{tag} class {class}: baseline fixture"
    );
    let (lo, hi, label, gens) = (frame.lo.clone(), frame.hi.clone(), frame.t, state.n_gens);
    drop(state);
    // This is a fixed-node internal original-data gate. Its debug-build wall
    // allowance is unrelated to the official30s instance budget in the
    // cohort manifest, and produces no wall-time/competition score claim.
    let policy = TierPolicy::clamped(node_cap, 120.0)
        .with_state(mode)
        .with_crosscheck_every(1);
    let tier = run_tier_on_frame(net, frame, &[class], &[-1.0], &[class], None, &policy);
    assert_eq!(tier.classes.len(), 1);
    assert_eq!(tier.classes_built, 1);
    let run = tier.classes[0]
        .result
        .as_ref()
        .expect("singleton production original-data class arrived");
    assert_eq!(run.state, mode);
    assert_eq!(run.unstable, unstable);
    assert!(!run.streamed_rows);
    assert_eq!((run.state_gens, run.depth_cap), (gens, depth));
    assert_eq!(
        run.root_g.to_bits(),
        root.to_bits(),
        "production slopes must equal original class chart"
    );
    assert_eq!(
        run.search.root_certified.to_bits(),
        baseline.root_certified.to_bits()
    );
    assert_eq!(
        run.search.closed(),
        expected_closed,
        "{tag} class {class}: scoped proposals changed correctness"
    );
    assert_eq!(run.search.crosscheck_failures, 0);
    assert_eq!(
        run.search.crosschecks, run.search.certified_prunes,
        "every actual production prune must independently replay at its actual multipliers"
    );
    assert_eq!(ledger_closes(&run.search.levels), expected_closed);
    let relation = run.search.relations;
    assert!(
        relation.scope_census > 0,
        "complete frame preflight must be reported"
    );
    assert!(relation.work_reserved <= 262_144);
    if relation.scope_over_cap > 0 {
        assert!(relation.scope_census > 262_144 && relation.refused > 0);
        assert_eq!(
            relation.work_reserved, 0,
            "oversized scope cannot be scanned"
        );
        assert_eq!(
            relation.used, 0,
            "preflight refusal cannot claim proposal arrival"
        );
    } else {
        assert!(relation.scope_census <= 262_144 && relation.work_reserved > 0);
        assert!(
            relation.attempted > 0 || relation.refused > 0,
            "admitted inspection must report discovery or an honest refusal"
        );
    }
    if expected_closed {
        assert_eq!(tier.closed, vec![class]);
        assert_closed_classes_hold(
            net,
            &lo,
            &hi,
            label,
            &tier.closed,
            "checked relation Cora tier",
        );
    } else {
        assert_eq!((tier.closed).len(), 0);
    }
    eprintln!(
        "[fd-relations-real-cora] tag={tag} class={class} state={} unstable={unstable} state_gens={gens} baseline_nodes={} relation_nodes={} same_root_bits=1 original_replay={}/{} receipt={:?}",
        mode.token(),
        baseline.nodes,
        run.search.nodes,
        run.search.crosschecks,
        run.search.certified_prunes,
        run.search.relations
    );
}

#[test]
fn frozen_dual_shared_relations_match_real_cora_charts_and_original_replay() {
    for (tag, classes) in [("img321", &[4, 2][..]), ("img141", &[0][..])] {
        let fixture = load_p1(tag);
        let (net, input) = p1_setup(&fixture);
        let frame = FrozenDualFrame::new(&net, input, fixture.label)
            .expect("pinned original real Cora frame");
        for &class in classes {
            compare_real_cora_relation_chart(
                &net,
                frame.clone(),
                tag,
                class,
                5_000,
                true,
                ny_propagate::margin_row::zono_bab::FrozenDualStateMode::Shared,
            );
        }
    }
}

#[test]
fn frozen_dual_shared_relations_preserve_a_witnessed_real_cora_sat_chart() {
    let fixture = load_p1("img141");
    let (net, input) = p1_setup(&fixture);
    let center: Vec<_> = input
        .lo
        .iter()
        .zip(&input.hi)
        .map(|(&l, &h)| f64::midpoint(l, h))
        .collect();
    let values = dd_forward(&net, &center).expect("original real network witness");
    let logits = &values[net.ops.len()];
    let mut labels: Vec<_> = (0..logits.len()).collect();
    labels.sort_by(|&a, &b| {
        logits[b]
            .to_f64()
            .total_cmp(&logits[a].to_f64())
            .then(a.cmp(&b))
    });
    let (top, runner_up) = (labels[0], labels[1]);
    assert!(
        (logits[runner_up] - logits[top]).to_f64() < 0.0,
        "original center is a SAT witness"
    );
    let frame = FrozenDualFrame::new(&net, input, runner_up)
        .expect("derived swapped-target negative control");
    compare_real_cora_relation_chart(
        &net,
        frame,
        "img141-swapped-target",
        top,
        256,
        false,
        ny_propagate::margin_row::zono_bab::FrozenDualStateMode::Shared,
    );
}

#[test]
fn frozen_dual_default_per_class_relations_match_real_cora_and_original_replay() {
    for (tag, classes) in [("img321", &[4, 2][..]), ("img141", &[0][..])] {
        let fixture = load_p1(tag);
        let (net, input) = p1_setup(&fixture);
        let frame = FrozenDualFrame::new(&net, input, fixture.label).unwrap();
        assert_eq!(frame.n_unstable, if tag == "img321" { 172 } else { 280 });
        for &class in classes {
            compare_real_cora_relation_chart(
                &net,
                frame.clone(),
                tag,
                class,
                5_000,
                true,
                ny_propagate::margin_row::zono_bab::FrozenDualStateMode::default(),
            );
        }
    }
}

#[test]
fn frozen_dual_default_per_class_relations_preserve_a_witnessed_real_cora_sat_chart() {
    let fixture = load_p1("img141");
    let (net, input) = p1_setup(&fixture);
    let center: Vec<_> = input
        .lo
        .iter()
        .zip(&input.hi)
        .map(|(&l, &h)| f64::midpoint(l, h))
        .collect();
    let values = dd_forward(&net, &center).unwrap();
    let logits = &values[net.ops.len()];
    let mut labels: Vec<_> = (0..logits.len()).collect();
    labels.sort_by(|&a, &b| {
        logits[b]
            .to_f64()
            .total_cmp(&logits[a].to_f64())
            .then(a.cmp(&b))
    });
    let (top, runner_up) = (labels[0], labels[1]);
    assert!((logits[runner_up] - logits[top]).to_f64() < 0.0);
    let frame = FrozenDualFrame::new(&net, input, runner_up).unwrap();
    compare_real_cora_relation_chart(
        &net,
        frame,
        "img141-swapped-target",
        top,
        256,
        false,
        ny_propagate::margin_row::zono_bab::FrozenDualStateMode::default(),
    );
}
