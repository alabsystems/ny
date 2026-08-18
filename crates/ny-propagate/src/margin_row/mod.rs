// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Margin-row twin-wall BaB lane (#twinwall).
//!
//! Production Rust implementation of THE verified twin-wall mechanism
//! (validated against an exact-rational falsifier harness during
//! development): for the
//! cifar100/tinyimagenet resnet family, per BaB domain, ONE sparse CROWN
//! backward pass of `nf + a few` MARGIN rows — the margin functional rows
//! seeded at the output, composed through per-domain-refreshed HEAD gates and
//! FROZEN root trunk gates with PIECE-FIXED overrides for split neurons —
//! concretized against the input box. Best-first BaB with exact batched
//! candidate scoring closes the failing classes.
//!
//! Every verdict-feeding bound runs in [`RoundMode::Outward`]: f64 accumulate
//! with certified error carry + directed rounding toward -inf (the moat:
//! never overstate a lower bound; NaN/Inf/unsupported fail closed to
//! Unknown). `RoundMode::Parity` reproduces the Python reference bit-nearly
//! for differential tests only.
//!
//! Production authority was granted 2026-07-18 after the directed-rounding and
//! strict ONNX/VNNLIB-adapter proof obligations were discharged.  The trusted
//! constant and detailed evidence trail live with [`margin_row_bab_enabled`];
//! no environment variable can promote an unauthorised lane to an `Unsat`
//! verdict.

pub mod bab;
pub mod bounds;
pub mod engine;
pub(crate) mod gpu_seam;
pub mod hyperplane_probe;
pub mod net;
pub mod prof;
pub mod root;
pub mod rounding;
pub mod spec;

#[cfg(test)]
mod tests;

use std::time::Instant;

pub use bab::{BabConfig, BabStats, ClassBabStats, MarginRowBab, MarginRowOutcome};
pub use net::TwinNet;
pub use root::RootGates;
pub use rounding::RoundMode;
pub use spec::{TwinOpSpec, TwinSpec};

/// Production gate for the vnncomp wiring.
///
/// Authority granted 2026-07-18 after BOTH quarantine proof obligations were
/// discharged (adversarial + formal verification, ~850k zero-violation enclosure
/// checks; commits 5628ed6f/55455cbb/b7342fdf/8841eaa5):
///   1. margin-coefficient directed-rounding proof — the backward-bias running-
///      accumulator under-count (a REAL false-UNSAT, 351-witness oracle) is fixed
///      (2u*|b_running| per step), and the root-tableau outward headroom is now
///      depth-scaled (gamma_n(n_in+16+8*trunk_relus)) so soundness is depth-
///      independent by construction. Both guarded by discriminating oracles.
///   2. strict ONNX/vnnlib adapter validation — the input box AND the output
///      robustness property are cross-checked against the trusted load_vnnlib spec
///      (set-equality of the adversarial set), fail-closed on any divergence.
///
/// Kept a trusted code constant (NOT an env var) so no untrusted process can flip
/// the authority switch. The lane is strictly-additive (only turns timeout/unknown
/// into a certified `unsat`, never `sat`), so enabling it cannot regress a verdict
/// except via the budget reserve, which only applies to the twin-wall net family.
///
/// Measured behind this gate (#epoch-bab, docs/EPOCH_BAB_STATE_2026-07-18.md):
/// +5 `metaroom_2023` UNSAT instances the internal verifier times out on, each
/// closed at the ROOT in ~10s, every one confirmed against the official 2025
/// results (>=6 tools unsat, 0 sat). The shipped path defaults to a 45 s
/// reserve. Concurrent execution is a separate performance option that requires
/// `NY_MARGIN_ROW_CONCURRENT=1` and a GPU-device preset; it is off by default.
pub fn margin_row_bab_enabled() -> bool {
    true
}

/// Typed preset route for the SOUND f32 root tableau (`margin_row.root_f32`).
///
/// Set ONCE by the CLI from the category preset before any lane runs; read by
/// [`root::RootGates::build_retaining`] alongside `NY_MARGIN_ROW_ROOT_F32`.
/// This exists because the f32 gate lives four crates below the preset loader
/// and the workspace forbids writing process environment from code.
///
/// SOUNDNESS: arming it can only LOOSEN the root tableau — the f32 rounding is
/// charged into a certified additive concretize slack that is subtracted from
/// every lower endpoint and added to every upper one. A wrong value here costs
/// proofs; it cannot manufacture one. The lane itself remains fail-closed
/// (`Unsat` or `Unknown` only).
static ROOT_F32_PRESET: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Arm/disarm the typed preset route for the f32 root tableau.
pub fn set_root_f32_preset(on: bool) {
    ROOT_F32_PRESET.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Is the typed preset route armed?
pub(crate) fn root_f32_preset() -> bool {
    ROOT_F32_PRESET.load(std::sync::atomic::Ordering::Relaxed)
}

/// End-to-end lane: compile, build certified root gates, run the BaB.
/// `t` = true class, `adv` = adversarial classes of the robustness
/// disjunction. Only ever returns `Unsat` or `Unknown` (fail-closed).
pub fn run_margin_row_lane(
    spec: &TwinSpec,
    lo: &[f64],
    hi: &[f64],
    t: usize,
    adv: &[usize],
    deadline: Option<Instant>,
    max_expansions: usize,
) -> MarginRowOutcome {
    lane_impl(spec, lo, hi, t, adv, deadline, max_expansions)
}

/// Quarantined compatibility entry for the unproven external-head experiment.
///
/// `external_head` is unconditionally discarded. In particular,
/// `NY_MARGIN_ROW_HEAD_INJECT` has no authority: no environment value can move
/// a bound or verdict through this entry point. This function is a hard
/// semantic alias of [`run_margin_row_lane`] until a typed, provenance-checked
/// `CertifiedHeadBox` exists. See `docs/MARGIN_ROW_ROOT_JOINT_COUPLING.md`.
pub fn run_margin_row_lane_with_head(
    spec: &TwinSpec,
    lo: &[f64],
    hi: &[f64],
    t: usize,
    adv: &[usize],
    deadline: Option<Instant>,
    max_expansions: usize,
    external_head: Option<(Vec<f64>, Vec<f64>)>,
) -> MarginRowOutcome {
    drop(external_head);
    run_margin_row_lane(spec, lo, hi, t, adv, deadline, max_expansions)
}

#[allow(clippy::too_many_arguments)]
fn lane_impl(
    spec: &TwinSpec,
    lo: &[f64],
    hi: &[f64],
    t: usize,
    adv: &[usize],
    deadline: Option<Instant>,
    max_expansions: usize,
) -> MarginRowOutcome {
    if prof::enabled() {
        prof::reset();
    }
    let net = match TwinNet::compile(spec) {
        Ok(n) => n,
        Err(e) => {
            return MarginRowOutcome::Unknown {
                reason: format!("compile: {e}"),
                stats: None,
            }
        }
    };
    // Tier-0 (#epoch-bab): the rank-1 candidate ranker scores a WIDE trunk
    // universe in O(nf·n_in) each and sends only the best `k` to the exact
    // Tier-1 pass. DEFAULT OFF; `NY_EPOCH_TIER0=k` enables it.
    //
    // Ranker-only by construction: candidate ORDERING cannot move a verdict
    // — every pushed bound and every closure still flows through the
    // unchanged certified Outward pass, so the worst a bad ranking can do
    // is waste budget.
    //
    // Held OFF by the completed same-binary A/B on the cifar100 pyrat-easy
    // tier (docs/EPOCH_BAB_STATE_2026-07-18.md). Established: expansion
    // counts IDENTICAL wherever both lanes close (2, 13, 29, 23, 31 — the
    // verified tree is reproduced exactly), `score_candidates` 124 → 66
    // ms/call, and every shared closure is reached faster. NOT established:
    // a net closure gain — the tier count is 6 = 6 (Tier-0 gains prop_1588,
    // loses prop_4605; wider ranking explores a different tree, better on
    // some instances and worse on others). Since scoring is 10 points per
    // solve against a small time bonus, closure count is the load-bearing
    // number and it is a wash, so this stays gated until an A/B over the
    // full wall set — not a 13-instance tier — shows a net win.
    let tier0_exact: usize = std::env::var("NY_EPOCH_TIER0")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let tier0_universe: usize = std::env::var("NY_EPOCH_UNIVERSE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);
    // Tier-2 (#epoch-bab): `NY_EPOCH_DEPTH=d` re-linearizes a domain's
    // subtree once it carries `d` trunk splits (0/unset = off);
    // `NY_EPOCH_ATTEMPTS` caps Unknown-ending rebuilds per run.
    let epoch_depth: usize = std::env::var("NY_EPOCH_DEPTH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let epoch_max_attempts: usize = std::env::var("NY_EPOCH_ATTEMPTS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);
    let retain_cfg = (tier0_exact > 0 || epoch_depth > 0).then(|| root::RetainCfg {
        per_layer: std::env::var("NY_EPOCH_RETAIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(128),
        ..root::RetainCfg::default()
    });
    let build_t0 = Instant::now();
    let (root, retained) = {
        let _t = prof::Timer::start(prof::Phase::RootGate);
        match RootGates::build_retaining(
            &net,
            lo,
            hi,
            RoundMode::Outward,
            deadline,
            retain_cfg.as_ref(),
            &[],
        ) {
            Ok(r) => r,
            Err(e) => {
                return MarginRowOutcome::Unknown {
                    reason: format!("root gates: {e}"),
                    stats: None,
                }
            }
        }
    };
    let root_build_secs = build_t0.elapsed().as_secs_f64();
    // READ-ONLY diagnostic (#hyperplane-probe): default-off behind
    // `NY_HYPERPLANE_PROBE`. Computes and logs the trunk->head Jacobian
    // spectrum, then returns. Touches no bound, gate, or verdict; when the env
    // var is unset this closure is never entered, so behavior is byte-identical.
    if hyperplane_probe::enabled() {
        hyperplane_probe::run(&net, &root, t, adv);
    }
    // y-pack cache capacity (trunk-sets). Bit-identical to the serial oracle
    // (a hit returns the SAME deterministic pack a miss would rebuild), so a
    // larger cache only trades memory for fewer recomputations — no bound
    // moves. Overridable for A/B measurement (`NY_MARGIN_ROW_LRU`).
    let lru_cap = std::env::var("NY_MARGIN_ROW_LRU")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);
    // Parallel best-first frontier width. The legacy banked exact-timeout wall
    // sweep used all available Rayon workers and the serial-vs-parallel gates
    // prove every per-domain bound bit-identical; make that measured recipe the
    // margin-row-only default. `NY_MARGIN_ROW_PARALLEL=0` is the serial kill
    // switch, `=1` explicitly selects all workers, and `=N` pins a width.
    let parallel = std::env::var("NY_MARGIN_ROW_PARALLEL").ok();
    let frontier =
        margin_row_frontier_from_env(parallel.as_deref(), rayon::current_num_threads().max(1));
    // Default-OFF cross-domain score-row stack. Only exact `=1` arms this
    // canary; unset, malformed, and friendly truthy spellings all retain the
    // established independent `score_candidates` calls.
    let domain_stack = margin_row_domain_stack_from_env(
        std::env::var("NY_MARGIN_ROW_DOMAIN_STACK").ok().as_deref(),
    );
    let cfg = BabConfig {
        max_expansions,
        deadline,
        lru_cap,
        frontier,
        domain_stack,
        tier0_exact,
        tier0_universe,
        retained: retained.map(std::sync::Arc::new),
        epoch_depth,
        epoch_max_attempts,
        root_build_secs,
        retain_cfg,
        ..BabConfig::default()
    };
    // Default-OFF bounded experiment: replace the joint multi-margin tree with
    // a conjunction of independent full-box class proofs. Unset/0 preserves
    // the established joint call exactly.
    let classwise =
        margin_row_classwise_from_env(std::env::var("NY_MARGIN_ROW_CLASSWISE").ok().as_deref());
    let out = if classwise {
        MarginRowBab::run_classwise(&net, &root, t, adv, cfg)
    } else {
        MarginRowBab::run(&net, &root, t, adv, cfg)
    };
    if prof::enabled() {
        eprint!("{}", prof::dump());
    }
    out
}

fn margin_row_classwise_from_env(value: Option<&str>) -> bool {
    value.is_some_and(|value| {
        let value = value.trim();
        value == "1"
            || value.eq_ignore_ascii_case("true")
            || value.eq_ignore_ascii_case("on")
            || value.eq_ignore_ascii_case("yes")
    })
}

fn margin_row_domain_stack_from_env(value: Option<&str>) -> bool {
    value.is_some_and(|value| value.trim() == "1")
}

fn margin_row_frontier_from_env(value: Option<&str>, available: usize) -> usize {
    let available = available.max(1);
    match value.map(str::trim) {
        None => available,
        Some("") | Some("0") => 1,
        Some(v)
            if v.eq_ignore_ascii_case("false")
                || v.eq_ignore_ascii_case("off")
                || v.eq_ignore_ascii_case("no") =>
        {
            1
        }
        Some("1") => available,
        Some(v)
            if v.eq_ignore_ascii_case("true")
                || v.eq_ignore_ascii_case("on")
                || v.eq_ignore_ascii_case("yes") =>
        {
            available
        }
        Some(v) => v.parse::<usize>().unwrap_or(1).max(1),
    }
}
