// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Dark, print-only per-node A-matrix telemetry for the iteration-0 α parity
//! investigation (#iter0-alpha-parity, docs/SYSTEM_DESIGN_ONE_PIPELINE P0-1).
//!
//! `NY_ITER0_PARITY_TRACE=1` enables; the declared `false` default emits no
//! output. WHY: the cifar100 root α loop evaluates its iteration-0 bound at
//! -2.15e23 against a -1989.90 pre-loop CROWN baseline
//! (docs/ROOT_ALPHA_STEP_EXPLODES_AND_STALLS_2026-07-29.md). Localizing WHERE
//! the 20 orders of magnitude enter requires the per-node accumulated
//! coefficient magnitudes of BOTH backward folds (the fixed-slope Graph-CROWN
//! baseline and the α loop's fold) over the SAME run — wall-clock profiling
//! and the returned (elementwise-best) bounds cannot see it.
//!
//! One stderr line per node visited by a backward walk:
//!
//! ```text
//! [iter0-parity] walk=<id> pass=<label> node=<name> op=<layer> repr=<Dense|Patches>
//!     rows=<n> max_row_l1_lo=<v> max_row_l1_up=<v> bias_lo=[min,max] bias_up=[min,max]
//!     coeff_err_lo=<max-row-sum|-> coeff_err_up=<max-row-sum|->
//! ```
//!
//! Walk ids are process-global and monotonically increasing: one id per
//! backward walk, claimed at the walk's seed, so interleaved walks (reference
//! collection, pre-loop baseline, α iterations, gradient passes) separate
//! cleanly offline. Print-only: nothing here feeds any bound, verdict, or
//! schedule decision. The gate is checked FIRST at every site, so the
//! declared-false path is one latched-string compare — the O(nnz) stat
//! reductions below run only when armed. Armed-vs-unarmed deadline/verdict
//! parity is not claimed.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

use crate::bounds::patches::{CrownBounds, PatchesLinearBounds};
use crate::bounds::LinearBounds;

/// Pure gate predicate: exactly `"1"` enables (same idiom as
/// `phase_telemetry::gate_on`).
fn gate_on(raw: Option<&str>) -> bool {
    raw == Some("1")
}

/// Latched RAW env string (lever-debt batch B1 preparation), read once through
/// the ny-levers chokepoint's raw view. The gate is checked per backward walk
/// inside the alpha loop — hot — so the string is latched and the decision is
/// derived per call by [`gate_on`]. This remains process-wide; Phase 2 must
/// replace it with an injected per-run `LeverSet`.
fn env_raw() -> Option<&'static str> {
    static RAW: OnceLock<Option<String>> = OnceLock::new();
    RAW.get_or_init(|| ny_levers::read_raw(&ny_levers::decls::telemetry::ITER0_PARITY_TRACE))
        .as_deref()
}

/// Process-wide gate over the latched raw string. One latched-string compare
/// when unset.
pub(crate) fn iter0_parity_trace_enabled() -> bool {
    gate_on(env_raw())
}

/// Claim a fresh walk id at a backward-walk seed. Only meaningful (and only
/// called) when the gate is armed.
pub(crate) fn next_walk_id() -> u64 {
    static WALK: AtomicU64 = AtomicU64::new(0);
    WALK.fetch_add(1, Ordering::Relaxed)
}

/// max over rows of the L1 norm of the row's coefficients.
fn max_row_l1(a: &ndarray::Array2<f32>) -> f32 {
    a.rows()
        .into_iter()
        .map(|row| row.iter().map(|v| v.abs()).sum::<f32>())
        .fold(0.0f32, f32::max)
}

fn minmax(b: &ndarray::Array1<f32>) -> (f32, f32) {
    b.iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        })
}

fn dense_line(walk: u64, pass: &str, node: &str, op: &str, lb: &LinearBounds) -> String {
    let (blo_min, blo_max) = minmax(&lb.lower_b);
    let (bup_min, bup_max) = minmax(&lb.upper_b);
    let err_lo = lb
        .lower_a_err
        .as_ref()
        .map_or_else(|| "-".to_string(), |e| format!("{:.3e}", max_row_l1(e)));
    let err_up = lb
        .upper_a_err
        .as_ref()
        .map_or_else(|| "-".to_string(), |e| format!("{:.3e}", max_row_l1(e)));
    format!(
        "[iter0-parity] walk={walk} pass={pass} node={node} op={op} repr=Dense rows={} \
         max_row_l1_lo={:.6e} max_row_l1_up={:.6e} bias_lo=[{:.3e},{:.3e}] \
         bias_up=[{:.3e},{:.3e}] coeff_err_lo={err_lo} coeff_err_up={err_up}",
        lb.lower_a.nrows(),
        max_row_l1(&lb.lower_a),
        max_row_l1(&lb.upper_a),
        blo_min,
        blo_max,
        bup_min,
        bup_max,
    )
}

/// Patches side: the per-row L1 is not directly addressable without a dense
/// materialization (which the trace must not force — it would change the
/// walk's own memory/time behavior). Report the global |coeff| sum and max —
/// enough to see a >=100x inter-node jump — plus the identity flag, the
/// explicit-rows dimensionality, and the per-row certified coeff_err extrema.
fn patches_line(walk: u64, pass: &str, node: &str, op: &str, pb: &PatchesLinearBounds) -> String {
    let side = |p: &crate::bounds::patches::PatchesData| -> (String, String) {
        let mag = match &p.patches {
            Some(arr) => {
                let (sum, max) = arr.iter().fold((0.0f64, 0.0f32), |(s, m), &v| {
                    (s + f64::from(v.abs()), m.max(v.abs()))
                });
                format!(
                    "ndim={} abs_sum={:.6e} abs_max={:.3e}",
                    arr.ndim(),
                    sum,
                    max
                )
            }
            None => format!("identity={}", p.identity),
        };
        let err = p.coeff_err.as_ref().map_or_else(
            || "-".to_string(),
            |e| {
                let (mn, mx) = minmax(e);
                format!("[{mn:.3e},{mx:.3e}]")
            },
        );
        (mag, err)
    };
    let (lo_mag, lo_err) = side(&pb.lower_a);
    let (up_mag, up_err) = side(&pb.upper_a);
    let (blo_min, blo_max) = minmax(&pb.lower_b);
    let (bup_min, bup_max) = minmax(&pb.upper_b);
    format!(
        "[iter0-parity] walk={walk} pass={pass} node={node} op={op} repr=Patches rows={} \
         lo({lo_mag}) up({up_mag}) bias_lo=[{blo_min:.3e},{blo_max:.3e}] \
         bias_up=[{bup_min:.3e},{bup_max:.3e}] coeff_err_lo={lo_err} coeff_err_up={up_err}",
        pb.row_count,
    )
}

/// Emit one per-node line for the accumulated bounds ARRIVING at `node` (i.e.
/// after all consumers merged, before this node's own backward transform).
/// Caller must have checked [`iter0_parity_trace_enabled`] — this asserts
/// nothing and prints unconditionally.
pub(crate) fn trace_node(walk: u64, pass: &str, node: &str, op: &str, cb: &CrownBounds) {
    let line = match cb {
        CrownBounds::Dense(lb) => dense_line(walk, pass, node, op, lb),
        CrownBounds::Patches(pb) => patches_line(walk, pass, node, op, pb),
    };
    eprintln!("{line}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_requires_exactly_one() {
        assert!(gate_on(Some("1")));
        assert!(!gate_on(Some("true")));
        assert!(!gate_on(Some("0")));
        assert!(!gate_on(None));
    }

    #[test]
    fn dense_line_reports_row_l1_and_bias_range() {
        let lb = LinearBounds::identity(2);
        let line = dense_line(7, "loop", "n", "ReLU", &lb);
        assert!(line.contains("walk=7"));
        assert!(line.contains("repr=Dense"));
        assert!(line.contains("max_row_l1_lo=1.000000e0"));
    }
}
