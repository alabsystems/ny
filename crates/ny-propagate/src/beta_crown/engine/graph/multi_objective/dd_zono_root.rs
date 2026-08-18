// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Root wiring for the certified double-double zonotope (`#dd-zonotope`,
//! default-ON for the fail-closed sparse-input detector; `NY_DD_ZONOTOPE=0`
//! is the kill switch).
//!
//! # Where this sits and why
//!
//! The scoping plan proposed TWO integration points: supply
//! `initial_node_bounds` from the zonotope (`shared/init.rs`) and intersect the
//! root MARGIN (`multi_objective/root.rs`). This module implements only the
//! second, and deliberately runs it BEFORE the bootstrap:
//!
//! * The margin route is the one the reference probe MEASURED — all five
//!   unsat-side vggnet16 specs it ran were proven by the certified margin
//!   alone, with no BaB. Feeding tight intermediates into ny's own CROWN would
//!   leave CROWN's certified rounding channel (the same product of `|W|` L1
//!   norms, i.e. the same `~2^66` amplification at f64) in charge of the final
//!   margin — reasoned, NOT measured (scoping risk R8), so it is not shipped.
//! * `vggnet16` currently TIMES OUT inside `compute_root_objective_bounds`
//!   (spec1: 20:27 wall against a 1200 s budget). An intersect placed after
//!   that call would never be reached. The pass therefore runs first, out of
//!   its own bounded slice of the global deadline.
//!
//! # Soundness contract
//!
//! * The explicit `NY_DD_ZONOTOPE=0` kill switch short-circuits before any
//!   allocation. With no preset or environment override, the built-in detector
//!   remains narrow (large image input, `k <= 128`, full op-surface support)
//!   and declines unsupported/non-sparse instances before allocation.
//! * Typed preset plumbing can broaden only the detector's admission/resource
//!   caps. No shipped preset currently does so; a future arming must state and
//!   measure the larger category scope. The soundness gates remain outside the
//!   preset surface.
//! * The self-policing precision gate refuses when the CERTIFIED rounding
//!   half-width is not far below the margin — "is double-double enough for
//!   THIS network?" is answered at runtime.
//! * Publishing is INTERSECT-ONLY: `lower = max(old, new)`,
//!   `upper = min(old, new)` over two certified enclosures of the same
//!   quantity. A refusal leaves the existing bounds untouched.

use std::time::{Duration, Instant};

use ny_core::{f64_to_f32_down, f64_to_f32_up};
use ny_tensor::BoundedTensor;

use crate::dd_zonotope::{
    dd_zonotope_enabled, dd_zonotope_margins, DdZonoConfig, DdZonoMargin, DdZonoPlan,
};
use crate::GraphNetwork;

/// A certified margin the root pass may use.
pub(super) struct DdZonoRootResult {
    pub(super) margin: DdZonoMargin,
    /// Certified enclosure of the graph output, ready for
    /// `build_result_with_bounds`.
    pub(super) output: Option<BoundedTensor>,
}

/// Wall-clock slice granted to the certified pass, as a fraction of the
/// remaining global budget, capped by `NY_DD_ZONOTOPE_SECS`.
fn budget(now: Instant, deadline: Option<Instant>) -> Option<Instant> {
    let cap = Duration::from_secs(
        std::env::var("NY_DD_ZONOTOPE_SECS")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(900),
    );
    match deadline {
        // Reserve 40% of what is left for the existing pipeline, so a refusal
        // never costs the whole budget.
        Some(d) => {
            let remaining = d.saturating_duration_since(now);
            if remaining < Duration::from_secs(5) {
                return None;
            }
            Some(now + cap.min(remaining.mul_f32(0.6)))
        }
        None => Some(now + cap),
    }
}

/// Run the certified double-double zonotope at the BaB root.
///
/// Returns `None` for every refusal — gate off, detector declined, unsupported
/// op, cap exceeded, deadline expiry, or the self-policing precision gate.
pub(super) fn run_dd_zono_root(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    objectives: &[Vec<f32>],
    deadline: Option<Instant>,
    config: &crate::BetaCrownConfig,
) -> Option<DdZonoRootResult> {
    if !dd_zonotope_enabled() {
        return None;
    }
    // Admission caps only (#metaroom-ddzono): the preset may resize the
    // detector's blast-radius/resource caps; env knobs keep precedence and the
    // soundness gates are not reachable from here.
    let cfg = DdZonoConfig::from_env().with_admission_overrides(
        config.dd_zonotope_min_input_numel,
        config.dd_zonotope_max_k,
        config.dd_zonotope_max_generators,
        config.dd_zonotope_collect_interm,
    );
    let plan = DdZonoPlan::detect(graph, input, &cfg)?;
    let now = Instant::now();
    let pass_deadline = budget(now, deadline)?;
    eprintln!(
        "[dd-zonotope] admitted: k={} input={} budget={:.1}s",
        plan.k(),
        input.len(),
        pass_deadline.saturating_duration_since(now).as_secs_f32()
    );

    let margin =
        match dd_zonotope_margins(graph, input, objectives, &plan, &cfg, Some(pass_deadline)) {
            Ok(Some(m)) => m,
            Ok(None) => {
                eprintln!("[dd-zonotope] refused (fail-closed); bounds unchanged");
                return None;
            }
            Err(e) => {
                eprintln!("[dd-zonotope] error ({e}); bounds unchanged");
                return None;
            }
        };

    for (i, ((&l, &u), (&hw, &rw))) in margin
        .lower
        .iter()
        .zip(margin.upper.iter())
        .zip(
            margin
                .rounding_half_width
                .iter()
                .zip(margin.relax_half_width.iter()),
        )
        .enumerate()
    {
        eprintln!(
            "[dd-zonotope] obj[{i}] certified=[{l:.9}, {u:.9}] relax_hw={rw:.4e} rounding_hw={hw:.4e}"
        );
    }
    eprintln!(
        "[dd-zonotope] gens={} wall={:.1}s",
        margin.n_generators,
        margin.wall.as_secs_f32()
    );

    // SELF-POLICING PRECISION GATE. The `~2^66` amplification was MEASURED on
    // vgg16-7 only; a deeper or larger-weight network could exceed
    // double-double. Refusing here turns that from an unsound assumption into
    // an ordinary decline.
    if !margin.precision_ok(cfg.precision_ratio) {
        eprintln!(
            "[dd-zonotope] PRECISION GATE refused: certified rounding half-width is not \
             below {:.1e} x |margin|; bounds unchanged",
            cfg.precision_ratio
        );
        return None;
    }

    let output = build_output_tensor(&margin);
    Some(DdZonoRootResult { margin, output })
}

fn build_output_tensor(margin: &DdZonoMargin) -> Option<BoundedTensor> {
    if margin.output_lower.is_empty() || margin.output_shape.is_empty() {
        return None;
    }
    let shape = ndarray::IxDyn(&margin.output_shape);
    let lower = ndarray::ArrayD::from_shape_vec(shape.clone(), margin.output_lower.clone()).ok()?;
    let upper = ndarray::ArrayD::from_shape_vec(shape, margin.output_upper.clone()).ok()?;
    BoundedTensor::new(lower, upper).ok()
}

/// Element-wise INTERSECTION of the existing certified objective bounds with
/// the zonotope's.
///
/// Both operands are certified enclosures of the same quantity, so keeping the
/// tighter side of each is sound. A non-finite or inverted zonotope entry is
/// skipped rather than published.
pub(super) fn intersect_objective_bounds(
    existing: &mut [(f32, f32)],
    margin: &DdZonoMargin,
) -> usize {
    if existing.len() != margin.lower.len() {
        return 0;
    }
    let mut tightened = 0usize;
    for (i, slot) in existing.iter_mut().enumerate() {
        let lo = margin.lower[i];
        let hi = margin.upper[i];
        if !lo.is_finite() || !hi.is_finite() || lo > hi {
            continue;
        }
        // Round OUTWARD on the f64 -> f32 narrowing so the published f32
        // interval still encloses the certified f64 one.
        let lo32 = f64_to_f32_down(lo);
        let hi32 = f64_to_f32_up(hi);
        if !lo32.is_finite() || !hi32.is_finite() || lo32 > hi32 {
            continue;
        }
        let before = *slot;
        if lo32 > slot.0 {
            slot.0 = lo32;
        }
        if hi32 < slot.1 {
            slot.1 = hi32;
        }
        // An inverted intersection means the two certified enclosures are
        // disjoint, which cannot happen for sound operands. Restore rather than
        // publish an impossible interval.
        if slot.0 > slot.1 {
            *slot = before;
            continue;
        }
        if *slot != before {
            tightened += 1;
        }
    }
    tightened
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dd_zono_root_gate_defaults_on_with_kill_switch() {
        // DEFAULT-ON since the 2026-07-23 GO-AFTER-FIXES audit (underflow floor
        // applied). `dd_zonotope_enabled()` is a process-wide OnceLock, so this
        // asserts the shipped default (no test may set the variable). The
        // `NY_DD_ZONOTOPE=0` kill switch restores the inert path; safety for
        // non-sparse instances comes from `DdZonoPlan::detect` failing closed
        // (k <= 128, exact-decimal box, full op-surface check), not from the
        // gate.
        assert!(
            dd_zonotope_enabled() || std::env::var("NY_DD_ZONOTOPE").as_deref() == Ok("0"),
            "the #dd-zonotope gate must default to ON (kill switch NY_DD_ZONOTOPE=0)"
        );
    }

    fn margin(lower: Vec<f64>, upper: Vec<f64>) -> DdZonoMargin {
        let n = lower.len();
        DdZonoMargin {
            lower,
            upper,
            rounding_half_width: vec![1e-12; n],
            relax_half_width: vec![1e-6; n],
            output_lower: vec![],
            output_upper: vec![],
            output_shape: vec![],
            interm: std::collections::HashMap::new(),
            n_generators: 1,
            wall: Duration::from_secs(0),
        }
    }

    #[test]
    fn intersection_keeps_the_tighter_side_only() {
        let mut b = vec![(-1.0e18_f32, 1.0e18_f32), (0.5, 2.0)];
        let m = margin(vec![1.6, 0.1], vec![1.7, 1.0]);
        let n = intersect_objective_bounds(&mut b, &m);
        assert_eq!(n, 2);
        // Objective 0: both sides tighten.
        assert!(b[0].0 > 1.59 && b[0].0 <= 1.6);
        assert!(b[0].1 >= 1.7 && b[0].1 < 1.71);
        // Objective 1: the zonotope lower is LOOSER, so it must not be taken;
        // the upper is tighter and is.
        assert_eq!(b[1].0, 0.5);
        assert!(b[1].1 >= 1.0 && b[1].1 < 1.001);
    }

    #[test]
    fn intersection_never_widens() {
        let mut b = vec![(1.0_f32, 2.0_f32)];
        let before = b.clone();
        let m = margin(vec![-5.0], vec![9.0]);
        assert_eq!(intersect_objective_bounds(&mut b, &m), 0);
        assert_eq!(b, before);
    }

    #[test]
    fn intersection_skips_non_finite_and_inverted_entries() {
        let mut b = vec![(0.0_f32, 1.0_f32), (0.0, 1.0), (0.0, 1.0)];
        let before = b.clone();
        let mut m = margin(vec![f64::NAN, 3.0, f64::NEG_INFINITY], vec![1.0, 2.0, 1.0]);
        m.rounding_half_width = vec![1e-12; 3];
        m.relax_half_width = vec![1e-6; 3];
        // entry 1 is inverted (lo > hi); entry 2 has a non-finite lower.
        assert_eq!(intersect_objective_bounds(&mut b, &m), 0);
        assert_eq!(b, before);
    }

    #[test]
    fn intersection_declines_on_a_length_mismatch() {
        let mut b = vec![(0.0_f32, 1.0_f32)];
        let before = b.clone();
        let m = margin(vec![0.5, 0.5], vec![0.6, 0.6]);
        assert_eq!(intersect_objective_bounds(&mut b, &m), 0);
        assert_eq!(b, before);
    }
}
