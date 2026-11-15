// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #sbench-f64-maximin: the LP-maximin step and basin hopping on the float64
//! lift of #sbench-f64-attack (`attack.f64_lift_step: lp_maximin`,
//! `attack.f64_basin_hops`).
//!
//! # Why
//!
//! SoundnessBench `model_residual / property_033` is truth-SAT: two witnesses
//! found off-box replay as violated on the organisers' x86-64 ONNX Runtime
//! 1.26.0 at +4.9e-6 and +3.5e-6 (`scratchpad/vbs/frontier/rest.md` §2). Both
//! came from a sequential LP that MAXIMISES THE MINIMUM exact-f64 margin over
//! all twelve conjuncts, and the from-scratch one needed 60 basin hops on top.
//! ny's post-BaB tail stalls about 9e-6 short (PRIOR, pin c73c289c2,
//! `scratchpad/refine2/sbs_9.raw`), and #sbench-f64-attack's min-norm lift is
//! a LOCAL Newton move on the active set: it lifts the lagging rows toward a
//! target, so it can only climb the basin it starts in.
//!
//! # What it does
//!
//! [`lp_maximin_lift_core`] runs, on the f32 witness lattice:
//!
//! * **the LP-maximin step** — at the iterate `x`, with exact-f64 margins `m`
//!   and exact Jacobian rows `J` of EVERY conjunct, solve
//!   `max t  s.t.  m_i + J_i . D >= t,  lo <= x + D <= hi,  |D_d| <= rho * w_d`
//!   ([`solve_maximin_lp`], a dense bounded-variable simplex; the LP is
//!   non-authoritative, so plain f64 is enough), round `x + D` to f32 toward
//!   the dual-weighted gradient of the binding conjuncts (C8's favourable
//!   rounding), and ACCEPT the move iff the TRUE exact-f64 minimum margin at
//!   the rounded point strictly improves — otherwise halve `rho`;
//! * **basin hopping** — when a run stalls (five consecutive iterations after
//!   its first accepted step without a `1e-9` gain) or its trust region
//!   collapses, kick the incumbent by uniform noise of a log-uniform radius
//!   `10^U(-5,-2)` of each coordinate's width, run the SLP again from there,
//!   and keep the better incumbent — at most `attack.f64_basin_hops` times.
//!
//! The trusted forward is consulted exactly as in #sbench-f64-attack: only at
//! points whose exact margin is already `>= 0`, at most once per point; a
//! point it confirms at zero tolerance becomes a candidate ranked by
//! `min(trusted margin, exact margin)`; the lift stops early once a confirmed
//! point clears the (escalating) `f64_target_margin_ulps` target.
//!
//! # Soundness
//!
//! Attack-only, and the exit is #sbench-f64-attack's, unchanged: a point
//! leaves only as `F64RepairOutcome::violation`, which is `Some` ONLY for a
//! point the trusted forward confirmed at zero tolerance (the unchanged
//! `property_violated_f64` on the witness input view) AND whose exact-f64
//! margin is `>= 0`. The caller then applies the same local true-f64 precheck
//! as every post-BaB lineage (`emit_found!`) and the unchanged
//! `gate_sat_with_trusted_oracle`. Nothing here returns a verdict, so no
//! `unsat` can come from it; the LP, the rounding and the kicks only choose
//! which points are TRIED.

use std::time::Instant;

use ny_core::f32_to_f64_exact;
use ny_onnx::vnnlib::VnnLibSpec;
use ny_propagate::{PointVjpF64Plan, PointVjpF64Tape};

use super::super::{clamp_inside_box, clamp_to_box};
use super::{
    exact_margins, margin_output_row, min_of, repair_conjuncts, round_toward, target_delta_for,
    trusted_check, F64LiftStats, F64RepairOutcome, MAX_DIM, NEAR_BAND, TARGET_ESCALATIONS,
};

/// Initial trust radius of the first SLP run, as a fraction of each
/// coordinate's box width (the reference SLP's `r = 0.05` on a width-1 box).
const RHO_INIT: f64 = 0.05;
/// Largest trust radius of the first run.
const RHO_MAX: f64 = 0.5;
/// Largest trust radius inside a basin hop; a hop's run starts at its kick
/// radius (the reference `hop.py`).
const HOP_RHO_MAX: f64 = 0.05;
/// Trust-radius growth on an accepted step.
const RHO_GROW: f64 = 1.5;
/// Trust-radius shrink on a rejected step.
const RHO_SHRINK: f64 = 0.5;
/// A run ends when its trust radius falls below this fraction of the width
/// (far below the f32 spacing of any coordinate of a unit-scale box).
pub(super) const RHO_MIN: f64 = 1e-9;
/// A run has stalled after this many consecutive iterations (counted from its
/// first accepted step) that did not raise the exact minimum margin by
/// [`STALL_EPS`].
pub(super) const STALL_WINDOW: usize = 5;
/// The stall threshold on the exact minimum margin.
pub(super) const STALL_EPS: f64 = 1e-9;
/// Kick radius range, `log10` of the fraction of each coordinate's width.
pub(super) const HOP_LOG10_MIN: f64 = -5.0;
/// See [`HOP_LOG10_MIN`].
pub(super) const HOP_LOG10_MAX: f64 = -2.0;
/// SLP iteration cap inside one hop (the reference `hop.py`'s 150).
pub(super) const HOP_SLP_ITERS: usize = 150;

/// Pivot tolerance on the scaled simplex tableau.
const LP_PIVOT_TOL: f64 = 1e-9;
/// Reduced-cost optimality tolerance on the scaled tableau.
const LP_COST_TOL: f64 = 1e-11;
/// Ratios within this band of each other are ties.
const LP_RATIO_TIE: f64 = 1e-12;
/// A step shorter than this is degenerate.
const LP_DEGENERATE_STEP: f64 = 1e-14;
/// Consecutive degenerate pivots before the pricing switches to Bland's rule
/// (which cannot cycle).
const LP_DEGENERATE_SWITCH: usize = 32;

/// A solution of the maximin step LP.
#[derive(Debug, Clone)]
pub(super) struct MaximinLp {
    /// The step `D`, inside `[lower, upper]` coordinate-wise.
    pub(super) delta: Vec<f64>,
    /// `min_i (m_i + J_i . D)`, recomputed in f64 from the inputs (the
    /// tests' handle on optimality; the lift judges steps exactly instead).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) predicted: f64,
    /// The row multipliers (`>= 0`; they sum to one at a non-degenerate
    /// optimum): the weights of the maximin subgradient.
    pub(super) duals: Vec<f64>,
    /// Basis changes plus bound flips.
    pub(super) pivots: usize,
    /// Whether the simplex proved optimality (otherwise the returned step is
    /// feasible but was cut short by the pivot cap or the deadline).
    pub(super) optimal: bool,
}

/// Solve `max t  s.t.  margins[i] + rows[i] . D >= t` for every row,
/// `lower <= D <= upper`, by a dense bounded-variable primal simplex.
///
/// NON-AUTHORITATIVE: the result only proposes a step; the caller evaluates
/// the step's point exactly and keeps it only on a true improvement, so a
/// numerically poor solve costs a rejected step and never a wrong answer.
/// Returns `None` on malformed input (arity, non-finite values, an empty box
/// the wrong way round) or when every movable column and every row offset is
/// zero (no step can matter).
///
/// Formulation. `D = lower + w * z` with `z in [0,1]` for every coordinate
/// that can move and has a non-zero coefficient (`w = upper - lower`); every
/// other coordinate gets `D = 0` (clamped into its interval). With
/// `b_i = m_i + J_i . lower`, `t0 = min_i b_i` and `t = t0 + S * tau`
/// (`tau >= 0` cuts nothing off: `z = 0, t = t0` is feasible), each row reads
/// `-(J_i w / S) . z + tau + s_i = (b_i - t0) / S` with a slack `s_i >= 0`,
/// scaled by `S = max(max_i (b_i - t0), max |J_id w_d|)` so the tableau is
/// O(1). The slack basis is feasible at `z = 0`, so no phase 1 is needed.
pub(super) fn solve_maximin_lp(
    margins: &[f64],
    rows: &[Vec<f64>],
    lower: &[f64],
    upper: &[f64],
    deadline: Option<Instant>,
) -> Option<MaximinLp> {
    let k = margins.len();
    let n = lower.len();
    if k == 0
        || rows.len() != k
        || upper.len() != n
        || margins.iter().any(|m| !m.is_finite())
        || rows
            .iter()
            .any(|row| row.len() != n || row.iter().any(|v| !v.is_finite()))
        || lower
            .iter()
            .zip(upper)
            .any(|(lo, hi)| !(lo.is_finite() && hi.is_finite() && lo <= hi))
    {
        return None;
    }

    // Columns the LP can use: positive width and a non-zero coefficient in
    // some row. The rest stay at `D = 0`.
    let cols: Vec<usize> = (0..n)
        .filter(|&d| upper[d] > lower[d] && rows.iter().any(|row| row[d] != 0.0))
        .collect();
    let nz = cols.len();
    let mut base = vec![0.0f64; n];
    for d in 0..n {
        base[d] = 0.0f64.clamp(lower[d], upper[d]);
    }
    for &d in &cols {
        base[d] = lower[d];
    }
    // b_i = m_i + J_i . base
    let b: Vec<f64> = rows
        .iter()
        .zip(margins)
        .map(|(row, &m)| m + row.iter().zip(&base).map(|(j, x)| j * x).sum::<f64>())
        .collect();
    let t0 = min_of(&b);
    let mut scale = b.iter().map(|&v| v - t0).fold(0.0f64, f64::max);
    for row in rows {
        for &d in &cols {
            scale = scale.max((row[d] * (upper[d] - lower[d])).abs());
        }
    }
    if !(scale.is_finite() && scale > 0.0) {
        return None;
    }

    let tau = nz;
    let slack0 = nz + 1;
    let ncols = nz + 1 + k;
    let mut tab = vec![0.0f64; k * ncols];
    for i in 0..k {
        let row = &mut tab[i * ncols..(i + 1) * ncols];
        for (c, &d) in cols.iter().enumerate() {
            row[c] = -rows[i][d] * (upper[d] - lower[d]) / scale;
        }
        row[tau] = 1.0;
        row[slack0 + i] = 1.0;
    }
    // Values of the basic variables (the slacks, initially).
    let mut beta: Vec<f64> = b.iter().map(|&v| (v - t0) / scale).collect();
    let mut basis: Vec<usize> = (slack0..slack0 + k).collect();
    let mut is_basic = vec![false; ncols];
    for &j in &basis {
        is_basic[j] = true;
    }
    let mut at_upper = vec![false; ncols];
    let upper_bound = |j: usize| if j < nz { 1.0 } else { f64::INFINITY };
    // Reduced costs of `max tau` (the basis costs are zero at the start).
    let mut cost = vec![0.0f64; ncols];
    cost[tau] = 1.0;

    let max_pivots = 8 * (ncols + k) + 64;
    let mut pivots = 0usize;
    let mut degenerate_run = 0usize;
    let mut bland = false;
    let mut optimal = false;
    let mut column = vec![0.0f64; k];
    while pivots < max_pivots {
        if pivots.is_multiple_of(16) && deadline.is_some_and(|limit| Instant::now() >= limit) {
            break;
        }
        // Pricing: a nonbasic column that raises `tau` when moved off its bound.
        let mut entering: Option<(usize, f64)> = None;
        let mut best_score = 0.0f64;
        for j in 0..ncols {
            if is_basic[j] {
                continue;
            }
            let dj = cost[j];
            let (eligible, sigma) = if at_upper[j] {
                (dj < -LP_COST_TOL, -1.0)
            } else {
                (dj > LP_COST_TOL, 1.0)
            };
            if !eligible {
                continue;
            }
            if bland {
                entering = Some((j, sigma));
                break;
            }
            if dj.abs() > best_score {
                best_score = dj.abs();
                entering = Some((j, sigma));
            }
        }
        let Some((j, sigma)) = entering else {
            optimal = true;
            break;
        };
        for i in 0..k {
            column[i] = tab[i * ncols + j];
        }

        // Ratio test over the basic variables (moving column j by sigma*theta
        // moves basic i by -sigma*theta*column[i]).
        let mut leave: Option<(usize, f64, bool)> = None;
        for i in 0..k {
            let alpha = sigma * column[i];
            let bound = upper_bound(basis[i]);
            let (ratio, to_upper) = if alpha > LP_PIVOT_TOL {
                (beta[i].max(0.0) / alpha, false)
            } else if alpha < -LP_PIVOT_TOL && bound.is_finite() {
                ((bound - beta[i]).max(0.0) / -alpha, true)
            } else {
                continue;
            };
            let take = match leave {
                None => true,
                Some((r, best, _)) => {
                    if ratio < best - LP_RATIO_TIE {
                        true
                    } else if ratio <= best + LP_RATIO_TIE {
                        if bland {
                            basis[i] < basis[r]
                        } else {
                            alpha.abs() > column[r].abs()
                        }
                    } else {
                        false
                    }
                }
            };
            if take {
                leave = Some((i, ratio, to_upper));
            }
        }
        let flip = upper_bound(j);
        let theta = match leave {
            Some((_, ratio, _)) if ratio < flip => ratio,
            _ if flip.is_finite() => flip,
            // Unbounded: impossible for a well-posed step LP; numerical trouble.
            _ => return None,
        };
        if theta <= LP_DEGENERATE_STEP {
            degenerate_run += 1;
            if degenerate_run >= LP_DEGENERATE_SWITCH {
                bland = true;
            }
        } else {
            degenerate_run = 0;
        }
        for i in 0..k {
            beta[i] -= sigma * theta * column[i];
        }
        match leave {
            Some((r, ratio, to_upper)) if ratio < flip => {
                let entering_value =
                    (if at_upper[j] { upper_bound(j) } else { 0.0 }) + sigma * theta;
                let leaving = basis[r];
                is_basic[leaving] = false;
                at_upper[leaving] = to_upper;
                let pivot = column[r];
                for c in 0..ncols {
                    tab[r * ncols + c] /= pivot;
                }
                let pivot_row: Vec<f64> = tab[r * ncols..(r + 1) * ncols].to_vec();
                for i in 0..k {
                    let factor = column[i];
                    if i == r || factor == 0.0 {
                        continue;
                    }
                    for c in 0..ncols {
                        tab[i * ncols + c] -= factor * pivot_row[c];
                    }
                }
                let factor = cost[j];
                if factor != 0.0 {
                    for c in 0..ncols {
                        cost[c] -= factor * pivot_row[c];
                    }
                }
                cost[j] = 0.0;
                for i in 0..k {
                    tab[i * ncols + j] = if i == r { 1.0 } else { 0.0 };
                }
                basis[r] = j;
                is_basic[j] = true;
                at_upper[j] = false;
                beta[r] = entering_value;
            }
            _ => {
                at_upper[j] = !at_upper[j];
            }
        }
        pivots += 1;
    }

    // Primal values of the scaled columns.
    let mut value = vec![0.0f64; ncols];
    for j in 0..ncols {
        if !is_basic[j] && at_upper[j] {
            value[j] = upper_bound(j);
        }
    }
    for (i, &bv) in basis.iter().enumerate() {
        value[bv] = beta[i];
    }
    let mut delta = base;
    for (c, &d) in cols.iter().enumerate() {
        let z = value[c].clamp(0.0, 1.0);
        delta[d] = (lower[d] + z * (upper[d] - lower[d])).clamp(lower[d], upper[d]);
    }
    let predicted = min_of(
        &rows
            .iter()
            .zip(margins)
            .map(|(row, &m)| m + row.iter().zip(&delta).map(|(j, x)| j * x).sum::<f64>())
            .collect::<Vec<f64>>(),
    );
    let duals: Vec<f64> = (0..k).map(|i| (-cost[slack0 + i]).max(0.0)).collect();
    let finite = predicted.is_finite() && delta.iter().all(|v| v.is_finite());
    if !finite {
        return None;
    }
    Some(MaximinLp {
        delta,
        predicted,
        duals,
        pivots,
        optimal,
    })
}

/// The favourable-rounding direction of a maximin step: the dual-weighted
/// gradient of the binding conjuncts (C8 weights the lagging rows by their
/// shortfall; here the LP's own multipliers say which rows bind). When the
/// multipliers are degenerate, the rows the step leaves lowest weigh equally.
pub(super) fn rounding_direction(lp: &MaximinLp, margins: &[f64], rows: &[Vec<f64>]) -> Vec<f64> {
    let dim = lp.delta.len();
    let total: f64 = lp.duals.iter().sum();
    let weights: Vec<f64> = if total.is_finite() && total > 0.0 {
        lp.duals.clone()
    } else {
        let predicted: Vec<f64> = rows
            .iter()
            .zip(margins)
            .map(|(row, &m)| m + row.iter().zip(&lp.delta).map(|(j, x)| j * x).sum::<f64>())
            .collect();
        let floor = min_of(&predicted);
        let band = 1e-12 * (1.0 + floor.abs());
        predicted
            .iter()
            .map(|&p| if p <= floor + band { 1.0 } else { 0.0 })
            .collect()
    };
    let mut direction = vec![0.0f64; dim];
    for (row, &w) in rows.iter().zip(&weights) {
        if w == 0.0 {
            continue;
        }
        for (d, &j) in row.iter().enumerate().take(dim) {
            direction[d] += w * j;
        }
    }
    direction
}

/// The hop kicks' deterministic generator (xorshift64*, seeded through
/// splitmix64): a verdict stays a pure function of the instance.
pub(super) struct HopRng(u64);

impl HopRng {
    pub(super) fn new(seed: u64) -> Self {
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        Self(if z == 0 { 0x2545_F491_4F6C_DD1D } else { z })
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform on `[0, 1)`.
    pub(super) fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// An iterate: an f32 point, its exact per-conjunct margins, their minimum,
/// and the evaluation record the Jacobian is taken from.
struct Iterate<T> {
    x: Vec<f32>,
    margins: Vec<f64>,
    min: f64,
    tape: T,
}

/// The lift's state across SLP runs and hops.
struct Lift<'a, T> {
    eval: &'a mut dyn FnMut(&[f32]) -> Option<(Vec<f64>, T)>,
    jacobian: &'a mut dyn FnMut(&T) -> Option<Vec<Vec<f64>>>,
    trusted: &'a mut dyn FnMut(&[f32]) -> Option<(bool, f64)>,
    box_lo: &'a [f32],
    box_hi: &'a [f32],
    width: Vec<f64>,
    deadline: Instant,
    outcome: F64RepairOutcome,
    stats: F64LiftStats,
    target: f64,
    escalations: u32,
    best_point: Vec<f32>,
    best_candidate: Option<(Vec<f32>, f64)>,
    last_checked: Option<Vec<f32>>,
}

impl<T> Lift<'_, T> {
    /// Track the best exact point seen.
    fn note_point(&mut self, x: &[f32], min: f64) {
        if min > self.outcome.best_margin_f64 {
            self.outcome.best_margin_f64 = min;
            self.best_point = x.to_vec();
        }
    }

    /// Consult the trusted forward once at `x` when its exact minimum margin
    /// is `>= 0` (#sbench-f64-attack's rule, unchanged). `Err` stops the lift.
    fn consult(&mut self, x: &[f32], f64_min: f64) -> Result<(), &'static str> {
        if f64_min < 0.0 || self.last_checked.as_deref() == Some(x) {
            return Ok(());
        }
        self.outcome.trusted_evals += 1;
        self.last_checked = Some(x.to_vec());
        let Some((violated, trusted_min)) = (self.trusted)(x) else {
            return Err("trusted forward failed");
        };
        if violated {
            self.outcome.candidates += 1;
            let joint = trusted_min.min(f64_min);
            if self
                .best_candidate
                .as_ref()
                .is_none_or(|(_, best_joint)| joint > *best_joint)
            {
                self.best_candidate = Some((x.to_vec(), joint));
                self.outcome.best_margin_ort = trusted_min;
            }
        } else if self.best_candidate.is_none() && trusted_min > self.outcome.best_margin_ort {
            self.outcome.best_margin_ort = trusted_min;
        }
        if f64_min >= self.target {
            if violated {
                return Err("target reached");
            }
            // Exact f64 clears the target, the trusted forward does not:
            // raise the target and keep lifting (at most TARGET_ESCALATIONS).
            if self.escalations >= TARGET_ESCALATIONS {
                return Err("target escalations exhausted");
            }
            self.escalations += 1;
            self.target *= 2.0;
        }
        Ok(())
    }

    /// One SLP run from `start`. A move is ACCEPTED only when the exact
    /// minimum margin at the rounded point is strictly larger than the
    /// current one; the Jacobian is recomputed only after an acceptance.
    /// Returns the run's last accepted iterate; `Err` stops the whole lift.
    fn slp(
        &mut self,
        start: Iterate<T>,
        rho_init: f64,
        rho_max: f64,
        max_iters: usize,
    ) -> Result<Iterate<T>, &'static str> {
        let mut cur = start;
        let mut rho = rho_init;
        // A consult-driven stop ("target reached", "target escalations
        // exhausted", "trusted forward failed") records THIS run's radius, as
        // every other exit does (telemetry only).
        if let Err(stop) = self.consult(&cur.x, cur.min) {
            self.stats.rho_final = rho;
            return Err(stop);
        }
        let dim = cur.x.len();
        let mut rows: Option<Vec<Vec<f64>>> = None;
        let mut accepted_in_run = false;
        let mut unproductive = 0usize;
        for _ in 0..max_iters {
            if Instant::now() >= self.deadline {
                self.stats.rho_final = rho;
                return Err("deadline");
            }
            self.stats.slp_iters += 1;
            self.outcome.newton_iters += 1;
            if rows.is_none() {
                let Some(fresh) = (self.jacobian)(&cur.tape) else {
                    self.stats.rho_final = rho;
                    return Err("exact vjp failed");
                };
                if fresh.len() != cur.margins.len() || fresh.iter().any(|row| row.len() != dim) {
                    self.stats.rho_final = rho;
                    return Err("exact vjp shape mismatch");
                }
                rows = Some(fresh);
            }
            let jac = rows.as_ref().expect("the Jacobian was computed above");
            let lower: Vec<f64> = (0..dim)
                .map(|d| {
                    (f32_to_f64_exact(self.box_lo[d]) - f32_to_f64_exact(cur.x[d]))
                        .max(-rho * self.width[d])
                        .min(0.0)
                })
                .collect();
            let upper: Vec<f64> = (0..dim)
                .map(|d| {
                    (f32_to_f64_exact(self.box_hi[d]) - f32_to_f64_exact(cur.x[d]))
                        .min(rho * self.width[d])
                        .max(0.0)
                })
                .collect();
            self.stats.lp_solves += 1;
            let mut next: Option<Iterate<T>> = None;
            match solve_maximin_lp(&cur.margins, jac, &lower, &upper, Some(self.deadline)) {
                None => self.stats.lp_failures += 1,
                Some(lp) => {
                    self.stats.lp_pivots += lp.pivots;
                    if !lp.optimal {
                        self.stats.lp_cut_short += 1;
                    }
                    let direction = rounding_direction(&lp, &cur.margins, jac);
                    let candidate: Vec<f32> = (0..dim)
                        .map(|d| {
                            round_toward(
                                f32_to_f64_exact(cur.x[d]) + lp.delta[d],
                                direction[d],
                                self.box_lo[d],
                                self.box_hi[d],
                            )
                        })
                        .collect();
                    if candidate != cur.x {
                        if let Some((margins, tape)) = (self.eval)(&candidate) {
                            let min = min_of(&margins);
                            self.note_point(&candidate, min);
                            if margins.len() == cur.margins.len() && min > cur.min {
                                next = Some(Iterate {
                                    x: candidate,
                                    margins,
                                    min,
                                    tape,
                                });
                            }
                        }
                    }
                }
            }
            match next {
                Some(accepted) => {
                    let gain = accepted.min - cur.min;
                    cur = accepted;
                    rows = None;
                    self.stats.accepted += 1;
                    accepted_in_run = true;
                    rho = (rho * RHO_GROW).min(rho_max);
                    unproductive = if gain < STALL_EPS {
                        unproductive + 1
                    } else {
                        0
                    };
                    if let Err(stop) = self.consult(&cur.x, cur.min) {
                        self.stats.rho_final = rho;
                        return Err(stop);
                    }
                }
                None => {
                    rho *= RHO_SHRINK;
                    if accepted_in_run {
                        unproductive += 1;
                    }
                }
            }
            if rho < RHO_MIN || unproductive >= STALL_WINDOW {
                break;
            }
        }
        self.stats.rho_final = rho;
        Ok(cur)
    }

    /// The first SLP run, then the basin hops. Returns the stop reason.
    fn run(
        &mut self,
        start: Iterate<T>,
        basin_hops: usize,
        rng_seed: u64,
        max_iters: usize,
    ) -> &'static str {
        let mut incumbent = match self.slp(start, RHO_INIT, RHO_MAX, max_iters.max(1)) {
            Ok(iterate) => iterate,
            Err(stop) => return stop,
        };
        if basin_hops == 0 {
            return "slp converged";
        }
        let mut rng = HopRng::new(rng_seed);
        let hop_iters = HOP_SLP_ITERS.min(max_iters.max(1));
        for _ in 0..basin_hops {
            if Instant::now() >= self.deadline {
                return "deadline";
            }
            self.stats.hops += 1;
            let radius = 10f64.powf(HOP_LOG10_MIN + (HOP_LOG10_MAX - HOP_LOG10_MIN) * rng.unit());
            let kicked: Vec<f32> = (0..incumbent.x.len())
                .map(|d| {
                    let kick = (2.0 * rng.unit() - 1.0) * radius * self.width[d];
                    let moved = (f32_to_f64_exact(incumbent.x[d]) + kick) as f32;
                    clamp_to_box(moved, self.box_lo[d], self.box_hi[d])
                })
                .collect();
            let Some((margins, tape)) = (self.eval)(&kicked) else {
                continue;
            };
            if margins.len() != incumbent.margins.len() {
                continue;
            }
            let min = min_of(&margins);
            self.note_point(&kicked, min);
            let start = Iterate {
                x: kicked,
                margins,
                min,
                tape,
            };
            let local = match self.slp(start, radius, HOP_RHO_MAX, hop_iters) {
                Ok(iterate) => iterate,
                Err(stop) => return stop,
            };
            if local.min > incumbent.min {
                incumbent = local;
                self.stats.hops_improved += 1;
            }
        }
        "hops exhausted"
    }

    /// C8's closing rule (the best exact point that cleared zero but was never
    /// shown to the trusted forward gets its one consultation), then the
    /// outcome.
    fn finish(mut self, stop: &'static str) -> (F64RepairOutcome, F64LiftStats) {
        if self.outcome.best_margin_f64 >= 0.0
            && self.last_checked.as_deref() != Some(&self.best_point[..])
        {
            let point = self.best_point.clone();
            let best = self.outcome.best_margin_f64;
            let _ = self.consult(&point, best);
        }
        self.outcome.stop = stop;
        self.stats.stop = stop;
        self.outcome.violation = self
            .best_candidate
            .as_ref()
            .and_then(|(point, _)| clamp_inside_box(point, self.box_lo, self.box_hi));
        if self.outcome.best_margin_f64 > self.outcome.initial_margin_f64 {
            self.outcome.guidance = Some((self.best_point.clone(), self.outcome.best_margin_f64));
        }
        (self.outcome, self.stats)
    }
}

/// The LP-maximin lift over caller-supplied oracles (module docs).
///
/// * `eval(x)` — the EXACT per-conjunct margins at the f32 point `x` (at its
///   witness input view) and the record `jacobian` needs; `None` fails the
///   point.
/// * `jacobian(record)` — one exact gradient row `d margin_i / d x` per
///   conjunct, flat over the input.
/// * `trusted(x)` — `(violated at zero tolerance, trusted min margin)`; it is
///   called ONLY at points whose exact minimum margin is `>= 0`.
///
/// `base_target` is the first stopping target (`f64_target_margin_ulps` f32
/// ULPs at the output magnitude); `max_iters` caps the first SLP run
/// ([`HOP_SLP_ITERS`] or less caps each hop's run). Returns `None` only on
/// malformed input or a seed `eval` refuses.
#[allow(clippy::too_many_arguments)]
pub(super) fn lp_maximin_lift_core<'a, T>(
    eval: &'a mut dyn FnMut(&[f32]) -> Option<(Vec<f64>, T)>,
    jacobian: &'a mut dyn FnMut(&T) -> Option<Vec<Vec<f64>>>,
    trusted: &'a mut dyn FnMut(&[f32]) -> Option<(bool, f64)>,
    box_lo: &'a [f32],
    box_hi: &'a [f32],
    seed: &[f32],
    base_target: f64,
    basin_hops: usize,
    rng_seed: u64,
    deadline: Instant,
    max_iters: usize,
) -> Option<(F64RepairOutcome, F64LiftStats)> {
    let dim = box_lo.len();
    if dim == 0
        || seed.len() != dim
        || box_hi.len() != dim
        || !(base_target.is_finite() && base_target > 0.0)
    {
        return None;
    }
    let width: Vec<f64> = box_lo
        .iter()
        .zip(box_hi)
        .map(|(&lo, &hi)| (f32_to_f64_exact(hi) - f32_to_f64_exact(lo)).max(0.0))
        .collect();
    if width.iter().all(|&w| w <= 0.0) {
        return None;
    }
    let x: Vec<f32> = seed
        .iter()
        .enumerate()
        .map(|(d, &v)| clamp_to_box(v, box_lo[d], box_hi[d]))
        .collect();
    let (margins, tape) = eval(&x)?;
    if margins.is_empty() || margins.iter().any(|m| !m.is_finite()) {
        return None;
    }
    let min = min_of(&margins);
    let mut lift = Lift {
        eval,
        jacobian,
        trusted,
        box_lo,
        box_hi,
        width,
        deadline,
        outcome: F64RepairOutcome {
            violation: None,
            guidance: None,
            initial_margin_f64: min,
            best_margin_f64: min,
            best_margin_ort: f64::NEG_INFINITY,
            target_delta: base_target,
            newton_iters: 0,
            trusted_evals: 0,
            candidates: 0,
            stop: "not started",
        },
        stats: F64LiftStats::default(),
        target: base_target,
        escalations: 0,
        best_point: x.clone(),
        best_candidate: None,
        last_checked: None,
    };
    let start = Iterate {
        x,
        margins,
        min,
        tape,
    };
    let stop = lift.run(start, basin_hops, rng_seed, max_iters);
    Some(lift.finish(stop))
}

/// The LP-maximin lift of one seed lineage on the exact-f64 twin: the same
/// scope, target and trusted-forward contract as
/// `f64_target_attack::f64_target_margin_repair` (single conjunction, at most
/// `MAX_DIM` inputs, a seed within `NEAR_BAND` of zero), with the step rule
/// and the hops of this module. `trusted_forward` returns the trusted flat
/// outputs at an f32 point (ONNX Runtime on the scored path).
///
/// `outcome.violation` is `Some` ONLY for a point the trusted forward
/// confirms at zero tolerance and whose exact-f64 margin is `>= 0`.
#[allow(clippy::too_many_arguments)]
pub(super) fn f64_lp_maximin_lift<F>(
    plan: &PointVjpF64Plan,
    mut trusted_forward: F,
    spec: &VnnLibSpec,
    box_lo: &[f32],
    box_hi: &[f32],
    emit_pin: &[Option<f64>],
    seed: &[f32],
    target_ulps: u32,
    basin_hops: usize,
    rng_seed: u64,
    deadline: Instant,
    max_iters: usize,
) -> Option<(F64RepairOutcome, F64LiftStats)>
where
    F: FnMut(&[f32]) -> Option<Vec<f64>>,
{
    let dim = box_lo.len();
    if dim == 0
        || dim > MAX_DIM
        || seed.len() != dim
        || box_hi.len() != dim
        || emit_pin.len() != dim
        || plan.input_dim() != dim
        || target_ulps == 0
    {
        return None;
    }
    let conjuncts = repair_conjuncts(spec)?;
    let output_dim = plan.output_dim();
    let conjunct_rows: Vec<Vec<f64>> = conjuncts
        .iter()
        .map(|c| margin_output_row(c, output_dim))
        .collect::<Option<Vec<Vec<f64>>>>()?;
    let x0: Vec<f32> = seed
        .iter()
        .enumerate()
        .map(|(d, &v)| clamp_to_box(v, box_lo[d], box_hi[d]))
        .collect();
    let (seed_margins, seed_outputs, _) = exact_margins(plan, conjuncts, emit_pin, &x0)?;
    if min_of(&seed_margins) < -NEAR_BAND {
        return None;
    }
    let base_target = target_delta_for(target_ulps, &seed_outputs, &conjunct_rows);
    if !(base_target.is_finite() && base_target > 0.0) {
        return None;
    }
    let mut eval = |x: &[f32]| {
        exact_margins(plan, conjuncts, emit_pin, x).map(|(margins, _, tape)| (margins, tape))
    };
    let mut jacobian = |tape: &PointVjpF64Tape| plan.vjp_rows(tape, &conjunct_rows);
    let mut trusted = |x: &[f32]| trusted_check(&mut trusted_forward, spec, conjuncts, emit_pin, x);
    lp_maximin_lift_core::<PointVjpF64Tape>(
        &mut eval,
        &mut jacobian,
        &mut trusted,
        box_lo,
        box_hi,
        &x0,
        base_target,
        basin_hops,
        rng_seed,
        deadline,
        max_iters,
    )
}

#[cfg(test)]
#[path = "f64_lp_maximin_tests.rs"]
mod tests;
