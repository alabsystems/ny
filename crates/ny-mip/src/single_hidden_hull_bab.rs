// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #hull-bab — wave-6 C4 phase 2: a deadline-bounded, best-first ReLU-split
//! branch-and-bound over the unstable NEGATIVE-coefficient hidden neurons,
//! with the certified per-neuron hull bound at every node
//! (`scratchpad/vbs/diag6/safenlp.md` Plan 2).
//!
//! # Why only `c_j < 0` neurons are branched
//!
//! With `m(x) = d + sum_j c_j relu(z_j(x))`, a `c_j > 0` term is CONVEX and its
//! LP relaxation (`s >= 0, s >= z`, i.e. one slope `alpha in [0,1]` in the
//! certificate) is already exact for minimization. Only the concave `c_j < 0`
//! terms carry a relaxation gap, so fixing all of them makes the node LP exact.
//!
//! # A node, and what its bound means
//!
//! A node is a set of splits `(k, phase)`. Its REGION is
//!
//! ```text
//! R = B  ∩  { x : sigma_k z_k(x) >= 0  for every split k },
//! sigma_k = +1 (active, z_k >= 0) | -1 (inactive, z_k <= 0).
//! ```
//!
//! The two children of a node split on one free neuron `j` with
//! `R ∩ {z_j >= 0}` and `R ∩ {z_j <= 0}`; their union is `R` (the boundary
//! lies in both), so the leaves of any tree cover `B`. The unsafe row is empty
//! iff `m > 0` on every leaf region.
//!
//! # The node certificate (the ONLY authority)
//!
//! [`certify_hull_node_lower_bound`] extends the root closed form
//! ([`super::certify_hull_lower_bound`]) with ONE Lagrangian term per split:
//!
//! * a split-ACTIVE neuron contributes `c_k z_k(x)` exactly (on `R`,
//!   `relu(z_k) = z_k`); a split-INACTIVE neuron contributes `0` (on `R`,
//!   `relu(z_k) = 0`). The downward-rounded `c_k` stays a valid lower
//!   coefficient because `relu(z_k) >= 0` on `R` in both cases;
//! * free neurons are relaxed over the WHOLE box exactly as at the root
//!   (`alpha` for `c_j > 0`, the affine majorant `g_j.x + h_j` for `c_j < 0`);
//! * every split's halfspace is charged by `beta_k >= 0` (weak duality):
//!   on `R`, `sigma_k z_k(x) >= 0`, so subtracting `beta_k sigma_k z_k(x)`
//!   can only LOWER the function.
//!
//! The sum is an affine minorant of `m` on `R`, minimized in closed form over
//! the box `B ⊇ R` with the same outward (directed) rounding as the root.
//! Negative or NaN `beta` is clamped to `0` (a negative `beta` would be
//! unsound); an infinite `beta`, a malformed split list (duplicate or
//! out-of-range neuron) or any non-finite intermediate returns `None`.
//! With no splits the computation is operation-for-operation the root
//! certifier's (pinned by a bit-identity test).
//!
//! A node's EFFECTIVE bound is `max(own certificate, parent's effective
//! bound)` — valid because the child's region is a subset of the parent's.
//!
//! # The proposer (NON-authoritative)
//!
//! Each node builds the root's compact projected hull LP with the split
//! neurons turned exact (active -> linear, inactive -> dropped), plus one
//! ELASTIC halfspace row per split, `sigma_k w_k.x + e_k >= -sigma_k b_k`
//! with `e_k >= 0` priced at a large penalty. The elastic column keeps every
//! node LP feasible, so an EMPTY region yields large duals (and a large,
//! still-certified bound) instead of "no duals". The row duals are mapped to
//! `(alpha, g)` as at the root and to `beta_k = y_k`; the certifier never
//! sees the elastic columns, the LP value or the float classification. The
//! triangle multipliers (with `beta = 0`) always run as a fallback candidate.
//!
//! # The search
//!
//! Best-first on the lowest effective bound, children bounded in parallel
//! batches (`std::thread::scope`, one AY session per node), deadline- and
//! node-cap-bounded. `Proved` is reported ONLY when the frontier is empty,
//! i.e. every leaf's effective bound is `> 0`. A node with no free concave
//! neuron left and a non-positive bound is `Exhausted` (its LP was exact, so
//! the row is very likely SAT); the search then stops UNDECIDED. Nothing here
//! can publish `sat`.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::time::{Duration, Instant};

use ay_milp::{Col, SolveOpts};

use super::{
    add_down, add_up, build_hull_lp, classify, guarded_hull_proposal, majorant_offset_upper,
    mul_down, mul_up, multipliers_from_duals, preactivation_bounds, require_hull_environment,
    sanitize_alpha, scale_interval, triangle_multipliers, HullBoundRoute, HullLp, HullLpStatus,
    HullMarginProblem, HullMultipliers, Role,
};
use crate::lp_propose_certify::{propose_then_certify_min, ReplayPolicy};

/// Default cap on bounded nodes (root included). Memory per open node is its
/// split list (at most one `(u32, bool)` per concave neuron).
pub const HULL_BAB_MAX_NODES: usize = 50_000;
/// Upper bound on parallel node workers.
pub const HULL_BAB_MAX_WORKERS: usize = 8;
/// Elastic-row price scale: `penalty_k = scale * (1 + sum_j |c_j| |w_j|_1) / |w_k|_1`.
/// Any positive value is sound (it only shapes the proposer); it should sit
/// well above the optimal halfspace multipliers so a feasible node's LP is
/// not relaxed by the elastic columns.
pub const HULL_BAB_ELASTIC_SCALE: f64 = 256.0;
/// Stack size for node worker threads (AY's float simplex is iterative; this
/// is headroom, not a measured need).
const HULL_BAB_WORKER_STACK_BYTES: usize = 16 << 20;
// The optional multiplier cache is bounded independently of the search tree.
// Oversized candidates simply retain the historical cold-LP path.
const WARM_MAX_CANDIDATE_CELLS: usize = 65_536;
const WARM_MAX_FRONTIER_CELLS: usize = 8 * 1024 * 1024;
const WARM_MAX_COORDINATES: usize = 256;

#[derive(Debug, Clone, PartialEq)]
struct WarmCandidate {
    multipliers: HullMultipliers,
    beta: Vec<f64>,
}

impl WarmCandidate {
    fn cells(&self) -> usize {
        self.multipliers.alpha.len() + self.multipliers.g.len() + self.beta.len()
    }
}

/// One ReLU split: hidden neuron `neuron` fixed to `z >= 0` (`active`) or
/// `z <= 0` (`!active`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HullSplit {
    /// Hidden neuron index.
    pub neuron: u32,
    /// `true`: `z_neuron >= 0`; `false`: `z_neuron <= 0`.
    pub active: bool,
}

/// Per-neuron phase of a node, or `None` on an out-of-range or duplicate
/// neuron (fail closed).
fn split_phases(hidden: usize, splits: &[HullSplit]) -> Option<Vec<Option<bool>>> {
    let mut phases = vec![None; hidden];
    for split in splits {
        let k = usize::try_from(split.neuron).ok()?;
        let slot = phases.get_mut(k)?;
        if slot.is_some() {
            return None;
        }
        *slot = Some(split.active);
    }
    Some(phases)
}

/// `beta` as the certifier uses it: `None` for `+-inf`, `0` for NaN or any
/// non-positive value (a negative multiplier would be UNSOUND), else itself.
#[inline]
fn sanitize_beta(raw: f64) -> Option<f64> {
    if raw.is_infinite() {
        None
    } else if raw > 0.0 {
        Some(raw)
    } else {
        Some(0.0)
    }
}

/// THE NODE AUTHORITY: a certified lower bound on `min_{x in R} m(x)` (real
/// arithmetic) for the node region `R` defined by `splits`, valid for ANY
/// `multipliers` and ANY `beta` (see the module doc). `beta[k]` charges
/// `splits[k]`.
///
/// Returns `None` on a shape mismatch, a malformed split list, an infinite
/// `beta`, a non-finite `g`, or any non-finite intermediate — never an unsound
/// number. With `splits` empty it performs exactly the root certifier's
/// operations.
#[must_use]
pub fn certify_hull_node_lower_bound(
    problem: &HullMarginProblem,
    splits: &[HullSplit],
    multipliers: &HullMultipliers,
    beta: &[f64],
) -> Option<f64> {
    require_hull_environment().ok()?;
    let n = problem.input_dim;
    let h = problem.hidden_dim;
    if multipliers.alpha.len() != h
        || multipliers.g.len() != h.checked_mul(n)?
        || beta.len() != splits.len()
    {
        return None;
    }
    let phases = split_phases(h, splits)?;
    let mut coef_lo = vec![0.0_f64; n];
    let mut coef_hi = vec![0.0_f64; n];
    let mut constant = problem.d;
    for j in 0..h {
        let cj = problem.c[j];
        let w = problem.w_row(j);
        let bj = problem.b1[j];
        match phases[j] {
            Some(true) => {
                // On R: z_j >= 0, so c_j relu(z_j) = c_j z_j exactly (and the
                // down-rounded c_j is a valid lower coefficient: z_j >= 0).
                if cj != 0.0 {
                    for i in 0..n {
                        coef_lo[i] = add_down(coef_lo[i], mul_down(cj, w[i]));
                        coef_hi[i] = add_up(coef_hi[i], mul_up(cj, w[i]));
                    }
                    constant = add_down(constant, mul_down(cj, bj));
                }
            }
            // On R: z_j <= 0, so relu(z_j) = 0 and the term vanishes.
            Some(false) => {}
            None => {
                // Free neuron: exactly the root certifier's treatment, valid
                // on the whole box and hence on R.
                if cj > 0.0 {
                    let alpha = sanitize_alpha(multipliers.alpha[j]);
                    if alpha == 0.0 {
                        continue;
                    }
                    let (k_lo, k_hi) = (mul_down(cj, alpha), mul_up(cj, alpha));
                    for i in 0..n {
                        let (lo, hi) = scale_interval(k_lo, k_hi, w[i]);
                        coef_lo[i] = add_down(coef_lo[i], lo);
                        coef_hi[i] = add_up(coef_hi[i], hi);
                    }
                    constant = add_down(constant, scale_interval(k_lo, k_hi, bj).0);
                } else if cj < 0.0 {
                    let g = &multipliers.g[j * n..(j + 1) * n];
                    if g.iter().any(|v| !v.is_finite()) {
                        return None;
                    }
                    let hj = majorant_offset_upper(w, bj, g, &problem.lower, &problem.upper)?;
                    for i in 0..n {
                        coef_lo[i] = add_down(coef_lo[i], mul_down(cj, g[i]));
                        coef_hi[i] = add_up(coef_hi[i], mul_up(cj, g[i]));
                    }
                    constant = add_down(constant, mul_down(cj, hj));
                }
            }
        }
    }
    // Weak duality for the split halfspaces: on R, sigma_k z_k(x) >= 0, so
    // `- beta_k sigma_k (w_k.x + b_k)` is <= 0 there. `q = -sigma_k beta_k`
    // is exact (a sign flip).
    for (split, &raw) in splits.iter().zip(beta) {
        let b = sanitize_beta(raw)?;
        if b == 0.0 {
            continue;
        }
        let k = usize::try_from(split.neuron).ok()?;
        let q = if split.active { -b } else { b };
        let w = problem.w_row(k);
        for i in 0..n {
            coef_lo[i] = add_down(coef_lo[i], mul_down(q, w[i]));
            coef_hi[i] = add_up(coef_hi[i], mul_up(q, w[i]));
        }
        constant = add_down(constant, mul_down(q, problem.b1[k]));
    }
    if !constant.is_finite() || coef_lo.iter().chain(&coef_hi).any(|v| !v.is_finite()) {
        return None;
    }
    let mut total = constant;
    for i in 0..n {
        let (l, u) = (problem.lower[i], problem.upper[i]);
        let term = mul_down(coef_lo[i], l)
            .min(mul_down(coef_lo[i], u))
            .min(mul_down(coef_hi[i], l))
            .min(mul_down(coef_hi[i], u));
        total = add_down(total, term);
    }
    total.is_finite().then_some(total)
}

/// Roles with the split neurons made exact: active -> `Live`, inactive ->
/// `Dead` (proposer shaping only; the certifier reads the splits itself).
fn effective_roles(roles: &[Role], phases: &[Option<bool>]) -> Vec<Role> {
    roles
        .iter()
        .zip(phases)
        .map(|(&role, &phase)| match (role, phase) {
            (Role::Inert, _) => Role::Inert,
            (_, Some(true)) => Role::Live,
            (_, Some(false)) => Role::Dead,
            (role, None) => role,
        })
        .collect()
}

/// The node proposer LP: the root hull LP over `effective_roles` plus one
/// elastic halfspace row per split. Returns the LP and, per split, the row
/// index whose dual is that split's `beta` (`None` for a neuron with an
/// all-zero weight row, whose halfspace does not depend on `x`).
fn build_node_lp(
    problem: &HullMarginProblem,
    roles: &[Role],
    bounds: &[(f64, f64)],
    splits: &[HullSplit],
) -> Option<(HullLp, Vec<Option<usize>>)> {
    let mut lp = build_hull_lp(problem, roles, bounds)?;
    let n = problem.input_dim;
    // `build_hull_lp` adds the n input columns FIRST, so column i is x_i.
    if lp.model.num_cols() < n {
        return None;
    }
    let mut scale = 1.0_f64;
    for j in 0..problem.hidden_dim {
        let norm: f64 = problem.w_row(j).iter().map(|v| v.abs()).sum();
        scale += problem.c[j].abs() * norm;
    }
    if !scale.is_finite() {
        return None;
    }
    let mut split_rows = Vec::with_capacity(splits.len());
    for split in splits {
        let k = usize::try_from(split.neuron).ok()?;
        let w = problem.w_row(k);
        let norm: f64 = w.iter().map(|v| v.abs()).sum();
        if norm.partial_cmp(&0.0) != Some(Ordering::Greater) || !norm.is_finite() {
            split_rows.push(None);
            continue;
        }
        let penalty = HULL_BAB_ELASTIC_SCALE * scale / norm;
        let sigma = if split.active { 1.0 } else { -1.0 };
        let rhs = -sigma * problem.b1[k];
        if !penalty.is_finite() || !rhs.is_finite() {
            return None;
        }
        let mut coeffs: Vec<(Col, f64)> = Vec::with_capacity(n + 1);
        for (i, &wi) in w.iter().enumerate() {
            if wi != 0.0 {
                coeffs.push((lp.model.col_at(i)?, sigma * wi));
            }
        }
        let slack = lp.model.add_col(0.0, f64::INFINITY);
        lp.objective.push((slack, penalty));
        coeffs.push((slack, 1.0));
        let row = lp.model.add_row(rhs, f64::INFINITY, &coeffs);
        split_rows.push(Some(row.index()));
    }
    Some((lp, split_rows))
}

/// Map a node LP's row duals to certifier multipliers and split `beta`s.
fn node_multipliers_from_duals(
    problem: &HullMarginProblem,
    roles: &[Role],
    lp: &HullLp,
    split_rows: &[Option<usize>],
    y: &[f64],
) -> Option<(HullMultipliers, Vec<f64>)> {
    // Checks `y.len() == num_rows`, so every `split_rows` index is in range.
    let multipliers = multipliers_from_duals(problem, roles, lp, y)?;
    let beta = split_rows
        .iter()
        .map(|row| {
            row.and_then(|r| y.get(r).copied())
                .filter(|v| v.is_finite() && *v > 0.0)
                .unwrap_or(0.0)
        })
        .collect();
    Some((multipliers, beta))
}

/// Plain-float box minimizer of the node's affine minorant (branching only;
/// flat coordinates go to the box midpoint).
fn node_coefficients(
    problem: &HullMarginProblem,
    phases: &[Option<bool>],
    splits: &[HullSplit],
    multipliers: &HullMultipliers,
    beta: &[f64],
) -> Vec<f64> {
    let n = problem.input_dim;
    let mut coef = vec![0.0_f64; n];
    for (j, phase) in phases.iter().enumerate() {
        let cj = problem.c[j];
        let w = problem.w_row(j);
        match phase {
            Some(true) => {
                for i in 0..n {
                    coef[i] += cj * w[i];
                }
            }
            Some(false) => {}
            None => {
                if cj > 0.0 {
                    let alpha = sanitize_alpha(multipliers.alpha[j]);
                    for i in 0..n {
                        coef[i] += cj * alpha * w[i];
                    }
                } else if cj < 0.0 {
                    let g = &multipliers.g[j * n..(j + 1) * n];
                    for i in 0..n {
                        coef[i] += cj * g[i];
                    }
                }
            }
        }
    }
    for (split, &raw) in splits.iter().zip(beta) {
        let b = if raw.is_finite() && raw > 0.0 {
            raw
        } else {
            0.0
        };
        if b == 0.0 {
            continue;
        }
        let q = if split.active { -b } else { b };
        let w = problem.w_row(split.neuron as usize);
        for i in 0..n {
            coef[i] += q * w[i];
        }
    }
    coef
}

fn node_minimizer(
    problem: &HullMarginProblem,
    phases: &[Option<bool>],
    splits: &[HullSplit],
    multipliers: &HullMultipliers,
    beta: &[f64],
) -> Vec<f64> {
    let coef = node_coefficients(problem, phases, splits, multipliers, beta);
    let magnitude = coef.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    let tol = 1e-12 * (1.0 + magnitude);
    (0..problem.input_dim)
        .map(|i| {
            let (l, u) = (problem.lower[i], problem.upper[i]);
            if coef[i] > tol {
                l
            } else if coef[i] < -tol {
                u
            } else {
                f64::midpoint(l, u)
            }
        })
        .collect()
}

/// Nominal exact-coordinate maximization of a piecewise-affine box minimum.
/// The finite endpoints and coefficient-zero breakpoints contain its maximum
/// on a bounded interval. One finite tail point is also tried for beta>=0.
/// This proposes NUMBERS ONLY; every retained proposal is independently replayed.
fn coordinate_proposal(
    problem: &HullMarginProblem,
    coefficients: &[f64],
    direction: &[f64],
    constant: f64,
    current: f64,
    upper: Option<f64>,
    deadline: Instant,
) -> Option<f64> {
    if !current.is_finite()
        || !constant.is_finite()
        || coefficients.len() != problem.input_dim
        || direction.len() != problem.input_dim
    {
        return None;
    }
    let base: Vec<_> = coefficients
        .iter()
        .zip(direction)
        .map(|(&a, &d)| a - current * d)
        .collect();
    if base.iter().chain(direction).any(|v| !v.is_finite()) {
        return None;
    }
    let mut candidates = vec![0.0, current];
    if let Some(cap) = upper {
        candidates.push(cap);
    }
    for (&a, &d) in base.iter().zip(direction) {
        let point = -a / d;
        if point.is_finite() && point >= 0.0 && upper.is_none_or(|cap| point <= cap) {
            candidates.push(point);
        }
    }
    if upper.is_none() {
        let last = candidates.iter().copied().fold(0.0_f64, f64::max);
        let tail = last + last.abs().max(1.0);
        if tail.is_finite() {
            candidates.push(tail);
        }
    }
    let mut best: Option<(f64, f64)> = None;
    for point in candidates {
        if Instant::now() >= deadline {
            return None;
        }
        let mut score = constant * point;
        for i in 0..problem.input_dim {
            let a = base[i] + point * direction[i];
            score += (a * problem.lower[i]).min(a * problem.upper[i]);
        }
        if score.is_finite() && best.is_none_or(|(value, _)| score > value) {
            best = Some((score, point));
        }
    }
    best.map(|(_, point)| point)
}

/// Reuse the parent's multipliers under the child's exact phase substitution.
/// Only positive-head alpha and nonnegative split beta coordinates are varied;
/// the expensive negative-head majorants remain cached. Every sweep is replayed
/// by the unchanged directed-rounding node authority. Interrupted optional work
/// returns the last completed certificate, never a nominal coordinate score.
fn warm_child(
    problem: &HullMarginProblem,
    roles: &[Role],
    splits: &[HullSplit],
    seed: &WarmCandidate,
    deadline: Instant,
) -> Option<(f64, WarmCandidate)> {
    if Instant::now() >= deadline
        || seed.cells() > WARM_MAX_CANDIDATE_CELLS
        || roles.len() != problem.hidden_dim
        || seed.multipliers.alpha.len() != problem.hidden_dim
        || seed.multipliers.g.len() != problem.hidden_dim.checked_mul(problem.input_dim)?
        || seed.beta.len().checked_add(1)? != splits.len()
    {
        return None;
    }
    let phases = split_phases(problem.hidden_dim, splits)?;
    let mut current = seed.clone();
    current.beta.push(0.0);
    let value =
        certify_hull_node_lower_bound(problem, splits, &current.multipliers, &current.beta)?;
    if Instant::now() >= deadline {
        return None;
    }
    let mut best = (value, current.clone());
    let mut count = 0usize;
    for _ in 0..3 {
        let mut coefficients = node_coefficients(
            problem,
            &phases,
            splits,
            &current.multipliers,
            &current.beta,
        );
        for k in 0..splits.len() + problem.hidden_dim {
            if count >= WARM_MAX_COORDINATES || Instant::now() >= deadline {
                return Some(best);
            }
            let (direction, constant, old, cap) = if k < splits.len() {
                let split = splits[k];
                let j = split.neuron as usize;
                let sign = if split.active { -1.0 } else { 1.0 };
                (
                    problem
                        .w_row(j)
                        .iter()
                        .map(|w| sign * w)
                        .collect::<Vec<_>>(),
                    sign * problem.b1[j],
                    current.beta[k],
                    None,
                )
            } else {
                let j = k - splits.len();
                if roles[j] != Role::PosUnstable || phases[j].is_some() {
                    continue;
                }
                (
                    problem
                        .w_row(j)
                        .iter()
                        .map(|w| problem.c[j] * w)
                        .collect::<Vec<_>>(),
                    problem.c[j] * problem.b1[j],
                    sanitize_alpha(current.multipliers.alpha[j]),
                    Some(1.0),
                )
            };
            count += 1;
            let Some(next) = coordinate_proposal(
                problem,
                &coefficients,
                &direction,
                constant,
                old,
                cap,
                deadline,
            ) else {
                return Some(best);
            };
            for (a, d) in coefficients.iter_mut().zip(direction) {
                *a += (next - old) * d;
            }
            if k < splits.len() {
                current.beta[k] = next;
            } else {
                current.multipliers.alpha[k - splits.len()] = next;
            }
        }
        let Some(value) =
            certify_hull_node_lower_bound(problem, splits, &current.multipliers, &current.beta)
        else {
            return Some(best);
        };
        if Instant::now() >= deadline {
            return Some(best);
        }
        if value > best.0 {
            best = (value, current.clone());
        }
        if best.0 > 0.0 {
            break;
        }
    }
    Some(best)
}

/// The branching neuron: among free concave unstable neurons, the largest
/// weighted hull gap `|c_j| (g_j.x^ + h_j - relu(z_j(x^)))` at the
/// minorant's minimizer `x^`; when every gap is ~0, the largest static
/// triangle gap `|c_j| u_j (-l_j) / (u_j - l_j)`. `None` when no free concave
/// unstable neuron remains. Heuristic only: it never affects soundness.
fn choose_branch(
    problem: &HullMarginProblem,
    roles: &[Role],
    bounds: &[(f64, f64)],
    x_hat: &[f64],
    multipliers: &HullMultipliers,
) -> Option<usize> {
    let n = problem.input_dim;
    let mut best_gap: Option<(f64, usize)> = None;
    let mut best_static: Option<(f64, usize)> = None;
    for (j, role) in roles.iter().enumerate() {
        if *role != Role::NegUnstable {
            continue;
        }
        let abs_c = problem.c[j].abs();
        let (lo, hi) = bounds[j];
        let static_gap = abs_c * hi * (-lo) / (hi - lo);
        let static_gap = if static_gap.is_finite() {
            static_gap
        } else {
            0.0
        };
        if best_static.is_none_or(|(s, _)| static_gap > s) {
            best_static = Some((static_gap, j));
        }
        let w = problem.w_row(j);
        let g = &multipliers.g[j * n..(j + 1) * n];
        let Some(hj) = majorant_offset_upper(w, problem.b1[j], g, &problem.lower, &problem.upper)
        else {
            continue;
        };
        let mut z = problem.b1[j];
        let mut gx = 0.0_f64;
        for i in 0..n {
            z += w[i] * x_hat[i];
            gx += g[i] * x_hat[i];
        }
        let gap = abs_c * (gx + hj - z.max(0.0));
        if gap.is_finite() && best_gap.is_none_or(|(s, _)| gap > s) {
            best_gap = Some((gap, j));
        }
    }
    match best_gap {
        Some((gap, j)) if gap > 1e-12 => Some(j),
        _ => best_static.map(|(_, j)| j),
    }
}

/// What bounding one node produced.
#[derive(Debug, Clone, PartialEq)]
struct NodeOutcome {
    /// Best certified value over the triangle and LP candidates (own, not
    /// yet combined with the parent's).
    certified: Option<f64>,
    /// Independently checked signed mask floor; never the old tuple's scalar.
    cached: Option<f64>,
    route: Option<HullBoundRoute>,
    lp_status: HullLpStatus,
    /// Plain-float LP value. TELEMETRY ONLY.
    lp_value: Option<f64>,
    /// Neuron to split next; `None` when no free concave neuron remains.
    branch: Option<usize>,
    warm: Option<WarmCandidate>,
}

impl NodeOutcome {
    fn unsupported_environment() -> Self {
        Self {
            certified: None,
            cached: None,
            route: None,
            lp_status: HullLpStatus::UnsupportedEnvironment,
            lp_value: None,
            branch: None,
            warm: None,
        }
    }
}

/// Bound one node: triangle candidate, then the elastic node LP through the
/// propose/certify seam; every candidate goes through
/// [`certify_hull_node_lower_bound`].
fn bound_node_counted(
    problem: &HullMarginProblem,
    roles: &[Role],
    bounds: &[(f64, f64)],
    splits: &[HullSplit],
    deadline: Instant,
    seed: Option<&WarmCandidate>,
    calls: Option<&AtomicUsize>,
) -> NodeOutcome {
    bound_node_cached(problem, roles, bounds, splits, deadline, seed, calls, None)
}

#[allow(clippy::too_many_arguments)]
fn bound_node_cached(
    problem: &HullMarginProblem,
    roles: &[Role],
    bounds: &[(f64, f64)],
    splits: &[HullSplit],
    deadline: Instant,
    seed: Option<&WarmCandidate>,
    calls: Option<&AtomicUsize>,
    cache: Option<&super::mip_cuts::SignedChildBound<'_>>,
) -> NodeOutcome {
    bound_node_cached_native(
        problem, roles, bounds, splits, deadline, seed, calls, cache, None,
    )
}
fn bound_node_cached_native(
    problem: &HullMarginProblem,
    roles: &[Role],
    bounds: &[(f64, f64)],
    splits: &[HullSplit],
    deadline: Instant,
    seed: Option<&WarmCandidate>,
    calls: Option<&AtomicUsize>,
    cache: Option<&super::mip_cuts::SignedChildBound<'_>>,
    native: Option<&super::mip_cuts::SignedNativeRows<'_>>,
) -> NodeOutcome {
    if require_hull_environment().is_err() {
        return NodeOutcome::unsupported_environment();
    }
    let cache = cache.filter(|cache| cache.matches_problem(problem));
    let Some(phases) = split_phases(problem.hidden_dim, splits) else {
        return NodeOutcome {
            certified: None,
            cached: None,
            route: None,
            lp_status: HullLpStatus::NotBuilt,
            lp_value: None,
            branch: None,
            warm: None,
        };
    };
    let node_roles = effective_roles(roles, &phases);
    let zero_beta = vec![0.0_f64; splits.len()];
    let triangle = triangle_multipliers(problem, &node_roles, bounds);
    let mut best: Option<(f64, HullBoundRoute)> =
        certify_hull_node_lower_bound(problem, splits, &triangle, &zero_beta)
            .map(|v| (v, HullBoundRoute::Triangle));
    let mut best_multipliers = triangle;
    let mut best_beta = zero_beta;

    if let Some((value, proposal)) =
        seed.and_then(|seed| warm_child(problem, roles, splits, seed, deadline))
    {
        if best.is_none_or(|(bound, _)| value > bound) {
            best = Some((value, HullBoundRoute::Subgradient));
            best_multipliers = proposal.multipliers;
            best_beta = proposal.beta;
        }
    }
    if require_hull_environment().is_err() {
        return NodeOutcome::unsupported_environment();
    }
    let mut cached = None;
    if !best.is_some_and(|(value, _)| value > 0.) {
        if let Some(cache) = cache {
            cached = cache.lower_zero(splits, deadline, &mut || Instant::now() < deadline);
            // The completed first pass precharges this bounded beta scan too.
            if cached.is_some() && best_beta.iter().any(|b| *b != 0.) {
                if let Some(value) = cache.lower(splits, &best_beta, deadline, &mut || {
                    Instant::now() < deadline
                }) {
                    cached = Some(cached.map_or(value, |old| old.max(value)));
                }
            }
        }
    }
    // A completed closed-form child certificate already discharges the node.
    // The proposer LP cannot add authority and need not be constructed at all.
    if !splits.is_empty()
        && Instant::now() < deadline
        && (best.is_some_and(|(value, _)| value > 0.0) || cached.is_some_and(|v| v > 0.))
    {
        if cached.is_some_and(|v| v > 0.) && !best.is_some_and(|(v, _)| v > 0.) {
            if let Some(cache) = cache {
                cache.skipped_lp();
            }
        }
        return NodeOutcome {
            certified: best.map(|v| v.0),
            cached,
            route: best.map(|v| v.1),
            lp_status: HullLpStatus::Skipped,
            lp_value: None,
            branch: None,
            warm: None,
        };
    }

    let free_unstable = node_roles
        .iter()
        .any(|r| matches!(r, Role::PosUnstable | Role::NegUnstable));
    let mut lp_value = None;
    let lp_status = if !free_unstable && splits.is_empty() {
        HullLpStatus::Empty
    } else if Instant::now() >= deadline {
        HullLpStatus::Skipped
    } else {
        match build_node_lp(problem, &node_roles, bounds, splits) {
            None => HullLpStatus::NotBuilt,
            Some((mut lp, split_rows)) => {
                let old = |lp: &HullLp, y: &[f64]| -> Option<f64> {
                    require_hull_environment().ok()?;
                    let (m, beta) =
                        node_multipliers_from_duals(problem, &node_roles, lp, &split_rows, y)?;
                    certify_hull_node_lower_bound(problem, splits, &m, &beta)
                };
                let run = |lp: &HullLp,
                           certify: &mut dyn FnMut(&[f64]) -> Option<f64>,
                           solve_end: Instant| {
                    let remaining = solve_end.saturating_duration_since(Instant::now());
                    let opts = SolveOpts::new()
                        .with_time_limit(remaining.max(Duration::from_millis(1)))
                        .with_deadline(solve_end);
                    guarded_hull_proposal(|| {
                        if let Some(calls) = calls {
                            calls.fetch_add(1, AtomicOrdering::Relaxed);
                        }
                        propose_then_certify_min(
                            &lp.model,
                            &lp.objective,
                            &opts,
                            ReplayPolicy::Skip,
                            certify,
                        )
                    })
                };
                let native = native.filter(|rows| Instant::now() < rows.end());
                let (proposal, native_committed) = if let Some(native) = native {
                    use super::mip_cuts::NativeVisit;
                    let native_end = deadline.min(native.end());
                    let native_run =
                        |lp: &HullLp, certify: &mut dyn FnMut(&[f64]) -> Option<f64>| {
                            run(lp, certify, native_end)
                        };
                    match native.with_node(
                        problem,
                        splits,
                        &mut lp,
                        &split_rows,
                        native_end,
                        &mut || Instant::now() < native_end,
                        old,
                        native_run,
                    ) {
                        NativeVisit::Declined => {
                            let mut certify = |y: &[f64]| old(&lp, y);
                            (Some(run(&lp, &mut certify, deadline)), false)
                        }
                        NativeVisit::Stopped => (None, false),
                        NativeVisit::Committed(result) => (result, true),
                    }
                } else {
                    let mut certify = |y: &[f64]| old(&lp, y);
                    (Some(run(&lp, &mut certify, deadline)), false)
                };
                match proposal {
                    None => HullLpStatus::Skipped,
                    Some(Err(_)) => return NodeOutcome::unsupported_environment(),
                    Some(Ok(Err(_))) => HullLpStatus::Error,
                    Some(Ok(Ok(outcome))) => {
                        lp_value = outcome.float_dual_value.map(|v| v + lp.constant);
                        if !outcome.duals_available {
                            HullLpStatus::NoDuals
                        } else if let (Some(value), Some(y)) =
                            (outcome.closed_form, outcome.best_duals.as_deref())
                        {
                            let mut old_decoded = false;
                            if let Some((m, beta)) = node_multipliers_from_duals(
                                problem,
                                &node_roles,
                                &lp,
                                &split_rows,
                                y,
                            ) {
                                old_decoded = true;
                                // Fresh LP beta remains advice for the independent cache floor.
                                if let Some(cache) = cache {
                                    if let Some(candidate) =
                                        cache.lower(splits, &beta, deadline, &mut || {
                                            Instant::now() < deadline
                                        })
                                    {
                                        cached = Some(
                                            cached.map_or(candidate, |old| old.max(candidate)),
                                        );
                                    }
                                }
                                // These remain warm/branching ADVICE. The new
                                // scalar may be attained by the independent pricer.
                                if best.is_none_or(|(b, _)| value > b) {
                                    best_multipliers = m;
                                    best_beta = beta;
                                }
                            }
                            if (native_committed || old_decoded)
                                && value.is_finite()
                                && best.is_none_or(|(b, _)| value > b)
                            {
                                best = Some((value, HullBoundRoute::Lp));
                            }
                            HullLpStatus::Solved
                        } else {
                            HullLpStatus::NoCertificate
                        }
                    }
                }
            }
        }
    };
    if require_hull_environment().is_err() {
        return NodeOutcome::unsupported_environment();
    }
    let x_hat = node_minimizer(problem, &phases, splits, &best_multipliers, &best_beta);
    let branch = choose_branch(problem, &node_roles, bounds, &x_hat, &best_multipliers);
    NodeOutcome {
        certified: best.map(|(v, _)| v),
        cached,
        route: best.map(|(_, r)| r),
        lp_status,
        lp_value,
        branch,
        warm: (problem
            .hidden_dim
            .saturating_mul(problem.input_dim)
            .saturating_add(problem.hidden_dim)
            .saturating_add(splits.len())
            <= WARM_MAX_CANDIDATE_CELLS)
            .then_some(WarmCandidate {
                multipliers: best_multipliers,
                beta: best_beta,
            }),
    }
}

/// Bound every node spec, in parallel chunks when `workers > 1`. An entry is
/// `None` only when its worker thread could not be spawned or panicked; the
/// caller treats that as a stop, never as a bound.
fn bound_many(
    problem: &HullMarginProblem,
    roles: &[Role],
    bounds: &[(f64, f64)],
    specs: &[Vec<HullSplit>],
    seeds: &[Option<&WarmCandidate>],
    deadline: Instant,
    workers: usize,
    calls: &AtomicUsize,
    cache: Option<&super::mip_cuts::SignedChildBound<'_>>,
) -> Vec<Option<NodeOutcome>> {
    bound_many_native(
        problem, roles, bounds, specs, seeds, deadline, workers, calls, cache, None,
    )
}
fn bound_many_native(
    problem: &HullMarginProblem,
    roles: &[Role],
    bounds: &[(f64, f64)],
    specs: &[Vec<HullSplit>],
    seeds: &[Option<&WarmCandidate>],
    deadline: Instant,
    workers: usize,
    calls: &AtomicUsize,
    cache: Option<&super::mip_cuts::SignedChildBound<'_>>,
    native: Option<&super::mip_cuts::SignedNativeRows<'_>>,
) -> Vec<Option<NodeOutcome>> {
    let workers = workers.clamp(1, specs.len().max(1));
    if workers == 1 {
        return specs
            .iter()
            .zip(seeds)
            .map(|(splits, seed)| {
                Some(bound_node_cached_native(
                    problem,
                    roles,
                    bounds,
                    splits,
                    deadline,
                    *seed,
                    Some(calls),
                    cache,
                    native,
                ))
            })
            .collect();
    }
    let chunk = specs.len().div_ceil(workers);
    std::thread::scope(|scope| {
        let handles: Vec<_> = specs
            .chunks(chunk)
            .zip(seeds.chunks(chunk))
            .map(|(part, seeds)| {
                let handle = std::thread::Builder::new()
                    .name("ny-hull-bab".to_string())
                    .stack_size(HULL_BAB_WORKER_STACK_BYTES)
                    .spawn_scoped(scope, move || {
                        part.iter()
                            .zip(seeds)
                            .map(|(splits, seed)| {
                                bound_node_cached_native(
                                    problem,
                                    roles,
                                    bounds,
                                    splits,
                                    deadline,
                                    *seed,
                                    Some(calls),
                                    cache,
                                    native,
                                )
                            })
                            .collect::<Vec<_>>()
                    });
                (part.len(), handle.ok())
            })
            .collect();
        let mut out = Vec::with_capacity(specs.len());
        for (len, handle) in handles {
            match handle.map(|h| h.join()) {
                Some(Ok(results)) if results.len() == len => {
                    out.extend(results.into_iter().map(Some));
                }
                _ => out.extend(std::iter::repeat_with(|| None).take(len)),
            }
        }
        out
    })
}

/// Why the search stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HullBabStop {
    /// A coordinator or node worker refused binary64 arithmetic assumptions.
    UnsupportedEnvironment,
    /// The frontier emptied: every leaf's effective bound is `> 0`.
    Proved,
    /// The deadline passed with open nodes.
    Deadline,
    /// The node cap was reached with open nodes.
    NodeCap,
    /// A node with no free concave neuron still has a non-positive bound
    /// (its LP is exact: the row is very likely SAT).
    Exhausted,
    /// A node worker could not be spawned or panicked.
    WorkerFailure,
}

impl HullBabStop {
    /// Stable lowercase label for observables.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::UnsupportedEnvironment => "unsupported_environment",
            Self::Proved => "proved",
            Self::Deadline => "deadline",
            Self::NodeCap => "node_cap",
            Self::Exhausted => "exhausted",
            Self::WorkerFailure => "worker_failure",
        }
    }
}

/// Search limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HullBabOptions {
    /// Cap on bounded nodes (root included); soft by at most one batch.
    pub max_nodes: usize,
    /// Parallel node workers (`>= 1`).
    pub workers: usize,
}

impl Default for HullBabOptions {
    fn default() -> Self {
        Self {
            max_nodes: HULL_BAB_MAX_NODES,
            workers: 1,
        }
    }
}

/// The outcome of one hull BaB.
#[derive(Debug, Clone, PartialEq)]
pub struct HullBabReport {
    /// Unstable neurons with a nonzero margin coefficient (float estimate).
    pub unstable: usize,
    /// Of those, the concave (`c_j < 0`) ones — the branching candidates.
    pub neg_unstable: usize,
    /// Of those, the convex (`c_j > 0`) ones.
    pub pos_unstable: usize,
    /// The root's certified bound.
    pub root_certified: Option<f64>,
    /// Effective checked root/cache floor before optional native pricing.
    /// Telemetry only; it does not establish a verifying stop.
    pub root_before_native: Option<f64>,
    /// Which proposer produced the root's certified bound.
    pub root_route: Option<HullBoundRoute>,
    /// The root's plain-float LP value. TELEMETRY ONLY.
    pub root_lp_value: Option<f64>,
    /// What happened to the root LP proposer.
    pub root_lp_status: HullLpStatus,
    /// Why the search stopped. `Proved` is the ONLY verifying stop.
    pub stop: HullBabStop,
    /// Nodes bounded (root included).
    pub nodes: usize,
    /// Leaves closed with an effective bound `> 0`.
    pub closed: usize,
    /// Open nodes left on the frontier.
    pub open: usize,
    /// Lowest effective bound on the open frontier (`None` when empty).
    pub worst: Option<f64>,
    /// Deepest bounded node.
    pub max_depth: usize,
    /// Failed LP proposals; intentional warm/deadline skips are not failures.
    pub lp_failures: usize,
    /// Actual calls into the LP proposal seam, including failed/worker calls.
    pub proposal_lp_calls: usize,
    /// Of those, root calls; zero when an opaque completed root is reused.
    pub root_lp_calls: usize,
    /// Workers used.
    pub workers: usize,
}

impl HullBabReport {
    fn unsupported_environment(workers: usize) -> Self {
        Self {
            unstable: 0,
            neg_unstable: 0,
            pos_unstable: 0,
            root_certified: None,
            root_before_native: None,
            root_route: None,
            root_lp_value: None,
            root_lp_status: HullLpStatus::UnsupportedEnvironment,
            stop: HullBabStop::UnsupportedEnvironment,
            nodes: 0,
            closed: 0,
            open: 1,
            worst: None,
            max_depth: 0,
            lp_failures: 0,
            proposal_lp_calls: 0,
            root_lp_calls: 0,
            workers,
        }
    }

    /// True iff every leaf was certified `> 0`: the unsafe row is empty.
    #[must_use]
    pub fn verified(&self) -> bool {
        self.stop == HullBabStop::Proved && self.open == 0
    }
}

/// An open node on the best-first frontier.
#[derive(Debug, Clone)]
struct OpenNode {
    /// Effective bound: max(own certificate, parent's), `-inf` if none.
    bound: f64,
    seq: u64,
    depth: usize,
    splits: Vec<HullSplit>,
    branch: Option<usize>,
    warm: Option<WarmCandidate>,
}

impl PartialEq for OpenNode {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for OpenNode {}

impl PartialOrd for OpenNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OpenNode {
    /// `BinaryHeap` pops the maximum: the LOWEST bound is the maximum, ties
    /// broken toward the OLDEST node (deterministic).
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .bound
            .total_cmp(&self.bound)
            .then_with(|| other.seq.cmp(&self.seq))
    }
}

fn lp_failed(status: HullLpStatus) -> bool {
    !matches!(
        status,
        HullLpStatus::Solved | HullLpStatus::Empty | HullLpStatus::Skipped
    )
}

/// Run the hull BaB until the frontier empties (`Proved`), the deadline
/// passes, the node cap is reached, a node is exhausted, or a worker fails.
/// Never panics on admitted input by itself; `Proved` requires every leaf's
/// effective certified bound to be `> 0`.
#[must_use]
pub fn hull_bab(
    problem: &HullMarginProblem,
    deadline: Instant,
    options: HullBabOptions,
) -> HullBabReport {
    if require_hull_environment().is_err() {
        return HullBabReport::unsupported_environment(options.workers.max(1));
    }
    let bounds = preactivation_bounds(problem);
    let roles = classify(problem, &bounds);
    let calls = AtomicUsize::new(0);
    let root = bound_node_counted(problem, &roles, &bounds, &[], deadline, None, Some(&calls));
    let root_lp_calls = calls.load(AtomicOrdering::Relaxed);
    run_tree(
        problem,
        deadline,
        options,
        &bounds,
        &roles,
        root,
        &calls,
        root_lp_calls,
    )
}

fn run_tree(
    problem: &HullMarginProblem,
    deadline: Instant,
    options: HullBabOptions,
    bounds: &[(f64, f64)],
    roles: &[Role],
    root: NodeOutcome,
    calls: &AtomicUsize,
    root_lp_calls: usize,
) -> HullBabReport {
    run_tree_with_cache(
        problem,
        deadline,
        options,
        bounds,
        roles,
        root,
        calls,
        root_lp_calls,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_tree_with_cache(
    problem: &HullMarginProblem,
    deadline: Instant,
    options: HullBabOptions,
    bounds: &[(f64, f64)],
    roles: &[Role],
    root: NodeOutcome,
    calls: &AtomicUsize,
    root_lp_calls: usize,
    cache: Option<&super::mip_cuts::SignedChildBound<'_>>,
) -> HullBabReport {
    run_tree_with_cache_native(
        problem,
        deadline,
        options,
        bounds,
        roles,
        root,
        calls,
        root_lp_calls,
        cache,
        None,
    )
}
#[allow(clippy::too_many_arguments)]
fn run_tree_with_cache_native(
    problem: &HullMarginProblem,
    deadline: Instant,
    options: HullBabOptions,
    bounds: &[(f64, f64)],
    roles: &[Role],
    root: NodeOutcome,
    calls: &AtomicUsize,
    root_lp_calls: usize,
    cache: Option<&super::mip_cuts::SignedChildBound<'_>>,
    native: Option<&super::mip_cuts::SignedNativeRows<'_>>,
) -> HullBabReport {
    let neg_unstable = roles.iter().filter(|r| **r == Role::NegUnstable).count();
    let pos_unstable = roles.iter().filter(|r| **r == Role::PosUnstable).count();
    let workers = options.workers.clamp(1, HULL_BAB_MAX_WORKERS);
    if root.lp_status == HullLpStatus::UnsupportedEnvironment {
        let mut report = HullBabReport::unsupported_environment(workers);
        report.proposal_lp_calls = calls.load(AtomicOrdering::Relaxed);
        report.root_lp_calls = root_lp_calls;
        return report;
    }
    let mut report = HullBabReport {
        unstable: neg_unstable + pos_unstable,
        neg_unstable,
        pos_unstable,
        root_certified: root.certified,
        root_before_native: None,
        root_route: root.route,
        root_lp_value: root.lp_value,
        root_lp_status: root.lp_status,
        stop: HullBabStop::Deadline,
        nodes: 1,
        closed: 0,
        open: 0,
        worst: None,
        max_depth: 0,
        lp_failures: usize::from(lp_failed(root.lp_status)),
        proposal_lp_calls: 0,
        root_lp_calls,
        workers,
    };
    let root_bound = root
        .certified
        .unwrap_or(f64::NEG_INFINITY)
        .max(root.cached.unwrap_or(f64::NEG_INFINITY));
    let mut frontier = BinaryHeap::new();
    let mut seq = 0u64;
    let mut warm_cells = 0usize;
    if root_bound > 0.0 {
        report.closed = 1;
    } else {
        warm_cells = root.warm.as_ref().map_or(0, WarmCandidate::cells);
        frontier.push(OpenNode {
            bound: root_bound,
            seq,
            depth: 0,
            splits: Vec::new(),
            branch: root.branch,
            warm: root.warm,
        });
    }

    let stop = loop {
        if require_hull_environment().is_err() {
            break HullBabStop::UnsupportedEnvironment;
        }
        if frontier.is_empty() {
            break HullBabStop::Proved;
        }
        if Instant::now() >= deadline {
            break HullBabStop::Deadline;
        }
        if report.nodes >= options.max_nodes {
            break HullBabStop::NodeCap;
        }
        let mut batch: Vec<OpenNode> = Vec::with_capacity(workers);
        while batch.len() < workers {
            match frontier.pop() {
                Some(node) => batch.push(node),
                None => break,
            }
        }
        if batch.iter().any(|node| node.branch.is_none()) {
            frontier.extend(batch);
            break HullBabStop::Exhausted;
        }
        for node in &batch {
            warm_cells -= node.warm.as_ref().map_or(0, WarmCandidate::cells);
        }
        let mut specs: Vec<Vec<HullSplit>> = Vec::with_capacity(2 * batch.len());
        let mut parents: Vec<usize> = Vec::with_capacity(2 * batch.len());
        for (index, node) in batch.iter().enumerate() {
            let Some(j) = node.branch else { continue };
            let Ok(neuron) = u32::try_from(j) else {
                continue;
            };
            for active in [true, false] {
                let mut splits = node.splits.clone();
                splits.push(HullSplit { neuron, active });
                specs.push(splits);
                parents.push(index);
            }
        }
        if specs.len() != 2 * batch.len() {
            // A branch index that does not fit u32 (cannot happen under the
            // admission caps): stop rather than drop a region.
            frontier.extend(batch);
            break HullBabStop::WorkerFailure;
        }
        let seeds: Vec<_> = parents
            .iter()
            .map(|&parent| batch[parent].warm.as_ref())
            .collect();
        let bounded = if native.is_some() {
            bound_many_native(
                problem, roles, bounds, &specs, &seeds, deadline, workers, calls, cache, native,
            )
        } else {
            bound_many(
                problem, roles, bounds, &specs, &seeds, deadline, workers, calls, cache,
            )
        };
        let outcomes: Option<Vec<NodeOutcome>> = bounded.into_iter().collect();
        let outcomes = match outcomes {
            Some(outcomes) if outcomes.len() == specs.len() => outcomes,
            _ => {
                // Never drop a region: the parents go back and the search stops.
                frontier.extend(batch);
                break HullBabStop::WorkerFailure;
            }
        };
        if require_hull_environment().is_err()
            || outcomes
                .iter()
                .any(|outcome| outcome.lp_status == HullLpStatus::UnsupportedEnvironment)
        {
            // These parent regions remain uncovered. Never drop them because
            // a worker used an unsupported environment.
            frontier.extend(batch);
            break HullBabStop::UnsupportedEnvironment;
        }
        for ((splits, parent), mut outcome) in specs.into_iter().zip(parents).zip(outcomes) {
            let parent_node = &batch[parent];
            let depth = parent_node.depth + 1;
            report.nodes += 1;
            report.max_depth = report.max_depth.max(depth);
            report.lp_failures += usize::from(lp_failed(outcome.lp_status));
            let own = outcome
                .certified
                .unwrap_or(f64::NEG_INFINITY)
                .max(outcome.cached.unwrap_or(f64::NEG_INFINITY));
            let bound = own.max(parent_node.bound);
            if bound > 0.0 {
                report.closed += 1;
            } else {
                seq += 1;
                if let Some(seed) = &outcome.warm {
                    if seed.cells() > WARM_MAX_FRONTIER_CELLS - warm_cells {
                        outcome.warm = None;
                    } else {
                        warm_cells += seed.cells();
                    }
                }
                frontier.push(OpenNode {
                    bound,
                    seq,
                    depth,
                    splits,
                    branch: outcome.branch,
                    warm: outcome.warm,
                });
            }
        }
    };
    report.stop = stop;
    report.open = frontier.len();
    report.worst = frontier.peek().map(|node| node.bound);
    report.proposal_lp_calls = calls.load(AtomicOrdering::Relaxed);
    // Empty-frontier processing can cross the clock after the last child.
    // A completed numerical cover is never permission for late publication.
    if Instant::now() >= deadline
        || ny_core::instance_budget::deadline().is_some_and(|d| Instant::now() >= d)
    {
        report.stop = HullBabStop::Deadline;
    }
    if require_hull_environment().is_err() {
        report.stop = HullBabStop::UnsupportedEnvironment;
    }
    report
}

/// Test-only access to the SAME production native worker; creates no token,
/// quote, Model or alternate pricing path. Callers must supply a genuinely
/// committed/bound source-licensed handle from the normal publication route.
#[cfg(test)]
pub(in crate::single_hidden_hull) fn native_rows_test_node(
    problem: &HullMarginProblem,
    splits: &[HullSplit],
    native: &super::mip_cuts::SignedNativeRows<'_>,
    end: Instant,
) -> (Option<f64>, Option<f64>, HullLpStatus, usize) {
    let bounds = preactivation_bounds(problem);
    let roles = classify(problem, &bounds);
    let calls = AtomicUsize::new(0);
    let out = bound_node_cached_native(
        problem,
        &roles,
        &bounds,
        splits,
        end,
        None,
        Some(&calls),
        None,
        Some(native),
    );
    (
        out.certified,
        out.cached,
        out.lp_status,
        calls.load(AtomicOrdering::Relaxed),
    )
}

/// Exercise the actual cache-before-LP worker without a branch-order assumption.
#[cfg(test)]
pub(in crate::single_hidden_hull) fn signed_child_test_node(
    problem: &HullMarginProblem,
    splits: &[HullSplit],
    cache: &super::mip_cuts::SignedChildBound<'_>,
    end: Instant,
) -> (Option<f64>, Option<f64>, HullLpStatus, usize) {
    let bounds = preactivation_bounds(problem);
    let roles = classify(problem, &bounds);
    let calls = AtomicUsize::new(0);
    let out = bound_node_cached(
        problem,
        &roles,
        &bounds,
        splits,
        end,
        None,
        Some(&calls),
        Some(cache),
    );
    (
        out.certified,
        out.cached,
        out.lp_status,
        calls.load(AtomicOrdering::Relaxed),
    )
}

#[cfg(test)]
#[path = "single_hidden_hull_bab_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "single_hidden_hull_bab_reuse_tests.rs"]
mod reuse_tests;

#[path = "single_hidden_hull_bab_seed.rs"]
mod seed;
pub(super) use seed::{from_completed, from_completed_with_cache, from_completed_with_native};
