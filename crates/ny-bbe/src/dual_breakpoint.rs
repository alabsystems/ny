// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Untrusted concave dual search with exact breakpoint placement along each ray.
//!
//! The objective is a negative L1 norm plus a linear multiplier price. Along
//! any feasible ray its slope only decreases: a reduced cost crossing zero
//! drops the slope by twice the absolute reduced-cost direction. Sorting these
//! crossings locates the ray maximum without a learning rate or a grid. The
//! search first follows a projected supergradient, then tries coordinate rays
//! when simultaneous motion stalls at a kink. If both stall, a bounded pair
//! rescue follows directions that cancel a nearly zero reduced cost. Every
//! accepted point strictly improves a fresh evaluation from the request's
//! initial reduced costs.
//!
//! This is f64 SEARCH arithmetic without a rounding certificate. Only finite,
//! nonnegative multiplier hints leave this module; its gain has no verdict
//! authority. Buffers use O(generators + split depth) memory. A finite search
//! ceiling bounds rays whose dual objective is unbounded; initial multipliers
//! above that ceiling are preserved and may move down.

use std::time::Instant;

use ny_core::search::{
    DualMultipliers, DualSearchOutcome, DualSearchRequest, Refusal, SupportSearchStats, Untrusted,
};

mod support;
use support::Support;

/// Search limit, never a constraint or assumption in the certified problem.
const MULTIPLIER_LIMIT: f64 = 1.0e12;
const POLL_MASK: usize = 255;

/// Pair rescue admission: maximum generators in a request. These are search
/// work controls, never restrictions on the certified frozen problem.
pub const MAX_PAIR_GENERATORS: usize = 16_384;
/// Pair rescue admission: maximum generator-by-row terms.
pub const MAX_PAIR_ROW_TERMS: usize = 250_000;
/// Maximum pair-column inspections in one rescue round, excluding its one
/// gradient pass. An inspection reads at most two row coefficients.
pub const MAX_PAIR_GENERATOR_SCANS: usize = 131_072;
/// Maximum calls to the existing breakpoint ray solver in one rescue round.
pub const MAX_PAIR_RAYS: usize = 8;
const PAIR_COORDINATES: usize = 8;
const PAIR_RAYS_PER_PAIR: usize = 2;
const KINK_REL_TOL: f64 = 64.0 * f64::EPSILON;

#[derive(Clone, Copy)]
struct PairCoordinate {
    row: usize,
    far: bool,
    gradient: f64,
    pressure: f64,
    coupling: f64,
}

impl PairCoordinate {
    const EMPTY: Self = Self {
        row: 0,
        far: false,
        gradient: 0.0,
        pressure: 0.0,
        coupling: 0.0,
    };

    fn coefficient(self, req: &DualSearchRequest<'_>, j: usize) -> f64 {
        let row = &req.rows[self.row];
        row.sign * if self.far { -row.a[j] } else { row.a[j] }
    }

    fn preferred_to(self, other: Self) -> bool {
        self.pressure > other.pressure
            || (self.pressure == other.pressure
                && (self.coupling > other.coupling
                    || (self.coupling == other.coupling
                        && self.gradient.abs() > other.gradient.abs())))
    }
}

/// Relative proximity only guides untrusted search. The ray and final fresh
/// score still use the actual reduced cost, including a nonzero residual.
fn near_kink(rc: f64, initial: f64) -> bool {
    rc.abs() <= KINK_REL_TOL * (1.0 + initial.abs())
}

fn pair_admitted(n: usize, depth: usize) -> bool {
    depth >= 2 && n <= MAX_PAIR_GENERATORS && n.saturating_mul(depth) <= MAX_PAIR_ROW_TERMS
}

#[derive(Clone, Copy)]
struct Crossing {
    at: f64,
    drop: f64,
}

#[derive(Debug, Clone, Copy)]
enum Halt {
    Deadline,
    Nonfinite,
}

fn poll(deadline: Option<Instant>) -> Result<(), Halt> {
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        Err(Halt::Deadline)
    } else {
        Ok(())
    }
}

fn finite(value: f64) -> Result<f64, Halt> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(Halt::Nonfinite)
    }
}

/// Dense search admission baseline, including the result. Optional support
/// compilation is admitted separately against the SAME request cap and falls
/// back to this representation if either of its phases cannot fit.
pub fn working_bytes(generators: usize, depth: usize) -> u64 {
    let generators = u64::try_from(generators).unwrap_or(u64::MAX);
    let depth = u64::try_from(depth).unwrap_or(u64::MAX);
    generators
        .saturating_mul(40)
        .saturating_add(depth.saturating_mul(48))
        .saturating_add(1024)
}

fn zeros(len: usize, need: u64, cap: u64) -> Result<Vec<f64>, Refusal> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| Refusal::Memory { need, cap })?;
    values.resize(len, 0.0);
    Ok(values)
}

struct Work {
    support: Support,
    support_stats: SupportSearchStats,
    rc: Vec<f64>,
    rc_dir: Vec<f64>,
    candidate_rc: Vec<f64>,
    l0: Vec<f64>,
    l1: Vec<f64>,
    d0: Vec<f64>,
    d1: Vec<f64>,
    candidate_l0: Vec<f64>,
    candidate_l1: Vec<f64>,
    crossings: Vec<Crossing>,
    objective: f64,
    pair_scans: usize,
    pair_rays: usize,
}

impl Work {
    #[cfg(test)]
    fn new(req: &DualSearchRequest<'_>, objective: f64, need: u64) -> Result<Self, Refusal> {
        Self::with_support(
            req,
            objective,
            need,
            Support::dense(),
            SupportSearchStats::default(),
        )
    }

    fn with_support(
        req: &DualSearchRequest<'_>,
        objective: f64,
        need: u64,
        support: Support,
        support_stats: SupportSearchStats,
    ) -> Result<Self, Refusal> {
        let n = support.len(req.rc.len());
        let d = req.rows.len();
        let cap = req.mem_cap_bytes;
        let mut crossings = Vec::new();
        crossings
            .try_reserve_exact(n)
            .map_err(|_| Refusal::Memory { need, cap })?;
        let mut out = Self {
            support,
            support_stats,
            rc: zeros(n, need, cap)?,
            rc_dir: zeros(n, need, cap)?,
            candidate_rc: zeros(n, need, cap)?,
            l0: zeros(d, need, cap)?,
            l1: zeros(d, need, cap)?,
            d0: zeros(d, need, cap)?,
            d1: zeros(d, need, cap)?,
            candidate_l0: zeros(d, need, cap)?,
            candidate_l1: zeros(d, need, cap)?,
            crossings,
            objective,
            pair_scans: 0,
            pair_rays: 0,
        };
        for (j, rc) in out.rc.iter_mut().enumerate() {
            *rc = req.rc[out.support.original(j)];
        }
        out.l0.copy_from_slice(req.l0);
        out.l1.copy_from_slice(req.l1);
        if out.support_stats.compiled != 0 {
            let mut norm = out.support.norm();
            for (j, &rc) in out.rc.iter().enumerate() {
                norm += out.support.weight(j) as f64 * rc.abs();
            }
            let price: f64 = req
                .rows
                .iter()
                .enumerate()
                .map(|(i, row)| req.l0[i] * row.b0 + req.l1[i] * row.b1)
                .sum();
            out.objective = -norm - price;
            if !out.objective.is_finite() {
                return Err(Refusal::Invalid(
                    "compiled starting support overflow".into(),
                ));
            }
        }
        let bytes = out.support.bytes()
            + (out.rc.capacity()
                + out.rc_dir.capacity()
                + out.candidate_rc.capacity()
                + out.l0.capacity()
                + out.l1.capacity()
                + out.d0.capacity()
                + out.d1.capacity()
                + out.candidate_l0.capacity()
                + out.candidate_l1.capacity()) as u64
                * 8
            + out.crossings.capacity() as u64 * 16
            + 1024;
        out.support_stats.peak_bytes = out.support_stats.peak_bytes.max(bytes);
        Ok(out)
    }

    /// One supergradient of the negative L1 norm; at zero use its midpoint.
    fn projected_direction(&mut self, req: &DualSearchRequest<'_>) -> Result<(), Halt> {
        let mut scale = 0.0f64;
        for (i, row) in req.rows.iter().enumerate() {
            poll(req.deadline)?;
            let mut dot = self.support.dot(i);
            for (j, &rc) in self.rc.iter().enumerate() {
                if j & POLL_MASK == 0 {
                    poll(req.deadline)?;
                }
                // f64::signum(0) is +1: explicit zero is load-bearing.
                let s = if rc > 0.0 {
                    1.0
                } else if rc < 0.0 {
                    -1.0
                } else {
                    0.0
                };
                let a = row.a[self.support.original(j)];
                self.support_stats.search_reads += 1;
                self.support_stats.dense_search_reads += self.support.weight(j) as u64;
                dot += (s * a) * self.support.weight(j) as f64;
            }
            self.support_stats.dense_search_reads +=
                self.support_stats.untouched_columns + self.support_stats.sign_folded_columns;
            let g0 = finite(-row.sign * dot - row.b0)?;
            let g1 = finite(row.sign * dot - row.b1)?;
            self.d0[i] = project(g0, self.l0[i], req.l0[i]);
            self.d1[i] = if row.vacuous {
                0.0
            } else {
                project(g1, self.l1[i], req.l1[i])
            };
            scale = scale.max(self.d0[i].abs()).max(self.d1[i].abs());
        }
        if scale > 0.0 {
            for i in 0..req.rows.len() {
                if i & POLL_MASK == 0 {
                    poll(req.deadline)?;
                }
                self.d0[i] /= scale;
                self.d1[i] /= scale;
            }
        }
        Ok(())
    }

    /// Maximize the piecewise-linear objective on the finite feasible ray.
    fn ray(&mut self, req: &DualSearchRequest<'_>) -> Result<bool, Halt> {
        poll(req.deadline)?;
        self.rc_dir.fill(0.0);
        self.crossings.clear();
        let mut limit = f64::MAX;
        let mut price_dir = 0.0;
        let mut folded_dir = 0.0;
        let mut moving = false;
        for (i, row) in req.rows.iter().enumerate() {
            poll(req.deadline)?;
            for (value, dir, initial) in [
                (self.l0[i], self.d0[i], req.l0[i]),
                (self.l1[i], self.d1[i], req.l1[i]),
            ] {
                if dir > 0.0 {
                    limit = limit.min((initial.max(MULTIPLIER_LIMIT) - value) / dir);
                    moving = true;
                } else if dir < 0.0 {
                    limit = limit.min(-value / dir);
                    moving = true;
                }
            }
            price_dir += self.d0[i] * row.b0 + self.d1[i] * row.b1;
            let coeff = row.sign * (self.d0[i] - self.d1[i]);
            if coeff != 0.0 {
                folded_dir += coeff * self.support.dot(i);
                for j in 0..self.rc.len() {
                    if j & POLL_MASK == 0 {
                        poll(req.deadline)?;
                    }
                    self.rc_dir[j] += coeff * row.a[self.support.original(j)];
                    self.support_stats.search_reads += 1;
                    self.support_stats.dense_search_reads += self.support.weight(j) as u64;
                }
                self.support_stats.dense_search_reads +=
                    self.support_stats.untouched_columns + self.support_stats.sign_folded_columns;
            }
        }
        if !moving || limit <= 0.0 {
            return Ok(false);
        }
        finite(limit)?;
        let mut slope = -finite(price_dir)? - finite(folded_dir)?;
        let mut ungrouped_events = 0u64;
        for (j, (&rc, &dir)) in self.rc.iter().zip(&self.rc_dir).enumerate() {
            if j & POLL_MASK == 0 {
                poll(req.deadline)?;
            }
            finite(dir)?;
            let weight = self.support.weight(j) as f64;
            slope -= weight
                * if rc > 0.0 {
                    dir
                } else if rc < 0.0 {
                    -dir
                } else {
                    dir.abs()
                };
            if dir != 0.0 {
                let at = -rc / dir;
                if at.is_finite() && at > 0.0 && at <= limit {
                    self.crossings.push(Crossing {
                        at,
                        drop: finite(weight * (2.0 * dir.abs()))?,
                    });
                    ungrouped_events += self.support.weight(j) as u64;
                }
            }
        }
        finite(slope)?;
        if slope <= 0.0 {
            return Ok(false);
        }
        poll(req.deadline)?;
        self.support_stats.events += self.crossings.len() as u64;
        self.support_stats.ungrouped_events += ungrouped_events;
        self.crossings
            .sort_unstable_by(|a, b| a.at.total_cmp(&b.at));
        poll(req.deadline)?;
        let mut step = limit;
        for (i, crossing) in self.crossings.iter().enumerate() {
            if i & POLL_MASK == 0 {
                poll(req.deadline)?;
            }
            slope -= crossing.drop;
            if slope <= 0.0 {
                step = crossing.at;
                break;
            }
        }
        // Recompute from the supplied base, rather than accumulating reduced
        // cost updates from earlier iterations. An untrusted numeric artifact
        // therefore cannot be amplified by repeated recursive updates.
        let objective = self.evaluate_candidate(req, step)?;
        if objective > self.objective {
            std::mem::swap(&mut self.rc, &mut self.candidate_rc);
            std::mem::swap(&mut self.l0, &mut self.candidate_l0);
            std::mem::swap(&mut self.l1, &mut self.candidate_l1);
            self.objective = objective;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn evaluate_candidate(&mut self, req: &DualSearchRequest<'_>, step: f64) -> Result<f64, Halt> {
        for (j, rc) in self.candidate_rc.iter_mut().enumerate() {
            *rc = req.rc[self.support.original(j)];
        }
        let mut price = 0.0;
        let mut norm = self.support.norm();
        for (i, row) in req.rows.iter().enumerate() {
            poll(req.deadline)?;
            let l0 =
                finite(self.l0[i] + step * self.d0[i])?.clamp(0.0, req.l0[i].max(MULTIPLIER_LIMIT));
            let l1 =
                finite(self.l1[i] + step * self.d1[i])?.clamp(0.0, req.l1[i].max(MULTIPLIER_LIMIT));
            self.candidate_l0[i] = l0;
            self.candidate_l1[i] = l1;
            price += l0 * row.b0 + l1 * row.b1;
            let coeff = finite(row.sign * ((l0 - req.l0[i]) - (l1 - req.l1[i])))?;
            if coeff != 0.0 {
                norm += coeff * self.support.dot(i);
                for j in 0..self.candidate_rc.len() {
                    if j & POLL_MASK == 0 {
                        poll(req.deadline)?;
                    }
                    self.candidate_rc[j] += coeff * row.a[self.support.original(j)];
                    self.support_stats.search_reads += 1;
                    self.support_stats.dense_search_reads += self.support.weight(j) as u64;
                }
                self.support_stats.dense_search_reads +=
                    self.support_stats.untouched_columns + self.support_stats.sign_folded_columns;
            }
        }
        for (j, &value) in self.candidate_rc.iter().enumerate() {
            if j & POLL_MASK == 0 {
                poll(req.deadline)?;
            }
            norm += self.support.weight(j) as f64 * finite(value)?.abs();
        }
        finite(-norm - price)
    }

    /// A projected joint direction can cross a kink immediately, even while
    /// a single coordinate admits a useful ray. Test both feasible directions
    /// in deterministic row order and accept the first strict improvement.
    fn rescue(&mut self, req: &DualSearchRequest<'_>) -> Result<bool, Halt> {
        for i in 0..req.rows.len() {
            poll(req.deadline)?;
            for far in [false, true] {
                if far && req.rows[i].vacuous {
                    continue;
                }
                for sign in [1.0, -1.0] {
                    self.d0.fill(0.0);
                    self.d1.fill(0.0);
                    let (value, initial) = if far {
                        (self.l1[i], req.l1[i])
                    } else {
                        (self.l0[i], req.l0[i])
                    };
                    if project(sign, value, initial) == 0.0 {
                        continue;
                    }
                    if far {
                        self.d1[i] = sign;
                    } else {
                        self.d0[i] = sign;
                    }
                    if self.ray(req)? {
                        return Ok(true);
                    }
                }
            }
        }
        Ok(false)
    }

    /// The midpoint supergradient and every individual coordinate may fail
    /// while a coupled feasible direction improves this nonseparable L1
    /// objective. Follow a small collection of sign-manifold edges: cancel
    /// one nearly zero reduced cost with two coordinates, then let the exact
    /// ray solver account for every other cost. This is a bounded heuristic,
    /// not an optimality test or a general manifold-sampling implementation.
    fn pair_rescue(&mut self, req: &DualSearchRequest<'_>) -> Result<bool, Halt> {
        poll(req.deadline)?;
        self.pair_scans = 0;
        self.pair_rays = 0;
        if !pair_admitted(self.rc.len(), req.rows.len()) {
            return Ok(false);
        }
        let mut coordinates = [PairCoordinate::EMPTY; PAIR_COORDINATES];
        let mut count = 0;
        // One admitted n*D pass chooses coordinates with gradient pressure
        // and coupling at the current kinks. Fixed arrays add no heap work.
        for (i, row) in req.rows.iter().enumerate() {
            poll(req.deadline)?;
            let mut dot = self.support.dot(i);
            let mut coupling: f64 = 0.0;
            for j in 0..self.rc.len() {
                if j & POLL_MASK == 0 {
                    poll(req.deadline)?;
                }
                let rc = self.rc[j];
                let original = self.support.original(j);
                let a = row.a[original];
                self.support_stats.search_reads += 1;
                self.support_stats.dense_search_reads += self.support.weight(j) as u64;
                dot += self.support.weight(j) as f64
                    * if rc > 0.0 {
                        a
                    } else if rc < 0.0 {
                        -a
                    } else {
                        0.0
                    };
                if near_kink(rc, req.rc[original]) {
                    coupling = coupling.max(a.abs());
                }
            }
            self.support_stats.dense_search_reads +=
                self.support_stats.untouched_columns + self.support_stats.sign_folded_columns;
            if coupling == 0.0 {
                continue;
            }
            for far in [false, true] {
                if far && row.vacuous {
                    continue;
                }
                let (gradient, value, initial) = if far {
                    (finite(row.sign * dot - row.b1)?, self.l1[i], req.l1[i])
                } else {
                    (finite(-row.sign * dot - row.b0)?, self.l0[i], req.l0[i])
                };
                let coordinate = PairCoordinate {
                    row: i,
                    far,
                    gradient,
                    pressure: project(gradient, value, initial).abs(),
                    coupling,
                };
                let mut at = 0;
                while at < count && !coordinate.preferred_to(coordinates[at]) {
                    at += 1;
                }
                if at == PAIR_COORDINATES {
                    continue;
                }
                let next_count = (count + 1).min(PAIR_COORDINATES);
                for k in (at + 1..next_count).rev() {
                    coordinates[k] = coordinates[k - 1];
                }
                coordinates[at] = coordinate;
                count = next_count;
            }
        }
        for first in 0..count {
            for second in first + 1..count {
                poll(req.deadline)?;
                let a = coordinates[first];
                let b = coordinates[second];
                // Near and far cuts of one row have opposite coefficients;
                // reserve the bounded work for coupling distinct rows.
                if a.row == b.row {
                    continue;
                }
                let mut tried = [(0.0, 0.0); PAIR_RAYS_PER_PAIR];
                let mut trials = 0;
                for j in 0..self.rc.len() {
                    if self.pair_scans == MAX_PAIR_GENERATOR_SCANS
                        || self.pair_rays == MAX_PAIR_RAYS
                    {
                        return Ok(false);
                    }
                    self.pair_scans += 1;
                    if j & POLL_MASK == 0 {
                        poll(req.deadline)?;
                    }
                    let original = self.support.original(j);
                    if !near_kink(self.rc[j], req.rc[original]) {
                        continue;
                    }
                    self.support_stats.search_reads += 2;
                    self.support_stats.dense_search_reads += 2 * self.support.weight(j) as u64;
                    let ca = a.coefficient(req, original);
                    let cb = b.coefficient(req, original);
                    if ca == 0.0 || cb == 0.0 {
                        continue;
                    }
                    // Normalize before any product; huge finite coefficients
                    // cannot overflow while constructing a cancellation ray.
                    let scale = ca.abs().max(cb.abs());
                    let pa = cb / scale;
                    let pb = -ca / scale;
                    if pa == 0.0 || pb == 0.0 {
                        continue;
                    }
                    for orientation in [1.0, -1.0] {
                        let pa = orientation * pa;
                        let pb = orientation * pb;
                        let (va, ia) = if a.far {
                            (self.l1[a.row], req.l1[a.row])
                        } else {
                            (self.l0[a.row], req.l0[a.row])
                        };
                        let (vb, ib) = if b.far {
                            (self.l1[b.row], req.l1[b.row])
                        } else {
                            (self.l0[b.row], req.l0[b.row])
                        };
                        let pressure = a.gradient * pa + b.gradient * pb;
                        if project(pa, va, ia) == 0.0
                            || project(pb, vb, ib) == 0.0
                            || !pressure.is_finite()
                            || pressure <= 0.0
                            || tried[..trials].contains(&(pa, pb))
                        {
                            continue;
                        }
                        self.d0.fill(0.0);
                        self.d1.fill(0.0);
                        if a.far {
                            self.d1[a.row] = pa;
                        } else {
                            self.d0[a.row] = pa;
                        }
                        if b.far {
                            self.d1[b.row] = pb;
                        } else {
                            self.d0[b.row] = pb;
                        }
                        self.pair_rays += 1;
                        tried[trials] = (pa, pb);
                        trials += 1;
                        if self.ray(req)? {
                            return Ok(true);
                        }
                        // Only one orientation has positive midpoint pressure.
                        break;
                    }
                    if trials == PAIR_RAYS_PER_PAIR {
                        break;
                    }
                }
            }
        }
        Ok(false)
    }
}

fn project(gradient: f64, value: f64, initial: f64) -> f64 {
    if (gradient < 0.0 && value == 0.0)
        || (gradient > 0.0 && value >= initial.max(MULTIPLIER_LIMIT))
    {
        0.0
    } else {
        gradient
    }
}

fn validate(req: &DualSearchRequest<'_>) -> Result<f64, Refusal> {
    let invalid = |why: &str| Refusal::Invalid(why.to_owned());
    if poll(req.deadline).is_err() {
        return Err(Refusal::Deadline);
    }
    if req.rows.len() != req.l0.len() || req.rows.len() != req.l1.len() {
        return Err(invalid("dual row and multiplier lengths differ"));
    }
    if req.rows.is_empty() || req.rc.is_empty() {
        return Err(Refusal::Nothing("no dual rows or generators"));
    }
    let mut norm = 0.0;
    for (j, &rc) in req.rc.iter().enumerate() {
        if j & POLL_MASK == 0 && poll(req.deadline).is_err() {
            return Err(Refusal::Deadline);
        }
        if !rc.is_finite() {
            return Err(invalid("nonfinite dual reduced cost"));
        }
        norm += rc.abs();
    }
    let mut price = 0.0;
    for (i, row) in req.rows.iter().enumerate() {
        if poll(req.deadline).is_err() {
            return Err(Refusal::Deadline);
        }
        if row.a.len() != req.rc.len()
            || !(row.sign == 1.0 || row.sign == -1.0)
            || !row.b0.is_finite()
            || !row.b1.is_finite()
        {
            return Err(invalid(
                "malformed dual row, orientation, or right hand side",
            ));
        }
        // `!(finite && x >= 0)` in its minimal form: a finite value fails
        // `x >= 0` exactly when `x < 0`, and NaN/inf fail `is_finite`.
        if !req.l0[i].is_finite() || req.l0[i] < 0.0 || !req.l1[i].is_finite() || req.l1[i] < 0.0 {
            return Err(invalid("dual multipliers must be finite and nonnegative"));
        }
        for (j, &a) in row.a.iter().enumerate() {
            if j & POLL_MASK == 0 && poll(req.deadline).is_err() {
                return Err(Refusal::Deadline);
            }
            if !a.is_finite() {
                return Err(invalid("nonfinite dual row coefficient"));
            }
        }
        price += req.l0[i] * row.b0 + req.l1[i] * row.b1;
    }
    let objective = -norm - price;
    if !objective.is_finite() {
        return Err(invalid("dual starting objective is nonfinite"));
    }
    Ok(objective)
}

/// Search a box-support dual, returning only an untrusted multiplier hint.
pub fn search(req: &DualSearchRequest<'_>) -> Result<DualSearchOutcome, Refusal> {
    // Validate and admit before any search allocation.
    let baseline = validate(req)?;
    let need = working_bytes(req.rc.len(), req.rows.len());
    if need > req.mem_cap_bytes {
        return Err(Refusal::Memory {
            need,
            cap: req.mem_cap_bytes,
        });
    }
    if poll(req.deadline).is_err() {
        return Err(Refusal::Deadline);
    }
    let mut receipt = SupportSearchStats::default();
    let support = Support::compile(req, &mut receipt);
    let mut work = Work::with_support(req, baseline, need, support, receipt)?;
    let search_baseline = work.objective;
    let mut steps = 0;
    let mut stop = "steps";
    for _ in 0..req.steps {
        let moved = work
            .projected_direction(req)
            .and_then(|()| work.ray(req))
            .and_then(|moved| if moved { Ok(true) } else { work.rescue(req) })
            .and_then(|moved| {
                if moved {
                    Ok(true)
                } else {
                    work.pair_rescue(req)
                }
            });
        match moved {
            Ok(true) => steps += 1,
            Ok(false) => {
                stop = "stationary";
                break;
            }
            Err(Halt::Deadline) => {
                stop = "deadline";
                break;
            }
            Err(Halt::Nonfinite) => {
                stop = "nonfinite";
                break;
            }
        }
    }
    Ok(DualSearchOutcome {
        multipliers: Untrusted::new(DualMultipliers {
            l0: work.l0,
            l1: work.l1,
        }),
        steps,
        gain: Untrusted::new(work.objective - search_baseline),
        stop,
        support: work.support_stats,
    })
}

#[cfg(test)]
mod tests;
