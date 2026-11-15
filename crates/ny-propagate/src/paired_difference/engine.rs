// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Certified signed-strip transport for pairs of nearby ReLU networks.
//!
//! A state encloses each tower and their SIGNED preactivation difference.
//! Backward substitution uses a plane certified over the strip-cut paired
//! activation polygon, rather than treating the towers as independent. The
//! geometry kernel repairs arbitrary proposed slopes by checking every vertex.
//! All affine substitution errors are charged against certified source boxes.
//! Input splitting closes BOTH children and never discards queued domains.
//! The result is proof-only: failure, expiry and resource caps return no proof.
//!
//! The std-only engine is also compiled verbatim by tools/paired-difference;
//! the GraphNetwork admission adapter is the sole feature-specific module.

#[path = "forward.rs"]
mod forward;
#[path = "forward_affine_hull.rs"]
mod forward_affine_hull;
#[path = "forward_coupled_hull.rs"]
mod forward_coupled_hull;
#[path = "forward_meet.rs"]
mod forward_meet;
#[path = "lp_affine_hull.rs"]
mod lp_affine_hull;
#[path = "lp_certificate.rs"]
mod lp_certificate;
#[path = "lp_coupled_affine.rs"]
mod lp_coupled_affine;
#[path = "lp_cover.rs"]
mod lp_cover;
#[path = "lp_group.rs"]
mod lp_group;
#[path = "lp_model.rs"]
mod lp_model;
#[path = "lp_native.rs"]
mod lp_native;
#[path = "lp_optimize.rs"]
mod lp_optimize;
#[path = "lp_phase.rs"]
mod lp_phase;
#[path = "lp_template.rs"]
mod lp_template;
#[path = "refine.rs"]
mod refine;
pub use lp_model::{
    LpProofReport, OutputTemplateCover, PreparedRootRelaxation, RootRelaxationReport,
    SoftGuardCover, SoftGuardLeaf,
};
pub use lp_native::NativeLpReport;
pub use lp_phase::{
    PairedPhase, PhaseChildCertificate, PhaseChildStatus, PhaseCoverReport,
    PhaseInfeasibilityReport, PhaseLpReport, PhaseProofNode, PhaseProofTree, PhaseTreeReport,
};
#[path = "search.rs"]
mod search;
pub use search::ParallelWorkers;

use crate::paired_relu_hull::{PairedReluHull, PairedReluRanges};
use std::time::{Duration, Instant};

/// One stage of a paired chain. `Affine` (112 bytes) is the large variant
/// and also the common one: a chain alternates affine and ReLU stages and is
/// built once per instance, so boxing the affine payload would add an
/// allocation and an indirection on every affine stage of the propagation hot
/// loop to save ~100 bytes per ReLU stage of a handful-element `Vec`.
#[derive(Clone, Debug)]
#[expect(
    clippy::large_enum_variant,
    reason = "few stages per chain, Affine is the common variant; boxing costs an indirection per affine stage"
)]
pub enum Stage {
    Affine {
        rows: usize,
        cols: usize,
        wf: Vec<f64>,
        wg: Vec<f64>,
        bf: Vec<f64>,
        bg: Vec<f64>,
    },
    Relu {
        width: usize,
    },
}

#[derive(Clone, Debug)]
pub struct PairedChain {
    pub input_dim: usize,
    pub output_dim: usize,
    pub stages: Vec<Stage>,
}

#[derive(Clone, Debug)]
pub struct BoxDomain {
    pub lo: Vec<f64>,
    pub hi: Vec<f64>,
}

#[derive(Clone, Debug)]
struct Enclosure {
    f_lo: Vec<f64>,
    f_hi: Vec<f64>,
    g_lo: Vec<f64>,
    g_hi: Vec<f64>,
    d_lo: Vec<f64>,
    d_hi: Vec<f64>,
    hulls: Option<Vec<PairedReluHull>>,
}

#[derive(Clone, Debug)]
pub struct BoundReport {
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
    pub split_scores: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct SearchReport {
    pub proved: bool,
    pub evaluated: u64,
    pub closed: u64,
    pub open: usize,
    pub max_depth: usize,
    pub elapsed: Duration,
    pub reason: &'static str,
    /// Maximum certified absolute endpoint among ALL closed leaves.
    pub worst_closed_bound: f64,
}

fn up(x: f64) -> f64 {
    if x == 0.0 {
        0.0
    } else {
        x.next_up()
    }
}
// Always step for a rounded addition/product: an exact zero may hide
// cancellation or underflow. Only structural zero operations skip rounding.
fn add_down(a: f64, b: f64) -> f64 {
    if b == 0.0 {
        a
    } else {
        (a + b).next_down()
    }
}
fn add_up(a: f64, b: f64) -> f64 {
    if b == 0.0 {
        a
    } else {
        (a + b).next_up()
    }
}
fn mul_down(a: f64, b: f64) -> f64 {
    if a == 0.0 || b == 0.0 {
        0.0
    } else {
        (a * b).next_down()
    }
}
fn mul_up(a: f64, b: f64) -> f64 {
    if a == 0.0 || b == 0.0 {
        0.0
    } else {
        (a * b).next_up()
    }
}
fn magnitude(l: f64, h: f64) -> f64 {
    l.abs().max(h.abs())
}

fn dot_error(n: usize, absolute_sum: f64) -> f64 {
    let ne = (2 * n + 2) as f64 * f64::EPSILON;
    up((ne / (1.0 - ne)) * absolute_sum * (1.0 + 1e-12) + (4 * n + 8) as f64 * f64::from_bits(1))
}

impl PairedChain {
    /// Validate the whole transport program before any bound can be returned.
    pub fn validate(&self) -> Result<(), String> {
        if self.input_dim == 0
            || self.input_dim > 16
            || self.stages.is_empty()
            || self.stages.len() > 128
        {
            return Err("paired program exceeds structural caps".into());
        }
        let mut width = self.input_dim;
        let mut total = 0usize;
        for stage in &self.stages {
            match stage {
                Stage::Affine {
                    rows,
                    cols,
                    wf,
                    wg,
                    bf,
                    bg,
                } => {
                    if *cols != width
                        || *rows == 0
                        || *rows > 512
                        || wf.len() != rows * cols
                        || wg.len() != wf.len()
                        || bf.len() != *rows
                        || bg.len() != *rows
                    {
                        return Err("paired affine shape mismatch".into());
                    }
                    if wf
                        .iter()
                        .chain(wg)
                        .chain(bf)
                        .chain(bg)
                        .any(|v| !v.is_finite() || v.abs() > 1e100)
                    {
                        return Err("nonfinite/oversized paired coefficient".into());
                    }
                    width = *rows;
                    total += rows;
                }
                Stage::Relu { width: n } if *n == width => {
                    total += n;
                }
                Stage::Relu { .. } => return Err("paired ReLU shape mismatch".into()),
            }
        }
        if width != self.output_dim || self.output_dim == 0 || self.output_dim > 64 || total > 10000
        {
            return Err("paired output/size mismatch".into());
        }
        Ok(())
    }

    fn input_state(&self, domain: &BoxDomain) -> Option<Enclosure> {
        if domain.lo.len() != self.input_dim
            || domain.hi.len() != self.input_dim
            || domain.lo.iter().zip(&domain.hi).any(|(&l, &h)| {
                !l.is_finite() || !h.is_finite() || l > h || magnitude(l, h) > 1e100
            })
        {
            return None;
        }
        Some(Enclosure {
            f_lo: domain.lo.clone(),
            f_hi: domain.hi.clone(),
            g_lo: domain.lo.clone(),
            g_hi: domain.hi.clone(),
            d_lo: vec![0.0; self.input_dim],
            d_hi: vec![0.0; self.input_dim],
            hulls: None,
        })
    }

    /// Concrete real-valued evaluation of exact stored coefficients (diagnostic).
    pub fn forward(&self, input: &[f64]) -> Option<(Vec<f64>, Vec<f64>)> {
        self.validate().ok()?;
        if input.len() != self.input_dim || input.iter().any(|x| !x.is_finite()) {
            return None;
        }
        let (mut f, mut g) = (input.to_vec(), input.to_vec());
        for stage in &self.stages {
            match stage {
                Stage::Affine {
                    rows,
                    cols,
                    wf,
                    wg,
                    bf,
                    bg,
                } => {
                    let mut nf = bf.clone();
                    let mut ng = bg.clone();
                    for i in 0..*rows {
                        for j in 0..*cols {
                            nf[i] += wf[i * cols + j] * f[j];
                            ng[i] += wg[i * cols + j] * g[j];
                        }
                    }
                    f = nf;
                    g = ng;
                }
                Stage::Relu { .. } => {
                    for x in f.iter_mut().chain(&mut g) {
                        *x = x.max(0.0);
                    }
                }
            }
        }
        Some((f, g))
    }

    fn back(
        &self,
        before: usize,
        cf: Vec<f64>,
        cg: Vec<f64>,
        k: f64,
        states: &[Enclosure],
        domain: &BoxDomain,
        paired: bool,
        deadline: Option<Instant>,
    ) -> Option<(f64, Vec<f64>)> {
        self.back_with_root_cost(before, cf, cg, k, states, domain, paired, deadline, None)
    }

    /// Certifies a layer objective minus an arbitrary exact-f64 root linear
    /// form. The root coefficients are proposals, never claimed bounds.
    fn back_with_root_cost(
        &self,
        before: usize,
        mut cf: Vec<f64>,
        mut cg: Vec<f64>,
        mut k: f64,
        states: &[Enclosure],
        domain: &BoxDomain,
        paired: bool,
        deadline: Option<Instant>,
        root_cost: Option<&[f64]>,
    ) -> Option<(f64, Vec<f64>)> {
        if root_cost.is_some_and(|a| a.len() != self.input_dim || a.iter().any(|x| !x.is_finite()))
        {
            return None;
        }
        for s in (0..before).rev() {
            if deadline.is_some_and(|d| Instant::now() >= d) {
                return None;
            }
            let source = &states[s];
            match &self.stages[s] {
                Stage::Affine {
                    rows,
                    cols,
                    wf,
                    wg,
                    bf,
                    bg,
                } => {
                    if cf.len() != *rows || cg.len() != *rows {
                        return None;
                    }
                    for i in 0..*rows {
                        k = add_down(k, mul_down(cf[i], bf[i]));
                        k = add_down(k, mul_down(cg[i], bg[i]));
                    }
                    let mut nf = vec![0.0; *cols];
                    let mut ng = vec![0.0; *cols];
                    let mut charge = 0.0;
                    for j in 0..*cols {
                        let (mut sf, mut sg) = (0.0, 0.0);
                        for i in 0..*rows {
                            let pf = cf[i] * wf[i * cols + j];
                            let pg = cg[i] * wg[i * cols + j];
                            nf[j] += pf;
                            ng[j] += pg;
                            sf = add_up(sf, mul_up(cf[i].abs(), wf[i * cols + j].abs()));
                            sg = add_up(sg, mul_up(cg[i].abs(), wg[i * cols + j].abs()));
                        }
                        charge = add_up(
                            charge,
                            mul_up(
                                dot_error(*rows, sf),
                                magnitude(source.f_lo[j], source.f_hi[j]),
                            ),
                        );
                        charge = add_up(
                            charge,
                            mul_up(
                                dot_error(*rows, sg),
                                magnitude(source.g_lo[j], source.g_hi[j]),
                            ),
                        );
                    }
                    k = add_down(k, -charge);
                    cf = nf;
                    cg = ng;
                }
                Stage::Relu { width } => {
                    for j in 0..*width {
                        let r = PairedReluRanges {
                            lf: source.f_lo[j],
                            uf: source.f_hi[j],
                            lg: source.g_lo[j],
                            ug: source.g_hi[j],
                            dl: source.d_lo[j],
                            du: source.d_hi[j],
                        };
                        if (r.lf >= 0.0 || r.uf <= 0.0) && (r.lg >= 0.0 || r.ug <= 0.0) {
                            if r.uf <= 0.0 {
                                cf[j] = 0.0;
                            }
                            if r.ug <= 0.0 {
                                cg[j] = 0.0;
                            }
                            continue;
                        }
                        let sf = relu_slope(cf[j], r.lf, r.uf);
                        let sg = relu_slope(cg[j], r.lg, r.ug);
                        let mut candidates = vec![(cf[j] * sf, cg[j] * sg)];
                        if paired && cf[j] * cg[j] < 0.0 {
                            let c = cf[j].signum() * cf[j].abs().min(cg[j].abs());
                            let rf = cf[j] - c;
                            let rg = cg[j] + c;
                            let sd = if r.dl >= 0.0 {
                                if c > 0.0 {
                                    0.0
                                } else {
                                    1.0
                                }
                            } else if r.du <= 0.0 {
                                if c > 0.0 {
                                    1.0
                                } else {
                                    0.0
                                }
                            } else if c > 0.0 {
                                -r.dl / (r.du - r.dl)
                            } else {
                                r.du / (r.du - r.dl)
                            };
                            for a in [sd, 0.5, 0.0, 1.0] {
                                candidates.push((
                                    c * a + rf * relu_slope(rf, r.lf, r.uf),
                                    -c * a + rg * relu_slope(rg, r.lg, r.ug),
                                ));
                            }
                        }
                        let mut best = None;
                        let mf = r.lf * 0.5 + r.uf * 0.5;
                        let mg = r.lg * 0.5 + r.ug * 0.5;
                        for (af, ag) in candidates {
                            let intercept = if paired && cf[j] * cg[j] < 0.0 {
                                source.hulls.as_ref()?[j]
                                    .certified_intercept(cf[j], cg[j], af, ag)?
                            } else {
                                independent_intercept(cf[j], af, r.lf, r.uf, cg[j], ag, r.lg, r.ug)
                            };
                            let score = af * mf + ag * mg + intercept;
                            if score.is_finite()
                                && best.as_ref().is_none_or(|&(_, _, _, old)| score > old)
                            {
                                best = Some((af, ag, intercept, score));
                            }
                        }
                        let (af, ag, intercept, _) = best?;
                        cf[j] = af;
                        cg[j] = ag;
                        k = add_down(k, intercept);
                    }
                }
            }
            if !k.is_finite()
                || cf
                    .iter()
                    .chain(&cg)
                    .any(|x| !x.is_finite() || x.abs() > 1e150)
            {
                return None;
            }
        }
        let mut combined = vec![0.0; self.input_dim];
        for j in 0..self.input_dim {
            combined[j] = cf[j] + cg[j];
            let err = dot_error(1, add_up(cf[j].abs(), cg[j].abs()));
            k = add_down(k, -mul_up(err, magnitude(domain.lo[j], domain.hi[j])));
            if let Some(a) = root_cost {
                if a[j] != 0.0 {
                    let err = dot_error(1, add_up(combined[j].abs(), a[j].abs()));
                    combined[j] -= a[j];
                    k = add_down(k, -mul_up(err, magnitude(domain.lo[j], domain.hi[j])));
                }
            }
            k = add_down(
                k,
                mul_down(
                    combined[j],
                    if combined[j] >= 0.0 {
                        domain.lo[j]
                    } else {
                        domain.hi[j]
                    },
                ),
            );
        }
        k.is_finite().then_some((k, combined))
    }

    /// Rigorous enclosures for both signs of every difference output.
    pub fn bound(&self, domain: &BoxDomain, paired: bool) -> Option<BoundReport> {
        self.bound_until(domain, paired, None, None)
    }

    /// Propagate source-certified multivariate affine/ReLU hulls before the
    /// native backward bound. Original source bounds remain in the meet.
    pub fn bound_strong_until(&self, domain: &BoxDomain, deadline: Instant) -> Option<BoundReport> {
        self.validate().ok()?;
        if !lp_certificate::suitable_fp_environment() || Instant::now() >= deadline {
            return None;
        }
        let original = forward::states(self, domain, Some(deadline), true)?;
        let reference = forward::states_strong_with_reference(
            self,
            domain,
            Some(deadline),
            true,
            Some(&original),
        )?;
        let report =
            self.bound_with_reference_until(domain, true, Some(deadline), None, Some(&reference))?;
        (Instant::now() < deadline).then_some(report)
    }

    fn bound_until(
        &self,
        domain: &BoxDomain,
        paired: bool,
        deadline: Option<Instant>,
        band_epsilon: Option<f64>,
    ) -> Option<BoundReport> {
        self.bound_with_reference_until(domain, paired, deadline, band_epsilon, None)
    }

    fn bound_with_reference_until(
        &self,
        domain: &BoxDomain,
        paired: bool,
        deadline: Option<Instant>,
        band_epsilon: Option<f64>,
        reference: Option<&[Enclosure]>,
    ) -> Option<BoundReport> {
        if !lp_certificate::suitable_fp_environment() {
            return None;
        }
        self.validate().ok()?;
        if band_epsilon.is_some_and(|epsilon| !epsilon.is_finite() || epsilon <= 0.0) {
            return None;
        }
        let states = forward::states_with_reference(self, domain, deadline, paired, reference)?;
        let output = states.last()?;
        let mut lower = output.d_lo.clone();
        let mut upper = output.d_hi.clone();
        if lower.len() != self.output_dim
            || upper.len() != self.output_dim
            || lower
                .iter()
                .zip(&upper)
                .any(|(&lo, &hi)| !lo.is_finite() || !hi.is_finite() || lo > hi)
        {
            return None;
        }
        // These forward intervals already enclose every reachable difference.
        // A search threshold can save refinement work; it is not proof authority.
        // The public full-bound API passes None and still refines every output.
        let forward_in_band =
            |lo: f64, hi: f64| band_epsilon.is_some_and(|epsilon| lo > -epsilon && hi < epsilon);
        if lower
            .iter()
            .zip(&upper)
            .all(|(&lo, &hi)| forward_in_band(lo, hi))
        {
            if deadline.is_some_and(|d| Instant::now() >= d) {
                return None;
            }
            return Some(BoundReport {
                lower,
                upper,
                // Every objective is certified, so this box will close and
                // the unused branching scores need no backward sensitivities.
                split_scores: domain
                    .lo
                    .iter()
                    .zip(&domain.hi)
                    .map(|(&lo, &hi)| hi - lo)
                    .collect(),
            });
        }
        let mut sensitivity = vec![0.0_f64; self.input_dim];
        for i in 0..self.output_dim {
            if deadline.is_some_and(|d| Instant::now() >= d) {
                return None;
            }
            if forward_in_band(lower[i], upper[i]) {
                // Keep the rigorous forward enclosure; compute both backward
                // signs only for an output whose forward band remains open.
                // Dropping a closed output's sensitivity can change splitting,
                // but never the certified enclosure or complete cover argument.
                continue;
            }
            let mut cf = vec![0.0; self.output_dim];
            let mut cg = cf.clone();
            cf[i] = 1.0;
            cg[i] = -1.0;
            let (lb, lg) = self.back(
                self.stages.len(),
                cf.clone(),
                cg.clone(),
                0.0,
                &states,
                domain,
                paired,
                deadline,
            )?;
            lower[i] = lower[i].max(lb);
            cf[i] = -1.0;
            cg[i] = 1.0;
            let (negative_ub, ug) = self.back(
                self.stages.len(),
                cf,
                cg,
                0.0,
                &states,
                domain,
                paired,
                deadline,
            )?;
            let ub = -negative_ub;
            for j in 0..self.input_dim {
                sensitivity[j] = sensitivity[j].max(lg[j].abs()).max(ug[j].abs());
            }
            upper[i] = upper[i].min(ub);
            if lower[i] > upper[i] || !lower[i].is_finite() || !upper[i].is_finite() {
                return None;
            }
        }
        // These coefficients select a search axis only. They carry no proof
        // authority, so arbitrary finite ranking changes remain sound.
        let any_sensitivity = sensitivity.iter().any(|&x| x > 0.0);
        let split_scores = domain
            .lo
            .iter()
            .zip(&domain.hi)
            .enumerate()
            .map(|(j, (&l, &h))| (h - l) * if any_sensitivity { sensitivity[j] } else { 1.0 })
            .collect();
        Some(BoundReport {
            lower,
            upper,
            split_scores,
        })
    }

    /// Prove |f-g| < epsilon everywhere by a complete, bounded input tree.
    pub fn prove(
        &self,
        root: &BoxDomain,
        epsilon: f64,
        deadline: Instant,
        max_domains: u64,
        paired: bool,
    ) -> SearchReport {
        let start = Instant::now();
        let mut report = SearchReport {
            proved: false,
            evaluated: 0,
            closed: 0,
            open: 1,
            max_depth: 0,
            elapsed: Duration::ZERO,
            reason: "invalid",
            worst_closed_bound: 0.0,
        };
        if self.validate().is_err()
            || self.input_state(root).is_none()
            || !epsilon.is_finite()
            || epsilon <= 0.0
            || max_domains == 0
        {
            return report;
        }
        let mut stack = vec![(root.clone(), 0usize)];
        report.reason = "domain_cap";
        while let Some((domain, depth)) = stack.pop() {
            if Instant::now() >= deadline {
                stack.push((domain, depth));
                report.reason = "deadline";
                break;
            }
            if report.evaluated >= max_domains {
                stack.push((domain, depth));
                break;
            }
            let Some(bounds) = self.bound_until(&domain, paired, Some(deadline), Some(epsilon))
            else {
                stack.push((domain, depth));
                report.reason = if Instant::now() >= deadline {
                    "deadline"
                } else {
                    "bound_refused"
                };
                break;
            };
            if Instant::now() >= deadline {
                stack.push((domain, depth));
                report.reason = "deadline";
                break;
            }
            report.evaluated += 1;
            report.max_depth = report.max_depth.max(depth);
            let worst = bounds
                .lower
                .iter()
                .chain(&bounds.upper)
                .map(|v| v.abs())
                .fold(0.0, f64::max);
            if worst < epsilon {
                report.closed += 1;
                report.worst_closed_bound = report.worst_closed_bound.max(worst);
                continue;
            }
            if depth >= 96 {
                stack.push((domain, depth));
                report.reason = "depth_cap";
                break;
            }
            let axis = match search::split_axis(&domain, root, &bounds, depth) {
                Ok(axis) => axis,
                Err(reason) => {
                    stack.push((domain, depth));
                    report.reason = reason;
                    break;
                }
            };
            let mid = domain.lo[axis] * 0.5 + domain.hi[axis] * 0.5;
            if !(domain.lo[axis] < mid && mid < domain.hi[axis]) {
                stack.push((domain, depth));
                report.reason = "unsplittable";
                break;
            }
            let mut left = domain.clone();
            let mut right = domain;
            left.hi[axis] = mid;
            right.lo[axis] = mid;
            // Overlap at mid. Their union is EXACTLY the parent's closed box.
            stack.push((right, depth + 1));
            stack.push((left, depth + 1));
        }
        if stack.is_empty() && Instant::now() >= deadline {
            stack.push((root.clone(), 0));
            report.reason = "deadline";
        }
        report.open = stack.len();
        report.proved = stack.is_empty();
        if report.proved {
            report.reason = "covered";
        }
        report.elapsed = start.elapsed();
        report
    }
}

fn relu_slope(c: f64, l: f64, u: f64) -> f64 {
    if l >= 0.0 {
        1.0
    } else if u <= 0.0 {
        0.0
    } else if c < 0.0 {
        u / (u - l)
    } else if u >= -l {
        1.0
    } else {
        0.0
    }
}

fn independent_intercept(
    cf: f64,
    af: f64,
    lf: f64,
    uf: f64,
    cg: f64,
    ag: f64,
    lg: f64,
    ug: f64,
) -> f64 {
    fn one(c: f64, a: f64, l: f64, u: f64) -> f64 {
        if c == 0.0 && a == 0.0 || l >= 0.0 && c == a || u <= 0.0 && a == 0.0 {
            return 0.0;
        }
        let mut k = f64::INFINITY;
        for z in [l, u, 0.0] {
            if z >= l && z <= u {
                k = k.min(add_down(mul_down(c, z.max(0.0)), -mul_up(a, z)));
            }
        }
        k
    }
    add_down(one(cf, af, lf, uf), one(cg, ag, lg, ug))
}

#[cfg(test)]
#[path = "engine_tests.rs"]
mod tests;
