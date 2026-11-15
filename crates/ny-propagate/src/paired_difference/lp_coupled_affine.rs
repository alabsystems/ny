// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unconditional paired affine-input ReLU supports from four closed phases.
//!
//! Let zF=WF*u+bF, zG=WG*v+bG with each (u_j,v_j) in its source
//! rectangle intersected with a certified SIGNED difference strip. For phase
//! sF,sG in {0,1}, write eta=2*s-1 and choose arbitrary lambdaF,lambdaG>=0.
//! On that phase lambdaF*etaF*zF+lambdaG*etaG*zG>=0. Consequently
//!
//! ```text
//! cf*relu(zF)+cg*relu(zG)-aF*u-aG*v
//!   >= pF*bF+pG*bG + sum_j min_polygon_j
//!        ((pF*WF_j-aF_j)*u_j+(pG*WG_j-aG_j)*v_j),
//! ```
//!
//! where pF=cf*sF-lambdaF*etaF and likewise for G. The MINIMUM of
//! ALL FOUR separately replayed phase bounds is an unconditional root cut.
//! No phase, including either mixed phase or a zero boundary, is discarded.
//! Source inputs are relaxed jointly by neuron but independently across
//! neurons; ignoring additional correlations enlarges the domain safely.
//!
//! Numerical normals, shared Lovasz fractions, polygon hints and a bounded
//! two-variable coordinate search choose proposals only. Exact authored
//! affine-run composition intervals enclose the true source coefficients.
//! Residual coefficient intervals are charged against source boxes; their
//! nominal paired polygon minima are certified by the existing geometric
//! replay. A failed/expired phase refuses the entire unconditional support.
//! This combines established disjunctive and weak-dual constructions and
//! makes no novelty, complete-solver or competition-score claim.

use super::lp_affine_hull::{compose, separating_slopes, ComposedAffine};
use super::lp_certificate::{suitable_fp_environment, LinearRow};
use super::*;

const MAX_WIDTH: usize = 512;
const MAX_NEURONS: usize = 8;
const MAX_INTERMEDIATE_NEURONS: usize = 4;
// At most 8 neurons * 4 normals * 3 source-plane proposals. The source LP
// builder separately enforces its global authenticated row/term caps.
const MAX_ROWS: usize = 96;
const MAX_MAGNITUDE: f64 = 1e100;
const PHASES: [(bool, bool); 4] = [(false, false), (true, false), (false, true), (true, true)];

fn alive(deadline: Option<Instant>) -> bool {
    deadline.is_none_or(|d| Instant::now() < d)
}
fn admissible(x: f64) -> bool {
    x.is_finite() && x.abs() <= MAX_MAGNITUDE
}

#[derive(Clone, Copy)]
struct Interval {
    lo: f64,
    hi: f64,
}
impl Interval {
    fn point(x: f64) -> Self {
        Self { lo: x, hi: x }
    }
    fn valid(self) -> bool {
        self.lo.is_finite() && self.hi.is_finite() && self.lo <= self.hi
    }
    fn add(self, b: Self) -> Option<Self> {
        let lo = if self.lo == -b.lo {
            0.0
        } else {
            add_down(self.lo, b.lo)
        };
        let hi = if self.hi == -b.hi {
            0.0
        } else {
            add_up(self.hi, b.hi)
        };
        let r = Self { lo, hi };
        r.valid().then_some(r)
    }
    fn neg(self) -> Self {
        Self {
            lo: -self.hi,
            hi: -self.lo,
        }
    }
    fn sub(self, b: Self) -> Option<Self> {
        self.add(b.neg())
    }
    fn mul(self, b: Self) -> Option<Self> {
        if (self.lo == 0.0 && self.hi == 0.0) || (b.lo == 0.0 && b.hi == 0.0) {
            return Some(Self::point(0.0));
        }
        if self.lo == 1.0 && self.hi == 1.0 {
            return Some(b);
        }
        if b.lo == 1.0 && b.hi == 1.0 {
            return Some(self);
        }
        if self.lo == -1.0 && self.hi == -1.0 {
            return Some(b.neg());
        }
        if b.lo == -1.0 && b.hi == -1.0 {
            return Some(self.neg());
        }
        let pairs = [
            (self.lo, b.lo),
            (self.lo, b.hi),
            (self.hi, b.lo),
            (self.hi, b.hi),
        ];
        let r = Self {
            lo: pairs
                .iter()
                .map(|&(a, b)| mul_down(a, b))
                .fold(f64::INFINITY, f64::min),
            hi: pairs
                .iter()
                .map(|&(a, b)| mul_up(a, b))
                .fold(f64::NEG_INFINITY, f64::max),
        };
        r.valid().then_some(r)
    }
    fn error(self, nominal: f64) -> Option<f64> {
        let e = self.sub(Self::point(nominal))?;
        let r = e.lo.abs().max(e.hi.abs());
        r.is_finite().then_some(r)
    }
}

struct Input {
    ranges: PairedReluRanges,
    hull: PairedReluHull,
    // Nearest-rounded intersections are strictly proposal-only. Even an
    // imperfect hint set cannot bypass the independent geometric replay.
    hints: Vec<(f64, f64)>,
}

fn inputs(ranges: &[PairedReluRanges], deadline: Option<Instant>) -> Option<Vec<Input>> {
    if ranges.is_empty() || ranges.len() > MAX_WIDTH {
        return None;
    }
    let mut result = Vec::with_capacity(ranges.len());
    for &r in ranges {
        if !alive(deadline)
            || [r.lf, r.uf, r.lg, r.ug, r.dl, r.du]
                .iter()
                .any(|&v| !admissible(v))
        {
            return None;
        }
        let hull = r.prepare()?;
        let mut hints = Vec::new();
        let mut add = |x: f64, y: f64| {
            if x.is_finite()
                && y.is_finite()
                && x >= r.lf
                && x <= r.uf
                && y >= r.lg
                && y <= r.ug
                && x - y >= r.dl
                && x - y <= r.du
                && !hints.contains(&(x, y))
            {
                hints.push((x, y));
            }
        };
        for x in [r.lf, r.uf] {
            for y in [r.lg, r.ug] {
                add(x, y);
            }
            for d in [r.dl, r.du] {
                add(x, x - d);
            }
        }
        for y in [r.lg, r.ug] {
            for d in [r.dl, r.du] {
                add(y + d, y);
            }
        }
        if hints.is_empty() {
            hints.push((0.5 * r.lf + 0.5 * r.uf, 0.5 * r.lg + 0.5 * r.ug));
        }
        result.push(Input {
            ranges: r,
            hull,
            hints,
        });
    }
    Some(result)
}

struct AffinePair<'a> {
    wf: &'a [f64],
    wg: &'a [f64],
    bf: f64,
    bg: f64,
    wfl: &'a [f64],
    wfu: &'a [f64],
    wgl: &'a [f64],
    wgu: &'a [f64],
    bfi: Interval,
    bgi: Interval,
}
impl AffinePair<'_> {
    fn valid(&self, n: usize) -> bool {
        n > 0
            && n <= MAX_WIDTH
            && [
                self.wf.len(),
                self.wg.len(),
                self.wfl.len(),
                self.wfu.len(),
                self.wgl.len(),
                self.wgu.len(),
            ]
            .iter()
            .all(|&m| m == n)
            && self.bfi.valid()
            && self.bgi.valid()
            && [
                self.bf,
                self.bg,
                self.bfi.lo,
                self.bfi.hi,
                self.bgi.lo,
                self.bgi.hi,
            ]
            .iter()
            .all(|&v| admissible(v))
            && self
                .wf
                .iter()
                .chain(self.wg)
                .chain(self.wfl)
                .chain(self.wfu)
                .chain(self.wgl)
                .chain(self.wgu)
                .all(|&v| admissible(v))
            && self.wfl.iter().zip(self.wfu).all(|(&l, &u)| l <= u)
            && self.wgl.iter().zip(self.wgu).all(|(&l, &u)| l <= u)
    }
}

fn phase_intercept(
    source: &AffinePair<'_>,
    input: &[Input],
    cf: f64,
    cg: f64,
    af: &[f64],
    ag: &[f64],
    phase: (bool, bool),
    lambda: [f64; 2],
    deadline: Option<Instant>,
) -> Option<f64> {
    let eta = |s: bool| if s { 1.0 } else { -1.0 };
    let sf = if phase.0 { cf } else { 0.0 };
    let sg = if phase.1 { cg } else { 0.0 };
    let pf = Interval::point(sf).sub(Interval::point(lambda[0] * eta(phase.0)))?;
    let pg = Interval::point(sg).sub(Interval::point(lambda[1] * eta(phase.1)))?;
    let pfn = sf - lambda[0] * eta(phase.0);
    let pgn = sg - lambda[1] * eta(phase.1);
    let mut lower = pf.mul(source.bfi)?.add(pg.mul(source.bgi)?)?.lo;
    for j in 0..input.len() {
        if !alive(deadline) {
            return None;
        }
        let rf = pf
            .mul(Interval {
                lo: source.wfl[j],
                hi: source.wfu[j],
            })?
            .sub(Interval::point(af[j]))?;
        let rg = pg
            .mul(Interval {
                lo: source.wgl[j],
                hi: source.wgu[j],
            })?
            .sub(Interval::point(ag[j]))?;
        let rfn = pfn * source.wf[j] - af[j];
        let rgn = pgn * source.wg[j] - ag[j];
        if !rfn.is_finite() || !rgn.is_finite() {
            return None;
        }
        let r = input[j].ranges;
        let charge = add_up(
            mul_up(rf.error(rfn)?, magnitude(r.lf, r.uf)),
            mul_up(rg.error(rgn)?, magnitude(r.lg, r.ug)),
        );
        // cf=cg=0 makes this the exact-f64 LINEAR polygon support oracle;
        // the extra axis subdivisions preserve the same polygon minimum.
        let nominal = input[j].hull.certified_intercept(0.0, 0.0, -rfn, -rgn)?;
        let repaired = if charge == 0.0 {
            nominal
        } else {
            add_down(nominal, -charge)
        };
        lower = add_down(lower, repaired);
    }
    (admissible(lower) && alive(deadline)).then_some(lower)
}

/// Certify an arbitrary exact-f64 support of an AUTHORED single affine pair.
/// This arithmetic oracle authenticates no network or supplied source bounds.
/// Production rows below use only the source builder's private bound/binding
/// maps and the interval-enclosed authored affine-run composition.
#[cfg(test)]
pub(super) fn certified_intercept(
    wf: &[f64],
    bf: f64,
    wg: &[f64],
    bg: f64,
    ranges: &[PairedReluRanges],
    cf: f64,
    cg: f64,
    af: &[f64],
    ag: &[f64],
    lambdas: [[f64; 2]; 4],
    deadline: Option<Instant>,
) -> Option<f64> {
    let source = AffinePair {
        wf,
        wg,
        bf,
        bg,
        wfl: wf,
        wfu: wf,
        wgl: wg,
        wgu: wg,
        bfi: Interval::point(bf),
        bgi: Interval::point(bg),
    };
    if !source.valid(ranges.len())
        || af.len() != ranges.len()
        || ag.len() != ranges.len()
        || af.iter().chain(ag).any(|&v| !admissible(v))
        || !admissible(cf)
        || !admissible(cg)
        || lambdas.iter().flatten().any(|&v| !admissible(v) || v < 0.0)
        || !alive(deadline)
        || !suitable_fp_environment()
    {
        return None;
    }
    let prepared = inputs(ranges, deadline)?;
    let mut result = f64::INFINITY;
    for (phase, lambda) in PHASES.into_iter().zip(lambdas) {
        result = result.min(phase_intercept(
            &source, &prepared, cf, cg, af, ag, phase, lambda, deadline,
        )?);
    }
    (admissible(result) && alive(deadline)).then_some(result)
}

// The concave numerical weak dual and a selected subgradient. This is never
// returned as an intercept. Native coordinate ascent chooses two multipliers.
fn hint_value(
    source: &AffinePair<'_>,
    input: &[Input],
    cf: f64,
    cg: f64,
    af: &[f64],
    ag: &[f64],
    phase: (bool, bool),
    lambda: [f64; 2],
) -> (f64, [f64; 2]) {
    let ef = if phase.0 { 1.0 } else { -1.0 };
    let eg = if phase.1 { 1.0 } else { -1.0 };
    let pf = (if phase.0 { cf } else { 0.0 }) - lambda[0] * ef;
    let pg = (if phase.1 { cg } else { 0.0 }) - lambda[1] * eg;
    let mut value = pf * source.bf + pg * source.bg;
    let mut gradient = [-ef * source.bf, -eg * source.bg];
    for (j, p) in input.iter().enumerate() {
        let rf = pf * source.wf[j] - af[j];
        let rg = pg * source.wg[j] - ag[j];
        let mut best = (f64::INFINITY, 0.0, 0.0);
        for &(x, y) in &p.hints {
            let v = rf * x + rg * y;
            if v < best.0 {
                best = (v, x, y);
            }
        }
        value += best.0;
        gradient[0] -= ef * source.wf[j] * best.1;
        gradient[1] -= eg * source.wg[j] * best.2;
    }
    (value, gradient)
}

fn multipliers(
    source: &AffinePair<'_>,
    input: &[Input],
    cf: f64,
    cg: f64,
    af: &[f64],
    ag: &[f64],
    phase: (bool, bool),
    deadline: Option<Instant>,
) -> Option<Vec<[f64; 2]>> {
    let unit = cf.abs().max(cg.abs()).max(1.0);
    let cap = 64.0 * unit;
    let mut candidates = Vec::new();
    let mut best = ([0.0, 0.0], f64::NEG_INFINITY);
    for f in [0.0, cf.abs(), cg.abs(), unit] {
        for g in [0.0, cf.abs(), cg.abs(), unit] {
            if !alive(deadline) {
                return None;
            }
            let lambda = [f, g];
            let value = hint_value(source, input, cf, cg, af, ag, phase, lambda).0;
            if value.is_finite() && value > best.1 {
                best = (lambda, value);
            }
        }
    }
    candidates.push(best.0);
    let mut current = best.0;
    for _ in 0..2 {
        for axis in 0..2 {
            if !alive(deadline) {
                return None;
            }
            let mut probe = current;
            probe[axis] = 0.0;
            let (_, grad) = hint_value(source, input, cf, cg, af, ag, phase, probe);
            if !grad[axis].is_finite() {
                return None;
            }
            if grad[axis] <= 0.0 {
                current = probe;
                continue;
            }
            let mut low = 0.0;
            let mut high = unit;
            loop {
                if !alive(deadline) {
                    return None;
                }
                probe[axis] = high;
                let grad = hint_value(source, input, cf, cg, af, ag, phase, probe).1[axis];
                if !grad.is_finite() {
                    return None;
                }
                if grad <= 0.0 || high >= cap {
                    break;
                }
                low = high;
                high = (high * 2.0).min(cap);
            }
            for _ in 0..28 {
                if !alive(deadline) {
                    return None;
                }
                let mid = 0.5 * low + 0.5 * high;
                probe[axis] = mid;
                let grad = hint_value(source, input, cf, cg, af, ag, phase, probe).1[axis];
                if !grad.is_finite() {
                    return None;
                }
                if grad > 0.0 {
                    low = mid;
                } else {
                    high = mid;
                }
            }
            probe[axis] = 0.5 * low + 0.5 * high;
            current = probe;
        }
        if !candidates.contains(&current) {
            candidates.push(current);
        }
    }
    Some(candidates)
}

fn optimized_intercept(
    source: &AffinePair<'_>,
    input: &[Input],
    cf: f64,
    cg: f64,
    af: &[f64],
    ag: &[f64],
    deadline: Option<Instant>,
) -> Option<f64> {
    let mut result = f64::INFINITY;
    for phase in PHASES {
        let mut lower = f64::NEG_INFINITY;
        for lambda in multipliers(source, input, cf, cg, af, ag, phase, deadline)? {
            lower = lower.max(phase_intercept(
                source, input, cf, cg, af, ag, phase, lambda, deadline,
            )?);
        }
        result = result.min(lower);
    }
    (admissible(result) && alive(deadline)).then_some(result)
}

fn affine_row<'a>(
    f: &'a ComposedAffine,
    g: &'a ComposedAffine,
    j: usize,
) -> Option<AffinePair<'a>> {
    if f.rows != g.rows || f.cols != g.cols || j >= f.rows {
        return None;
    }
    let a = j * f.cols;
    let b = a + f.cols;
    Some(AffinePair {
        wf: &f.weights[a..b],
        wg: &g.weights[a..b],
        bf: f.bias[j],
        bg: g.bias[j],
        wfl: &f.weight_lo[a..b],
        wfu: &f.weight_hi[a..b],
        wgl: &g.weight_lo[a..b],
        wgu: &g.weight_hi[a..b],
        bfi: Interval {
            lo: f.bias_lo[j],
            hi: f.bias_hi[j],
        },
        bgi: Interval {
            lo: g.bias_lo[j],
            hi: g.bias_hi[j],
        },
    })
}

fn source_center(
    chain: &PairedChain,
    domain: &BoxDomain,
    before: usize,
    deadline: Option<Instant>,
) -> Option<(Vec<f64>, Vec<f64>)> {
    let x: Vec<_> = domain
        .lo
        .iter()
        .zip(&domain.hi)
        .map(|(&l, &u)| 0.5 * l + 0.5 * u)
        .collect();
    let (mut f, mut g) = (x.clone(), x);
    for stage in chain.stages.iter().take(before) {
        if !alive(deadline) {
            return None;
        }
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
                for v in f.iter_mut().chain(&mut g) {
                    *v = v.max(0.0);
                }
            }
        }
        if f.iter().chain(&g).any(|v| !v.is_finite()) {
            return None;
        }
    }
    Some((f, g))
}

/// Certified `cf*rF + cg*rG >= af*u + ag*v + k`, where `u,v` are the
/// source towers at `start` and `rF,rG` are neuron outputs at `stage + 1`.
/// The fields are private to the source-owned engine; callers cannot import
/// a support or replace its right-hand side through a public certificate.
pub(super) struct Support {
    pub(super) start: usize,
    pub(super) neuron: usize,
    pub(super) cf: f64,
    pub(super) cg: f64,
    pub(super) af: Vec<f64>,
    pub(super) ag: Vec<f64>,
    pub(super) k: f64,
}

/// Source-owned supports; no replacement row or phase premise can
/// enter through a proposal artifact. This bounded portfolio introduces no
/// variables and preserves every existing authored equation/variable index.
/// Exactly opposite +/-1 output normals can also be transported directly as
/// lower/upper difference forms by a separately rounding-repaired caller.
pub(super) fn supports(
    chain: &PairedChain,
    domain: &BoxDomain,
    states: &[Enclosure],
    stage: usize,
    deadline: Option<Instant>,
) -> Option<Vec<Support>> {
    supports_mode(chain, domain, states, stage, deadline, false)
}

/// The same certifier with only exact +/-D normals. Forward callers need no
/// asymmetric LP normals; skipping those proposals affects cost, not proof
/// authority or the mandatory four-phase coverage of every returned support.
pub(super) fn supports_difference(
    chain: &PairedChain,
    domain: &BoxDomain,
    states: &[Enclosure],
    stage: usize,
    deadline: Option<Instant>,
) -> Option<Vec<Support>> {
    supports_mode(chain, domain, states, stage, deadline, true)
}

fn supports_mode(
    chain: &PairedChain,
    domain: &BoxDomain,
    states: &[Enclosure],
    stage: usize,
    deadline: Option<Instant>,
    difference_only: bool,
) -> Option<Vec<Support>> {
    if !alive(deadline) || !suitable_fp_environment() {
        return None;
    }
    chain.validate().ok()?;
    chain.input_state(domain)?;
    let Stage::Relu { width } = chain.stages.get(stage)? else {
        return None;
    };
    // Forward transport calls this before the target ReLU has produced an
    // output state. Its authenticated source prefix is sufficient: no later
    // enclosure is consulted by the certifier.
    if states.len() <= stage || states.len() > chain.stages.len() + 1 {
        return None;
    }
    let last_relu = !chain.stages[stage + 1..]
        .iter()
        .any(|s| matches!(s, Stage::Relu { .. }));
    let neuron_limit = if last_relu {
        MAX_NEURONS
    } else {
        MAX_INTERMEDIATE_NEURONS
    };
    let mut start = stage;
    while start > 0 && matches!(chain.stages[start - 1], Stage::Affine { .. }) {
        start -= 1;
    }
    if start == stage {
        return Some(Vec::new());
    }
    let s = &states[start];
    let n = s.f_lo.len();
    if [
        s.f_hi.len(),
        s.g_lo.len(),
        s.g_hi.len(),
        s.d_lo.len(),
        s.d_hi.len(),
    ]
    .iter()
    .any(|&m| m != n)
    {
        return None;
    }
    let pre = &states[stage];
    if [
        pre.f_lo.len(),
        pre.f_hi.len(),
        pre.g_lo.len(),
        pre.g_hi.len(),
    ]
    .iter()
    .any(|&m| m != *width)
    {
        return None;
    }
    let ranges: Vec<_> = (0..n)
        .map(|j| PairedReluRanges {
            lf: s.f_lo[j],
            uf: s.f_hi[j],
            lg: s.g_lo[j],
            ug: s.g_hi[j],
            dl: s.d_lo[j],
            du: s.d_hi[j],
        })
        .collect();
    let prepared = inputs(&ranges, deadline)?;
    let f = compose(chain, start, stage, n, false, deadline)?;
    let g = compose(chain, start, stage, n, true, deadline)?;
    if f.rows != *width || g.rows != *width {
        return None;
    }
    let center = source_center(chain, domain, start, deadline).unwrap_or_else(|| {
        (
            (0..n).map(|j| 0.5 * s.f_lo[j] + 0.5 * s.f_hi[j]).collect(),
            (0..n).map(|j| 0.5 * s.g_lo[j] + 0.5 * s.g_hi[j]).collect(),
        )
    });
    if !alive(deadline) {
        return None;
    }
    let bf: Vec<_> = s.f_lo.iter().copied().zip(s.f_hi.iter().copied()).collect();
    let bg: Vec<_> = s.g_lo.iter().copied().zip(s.g_hi.iter().copied()).collect();
    let mut end = stage + 1;
    while matches!(chain.stages.get(end), Some(Stage::Affine { .. })) {
        end += 1;
    }
    let head = if end > stage + 1 {
        Some((
            compose(chain, stage + 1, end, *width, false, deadline)?,
            compose(chain, stage + 1, end, *width, true, deadline)?,
        ))
    } else {
        None
    };
    let mut ranking = Vec::new();
    for j in 0..*width {
        let p = &states[stage];
        if (p.f_lo[j] >= 0.0 || p.f_hi[j] <= 0.0) && (p.g_lo[j] >= 0.0 || p.g_hi[j] <= 0.0) {
            continue;
        }
        let gap = |l: f64, u: f64| {
            if l < 0.0 && u > 0.0 {
                u * (-l / (u - l))
            } else {
                0.0
            }
        };
        let weight = head
            .as_ref()
            .map(|(f, g)| {
                (0..f.rows)
                    .map(|i| f.weights[i * f.cols + j].abs() + g.weights[i * g.cols + j].abs())
                    .fold(0.0, f64::max)
            })
            .unwrap_or(1.0);
        let score = weight * gap(p.f_lo[j], p.f_hi[j]).max(gap(p.g_lo[j], p.g_hi[j]));
        ranking.push((j, if score.is_finite() { score } else { 0.0 }));
    }
    ranking.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let mut selected: Vec<_> = ranking
        .iter()
        .take(neuron_limit - 2)
        .map(|&(j, _)| j)
        .collect();
    for j in [
        ranking.iter().map(|&(j, _)| j).min(),
        ranking.iter().map(|&(j, _)| j).max(),
    ]
    .into_iter()
    .flatten()
    {
        if !selected.contains(&j) {
            selected.push(j);
        }
    }
    let mut result = Vec::new();
    for j in selected {
        if !alive(deadline) {
            return None;
        }
        let source = affine_row(&f, &g, j)?;
        if !source.valid(n) {
            return None;
        }
        let sep_f = separating_slopes(source.wf, source.bf, &bf, &center.0)?;
        let sep_g = separating_slopes(source.wg, source.bg, &bg, &center.1)?;
        let mut normals = vec![(1.0, -1.0), (-1.0, 1.0)];
        if let Some((f, g)) = head.as_ref().filter(|_| !difference_only) {
            if let Some(i) = (0..f.rows).max_by(|&i, &k| {
                (f.weights[i * f.cols + j].abs() + g.weights[i * g.cols + j].abs())
                    .total_cmp(&(f.weights[k * f.cols + j].abs() + g.weights[k * g.cols + j].abs()))
            }) {
                let a = f.weights[i * f.cols + j];
                let b = -g.weights[i * g.cols + j];
                let scale = a.abs().max(b.abs());
                if scale > 0.0 && scale.is_finite() {
                    for normal in [(a / scale, b / scale), (-a / scale, -b / scale)] {
                        if !normals.contains(&normal) {
                            normals.push(normal);
                        }
                    }
                }
            }
        }
        for (cf, cg) in normals {
            // Shared half-slopes preserve common/difference structure even
            // when the two numerical Lovasz chains select different sectors.
            // They are advice only and receive the identical four-phase replay.
            let mut proposals = vec![(
                source.wf.iter().map(|&w| cf * 0.5 * w).collect::<Vec<_>>(),
                source.wg.iter().map(|&w| cg * 0.5 * w).collect::<Vec<_>>(),
            )];
            for (slopes, weights) in [(&sep_f, source.wf), (&sep_g, source.wg)] {
                let fraction: Vec<_> = slopes
                    .iter()
                    .zip(weights)
                    .map(|(&a, &w)| {
                        if w == 0.0 {
                            0.0
                        } else {
                            (a / w).clamp(0.0, 1.0)
                        }
                    })
                    .collect();
                let af: Vec<_> = source
                    .wf
                    .iter()
                    .zip(&fraction)
                    .map(|(&w, &a)| cf * a * w)
                    .collect();
                let ag: Vec<_> = source
                    .wg
                    .iter()
                    .zip(&fraction)
                    .map(|(&w, &a)| cg * a * w)
                    .collect();
                if !proposals.contains(&(af.clone(), ag.clone())) {
                    proposals.push((af, ag));
                }
            }
            for (af, ag) in proposals {
                if af.iter().chain(&ag).any(|&v| !admissible(v)) {
                    return None;
                }
                let intercept =
                    optimized_intercept(&source, &prepared, cf, cg, &af, &ag, deadline)?;
                result.push(Support {
                    start,
                    neuron: j,
                    cf,
                    cg,
                    af,
                    ag,
                    k: intercept,
                });
                if result.len() >= MAX_ROWS {
                    return alive(deadline).then_some(result);
                }
            }
        }
    }
    alive(deadline).then_some(result)
}

/// Bind the same certified supports to the authenticated LP variables. Keep
/// duplicate source indices: source F/G may alias the shared root input, and
/// approximate coefficient coalescing must not rewrite authoritative rows.
pub(super) fn rows(
    chain: &PairedChain,
    domain: &BoxDomain,
    states: &[Enclosure],
    stage: usize,
    bindings: &[(Vec<usize>, Vec<usize>)],
    deadline: Option<Instant>,
) -> Option<Vec<LinearRow>> {
    if bindings.len() != states.len() {
        return None;
    }
    let portfolio = supports(chain, domain, states, stage, deadline)?;
    let mut result = Vec::with_capacity(portfolio.len());
    for support in portfolio {
        if !alive(deadline) {
            return None;
        }
        let source = bindings.get(support.start)?;
        let output = bindings.get(stage.checked_add(1)?)?;
        if source.0.len() != support.af.len() || source.1.len() != support.ag.len() {
            return None;
        }
        let mut terms = vec![
            (*output.0.get(support.neuron)?, support.cf),
            (*output.1.get(support.neuron)?, support.cg),
        ];
        terms.extend(
            source
                .0
                .iter()
                .copied()
                .zip(support.af.into_iter().map(|a| -a))
                .filter(|&(_, a)| a != 0.0),
        );
        terms.extend(
            source
                .1
                .iter()
                .copied()
                .zip(support.ag.into_iter().map(|a| -a))
                .filter(|&(_, a)| a != 0.0),
        );
        result.push(LinearRow {
            terms,
            rhs: support.k,
        });
    }
    alive(deadline).then_some(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn ranges(n: usize, delta: f64) -> Vec<PairedReluRanges> {
        (0..n)
            .map(|_| PairedReluRanges {
                lf: -1.0,
                uf: 1.0,
                lg: -1.0,
                ug: 1.0,
                dl: -delta,
                du: delta,
            })
            .collect()
    }
    fn exact<'a>(wf: &'a [f64], bf: f64, wg: &'a [f64], bg: f64) -> AffinePair<'a> {
        AffinePair {
            wf,
            wg,
            bf,
            bg,
            wfl: wf,
            wfu: wf,
            wgl: wg,
            wgu: wg,
            bfi: Interval::point(bf),
            bgi: Interval::point(bg),
        }
    }

    #[test]
    fn identical_affine_twins_certify_zero_difference_from_input_coupling() {
        let wf = [1.0, -0.5, 0.25];
        let source = exact(&wf, -0.125, &wf, -0.125);
        let input = inputs(&ranges(3, 0.0), None).unwrap();
        let af = [0.5, -0.25, 0.125];
        let ag = [-0.5, 0.25, -0.125];
        let k = optimized_intercept(&source, &input, 1.0, -1.0, &af, &ag, None).unwrap();
        assert!(k <= 0.0 && k > -1e-12, "k={k}");
    }

    #[test]
    fn arbitrary_two_input_planes_and_duals_cover_all_dyadic_phase_traces() {
        let wf = [1.0, -0.5];
        let wg = [1.125, -0.5];
        let r = ranges(2, 0.25);
        let af = [0.5, -0.25];
        let ag = [-0.5, 0.25];
        let k = certified_intercept(
            &wf,
            0.25,
            &wg,
            0.125,
            &r,
            1.0,
            -1.0,
            &af,
            &ag,
            [[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
            None,
        )
        .unwrap();
        let mut phases = [false; 4];
        for i in -4..=4 {
            for j in -4..=4 {
                for p in -4..=4 {
                    for q in -4..=4 {
                        let u = [i as f64 / 4.0, j as f64 / 4.0];
                        let v = [p as f64 / 4.0, q as f64 / 4.0];
                        if (u[0] - v[0]).abs() > 0.25 || (u[1] - v[1]).abs() > 0.25 {
                            continue;
                        }
                        let zf = wf[0] * u[0] + wf[1] * u[1] + 0.25;
                        let zg = wg[0] * v[0] + wg[1] * v[1] + 0.125;
                        phases[usize::from(zf >= 0.0) + 2 * usize::from(zg >= 0.0)] = true;
                        let value = zf.max(0.0)
                            - zg.max(0.0)
                            - af[0] * u[0]
                            - af[1] * u[1]
                            - ag[0] * v[0]
                            - ag[1] * v[1];
                        assert!(value >= k);
                    }
                }
            }
        }
        assert!(phases.iter().all(|&p| p));
    }

    #[test]
    fn three_input_polygon_product_covers_boundary_and_mixed_traces() {
        let wf = [1.0, 0.5, -0.25];
        let wg = [0.875, 0.5, -0.125];
        let af = [0.25, -0.5, 0.125];
        let ag = [-0.125, 0.25, -0.5];
        let k = certified_intercept(
            &wf,
            0.0,
            &wg,
            0.125,
            &ranges(3, 0.25),
            2.0,
            -3.0,
            &af,
            &ag,
            [[1.0, 0.5], [0.0, 1.0], [2.0, 0.0], [0.5, 0.25]],
            None,
        )
        .unwrap();
        let pairs = [
            (-1.0, -1.0),
            (0.0, 0.0),
            (1.0, 1.0),
            (0.0, 0.25),
            (0.25, 0.0),
        ];
        for a in pairs {
            for b in pairs {
                for c in pairs {
                    let u = [a.0, b.0, c.0];
                    let v = [a.1, b.1, c.1];
                    let mut zf = 0.0;
                    let mut zg = 0.125;
                    let mut value = 0.0;
                    for j in 0..3 {
                        zf += wf[j] * u[j];
                        zg += wg[j] * v[j];
                        value -= af[j] * u[j] + ag[j] * v[j];
                    }
                    value += 2.0 * zf.max(0.0) - 3.0 * zg.max(0.0);
                    assert!(value >= k);
                }
            }
        }
    }

    #[test]
    fn an_unsafe_mixed_phase_cannot_be_dropped_from_the_root_support() {
        let wf = [1.0];
        let wg = [-1.0];
        let source = exact(&wf, 0.0, &wg, 0.0);
        let input = inputs(&ranges(1, 0.0), None).unwrap();
        let lower = optimized_intercept(&source, &input, 1.0, -1.0, &[0.0], &[0.0], None).unwrap();
        assert!(lower <= -1.0);
        let same = phase_intercept(
            &source,
            &input,
            1.0,
            -1.0,
            &[0.0],
            &[0.0],
            (true, true),
            [2.0, 0.0],
            None,
        )
        .unwrap();
        assert!(same > -1e-12); // Using this conditional value at root is false.
        assert!(-1.0 < same); // Exact u=v=-1 gives rF-rG=-1.
    }

    #[test]
    fn two_and_three_input_closed_phase_vertices_are_enclosed_exactly() {
        fn det(m: [[f64; 3]; 3]) -> f64 {
            m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
        }
        for (wf, wg, zbound) in [
            ([1.0, 1.0, 0.0], [1.0, -1.0, 0.0], 0.0),
            ([1.0, 0.5, 0.25], [1.0, -0.5, 0.25], 1.0),
        ] {
            let bounds = [1.0, 1.0, zbound];
            let r: Vec<_> = bounds
                .iter()
                .map(|&b| PairedReluRanges {
                    lf: -b,
                    uf: b,
                    lg: -b,
                    ug: b,
                    dl: 0.0,
                    du: 0.0,
                })
                .collect();
            let af = [0.25, 0.125, 0.25];
            let ag = [-0.5, 0.125, -0.125];
            let lower = certified_intercept(
                &wf,
                0.125,
                &wg,
                -0.125,
                &r,
                2.0,
                -1.0,
                &af,
                &ag,
                [[1.0, 0.5], [0.5, 0.0], [2.0, 0.0], [0.0, 1.0]],
                None,
            )
            .unwrap();
            let mut planes = Vec::new();
            for axis in 0..3 {
                for rhs in [-bounds[axis], bounds[axis]] {
                    let mut a = [0.0; 3];
                    a[axis] = 1.0;
                    planes.push((a, rhs));
                }
            }
            planes.push((wf, -0.125));
            planes.push((wg, 0.125));
            let mut minimum = f64::INFINITY;
            let mut phases = [false; 4];
            let mut vertices = 0;
            for a in 0..planes.len() {
                for b in a + 1..planes.len() {
                    for c in b + 1..planes.len() {
                        let m = [planes[a].0, planes[b].0, planes[c].0];
                        let rhs = [planes[a].1, planes[b].1, planes[c].1];
                        let denominator = det(m);
                        if denominator == 0.0 {
                            continue;
                        }
                        // Nonzero determinants in these deliberately chosen fixtures
                        // are signed powers of two. Cramer's rule therefore computes
                        // every phase/box intersection as an EXACT dyadic f64.
                        assert_eq!(denominator.abs().to_bits() & ((1_u64 << 52) - 1), 0);
                        let mut x = [0.0; 3];
                        for axis in 0..3 {
                            let mut q = m;
                            for row in 0..3 {
                                q[row][axis] = rhs[row];
                            }
                            x[axis] = det(q) / denominator;
                        }
                        if (0..3).any(|i| x[i] < -bounds[i] || x[i] > bounds[i]) {
                            continue;
                        }
                        let zf = 0.125 + (0..3).map(|i| wf[i] * x[i]).sum::<f64>();
                        let zg = -0.125 + (0..3).map(|i| wg[i] * x[i]).sum::<f64>();
                        let value = 2.0 * zf.max(0.0)
                            - zg.max(0.0)
                            - (0..3).map(|i| (af[i] + ag[i]) * x[i]).sum::<f64>();
                        minimum = minimum.min(value);
                        vertices += 1;
                        for (index, (f, g)) in PHASES.iter().copied().enumerate() {
                            if (if f { zf >= 0.0 } else { zf <= 0.0 })
                                && (if g { zg >= 0.0 } else { zg <= 0.0 })
                            {
                                phases[index] = true;
                            }
                        }
                    }
                }
            }
            // A piecewise affine objective reaches its minimum on the union
            // of vertices of these four CLOSED phase polytopes. This checks
            // the whole fixture domain, rather than only a sampling grid.
            assert!(vertices >= 8 && phases.iter().all(|&p| p));
            assert!(lower <= minimum);
        }
    }

    #[test]
    fn scalar_rounding_composition_enclosures_and_underflow_are_not_omitted() {
        let wf = [1.0 + 2.0_f64.powi(-52)];
        let wg = [0.0];
        let r = [PairedReluRanges {
            lf: 1.0,
            uf: 1.0,
            lg: 0.0,
            ug: 0.0,
            dl: 1.0,
            du: 1.0,
        }];
        let k = certified_intercept(
            &wf,
            0.0,
            &wg,
            0.0,
            &r,
            -1.0,
            0.0,
            &[-1.0],
            &[0.0],
            [[0.0; 2]; 4],
            None,
        )
        .unwrap();
        assert!(k <= -2.0_f64.powi(-52));
        let tiny = Interval::point(f64::from_bits(1))
            .mul(Interval::point(0.5))
            .unwrap();
        assert!(tiny.lo <= 0.0 && tiny.hi >= f64::from_bits(1));
    }

    #[test]
    fn malformed_coefficients_empty_domains_and_expiry_refuse_all_phase_authority() {
        let call = |r: &[PairedReluRanges], l, deadline| {
            certified_intercept(
                &[1.0],
                0.0,
                &[1.0],
                0.0,
                r,
                1.0,
                -1.0,
                &[0.0],
                &[0.0],
                l,
                deadline,
            )
        };
        assert!(call(&ranges(1, 0.0), [[0.0; 2]; 4], Some(Instant::now())).is_none());
        let mut lambdas = [[0.0; 2]; 4];
        lambdas[2][1] = -1.0;
        assert!(call(&ranges(1, 0.0), lambdas, None).is_none());
        lambdas[2][1] = f64::NAN;
        assert!(call(&ranges(1, 0.0), lambdas, None).is_none());
        let mut bad = ranges(1, 0.0);
        bad[0].dl = 3.0;
        bad[0].du = 4.0;
        assert!(call(&bad, [[0.0; 2]; 4], None).is_none());
        assert!(call(&[], [[0.0; 2]; 4], None).is_none());
    }

    #[test]
    fn intermediate_prefix_has_bounded_difference_only_and_half_slope_supports() {
        let weights: Vec<_> = (0..8).map(|j| 1.0 + j as f64 / 8.0).collect();
        let bias: Vec<_> = (0..8).map(|j| (j as f64 - 3.5) / 8.0).collect();
        let mut head_g = vec![0.125; 8];
        head_g[0] = 0.25;
        let chain = PairedChain {
            input_dim: 1,
            output_dim: 1,
            stages: vec![
                Stage::Affine {
                    rows: 8,
                    cols: 1,
                    wf: weights.clone(),
                    wg: weights.clone(),
                    bf: bias.clone(),
                    bg: bias,
                },
                Stage::Relu { width: 8 },
                Stage::Affine {
                    rows: 1,
                    cols: 8,
                    wf: vec![0.125; 8],
                    wg: head_g,
                    bf: vec![-0.125],
                    bg: vec![-0.125],
                },
                Stage::Relu { width: 1 },
            ],
        };
        let domain = BoxDomain {
            lo: vec![-1.0],
            hi: vec![1.0],
        };
        let states = forward::states(&chain, &domain, None, true).unwrap();
        let difference = supports_difference(&chain, &domain, &states[..2], 1, None).unwrap();
        let full = supports(&chain, &domain, &states[..2], 1, None).unwrap();
        assert!(!difference.is_empty() && difference.len() <= MAX_INTERMEDIATE_NEURONS * 2 * 3);
        assert!(
            !full.is_empty()
                && full.len() <= MAX_INTERMEDIATE_NEURONS * 4 * 3
                && full.len() <= MAX_ROWS
        );
        let mut selected = Vec::new();
        for support in &difference {
            assert_eq!(support.start, 0);
            assert!(matches!(
                (support.cf, support.cg),
                (1.0, -1.0) | (-1.0, 1.0)
            ));
            if !selected.contains(&support.neuron) {
                selected.push(support.neuron);
                // Uniform shared half-slopes are the FIRST candidate for the
                // neuron, before either numerical center-derived Lovasz plane.
                assert_eq!(support.af, vec![0.5 * weights[support.neuron]]);
                assert_eq!(support.ag, vec![-0.5 * weights[support.neuron]]);
            }
            for i in -8..=8 {
                let x = i as f64 / 8.0;
                let pre = weights[support.neuron] * x
                    + chain
                        .stages
                        .first()
                        .and_then(|s| match s {
                            Stage::Affine { bf, .. } => Some(bf[support.neuron]),
                            _ => None,
                        })
                        .unwrap();
                let lhs = support.cf * pre.max(0.0) + support.cg * pre.max(0.0)
                    - support.af[0] * x
                    - support.ag[0] * x;
                assert!(lhs >= support.k);
            }
        }
        assert!(
            selected.len() <= MAX_INTERMEDIATE_NEURONS
                && selected.contains(&0)
                && selected.contains(&7)
        );
        assert!(full
            .iter()
            .any(|s| (s.cf, s.cg) != (1.0, -1.0) && (s.cf, s.cg) != (-1.0, 1.0)));
        assert!(
            supports_difference(&chain, &domain, &states[..2], 1, Some(Instant::now())).is_none()
        );
    }

    fn dyadic_parts(x: f64) -> (i128, i32) {
        assert!(x.is_finite());
        if x == 0.0 {
            return (0, 0);
        }
        let bits = x.to_bits();
        let exponent = ((bits >> 52) & 2047) as i32;
        let mut mantissa = (bits & ((1_u64 << 52) - 1)) as i128;
        let mut power = if exponent == 0 {
            -1074
        } else {
            mantissa |= 1_i128 << 52;
            exponent - 1023 - 52
        };
        let zeros = mantissa.trailing_zeros();
        mantissa >>= zeros;
        power += zeros as i32;
        if bits >> 63 != 0 {
            mantissa = -mantissa;
        }
        (mantissa, power)
    }

    // Independent exact-integer row replay. All trace arithmetic in the
    // fixture is dyadic with fewer than 53 significant bits. A proposed f64
    // row coefficient is itself an exact dyadic rational; 2^-120 is a fine
    // enough common unit for its products with these small source traces.
    fn exact_row_holds(row: &LinearRow, values: &[f64]) -> bool {
        let mut lhs = 0_i128;
        for &(index, coefficient) in &row.terms {
            let (a, ea) = dyadic_parts(coefficient);
            let (b, eb) = dyadic_parts(values[index]);
            if a == 0 || b == 0 {
                continue;
            }
            let power = ea + eb + 120;
            assert!(
                (0..127).contains(&power),
                "unsupported fixture exponent {power}"
            );
            let term = a
                .checked_mul(b)
                .unwrap()
                .checked_mul(1_i128 << power)
                .unwrap();
            lhs = lhs.checked_add(term).unwrap();
        }
        let (m, e) = dyadic_parts(row.rhs);
        let power = e + 120;
        let ceil = if m == 0 {
            0
        } else if power >= 0 {
            assert!(power < 127);
            m.checked_mul(1_i128 << power).unwrap()
        } else {
            let shift = -power;
            if shift >= 127 {
                if m > 0 {
                    1
                } else {
                    0
                }
            } else {
                let divisor = 1_i128 << shift;
                let q = m / divisor;
                q + i128::from(m > 0 && m % divisor != 0)
            }
        };
        lhs >= ceil
    }

    #[test]
    fn source_composition_binding_and_prefix_supports_contain_exact_dyadic_traces() {
        let delta = 1.0 / 64.0;
        let chain = PairedChain {
            input_dim: 2,
            output_dim: 1,
            stages: vec![
                Stage::Affine {
                    rows: 3,
                    cols: 2,
                    wf: vec![1.0, 0.5, -0.5, 1.0, 0.25, -0.75],
                    wg: vec![1.0 + delta, 0.5, -0.5, 1.0 - delta, 0.25, -0.75],
                    bf: vec![0.25, -0.5, 0.0],
                    bg: vec![0.25, -0.5, delta],
                },
                Stage::Relu { width: 3 },
                Stage::Affine {
                    rows: 2,
                    cols: 3,
                    wf: vec![1.0, -0.5, 0.25, -0.25, 0.75, 0.5],
                    wg: vec![1.0 + delta, -0.5, 0.25, -0.25, 0.75 - delta, 0.5],
                    bf: vec![0.0, 0.125],
                    bg: vec![delta, 0.125],
                },
                Stage::Affine {
                    rows: 2,
                    cols: 2,
                    wf: vec![0.5, 0.0, 0.0, 0.5],
                    wg: vec![0.5, 0.0, 0.0, 0.5],
                    bf: vec![-0.125, 0.25],
                    bg: vec![-0.125 + delta, 0.25],
                },
                Stage::Relu { width: 2 },
                Stage::Affine {
                    rows: 1,
                    cols: 2,
                    wf: vec![1.0, -0.5],
                    wg: vec![1.0, -0.5 + delta],
                    bf: vec![0.0],
                    bg: vec![0.0],
                },
            ],
        };
        let domain = BoxDomain {
            lo: vec![-1.0, -1.0],
            hi: vec![1.0, 1.0],
        };
        let states = forward::states(&chain, &domain, None, true).unwrap();
        let mut bindings = vec![(vec![0, 1], vec![0, 1])];
        let mut next = 2;
        for s in states.iter().skip(1) {
            let f = (next..next + s.f_lo.len()).collect();
            next += s.f_lo.len();
            let g = (next..next + s.g_lo.len()).collect();
            next += s.g_lo.len();
            bindings.push((f, g));
        }
        let full = supports(&chain, &domain, &states, 4, None).unwrap();
        let prefix = supports(&chain, &domain, &states[..5], 4, None).unwrap();
        assert!(!full.is_empty() && full.len() <= MAX_ROWS);
        assert_eq!(full.len(), prefix.len());
        for (a, b) in full.iter().zip(&prefix) {
            assert_eq!(a.start, 2);
            assert_eq!(
                (a.start, a.neuron, a.cf, a.cg, a.k),
                (b.start, b.neuron, b.cf, b.cg, b.k)
            );
            assert_eq!(a.af, b.af);
            assert_eq!(a.ag, b.ag);
        }
        let rows = rows(&chain, &domain, &states, 4, &bindings, None).unwrap();
        assert_eq!(rows.len(), full.len());
        for i in -4..=4 {
            for j in -4..=4 {
                let mut f = vec![i as f64 / 4.0, j as f64 / 4.0];
                let mut g = f.clone();
                let mut values = vec![0.0; next];
                values[0] = f[0];
                values[1] = f[1];
                for (stage_index, stage) in chain.stages.iter().enumerate() {
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
                            for a in 0..*rows {
                                for b in 0..*cols {
                                    nf[a] += wf[a * cols + b] * f[b];
                                    ng[a] += wg[a * cols + b] * g[b];
                                }
                            }
                            f = nf;
                            g = ng;
                        }
                        Stage::Relu { .. } => {
                            for v in f.iter_mut().chain(&mut g) {
                                *v = v.max(0.0);
                            }
                        }
                    }
                    for (&index, &value) in bindings[stage_index + 1]
                        .0
                        .iter()
                        .zip(&f)
                        .chain(bindings[stage_index + 1].1.iter().zip(&g))
                    {
                        values[index] = value;
                    }
                }
                for row in &rows {
                    assert!(
                        exact_row_holds(row, &values),
                        "source row rejected exact grid trace at ({i},{j})"
                    );
                }
            }
        }
        assert!(supports(&chain, &domain, &states[..4], 4, None).is_none());
        assert!(supports(&chain, &domain, &states, 4, Some(Instant::now())).is_none());
    }
}
