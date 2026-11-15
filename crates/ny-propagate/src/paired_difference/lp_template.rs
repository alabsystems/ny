// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unconditional causal activation-template difference cuts.
//!
//! A fixed bit s defines the EXACT identity r=s*z+e with e=r-s*z>=0
//! for every ReLU graph point, regardless of its actual activation phase.
//! Propagating these identities through the authored affine prefix gives
//! each preactivation as root-affine template plus a linear combination of
//! every earlier slack. For a pair, write delta=t(x)+sum c_i*e_i and certify
//! L<=t(x)<=U over the complete root box. Signed ReLU monotonicity and
//! subadditivity then give unconditional inequalities
//!
//! Dpost <= max(0,U)+sum max(c_i,0)*e_i,
//! Dpost >= min(0,L)+sum min(c_i,0)*e_i.
//!
//! Here interval-enclosed causal coefficients are replaced by their upper
//! positive/lower negative endpoints because every slack is nonnegative.
//! Root coefficient/bias intervals are evaluated outward on the root box.
//! Thus every authored multiplication and coefficient sum is repaired, even
//! in long prefixes and cancellation/underflow cases. The exact-f64 row
//! uses duplicate post/pre terms rather than approximately coalescing them.
//! No mask is a phase assumption and no earlier activation is discarded.
//!
//! A second bound retains the individual root templates HF and HG. By ReLU
//! monotonicity and its unit Lipschitz constant, the SAME causal penalties
//! bound Dpost above/below relative to rho(HF)-rho(HG). Independently enclose
//! that root pair with rho(HFhi)-rho(HGlo) and rho(HFlo)-rho(HGhi), then meet
//! the old clipped-difference constants before emitting the rows. This keeps
//! their coefficients unchanged and strengthens only their right-hand sides.
//! In particular, two globally inactive root templates give root pair zero,
//! even when their affine difference has a wide interval crossing zero.
//!
//! Masks from a source midpoint are proposals only. The bounded portfolio
//! uses shared F masks, shared G masks, and each tower's own masks, in that
//! order, deduplicating identical templates. A mismatch merely has a positive
//! slack and is already covered by the inequality.
//! This is an application of established slack/Lagrangian constructions;
//! no novelty, solver-completeness or competition-score claim is made.

use super::lp_certificate::{suitable_fp_environment, LinearRow};
use super::*;

const MAX_VARIABLES: usize = 8_192;
const MAX_ROWS: usize = 100_000;
const MAX_TERMS: usize = 5_000_000;
const MAX_LIVE_COEFFICIENTS: usize = 5_000_000;
const MAX_SLACKS: usize = 8_192;
const MAX_MAGNITUDE: f64 = 1e100;

#[path = "lp_template_output.rs"]
mod output;
#[path = "lp_template_soft_guard.rs"]
mod soft_guard;

pub(super) use soft_guard::SoftGuardCover;

/// Source-owned complete covers, retained for independent reproduction.
/// These are diagnostics; the emitted GEQ is reconstructed from the source.
pub(super) struct OutputCover {
    pub(super) output: usize,
    pub(super) sign: i8,
    pub(super) portfolio: usize,
    pub(super) upper: f64,
    pub(super) nodes: usize,
    pub(super) leaves: Vec<BoxDomain>,
}

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
        admissible(self.lo) && admissible(self.hi) && self.lo <= self.hi
    }
    fn neg(self) -> Self {
        Self {
            lo: -self.hi,
            hi: -self.lo,
        }
    }
    fn add(self, b: Self) -> Option<Self> {
        let lo = if self.lo == 0.0 {
            b.lo
        } else if b.lo == 0.0 {
            self.lo
        } else if self.lo == -b.lo {
            0.0
        } else {
            add_down(self.lo, b.lo)
        };
        let hi = if self.hi == 0.0 {
            b.hi
        } else if b.hi == 0.0 {
            self.hi
        } else if self.hi == -b.hi {
            0.0
        } else {
            add_up(self.hi, b.hi)
        };
        let r = Self { lo, hi };
        r.valid().then_some(r)
    }
    fn sub(self, b: Self) -> Option<Self> {
        self.add(b.neg())
    }
    fn scale(self, w: f64) -> Option<Self> {
        if !admissible(w) || !self.valid() {
            return None;
        }
        if w == 0.0 || (self.lo == 0.0 && self.hi == 0.0) {
            return Some(Self::point(0.0));
        }
        if w == 1.0 {
            return Some(self);
        }
        if w == -1.0 {
            return Some(self.neg());
        }
        let (l, u) = if w >= 0.0 {
            (self.lo, self.hi)
        } else {
            (self.hi, self.lo)
        };
        let r = Self {
            lo: mul_down(w, l),
            hi: mul_up(w, u),
        };
        r.valid().then_some(r)
    }
    fn product(self, b: Self) -> Option<Self> {
        if !self.valid() || !b.valid() {
            return None;
        }
        if (self.lo == 0.0 && self.hi == 0.0) || (b.lo == 0.0 && b.hi == 0.0) {
            return Some(Self::point(0.0));
        }
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for (a, b) in [
            (self.lo, b.lo),
            (self.lo, b.hi),
            (self.hi, b.lo),
            (self.hi, b.hi),
        ] {
            lo = lo.min(mul_down(a, b));
            hi = hi.max(mul_up(a, b));
        }
        let r = Self { lo, hi };
        r.valid().then_some(r)
    }
}

struct Expr {
    root: Vec<Interval>,
    bias: Interval,
    slack: Vec<Interval>,
}

#[derive(Clone, PartialEq, Eq)]
pub(super) struct Mask {
    pub(super) stage: usize,
    pub(super) f: Vec<u8>,
    pub(super) g: Vec<u8>,
}

#[derive(Clone, Copy)]
struct Slack {
    pre: usize,
    post: usize,
    mask: u8,
}

fn count(expressions: &[Expr]) -> Option<usize> {
    expressions.iter().try_fold(0usize, |n, e| {
        n.checked_add(e.root.len().checked_add(e.slack.len())?.checked_add(1)?)
    })
}

fn authored_affine(
    source: &[Expr],
    weights: &[f64],
    bias: &[f64],
    rows: usize,
    cols: usize,
    deadline: Option<Instant>,
) -> Option<Vec<Expr>> {
    if source.len() != cols
        || cols == 0
        || rows == 0
        || weights.len() != rows.checked_mul(cols)?
        || bias.len() != rows
    {
        return None;
    }
    let root = source.first()?.root.len();
    let slack = source.first()?.slack.len();
    if source
        .iter()
        .any(|e| e.root.len() != root || e.slack.len() != slack)
    {
        return None;
    }
    let required = rows.checked_mul(root.checked_add(slack)?.checked_add(1)?)?;
    if required > MAX_LIVE_COEFFICIENTS {
        return None;
    }
    let mut result = Vec::with_capacity(rows);
    for i in 0..rows {
        if !alive(deadline) || !admissible(bias[i]) {
            return None;
        }
        let mut e = Expr {
            root: vec![Interval::point(0.0); root],
            bias: Interval::point(bias[i]),
            slack: vec![Interval::point(0.0); slack],
        };
        for j in 0..cols {
            if !alive(deadline) {
                return None;
            }
            let w = weights[i * cols + j];
            if !admissible(w) {
                return None;
            }
            if w == 0.0 {
                continue;
            }
            e.bias = e.bias.add(source[j].bias.scale(w)?)?;
            for k in 0..root {
                e.root[k] = e.root[k].add(source[j].root[k].scale(w)?)?;
            }
            for k in 0..slack {
                e.slack[k] = e.slack[k].add(source[j].slack[k].scale(w)?)?;
            }
        }
        result.push(e);
    }
    alive(deadline).then_some(result)
}

fn apply_mask(mut source: Vec<Expr>, mask: &[u8], deadline: Option<Instant>) -> Option<Vec<Expr>> {
    if source.len() != mask.len() || mask.iter().any(|&s| s > 1) {
        return None;
    }
    let old = source.first()?.slack.len();
    let total = old.checked_add(source.len())?;
    if total > MAX_SLACKS {
        return None;
    }
    let width = source.len();
    let root = source.first()?.root.len();
    if width.checked_mul(root.checked_add(total)?.checked_add(1)?)? > MAX_LIVE_COEFFICIENTS {
        return None;
    }
    for (j, e) in source.iter_mut().enumerate() {
        if !alive(deadline) || e.slack.len() != old || e.root.len() != root {
            return None;
        }
        if mask[j] == 0 {
            e.root.fill(Interval::point(0.0));
            e.bias = Interval::point(0.0);
            e.slack.fill(Interval::point(0.0));
        }
        e.slack.resize(total, Interval::point(0.0));
        e.slack[old + j] = Interval::point(1.0);
    }
    alive(deadline).then_some(source)
}

fn root_difference(
    f: &Expr,
    g: &Expr,
    domain: &BoxDomain,
    deadline: Option<Instant>,
) -> Option<Interval> {
    if f.root.len() != domain.lo.len()
        || g.root.len() != domain.lo.len()
        || domain.hi.len() != domain.lo.len()
    {
        return None;
    }
    let mut result = f.bias.sub(g.bias)?;
    for j in 0..domain.lo.len() {
        if !alive(deadline) {
            return None;
        }
        let root = Interval {
            lo: domain.lo[j],
            hi: domain.hi[j],
        };
        result = result.add(f.root[j].sub(g.root[j])?.product(root)?)?;
    }
    (result.valid() && alive(deadline)).then_some(result)
}

/// Bounds of the individual ROOT affine template, not the network value:
/// earlier nonnegative slack contributions remain explicit in the cut.
fn root_value(e: &Expr, domain: &BoxDomain, deadline: Option<Instant>) -> Option<Interval> {
    if e.root.len() != domain.lo.len() || domain.hi.len() != domain.lo.len() {
        return None;
    }
    let mut result = e.bias;
    for j in 0..domain.lo.len() {
        if !alive(deadline) {
            return None;
        }
        result = result.add(e.root[j].product(Interval {
            lo: domain.lo[j],
            hi: domain.hi[j],
        })?)?;
    }
    (result.valid() && alive(deadline)).then_some(result)
}

fn append_slack(terms: &mut Vec<(usize, f64)>, slack: Slack, coefficient: f64) -> Option<()> {
    if !admissible(coefficient) || coefficient < 0.0 || slack.mask > 1 {
        return None;
    }
    if coefficient != 0.0 {
        terms.push((slack.post, coefficient));
        if slack.mask == 1 {
            terms.push((slack.pre, -coefficient));
        }
    }
    Some(())
}

fn pair_rows(
    f: &Expr,
    g: &Expr,
    fs: &[Slack],
    gs: &[Slack],
    post_f: usize,
    post_g: usize,
    domain: &BoxDomain,
    deadline: Option<Instant>,
) -> Option<[LinearRow; 2]> {
    if f.slack.len() != fs.len() || g.slack.len() != gs.len() {
        return None;
    }
    let template = root_difference(f, g, domain, deadline)?;
    let rf = root_value(f, domain, deadline)?;
    let rg = root_value(g, domain, deadline)?;
    // rho(z+t) <= rho(z)+sum max(K,0)e and the lower analogue
    // preserve exactly the SAME slack penalties used by the old delta clip.
    // Outward interval subtraction encloses the paired ROOT ReLU value;
    // exact zero identities avoid a spurious subnormal for inactive pairs.
    let root_pair = Interval {
        lo: rf.lo.max(0.0),
        hi: rf.hi.max(0.0),
    }
    .sub(Interval {
        lo: rg.lo.max(0.0),
        hi: rg.hi.max(0.0),
    })?;
    let upper_constant = template.hi.max(0.0).min(root_pair.hi);
    let lower_constant = template.lo.min(0.0).max(root_pair.lo);
    let mut upper = vec![(post_g, 1.0), (post_f, -1.0)];
    let mut lower = vec![(post_f, 1.0), (post_g, -1.0)];
    for (coefficient, slack) in f.slack.iter().copied().zip(fs.iter().copied()).chain(
        g.slack
            .iter()
            .copied()
            .map(Interval::neg)
            .zip(gs.iter().copied()),
    ) {
        if !alive(deadline) || !coefficient.valid() {
            return None;
        }
        append_slack(&mut upper, slack, coefficient.hi.max(0.0))?;
        append_slack(&mut lower, slack, -coefficient.lo.min(0.0))?;
    }
    Some([
        LinearRow {
            terms: upper,
            rhs: -upper_constant,
        },
        LinearRow {
            terms: lower,
            rhs: lower_constant,
        },
    ])
}

fn valid_bindings(chain: &PairedChain, bindings: &[(Vec<usize>, Vec<usize>)]) -> Option<()> {
    if bindings.len() != chain.stages.len().checked_add(1)?
        || bindings.first()?.0 != bindings.first()?.1
    {
        return None;
    }
    let mut width = chain.input_dim;
    for (s, (f, g)) in bindings.iter().enumerate() {
        if f.len() != width || g.len() != width || f.iter().chain(g).any(|&v| v >= MAX_VARIABLES) {
            return None;
        }
        if let Some(stage) = chain.stages.get(s) {
            width = match stage {
                Stage::Affine { rows, .. } => *rows,
                Stage::Relu { width } => *width,
            };
        }
    }
    Some(())
}

fn center_masks(
    chain: &PairedChain,
    domain: &BoxDomain,
    deadline: Option<Instant>,
) -> Option<Vec<Mask>> {
    let mut f: Vec<_> = domain
        .lo
        .iter()
        .zip(&domain.hi)
        .map(|(&l, &u)| 0.5 * l + 0.5 * u)
        .collect();
    let mut g = f.clone();
    let mut masks = Vec::new();
    for (s, stage) in chain.stages.iter().enumerate() {
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
                    if !alive(deadline) {
                        return None;
                    }
                    for j in 0..*cols {
                        nf[i] += wf[i * cols + j] * f[j];
                        ng[i] += wg[i * cols + j] * g[j];
                    }
                }
                if nf.iter().chain(&ng).any(|x| !x.is_finite()) {
                    return None;
                }
                f = nf;
                g = ng;
            }
            Stage::Relu { .. } => {
                let sf: Vec<_> = f.iter().map(|&z| u8::from(z >= 0.0)).collect();
                let sg: Vec<_> = g.iter().map(|&z| u8::from(z >= 0.0)).collect();
                masks.push(Mask {
                    stage: s,
                    f: sf,
                    g: sg,
                });
                for z in f.iter_mut().chain(&mut g) {
                    *z = z.max(0.0);
                }
            }
        }
    }
    alive(deadline).then_some(masks)
}

/// The explicit-mask replay is private and accepts no authored rows, bounds
/// or arithmetic coefficients from a certificate. Bits define slack bases,
/// not phase premises; wrong widths/stages/bits nevertheless refuse the call.
#[cfg(test)]
pub(super) fn rows_with_masks(
    chain: &PairedChain,
    domain: &BoxDomain,
    bindings: &[(Vec<usize>, Vec<usize>)],
    masks: &[Mask],
    deadline: Option<Instant>,
) -> Option<Vec<LinearRow>> {
    rows_with_masks_impl(chain, domain, bindings, masks, false, 0, deadline)
        .map(|(rows, _, _)| rows)
}

fn rows_with_masks_impl(
    chain: &PairedChain,
    domain: &BoxDomain,
    bindings: &[(Vec<usize>, Vec<usize>)],
    masks: &[Mask],
    include_output: bool,
    portfolio: usize,
    deadline: Option<Instant>,
) -> Option<(Vec<LinearRow>, Vec<LinearRow>, Vec<OutputCover>)> {
    if !alive(deadline) || !suitable_fp_environment() {
        return None;
    }
    chain.validate().ok()?;
    chain.input_state(domain)?;
    valid_bindings(chain, bindings)?;
    let expected: Vec<_> = chain
        .stages
        .iter()
        .enumerate()
        .filter_map(|(s, stage)| {
            if let Stage::Relu { width } = stage {
                Some((s, *width))
            } else {
                None
            }
        })
        .collect();
    if expected.len() != masks.len()
        || expected.iter().zip(masks).any(|(&(s, width), mask)| {
            mask.stage != s
                || mask.f.len() != width
                || mask.g.len() != width
                || mask.f.iter().chain(&mask.g).any(|&v| v > 1)
        })
    {
        return None;
    }
    let mut f = Vec::with_capacity(chain.input_dim);
    let mut g = Vec::with_capacity(chain.input_dim);
    for j in 0..chain.input_dim {
        let mut root = vec![Interval::point(0.0); chain.input_dim];
        root[j] = Interval::point(1.0);
        f.push(Expr {
            root: root.clone(),
            bias: Interval::point(0.0),
            slack: Vec::new(),
        });
        g.push(Expr {
            root,
            bias: Interval::point(0.0),
            slack: Vec::new(),
        });
    }
    let mut fs = Vec::new();
    let mut gs = Vec::new();
    let mut rows = Vec::new();
    let mut output_rows = Vec::new();
    let mut covers = Vec::new();
    let mut terms = 0usize;
    let mut mask_index = 0usize;
    for (s, stage) in chain.stages.iter().enumerate() {
        if !alive(deadline) {
            return None;
        }
        match stage {
            Stage::Affine {
                rows: height,
                cols,
                wf,
                wg,
                bf,
                bg,
            } => {
                let old = count(&f)?.checked_add(count(&g)?)?;
                let required_f = height.checked_mul(
                    chain
                        .input_dim
                        .checked_add(f.first()?.slack.len())?
                        .checked_add(1)?,
                )?;
                let required_g = height.checked_mul(
                    chain
                        .input_dim
                        .checked_add(g.first()?.slack.len())?
                        .checked_add(1)?,
                )?;
                if old.checked_add(required_f)?.checked_add(required_g)? > MAX_LIVE_COEFFICIENTS {
                    return None;
                }
                let nf = authored_affine(&f, wf, bf, *height, *cols, deadline)?;
                let ng = authored_affine(&g, wg, bg, *height, *cols, deadline)?;
                f = nf;
                g = ng;
            }
            Stage::Relu { width } => {
                let mask = masks.get(mask_index)?;
                mask_index += 1;
                if f.len() != *width || g.len() != *width {
                    return None;
                }
                for j in 0..*width {
                    if !alive(deadline) || rows.len().checked_add(2)? > MAX_ROWS {
                        return None;
                    }
                    for row in pair_rows(
                        &f[j],
                        &g[j],
                        &fs,
                        &gs,
                        bindings[s + 1].0[j],
                        bindings[s + 1].1[j],
                        domain,
                        deadline,
                    )? {
                        terms = terms.checked_add(row.terms.len())?;
                        if terms > MAX_TERMS {
                            return None;
                        }
                        rows.push(row);
                    }
                }
                // There is no later ReLU cut to consume the terminal mask or
                // the final affine output head. Avoid an unused composition.
                if mask_index == masks.len() {
                    if include_output {
                        if let Some(generated) = output::rows(
                            chain, domain, bindings, s, &f, &g, &fs, &gs, portfolio, deadline,
                        ) {
                            (output_rows, covers) = generated;
                        }
                    }
                    break;
                }
                let required_f = width.checked_mul(
                    chain
                        .input_dim
                        .checked_add(f.first()?.slack.len().checked_add(*width)?)?
                        .checked_add(1)?,
                )?;
                let required_g = width.checked_mul(
                    chain
                        .input_dim
                        .checked_add(g.first()?.slack.len().checked_add(*width)?)?
                        .checked_add(1)?,
                )?;
                if required_f.checked_add(required_g)? > MAX_LIVE_COEFFICIENTS {
                    return None;
                }
                f = apply_mask(f, &mask.f, deadline)?;
                g = apply_mask(g, &mask.g, deadline)?;
                for j in 0..*width {
                    fs.push(Slack {
                        pre: bindings[s].0[j],
                        post: bindings[s + 1].0[j],
                        mask: mask.f[j],
                    });
                    gs.push(Slack {
                        pre: bindings[s].1[j],
                        post: bindings[s + 1].1[j],
                        mask: mask.g[j],
                    });
                }
            }
        }
    }
    (alive(deadline) && mask_index == masks.len()).then_some((rows, output_rows, covers))
}

/// Build up to three source-selected center templates, shared F first, then
/// shared G and own tower masks. Identical whole-prefix templates are removed.
/// Every row replays the WHOLE authored prefix and covers all actual phases.
#[cfg(test)]
pub(super) fn rows(
    chain: &PairedChain,
    domain: &BoxDomain,
    bindings: &[(Vec<usize>, Vec<usize>)],
    deadline: Option<Instant>,
) -> Option<Vec<LinearRow>> {
    rows_portfolio_impl(chain, domain, bindings, false, deadline).map(|(rows, _, _)| rows)
}

/// Retain the old causal-row prefix and build whole-output supports from the
/// SAME prefix expressions, once per source-selected portfolio. No new LP
/// variables, authored equalities, or binding interpretations are introduced.
pub(super) fn rows_with_output(
    chain: &PairedChain,
    domain: &BoxDomain,
    bindings: &[(Vec<usize>, Vec<usize>)],
    deadline: Option<Instant>,
) -> Option<(Vec<LinearRow>, Vec<LinearRow>, Vec<OutputCover>)> {
    rows_portfolio_impl(chain, domain, bindings, true, deadline)
}

/// Preserve every old row before attempting optional own-mask soft-guard
/// supports. Charged supports have larger slack penalties and are APPENDED,
/// never substituted for the existing complete-cover output rows.
pub(super) fn rows_with_output_and_guards(
    chain: &PairedChain,
    domain: &BoxDomain,
    bindings: &[(Vec<usize>, Vec<usize>)],
    deadline: Option<Instant>,
) -> Option<(
    Vec<LinearRow>,
    Vec<LinearRow>,
    Vec<LinearRow>,
    Vec<OutputCover>,
    Vec<SoftGuardCover>,
)> {
    let (prefix, outputs, output_covers) = rows_with_output(chain, domain, bindings, deadline)?;
    // Optional expiry/refusal does not replace or partially modify the old
    // family. The enclosing model/checker still enforces its final deadline.
    let generated = (|| {
        let own = center_masks(chain, domain, deadline)?;
        let mut sf = own.clone();
        let mut sg = own.clone();
        for mask in &mut sf {
            mask.g = mask.f.clone();
        }
        for mask in &mut sg {
            mask.f = mask.g.clone();
        }
        let mut unique = Vec::new();
        for masks in [sf, sg, own.clone()] {
            if !unique.contains(&masks) {
                unique.push(masks);
            }
        }
        let reference_portfolio = unique.iter().position(|m| *m == own)?;
        soft_guard::rows(
            chain,
            domain,
            bindings,
            &own,
            reference_portfolio,
            &output_covers,
            deadline,
        )
    })();
    let (mut guards, mut guard_covers) = generated.unwrap_or_default();
    let old_terms = prefix
        .iter()
        .chain(&outputs)
        .try_fold(0usize, |n, row| n.checked_add(row.terms.len()))?;
    let mut guard_terms = guards
        .iter()
        .try_fold(0usize, |n, row| n.checked_add(row.terms.len()))?;
    let old_rows = prefix.len().checked_add(outputs.len())?;
    while !guards.is_empty()
        && (guard_terms > MAX_TERMS.checked_sub(old_terms)?
            || old_rows.checked_add(guards.len())? > MAX_ROWS)
    {
        guard_terms = guard_terms.checked_sub(guards.pop()?.terms.len())?;
        guard_covers.pop()?;
    }
    Some((prefix, outputs, guards, output_covers, guard_covers))
}

fn rows_portfolio_impl(
    chain: &PairedChain,
    domain: &BoxDomain,
    bindings: &[(Vec<usize>, Vec<usize>)],
    include_output: bool,
    deadline: Option<Instant>,
) -> Option<(Vec<LinearRow>, Vec<LinearRow>, Vec<OutputCover>)> {
    if !alive(deadline) || !suitable_fp_environment() {
        return None;
    }
    chain.validate().ok()?;
    chain.input_state(domain)?;
    valid_bindings(chain, bindings)?;
    let own = center_masks(chain, domain, deadline)?;
    let mut shared_f = own.clone();
    for mask in &mut shared_f {
        mask.g = mask.f.clone();
    }
    let mut shared_g = own.clone();
    for mask in &mut shared_g {
        mask.f = mask.g.clone();
    }
    let mut templates = Vec::new();
    for candidate in [shared_f, shared_g, own] {
        if !templates.contains(&candidate) {
            templates.push(candidate);
        }
    }
    let mut result = Vec::new();
    let mut output_result = Vec::new();
    let mut covers = Vec::new();
    let mut terms = 0usize;
    let mut output_terms = 0usize;
    for (portfolio, masks) in templates.into_iter().enumerate() {
        if !alive(deadline) {
            return None;
        }
        let (prefix, output, source_covers) = rows_with_masks_impl(
            chain,
            domain,
            bindings,
            &masks,
            include_output,
            portfolio,
            deadline,
        )?;
        for row in prefix {
            if result.len().checked_add(1)? > MAX_ROWS {
                return None;
            }
            terms = terms.checked_add(row.terms.len())?;
            if terms > MAX_TERMS {
                return None;
            }
            result.push(row);
        }
        if output.len() != source_covers.len() {
            return None;
        }
        for (row, cover) in output.into_iter().zip(source_covers) {
            let Some(next_terms) = output_terms.checked_add(row.terms.len()) else {
                continue;
            };
            if next_terms > MAX_TERMS || output_result.len() >= 30 {
                continue;
            }
            output_terms = next_terms;
            output_result.push(row);
            covers.push(cover);
        }
    }
    // Added cuts cannot consume the term allowance needed by the unchanged
    // old prefix. Drop optional output suffix rows first if the combined cap
    // is exceeded; covers always correspond one-for-one to retained rows.
    while output_terms > MAX_TERMS - terms {
        output_terms -= output_result.pop()?.terms.len();
        covers.pop()?;
    }
    alive(deadline).then_some((result, output_result, covers))
}

#[cfg(test)]
#[path = "lp_template_tests.rs"]
mod tests;
