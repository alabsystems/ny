// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Whole two-component mass, three exact negative choices and a checked tail.
//! This distinct witness is never an old full-mask Component certificate.
use super::super::certificate::I;
use super::*;
#[path = "single_hidden_hull_mip_signed_partial_certificate.rs"]
mod certificate;
const SELECTED: usize = 3;
#[derive(Debug)]
pub(super) struct Tail {
    pub head: usize,
    pub g: Vec<f64>,
    pub upper: f64,
}
#[derive(Debug)]
pub(super) struct Witness {
    pub negative: Vec<usize>,
    // Component is only nominal steering storage. Its selected negatives are
    // NOT the full negative mass, and Component::valid/mask_lower are not used.
    pub body: Component,
    pub tail: Vec<Tail>,
    pub tail_constant: I,
    pub tail_coeff: Vec<I>,
}
/// The exact partial-only fields retained alongside a common Component.
/// Moving these descriptors does not allocate or re-run any producer/checker.
#[derive(Debug)]
pub(super) struct Details {
    pub negative: Vec<usize>,
    pub tail: Vec<Tail>,
    pub tail_constant: I,
    pub tail_coeff: Vec<I>,
}
impl Witness {
    pub(super) fn into_parts(self) -> (Component, Details) {
        let Self {
            negative,
            body,
            tail,
            tail_constant,
            tail_coeff,
        } = self;
        (
            body,
            Details {
                negative,
                tail,
                tail_constant,
                tail_coeff,
            },
        )
    }
}
impl Details {
    pub(super) fn bits<'a>(&'a self, c: &'a Component) -> impl Iterator<Item = u64> + 'a {
        [
            self.negative.len() as u64,
            c.negative.len() as u64,
            c.positive.len() as u64,
            c.positive_units.len() as u64,
            c.g.len() as u64,
            c.alpha.len() as u64,
            self.tail.len() as u64,
            self.tail_coeff.len() as u64,
            self.tail_constant.lo.to_bits(),
            self.tail_constant.hi.to_bits(),
        ]
        .into_iter()
        .chain(
            self.negative
                .iter()
                .chain(&c.negative)
                .chain(&c.positive)
                .map(|v| *v as u64),
        )
        .chain(c.positive_units.iter().map(|v| u64::from(*v)))
        .chain(c.g.iter().chain(&c.alpha).map(|v| v.to_bits()))
        .chain(self.tail.iter().flat_map(|t| {
            [t.head as u64, t.g.len() as u64, t.upper.to_bits()]
                .into_iter()
                .chain(t.g.iter().map(|v| v.to_bits()))
        }))
        .chain(
            self.tail_coeff
                .iter()
                .flat_map(|v| [v.lo.to_bits(), v.hi.to_bits()]),
        )
    }
}
/// All scans, allocations, source tail checks, interval supports and integrity
/// passes are covered before build. Same 4M subcap and 64M/8MiB parent ledger.
pub(super) fn quote(
    p: &HullMarginProblem,
    a: &Component,
    b: &Component,
) -> Option<(usize, usize, usize)> {
    let k = a.negative.len().checked_add(b.negative.len())?;
    let pos = a.positive.len().checked_add(b.positive.len())?;
    if !(9..=16).contains(&k) || pos == 0 || pos > 32 {
        return None;
    }
    let masks = 1usize.checked_shl(SELECTED as u32)?;
    let width = p.input_dim.checked_add(1)?;
    let per = width
        .checked_mul(pos.checked_add(k)?.checked_add(3)?)?
        .checked_mul(4)?
        .checked_add(pos.min(4).checked_mul(width)?.checked_mul(20)?)?;
    let terms = p
        .hidden_dim
        .checked_add(p.input_dim)?
        .checked_add(pos)?
        .checked_add(k)?
        .checked_add(16)?;
    let tail = k.checked_mul(width)?.checked_mul(64)?;
    let tail_work = tail.checked_mul(2)?;
    let work = masks
        .checked_mul(per)?
        .checked_mul(4)?
        .checked_add(terms.checked_mul(128)?)?
        .checked_add(tail_work)?;
    let bytes = masks
        .checked_mul(pos)?
        .checked_mul(32)?
        .checked_add(terms.checked_mul(256)?)?
        .checked_add(tail)?
        .checked_add(8192)?;
    (work <= resources::MAX_JOINT_WORK).then_some((masks, work, bytes))
}
fn energy(p: &HullMarginProblem, j: usize) -> Option<f64> {
    let mut value = p.b1.get(j)?.abs();
    for i in 0..p.input_dim {
        value = finite(value + p.w_row(j)[i].abs() * (p.upper[i] - p.lower[i]))?;
    }
    finite(value * p.c.get(j)?.abs())
}
fn selected(p: &HullMarginProblem, negative: &[usize]) -> Option<Vec<usize>> {
    let mut out = Vec::with_capacity(SELECTED);
    for _ in 0..SELECTED {
        let mut best = None;
        for &j in negative {
            if out.contains(&j) {
                continue;
            }
            let score = energy(p, j)?;
            if best.is_none_or(|(_, v)| score > v) {
                best = Some((j, score));
            }
        }
        out.push(best?.0);
    }
    Some(out)
}
fn geometry(
    row: &SignedMajorant,
    cuts: &HullRootMipCuts<'_>,
    choice: &resources::Choice,
) -> Option<Witness> {
    let p = cuts.problem;
    if !choice.partial || choice.left >= choice.right || choice.masks != 8 {
        return None;
    }
    let a = row.plan.components.get(choice.left)?;
    let b = row.plan.components.get(choice.right)?;
    let negative: Vec<_> = a.negative.iter().chain(&b.negative).copied().collect();
    if !(9..=16).contains(&negative.len())
        || negative
            .iter()
            .enumerate()
            .any(|(i, j)| negative[..i].contains(j))
        || negative
            .iter()
            .any(|j| p.c.get(*j).is_none_or(|c| !c.is_finite() || *c >= 0.))
    {
        return None;
    }
    let mut positive = Vec::with_capacity(a.positive.len() + b.positive.len());
    let mut units: Vec<u8> = Vec::with_capacity(positive.capacity());
    for c in [a, b] {
        for (&j, &u) in c.positive.iter().zip(&c.positive_units) {
            if !matches!(u, 1 | 2) || p.c.get(j).is_none_or(|v| !v.is_finite() || *v <= 0.) {
                return None;
            }
            if let Some(i) = positive.iter().position(|old| *old == j) {
                units[i] = units[i].checked_add(u)?;
                if units[i] > 2 {
                    return None;
                }
            } else {
                positive.push(j);
                units.push(u);
            }
        }
    }
    if positive.is_empty()
        || positive.len() > 32
        || a.g.len() != p.input_dim
        || b.g.len() != p.input_dim
    {
        return None;
    }
    let g: Vec<_> =
        a.g.iter()
            .zip(&b.g)
            .map(|(x, y)| finite(x + y))
            .collect::<Option<_>>()?;
    let initial: Vec<_> = positive
        .iter()
        .map(|&j| cuts.multipliers.alpha.get(j).copied().map(sanitize_alpha))
        .collect::<Option<_>>()?;
    let body = Component {
        negative: selected(p, &negative)?,
        positive,
        positive_units: units,
        g,
        alpha: initial.repeat(1usize.checked_shl(SELECTED as u32)?),
    };
    Some(Witness {
        negative,
        body,
        tail: Vec::with_capacity(16 - SELECTED),
        tail_constant: I::zero(),
        tail_coeff: vec![I::zero(); p.input_dim],
    })
}
fn prepare_tail(
    row: &SignedMajorant,
    cuts: &HullRootMipCuts<'_>,
    w: &mut Witness,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<(f64, Vec<f64>)> {
    let p = cuts.problem;
    let d = p.input_dim;
    let mut coeff = vec![0.; d];
    let mut constant = 0.;
    require_hull_environment().ok()?;
    for &j in &w.negative {
        if w.body.negative.contains(&j) {
            continue;
        }
        if !live(end) || !poll() {
            return None;
        }
        let g = cuts
            .multipliers
            .g
            .get(j.checked_mul(d)?..(j + 1).checked_mul(d)?)?
            .to_vec();
        if g.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let h = majorant_offset_upper(p.w_row(j), p.b1[j], &g, &p.lower, &p.upper)?;
        let c = I::point(p.c[j])?;
        w.tail_constant = w.tail_constant.add(c.scale(h)?)?;
        constant = finite(constant + p.c[j] * h)?;
        for i in 0..d {
            w.tail_coeff[i] = w.tail_coeff[i].add(c.scale(g[i])?)?;
            coeff[i] = finite(coeff[i] + p.c[j] * g[i])?;
        }
        if !live(end) || !poll() {
            return None;
        }
        w.tail.push(Tail {
            head: j,
            g,
            upper: h,
        });
        row.budget
            .state
            .lock()
            .ok()?
            .component_rows
            .joint
            .partial_tail_rechecks += 1;
    }
    require_hull_environment().ok()?;
    (live(end) && poll()).then_some((constant, coeff))
}
pub(super) fn produce(
    row: &SignedMajorant,
    cuts: &HullRootMipCuts<'_>,
    choice: &resources::Choice,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<(Witness, f64)> {
    let p = cuts.problem;
    let mut w = geometry(row, cuts, choice)?;
    {
        let mut c = row.budget.state.lock().ok()?;
        let r = &mut c.component_rows.joint;
        r.partial_selected_heads = w.body.negative.len();
        r.partial_tail_heads = w.negative.len() - w.body.negative.len();
    }
    let (constant, coeff) = prepare_tail(row, cuts, &mut w, end, poll)?;
    let mut coordinates = 0;
    let steered = steering::steer_with_tail(
        p,
        &mut w.body,
        Some((constant, &coeff)),
        end,
        poll,
        &mut coordinates,
    );
    row.budget
        .state
        .lock()
        .ok()?
        .component_rows
        .joint
        .coordinates += coordinates;
    steered?;
    let lower = certificate::check(row, cuts, &w, end, poll)?;
    Some((w, lower))
}
#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_partial_retention_tests.rs"]
mod retention_tests;
#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_partial_test_fixture.rs"]
mod test_fixture;
#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_partial_tests.rs"]
mod tests;
