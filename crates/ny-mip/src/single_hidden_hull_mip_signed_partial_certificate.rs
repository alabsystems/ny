// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Complete selected-mask coverage with every unselected original negative.
use super::super::super::certificate::I;
use super::*;
fn mask_lower(
    cuts: &HullRootMipCuts<'_>,
    w: &Witness,
    tail_constant: I,
    tail_coeff: &[I],
    mask: usize,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<f64> {
    let p = cuts.problem;
    let c = &w.body;
    let mut coeff = tail_coeff.to_vec();
    let mut constant = tail_constant;
    if coeff.len() != p.input_dim || c.g.len() != p.input_dim {
        return None;
    }
    for i in 0..p.input_dim {
        coeff[i] = coeff[i].sub(I::point(c.g[i])?)?;
    }
    let alpha = c
        .alpha
        .get(mask.checked_mul(c.positive.len())?..(mask + 1).checked_mul(c.positive.len())?)?;
    for (slot, (&j, &a)) in c.positive.iter().zip(alpha).enumerate() {
        if !live(end) || !poll() || !a.is_finite() || !(0. ..=1.).contains(&a) {
            return None;
        }
        let k = c.coefficient(p, slot)?.scale(a)?;
        constant = constant.add(k.scale(p.b1[j])?)?;
        for i in 0..p.input_dim {
            coeff[i] = coeff[i].add(k.scale(p.w_row(j)[i])?)?;
        }
    }
    for (bit, &j) in c.negative.iter().enumerate() {
        if mask & (1usize.checked_shl(bit as u32)?) == 0 {
            continue;
        }
        if !live(end) || !poll() {
            return None;
        }
        let k = I::point(p.c[j])?;
        constant = constant.add(k.scale(p.b1[j])?)?;
        for i in 0..p.input_dim {
            coeff[i] = coeff[i].add(k.scale(p.w_row(j)[i])?)?;
        }
    }
    let mut lower = constant.lo;
    for i in 0..p.input_dim {
        lower = finite(add_down(lower, coeff[i].minimum(p.lower[i], p.upper[i])?))?;
    }
    (live(end) && poll()).then_some(lower)
}
fn valid(p: &HullMarginProblem, w: &Witness) -> Option<()> {
    let c = &w.body;
    if !(9..=16).contains(&w.negative.len())
        || c.negative.len() != SELECTED
        || c.g.len() != p.input_dim
        || c.g.iter().any(|v| !v.is_finite())
        || c.positive.is_empty()
        || c.positive.len() > 32
        || c.positive.len() != c.positive_units.len()
        || c.alpha.len() != 8usize.checked_mul(c.positive.len())?
        || w.tail.len() + SELECTED != w.negative.len()
        || w.tail_coeff.len() != p.input_dim
    {
        return None;
    }
    for (i, &j) in w.negative.iter().enumerate() {
        if w.negative[..i].contains(&j) || p.c.get(j).is_none_or(|v| !v.is_finite() || *v >= 0.) {
            return None;
        }
    }
    for (i, &j) in c.negative.iter().enumerate() {
        if c.negative[..i].contains(&j) || !w.negative.contains(&j) {
            return None;
        }
    }
    for (i, (&j, &u)) in c.positive.iter().zip(&c.positive_units).enumerate() {
        if c.positive[..i].contains(&j)
            || !matches!(u, 1 | 2)
            || p.c.get(j).is_none_or(|v| !v.is_finite() || *v <= 0.)
        {
            return None;
        }
    }
    Some(())
}
fn tail(
    row: &SignedMajorant,
    cuts: &HullRootMipCuts<'_>,
    w: &Witness,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<(I, Vec<I>)> {
    let p = cuts.problem;
    valid(p, w)?;
    let mut constant = I::zero();
    let mut coeff = vec![I::zero(); p.input_dim];
    let mut at = 0usize;
    for &j in &w.negative {
        if w.body.negative.contains(&j) {
            continue;
        }
        if !live(end) || !poll() {
            return None;
        }
        let t = w.tail.get(at)?;
        at += 1;
        let original = cuts
            .multipliers
            .g
            .get(j.checked_mul(p.input_dim)?..(j + 1).checked_mul(p.input_dim)?)?;
        if t.head != j
            || t.g.len() != p.input_dim
            || t.g
                .iter()
                .zip(original)
                .any(|(a, b)| a.to_bits() != b.to_bits())
        {
            return None;
        }
        let h = majorant_offset_upper(p.w_row(j), p.b1[j], &t.g, &p.lower, &p.upper)?;
        if h.to_bits() != t.upper.to_bits() {
            return None;
        }
        let c = I::point(p.c[j])?;
        constant = constant.add(c.scale(h)?)?;
        for i in 0..p.input_dim {
            coeff[i] = coeff[i].add(c.scale(t.g[i])?)?;
        }
        if !live(end) || !poll() {
            return None;
        }
        row.budget
            .state
            .lock()
            .ok()?
            .component_rows
            .joint
            .partial_tail_rechecks += 1;
    }
    if at != w.tail.len()
        || constant.lo.to_bits() != w.tail_constant.lo.to_bits()
        || constant.hi.to_bits() != w.tail_constant.hi.to_bits()
        || coeff
            .iter()
            .zip(&w.tail_coeff)
            .any(|(a, b)| a.lo.to_bits() != b.lo.to_bits() || a.hi.to_bits() != b.hi.to_bits())
    {
        return None;
    }
    Some((constant, coeff))
}
pub(super) fn check(
    row: &SignedMajorant,
    cuts: &HullRootMipCuts<'_>,
    w: &Witness,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<f64> {
    require_hull_environment().ok()?;
    let (constant, coeff) = tail(row, cuts, w, end, poll)?;
    let mut lower = f64::INFINITY;
    for mask in 0..8 {
        let value = mask_lower(cuts, w, constant, &coeff, mask, end, poll)?;
        lower = lower.min(finite(value)?);
        let mut c = row.budget.state.lock().ok()?;
        let r = &mut c.component_rows.joint;
        r.checked_masks += 1;
        r.partial_checked_masks += 1;
    }
    require_hull_environment().ok()?;
    (live(end) && poll()).then_some(lower)
}
