// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Reuse ONE checked graph inequality for exact original-affine negative aliases.
use super::*;
pub(super) const MAX_ROW_TERMS: usize = 64;
/// Entire optional scan/copy batch is prepaid even if there are no aliases.
/// N is the authenticated captured negative list length, not all signed heads.
pub(super) fn quote(d: usize, q: usize, n: usize) -> Option<(usize, usize)> {
    if n == 0 {
        return Some((0, 0));
    }
    let width = d.checked_add(1)?;
    let terms = d.checked_add(q)?.checked_add(1)?;
    if n > 512 || terms > MAX_ROW_TERMS {
        return None;
    }
    let work = n
        .checked_mul(width)?
        .checked_mul(16)?
        .checked_add(n.checked_mul(terms)?.checked_mul(8)?)?
        .checked_add(n.checked_mul(64)?)?
        .checked_add(1024)?;
    let bytes = n
        .checked_mul(terms.checked_mul(16)?.checked_add(128)?)?
        .checked_add(4096)?;
    Some((work, bytes))
}
/// Called inside the paid canonical check AFTER its fresh original support.
/// The caller supplies the captured list without relocking its held owner.
/// Ascending membership makes duplicate validation linear, not quadratic.
pub(super) fn collect(
    p: &HullMarginProblem,
    head: usize,
    negative: &[usize],
    limit: usize,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<Vec<usize>> {
    if limit == 0
        || limit > p.hidden_dim
        || negative.len() != limit
        || head >= p.hidden_dim
        || p.c.len() != p.hidden_dim
        || p.b1.len() != p.hidden_dim
        || p.w1.len() != p.input_dim.checked_mul(p.hidden_dim)?
        || p.c[head].partial_cmp(&0.) != Some(std::cmp::Ordering::Less)
        || !p.c[head].is_finite()
        || !p.b1[head].is_finite()
    {
        return None;
    }
    let mut aliases = Vec::with_capacity(limit);
    let (mut previous, mut canonical) = (None, false);
    for &j in negative {
        if !live(end)
            || !poll()
            || j >= p.hidden_dim
            || previous.is_some_and(|old| j <= old)
            || p.c[j].partial_cmp(&0.) != Some(std::cmp::Ordering::Less)
            || !p.c[j].is_finite()
        {
            return None;
        }
        previous = Some(j);
        canonical |= j == head;
        let mut equal = p.b1[j].to_bits() == p.b1[head].to_bits();
        // Always price the complete admitted D scan, including near misses.
        for i in 0..p.input_dim {
            if !live(end) || !poll() || !p.w_row(head)[i].is_finite() {
                return None;
            }
            equal &= p.w_row(j)[i].to_bits() == p.w_row(head)[i].to_bits();
        }
        if equal {
            aliases.push(j);
        }
    }
    (canonical && !aliases.is_empty() && live(end) && poll()).then_some(aliases)
}
/// Canonical is built ONCE by the original exporter from the sealed proposal.
/// Each additional row is a memcpy clone plus one authenticated target-column
/// replacement; no repeated lambda/input arithmetic or fresh row authority.
/// The private caller constructs canonical directly from the sealed token; this
/// helper does not accept an externally supplied coefficient vector.
pub(super) fn expand(
    row: &SignedMajorant,
    cuts: &HullRootMipCuts<'_>,
    parts: &MipParts,
    binding: &binding::SignedPhaseBinding<'_>,
    token: &kernel::Checked,
    canonical: &RowSpec,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<Vec<RowSpec>> {
    let end = end
        .min(token.end)
        .min(row.authority())
        .min(row.budget.search_end)
        .min(cuts.deadline);
    if !binding.matches(cuts, parts)
        || row.source != std::ptr::from_ref(cuts.problem) as usize
        || !token.owned(row)
        || token.aliases.is_empty()
        || token.aliases.len() > token.alias_limit
        || canonical.coeffs.len() > MAX_ROW_TERMS
        || canonical.lb != f64::NEG_INFINITY
        || canonical.ub.to_bits() != token.upper.to_bits()
        || canonical.coeffs.first().copied()
            != Some((binding.head_column(token.proposal.head)?.0, 1.))
    {
        return None;
    }
    let mut out = Vec::with_capacity(token.aliases.len());
    for &head in &token.aliases {
        if !live(end) || !poll() {
            return None;
        }
        let target = binding.head_column(head)?.0;
        let mut coeffs = canonical.coeffs.clone();
        coeffs.first_mut()?.0 = target;
        out.push(RowSpec {
            lb: canonical.lb,
            ub: canonical.ub,
            coeffs,
        });
    }
    (live(end) && poll()).then_some(out)
}
#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_domination_alias_tests.rs"]
mod tests;
