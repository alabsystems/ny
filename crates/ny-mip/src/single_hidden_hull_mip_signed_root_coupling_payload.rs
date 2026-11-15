// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! A prepaid maximum of live nominal Vec payloads, never a refund or RSS bound.
use super::*;
use std::mem::size_of;

/// Include unused allocated slots, including the Component Vec's five headers.
/// This reads O(C) descriptors and never scans Source or an alpha array.
pub(super) fn heap(plan: &Plan) -> Option<usize> {
    let mut bytes = plan
        .components
        .capacity()
        .checked_mul(size_of::<Component>())?
        .checked_add(plan.leftover.capacity().checked_mul(size_of::<usize>())?)?;
    for c in &plan.components {
        bytes = bytes
            .checked_add(c.negative.capacity().checked_mul(size_of::<usize>())?)?
            .checked_add(c.positive.capacity().checked_mul(size_of::<usize>())?)?
            .checked_add(c.positive_units.capacity().checked_mul(size_of::<u8>())?)?
            .checked_add(c.g.capacity().checked_mul(size_of::<f64>())?)?
            .checked_add(c.alpha.capacity().checked_mul(size_of::<f64>())?)?;
    }
    Some(bytes)
}

/// Geometry uses exact-size retained buffers; the old larger word bound stays.
/// 8*words already overprices u8 units and five length words per Component.
pub(super) fn plan_bound(words: usize, old_components: usize) -> Option<usize> {
    let word = size_of::<usize>()
        .max(size_of::<f64>())
        .max(size_of::<u64>());
    words
        .checked_mul(word)?
        .checked_add(
            old_components
                .checked_add(14)?
                .checked_mul(size_of::<Component>())?,
        )?
        .checked_add(128)
}

/// Same serialized value sequence as signed::witness; borrowed, no Vec scratch.
fn bits(plan: &Plan) -> impl Iterator<Item = u64> + '_ {
    [plan.components.len() as u64, plan.leftover.len() as u64]
        .into_iter()
        .chain(plan.leftover.iter().map(|v| *v as u64))
        .chain(plan.components.iter().flat_map(|c| {
            [
                c.negative.len() as u64,
                c.positive.len() as u64,
                c.positive_units.len() as u64,
                c.g.len() as u64,
                c.alpha.len() as u64,
            ]
            .into_iter()
            .chain(c.negative.iter().chain(&c.positive).map(|v| *v as u64))
            .chain(c.positive_units.iter().map(|v| *v as u64))
            .chain(c.g.iter().chain(&c.alpha).map(|v| v.to_bits()))
        }))
}
pub(super) fn matches(plan: &Plan, expected: &[u64]) -> bool {
    bits(plan).eq(expected.iter().copied())
}
pub(super) fn capacity_matches(
    plan: &Plan,
    recorded: usize,
    old_words: usize,
    old_components: usize,
    d: usize,
) -> bool {
    let limit = old_words
        .checked_add(4096)
        .and_then(|n| 14usize.checked_mul(d.checked_add(5)?)?.checked_add(n))
        .and_then(|n| plan_bound(n, old_components));
    heap(plan)
        .zip(limit)
        .is_some_and(|(actual, ceiling)| actual == recorded && actual <= ceiling)
}
pub(super) fn seal(plan: &Plan) -> Option<Box<[u64]>> {
    let mut out = vec![0; resources::words(plan)?].into_boxed_slice();
    let mut values = bits(plan);
    for v in &mut out {
        *v = values.next()?;
    }
    values.next().is_none().then_some(out)
}

/// One boxed old witness plus NINE complete candidate-Plan payload allowances.
/// At most three nonsharing whole Plans + two grown Vec witnesses coexist;
/// 3B+32N<=7B. Two further B cover component/trial and handoff scratch.
/// Half-sharing's independent simultaneous-family quote is kept unchanged.
/// Source signature storage is allocated once in the retained base path and
/// shared by Arc; this stage allocates no additional Source-sized signature.
pub(super) fn quote(
    plan: &Plan,
    d: usize,
    h: usize,
    p: usize,
    c: usize,
    old: usize,
    new: usize,
) -> Option<usize> {
    if c != plan.components.len() || new < old || resources::words(plan)? != old {
        return None;
    }
    let owned = plan_bound(new, c)?;
    if heap(plan)? > owned {
        return None;
    }
    old.checked_mul(size_of::<u64>())?
        .checked_add(owned.checked_mul(9)?)?
        .checked_add(
            c.checked_add(14)?
                .checked_mul(d.checked_add(1)?)?
                .checked_add(h)?
                .checked_add(p.checked_mul(2)?)?
                .checked_add(1)?
                .checked_mul(128)?,
        )?
        .checked_add(8192)?
        .checked_add(1536)
}

#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_root_coupling_payload_tests.rs"]
mod tests;
