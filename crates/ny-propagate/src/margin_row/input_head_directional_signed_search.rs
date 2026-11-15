// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! One original-q anchor scans every head before the bounded local shortlist.
use super::*;

pub(super) fn anchor(
    source: &Source,
    budget: &mut Budget,
) -> Option<(Direction, signed::PositiveTerms)> {
    let h = source.spec.heads;
    budget.charge(h.checked_mul(3)?)?;
    if source.q.len() != h || source.y.len() != h {
        return None;
    }
    let mut positive = Vec::with_capacity(h);
    let mut negative = Vec::with_capacity(h);
    for (head, &q) in source.q.iter().enumerate() {
        if head & 31 == 0 {
            budget.charge(0)?;
        }
        let point = signed::point(q)?;
        if q.lo > 0.0 && point > 0.0 {
            positive.push((head, point));
        } else if q.hi < 0.0 && point < 0.0 && source.y[head].lo < 0.0 && source.y[head].hi > 0.0 {
            let score = -point * source.y[head].hi;
            if good(score) {
                negative.push((head, -point, score));
            }
        }
    }
    if positive.is_empty() || negative.is_empty() {
        return None;
    }
    let log = usize::BITS as usize - h.max(1).leading_zeros() as usize;
    budget.charge(h.checked_mul(log.checked_add(1)?)?)?;
    negative.sort_by(|a, b| b.2.total_cmp(&a.2).then(a.0.cmp(&b.0)));
    negative.truncate(MAX_SUPPORT);
    negative.sort_by_key(|&(head, _, _)| head);
    Some((
        Direction {
            heads: negative
                .into_iter()
                .map(|(head, tau, _)| (head, tau))
                .collect(),
        },
        signed::PositiveTerms(positive),
    ))
}
