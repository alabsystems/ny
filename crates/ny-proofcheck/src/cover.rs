// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The per-(region, disjunct) cover check (brief §4.4). Independent code: it
//! shares nothing with the producer's `CoverLedger`.
//!
//! A member is a node's literal set, keyed by CHECKER-resolved site identity
//! (a Relu input tensor and flat index, or an input index and exact
//! threshold). The check rebuilds a binary case split: a sub-problem is the
//! set of members consistent with the literals chosen so far;
//!
//! 1. a member whose every literal is already chosen covers the sub-cell
//!    (overlaps are allowed — they never hurt soundness);
//! 2. no member left is a GAP, reported with its witness assignment;
//! 3. otherwise split on the unchosen site present in the most members
//!    (ties: the smallest site key); a member with the opposite side leaves
//!    that half, a member lacking the site goes to BOTH halves.
//!
//! Each step is the identity `cell(S) = cell(S ∧ s≥0) ∪ cell(S ∧ s≤0)`,
//! true for ANY site, so a producer that mislabels or reorders its splits
//! can only produce a gap, never a false cover.

use std::collections::BTreeMap;

/// A checker-resolved site: `(kind, a, b)` — kind 0 is a Relu input
/// `(tensor, index)`, kind 1 an input threshold `(index, t.to_bits())`.
pub type SiteKey = (u8, u64, u64);

/// A literal: the site and its side (`true` = `>=`).
pub type CoverLit = (SiteKey, bool);

/// The answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoverResult {
    /// The members cover the region.
    Covered,
    /// A cell no member covers, as its assignment.
    Gap(Vec<CoverLit>),
    /// The sub-problem budget ran out.
    Budget,
}

/// Default sub-problem budget.
pub const DEFAULT_BUDGET: usize = 10_000_000;

/// Decide whether `members` (each sorted by site, one side per site) cover
/// the root cell.
#[must_use]
pub fn check_cover(members: &[Vec<CoverLit>], budget: usize) -> CoverResult {
    struct Work {
        members: Vec<usize>,
        path: Vec<CoverLit>,
    }
    let side_in = |lits: &[CoverLit], site: SiteKey| -> Option<bool> {
        lits.binary_search_by(|l| l.0.cmp(&site))
            .ok()
            .map(|i| lits[i].1)
    };
    let mut stack = vec![Work {
        members: (0..members.len()).collect(),
        path: Vec::new(),
    }];
    let mut steps = 0usize;
    while let Some(Work { members: mem, path }) = stack.pop() {
        steps += 1;
        if steps > budget {
            return CoverResult::Budget;
        }
        if mem.is_empty() {
            return CoverResult::Gap(path);
        }
        let mut sorted_path = path.clone();
        sorted_path.sort_unstable();
        let chosen = |s: SiteKey| sorted_path.binary_search_by(|l| l.0.cmp(&s)).is_ok();
        if mem
            .iter()
            .any(|&m| members[m].iter().all(|&(s, _)| chosen(s)))
        {
            continue;
        }
        let mut counts: BTreeMap<SiteKey, usize> = BTreeMap::new();
        for &m in &mem {
            for &(s, _) in &members[m] {
                if !chosen(s) {
                    *counts.entry(s).or_insert(0) += 1;
                }
            }
        }
        // Most frequent; ties resolve to the smallest key (BTreeMap order).
        let mut best: Option<(SiteKey, usize)> = None;
        for (&s, &c) in &counts {
            if best.is_none_or(|(_, bc)| c > bc) {
                best = Some((s, c));
            }
        }
        let Some((site, _)) = best else {
            // Every member is consistent and fully chosen was handled above;
            // nothing left to split on means the cell is uncovered.
            return CoverResult::Gap(path);
        };
        let mut pos = Vec::new();
        let mut neg = Vec::new();
        for &m in &mem {
            match side_in(&members[m], site) {
                Some(true) => pos.push(m),
                Some(false) => neg.push(m),
                None => {
                    pos.push(m);
                    neg.push(m);
                }
            }
        }
        let mut pn = path.clone();
        pn.push((site, false));
        let mut pp = path;
        pp.push((site, true));
        stack.push(Work {
            members: neg,
            path: pn,
        });
        stack.push(Work {
            members: pos,
            path: pp,
        });
    }
    CoverResult::Covered
}

// ---------------------------------------------------------------------------
// B2.2: the box-aware (hybrid) cover
// ---------------------------------------------------------------------------

/// One member of a box-aware cover problem: a node's region as the CHECKER
/// resolved it — its box (the VNN-LIB region box cut by the node's input
/// threshold literals, [`crate::check`]'s `node_box`) and its Relu literals
/// (sorted by site, one side per site).
#[derive(Debug, Clone, PartialEq)]
pub struct BoxMember {
    /// Box lower ends.
    pub lo: Vec<f64>,
    /// Box upper ends.
    pub hi: Vec<f64>,
    /// Relu literals, sorted by site (kind-0 [`SiteKey`]s only).
    pub relu: Vec<CoverLit>,
}

/// The answer of [`check_box_cover`].
#[derive(Debug, Clone, PartialEq)]
pub enum BoxCoverResult {
    /// The members cover the region.
    Covered,
    /// A cell no member covers: a sub-box of positive measure (degenerate
    /// only where the region itself is) under a Relu literal assignment.
    /// The box ends are boxed slices (fixed length) so the variant stays
    /// within clippy::large_enum_variant's 64-byte workspace threshold.
    Gap {
        /// Gap box lower ends.
        lo: Box<[f64]>,
        /// Gap box upper ends.
        hi: Box<[f64]>,
        /// Relu assignment of the gap cell.
        path: Vec<CoverLit>,
    },
    /// The sub-problem budget ran out.
    Budget,
}

/// Whether member box `(mlo, mhi)` meets box `[lo, hi]` with positive measure
/// in every dimension where the box has positive width, and contains the
/// box's coordinate in every degenerate one.
fn meets(mlo: &[f64], mhi: &[f64], lo: &[f64], hi: &[f64]) -> bool {
    lo.iter().zip(hi).enumerate().all(|(d, (&l, &h))| {
        if l < h {
            mlo[d] < h && mhi[d] > l
        } else {
            mlo[d] <= l && mhi[d] >= h
        }
    })
}

fn holds_box(mlo: &[f64], mhi: &[f64], lo: &[f64], hi: &[f64]) -> bool {
    lo.iter()
        .zip(hi)
        .enumerate()
        .all(|(d, (&l, &h))| mlo[d] <= l && mhi[d] >= h)
}

/// A value strictly inside `(lo[d], hi[d])` at which NO member interval
/// crosses (each lies entirely on one closed side), chosen for balance; the
/// dimension with the best balance wins, ties to the smallest dimension.
fn clean_cut(members: &[&BoxMember], lo: &[f64], hi: &[f64]) -> Option<(usize, f64)> {
    let mut best: Option<(usize, usize, f64)> = None;
    for d in 0..lo.len() {
        if lo[d].partial_cmp(&hi[d]) != Some(std::cmp::Ordering::Less) {
            continue;
        }
        let mut spans: Vec<(f64, f64)> = members
            .iter()
            .map(|m| (m.lo[d].max(lo[d]), m.hi[d].min(hi[d])))
            .collect();
        spans.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
        let n = spans.len();
        let mut reach = f64::NEG_INFINITY;
        let mut i = 0usize;
        while i < n {
            let v = spans[i].0;
            if i > 0 && v > lo[d] && v < hi[d] && reach <= v {
                let balance = i.min(n - i);
                if best.is_none_or(|(b, _, _)| balance > b) {
                    best = Some((balance, d, v));
                }
            }
            while i < n && spans[i].0 == v {
                reach = reach.max(spans[i].1);
                i += 1;
            }
        }
    }
    best.map(|(_, d, v)| (d, v))
}

/// Any interior member face (members crossing it go to both halves): the
/// dimension with the most interior faces, at their median.
fn any_cut(members: &[&BoxMember], lo: &[f64], hi: &[f64]) -> Option<(usize, f64)> {
    let mut best: Option<(usize, usize, f64)> = None;
    for d in 0..lo.len() {
        if lo[d].partial_cmp(&hi[d]) != Some(std::cmp::Ordering::Less) {
            continue;
        }
        let mut faces: Vec<f64> = members
            .iter()
            .flat_map(|m| [m.lo[d], m.hi[d]])
            .filter(|&v| v > lo[d] && v < hi[d])
            .collect();
        if faces.is_empty() {
            continue;
        }
        faces.sort_by(f64::total_cmp);
        let count = faces.len();
        if best.is_none_or(|(c, _, _)| count > c) {
            best = Some((count, d, faces[count / 2]));
        }
    }
    best.map(|(_, d, v)| (d, v))
}

/// B2.2: decide whether `members` cover the region `[root_lo, root_hi]` (under
/// no Relu literal). Independent code (the producer's box gate is
/// `certify::boxes`; nothing is shared).
///
/// Each step is one of two identities, true for ANY cut the step picks:
/// `[a, b] = [a, v] ∪ [v, b]` for CLOSED intervals, or
/// `cell(P) = cell(P ∧ z ≥ 0) ∪ cell(P ∧ z ≤ 0)`. A cell is accepted only when
/// some member's box contains the cell's box AND every Relu literal of the
/// member is on the cell's path, so acceptance is a true containment of the
/// cell in the member's region. Members whose box meets the cell with zero
/// measure are dropped: a finite union of CLOSED regions that covers a dense
/// subset of a box covers the box, so this costs completeness only on
/// Relu-constrained cells, and never soundness.
#[must_use]
pub fn check_box_cover(
    root_lo: &[f64],
    root_hi: &[f64],
    members: &[BoxMember],
    budget: usize,
) -> BoxCoverResult {
    struct Work {
        lo: Vec<f64>,
        hi: Vec<f64>,
        path: Vec<CoverLit>,
        live: Vec<usize>,
    }
    let dims = root_lo.len();
    let mut stack = vec![Work {
        lo: root_lo.to_vec(),
        hi: root_hi.to_vec(),
        path: Vec::new(),
        live: (0..members.len())
            .filter(|&m| members[m].lo.len() == dims && members[m].hi.len() == dims)
            .collect(),
    }];
    let mut steps = 0usize;
    while let Some(Work { lo, hi, path, live }) = stack.pop() {
        steps = steps.saturating_add(1 + live.len() / 64);
        if steps > budget {
            return BoxCoverResult::Budget;
        }
        let side_on_path = |site: SiteKey| -> Option<bool> {
            path.binary_search_by(|l| l.0.cmp(&site))
                .ok()
                .map(|i| path[i].1)
        };
        let live: Vec<usize> = live
            .into_iter()
            .filter(|&m| {
                let mb = &members[m];
                meets(&mb.lo, &mb.hi, &lo, &hi)
                    && mb
                        .relu
                        .iter()
                        .all(|&(site, side)| side_on_path(site).is_none_or(|p| p == side))
            })
            .collect();
        if live.is_empty() {
            return BoxCoverResult::Gap {
                lo: lo.into_boxed_slice(),
                hi: hi.into_boxed_slice(),
                path,
            };
        }
        if live.iter().any(|&m| {
            let mb = &members[m];
            holds_box(&mb.lo, &mb.hi, &lo, &hi)
                && mb
                    .relu
                    .iter()
                    .all(|&(site, _)| side_on_path(site).is_some())
        }) {
            continue;
        }
        let refs: Vec<&BoxMember> = live.iter().map(|&m| &members[m]).collect();
        let geometric = clean_cut(&refs, &lo, &hi);
        let relu_site = if geometric.is_none() {
            let mut counts: BTreeMap<SiteKey, usize> = BTreeMap::new();
            for mb in &refs {
                for &(site, _) in &mb.relu {
                    if side_on_path(site).is_none() {
                        *counts.entry(site).or_insert(0) += 1;
                    }
                }
            }
            let mut best: Option<(SiteKey, usize)> = None;
            for (&site, &c) in &counts {
                if best.is_none_or(|(_, bc)| c > bc) {
                    best = Some((site, c));
                }
            }
            best.map(|(site, _)| site)
        } else {
            None
        };
        if let Some(site) = relu_site {
            for side in [false, true] {
                let mut p = path.clone();
                let at = p.partition_point(|l| l.0 < site);
                p.insert(at, (site, side));
                stack.push(Work {
                    lo: lo.clone(),
                    hi: hi.clone(),
                    path: p,
                    live: live.clone(),
                });
            }
            continue;
        }
        let Some((d, v)) = geometric.or_else(|| any_cut(&refs, &lo, &hi)) else {
            // Every live member meets the cell with positive measure, none
            // holds it, and none has an interior face or an undecided Relu
            // literal: impossible for a covered cell, so report it.
            return BoxCoverResult::Gap {
                lo: lo.into_boxed_slice(),
                hi: hi.into_boxed_slice(),
                path,
            };
        };
        let mut left_hi = hi.clone();
        left_hi[d] = v;
        let mut right_lo = lo.clone();
        right_lo[d] = v;
        let left: Vec<usize> = live
            .iter()
            .copied()
            .filter(|&m| members[m].lo[d] < v)
            .collect();
        let right: Vec<usize> = live
            .iter()
            .copied()
            .filter(|&m| members[m].hi[d] > v)
            .collect();
        stack.push(Work {
            lo: right_lo,
            hi,
            path: path.clone(),
            live: right,
        });
        stack.push(Work {
            lo,
            hi: left_hi,
            path,
            live: left,
        });
    }
    BoxCoverResult::Covered
}

#[cfg(test)]
mod tests {
    use super::{check_cover, CoverResult, DEFAULT_BUDGET};

    fn s(i: u64) -> (u8, u64, u64) {
        (0, 0, i)
    }

    #[test]
    fn a_split_tree_covers_and_a_hole_does_not() {
        let leaves = vec![
            vec![(s(1), true)],
            vec![(s(1), false), (s(2), true)],
            vec![(s(1), false), (s(2), false)],
        ];
        assert_eq!(check_cover(&leaves, DEFAULT_BUDGET), CoverResult::Covered);
        let hole = vec![vec![(s(1), true)], vec![(s(1), false), (s(2), true)]];
        assert!(matches!(
            check_cover(&hole, DEFAULT_BUDGET),
            CoverResult::Gap(_)
        ));
    }

    fn bx(lo: &[f64], hi: &[f64]) -> super::BoxMember {
        super::BoxMember {
            lo: lo.to_vec(),
            hi: hi.to_vec(),
            relu: Vec::new(),
        }
    }

    /// B2.2: a bisection tree, an arbitrary (non-tree) overlapping tiling and a
    /// degenerate coordinate all cover; a one-ulp sliver, a dropped box and a
    /// box outside the region do not.
    #[test]
    fn box_cover_accepts_tilings_and_refuses_slivers() {
        use super::{check_box_cover, BoxCoverResult};
        let (lo, hi) = ([0.0, 0.0], [1.0, 1.0]);
        let tree = vec![
            bx(&[0.0, 0.0], &[0.5, 0.5]),
            bx(&[0.0, 0.5], &[0.5, 1.0]),
            bx(&[0.5, 0.0], &[1.0, 0.25]),
            bx(&[0.5, 0.25], &[1.0, 1.0]),
        ];
        assert_eq!(
            check_box_cover(&lo, &hi, &tree, DEFAULT_BUDGET),
            BoxCoverResult::Covered
        );
        // Overlapping, non-guillotine pinwheel of four boxes plus the centre.
        let pinwheel = vec![
            bx(&[0.0, 0.0], &[0.75, 0.25]),
            bx(&[0.75, 0.0], &[1.0, 0.75]),
            bx(&[0.25, 0.75], &[1.0, 1.0]),
            bx(&[0.0, 0.25], &[0.25, 1.0]),
            bx(&[0.2, 0.2], &[0.8, 0.8]),
        ];
        assert_eq!(
            check_box_cover(&lo, &hi, &pinwheel, DEFAULT_BUDGET),
            BoxCoverResult::Covered
        );
        // A one-ulp sliver between two halves.
        let sliver = vec![
            bx(&[0.0, 0.0], &[0.5, 1.0]),
            bx(&[f64::from_bits(0.5f64.to_bits() + 1), 0.0], &[1.0, 1.0]),
        ];
        assert!(matches!(
            check_box_cover(&lo, &hi, &sliver, DEFAULT_BUDGET),
            BoxCoverResult::Gap { .. }
        ));
        // A dropped box.
        assert!(matches!(
            check_box_cover(&lo, &hi, &tree[..3], DEFAULT_BUDGET),
            BoxCoverResult::Gap { .. }
        ));
        // A degenerate region coordinate is covered by a member containing it.
        let flat = vec![bx(&[0.0, 0.25], &[0.5, 0.25]), bx(&[0.5, 0.0], &[1.0, 1.0])];
        assert_eq!(
            check_box_cover(&[0.0, 0.25], &[1.0, 0.25], &flat, DEFAULT_BUDGET),
            BoxCoverResult::Covered
        );
        let off = vec![bx(&[0.0, 0.0], &[1.0, 0.2])];
        assert!(matches!(
            check_box_cover(&[0.0, 0.25], &[1.0, 0.25], &off, DEFAULT_BUDGET),
            BoxCoverResult::Gap { .. }
        ));
    }

    /// B2.2: a member carrying Relu literals covers only its literal's cell;
    /// both halves of a site are needed, on top of the geometric cover.
    #[test]
    fn box_cover_splits_relu_literals_inside_boxes() {
        use super::{check_box_cover, BoxCoverResult, BoxMember};
        let site = (0u8, 0u64, 3u64);
        let half = |pos: bool, lo: f64, hi: f64| BoxMember {
            lo: vec![lo],
            hi: vec![hi],
            relu: vec![(site, pos)],
        };
        let full = vec![
            half(true, 0.0, 1.0),
            half(false, 0.0, 0.5),
            half(false, 0.5, 1.0),
        ];
        assert_eq!(
            check_box_cover(&[0.0], &[1.0], &full, DEFAULT_BUDGET),
            BoxCoverResult::Covered
        );
        let hole = vec![half(true, 0.0, 1.0), half(false, 0.0, 0.5)];
        match check_box_cover(&[0.0], &[1.0], &hole, DEFAULT_BUDGET) {
            BoxCoverResult::Gap { lo, hi, path } => {
                assert_eq!((lo.to_vec(), hi.to_vec()), (vec![0.5], vec![1.0]));
                assert_eq!(path, vec![(site, false)]);
            }
            other => panic!("expected a gap, got {other:?}"),
        }
    }
}
