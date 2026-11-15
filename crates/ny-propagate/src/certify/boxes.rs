// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The BOX cover gate for input-split trees (U5 of ENGINE_PROGRAM §1.6, the
//! graph lane's input-split route).
//!
//! # The claim
//!
//! An input-split tree closes leaves that are axis-aligned BOXES of the input
//! space. `unsat` needs the closed boxes to cover the frame's root box. The
//! gate proves exactly that, with EXACT arithmetic: every test below is a
//! comparison between two `f64` values that are themselves exact images of the
//! lane's binary32 coordinates (`f64::from(f32)` is exact), so there is no
//! rounding anywhere in the check and nothing to charge. There is no midpoint
//! arithmetic: the gate never recomputes a split point, it only reads the
//! boundaries the lane recorded.
//!
//! # The check
//!
//! `covered(B, M)` for a closed box `B` and a set `M` of recorded leaf boxes:
//!
//! * some `L ∈ M` contains `B` (`L.lo ≤ B.lo` and `L.hi ≥ B.hi` in every
//!   dimension) ⇒ covered;
//! * `M` empty ⇒ refused (`Uncovered`);
//! * otherwise pick a dimension `d` and a value `v` strictly inside
//!   `(B.lo[d], B.hi[d])` that is a boundary of some member, and require
//!   `covered(B ∩ {x_d ≤ v}, M_left)` and `covered(B ∩ {x_d ≥ v}, M_right)`.
//!
//! Soundness is the identity `[a, v] ∪ [v, b] = [a, b]` of CLOSED intervals:
//! each accepted box is covered either by one recorded leaf or by the two
//! closed halves, whatever `v` is. `M_left`/`M_right` keep the members whose
//! overlap with the half has positive measure (or contains a degenerate
//! coordinate); that choice affects completeness only, never soundness,
//! because acceptance is always by the containment test above.
//!
//! Overlapping and extra leaves are tolerated (a fact about a box is true
//! whether or not the box was committed to the tree). The cut prefers a value
//! that crosses NO member (a guillotine cut: every member lies on one side),
//! choosing the most balanced one; a genuine bisection tree always has one, so
//! the reconstruction retraces the tree. Only when no such cut exists does it
//! cut through members (which then go to both halves); that always makes
//! progress, because each half has strictly fewer interior boundaries in `d`.
//! Cost is charged to a [`WorkBudget`]; running out is `Budget`, which the
//! graph lane reads as "not checked" (the counted legacy door), never a proof.
//!
//! # Clips
//!
//! The lane's input clip replaces a child box `P` by a smaller `Q ⊆ P` after
//! its certified linear bound excluded `P ∖ Q`. [`BoxLedger::clip`] records the
//! excluded part as at most `2·D` closed SLABS (the standard box-difference
//! decomposition, again comparisons only), each carrying the clip's receipt,
//! so the tree below `Q` plus the slabs covers `P` exactly.

use std::time::Instant;

use super::cover::{CoverRefusal, LeafEvidence, TreeCover, WorkBudget, COVER_WORK_BUDGET};
use super::value::{closed, CertReceipt};

/// One recorded leaf box.
#[derive(Debug, Clone)]
struct BoxLeaf {
    lo: Box<[f64]>,
    hi: Box<[f64]>,
    evidence: LeafEvidence,
}

/// The ledger one input-split frame writes as it closes boxes.
#[derive(Debug)]
pub(crate) struct BoxLedger {
    dims: usize,
    leaves: Vec<BoxLeaf>,
    first_refusal: Option<CoverRefusal>,
    recorded: usize,
}

fn well_formed(lo: &[f64], hi: &[f64], dims: usize) -> bool {
    lo.len() == dims
        && hi.len() == dims
        && lo
            .iter()
            .zip(hi)
            .all(|(l, h)| l.is_finite() && h.is_finite() && l <= h)
}

impl BoxLedger {
    /// An empty ledger over `dims`-dimensional boxes.
    pub(crate) fn new(dims: usize) -> Self {
        Self {
            dims,
            leaves: Vec::new(),
            first_refusal: None,
            recorded: 0,
        }
    }

    /// Dimension count of the frame.
    pub(crate) fn dims(&self) -> usize {
        self.dims
    }

    /// Leaves recorded so far (including refused ones).
    pub(crate) fn recorded(&self) -> usize {
        self.recorded
    }

    /// Coordinates held (the recorder's memory admission reads it).
    pub(crate) fn coordinate_count(&self) -> usize {
        self.leaves
            .len()
            .saturating_mul(2)
            .saturating_mul(self.dims)
    }

    fn refuse(&mut self, refusal: CoverRefusal) {
        if self.first_refusal.is_none() {
            self.first_refusal = Some(refusal);
        }
    }

    /// Record one closed box. A malformed box or missing evidence is kept as
    /// the ledger's first refusal: the lane CLOSED that region, so the gate
    /// must not pretend the tree is merely incomplete.
    pub(crate) fn leaf(&mut self, lo: &[f64], hi: &[f64], evidence: Option<LeafEvidence>) {
        let ordinal = self.recorded;
        self.recorded += 1;
        if !well_formed(lo, hi, self.dims) {
            self.refuse(CoverRefusal::BadBox { leaf: ordinal });
            return;
        }
        let Some(evidence) = evidence else {
            self.refuse(CoverRefusal::UncertifiedLeaf { leaf: ordinal });
            return;
        };
        self.leaves.push(BoxLeaf {
            lo: lo.into(),
            hi: hi.into(),
            evidence,
        });
    }

    /// Record a clip of box `pre` to box `post`: the part of `pre` outside
    /// `post` is closed by `evidence`, as at most `2·dims` slabs. Returns the
    /// number of slabs recorded. A `post` that is empty in some dimension
    /// (`post.lo > post.hi`) closes all of `pre`; a `post` reaching outside
    /// `pre` is clamped (the outside part is not `pre`'s concern).
    pub(crate) fn clip(
        &mut self,
        pre_lo: &[f64],
        pre_hi: &[f64],
        post_lo: &[f64],
        post_hi: &[f64],
        evidence: &LeafEvidence,
    ) -> usize {
        let finite = |v: &[f64]| v.len() == self.dims && v.iter().all(|x| x.is_finite());
        if !well_formed(pre_lo, pre_hi, self.dims) || !finite(post_lo) || !finite(post_hi) {
            let ordinal = self.recorded;
            self.recorded += 1;
            self.refuse(CoverRefusal::BadBox { leaf: ordinal });
            return 0;
        }
        let mut lo: Vec<f64> = pre_lo.to_vec();
        let mut hi: Vec<f64> = pre_hi.to_vec();
        let mut slabs = 0usize;
        for d in 0..self.dims {
            // Invariant: `pre` = (recorded slabs) ∪ [lo, hi], closed boxes
            // sharing faces; `lo[d] <= hi[d]` throughout.
            if post_lo[d] > lo[d] {
                let cut = post_lo[d].min(hi[d]);
                let mut slab_hi = hi.clone();
                slab_hi[d] = cut;
                self.leaf(&lo, &slab_hi, Some(evidence.clone()));
                slabs += 1;
                lo[d] = cut;
            }
            if post_hi[d] < hi[d] {
                let cut = post_hi[d].max(lo[d]);
                let mut slab_lo = lo.clone();
                slab_lo[d] = cut;
                self.leaf(&slab_lo, &hi, Some(evidence.clone()));
                slabs += 1;
                hi[d] = cut;
            }
        }
        slabs
    }

    /// U5 for an input-split frame: the certified boxes cover the root box.
    /// `classes` is what the cover is claimed for (the frame's objectives).
    pub(crate) fn prove(
        &self,
        root_lo: &[f64],
        root_hi: &[f64],
        classes: &[usize],
        deadline: Option<Instant>,
    ) -> Result<TreeCover, CoverRefusal> {
        self.prove_within(root_lo, root_hi, classes, COVER_WORK_BUDGET, deadline)
    }

    /// [`Self::prove`] with an explicit work budget.
    pub(crate) fn prove_within(
        &self,
        root_lo: &[f64],
        root_hi: &[f64],
        classes: &[usize],
        budget: usize,
        deadline: Option<Instant>,
    ) -> Result<TreeCover, CoverRefusal> {
        if let Some(refusal) = &self.first_refusal {
            return Err(refusal.clone());
        }
        if self.leaves.is_empty() {
            return Err(CoverRefusal::NoLeaves);
        }
        if !well_formed(root_lo, root_hi, self.dims) {
            return Err(CoverRefusal::BadBox { leaf: usize::MAX });
        }
        let mut min_margin = f64::INFINITY;
        let mut lane_leaves = 0usize;
        for (leaf, b) in self.leaves.iter().enumerate() {
            match &b.evidence {
                LeafEvidence::Bound(c) => {
                    if c.receipt() != CertReceipt::GraphLaneBound || !closed(c) {
                        return Err(CoverRefusal::UncertifiedLeaf { leaf });
                    }
                    min_margin = min_margin.min(c.value());
                }
                LeafEvidence::LaneClose(
                    CertReceipt::GraphLaneClip
                    | CertReceipt::GraphLaneInfeasible
                    | CertReceipt::GraphLaneScreen,
                ) => lane_leaves += 1,
                LeafEvidence::LaneClose(_) | LeafEvidence::Empty(_) => {
                    return Err(CoverRefusal::UncertifiedLeaf { leaf });
                }
                LeafEvidence::Subtree(_) => return Err(CoverRefusal::SubtreeMismatch { leaf }),
            }
        }
        let boxes: Vec<(&[f64], &[f64])> =
            self.leaves.iter().map(|b| (&b.lo[..], &b.hi[..])).collect();
        let max_depth = check_box_cover(
            &boxes,
            root_lo,
            root_hi,
            &mut WorkBudget::new(budget, deadline),
        )?;
        Ok(TreeCover::checked(
            classes,
            self.leaves.len(),
            max_depth,
            min_margin,
            0,
            lane_leaves,
        ))
    }
}

/// Whether member `m` overlaps box `[lo, hi]` with positive measure in every
/// dimension where the box has positive width, and contains the box's
/// coordinate in every degenerate dimension.
fn relevant(m: (&[f64], &[f64]), lo: &[f64], hi: &[f64]) -> bool {
    (0..lo.len()).all(|d| {
        if lo[d] < hi[d] {
            m.0[d] < hi[d] && m.1[d] > lo[d]
        } else {
            m.0[d] <= lo[d] && m.1[d] >= hi[d]
        }
    })
}

fn contains(m: (&[f64], &[f64]), lo: &[f64], hi: &[f64]) -> bool {
    (0..lo.len()).all(|d| m.0[d] <= lo[d] && m.1[d] >= hi[d])
}

/// A cut `x_d = v` of a sub-box.
#[derive(Debug, Clone, Copy)]
struct Cut {
    dim: usize,
    value: f64,
}

/// The most balanced cut that crosses no member (every member lies entirely
/// on one closed side), or `None`.
fn guillotine_cut(
    boxes: &[(&[f64], &[f64])],
    members: &[usize],
    lo: &[f64],
    hi: &[f64],
    budget: &mut WorkBudget,
) -> Result<Option<Cut>, CoverRefusal> {
    let mut best: Option<(usize, Cut)> = None;
    let n = members.len();
    for d in 0..lo.len() {
        if lo[d] >= hi[d] {
            continue;
        }
        // Skip a dimension in which no member has an interior boundary.
        budget.charge(n)?;
        if !members.iter().any(|&m| {
            let (l, h) = (boxes[m].0[d], boxes[m].1[d]);
            (l > lo[d] && l < hi[d]) || (h > lo[d] && h < hi[d])
        }) {
            continue;
        }
        let mut order: Vec<(f64, f64)> = members
            .iter()
            .map(|&m| (boxes[m].0[d].max(lo[d]), boxes[m].1[d].min(hi[d])))
            .collect();
        budget.charge(n.saturating_mul(usize::BITS as usize))?;
        order.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut max_hi = f64::NEG_INFINITY;
        let mut i = 0usize;
        while i < n {
            let v = order[i].0;
            // `order[..i]` are the members with lo < v.
            if i > 0 && v > lo[d] && v < hi[d] && max_hi <= v {
                let balance = i.min(n - i);
                if best.is_none_or(|(b, _)| balance > b) {
                    best = Some((balance, Cut { dim: d, value: v }));
                }
            }
            // Consume every member that starts at `v`.
            while i < n && order[i].0 == v {
                max_hi = max_hi.max(order[i].1);
                i += 1;
            }
        }
    }
    Ok(best.map(|(_, cut)| cut))
}

/// Any cut at an interior member boundary (members crossing it go to both
/// halves): the dimension with the most interior boundaries, at their median.
fn crossing_cut(
    boxes: &[(&[f64], &[f64])],
    members: &[usize],
    lo: &[f64],
    hi: &[f64],
    budget: &mut WorkBudget,
) -> Result<Option<Cut>, CoverRefusal> {
    let mut best: Option<(usize, Cut)> = None;
    for d in 0..lo.len() {
        if lo[d] >= hi[d] {
            continue;
        }
        budget.charge(members.len())?;
        let mut inner: Vec<f64> = Vec::new();
        for &m in members {
            for v in [boxes[m].0[d], boxes[m].1[d]] {
                if v > lo[d] && v < hi[d] {
                    inner.push(v);
                }
            }
        }
        if inner.is_empty() {
            continue;
        }
        inner.sort_by(f64::total_cmp);
        let count = inner.len();
        if best.is_none_or(|(c, _)| count > c) {
            best = Some((
                count,
                Cut {
                    dim: d,
                    value: inner[count / 2],
                },
            ));
        }
    }
    Ok(best.map(|(_, cut)| cut))
}

/// The recursive cover reconstruction (see the module docs), iterative.
/// Returns the deepest cut level reached.
fn check_box_cover(
    boxes: &[(&[f64], &[f64])],
    root_lo: &[f64],
    root_hi: &[f64],
    budget: &mut WorkBudget,
) -> Result<usize, CoverRefusal> {
    struct Work {
        lo: Vec<f64>,
        hi: Vec<f64>,
        members: Vec<usize>,
        depth: usize,
    }
    let dims = root_lo.len();
    budget.charge(boxes.len().saturating_mul(dims))?;
    let members: Vec<usize> = (0..boxes.len())
        .filter(|&m| relevant(boxes[m], root_lo, root_hi))
        .collect();
    let mut stack = vec![Work {
        lo: root_lo.to_vec(),
        hi: root_hi.to_vec(),
        members,
        depth: 0,
    }];
    let mut max_depth = 0usize;
    while let Some(Work {
        lo,
        hi,
        members,
        depth,
    }) = stack.pop()
    {
        max_depth = max_depth.max(depth);
        budget.charge(members.len().saturating_mul(dims))?;
        if members.is_empty() {
            return Err(CoverRefusal::Uncovered { depth });
        }
        if members.iter().any(|&m| contains(boxes[m], &lo, &hi)) {
            continue;
        }
        let cut = match guillotine_cut(boxes, &members, &lo, &hi, budget)? {
            Some(cut) => cut,
            None => match crossing_cut(boxes, &members, &lo, &hi, budget)? {
                Some(cut) => cut,
                // Every member overlaps with positive measure and none
                // contains the box, so some member has an interior boundary;
                // reaching here means the box is not covered.
                None => return Err(CoverRefusal::Uncovered { depth }),
            },
        };
        let Cut { dim, value } = cut;
        let mut left_hi = hi.clone();
        left_hi[dim] = value;
        let mut right_lo = lo.clone();
        right_lo[dim] = value;
        let left: Vec<usize> = members
            .iter()
            .copied()
            .filter(|&m| boxes[m].0[dim] < value)
            .collect();
        let right: Vec<usize> = members
            .iter()
            .copied()
            .filter(|&m| boxes[m].1[dim] > value)
            .collect();
        stack.push(Work {
            lo: right_lo,
            hi,
            members: right,
            depth: depth + 1,
        });
        stack.push(Work {
            lo,
            hi: left_hi,
            members: left,
            depth: depth + 1,
        });
    }
    Ok(max_depth)
}
