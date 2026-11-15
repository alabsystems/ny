// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The PATH-BASED cover gate (JS F1; U5 of ENGINE_PROGRAM §1.6).
//!
//! # Why paths and not depths
//!
//! The margin-row ledger this replaces checked the exact-integer Kraft equality
//! `Σ 2^(D - d_i) = 2^D` over the recorded leaf DEPTHS. That is a depth
//! multiset: it passes a duplicated leaf plus a missing sibling at the same
//! depth, and it passes siblings that split DIFFERENT sites — leaves `(n5 off)`
//! and `(n7 on)` sum to 1 and leave `n5 on ∧ n7 off` uncovered. It also never
//! gated: `Unsat` was returned whatever it said.
//!
//! # What is checked here
//!
//! Every certified leaf carries the set of split LITERALS on its path below the
//! frame root — `(site, side)`, with the side being one closed half-space of a
//! ReLU or head pre-activation. The gate proves that the leaves' regions cover
//! the frame root by rebuilding a binary case split over them:
//!
//! * a sub-problem is a set of leaves that all contain the literals consumed so
//!   far;
//! * if exactly one leaf has no literal left, it covers the sub-region;
//!   more than one such leaf, or such a leaf beside others, is a duplicate or an
//!   overlap and is refused (prefix-freeness);
//! * otherwise some site must occur in EVERY leaf of the sub-problem (the
//!   sub-tree's own split); the leaves are partitioned by its side and BOTH
//!   halves must be covered recursively. No common site, or an empty half, is
//!   refused.
//!
//! Soundness does not depend on which common site is chosen, nor on the order
//! the tree was built in: each step is the identity
//! `region(S ∧ s≥0) ∪ region(S ∧ s≤0) = region(S)`, true for any site. For a
//! genuine split tree any choice succeeds (every root-to-leaf path through a
//! site that is on all of them can be rotated to the top), so the gate refuses
//! only enumerations that do not cover. Kraft = 1 is implied.
//!
//! Complexity: `O(L · d · log d)` per level for `L` leaves of depth `≤ d`,
//! index vectors only — small next to the search that produced the leaves.
//!
//! # Two policies: a partition (margin-row) and a cover (graph lane)
//!
//! The margin-row lane records exactly the leaves its enumeration closed, so
//! its gate is the strict PARTITION check above (a duplicate or an overlap is a
//! bookkeeping defect and is refused). The graph (β-CROWN) lane records
//! FACTS where they are established — a certified bound on a domain, a split
//! child whose region the lane's certified node bounds show EMPTY — and some of
//! those facts belong to children a retried or speculative evaluation built
//! and then discarded. A fact about a region is true whether or not the region
//! was committed to the tree, so for that lane the gate is an OVERLAP-TOLERANT
//! COVER check ([`CoverPolicy::GraphLane`]): a Shannon expansion over the
//! recorded literals that accepts iff every total assignment of the split sites
//! is consistent with some certified leaf. Extra or overlapping facts can never
//! make it accept a region nobody closed: the only way to accept is to reach,
//! on every branch, a leaf ALL of whose literals hold. Each expansion step is
//! the same identity `region(S ∧ s≥0) ∪ region(S ∧ s≤0) = region(S)`, and a
//! leaf's literals are real half-spaces of fixed functions of the input, so a
//! Boolean cover is a real cover. Its cost is bounded by a work budget; running
//! out is [`CoverRefusal::Budget`], which the graph lane reads as "the ledger
//! could not be checked" (legacy door, counted), never as a proof.

use std::fmt;
use std::time::Instant;

use super::sink::{LeafClose, LeafRecord};
use super::token::{TokenScope, UnsatToken};
use super::value::{closed, CertReceipt, Certified};

/// One split site of a tree. Ordering is total and arbitrary but fixed; the
/// gate never depends on it for soundness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CoverSite {
    /// A trunk ReLU pre-activation: `(unstable-list layer, position in that
    /// layer's unstable list)` in the frame root's numbering.
    Trunk {
        /// Layer index among the frame's trunk ReLU layers.
        layer: u32,
        /// Position in that layer's unstable list.
        pos: u32,
    },
    /// A head pre-activation `y_i`.
    Head {
        /// Head neuron index.
        index: u32,
    },
    /// A graph-lane (β-CROWN) ReLU/Sign split at zero: the pre-activation of
    /// neuron `neuron` of the activation node the frame numbered `node`.
    GraphRelu {
        /// The frame's id for the split node's name.
        node: u32,
        /// Flat neuron index in the node's output.
        neuron: u32,
    },
    /// A graph-lane GenBaB split of a nonlinear node's pre-activation at an
    /// arbitrary point. The two closed half-lines of ONE point cover the line,
    /// so the point is part of the site: `point_bits` is the split point's
    /// IEEE-754 binary32 encoding, and `input` selects which input of a binary
    /// node the split constrains (`0` for a unary node).
    GraphPoint {
        /// The frame's id for the split node's name.
        node: u32,
        /// Which input of the node the split constrains.
        input: u32,
        /// Flat neuron index.
        neuron: u32,
        /// The split point, as binary32 bits.
        point_bits: u32,
    },
}

/// A split literal: `positive` is the `>= 0` half-space, otherwise `<= 0`.
/// The two halves of one site cover the whole space, which is the only fact
/// the gate uses about them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CoverLiteral {
    /// The site.
    pub site: CoverSite,
    /// Which closed half-space.
    pub positive: bool,
}

/// A leaf's region below the frame root, as its set of split literals.
///
/// Canonical: sorted by site, at most one literal per site. A repeated literal
/// collapses; the same site on both sides is refused at construction (the
/// enumeration never splits a fixed site, so that is a bookkeeping defect).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LeafPath {
    lits: Box<[CoverLiteral]>,
}

impl LeafPath {
    /// Canonicalize a literal list. `None` when one site carries both sides.
    #[must_use]
    pub fn from_literals(mut lits: Vec<CoverLiteral>) -> Option<Self> {
        lits.sort_unstable();
        lits.dedup();
        if lits.windows(2).any(|w| w[0].site == w[1].site) {
            return None;
        }
        Some(Self {
            lits: lits.into_boxed_slice(),
        })
    }

    /// The frame root itself (no literal).
    #[must_use]
    pub fn root() -> Self {
        Self { lits: Box::new([]) }
    }

    /// Number of literals.
    #[must_use]
    pub fn depth(&self) -> usize {
        self.lits.len()
    }

    /// The literals, sorted by site.
    #[must_use]
    pub fn literals(&self) -> &[CoverLiteral] {
        &self.lits
    }

    fn side_of(&self, site: CoverSite) -> Option<bool> {
        self.lits
            .binary_search_by(|l| l.site.cmp(&site))
            .ok()
            .map(|i| self.lits[i].positive)
    }
}

/// Why a leaf is closed.
#[derive(Debug, Clone, PartialEq)]
pub enum LeafEvidence {
    /// The leaf's certified margin lower bound, strictly positive.
    Bound(Certified<f64>),
    /// The leaf's region is certified EMPTY (a strictly positive emptiness
    /// slack between certified outward bounds). An error is never a close.
    Empty(Certified<f64>),
    /// A nested frame (an epoch subtree) authorized its own root — exactly this
    /// leaf's region — for every class of this frame.
    Subtree(UnsatToken),
    /// A close the graph lane decided by a certificate that carries no numeric
    /// margin, named by its receipt: the lane's certified node bounds contradict
    /// the leaf's split constraints ([`CertReceipt::GraphLaneInfeasible`]), an
    /// input clip excluded the leaf's region with the lane's certified linear
    /// bound ([`CertReceipt::GraphLaneClip`]), or the lane's batched IBP
    /// prescreen verified the leaf ([`CertReceipt::GraphLaneScreen`]). Accepted only under
    /// [`CoverPolicy::GraphLane`], and counted separately
    /// ([`TreeCover::lane_leaves`]) so a proof resting on them is visible.
    LaneClose(CertReceipt),
}

/// Why the gate refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoverRefusal {
    /// No leaf was recorded.
    NoLeaves,
    /// A leaf's path could not be built (a site outside the encodable range,
    /// a head prefix that is not the frame root's, or one site on both sides).
    BadPath {
        /// Leaf ordinal.
        leaf: usize,
    },
    /// A leaf's evidence is missing or is not a certified close.
    UncertifiedLeaf {
        /// Leaf ordinal.
        leaf: usize,
    },
    /// A subtree leaf's token is not a sub-domain proof for every class of
    /// this frame.
    SubtreeMismatch {
        /// Leaf ordinal.
        leaf: usize,
    },
    /// A duplicate leaf, or a leaf that also has descendants.
    Overlap {
        /// Split depth of the sub-problem.
        depth: usize,
    },
    /// A sub-region no leaf covers.
    Uncovered {
        /// Split depth of the sub-problem.
        depth: usize,
    },
    /// The leaves of a sub-problem share no split site (e.g. siblings that
    /// split different neurons).
    NoCommonSplit {
        /// Split depth of the sub-problem.
        depth: usize,
        /// Leaves in the sub-problem.
        leaves: usize,
    },
    /// The check ran out of its work budget or its deadline before deciding.
    /// NOT a finding about the tree: the graph lane reads it as "the ledger
    /// could not be checked" and publishes through the counted legacy door.
    Budget {
        /// Work units spent when it stopped.
        work: usize,
    },
    /// A box leaf's dimension count differs from the frame root's, or a box
    /// coordinate is non-finite or inverted.
    BadBox {
        /// Leaf ordinal.
        leaf: usize,
    },
}

impl fmt::Display for CoverRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoLeaves => f.write_str("no_leaves"),
            Self::BadPath { leaf } => write!(f, "bad_path leaf={leaf}"),
            Self::UncertifiedLeaf { leaf } => write!(f, "uncertified_leaf leaf={leaf}"),
            Self::SubtreeMismatch { leaf } => write!(f, "subtree_mismatch leaf={leaf}"),
            Self::Overlap { depth } => write!(f, "overlap depth={depth}"),
            Self::Uncovered { depth } => write!(f, "uncovered depth={depth}"),
            Self::NoCommonSplit { depth, leaves } => {
                write!(f, "no_common_split depth={depth} leaves={leaves}")
            }
            Self::Budget { work } => write!(f, "budget work={work}"),
            Self::BadBox { leaf } => write!(f, "bad_box leaf={leaf}"),
        }
    }
}

impl std::error::Error for CoverRefusal {}

/// Which gate a ledger runs (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub(crate) enum CoverPolicy {
    /// The margin-row lane: a strict PARTITION of margin-row certified leaves
    /// (bound, emptiness, nested subtree). The historical gate, unchanged.
    #[default]
    MarginRow,
    /// The graph (β-CROWN) lane: an overlap-tolerant COVER of graph-lane
    /// certified leaves (bound, lane close). No nested subtree tokens.
    GraphLane,
}

impl CoverPolicy {
    /// The receipt a bound leaf must carry under this policy.
    const fn bound_receipt(self) -> CertReceipt {
        match self {
            Self::MarginRow => CertReceipt::MarginRowOutward,
            Self::GraphLane => CertReceipt::GraphLaneBound,
        }
    }

    /// The receipt an emptiness leaf must carry under this policy: the
    /// margin-row lane's certified emptiness slack, or (#vit-v3-bankable) the
    /// graph lane's certified emptiness slack. Each lane's slack is accepted
    /// only by its own gate, as with bound leaves.
    const fn emptiness_receipt(self) -> CertReceipt {
        match self {
            Self::MarginRow => CertReceipt::MarginRowEmptiness,
            Self::GraphLane => CertReceipt::GraphLaneEmptiness,
        }
    }

    /// Whether a margin-free lane close with receipt `r` is evidence here.
    fn accepts_lane_close(self, r: CertReceipt) -> bool {
        matches!(
            (self, r),
            (
                Self::GraphLane,
                CertReceipt::GraphLaneInfeasible
                    | CertReceipt::GraphLaneClip
                    | CertReceipt::GraphLaneScreen
            )
        )
    }
}

/// Default work budget of one overlap-tolerant cover check, in member visits.
pub(crate) const COVER_WORK_BUDGET: usize = 200_000_000;

/// How often (in work units) a budgeted check consults its deadline.
const DEADLINE_POLL_WORK: usize = 1 << 16;

/// A work counter with a budget and an optional deadline, shared by the
/// overlap-tolerant path cover and the box cover.
pub(crate) struct WorkBudget {
    spent: usize,
    budget: usize,
    deadline: Option<Instant>,
    next_poll: usize,
}

impl WorkBudget {
    /// A budget of `budget` work units, stopping early at `deadline`.
    pub(crate) fn new(budget: usize, deadline: Option<Instant>) -> Self {
        Self {
            spent: 0,
            budget,
            deadline,
            next_poll: DEADLINE_POLL_WORK,
        }
    }

    /// Charge `units`; `Err(Budget)` once the budget or the deadline is spent.
    pub(crate) fn charge(&mut self, units: usize) -> Result<(), CoverRefusal> {
        self.spent = self.spent.saturating_add(units.max(1));
        if self.spent > self.budget {
            return Err(CoverRefusal::Budget { work: self.spent });
        }
        if self.spent >= self.next_poll {
            self.next_poll = self.spent.saturating_add(DEADLINE_POLL_WORK);
            if self.deadline.is_some_and(|d| Instant::now() >= d) {
                return Err(CoverRefusal::Budget { work: self.spent });
            }
        }
        Ok(())
    }
}

/// A successful cover proof for a set of classes. Private fields, and only the
/// cover gates build one: [`CoverLedger::prove`] (paths) and the box gate
/// (`certify::boxes`, input-split trees), through [`TreeCover::checked`].
#[derive(Debug, Clone, PartialEq)]
pub struct TreeCover {
    classes: Box<[usize]>,
    leaves: usize,
    max_depth: usize,
    min_margin: f64,
    empty_leaves: usize,
    subtree_leaves: usize,
    lane_leaves: usize,
}

impl TreeCover {
    /// Assemble a cover a gate in this module has just CHECKED. `pub(super)`:
    /// only `certify`'s own gates reach it, after their check succeeded.
    pub(super) fn checked(
        classes: &[usize],
        leaves: usize,
        max_depth: usize,
        min_margin: f64,
        empty_leaves: usize,
        lane_leaves: usize,
    ) -> Self {
        let mut sorted: Vec<usize> = classes.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        Self {
            classes: sorted.into_boxed_slice(),
            leaves,
            max_depth,
            min_margin,
            empty_leaves,
            subtree_leaves: 0,
            lane_leaves,
        }
    }

    /// The same checked cover, claimed for `classes` instead. `pub(crate)`
    /// and used only by the graph lane's instance aggregation
    /// (`certify::graph`), which proved that a ledgered invocation's region and
    /// objective entail each class it relabels to.
    pub(crate) fn relabeled(&self, classes: &[usize]) -> Self {
        let mut sorted: Vec<usize> = classes.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        Self {
            classes: sorted.into_boxed_slice(),
            ..self.clone()
        }
    }

    /// Leaves closed by a margin-free graph-lane certificate (an inverted
    /// certified interval, an input clip, or the batched IBP prescreen).
    #[must_use]
    pub fn lane_leaves(&self) -> usize {
        self.lane_leaves
    }

    /// The classes every leaf certifies.
    #[must_use]
    pub fn classes(&self) -> &[usize] {
        &self.classes
    }

    /// Leaves in the cover.
    #[must_use]
    pub fn leaves(&self) -> usize {
        self.leaves
    }

    /// Deepest leaf.
    #[must_use]
    pub fn max_depth(&self) -> usize {
        self.max_depth
    }

    /// Smallest certified margin over bound leaves and subtree tokens
    /// (`+inf` when every leaf is an emptiness close).
    #[must_use]
    pub fn min_margin(&self) -> f64 {
        self.min_margin
    }

    /// Leaves closed by a certified emptiness slack.
    #[must_use]
    pub fn empty_leaves(&self) -> usize {
        self.empty_leaves
    }

    /// Leaves closed by a nested subtree token.
    #[must_use]
    pub fn subtree_leaves(&self) -> usize {
        self.subtree_leaves
    }
}

/// The ledger one tree frame writes as it closes leaves.
#[derive(Debug, Default)]
pub struct CoverLedger {
    leaves: Vec<(LeafPath, LeafEvidence)>,
    first_refusal: Option<CoverRefusal>,
    recorded: usize,
    policy: CoverPolicy,
}

impl CoverLedger {
    /// An empty ledger (the margin-row PARTITION policy).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// An empty graph-lane ledger (the overlap-tolerant COVER policy; see the
    /// module docs). `pub(crate)`: only the graph lane's recorder builds one.
    #[must_use]
    pub(crate) fn for_graph_lane() -> Self {
        Self {
            policy: CoverPolicy::GraphLane,
            ..Self::default()
        }
    }

    /// Record one closed leaf. A missing path or missing evidence is kept as
    /// the ledger's first refusal: the leaf WAS closed by the search, so the
    /// gate must not later pretend the tree is merely incomplete.
    pub fn leaf(&mut self, path: Option<LeafPath>, evidence: Option<LeafEvidence>) {
        let ordinal = self.recorded;
        self.recorded += 1;
        let refusal = match (path, evidence) {
            (Some(path), Some(evidence)) => {
                self.leaves.push((path, evidence));
                return;
            }
            (None, _) => CoverRefusal::BadPath { leaf: ordinal },
            (Some(_), None) => CoverRefusal::UncertifiedLeaf { leaf: ordinal },
        };
        if self.first_refusal.is_none() {
            self.first_refusal = Some(refusal);
        }
    }

    /// Leaves recorded so far (including refused ones).
    #[must_use]
    pub fn recorded(&self) -> usize {
        self.recorded
    }

    /// The recorded leaves as proof HINTS (path, close kind, certified
    /// margin), in recording order, for a write-only [`super::ProofSink`].
    /// Nothing the gate decides reads these records back.
    pub fn records(&self) -> impl Iterator<Item = LeafRecord> + '_ {
        self.leaves.iter().map(|(path, evidence)| {
            let (close, margin) = match evidence {
                LeafEvidence::Bound(c) => (LeafClose::Bound, c.value()),
                LeafEvidence::Empty(c) => (LeafClose::Empty, c.value()),
                LeafEvidence::Subtree(token) => (LeafClose::Subtree, token.min_margin()),
                LeafEvidence::LaneClose(_) => (LeafClose::Lane, f64::NAN),
            };
            LeafRecord {
                path: path.clone(),
                margin,
                close,
            }
        })
    }

    /// Certified margin values of every bound leaf, for the G5 histogram.
    pub fn bound_margins(&self) -> impl Iterator<Item = f64> + '_ {
        self.leaves.iter().filter_map(|(_, ev)| match ev {
            LeafEvidence::Bound(c) => Some(c.value()),
            LeafEvidence::Empty(_) | LeafEvidence::Subtree(_) | LeafEvidence::LaneClose(_) => None,
        })
    }

    /// The legacy depth-multiset Kraft equality `Σ 2^(D-d) = 2^D`, kept as an
    /// observable. NOT a cover certificate (see the module docs); the gate is
    /// [`Self::prove`].
    #[must_use]
    pub fn kraft_ok(&self) -> bool {
        if self.first_refusal.is_some() || self.leaves.is_empty() {
            return false;
        }
        let Some(dmax) = self.leaves.iter().map(|(p, _)| p.depth()).max() else {
            return false;
        };
        if dmax > 120 {
            return false;
        }
        let target: u128 = 1u128 << dmax;
        let mut sum: u128 = 0;
        for (p, _) in &self.leaves {
            let Some(s) = sum.checked_add(1u128 << (dmax - p.depth())) else {
                return false;
            };
            sum = s;
        }
        sum == target
    }

    /// U5: prove that the certified leaves cover the frame root for `classes`.
    ///
    /// Under the margin-row policy this is the strict partition gate; under
    /// the graph-lane policy it is the overlap-tolerant cover gate with the
    /// default work budget and no deadline ([`Self::prove_within`]).
    pub fn prove(&self, classes: &[usize]) -> Result<TreeCover, CoverRefusal> {
        self.prove_within(classes, COVER_WORK_BUDGET, None)
    }

    /// [`Self::prove`] with an explicit work budget and deadline for the
    /// graph-lane cover check (the margin-row partition check ignores both: it
    /// is linear per level and never ran out of anything).
    pub(crate) fn prove_within(
        &self,
        classes: &[usize],
        budget: usize,
        deadline: Option<Instant>,
    ) -> Result<TreeCover, CoverRefusal> {
        self.prove_collecting(classes, budget, deadline, None)
    }

    /// #vit-v3-bankable: [`Self::prove_within`] under the graph-lane policy,
    /// also returning the WITNESSES — the recorded leaves (indices in
    /// recording order of the accepted leaves, see [`Self::witness_path`])
    /// that closed some branch of the cover's Shannon expansion. Every other
    /// recorded fact is a tolerated extra. The decision is exactly
    /// `prove_within`'s (the witness is the member the `covered` test already
    /// found); a certify-once replay re-certifies precisely these leaves.
    /// `None` witnesses for the margin-row partition policy (every leaf is a
    /// witness there).
    pub(crate) fn prove_with_witnesses(
        &self,
        classes: &[usize],
        budget: usize,
        deadline: Option<Instant>,
    ) -> Result<(TreeCover, Vec<usize>), CoverRefusal> {
        let mut witnesses = Vec::new();
        let cover = self.prove_collecting(classes, budget, deadline, Some(&mut witnesses))?;
        witnesses.sort_unstable();
        witnesses.dedup();
        Ok((cover, witnesses))
    }

    /// The path of accepted leaf `index` (a witness index from
    /// [`Self::prove_with_witnesses`]).
    pub(crate) fn witness_path(&self, index: usize) -> Option<&LeafPath> {
        self.leaves.get(index).map(|(path, _)| path)
    }

    /// Whether accepted leaf `index` closed by a certified emptiness slack.
    pub(crate) fn witness_is_empty(&self, index: usize) -> bool {
        matches!(self.leaves.get(index), Some((_, LeafEvidence::Empty(_))))
    }

    fn prove_collecting(
        &self,
        classes: &[usize],
        budget: usize,
        deadline: Option<Instant>,
        witnesses: Option<&mut Vec<usize>>,
    ) -> Result<TreeCover, CoverRefusal> {
        if let Some(refusal) = &self.first_refusal {
            return Err(refusal.clone());
        }
        if self.leaves.is_empty() {
            return Err(CoverRefusal::NoLeaves);
        }
        let policy = self.policy;
        let mut min_margin = f64::INFINITY;
        let mut empty_leaves = 0usize;
        let mut subtree_leaves = 0usize;
        let mut lane_leaves = 0usize;
        for (leaf, (_, evidence)) in self.leaves.iter().enumerate() {
            match evidence {
                LeafEvidence::Bound(c) => {
                    if c.receipt() != policy.bound_receipt() || !closed(c) {
                        return Err(CoverRefusal::UncertifiedLeaf { leaf });
                    }
                    min_margin = min_margin.min(c.value());
                }
                LeafEvidence::Empty(c) => {
                    if c.receipt() != policy.emptiness_receipt() || !closed(c) {
                        return Err(CoverRefusal::UncertifiedLeaf { leaf });
                    }
                    empty_leaves += 1;
                }
                LeafEvidence::Subtree(token) => {
                    if policy != CoverPolicy::MarginRow
                        || token.scope() != TokenScope::Subdomain
                        || !token.covers_all(classes)
                    {
                        return Err(CoverRefusal::SubtreeMismatch { leaf });
                    }
                    let m = token.min_margin();
                    if m.is_nan() || m <= 0.0 {
                        return Err(CoverRefusal::SubtreeMismatch { leaf });
                    }
                    min_margin = min_margin.min(m);
                    subtree_leaves += 1;
                }
                LeafEvidence::LaneClose(receipt) => {
                    if !policy.accepts_lane_close(*receipt) {
                        return Err(CoverRefusal::UncertifiedLeaf { leaf });
                    }
                    lane_leaves += 1;
                }
            }
        }
        let paths: Vec<&LeafPath> = self.leaves.iter().map(|(p, _)| p).collect();
        match policy {
            CoverPolicy::MarginRow => check_partition(&paths)?,
            CoverPolicy::GraphLane => {
                check_cover(&paths, &mut WorkBudget::new(budget, deadline), witnesses)?;
            }
        }
        let mut sorted: Vec<usize> = classes.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        Ok(TreeCover {
            classes: sorted.into_boxed_slice(),
            leaves: paths.len(),
            max_depth: paths.iter().map(|p| p.depth()).max().unwrap_or(0),
            min_margin,
            empty_leaves,
            subtree_leaves,
            lane_leaves,
        })
    }

    /// Whether [`Self::prove`] succeeds (for observables and tests).
    #[must_use]
    pub fn proves_cover(&self, classes: &[usize]) -> bool {
        self.prove(classes).is_ok()
    }
}

// Original-query paid partition recorder. This is an allocation/coverage
// adapter, never a new numerical or token authority. Legacy CoverLedger
// methods and their policy remain unchanged.
#[derive(Debug)]
pub(crate) enum PaidCoverError {
    Resource(super::work::WorkRefusal),
    Cover(CoverRefusal),
    Allocation,
}
impl From<super::work::WorkRefusal> for PaidCoverError {
    fn from(value: super::work::WorkRefusal) -> Self {
        Self::Resource(value)
    }
}
impl From<CoverRefusal> for PaidCoverError {
    fn from(value: CoverRefusal) -> Self {
        Self::Cover(value)
    }
}
impl fmt::Display for PaidCoverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "paid cover: {self:?}")
    }
}
impl std::error::Error for PaidCoverError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Resource(error) => Some(error),
            Self::Cover(error) => Some(error),
            Self::Allocation => None,
        }
    }
}

pub(crate) struct PaidCoverLedger {
    ledger: CoverLedger,
    paths: Vec<super::work::PersistentReservation>,
    path_bytes: usize,
    budget: super::work::WorkBudget,
    deadline: Instant,
    resident: super::work::PersistentReservation,
}
/// The class allocation and original resource authority outlive every borrow
/// of the checked cover. No detached token is constructed by this adapter.
pub(crate) struct PaidTreeCover(super::work::RetainedOwner<PaidTreeCoverData>);
pub(crate) struct PaidTreeCoverData {
    cover: TreeCover,
    budget: super::work::WorkBudget,
    deadline: Instant,
}
impl std::ops::Deref for PaidTreeCover {
    type Target = PaidTreeCoverData;

    fn deref(&self) -> &PaidTreeCoverData {
        &self.0
    }
}
const PAID_COVER_META: usize = 4096;
const PAID_COVER_POLL: usize = 256;
fn paid_add(a: usize, b: usize) -> Result<usize, PaidCoverError> {
    a.checked_add(b)
        .ok_or(super::work::WorkRefusal::Overflow.into())
}
fn paid_mul(a: usize, b: usize) -> Result<usize, PaidCoverError> {
    a.checked_mul(b)
        .ok_or(super::work::WorkRefusal::Overflow.into())
}
fn paid_work(n: usize, factor: u64) -> Result<u64, PaidCoverError> {
    u64::try_from(n)
        .ok()
        .and_then(|n| n.checked_mul(factor))
        .ok_or(super::work::WorkRefusal::Overflow.into())
}
fn paid_checkpoint(budget: &super::work::WorkBudget, at: Instant) -> Result<(), PaidCoverError> {
    budget.checkpoint_for(super::work::LeaseClass::Replay)?;
    if Instant::now() >= at {
        return Err(super::work::WorkRefusal::Deadline.into());
    }
    Ok(())
}
fn paid_poll(
    budget: &super::work::WorkBudget,
    at: Instant,
    i: usize,
) -> Result<(), PaidCoverError> {
    if i.is_multiple_of(PAID_COVER_POLL) {
        paid_checkpoint(budget, at)?;
    }
    Ok(())
}
fn paid_vec<T>(capacity: usize) -> Result<Vec<T>, PaidCoverError> {
    let mut v = Vec::new();
    v.try_reserve_exact(capacity)
        .map_err(|_| PaidCoverError::Allocation)?;
    if v.capacity() > capacity {
        return Err(PaidCoverError::Allocation);
    }
    Ok(v)
}
impl PaidCoverLedger {
    /// Reserve the complete recorder capacity before any leaf insertion.
    /// A binary tree with at most E expansions has at most E+1 leaves.
    /// Capacity is a checked caller bound, not permission to skip a leaf.
    pub(crate) fn new(
        budget: &super::work::WorkBudget,
        capacity: usize,
        requested: Instant,
    ) -> Result<Self, PaidCoverError> {
        use super::work::{LeaseClass, WorkRequest};
        paid_checkpoint(budget, requested)?;
        let bytes = paid_add(
            PAID_COVER_META,
            paid_add(
                size_of::<Self>(),
                paid_mul(
                    capacity,
                    paid_add(
                        size_of::<(LeafPath, LeafEvidence)>(),
                        size_of::<super::work::PersistentReservation>(),
                    )?,
                )?,
            )?,
        )?;
        let mut lease = budget.lease_for(
            LeaseClass::Replay,
            WorkRequest {
                persistent_bytes: bytes,
                work: paid_work(capacity, 4)?,
                ..WorkRequest::default()
            },
        )?;
        let at = lease.deadline().map_or(requested, |d| d.min(requested));
        let leaves = paid_vec(capacity)?;
        let paths = paid_vec(capacity)?;
        paid_checkpoint(budget, at)?;
        lease.mark_materialized()?;
        let resident = lease.commit_persistent()?;
        Ok(Self {
            ledger: CoverLedger {
                leaves,
                ..CoverLedger::new()
            },
            paths,
            path_bytes: 0,
            budget: budget.clone(),
            deadline: at,
            resident,
        })
    }
    pub(crate) fn deadline(&self, requested: Option<Instant>) -> Instant {
        requested.map_or(self.deadline, |d| d.min(self.deadline))
    }
    pub(crate) fn checkpoint(&self, requested: Option<Instant>) -> Result<(), PaidCoverError> {
        if !self.resident.shares_ledger(&self.budget) {
            return Err(super::work::WorkRefusal::Unavailable.into());
        }
        paid_checkpoint(&self.budget, self.deadline(requested))
    }
    pub(crate) fn recorded(&self) -> usize {
        self.ledger.recorded()
    }
    pub(crate) fn capacity(&self) -> usize {
        self.ledger.leaves.capacity().min(self.paths.capacity())
    }
    pub(crate) fn resident_bytes(&self) -> Result<usize, PaidCoverError> {
        paid_add(self.resident.bytes(), self.path_bytes)
    }
    pub(crate) fn ledger(
        &self,
        requested: Option<Instant>,
    ) -> Result<&CoverLedger, PaidCoverError> {
        self.checkpoint(requested)?;
        Ok(&self.ledger)
    }
    /// Clone a genuine producer-owned path only after paying its capacity.
    /// Missing path/evidence is recorded as the same first refusal as legacy.
    /// A resource refusal leaves all earlier recorded leaves unchanged.
    pub(crate) fn leaf(
        &mut self,
        path: Option<&LeafPath>,
        evidence: Option<LeafEvidence>,
        requested: Option<Instant>,
    ) -> Result<(), PaidCoverError> {
        use super::work::{LeaseClass, WorkRequest};
        let at = self.deadline(requested);
        self.checkpoint(Some(at))?;
        let ordinal = self.ledger.recorded;
        let (path, evidence) = match (path, evidence) {
            (Some(p), Some(e)) => (p, e),
            (None, _) => {
                self.ledger.leaf(None, None);
                self.deadline = at;
                return Ok(());
            }
            (Some(_), None) => {
                self.ledger.recorded += 1;
                if self.ledger.first_refusal.is_none() {
                    self.ledger.first_refusal =
                        Some(CoverRefusal::UncertifiedLeaf { leaf: ordinal });
                }
                self.deadline = at;
                return Ok(());
            }
        };
        if self.ledger.leaves.len() >= self.capacity() {
            return Err(super::work::WorkRefusal::PersistentLimit.into());
        }
        let bytes = paid_add(
            PAID_COVER_META,
            paid_mul(path.depth(), size_of::<CoverLiteral>())?,
        )?;
        let mut lease = self.budget.lease_for(
            LeaseClass::Replay,
            WorkRequest {
                persistent_bytes: bytes,
                work: paid_work(path.depth(), 8)?,
                ..WorkRequest::default()
            },
        )?;
        let at = lease.deadline().map_or(at, |d| d.min(at));
        let mut lits = paid_vec(path.depth())?;
        for (i, &lit) in path.literals().iter().enumerate() {
            paid_poll(&self.budget, at, i)?;
            lits.push(lit);
        }
        let path = LeafPath {
            lits: lits.into_boxed_slice(),
        };
        paid_checkpoint(&self.budget, at)?;
        lease.mark_materialized()?;
        let resident = lease.commit_persistent()?;
        let next_bytes = paid_add(self.path_bytes, resident.bytes())?;
        self.ledger.leaves.push((path, evidence));
        self.paths.push(resident);
        if let Err(error) = paid_checkpoint(&self.budget, at) {
            self.ledger.leaves.pop();
            self.paths.pop();
            return Err(error);
        }
        self.ledger.recorded += 1;
        self.path_bytes = next_bytes;
        self.deadline = at;
        Ok(())
    }
    /// Allocation-free undo for a caller's uncommitted batch. The caller
    /// prepays forward/undo metadata work; cancelled rollback must finish.
    /// No work allowance or inherited deadline is refunded.
    pub(crate) fn rollback_to(&mut self, recorded: usize) {
        assert!(self.ledger.first_refusal.is_none());
        assert_eq!(self.ledger.recorded, self.ledger.leaves.len());
        assert!(recorded <= self.ledger.recorded);
        while self.ledger.recorded > recorded {
            self.ledger.leaves.pop();
            let claim = self.paths.pop().expect("paid path/leaf alignment");
            self.path_bytes -= claim.bytes();
            self.ledger.recorded -= 1;
        }
    }
    pub(crate) fn prove(
        &self,
        classes: &[usize],
        requested: Option<Instant>,
    ) -> Result<PaidTreeCover, PaidCoverError> {
        use super::work::{LeaseClass, WorkRequest};
        let at = self.deadline(requested);
        self.checkpoint(Some(at))?;
        if let Some(r) = &self.ledger.first_refusal {
            return Err(r.clone().into());
        }
        let count = self.ledger.leaves.len();
        if count == 0 {
            return Err(CoverRefusal::NoLeaves.into());
        }
        // This enclosing owner replaces the already priced reservation Arc.
        // Check its full aligned header/control bound before any census debit.
        super::work::RetainedOwner::<PaidTreeCoverData>::check_metadata(PAID_COVER_META)?;
        // O(1) count is admitted before the metadata census.
        let mut census = self.budget.lease_for(
            LeaseClass::Replay,
            WorkRequest {
                work: paid_work(count, 16)?,
                ..WorkRequest::default()
            },
        )?;
        let mut depth = 0usize;
        for (i, (p, _)) in self.ledger.leaves.iter().enumerate() {
            paid_poll(&self.budget, at, i)?;
            depth = depth.max(p.depth());
        }
        census.complete()?;
        let d = paid_add(depth, 1)?;
        let scratch = paid_add(
            PAID_COVER_META,
            paid_add(
                paid_mul(count, size_of::<usize>())?,
                paid_add(
                    paid_mul(d, size_of::<PaidPartitionFrame>())?,
                    paid_mul(depth, size_of::<CoverSite>())?,
                )?,
            )?,
        )?;
        let persistent = paid_add(
            PAID_COVER_META,
            paid_add(
                paid_add(size_of::<PaidTreeCover>(), size_of::<PaidTreeCoverData>())?,
                paid_mul(paid_mul(classes.len(), 2)?, size_of::<usize>())?,
            )?,
        )?;
        // At each depth member visits sum to L. At most d candidates per
        // node, each checking d consumed sites and a binary path lookup.
        // The cubic allowance also pays evidence/partition/rollback scans.
        let visits = paid_mul(count, paid_mul(d, paid_mul(d, d)?)?)?;
        let class_work = paid_mul(classes.len(), paid_add(usize::BITS as usize, 1)?)?;
        let mut lease = self.budget.lease_for(
            LeaseClass::Replay,
            WorkRequest {
                scratch_bytes: scratch,
                persistent_bytes: persistent,
                work: paid_work(
                    paid_add(
                        visits,
                        paid_add(class_work, paid_mul(count, paid_add(classes.len(), 1)?)?)?,
                    )?,
                    256,
                )?,
                ..WorkRequest::default()
            },
        )?;
        let at = lease.deadline().map_or(at, |d| d.min(at));
        let mut min_margin = f64::INFINITY;
        let mut empty = 0usize;
        let mut subtree = 0usize;
        for (i, (_, ev)) in self.ledger.leaves.iter().enumerate() {
            paid_poll(&self.budget, at, i)?;
            match ev {
                LeafEvidence::Bound(c)
                    if c.receipt() == CertReceipt::MarginRowOutward && closed(c) =>
                {
                    min_margin = min_margin.min(c.value());
                }
                LeafEvidence::Empty(c)
                    if c.receipt() == CertReceipt::MarginRowEmptiness && closed(c) =>
                {
                    empty += 1;
                }
                LeafEvidence::Subtree(t)
                    if t.scope() == TokenScope::Subdomain && t.min_margin() > 0.0 =>
                {
                    for (j, class) in classes.iter().enumerate() {
                        paid_poll(&self.budget, at, j)?;
                        if t.classes().binary_search(class).is_err() {
                            return Err(CoverRefusal::SubtreeMismatch { leaf: i }.into());
                        }
                    }
                    min_margin = min_margin.min(t.min_margin());
                    subtree += 1;
                }
                LeafEvidence::Subtree(_) => {
                    return Err(CoverRefusal::SubtreeMismatch { leaf: i }.into())
                }
                _ => return Err(CoverRefusal::UncertifiedLeaf { leaf: i }.into()),
            }
        }
        paid_partition(&self.ledger.leaves, depth, &self.budget, at)?;
        // Cooperative in-place heap sort gives O(K log K) for full LLM
        // vocabularies, without an opaque unpolled comparison loop.
        let mut sorted = paid_vec(classes.len())?;
        for (i, &class) in classes.iter().enumerate() {
            paid_poll(&self.budget, at, i)?;
            sorted.push(class);
        }
        paid_sort_classes(&mut sorted, &self.budget, at)?;
        let mut unique = 0usize;
        for i in 0..sorted.len() {
            paid_poll(&self.budget, at, i)?;
            if unique == 0 || sorted[i] != sorted[unique - 1] {
                sorted[unique] = sorted[i];
                unique += 1;
            }
        }
        sorted.truncate(unique);
        // The full old Vec and new Box overlap is priced above, including
        // allocator shrinking after duplicate elimination.
        let cover = TreeCover {
            classes: sorted.into_boxed_slice(),
            leaves: count,
            max_depth: depth,
            min_margin,
            empty_leaves: empty,
            subtree_leaves: subtree,
            lane_leaves: 0,
        };
        paid_checkpoint(&self.budget, at)?;
        lease.mark_materialized()?;
        let result = PaidTreeCover(lease.commit_retained_with(|| PaidTreeCoverData {
            cover,
            budget: self.budget.clone(),
            deadline: at,
        })?);
        result.checkpoint(None)?;
        Ok(result)
    }
}
impl PaidTreeCover {
    pub(crate) fn checkpoint(&self, requested: Option<Instant>) -> Result<(), PaidCoverError> {
        if !self.0.shares_ledger(&self.budget) {
            return Err(super::work::WorkRefusal::Unavailable.into());
        }
        paid_checkpoint(
            &self.budget,
            requested.map_or(self.deadline, |d| d.min(self.deadline)),
        )
    }
    pub(crate) fn cover(&self, requested: Option<Instant>) -> Result<&TreeCover, PaidCoverError> {
        self.checkpoint(requested)?;
        Ok(&self.cover)
    }
    pub(crate) fn deadline(&self) -> Instant {
        self.deadline
    }
    pub(crate) fn resident_bytes(&self) -> usize {
        self.0.bytes()
    }
}
fn paid_sort_classes(
    v: &mut [usize],
    budget: &super::work::WorkBudget,
    at: Instant,
) -> Result<(), PaidCoverError> {
    fn sift(
        v: &mut [usize],
        mut parent: usize,
        end: usize,
        budget: &super::work::WorkBudget,
        at: Instant,
        units: &mut usize,
    ) -> Result<(), PaidCoverError> {
        loop {
            paid_poll(budget, at, *units)?;
            *units = units.wrapping_add(1);
            let Some(left) = parent.checked_mul(2).and_then(|p| p.checked_add(1)) else {
                break;
            };
            if left >= end {
                break;
            }
            let mut child = left;
            if child + 1 < end && v[child] < v[child + 1] {
                child += 1
            }
            if v[parent] >= v[child] {
                break;
            }
            v.swap(parent, child);
            parent = child;
        }
        Ok(())
    }
    let mut units = 0usize;
    for parent in (0..v.len() / 2).rev() {
        let end = v.len();
        sift(v, parent, end, budget, at, &mut units)?;
    }
    for end in (1..v.len()).rev() {
        v.swap(0, end);
        sift(v, 0, end, budget, at, &mut units)?;
    }
    paid_checkpoint(budget, at)
}
#[derive(Clone, Copy)]
struct PaidPartitionFrame {
    lo: usize,
    hi: usize,
    depth: usize,
    stage: u8,
    site: Option<CoverSite>,
    cut: usize,
}
fn paid_partition(
    leaves: &[(LeafPath, LeafEvidence)],
    depth_bound: usize,
    budget: &super::work::WorkBudget,
    at: Instant,
) -> Result<(), PaidCoverError> {
    let mut members = paid_vec(leaves.len())?;
    for i in 0..leaves.len() {
        paid_poll(budget, at, i)?;
        members.push(i);
    }
    let mut consumed = paid_vec(depth_bound)?;
    let mut stack = paid_vec(paid_add(depth_bound, 1)?)?;
    stack.push(PaidPartitionFrame {
        lo: 0,
        hi: members.len(),
        depth: 0,
        stage: 0,
        site: None,
        cut: 0,
    });
    let mut units = 0usize;
    while let Some(frame) = stack.last().copied() {
        paid_poll(budget, at, units)?;
        units = units.wrapping_add(1);
        if frame.stage == 1 {
            stack.last_mut().unwrap().stage = 2;
            consumed.truncate(frame.depth);
            consumed.push(frame.site.unwrap());
            stack.push(PaidPartitionFrame {
                lo: frame.cut,
                hi: frame.hi,
                depth: frame.depth + 1,
                stage: 0,
                site: None,
                cut: 0,
            });
            continue;
        }
        if frame.stage == 2 {
            stack.pop();
            consumed.truncate(frame.depth);
            continue;
        }
        if frame.lo == frame.hi {
            return Err(CoverRefusal::Uncovered { depth: frame.depth }.into());
        }
        let mut full = 0usize;
        for &m in &members[frame.lo..frame.hi] {
            paid_poll(budget, at, units)?;
            units = units.wrapping_add(1);
            full += usize::from(leaves[m].0.depth() == frame.depth);
        }
        if full > 0 {
            if frame.hi - frame.lo != 1 {
                return Err(CoverRefusal::Overlap { depth: frame.depth }.into());
            }
            stack.pop();
            consumed.truncate(frame.depth);
            continue;
        }
        let first = &leaves[members[frame.lo]].0;
        let mut site = None;
        'candidate: for lit in first.literals() {
            for &old in &consumed[..frame.depth] {
                paid_poll(budget, at, units)?;
                units = units.wrapping_add(1);
                if old == lit.site {
                    continue 'candidate;
                }
            }
            for &m in &members[frame.lo..frame.hi] {
                paid_poll(budget, at, units)?;
                units = units.wrapping_add(1);
                if leaves[m].0.side_of(lit.site).is_none() {
                    continue 'candidate;
                }
            }
            site = Some(lit.site);
            break;
        }
        let Some(site) = site else {
            return Err(CoverRefusal::NoCommonSplit {
                depth: frame.depth,
                leaves: frame.hi - frame.lo,
            }
            .into());
        };
        let mut cut = frame.lo;
        for i in frame.lo..frame.hi {
            paid_poll(budget, at, units)?;
            units = units.wrapping_add(1);
            if leaves[members[i]].0.side_of(site) == Some(true) {
                members.swap(cut, i);
                cut += 1;
            }
        }
        let top = stack.last_mut().unwrap();
        top.stage = 1;
        top.site = Some(site);
        top.cut = cut;
        consumed.truncate(frame.depth);
        consumed.push(site);
        stack.push(PaidPartitionFrame {
            lo: frame.lo,
            hi: cut,
            depth: frame.depth + 1,
            stage: 0,
            site: None,
            cut: 0,
        });
    }
    paid_checkpoint(budget, at)
}

#[cfg(test)]
#[path = "cover_paid_tests.rs"]
mod paid_tests;

/// The case-split reconstruction (see the module docs). Iterative, so a deep
/// tree cannot overflow the stack.
fn check_partition(paths: &[&LeafPath]) -> Result<(), CoverRefusal> {
    struct Work {
        members: Vec<usize>,
        consumed: Vec<CoverSite>,
    }
    let mut stack = vec![Work {
        members: (0..paths.len()).collect(),
        consumed: Vec::new(),
    }];
    while let Some(Work { members, consumed }) = stack.pop() {
        let depth = consumed.len();
        if members.is_empty() {
            return Err(CoverRefusal::Uncovered { depth });
        }
        // Every member contains every consumed site (by construction), so a
        // member whose literal count equals `depth` has nothing left: it
        // covers the whole sub-region.
        let full = members
            .iter()
            .filter(|&&m| paths[m].depth() == depth)
            .count();
        if full > 0 {
            if members.len() == 1 {
                continue;
            }
            return Err(CoverRefusal::Overlap { depth });
        }
        let first = paths[members[0]];
        let site = first
            .literals()
            .iter()
            .map(|l| l.site)
            .filter(|s| consumed.binary_search(s).is_err())
            .find(|s| members.iter().all(|&m| paths[m].side_of(*s).is_some()));
        let Some(site) = site else {
            return Err(CoverRefusal::NoCommonSplit {
                depth,
                leaves: members.len(),
            });
        };
        let (positive, negative): (Vec<usize>, Vec<usize>) = members
            .iter()
            .copied()
            .partition(|&m| paths[m].side_of(site) == Some(true));
        let mut next = consumed;
        let at = next.binary_search(&site).unwrap_or_else(|i| i);
        next.insert(at, site);
        stack.push(Work {
            members: negative,
            consumed: next.clone(),
        });
        stack.push(Work {
            members: positive,
            consumed: next,
        });
    }
    Ok(())
}

/// The OVERLAP-TOLERANT cover check (graph-lane policy; see the module docs):
/// a Shannon expansion over the recorded literals. A sub-problem is a partial
/// assignment of sites plus the leaves CONSISTENT with it (no literal of theirs
/// contradicts it). It is covered when some member has every literal assigned
/// — its region contains the sub-region — and otherwise it is split on an
/// unassigned site: members carrying the site go to their side, members not
/// carrying it go to BOTH sides. An empty member set is a region no certified
/// leaf covers. Iterative, so a deep tree cannot overflow the stack.
///
/// With `witnesses`, the index of the member that closed each covered
/// sub-problem is pushed (the same member the `covered` test finds; the
/// decision does not depend on whether they are collected).
fn check_cover(
    paths: &[&LeafPath],
    budget: &mut WorkBudget,
    mut witnesses: Option<&mut Vec<usize>>,
) -> Result<(), CoverRefusal> {
    struct Work {
        members: Vec<usize>,
        /// Assigned sites and their sides, sorted by site.
        assigned: Vec<(CoverSite, bool)>,
    }
    fn assigned_side(assigned: &[(CoverSite, bool)], site: CoverSite) -> Option<bool> {
        assigned
            .binary_search_by(|(s, _)| s.cmp(&site))
            .ok()
            .map(|i| assigned[i].1)
    }
    let mut stack = vec![Work {
        members: (0..paths.len()).collect(),
        assigned: Vec::new(),
    }];
    while let Some(Work { members, assigned }) = stack.pop() {
        let depth = assigned.len();
        budget.charge(members.len())?;
        if members.is_empty() {
            return Err(CoverRefusal::Uncovered { depth });
        }
        // Members are consistent with `assigned` by construction, so a member
        // whose every literal is assigned has every literal HOLD: it covers.
        let covered = members.iter().copied().find(|&m| {
            paths[m]
                .literals()
                .iter()
                .all(|l| assigned_side(&assigned, l.site).is_some())
        });
        if let Some(witness) = covered {
            if let Some(out) = witnesses.as_deref_mut() {
                out.push(witness);
            }
            continue;
        }
        // Split site: an unassigned site on EVERY member when there is one (a
        // genuine tree's own split; nothing is duplicated), else the most
        // frequent unassigned site (ties to the smallest, for determinism).
        let first = paths[members[0]];
        let common = first
            .literals()
            .iter()
            .map(|l| l.site)
            .filter(|s| assigned_side(&assigned, *s).is_none())
            .find(|s| members.iter().all(|&m| paths[m].side_of(*s).is_some()));
        let site = match common {
            Some(site) => site,
            None => {
                let mut counts: std::collections::BTreeMap<CoverSite, usize> =
                    std::collections::BTreeMap::new();
                for &m in &members {
                    budget.charge(paths[m].depth())?;
                    for lit in paths[m].literals() {
                        if assigned_side(&assigned, lit.site).is_none() {
                            *counts.entry(lit.site).or_default() += 1;
                        }
                    }
                }
                // Every member has an unassigned literal (none covered), so
                // the map is non-empty; `max_by` keeps the LAST maximum, so
                // iterate in reverse to tie-break toward the smallest site.
                let Some((&site, _)) = counts.iter().rev().max_by(|a, b| a.1.cmp(b.1)) else {
                    return Err(CoverRefusal::Uncovered { depth });
                };
                site
            }
        };
        let mut positive = Vec::with_capacity(members.len());
        let mut negative = Vec::with_capacity(members.len());
        for &m in &members {
            match paths[m].side_of(site) {
                Some(true) => positive.push(m),
                Some(false) => negative.push(m),
                None => {
                    positive.push(m);
                    negative.push(m);
                }
            }
        }
        let at = assigned
            .binary_search_by(|(s, _)| s.cmp(&site))
            .unwrap_or_else(|i| i);
        let mut assigned_neg = assigned.clone();
        assigned_neg.insert(at, (site, false));
        let mut assigned_pos = assigned;
        assigned_pos.insert(at, (site, true));
        stack.push(Work {
            members: negative,
            assigned: assigned_neg,
        });
        stack.push(Work {
            members: positive,
            assigned: assigned_pos,
        });
    }
    Ok(())
}
