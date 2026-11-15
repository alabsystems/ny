// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `Certified<T>` and `SearchValue`: the type split between a value a trusted
//! evaluator certified and a value that is only good for search.

/// Which trusted evaluator produced a [`Certified`] value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CertReceipt {
    /// A margin-row bound computed by the certified-outward engine: a root
    /// `dj`, a domain evaluation, a child score, or a parent lift.
    MarginRowOutward,
    /// A margin-row emptiness slack: a certified outward LOWER bound minus a
    /// certified outward UPPER bound of one coordinate over a region. A
    /// positive slack proves the region empty (a finite dual / Farkas-style
    /// certificate, never an error path).
    MarginRowEmptiness,
    /// A frozen-dual root or prune value (`zono_state::problem`'s Higham
    /// certificate, strict `> 0`).
    FrozenDual,
    /// A graph-lane (β-CROWN) objective bound strictly past its threshold on
    /// one domain: the margin `lower − threshold` (or `threshold − upper` in
    /// upper-bound mode), in `f64`, of a bound pair the lane's own CROWN / α /
    /// β evaluator produced and its own closure rule
    /// (`domain_is_verified`) accepted. The arithmetic is the lane's, trusted
    /// exactly as it is today; what the receipt adds is that the close is on
    /// the record, attached to the region it closes.
    GraphLaneBound,
    /// A graph-lane domain whose region the lane's certified node bounds show
    /// EMPTY: a split half-space that misses the certified pre-activation
    /// interval, or a constraint application that inverted a certified
    /// interval (`InfeasibleDomain`). No numeric slack survives that path, so
    /// the leaf is a margin-free lane close.
    GraphLaneInfeasible,
    /// A part of an input-split child box that the lane's certified linear
    /// bound excluded when it clipped the box (the clip's own closure).
    GraphLaneClip,
    /// An input-split child box the lane's batched IBP prescreen verified
    /// (the prescreen reports its verdict mask, not the margin).
    GraphLaneScreen,
    /// #vit-v3-bankable (M2c's certified-infeasibility rule on the graph
    /// lane): a graph-lane emptiness SLACK. One split literal of the region
    /// contradicts the lane's certified bound on that pre-activation — an
    /// active literal `z >= 0` against a certified upper `u < 0` (slack `-u`),
    /// an inactive literal `z <= 0` against a certified lower `l > 0` (slack
    /// `l`) — or a constraint application inverted a certified interval by
    /// `l - u > 0`. A positive slack between certified outward bounds proves
    /// the region empty: a finite interval (Farkas-style) certificate, never
    /// an error path. The arithmetic is the lane's, trusted exactly as its
    /// bound closes are; what the receipt adds is the NUMBER, so the close is
    /// strict, auditable and re-derivable by an offline checker.
    GraphLaneEmptiness,
}

impl CertReceipt {
    /// Stable token for observables.
    #[must_use]
    pub const fn token(self) -> &'static str {
        match self {
            Self::MarginRowOutward => "margin_row_outward",
            Self::MarginRowEmptiness => "margin_row_emptiness",
            Self::FrozenDual => "frozen_dual",
            Self::GraphLaneBound => "graph_lane_bound",
            Self::GraphLaneInfeasible => "graph_lane_infeasible",
            Self::GraphLaneClip => "graph_lane_clip",
            Self::GraphLaneScreen => "graph_lane_screen",
            Self::GraphLaneEmptiness => "graph_lane_emptiness",
        }
    }
}

/// A value certified by trusted (TCB) arithmetic.
///
/// The fields are private and the only constructor is visible to
/// `crate::certify` alone, so a `Certified` can come from nowhere but a
/// `certify::*` adapter that wraps an existing certified evaluator. Reading the
/// value is unrestricted; minting one is not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Certified<T> {
    v: T,
    receipt: CertReceipt,
}

impl<T: Copy> Certified<T> {
    /// The only constructor. `pub(in crate::certify)`: see the type docs.
    pub(in crate::certify) fn new(v: T, receipt: CertReceipt) -> Self {
        Self { v, receipt }
    }

    /// The certified value.
    #[must_use]
    pub fn value(&self) -> T {
        self.v
    }

    /// Which evaluator certified it.
    #[must_use]
    pub fn receipt(&self) -> CertReceipt {
        self.receipt
    }
}

/// THE closure predicate: strict (`> 0`), finite, and it takes no mode flag.
///
/// This replaces the margin-row `closed(mode, b)` split in which the `Parity`
/// branch closed on `b >= 0`: a zero margin admits a counterexample, and a
/// value that was not produced by the certified path cannot reach this
/// function at all (it is a [`SearchValue`], which has no closure predicate).
#[must_use]
pub fn closed(b: &Certified<f64>) -> bool {
    b.v.is_finite() && b.v > 0.0
}

/// A value produced by an UNCERTIFIED evaluation (round-to-nearest parity
/// arithmetic, a future f32 or device search). It ranks, schedules and hints;
/// it has no closure predicate, and there is no conversion into
/// [`Certified`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchValue(f64);

impl SearchValue {
    /// Wrap a search value.
    #[must_use]
    pub const fn new(v: f64) -> Self {
        Self(v)
    }

    /// The raw value, for ranking and scheduling only.
    #[must_use]
    pub const fn hint(self) -> f64 {
        self.0
    }
}
