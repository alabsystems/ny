// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Compile-time dispatch-coverage matrix for backward bound propagation.
//!
//! # The bug class this exists to kill
//!
//! Three separate silent-degradation bugs landed in one week, all with the same
//! shape: a [`Layer`] variant handled in the DENSE CROWN dispatcher and quietly
//! **absent** from the BATCHED one, so every node of that kind fell through to a
//! generic fallback that is *sound but much looser* — with **no test and no
//! compile error** to notice.
//!
//! * `Where` had no batched arm at all, so a causal / padding attention mask
//!   blurred into the condition-blind union `[min(t,f), max(t,f)]` at every
//!   position.
//! * `Concat` / `Sub` / `Div` / `Gather` / pooling arms were missing from the
//!   batched dispatcher; `ConcatLayer::propagate_linear_batched_binary` was
//!   implemented and tested with **zero production callers**.
//! * `supports_batched_crown` omitted `CumSum`, whose batched backward was
//!   implemented and dispatched — so one `CumSum` node disqualified an entire
//!   network from the batched path.
//!
//! The pre-existing coverage guard (`network::dispatch_coverage_tests`) polices
//! the *dense* family of dispatch sites by parsing their source text. It does
//! not police the two dispatchers where this bug class actually lives. This
//! module does.
//!
//! # What it guarantees
//!
//! For **every** `Layer` variant, and for each of four dispatchers
//! ([`Dispatcher`]), the matrix records one of three verdicts
//! ([`DispatchVerdict`]): `Supported`, `FallsBack(reason)`, or
//! `Refused(reason)`. Two mechanisms keep that record honest:
//!
//! 1. **Compile time.** Each classifier in [`classify`] is an exhaustive
//!    `match &Layer` with **no wildcard arm**, in the style ny already uses in
//!    `backward_dispatch/dispatch.rs` and `ny-trace-bridge`'s `coverage.rs`.
//!    Adding a variant to `Layer` fails to compile in all four until someone
//!    classifies it in each.
//! 2. **Run time.** The tests in [`tests`] execute the real dispatchers over
//!    the canonical sample catalogue and fail when a variant marked `Supported`
//!    actually falls through — the specific failure the compile-time half
//!    cannot see. See [`tests`] for the cross-check inventory.
//!
//! # What a verdict does NOT mean
//!
//! `Supported` is a statement about the *dispatcher*, never about bound
//! quality: it means an arm exists that produces real linear bounds for the
//! variant, not that it succeeds for every node's shape or configuration. Every
//! `FallsBack` route in ny is a **valid enclosure** — looser, never wrong — and
//! every `Refused` route fails closed. Nothing in this module relaxes a bound;
//! it only records which route a node takes, so a route silently getting
//! *looser* becomes a test failure instead of a quiet regression.

use crate::layers::Layer;

mod catalogue;
mod classify;
#[cfg(test)]
mod tests;

/// Number of variants in [`Layer`].
///
/// Pinned here and asserted against [`catalogue::canonical_layers`] in tests.
/// The four exhaustive matches in [`classify`] are the primary gate; this
/// constant is the second line of defence, catching a variant that was
/// classified but never given a sample to probe with.
pub const LAYER_VARIANT_COUNT: usize = 101;

/// What a dispatcher does with a given [`Layer`] variant.
///
/// The ordering is from most to least capable. Only [`Self::Supported`] means
/// the variant reaches a real relaxation on that path; the other two are both
/// sound, and differ in whether the node still gets a (looser) bound or the
/// dispatcher fails closed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchVerdict {
    /// The dispatcher has a real arm producing linear bounds for this variant.
    Supported,
    /// The dispatcher deliberately routes this variant to a generic fallback —
    /// IBP concretization, partial CROWN, or a fixed-slope relaxation. Sound,
    /// and a valid enclosure, but strictly looser than a real arm. The string
    /// says *why* the arm is absent, so the record is a decision and not a gap.
    FallsBack(&'static str),
    /// The dispatcher refuses this variant outright (fail-closed error). The
    /// string says why no sound lowering exists on this path.
    Refused(&'static str),
}

impl DispatchVerdict {
    /// Whether this variant reaches a real arm on the dispatcher.
    #[must_use]
    pub const fn is_supported(self) -> bool {
        matches!(self, Self::Supported)
    }

    /// Whether this variant takes the dispatcher's generic (looser) fallback.
    #[must_use]
    pub const fn falls_back(self) -> bool {
        matches!(self, Self::FallsBack(_))
    }

    /// Short, stable lowercase label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Supported => "supported",
            Self::FallsBack(_) => "falls_back",
            Self::Refused(_) => "refused",
        }
    }

    /// The recorded reason, for the two non-`Supported` verdicts.
    #[must_use]
    pub const fn reason(self) -> Option<&'static str> {
        match self {
            Self::Supported => None,
            Self::FallsBack(reason) | Self::Refused(reason) => Some(reason),
        }
    }
}

/// The four backward-propagation dispatchers this matrix covers.
///
/// Each names one *routing decision point* — the place where a node's `Layer`
/// variant selects a code path — not a whole propagation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dispatcher {
    /// `network::backward_dispatch::dispatch_backward_layer`, the canonical
    /// per-node dense CROWN backward every scalar dispatch site routes through.
    DenseCrownBackward,
    /// The batched CROWN backward path: `Layer::propagate_crown_backward_batched`
    /// for unary nodes plus `GraphNetwork::dispatch_special_batched_operator`
    /// for the multi-input and graph-structured ones, gated for admission by
    /// `Layer::supports_batched_crown`.
    BatchedCrownBackward,
    /// `graph_alpha::propagate_dag::init_dag_alpha_state` — which variants are
    /// alpha-bearing, i.e. register optimizable parameters and hold a graph in
    /// the α-CROWN optimizer rather than letting it leave through the
    /// no-optimizable-state early return.
    AlphaEntry,
    /// `core::graph::crown_block_wise::crown_backward_within_block_with_engine`,
    /// the per-block CROWN backward used by block-wise verification.
    BlockWise,
}

impl Dispatcher {
    /// Every dispatcher, in declaration order.
    pub const ALL: [Self; 4] = [
        Self::DenseCrownBackward,
        Self::BatchedCrownBackward,
        Self::AlphaEntry,
        Self::BlockWise,
    ];

    /// Short, stable lowercase label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DenseCrownBackward => "dense_crown_backward",
            Self::BatchedCrownBackward => "batched_crown_backward",
            Self::AlphaEntry => "alpha_entry",
            Self::BlockWise => "block_wise",
        }
    }
}

/// Classify one `layer` against one `dispatcher`.
///
/// Delegates to the exhaustive matches in [`classify`], each of which fails to
/// compile when a `Layer` variant is added and left unclassified.
#[must_use]
pub fn verdict(dispatcher: Dispatcher, layer: &Layer) -> DispatchVerdict {
    match dispatcher {
        Dispatcher::DenseCrownBackward => classify::dense_crown_backward(layer),
        Dispatcher::BatchedCrownBackward => classify::batched_crown_backward(layer),
        Dispatcher::AlphaEntry => classify::alpha_entry(layer),
        Dispatcher::BlockWise => classify::block_wise(layer),
    }
}

/// One row of the coverage matrix: a variant's verdict on every dispatcher.
#[derive(Debug, Clone)]
pub struct DispatchCoverageRow {
    /// The `Layer` variant's name, as declared in the enum.
    pub layer: &'static str,
    /// Verdicts in [`Dispatcher::ALL`] order.
    pub verdicts: [DispatchVerdict; 4],
}

impl DispatchCoverageRow {
    /// This row's verdict on one dispatcher.
    #[must_use]
    pub fn verdict(&self, dispatcher: Dispatcher) -> DispatchVerdict {
        let index = Dispatcher::ALL
            .iter()
            .position(|d| *d == dispatcher)
            .unwrap_or(0);
        self.verdicts[index]
    }

    /// Whether this variant reaches a real arm on the dense dispatcher but
    /// falls back on the batched one.
    ///
    /// This predicate IS the bug class: every one of the three regressions was
    /// a variant for which it silently became true. It stays true for the
    /// remaining, deliberately-declared gaps, so it is a census — not an
    /// assertion — and the tests pin the census rather than demanding it be
    /// empty.
    #[must_use]
    pub fn dense_only(&self) -> bool {
        self.verdict(Dispatcher::DenseCrownBackward).is_supported()
            && !self
                .verdict(Dispatcher::BatchedCrownBackward)
                .is_supported()
    }
}

/// `Layer` variants for which no value can be constructed, as
/// `(variant, reason)`.
///
/// [`coverage`] can only report on variants it can build a sample of, so these
/// are the difference between its row count and [`LAYER_VARIANT_COUNT`]. They
/// are NOT a hole in the guard: the classifiers match over the *type*, so these
/// variants are still classified at compile time and a new variant still breaks
/// the build. What they lack is a runtime probe.
#[must_use]
pub fn unconstructible_variants() -> Vec<(&'static str, &'static str)> {
    catalogue::UNCONSTRUCTIBLE_VARIANTS.to_vec()
}

/// The per-variant, per-dispatcher coverage matrix.
///
/// Rows are in `Layer` declaration order. Deterministic, and exactly
/// `LAYER_VARIANT_COUNT - unconstructible_variants().len()` rows long — see
/// [`unconstructible_variants`] for the difference.
#[must_use]
pub fn coverage() -> Vec<DispatchCoverageRow> {
    catalogue::canonical_layers()
        .into_iter()
        .map(|(name, layer)| DispatchCoverageRow {
            layer: name,
            verdicts: [
                classify::dense_crown_backward(&layer),
                classify::batched_crown_backward(&layer),
                classify::alpha_entry(&layer),
                classify::block_wise(&layer),
            ],
        })
        .collect()
}

/// Variants that reach a real DENSE arm but take the batched path's generic
/// fallback — the exact shape of all three historical regressions.
///
/// Callers use this to see, at a glance, where a batched verdict is still
/// paying a looseness penalty a dense verdict is not.
#[must_use]
pub fn dense_only_variants() -> Vec<&'static str> {
    coverage()
        .into_iter()
        .filter(DispatchCoverageRow::dense_only)
        .map(|row| row.layer)
        .collect()
}
