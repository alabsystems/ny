// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `certify`: the trusted verdict facade (ENGINE_PROGRAM §1.6, milestones M2b
//! and M2c).
//!
//! This module is thin on purpose. It adds NO bound arithmetic: every value it
//! accepts was produced by an existing certified evaluator (the margin-row
//! Outward engine, the certified emptiness tests, the frozen-dual Higham
//! certificate). What it adds is TYPES that make three mistakes
//! unrepresentable:
//!
//! 1. **closing on an uncertified value.** [`closed`] takes only a
//!    [`Certified<f64>`] and is strict (`> 0`) with no mode flag. A parity or
//!    search value is a [`SearchValue`], which has no closure predicate and no
//!    conversion into `Certified`.
//! 2. **publishing an `unsat` whose leaves do not cover the box.** A tree's
//!    leaves are recorded with their PATHS in a [`CoverLedger`] and must pass
//!    [`CoverLedger::prove`] (prefix-free, siblings split the same site on
//!    opposite sides) before a [`TreeCover`] exists.
//! 3. **publishing an `unsat` from anywhere but a token.** An [`UnsatToken`]
//!    has exactly one constructor, private to this module, reached only
//!    through [`authorize_unsat`] (every class covered, every guard clear) or
//!    [`legacy_uncovered_unsat`] (the census of pre-M2 lanes that still have
//!    no cover, frozen by a ratchet in `ny-cli` whose ceiling may only go
//!    down). `ny-cli`'s `VnncompResult::Unsat` carries one.
//!
//! A property whose unsafe region is empty as a FORMULA needs no bound at all:
//! [`exact`] re-checks a contradicting pair of scalar atoms itself and hands
//! the cover to [`authorize_unsat`] ([`authorize_scalar_emptiness`]).
//!
//! The graph (β-CROWN) branch-and-bound drivers write a COMPLETENESS LEDGER
//! ([`graph`]): every close, with the split literals (ReLU / GenBaB trees) or
//! the box (input-split trees) of the region it closes and the lane's own
//! closing evidence. A frame whose leaves do not cover its root refuses the
//! driver's `Verified`; a proved frame is captured for the publication path,
//! where [`authorize_graph_instance`] mints an instance token only if every
//! clause of the instance is entailed by a captured cover on a box containing
//! the declared one.
//!
//! # Rules this module keeps (and ratchets)
//!
//! * it never names the proof profile: the same certification runs in both
//!   profiles, and nothing here can learn which one is active;
//! * it never uses fused multiply-add, nor the TF32 search GEMM;
//! * `UnsatToken::mint` has exactly two callers;
//! * every adapter that mints a cover payload is `pub(crate)`;
//! * every [`ProofSink`] method returns `()` (sinks are write-only); the
//!   process sink is installed by the ENTRY POINT ([`install_proof_sink`]),
//!   fed by a lane only with a PROVED, authorized frame ([`stream_frame`]:
//!   its leaves and its write-only [`HintLedger`]'s v2 nodes) and closed
//!   after publication ([`finish_stream`]); hint capture is switched on by
//!   the entry point's Proof arm alone ([`enable_hint_capture`]).
//!
//! B2.3: the margin-row lane's remaining routes stream checkable proofs too —
//! the frozen-dual tier's prunes ([`FdProofTrace`]: one state record and one
//! node per prune, with its dual point), and the wave engine's leaves (its
//! per-class ledgers' leaves pushed into the frame's [`HintLedger`]). Class
//! trees and nested epoch frames already stream their own authorized frames
//! (absolute literals under their prefix).
//!
//! B2.2: every unsat-publishing lane also keeps, under the Proof profile only,
//! a write-only PROOF TAPE of its closes (`certify::lane_proof`) — the graph ledger's
//! frames, the multi-objective driver's domains and the lowdim tree — which
//! the entry point streams after publication as bundle nodes for the
//! independent checker ([`stream_lane_proofs`]).
//!
//! The rules are checked by the crate's section-1.7 ratchet integration test.
//!
//! # What does not compile (and must not)
//!
//! A search value has no closure predicate:
//!
//! ```compile_fail
//! use ny_propagate::certify::{closed, SearchValue};
//! let s = SearchValue::new(1.0);
//! let _ = closed(&s);
//! ```
//!
//! A raw `f64` is not a certified value either:
//!
//! ```compile_fail
//! let _ = ny_propagate::certify::closed(&1.0_f64);
//! ```
//!
//! `Certified` cannot be built outside this module, by constructor or by
//! literal:
//!
//! ```compile_fail
//! use ny_propagate::certify::{CertReceipt, Certified};
//! let _ = Certified::new(1.0_f64, CertReceipt::MarginRowOutward);
//! ```
//!
//! ```compile_fail
//! use ny_propagate::certify::{CertReceipt, Certified};
//! let _ = Certified { v: 1.0_f64, receipt: CertReceipt::MarginRowOutward };
//! ```
//!
//! Nor can an `UnsatToken`, a `TreeCover` or an exact-emptiness certificate:
//!
//! ```compile_fail
//! use ny_propagate::certify::{TokenScope, UnsatSource, UnsatToken};
//! let _ = UnsatToken::mint(UnsatSource::Authorized, TokenScope::Instance, Box::new([]), 1.0);
//! ```
//!
//! ```compile_fail
//! use ny_propagate::certify::TreeCover;
//! let _ = TreeCover { classes: Box::new([]), leaves: 1, max_depth: 0, min_margin: 1.0,
//!     empty_leaves: 0, subtree_leaves: 0 };
//! ```
//!
//! ```compile_fail
//! use ny_propagate::certify::{ExactEmptiness, ScalarAtom, ScalarRel};
//! let a = ScalarAtom { rel: ScalarRel::Gt, constant: 1.0 };
//! let _ = ExactEmptiness { lower: a, upper: a, slack: 1.0 };
//! ```
//!
//! ```compile_fail
//! use ny_propagate::certify::exact::scalar_emptiness;
//! let _ = scalar_emptiness(&[]);
//! ```
//!
//! Nor can another crate launder a number of its choosing through the
//! margin-row adapters, which wrap values only this crate's certified lanes
//! produce (they are `pub(crate)`; see `certify::margin_row`):
//!
//! ```compile_fail
//! use ny_propagate::certify::margin_row;
//! use ny_propagate::margin_row::RoundMode;
//! let _ = margin_row::outward_bound(RoundMode::Outward, 1.0);
//! ```
//!
//! ```compile_fail
//! use ny_propagate::certify::{authorize_unsat, margin_row, GuardReport, TokenScope};
//! use ny_propagate::margin_row::RoundMode;
//! let cover = margin_row::root_class(RoundMode::Outward, 3, 1.0).unwrap();
//! let _ = authorize_unsat(TokenScope::Instance, &[3], &[cover], &GuardReport::new());
//! ```
//!
//! Nor can another crate record a graph-lane close or build a captured cover:
//!
//! ```compile_fail
//! use ny_propagate::certify::graph;
//! let _ = graph::graph_bound_evidence(false, 1.0, 2.0, 0.0);
//! ```
//!
//! ```compile_fail
//! use ny_propagate::certify::GraphInvocationCover;
//! let _ = GraphInvocationCover { lane: "relu_split" };
//! ```
//!
//! The graph instance door re-checks entailment; with nothing captured there
//! is no token:
//!
//! ```
//! use ny_propagate::certify::{
//!     authorize_graph_instance, GraphCoverCapture, GraphInstanceProperty, GraphObjective,
//! };
//!
//! let atom = GraphObjective::new(&[1.0, -1.0], 0.0, false).expect("finite");
//! let property =
//!     GraphInstanceProperty::new(vec![0.0], vec![1.0], vec![vec![atom]]).expect("well formed");
//! assert!(authorize_graph_instance(&property, &GraphCoverCapture::default()).is_err());
//! ```
//!
//! The exact entry RE-CHECKS what it is given; a satisfiable group gets no
//! token:
//!
//! ```
//! use ny_propagate::certify::{authorize_scalar_emptiness, ScalarAtom, ScalarRel};
//!
//! let gt = |c| ScalarAtom { rel: ScalarRel::Gt, constant: c };
//! let lt = |c| ScalarAtom { rel: ScalarRel::Lt, constant: c };
//! // t > 0.05 and t < -0.05: empty.
//! assert!(authorize_scalar_emptiness(&[vec![gt(0.05), lt(-0.05)]]).is_ok());
//! // t > -0.05 and t < 0.05: t = 0 satisfies both.
//! assert!(authorize_scalar_emptiness(&[vec![gt(-0.05), lt(0.05)]]).is_err());
//! ```
//!
//! The closure DECISION is public and mints nothing:
//!
//! ```
//! use ny_propagate::certify::margin_row;
//! use ny_propagate::margin_row::RoundMode;
//!
//! assert!(margin_row::closes(RoundMode::Outward, 0.25));
//! // Strict: a zero margin admits a counterexample.
//! assert!(!margin_row::closes(RoundMode::Outward, 0.0));
//! // A parity value never certifies, so it can never close.
//! assert!(!margin_row::closes(RoundMode::Parity, 0.25));
//! ```

mod boxes;
mod cover;
pub mod exact;
pub(crate) mod fd_proof;
pub mod graph;
pub(crate) mod lane_proof;
pub mod margin_row;
pub mod residency;
pub mod scope;
mod sink;
mod token;
mod value;

pub(crate) use token::aligned_single_class_covers;
pub mod work;

#[cfg(test)]
mod graph_rows_tests;
#[cfg(test)]
mod graph_tests;
#[cfg(test)]
mod lane_proof_tests;
#[cfg(test)]
mod tests;

pub use cover::{
    CoverLedger, CoverLiteral, CoverRefusal, CoverSite, LeafEvidence, LeafPath, TreeCover,
};
pub(crate) use cover::{PaidCoverError, PaidCoverLedger, PaidTreeCover};
pub use exact::{
    authorize_scalar_emptiness, ExactEmptiness, ExactRefusal, ScalarAtom, ScalarRel,
    UNSAFE_CONJUNCTION,
};
pub use fd_proof::FdProofTrace;
pub use graph::{
    authorize_graph_instance, reset_graph_cover_capture, take_graph_cover_capture,
    GraphCoverCapture, GraphInstanceProperty, GraphInstanceRefusal, GraphInvocationCover,
    GraphObjective,
};
pub use lane_proof::{
    reset_lane_proof_capture, stream_lane_proofs, stream_lane_proofs_installed,
    take_lane_proof_capture, LaneProofCapture, LaneProofInstance, LaneProofSummary,
};
#[cfg(test)]
pub(crate) use sink::force_hint_capture_for_test;
pub use sink::{
    enable_hint_capture, finish_stream, hint_capture_enabled, install_proof_sink,
    proof_sink_installed, reserve_node_id, stream_cover_into, stream_frame, stream_opaque_claim,
    stream_sites, with_thread_sink, AbsLit, AbsSite, AlphaRecord, FdLambdaRecord, FdNodeHint,
    FdStateRecord, FrameSplit, HintLedger, InputLit, LeafClose, LeafHint, LeafRecord, NodeClose,
    NodeRecord, NullSink, PreHint, ProofSink, RecordingSink, RetireHint, SiteMap, SiteTensorRecord,
    SitesRecord, VerdictRecord,
};
pub use token::{
    authorize_unsat, legacy_uncovered_unsat, ratify, AuthorizeRefusal, ClassCover, GuardId,
    GuardReport, GuardTrip, LegacyUnsatAuthority, TokenScope, UnsatSource, UnsatToken,
};
pub use value::{closed, CertReceipt, Certified, SearchValue};
