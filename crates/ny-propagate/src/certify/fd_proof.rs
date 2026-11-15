// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! B2.3 (ENGINE_PROGRAM §1.7 B1/B2): the frozen-dual tier's WRITE-ONLY proof
//! trace — what the independent checker needs to re-derive a frozen-dual
//! class close prune by prune.
//!
//! The tier (`margin_row::zono_bab`) closes a class by a level-synchronous
//! BaB over a split ORDER of the state's unstable rows: a node at depth `D`
//! fixes the sides of `order[0..D]`, and the class closes when every node of
//! the last level is pruned on a certified bound `> 0` and the level ledger
//! closes. Its prunes are therefore the leaves of a complete binary split
//! tree whose split sites are ABSOLUTE Relu neurons. Under the Proof profile
//! only (hint capture on), the tier records every prune — its sides along
//! the order and the dual point `(L0, L1)` handed to the certified evaluator
//! (`SearchOptions::record_prunes`, a write-only side channel of the search;
//! 0 under Competition) — and this trace carries them, with the state's
//! identity, to the stream:
//!
//! * one [`FdStateRecord`] (the class, the sha256 of the certified state the
//!   prunes rest on, its sizes and root, the order as absolute sites, the
//!   slopes `λ` and certified boxes of every relaxation symbol — for a
//!   #breakpoint-dual RANKED compact state, the omitted rows' class
//!   relaxation included, `#fd-compact-checkable`);
//! * one `node` per prune: the frame prefix plus `(order[i], side_i)` for
//!   `i < D` in branch order, `close: bound`, `src: fd_prune`, discharging
//!   the class, with the dual point as `hint.fd` (aligned with the literals;
//!   zero multipliers on the prefix).
//!
//! A class whose state root already certified (no level) streams ONE node:
//! the frame prefix with an empty dual point.
//!
//! # Why nothing here can move a verdict
//!
//! The trace has private fields and no reader: its public methods return a
//! class index or a flag (ratchet `fd_proof_trace_exposes_no_reader`), and
//! [`FdProofTrace::stream_into`] hands its records only to a write-only
//! [`ProofSink`]. It is built from a FINISHED search (its prunes are moved
//! out of the report the cover adapter reads, so that report is
//! bit-identical in both profiles) and streamed only after the frame's token
//! was minted. A trace that is not complete (the record budget ran out, or
//! the class ran on a #breakpoint-dual RANKED compact state whose slope table
//! does not cover every relaxation symbol) is never streamed: the lane falls
//! back to its frozen-dual CLAIM, which the checker can only report
//! unconfirmed. Since `#fd-compact-checkable` a ranked compact state streams
//! when the tier recorded (Proof profile only) and VERIFIED its class
//! relaxation: the slope of every omitted row, so the `hint.fd_lambda` table
//! describes exactly the relaxation the prunes were certified with.

use std::sync::atomic::{AtomicU64, Ordering};

use super::sink::{
    next_node_id, note_node_streamed, AbsLit, AbsSite, FdLambdaRecord, FdNodeHint, FdStateRecord,
    NodeClose, NodeRecord, ProofSink,
};
use crate::margin_row::zono_bab::PrunedNode;
use crate::margin_row::zono_state::Side;

/// Process-unique frozen-dual state ids.
static NEXT_FD_STATE_ID: AtomicU64 = AtomicU64::new(0);

/// One frozen-dual class close, prune by prune (see the module docs).
#[derive(Debug, Clone, PartialEq)]
pub struct FdProofTrace {
    class: usize,
    t: usize,
    state_sha256: String,
    n_gens: usize,
    in_gens: usize,
    unstable: usize,
    relaxed: usize,
    root: f64,
    order: Vec<AbsSite>,
    lambda: Vec<FdLambdaRecord>,
    prunes: Vec<PrunedNode>,
    root_closed: bool,
    complete: bool,
}

/// What the tier hands over to build a trace (crate-internal plumbing).
pub(crate) struct FdProofParts {
    pub(crate) class: usize,
    pub(crate) t: usize,
    pub(crate) state_sha256: String,
    pub(crate) n_gens: usize,
    pub(crate) in_gens: usize,
    pub(crate) unstable: usize,
    /// Relaxation symbols the slope table covers (`unstable` for a full
    /// state; every unstable neuron of the frame for a compact state with
    /// its class relaxation).
    pub(crate) relaxed: usize,
    pub(crate) root: f64,
    pub(crate) order: Vec<AbsSite>,
    pub(crate) lambda: Vec<FdLambdaRecord>,
    pub(crate) prunes: Vec<PrunedNode>,
    pub(crate) root_closed: bool,
    pub(crate) complete: bool,
}

impl FdProofTrace {
    /// Assemble a trace from a finished search (crate-internal: only the
    /// tier builds one).
    #[must_use]
    pub(crate) fn new(parts: FdProofParts) -> Self {
        Self {
            class: parts.class,
            t: parts.t,
            state_sha256: parts.state_sha256,
            n_gens: parts.n_gens,
            in_gens: parts.in_gens,
            unstable: parts.unstable,
            relaxed: parts.relaxed,
            root: parts.root,
            order: parts.order,
            lambda: parts.lambda,
            prunes: parts.prunes,
            root_closed: parts.root_closed,
            complete: parts.complete,
        }
    }

    /// The adversarial class whose close this trace re-derives.
    #[must_use]
    pub fn class(&self) -> usize {
        self.class
    }

    /// Whether every certified leaf of the class's tree was recorded AND
    /// the slope table covers every relaxation symbol the tier certified
    /// with (only a complete trace may replace the class's claim in the
    /// stream).
    #[must_use]
    pub fn complete(&self) -> bool {
        self.complete
    }

    /// `#fd-compact-checkable`: how many relaxation symbols the slope table
    /// covers (a count; the table itself leaves only through `stream_into`).
    #[must_use]
    pub fn relaxed(&self) -> usize {
        self.relaxed
    }

    /// Stream the state record and one node per prune into `sink`, every
    /// node under the frame's `prefix` (absolute literals of the frame root).
    pub(crate) fn stream_into(self, prefix: &[AbsLit], sink: &mut dyn ProofSink) {
        let id = NEXT_FD_STATE_ID.fetch_add(1, Ordering::Relaxed);
        let prunes = if self.root_closed {
            1
        } else {
            self.prunes.len() as u64
        };
        sink.fd_state(&FdStateRecord {
            id,
            class: self.class,
            t: self.t,
            state_sha256: self.state_sha256,
            n_gens: self.n_gens,
            in_gens: self.in_gens,
            unstable: self.unstable,
            relaxed: self.relaxed,
            root: self.root,
            order: self.order.clone(),
            lambda: self.lambda,
            prunes,
        });
        let class = self.class;
        let node = |sink: &mut dyn ProofSink, own: Vec<AbsLit>, l0: &[f64], l1: &[f64], g: f64| {
            let mut lits = prefix.to_vec();
            let mut m0 = vec![0.0; prefix.len()];
            let mut m1 = vec![0.0; prefix.len()];
            lits.extend(own);
            m0.extend_from_slice(l0);
            m1.extend_from_slice(l1);
            sink.node(&NodeRecord {
                id: next_node_id(),
                region: 0,
                lits,
                xlits: Vec::new(),
                close: NodeClose::Bound,
                src: "fd_prune",
                dis: Some(vec![class]),
                beta: Vec::new(),
                alpha: None,
                lift: false,
                margin: g,
                empty_site: None,
                fd: Some(FdNodeHint {
                    state: id,
                    l0: m0,
                    l1: m1,
                }),
            });
            note_node_streamed();
        };
        if self.root_closed {
            node(sink, Vec::new(), &[], &[], self.root);
            return;
        }
        for p in &self.prunes {
            let lits: Option<Vec<AbsLit>> = p
                .sides
                .iter()
                .enumerate()
                .map(|(i, side)| {
                    self.order.get(i).map(|&site| AbsLit {
                        site,
                        positive: *side == Side::On,
                    })
                })
                .collect();
            // A prune deeper than the recorded order cannot be translated;
            // `complete` already excluded such a trace from streaming, so
            // this is unreachable on a streamed trace.
            let Some(lits) = lits else {
                continue;
            };
            node(sink, lits, &p.l0, &p.l1, p.certified);
        }
    }
}
