// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The graph-lane (β-CROWN) COMPLETENESS LEDGER: the cover gate for the
//! branch-and-bound drivers that published `unsat` through the
//! `BetaCrownVerdict` legacy-uncovered door (ENGINE_PROGRAM §1.6 U5/U6).
//!
//! # What was uncovered, and what this adds
//!
//! The graph lane decides `Verified` when its queue drains with no unresolved
//! flag. Nothing checked that the closed leaves actually tile the root: a
//! child dropped by a bookkeeping path (`Ok(None)` read as prune-and-forget, a
//! misaligned batch result, a lost retry) left no trace, and the verdict
//! crossed into `ny-cli` as a status string. This module puts every close ON
//! THE RECORD, attached to the region it closes, and checks the record before
//! the lane may keep its `Verified`:
//!
//! * **ReLU / GenBaB split trees** record each closed region as its split
//!   LITERALS (`(node, neuron, side)` at zero, or at a GenBaB point) and are
//!   checked by the overlap-tolerant path cover of `certify::cover`;
//! * **input-split trees** record each closed region as its BOX (and each
//!   input clip as the slabs it excluded) and are checked by the exact box
//!   cover of `certify::boxes`.
//!
//! The closing EVIDENCE is the lane's own — its certified bound strictly past
//! the threshold ([`CertReceipt::GraphLaneBound`]), its certified emptiness
//! ([`CertReceipt::GraphLaneInfeasible`]), its clip
//! ([`CertReceipt::GraphLaneClip`]), its batched IBP prescreen
//! ([`CertReceipt::GraphLaneScreen`]). The graph lane's arithmetic is trusted
//! exactly as it is today (including whatever its own soundness gates admit
//! for device GEMM); the ledger removes the COMPLETENESS trust, and adds no
//! arithmetic of its own beyond exact comparisons.
//!
//! # Scope of a frame, and why facts may overlap
//!
//! A driver invocation ARMS a frame on its verifier's [`GraphCoverSlot`]
//! (root box, objective, threshold, direction). Recording sites deep in the
//! batch processors — including parallel ones — add facts to the armed frame
//! through `&self`. A fact is recorded where it is established, so a retried
//! or speculative evaluation can leave facts about regions the committed tree
//! never used; they are TRUE facts about those regions, and the cover gates
//! tolerate them. What they cannot do is close a region nobody closed.
//!
//! Every record carries the objective, threshold and direction it was decided
//! for; a record for anything else POISONS the frame (the frame then cannot
//! be checked and the lane's `Verified` stays on the counted legacy door). A
//! nested arm on the same verifier poisons the outer frame for the same
//! reason. An encoding the ledger cannot express (a normalization split, a
//! non-finite GenBaB point, an unknown node, a GenBaB empty child whose half
//! space is not recorded) also leaves the frame UNBUILT — never refused.
//!
//! # Outcomes
//!
//! When the driver's own result is `Verified`, [`GraphCoverScope::finish`]
//! checks the frame:
//!
//! * **proved** — the cover and its problem (root box, objective) are
//!   deposited into the process CAPTURE for the publication path;
//! * **refused** — the lane closed a tree whose certified leaves do not cover
//!   its root: the driver DOWNGRADES its own `Verified` to `Unknown` (a
//!   latent soundness finding, printed as `[cover-gate] REFUSED`);
//! * **unbuilt** — the lane's `Verified` stands and publishes through the
//!   counted legacy door.
//!
//! # Publication
//!
//! `ny-cli` resets the capture before the in-process verifier runs and takes
//! it after. [`authorize_graph_instance`] — the one public door of this module
//! that returns a token — mints an INSTANCE-scoped token only if every clause
//! of the instance's unsafe region (as the caller parsed it, the U1 trust the
//! margin-row lane places in its objective rows) is entailed by a captured
//! cover on a root box that CONTAINS the declared box. It re-checks the
//! entailment itself and mints through `authorize_unsat`; a captured cover is
//! unforgeable ([`GraphInvocationCover`] has private fields and is built only
//! here, after a gate passed).
//!
//! # What the door still trusts (adversarial review)
//!
//! * **The same network, in the same coordinates.** A captured cover names its
//!   root box and objective row, not the graph. The door therefore trusts
//!   that a ledgered driver ran on the instance's own network (or its exact
//!   `from_sequential` conversion) with the instance's input order — the
//!   premise the legacy door rested on. The drivers are reached only through
//!   `ny-cli`'s `dispatch_graph_constraint` with the model graph; relational
//!   rewrites (dual-network properties) and per-clause input boxes are refused
//!   by the property builder, and a row of another length or a box that does
//!   not contain the declared one never entails.
//! * **The lane's evaluator.** A graph-lane bound close is the lane's own CROWN
//!   / α / β value (including the uncharged BN-fold rounding the M2e item
//!   records for BN models). The ledger adds completeness, not arithmetic.
//! * **No guards.** Unlike the margin-row lane (G2 containment, G3 spec
//!   agreement), a graph token passes no downgrade-only guard: none exists
//!   for the graph lane yet.
//! * **Proof stream (B2.2).** Under the Proof profile (the lane proof capture
//!   armed, `certify::lane_proof`) every frame also keeps a write-only TAPE of
//!   the same closes — split literals with the lane's β, or boxes and clip
//!   slabs — deposited with the frame's outcome (complete when proved) and
//!   streamed after publication as `ny-proof-bundle/v2` nodes the independent
//!   checker re-derives. Under Competition no tape exists (one `Option` test
//!   per record).

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use ny_tensor::BoundedTensor;

use super::boxes::BoxLedger;
use super::cover::{
    CoverLedger, CoverLiteral, CoverRefusal, CoverSite, LeafEvidence, LeafPath, TreeCover,
    COVER_WORK_BUDGET,
};
use super::lane_proof::{TapeClose, TapeRecorder};
use super::token::{
    authorize_unsat, AuthorizeRefusal, ClassCover, GuardReport, TokenScope, UnsatToken,
};
use super::value::{CertReceipt, Certified};
use crate::beta_crown::branching::{GenBabConstraint, GraphNeuronConstraint, GraphSplitHistory};
use crate::beta_crown::state::GraphBetaState;
use crate::GraphNetwork;

/// Most literals one path frame may hold before it gives up (memory
/// admission: about 100 MB of literals). Past it the frame is UNBUILT.
pub(crate) const GRAPH_LEDGER_MAX_LITERALS: usize = 4 << 20;

/// Most leaves one frame may hold before it gives up.
pub(crate) const GRAPH_LEDGER_MAX_LEAVES: usize = 1 << 20;

/// Most box coordinates one box frame may hold before it gives up (64 MB).
pub(crate) const GRAPH_LEDGER_MAX_COORDINATES: usize = 8 << 20;

/// Longest the cover check of one frame may run, on top of the driver's own
/// deadline: the check runs after the search decided, so it must never cost
/// the verdict its publication window. Running out leaves the frame UNBUILT.
pub(crate) const GRAPH_COVER_CHECK_WALL: std::time::Duration = std::time::Duration::from_secs(2);

/// Most captured covers one run keeps (a capture past it keeps the first ones).
const CAPTURE_CAP: usize = 1024;

/// One objective in its LOWER-BOUND reading: certifying it on a region proves
/// `lower(row · y) > threshold` there.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphObjective {
    row: Box<[f32]>,
    threshold: f32,
}

impl GraphObjective {
    /// Normalize a lane objective. Upper-bound mode proves
    /// `upper(row · y) < threshold`, which is the claim
    /// `lower((−row) · y) > −threshold`; binary32 negation is exact. `None`
    /// for an empty row or any non-finite value.
    #[must_use]
    pub fn new(row: &[f32], threshold: f32, verify_upper: bool) -> Option<Self> {
        if row.is_empty() || !threshold.is_finite() || row.iter().any(|v| !v.is_finite()) {
            return None;
        }
        if verify_upper {
            Some(Self {
                row: row.iter().map(|v| -v).collect(),
                threshold: -threshold,
            })
        } else {
            Some(Self {
                row: row.into(),
                threshold,
            })
        }
    }

    /// Whether certifying `self` on a region excludes the unsafe atom whose
    /// excluding objective is `atom`: the same row (compared as values, so the
    /// two zeros agree) and a threshold at least `atom`'s
    /// (`row · y > self.threshold ≥ atom.threshold`).
    #[must_use]
    pub fn entails(&self, atom: &GraphObjective) -> bool {
        self.row.len() == atom.row.len()
            && self.row.iter().zip(atom.row.iter()).all(|(a, b)| a == b)
            && self.threshold >= atom.threshold
    }
}

/// The closing evidence of a graph-lane bound close: `Some` exactly when the
/// lane's closure rule (`domain_is_verified_for_mode`: finite, `lower ≤
/// upper`, strictly past the threshold in the verification direction) holds.
/// `pub(crate)`: it TRUSTS the caller to pass a bound pair the lane's
/// certified evaluator produced.
#[must_use]
pub(crate) fn graph_bound_evidence(
    verify_upper: bool,
    lower: f32,
    upper: f32,
    threshold: f32,
) -> Option<LeafEvidence> {
    if !lower.is_finite() || !upper.is_finite() || !threshold.is_finite() || lower > upper {
        return None;
    }
    // The sign of an f64 difference of two binary32 values is exact: it is
    // zero only when they are equal, and rounding never flips a sign.
    let margin = if verify_upper {
        f64::from(threshold) - f64::from(upper)
    } else {
        f64::from(lower) - f64::from(threshold)
    };
    (margin > 0.0).then(|| LeafEvidence::Bound(Certified::new(margin, CertReceipt::GraphLaneBound)))
}

/// The evidence of a region the lane's certified node bounds show EMPTY.
/// `pub(crate)`: it trusts the caller's `InfeasibleDomain` / empty-half-space
/// decision.
#[must_use]
pub(crate) fn graph_infeasible_evidence() -> LeafEvidence {
    LeafEvidence::LaneClose(CertReceipt::GraphLaneInfeasible)
}

/// The evidence of a region an input clip excluded. `pub(crate)`: it trusts
/// the caller's clip.
#[must_use]
pub(crate) fn graph_clip_evidence() -> LeafEvidence {
    LeafEvidence::LaneClose(CertReceipt::GraphLaneClip)
}

/// The evidence of an input box the lane's batched IBP prescreen verified.
/// `pub(crate)`: it trusts the caller's prescreen mask.
#[must_use]
pub(crate) fn graph_screen_evidence() -> LeafEvidence {
    LeafEvidence::LaneClose(CertReceipt::GraphLaneScreen)
}

/// #vit-v3-bankable: the closing evidence of a lower-bound-mode close whose
/// certificate is a LOWER bound alone (a root-terminal `Verified`, whose
/// closing root site records only the objective lowers it rests on): `Some`
/// exactly when `lower` is finite and strictly past `threshold`. The margin is
/// the exact `f64` difference of two binary32 values. `pub(crate)`: it trusts
/// the caller to pass a lower bound the lane's certified evaluator produced.
#[must_use]
pub(crate) fn graph_lower_evidence(lower: f32, threshold: f32) -> Option<LeafEvidence> {
    if !lower.is_finite() || !threshold.is_finite() {
        return None;
    }
    let margin = f64::from(lower) - f64::from(threshold);
    (margin > 0.0).then(|| LeafEvidence::Bound(Certified::new(margin, CertReceipt::GraphLaneBound)))
}

/// #vit-v3-bankable (M2c's certified-infeasibility rule): the evidence of a
/// region the lane's certified node bounds show EMPTY, WITH its slack
/// ([`CertReceipt::GraphLaneEmptiness`]): `Some` exactly when `slack` is finite
/// and strictly positive. An emptiness claim without a positive slack is not
/// evidence (the multi-objective frame then leaves itself UNBUILT; an error is
/// never a close). `pub(crate)`: it trusts the caller to pass a slack between
/// bounds the lane's certified evaluator produced.
#[must_use]
pub(crate) fn graph_emptiness_evidence(slack: f64) -> Option<LeafEvidence> {
    (slack.is_finite() && slack > 0.0)
        .then(|| LeafEvidence::Empty(Certified::new(slack, CertReceipt::GraphLaneEmptiness)))
}

/// Why a history cannot be written as a path (the frame is then UNBUILT).
type Unencodable = &'static str;

fn node_id(nodes: &HashMap<String, u32>, name: &str) -> Result<u32, Unencodable> {
    nodes.get(name).copied().ok_or("unknown_split_node")
}

fn relu_literal(
    nodes: &HashMap<String, u32>,
    c: &GraphNeuronConstraint,
) -> Result<CoverLiteral, Unencodable> {
    Ok(CoverLiteral {
        site: CoverSite::GraphRelu {
            node: node_id(nodes, &c.node_name)?,
            neuron: u32::try_from(c.neuron_idx).map_err(|_| "neuron_index_overflow")?,
        },
        positive: c.is_active,
    })
}

fn point_literal(
    nodes: &HashMap<String, u32>,
    c: &GenBabConstraint,
) -> Result<CoverLiteral, Unencodable> {
    if !c.split_point.is_finite() {
        return Err("non_finite_split_point");
    }
    Ok(CoverLiteral {
        site: CoverSite::GraphPoint {
            node: node_id(nodes, &c.node_name)?,
            input: u32::try_from(c.input_index.unwrap_or(0)).map_err(|_| "input_index_overflow")?,
            neuron: u32::try_from(c.neuron_idx).map_err(|_| "neuron_index_overflow")?,
            point_bits: c.split_point.to_bits(),
        },
        positive: c.is_upper_branch,
    })
}

/// A domain's region below the frame root, from the lane's OWN split history
/// (plus the split that creates a child, when the fact is about a child the
/// lane never built). Normalization (`inv_rms`) windows are not half-spaces of
/// one point, so a history carrying one is unencodable.
fn history_path(
    nodes: &HashMap<String, u32>,
    history: &GraphSplitHistory,
    extra: Option<&GraphNeuronConstraint>,
) -> Result<LeafPath, Unencodable> {
    if history.has_norm_inv_rms_constraints() {
        return Err("norm_split");
    }
    let mut lits = Vec::with_capacity(
        history.constraints.len() + history.genbab_constraints.len() + usize::from(extra.is_some()),
    );
    for c in history.constraints.iter().chain(extra) {
        lits.push(relu_literal(nodes, c)?);
    }
    for c in &history.genbab_constraints {
        lits.push(point_literal(nodes, c)?);
    }
    LeafPath::from_literals(lits).ok_or("contradictory_path")
}

fn box_f64(t: &BoundedTensor) -> (Vec<f64>, Vec<f64>) {
    (
        t.lower().iter().map(|&v| f64::from(v)).collect(),
        t.upper().iter().map(|&v| f64::from(v)).collect(),
    )
}

/// The objective a record was decided for (the lane's raw values).
#[derive(Debug, Clone, Copy)]
pub(crate) struct GraphRecordKey<'a> {
    /// The objective row the lane bounded.
    pub(crate) objective: &'a [f32],
    /// The lane's threshold.
    pub(crate) threshold: f32,
    /// The lane's direction (`verify_upper_bound`).
    pub(crate) verify_upper: bool,
}

enum FrameLedger {
    Paths {
        nodes: HashMap<String, u32>,
        ledger: CoverLedger,
    },
    Boxes {
        ledger: BoxLedger,
    },
}

struct GraphFrame {
    lane: &'static str,
    key_row: Box<[f32]>,
    key_threshold_bits: u32,
    key_upper: bool,
    objective: Option<GraphObjective>,
    root_lo: Box<[f64]>,
    root_hi: Box<[f64]>,
    ledger: FrameLedger,
    unbuilt: Option<String>,
    records: usize,
    literals: usize,
    /// B2.2: the write-only proof tape (Proof profile only).
    tape: Option<TapeRecorder>,
}

/// The tape form of a leaf's evidence: its close kind and margin hint, or
/// `None` for evidence a tape does not stream.
fn tape_close(evidence: Option<&LeafEvidence>) -> Option<(TapeClose, f64)> {
    match evidence? {
        LeafEvidence::Bound(c) => Some((TapeClose::Bound, c.value())),
        LeafEvidence::LaneClose(CertReceipt::GraphLaneInfeasible) => {
            Some((TapeClose::Empty, f64::NAN))
        }
        LeafEvidence::LaneClose(CertReceipt::GraphLaneClip) => Some((TapeClose::Clip, f64::NAN)),
        LeafEvidence::LaneClose(CertReceipt::GraphLaneScreen) => Some((TapeClose::Bound, f64::NAN)),
        LeafEvidence::LaneClose(_) | LeafEvidence::Empty(_) | LeafEvidence::Subtree(_) => None,
    }
}

impl GraphFrame {
    #[allow(clippy::too_many_arguments)]
    fn new(
        lane: &'static str,
        input: &BoundedTensor,
        objective: &[f32],
        threshold: f32,
        verify_upper: bool,
        ledger: FrameLedger,
        tape: Option<TapeRecorder>,
    ) -> Self {
        let (root_lo, root_hi) = box_f64(input);
        let normalized = GraphObjective::new(objective, threshold, verify_upper);
        // A box frame needs a finite root (the box gate compares finite
        // coordinates only); a path frame uses its root box only for the
        // capture's containment test, where an infinite endpoint is exact.
        let finite_root = root_lo.iter().chain(root_hi.iter()).all(|v| v.is_finite());
        let unbuilt = if normalized.is_none() {
            Some("non_finite_objective".to_string())
        } else if matches!(ledger, FrameLedger::Boxes { .. }) && !finite_root {
            Some("non_finite_root_box".to_string())
        } else {
            None
        };
        Self {
            lane,
            key_row: objective.into(),
            key_threshold_bits: threshold.to_bits(),
            key_upper: verify_upper,
            objective: normalized,
            root_lo: root_lo.into(),
            root_hi: root_hi.into(),
            ledger,
            unbuilt,
            records: 0,
            literals: 0,
            tape,
        }
    }

    fn give_up(&mut self, reason: impl Into<String>) {
        if self.unbuilt.is_none() {
            self.unbuilt = Some(reason.into());
        }
        // The tape keeps the closes recorded so far (true facts), but it can
        // no longer be complete.
        if let Some(tape) = &self.tape {
            tape.lose("frame_unbuilt");
        }
        // Free what the frame held: it can no longer be checked.
        self.literals = 0;
        match &mut self.ledger {
            FrameLedger::Paths { ledger, .. } => *ledger = CoverLedger::for_graph_lane(),
            FrameLedger::Boxes { ledger } => *ledger = BoxLedger::new(ledger.dims()),
        }
    }

    /// Whether a record is for this frame's objective; a foreign record
    /// poisons the frame.
    fn admit(&mut self, key: GraphRecordKey<'_>) -> bool {
        if self.unbuilt.is_some() {
            return false;
        }
        let same = key.verify_upper == self.key_upper
            && key.threshold.to_bits() == self.key_threshold_bits
            && key.objective.len() == self.key_row.len()
            && key
                .objective
                .iter()
                .zip(self.key_row.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits());
        if !same {
            self.give_up("foreign_record");
            return false;
        }
        self.records += 1;
        true
    }

    fn admit_memory(&mut self) {
        let over = match &self.ledger {
            FrameLedger::Paths { ledger, .. } => {
                ledger.recorded() > GRAPH_LEDGER_MAX_LEAVES
                    || self.literals > GRAPH_LEDGER_MAX_LITERALS
            }
            FrameLedger::Boxes { ledger } => {
                ledger.recorded() > GRAPH_LEDGER_MAX_LEAVES
                    || ledger.coordinate_count() > GRAPH_LEDGER_MAX_COORDINATES
            }
        };
        if over {
            self.give_up("ledger_memory_cap");
        }
    }

    fn settle(&mut self, recorded: Result<(), Unencodable>) {
        match recorded {
            Ok(()) => self.admit_memory(),
            Err(reason) => self.give_up(reason),
        }
    }

    fn path_leaf(
        &mut self,
        history: &GraphSplitHistory,
        extra: Option<&GraphNeuronConstraint>,
        evidence: Option<LeafEvidence>,
        beta: Option<&GraphBetaState>,
    ) {
        if let (Some(tape), Some((close, margin))) = (&self.tape, tape_close(evidence.as_ref())) {
            tape.note_splits(history, extra, beta, close, vec![vec![0]], margin, false);
        }
        let recorded = match &mut self.ledger {
            FrameLedger::Paths { nodes, ledger } => {
                history_path(nodes, history, extra).map(|path| {
                    self.literals = self.literals.saturating_add(path.depth());
                    ledger.leaf(Some(path), evidence);
                })
            }
            FrameLedger::Boxes { .. } => Err("path_record_on_box_frame"),
        };
        self.settle(recorded);
    }

    fn box_leaf(&mut self, input: &BoundedTensor, evidence: Option<LeafEvidence>) {
        let recorded = match &mut self.ledger {
            FrameLedger::Boxes { ledger } => {
                let (lo, hi) = box_f64(input);
                if let (Some(tape), Some((close, margin))) =
                    (&self.tape, tape_close(evidence.as_ref()))
                {
                    tape.note_box(&lo, &hi, close, vec![vec![0]], margin);
                }
                ledger.leaf(&lo, &hi, evidence);
                Ok(())
            }
            FrameLedger::Paths { .. } => Err("box_record_on_path_frame"),
        };
        self.settle(recorded);
    }

    fn box_clip(&mut self, pre: &BoundedTensor, post: &BoundedTensor) {
        let recorded = match &mut self.ledger {
            FrameLedger::Boxes { ledger } => {
                let (pre_lo, pre_hi) = box_f64(pre);
                let (post_lo, post_hi) = box_f64(post);
                if let Some(tape) = &self.tape {
                    tape.note_clip(&pre_lo, &pre_hi, &post_lo, &post_hi, &[vec![0]]);
                }
                ledger.clip(&pre_lo, &pre_hi, &post_lo, &post_hi, &graph_clip_evidence());
                Ok(())
            }
            FrameLedger::Paths { .. } => Err("box_record_on_path_frame"),
        };
        self.settle(recorded);
    }

    /// Check the frame (the driver decided `Verified`).
    fn check(self, deadline: Option<Instant>) -> GraphCoverVerdict {
        // B2.2: the tape follows the frame's outcome. Proved: complete.
        // Unbuilt: the lane's `Verified` stands on the legacy door and the
        // tape's closes are still true facts (streamed, never complete).
        // Refused: the lane withdraws `Verified`; nothing is deposited.
        let tape = self.tape;
        if let Some(reason) = self.unbuilt {
            if let Some(tape) = &tape {
                tape.deposit(false);
            }
            return GraphCoverVerdict::Unbuilt(reason);
        }
        let Some(objective) = self.objective else {
            if let Some(tape) = &tape {
                tape.deposit(false);
            }
            return GraphCoverVerdict::Unbuilt("non_finite_objective".to_string());
        };
        let now = Instant::now();
        // An unrepresentable wall is an expired one (fail-closed to UNBUILT).
        let wall = now.checked_add(GRAPH_COVER_CHECK_WALL).unwrap_or(now);
        let deadline = Some(deadline.map_or(wall, |d| d.min(wall)));
        let proved = match &self.ledger {
            FrameLedger::Paths { ledger, .. } => {
                ledger.prove_within(&[0], COVER_WORK_BUDGET, deadline)
            }
            FrameLedger::Boxes { ledger } => {
                ledger.prove(&self.root_lo, &self.root_hi, &[0], deadline)
            }
        };
        match proved {
            Ok(cover) => {
                let proof = GraphInvocationCover {
                    lane: self.lane,
                    root_lo: self.root_lo,
                    root_hi: self.root_hi,
                    objective,
                    cover,
                };
                let summary = GraphFrameSummary {
                    leaves: proof.cover.leaves(),
                    lane_leaves: proof.cover.lane_leaves(),
                    max_depth: proof.cover.max_depth(),
                    min_margin: proof.cover.min_margin(),
                    records: self.records,
                };
                deposit_graph_cover(proof);
                if let Some(tape) = &tape {
                    tape.deposit(true);
                }
                GraphCoverVerdict::Proved(summary)
            }
            Err(CoverRefusal::Budget { work }) => {
                if let Some(tape) = &tape {
                    tape.deposit(false);
                }
                GraphCoverVerdict::Unbuilt(format!("cover_check_budget work={work}"))
            }
            Err(refusal) => GraphCoverVerdict::Refused(refusal),
        }
    }
}

/// What a proved frame rested on (the lane's observable line).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct GraphFrameSummary {
    /// Certified leaves in the cover.
    pub(crate) leaves: usize,
    /// Of which margin-free lane closes (infeasible / clip / screen).
    pub(crate) lane_leaves: usize,
    /// Deepest path, or deepest box cut level.
    pub(crate) max_depth: usize,
    /// Smallest certified margin (`+inf` when every leaf is a lane close).
    pub(crate) min_margin: f64,
    /// Facts recorded (including tolerated extras).
    pub(crate) records: usize,
}

/// The outcome of a frame at the driver's exit.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum GraphCoverVerdict {
    /// This scope armed no frame (nested arm, or the result was not
    /// `Verified`): nothing to do.
    NotArmed,
    /// The frame could not be checked: the lane's `Verified` stands on the
    /// counted legacy door.
    Unbuilt(String),
    /// The cover gate REFUSED: the lane's certified leaves do not cover its
    /// root. The driver must not keep `Verified`.
    Refused(CoverRefusal),
    /// Proved, and deposited into the capture.
    Proved(GraphFrameSummary),
}

/// The frame a driver armed: one objective (a path or box frame), or — for
/// the multi-objective driver (#vit-v3-bankable) — one path ledger per
/// objective ROW over a shared node numbering ([`RowsFrame`]).
///
/// Both frames are boxed: they are 488 and 200 bytes, and the slot holds one
/// for the life of a scope (clippy::large_enum_variant, workspace threshold 64).
enum ArmedFrame {
    Single(Box<GraphFrame>),
    Rows(Box<RowsFrame>),
}

impl ArmedFrame {
    fn give_up(&mut self, reason: &str) {
        match self {
            Self::Single(frame) => frame.give_up(reason),
            Self::Rows(frame) => frame.give_up(reason),
        }
    }
}

/// The per-verifier recorder. Idle (one relaxed atomic load per recording
/// site) unless a driver armed a frame.
#[derive(Default)]
pub(crate) struct GraphCoverSlot {
    armed: AtomicBool,
    frame: Mutex<Option<ArmedFrame>>,
}

/// An armed frame's lifetime. Dropping it disarms; [`Self::finish`] checks.
pub(crate) struct GraphCoverScope<'a> {
    slot: &'a GraphCoverSlot,
    owner: bool,
}

impl GraphCoverSlot {
    fn lock(&self) -> std::sync::MutexGuard<'_, Option<ArmedFrame>> {
        self.frame
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Install `frame`; `true` when this call owns it. A second driver on the
    /// same verifier while a frame is armed (of either kind) would record into
    /// the outer frame: the outer frame is poisoned and the nested driver gets
    /// no frame of its own.
    fn arm_frame(&self, frame: ArmedFrame) -> bool {
        let mut guard = self.lock();
        if let Some(outer) = guard.as_mut() {
            outer.give_up("nested_driver");
            return false;
        }
        *guard = Some(frame);
        self.armed.store(true, Ordering::Release);
        true
    }

    fn arm(&self, frame: GraphFrame) -> GraphCoverScope<'_> {
        GraphCoverScope {
            slot: self,
            owner: self.arm_frame(ArmedFrame::Single(Box::new(frame))),
        }
    }

    /// Arm a split-tree (path) frame: ReLU / GenBaB splits of `graph` below
    /// the root box `input`, for one objective.
    pub(crate) fn arm_paths(
        &self,
        lane: &'static str,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        objective: &[f32],
        threshold: f32,
        verify_upper: bool,
    ) -> GraphCoverScope<'_> {
        let tape =
            TapeRecorder::for_graph(lane, graph, input, &[(objective, threshold, verify_upper)]);
        self.arm_paths_inner(
            lane,
            graph.node_order.iter().map(String::as_str),
            input,
            objective,
            threshold,
            verify_upper,
            tape,
        )
    }

    /// [`Self::arm_paths`] over an explicit node-name list (the frame numbers
    /// the names in order; a repeated name keeps its first id). Test face:
    /// production arms through [`Self::arm_paths`].
    #[cfg(test)]
    pub(crate) fn arm_paths_named<'n>(
        &self,
        lane: &'static str,
        names: impl IntoIterator<Item = &'n str> + Clone,
        input: &BoundedTensor,
        objective: &[f32],
        threshold: f32,
        verify_upper: bool,
    ) -> GraphCoverScope<'_> {
        let tape = TapeRecorder::for_names(
            lane,
            names.clone(),
            input,
            &[(objective, threshold, verify_upper)],
        );
        self.arm_paths_inner(lane, names, input, objective, threshold, verify_upper, tape)
    }

    #[allow(clippy::too_many_arguments)]
    fn arm_paths_inner<'n>(
        &self,
        lane: &'static str,
        names: impl IntoIterator<Item = &'n str>,
        input: &BoundedTensor,
        objective: &[f32],
        threshold: f32,
        verify_upper: bool,
        tape: Option<TapeRecorder>,
    ) -> GraphCoverScope<'_> {
        let mut nodes: HashMap<String, u32> = HashMap::new();
        for (id, name) in names.into_iter().enumerate() {
            let id = u32::try_from(id).unwrap_or(u32::MAX);
            nodes.entry(name.to_string()).or_insert(id);
        }
        self.arm(GraphFrame::new(
            lane,
            input,
            objective,
            threshold,
            verify_upper,
            FrameLedger::Paths {
                nodes,
                ledger: CoverLedger::for_graph_lane(),
            },
            tape,
        ))
    }

    /// Arm an input-split (box) frame below the root box `input`.
    pub(crate) fn arm_boxes(
        &self,
        lane: &'static str,
        input: &BoundedTensor,
        objective: &[f32],
        threshold: f32,
        verify_upper: bool,
    ) -> GraphCoverScope<'_> {
        let tape = TapeRecorder::for_names(
            lane,
            std::iter::empty(),
            input,
            &[(objective, threshold, verify_upper)],
        );
        self.arm(GraphFrame::new(
            lane,
            input,
            objective,
            threshold,
            verify_upper,
            FrameLedger::Boxes {
                ledger: BoxLedger::new(input.len()),
            },
            tape,
        ))
    }

    fn with_frame(&self, f: impl FnOnce(&mut GraphFrame)) {
        if !self.armed.load(Ordering::Acquire) {
            return;
        }
        let mut guard = self.lock();
        match guard.as_mut() {
            Some(ArmedFrame::Single(frame)) => f(frame),
            // A one-objective record on a multi-objective frame: the frame
            // cannot tell which row it closes, so it can no longer be checked.
            Some(ArmedFrame::Rows(frame)) => frame.give_up("foreign_record"),
            None => {}
        }
    }

    /// Whether a frame is armed (lets a caller skip preparing a record).
    pub(crate) fn is_armed(&self) -> bool {
        self.armed.load(Ordering::Acquire)
    }

    /// A domain (split history `history`) closed by the bound pair
    /// `(lower, upper)` the lane's closure rule accepted.
    pub(crate) fn note_bound(
        &self,
        key: GraphRecordKey<'_>,
        history: &GraphSplitHistory,
        beta: Option<&GraphBetaState>,
        lower: f32,
        upper: f32,
    ) {
        self.with_frame(|frame| {
            if frame.admit(key) {
                let evidence = graph_bound_evidence(key.verify_upper, lower, upper, key.threshold);
                frame.path_leaf(history, None, evidence, beta);
            }
        });
    }

    /// A domain (split history `history`, plus the split `extra` that makes
    /// the child when the fact is about a child the lane never built) whose
    /// region the lane's certified node bounds show EMPTY.
    pub(crate) fn note_infeasible(
        &self,
        key: GraphRecordKey<'_>,
        history: &GraphSplitHistory,
        extra: Option<&GraphNeuronConstraint>,
    ) {
        self.with_frame(|frame| {
            if frame.admit(key) {
                frame.path_leaf(history, extra, Some(graph_infeasible_evidence()), None);
            }
        });
    }

    /// An input box closed by the bound pair `(lower, upper)`.
    pub(crate) fn note_box_bound(
        &self,
        key: GraphRecordKey<'_>,
        input: &BoundedTensor,
        lower: f32,
        upper: f32,
    ) {
        self.with_frame(|frame| {
            if frame.admit(key) {
                let evidence = graph_bound_evidence(key.verify_upper, lower, upper, key.threshold);
                frame.box_leaf(input, evidence);
            }
        });
    }

    /// An input box the lane's clip closed whole (the clip verified it).
    pub(crate) fn note_box_clip_closed(&self, key: GraphRecordKey<'_>, input: &BoundedTensor) {
        self.with_frame(|frame| {
            if frame.admit(key) {
                frame.box_leaf(input, Some(graph_clip_evidence()));
            }
        });
    }

    /// An input box the lane's batched IBP prescreen verified.
    pub(crate) fn note_box_screened(&self, key: GraphRecordKey<'_>, input: &BoundedTensor) {
        self.with_frame(|frame| {
            if frame.admit(key) {
                frame.box_leaf(input, Some(graph_screen_evidence()));
            }
        });
    }

    /// An input clip of `pre` to `post`: the part of `pre` outside `post` is
    /// closed by the clip.
    pub(crate) fn note_box_clip(
        &self,
        key: GraphRecordKey<'_>,
        pre: &BoundedTensor,
        post: &BoundedTensor,
    ) {
        self.with_frame(|frame| {
            if frame.admit(key) {
                frame.box_clip(pre, post);
            }
        });
    }

    /// A close the ledger cannot express: the frame is UNBUILT (its
    /// `Verified` publishes through the counted legacy door, never refused).
    pub(crate) fn note_unbuildable(&self, reason: &'static str) {
        if !self.armed.load(Ordering::Acquire) {
            return;
        }
        if let Some(frame) = self.lock().as_mut() {
            frame.give_up(reason);
        }
    }

    fn take(&self) -> Option<ArmedFrame> {
        let mut guard = self.lock();
        self.armed.store(false, Ordering::Release);
        guard.take()
    }
}

impl GraphCoverScope<'_> {
    /// Close the frame. When the driver's result is `verified`, check it (the
    /// check never runs past `deadline` nor past [`GRAPH_COVER_CHECK_WALL`]);
    /// otherwise discard it.
    pub(crate) fn finish(self, verified: bool, deadline: Option<Instant>) -> GraphCoverVerdict {
        if !self.owner {
            return GraphCoverVerdict::NotArmed;
        }
        let Some(frame) = self.slot.take() else {
            return GraphCoverVerdict::NotArmed;
        };
        if !verified {
            return GraphCoverVerdict::NotArmed;
        }
        match frame {
            ArmedFrame::Single(frame) => frame.check(deadline),
            // Unreachable by construction (a one-objective scope owns a
            // one-objective frame); fail to UNBUILT, never to a proof.
            ArmedFrame::Rows(_) => GraphCoverVerdict::Unbuilt("frame_kind_mismatch".to_string()),
        }
    }
}

impl Drop for GraphCoverScope<'_> {
    fn drop(&mut self) {
        if self.owner {
            // Idempotent: `finish` already took the frame on its path.
            drop(self.slot.take());
        }
    }
}

// ---------------------------------------------------------------------------
// #vit-v3-bankable: the multi-objective (per-ROW) frame
// ---------------------------------------------------------------------------

/// Most objective rows one rows frame may carry (a vit property has 9).
pub(crate) const GRAPH_ROWS_MAX: usize = 4096;

/// One objective row of a [`RowsFrame`]: its threshold, its lower-bound
/// normalization (the capture's objective), and its own overlap-tolerant
/// ledger.
struct RowLedger {
    threshold: f32,
    objective: Option<GraphObjective>,
    ledger: CoverLedger,
}

/// The multi-objective graph BaB frame (lower-bound mode, DISJUNCTIVE
/// semantics: the property is safe on a region only when EVERY row is
/// certified there).
///
/// A row is closed on a region when the lane certified THAT row there, so the
/// frame keeps one path ledger per row over one node numbering, and a record
/// is a FACT about one region for the rows it proves:
///
/// * a bound record (`bounds[j]` for every row) adds a leaf to row `j`'s
///   ledger exactly when `bounds[j]` passes the lane's closure rule
///   ([`graph_bound_evidence`]); a row it does not prove gets nothing (it may
///   be closed below or above this region, which its own ledger decides);
/// * an emptiness record with a positive certified slack adds a leaf to EVERY
///   row's ledger ([`graph_emptiness_evidence`]); without one the frame gives
///   up (an error is never a close);
/// * a root-terminal record carries lowers only ([`graph_lower_evidence`]).
///
/// The frame is proved when EVERY row's ledger covers the root; each row's
/// cover is then its own [`GraphInvocationCover`], so the instance door treats
/// a vit property (one single-atom clause per row) exactly like any other
/// graph cover. Anything the frame cannot express leaves it UNBUILT, never
/// refused.
struct RowsFrame {
    lane: &'static str,
    nodes: HashMap<String, u32>,
    names: Vec<String>,
    rows: Vec<RowLedger>,
    root_lo: Box<[f64]>,
    root_hi: Box<[f64]>,
    unbuilt: Option<String>,
    records: usize,
    empty_records: usize,
    leaves: usize,
    literals: usize,
}

impl RowsFrame {
    fn new<'n>(
        lane: &'static str,
        names: impl IntoIterator<Item = &'n str>,
        input: &BoundedTensor,
        objectives: &[Vec<f32>],
        thresholds: &[f32],
    ) -> Self {
        let mut nodes: HashMap<String, u32> = HashMap::new();
        let mut id_names: Vec<String> = Vec::new();
        for name in names {
            if nodes.contains_key(name) {
                continue;
            }
            let id = u32::try_from(id_names.len()).unwrap_or(u32::MAX);
            nodes.insert(name.to_string(), id);
            id_names.push(name.to_string());
        }
        let (root_lo, root_hi) = box_f64(input);
        let mut unbuilt = None;
        if objectives.is_empty() || objectives.len() > GRAPH_ROWS_MAX {
            unbuilt = Some("row_count".to_string());
        } else if objectives.len() != thresholds.len() {
            unbuilt = Some("row_threshold_mismatch".to_string());
        }
        let rows: Vec<RowLedger> = objectives
            .iter()
            .zip(thresholds.iter())
            .map(|(row, &threshold)| RowLedger {
                threshold,
                objective: GraphObjective::new(row, threshold, false),
                ledger: CoverLedger::for_graph_lane(),
            })
            .collect();
        if unbuilt.is_none() && rows.iter().any(|r| r.objective.is_none()) {
            unbuilt = Some("non_finite_objective".to_string());
        }
        Self {
            lane,
            nodes,
            names: id_names,
            rows,
            root_lo: root_lo.into(),
            root_hi: root_hi.into(),
            unbuilt,
            records: 0,
            empty_records: 0,
            leaves: 0,
            literals: 0,
        }
    }

    fn give_up(&mut self, reason: impl Into<String>) {
        if self.unbuilt.is_none() {
            self.unbuilt = Some(reason.into());
        }
        // Free what the frame held: it can no longer be checked.
        for row in &mut self.rows {
            row.ledger = CoverLedger::for_graph_lane();
        }
        self.leaves = 0;
        self.literals = 0;
    }

    fn admit_memory(&mut self) {
        if self.leaves > GRAPH_LEDGER_MAX_LEAVES || self.literals > GRAPH_LEDGER_MAX_LITERALS {
            self.give_up("ledger_memory_cap");
        }
    }

    fn path(
        &mut self,
        history: &GraphSplitHistory,
        extra: Option<&GraphNeuronConstraint>,
    ) -> Option<LeafPath> {
        if self.unbuilt.is_some() {
            return None;
        }
        match history_path(&self.nodes, history, extra) {
            Ok(path) => Some(path),
            Err(reason) => {
                self.give_up(reason);
                None
            }
        }
    }

    fn push(&mut self, row: usize, path: &LeafPath, evidence: LeafEvidence) {
        self.leaves = self.leaves.saturating_add(1);
        self.literals = self.literals.saturating_add(path.depth());
        self.rows[row]
            .ledger
            .leaf(Some(path.clone()), Some(evidence));
    }

    fn record_bounds(&mut self, history: &GraphSplitHistory, bounds: &[(f32, f32)]) {
        if self.unbuilt.is_some() {
            return;
        }
        if bounds.len() != self.rows.len() {
            self.give_up("row_count_mismatch");
            return;
        }
        let Some(path) = self.path(history, None) else {
            return;
        };
        self.records += 1;
        for (row, &(lower, upper)) in bounds.iter().enumerate() {
            if let Some(evidence) =
                graph_bound_evidence(false, lower, upper, self.rows[row].threshold)
            {
                self.push(row, &path, evidence);
            }
        }
        self.admit_memory();
    }

    fn record_lowers(&mut self, history: &GraphSplitHistory, lowers: &[f32]) {
        if self.unbuilt.is_some() {
            return;
        }
        if lowers.len() != self.rows.len() {
            self.give_up("row_count_mismatch");
            return;
        }
        let Some(path) = self.path(history, None) else {
            return;
        };
        self.records += 1;
        for (row, &lower) in lowers.iter().enumerate() {
            if let Some(evidence) = graph_lower_evidence(lower, self.rows[row].threshold) {
                self.push(row, &path, evidence);
            }
        }
        self.admit_memory();
    }

    fn record_empty(
        &mut self,
        history: &GraphSplitHistory,
        extra: Option<&GraphNeuronConstraint>,
        slack: Option<f64>,
    ) {
        if self.unbuilt.is_some() {
            return;
        }
        let Some(evidence) = slack.and_then(graph_emptiness_evidence) else {
            self.give_up("uncertified_emptiness");
            return;
        };
        let Some(path) = self.path(history, extra) else {
            return;
        };
        self.records += 1;
        self.empty_records += 1;
        for row in 0..self.rows.len() {
            self.push(row, &path, evidence.clone());
        }
        self.admit_memory();
    }

    fn region_literal(&self, lit: &CoverLiteral) -> RegionLiteral {
        let name = |node: u32| {
            usize::try_from(node)
                .ok()
                .and_then(|i| self.names.get(i))
                .cloned()
                .unwrap_or_default()
        };
        match lit.site {
            CoverSite::GraphRelu { node, neuron } => RegionLiteral::Relu {
                node: name(node),
                neuron: usize::try_from(neuron).unwrap_or(usize::MAX),
                active: lit.positive,
            },
            CoverSite::GraphPoint { node, .. } => RegionLiteral::Other { node: name(node) },
            CoverSite::Trunk { .. } | CoverSite::Head { .. } => RegionLiteral::Other {
                node: String::new(),
            },
        }
    }

    /// Check every row (the driver decided `Verified`).
    fn check(self, deadline: Option<Instant>) -> GraphRowsVerdict {
        if let Some(reason) = &self.unbuilt {
            return GraphRowsVerdict::Unbuilt(reason.clone());
        }
        let now = Instant::now();
        // An unrepresentable wall is an expired one (fail-closed to UNBUILT).
        let wall = now.checked_add(GRAPH_COVER_CHECK_WALL).unwrap_or(now);
        let deadline = Some(deadline.map_or(wall, |d| d.min(wall)));
        let mut covers = Vec::with_capacity(self.rows.len());
        let mut by_path: HashMap<LeafPath, (Vec<usize>, bool)> = HashMap::new();
        let mut summary = GraphRowsSummary {
            rows: self.rows.len(),
            leaves: 0,
            min_row_leaves: usize::MAX,
            empty_leaves: 0,
            max_depth: 0,
            min_margin: f64::INFINITY,
            records: self.records,
            empty_records: self.empty_records,
            regions: 0,
        };
        for (row, slot) in self.rows.iter().enumerate() {
            let Some(objective) = slot.objective.clone() else {
                return GraphRowsVerdict::Unbuilt("non_finite_objective".to_string());
            };
            match slot
                .ledger
                .prove_with_witnesses(&[0], COVER_WORK_BUDGET, deadline)
            {
                Ok((cover, witnesses)) => {
                    summary.leaves += cover.leaves();
                    summary.min_row_leaves = summary.min_row_leaves.min(cover.leaves());
                    summary.empty_leaves += cover.empty_leaves();
                    summary.max_depth = summary.max_depth.max(cover.max_depth());
                    summary.min_margin = summary.min_margin.min(cover.min_margin());
                    for index in witnesses {
                        let Some(path) = slot.ledger.witness_path(index) else {
                            return GraphRowsVerdict::Unbuilt("witness_index".to_string());
                        };
                        let empty = slot.ledger.witness_is_empty(index);
                        let entry = by_path.entry(path.clone()).or_insert((Vec::new(), true));
                        entry.0.push(row);
                        entry.1 &= empty;
                    }
                    covers.push(GraphInvocationCover {
                        lane: self.lane,
                        root_lo: self.root_lo.clone(),
                        root_hi: self.root_hi.clone(),
                        objective,
                        cover,
                    });
                }
                Err(CoverRefusal::Budget { work }) => {
                    return GraphRowsVerdict::Unbuilt(format!(
                        "cover_check_budget row={row} work={work}"
                    ));
                }
                Err(refusal) => return GraphRowsVerdict::Refused { row, refusal },
            }
        }
        let mut regions: Vec<RowsRegion> = by_path
            .into_iter()
            .map(|(path, (mut rows, empty))| {
                rows.sort_unstable();
                rows.dedup();
                RowsRegion {
                    literals: path
                        .literals()
                        .iter()
                        .map(|lit| self.region_literal(lit))
                        .collect(),
                    rows,
                    empty,
                }
            })
            .collect();
        // Deterministic order: shallowest first, then by literal list.
        regions.sort_by(|a, b| {
            a.literals
                .len()
                .cmp(&b.literals.len())
                .then_with(|| a.literals.cmp(&b.literals))
                .then_with(|| a.rows.cmp(&b.rows))
        });
        summary.regions = regions.len();
        if summary.min_row_leaves == usize::MAX {
            summary.min_row_leaves = 0;
        }
        GraphRowsVerdict::Proved(Box::new(ProvedRowsCover {
            covers,
            regions,
            summary,
        }))
    }
}

/// One literal of a witness region, in the lane's own vocabulary.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum RegionLiteral {
    /// A ReLU/Sign split at zero: `active` is the `z >= 0` half-space of the
    /// pre-activation of neuron `neuron` of activation node `node`.
    Relu {
        /// The activation node's name.
        node: String,
        /// Flat neuron index.
        neuron: usize,
        /// Which closed half-space.
        active: bool,
    },
    /// A literal the multi-objective lane cannot rebuild by `with_constraint`
    /// (a GenBaB point split): a region carrying one cannot be replayed.
    Other {
        /// The node's name (empty when the site names none).
        node: String,
    },
}

/// A witness region of a proved rows frame: its literals and the rows whose
/// cover it closes (`empty` when every one of those closes was an emptiness
/// slack).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RowsRegion {
    /// The region's split literals (canonical order).
    pub(crate) literals: Vec<RegionLiteral>,
    /// Rows this region closes in their covers.
    pub(crate) rows: Vec<usize>,
    /// Whether every close recorded for it was an emptiness slack.
    pub(crate) empty: bool,
}

/// What a proved rows frame rested on (the lane's observable line).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct GraphRowsSummary {
    /// Objective rows.
    pub(crate) rows: usize,
    /// Leaves over every row's cover.
    pub(crate) leaves: usize,
    /// The smallest row cover.
    pub(crate) min_row_leaves: usize,
    /// Of `leaves`, emptiness-slack leaves.
    pub(crate) empty_leaves: usize,
    /// Deepest leaf of any row's cover.
    pub(crate) max_depth: usize,
    /// Smallest certified margin (`+inf` when every leaf is an emptiness close).
    pub(crate) min_margin: f64,
    /// Facts recorded (including tolerated extras).
    pub(crate) records: usize,
    /// Of `records`, emptiness records.
    pub(crate) empty_records: usize,
    /// Distinct witness regions over all rows.
    pub(crate) regions: usize,
}

/// A proved rows frame, not yet deposited: the per-row covers and the witness
/// regions a certify-once replay re-certifies before the driver keeps its
/// `Verified`. Private fields; built only by a frame whose every row passed
/// its gate.
#[derive(Debug)]
pub(crate) struct ProvedRowsCover {
    covers: Vec<GraphInvocationCover>,
    regions: Vec<RowsRegion>,
    summary: GraphRowsSummary,
}

impl ProvedRowsCover {
    /// The proof's summary.
    pub(crate) fn summary(&self) -> GraphRowsSummary {
        self.summary
    }

    /// The witness regions (shallowest first).
    pub(crate) fn regions(&self) -> &[RowsRegion] {
        &self.regions
    }

    /// Deposit every row's cover into the process capture (the publication
    /// path's instance door then entails one single-atom clause per row).
    pub(crate) fn deposit(self) -> GraphRowsSummary {
        for cover in self.covers {
            deposit_graph_cover(cover);
        }
        self.summary
    }
}

/// The outcome of a rows frame at the driver's exit.
#[derive(Debug)]
pub(crate) enum GraphRowsVerdict {
    /// This scope armed no frame, or the result was not `Verified`.
    NotArmed,
    /// The frame could not be checked: the lane's `Verified` stands on the
    /// counted legacy door.
    Unbuilt(String),
    /// Row `row`'s certified leaves do not cover the root: the driver must not
    /// keep `Verified`.
    Refused {
        /// The row whose cover failed.
        row: usize,
        /// Why.
        refusal: CoverRefusal,
    },
    /// Every row proved; not yet deposited. Boxed: 120 bytes against the
    /// other variants' 32 (clippy::large_enum_variant, workspace threshold 64).
    Proved(Box<ProvedRowsCover>),
}

/// An armed rows frame's lifetime. Dropping it disarms; [`Self::finish`]
/// checks.
pub(crate) struct GraphRowsScope<'a> {
    slot: &'a GraphCoverSlot,
    owner: bool,
}

impl GraphRowsScope<'_> {
    /// Whether this scope owns an armed frame (a nested arm owns none).
    pub(crate) fn owns_frame(&self) -> bool {
        self.owner
    }

    /// Close the frame. When the driver's result is `verified`, check every
    /// row (never past `deadline` nor past [`GRAPH_COVER_CHECK_WALL`]);
    /// otherwise discard it.
    pub(crate) fn finish(self, verified: bool, deadline: Option<Instant>) -> GraphRowsVerdict {
        if !self.owner {
            return GraphRowsVerdict::NotArmed;
        }
        let Some(frame) = self.slot.take() else {
            return GraphRowsVerdict::NotArmed;
        };
        if !verified {
            return GraphRowsVerdict::NotArmed;
        }
        match frame {
            ArmedFrame::Rows(frame) => frame.check(deadline),
            ArmedFrame::Single(_) => GraphRowsVerdict::Unbuilt("frame_kind_mismatch".to_string()),
        }
    }
}

impl Drop for GraphRowsScope<'_> {
    fn drop(&mut self) {
        if self.owner {
            drop(self.slot.take());
        }
    }
}

impl GraphCoverSlot {
    /// #vit-v3-bankable: arm a multi-objective (per-row) path frame for the
    /// graph multi-objective BaB driver: ReLU splits of the graph whose nodes
    /// are `names` (numbered in order), below the root box `input`, one ledger
    /// per `(objectives[j], thresholds[j])` in lower-bound mode.
    pub(crate) fn arm_rows<'n>(
        &self,
        lane: &'static str,
        names: impl IntoIterator<Item = &'n str>,
        input: &BoundedTensor,
        objectives: &[Vec<f32>],
        thresholds: &[f32],
    ) -> GraphRowsScope<'_> {
        GraphRowsScope {
            slot: self,
            owner: self.arm_frame(ArmedFrame::Rows(Box::new(RowsFrame::new(
                lane, names, input, objectives, thresholds,
            )))),
        }
    }

    fn with_rows(&self, f: impl FnOnce(&mut RowsFrame)) {
        if !self.armed.load(Ordering::Acquire) {
            return;
        }
        let mut guard = self.lock();
        match guard.as_mut() {
            Some(ArmedFrame::Rows(frame)) => f(frame),
            // A multi-objective record on a one-objective frame is foreign.
            Some(ArmedFrame::Single(frame)) => frame.give_up("foreign_record"),
            None => {}
        }
    }

    /// Whether a multi-objective rows frame is armed (tests).
    #[cfg(test)]
    pub(crate) fn rows_armed(&self) -> bool {
        if !self.armed.load(Ordering::Acquire) {
            return false;
        }
        matches!(self.lock().as_ref(), Some(ArmedFrame::Rows(_)))
    }

    /// A domain (split history `history`) whose objective bound pairs are
    /// `bounds` (one per row): each row the pair proves gets a leaf.
    pub(crate) fn note_rows_bounds(&self, history: &GraphSplitHistory, bounds: &[(f32, f32)]) {
        self.with_rows(|frame| frame.record_bounds(history, bounds));
    }

    /// A root-terminal close resting on objective LOWERS only.
    pub(crate) fn note_rows_lowers(&self, history: &GraphSplitHistory, lowers: &[f32]) {
        self.with_rows(|frame| frame.record_lowers(history, lowers));
    }

    /// A region (the domain `history`, narrowed by `extra` for a child the
    /// lane never built) the lane's certified node bounds show EMPTY, with its
    /// certified `slack` (`None` = no slack survived: the frame gives up).
    pub(crate) fn note_rows_empty(
        &self,
        history: &GraphSplitHistory,
        extra: Option<&GraphNeuronConstraint>,
        slack: Option<f64>,
    ) {
        self.with_rows(|frame| frame.record_empty(history, extra, slack));
    }
}

/// What one ledgered graph BaB invocation proved: its gate-checked cover, the
/// root box it covers and the objective every leaf certified. Private fields;
/// built only by a frame whose gate passed.
#[derive(Debug, Clone)]
pub struct GraphInvocationCover {
    lane: &'static str,
    root_lo: Box<[f64]>,
    root_hi: Box<[f64]>,
    objective: GraphObjective,
    cover: TreeCover,
}

impl GraphInvocationCover {
    /// The driver that proved it.
    #[must_use]
    pub fn lane(&self) -> &'static str {
        self.lane
    }

    /// Certified leaves in the cover.
    #[must_use]
    pub fn leaves(&self) -> usize {
        self.cover.leaves()
    }

    /// Of which margin-free lane closes.
    #[must_use]
    pub fn lane_leaves(&self) -> usize {
        self.cover.lane_leaves()
    }

    /// Smallest certified margin the cover rests on.
    #[must_use]
    pub fn min_margin(&self) -> f64 {
        self.cover.min_margin()
    }

    /// Whether the proved root box contains the box `[lo, hi]` (exact `f64`
    /// comparisons; the dimension counts must agree).
    fn contains_box(&self, lo: &[f64], hi: &[f64]) -> bool {
        self.root_lo.len() == lo.len()
            && self.root_hi.len() == hi.len()
            && (0..lo.len()).all(|d| self.root_lo[d] <= lo[d] && self.root_hi[d] >= hi[d])
    }
}

/// The process capture: `None` until an entry point arms it.
static CAPTURE: Mutex<Option<Vec<GraphInvocationCover>>> = Mutex::new(None);

/// Serializes the unit tests that arm the PROCESS capture (the certify tests
/// and the drivers' end-to-end ledger tests share it).
#[cfg(test)]
pub(crate) static CAPTURE_TEST_LOCK: Mutex<()> = Mutex::new(());

fn capture_lock() -> std::sync::MutexGuard<'static, Option<Vec<GraphInvocationCover>>> {
    CAPTURE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn deposit_graph_cover(cover: GraphInvocationCover) {
    let mut guard = capture_lock();
    if let Some(entries) = guard.as_mut() {
        if entries.len() < CAPTURE_CAP {
            entries.push(cover);
        }
    }
}

/// Arm (and empty) the process capture. The publication path calls this
/// before the in-process verifier runs; with no capture armed (library
/// callers, unit tests of the drivers) a proved frame deposits nothing.
pub fn reset_graph_cover_capture() {
    *capture_lock() = Some(Vec::new());
}

/// Disarm the capture and return what it holds.
#[must_use]
pub fn take_graph_cover_capture() -> GraphCoverCapture {
    GraphCoverCapture {
        entries: capture_lock().take().unwrap_or_default(),
    }
}

/// The covers one run's ledgered graph invocations proved.
#[derive(Debug, Clone, Default)]
pub struct GraphCoverCapture {
    entries: Vec<GraphInvocationCover>,
}

impl GraphCoverCapture {
    /// Captured covers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// No cover was captured.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The most leaves any captured cover for exactly `objective` rests on
    /// (an observable for the lane line and the tests), or `None`.
    #[must_use]
    pub fn leaves_for(&self, objective: &GraphObjective) -> Option<usize> {
        self.entries
            .iter()
            .filter(|entry| entry.objective == *objective)
            .map(|entry| entry.cover.leaves())
            .max()
    }
}

/// An instance's unsafe region as the publication path parsed it: the
/// declared input box, and a DISJUNCTION of clauses, each a CONJUNCTION of
/// atoms, every atom given as the objective whose certification excludes it.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphInstanceProperty {
    pub(super) lo: Box<[f64]>,
    pub(super) hi: Box<[f64]>,
    pub(super) clauses: Vec<Vec<GraphObjective>>,
}

impl GraphInstanceProperty {
    /// `None` unless the box is finite, non-empty and non-inverted, and there
    /// is at least one clause and no empty clause (an empty conjunction is the
    /// whole space: nothing could exclude it).
    #[must_use]
    pub fn new(lo: Vec<f64>, hi: Vec<f64>, clauses: Vec<Vec<GraphObjective>>) -> Option<Self> {
        let box_ok = !lo.is_empty()
            && lo.len() == hi.len()
            && lo
                .iter()
                .zip(&hi)
                .all(|(l, h)| l.is_finite() && h.is_finite() && l <= h);
        if !box_ok || clauses.is_empty() || clauses.iter().any(Vec::is_empty) {
            return None;
        }
        Some(Self {
            lo: lo.into(),
            hi: hi.into(),
            clauses,
        })
    }

    /// Clauses of the unsafe region.
    #[must_use]
    pub fn clause_count(&self) -> usize {
        self.clauses.len()
    }
}

/// Why [`authorize_graph_instance`] refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphInstanceRefusal {
    /// No ledgered graph invocation proved a cover in this run.
    NothingCaptured,
    /// No captured cover entails this clause on the declared box.
    ClauseUncovered {
        /// The clause.
        clause: usize,
    },
    /// The aggregation refused.
    Authorize(AuthorizeRefusal),
}

impl fmt::Display for GraphInstanceRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NothingCaptured => f.write_str("nothing_captured"),
            Self::ClauseUncovered { clause } => write!(f, "clause_uncovered clause={clause}"),
            Self::Authorize(refusal) => write!(f, "authorize {refusal}"),
        }
    }
}

/// The graph lane's ONE public token door: an INSTANCE-scoped token when every
/// clause of `property` is entailed by a captured cover — a cover proved on a
/// root box containing the declared box, for an objective that excludes some
/// atom of the clause. Re-checks both itself and mints through
/// `authorize_unsat` (one `ClassCover::Tree` per clause); nothing the caller
/// supplies is taken as proof, and a captured cover cannot be forged.
pub fn authorize_graph_instance(
    property: &GraphInstanceProperty,
    capture: &GraphCoverCapture,
) -> Result<UnsatToken, GraphInstanceRefusal> {
    if capture.entries.is_empty() {
        return Err(GraphInstanceRefusal::NothingCaptured);
    }
    let mut covers = Vec::with_capacity(property.clauses.len());
    for (clause, atoms) in property.clauses.iter().enumerate() {
        // EVERY entailing cover, not the first: the capture's order is the
        // drivers' deposit order, and `authorize_unsat` takes the best margin
        // per class, so the token (and its `min_margin` observable) is a
        // function of the captured SET, whatever order it was deposited in.
        let before = covers.len();
        covers.extend(
            capture
                .entries
                .iter()
                .filter(|entry| {
                    entry.contains_box(&property.lo, &property.hi)
                        && atoms.iter().any(|atom| entry.objective.entails(atom))
                })
                .map(|entry| ClassCover::Tree(entry.cover.relabeled(&[clause]))),
        );
        if covers.len() == before {
            return Err(GraphInstanceRefusal::ClauseUncovered { clause });
        }
    }
    let required: Vec<usize> = (0..property.clauses.len()).collect();
    authorize_unsat(
        TokenScope::Instance,
        &required,
        &covers,
        &GuardReport::new(),
    )
    .map_err(GraphInstanceRefusal::Authorize)
}
