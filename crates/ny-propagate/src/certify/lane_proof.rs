// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! B2.2 — LANE PROOF TAPES: the one lane-agnostic leaf certificate every
//! unsat-publishing lane streams under the Proof profile, for the independent
//! checker (`ny proof-check`, `ny-proof-bundle/v2`) to re-derive.
//!
//! # The certificate
//!
//! A lane's proof is a set of LEAVES. A leaf is
//!
//! * a REGION: an input box (input-split and lowdim leaves, written as the
//!   box's faces inside the declared box — `x_i >= t` / `x_i <= t` literals
//!   over the ONNX input's flat index), or a set of Relu split literals with
//!   ABSOLUTE neuron ids (`z >= 0` / `z <= 0` on an ONNX Relu node's
//!   `input[0]`, flat C-order index), or both;
//! * the DISCHARGE: which of the instance's VNN-LIB clauses the lane
//!   certified to have no counterexample in the region — derived from the
//!   lane's own objective rows by exact entailment (the same rule the graph
//!   instance door uses), never from a number the lane reports;
//! * the DUAL POINT as a hint: the β of every split literal (the lane's
//!   Lagrangian multipliers). By weak duality ANY `β >= 0` gives a valid
//!   bound, so a wrong hint can only cost a confirmation.
//!
//! The checker confirms each discharge with its OWN outward-rounded
//! back-substitution and its own input clip (the per-leaf LINEAR LOWER-BOUND
//! certificate: coefficients over the input plus the split multipliers,
//! concretized over the region with directed rounding), and proves COVER per
//! clause itself: a box tiling geometrically (any tiling), a split tree by
//! the case-split identity. Nothing a tape says is trusted.
//!
//! # Write-only, and dark in Competition
//!
//! A lane builds a [`TapeRecorder`] only when the process capture is armed
//! ([`reset_lane_proof_capture`], which arms iff the entry point's Proof arm
//! switched hint capture on). Under Competition nothing is armed, no
//! recorder exists, and every recording site is one `Option` test: the
//! verdict path is bit-identical. A tape has no reader: its leaves leave the
//! process only through [`stream_lane_proofs`] into a write-only
//! [`ProofSink`], after publication; the only values returned are the COUNTS
//! of what was streamed (observables for the entry point's line).
//!
//! # What a tape claims, and what a missing tape means
//!
//! A tape deposited by a gate-checked frame (the graph ledger proved its
//! cover) or by a lane whose queue drained with every fate recorded (lowdim)
//! is COMPLETE. Any other tape (the multi-objective lane records its closes
//! at the domain level without a gate of its own) is streamed as is. For
//! every clause no complete tape discharges, [`stream_lane_proofs`] adds ONE
//! `opaque` node over the whole region: the lane claims it (it published
//! `unsat`), with no certificate the checker can confirm. So a published
//! lane `unsat` always has a claimed cover, and the checker's verdict is
//! `verified` exactly when the streamed certificates cover by themselves,
//! `unconfirmed` otherwise — never `invalid` for a lane that merely streams
//! less than it proved.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use ny_tensor::BoundedTensor;

use super::graph::{GraphInstanceProperty, GraphObjective};
use super::sink::{
    hint_capture_enabled, next_node_id, note_node_streamed, with_installed_sink, AbsLit, AbsSite,
    InputLit, NodeClose, NodeRecord, ProofSink, SiteTensorRecord, SitesRecord,
};
use crate::beta_crown::branching::{GraphNeuronConstraint, GraphSplitHistory};
use crate::beta_crown::state::GraphBetaState;
use crate::layers::Layer;
use crate::GraphNetwork;

/// Most leaves one tape keeps; past it the tape is LOSSY (streamed, never
/// complete).
pub(crate) const TAPE_MAX_LEAVES: usize = 1 << 20;

/// Most literals plus box coordinates one tape keeps (about 64 MB).
pub(crate) const TAPE_MAX_WORDS: usize = 8 << 20;

/// One split literal in the tape's own node numbering: the pre-activation
/// (`input[0]`) of neuron `neuron` of activation node `node` is `>= 0` when
/// `positive`, else `<= 0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TapeLit {
    pub(crate) node: u32,
    pub(crate) neuron: u32,
    pub(crate) positive: bool,
}

/// A leaf's region, in the lane's coordinates.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum TapeRegion {
    /// Split literals below the lane's root box, branch order. `opaque`
    /// names a literal the tape cannot express (a GenBaB point, a norm
    /// window, an unknown node): the leaf is then streamed as an `opaque`
    /// claim, never dropped.
    Splits {
        lits: Vec<TapeLit>,
        opaque: Option<&'static str>,
    },
    /// An input box over the FULL input dimension.
    Box { lo: Box<[f64]>, hi: Box<[f64]> },
}

/// How the lane closed a leaf.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TapeClose {
    /// A certified bound strictly past the threshold (or a batched screen).
    Bound,
    /// The region is certified empty (claims every clause).
    Empty,
    /// An input clip excluded the region (the checker's clip re-derives it).
    Clip,
}

/// One recorded leaf.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TapeLeaf {
    pub(crate) region: TapeRegion,
    pub(crate) close: TapeClose,
    /// For EVERY group, SOME objective in it is certified on the region (the
    /// lane may not know which: a lowdim conjunction closes when any conjunct
    /// does). Each objective alone is a singleton group.
    pub(crate) groups: Vec<Vec<u32>>,
    /// β aligned with a `Splits` region's literals (empty when unknown).
    pub(crate) beta: Vec<f64>,
    /// The lane's certified margin (hint; `NaN` when none).
    pub(crate) margin: f64,
    /// The bound may have come from an ancestor region (hint).
    pub(crate) lift: bool,
}

#[derive(Debug, Default)]
struct TapeState {
    leaves: Vec<TapeLeaf>,
    words: usize,
    lost: Option<&'static str>,
    deposited: bool,
}

/// A lane invocation's WRITE-ONLY proof tape. Built only while the process
/// capture is armed (see the module docs); its leaves are private and leave
/// only through [`stream_lane_proofs`].
#[derive(Debug)]
pub(crate) struct TapeRecorder {
    lane: &'static str,
    root_lo: Box<[f64]>,
    root_hi: Box<[f64]>,
    objectives: Vec<Option<GraphObjective>>,
    node_ids: HashMap<String, u32>,
    node_names: Vec<String>,
    node_sizes: Vec<Option<usize>>,
    state: Mutex<TapeState>,
}

fn box_f64(t: &BoundedTensor) -> (Vec<f64>, Vec<f64>) {
    (
        t.lower().iter().map(|&v| f64::from(v)).collect(),
        t.upper().iter().map(|&v| f64::from(v)).collect(),
    )
}

/// Flat sizes of every activation node's PRE-activation (its output has the
/// same shape), from one interval pass over `graph` at `input`. Proof-profile
/// only (a recorder is built only then). A graph the pass cannot handle
/// yields no sizes, and literals on those nodes stream as `opaque` claims.
fn activation_sizes(graph: &GraphNetwork, input: &BoundedTensor) -> HashMap<String, usize> {
    let mut out = HashMap::new();
    let Ok(bounds) = graph.collect_block_ibp_bounds(&graph.node_order, input) else {
        return out;
    };
    for name in &graph.node_order {
        let Some(node) = graph.nodes.get(name) else {
            continue;
        };
        if !matches!(node.layer, Layer::ReLU(_)) {
            continue;
        }
        if let Some(b) = bounds.get(name) {
            out.insert(name.clone(), b.len());
        }
    }
    out
}

impl TapeRecorder {
    fn build(
        lane: &'static str,
        root_lo: Vec<f64>,
        root_hi: Vec<f64>,
        objectives: Vec<Option<GraphObjective>>,
        node_names: Vec<String>,
        sizes: &HashMap<String, usize>,
    ) -> Self {
        let mut node_ids = HashMap::with_capacity(node_names.len());
        for (id, name) in node_names.iter().enumerate() {
            node_ids
                .entry(name.clone())
                .or_insert(u32::try_from(id).unwrap_or(u32::MAX));
        }
        let node_sizes = node_names.iter().map(|n| sizes.get(n).copied()).collect();
        Self {
            lane,
            root_lo: root_lo.into(),
            root_hi: root_hi.into(),
            objectives,
            node_ids,
            node_names,
            node_sizes,
            state: Mutex::new(TapeState::default()),
        }
    }

    /// A tape for a graph lane's split tree (or input boxes) below `input`,
    /// proving the lane objectives `(row, threshold, verify_upper)` (each
    /// normalized to its lower-bound reading). `None` unless the process
    /// capture is armed.
    pub(crate) fn for_graph(
        lane: &'static str,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        objectives: &[(&[f32], f32, bool)],
    ) -> Option<Self> {
        if !lane_tape_armed() {
            return None;
        }
        let (lo, hi) = box_f64(input);
        let objs = objectives
            .iter()
            .map(|&(row, thr, upper)| GraphObjective::new(row, thr, upper))
            .collect();
        let sizes = activation_sizes(graph, input);
        Some(Self::build(
            lane,
            lo,
            hi,
            objs,
            graph.node_order.clone(),
            &sizes,
        ))
    }

    /// A tape over explicit node names with no size information (literals
    /// therefore stream as `opaque` claims) — the graph ledger's name-list
    /// arm. `None` unless armed.
    pub(crate) fn for_names<'n>(
        lane: &'static str,
        names: impl IntoIterator<Item = &'n str>,
        input: &BoundedTensor,
        objectives: &[(&[f32], f32, bool)],
    ) -> Option<Self> {
        if !lane_tape_armed() {
            return None;
        }
        let (lo, hi) = box_f64(input);
        let objs = objectives
            .iter()
            .map(|&(row, thr, upper)| GraphObjective::new(row, thr, upper))
            .collect();
        Some(Self::build(
            lane,
            lo,
            hi,
            objs,
            names.into_iter().map(str::to_string).collect(),
            &HashMap::new(),
        ))
    }

    /// A box-only tape (lowdim): root box `[lo, hi]` over the full input,
    /// objectives in their lower-bound reading `objective · y > threshold`
    /// (f64 rows that must be exact binary32 values, as the lane's f32
    /// objectives widened; an inexact row is not an instance objective and
    /// discharges nothing). `None` unless armed.
    pub(crate) fn for_boxes(
        lane: &'static str,
        lo: &[f64],
        hi: &[f64],
        objectives: &[Vec<f64>],
        thresholds: &[f64],
    ) -> Option<Self> {
        if !lane_tape_armed() {
            return None;
        }
        let exact32 = |v: f64| {
            let f = v as f32;
            (f64::from(f) == v).then_some(f)
        };
        let objs = objectives
            .iter()
            .zip(thresholds)
            .map(|(row, &thr)| {
                let row32: Option<Vec<f32>> = row.iter().map(|&v| exact32(v)).collect();
                let thr32 = exact32(thr)?;
                GraphObjective::new(&row32?, thr32, false)
            })
            .collect();
        Some(Self::build(
            lane,
            lo.to_vec(),
            hi.to_vec(),
            objs,
            Vec::new(),
            &HashMap::new(),
        ))
    }

    /// Objectives the tape was built for (a caller's index-alignment check;
    /// a count, never tape data).
    pub(crate) fn objective_count(&self) -> usize {
        self.objectives.len()
    }

    fn push(&self, leaf: TapeLeaf) {
        let words = match &leaf.region {
            TapeRegion::Splits { lits, .. } => lits.len() + 1,
            TapeRegion::Box { lo, .. } => 2 * lo.len() + 1,
        };
        let mut st = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if st.deposited || st.lost.is_some() {
            return;
        }
        if st.leaves.len() >= TAPE_MAX_LEAVES || st.words.saturating_add(words) > TAPE_MAX_WORDS {
            st.lost = Some("tape_capacity");
            st.leaves = Vec::new();
            return;
        }
        st.words += words;
        st.leaves.push(leaf);
    }

    /// The tape missed a close it cannot express: it is never complete.
    pub(crate) fn lose(&self, why: &'static str) {
        let mut st = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if st.lost.is_none() {
            st.lost = Some(why);
        }
    }

    fn lits_of(
        &self,
        history: &GraphSplitHistory,
        extra: Option<&GraphNeuronConstraint>,
    ) -> (Vec<TapeLit>, Option<&'static str>) {
        let mut opaque = None;
        if history.has_norm_inv_rms_constraints() {
            opaque = Some("norm_split");
        }
        if !history.genbab_constraints.is_empty() {
            opaque = Some("genbab_point");
        }
        let mut lits = Vec::with_capacity(history.constraints.len() + 1);
        for c in history.constraints.iter().chain(extra) {
            let Some(&node) = self.node_ids.get(&c.node_name) else {
                opaque = Some("unknown_split_node");
                continue;
            };
            let Ok(neuron) = u32::try_from(c.neuron_idx) else {
                opaque = Some("neuron_index_overflow");
                continue;
            };
            lits.push(TapeLit {
                node,
                neuron,
                positive: c.is_active,
            });
        }
        (lits, opaque)
    }

    fn beta_of(
        history: &GraphSplitHistory,
        extra: Option<&GraphNeuronConstraint>,
        beta: Option<&GraphBetaState>,
    ) -> Vec<f64> {
        let Some(beta) = beta else {
            return Vec::new();
        };
        history
            .constraints
            .iter()
            .chain(extra)
            .map(|c| {
                beta.entries
                    .iter()
                    .find(|e| {
                        e.node_name == c.node_name
                            && e.neuron_idx == c.neuron_idx
                            && e.split_point == 0.0
                    })
                    .map_or(0.0, |e| {
                        let v = f64::from(e.value);
                        if v.is_finite() && v > 0.0 {
                            v
                        } else {
                            0.0
                        }
                    })
            })
            .collect()
    }

    /// A split-tree leaf: the domain with split history `history` (plus
    /// `extra`, a child half-space the lane never built), closed by `close`
    /// for the objective groups `groups`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn note_splits(
        &self,
        history: &GraphSplitHistory,
        extra: Option<&GraphNeuronConstraint>,
        beta: Option<&GraphBetaState>,
        close: TapeClose,
        groups: Vec<Vec<u32>>,
        margin: f64,
        lift: bool,
    ) {
        let (lits, opaque) = self.lits_of(history, extra);
        let beta = if opaque.is_none() {
            Self::beta_of(history, extra, beta)
        } else {
            Vec::new()
        };
        self.push(TapeLeaf {
            region: TapeRegion::Splits { lits, opaque },
            close,
            groups,
            beta,
            margin,
            lift,
        });
    }

    /// An input box `[lo, hi]` (full input dimension) closed by `close`.
    pub(crate) fn note_box(
        &self,
        lo: &[f64],
        hi: &[f64],
        close: TapeClose,
        groups: Vec<Vec<u32>>,
        margin: f64,
    ) {
        if lo.len() != self.root_lo.len() || hi.len() != self.root_lo.len() {
            self.lose("box_dimension");
            return;
        }
        self.push(TapeLeaf {
            region: TapeRegion::Box {
                lo: lo.into(),
                hi: hi.into(),
            },
            close,
            groups,
            beta: Vec::new(),
            margin,
            lift: false,
        });
    }

    /// An input clip of box `pre` to box `post`: the part of `pre` outside
    /// `post` is closed, recorded as at most `2·D` slabs (the box-difference
    /// decomposition; comparisons only).
    pub(crate) fn note_clip(
        &self,
        pre_lo: &[f64],
        pre_hi: &[f64],
        post_lo: &[f64],
        post_hi: &[f64],
        groups: &[Vec<u32>],
    ) {
        let d = self.root_lo.len();
        if [pre_lo, pre_hi, post_lo, post_hi]
            .iter()
            .any(|v| v.len() != d || v.iter().any(|x| !x.is_finite()))
        {
            self.lose("clip_box");
            return;
        }
        let mut lo = pre_lo.to_vec();
        let mut hi = pre_hi.to_vec();
        for k in 0..d {
            if post_lo[k] > lo[k] {
                let cut = post_lo[k].min(hi[k]);
                let mut slab_hi = hi.clone();
                slab_hi[k] = cut;
                self.note_box(&lo, &slab_hi, TapeClose::Clip, groups.to_vec(), f64::NAN);
                lo[k] = cut;
            }
            if post_hi[k] < hi[k] {
                let cut = post_hi[k].max(lo[k]);
                let mut slab_lo = lo.clone();
                slab_lo[k] = cut;
                self.note_box(&slab_lo, &hi, TapeClose::Clip, groups.to_vec(), f64::NAN);
                hi[k] = cut;
            }
        }
    }

    /// Hand the tape to the process capture (once; later records are
    /// ignored). `complete`: the lane's own gate (or its drained queue)
    /// established that the leaves cover the root for every objective.
    pub(crate) fn deposit(&self, complete: bool) {
        let (leaves, lost, words) = {
            let mut st = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if st.deposited {
                return;
            }
            st.deposited = true;
            (std::mem::take(&mut st.leaves), st.lost, st.words)
        };
        let frame = TapeFrame {
            lane: self.lane,
            root_lo: self.root_lo.clone(),
            root_hi: self.root_hi.clone(),
            objectives: self.objectives.clone(),
            node_names: self.node_names.clone(),
            node_sizes: self.node_sizes.clone(),
            leaves,
            complete: complete && lost.is_none(),
            lost,
            words,
        };
        let mut guard = capture_lock();
        if let Some(frames) = guard.as_mut() {
            admit_frame(frames, frame, CAPTURE_MAX_WORDS);
        }
    }
}

/// review(b2-lanes): keep `frame` in the capture within the process-wide
/// budget of `max_words` literal/coordinate words across ALL held tapes. A
/// tape past the budget is kept as LOST (no leaves, never complete: its
/// clauses fall back to the lane's opaque claim), so a Proof run that
/// deposits many large tapes degrades to `unconfirmed` instead of holding up
/// to `CAPTURE_CAP` full tapes in memory (this host has no memory governor
/// without a cgroup).
fn admit_frame(frames: &mut Vec<TapeFrame>, mut frame: TapeFrame, max_words: usize) {
    if frames.len() >= CAPTURE_CAP {
        return;
    }
    let held = frames.iter().fold(0usize, |s, f| s.saturating_add(f.words));
    if held.saturating_add(frame.words) > max_words {
        frame.leaves = Vec::new();
        frame.words = 0;
        frame.complete = false;
        frame.lost = Some(frame.lost.unwrap_or("capture_capacity"));
    }
    frames.push(frame);
}

/// A deposited tape.
#[derive(Debug, Clone)]
struct TapeFrame {
    lane: &'static str,
    root_lo: Box<[f64]>,
    root_hi: Box<[f64]>,
    objectives: Vec<Option<GraphObjective>>,
    node_names: Vec<String>,
    node_sizes: Vec<Option<usize>>,
    leaves: Vec<TapeLeaf>,
    complete: bool,
    lost: Option<&'static str>,
    /// Literal and box-coordinate words the leaves hold (the tape's count).
    words: usize,
}

/// Most tapes one run keeps.
const CAPTURE_CAP: usize = 4096;

/// Most literal/coordinate words ALL captured tapes hold together (four full
/// tapes; see [`admit_frame`]).
const CAPTURE_MAX_WORDS: usize = 4 * TAPE_MAX_WORDS;

/// Whether lanes build tapes (the process capture is armed).
static ARMED: AtomicBool = AtomicBool::new(false);

/// The process capture: `None` until an entry point arms it.
static CAPTURE: Mutex<Option<Vec<TapeFrame>>> = Mutex::new(None);

fn capture_lock() -> std::sync::MutexGuard<'static, Option<Vec<TapeFrame>>> {
    CAPTURE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Whether a lane should build a tape now (one relaxed load; `false` under
/// the Competition profile, where nothing ever arms the capture).
#[must_use]
pub(crate) fn lane_tape_armed() -> bool {
    ARMED.load(Ordering::Acquire)
}

/// Arm (and empty) the process capture — iff hint capture is on (the entry
/// point's Proof arm). The publication path calls this before the lanes run;
/// under Competition it arms nothing, so no lane builds a tape.
pub fn reset_lane_proof_capture() {
    let mut guard = capture_lock();
    if hint_capture_enabled() {
        *guard = Some(Vec::new());
        ARMED.store(true, Ordering::Release);
    } else {
        *guard = None;
        ARMED.store(false, Ordering::Release);
    }
}

/// Disarm the capture and return what the lanes deposited.
#[must_use]
pub fn take_lane_proof_capture() -> LaneProofCapture {
    let mut guard = capture_lock();
    ARMED.store(false, Ordering::Release);
    LaneProofCapture {
        frames: guard.take().unwrap_or_default(),
    }
}

/// Test-only: arm the capture regardless of the hint-capture flag.
#[cfg(test)]
pub(crate) fn arm_lane_proof_capture_for_test() {
    *capture_lock() = Some(Vec::new());
    ARMED.store(true, Ordering::Release);
}

/// The tapes one run's lanes deposited (opaque: streamed, never read).
#[derive(Debug, Clone, Default)]
pub struct LaneProofCapture {
    frames: Vec<TapeFrame>,
}

impl LaneProofCapture {
    /// Tapes held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// No tape was deposited.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Append another capture's tapes (a process may run the lanes twice).
    pub fn absorb(&mut self, other: LaneProofCapture) {
        self.frames.extend(other.frames);
    }

    /// TEST-HARNESS face (production never calls it): keep only the tapes
    /// carrying one of the lower-bound objectives `(row, threshold)` in
    /// `wanted`. The capture is PROCESS-global, so while a test has it armed
    /// every lane that other tests of the same binary run concurrently
    /// deposits into it too (review(b2-lanes)); a test must judge its own
    /// tapes only. Returns nothing (write-only: it can only drop tapes).
    #[doc(hidden)]
    pub fn retain_objectives(&mut self, wanted: &[(&[f32], f32)]) {
        let wanted: Vec<GraphObjective> = wanted
            .iter()
            .filter_map(|&(row, threshold)| GraphObjective::new(row, threshold, false))
            .collect();
        self.frames
            .retain(|f| f.objectives.iter().flatten().any(|o| wanted.contains(o)));
    }
}

/// What the ENTRY POINT supplies to translate tapes into bundle nodes: the
/// instance's declared box and clauses (as the publication path parsed them)
/// and, for every graph activation node the entry point resolved to an ONNX
/// Relu node, the ONNX name of that node's `input[0]`.
#[derive(Debug, Clone)]
pub struct LaneProofInstance {
    lo: Vec<f64>,
    hi: Vec<f64>,
    clauses: Vec<Vec<GraphObjective>>,
    relu_sites: HashMap<String, String>,
}

impl LaneProofInstance {
    /// From the instance property and the Relu site map.
    #[must_use]
    pub fn new(property: &GraphInstanceProperty, relu_sites: HashMap<String, String>) -> Self {
        Self {
            lo: property.lo.to_vec(),
            hi: property.hi.to_vec(),
            clauses: property.clauses.clone(),
            relu_sites,
        }
    }

    /// Clause ids (`0..clauses`), the verdict record's `required`.
    #[must_use]
    pub fn clause_ids(&self) -> Vec<usize> {
        (0..self.clauses.len()).collect()
    }
}

/// Counts of one [`stream_lane_proofs`] call (the entry point's observable).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LaneProofSummary {
    /// Tapes in the capture.
    pub tapes: usize,
    /// Tapes whose root box contains the declared box and that discharge at
    /// least one clause.
    pub tapes_used: usize,
    /// Node records streamed (certificates plus opaque claims).
    pub nodes: u64,
    /// Of which `opaque` claims (untranslatable leaves and fallbacks).
    pub opaque: u64,
    /// Clauses no complete tape discharges (each got an `opaque` root claim).
    pub fallback_clauses: usize,
    /// Relu site tensors named in the `sites` record.
    pub sites: usize,
    /// Tapes that lost a close (capacity or an unbuilt frame): streamed, never
    /// complete.
    pub lossy: usize,
}

fn contains_box(frame: &TapeFrame, lo: &[f64], hi: &[f64]) -> bool {
    frame.root_lo.len() == lo.len()
        && (0..lo.len()).all(|d| frame.root_lo[d] <= lo[d] && frame.root_hi[d] >= hi[d])
}

/// Build the nodes (and sites) of every tape against `inst`, in memory.
fn translate(
    capture: &LaneProofCapture,
    inst: &LaneProofInstance,
) -> (SitesRecord, Vec<NodeRecord>, LaneProofSummary) {
    let mut summary = LaneProofSummary {
        tapes: capture.frames.len(),
        ..LaneProofSummary::default()
    };
    // Site ids in first-use order; a site whose size the tapes disagree on is
    // refused (its literals become opaque claims).
    let mut site_ids: BTreeMap<String, (u32, usize)> = BTreeMap::new();
    let mut site_order: Vec<(String, usize)> = Vec::new();
    let mut nodes: Vec<NodeRecord> = Vec::new();
    let mut complete: BTreeSet<usize> = BTreeSet::new();
    let n_clauses = inst.clauses.len();
    let opaque_node = |dis: Vec<usize>, src: &'static str| NodeRecord {
        id: next_node_id(),
        region: 0,
        lits: Vec::new(),
        xlits: Vec::new(),
        close: NodeClose::Opaque,
        src,
        dis: Some(dis),
        beta: Vec::new(),
        alpha: None,
        lift: false,
        margin: f64::NAN,
        empty_site: None,
        fd: None,
    };
    for frame in &capture.frames {
        if frame.lost.is_some() {
            summary.lossy += 1;
        }
        if !contains_box(frame, &inst.lo, &inst.hi) {
            continue;
        }
        // Clauses each objective entails (some atom of the clause).
        let entails: Vec<BTreeSet<usize>> = frame
            .objectives
            .iter()
            .map(|o| {
                let Some(o) = o else {
                    return BTreeSet::new();
                };
                (0..n_clauses)
                    .filter(|&k| inst.clauses[k].iter().any(|atom| o.entails(atom)))
                    .collect()
            })
            .collect();
        let frame_clauses: BTreeSet<usize> = entails.iter().flatten().copied().collect();
        if frame_clauses.is_empty() {
            continue;
        }
        summary.tapes_used += 1;
        // A COMPLETE tape discharges clause k on its whole root only if every
        // leaf's groups reach k (checked leaf by leaf below).
        let mut all_leaves_reach: BTreeMap<usize, bool> =
            frame_clauses.iter().map(|&k| (k, true)).collect();
        for leaf in &frame.leaves {
            let dis: BTreeSet<usize> = if leaf.close == TapeClose::Empty {
                (0..n_clauses).collect()
            } else {
                (0..n_clauses)
                    .filter(|&k| {
                        leaf.groups.iter().any(|g| {
                            !g.is_empty()
                                && g.iter().all(|&j| {
                                    entails.get(j as usize).is_some_and(|e| e.contains(&k))
                                })
                        })
                    })
                    .collect()
            };
            for (k, reach) in &mut all_leaves_reach {
                if !dis.contains(k) {
                    *reach = false;
                }
            }
            if dis.is_empty() {
                continue;
            }
            let dis_vec: Vec<usize> = dis.iter().copied().collect();
            // Region.
            let mut lits = Vec::new();
            let mut xlits = Vec::new();
            let mut beta = Vec::new();
            let mut untranslatable: Option<&'static str> = None;
            match &leaf.region {
                TapeRegion::Splits { lits: tl, opaque } => {
                    if let Some(why) = opaque {
                        untranslatable = Some(why);
                    }
                    for (i, l) in tl.iter().enumerate() {
                        let name = frame.node_names.get(l.node as usize);
                        let size = frame.node_sizes.get(l.node as usize).copied().flatten();
                        let site = name.and_then(|n| inst.relu_sites.get(n));
                        let (Some(site), Some(size)) = (site, size) else {
                            untranslatable.get_or_insert("unresolved_site");
                            continue;
                        };
                        if l.neuron as usize >= size {
                            untranslatable.get_or_insert("neuron_out_of_range");
                            continue;
                        }
                        let next = u32::try_from(site_order.len()).unwrap_or(u32::MAX);
                        let entry = site_ids.entry(site.clone()).or_insert_with(|| {
                            site_order.push((site.clone(), size));
                            (next, size)
                        });
                        if entry.1 != size {
                            untranslatable.get_or_insert("site_size_disagrees");
                            continue;
                        }
                        lits.push(AbsLit {
                            site: AbsSite {
                                op: entry.0,
                                index: l.neuron,
                            },
                            positive: l.positive,
                        });
                        beta.push(leaf.beta.get(i).copied().unwrap_or(0.0));
                    }
                }
                TapeRegion::Box { lo, hi } => {
                    let mut outside = false;
                    for d in 0..lo.len().min(inst.lo.len()) {
                        if lo[d] > inst.hi[d] || hi[d] < inst.lo[d] || lo[d] > hi[d] {
                            outside = true;
                            break;
                        }
                        // review(b2-lanes): a leaf of ZERO width in a
                        // coordinate where the declared box has positive
                        // width (a clip shrank a face onto the other) has
                        // zero measure: the checker's box cover never accepts
                        // such a member (it holds no cell of positive width
                        // there), so it covers nothing the verdict needs —
                        // and streamed strictly inside it would carry BOTH
                        // `x_d >= t` and `x_d <= t`, which the checker
                        // refuses as `both_sides`, malforming the WHOLE
                        // bundle. Skip it like a leaf outside the box.
                        if lo[d] == hi[d] && inst.lo[d] < inst.hi[d] {
                            outside = true;
                            break;
                        }
                        let Ok(index) = u32::try_from(d) else {
                            untranslatable.get_or_insert("input_index_overflow");
                            continue;
                        };
                        if lo[d] > inst.lo[d] {
                            xlits.push(InputLit {
                                index,
                                threshold: lo[d],
                                positive: true,
                            });
                        }
                        if hi[d] < inst.hi[d] {
                            xlits.push(InputLit {
                                index,
                                threshold: hi[d],
                                positive: false,
                            });
                        }
                    }
                    if outside {
                        // The leaf lies outside the declared box (or has zero
                        // measure in it): it covers nothing the instance
                        // needs.
                        continue;
                    }
                }
            }
            if let Some(why) = untranslatable {
                summary.opaque += 1;
                nodes.push(opaque_node(dis_vec, why));
                continue;
            }
            let (close, dis_field) = match leaf.close {
                TapeClose::Empty => (NodeClose::Empty, None),
                TapeClose::Bound | TapeClose::Clip => (NodeClose::Bound, Some(dis_vec)),
            };
            if beta.len() != lits.len() || beta.iter().all(|b| *b == 0.0) {
                beta = Vec::new();
            }
            nodes.push(NodeRecord {
                id: next_node_id(),
                region: 0,
                lits,
                xlits,
                close,
                src: frame.lane,
                dis: dis_field,
                beta,
                alpha: None,
                lift: leaf.lift,
                margin: leaf.margin,
                empty_site: None,
                fd: None,
            });
        }
        if frame.complete && !frame.leaves.is_empty() {
            for (k, reach) in all_leaves_reach {
                if reach {
                    complete.insert(k);
                }
            }
        }
    }
    // Every clause no complete tape discharges: the lane's opaque claim.
    for k in 0..n_clauses {
        if !complete.contains(&k) {
            summary.fallback_clauses += 1;
            summary.opaque += 1;
            nodes.push(opaque_node(vec![k], "lane_claim"));
        }
    }
    summary.nodes = nodes.len() as u64;
    summary.sites = site_order.len();
    let sites = SitesRecord {
        tensors: site_order
            .into_iter()
            .enumerate()
            .map(|(id, (name, size))| {
                let id = u32::try_from(id).unwrap_or(u32::MAX);
                SiteTensorRecord {
                    id,
                    name,
                    size,
                    op: id,
                }
            })
            .collect(),
    };
    (sites, nodes, summary)
}

/// Stream the capture's tapes into `sink` as `ny-proof-bundle/v2` records: the
/// `sites` record of every Relu site a literal names, then one `node` per
/// leaf (and the opaque claims of the module docs). Write-only: the summary
/// counts what was streamed; nothing the sink does comes back.
pub fn stream_lane_proofs(
    capture: &LaneProofCapture,
    inst: &LaneProofInstance,
    sink: &mut dyn ProofSink,
) -> LaneProofSummary {
    let (sites, nodes, summary) = translate(capture, inst);
    sink.sites(&sites);
    for node in &nodes {
        sink.node(node);
        note_node_streamed();
    }
    summary
}

/// [`stream_lane_proofs`] into the installed sink (no-op without one; the
/// summary still counts what WOULD have been streamed).
pub fn stream_lane_proofs_installed(
    capture: &LaneProofCapture,
    inst: &LaneProofInstance,
) -> LaneProofSummary {
    let (sites, nodes, summary) = translate(capture, inst);
    with_installed_sink(|sink| {
        sink.sites(&sites);
        for node in &nodes {
            sink.node(node);
            note_node_streamed();
        }
    });
    summary
}

#[cfg(test)]
mod capture_budget_tests {
    use super::{admit_frame, TapeClose, TapeFrame, TapeLeaf, TapeRegion};

    fn frame(words: usize) -> TapeFrame {
        TapeFrame {
            lane: "t",
            root_lo: vec![-1.0].into(),
            root_hi: vec![1.0].into(),
            objectives: Vec::new(),
            node_names: Vec::new(),
            node_sizes: Vec::new(),
            leaves: vec![TapeLeaf {
                region: TapeRegion::Box {
                    lo: vec![-1.0].into(),
                    hi: vec![1.0].into(),
                },
                close: TapeClose::Bound,
                groups: vec![vec![0]],
                beta: Vec::new(),
                margin: f64::NAN,
                lift: false,
            }],
            complete: true,
            lost: None,
            words,
        }
    }

    /// review(b2-lanes): the capture holds at most `max_words` words across
    /// ALL tapes; a tape past the budget is kept LOST (no leaves, never
    /// complete), so its clauses fall back to the lane's opaque claim and
    /// memory stays bounded however many tapes a Proof run deposits.
    #[test]
    fn capture_word_budget_keeps_a_tape_past_it_as_lost() {
        let mut frames = Vec::new();
        admit_frame(&mut frames, frame(10), 15);
        admit_frame(&mut frames, frame(10), 15);
        admit_frame(&mut frames, frame(5), 15);
        assert_eq!(frames.len(), 3);
        assert!(frames[0].complete && frames[0].lost.is_none());
        assert_eq!(frames[0].leaves.len(), 1);
        assert!(!frames[1].complete, "past the budget: never complete");
        assert_eq!(frames[1].lost, Some("capture_capacity"));
        assert!(frames[1].leaves.is_empty(), "its leaves are freed");
        assert_eq!(frames[1].words, 0);
        // A tape that still fits after the lost one is kept whole.
        assert!(frames[2].complete && frames[2].leaves.len() == 1);
    }
}
