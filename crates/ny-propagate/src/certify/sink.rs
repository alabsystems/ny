// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Proof sinks (ENGINE_PROGRAM §1.4, §1.7 item B1): WRITE-ONLY recipients of
//! proof hints for an offline checker (B2.1 `ny proof-check`).
//!
//! Every [`ProofSink`] method returns `()` (ratcheted by the crate's
//! section-1.7 integration test), so no verdict path can read anything back
//! from a sink: a sink receives a leaf only after the cover ledger counted it,
//! and removing the sink leaves every verdict-bearing value bit-identical. The
//! proof profile chooses WHICH sink is wired — a recording sink under `Proof`,
//! [`NullSink`] under `Competition` — and nothing else.
//!
//! # The process sink (wired by the entry point)
//!
//! This module never learns the profile. The ENTRY POINT (ny-cli, the only
//! code that resolves it) installs ONE sink per process with
//! [`install_proof_sink`] before any lane runs and, under Proof only, turns on
//! hint capture ([`enable_hint_capture`]); the lanes stream into whatever was
//! installed through [`stream_frame`] (after a frame's cover was PROVED and
//! its token authorized) and the entry point closes the stream with
//! [`finish_stream`] after publication. With nothing installed (unit tests,
//! library callers) both are no-ops. Every function here returns `()` or
//! reports only what the ENTRY POINT did; none returns anything a sink
//! produced.
//!
//! # Proof-bundle v2 hints (B2.1)
//!
//! A frame's [`HintLedger`] collects, write-only, what the offline checker
//! needs to RE-DERIVE each close with its own code: the leaf's split literals
//! in branch order (frame-relative; translated to absolute ONNX-level sites by
//! a [`SiteMap`] only at stream time), the β the leaf's bound was scored
//! under, the classes it discharges, and the frame's root α. The ledger has
//! no reader: nothing a hint says can reach a bound, a gate or a verdict.
//! Under Competition hint capture is off and every push site returns before
//! building a hint.

use std::cell::RefCell;
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use super::cover::{CoverLedger, CoverSite, LeafPath};

/// How a streamed leaf was closed (the offline checker re-derives each kind
/// with its own certificate).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeafClose {
    /// A strictly positive certified margin lower bound.
    Bound,
    /// A strictly positive certified emptiness slack.
    Empty,
    /// A nested frame's sub-domain token (its own leaves are not streamed).
    Subtree,
    /// A margin-free graph-lane close (an inverted certified interval or an
    /// input clip); its margin field is `NaN`.
    Lane,
}

impl LeafClose {
    /// Stable token for the proof bundle.
    #[must_use]
    pub const fn token(self) -> &'static str {
        match self {
            Self::Bound => "bound",
            Self::Empty => "empty",
            Self::Subtree => "subtree",
            Self::Lane => "lane",
        }
    }
}

/// One certified leaf, as a positional record (its frame-relative path, how
/// it closed and its certified margin). Kept for in-memory recording sinks;
/// the v2 bundle writes [`NodeRecord`]s instead, because a frame-relative
/// position cannot be resolved by an offline checker.
#[derive(Debug, Clone, PartialEq)]
pub struct LeafRecord {
    /// The leaf's path below the frame root.
    pub path: LeafPath,
    /// Its certified margin (or emptiness slack, or the subtree token's
    /// minimum margin).
    pub margin: f64,
    /// How it closed.
    pub close: LeafClose,
}

/// The verdict a proof stream closes with.
#[derive(Debug, Clone, PartialEq)]
pub struct VerdictRecord {
    /// Classes the verdict covers.
    pub classes: Vec<usize>,
    /// Smallest certified margin it rests on.
    pub min_margin: f64,
    /// Node records this process streamed.
    pub nodes: usize,
}

/// An ABSOLUTE ReLU site: the Relu op's index in the lane's op list (the key
/// of the bundle's `sites` record, which names the ONNX Relu node's
/// `input[0]`) and the flat C-order neuron index in that input tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AbsSite {
    /// Relu op index.
    pub op: u32,
    /// Flat neuron index in the Relu's input tensor.
    pub index: u32,
}

/// An absolute literal: `positive` is `z >= 0`, else `z <= 0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AbsLit {
    /// The site.
    pub site: AbsSite,
    /// Which closed half-space.
    pub positive: bool,
}

/// An absolute input-threshold literal (B2.2 input-split and lowdim leaves):
/// `x_index >= threshold` when `positive`, else `x_index <= threshold`, with
/// `index` the flat C-order index of the ONNX graph's first non-initializer
/// input (batch stripped: the bundle's `onnx-input0-flat/v1` space). The
/// threshold is written as its shortest round-trip decimal, so the checker's
/// `f64::from_str` reads back exactly this value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InputLit {
    /// Flat input index.
    pub index: u32,
    /// The threshold.
    pub threshold: f64,
    /// Which closed half-space.
    pub positive: bool,
}

/// A v2 node's close kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeClose {
    /// A certified bound on the node's region.
    Bound,
    /// A certified empty region (claims every clause).
    Empty,
    /// Columns retired at an internal (still open) node.
    Retire,
    /// A class discharged at the frame root.
    Root,
    /// A class the frozen-dual tier closed.
    Fd,
    /// A nested frame's leaf; the nested frame's nodes cover its cell.
    Subtree,
    /// A class a route closed without streaming a checkable certificate.
    Opaque,
}

impl NodeClose {
    /// Stable token for the bundle.
    #[must_use]
    pub const fn token(self) -> &'static str {
        match self {
            Self::Bound => "bound",
            Self::Empty => "empty",
            Self::Retire => "retire",
            Self::Root => "root",
            Self::Fd => "fd",
            Self::Subtree => "subtree",
            Self::Opaque => "opaque",
        }
    }
}

/// B2.3: a frozen-dual prune's checker hint — the state it was certified in
/// (a [`FdStateRecord`] id) and the dual point `(L0, L1)` handed to the
/// certified evaluator, aligned with the node's literals.
#[derive(Debug, Clone, PartialEq)]
pub struct FdNodeHint {
    /// The FD state id.
    pub state: u64,
    /// Sign-cut multipliers.
    pub l0: Vec<f64>,
    /// Far-cut multipliers.
    pub l1: Vec<f64>,
}

/// B2.3: one Relu op's relaxation of a frozen-dual state: the slope `λ` and
/// the certified box of every unstable neuron (hints: the checker rebuilds
/// its own state and may only borrow the slopes).
#[derive(Debug, Clone, PartialEq)]
pub struct FdLambdaRecord {
    /// Relu op (the `sites` id).
    pub op: u32,
    /// Neurons.
    pub idx: Vec<u32>,
    /// Slopes in `[0, 1]`.
    pub val: Vec<f64>,
    /// Certified box lower ends (diagnostic).
    pub lo: Vec<f64>,
    /// Certified box upper ends (diagnostic).
    pub hi: Vec<f64>,
}

/// B2.3: one frozen-dual state's identity and hints, streamed before its
/// prunes: the class, the sha256 of the certified state the prunes rest on,
/// its sizes and root, its split order as absolute sites, and its slopes.
#[derive(Debug, Clone, PartialEq)]
pub struct FdStateRecord {
    /// Process-unique id (the prunes' `FdNodeHint::state`).
    pub id: u64,
    /// Adversarial class of the margin `y_t - y_class`.
    pub class: usize,
    /// True class.
    pub t: usize,
    /// sha256 (hex) of the certified state content the prunes rest on.
    pub state_sha256: String,
    /// Generator count.
    pub n_gens: usize,
    /// Input symbols.
    pub in_gens: usize,
    /// Unstable rows.
    pub unstable: usize,
    /// `#fd-compact-checkable`: relaxation symbols the slope table covers
    /// (`unstable` for a full state; every unstable neuron of the frame for
    /// a ranked compact state streamed with its class relaxation).
    pub relaxed: usize,
    /// The certified state root.
    pub root: f64,
    /// The split order (depth `i` splits `order[i]`), absolute sites.
    pub order: Vec<AbsSite>,
    /// Slopes and boxes, per Relu op.
    pub lambda: Vec<FdLambdaRecord>,
    /// Prune nodes that follow.
    pub prunes: u64,
}

/// One v2 node: a region (the FULL literal set from the instance root, in
/// branch order) and the classes it discharges.
#[derive(Debug, Clone, PartialEq)]
pub struct NodeRecord {
    /// Process-unique id.
    pub id: u64,
    /// Input region (0: the lane's single box).
    pub region: u32,
    /// Literals in branch order.
    pub lits: Vec<AbsLit>,
    /// Input-threshold literals (B2.2; empty for a Relu-split region).
    pub xlits: Vec<InputLit>,
    /// Close kind.
    pub close: NodeClose,
    /// Producing route (diagnostic).
    pub src: &'static str,
    /// Discharged classes; `None` = every clause (emptiness).
    pub dis: Option<Vec<usize>>,
    /// β aligned with `lits` (hint).
    pub beta: Vec<f64>,
    /// α table id (hint).
    pub alpha: Option<u64>,
    /// The bound came from an ancestor region (hint).
    pub lift: bool,
    /// The producer's certified margin (hint).
    pub margin: f64,
    /// A site the producer found empty (hint).
    pub empty_site: Option<AbsSite>,
    /// B2.3: the frozen-dual prune hint (`None` for every other node).
    pub fd: Option<FdNodeHint>,
}

/// One α table entry set: slopes of the unstable neurons of one Relu site at
/// a frame root (`start: "*"`: valid for every objective).
#[derive(Debug, Clone, PartialEq)]
pub struct AlphaRecord {
    /// Table id (a node's `hint.alpha`).
    pub id: u64,
    /// Relu op.
    pub op: u32,
    /// Neurons.
    pub idx: Vec<u32>,
    /// Slopes.
    pub val: Vec<f64>,
}

/// One entry of the bundle's `sites` record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SiteTensorRecord {
    /// Id literals use (the Relu op index).
    pub id: u32,
    /// ONNX name of the Relu node's `input[0]`.
    pub name: String,
    /// Flat size.
    pub size: usize,
    /// Relu op index.
    pub op: u32,
}

/// The bundle's `sites` record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SitesRecord {
    /// Every Relu site of the network.
    pub tensors: Vec<SiteTensorRecord>,
}

/// A write-only proof hint sink. Every method returns `()`.
pub trait ProofSink: Send {
    /// One certified leaf, after the ledger counted it (positional).
    fn leaf(&mut self, leaf: &LeafRecord);
    /// One v2 node, after its frame's token was authorized.
    fn node(&mut self, node: &NodeRecord);
    /// One α table entry set.
    fn alpha(&mut self, alpha: &AlphaRecord);
    /// The network's Relu sites (once, before any node).
    fn sites(&mut self, sites: &SitesRecord);
    /// B2.3: one frozen-dual state's identity and hints (before its prunes).
    fn fd_state(&mut self, state: &FdStateRecord);
    /// The verdict, after publication.
    fn finish(&mut self, verdict: &VerdictRecord);
}

/// The Competition profile's sink: discards everything.
#[derive(Debug, Default, Clone, Copy)]
pub struct NullSink;

impl ProofSink for NullSink {
    fn leaf(&mut self, _leaf: &LeafRecord) {}

    fn node(&mut self, _node: &NodeRecord) {}

    fn alpha(&mut self, _alpha: &AlphaRecord) {}

    fn sites(&mut self, _sites: &SitesRecord) {}

    fn fd_state(&mut self, _state: &FdStateRecord) {}

    fn finish(&mut self, _verdict: &VerdictRecord) {}
}

/// An in-memory sink (tests and proof-mode audits).
#[derive(Debug, Default, Clone)]
pub struct RecordingSink {
    /// Leaves received, in order.
    pub leaves: Vec<LeafRecord>,
    /// Nodes received, in order.
    pub nodes: Vec<NodeRecord>,
    /// α records received, in order.
    pub alphas: Vec<AlphaRecord>,
    /// Sites records received.
    pub sites: Vec<SitesRecord>,
    /// B2.3: frozen-dual state records received, in order.
    pub fd_states: Vec<FdStateRecord>,
    /// The closing verdict, once received.
    pub verdict: Option<VerdictRecord>,
}

impl ProofSink for RecordingSink {
    fn leaf(&mut self, leaf: &LeafRecord) {
        self.leaves.push(leaf.clone());
    }

    fn node(&mut self, node: &NodeRecord) {
        self.nodes.push(node.clone());
    }

    fn alpha(&mut self, alpha: &AlphaRecord) {
        self.alphas.push(alpha.clone());
    }

    fn sites(&mut self, sites: &SitesRecord) {
        self.sites.push(sites.clone());
    }

    fn fd_state(&mut self, state: &FdStateRecord) {
        self.fd_states.push(state.clone());
    }

    fn finish(&mut self, verdict: &VerdictRecord) {
        self.verdict = Some(verdict.clone());
    }
}

/// The process's installed sink (first install wins).
static INSTALLED: OnceLock<Mutex<Box<dyn ProofSink>>> = OnceLock::new();

/// Node records handed to a sink by this process (the verdict record's count).
static NODES_STREAMED: AtomicU64 = AtomicU64::new(0);

/// Process-unique node ids.
static NEXT_NODE_ID: AtomicU64 = AtomicU64::new(0);

/// Process-unique α table ids.
static NEXT_ALPHA_ID: AtomicU64 = AtomicU64::new(0);

/// Whether this process captures v2 hints (set by the entry point's Proof
/// arm only).
static HINT_CAPTURE: AtomicBool = AtomicBool::new(false);

thread_local! {
    /// A thread-scoped sink that takes precedence over the process sink
    /// (see [`with_thread_sink`]).
    static SCOPED: RefCell<Option<Box<dyn ProofSink>>> = const { RefCell::new(None) };
}

#[cfg(test)]
thread_local! {
    /// Test-only capture override for this thread's NEW ledgers.
    static FORCE_CAPTURE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Test-only: force hint capture on (`Some(true)`), off, or back to the
/// process flag (`None`) for ledgers created on this thread.
#[cfg(test)]
pub(crate) fn force_hint_capture_for_test(v: Option<bool>) {
    FORCE_CAPTURE.with(|c| c.set(v));
}

/// Turn on v2 hint capture for this process. The ENTRY POINT calls this
/// under the Proof profile only (ratchet: it is called from
/// `ny-cli/src/commands/proof_bundle.rs` and nowhere else). Capture changes
/// what ledgers RECORD, never what the lane computes.
pub fn enable_hint_capture() {
    HINT_CAPTURE.store(true, Ordering::Relaxed);
}

/// Whether new hint ledgers capture (an observable; no verdict path reads it).
#[must_use]
pub fn hint_capture_enabled() -> bool {
    #[cfg(test)]
    if let Some(v) = FORCE_CAPTURE.with(std::cell::Cell::get) {
        return v;
    }
    HINT_CAPTURE.load(Ordering::Relaxed)
}

/// Install the process's proof sink. The ENTRY POINT calls this once, before
/// any lane runs; the FIRST call wins and a later one is ignored (returns
/// `false`), so no lane can swap the entry point's choice mid-run.
pub fn install_proof_sink(sink: Box<dyn ProofSink>) -> bool {
    INSTALLED.set(Mutex::new(sink)).is_ok()
}

/// Whether an entry point installed a sink in this process (an observable
/// for the entry point's own line; no verdict path reads it).
#[must_use]
pub fn proof_sink_installed() -> bool {
    INSTALLED.get().is_some()
}

/// Run `f` with `sink` as THIS thread's sink (it takes precedence over the
/// process sink for streams issued on this thread), then hand the sink back.
/// For in-process harnesses that must not share the process sink with
/// concurrently running work. Write-only like every sink.
pub fn with_thread_sink<R>(
    sink: Box<dyn ProofSink>,
    f: impl FnOnce() -> R,
) -> (R, Box<dyn ProofSink>) {
    SCOPED.with(|s| *s.borrow_mut() = Some(sink));
    let r = f();
    let back = SCOPED
        .with(|s| s.borrow_mut().take())
        .unwrap_or_else(|| Box::new(NullSink));
    (r, back)
}

fn with_installed(f: impl FnOnce(&mut dyn ProofSink)) {
    let mut f = Some(f);
    SCOPED.with(|s| {
        if let Ok(mut slot) = s.try_borrow_mut() {
            if let Some(sink) = slot.as_mut() {
                if let Some(f) = f.take() {
                    f(&mut **sink);
                }
            }
        }
    });
    let Some(f) = f else {
        return;
    };
    let Some(slot) = INSTALLED.get() else {
        return;
    };
    // A poisoned slot (a sink panicked mid-write) streams nothing more: the
    // sink is write-only, so losing hints can never touch a verdict.
    if let Ok(mut sink) = slot.lock() {
        f(&mut **sink);
    }
}

/// Stream every leaf of `ledger` into [`ProofSink::leaf`] of `sink`, in ledger
/// order, and return how many were streamed. The ledger must already have
/// PROVED its cover: a caller streams only what the gate accepted.
pub fn stream_cover_into(ledger: &CoverLedger, sink: &mut dyn ProofSink) -> usize {
    let mut n = 0usize;
    for record in ledger.records() {
        sink.leaf(&record);
        n += 1;
    }
    n
}

/// Stream the network's Relu sites into the installed sink (no-op without
/// one).
pub fn stream_sites(sites: &SitesRecord) {
    with_installed(|sink| sink.sites(sites));
}

/// A process-unique node id (B2.2 lane tapes share the margin-row stream's
/// id space, so one bundle never repeats an id).
pub(crate) fn next_node_id() -> u64 {
    NEXT_NODE_ID.fetch_add(1, Ordering::Relaxed)
}

/// A process-unique node id for a record the ENTRY POINT's bundle writer
/// adds itself (the whole-region claim it completes a published `unsat`
/// with, `ny-cli`'s `proof_bundle`): drawn from the same counter as every
/// lane's nodes, so one bundle never repeats an id even while a lane thread
/// is still streaming. Returns a fresh counter value, never anything a sink
/// received.
#[must_use]
pub fn reserve_node_id() -> u64 {
    next_node_id()
}

/// Count one node handed to a sink (the verdict record's node count).
pub(crate) fn note_node_streamed() {
    NODES_STREAMED.fetch_add(1, Ordering::Relaxed);
}

/// Run `f` against the installed sink (thread sink first, then the process
/// sink); a no-op without one. B2.2 lane tapes stream through this.
pub(crate) fn with_installed_sink(f: impl FnOnce(&mut dyn ProofSink)) {
    with_installed(f);
}

/// One split of a frame, frame-relative, in branch order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameSplit {
    /// A trunk split: the frame root's `(unstable-list layer, position)`.
    Trunk {
        /// Layer.
        layer: u32,
        /// Position in the layer's unstable list.
        pos: u32,
        /// `z >= 0`.
        positive: bool,
    },
    /// A head split `y_index`.
    Head {
        /// Head neuron.
        index: u32,
        /// `y >= 0`.
        positive: bool,
    },
}

/// A frame's site translation and root α, built at STREAM time from the
/// frame root (see `certify::margin_row`'s `SiteMap::from_root`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SiteMap {
    /// Per trunk layer: the Relu op and the unstable neuron list.
    trunk: Vec<(u32, Vec<u32>)>,
    /// The head Relu op.
    head_op: u32,
    /// The root α per site: `(op, neurons, slopes)`.
    alpha: Vec<(u32, Vec<u32>, Vec<f64>)>,
}

impl SiteMap {
    /// Assemble a map (crate-internal: built from a frame root).
    pub(crate) fn new(
        trunk: Vec<(u32, Vec<u32>)>,
        head_op: u32,
        alpha: Vec<(u32, Vec<u32>, Vec<f64>)>,
    ) -> Self {
        Self {
            trunk,
            head_op,
            alpha,
        }
    }

    fn lit(&self, s: FrameSplit) -> Option<AbsLit> {
        match s {
            FrameSplit::Trunk {
                layer,
                pos,
                positive,
            } => {
                let (op, unst) = self.trunk.get(layer as usize)?;
                Some(AbsLit {
                    site: AbsSite {
                        op: *op,
                        index: *unst.get(pos as usize)?,
                    },
                    positive,
                })
            }
            FrameSplit::Head { index, positive } => Some(AbsLit {
                site: AbsSite {
                    op: self.head_op,
                    index,
                },
                positive,
            }),
        }
    }

    fn cover_lit(&self, site: CoverSite, positive: bool) -> Option<AbsLit> {
        match site {
            CoverSite::Trunk { layer, pos } => self.lit(FrameSplit::Trunk {
                layer,
                pos,
                positive,
            }),
            CoverSite::Head { index } => self.lit(FrameSplit::Head { index, positive }),
            // #graph-bab-cover-ledger sites have no absolute translation in a
            // margin-row frame's map: untranslatable, so the caller claims the
            // cell opaque (the checker can only report it unconfirmed).
            CoverSite::GraphRelu { .. } | CoverSite::GraphPoint { .. } => None,
        }
    }

    /// Translate a frame path to absolute literals (branch order kept).
    #[must_use]
    pub fn absolute(&self, splits: &[FrameSplit]) -> Option<Vec<AbsLit>> {
        splits.iter().map(|s| self.lit(*s)).collect()
    }
}

/// A leaf hint (frame-relative, branch order).
#[derive(Debug, Clone, PartialEq)]
pub struct LeafHint {
    /// Splits below the frame root, branch order (the child's split last).
    pub splits: Vec<FrameSplit>,
    /// β aligned with `splits`.
    pub beta: Vec<f64>,
    /// How it closed.
    pub close: LeafClose,
    /// Classes discharged (ignored for `Empty` / `Subtree`).
    pub dis: Vec<usize>,
    /// The producer's certified margin.
    pub margin: f64,
    /// The closing bound came from an ancestor region.
    pub lift: bool,
    /// Index into `splits` of the site found empty.
    pub empty_split: Option<usize>,
}

/// A retirement hint: `dis` proved on the evaluated domain's region.
#[derive(Debug, Clone, PartialEq)]
pub struct RetireHint {
    /// The evaluated domain's splits, branch order.
    pub splits: Vec<FrameSplit>,
    /// β aligned with `splits`.
    pub beta: Vec<f64>,
    /// Newly retired classes.
    pub dis: Vec<usize>,
    /// The certified composite (smallest over `dis`).
    pub margin: f64,
}

/// A class the frame discharged before (or outside) its tree.
#[derive(Debug, Clone, PartialEq)]
pub enum PreHint {
    /// Certified root bound.
    Root {
        /// Class.
        class: usize,
        /// Certified bound.
        margin: f64,
    },
    /// Frozen-dual tier close.
    FrozenDual {
        /// Class.
        class: usize,
        /// Certified value.
        margin: f64,
    },
    /// B2.3: a frozen-dual close with its COMPLETE prune trace: streamed as
    /// one state record and one checkable node per prune instead of the
    /// claim above.
    FrozenDualProof(Box<super::fd_proof::FdProofTrace>),
    /// A route that streams no checkable certificate.
    Opaque {
        /// Classes.
        classes: Vec<usize>,
        /// Route token.
        route: &'static str,
    },
}

#[derive(Debug, Clone, PartialEq)]
enum FrameHint {
    /// A leaf, with its OWN slope table when the route searched per-domain
    /// slopes (the wave engine); `None` uses the frame's root α table.
    Leaf(LeafHint, Option<Vec<(u32, Vec<u32>, Vec<f64>)>>),
    Retire(RetireHint),
}

/// review(b2-fd): the most bytes of per-leaf slope tables one frame's hint
/// ledger keeps (Proof profile only; Competition captures nothing). A wave
/// leaf's own table covers EVERY searched site of the universe, so without a
/// ceiling the ledger grows as leaves × unstable neurons with no admission
/// (a cifar100-sized wave tree reaches tens of GB on a host with no memory
/// governor). Past the ceiling a leaf keeps its hint and streams under the
/// frame's root α table instead: the checker may then leave it unconfirmed,
/// never wrong (a hint can only cost a confirmation).
pub const LEAF_ALPHA_TABLE_BYTES: u64 = 1 << 30;

/// Bytes one per-leaf slope table holds (`u32` neuron + `f64` slope per
/// entry, plus the per-op vectors' headers).
fn leaf_alpha_bytes(alpha: &[(u32, Vec<u32>, Vec<f64>)]) -> u64 {
    alpha.iter().fold(0u64, |acc, (_, idx, val)| {
        acc.saturating_add(52)
            .saturating_add((idx.len() as u64).saturating_mul(4))
            .saturating_add((val.len() as u64).saturating_mul(8))
    })
}

/// A frame's WRITE-ONLY hint ledger. Private fields; no method returns hint
/// data (ratchet `hint_ledger_exposes_no_reader`). Consumed by
/// [`HintLedger::stream_into`].
#[derive(Debug, Clone, PartialEq)]
pub struct HintLedger {
    capture: bool,
    prefix: Vec<AbsLit>,
    hints: Vec<FrameHint>,
    pres: Vec<PreHint>,
    /// review(b2-fd): bytes of per-leaf slope tables kept so far, and the
    /// ceiling ([`LEAF_ALPHA_TABLE_BYTES`]; a test may lower it).
    leaf_alpha_bytes: u64,
    leaf_alpha_cap: u64,
}

impl Default for HintLedger {
    fn default() -> Self {
        Self::new()
    }
}

impl HintLedger {
    /// A ledger that captures iff hint capture is on for this process.
    #[must_use]
    pub fn new() -> Self {
        Self {
            capture: hint_capture_enabled(),
            prefix: Vec::new(),
            hints: Vec::new(),
            pres: Vec::new(),
            leaf_alpha_bytes: 0,
            leaf_alpha_cap: LEAF_ALPHA_TABLE_BYTES,
        }
    }

    /// review(b2-fd) TEST-ONLY: lower this ledger's per-leaf slope-table
    /// ceiling (the pin of the fallback past it).
    #[cfg(test)]
    pub(crate) fn set_leaf_alpha_cap_for_test(&mut self, bytes: u64) {
        self.leaf_alpha_cap = bytes;
    }

    /// Whether this ledger captures (callers skip BUILDING a hint otherwise).
    #[must_use]
    pub fn capturing(&self) -> bool {
        self.capture
    }

    /// Record one leaf hint (after the cover ledger counted the leaf).
    pub fn push_leaf(&mut self, hint: LeafHint) {
        if self.capture {
            self.hints.push(FrameHint::Leaf(hint, None));
        }
    }

    /// B2.3: record one leaf hint with the leaf's OWN slope table
    /// `(Relu op, neurons, slopes)` — the slopes its certified bound was
    /// computed under (the wave engine searches them per domain). Streamed as
    /// the leaf's own `hint.alpha` table; untrusted like every hint.
    ///
    /// review(b2-fd): the tables are bounded per frame by
    /// [`LEAF_ALPHA_TABLE_BYTES`]; a leaf whose table would pass the ceiling
    /// is kept as a plain leaf (the frame's root α table), never dropped.
    pub fn push_leaf_with_alpha(&mut self, hint: LeafHint, alpha: Vec<(u32, Vec<u32>, Vec<f64>)>) {
        if !self.capture {
            return;
        }
        let bytes = leaf_alpha_bytes(&alpha);
        let total = self.leaf_alpha_bytes.saturating_add(bytes);
        if total > self.leaf_alpha_cap {
            self.hints.push(FrameHint::Leaf(hint, None));
            return;
        }
        self.leaf_alpha_bytes = total;
        self.hints.push(FrameHint::Leaf(hint, Some(alpha)));
    }

    /// Record one retirement hint.
    pub fn push_retire(&mut self, hint: RetireHint) {
        if self.capture {
            self.hints.push(FrameHint::Retire(hint));
        }
    }

    /// Record one pre-tree class discharge.
    pub fn push_pre(&mut self, hint: PreHint) {
        if self.capture {
            self.pres.push(hint);
        }
    }

    /// Set the absolute literals of the frame root (a nested frame's parent
    /// leaf, translated with the PARENT's root).
    pub fn set_prefix(&mut self, prefix: Vec<AbsLit>) {
        if self.capture {
            self.prefix = prefix;
        }
    }

    /// Stream this frame's nodes (and its α table) into `sink`. `ledger` is
    /// the frame's PROVED cover: every recorded leaf is guaranteed a node —
    /// one whose hint was lost is streamed from its canonical path with the
    /// frame's `classes` (over-claiming costs confirmations, never cover).
    pub fn stream_into(
        self,
        ledger: &CoverLedger,
        classes: &[usize],
        sites: &SiteMap,
        sink: &mut dyn ProofSink,
    ) {
        if !self.capture {
            return;
        }
        let alpha_id = if sites.alpha.is_empty() {
            None
        } else {
            let id = NEXT_ALPHA_ID.fetch_add(1, Ordering::Relaxed);
            for (op, idx, val) in &sites.alpha {
                sink.alpha(&AlphaRecord {
                    id,
                    op: *op,
                    idx: idx.clone(),
                    val: val.clone(),
                });
            }
            Some(id)
        };
        let prefix = &self.prefix;
        let emit = |sink: &mut dyn ProofSink,
                    own: Vec<AbsLit>,
                    own_beta: &[f64],
                    close: NodeClose,
                    src: &'static str,
                    dis: Option<Vec<usize>>,
                    lift: bool,
                    margin: f64,
                    empty_site: Option<AbsSite>,
                    alpha: Option<u64>| {
            let mut lits = prefix.clone();
            let mut beta = vec![0.0; prefix.len()];
            if own_beta.len() == own.len() {
                beta.extend_from_slice(own_beta);
            } else {
                beta.extend(std::iter::repeat_n(0.0, own.len()));
            }
            lits.extend(own);
            let node = NodeRecord {
                id: NEXT_NODE_ID.fetch_add(1, Ordering::Relaxed),
                region: 0,
                lits,
                xlits: Vec::new(),
                close,
                src,
                dis,
                beta,
                alpha,
                lift,
                margin,
                empty_site,
                fd: None,
            };
            sink.node(&node);
            NODES_STREAMED.fetch_add(1, Ordering::Relaxed);
        };
        for pre in self.pres {
            match pre {
                PreHint::Root { class, margin } => emit(
                    sink,
                    Vec::new(),
                    &[],
                    NodeClose::Root,
                    "root",
                    Some(vec![class]),
                    false,
                    margin,
                    None,
                    alpha_id,
                ),
                PreHint::FrozenDual { class, margin } => emit(
                    sink,
                    Vec::new(),
                    &[],
                    NodeClose::Fd,
                    "fd",
                    Some(vec![class]),
                    false,
                    margin,
                    None,
                    alpha_id,
                ),
                PreHint::Opaque { classes, route } => emit(
                    sink,
                    Vec::new(),
                    &[],
                    NodeClose::Opaque,
                    route,
                    Some(classes),
                    false,
                    f64::NAN,
                    None,
                    alpha_id,
                ),
                // B2.3: a frozen-dual close as its state record and one
                // checkable node per prune (the prunes carry their own dual
                // point, not the frame's α table).
                PreHint::FrozenDualProof(trace) => trace.stream_into(prefix, sink),
            }
        }
        let mut streamed_leaf_sets: HashSet<Vec<AbsLit>> = HashSet::new();
        for h in self.hints {
            match h {
                FrameHint::Leaf(leaf, own_alpha) => {
                    let Some(own) = sites.absolute(&leaf.splits) else {
                        // Untranslatable: claim the frame cell as opaque so
                        // the cover stays claimed (the checker can only
                        // report it unconfirmed).
                        emit(
                            sink,
                            Vec::new(),
                            &[],
                            NodeClose::Opaque,
                            "untranslatable",
                            Some(classes.to_vec()),
                            false,
                            f64::NAN,
                            None,
                            alpha_id,
                        );
                        continue;
                    };
                    let mut set = own.clone();
                    set.sort_unstable();
                    streamed_leaf_sets.insert(set);
                    let empty_site = leaf.empty_split.and_then(|i| own.get(i)).map(|l| l.site);
                    let (close, dis) = match leaf.close {
                        LeafClose::Bound => (NodeClose::Bound, Some(leaf.dis.clone())),
                        LeafClose::Empty => (NodeClose::Empty, None),
                        LeafClose::Subtree => (NodeClose::Subtree, Some(Vec::new())),
                        // A margin-free graph-lane close states no certificate
                        // this bundle can carry: claim it opaque for every
                        // class (unconfirmed, never verified).
                        LeafClose::Lane => (NodeClose::Opaque, Some(classes.to_vec())),
                    };
                    // B2.3: a leaf certified under its own slopes streams
                    // them as its own α table.
                    let leaf_alpha = match own_alpha {
                        Some(tables) if !tables.is_empty() => {
                            let id = NEXT_ALPHA_ID.fetch_add(1, Ordering::Relaxed);
                            for (op, idx, val) in tables {
                                sink.alpha(&AlphaRecord { id, op, idx, val });
                            }
                            Some(id)
                        }
                        _ => alpha_id,
                    };
                    emit(
                        sink,
                        own,
                        &leaf.beta,
                        close,
                        "tree",
                        dis,
                        leaf.lift,
                        leaf.margin,
                        empty_site,
                        leaf_alpha,
                    );
                }
                FrameHint::Retire(r) => {
                    let Some(own) = sites.absolute(&r.splits) else {
                        emit(
                            sink,
                            Vec::new(),
                            &[],
                            NodeClose::Opaque,
                            "untranslatable",
                            Some(r.dis),
                            false,
                            f64::NAN,
                            None,
                            alpha_id,
                        );
                        continue;
                    };
                    emit(
                        sink,
                        own,
                        &r.beta,
                        NodeClose::Retire,
                        "retire",
                        Some(r.dis.clone()),
                        false,
                        r.margin,
                        None,
                        alpha_id,
                    );
                }
            }
        }
        // Every proved leaf gets a node, even one whose hint was lost.
        for rec in ledger.records() {
            let own: Option<Vec<AbsLit>> = rec
                .path
                .literals()
                .iter()
                .map(|l| sites.cover_lit(l.site, l.positive))
                .collect();
            let Some(own) = own else {
                emit(
                    sink,
                    Vec::new(),
                    &[],
                    NodeClose::Opaque,
                    "untranslatable",
                    Some(classes.to_vec()),
                    false,
                    f64::NAN,
                    None,
                    alpha_id,
                );
                continue;
            };
            let mut set = own.clone();
            set.sort_unstable();
            if streamed_leaf_sets.contains(&set) {
                continue;
            }
            let (close, dis) = match rec.close {
                LeafClose::Bound => (NodeClose::Bound, Some(classes.to_vec())),
                LeafClose::Empty => (NodeClose::Empty, None),
                LeafClose::Subtree => (NodeClose::Subtree, Some(Vec::new())),
                // As above: a graph-lane close is claimed opaque.
                LeafClose::Lane => (NodeClose::Opaque, Some(classes.to_vec())),
            };
            emit(
                sink,
                own,
                &[],
                close,
                "tree_unhinted",
                dis,
                true,
                rec.margin,
                None,
                alpha_id,
            );
        }
    }
}

/// Stream a PROVED frame into the installed sink (no-op without one or
/// without capture): its positional leaves ([`ProofSink::leaf`], for
/// recording sinks) and its v2 nodes. A lane calls this only after the
/// frame's token was authorized.
pub fn stream_frame(ledger: &CoverLedger, hints: HintLedger, classes: &[usize], sites: &SiteMap) {
    let capture = hints.capturing();
    with_installed(move |sink| {
        stream_cover_into(ledger, sink);
        if capture {
            hints.stream_into(ledger, classes, sites, sink);
        }
    });
}

/// review(eng14): a route that publishes `unsat` OUTSIDE every frame (no
/// cover ledger and no hint ledger — the separately sealed dense input-head
/// route) claims the classes its token covers as ONE `opaque` node over the
/// whole region (`route` is its `src`), so every class a published token
/// relies on is claimed in the stream: the checker then reports the claim
/// `unconfirmed`, never a cover gap. A no-op without capture or without a
/// sink; write-only, so a caller streams only after its token exists.
pub fn stream_opaque_claim(classes: Vec<usize>, route: &'static str) {
    if !hint_capture_enabled() || classes.is_empty() {
        return;
    }
    with_installed(move |sink| {
        let node = NodeRecord {
            id: NEXT_NODE_ID.fetch_add(1, Ordering::Relaxed),
            region: 0,
            lits: Vec::new(),
            xlits: Vec::new(),
            close: NodeClose::Opaque,
            src: route,
            dis: Some(classes),
            beta: Vec::new(),
            alpha: None,
            lift: false,
            margin: f64::NAN,
            empty_site: None,
            fd: None,
        };
        sink.node(&node);
        NODES_STREAMED.fetch_add(1, Ordering::Relaxed);
    });
}

/// Close the installed sink's stream with the PUBLISHED verdict (no-op
/// without a sink). The entry point calls this after RESULTS_FILE was written.
pub fn finish_stream(classes: Vec<usize>, min_margin: f64) {
    with_installed(|sink| {
        sink.finish(&VerdictRecord {
            classes,
            min_margin,
            nodes: NODES_STREAMED.load(Ordering::Relaxed) as usize,
        });
    });
}
