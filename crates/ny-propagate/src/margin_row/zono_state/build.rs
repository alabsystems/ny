// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The reverse-mode builder of the per-class frozen alpha-zonotope state.
//!
//! # The representation (exact real arithmetic)
//!
//! Fix the adversarial class `j` and, for every unstable ReLU neuron `k`
//! (`l_k < 0 < u_k` on its certified box), a slope `lambda_k in [0, 1]` and
//! `mu_k >= max((1 - lambda_k) u_k, -lambda_k l_k) / 2`. For every input `x`
//! in the declared box define the generator vector `e(x)`:
//!
//! * input symbols `e_i = (x_i - mid_i) / rad_i` for every non-degenerate
//!   input dimension, in `[-1, 1]` because `mid +- rad` covers the box;
//! * one relaxation symbol per unstable neuron,
//!   `e_k = (relu(z_k(x)) - lambda_k z_k(x) - mu_k) / mu_k`, in `[-1, 1]`
//!   because `relu(z) - lambda z in [0, 2 mu]` for `z in [l, u]`.
//!
//! Substituting `relu(z_k) = lambda_k z_k + mu_k + mu_k e_k` (unstable),
//! `relu(z) = z` (active) and `relu(z) = 0` (dead) into the network is an
//! IDENTITY, so every pre-activation and the margin `m_j = y_t - y_j` are
//! exactly affine in `e(x)`. The state stores, for the margin and for every
//! unstable neuron, the float coefficients of that affine form plus a
//! remainder that bounds the difference:
//!
//! ```text
//!   |m_j(x) - c_obj - d . e(x)|  <= r_obj
//!   |z_k(x) - c_k   - a_k . e(x)| <= r_k        for every x in the box.
//! ```
//!
//! # Direction-adaptive lambda
//!
//! The margin row is built FIRST, by one backward pass from the logits that
//! chooses `lambda_k = alpha_k` where the margin's coefficient on the neuron's
//! output is positive and the chord `s_k` otherwise
//! (`alpha_crown.py:2398-2430`, the P1/P1x replicas' `dir_adaptive_lambda`).
//! With that choice the state's root `c_obj - sum|d| - r_obj` is the
//! alpha-CROWN margin bound on the same boxes (up to rounding charges). Every
//! unstable layer's rows are then built with those SAME slopes by one
//! identity-seeded backward pass per layer (chunked), the rows the rank-1
//! split substitution needs. The slopes only have to lie in `[0, 1]`; how they
//! were chosen is irrelevant to soundness.
//!
//! # Rounding and model errors (the certificate)
//!
//! Every backward step keeps the invariant, per seed column `r`,
//!
//! ```text
//!   s_r . v_start(x) = sum_{t in frontier} L_t[:, r] . v_t(x)
//!                      + sum_k C_k[r] e_k(x) + K_r + eps_r(x),   |eps_r| <= delta_r
//! ```
//!
//! where `L_t`, `C_k`, `K_r` are the STORED floats. Each step moves one tensor
//! out of the frontier and charges into `delta_r` a bound on the exact
//! discrepancy it introduces, always as `|coefficient error| x mag`, with
//! `mag >= |v_t(x)|` the frame's certified magnitude:
//!
//! * affine ops: `gamma_K |L_t|^T (|W| mag_in)` for the float matrix product
//!   (Higham, ANY summation order, FMA included), `rel |L_t|^T (|W| mag_in)`
//!   plus `|L_t|^T bias_err` for the certified BatchNorm-fold model error, and
//!   `gamma_{n+2} (|K_old| + sum|L_t b|)` for the bias accumulation;
//! * ReLU: `2u |fl(lambda L)| mag_z` for the slope product, `3u |C|` for the
//!   relaxation-symbol products (coefficient and constant), and
//!   `gamma_{p+2}(|K_old| + sum|C|)` for the constant sum;
//! * merges (a tensor with two consumers): `2u |L_new|^T mag`;
//! * the input: `gamma_{n+3}(|K| + sum|L_i mid_i|)` for the centre and
//!   `2u sum|L_i rad_i|` for the input-symbol products.
//!
//! The published remainder is `next_up(delta * SAFE + TINY)`. Any NaN/Inf, a
//! shape mismatch, a memory-admission refusal or the deadline refuses the
//! whole build: nothing partial is ever returned.
//!
//! # The shared trunk (`#fd-shared-state`, ENGINE_PROGRAM M3)
//!
//! A row pass seeded at layer `m`'s pre-activation walks only the ops BEFORE
//! that tensor, so a row reads the slopes of the layers upstream of its own
//! and nothing else: the rows depend on the adversarial class ONLY through
//! those slopes. The per-class build therefore splits into
//!
//! * a TRUNK (`FrozenDualTrunk`): the slopes of every layer some unstable
//!   row may read, fixed ONCE for all classes (`TrunkRelax`), and every row
//!   built once under them; and
//! * a per-class TAIL (`ClassTail`): one margin pass that READS every trunk
//!   slope and re-chooses (direction-adaptively) only the TAIL layers, the
//!   layers no row reads (`tail_layers`, e.g. the head ReLU).
//!
//! The assembled state (`FrozenDualTrunk::state`) is exactly what
//! [`FrozenDualState::build`] computes for a margin pass whose slopes were
//! fixed in advance on the trunk layers. Every relaxation symbol `e_k` is
//! defined by ONE `(lambda_k, mu_k)` that every row and the margin row use,
//! and both halves of that are ENFORCED, fail-closed, rather than trusted:
//! the trunk's row passes refuse the moment one reaches a tail layer
//! (`Choose::Forbid`), and the assembly refuses a tail whose slope or `mu`
//! differs, in any bit, from the trunk's on every other layer. The assembly
//! also binds the pieces to each other: it refuses a tail of another class
//! than the one asked for, and a frame or tail whose generator layout is not
//! the trunk's. So the
//! containment claim and every charge above hold verbatim. How the trunk slopes were chosen (the joint
//! margin's direction-adaptive choice, one class's own choice, anything in
//! `[0, 1]`) is irrelevant to soundness and costs only tightness, the
//! "shared-lambda root loss". With the trunk chosen as one class's own slopes
//! (`TrunkRelax::anchor`) that class's shared state IS its per-class state,
//! bit for bit.

use std::time::Instant;

use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;

use crate::zono_core::rounding::{gamma, next_down, next_up, TINY, UNIT};
use crate::zono_core::ZonoRefusal;

use super::super::net::{conv_apply_backward, TwinNet, TwinOp};
use super::super::zono_bab::FROZEN_DUAL_MAX_DEPTH;
use super::frame::{FrozenDualFrame, ReluSite, MAX_DIM, SAFE};
use super::order::{split_score, SplitScoreScratch};

pub(crate) mod progressive;

/// Default live-state ceiling (bytes) before the headroom clamp.
pub const FROZEN_DUAL_DEFAULT_MAX_STATE_BYTES: u64 = 4 << 30;

/// Byte ceiling when the process memory headroom is unobservable (no cgroup
/// and no `MemAvailable`): competition runs uncapped, so this is the guard.
pub const FROZEN_DUAL_UNOBSERVED_HEADROOM_CAP: u64 = 4 << 30;

/// Default seed-column chunk of the per-layer row passes.
pub const FROZEN_DUAL_DEFAULT_CHUNK_COLS: usize = 256;

/// Build options.
#[derive(Debug, Clone, Copy)]
pub struct FrozenDualOptions {
    /// Refuse (`ZonoRefusal::Deadline`) once this passes.
    pub deadline: Option<Instant>,
    /// Configured live-state ceiling, clamped by [`admitted_bytes`].
    pub max_state_bytes: u64,
    /// Seed columns per backward pass of the row build (>= 1).
    pub chunk_cols: usize,
}

impl Default for FrozenDualOptions {
    fn default() -> Self {
        Self {
            deadline: None,
            max_state_bytes: FROZEN_DUAL_DEFAULT_MAX_STATE_BYTES,
            chunk_cols: FROZEN_DUAL_DEFAULT_CHUNK_COLS,
        }
    }
}

/// The byte budget actually admitted: the configured ceiling, clamped to half
/// the process memory headroom (cgroup- or `MemAvailable`-derived), or to
/// [`FROZEN_DUAL_UNOBSERVED_HEADROOM_CAP`] when no headroom is observable.
/// ny has no memory governor without a cgroup, so this IS the guard.
#[must_use]
pub fn admitted_bytes(max_bytes: u64) -> u64 {
    match crate::network::crown_memory::process_memory_headroom_bytes() {
        Some(headroom) => max_bytes.min(headroom / 2),
        None => max_bytes.min(FROZEN_DUAL_UNOBSERVED_HEADROOM_CAP),
    }
}

/// Estimated peak bytes of one state build (rows, objective, pass working set).
#[must_use]
pub fn estimate_state_bytes(frame: &FrozenDualFrame, chunk_cols: usize) -> u64 {
    let n = frame.n_gens as u64;
    let s = frame.n_unstable as u64;
    let rows = s.saturating_mul(n).saturating_mul(8);
    let per_row = s.saturating_mul(8 * 8);
    let chunk = chunk_cols.max(1) as u64;
    // Frontier matrices of a residual pass: at most ~6 live (tensor x chunk)
    // coefficient matrices, plus the input coefficient block and one assembly
    // row per column.
    let pass = (frame.max_width as u64)
        .saturating_mul(chunk)
        .saturating_mul(8 * 6)
        .saturating_add((frame.n_in as u64).saturating_mul(chunk).saturating_mul(8))
        .saturating_add(n.saturating_mul(chunk).saturating_mul(8));
    rows.saturating_add(per_row)
        .saturating_add(n.saturating_mul(8))
        .saturating_add(pass)
}

/// Heap storage for the per-layer slopes, including the outer vector tables.
fn estimated_relax_heap_bytes(frame: &FrozenDualFrame) -> u64 {
    (frame.n_unstable as u64)
        .saturating_mul(2 * size_of::<f64>() as u64)
        .saturating_add(
            (frame.layers.len() as u64).saturating_mul(2 * size_of::<Vec<f64>>() as u64),
        )
}

/// Estimated shared-trunk build peak. Unlike a class state, the trunk retains
/// nested slopes and a tail mask. Row validation also flattens both slopes
/// while the rows and nested slopes remain live.
#[must_use]
pub(crate) fn estimate_trunk_state_bytes(frame: &FrozenDualFrame, chunk_cols: usize) -> u64 {
    let n = frame.n_gens as u64;
    let s = frame.n_unstable as u64;
    let layers = frame.layers.len() as u64;
    let chunk = chunk_cols.clamp(1, MAX_DIM) as u64;
    let rows = s.saturating_mul(n).saturating_mul(8);
    let metadata = s.saturating_mul(
        4 * size_of::<f64>() as u64 + 3 * size_of::<usize>() as u64 + size_of::<ReluSite>() as u64,
    );
    let pass = (frame.max_width as u64)
        .saturating_mul(chunk)
        .saturating_mul(8 * 6)
        .saturating_add((frame.n_in as u64).saturating_mul(chunk).saturating_mul(8))
        .saturating_add(n.saturating_mul(chunk).saturating_mul(8));
    let validation = s.saturating_mul(2 * size_of::<f64>() as u64);
    let digest = s
        .saturating_mul(n)
        .div_ceil(DIGEST_CHUNK as u64)
        .saturating_mul(size_of::<u64>() as u64);
    rows.saturating_add(metadata)
        .saturating_add(estimated_relax_heap_bytes(frame))
        .saturating_add(layers.saturating_mul(size_of::<bool>() as u64))
        .saturating_add(size_of::<FrozenDualTrunk>() as u64)
        .saturating_add(pass.saturating_add(layers).max(validation).max(digest))
}

/// Allocated vector payload, excluding its inline header.
fn vector_bytes<T>(v: &Vec<T>) -> u64 {
    (v.capacity() as u64).saturating_mul(size_of::<T>() as u64)
}

/// Peak bytes of exact streaming row construction, including all pass
/// buffers, the current assembled chunk, retained search rows, objective,
/// class relaxation metadata and the parallel scorer's private buffers.
/// Reverse passes keep the requested chunk width; assembly and scoring use
/// at most eight rows at a time after that pass, before the next one starts.
#[must_use]
pub fn estimate_ranked_state_bytes(
    frame: &FrozenDualFrame,
    chunk_cols: usize,
    max_rows: usize,
) -> u64 {
    let n = frame.n_gens as u64;
    let s = frame.n_unstable as u64;
    let chunk = (chunk_cols.clamp(1, MAX_DIM) as u64).min(s).max(1);
    let retained = (max_rows.min(FROZEN_DUAL_MAX_DEPTH) as u64).min(s);
    let pass = (frame.max_width as u64)
        .saturating_mul(chunk)
        .saturating_mul(8 * 6)
        .saturating_add((frame.n_in as u64).saturating_mul(chunk).saturating_mul(8))
        .saturating_add(n.saturating_mul(chunk).saturating_mul(8));
    let batch = chunk.min(s).min(8);
    let rows = n.saturating_mul(retained).saturating_mul(8);
    // Three n-float vectors and two n-pair vectors need 56n bytes;
    // 80n also reserves temporary scratch for stable breakpoint sorting.
    let scoring = n.saturating_mul(batch).saturating_mul(80);
    // Scoring keeps only PassOut (input and relaxation coefficients plus
    // constants/deltas), the eight-row assembly buffer and scorer scratch.
    // Reverse frontier buffers have been released before this phase starts.
    let score_phase = (frame.n_in as u64)
        .saturating_add(s)
        .saturating_mul(chunk)
        .saturating_mul(8)
        .saturating_add(chunk.saturating_mul(16))
        .saturating_add(n.saturating_mul(batch).saturating_mul(8))
        .saturating_add(scoring);
    let metadata = s
        .saturating_mul(16)
        .saturating_add(retained.saturating_mul(256))
        .saturating_add((frame.layers.len() as u64).saturating_mul(128))
        .saturating_add((frame.mag.len() as u64).saturating_mul(96))
        .saturating_add(chunk.saturating_mul(32))
        .saturating_add(4096);
    pass.max(score_phase)
        .saturating_add(rows)
        .saturating_add(metadata)
        .saturating_add(n.saturating_mul(8))
}

/// Build statistics (telemetry only; never read by a bound).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FrozenDualBuildStats {
    /// Wall milliseconds of the whole build.
    pub build_ms: f64,
    /// Backward passes run (objective + row chunks).
    pub passes: usize,
    /// Seed columns walked (1 + unstable count).
    pub columns: usize,
    /// Admission estimate (bytes).
    pub estimate_bytes: u64,
    /// Admitted budget (bytes).
    pub budget_bytes: u64,
}

/// The certified per-class query state (see the module docs for the claim).
#[derive(Debug, Clone, PartialEq)]
pub struct FrozenDualState {
    /// True class.
    pub t: usize,
    /// Adversarial class `j` of the margin `y_t - y_j`.
    pub class: usize,
    /// Generator count `n` (input symbols first, then relaxation symbols in
    /// the frame's layer order).
    pub n_gens: usize,
    /// Input symbol count.
    pub n_in_gens: usize,
    /// Margin centre.
    pub c_obj: f64,
    /// Margin generator coefficients (`n`).
    pub d: Vec<f64>,
    /// Margin remainder.
    pub r_obj: f64,
    /// Unstable-neuron rows, row-major `S x n`; row `k` is the neuron whose
    /// relaxation symbol is generator column `enc[k]`.
    pub a: Vec<f64>,
    /// Row centres (`S`).
    pub c_in: Vec<f64>,
    /// Row remainders (`S`).
    pub r: Vec<f64>,
    /// Chosen slope per row, in `[0, 1]`.
    pub lam: Vec<f64>,
    /// Certified `mu` per row (`> 0`).
    pub mu: Vec<f64>,
    /// Certified pre-activation box per row, lower.
    pub lo: Vec<f64>,
    /// Certified pre-activation box per row, upper.
    pub hi: Vec<f64>,
    /// The row's own relaxation-symbol column (`n_in_gens + k`).
    pub enc: Vec<usize>,
    /// Frame layer index of the row.
    pub layer: Vec<usize>,
    /// Where that layer sits.
    pub site: Vec<ReluSite>,
    /// Neuron index within its layer.
    pub neuron: Vec<usize>,
    /// Build telemetry.
    pub stats: FrozenDualBuildStats,
}

/// Search-only compact state, retaining the exact first requested rows (up
/// to the established [`FROZEN_DUAL_MAX_DEPTH`])
/// rows of the full state's [`super::order::static_order`]. Generator columns
/// keep their original global indices; `state` row indices are compact.
///
/// Use `0..state.num_unstable()` with [`super::problem::FrozenDualProblem`].
/// The full-state point checker reconstructs witnesses from every unstable
/// row and must not be called on this compact state. Its certificates are
/// checked through the unchanged problem evaluator; row/objective equality
/// against the full state is independently tested.
#[derive(Debug, Clone, PartialEq)]
pub struct RankedFrozenDualState {
    pub state: FrozenDualState,
    /// Original full-state row indices, in exact static score order.
    pub original_rows: Vec<usize>,
    /// All unstable rows generated and scored, including discarded rows.
    pub total_unstable: usize,
}

struct RankedRow {
    original: usize,
    slot: usize,
    score: f64,
    centre: f64,
    remainder: f64,
    lam: f64,
    mu: f64,
    lo: f64,
    hi: f64,
    enc: usize,
    layer: usize,
    site: ReluSite,
    neuron: usize,
}

/// Descending score, then ascending original row index: exactly the full
/// static order, including infinities and signed zero.
fn ranked_cmp(a: &RankedRow, b: &RankedRow) -> std::cmp::Ordering {
    b.score
        .total_cmp(&a.score)
        .then(a.original.cmp(&b.original))
}

/// Per-class relaxation: slope and certified `mu` per unstable position, per
/// frame layer.
#[derive(Debug, Clone)]
pub(crate) struct ClassRelax {
    pub(crate) lam: Vec<Vec<f64>>,
    pub(crate) mu: Vec<Vec<f64>>,
}

/// Certified `mu >= max((1 - lambda) u, -lambda l) / 2` for `lambda in [0, 1]`,
/// `l < 0 < u`. `(1 - lambda) u` carries two roundings, `-lambda l` one, so the
/// float `mm` is within `u(|u| + |l|)` of the exact half-maximum; the charge
/// below is four times that plus a relative `4u` and the underflow floor.
#[must_use]
pub fn certified_mu(lambda: f64, l: f64, u: f64) -> f64 {
    let v1 = (1.0 - lambda) * u;
    let v2 = -lambda * l;
    let mm = 0.5 * v1.max(v2);
    next_up(mm + 4.0 * UNIT * mm.abs() + 4.0001 * UNIT * (u.abs() + l.abs()) + TINY)
}

impl ClassRelax {
    fn heap_bytes(&self) -> u64 {
        vector_bytes(&self.lam)
            .saturating_add(vector_bytes(&self.mu))
            .saturating_add(
                self.lam
                    .iter()
                    .map(vector_bytes)
                    .fold(0, u64::saturating_add),
            )
            .saturating_add(
                self.mu
                    .iter()
                    .map(vector_bytes)
                    .fold(0, u64::saturating_add),
            )
    }

    /// Every unstable neuron starts at its chord (overwritten by the margin
    /// pass wherever it chooses alpha).
    fn chord(frame: &FrozenDualFrame) -> Self {
        let mut lam = Vec::with_capacity(frame.layers.len());
        let mut mu = Vec::with_capacity(frame.layers.len());
        for layer in &frame.layers {
            let ls: Vec<f64> = layer.unst.iter().map(|&j| layer.s[j]).collect();
            let ms: Vec<f64> = layer
                .unst
                .iter()
                .zip(&ls)
                .map(|(&j, &lm)| certified_mu(lm, layer.l[j], layer.u[j]))
                .collect();
            lam.push(ls);
            mu.push(ms);
        }
        Self { lam, mu }
    }

    /// One slope and one `mu` per unstable position of every frame layer.
    fn matches(&self, frame: &FrozenDualFrame) -> bool {
        self.lam.len() == frame.layers.len()
            && self.mu.len() == frame.layers.len()
            && frame.layers.iter().enumerate().all(|(li, layer)| {
                self.lam[li].len() == layer.unst.len() && self.mu[li].len() == layer.unst.len()
            })
    }
}

/// B2.3 x #breakpoint-dual (`#fd-compact-checkable`, ENGINE_PROGRAM §1.7
/// B2): the CLASS RELAXATION of a ranked compact state.
///
/// A ranked state ([`FrozenDualState::build_ranked`]) keeps rows for the
/// retained split candidates only, but EVERY unstable neuron of the frame
/// keeps its relaxation symbol `e_k = (relu(z_k) - lambda_k z_k - mu_k) / mu_k`
/// at the slope the class's margin pass chose. The state's margin row (and
/// therefore every certified prune) rests on all of those slopes, the omitted
/// rows' included, so a proof trace of a compact state describes the
/// relaxation the tier certified with only when it carries this table: one
/// `(lambda, mu)` per unstable position of every frame layer, exactly as the
/// build used them (the margin pass writes them; the row passes only read).
///
/// Built only on request ([`FrozenDualState::build_ranked_with_relax`], the
/// Proof profile's hint capture); nothing in the search reads it. Before a
/// trace may use it, [`CompactRelax::verify`] re-derives the state's margin
/// row from the table alone and compares every bit, so a table that does not
/// describe the state can only leave the trace incomplete (the class keeps
/// its opaque claim), never stream a mismatched certificate.
#[derive(Debug, Clone)]
pub(crate) struct CompactRelax {
    /// Adversarial class of the margin the table was chosen for.
    class: usize,
    /// True class.
    t: usize,
    /// [`layout_digest`] of the frame the build ran on.
    layout: u64,
    /// Every layer's slope and `mu` exactly as the build used them.
    relax: ClassRelax,
}

impl CompactRelax {
    /// Owned heap bytes (the nested slope tables) plus the header.
    #[must_use]
    pub(crate) fn heap_bytes(&self) -> u64 {
        self.relax
            .heap_bytes()
            .saturating_add(size_of::<Self>() as u64)
    }

    /// Table entries: one per unstable position of every frame layer.
    #[must_use]
    pub(crate) fn len(&self) -> usize {
        self.relax.lam.iter().map(Vec::len).sum()
    }

    /// The slope and certified `mu` of unstable position `p` of frame layer
    /// `li` (`None` outside the table).
    #[must_use]
    pub(crate) fn slope(&self, li: usize, p: usize) -> Option<(f64, f64)> {
        let lam = *self.relax.lam.get(li)?.get(p)?;
        let mu = *self.relax.mu.get(li)?.get(p)?;
        Some((lam, mu))
    }

    /// Whether the table describes EXACTLY the relaxation `state` (a ranked
    /// compact state of `frame`) was certified with, re-derived rather than
    /// trusted. In order, each refusal naming itself:
    ///
    /// * `table_class` — the table was chosen for another class (or true
    ///   class) than the state's;
    /// * `table_layout` — another generator layout than the frame's;
    /// * `table_shape` — not one slope and one `mu` per unstable position
    ///   of every frame layer (a dropped or extra entry);
    /// * `table_range` — a slope outside `[0, 1]` or a `mu` that is not
    ///   finite and positive;
    /// * `table_row` — a retained row's slope, `mu`, box, column or site
    ///   differs, in any bit, from the table's entry for its neuron;
    /// * `table_mismatch` — the margin row replayed from the table ALONE
    ///   (one reverse pass that re-chooses no slope) is not the state's
    ///   `(c_obj, d, r_obj)` bit for bit: some slope the margin pass read
    ///   is not the table's, the omitted rows' included;
    /// * `table_choice` (review of `#fd-compact-checkable`) — the margin pass
    ///   RE-RUN from the chord start (re-choosing every slope exactly as the
    ///   build did) does not write the table entry for entry, or does not
    ///   reproduce the state's margin row. The replay above constrains only
    ///   the slopes the margin row reads with a NONZERO coefficient: a neuron
    ///   whose margin coefficient is exactly zero (every downstream path
    ///   through a dead neuron or a slope of 0) leaves no trace in
    ///   `(c_obj, d, r_obj)` whatever its table entry says, while the
    ///   retained rows of the layers above it read that slope. This check
    ///   pins those entries too, so EVERY entry is the build's own;
    /// * `deadline` / `table_replay` — a replay ran out of time or refused.
    ///
    /// The replay is the build's own margin pass (`margin_row` with
    /// `Choose::Never`): the same seed, the same ops, the same assembly, each
    /// slope read from the table where the build wrote it, so an exact table
    /// reproduces every bit. The re-choice is the same pass with
    /// `Choose::Every` over a fresh chord relaxation, exactly as
    /// `build_ranked` starts; the row passes only READ the relaxation, so
    /// what it writes is what every retained row read. Its transient second
    /// table (two `f64` per neuron) stays within the trace's own estimate
    /// (`fd_proof_table_bytes` charges more per neuron for the records that
    /// follow, and the two are never alive together).
    pub(crate) fn verify(
        &mut self,
        net: &TwinNet,
        frame: &FrozenDualFrame,
        state: &FrozenDualState,
        deadline: Option<Instant>,
    ) -> Result<(), &'static str> {
        if self.class != state.class || self.t != state.t || frame.t != state.t {
            return Err("table_class");
        }
        if self.layout != layout_digest(frame)
            || state.n_gens != frame.n_gens
            || state.n_in_gens != frame.n_in_gens
        {
            return Err("table_layout");
        }
        if !self.relax.matches(frame) {
            return Err("table_shape");
        }
        if self
            .relax
            .mu
            .iter()
            .flatten()
            .any(|m| !(m.is_finite() && *m > 0.0))
            || self
                .relax
                .lam
                .iter()
                .flatten()
                .any(|l| !(0.0..=1.0).contains(l))
        {
            return Err("table_range");
        }
        let rows = state.num_unstable();
        let same_len = [
            state.lam.len(),
            state.mu.len(),
            state.lo.len(),
            state.hi.len(),
            state.enc.len(),
            state.layer.len(),
            state.site.len(),
            state.neuron.len(),
        ]
        .iter()
        .all(|&n| n == rows);
        if !same_len {
            return Err("table_row");
        }
        for k in 0..rows {
            let li = state.layer[k];
            let j = state.neuron[k];
            let Some(layer) = frame.layers.get(li) else {
                return Err("table_row");
            };
            let p = layer.pos.get(j).copied().unwrap_or(usize::MAX);
            let Some((lam, mu)) = self.slope(li, p) else {
                return Err("table_row");
            };
            let agrees = state.enc[k] == layer.col0 + p
                && state.site[k] == layer.site
                && state.lam[k].to_bits() == lam.to_bits()
                && state.mu[k].to_bits() == mu.to_bits()
                && state.lo[k].to_bits() == layer.l[j].to_bits()
                && state.hi[k].to_bits() == layer.u[j].to_bits();
            if !agrees {
                return Err("table_row");
            }
        }
        let (c_obj, d, r_obj) = match margin_row(
            net,
            frame,
            &mut self.relax,
            self.class,
            Choose::Never,
            deadline,
        ) {
            Ok(m) => m,
            Err(ZonoRefusal::Deadline) => return Err("deadline"),
            Err(_) => return Err("table_replay"),
        };
        if c_obj.to_bits() != state.c_obj.to_bits()
            || r_obj.to_bits() != state.r_obj.to_bits()
            || !same_bits(&d, &state.d)
        {
            return Err("table_mismatch");
        }
        drop(d);
        let mut chosen = ClassRelax::chord(frame);
        let (c_obj, d, r_obj) =
            match margin_row(net, frame, &mut chosen, self.class, Choose::Every, deadline) {
                Ok(m) => m,
                Err(ZonoRefusal::Deadline) => return Err("deadline"),
                Err(_) => return Err("table_replay"),
            };
        let written = |a: &[Vec<f64>], b: &[Vec<f64>]| {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| same_bits(x, y))
        };
        if c_obj.to_bits() != state.c_obj.to_bits()
            || r_obj.to_bits() != state.r_obj.to_bits()
            || !same_bits(&d, &state.d)
            || !written(&chosen.lam, &self.relax.lam)
            || !written(&chosen.mu, &self.relax.mu)
        {
            return Err("table_choice");
        }
        Ok(())
    }

    /// TEST-ONLY fault injection: overwrite one recorded slope (its `mu`
    /// stays, so only the slope is wrong).
    #[cfg(test)]
    pub(crate) fn perturb_slope_for_test(&mut self, li: usize, p: usize, lam: f64) {
        self.relax.lam[li][p] = lam;
    }

    /// TEST-ONLY fault injection: drop one recorded entry (slope and `mu`).
    #[cfg(test)]
    pub(crate) fn drop_entry_for_test(&mut self, li: usize, p: usize) {
        self.relax.lam[li].remove(p);
        self.relax.mu[li].remove(p);
    }

    /// TEST-ONLY fault injection: relabel the table to another class.
    #[cfg(test)]
    pub(crate) fn relabel_for_test(&mut self, class: usize) {
        self.class = class;
    }

    /// TEST-ONLY: the table's class.
    #[cfg(test)]
    pub(crate) fn class_for_test(&self) -> usize {
        self.class
    }
}

/// Which frame layers a reverse pass re-CHOOSES the slope of (from the sign
/// of the incoming coefficient, written into the relaxation) instead of
/// reading the one it was given.
#[derive(Debug, Clone, Copy)]
enum Choose<'a> {
    /// No layer: every slope is read (the row passes).
    Never,
    /// Every layer (the per-class margin pass, and a trunk's joint pass).
    Every,
    /// Exactly the layers flagged `true` (a class tail over a shared trunk);
    /// every other layer's slope is read.
    Layers(&'a [bool]),
    /// No layer, and REACHING an unstable neuron of a layer flagged `true`
    /// refuses the pass: a shared trunk's row passes, whose rows must never
    /// read a tail slope (each class re-chooses those, so such a row would
    /// not share its relaxation symbols with every class's margin row).
    Forbid(&'a [bool]),
}

impl Choose<'_> {
    fn at(self, li: usize) -> bool {
        match self {
            Self::Never | Self::Forbid(_) => false,
            Self::Every => true,
            Self::Layers(mask) => mask.get(li).copied().unwrap_or(false),
        }
    }

    fn any(self) -> bool {
        match self {
            Self::Never | Self::Forbid(_) => false,
            Self::Every => true,
            Self::Layers(mask) => mask.contains(&true),
        }
    }

    fn forbids(self, li: usize) -> bool {
        match self {
            Self::Forbid(mask) => mask.get(li).copied().unwrap_or(false),
            Self::Never | Self::Every | Self::Layers(_) => false,
        }
    }
}

/// Output of one reverse pass.
struct PassOut {
    /// Coefficients on the flat input `x`, `(n_in, R)`.
    x: Array2<f64>,
    /// Relaxation-symbol coefficients per frame layer, `(unst, R)`; `None` for
    /// a layer the pass never reached.
    enew: Vec<Option<Array2<f64>>>,
    /// Constant per column.
    konst: Vec<f64>,
    /// Accumulated certified discrepancy per column (before `SAFE`).
    delta: Vec<f64>,
}

fn check_deadline(deadline: Option<Instant>) -> Result<(), ZonoRefusal> {
    if deadline.is_some_and(|d| Instant::now() >= d) {
        Err(ZonoRefusal::Deadline)
    } else {
        Ok(())
    }
}

/// Accumulate `v` into the frontier slot of tensor `t`, charging the float
/// addition (`|fl(a + b) - (a + b)| <= u |a + b| <= 2u |fl(a + b)|`) against
/// the tensor's magnitude.
fn merge(
    coef: &mut [Option<Array2<f64>>],
    t: usize,
    v: Array2<f64>,
    mag: &[f64],
    delta: &mut [f64],
) -> Result<(), ZonoRefusal> {
    match coef[t].take() {
        None => coef[t] = Some(v),
        Some(mut acc) => {
            if acc.dim() != v.dim() || acc.nrows() != mag.len() {
                return Err(ZonoRefusal::Shape(format!(
                    "frozen-dual merge into tensor {t}: {:?} vs {:?}",
                    acc.dim(),
                    v.dim()
                )));
            }
            acc += &v;
            charge_rows(delta, &acc, mag, 2.0 * UNIT)?;
            coef[t] = Some(acc);
        }
    }
    Ok(())
}

/// `delta[r] += factor * sum_j |m[j, r]| * w[j]`. A weight vector whose
/// length is not the row count REFUSES: a silently truncated zip would drop
/// charges, which is the one failure here that cannot fail loudly on its own.
fn charge_rows(
    delta: &mut [f64],
    m: &Array2<f64>,
    w: &[f64],
    factor: f64,
) -> Result<(), ZonoRefusal> {
    if w.len() != m.nrows() || delta.len() != m.ncols() {
        return Err(ZonoRefusal::Internal(format!(
            "frozen-dual charge: {} weights / {} deltas for a {:?} matrix",
            w.len(),
            delta.len(),
            m.dim()
        )));
    }
    let r = m.ncols();
    let mut acc = vec![0.0f64; r];
    for (row, &wj) in m.outer_iter().zip(w) {
        if wj == 0.0 {
            continue;
        }
        for (a, &v) in acc.iter_mut().zip(row.iter()) {
            *a += v.abs() * wj;
        }
    }
    for (d, a) in delta.iter_mut().zip(&acc) {
        *d += factor * a;
    }
    Ok(())
}

/// Fold `sum_j L[j, r] b[j]` into `konst[r]` with its charges: the float dot
/// and add (`gamma_{n+2}(|K_old| + sum|L b|)`) and the certified model error
/// `sum_j |L[j, r]| berr[j]`. `b` / `berr` are per ROW of `l`.
fn fold_bias(
    konst: &mut [f64],
    delta: &mut [f64],
    l: &Array2<f64>,
    b: &dyn Fn(usize) -> f64,
    berr: &dyn Fn(usize) -> f64,
) {
    debug_assert!(konst.len() == l.ncols() && delta.len() == l.ncols());
    let r = l.ncols();
    let n = l.nrows();
    let mut sum = vec![0.0f64; r];
    let mut abs = vec![0.0f64; r];
    let mut err = vec![0.0f64; r];
    for (j, row) in l.outer_iter().enumerate() {
        let bj = b(j);
        let ej = berr(j);
        if bj == 0.0 && ej == 0.0 {
            continue;
        }
        for c in 0..r {
            let v = row[c];
            if bj != 0.0 {
                let p = v * bj;
                sum[c] += p;
                abs[c] += p.abs();
            }
            if ej != 0.0 {
                err[c] += v.abs() * ej;
            }
        }
    }
    let g = gamma(n + 2);
    for c in 0..r {
        delta[c] += g * (konst[c].abs() + abs[c]) + err[c];
        konst[c] += sum[c];
    }
}

/// One reverse pass from tensor `start` seeded with `seed` (`(tsize[start], R)`).
///
/// A pass that chooses any slope is a margin pass: `R` must be 1, and each
/// unstable neuron of a layer `choose` names gets its slope from the sign of
/// the incoming coefficient, written into `relax`. Every other slope is read
/// (`Choose::Never`: `relax` is read only). With `Choose::Every` this is the
/// pre-M3 `choose = true` pass operation for operation (pinned by
/// `fd_trunk_build_refactor_is_bit_identical` against the frozen copy in
/// this file's test-only `pre_m3` module).
fn reverse_pass(
    net: &TwinNet,
    frame: &FrozenDualFrame,
    relax: &mut ClassRelax,
    choose: Choose<'_>,
    start: usize,
    seed: Array2<f64>,
    deadline: Option<Instant>,
) -> Result<PassOut, ZonoRefusal> {
    let r = seed.ncols();
    if choose.any() && r != 1 {
        return Err(ZonoRefusal::Internal(
            "frozen-dual: the slope-choosing pass takes one column".to_string(),
        ));
    }
    if start > net.ops.len() || seed.nrows() != net.tsize[start] {
        return Err(ZonoRefusal::Shape(format!(
            "frozen-dual seed: {} rows at tensor {start}",
            seed.nrows()
        )));
    }
    let mut coef: Vec<Option<Array2<f64>>> = vec![None; start + 1];
    coef[start] = Some(seed);
    let mut enew: Vec<Option<Array2<f64>>> = vec![None; frame.layers.len()];
    let mut konst = vec![0.0f64; r];
    let mut delta = vec![0.0f64; r];
    for k in (0..start).rev() {
        let Some(cur) = coef[k + 1].take() else {
            continue;
        };
        check_deadline(deadline)?;
        match &net.ops[k] {
            TwinOp::Conv(cv) => {
                let n_t = net.tsize[k + 1];
                let p = (cv.oshape.1 * cv.oshape.2).max(1);
                if cv.bias.len() * p != n_t || cv.bias_err.len() * p != n_t {
                    return Err(ZonoRefusal::Shape(format!("frozen-dual conv op {k} bias")));
                }
                fold_bias(
                    &mut konst,
                    &mut delta,
                    &cur,
                    &|j: usize| cv.bias[j / p],
                    &|j: usize| cv.bias_err[j / p],
                );
                let rel = cv.weight_rel_err;
                let g = next_up(gamma(cv.k_bwd + 2) + rel + gamma(cv.k_bwd + 2) * rel);
                charge_rows(&mut delta, &cur, &frame.wmag[k], g)?;
                let mut lnew = Array2::<f64>::zeros((net.tsize[cv.input], r));
                conv_apply_backward(cv, &cur, &mut lnew, false);
                merge(&mut coef, cv.input, lnew, &frame.mag[cv.input], &mut delta)?;
            }
            TwinOp::ChannelAffine {
                input,
                scale,
                shift,
                scale_rel_err,
                shift_err,
            } => {
                if shift.len() != cur.nrows()
                    || shift_err.len() != cur.nrows()
                    || scale.len() != cur.nrows()
                {
                    return Err(ZonoRefusal::Shape(format!("frozen-dual affine op {k}")));
                }
                fold_bias(
                    &mut konst,
                    &mut delta,
                    &cur,
                    &|j: usize| shift[j],
                    &|j: usize| shift_err[j],
                );
                // Model error of the scale: rel * sum_j |L_j| |s_j| mag_j.
                charge_rows(&mut delta, &cur, &frame.wmag[k], *scale_rel_err)?;
                let mut lnew = cur;
                for (mut row, &sj) in lnew.outer_iter_mut().zip(scale.iter()) {
                    row.mapv_inplace(|v| v * sj);
                }
                // One rounding per product, charged on the stored result.
                charge_rows(&mut delta, &lnew, &frame.mag[*input], 2.0 * UNIT)?;
                merge(&mut coef, *input, lnew, &frame.mag[*input], &mut delta)?;
            }
            TwinOp::Relu { input, .. } => {
                let li = frame.layer_of_op[k];
                let layer = frame.layers.get(li).ok_or_else(|| {
                    ZonoRefusal::Internal(format!("frozen-dual: relu op {k} has no frame layer"))
                })?;
                let n = layer.l.len();
                let np = layer.unst.len();
                if cur.nrows() != n || frame.mag[*input].len() != n {
                    return Err(ZonoRefusal::Shape(format!("frozen-dual relu op {k}")));
                }
                if np > 0 && choose.forbids(li) {
                    return Err(ZonoRefusal::Internal(format!(
                        "frozen-dual trunk: a row pass reaches tail layer {li}"
                    )));
                }
                let chosen = choose.at(li);
                let mut lz = Array2::<f64>::zeros((n, r));
                let mut cm = Array2::<f64>::zeros((np, r));
                let magz = &frame.mag[*input];
                for j in 0..n {
                    if layer.l[j] >= 0.0 {
                        // Active on the whole box: relu(z) = z exactly.
                        lz.row_mut(j).assign(&cur.row(j));
                        continue;
                    }
                    if layer.u[j] <= 0.0 {
                        // Dead on the whole box: relu(z) = 0 exactly.
                        continue;
                    }
                    let p = layer.pos[j];
                    if chosen {
                        let lam = if cur[[j, 0]] > 0.0 {
                            layer.alpha[j]
                        } else {
                            layer.s[j]
                        };
                        relax.lam[li][p] = lam;
                        relax.mu[li][p] = certified_mu(lam, layer.l[j], layer.u[j]);
                    }
                    let lam = relax.lam[li][p];
                    let mu = relax.mu[li][p];
                    for c in 0..r {
                        let v = cur[[j, c]];
                        let zc = lam * v;
                        let ec = v * mu;
                        lz[[j, c]] = zc;
                        cm[[p, c]] = ec;
                        delta[c] += 2.0 * UNIT * zc.abs() * magz[j] + 3.0 * UNIT * ec.abs();
                    }
                }
                let g = gamma(np + 2);
                for c in 0..r {
                    let mut sum = 0.0f64;
                    let mut abs = 0.0f64;
                    for p in 0..np {
                        let v = cm[[p, c]];
                        sum += v;
                        abs += v.abs();
                    }
                    delta[c] += g * (konst[c].abs() + abs);
                    konst[c] += sum;
                }
                enew[li] = Some(cm);
                merge(&mut coef, *input, lz, magz, &mut delta)?;
            }
            TwinOp::Add { lhs, rhs } => {
                let copy = cur.clone();
                merge(&mut coef, *lhs, cur, &frame.mag[*lhs], &mut delta)?;
                merge(&mut coef, *rhs, copy, &frame.mag[*rhs], &mut delta)?;
            }
            TwinOp::Flatten { input } => {
                merge(&mut coef, *input, cur, &frame.mag[*input], &mut delta)?;
            }
            TwinOp::Gemm {
                input,
                weight,
                bias,
                bias_err,
                weight_rel_err,
                shape,
            } => {
                let (no, ni) = *shape;
                if cur.nrows() != no
                    || weight.len() != no * ni
                    || bias.len() != no
                    || bias_err.len() != no
                {
                    return Err(ZonoRefusal::Shape(format!("frozen-dual gemm op {k}")));
                }
                fold_bias(
                    &mut konst,
                    &mut delta,
                    &cur,
                    &|o: usize| bias[o],
                    &|o: usize| bias_err[o],
                );
                let rel = *weight_rel_err;
                let g = next_up(gamma(no + 2) + rel + gamma(no + 2) * rel);
                charge_rows(&mut delta, &cur, &frame.wmag[k], g)?;
                let w = ArrayView2::from_shape((no, ni), weight.as_slice())
                    .map_err(|e| ZonoRefusal::Shape(format!("frozen-dual gemm view: {e}")))?;
                let prod = w.t().dot(&cur);
                let lnew = if prod.is_standard_layout() {
                    prod
                } else {
                    prod.as_standard_layout().into_owned()
                };
                merge(&mut coef, *input, lnew, &frame.mag[*input], &mut delta)?;
            }
        }
    }
    let x = match coef[0].take() {
        Some(x) => x,
        None => Array2::<f64>::zeros((net.n_in, r)),
    };
    Ok(PassOut {
        x,
        enew,
        konst,
        delta,
    })
}

/// Assemble column `col` of a pass into a generator row: returns the centre
/// and the published remainder, writing the `n_gens` coefficients into `row`.
fn assemble(
    frame: &FrozenDualFrame,
    out: &PassOut,
    col: usize,
    row: &mut [f64],
) -> Result<(f64, f64), ZonoRefusal> {
    row.fill(0.0);
    let mut prod_abs = 0.0f64;
    for (gi, &dim) in frame.in_cols.iter().enumerate() {
        let v = out.x[[dim, col]] * frame.rad[dim];
        row[gi] = v;
        prod_abs += v.abs();
    }
    let mut c = out.konst[col];
    let mut cabs = c.abs();
    for i in 0..frame.n_in {
        let term = out.x[[i, col]] * frame.mid[i];
        c += term;
        cabs += term.abs();
    }
    for (li, layer) in frame.layers.iter().enumerate() {
        if let Some(cm) = out.enew[li].as_ref() {
            for p in 0..layer.unst.len() {
                row[layer.col0 + p] = cm[[p, col]];
            }
        }
    }
    let charge = 2.0 * UNIT * prod_abs + gamma(frame.n_in + 3) * cabs;
    let rem = next_up((out.delta[col] + charge) * SAFE + TINY);
    if !c.is_finite() || !rem.is_finite() || row.iter().any(|v| !v.is_finite()) {
        return Err(ZonoRefusal::NonFinite("frozen-dual state row"));
    }
    Ok((c, rem))
}

/// `class` is an adversarial class of the frame's true class on `net`.
fn check_class(net: &TwinNet, frame: &FrozenDualFrame, class: usize) -> Result<(), ZonoRefusal> {
    let t = frame.t;
    if class >= frame.n_out || class == t || frame.n_out != net.n_out {
        return Err(ZonoRefusal::Shape(format!(
            "frozen-dual: class {class} vs true class {t} of {}",
            frame.n_out
        )));
    }
    Ok(())
}

/// The frame was built for `net`.
fn check_frame(net: &TwinNet, frame: &FrozenDualFrame) -> Result<(), ZonoRefusal> {
    if frame.out_tensor != net.ops.len() || frame.mag.len() != net.ops.len() + 1 {
        return Err(ZonoRefusal::Internal(
            "frozen-dual: frame was built for a different net".to_string(),
        ));
    }
    Ok(())
}

/// The margin row of `y_t - y_class`: ONE reverse pass from the logits under
/// `relax`, re-choosing the slopes of the layers `choose` names (written into
/// `relax`). Returns the centre, the `n_gens` coefficients and the published
/// remainder.
fn margin_row(
    net: &TwinNet,
    frame: &FrozenDualFrame,
    relax: &mut ClassRelax,
    class: usize,
    choose: Choose<'_>,
    deadline: Option<Instant>,
) -> Result<(f64, Vec<f64>, f64), ZonoRefusal> {
    let mut seed = Array2::<f64>::zeros((frame.n_out, 1));
    seed[[frame.t, 0]] = 1.0;
    seed[[class, 0]] = -1.0;
    let out = reverse_pass(net, frame, relax, choose, frame.out_tensor, seed, deadline)?;
    let mut d = vec![0.0f64; frame.n_gens];
    let (c_obj, r_obj) = assemble(frame, &out, 0, &mut d)?;
    Ok((c_obj, d, r_obj))
}

/// Every unstable row of a state. Depends on the adversarial class ONLY
/// through the slopes the row passes read (see [`tail_layers`]).
#[derive(Debug, Clone)]
struct Rows {
    a: Vec<f64>,
    c_in: Vec<f64>,
    r: Vec<f64>,
    lo: Vec<f64>,
    hi: Vec<f64>,
    enc: Vec<usize>,
    layer: Vec<usize>,
    site: Vec<ReluSite>,
    neuron: Vec<usize>,
    /// Backward passes run.
    passes: usize,
}

impl Rows {
    /// Cloning a vector copies its initialized elements, not spare capacity.
    fn payload_bytes(&self, capacity: bool) -> u64 {
        fn count<T>(v: &Vec<T>, capacity: bool) -> u64 {
            (if capacity { v.capacity() } else { v.len() } as u64)
                .saturating_mul(size_of::<T>() as u64)
        }
        count(&self.a, capacity)
            .saturating_add(count(&self.c_in, capacity))
            .saturating_add(count(&self.r, capacity))
            .saturating_add(count(&self.lo, capacity))
            .saturating_add(count(&self.hi, capacity))
            .saturating_add(count(&self.enc, capacity))
            .saturating_add(count(&self.layer, capacity))
            .saturating_add(count(&self.site, capacity))
            .saturating_add(count(&self.neuron, capacity))
    }
}

/// One identity-seeded pass per unstable layer (chunked) under `relax`, which
/// every pass only READS. `forbid` (per frame layer; empty for a per-class
/// build) names the layers no row may read: a pass that reaches one refuses
/// the whole build (`Choose::Forbid`).
fn build_rows(
    net: &TwinNet,
    frame: &FrozenDualFrame,
    relax: &mut ClassRelax,
    chunk: usize,
    deadline: Option<Instant>,
    forbid: &[bool],
) -> Result<Rows, ZonoRefusal> {
    let choose = if forbid.is_empty() {
        Choose::Never
    } else {
        Choose::Forbid(forbid)
    };
    let n = frame.n_gens;
    let s_total = frame.n_unstable;
    let mut passes = 0usize;
    let mut a = vec![0.0f64; s_total * n];
    let mut c_in = vec![0.0f64; s_total];
    let mut r = vec![0.0f64; s_total];
    let mut lo = vec![0.0f64; s_total];
    let mut hi = vec![0.0f64; s_total];
    let mut enc = vec![0usize; s_total];
    let mut layer_of = vec![0usize; s_total];
    let mut site = vec![ReluSite::Head; s_total];
    let mut neuron = vec![0usize; s_total];
    for (li, layer) in frame.layers.iter().enumerate() {
        let np = layer.unst.len();
        let base = layer.col0 - frame.n_in_gens;
        for (p, &j) in layer.unst.iter().enumerate() {
            let k = base + p;
            lo[k] = layer.l[j];
            hi[k] = layer.u[j];
            enc[k] = layer.col0 + p;
            layer_of[k] = li;
            site[k] = layer.site;
            neuron[k] = j;
        }
        let width = net.tsize[layer.input];
        let mut p0 = 0usize;
        while p0 < np {
            check_deadline(deadline)?;
            let w = (np - p0).min(chunk);
            let mut seed = Array2::<f64>::zeros((width, w));
            for c in 0..w {
                seed[[layer.unst[p0 + c], c]] = 1.0;
            }
            let out = reverse_pass(net, frame, relax, choose, layer.input, seed, deadline)?;
            passes += 1;
            for c in 0..w {
                let k = base + p0 + c;
                let (ck, rk) = assemble(frame, &out, c, &mut a[k * n..(k + 1) * n])?;
                // A row may not reference its own or any downstream
                // relaxation symbol: the pass stops at the layer's input.
                if a[k * n + enc[k]] != 0.0 {
                    return Err(ZonoRefusal::Internal(format!(
                        "frozen-dual: row {k} references its own symbol"
                    )));
                }
                c_in[k] = ck;
                r[k] = rk;
            }
            p0 += w;
        }
    }
    Ok(Rows {
        a,
        c_in,
        r,
        lo,
        hi,
        enc,
        layer: layer_of,
        site,
        neuron,
        passes,
    })
}

/// The slope and certified `mu` of every unstable row under `relax`, in row
/// order (row `k` of layer `li` at position `p` is `col0 - n_in_gens + p`).
fn row_relax(frame: &FrozenDualFrame, relax: &ClassRelax) -> (Vec<f64>, Vec<f64>) {
    let s_total = frame.n_unstable;
    let mut lam = vec![0.0f64; s_total];
    let mut mu = vec![0.0f64; s_total];
    for (li, layer) in frame.layers.iter().enumerate() {
        let base = layer.col0 - frame.n_in_gens;
        let n = layer.unst.len();
        lam[base..base + n].copy_from_slice(&relax.lam[li][..n]);
        mu[base..base + n].copy_from_slice(&relax.mu[li][..n]);
    }
    (lam, mu)
}

/// Every slope in `[0, 1]` and every `mu` finite and positive, or refuse.
fn check_relaxation(lam: &[f64], mu: &[f64]) -> Result<(), ZonoRefusal> {
    if mu.iter().any(|m| !(m.is_finite() && *m > 0.0))
        || lam.iter().any(|l| !(0.0..=1.0).contains(l))
    {
        return Err(ZonoRefusal::NonFinite("frozen-dual relaxation"));
    }
    Ok(())
}

/// The certified root bound `c_obj - sum|d| - r_obj` of a margin row over
/// `n_gens` generators, rounded down with its own summation charge
/// ([`FrozenDualState::root_bound`] and [`ClassTail::root_bound`] are this,
/// so a tail's root IS the root of the state it assembles into).
fn root_bound_of(c_obj: f64, d: &[f64], r_obj: f64, n_gens: usize) -> f64 {
    let absd: f64 = d.iter().map(|v| v.abs()).sum();
    let raw = c_obj - absd - r_obj;
    let charge = gamma(n_gens + 4) * (c_obj.abs() + absd + r_obj) * SAFE;
    let v = next_down(next_down(raw - charge) - TINY);
    if v.is_finite() {
        v
    } else {
        f64::NEG_INFINITY
    }
}

impl FrozenDualState {
    /// Build the certified state of the margin `y_t - y_class` over `frame`.
    ///
    /// Fail-closed: any refusal returns `Err` and nothing is published.
    ///
    /// ENGINE_PROGRAM M3 factored the body into `margin_row` and
    /// `build_rows` (shared with the crate's `FrozenDualTrunk`) without moving
    /// an operation: `fd_trunk_build_refactor_is_bit_identical` pins every
    /// field against the frozen pre-M3 copy in this file's test-only `pre_m3`
    /// module.
    pub fn build(
        net: &TwinNet,
        frame: &FrozenDualFrame,
        class: usize,
        opts: &FrozenDualOptions,
    ) -> Result<Self, ZonoRefusal> {
        let t0 = Instant::now();
        check_class(net, frame, class)?;
        check_frame(net, frame)?;
        let chunk = opts.chunk_cols.clamp(1, MAX_DIM);
        let estimate = estimate_state_bytes(frame, chunk);
        let budget = admitted_bytes(opts.max_state_bytes);
        if estimate > budget {
            return Err(ZonoRefusal::Memory {
                estimate_bytes: estimate,
                budget_bytes: budget,
            });
        }
        check_deadline(opts.deadline)?;
        let mut relax = ClassRelax::chord(frame);

        // ---- the margin row (chooses lambda) -----------------------------
        let margin = margin_row(net, frame, &mut relax, class, Choose::Every, opts.deadline)?;

        // ---- one identity-seeded pass per unstable layer (chunked) -------
        let rows = build_rows(net, frame, &mut relax, chunk, opts.deadline, &[])?;
        let (lam, mu) = row_relax(frame, &relax);
        check_relaxation(&lam, &mu)?;
        let passes = 1 + rows.passes;
        let build_ms = t0.elapsed().as_secs_f64() * 1e3;
        Ok(Self::from_parts(
            frame,
            class,
            margin,
            rows,
            (lam, mu),
            FrozenDualBuildStats {
                build_ms,
                passes,
                columns: 1 + frame.n_unstable,
                estimate_bytes: estimate,
                budget_bytes: budget,
            },
        ))
    }

    /// Assemble a state from its margin row, its rows and its row slopes.
    fn from_parts(
        frame: &FrozenDualFrame,
        class: usize,
        margin: (f64, Vec<f64>, f64),
        rows: Rows,
        relax: (Vec<f64>, Vec<f64>),
        stats: FrozenDualBuildStats,
    ) -> Self {
        let (c_obj, d, r_obj) = margin;
        let (lam, mu) = relax;
        Self {
            t: frame.t,
            class,
            n_gens: frame.n_gens,
            n_in_gens: frame.n_in_gens,
            c_obj,
            d,
            r_obj,
            a: rows.a,
            c_in: rows.c_in,
            r: rows.r,
            lam,
            mu,
            lo: rows.lo,
            hi: rows.hi,
            enc: rows.enc,
            layer: rows.layer,
            site: rows.site,
            neuron: rows.neuron,
            stats,
        }
    }

    /// Number of unstable-neuron rows `S`.
    #[must_use]
    pub fn num_unstable(&self) -> usize {
        self.c_in.len()
    }

    /// Row `k` (`n_gens` coefficients).
    #[must_use]
    pub fn row(&self, k: usize) -> &[f64] {
        &self.a[k * self.n_gens..(k + 1) * self.n_gens]
    }

    /// The certified root bound `c_obj - sum|d| - r_obj`, rounded down with
    /// its own summation charge: a valid lower bound of the margin over the
    /// whole box, equal to alpha-CROWN on the frame's boxes up to rounding.
    #[must_use]
    pub fn root_bound(&self) -> f64 {
        root_bound_of(self.c_obj, &self.d, self.r_obj, self.n_gens)
    }

    /// Largest row remainder.
    #[must_use]
    pub fn max_row_remainder(&self) -> f64 {
        self.r.iter().copied().fold(0.0, f64::max)
    }

    /// Generate and score every row using the full builder's objective,
    /// reverse chunks, assembly arithmetic and exact split score, retaining
    /// only the best rows, up to [`FROZEN_DUAL_MAX_DEPTH`]. Ties use the
    /// original row index.
    ///
    /// This is a search-only state: omitted rows still own their original
    /// generator columns. See [`RankedFrozenDualState`] for replay guidance.
    pub fn build_ranked(
        net: &TwinNet,
        frame: &FrozenDualFrame,
        class: usize,
        opts: &FrozenDualOptions,
        max_rows: usize,
    ) -> Result<RankedFrozenDualState, ZonoRefusal> {
        Self::build_ranked_inner(net, frame, class, opts, max_rows, false).map(|(state, _)| state)
    }

    /// B2.3 x #breakpoint-dual (`#fd-compact-checkable`): [`Self::build_ranked`]
    /// operation for operation, that ALSO hands back the compact state's
    /// CLASS RELAXATION ([`CompactRelax`]: the slope and certified `mu` of
    /// every unstable neuron of the frame, the omitted ones included).
    /// Crate-internal: only the frozen-dual tier asks, and only while the
    /// Proof profile's hint capture is on; Competition calls
    /// [`Self::build_ranked`], which drops the relaxation as it always did.
    pub(crate) fn build_ranked_with_relax(
        net: &TwinNet,
        frame: &FrozenDualFrame,
        class: usize,
        opts: &FrozenDualOptions,
        max_rows: usize,
    ) -> Result<(RankedFrozenDualState, CompactRelax), ZonoRefusal> {
        let (state, relax) = Self::build_ranked_inner(net, frame, class, opts, max_rows, true)?;
        let relax = relax.ok_or_else(|| {
            ZonoRefusal::Internal("frozen-dual: the compact relaxation was not kept".to_string())
        })?;
        Ok((
            state,
            CompactRelax {
                class,
                t: frame.t,
                layout: layout_digest(frame),
                relax,
            },
        ))
    }

    /// The ranked build; `keep_relax` hands the class relaxation back (it is
    /// otherwise dropped where it always was, at the end of the build).
    fn build_ranked_inner(
        net: &TwinNet,
        frame: &FrozenDualFrame,
        class: usize,
        opts: &FrozenDualOptions,
        max_rows: usize,
        keep_relax: bool,
    ) -> Result<(RankedFrozenDualState, Option<ClassRelax>), ZonoRefusal> {
        let t0 = Instant::now();
        let t = frame.t;
        if class >= frame.n_out || class == t || frame.n_out != net.n_out {
            return Err(ZonoRefusal::Shape(format!(
                "frozen-dual: class {class} vs true class {t} of {}",
                frame.n_out
            )));
        }
        if frame.out_tensor != net.ops.len() || frame.mag.len() != net.ops.len() + 1 {
            return Err(ZonoRefusal::Internal(
                "frozen-dual: frame was built for a different net".to_string(),
            ));
        }
        let chunk = opts.chunk_cols.clamp(1, MAX_DIM);
        let n = frame.n_gens;
        let total = frame.n_unstable;
        let retain = max_rows.min(FROZEN_DUAL_MAX_DEPTH).min(total);
        let estimate = estimate_ranked_state_bytes(frame, chunk, retain);
        let budget = admitted_bytes(opts.max_state_bytes);
        if estimate > budget {
            return Err(ZonoRefusal::Memory {
                estimate_bytes: estimate,
                budget_bytes: budget,
            });
        }
        check_deadline(opts.deadline)?;
        let mut relax = ClassRelax::chord(frame);
        let mut passes = 0usize;

        // Preserve the adaptive objective pass and its exact arithmetic.
        let mut seed = Array2::<f64>::zeros((frame.n_out, 1));
        seed[[t, 0]] = 1.0;
        seed[[class, 0]] = -1.0;
        let out = reverse_pass(
            net,
            frame,
            &mut relax,
            Choose::Every,
            frame.out_tensor,
            seed,
            opts.deadline,
        )?;
        passes += 1;
        let mut d = vec![0.0f64; n];
        let (c_obj, r_obj) = assemble(frame, &out, 0, &mut d)?;
        drop(out);
        check_deadline(opts.deadline)?;
        if relax
            .mu
            .iter()
            .flatten()
            .any(|m| !(m.is_finite() && *m > 0.0))
            || relax.lam.iter().flatten().any(|l| !(0.0..=1.0).contains(l))
        {
            return Err(ZonoRefusal::NonFinite("frozen-dual relaxation"));
        }
        let base_score = c_obj - d.iter().map(|v| v.abs()).sum::<f64>();
        let mut a = vec![0.0f64; retain * n];
        let mut rows: Vec<RankedRow> = Vec::with_capacity(retain);

        for (li, layer) in frame.layers.iter().enumerate() {
            let np = layer.unst.len();
            let base = layer.col0 - frame.n_in_gens;
            let width = net.tsize[layer.input];
            let mut p0 = 0usize;
            while p0 < np {
                check_deadline(opts.deadline)?;
                let w = (np - p0).min(chunk);
                let mut seed = Array2::<f64>::zeros((width, w));
                for c in 0..w {
                    seed[[layer.unst[p0 + c], c]] = 1.0;
                }
                let out = reverse_pass(
                    net,
                    frame,
                    &mut relax,
                    Choose::Never,
                    layer.input,
                    seed,
                    opts.deadline,
                )?;
                passes += 1;

                // PassOut stays alive; its reverse frontier has been dropped.
                // Bounded parallel score batches avoid one scorer per full
                // reverse seed chunk while retaining the original BLAS shape.
                let batch = w.min(8);
                let mut assembled = vec![0.0f64; batch * n];
                let mut centres = vec![0.0f64; batch];
                let mut remainders = vec![0.0f64; batch];
                let mut c0 = 0usize;
                while c0 < w {
                    check_deadline(opts.deadline)?;
                    let count = (w - c0).min(batch);
                    for i in 0..count {
                        let c = c0 + i;
                        let k = base + p0 + c;
                        let row = &mut assembled[i * n..(i + 1) * n];
                        let (ck, rk) = assemble(frame, &out, c, row)?;
                        let enc = layer.col0 + p0 + c;
                        if row[enc] != 0.0 {
                            return Err(ZonoRefusal::Internal(format!(
                                "frozen-dual: row {k} references its own symbol"
                            )));
                        }
                        centres[i] = ck;
                        remainders[i] = rk;
                    }
                    check_deadline(opts.deadline)?;
                    let scores: Vec<f64> = (0..count)
                        .into_par_iter()
                        .map_init(
                            || SplitScoreScratch::new(n),
                            |scratch, i| {
                                let p = p0 + c0 + i;
                                split_score(
                                    &assembled[i * n..(i + 1) * n],
                                    &d,
                                    c_obj,
                                    base_score,
                                    relax.lam[li][p],
                                    relax.mu[li][p],
                                    centres[i],
                                    layer.col0 + p,
                                    scratch,
                                )
                            },
                        )
                        .collect();
                    check_deadline(opts.deadline)?;
                    for i in 0..count {
                        let p = p0 + c0 + i;
                        let j = layer.unst[p];
                        let mut candidate = RankedRow {
                            original: base + p,
                            slot: rows.len(),
                            score: scores[i],
                            centre: centres[i],
                            remainder: remainders[i],
                            lam: relax.lam[li][p],
                            mu: relax.mu[li][p],
                            lo: layer.l[j],
                            hi: layer.u[j],
                            enc: layer.col0 + p,
                            layer: li,
                            site: layer.site,
                            neuron: j,
                        };
                        let slot = if rows.len() < retain {
                            let slot = rows.len();
                            rows.push(candidate);
                            Some(slot)
                        } else if let Some((worst, prior)) = rows
                            .iter()
                            .enumerate()
                            .max_by(|(_, x), (_, y)| ranked_cmp(x, y))
                        {
                            if ranked_cmp(&candidate, prior).is_lt() {
                                candidate.slot = prior.slot;
                                let slot = candidate.slot;
                                rows[worst] = candidate;
                                Some(slot)
                            } else {
                                None
                            }
                        } else {
                            None
                        };
                        if let Some(slot) = slot {
                            a[slot * n..(slot + 1) * n]
                                .copy_from_slice(&assembled[i * n..(i + 1) * n]);
                        }
                    }
                    c0 += count;
                }
                p0 += w;
            }
        }
        check_deadline(opts.deadline)?;
        rows.sort_by(ranked_cmp);
        // Permute dense blocks in place. No second K×n matrix is allocated.
        for target in 0..rows.len() {
            let source = rows[target].slot;
            if source != target {
                for j in 0..n {
                    a.swap(target * n + j, source * n + j);
                }
                for row in &mut rows {
                    if row.slot == source {
                        row.slot = target;
                    } else if row.slot == target {
                        row.slot = source;
                    }
                }
            }
        }
        check_deadline(opts.deadline)?;
        let original_rows = rows.iter().map(|row| row.original).collect();
        let state = Self {
            t,
            class,
            n_gens: n,
            n_in_gens: frame.n_in_gens,
            c_obj,
            d,
            r_obj,
            a,
            c_in: rows.iter().map(|row| row.centre).collect(),
            r: rows.iter().map(|row| row.remainder).collect(),
            lam: rows.iter().map(|row| row.lam).collect(),
            mu: rows.iter().map(|row| row.mu).collect(),
            lo: rows.iter().map(|row| row.lo).collect(),
            hi: rows.iter().map(|row| row.hi).collect(),
            enc: rows.iter().map(|row| row.enc).collect(),
            layer: rows.iter().map(|row| row.layer).collect(),
            site: rows.iter().map(|row| row.site).collect(),
            neuron: rows.iter().map(|row| row.neuron).collect(),
            stats: FrozenDualBuildStats {
                build_ms: t0.elapsed().as_secs_f64() * 1e3,
                passes,
                columns: 1 + total,
                estimate_bytes: estimate,
                budget_bytes: budget,
            },
        };
        check_deadline(opts.deadline)?;
        Ok((
            RankedFrozenDualState {
                state,
                original_rows,
                total_unstable: total,
            },
            keep_relax.then_some(relax),
        ))
    }

    /// Live bytes owned by the state, including vector capacities and row
    /// metadata. Allocator bookkeeping is covered by build admission slack.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        let floats = self
            .a
            .capacity()
            .saturating_add(self.d.capacity())
            .saturating_add(self.c_in.capacity())
            .saturating_add(self.r.capacity())
            .saturating_add(self.lam.capacity())
            .saturating_add(self.mu.capacity())
            .saturating_add(self.lo.capacity())
            .saturating_add(self.hi.capacity());
        let indices = self
            .enc
            .capacity()
            .saturating_add(self.layer.capacity())
            .saturating_add(self.neuron.capacity());
        (floats as u64)
            .saturating_mul(size_of::<f64>() as u64)
            .saturating_add((indices as u64).saturating_mul(size_of::<usize>() as u64))
            .saturating_add(
                (self.site.capacity() as u64).saturating_mul(size_of::<ReluSite>() as u64),
            )
            .saturating_add(size_of::<Self>() as u64)
    }
}

// ---------------------------------------------------------------------------
// The shared trunk (`#fd-shared-state`, ENGINE_PROGRAM M3)
// ---------------------------------------------------------------------------

/// The layers whose slope NO unstable row reads (the class TAIL).
///
/// A row pass seeded at layer `m`'s pre-activation (tensor `m.input`) visits
/// only ops `k < m.input`, so layer `li`'s ReLU (op `li.op`) can be read by
/// `m`'s rows only when `li.op < m.input`. A layer is a tail layer when that
/// holds for NO layer that has an unstable neuron (only those run row
/// passes). Conservative by construction: a layer that merely precedes an
/// unstable layer stays shared even if no path reaches it; misclassifying
/// the other way is what [`FrozenDualTrunk::state`]'s slope check refuses.
#[must_use]
pub(crate) fn tail_layers(frame: &FrozenDualFrame) -> Vec<bool> {
    frame
        .layers
        .iter()
        .map(|layer| {
            frame
                .layers
                .iter()
                .all(|m| m.unst.is_empty() || layer.op >= m.input)
        })
        .collect()
}

/// Layout digest of a frame: everything that fixes what a generator column
/// MEANS (the input symbols' `in_cols`, `mid` and `rad`; each layer's op,
/// input tensor, `col0` and unstable set) plus the shape counts. A trunk's
/// rows and a tail's margin row are ONE state only when their columns name
/// the same symbols, so [`FrozenDualTrunk::state`] refuses a frame or a tail
/// whose layout is not the trunk's. `O(n_in + S + layers)`: cheap enough to
/// take on every tail. (Boxes and magnitudes are deliberately not in it:
/// each object's own charges and `mu` are certified on the frame it was
/// built on, so two valid frames of one layout still assemble soundly.)
fn layout_digest(frame: &FrozenDualFrame) -> u64 {
    let mut h = 0x1357_9BDF_2468_ACE0u64;
    for v in [
        frame.n_in,
        frame.n_in_gens,
        frame.n_unstable,
        frame.n_gens,
        frame.t,
        frame.n_out,
        frame.out_tensor,
        frame.layers.len(),
    ] {
        h = digest_mix(h, v as u64);
    }
    for &c in &frame.in_cols {
        h = digest_mix(h, c as u64);
    }
    h = digest_mix(h, digest_f64(&frame.mid));
    h = digest_mix(h, digest_f64(&frame.rad));
    for layer in &frame.layers {
        for v in [
            layer.op,
            layer.input,
            layer.col0,
            layer.l.len(),
            layer.unst.len(),
        ] {
            h = digest_mix(h, v as u64);
        }
        for &j in &layer.unst {
            h = digest_mix(h, j as u64);
        }
    }
    h
}

/// One class's margin row under a given relaxation: its own (per-class,
/// [`class_margin`]) or a trunk's with the tail re-chosen
/// ([`TrunkRelax::tail`]).
#[derive(Debug, Clone)]
pub(crate) struct ClassTail {
    /// Adversarial class of the margin `y_t - y_class`.
    pub(crate) class: usize,
    /// Margin centre.
    pub(crate) c_obj: f64,
    /// Margin generator coefficients (`n`).
    pub(crate) d: Vec<f64>,
    /// Margin remainder.
    pub(crate) r_obj: f64,
    /// Every layer's slope and `mu` exactly as the margin pass used them.
    relax: ClassRelax,
    /// [`layout_digest`] of the frame the margin pass ran on.
    layout: u64,
    /// Wall seconds of the pass.
    pub(crate) secs: f64,
}

impl ClassTail {
    /// Heap storage retained while another class is materialized or searched.
    #[must_use]
    pub(crate) fn heap_bytes(&self) -> u64 {
        vector_bytes(&self.d).saturating_add(self.relax.heap_bytes())
    }

    /// Estimated owned bytes of a newly constructed tail.
    #[must_use]
    pub(crate) fn estimated_bytes(frame: &FrozenDualFrame) -> u64 {
        (frame.n_gens as u64)
            .saturating_mul(size_of::<f64>() as u64)
            .saturating_add(estimated_relax_heap_bytes(frame))
            .saturating_add(size_of::<Self>() as u64)
    }

    /// The certified root of this margin row, by the same arithmetic as
    /// [`FrozenDualState::root_bound`]: the root of the state it assembles
    /// into, bit for bit.
    #[must_use]
    pub(crate) fn root_bound(&self) -> f64 {
        root_bound_of(self.c_obj, &self.d, self.r_obj, self.d.len())
    }

    /// TEST-ONLY fault injection: overwrite one slope the margin pass used.
    #[cfg(test)]
    pub(crate) fn perturb_slope_for_test(&mut self, li: usize, p: usize, lam: f64) {
        self.relax.lam[li][p] = lam;
    }
}

/// The class's OWN direction-adaptive margin row: today's per-class margin
/// pass (chord start, every layer chosen), so its root is the per-class
/// state's root bit for bit, and its slopes are a trunk candidate
/// ([`TrunkRelax::anchor`]).
pub(crate) fn class_margin(
    net: &TwinNet,
    frame: &FrozenDualFrame,
    class: usize,
    deadline: Option<Instant>,
) -> Result<ClassTail, ZonoRefusal> {
    let t0 = Instant::now();
    check_class(net, frame, class)?;
    check_frame(net, frame)?;
    check_deadline(deadline)?;
    let mut relax = ClassRelax::chord(frame);
    let (c_obj, d, r_obj) = margin_row(net, frame, &mut relax, class, Choose::Every, deadline)?;
    Ok(ClassTail {
        class,
        c_obj,
        d,
        r_obj,
        relax,
        layout: layout_digest(frame),
        secs: t0.elapsed().as_secs_f64(),
    })
}

/// The slopes a shared trunk fixes for every class, and which layers each
/// class still re-chooses ([`tail_layers`]). UNTRUSTED in the sense that
/// matters: any slope in `[0, 1]` gives a sound state (the builder certifies
/// `mu` from the certified box), so the choice moves only tightness.
#[derive(Debug, Clone)]
pub(crate) struct TrunkRelax {
    relax: ClassRelax,
    tail: Vec<bool>,
}

impl TrunkRelax {
    fn heap_bytes(&self) -> u64 {
        self.relax
            .heap_bytes()
            .saturating_add(vector_bytes(&self.tail))
    }

    /// Estimated owned bytes of a newly constructed slope candidate.
    #[must_use]
    pub(crate) fn estimated_bytes(frame: &FrozenDualFrame) -> u64 {
        estimated_relax_heap_bytes(frame)
            .saturating_add(frame.layers.len() as u64)
            .saturating_add(size_of::<Self>() as u64)
    }

    /// The JOINT margin's direction-adaptive slopes: one pass seeded with
    /// `sum_j (y_t - y_j)` over `classes` chooses every slope from the sign of
    /// the SUMMED coefficient (the CROWN choice for the sum of the open
    /// margins).
    pub(crate) fn joint(
        net: &TwinNet,
        frame: &FrozenDualFrame,
        classes: &[usize],
        deadline: Option<Instant>,
    ) -> Result<Self, ZonoRefusal> {
        check_frame(net, frame)?;
        if classes.is_empty() {
            return Err(ZonoRefusal::Internal(
                "frozen-dual trunk: no class to share slopes across".to_string(),
            ));
        }
        let mut seed = Array2::<f64>::zeros((frame.n_out, 1));
        for &class in classes {
            check_class(net, frame, class)?;
            seed[[frame.t, 0]] += 1.0;
            seed[[class, 0]] -= 1.0;
        }
        check_deadline(deadline)?;
        let mut relax = ClassRelax::chord(frame);
        reverse_pass(
            net,
            frame,
            &mut relax,
            Choose::Every,
            frame.out_tensor,
            seed,
            deadline,
        )?;
        Ok(Self {
            relax,
            tail: tail_layers(frame),
        })
    }

    /// One class's OWN slopes (from [`class_margin`]): that class's shared
    /// state is then its per-class state, bit for bit.
    #[must_use]
    pub(crate) fn anchor(frame: &FrozenDualFrame, own: &ClassTail) -> Self {
        Self {
            relax: own.relax.clone(),
            tail: tail_layers(frame),
        }
    }

    /// TEST-ONLY fault injection: arbitrary slopes `lam(layer, position)`,
    /// NOT validated here, with `mu` certified from the frame's box (or the
    /// raw `mu` given). The builder must refuse the invalid ones and stay
    /// sound on every valid one.
    #[cfg(test)]
    pub(crate) fn from_slopes(
        frame: &FrozenDualFrame,
        lam: impl Fn(usize, usize) -> f64,
        mu: Option<f64>,
    ) -> Self {
        let mut relax = ClassRelax::chord(frame);
        for (li, layer) in frame.layers.iter().enumerate() {
            for (p, &j) in layer.unst.iter().enumerate() {
                let l = lam(li, p);
                relax.lam[li][p] = l;
                relax.mu[li][p] = mu.unwrap_or_else(|| certified_mu(l, layer.l[j], layer.u[j]));
            }
        }
        Self {
            relax,
            tail: tail_layers(frame),
        }
    }

    /// TEST-ONLY fault injection: replace the tail mask (a mask that marks a
    /// layer some row reads must make the trunk build refuse).
    #[cfg(test)]
    #[must_use]
    pub(crate) fn with_tail_mask_for_test(mut self, tail: Vec<bool>) -> Self {
        self.tail = tail;
        self
    }

    /// TEST-ONLY: the slope fixed at `(layer, position)`.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn slope(&self, li: usize, p: usize) -> Option<f64> {
        self.relax.lam.get(li)?.get(p).copied()
    }

    /// `class`'s TAIL under this trunk: one margin pass that READS every
    /// trunk slope and re-chooses only the tail layers.
    pub(crate) fn tail(
        &self,
        net: &TwinNet,
        frame: &FrozenDualFrame,
        class: usize,
        deadline: Option<Instant>,
    ) -> Result<ClassTail, ZonoRefusal> {
        let t0 = Instant::now();
        check_class(net, frame, class)?;
        check_frame(net, frame)?;
        self.check_shape(frame)?;
        check_deadline(deadline)?;
        let mut relax = self.relax.clone();
        let (c_obj, d, r_obj) = margin_row(
            net,
            frame,
            &mut relax,
            class,
            Choose::Layers(&self.tail),
            deadline,
        )?;
        Ok(ClassTail {
            class,
            c_obj,
            d,
            r_obj,
            relax,
            layout: layout_digest(frame),
            secs: t0.elapsed().as_secs_f64(),
        })
    }

    fn check_shape(&self, frame: &FrozenDualFrame) -> Result<(), ZonoRefusal> {
        if self.tail.len() == frame.layers.len() && self.relax.matches(frame) {
            Ok(())
        } else {
            Err(ZonoRefusal::Internal(
                "frozen-dual trunk: relaxation does not match the frame".to_string(),
            ))
        }
    }
}

/// Equal length and equal in every bit.
fn same_bits(x: &[f64], y: &[f64]) -> bool {
    x.len() == y.len() && x.iter().zip(y).all(|(a, b)| a.to_bits() == b.to_bits())
}

/// Words per digest chunk: FIXED, so the digest does not depend on the
/// thread count.
const DIGEST_CHUNK: usize = 1 << 16;

fn digest_mix(h: u64, w: u64) -> u64 {
    let mut x = (h ^ w).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 31;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^ (x >> 29)
}

fn digest_f64(v: &[f64]) -> u64 {
    v.par_chunks(DIGEST_CHUNK)
        .map(|c| {
            c.iter()
                .fold(0x243F_6A88_85A3_08D3u64, |h, x| digest_mix(h, x.to_bits()))
        })
        .collect::<Vec<u64>>()
        .into_iter()
        .fold(v.len() as u64, digest_mix)
}

/// Content digest of a trunk: every row coefficient, centre and remainder,
/// and every shared slope and `mu`. A corruption GUARD for the one certified
/// object that outlives a class (not a MAC).
fn trunk_digest(rows: &Rows, trunk: &TrunkRelax) -> u64 {
    let mut h = digest_f64(&rows.a);
    h = digest_mix(h, digest_f64(&rows.c_in));
    h = digest_mix(h, digest_f64(&rows.r));
    for (lam, mu) in trunk.relax.lam.iter().zip(&trunk.relax.mu) {
        h = digest_mix(h, digest_f64(lam));
        h = digest_mix(h, digest_f64(mu));
    }
    h
}

/// Every unstable row of every open class's state, built ONCE under a
/// [`TrunkRelax`] (`#fd-shared-state`, ENGINE_PROGRAM M3).
#[derive(Debug)]
pub(crate) struct FrozenDualTrunk {
    t: usize,
    n_gens: usize,
    trunk: TrunkRelax,
    rows: Rows,
    digest: u64,
    /// [`layout_digest`] of the frame the rows were built on.
    layout: u64,
    /// Build telemetry of the row passes (`passes`, `columns` = rows).
    pub(crate) stats: FrozenDualBuildStats,
}

impl FrozenDualTrunk {
    /// Build every unstable row under `trunk`'s slopes. Fail-closed exactly
    /// like [`FrozenDualState::build`]: the same admission (the same
    /// estimate), deadline and refusals, and a slope outside `[0, 1]` or a
    /// non-positive `mu` ANYWHERE refuses the whole trunk.
    pub(crate) fn build(
        net: &TwinNet,
        frame: &FrozenDualFrame,
        trunk: TrunkRelax,
        opts: &FrozenDualOptions,
    ) -> Result<Self, ZonoRefusal> {
        let t0 = Instant::now();
        check_frame(net, frame)?;
        trunk.check_shape(frame)?;
        let chunk = opts.chunk_cols.clamp(1, MAX_DIM);
        let estimated_relax =
            TrunkRelax::estimated_bytes(frame).saturating_sub(size_of::<TrunkRelax>() as u64);
        let estimate = estimate_trunk_state_bytes(frame, chunk)
            .saturating_add(trunk.heap_bytes().saturating_sub(estimated_relax));
        let budget = admitted_bytes(opts.max_state_bytes);
        if estimate > budget {
            return Err(ZonoRefusal::Memory {
                estimate_bytes: estimate,
                budget_bytes: budget,
            });
        }
        check_deadline(opts.deadline)?;
        let mut trunk = trunk;
        // The rows may read no tail slope: enforced per pass, fail-closed,
        // rather than trusted to `tail_layers`'s index argument.
        let forbid = trunk.tail.clone();
        let rows = build_rows(net, frame, &mut trunk.relax, chunk, opts.deadline, &forbid)?;
        drop(forbid);
        let (lam, mu) = row_relax(frame, &trunk.relax);
        check_relaxation(&lam, &mu)?;
        drop((lam, mu));
        let digest = trunk_digest(&rows, &trunk);
        let passes = rows.passes;
        Ok(Self {
            t: frame.t,
            n_gens: frame.n_gens,
            trunk,
            rows,
            digest,
            layout: layout_digest(frame),
            stats: FrozenDualBuildStats {
                build_ms: t0.elapsed().as_secs_f64() * 1e3,
                passes,
                columns: frame.n_unstable,
                estimate_bytes: estimate,
                budget_bytes: budget,
            },
        })
    }

    /// TEST-ONLY: the shared slopes the rows were built with.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn relax(&self) -> &TrunkRelax {
        &self.trunk
    }

    /// Unstable rows `S`.
    #[must_use]
    pub(crate) fn num_unstable(&self) -> usize {
        self.rows.c_in.len()
    }

    /// Owned live bytes, including row capacities, metadata, nested slopes
    /// and the tail mask. A class copy has different storage: its slopes are
    /// flattened and the source vectors' spare capacity is not copied.
    #[must_use]
    pub(crate) fn bytes(&self) -> u64 {
        (size_of::<Self>() as u64)
            .saturating_add(self.rows.payload_bytes(true))
            .saturating_add(self.trunk.heap_bytes())
    }

    /// Additional bytes at materialization's peak: copied rows, objective,
    /// and both flattened slope vectors. Flattening precedes the row copy,
    /// and those same vectors become the state's `lam` and `mu` fields.
    /// Digest scratch is released before this phase and is bounded by the
    /// copied coefficient payload, so it does not increase this peak.
    #[must_use]
    pub(crate) fn state_copy_bytes(&self, tail: &ClassTail) -> u64 {
        self.rows
            .payload_bytes(false)
            .saturating_add((tail.d.len() as u64).saturating_mul(8))
            .saturating_add((self.num_unstable() as u64).saturating_mul(2 * 8))
            .saturating_add(size_of::<FrozenDualState>() as u64)
    }

    /// TEST-ONLY: enlarge spare allocations without changing any value.
    #[cfg(test)]
    pub(crate) fn reserve_storage_for_test(&mut self) {
        fn grow<T>(v: &mut Vec<T>) {
            v.reserve_exact(v.capacity().saturating_sub(v.len()).saturating_add(1));
        }
        grow(&mut self.rows.a);
        grow(&mut self.rows.site);
        grow(&mut self.trunk.relax.lam);
        if let Some(mu) = self.trunk.relax.mu.first_mut() {
            grow(mu);
        }
        grow(&mut self.trunk.tail);
    }

    /// Class `class`'s state over this trunk: the trunk's rows (COPIED) with
    /// `tail`'s margin row and slopes.
    ///
    /// Refuses, and never assembles an inconsistent state, when `tail` is not
    /// `class`'s (a caller indexes tails by position and then closes the class
    /// it ASKED for), when `frame` or the frame `tail` ran on has another
    /// layout than the trunk's (a column would name another symbol), when the
    /// tail's slope or `mu` differs in ANY bit from the trunk's on a layer a
    /// row may read (the rows' relaxation symbols would not be the margin
    /// row's), when the trunk's content changed since it was built, or on any
    /// shape mismatch.
    pub(crate) fn state(
        &self,
        frame: &FrozenDualFrame,
        class: usize,
        tail: &ClassTail,
    ) -> Result<FrozenDualState, ZonoRefusal> {
        let t0 = Instant::now();
        if frame.t != self.t
            || frame.n_gens != self.n_gens
            || frame.n_unstable != self.rows.c_in.len()
            || tail.d.len() != self.n_gens
            || !tail.relax.matches(frame)
        {
            return Err(ZonoRefusal::Internal(
                "frozen-dual trunk: the frame or the tail does not match the trunk".to_string(),
            ));
        }
        if tail.class >= frame.n_out || tail.class == frame.t {
            return Err(ZonoRefusal::Shape(format!(
                "frozen-dual trunk: class {} vs true class {}",
                tail.class, frame.t
            )));
        }
        if tail.class != class {
            return Err(ZonoRefusal::Internal(format!(
                "frozen-dual trunk: a tail of class {} offered as class {class}",
                tail.class
            )));
        }
        self.trunk.check_shape(frame)?;
        if layout_digest(frame) != self.layout || tail.layout != self.layout {
            return Err(ZonoRefusal::Internal(
                "frozen-dual trunk: the frame or the tail has another layout than the trunk"
                    .to_string(),
            ));
        }
        for (li, &is_tail) in self.trunk.tail.iter().enumerate() {
            if is_tail {
                continue;
            }
            if !same_bits(&self.trunk.relax.lam[li], &tail.relax.lam[li])
                || !same_bits(&self.trunk.relax.mu[li], &tail.relax.mu[li])
            {
                return Err(ZonoRefusal::Internal(format!(
                    "frozen-dual trunk: class {} changes the shared slope of layer {li}",
                    tail.class
                )));
            }
        }
        if trunk_digest(&self.rows, &self.trunk) != self.digest {
            return Err(ZonoRefusal::Internal(
                "frozen-dual trunk: the shared rows changed after they were built".to_string(),
            ));
        }
        let (lam, mu) = row_relax(frame, &tail.relax);
        check_relaxation(&lam, &mu)?;
        let rows = self.rows.clone();
        let build_ms = (tail.secs + t0.elapsed().as_secs_f64()) * 1e3;
        Ok(FrozenDualState::from_parts(
            frame,
            tail.class,
            (tail.c_obj, tail.d.clone(), tail.r_obj),
            rows,
            (lam, mu),
            FrozenDualBuildStats {
                build_ms,
                passes: 1,
                columns: 1,
                estimate_bytes: self.stats.estimate_bytes,
                budget_bytes: self.stats.budget_bytes,
            },
        ))
    }

    /// TEST-ONLY fault injection: flip the low bit of row coefficient `idx`
    /// (the digest guard must then refuse every later class).
    #[cfg(test)]
    pub(crate) fn corrupt_for_test(&mut self, idx: usize) {
        if let Some(v) = self.rows.a.get_mut(idx) {
            *v = f64::from_bits(v.to_bits() ^ 1);
        }
    }

    /// TEST-ONLY: the shared rows (row-major `S x n`).
    #[cfg(test)]
    pub(crate) fn rows_for_test(&self) -> &[f64] {
        &self.rows.a
    }
}

/// The frozen pre-M3 builder, the reference of the refactor's bit-identity
/// pin. Inline (like M1's `eval_node_pre_m1`): a `build/` directory is
/// gitignored in this repository, so a child FILE here would be dropped by a
/// plain `git add` and the test build would lose the module.
///
/// TEST-ONLY: a FROZEN verbatim copy of the pre-M3 `reverse_pass` and
/// `FrozenDualState::build` (engine program base `b7110db19`), the reference
/// the refactored builder is pinned against bit for bit
/// (`fd_trunk_build_refactor_is_bit_identical`). ENGINE_PROGRAM M3 split the
/// build into a margin row plus the row passes so a SHARED trunk can reuse the
/// latter; the per-class default must not move a single bit. Only the names
/// changed here (`reverse_pass` -> `reverse_pass_pre_m3`, the method `build` ->
/// the free function `build_pre_m3`, `Self` -> `FrozenDualState`). Never edit
/// the bodies: a drift here silently weakens the pin.
#[cfg(test)]
pub(crate) mod pre_m3 {
    use std::time::Instant;

    use ndarray::{Array2, ArrayView2};

    use crate::margin_row::net::{conv_apply_backward, TwinNet, TwinOp};
    use crate::zono_core::rounding::{gamma, next_up, UNIT};
    use crate::zono_core::ZonoRefusal;

    use super::super::frame::{FrozenDualFrame, ReluSite, MAX_DIM};
    use super::{
        admitted_bytes, assemble, certified_mu, charge_rows, check_deadline, estimate_state_bytes,
        fold_bias, merge, ClassRelax, FrozenDualBuildStats, FrozenDualOptions, FrozenDualState,
        PassOut,
    };

    pub(super) fn reverse_pass_pre_m3(
        net: &TwinNet,
        frame: &FrozenDualFrame,
        relax: &mut ClassRelax,
        choose: bool,
        start: usize,
        seed: Array2<f64>,
        deadline: Option<Instant>,
    ) -> Result<PassOut, ZonoRefusal> {
        let r = seed.ncols();
        if choose && r != 1 {
            return Err(ZonoRefusal::Internal(
                "frozen-dual: the slope-choosing pass takes one column".to_string(),
            ));
        }
        if start > net.ops.len() || seed.nrows() != net.tsize[start] {
            return Err(ZonoRefusal::Shape(format!(
                "frozen-dual seed: {} rows at tensor {start}",
                seed.nrows()
            )));
        }
        let mut coef: Vec<Option<Array2<f64>>> = vec![None; start + 1];
        coef[start] = Some(seed);
        let mut enew: Vec<Option<Array2<f64>>> = vec![None; frame.layers.len()];
        let mut konst = vec![0.0f64; r];
        let mut delta = vec![0.0f64; r];
        for k in (0..start).rev() {
            let Some(cur) = coef[k + 1].take() else {
                continue;
            };
            check_deadline(deadline)?;
            match &net.ops[k] {
                TwinOp::Conv(cv) => {
                    let n_t = net.tsize[k + 1];
                    let p = (cv.oshape.1 * cv.oshape.2).max(1);
                    if cv.bias.len() * p != n_t || cv.bias_err.len() * p != n_t {
                        return Err(ZonoRefusal::Shape(format!("frozen-dual conv op {k} bias")));
                    }
                    fold_bias(
                        &mut konst,
                        &mut delta,
                        &cur,
                        &|j: usize| cv.bias[j / p],
                        &|j: usize| cv.bias_err[j / p],
                    );
                    let rel = cv.weight_rel_err;
                    let g = next_up(gamma(cv.k_bwd + 2) + rel + gamma(cv.k_bwd + 2) * rel);
                    charge_rows(&mut delta, &cur, &frame.wmag[k], g)?;
                    let mut lnew = Array2::<f64>::zeros((net.tsize[cv.input], r));
                    conv_apply_backward(cv, &cur, &mut lnew, false);
                    merge(&mut coef, cv.input, lnew, &frame.mag[cv.input], &mut delta)?;
                }
                TwinOp::ChannelAffine {
                    input,
                    scale,
                    shift,
                    scale_rel_err,
                    shift_err,
                } => {
                    if shift.len() != cur.nrows()
                        || shift_err.len() != cur.nrows()
                        || scale.len() != cur.nrows()
                    {
                        return Err(ZonoRefusal::Shape(format!("frozen-dual affine op {k}")));
                    }
                    fold_bias(
                        &mut konst,
                        &mut delta,
                        &cur,
                        &|j: usize| shift[j],
                        &|j: usize| shift_err[j],
                    );
                    // Model error of the scale: rel * sum_j |L_j| |s_j| mag_j.
                    charge_rows(&mut delta, &cur, &frame.wmag[k], *scale_rel_err)?;
                    let mut lnew = cur;
                    for (mut row, &sj) in lnew.outer_iter_mut().zip(scale.iter()) {
                        row.mapv_inplace(|v| v * sj);
                    }
                    // One rounding per product, charged on the stored result.
                    charge_rows(&mut delta, &lnew, &frame.mag[*input], 2.0 * UNIT)?;
                    merge(&mut coef, *input, lnew, &frame.mag[*input], &mut delta)?;
                }
                TwinOp::Relu { input, .. } => {
                    let li = frame.layer_of_op[k];
                    let layer = frame.layers.get(li).ok_or_else(|| {
                        ZonoRefusal::Internal(format!(
                            "frozen-dual: relu op {k} has no frame layer"
                        ))
                    })?;
                    let n = layer.l.len();
                    let np = layer.unst.len();
                    if cur.nrows() != n || frame.mag[*input].len() != n {
                        return Err(ZonoRefusal::Shape(format!("frozen-dual relu op {k}")));
                    }
                    let mut lz = Array2::<f64>::zeros((n, r));
                    let mut cm = Array2::<f64>::zeros((np, r));
                    let magz = &frame.mag[*input];
                    for j in 0..n {
                        if layer.l[j] >= 0.0 {
                            // Active on the whole box: relu(z) = z exactly.
                            lz.row_mut(j).assign(&cur.row(j));
                            continue;
                        }
                        if layer.u[j] <= 0.0 {
                            // Dead on the whole box: relu(z) = 0 exactly.
                            continue;
                        }
                        let p = layer.pos[j];
                        if choose {
                            let lam = if cur[[j, 0]] > 0.0 {
                                layer.alpha[j]
                            } else {
                                layer.s[j]
                            };
                            relax.lam[li][p] = lam;
                            relax.mu[li][p] = certified_mu(lam, layer.l[j], layer.u[j]);
                        }
                        let lam = relax.lam[li][p];
                        let mu = relax.mu[li][p];
                        for c in 0..r {
                            let v = cur[[j, c]];
                            let zc = lam * v;
                            let ec = v * mu;
                            lz[[j, c]] = zc;
                            cm[[p, c]] = ec;
                            delta[c] += 2.0 * UNIT * zc.abs() * magz[j] + 3.0 * UNIT * ec.abs();
                        }
                    }
                    let g = gamma(np + 2);
                    for c in 0..r {
                        let mut sum = 0.0f64;
                        let mut abs = 0.0f64;
                        for p in 0..np {
                            let v = cm[[p, c]];
                            sum += v;
                            abs += v.abs();
                        }
                        delta[c] += g * (konst[c].abs() + abs);
                        konst[c] += sum;
                    }
                    enew[li] = Some(cm);
                    merge(&mut coef, *input, lz, magz, &mut delta)?;
                }
                TwinOp::Add { lhs, rhs } => {
                    let copy = cur.clone();
                    merge(&mut coef, *lhs, cur, &frame.mag[*lhs], &mut delta)?;
                    merge(&mut coef, *rhs, copy, &frame.mag[*rhs], &mut delta)?;
                }
                TwinOp::Flatten { input } => {
                    merge(&mut coef, *input, cur, &frame.mag[*input], &mut delta)?;
                }
                TwinOp::Gemm {
                    input,
                    weight,
                    bias,
                    bias_err,
                    weight_rel_err,
                    shape,
                } => {
                    let (no, ni) = *shape;
                    if cur.nrows() != no
                        || weight.len() != no * ni
                        || bias.len() != no
                        || bias_err.len() != no
                    {
                        return Err(ZonoRefusal::Shape(format!("frozen-dual gemm op {k}")));
                    }
                    fold_bias(
                        &mut konst,
                        &mut delta,
                        &cur,
                        &|o: usize| bias[o],
                        &|o: usize| bias_err[o],
                    );
                    let rel = *weight_rel_err;
                    let g = next_up(gamma(no + 2) + rel + gamma(no + 2) * rel);
                    charge_rows(&mut delta, &cur, &frame.wmag[k], g)?;
                    let w = ArrayView2::from_shape((no, ni), weight.as_slice())
                        .map_err(|e| ZonoRefusal::Shape(format!("frozen-dual gemm view: {e}")))?;
                    let prod = w.t().dot(&cur);
                    let lnew = if prod.is_standard_layout() {
                        prod
                    } else {
                        prod.as_standard_layout().into_owned()
                    };
                    merge(&mut coef, *input, lnew, &frame.mag[*input], &mut delta)?;
                }
            }
        }
        let x = match coef[0].take() {
            Some(x) => x,
            None => Array2::<f64>::zeros((net.n_in, r)),
        };
        Ok(PassOut {
            x,
            enew,
            konst,
            delta,
        })
    }

    /// The pre-M3 `FrozenDualState::build`, verbatim.
    pub(crate) fn build_pre_m3(
        net: &TwinNet,
        frame: &FrozenDualFrame,
        class: usize,
        opts: &FrozenDualOptions,
    ) -> Result<FrozenDualState, ZonoRefusal> {
        let t0 = Instant::now();
        let t = frame.t;
        if class >= frame.n_out || class == t || frame.n_out != net.n_out {
            return Err(ZonoRefusal::Shape(format!(
                "frozen-dual: class {class} vs true class {t} of {}",
                frame.n_out
            )));
        }
        if frame.out_tensor != net.ops.len() || frame.mag.len() != net.ops.len() + 1 {
            return Err(ZonoRefusal::Internal(
                "frozen-dual: frame was built for a different net".to_string(),
            ));
        }
        let chunk = opts.chunk_cols.clamp(1, MAX_DIM);
        let estimate = estimate_state_bytes(frame, chunk);
        let budget = admitted_bytes(opts.max_state_bytes);
        if estimate > budget {
            return Err(ZonoRefusal::Memory {
                estimate_bytes: estimate,
                budget_bytes: budget,
            });
        }
        check_deadline(opts.deadline)?;
        let n = frame.n_gens;
        let s_total = frame.n_unstable;
        let mut relax = ClassRelax::chord(frame);
        let mut passes = 0usize;

        // ---- the margin row (chooses lambda) -----------------------------
        let mut seed = Array2::<f64>::zeros((frame.n_out, 1));
        seed[[t, 0]] = 1.0;
        seed[[class, 0]] = -1.0;
        let out = reverse_pass_pre_m3(
            net,
            frame,
            &mut relax,
            true,
            frame.out_tensor,
            seed,
            opts.deadline,
        )?;
        passes += 1;
        let mut d = vec![0.0f64; n];
        let (c_obj, r_obj) = assemble(frame, &out, 0, &mut d)?;
        drop(out);
        // A final reverse operation or assembly can use the remaining wall.
        // Refuse before allocating the dense row matrix in that case.
        check_deadline(opts.deadline)?;

        // ---- one identity-seeded pass per unstable layer (chunked) -------
        let mut a = vec![0.0f64; s_total * n];
        let mut c_in = vec![0.0f64; s_total];
        let mut r = vec![0.0f64; s_total];
        let mut lam = vec![0.0f64; s_total];
        let mut mu = vec![0.0f64; s_total];
        let mut lo = vec![0.0f64; s_total];
        let mut hi = vec![0.0f64; s_total];
        let mut enc = vec![0usize; s_total];
        let mut layer_of = vec![0usize; s_total];
        let mut site = vec![ReluSite::Head; s_total];
        let mut neuron = vec![0usize; s_total];
        for (li, layer) in frame.layers.iter().enumerate() {
            let np = layer.unst.len();
            let base = layer.col0 - frame.n_in_gens;
            for (p, &j) in layer.unst.iter().enumerate() {
                let k = base + p;
                lam[k] = relax.lam[li][p];
                mu[k] = relax.mu[li][p];
                lo[k] = layer.l[j];
                hi[k] = layer.u[j];
                enc[k] = layer.col0 + p;
                layer_of[k] = li;
                site[k] = layer.site;
                neuron[k] = j;
            }
            let width = net.tsize[layer.input];
            let mut p0 = 0usize;
            while p0 < np {
                check_deadline(opts.deadline)?;
                let w = (np - p0).min(chunk);
                let mut seed = Array2::<f64>::zeros((width, w));
                for c in 0..w {
                    seed[[layer.unst[p0 + c], c]] = 1.0;
                }
                let out = reverse_pass_pre_m3(
                    net,
                    frame,
                    &mut relax,
                    false,
                    layer.input,
                    seed,
                    opts.deadline,
                )?;
                passes += 1;
                for c in 0..w {
                    let k = base + p0 + c;
                    let (ck, rk) = assemble(frame, &out, c, &mut a[k * n..(k + 1) * n])?;
                    // A row may not reference its own or any downstream
                    // relaxation symbol: the pass stops at the layer's input.
                    if a[k * n + enc[k]] != 0.0 {
                        return Err(ZonoRefusal::Internal(format!(
                            "frozen-dual: row {k} references its own symbol"
                        )));
                    }
                    c_in[k] = ck;
                    r[k] = rk;
                }
                p0 += w;
            }
        }
        if mu.iter().any(|m| !(m.is_finite() && *m > 0.0))
            || lam.iter().any(|l| !(0.0..=1.0).contains(l))
        {
            return Err(ZonoRefusal::NonFinite("frozen-dual relaxation"));
        }
        let build_ms = t0.elapsed().as_secs_f64() * 1e3;
        Ok(FrozenDualState {
            t,
            class,
            n_gens: n,
            n_in_gens: frame.n_in_gens,
            c_obj,
            d,
            r_obj,
            a,
            c_in,
            r,
            lam,
            mu,
            lo,
            hi,
            enc,
            layer: layer_of,
            site,
            neuron,
            stats: FrozenDualBuildStats {
                build_ms,
                passes,
                columns: 1 + s_total,
                estimate_bytes: estimate,
                budget_bytes: budget,
            },
        })
    }
}
