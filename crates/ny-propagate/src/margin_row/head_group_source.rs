// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Private ownership and construction provenance for the optional head pass.
//!
//! A digest is a mutation detector, never imported proof authority. Authority
//! comes from this module's private generation, minted before a successful
//! unsplit outward source construction in the same owned invocation. Network
//! parameters and the declared input box remain immutable throughout that
//! invocation; trusted root refinements may publish new, valid frozen gates.
//! A pack then freezes that exact gate generation for immediate replay. No
//! deserializer, public constructor, raw-row adapter, or certificate importer
//! can create either provenance value.

use std::cell::Cell;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use sha2::{Digest, Sha256};

use super::engine::{BackwardEngine, PassOut};
use super::errlane::ErrLane;
use super::net::{TwinNet, TwinOp};
use super::root::RootGates;
use super::rounding::RoundMode;

thread_local! {
    static ARMED: Cell<bool> = const { Cell::new(false) };
}

/// Entry-scoped permission to capture native source provenance. The dark lane
/// performs no hashing or FP pool qualification. This guard cannot move to
/// another thread because the switch is local to its calling thread.
pub(super) struct HeadGroupSourceGuard {
    previous: bool,
    _thread: PhantomData<Rc<()>>,
}

impl HeadGroupSourceGuard {
    pub(super) fn enter(enabled: bool) -> Self {
        Self {
            previous: ARMED.with(|armed| armed.replace(enabled)),
            _thread: PhantomData,
        }
    }
}

impl Drop for HeadGroupSourceGuard {
    fn drop(&mut self) {
        ARMED.with(|armed| armed.set(self.previous));
    }
}

struct SourceGeneration {
    net_addr: usize,
    n_in: usize,
    n_y: usize,
    source: [u8; 32],
    input_box: [u8; 32],
    // None records a failed completion too: a later, plausibly shaped frame
    // cannot retry this private build transaction and mint new authority.
    input_rows: OnceLock<Option<CompletedInputRows>>,
}

struct CompletedInputRows {
    #[cfg_attr(not(test), expect(dead_code, reason = "bounded input-row integration"))]
    gates: [u8; 32],
    #[cfg_attr(not(test), expect(dead_code, reason = "bounded input-row integration"))]
    deadline: Option<Instant>,
}

/// Private source-build generation. Only the normal builder attaches this
/// value to its completed root. Scratch, imported and split frames have none.
#[derive(Clone)]
pub(super) struct RootSource(Arc<SourceGeneration>);

impl RootSource {
    /// Called once by the original unsplit builder immediately after success.
    /// Hashing streams bounded chunks into fixed stack storage; it allocates
    /// no temporary vectors. This fixed owner/hash footprint is included in
    /// the input-product resident floor, independently of row payloads.
    pub(super) fn complete_input_rows(
        &self,
        root: &RootGates,
        original_deadline: Option<Instant>,
        _authority: super::root::CompletedOriginalRootAuthority,
    ) -> bool {
        if self.0.input_rows.get().is_some() {
            return false;
        }
        let completed = input_rows_gates_digest(root, original_deadline)
            .ok()
            .map(|gates| CompletedInputRows {
                gates,
                deadline: original_deadline,
            });
        let qualified = completed.is_some();
        self.0.input_rows.set(completed).is_ok() && qualified
    }

    pub(super) fn input_rows_resident_bytes(&self) -> usize {
        // Arc counters/alignment, fixed fingerprint state and final digests.
        // No source arrays are copied or charged a second time here.
        size_of::<SourceGeneration>() + size_of::<[usize; 8]>() + size_of::<Hash>() + 3 * 32
    }

    #[cfg_attr(not(test), expect(dead_code, reason = "bounded input-row integration"))]
    pub(super) fn input_rows_deadline(&self, requested: Instant) -> ny_core::Result<Instant> {
        let completed = self
            .0
            .input_rows
            .get()
            .and_then(Option::as_ref)
            .ok_or_else(|| {
                ny_core::NyError::InvalidSpec(
                    "input rows: original builder generation was not completed".into(),
                )
            })?;
        let deadline = completed
            .deadline
            .map_or(requested, |original| original.min(requested));
        super::net::check_deadline(deadline)?;
        Ok(deadline)
    }

    /// Reuses this private completed UNSPLIT builder generation for the first
    /// tile-local input-row producer. This does not widen native-head shape or
    /// source caps and does not mint a generation from numerical gate checks.
    /// A scalable independent original-root owner remains a separate frontier.
    #[cfg_attr(not(test), expect(dead_code, reason = "bounded input-row integration"))]
    pub(super) fn authenticate_input_rows(
        &self,
        net: &TwinNet,
        root: &RootGates,
        deadline: Instant,
    ) -> ny_core::Result<Instant> {
        let completed = self
            .0
            .input_rows
            .get()
            .and_then(Option::as_ref)
            .ok_or_else(|| {
                ny_core::NyError::InvalidSpec(
                    "input rows: original builder generation was not completed".into(),
                )
            })?;
        let deadline = self.input_rows_deadline(deadline)?;
        super::net::check_deadline(deadline)?;
        if !root.mode.outward()
            || !shape_admitted(net)
            || root.root_precision_policy.box_source != super::RootBoxSource::Tableau
            || matches!(
                root.root_precision_policy.err_lane,
                ErrLane::Envelope | ErrLane::PerChannel
            )
            || self.0.net_addr != std::ptr::from_ref(net) as usize
            || self.0.n_in != net.n_in
            || self.0.n_y != net.n_y
            || self.0.input_box != box_digest_with_deadline(&root.lo, &root.hi, deadline)?
            || self.0.source != source_digest_with_deadline(net, deadline)?
            || completed.gates != input_rows_gates_digest(root, Some(deadline))?
        {
            return Err(ny_core::NyError::InvalidSpec(
                "input rows: stale or unsupported original unsplit builder authority".into(),
            ));
        }
        super::net::check_deadline(deadline)?;
        Ok(deadline)
    }

    /// Must run BEFORE any source gate construction. Failure disables only
    /// the optional head producer; it never creates a replacement authority.
    pub(super) fn begin(
        net: &TwinNet,
        lo: &[f64],
        hi: &[f64],
        mode: RoundMode,
        splits: &[(usize, usize, i8)],
    ) -> Option<Self> {
        if !ARMED.with(Cell::get)
            || !mode.outward()
            || !splits.is_empty()
            || !shape_admitted(net)
            || lo.len() != net.n_in
            || hi.len() != net.n_in
            || lo
                .iter()
                .zip(hi)
                .any(|(&l, &u)| !l.is_finite() || !u.is_finite() || l > u)
            || !qualified_pool()
        {
            return None;
        }
        Some(Self(Arc::new(SourceGeneration {
            net_addr: std::ptr::from_ref(net) as usize,
            n_in: net.n_in,
            n_y: net.n_y,
            source: source_digest(net),
            input_box: box_digest(lo, hi),
            input_rows: OnceLock::new(),
        })))
    }

    fn matches(&self, eng: &BackwardEngine<'_>) -> bool {
        eng.root.mode.outward()
            && super::head_group::root_lines_qualified(eng.root)
            && shape_admitted(eng.net)
            && self.0.net_addr == std::ptr::from_ref(eng.net) as usize
            && self.0.input_box == box_digest(&eng.root.lo, &eng.root.hi)
            && self.0.source == source_digest(eng.net)
    }
}

/// Provenance of one identity-seeded, unconditional pair of source rows.
/// `bab::build_pack` captures this BEFORE producing those rows and attaches
/// it only for `(domain=None, splits=[])`; raw rows remain unbound.
#[derive(Clone)]
pub(super) struct YPackSource {
    root: RootSource,
    gates: [u8; 32],
    rows: Option<[u8; 32]>,
}

/// Separate provenance for a directly constructed scalar head box. It never
/// represents identity-seeded affine rows or grants YPack authority. Only the
/// private direct producer constructs the numerical enclosure that uses it.
#[derive(Clone)]
pub(super) struct RootFunctionalSource {
    root: RootSource,
    gates: [u8; 32],
    target: usize,
    adversaries: Vec<usize>,
}

impl RootFunctionalSource {
    pub(super) fn capture(
        eng: &BackwardEngine<'_>,
        target: usize,
        adversaries: &[usize],
    ) -> Option<Self> {
        let source = eng.root.head_group_source.as_ref()?;
        if adversaries.is_empty()
            || target >= eng.net.n_out
            || adversaries
                .iter()
                .enumerate()
                .any(|(i, &a)| a >= eng.net.n_out || a == target || adversaries[..i].contains(&a))
            || matches!(eng.err_lane(), ErrLane::Envelope | ErrLane::PerChannel)
            || !source.matches(eng)
            || !qualified_pool()
        {
            return None;
        }
        Some(Self {
            root: source.clone(),
            gates: gates_digest(eng.root),
            target,
            adversaries: adversaries.to_vec(),
        })
    }

    pub(super) fn matches(
        &self,
        eng: &BackwardEngine<'_>,
        target: usize,
        adversaries: &[usize],
    ) -> bool {
        eng.root.head_group_source.as_ref().is_some_and(|source| {
            self.target == target
                && self.adversaries == adversaries
                && Arc::ptr_eq(&self.root.0, &source.0)
                && !matches!(eng.err_lane(), ErrLane::Envelope | ErrLane::PerChannel)
                && self.root.matches(eng)
                && self.gates == gates_digest(eng.root)
        })
    }

    /// This producer already validated the source bytes at capture. Within
    /// its immutable engine borrow, internal construction/replay need only
    /// check the private generation, frozen gates and original row mapping.
    /// Reuse/publication still call `matches` to repeat the full source guard.
    pub(super) fn same_frame(
        &self,
        eng: &BackwardEngine<'_>,
        target: usize,
        adversaries: &[usize],
    ) -> bool {
        eng.root.head_group_source.as_ref().is_some_and(|source| {
            self.target == target
                && self.adversaries == adversaries
                && Arc::ptr_eq(&self.root.0, &source.0)
                && self.root.0.net_addr == std::ptr::from_ref(eng.net) as usize
                && eng.root.mode.outward()
                && !matches!(eng.err_lane(), ErrLane::Envelope | ErrLane::PerChannel)
                && self.gates == gates_digest(eng.root)
        })
    }
}

impl YPackSource {
    /// Compare two already sealed private snapshots without rehashing their
    /// shared immutable source. A distinct build generation never matches,
    /// even when every network, gate and row byte is otherwise identical.
    pub(super) fn same_snapshot(&self, other: &Self) -> bool {
        self.rows.is_some()
            && other.rows.is_some()
            && self.rows == other.rows
            && self.gates == other.gates
            && Arc::ptr_eq(&self.root.0, &other.root.0)
    }

    pub(super) fn capture(eng: &BackwardEngine<'_>) -> Option<Self> {
        let source = eng.root.head_group_source.as_ref()?;
        // Only the repaired per-column producer carries the absolute
        // underflow charges required by arbitrary signed whole functionals.
        // Audit publishes that same lane; its shadows provide no authority.
        if matches!(eng.err_lane(), ErrLane::Envelope | ErrLane::PerChannel)
            || !source.matches(eng)
            || !qualified_pool()
        {
            return None;
        }
        Some(Self {
            root: source.clone(),
            gates: gates_digest(eng.root),
            rows: None,
        })
    }

    /// Seal the exact owned rows after the native producer completes. There
    /// is no re-seal operation: subsequent mutation invalidates this pack.
    pub(super) fn seal_rows(&mut self, al: &PassOut, au: &PassOut, ly: &[f64], uy: &[f64]) -> bool {
        if self.rows.is_some()
            || !rows_valid(al, self.root.0.n_in, self.root.0.n_y)
            || !rows_valid(au, self.root.0.n_in, self.root.0.n_y)
            || ly.len() != self.root.0.n_y
            || uy.len() != self.root.0.n_y
            || ly
                .iter()
                .zip(uy)
                .any(|(&l, &u)| !l.is_finite() || !u.is_finite() || l > u)
        {
            return false;
        }
        self.rows = Some(rows_digest(al, au, ly, uy));
        true
    }

    pub(super) fn matches(
        &self,
        eng: &BackwardEngine<'_>,
        al: &PassOut,
        au: &PassOut,
        ly: &[f64],
        uy: &[f64],
    ) -> bool {
        let Some(source) = eng.root.head_group_source.as_ref() else {
            return false;
        };
        self.rows == Some(rows_digest(al, au, ly, uy))
            && !matches!(eng.err_lane(), ErrLane::Envelope | ErrLane::PerChannel)
            && Arc::ptr_eq(&self.root.0, &source.0)
            && self.root.matches(eng)
            && self.gates == gates_digest(eng.root)
    }
}

fn rows_valid(pass: &PassOut, n_in: usize, n_y: usize) -> bool {
    let Some(error) = &pass.e else {
        return false;
    };
    pass.a.dim() == (n_in, n_y)
        && error.dim() == (n_in, n_y)
        && pass.b.len() == n_y
        && pass.eb.len() == n_y
        && pass.a.iter().chain(&pass.b).all(|v| v.is_finite())
        && error
            .iter()
            .chain(&pass.eb)
            .all(|v| v.is_finite() && *v >= 0.0)
}

fn rows_digest(al: &PassOut, au: &PassOut, ly: &[f64], uy: &[f64]) -> [u8; 32] {
    let mut hash = Hash::new(b"ny/native-head-source-owned-rows/v1");
    for pass in [al, au] {
        hash.matrix(&pass.a);
        hash.flag(pass.e.is_some());
        if let Some(error) = &pass.e {
            hash.matrix(error);
        }
        hash.floats(&pass.b);
        hash.floats(&pass.eb);
    }
    hash.floats(ly);
    hash.floats(uy);
    hash.finish()
}

fn shape_admitted(net: &TwinNet) -> bool {
    (1..=4096).contains(&net.n_in) && (1..=512).contains(&net.n_y) && (2..=10).contains(&net.n_out)
}

/// The source kernels use Rayon. Caller-only qualification would leave the
/// actual worker arithmetic unchecked because FP control is thread-local.
fn qualified_pool() -> bool {
    ny_core::require_f64_interval_proof_environment().is_ok()
        && rayon::broadcast(|_| ny_core::require_f64_interval_proof_environment())
            .into_iter()
            .all(|result| result.is_ok())
}

struct Hash(Sha256);

impl Hash {
    fn new(domain: &[u8]) -> Self {
        let mut hash = Sha256::new();
        hash.update(domain);
        Self(hash)
    }

    fn word(&mut self, word: usize) {
        self.0.update(word.to_ne_bytes());
    }

    fn flag(&mut self, flag: bool) {
        self.word(usize::from(flag));
    }

    fn scalar(&mut self, value: f64) {
        self.0.update(value.to_bits().to_ne_bytes());
    }

    // Native bytes are deliberate: these are private same-process mutation
    // detectors, never portable serialized claims. Streaming whole slices
    // avoids millions of tiny hash updates for a wide authored head matrix.
    fn floats(&mut self, values: &[f64]) {
        self.word(values.len());
        self.0.update(bytemuck::cast_slice::<f64, u8>(values));
    }

    fn indices(&mut self, values: &[usize]) {
        self.word(values.len());
        for &value in values {
            self.word(value);
        }
    }

    fn matrix(&mut self, values: &ndarray::Array2<f64>) {
        self.indices(&[values.nrows(), values.ncols()]);
        if let Some(slice) = values.as_slice() {
            self.floats(slice);
        } else {
            self.word(values.len());
            for &value in values {
                self.scalar(value);
            }
        }
    }

    fn finish(self) -> [u8; 32] {
        self.0.finalize().into()
    }
}

fn box_digest(lo: &[f64], hi: &[f64]) -> [u8; 32] {
    let mut hash = Hash::new(b"ny/native-head-source-box/v1");
    hash.floats(lo);
    hash.floats(hi);
    hash.finish()
}

#[cfg_attr(not(test), expect(dead_code, reason = "bounded input-row integration"))]
fn box_digest_with_deadline(
    lo: &[f64],
    hi: &[f64],
    deadline: Instant,
) -> ny_core::Result<[u8; 32]> {
    let mut hash = Hash::new(b"ny/native-head-source-box/v1");
    hash_floats_with_deadline(&mut hash, lo, deadline)?;
    hash_floats_with_deadline(&mut hash, hi, deadline)?;
    Ok(hash.finish())
}

/// Numerical inputs actually read by the input-row producer. xabs is excluded
/// deliberately: the new pointwise charge reads original endpoints, and the
/// new patch scalar concretization computes its own outward magnitude bound.
/// Gate bounds/lines cannot be re-authenticated after a conditional mutation.
fn input_rows_gates_digest(
    root: &RootGates,
    deadline: Option<Instant>,
) -> ny_core::Result<[u8; 32]> {
    let check = || deadline.map_or(Ok(()), super::net::check_deadline);
    let floats = |hash: &mut Hash, values: &[f64]| -> ny_core::Result<()> {
        hash.word(values.len());
        for chunk in values.chunks(4096) {
            check()?;
            hash.0.update(bytemuck::cast_slice::<f64, u8>(chunk));
        }
        check()
    };
    check()?;
    let mut hash = Hash::new(b"ny/original-input-row-completed-gates/v1");
    hash.flag(root.mode.outward());
    floats(&mut hash, &root.mid)?;
    floats(&mut hash, &root.rad)?;
    hash.word(root.layers.len());
    for layer in &root.layers {
        check()?;
        hash.word(layer.op);
        hash.word(layer.n);
        for values in [
            &layer.l,
            &layer.u,
            &layer.alpha,
            &layer.s,
            &layer.c,
            &layer.ms,
        ] {
            floats(&mut hash, values)?;
        }
        hash.word(layer.unst.len());
        for chunk in layer.unst.chunks(4096) {
            check()?;
            for &neuron in chunk {
                hash.word(neuron);
            }
        }
    }
    let policy = root.root_precision_policy;
    hash.flag(policy.root_f32);
    hash.flag(policy.tableau_f32);
    hash.flag(policy.ibp_seed);
    hash.0.update(policy.err_lane.token().as_bytes());
    hash.0.update(policy.box_source.token().as_bytes());
    check()?;
    Ok(hash.finish())
}

/// Includes the entire compiled trunk, its geometry/maps, both directions'
/// weights, all fold uncertainty, and both authored head maps. Head-only
/// identity would miss a changed prefix while retaining plausible head rows.
fn source_digest(net: &TwinNet) -> [u8; 32] {
    let mut hash = Hash::new(b"ny/native-head-source-network/v1");
    hash.word(net.n_in);
    hash.word(net.n_y);
    hash.word(net.n_out);
    hash.word(net.i_gemm1);
    hash.indices(&net.tsize);
    hash.indices(&net.trunk_relus);
    hash.word(net.ops.len());
    for op in &net.ops {
        match op {
            TwinOp::Conv(conv) => {
                hash.word(0);
                hash.word(conv.input);
                hash.indices(&[conv.kernel.0, conv.kernel.1, conv.kernel.2, conv.kernel.3]);
                hash.indices(&[conv.ishape.0, conv.ishape.1, conv.ishape.2]);
                hash.indices(&[conv.oshape.0, conv.oshape.1, conv.oshape.2]);
                hash.indices(&[conv.stride.0, conv.stride.1]);
                hash.indices(&[conv.pads.0, conv.pads.1, conv.pads.2, conv.pads.3]);
                hash.flag(conv.transposed);
                hash.floats(&conv.wmat);
                hash.floats(&conv.wt);
                hash.floats(&conv.bias);
                hash.floats(&conv.bias_err);
                hash.scalar(conv.weight_rel_err);
                hash.indices(&conv.gather);
                hash.word(conv.back_taps.len());
                for taps in &conv.back_taps {
                    hash.word(taps.len());
                    for &(row, col, spatial) in taps {
                        hash.indices(&[row, col, spatial]);
                    }
                }
                hash.word(conv.k_fwd);
                hash.word(conv.k_bwd);
            }
            TwinOp::ChannelAffine {
                input,
                scale,
                shift,
                scale_rel_err,
                shift_err,
            } => {
                hash.word(1);
                hash.word(*input);
                hash.floats(scale);
                hash.floats(shift);
                hash.scalar(*scale_rel_err);
                hash.floats(shift_err);
            }
            TwinOp::Relu { input, layer } => {
                hash.word(2);
                hash.word(*input);
                hash.word(*layer);
            }
            TwinOp::Add { lhs, rhs } => {
                hash.word(3);
                hash.word(*lhs);
                hash.word(*rhs);
            }
            TwinOp::Flatten { input } => {
                hash.word(4);
                hash.word(*input);
            }
            TwinOp::Gemm {
                input,
                weight,
                bias,
                bias_err,
                weight_rel_err,
                shape,
            } => {
                hash.word(5);
                hash.word(*input);
                hash.floats(weight);
                hash.floats(bias);
                hash.floats(bias_err);
                hash.scalar(*weight_rel_err);
                hash.indices(&[shape.0, shape.1]);
            }
        }
    }
    hash.finish()
}

// Byte-identical same-process fingerprint with bounded original-deadline
// checks. Existing head hashing/caps/semantics remain unchanged.
#[cfg_attr(not(test), expect(dead_code, reason = "bounded input-row integration"))]
fn hash_floats_with_deadline(
    hash: &mut Hash,
    values: &[f64],
    deadline: Instant,
) -> ny_core::Result<()> {
    hash.word(values.len());
    for chunk in values.chunks(4096) {
        super::net::check_deadline(deadline)?;
        hash.0.update(bytemuck::cast_slice::<f64, u8>(chunk));
    }
    super::net::check_deadline(deadline)
}
#[cfg_attr(not(test), expect(dead_code, reason = "bounded input-row integration"))]
fn hash_indices_with_deadline(
    hash: &mut Hash,
    values: &[usize],
    deadline: Instant,
) -> ny_core::Result<()> {
    hash.word(values.len());
    for chunk in values.chunks(4096) {
        super::net::check_deadline(deadline)?;
        for &value in chunk {
            hash.word(value);
        }
    }
    super::net::check_deadline(deadline)
}

fn source_digest_with_deadline(net: &TwinNet, deadline: Instant) -> ny_core::Result<[u8; 32]> {
    super::net::check_deadline(deadline)?;
    let mut hash = Hash::new(b"ny/native-head-source-network/v1");
    hash.word(net.n_in);
    hash.word(net.n_y);
    hash.word(net.n_out);
    hash.word(net.i_gemm1);
    hash_indices_with_deadline(&mut hash, &net.tsize, deadline)?;
    hash_indices_with_deadline(&mut hash, &net.trunk_relus, deadline)?;
    hash.word(net.ops.len());
    for op in &net.ops {
        super::net::check_deadline(deadline)?;
        match op {
            TwinOp::Conv(conv) => {
                hash.word(0);
                hash.word(conv.input);
                hash_indices_with_deadline(
                    &mut hash,
                    &[conv.kernel.0, conv.kernel.1, conv.kernel.2, conv.kernel.3],
                    deadline,
                )?;
                hash_indices_with_deadline(
                    &mut hash,
                    &[conv.ishape.0, conv.ishape.1, conv.ishape.2],
                    deadline,
                )?;
                hash_indices_with_deadline(
                    &mut hash,
                    &[conv.oshape.0, conv.oshape.1, conv.oshape.2],
                    deadline,
                )?;
                hash_indices_with_deadline(&mut hash, &[conv.stride.0, conv.stride.1], deadline)?;
                hash_indices_with_deadline(
                    &mut hash,
                    &[conv.pads.0, conv.pads.1, conv.pads.2, conv.pads.3],
                    deadline,
                )?;
                hash.flag(conv.transposed);
                hash_floats_with_deadline(&mut hash, &conv.wmat, deadline)?;
                hash_floats_with_deadline(&mut hash, &conv.wt, deadline)?;
                hash_floats_with_deadline(&mut hash, &conv.bias, deadline)?;
                hash_floats_with_deadline(&mut hash, &conv.bias_err, deadline)?;
                hash.scalar(conv.weight_rel_err);
                hash_indices_with_deadline(&mut hash, &conv.gather, deadline)?;
                hash.word(conv.back_taps.len());
                for (i, taps) in conv.back_taps.iter().enumerate() {
                    if i.is_multiple_of(4096) {
                        super::net::check_deadline(deadline)?;
                    }
                    hash.word(taps.len());
                    for (i, &(row, col, spatial)) in taps.iter().enumerate() {
                        if i.is_multiple_of(4096) {
                            super::net::check_deadline(deadline)?;
                        }
                        hash_indices_with_deadline(&mut hash, &[row, col, spatial], deadline)?;
                    }
                }
                hash.word(conv.k_fwd);
                hash.word(conv.k_bwd);
            }
            TwinOp::ChannelAffine {
                input,
                scale,
                shift,
                scale_rel_err,
                shift_err,
            } => {
                hash.word(1);
                hash.word(*input);
                hash_floats_with_deadline(&mut hash, scale, deadline)?;
                hash_floats_with_deadline(&mut hash, shift, deadline)?;
                hash.scalar(*scale_rel_err);
                hash_floats_with_deadline(&mut hash, shift_err, deadline)?;
            }
            TwinOp::Relu { input, layer } => {
                hash.word(2);
                hash.word(*input);
                hash.word(*layer);
            }
            TwinOp::Add { lhs, rhs } => {
                hash.word(3);
                hash.word(*lhs);
                hash.word(*rhs);
            }
            TwinOp::Flatten { input } => {
                hash.word(4);
                hash.word(*input);
            }
            TwinOp::Gemm {
                input,
                weight,
                bias,
                bias_err,
                weight_rel_err,
                shape,
            } => {
                hash.word(5);
                hash.word(*input);
                hash_floats_with_deadline(&mut hash, weight, deadline)?;
                hash_floats_with_deadline(&mut hash, bias, deadline)?;
                hash_floats_with_deadline(&mut hash, bias_err, deadline)?;
                hash.scalar(*weight_rel_err);
                hash_indices_with_deadline(&mut hash, &[shape.0, shape.1], deadline)?;
            }
        }
    }
    super::net::check_deadline(deadline)?;
    Ok(hash.finish())
}

fn gates_digest(root: &RootGates) -> [u8; 32] {
    let mut hash = Hash::new(b"ny/native-head-source-gates/v1");
    hash.flag(root.mode.outward());
    hash.floats(&root.mid);
    hash.floats(&root.rad);
    hash.floats(&root.xabs);
    hash.floats(&root.lo);
    hash.floats(&root.hi);
    hash.word(root.layers.len());
    for layer in &root.layers {
        hash.word(layer.op);
        hash.word(layer.n);
        hash.floats(&layer.l);
        hash.floats(&layer.u);
        hash.floats(&layer.alpha);
        hash.floats(&layer.s);
        hash.floats(&layer.c);
        hash.floats(&layer.ms);
        hash.indices(&layer.unst);
        hash.flag(layer.clip_rows.is_some());
        if let Some(rows) = &layer.clip_rows {
            hash.matrix(&rows.m);
            hash.matrix(&rows.d);
            hash.floats(&rows.sl_lo);
            hash.floats(&rows.sl_up);
            hash.floats(&rows.bm);
            hash.floats(&rows.bd);
        }
    }
    hash.flag(root.head_alpha.is_some());
    if let Some(alpha) = &root.head_alpha {
        hash.floats(alpha);
    }
    // These policies select a different source/error producer. The values
    // are frozen, and their equality is pinned with the exact gate frame.
    let policy = root.root_precision_policy;
    hash.flag(policy.root_f32);
    hash.flag(policy.tableau_f32);
    hash.flag(policy.ibp_seed);
    hash.0.update(policy.err_lane.token().as_bytes());
    hash.0.update(policy.box_source.token().as_bytes());
    hash.flag(root.spec_alpha_policy.is_some());
    hash.finish()
}
