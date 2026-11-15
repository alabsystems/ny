// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Paid original-DAG root construction and complete retained row coverage.
//!
//! This is a numerical source, not a RootGates/RetainCfg completion token.
//! Reference boxes are released at their original last consumer, and each
//! ReLU moves its endpoints through one refinement transaction for every tile.

use std::mem::size_of;
use std::sync::Arc;
use std::time::Instant;

use ny_core::{NyError, Result};

use super::compiler::OriginalModelIdentity;
use super::ir::{checkpoint, earliest, invalid, refusal};
use super::liveness::OriginalLiveness;
use super::prefix::{CertifiedTensorBox, OriginalRoot, PreparedPrefix};
use super::reference_execution::{
    PreparedReferenceArithmetic, PreparedReferenceReceipt, ReferenceExecution,
};
use crate::certify::work::{LeaseClass, PersistentReservation, WorkBudget, WorkLease, WorkRequest};
use crate::margin_row::engine::run_prepared_prefix_input_rows;
use crate::margin_row::errlane::ErrLane;
use crate::margin_row::input_rows::InputRowPair;
use crate::margin_row::net::TwinOp;
use crate::margin_row::patch::{run_prepared_prefix_patch_input_rows, PreparedPatchPlan};
use crate::margin_row::root::LayerGates;
use crate::margin_row::root_box::ROOT_SOURCE_BYTES_CAP;

const META: usize = 256;
const POLL: usize = 256;

/// Internal deterministic scheduling. This does not reinterpret precision,
/// clipping, ranker retention or source lever requests.
#[derive(Clone, Copy, Debug)]
pub(in crate::margin_row) struct PreparedConstructionOptions {
    pub(in crate::margin_row) reference_arithmetic: PreparedReferenceArithmetic,
    pub(in crate::margin_row) keep_input_rows: bool,
    pub(in crate::margin_row) row_f32: bool,
    pub(in crate::margin_row) prefer_patch: bool,
    pub(in crate::margin_row) dense_tile: usize,
    pub(in crate::margin_row) patch_tile: usize,
    pub(in crate::margin_row) patch_ratio: f64,
}
impl Default for PreparedConstructionOptions {
    fn default() -> Self {
        Self {
            reference_arithmetic: PreparedReferenceArithmetic::F64,
            keep_input_rows: false,
            row_f32: false,
            prefer_patch: true,
            dense_tile: 64,
            patch_tile: 2048,
            patch_ratio: 0.9,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(in crate::margin_row) struct PreparedConstructionStats {
    pub(in crate::margin_row) derived_ops: usize,
    pub(in crate::margin_row) refined_rows: usize,
    pub(in crate::margin_row) dense_tiles: usize,
    pub(in crate::margin_row) patch_tiles: usize,
    pub(in crate::margin_row) f32_tiles: usize,
    pub(in crate::margin_row) f32_coefficient_cells: usize,
    pub(in crate::margin_row) f32_bias_cells: usize,
    pub(in crate::margin_row) f32_changed_cells: usize,
    pub(in crate::margin_row) f32_payload_bytes_before_total: usize,
    pub(in crate::margin_row) f32_payload_bytes_after_total: usize,
    pub(in crate::margin_row) f32_nominal_allocation_bytes_total: usize,
    pub(in crate::margin_row) f32_max_admitted_peak_bytes: usize,
    pub(in crate::margin_row) reference_peak_bytes: usize,
    pub(in crate::margin_row) max_live_reference_boxes: usize,
    pub(in crate::margin_row) released_reference_boxes: usize,
}

/// The Arc allocation itself has a last-owner reservation. Sharing this owner
/// cannot detach the row payload, model, prefix or its paid wrapper lifetime.
#[derive(Clone)]
pub(in crate::margin_row) struct PreparedRowTile(Arc<TileInner>);
struct TileInner {
    rows: InputRowPair<'static>,
    _resident: PersistentReservation,
}
impl PreparedRowTile {
    pub(in crate::margin_row) fn rows(&self) -> &InputRowPair<'static> {
        &self.0.rows
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::margin_row) struct PreparedRowLocation {
    pub(in crate::margin_row) target: usize,
    pub(in crate::margin_row) tile: usize,
    pub(in crate::margin_row) column: usize,
}

/// Producer-sealed coverage for one original ReLU. Locations are in sorted
/// original-ID order and point to actual ordered columns, never compacted IDs.
pub(in crate::margin_row) struct PreparedLayerProducts {
    op: usize,
    layer: usize,
    prefix_before: PreparedPrefix,
    tiles: Vec<PreparedRowTile>,
    coverage: Vec<PreparedRowLocation>,
    keep_input_rows: bool,
    _resident: PersistentReservation,
}
impl PreparedLayerProducts {
    pub(in crate::margin_row) fn op(&self) -> usize {
        self.op
    }
    pub(in crate::margin_row) fn layer(&self) -> usize {
        self.layer
    }
    pub(in crate::margin_row) fn prefix_before(&self) -> &PreparedPrefix {
        &self.prefix_before
    }
    pub(in crate::margin_row) fn tiles(&self) -> &[PreparedRowTile] {
        &self.tiles
    }
    pub(in crate::margin_row) fn coverage(&self) -> &[PreparedRowLocation] {
        &self.coverage
    }
    pub(in crate::margin_row) fn row(
        &self,
        target: usize,
    ) -> Option<(&InputRowPair<'static>, usize)> {
        let i = self
            .coverage
            .binary_search_by_key(&target, |p| p.target)
            .ok()?;
        let location = self.coverage[i];
        Some((self.tiles.get(location.tile)?.rows(), location.column))
    }

    fn validate(
        &self,
        final_prefix: &PreparedPrefix,
        gate: &LayerGates,
        at: Instant,
    ) -> Result<()> {
        checkpoint(final_prefix.root().budget(), Some(at))?;
        if !self.prefix_before.root().same_root(final_prefix.root())
            || self.prefix_before.layer_count() != self.layer
            || self.prefix_before.for_relu(self.op).is_err()
        {
            return Err(invalid("constructed layer prefix/op identity changed"));
        }
        if gate.op != self.op {
            return Err(invalid("constructed layer gate op changed"));
        }
        if !self.keep_input_rows {
            if !self.tiles.is_empty() || !self.coverage.is_empty() {
                return Err(invalid("unrequested persistent row delivery"));
            }
            return Ok(());
        }
        if gate.unst.len() != self.coverage.len() {
            return Err(invalid("incomplete final unstable row coverage"));
        }
        for (i, (&target, location)) in gate.unst.iter().zip(&self.coverage).enumerate() {
            poll(final_prefix.root().budget(), at, i)?;
            let rows = self
                .tiles
                .get(location.tile)
                .ok_or_else(|| invalid("missing constructed row tile"))?
                .rows();
            if location.target != target || rows.target(location.column) != Some(target) {
                return Err(invalid("constructed row original-ID mapping changed"));
            }
        }
        for tile in &self.tiles {
            tile.rows().validate_prepared_prefix_with_deadline(
                &self.prefix_before,
                self.op,
                tile.rows().targets(),
                at,
            )?;
        }
        checkpoint(final_prefix.root().budget(), Some(at))
    }
}

/// Only a completed builder can mint this result. Persistent rows, if asked
/// for, cover every final unstable row of every completed original layer.
/// Root precision/ranker/clip adapters must establish their own delivery.
pub(in crate::margin_row) struct PreparedRootConstruction {
    prefix: PreparedPrefix,
    trunk_output: CertifiedTensorBox,
    layers: Vec<PreparedLayerProducts>,
    keep_input_rows: bool,
    deadline: Instant,
    stats: PreparedConstructionStats,
    reference: ReferenceExecution,
    _resident: PersistentReservation,
}
impl PreparedRootConstruction {
    pub(in crate::margin_row) fn prefix(&self) -> &PreparedPrefix {
        &self.prefix
    }
    pub(in crate::margin_row) fn trunk_output(&self) -> &CertifiedTensorBox {
        &self.trunk_output
    }
    pub(in crate::margin_row) fn layer_products(
        &self,
        layer: usize,
    ) -> Option<&PreparedLayerProducts> {
        self.layers.get(layer)
    }
    pub(in crate::margin_row) fn layers(&self) -> &[PreparedLayerProducts] {
        &self.layers
    }
    pub(in crate::margin_row) fn identity(&self) -> &OriginalModelIdentity {
        self.prefix.root().model().identity()
    }
    pub(in crate::margin_row) fn reference_receipt(&self) -> PreparedReferenceReceipt {
        self.reference.receipt()
    }
    pub(in crate::margin_row) fn reference_checkpoint(
        &self,
        requested: Option<Instant>,
    ) -> Result<()> {
        let at = self.deadline(requested)?;
        if !self._resident.shares_ledger(self.prefix.root().budget())
            || self._resident.bytes() < self.reference.record_bytes()?
        {
            return Err(invalid(
                "reference construction lost original ledger/capacity",
            ));
        }
        self.reference.checkpoint(self.prefix.root(), at)
    }
    pub(in crate::margin_row) fn validate_reference(
        &self,
        requested: Option<Instant>,
    ) -> Result<()> {
        self.reference_checkpoint(requested)?;
        self.reference
            .validate(self.prefix.root(), self.deadline(requested)?)
    }
    pub(in crate::margin_row) fn stats(&self) -> &PreparedConstructionStats {
        &self.stats
    }
    pub(in crate::margin_row) fn deadline(&self, requested: Option<Instant>) -> Result<Instant> {
        self.prefix
            .deadline(earliest(Some(self.deadline), requested))?
            .ok_or_else(|| invalid("missing constructed source deadline"))
    }
    pub(in crate::margin_row) fn coverage_complete(
        &self,
        requested: Option<Instant>,
    ) -> Result<bool> {
        let at = self.deadline(requested)?;
        if !self.keep_input_rows {
            return Ok(false);
        }
        if self.layers.len() != self.prefix.layer_count() {
            return Err(invalid("constructed final prefix layer census changed"));
        }
        // Private successful-production counters bound both the retained
        // ordered target scan and final mapping scan. Admit BEFORE traversing
        // any layer/tile census, including repeated refused coverage queries.
        let count = self
            .stats
            .refined_rows
            .checked_mul(2)
            .and_then(|n| n.checked_add(self.stats.dense_tiles))
            .and_then(|n| n.checked_add(self.stats.patch_tiles))
            .and_then(|n| n.checked_add(self.layers.len()))
            .ok_or_else(overflow)?;
        let mut validation = admit(self.prefix.root().budget(), 0, 0, charged(count, 128)?, at)?;
        let at = min_lease(at, &validation);
        let frame = self.prefix.frame(Some(at))?;
        for layer in &self.layers {
            layer.validate(
                &self.prefix,
                frame
                    .gate(layer.layer)
                    .ok_or_else(|| invalid("missing final constructed layer gate"))?,
                at,
            )?;
        }
        validation.complete().map_err(refusal)?;
        Ok(true)
    }
}

pub(in crate::margin_row) fn build_prepared_root(
    root: OriginalRoot,
    options: PreparedConstructionOptions,
    requested: Option<Instant>,
) -> Result<PreparedRootConstruction> {
    build_inner(root, options, requested, None)
}

// The observer is private and only used by controls to cancel the original
// authority during actual DAG construction, rather than fabricating products.
fn build_inner(
    root: OriginalRoot,
    options: PreparedConstructionOptions,
    requested: Option<Instant>,
    mut after_op: Option<&mut dyn FnMut(usize)>,
) -> Result<PreparedRootConstruction> {
    let budget = root.budget();
    let mut at = root
        .deadline(requested)?
        .ok_or_else(|| invalid("prepared construction requires finite inherited deadline"))?;
    ny_core::require_f64_interval_proof_environment()?;
    let policy = root.policy();
    if policy.root_f32 || policy.tableau_f32 || policy.err_lane != ErrLane::PerCol {
        return Err(invalid(
            "prepared construction needs f64 root/tableau and per-column policy",
        ));
    }
    if options.dense_tile == 0
        || options.dense_tile > 64
        || options.patch_tile == 0
        || options.patch_tile > 2048
        || !options.patch_ratio.is_finite()
        || !(0.05..=1.0).contains(&options.patch_ratio)
        || (options.row_f32 && !options.keep_input_rows)
    {
        return Err(invalid(
            "unsupported prepared construction product/schedule request",
        ));
    }
    // All aliases of the original budget see this monotone restriction. The
    // combined live peak is admitted atomically for every subsequent phase.
    budget
        .tighten_total_byte_limit(ROOT_SOURCE_BYTES_CAP)
        .map_err(refusal)?;
    let net = root.model().net();
    let liveness = OriginalLiveness::new(root.model().clone(), Some(at))?;
    if !liveness.same_model(root.model()) {
        return Err(invalid("constructed liveness owner changed"));
    }
    let tensor_count = net.i_gemm1.checked_add(1).ok_or_else(overflow)?;
    let layer_count = net.trunk_relus.len();
    let construction_owner_bytes = owner::<PreparedRootConstruction>()?;
    let result_bytes = bytes::<PreparedLayerProducts>(layer_count)?
        .checked_add(ReferenceExecution::allocation_bytes(tensor_count)?)
        .and_then(|n| n.checked_add(construction_owner_bytes))
        .ok_or_else(overflow)?;
    let registry_bytes = bytes::<Option<CertifiedTensorBox>>(tensor_count)?
        .checked_add(META)
        .ok_or_else(overflow)?;
    let mut whole = admit(
        budget,
        result_bytes,
        registry_bytes,
        charged(tensor_count, 4096)?,
        at,
    )?;
    at = min_lease(at, &whole);
    let mut registry = exact_vec(tensor_count)?;
    for i in 0..tensor_count {
        poll(budget, at, i)?;
        registry.push(None);
    }
    let mut layers = exact_vec(layer_count)?;
    let mut reference =
        ReferenceExecution::new(root.clone(), options.reference_arithmetic, tensor_count, at)?;
    whole.mark_materialized().map_err(refusal)?;
    let mut prefix = root.empty_prefix_with_deadline(Some(at))?;
    registry[0] = Some(root.input_box_with_deadline(Some(at))?);
    let mut stats = PreparedConstructionStats::default();
    let mut reference_live = registry[0].as_ref().unwrap().resident_bytes();
    let mut reference_boxes = 1usize;
    record_peak(&mut stats, reference_live, reference_boxes);
    for op in 0..net.i_gemm1 {
        checkpoint(budget, Some(at))?;
        let output = match &net.ops[op] {
            TwinOp::Relu { input, layer } => {
                if *layer != layers.len() || prefix.layer_count() != *layer {
                    return Err(invalid("constructed original ReLU layer order changed"));
                }
                let pre = registry
                    .get_mut(*input)
                    .and_then(Option::take)
                    .ok_or_else(|| invalid("missing live original preactivation"))?;
                let n = pre.lo().len();
                // Pay the worst-case ID capacity and complete selection scan
                // before allocating or reading every reference neuron.
                let mut selection = admit(
                    budget,
                    0,
                    bytes::<usize>(n)?.checked_add(META).ok_or_else(overflow)?,
                    charged(n, 32)?,
                    at,
                )?;
                at = min_lease(at, &selection);
                let mut targets = exact_vec(n)?;
                selection.mark_materialized().map_err(refusal)?;
                for i in 0..n {
                    poll(budget, at, i)?;
                    if pre.lo()[i] < 0.0 && pre.hi()[i] > 0.0 {
                        targets.push(i);
                    }
                }
                let count = targets.len();
                let min_tile = options.dense_tile.min(options.patch_tile);
                let tile_capacity = count.div_ceil(min_tile);
                let kept_capacity = if options.keep_input_rows {
                    tile_capacity
                } else {
                    0
                };
                let map_capacity = if options.keep_input_rows { count } else { 0 };
                let product_bytes = bytes::<PreparedRowTile>(kept_capacity)?
                    .checked_add(bytes::<PreparedRowLocation>(map_capacity)?)
                    .and_then(|v| v.checked_add(META))
                    .ok_or_else(overflow)?;
                let mut products = admit(budget, product_bytes, 0, charged(count, 128)?, at)?;
                at = min_lease(at, &products);
                let mut tiles = exact_vec(kept_capacity)?;
                let mut coverage = exact_vec(map_capacity)?;
                products.mark_materialized().map_err(refusal)?;
                let before = prefix.clone();
                let mut accumulator = prefix.begin_refinement(pre, op, Some(at))?;
                // A declined paid plan may use the genuine bounded dense
                // producer. Numerical producer errors abort the entire source.
                let mut plan = if count != 0 && options.prefer_patch {
                    match PreparedPatchPlan::build(&prefix, options.patch_ratio, Some(at)) {
                        Ok(plan) => Some(plan),
                        Err(error) => {
                            checkpoint(budget, Some(at))?;
                            if matches!(error, NyError::DeadlineExceeded(_)) {
                                return Err(error);
                            }
                            None
                        }
                    }
                } else {
                    None
                };
                let mut position = 0;
                while position < count {
                    checkpoint(budget, Some(at))?;
                    let width = if plan.is_some() {
                        options.patch_tile
                    } else {
                        options.dense_tile
                    };
                    let end = position.checked_add(width).ok_or_else(overflow)?.min(count);
                    let mut row_pair = if let Some(patch) = &plan {
                        let result = match run_prepared_prefix_patch_input_rows(
                            patch,
                            &targets[position..end],
                            Some(at),
                        ) {
                            Ok(rows) => rows,
                            Err(error) if patch_can_fallback(&error) => {
                                checkpoint(budget, Some(at))?;
                                drop(plan.take());
                                continue;
                            }
                            Err(error) => return Err(error),
                        };
                        stats.patch_tiles =
                            stats.patch_tiles.checked_add(1).ok_or_else(overflow)?;
                        result
                    } else {
                        let result = run_prepared_prefix_input_rows(
                            &prefix,
                            &targets[position..end],
                            Some(at),
                        )?;
                        stats.dense_tiles =
                            stats.dense_tiles.checked_add(1).ok_or_else(overflow)?;
                        result
                    };
                    at = at.min(row_pair.deadline());
                    if options.row_f32 {
                        let receipt = budget.receipt().map_err(refusal)?;
                        let current = receipt
                            .live_scratch_bytes
                            .checked_add(receipt.live_persistent_bytes)
                            .ok_or_else(overflow)?;
                        let transfer = row_pair.into_f32(current, ROOT_SOURCE_BYTES_CAP, at)?;
                        stats.f32_tiles = stats.f32_tiles.checked_add(1).ok_or_else(overflow)?;
                        stats.f32_coefficient_cells = stats
                            .f32_coefficient_cells
                            .checked_add(transfer.receipt.coefficient_cells_stored)
                            .ok_or_else(overflow)?;
                        stats.f32_bias_cells = stats
                            .f32_bias_cells
                            .checked_add(transfer.receipt.bias_cells_stored)
                            .ok_or_else(overflow)?;
                        stats.f32_changed_cells = stats
                            .f32_changed_cells
                            .checked_add(transfer.receipt.changed_cells)
                            .ok_or_else(overflow)?;
                        stats.f32_payload_bytes_before_total = stats
                            .f32_payload_bytes_before_total
                            .checked_add(transfer.receipt.old_payload_bytes)
                            .ok_or_else(overflow)?;
                        stats.f32_payload_bytes_after_total = stats
                            .f32_payload_bytes_after_total
                            .checked_add(transfer.receipt.new_payload_bytes)
                            .ok_or_else(overflow)?;
                        stats.f32_nominal_allocation_bytes_total = stats
                            .f32_nominal_allocation_bytes_total
                            .checked_add(transfer.receipt.new_nominal_allocation_bytes)
                            .ok_or_else(overflow)?;
                        stats.f32_max_admitted_peak_bytes = stats
                            .f32_max_admitted_peak_bytes
                            .max(transfer.receipt.peak_admitted_bytes);
                        at = at.min(transfer.receipt.deadline);
                        row_pair = transfer.rows;
                    }
                    accumulator.apply_rows(&row_pair, Some(at))?;
                    stats.refined_rows = stats
                        .refined_rows
                        .checked_add(end - position)
                        .ok_or_else(overflow)?;
                    if options.keep_input_rows {
                        let mut header = admit(budget, owner::<TileInner>()?, 0, 1, at)?;
                        at = min_lease(at, &header);
                        checkpoint(budget, Some(at))?;
                        header.mark_materialized().map_err(refusal)?;
                        let resident = header.commit_persistent().map_err(refusal)?;
                        if tiles.len() >= tiles.capacity() {
                            return Err(invalid("constructed tile header capacity exceeded"));
                        }
                        tiles.push(PreparedRowTile(Arc::new(TileInner {
                            rows: row_pair,
                            _resident: resident,
                        })));
                    }
                    position = end;
                }
                drop(plan);
                let refined = accumulator.finish()?;
                if options.keep_input_rows {
                    for (tile_index, tile) in tiles.iter().enumerate() {
                        checkpoint(budget, Some(at))?;
                        tile.rows()
                            .validate_prepared_prefix(&before, op, tile.rows().targets())?;
                        for (column, &target) in tile.rows().targets().iter().enumerate() {
                            poll(budget, at, column)?;
                            if refined.lo()[target] < 0.0 && refined.hi()[target] > 0.0 {
                                coverage.push(PreparedRowLocation {
                                    target,
                                    tile: tile_index,
                                    column,
                                });
                            }
                        }
                    }
                    // The certified tightened box already proves these phases.
                    // Drop solved-only row allocations BEFORE paying the next
                    // gate/output overlap; no second tile vector is allocated.
                    let mut mapped = 0;
                    let mut retained = 0;
                    for old_tile in 0..tiles.len() {
                        poll(budget, at, old_tile)?;
                        if coverage.get(mapped).is_some_and(|p| p.tile == old_tile) {
                            while let Some(location) = coverage.get_mut(mapped) {
                                if location.tile != old_tile {
                                    break;
                                }
                                location.tile = retained;
                                mapped += 1;
                            }
                            if retained != old_tile {
                                tiles.swap(retained, old_tile);
                            }
                            retained += 1;
                        }
                    }
                    if mapped != coverage.len() {
                        return Err(invalid("incomplete constructed tile compaction"));
                    }
                    tiles.truncate(retained);
                    checkpoint(budget, Some(at))?;
                }
                let (next, output) = prefix.commit_relu(&refined, op, None)?;
                at = next
                    .deadline(Some(at))?
                    .ok_or_else(|| invalid("missing committed prefix deadline"))?;
                let resident = products.commit_persistent().map_err(refusal)?;
                let product = PreparedLayerProducts {
                    op,
                    layer: *layer,
                    prefix_before: before,
                    tiles,
                    coverage,
                    keep_input_rows: options.keep_input_rows,
                    _resident: resident,
                };
                product.validate(
                    &next,
                    next.gate(*layer)
                        .ok_or_else(|| invalid("missing committed original gate"))?,
                    at,
                )?;
                layers.push(product);
                registry[*input] = Some(refined);
                prefix = next;
                selection.complete().map_err(refusal)?;
                output
            }
            original => {
                let (left, right) = match original {
                    TwinOp::Conv(c) => (c.input, None),
                    TwinOp::ChannelAffine { input, .. } | TwinOp::Flatten { input } => {
                        (*input, None)
                    }
                    TwinOp::Add { lhs, rhs } => (*lhs, Some(*rhs)),
                    _ => {
                        return Err(invalid(
                            "unexpected original operator before first head Gemm",
                        ))
                    }
                };
                let first = registry
                    .get(left)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| invalid("missing live original lhs"))?;
                let second = right
                    .map(|id| {
                        registry
                            .get(id)
                            .and_then(Option::as_ref)
                            .ok_or_else(|| invalid("missing live original rhs"))
                    })
                    .transpose()?;
                if matches!(original, TwinOp::Conv(_) | TwinOp::ChannelAffine { .. }) {
                    let (output, observed) = match options.reference_arithmetic {
                        PreparedReferenceArithmetic::Binary32Certified => {
                            let (output, receipt) =
                                root.derive_reference_f32_with_deadline(op, first, Some(at))?;
                            (output, Some(receipt))
                        }
                        PreparedReferenceArithmetic::F64 => (
                            root.derive_with_deadline(op, first, second, Some(at))?,
                            None,
                        ),
                    };
                    reference.record(op, first, &output, observed)?;
                    output
                } else {
                    root.derive_with_deadline(op, first, second, Some(at))?
                }
            }
        };
        at = root
            .deadline(earliest(Some(at), output.deadline()?))?
            .ok_or_else(|| invalid("missing derived source deadline"))?;
        stats.derived_ops = stats.derived_ops.checked_add(1).ok_or_else(overflow)?;
        reference_live = reference_live
            .checked_add(output.resident_bytes())
            .ok_or_else(overflow)?;
        reference_boxes = reference_boxes.checked_add(1).ok_or_else(overflow)?;
        record_peak(&mut stats, reference_live, reference_boxes);
        registry[op + 1] = Some(output);
        for (i, &tensor) in liveness
            .release_after(op)
            .ok_or_else(|| invalid("missing original release boundary"))?
            .iter()
            .enumerate()
        {
            poll(budget, at, i)?;
            if let Some(value) = registry.get_mut(tensor).and_then(Option::take) {
                reference_live = reference_live
                    .checked_sub(value.resident_bytes())
                    .ok_or_else(overflow)?;
                reference_boxes = reference_boxes.checked_sub(1).ok_or_else(overflow)?;
                stats.released_reference_boxes = stats
                    .released_reference_boxes
                    .checked_add(1)
                    .ok_or_else(overflow)?;
                drop(value);
            }
        }
        if let Some(observer) = after_op.as_mut() {
            observer(op);
        }
        checkpoint(budget, Some(at))?;
    }
    let head_input = match net.ops.get(net.i_gemm1) {
        Some(TwinOp::Gemm { input, .. }) => *input,
        _ => return Err(invalid("missing original first head Gemm")),
    };
    let trunk_output = registry
        .get_mut(head_input)
        .and_then(Option::take)
        .ok_or_else(|| invalid("missing live original head input"))?;
    if prefix.layer_count() != layer_count || layers.len() != layer_count {
        return Err(invalid("incomplete constructed original trunk"));
    }
    for (i, value) in registry.iter_mut().enumerate() {
        poll(budget, at, i)?;
        drop(value.take());
    }
    drop(registry);
    drop(liveness);
    checkpoint(budget, Some(at))?;
    reference.finish(&root, at)?;
    let resident = whole.commit_persistent().map_err(refusal)?;
    let result = PreparedRootConstruction {
        prefix,
        trunk_output,
        layers,
        keep_input_rows: options.keep_input_rows,
        deadline: at,
        stats,
        reference,
        _resident: resident,
    };
    if options.keep_input_rows && !result.coverage_complete(Some(at))? {
        return Err(invalid("requested constructed coverage was not completed"));
    }
    result.validate_reference(Some(at))?;
    Ok(result)
}

fn record_peak(stats: &mut PreparedConstructionStats, bytes: usize, boxes: usize) {
    stats.reference_peak_bytes = stats.reference_peak_bytes.max(bytes);
    stats.max_live_reference_boxes = stats.max_live_reference_boxes.max(boxes);
}
fn patch_can_fallback(error: &NyError) -> bool {
    match error {
        NyError::CpuMemoryExceeded { .. } => true,
        NyError::UnsupportedConfiguration(message) => {
            message.contains("prepared patch structural refusal")
                || message.starts_with("margin_row prepared patch admission:")
        }
        _ => false,
    }
}
fn min_lease(at: Instant, lease: &WorkLease) -> Instant {
    lease.deadline().map_or(at, |d| at.min(d))
}
fn admit(
    budget: &WorkBudget,
    persistent: usize,
    scratch: usize,
    work: u64,
    at: Instant,
) -> Result<WorkLease> {
    checkpoint(budget, Some(at))?;
    let lease = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                persistent_bytes: persistent,
                scratch_bytes: scratch,
                work,
                ..WorkRequest::default()
            },
        )
        .map_err(refusal)?;
    checkpoint(budget, Some(min_lease(at, &lease)))?;
    Ok(lease)
}
fn poll(budget: &WorkBudget, at: Instant, i: usize) -> Result<()> {
    if i.is_multiple_of(POLL) {
        checkpoint(budget, Some(at))?;
    }
    Ok(())
}
fn bytes<T>(n: usize) -> Result<usize> {
    n.checked_mul(size_of::<T>()).ok_or_else(overflow)
}
fn owner<T>() -> Result<usize> {
    size_of::<T>().checked_add(META).ok_or_else(overflow)
}
fn charged(n: usize, cost: usize) -> Result<u64> {
    u64::try_from(n.checked_mul(cost).ok_or_else(overflow)?).map_err(|_| overflow())
}
fn exact_vec<T>(n: usize) -> Result<Vec<T>> {
    let mut v = Vec::new();
    v.try_reserve_exact(n)
        .map_err(|_| invalid("constructed allocation refused"))?;
    if v.capacity() != n {
        return Err(invalid(
            "constructed actual vector capacity exceeds paid bound",
        ));
    }
    Ok(v)
}
fn overflow() -> NyError {
    invalid("constructed capacity/work overflow")
}

#[cfg(test)]
#[path = "construction_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "construction_alternative_rows_tests.rs"]
mod alternative_rows_tests;
