// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Paid consumers of complete original prepared rows.
//!
//! Lower and upper A/E/b/eb remain independent producer-owned planes. Ranker
//! f32 packs are advice only. A clipping owner retains the actual immutable
//! source, ordered original IDs and row claims; a digest is a cache key, never
//! source authority. Ordinary RootGates, wave and rebake adapters are separate.

use std::cmp::Ordering;
use std::mem::size_of;
use std::sync::Arc;
use std::time::Instant;

use ny_core::{NyError, Result};
use sha2::{Digest, Sha256};

use super::ir::{checkpoint, invalid, refusal};
use super::publication::PreparedRootSource;
use crate::certify::work::{LeaseClass, PersistentReservation, WorkLease, WorkRequest};
use crate::margin_row::clip::Halfspace;
use crate::margin_row::input_clipping::InputSplitSide;
use crate::margin_row::input_rows::{InputRowPair, InputRowView, InputRowsBacking};
use crate::margin_row::root::RetainCfg;
use crate::margin_row::rounding::next_down;

const META: usize = 256;
const POLL: usize = 256;

/// A location in genuine, immutable construction tiles. Only the checked
/// constructor/filter can create a selection; callers cannot replace planes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RowLocation {
    layer: usize,
    target: usize,
    unstable_position: usize,
    tile: usize,
    column: usize,
}

/// All final unstable rows, or an authenticated ordered subsequence thereof.
/// This owner is deliberately not a process-wide calibration flag.
#[derive(Clone)]
pub(in crate::margin_row) struct PreparedClipRows(Arc<ClipRowsInner>);

struct ClipRowsInner {
    source: PreparedRootSource,
    locations: Vec<RowLocation>,
    deadline: Instant,
    digest: [u8; 32],
    resident: PersistentReservation,
}

impl PreparedClipRows {
    pub(in crate::margin_row) fn same_owner(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
    pub(in crate::margin_row) fn source(&self) -> &PreparedRootSource {
        &self.0.source
    }
    pub(in crate::margin_row) fn len(&self) -> usize {
        self.0.locations.len()
    }
    pub(in crate::margin_row) fn is_empty(&self) -> bool {
        self.0.locations.is_empty()
    }
    pub(in crate::margin_row) fn ids(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.0.locations.iter().map(|p| (p.layer, p.target))
    }
    pub(in crate::margin_row) fn digest(&self) -> &[u8; 32] {
        &self.0.digest
    }
    /// This selection's metadata claim only. Source and rows remain on their
    /// original independent owners and ledger, including after producer drop.
    pub(in crate::margin_row) fn resident_bytes(&self) -> usize {
        self.0.resident.bytes()
    }
    pub(in crate::margin_row) fn deadline(&self, requested: Option<Instant>) -> Result<Instant> {
        self.0.source.deadline(Some(
            requested.map_or(self.0.deadline, |d| d.min(self.0.deadline)),
        ))
    }
    pub(in crate::margin_row) fn validate_source(
        &self,
        source: &PreparedRootSource,
        requested: Option<Instant>,
    ) -> Result<()> {
        let at = self.deadline(requested)?;
        self.0.source.checkpoint(Some(at))?;
        if !self.source().same_source(source) || !self.0.resident.shares_ledger(source.budget()) {
            return Err(invalid("prepared clipping foreign source/ledger"));
        }
        Ok(())
    }
    /// Borrow explicit original rows, not reconstructed M/D or scalar bounds.
    pub(in crate::margin_row) fn row(
        &self,
        layer: usize,
        target: usize,
        requested: Option<Instant>,
    ) -> Result<(&InputRowPair<'static>, usize)> {
        let at = self.deadline(requested)?;
        self.validate_source(&self.0.source, Some(at))?;
        let i = self
            .0
            .locations
            .binary_search_by_key(&(layer, target), |p| (p.layer, p.target))
            .map_err(|_| invalid("prepared clipping target absent from ordered selection"))?;
        checked_row(&self.0.source, self.0.locations[i], at)
    }
    /// Paid immutable subsequence. Matrices and complete prefix gate arrays
    /// are shared through the source; only selected location metadata is copied.
    pub(in crate::margin_row) fn filter(
        &self,
        ordered_ids: &[(usize, usize)],
        requested: Option<Instant>,
    ) -> Result<Self> {
        let mut at = self.deadline(requested)?;
        self.validate_source(&self.0.source, Some(at))?;
        let capacity = add(
            shared_owner::<ClipRowsInner>()?,
            bytes::<RowLocation>(ordered_ids.len())?,
        )?;
        let mut lease = admit(
            &self.0.source,
            capacity,
            0,
            work(ordered_ids.len(), 128)?,
            at,
        )?;
        at = min_lease(at, &lease);
        let mut locations = exact_vec(ordered_ids.len())?;
        let mut previous = None;
        for (i, &key) in ordered_ids.iter().enumerate() {
            poll(&self.0.source, at, i)?;
            if previous.is_some_and(|p| p >= key) {
                return Err(invalid(
                    "prepared clipping filter is not an ordered unique subsequence",
                ));
            }
            let found = self
                .0
                .locations
                .binary_search_by_key(&key, |p| (p.layer, p.target))
                .map_err(|_| invalid("prepared clipping filter contains an absent original ID"))?;
            locations.push(self.0.locations[found]);
            previous = Some(key);
        }
        let digest = content_digest(&self.0.source, &locations, at)?;
        self.0.source.checkpoint(Some(at))?;
        lease.mark_materialized().map_err(refusal)?;
        let resident = lease.commit_persistent().map_err(refusal)?;
        let result = Self(Arc::new(ClipRowsInner {
            source: self.0.source.clone(),
            locations,
            deadline: at,
            digest,
            resident,
        }));
        result.validate_source(&self.0.source, Some(at))?;
        Ok(result)
    }
    pub(in crate::margin_row) fn halfspaces(
        &self,
        splits: &[PreparedSplit],
        requested: Option<Instant>,
    ) -> Result<PreparedHalfspaceBatch> {
        build_halfspaces(self, splits, requested)
    }
}

/// Build all final unstable rows. Missing persistent coverage is a refusal,
/// never an empty success that silently loses requested clipping capability.
pub(in crate::margin_row) fn build_prepared_clip_rows(
    source: &PreparedRootSource,
    requested: Option<Instant>,
) -> Result<PreparedClipRows> {
    let mut at = source.deadline(requested)?;
    source.checkpoint(Some(at))?;
    if !source.construction().coverage_complete(Some(at))? {
        return Err(invalid(
            "prepared clipping requires complete requested input rows",
        ));
    }
    let upper = source.construction().stats().refined_rows;
    let mut lease = admit(
        source,
        add(
            shared_owner::<ClipRowsInner>()?,
            bytes::<RowLocation>(upper)?,
        )?,
        0,
        work(upper, 256)?,
        at,
    )?;
    at = min_lease(at, &lease);
    let mut locations = exact_vec(upper)?;
    let frame = source.construction().prefix().frame(Some(at))?;
    for layer in source.construction().layers() {
        frame.checkpoint()?;
        let gate = frame
            .gate(layer.layer())
            .ok_or_else(|| invalid("prepared clipping missing final gate"))?;
        if gate.unst.len() != layer.coverage().len() || gate.op != layer.op() {
            return Err(invalid(
                "prepared clipping incomplete final unstable coverage",
            ));
        }
        for (pos, (&target, row)) in gate.unst.iter().zip(layer.coverage()).enumerate() {
            poll(source, at, pos)?;
            if target != row.target || locations.len() == upper {
                return Err(invalid(
                    "prepared clipping original coverage/capacity changed",
                ));
            }
            let location = RowLocation {
                layer: layer.layer(),
                target,
                unstable_position: pos,
                tile: row.tile,
                column: row.column,
            };
            checked_row(source, location, at)?;
            locations.push(location);
        }
    }
    drop(frame);
    let digest = content_digest(source, &locations, at)?;
    source.checkpoint(Some(at))?;
    lease.mark_materialized().map_err(refusal)?;
    let resident = lease.commit_persistent().map_err(refusal)?;
    let result = PreparedClipRows(Arc::new(ClipRowsInner {
        source: source.clone(),
        locations,
        deadline: at,
        digest,
        resident,
    }));
    result.validate_source(source, Some(at))?;
    Ok(result)
}

/// Genuine explicit rows plus separately stored ranker advice. Advice cannot
/// be converted into a clipping certificate and never averages row constants.
pub(in crate::margin_row) struct PreparedRetainedLayer {
    layer: usize,
    eligible: usize,
    idx: Vec<usize>,
    unstable_positions: Vec<usize>,
    lower_advice: Vec<f32>,
    upper_advice: Vec<f32>,
    naug: usize,
}
impl PreparedRetainedLayer {
    pub(in crate::margin_row) fn layer(&self) -> usize {
        self.layer
    }
    pub(in crate::margin_row) fn eligible(&self) -> usize {
        self.eligible
    }
    pub(in crate::margin_row) fn ids(&self) -> &[usize] {
        &self.idx
    }
    pub(in crate::margin_row) fn unstable_positions(&self) -> &[usize] {
        &self.unstable_positions
    }
    pub(in crate::margin_row) fn naug(&self) -> usize {
        self.naug
    }
    pub(in crate::margin_row) fn lower_advice(&self) -> &[f32] {
        &self.lower_advice
    }
    pub(in crate::margin_row) fn upper_advice(&self) -> &[f32] {
        &self.upper_advice
    }
    pub(in crate::margin_row) fn bytes(&self) -> usize {
        (self.lower_advice().len() + self.upper_advice().len()) * size_of::<f32>()
    }
}

#[derive(Clone)]
pub(in crate::margin_row) struct PreparedRetainedRows(Arc<RetainedRowsInner>);

struct RetainedRowsInner {
    source: PreparedRootSource,
    layers: Vec<PreparedRetainedLayer>,
    cfg: RetainCfg,
    payload_bytes: usize,
    digest: [u8; 32],
    deadline: Instant,
    resident: PersistentReservation,
}
impl PreparedRetainedRows {
    pub(in crate::margin_row) fn same_owner(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
    pub(in crate::margin_row) fn source(&self) -> &PreparedRootSource {
        &self.0.source
    }
    pub(in crate::margin_row) fn layers(&self) -> &[PreparedRetainedLayer] {
        &self.0.layers
    }
    pub(in crate::margin_row) fn payload_bytes(&self) -> usize {
        self.0.payload_bytes
    }
    pub(in crate::margin_row) fn digest(&self) -> &[u8; 32] {
        &self.0.digest
    }
    pub(in crate::margin_row) fn cfg(&self) -> RetainCfg {
        self.0.cfg
    }
    pub(in crate::margin_row) fn resident_bytes(&self) -> usize {
        self.0.resident.bytes()
    }
    pub(in crate::margin_row) fn validate_source(
        &self,
        source: &PreparedRootSource,
        requested: Option<Instant>,
    ) -> Result<()> {
        let at = self.deadline(requested)?;
        self.0.source.checkpoint(Some(at))?;
        if !self.source().same_source(source) || !self.0.resident.shares_ledger(source.budget()) {
            return Err(invalid("prepared retention foreign source/ledger"));
        }
        Ok(())
    }
    pub(in crate::margin_row) fn deadline(&self, requested: Option<Instant>) -> Result<Instant> {
        self.0.source.deadline(Some(
            requested.map_or(self.0.deadline, |d| d.min(self.0.deadline)),
        ))
    }
    pub(in crate::margin_row) fn row(
        &self,
        layer: usize,
        target: usize,
        requested: Option<Instant>,
    ) -> Result<(&InputRowPair<'static>, usize)> {
        let mut at = self.deadline(requested)?;
        self.validate_source(self.source(), Some(at))?;
        // Two immutable usize-indexed binary searches take at most BITS+1
        // comparisons each. Pay their bounded control/lookup work on the
        // original ledger before searching; this creates no row allocation.
        let mut lookup = self
            .source()
            .budget()
            .lease_for(
                LeaseClass::Fallback,
                WorkRequest {
                    work: 1024 + 128 * (u64::from(usize::BITS) + 1),
                    ..WorkRequest::default()
                },
            )
            .map_err(refusal)?;
        at = self.deadline(Some(lookup.deadline().map_or(at, |d| d.min(at))))?;
        let retained = self
            .0
            .layers
            .get(layer)
            .ok_or_else(|| invalid("prepared retained missing layer"))?;
        if retained.idx.binary_search(&target).is_err()
            || !self.0.resident.shares_ledger(self.0.source.budget())
        {
            return Err(invalid("prepared retained missing original target/ledger"));
        }
        let products = self
            .0
            .source
            .construction()
            .layer_products(layer)
            .ok_or_else(|| invalid("prepared retained source layer absent"))?;
        let (rows, column) = products
            .row(target)
            .ok_or_else(|| invalid("prepared retained source row absent"))?;
        rows.checkpoint(at)?;
        if rows.target(column) != Some(target) {
            return Err(invalid("prepared retained column identity changed"));
        }
        self.validate_source(self.source(), Some(at))?;
        lookup.complete().map_err(refusal)?;
        Ok((rows, column))
    }
}

#[derive(Clone, Copy)]
struct Score {
    score: f64,
    target: usize,
    pos: usize,
}

/// c*(u-l), descending score and ascending original-ID ties, then ascending
/// selected IDs. The global byte budget is strict across first-come layers;
/// it counts the actual two f32 advice packs, not the already-owned source.
pub(in crate::margin_row) fn build_prepared_retained(
    source: &PreparedRootSource,
    cfg: RetainCfg,
    requested: Option<Instant>,
) -> Result<PreparedRetainedRows> {
    let mut at = source.deadline(requested)?;
    source.checkpoint(Some(at))?;
    if !source.construction().coverage_complete(Some(at))? {
        return Err(invalid(
            "prepared retention requires complete requested input rows",
        ));
    }
    let naug = add(source.construction().prefix().root().lo().len(), 1)?;
    let row_bytes = bytes::<f32>(mul(naug, 2)?)?;
    let upper = source.construction().stats().refined_rows;
    let count = source.construction().layers().len();
    let max_rows = upper
        .min(mul(count, cfg.per_layer)?)
        .min(cfg.budget_bytes / row_bytes);
    let persistent = add(
        add(
            shared_owner::<RetainedRowsInner>()?,
            bytes::<PreparedRetainedLayer>(count)?,
        )?,
        add(
            bytes::<usize>(mul(max_rows, 2)?)?,
            mul(max_rows, row_bytes)?,
        )?,
    )?;
    // Sorting and packing overlap the retained payload. Price all of it before
    // the first allocation; loops and sort comparisons poll the original token.
    let score_capacity = if max_rows == 0 { 0 } else { upper };
    let scratch = add(
        bytes::<Score>(score_capacity)?,
        add(bytes::<RowLocation>(max_rows)?, owner::<Sha256>()?)?,
    )?;
    let sort_work = work(score_capacity, mul(usize::BITS as usize + 1, 64)?)?
        .checked_add(work(count, 128)?)
        .ok_or_else(overflow)?;
    let pack_work = work(mul(max_rows, naug)?, 96)?;
    let mut lease = admit(
        source,
        persistent,
        scratch,
        sort_work.checked_add(pack_work).ok_or_else(overflow)?,
        at,
    )?;
    at = min_lease(at, &lease);
    let frame = source.construction().prefix().frame(Some(at))?;
    let mut scored = exact_vec::<Score>(score_capacity)?;
    let mut selected_locations = exact_vec::<RowLocation>(max_rows)?;
    let mut layers = exact_vec::<PreparedRetainedLayer>(count)?;
    let mut payload_bytes = 0;
    for products in source.construction().layers() {
        frame.checkpoint()?;
        let gate = frame
            .gate(products.layer())
            .ok_or_else(|| invalid("prepared retention missing final gate"))?;
        scored.clear();
        let available = (cfg.budget_bytes - payload_bytes) / row_bytes;
        let take = cfg.per_layer.min(gate.unst.len()).min(available);
        if take > 0 {
            for (pos, &target) in gate.unst.iter().enumerate() {
                poll(source, at, pos)?;
                let score = gate.c[target] * (gate.u[target] - gate.l[target]);
                if score.is_nan() || score < 0.0 || scored.len() == upper {
                    return Err(invalid("prepared retention invalid slack score/capacity"));
                }
                scored.push(Score { score, target, pos });
            }
            sort_paid(&mut scored, source, at, |a, b| {
                b.score
                    .total_cmp(&a.score)
                    .then_with(|| a.target.cmp(&b.target))
            })?;
            scored.truncate(take);
            sort_paid(&mut scored, source, at, |a, b| a.target.cmp(&b.target))?;
        }
        let mut layer = PreparedRetainedLayer {
            layer: products.layer(),
            eligible: gate.unst.len(),
            idx: exact_vec(take)?,
            unstable_positions: exact_vec(take)?,
            lower_advice: exact_vec(mul(take, naug)?)?,
            upper_advice: exact_vec(mul(take, naug)?)?,
            naug,
        };
        // Empty/budget-exhausted layers remain explicit with eligible census.
        for (i, selected) in scored.iter().take(take).enumerate() {
            poll(source, at, i)?;
            let (rows, column) = products
                .row(selected.target)
                .ok_or_else(|| invalid("prepared retention absent source row"))?;
            rows.validate_prepared_prefix_with_deadline(
                products.prefix_before(),
                products.op(),
                rows.targets(),
                at,
            )?;
            at = at.min(rows.deadline());
            layer.idx.push(selected.target);
            layer.unstable_positions.push(selected.pos);
            let mapping = products
                .coverage()
                .get(selected.pos)
                .ok_or_else(|| invalid("prepared retention absent ordered source mapping"))?;
            if mapping.target != selected.target || mapping.column != column {
                return Err(invalid(
                    "prepared retention original source mapping changed",
                ));
            }
            selected_locations.push(RowLocation {
                layer: products.layer(),
                target: selected.target,
                unstable_position: selected.pos,
                tile: mapping.tile,
                column,
            });
            append_advice(
                rows.lower(column)
                    .ok_or_else(|| invalid("prepared retention absent lower plane"))?,
                layer.naug(),
                &mut layer.lower_advice,
                source,
                at,
            )?;
            append_advice(
                rows.upper(column)
                    .ok_or_else(|| invalid("prepared retention absent upper plane"))?,
                layer.naug(),
                &mut layer.upper_advice,
                source,
                at,
            )?;
        }
        payload_bytes = add(payload_bytes, layer.bytes())?;
        if payload_bytes > cfg.budget_bytes {
            return Err(invalid("prepared retention global payload budget exceeded"));
        }
        layers.push(layer);
    }
    drop(frame);
    drop(scored);
    let content = content_digest(source, &selected_locations, at)?;
    drop(selected_locations);
    let mut hash = Sha256::new();
    hash.update(b"NY_PREPARED_RANKER_ADVICE_V1");
    hash.update(content);
    put_usize(&mut hash, cfg.per_layer)?;
    put_usize(&mut hash, cfg.budget_bytes)?;
    for layer in &layers {
        put_usize(&mut hash, layer.layer())?;
        put_usize(&mut hash, layer.eligible())?;
        for advice in [&layer.lower_advice, &layer.upper_advice] {
            put_usize(&mut hash, advice.len())?;
            for (i, value) in advice.iter().enumerate() {
                poll(source, at, i)?;
                hash.update(value.to_bits().to_le_bytes());
            }
        }
    }
    let digest = hash.finalize().into();
    source.checkpoint(Some(at))?;
    lease.mark_materialized().map_err(refusal)?;
    let resident = lease.commit_persistent().map_err(refusal)?;
    let result = PreparedRetainedRows(Arc::new(RetainedRowsInner {
        source: source.clone(),
        layers,
        cfg,
        payload_bytes,
        digest,
        deadline: at,
        resident,
    }));
    result.validate_source(source, Some(at))?;
    Ok(result)
}

fn append_advice(
    row: InputRowView<'_>,
    naug: usize,
    output: &mut Vec<f32>,
    source: &PreparedRootSource,
    at: Instant,
) -> Result<()> {
    let start = output.len();
    for i in 0..naug {
        poll(source, at, i)?;
        output.push(0.0);
    }
    for (count, (index, coefficient, _)) in row.entries().enumerate() {
        poll(source, at, count)?;
        output[start + index] = advice_cast(coefficient)?;
    }
    output[start + naug - 1] = advice_cast(row.bias())?;
    Ok(())
}
#[allow(clippy::cast_possible_truncation)]
fn advice_cast(value: f64) -> Result<f32> {
    let result = value as f32;
    if !result.is_finite() {
        return Err(invalid(
            "prepared ranker advice cannot represent finite original coefficient",
        ));
    }
    Ok(result)
}

/// Original branch request. Values are declarations; the clip owner resolves
/// them against its sealed original scope before any halfspace is published.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PreparedSplit {
    /// Original trunk ReLU layer index, as reported by the retained-row owner.
    pub layer: usize,
    /// Original preactivation index within that layer.
    pub target: usize,
    /// The declared closed phase whose necessary input constraint is requested.
    pub side: InputSplitSide,
}

#[derive(Clone)]
pub(in crate::margin_row) struct PreparedHalfspaceBatch(Arc<HalfspaceBatchInner>);

struct HalfspaceBatchInner {
    source: PreparedRootSource,
    splits: Vec<PreparedSplit>,
    halfspaces: Vec<Halfspace>,
    charged_errors: Vec<f64>,
    source_digest: [u8; 32],
    digest: [u8; 32],
    deadline: Instant,
    resident: PersistentReservation,
}
impl PreparedHalfspaceBatch {
    pub(in crate::margin_row) fn same_owner(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
    pub(in crate::margin_row) fn source(&self) -> &PreparedRootSource {
        &self.0.source
    }
    pub(in crate::margin_row) fn splits(&self) -> &[PreparedSplit] {
        &self.0.splits
    }
    pub(in crate::margin_row) fn halfspaces(&self) -> &[Halfspace] {
        &self.0.halfspaces
    }
    pub(in crate::margin_row) fn charged_errors(&self) -> &[f64] {
        &self.0.charged_errors
    }
    pub(in crate::margin_row) fn source_digest(&self) -> &[u8; 32] {
        &self.0.source_digest
    }
    pub(in crate::margin_row) fn digest(&self) -> &[u8; 32] {
        &self.0.digest
    }
    pub(in crate::margin_row) fn resident_bytes(&self) -> usize {
        self.0.resident.bytes()
    }
    pub(in crate::margin_row) fn deadline(&self, requested: Option<Instant>) -> Result<Instant> {
        self.0.source.deadline(Some(
            requested.map_or(self.0.deadline, |d| d.min(self.0.deadline)),
        ))
    }
    pub(in crate::margin_row) fn validate_source(
        &self,
        source: &PreparedRootSource,
        requested: Option<Instant>,
    ) -> Result<()> {
        let at = self.deadline(requested)?;
        self.0.source.checkpoint(Some(at))?;
        if !self.0.source.same_source(source) || !self.0.resident.shares_ledger(source.budget()) {
            return Err(invalid("prepared halfspaces foreign source/ledger"));
        }
        Ok(())
    }
}

fn build_halfspaces(
    clip: &PreparedClipRows,
    requests: &[PreparedSplit],
    requested: Option<Instant>,
) -> Result<PreparedHalfspaceBatch> {
    let source = &clip.0.source;
    let mut at = clip.deadline(requested)?;
    clip.validate_source(source, Some(at))?;
    let n = source.construction().prefix().root().lo().len();
    let payload = add(
        bytes::<PreparedSplit>(requests.len())?,
        add(
            bytes::<Halfspace>(requests.len())?,
            add(
                bytes::<f64>(requests.len())?,
                bytes::<f64>(mul(requests.len(), n)?)?,
            )?,
        )?,
    )?;
    let mut lease = admit(
        source,
        add(shared_owner::<HalfspaceBatchInner>()?, payload)?,
        owner::<Sha256>()?,
        work(mul(requests.len(), add(n, 1)?)?, 128)?,
        at,
    )?;
    at = min_lease(at, &lease);
    let mut splits = exact_vec(requests.len())?;
    let mut halfspaces = exact_vec(requests.len())?;
    let mut charged_errors = exact_vec(requests.len())?;
    let mut hash = Sha256::new();
    hash.update(b"NY_PREPARED_SPLIT_HALFSPACES_V1");
    hash.update(clip.0.digest);
    put_usize(&mut hash, requests.len())?;
    for (i, &split) in requests.iter().enumerate() {
        poll(source, at, i)?;
        let (rows, column) = clip.row(split.layer, split.target, Some(at))?;
        at = at.min(rows.deadline());
        let row = match split.side {
            InputSplitSide::Inactive => rows.lower(column),
            InputSplitSide::Active => rows.upper(column),
        }
        .ok_or_else(|| invalid("prepared halfspace missing independent plane"))?;
        // Producer error is bounded over its ORIGINAL declared input domain.
        // Any later dual/dot arithmetic still needs its own outward consumer.
        let eta = row.pointwise_error(at)?;
        let mut a = exact_vec(n)?;
        for j in 0..n {
            poll(source, at, j)?;
            a.push(0.0);
        }
        for (j, (index, coefficient, _)) in row.entries().enumerate() {
            poll(source, at, j)?;
            a[index] = match split.side {
                InputSplitSide::Inactive => coefficient,
                InputSplitSide::Active => -coefficient,
            };
        }
        let bias = match split.side {
            InputSplitSide::Inactive => row.bias(),
            InputSplitSide::Active => -row.bias(),
        };
        let d = if eta == 0.0 {
            bias
        } else {
            next_down(bias - eta)
        };
        if !d.is_finite() {
            return Err(invalid("prepared halfspace non-finite directed constant"));
        }
        put_usize(&mut hash, split.layer)?;
        put_usize(&mut hash, split.target)?;
        hash.update([match split.side {
            InputSplitSide::Inactive => 0,
            InputSplitSide::Active => 1,
        }]);
        hash_f64(&mut hash, &a, source, at)?;
        hash.update(d.to_bits().to_le_bytes());
        hash.update(eta.to_bits().to_le_bytes());
        splits.push(split);
        halfspaces.push(Halfspace { a, d });
        charged_errors.push(eta);
    }
    source.checkpoint(Some(at))?;
    let digest = hash.finalize().into();
    lease.mark_materialized().map_err(refusal)?;
    let resident = lease.commit_persistent().map_err(refusal)?;
    let result = PreparedHalfspaceBatch(Arc::new(HalfspaceBatchInner {
        source: source.clone(),
        splits,
        halfspaces,
        charged_errors,
        source_digest: clip.0.digest,
        digest,
        deadline: at,
        resident,
    }));
    result.validate_source(source, Some(at))?;
    Ok(result)
}

fn checked_row(
    source: &PreparedRootSource,
    p: RowLocation,
    at: Instant,
) -> Result<(&InputRowPair<'static>, usize)> {
    let layer = source
        .construction()
        .layer_products(p.layer)
        .ok_or_else(|| invalid("prepared row missing original layer"))?;
    let location = layer
        .coverage()
        .get(p.unstable_position)
        .ok_or_else(|| invalid("prepared row unstable position absent"))?;
    let rows = layer
        .tiles()
        .get(p.tile)
        .ok_or_else(|| invalid("prepared row missing original tile"))?
        .rows();
    if location.target != p.target
        || location.tile != p.tile
        || location.column != p.column
        || rows.target(p.column) != Some(p.target)
    {
        return Err(invalid(
            "prepared row exact scope/order/backing mapping changed",
        ));
    }
    rows.checkpoint(at)?;
    // The construction seal authenticated every tile once. Exact private
    // locations above select those same immutable arrays without an O(depth)
    // or full tile validation scan on every subsequent row lookup.
    Ok((rows, p.column))
}

/// Complete content key: model/fold/layout identity, declared domain/policy,
/// final gates, producer prefix/op and ordered effective support with every
/// nominal/error coefficient and constant. Different scopes still require
/// same_source validation even when mathematically identical hashes agree.
fn content_digest(
    source: &PreparedRootSource,
    locations: &[RowLocation],
    mut at: Instant,
) -> Result<[u8; 32]> {
    // Pay the compiler-header census before visiting tensor metadata. It is
    // not postallocation admission and does not copy compiled parameters.
    let net = source.construction().prefix().root().model().net();
    let mut header = admit(
        source,
        0,
        0,
        work(add(net.trunk_relus.len(), locations.len())?, 128)?,
        at,
    )?;
    at = min_lease(at, &header);
    let mut gate_cells = 0;
    for (i, &op) in net.trunk_relus.iter().enumerate() {
        poll(source, at, i)?;
        gate_cells = add(
            gate_cells,
            *net.tsize
                .get(add(op, 1)?)
                .ok_or_else(|| invalid("prepared hash missing original tensor geometry"))?,
        )?;
    }
    let mut target_cells = 0;
    for (i, &location) in locations.iter().enumerate() {
        poll(source, at, i)?;
        let (rows, _) = checked_row(source, location, at)?;
        target_cells = add(target_cells, rows.targets().len())?;
    }
    header.complete().map_err(refusal)?;
    let n = net.n_in;
    let row_work = work(add(mul(locations.len(), add(n, 1)?)?, target_cells)?, 256)?;
    let base_work = work(add(gate_cells, n)?, 256)?;
    let mut lease = admit(
        source,
        0,
        owner::<Sha256>()?,
        row_work.checked_add(base_work).ok_or_else(overflow)?,
        at,
    )?;
    at = min_lease(at, &lease);
    let frame = source.construction().prefix().frame(Some(at))?;
    let root = frame.root();
    let mut hash = Sha256::new();
    hash.update(b"NY_PREPARED_EXPLICIT_INPUT_ROWS_V1");
    hash.update(b"BINARY64_PARAMETERS_WITH_EXPLICIT_FOLD_UNCERTAINTY");
    hash.update(source.construction().identity().source_digest());
    hash.update(source.construction().identity().compiled_digest());
    put_usize(&mut hash, n)?;
    put_usize(&mut hash, frame.layers().len())?;
    let policy = root.policy();
    hash.update([
        u8::from(policy.root_f32),
        u8::from(policy.tableau_f32),
        u8::from(policy.ibp_seed),
    ]);
    hash.update(policy.err_lane.token().as_bytes());
    hash.update([0]);
    hash.update(policy.box_source.token().as_bytes());
    for values in [root.lo(), root.hi(), root.mid(), root.rad(), root.xabs()] {
        hash_f64(&mut hash, values, source, at)?;
    }
    for gate in frame.layers() {
        frame.checkpoint()?;
        put_usize(&mut hash, gate.op)?;
        put_usize(&mut hash, gate.n)?;
        for values in [&gate.l, &gate.u, &gate.alpha, &gate.s, &gate.c, &gate.ms] {
            hash_f64(&mut hash, values, source, at)?;
        }
        put_usize(&mut hash, gate.unst.len())?;
        for (i, &id) in gate.unst.iter().enumerate() {
            poll(source, at, i)?;
            put_usize(&mut hash, id)?;
        }
    }
    put_usize(&mut hash, locations.len())?;
    for (i, &location) in locations.iter().enumerate() {
        poll(source, at, i)?;
        let layer = source
            .construction()
            .layer_products(location.layer)
            .ok_or_else(|| invalid("prepared hash missing original layer"))?;
        let (rows, column) = checked_row(source, location, at)?;
        at = at.min(rows.deadline());
        for value in [
            location.layer,
            layer.op(),
            layer.prefix_before().layer_count(),
            location.target,
            location.unstable_position,
            location.tile,
            location.column,
            rows.columns(),
        ] {
            put_usize(&mut hash, value)?;
        }
        hash.update([match rows.backing() {
            InputRowsBacking::F64 => 64,
            InputRowsBacking::F32 => 32,
        }]);
        put_usize(&mut hash, rows.targets().len())?;
        for (j, &target) in rows.targets().iter().enumerate() {
            poll(source, at, j)?;
            put_usize(&mut hash, target)?;
        }
        for row in [rows.lower(column), rows.upper(column)] {
            let row = row.ok_or_else(|| invalid("prepared hash missing independent row plane"))?;
            hash.update(row.bias().to_bits().to_le_bytes());
            hash.update(row.bias_error().to_bits().to_le_bytes());
            let mut entries = 0usize;
            for (j, (input, coefficient, error)) in row.entries().enumerate() {
                poll(source, at, j)?;
                put_usize(&mut hash, input)?;
                hash.update(coefficient.to_bits().to_le_bytes());
                hash.update(error.to_bits().to_le_bytes());
                entries = add(entries, 1)?;
            }
            put_usize(&mut hash, entries)?;
        }
    }
    source.checkpoint(Some(at))?;
    lease.complete().map_err(refusal)?;
    Ok(hash.finalize().into())
}
fn hash_f64(
    hash: &mut Sha256,
    values: &[f64],
    source: &PreparedRootSource,
    at: Instant,
) -> Result<()> {
    put_usize(hash, values.len())?;
    for (i, &v) in values.iter().enumerate() {
        poll(source, at, i)?;
        hash.update(v.to_bits().to_le_bytes());
    }
    Ok(())
}
fn put_usize(hash: &mut Sha256, value: usize) -> Result<()> {
    hash.update(u64::try_from(value).map_err(|_| overflow())?.to_le_bytes());
    Ok(())
}

/// In-place heapsort with cooperative comparisons; no unpolled library sort
/// or hidden temporary score allocation bypasses the admitted frontier.
fn sort_paid<T>(
    values: &mut [T],
    source: &PreparedRootSource,
    at: Instant,
    cmp: impl Fn(&T, &T) -> Ordering,
) -> Result<()> {
    let mut comparisons = 0;
    for start in (0..values.len() / 2).rev() {
        sift(
            values,
            start,
            values.len(),
            &cmp,
            source,
            at,
            &mut comparisons,
        )?;
    }
    for end in (1..values.len()).rev() {
        values.swap(0, end);
        sift(values, 0, end, &cmp, source, at, &mut comparisons)?;
    }
    source.checkpoint(Some(at))
}
fn sift<T>(
    values: &mut [T],
    mut root: usize,
    end: usize,
    cmp: &impl Fn(&T, &T) -> Ordering,
    source: &PreparedRootSource,
    at: Instant,
    comparisons: &mut usize,
) -> Result<()> {
    while let Some(left) = root
        .checked_mul(2)
        .and_then(|r| r.checked_add(1))
        .filter(|l| *l < end)
    {
        poll(source, at, *comparisons)?;
        *comparisons = comparisons.checked_add(2).ok_or_else(overflow)?;
        let child = if left + 1 < end && cmp(&values[left], &values[left + 1]) == Ordering::Less {
            left + 1
        } else {
            left
        };
        if cmp(&values[root], &values[child]) != Ordering::Less {
            break;
        }
        values.swap(root, child);
        root = child;
    }
    Ok(())
}

fn admit(
    source: &PreparedRootSource,
    persistent: usize,
    scratch: usize,
    work: u64,
    at: Instant,
) -> Result<WorkLease> {
    source.checkpoint(Some(at))?;
    let lease = source
        .budget()
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
    checkpoint(source.budget(), Some(min_lease(at, &lease)))?;
    Ok(lease)
}
fn min_lease(at: Instant, lease: &WorkLease) -> Instant {
    lease.deadline().map_or(at, |d| at.min(d))
}
fn poll(source: &PreparedRootSource, at: Instant, i: usize) -> Result<()> {
    if i.is_multiple_of(POLL) {
        checkpoint(source.budget(), Some(at))?;
    }
    Ok(())
}
fn exact_vec<T>(n: usize) -> Result<Vec<T>> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(n)
        .map_err(|_| invalid("prepared consumer allocation refused"))?;
    if result.capacity() != n {
        return Err(invalid(
            "prepared consumer actual capacity exceeds paid prediction",
        ));
    }
    Ok(result)
}
fn shared_owner<T>() -> Result<usize> {
    // The actual Arc inner, two reference counters and reservation/allocator
    // metadata are admitted before Arc::new. Clone copies one pointer only;
    // every payload and header claim remains charged through its last alias.
    add(owner::<T>()?, mul(2, size_of::<usize>())?)
}
fn owner<T>() -> Result<usize> {
    add(size_of::<T>(), META)
}
fn bytes<T>(n: usize) -> Result<usize> {
    mul(n, size_of::<T>())
}
fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b).ok_or_else(overflow)
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b).ok_or_else(overflow)
}
fn work(n: usize, cost: usize) -> Result<u64> {
    u64::try_from(mul(n, cost)?).map_err(|_| overflow())
}
fn overflow() -> NyError {
    invalid("prepared consumer capacity/work overflow")
}

#[cfg(test)]
#[path = "retained_tests.rs"]
mod tests;
