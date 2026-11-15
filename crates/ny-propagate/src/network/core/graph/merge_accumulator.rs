// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::mem::size_of;
use std::time::Instant;

use ndarray::{Array, Array1, Array2, Dimension, Ix2, Zip};
use ny_tensor::{next_down_f32, next_up_f32};
use tracing::warn;

use crate::bounds::patches::{
    CrownBounds, PatchesMaterializationDeadline, PatchesMaterializationPurpose,
};
use crate::bounds::{LinearBounds, LinearBounds64};
use ny_core::{
    dd::{next_down_f64, next_up_f64, two_sum},
    f32_to_f64_exact, f64_to_f32_down, f64_to_f32_up, require_f64_interval_proof_environment,
    NyError, Result,
};

/// Vec-backed indexed storage for the hot backward loop.
/// When present, all operations use O(1) indexed access instead of HashMap lookups.
struct IndexedStorage {
    name_to_idx: HashMap<String, usize>,
    pending: Vec<Option<CrownBounds>>,
    merged_dense: Vec<Option<LinearBounds64>>,
    /// Reverse map from index to name, for drain().
    idx_to_name: Vec<String>,
}

/// Multiple of the summed dense f32 pairs that a Patches→Dense promotion at a
/// merge point actually holds live at its peak (#conv-crown-residual).
///
/// Counted in `8·rows·cols` pair-equivalents: incoming A pair + incoming
/// certified-error pair + pending A pair + pending error pair + the f64
/// accumulator + the f64 roundoff buffers. See
/// [`CrownMergeAccumulator::guard_patches_dense_promotion`].
const MERGE_PROMOTION_PEAK_MULTIPLE: usize = 4;

/// `CpuMemoryExceeded` site of the hard-authority native Patches merge receipt
/// (#patches-merge-expiry). Distinct from the Dense promotion site so a
/// refusal names the check that actually refused.
pub(crate) const NATIVE_PATCHES_MERGE_SITE: &str =
    "CrownMergeAccumulator native patches merge (hard finite authority)";

/// `[deadline-preserve] patches-merge-graph: saved=N discarded=M` — engagement
/// observable of the hard-authority native merge (#patches-merge-expiry).
/// `saved` counts residual joins that kept the native route under a live hard
/// deadline; `discarded` counts joins refused because the deadline expired or
/// the `NY_PATCHES_FINITE_EXPIRY` lever is disarmed. Observation only.
static PATCHES_MERGE_GRAPH_PRESERVE: super::backward_helpers::DeadlinePreserveCounters =
    super::backward_helpers::DeadlinePreserveCounters::new("patches-merge-graph");

/// Test-only view of the `patches-merge-graph` engagement counters.
#[cfg(test)]
pub(crate) fn patches_merge_graph_preserve_snapshot() -> (u64, u64) {
    PATCHES_MERGE_GRAPH_PRESERVE.snapshot()
}

/// Outcome of [`CrownMergeAccumulator::try_hard_authority_patches_merge`].
enum HardAuthorityPatchesMerge {
    /// The native merge published in place.
    Merged,
    /// No native merge is available (lever disarmed, deadline expired, no
    /// pending Patches carrier, or a pair the operator declines): the legacy
    /// Dense promotion runs exactly as before.
    Declined,
    /// The native merge was available but its transient footprint exceeds the
    /// budget. The guarded Dense promotion is still tried; this refusal is
    /// reported only if that guard refuses on memory too.
    OverBudget(NyError),
}

/// Admission receipt for the temporary Dense zero-coefficient carrier used to
/// accumulate a concretized bias at `NETWORK_INPUT`.
///
/// Keeping this local to the merge accumulator lets us check a pending
/// Patches→Dense promotion before allocating either coefficient matrix.  The
/// stored byte count also turns an allocator refusal into the same structured
/// memory error as an up-front budget refusal.
#[derive(Clone, Copy)]
struct DenseBiasAdmission {
    required_bytes: usize,
    budget_bytes: usize,
}

impl DenseBiasAdmission {
    fn allocation_error(self, site: &'static str) -> NyError {
        NyError::CpuMemoryExceeded {
            required_bytes: self.required_bytes,
            budget_bytes: self.budget_bytes,
            site,
        }
    }

    fn reconcile_capacity(
        self,
        allocated_elements: usize,
        remaining_elements: usize,
        site: &'static str,
    ) -> Result<()> {
        let required_bytes = allocated_elements
            .checked_add(remaining_elements)
            .and_then(|elements| elements.checked_mul(size_of::<f32>()))
            .unwrap_or(usize::MAX);
        if required_bytes > self.budget_bytes {
            return Err(NyError::CpuMemoryExceeded {
                required_bytes,
                budget_bytes: self.budget_bytes,
                site,
            });
        }
        Ok(())
    }
}

/// Unforgeable proof that ONE node is the sole remaining carrier of every
/// output row's dependence on the network input.
///
/// #gpu-suffix-frontier (soundness). The DAG alpha-CROWN backward walk may only
/// be TRUNCATED — a concretized suffix result returned as the final output
/// bound — when no other branch still holds a pending relation. Truncating with
/// a non-empty frontier silently assigns ZERO coefficients and ZERO bias to
/// every output row that reaches the input through an abandoned branch. Those
/// rows concretize to +/-(floating-point residue) and are published as a tiny
/// symmetric CERTIFIED interval: a false proof (see the regression test
/// `gpu_suffix_shortcut_refuses_with_a_pending_sibling_branch`).
///
/// The inner field is private to this module, so this token CANNOT be
/// constructed anywhere else in the workspace. Any present or future shortcut
/// that ends the walk early must obtain it from
/// [`CrownMergeAccumulator::sole_carrier_witness`], which issues it only when
/// the frontier is provably empty. This is the compiler-enforced replacement
/// for "remember to check the frontier".
#[derive(Debug)]
pub(crate) struct SoleCarrierWitness(());

/// Keeps single-parent nodes in their original carrier while merge points
/// accumulate dense bounds in f64 until the node is consumed.
///
/// Supports an optional indexed mode (`new_indexed`) that replaces HashMap
/// lookups with Vec index operations for the graph backward hot loop.
#[derive(Default)]
pub(crate) struct CrownMergeAccumulator {
    pending: HashMap<String, CrownBounds>,
    merged_dense: HashMap<String, LinearBounds64>,
    /// When Some, all operations use Vec-backed indexed storage.
    indexed: Option<IndexedStorage>,
}

impl CrownMergeAccumulator {
    #[inline]
    fn check_deadline(deadline: Option<Instant>, phase: &'static str) -> Result<()> {
        if deadline.is_some_and(|limit| Instant::now() >= limit) {
            Err(NyError::DeadlineExceeded(format!(
                "CrownMergeAccumulator: deadline exceeded {phase}"
            )))
        } else {
            Ok(())
        }
    }

    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Create an indexed accumulator for the graph backward hot loop.
    ///
    /// All node names in `exec_order` plus `NETWORK_INPUT` get assigned
    /// sequential indices. Operations use O(1) Vec access instead of
    /// HashMap lookups.
    pub(crate) fn new_indexed(exec_order: &[String]) -> Self {
        use super::NETWORK_INPUT;
        let capacity = exec_order.len() + 1; // +1 for NETWORK_INPUT
        let mut name_to_idx = HashMap::with_capacity(capacity);
        let mut idx_to_name = Vec::with_capacity(capacity);

        for (i, name) in exec_order.iter().enumerate() {
            name_to_idx.insert(name.clone(), i);
            idx_to_name.push(name.clone());
        }
        let ni_idx = exec_order.len();
        name_to_idx.insert(NETWORK_INPUT.to_string(), ni_idx);
        idx_to_name.push(NETWORK_INPUT.to_string());

        Self {
            pending: HashMap::new(),
            merged_dense: HashMap::new(),
            indexed: Some(IndexedStorage {
                name_to_idx,
                pending: vec![None; capacity],
                merged_dense: vec![None; capacity],
                idx_to_name,
            }),
        }
    }

    pub(crate) fn insert(&mut self, key: String, bounds: CrownBounds) {
        if let Some(ref mut idx_store) = self.indexed {
            if let Some(&i) = idx_store.name_to_idx.get(&key) {
                debug_assert!(
                    idx_store.pending[i].is_none() && idx_store.merged_dense[i].is_none(),
                    "duplicate CrownMergeAccumulator insert for key {key}",
                );
                idx_store.pending[i] = Some(bounds);
                return;
            }
            // Fall through to HashMap for keys not in exec_order (shouldn't happen normally)
        }
        debug_assert!(
            !self.pending.contains_key(&key) && !self.merged_dense.contains_key(&key),
            "duplicate CrownMergeAccumulator insert for key {key}",
        );
        self.pending.insert(key, bounds);
    }

    pub(crate) fn contains_key(&self, key: &str) -> bool {
        if let Some(ref idx_store) = self.indexed {
            if let Some(&i) = idx_store.name_to_idx.get(key) {
                return idx_store.pending[i].is_some() || idx_store.merged_dense[i].is_some();
            }
        }
        self.pending.contains_key(key) || self.merged_dense.contains_key(key)
    }

    /// Issue a [`SoleCarrierWitness`] iff truncating the walk here is sound.
    ///
    /// The caller must have ALREADY taken the current node's own entry (the
    /// walk does this via `take_by_idx_with_deadline`), so an empty accumulator
    /// means no sibling branch is still pending. `input_accumulated` covers the
    /// other half: coefficients that already reached NETWORK_INPUT directly are
    /// held outside this accumulator, and a truncating shortcut would drop them
    /// too.
    pub(crate) fn sole_carrier_witness(
        &self,
        input_accumulated: bool,
    ) -> Option<SoleCarrierWitness> {
        (self.is_empty() && !input_accumulated).then_some(SoleCarrierWitness(()))
    }

    pub(crate) fn is_empty(&self) -> bool {
        if let Some(ref idx_store) = self.indexed {
            return idx_store.pending.iter().all(Option::is_none)
                && idx_store.merged_dense.iter().all(Option::is_none)
                && self.pending.is_empty()
                && self.merged_dense.is_empty();
        }
        self.pending.is_empty() && self.merged_dense.is_empty()
    }

    pub(crate) fn has_only_key(&self, key: &str) -> bool {
        if let Some(ref idx_store) = self.indexed {
            if let Some(&i) = idx_store.name_to_idx.get(key) {
                let has_this =
                    idx_store.pending[i].is_some() || idx_store.merged_dense[i].is_some();
                if !has_this {
                    return false;
                }
                let total_indexed = idx_store.pending.iter().filter(|x| x.is_some()).count()
                    + idx_store
                        .merged_dense
                        .iter()
                        .filter(|x| x.is_some())
                        .count();
                let total_hash = self.pending.len() + self.merged_dense.len();
                return total_indexed + total_hash == 1;
            }
        }
        self.pending.len() + self.merged_dense.len() == 1 && self.contains_key(key)
    }

    pub(crate) fn take(&mut self, key: &str) -> Result<Option<CrownBounds>> {
        self.take_with_deadline(key, None)
    }

    /// Transactional finite-authority take.  An f64 sidecar is fully downcast
    /// and checked before its source slot is removed; a Patches/Dense carrier
    /// is moved only after the deadline checkpoint.
    pub(crate) fn take_with_deadline(
        &mut self,
        key: &str,
        deadline: Option<Instant>,
    ) -> Result<Option<CrownBounds>> {
        self.take_with_deadline_and_resident(key, deadline, 0)
    }

    /// Transactional finite-authority take while another request-owned
    /// logical payload remains live. `retained_base_bytes` excludes the
    /// carrier stored under `key`; an f64 sidecar and its staged f32 result are
    /// charged here before the source slot can be removed.
    pub(crate) fn take_with_deadline_and_resident(
        &mut self,
        key: &str,
        deadline: Option<Instant>,
        retained_base_bytes: usize,
    ) -> Result<Option<CrownBounds>> {
        // `None` is the exact legacy policy: external finite-authority
        // receipts must not add a new refusal to the unbounded route.
        let retained_base_bytes = if deadline.is_some() {
            retained_base_bytes
        } else {
            0
        };
        Self::check_deadline(deadline, "before take")?;
        if let Some(ref mut idx_store) = self.indexed {
            if let Some(&i) = idx_store.name_to_idx.get(key) {
                return Self::take_from_vecs_with_deadline(
                    &mut idx_store.pending,
                    &mut idx_store.merged_dense,
                    i,
                    deadline,
                    retained_base_bytes,
                );
            }
        }
        let pending = self.pending.get(key);
        let merged = self.merged_dense.get(key);
        debug_assert!(
            pending.is_none() || merged.is_none(),
            "CrownMergeAccumulator key {key} existed in both stores",
        );
        if pending.is_some() {
            Self::check_deadline(deadline, "before pending take publication")?;
            return Ok(self.pending.remove(key));
        }
        let Some(merged) = merged else {
            return Ok(None);
        };
        let staged = Self::downcast_dense_ref_with_deadline(merged, deadline, retained_base_bytes)?;
        Self::check_deadline(deadline, "after f64 take downcast")?;
        let removed = self.merged_dense.remove(key);
        debug_assert!(removed.is_some());
        Ok(Some(CrownBounds::Dense(staged)))
    }

    /// Direct-index take for the hot loop where the caller already knows the index.
    /// Avoids the name_to_idx HashMap lookup.
    #[cfg(test)]
    #[inline]
    pub(crate) fn take_by_idx(&mut self, idx: usize) -> Result<Option<CrownBounds>> {
        self.take_by_idx_with_deadline(idx, None)
    }

    #[inline]
    pub(crate) fn take_by_idx_with_deadline(
        &mut self,
        idx: usize,
        deadline: Option<Instant>,
    ) -> Result<Option<CrownBounds>> {
        if let Some(ref mut idx_store) = self.indexed {
            return Self::take_from_vecs_with_deadline(
                &mut idx_store.pending,
                &mut idx_store.merged_dense,
                idx,
                deadline,
                0,
            );
        }
        Err(NyError::InvalidSpec(
            "take_by_idx called on non-indexed CrownMergeAccumulator".to_string(),
        ))
    }

    fn take_from_vecs_with_deadline(
        pending: &mut [Option<CrownBounds>],
        merged_dense: &mut [Option<LinearBounds64>],
        i: usize,
        deadline: Option<Instant>,
        retained_base_bytes: usize,
    ) -> Result<Option<CrownBounds>> {
        Self::check_deadline(deadline, "before indexed take")?;
        let p = pending[i].as_ref();
        let m = merged_dense[i].as_ref();
        debug_assert!(
            p.is_none() || m.is_none(),
            "CrownMergeAccumulator indexed slot {i} existed in both stores",
        );
        if p.is_some() {
            Self::check_deadline(deadline, "before indexed pending publication")?;
            return Ok(pending[i].take());
        }
        let Some(m) = m else {
            return Ok(None);
        };
        let staged = Self::downcast_dense_ref_with_deadline(m, deadline, retained_base_bytes)?;
        Self::check_deadline(deadline, "after indexed f64 take downcast")?;
        let removed = merged_dense[i].take();
        debug_assert!(removed.is_some());
        Ok(Some(CrownBounds::Dense(staged)))
    }

    /// `publishes_error_planes` is whether the downcast will emit coefficient
    /// error planes: a carried certificate OR an inexact coefficient cast.
    fn guard_sidecar_downcast_staging(
        bounds: &LinearBounds64,
        retained_base_bytes: usize,
        publishes_error_planes: bool,
    ) -> Result<()> {
        let pair_bytes = Self::linear64_dense_pair_bytes(bounds);
        let matrix_bytes = pair_bytes / 2;
        let output_error_bytes = if publishes_error_planes {
            matrix_bytes.saturating_mul(2)
        } else {
            0
        };
        let output_bytes = pair_bytes
            .saturating_add(output_error_bytes)
            .saturating_add(
                bounds
                    .num_outputs()
                    .saturating_mul(2)
                    .saturating_mul(size_of::<f32>()),
            );
        let required_bytes = retained_base_bytes
            .saturating_add(Self::linear64_memory_bytes(bounds))
            .saturating_add(output_bytes);
        let budget_bytes = crate::network::crown_memory::cpu_crown_dense_budget_bytes();
        if required_bytes > budget_bytes {
            return Err(NyError::CpuMemoryExceeded {
                required_bytes,
                budget_bytes,
                site: "CrownMergeAccumulator f64 take downcast staging",
            });
        }
        Ok(())
    }

    /// Stage a Dense snapshot of the whole frontier without removing any
    /// pending carrier or f64 sidecar.
    ///
    /// This is the transactional half of graph-frontier draining.  Every
    /// Patches materialization observes the same absolute deadline; f64
    /// downcasts are bracketed by checkpoints.  The caller may clear the
    /// source only after all later concretization/merge work succeeds.
    pub(crate) fn snapshot_dense_with_deadline(
        &self,
        deadline: Option<Instant>,
    ) -> Result<Vec<(String, LinearBounds)>> {
        Self::check_deadline(deadline, "before frontier snapshot")?;
        self.guard_frontier_snapshot_budget()?;
        let source_frontier_bytes = self.logical_frontier_payload_bytes();
        let mut staged_payload_bytes = 0usize;

        let count = self.pending.len()
            + self.merged_dense.len()
            + self.indexed.as_ref().map_or(0, |storage| {
                storage
                    .pending
                    .iter()
                    .filter(|entry| entry.is_some())
                    .count()
                    + storage
                        .merged_dense
                        .iter()
                        .filter(|entry| entry.is_some())
                        .count()
            });
        let budget_bytes = crate::network::crown_memory::cpu_crown_dense_budget_bytes();
        let mut staged = Vec::new();
        staged
            .try_reserve_exact(count)
            .map_err(|_| NyError::CpuMemoryExceeded {
                required_bytes: count.saturating_mul(size_of::<(String, LinearBounds)>()),
                budget_bytes,
                site: "CrownMergeAccumulator frontier snapshot entries",
            })?;

        if let Some(ref storage) = self.indexed {
            for (index, bounds) in storage.pending.iter().enumerate() {
                if let Some(bounds) = bounds {
                    Self::stage_snapshot_crown(
                        &mut staged,
                        &mut staged_payload_bytes,
                        source_frontier_bytes,
                        &storage.idx_to_name[index],
                        bounds,
                        deadline,
                    )?;
                }
            }
            for (index, bounds) in storage.merged_dense.iter().enumerate() {
                if let Some(bounds) = bounds {
                    Self::stage_snapshot_sidecar(
                        &mut staged,
                        &mut staged_payload_bytes,
                        source_frontier_bytes,
                        &storage.idx_to_name[index],
                        bounds,
                        deadline,
                    )?;
                }
            }
        }
        for (name, bounds) in &self.pending {
            Self::stage_snapshot_crown(
                &mut staged,
                &mut staged_payload_bytes,
                source_frontier_bytes,
                name,
                bounds,
                deadline,
            )?;
        }
        for (name, bounds) in &self.merged_dense {
            Self::stage_snapshot_sidecar(
                &mut staged,
                &mut staged_payload_bytes,
                source_frontier_bytes,
                name,
                bounds,
                deadline,
            )?;
        }
        Self::check_deadline(deadline, "before frontier snapshot publication")?;
        Ok(staged)
    }

    fn stage_snapshot_crown(
        staged: &mut Vec<(String, LinearBounds)>,
        staged_payload_bytes: &mut usize,
        source_frontier_bytes: usize,
        name: &str,
        bounds: &CrownBounds,
        deadline: Option<Instant>,
    ) -> Result<()> {
        Self::check_deadline(deadline, "before frontier carrier staging")?;
        let retained_base_bytes = source_frontier_bytes
            .saturating_sub(bounds.memory_bytes())
            .saturating_add(*staged_payload_bytes);
        let dense = match bounds {
            CrownBounds::Dense(bounds) => {
                bounds.try_clone_with_deadline(deadline, retained_base_bytes)?
            }
            CrownBounds::Patches(bounds) => bounds
                .to_dense_with_deadline_and_resident_for_purpose(
                    deadline,
                    retained_base_bytes,
                    PatchesMaterializationPurpose::Other,
                )?,
        };
        Self::check_deadline(deadline, "after frontier carrier staging")?;
        *staged_payload_bytes = (*staged_payload_bytes).saturating_add(dense.memory_bytes());
        staged.push((name.to_string(), dense));
        Self::check_deadline(deadline, "after frontier carrier publication")?;
        Ok(())
    }

    fn stage_snapshot_sidecar(
        staged: &mut Vec<(String, LinearBounds)>,
        staged_payload_bytes: &mut usize,
        source_frontier_bytes: usize,
        name: &str,
        bounds: &LinearBounds64,
        deadline: Option<Instant>,
    ) -> Result<()> {
        Self::check_deadline(deadline, "before frontier f64 downcast")?;
        let retained_base_bytes = source_frontier_bytes
            .saturating_sub(Self::linear64_memory_bytes(bounds))
            .saturating_add(*staged_payload_bytes);
        let dense = Self::downcast_dense_ref_with_deadline(bounds, deadline, retained_base_bytes)?;
        Self::check_deadline(deadline, "after frontier f64 downcast")?;
        *staged_payload_bytes = (*staged_payload_bytes).saturating_add(dense.memory_bytes());
        staged.push((name.to_string(), dense));
        Self::check_deadline(deadline, "after frontier f64 publication")?;
        Ok(())
    }

    /// Logical payload retained by the complete pending/merged frontier.
    ///
    /// Callers staging another request-owned carrier under a finite deadline
    /// include this base in that carrier's admission receipt so the clone does
    /// not pretend the live accumulator disappeared during allocation.
    pub(crate) fn logical_frontier_payload_bytes(&self) -> usize {
        let mut bytes = self.pending.values().fold(0usize, |sum, bounds| {
            sum.saturating_add(bounds.memory_bytes())
        });
        bytes = self.merged_dense.values().fold(bytes, |sum, bounds| {
            sum.saturating_add(Self::linear64_memory_bytes(bounds))
        });
        if let Some(ref storage) = self.indexed {
            bytes = storage.pending.iter().flatten().fold(bytes, |sum, bounds| {
                sum.saturating_add(bounds.memory_bytes())
            });
            bytes = storage
                .merged_dense
                .iter()
                .flatten()
                .fold(bytes, |sum, bounds| {
                    sum.saturating_add(Self::linear64_memory_bytes(bounds))
                });
        }
        bytes
    }

    /// Clear a frontier only after a fully staged replacement has been
    /// published by the caller.
    pub(crate) fn clear(&mut self) {
        if let Some(ref mut storage) = self.indexed {
            storage.pending.fill_with(|| None);
            storage.merged_dense.fill_with(|| None);
        }
        self.pending.clear();
        self.merged_dense.clear();
    }

    fn guard_frontier_snapshot_budget(&self) -> Result<()> {
        let mut required_bytes = 0usize;
        let mut max_patches_pair = 0usize;
        if let Some(ref storage) = self.indexed {
            for bounds in storage.pending.iter().flatten() {
                Self::account_snapshot_crown(bounds, &mut required_bytes, &mut max_patches_pair)?;
            }
            for bounds in storage.merged_dense.iter().flatten() {
                Self::account_snapshot_sidecar(bounds, &mut required_bytes);
            }
        }
        for bounds in self.pending.values() {
            Self::account_snapshot_crown(bounds, &mut required_bytes, &mut max_patches_pair)?;
        }
        for bounds in self.merged_dense.values() {
            Self::account_snapshot_sidecar(bounds, &mut required_bytes);
        }

        // The largest pollable Patches conversion may transiently hold its
        // scatter/error workspaces in addition to every retained source and
        // already-staged output above.
        required_bytes = required_bytes
            .saturating_add(max_patches_pair.saturating_mul(MERGE_PROMOTION_PEAK_MULTIPLE));
        let budget_bytes = crate::network::crown_memory::cpu_crown_dense_budget_bytes();
        if required_bytes > budget_bytes {
            return Err(NyError::CpuMemoryExceeded {
                required_bytes,
                budget_bytes,
                site: "CrownMergeAccumulator transactional frontier snapshot",
            });
        }
        Ok(())
    }

    fn account_snapshot_crown(
        bounds: &CrownBounds,
        required_bytes: &mut usize,
        max_patches_pair: &mut usize,
    ) -> Result<()> {
        *required_bytes = required_bytes.saturating_add(bounds.memory_bytes());
        match bounds {
            CrownBounds::Dense(bounds) => {
                *required_bytes = required_bytes.saturating_add(bounds.memory_bytes());
            }
            CrownBounds::Patches(bounds) => {
                let pair = bounds.dense_pair_bytes()?;
                *max_patches_pair = (*max_patches_pair).max(pair);
                *required_bytes = required_bytes
                    .saturating_add(pair.saturating_mul(2))
                    .saturating_add(
                        bounds
                            .row_count
                            .saturating_mul(2)
                            .saturating_mul(size_of::<f32>()),
                    );
            }
        }
        Ok(())
    }

    fn account_snapshot_sidecar(bounds: &LinearBounds64, required_bytes: &mut usize) {
        let pair_bytes = Self::linear64_dense_pair_bytes(bounds);
        let matrix_bytes = pair_bytes / 2;
        let output_error_bytes = if bounds.lower_a_err().is_some() || bounds.upper_a_err().is_some()
        {
            matrix_bytes.saturating_mul(2)
        } else {
            0
        };
        let output_bytes = pair_bytes.saturating_add(output_error_bytes);
        *required_bytes = required_bytes
            .saturating_add(Self::linear64_memory_bytes(bounds))
            .saturating_add(output_bytes)
            .saturating_add(
                bounds
                    .num_outputs()
                    .saturating_mul(2)
                    .saturating_mul(size_of::<f32>()),
            );
    }

    /// Merge a CrownBounds contribution, preserving patches when compatible.
    ///
    /// Policy:
    /// - Patches + Patches (compatible): merge in-place in `pending`
    /// - Patches + Patches (incompatible): convert both to dense, route to f64
    /// - Dense + Dense: route to the checked Dense merge implementation
    /// - Mixed Dense/Patches: convert both to dense, route to f64
    ///
    /// Patches-native residual merge for CNN DAGs.
    #[cfg(test)]
    pub(crate) fn merge_crown(&mut self, key: &str, new_bounds: CrownBounds) -> Result<()> {
        self.merge_crown_with_deadline(key, new_bounds, None)
    }

    /// Deadline-aware merge publication.
    ///
    /// A `Some` deadline is HARD authority here. Hard authority keeps the
    /// native in-place Patches merge only while `NY_PATCHES_FINITE_EXPIRY` is
    /// armed (the shipped default), the deadline has not expired, and the
    /// merge's own transient footprint fits the CROWN dense budget; otherwise
    /// it takes the exact Dense promotion, which is pollable and keeps the
    /// pending carrier borrowed until both the dense conversion and f64
    /// sidecar are complete. Either way a deadline or memory refusal cannot
    /// erase or partially rewrite pending state.
    #[cfg(test)]
    pub(crate) fn merge_crown_with_deadline(
        &mut self,
        key: &str,
        new_bounds: CrownBounds,
        deadline: Option<Instant>,
    ) -> Result<()> {
        self.merge_crown_with_deadline_authority(key, new_bounds, deadline, deadline.is_some())
    }

    /// Merge with an explicit distinction between caller/cap authority and an
    /// internal scheduling timestamp.
    ///
    /// Both deadline classes are checked before admission and remain available
    /// to the cooperative Dense promotion path. An internal collector
    /// timestamp (soft) always keeps the historical native Patches merge: it
    /// must not turn two compatible residual carriers into an otherwise
    /// unnecessary dense allocation. Hard authority keeps it too, decided by
    /// EXPIRY rather than PRESENCE — see
    /// [`Self::try_hard_authority_patches_merge`]. The in-place merge is an
    /// indivisible scheduling unit, so after its single admission check it
    /// publishes the complete result without a post-work expiry error.
    pub(crate) fn merge_crown_with_deadline_authority(
        &mut self,
        key: &str,
        new_bounds: CrownBounds,
        deadline: Option<Instant>,
        deadline_is_hard: bool,
    ) -> Result<()> {
        Self::check_deadline(deadline, "before merge")?;
        let mut native_merge_refusal = None;
        if let CrownBounds::Patches(ref new_pb) = new_bounds {
            if !deadline_is_hard {
                // This legacy native merge has no cooperative polling seam and
                // mutates only when the complete owned replacement arrays are
                // ready. Treat it as one admitted soft scheduling unit: a
                // post-work deadline error would report failure after state had
                // already published and cannot be rolled back.
                if self.try_patches_merge(key, new_pb)? {
                    return Ok(());
                }
            } else {
                match self.try_hard_authority_patches_merge(key, new_pb, deadline)? {
                    HardAuthorityPatchesMerge::Merged => return Ok(()),
                    HardAuthorityPatchesMerge::Declined => {}
                    HardAuthorityPatchesMerge::OverBudget(refusal) => {
                        native_merge_refusal = Some(refusal);
                    }
                }
            }
        }
        // A promotion is required when EITHER carrier is Patches.  In
        // particular, a Patches contribution may arrive first and be followed
        // by Dense; guarding only `new_bounds` left that arrival order able to
        // materialize the pending relation without a receipt.
        //
        // A native merge refused on memory still falls through to this guarded
        // promotion: for 7-D explicit-row carriers (or kernels wider than the
        // input) the Dense pair can be the SMALLER allocation, and refusing
        // outright would regress those joins below the legacy hard route. When
        // both refuse, report the native merge's receipt — it names the route
        // the join should have taken and the budget that refused it.
        if let Err(promotion_refusal) = self.guard_patches_dense_promotion(key, &new_bounds) {
            return Err(match native_merge_refusal {
                Some(native) if promotion_refusal.is_cpu_memory_exceeded() => native,
                _ => promotion_refusal,
            });
        }
        let new_lb = new_bounds
            .into_dense_with_deadline_for_purpose(deadline, PatchesMaterializationPurpose::Other)?;
        Self::check_deadline(deadline, "after incoming dense materialization")?;
        self.merge_dense_after_promotion_guard_with_deadline(key, new_lb, deadline)
    }

    /// Refuse an over-budget Patches→Dense promotion at a DAG merge point
    /// (#conv-crown-residual).
    ///
    /// When [`try_patches_merge`](Self::try_patches_merge) declines — or either
    /// arrival order mixes Dense and Patches — the merge must account for
    /// BOTH carriers' dense `[rows x cols]` pairs before converting either one.
    /// [`merge_dense`](Self::merge_dense) then widens the accumulator to f64,
    /// so several multiples of an f32 pair are live at once.
    ///
    /// Nothing else on this path bounded that allocation. Every other
    /// densification site in the backward walk is guarded — `#patches-row-range`
    /// at the step gate, `guard_dense_materialization_budget` in the patches
    /// step — but a merge point was not, because before residual joins ran in
    /// patches form this fallback only saw carriers that had already paid for a
    /// dense pair. Now that a residual `Add` duplicates a patch relation down
    /// two branches that rejoin, a wide feature map can arrive here in patches
    /// form and ask for a multi-GiB promotion.
    ///
    /// Returning the structured `CpuMemoryExceeded` degrades this target to
    /// sound IBP (every caller maps it that way) instead of taking the process
    /// down. Under-budget merges are unchanged.
    ///
    /// The estimate deliberately models the TRANSIENT PEAK, not one pair. For a
    /// `[rows x cols]` promotion the following are live simultaneously, each an
    /// `8·rows·cols` f32-pair equivalent: the incoming carrier's dense A pair
    /// and its certified-error pair (`to_dense` emits the error pair whenever
    /// the carrier has one, which a post-merge carrier always does), the same
    /// two for the pending carrier, the f64 accumulator
    /// (`widen_linear64_exact`), and the f64 per-element roundoff buffers
    /// (`accumulate_coeff_array`). [`MERGE_PROMOTION_PEAK_MULTIPLE`] captures
    /// that. Charging one pair would under-count by roughly 4x and let exactly
    /// the allocation this guard exists to stop through.
    fn guard_patches_dense_promotion(&self, key: &str, incoming: &CrownBounds) -> Result<()> {
        let pending = self.peek_pending(key);
        if !incoming.is_patches() && !pending.is_some_and(CrownBounds::is_patches) {
            return Ok(());
        }

        let incoming_pair = Self::crown_dense_pair_bytes(incoming)?;
        let (existing_pair, retained_existing) = match pending {
            Some(bounds) => (Self::crown_dense_pair_bytes(bounds)?, bounds.memory_bytes()),
            None => self.peek_merged_dense(key).map_or((0, 0), |bounds| {
                (
                    Self::linear64_dense_pair_bytes(bounds),
                    Self::linear64_memory_bytes(bounds),
                )
            }),
        };
        let retained_sources = incoming.memory_bytes().saturating_add(retained_existing);
        self.guard_dense_promotion_receipts(incoming_pair, existing_pair, retained_sources)
    }

    /// Direct test-only Dense merges bypass the Crown merge, so independently
    /// guard the only promotion that can occur there: a pending Patches
    /// carrier materialized beside the incoming Dense pair.
    #[cfg(test)]
    fn guard_dense_against_pending_patches(
        &self,
        key: &str,
        incoming: &LinearBounds,
    ) -> Result<()> {
        self.guard_dense_pair_against_pending_patches(
            key,
            Self::linear_dense_pair_bytes(incoming),
            incoming.memory_bytes(),
        )
    }

    fn guard_dense_pair_against_pending_patches(
        &self,
        key: &str,
        incoming_pair: usize,
        incoming_live_bytes: usize,
    ) -> Result<()> {
        let Some(existing @ CrownBounds::Patches(_)) = self.peek_pending(key) else {
            return Ok(());
        };
        let pending_pair = Self::crown_dense_pair_bytes(existing)?;
        self.guard_dense_promotion_receipts(
            incoming_pair,
            pending_pair,
            incoming_live_bytes.saturating_add(existing.memory_bytes()),
        )
    }

    fn crown_dense_pair_bytes(bounds: &CrownBounds) -> Result<usize> {
        match bounds {
            CrownBounds::Dense(bounds) => Ok(Self::linear_dense_pair_bytes(bounds)),
            CrownBounds::Patches(bounds) => bounds.dense_pair_bytes(),
        }
    }

    fn linear_dense_pair_bytes(bounds: &LinearBounds) -> usize {
        crate::network::crown_memory::dense_pair_bytes(bounds.num_outputs(), bounds.num_inputs())
            .unwrap_or(usize::MAX)
    }

    fn linear64_dense_pair_bytes(bounds: &LinearBounds64) -> usize {
        crate::network::crown_memory::dense_pair_bytes(bounds.num_outputs(), bounds.num_inputs())
            .unwrap_or(usize::MAX)
    }

    fn linear64_memory_bytes(bounds: &LinearBounds64) -> usize {
        [
            bounds.lower_a().len(),
            bounds.upper_a().len(),
            bounds.lower_b().len(),
            bounds.upper_b().len(),
            bounds.lower_a_err().map_or(0, Array2::len),
            bounds.upper_a_err().map_or(0, Array2::len),
        ]
        .into_iter()
        .fold(0usize, usize::saturating_add)
        .saturating_mul(size_of::<f64>())
    }

    /// Every allocation keeps the entire nominal output priced. Allocator
    /// overage is added separately: extra capacity in an early plane must not
    /// subtract bytes from the later planes that still need to be allocated.
    fn staging_elements_after_capacity(
        nominal_elements: usize,
        capacity_overage_elements: &mut usize,
        requested_elements: usize,
        actual_capacity_elements: usize,
    ) -> usize {
        *capacity_overage_elements = (*capacity_overage_elements)
            .saturating_add(actual_capacity_elements.saturating_sub(requested_elements));
        nominal_elements.saturating_add(*capacity_overage_elements)
    }

    fn try_collect_f64_with_deadline(
        values: impl Iterator<Item = f64>,
        len: usize,
        deadline: &mut PatchesMaterializationDeadline,
        base_bytes: usize,
        total_output_elements: usize,
        capacity_overage_elements: &mut usize,
        site: &'static str,
    ) -> Result<Vec<f64>> {
        deadline.checkpoint(site)?;
        let budget_bytes = crate::network::crown_memory::cpu_crown_dense_budget_bytes();
        let mut output = Vec::new();
        output
            .try_reserve_exact(len)
            .map_err(|_| NyError::CpuMemoryExceeded {
                required_bytes: base_bytes
                    .saturating_add(total_output_elements.saturating_mul(size_of::<f64>())),
                budget_bytes,
                site,
            })?;
        let staged_elements = Self::staging_elements_after_capacity(
            total_output_elements,
            capacity_overage_elements,
            len,
            output.capacity(),
        );
        let required_bytes =
            base_bytes.saturating_add(staged_elements.saturating_mul(size_of::<f64>()));
        if required_bytes > budget_bytes {
            return Err(NyError::CpuMemoryExceeded {
                required_bytes,
                budget_bytes,
                site,
            });
        }
        for value in values {
            deadline.work(1, site)?;
            output.push(value);
        }
        deadline.checkpoint(site)?;
        Ok(output)
    }

    /// Decode the binary32 representation so a subnormal operand cannot be
    /// erased by a hardware widening conversion. The unbounded route keeps its
    /// existing allocation policy; the finite route uses the paid collector.
    fn widen_linear64_exact(bounds: &LinearBounds) -> LinearBounds64 {
        LinearBounds64 {
            lower_a: bounds.lower_a().mapv(f32_to_f64_exact),
            lower_b: bounds.lower_b().mapv(f32_to_f64_exact),
            upper_a: bounds.upper_a().mapv(f32_to_f64_exact),
            upper_b: bounds.upper_b().mapv(f32_to_f64_exact),
            lower_a_err: bounds.lower_a_err().map(|e| e.mapv(f32_to_f64_exact)),
            upper_a_err: bounds.upper_a_err().map(|e| e.mapv(f32_to_f64_exact)),
        }
    }

    fn try_widen_linear64_with_deadline(
        bounds: &LinearBounds,
        deadline: Option<Instant>,
        retained_base_bytes: usize,
    ) -> Result<LinearBounds64> {
        const SITE: &str = "CrownMergeAccumulator finite f32->f64 sidecar staging";
        let source_bytes = bounds.memory_bytes();
        let total_output_elements = source_bytes / size_of::<f32>();
        let base_bytes = retained_base_bytes.saturating_add(source_bytes);
        let required_bytes =
            base_bytes.saturating_add(total_output_elements.saturating_mul(size_of::<f64>()));
        let budget_bytes = crate::network::crown_memory::cpu_crown_dense_budget_bytes();
        if required_bytes > budget_bytes {
            return Err(NyError::CpuMemoryExceeded {
                required_bytes,
                budget_bytes,
                site: SITE,
            });
        }

        let mut poll = PatchesMaterializationDeadline::new(deadline);
        let mut capacity_overage = 0usize;
        let lower_a = Self::try_collect_f64_with_deadline(
            bounds
                .lower_a()
                .iter()
                .map(|&value| f32_to_f64_exact(value)),
            bounds.lower_a().len(),
            &mut poll,
            base_bytes,
            total_output_elements,
            &mut capacity_overage,
            SITE,
        )?;
        let lower_b = Self::try_collect_f64_with_deadline(
            bounds
                .lower_b()
                .iter()
                .map(|&value| f32_to_f64_exact(value)),
            bounds.lower_b().len(),
            &mut poll,
            base_bytes,
            total_output_elements,
            &mut capacity_overage,
            SITE,
        )?;
        let upper_a = Self::try_collect_f64_with_deadline(
            bounds
                .upper_a()
                .iter()
                .map(|&value| f32_to_f64_exact(value)),
            bounds.upper_a().len(),
            &mut poll,
            base_bytes,
            total_output_elements,
            &mut capacity_overage,
            SITE,
        )?;
        let upper_b = Self::try_collect_f64_with_deadline(
            bounds
                .upper_b()
                .iter()
                .map(|&value| f32_to_f64_exact(value)),
            bounds.upper_b().len(),
            &mut poll,
            base_bytes,
            total_output_elements,
            &mut capacity_overage,
            SITE,
        )?;
        let lower_a_err = bounds
            .lower_a_err()
            .map(|source| {
                Self::try_collect_f64_with_deadline(
                    source.iter().map(|&value| f32_to_f64_exact(value)),
                    source.len(),
                    &mut poll,
                    base_bytes,
                    total_output_elements,
                    &mut capacity_overage,
                    SITE,
                )
            })
            .transpose()?;
        let upper_a_err = bounds
            .upper_a_err()
            .map(|source| {
                Self::try_collect_f64_with_deadline(
                    source.iter().map(|&value| f32_to_f64_exact(value)),
                    source.len(),
                    &mut poll,
                    base_bytes,
                    total_output_elements,
                    &mut capacity_overage,
                    SITE,
                )
            })
            .transpose()?;
        poll.checkpoint(SITE)?;
        Ok(LinearBounds64 {
            lower_a: Array2::from_shape_vec(bounds.lower_a().raw_dim(), lower_a)
                .map_err(|error| NyError::InternalError(format!("{SITE}: {error}")))?,
            lower_b: Array1::from_vec(lower_b),
            upper_a: Array2::from_shape_vec(bounds.upper_a().raw_dim(), upper_a)
                .map_err(|error| NyError::InternalError(format!("{SITE}: {error}")))?,
            upper_b: Array1::from_vec(upper_b),
            lower_a_err: lower_a_err
                .map(|values| Array2::from_shape_vec(bounds.lower_a().raw_dim(), values))
                .transpose()
                .map_err(|error| NyError::InternalError(format!("{SITE}: {error}")))?,
            upper_a_err: upper_a_err
                .map(|values| Array2::from_shape_vec(bounds.upper_a().raw_dim(), values))
                .transpose()
                .map_err(|error| NyError::InternalError(format!("{SITE}: {error}")))?,
        })
    }

    fn try_clone_linear64_with_deadline(
        bounds: &LinearBounds64,
        deadline: Option<Instant>,
        retained_base_bytes: usize,
    ) -> Result<LinearBounds64> {
        const SITE: &str = "CrownMergeAccumulator finite f64 sidecar clone";
        let source_bytes = Self::linear64_memory_bytes(bounds);
        let total_output_elements = source_bytes / size_of::<f64>();
        let base_bytes = retained_base_bytes.saturating_add(source_bytes);
        let required_bytes = base_bytes.saturating_add(source_bytes);
        let budget_bytes = crate::network::crown_memory::cpu_crown_dense_budget_bytes();
        if required_bytes > budget_bytes {
            return Err(NyError::CpuMemoryExceeded {
                required_bytes,
                budget_bytes,
                site: SITE,
            });
        }
        let mut poll = PatchesMaterializationDeadline::new(deadline);
        let mut capacity_overage = 0usize;
        let lower_a = Self::try_collect_f64_with_deadline(
            bounds.lower_a().iter().copied(),
            bounds.lower_a().len(),
            &mut poll,
            base_bytes,
            total_output_elements,
            &mut capacity_overage,
            SITE,
        )?;
        let lower_b = Self::try_collect_f64_with_deadline(
            bounds.lower_b().iter().copied(),
            bounds.lower_b().len(),
            &mut poll,
            base_bytes,
            total_output_elements,
            &mut capacity_overage,
            SITE,
        )?;
        let upper_a = Self::try_collect_f64_with_deadline(
            bounds.upper_a().iter().copied(),
            bounds.upper_a().len(),
            &mut poll,
            base_bytes,
            total_output_elements,
            &mut capacity_overage,
            SITE,
        )?;
        let upper_b = Self::try_collect_f64_with_deadline(
            bounds.upper_b().iter().copied(),
            bounds.upper_b().len(),
            &mut poll,
            base_bytes,
            total_output_elements,
            &mut capacity_overage,
            SITE,
        )?;
        let lower_a_err = bounds
            .lower_a_err()
            .map(|source| {
                Self::try_collect_f64_with_deadline(
                    source.iter().copied(),
                    source.len(),
                    &mut poll,
                    base_bytes,
                    total_output_elements,
                    &mut capacity_overage,
                    SITE,
                )
            })
            .transpose()?;
        let upper_a_err = bounds
            .upper_a_err()
            .map(|source| {
                Self::try_collect_f64_with_deadline(
                    source.iter().copied(),
                    source.len(),
                    &mut poll,
                    base_bytes,
                    total_output_elements,
                    &mut capacity_overage,
                    SITE,
                )
            })
            .transpose()?;
        poll.checkpoint(SITE)?;
        Ok(LinearBounds64 {
            lower_a: Array2::from_shape_vec(bounds.lower_a().raw_dim(), lower_a)
                .map_err(|error| NyError::InternalError(format!("{SITE}: {error}")))?,
            lower_b: Array1::from_vec(lower_b),
            upper_a: Array2::from_shape_vec(bounds.upper_a().raw_dim(), upper_a)
                .map_err(|error| NyError::InternalError(format!("{SITE}: {error}")))?,
            upper_b: Array1::from_vec(upper_b),
            lower_a_err: lower_a_err
                .map(|values| Array2::from_shape_vec(bounds.lower_a().raw_dim(), values))
                .transpose()
                .map_err(|error| NyError::InternalError(format!("{SITE}: {error}")))?,
            upper_a_err: upper_a_err
                .map(|values| Array2::from_shape_vec(bounds.upper_a().raw_dim(), values))
                .transpose()
                .map_err(|error| NyError::InternalError(format!("{SITE}: {error}")))?,
        })
    }

    /// Build the f64 sidecar while borrowing the pending carrier.  Patches
    /// conversion is fallible, so keeping ownership in the store until this
    /// returns is the accumulator's atomic-publication boundary.
    fn accumulator_from_pending_with_deadline(
        bounds: &CrownBounds,
        retained_base_bytes: usize,
        deadline: Option<Instant>,
    ) -> Result<LinearBounds64> {
        Self::check_deadline(deadline, "before pending-carrier conversion")?;
        let staged = match bounds {
            CrownBounds::Dense(bounds) => {
                if deadline.is_some() {
                    Self::try_widen_linear64_with_deadline(bounds, deadline, retained_base_bytes)?
                } else {
                    Self::widen_linear64_exact(bounds)
                }
            }
            CrownBounds::Patches(bounds) => {
                if deadline.is_some() {
                    let dense = bounds.to_dense_with_deadline_and_resident_for_purpose(
                        deadline,
                        retained_base_bytes,
                        PatchesMaterializationPurpose::Other,
                    )?;
                    let widen_retained = retained_base_bytes.saturating_add(bounds.memory_bytes());
                    Self::try_widen_linear64_with_deadline(&dense, deadline, widen_retained)?
                } else {
                    let dense = bounds.to_dense_with_deadline_for_purpose(
                        None,
                        PatchesMaterializationPurpose::Other,
                    )?;
                    Self::widen_linear64_exact(&dense)
                }
            }
        };
        Self::check_deadline(deadline, "after f64 sidecar staging")?;
        Ok(staged)
    }

    fn guard_finite_sidecar_staging(
        existing: &LinearBounds64,
        incoming: &LinearBounds,
    ) -> Result<()> {
        let budget_bytes = crate::network::crown_memory::cpu_crown_dense_budget_bytes();
        let coefficient_elements = existing.num_outputs().saturating_mul(existing.num_inputs());
        // Two f64 roundoff matrices remain live while up to two certified-error
        // matrices are allocated/updated. Charge all four workspaces.
        let accumulation_workspace_bytes = coefficient_elements
            .saturating_mul(4)
            .saturating_mul(size_of::<f64>());
        let required_bytes = Self::linear64_memory_bytes(existing)
            .saturating_mul(2)
            .saturating_add(incoming.memory_bytes())
            .saturating_add(accumulation_workspace_bytes);
        if required_bytes > budget_bytes {
            return Err(NyError::CpuMemoryExceeded {
                required_bytes,
                budget_bytes,
                site: "CrownMergeAccumulator finite f64 sidecar staging",
            });
        }
        Ok(())
    }

    /// Receipt the first finite merge sidecar before widening a pending f32 or
    /// Patches carrier. The pending source and incoming Dense contribution stay
    /// live until the completed f64 relation passes its deadline checkpoint.
    fn guard_initial_sidecar_staging(
        existing: &CrownBounds,
        incoming: &LinearBounds,
    ) -> Result<()> {
        let existing_pair = Self::crown_dense_pair_bytes(existing)?;
        let incoming_pair = Self::linear_dense_pair_bytes(incoming);
        let workspace_pair = existing_pair.max(incoming_pair);
        // One f32 coefficient pair occupies the same bytes as one f64 matrix;
        // four matrices cover lower/upper roundoff and lower/upper error work.
        let accumulation_workspace_bytes = workspace_pair.saturating_mul(4);
        let rows = match existing {
            CrownBounds::Dense(bounds) => bounds.num_outputs(),
            CrownBounds::Patches(bounds) => bounds.row_count,
        };
        let bias64_bytes = rows.saturating_mul(2).saturating_mul(size_of::<f64>());
        let (sidecar_bytes, materialized_patches_bytes) = match existing {
            CrownBounds::Dense(bounds) => (bounds.memory_bytes().saturating_mul(2), 0),
            CrownBounds::Patches(_) => (
                // Coefficients plus a worst-case certified-error pair in f64.
                existing_pair.saturating_mul(4).saturating_add(bias64_bytes),
                // The pollable conversion's f32 Dense result remains live while
                // it is widened into the f64 sidecar.
                existing_pair
                    .saturating_mul(2)
                    .saturating_add(rows.saturating_mul(2).saturating_mul(size_of::<f32>())),
            ),
        };
        let required_bytes = existing
            .memory_bytes()
            .saturating_add(incoming.memory_bytes())
            .saturating_add(sidecar_bytes)
            .saturating_add(materialized_patches_bytes)
            .saturating_add(accumulation_workspace_bytes);
        let budget_bytes = crate::network::crown_memory::cpu_crown_dense_budget_bytes();
        if required_bytes > budget_bytes {
            return Err(NyError::CpuMemoryExceeded {
                required_bytes,
                budget_bytes,
                site: "CrownMergeAccumulator initial finite f64 sidecar staging",
            });
        }
        Ok(())
    }

    fn guard_dense_promotion_receipts(
        &self,
        incoming_pair: usize,
        pending_pair: usize,
        retained_source_bytes: usize,
    ) -> Result<()> {
        let budget = crate::network::crown_memory::cpu_crown_dense_budget_bytes();
        let pairs = incoming_pair.saturating_add(pending_pair);
        // Retained structured/Dense sources remain live while both pollable
        // materializations and the f64 accumulator are staged.  The historical
        // pair multiple accounts for the dense/error pairs and f64 roundoff
        // buffers; adding the actual source payload closes the Anchored sidecar
        // peak that pair-only accounting omitted.
        let required = retained_source_bytes
            .saturating_add(pairs.saturating_mul(MERGE_PROMOTION_PEAK_MULTIPLE));
        if required > budget {
            return Err(NyError::CpuMemoryExceeded {
                required_bytes: required,
                budget_bytes: budget,
                site: "CrownMergeAccumulator::merge_crown patches->dense promotion",
            });
        }
        Ok(())
    }

    /// Borrow the pending carrier for `key` without removing it, across both
    /// the indexed and HashMap storage modes.
    fn peek_pending(&self, key: &str) -> Option<&CrownBounds> {
        if let Some(ref idx_store) = self.indexed {
            if let Some(&i) = idx_store.name_to_idx.get(key) {
                return idx_store.pending[i].as_ref();
            }
        }
        self.pending.get(key)
    }

    /// Borrow the f64 merge sidecar for `key` across both storage modes.  A
    /// third or later Patches parent must include this already-live relation in
    /// the same transient-peak receipt as a still-pending carrier.
    fn peek_merged_dense(&self, key: &str) -> Option<&LinearBounds64> {
        if let Some(ref idx_store) = self.indexed {
            if let Some(&i) = idx_store.name_to_idx.get(key) {
                return idx_store.merged_dense[i].as_ref();
            }
        }
        self.merged_dense.get(key)
    }

    fn try_patches_merge(
        &mut self,
        key: &str,
        new_pb: &crate::bounds::patches::PatchesLinearBounds,
    ) -> Result<bool> {
        if let Some(ref mut idx_store) = self.indexed {
            if let Some(&i) = idx_store.name_to_idx.get(key) {
                if let Some(CrownBounds::Patches(ref mut existing)) = idx_store.pending[i] {
                    if existing.try_merge_inplace(new_pb)? {
                        return Ok(true);
                    }
                }
                return Ok(false);
            }
        }
        if let Some(CrownBounds::Patches(ref mut existing)) = self.pending.get_mut(key) {
            if existing.try_merge_inplace(new_pb)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// #patches-merge-expiry: the native in-place Patches merge under HARD
    /// finite authority (audit item N6, `docs/DEADLINE_PRESENCE_AUDIT_2026-08-24.md`).
    ///
    /// `7b0140d09` skipped the native merge whenever the authority was hard,
    /// i.e. on deadline PRESENCE. Every CROWN-IBP collection under a caller
    /// deadline or collector cap is hard, so every residual join promoted both
    /// carriers to Dense f64: on yolo_2023 that asks ~22.5 GB for `Conv_5`
    /// against a ~12 GB budget, and `Conv_5`/`Add_8`/`Conv_12` fall back to
    /// IBP where they had completed CROWN in seconds before the skip.
    ///
    /// The decision is the shared `NY_PATCHES_FINITE_EXPIRY` predicate of the
    /// graph-lane set-mates (`finite_authority_refuses_native_patches`), with
    /// its own `patches-merge-graph` counters: lever off => refuse on presence,
    /// byte-identical to the historical guard; armed (shipped default) =>
    /// refuse only on EXPIRY.
    ///
    /// The invariant the skip protected is kept, restated: the native merge
    /// neither polls a deadline nor charges its allocations (padding copies
    /// and outward-rounded sums allocate infallibly). So under hard authority
    /// its whole transient footprint (`native_merge_peak_bytes`, <= 12x the
    /// largest padded side) is charged against `cpu_crown_dense_budget_bytes()`
    /// before entry, and one deadline check admits it as a single bounded unit
    /// — the same accepted "overrun by one indivisible step" tradeoff as every
    /// other armed set-mate. Nothing is mutated before the merge publishes.
    fn try_hard_authority_patches_merge(
        &mut self,
        key: &str,
        new_pb: &crate::bounds::patches::PatchesLinearBounds,
        deadline: Option<Instant>,
    ) -> Result<HardAuthorityPatchesMerge> {
        // Carrier check FIRST, as at the residual fan-out: the counters then
        // mean "a structured merge was at stake", not "this helper ran".
        let Some(CrownBounds::Patches(existing)) = self.peek_pending(key) else {
            return Ok(HardAuthorityPatchesMerge::Declined);
        };
        if super::backward_helpers::finite_authority_refuses_native_patches(
            true,
            deadline,
            &PATCHES_MERGE_GRAPH_PRESERVE,
        ) {
            return Ok(HardAuthorityPatchesMerge::Declined);
        }
        // A carrier pair the native operator would decline (anchored, sparse,
        // incomparable padding, mismatched metadata) — or one whose preflight
        // cannot authenticate it — keeps the legacy hard route exactly.
        let Ok(Some(required_bytes)) = existing.native_merge_peak_bytes(new_pb) else {
            return Ok(HardAuthorityPatchesMerge::Declined);
        };
        let budget_bytes = crate::network::crown_memory::cpu_crown_dense_budget_bytes();
        if required_bytes > budget_bytes {
            return Ok(HardAuthorityPatchesMerge::OverBudget(
                NyError::CpuMemoryExceeded {
                    required_bytes,
                    budget_bytes,
                    site: NATIVE_PATCHES_MERGE_SITE,
                },
            ));
        }
        Self::check_deadline(deadline, "before hard-authority native Patches merge")?;
        Ok(if self.try_patches_merge(key, new_pb)? {
            HardAuthorityPatchesMerge::Merged
        } else {
            HardAuthorityPatchesMerge::Declined
        })
    }

    /// Fallibly construct the zero-coefficient Dense carrier used to add a
    /// concretized bias at `NETWORK_INPUT`.
    ///
    /// The historical helper allocated two `Array2::zeros` matrices and cloned
    /// both biases before the merge accumulator could inspect a pending
    /// Patches carrier.  This constructor authenticates shapes, receipts the
    /// whole owned carrier, and uses `try_reserve_exact` for every potentially
    /// large buffer.  Capacity rounding is reconciled after each reserve while
    /// the minimum size of all remaining buffers is still included.
    pub(crate) fn try_dense_bias_bounds_with_deadline(
        bias_lower: &Array1<f32>,
        bias_upper: &Array1<f32>,
        output_dim: usize,
        input_dim: usize,
        deadline: Option<Instant>,
    ) -> Result<LinearBounds> {
        const SITE: &str = "CrownMergeAccumulator::try_dense_bias_bounds NETWORK_INPUT carrier";
        let mut poll = PatchesMaterializationDeadline::new(deadline);
        poll.checkpoint(SITE)?;

        if bias_lower.len() != output_dim || bias_upper.len() != output_dim {
            return Err(NyError::InvalidSpec(format!(
                "NETWORK_INPUT bias shape mismatch: expected {output_dim} rows, got lower={} upper={}",
                bias_lower.len(),
                bias_upper.len(),
            )));
        }

        let budget_bytes = crate::network::crown_memory::cpu_crown_dense_budget_bytes();
        let Some(coefficient_elements) = output_dim.checked_mul(input_dim) else {
            return Err(NyError::CpuMemoryExceeded {
                required_bytes: usize::MAX,
                budget_bytes,
                site: SITE,
            });
        };
        let required_elements = coefficient_elements
            .checked_mul(2)
            .and_then(|elements| {
                output_dim
                    .checked_mul(2)
                    .and_then(|bias| elements.checked_add(bias))
            })
            .unwrap_or(usize::MAX);
        let required_bytes = required_elements.saturating_mul(size_of::<f32>());
        let admission = DenseBiasAdmission {
            required_bytes,
            budget_bytes,
        };
        if required_bytes > budget_bytes {
            return Err(admission.allocation_error(SITE));
        }

        let mut lower_a_data = Vec::new();
        lower_a_data
            .try_reserve_exact(coefficient_elements)
            .map_err(|_| admission.allocation_error(SITE))?;
        admission.reconcile_capacity(
            lower_a_data.capacity(),
            coefficient_elements.saturating_add(output_dim.saturating_mul(2)),
            SITE,
        )?;
        for _ in 0..coefficient_elements {
            poll.work(1, SITE)?;
            lower_a_data.push(0.0);
        }

        let mut upper_a_data = Vec::new();
        upper_a_data
            .try_reserve_exact(coefficient_elements)
            .map_err(|_| admission.allocation_error(SITE))?;
        admission.reconcile_capacity(
            lower_a_data
                .capacity()
                .saturating_add(upper_a_data.capacity()),
            output_dim.saturating_mul(2),
            SITE,
        )?;
        for _ in 0..coefficient_elements {
            poll.work(1, SITE)?;
            upper_a_data.push(0.0);
        }

        let mut lower_b_data = Vec::new();
        lower_b_data
            .try_reserve_exact(output_dim)
            .map_err(|_| admission.allocation_error(SITE))?;
        admission.reconcile_capacity(
            lower_a_data
                .capacity()
                .saturating_add(upper_a_data.capacity())
                .saturating_add(lower_b_data.capacity()),
            output_dim,
            SITE,
        )?;
        for &value in bias_lower {
            poll.work(1, SITE)?;
            lower_b_data.push(value);
        }

        let mut upper_b_data = Vec::new();
        upper_b_data
            .try_reserve_exact(output_dim)
            .map_err(|_| admission.allocation_error(SITE))?;
        admission.reconcile_capacity(
            lower_a_data
                .capacity()
                .saturating_add(upper_a_data.capacity())
                .saturating_add(lower_b_data.capacity())
                .saturating_add(upper_b_data.capacity()),
            0,
            SITE,
        )?;
        for &value in bias_upper {
            poll.work(1, SITE)?;
            upper_b_data.push(value);
        }

        // Match `LinearBounds::new_or_conservative`: a NaN in either bias
        // degrades the whole relation instead of publishing NaN.  Coefficients
        // are known-finite zeros, so this in-place firewall needs no extra
        // allocation.
        let mut has_nan = false;
        for &value in lower_b_data.iter().chain(upper_b_data.iter()) {
            poll.work(1, SITE)?;
            has_nan |= value.is_nan();
        }
        if has_nan {
            for value in &mut lower_b_data {
                poll.work(1, SITE)?;
                *value = f32::NEG_INFINITY;
            }
            for value in &mut upper_b_data {
                poll.work(1, SITE)?;
                *value = f32::INFINITY;
            }
        }

        poll.checkpoint(SITE)?;
        let lower_a =
            Array2::from_shape_vec((output_dim, input_dim), lower_a_data).map_err(|error| {
                NyError::InternalError(format!(
                    "{SITE}: checked lower coefficient shape construction failed: {error}"
                ))
            })?;
        let upper_a =
            Array2::from_shape_vec((output_dim, input_dim), upper_a_data).map_err(|error| {
                NyError::InternalError(format!(
                    "{SITE}: checked upper coefficient shape construction failed: {error}"
                ))
            })?;
        let bounds = LinearBounds::from_prevalidated_parts(
            lower_a,
            Array1::from_vec(lower_b_data),
            upper_a,
            Array1::from_vec(upper_b_data),
        )?;
        poll.checkpoint(SITE)?;
        Ok(bounds)
    }

    /// Merge a concretized bias only after any pending Patches promotion has
    /// paid its receipt.  In particular, a zero-budget refusal occurs before
    /// either Dense coefficient matrix is allocated and leaves the pending
    /// carrier untouched.
    pub(crate) fn merge_dense_bias_with_deadline(
        &mut self,
        key: &str,
        bias_lower: &Array1<f32>,
        bias_upper: &Array1<f32>,
        output_dim: usize,
        input_dim: usize,
        deadline: Option<Instant>,
    ) -> Result<()> {
        Self::check_deadline(deadline, "before NETWORK_INPUT bias merge")?;
        if !self.contains_key(key) {
            return Err(NyError::InvalidSpec(format!(
                "CrownMergeAccumulator bias merge expected existing entry for key {key}",
            )));
        }
        if bias_lower.len() != output_dim || bias_upper.len() != output_dim {
            return Err(NyError::InvalidSpec(format!(
                "NETWORK_INPUT bias shape mismatch: expected {output_dim} rows, got lower={} upper={}",
                bias_lower.len(),
                bias_upper.len(),
            )));
        }

        let incoming_pair = crate::network::crown_memory::dense_pair_bytes(output_dim, input_dim)
            .unwrap_or(usize::MAX);
        let incoming_live_bytes = incoming_pair.saturating_add(
            output_dim
                .saturating_mul(2)
                .saturating_mul(size_of::<f32>()),
        );
        self.guard_dense_pair_against_pending_patches(key, incoming_pair, incoming_live_bytes)?;
        let dense_bias = Self::try_dense_bias_bounds_with_deadline(
            bias_lower, bias_upper, output_dim, input_dim, deadline,
        )?;
        Self::check_deadline(deadline, "after NETWORK_INPUT bias allocation")?;
        self.merge_dense_after_promotion_guard_with_deadline(key, dense_bias, deadline)
    }

    /// Test-only direct Dense entry retained for merge/promotion regressions.
    /// Production callers use the deadline-aware Crown or bias merge paths,
    /// which both enforce the same Patches-promotion receipt before materialization.
    #[cfg(test)]
    pub(crate) fn merge_dense(&mut self, key: &str, new_bounds: LinearBounds) -> Result<()> {
        self.guard_dense_against_pending_patches(key, &new_bounds)?;
        self.merge_dense_after_promotion_guard_with_deadline(key, new_bounds, None)
    }

    /// Merge a Dense contribution after any Patches materialization has paid
    /// its transient-peak receipt.  Pending state remains borrowed until every
    /// fallible carrier conversion succeeds; only the final publication moves
    /// the key from `pending` to `merged_dense`.
    fn merge_dense_after_promotion_guard_with_deadline(
        &mut self,
        key: &str,
        new_bounds: LinearBounds,
        deadline: Option<Instant>,
    ) -> Result<()> {
        Self::check_deadline(deadline, "before f64 accumulation")?;
        Self::require_scalar_merge_environment()?;
        if let Some(ref mut idx_store) = self.indexed {
            if let Some(&i) = idx_store.name_to_idx.get(key) {
                if let Some(ref mut existing) = idx_store.merged_dense[i] {
                    if deadline.is_some() {
                        // Publish only after the finite transaction completes.
                        // The staged sidecar keeps the exact current relation
                        // available if the post-accumulation checkpoint expires.
                        Self::guard_finite_sidecar_staging(existing, &new_bounds)?;
                        let original_bytes = Self::linear64_memory_bytes(existing);
                        let mut staged = Self::try_clone_linear64_with_deadline(
                            existing,
                            deadline,
                            new_bounds.memory_bytes(),
                        )?;
                        let live_base_bytes = original_bytes
                            .saturating_add(Self::linear64_memory_bytes(&staged))
                            .saturating_add(new_bounds.memory_bytes());
                        Self::accumulate_linear_bounds64_with_deadline(
                            &mut staged,
                            &new_bounds,
                            deadline,
                            live_base_bytes,
                        )?;
                        Self::check_deadline(deadline, "after f64 accumulation")?;
                        *existing = staged;
                    } else {
                        Self::accumulate_linear_bounds64(existing, &new_bounds);
                    }
                    return Ok(());
                }
                if let Some(existing_bounds) = idx_store.pending[i].as_ref() {
                    if deadline.is_some() {
                        Self::guard_initial_sidecar_staging(existing_bounds, &new_bounds)?;
                    }
                    let mut accumulator = Self::accumulator_from_pending_with_deadline(
                        existing_bounds,
                        new_bounds.memory_bytes(),
                        deadline,
                    )?;
                    if deadline.is_some() {
                        let live_base_bytes = existing_bounds
                            .memory_bytes()
                            .saturating_add(Self::linear64_memory_bytes(&accumulator))
                            .saturating_add(new_bounds.memory_bytes());
                        Self::accumulate_linear_bounds64_with_deadline(
                            &mut accumulator,
                            &new_bounds,
                            deadline,
                            live_base_bytes,
                        )?;
                    } else {
                        Self::accumulate_linear_bounds64(&mut accumulator, &new_bounds);
                    }
                    Self::check_deadline(deadline, "after staged f64 accumulation")?;
                    // All receipts and fallible conversions completed while the
                    // exact pending carrier was still present.
                    let removed = idx_store.pending[i].take();
                    debug_assert!(removed.is_some());
                    idx_store.merged_dense[i] = Some(accumulator);
                    return Ok(());
                }
                let _ = new_bounds;
                return Err(NyError::InvalidSpec(format!(
                    "CrownMergeAccumulator merge expected existing entry for key {key}",
                )));
            }
        }

        if let Some(existing) = self.merged_dense.get_mut(key) {
            if deadline.is_some() {
                Self::guard_finite_sidecar_staging(existing, &new_bounds)?;
                let original_bytes = Self::linear64_memory_bytes(existing);
                let mut staged = Self::try_clone_linear64_with_deadline(
                    existing,
                    deadline,
                    new_bounds.memory_bytes(),
                )?;
                let live_base_bytes = original_bytes
                    .saturating_add(Self::linear64_memory_bytes(&staged))
                    .saturating_add(new_bounds.memory_bytes());
                Self::accumulate_linear_bounds64_with_deadline(
                    &mut staged,
                    &new_bounds,
                    deadline,
                    live_base_bytes,
                )?;
                Self::check_deadline(deadline, "after f64 accumulation")?;
                *existing = staged;
            } else {
                Self::accumulate_linear_bounds64(existing, &new_bounds);
            }
            return Ok(());
        }

        if let Some(existing_bounds) = self.pending.get(key) {
            if deadline.is_some() {
                Self::guard_initial_sidecar_staging(existing_bounds, &new_bounds)?;
            }
            let mut accumulator = Self::accumulator_from_pending_with_deadline(
                existing_bounds,
                new_bounds.memory_bytes(),
                deadline,
            )?;
            if deadline.is_some() {
                let live_base_bytes = existing_bounds
                    .memory_bytes()
                    .saturating_add(Self::linear64_memory_bytes(&accumulator))
                    .saturating_add(new_bounds.memory_bytes());
                Self::accumulate_linear_bounds64_with_deadline(
                    &mut accumulator,
                    &new_bounds,
                    deadline,
                    live_base_bytes,
                )?;
            } else {
                Self::accumulate_linear_bounds64(&mut accumulator, &new_bounds);
            }
            Self::check_deadline(deadline, "after staged f64 accumulation")?;
            // Publish only after the conversion above succeeds.  Removing the
            // entry before `to_dense` used to lose a Patches receipt on error.
            let removed = self.pending.remove(key);
            debug_assert!(removed.is_some());
            self.merged_dense.insert(key.to_string(), accumulator);
            return Ok(());
        }

        let _ = new_bounds;
        Err(NyError::InvalidSpec(format!(
            "CrownMergeAccumulator merge expected existing entry for key {key}",
        )))
    }

    fn downcast_dense_ref_with_deadline(
        bounds: &LinearBounds64,
        deadline: Option<Instant>,
        retained_base_bytes: usize,
    ) -> Result<LinearBounds> {
        const SITE: &str = "CrownMergeAccumulator f64->f32 downcast staging";
        let carries_err = bounds.lower_a_err().is_some() || bounds.upper_a_err().is_some();
        Self::guard_sidecar_downcast_staging(bounds, retained_base_bytes, carries_err)?;
        let mut deadline_state = PatchesMaterializationDeadline::new(deadline);
        deadline_state.checkpoint(SITE)?;
        Self::require_scalar_merge_environment()?;

        // Read the carried certified coefficient error (f64) BEFORE consuming the
        // relation (#vnncomp-aw-soundness).  The borrowed form is also used by
        // transactional frontier staging, where the original sidecar must stay
        // available until every fallible materialization succeeds.
        let lower_err_f64 = bounds.lower_a_err();
        let upper_err_f64 = bounds.upper_a_err();
        let lower_a = bounds.lower_a();
        let lower_b = bounds.lower_b();
        let upper_a = bounds.upper_a();
        let upper_b = bounds.upper_b();
        // Coefficients multiply SIGNED inputs, so a stored coefficient that
        // differs from its f64 source is not a lower/upper affine bound in
        // either rounding direction (`min(c*l, c*u)` is concave in `c`). Every
        // such gap is paid through the error plane; an exact fan-in sum (every
        // coefficient representable, nothing carried) needs no plane.
        let has_err = carries_err
            || Self::any_coefficient_cast_inexact(lower_a, &mut deadline_state, SITE)?
            || Self::any_coefficient_cast_inexact(upper_a, &mut deadline_state, SITE)?;
        if has_err && !carries_err {
            // Price the two planes the cast gaps now need before allocating.
            Self::guard_sidecar_downcast_staging(bounds, retained_base_bytes, true)?;
        }
        let (num_outputs, num_inputs) = (lower_a.nrows(), lower_a.ncols());
        let coefficient_elements =
            num_outputs
                .checked_mul(num_inputs)
                .ok_or_else(|| NyError::CpuMemoryExceeded {
                    required_bytes: usize::MAX,
                    budget_bytes: crate::network::crown_memory::cpu_crown_dense_budget_bytes(),
                    site: SITE,
                })?;
        let output_elements = coefficient_elements
            .saturating_mul(if has_err { 4 } else { 2 })
            .saturating_add(num_outputs.saturating_mul(2));
        let mut capacity_overage_elements = 0usize;
        let mut reserve = |len: usize| -> Result<Vec<f32>> {
            deadline_state.checkpoint(SITE)?;
            let mut values = Vec::new();
            values
                .try_reserve_exact(len)
                .map_err(|_| NyError::CpuMemoryExceeded {
                    required_bytes: retained_base_bytes
                        .saturating_add(Self::linear64_memory_bytes(bounds))
                        .saturating_add(output_elements.saturating_mul(size_of::<f32>())),
                    budget_bytes: crate::network::crown_memory::cpu_crown_dense_budget_bytes(),
                    site: SITE,
                })?;
            let staged_elements = Self::staging_elements_after_capacity(
                output_elements,
                &mut capacity_overage_elements,
                len,
                values.capacity(),
            );
            let required_bytes = retained_base_bytes
                .saturating_add(Self::linear64_memory_bytes(bounds))
                .saturating_add(staged_elements.saturating_mul(size_of::<f32>()));
            let budget_bytes = crate::network::crown_memory::cpu_crown_dense_budget_bytes();
            if required_bytes > budget_bytes {
                return Err(NyError::CpuMemoryExceeded {
                    required_bytes,
                    budget_bytes,
                    site: SITE,
                });
            }
            deadline_state.checkpoint(SITE)?;
            Ok(values)
        };

        let mut out_lower_a = reserve(coefficient_elements)?;
        let mut out_upper_a = reserve(coefficient_elements)?;
        let mut out_lower_b = reserve(num_outputs)?;
        let mut out_upper_b = reserve(num_outputs)?;
        let mut out_lower_err = if has_err {
            Some(reserve(coefficient_elements)?)
        } else {
            None
        };
        let mut out_upper_err = if has_err {
            Some(reserve(coefficient_elements)?)
        } else {
            None
        };
        for row in 0..num_outputs {
            deadline_state.work(1, SITE)?;
            let row_start = out_lower_a.len();
            let lower_bias = Self::downcast_lower_bias(lower_b[row]);
            let upper_bias = Self::downcast_upper_bias(upper_b[row]);
            let mut row_valid = lower_bias.is_some() && upper_bias.is_some();
            for col in 0..num_inputs {
                deadline_state.work(1, SITE)?;
                let lower_value = Self::downcast_coeff(lower_a[[row, col]]);
                let upper_value = Self::downcast_coeff(upper_a[[row, col]]);
                row_valid &= lower_value.is_some() && upper_value.is_some();
                let stored_lower = lower_value.unwrap_or(0.0);
                let stored_upper = upper_value.unwrap_or(0.0);
                out_lower_a.push(stored_lower);
                out_upper_a.push(stored_upper);
                if let Some(errors) = out_lower_err.as_mut() {
                    let carried = lower_err_f64.map(|e| e[[row, col]]).unwrap_or(0.0);
                    let cast_gap = Self::coefficient_cast_gap_up(stored_lower, lower_a[[row, col]]);
                    errors.push(Self::err_to_f32(Self::add_nonnegative_f64_up(
                        carried, cast_gap,
                    )));
                }
                if let Some(errors) = out_upper_err.as_mut() {
                    let carried = upper_err_f64.map(|e| e[[row, col]]).unwrap_or(0.0);
                    let cast_gap = Self::coefficient_cast_gap_up(stored_upper, upper_a[[row, col]]);
                    errors.push(Self::err_to_f32(Self::add_nonnegative_f64_up(
                        carried, cast_gap,
                    )));
                }
            }

            if !row_valid {
                warn!(
                    row,
                    "CrownMergeAccumulator f64->f32 row downcast failed; returning conservative row"
                );
                out_lower_a[row_start..].fill(0.0);
                out_upper_a[row_start..].fill(0.0);
                if let Some(errors) = out_lower_err.as_mut() {
                    errors[row_start..].fill(f32::INFINITY);
                }
                if let Some(errors) = out_upper_err.as_mut() {
                    errors[row_start..].fill(f32::INFINITY);
                }
                out_lower_b.push(f32::NEG_INFINITY);
                out_upper_b.push(f32::INFINITY);
            } else {
                out_lower_b.push(lower_bias.expect("row validity checked lower bias"));
                out_upper_b.push(upper_bias.expect("row validity checked upper bias"));
            }
        }

        deadline_state.checkpoint(SITE)?;
        let lower_a = Array2::from_shape_vec((num_outputs, num_inputs), out_lower_a)
            .map_err(|error| NyError::InternalError(format!("{SITE}: lower shape: {error}")))?;
        let upper_a = Array2::from_shape_vec((num_outputs, num_inputs), out_upper_a)
            .map_err(|error| NyError::InternalError(format!("{SITE}: upper shape: {error}")))?;
        let lower_err = out_lower_err
            .map(|values| Array2::from_shape_vec((num_outputs, num_inputs), values))
            .transpose()
            .map_err(|error| {
                NyError::InternalError(format!("{SITE}: lower error shape: {error}"))
            })?;
        let upper_err = out_upper_err
            .map(|values| Array2::from_shape_vec((num_outputs, num_inputs), values))
            .transpose()
            .map_err(|error| {
                NyError::InternalError(format!("{SITE}: upper error shape: {error}"))
            })?;
        let dense = LinearBounds::from_prevalidated_parts_with_optional_err(
            lower_a,
            Array1::from_vec(out_lower_b),
            upper_a,
            Array1::from_vec(out_upper_b),
            lower_err,
            upper_err,
        )?;
        deadline_state.checkpoint(SITE)?;
        Ok(dense)
    }

    /// The complete f64 difference is charged, including the subtraction's
    /// residual. Under the checked binary64 environment, TwoSum is exact and
    /// `|hi + lo| <= |hi| + |lo|`; the latter is accumulated upward.
    #[inline]
    fn coefficient_cast_gap_up(stored: f32, source: f64) -> f64 {
        let (gap, residual) = two_sum(f32_to_f64_exact(stored), -source);
        Self::add_nonnegative_f64_up(gap.abs(), residual.abs())
    }

    /// Publish a non-negative f64 error as an outward binary32 value. A
    /// nonzero subnormal radius must not be flushed or consumed as zero in a
    /// later f32 kernel, so its ceiling is at least MIN_POSITIVE. Exact zero
    /// remains zero and does not turn an exact side into an artificial error.
    #[inline]
    fn err_to_f32(e: f64) -> f32 {
        if !e.is_finite() || e < 0.0 {
            return f32::INFINITY;
        }
        f64_to_f32_up(e)
    }

    fn require_scalar_merge_environment() -> Result<()> {
        #[cfg(test)]
        if scalar_cast_tests::environment_refused_for_test() {
            return Err(NyError::SoundnessRefusal(
                "scalar graph merge requires a qualified binary64 environment".into(),
            ));
        }
        require_f64_interval_proof_environment()
    }

    /// Add non-negative error magnitudes with a certified upward result.
    /// `TwoSum` exposes the exact residual: a positive residual means the
    /// rounded sum is below the exact sum and needs one upward step; a
    /// non-positive residual means the rounded sum is already an upper bound.
    #[inline]
    fn add_nonnegative_f64_up(a: f64, b: f64) -> f64 {
        if !a.is_finite() || !b.is_finite() || a < 0.0 || b < 0.0 {
            return f64::INFINITY;
        }
        let (sum, residual) = two_sum(a, b);
        if !sum.is_finite() {
            f64::INFINITY
        } else if residual > 0.0 {
            next_up_f64(sum)
        } else {
            sum
        }
    }

    /// True when publishing a coefficient changes its exact f64 value. This
    /// uses the publication helper itself, so normal rounding AND deliberate
    /// removal of binary32 subnormals are both paid. Invalid sources fail their
    /// row's downcast and yield vacuous biases instead.
    fn any_coefficient_cast_inexact(
        source: &Array2<f64>,
        deadline_state: &mut PatchesMaterializationDeadline,
        site: &'static str,
    ) -> Result<bool> {
        for &value in source {
            deadline_state.work(1, site)?;
            if Self::downcast_coeff(value).is_some_and(|stored| f32_to_f64_exact(stored) != value) {
                return Ok(true);
            }
        }
        deadline_state.checkpoint(site)?;
        Ok(false)
    }

    /// Normal coefficients retain the nearest f32. Tiny nonzero values use
    /// signed zero rather than a binary32-subnormal operand that a later f32
    /// kernel may erase. Every changed value is certified by the full gap in
    /// the coefficient-error plane; exact normal fan-in remains error-free.
    fn downcast_coeff(value: f64) -> Option<f32> {
        if !value.is_finite() {
            return None;
        }
        let magnitude_bits = value.to_bits() & 0x7fff_ffff_ffff_ffff;
        if magnitude_bits < f32_to_f64_exact(f32::MIN_POSITIVE).to_bits() {
            return Some(f32::from_bits(((value.to_bits() >> 63) as u32) << 31));
        }
        let cast = value as f32;
        // The range check and this final representation check jointly ensure
        // a binary32-subnormal operand cannot escape this boundary.
        if cast.is_finite() && (cast.to_bits() & 0x7fff_ffff) < f32::MIN_POSITIVE.to_bits() {
            return Some(f32::from_bits(cast.to_bits() & 0x8000_0000));
        }
        cast.is_finite().then_some(cast)
    }

    /// Biases are additive constants, so a directed step is a certificate.
    /// Keep the historical normal-value step and use normal outward endpoints
    /// near underflow, where a one-subnormal step could be flushed away.
    fn downcast_bias_directed(value: f64, is_lower: bool) -> Option<f32> {
        if !value.is_finite() {
            return None;
        }
        if value.abs() <= f32_to_f64_exact(f32::MIN_POSITIVE) {
            return Some(if is_lower {
                f64_to_f32_down(value)
            } else {
                f64_to_f32_up(value)
            });
        }
        let cast = Self::downcast_coeff(value)?;
        let directed = if is_lower {
            next_down_f32(cast)
        } else {
            next_up_f32(cast)
        };
        // One step from MIN_POSITIVE can itself enter the subnormal range.
        // Widen that representation outward once more to a normal endpoint.
        Some(if is_lower {
            f64_to_f32_down(f32_to_f64_exact(directed))
        } else {
            f64_to_f32_up(f32_to_f64_exact(directed))
        })
    }

    fn downcast_lower_bias(value: f64) -> Option<f32> {
        if value == f64::NEG_INFINITY {
            return Some(f32::NEG_INFINITY);
        }
        Self::downcast_bias_directed(value, true)
    }

    fn downcast_upper_bias(value: f64) -> Option<f32> {
        if value == f64::INFINITY {
            return Some(f32::INFINITY);
        }
        Self::downcast_bias_directed(value, false)
    }

    fn accumulate_linear_bounds64_with_deadline(
        existing: &mut LinearBounds64,
        new_bounds: &LinearBounds,
        deadline: Option<Instant>,
        live_base_bytes: usize,
    ) -> Result<()> {
        const SITE: &str = "CrownMergeAccumulator finite f64 accumulation";
        let mut poll = PatchesMaterializationDeadline::new(deadline);
        poll.checkpoint(SITE)?;
        if existing.num_outputs() != new_bounds.num_outputs()
            || existing.num_inputs() != new_bounds.num_inputs()
            || existing.lower_b().len() != new_bounds.lower_b().len()
            || existing.upper_b().len() != new_bounds.upper_b().len()
        {
            warn!(
                existing_shape = ?existing.lower_a().shape(),
                new_shape = ?new_bounds.lower_a().shape(),
                "CrownMergeAccumulator shape mismatch; widening accumulator to infinities"
            );
            for value in existing.lower_a.iter_mut() {
                poll.work(1, SITE)?;
                *value = f64::NEG_INFINITY;
            }
            for value in existing.lower_b.iter_mut() {
                poll.work(1, SITE)?;
                *value = f64::NEG_INFINITY;
            }
            for value in existing.upper_a.iter_mut() {
                poll.work(1, SITE)?;
                *value = f64::INFINITY;
            }
            for value in existing.upper_b.iter_mut() {
                poll.work(1, SITE)?;
                *value = f64::INFINITY;
            }
            poll.checkpoint(SITE)?;
            return Ok(());
        }

        let n_out = existing.num_outputs();
        let n_in = existing.num_inputs();
        let coefficient_elements = n_out.saturating_mul(n_in);
        let total_workspace_elements = coefficient_elements.saturating_mul(4);
        let mut capacity_overage = 0usize;
        let mut make_workspace = || -> Result<Array2<f64>> {
            let values = Self::try_collect_f64_with_deadline(
                std::iter::repeat_n(0.0, coefficient_elements),
                coefficient_elements,
                &mut poll,
                live_base_bytes,
                total_workspace_elements,
                &mut capacity_overage,
                SITE,
            )?;
            Array2::from_shape_vec((n_out, n_in), values)
                .map_err(|error| NyError::InternalError(format!("{SITE}: {error}")))
        };
        let mut lower_roundoff = make_workspace()?;
        let mut upper_roundoff = make_workspace()?;
        let lower_error_spare = make_workspace()?;
        let upper_error_spare = make_workspace()?;
        for ((existing_value, &new_value), roundoff) in existing
            .lower_a
            .iter_mut()
            .zip(new_bounds.lower_a().iter())
            .zip(lower_roundoff.iter_mut())
        {
            poll.work(1, SITE)?;
            Self::accumulate_coefficient(existing_value, new_value, roundoff);
        }
        for ((existing_value, &new_value), roundoff) in existing
            .upper_a
            .iter_mut()
            .zip(new_bounds.upper_a().iter())
            .zip(upper_roundoff.iter_mut())
        {
            poll.work(1, SITE)?;
            Self::accumulate_coefficient(existing_value, new_value, roundoff);
        }
        Self::accumulate_bias_with_deadline(
            &mut existing.lower_b,
            new_bounds.lower_b(),
            true,
            &mut poll,
            SITE,
        )?;
        Self::accumulate_bias_with_deadline(
            &mut existing.upper_b,
            new_bounds.upper_b(),
            false,
            &mut poll,
            SITE,
        )?;
        Self::accumulate_err_with_deadline(
            &mut existing.lower_a_err,
            new_bounds.lower_a_err(),
            &lower_roundoff,
            lower_error_spare,
            n_out,
            n_in,
            &mut poll,
            SITE,
        )?;
        Self::accumulate_err_with_deadline(
            &mut existing.upper_a_err,
            new_bounds.upper_a_err(),
            &upper_roundoff,
            upper_error_spare,
            n_out,
            n_in,
            &mut poll,
            SITE,
        )?;
        poll.checkpoint(SITE)?;
        Ok(())
    }

    #[inline]
    fn accumulate_coefficient(existing_value: &mut f64, new_value: f32, roundoff: &mut f64) {
        if existing_value.is_nan() || new_value.is_nan() {
            *existing_value = f64::NAN;
            *roundoff = 0.0;
            return;
        }
        let (sum, residual) = two_sum(*existing_value, f32_to_f64_exact(new_value));
        *existing_value = sum;
        *roundoff = if sum.is_finite() { residual.abs() } else { 0.0 };
    }

    fn accumulate_bias_with_deadline(
        existing: &mut Array1<f64>,
        new: &Array1<f32>,
        is_lower: bool,
        poll: &mut PatchesMaterializationDeadline,
        site: &'static str,
    ) -> Result<()> {
        let nan_fallback = if is_lower {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
        for (existing_value, &new_value) in existing.iter_mut().zip(new.iter()) {
            poll.work(1, site)?;
            if existing_value.is_nan() || new_value.is_nan() {
                *existing_value = nan_fallback;
                continue;
            }
            let (sum, residual) = two_sum(*existing_value, f32_to_f64_exact(new_value));
            *existing_value = if sum.is_nan() {
                nan_fallback
            } else if !sum.is_finite() {
                sum
            } else if is_lower && residual < 0.0 {
                next_down_f64(sum)
            } else if !is_lower && residual > 0.0 {
                next_up_f64(sum)
            } else {
                sum
            };
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn accumulate_err_with_deadline(
        existing_err: &mut Option<Array2<f64>>,
        new_err: Option<&Array2<f32>>,
        roundoff: &Array2<f64>,
        mut spare: Array2<f64>,
        n_out: usize,
        n_in: usize,
        poll: &mut PatchesMaterializationDeadline,
        site: &'static str,
    ) -> Result<()> {
        let mut roundoff_has = false;
        for &value in roundoff {
            poll.work(1, site)?;
            roundoff_has |= value != 0.0;
        }
        if existing_err.is_none() && new_err.is_none() && !roundoff_has {
            return Ok(());
        }
        if existing_err
            .as_ref()
            .is_some_and(|error| error.shape() != [n_out, n_in])
            || new_err.is_some_and(|error| error.shape() != [n_out, n_in])
        {
            for value in &mut spare {
                poll.work(1, site)?;
                *value = f64::INFINITY;
            }
            *existing_err = Some(spare);
            return Ok(());
        }
        let accumulator = existing_err.get_or_insert(spare);
        for (value, &roundoff_value) in accumulator.iter_mut().zip(roundoff.iter()) {
            poll.work(1, site)?;
            *value = Self::add_nonnegative_f64_up(*value, roundoff_value);
        }
        if let Some(new_error) = new_err {
            for (value, &new_value) in accumulator.iter_mut().zip(new_error.iter()) {
                poll.work(1, site)?;
                *value = Self::add_nonnegative_f64_up(*value, f32_to_f64_exact(new_value));
            }
        }
        Ok(())
    }

    fn accumulate_linear_bounds64(existing: &mut LinearBounds64, new_bounds: &LinearBounds) {
        if existing.num_outputs() != new_bounds.num_outputs()
            || existing.num_inputs() != new_bounds.num_inputs()
            || existing.lower_b().len() != new_bounds.lower_b().len()
            || existing.upper_b().len() != new_bounds.upper_b().len()
        {
            warn!(
                existing_shape = ?existing.lower_a().shape(),
                new_shape = ?new_bounds.lower_a().shape(),
                existing_lower_bias = existing.lower_b().len(),
                new_lower_bias = new_bounds.lower_b().len(),
                existing_upper_bias = existing.upper_b().len(),
                new_upper_bias = new_bounds.upper_b().len(),
                "CrownMergeAccumulator shape mismatch; widening accumulator to infinities"
            );
            Self::widen_to_infinities(existing);
            return;
        }

        // Accumulate the coefficients in f64, capturing the per-element f64
        // accumulation roundoff so we can fold it into the certified error
        // (#vnncomp-aw-soundness). f32→f64 widening of `new` is exact, so the only
        // new error introduced by `existing + new` is the single f64 add's exact
        // `TwoSum` residual. We add its magnitude OUTWARD into the error
        // accumulator below, alongside the incoming contribution's certified
        // coefficient error — neither may be silently dropped at a DAG merge.
        let lower_roundoff =
            Self::accumulate_coeff_array(existing.lower_a_mut(), new_bounds.lower_a());
        let upper_roundoff =
            Self::accumulate_coeff_array(existing.upper_a_mut(), new_bounds.upper_a());
        Self::accumulate_array(existing.lower_b_mut(), new_bounds.lower_b(), true);
        Self::accumulate_array(existing.upper_b_mut(), new_bounds.upper_b(), false);

        // Carry the certified coefficient error: existing_err + new_err + roundoff.
        // Every non-negative f64 addition is rounded upward here; a final f32 cast
        // cannot by itself cover arbitrarily many small addends lost at a large
        // accumulator magnitude.
        let n_out = existing.num_outputs();
        let n_in = existing.num_inputs();
        Self::accumulate_err(
            &mut existing.lower_a_err,
            new_bounds.lower_a_err(),
            &lower_roundoff,
            n_out,
            n_in,
        );
        Self::accumulate_err(
            &mut existing.upper_a_err,
            new_bounds.upper_a_err(),
            &upper_roundoff,
            n_out,
            n_in,
        );
    }

    /// Accumulate `existing += new` (f32→f64 exact widening) element-wise with the
    /// NaN→±inf firewall, returning the exact magnitude of the binary64 add
    /// residual from `TwoSum` (0 where the result is non-finite — those rows
    /// degrade via the bias/err). This is the coefficient-array analogue of
    /// [`accumulate_array`] that additionally reports the introduced roundoff so
    /// it can be folded into the certified coefficient error.
    fn accumulate_coeff_array(
        existing: &mut Array<f64, Ix2>,
        new: &Array<f32, Ix2>,
    ) -> Array<f64, Ix2> {
        let mut roundoff = Array::<f64, _>::zeros(existing.raw_dim());
        Zip::from(existing).and(new).and(&mut roundoff).for_each(
            |existing_value, &new_value, ro| {
                if existing_value.is_nan() || new_value.is_nan() {
                    // A NaN coefficient is never sound; widen the accumulator
                    // coefficient and let concretize degrade the row.
                    *existing_value = f64::NAN;
                    *ro = 0.0;
                    return;
                }
                let (sum, residual) = two_sum(*existing_value, f32_to_f64_exact(new_value));
                if sum.is_finite() {
                    // `sum + residual` is the exact real addition, including
                    // cancellation and subnormal cases.
                    *ro = residual.abs();
                    *existing_value = sum;
                } else {
                    // Inf coefficient: row will degrade at concretize; no finite
                    // roundoff to carry.
                    *ro = 0.0;
                    *existing_value = sum;
                }
            },
        );
        roundoff
    }

    /// Accumulate the certified coefficient error in f64:
    /// `existing_err += new_err + roundoff`. `new_err` is the f32 incoming error
    /// (exact f32→f64 widening); `roundoff` is the f64 add roundoff bound from
    /// [`accumulate_coeff_array`]. The result is allocated lazily: if there is no
    /// error to carry (both sides None and roundoff all-zero) `existing_err` stays
    /// `None` (exact). All entries stay non-negative; a non-finite entry marks the
    /// row for degradation at downcast/concretize.
    fn accumulate_err(
        existing_err: &mut Option<Array2<f64>>,
        new_err: Option<&Array2<f32>>,
        roundoff: &Array<f64, Ix2>,
        n_out: usize,
        n_in: usize,
    ) {
        let new_has = new_err.is_some();
        let roundoff_has = roundoff.iter().any(|&v| v != 0.0);
        if existing_err.is_none() && !new_has && !roundoff_has {
            // Nothing to carry; keep exact.
            return;
        }
        let acc = existing_err.get_or_insert_with(|| Array2::<f64>::zeros((n_out, n_in)));
        if acc.shape() != [n_out, n_in] {
            // Shape drift (should not happen): degrade to a fully-degraded error so
            // concretize widens every row rather than under-counting.
            *acc = Array2::<f64>::from_elem((n_out, n_in), f64::INFINITY);
            return;
        }
        Zip::from(acc).and(roundoff).for_each(|a, &ro| {
            *a = Self::add_nonnegative_f64_up(*a, ro);
        });
        if let Some(ne) = new_err {
            if ne.shape() == [n_out, n_in] {
                let acc = existing_err.as_mut().expect("allocated above");
                Zip::from(acc).and(ne).for_each(|a, &e| {
                    *a = Self::add_nonnegative_f64_up(*a, f32_to_f64_exact(e));
                });
            } else {
                // Incoming error shape mismatch: cannot map it soundly; degrade.
                *existing_err = Some(Array2::<f64>::from_elem((n_out, n_in), f64::INFINITY));
            }
        }
    }

    fn widen_to_infinities(existing: &mut LinearBounds64) {
        *existing.lower_a_mut() = Array::from_elem(existing.lower_a().raw_dim(), f64::NEG_INFINITY);
        *existing.lower_b_mut() = Array::from_elem(existing.lower_b().raw_dim(), f64::NEG_INFINITY);
        *existing.upper_a_mut() = Array::from_elem(existing.upper_a().raw_dim(), f64::INFINITY);
        *existing.upper_b_mut() = Array::from_elem(existing.upper_b().raw_dim(), f64::INFINITY);
    }

    fn accumulate_array<D: Dimension>(
        existing: &mut Array<f64, D>,
        new: &Array<f32, D>,
        is_lower: bool,
    ) {
        let nan_fallback = if is_lower {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
        Zip::from(existing)
            .and(new)
            .for_each(|existing_value, &new_value| {
                if existing_value.is_nan() || new_value.is_nan() {
                    *existing_value = nan_fallback;
                    return;
                }
                let (sum, residual) = two_sum(*existing_value, f32_to_f64_exact(new_value));
                *existing_value = if sum.is_nan() {
                    nan_fallback
                } else if !sum.is_finite() {
                    sum
                } else if is_lower && residual < 0.0 {
                    next_down_f64(sum)
                } else if !is_lower && residual > 0.0 {
                    next_up_f64(sum)
                } else {
                    sum
                };
            });
    }
}

#[cfg(test)]
#[path = "merge_accumulator_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "merge_accumulator_scalar_cast_tests.rs"]
mod scalar_cast_tests;
