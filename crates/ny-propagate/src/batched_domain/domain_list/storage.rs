// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Domain list storage operations: construction, pick_out, and add.
//!
//! Implements `DomainList::{new, len, is_empty, pick_out, pick_out_batched, add}` —
//! the core CRUD operations for the branch-and-bound domain queue.

use super::super::options::BatchedDomainOptions;
use super::alpha_queue::{allocate_graph_local_queue_identity, packed_graph_alpha_queue_enabled};
use super::filter::filter_batch;
use super::grouped::{
    GroupedDisjunctiveLayout, GroupedDisjunctiveStorage, GroupedDomainId, GroupedLeaseState,
    PackedGroupedBounds, PickedGroupedDomains, SealedGroupedQueueEntry,
};
use super::picked::PickedDomains;
use super::processed::ProcessedDomains;
use super::types::{DomainListConfig, DomainMetadata};
use super::DomainList;
use ndarray::{ArrayD, IxDyn};
use ny_core::{NyError, Result};
use ny_tensor::{create_tensor_storage, TreeTraversal};
use std::collections::HashMap;
use std::sync::Arc;

impl DomainList {
    fn restore_popped_storage(
        storage: &mut (dyn ny_tensor::TensorStorage + Send),
        popped: &ArrayD<f32>,
        remaining_len: usize,
        traversal: TreeTraversal,
    ) -> Result<()> {
        let remaining = storage.pop(remaining_len)?;
        match traversal {
            TreeTraversal::DepthFirst => {
                storage.append(&remaining)?;
                storage.append(popped)?;
            }
            TreeTraversal::BreadthFirst => {
                storage.append(popped)?;
                storage.append(&remaining)?;
            }
        }
        Ok(())
    }

    // Justification: rollback seam restores already-materialized lower/upper/input/global
    // tensors plus metadata without repacking into another struct solely to satisfy the lint.
    #[allow(clippy::too_many_arguments)]
    fn restore_popped_batch(
        &mut self,
        layer_lowers: &HashMap<String, ArrayD<f32>>,
        layer_uppers: &HashMap<String, ArrayD<f32>>,
        input_lowers: &ArrayD<f32>,
        input_uppers: &ArrayD<f32>,
        global_lbs: &ArrayD<f32>,
        global_ubs: &ArrayD<f32>,
        metadata: Vec<DomainMetadata>,
    ) -> Result<()> {
        let traversal = self.config.traversal;
        let remaining_len = self.metadata.len();

        for name in &self.config.layer_names {
            let lowers = layer_lowers.get(name).ok_or_else(|| {
                NyError::InternalError(format!(
                    "DomainList::restore_popped_batch missing lower bounds for '{name}'"
                ))
            })?;
            let uppers = layer_uppers.get(name).ok_or_else(|| {
                NyError::InternalError(format!(
                    "DomainList::restore_popped_batch missing upper bounds for '{name}'"
                ))
            })?;
            let lower_storage = self.layer_lowers.get_mut(name).ok_or_else(|| {
                NyError::InternalError(format!(
                    "DomainList::restore_popped_batch missing lower storage for '{name}'"
                ))
            })?;
            Self::restore_popped_storage(lower_storage.as_mut(), lowers, remaining_len, traversal)?;

            let upper_storage = self.layer_uppers.get_mut(name).ok_or_else(|| {
                NyError::InternalError(format!(
                    "DomainList::restore_popped_batch missing upper storage for '{name}'"
                ))
            })?;
            Self::restore_popped_storage(upper_storage.as_mut(), uppers, remaining_len, traversal)?;
        }

        Self::restore_popped_storage(
            self.input_lowers.as_mut(),
            input_lowers,
            remaining_len,
            traversal,
        )?;
        Self::restore_popped_storage(
            self.input_uppers.as_mut(),
            input_uppers,
            remaining_len,
            traversal,
        )?;
        Self::restore_popped_storage(
            self.global_lbs.as_mut(),
            global_lbs,
            remaining_len,
            traversal,
        )?;
        Self::restore_popped_storage(
            self.global_ubs.as_mut(),
            global_ubs,
            remaining_len,
            traversal,
        )?;

        match traversal {
            TreeTraversal::DepthFirst => self.metadata.extend(metadata),
            TreeTraversal::BreadthFirst => {
                let mut restored = metadata;
                restored.append(&mut self.metadata);
                self.metadata = restored;
            }
        }

        Ok(())
    }

    fn restore_picked_batch(&mut self, picked: PickedDomains) -> Result<()> {
        let global_lbs = ArrayD::from_shape_vec(IxDyn(&[picked.batch_size, 1]), picked.global_lbs)
            .map_err(|error| {
                NyError::InternalError(format!(
                    "DomainList::restore_picked_batch lower shape mismatch: {error}"
                ))
            })?;
        let global_ubs = ArrayD::from_shape_vec(IxDyn(&[picked.batch_size, 1]), picked.global_ubs)
            .map_err(|error| {
                NyError::InternalError(format!(
                    "DomainList::restore_picked_batch upper shape mismatch: {error}"
                ))
            })?;
        self.restore_popped_batch(
            &picked.layer_lowers,
            &picked.layer_uppers,
            &picked.input_lowers,
            &picked.input_uppers,
            &global_lbs,
            &global_ubs,
            picked.metadata,
        )
    }

    fn restore_popped_grouped(
        &mut self,
        row_lowers: &ArrayD<f32>,
        row_uppers: &ArrayD<f32>,
    ) -> Result<()> {
        let traversal = self.config.traversal;
        let grouped = self.grouped.as_mut().ok_or_else(|| {
            NyError::InternalError(
                "DomainList::restore_popped_grouped missing grouped storage".to_string(),
            )
        })?;
        let remaining_len = grouped.row_lowers.len();
        if grouped.row_uppers.len() != remaining_len {
            return Err(NyError::InternalError(format!(
                "DomainList::restore_popped_grouped row storage mismatch \
                 (lower={}, upper={})",
                remaining_len,
                grouped.row_uppers.len()
            )));
        }
        Self::restore_popped_storage(
            grouped.row_lowers.as_mut(),
            row_lowers,
            remaining_len,
            traversal,
        )?;
        Self::restore_popped_storage(
            grouped.row_uppers.as_mut(),
            row_uppers,
            remaining_len,
            traversal,
        )
    }

    fn empty_picked_domains() -> PickedDomains {
        PickedDomains {
            batch_size: 0,
            layer_lowers: HashMap::new(),
            layer_uppers: HashMap::new(),
            input_lowers: ArrayD::zeros(IxDyn(&[0])),
            input_uppers: ArrayD::zeros(IxDyn(&[0])),
            global_lbs: Vec::new(),
            global_ubs: Vec::new(),
            metadata: Vec::new(),
        }
    }

    /// Create a new domain list with the given configuration.
    ///
    /// Returns an error if `config.layer_names` references a layer not present
    /// in `config.layer_shapes`.
    pub fn new(config: DomainListConfig) -> Result<Self> {
        Self::new_with_grouped_layout(config, None)
    }

    /// Create a clause-aware DomainList sidecar for the future grouped GPU
    /// executor. Production routing remains default-off until GPU qualification.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn new_grouped(
        config: DomainListConfig,
        layout: GroupedDisjunctiveLayout,
    ) -> Result<Self> {
        Self::new_with_grouped_layout(config, Some(layout))
    }

    fn new_with_grouped_layout(
        config: DomainListConfig,
        grouped_layout: Option<GroupedDisjunctiveLayout>,
    ) -> Result<Self> {
        let traversal = config.traversal;
        let initial_capacity = config.initial_capacity.max(1);

        // Create storage for each layer
        let mut layer_lowers = HashMap::new();
        let mut layer_uppers = HashMap::new();
        for name in &config.layer_names {
            let element_shape = config.layer_shapes.get(name).ok_or_else(|| {
                NyError::InvalidSpec(format!("DomainListConfig missing layer shape for '{name}'"))
            })?;
            let mut full_shape = vec![0];
            full_shape.extend_from_slice(element_shape);
            layer_lowers.insert(name.clone(), create_tensor_storage(&full_shape, traversal)?);
            layer_uppers.insert(name.clone(), create_tensor_storage(&full_shape, traversal)?);
        }

        // Input bounds storage
        let mut input_shape = vec![0];
        input_shape.extend_from_slice(&config.input_shape);
        let input_lowers = create_tensor_storage(&input_shape, traversal)?;
        let input_uppers = create_tensor_storage(&input_shape, traversal)?;

        // Global bounds storage: [batch, 1]
        let global_lbs = create_tensor_storage(&[0, 1], traversal)?;
        let global_ubs = create_tensor_storage(&[0, 1], traversal)?;

        let grouped = match grouped_layout {
            Some(layout) => {
                let row_count = layout.row_count();
                Some(GroupedDisjunctiveStorage {
                    layout,
                    row_lowers: create_tensor_storage(&[0, row_count], traversal)?,
                    row_uppers: create_tensor_storage(&[0, row_count], traversal)?,
                    next_lease_id: 1,
                    next_domain_id: 1,
                    active_leases: HashMap::new(),
                    unresolved_dropped: 0,
                    search_started: false,
                    queue_token: Arc::new(()),
                })
            }
            None => None,
        };

        Ok(Self {
            config,
            alpha_queue_identity: allocate_graph_local_queue_identity()?,
            layer_lowers,
            layer_uppers,
            input_lowers,
            input_uppers,
            global_lbs,
            global_ubs,
            grouped,
            metadata: Vec::with_capacity(initial_capacity),
            evicted: 0,
        })
    }

    /// Number of domains currently stored.
    pub fn len(&self) -> usize {
        self.metadata.len()
    }

    /// Cumulative count of unverified domains removed by queue-cap eviction.
    ///
    /// Evicted domains are unexplored search space: when this is nonzero, a
    /// drained queue does not prove the property, and the BaB loop must
    /// report Unknown instead of Verified.
    pub fn evicted_count(&self) -> usize {
        self.evicted
    }

    /// Check if the domain list is empty.
    pub fn is_empty(&self) -> bool {
        self.metadata.is_empty()
    }

    /// Pick out `batch_size` domains for GPU processing.
    ///
    /// Returns domains in order determined by traversal mode:
    /// - DFS: pops from end (most recent)
    /// - BFS: pops from start (oldest)
    pub fn pick_out(&mut self, batch_size: usize) -> Result<PickedDomains> {
        if self.grouped.is_some() {
            return Err(NyError::InvalidSpec(
                "DomainList::pick_out cannot discard grouped row state; \
                 use pick_out_grouped"
                    .to_string(),
            ));
        }
        self.pick_out_scalar(batch_size)
    }

    fn pick_out_scalar(&mut self, batch_size: usize) -> Result<PickedDomains> {
        let batch_size = batch_size.min(self.len());
        if batch_size == 0 {
            return Ok(Self::empty_picked_domains());
        }

        // Pop layer bounds
        let mut layer_lowers = HashMap::new();
        let mut layer_uppers = HashMap::new();
        for name in &self.config.layer_names {
            if let Some(storage) = self.layer_lowers.get_mut(name) {
                layer_lowers.insert(name.clone(), storage.pop(batch_size)?);
            }
            if let Some(storage) = self.layer_uppers.get_mut(name) {
                layer_uppers.insert(name.clone(), storage.pop(batch_size)?);
            }
        }

        // Pop input bounds
        let input_lowers = self.input_lowers.pop(batch_size)?;
        let input_uppers = self.input_uppers.pop(batch_size)?;

        // Pop global bounds
        let global_lbs_tensor = self.global_lbs.pop(batch_size)?;
        let global_ubs_tensor = self.global_ubs.pop(batch_size)?;
        let global_lbs: Vec<f32> = global_lbs_tensor.iter().copied().collect();
        let global_ubs: Vec<f32> = global_ubs_tensor.iter().copied().collect();

        // Pop metadata based on traversal mode
        let mut metadata: Vec<DomainMetadata> = match self.config.traversal {
            TreeTraversal::DepthFirst => {
                // Pop from end
                let start = self.metadata.len() - batch_size;
                self.metadata.drain(start..).collect()
            }
            TreeTraversal::BreadthFirst => {
                // Pop from start
                self.metadata.drain(..batch_size).collect()
            }
        };

        // Defense-in-depth: reject stored non-finite data before branching (#3115).
        let validation = (|| -> Result<()> {
            super::super::utils::validate_global_bounds_finite(
                &global_lbs,
                &global_ubs,
                "DomainList::pick_out",
            )?;
            super::super::utils::validate_named_batched_tensors_finite(
                &layer_lowers,
                "DomainList::pick_out",
                "layer_lower",
                None,
            )?;
            super::super::utils::validate_named_batched_tensors_finite(
                &layer_uppers,
                "DomainList::pick_out",
                "layer_upper",
                None,
            )?;
            super::super::utils::validate_batched_tensor_finite(
                &input_lowers,
                "DomainList::pick_out",
                "input_lowers",
                None,
            )?;
            super::super::utils::validate_batched_tensor_finite(
                &input_uppers,
                "DomainList::pick_out",
                "input_uppers",
                None,
            )?;
            super::super::utils::validate_pick_out_metadata_finite(&metadata)?;
            for item in &metadata {
                item.validate_queued_alpha_state(self.alpha_queue_identity)?;
            }
            Ok(())
        })();
        if let Err(err) = validation {
            self.restore_popped_batch(
                &layer_lowers,
                &layer_uppers,
                &input_lowers,
                &input_uppers,
                &global_lbs_tensor,
                &global_ubs_tensor,
                metadata,
            )?;
            return Err(err);
        }
        let unpack_result = metadata.iter_mut().try_for_each(|metadata| {
            metadata.unpack_alpha_state_after_dequeue(self.alpha_queue_identity)
        });
        if let Err(err) = unpack_result {
            self.restore_popped_batch(
                &layer_lowers,
                &layer_uppers,
                &input_lowers,
                &input_uppers,
                &global_lbs_tensor,
                &global_ubs_tensor,
                metadata,
            )?;
            return Err(err);
        }

        Ok(PickedDomains {
            batch_size,
            layer_lowers,
            layer_uppers,
            input_lowers,
            input_uppers,
            global_lbs,
            global_ubs,
            metadata,
        })
    }

    /// Pick a batch while moving its clause layout and all packed row bounds
    /// with the scalar DomainList state.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn pick_out_grouped(&mut self, batch_size: usize) -> Result<PickedGroupedDomains> {
        self.validate_grouped_alignment()?;
        let batch_size = batch_size.min(self.len());
        let lease_ids = if batch_size == 0 {
            None
        } else {
            let grouped = self.grouped.as_ref().ok_or_else(|| {
                NyError::InvalidSpec(
                    "DomainList::pick_out_grouped called on a scalar DomainList".to_string(),
                )
            })?;
            let lease_id = grouped.next_lease_id;
            let next_lease_id = lease_id.checked_add(1).ok_or_else(|| {
                NyError::InternalError("grouped DomainList lease counter overflow".to_string())
            })?;
            if grouped.active_leases.contains_key(&lease_id) {
                return Err(NyError::InternalError(format!(
                    "grouped DomainList duplicate lease id {lease_id}"
                )));
            }
            let domain_count = u64::try_from(batch_size).map_err(|_| {
                NyError::InternalError(
                    "grouped DomainList batch size does not fit domain ID counter".to_string(),
                )
            })?;
            let first_domain_id = grouped.next_domain_id;
            let next_domain_id = first_domain_id.checked_add(domain_count).ok_or_else(|| {
                NyError::InternalError("grouped DomainList domain ID counter overflow".to_string())
            })?;
            Some((lease_id, next_lease_id, first_domain_id, next_domain_id))
        };
        let (row_lowers, row_uppers, layout, queue_token) = {
            let grouped = self.grouped.as_mut().ok_or_else(|| {
                NyError::InvalidSpec(
                    "DomainList::pick_out_grouped called on a scalar DomainList".to_string(),
                )
            })?;
            let row_lowers = grouped.row_lowers.pop(batch_size)?;
            let row_uppers = match grouped.row_uppers.pop(batch_size) {
                Ok(row_uppers) => row_uppers,
                Err(error) => {
                    let remaining_len = grouped.row_lowers.len();
                    Self::restore_popped_storage(
                        grouped.row_lowers.as_mut(),
                        &row_lowers,
                        remaining_len,
                        self.config.traversal,
                    )?;
                    return Err(error);
                }
            };
            (
                row_lowers,
                row_uppers,
                grouped.layout.clone(),
                Arc::clone(&grouped.queue_token),
            )
        };

        let picked = match self.pick_out_scalar(batch_size) {
            Ok(picked) => picked,
            Err(error) => {
                self.restore_popped_grouped(&row_lowers, &row_uppers)?;
                return Err(error);
            }
        };
        let row_bounds = PackedGroupedBounds::new(row_lowers, row_uppers);
        let summaries = match row_bounds.summaries(&layout, picked.batch_size) {
            Ok(summaries) => summaries,
            Err(error) => {
                let restore_lowers = row_bounds.row_lowers().clone();
                let restore_uppers = row_bounds.row_uppers().clone();
                self.restore_picked_batch(picked)?;
                self.restore_popped_grouped(&restore_lowers, &restore_uppers)?;
                return Err(error);
            }
        };

        let (domain_ids, authority) =
            if let Some((lease_id, next_lease_id, first_domain_id, next_domain_id)) = lease_ids {
                let domain_ids: Vec<GroupedDomainId> = (first_domain_id..next_domain_id)
                    .map(GroupedDomainId::from_queue_counter)
                    .collect();
                let lease = match GroupedLeaseState::new(&picked, &domain_ids, &summaries) {
                    Ok(lease) => lease,
                    Err(error) => {
                        let restore_lowers = row_bounds.row_lowers().clone();
                        let restore_uppers = row_bounds.row_uppers().clone();
                        self.restore_picked_batch(picked)?;
                        self.restore_popped_grouped(&restore_lowers, &restore_uppers)?;
                        return Err(error);
                    }
                };
                let grouped = self.grouped.as_mut().ok_or_else(|| {
                    NyError::InternalError(
                        "DomainList::pick_out_grouped lost grouped storage".to_string(),
                    )
                })?;
                grouped.next_lease_id = next_lease_id;
                grouped.next_domain_id = next_domain_id;
                grouped.active_leases.insert(lease_id, lease);
                (domain_ids, Some((lease_id, queue_token)))
            } else {
                (Vec::new(), None)
            };

        Ok(PickedGroupedDomains::new(
            picked, row_bounds, layout, domain_ids, authority,
        ))
    }

    /// Convenience wrapper around `pick_out` that accepts GPU batch options.
    ///
    /// Reference: `designs/2026-02-03-batched-domain-pickout-gpu-transfer.md`,
    /// alpha-beta-CROWN `complete_verifier/branching_domains.py`:270-305
    pub fn pick_out_batched(
        &mut self,
        batch_size: usize,
        _options: BatchedDomainOptions,
    ) -> Result<PickedDomains> {
        self.pick_out(batch_size)
    }

    /// Add processed domains back to the list.
    ///
    /// Only domains where `keep_mask[i]` is true are added.
    pub fn add(&mut self, processed: ProcessedDomains) -> Result<()> {
        if self.grouped.is_some() {
            return Err(NyError::InvalidSpec(
                "DomainList::add cannot bypass sealed grouped evaluation".to_string(),
            ));
        }
        self.add_impl(processed, None)
    }

    pub(super) fn append_sealed_grouped_queued(
        &mut self,
        entry: SealedGroupedQueueEntry,
    ) -> Result<()> {
        let (processed, row_bounds) = entry.into_queued_payload()?;
        self.add_impl(processed, Some(row_bounds))
    }

    fn add_impl(
        &mut self,
        mut processed: ProcessedDomains,
        grouped_bounds: Option<PackedGroupedBounds>,
    ) -> Result<()> {
        if self.grouped.is_some() != grouped_bounds.is_some() {
            return Err(NyError::InvalidSpec(
                "DomainList::add grouped sidecar mismatch".to_string(),
            ));
        }
        let batch_size = processed.global_lbs.len();
        if processed.global_ubs.len() != batch_size {
            return Err(NyError::InvalidSpec(format!(
                "DomainList::add global bound length mismatch (lower={}, upper={})",
                batch_size,
                processed.global_ubs.len()
            )));
        }
        if processed.metadata.len() != batch_size {
            return Err(NyError::InvalidSpec(format!(
                "DomainList::add metadata length mismatch (global={}, metadata={})",
                batch_size,
                processed.metadata.len()
            )));
        }
        if processed.keep_mask.len() != batch_size {
            return Err(NyError::InvalidSpec(format!(
                "DomainList::add keep_mask length mismatch (global={}, keep_mask={})",
                batch_size,
                processed.keep_mask.len()
            )));
        }
        let input_lower_batch = processed.input_lowers.shape().first().copied().unwrap_or(0);
        let input_upper_batch = processed.input_uppers.shape().first().copied().unwrap_or(0);
        if input_lower_batch != batch_size || input_upper_batch != batch_size {
            return Err(NyError::InvalidSpec(format!(
                "DomainList::add input bound batch mismatch (global={}, input_lower={}, input_upper={})",
                batch_size, input_lower_batch, input_upper_batch
            )));
        }

        let configured_layers: std::collections::HashSet<&str> =
            self.config.layer_names.iter().map(String::as_str).collect();
        for layer_name in processed.layer_lowers.keys() {
            if !configured_layers.contains(layer_name.as_str()) {
                return Err(NyError::InvalidSpec(format!(
                    "DomainList::add received unconfigured layer lower bounds for '{layer_name}'"
                )));
            }
        }
        for layer_name in processed.layer_uppers.keys() {
            if !configured_layers.contains(layer_name.as_str()) {
                return Err(NyError::InvalidSpec(format!(
                    "DomainList::add received unconfigured layer upper bounds for '{layer_name}'"
                )));
            }
        }

        if batch_size == 0 {
            return Ok(());
        }

        // Reject domains with NaN/Inf global bounds (#2246).
        super::super::utils::validate_global_bounds_finite(
            &processed.global_lbs,
            &processed.global_ubs,
            "DomainList::add",
        )?;

        // Reject non-finite kept layer/input tensors (#3115).
        // Uses keep_mask so dropped rows don't trigger rejection.
        let mask = &processed.keep_mask;
        super::super::utils::validate_named_batched_tensors_finite(
            &processed.layer_lowers,
            "DomainList::add",
            "layer_lower",
            Some(mask),
        )?;
        super::super::utils::validate_named_batched_tensors_finite(
            &processed.layer_uppers,
            "DomainList::add",
            "layer_upper",
            Some(mask),
        )?;
        super::super::utils::validate_batched_tensor_finite(
            &processed.input_lowers,
            "DomainList::add",
            "input_lowers",
            Some(mask),
        )?;
        super::super::utils::validate_batched_tensor_finite(
            &processed.input_uppers,
            "DomainList::add",
            "input_uppers",
            Some(mask),
        )?;
        super::super::utils::validate_add_metadata_finite(&processed.metadata, mask)?;

        // Count domains to keep
        let keep_count = processed.keep_mask.iter().filter(|&&x| x).count();
        if keep_count == 0 {
            return Ok(());
        }

        // DomainList stores per-layer tensors for every kept domain when
        // `config.layer_names` is non-empty. Missing layer tensors here would
        // desynchronize layer storage length from metadata/global/input storage.
        for name in &self.config.layer_names {
            let has_lower = processed.layer_lowers.contains_key(name);
            let has_upper = processed.layer_uppers.contains_key(name);
            if !has_lower && !has_upper {
                return Err(NyError::InvalidSpec(format!(
                    "DomainList::add missing layer bounds for configured layer '{name}'"
                )));
            }
            if has_lower != has_upper {
                return Err(NyError::InvalidSpec(format!(
                    "DomainList::add incomplete layer bounds for '{name}' (lower={}, upper={})",
                    has_lower, has_upper
                )));
            }
        }

        // C2 packed graph-alpha queue state is dark by default. Pack only kept
        // metadata, before mutating any tensor storage, so a validation/format
        // refusal leaves the queue transaction untouched.
        if packed_graph_alpha_queue_enabled() {
            for (metadata, keep) in processed.metadata.iter_mut().zip(&processed.keep_mask) {
                if *keep {
                    metadata.pack_alpha_state_for_queue(self.alpha_queue_identity)?;
                }
            }
        }

        // Filter and add layer bounds
        for name in &self.config.layer_names {
            let lowers = processed.layer_lowers.get(name).ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "DomainList::add missing lower bounds for configured layer '{name}'"
                ))
            })?;
            let uppers = processed.layer_uppers.get(name).ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "DomainList::add missing upper bounds for configured layer '{name}'"
                ))
            })?;
            let filtered_lowers = filter_batch(lowers, &processed.keep_mask)?;
            let filtered_uppers = filter_batch(uppers, &processed.keep_mask)?;
            if let Some(storage) = self.layer_lowers.get_mut(name) {
                storage.append(&filtered_lowers)?;
            }
            if let Some(storage) = self.layer_uppers.get_mut(name) {
                storage.append(&filtered_uppers)?;
            }
        }

        // Filter and add input bounds
        let filtered_input_lowers = filter_batch(&processed.input_lowers, &processed.keep_mask)?;
        let filtered_input_uppers = filter_batch(&processed.input_uppers, &processed.keep_mask)?;
        self.input_lowers.append(&filtered_input_lowers)?;
        self.input_uppers.append(&filtered_input_uppers)?;

        // Filter and add global bounds
        let filtered_lbs: Vec<f32> = processed
            .global_lbs
            .iter()
            .zip(&processed.keep_mask)
            .filter(|(_, &keep)| keep)
            .map(|(&lb, _)| lb)
            .collect();
        let filtered_ubs: Vec<f32> = processed
            .global_ubs
            .iter()
            .zip(&processed.keep_mask)
            .filter(|(_, &keep)| keep)
            .map(|(&ub, _)| ub)
            .collect();

        let lbs_tensor = ArrayD::from_shape_vec(IxDyn(&[filtered_lbs.len(), 1]), filtered_lbs)
            .map_err(|e| {
                NyError::InvalidSpec(format!(
                    "failed to build global lower-bound tensor [n,1]: {e}"
                ))
            })?;
        let ubs_tensor = ArrayD::from_shape_vec(IxDyn(&[filtered_ubs.len(), 1]), filtered_ubs)
            .map_err(|e| {
                NyError::InvalidSpec(format!(
                    "failed to build global upper-bound tensor [n,1]: {e}"
                ))
            })?;
        self.global_lbs.append(&lbs_tensor)?;
        self.global_ubs.append(&ubs_tensor)?;

        // Apply exactly the same compaction mask to the packed per-row state.
        if let Some(row_bounds) = grouped_bounds.as_ref() {
            let filtered_row_lowers = filter_batch(row_bounds.row_lowers(), &processed.keep_mask)?;
            let filtered_row_uppers = filter_batch(row_bounds.row_uppers(), &processed.keep_mask)?;
            let grouped = self.grouped.as_mut().ok_or_else(|| {
                NyError::InternalError(
                    "DomainList::add_impl lost grouped sidecar storage".to_string(),
                )
            })?;
            grouped.row_lowers.append(&filtered_row_lowers)?;
            grouped.row_uppers.append(&filtered_row_uppers)?;
        }

        // Filter and add metadata
        let filtered_metadata: Vec<DomainMetadata> = processed
            .metadata
            .into_iter()
            .zip(&processed.keep_mask)
            .filter(|(_, &keep)| keep)
            .map(|(m, _)| m)
            .collect();
        self.metadata.extend(filtered_metadata);

        // Evict lowest-priority domains when queue exceeds max_queue_size (#2326).
        // Eviction discards unverified domains; it is recorded in
        // `evicted_count()` so the BaB loop reports Unknown rather than
        // Verified when the truncated queue drains.
        self.evict_excess_domains()?;
        self.validate_grouped_alignment()?;

        Ok(())
    }
}
