// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::mem::size_of;

use ny_core::{
    GpuBabBoundArenaRange, GpuBabBoundSplitHistoryLiteral, GpuBabBoundSplitHistoryPhase,
    GPU_BAB_BOUND_MAX_APPEND_SPLITS, GPU_BAB_BOUND_MAX_ARENA_VALUES,
    GPU_BAB_BOUND_MAX_SPLIT_HISTORY_WORDS, GPU_BAB_BOUND_SPLIT_HISTORY_RECORD_WORDS,
};

use crate::beta_crown::branching::GraphSplitHistory;
use crate::beta_crown::state::GraphBetaState;
use crate::resident_bab_wire::v1::RESIDENT_BAB_MAX_NODE_NAME_BYTES_V1;

use super::budget::{
    checked_add, checked_elements, checked_hash_entries, invalid, poll_scaled,
    ResidentBabHostBudgetV1,
};
use super::ResidentBabComposeErrorV1;
use super::{ResidentBabHistoryBetaV1, ResidentBabReluSiteV1};

const SPLIT_HISTORY_TAG_V1: u32 = 0x4E59_0100;
const SPLIT_HISTORY_TAG_MASK_V1: u32 = !1;

pub(super) fn history_nominal_bytes(
    site_count: usize,
    history_count: usize,
    beta_len: usize,
) -> Result<usize, ResidentBabComposeErrorV1> {
    let history_words = history_count
        .checked_mul(GPU_BAB_BOUND_SPLIT_HISTORY_RECORD_WORDS)
        .ok_or_else(|| invalid("retained-BaB nominal history word count overflows"))?;
    let mut bytes = size_of::<ResidentBabHistoryBetaV1>();
    for charge in [
        size_of::<HashMap<&str, usize>>(),
        size_of::<HashSet<u32>>(),
        size_of::<HashSet<(usize, usize)>>(),
        checked_elements::<GpuBabBoundArenaRange>(site_count)?,
        checked_elements::<u32>(history_words)?,
        checked_elements::<f32>(beta_len)?,
        checked_hash_entries::<&str, usize>(site_count)?,
        checked_hash_entries::<u32, ()>(site_count)?,
        checked_hash_entries::<(usize, usize), ()>(history_count)?,
    ] {
        checked_add(&mut bytes, charge)?;
    }
    Ok(bytes)
}

/// Compose the exact four-u32 history records and the canonical dense signed
/// beta table for one domain.
///
/// The beta table contains one full-width row per `sites` entry. Unconstrained
/// cells are canonical positive zero. A constrained cell preserves
/// `value * sign` bit-for-bit, including negative zero for an inactive
/// zero-valued multiplier. The sparse history and `GraphBetaState` must be an
/// ordered one-to-one association; permissive epsilon matching or duplicate
/// overwrite is never used at this boundary.
pub(super) fn compose_history_beta_v1(
    sites: &[ResidentBabReluSiteV1],
    history: &GraphSplitHistory,
    beta_state: &GraphBetaState,
    budget: &mut ResidentBabHostBudgetV1,
    check: &mut dyn FnMut(&'static str) -> ny_core::Result<()>,
) -> Result<ResidentBabHistoryBetaV1, ResidentBabComposeErrorV1> {
    if sites.is_empty() {
        return Err(invalid(
            "retained-BaB v1 requires at least one true ReLU site",
        ));
    }
    if !history.is_pure_relu_at_zero() || !history.genbab_split_ids.is_empty() {
        return Err(invalid(
            "retained-BaB v1 accepts only pure ReLU-at-positive-zero histories with no GenBaB IDs",
        ));
    }
    if history.constraints.len() != beta_state.entries.len() {
        return Err(invalid(format!(
            "retained-BaB history/beta cardinality mismatch: history={}, beta={}",
            history.constraints.len(),
            beta_state.entries.len()
        )));
    }

    let history_word_count = history
        .constraints
        .len()
        .checked_mul(GPU_BAB_BOUND_SPLIT_HISTORY_RECORD_WORDS)
        .ok_or_else(|| invalid("retained-BaB history word count overflows usize"))?;
    if history_word_count > GPU_BAB_BOUND_MAX_SPLIT_HISTORY_WORDS {
        return Err(invalid(
            "retained-BaB history exceeds the core split-history word cap",
        ));
    }

    let mut sealed_beta_len = 0usize;
    for (site_index, site) in sites.iter().enumerate() {
        poll_scaled(check, "resident ReLU-site cap validation", site_index)?;
        if site.node_name.is_empty()
            || site.node_name.len() > RESIDENT_BAB_MAX_NODE_NAME_BYTES_V1
            || site.topology_node_id == u32::MAX
            || site.preactivation_width == 0
        {
            return Err(invalid(format!(
                "retained-BaB ReLU site {site_index} exceeds the v1 name/width/ID bounds"
            )));
        }
        sealed_beta_len = sealed_beta_len
            .checked_add(site.preactivation_width)
            .ok_or_else(|| invalid("retained-BaB dense beta length overflows usize"))?;
        if sealed_beta_len > GPU_BAB_BOUND_MAX_ARENA_VALUES {
            return Err(invalid(
                "retained-BaB dense beta exceeds the core arena cap",
            ));
        }
    }
    check("resident ReLU-site cap validation final")?;
    for (literal_index, constraint) in history.constraints.iter().enumerate() {
        poll_scaled(check, "resident history-name cap validation", literal_index)?;
        if constraint.node_name.is_empty()
            || constraint.node_name.len() > RESIDENT_BAB_MAX_NODE_NAME_BYTES_V1
        {
            return Err(invalid(format!(
                "retained-BaB history literal {literal_index} has a name outside the v1 bound"
            )));
        }
    }
    check("resident history-name cap validation final")?;
    for (entry_index, entry) in beta_state.entries.iter().enumerate() {
        poll_scaled(check, "resident beta-name cap validation", entry_index)?;
        if entry.node_name.is_empty() || entry.node_name.len() > RESIDENT_BAB_MAX_NODE_NAME_BYTES_V1
        {
            return Err(invalid(format!(
                "retained-BaB beta entry {entry_index} has a name outside the v1 bound"
            )));
        }
    }
    check("resident beta-name cap validation final")?;

    let mut site_by_name = HashMap::new();
    site_by_name
        .try_reserve(sites.len())
        .map_err(|_| ResidentBabComposeErrorV1::AllocationRefused("ReLU-site index"))?;
    budget.charge_hash_capacity::<&str, usize>(sites.len(), site_by_name.capacity())?;
    check("resident ReLU-site index reserve")?;
    let mut topology_ids = HashSet::new();
    topology_ids
        .try_reserve(sites.len())
        .map_err(|_| ResidentBabComposeErrorV1::AllocationRefused("topology-ID index"))?;
    budget.charge_hash_capacity::<u32, ()>(sites.len(), topology_ids.capacity())?;
    check("resident topology-ID index reserve")?;

    let mut beta_len = 0usize;
    let mut beta_rows = Vec::new();
    budget.reserve_vec(&mut beta_rows, sites.len(), "beta row metadata")?;
    check("resident beta row reserve")?;
    for (site_index, site) in sites.iter().enumerate() {
        poll_scaled(check, "resident ReLU-site validation", site_index)?;
        check("resident ReLU-site name indexing")?;
        if site.node_name.is_empty()
            || site.topology_node_id == u32::MAX
            || site.preactivation_width == 0
        {
            return Err(invalid(format!(
                "retained-BaB ReLU site {site_index} has an empty name, reserved ID, or zero width"
            )));
        }
        if site_by_name
            .insert(site.node_name.as_str(), site_index)
            .is_some()
            || !topology_ids.insert(site.topology_node_id)
        {
            return Err(invalid(
                "retained-BaB ReLU sites repeat a name or topology ID",
            ));
        }
        let start = beta_len;
        beta_len = beta_len
            .checked_add(site.preactivation_width)
            .ok_or_else(|| invalid("retained-BaB dense beta length overflows usize"))?;
        if beta_len > GPU_BAB_BOUND_MAX_ARENA_VALUES {
            return Err(invalid(
                "retained-BaB dense beta exceeds the core arena cap",
            ));
        }
        beta_rows.push(GpuBabBoundArenaRange {
            start,
            len: site.preactivation_width,
        });
    }
    check("resident ReLU-site validation final")?;

    let mut history_words = Vec::new();
    budget.reserve_vec(
        &mut history_words,
        history_word_count,
        "split-history words",
    )?;
    check("resident history word reserve")?;
    let mut beta = Vec::new();
    budget.reserve_vec(&mut beta, beta_len, "dense signed beta")?;
    check("resident dense beta reserve")?;
    for beta_index in 0..beta_len {
        poll_scaled(check, "resident dense beta initialization", beta_index)?;
        beta.push(0.0);
    }
    check("resident dense beta initialization final")?;

    let mut seen_literals = HashSet::new();
    seen_literals
        .try_reserve(history.constraints.len())
        .map_err(|_| ResidentBabComposeErrorV1::AllocationRefused("history literal index"))?;
    budget.charge_hash_capacity::<(usize, usize), ()>(
        history.constraints.len(),
        seen_literals.capacity(),
    )?;
    check("resident history literal index reserve")?;

    for (literal_index, (constraint, entry)) in history
        .constraints
        .iter()
        .zip(beta_state.entries.iter())
        .enumerate()
    {
        poll_scaled(check, "resident history/beta validation", literal_index)?;
        check("resident history/beta name validation")?;
        let site_index = site_by_name
            .get(constraint.node_name.as_str())
            .copied()
            .ok_or_else(|| {
                invalid(format!(
                    "retained-BaB history literal {literal_index} names a node outside the serialized ReLU sites"
                ))
            })?;
        let site = &sites[site_index];
        if constraint.neuron_idx >= site.preactivation_width {
            return Err(invalid(format!(
                "retained-BaB history literal {literal_index} neuron {} exceeds ReLU width {}",
                constraint.neuron_idx, site.preactivation_width
            )));
        }
        if !seen_literals.insert((site_index, constraint.neuron_idx)) {
            return Err(invalid(
                "retained-BaB history repeats a ReLU-site/neuron literal",
            ));
        }
        if !constraint.score.is_finite() {
            return Err(invalid(format!(
                "retained-BaB history literal {literal_index} has a nonfinite score"
            )));
        }

        let expected_sign = if constraint.is_active {
            1.0f32
        } else {
            -1.0f32
        };
        if entry.node_name != constraint.node_name
            || entry.neuron_idx != constraint.neuron_idx
            || entry.split_point.to_bits() != 0.0f32.to_bits()
            || entry.sign.to_bits() != expected_sign.to_bits()
            || !entry.value.is_finite()
            || entry.value < 0.0
            || (entry.value == 0.0 && entry.value.to_bits() != 0.0f32.to_bits())
        {
            return Err(invalid(format!(
                "retained-BaB history literal {literal_index} does not exactly match its beta entry"
            )));
        }

        let phase = if constraint.is_active {
            GpuBabBoundSplitHistoryPhase::Active
        } else {
            GpuBabBoundSplitHistoryPhase::Inactive
        };
        let topology_node_id = site.topology_node_id;
        let neuron_index = u32::try_from(constraint.neuron_idx).map_err(|_| {
            invalid(format!(
                "retained-BaB history literal {literal_index} neuron does not fit u32"
            ))
        })?;
        history_words.extend_from_slice(
            &GpuBabBoundSplitHistoryLiteral {
                phase,
                topology_node_id,
                neuron_index,
                score: constraint.score,
            }
            .encode_words()?,
        );

        let beta_index = beta_rows[site_index]
            .start
            .checked_add(constraint.neuron_idx)
            .ok_or_else(|| invalid("retained-BaB beta index overflows usize"))?;
        beta[beta_index] = entry.signed_value();
    }
    check("resident history/beta validation final")?;

    Ok(ResidentBabHistoryBetaV1 {
        history_words,
        beta,
        beta_rows,
    })
}

/// Validate one append-only child history against an exact parent prefix and
/// return its stable branch pattern. Bit zero is the oldest suffix literal.
///
/// This helper deliberately supports the core's generic `k <= 16` structural
/// contract even though the first live adapter admits only whole-parent `k=1`
/// groups.
fn validate_append_lengths_v1(
    parent_len: usize,
    child_len: usize,
) -> Result<usize, ResidentBabComposeErrorV1> {
    if parent_len > child_len
        || parent_len > GPU_BAB_BOUND_MAX_SPLIT_HISTORY_WORDS
        || child_len > GPU_BAB_BOUND_MAX_SPLIT_HISTORY_WORDS
        || !parent_len.is_multiple_of(GPU_BAB_BOUND_SPLIT_HISTORY_RECORD_WORDS)
        || !child_len.is_multiple_of(GPU_BAB_BOUND_SPLIT_HISTORY_RECORD_WORDS)
    {
        return Err(invalid(
            "retained-BaB child history does not preserve the exact record-aligned parent prefix",
        ));
    }
    let suffix_words = child_len - parent_len;
    let suffix_records = suffix_words / GPU_BAB_BOUND_SPLIT_HISTORY_RECORD_WORDS;
    if suffix_records == 0 || suffix_records > GPU_BAB_BOUND_MAX_APPEND_SPLITS {
        return Err(invalid(
            "retained-BaB append suffix must contain between one and sixteen literals",
        ));
    }
    Ok(suffix_records)
}

pub(in crate::beta_crown::engine::graph) fn validate_append_suffix_v1(
    parent: &[u32],
    child: &[u32],
    check: &mut dyn FnMut(&'static str) -> ny_core::Result<()>,
) -> Result<u64, ResidentBabComposeErrorV1> {
    validate_append_lengths_v1(parent.len(), child.len())?;
    for (word_index, (&parent_word, &child_word)) in
        parent.iter().zip(&child[..parent.len()]).enumerate()
    {
        poll_scaled(
            check,
            "resident append-history prefix validation",
            word_index,
        )?;
        if parent_word != child_word {
            return Err(invalid(
                "retained-BaB child history does not preserve the exact record-aligned parent prefix",
            ));
        }
    }
    check("resident append-history prefix validation final")?;
    let suffix = &child[parent.len()..];

    let mut pattern = 0u64;
    // `as_chunks::<N>()` (the tippy suggestion) reshapes this validation
    // walk's types; keep `chunks_exact` until the public pin's clippy also
    // carries the lint.
    #[allow(unknown_lints)] // stock 1.95 clippy (public pin) does not know the lint below
    #[allow(clippy::chunks_exact_to_as_chunks)]
    for (offset, record) in suffix
        .chunks_exact(GPU_BAB_BOUND_SPLIT_HISTORY_RECORD_WORDS)
        .enumerate()
    {
        poll_scaled(check, "resident append-history validation", offset)?;
        if record[0] & SPLIT_HISTORY_TAG_MASK_V1 != SPLIT_HISTORY_TAG_V1
            || record[1] == u32::MAX
            || !f32::from_bits(record[3]).is_finite()
        {
            return Err(invalid(format!(
                "retained-BaB append suffix record {offset} is malformed"
            )));
        }
        pattern |= u64::from(record[0] & 1) << offset;
    }
    check("resident append-history validation final")?;
    Ok(pattern)
}

#[cfg(test)]
mod tests {
    use super::super::budget::RESIDENT_BAB_COMPOSE_POLL_STRIDE;
    use super::*;
    use crate::beta_crown::branching::GraphNeuronConstraint;
    use crate::beta_crown::state::GraphBetaEntry;

    fn compose_for_test(
        sites: &[ResidentBabReluSiteV1],
        history: &GraphSplitHistory,
        beta: &GraphBetaState,
    ) -> Result<ResidentBabHistoryBetaV1, ResidentBabComposeErrorV1> {
        let beta_len = sites.iter().try_fold(0usize, |total, site| {
            total.checked_add(site.preactivation_width)
        });
        let nominal = history_nominal_bytes(
            sites.len(),
            history.constraints.len(),
            beta_len.expect("small fixture beta length"),
        )
        .expect("small fixture nominal bytes");
        let mut budget = ResidentBabHostBudgetV1::begin(
            super::super::ResidentBabAdapterHostCapV1 {
                limit_bytes: 1 << 20,
                resident_bytes_before: 0,
            },
            nominal,
        )?;
        let mut check = |_| Ok(());
        compose_history_beta_v1(sites, history, beta, &mut budget, &mut check)
    }

    fn append_for_test(parent: &[u32], child: &[u32]) -> Result<u64, ResidentBabComposeErrorV1> {
        let mut check = |_| Ok(());
        validate_append_suffix_v1(parent, child, &mut check)
    }

    fn sites() -> Vec<ResidentBabReluSiteV1> {
        vec![
            ResidentBabReluSiteV1 {
                topology_node_id: 3,
                node_name: "relu_a".to_string(),
                preactivation_width: 3,
            },
            ResidentBabReluSiteV1 {
                topology_node_id: 8,
                node_name: "relu_b".to_string(),
                preactivation_width: 2,
            },
        ]
    }

    fn history_beta() -> (GraphSplitHistory, GraphBetaState) {
        let mut history = GraphSplitHistory::new();
        history.add_constraint(
            GraphNeuronConstraint::new("relu_b".to_string(), 1, false, -0.0).unwrap(),
        );
        history.add_constraint(
            GraphNeuronConstraint::new("relu_a".to_string(), 2, true, 0.25).unwrap(),
        );
        let beta = GraphBetaState::from_entries(vec![
            GraphBetaEntry::new("relu_b".to_string(), 1, 0.0, 0.0, -1.0).unwrap(),
            GraphBetaEntry::new("relu_a".to_string(), 2, 0.0, 1.5, 1.0).unwrap(),
        ]);
        (history, beta)
    }

    #[test]
    fn dense_beta_uses_fold_rows_and_preserves_signed_zero() {
        let (history, beta) = history_beta();
        let composed = compose_for_test(&sites(), &history, &beta).unwrap();
        assert_eq!(
            composed.beta_rows,
            vec![
                GpuBabBoundArenaRange { start: 0, len: 3 },
                GpuBabBoundArenaRange { start: 3, len: 2 },
            ]
        );
        assert_eq!(composed.beta.len(), 5);
        assert_eq!(composed.beta[0].to_bits(), 0.0f32.to_bits());
        assert_eq!(composed.beta[2].to_bits(), 1.5f32.to_bits());
        assert_eq!(composed.beta[4].to_bits(), (-0.0f32).to_bits());
        assert_eq!(composed.history_words[0], SPLIT_HISTORY_TAG_V1);
        assert_eq!(composed.history_words[1], 8);
        assert_eq!(composed.history_words[2], 1);
        assert_eq!(composed.history_words[3], (-0.0f32).to_bits());
        assert_eq!(composed.history_words[4], SPLIT_HISTORY_TAG_V1 | 1);
    }

    #[test]
    fn nominal_history_charge_includes_all_three_live_index_owners() {
        let site_count = 2usize;
        let history_count = 2usize;
        let beta_len = 5usize;
        let index_owner_bytes = size_of::<HashMap<&str, usize>>()
            + size_of::<HashSet<u32>>()
            + size_of::<HashSet<(usize, usize)>>();
        let mut expected = size_of::<ResidentBabHistoryBetaV1>() + index_owner_bytes;
        for charge in [
            checked_elements::<GpuBabBoundArenaRange>(site_count).unwrap(),
            checked_elements::<u32>(history_count * GPU_BAB_BOUND_SPLIT_HISTORY_RECORD_WORDS)
                .unwrap(),
            checked_elements::<f32>(beta_len).unwrap(),
            checked_hash_entries::<&str, usize>(site_count).unwrap(),
            checked_hash_entries::<u32, ()>(site_count).unwrap(),
            checked_hash_entries::<(usize, usize), ()>(history_count).unwrap(),
        ] {
            expected += charge;
        }
        let nominal = history_nominal_bytes(site_count, history_count, beta_len).unwrap();
        assert_eq!(nominal, expected);

        let cap = super::super::ResidentBabAdapterHostCapV1 {
            limit_bytes: nominal - 1,
            resident_bytes_before: 0,
        };
        assert!(matches!(
            ResidentBabHostBudgetV1::begin(cap, nominal),
            Err(ResidentBabComposeErrorV1::Capacity {
                required_bytes,
                limit_bytes,
            }) if required_bytes == nominal && limit_bytes == nominal - 1
        ));
        let exact = ResidentBabHostBudgetV1::begin(
            super::super::ResidentBabAdapterHostCapV1 {
                limit_bytes: nominal,
                resident_bytes_before: 0,
            },
            nominal,
        )
        .unwrap();
        assert_eq!(exact.peak_bytes(), nominal);
    }

    #[test]
    fn beta_magnitude_requires_positive_zero_before_phase_signing() {
        let (history, _) = history_beta();
        let negative_zero = GraphBetaState::from_entries(vec![
            GraphBetaEntry::new("relu_b".to_string(), 1, 0.0, -0.0, -1.0).unwrap(),
            GraphBetaEntry::new("relu_a".to_string(), 2, 0.0, 1.5, 1.0).unwrap(),
        ]);
        assert!(compose_for_test(&sites(), &history, &negative_zero).is_err());

        let active_zero = GraphBetaState::from_entries(vec![
            GraphBetaEntry::new("relu_b".to_string(), 1, 0.0, 0.0, -1.0).unwrap(),
            GraphBetaEntry::new("relu_a".to_string(), 2, 0.0, 0.0, 1.0).unwrap(),
        ]);
        let composed = compose_for_test(&sites(), &history, &active_zero).unwrap();
        assert_eq!(composed.beta[2].to_bits(), 0.0f32.to_bits());
        assert_eq!(composed.beta[4].to_bits(), (-0.0f32).to_bits());
    }

    #[test]
    fn epsilon_split_point_is_rejected() {
        let (history, _) = history_beta();
        let beta = GraphBetaState::from_entries(vec![
            GraphBetaEntry::new("relu_b".to_string(), 1, 1.0e-7, 0.0, -1.0).unwrap(),
            GraphBetaEntry::new("relu_a".to_string(), 2, 0.0, 1.5, 1.0).unwrap(),
        ]);
        let error = compose_for_test(&sites(), &history, &beta).unwrap_err();
        assert!(error.to_string().contains("exactly match"));
    }

    #[test]
    fn reordered_or_extra_beta_is_rejected() {
        let (history, _) = history_beta();
        let reordered = GraphBetaState::from_entries(vec![
            GraphBetaEntry::new("relu_a".to_string(), 2, 0.0, 1.5, 1.0).unwrap(),
            GraphBetaEntry::new("relu_b".to_string(), 1, 0.0, 0.0, -1.0).unwrap(),
        ]);
        assert!(compose_for_test(&sites(), &history, &reordered).is_err());

        let mut extra = reordered;
        extra
            .entries
            .push(GraphBetaEntry::new("relu_a".to_string(), 0, 0.0, 0.0, 1.0).unwrap());
        assert!(compose_for_test(&sites(), &history, &extra).is_err());
    }

    #[test]
    fn stale_genbab_split_id_is_rejected_even_without_genbab_constraints() {
        let (mut history, beta) = history_beta();
        history.genbab_split_ids.push(0);
        let error = compose_for_test(&sites(), &history, &beta).unwrap_err();
        assert!(error.to_string().contains("no GenBaB IDs"));
    }

    #[test]
    fn relu_site_validation_deadline_fires_at_second_stride() {
        let sites: Vec<_> = (0..=RESIDENT_BAB_COMPOSE_POLL_STRIDE)
            .map(|index| ResidentBabReluSiteV1 {
                topology_node_id: u32::try_from(index).unwrap(),
                node_name: format!("relu_{index}"),
                preactivation_width: 1,
            })
            .collect();
        let history = GraphSplitHistory::new();
        let beta = GraphBetaState::default();
        let nominal = history_nominal_bytes(sites.len(), 0, sites.len()).unwrap();
        let mut budget = ResidentBabHostBudgetV1::begin(
            super::super::ResidentBabAdapterHostCapV1 {
                limit_bytes: 16 << 20,
                resident_bytes_before: 0,
            },
            nominal,
        )
        .unwrap();
        let mut stride_checks = 0usize;
        let mut check = |label| {
            if label == "resident ReLU-site validation" {
                stride_checks += 1;
                if stride_checks == 2 {
                    return Err(ny_core::NyError::DeadlineExceeded(
                        "injected ReLU-site deadline".to_string(),
                    ));
                }
            }
            Ok(())
        };
        assert!(matches!(
            compose_history_beta_v1(&sites, &history, &beta, &mut budget, &mut check),
            Err(ResidentBabComposeErrorV1::Deadline(
                ny_core::NyError::DeadlineExceeded(_)
            ))
        ));
        assert_eq!(stride_checks, 2);
    }

    #[test]
    fn dense_beta_initialization_deadline_fires_at_second_stride() {
        let beta_len = RESIDENT_BAB_COMPOSE_POLL_STRIDE + 1;
        let sites = [ResidentBabReluSiteV1 {
            topology_node_id: 3,
            node_name: "relu".to_string(),
            preactivation_width: beta_len,
        }];
        let history = GraphSplitHistory::new();
        let beta = GraphBetaState::default();
        let nominal = history_nominal_bytes(sites.len(), 0, beta_len).unwrap();
        let mut budget = ResidentBabHostBudgetV1::begin(
            super::super::ResidentBabAdapterHostCapV1 {
                limit_bytes: 16 << 20,
                resident_bytes_before: 0,
            },
            nominal,
        )
        .unwrap();
        let mut stride_checks = 0usize;
        let mut check = |label| {
            if label == "resident dense beta initialization" {
                stride_checks += 1;
                if stride_checks == 2 {
                    return Err(ny_core::NyError::DeadlineExceeded(
                        "injected dense-beta initialization deadline".to_string(),
                    ));
                }
            }
            Ok(())
        };
        assert!(matches!(
            compose_history_beta_v1(&sites, &history, &beta, &mut budget, &mut check),
            Err(ResidentBabComposeErrorV1::Deadline(
                ny_core::NyError::DeadlineExceeded(_)
            ))
        ));
        assert_eq!(stride_checks, 2);
    }

    #[test]
    fn append_suffix_supports_sixteen_and_uses_oldest_bit_zero() {
        let mut child = Vec::new();
        for index in 0..GPU_BAB_BOUND_MAX_APPEND_SPLITS {
            let phase = if index % 3 == 0 {
                GpuBabBoundSplitHistoryPhase::Active
            } else {
                GpuBabBoundSplitHistoryPhase::Inactive
            };
            child.extend_from_slice(
                &GpuBabBoundSplitHistoryLiteral {
                    phase,
                    topology_node_id: u32::try_from(index).unwrap(),
                    neuron_index: 0,
                    score: index as f32,
                }
                .encode_words()
                .unwrap(),
            );
        }
        let pattern = append_for_test(&[], &child).unwrap();
        let expected = (0..GPU_BAB_BOUND_MAX_APPEND_SPLITS)
            .filter(|index| index % 3 == 0)
            .fold(0u64, |bits, index| bits | (1u64 << index));
        assert_eq!(pattern, expected);
    }

    #[test]
    fn append_suffix_rejects_prefix_mutation_and_seventeen_records() {
        let (history, beta) = history_beta();
        let parent = compose_for_test(&sites(), &history, &beta)
            .unwrap()
            .history_words;
        let mut child = parent.clone();
        child.extend_from_slice(
            &GpuBabBoundSplitHistoryLiteral {
                phase: GpuBabBoundSplitHistoryPhase::Active,
                topology_node_id: 3,
                neuron_index: 0,
                score: 1.0,
            }
            .encode_words()
            .unwrap(),
        );
        let mut wrong_parent = parent.clone();
        wrong_parent[3] ^= 1;
        assert!(append_for_test(&wrong_parent, &child).is_err());

        let record = child[parent.len()..].to_vec();
        let mut too_many = parent.clone();
        for _ in 0..=GPU_BAB_BOUND_MAX_APPEND_SPLITS {
            too_many.extend_from_slice(&record);
        }
        assert!(append_for_test(&parent, &too_many).is_err());
    }

    #[test]
    fn append_suffix_enforces_the_core_history_word_cap_before_prefix_scan() {
        let parent_len =
            GPU_BAB_BOUND_MAX_SPLIT_HISTORY_WORDS - GPU_BAB_BOUND_SPLIT_HISTORY_RECORD_WORDS;
        assert_eq!(
            validate_append_lengths_v1(parent_len, GPU_BAB_BOUND_MAX_SPLIT_HISTORY_WORDS).unwrap(),
            1
        );
        assert!(validate_append_lengths_v1(
            GPU_BAB_BOUND_MAX_SPLIT_HISTORY_WORDS,
            GPU_BAB_BOUND_MAX_SPLIT_HISTORY_WORDS + GPU_BAB_BOUND_SPLIT_HISTORY_RECORD_WORDS,
        )
        .is_err());
    }

    #[test]
    fn append_prefix_validation_deadline_fires_at_second_stride() {
        let parent_len =
            RESIDENT_BAB_COMPOSE_POLL_STRIDE + GPU_BAB_BOUND_SPLIT_HISTORY_RECORD_WORDS;
        let parent = vec![0u32; parent_len];
        let mut child = parent.clone();
        child.extend_from_slice(
            &GpuBabBoundSplitHistoryLiteral {
                phase: GpuBabBoundSplitHistoryPhase::Active,
                topology_node_id: 3,
                neuron_index: 0,
                score: 1.0,
            }
            .encode_words()
            .unwrap(),
        );
        let mut prefix_checks = 0usize;
        let mut suffix_checks = 0usize;
        let mut check = |label| {
            if label == "resident append-history prefix validation" {
                prefix_checks += 1;
                if prefix_checks == 2 {
                    return Err(ny_core::NyError::DeadlineExceeded(
                        "injected append-prefix deadline".to_string(),
                    ));
                }
            } else if label == "resident append-history validation" {
                suffix_checks += 1;
            }
            Ok(())
        };
        assert!(matches!(
            validate_append_suffix_v1(&parent, &child, &mut check),
            Err(ResidentBabComposeErrorV1::Deadline(
                ny_core::NyError::DeadlineExceeded(_)
            ))
        ));
        assert_eq!(prefix_checks, 2);
        assert_eq!(suffix_checks, 0);
    }
}
