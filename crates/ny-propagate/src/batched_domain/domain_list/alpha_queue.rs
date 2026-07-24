// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! DomainMetadata-only adapter between runtime and packed graph alpha state.

use ny_core::{NyError, Result};
use std::mem::size_of;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::beta_crown::state::{
    GraphAlphaStateByteCensus, GraphAlphaStateRepresentation, GraphDomainAlphaState,
    PackedGraphDomainAlphaState,
};

pub(super) const PACKED_GRAPH_ALPHA_QUEUE_ENV: &str = "NY_PACKED_GRAPH_ALPHA_QUEUE";

static NEXT_GRAPH_LOCAL_QUEUE_IDENTITY: AtomicU64 = AtomicU64::new(1);

pub(super) fn packed_graph_alpha_queue_enabled() -> bool {
    std::env::var(PACKED_GRAPH_ALPHA_QUEUE_ENV).ok().as_deref() == Some("1")
}

/// Allocate an immutable process-local identity for one graph-local DomainList.
pub(super) fn allocate_graph_local_queue_identity() -> Result<u64> {
    NEXT_GRAPH_LOCAL_QUEUE_IDENTITY
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |identity| {
            identity.checked_add(1)
        })
        .map_err(|_| {
            NyError::InternalError(
                "exhausted graph-local DomainList alpha queue identities".to_string(),
            )
        })
}

/// Exactly one canonical alpha state is owned by DomainMetadata at a time.
#[derive(Debug, Clone)]
// Runtime stays inline so gate-off enqueue remains allocation-free. Boxing it
// would add a heap allocation solely to equalize variant sizes; packed is
// already boxed, so the enum itself is no larger than the runtime payload.
#[allow(clippy::large_enum_variant)]
pub(crate) enum QueuedGraphAlphaState {
    Runtime(GraphDomainAlphaState),
    Packed(Box<PackedGraphDomainAlphaState>),
}

impl QueuedGraphAlphaState {
    pub(super) fn pack_for_queue(&mut self, queue_identity: u64) -> Result<()> {
        match self {
            Self::Runtime(runtime) => {
                let packed = PackedGraphDomainAlphaState::pack(runtime, queue_identity)?;
                *self = Self::Packed(Box::new(packed));
            }
            Self::Packed(packed) => packed.validate(queue_identity)?,
        }
        Ok(())
    }

    pub(super) fn validate(&self, queue_identity: u64) -> Result<()> {
        match self {
            Self::Runtime(_) => Ok(()),
            Self::Packed(packed) => packed.validate(queue_identity),
        }
    }

    pub(super) fn unpack_after_dequeue(&mut self, queue_identity: u64) -> Result<()> {
        if let Self::Packed(packed) = self {
            let runtime = packed.unpack(queue_identity)?;
            *self = Self::Runtime(runtime);
        }
        Ok(())
    }

    pub(super) fn runtime(&self) -> Option<&GraphDomainAlphaState> {
        match self {
            Self::Runtime(runtime) => Some(runtime),
            Self::Packed(_) => None,
        }
    }

    pub(super) fn representation(&self) -> GraphAlphaStateRepresentation {
        match self {
            Self::Runtime(_) => GraphAlphaStateRepresentation::Runtime,
            Self::Packed(_) => GraphAlphaStateRepresentation::Packed,
        }
    }

    pub(super) fn byte_census(&self) -> GraphAlphaStateByteCensus {
        let metadata_slot_bytes = size_of::<Option<Self>>();
        match self {
            Self::Runtime(runtime) => runtime.runtime_byte_census().with_additional_fixed_bytes(
                metadata_slot_bytes.saturating_sub(size_of::<GraphDomainAlphaState>()),
            ),
            Self::Packed(packed) => packed
                .byte_census()
                .with_additional_fixed_bytes(metadata_slot_bytes),
        }
    }

    #[cfg(test)]
    pub(super) fn corrupt_packed_layout_for_test(&mut self) {
        if let Self::Packed(packed) = self {
            packed.corrupt_queue_layout_fingerprint_for_test();
        }
    }
}

impl From<GraphDomainAlphaState> for QueuedGraphAlphaState {
    fn from(runtime: GraphDomainAlphaState) -> Self {
        Self::Runtime(runtime)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::beta_crown::state::AlphaNeuronState;

    #[test]
    fn alpha_heavy_queue_state_estimates_at_least_twenty_five_percent_reduction() {
        let mut state = GraphDomainAlphaState::empty();
        for node_idx in 0..8 {
            let node_name = format!("residual-block-{node_idx:02}-relu");
            for neuron_idx in 0..1024 {
                state.insert(
                    node_name.clone(),
                    neuron_idx,
                    AlphaNeuronState::new((neuron_idx % 101) as f32 / 100.0),
                );
            }
        }

        let mut queued_state = QueuedGraphAlphaState::Runtime(state);
        let runtime = queued_state.byte_census();
        queued_state.pack_for_queue(29).unwrap();
        let packed = queued_state.byte_census();
        let reduction_percent = 100.0
            * (1.0 - packed.estimated_total_bytes as f64 / runtime.estimated_total_bytes as f64);
        eprintln!(
            "graph-alpha queue census: runtime={} packed={} reduction={reduction_percent:.2}%",
            runtime.estimated_total_bytes, packed.estimated_total_bytes
        );
        assert_eq!(
            runtime.representation,
            GraphAlphaStateRepresentation::Runtime
        );
        assert_eq!(packed.representation, GraphAlphaStateRepresentation::Packed);
        assert!(
            packed.estimated_total_bytes * 4 <= runtime.estimated_total_bytes * 3,
            "packed={} runtime={} reduction={:.1}%",
            packed.estimated_total_bytes,
            runtime.estimated_total_bytes,
            reduction_percent
        );
    }
}
