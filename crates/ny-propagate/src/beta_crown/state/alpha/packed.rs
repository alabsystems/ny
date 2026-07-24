// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Packed queue representation for graph-domain alpha optimizer state.
//!
//! The runtime representation is optimized for keyed mutation. The queue
//! representation is optimized for resident bytes and is never used directly
//! by bound propagation or optimization.
//!
//! `queue_layout_fingerprint` binds the format version, an immutable
//! graph-local `DomainList` identity, and the deterministic node layout. The
//! identity is a process-local nonce, not a stable model fingerprint. Packed
//! values are crate-private, are created only while `DomainMetadata` enters one
//! `DomainList`, and are eliminated before metadata leaves `pick_out`. They are
//! not serialized, cached, or transferable between verifier/graph instances.
//! If that confinement changes, the format must add a stable graph identity
//! before reuse is allowed.

use std::collections::HashMap;
use std::mem::size_of;

use ny_core::{NyError, Result};
use sha2::{Digest, Sha256};

use super::graph_init::GraphDomainAlphaState;
use super::neuron::AlphaNeuronState;

/// In-memory format version for packed graph alpha queue state.
pub const PACKED_GRAPH_ALPHA_FORMAT_VERSION: u32 = 1;

const QUEUE_LAYOUT_FINGERPRINT_DOMAIN: &[u8] = b"ny-packed-graph-alpha-queue-layout-v1";

/// Which graph alpha representation a byte census describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphAlphaStateRepresentation {
    /// Hash-map runtime representation used by propagation and optimizers.
    Runtime,
    /// Structure-of-arrays representation used while a domain is queued.
    Packed,
}

/// Explicit owned-byte estimate for one graph-domain alpha state.
///
/// Hash-table bytes are estimates based on reported table capacities and one
/// control byte per usable bucket. Packed bytes use the actual vector/string
/// capacities. The census excludes allocator headers, which are
/// implementation-specific.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphAlphaStateByteCensus {
    /// Representation being measured.
    pub representation: GraphAlphaStateRepresentation,
    /// Inline struct and vector/hash-map header bytes.
    pub fixed_bytes: usize,
    /// Node names, node table entries, and packed node offsets.
    pub node_layout_bytes: usize,
    /// Neuron keys/indices at allocated capacity.
    pub neuron_index_bytes: usize,
    /// Six optimizer fields per allocated lower/upper neuron.
    pub value_bytes: usize,
    /// Estimated hash-table control bytes; zero for packed vectors.
    pub hash_control_bytes: usize,
    /// Sum of all census components.
    pub estimated_total_bytes: usize,
}

impl GraphAlphaStateByteCensus {
    fn new(
        representation: GraphAlphaStateRepresentation,
        fixed_bytes: usize,
        node_layout_bytes: usize,
        neuron_index_bytes: usize,
        value_bytes: usize,
        hash_control_bytes: usize,
    ) -> Self {
        let estimated_total_bytes = fixed_bytes
            .saturating_add(node_layout_bytes)
            .saturating_add(neuron_index_bytes)
            .saturating_add(value_bytes)
            .saturating_add(hash_control_bytes);
        Self {
            representation,
            fixed_bytes,
            node_layout_bytes,
            neuron_index_bytes,
            value_bytes,
            hash_control_bytes,
            estimated_total_bytes,
        }
    }

    pub(crate) fn with_additional_fixed_bytes(mut self, bytes: usize) -> Self {
        self.fixed_bytes = self.fixed_bytes.saturating_add(bytes);
        self.estimated_total_bytes = self.estimated_total_bytes.saturating_add(bytes);
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
struct PackedAlphaValues {
    alpha: Vec<f32>,
    grad: Vec<f32>,
    velocity: Vec<f32>,
    adam_m: Vec<f32>,
    adam_v: Vec<f32>,
    adam_v_max: Vec<f32>,
}

impl PackedAlphaValues {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            alpha: Vec::with_capacity(capacity),
            grad: Vec::with_capacity(capacity),
            velocity: Vec::with_capacity(capacity),
            adam_m: Vec::with_capacity(capacity),
            adam_v: Vec::with_capacity(capacity),
            adam_v_max: Vec::with_capacity(capacity),
        }
    }

    fn push(&mut self, state: &AlphaNeuronState) {
        self.alpha.push(state.alpha);
        self.grad.push(state.grad);
        self.velocity.push(state.velocity);
        self.adam_m.push(state.adam_m);
        self.adam_v.push(state.adam_v);
        self.adam_v_max.push(state.adam_v_max);
    }

    fn allocated_bytes(&self) -> usize {
        [
            self.alpha.capacity(),
            self.grad.capacity(),
            self.velocity.capacity(),
            self.adam_m.capacity(),
            self.adam_v.capacity(),
            self.adam_v_max.capacity(),
        ]
        .into_iter()
        .sum::<usize>()
        .saturating_mul(size_of::<f32>())
    }

    fn validate(&self, expected_len: usize, side: &str) -> Result<()> {
        let fields = [
            ("alpha", &self.alpha),
            ("grad", &self.grad),
            ("velocity", &self.velocity),
            ("adam_m", &self.adam_m),
            ("adam_v", &self.adam_v),
            ("adam_v_max", &self.adam_v_max),
        ];
        for (field, values) in fields {
            if values.len() != expected_len {
                return Err(packed_error(format!(
                    "{side}.{field} length {} != layout length {expected_len}",
                    values.len()
                )));
            }
            if values.iter().any(|value| !value.is_finite()) {
                return Err(packed_error(format!(
                    "{side}.{field} contains a non-finite value"
                )));
            }
        }
        if self.alpha.iter().any(|alpha| !(0.0..=1.0).contains(alpha)) {
            return Err(packed_error(format!(
                "{side}.alpha violates the [0,1] relaxation contract"
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
struct PackedGraphAlphaSide {
    node_names: Vec<String>,
    /// Start offset for every node plus a final sentinel.
    node_offsets: Vec<u32>,
    neuron_indices: Vec<u32>,
    values: PackedAlphaValues,
}

impl PackedGraphAlphaSide {
    fn pack(maps: &HashMap<String, HashMap<usize, AlphaNeuronState>>, side: &str) -> Result<Self> {
        let entry_count = maps.values().try_fold(0usize, |total, neurons| {
            total
                .checked_add(neurons.len())
                .ok_or_else(|| packed_error(format!("{side} entry count overflow")))
        })?;
        if entry_count > u32::MAX as usize {
            return Err(packed_error(format!(
                "{side} has {entry_count} entries, exceeding the u32 packed offset format"
            )));
        }

        let mut ordered_nodes: Vec<&String> = maps.keys().collect();
        ordered_nodes.sort_unstable();
        let mut node_names = Vec::with_capacity(ordered_nodes.len());
        let mut node_offsets = Vec::with_capacity(ordered_nodes.len() + 1);
        let mut neuron_indices = Vec::with_capacity(entry_count);
        let mut values = PackedAlphaValues::with_capacity(entry_count);
        node_offsets.push(0);

        for node_name in ordered_nodes {
            node_names.push(node_name.clone());
            let neuron_map = maps
                .get(node_name)
                .expect("node name was collected from the same map");
            let mut ordered_neurons: Vec<usize> = neuron_map.keys().copied().collect();
            ordered_neurons.sort_unstable();
            for neuron_idx in ordered_neurons {
                let packed_idx = u32::try_from(neuron_idx).map_err(|_| {
                    packed_error(format!(
                        "{side}.{node_name} neuron index {neuron_idx} exceeds u32"
                    ))
                })?;
                neuron_indices.push(packed_idx);
                values.push(
                    neuron_map
                        .get(&neuron_idx)
                        .expect("neuron index was collected from the same map"),
                );
            }
            node_offsets.push(
                u32::try_from(neuron_indices.len())
                    .map_err(|_| packed_error(format!("{side} packed offset exceeds u32")))?,
            );
        }

        Ok(Self {
            node_names,
            node_offsets,
            neuron_indices,
            values,
        })
    }

    fn validate(&self, side: &str) -> Result<()> {
        if self.node_offsets.len() != self.node_names.len().saturating_add(1) {
            return Err(packed_error(format!(
                "{side} node offset count {} != node count {} + sentinel",
                self.node_offsets.len(),
                self.node_names.len()
            )));
        }
        if self.node_offsets.first().copied() != Some(0) {
            return Err(packed_error(format!(
                "{side} node offsets must start at zero"
            )));
        }
        if self
            .node_offsets
            .last()
            .copied()
            .map(|value| value as usize)
            != Some(self.neuron_indices.len())
        {
            return Err(packed_error(format!(
                "{side} final node offset does not match neuron index length"
            )));
        }
        if self.node_names.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(packed_error(format!(
                "{side} node names are not strictly sorted and unique"
            )));
        }
        for (node_idx, offsets) in self.node_offsets.windows(2).enumerate() {
            let start = offsets[0] as usize;
            let end = offsets[1] as usize;
            if start > end || end > self.neuron_indices.len() {
                return Err(packed_error(format!(
                    "{side} node {} has invalid offset range {start}..{end}",
                    self.node_names[node_idx]
                )));
            }
            if self.neuron_indices[start..end]
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            {
                return Err(packed_error(format!(
                    "{side} node {} neuron indices are not strictly sorted and unique",
                    self.node_names[node_idx]
                )));
            }
        }
        self.values.validate(self.neuron_indices.len(), side)
    }

    fn update_fingerprint(&self, hasher: &mut Sha256, side: &[u8]) {
        hasher.update((side.len() as u64).to_le_bytes());
        hasher.update(side);
        hasher.update((self.node_names.len() as u64).to_le_bytes());
        for node_name in &self.node_names {
            hasher.update((node_name.len() as u64).to_le_bytes());
            hasher.update(node_name.as_bytes());
        }
        hasher.update((self.node_offsets.len() as u64).to_le_bytes());
        for offset in &self.node_offsets {
            hasher.update(offset.to_le_bytes());
        }
        hasher.update((self.neuron_indices.len() as u64).to_le_bytes());
        for neuron_idx in &self.neuron_indices {
            hasher.update(neuron_idx.to_le_bytes());
        }
    }

    fn unpack(&self) -> HashMap<String, HashMap<usize, AlphaNeuronState>> {
        let mut maps = HashMap::with_capacity(self.node_names.len());
        for (node_idx, node_name) in self.node_names.iter().enumerate() {
            let start = self.node_offsets[node_idx] as usize;
            let end = self.node_offsets[node_idx + 1] as usize;
            let mut neurons = HashMap::with_capacity(end - start);
            for packed_idx in start..end {
                neurons.insert(
                    self.neuron_indices[packed_idx] as usize,
                    AlphaNeuronState {
                        alpha: self.values.alpha[packed_idx],
                        grad: self.values.grad[packed_idx],
                        velocity: self.values.velocity[packed_idx],
                        adam_m: self.values.adam_m[packed_idx],
                        adam_v: self.values.adam_v[packed_idx],
                        adam_v_max: self.values.adam_v_max[packed_idx],
                    },
                );
            }
            maps.insert(node_name.clone(), neurons);
        }
        maps
    }

    fn layout_allocated_bytes(&self) -> usize {
        let node_headers = self
            .node_names
            .capacity()
            .saturating_mul(size_of::<String>());
        let node_text = self.node_names.iter().map(String::capacity).sum::<usize>();
        let offsets = self
            .node_offsets
            .capacity()
            .saturating_mul(size_of::<u32>());
        node_headers
            .saturating_add(node_text)
            .saturating_add(offsets)
    }
}

/// Compact, deterministic structure-of-arrays representation used in DomainList.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PackedGraphDomainAlphaState {
    format_version: u32,
    queue_identity: u64,
    queue_layout_fingerprint: [u8; 32],
    lower: PackedGraphAlphaSide,
    upper: PackedGraphAlphaSide,
}

impl PackedGraphDomainAlphaState {
    /// Pack a runtime graph alpha state without changing any field value.
    pub(crate) fn pack(runtime: &GraphDomainAlphaState, queue_identity: u64) -> Result<Self> {
        let lower = PackedGraphAlphaSide::pack(&runtime.neurons, "lower")?;
        let upper = PackedGraphAlphaSide::pack(&runtime.upper_neurons, "upper")?;
        let mut packed = Self {
            format_version: PACKED_GRAPH_ALPHA_FORMAT_VERSION,
            queue_identity,
            queue_layout_fingerprint: [0; 32],
            lower,
            upper,
        };
        packed.queue_layout_fingerprint = packed.compute_queue_layout_fingerprint();
        packed.validate(queue_identity)?;
        Ok(packed)
    }

    /// Validate queue ownership, format, deterministic layout, field lengths, and values.
    pub(crate) fn validate(&self, expected_queue_identity: u64) -> Result<()> {
        if self.format_version != PACKED_GRAPH_ALPHA_FORMAT_VERSION {
            return Err(packed_error(format!(
                "format version {} != supported version {}",
                self.format_version, PACKED_GRAPH_ALPHA_FORMAT_VERSION
            )));
        }
        if self.queue_identity == 0 || self.queue_identity != expected_queue_identity {
            return Err(packed_error(format!(
                "queue identity {} != expected graph-local queue identity {expected_queue_identity}",
                self.queue_identity
            )));
        }
        self.lower.validate("lower")?;
        self.upper.validate("upper")?;
        let observed = self.compute_queue_layout_fingerprint();
        if observed != self.queue_layout_fingerprint {
            return Err(packed_error(
                "queue/layout fingerprint does not match format, identity, and node layout",
            ));
        }
        Ok(())
    }

    /// Validate and reconstruct the mutable hash-map optimizer representation.
    pub(crate) fn unpack(&self, expected_queue_identity: u64) -> Result<GraphDomainAlphaState> {
        self.validate(expected_queue_identity)?;
        Ok(GraphDomainAlphaState {
            neurons: self.lower.unpack(),
            upper_neurons: self.upper.unpack(),
        })
    }

    /// Packed format version carried by this value.
    #[cfg(test)]
    pub(crate) fn format_version(&self) -> u32 {
        self.format_version
    }

    /// Graph-local DomainList identity carried by this packed state.
    #[cfg(test)]
    pub(crate) fn queue_identity(&self) -> u64 {
        self.queue_identity
    }

    /// SHA-256 fingerprint of version, queue identity, nodes, offsets, and indices.
    #[cfg(test)]
    pub(crate) fn queue_layout_fingerprint(&self) -> [u8; 32] {
        self.queue_layout_fingerprint
    }

    /// Owned-byte census for the packed representation.
    pub(crate) fn byte_census(&self) -> GraphAlphaStateByteCensus {
        let fixed_bytes = size_of::<Self>();
        let node_layout_bytes = self
            .lower
            .layout_allocated_bytes()
            .saturating_add(self.upper.layout_allocated_bytes());
        let neuron_index_bytes = self
            .lower
            .neuron_indices
            .capacity()
            .saturating_add(self.upper.neuron_indices.capacity())
            .saturating_mul(size_of::<u32>());
        let value_bytes = self
            .lower
            .values
            .allocated_bytes()
            .saturating_add(self.upper.values.allocated_bytes());
        GraphAlphaStateByteCensus::new(
            GraphAlphaStateRepresentation::Packed,
            fixed_bytes,
            node_layout_bytes,
            neuron_index_bytes,
            value_bytes,
            0,
        )
    }

    #[cfg(test)]
    pub(crate) fn corrupt_queue_layout_fingerprint_for_test(&mut self) {
        self.queue_layout_fingerprint[0] ^= 1;
    }

    fn compute_queue_layout_fingerprint(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(QUEUE_LAYOUT_FINGERPRINT_DOMAIN);
        hasher.update(self.format_version.to_le_bytes());
        hasher.update(self.queue_identity.to_le_bytes());
        self.lower.update_fingerprint(&mut hasher, b"lower");
        self.upper.update_fingerprint(&mut hasher, b"upper");
        hasher.finalize().into()
    }
}

impl GraphDomainAlphaState {
    /// Estimated owned bytes for the mutable hash-map runtime representation.
    pub fn runtime_byte_census(&self) -> GraphAlphaStateByteCensus {
        let mut node_layout_bytes = 0usize;
        let mut neuron_index_bytes = 0usize;
        let mut value_bytes = 0usize;
        let mut hash_control_bytes = 0usize;

        for side in [&self.neurons, &self.upper_neurons] {
            node_layout_bytes = node_layout_bytes.saturating_add(side.capacity().saturating_mul(
                size_of::<String>().saturating_add(size_of::<HashMap<usize, AlphaNeuronState>>()),
            ));
            hash_control_bytes = hash_control_bytes.saturating_add(side.capacity());
            for (node_name, neurons) in side {
                node_layout_bytes = node_layout_bytes.saturating_add(node_name.capacity());
                neuron_index_bytes = neuron_index_bytes
                    .saturating_add(neurons.capacity().saturating_mul(size_of::<usize>()));
                value_bytes = value_bytes.saturating_add(
                    neurons
                        .capacity()
                        .saturating_mul(size_of::<AlphaNeuronState>()),
                );
                hash_control_bytes = hash_control_bytes.saturating_add(neurons.capacity());
            }
        }

        GraphAlphaStateByteCensus::new(
            GraphAlphaStateRepresentation::Runtime,
            size_of::<Self>(),
            node_layout_bytes,
            neuron_index_bytes,
            value_bytes,
            hash_control_bytes,
        )
    }
}

fn packed_error(detail: impl Into<String>) -> NyError {
    NyError::InvalidSpec(format!(
        "packed graph alpha queue state invalid: {}",
        detail.into()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::beta_crown::config::AdaptiveOptConfig;

    const QUEUE_ID: u64 = 17;

    fn neuron(seed: f32) -> AlphaNeuronState {
        AlphaNeuronState {
            alpha: seed,
            grad: -seed,
            velocity: seed * 2.0,
            adam_m: -seed * 3.0,
            adam_v: seed * seed,
            adam_v_max: seed * seed + 0.25,
        }
    }

    fn assert_neuron_bits_eq(left: &AlphaNeuronState, right: &AlphaNeuronState) {
        assert_eq!(left.alpha.to_bits(), right.alpha.to_bits());
        assert_eq!(left.grad.to_bits(), right.grad.to_bits());
        assert_eq!(left.velocity.to_bits(), right.velocity.to_bits());
        assert_eq!(left.adam_m.to_bits(), right.adam_m.to_bits());
        assert_eq!(left.adam_v.to_bits(), right.adam_v.to_bits());
        assert_eq!(left.adam_v_max.to_bits(), right.adam_v_max.to_bits());
    }

    fn assert_state_bits_eq(left: &GraphDomainAlphaState, right: &GraphDomainAlphaState) {
        for (left_side, right_side) in [
            (&left.neurons, &right.neurons),
            (&left.upper_neurons, &right.upper_neurons),
        ] {
            assert_eq!(left_side.len(), right_side.len());
            for (node_name, left_neurons) in left_side {
                let right_neurons = right_side
                    .get(node_name)
                    .expect("roundtrip must preserve every node");
                assert_eq!(left_neurons.len(), right_neurons.len());
                for (neuron_idx, left_neuron) in left_neurons {
                    let right_neuron = right_neurons
                        .get(neuron_idx)
                        .expect("roundtrip must preserve every neuron index");
                    assert_neuron_bits_eq(left_neuron, right_neuron);
                }
            }
        }
    }

    #[test]
    fn roundtrip_preserves_all_fields_and_has_deterministic_layout() {
        let mut state = GraphDomainAlphaState::empty();
        state
            .neurons
            .entry("relu-z".to_string())
            .or_default()
            .insert(9, neuron(0.75));
        state
            .neurons
            .entry("relu-a".to_string())
            .or_default()
            .insert(3, neuron(0.25));
        state
            .neurons
            .entry("relu-a".to_string())
            .or_default()
            .insert(1, neuron(-0.0));
        state
            .upper_neurons
            .entry("relu-a".to_string())
            .or_default()
            .insert(7, neuron(0.5));
        state
            .upper_neurons
            .insert("empty-node".to_string(), HashMap::new());

        let packed = PackedGraphDomainAlphaState::pack(&state, QUEUE_ID).unwrap();
        let packed_again = PackedGraphDomainAlphaState::pack(&state, QUEUE_ID).unwrap();
        assert_eq!(packed, packed_again);
        assert_eq!(packed.format_version(), PACKED_GRAPH_ALPHA_FORMAT_VERSION);
        assert_eq!(packed.queue_identity(), QUEUE_ID);
        assert_ne!(packed.queue_layout_fingerprint(), [0; 32]);

        let restored = packed.unpack(QUEUE_ID).unwrap();
        assert_state_bits_eq(&state, &restored);
        assert_eq!(
            restored.neuron("relu-a", 1).unwrap().alpha.to_bits(),
            (-0.0f32).to_bits(),
            "valid alpha bits must not be re-sanitized during unpack"
        );
    }

    #[test]
    fn roundtrip_preserves_exact_adam_trajectory() {
        let mut runtime = GraphDomainAlphaState::empty();
        for node in ["relu-b", "relu-a"] {
            for idx in [9, 2, 5] {
                runtime
                    .neurons
                    .entry(node.to_string())
                    .or_default()
                    .insert(idx, neuron(0.2 + idx as f32 * 0.01));
                runtime
                    .upper_neurons
                    .entry(node.to_string())
                    .or_default()
                    .insert(idx, neuron(0.7 - idx as f32 * 0.01));
            }
        }
        let mut restored = PackedGraphDomainAlphaState::pack(&runtime, QUEUE_ID)
            .unwrap()
            .unpack(QUEUE_ID)
            .unwrap();
        let config = AdaptiveOptConfig::default();

        for step in 1..=4 {
            for node in ["relu-a", "relu-b"] {
                for idx in [2, 5, 9] {
                    let grad = (step as f32 * 0.125) - idx as f32 * 0.01;
                    runtime.accumulate_grad(node, idx, grad);
                    runtime.accumulate_grad_upper(node, idx, -grad);
                    restored.accumulate_grad(node, idx, grad);
                    restored.accumulate_grad_upper(node, idx, -grad);
                }
            }
            assert_eq!(
                runtime.gradient_step_adam(&config, step).to_bits(),
                restored.gradient_step_adam(&config, step).to_bits()
            );
            assert_state_bits_eq(&runtime, &restored);
            runtime.zero_grad();
            restored.zero_grad();
        }
    }

    #[test]
    fn validation_refuses_layout_and_value_corruption() {
        let mut state = GraphDomainAlphaState::empty();
        state.insert("relu".to_string(), 4, AlphaNeuronState::new(0.4));
        let packed = PackedGraphDomainAlphaState::pack(&state, QUEUE_ID).unwrap();

        let mut bad_version = packed.clone();
        bad_version.format_version += 1;
        assert!(bad_version.validate(QUEUE_ID).is_err());

        assert!(
            packed.validate(QUEUE_ID + 1).is_err(),
            "a packed value must be bound to its originating DomainList"
        );

        let mut bad_layout = packed.clone();
        bad_layout.lower.neuron_indices[0] += 1;
        assert!(bad_layout.validate(QUEUE_ID).is_err());

        let mut bad_alpha = packed;
        bad_alpha.upper.values.alpha[0] = f32::NAN;
        assert!(bad_alpha.validate(QUEUE_ID).is_err());
        assert!(bad_alpha.unpack(QUEUE_ID).is_err());
    }
}
