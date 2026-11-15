// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! One-traversal observations of an immutable original decoder block.
//!
//! Only the graph input boundary and selected output are changed. Original
//! layer specifications, coefficients, positional tables and their identities
//! remain borrowed from the lowering. These observations carry no proof or
//! deployed-kernel qualification.

use super::model::{linear_census, point_identity};
use super::{add, mul, DecoderContractRefusal as Refusal, DecoderOptions, LoweredDecoder, Result};
use crate::{DataType, TensorSpec};
use ndarray::ArrayD;
use ny_propagate::certify::work::{
    LeaseClass, PersistentReservation, WorkBudget, WorkLease, WorkRequest,
};
use ny_propagate::network::GraphPointTrace;
use ny_propagate::GraphNetwork;
use ny_tensor::BoundedTensor;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::time::Instant;

const MAX_NAME_BYTES: usize = 4096;
const MAX_RANK: usize = 16;

#[derive(Debug, Clone, Serialize)]
pub struct ObservationTensorIdentity {
    pub node: String,
    pub tensor: String,
    pub original_shape: Vec<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ObservationBoundary {
    pub schema_version: u32,
    pub certificate_authority: bool,
    pub original_lowering_digest: [u8; 32],
    pub original_block_digest: [u8; 32],
    pub subset_digest: [u8; 32],
    pub reference_profile: String,
    pub block_index: usize,
    pub input_tensor: String,
    pub input_shape: Vec<usize>,
    pub positions: Vec<u64>,
    pub outputs: Vec<ObservationTensorIdentity>,
}

/// Owns one converted original block and its admitted resident metadata.
/// Compact Linear owners retain their original shared-ledger reservations.
#[derive(Debug)]
pub struct DecoderBlockObserver {
    graph: GraphNetwork,
    boundary: ObservationBoundary,
    budget: WorkBudget,
    graph_reservation: PersistentReservation,
    metadata_reservation: PersistentReservation,
    linear_limit: usize,
    point_scratch: usize,
    point_work: u64,
    trace_bytes: usize,
    output_elements: usize,
    result_bytes: usize,
}

#[derive(Debug)]
pub struct ObservedDecoderTensor {
    identity: ObservationTensorIdentity,
    values: ArrayD<f32>,
    value_digest: [u8; 32],
}
impl ObservedDecoderTensor {
    pub fn identity(&self) -> &ObservationTensorIdentity {
        &self.identity
    }
    pub fn values(&self) -> &ArrayD<f32> {
        &self.values
    }
    pub fn value_digest(&self) -> &[u8; 32] {
        &self.value_digest
    }
}

#[derive(Debug)]
pub struct DecoderBlockObservation {
    boundary: ObservationBoundary,
    tensors: Vec<ObservedDecoderTensor>,
    trace: GraphPointTrace,
    input_digest: [u8; 32],
    outputs_digest: [u8; 32],
    execution_digest: [u8; 32],
    reservation: PersistentReservation,
}
impl DecoderBlockObservation {
    pub fn boundary(&self) -> &ObservationBoundary {
        &self.boundary
    }
    /// Original lowering order, independent of the graph's topological order.
    pub fn tensors(&self) -> &[ObservedDecoderTensor] {
        &self.tensors
    }
    pub fn execution_trace(&self) -> &GraphPointTrace {
        &self.trace
    }
    pub fn input_digest(&self) -> &[u8; 32] {
        &self.input_digest
    }
    pub fn outputs_digest(&self) -> &[u8; 32] {
        &self.outputs_digest
    }
    pub fn execution_digest(&self) -> &[u8; 32] {
        &self.execution_digest
    }
    pub fn reserved_bytes(&self) -> usize {
        self.reservation.bytes()
    }
}

fn deadline(options: &DecoderOptions, lease: Option<Instant>) -> Option<Instant> {
    [
        options.deadline,
        lease,
        ny_core::instance_budget::deadline(),
    ]
    .into_iter()
    .flatten()
    .min()
}

fn shape_matches(actual: &[usize], original: &[usize]) -> bool {
    actual == original || (original.first() == Some(&1) && actual == &original[1..])
}

fn elements(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |n, &d| mul(n, d))
}

fn copy_identity(
    source: &ObservationTensorIdentity,
    lease: &WorkLease,
    limit: Option<Instant>,
) -> Result<ObservationTensorIdentity> {
    point_identity::poll(lease, limit)?;
    Ok(ObservationTensorIdentity {
        node: point_identity::copy_name(&source.node, lease, limit)?,
        tensor: point_identity::copy_name(&source.tensor, lease, limit)?,
        original_shape: source.original_shape.clone(),
    })
}

fn copy_positions(source: &[u64], lease: &WorkLease, limit: Option<Instant>) -> Result<Vec<u64>> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(source.len())
        .map_err(|e| Refusal::Build(format!("position metadata allocation: {e}")))?;
    for chunk in source.chunks(256) {
        point_identity::poll(lease, limit)?;
        result.extend_from_slice(chunk);
    }
    Ok(result)
}

fn copy_boundary(
    source: &ObservationBoundary,
    lease: &WorkLease,
    limit: Option<Instant>,
) -> Result<ObservationBoundary> {
    let mut outputs = Vec::new();
    outputs
        .try_reserve_exact(source.outputs.len())
        .map_err(|e| Refusal::Build(format!("boundary metadata allocation: {e}")))?;
    for output in &source.outputs {
        outputs.push(copy_identity(output, lease, limit)?);
    }
    Ok(ObservationBoundary {
        schema_version: source.schema_version,
        certificate_authority: false,
        original_lowering_digest: source.original_lowering_digest,
        original_block_digest: source.original_block_digest,
        subset_digest: source.subset_digest,
        reference_profile: point_identity::copy_name(&source.reference_profile, lease, limit)?,
        block_index: source.block_index,
        input_tensor: point_identity::copy_name(&source.input_tensor, lease, limit)?,
        input_shape: source.input_shape.clone(),
        positions: copy_positions(&source.positions, lease, limit)?,
        outputs,
    })
}

fn hash_field(
    hash: &mut Sha256,
    bytes: &[u8],
    lease: &WorkLease,
    limit: Option<Instant>,
) -> Result<()> {
    hash.update((bytes.len() as u64).to_le_bytes());
    for chunk in bytes.chunks(4096) {
        point_identity::poll(lease, limit)?;
        hash.update(chunk);
    }
    Ok(())
}

fn boundary_digest(
    boundary: &ObservationBoundary,
    lease: &WorkLease,
    limit: Option<Instant>,
) -> Result<[u8; 32]> {
    let mut hash = Sha256::new();
    hash_field(
        &mut hash,
        b"ny-diagnostic-original-decoder-block-subset/v1",
        lease,
        limit,
    )?;
    hash.update(boundary.schema_version.to_le_bytes());
    hash.update(boundary.original_lowering_digest);
    hash.update(boundary.original_block_digest);
    hash.update((boundary.block_index as u64).to_le_bytes());
    hash_field(
        &mut hash,
        boundary.reference_profile.as_bytes(),
        lease,
        limit,
    )?;
    hash_field(&mut hash, boundary.input_tensor.as_bytes(), lease, limit)?;
    hash.update((boundary.input_shape.len() as u64).to_le_bytes());
    for &d in &boundary.input_shape {
        hash.update((d as u64).to_le_bytes());
    }
    hash.update((boundary.positions.len() as u64).to_le_bytes());
    for positions in boundary.positions.chunks(256) {
        point_identity::poll(lease, limit)?;
        for &position in positions {
            hash.update(position.to_le_bytes());
        }
    }
    hash.update((boundary.outputs.len() as u64).to_le_bytes());
    for output in &boundary.outputs {
        hash_field(&mut hash, output.node.as_bytes(), lease, limit)?;
        hash_field(&mut hash, output.tensor.as_bytes(), lease, limit)?;
        hash.update((output.original_shape.len() as u64).to_le_bytes());
        for &d in &output.original_shape {
            hash.update((d as u64).to_le_bytes());
        }
    }
    point_identity::poll(lease, limit)?;
    Ok(hash.finalize().into())
}

impl LoweredDecoder {
    /// Prepare ALL original intermediate outputs of one decoder block.
    /// The caller supplies that block's input coordinates, rather than tokens.
    /// Complete ordinary graph construction and evaluation remain unchanged.
    pub fn observe_block(
        &self,
        block_index: usize,
        options: &DecoderOptions,
    ) -> Result<DecoderBlockObserver> {
        super::checkpoint(options)?;
        if block_index >= self.block_outputs.len() {
            return Err(Refusal::Tensor("observation block index".into()));
        }
        // This first lease pays metadata inspection even when later admission
        // refuses. Closed lowering names/ranks are checked before any copying.
        let inspect_work = add(
            mul(self.graph_model.network.layers.len(), 32)?,
            mul(
                options.max_nodes.min(self.graph_model.network.layers.len()),
                8192,
            )?,
        )? as u64;
        let inspect_request = WorkRequest {
            scratch_bytes: 4096,
            work: inspect_work,
            ..WorkRequest::default()
        };
        super::admit(options, &inspect_request)?;
        let mut inspection = self
            .budget
            .lease_for(LeaseClass::Optional, inspect_request)
            .map_err(Refusal::Budget)?;
        let limit = deadline(options, inspection.deadline());
        let block_name = format!("block.{block_index}");
        let block = self
            .manifest
            .blocks
            .iter()
            .find(|b| b.name == block_name)
            .ok_or_else(|| Refusal::Tensor("original block manifest missing".into()))?;
        let first = block
            .operations
            .first()
            .ok_or_else(|| Refusal::Tensor("empty observation block".into()))?;
        let start = self
            .graph_model
            .network
            .layers
            .iter()
            .position(|s| s.name == first.name)
            .ok_or_else(|| Refusal::Tensor("original block layer missing".into()))?;
        let end = add(start, block.operations.len())?;
        let layers = self
            .graph_model
            .network
            .layers
            .get(start..end)
            .ok_or_else(|| Refusal::Tensor("original block layer range".into()))?;
        let input_name = if block_index == 0 {
            "embedded_tokens"
        } else {
            &self.block_outputs[block_index - 1]
        };
        let mut metadata = 32 * 1024usize;
        let mut output_elements = 0usize;
        for (spec, op) in layers.iter().zip(&block.operations) {
            point_identity::poll(&inspection, limit)?;
            if spec.name != op.name
                || spec.outputs.len() != 1
                || spec.outputs[0] != op.output
                || spec.inputs != op.inputs
                || spec.name.len() > MAX_NAME_BYTES
                || op.output.len() > MAX_NAME_BYTES
                || op.output_shape.len() > MAX_RANK
            {
                return Err(Refusal::Tensor(
                    "original observation operation mismatch".into(),
                ));
            }
            let shape = self
                .graph_model
                .tensor_shapes
                .get(&op.output)
                .ok_or_else(|| Refusal::Tensor("observation shape missing".into()))?;
            if shape != &op.output_shape || shape.iter().any(|&d| d <= 0) {
                return Err(Refusal::Tensor(
                    "original observation shape mismatch".into(),
                ));
            }
            let count = shape.iter().try_fold(1usize, |n, &d| {
                mul(
                    n,
                    usize::try_from(d).map_err(|_| Refusal::Resource("observation shape"))?,
                )
            })?;
            output_elements = add(output_elements, count)?;
            metadata = add(
                metadata,
                add(1024, mul(add(spec.name.len(), op.output.len())?, 4)?)?,
            )?;
            // Every external dynamic input is the exact archived boundary;
            // weights/constants are the unchanged original immutable store.
            for name in &spec.inputs {
                point_identity::poll(&inspection, limit)?;
                if name != input_name
                    && self.graph_model.weights.get(name).is_none()
                    && !block.operations.iter().any(|p| p.output == *name)
                {
                    return Err(Refusal::Tensor("unbound observation block input".into()));
                }
            }
        }
        metadata = add(metadata, mul(self.sequence, 16)?)?;
        metadata = add(
            metadata,
            mul(self.manifest.parameters.reference_profile.len(), 4)?,
        )?;
        let request = WorkRequest {
            scratch_bytes: metadata,
            persistent_bytes: metadata,
            work: mul(metadata, 128)? as u64,
            proposals: 0,
        };
        super::admit(options, &request)?;
        let mut lease = self
            .budget
            .lease_for(LeaseClass::Optional, request)
            .map_err(Refusal::Budget)?;
        let limit = deadline(options, lease.deadline());
        let mut outputs = Vec::new();
        outputs
            .try_reserve_exact(layers.len())
            .map_err(|e| Refusal::Build(format!("observation metadata allocation: {e}")))?;
        for op in &block.operations {
            point_identity::poll(&lease, limit)?;
            outputs.push(ObservationTensorIdentity {
                node: point_identity::copy_name(&op.name, &lease, limit)?,
                tensor: point_identity::copy_name(&op.output, &lease, limit)?,
                original_shape: op.output_shape.iter().map(|&d| d as usize).collect(),
            });
        }
        let mut boundary = ObservationBoundary {
            schema_version: 1,
            certificate_authority: false,
            original_lowering_digest: self.manifest.lowering_digest,
            original_block_digest: block.digest,
            subset_digest: [0; 32],
            reference_profile: point_identity::copy_name(
                &self.manifest.parameters.reference_profile,
                &lease,
                limit,
            )?,
            block_index,
            input_tensor: point_identity::copy_name(input_name, &lease, limit)?,
            input_shape: vec![1, self.sequence, self.manifest.parameters.hidden_size],
            positions: copy_positions(&self.manifest.positions, &lease, limit)?,
            outputs,
        };
        boundary.subset_digest = boundary_digest(&boundary, &lease, limit)?;
        let input_spec = [TensorSpec {
            name: boundary.input_tensor.clone(),
            shape: boundary.input_shape.iter().map(|&d| d as i64).collect(),
            dtype: DataType::Float32,
        }];
        let final_output = boundary.outputs.last().expect("nonempty original block");
        let output_spec = [TensorSpec {
            name: final_output.tensor.clone(),
            shape: final_output
                .original_shape
                .iter()
                .map(|&d| d as i64)
                .collect(),
            dtype: DataType::Float32,
        }];
        let mut inputs = self.graph_model.graph_build_inputs();
        inputs.layers = layers;
        inputs.inputs = &input_spec;
        inputs.outputs = &output_spec;
        // Same policy/ledger/census as ordinary graph construction. The full
        // original plan is a conservative admission for this smaller sub-DAG.
        let (graph, graph_reservation, plan) = self.build_observation_graph(&inputs, options)?;
        for output in &boundary.outputs {
            point_identity::poll(&lease, limit)?;
            if !graph.contains_node(&output.node) {
                return Err(Refusal::Unsupported(format!(
                    "original observation node omitted during conversion: {}",
                    output.node,
                )));
            }
        }
        let trace = graph
            .point_trace_admission(None, limit)
            .map_err(|e| Refusal::Build(e.to_string()))?;
        let point_scratch = add(plan.point_scratch, trace.scratch_bytes)?;
        // All output payloads, both temporary slot and result carriers, the
        // copied boundary, and trace survive under the one execution lease.
        let result_bytes = add(
            add(mul(output_elements, 4)?, mul(metadata, 2)?)?,
            add(
                trace.persistent_bytes,
                mul(
                    boundary.outputs.len(),
                    size_of::<Option<ObservedDecoderTensor>>() + size_of::<ObservedDecoderTensor>(),
                )?,
            )?,
        )?;
        let point_work = plan
            .point_work
            .checked_add(trace.work)
            .and_then(|n| n.checked_add(mul(output_elements, 128).ok()? as u64))
            .and_then(|n| {
                n.checked_add(mul(elements(&boundary.input_shape).ok()?, 128).ok()? as u64)
            })
            .and_then(|n| n.checked_add(mul(metadata, 128).ok()? as u64))
            .ok_or(Refusal::Resource("observation execution work"))?;
        inspection.complete().map_err(Refusal::Budget)?;
        lease.complete().map_err(Refusal::Budget)?;
        let metadata_reservation = lease.commit_persistent().map_err(Refusal::Budget)?;
        super::checkpoint(options)?;
        Ok(DecoderBlockObserver {
            graph,
            boundary,
            budget: self.budget.clone(),
            graph_reservation,
            metadata_reservation,
            linear_limit: plan.linear_retained,
            point_scratch,
            point_work,
            trace_bytes: trace.persistent_bytes,
            output_elements,
            result_bytes,
        })
    }
}

impl DecoderBlockObserver {
    pub fn boundary(&self) -> &ObservationBoundary {
        &self.boundary
    }
    pub fn graph(&self) -> &GraphNetwork {
        &self.graph
    }
    pub fn reserved_bytes(&self) -> Result<usize> {
        add(
            add(
                self.graph_reservation.bytes(),
                self.metadata_reservation.bytes(),
            )?,
            linear_census(&self.graph)?,
        )
    }
    pub fn point_work_bound(&self) -> u64 {
        self.point_work
    }

    /// One actual concrete graph traversal; every original stage is copied and
    /// hashed once after its final point centering, under the original ledger.
    pub fn point(
        &self,
        input: &ArrayD<f32>,
        options: &DecoderOptions,
    ) -> Result<DecoderBlockObservation> {
        if !shape_matches(input.shape(), &self.boundary.input_shape) {
            return Err(Refusal::Tensor("observation boundary shape".into()));
        }
        if linear_census(&self.graph)? > self.linear_limit {
            return Err(Refusal::Resource(
                "observation raw graph exceeded admitted caches",
            ));
        }
        let request = WorkRequest {
            scratch_bytes: self.point_scratch,
            persistent_bytes: self.result_bytes,
            work: self.point_work,
            proposals: 1,
        };
        super::admit(options, &request)?;
        if add(self.reserved_bytes()?, self.result_bytes)? > options.max_resident_bytes {
            return Err(Refusal::Resource("observation result residency cap"));
        }
        let mut lease = self
            .budget
            .lease_for(LeaseClass::Optional, request)
            .map_err(Refusal::Budget)?;
        let limit = deadline(options, lease.deadline());
        let input_digest = point_identity::array_digest(input, &lease, limit)?;
        let input = point_identity::copy_array(input, &lease, limit)?;
        let input = BoundedTensor::concrete_with_poll(input, || {
            point_identity::poll(&lease, limit)
                .map_err(|e| ny_core::NyError::DeadlineExceeded(e.to_string()))
        })
        .map_err(|e| Refusal::Build(e.to_string()))?;
        let mut captured = Vec::new();
        captured
            .try_reserve_exact(self.boundary.outputs.len())
            .map_err(|e| Refusal::Build(format!("observation output allocation: {e}")))?;
        captured.resize_with(self.boundary.outputs.len(), || None);
        let mut callback_refusal = None;
        let result =
            self.graph
                .propagate_concrete_point_with_observer(&input, limit, |node, bounds| {
                    let capture = (|| -> Result<()> {
                        point_identity::poll(&lease, limit)?;
                        let Some(index) = self.boundary.outputs.iter().position(|o| o.node == node)
                        else {
                            return Ok(());
                        };
                        let identity = &self.boundary.outputs[index];
                        if captured[index].is_some()
                            || !shape_matches(bounds.shape(), &identity.original_shape)
                        {
                            return Err(Refusal::Tensor(
                                "observation stage shape or duplicate/nonpoint output".into(),
                            ));
                        }
                        for (index, (l, u)) in bounds.lower().iter().zip(bounds.upper()).enumerate()
                        {
                            if index % 1024 == 0 {
                                point_identity::poll(&lease, limit)?;
                            }
                            if l.to_bits() != u.to_bits() {
                                return Err(Refusal::Tensor("nonpoint observed output".into()));
                            }
                        }
                        let values = point_identity::copy_array(bounds.lower(), &lease, limit)?;
                        let value_digest = point_identity::array_digest(&values, &lease, limit)?;
                        captured[index] = Some(ObservedDecoderTensor {
                            identity: copy_identity(identity, &lease, limit)?,
                            values,
                            value_digest,
                        });
                        Ok(())
                    })();
                    capture.map_err(|error| {
                        callback_refusal = Some(error);
                        ny_core::NyError::InvalidSpec("diagnostic observer refused".into())
                    })
                });
        if let Some(error) = callback_refusal {
            return Err(error);
        }
        let (final_point, trace) = result.map_err(|e| Refusal::Build(e.to_string()))?;
        if trace
            .retained_bytes_with_deadline(limit)
            .map_err(|e| Refusal::Build(e.to_string()))?
            > self.trace_bytes
        {
            return Err(Refusal::Resource("observation trace allocation mismatch"));
        }
        let mut tensors = Vec::new();
        tensors
            .try_reserve_exact(captured.len())
            .map_err(|e| Refusal::Build(format!("observation result allocation: {e}")))?;
        let mut observed_elements = 0;
        let mut hash = Sha256::new();
        hash.update(b"ny-decoder-original-block-all-output-bits/v1");
        hash.update(self.boundary.subset_digest);
        for tensor in captured {
            point_identity::poll(&lease, limit)?;
            let tensor =
                tensor.ok_or_else(|| Refusal::Tensor("missing original observed stage".into()))?;
            observed_elements = add(observed_elements, tensor.values.len())?;
            hash.update((tensor.identity.tensor.len() as u64).to_le_bytes());
            hash.update(tensor.identity.tensor.as_bytes());
            hash.update((tensor.values.ndim() as u64).to_le_bytes());
            for &dimension in tensor.values.shape() {
                hash.update((dimension as u64).to_le_bytes());
            }
            hash.update(tensor.value_digest);
            tensors.push(tensor);
        }
        if observed_elements != self.output_elements {
            return Err(Refusal::Tensor("observation stage element census".into()));
        }
        let outputs_digest = hash.finalize().into();
        let execution_digest = point_identity::observation_digest(
            &self.boundary.subset_digest,
            &self.boundary.input_tensor,
            &input_digest,
            &outputs_digest,
            &trace,
            &lease,
            limit,
        )?;
        let boundary = copy_boundary(&self.boundary, &lease, limit)?;
        drop(final_point);
        drop(input);
        if linear_census(&self.graph)? > self.linear_limit {
            return Err(Refusal::Resource(
                "observation point cache allocation mismatch",
            ));
        }
        lease.complete().map_err(Refusal::Budget)?;
        let reservation = lease.commit_persistent().map_err(Refusal::Budget)?;
        super::checkpoint(options)?;
        Ok(DecoderBlockObservation {
            boundary,
            tensors,
            trace,
            input_digest,
            outputs_digest,
            execution_digest,
            reservation,
        })
    }
}
