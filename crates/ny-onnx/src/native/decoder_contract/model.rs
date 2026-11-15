// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Named production graph handoff and concrete parity route with shared leases.
use super::{
    add, mul, DecoderContractRefusal as Refusal, DecoderOptions, LoweringManifest, Result,
};
use ndarray::{ArrayD, IxDyn};
use ny_build::{GraphBuildInputs, GraphModel, GraphNetworkOptions};
use ny_core::LayerType;
use ny_propagate::certify::work::{
    LeaseClass, PersistentReservation, WorkBudget, WorkLease, WorkRequest,
};
use ny_propagate::layers::{CausalSoftmaxLayer, PointLinearBuildPolicy, PointLinearLayer};
use ny_propagate::network::GraphPointTrace;
use ny_propagate::{GraphNetwork, Layer};
use ny_tensor::BoundedTensor;

pub(super) mod point_identity;

pub(super) struct GraphPlan {
    pub(super) request: WorkRequest,
    pub(super) linear_retained: usize,
    pub(super) point_scratch: usize,
    pub(super) point_work: u64,
    linear_construction_work: u64,
    linear_construction_scratch: usize,
    linear_count: u64,
}

/// The owned original coefficients and exact lowering manifest travel together.
/// No Clone: duplicating a graph/head cannot silently share one byte charge.
#[derive(Debug)]
pub struct LoweredDecoder {
    pub(super) graph_model: GraphModel,
    pub(super) manifest: LoweringManifest,
    pub(super) embedding: String,
    pub(super) hidden_output: String,
    pub(super) logits_output: String,
    pub(super) block_outputs: Vec<String>,
    pub(super) sequence: usize,
    pub(super) budget: WorkBudget,
    pub(super) weight_reservation: PersistentReservation,
    pub(super) extra_reservation: PersistentReservation,
    pub(super) resident_bytes: usize,
}
impl LoweredDecoder {
    pub fn manifest(&self) -> &LoweringManifest {
        &self.manifest
    }
    pub fn graph_model(&self) -> &GraphModel {
        &self.graph_model
    }
    pub fn reserved_bytes(&self) -> usize {
        self.weight_reservation.bytes() + self.extra_reservation.bytes()
    }
    pub fn sequence_length(&self) -> usize {
        self.sequence
    }
    pub fn hidden_output(&self) -> &str {
        &self.hidden_output
    }
    pub fn logits_output(&self) -> &str {
        &self.logits_output
    }
    pub fn block_outputs(&self) -> &[String] {
        &self.block_outputs
    }
    pub fn budget(&self) -> WorkBudget {
        self.budget.clone()
    }

    /// Pre-allocation capacity/work for the complete converted graph, including
    /// every layer even when an early block output is selected. Persistent bytes
    /// cover one compact original copy and graph/non-affine metadata. Scratch
    /// covers canonical copying and non-affine construction. Source WeightStore
    /// remains separately charged. No signed/spectral/CROWN cache is constructed.
    pub fn graph_admission(&self, options: &DecoderOptions) -> Result<WorkRequest> {
        self.graph_plan(options).map(|p| p.request)
    }

    pub(super) fn graph_plan(&self, options: &DecoderOptions) -> Result<GraphPlan> {
        super::checkpoint(options)?;
        if self.graph_model.network.layers.len() > options.max_nodes
            || self.sequence > options.max_sequence
        {
            return Err(Refusal::Resource("graph shape cap"));
        }
        let mut resident = size_of::<GraphNetwork>();
        let mut metadata = 0usize;
        let mut temporary = 0usize;
        let mut linear_retained = 0usize;
        let mut linear_construction_work = 0u64;
        let mut linear_construction_scratch = 0usize;
        let mut linear_count = 0u64;
        let mut point_temporary = 0usize;
        let mut construction_work = 0u64;
        let mut point_work = 0u64;
        for spec in &self.graph_model.network.layers {
            super::checkpoint(options)?;
            let names = spec
                .inputs
                .iter()
                .chain(&spec.outputs)
                .try_fold(spec.name.len(), |n, s| add(n, s.len()))?;
            // GraphNode/Layer inline storage, names, hash maps, declared shapes,
            // transform vectors and scalar nonlinear cache/control metadata.
            let node_metadata = add(add(size_of::<Layer>(), 4096)?, mul(names, 8)?)?;
            metadata = add(metadata, node_metadata)?;
            let mut parameter_bytes = 0usize;
            match &spec.layer_type {
                LayerType::Linear => {
                    let name = spec
                        .inputs
                        .get(1)
                        .ok_or(Refusal::Resource("linear parameter input"))?;
                    let weight = self
                        .graph_model
                        .weights
                        .get(name)
                        .ok_or(Refusal::Resource("linear parameter missing"))?;
                    if weight.ndim() != 2 {
                        return Err(Refusal::Resource("linear parameter rank"));
                    }
                    let (out, inner) = (weight.shape()[0], weight.shape()[1]);
                    let bias = spec.inputs.len() == 3;
                    let original = PointLinearLayer::construction_request(out, inner, bias)
                        .map_err(|e| Refusal::Build(e.to_string()))?;
                    linear_retained = add(linear_retained, original.persistent_bytes)?;
                    resident = add(resident, original.persistent_bytes)?;
                    linear_construction_scratch =
                        linear_construction_scratch.max(original.scratch_bytes);
                    linear_construction_work = linear_construction_work
                        .checked_add(original.work)
                        .ok_or(Refusal::Resource("original coefficient construction work"))?;
                    linear_count = linear_count
                        .checked_add(original.proposals)
                        .ok_or(Refusal::Resource("original coefficient constructor count"))?;
                    construction_work = construction_work
                        .checked_add(original.work)
                        .ok_or(Refusal::Resource("graph construction work"))?;
                    let point =
                        PointLinearLayer::execution_request(out, inner, self.sequence, bias)
                            .map_err(|e| Refusal::Build(e.to_string()))?;
                    point_temporary = point_temporary.max(point.scratch_bytes);
                    point_work = point_work
                        .checked_add(point.work)
                        .ok_or(Refusal::Resource("point projection work"))?;
                }
                LayerType::LayerNorm | LayerType::RMSNorm => {
                    for input in spec.inputs.iter().skip(1) {
                        let parameter = self
                            .graph_model
                            .weights
                            .get(input)
                            .ok_or(Refusal::Resource("normalization parameter missing"))?;
                        parameter_bytes = add(parameter_bytes, mul(parameter.len(), 4)?)?;
                    }
                }
                LayerType::Add | LayerType::Sub | LayerType::Mul => {
                    for input in &spec.inputs {
                        if let Some(constant) = self.graph_model.weights.get(input) {
                            parameter_bytes = add(parameter_bytes, mul(constant.len(), 4)?)?;
                        }
                    }
                }
                LayerType::Reshape
                | LayerType::Transpose
                | LayerType::Slice
                | LayerType::Concat
                | LayerType::Tile
                | LayerType::MatMul
                | LayerType::CausalSoftmax
                | LayerType::GELU
                | LayerType::SiLU => {}
                _ => {
                    return Err(Refusal::Unsupported(format!(
                        "no decoder graph allocation model for {}",
                        spec.layer_type
                    )))
                }
            }
            resident = add(resident, parameter_bytes)?;
            temporary = temporary.max(mul(parameter_bytes, 3)?);
            construction_work = construction_work
                .checked_add((parameter_bytes / 4) as u64)
                .ok_or(Refusal::Resource("graph parameter work"))?;
            // Full normalization/GELU/SiLU/mask/shape point processing, including
            // GELU's scalar OnceLock critical-point bracket. The large GELU
            // CROWN tangent tables are not touched by this managed point route.
            if let Some(output) = spec
                .outputs
                .first()
                .and_then(|s| self.graph_model.tensor_shapes.get(s))
            {
                let elements = output.iter().try_fold(1usize, |n, &d| {
                    mul(
                        n,
                        usize::try_from(d).map_err(|_| Refusal::Resource("point shape"))?,
                    )
                })?;
                point_work = point_work
                    .checked_add(mul(elements, 32)? as u64)
                    .and_then(|n| n.checked_add(4096))
                    .ok_or(Refusal::Resource("point scalar work"))?;
            }
        }
        resident = add(resident, metadata)?;
        let scratch = add(
            add(temporary, mul(metadata, 2)?)?,
            linear_construction_scratch,
        )?;
        construction_work = construction_work
            .checked_add(metadata as u64)
            .ok_or(Refusal::Resource("graph metadata work"))?;
        let mut scratch_elements = 0usize;
        for shape in self.graph_model.tensor_shapes.values() {
            let elements = shape.iter().try_fold(1usize, |n, &d| {
                mul(
                    n,
                    usize::try_from(d).map_err(|_| Refusal::Resource("dynamic point shape"))?,
                )
            })?;
            scratch_elements = add(scratch_elements, elements)?;
        }
        // All cached endpoints/centers plus the largest padded forward product.
        let point_scratch = add(mul(scratch_elements, 64)?, point_temporary)?;
        let p = &self.manifest.parameters;
        let attention_terms = mul(
            mul(
                mul(mul(p.num_layers, p.query_heads)?, self.sequence)?,
                self.sequence,
            )?,
            mul(p.head_dim, 4)?,
        )?;
        // The explicitly noncertifying graph point branch now evaluates each
        // visible score once. Preserve QK/PV pricing and charge its full dense
        // scans/copies plus actual causal triangle using the operator's census.
        let causal_work = CausalSoftmaxLayer::point_work_bound(
            &[p.query_heads, self.sequence, self.sequence],
            None,
        )
        .map_err(|e| Refusal::Build(e.to_string()))?
        .checked_mul(p.num_layers as u64)
        .ok_or(Refusal::Resource("point causal work"))?;
        point_work = point_work
            .checked_add(attention_terms as u64)
            .and_then(|n| n.checked_add(causal_work))
            .ok_or(Refusal::Resource("point attention work"))?;
        let request = WorkRequest {
            scratch_bytes: scratch,
            persistent_bytes: resident,
            work: construction_work,
            proposals: linear_count
                .checked_add(1)
                .ok_or(Refusal::Resource("graph construction count"))?,
        };
        super::admit(options, &request)?;
        if add(self.resident_bytes, resident)? > options.max_resident_bytes {
            return Err(Refusal::Resource("materialized graph admission"));
        }
        Ok(GraphPlan {
            request,
            linear_retained,
            point_scratch,
            point_work,
            linear_construction_work,
            linear_construction_scratch,
            linear_count,
        })
    }

    /// Shared compact construction for whole-model and original block diagnostics.
    /// Point coefficient owners retain their own exact reservations. The outer
    /// lease covers graph/non-affine metadata only; source coefficients remain
    /// in this model's separate original ledger owner throughout construction.
    pub(super) fn build_observation_graph(
        &self,
        inputs: &GraphBuildInputs<'_>,
        options: &DecoderOptions,
    ) -> Result<(GraphNetwork, PersistentReservation, GraphPlan)> {
        let (graph, mut lease, plan) = self.build_compact_graph(inputs, options)?;
        lease.complete().map_err(Refusal::Budget)?;
        let reservation = lease.commit_persistent().map_err(Refusal::Budget)?;
        super::checkpoint(options)?;
        Ok((graph, reservation, plan))
    }

    fn build_compact_graph(
        &self,
        inputs: &GraphBuildInputs<'_>,
        options: &DecoderOptions,
    ) -> Result<(GraphNetwork, WorkLease, GraphPlan)> {
        if !std::ptr::eq(inputs.weights, &raw const self.graph_model.weights) {
            return Err(Refusal::Tensor(
                "observation graph requires original coefficient store".into(),
            ));
        }
        let plan = self.graph_plan(options)?;
        let aggregate = WorkRequest {
            persistent_bytes: plan
                .request
                .persistent_bytes
                .checked_sub(plan.linear_retained)
                .ok_or(Refusal::Resource("compact graph owner accounting"))?,
            scratch_bytes: plan
                .request
                .scratch_bytes
                .checked_sub(plan.linear_construction_scratch)
                .ok_or(Refusal::Resource("compact graph scratch accounting"))?,
            work: plan
                .request
                .work
                .checked_sub(plan.linear_construction_work)
                .ok_or(Refusal::Resource("compact graph work accounting"))?,
            proposals: plan
                .request
                .proposals
                .checked_sub(plan.linear_count)
                .ok_or(Refusal::Resource("compact graph constructor accounting"))?,
        };
        let lease = self
            .budget
            .lease_for(LeaseClass::Optional, aggregate)
            .map_err(Refusal::Budget)?;
        let deadline = [
            options.deadline,
            lease.deadline(),
            ny_core::instance_budget::deadline(),
        ]
        .into_iter()
        .flatten()
        .min();
        let policy = PointLinearBuildPolicy {
            budget: &self.budget,
            deadline,
            max_resident_bytes: options.max_resident_bytes,
            max_scratch_bytes: options.max_scratch_bytes,
            max_work: options.max_terms,
        };
        let graph = ny_build::build_graph_network_with_linear_storage(
            inputs,
            GraphNetworkOptions::default(),
            ny_build::LinearStoragePolicy::OriginalPoint(policy),
        )
        .map_err(|e| Refusal::Build(e.to_string()))?;
        for name in graph.node_names() {
            let node = graph
                .node(name)
                .ok_or_else(|| Refusal::Build("graph node disappeared".into()))?;
            if matches!(node.layer(), Layer::OpaqueSkip(_) | Layer::Linear(_)) {
                return Err(Refusal::Unsupported(format!(
                    "noncompact or opaque graph node {name}"
                )));
            }
        }
        if linear_census(&graph)? > plan.linear_retained {
            return Err(Refusal::Resource(
                "compact original allocation model mismatch",
            ));
        }
        lease.checkpoint().map_err(Refusal::Budget)?;
        super::checkpoint(options)?;
        Ok((graph, lease, plan))
    }

    /// Original immutable embedding rows. Tokenization is outside this API.
    pub fn embedded_tokens(
        &self,
        tokens: &[usize],
        options: &DecoderOptions,
    ) -> Result<EmbeddedInput> {
        super::checkpoint(options)?;
        if tokens.len() != self.sequence
            || tokens.len() > options.max_sequence
            || tokens
                .iter()
                .any(|&id| id >= self.manifest.parameters.vocab_size)
        {
            return Err(Refusal::Tensor("token IDs or sequence length".into()));
        }
        let p = &self.manifest.parameters;
        let bytes = mul(mul(tokens.len(), p.hidden_size)?, 4)?;
        super::admit(
            options,
            &WorkRequest {
                scratch_bytes: 0,
                persistent_bytes: bytes,
                work: (bytes / 4) as u64,
                proposals: 1,
            },
        )?;
        if add(self.resident_bytes, bytes)? > options.max_resident_bytes {
            return Err(Refusal::Resource("embedded input residency"));
        }
        let mut lease = self
            .budget
            .lease_for(
                LeaseClass::Optional,
                WorkRequest {
                    scratch_bytes: 0,
                    persistent_bytes: bytes,
                    work: (tokens.len() * p.hidden_size) as u64,
                    proposals: 1,
                },
            )
            .map_err(Refusal::Budget)?;
        let embedding = self
            .graph_model
            .weights
            .get(&self.embedding)
            .ok_or_else(|| Refusal::Tensor("original embedding missing".into()))?;
        let flat = embedding
            .as_slice()
            .ok_or_else(|| Refusal::Tensor("embedding contiguous layout".into()))?;
        let mut values = Vec::with_capacity(bytes / 4);
        for &id in tokens {
            lease.checkpoint().map_err(Refusal::Budget)?;
            super::checkpoint(options)?;
            values.extend_from_slice(&flat[id * p.hidden_size..(id + 1) * p.hidden_size]);
        }
        let values = ArrayD::from_shape_vec(IxDyn(&[tokens.len(), p.hidden_size]), values)
            .map_err(|e| Refusal::Tensor(e.to_string()))?;
        lease.complete().map_err(Refusal::Budget)?;
        let reservation = lease.commit_persistent().map_err(Refusal::Budget)?;
        Ok(EmbeddedInput {
            values,
            reservation,
        })
    }

    /// Production graph build for logits, final hidden state, or a named block.
    /// Output aliases are `logits`, `hidden`, and `block.N`; exact tensor names
    /// in the manifest are also accepted. Any conservative opaque skip refuses
    /// this faithful-operator route instead of silently qualifying the graph.
    pub fn graph_for(&self, output: &str, options: &DecoderOptions) -> Result<DecoderGraph> {
        super::checkpoint(options)?;
        if self.graph_model.network.layers.len() > options.max_nodes
            || self.sequence > options.max_sequence
        {
            return Err(Refusal::Resource("graph shape cap"));
        }
        let tensor = if output == "logits" {
            self.logits_output.as_str()
        } else if output == "hidden" {
            self.hidden_output.as_str()
        } else if let Some(n) = output.strip_prefix("block.") {
            let index = n
                .parse::<usize>()
                .map_err(|_| Refusal::Tensor("block output index".into()))?;
            self.block_outputs
                .get(index)
                .map(String::as_str)
                .ok_or_else(|| Refusal::Tensor("block output index".into()))?
        } else {
            output
        };
        let spec = self
            .graph_model
            .network
            .outputs
            .iter()
            .find(|s| s.name == tensor)
            .ok_or_else(|| Refusal::Tensor(format!("unknown output {output}")))?;
        let mut inputs: GraphBuildInputs<'_> = self.graph_model.graph_build_inputs();
        inputs.outputs = std::slice::from_ref(spec);
        let (graph, mut lease, plan) = self.build_compact_graph(&inputs, options)?;
        let output_elements = spec.shape.iter().try_fold(1usize, |n, &d| {
            mul(
                n,
                usize::try_from(d).map_err(|_| Refusal::Resource("output shape"))?,
            )
        })?;
        let p = &self.manifest.parameters;
        let deadline = [
            options.deadline,
            lease.deadline(),
            ny_core::instance_budget::deadline(),
        ]
        .into_iter()
        .flatten()
        .min();
        let trace = graph
            .point_trace_admission(None, deadline)
            .map_err(|e| Refusal::Build(e.to_string()))?;
        // Completed-arm names and exact input/output bit hashing share the
        // original execution lease. No tensor-sized serialization is needed.
        let hash_work = mul(
            add(mul(self.sequence, p.hidden_size)?, output_elements)?,
            64,
        )? as u64;
        let point_work = plan
            .point_work
            .checked_add(trace.work)
            .and_then(|n| n.checked_add(hash_work))
            .and_then(|n| n.checked_add((tensor.len() as u64).checked_mul(64)?))
            .and_then(|n| n.checked_add(4096))
            .ok_or(Refusal::Resource("point identity work"))?;
        let point_scratch = add(plan.point_scratch, trace.scratch_bytes)?;
        let selected_output = point_identity::copy_name(tensor, &lease, deadline)?;
        let linear_retained = linear_census(&graph)?;
        lease.complete().map_err(Refusal::Budget)?;
        let reservation = lease.commit_persistent().map_err(Refusal::Budget)?;
        super::checkpoint(options)?;
        Ok(DecoderGraph {
            graph,
            reservation,
            budget: self.budget.clone(),
            point_scratch,
            point_work,
            point_trace_bytes: trace.persistent_bytes,
            lowering_digest: self.manifest.lowering_digest,
            selected_output,
            linear_retained,
            output_elements,
            sequence: self.sequence,
            hidden_size: p.hidden_size,
        })
    }
}

#[derive(Debug)]
pub struct EmbeddedInput {
    values: ArrayD<f32>,
    reservation: PersistentReservation,
}
impl EmbeddedInput {
    pub fn values(&self) -> &ArrayD<f32> {
        &self.values
    }
    pub fn reserved_bytes(&self) -> usize {
        self.reservation.bytes()
    }
}

#[derive(Debug)]
pub struct DecoderGraph {
    graph: GraphNetwork,
    reservation: PersistentReservation,
    budget: WorkBudget,
    point_scratch: usize,
    point_work: u64,
    point_trace_bytes: usize,
    lowering_digest: [u8; 32],
    selected_output: String,
    linear_retained: usize,
    output_elements: usize,
    sequence: usize,
    hidden_size: usize,
}
impl DecoderGraph {
    /// Read-only graph access does not carry admission for direct execution or
    /// certified backward calls. Point-only affine layers explicitly refuse
    /// interval/CROWN/proof paths until separately qualified derived views exist.
    /// Raw diagnostic calls require external admission; use `point` for the
    /// managed route sharing the original model ledger.
    pub fn graph(&self) -> &GraphNetwork {
        &self.graph
    }
    pub fn reserved_bytes(&self) -> usize {
        // Compact coefficient leases live on graph nodes, not this outer
        // metadata owner. Both were admitted from the same original budget.
        self.reservation.bytes() + self.linear_retained
    }
    pub fn linear_retained_bytes(&self) -> Result<usize> {
        linear_census(&self.graph)
    }
    pub fn point_scratch_bytes(&self) -> usize {
        self.point_scratch
    }
    pub fn point_work_bound(&self) -> u64 {
        self.point_work
    }
    /// Evaluate the ACTUAL converted graph using its concrete point route.
    /// This result is a parity observation only and never supplies a proof.
    pub fn point(
        &self,
        embedding: &ArrayD<f32>,
        options: &DecoderOptions,
    ) -> Result<PointObservation> {
        // Check the immutable original-owner census against the admitted graph.
        // No derived coefficient cache is authorized by this point lease.
        if linear_census(&self.graph)? > self.linear_retained {
            return Err(Refusal::Resource(
                "raw graph exceeded point cache admission",
            ));
        }
        if embedding.shape() != [self.sequence, self.hidden_size] {
            return Err(Refusal::Tensor("point embedding shape/finiteness".into()));
        }
        if self.point_scratch > options.max_scratch_bytes {
            return Err(Refusal::Resource("point scratch cap"));
        }
        let result_bytes = add(
            add(
                add(
                    mul(self.output_elements, 4)?,
                    size_of::<PointObservation>() + 128,
                )?,
                self.point_trace_bytes,
            )?,
            self.selected_output.len(),
        )?;
        if add(self.reserved_bytes(), result_bytes)? > options.max_resident_bytes {
            return Err(Refusal::Resource("point result residency cap"));
        }
        super::admit(
            options,
            &WorkRequest {
                scratch_bytes: self.point_scratch,
                persistent_bytes: result_bytes,
                work: self.point_work,
                proposals: 1,
            },
        )?;
        let mut lease = self
            .budget
            .lease_for(
                LeaseClass::Optional,
                WorkRequest {
                    scratch_bytes: self.point_scratch,
                    persistent_bytes: result_bytes,
                    work: self.point_work,
                    proposals: 1,
                },
            )
            .map_err(Refusal::Budget)?;
        let deadline = [
            options.deadline,
            lease.deadline(),
            ny_core::instance_budget::deadline(),
        ]
        .into_iter()
        .flatten()
        .min();
        let input_digest = point_identity::array_digest(embedding, &lease, deadline)?;
        let copied_input = point_identity::copy_array(embedding, &lease, deadline)?;
        let input = BoundedTensor::concrete_with_poll(copied_input, || {
            point_identity::poll(&lease, deadline)
                .map_err(|e| ny_core::NyError::DeadlineExceeded(e.to_string()))
        })
        .map_err(|e| Refusal::Build(e.to_string()))?;
        let (bounds, trace) = self
            .graph
            .propagate_concrete_point_with_trace(&input, None, deadline)
            .map_err(|e| Refusal::Build(e.to_string()))?;
        if trace
            .retained_bytes_with_deadline(deadline)
            .map_err(|e| Refusal::Build(e.to_string()))?
            > self.point_trace_bytes
        {
            return Err(Refusal::Resource("point trace storage model mismatch"));
        }
        let values = point_identity::copy_array(bounds.lower(), &lease, deadline)?;
        let value_digest = point_identity::array_digest(&values, &lease, deadline)?;
        let selected_output = point_identity::copy_name(&self.selected_output, &lease, deadline)?;
        let execution_digest = point_identity::observation_digest(
            &self.lowering_digest,
            &selected_output,
            &input_digest,
            &value_digest,
            &trace,
            &lease,
            deadline,
        )?;
        drop(bounds);
        drop(input);
        if linear_census(&self.graph)? > self.linear_retained {
            return Err(Refusal::Resource("point cache allocation model mismatch"));
        }
        lease.complete().map_err(Refusal::Budget)?;
        let reservation = lease.commit_persistent().map_err(Refusal::Budget)?;
        super::checkpoint(options)?;
        Ok(PointObservation {
            values,
            trace,
            selected_output,
            lowering_digest: self.lowering_digest,
            input_digest,
            value_digest,
            execution_digest,
            reservation,
        })
    }
}

pub(super) fn linear_census(graph: &GraphNetwork) -> Result<usize> {
    graph.node_names().iter().try_fold(0usize, |bytes, name| {
        let node = graph
            .node(name)
            .ok_or_else(|| Refusal::Build("graph node disappeared".into()))?;
        if let Layer::Linear(linear) = node.layer() {
            add(
                bytes,
                linear
                    .retained_bytes()
                    .map_err(|e| Refusal::Build(e.to_string()))?,
            )
        } else if let Layer::PointLinear(linear) = node.layer() {
            add(bytes, linear.retained_bytes())
        } else {
            Ok(bytes)
        }
    })
}
#[derive(Debug)]
pub struct PointObservation {
    values: ArrayD<f32>,
    trace: GraphPointTrace,
    selected_output: String,
    lowering_digest: [u8; 32],
    input_digest: [u8; 32],
    value_digest: [u8; 32],
    execution_digest: [u8; 32],
    reservation: PersistentReservation,
}
impl PointObservation {
    pub fn values(&self) -> &ArrayD<f32> {
        &self.values
    }
    pub fn reserved_bytes(&self) -> usize {
        self.reservation.bytes()
    }
    pub fn execution_trace(&self) -> &GraphPointTrace {
        &self.trace
    }
    pub fn selected_output(&self) -> &str {
        &self.selected_output
    }
    pub fn lowering_digest(&self) -> &[u8; 32] {
        &self.lowering_digest
    }
    pub fn input_digest(&self) -> &[u8; 32] {
        &self.input_digest
    }
    pub fn value_digest(&self) -> &[u8; 32] {
        &self.value_digest
    }
    /// Binds the original lowering, selected tensor, exact point bits and
    /// actual completed software arms. This is diagnostic identity, not proof.
    pub fn execution_digest(&self) -> &[u8; 32] {
        &self.execution_digest
    }
}
