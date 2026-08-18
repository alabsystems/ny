// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Exact affine closure for one conjunctive candidate-violation region.
//!
//! This is a deliberately small pre-BaB authority lane. It accepts only a
//! sequential network that becomes Linear/ReLU after exact constant folding,
//! uses sound IBP to build exact stable phases or continuous Planet outer hulls
//! for unstable ReLUs, and asks AY whether the non-strict closure of the
//! VNN-LIB unsafe region is infeasible.
//! A successful result crosses AY's exact root-Farkas verifier and an
//! independent ny-cert replay inside ny-mip. Every refusal is non-terminal.

use std::path::Path;
use std::time::{Duration, Instant};

use ny_mip::encoder::MipEncoder;
use ny_mip::{
    certify_continuous_root_infeasibility_with_ay_until_admission,
    CertifiedContinuousRootInfeasibility, CertifiedLinearLowerWorkerAdmission,
};
#[cfg(test)]
use ny_onnx::vnnlib::OutputConstraint;
use ny_onnx::vnnlib::{
    load_vnnlib_with_certified_affine_property, CertifiedRelationalOutputAtom, VnnLibSpec,
};
use ny_onnx::{
    load_onnx_with_config, AttributeValue, BatchNormFoldingPolicy, DataType, OnnxLoadConfig,
    OnnxOptimizationFlag,
};
use ny_propagate::{Layer, Network, VerificationArtifactAuthority};
use ny_tensor::BoundedTensor;
use tracing::debug;

use super::mip_preprocess::{
    bounded_tensor_to_bounds, fold_constant_layers, strip_shape_layers, FoldedMipNetwork,
};
const MAX_RAW_LAYERS: usize = 256;
const MAX_SOURCE_ELEMENTS: usize = 1_000_000;
const MAX_MODEL_SOURCE_BYTES: u64 = 64 * 1024 * 1024;
const MAX_PROPERTY_SOURCE_BYTES: u64 = 64 * 1024 * 1024;
const MAX_CUMULATIVE_ACTIVATION_ELEMENTS: usize = 1_000_000;
const MAX_COLUMNS: usize = 4_096;
const MAX_ROWS: usize = 8_192;
const MAX_NONZEROS: usize = 1_000_000;
const MAX_OUTPUT_CONSTRAINTS: usize = 128;
const PROOF_SHARE: f64 = 0.05;
const PROOF_CAP: Duration = Duration::from_secs(2);
const MIN_PROOF_SLICE: Duration = Duration::from_millis(10);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum AffineRootFarkasDecline {
    ArtifactAuthority,
    Deadline,
    WorkerBusy,
    PeeledOutput,
    UntrustedModelSource,
    UncertifiedPropertySource,
    UnsupportedProperty,
    UnsupportedLayer,
    InvalidModel,
    ResourceLimit,
    InexactReluHull,
    NotProved,
    SolverOrReplay,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum AffineRootFarkasAttempt {
    Proved(CertifiedContinuousRootInfeasibility),
    Declined(AffineRootFarkasDecline),
}

impl AffineRootFarkasAttempt {
    fn declined(reason: AffineRootFarkasDecline) -> Self {
        debug!(?reason, "exact affine root-Farkas lane declined");
        Self::Declined(reason)
    }
}

/// Try the exact affine closure using freshly source-authenticated inputs.
///
/// After cheap policy and file-metadata gates, this function acquires the
/// process-wide exact-worker lease before reparsing VNN-LIB or reloading ONNX.
/// A detached AY worker therefore sheds this entire optional lane. The local
/// clock includes source authentication, folding, and sound IBP; bounded
/// synchronous preprocessing cannot be interrupted, but an expired slice
/// prevents solver launch and prevents a late result from gaining authority.
pub(super) fn try_affine_root_farkas(
    model_path: &Path,
    onnx_load_config: &OnnxLoadConfig,
    property_path: Option<&Path>,
    input: &BoundedTensor,
    spec: &VnnLibSpec,
    authority: VerificationArtifactAuthority,
    overall_deadline: Option<Instant>,
    sigmoid_peeled: bool,
) -> AffineRootFarkasAttempt {
    if authority != VerificationArtifactAuthority::VerdictOnly {
        return AffineRootFarkasAttempt::declined(AffineRootFarkasDecline::ArtifactAuthority);
    }
    if sigmoid_peeled {
        return AffineRootFarkasAttempt::declined(AffineRootFarkasDecline::PeeledOutput);
    }
    let Some(proof_deadline) = affine_proof_deadline(Instant::now(), overall_deadline) else {
        return AffineRootFarkasAttempt::declined(AffineRootFarkasDecline::Deadline);
    };
    let constraint_count = match authoritative_constraints(spec, input) {
        Ok(count) => count,
        Err(reason) => return AffineRootFarkasAttempt::declined(reason),
    };
    if let Err(reason) = preflight_source_authority(model_path, onnx_load_config, property_path) {
        return AffineRootFarkasAttempt::declined(reason);
    }
    let Some(admission) = CertifiedLinearLowerWorkerAdmission::try_acquire() else {
        return AffineRootFarkasAttempt::declined(AffineRootFarkasDecline::WorkerBusy);
    };
    let (exact_input_bounds, constraints) =
        match certified_source_property(property_path, spec, input, constraint_count) {
            Ok(property) => property,
            Err(reason) => return AffineRootFarkasAttempt::declined(reason),
        };
    if Instant::now() >= proof_deadline {
        return AffineRootFarkasAttempt::declined(AffineRootFarkasDecline::Deadline);
    }
    let network = match strict_reload_affine_network(model_path, onnx_load_config) {
        Ok(network) => network,
        Err(reason) => return AffineRootFarkasAttempt::declined(reason),
    };
    try_sequential_affine_root_farkas(
        &network,
        input,
        spec,
        &constraints,
        &exact_input_bounds,
        proof_deadline,
        admission,
    )
}

fn affine_proof_deadline(now: Instant, overall_deadline: Option<Instant>) -> Option<Instant> {
    let slice = match overall_deadline {
        Some(deadline) => deadline.checked_duration_since(now)?.mul_f64(PROOF_SHARE),
        None => PROOF_CAP,
    }
    .min(PROOF_CAP);
    if slice < MIN_PROOF_SLICE {
        return None;
    }
    let local = now.checked_add(slice)?;
    Some(overall_deadline.map_or(local, |outer| local.min(outer)))
}

fn preflight_source_authority(
    model_path: &Path,
    caller_config: &OnnxLoadConfig,
    property_path: Option<&Path>,
) -> Result<(), AffineRootFarkasDecline> {
    preflight_model_source(model_path, caller_config)?;
    let path = property_path.ok_or(AffineRootFarkasDecline::UncertifiedPropertySource)?;
    if std::fs::metadata(path)
        .map(|metadata| !metadata.is_file() || metadata.len() > MAX_PROPERTY_SOURCE_BYTES)
        .unwrap_or(true)
    {
        return Err(AffineRootFarkasDecline::UncertifiedPropertySource);
    }
    Ok(())
}

fn preflight_model_source(
    model_path: &Path,
    caller_config: &OnnxLoadConfig,
) -> Result<(), AffineRootFarkasDecline> {
    if model_path
        .extension()
        .and_then(|extension| extension.to_str())
        != Some("onnx")
        || caller_config.has_optimization_flag(OnnxOptimizationFlag::MergeLinear)
        || std::fs::metadata(model_path)
            .map(|metadata| !metadata.is_file() || metadata.len() > MAX_MODEL_SOURCE_BYTES)
            .unwrap_or(true)
    {
        return Err(AffineRootFarkasDecline::UntrustedModelSource);
    }
    Ok(())
}

fn certified_source_property(
    property_path: Option<&Path>,
    expected_spec: &VnnLibSpec,
    actual_input: &BoundedTensor,
    expected_constraint_count: usize,
) -> Result<(Vec<(f64, f64)>, Vec<CertifiedRelationalOutputAtom>), AffineRootFarkasDecline> {
    let path = property_path.ok_or(AffineRootFarkasDecline::UncertifiedPropertySource)?;
    let (source_spec, certified, certified_outputs) =
        load_vnnlib_with_certified_affine_property(path).map_err(|error| {
            debug!(%error, path = %path.display(), "exact affine property-source reparse declined");
            AffineRootFarkasDecline::UncertifiedPropertySource
        })?;
    if !affine_specs_match(&source_spec, expected_spec)
        || certified.len() != expected_spec.num_inputs
        || certified.len() != actual_input.len()
        || certified_outputs.len() != expected_constraint_count
    {
        return Err(AffineRootFarkasDecline::UncertifiedPropertySource);
    }

    let mut bounds = Vec::with_capacity(certified.len());
    for (((&lower, &upper), &actual_lower), &actual_upper) in certified
        .lower()
        .iter()
        .zip(certified.upper())
        .zip(actual_input.lower())
        .zip(actual_input.upper())
    {
        if !lower.is_finite()
            || !upper.is_finite()
            || lower > upper
            || !actual_lower.is_finite()
            || !actual_upper.is_finite()
            || f64::from(actual_lower) > lower
            || f64::from(actual_upper) < upper
        {
            return Err(AffineRootFarkasDecline::UncertifiedPropertySource);
        }
        bounds.push((lower, upper));
    }
    Ok((bounds, certified_outputs.atoms().to_vec()))
}

fn affine_specs_match(source: &VnnLibSpec, expected: &VnnLibSpec) -> bool {
    let same_pairs = |left: &[(f64, f64)], right: &[(f64, f64)]| {
        left.len() == right.len()
            && left.iter().zip(right).all(|(left, right)| {
                left.0.to_bits() == right.0.to_bits() && left.1.to_bits() == right.1.to_bits()
            })
    };
    source.num_inputs == expected.num_inputs
        && source.num_outputs == expected.num_outputs
        && same_pairs(&source.input_bounds, &expected.input_bounds)
        && source.output_constraints == expected.output_constraints
        && source.output_constraint_clauses == expected.output_constraint_clauses
        && source.is_disjunction == expected.is_disjunction
        && source.version == expected.version
        && source.per_clause_input_bounds == expected.per_clause_input_bounds
        && same_pairs(
            &source.declared_input_bounds,
            &expected.declared_input_bounds,
        )
        && source.dual_network.is_none()
        && expected.dual_network.is_none()
}

fn strict_reload_affine_network(
    model_path: &Path,
    caller_config: &OnnxLoadConfig,
) -> Result<Network, AffineRootFarkasDecline> {
    preflight_model_source(model_path, caller_config)?;

    // This reload, not the caller's already-converted model, is the authority
    // boundary. Preserve authored BN nodes, seal raw FLOAT initializers, and
    // make any custom converter an ordinary load failure under provenance.
    let config = caller_config
        .clone()
        .with_batch_norm_folding_policy(BatchNormFoldingPolicy::PreserveRaw)
        .with_require_authored_float32_initializers(true);
    let model = load_onnx_with_config(model_path, &config).map_err(|error| {
        debug!(%error, path = %model_path.display(), "strict affine ONNX reload declined");
        AffineRootFarkasDecline::UntrustedModelSource
    })?;
    if !strict_onnx_affine_sources_are_authored(&model) {
        return Err(AffineRootFarkasDecline::UntrustedModelSource);
    }
    let network = model.to_propagate_network().map_err(|error| {
        debug!(%error, "strict affine ONNX sequential conversion declined");
        AffineRootFarkasDecline::UntrustedModelSource
    })?;
    if !raw_network_within_limits(&network) {
        return Err(AffineRootFarkasDecline::ResourceLimit);
    }
    Ok(network)
}

fn strict_onnx_affine_sources_are_authored(model: &ny_onnx::OnnxModel) -> bool {
    if model.authored_float32_initializers_match_current() != Some(true)
        || model.original_network_topology_matches_current() != Some(true)
        || model.network.inputs.len() != 1
        || model.network.outputs.len() != 1
        || model.network.inputs[0].dtype != DataType::Float32
        || model.network.outputs[0].dtype != DataType::Float32
        || model.network.layers.is_empty()
        || model.network.layers.len() > MAX_RAW_LAYERS
    {
        return false;
    }

    let authored_float = |name: &str| {
        model.weights.get(name).is_some()
            && model.original_float32_initializer_matches_current(name) == Some(true)
    };
    let mut current = model.network.inputs[0].name.as_str();
    if current.is_empty() {
        return false;
    }
    for layer in &model.network.layers {
        if layer.weights.is_some() || layer.outputs.len() != 1 || layer.outputs[0].is_empty() {
            return false;
        }
        let topology_ok = match layer.layer_type {
            ny_core::LayerType::Linear => {
                (layer.inputs.len() == 2 || layer.inputs.len() == 3)
                    && layer.inputs[0] == current
                    && layer.inputs[1..].iter().all(|name| authored_float(name))
                    && strict_gemm_attributes(&layer.attributes)
            }
            ny_core::LayerType::MatMul => {
                layer.inputs.len() == 2
                    && layer
                        .inputs
                        .iter()
                        .filter(|name| name.as_str() == current)
                        .count()
                        == 1
                    && layer
                        .inputs
                        .iter()
                        .filter(|name| name.as_str() != current)
                        .all(|name| authored_float(name))
                    && layer.attributes.is_empty()
            }
            ny_core::LayerType::Add => {
                layer.inputs.len() == 2
                    && layer
                        .inputs
                        .iter()
                        .filter(|name| name.as_str() == current)
                        .count()
                        == 1
                    && layer
                        .inputs
                        .iter()
                        .filter(|name| name.as_str() != current)
                        .all(|name| authored_float(name))
                    && layer.attributes.is_empty()
            }
            ny_core::LayerType::Sub => {
                layer.inputs.len() == 2
                    && layer.inputs[0] == current
                    && authored_float(&layer.inputs[1])
                    && layer.attributes.is_empty()
            }
            ny_core::LayerType::ReLU => {
                layer.inputs.len() == 1 && layer.inputs[0] == current && layer.attributes.is_empty()
            }
            ny_core::LayerType::Flatten => {
                layer.inputs.len() == 1
                    && layer.inputs[0] == current
                    && layer.attributes.iter().all(|(name, value)| {
                        name == "axis" && matches!(value, AttributeValue::Int(_))
                    })
            }
            ny_core::LayerType::Reshape => {
                layer.inputs.len() == 2
                    && layer.inputs[0] == current
                    && model.weights.get(&layer.inputs[1]).is_some()
                    && layer.attributes.iter().all(|(name, value)| {
                        name == "allowzero" && matches!(value, AttributeValue::Int(_))
                    })
            }
            _ => false,
        };
        if !topology_ok {
            return false;
        }
        current = layer.outputs[0].as_str();
    }
    current == model.network.outputs[0].name
}

fn strict_gemm_attributes(attributes: &std::collections::HashMap<String, AttributeValue>) -> bool {
    attributes
        .iter()
        .all(|(name, value)| match (name.as_str(), value) {
            ("alpha" | "beta", AttributeValue::Float(value)) => *value == 1.0,
            ("transA", AttributeValue::Int(value)) => *value == 0,
            ("transB", AttributeValue::Int(value)) => *value == 0 || *value == 1,
            _ => false,
        })
}

fn try_sequential_affine_root_farkas(
    network: &Network,
    input: &BoundedTensor,
    spec: &VnnLibSpec,
    constraints: &[CertifiedRelationalOutputAtom],
    exact_input_bounds: &[(f64, f64)],
    proof_deadline: Instant,
    admission: CertifiedLinearLowerWorkerAdmission,
) -> AffineRootFarkasAttempt {
    if !raw_network_within_limits(network)
        || !original_activation_volume_within_limits(network, input.len())
    {
        return AffineRootFarkasAttempt::declined(AffineRootFarkasDecline::ResourceLimit);
    }

    let stripped = strip_shape_layers(network);
    let folded = match fold_constant_layers(&stripped) {
        Ok(folded) => folded,
        Err(_) => return AffineRootFarkasAttempt::declined(AffineRootFarkasDecline::InvalidModel),
    };
    if !folded_network_within_limits(&folded, input.len(), constraints, spec.num_outputs) {
        return AffineRootFarkasAttempt::declined(AffineRootFarkasDecline::ResourceLimit);
    }
    if Instant::now() >= proof_deadline {
        return AffineRootFarkasAttempt::declined(AffineRootFarkasDecline::Deadline);
    }
    let relu_count = folded
        .layers()
        .iter()
        .filter(|layer| matches!(layer, Layer::ReLU(_)))
        .count();
    let relu_bounds = match sound_relu_bounds(network, input, relu_count) {
        Ok(bounds) => bounds,
        Err(reason) => return AffineRootFarkasAttempt::declined(reason),
    };
    if Instant::now() >= proof_deadline {
        return AffineRootFarkasAttempt::declined(AffineRootFarkasDecline::Deadline);
    }

    let mut encoder = match MipEncoder::new_with_f64_bounds(exact_input_bounds) {
        Ok(encoder) => encoder,
        Err(_) => return AffineRootFarkasAttempt::declined(AffineRootFarkasDecline::InvalidModel),
    };
    let mut bias_index = 0_usize;
    let mut relu_index = 0_usize;
    for layer in folded.layers() {
        let encoded: Result<(), AffineRootFarkasDecline> = match layer {
            Layer::Linear(linear) => {
                let Some(bias) = folded.exact_biases().get(bias_index) else {
                    return AffineRootFarkasAttempt::declined(
                        AffineRootFarkasDecline::InvalidModel,
                    );
                };
                bias_index += 1;
                let weights = linear
                    .weight()
                    .iter()
                    .copied()
                    .map(f64::from)
                    .collect::<Vec<_>>();
                encoder
                    .encode_linear(&weights, bias, linear.out_features())
                    .map_err(|_| AffineRootFarkasDecline::InvalidModel)
            }
            Layer::ReLU(_) => {
                let Some(bounds) = relu_bounds.get(relu_index) else {
                    return AffineRootFarkasAttempt::declined(
                        AffineRootFarkasDecline::InvalidModel,
                    );
                };
                relu_index += 1;
                encoder
                    .encode_relu_continuous_outer(bounds)
                    .map_err(|_| AffineRootFarkasDecline::InexactReluHull)
            }
            _ => {
                return AffineRootFarkasAttempt::declined(AffineRootFarkasDecline::UnsupportedLayer)
            }
        };
        if let Err(reason) = encoded {
            return AffineRootFarkasAttempt::declined(reason);
        }
    }
    if bias_index != folded.exact_biases().len()
        || relu_index != relu_bounds.len()
        || encoder.num_binary_vars() != 0
    {
        return AffineRootFarkasAttempt::declined(AffineRootFarkasDecline::InvalidModel);
    }
    encoder.finalize();
    if encoder.output_vars().len() != spec.num_outputs {
        return AffineRootFarkasAttempt::declined(AffineRootFarkasDecline::InvalidModel);
    }
    if add_output_closure(&mut encoder, constraints).is_err() {
        return AffineRootFarkasAttempt::declined(AffineRootFarkasDecline::UnsupportedProperty);
    }
    let parts = encoder.into_parts();
    if !problem_within_limits(&parts.problem) || !parts.binary_vars.is_empty() {
        return AffineRootFarkasAttempt::declined(AffineRootFarkasDecline::ResourceLimit);
    }
    if Instant::now() >= proof_deadline {
        return AffineRootFarkasAttempt::declined(AffineRootFarkasDecline::Deadline);
    }

    match certify_continuous_root_infeasibility_with_ay_until_admission(
        &parts.problem,
        proof_deadline,
        admission,
    ) {
        Ok(Some(certificate)) => AffineRootFarkasAttempt::Proved(certificate),
        Ok(None) => AffineRootFarkasAttempt::declined(AffineRootFarkasDecline::NotProved),
        Err(error) => {
            debug!(%error, "exact affine root-Farkas solve/replay failed closed");
            AffineRootFarkasAttempt::declined(AffineRootFarkasDecline::SolverOrReplay)
        }
    }
}

fn authoritative_constraints(
    spec: &VnnLibSpec,
    input: &BoundedTensor,
) -> Result<usize, AffineRootFarkasDecline> {
    if spec.is_disjunction
        || spec.dual_network.is_some()
        || spec.num_inputs == 0
        || spec.num_outputs == 0
        || spec.num_inputs != input.len()
        || spec.num_inputs != spec.input_bounds.len()
        || spec.num_inputs > MAX_COLUMNS
        || spec.num_outputs > MAX_COLUMNS
        || spec
            .per_clause_input_bounds
            .iter()
            .any(|bounds| !bounds.is_empty())
        || spec.validate_input_bounds().is_err()
        || spec.validate_output_indices().is_err()
    {
        return Err(AffineRootFarkasDecline::UnsupportedProperty);
    }
    for (&actual_lower, &actual_upper) in input.lower().iter().zip(input.upper()) {
        if !actual_lower.is_finite() || !actual_upper.is_finite() {
            return Err(AffineRootFarkasDecline::UnsupportedProperty);
        }
    }

    let count = if spec.output_constraint_clauses.is_empty() {
        spec.output_constraints.len()
    } else {
        spec.output_constraint_clauses
            .iter()
            .try_fold(0_usize, |count, clause| count.checked_add(clause.len()))
            .ok_or(AffineRootFarkasDecline::ResourceLimit)?
    };
    if count == 0 || count > MAX_OUTPUT_CONSTRAINTS {
        return Err(AffineRootFarkasDecline::UnsupportedProperty);
    }
    let mut constraints = Vec::with_capacity(count);
    if spec.output_constraint_clauses.is_empty() {
        constraints.extend(spec.output_constraints.iter());
    } else {
        constraints.extend(
            spec.output_constraint_clauses
                .iter()
                .flat_map(|clause| clause.iter()),
        );
    }
    // Ordinary VnnLibSpec constant literals are nearest-f64, not source-exact
    // outward enclosures. Until the exact source parser exports general output
    // atoms, only zero-constant relational comparisons may enter proof IR.
    if constraints
        .iter()
        .any(|constraint| !constraint.is_relational())
    {
        return Err(AffineRootFarkasDecline::UnsupportedProperty);
    }
    Ok(count)
}

fn raw_network_within_limits(network: &Network) -> bool {
    if network.layers().len() > MAX_RAW_LAYERS {
        return false;
    }
    let mut elements = 0_usize;
    for layer in network.layers() {
        let add = match layer {
            Layer::Linear(linear) => linear
                .weight()
                .len()
                .checked_add(linear.bias().map_or(0, |bias| bias.len())),
            Layer::AddConstant(add) => Some(add.constant().len()),
            Layer::SubConstant(sub) => Some(sub.constant().len()),
            Layer::ReLU(_) | Layer::Flatten(_) | Layer::Reshape(_) => Some(0),
            _ => return false,
        };
        let Some(next) = add.and_then(|add| elements.checked_add(add)) else {
            return false;
        };
        if next > MAX_SOURCE_ELEMENTS {
            return false;
        }
        elements = next;
    }
    true
}

fn original_activation_volume_within_limits(network: &Network, input_dim: usize) -> bool {
    if input_dim == 0 || input_dim > MAX_COLUMNS {
        return false;
    }
    let mut current_dim = input_dim;
    let mut cumulative = 0_usize;
    for layer in network.layers() {
        current_dim = match layer {
            Layer::Linear(linear) if linear.in_features() == current_dim => linear.out_features(),
            Layer::ReLU(_)
            | Layer::Flatten(_)
            | Layer::Reshape(_)
            | Layer::AddConstant(_)
            | Layer::SubConstant(_) => current_dim,
            _ => return false,
        };
        let Some(next) = cumulative.checked_add(current_dim) else {
            return false;
        };
        if current_dim == 0
            || current_dim > MAX_COLUMNS
            || next > MAX_CUMULATIVE_ACTIVATION_ELEMENTS
        {
            return false;
        }
        cumulative = next;
    }
    true
}

fn sound_relu_bounds(
    original: &Network,
    input: &BoundedTensor,
    expected_count: usize,
) -> Result<Vec<Vec<(f64, f64)>>, AffineRootFarkasDecline> {
    if expected_count == 0 {
        return Ok(Vec::new());
    }
    let outputs = original
        .collect_ibp_bounds_sound(input)
        .map_err(|_| AffineRootFarkasDecline::InvalidModel)?;
    if outputs.len() != original.layers().len() {
        return Err(AffineRootFarkasDecline::InvalidModel);
    }
    let mut bounds = Vec::with_capacity(expected_count);
    for (index, layer) in original.layers().iter().enumerate() {
        if !matches!(layer, Layer::ReLU(_)) {
            continue;
        }
        let preactivation = if index == 0 {
            input
        } else {
            outputs
                .get(index - 1)
                .ok_or(AffineRootFarkasDecline::InvalidModel)?
        };
        let layer_bounds = bounded_tensor_to_bounds(preactivation)
            .map_err(|_| AffineRootFarkasDecline::InvalidModel)?;
        if layer_bounds.iter().any(|bound| {
            !bound.lower().is_finite()
                || !bound.upper().is_finite()
                || bound.lower() > bound.upper()
        }) {
            return Err(AffineRootFarkasDecline::InvalidModel);
        }
        bounds.push(
            layer_bounds
                .iter()
                .map(|bound| (f64::from(bound.lower()), f64::from(bound.upper())))
                .collect(),
        );
    }
    if bounds.len() != expected_count {
        return Err(AffineRootFarkasDecline::InvalidModel);
    }
    Ok(bounds)
}

fn folded_network_within_limits(
    network: &FoldedMipNetwork,
    input_dim: usize,
    constraints: &[CertifiedRelationalOutputAtom],
    expected_output_dim: usize,
) -> bool {
    if input_dim == 0 || input_dim > MAX_COLUMNS {
        return false;
    }
    let mut current_dim = input_dim;
    let mut columns = input_dim;
    let mut rows = constraints.len();
    let Some(mut nonzeros) = constraints
        .iter()
        .map(|_| 2_usize)
        .try_fold(0_usize, |sum, count| sum.checked_add(count))
    else {
        return false;
    };
    let mut bias_index = 0_usize;
    for layer in network.layers() {
        match layer {
            Layer::Linear(linear) => {
                let (out_dim, in_dim) = linear.weight().dim();
                let Some(bias) = network.exact_biases().get(bias_index) else {
                    return false;
                };
                bias_index += 1;
                if out_dim == 0
                    || in_dim != current_dim
                    || bias.len() != out_dim
                    || linear.weight().iter().any(|value| !value.is_finite())
                    || bias.iter().any(|value| !value.is_finite())
                {
                    return false;
                }
                let Some(next_columns) = columns.checked_add(out_dim) else {
                    return false;
                };
                let Some(next_rows) = rows.checked_add(out_dim) else {
                    return false;
                };
                let weight_nonzeros = linear
                    .weight()
                    .iter()
                    .filter(|&&value| value != 0.0)
                    .count();
                let Some(next_nonzeros) = nonzeros
                    .checked_add(weight_nonzeros)
                    .and_then(|count| count.checked_add(out_dim))
                else {
                    return false;
                };
                columns = next_columns;
                rows = next_rows;
                nonzeros = next_nonzeros;
                current_dim = out_dim;
            }
            Layer::ReLU(_) => {
                // Worst case: every coordinate is unstable. The continuous
                // hull adds one y column, y>=x (two nonzeros), and its upper
                // facet (two nonzeros) per coordinate. y>=0 is the column LB.
                let Some((next_columns, next_rows, next_nonzeros)) =
                    add_relu_outer_resources(columns, rows, nonzeros, current_dim)
                else {
                    return false;
                };
                columns = next_columns;
                rows = next_rows;
                nonzeros = next_nonzeros;
            }
            _ => return false,
        }
        if columns > MAX_COLUMNS || rows > MAX_ROWS || nonzeros > MAX_NONZEROS {
            return false;
        }
    }
    bias_index == network.exact_biases().len()
        && current_dim == expected_output_dim
        && rows <= MAX_ROWS
        && nonzeros <= MAX_NONZEROS
}

fn add_relu_outer_resources(
    columns: usize,
    rows: usize,
    nonzeros: usize,
    width: usize,
) -> Option<(usize, usize, usize)> {
    let added_rows = width.checked_mul(2)?;
    let added_nonzeros = width.checked_mul(4)?;
    Some((
        columns.checked_add(width)?,
        rows.checked_add(added_rows)?,
        nonzeros.checked_add(added_nonzeros)?,
    ))
}

fn add_output_closure(
    encoder: &mut MipEncoder,
    constraints: &[CertifiedRelationalOutputAtom],
) -> Result<(), ()> {
    for constraint in constraints {
        let result = match constraint {
            CertifiedRelationalOutputAtom::LessEq(i, j)
            | CertifiedRelationalOutputAtom::LessThan(i, j) => encoder.constrain_output_leq(*i, *j),
            CertifiedRelationalOutputAtom::GreaterEq(i, j)
            | CertifiedRelationalOutputAtom::GreaterThan(i, j) => {
                encoder.constrain_output_geq(*i, *j)
            }
        };
        result.map_err(|_| ())?;
    }
    Ok(())
}

fn problem_within_limits(problem: &ny_mip::ir::MilpProblem) -> bool {
    if problem.num_cols() > MAX_COLUMNS || problem.num_rows() > MAX_ROWS {
        return false;
    }
    problem
        .rows()
        .iter()
        .try_fold(0_usize, |count, row| count.checked_add(row.coeffs.len()))
        .is_some_and(|count| count <= MAX_NONZEROS)
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::sync::Mutex;

    use ndarray::{array, Array1, Array2};
    use ny_onnx::{onnx_proto, ShapeInferencePolicy};
    use ny_propagate::layers::{AddConstantLayer, LinearLayer, ReLULayer, ReshapeLayer};
    use prost::Message;

    use super::*;

    static AFFINE_ROOT_TEST_LOCK: Mutex<()> = Mutex::new(());

    fn bounded(lower: Vec<f32>, upper: Vec<f32>) -> BoundedTensor {
        BoundedTensor::new(
            Array1::from_vec(lower).into_dyn(),
            Array1::from_vec(upper).into_dyn(),
        )
        .expect("ordered finite test box")
    }

    fn input_from_spec(spec: &VnnLibSpec) -> BoundedTensor {
        let (lower, upper) = spec.split_input_bounds_f32();
        bounded(lower, upper)
    }

    fn linear(weight: Array2<f32>, bias: Option<Vec<f32>>) -> Layer {
        Layer::Linear(
            LinearLayer::new(weight, bias.map(Array1::from_vec)).expect("valid finite Linear"),
        )
    }

    fn sequential(layers: impl IntoIterator<Item = Layer>) -> Network {
        let mut network = Network::new();
        for layer in layers {
            network.add_layer(layer);
        }
        network
    }

    fn spec(
        input_bounds: Vec<(f64, f64)>,
        num_outputs: usize,
        constraints: Vec<OutputConstraint>,
    ) -> VnnLibSpec {
        let mut spec = VnnLibSpec::new();
        spec.num_inputs = input_bounds.len();
        spec.num_outputs = num_outputs;
        spec.input_bounds = input_bounds;
        spec.output_constraints = constraints;
        spec
    }

    fn core_attempt(
        network: &Network,
        input: &BoundedTensor,
        spec: &VnnLibSpec,
        exact_input_bounds: &[(f64, f64)],
    ) -> AffineRootFarkasAttempt {
        if let Err(reason) = authoritative_constraints(spec, input) {
            return AffineRootFarkasAttempt::Declined(reason);
        }
        let constraints = match test_certified_constraints(spec) {
            Some(constraints) => constraints,
            None => {
                return AffineRootFarkasAttempt::Declined(
                    AffineRootFarkasDecline::UnsupportedProperty,
                )
            }
        };
        let Some(admission) = CertifiedLinearLowerWorkerAdmission::try_acquire() else {
            return AffineRootFarkasAttempt::Declined(AffineRootFarkasDecline::WorkerBusy);
        };
        try_sequential_affine_root_farkas(
            network,
            input,
            spec,
            &constraints,
            exact_input_bounds,
            Instant::now() + Duration::from_mins(1),
            admission,
        )
    }

    fn test_certified_constraints(spec: &VnnLibSpec) -> Option<Vec<CertifiedRelationalOutputAtom>> {
        spec.output_constraints
            .iter()
            .map(|constraint| match constraint {
                OutputConstraint::LessEq(i, j) => {
                    Some(CertifiedRelationalOutputAtom::LessEq(*i, *j))
                }
                OutputConstraint::GreaterEq(i, j) => {
                    Some(CertifiedRelationalOutputAtom::GreaterEq(*i, *j))
                }
                OutputConstraint::LessThan(i, j) => {
                    Some(CertifiedRelationalOutputAtom::LessThan(*i, *j))
                }
                OutputConstraint::GreaterThan(i, j) => {
                    Some(CertifiedRelationalOutputAtom::GreaterThan(*i, *j))
                }
                _ => None,
            })
            .collect()
    }

    fn tensor_value_info(name: &str, shape: &[i64]) -> onnx_proto::ValueInfoProto {
        let dim = shape
            .iter()
            .map(|&value| onnx_proto::tensor_shape_proto::Dimension {
                value: Some(onnx_proto::tensor_shape_proto::dimension::Value::DimValue(
                    value,
                )),
            })
            .collect();
        onnx_proto::ValueInfoProto {
            name: name.to_string(),
            r#type: Some(onnx_proto::TypeProto {
                tensor_type: Some(onnx_proto::TensorTypeProto {
                    elem_type: 1,
                    shape: Some(onnx_proto::TensorShapeProto { dim }),
                }),
            }),
        }
    }

    fn tensor_f32(name: &str, shape: &[i64], values: &[f32]) -> onnx_proto::TensorProto {
        onnx_proto::TensorProto {
            dims: shape.to_vec(),
            data_type: 1,
            name: name.to_string(),
            float_data: values.to_vec(),
            ..Default::default()
        }
    }

    fn node(name: &str, op_type: &str, inputs: &[&str], output: &str) -> onnx_proto::NodeProto {
        onnx_proto::NodeProto {
            input: inputs.iter().map(|input| (*input).to_string()).collect(),
            output: vec![output.to_string()],
            name: name.to_string(),
            op_type: op_type.to_string(),
            ..Default::default()
        }
    }

    fn int_attr(name: &str, value: i64) -> onnx_proto::AttributeProto {
        onnx_proto::AttributeProto {
            name: name.to_string(),
            i: Some(value),
            r#type: onnx_proto::attribute_type::INT,
            ..Default::default()
        }
    }

    fn onnx_bytes(
        nodes: Vec<onnx_proto::NodeProto>,
        initializers: Vec<onnx_proto::TensorProto>,
        outputs: usize,
    ) -> Vec<u8> {
        let graph = onnx_proto::GraphProto {
            node: nodes,
            name: "affine-root-farkas-fixture".to_string(),
            initializer: initializers,
            input: vec![tensor_value_info("X", &[1, 1])],
            output: vec![tensor_value_info("Y", &[1, outputs as i64])],
            ..Default::default()
        };
        onnx_proto::ModelProto {
            ir_version: 9,
            opset_import: vec![onnx_proto::OperatorSetIdProto {
                domain: String::new(),
                version: 13,
            }],
            producer_name: "ny-affine-root-farkas-test".to_string(),
            graph: Some(graph),
            ..Default::default()
        }
        .encode_to_vec()
    }

    fn direct_gemm_bytes(weights: &[f32], bias: &[f32]) -> Vec<u8> {
        let mut gemm = node("gemm", "Gemm", &["X", "W", "B"], "Y");
        gemm.attribute.push(int_attr("transB", 1));
        onnx_bytes(
            vec![gemm],
            vec![
                tensor_f32("W", &[weights.len() as i64, 1], weights),
                tensor_f32("B", &[bias.len() as i64], bias),
            ],
            weights.len(),
        )
    }

    fn temp_file(suffix: &str, bytes: &[u8]) -> tempfile::NamedTempFile {
        let mut file = tempfile::Builder::new()
            .suffix(suffix)
            .tempfile()
            .expect("create temporary fixture");
        file.write_all(bytes).expect("write temporary fixture");
        file.flush().expect("flush temporary fixture");
        file
    }

    fn temp_vnnlib(source: &str) -> tempfile::NamedTempFile {
        temp_file(".vnnlib", source.as_bytes())
    }

    fn test_load_config() -> OnnxLoadConfig {
        OnnxLoadConfig::default().with_shape_inference_policy(ShapeInferencePolicy::Skip)
    }

    #[test]
    fn coupled_empty_affine_region_proves_but_feasible_sibling_declines() {
        let _guard = AFFINE_ROOT_TEST_LOCK.lock().unwrap();
        let network = sequential([linear(
            array![[1.0], [-1.0], [1.0], [-1.0]],
            Some(vec![0.0, -1.0, 0.0, 1.0]),
        )]);
        let input = bounded(vec![-1.0], vec![1.0]);
        let empty = spec(
            vec![(-1.0, 1.0)],
            4,
            vec![
                OutputConstraint::LessEq(0, 1),
                OutputConstraint::GreaterEq(2, 3),
            ],
        );
        assert!(matches!(
            core_attempt(&network, &input, &empty, &[(-1.0, 1.0)]),
            AffineRootFarkasAttempt::Proved(_)
        ));

        let feasible = spec(
            vec![(-1.0, 1.0)],
            4,
            vec![
                OutputConstraint::LessEq(0, 3),
                OutputConstraint::GreaterEq(2, 1),
            ],
        );
        assert_eq!(
            core_attempt(&network, &input, &feasible, &[(-1.0, 1.0)]),
            AffineRootFarkasAttempt::Declined(AffineRootFarkasDecline::NotProved)
        );
    }

    #[test]
    fn stable_relu_shape_alignment_and_exact_constant_fold_prove() {
        let _guard = AFFINE_ROOT_TEST_LOCK.lock().unwrap();
        let input = bounded(vec![0.0], vec![1.0]);
        let active = sequential([
            linear(array![[1.0]], Some(vec![1.0])),
            Layer::Reshape(ReshapeLayer::new(vec![1])),
            Layer::ReLU(ReLULayer::new()),
            linear(array![[1.0], [0.0]], Some(vec![0.0, 0.0])),
        ]);
        let active_empty = spec(vec![(0.0, 1.0)], 2, vec![OutputConstraint::LessEq(0, 1)]);
        assert!(matches!(
            core_attempt(&active, &input, &active_empty, &[(0.0, 1.0)]),
            AffineRootFarkasAttempt::Proved(_)
        ));

        let inactive = sequential([
            linear(array![[1.0]], Some(vec![-2.0])),
            Layer::ReLU(ReLULayer::new()),
            linear(array![[1.0], [0.0]], Some(vec![0.0, 1.0])),
        ]);
        let inactive_empty = spec(vec![(0.0, 1.0)], 2, vec![OutputConstraint::GreaterEq(0, 1)]);
        assert!(matches!(
            core_attempt(&inactive, &input, &inactive_empty, &[(0.0, 1.0)]),
            AffineRootFarkasAttempt::Proved(_)
        ));

        let folded = sequential([
            linear(array![[1.0], [0.0]], None),
            Layer::AddConstant(AddConstantLayer::new(array![1.0_f32, 0.0].into_dyn())),
            Layer::Reshape(ReshapeLayer::new(vec![2])),
        ]);
        let folded_empty = spec(vec![(0.0, 1.0)], 2, vec![OutputConstraint::LessEq(0, 1)]);
        assert!(matches!(
            core_attempt(&folded, &input, &folded_empty, &[(0.0, 1.0)]),
            AffineRootFarkasAttempt::Proved(_)
        ));
    }

    #[test]
    fn unstable_relu_outer_hull_proves_and_feasible_sibling_declines() {
        let _guard = AFFINE_ROOT_TEST_LOCK.lock().unwrap();
        let input = bounded(vec![-1.0], vec![1.0]);
        let empty = sequential([
            linear(array![[1.0]], None),
            Layer::ReLU(ReLULayer::new()),
            linear(array![[1.0], [0.0]], Some(vec![0.0, -1.0])),
        ]);
        let output_leq_sibling = spec(vec![(-1.0, 1.0)], 2, vec![OutputConstraint::LessEq(0, 1)]);
        assert!(matches!(
            core_attempt(&empty, &input, &output_leq_sibling, &[(-1.0, 1.0)]),
            AffineRootFarkasAttempt::Proved(_)
        ));

        let feasible = sequential([
            linear(array![[1.0]], None),
            Layer::ReLU(ReLULayer::new()),
            linear(array![[1.0], [0.0]], Some(vec![0.0, 1.0])),
        ]);
        assert_eq!(
            core_attempt(&feasible, &input, &output_leq_sibling, &[(-1.0, 1.0)]),
            AffineRootFarkasAttempt::Declined(AffineRootFarkasDecline::NotProved)
        );

        let extreme_input = bounded(vec![-f32::MAX], vec![f32::MIN_POSITIVE]);
        let extreme = sequential([
            Layer::ReLU(ReLULayer::new()),
            linear(array![[1.0], [0.0]], Some(vec![0.0, -1.0])),
        ]);
        let extreme_bounds = (f64::from(-f32::MAX), f64::from(f32::MIN_POSITIVE));
        let extreme_spec = spec(
            vec![extreme_bounds],
            2,
            vec![OutputConstraint::LessEq(0, 1)],
        );
        assert_eq!(
            core_attempt(&extreme, &extreme_input, &extreme_spec, &[extreme_bounds]),
            AffineRootFarkasAttempt::Declined(AffineRootFarkasDecline::InexactReluHull),
            "an extreme binary32 exponent gap whose u-l is inexact in f64 must fail closed"
        );

        let fixed_zero = sequential([linear(array![[0.0], [0.0]], None)]);
        let boundary = spec(vec![(-1.0, 1.0)], 2, vec![OutputConstraint::LessThan(0, 1)]);
        assert_eq!(
            core_attempt(&fixed_zero, &input, &boundary, &[(-1.0, 1.0)]),
            AffineRootFarkasAttempt::Declined(AffineRootFarkasDecline::NotProved),
            "strict-only boundary emptiness must not cross the non-strict closure gate"
        );

        let admission = CertifiedLinearLowerWorkerAdmission::try_acquire()
            .expect("the exact worker slot is free after preceding attempts");
        assert_eq!(
            core_attempt(&fixed_zero, &input, &output_leq_sibling, &[(-1.0, 1.0)]),
            AffineRootFarkasAttempt::Declined(AffineRootFarkasDecline::WorkerBusy)
        );
        drop(admission);
    }

    #[test]
    fn property_and_activation_preflights_fail_closed_before_ibp() {
        let input = bounded(vec![-1.0], vec![1.0]);
        let constant = spec(
            vec![(-1.0, 1.0)],
            2,
            vec![OutputConstraint::LessEqConst(0, 0.1)],
        );
        assert!(matches!(
            authoritative_constraints(&constant, &input),
            Err(AffineRootFarkasDecline::UnsupportedProperty)
        ));

        let mut disjunction = spec(vec![(-1.0, 1.0)], 2, vec![OutputConstraint::LessEq(0, 1)]);
        disjunction.is_disjunction = true;
        assert!(matches!(
            authoritative_constraints(&disjunction, &input),
            Err(AffineRootFarkasDecline::UnsupportedProperty)
        ));

        let mut high_volume = Network::new();
        high_volume.add_layer(linear(Array2::zeros((MAX_COLUMNS, 1)), None));
        for _ in 0..244 {
            high_volume.add_layer(Layer::ReLU(ReLULayer::new()));
        }
        assert!(raw_network_within_limits(&high_volume));
        assert!(high_volume.layers().len() <= MAX_RAW_LAYERS);
        assert!(!original_activation_volume_within_limits(&high_volume, 1));

        assert_eq!(add_relu_outer_resources(1, 2, 3, 4), Some((5, 10, 19)));
        assert_eq!(
            add_relu_outer_resources(usize::MAX, 0, 0, 1),
            None,
            "column accounting must fail closed on overflow"
        );
        assert_eq!(
            add_relu_outer_resources(0, usize::MAX, 0, 1),
            None,
            "row accounting must fail closed on overflow"
        );
        assert_eq!(
            add_relu_outer_resources(0, 0, usize::MAX, 1),
            None,
            "nonzero accounting must fail closed on overflow"
        );
        assert!(
            add_relu_outer_resources(MAX_COLUMNS, 0, 0, 1)
                .is_some_and(|(columns, _, _)| columns > MAX_COLUMNS),
            "the caller's hard cap must reject the overflow-checked result"
        );
    }

    #[test]
    fn exact_decimal_input_is_source_reparsed_outward() {
        let property = temp_vnnlib(
            "(declare-const X_0 Real)\n\
             (declare-const Y_0 Real)\n\
             (declare-const Y_1 Real)\n\
             (assert (= X_0 0.1))\n\
             (assert (<= Y_0 Y_1))\n",
        );
        let ordinary = ny_onnx::vnnlib::load_vnnlib(property.path()).expect("ordinary parse");
        let input = input_from_spec(&ordinary);
        let (certified, outputs) =
            certified_source_property(Some(property.path()), &ordinary, &input, 1)
                .expect("direct exact-decimal box and relational output");
        assert!(certified[0].0 < ordinary.input_bounds[0].0);
        assert!(certified[0].1 >= ordinary.input_bounds[0].1);
        assert_ne!(certified[0].0.to_bits(), certified[0].1.to_bits());
        assert_eq!(outputs, vec![CertifiedRelationalOutputAtom::LessEq(0, 1)]);
    }

    #[test]
    fn decimal_output_cancellation_cannot_enter_proof_ir() {
        let property = temp_vnnlib(
            "(declare-const X_0 Real)\n\
             (declare-const Y_0 Real)\n\
             (declare-const Y_1 Real)\n\
             (assert (>= X_0 -1.0))\n\
             (assert (<= X_0 1.0))\n\
             (assert (<= (+ Y_0 1.0) (+ Y_1 1.00000000000000001)))\n",
        );
        let ordinary = ny_onnx::vnnlib::load_vnnlib(property.path())
            .expect("ordinary parser normalizes the f64 collision");
        assert_eq!(
            ordinary.output_constraints.as_slice(),
            &[OutputConstraint::LessEq(0, 1)]
        );
        assert!(matches!(
            certified_source_property(
                Some(property.path()),
                &ordinary,
                &input_from_spec(&ordinary),
                1,
            ),
            Err(AffineRootFarkasDecline::UncertifiedPropertySource)
        ));
    }

    #[test]
    fn strict_reload_accepts_direct_authored_gemm_and_rejects_surrogates() {
        let config = test_load_config();
        let direct = temp_file(".onnx", &direct_gemm_bytes(&[1.0, -1.0], &[0.0, 0.0]));
        let direct_network = strict_reload_affine_network(direct.path(), &config)
            .expect("direct authored Gemm is a sealed affine source");
        assert!(matches!(direct_network.layers(), [Layer::Linear(_)]));

        let mut derived_gemm = node("gemm", "Gemm", &["X", "W", "B"], "Y");
        derived_gemm.attribute.push(int_attr("transB", 1));
        let derived = temp_file(
            ".onnx",
            &onnx_bytes(
                vec![node("derive_w", "Add", &["WA", "WB"], "W"), derived_gemm],
                vec![
                    tensor_f32("WA", &[2, 1], &[0.5, -0.5]),
                    tensor_f32("WB", &[2, 1], &[0.5, -0.5]),
                    tensor_f32("B", &[2], &[0.0, 0.0]),
                ],
                2,
            ),
        );
        assert!(
            matches!(
                strict_reload_affine_network(derived.path(), &config),
                Err(AffineRootFarkasDecline::UntrustedModelSource)
            ),
            "a constant-fold result must not masquerade as an authored parameter"
        );

        let mut gemm = node("gemm", "Gemm", &["X", "W", "B"], "H");
        gemm.attribute.push(int_attr("transB", 1));
        let batch_norm = node("bn", "BatchNormalization", &["H", "S", "BB", "M", "V"], "Y");
        let bn = temp_file(
            ".onnx",
            &onnx_bytes(
                vec![gemm, batch_norm],
                vec![
                    tensor_f32("W", &[2, 1], &[1.0, -1.0]),
                    tensor_f32("B", &[2], &[0.0, 0.0]),
                    tensor_f32("S", &[2], &[1.0, 1.0]),
                    tensor_f32("BB", &[2], &[0.0, 0.0]),
                    tensor_f32("M", &[2], &[0.0, 0.0]),
                    tensor_f32("V", &[2], &[1.0, 1.0]),
                ],
                2,
            ),
        );
        assert!(
            matches!(
                strict_reload_affine_network(bn.path(), &config),
                Err(AffineRootFarkasDecline::UntrustedModelSource)
            ),
            "PreserveRaw must expose and reject BN rather than certify its rounded fold"
        );

        let nnet = temp_file(".nnet", b"not consulted");
        assert!(matches!(
            strict_reload_affine_network(nnet.path(), &config),
            Err(AffineRootFarkasDecline::UntrustedModelSource)
        ));
        let merge = config
            .clone()
            .with_optimization_flag(OnnxOptimizationFlag::MergeLinear);
        assert!(matches!(
            strict_reload_affine_network(direct.path(), &merge),
            Err(AffineRootFarkasDecline::UntrustedModelSource)
        ));

        let float32_bytes = direct_gemm_bytes(&[1.0, -1.0], &[0.0, 0.0]);
        for (input_dtype, output_dtype) in [(7, 1), (1, 7)] {
            let mut proto = onnx_proto::ModelProto::decode(float32_bytes.as_slice())
                .expect("decode dtype fixture");
            let graph = proto.graph.as_mut().expect("fixture graph");
            graph.input[0]
                .r#type
                .as_mut()
                .and_then(|value| value.tensor_type.as_mut())
                .expect("fixture input tensor type")
                .elem_type = input_dtype;
            graph.output[0]
                .r#type
                .as_mut()
                .and_then(|value| value.tensor_type.as_mut())
                .expect("fixture output tensor type")
                .elem_type = output_dtype;
            let non_float = temp_file(".onnx", &proto.encode_to_vec());
            assert!(matches!(
                strict_reload_affine_network(non_float.path(), &config),
                Err(AffineRootFarkasDecline::UntrustedModelSource)
            ));
        }
    }

    #[test]
    fn production_strict_reload_and_source_property_prove_only_empty_region() {
        let _guard = AFFINE_ROOT_TEST_LOCK.lock().unwrap();
        let model = temp_file(
            ".onnx",
            &direct_gemm_bytes(&[1.0, -1.0, 1.0, -1.0], &[0.0, -1.0, 0.0, 1.0]),
        );
        let empty_property = temp_vnnlib(
            "(declare-const X_0 Real)\n\
             (declare-const Y_0 Real)\n\
             (declare-const Y_1 Real)\n\
             (declare-const Y_2 Real)\n\
             (declare-const Y_3 Real)\n\
             (assert (>= X_0 -1.0))\n\
             (assert (<= X_0 1.0))\n\
             (assert (and (<= Y_0 Y_1) (>= Y_2 Y_3)))\n",
        );
        let empty = ny_onnx::vnnlib::load_vnnlib(empty_property.path()).expect("empty property");
        let input = input_from_spec(&empty);
        let config = test_load_config();
        assert!(matches!(
            try_affine_root_farkas(
                model.path(),
                &config,
                Some(empty_property.path()),
                &input,
                &empty,
                VerificationArtifactAuthority::VerdictOnly,
                Some(Instant::now() + Duration::from_mins(1)),
                false,
            ),
            AffineRootFarkasAttempt::Proved(_)
        ));

        let feasible_property = temp_vnnlib(
            "(declare-const X_0 Real)\n\
             (declare-const Y_0 Real)\n\
             (declare-const Y_1 Real)\n\
             (declare-const Y_2 Real)\n\
             (declare-const Y_3 Real)\n\
             (assert (>= X_0 -1.0))\n\
             (assert (<= X_0 1.0))\n\
             (assert (and (<= Y_0 Y_3) (>= Y_2 Y_1)))\n",
        );
        let feasible =
            ny_onnx::vnnlib::load_vnnlib(feasible_property.path()).expect("feasible property");
        assert_eq!(
            try_affine_root_farkas(
                model.path(),
                &config,
                Some(feasible_property.path()),
                &input_from_spec(&feasible),
                &feasible,
                VerificationArtifactAuthority::VerdictOnly,
                Some(Instant::now() + Duration::from_mins(1)),
                false,
            ),
            AffineRootFarkasAttempt::Declined(AffineRootFarkasDecline::NotProved)
        );

        let admission = CertifiedLinearLowerWorkerAdmission::try_acquire()
            .expect("worker lease is free after synchronous proof attempts");
        assert_eq!(
            try_affine_root_farkas(
                model.path(),
                &config,
                Some(empty_property.path()),
                &input,
                &empty,
                VerificationArtifactAuthority::VerdictOnly,
                Some(Instant::now() + Duration::from_mins(1)),
                false,
            ),
            AffineRootFarkasAttempt::Declined(AffineRootFarkasDecline::WorkerBusy),
            "valid source files must be shed before VNN-LIB reparse or ONNX reload"
        );
        drop(admission);
    }

    #[test]
    fn authority_and_deadline_decline_before_filesystem_authority() {
        let input = bounded(vec![-1.0], vec![1.0]);
        let property = spec(vec![(-1.0, 1.0)], 2, vec![OutputConstraint::LessEq(0, 1)]);
        assert_eq!(
            try_affine_root_farkas(
                Path::new("missing.onnx"),
                &test_load_config(),
                None,
                &input,
                &property,
                VerificationArtifactAuthority::CertificateExport,
                Some(Instant::now() + Duration::from_mins(1)),
                false,
            ),
            AffineRootFarkasAttempt::Declined(AffineRootFarkasDecline::ArtifactAuthority)
        );
        assert_eq!(
            try_affine_root_farkas(
                Path::new("missing.onnx"),
                &test_load_config(),
                None,
                &input,
                &property,
                VerificationArtifactAuthority::VerdictOnly,
                Some(
                    Instant::now()
                        .checked_sub(Duration::from_millis(1))
                        .expect("one millisecond fits before now"),
                ),
                false,
            ),
            AffineRootFarkasAttempt::Declined(AffineRootFarkasDecline::Deadline)
        );
    }
}
