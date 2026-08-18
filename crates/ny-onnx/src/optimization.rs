// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Joint ONNX + VNN-LIB optimization passes.

use crate::vnnlib::{OutputConstraint, VnnLibSpec};
use crate::{GraphNetworkOptions, OnnxLoadConfig, OnnxModel};
use ny_core::LayerType;
use sha2::{Digest, Sha256};
use tracing::{debug, info, warn};

/// Dark gate for [`strip_terminal_softmax`].
///
/// Armed ONLY by the byte string `"1"`. Unset, empty, `"0"`, `"01"`, `"true"`,
/// `" 1"`, and non-UTF-8 all mean OFF. Compared as [`OsStr`] so a non-UTF-8
/// value cannot be lossily coerced into an arming value.
pub const STRIP_TERMINAL_SOFTMAX_ENV: &str = "NY_STRIP_TERMINAL_SOFTMAX";

// Exact VNN-COMP traffic model whose terminal-logit arithmetic has a checked
// lattice certificate. The hash authenticates the bytes that the caller also
// parsed; a filename/category match is intentionally insufficient proof
// authority.
const TRAFFIC_30_MODEL_SHA256: &str =
    "9fb711d9b0019889d648df23ce477a083f0f6d6ef13f732de284a607f0abdce6";

// A deliberately conservative upper bound around the measured f32
// exp/division tie window (~1.2e-7). A certificate must prove that every pair
// of distinct logits is separated by strictly more than this value.
const F32_SOFTMAX_TIE_WINDOW_UPPER_BOUND: f64 = 1.0e-6;

#[derive(Debug, Clone, Copy)]
struct CertifiedLogitLattice {
    model_sha256: &'static str,
    min_distinct_gap: f64,
    rule: &'static str,
}

const TRAFFIC_30_LOGIT_LATTICE: CertifiedLogitLattice = CertifiedLogitLattice {
    model_sha256: TRAFFIC_30_MODEL_SHA256,
    min_distinct_gap: 2.0,
    // For these exact bytes, the tensor feeding the terminal Softmax is a
    // bias-free 23,328-by-43 MatMul. Its shared input is the output of Sign
    // (each entry is -1, 0, or 1), and every matrix entry is exactly -1 or 1.
    // Therefore every pairwise logit difference is a sum of terms in
    // {-2, 0, 2}: it is either zero or has magnitude >= 2. All partial sums
    // are bounded by 23,328 < 2^24, so f32 represents the integer arithmetic
    // exactly irrespective of accumulation order.
    rule: "bias-free Sign x {-1,+1} MatMul; pairwise differences lie in 2Z",
};

fn authenticate_logit_lattice(model_bytes: &[u8]) -> Result<CertifiedLogitLattice, String> {
    let digest = format!("{:x}", Sha256::digest(model_bytes));
    let certificate = match digest.as_str() {
        TRAFFIC_30_MODEL_SHA256 => TRAFFIC_30_LOGIT_LATTICE,
        _ => {
            return Err(format!(
                "terminal Softmax model SHA-256 {digest} has no certified logit-lattice rule"
            ));
        }
    };
    if certificate.min_distinct_gap <= F32_SOFTMAX_TIE_WINDOW_UPPER_BOUND {
        return Err(
            "terminal Softmax lattice certificate does not clear the f32 tie window".to_string(),
        );
    }
    Ok(certificate)
}

/// Whether the [`STRIP_TERMINAL_SOFTMAX_ENV`] dark gate is armed.
///
/// Exact-match by construction: any byte sequence other than `1` is OFF.
fn strip_terminal_softmax_armed() -> bool {
    ny_levers::read(&ny_levers::decls::onnx::STRIP_TERMINAL_SOFTMAX)
        .value
        .as_bool()
}

#[derive(Debug, Clone)]
pub struct PeelOffReport {
    pub peeled: bool,
    pub layer_type: Option<LayerType>,
    pub reason: Option<String>,
}

impl PeelOffReport {
    fn skipped(reason: impl Into<String>) -> Self {
        Self {
            peeled: false,
            layer_type: None,
            reason: Some(reason.into()),
        }
    }

    fn peeled(layer_type: LayerType) -> Self {
        Self {
            peeled: true,
            layer_type: Some(layer_type),
            reason: None,
        }
    }
}

/// Compatibility seam for the quarantined constant-threshold terminal-Sigmoid
/// peel (#cgan-sigmoid-peel).
///
/// Even with `NY_SIGMOID_PEEL=1`, this entry point currently declines without
/// mutation. A measured original-graph differential rejected a peeled witness;
/// until region equivalence is independently repaired, a transformed UNSAT has
/// no proof authority. Provably exact relational Sigmoid comparisons remain
/// available only through the explicit legacy peel below.
pub fn peel_off_terminal_sigmoid_auto(
    model: &mut OnnxModel,
    vnnlib: &mut VnnLibSpec,
) -> PeelOffReport {
    // QUARANTINED pending root-cause (2026-07-17): with the peel ON, the cgan
    // upsample row's internal attack finds a witness that the trusted-ORT gate
    // REJECTS on the original graph (false counterexample) — the rewritten
    // clause region does not match the original on that row. The sat side is
    // ORT-gated, but a subset-direction defect would make the UNSAT side
    // unsound with no gate to catch it, so the path stays quarantined until the
    // region-equivalence bug is found and a differential region test lands.
    if std::env::var("NY_SIGMOID_PEEL").ok().as_deref() != Some("1") {
        return PeelOffReport::skipped("NY_SIGMOID_PEEL not enabled (opt-in)");
    }
    let Some(terminal) = terminal_layer_type(model) else {
        return PeelOffReport::skipped("no single terminal layer");
    };
    if terminal != LayerType::Sigmoid {
        return PeelOffReport::skipped("terminal layer is not Sigmoid");
    }
    let all = vnnlib
        .output_constraints
        .iter()
        .chain(vnnlib.output_constraint_clauses.iter().flatten());
    let mut any = false;
    for constraint in all {
        any = true;
        if !is_constant_threshold(constraint) {
            return PeelOffReport::skipped("non-constant-threshold constraint");
        }
    }
    if !any {
        return PeelOffReport::skipped("no output constraints to rewrite");
    }
    PeelOffReport::skipped(
        "constant-threshold terminal-Sigmoid peel is quarantined pending region-equivalence proof",
    )
}

fn terminal_layer_type(model: &OnnxModel) -> Option<LayerType> {
    if model.network.outputs.len() != 1 {
        return None;
    }
    let output_name = &model.network.outputs[0].name;
    model
        .network
        .layers
        .iter()
        .find(|layer| layer.outputs.iter().any(|o| o == output_name))
        .map(|layer| layer.layer_type.clone())
}

fn is_constant_threshold(constraint: &OutputConstraint) -> bool {
    matches!(
        constraint,
        OutputConstraint::LessEqConst(_, _)
            | OutputConstraint::GreaterEqConst(_, _)
            | OutputConstraint::LessThanConst(_, _)
            | OutputConstraint::GreaterThanConst(_, _)
    )
}

/// Authenticate that the exact terminal Softmax-family activation has one
/// normalization group in ny's concrete verification shape.
fn validate_terminal_normalization_single_group(
    model: &OnnxModel,
    expected: &LayerType,
) -> Result<(), String> {
    if !matches!(expected, LayerType::Softmax | LayerType::LogSoftmax) {
        return Err("single-group authentication requires Softmax or LogSoftmax".to_string());
    }
    if model.network.outputs.len() != 1 {
        return Err("multiple outputs not supported".to_string());
    }
    let output = &model.network.outputs[0];
    let layer = model
        .network
        .layers
        .iter()
        .find(|layer| layer.outputs.iter().any(|name| name == &output.name))
        .ok_or_else(|| "output tensor is not produced by a layer".to_string())?;
    if &layer.layer_type != expected {
        return Err(format!("terminal layer is not exactly {expected:?}"));
    }
    if layer.inputs.len() != 1 || layer.outputs.len() != 1 {
        return Err(format!(
            "only single-input/output {expected:?} layers can be peeled"
        ));
    }

    // The ONNX loader seals the authored/defaulted axis into every standard
    // Softmax/LogSoftmax LayerSpec. Missing or non-integer metadata is not
    // guessed here.
    let axis = match layer.attributes.get("axis") {
        Some(crate::AttributeValue::Int(axis)) => *axis,
        _ => {
            return Err(format!(
                "terminal {expected:?} has no authenticated integer axis"
            ));
        }
    };
    // Authenticate both sides of the shape-preserving activation.  In
    // particular, do not trust only the authored graph-output annotation: the
    // loader deliberately preserves a conflicting positive proto dimension
    // over inferred metadata, so a stale `[1, N]` output annotation could hide
    // a real `[B, N]` Softmax input and falsely claim there is one group.
    let shapes = model.tensor_shapes();
    let input_shape = shapes.get(&layer.inputs[0]).ok_or_else(|| {
        format!("terminal {expected:?} has no authenticated inferred input shape")
    })?;
    let output_shape = shapes.get(&output.name).ok_or_else(|| {
        format!("terminal {expected:?} has no authenticated inferred output shape")
    })?;
    if input_shape.is_empty() || input_shape.iter().any(|dimension| *dimension <= 0) {
        return Err(format!("terminal {expected:?} input shape is not concrete"));
    }
    if output_shape.is_empty() || output_shape.iter().any(|dimension| *dimension <= 0) {
        return Err(format!(
            "terminal {expected:?} output shape is not concrete"
        ));
    }
    if input_shape != output_shape {
        return Err(format!(
            "terminal {expected:?} inferred input/output shapes disagree"
        ));
    }
    let shape = input_shape;
    let rank = i64::try_from(shape.len())
        .map_err(|_| format!("terminal {expected:?} rank overflows i64"))?;
    let resolved_axis = if axis < 0 {
        axis.checked_add(rank)
            .ok_or_else(|| format!("terminal {expected:?} axis overflows"))?
    } else {
        axis
    };
    if !(0..rank).contains(&resolved_axis) {
        return Err(format!(
            "terminal {expected:?} axis is outside its output rank"
        ));
    }
    let resolved_axis = usize::try_from(resolved_axis)
        .map_err(|_| format!("terminal {expected:?} axis cannot be represented"))?;

    // Every coordinate outside `axis` selects an independent normalization
    // group. Requiring their product to be one proves every relational atom's
    // two indices share a denominator. Checked multiplication fails closed.
    let mut groups = 1usize;
    for (dimension_index, &dimension) in shape.iter().enumerate() {
        if dimension_index == resolved_axis {
            continue;
        }
        let dimension = usize::try_from(dimension)
            .map_err(|_| format!("terminal {expected:?} dimension cannot be represented"))?;
        groups = groups
            .checked_mul(dimension)
            .ok_or_else(|| format!("terminal {expected:?} group count overflows"))?;
    }
    if groups != 1 {
        return Err(format!(
            "terminal {expected:?} has more than one normalization group"
        ));
    }
    Ok(())
}

/// Authenticate that no layer other than `producer_idx` reads the graph output,
/// i.e. the activation really is terminal.
///
/// A downstream consumer computes on PROBABILITIES; deleting the node silently
/// changes what it receives.
fn validate_terminal_activation_has_no_other_consumers(
    model: &OnnxModel,
    producer_idx: usize,
    output_name: &str,
) -> Result<(), String> {
    let has_other_consumers = model.network.layers.iter().enumerate().any(|(idx, layer)| {
        idx != producer_idx && layer.inputs.iter().any(|input| input == output_name)
    });
    if has_other_consumers {
        return Err("terminal Softmax output is consumed by other layers".to_string());
    }
    Ok(())
}

/// Authenticate that the property is EXACTLY an argmax-complement disjunction
/// over one true label, returning that label.
///
/// # Why this is load-bearing (measured, not hypothetical)
///
/// The real-arithmetic order-preservation proof (`softmax(z)_i >= softmax(z)_j`
/// iff `z_i >= z_j`) is exact, but the VNN-COMP reference checker evaluates the
/// ONNX in FLOAT32, and `expf` UNDERFLOWS TO EXACTLY `0.0f`. Measured on
/// `3_30_30_QConv_16_3_QConv_32_2_Dense_43_ep_30.onnx` over 301 points of the
/// real `model_30_idx_1703_eps_1.00000` input box (ORT 1.19.2): logit magnitudes
/// reach 2590 and **42 of 43 softmax outputs are exactly `0.0f` at every single
/// point** — the float output is literally one-hot.
///
/// So for a bare pairwise atom between two NON-argmax classes, e.g.
/// `(>= Y[0,0] Y[0,42])`:
///   * on the float32 ONNX, `p_0 >= p_42` holds at 301/301 sampled points
///     (`p_0 == p_42 == 0.0f`) — the property is SAT;
///   * on the peeled logits, `z_0 >= z_42` holds at 0/301 points
///     (midpoint `z_0 = 296`, `z_42 = 460`) — the property looks UNSAT.
///
/// Stripping the Softmax there manufactures a FALSE UNSAT. That atom satisfies
/// every model-side guard (terminal Softmax, one normalization group, shape
/// `[1, 43]`, 43 outputs, bare pairwise `GreaterEq`), so the model-side guards
/// alone DO NOT make the strip sound under the semantics that get scored.
///
/// The argmax-complement shape is what rescues it. With unsafe =
/// `∃ i != t. Y_i >= Y_t`:
///   * (SAT direction) `z_i >= z_t` implies `e_i >= e_t` implies `p_i >= p_t`,
///     because ORT's max-subtracted `expf` and the following division are both
///     monotone under round-to-nearest. Logit-unsafe always implies float-unsafe.
///   * (UNSAT direction) suppose `p_i >= p_t` for some `i != t`. If `t` is not
///     the strict logit argmax then some `z_j >= z_t` and logit-unsafe already
///     holds. If `t` IS the strict argmax then ORT's shift makes `e_t = expf(0)`
///     exactly `1.0f`, the maximum, so `p_i >= p_t` forces
///     `expf(z_i - z_t) >= 1.0f` — impossible unless `z_t - z_i` falls inside the
///     float32 tie window (~1.2e-7 absolute, covering both the `expf` and the
///     `/S` rounding).
///
/// Hence float-unsafe implies logit-unsafe except inside that tie window. See
/// the crate docs on the residual: measured minimum logit gap on this corpus is
/// `2.0` (these are Sign/binarized nets with integral logits), seven orders of
/// magnitude clear of the window — but that is a MEASUREMENT, not a proof, which
/// is exactly why this transform stays dark.
fn validate_argmax_complement_disjunction(vnnlib: &VnnLibSpec) -> Result<usize, String> {
    // A dual-network property compares outputs of two DIFFERENT invocations with
    // two DIFFERENT denominators; order preservation does not hold across them.
    // Checked EXPLICITLY rather than relying on the parser leaving the
    // constraint lists empty for dual specs.
    if vnnlib.dual_network.is_some() {
        return Err("dual-network spec: outputs may not share a softmax denominator".to_string());
    }
    if vnnlib.num_outputs < 2 {
        return Err("argmax-complement requires at least 2 outputs".to_string());
    }
    // Per-clause input boxes make clauses non-interchangeable; the argmax
    // reading below assumes one shared input region.
    if !vnnlib.per_clause_input_bounds.iter().all(|m| m.is_empty()) {
        return Err("per-clause input bounds are not supported by the strip".to_string());
    }
    if !vnnlib.is_disjunction {
        return Err("output constraints are not a top-level disjunction".to_string());
    }
    let clauses = &vnnlib.output_constraint_clauses;
    if clauses.len() != vnnlib.num_outputs - 1 {
        return Err(format!(
            "argmax-complement needs exactly {} clauses, found {}",
            vnnlib.num_outputs - 1,
            clauses.len()
        ));
    }

    let mut true_label: Option<usize> = None;
    let mut seen = vec![false; vnnlib.num_outputs];
    for clause in clauses {
        if clause.len() != 1 {
            return Err("argmax-complement clauses must be single atoms".to_string());
        }
        // NON-STRICT `>=` only. A strict atom is true on logits and false on
        // floats under a tie, which is the opposite error (a false SAT).
        let OutputConstraint::GreaterEq(challenger, label) = clause[0] else {
            return Err(
                "argmax-complement atoms must be non-strict output-vs-output GreaterEq".to_string(),
            );
        };
        if challenger == label {
            return Err("argmax-complement atom compares an output to itself".to_string());
        }
        if challenger >= vnnlib.num_outputs || label >= vnnlib.num_outputs {
            return Err("argmax-complement atom indexes outside the output vector".to_string());
        }
        match true_label {
            None => true_label = Some(label),
            Some(existing) if existing != label => {
                return Err(
                    "argmax-complement atoms do not share one right-hand true label".to_string(),
                );
            }
            Some(_) => {}
        }
        if seen[challenger] {
            return Err("argmax-complement repeats a challenger class".to_string());
        }
        seen[challenger] = true;
    }

    let true_label = true_label.ok_or_else(|| "no output constraints to rewrite".to_string())?;
    for (index, hit) in seen.iter().enumerate() {
        if index == true_label {
            if *hit {
                return Err("argmax-complement challenges the true label itself".to_string());
            }
        } else if !*hit {
            return Err(format!(
                "argmax-complement omits challenger class {index}: the property is not the \
                 complement of `class {true_label} is the argmax`"
            ));
        }
    }

    // The flat list is a redundant second representation of the same property.
    // Accept only "absent" or "exactly the clause concatenation" — a divergent
    // flat list would leave a second, unvalidated reading of the spec behind.
    if !vnnlib.output_constraints.is_empty() {
        let concatenated: Vec<OutputConstraint> = clauses.iter().flatten().cloned().collect();
        if vnnlib.output_constraints != concatenated {
            return Err(
                "flat output constraints disagree with the disjunctive clauses".to_string(),
            );
        }
    }

    Ok(true_label)
}

/// Full strip predicate: model side AND spec side. Returns the producer index,
/// the pre-softmax tensor name, and the authenticated true label.
fn strip_terminal_softmax_guard(
    model: &OnnxModel,
    vnnlib: &VnnLibSpec,
) -> Result<(usize, String, usize), String> {
    validate_terminal_normalization_single_group(model, &LayerType::Softmax)?;

    let output_name = model.network.outputs[0].name.clone();
    let (producer_idx, producer) = model
        .network
        .layers
        .iter()
        .enumerate()
        .find(|(_, layer)| layer.outputs.iter().any(|name| name == &output_name))
        .ok_or_else(|| "terminal Softmax producer disappeared".to_string())?;
    let pre_softmax = producer.inputs[0].clone();

    validate_terminal_activation_has_no_other_consumers(model, producer_idx, &output_name)?;

    // Pin the verification layout: the constraint index k must name coordinate k
    // of the pre-softmax tensor. `validate_terminal_normalization_single_group`
    // already proved input shape == output shape and one normalization group;
    // requiring exactly `[1, num_outputs]` also pins rank and layout, so a
    // spec/graph output-count mismatch cannot slip through.
    let shapes = model.tensor_shapes();
    let input_shape = shapes
        .get(&pre_softmax)
        .ok_or_else(|| "terminal Softmax input shape disappeared".to_string())?;
    let expected = [1i64, i64::try_from(vnnlib.num_outputs).unwrap_or(i64::MAX)];
    if input_shape.as_slice() != expected {
        return Err(format!(
            "terminal Softmax shape {input_shape:?} is not exactly [1, {}]",
            vnnlib.num_outputs
        ));
    }

    let true_label = validate_argmax_complement_disjunction(vnnlib)?;
    Ok((producer_idx, pre_softmax, true_label))
}

/// Dark-gated strip of a terminal Softmax for argmax-complement properties.
///
/// DEFAULT OFF. Runs only when [`STRIP_TERMINAL_SOFTMAX_ENV`] is exactly `"1"`;
/// with the gate unset this function reads nothing and mutates nothing.
///
/// When armed it still refuses unless EVERY guard holds — the union of the
/// model-side normalization-group authentication and the spec-side
/// argmax-complement authentication (see
/// [`validate_argmax_complement_disjunction`] for why the spec-side half is not
/// optional). The rewrite itself is the identity on the constraint list: index
/// `k` of the pre-softmax tensor is the same coordinate as index `k` of the
/// softmax output, which the shape guard proves.
///
/// Every outcome is logged. A silent refusal is a debugging trap.
pub fn strip_terminal_softmax(model: &mut OnnxModel, vnnlib: &mut VnnLibSpec) -> PeelOffReport {
    if !strip_terminal_softmax_armed() {
        // Dark path: no reads of `model`/`vnnlib`, no writes, no log noise.
        debug!(
            gate = STRIP_TERMINAL_SOFTMAX_ENV,
            "terminal Softmax strip is dark (gate not set to exactly \"1\")"
        );
        return PeelOffReport::skipped(format!(
            "{STRIP_TERMINAL_SOFTMAX_ENV} is not exactly \"1\" (dark by default)"
        ));
    }

    match strip_terminal_softmax_guard(model, vnnlib) {
        Ok(_) => {}
        Err(reason) => {
            warn!(
                gate = STRIP_TERMINAL_SOFTMAX_ENV,
                reason = %reason,
                "terminal Softmax strip REFUSED (fail-closed; model and spec untouched)"
            );
            return PeelOffReport::skipped(reason);
        }
    }

    // This legacy entry point has no access to the exact bytes parsed into
    // `model`, so it cannot authenticate a model-specific gap certificate.
    // The environment gate remains dark by default, but arming it alone must
    // never grant UNSAT authority inside the f32 Softmax tie window.
    let reason =
        "terminal Softmax strip requires exact model bytes and a certified logit-lattice rule";
    warn!(
        gate = STRIP_TERMINAL_SOFTMAX_ENV,
        reason, "terminal Softmax strip REFUSED (no model-byte authentication)"
    );
    PeelOffReport::skipped(reason)
}

/// Peel an authenticated terminal Softmax only when its verification shape is
/// one normalization group, every property atom rewrites, and the exact model
/// bytes have a model-specific lattice certificate clearing the f32 tie window.
fn peel_off_terminal_softmax_single_group_unbound(
    model: &mut OnnxModel,
    vnnlib: &mut VnnLibSpec,
    model_bytes: &[u8],
) -> PeelOffReport {
    let certificate = match authenticate_logit_lattice(model_bytes) {
        Ok(certificate) => certificate,
        Err(reason) => return PeelOffReport::skipped(reason),
    };
    if let Err(reason) = validate_certified_lattice_structure(model, certificate) {
        return PeelOffReport::skipped(reason);
    }
    peel_off_terminal_softmax_with_certificate(model, vnnlib, certificate)
}

/// Parse and peel one model from the exact same authenticated byte slice.
/// There is intentionally no public `(model, bytes)` API: callers cannot use
/// certified bytes as authority for an independently constructed model.
pub fn load_and_peel_terminal_softmax_single_group(
    name: &str,
    model_bytes: &[u8],
    config: &OnnxLoadConfig,
    vnnlib: &mut VnnLibSpec,
) -> ny_core::Result<(OnnxModel, PeelOffReport)> {
    let mut model = crate::load_onnx_bytes_with_config(name, model_bytes, config)?;
    let report = peel_off_terminal_softmax_single_group_unbound(&mut model, vnnlib, model_bytes);
    Ok((model, report))
}

/// As above, while retaining an immutable conversion of the authenticated
/// original graph for trusted SAT replay before consuming the model in peel.
pub fn load_and_peel_terminal_softmax_single_group_with_original_graph(
    name: &str,
    model_bytes: &[u8],
    config: &OnnxLoadConfig,
    graph_options: GraphNetworkOptions,
    vnnlib: &mut VnnLibSpec,
) -> ny_core::Result<(ny_propagate::GraphNetwork, PeelOffReport)> {
    let mut model = crate::load_onnx_bytes_with_config(name, model_bytes, config)?;
    let graph = model.to_graph_network_with_options(graph_options)?;
    let report = peel_off_terminal_softmax_single_group_unbound(&mut model, vnnlib, model_bytes);
    Ok((graph, report))
}

fn validate_certified_lattice_structure(
    model: &OnnxModel,
    certificate: CertifiedLogitLattice,
) -> Result<(), String> {
    if certificate.model_sha256 != TRAFFIC_30_MODEL_SHA256 {
        return Err("unknown terminal Softmax lattice certificate".to_string());
    }
    let output_name = model
        .network
        .outputs
        .first()
        .ok_or_else(|| "certified model has no output".to_string())?
        .name
        .as_str();
    let softmax = model
        .network
        .layers
        .iter()
        .find(|layer| layer.outputs.iter().any(|output| output == output_name))
        .ok_or_else(|| "certified model terminal producer is missing".to_string())?;
    let logits = softmax
        .inputs
        .first()
        .ok_or_else(|| "certified model Softmax input is missing".to_string())?;
    let matmul = model
        .network
        .layers
        .iter()
        .find(|layer| layer.outputs.iter().any(|output| output == logits))
        .ok_or_else(|| "certified model logit producer is missing".to_string())?;
    if matmul.layer_type != LayerType::MatMul || matmul.inputs.len() != 2 {
        return Err("certified model logits are not a bias-free two-input MatMul".to_string());
    }
    let sign_input = &matmul.inputs[0];
    let sign = model
        .network
        .layers
        .iter()
        .find(|layer| layer.outputs.iter().any(|output| output == sign_input))
        .ok_or_else(|| "certified model MatMul activation producer is missing".to_string())?;
    if sign.layer_type != LayerType::Sign {
        return Err("certified model MatMul activation is not the output of Sign".to_string());
    }
    let weights = model
        .weights
        .get(&matmul.inputs[1])
        .ok_or_else(|| "certified model MatMul weights are missing".to_string())?;
    if weights.shape() != [23_328, 43]
        || !weights.iter().all(|value| {
            value.to_bits() == 1.0f32.to_bits() || value.to_bits() == (-1.0f32).to_bits()
        })
    {
        return Err(
            "certified model MatMul weights are not exactly [23328,43] in {-1,+1}".to_string(),
        );
    }
    Ok(())
}

fn peel_off_terminal_softmax_with_certificate(
    model: &mut OnnxModel,
    vnnlib: &mut VnnLibSpec,
    certificate: CertifiedLogitLattice,
) -> PeelOffReport {
    if let Err(reason) = validate_terminal_normalization_single_group(model, &LayerType::Softmax) {
        return PeelOffReport::skipped(reason);
    }
    // This entry point is the typed traffic_signs_recognition_2023 treatment,
    // not a generic Softmax optimizer. Seal the benchmark's exact verified
    // output layout so an accidental future category/model reuse cannot widen
    // the authority granted by the VNN-COMP router.
    let output_name = &model.network.outputs[0].name;
    let Some(layer) = model
        .network
        .layers
        .iter()
        .find(|layer| layer.outputs.iter().any(|name| name == output_name))
    else {
        return PeelOffReport::skipped("traffic terminal Softmax producer disappeared");
    };
    let Some(input_shape) = model.tensor_shapes().get(&layer.inputs[0]) else {
        return PeelOffReport::skipped("traffic terminal Softmax input shape disappeared");
    };
    if input_shape.as_slice() != [1, 43] {
        return PeelOffReport::skipped("traffic terminal Softmax shape is not exactly [1, 43]");
    }
    if vnnlib.num_outputs != 43 {
        return PeelOffReport::skipped("traffic VNN-LIB output count is not exactly 43");
    }
    // The model-side guards above are NOT sufficient under the float32 semantics
    // the VNN-COMP reference checker actually runs: `expf` underflows to exactly
    // 0.0f, and 42 of 43 softmax outputs are measured to be exactly 0.0f at every
    // sampled point of the real input boxes. A bare pairwise atom between two
    // non-argmax classes is therefore SAT on the original ONNX and UNSAT on the
    // logits. Only the argmax-complement shape closes that gap.
    // See `validate_argmax_complement_disjunction` for the measurement and proof.
    if let Err(reason) = validate_argmax_complement_disjunction(vnnlib) {
        warn!(
            reason = %reason,
            "traffic terminal Softmax peel REFUSED (not an argmax-complement property)"
        );
        return PeelOffReport::skipped(reason);
    }
    let (producer_idx, pre_softmax, true_label) = match strip_terminal_softmax_guard(model, vnnlib)
    {
        Ok(plan) => plan,
        Err(reason) => return PeelOffReport::skipped(reason),
    };
    info!(
        model_sha256 = certificate.model_sha256,
        lattice_rule = certificate.rule,
        certified_min_distinct_gap = certificate.min_distinct_gap,
        f32_tie_window_upper_bound = F32_SOFTMAX_TIE_WINDOW_UPPER_BOUND,
        graph_output = %model.network.outputs[0].name,
        pre_softmax_output = %pre_softmax,
        true_label,
        num_outputs = vnnlib.num_outputs,
        "stripping terminal Softmax under authenticated lattice certificate"
    );
    model.network.outputs[0].name = pre_softmax;
    model.network.layers.remove(producer_idx);
    PeelOffReport::peeled(LayerType::Softmax)
}

/// Attempt a sound terminal activation peel and VNN-LIB rewrite.
///
/// Softmax/LogSoftmax are refused here because this legacy API has no exact
/// model-byte lattice certificate. The typed traffic entry point above owns
/// the sole certified Softmax admission.
/// Sigmoid is admitted only for relational comparisons, where applying the
/// same strictly increasing scalar map to both operands is exactly invertible.
/// Constant-threshold Sigmoid rewriting remains quarantined.
///
/// Both spec representations are rewritten: the flat `output_constraints` AND
/// the disjunctive `output_constraint_clauses` (#cgan-sigmoid-peel — the cgan
/// band specs `Y>=b OR Y<=a` live ONLY in the clause list, with an empty flat
/// list). The peel fires when at least one representation is non-empty and
/// EVERY present constraint rewrites; otherwise it is skipped fail-closed.
pub fn peel_off_last_softmax_layer(
    model: &mut OnnxModel,
    vnnlib: &mut VnnLibSpec,
) -> PeelOffReport {
    if model.network.outputs.len() != 1 {
        return PeelOffReport::skipped("multiple outputs not supported");
    }

    if vnnlib.output_constraints.is_empty() && vnnlib.output_constraint_clauses.is_empty() {
        return PeelOffReport::skipped("no output constraints to rewrite");
    }

    let output_name = model.network.outputs[0].name.clone();
    let mut layer_idx = None;
    let mut layer_type = None;
    let mut layer_inputs: Option<Vec<String>> = None;
    let mut layer_outputs: Option<Vec<String>> = None;

    for (idx, layer) in model.network.layers.iter().enumerate() {
        if layer.outputs.iter().any(|o| o == &output_name) {
            layer_idx = Some(idx);
            layer_type = Some(layer.layer_type.clone());
            layer_inputs = Some(layer.inputs.clone());
            layer_outputs = Some(layer.outputs.clone());
            break;
        }
    }

    let Some(layer_idx) = layer_idx else {
        return PeelOffReport::skipped("output tensor is not produced by a layer");
    };
    let Some(layer_type) = layer_type else {
        return PeelOffReport::skipped("missing layer type for output");
    };
    let Some(layer_inputs) = layer_inputs else {
        return PeelOffReport::skipped("missing layer inputs");
    };
    let Some(layer_outputs) = layer_outputs else {
        return PeelOffReport::skipped("missing layer outputs");
    };

    if layer_outputs.len() != 1 || layer_inputs.len() != 1 {
        return PeelOffReport::skipped("only single-input/output layers can be peeled");
    }

    match &layer_type {
        LayerType::Softmax | LayerType::LogSoftmax => {
            return PeelOffReport::skipped(
                "terminal Softmax-family peel requires exact model bytes and a certified logit-lattice rule",
            );
        }
        LayerType::Sigmoid => {}
        _ => return PeelOffReport::skipped("terminal layer is not Softmax/LogSoftmax/Sigmoid"),
    }

    let has_other_consumers = model
        .network
        .layers
        .iter()
        .enumerate()
        .any(|(idx, layer)| idx != layer_idx && layer.inputs.iter().any(|i| i == &output_name));
    if has_other_consumers {
        return PeelOffReport::skipped("output tensor is consumed by other layers");
    }

    // Rewrite BOTH representations up front; mutate the spec only when every
    // present constraint rewrote (fail closed — a half-rewritten spec would
    // mix pre- and post-sigmoid coordinate systems).
    let new_flat = if vnnlib.output_constraints.is_empty() {
        Vec::new()
    } else {
        match rewrite_constraints_for_layer(layer_type.clone(), &vnnlib.output_constraints) {
            Ok(Some(constraints)) => constraints,
            Ok(None) => {
                return PeelOffReport::skipped("output constraints incompatible with peeling");
            }
            Err(reason) => {
                return PeelOffReport::skipped(format!(
                    "output constraints unsupported: {}",
                    reason
                ));
            }
        }
    };

    let mut new_clauses = Vec::with_capacity(vnnlib.output_constraint_clauses.len());
    for clause in &vnnlib.output_constraint_clauses {
        // The peel rewrites each ATOM in place and preserves the boolean
        // structure (clause count, order, and `is_disjunction`), so the
        // parallel `per_clause_input_bounds` stay aligned.
        if clause.is_empty() {
            new_clauses.push(Vec::new());
            continue;
        }
        match rewrite_constraints_for_layer(layer_type.clone(), clause) {
            Ok(Some(constraints)) => new_clauses.push(constraints),
            Ok(None) => {
                return PeelOffReport::skipped("clause constraints incompatible with peeling");
            }
            Err(reason) => {
                return PeelOffReport::skipped(format!(
                    "clause constraints unsupported: {}",
                    reason
                ));
            }
        }
    }

    let new_output_name = layer_inputs[0].clone();
    debug!(
        "Peeling off {:?} layer {} -> output {}",
        layer_type, output_name, new_output_name
    );

    vnnlib.output_constraints = new_flat;
    vnnlib.output_constraint_clauses = new_clauses;
    model.network.outputs[0].name = new_output_name;
    model.network.layers.remove(layer_idx);

    PeelOffReport::peeled(layer_type)
}

fn rewrite_constraints_for_layer(
    layer_type: LayerType,
    constraints: &[OutputConstraint],
) -> Result<Option<Vec<OutputConstraint>>, String> {
    if constraints.is_empty() {
        return Ok(None);
    }

    match layer_type {
        LayerType::Softmax | LayerType::LogSoftmax | LayerType::Sigmoid => {
            Ok(rewrite_relational_only(constraints))
        }
        _ => Ok(None),
    }
}

fn rewrite_relational_only(constraints: &[OutputConstraint]) -> Option<Vec<OutputConstraint>> {
    if constraints.iter().all(is_relational_only) {
        Some(constraints.to_vec())
    } else {
        None
    }
}

fn is_relational_only(constraint: &OutputConstraint) -> bool {
    matches!(
        constraint,
        OutputConstraint::LessEq(_, _)
            | OutputConstraint::GreaterEq(_, _)
            | OutputConstraint::LessThan(_, _)
            | OutputConstraint::GreaterThan(_, _)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vnnlib::VnnLibSpec;
    use crate::{DataType, LayerSpec, Network, OnnxModel, TensorSpec, WeightStore};
    use ny_core::LayerType as LT;
    use std::collections::HashMap;

    fn peel_certified_traffic_fixture(
        model: &mut OnnxModel,
        spec: &mut VnnLibSpec,
    ) -> PeelOffReport {
        peel_off_terminal_softmax_with_certificate(model, spec, TRAFFIC_30_LOGIT_LATTICE)
    }

    fn layer(name: &str, layer_type: LT, inputs: &[&str], outputs: &[&str]) -> LayerSpec {
        LayerSpec {
            name: name.to_string(),
            layer_type,
            inputs: inputs.iter().map(|s| (*s).to_string()).collect(),
            outputs: outputs.iter().map(|s| (*s).to_string()).collect(),
            weights: None,
            attributes: HashMap::new(),
        }
    }

    /// 1-d fixture: <something> -> Sigmoid -> y.
    fn sigmoid_terminal_model() -> OnnxModel {
        let network = Network {
            name: "sigmoid_fixture".to_string(),
            inputs: vec![TensorSpec {
                name: "x".to_string(),
                shape: vec![1],
                dtype: DataType::Float32,
            }],
            outputs: vec![TensorSpec {
                name: "y".to_string(),
                shape: vec![1],
                dtype: DataType::Float32,
            }],
            layers: vec![
                layer("relu", LT::ReLU, &["x"], &["z"]),
                layer("sig", LT::Sigmoid, &["z"], &["y"]),
            ],
            param_count: 0,
        };
        OnnxModel::empty_with_network(network, WeightStore::new())
    }

    fn softmax_terminal_model(layer_type: LT, shape: Vec<i64>, axis: Option<i64>) -> OnnxModel {
        let mut terminal = layer("terminal", layer_type, &["z"], &["y"]);
        if let Some(axis) = axis {
            terminal
                .attributes
                .insert("axis".to_string(), crate::AttributeValue::Int(axis));
        }
        let network = Network {
            name: "softmax_fixture".to_string(),
            inputs: vec![TensorSpec {
                name: "z".to_string(),
                shape: shape.clone(),
                dtype: DataType::Float32,
            }],
            outputs: vec![TensorSpec {
                name: "y".to_string(),
                shape,
                dtype: DataType::Float32,
            }],
            layers: vec![terminal],
            param_count: 0,
        };
        let inferred_shape = network.outputs[0].shape.clone();
        OnnxModel::empty_with_network(network, WeightStore::new()).with_tensor_shapes(
            HashMap::from([
                ("z".to_string(), inferred_shape.clone()),
                ("y".to_string(), inferred_shape),
            ]),
        )
    }

    fn relational_spec(num_outputs: usize) -> VnnLibSpec {
        let mut spec = VnnLibSpec::new();
        spec.num_inputs = 1;
        spec.num_outputs = num_outputs;
        spec.input_bounds = vec![(0.0, 0.0)];
        spec.output_constraints = vec![OutputConstraint::GreaterEq(0, num_outputs - 1)];
        spec
    }

    /// The 42 atoms of the real traffic property: `Y_i >= Y_t` for every
    /// `i != t`, i.e. the complement of "class `t` is the strict argmax".
    fn argmax_complement_atoms(num_outputs: usize, true_label: usize) -> Vec<OutputConstraint> {
        (0..num_outputs)
            .filter(|index| *index != true_label)
            .map(|index| OutputConstraint::GreaterEq(index, true_label))
            .collect()
    }

    /// Exactly the shape `parse_vnnlib` produces for the staged traffic corpus
    /// (verified against `model_30_idx_1703_eps_1.00000.vnnlib`: 43 outputs,
    /// `is_disjunction`, 42 singleton clauses, flat list == clause concatenation,
    /// no per-clause input bounds, no dual network).
    fn argmax_complement_spec(num_outputs: usize, true_label: usize) -> VnnLibSpec {
        let atoms = argmax_complement_atoms(num_outputs, true_label);
        let mut spec = VnnLibSpec::new();
        spec.num_inputs = 1;
        spec.num_outputs = num_outputs;
        spec.input_bounds = vec![(0.0, 1.0)];
        spec.output_constraint_clauses = atoms.iter().cloned().map(|atom| vec![atom]).collect();
        spec.output_constraints = atoms;
        spec.is_disjunction = true;
        spec
    }

    /// A minimal two-network relational spec. Cross-network atoms compare
    /// outputs of two DIFFERENT invocations, hence two different softmax
    /// denominators (e.g. `(assert (< Y_f[0,3] Y_g[0,3]))` in
    /// `monotonic_acasxu_2026`).
    fn dual_network_stub() -> crate::vnnlib::DualNetworkSpec {
        crate::vnnlib::DualNetworkSpec {
            networks: Vec::new(),
            property: crate::vnnlib::DualNetworkProperty::MonotonicGreaterEq {
                output: 3,
                varying_input: 0,
                strict_unsafe: true,
            },
            shared_input_coupling: true,
            f_input_bounds: vec![(0.0, 1.0)],
            g_input_bounds: vec![(0.0, 1.0)],
            validation: crate::vnnlib::DualNetworkValidation {
                input_equalities: Vec::new(),
                f_input_ge_g_input: Vec::new(),
                g_input_ge_f_input: Vec::new(),
                isomorphic_output_safe_complement: false,
                monotonic_output_relation_count: 1,
                unsupported_output_relation: false,
                isomorphic_output_atoms: Vec::new(),
                isomorphic_output_is_conjunction: false,
            },
            formula_dnf: None,
        }
    }

    fn band_clause_spec(low: f64, high: f64) -> VnnLibSpec {
        let mut spec = VnnLibSpec::new();
        spec.num_inputs = 1;
        spec.num_outputs = 1;
        spec.input_bounds = vec![(-1.0, 1.0)];
        // cgan band shape: unsafe = (Y >= high) OR (Y <= low); flat list EMPTY.
        spec.output_constraint_clauses = vec![
            vec![OutputConstraint::GreaterEqConst(0, high)],
            vec![OutputConstraint::LessEqConst(0, low)],
        ];
        spec.is_disjunction = true;
        spec
    }

    /// Run `f` with `NY_SIGMOID_PEEL` set (`Some`) or removed (`None`),
    /// serialized + restored via the blessed env choke point (clippy env wall).
    fn with_sigmoid_peel_env<T>(value: Option<&str>, f: impl FnOnce() -> T) -> T {
        match value {
            Some(v) => ny_test_utils::env::with_serialized_env_vars(&[("NY_SIGMOID_PEEL", v)], f),
            None => ny_test_utils::env::with_serialized_env_vars_removed(&["NY_SIGMOID_PEEL"], f),
        }
    }

    /// The known-bad constant-threshold lane stays quarantined even when its
    /// historical environment gate is explicitly armed.
    #[test]
    fn constant_threshold_sigmoid_peel_is_quarantined_without_mutation() {
        with_sigmoid_peel_env(Some("1"), || {
            let mut model = sigmoid_terminal_model();
            let (low, high) = (0.1996644288301468, 0.23966443538665771);
            let mut spec = band_clause_spec(low, high);

            let report = peel_off_terminal_sigmoid_auto(&mut model, &mut spec);
            assert!(!report.peeled);
            assert_eq!(
                report.reason.as_deref(),
                Some(
                    "constant-threshold terminal-Sigmoid peel is quarantined pending region-equivalence proof"
                )
            );
            assert_eq!(model.network.layers.len(), 2);
            assert_eq!(model.network.outputs[0].name, "y");
            assert!(spec.output_constraints.is_empty());
            assert_eq!(
                spec.output_constraint_clauses,
                vec![
                    vec![OutputConstraint::GreaterEqConst(0, high)],
                    vec![OutputConstraint::LessEqConst(0, low)],
                ]
            );
        });
    }

    /// Auto-peel gates: non-Sigmoid terminal, relational constraint present,
    /// out-of-range threshold, and a disabled environment gate all skip.
    #[test]
    fn auto_peel_gates_fail_closed() {
        with_sigmoid_peel_env(Some("1"), || {
            // Non-Sigmoid terminal.
            let mut model = sigmoid_terminal_model();
            model.network.outputs[0].name = "z".to_string(); // ReLU is now terminal
            let mut spec = band_clause_spec(0.2, 0.8);
            assert!(!peel_off_terminal_sigmoid_auto(&mut model, &mut spec).peeled);

            // Relational constraint present -> not all-constant -> skip.
            let mut model = sigmoid_terminal_model();
            let mut spec = band_clause_spec(0.2, 0.8);
            spec.output_constraint_clauses[0].push(OutputConstraint::LessEq(0, 0));
            assert!(!peel_off_terminal_sigmoid_auto(&mut model, &mut spec).peeled);
            assert_eq!(model.network.layers.len(), 2, "model must be untouched");

            // Constant threshold outside (0, 1): the quarantined lane still
            // skips without inspecting/rewriting its value.
            let mut model = sigmoid_terminal_model();
            let mut spec = band_clause_spec(0.2, 1.5);
            assert!(!peel_off_terminal_sigmoid_auto(&mut model, &mut spec).peeled);
            assert_eq!(model.network.layers.len(), 2);
            match spec.output_constraint_clauses[0][0] {
                OutputConstraint::GreaterEqConst(0, t) => assert_eq!(t, 1.5),
                ref other => panic!("unexpected constraint {other:?}"),
            }

            // Empty spec -> skip.
            let mut model = sigmoid_terminal_model();
            let mut spec = VnnLibSpec::new();
            assert!(!peel_off_terminal_sigmoid_auto(&mut model, &mut spec).peeled);
        });

        // Disabled gate.
        with_sigmoid_peel_env(Some("0"), || {
            let mut model = sigmoid_terminal_model();
            let mut spec = band_clause_spec(0.2, 0.8);
            let report = peel_off_terminal_sigmoid_auto(&mut model, &mut spec);
            assert!(!report.peeled);
            assert_eq!(
                report.reason.as_deref(),
                Some("NY_SIGMOID_PEEL not enabled (opt-in)")
            );
            assert_eq!(model.network.layers.len(), 2);
        });
    }

    #[test]
    fn rewrite_softmax_rejects_constants() {
        let constraints = vec![OutputConstraint::LessEqConst(0, 0.5)];
        let rewritten = rewrite_constraints_for_layer(LayerType::Softmax, &constraints)
            .expect("rewrite should not error");
        assert!(rewritten.is_none());
    }

    #[test]
    fn rewrite_softmax_accepts_relational() {
        let constraints = vec![OutputConstraint::LessEq(0, 1)];
        let rewritten = rewrite_constraints_for_layer(LayerType::Softmax, &constraints)
            .expect("rewrite should not error");
        assert_eq!(rewritten, Some(constraints));
    }

    #[test]
    fn traffic_softmax_peel_accepts_exact_single_group() {
        let mut model = softmax_terminal_model(LT::Softmax, vec![1, 43], Some(-1));
        let mut spec = argmax_complement_spec(43, 25);

        let report = peel_certified_traffic_fixture(&mut model, &mut spec);

        assert!(
            report.peeled,
            "single-group Softmax declined: {:?}",
            report.reason
        );
        assert_eq!(report.layer_type, Some(LT::Softmax));
        assert!(model.network.layers.is_empty());
        assert_eq!(model.network.outputs[0].name, "z");
        assert_eq!(spec.output_constraints, argmax_complement_atoms(43, 25));
    }

    #[test]
    fn traffic_softmax_peel_rejects_unsupported_near_tie_model_without_mutation() {
        // A real-arithmetic strict ordering can collapse in f32 before the
        // division: exp(-5e-8) rounds to exp(0) here. A synthetic/unknown
        // model therefore cannot borrow the certified traffic model's gap.
        let near_tie_gap = 1.0e-8_f32;
        assert_eq!(1.0_f32 - near_tie_gap, 1.0_f32);

        let mut model = softmax_terminal_model(LT::Softmax, vec![1, 43], Some(-1));
        let mut spec = argmax_complement_spec(43, 25);
        let before = fingerprint(&model, &spec);
        let report = peel_off_terminal_softmax_single_group_unbound(
            &mut model,
            &mut spec,
            b"unsupported synthetic model with near-tied logits",
        );

        assert!(!report.peeled);
        assert!(report
            .reason
            .as_deref()
            .is_some_and(|reason| reason.contains("no certified logit-lattice rule")));
        assert_eq!(before, fingerprint(&model, &spec));
    }

    #[test]
    fn official_certificate_rejects_an_independently_constructed_model() {
        let model = softmax_terminal_model(LT::Softmax, vec![1, 43], Some(-1));
        let error = validate_certified_lattice_structure(&model, TRAFFIC_30_LOGIT_LATTICE)
            .expect_err("official-byte certificate must not authorize a mismatched model");
        assert!(error.contains("logit producer") || error.contains("MatMul"));
    }

    /// REGRESSION for a MEASURED false UNSAT.
    ///
    /// `(assert (and (>= Y[0,0] Y[0,42])))` passes every model-side guard, and
    /// the previous spec-side guard (bare pairwise, no constants) admitted it.
    /// On the real 3_30_30 traffic model over 301 points of the real
    /// `model_30_idx_1703_eps_1.00000` box, ORT 1.19.2 gives
    /// `p_0 == p_42 == 0.0f` at 301/301 points (SAT) while `z_0 >= z_42` holds at
    /// 0/301 points (UNSAT). Peeling here manufactures an unsound `unsat`.
    #[test]
    fn traffic_softmax_peel_rejects_non_argmax_pair_without_mutation() {
        let mut model = softmax_terminal_model(LT::Softmax, vec![1, 43], Some(-1));
        let mut spec = relational_spec(43);

        let report = peel_certified_traffic_fixture(&mut model, &mut spec);

        assert!(
            !report.peeled,
            "peeled a non-argmax pair: this is the measured false-UNSAT shape"
        );
        assert_eq!(model.network.layers.len(), 1, "model must be untouched");
        assert_eq!(model.network.outputs[0].name, "y");
        assert_eq!(
            spec.output_constraints,
            vec![OutputConstraint::GreaterEq(0, 42)],
            "spec must be untouched"
        );
    }

    #[test]
    fn traffic_softmax_peel_accepts_loader_pinned_symbolic_batch() {
        let mut model = softmax_terminal_model(LT::Softmax, vec![-1, 43], Some(-1))
            .with_tensor_shapes(HashMap::from([
                ("z".to_string(), vec![1, 43]),
                ("y".to_string(), vec![1, 43]),
            ]));
        let mut spec = argmax_complement_spec(43, 25);

        let report = peel_certified_traffic_fixture(&mut model, &mut spec);

        assert!(
            report.peeled,
            "batch-1 inferred traffic shape declined: {:?}",
            report.reason
        );
        assert_eq!(report.layer_type, Some(LT::Softmax));
    }

    #[test]
    fn traffic_softmax_peel_seals_exact_benchmark_layout() {
        let mut wrong_shape = softmax_terminal_model(LT::Softmax, vec![1, 42], Some(-1));
        let mut shape_spec = argmax_complement_spec(42, 25);
        let shape_report = peel_certified_traffic_fixture(&mut wrong_shape, &mut shape_spec);
        assert!(!shape_report.peeled);
        assert_eq!(
            shape_report.reason.as_deref(),
            Some("traffic terminal Softmax shape is not exactly [1, 43]")
        );

        let mut wrong_spec = softmax_terminal_model(LT::Softmax, vec![1, 43], Some(-1));
        let mut count_spec = argmax_complement_spec(42, 25);
        let count_report = peel_certified_traffic_fixture(&mut wrong_spec, &mut count_spec);
        assert!(!count_report.peeled);
        assert_eq!(
            count_report.reason.as_deref(),
            Some("traffic VNN-LIB output count is not exactly 43")
        );
    }

    #[test]
    fn traffic_softmax_peel_rejects_other_terminal_activations_without_mutation() {
        for layer_type in [LT::LogSoftmax, LT::Sigmoid] {
            let mut model = softmax_terminal_model(layer_type.clone(), vec![1, 4], Some(-1));
            let mut spec = relational_spec(4);

            let report = peel_certified_traffic_fixture(&mut model, &mut spec);

            assert!(!report.peeled, "unexpectedly peeled {layer_type:?}");
            assert_eq!(model.network.layers.len(), 1);
            assert_eq!(model.network.outputs[0].name, "y");
            assert_eq!(
                spec.output_constraints,
                vec![OutputConstraint::GreaterEq(0, 3)]
            );
        }
    }

    #[test]
    fn traffic_softmax_peel_rejects_cross_group_semantics_without_mutation() {
        // Axis 1 on [2, 2] creates two independent groups.  For logits
        // [0, -10, 1, 10], softmax(z)[0] > softmax(z)[2] even though z0 < z2,
        // so copying a relational atom across these groups would be unsound.
        let mut model = softmax_terminal_model(LT::Softmax, vec![2, 2], Some(1));
        let mut spec = relational_spec(4);
        spec.output_constraints = vec![OutputConstraint::GreaterEq(0, 2)];

        let report = peel_certified_traffic_fixture(&mut model, &mut spec);

        assert!(!report.peeled);
        assert_eq!(
            report.reason.as_deref(),
            Some("terminal Softmax has more than one normalization group")
        );
        assert_eq!(model.network.layers.len(), 1);
        assert_eq!(model.network.outputs[0].name, "y");
        assert_eq!(
            spec.output_constraints,
            vec![OutputConstraint::GreaterEq(0, 2)]
        );
    }

    #[test]
    fn traffic_softmax_peel_rejects_conflicting_output_annotation() {
        // A stale authored graph-output annotation must not turn a real
        // two-batch Softmax into a purported single normalization group.
        let mut model =
            softmax_terminal_model(LT::Softmax, vec![1, 2], Some(1)).with_tensor_shapes(
                HashMap::from([("z".to_string(), vec![2, 2]), ("y".to_string(), vec![1, 2])]),
            );
        let mut spec = relational_spec(2);

        let report = peel_certified_traffic_fixture(&mut model, &mut spec);

        assert!(!report.peeled);
        assert_eq!(
            report.reason.as_deref(),
            Some("terminal Softmax inferred input/output shapes disagree")
        );
        assert_eq!(model.network.layers.len(), 1);
        assert_eq!(model.network.outputs[0].name, "y");
        assert_eq!(
            spec.output_constraints,
            vec![OutputConstraint::GreaterEq(0, 1)]
        );
    }

    #[test]
    fn traffic_softmax_peel_requires_authenticated_axis_and_concrete_shape() {
        for (shape, axis) in [(vec![1, 4], None), (vec![-1, 4], Some(-1))] {
            let mut model = softmax_terminal_model(LT::Softmax, shape, axis);
            let mut spec = relational_spec(4);

            let report = peel_certified_traffic_fixture(&mut model, &mut spec);

            assert!(!report.peeled);
            assert_eq!(model.network.layers.len(), 1);
            assert_eq!(model.network.outputs[0].name, "y");
        }
    }

    #[test]
    fn legacy_softmax_family_peel_requires_single_group() {
        for layer_type in [LT::Softmax, LT::LogSoftmax] {
            let mut model = softmax_terminal_model(layer_type.clone(), vec![2, 2], Some(1));
            let mut spec = relational_spec(4);
            spec.output_constraints = vec![OutputConstraint::GreaterEq(0, 2)];

            let report = peel_off_last_softmax_layer(&mut model, &mut spec);

            assert!(!report.peeled, "cross-group {layer_type:?} was peeled");
            assert_eq!(model.network.layers.len(), 1);
            assert_eq!(model.network.outputs[0].name, "y");
            assert_eq!(
                spec.output_constraints,
                vec![OutputConstraint::GreaterEq(0, 2)]
            );
        }
    }

    #[test]
    fn legacy_logsoftmax_peel_refuses_without_model_byte_certificate() {
        let mut model = softmax_terminal_model(LT::LogSoftmax, vec![1, 4], Some(-1));
        let mut spec = relational_spec(4);

        let report = peel_off_last_softmax_layer(&mut model, &mut spec);

        assert!(!report.peeled);
        assert_eq!(
            report.reason.as_deref(),
            Some(
                "terminal Softmax-family peel requires exact model bytes and a certified logit-lattice rule"
            )
        );
        assert_eq!(model.network.layers.len(), 1);
        assert_eq!(model.network.outputs[0].name, "y");
    }

    #[test]
    fn legacy_sigmoid_peel_keeps_only_exact_relational_comparisons() {
        let mut relational_model = softmax_terminal_model(LT::Sigmoid, vec![1, 2], None);
        let mut relational_spec = relational_spec(2);
        let relational_report =
            peel_off_last_softmax_layer(&mut relational_model, &mut relational_spec);
        assert!(relational_report.peeled);
        assert_eq!(relational_report.layer_type, Some(LT::Sigmoid));
        assert_eq!(
            relational_spec.output_constraints,
            vec![OutputConstraint::GreaterEq(0, 1)]
        );

        let mut constant_model = sigmoid_terminal_model();
        let mut constant_spec = band_clause_spec(0.2, 0.8);
        let constant_report = peel_off_last_softmax_layer(&mut constant_model, &mut constant_spec);
        assert!(!constant_report.peeled);
        assert_eq!(constant_model.network.layers.len(), 2);
        assert_eq!(constant_model.network.outputs[0].name, "y");
        assert_eq!(
            constant_spec.output_constraint_clauses,
            band_clause_spec(0.2, 0.8).output_constraint_clauses
        );
    }

    #[test]
    fn rewrite_sigmoid_rejects_constants() {
        let constraints = vec![OutputConstraint::LessEqConst(0, 0.25)];
        let rewritten = rewrite_constraints_for_layer(LayerType::Sigmoid, &constraints)
            .expect("constant quarantine is a clean decline");
        assert!(rewritten.is_none());
    }

    #[test]
    fn rewrite_sigmoid_accepts_relational() {
        let constraints = vec![OutputConstraint::GreaterThan(0, 1)];
        let rewritten = rewrite_constraints_for_layer(LayerType::Sigmoid, &constraints)
            .expect("relational rewrite should not error");
        assert_eq!(rewritten, Some(constraints));
    }

    // ---------------------------------------------------------------------
    // NY_STRIP_TERMINAL_SOFTMAX — dark-gated argmax-complement Softmax strip
    // ---------------------------------------------------------------------

    /// Run `f` with `NY_STRIP_TERMINAL_SOFTMAX` set (`Some`) or removed (`None`),
    /// serialized + restored via the blessed env choke point (clippy env wall).
    fn with_strip_env<T>(value: Option<&str>, f: impl FnOnce() -> T) -> T {
        match value {
            Some(v) => {
                ny_test_utils::env::with_serialized_env_vars(&[(STRIP_TERMINAL_SOFTMAX_ENV, v)], f)
            }
            None => ny_test_utils::env::with_serialized_env_vars_removed(
                &[STRIP_TERMINAL_SOFTMAX_ENV],
                f,
            ),
        }
    }

    /// A snapshot of everything the strip is allowed to touch, for byte-identity
    /// assertions on the dark path.
    fn fingerprint(model: &OnnxModel, spec: &VnnLibSpec) -> String {
        format!(
            "{:?}|{:?}|{:?}|{:?}|{:?}|{}|{:?}|{:?}",
            model.network.outputs,
            model
                .network
                .layers
                .iter()
                .map(|l| (&l.name, &l.layer_type, &l.inputs, &l.outputs))
                .collect::<Vec<_>>(),
            model.network.inputs,
            spec.output_constraints,
            spec.output_constraint_clauses,
            spec.is_disjunction,
            spec.input_bounds,
            spec.num_outputs,
        )
    }

    /// (a) GATE UNSET => the transform never runs and the model/spec are
    /// byte-identical afterwards, even for an input it would otherwise strip.
    #[test]
    fn strip_terminal_softmax_is_dark_by_default() {
        with_strip_env(None, || {
            let mut model = softmax_terminal_model(LT::Softmax, vec![1, 43], Some(-1));
            let mut spec = argmax_complement_spec(43, 25);
            let before = fingerprint(&model, &spec);

            let report = strip_terminal_softmax(&mut model, &mut spec);

            assert!(!report.peeled, "dark path must never strip");
            assert_eq!(
                report.reason.as_deref(),
                Some("NY_STRIP_TERMINAL_SOFTMAX is not exactly \"1\" (dark by default)")
            );
            assert_eq!(
                before,
                fingerprint(&model, &spec),
                "dark path must be byte-identical"
            );
        });
    }

    /// (a) The gate is armed by the EXACT byte string "1" and nothing else.
    /// Every near-miss leaves the model and spec byte-identical.
    #[test]
    fn strip_terminal_softmax_gate_parses_exactly_one() {
        for value in [
            "", "0", "01", "1 ", " 1", "true", "TRUE", "yes", "on", "2", "1\n",
        ] {
            with_strip_env(Some(value), || {
                let mut model = softmax_terminal_model(LT::Softmax, vec![1, 43], Some(-1));
                let mut spec = argmax_complement_spec(43, 25);
                let before = fingerprint(&model, &spec);

                let report = strip_terminal_softmax(&mut model, &mut spec);

                assert!(!report.peeled, "gate armed by {value:?}");
                assert_eq!(
                    before,
                    fingerprint(&model, &spec),
                    "mutation under gate value {value:?}"
                );
            });
        }

        // Non-UTF-8 must not be lossily coerced into the arming value.
        #[cfg(unix)]
        {
            use std::ffi::OsStr;
            use std::os::unix::ffi::OsStrExt;

            let raw = OsStr::from_bytes(b"1\xff");
            ny_test_utils::env::with_serialized_env_vars_os(
                &[(STRIP_TERMINAL_SOFTMAX_ENV, raw)],
                || {
                    let mut model = softmax_terminal_model(LT::Softmax, vec![1, 43], Some(-1));
                    let mut spec = argmax_complement_spec(43, 25);
                    let before = fingerprint(&model, &spec);
                    assert!(!strip_terminal_softmax(&mut model, &mut spec).peeled);
                    assert_eq!(before, fingerprint(&model, &spec));
                },
            );
        }
    }

    /// (c) Even an otherwise-qualified armed legacy request lacks the exact
    /// model bytes needed for lattice authentication and must fail closed.
    #[test]
    fn strip_terminal_softmax_armed_without_model_bytes_still_refuses() {
        with_strip_env(Some("1"), || {
            let mut model = softmax_terminal_model(LT::Softmax, vec![1, 43], Some(-1));
            let mut spec = argmax_complement_spec(43, 25);
            let before = fingerprint(&model, &spec);

            let report = strip_terminal_softmax(&mut model, &mut spec);

            assert!(!report.peeled);
            assert_eq!(
                report.reason.as_deref(),
                Some(
                    "terminal Softmax strip requires exact model bytes and a certified logit-lattice rule"
                )
            );
            assert_eq!(before, fingerprint(&model, &spec));
        });

        // Ordinary well-separated values still illustrate the real-arithmetic
        // identity. This is not used as proof authority for arbitrary f32
        // models; only the hash-bound lattice certificate can exclude ties.
        let logits = [3.5f64, -2.0, 0.0, 9.25, -7.5, 1.0, 9.24];
        let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let denominator: f64 = logits.iter().map(|z| (z - max).exp()).sum();
        let probabilities: Vec<f64> = logits
            .iter()
            .map(|z| (z - max).exp() / denominator)
            .collect();
        for i in 0..logits.len() {
            for j in 0..logits.len() {
                assert_eq!(
                    logits[i] >= logits[j],
                    probabilities[i] >= probabilities[j],
                    "order not preserved for ({i}, {j})"
                );
            }
        }
        let true_label = 3;
        assert!(
            (0..logits.len())
                .filter(|i| *i != true_label)
                .all(|i| logits[i] < logits[true_label]),
            "fixture must make the true label the strict argmax"
        );
        assert!(
            (0..probabilities.len())
                .filter(|i| *i != true_label)
                .all(|i| probabilities[i] < probabilities[true_label]),
            "argmax-complement verdict must agree pre- and post-softmax"
        );
    }

    /// (b) REJECT: comparison against a CONSTANT. Softmax is order-preserving,
    /// not value-preserving: for z = [0.6, 10], `z_0 >= 0.5` is true while
    /// `softmax(z)_0 ~= 8.3e-5 >= 0.5` is false.
    #[test]
    fn strip_terminal_softmax_rejects_constant_comparison() {
        with_strip_env(Some("1"), || {
            let mut model = softmax_terminal_model(LT::Softmax, vec![1, 43], Some(-1));
            let mut spec = argmax_complement_spec(43, 25);
            spec.output_constraint_clauses[0] = vec![OutputConstraint::GreaterEqConst(0, 0.5)];
            spec.output_constraints[0] = OutputConstraint::GreaterEqConst(0, 0.5);
            let before = fingerprint(&model, &spec);

            let report = strip_terminal_softmax(&mut model, &mut spec);

            assert!(!report.peeled);
            assert_eq!(
                report.reason.as_deref(),
                Some("argmax-complement atoms must be non-strict output-vs-output GreaterEq")
            );
            assert_eq!(before, fingerprint(&model, &spec));
        });
    }

    /// (b) REJECT: the MEASURED false-UNSAT shape — a bare pairwise comparison
    /// between two non-argmax classes. `p_0 == p_42 == 0.0f` at 301/301 sampled
    /// points of the real box (SAT) while `z_0 >= z_42` holds at 0/301 (UNSAT).
    #[test]
    fn strip_terminal_softmax_rejects_non_argmax_pairwise_comparison() {
        with_strip_env(Some("1"), || {
            let mut model = softmax_terminal_model(LT::Softmax, vec![1, 43], Some(-1));
            let mut spec = relational_spec(43);
            let before = fingerprint(&model, &spec);

            let report = strip_terminal_softmax(&mut model, &mut spec);

            assert!(
                !report.peeled,
                "stripped the measured false-UNSAT shape (>= Y[0,0] Y[0,42])"
            );
            assert_eq!(before, fingerprint(&model, &spec));
        });
    }

    /// (b) REJECT: a spec that is nearly the argmax complement but omits one
    /// challenger class, so it is NOT `not-argmax(t)` and the float saturation
    /// argument does not close.
    #[test]
    fn strip_terminal_softmax_rejects_incomplete_argmax_complement() {
        with_strip_env(Some("1"), || {
            let mut model = softmax_terminal_model(LT::Softmax, vec![1, 43], Some(-1));
            let mut spec = argmax_complement_spec(43, 25);
            // Drop class 7's clause and duplicate class 0 to keep the count.
            spec.output_constraint_clauses[7] = vec![OutputConstraint::GreaterEq(0, 25)];
            spec.output_constraints = spec
                .output_constraint_clauses
                .iter()
                .flatten()
                .cloned()
                .collect();
            let before = fingerprint(&model, &spec);

            let report = strip_terminal_softmax(&mut model, &mut spec);

            assert!(!report.peeled);
            assert_eq!(
                report.reason.as_deref(),
                Some("argmax-complement repeats a challenger class")
            );
            assert_eq!(before, fingerprint(&model, &spec));
        });
    }

    /// (b) REJECT: atoms that do not share ONE right-hand true label — i.e. not
    /// an argmax complement, and in general mixing saturated classes.
    #[test]
    fn strip_terminal_softmax_rejects_mixed_right_hand_labels() {
        with_strip_env(Some("1"), || {
            let mut model = softmax_terminal_model(LT::Softmax, vec![1, 43], Some(-1));
            let mut spec = argmax_complement_spec(43, 25);
            spec.output_constraint_clauses[0] = vec![OutputConstraint::GreaterEq(0, 24)];
            spec.output_constraints[0] = OutputConstraint::GreaterEq(0, 24);
            let before = fingerprint(&model, &spec);

            let report = strip_terminal_softmax(&mut model, &mut spec);

            assert!(!report.peeled);
            assert_eq!(
                report.reason.as_deref(),
                Some("argmax-complement atoms do not share one right-hand true label")
            );
            assert_eq!(before, fingerprint(&model, &spec));
        });
    }

    /// (b) REJECT: STRICT atoms. Under a float tie a strict atom is true on
    /// logits and false on probabilities — the opposite (false-SAT) error.
    #[test]
    fn strip_terminal_softmax_rejects_strict_atoms() {
        with_strip_env(Some("1"), || {
            let mut model = softmax_terminal_model(LT::Softmax, vec![1, 43], Some(-1));
            let mut spec = argmax_complement_spec(43, 25);
            spec.output_constraint_clauses[0] = vec![OutputConstraint::GreaterThan(0, 25)];
            spec.output_constraints[0] = OutputConstraint::GreaterThan(0, 25);
            let before = fingerprint(&model, &spec);

            let report = strip_terminal_softmax(&mut model, &mut spec);

            assert!(!report.peeled);
            assert_eq!(
                report.reason.as_deref(),
                Some("argmax-complement atoms must be non-strict output-vs-output GreaterEq")
            );
            assert_eq!(before, fingerprint(&model, &spec));
        });
    }

    /// (b) REJECT: a CONJUNCTION of the same atoms. `is_disjunction == false`
    /// is a different property that is not the argmax complement.
    #[test]
    fn strip_terminal_softmax_rejects_conjunctive_property() {
        with_strip_env(Some("1"), || {
            let mut model = softmax_terminal_model(LT::Softmax, vec![1, 43], Some(-1));
            let mut spec = argmax_complement_spec(43, 25);
            spec.is_disjunction = false;
            let before = fingerprint(&model, &spec);

            let report = strip_terminal_softmax(&mut model, &mut spec);

            assert!(!report.peeled);
            assert_eq!(
                report.reason.as_deref(),
                Some("output constraints are not a top-level disjunction")
            );
            assert_eq!(before, fingerprint(&model, &spec));
        });
    }

    /// (b) REJECT: MULTI-SOFTMAX / cross-denominator. Two normalization groups
    /// mean two different denominators, and order is not preserved across them.
    #[test]
    fn strip_terminal_softmax_rejects_multiple_normalization_groups() {
        with_strip_env(Some("1"), || {
            // [2, 2] with axis 1 is two groups of two.
            let mut model = softmax_terminal_model(LT::Softmax, vec![2, 2], Some(1));
            let mut spec = argmax_complement_spec(4, 3);
            let before = fingerprint(&model, &spec);

            let report = strip_terminal_softmax(&mut model, &mut spec);

            assert!(!report.peeled);
            assert_eq!(
                report.reason.as_deref(),
                Some("terminal Softmax has more than one normalization group")
            );
            assert_eq!(before, fingerprint(&model, &spec));
        });

        // The concrete counterexample the group check exists to block: with two
        // denominators, equal logits give unequal probabilities.
        let group_a = [10.0f64, 0.0];
        let group_b = [10.0f64, 10.0];
        let softmax = |row: [f64; 2]| {
            let m = row[0].max(row[1]);
            let s: f64 = row.iter().map(|z| (z - m).exp()).sum();
            [(row[0] - m).exp() / s, (row[1] - m).exp() / s]
        };
        let (pa, pb) = (softmax(group_a), softmax(group_b));
        assert!(group_a[0] <= group_b[0], "logits are equal, so `<=` holds");
        assert!(
            pa[0] > pb[0],
            "probabilities violate the same atom across denominators"
        );
    }

    /// (b) REJECT: a Softmax that is NOT terminal — another layer consumes its
    /// output, and that layer computes on probabilities.
    #[test]
    fn strip_terminal_softmax_rejects_non_terminal_softmax() {
        with_strip_env(Some("1"), || {
            let mut model = softmax_terminal_model(LT::Softmax, vec![1, 43], Some(-1));
            // A second consumer of the softmax output "y".
            model
                .network
                .layers
                .push(layer("downstream", LT::ReLU, &["y"], &["w"]));
            let mut spec = argmax_complement_spec(43, 25);
            let before = fingerprint(&model, &spec);

            let report = strip_terminal_softmax(&mut model, &mut spec);

            assert!(!report.peeled);
            assert_eq!(
                report.reason.as_deref(),
                Some("terminal Softmax output is consumed by other layers")
            );
            assert_eq!(before, fingerprint(&model, &spec));
        });
    }

    /// (b) REJECT: WRONG AXIS. `[1, 43]` normalized along axis 0 is 43 groups of
    /// one, so the compared indices do not share a denominator (each softmax
    /// output is identically 1.0).
    #[test]
    fn strip_terminal_softmax_rejects_wrong_axis() {
        with_strip_env(Some("1"), || {
            let mut model = softmax_terminal_model(LT::Softmax, vec![1, 43], Some(0));
            let mut spec = argmax_complement_spec(43, 25);
            let before = fingerprint(&model, &spec);

            let report = strip_terminal_softmax(&mut model, &mut spec);

            assert!(!report.peeled);
            assert_eq!(
                report.reason.as_deref(),
                Some("terminal Softmax has more than one normalization group")
            );
            assert_eq!(before, fingerprint(&model, &spec));
        });

        // And a missing axis attribute is never defaulted.
        with_strip_env(Some("1"), || {
            let mut model = softmax_terminal_model(LT::Softmax, vec![1, 43], None);
            let mut spec = argmax_complement_spec(43, 25);
            let before = fingerprint(&model, &spec);

            let report = strip_terminal_softmax(&mut model, &mut spec);

            assert!(!report.peeled);
            assert_eq!(
                report.reason.as_deref(),
                Some("terminal Softmax has no authenticated integer axis")
            );
            assert_eq!(before, fingerprint(&model, &spec));
        });
    }

    /// (b) REJECT: DUAL-NETWORK specs, explicitly. Two network invocations have
    /// two denominators. Previously blocked only indirectly, by the parser
    /// leaving both constraint lists empty.
    #[test]
    fn strip_terminal_softmax_rejects_dual_network_spec() {
        with_strip_env(Some("1"), || {
            let mut model = softmax_terminal_model(LT::Softmax, vec![1, 43], Some(-1));
            let mut spec = argmax_complement_spec(43, 25);
            spec.dual_network = Some(dual_network_stub());
            let before = fingerprint(&model, &spec);

            let report = strip_terminal_softmax(&mut model, &mut spec);

            assert!(!report.peeled);
            assert_eq!(
                report.reason.as_deref(),
                Some("dual-network spec: outputs may not share a softmax denominator")
            );
            assert_eq!(before, fingerprint(&model, &spec));
        });
    }

    /// (b) REJECT: a non-Softmax terminal activation is never stripped by this
    /// entry point, whatever the spec looks like.
    #[test]
    fn strip_terminal_softmax_rejects_other_terminal_activations() {
        with_strip_env(Some("1"), || {
            for layer_type in [LT::LogSoftmax, LT::Sigmoid, LT::ReLU] {
                let mut model = softmax_terminal_model(layer_type.clone(), vec![1, 43], Some(-1));
                let mut spec = argmax_complement_spec(43, 25);
                let before = fingerprint(&model, &spec);

                let report = strip_terminal_softmax(&mut model, &mut spec);

                assert!(!report.peeled, "stripped {layer_type:?}");
                assert_eq!(before, fingerprint(&model, &spec));
            }
        });
    }

    /// (b) REJECT: a flat constraint list that disagrees with the clauses would
    /// leave a second, unvalidated reading of the property behind.
    #[test]
    fn strip_terminal_softmax_rejects_divergent_flat_list() {
        with_strip_env(Some("1"), || {
            let mut model = softmax_terminal_model(LT::Softmax, vec![1, 43], Some(-1));
            let mut spec = argmax_complement_spec(43, 25);
            spec.output_constraints
                .push(OutputConstraint::GreaterEq(0, 1));
            let before = fingerprint(&model, &spec);

            let report = strip_terminal_softmax(&mut model, &mut spec);

            assert!(!report.peeled);
            assert_eq!(
                report.reason.as_deref(),
                Some("flat output constraints disagree with the disjunctive clauses")
            );
            assert_eq!(before, fingerprint(&model, &spec));
        });
    }

    /// (b) REJECT: per-clause input boxes make clauses non-interchangeable.
    #[test]
    fn strip_terminal_softmax_rejects_per_clause_input_bounds() {
        with_strip_env(Some("1"), || {
            let mut model = softmax_terminal_model(LT::Softmax, vec![1, 43], Some(-1));
            let mut spec = argmax_complement_spec(43, 25);
            let mut bounds = std::collections::BTreeMap::new();
            bounds.insert(0usize, (0.0, 0.5));
            spec.per_clause_input_bounds = vec![bounds];
            let before = fingerprint(&model, &spec);

            let report = strip_terminal_softmax(&mut model, &mut spec);

            assert!(!report.peeled);
            assert_eq!(
                report.reason.as_deref(),
                Some("per-clause input bounds are not supported by the strip")
            );
            assert_eq!(before, fingerprint(&model, &spec));
        });
    }

    /// (b) REJECT: a spec whose output count disagrees with the softmax width.
    #[test]
    fn strip_terminal_softmax_rejects_output_count_mismatch() {
        with_strip_env(Some("1"), || {
            let mut model = softmax_terminal_model(LT::Softmax, vec![1, 43], Some(-1));
            let mut spec = argmax_complement_spec(10, 3);
            let before = fingerprint(&model, &spec);

            let report = strip_terminal_softmax(&mut model, &mut spec);

            assert!(!report.peeled);
            assert_eq!(
                report.reason.as_deref(),
                Some("terminal Softmax shape [1, 43] is not exactly [1, 10]")
            );
            assert_eq!(before, fingerprint(&model, &spec));
        });
    }
}
