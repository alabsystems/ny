// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Joint ONNX + VNN-LIB optimization passes.

use crate::vnnlib::{OutputConstraint, VnnLibSpec};
use crate::OnnxModel;
use ny_core::LayerType;
use tracing::debug;

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

/// Default-ON auto-peel of a terminal Sigmoid for exactly-invertible specs
/// (#cgan-sigmoid-peel). Kill-switch: `NY_SIGMOID_PEEL=0`.
///
/// Gate: the network's terminal layer is a Sigmoid AND every output constraint
/// — flat and inside every disjunctive clause — is a CONSTANT-threshold
/// constraint. In that case the peel is an exact spec rewrite (see
/// `rewrite_sigmoid_constraints` for the monotonicity algebra and the outward
/// threshold rounding), so it is safe to apply without an explicit config
/// flag. Softmax/LogSoftmax peels (and relational sigmoid specs) remain
/// opt-in via `peel_off_last_softmax_layer`.
pub fn peel_off_terminal_sigmoid_auto(
    model: &mut OnnxModel,
    vnnlib: &mut VnnLibSpec,
) -> PeelOffReport {
    // OPT-IN pending root-cause (2026-07-17): with the peel ON, the cgan
    // upsample row's internal attack finds a witness that the trusted-ORT gate
    // REJECTS on the original graph (false counterexample) — the rewritten
    // clause region does not match the original on that row. The sat side is
    // ORT-gated, but a subset-direction defect would make the UNSAT side
    // unsound with no gate to catch it, so the peel stays off by default until
    // the region-equivalence bug is found and a differential region test lands.
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
    peel_off_last_softmax_layer(model, vnnlib)
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

/// Attempt to peel off a terminal Softmax/LogSoftmax/Sigmoid layer and rewrite VNN-LIB constraints.
///
/// This mirrors alpha-beta-CROWN's `peel_off_last_softmax_layer` behavior for classification specs.
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

    match layer_type {
        LayerType::Softmax | LayerType::LogSoftmax | LayerType::Sigmoid => {}
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
        LayerType::Softmax | LayerType::LogSoftmax => Ok(rewrite_relational_only(constraints)),
        LayerType::Sigmoid => rewrite_sigmoid_constraints(constraints).map(Some),
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

/// Rewrite sigmoid-output constraints into pre-sigmoid coordinates.
///
/// Algebra: `sigmoid` is strictly increasing and bijective onto (0, 1), so for
/// a constant threshold `c` in (0, 1) and pre-activation `z`:
///   `sigmoid(z) <= c  <=>  z <= logit(c)`      (LessEq / LessThan)
///   `sigmoid(z) >= c  <=>  z >= logit(c)`      (GreaterEq / GreaterThan)
/// — directions are PRESERVED (an increasing map never flips inequalities),
/// and relational constraints `sigmoid(z_i) <= sigmoid(z_j) <=> z_i <= z_j`
/// pass through untouched.
///
/// Rounding (the moat): `logit(c)` is generally not an f64; rounding it to
/// NEAREST could shrink the asserted (unsafe) region by up to half an ulp,
/// which is the unsound direction for an UNSAT verdict. Thresholds are
/// therefore rounded OUTWARD — upper bounds up, lower bounds down — so the
/// peeled unsafe region is a SUPERSET of the sigmoid one: any input proven to
/// avoid the peeled region also avoids the original (UNSAT is preserved).
/// The <=1-ulp enlargement cannot mint a wrong SAT either: counterexample
/// candidates are re-validated against the concrete network output downstream.
fn rewrite_sigmoid_constraints(
    constraints: &[OutputConstraint],
) -> Result<Vec<OutputConstraint>, String> {
    let mut rewritten = Vec::with_capacity(constraints.len());
    for constraint in constraints {
        let mapped = match *constraint {
            OutputConstraint::LessEq(i, j) => OutputConstraint::LessEq(i, j),
            OutputConstraint::GreaterEq(i, j) => OutputConstraint::GreaterEq(i, j),
            OutputConstraint::LessThan(i, j) => OutputConstraint::LessThan(i, j),
            OutputConstraint::GreaterThan(i, j) => OutputConstraint::GreaterThan(i, j),
            OutputConstraint::LessEqConst(i, c) => {
                let rhs = inv_sigmoid_outward(c, RoundOutward::Up)
                    .ok_or_else(|| "sigmoid constraint must be in (0, 1)".to_string())?;
                OutputConstraint::LessEqConst(i, rhs)
            }
            OutputConstraint::GreaterEqConst(i, c) => {
                let rhs = inv_sigmoid_outward(c, RoundOutward::Down)
                    .ok_or_else(|| "sigmoid constraint must be in (0, 1)".to_string())?;
                OutputConstraint::GreaterEqConst(i, rhs)
            }
            OutputConstraint::LessThanConst(i, c) => {
                let rhs = inv_sigmoid_outward(c, RoundOutward::Up)
                    .ok_or_else(|| "sigmoid constraint must be in (0, 1)".to_string())?;
                OutputConstraint::LessThanConst(i, rhs)
            }
            OutputConstraint::GreaterThanConst(i, c) => {
                let rhs = inv_sigmoid_outward(c, RoundOutward::Down)
                    .ok_or_else(|| "sigmoid constraint must be in (0, 1)".to_string())?;
                OutputConstraint::GreaterThanConst(i, rhs)
            }
        };
        rewritten.push(mapped);
    }
    Ok(rewritten)
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

#[derive(Clone, Copy)]
enum RoundOutward {
    /// Threshold bounds the region from ABOVE (`z <= t`): round up.
    Up,
    /// Threshold bounds the region from BELOW (`z >= t`): round down.
    Down,
}

/// `logit(value)` rounded one ulp OUTWARD so the rewritten region encloses the
/// exact one (see `rewrite_sigmoid_constraints` for the direction argument).
fn inv_sigmoid_outward(value: f64, direction: RoundOutward) -> Option<f64> {
    let logit = inv_sigmoid(value)?;
    // The nearest-rounded `value/(1-value)` quotient plus `ln` carry up to a
    // few ulps of combined error, so a single next_up/next_down does NOT
    // dominate it (mpmath sweep: 4,702/12,010 thresholds landed inside the
    // exact logit under 1-ulp compensation; worst deficit 1.665e-16 absolute).
    // Nudge by 4eps*(1+|logit|) — an absolute + relative margin that dominates
    // the measured worst case with >2x headroom — then round one more ulp out.
    let margin = 4.0 * f64::EPSILON * (1.0 + logit.abs());
    let rounded = match direction {
        RoundOutward::Up => (logit + margin).next_up(),
        RoundOutward::Down => (logit - margin).next_down(),
    };
    rounded.is_finite().then_some(rounded)
}

fn inv_sigmoid(value: f64) -> Option<f64> {
    if !(0.0 < value && value < 1.0) {
        return None;
    }
    let odds = value / (1.0 - value);
    let logit = odds.ln();
    if logit.is_finite() {
        Some(logit)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vnnlib::VnnLibSpec;
    use crate::{DataType, LayerSpec, Network, OnnxModel, TensorSpec, WeightStore};
    use ny_core::LayerType as LT;
    use std::collections::HashMap;

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

    /// Clause-only cgan band spec: the peel fires, removes the Sigmoid, and
    /// rewrites each clause threshold to an OUTWARD-rounded logit
    /// (#cgan-sigmoid-peel).
    #[test]
    fn peel_clause_band_spec_rewrites_thresholds_outward() {
        with_sigmoid_peel_env(Some("1"), || {
            let mut model = sigmoid_terminal_model();
            let (low, high) = (0.1996644288301468, 0.23966443538665771);
            let mut spec = band_clause_spec(low, high);

            let report = peel_off_terminal_sigmoid_auto(&mut model, &mut spec);
            assert!(report.peeled, "expected peel, got {:?}", report.reason);
            assert_eq!(report.layer_type, Some(LayerType::Sigmoid));

            // Network: Sigmoid removed, output moved to its input.
            assert_eq!(model.network.layers.len(), 1);
            assert_eq!(model.network.outputs[0].name, "z");

            // Spec: clause structure preserved, thresholds -> logit +/- 1 ulp.
            assert!(spec.output_constraints.is_empty());
            assert_eq!(spec.output_constraint_clauses.len(), 2);
            assert!(spec.is_disjunction);
            let exact_high = (high / (1.0 - high)).ln();
            let exact_low = (low / (1.0 - low)).ln();
            match spec.output_constraint_clauses[0][0] {
                OutputConstraint::GreaterEqConst(0, t) => {
                    // Lower-bounding threshold: rounded DOWN (region superset).
                    assert!(t <= exact_high, "GE threshold {t} not rounded down");
                    let slack = 8.0 * f64::EPSILON * (1.0 + exact_high.abs());
                    assert!(t >= exact_high - slack, "over-rounded");
                }
                ref other => panic!("unexpected constraint {other:?}"),
            }
            match spec.output_constraint_clauses[1][0] {
                OutputConstraint::LessEqConst(0, t) => {
                    // Upper-bounding threshold: rounded UP (region superset).
                    assert!(t >= exact_low, "LE threshold {t} not rounded up");
                    let slack = 8.0 * f64::EPSILON * (1.0 + exact_low.abs());
                    assert!(t <= exact_low + slack, "over-rounded");
                }
                ref other => panic!("unexpected constraint {other:?}"),
            }
        });
    }

    /// Auto-peel gates: non-Sigmoid terminal, relational constraint present,
    /// out-of-range threshold, and the NY_SIGMOID_PEEL=0 kill-switch all skip.
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

            // Threshold outside (0, 1): rewrite errors -> skip, spec untouched.
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

        // Kill-switch.
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

    /// Verdict equivalence on a real Gemm+Sigmoid fixture: for a grid of band
    /// thresholds, the clause-infeasibility decision from the UNPEELED graph's
    /// output interval equals the decision from the PEELED graph's interval
    /// against the rewritten thresholds (sigmoid is monotone, so
    /// `sup y < b <=> sup z < logit(b)` away from ulp boundaries).
    #[test]
    fn peel_verdict_equivalence_gemm_sigmoid_fixture() {
        use crate::onnx_proto::{
            attribute_type, AttributeProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto,
            TensorProto, TensorShapeProto, TensorTypeProto, TypeProto, ValueInfoProto,
        };
        use ny_tensor::BoundedTensor;
        use prost::Message;

        // Gemm(w=2, b=0.25) -> Sigmoid, 1-d.
        let value_info = |name: &str, shape: &[i64]| {
            ValueInfoProto {
            name: name.to_string(),
            r#type: Some(TypeProto {
                tensor_type: Some(TensorTypeProto {
                    elem_type: 1,
                    shape: Some(TensorShapeProto {
                        dim: shape
                            .iter()
                            .map(|&value| crate::onnx_proto::tensor_shape_proto::Dimension {
                                value: Some(
                                    crate::onnx_proto::tensor_shape_proto::dimension::Value::DimValue(value),
                                ),
                            })
                            .collect(),
                    }),
                }),
            }),
        }
        };
        let mut gemm = NodeProto {
            input: vec!["x".into(), "w".into(), "b".into()],
            output: vec!["z".into()],
            name: "gemm".into(),
            op_type: "Gemm".into(),
            domain: String::new(),
            attribute: vec![AttributeProto {
                name: "transB".into(),
                r#type: attribute_type::INT,
                i: 1,
                ..Default::default()
            }],
        };
        gemm.attribute.push(AttributeProto {
            name: "alpha".into(),
            r#type: attribute_type::FLOAT,
            f: 1.0,
            ..Default::default()
        });
        let sigmoid = NodeProto {
            input: vec!["z".into()],
            output: vec!["y".into()],
            name: "sig".into(),
            op_type: "Sigmoid".into(),
            domain: String::new(),
            attribute: Vec::new(),
        };
        let graph = GraphProto {
            node: vec![gemm, sigmoid],
            name: "peel_fixture".into(),
            initializer: vec![
                TensorProto {
                    dims: vec![1, 1],
                    data_type: 1,
                    name: "w".into(),
                    float_data: vec![2.0],
                    ..Default::default()
                },
                TensorProto {
                    dims: vec![1],
                    data_type: 1,
                    name: "b".into(),
                    float_data: vec![0.25],
                    ..Default::default()
                },
            ],
            input: vec![value_info("x", &[1, 1])],
            output: vec![value_info("y", &[1, 1])],
            ..Default::default()
        };
        let model_proto = ModelProto {
            ir_version: 8,
            opset_import: vec![OperatorSetIdProto {
                domain: String::new(),
                version: 13,
            }],
            graph: Some(graph),
            ..Default::default()
        };
        let mut bytes = Vec::new();
        model_proto.encode(&mut bytes).expect("encode fixture");

        // x in [-0.4, 0.3] => z in [-0.55, 0.85] => y in [~0.366, ~0.700].
        let lower = ndarray::arr1(&[-0.4_f32]).into_dyn();
        let upper = ndarray::arr1(&[0.3_f32]).into_dyn();
        let box_input = BoundedTensor::new(lower, upper).expect("box");

        let unpeeled = crate::load_onnx_bytes("peel_eq_unpeeled", &bytes).expect("load");
        let y_out = unpeeled
            .to_graph_network()
            .expect("graph")
            .propagate_ibp(&box_input)
            .expect("ibp y");
        let (y_lo, y_hi) = (y_out.lower()[[0]] as f64, y_out.upper()[[0]] as f64);

        let thresholds = [0.1, 0.3, 0.36, 0.5, 0.65, 0.71, 0.9];
        for &low in &thresholds {
            for &high in &thresholds {
                let mut peeled_model =
                    crate::load_onnx_bytes("peel_eq_peeled", &bytes).expect("load");
                let mut spec = band_clause_spec(low, high);
                let report = peel_off_last_softmax_layer(&mut peeled_model, &mut spec);
                assert!(report.peeled, "peel failed: {:?}", report.reason);

                let z_out = peeled_model
                    .to_graph_network()
                    .expect("graph")
                    .propagate_ibp(&box_input)
                    .expect("ibp z");
                let (z_lo, z_hi) = (z_out.lower()[[0]] as f64, z_out.upper()[[0]] as f64);

                let (t_high, t_low) = match (
                    &spec.output_constraint_clauses[0][0],
                    &spec.output_constraint_clauses[1][0],
                ) {
                    (
                        OutputConstraint::GreaterEqConst(0, th),
                        OutputConstraint::LessEqConst(0, tl),
                    ) => (*th, *tl),
                    other => panic!("unexpected rewritten clauses {other:?}"),
                };

                // Clause 1: unsafe iff some y >= high (resp. z >= t_high).
                let ge_feasible_y = y_hi >= high;
                let ge_feasible_z = z_hi >= t_high;
                assert_eq!(
                    ge_feasible_y, ge_feasible_z,
                    "GE clause decision diverged at high={high}: y_hi={y_hi}, z_hi={z_hi}, t={t_high}"
                );
                // Clause 2: unsafe iff some y <= low (resp. z <= t_low).
                let le_feasible_y = y_lo <= low;
                let le_feasible_z = z_lo <= t_low;
                assert_eq!(
                    le_feasible_y, le_feasible_z,
                    "LE clause decision diverged at low={low}: y_lo={y_lo}, z_lo={z_lo}, t={t_low}"
                );
            }
        }
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
    fn rewrite_sigmoid_maps_constants() {
        let constraints = vec![OutputConstraint::LessEqConst(0, 0.25)];
        let rewritten = rewrite_constraints_for_layer(LayerType::Sigmoid, &constraints)
            .expect("rewrite should not error")
            .expect("should rewrite sigmoid constraint");
        assert!(matches!(
            rewritten[0],
            OutputConstraint::LessEqConst(_, rhs) if rhs < 0.0
        ));
    }

    #[test]
    fn rewrite_sigmoid_rejects_out_of_range() {
        let constraints = vec![OutputConstraint::GreaterEqConst(0, 1.5)];
        let rewritten = rewrite_constraints_for_layer(LayerType::Sigmoid, &constraints);
        assert!(rewritten.is_err());
    }
}
