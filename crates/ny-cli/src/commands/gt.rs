// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `ny gt` — geometric ground-truth utilities
//! (`docs/GEOMETRIC_GROUND_TRUTH_PLAN.md`, M2 CLI surface).
//!
//! * `ny gt eval <spec.gt.json> --at x,y,z` — evaluate a `.gt.json` spec at a
//!   point: the EXACT rational reference residual (rounded once to f64) next
//!   to the graph's sound zero-width-IBP enclosure of the same value.
//! * `ny gt verify <model.onnx> <spec> --property dominates|absbound:<eps>
//!   --input-bounds "lo,hi;lo,hi;lo,hi"` — verify the network `f` against the
//!   ground truth `g` on a box via the difference network
//!   ([`verify_against_ground_truth`]). `<spec>` is a `.gt.json` sidecar, or a
//!   VNN-LIB 2.0 dual-network file with a `(ground-truth "…")` relation, in
//!   which case the property and input box come from the file.
//! * `--emit-cert <path>` (dominance only) — additionally derive an
//!   exact-rational, self-checked entailment/Farkas certificate
//!   ([`certify_dominance`], plane/linear and single-level quadratic —
//!   sphere/cylinder/cone — ground truths) and write it as
//!   `.cert.json`.
//! * `--escalate smt` — Route B of the plan: when CROWN answers *unknown*,
//!   re-ask the question exactly — the difference network becomes one
//!   QF_LRA/QF_NRA query to the AY solver ([`SmtEscalation`]). `unsat` proves
//!   the property (Alethe certificate on disk, path reported); `sat` models
//!   are re-validated in exact rational arithmetic before being reported as
//!   witnesses (a placeholder model that fails validation is reported as
//!   "violation exists" and the exit code stays 2 — exit 1 always means a
//!   *validated* witness). The engine that decided is always reported.
//!
//! Exit codes: 0 = property proved, 1 = falsified with a witness, 2 = unknown
//! (sound non-answer), 4 = invalid invocation or operational error.

use anyhow::{anyhow, bail, Context, Result};
use clap::Subcommand;
use ndarray::Array1;
use std::path::{Path, PathBuf};

use ny_core::Bound;
use ny_groundtruth::{
    certify_dominance, verify_against_ground_truth, verify_whole_field_tolerance, EscalateOptions,
    GroundTruthOutcome, GroundTruthSpec, Relation, SmtEscalation, SmtVerdict, WholeFieldOutcome,
};
use ny_onnx::vnnlib::{DualNetworkProperty, NetworkRelation};
use ny_propagate::GraphNetwork;
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use crate::commands::vnncomp::load_graph_network;

/// `ny gt` subcommands.
#[derive(Debug, Subcommand)]
#[allow(clippy::large_enum_variant)] // CLI subcommands — size difference is acceptable
pub(crate) enum GtAction {
    /// Evaluate a ground-truth spec at a point (exact reference + sound graph enclosure)
    Eval {
        /// Path to the .gt.json ground-truth spec
        spec: PathBuf,

        /// Evaluation point as "x,y,z"
        #[arg(long)]
        at: String,
    },

    /// Verify an ONNX network f against a ground-truth spec g (difference network)
    Verify {
        /// Path to the ONNX model for f
        model: PathBuf,

        /// Ground-truth spec: a .gt.json sidecar (needs --property/--input-bounds),
        /// or a .vnnlib dual-network file with a (ground-truth "...") relation
        spec: PathBuf,

        /// Property to check against a .gt.json spec: `dominates` (f >= g) or
        /// `absbound:<eps>` (|f - g| <= eps)
        #[arg(long)]
        property: Option<String>,

        /// Input box for a .gt.json spec: one `lo,hi` pair per input dimension,
        /// `;`-separated (e.g. "-0.5,1.5;-1.25,0.75;-1,1")
        #[arg(long)]
        input_bounds: Option<String>,

        /// Also emit an exact-rational dominance certificate (.cert.json).
        /// Dominance only; certifiable ground truths are the pure-Linear
        /// (plane) builders and the single-level quadratic (SPHERE /
        /// CYLINDER / CONE) builders — see ny-groundtruth::cert for the
        /// remaining nested-square (TORUS) obstruction.
        #[arg(long, value_name = "PATH")]
        emit_cert: Option<PathBuf>,

        /// Escalate unknown verdicts to an exact engine: `smt` re-asks the
        /// difference-network question as one exact AY query (unsat proves
        /// the property with an Alethe certificate; sat witnesses are
        /// validated in exact rational arithmetic). Needs `ay` (via $NY_AY,
        /// PATH, or the Trust stage2 sysroot).
        #[arg(long, value_name = "ENGINE")]
        escalate: Option<String>,
    },

    /// Whole-field "no-escape" tolerance: prove |measured − nominal| ≤ T over
    /// the ENTIRE surface region (continuous, not point-sampled)
    Wholefield {
        /// Path to the ONNX model for the measured-field surrogate f
        model: PathBuf,

        /// Nominal ground-truth spec: a .gt.json sidecar (plane/sphere/… g)
        spec: PathBuf,

        /// Whole-field tolerance T (deviation units, e.g. mm): the proved
        /// bound `|f − g| ≤ T` over the whole region
        #[arg(long)]
        tol: f64,

        /// Surface-parameter box: one `lo,hi` pair per input dimension,
        /// `;`-separated (e.g. "-0.5,1.5;-1.25,0.75;-1,1")
        #[arg(long)]
        input_bounds: String,

        /// Escalate a CROWN-loose (unknown) verdict to the exact AY engine:
        /// `smt` re-asks `∃ x: |f − g| > T` exactly (unsat proves conformance
        /// with an Alethe certificate). Needs `ay` (via $NY_AY / PATH / Trust
        /// stage2 sysroot).
        #[arg(long, value_name = "ENGINE")]
        escalate: Option<String>,
    },
}

/// Dispatch an `ny gt` action. Returns the process exit code (0 proved,
/// 1 falsified, 2 unknown).
pub(crate) fn handle_gt_command(action: GtAction) -> Result<i32> {
    match action {
        GtAction::Eval { spec, at } => {
            let point = parse_point(&at)?;
            let (reference, (lo, hi)) = eval_spec(&spec, point)?;
            println!("exact reference:  {reference}");
            println!("graph enclosure:  [{lo}, {hi}]");
            Ok(0)
        }
        GtAction::Verify {
            model,
            spec,
            property,
            input_bounds,
            emit_cert,
            escalate,
        } => {
            let escalate_smt = match escalate.as_deref() {
                None => false,
                Some("smt") => true,
                Some(other) => {
                    bail!("unsupported --escalate engine `{other}` (only `smt` is supported)")
                }
            };
            run_gt_verify(
                &model,
                &spec,
                property.as_deref(),
                input_bounds.as_deref(),
                emit_cert.as_deref(),
                escalate_smt,
            )
        }
        GtAction::Wholefield {
            model,
            spec,
            tol,
            input_bounds,
            escalate,
        } => {
            let escalate_smt = match escalate.as_deref() {
                None => false,
                Some("smt") => true,
                Some(other) => {
                    bail!("unsupported --escalate engine `{other}` (only `smt` is supported)")
                }
            };
            run_gt_wholefield(&model, &spec, tol, &input_bounds, escalate_smt)
        }
    }
}

/// `ny gt wholefield` core: prove `|f − g| ≤ tol` over the WHOLE surface region
/// (continuous, not point-sampled) via the whole-field tolerance semantics
/// ([`verify_whole_field_tolerance`]). Returns the exit code (0 conforms /
/// 1 violates / 2 unknown).
///
/// Honest scope: the guarantee is over the surrogate `f` (whole-field,
/// certified); the surrogate's own fidelity to the physical surface between
/// measured points is the caller's modeling assumption — the same boundary as
/// the ground-truth M-series.
fn run_gt_wholefield(
    model: &Path,
    spec_path: &Path,
    tol: f64,
    input_bounds: &str,
    escalate_smt: bool,
) -> Result<i32> {
    let g_spec = GroundTruthSpec::load(spec_path)?;
    let bounds = parse_bounds(input_bounds)?;
    let g = g_spec.build()?;
    let f = load_graph_network(model)?;

    let outcome = verify_whole_field_tolerance(&f, &g, &bounds, tol)?;
    match outcome {
        WholeFieldOutcome::Conforms { cert } => {
            println!(
                "✓ whole-field conformance: |measured − nominal| ≤ {tol} mm proved over the \
                 ENTIRE surface region (certificate)"
            );
            println!(
                "  certified deviation band {:?}; max |f − g| ≤ {} mm over the whole box",
                cert.deviation_bounds,
                cert.max_abs_deviation()
            );
            println!("engine: crown (bound propagation decided)");
            Ok(0)
        }
        WholeFieldOutcome::Violates {
            witness,
            witness_region,
            difference,
        } => {
            println!("✗ VIOLATION region near {witness:?}: deviation may exceed {tol} mm");
            println!(
                "  certified out-of-tolerance at x* = {witness:?} (locator region {witness_region:?}); \
                 sound enclosure of f − g there is [{}, {}] mm",
                difference.lower(),
                difference.upper()
            );
            println!("engine: crown (grid witness certified the violation)");
            Ok(1)
        }
        WholeFieldOutcome::Unknown { deviation_bounds } => {
            println!(
                "unknown: CROWN could not fit the deviation field inside ±{tol} mm and no \
                 certain violation was found; best f − g bounds {deviation_bounds:?}"
            );
            if escalate_smt {
                // Route B: re-ask |f − g| ≤ tol exactly over the difference net.
                return escalate_to_smt(&f, &g, Relation::AbsBound(tol), &bounds, true);
            }
            println!("  (re-run with `--escalate smt` to decide it exactly, or refine the region)");
            Ok(2)
        }
        other => {
            println!("unknown: unrecognized whole-field outcome {other:?}");
            Ok(2)
        }
    }
}

/// `ny gt eval` core: exact reference value + sound zero-width enclosure.
fn eval_spec(spec_path: &Path, point: [f64; 3]) -> Result<(f64, (f32, f32))> {
    let spec = GroundTruthSpec::load(spec_path)?;
    let reference = spec.reference_eval(point)?;
    let graph = spec.build()?;
    let enclosure = point_enclosure(&graph, point)?;
    Ok((reference, enclosure))
}

/// Sound zero-width-IBP enclosure of a single-output graph at a point.
fn point_enclosure(graph: &GraphNetwork, point: [f64; 3]) -> Result<(f32, f32)> {
    let arr = Array1::from(vec![point[0] as f32, point[1] as f32, point[2] as f32]).into_dyn();
    let tensor = BoundedTensor::new(arr.clone(), arr)?;
    let out = graph.propagate_ibp(&tensor)?;
    let (Some(&lo), Some(&hi)) = (out.lower().first(), out.upper().first()) else {
        bail!("ground-truth graph produced no output");
    };
    Ok((lo, hi))
}

/// `ny gt verify` core. Returns the exit code.
fn run_gt_verify(
    model: &Path,
    spec_path: &Path,
    property: Option<&str>,
    input_bounds: Option<&str>,
    emit_cert: Option<&Path>,
    escalate_smt: bool,
) -> Result<i32> {
    let is_vnnlib = spec_path
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("vnnlib"));

    let (g_spec, relation, bounds, strict_unsafe) = if is_vnnlib {
        if property.is_some() || input_bounds.is_some() {
            bail!(
                "--property/--input-bounds are read from the VNN-LIB file; \
                 pass them only with a .gt.json spec"
            );
        }
        load_vnnlib_ground_truth(spec_path)?
    } else {
        let property =
            property.ok_or_else(|| anyhow!("--property is required with a .gt.json spec"))?;
        let bounds_str = input_bounds
            .ok_or_else(|| anyhow!("--input-bounds is required with a .gt.json spec"))?;
        (
            GroundTruthSpec::load(spec_path)?,
            parse_property(property)?,
            parse_bounds(bounds_str)?,
            true,
        )
    };

    let g = g_spec.build()?;
    let f = load_graph_network(model)?;

    if emit_cert.is_some() && !matches!(relation, Relation::Dominates) {
        bail!("--emit-cert covers the dominance property only");
    }

    let outcome = verify_against_ground_truth(&f, &g, relation, &bounds)?;
    let mut code = report_outcome(&outcome, strict_unsafe);
    if code != 2 {
        println!("engine: crown (bound propagation decided)");
    }

    if let Some(cert_path) = emit_cert {
        // The certificate is derived independently (exact-rational CROWN,
        // self-checked); when it closes the property it is itself a proof of
        // f - g >= 0 (strictly positive bound also settles a non-strict
        // unsafe clause), so it may upgrade an Unknown float verdict.
        let cert = certify_dominance(&f, &g, &bounds)
            .map_err(|e| anyhow!("certificate not emitted: {e}"))?;
        std::fs::write(cert_path, &cert.certificate_json)
            .with_context(|| format!("writing certificate to {}", cert_path.display()))?;
        println!(
            "certificate: exact lower bound {} on f - g; written to {}",
            cert.lower_bound,
            cert_path.display()
        );
        if code == 2 && (strict_unsafe || rational_is_positive(&cert.lower_bound)) {
            println!("exact certificate closes the property: proved");
            println!("engine: exact-rational CROWN certificate");
            code = 0;
        }
    }

    if code == 2 && escalate_smt {
        code = escalate_to_smt(&f, &g, relation, &bounds, strict_unsafe)?;
    }
    Ok(code)
}

/// Route B: re-ask the unknown as one exact AY query over the difference
/// network. Returns the new exit code (0 proved / 1 validated witness /
/// 2 still unknown). Exit 1 is reserved for *validated* witnesses: a `sat`
/// whose placeholder model fails exact validation is reported honestly as
/// "violation exists" and keeps exit 2.
fn escalate_to_smt(
    f: &GraphNetwork,
    g: &GraphNetwork,
    relation: Relation,
    bounds: &[Bound],
    strict_unsafe: bool,
) -> Result<i32> {
    let Some(solver) = SmtEscalation::locate() else {
        bail!(
            "--escalate smt: no `ay` solver found (set NY_AY, put `ay` on PATH, \
             or build the Trust stage2 sysroot)"
        );
    };
    let options = EscalateOptions {
        // Non-strict unsafe clause (f <= g unsafe): safety needs the strict
        // margin f - g > 0, so the violation is encoded as f - g <= 0.
        require_strict_margin: !strict_unsafe,
        ..EscalateOptions::default()
    };
    let verdict = solver
        .escalate(f, g, relation, bounds, &options)
        .map_err(|e| anyhow!("smt escalation failed: {e}"))?;
    match verdict {
        SmtVerdict::Proved { query, certificate } => {
            println!(
                "smt escalation: unsat — the relation holds on the whole box \
                 (exact real semantics); query: {}",
                query.display()
            );
            match certificate {
                Some(cert) => println!("certificate: Alethe proof at {}", cert.display()),
                None => println!("certificate: MISSING (ay did not leave an Alethe file)"),
            }
            println!("engine: smt (ay, escalated after crown unknown)");
            Ok(0)
        }
        SmtVerdict::Falsified {
            witness,
            witness_exact,
            output_index,
            difference_exact,
            query,
        } => {
            println!(
                "smt escalation: falsified at x* = {witness:?} (exact {witness_exact:?}); \
                 output {output_index} of f - g is {difference_exact} (validated in exact \
                 rational arithmetic); query: {}",
                query.display()
            );
            println!("engine: smt (ay, escalated after crown unknown)");
            Ok(1)
        }
        SmtVerdict::ViolationExists { detail, query } => {
            println!(
                "smt escalation: ay reports sat (a violation exists), but the printed \
                 model did not validate: {detail}; query: {}",
                query.display()
            );
            println!(
                "unknown: existence-only evidence is not reported as a witness (exit 1 \
                 is reserved for validated counterexamples)"
            );
            println!("engine: smt (ay, escalated after crown unknown)");
            Ok(2)
        }
        SmtVerdict::Unknown { reason, query } => {
            println!(
                "smt escalation: unknown ({reason}); query: {}",
                query.display()
            );
            println!("engine: none (crown unknown, smt unknown)");
            Ok(2)
        }
        other => {
            println!("smt escalation: unrecognized verdict {other:?}");
            Ok(2)
        }
    }
}

/// Resolve a VNN-LIB dual-network file with a ground-truth relation into
/// `(sidecar spec, relation, input box, strict_unsafe)`.
fn load_vnnlib_ground_truth(
    vnnlib_path: &Path,
) -> Result<(GroundTruthSpec, Relation, Vec<Bound>, bool)> {
    let spec = ny_onnx::vnnlib::load_vnnlib(vnnlib_path)?;
    let dual = spec
        .dual_network
        .as_ref()
        .ok_or_else(|| anyhow!("VNN-LIB file has no validated dual-network property"))?;

    let DualNetworkProperty::DominatesSecond { strict_unsafe } = dual.property else {
        bail!(
            "VNN-LIB dual-network property is not ground-truth dominance \
             (got {:?})",
            dual.property
        );
    };
    let gt_path = dual
        .networks
        .iter()
        .find_map(|network| match &network.relation_to {
            Some((NetworkRelation::GroundTruth(path), _)) => Some(path.clone()),
            _ => None,
        })
        .ok_or_else(|| anyhow!("no (ground-truth \"...\") relation among declared networks"))?;

    // SOUNDNESS GATE: the difference network evaluates f and g at the SAME
    // point, which the VNN-LIB semantics only authorize when every input is
    // explicitly equality-coupled with matching boxes.
    if !dual.shared_input_coupling {
        bail!(
            "ground-truth dominance requires explicit input coupling \
             (assert (== X_f[i] X_g[i]) for every i) with matching bounds"
        );
    }

    // Outward-directed f64 -> f32 rounding: the verified region is a superset
    // of the declared one (sound; mirrors split_input_bounds_f32).
    let mut bounds = Vec::with_capacity(dual.f_input_bounds.len());
    for (i, &(lo, hi)) in dual.f_input_bounds.iter().enumerate() {
        if !lo.is_finite() || !hi.is_finite() {
            bail!("input {i} has non-finite bounds [{lo}, {hi}]; the box must be finite");
        }
        let lo32 = {
            let v = lo as f32;
            if v.is_finite() {
                next_down_f32(v)
            } else {
                f32::NEG_INFINITY
            }
        };
        let hi32 = {
            let v = hi as f32;
            if v.is_finite() {
                next_up_f32(v)
            } else {
                f32::INFINITY
            }
        };
        bounds.push(Bound::new_allow_infinite(lo32, hi32));
    }

    // Resolve the sidecar path against the VNN-LIB file's directory.
    let sidecar = {
        let p = Path::new(&gt_path);
        if p.is_absolute() {
            p.to_path_buf()
        } else {
            vnnlib_path.parent().unwrap_or(Path::new(".")).join(p)
        }
    };
    let g_spec = GroundTruthSpec::load(&sidecar)?;
    Ok((g_spec, Relation::Dominates, bounds, strict_unsafe))
}

/// Print the outcome and derive the exit code. `strict_unsafe` is true when
/// proving `f − g ≥ 0` settles the property (the unsafe clause was strict, or
/// the caller asked for the dominance relation directly); when false, an
/// exactly-zero margin remains unsafe and the verdict is downgraded to
/// unknown unless every certified lower bound is strictly positive.
fn report_outcome(outcome: &GroundTruthOutcome, strict_unsafe: bool) -> i32 {
    match outcome {
        GroundTruthOutcome::Verified { difference_bounds } => {
            if !strict_unsafe && difference_bounds.iter().any(|bound| bound.lower() <= 0.0) {
                println!(
                    "unknown: proved f - g >= 0, but the unsafe clause is non-strict \
                     (f <= g unsafe) and the certified margin is not strictly positive: \
                     {difference_bounds:?}"
                );
                return 2;
            }
            println!("verified: f - g bounds {difference_bounds:?}");
            0
        }
        GroundTruthOutcome::Falsified {
            witness,
            difference,
        } => {
            println!(
                "falsified: at x* = {witness:?} the sound enclosure of f - g is \
                 [{}, {}], certainly violating the relation",
                difference.lower(),
                difference.upper()
            );
            1
        }
        GroundTruthOutcome::Unknown { difference_bounds } => {
            println!("unknown: best f - g bounds {difference_bounds:?}");
            2
        }
        other => {
            println!("unknown: unrecognized outcome {other:?}");
            2
        }
    }
}

/// Parse "x,y,z" into a 3-D point.
fn parse_point(text: &str) -> Result<[f64; 3]> {
    let parts: Vec<f64> = text
        .split(',')
        .map(|part| {
            part.trim()
                .parse::<f64>()
                .with_context(|| format!("bad coordinate `{part}` in --at"))
        })
        .collect::<Result<_>>()?;
    let [x, y, z] = parts.as_slice() else {
        bail!(
            "--at needs exactly three comma-separated coordinates, got {}",
            parts.len()
        );
    };
    Ok([*x, *y, *z])
}

/// Parse `dominates` / `absbound:<eps>` into a [`Relation`].
fn parse_property(text: &str) -> Result<Relation> {
    if text.eq_ignore_ascii_case("dominates") {
        return Ok(Relation::Dominates);
    }
    if let Some(eps) = text
        .strip_prefix("absbound:")
        .or_else(|| text.strip_prefix("ABSBOUND:"))
    {
        let eps: f64 = eps
            .parse()
            .with_context(|| format!("bad epsilon `{eps}` in --property absbound:<eps>"))?;
        return Ok(Relation::AbsBound(eps));
    }
    bail!("unknown --property `{text}` (expected `dominates` or `absbound:<eps>`)")
}

/// Parse `lo,hi;lo,hi;...` into a box, with outward-directed f64 -> f32
/// rounding (the verified region is a superset of the requested one).
fn parse_bounds(text: &str) -> Result<Vec<Bound>> {
    let mut bounds = Vec::new();
    for (i, pair) in text.split(';').enumerate() {
        let Some((lo, hi)) = pair.split_once(',') else {
            bail!("--input-bounds dimension {i} is not a `lo,hi` pair: `{pair}`");
        };
        let lo: f64 = lo
            .trim()
            .parse()
            .with_context(|| format!("bad lower bound in dimension {i}"))?;
        let hi: f64 = hi
            .trim()
            .parse()
            .with_context(|| format!("bad upper bound in dimension {i}"))?;
        if !lo.is_finite() || !hi.is_finite() || lo > hi {
            bail!("--input-bounds dimension {i} is not a finite ordered interval: [{lo}, {hi}]");
        }
        let lo32 = next_down_f32(lo as f32);
        let hi32 = next_up_f32(hi as f32);
        bounds.push(Bound::new(lo32, hi32));
    }
    if bounds.is_empty() {
        bail!("--input-bounds must contain at least one `lo,hi` pair");
    }
    Ok(bounds)
}

/// Is a `"n/d"` / `"n"` exact-rational string strictly positive? (The
/// denominator emitted by ny-cert is always positive, so the sign and
/// zero-ness live entirely in the numerator.)
fn rational_is_positive(text: &str) -> bool {
    let numerator = text.split('/').next().unwrap_or("");
    !numerator.starts_with('-') && numerator.chars().any(|c| c.is_ascii_digit() && c != '0')
}

#[cfg(test)]
mod tests {
    use super::*;
    use ny_onnx::onnx_proto::{
        tensor_shape_proto, AttributeProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto,
        TensorProto, TensorShapeProto, TensorTypeProto, TypeProto, ValueInfoProto,
    };
    use prost::Message;

    fn f32_value_info(name: &str, shape: &[i64]) -> ValueInfoProto {
        ValueInfoProto {
            name: name.to_string(),
            r#type: Some(TypeProto {
                tensor_type: Some(TensorTypeProto {
                    elem_type: 1,
                    shape: Some(TensorShapeProto {
                        dim: shape
                            .iter()
                            .map(|dim| tensor_shape_proto::Dimension {
                                value: Some(tensor_shape_proto::dimension::Value::DimValue(*dim)),
                            })
                            .collect(),
                    }),
                }),
            }),
        }
    }

    fn f32_tensor(name: &str, dims: Vec<i64>, data: Vec<f32>) -> TensorProto {
        TensorProto {
            dims,
            data_type: 1,
            name: name.to_string(),
            float_data: data,
            ..Default::default()
        }
    }

    fn gemm(name: &str, x: &str, w: &str, b: &str, y: &str) -> NodeProto {
        NodeProto {
            op_type: "Gemm".to_string(),
            input: vec![x.to_string(), w.to_string(), b.to_string()],
            output: vec![y.to_string()],
            name: name.to_string(),
            domain: String::new(),
            attribute: vec![AttributeProto {
                name: "transB".to_string(),
                i: Some(1),
                r#type: ny_onnx::onnx_proto::attribute_type::INT,
                ..Default::default()
            }],
        }
    }

    /// f(x) = |x0 − 0.5| + |x1 + 0.25| − 6, a hand-weighted 3->4->1 FC-ReLU
    /// surrogate (the "fitted" cylinder score net for the a3d acceptance
    /// test): on the box around the fitted axis it stays within [−6, −4],
    /// dominating the cylinder residual (∈ [−9, −7]) with margin ≥ 3.
    fn surrogate_onnx_bytes() -> Vec<u8> {
        let w1 = f32_tensor(
            "w1",
            vec![4, 3],
            vec![
                1.0, 0.0, 0.0, //
                -1.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, //
                0.0, -1.0, 0.0,
            ],
        );
        let b1 = f32_tensor("b1", vec![4], vec![-0.5, 0.5, 0.25, -0.25]);
        let w2 = f32_tensor("w2", vec![1, 4], vec![1.0, 1.0, 1.0, 1.0]);
        let b2 = f32_tensor("b2", vec![1], vec![-6.0]);
        let graph = GraphProto {
            node: vec![
                gemm("gemm1", "input", "w1", "b1", "z1"),
                NodeProto {
                    op_type: "Relu".to_string(),
                    input: vec!["z1".to_string()],
                    output: vec!["a1".to_string()],
                    name: "relu1".to_string(),
                    domain: String::new(),
                    attribute: Vec::new(),
                },
                gemm("gemm2", "a1", "w2", "b2", "output"),
            ],
            name: "cylinder_surrogate".to_string(),
            initializer: vec![w1, b1, w2, b2],
            sparse_initializer: Vec::new(),
            input: vec![f32_value_info("input", &[1, 3])],
            output: vec![f32_value_info("output", &[1, 1])],
            value_info: Vec::new(),
        };
        let model = ModelProto {
            ir_version: 9,
            opset_import: vec![OperatorSetIdProto {
                domain: String::new(),
                version: 17,
            }],
            producer_name: "ny-gt-test".to_string(),
            producer_version: String::new(),
            domain: String::new(),
            model_version: 1,
            doc_string: String::new(),
            graph: Some(graph),
        };
        let mut bytes = Vec::new();
        model.encode(&mut bytes).expect("encode surrogate onnx");
        bytes
    }

    /// The realistic fitted-cylinder parameter set for the a3d acceptance
    /// test: axis z, radius 3.0, axis point (0.5, −0.25, 0) — all exactly
    /// f32-representable, as a3d exports them.
    fn cylinder_spec() -> GroundTruthSpec {
        GroundTruthSpec::cylinder([0.0, 0.0, 1.0], [0.5, -0.25, 0.0], 3.0)
    }

    const SURROGATE_BOX: &str = "-0.5,1.5;-1.25,0.75;-1,1";

    #[test]
    fn a3d_cylinder_acceptance_end_to_end_via_cli_path() {
        // M2 deliverable 4: FC-ReLU surrogate (ONNX) vs cylinder_residual
        // from a realistic fitted set, through the CLI verify path
        // (.gt.json sidecar + ONNX loader + difference network + CROWN).
        let dir = tempfile::tempdir().expect("tempdir");
        let model = dir.path().join("surrogate.onnx");
        std::fs::write(&model, surrogate_onnx_bytes()).expect("write onnx");
        let sidecar = dir.path().join("cyl.gt.json");
        std::fs::write(&sidecar, cylinder_spec().to_json_string().unwrap()).expect("write sidecar");

        // Dominates: f − g ≥ 3 on the box; must verify (exit 0).
        let code = run_gt_verify(
            &model,
            &sidecar,
            Some("dominates"),
            Some(SURROGATE_BOX),
            None,
            false,
        )
        .expect("verify runs");
        assert_eq!(code, 0, "dominance must be Verified");

        // AbsBound with a too-small eps: |f − g| ≤ 1 is certainly violated
        // (f − g ≥ 3 everywhere) — the witness search must falsify (exit 1).
        let code = run_gt_verify(
            &model,
            &sidecar,
            Some("absbound:1.0"),
            Some(SURROGATE_BOX),
            None,
            false,
        )
        .expect("verify runs");
        assert_eq!(code, 1, "absbound:1 must be Falsified with a witness");
    }

    #[test]
    fn vnnlib_ground_truth_reaches_difference_verify_path() {
        // M2 deliverable 2 (end-to-end): a dual-network VNN-LIB property with
        // a ground-truth second network resolves through the .gt.json loader
        // and reaches the difference-network verify path.
        let dir = tempfile::tempdir().expect("tempdir");
        let model = dir.path().join("surrogate.onnx");
        std::fs::write(&model, surrogate_onnx_bytes()).expect("write onnx");
        let sidecar = dir.path().join("cyl.gt.json");
        std::fs::write(&sidecar, cylinder_spec().to_json_string().unwrap()).expect("write sidecar");
        let vnnlib = dir.path().join("dominates.vnnlib");
        std::fs::write(
            &vnnlib,
            r#"
(vnnlib-version 2.0)
(declare-network f
  (declare-input X_f Float32 [3])
  (declare-output Y_f Float32 [1])
)
(declare-network g
  (ground-truth "cyl.gt.json")
  (declare-input X_g Float32 [3])
  (declare-output Y_g Float32 [1])
)
(assert (>= X_f[0] -0.5)) (assert (<= X_f[0] 1.5))
(assert (>= X_g[0] -0.5)) (assert (<= X_g[0] 1.5))
(assert (>= X_f[1] -1.25)) (assert (<= X_f[1] 0.75))
(assert (>= X_g[1] -1.25)) (assert (<= X_g[1] 0.75))
(assert (>= X_f[2] -1)) (assert (<= X_f[2] 1))
(assert (>= X_g[2] -1)) (assert (<= X_g[2] 1))
(assert (== X_f[0] X_g[0]))
(assert (== X_f[1] X_g[1]))
(assert (== X_f[2] X_g[2]))
(assert (< Y_f[0] Y_g[0]))
"#,
        )
        .expect("write vnnlib");

        let code = run_gt_verify(&model, &vnnlib, None, None, None, false).expect("verify runs");
        assert_eq!(code, 0, "vnnlib ground-truth dominance must be Verified");
    }

    #[test]
    fn emit_cert_writes_selfchecked_plane_certificate() {
        // M3 (cert half): the plane ground truth certifies end-to-end with a
        // .cert.json; the quadric cylinder is refused with the documented
        // obstruction.
        let dir = tempfile::tempdir().expect("tempdir");
        let model = dir.path().join("surrogate.onnx");
        std::fs::write(&model, surrogate_onnx_bytes()).expect("write onnx");

        // Plane below the surrogate: g(x) = x2 − 8; f − g ≥ 1 on the box.
        let plane = dir.path().join("plane.gt.json");
        std::fs::write(
            &plane,
            GroundTruthSpec::plane([0.0, 0.0, 1.0], -8.0)
                .to_json_string()
                .unwrap(),
        )
        .expect("write plane sidecar");
        let cert_path = dir.path().join("dominance.cert.json");
        let code = run_gt_verify(
            &model,
            &plane,
            Some("dominates"),
            Some(SURROGATE_BOX),
            Some(&cert_path),
            false,
        )
        .expect("verify + certify runs");
        assert_eq!(code, 0, "plane dominance must be Verified");
        let cert: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&cert_path).unwrap())
                .expect("cert json parses");
        assert_eq!(cert["format"], "ny-cert/ground-truth-dominance/v1");
        assert!(cert["entailment"].is_object() && cert["farkas"].is_object());

        // The quadric side CERTIFIES since M3 (`f647603a`): the pow2 envelope
        // (`pow2_tangent` / `pow2_secant`, sorry-free in the exact Lake-pinned
        // Clean module `Crownproof.Pow2Envelope`) un-refused
        // `QuadraticSideNotYetCertifiable` for `PowConstant(2)` g. This
        // assertion used to require a refusal and was simply never updated
        // when that capability landed, so it has been failing on trunk since.
        let cyl = dir.path().join("cyl.gt.json");
        std::fs::write(&cyl, cylinder_spec().to_json_string().unwrap()).unwrap();
        let quad_cert = dir.path().join("quadric.cert.json");
        let code = run_gt_verify(
            &model,
            &cyl,
            Some("dominates"),
            Some(SURROGATE_BOX),
            Some(&quad_cert),
            false,
        )
        .expect("quadric dominance certifies since M3");
        assert_eq!(code, 0, "quadric dominance must be Verified");
        let qcert: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&quad_cert).unwrap())
                .expect("quadric cert json parses");
        assert_eq!(qcert["format"], "ny-cert/ground-truth-dominance/v1");
        assert!(
            qcert["entailment"].is_object() && qcert["farkas"].is_object(),
            "quadric cert must carry the entailment + Farkas objects"
        );
    }

    #[test]
    fn gt_eval_reports_exact_reference_inside_enclosure() {
        let dir = tempfile::tempdir().expect("tempdir");
        let sidecar = dir.path().join("cyl.gt.json");
        std::fs::write(&sidecar, cylinder_spec().to_json_string().unwrap()).unwrap();
        let x = [1.5, -0.25, 0.5];
        let (reference, (lo, hi)) = eval_spec(&sidecar, x).expect("eval runs");
        // Exact: (1.5-0.5)^2 + 0^2 - 9 = -8.
        assert_eq!(reference, -8.0);
        assert!(f64::from(lo) <= reference && reference <= f64::from(hi));
    }

    #[test]
    fn parse_helpers_reject_malformed_input() {
        assert!(parse_point("1,2").is_err());
        assert!(parse_point("1,2,three").is_err());
        assert_eq!(parse_point(" 1 , 2 , 3 ").unwrap(), [1.0, 2.0, 3.0]);

        assert!(matches!(
            parse_property("dominates").unwrap(),
            Relation::Dominates
        ));
        assert!(matches!(
            parse_property("absbound:0.5").unwrap(),
            Relation::AbsBound(eps) if (eps - 0.5).abs() < 1e-12
        ));
        assert!(parse_property("gteq").is_err());

        assert!(parse_bounds("0,1;2,1").is_err(), "inverted interval");
        assert!(parse_bounds("0;1").is_err(), "missing pair");
        let b = parse_bounds("0,1;-2,3").unwrap();
        assert_eq!(b.len(), 2);
        assert!(
            b[0].lower() <= 0.0 && b[0].upper() >= 1.0,
            "outward rounded"
        );
    }

    #[test]
    fn escalate_rejects_unknown_engines() {
        let err = handle_gt_command(GtAction::Verify {
            model: PathBuf::from("f.onnx"),
            spec: PathBuf::from("g.gt.json"),
            property: Some("dominates".to_string()),
            input_bounds: Some("0,1".to_string()),
            emit_cert: None,
            escalate: Some("z3".to_string()),
        })
        .expect_err("unknown engine must be rejected");
        assert!(
            err.to_string().contains("only `smt` is supported"),
            "got: {err}"
        );
    }

    /// f(x) = |x0 − 0.5| + |x1 + 0.25| − 8 as ONNX: same absolute-value pair
    /// structure as [`surrogate_onnx_bytes`] with the read-out bias lowered
    /// so the CROWN margin goes negative while true dominance survives.
    #[cfg(feature = "external-ay")]
    fn tight_surrogate_onnx_bytes() -> Vec<u8> {
        let w1 = f32_tensor(
            "w1",
            vec![4, 3],
            vec![
                1.0, 0.0, 0.0, //
                -1.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, //
                0.0, -1.0, 0.0,
            ],
        );
        let b1 = f32_tensor("b1", vec![4], vec![-0.5, 0.5, 0.25, -0.25]);
        let w2 = f32_tensor("w2", vec![1, 4], vec![1.0, 1.0, 1.0, 1.0]);
        let b2 = f32_tensor("b2", vec![1], vec![-8.0]);
        let graph = GraphProto {
            node: vec![
                gemm("gemm1", "input", "w1", "b1", "z1"),
                NodeProto {
                    op_type: "Relu".to_string(),
                    input: vec!["z1".to_string()],
                    output: vec!["a1".to_string()],
                    name: "relu1".to_string(),
                    domain: String::new(),
                    attribute: Vec::new(),
                },
                gemm("gemm2", "a1", "w2", "b2", "output"),
            ],
            name: "tight_cylinder_surrogate".to_string(),
            initializer: vec![w1, b1, w2, b2],
            sparse_initializer: Vec::new(),
            input: vec![f32_value_info("input", &[1, 3])],
            output: vec![f32_value_info("output", &[1, 1])],
            value_info: Vec::new(),
        };
        let model = ModelProto {
            ir_version: 9,
            opset_import: vec![OperatorSetIdProto {
                domain: String::new(),
                version: 17,
            }],
            producer_name: "ny-gt-test".to_string(),
            producer_version: String::new(),
            domain: String::new(),
            model_version: 1,
            doc_string: String::new(),
            graph: Some(graph),
        };
        let mut bytes = Vec::new();
        model.encode(&mut bytes).expect("encode surrogate onnx");
        bytes
    }

    /// A whole-field plane-tracking surrogate `f(x) = x2 + bias` as ONNX: one
    /// Gemm with weight `[0, 0, 1]` (reads the third coordinate) and the given
    /// read-out bias. Against the nominal plane `g(x) = x2` the deviation is
    /// the constant `bias` — a clean whole-field conform/violate demo.
    fn plane_surrogate_onnx_bytes(bias: f32) -> Vec<u8> {
        let w = f32_tensor("w", vec![1, 3], vec![0.0, 0.0, 1.0]);
        let b = f32_tensor("b", vec![1], vec![bias]);
        let graph = GraphProto {
            node: vec![gemm("gemm", "input", "w", "b", "output")],
            name: "plane_surrogate".to_string(),
            initializer: vec![w, b],
            sparse_initializer: Vec::new(),
            input: vec![f32_value_info("input", &[1, 3])],
            output: vec![f32_value_info("output", &[1, 1])],
            value_info: Vec::new(),
        };
        let model = ModelProto {
            ir_version: 9,
            opset_import: vec![OperatorSetIdProto {
                domain: String::new(),
                version: 17,
            }],
            producer_name: "ny-gt-test".to_string(),
            producer_version: String::new(),
            domain: String::new(),
            model_version: 1,
            doc_string: String::new(),
            graph: Some(graph),
        };
        let mut bytes = Vec::new();
        model
            .encode(&mut bytes)
            .expect("encode plane surrogate onnx");
        bytes
    }

    #[test]
    fn wholefield_conforms_and_violates_through_cli_path() {
        // Wave-3 whole-field deliverable: |measured − nominal| ≤ T over the
        // WHOLE region, through the CLI path (.gt.json nominal + ONNX field +
        // difference network + CROWN), reporting Conforms / Violates.
        let dir = tempfile::tempdir().expect("tempdir");
        // Surrogate f = x2 + 0.05 (constant 0.05 deviation from the plane).
        let model = dir.path().join("field.onnx");
        std::fs::write(&model, plane_surrogate_onnx_bytes(0.05)).expect("write onnx");
        // Nominal g = x2 (the plane z = 0), exact-constant sidecar.
        let sidecar = dir.path().join("plane.gt.json");
        std::fs::write(
            &sidecar,
            GroundTruthSpec::plane([0.0, 0.0, 1.0], 0.0)
                .to_json_string()
                .unwrap(),
        )
        .expect("write sidecar");

        // Conforms: |f − g| ≡ 0.05 ≤ 0.10 over the whole box (exit 0).
        let code =
            run_gt_wholefield(&model, &sidecar, 0.10, SURROGATE_BOX, false).expect("verify runs");
        assert_eq!(code, 0, "whole-field |f − g| ≤ 0.10 must Conform");

        // Violates: |f − g| ≡ 0.05 exceeds 0.02 everywhere (exit 1).
        let code =
            run_gt_wholefield(&model, &sidecar, 0.02, SURROGATE_BOX, false).expect("verify runs");
        assert_eq!(code, 1, "whole-field tol 0.02 must report a VIOLATION");
    }

    #[test]
    fn wholefield_rejects_unknown_escalate_engine() {
        let err = handle_gt_command(GtAction::Wholefield {
            model: PathBuf::from("f.onnx"),
            spec: PathBuf::from("g.gt.json"),
            tol: 0.1,
            input_bounds: "0,1".to_string(),
            escalate: Some("z3".to_string()),
        })
        .expect_err("unknown engine must be rejected");
        assert!(
            err.to_string().contains("only `smt` is supported"),
            "got: {err}"
        );
    }

    #[test]
    #[cfg(feature = "external-ay")]
    fn escalate_smt_decides_what_crown_cannot() {
        // Route B end to end through the CLI path. With u = x0 − 0.5 and
        // v = x1 + 0.25 (both ranging over [−1, 1] on SURROGATE_BOX):
        //   f − g = |u| + |v| − u² − v² + 1, true minimum 1 > 0 (dominance
        //   holds), but CROWN's certified lower bound is −1 (the −t² secant
        //   relaxation is loose at the interior binding point t = 0), so the
        //   float path is Unknown (exit 2). The exact SMT query is unsat.
        SmtEscalation::locate()
            .expect("external-ay conformance requires pinned ay via NY_AY or PATH");
        let dir = tempfile::tempdir().expect("tempdir");
        let model = dir.path().join("tight_surrogate.onnx");
        std::fs::write(&model, tight_surrogate_onnx_bytes()).expect("write onnx");
        let sidecar = dir.path().join("cyl.gt.json");
        std::fs::write(&sidecar, cylinder_spec().to_json_string().unwrap()).expect("write sidecar");

        // Without escalation: honest unknown.
        let code = run_gt_verify(
            &model,
            &sidecar,
            Some("dominates"),
            Some(SURROGATE_BOX),
            None,
            false,
        )
        .expect("verify runs");
        assert_eq!(
            code, 2,
            "CROWN alone must be Unknown (else the case is stale)"
        );

        // With --escalate smt: proved exactly.
        let code = run_gt_verify(
            &model,
            &sidecar,
            Some("dominates"),
            Some(SURROGATE_BOX),
            None,
            true,
        )
        .expect("verify + escalation runs");
        assert_eq!(code, 0, "SMT escalation must prove the dominance");
    }
}
