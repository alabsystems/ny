// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #cctsdb-subbox — spec-aware SUB-BOX DISJUNCTION for patch-attack models
//! whose graph cannot be built as one unit.
//!
//! # The defect this closes
//!
//! `cctsdb_yolo_2023`'s `patch-1.onnx` does not build: `Expand_102`'s shape
//! input is produced at runtime by a `Slice` whose start is the spec-bounded
//! patch x-coordinate, so the shape is not resolvable once for the whole box.
//! ny therefore degrades to `#graph-free-sat`, which can only ever mint `sat`,
//! and the 11 GT-`unsat` rows discard their entire 350 s budget.
//!
//! # Why one graph cannot work, and two sub-boxes can
//!
//! ONNX `Slice` CLAMPS its indices, so `Slice(t, start=x, end=x+1, axis=a)`
//! has extent `clamp(x+1,0,L) - clamp(x,0,L)` where `L = shape(t)[a]` — that
//! is 1 for `x < L` and 0 for `x >= L`. The declared box `x in [0, 62]`
//! straddles both regimes (the earlier session's "the extent is a constant"
//! rule was RETRACTED for exactly this reason; it would have applied the patch
//! for every x where the model applies none for 60 of 63 values, a route to a
//! FALSE `unsat`). Split the box at `L` and each half has ONE regime:
//!
//! * `S_high = { trunc(x) >= L }` — the extent is 0, so the `ScatterND` index
//!   tensor is empty, `ScatterND(data, [], [])` IS `data`, and the whole patch
//!   cone is a proven NO-OP that prunes away. That excised graph builds, and
//!   the ordinary cell driver decides it.
//! * `S_low  = { trunc(x) <  L }` — the patch DOES apply. It applies as a 0/1
//!   mask multiplied elementwise into a contiguous window of the flat network
//!   input, and the detector proves that mask is a function of the two integer
//!   coordinates ALONE. So on one integer cell the patched model is exactly the
//!   EXCISED model run on the declared input with the masked coordinates set to
//!   zero — a change to the SPEC's point bounds, not to the graph. ONE graph
//!   build therefore serves both halves.
//!
//! (The route that does NOT work, recorded so it is not re-attempted: rewriting
//! the runtime-start `Slice`s to `Gather`s is correct at the ONNX level — ORT
//! confirms bit-equality for `x < L` — but ny's `GatherLayer` carries no
//! runtime-index channel at all. A loaded `Gather` has exactly ONE graph edge,
//! so the index tensor is simply absent and the per-cell evaluator's
//! `Gather requires constant indices` refusal is unreachable-by-fixing.)
//!
//! The two halves are COVERING and DISJOINT over the declared box's integer
//! cells, so the property holds on the box iff it holds on both halves. `unsat`
//! is minted ONLY when BOTH halves are refuted; anything else declines and the
//! caller's behaviour is unchanged. This driver can mint NO other verdict — a
//! violated cell DECLINES rather than claiming `sat`, and the unchanged
//! `#graph-free-sat` lane below finds and ORT-gates the counterexample exactly
//! as it does today.
//!
//! # Soundness, in the order the checks run
//!
//! 1. **Cheapest, most selective gate first.** A spec without exactly two
//!    non-negative, integer-ranged free dims (every other dim a point) leaves
//!    immediately, before a byte of the model is decoded — this function is
//!    consulted for every model that fails to load, in every benchmark.
//! 2. **The split is READ from the ONNX, never assumed.** `L` is resolved by a
//!    small static shape/value evaluator over the shape cone
//!    ([`Facts::static_shape`]); the detector declines if it cannot resolve it.
//! 3. **The template is verified structurally, not by node name.** Every
//!    runtime-start `Slice` must have the exact
//!    `starts = Unsqueeze(Cast(Gather(input, d)))`,
//!    `ends = Unsqueeze(Add(same, k))` for any read `k >= 1`, single-element `axes`, unit `steps`
//!    form; exactly one coordinate may clamp and all of its selections must
//!    share one axis length; there must be exactly one `ScatterND`, writing
//!    into a literal all-ones tensor (either a `ConstantOfShape` filling with
//!    1.0, or a dense `Constant`/initializer verified elementwise), whose index/update cone
//!    depends on the network input ONLY through those two coordinates (cutting
//!    at `Shape`, which reads no values); and the mask's only consumer must be
//!    an elementwise `Mul` against a contiguous static window of the flat
//!    input that does not overlap the free coordinates.
//! 4. **Fires only where ny has no graph.** Checked by re-asking the loader,
//!    so no row that ny can already decide is ever diverted here.
//! 5. **Everything is then CROSS-CHECKED BY EXECUTION** — the method rule this
//!    cone exists to enforce ("structure tells you what a graph says; only
//!    execution tells you what it means"). Because both free coordinates reach
//!    the graph exclusively through `Cast`-to-integer, both models are constant
//!    on each integer cell, so ONE representative per cell is EXHAUSTIVE. Every
//!    cell of every sub-box is run through ONNX Runtime on the ORIGINAL model
//!    and on the transformed instance, and the outputs must be BIT-EQUAL. The
//!    mask itself is READ from the model by ORT, never reconstructed from a
//!    reading of the exporter's intent. Any mismatch, any ORT error, any
//!    non-0/1 mask, and the driver declines.
//! 6. **The verdict comes from the existing cell driver**, over a tightened
//!    CLONE of the spec's input bounds, and only `Verified` counts.
//!    Violated / indeterminate / timeout / load failure all DECLINE.
//!    `NY_NO_CELL_ENUM=1` disables this path too, so one switch still turns the
//!    whole cell-evaluation authority off.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::Instant;

use ny_onnx::onnx_proto::{GraphProto, ModelProto, NodeProto, TensorProto};
use ny_onnx::vnnlib::VnnLibSpec;
use prost::Message;

/// Recursion cap for the static shape/value evaluator.
const EVAL_DEPTH_MAX: usize = 64;
/// Largest integer vector the evaluator will materialise (shape-cone tensors
/// are tiny; anything larger is not a shape computation).
const EVAL_LEN_MAX: usize = 4096;
/// Largest number of integer cells the ORT equivalence gate will replay.
/// The cell-enumeration driver's own budget is 16_384; this is the same order
/// and each point costs ~0.25 ms per model on the cctsdb network.
const EQUIV_CELL_MAX: usize = 16_384;

/// Per-sub-box outcome, logged as the engagement observable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SubBoxVerdict {
    /// Every cell of this sub-box is definitely safe.
    Refuted,
    /// Anything else: the driver may not mint `unsat`.
    Declined,
}

/// The structural facts the driver needs, all READ from the model.
#[derive(Debug)]
struct Plan {
    /// Spec input index of the coordinate whose selection axis can clamp.
    split_dim: usize,
    /// Length of that coordinate's selection axis, read from the ONNX.
    split_len: i64,
    /// Declared upper bound of the split coordinate (integral, `>= split_len`).
    split_hi: f64,
    /// Spec input index of the OTHER free coordinate (never clamps).
    other_dim: usize,
    /// Node index of the single `ScatterND`.
    scatternd: usize,
    /// Tensor name of the `ScatterND` output — the applied patch mask.
    mask_out: String,
    /// Spec input index that the mask's flat element 0 multiplies. The image
    /// occupies `image_offset .. image_offset + mask_len` of the flat input.
    image_offset: usize,
    /// Element count of the mask (== the image window length).
    mask_len: usize,
    /// The free spec dims, ascending.
    free_dims: Vec<usize>,
}

/// Try to decide an instance by the two-sub-box disjunction.
///
/// Returns `true` ONLY when both sub-boxes are refuted; every other situation
/// returns `false` and the caller must continue with its unchanged behaviour.
pub(crate) fn cctsdb_subbox_refutes(onnx: &Path, vnnlib: &Path, deadline: Option<Instant>) -> bool {
    // #phase-cost: see `ny_core::phase_cost` (observability only).
    let _phase_cost = ny_core::phase_cost::enter(ny_core::phase_cost::CostPhase::Exact);
    match run(onnx, vnnlib, deadline) {
        Some(()) => true,
        None => false,
    }
}

fn decline(reason: &str) -> Option<()> {
    eprintln!("#cctsdb-subbox: DECLINED ({reason}); falling through unchanged");
    None
}

fn run(onnx: &Path, vnnlib: &Path, deadline: Option<Instant>) -> Option<()> {
    // Same kill switch as the driver that produces the verdict, so
    // `NY_NO_CELL_ENUM=1` still turns the whole cell-enumeration authority off
    // in one place.
    if crate::preset_gates::no_cell_enum() {
        return None;
    }
    // CHEAPEST, MOST SELECTIVE GATE FIRST. This function is consulted for
    // EVERY model that fails to load, in every benchmark, several of which run
    // on 20 s budgets — so the spec parse (which the falsification lane below
    // performs anyway) screens out everything that is not a two-free-integer-
    // dim patch spec before a single byte of the model is decoded.
    let spec = ny_onnx::vnnlib::load_vnnlib(vnnlib).ok()?;
    let candidate = free_integer_dims(&spec).is_some();
    if !candidate {
        return None;
    }
    let bytes = std::fs::read(onnx).ok()?;
    let model = ModelProto::decode(bytes.as_slice()).ok()?;

    let Some(plan) = detect(&model, &spec) else {
        {
            // Loud only for specs that ARE this shape: a two-free-integer-dim
            // patch spec whose graph does not match the template is exactly
            // the case a future reader needs to see.
            eprintln!(
                "#cctsdb-subbox: DECLINED (spec qualifies but the graph does not match the \
                 checked patch-selection template); falling through unchanged"
            );
        }
        return None;
    };
    // Fire ONLY where ny has no graph. Checked, not assumed: the caller's arm
    // is reached for several unrelated reasons, and a row that ny can already
    // decide through the ordinary pipeline must never be diverted here.
    if crate::commands::vnncomp::load_graph_network(onnx).is_ok() {
        return decline("ny CAN build a graph for this model; the ordinary pipeline owns it");
    }
    eprintln!(
        "#cctsdb-subbox: qualified — free dims {:?}, split dim X_{} at axis length {} \
         (read from the ONNX), mask of {} element(s) over flat input window [{}, {})",
        plan.free_dims,
        plan.split_dim,
        plan.split_len,
        plan.mask_len,
        plan.image_offset,
        plan.image_offset + plan.mask_len
    );

    // Sub-boxes, covering and disjoint on the split coordinate's integer
    // cells: S_low covers trunc(x) in [0, L-1], S_high covers
    // trunc(x) in [L, floor(hi)]. Both keep every other declared bound.
    let split_len_f = plan.split_len as f64;
    let low = tightened(&spec, plan.split_dim, 0.0, split_len_f - 1.0)?;
    let high = tightened(&spec, plan.split_dim, split_len_f, plan.split_hi)?;

    // ONE rewrite serves both halves: delete the ScatterND, so the mask is the
    // literal all-ones `ConstantOfShape` and `Mul` by it is an identity. On
    // S_high that IS the model (the index tensor is empty there). On S_low the
    // patch is re-applied per cell as a change to the SPEC's declared point
    // bounds — see `decide_low`.
    let excised = rewrite_excise_scatternd(&model, &plan)?;
    let mut excised_bytes = Vec::new();
    if excised.encode(&mut excised_bytes).is_err() {
        return decline("the excised proto did not encode");
    }
    let Ok(mut temp) = tempfile::Builder::new()
        .prefix("ny-cctsdb-subbox-")
        .suffix(".onnx")
        .tempfile()
    else {
        return decline("no temp file for the excised model");
    };
    {
        use std::io::Write as _;
        if temp.write_all(&excised_bytes).is_err() || temp.flush().is_err() {
            return decline("could not write the excised model");
        }
    }
    // Proof authority for `CctsdbSubbox`: the certified lowering (every
    // authored BatchNormalization kept; #spine-legacy-lanes M2e).
    let graph = match crate::commands::vnncomp::load_certified_graph_network(temp.path()) {
        Ok(graph) => graph,
        Err(err) => return decline(&format!("the excised model does not build: {err}")),
    };

    let n = spec.input_bounds.len();
    let Ok(mut oracle) = Oracle::new(onnx, &excised_bytes, &model, &plan, n) else {
        return decline("ONNX Runtime is unavailable for the equivalence gate");
    };

    let low_verdict = decide_low(&graph, &mut oracle, &low, &plan, deadline);
    eprintln!("#cctsdb-subbox: S_low verdict = {low_verdict:?}");
    if low_verdict != SubBoxVerdict::Refuted {
        return decline("S_low not refuted");
    }

    let high_verdict = decide_high(&graph, &mut oracle, &high, &plan, deadline);
    eprintln!("#cctsdb-subbox: S_high verdict = {high_verdict:?}");
    if high_verdict != SubBoxVerdict::Refuted {
        return decline("S_high not refuted");
    }

    // AND-combiner: both halves refuted and the halves cover the declared box,
    // so the property holds on the whole box.
    eprintln!(
        "#cctsdb-subbox: BOTH sub-boxes refuted (S_low: trunc(X_{}) < {}, \
         S_high: trunc(X_{}) >= {}) — covering and disjoint, so the declared box is refuted",
        plan.split_dim, plan.split_len, plan.split_dim, plan.split_len
    );
    Some(())
}

/// Tightened CLONE of the spec's input box on one dimension.
///
/// `cell_enum::detect` reads `vnnlib.input_bounds` directly and refuses any
/// per-clause box, so a tightened clone is the only way to hand it a sub-box.
fn tightened(spec: &VnnLibSpec, dim: usize, lo: f64, hi: f64) -> Option<VnnLibSpec> {
    if !(lo.is_finite() && hi.is_finite() && lo <= hi) {
        return None;
    }
    let (dlo, dhi) = *spec.input_bounds.get(dim)?;
    // A sub-box must be INSIDE the declared box; a widened clone would decide
    // points the spec never declared.
    if lo < dlo || hi > dhi {
        return None;
    }
    let mut out = spec.clone();
    out.input_bounds[dim] = (lo, hi);
    if let Some(slot) = out.declared_input_bounds.get_mut(dim) {
        *slot = (lo, hi);
    }
    Some(out)
}

/// The three ONNX-Runtime sessions the driver measures against.
///
/// `original` is the ground truth; `excised` is the graph ny actually reasons
/// over; `mask_probe` is `original` with its output replaced by the ScatterND
/// result, so the applied patch mask is READ from the model rather than
/// reconstructed from a reading of the exporter's intent.
struct Oracle {
    original: ny_onnx::diff::OrtForward,
    excised: ny_onnx::diff::OrtForward,
    mask_probe: ny_onnx::diff::OrtForward,
}

impl Oracle {
    fn new(
        original: &Path,
        excised_bytes: &[u8],
        model: &ModelProto,
        plan: &Plan,
        n: usize,
    ) -> Result<Self, String> {
        let probe = mask_probe_model(model, plan).ok_or("could not build the mask probe model")?;
        let mut probe_bytes = Vec::new();
        probe
            .encode(&mut probe_bytes)
            .map_err(|e| format!("mask probe proto did not encode: {e}"))?;
        Ok(Self {
            original: ny_onnx::diff::OrtForward::from_path(original, n)
                .map_err(|e| format!("original model: {e}"))?,
            excised: ny_onnx::diff::OrtForward::from_model_bytes(excised_bytes, n, false)
                .map_err(|e| format!("excised model: {e}"))?,
            mask_probe: ny_onnx::diff::OrtForward::from_model_bytes(&probe_bytes, n, false)
                .map_err(|e| format!("mask probe model: {e}"))?,
        })
    }
}

/// `model` with its graph output replaced by the ScatterND result, so one ORT
/// forward reports the mask the ORIGINAL model applies at a given cell.
fn mask_probe_model(model: &ModelProto, plan: &Plan) -> Option<ModelProto> {
    let mut out = model.clone();
    let graph = out.graph.as_mut()?;
    graph.output = vec![ny_onnx::onnx_proto::ValueInfoProto {
        name: plan.mask_out.clone(),
        r#type: Some(ny_onnx::onnx_proto::TypeProto {
            tensor_type: Some(ny_onnx::onnx_proto::TensorTypeProto {
                elem_type: 1, // FLOAT
                shape: None,
            }),
        }),
    }];
    Some(out)
}

/// The f32 point ORT is run at for one integer cell: every declared bound's
/// lower endpoint, with the two free coordinates set to the cell's
/// representative.
fn cell_point_f32(spec: &VnnLibSpec, plan: &Plan, rep_split: f64, rep_other: f64) -> Vec<f32> {
    let mut point: Vec<f32> = spec.input_bounds.iter().map(|&(lo, _)| lo as f32).collect();
    point[plan.split_dim] = rep_split as f32;
    point[plan.other_dim] = rep_other as f32;
    point
}

/// Integer cell values of a dimension inside a (sub-)box, with the
/// representative the cell driver would pick for each.
fn cell_reps(spec: &VnnLibSpec, dim: usize) -> Option<Vec<(i64, f64)>> {
    let (lo, hi) = *spec.input_bounds.get(dim)?;
    let (v_min, v_max) = (lo.floor(), hi.floor());
    let count = (v_max - v_min) as usize + 1;
    if count > EQUIV_CELL_MAX {
        return None;
    }
    let mut out = Vec::with_capacity(count);
    let mut v = v_min;
    #[allow(clippy::while_float)]
    while v <= v_max {
        // Same rule as `cell_enum::cell_representative`: any point of
        // `[max(lo,v), min(hi,v+1))` induces the same value at every Trunc
        // output, so it decides the WHOLE cell.
        let rep = [v + 0.5, v, lo, hi]
            .into_iter()
            .find(|&c| c >= lo && c <= hi && c >= v && c < v + 1.0)?;
        out.push((v as i64, rep));
        v += 1.0;
    }
    Some(out)
}

/// S_high: `trunc(x) >= L`, where the ONNX `Slice`'s clamping makes the patch
/// index tensor empty, `ScatterND` writes nothing and the excised graph IS the
/// model. Checked by running EVERY integer cell of the sub-box through ORT on
/// both models and requiring bit-equal outputs — exhaustive because both
/// coordinates reach the graph only through `Cast`-to-integer, so both models
/// are constant on each cell. The verdict then comes from the unchanged cell
/// driver over the tightened box.
fn decide_high(
    graph: &ny_propagate::GraphNetwork,
    oracle: &mut Oracle,
    sub_spec: &VnnLibSpec,
    plan: &Plan,
    deadline: Option<Instant>,
) -> SubBoxVerdict {
    let (Some(split_cells), Some(other_cells)) = (
        cell_reps(sub_spec, plan.split_dim),
        cell_reps(sub_spec, plan.other_dim),
    ) else {
        eprintln!("#cctsdb-subbox: S_high declined — cell lattice not enumerable");
        return SubBoxVerdict::Declined;
    };
    let mut checked = 0usize;
    for &(_, rep_split) in &split_cells {
        for &(_, rep_other) in &other_cells {
            if deadline.is_some_and(|d| Instant::now() >= d) {
                eprintln!("#cctsdb-subbox: S_high declined — deadline during the ORT gate");
                return SubBoxVerdict::Declined;
            }
            let point = cell_point_f32(sub_spec, plan, rep_split, rep_other);
            let (Ok(a), Ok(b)) = (oracle.original.run(&point), oracle.excised.run(&point)) else {
                eprintln!("#cctsdb-subbox: S_high declined — an ORT forward failed");
                return SubBoxVerdict::Declined;
            };
            if !bit_equal(&a, &b) {
                eprintln!(
                    "#cctsdb-subbox: S_high declined — the excised model DIFFERS from the \
                     original at cell ({rep_split}, {rep_other}): {a:?} vs {b:?}"
                );
                return SubBoxVerdict::Declined;
            }
            checked += 1;
        }
    }
    eprintln!(
        "#cctsdb-subbox: S_high ORT equivalence gate PASSED — {checked} integer cell(s), \
         original vs excised bit-equal on every one"
    );

    let n = sub_spec.input_bounds.len();
    let Some(result) =
        crate::commands::beta_crown::cell_enumeration_verdict(graph, &[n], sub_spec, deadline)
    else {
        eprintln!("#cctsdb-subbox: S_high declined — cell enumeration did not apply/decide");
        return SubBoxVerdict::Declined;
    };
    if result.result == ny_propagate::BabVerificationStatus::Verified {
        eprintln!(
            "#cctsdb-subbox: S_high REFUTED by cell enumeration — {} domains explored, \
             {} verified, {:.2}s",
            result.domains_explored,
            result.domains_verified,
            result.time_elapsed.as_secs_f64()
        );
        SubBoxVerdict::Refuted
    } else {
        eprintln!(
            "#cctsdb-subbox: S_high declined — cell enumeration returned {:?}",
            result.result
        );
        SubBoxVerdict::Declined
    }
}

/// S_low: `trunc(x) < L`, where the patch DOES apply.
///
/// The patch is a 0/1 mask multiplied elementwise into a contiguous window of
/// the flat network input, and the detector proved that mask is a function of
/// the two integer coordinates ALONE. So on one integer cell the patched model
/// is exactly the EXCISED model run on the declared input with the masked
/// coordinates set to zero — a change to the SPEC's point bounds, not to the
/// graph. That is what lets all 3 x-cells share ONE graph build.
///
/// The mask itself is READ from the model by ORT (`Oracle::mask_probe`), never
/// reconstructed, and each cell's transformation is then confirmed against the
/// ORIGINAL model by a bit-equality check. Per cell the whole input is a point,
/// so the sound f64 forward decides it outright.
fn decide_low(
    graph: &ny_propagate::GraphNetwork,
    oracle: &mut Oracle,
    sub_spec: &VnnLibSpec,
    plan: &Plan,
    deadline: Option<Instant>,
) -> SubBoxVerdict {
    let (Some(split_cells), Some(other_cells)) = (
        cell_reps(sub_spec, plan.split_dim),
        cell_reps(sub_spec, plan.other_dim),
    ) else {
        eprintln!("#cctsdb-subbox: S_low declined — cell lattice not enumerable");
        return SubBoxVerdict::Declined;
    };
    let n = sub_spec.input_bounds.len();
    let mut decided = 0usize;
    let started = Instant::now();
    for &(vx, rep_split) in &split_cells {
        for &(vy, rep_other) in &other_cells {
            if deadline.is_some_and(|d| Instant::now() >= d) {
                eprintln!("#cctsdb-subbox: S_low declined — deadline after {decided} cell(s)");
                return SubBoxVerdict::Declined;
            }
            let point = cell_point_f32(sub_spec, plan, rep_split, rep_other);
            let Ok(mask) = oracle.mask_probe.run(&point) else {
                eprintln!("#cctsdb-subbox: S_low declined — the mask probe forward failed");
                return SubBoxVerdict::Declined;
            };
            if mask.len() != plan.mask_len {
                eprintln!(
                    "#cctsdb-subbox: S_low declined — mask has {} element(s), expected {}",
                    mask.len(),
                    plan.mask_len
                );
                return SubBoxVerdict::Declined;
            }
            // Only a 0/1 mask is expressible as "zero these declared inputs".
            // Any other multiplier and the driver has no argument: decline.
            if mask.iter().any(|&m| m != 0.0 && m != 1.0) {
                eprintln!(
                    "#cctsdb-subbox: S_low declined — the mask at cell ({vx}, {vy}) is not 0/1"
                );
                return SubBoxVerdict::Declined;
            }

            // The transformed instance: same declared box, masked coordinates
            // pinned to zero, both coordinates pinned to this cell.
            let mut cell_spec = sub_spec.clone();
            for (i, &m) in mask.iter().enumerate() {
                if m == 0.0 {
                    cell_spec.input_bounds[plan.image_offset + i] = (0.0, 0.0);
                }
            }
            cell_spec.input_bounds[plan.split_dim] = (rep_split, rep_split);
            cell_spec.input_bounds[plan.other_dim] = (rep_other, rep_other);

            // EXECUTION gate: the excised model on the transformed point must
            // reproduce the ORIGINAL model on this cell, bit for bit.
            let transformed: Vec<f32> = cell_spec
                .input_bounds
                .iter()
                .map(|&(lo, _)| lo as f32)
                .collect();
            let (Ok(a), Ok(b)) = (
                oracle.original.run(&point),
                oracle.excised.run(&transformed),
            ) else {
                eprintln!("#cctsdb-subbox: S_low declined — an ORT forward failed");
                return SubBoxVerdict::Declined;
            };
            if !bit_equal(&a, &b) {
                eprintln!(
                    "#cctsdb-subbox: S_low declined — the transformed instance DIFFERS from the \
                     original at cell ({vx}, {vy}): {a:?} vs {b:?}"
                );
                return SubBoxVerdict::Declined;
            }

            // Sound f64 decision on the fully pinned point.
            let values: Vec<f64> = cell_spec.input_bounds.iter().map(|&(lo, _)| lo).collect();
            match crate::commands::beta_crown::f64_point_definitely_safe(
                graph,
                &[n],
                &values,
                &cell_spec,
            ) {
                Some(true) => decided += 1,
                Some(false) => {
                    eprintln!(
                        "#cctsdb-subbox: S_low declined — cell ({vx}, {vy}) is not definitely \
                         safe (the row may well be `sat`; this driver never mints one)"
                    );
                    return SubBoxVerdict::Declined;
                }
                None => {
                    eprintln!(
                        "#cctsdb-subbox: S_low declined — the f64 forward did not decide cell \
                         ({vx}, {vy})"
                    );
                    return SubBoxVerdict::Declined;
                }
            }
        }
    }
    eprintln!(
        "#cctsdb-subbox: S_low REFUTED — {decided} integer cell(s), each ORT-confirmed against \
         the original and definitely safe under the sound f64 forward, {:.2}s",
        started.elapsed().as_secs_f64()
    );
    SubBoxVerdict::Refuted
}

fn bit_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

/// Indexed read-only view of the graph plus a memoising shape/value evaluator.
struct Facts<'g> {
    graph: &'g GraphProto,
    /// tensor name -> producing node index
    producer: HashMap<&'g str, usize>,
    /// tensor name -> consuming node indices
    consumers: HashMap<&'g str, Vec<usize>>,
    /// initializer name -> tensor
    init: HashMap<&'g str, &'g TensorProto>,
    /// the single activation input's name
    input_name: &'g str,
}

impl<'g> Facts<'g> {
    fn new(graph: &'g GraphProto) -> Option<Self> {
        let init: HashMap<&str, &TensorProto> = graph
            .initializer
            .iter()
            .map(|t| (t.name.as_str(), t))
            .collect();
        // Exactly one activation input (initializers may also be listed).
        let activations: Vec<&str> = graph
            .input
            .iter()
            .map(|v| v.name.as_str())
            .filter(|name| !init.contains_key(name))
            .collect();
        if activations.len() != 1 {
            return None;
        }
        let mut producer = HashMap::new();
        let mut consumers: HashMap<&str, Vec<usize>> = HashMap::new();
        for (idx, node) in graph.node.iter().enumerate() {
            for out in &node.output {
                if producer.insert(out.as_str(), idx).is_some() {
                    // Two producers for one tensor: not a well-formed graph.
                    return None;
                }
            }
            for inp in &node.input {
                consumers.entry(inp.as_str()).or_default().push(idx);
            }
        }
        Some(Self {
            graph,
            producer,
            consumers,
            init,
            input_name: activations[0],
        })
    }

    fn node(&self, idx: usize) -> &'g NodeProto {
        &self.graph.node[idx]
    }

    fn producer_of(&self, name: &str) -> Option<&'g NodeProto> {
        self.producer.get(name).map(|&i| self.node(i))
    }

    fn consumers_of(&self, name: &str) -> &[usize] {
        self.consumers
            .get(name)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// A node attribute by name.
    fn attr(node: &'g NodeProto, name: &str) -> Option<&'g ny_onnx::onnx_proto::AttributeProto> {
        node.attribute.iter().find(|a| a.name == name)
    }

    /// The integer payload of a tensor that is an initializer or a `Constant`
    /// node's `value`. `None` for anything non-integral or absent.
    fn const_i64s(&self, name: &str) -> Option<Vec<i64>> {
        let tensor = if let Some(t) = self.init.get(name) {
            *t
        } else {
            let node = self.producer_of(name)?;
            if node.op_type != "Constant" {
                return None;
            }
            Self::attr(node, "value")?.t.as_ref()?
        };
        tensor_i64s(tensor)
    }

    /// The `f32` contents of a statically-known tensor — an initializer or a
    /// `Constant` node. The `f32` twin of [`Self::const_i64s`], used to verify
    /// that a scattered-into tensor is literally all ones.
    fn const_f32s(&self, name: &str) -> Option<Vec<f32>> {
        let tensor = if let Some(t) = self.init.get(name) {
            *t
        } else {
            let node = self.producer_of(name)?;
            if node.op_type != "Constant" {
                return None;
            }
            Self::attr(node, "value")?.t.as_ref()?
        };
        tensor_f32s(tensor)
    }

    /// Static shape of a tensor, per-dimension, `None` for a dim that is not
    /// statically resolvable. Returns `None` for a tensor whose RANK is not
    /// resolvable.
    fn static_shape(&self, name: &str, depth: usize) -> Option<Vec<Option<i64>>> {
        if depth > EVAL_DEPTH_MAX {
            return None;
        }
        if name == self.input_name {
            let vi = self.graph.input.iter().find(|v| v.name == name)?;
            let shape = &vi.r#type.as_ref()?.tensor_type.as_ref()?.shape.as_ref()?;
            return Some(
                shape
                    .dim
                    .iter()
                    .map(|d| match &d.value {
                        Some(
                            ny_onnx::onnx_proto::tensor_shape_proto::dimension::Value::DimValue(v),
                        ) => Some(*v),
                        _ => None,
                    })
                    .collect(),
            );
        }
        if let Some(t) = self.init.get(name) {
            return Some(t.dims.iter().map(|&d| Some(d)).collect());
        }
        let node = self.producer_of(name)?;
        match node.op_type.as_str() {
            "Constant" => {
                let t = Self::attr(node, "value")?.t.as_ref()?;
                Some(t.dims.iter().map(|&d| Some(d)).collect())
            }
            "ConstantOfShape" => {
                let dims = self.eval_i64s(&node.input[0], depth + 1)?;
                Some(dims.into_iter().map(Some).collect())
            }
            "Range" => {
                let n = self.range_len(node, depth + 1)?;
                Some(vec![Some(n)])
            }
            "Reshape" => {
                let target = self.eval_i64s(node.input.get(1)?, depth + 1)?;
                let src = self.static_shape(&node.input[0], depth + 1);
                let mut out: Vec<Option<i64>> = Vec::with_capacity(target.len());
                let mut hole = None;
                for (axis, &d) in target.iter().enumerate() {
                    match d {
                        -1 => {
                            hole = Some(axis);
                            out.push(None);
                        }
                        0 => out.push(src.as_ref().and_then(|s| s.get(axis).copied()).flatten()),
                        d if d > 0 => out.push(Some(d)),
                        _ => return None,
                    }
                }
                if let Some(axis) = hole {
                    // Resolve the single -1 from the source element count.
                    let src = src?;
                    let total: i64 = src
                        .iter()
                        .copied()
                        .collect::<Option<Vec<i64>>>()?
                        .iter()
                        .product();
                    let known: i64 = out
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| *i != axis)
                        .map(|(_, d)| *d)
                        .collect::<Option<Vec<i64>>>()?
                        .iter()
                        .product();
                    if known <= 0 || total % known != 0 {
                        return None;
                    }
                    out[axis] = Some(total / known);
                }
                Some(out)
            }
            "Unsqueeze" => {
                let src = self.static_shape(&node.input[0], depth + 1)?;
                let axes = match Self::attr(node, "axes") {
                    Some(a) => a.ints.clone(),
                    None => self.eval_i64s(node.input.get(1)?, depth + 1)?,
                };
                let rank = src.len() as i64 + axes.len() as i64;
                let mut out: Vec<Option<i64>> = Vec::new();
                let mut norm: Vec<i64> = axes
                    .iter()
                    .map(|&a| if a < 0 { a + rank } else { a })
                    .collect();
                norm.sort_unstable();
                let mut it = src.into_iter();
                for axis in 0..rank {
                    if norm.binary_search(&axis).is_ok() {
                        out.push(Some(1));
                    } else {
                        out.push(it.next()?);
                    }
                }
                Some(out)
            }
            // ScatterND writes INTO `data` and returns a tensor of exactly
            // `data`'s shape, whatever the indices are.
            "ScatterND" | "Cast" | "Identity" => self.static_shape(&node.input[0], depth + 1),
            "Slice" => {
                // Shape is preserved on every axis this Slice does NOT cut;
                // the cut axis is left unresolved (that is exactly the
                // clamping-dependent extent this module reasons about).
                let mut src = self.static_shape(&node.input[0], depth + 1)?;
                let axes = self.eval_i64s(node.input.get(3)?, depth + 1)?;
                let rank = src.len() as i64;
                for a in axes {
                    let a = if a < 0 { a + rank } else { a };
                    let a = usize::try_from(a).ok()?;
                    *src.get_mut(a)? = None;
                }
                Some(src)
            }
            "Shape" => {
                let src = self.static_shape(&node.input[0], depth + 1)?;
                Some(vec![Some(src.len() as i64)])
            }
            _ => None,
        }
    }

    fn range_len(&self, node: &'g NodeProto, depth: usize) -> Option<i64> {
        let start = *self.eval_i64s(&node.input[0], depth)?.first()?;
        let limit = *self.eval_i64s(node.input.get(1)?, depth)?.first()?;
        let delta = *self.eval_i64s(node.input.get(2)?, depth)?.first()?;
        if delta <= 0 {
            return None;
        }
        Some(((limit - start).max(0) + delta - 1) / delta)
    }

    /// Evaluate a shape-cone integer tensor.
    fn eval_i64s(&self, name: &str, depth: usize) -> Option<Vec<i64>> {
        if depth > EVAL_DEPTH_MAX {
            return None;
        }
        if let Some(v) = self.const_i64s(name) {
            return (v.len() <= EVAL_LEN_MAX).then_some(v);
        }
        let node = self.producer_of(name)?;
        match node.op_type.as_str() {
            "Cast" => self.eval_i64s(&node.input[0], depth + 1),
            "Shape" => {
                let shape = self.static_shape(&node.input[0], depth + 1)?;
                shape.into_iter().collect()
            }
            "Range" => {
                let start = *self.eval_i64s(&node.input[0], depth + 1)?.first()?;
                let delta = *self.eval_i64s(node.input.get(2)?, depth + 1)?.first()?;
                let n = self.range_len(node, depth + 1)?;
                let n = usize::try_from(n).ok()?;
                if n > EVAL_LEN_MAX {
                    return None;
                }
                Some((0..n).map(|i| start + delta * i as i64).collect())
            }
            "Gather" => {
                let data = self.eval_i64s(&node.input[0], depth + 1)?;
                let idx = self.eval_i64s(node.input.get(1)?, depth + 1)?;
                let axis = Self::attr(node, "axis").map(|a| a.i_value()).unwrap_or(0);
                if axis != 0 {
                    return None;
                }
                idx.into_iter()
                    .map(|i| {
                        let i = if i < 0 { i + data.len() as i64 } else { i };
                        usize::try_from(i).ok().and_then(|i| data.get(i).copied())
                    })
                    .collect()
            }
            "Concat" => {
                let axis = Self::attr(node, "axis").map(|a| a.i_value()).unwrap_or(0);
                if axis != 0 {
                    return None;
                }
                let mut out = Vec::new();
                for inp in &node.input {
                    out.extend(self.eval_i64s(inp, depth + 1)?);
                    if out.len() > EVAL_LEN_MAX {
                        return None;
                    }
                }
                Some(out)
            }
            "Add" | "Sub" | "Mul" => {
                let a = self.eval_i64s(&node.input[0], depth + 1)?;
                let b = self.eval_i64s(node.input.get(1)?, depth + 1)?;
                let n = a.len().max(b.len());
                if (a.len() != n && a.len() != 1) || (b.len() != n && b.len() != 1) {
                    return None;
                }
                (0..n)
                    .map(|i| {
                        let x = a[if a.len() == 1 { 0 } else { i }];
                        let y = b[if b.len() == 1 { 0 } else { i }];
                        match node.op_type.as_str() {
                            "Add" => x.checked_add(y),
                            "Sub" => x.checked_sub(y),
                            _ => x.checked_mul(y),
                        }
                    })
                    .collect()
            }
            _ => None,
        }
    }

    /// If `name` is a pure reshape of a CONTIGUOUS, unit-step, static window
    /// of the flat network input, return `(start, len)` in flat-input
    /// coordinates. `None` for anything else — including any window whose
    /// bounds are not read from constants.
    fn contiguous_input_window(&self, name: &str, depth: usize) -> Option<(i64, i64)> {
        if depth > EVAL_DEPTH_MAX {
            return None;
        }
        let node = self.producer_of(name)?;
        match node.op_type.as_str() {
            // Reshape/Identity/Cast preserve the flat element order, so the
            // window is unchanged. (Cast is included only because a no-op cast
            // is common in exporters; it does not move elements.)
            "Reshape" | "Identity" => self.contiguous_input_window(&node.input[0], depth + 1),
            "Slice" => {
                if node.input.first()?.as_str() != self.input_name {
                    return None;
                }
                let starts = self.const_i64s(node.input.get(1)?)?;
                let ends = self.const_i64s(node.input.get(2)?)?;
                let axes = match node.input.get(3) {
                    Some(a) => self.const_i64s(a)?,
                    None => vec![0],
                };
                let steps = match node.input.get(4) {
                    Some(st) => self.const_i64s(st)?,
                    None => vec![1],
                };
                // The input is flat (rank 1, checked by the caller), so the
                // only admissible cut is a unit-step window on axis 0.
                if starts.len() != 1 || ends.len() != 1 || axes != [0] || steps != [1] {
                    return None;
                }
                let input_len = *self
                    .static_shape(self.input_name, depth + 1)?
                    .first()?
                    .as_ref()?;
                // ONNX Slice counts a NEGATIVE start/end from the end of the
                // axis (`v + L`), and clamps into `[0, L]` only afterwards.
                // Clamping a negative value straight to 0 (as an earlier version did)
                // silently rewrites "two before the end" as "the very beginning",
                // handing back a window that is both wrong and
                // WIDER than the real one. Here that failed safe (the window
                // length then disagrees with the mask, and the per-cell ORT gate
                // stands behind that), but a wrong window computed on purpose is
                // not something to leave in place because the net below it holds.
                // `saturating_add` because `v + L` on an i64::MIN sentinel would
                // otherwise overflow.
                let resolve = |v: i64| -> i64 {
                    let shifted = if v < 0 {
                        v.saturating_add(input_len)
                    } else {
                        v
                    };
                    shifted.clamp(0, input_len)
                };
                let start = resolve(starts[0]);
                let end = resolve(ends[0]);
                if end < start {
                    return None;
                }
                Some((start, end - start))
            }
            _ => None,
        }
    }

    /// Backward-reachable NODE set from `roots`, CUTTING at `Shape` nodes
    /// (a `Shape` output is a function of its input's shape, never its values),
    /// i.e. the set of nodes whose VALUES the roots can depend on.
    fn value_cone(&self, roots: &[&str]) -> HashSet<usize> {
        let mut seen: HashSet<usize> = HashSet::new();
        let mut stack: Vec<&str> = roots.to_vec();
        while let Some(name) = stack.pop() {
            let Some(&idx) = self.producer.get(name) else {
                continue;
            };
            if !seen.insert(idx) {
                continue;
            }
            // A `Shape` output is a function of its input's SHAPE, never of
            // its values, so the value dependence stops here.
            if self.node(idx).op_type == "Shape" {
                continue;
            }
            stack.extend(self.node(idx).input.iter().map(String::as_str));
        }
        seen
    }
}

fn tensor_i64s(t: &TensorProto) -> Option<Vec<i64>> {
    // 6 = INT32, 7 = INT64, 9 = BOOL, 11 = DOUBLE (not accepted here).
    match t.data_type {
        7 => {
            if !t.int64_data.is_empty() {
                Some(t.int64_data.clone())
            } else if !t.raw_data.is_empty() {
                if !t.raw_data.len().is_multiple_of(8) {
                    return None;
                }
                Some(
                    t.raw_data
                        .as_chunks::<8>()
                        .0
                        .iter()
                        .map(|c| i64::from_le_bytes(*c))
                        .collect(),
                )
            } else {
                Some(Vec::new())
            }
        }
        6 => {
            if !t.int32_data.is_empty() {
                Some(t.int32_data.iter().map(|&v| i64::from(v)).collect())
            } else if !t.raw_data.is_empty() {
                if !t.raw_data.len().is_multiple_of(4) {
                    return None;
                }
                Some(
                    t.raw_data
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|c| i64::from(i32::from_le_bytes(*c)))
                        .collect(),
                )
            } else {
                Some(Vec::new())
            }
        }
        _ => None,
    }
}

fn tensor_f32s(t: &TensorProto) -> Option<Vec<f32>> {
    if t.data_type != 1 {
        return None;
    }
    if !t.float_data.is_empty() {
        Some(t.float_data.clone())
    } else if !t.raw_data.is_empty() {
        if !t.raw_data.len().is_multiple_of(4) {
            return None;
        }
        Some(
            t.raw_data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect(),
        )
    } else {
        Some(Vec::new())
    }
}

/// The guarded detector. Every condition is CHECKED; any doubt returns `None`.
/// The spec-side precondition: EXACTLY two free dimensions, both finite,
/// non-negative (so `trunc == floor` and the integer cells are `[v, v+1)`) and
/// integer-ranged (so the split point can land on a cell boundary rather than
/// inside a cell). Every other declared dimension must be a finite point.
///
/// Returns the free dims in ascending order, or `None`.
fn free_integer_dims(spec: &VnnLibSpec) -> Option<Vec<usize>> {
    let mut free_dims = Vec::new();
    for (idx, &(lo, hi)) in spec.input_bounds.iter().enumerate() {
        if !(lo.is_finite() && hi.is_finite() && lo <= hi) {
            return None;
        }
        if lo < hi {
            if lo < 0.0 || lo.fract() != 0.0 || hi.fract() != 0.0 {
                return None;
            }
            free_dims.push(idx);
        }
    }
    // EXACTLY two free dims: one patch coordinate per axis. Anything else is
    // a different spec shape and this driver has no argument for it.
    (free_dims.len() == 2).then_some(free_dims)
}

/// Fail-closed exit from the detector, tagged with its source line so a
/// future decline can be located without re-deriving the whole precondition
/// list. Deliberately quiet: this function is consulted for EVERY model that
/// fails to load, in every benchmark, and almost all of them are not
/// patch-attack graphs at all.
fn no<T>(line: u32) -> Option<T> {
    tracing::debug!("#cctsdb-subbox: detector declined at cctsdb_subbox.rs:{line}");
    None
}

fn detect(model: &ModelProto, spec: &VnnLibSpec) -> Option<Plan> {
    // ---- spec shape -------------------------------------------------------
    if spec.dual_network.is_some() {
        return no(line!());
    }
    if spec
        .per_clause_input_bounds
        .iter()
        .any(|bounds| !bounds.is_empty())
    {
        return no(line!());
    }
    if spec.output_constraints.is_empty() && spec.output_constraint_clauses.is_empty() {
        return no(line!());
    }
    let mut free_dims = free_integer_dims(spec)?;

    // ---- graph ------------------------------------------------------------
    let graph = model.graph.as_ref()?;
    let facts = Facts::new(graph)?;
    let n = spec.input_bounds.len();
    let input_shape = facts.static_shape(facts.input_name, 0)?;
    let declared: i64 = input_shape
        .iter()
        .copied()
        .collect::<Option<Vec<i64>>>()?
        .iter()
        .product();
    if declared != i64::try_from(n).ok()? {
        return no(line!());
    }

    // Every consumer of the network input is either a fully static Slice (the
    // image) or one of the two coordinate Gathers. That pins the ONLY way a
    // free dim can reach the graph, which is what makes the per-cell
    // piecewise-constant argument (and hence the exhaustive equivalence gate)
    // valid.
    let mut coord_cast: HashMap<usize, String> = HashMap::new();
    let mut coord_gathers: HashSet<usize> = HashSet::new();
    for &ci in facts.consumers_of(facts.input_name) {
        let node = facts.node(ci);
        match node.op_type.as_str() {
            "Slice" => {
                // All of starts/ends/axes/steps constant => value-independent
                // of the free dims.
                for arg in node.input.iter().skip(1) {
                    facts.const_i64s(arg)?;
                }
            }
            "Gather" => {
                let idx = facts.const_i64s(node.input.get(1)?)?;
                if idx.len() != 1 {
                    return no(line!());
                }
                let dim = usize::try_from(idx[0]).ok()?;
                if !free_dims.contains(&dim) {
                    return no(line!());
                }
                // The gathered scalar must be consumed by exactly one Cast to
                // an integer type — that Cast IS the trunc gate.
                let consumers = facts.consumers_of(&node.output[0]);
                if consumers.len() != 1 {
                    return no(line!());
                }
                let cast = facts.node(consumers[0]);
                if cast.op_type != "Cast" {
                    return no(line!());
                }
                // 6 = INT32, 7 = INT64.
                let to = Facts::attr(cast, "to")?.i_value();
                if to != 6 && to != 7 {
                    return no(line!());
                }
                if coord_cast.insert(dim, cast.output[0].clone()).is_some() {
                    return no(line!());
                }
                coord_gathers.insert(ci);
            }
            _ => return None,
        }
    }
    if coord_cast.len() != 2 {
        return no(line!());
    }

    // ---- the runtime-start selections ------------------------------------
    let mut runtime_slices: Vec<(usize, i64)> = Vec::new();
    // dim -> distinct axis lengths seen for that coordinate's selections
    let mut per_dim_lengths: HashMap<usize, Vec<i64>> = HashMap::new();
    for (idx, node) in graph.node.iter().enumerate() {
        if node.op_type != "Slice" {
            continue;
        }
        if node.input.len() != 5 {
            // A Slice without explicit axes/steps: fall closed rather than
            // guess the defaults.
            if facts.const_i64s(node.input.get(1)?).is_none() {
                return no(line!());
            }
            continue;
        }
        if facts.const_i64s(&node.input[1]).is_some() {
            continue; // static start: untouched by both rewrites
        }
        // starts = Unsqueeze(cast_d) with axes [0]
        let starts = facts.producer_of(&node.input[1])?;
        if starts.op_type != "Unsqueeze" || !unsqueeze_axes_is(&facts, starts, 0) {
            return no(line!());
        }
        let coord = starts.input.first()?.as_str();
        let dim = *coord_cast
            .iter()
            .find(|(_, cast_out)| cast_out.as_str() == coord)
            .map(|(dim, _)| dim)?;
        // ends = Unsqueeze(Add(cast_d, k)) with axes [0], any k >= 1
        let ends = facts.producer_of(&node.input[2])?;
        if ends.op_type != "Unsqueeze" || !unsqueeze_axes_is(&facts, ends, 0) {
            return no(line!());
        }
        let add = facts.producer_of(ends.input.first()?)?;
        if add.op_type != "Add" {
            return no(line!());
        }
        let (a, b) = (add.input.first()?.as_str(), add.input.get(1)?.as_str());
        let offset = if a == coord {
            facts.const_i64s(b)?
        } else if b == coord {
            facts.const_i64s(a)?
        } else {
            return no(line!());
        };
        // The patch side `k` is READ, not assumed to be 1. The split-at-`L`
        // argument this module rests on holds for every `k >= 1`: the extent is
        // `clamp(d+k,0,L) - clamp(d,0,L)`, which for `d >= L` is `L - L = 0`
        // (nothing is written, so the ScatterND is a no-op and the excision is
        // valid) and for `0 <= d < L` is `min(d+k,L) - d >= 1` (something is
        // written). So `extent == 0` iff `d >= L` INDEPENDENTLY of `k`, and the
        // split point stays exactly `L`. What `k > 1` does change is that the
        // extent VARIES across the low region (k=3, L=3 gives 3, 2, 1 at d=0,1,2
        // instead of a constant 1) — which costs nothing here, because `decide_low`
        // reads the mask from ORT and decides SEPARATELY FOR EVERY INTEGER CELL,
        // so a per-cell extent is simply observed rather than assumed. Both lanes
        // then bit-compare the rewritten model against the ORIGINAL on every cell
        // before any verdict is claimed, so a `k` this reasoning did not
        // anticipate DECLINES at that gate rather than returning a verdict.
        if offset.len() != 1 || offset[0] < 1 {
            return no(line!());
        }
        // axes = [a], steps = [1]
        let axes = facts.const_i64s(&node.input[3])?;
        let steps = facts.const_i64s(&node.input[4])?;
        if axes.len() != 1 || steps.len() != 1 || steps[0] != 1 {
            return no(line!());
        }
        let data_shape = facts.static_shape(&node.input[0], 0)?;
        let rank = data_shape.len() as i64;
        let axis = if axes[0] < 0 { axes[0] + rank } else { axes[0] };
        let axis_usize = usize::try_from(axis).ok()?;
        let len = (*data_shape.get(axis_usize)?)?;
        if len <= 0 {
            return no(line!());
        }
        runtime_slices.push((idx, axis));
        per_dim_lengths.entry(dim).or_default().push(len);
    }
    if runtime_slices.is_empty() {
        return no(line!());
    }

    // ---- which coordinate clamps, and where ------------------------------
    let mut split: Option<(usize, i64)> = None;
    for (&dim, lengths) in &per_dim_lengths {
        let hi = spec.input_bounds[dim].1;
        let clamps: Vec<i64> = lengths
            .iter()
            .copied()
            .filter(|&len| hi >= len as f64)
            .collect();
        if clamps.is_empty() {
            continue;
        }
        // Every selection of the clamping coordinate must share ONE axis
        // length, or there is more than one shape regime and a two-way split
        // is not covering.
        if lengths.iter().any(|&len| len != lengths[0]) {
            return no(line!());
        }
        if split.is_some() {
            return no(line!()); // both coordinates clamp: not a two-sub-box problem
        }
        split = Some((dim, lengths[0]));
    }
    let (split_dim, split_len) = split?;
    let (split_lo, split_hi) = spec.input_bounds[split_dim];
    if split_lo != 0.0 || split_len < 1 || split_hi < split_len as f64 {
        return no(line!());
    }

    // ---- the ScatterND excision's structural preconditions ----------------
    let scatternds: Vec<usize> = graph
        .node
        .iter()
        .enumerate()
        .filter(|(_, n)| n.op_type == "ScatterND")
        .map(|(i, _)| i)
        .collect();
    if scatternds.len() != 1 {
        return no(line!());
    }
    let sn_idx = scatternds[0];
    let sn = facts.node(sn_idx);
    if sn.input.len() != 3 || sn.output.len() != 1 {
        return no(line!());
    }
    // The scattered-into tensor is the all-ones mask: ConstantOfShape with a
    // literal 1.0 value attribute. That is the fact the "the patch cone is a
    // no-op when nothing is written" claim stands on.
    // Two spellings of the same fact, and it is the FACT that is checked, not
    // the spelling: `ConstantOfShape` fills a shape with one literal value,
    // while a `Constant`/initializer carries the tensor densely. patch-1 uses
    // the former and patch-3 the latter for an identical all-ones mask, so
    // keying on the op type alone would decline a graph that satisfies the
    // precondition outright. What the excision argument needs is only that
    // every element written into is 1.0.
    let all_ones = match facts.producer_of(&sn.input[0]) {
        Some(node) if node.op_type == "ConstantOfShape" => Facts::attr(node, "value")
            .and_then(|a| a.t.as_ref())
            .and_then(tensor_f32s)
            .is_some_and(|v| v.len() == 1 && v[0] == 1.0),
        // A dense literal (`Constant` node or initializer): every element must
        // be exactly 1.0. An empty tensor is NOT all-ones for this purpose —
        // it would make the mask vacuous rather than neutral.
        _ => facts
            .const_f32s(&sn.input[0])
            .is_some_and(|v| !v.is_empty() && v.iter().all(|&m| m == 1.0)),
    };
    if !all_ones {
        return no(line!());
    }
    // The ScatterND output must not itself be a graph output (the excision
    // rewires consumers).
    if graph.output.iter().any(|o| o.name == sn.output[0]) {
        return no(line!());
    }
    // The index/update cone must depend on the network input ONLY through the
    // two coordinate casts — cutting at `Shape`, which reads no values. This
    // is what makes the index tensor a function of (trunc x, trunc y) alone,
    // and therefore what makes one representative per integer cell an
    // EXHAUSTIVE check of the rewrite.
    let cone = facts.value_cone(&[sn.input[1].as_str(), sn.input[2].as_str()]);
    for &ni in &cone {
        let node = facts.node(ni);
        if node.input.iter().any(|i| i == facts.input_name) && !coord_gathers.contains(&ni) {
            return no(line!());
        }
    }
    // ...and it must actually depend on the clamping selection, or the
    // "extent 0 empties the indices" argument does not apply to this graph.
    let clamping_nodes: Vec<usize> = runtime_slices
        .iter()
        .filter_map(|&(idx, _)| {
            let node = facts.node(idx);
            let starts = facts.producer_of(&node.input[1])?;
            let coord = starts.input.first()?.as_str();
            let dim = *coord_cast
                .iter()
                .find(|(_, cast_out)| cast_out.as_str() == coord)
                .map(|(dim, _)| dim)?;
            (dim == split_dim).then_some(idx)
        })
        .collect();
    if clamping_nodes.is_empty() || !clamping_nodes.iter().any(|ni| cone.contains(ni)) {
        return no(line!());
    }

    // ---- how the mask reaches the image ----------------------------------
    // The mask's ONLY consumer must be an elementwise `Mul` against a
    // contiguous window of the flat network input. That window is what lets a
    // per-cell mask be applied as a change to the SPEC's point bounds instead
    // of as a change to the graph: `image * mask` with a 0/1 mask is exactly
    // "zero these declared input coordinates".
    let mask_out = sn.output[0].clone();
    let mask_consumers = facts.consumers_of(&mask_out);
    if mask_consumers.len() != 1 {
        return no(line!());
    }
    let mul = facts.node(mask_consumers[0]);
    if mul.op_type != "Mul" || mul.input.len() != 2 {
        return no(line!());
    }
    let image = if mul.input[0] == mask_out {
        mul.input[1].as_str()
    } else if mul.input[1] == mask_out {
        mul.input[0].as_str()
    } else {
        return no(line!());
    };
    let mask_shape: Vec<i64> = match facts
        .static_shape(&mask_out, 0)
        .and_then(|s| s.into_iter().collect::<Option<Vec<i64>>>())
    {
        Some(v) => v,
        None => return no(line!()),
    };
    let image_shape: Vec<i64> = match facts
        .static_shape(image, 0)
        .and_then(|s| s.into_iter().collect::<Option<Vec<i64>>>())
    {
        Some(v) => v,
        None => return no(line!()),
    };
    // Elementwise, never broadcast: a broadcasting Mul would not be a
    // coordinate-for-coordinate scaling of the declared inputs.
    if mask_shape != image_shape || mask_shape.iter().any(|&d| d <= 0) {
        return no(line!());
    }
    let mask_len: i64 = mask_shape.iter().product();
    let Some((image_offset, window_len)) = facts.contiguous_input_window(image, 0) else {
        return no(line!());
    };
    if window_len != mask_len || image_offset < 0 {
        return no(line!());
    }
    let image_offset = usize::try_from(image_offset).ok()?;
    let mask_len = usize::try_from(mask_len).ok()?;
    if image_offset.checked_add(mask_len)? > n {
        return no(line!());
    }
    // The free coordinates must lie OUTSIDE the image window: the S_low route
    // re-applies the patch by zeroing declared input coordinates, and zeroing
    // one that is also a patch coordinate would silently rewrite the cell it
    // is meant to be evaluating.
    if free_dims
        .iter()
        .any(|&d| d >= image_offset && d < image_offset + mask_len)
    {
        return no(line!());
    }

    free_dims.sort_unstable();
    let other_dim = *free_dims.iter().find(|&&d| d != split_dim)?;
    Some(Plan {
        split_dim,
        split_len,
        split_hi,
        other_dim,
        scatternd: sn_idx,
        mask_out,
        image_offset,
        mask_len,
        free_dims,
    })
}

fn unsqueeze_axes_is(facts: &Facts<'_>, node: &NodeProto, axis: i64) -> bool {
    if let Some(a) = Facts::attr(node, "axes") {
        return a.ints == [axis];
    }
    match node.input.get(1).and_then(|n| facts.const_i64s(n)) {
        Some(v) => v == [axis],
        None => false,
    }
}

// ---------------------------------------------------------------------------
// Rewrites
// ---------------------------------------------------------------------------

/// S_high: `ScatterND(data, indices, updates)` with an EMPTY index tensor is
/// `data`, so the node is deleted and its consumers read `data` directly; the
/// now-unreachable patch cone is then dropped.
///
/// Justified on S_high because the split coordinate's selection has
/// `trunc(x) >= L`, so ONNX `Slice`'s clamping makes its extent
/// `clamp(x+1,0,L) - clamp(x,0,L) = 0`, which empties the index tensor the
/// detector proved is a function of that selection.
fn rewrite_excise_scatternd(model: &ModelProto, plan: &Plan) -> Option<ModelProto> {
    let mut out = model.clone();
    let graph = out.graph.as_mut()?;
    let sn = graph.node.get(plan.scatternd)?;
    let data = sn.input.first()?.clone();
    let produced = sn.output.first()?.clone();
    graph.node.remove(plan.scatternd);
    for node in &mut graph.node {
        for inp in &mut node.input {
            if *inp == produced {
                *inp = data.clone();
            }
        }
    }
    // Dead-node elimination from the graph outputs. Without it the loader
    // still walks the (now unreachable) patch cone and refuses at the very
    // `Expand` this whole module exists to route around — measured.
    let producer: HashMap<String, usize> = graph
        .node
        .iter()
        .enumerate()
        .flat_map(|(i, n)| n.output.iter().map(move |o| (o.clone(), i)))
        .collect();
    let mut live: HashSet<usize> = HashSet::new();
    let mut stack: Vec<String> = graph.output.iter().map(|o| o.name.clone()).collect();
    while let Some(name) = stack.pop() {
        let Some(&idx) = producer.get(&name) else {
            continue;
        };
        if !live.insert(idx) {
            continue;
        }
        stack.extend(graph.node[idx].input.iter().cloned());
    }
    let kept: Vec<NodeProto> = graph
        .node
        .iter()
        .enumerate()
        .filter(|(i, _)| live.contains(i))
        .map(|(_, n)| n.clone())
        .collect();
    graph.node = kept;
    Some(out)
}

// ---------------------------------------------------------------------------
// The execution cross-check
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests;
