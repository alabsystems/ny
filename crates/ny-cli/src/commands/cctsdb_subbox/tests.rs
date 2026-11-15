// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #cctsdb-subbox pins.
//!
//! The positives run against the real `cctsdb_yolo_2023` `patch-1.onnx`
//! (skipped when the gitignored benchmark checkout is absent — a worktree has
//! no `benchmarks/`). The negatives are the soundness argument: each one
//! perturbs exactly ONE structural fact the rewrites stand on and asserts the
//! detector declines.

use std::path::{Path, PathBuf};

use ny_onnx::onnx_proto::{ModelProto, NodeProto, TensorProto};
use ny_onnx::vnnlib::VnnLibSpec;
use prost::Message;

use super::*;

fn bench_dir() -> PathBuf {
    // `cargo test -p ny-cli` runs with CWD at the PACKAGE root, not the repo
    // root, and `benchmarks/` is gitignored and lives only in the main tree.
    // A relative path here silently skips every fixture test (the
    // matched-nothing trap wearing a different hat).
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmarks/vnncomp2025/benchmarks/cctsdb_yolo_2023")
}

fn load_model(name: &str) -> Option<ModelProto> {
    let path = bench_dir().join("onnx").join(name);
    if !path.is_file() {
        return None;
    }
    let bytes = std::fs::read(path).expect("read model");
    Some(ModelProto::decode(bytes.as_slice()).expect("decode model"))
}

fn patch1() -> Option<ModelProto> {
    load_model("patch-1.onnx")
}

fn spec_named(name: &str) -> Option<VnnLibSpec> {
    let path = bench_dir().join("vnnlib").join(name);
    if !path.is_file() {
        return None;
    }
    Some(ny_onnx::vnnlib::load_vnnlib(path).expect("parse cctsdb spec"))
}

fn patch1_spec() -> Option<VnnLibSpec> {
    spec_named("spec_onnx_patch-1_idx_01045_0.vnnlib")
}

fn fixture() -> Option<(ModelProto, VnnLibSpec)> {
    Some((patch1()?, patch1_spec()?))
}

/// The load-bearing structural facts, all READ from the shipped model: the
/// split coordinate is `X_12288` and its selection axis is 3 long.
#[test]
fn detect_reads_the_split_from_the_shipped_cctsdb_model() {
    let Some((model, spec)) = fixture() else {
        eprintln!("skipping: cctsdb benchmark checkout absent");
        return;
    };
    let plan = detect(&model, &spec).expect("cctsdb patch-1 qualifies");
    assert_eq!(plan.free_dims, vec![12288, 12289]);
    assert_eq!(plan.split_dim, 12288);
    assert_eq!(
        plan.split_len, 3,
        "the split point must be READ from Shape(mask)[1], not assumed"
    );
    assert_eq!(plan.split_hi, 62.0);
    assert_eq!(plan.mask_out, "363");
}

/// The two sub-boxes are covering and disjoint over the DECLARED box, on the
/// integer cells the driver actually enumerates.
#[test]
fn sub_boxes_cover_the_declared_box_and_do_not_overlap() {
    let Some((model, spec)) = fixture() else {
        eprintln!("skipping: cctsdb benchmark checkout absent");
        return;
    };
    let plan = detect(&model, &spec).expect("qualifies");
    let l = plan.split_len as f64;
    let low = tightened(&spec, plan.split_dim, 0.0, l - 1.0).expect("S_low box");
    let high = tightened(&spec, plan.split_dim, l, plan.split_hi).expect("S_high box");
    let (dlo, dhi) = spec.input_bounds[plan.split_dim];

    // Cell v of the driver covers the whole trunc-cell [v, v+1).
    let low_cells: Vec<i64> = (low.input_bounds[plan.split_dim].0 as i64
        ..=low.input_bounds[plan.split_dim].1 as i64)
        .collect();
    let high_cells: Vec<i64> = (high.input_bounds[plan.split_dim].0 as i64
        ..=high.input_bounds[plan.split_dim].1 as i64)
        .collect();
    assert_eq!(low_cells, vec![0, 1, 2]);
    assert_eq!(high_cells.first(), Some(&3));
    assert_eq!(high_cells.last(), Some(&62));
    assert!(
        low_cells.iter().all(|v| !high_cells.contains(v)),
        "sub-boxes must be disjoint"
    );
    // Every trunc-cell of the declared box is in exactly one half.
    for v in dlo as i64..=dhi as i64 {
        assert_eq!(
            usize::from(low_cells.contains(&v)) + usize::from(high_cells.contains(&v)),
            1,
            "cell {v} is not covered exactly once"
        );
    }
    // Every other dim is untouched.
    for (idx, &(lo, hi)) in spec.input_bounds.iter().enumerate() {
        if idx == plan.split_dim {
            continue;
        }
        assert_eq!(low.input_bounds[idx], (lo, hi));
        assert_eq!(high.input_bounds[idx], (lo, hi));
    }
}

/// A sub-box may never be WIDER than the declared box.
#[test]
fn tightened_refuses_a_box_outside_the_declared_one() {
    let mut spec = VnnLibSpec::new();
    spec.input_bounds = vec![(0.0, 5.0)];
    assert!(tightened(&spec, 0, 0.0, 6.0).is_none());
    assert!(tightened(&spec, 0, -1.0, 5.0).is_none());
    assert!(tightened(&spec, 0, f64::NAN, 5.0).is_none());
    assert!(tightened(&spec, 0, 4.0, 3.0).is_none());
    assert!(tightened(&spec, 1, 0.0, 1.0).is_none());
    assert_eq!(
        tightened(&spec, 0, 1.0, 2.0).expect("inside").input_bounds[0],
        (1.0, 2.0)
    );
}

/// The mask-to-input mapping is READ from the graph: the ScatterND result is
/// multiplied elementwise into the first 12288 flat inputs.
#[test]
fn mask_window_is_read_from_the_graphs_own_mul() {
    let Some((model, spec)) = fixture() else {
        eprintln!("skipping: cctsdb benchmark checkout absent");
        return;
    };
    let plan = detect(&model, &spec).expect("qualifies");
    assert_eq!(plan.image_offset, 0);
    assert_eq!(plan.mask_len, 3 * 64 * 64);
    assert_eq!(plan.other_dim, 12289);
    // The window must sit inside the declared input and must not overlap the
    // free coordinates, or zeroing a masked coordinate would overwrite one.
    assert!(plan.image_offset + plan.mask_len <= spec.input_bounds.len());
    assert!(plan.split_dim >= plan.image_offset + plan.mask_len);
    assert!(plan.other_dim >= plan.image_offset + plan.mask_len);
}

/// The mask probe model reports the ScatterND result and nothing else.
#[test]
fn mask_probe_model_outputs_the_scatternd_result() {
    let Some((model, spec)) = fixture() else {
        eprintln!("skipping: cctsdb benchmark checkout absent");
        return;
    };
    let plan = detect(&model, &spec).expect("qualifies");
    let probe = mask_probe_model(&model, &plan).expect("probe model");
    let graph = probe.graph.as_ref().unwrap();
    assert_eq!(graph.output.len(), 1);
    assert_eq!(graph.output[0].name, plan.mask_out);
    // Every other node survives: the probe must not change what is computed.
    assert_eq!(graph.node.len(), model.graph.as_ref().unwrap().node.len());
}

/// The S_high rewrite deletes the ScatterND, rewires its consumers to the
/// mask, and drops the unreachable patch cone.
#[test]
fn scatternd_excision_rewires_consumers_and_prunes_the_patch_cone() {
    let Some((model, spec)) = fixture() else {
        eprintln!("skipping: cctsdb benchmark checkout absent");
        return;
    };
    let plan = detect(&model, &spec).expect("qualifies");
    let before = model.graph.as_ref().unwrap();
    let sn = &before.node[plan.scatternd];
    let (data, produced) = (sn.input[0].clone(), sn.output[0].clone());
    let consumers: Vec<String> = before
        .node
        .iter()
        .filter(|n| n.input.contains(&produced))
        .map(|n| n.name.clone())
        .collect();
    assert_ne!((consumers).len(), 0);

    let high = rewrite_excise_scatternd(&model, &plan).expect("rewrite");
    let after = high.graph.as_ref().unwrap();
    assert!(after.node.iter().all(|n| n.op_type != "ScatterND"));
    assert!(
        after.node.iter().all(|n| !n.input.contains(&produced)),
        "no node may still read the deleted ScatterND output"
    );
    for name in &consumers {
        let node = after
            .node
            .iter()
            .find(|n| &n.name == name)
            .expect("consumer survives");
        assert!(node.input.contains(&data), "consumer now reads the mask");
    }
    assert!(
        after.node.len() < before.node.len() - 1,
        "the unreachable patch cone must prune (before {} after {})",
        before.node.len(),
        after.node.len()
    );
    // Every surviving node's inputs still resolve.
    let produced_names: HashSet<&str> = after
        .node
        .iter()
        .flat_map(|n| n.output.iter().map(String::as_str))
        .collect();
    let inits: HashSet<&str> = after.initializer.iter().map(|t| t.name.as_str()).collect();
    let graph_inputs: HashSet<&str> = after.input.iter().map(|v| v.name.as_str()).collect();
    for node in &after.node {
        for inp in &node.input {
            assert!(
                produced_names.contains(inp.as_str())
                    || inits.contains(inp.as_str())
                    || graph_inputs.contains(inp.as_str())
                    || inp.is_empty(),
                "node {} reads unproduced tensor {inp}",
                node.name
            );
        }
    }
    for out in &after.output {
        assert!(produced_names.contains(out.name.as_str()));
    }
}

// -- negatives: each removes ONE fact a rewrite stands on ------------------

fn assert_declines(mutate: impl FnOnce(&mut ModelProto), why: &str) {
    let Some((mut model, spec)) = fixture() else {
        eprintln!("skipping: cctsdb benchmark checkout absent");
        return;
    };
    assert!(
        detect(&model, &spec).is_some(),
        "control: unmutated qualifies"
    );
    mutate(&mut model);
    assert!(detect(&model, &spec).is_none(), "must decline: {why}");
}

#[test]
fn declines_when_the_mask_is_not_all_ones() {
    assert_declines(
        |model| {
            let graph = model.graph.as_mut().unwrap();
            let sn_data = graph
                .node
                .iter()
                .find(|n| n.op_type == "ScatterND")
                .unwrap()
                .input[0]
                .clone();
            let mask = graph
                .node
                .iter_mut()
                .find(|n| n.output.first() == Some(&sn_data))
                .unwrap();
            let value = mask
                .attribute
                .iter_mut()
                .find(|a| a.name == "value")
                .unwrap();
            value.t = Some(TensorProto {
                data_type: 1,
                dims: vec![1],
                float_data: vec![0.5],
                ..Default::default()
            });
        },
        "a mask that is not all-ones breaks the no-op argument",
    );
}

#[test]
fn declines_when_a_second_scatternd_exists() {
    assert_declines(
        |model| {
            let graph = model.graph.as_mut().unwrap();
            let sn = graph
                .node
                .iter()
                .find(|n| n.op_type == "ScatterND")
                .unwrap()
                .clone();
            let mut extra = sn;
            extra.name = "ScatterND_extra".to_string();
            extra.output = vec!["ny_extra_scatter_out".to_string()];
            graph.node.push(extra);
        },
        "two ScatterNDs mean the single-excision argument does not apply",
    );
}

#[test]
fn declines_when_a_runtime_slice_has_a_non_unit_step() {
    assert_declines(
        |model| {
            let graph = model.graph.as_mut().unwrap();
            // Slice_59 selects on the clamping axis; give it step 2 by
            // repointing its `steps` input at a fresh constant.
            let node = graph
                .node
                .iter_mut()
                .find(|n| n.name == "Slice_59")
                .expect("Slice_59");
            node.input[4] = "ny_step_two".to_string();
            let mut step = NodeProto {
                op_type: "Constant".to_string(),
                name: "ny_step_two_const".to_string(),
                output: vec!["ny_step_two".to_string()],
                ..Default::default()
            };
            step.attribute.push(ny_onnx::onnx_proto::AttributeProto {
                name: "value".to_string(),
                r#type: ny_onnx::onnx_proto::attribute_type::TENSOR,
                t: Some(TensorProto {
                    data_type: 7,
                    dims: vec![1],
                    int64_data: vec![2],
                    ..Default::default()
                }),
                ..Default::default()
            });
            graph.node.insert(0, step);
        },
        "step != 1 is not a single-slab selection, so Gather is not equivalent",
    );
}

/// `k >= 1` is the precondition, and `k = 0` is the boundary it draws.
///
/// This mutation used to set `k = 2` and assert a decline, on the reading that
/// only a single-slab patch is admissible. patch-3 refuted that: `k = 3` is a
/// real shipped model that the driver decides soundly, because the extent-zero
/// boundary sits at `L` for every `k >= 1` and the low region is decided
/// per-cell. See `detects_patch_3_whose_patch_is_wider_than_one_slab`.
///
/// `k = 0` is genuinely different, and stays refused: the extent is then
/// `clamp(d,0,L) - clamp(d,0,L) = 0` for EVERY `d`, so nothing is ever written,
/// the split at `L` separates nothing, and the "something is written below `L`"
/// half of the argument has no content. Declining is the honest answer for a
/// premise the module cannot state.
#[test]
fn declines_when_the_slice_end_offset_is_not_positive() {
    assert_declines(
        |model| {
            let graph = model.graph.as_mut().unwrap();
            let add = graph
                .node
                .iter_mut()
                .find(|n| n.name == "Add_28")
                .expect("Add_28 computes x+1");
            add.input[1] = "ny_zero".to_string();
            let mut zero = NodeProto {
                op_type: "Constant".to_string(),
                name: "ny_zero_const".to_string(),
                output: vec!["ny_zero".to_string()],
                ..Default::default()
            };
            zero.attribute.push(ny_onnx::onnx_proto::AttributeProto {
                name: "value".to_string(),
                r#type: ny_onnx::onnx_proto::attribute_type::TENSOR,
                t: Some(TensorProto {
                    data_type: 7,
                    int64_data: vec![0],
                    ..Default::default()
                }),
                ..Default::default()
            });
            graph.node.insert(0, zero);
        },
        "end = start + 0 is an empty selection at every coordinate",
    );
}

#[test]
fn declines_when_the_index_cone_reads_input_values() {
    assert_declines(
        |model| {
            let graph = model.graph.as_mut().unwrap();
            // Make the ScatterND updates depend on the raw network input
            // rather than only on shapes and the two coordinates.
            let input = graph.input[0].name.clone();
            let sn = graph
                .node
                .iter_mut()
                .find(|n| n.op_type == "ScatterND")
                .unwrap();
            sn.input[2] = input;
        },
        "an index/update cone that reads input values is not a function of \
         (trunc x, trunc y), so per-cell replay would not be exhaustive",
    );
}

#[test]
fn declines_on_a_spec_with_a_negative_free_dim() {
    let Some((model, mut spec)) = fixture() else {
        eprintln!("skipping: cctsdb benchmark checkout absent");
        return;
    };
    assert!(detect(&model, &spec).is_some());
    spec.input_bounds[12288] = (-1.0, 62.0);
    assert!(
        detect(&model, &spec).is_none(),
        "trunc != floor on a negative axis; the integer-cell argument fails"
    );
}

#[test]
fn declines_on_a_spec_with_a_non_integer_free_range() {
    let Some((model, mut spec)) = fixture() else {
        eprintln!("skipping: cctsdb benchmark checkout absent");
        return;
    };
    spec.input_bounds[12288] = (0.0, 62.5);
    assert!(
        detect(&model, &spec).is_none(),
        "a non-integral endpoint means the split point need not fall on a cell boundary"
    );
}

#[test]
fn declines_on_a_spec_with_more_than_two_free_dims() {
    let Some((model, mut spec)) = fixture() else {
        eprintln!("skipping: cctsdb benchmark checkout absent");
        return;
    };
    spec.input_bounds[0] = (0.0, 1.0);
    assert!(detect(&model, &spec).is_none());
}

#[test]
fn declines_when_the_split_coordinate_never_clamps() {
    let Some((model, mut spec)) = fixture() else {
        eprintln!("skipping: cctsdb benchmark checkout absent");
        return;
    };
    // x in [0, 2] never reaches the 3-long axis, so there is no second regime
    // to split off and this driver has nothing to contribute.
    spec.input_bounds[12288] = (0.0, 2.0);
    assert!(detect(&model, &spec).is_none());
}

#[test]
fn declines_on_per_clause_input_boxes_and_dual_networks() {
    let Some((model, mut clause)) = fixture() else {
        eprintln!("skipping: cctsdb benchmark checkout absent");
        return;
    };
    assert!(
        detect(&model, &clause).is_some(),
        "control: unmutated qualifies"
    );
    clause.per_clause_input_bounds = vec![[(0usize, (0.0, 1.0))].into_iter().collect()];
    assert!(detect(&model, &clause).is_none());
}

// -- the static evaluator ---------------------------------------------------

/// `L` is READ, not hardcoded: the same graph with a 5-channel mask reports 5.
#[test]
fn split_length_follows_the_models_own_mask_shape() {
    let Some((mut model, spec)) = fixture() else {
        eprintln!("skipping: cctsdb benchmark checkout absent");
        return;
    };
    assert_eq!(detect(&model, &spec).unwrap().split_len, 3);
    {
        let graph = model.graph.as_mut().unwrap();
        // Reshape_6's target shape defines the mask shape [1, 3, 64, 64].
        // Keep the ELEMENT COUNT (the image window is 12288 either way) and
        // keep the y axis longer than the declared y range, so the only thing
        // that changes is the x axis the split is read from.
        let reshape_shape = graph
            .node
            .iter()
            .find(|n| n.name == "Reshape_6")
            .expect("Reshape_6")
            .input[1]
            .clone();
        let konst = graph
            .node
            .iter_mut()
            .find(|n| n.output.first() == Some(&reshape_shape))
            .expect("its shape constant");
        let value = konst
            .attribute
            .iter_mut()
            .find(|a| a.name == "value")
            .unwrap();
        value.t = Some(TensorProto {
            data_type: 7,
            dims: vec![4],
            int64_data: vec![1, 6, 64, 32],
            ..Default::default()
        });
    }
    assert_eq!(
        detect(&model, &spec).expect("still qualifies").split_len,
        6,
        "the split point must track Shape(mask)[1]"
    );
}

/// The driver is inert on a model that loads: it never fires where ny already
/// has a graph.
#[test]
fn refutes_returns_false_for_a_model_that_builds() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmarks/vnncomp2025/benchmarks/acasxu_2023");
    let onnx = root.join("onnx/ACASXU_run2a_1_1_batch_2000.onnx");
    let vnnlib = root.join("vnnlib/prop_1.vnnlib");
    if !onnx.is_file() || !vnnlib.is_file() {
        eprintln!("skipping: acasxu fixture absent");
        return;
    }
    assert!(!cctsdb_subbox_refutes(&onnx, &vnnlib, None));
}

/// patch-3 applies a 3-slab-wide patch (`end = start + 3`), and it QUALIFIES.
///
/// This test asserted the opposite until 2026-09-02, on the reasoning that the
/// clamped extent takes four values (3, 2, 1, 0) across the declared box rather
/// than two, so "a two-way split is not covering". **That reasoning was wrong,
/// and it cost three banked-unsat rows.** The split is not on the extent's
/// VALUE, it is on whether the extent is ZERO:
///
/// * `extent(d) = clamp(d+k,0,L) - clamp(d,0,L)`, so `extent == 0` exactly when
///   `d >= L`, for EVERY `k >= 1`. The split point is therefore still `L`, and
///   the two sub-boxes are still covering and disjoint.
/// * Above `L` nothing is written, so the ScatterND is a no-op and the excision
///   is valid — unchanged by `k`.
/// * Below `L` the extent does vary (3, 2, 1 at d = 0, 1, 2), and that costs
///   nothing, because `decide_low` reads the mask from ORT and decides
///   SEPARATELY FOR EVERY INTEGER CELL. A per-cell extent is observed, never
///   assumed.
///
/// The general lesson, and why this comment is long: a precondition stated over
/// a quantity the algorithm never actually branches on will decline sound
/// inputs forever, and it reads as principled while doing it.
#[test]
fn detects_patch_3_whose_patch_is_wider_than_one_slab() {
    let (Some(model), Some(spec)) = (
        load_model("patch-3.onnx"),
        spec_named("spec_onnx_patch-3_idx_00303_0.vnnlib"),
    ) else {
        eprintln!("skipping: cctsdb benchmark checkout absent");
        return;
    };
    assert_eq!(
        free_integer_dims(&spec).expect("patch-3's spec has the same two free dims"),
        vec![12288, 12289]
    );
    let plan = detect(&model, &spec).expect("patch-3 qualifies: k > 1 is read, not assumed");
    // Same split coordinate and same split point as patch-1 — which is the
    // whole claim: `k` does not move the extent-zero boundary.
    assert_eq!(plan.split_dim, 12288);
    assert_eq!(
        plan.split_len, 3,
        "the extent-zero boundary is the axis length L, independent of k"
    );
    assert_eq!(plan.split_hi, 62.0);
    assert_eq!(plan.mask_len, 12288);
}

/// patch-3 writes into a DENSE all-ones `Constant` where patch-1 uses a
/// `ConstantOfShape` filled with 1.0. Both are the same fact — every scattered-
/// into element is 1.0 — and the detector must key on the fact, not the op
/// type. Keying on `ConstantOfShape` alone declined patch-3 even after the `k`
/// generalization landed.
#[test]
fn accepts_a_dense_all_ones_constant_as_the_scattered_into_mask() {
    let (Some(model), Some(spec)) = (
        load_model("patch-3.onnx"),
        spec_named("spec_onnx_patch-3_idx_00303_0.vnnlib"),
    ) else {
        eprintln!("skipping: cctsdb benchmark checkout absent");
        return;
    };
    let graph = model.graph.as_ref().expect("graph");
    let sn = graph
        .node
        .iter()
        .find(|n| n.op_type == "ScatterND")
        .expect("patch-3 has exactly one ScatterND");
    let producer = graph
        .node
        .iter()
        .find(|n| n.output.iter().any(|o| *o == sn.input[0]))
        .expect("the scattered-into tensor has a producer");
    assert_eq!(
        producer.op_type, "Constant",
        "patch-3 spells the mask as a dense literal, NOT a ConstantOfShape"
    );
    assert!(detect(&model, &spec).is_some());
}

/// ONNX Slice counts a negative bound from the END of the axis (`v + L`) and
/// clamps only afterwards. patch-1's flat input is 12296 long and its image
/// window is `[0, 12288)`, so `end = -8` is a DIFFERENT SPELLING OF THE SAME
/// WINDOW and must produce an identical plan.
///
/// Until 2026-09-02 `contiguous_input_window` clamped the raw value into
/// `[0, L]` first, so `-8` became `0`, the window came out empty, and its length
/// then disagreed with the mask. That failed safe rather than returning a wrong
/// verdict — but only because a later length check and the per-cell ORT gate
/// caught it, which is not a reason to keep computing the wrong window.
#[test]
fn a_negative_slice_end_denotes_the_same_window() {
    let Some((model, spec)) = fixture() else {
        eprintln!("skipping: cctsdb benchmark checkout absent");
        return;
    };
    let control = detect(&model, &spec).expect("control: patch-1 qualifies unmutated");
    assert_eq!(control.image_offset, 0);
    assert_eq!(control.mask_len, 12288);

    let mut mutated = model;
    {
        let graph = mutated.graph.as_mut().expect("graph");
        let slice = graph
            .node
            .iter_mut()
            .find(|n| n.name == "Slice_4")
            .expect("Slice_4 cuts the image window out of the flat input");
        slice.input[2] = "ny_neg_end".to_string();
        let mut neg = NodeProto {
            op_type: "Constant".to_string(),
            name: "ny_neg_end_const".to_string(),
            output: vec!["ny_neg_end".to_string()],
            ..Default::default()
        };
        neg.attribute.push(ny_onnx::onnx_proto::AttributeProto {
            name: "value".to_string(),
            r#type: ny_onnx::onnx_proto::attribute_type::TENSOR,
            t: Some(TensorProto {
                data_type: 7,
                int64_data: vec![-8],
                ..Default::default()
            }),
            ..Default::default()
        });
        graph.node.insert(0, neg);
    }

    let got = detect(&mutated, &spec)
        .expect("-8 denotes 12288 on a 12296-long axis, so the plan is unchanged");
    assert_eq!(got.image_offset, control.image_offset);
    assert_eq!(got.mask_len, control.mask_len);
    assert_eq!(got.split_dim, control.split_dim);
    assert_eq!(got.split_len, control.split_len);
}

/// The free coordinates must never be inside the image window the mask zeroes.
#[test]
fn free_coordinates_lie_outside_the_masked_image_window() {
    let Some((model, spec)) = fixture() else {
        eprintln!("skipping: cctsdb benchmark checkout absent");
        return;
    };
    let plan = detect(&model, &spec).expect("qualifies");
    for &dim in &plan.free_dims {
        assert!(
            dim < plan.image_offset || dim >= plan.image_offset + plan.mask_len,
            "free dim {dim} lies inside the masked window"
        );
    }
}
