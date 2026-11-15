// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//
// #mip-conv-sound-enclosure RATCHET — every layer the MIP encoders ADMIT must
// have an IBP arm that is an ENCLOSURE, by name.
//
// Why this exists (review of bbe9e9657, 2026-09-12, finding "the `_` arm is a
// margin, not a certificate"): `collect_ibp_bounds_sound` (sequential lane)
// routes Linear / the Conv family / AveragePool to certified arms and hands
// EVERY OTHER layer to its plain `propagate_ibp` plus a 1-ULP outward margin;
// `propagate_node_ibp_with_engine_and_deadline` (graph lane) does the same
// with NO margin at all. Both are sound today only because every non-affine
// layer the two MIP encoders admit (`is_mip_encodable_layer`,
// `graph_unstable_relu_count`) has a plain arm that is exact or DIRECTED:
// ReLU is an exact max, Flatten/Reshape are relabels, AddConstant/SubConstant
// use `add_down/up_f32` / `sub_down/up_f32`, BatchNorm publishes directed f64
// endpoints, binary Add uses `add_down/up_f32`. Admit a layer with a
// round-to-NEAREST plain arm (MulConstant, Div, ...) and the Big-M box can be
// under-covered by more than the margin — and the certificate is checked
// against the encoded model, so that is a minted proof.
//
// So this test derives BOTH admitted sets and BOTH sound dispatchers from the
// SOURCE (`include_str!`, so a moved file is a compile error, never an empty
// grep), and asserts every admitted layer is either (a) an explicit
// `propagate_ibp_sound*` arm of that lane's dispatcher, or (b) frozen here as
// DirectedPlain with evidence strings that must be present in the layer's own
// source. Widening an admission set without adding a justified row fails at
// review time. Every check reports ALL of its violations at once.

use std::collections::BTreeSet;

const DISPATCH_SRC: &str = include_str!("dispatch.rs");
const SEQ_SOUND_SRC: &str = include_str!("../../../../ny-propagate/src/network/ibp/forward.rs");
const GRAPH_SOUND_SRC: &str =
    include_str!("../../../../ny-propagate/src/network/graph_alpha/bounds/ibp.rs");

/// Anchors are the fn signatures as written; a rename is a loud failure.
const SEQ_ADMISSION_ANCHOR: &str =
    "fn is_mip_encodable_layer(layer: &ny_propagate::Layer) -> bool {";
const GRAPH_ADMISSION_ANCHOR: &str = "pub(super) fn graph_unstable_relu_count(";
const SEQ_SOUND_ANCHOR: &str = "pub(super) fn collect_ibp_bounds_sound(";
const GRAPH_SOUND_ANCHOR: &str = "pub(super) fn propagate_node_ibp_with_engine_and_deadline(";

/// Why an admitted layer's IBP box is an enclosure in a given lane.
enum Arm {
    /// The lane's sound dispatcher names the layer explicitly and routes it to
    /// a `propagate_ibp_sound*` method.
    Certified,
    /// The layer falls to the dispatcher's `_` arm; its PLAIN arm is exact or
    /// directed, pinned by `evidence` strings that must appear in `file`
    /// (relative to `crates/ny-propagate/src/`).
    DirectedPlain {
        file: &'static str,
        evidence: &'static [&'static str],
    },
}

struct Entry {
    layer: &'static str,
    arm: Arm,
    why: &'static str,
}

/// Sequential MIP lane: `is_mip_encodable_layer` (dispatch.rs) admits, and
/// `Network::collect_ibp_bounds_sound` (forward.rs) bounds.
const SEQUENTIAL_TABLE: &[Entry] = &[
    Entry {
        layer: "Linear",
        arm: Arm::Certified,
        why: "explicit `propagate_ibp_sound` arm: Higham/abssum widening plus directed casts",
    },
    Entry {
        layer: "Conv2d",
        arm: Arm::Certified,
        why: "explicit `propagate_ibp_sound_with_engine` arm: certified γ_{K+2}·S + 2u·|y| term",
    },
    Entry {
        layer: "ReLU",
        arm: Arm::DirectedPlain {
            file: "layers/activations/relu/ibp.rs",
            evidence: &["nan_propagating_max_zero"],
        },
        why: "max(0, ·) is exact in f32 (no rounding), plus the lane's 1-ULP margin",
    },
    Entry {
        layer: "Flatten",
        arm: Arm::DirectedPlain {
            file: "layers/transform/reshape.rs",
            evidence: &["into_shape_with_order("],
        },
        why: "pure relabel of the same elements, no arithmetic",
    },
    Entry {
        layer: "Reshape",
        arm: Arm::DirectedPlain {
            file: "layers/transform/reshape.rs",
            evidence: &["into_shape_with_order("],
        },
        why: "pure relabel of the same elements, no arithmetic",
    },
    Entry {
        layer: "AddConstant",
        arm: Arm::DirectedPlain {
            file: "layers/arithmetic/add_constant.rs",
            evidence: &["add_down_f32(", "add_up_f32("],
        },
        why: "directed endpoint adds",
    },
    Entry {
        layer: "SubConstant",
        arm: Arm::DirectedPlain {
            file: "layers/arithmetic/sub_constant.rs",
            evidence: &["sub_down_f32(", "sub_up_f32("],
        },
        why: "directed endpoint subtracts (non-reverse only is admitted)",
    },
];

/// Graph MIP lane: `graph_unstable_relu_count` (dispatch.rs) admits, and
/// `propagate_node_ibp_with_engine_and_deadline` (graph_alpha/bounds/ibp.rs)
/// bounds — with NO margin on its `_` arm, so the plain arms below carry the
/// whole enclosure claim.
const GRAPH_TABLE: &[Entry] = &[
    Entry {
        layer: "Linear",
        arm: Arm::DirectedPlain {
            file: "layers/linear/ibp.rs",
            evidence: &["round_for_soundness_n_ulps_inplace("],
        },
        why: "the plain arm is internally certified: f64 accumulation + directed cast on 1-D, \
              n+2 ULP outward rounding on the GEMM path (the graph dispatcher calls \
              `propagate_ibp_with_engine`, not a `_sound` method)",
    },
    Entry {
        layer: "Conv2d",
        arm: Arm::Certified,
        why: "explicit `propagate_ibp_sound_with_engine_and_deadline` arm",
    },
    Entry {
        layer: "ReLU",
        arm: Arm::DirectedPlain {
            file: "layers/activations/relu/ibp.rs",
            evidence: &["nan_propagating_max_zero"],
        },
        why: "max(0, ·) is exact in f32; the graph `_` arm adds no margin and needs none here",
    },
    Entry {
        layer: "Flatten",
        arm: Arm::DirectedPlain {
            file: "layers/transform/reshape.rs",
            evidence: &["into_shape_with_order("],
        },
        why: "pure relabel of the same elements, no arithmetic",
    },
    Entry {
        layer: "Reshape",
        arm: Arm::DirectedPlain {
            file: "layers/transform/reshape.rs",
            evidence: &["into_shape_with_order("],
        },
        why: "pure relabel of the same elements, no arithmetic",
    },
    Entry {
        layer: "BatchNorm",
        arm: Arm::DirectedPlain {
            file: "layers/normalization/batch_norm/ibp.rs",
            evidence: &["f64_to_f32_down(", "f64_to_f32_up("],
        },
        why: "certified scale/bias model error, directed f64 endpoints published with \
              directed f64->f32 casts",
    },
    Entry {
        layer: "Add",
        arm: Arm::DirectedPlain {
            file: "layers/binary_ops/add.rs",
            evidence: &["add_down_f32(", "add_up_f32("],
        },
        why: "binary add with directed endpoint adds",
    },
];

/// Evidence sources are embedded at compile time: a table row naming a file
/// that moved is a compile error in this arm, never a silent empty grep.
fn evidence_source(file: &str) -> &'static str {
    match file {
        "layers/activations/relu/ibp.rs" => {
            include_str!("../../../../ny-propagate/src/layers/activations/relu/ibp.rs")
        }
        "layers/transform/reshape.rs" => {
            include_str!("../../../../ny-propagate/src/layers/transform/reshape.rs")
        }
        "layers/arithmetic/add_constant.rs" => {
            include_str!("../../../../ny-propagate/src/layers/arithmetic/add_constant.rs")
        }
        "layers/arithmetic/sub_constant.rs" => {
            include_str!("../../../../ny-propagate/src/layers/arithmetic/sub_constant.rs")
        }
        "layers/linear/ibp.rs" => {
            include_str!("../../../../ny-propagate/src/layers/linear/ibp.rs")
        }
        "layers/normalization/batch_norm/ibp.rs" => {
            include_str!("../../../../ny-propagate/src/layers/normalization/batch_norm/ibp.rs")
        }
        "layers/binary_ops/add.rs" => {
            include_str!("../../../../ny-propagate/src/layers/binary_ops/add.rs")
        }
        other => panic!(
            "ratchet table names a source file with no include_str! mapping: {other} — add it \
             to evidence_source()"
        ),
    }
}

/// The body of a MODULE-LEVEL fn: from its anchor to the first column-0 `}`.
/// (Brace counting is deliberately avoided: doc strings contain braces.)
fn fn_body<'a>(src: &'a str, anchor: &str) -> &'a str {
    let start = src.find(anchor).unwrap_or_else(|| {
        panic!("anchor {anchor:?} not found — the function was renamed or moved; re-anchor")
    });
    let rest = &src[start..];
    let end = rest
        .find("\n}\n")
        .expect("a module-level fn ends with a column-0 closing brace");
    &rest[..end]
}

/// Drop `//` comment lines so prose never counts as a match arm.
fn code_only(body: &str) -> String {
    body.lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Every `Layer::<Variant>` named in the body's CODE.
fn layer_variants(body: &str) -> BTreeSet<String> {
    let code = code_only(body);
    let mut out = BTreeSet::new();
    for seg in code.split("Layer::").skip(1) {
        let name: String = seg
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect();
        if !name.is_empty() {
            out.insert(name);
        }
    }
    out
}

/// The variants whose OWN match arm (text up to the next `Layer::` or the
/// `_ =>` wildcard) calls a `propagate_ibp_sound*` method.
fn certified_arms(sound_body: &str) -> BTreeSet<String> {
    let code = code_only(sound_body);
    let mut out = BTreeSet::new();
    for seg in code.split("Layer::").skip(1) {
        let name: String = seg
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect();
        let own_arm = seg.split("_ =>").next().unwrap_or("");
        if own_arm.contains("propagate_ibp_sound") {
            out.insert(name);
        }
    }
    out
}

/// All violations for one lane, reported together.
fn lane_violations(
    lane: &str,
    admission_body: &str,
    sound_body: &str,
    table: &[Entry],
) -> Vec<String> {
    let mut v = Vec::new();
    let admitted = layer_variants(admission_body);
    let certified = certified_arms(sound_body);

    // Positive controls on the parser itself: an empty set from a broken
    // anchor must fail here, never pass the checks below vacuously.
    for must in ["Conv2d", "ReLU", "Linear"] {
        if !admitted.contains(must) {
            v.push(format!(
                "{lane}: parser positive control — admitted set {admitted:?} lacks {must}; the \
                 admission fn body could not be parsed"
            ));
        }
    }
    if !certified.contains("Conv2d") {
        v.push(format!(
            "{lane}: parser positive control — certified arms {certified:?} lack Conv2d; the \
             sound dispatcher body could not be parsed"
        ));
    }

    let frozen: BTreeSet<String> = table.iter().map(|e| e.layer.to_string()).collect();
    for extra in admitted.difference(&frozen) {
        v.push(format!(
            "{lane}: `Layer::{extra}` is ADMITTED by the encoder but has no row in this \
             ratchet — prove its IBP arm is certified or directed (not round-to-nearest) and \
             add a justified Entry, or do not admit it"
        ));
    }
    for stale in frozen.difference(&admitted) {
        v.push(format!(
            "{lane}: ratchet row `{stale}` is no longer admitted by the encoder — remove it \
             (a ratchet may only describe the live admission set)"
        ));
    }

    for entry in table {
        match &entry.arm {
            Arm::Certified => {
                if !certified.contains(entry.layer) {
                    v.push(format!(
                        "{lane}: `Layer::{}` is frozen as Certified ({}) but the lane's sound \
                         dispatcher has no explicit `propagate_ibp_sound*` arm for it \
                         (certified arms: {certified:?})",
                        entry.layer, entry.why
                    ));
                }
            }
            Arm::DirectedPlain { file, evidence } => {
                if certified.contains(entry.layer) {
                    v.push(format!(
                        "{lane}: `Layer::{}` is frozen as DirectedPlain but the sound \
                         dispatcher now routes it to a certified arm — promote the row to \
                         Certified",
                        entry.layer
                    ));
                }
                let src = evidence_source(file);
                for needle in evidence.iter() {
                    if !src.contains(needle) {
                        v.push(format!(
                            "{lane}: `Layer::{}` DirectedPlain evidence {needle:?} is absent \
                             from ny-propagate/src/{file} ({}) — the directed arm changed; \
                             re-audit before trusting the `_` route",
                            entry.layer, entry.why
                        ));
                    }
                }
            }
        }
    }
    v
}

fn assert_no_violations(lane: &str, violations: &[String]) {
    assert!(
        violations.is_empty(),
        "{lane}: {} admitted-layer soundness violation(s):\n  - {}",
        violations.len(),
        violations.join("\n  - ")
    );
}

/// Sequential lane: `is_mip_encodable_layer` vs `collect_ibp_bounds_sound`.
#[test]
fn mip_admitted_layer_ratchet_sequential_lane_every_admitted_layer_has_a_certified_or_directed_arm()
{
    let admission = fn_body(DISPATCH_SRC, SEQ_ADMISSION_ANCHOR);
    let sound = fn_body(SEQ_SOUND_SRC, SEQ_SOUND_ANCHOR);
    let mut violations = lane_violations("sequential", admission, sound, SEQUENTIAL_TABLE);
    // The `_` arm's 1-ULP outward margin is part of the sequential argument
    // for the exact/relabel rows; its removal must be reviewed here.
    if !code_only(sound).contains("round_for_soundness_inplace()") {
        violations.push(
            "sequential: collect_ibp_bounds_sound's `_` arm no longer applies \
             `round_for_soundness_inplace()` — the 1-ULP margin the DirectedPlain rows cite \
             is gone"
                .to_string(),
        );
    }
    assert_no_violations("sequential", &violations);
}

/// Graph lane: `graph_unstable_relu_count` vs
/// `propagate_node_ibp_with_engine_and_deadline`.
#[test]
fn mip_admitted_layer_ratchet_graph_lane_every_admitted_layer_has_a_certified_or_directed_arm() {
    let admission = fn_body(DISPATCH_SRC, GRAPH_ADMISSION_ANCHOR);
    let sound = fn_body(GRAPH_SOUND_SRC, GRAPH_SOUND_ANCHOR);
    let violations = lane_violations("graph", admission, sound, GRAPH_TABLE);
    assert_no_violations("graph", &violations);
}

/// The ratchet must BITE: a fabricated admission of a layer with a
/// round-to-nearest plain arm (`MulConstant`) has to surface as a violation
/// naming that layer, in both lanes, against the live tables.
#[test]
fn mip_admitted_layer_ratchet_bites_on_a_round_to_nearest_admission() {
    let seq_admission = format!(
        "{}\n        Layer::MulConstant(_) => true,\n",
        fn_body(DISPATCH_SRC, SEQ_ADMISSION_ANCHOR)
    );
    let seq = lane_violations(
        "sequential",
        &seq_admission,
        fn_body(SEQ_SOUND_SRC, SEQ_SOUND_ANCHOR),
        SEQUENTIAL_TABLE,
    );
    assert!(
        seq.iter()
            .any(|m| m.contains("`Layer::MulConstant` is ADMITTED")),
        "sequential ratchet did not bite on a fabricated MulConstant admission: {seq:?}"
    );

    let graph_admission = format!(
        "{}\n            | Layer::MulConstant(_) => {{}}\n",
        fn_body(DISPATCH_SRC, GRAPH_ADMISSION_ANCHOR)
    );
    let graph = lane_violations(
        "graph",
        &graph_admission,
        fn_body(GRAPH_SOUND_SRC, GRAPH_SOUND_ANCHOR),
        GRAPH_TABLE,
    );
    assert!(
        graph
            .iter()
            .any(|m| m.contains("`Layer::MulConstant` is ADMITTED")),
        "graph ratchet did not bite on a fabricated MulConstant admission: {graph:?}"
    );
}
