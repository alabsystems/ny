// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The ONNX Runtime ORACLE for the checker's routing, reshaping and MatMul
//! forms (hermetic: committed fixtures, no Python at test time).
//!
//! `tests/data/ort_forms/gen_ort_forms.py` builds eight small models with the
//! official `onnx` helper (a writer independent of both the checker's reader
//! and `tests/support`'s writer) covering Slice (negative steps, INT64
//! extremes, negative axes, defaulted operands), Gather (negative, repeated
//! and scalar indices), Concat (inner axes, repeated operands), Reshape
//! (`0` copies, `-1`), Flatten (axes 0..=r and negative), MatMul with the
//! constant on either side (rank-1 promotion, batch broadcasting), a rank-1
//! and a `[3, 2]` graph input, and a Relu network through all of them; it
//! evaluates each with onnxruntime at four points and commits the outputs.
//! Every weight and input is a small dyadic rational, so ORT's float32 result
//! is the EXACT real value: the checker's certified point enclosure must
//! contain it with no tolerance, and on the pure-routing models it must
//! equal it bit for bit.

use std::path::PathBuf;

use ny_cert::json::{self, Json};
use ny_proofcheck::bundle::sha256_hex;
use ny_proofcheck::ibp::eval_point;
use ny_proofcheck::testkit::net_from_onnx;

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/ort_forms")
}

fn nums(v: &Json, what: &str) -> Vec<f64> {
    let Json::Arr(a) = v else {
        panic!("{what}: not an array");
    };
    a.iter()
        .map(|s| {
            s.as_str()
                .unwrap_or_else(|| panic!("{what}: numbers are decimal strings"))
                .parse::<f64>()
                .unwrap_or_else(|e| panic!("{what}: {e}"))
        })
        .collect()
}

#[test]
fn routing_forms_match_onnxruntime_on_committed_fixtures() {
    let dir = data_dir();
    let text = std::fs::read_to_string(dir.join("manifest.json")).expect("manifest.json");
    let m = json::from_str(&text).expect("manifest is JSON");
    let Json::Arr(fixtures) = &m["fixtures"] else {
        panic!("manifest has no fixtures");
    };
    let names: Vec<&str> = fixtures
        .iter()
        .map(|f| f["file"].as_str().expect("file"))
        .collect();
    assert_eq!(
        names,
        [
            "slice.onnx",
            "gather.onnx",
            "concat.onnx",
            "reshape.onnx",
            "matmul.onnx",
            "rank1.onnx",
            "nonbatch.onnx",
            "routing_mlp.onnx"
        ],
        "every generated fixture is checked"
    );
    let mut compared = 0usize;
    for f in fixtures {
        let file = f["file"].as_str().expect("file");
        let bytes = std::fs::read(dir.join(file)).unwrap_or_else(|e| panic!("{file}: {e}"));
        assert_eq!(
            Some(sha256_hex(&bytes).as_str()),
            f["sha256"].as_str(),
            "{file}: not the model onnxruntime evaluated"
        );
        let net = net_from_onnx(&bytes).unwrap_or_else(|e| panic!("{file}: refused: {e}"));
        let routing = matches!(f["routing_only"], Json::Bool(true));
        if routing {
            // Pure data movement: the lowering may contain nothing but
            // copies (Slice / Gather / Concat -> Select, reshapes -> Identity).
            for n in &net.nodes {
                assert!(
                    matches!(n.op.kind(), "Select" | "Identity"),
                    "{file}: a routing model lowered to {}",
                    n.op.kind()
                );
            }
        }
        let (Json::Arr(points), Json::Arr(outputs)) = (&f["points"], &f["outputs"]) else {
            panic!("{file}: points / outputs");
        };
        assert_eq!(points.len(), outputs.len());
        assert_ne!(points.as_slice(), [] as [Json; 0]);
        for (k, (p, y)) in points.iter().zip(outputs).enumerate() {
            let x = nums(p, file);
            let yo = nums(y, file);
            assert_eq!(net.n_in(), x.len(), "{file}: input size");
            assert_eq!(net.n_out(), yo.len(), "{file}: output size");
            let b = eval_point(&net, &x);
            let (lo, hi) = (&b.lo[net.output], &b.hi[net.output]);
            for (j, &v) in yo.iter().enumerate() {
                assert!(
                    lo[j] <= v && v <= hi[j],
                    "{file} point {k} y{j}: checker [{}, {}] misses onnxruntime {v}",
                    lo[j],
                    hi[j]
                );
                if routing {
                    assert!(
                        lo[j] == v && hi[j] == v,
                        "{file} point {k} y{j}: a copy is not exact: [{}, {}] vs {v}",
                        lo[j],
                        hi[j]
                    );
                }
            }
            compared += 1;
        }
    }
    assert_eq!(compared, 32, "8 fixtures x 4 points");
}
