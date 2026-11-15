// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! INTEGRATOR (explicit `external-vnncomp` lane: `required-features` in
//! Cargo.toml, absent from the default lane, a hard failure when selected
//! without its prerequisites): the checker's lowering of the REAL corpus
//! networks its routing / reshaping / MatMul forms newly admit, against ONNX
//! Runtime on the ORIGINAL model at eight points of each property's box
//! (relative error <= 1e-4: ORT runs in float32, the checker's enclosure
//! midpoint is f64).
//!
//! It needs the MAIN tree's gitignored `benchmarks/vnncomp2026/` (a `git
//! worktree` has none — the CLAUDE.md worktree trap; a symlink to the main
//! tree's copy works) and a `python3` on PATH with numpy + onnxruntime (on the
//! GB10, `<home>/x86emu/venv-arm/bin`):
//!
//! ```text
//! PATH=<home>/x86emu/venv-arm/bin:$PATH targo --unverified test --locked --offline \
//!     -p ny-proofcheck --features external-vnncomp --test real_lowering_ort -- --nocapture
//! ```

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use ny_cert::json::{self, Json};
use ny_proofcheck::ibp::eval_point;
use ny_proofcheck::onnx_wire::{parse_model, Dim};
use ny_proofcheck::testkit::net_from_onnx;
use ny_proofcheck::vnnlib;

fn corpus() -> PathBuf {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../benchmarks/vnncomp2026/benchmarks");
    assert!(
        dir.join("nn4sys/1.0/instances.csv").is_file(),
        "{} is missing: this integrator needs the MAIN tree's gitignored benchmarks/vnncomp2026 \
         (a git worktree has none); run it from <home>/ny or symlink the corpus",
        dir.display()
    );
    dir
}

/// `(category, onnx, vnnlib)` rows whose networks the routing forms admit:
/// each was exit 31 `onnx_lowering` before (Reshape to a non-`[1, N]` target,
/// Slice, Gather/Concat behind them), plus one previously admitted control.
const ROWS: &[(&str, &str, &str)] = &[
    (
        "soundnessbench_2026",
        "onnx/model_residual.onnx",
        "vnnlib/properties/property_001.vnnlib",
    ),
    (
        "soundnessbench_2026",
        "onnx/model.onnx",
        "vnnlib/properties/property_000.vnnlib",
    ),
    (
        "nn4sys",
        "onnx/pensieve_small_simple.onnx",
        "vnnlib/pensieve_simple_21.vnnlib",
    ),
    (
        "linearizenn_2024",
        "onnx/AllInOne_10_10.onnx",
        "vnnlib/prop_10_10_2.vnnlib",
    ),
    (
        "linearizenn_2024",
        "onnx/AllInOne_120_120.onnx",
        "vnnlib/prop_120_120_0.vnnlib",
    ),
    ("nn4sys", "onnx/lindex.onnx", "vnnlib/lindex_1.vnnlib"),
];

fn ort_outputs(onnx: &Path, dims: &[i64], pts: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let script = format!(
        "import json,sys,numpy as np,onnxruntime as ort\n\
         s=ort.InferenceSession(sys.argv[1],providers=['CPUExecutionProvider'])\n\
         pts=json.load(sys.stdin)\n\
         name=s.get_inputs()[0].name\n\
         out=[[repr(float(v)) for v in s.run(None,{{name:np.array([float(t) for t in p],dtype=np.float32).reshape({dims:?})}})[0].reshape(-1)] for p in pts]\n\
         print(json.dumps(out))\n"
    );
    let mut child = Command::new("python3")
        .arg("-c")
        .arg(script)
        .arg(onnx)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("python3 with numpy + onnxruntime is required for this integrator test");
    {
        use std::io::Write;
        let pts_json = Json::Arr(
            pts.iter()
                .map(|p| Json::Arr(p.iter().map(|v| Json::Str(format!("{v:?}"))).collect()))
                .collect(),
        );
        child
            .stdin
            .as_mut()
            .expect("stdin")
            .write_all(json::to_string(&pts_json).as_bytes())
            .expect("write points");
    }
    let o = child.wait_with_output().expect("python3");
    assert!(o.status.success(), "onnxruntime evaluation failed");
    let text = String::from_utf8(o.stdout).expect("utf-8");
    let Json::Arr(rows) = json::from_str(text.trim()).expect("ort json") else {
        panic!("ort output is not an array");
    };
    rows.iter()
        .map(|r| {
            let Json::Arr(v) = r else {
                panic!("ort row");
            };
            v.iter()
                .map(|s| {
                    s.as_str()
                        .expect("repr string")
                        .parse::<f64>()
                        .expect("f64")
                })
                .collect()
        })
        .collect()
}

#[test]
fn real_corpus_networks_lower_and_match_onnxruntime() {
    let root = corpus();
    for &(cat, onnx_rel, vnn_rel) in ROWS {
        let dir = root.join(cat).join("1.0");
        let onnx = dir.join(onnx_rel);
        let bytes = std::fs::read(&onnx).unwrap_or_else(|e| panic!("{}: {e}", onnx.display()));
        let net =
            net_from_onnx(&bytes).unwrap_or_else(|e| panic!("{cat}/{onnx_rel}: refused: {e}"));
        let spec = vnnlib::parse(
            &std::fs::read_to_string(dir.join(vnn_rel))
                .unwrap_or_else(|e| panic!("{vnn_rel}: {e}")),
        )
        .unwrap_or_else(|e| panic!("{vnn_rel}: {}", e.0));
        assert_eq!(
            (spec.n_x, spec.n_y),
            (net.n_in(), net.n_out()),
            "{cat}/{onnx_rel}: the spec's sizes vs the lowered network"
        );
        let reg = &spec.regions[0];
        let mut s: u64 = 0x0123_4567_89ab_cdef;
        let pts: Vec<Vec<f64>> = (0..8)
            .map(|_| {
                reg.lo_in
                    .iter()
                    .zip(&reg.hi_in)
                    .map(|(a, b)| {
                        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                        let u = ((s >> 11) as f64) / ((1u64 << 53) as f64);
                        f64::from((a + u * (b - a)) as f32)
                    })
                    .collect()
            })
            .collect();
        let dims: Vec<i64> = {
            let m = parse_model(&bytes).expect("model");
            let inits: Vec<&str> = m
                .graph
                .initializers
                .iter()
                .map(|t| t.name.as_str())
                .collect();
            let inp = m
                .graph
                .inputs
                .iter()
                .find(|i| !inits.contains(&i.name.as_str()))
                .expect("graph input");
            inp.dims
                .as_ref()
                .expect("shape")
                .iter()
                .map(|d| match d {
                    Dim::Value(v) => *v,
                    _ => 1,
                })
                .collect()
        };
        let ort = ort_outputs(&onnx, &dims, &pts);
        let mut worst = 0.0f64;
        for (p, y) in pts.iter().zip(&ort) {
            let b = eval_point(&net, p);
            assert_eq!(y.len(), net.n_out(), "{cat}/{onnx_rel}: ORT output size");
            for (j, yo) in y.iter().enumerate() {
                let (lo, hi) = (b.lo[net.output][j], b.hi[net.output][j]);
                let mid = 0.5 * lo + 0.5 * hi;
                let rel = (mid - yo).abs() / yo.abs().max(1.0);
                worst = worst.max(rel);
                assert!(
                    rel <= 1e-4,
                    "{cat}/{onnx_rel} y{j}: checker {mid} [{lo}, {hi}] vs ORT {yo} (rel {rel:e})"
                );
            }
        }
        let kinds: Vec<&str> = net.nodes.iter().map(|n| n.op.kind()).collect();
        eprintln!(
            "[real-lowering-ort] {cat}/{onnx_rel}: lowered ({} nodes, {} Select), n_in {} n_out {}, \
             8 points, worst rel {worst:e}",
            kinds.len(),
            kinds.iter().filter(|k| **k == "Select").count(),
            net.n_in(),
            net.n_out()
        );
    }
}
