// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Corpus comparison: on EVERY distinct model of eight VNN-COMP corpora, every
//! shape the static fallback fills must EQUAL the shape ONNX Runtime reports
//! for the same bytes.
//!
//! The ORT tables are committed (`testdata/static_infer_ort_reference.json`)
//! so this test needs no ONNX Runtime. They were produced by
//! `scripts/gen_static_shape_infer_reference.py`, which mirrors the loader's
//! ORT extraction step for step (opset normalisation, batch pin, value_info
//! and re-consumed-output stripping, intermediate exposure, informative-shape
//! filter, and the zero-dim rank probe, so a reference `[]` is a proven rank
//! 0 exactly as in the loader's table) and records the onnx / onnxruntime
//! versions it ran with. Each entry
//! also carries the SHA-256 of the decompressed model bytes, so a corpus that
//! changes under the table fails as STALE rather than comparing against the
//! wrong reference.
//!
//! `benchmarks/` is gitignored and lives only in the main checkout, so the
//! comparison is a conformance test (`EXTERNAL_CONFORMANCE.md`): it is
//! registered only in the `external-vnncomp` lane
//! (`cargo run -p ny-onnx --bin ny_onnx_conformance -- vnncomp`), where an
//! absent corpus, a missing model, a stale hash, a mismatch, a filled tensor
//! ORT has no shape for, or an ORT-resolved tensor left unknown all FAIL. It
//! is never an early-returning `ok`: the hermetic default suite runs only the
//! committed-reference well-formedness check, which needs no fixture.

#[cfg(feature = "external-vnncomp")]
use super::static_infer::infer_tensor_shapes_static;
use serde::Deserialize;
#[cfg(feature = "external-vnncomp")]
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
#[cfg(feature = "external-vnncomp")]
use std::collections::HashMap;
#[cfg(feature = "external-vnncomp")]
use std::io::Read;
#[cfg(feature = "external-vnncomp")]
use std::path::{Path, PathBuf};

const REFERENCE_JSON: &str = include_str!("testdata/static_infer_ort_reference.json");

#[derive(Deserialize)]
struct Reference {
    schema: u32,
    generator: String,
    onnx_version: String,
    onnxruntime_version: String,
    corpora: Vec<String>,
    years: Vec<String>,
    models: Vec<ReferenceModel>,
}

#[derive(Deserialize)]
struct ReferenceModel {
    sha256: String,
    corpus: String,
    paths: Vec<String>,
    ort_shapes: BTreeMap<String, Vec<i64>>,
    ort_error: Option<String>,
}

#[cfg(feature = "external-vnncomp")]
#[derive(Default)]
struct CorpusTally {
    models: usize,
    filled_equal: usize,
    left_unknown: usize,
}

// The fixture helpers below are also used by `ort_rank_probe_tests`'s
// cctsdb corpus test, which runs in the same `external-vnncomp` lane.
#[cfg(feature = "external-vnncomp")]
pub(super) fn benchmarks_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../benchmarks")
}

/// Both 2026 checkout spellings are in use (see `vnncomp2026_benchmarks_root`
/// in ny-cli); the reference records the plain-year spelling.
#[cfg(feature = "external-vnncomp")]
pub(super) fn candidate_paths(root: &Path, relative: &str) -> Vec<PathBuf> {
    let mut candidates = vec![root.join(relative)];
    if let Some(rest) = relative.strip_prefix("vnncomp2026/") {
        candidates.push(root.join("vnncomp2026_benchmarks").join(rest));
    }
    candidates
}

#[cfg(feature = "external-vnncomp")]
pub(super) fn read_model(path: &Path) -> Vec<u8> {
    let raw = std::fs::read(path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
    if path.extension().is_some_and(|ext| ext == "gz") {
        let mut bytes = Vec::new();
        flate2::read::GzDecoder::new(raw.as_slice())
            .read_to_end(&mut bytes)
            .unwrap_or_else(|err| panic!("gunzip {}: {err}", path.display()));
        bytes
    } else {
        raw
    }
}

#[cfg(feature = "external-vnncomp")]
pub(super) fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn load_reference() -> Reference {
    let reference: Reference =
        serde_json::from_str(REFERENCE_JSON).expect("committed ORT reference parses");
    assert_eq!(reference.schema, 1, "reference schema");
    assert_eq!(
        reference.generator, "scripts/gen_static_shape_infer_reference.py",
        "reference provenance"
    );
    assert!(
        !reference.onnx_version.is_empty() && !reference.onnxruntime_version.is_empty(),
        "the reference must record the onnx / onnxruntime versions that produced it"
    );
    assert_eq!(reference.years, ["vnncomp2025", "vnncomp2026"]);
    assert_eq!(
        reference.corpora,
        [
            "linearizenn_2024",
            "lsnc_relu",
            "acasxu_2023",
            "tllverifybench_2023",
            "cersyve",
            "dist_shift_2023",
            "collins_rul_cnn_2022",
            "safenlp_2024",
        ],
        "the eight corpora the fallback is pinned against"
    );
    for corpus in &reference.corpora {
        assert!(
            reference.models.iter().any(|model| &model.corpus == corpus),
            "reference carries no model for corpus {corpus}"
        );
    }
    reference
}

#[cfg(feature = "external-vnncomp")]
type ComparedTable<'a> = (&'a [String], HashMap<String, Vec<i64>>);

#[cfg(feature = "external-vnncomp")]
fn table_for<'a>(tables: &'a [ComparedTable<'_>], suffix: &str) -> &'a HashMap<String, Vec<i64>> {
    tables
        .iter()
        .find(|(paths, _)| paths.iter().any(|path| path.ends_with(suffix)))
        .map(|(_, table)| table)
        .unwrap_or_else(|| panic!("no compared model ends with {suffix}"))
}

#[test]
fn committed_ort_reference_is_well_formed() {
    let reference = load_reference();
    assert!(!reference.models.is_empty());
    for model in &reference.models {
        assert_eq!(model.sha256.len(), 64, "{:?}", model.paths);
        assert_ne!((model.paths).len(), 0);
        assert!(
            model.ort_error.is_some() || !model.ort_shapes.is_empty(),
            "{:?}: an ORT success with an empty table carries no evidence",
            model.paths
        );
    }
}

#[test]
#[cfg(feature = "external-vnncomp")]
fn static_fallback_equals_ort_on_every_tensor_it_fills_across_the_corpus() {
    let reference = load_reference();
    let root = benchmarks_root();
    assert!(
        root.is_dir(),
        "VNN-COMP benchmark fixture root is missing at {} (benchmarks/ is gitignored and lives \
         only in the main checkout; run benchmarks/download_benchmarks.sh or link it), so the \
         static-fallback comparison against {} ORT reference tables cannot run",
        root.display(),
        reference.models.len()
    );

    let mut missing = Vec::new();
    let mut stale = Vec::new();
    let mut mismatches = Vec::new();
    let mut unverifiable = Vec::new();
    let mut tallies: BTreeMap<&str, CorpusTally> = BTreeMap::new();
    let mut tables: Vec<ComparedTable<'_>> = Vec::new();

    for model in &reference.models {
        let label = model.paths.first().map_or("?", String::as_str);
        let Some(path) = model
            .paths
            .iter()
            .flat_map(|relative| candidate_paths(&root, relative))
            .find(|candidate| candidate.is_file())
        else {
            missing.push(label.to_string());
            continue;
        };
        let bytes = read_model(&path);
        let digest = sha256_hex(&bytes);
        if digest != model.sha256 {
            stale.push(format!(
                "{}: sha256 {digest} != reference {}",
                path.display(),
                model.sha256
            ));
            continue;
        }
        let table = infer_tensor_shapes_static(&bytes);
        let tally = tallies.entry(model.corpus.as_str()).or_default();
        tally.models += 1;
        if let Some(error) = &model.ort_error {
            if !table.shapes.is_empty() {
                unverifiable.push(format!(
                    "{label}: ORT errored ({error}), so none of the fallback's {} fills can be \
                     checked",
                    table.shapes.len()
                ));
            }
            continue;
        }
        for (tensor, dims) in &table.shapes {
            match model.ort_shapes.get(tensor) {
                Some(ort) if ort == dims => tally.filled_equal += 1,
                Some(ort) => mismatches.push(format!(
                    "{label}: '{tensor}' fallback {dims:?} vs ORT {ort:?}"
                )),
                None => unverifiable.push(format!(
                    "{label}: '{tensor}' fallback {dims:?} but ORT reports no shape"
                )),
            }
        }
        tally.left_unknown += model
            .ort_shapes
            .keys()
            .filter(|tensor| !table.shapes.contains_key(*tensor))
            .count();
        tables.push((model.paths.as_slice(), table.shapes));
    }

    eprintln!(
        "static fallback vs ORT {} (onnx {}): corpus / models / filled==ORT / left unknown",
        reference.onnxruntime_version, reference.onnx_version
    );
    for (corpus, tally) in &tallies {
        eprintln!(
            "  {corpus}: {} / {} / {}",
            tally.models, tally.filled_equal, tally.left_unknown
        );
    }

    assert!(
        missing.is_empty(),
        "reference models absent from {}: {missing:#?}",
        root.display()
    );
    assert!(stale.is_empty(), "reference tables are stale: {stale:#?}");
    assert!(mismatches.is_empty(), "WRONG SHAPES: {mismatches:#?}");
    assert!(
        unverifiable.is_empty(),
        "fills with no ORT shape to check against: {unverifiable:#?}"
    );
    let filled: usize = tallies.values().map(|tally| tally.filled_equal).sum();
    let left_unknown: usize = tallies.values().map(|tally| tally.left_unknown).sum();
    assert!(filled > 0, "the fallback filled nothing across the corpus");
    // Every op these corpora use is modelled, so coverage is complete; a rule
    // that silently stops firing fails here instead of costing verdicts.
    assert_eq!(
        left_unknown, 0,
        "tensors ORT resolves that the fallback left unknown"
    );

    // The load-bearing tensors of the two measured regressions, by name.
    let linearizenn = table_for(&tables, "linearizenn_2024/onnx/AllInOne_120_120.onnx");
    assert_eq!(
        linearizenn
            .get("/controller/MatMul_output_0")
            .map(Vec::as_slice),
        Some([1, 1].as_slice())
    );
    assert_eq!(
        linearizenn.get("/Concat_output_0").map(Vec::as_slice),
        Some([1, 2].as_slice())
    );
    let lsnc = table_for(&tables, "lsnc_relu/onnx/relu_quadrotor2d_state.onnx");
    for (tensor, dims) in [
        ("/Gather_output_0", &[1, 2][..]),
        ("/Concat_output_0", &[1, 4][..]),
        ("/controller/net/net/net.2_1/MatMul_output_0", &[2][..]),
        ("/Slice_1_output_0", &[1, 3][..]),
        ("/lyapunov/ReduceSum_output_0", &[1, 1][..]),
        ("123", &[1, 8][..]),
    ] {
        assert_eq!(
            lsnc.get(tensor).map(Vec::as_slice),
            Some(dims),
            "lsnc '{tensor}'"
        );
    }
}
