// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! B2.1 INTEGRATOR tests: the real corpus, the real binary. This target is in
//! the explicit `external-vnncomp` lane (`required-features` in Cargo.toml), so
//! it is absent from the default lane and a hard failure when selected without
//! its prerequisites.
//!
//! These run in the MAIN TREE only — a `git worktree` has no `benchmarks/`
//! (gitignored), and every fixture assertion below names that trap instead of
//! skipping. Run them with the RELEASE binary (the debug `ny` is far too slow
//! for resnet rows), through the box, e.g.
//!
//! ```text
//! scripts/ny_box.sh measure -- rustup run trust targo --unverified test --release \
//!     --locked --offline -p ny-cli --features external-vnncomp --test proof_check_real \
//!     -- --nocapture
//! ```
//!
//! The pre-registered rows are all FIELD-CORROBORATED banked unsats, so the
//! expected answer is known: `ny vnncomp v1 --proof-profile proof` must
//! publish `unsat` with a v2 bundle, and `ny proof-check` must never say
//! `invalid`, `refuted`, `malformed`, `unsupported` or `checker_fault` on
//! them (each is the soundness-emergency protocol). The registry's pass line
//! (>= 6/9 rows `rc=0`) is printed as a `[proof-check-real]` summary.
//!
//! LEGACY ROWS (certified-moat review, 2026-10-08): a row published through a
//! legacy door (the sequential engine, a root-only graph fast path, the
//! cctsdb sub-box refutation, the nonlinear interval BaB, the relational
//! implication, the margin-row lane's partial stream under the internal
//! verifier's legacy verdict) streams no checkable proof. Its bundle must be
//! WELL-FORMED all the same: exit 30 stays the emergency, and the expected
//! answer is `unconfirmed` (exit 10; 0 where the lane's tapes verify), or —
//! for a row whose network is outside the CHECKER's own lowering — its pinned
//! scope limit (exit 31, `onnx_lowering`, reached only after the bundle
//! parsed).
//! `proof_check_real_legacy_lane_bundles_are_well_formed` pins the rows the
//! ENG14 pin reproduced as `malformed` (`missing_sites`, or `unknown_field`
//! for a dual-network header) or `unsupported` (`uncovered_source`). They live
//! under `benchmarks/vnncomp2026/benchmarks/<category>/<layout>/` (the moat's
//! `1.0` layout; `2.0` for the 2026-only categories).

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repo root")
}

fn bench(category: &str) -> PathBuf {
    let dir = repo_root()
        .join("benchmarks/vnncomp2025/benchmarks")
        .join(category);
    assert!(
        dir.join("instances.csv").is_file(),
        "{} is missing: these integrator tests need the MAIN tree's gitignored benchmarks/ \
         (a git worktree has none — the CLAUDE.md worktree trap); run them from <home>/ny",
        dir.display()
    );
    dir
}

/// `(category, onnx, vnnlib, budget)` of a pre-registered row.
fn row(category: &str, needle: &str) -> (String, PathBuf, PathBuf, String) {
    let dir = bench(category);
    let csv = std::fs::read_to_string(dir.join("instances.csv")).expect("instances.csv");
    let line = csv
        .lines()
        .find(|l| l.contains(needle))
        .unwrap_or_else(|| panic!("{needle} is not in {category}/instances.csv"));
    let f: Vec<&str> = line.split(',').collect();
    assert_eq!(f.len(), 3, "instances.csv row shape: {line}");
    (
        category.to_string(),
        dir.join(f[0]),
        dir.join(f[1]),
        f[2].trim().to_string(),
    )
}

/// The §9 pre-registered field-corroborated banked unsats. The cct row is
/// named by the integrator from results.tsv BEFORE the run (it is listed
/// here once named; until then the summary line says so).
const ROWS: &[(&str, &str)] = &[
    ("cifar100_2024", "resnet_medium_prop_idx_6659_sidx_4583"),
    ("cifar100_2024", "resnet_medium_prop_idx_8762_sidx_3374"),
    ("cifar100_2024", "resnet_medium_prop_idx_5242_sidx_1208"),
    ("cifar100_2024", "resnet_medium_prop_idx_5831_sidx_5506"),
    ("cifar100_2024", "resnet_medium_prop_idx_1798_sidx_1061"),
    ("cifar100_2024", "resnet_large_prop_idx_9883_sidx_7162"),
    ("tinyimagenet_2024", "resnet_medium_prop_idx_2045_sidx_7367"),
    ("tinyimagenet_2024", "resnet_medium_prop_idx_4871_sidx_7562"),
];

const CCT_ROWS: &[(&str, &str)] = &[];

fn ny() -> Command {
    Command::new(env!("CARGO_BIN_EXE_ny"))
}

/// Run the scored entry point under the Proof profile; return the bundle.
fn produce_bundle(
    out: &Path,
    r: &(String, PathBuf, PathBuf, String),
    tag: &str,
) -> Option<PathBuf> {
    let results = out.join(format!("{tag}.txt"));
    let status = ny()
        .args(["vnncomp", "v1", &r.0])
        .arg(&r.1)
        .arg(&r.2)
        .arg(&results)
        .arg(&r.3)
        .args(["--proof-profile", "proof"])
        .status()
        .expect("run ny vnncomp v1");
    assert!(status.success(), "{tag}: ny vnncomp v1 failed: {status}");
    let result = std::fs::read_to_string(&results).unwrap_or_default();
    let first = result.lines().next().unwrap_or("").trim().to_string();
    eprintln!("[proof-check-real] {tag}: published `{first}`");
    let mut bundle = results.into_os_string();
    bundle.push(".proof.jsonl");
    let bundle = PathBuf::from(bundle);
    (first == "unsat" && bundle.is_file()).then_some(bundle)
}

fn check(bundle: &Path, extra: &[&str]) -> (i32, String) {
    let o = ny()
        .arg("proof-check")
        .arg(bundle)
        .args(["--threads", "16"])
        .args(extra)
        .output()
        .expect("run ny proof-check");
    let stderr = String::from_utf8_lossy(&o.stderr).to_string();
    let done = stderr
        .lines()
        .find(|l| l.starts_with("PROOF_CHECK_DONE rc="))
        .unwrap_or("")
        .to_string();
    (o.status.code().unwrap_or(-1), done)
}

/// Integrator: main tree, release binary, box; minutes per row.
#[test]
fn proof_check_real_margin_row_bundles_on_banked_unsats() {
    let out = tempfile::tempdir().expect("tempdir");
    let mut verified = 0usize;
    let mut ran = 0usize;
    for (k, (cat, needle)) in ROWS.iter().chain(CCT_ROWS).enumerate() {
        let r = row(cat, needle);
        let tag = format!("r{k}");
        let Some(bundle) = produce_bundle(out.path(), &r, &tag) else {
            eprintln!("[proof-check-real] {needle}: no unsat bundle this run (a timeout is not an emergency)");
            continue;
        };
        ran += 1;
        let (rc, done) = check(&bundle, &[]);
        eprintln!("[proof-check-real] {needle}: {done}");
        assert!(
            ![20, 21, 30, 31, 40].contains(&rc),
            "SOUNDNESS-EMERGENCY PROTOCOL: {needle} checked rc={rc}: {done}"
        );
        if rc == 0 {
            verified += 1;
        }
    }
    eprintln!(
        "[proof-check-real] summary verified={verified} checked={ran} registered={} cct_named={} \
         (pass line: >= 6/9 rc=0)",
        ROWS.len() + CCT_ROWS.len(),
        !CCT_ROWS.is_empty()
    );
    assert!(ran > 0, "no row published a bundle: VOID");
}

fn rewrite(bundle: &Path, dest: &Path, edit: impl FnOnce(&mut Vec<String>)) {
    use sha2::{Digest, Sha256};
    let text = std::fs::read_to_string(bundle).expect("bundle");
    let mut lines: Vec<String> = text.lines().map(str::to_string).collect();
    lines.pop(); // the end record
    edit(&mut lines);
    let mut body = String::new();
    for l in &lines {
        body.push_str(l);
        body.push('\n');
    }
    let end = serde_json::json!({
        "kind": "end",
        "lines": lines.len() + 1,
        "sha256": format!("{:x}", Sha256::digest(body.as_bytes())),
    });
    body.push_str(&end.to_string());
    body.push('\n');
    std::fs::write(dest, body).expect("write mutated bundle");
}

/// Integrator: main tree, release binary, box.
#[test]
fn proof_check_real_bundle_mutations_are_refused() {
    let out = tempfile::tempdir().expect("tempdir");
    let r = row(ROWS[0].0, ROWS[0].1);
    let bundle = produce_bundle(out.path(), &r, "m").expect("the first row publishes unsat");
    let text = std::fs::read_to_string(&bundle).expect("bundle");
    let node_idx: Vec<usize> = text
        .lines()
        .enumerate()
        .filter(|(_, l)| l.contains("\"kind\":\"node\"") && l.contains("\"close\":\"bound\""))
        .map(|(i, _)| i)
        .collect();
    assert!(
        !node_idx.is_empty(),
        "the bundle carries no bound node: VOID"
    );
    // Drop a node -> 20.
    let dropped = out.path().join("dropped.proof.jsonl");
    rewrite(&bundle, &dropped, |l| {
        l.remove(node_idx[0]);
    });
    assert_eq!(
        check(&dropped, &[]).0,
        20,
        "a dropped node must be a cover gap"
    );
    // Flip a side -> 20.
    let flipped = out.path().join("flipped.proof.jsonl");
    rewrite(&bundle, &flipped, |l| {
        let mut v: serde_json::Value = serde_json::from_str(&l[node_idx[0]]).expect("node");
        let lits = v["lits"].as_array_mut().expect("lits");
        let last = lits.last_mut().expect("a literal");
        let n = last.as_array().map_or(0, Vec::len);
        let side = last[n - 1].as_u64().unwrap_or(0);
        last[n - 1] = serde_json::json!(1 - side);
        l[node_idx[0]] = v.to_string();
    });
    assert_eq!(
        check(&flipped, &[]).0,
        20,
        "a flipped literal must be a cover gap"
    );
    // Truncate -> 30.
    let cut = out.path().join("cut.proof.jsonl");
    std::fs::write(&cut, &text.as_bytes()[..text.len() / 2]).expect("write");
    assert_eq!(check(&cut, &[]).0, 30, "a cut bundle is malformed");
}

/// Integrator: main tree; needs python3 with onnxruntime.
#[test]
fn proof_check_lowering_matches_ort_on_margin_row_nets() {
    // The G1 analogue: the checker's lowering against ONNX Runtime on the
    // ORIGINAL model, at 8 points of a real property box, relative error
    // <= 1e-4 (ORT runs in f32; the checker's enclosure midpoint in f64).
    for (cat, needle) in [ROWS[0], ROWS[5], ROWS[6]] {
        let r = row(cat, needle);
        let bytes = std::fs::read(&r.1).expect("onnx");
        let net = ny_proofcheck::testkit::net_from_onnx(&bytes).expect("the checker lowers it");
        let spec = ny_proofcheck::vnnlib::parse(&std::fs::read_to_string(&r.2).expect("vnnlib"))
            .expect("the checker parses it");
        let reg = &spec.regions[0];
        let mut pts: Vec<Vec<f64>> = Vec::new();
        let mut s: u64 = 0x0123_4567;
        for _ in 0..8 {
            pts.push(
                reg.lo_in
                    .iter()
                    .zip(&reg.hi_in)
                    .map(|(a, b)| {
                        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                        let u = ((s >> 11) as f64) / ((1u64 << 53) as f64);
                        ((a + u * (b - a)) as f32) as f64
                    })
                    .collect(),
            );
        }
        let dims = {
            let m = ny_proofcheck::onnx_wire::parse_model(&bytes).expect("model");
            let inp = &m.graph.inputs[0];
            inp.dims
                .as_ref()
                .expect("shape")
                .iter()
                .map(|d| match d {
                    ny_proofcheck::onnx_wire::Dim::Value(v) => *v,
                    _ => 1,
                })
                .collect::<Vec<i64>>()
        };
        let script = format!(
            "import json,sys,numpy as np,onnxruntime as ort\n\
             s=ort.InferenceSession(sys.argv[1]);pts=json.load(sys.stdin)\n\
             name=s.get_inputs()[0].name\n\
             out=[s.run(None,{{name:np.array(p,dtype=np.float32).reshape({dims:?})}})[0].reshape(-1).tolist() for p in pts]\n\
             print(json.dumps(out))\n"
        );
        let mut child = Command::new("python3")
            .arg("-c")
            .arg(script)
            .arg(&r.1)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .expect("python3 with onnxruntime is required for this integrator test");
        {
            use std::io::Write;
            let stdin = child.stdin.as_mut().expect("stdin");
            stdin
                .write_all(serde_json::to_string(&pts).expect("json").as_bytes())
                .expect("write points");
        }
        let o = child.wait_with_output().expect("python3");
        assert!(o.status.success(), "onnxruntime evaluation failed");
        let ort: Vec<Vec<f64>> = serde_json::from_slice(&o.stdout).expect("ort json");
        for (p, y) in pts.iter().zip(&ort) {
            let b = ny_proofcheck::ibp::eval_point(&net, p);
            for (j, yo) in y.iter().enumerate() {
                let mid = 0.5 * b.lo[net.output][j] + 0.5 * b.hi[net.output][j];
                let rel = (mid - yo).abs() / yo.abs().max(1.0);
                assert!(
                    rel <= 1e-4,
                    "{needle} y{j}: checker {mid} vs ORT {yo} (rel {rel:e})"
                );
            }
        }
        eprintln!("[proof-check-real] lowering matches ORT on {needle}");
    }
}

/// Integrator: main tree.
#[test]
fn proof_check_vnnlib_reader_agrees_with_the_lane_parse_on_the_corpus() {
    // The checker's own VNN-LIB reader against ny-onnx's trusted parser (the
    // G3 reference) on every cifar100 / tinyimagenet property: the same box
    // (the checker's outward box contains the parsed box, within one ulp) and
    // the same disjuncts. A disagreement is a U1 finding.
    use ny_onnx::vnnlib::{load_vnnlib, OutputConstraint};
    let mut n = 0usize;
    for cat in ["cifar100_2024", "tinyimagenet_2024"] {
        let dir = bench(cat).join("vnnlib");
        let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
            .expect("vnnlib dir")
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "vnnlib"))
            .collect();
        files.sort();
        for f in files {
            let mine = ny_proofcheck::vnnlib::parse(&std::fs::read_to_string(&f).expect("read"))
                .unwrap_or_else(|e| panic!("{}: checker refused: {}", f.display(), e.0));
            let theirs = load_vnnlib(&f).expect("ny-onnx parses it");
            assert_eq!(mine.regions.len(), 1, "{}", f.display());
            let reg = &mine.regions[0];
            assert_eq!(reg.lo_out.len(), theirs.input_bounds.len());
            for (i, (l, u)) in theirs.input_bounds.iter().enumerate() {
                assert!(
                    reg.lo_out[i] <= *l && *u <= reg.hi_out[i],
                    "{} X_{i}",
                    f.display()
                );
                assert!(l.next_down() <= reg.lo_out[i] && reg.hi_out[i] <= u.next_up());
            }
            let mut theirs_canon: Vec<String> = theirs
                .output_constraint_clauses
                .iter()
                .map(|clause| {
                    let mut atoms: Vec<String> = clause
                        .iter()
                        .map(|c| match *c {
                            OutputConstraint::GreaterEq(a, b) => {
                                ny_proofcheck::vnnlib::margin_clause(b, a)
                            }
                            OutputConstraint::LessEq(a, b) => {
                                ny_proofcheck::vnnlib::margin_clause(a, b)
                            }
                            ref other => panic!("{}: unexpected {other:?}", f.display()),
                        })
                        .collect();
                    atoms.sort();
                    atoms.join("&")
                })
                .collect();
            theirs_canon.sort();
            let mut mine_canon: Vec<String> =
                reg.disjuncts.iter().map(|d| d.canonical.clone()).collect();
            mine_canon.sort();
            assert_eq!(mine_canon, theirs_canon, "{}", f.display());
            n += 1;
        }
    }
    assert!(n > 100, "only {n} properties compared: VOID");
    eprintln!("[proof-check-real] vnnlib agreement on {n} properties");
}

/// `(tag, category, layout, onnx needle, vnnlib needle, expected)` of the
/// legacy rows (`scratchpad/legacy-bundle`: on the ENG14 pin
/// ny_eng14_8c2100b7f every one published `unsat` through the door named here
/// and its bundle was refused as noted; on fix/legacy-bundle-wellformed
/// a87ff7dfa every bundle is well-formed). `expected` is `unconfirmed` (exit
/// 10, or 0 where a lane tape verifies) or `scope:<code>`: the CHECKER's own
/// scope (exit 31 with that one reason, decided only AFTER the bundle parsed,
/// so the bundle is well-formed) — pinned so a lifted or a new limit shows up.
/// The checker's own lowering (b2-checker-lowering, 07f72c4a0: static
/// Reshape/Flatten, constant-index Slice/Gather, Concat, MatMul with a
/// constant on either side) lifted the nn4sys, sbench and linz limits: those
/// bundles now load and check `unconfirmed` (measured 31 -> 10 on the
/// legacy-bundle writer's bundles); ml4acopf, cctsdb and lsnc stay
/// `scope:onnx_lowering` with their NEXT refusal (Sigmoid, Cast, a
/// constant-only MatMul).
/// The `iso` row's ONNX field is the dual-network tuple of its
/// `instances.csv` and runs from its benchmark directory.
const LEGACY_ROWS: &[(&str, &str, &str, &str, &str, &str)] = &[
    // sequential engine, BetaCrownVerdict (ENG14 pin: missing_sites)
    (
        "safenlp",
        "safenlp_2024",
        ".",
        "medical/perturbations_0.onnx",
        "medical/hyperrectangle_1098.vnnlib",
        "unconfirmed",
    ),
    (
        "satrelu",
        "sat_relu",
        "1.0",
        "unsat_v5_c21_k81.onnx",
        "unsat_v5_c21_k81.vnnlib",
        "unconfirmed",
    ),
    (
        "malb",
        "malbeware",
        "1.0",
        "malware_malimg_family_scaled_linear-25.onnx",
        "malbeware_family-Adialer.C_label-0_eps-1_idx-3.vnnlib",
        "unconfirmed",
    ),
    // graph lane with no tape (root-only fast paths), BetaCrownVerdict (ENG14 pin: missing_sites)
    (
        "metaroom",
        "metaroom_2023",
        "1.0",
        "4cnn_ry_41_6_no_custom_OP.onnx",
        "spec_idx_19_eps_0.00000436.vnnlib",
        "unconfirmed",
    ),
    (
        "collins",
        "collins_rul_cnn_2022",
        "1.0",
        "NN_rul_small_window_20.onnx",
        "robustness_2perturbations_delta10_epsilon10_w20.vnnlib",
        "unconfirmed",
    ),
    (
        "nn4sys",
        "nn4sys",
        "1.0",
        "pensieve_small_simple.onnx",
        "pensieve_simple_21.vnnlib",
        "unconfirmed",
    ),
    (
        "ml4acopf",
        "ml4acopf_2024",
        "1.0",
        "14_ieee_ml4acopf.onnx",
        "14_ieee_prop3.vnnlib",
        "scope:onnx_lowering",
    ),
    (
        "sbench",
        "soundnessbench_2026",
        "1.0",
        "model_residual.onnx",
        "properties/property_059.vnnlib",
        "unconfirmed",
    ),
    (
        "linz",
        "linearizenn_2024",
        "1.0",
        "AllInOne_10_10.onnx",
        "prop_10_10_2.vnnlib",
        "unconfirmed",
    ),
    // the margin-row lane's partial stream under the internal verifier's legacy verdict (ENG14 pin: unsupported, uncovered_source)
    (
        "cct",
        "challenging_certified_training_2026",
        "1.0",
        "cifar10_eps8_cnn7.onnx",
        "cifar10_eps8_cnn7/cifar10_eps8_cnn7_idx5683_sample1.vnnlib",
        "unconfirmed",
    ),
    // CctsdbSubbox (ENG14 pin: missing_sites)
    (
        "cctsdb",
        "cctsdb_yolo_2023",
        "1.0",
        "patch-1.onnx",
        "spec_onnx_patch-1_idx_02037_0.vnnlib",
        "scope:onnx_lowering",
    ),
    // lane tapes under the legacy door (well-formed on the pin too; checker scope)
    (
        "lsnc",
        "lsnc_relu",
        "1.0",
        "relu_quadrotor2d_state.onnx",
        "quadrotor2d_state_23.vnnlib",
        "scope:onnx_lowering",
    ),
    // NonlinearIntervalBab, VNN-LIB 2.0, unrenderable: v2.1 (ENG14 pin: missing_sites)
    (
        "acc_nl",
        "adaptive_cruise_control_non_linear_2026",
        "2.0",
        "acc-2000000-64-64-64-64-retrain-100000-200000-0.9.onnx",
        "instance_13.vnnlib",
        "unconfirmed",
    ),
    // RelationalImplication, dual network: v2.1 (ENG14 pin: unknown_field `error` in the header)
    (
        "iso",
        "isomorphic_acasxu_2026",
        "2.0",
        "ACASXU_run2a_1_2_batch_2000_perturbed_1.onnx",
        "instance_1.vnnlib",
        "unconfirmed",
    ),
];

/// `(category, onnx field, vnnlib, budget, cwd)` of a legacy row.
fn legacy_row(
    cat: &str,
    layout: &str,
    onnx: &str,
    vnn: &str,
) -> (String, String, PathBuf, String, PathBuf) {
    let dir = repo_root()
        .join("benchmarks/vnncomp2026/benchmarks")
        .join(cat)
        .join(layout);
    assert!(
        dir.join("instances.csv").is_file(),
        "{} is missing: these integrator tests need the MAIN tree's gitignored benchmarks/ \
         (a git worktree has none — the CLAUDE.md worktree trap)",
        dir.display()
    );
    let csv = std::fs::read_to_string(dir.join("instances.csv")).expect("instances.csv");
    let line = csv
        .lines()
        .find(|l| l.contains(onnx) && l.contains(vnn))
        .unwrap_or_else(|| panic!("{onnx} / {vnn} is not in {cat}/{layout}/instances.csv"));
    let budget = line.rsplit(',').next().expect("budget").trim().to_string();
    if line.starts_with('"') {
        // A dual-network row: the quoted ONNX tuple, resolved from `dir`.
        let field = line[1..line.rfind("\",").expect("quoted field")].to_string();
        (
            cat.to_string(),
            field,
            dir.join("vnnlib").join(vnn),
            budget,
            dir,
        )
    } else {
        let f: Vec<&str> = line.split(',').collect();
        (
            cat.to_string(),
            dir.join(f[0]).display().to_string(),
            dir.join(f[1]),
            budget,
            repo_root(),
        )
    }
}

/// Integrator: main tree, release binary, box. Every legacy row's Proof
/// bundle is well-formed: `ny proof-check` exits 10 (or 0), NEVER 30.
#[test]
fn proof_check_real_legacy_lane_bundles_are_well_formed() {
    let out = tempfile::tempdir().expect("tempdir");
    let mut ran = 0usize;
    for (tag, cat, layout, onnx, vnn, expected) in LEGACY_ROWS {
        let (cat, field, vnnlib, budget, cwd) = legacy_row(cat, layout, onnx, vnn);
        let results = out.path().join(format!("{tag}.txt"));
        let o = ny()
            .current_dir(&cwd)
            .args(["vnncomp", "v1", &cat, &field])
            .arg(&vnnlib)
            .arg(&results)
            .arg(&budget)
            .args(["--proof-profile", "proof"])
            .output()
            .expect("run ny vnncomp v1");
        assert!(
            o.status.success(),
            "{tag}: ny vnncomp v1 failed: {}",
            o.status
        );
        let stderr = String::from_utf8_lossy(&o.stderr).into_owned();
        let first = std::fs::read_to_string(&results)
            .unwrap_or_default()
            .lines()
            .next()
            .unwrap_or("")
            .trim()
            .to_string();
        let door = stderr
            .lines()
            .find(|l| l.starts_with("[cover-gate] legacy-uncovered authority="))
            .unwrap_or("")
            .to_string();
        let artifact = stderr
            .lines()
            .find(|l| l.starts_with("[proof-artifact] artifact="))
            .unwrap_or("")
            .to_string();
        eprintln!("[proof-check-real] legacy {tag}: published `{first}` {door} {artifact}");
        if first != "unsat" {
            eprintln!("[proof-check-real] legacy {tag}: no unsat this run (not an emergency)");
            continue;
        }
        assert!(
            artifact.starts_with("[proof-artifact] artifact=written "),
            "{tag}: a published unsat must leave a written bundle: {artifact}"
        );
        let mut bundle = results.clone().into_os_string();
        bundle.push(".proof.jsonl");
        let bundle = PathBuf::from(bundle);
        let (rc, done) = check(&bundle, &[]);
        eprintln!("[proof-check-real] legacy {tag}: {done}");
        assert_ne!(
            rc, 30,
            "SOUNDNESS-EMERGENCY PROTOCOL: {tag} bundle is MALFORMED: {done}"
        );
        if let Some(code) = expected.strip_prefix("scope:") {
            let mut report = bundle.clone().into_os_string();
            report.push(".check.json");
            let report: serde_json::Value =
                serde_json::from_slice(&std::fs::read(&report).expect("report")).expect("json");
            let codes: Vec<&str> = report["reasons"]
                .as_array()
                .map(|r| r.iter().filter_map(|x| x["code"].as_str()).collect())
                .unwrap_or_default();
            assert!(
                rc == 31 && codes == [code],
                "{tag}: expected the checker's scope limit `{code}` (exit 31), got rc={rc} \
                 reasons={codes:?}: {done}"
            );
        } else {
            assert!(
                ![20, 21, 31, 40].contains(&rc),
                "SOUNDNESS-EMERGENCY PROTOCOL: {tag} checked rc={rc}: {done}"
            );
            assert!(
                [0, 10].contains(&rc),
                "{tag}: a legacy bundle checks unconfirmed (10) or verified (0): {done}"
            );
        }
        ran += 1;
    }
    eprintln!(
        "[proof-check-real] legacy summary checked={ran} registered={}",
        LEGACY_ROWS.len()
    );
    assert!(
        ran * 2 >= LEGACY_ROWS.len(),
        "only {ran} legacy rows published unsat: VOID"
    );
}
