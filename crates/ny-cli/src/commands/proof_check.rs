// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `ny proof-check` (ENGINE_PROGRAM §1.7 item B2, increment B2.1): the
//! offline, downgrade-only proof-bundle checker.
//!
//! This is the ONLY file in `ny-cli` that names the checker crate
//! (`ny_proofcheck`), and the only caller of it is the `Commands::ProofCheck`
//! dispatch in `main.rs`. The report it writes (`<BUNDLE>.check.json`) is
//! read by NO verdict path — it exists for the banking rule: an
//! uncorroborated margin-row `unsat` banks only with a Proof-profile bundle of
//! the same onnx/vnnlib sha256 from the same pin plus `PROOF_CHECK_DONE rc=0`
//! (runners match `*"PROOF_CHECK_DONE rc=0 "*`, never a bare `rc=`), with the
//! report's sha256 recorded in the bank provenance. All pinned by
//! `tests/proof_check_ratchet.rs`.
//!
//! Observables (stderr): `[proof-check] verdict=<v> exit=<code> ...` and the
//! `PROOF_CHECK_DONE rc=<code> verdict=<v> nodes= confirmed= unconfirmed=
//! cover=<ok|gap|unconfirmed> refuted=<0|1> wall_s= report= report_sha256=`
//! line (also the last stdout line unless `--json`, which prints the report).

use anyhow::{anyhow, Result};

use crate::subcommands::{OnOff, ProofCheckArgs};

/// Run the checker; the returned code is the process exit code (0 verified,
/// 10 unconfirmed, 20 invalid, 21 refuted, 30 malformed, 31 unsupported,
/// 40 checker fault). An unreadable input or unwritable report is an
/// operational `Err` (exit 4).
pub(crate) fn handle_proof_check(args: &ProofCheckArgs) -> Result<i32> {
    let mut a = ny_proofcheck::CheckArgs::new(args.bundle.clone());
    a.onnx.clone_from(&args.onnx);
    a.vnnlib.clone_from(&args.vnnlib);
    a.report.clone_from(&args.report);
    a.threads = args.threads;
    a.mem_gib = args.mem_gib;
    a.budget_s = args.budget_s;
    a.refine = args.refine == OnOff::On;
    a.refute_points = args.refute_points;
    a.checker_build = crate::build_id::current_line().to_string();
    let out = ny_proofcheck::check_bundle(&a).map_err(|e| anyhow!("proof-check: {e}"))?;
    eprintln!("{}", out.summary_line);
    eprintln!("{}", out.done_line);
    if args.json {
        println!("{}", out.report_json);
    } else {
        println!("{}", out.done_line);
    }
    Ok(out.exit_code)
}

/// Test faces of the checker for this module's test children: this file is
/// the ONE ny-cli source allowed to name the checker
/// (`tests/proof_check_ratchet.rs`), so a child test module reaches it
/// through `super::test_faces` and never names it.
#[cfg(test)]
pub(super) mod test_faces {
    use std::path::Path;

    /// A check's verdict token, report and DONE line.
    pub(crate) struct Checked {
        pub(crate) verdict: &'static str,
        pub(crate) report_json: String,
        pub(crate) done_line: String,
    }

    /// Check a bundle with the CLI's test settings (2 threads, 120 s).
    pub(crate) fn check(path: &Path) -> Checked {
        let mut args = ny_proofcheck::CheckArgs::new(path.to_path_buf());
        args.threads = Some(2);
        args.budget_s = 120.0;
        let o = ny_proofcheck::check_bundle(&args).expect("operational");
        Checked {
            verdict: o.verdict.token(),
            report_json: o.report_json,
            done_line: o.done_line,
        }
    }

    /// The checker's own canonical disjunct strings of a VNN-LIB text, per
    /// region.
    pub(crate) fn checker_disjuncts(text: &str) -> Vec<Vec<String>> {
        ny_proofcheck::vnnlib::parse(text)
            .expect("the checker parses it")
            .regions
            .iter()
            .map(|r| r.disjuncts.iter().map(|d| d.canonical.clone()).collect())
            .collect()
    }

    /// [`checker_disjuncts`], or `None` when the checker's own reader refuses
    /// the text (its `vnnlib_form` scope limit) — review(legacy-bundle): the
    /// corpus pin of the writer's fallback clauses
    /// (`legacy_corpus`, external-vnncomp lane only) reaches the reader only
    /// through here.
    #[cfg(feature = "external-vnncomp")]
    pub(crate) fn checker_disjuncts_if_parsed(text: &str) -> Option<Vec<Vec<String>>> {
        ny_proofcheck::vnnlib::parse(text).ok().map(|s| {
            s.regions
                .iter()
                .map(|r| r.disjuncts.iter().map(|d| d.canonical.clone()).collect())
                .collect()
        })
    }
}

/// B2.2 end to end: every lane's proof tape through the real loader, lane,
/// v2 writer and independent checker.
#[cfg(test)]
mod lanes_e2e;

/// B2.3 end to end: the margin-row lane's remaining routes (frozen-dual
/// prunes, wave leaves, classwise class trees, nested epoch subtrees) through
/// the real driver, the real v2 writer and the independent checker, plus
/// mutations of a real frozen-dual bundle.
#[cfg(test)]
mod routes_e2e;

/// review(legacy-bundle): the writer's fallback clauses (the legacy
/// completion's `clauses`) against the checker's own reader on the real
/// corpus (external-vnncomp lane: it reads the gitignored `benchmarks/`).
#[cfg(all(test, feature = "external-vnncomp"))]
mod legacy_corpus;

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::time::{Duration, Instant};

    use ny_onnx::onnx_proto::tensor_shape_proto::{dimension, Dimension};
    use ny_onnx::onnx_proto::{
        AttributeProto, GraphProto, ModelProto, NodeProto, TensorProto, TensorShapeProto,
        TensorTypeProto, TypeProto, ValueInfoProto,
    };
    use ny_propagate::margin_row::MarginRowOutcome;
    use prost::Message;

    use crate::commands::margin_row_bab::{extract_twin_spec_named, parse_vnnlib_robustness};
    use crate::commands::proof_bundle::{
        bundle_path, enable_hint_capture_for_test, margin_clause, sites_record, BundleWriter,
        Completion, Header, VerdictInfo, FINALISE_WAIT,
    };

    fn node(op: &str, inputs: &[&str], outputs: &[&str], attrs: Vec<AttributeProto>) -> NodeProto {
        NodeProto {
            op_type: op.to_owned(),
            input: inputs.iter().map(|s| (*s).to_owned()).collect(),
            output: outputs.iter().map(|s| (*s).to_owned()).collect(),
            attribute: attrs,
            ..Default::default()
        }
    }

    fn ints(name: &str, v: &[i64]) -> AttributeProto {
        AttributeProto {
            name: name.to_owned(),
            ints: v.to_vec(),
            r#type: 7,
            ..Default::default()
        }
    }

    fn int(name: &str, v: i64) -> AttributeProto {
        AttributeProto {
            name: name.to_owned(),
            i: Some(v),
            r#type: 2,
            ..Default::default()
        }
    }

    fn f32_tensor(name: &str, dims: &[i64], v: &[f32]) -> TensorProto {
        TensorProto {
            name: name.to_owned(),
            dims: dims.to_vec(),
            data_type: 1,
            float_data: v.to_vec(),
            ..Default::default()
        }
    }

    /// A margin-row-family net (2x6x6 input, a residual block, two trunk
    /// Relus, a 5-wide head, 4 classes) with deterministic f32 weights.
    fn twin_model(seed: u64) -> ModelProto {
        let mut s = seed | 1;
        let mut w = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    s = s
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    ((((s >> 33) % 1001) as f64 / 1000.0 - 0.5) * 0.7) as f32
                })
                .collect()
        };
        let dim = |d: i64| Dimension {
            value: Some(dimension::Value::DimValue(d)),
        };
        let vi = |name: &str, dims: &[i64]| ValueInfoProto {
            name: name.to_owned(),
            r#type: Some(TypeProto {
                tensor_type: Some(TensorTypeProto {
                    elem_type: 1,
                    shape: Some(TensorShapeProto {
                        dim: dims.iter().map(|&d| dim(d)).collect(),
                    }),
                }),
            }),
        };
        let initializer = vec![
            f32_tensor("c1w", &[4, 2, 3, 3], &w(72)),
            f32_tensor("c1b", &[4], &w(4)),
            f32_tensor("c2w", &[4, 4, 3, 3], &w(144)),
            f32_tensor("c2b", &[4], &w(4)),
            f32_tensor("g1w", &[5, 36], &w(180)),
            f32_tensor("g1b", &[5], &w(5)),
            f32_tensor("g2w", &[4, 5], &w(20)),
            f32_tensor("g2b", &[4], &w(4)),
        ];
        let nodes = vec![
            node(
                "Conv",
                &["x", "c1w", "c1b"],
                &["c1"],
                vec![ints("strides", &[2, 2]), ints("pads", &[1, 1, 1, 1])],
            ),
            node("Relu", &["c1"], &["r1"], Vec::new()),
            node(
                "Conv",
                &["r1", "c2w", "c2b"],
                &["c2"],
                vec![ints("pads", &[1, 1, 1, 1])],
            ),
            node("Add", &["c2", "r1"], &["a1"], Vec::new()),
            node("Relu", &["a1"], &["r2"], Vec::new()),
            node("Flatten", &["r2"], &["f"], vec![int("axis", 1)]),
            node(
                "Gemm",
                &["f", "g1w", "g1b"],
                &["g1"],
                vec![int("transB", 1)],
            ),
            node("Relu", &["g1"], &["h"], Vec::new()),
            node("Gemm", &["h", "g2w", "g2b"], &["y"], vec![int("transB", 1)]),
        ];
        ModelProto {
            ir_version: 8,
            graph: Some(GraphProto {
                node: nodes,
                initializer,
                input: vec![vi("x", &[1, 2, 6, 6])],
                output: vec![vi("y", &[1, 4])],
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn robustness_vnnlib(lo: &[f64], hi: &[f64], n_y: usize, t: usize) -> String {
        let mut s = String::new();
        for i in 0..lo.len() {
            s.push_str(&format!("(declare-const X_{i} Real)\n"));
        }
        for j in 0..n_y {
            s.push_str(&format!("(declare-const Y_{j} Real)\n"));
        }
        for i in 0..lo.len() {
            s.push_str(&format!(
                "(assert (>= X_{i} {:?}))\n(assert (<= X_{i} {:?}))\n",
                lo[i], hi[i]
            ));
        }
        s.push_str("(assert (or\n");
        for j in (0..n_y).filter(|&j| j != t) {
            s.push_str(&format!("    (and (>= Y_{j} Y_{t}))\n"));
        }
        s.push_str("))\n");
        s
    }

    /// Run the REAL margin-row lane with the REAL v2 writer on a synthetic
    /// net, then the checker on the bundle it produced. Returns the checker's
    /// verdict, or `None` when the lane did not prove the property.
    fn lane_then_check(
        dir: &Path,
        tag: &str,
        onnx: &Path,
        eps: f64,
    ) -> Option<ny_proofcheck::CheckOutcome> {
        let net = ny_proofcheck::testkit::net_from_onnx(&std::fs::read(onnx).expect("onnx"))
            .expect("the checker lowers the net");
        let centre: Vec<f64> = (0..72)
            .map(|i| f64::from((i * 37 % 64) as u32) / 64.0)
            .collect();
        let yb = ny_proofcheck::ibp::eval_point(&net, &centre);
        let (yl, yh) = (&yb.lo[net.output], &yb.hi[net.output]);
        let t = (0..4)
            .max_by(|&a, &b| (yl[a] + yh[a]).total_cmp(&(yl[b] + yh[b])))
            .expect("a class");
        let lo: Vec<f64> = centre.iter().map(|c| c - eps).collect();
        let hi: Vec<f64> = centre.iter().map(|c| c + eps).collect();
        let vnnlib = dir.join(format!("{tag}.vnnlib"));
        std::fs::write(&vnnlib, robustness_vnnlib(&lo, &hi, 4, t)).expect("vnnlib");
        let (spec, names) = extract_twin_spec_named(onnx).expect("the lane admits the net");
        let (plo, phi, pt, adv) = parse_vnnlib_robustness(&vnnlib).expect("the lane parses it");
        assert_eq!(pt, t);
        let results = dir.join(format!("{tag}.txt"));
        let header = Header {
            profile: ny_core::proof_profile::ResolvedProofProfile::resolve(
                ny_core::proof_profile::EntryPoint::VnncompV1,
                Some(ny_core::proof_profile::ProofProfile::Proof),
            ),
            onnx: onnx.to_path_buf(),
            vnnlib,
        };
        let (writer, sink) = BundleWriter::create(&results, header).expect("bundle");
        writer.clauses(pt, &adv);
        let deadline = Instant::now() + Duration::from_secs(60);
        let (out, _sink) = ny_propagate::certify::with_thread_sink(Box::new(sink), || {
            ny_propagate::certify::stream_sites(&sites_record(&names));
            ny_propagate::margin_row::run_margin_row_lane(
                &spec,
                &plo,
                &phi,
                pt,
                &adv,
                Some(deadline),
                4000,
            )
        });
        let MarginRowOutcome::Unsat(stats, token) = out else {
            let _ = writer.finalise("unknown", None, FINALISE_WAIT);
            return None;
        };
        eprintln!(
            "[e2e] eps={eps} lane unsat: expansions={} tree_classes={:?}",
            stats.expansions, stats.tree_classes
        );
        let _ = writer.finalise(
            "unsat",
            Some(VerdictInfo {
                source: token.source().into(),
                min_margin: token.min_margin(),
                completion: Completion::OwnStream,
            }),
            FINALISE_WAIT,
        );
        let mut args = ny_proofcheck::CheckArgs::new(bundle_path(&results));
        args.threads = Some(2);
        args.budget_s = 120.0;
        Some(ny_proofcheck::check_bundle(&args).expect("operational"))
    }

    /// END TO END: bundles the REAL margin-row lane and the REAL v2 writer
    /// produce on a synthetic net are checked by `ny proof-check`'s engine:
    /// every proved instance is verified or (for a node the checker's weaker
    /// relaxation cannot re-derive) unconfirmed — never invalid, refuted,
    /// malformed, unsupported or a checker fault, which would be an
    /// emitter/checker disagreement.
    #[test]
    fn proof_check_end_to_end_on_a_real_margin_row_bundle() {
        enable_hint_capture_for_test();
        let dir = tempfile::tempdir().expect("tempdir");
        let onnx = dir.path().join("twin.onnx");
        std::fs::write(&onnx, twin_model(0x5eed).encode_to_vec()).expect("onnx");
        let mut proved = 0usize;
        let mut verified = 0usize;
        let mut tree_nodes = 0u64;
        // A sweep from root-only proofs to boxes that need a tree.
        let widths = [
            1.0 / 512.0,
            1.0 / 128.0,
            1.0 / 64.0,
            3.0 / 160.0,
            1.0 / 48.0,
            7.0 / 320.0,
            3.0 / 128.0,
            1.0 / 40.0,
            7.0 / 256.0,
            1.0 / 32.0,
        ];
        let mut bound_nodes = 0usize;
        for (k, eps) in widths.into_iter().enumerate() {
            let Some(o) = lane_then_check(dir.path(), &format!("e{k}"), &onnx, eps) else {
                continue;
            };
            proved += 1;
            eprintln!("[e2e] eps={eps} {}", o.done_line);
            assert!(
                matches!(
                    o.verdict,
                    ny_proofcheck::Verdict::Verified | ny_proofcheck::Verdict::Unconfirmed
                ),
                "eps={eps}: the checker disagrees with the emitter: {}",
                o.report_json
            );
            if o.verdict == ny_proofcheck::Verdict::Verified {
                verified += 1;
            }
            let r = ny_cert::json::from_str(&o.report_json).expect("report json");
            tree_nodes += r["nodes"]["total"].as_u64().unwrap_or(0);
            let bundle = std::fs::read_to_string(dir.path().join(format!("e{k}.txt.proof.jsonl")))
                .expect("bundle");
            bound_nodes += bundle
                .lines()
                .filter(|l| l.contains("\"close\":\"bound\""))
                .count();
        }
        assert!(
            proved > 0,
            "the lane proved none of the instances: the test is VOID"
        );
        assert!(verified > 0, "no real bundle verified end to end");
        eprintln!(
            "[e2e] proved={proved} verified={verified} nodes={tree_nodes} tree_leaf_nodes={bound_nodes}"
        );
    }

    /// The producer's clause strings are the checker's canonical form, for
    /// both robustness spellings (`(>= Y_j Y_t)` and `(<= Y_t Y_j)`) and for
    /// either index order.
    #[test]
    fn proof_bundle_v2_clauses_use_the_checker_canonical_form() {
        for (t, j) in [(0usize, 3usize), (3, 0), (2, 7), (9, 1)] {
            for spelling in [format!("(>= Y_{j} Y_{t})"), format!("(<= Y_{t} Y_{j})")] {
                let text = format!(
                    "(declare-const X_0 Real)\n{}\n(assert (>= X_0 0))\n(assert (<= X_0 1))\n\
                     (assert (or (and {spelling})))\n",
                    (0..10)
                        .map(|k| format!("(declare-const Y_{k} Real)"))
                        .collect::<Vec<_>>()
                        .join("\n")
                );
                let spec = ny_proofcheck::vnnlib::parse(&text).expect("the checker parses it");
                assert_eq!(
                    spec.regions[0].disjuncts[0].canonical,
                    margin_clause(t, j),
                    "t={t} j={j} spelling {spelling}"
                );
                assert_eq!(
                    ny_proofcheck::vnnlib::margin_clause(t, j),
                    margin_clause(t, j)
                );
            }
        }
    }

    #[test]
    fn default_report_path_is_next_to_the_bundle() {
        let p = ny_proofcheck::default_report_path(&PathBuf::from("/r/x.proof.jsonl"));
        assert_eq!(
            p,
            PathBuf::from(format!("/r/x.proof.jsonl.{}", "check.json"))
        );
    }
}

#[cfg(test)]
mod prepared_export_replay_tests {
    use ny_proofcheck::{check_bundle, CheckArgs, CheckOutcome, Verdict};
    use std::path::Path;

    fn check(path: &Path) -> CheckOutcome {
        let mut args = CheckArgs::new(path.to_owned());
        args.threads = Some(1);
        args.mem_gib = 1.0 / 32.0;
        args.budget_s = 20.0;
        args.checker_build = "independent-prepared-export-control".into();
        check_bundle(&args).unwrap()
    }
    fn positive(kind: usize) {
        let _env = ny_test_utils::env::lock_env_shared();
        let fixture = crate::commands::proof_bundle::prepared_replay_tests::independent_case(kind);
        for path in &fixture.paths {
            let result = check(path);
            assert_eq!(result.verdict, Verdict::Verified, "{}", result.report_json);
            assert_eq!(result.exit_code, 0);
        }
    }
    fn negative(kind: usize) {
        let _env = ny_test_utils::env::lock_env_shared();
        let fixture = crate::commands::proof_bundle::prepared_replay_tests::independent_case(kind);
        for path in &fixture.paths {
            let result = check(path);
            assert_ne!(result.verdict, Verdict::Verified, "{}", result.report_json);
            assert_ne!(result.exit_code, 0);
        }
    }
    #[test]
    fn prepared_export_independent_native_root_original_arithmetic_and_strict_objective() {
        positive(0);
    }
    #[test]
    fn prepared_export_independent_native_tree_actual_two_child_phase_cover() {
        positive(1);
    }
    #[test]
    fn prepared_export_independent_complete_65_classes_and_nonzero_original_target() {
        positive(2);
    }
    #[test]
    fn prepared_export_independent_missing_child_duplicate_phase_and_partial_classes_refuse() {
        negative(3);
    }
    #[test]
    fn prepared_export_independent_huge_cached_margin_and_source_tag_cannot_prove_negative_original(
    ) {
        negative(4);
    }
    #[test]
    fn prepared_export_independent_foreign_site_width_and_strict_clause_remain_quarantined() {
        negative(5);
    }
}
