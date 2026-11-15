// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #admission-precision unit tests (run with `--bin ny`, never `--lib`).
//!
//! * SOURCE RATCHETS: every lane check refactored onto
//!   `ny_core::router::admission` still calls it (and no longer spells the
//!   inline expression it replaced), the shared checks and the one
//!   routing-shape extractor have EXACTLY the non-test callers listed here,
//!   and the scored entry point hands the router the extractor's read.
//! * THE ROUTER READS THE SHAPE: the start context's property read becomes
//!   the facts' archetype, shape and counts, and the banner names it.
//! * SITES ON REAL FIXTURE ROWS: the four consumers' census decisions over
//!   tracked and synthetic models × the corpus's property forms (golden).
//!
//! Hermetic: tracked `tests/models/` and properties written to a temporary
//! directory; no `benchmarks/` dependency.

use ny_core::router::decl::{AdmitSite, OpKind};
use ny_core::router::route::CENSUS_WIDEST_ENVELOPE;

use super::census_tests::{
    call_sites, cell, census_cells, code_only, fn_body, halfspace_property, read,
    robustness_property, row, twin_model_bytes, workspace_root, World,
};
use super::{policy_input, registry, StartContext};
use crate::commands::beta_crown::spec_shape::{read_spec, SpecRead};
use crate::plan_resolver::ModelFacts;

/// SOURCE RATCHET: each refactored lane check calls the shared predicate on
/// the one extractor's shape, and no longer spells the expression it
/// replaced. Mutation target: an inline copy restored in any of them.
#[test]
fn admission_precision_lanes_call_the_shared_predicates() {
    let dispatch = code_only(&read("crates/ny-cli/src/commands/beta_crown/dispatch.rs"));
    let routes = fn_body(&dispatch, "routes_to_relational_verifier");
    assert!(routes.contains("super::spec_shape::shape_of(vnnlib)"));
    assert!(routes.contains("ny_core::router::admission::relational_verifier("));
    assert!(!routes.contains("output_constraints.len() > 1"));

    let verify = code_only(&read("crates/ny-cli/src/commands/beta_crown/verify/mod.rs"));
    let ledger = fn_body(&verify, "verify_relational_constraints_with_ledger");
    assert!(ledger.contains("if enters_disjunctive_lane(vnnlib) {"));
    assert!(!ledger.contains("has_multi_constraint_disjunction()"));
    assert!(!ledger.contains("has_boxed_clause_disjunction()"));
    assert!(fn_body(&verify, "enters_disjunctive_lane").contains(
        "ny_core::router::admission::disjunctive_lane(&super::spec_shape::shape_of(vnnlib))"
    ));
    let upgrade = fn_body(&verify, "should_upgrade_sequential_conjunction_to_graph");
    assert!(upgrade.contains("ny_core::router::admission::sequential_conjunction_upgrade("));
    assert!(upgrade.contains("super::spec_shape::shape_of(vnnlib)"));
    assert!(!upgrade.contains("normalize_same_lhs_reduction"));
    assert!(fn_body(&verify, "same_lhs_reducible")
        .contains("sequential::normalize_same_lhs_reduction(vnnlib).is_some()"));

    let graph = code_only(&read(
        "crates/ny-cli/src/commands/beta_crown/verify/graph.rs",
    ));
    // #attack-concurrent extracted the lane's proof half (everything after the
    // upfront attack) into `verify_graph_relational_proof`, which the serial
    // order and the concurrent race both call; the route check moved with it.
    let relational = fn_body(&graph, "verify_graph_relational");
    assert!(relational.contains("verify_graph_relational_proof("));
    assert!(!relational.contains("total_constraint_count >= 2 &&"));
    let proof_half = fn_body(&graph, "verify_graph_relational_proof");
    assert!(proof_half.contains("multi_objective_route(vnnlib, use_relu_split)"));
    assert!(!proof_half.contains("total_constraint_count >= 2 &&"));
    let route = fn_body(&graph, "multi_objective_route");
    assert!(route.contains("graph_multi_objective(&super::super::spec_shape::shape_of(vnnlib))"));

    let load = code_only(&read("crates/ny-cli/src/commands/beta_crown/model_load.rs"));
    let load_model = fn_body(&load, "load_model");
    assert!(load_model.contains(".is_some_and(huge_boxed_clause_spec)"));
    assert!(fn_body(&load, "huge_boxed_clause_spec")
        .contains("ny_core::router::admission::huge_boxed_clause_disjunction("));
    assert!(fn_body(&load, "huge_boxed_clause_spec")
        .contains("super::spec_shape::shape_of(vnnlib).boxed_clauses"));
    assert!(load_model.contains("ny_core::router::admission::graph_network("));
    assert!(!load_model.contains(">= 512"));
    assert!(!load_model.contains("needs_graph || use_graph_for_conv"));

    // The D45 refusal classifier is the router's `nonlinear` read.
    let entry = code_only(&read("crates/ny-cli/src/commands/vnncomp.rs"));
    let d45 = fn_body(&entry, "affine_parser_refused_for_representability");
    assert!(d45.contains("spec_shape::affine_parser_refused_for_representability(msg)"));
    assert!(!d45.contains("msg.contains("));
    let shape = code_only(&read("crates/ny-cli/src/commands/beta_crown/spec_shape.rs"));
    let read_fn = fn_body(&shape, "read_spec");
    assert!(read_fn.contains("ny_onnx::vnnlib::load_vnnlib(path)"));
    assert!(read_fn.contains("SpecRead::Affine(shape_of(&spec))"));
    assert!(read_fn.contains("affine_parser_refused_for_representability(&error.to_string())"));
    let extractor = fn_body(&shape, "shape_of");
    for needle in [
        "vnnlib.has_multi_constraint_disjunction()",
        "vnnlib.has_boxed_clause_disjunction()",
        "classify_constraints(vnnlib).has_relational",
        "super::verify::same_lhs_reducible(vnnlib)",
        "super::count_perturbed_inputs(vnnlib)",
    ] {
        assert!(extractor.contains(needle), "the extractor reads `{needle}`");
    }

    // The router composes the SAME checks (ny-core).
    let core = code_only(&read("crates/ny-core/src/router/admission.rs"));
    let site = fn_body(&core, "site_reachable");
    for needle in [
        "reaches_disjunctive_lane",
        "disjunctive_mo_on_graph_spec",
        "sequential_input_split_upgrade",
        "dag_possible(ops, spec)",
        "graph_network_possible(ops, spec)",
    ] {
        assert!(site.contains(needle), "site_reachable composes `{needle}`");
    }
    let decl = code_only(&read("crates/ny-core/src/router/decl.rs"));
    assert!(fn_body(&decl, "evaluate").contains("super::admission::site_reachable("));
}

/// SOURCE RATCHET: the shared checks and the extractor have EXACTLY these
/// non-test callers in the binary. A new caller is a new lane reading the
/// shape (fine — add it here with its ratchet); a missing one is a lane that
/// stopped calling the check the router reads.
#[test]
fn admission_precision_shared_checks_have_exactly_these_callers() {
    let mut sites = call_sites(
        "crates/ny-cli/src",
        &[
            "admission::relational_verifier(",
            "admission::disjunctive_lane(",
            "admission::sequential_conjunction_upgrade(",
            "admission::graph_multi_objective(",
            "admission::huge_boxed_clause_disjunction(",
            "admission::graph_network(",
        ],
    );
    sites.sort();
    sites.dedup();
    let beta = "crates/ny-cli/src/commands/beta_crown";
    assert_eq!(
        sites,
        vec![
            (
                format!("{beta}/dispatch.rs"),
                "routes_to_relational_verifier".to_owned()
            ),
            (
                format!("{beta}/model_load.rs"),
                "huge_boxed_clause_spec".to_owned()
            ),
            (format!("{beta}/model_load.rs"), "load_model".to_owned()),
            (
                format!("{beta}/verify/graph.rs"),
                "multi_objective_route".to_owned()
            ),
            (
                format!("{beta}/verify/mod.rs"),
                "enters_disjunctive_lane".to_owned()
            ),
            (
                format!("{beta}/verify/mod.rs"),
                "should_upgrade_sequential_conjunction_to_graph".to_owned()
            ),
        ],
        "the lanes' calls into ny_core::router::admission"
    );
    let mut readers = call_sites("crates/ny-cli/src", &["spec_shape::read_spec("]);
    readers.sort();
    readers.dedup();
    assert_eq!(
        readers,
        vec![
            (
                "crates/ny-cli/src/commands/vnncomp.rs".to_owned(),
                "handle_vnncomp_command".to_owned()
            ),
            (
                "crates/ny-cli/src/commands/vnncomp_plan.rs".to_owned(),
                "print_router_plan".to_owned()
            ),
        ],
        "the router's start-time property read: the scored start and the plan printer (the \
         census calls it through its own import)"
    );
}

/// THE ROUTER READS THE SHAPE: the start context's read becomes the facts'
/// archetype, shape and counts — on known and on unknown model facts — and
/// the session banner names the archetype.
#[test]
fn admission_precision_router_reads_the_spec_at_start() {
    let world = World::new();
    let robust = world.file("p/robust.vnnlib", robustness_property(72).as_bytes());
    let read_robust = read_spec(&robust);
    let shape = read_robust.shape().expect("an affine property");
    assert_eq!(shape.archetype().token(), "robustness");
    let model = world.file("m/net.onnx", &twin_model_bytes(2));
    let scan = ModelFacts::from_onnx_file(&model).expect("the scan reads the fixture");
    let ctx = StartContext {
        model: Some(&scan),
        spec: read_robust,
        ..super::tests::printer_ctx()
    };
    let input = policy_input(&ctx);
    assert_eq!(input.facts.spec, read_robust.archetype());
    assert_eq!(input.facts.spec_shape, Some(shape));
    assert_eq!(input.facts.input_dims, Some(72));
    assert_eq!(input.facts.p_eff, Some(72));
    assert_eq!(input.facts.classes, Some(4));
    assert_eq!(input.facts.clauses, Some(3));
    let banner = super::telemetry::banner_line(
        super::RouterPolicy::OFF,
        ny_core::router::EffectiveMode::Shadow,
        ny_core::router::EffectiveMode::Shadow,
        "test",
        &input,
    );
    assert!(banner.ends_with(" spec=robustness"), "{banner}");
    // Unknown model facts keep the property's facts (logged; every
    // precondition refuses on unknown facts first).
    let unknown_model = policy_input(&StartContext {
        model: None,
        spec: read_robust,
        ..super::tests::printer_ctx()
    });
    assert!(!unknown_model.facts.is_known());
    assert_eq!(unknown_model.facts.spec_shape, Some(shape));
    // No read: nothing recorded, the banner says so.
    let unread = policy_input(&StartContext {
        model: Some(&scan),
        spec: SpecRead::Unknown("not-read"),
        ..super::tests::printer_ctx()
    });
    assert_eq!(unread.facts.spec, None);
    assert_eq!(unread.facts.spec_shape, None);
    assert_eq!(unread.facts.p_eff, None);
    let banner = super::telemetry::banner_line(
        super::RouterPolicy::OFF,
        ny_core::router::EffectiveMode::Shadow,
        ny_core::router::EffectiveMode::Shadow,
        "test",
        &unread,
    );
    assert!(banner.ends_with(" spec=-"), "{banner}");
}

/// A disjunction of output thresholds (a band) over `n_x` inputs.
fn band_property(n_x: usize) -> String {
    let mut s = halfspace_property(n_x);
    let at = s.rfind("(assert (<= Y_0").expect("the halfspace atom");
    s.truncate(at);
    s.push_str("(assert (or\n    (and (>= Y_0 0.5))\n    (and (<= Y_0 -0.5))\n))\n");
    s
}

/// A single atom comparing two outputs, no `or` (the two-class shape).
fn relational_atom_property(n_x: usize) -> String {
    let mut s = halfspace_property(n_x);
    let at = s.rfind("(assert (<= Y_0").expect("the halfspace atom");
    s.truncate(at);
    s.push_str("(assert (<= Y_0 Y_1))\n");
    s
}

/// SITES ON FIXTURE ROWS (golden): the four consumers' census decisions
/// (CR-RS's own `commit`, pretended sealed under the widest envelope) over a
/// residual conv net, a plain MLP and a plain conv net × robustness, a single
/// threshold, a threshold band and a two-class relational atom — each
/// decline is the site's own reason, and the census equals the router (these
/// parents have no probe).
#[test]
fn admission_precision_sites_on_fixture_rows() {
    let world = World::new();
    let twin = world.file("twin.onnx", &twin_model_bytes(2));
    let mlp = workspace_root().join("tests/models/simple_mlp.onnx");
    let conv = workspace_root().join("tests/models/conv_relu.onnx");
    let r = world.file("r.vnnlib", robustness_property(72).as_bytes());
    let h = world.file("h.vnnlib", halfspace_property(72).as_bytes());
    let b = world.file("b.vnnlib", band_property(72).as_bytes());
    let s = world.file("s.vnnlib", relational_atom_property(72).as_bytes());
    let rows = [
        row("twin_r", &twin, &r),
        row("twin_h", &twin, &h),
        row("twin_b", &twin, &b),
        row("twin_s", &twin, &s),
        row("mlp_r", &mlp, &r),
        row("mlp_h", &mlp, &h),
        row("mlp_b", &mlp, &b),
        row("mlp_s", &mlp, &s),
        row("conv_r", &conv, &r),
        row("conv_h", &conv, &h),
    ];
    let (_, cells) = census_cells(&rows, &world.options("sites"));
    // (row, fwd-root-screen, vit-cover-certify, cgan-child-zono, la-warm-frontier)
    let golden: [(&str, [u8; 4]); 10] = [
        ("twin_r", [1, 1, 1, 1]),
        ("twin_h", [0, 0, 1, 1]),
        ("twin_b", [1, 1, 1, 1]),
        ("twin_s", [0, 0, 1, 1]),
        ("mlp_r", [0, 1, 1, 0]),
        ("mlp_h", [0, 0, 0, 0]),
        ("mlp_b", [0, 1, 1, 0]),
        ("mlp_s", [0, 0, 0, 0]),
        ("conv_r", [1, 1, 1, 0]),
        ("conv_h", [0, 0, 1, 0]),
    ];
    let caps = [
        ("fwd-root-screen", AdmitSite::DisjunctiveLaneGraph),
        ("vit-cover-certify", AdmitSite::DisjunctiveMultiObjective),
        ("cgan-child-zono", AdmitSite::GraphInputSplit),
        ("la-warm-frontier", AdmitSite::DagGraph),
    ];
    let mut wrong = Vec::new();
    for (id, admits) in golden {
        for ((cap, site), want) in caps.iter().zip(admits) {
            let c = cell(&cells, id, cap);
            let got = (c[0].as_str(), c[1].as_str(), c[2].as_str());
            // An admitted row asks its parent (none has a probe: `-`); a
            // declined one never pays for a probe (`skipped`).
            let (expect, probe) = if want == 1 {
                ("1", "-")
            } else {
                ("0", "skipped")
            };
            if got != (expect, expect, probe) {
                wrong.push(format!("{id}/{cap}: {c:?}, want admit={expect}"));
            }
            if want == 0 && c[3] != format!("admit:decline:{}", site.decline()) {
                wrong.push(format!("{id}/{cap}: reasons {:?}", c[3]));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "site decisions on the fixture rows:\n  {}",
        wrong.join("\n  ")
    );
    // Every row's spec column is the entry point's read.
    for (id, property) in [("twin_r", &r), ("mlp_h", &h), ("mlp_b", &b), ("mlp_s", &s)] {
        assert_eq!(
            cell(&cells, id, "fwd-root-screen")[6],
            read_spec(property).token()
        );
    }
    assert!(cell(&cells, "mlp_s", "fwd-root-screen")[6].starts_with("robustness c=1 a=1 "));
    assert!(cell(&cells, "mlp_b", "fwd-root-screen")[6].starts_with("disjunctive c=2 a=2 "));
}

/// The registry declares a site for exactly the four consumers the census
/// review named (and, since #kfsb-production, the single-shot kFSB scorer),
/// each over the affine archetypes; no other routable capability changed its
/// precondition's operator requirement.
#[test]
fn admission_precision_registry_sites() {
    let mut sited = Vec::new();
    for decl in registry::REGISTRY {
        let Some(routable) = decl.routable else {
            continue;
        };
        if let Some(site) = routable.admit_when.site {
            assert_eq!(
                routable.admit_when.specs,
                ny_core::router::decl::SpecSet::AFFINE,
                "{}: the β-CROWN lanes parse every affine archetype",
                decl.id.as_str()
            );
            assert_eq!(
                routable.admit_when.ops_any.0,
                0,
                "{}: the site carries the operator terms",
                decl.id.as_str()
            );
            sited.push((decl.id.as_str(), site));
        }
    }
    assert_eq!(
        sited,
        vec![
            ("cgan-child-zono", AdmitSite::GraphInputSplit),
            ("fwd-root-screen", AdmitSite::DisjunctiveLaneGraph),
            ("la-warm-frontier", AdmitSite::DagGraph),
            ("vit-cover-certify", AdmitSite::DisjunctiveMultiObjective),
            // #kfsb-production: the graph kFSB selector's single-shot scorer.
            ("kfsb-single-shot", AdmitSite::GraphReluSplit),
        ]
    );
    let window = registry::find("bi-window-joint-refresh").expect("registered");
    let window = window.routable.expect("routable").admit_when;
    assert!(window.ops_any.contains(OpKind::Conv) && window.site.is_none());
    // The census envelope is unchanged by the new facts.
    assert!(!CENSUS_WIDEST_ENVELOPE.ops.contains(OpKind::Other));
}

/// A conjunction of two output thresholds (`Y_0 <= 1e6` AND `Y_1 <= 1e6`):
/// two atoms, no `or`.
fn conjunction_property(n_x: usize) -> String {
    let mut s = halfspace_property(n_x);
    let at = s.rfind("(assert (<= Y_0").expect("the halfspace atom");
    s.truncate(at);
    s.push_str("(assert (<= Y_0 1000000.0))\n(assert (<= Y_1 1000000.0))\n");
    s
}

/// #kfsb-production — THE CENSUS ADMITS THE RIGHT ROWS (golden). The
/// single-shot kFSB scorer's consumer is the shared graph kFSB selector,
/// inside a graph ReLU-split BaB: the DomainList lane on a graph model (any
/// property), and the conjunctive multi-objective lane's aggregation-critical
/// route, which a sequential model reaches through the CLI's ReLU-split
/// conjunction upgrade or the disjunctive lane's per-disjunct graph BaB.
/// So, CR-RS's own decision (pretended sealed under the widest envelope):
///
/// * a residual conv net and a plain conv net (graph models with a ReLU):
///   every property admitted;
/// * a plain MLP: admitted on robustness / a threshold band (both enter the
///   disjunctive lane) and on a two-atom conjunction (the upgrade); declined
///   on a single threshold and a single two-output atom (the sequential lane
///   keeps them: no graph BaB, no graph kFSB);
/// * a model with no ReLU at all: declined on every property (nothing to
///   split), with the site's own reason.
///
/// Mutation targets: dropping the conjunction-upgrade term (mlp_c goes 0),
/// dropping the ReLU-presence term (lin_* go 1), or the graph term (conv_h
/// goes 0) turns this red.
#[test]
fn kfsb_production_census_admits_the_graph_relu_split_rows() {
    let world = World::new();
    let twin = world.file("twin.onnx", &twin_model_bytes(2));
    let mlp = workspace_root().join("tests/models/simple_mlp.onnx");
    let conv = workspace_root().join("tests/models/conv_relu.onnx");
    let lin = workspace_root().join("tests/models/single_linear.onnx");
    let r = world.file("r.vnnlib", robustness_property(72).as_bytes());
    let h = world.file("h.vnnlib", halfspace_property(72).as_bytes());
    let b = world.file("b.vnnlib", band_property(72).as_bytes());
    let s = world.file("s.vnnlib", relational_atom_property(72).as_bytes());
    let c = world.file("c.vnnlib", conjunction_property(72).as_bytes());
    let rows = [
        row("twin_r", &twin, &r),
        row("twin_h", &twin, &h),
        row("twin_c", &twin, &c),
        row("mlp_r", &mlp, &r),
        row("mlp_h", &mlp, &h),
        row("mlp_b", &mlp, &b),
        row("mlp_s", &mlp, &s),
        row("mlp_c", &mlp, &c),
        row("conv_r", &conv, &r),
        row("conv_h", &conv, &h),
        row("lin_r", &lin, &r),
        row("lin_c", &lin, &c),
    ];
    let (_, cells) = census_cells(&rows, &world.options("kfsb"));
    let golden: [(&str, u8); 12] = [
        ("twin_r", 1),
        ("twin_h", 1),
        ("twin_c", 1),
        ("mlp_r", 1),
        ("mlp_h", 0),
        ("mlp_b", 1),
        ("mlp_s", 0),
        ("mlp_c", 1),
        ("conv_r", 1),
        ("conv_h", 1),
        ("lin_r", 0),
        ("lin_c", 0),
    ];
    let site = AdmitSite::GraphReluSplit;
    let mut wrong = Vec::new();
    let mut admitted = 0usize;
    for (id, want) in golden {
        let c = cell(&cells, id, "kfsb-single-shot");
        let got = (c[0].as_str(), c[1].as_str(), c[2].as_str());
        let (expect, probe) = if want == 1 {
            ("1", "-")
        } else {
            ("0", "skipped")
        };
        if got != (expect, expect, probe) {
            wrong.push(format!("{id}: {c:?}, want admit={expect}"));
        }
        if want == 0 && c[3] != format!("admit:decline:{}", site.decline()) {
            wrong.push(format!("{id}: reasons {:?}", c[3]));
        }
        admitted += usize::from(want);
    }
    assert!(
        wrong.is_empty(),
        "kfsb-single-shot census decisions on the fixture rows:\n  {}",
        wrong.join("\n  ")
    );
    // Non-vacuity: the golden both admits and declines.
    assert!(admitted > 0 && admitted < golden.len());
    // The scan really saw no ReLU in the linear fixture (the decline is the
    // operator term, not an unread model).
    assert!(
        !cell(&cells, "lin_r", "kfsb-single-shot")[5].contains("relu"),
        "{:?}",
        cell(&cells, "lin_r", "kfsb-single-shot")
    );
    assert!(cell(&cells, "mlp_r", "kfsb-single-shot")[5].contains("relu"));
    println!(
        "[kfsb-production] census admitted={admitted} of {}",
        golden.len()
    );
}
