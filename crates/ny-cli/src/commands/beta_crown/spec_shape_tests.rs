// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #admission-precision: the routing-shape extractor on every specification
//! FORM the 2025 + 2026 moat corpus carries (one hermetic miniature per form,
//! golden), its start-time read (name-blind, deterministic, fail-closed to
//! `unknown`), and the BIT-IDENTITY of every lane check refactored onto the
//! shared predicates: each refactored function against its pre-refactor
//! inline expression (verbatim oracles) on every fixture and on generated
//! specifications.

use std::collections::BTreeMap;

use ny_core::router::admission;
use ny_core::router::decl::SpecArchetype;
use ny_onnx::vnnlib::{parse_vnnlib, OutputConstraint, VnnLibSpec};

use super::spec_shape::{
    affine_parser_refused_for_representability, read_spec, shape_of, SpecRead, SPEC_READ_MAX_BYTES,
};

fn header(n_x: usize, n_y: usize) -> String {
    let mut s = String::new();
    for i in 0..n_x {
        s.push_str(&format!("(declare-const X_{i} Real)\n"));
    }
    for j in 0..n_y {
        s.push_str(&format!("(declare-const Y_{j} Real)\n"));
    }
    for i in 0..n_x {
        s.push_str(&format!(
            "(assert (>= X_{i} -1.0))\n(assert (<= X_{i} 1.0))\n"
        ));
    }
    s
}

/// `(name, property text, golden SpecRead token)` — one miniature per form
/// the moat corpus carries, named by the form (never by a category).
fn forms() -> Vec<(&'static str, String, &'static str)> {
    let h = |n_x, n_y| header(n_x, n_y);
    vec![
        // One output threshold (single-atom half-space).
        (
            "threshold_atom",
            h(5, 5) + "(assert (>= Y_0 3.991125645861615))\n",
            "halfspace c=1 a=1 bx=0 -",
        ),
        // A conjunction of output thresholds.
        (
            "threshold_conjunction",
            h(3, 2) + "(assert (>= Y_0 1.0))\n(assert (<= Y_1 0.0))\n",
            "halfspace c=1 a=2 bx=0 -",
        ),
        // One atom comparing two outputs, no `or` (the two-class shape).
        (
            "relational_atom",
            h(4, 2) + "(assert (<= Y_0 Y_1))\n",
            "robustness c=1 a=1 bx=0 rel+argmax+same_lhs",
        ),
        // Argmax robustness, `(>= Y_j Y_t)` disjuncts.
        (
            "argmax_ge",
            h(4, 4)
                + "(assert (or\n    (and (>= Y_1 Y_0))\n    (and (>= Y_2 Y_0))\n    (and (>= Y_3 Y_0))\n))\n",
            "robustness c=3 a=3 bx=0 or+multi+rel+argmax",
        ),
        // Argmax robustness with the true class on the LEFT, `(<= Y_t Y_j)`.
        (
            "argmax_le_true_left",
            h(4, 4)
                + "(assert (or\n  (and (<= Y_3 Y_0))\n  (and (<= Y_3 Y_1))\n  (and (<= Y_3 Y_2))\n))\n",
            // Every atom shares its left side too: the sequential reduction
            // reads the atoms, not the aggregation.
            "robustness c=3 a=3 bx=0 or+multi+rel+argmax+same_lhs",
        ),
        // `(<= Y_j Y_t)` disjuncts: a common LARGER side, so not argmax by
        // the offline rule (a disjunction of relational atoms).
        (
            "relational_common_larger",
            h(5, 5)
                + "(assert (or\n    (and (<= Y_1 Y_0))\n    (and (<= Y_2 Y_0))\n    (and (<= Y_3 Y_0))\n    (and (<= Y_4 Y_0))\n))\n",
            // Flipped to `(>= Y_0 Y_j)` the atoms share their left side.
            "disjunctive c=4 a=4 bx=0 or+multi+rel+same_lhs",
        ),
        // A disjunction of output thresholds (a band).
        (
            "threshold_band",
            h(5, 1)
                + "(assert (or\n    (and (>= Y_0 0.6903536319732666))\n    (and (<= Y_0 0.6603536605834961))\n))\n",
            "disjunctive c=2 a=2 bx=0 or+multi",
        ),
        // A disjunction of single thresholds on different outputs.
        (
            "threshold_disjunction",
            h(6, 4)
                + "(assert (or\n  (and (>= Y_0 1e-06))\n  (and (<= Y_2 -0.7))\n  (and (>= Y_3 0.7))\n))\n",
            "disjunctive c=3 a=3 bx=0 or+multi",
        ),
        // A disjunction of multi-atom clauses.
        (
            "clause_disjunction",
            h(4, 4)
                + "(assert (or\n    (and (<= Y_0 -0.03) (>= Y_1 -0.2) (<= Y_2 -0.9))\n    (and (>= Y_3 0.5) (<= Y_0 2.4))\n))\n",
            "disjunctive c=2 a=5 bx=0 or+multi",
        ),
        // A same-LHS relational conjunction (the sequential lane reduces it).
        (
            "relational_same_lhs_conjunction",
            h(5, 5)
                + "(assert (<= Y_0 Y_1))\n(assert (<= Y_0 Y_2))\n(assert (<= Y_0 Y_3))\n(assert (<= Y_0 Y_4))\n",
            "halfspace c=1 a=4 bx=0 rel+same_lhs",
        ),
        // A chained relational conjunction (no same-LHS reduction).
        (
            "relational_chain_conjunction",
            h(1, 3) + "(assert (<= Y_1 Y_0))\n(assert (<= Y_2 Y_1))\n",
            "halfspace c=1 a=2 bx=0 rel",
        ),
        // ONE clause carrying its own input box.
        (
            "boxed_single_clause",
            h(2, 1) + "(assert (or\n    (and (>= X_1 0.1) (<= X_1 0.5) (<= Y_0 -1e-5))\n))\n",
            "halfspace c=1 a=1 bx=1 boxed",
        ),
        // Clauses each carrying their own input box.
        (
            "boxed_clauses",
            h(2, 1)
                + "(assert (or\n    (and (<= X_1 0.5) (>= X_1 0.1) (<= Y_0 0.0))\n    (and (<= X_1 -0.1) (>= X_1 -0.5) (>= Y_0 1.0))\n))\n",
            "disjunctive c=2 a=2 bx=2 or+multi+boxed",
        ),
        // A pure input-box disjunction distributed over an output disjunction.
        (
            "input_box_case_split",
            "(declare-const X_0 Real)\n(declare-const X_1 Real)\n(declare-const Y_0 Real)\n\
             (declare-const Y_1 Real)\n(assert (<= X_0 1.0))\n(assert (>= X_0 -1.0))\n\
             (assert (or\n    (and (<= X_1 0.5) (>= X_1 0.1))\n    (and (<= X_1 -0.1) (>= X_1 -0.5))\n))\n\
             (assert (or\n    (and (<= Y_1 Y_0))\n    (and (<= Y_0 0.0))\n))\n"
                .to_owned(),
            "disjunctive c=4 a=4 bx=4 or+multi+boxed+rel",
        ),
        // VNN-LIB 2.0 tensor declarations, one output threshold.
        (
            "v2_tensor_threshold",
            "(vnnlib-version 2.0)\n(declare-input X Real [2 2])\n(declare-output Y Real [2])\n\
             (assert (>= X[0,0] -1))\n(assert (<= X[0,0] 1))\n(assert (<= Y[1] 0.5))\n"
                .to_owned(),
            "halfspace c=1 a=1 bx=0 -",
        ),
        // Non-linear arithmetic and strict input atoms (the D45 lanes').
        (
            "nonlinear",
            "(vnnlib-version <2.0>)\n(declare-network f\n    (declare-input X real [1,2])\n    \
             (declare-output Y real [1,1])\n)\n\
             (assert (and (>= X[0,0] 60.0) (<= X[0,0] 80.0)))\n\
             (assert (and (>= X[0,1] 80.0) (<= X[0,1] 120.0)))\n\
             (assert (and (> X[0,0] 0.0) (>= (* X[0,0] 200.0) (* X[0,1] X[0,1]))))\n\
             (assert (or (< Y[0,0] -100.001) (> Y[0,0] 100.001)))\n"
                .to_owned(),
            "nonlinear",
        ),
        // Two declared networks (the relational route's).
        (
            "two_networks",
            "(vnnlib-version 2.0)\n(declare-network f\n  (declare-input X_f Float32 [2])\n  \
             (declare-output Y_f Float32 [2])\n)\n(declare-network g\n  (isomorphic-to f)\n  \
             (declare-input X_g Float32 [2])\n  (declare-output Y_g Float32 [2])\n)\n\
             (assert (>= X_f[0] -1))\n(assert (<= X_f[0] 1))\n(assert (>= X_g[0] -1))\n\
             (assert (<= X_g[0] 1))\n(assert (>= X_f[1] 0))\n(assert (<= X_f[1] 2))\n\
             (assert (>= X_g[1] 0))\n(assert (<= X_g[1] 2))\n(assert (== X_f[0] X_g[0]))\n\
             (assert (== X_g[1] X_f[1]))\n(assert (<= Y_g[0] (+ Y_f[0] 0.01)))\n"
                .to_owned(),
            "relational",
        ),
    ]
}

/// GOLDEN: every corpus form reads to its token, the archetype is the
/// offline classifier's, and the per-form counts are pinned.
#[test]
fn spec_shape_golden_on_every_corpus_form() {
    let dir = tempfile::tempdir().expect("temporary property dir");
    let mut archetypes: BTreeMap<String, usize> = BTreeMap::new();
    let mut mismatches = Vec::new();
    for (name, text, golden) in forms() {
        let path = dir.path().join(format!("{name}.vnnlib"));
        std::fs::write(&path, &text).expect("write the fixture");
        let read = read_spec(&path);
        if read.token() != golden {
            mismatches.push(format!(
                "{name}: read {:?}, golden {golden:?}",
                read.token()
            ));
        }
        let key = read
            .archetype()
            .map_or_else(|| "unknown".to_owned(), |a| a.token().to_owned());
        *archetypes.entry(key).or_default() += 1;
        // The read is the parse plus the one extractor.
        if let Ok(spec) = parse_vnnlib(&text) {
            if spec.dual_network.is_none() {
                assert_eq!(read, SpecRead::Affine(shape_of(&spec)), "{name}");
            }
        }
    }
    assert!(
        mismatches.is_empty(),
        "the routing shape of a corpus form changed:\n  {}",
        mismatches.join("\n  ")
    );
    let counts: Vec<(String, usize)> = archetypes.into_iter().collect();
    assert_eq!(
        counts,
        vec![
            ("disjunctive".to_owned(), 6),
            ("halfspace".to_owned(), 6),
            ("nonlinear".to_owned(), 1),
            ("relational".to_owned(), 1),
            ("robustness".to_owned(), 3),
        ],
        "golden archetype counts over the corpus forms"
    );
}

/// NAME-BLIND + DETERMINISTIC: the same content under category-like and
/// random paths reads identically, twice.
#[test]
fn spec_shape_read_is_name_blind_and_deterministic() {
    let dir = tempfile::tempdir().expect("temporary property dir");
    for (name, text, _) in forms() {
        let mut reads = Vec::new();
        for rel in [
            format!("cifar100_2024/vnnlib/{name}.vnnlib"),
            format!("acasxu_2023/vnnlib/prop_1_{name}.vnnlib"),
            format!("zz/q{name}.txt"),
        ] {
            let path = dir.path().join(rel);
            std::fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
            std::fs::write(&path, &text).expect("write");
            reads.push(read_spec(&path));
            reads.push(read_spec(&path));
        }
        assert!(
            reads.windows(2).all(|w| w[0] == w[1]),
            "{name}: reads differ by name or by run: {reads:?}"
        );
    }
}

/// FAIL-CLOSED TO UNKNOWN: a missing file, an unparsable one and one above
/// the read ceiling are `unknown` (which defers every specification check to
/// the consumer's own site), never a guessed shape.
#[test]
fn spec_shape_read_refuses_what_it_cannot_read() {
    let dir = tempfile::tempdir().expect("temporary property dir");
    assert_eq!(
        read_spec(&dir.path().join("absent.vnnlib")),
        SpecRead::Unknown("spec-unreadable")
    );
    let garbage = dir.path().join("garbage.vnnlib");
    std::fs::write(&garbage, "(assert (>= Y_0").expect("write");
    assert_eq!(read_spec(&garbage), SpecRead::Unknown("spec-unparsed"));
    // Above the ceiling (a sparse file: no parse is attempted).
    let large = dir.path().join("large.vnnlib");
    let file = std::fs::File::create(&large).expect("create");
    file.set_len(SPEC_READ_MAX_BYTES + 1).expect("set_len");
    assert_eq!(read_spec(&large), SpecRead::Unknown("spec>16MiB"));
    let at_ceiling = dir.path().join("at_ceiling.vnnlib");
    let mut text = header(1, 1) + "(assert (>= Y_0 1.0))\n";
    text.push_str(&" ".repeat(64));
    std::fs::write(&at_ceiling, &text).expect("write");
    assert!(matches!(read_spec(&at_ceiling), SpecRead::Affine(_)));
    for read in [
        SpecRead::Unknown("x"),
        SpecRead::Nonlinear,
        SpecRead::Relational,
    ] {
        assert!(read.shape().is_none());
    }
    assert_eq!(SpecRead::Unknown("x").archetype(), None);
    assert_eq!(
        SpecRead::Nonlinear.archetype(),
        Some(SpecArchetype::Nonlinear)
    );
    // The D45 refusal classifier (the nonlinear lanes' entry and the
    // router's `nonlinear` archetype are this one function).
    assert!(affine_parser_refused_for_representability(
        "Invalid spec: Non-linear arithmetic in output constraint"
    ));
    assert!(affine_parser_refused_for_representability(
        "Strict input constraints are unsupported: affine input boxes"
    ));
    assert!(!affine_parser_refused_for_representability(
        "unexpected end of input"
    ));
}

// ---- bit-identity of the refactored lane checks ------------------------------------

/// `dispatch.rs` before #admission-precision.
fn old_routes_to_relational_verifier(spec: Option<&VnnLibSpec>, has_relational: bool) -> bool {
    spec.is_some_and(|vnnlib| {
        vnnlib.output_constraints.len() > 1
            || has_relational
            || vnnlib.has_boxed_clause_disjunction()
    })
}

/// `verify/mod.rs` `verify_relational_constraints_with_ledger` before.
fn old_disjunctive_lane(vnnlib: &VnnLibSpec) -> bool {
    vnnlib.has_multi_constraint_disjunction() || vnnlib.has_boxed_clause_disjunction()
}

/// `verify/mod.rs` `should_upgrade_sequential_conjunction_to_graph` before
/// (the same-LHS reduction through the lane's own function).
fn old_should_upgrade(vnnlib: &VnnLibSpec, use_relu_split: bool) -> bool {
    use super::constraint_plan::{classify_constraints, AggregationMode};
    if classify_constraints(vnnlib).aggregation != AggregationMode::Conjunctive
        || vnnlib.output_constraints.len() < 2
    {
        return false;
    }
    use_relu_split || !super::verify::same_lhs_reducible(vnnlib)
}

/// `verify/graph.rs` `verify_graph_relational` before: `Some(conjunctive)`
/// when the multi-objective branch runs.
fn old_graph_multi_objective(vnnlib: &VnnLibSpec, use_relu_split: bool) -> Option<bool> {
    use super::constraint_plan::{classify_constraints, AggregationMode};
    let is_disjunction = classify_constraints(vnnlib).aggregation == AggregationMode::Disjunctive;
    (vnnlib.output_constraints.len() >= 2 && (use_relu_split || !is_disjunction))
        .then_some(!is_disjunction)
}

/// `model_load.rs` before: the boxed-clause count test.
fn old_huge(vnnlib: &VnnLibSpec) -> bool {
    vnnlib
        .per_clause_input_bounds
        .iter()
        .filter(|b| !b.is_empty())
        .count()
        >= 512
}

/// The refactored lane functions and the shared checks, on `vnnlib`.
fn assert_bit_identical(name: &str, vnnlib: &VnnLibSpec) {
    let shape = shape_of(vnnlib);
    let classified = super::constraint_plan::classify_constraints(vnnlib).has_relational;
    for has_relational in [classified, true, false] {
        assert_eq!(
            super::dispatch::routes_to_relational_verifier(Some(vnnlib), has_relational),
            old_routes_to_relational_verifier(Some(vnnlib), has_relational),
            "{name}: routes_to_relational_verifier(has_relational={has_relational})"
        );
    }
    assert_eq!(
        super::dispatch::routes_to_relational_verifier(None, true),
        old_routes_to_relational_verifier(None, true)
    );
    assert_eq!(
        super::verify::enters_disjunctive_lane(vnnlib),
        old_disjunctive_lane(vnnlib),
        "{name}: disjunctive lane"
    );
    assert_eq!(
        admission::disjunctive_lane(&shape),
        old_disjunctive_lane(vnnlib)
    );
    for relu in [false, true] {
        assert_eq!(
            super::verify::should_upgrade_sequential_conjunction_to_graph(vnnlib, relu),
            old_should_upgrade(vnnlib, relu),
            "{name}: sequential upgrade (relu={relu})"
        );
        assert_eq!(
            super::verify::multi_objective_route(vnnlib, relu),
            old_graph_multi_objective(vnnlib, relu),
            "{name}: graph multi-objective (relu={relu})"
        );
    }
    assert_eq!(
        super::model_load::huge_boxed_clause_spec(vnnlib),
        old_huge(vnnlib),
        "{name}: huge boxed disjunction"
    );
    assert_eq!(
        admission::huge_boxed_clause_disjunction(shape.boxed_clauses),
        old_huge(vnnlib)
    );
    assert_eq!(shape.relational_atoms, classified, "{name}");
}

/// BIT-IDENTITY on every corpus form.
#[test]
fn spec_shape_lane_checks_are_bit_identical_on_every_form() {
    let mut checked = 0;
    for (name, text, _) in forms() {
        if let Ok(spec) = parse_vnnlib(&text) {
            assert_bit_identical(name, &spec);
            checked += 1;
        }
    }
    assert!(checked >= 15, "positive control: {checked} forms parsed");
}

/// BIT-IDENTITY on generated specifications: every atom kind, grouped and
/// ungrouped atoms, either aggregation, and clause-box counts around the
/// 512 threshold (programmatic specifications reach shapes no parser emits;
/// the checks must agree there too).
#[test]
fn spec_shape_lane_checks_are_bit_identical_on_generated_specs() {
    let mut state = 0x5eed_a0a0_u64;
    let mut next = move |n: u64| {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) % n.max(1)
    };
    let atom = |kind: u64, i: usize, j: usize| match kind {
        0 => OutputConstraint::LessEq(i, j),
        1 => OutputConstraint::GreaterEq(i, j),
        2 => OutputConstraint::LessThan(i, j),
        3 => OutputConstraint::GreaterThan(i, j),
        4 => OutputConstraint::LessEqConst(i, 0.5),
        5 => OutputConstraint::GreaterEqConst(i, -0.5),
        6 => OutputConstraint::LessThanConst(i, 1.0),
        _ => OutputConstraint::GreaterThanConst(i, -1.0),
    };
    for case in 0..4_000 {
        let n_clauses = next(4) as usize;
        let mut spec = VnnLibSpec::new();
        spec.num_inputs = 2;
        spec.num_outputs = 6;
        spec.input_bounds = vec![(-1.0, 1.0), (0.0, 0.0)];
        let common = next(6) as usize;
        for _ in 0..n_clauses.max(1) {
            let width = 1 + next(3) as usize;
            let clause: Vec<OutputConstraint> = (0..width)
                .map(|_| {
                    let kind = next(8);
                    let j = if next(2) == 0 {
                        common
                    } else {
                        next(6) as usize
                    };
                    atom(kind, next(6) as usize, j)
                })
                .collect();
            spec.output_constraint_clauses.push(clause);
        }
        if n_clauses == 0 {
            // Ungrouped atoms.
            spec.output_constraints = spec.output_constraint_clauses.remove(0);
        } else {
            spec.output_constraints = spec
                .output_constraint_clauses
                .iter()
                .flatten()
                .cloned()
                .collect();
        }
        spec.is_disjunction = next(2) == 0;
        let boxed = [0usize, 1, 511, 512, 513][next(5) as usize];
        if boxed > 0 && !spec.output_constraint_clauses.is_empty() {
            spec.per_clause_input_bounds = (0..boxed)
                .map(|k| {
                    let mut b = BTreeMap::new();
                    if k % 3 != 2 {
                        b.insert(0, (-0.5, 0.5));
                    }
                    b
                })
                .collect();
        }
        assert_bit_identical(&format!("generated#{case}"), &spec);
    }
}

// ---- review: the router's MODEL terms against the REAL loader -----------------------

/// The router's facts for `model` (the live glue: `router::policy_input` on a
/// start context holding the plan resolver's scan, exactly as the scored
/// entry point and the census build them).
fn router_ops(scan: &crate::plan_resolver::ModelFacts) -> (ny_core::router::decl::OpSet, bool) {
    let ctx = crate::router::StartContext {
        levers: None,
        authority: crate::router::Authority::Frozen,
        model: Some(scan),
        model_bytes: Some(scan.file_size_bytes),
        facts_reason: None,
        declared_networks: Some(1),
        relational_selected: Some(false),
        official_budget_secs: 100,
        internal_secs: 95,
        join_margin: std::time::Duration::from_secs(2),
        anchor: std::time::Instant::now(),
        target_host: Some(ny_core::router::facts::HostClass::M5),
        pinned_keys: Vec::new(),
        // #router-r6: no trial knob pinned.
        trial_pinned_keys: Vec::new(),
        spec: SpecRead::Unknown("review-model-terms"),
    };
    let facts = crate::router::policy_input(&ctx).facts;
    (facts.ops, facts.ops_complete)
}

/// REVIEW (#admission-precision): the sites' MODEL terms are UPPER BOUNDS of
/// the REAL loader's decision, not only of the literal dispatch model the
/// ny-core closure test checks. Every tracked fixture model is loaded by
/// `model_load::load_model` (no property, so the boxed-clause term is out)
/// under each policy that can route it to a graph (ReLU split, input split,
/// the MIP complete verifier, either VGG MaxPool rewrite). Whenever the loader
/// builds a `GraphNetwork`, the router's `graph_network_possible` on the
/// scan's operator signature must answer `true`; whenever the built graph
/// holds a node the #warm-dag guard refuses at (two or more inputs, a binary
/// or ternary layer), `dag_possible` must too. A failure here is a row the
/// router (and the census, and so the seal's moat) declines although its
/// consumer can engage. Mutation target: a kind dropped from `N_ARY_AT_LOAD`
/// or `CONV_ROUTE`.
#[test]
fn spec_shape_model_terms_bound_the_real_loader_on_every_tracked_model() {
    use ny_core::router::admission::{dag_possible, graph_network_possible, site_reachable};
    use ny_core::router::decl::AdmitSite;
    use ny_propagate::{BranchingHeuristic, VggMaxPoolRewriteMode};

    let models_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/models");
    let mut models: Vec<std::path::PathBuf> = std::fs::read_dir(&models_dir)
        .expect("the tracked fixture models")
        .map(|entry| entry.expect("dir entry").path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "onnx"))
        .collect();
    models.sort();
    assert!(
        models.len() >= 20,
        "positive control: {} tracked models",
        models.len()
    );
    // A non-boxed shape (the property is not handed to the loader, so the
    // huge boxed-clause term is false on both sides) that reaches every
    // specification term: a band disjunction of two clauses.
    let band = parse_vnnlib(
        &(header(2, 1) + "(assert (or\n    (and (>= Y_0 0.5))\n    (and (<= Y_0 -0.5))\n))\n"),
    )
    .expect("band");
    let shape = shape_of(&band);
    assert!(admission::reaches_disjunctive_lane(&shape) && shape.boxed_clauses == 0);
    // (ReLU split, branching, complete verifier, VGG MaxPool rewrite).
    type LoadPolicy = (
        bool,
        Option<BranchingHeuristic>,
        crate::CompleteVerifierArg,
        Option<VggMaxPoolRewriteMode>,
    );
    let policies: [LoadPolicy; 5] = [
        (true, None, crate::CompleteVerifierArg::Auto, None),
        (
            false,
            Some(BranchingHeuristic::InputSplit),
            crate::CompleteVerifierArg::Auto,
            None,
        ),
        (true, None, crate::CompleteVerifierArg::Mip, None),
        (
            true,
            None,
            crate::CompleteVerifierArg::Auto,
            Some(VggMaxPoolRewriteMode::Residual),
        ),
        (
            true,
            None,
            crate::CompleteVerifierArg::Auto,
            Some(VggMaxPoolRewriteMode::Sequential),
        ),
    ];
    let config = ny_onnx::OnnxLoadConfig::default();
    let mut violations = Vec::new();
    let (mut loads, mut graphs, mut dags, mut sequential, mut unscanned) = (0, 0, 0, 0, 0);
    for model in &models {
        let name = model
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("?")
            .to_owned();
        let Some(scan) = crate::plan_resolver::ModelFacts::from_onnx_file(model) else {
            unscanned += 1;
            continue;
        };
        let (ops, complete) = router_ops(&scan);
        if !complete {
            // The router refuses every capability on an incomplete signature.
            unscanned += 1;
            continue;
        }
        for (relu, heuristic, verifier, vgg) in &policies {
            let Ok((loaded, _)) = super::model_load::load_model(
                model,
                &config,
                None,
                super::TerminalPeelPolicy::Off,
                heuristic.as_ref(),
                *relu,
                true,
                None,
                false,
                *verifier,
                true,
                None,
                *vgg,
            ) else {
                continue;
            };
            loads += 1;
            let n_ary = match &loaded.model {
                super::BetaCrownModel::Graph(graph) => graph.node_names().iter().any(|node| {
                    graph.node(node).is_some_and(|node| {
                        node.inputs().len() >= 2
                            || node.layer().is_binary()
                            || node.layer().is_ternary()
                    })
                }),
                super::BetaCrownModel::Sequential(_) => false,
            };
            if loaded.is_graph {
                graphs += 1;
                if graph_network_possible(ops, Some(&shape)) != Some(true) {
                    violations.push(format!("{name} {relu}/{verifier:?}/{vgg:?}: the loader built a graph, the router's bound says no (ops {ops:?})"));
                }
                for site in [AdmitSite::GraphInputSplit, AdmitSite::DisjunctiveLaneGraph] {
                    if site_reachable(site, ops, Some(&shape)) != Some(true) {
                        violations.push(format!("{name}: {site:?} declined on a graph model"));
                    }
                }
            } else {
                sequential += 1;
            }
            if n_ary {
                dags += 1;
                if dag_possible(ops, Some(&shape)) != Some(true)
                    || site_reachable(AdmitSite::DagGraph, ops, Some(&shape)) != Some(true)
                {
                    violations.push(format!("{name} {relu}/{verifier:?}/{vgg:?}: the loaded graph has an n-ary node, the router's DAG bound says no (ops {ops:?})"));
                }
            }
        }
    }
    assert!(
        violations.is_empty(),
        "the router's model terms do not bound the real loader:\n  {}",
        violations.join("\n  ")
    );
    assert!(
        loads >= 40 && graphs >= 5 && dags >= 3 && sequential >= 5,
        "positive control: loads={loads} graphs={graphs} dags={dags} sequential={sequential} unscanned={unscanned}"
    );
}
