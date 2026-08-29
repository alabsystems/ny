// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Facade-contract tests for the facade-audit expansion (Alethe emission,
//! misc layer group, GenBaB config sub-types, `LinearBounds`,
//! `MulBinaryRelaxationMode`, and the f64 evaluation/propagation helpers).
//!
//! Every test constructs and calls EXCLUSIVELY through `ny_api` paths — if a
//! curated re-export is removed or renamed, these fail at compile time; the
//! runtime assertions additionally pin real propagation behaviour (sound
//! enclosures, non-empty proof artifacts), not just nameability.

use ndarray::{arr1, arr2, ArrayD};

use ny_api::bab::{
    AdaptiveOptConfig, AlphaCrownConfig, BetaCrownConfig, BetaCrownVerifier, BranchingHeuristic,
    BranchingPointMethod, CutEvictionPolicy, CutScoreWeights, DepthTwoBranchLookaheadConfig,
    DepthTwoBranchLookaheadMode, ExactVjpPgdConfig, GradientMethod, NonlinearBranchingConfig,
    NonlinearHeuristicMethod, PgdAlphaMode, PgdInitialization, PgdOptimizer,
    VerificationArtifactAuthority, ViolationWitness,
};
use ny_api::cert::{certify_graph_with_alethe, entailment_to_alethe, farkas_to_alethe, Rat};
use ny_api::graph::{GraphNetwork, GraphNode, SequentialNetwork};
use ny_api::layers::{
    CeilLayer, CompareLayer, CompareOp, EpsilonExpandLayer, FloorLayer, Layer, LinearLayer,
    ReLULayer, SignLayer, TruncLayer, WhereLayer,
};
use ny_api::probabilistic::{propagate_distribution, AnalyticDistribution, LinearBounds};
use ny_api::verify::{
    convert_network_to_f64, evaluate_network_f64, propagate_network_f64, F64PropagationMode,
    MulBinaryRelaxationMode, PropagationConfig, SequentialLayerF64,
};
use ny_api::{Bound, BoundedTensor, BoundedTensor64, VerificationSpec};

/// The misc layer group is constructible through the facade and wraps into the
/// same [`Layer`] enum the graph builders consume.
#[test]
fn misc_layer_group_constructs_through_facade() {
    let layers = [
        Layer::Floor(FloorLayer::new()),
        Layer::Ceil(CeilLayer::new()),
        Layer::Sign(SignLayer::new()),
        Layer::Trunc(TruncLayer::new()),
        Layer::Where(WhereLayer::new()),
        Layer::Compare(CompareLayer::new(0.5, CompareOp::Gt)),
        Layer::EpsilonExpand(
            EpsilonExpandLayer::try_new(arr1(&[0.25_f32, 0.25]).into_dyn())
                .expect("finite non-negative epsilons are valid"),
        ),
    ];
    assert_eq!(layers.len(), 7, "every misc-group layer constructed");
    // The Compare family round-trips its operator.
    let compare = CompareLayer::new(1.5, CompareOp::Le);
    assert_eq!(compare.op, CompareOp::Le);
    assert_eq!(compare.threshold, 1.5);
}

/// The GenBaB heuristic — and every curated `BetaCrownConfig` sub-config type —
/// is constructible through `ny_api::bab` alone (this is exactly the config nn
/// builds via direct ny-propagate deps today).
#[test]
fn genbab_config_constructs_through_facade() {
    let config = BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::GenBaB(NonlinearBranchingConfig {
            point_method: BranchingPointMethod::Uniform,
            method: NonlinearHeuristicMethod::Bbps,
            num_candidates: 4,
            ..Default::default()
        }),
        ..Default::default()
    };
    let BranchingHeuristic::GenBaB(nl) = &config.branching_heuristic else {
        panic!("GenBaB heuristic constructed through the facade");
    };
    assert_eq!(nl.num_candidates, 4);
    let _verifier = BetaCrownVerifier::new(config);

    // Compile-contract: the remaining sub-config / result payload types are
    // nameable and default-constructible through the facade.
    let _alpha: AlphaCrownConfig = AlphaCrownConfig::default();
    let _gm: GradientMethod = AlphaCrownConfig::default().gradient_method;
    let _adaptive: AdaptiveOptConfig = AdaptiveOptConfig::default();
    let _evict: CutEvictionPolicy = BetaCrownConfig::default().cut_eviction_policy;
    let _weights: CutScoreWeights = BetaCrownConfig::default().cut_score_weights;
    let _lookahead: DepthTwoBranchLookaheadConfig = DepthTwoBranchLookaheadConfig::default();
    let _mode: DepthTwoBranchLookaheadMode = DepthTwoBranchLookaheadConfig::default().mode;
    let _authority: VerificationArtifactAuthority =
        BetaCrownConfig::default().verification_artifact_authority;
    let _pgd_init: PgdInitialization = BetaCrownConfig::default().pgd_initialization;
    let _pgd_opt: PgdOptimizer = BetaCrownConfig::default().pgd_optimizer;
    let _pgd_alpha: PgdAlphaMode = BetaCrownConfig::default().pgd_alpha_mode;
    let _pgd_vjp: ExactVjpPgdConfig = BetaCrownConfig::default().pgd_exact_vjp;
    fn _witness_nameable(_: &ViolationWitness) {}
}

/// `LinearBounds` is constructible through `ny_api::probabilistic` and feeds
/// [`propagate_distribution`] — with identity coefficients over a uniform input
/// the output mean enclosure must contain the input's true mean.
#[test]
fn linear_bounds_feed_propagate_distribution() {
    let lb = LinearBounds::identity(2);
    let input = BoundedTensor::new(
        arr1(&[0.0_f32, 2.0]).into_dyn(),
        arr1(&[1.0_f32, 4.0]).into_dyn(),
    )
    .expect("valid box");
    let out = propagate_distribution(&lb, &AnalyticDistribution::UniformFromBounds, &input, 0.9)
        .expect("identity propagation succeeds");
    // Uniform means are (l+u)/2 = [0.5, 3.0]; the identity-map mean enclosure
    // must contain them (sound, directed outward).
    for (i, &mu) in [0.5_f32, 3.0].iter().enumerate() {
        assert!(
            out.mean_lower[i] <= mu && mu <= out.mean_upper[i],
            "mean enclosure [{}, {}] must contain {mu}",
            out.mean_lower[i],
            out.mean_upper[i]
        );
    }
}

/// `MulBinaryRelaxationMode` is the facade-visible `PropagationConfig` field
/// type: both modes are settable through `ny_api::verify` alone.
#[test]
fn mul_binary_relaxation_mode_is_configurable() {
    let config = PropagationConfig {
        mul_binary_relaxation: MulBinaryRelaxationMode::Middle,
        ..Default::default()
    };
    assert_eq!(
        config.mul_binary_relaxation,
        MulBinaryRelaxationMode::Middle
    );
    assert_eq!(
        PropagationConfig::default().mul_binary_relaxation,
        MulBinaryRelaxationMode::McCormick,
        "default relaxation is McCormick (classic CROWN)"
    );
}

/// The f64 helpers convert, evaluate, and propagate a Linear+ReLU chain through
/// facade paths only — and the propagated IBP box must ENCLOSE the exact f64
/// evaluation at interior points (a real soundness assertion, not nameability).
#[test]
fn f64_helpers_convert_evaluate_and_enclose() {
    // y = relu(2x - 1) over x in [0, 1] -> y in [0, 1].
    let mut net = SequentialNetwork::new();
    net.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[2.0_f32]]), Some(arr1(&[-1.0_f32]))).expect("valid linear"),
    ));
    net.add_layer(Layer::ReLU(ReLULayer::new()));

    let f64_layers: Vec<SequentialLayerF64> =
        convert_network_to_f64(net.layers()).expect("Linear+ReLU converts");
    assert_eq!(f64_layers.len(), 2, "both layers converted");

    let box64 = BoundedTensor64::new(
        ArrayD::from_shape_vec(vec![1], vec![0.0_f64]).unwrap(),
        ArrayD::from_shape_vec(vec![1], vec![1.0_f64]).unwrap(),
    )
    .expect("valid f64 box");
    let out = propagate_network_f64(&f64_layers, &box64, F64PropagationMode::Ibp)
        .expect("IBP propagates");

    for x in [0.0_f64, 0.25, 0.5, 0.75, 1.0] {
        let point = ArrayD::from_shape_vec(vec![1], vec![x]).unwrap();
        let y = evaluate_network_f64(&f64_layers, &point).expect("f64 eval");
        assert!(
            out.lower()[0] <= y[0] && y[0] <= out.upper()[0],
            "IBP box [{}, {}] must enclose f(x={x}) = {}",
            out.lower()[0],
            out.upper()[0],
            y[0]
        );
    }
}

/// E2E through `ny_api` paths only: `certify_graph_with_alethe` on a tiny
/// FC-ReLU graph emits the exact-rational certificate AND a non-empty Alethe
/// pair per conjunct.
#[test]
fn certify_graph_with_alethe_end_to_end() {
    // y = 2 * relu(x) + 1/2 over x in [1, 2] -> y in [5/2, 9/2]; assert y >= 0.
    let mut g = GraphNetwork::new();
    g.add_node(GraphNode::from_input(
        "lin1",
        Layer::Linear(
            LinearLayer::new(arr2(&[[1.0_f32]]), Some(arr1(&[0.0_f32]))).expect("valid linear"),
        ),
    ));
    g.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer::new()),
        vec!["lin1".to_string()],
    ));
    g.add_node(GraphNode::new(
        "lin2",
        Layer::Linear(
            LinearLayer::new(arr2(&[[2.0_f32]]), Some(arr1(&[0.5_f32]))).expect("valid linear"),
        ),
        vec!["relu".to_string()],
    ));
    g.set_output("lin2");

    let spec = VerificationSpec::new(
        vec![Bound::new(1.0, 2.0)],
        vec![Bound::new_allow_infinite(0.0, f32::INFINITY)],
    )
    .expect("valid spec");

    let out = certify_graph_with_alethe(&g, &spec).expect("certification runs");
    assert!(out.certified.eligible, "eligible: {}", out.certified.note);
    assert!(
        out.certified.result.is_verified(),
        "y >= 5/2 satisfies y >= 0"
    );
    assert!(
        out.certified.certificate_json.is_some(),
        "certificate emitted; note: {}",
        out.certified.note
    );
    let alethe = out
        .alethe
        .as_ref()
        .unwrap_or_else(|| panic!("Alethe set emitted; note: {}", out.alethe_note));
    assert_eq!(alethe.len(), 1, "single lower-bound conjunct");
    let pair = &alethe[0];
    assert!(
        !pair.entailment.problem.is_empty() && !pair.entailment.proof.is_empty(),
        "non-empty entailment SMT-LIB/Alethe pair"
    );
    assert!(
        !pair.farkas.problem.is_empty() && !pair.farkas.proof.is_empty(),
        "non-empty farkas SMT-LIB/Alethe pair"
    );
    assert!(
        pair.entailment.proof.contains("la_generic") && pair.entailment.proof.contains("(cl)"),
        "Alethe refutation shape: {}",
        pair.entailment.proof
    );
}

/// The standalone Alethe emitters are callable through `ny_api::cert` on a
/// certificate produced through the same facade (the shallow CROWN problem).
#[test]
fn standalone_alethe_emitters_through_facade() {
    use ny_api::cert::{check_entailment, Relu1Problem};
    let r = |n: i128, d: i128| Rat::new(n, d).unwrap();
    let problem = Relu1Problem {
        w1: vec![vec![r(1, 1), r(1, 1)], vec![r(1, 1), r(-1, 1)]],
        b1: vec![Rat::ZERO, Rat::ZERO],
        w2: vec![r(1, 1), r(-1, 1)],
        b2: r(5, 2),
        input_lower: vec![r(-1, 1), r(-1, 1)],
        input_upper: vec![r(1, 1), r(1, 1)],
        alpha: Some(vec![r(1, 2), r(1, 2)]),
    };
    let certified = problem.certify(Rat::ZERO).expect("certifies");
    check_entailment(&certified.entailment).expect("self-check passes");

    let ent = entailment_to_alethe(&certified.entailment).expect("entailment emits");
    assert!(ent.problem.contains("set-logic"), "SMT-LIB problem header");
    assert!(ent.proof.contains("la_generic"), "la_generic step present");
    let far = farkas_to_alethe(&certified.farkas).expect("farkas emits");
    assert!(far.proof.contains("(cl)"), "empty clause derived");
}
