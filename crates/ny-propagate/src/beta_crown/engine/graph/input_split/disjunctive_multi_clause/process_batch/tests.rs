// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::cell::Cell;
use std::collections::BinaryHeap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ndarray::{arr1, arr2, Array2, ArrayD, IxDyn};
use ny_core::Result;
use ny_tensor::BoundedTensor;
use ny_test_utils::CountingGemmEngine;

use super::*;
use crate::beta_crown::config::{BetaCrownConfig, InputClipType};
use crate::beta_crown::engine::graph::input_split::grouped_semantics::disjunctive_domain_verified;
use crate::beta_crown::engine::graph::input_split::shared::{
    extract_obj_bounds, graph_spec_ibp_fallback,
};
use crate::beta_crown::engine::graph::shared::state::GraphBabLifecycle;
use crate::beta_crown::engine::BetaCrownVerifier;
use crate::bounds::{GraphAlphaState, LinearBounds};
use crate::layers::{Conv2dLayer, ConvTranspose2dLayer, Layer, LinearLayer, ReLULayer};
use crate::{GraphNetwork, GraphNode};

fn build_disjunctive_batch_graph_4353() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "out",
        Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32]]), None).expect("valid output linear")),
    ));
    graph.set_output("out");
    graph
}

fn unresolved_multi_obj_domain_4353() -> MultiObjInputDomain {
    let input_bounds =
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
            .expect("finite parent bounds");
    MultiObjInputDomain {
        input_bounds: Arc::new(input_bounds),
        obj_bounds: vec![(-1.0, 1.0), (-1.0, 1.0)],
        linear_bounds: None,
        depth: 0,
        priority: 1.0,
        needs_bounding: false,
        node_bounds_override: None,
        inherited_alpha_state: None,
    }
}

fn disjunctive_baseline_gemm_calls_4353(
    graph: &GraphNetwork,
    spec_matrix: &Array2<f32>,
    thresholds: &[f32],
    clause_sizes: &[usize],
) -> usize {
    let left_child = BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[0.0_f32]).into_dyn())
        .expect("finite left child");
    let right_child = BoundedTensor::new(arr1(&[0.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
        .expect("finite right child");
    let engine = CountingGemmEngine::new();
    let baseline_verified: Vec<bool> = [&left_child, &right_child]
        .into_iter()
        .map(|child| {
            let (bounds, _) =
                graph_spec_ibp_fallback(graph, child, spec_matrix, Some(&engine), None)
                    .expect("per-child grouped IBP fallback should succeed");
            let obj_bounds = extract_obj_bounds(&bounds, thresholds.len()).unwrap();
            disjunctive_domain_verified(&obj_bounds, thresholds, clause_sizes)
        })
        .collect();
    assert_eq!(
        baseline_verified,
        vec![false, true],
        "baseline grouped split children should leave exactly one unresolved domain"
    );
    engine.gemm_calls()
}

fn assert_queued_grouped_child_4353(queue: &mut BinaryHeap<MultiObjInputDomain>) {
    let child = queue
        .pop()
        .expect("one unresolved grouped child should be queued");
    assert!(child.needs_bounding);
    assert_eq!(child.depth, 1);
    assert_eq!(child.obj_bounds, vec![(-1.0, 1.0), (-1.0, 1.0)]);
    assert!(child.linear_bounds.is_none());
    assert!(child.node_bounds_override.is_none());
    assert_eq!(child.input_bounds.lower()[[0]], -1.0);
    assert_eq!(child.input_bounds.upper()[[0]], 0.0);
}

mod input_leaf_escalation {
    use super::*;
    use crate::beta_crown::graph_mip_leaf::{
        GraphInputLeafRequest, GraphMipLeafOracle, GraphMipLeafRequest, GraphMipLeafVerdict,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Clone, Copy)]
    enum MockVerdict {
        Verified,
        Violated,
        Undecided,
    }

    impl MockVerdict {
        fn materialize(self) -> GraphMipLeafVerdict {
            match self {
                Self::Verified => GraphMipLeafVerdict::VerifiedAllRows,
                Self::Violated => GraphMipLeafVerdict::Violated {
                    witness: vec![0.0],
                    output: vec![0.0],
                },
                Self::Undecided => GraphMipLeafVerdict::Undecided,
            }
        }
    }

    struct MockInputLeafOracle {
        consults: AtomicUsize,
        legacy_consults: AtomicUsize,
        input_verdict: MockVerdict,
        legacy_verdict: MockVerdict,
    }

    impl MockInputLeafOracle {
        fn new(verify_all: bool) -> Self {
            Self::with_verdicts(
                if verify_all {
                    MockVerdict::Verified
                } else {
                    MockVerdict::Undecided
                },
                MockVerdict::Undecided,
            )
        }

        fn with_verdicts(input_verdict: MockVerdict, legacy_verdict: MockVerdict) -> Self {
            Self {
                consults: AtomicUsize::new(0),
                legacy_consults: AtomicUsize::new(0),
                input_verdict,
                legacy_verdict,
            }
        }
    }

    impl GraphMipLeafOracle for MockInputLeafOracle {
        fn solve_input_leaf(&self, req: &GraphInputLeafRequest<'_>) -> GraphMipLeafVerdict {
            self.consults.fetch_add(1, Ordering::SeqCst);
            assert_eq!(req.graph.output_name(), "out");
            assert_eq!(
                req.input_bounds.lower()[[0]].to_bits(),
                (-1.0_f32).to_bits()
            );
            assert_eq!(req.input_bounds.upper()[[0]].to_bits(), 1.0_f32.to_bits());
            assert_eq!(req.objectives.shape(), &[2, 1]);
            assert_eq!(req.objectives[[0, 0]].to_bits(), 1.0_f32.to_bits());
            assert_eq!(req.objectives[[1, 0]].to_bits(), 0.5_f32.to_bits());
            assert_eq!(
                req.advisory_objective_bounds,
                &[(-1.0_f32, 1.0_f32), (-1.0_f32, 1.0_f32)]
            );
            assert_eq!(req.thresholds, &[-0.1_f32, -0.1_f32]);
            assert_eq!(req.clause_sizes, &[1usize, 1usize]);
            assert_eq!(req.depth, 0);
            assert!(req.deadline.is_some());
            self.input_verdict.materialize()
        }

        fn solve_leaf(&self, _req: &GraphMipLeafRequest<'_>) -> GraphMipLeafVerdict {
            self.legacy_consults.fetch_add(1, Ordering::SeqCst);
            self.legacy_verdict.materialize()
        }
    }

    struct LegacyOnlyOracle;

    impl GraphMipLeafOracle for LegacyOnlyOracle {
        fn solve_leaf(&self, _req: &GraphMipLeafRequest<'_>) -> GraphMipLeafVerdict {
            GraphMipLeafVerdict::Undecided
        }
    }

    struct BatchOutcome {
        oracle: Arc<MockInputLeafOracle>,
        queue: BinaryHeap<MultiObjInputDomain>,
        domains_explored: usize,
        domains_verified: usize,
        gemm_calls: usize,
    }

    fn run_batch_with_verdict(gate: bool, verdict: MockVerdict) -> BatchOutcome {
        let oracle = Arc::new(MockInputLeafOracle::with_verdicts(
            verdict,
            MockVerdict::Undecided,
        ));
        let verifier = BetaCrownVerifier::new(BetaCrownConfig {
            reorder_bab: true,
            input_split_ibp_enhancement: true,
            enable_relaxed_clip: false,
            input_split_input_leaf_oracle: gate,
            ..Default::default()
        })
        .with_graph_mip_leaf_oracle(oracle.clone());
        let graph = build_disjunctive_batch_graph_4353();
        let spec_matrix = arr2(&[[1.0_f32], [0.5_f32]]);
        let thresholds = [-0.1_f32, -0.1_f32];
        let clause_sizes = [1usize, 1usize];
        let engine = CountingGemmEngine::new();
        let mut queue = BinaryHeap::new();
        let mut lifecycle = GraphBabLifecycle::new(Instant::now());
        let mut domains_verified_by_clip = 0usize;

        let result = process_disjunctive_domain_batch(
            &verifier,
            &graph,
            vec![unresolved_multi_obj_domain_4353()],
            &spec_matrix,
            &thresholds,
            &clause_sizes,
            Some(&engine),
            &|_input, _node_bounds| -> Result<MultiObjBounds> {
                panic!("reordered IBP prescreen must not call compute_bounds")
            },
            None,
            &WarmAlphaTelemetry::new(false),
            &FreshDomainClipTelemetry::new(false),
            None,
            Duration::from_secs(1),
            &mut queue,
            &mut lifecycle,
            &mut domains_verified_by_clip,
        )
        .expect("input-leaf batch processing should not error");
        assert!(result.is_none());
        assert_eq!(domains_verified_by_clip, 0);

        BatchOutcome {
            oracle,
            queue,
            domains_explored: lifecycle.domains_explored,
            domains_verified: lifecycle.domains_verified,
            gemm_calls: engine.gemm_calls(),
        }
    }

    fn run_batch(gate: bool, verify_all: bool) -> BatchOutcome {
        run_batch_with_verdict(
            gate,
            if verify_all {
                MockVerdict::Verified
            } else {
                MockVerdict::Undecided
            },
        )
    }

    #[test]
    fn legacy_oracle_gets_fail_closed_input_leaf_default() {
        let graph = build_disjunctive_batch_graph_4353();
        let input = unresolved_multi_obj_domain_4353();
        let objectives = arr2(&[[1.0_f32], [0.5_f32]]);
        let thresholds = [-0.1_f32, -0.1_f32];
        let clause_sizes = [1usize, 1usize];
        let request = GraphInputLeafRequest {
            graph: &graph,
            input_bounds: input.input_bounds.as_ref(),
            objectives: &objectives,
            advisory_objective_bounds: &input.obj_bounds,
            thresholds: &thresholds,
            clause_sizes: &clause_sizes,
            depth: input.depth,
            deadline: None,
        };

        assert!(matches!(
            LegacyOnlyOracle.solve_input_leaf(&request),
            GraphMipLeafVerdict::Undecided
        ));
    }

    #[test]
    fn verified_input_leaf_drops_domain_while_undecided_requeues_unchanged() {
        let verified = run_batch(true, true);
        assert_eq!(verified.oracle.consults.load(Ordering::SeqCst), 1);
        assert_eq!(verified.oracle.legacy_consults.load(Ordering::SeqCst), 0);
        assert_eq!(verified.domains_explored, 1);
        assert_eq!(verified.domains_verified, 1);
        assert!(verified.queue.is_empty());
        assert_eq!(
            verified.gemm_calls, 0,
            "the lightweight verdict must precede all rebound/big-M collection work"
        );

        let mut undecided = run_batch(true, false);
        assert_eq!(undecided.oracle.consults.load(Ordering::SeqCst), 1);
        assert_eq!(undecided.oracle.legacy_consults.load(Ordering::SeqCst), 0);
        assert_eq!(undecided.domains_explored, 1);
        assert_eq!(undecided.domains_verified, 1);
        assert_eq!(undecided.queue.len(), 1);
        assert!(undecided.gemm_calls > 0);
        assert_queued_grouped_child_4353(&mut undecided.queue);
    }

    #[test]
    fn advisory_input_leaf_violation_requeues_through_the_unchanged_path() {
        let mut advisory = run_batch_with_verdict(true, MockVerdict::Violated);
        assert_eq!(advisory.oracle.consults.load(Ordering::SeqCst), 1);
        assert_eq!(advisory.oracle.legacy_consults.load(Ordering::SeqCst), 0);
        assert_eq!(advisory.domains_explored, 1);
        assert_eq!(advisory.domains_verified, 1);
        assert_eq!(advisory.queue.len(), 1);
        assert!(advisory.gemm_calls > 0);
        assert_queued_grouped_child_4353(&mut advisory.queue);
    }

    #[test]
    fn input_leaf_gate_defaults_off_and_preserves_fallback_batch() {
        assert!(!BetaCrownConfig::default().input_split_input_leaf_oracle);

        let mut gate_off = run_batch(false, true);
        let mut explicit_undecided = run_batch(true, false);
        assert_eq!(gate_off.oracle.consults.load(Ordering::SeqCst), 0);
        assert_eq!(gate_off.oracle.legacy_consults.load(Ordering::SeqCst), 0);
        assert_eq!(
            gate_off.domains_explored,
            explicit_undecided.domains_explored
        );
        assert_eq!(
            gate_off.domains_verified,
            explicit_undecided.domains_verified
        );
        assert_eq!(gate_off.gemm_calls, explicit_undecided.gemm_calls);
        assert_eq!(gate_off.queue.len(), explicit_undecided.queue.len());
        assert_queued_grouped_child_4353(&mut gate_off.queue);
        assert_queued_grouped_child_4353(&mut explicit_undecided.queue);
    }

    #[test]
    fn input_leaf_result_completed_at_deadline_has_no_authority() {
        let oracle = Arc::new(MockInputLeafOracle::new(true));
        let verifier = BetaCrownVerifier::new(BetaCrownConfig {
            input_split_input_leaf_oracle: true,
            ..Default::default()
        })
        .with_graph_mip_leaf_oracle(oracle.clone());
        let graph = build_disjunctive_batch_graph_4353();
        let domain = unresolved_multi_obj_domain_4353();
        let objectives = arr2(&[[1.0_f32], [0.5_f32]]);
        let thresholds = [-0.1_f32, -0.1_f32];
        let clause_sizes = [1usize, 1usize];
        let deadline = Instant::now() + Duration::from_secs(5);
        let before = deadline
            .checked_sub(Duration::from_nanos(1))
            .expect("future deadline has a predecessor");
        let polls = Cell::new(0usize);

        let accepted = try_input_leaf_escalation_with_clock(
            &verifier,
            &graph,
            &domain,
            &objectives,
            &thresholds,
            &clause_sizes,
            deadline,
            || {
                let poll = polls.get();
                polls.set(poll + 1);
                if poll == 0 {
                    before
                } else {
                    deadline
                }
            },
        );

        assert!(!accepted, "a proof completed at the deadline is too late");
        assert_eq!(polls.get(), 2);
        assert_eq!(oracle.consults.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn input_leaf_at_deadline_is_rejected_before_consult() {
        let oracle = Arc::new(MockInputLeafOracle::new(true));
        let verifier = BetaCrownVerifier::new(BetaCrownConfig {
            input_split_input_leaf_oracle: true,
            ..Default::default()
        })
        .with_graph_mip_leaf_oracle(oracle.clone());
        let graph = build_disjunctive_batch_graph_4353();
        let domain = unresolved_multi_obj_domain_4353();
        let objectives = arr2(&[[1.0_f32], [0.5_f32]]);
        let thresholds = [-0.1_f32, -0.1_f32];
        let clause_sizes = [1usize, 1usize];
        let deadline = Instant::now() + Duration::from_secs(5);
        let polls = Cell::new(0usize);

        let accepted = try_input_leaf_escalation_with_clock(
            &verifier,
            &graph,
            &domain,
            &objectives,
            &thresholds,
            &clause_sizes,
            deadline,
            || {
                polls.set(polls.get() + 1);
                deadline
            },
        );

        assert!(!accepted);
        assert_eq!(polls.get(), 1);
        assert_eq!(oracle.consults.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn malformed_or_nonfinite_input_leaf_requests_are_never_consulted() {
        let oracle = Arc::new(MockInputLeafOracle::new(true));
        let verifier = BetaCrownVerifier::new(BetaCrownConfig {
            input_split_input_leaf_oracle: true,
            ..Default::default()
        })
        .with_graph_mip_leaf_oracle(oracle.clone());
        let graph = build_disjunctive_batch_graph_4353();
        let domain = unresolved_multi_obj_domain_4353();
        let objectives = arr2(&[[1.0_f32], [0.5_f32]]);
        let nonfinite_objectives = arr2(&[[f32::NAN], [0.5_f32]]);
        let thresholds = [-0.1_f32, -0.1_f32];
        let nonfinite_thresholds = [-0.1_f32, f32::INFINITY];
        let clause_sizes = [1usize, 1usize];
        let malformed_clause_sizes = [1usize];
        let deadline = Instant::now() + Duration::from_secs(5);

        assert!(!try_input_leaf_escalation(
            &verifier,
            &graph,
            &domain,
            &objectives,
            &thresholds,
            &malformed_clause_sizes,
            deadline,
        ));
        assert!(!try_input_leaf_escalation(
            &verifier,
            &graph,
            &domain,
            &nonfinite_objectives,
            &thresholds,
            &clause_sizes,
            deadline,
        ));
        assert!(!try_input_leaf_escalation(
            &verifier,
            &graph,
            &domain,
            &objectives,
            &nonfinite_thresholds,
            &clause_sizes,
            deadline,
        ));
        assert_eq!(oracle.consults.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn legacy_edge_oracle_result_completed_at_deadline_has_no_authority() {
        let oracle = Arc::new(MockInputLeafOracle::with_verdicts(
            MockVerdict::Undecided,
            MockVerdict::Verified,
        ));
        let verifier = BetaCrownVerifier::new(BetaCrownConfig {
            input_split_edge_milp: true,
            input_split_edge_milp_gap: 10.0,
            input_split_edge_milp_depth: 0,
            ..Default::default()
        })
        .with_graph_mip_leaf_oracle(oracle.clone());
        let graph = build_disjunctive_batch_graph_4353();
        let domain = unresolved_multi_obj_domain_4353();
        let objectives = arr2(&[[1.0_f32], [0.5_f32]]);
        let thresholds = [-0.1_f32, -0.1_f32];
        let clause_sizes = [1usize, 1usize];
        let deadline = Instant::now() + Duration::from_secs(5);
        let before = deadline
            .checked_sub(Duration::from_nanos(1))
            .expect("future deadline has a predecessor");
        let polls = Cell::new(0usize);

        let accepted = try_edge_milp_escalation_with_clock(
            &verifier,
            &graph,
            &domain,
            &objectives,
            &thresholds,
            &clause_sizes,
            None,
            deadline,
            || {
                let poll = polls.get();
                polls.set(poll + 1);
                if poll < 2 {
                    before
                } else {
                    deadline
                }
            },
        );

        assert!(!accepted, "a late edge-MILP proof has no authority");
        assert_eq!(polls.get(), 3);
        assert_eq!(oracle.consults.load(Ordering::SeqCst), 0);
        assert_eq!(oracle.legacy_consults.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn legacy_edge_oracle_is_not_started_at_the_deadline() {
        let oracle = Arc::new(MockInputLeafOracle::with_verdicts(
            MockVerdict::Undecided,
            MockVerdict::Verified,
        ));
        let verifier = BetaCrownVerifier::new(BetaCrownConfig {
            input_split_edge_milp: true,
            input_split_edge_milp_gap: 10.0,
            input_split_edge_milp_depth: 0,
            ..Default::default()
        })
        .with_graph_mip_leaf_oracle(oracle.clone());
        let graph = build_disjunctive_batch_graph_4353();
        let domain = unresolved_multi_obj_domain_4353();
        let objectives = arr2(&[[1.0_f32], [0.5_f32]]);
        let thresholds = [-0.1_f32, -0.1_f32];
        let clause_sizes = [1usize, 1usize];
        let deadline = Instant::now() + Duration::from_secs(5);
        let polls = Cell::new(0usize);

        let accepted = try_edge_milp_escalation_with_clock(
            &verifier,
            &graph,
            &domain,
            &objectives,
            &thresholds,
            &clause_sizes,
            None,
            deadline,
            || {
                polls.set(polls.get() + 1);
                deadline
            },
        );

        assert!(!accepted);
        assert_eq!(polls.get(), 1);
        assert_eq!(oracle.consults.load(Ordering::SeqCst), 0);
        assert_eq!(oracle.legacy_consults.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn input_leaf_oracle_survives_with_config_from_restart() {
        let oracle = Arc::new(MockInputLeafOracle::new(true));
        let original = BetaCrownVerifier::new(BetaCrownConfig {
            input_split_input_leaf_oracle: true,
            ..Default::default()
        })
        .with_graph_mip_leaf_oracle(oracle.clone());
        let restarted = original.with_config_from(BetaCrownConfig {
            input_split_input_leaf_oracle: true,
            ..Default::default()
        });
        let graph = build_disjunctive_batch_graph_4353();
        let cloned_graph = graph.clone();
        let domain = unresolved_multi_obj_domain_4353();
        let objectives = arr2(&[[1.0_f32], [0.5_f32]]);
        let thresholds = [-0.1_f32, -0.1_f32];
        let clause_sizes = [1usize, 1usize];
        let deadline = Instant::now() + Duration::from_secs(5);

        assert!(try_input_leaf_escalation(
            &original,
            &graph,
            &domain,
            &objectives,
            &thresholds,
            &clause_sizes,
            deadline,
        ));
        assert!(try_input_leaf_escalation(
            &restarted,
            &cloned_graph,
            &domain,
            &objectives,
            &thresholds,
            &clause_sizes,
            deadline,
        ));
        assert_eq!(
            oracle.consults.load(Ordering::SeqCst),
            2,
            "with_config_from must preserve the same runtime oracle Arc"
        );
    }
}

#[test]
fn edge_alpha_combination_canary_cannot_inherit_root_only_cgan_collector() {
    ny_test_utils::env::with_env_edits(|env| {
        for key in [
            "NY_NO_FORWARD_LINEAR_REF",
            "NY_NO_FORWARD_LINEAR_CONV_TRANSPOSE_REF",
            "NY_CROWN_IBP_SPARSE_RELU_ROWS",
            "NY_CROWN_IBP_DOWNSTREAM_RESWEEP",
            "NY_CROWN_IBP_COLLECTOR_CAP_SECS",
            "NY_DISABLE_CROWN_COLLECTION_CACHE",
        ] {
            env.remove(key);
        }

        let root = crate::bounds::AlphaCrownConfig {
            iterations: 100,
            fix_interm_bounds: false,
            cgan_sparse_target_complete_root: true,
            cgan_complete_crown_ibp_root: true,
            ..Default::default()
        };
        let deadline = Instant::now() + Duration::from_secs(5);
        let child = edge_alpha_child_config(&root, 7, deadline);

        assert!(
            root.cgan_sparse_target_complete_root,
            "fixture must model the armed root cGAN canary"
        );
        assert!(
            !child.cgan_sparse_target_complete_root,
            "an independently armed edge-alpha lane must not inherit root collector authority"
        );
        assert!(
            !child.cgan_complete_crown_ibp_root,
            "an independently armed edge-alpha lane must not inherit complete root authority"
        );
        assert!(
            !child.fix_interm_bounds,
            "the scope guard must preserve the edge pass's ordinary intermediate-bound policy"
        );
        assert_eq!(child.iterations, 7);
        assert_eq!(child.deadline, Some(deadline));

        let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![1.0_f32, -0.5, 0.25, 0.75])
            .expect("kernel");
        let conv = ConvTranspose2dLayer::with_input_shape(
            kernel,
            Some(arr1(&[0.1_f32])),
            (1, 1),
            (0, 0),
            2,
            2,
        )
        .expect("ConvTranspose2d");
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("convt", Layer::ConvTranspose2d(conv)));
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["convt".into()],
        ));
        // The typed cGAN transaction is deliberately restricted to the exact
        // sequential ConvTranspose2d + Conv2d image surface.  Keep this canary
        // inside that production eligibility contract instead of weakening
        // the root-only structural gate for a ConvTranspose-only lookalike.
        let conv = Conv2dLayer::with_input_shape(
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![0.75_f32]).expect("Conv2d kernel"),
            Some(arr1(&[-0.2_f32])),
            (1, 1),
            (0, 0),
            3,
            3,
        )
        .expect("Conv2d");
        graph.add_node(GraphNode::new(
            "conv",
            Layer::Conv2d(conv),
            vec!["relu".into()],
        ));
        graph.set_output("conv");
        let input = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[1, 2, 2]), -1.0_f32),
            ArrayD::from_elem(IxDyn(&[1, 2, 2]), 1.0_f32),
        )
        .expect("input");
        let exec_order = graph.exec_order().expect("execution order").to_vec();
        let (_root_bounds, root_source) = graph
            .collect_alpha_reference_bounds_with_engine_and_source(&input, &root, None, &exec_order)
            .expect("armed root collection");
        assert!(
            !root_source.is_crown_ibp(),
            "armed root must enter the distinct typed transaction"
        );
        let (_child_bounds, child_source) = graph
            .collect_alpha_reference_bounds_with_engine_and_source(
                &input,
                &child,
                None,
                &exec_order,
            )
            .expect("edge-child collection");
        assert!(
            child_source.is_crown_ibp(),
            "the edge child must take ordinary fix_interm_bounds=false CROWN-IBP, not the typed \
             root collector"
        );
    });
}

#[test]
fn test_process_disjunctive_domain_batch_reorder_batches_ibp_prescreen_4353() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        reorder_bab: true,
        input_split_ibp_enhancement: true,
        enable_relaxed_clip: false,
        ..Default::default()
    });
    let graph = build_disjunctive_batch_graph_4353();
    let spec_matrix = arr2(&[[1.0_f32], [0.5_f32]]);
    let thresholds = [-0.1_f32, -0.1_f32];
    let clause_sizes = [1usize, 1usize];
    let baseline_calls =
        disjunctive_baseline_gemm_calls_4353(&graph, &spec_matrix, &thresholds, &clause_sizes);

    let mut queue = BinaryHeap::new();
    let mut lifecycle = GraphBabLifecycle::new(Instant::now());
    let mut domains_verified_by_clip = 0usize;
    let batched_engine = CountingGemmEngine::new();
    let warm_alpha_telemetry = WarmAlphaTelemetry::new(false);
    let fresh_domain_clip_telemetry = FreshDomainClipTelemetry::new(false);

    let result = process_disjunctive_domain_batch(
        &verifier,
        &graph,
        vec![unresolved_multi_obj_domain_4353()],
        &spec_matrix,
        &thresholds,
        &clause_sizes,
        Some(&batched_engine),
        &|_input, _node_bounds| -> Result<MultiObjBounds> {
            panic!("reorder batched grouped prescreen should not call compute_bounds")
        },
        None,
        &warm_alpha_telemetry,
        &fresh_domain_clip_telemetry,
        None,
        Duration::from_secs(1),
        &mut queue,
        &mut lifecycle,
        &mut domains_verified_by_clip,
    )
    .expect("batched disjunctive processing should not error");

    assert!(result.is_none());
    assert_eq!(domains_verified_by_clip, 0);
    assert_eq!(lifecycle.domains_verified, 1);
    assert_eq!(queue.len(), 1);
    assert_queued_grouped_child_4353(&mut queue);
    assert!(
        batched_engine.gemm_calls() < baseline_calls,
        "batched grouped process path should reduce GEMM dispatches: batched={}, baseline={}",
        batched_engine.gemm_calls(),
        baseline_calls
    );
}

fn exact_one_row_plane_4353(a: f32, b: f32) -> LinearBounds {
    LinearBounds::new(arr2(&[[a]]), arr1(&[b]), arr2(&[[a]]), arr1(&[b]))
        .expect("valid exact one-row plane")
}

fn fresh_clip_parent_4353(rows: usize) -> MultiObjInputDomain {
    MultiObjInputDomain {
        input_bounds: Arc::new(
            BoundedTensor::new(arr1(&[0.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
                .expect("finite exact parent box"),
        ),
        obj_bounds: vec![(-1.0, 3.0); rows],
        // A batch rebound plane is deliberately present as a poison carrier.
        // The fresh dispatcher must discard it before proof or split scoring;
        // no child may inherit it.
        linear_bounds: Some(
            LinearBounds::new(
                Array2::zeros((rows, 1)),
                arr1(&vec![-100.0_f32; rows]),
                Array2::zeros((rows, 1)),
                arr1(&vec![100.0_f32; rows]),
            )
            .expect("valid deliberately uninformative rebound plane"),
        ),
        depth: 0,
        priority: 1.0,
        needs_bounding: false,
        node_bounds_override: None,
        inherited_alpha_state: None,
    }
}

fn fresh_clip_processing_verifier_4353() -> BetaCrownVerifier {
    // This is the effective internal verifier after the outer typed dispatcher
    // has validated and armed the run-local capability: both config-owned clip
    // flags are clear, while telemetry below alone authorizes the fresh route.
    BetaCrownVerifier::new(BetaCrownConfig {
        reorder_bab: true,
        input_split_ibp_enhancement: true,
        enable_relaxed_clip: false,
        input_split_fresh_domain_clip: false,
        relaxed_clip_iterations: 3,
        ..Default::default()
    })
}

#[test]
fn fresh_domain_clip_runs_on_exact_parent_before_split_and_drops_planes() {
    let verifier = fresh_clip_processing_verifier_4353();
    verifier
        .config
        .validate()
        .expect("effective processing verifier must be valid");
    let graph = build_disjunctive_batch_graph_4353();
    let spec_matrix = arr2(&[[1.0_f32]]);
    let thresholds = [0.4_f32];
    let clause_sizes = [1usize];
    let calls = Cell::new(0usize);
    let mut queue = BinaryHeap::new();
    let mut lifecycle = GraphBabLifecycle::new(Instant::now());
    let mut domains_verified_by_clip = 0usize;
    let warm_alpha_telemetry = WarmAlphaTelemetry::new(false);
    let fresh_domain_clip_telemetry = FreshDomainClipTelemetry::new(true);

    let result = process_disjunctive_domain_batch(
        &verifier,
        &graph,
        vec![fresh_clip_parent_4353(1)],
        &spec_matrix,
        &thresholds,
        &clause_sizes,
        None,
        &|exact, node_bounds| -> Result<MultiObjBounds> {
            calls.set(calls.get() + 1);
            assert!(
                node_bounds.is_none(),
                "fresh full-spec pass has no child cache"
            );
            assert_eq!(exact.lower()[[0]].to_bits(), 0.0_f32.to_bits());
            assert_eq!(exact.upper()[[0]].to_bits(), 1.0_f32.to_bits());
            Ok((vec![(0.0, 1.0)], Some(exact_one_row_plane_4353(1.0, 0.0))))
        },
        None,
        &warm_alpha_telemetry,
        &fresh_domain_clip_telemetry,
        None,
        Duration::from_secs(1),
        &mut queue,
        &mut lifecycle,
        &mut domains_verified_by_clip,
    )
    .expect("fresh current-domain processing should succeed");

    assert!(result.is_none());
    assert_eq!(
        calls.get(),
        1,
        "one non-domain-stacked full-spec CROWN pass per popped domain"
    );
    assert_eq!(fresh_domain_clip_telemetry.snapshot(), (1, 1, 0, 0, 1));
    assert_eq!(lifecycle.domains_explored, 1);
    assert_eq!(lifecycle.domains_verified, 0);
    assert_eq!(domains_verified_by_clip, 0);
    assert_eq!(queue.len(), 2, "both clipped children remain unresolved");

    let mut child_boxes = Vec::new();
    while let Some(child) = queue.pop() {
        assert!(child.needs_bounding);
        assert!(
            child.linear_bounds.is_none(),
            "no parent/fresh plane escapes"
        );
        assert!(child.node_bounds_override.is_none());
        child_boxes.push((
            child.input_bounds.lower()[[0]],
            child.input_bounds.upper()[[0]],
        ));
    }
    child_boxes.sort_by(|left, right| left.0.total_cmp(&right.0));
    let [(left_l, left_u), (right_l, right_u)] = child_boxes.as_slice() else {
        panic!("expected exactly two child boxes")
    };
    assert_eq!(left_l.to_bits(), 0.0_f32.to_bits());
    assert!(
        *left_u < 0.25,
        "split must follow pre-split clip, got {left_u}"
    );
    assert!((*left_u - *right_l).abs() <= 2.0e-6);
    assert!(
        *right_u < 0.5,
        "fresh clip must tighten x <= 0.4, got {right_u}"
    );
    assert!(*right_u >= 0.4 - 2.0e-5);
}

#[test]
fn fresh_domain_clip_terminal_authority_requires_all_clauses() {
    let verifier = fresh_clip_processing_verifier_4353();
    let graph = build_disjunctive_batch_graph_4353();
    let spec_matrix = arr2(&[[1.0_f32], [1.0_f32]]);
    let thresholds = [-1.0_f32, -2.0_f32];
    let clause_sizes = [1usize, 1usize];
    let calls = Cell::new(0usize);
    let mut queue = BinaryHeap::new();
    let mut lifecycle = GraphBabLifecycle::new(Instant::now());
    let mut domains_verified_by_clip = 0usize;
    let warm_alpha_telemetry = WarmAlphaTelemetry::new(false);
    let fresh_domain_clip_telemetry = FreshDomainClipTelemetry::new(true);

    let result = process_disjunctive_domain_batch(
        &verifier,
        &graph,
        vec![fresh_clip_parent_4353(2)],
        &spec_matrix,
        &thresholds,
        &clause_sizes,
        None,
        &|exact, _| -> Result<MultiObjBounds> {
            calls.set(calls.get() + 1);
            assert_eq!(exact.lower()[[0]], 0.0);
            assert_eq!(exact.upper()[[0]], 1.0);
            let plane = LinearBounds::new(
                arr2(&[[1.0_f32], [-1.0_f32]]),
                arr1(&[0.0_f32, 0.0_f32]),
                arr2(&[[1.0_f32], [-1.0_f32]]),
                arr1(&[0.0_f32, 0.0_f32]),
            )
            .expect("two exact affine refutations");
            Ok((vec![(-1.0, 1.0); 2], Some(plane)))
        },
        None,
        &warm_alpha_telemetry,
        &fresh_domain_clip_telemetry,
        None,
        Duration::from_secs(1),
        &mut queue,
        &mut lifecycle,
        &mut domains_verified_by_clip,
    )
    .expect("all-clause fresh refutation should succeed");

    assert!(result.is_none());
    assert_eq!(calls.get(), 1);
    assert!(queue.is_empty(), "terminal domain must not split or queue");
    assert_eq!(lifecycle.domains_explored, 1);
    assert_eq!(lifecycle.domains_verified, 1);
    assert_eq!(domains_verified_by_clip, 1);
    assert_eq!(fresh_domain_clip_telemetry.snapshot(), (1, 0, 1, 0, 0));
}

#[test]
fn fresh_domain_clip_missing_plane_skips_unchanged_and_still_drops_parent_plane() {
    let verifier = fresh_clip_processing_verifier_4353();
    let graph = build_disjunctive_batch_graph_4353();
    let spec_matrix = arr2(&[[1.0_f32]]);
    let thresholds = [2.0_f32];
    let clause_sizes = [1usize];
    let mut queue = BinaryHeap::new();
    let mut lifecycle = GraphBabLifecycle::new(Instant::now());
    let mut domains_verified_by_clip = 0usize;
    let warm_alpha_telemetry = WarmAlphaTelemetry::new(false);
    let fresh_domain_clip_telemetry = FreshDomainClipTelemetry::new(true);

    process_disjunctive_domain_batch(
        &verifier,
        &graph,
        vec![fresh_clip_parent_4353(1)],
        &spec_matrix,
        &thresholds,
        &clause_sizes,
        None,
        &|_, _| -> Result<MultiObjBounds> { Ok((vec![(0.0, 1.0)], None)) },
        None,
        &warm_alpha_telemetry,
        &fresh_domain_clip_telemetry,
        None,
        Duration::from_secs(1),
        &mut queue,
        &mut lifecycle,
        &mut domains_verified_by_clip,
    )
    .expect("missing full-spec planes must fail closed");

    assert_eq!(fresh_domain_clip_telemetry.snapshot(), (1, 0, 0, 1, 0));
    assert_eq!(domains_verified_by_clip, 0);
    assert_eq!(lifecycle.domains_verified, 0);
    assert_eq!(queue.len(), 2);
    let mut boxes = Vec::new();
    while let Some(child) = queue.pop() {
        assert!(
            child.linear_bounds.is_none(),
            "parent plane cannot leak on skip"
        );
        boxes.push((
            child.input_bounds.lower()[[0]],
            child.input_bounds.upper()[[0]],
        ));
    }
    boxes.sort_by(|left, right| left.0.total_cmp(&right.0));
    assert_eq!(boxes, vec![(0.0, 0.5), (0.5, 1.0)]);
}

#[test]
fn fresh_domain_clip_completed_at_deadline_loses_authority_deterministically() {
    let source = BoundedTensor::new(arr1(&[0.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
        .expect("finite exact source");
    let terminal = FreshDomainClipResult {
        bounds: source.clone(),
        status: FreshDomainClipStatus::AllClausesRefuted,
    };
    let deadline = Instant::now();

    let declined = decline_fresh_clip_after_deadline(&source, terminal, deadline, deadline);

    assert_eq!(declined.status, FreshDomainClipStatus::Skipped);
    assert_eq!(declined.bounds.lower()[[0]].to_bits(), 0.0_f32.to_bits());
    assert_eq!(declined.bounds.upper()[[0]].to_bits(), 1.0_f32.to_bits());
    let telemetry = FreshDomainClipTelemetry::new(true);
    telemetry.record(&source, &declined);
    assert_eq!(telemetry.snapshot(), (1, 0, 0, 1, 0));
}

/// Verify that `push_batched_relaxed_survivors` checks grouped disjunctive
/// verification on concretized post-clip bounds, not just box infeasibility.
///
/// Without #4367, children whose linear-bound concretization exceeds the grouped
/// thresholds would be pushed to the queue instead of being counted as verified.
///
/// Tests `push_batched_relaxed_survivors` directly with pre-built FlatPendingChild
/// objects to isolate the grouped verification logic from the pipeline.
#[test]
fn test_batched_relaxed_clip_checks_grouped_verification_after_clip_4367() {
    use super::super::push_survivors::push_batched_relaxed_survivors;

    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        enable_relaxed_clip: true,
        input_clip_type: InputClipType::Relaxed,
        relaxed_clip_iterations: 1,
        ..Default::default()
    });

    // Two clauses, each with one row.
    let thresholds = [0.5_f32, 0.5_f32];
    let clause_sizes = [1usize, 1usize];
    let shape = &[1usize]; // 1-dim input

    // Linear bounds: 2 rows (one per threshold), 1 col.
    // lb_row = 5.0 * x_lower + 1.0
    let linear_bounds = LinearBounds::new(
        arr2(&[[5.0_f32], [5.0_f32]]),
        arr1(&[1.0_f32, 1.0_f32]),
        arr2(&[[5.0_f32], [5.0_f32]]),
        arr1(&[1.0_f32, 1.0_f32]),
    )
    .expect("valid linear bounds");

    // Two survivors: one where concretization exceeds threshold, one where it doesn't.
    //
    // Child A [0.0, 1.0]: lb = 5*0 + 1 = 1.0 > 0.5 → should be verified
    // Child B [-0.5, 0.0]: lb = 5*(-0.5) + 1 = -1.5 < 0.5 → should NOT be verified
    //   (after clip, lb tightens toward threshold but doesn't exceed it)
    let survivors = vec![
        FlatPendingChild {
            flat_lower: arr1(&[0.0_f32]).into_dyn(),
            flat_upper: arr1(&[1.0_f32]).into_dyn(),
            obj_bounds: vec![(-1.0, 1.0), (-1.0, 1.0)],
            linear_bounds: Some(linear_bounds.clone()),
            depth: 1,
            priority: 1.0,
            inherited_alpha_state: None,
        },
        FlatPendingChild {
            flat_lower: arr1(&[-0.5_f32]).into_dyn(),
            flat_upper: arr1(&[0.0_f32]).into_dyn(),
            obj_bounds: vec![(-1.0, 1.0), (-1.0, 1.0)],
            linear_bounds: Some(linear_bounds),
            depth: 1,
            priority: 1.0,
            inherited_alpha_state: None,
        },
    ];

    let mut queue = BinaryHeap::new();
    let mut lifecycle = GraphBabLifecycle::new(Instant::now());
    let mut domains_verified_by_clip = 0usize;

    push_batched_relaxed_survivors(
        &verifier,
        survivors,
        shape,
        &thresholds,
        &clause_sizes,
        &mut queue,
        &mut lifecycle,
        &mut domains_verified_by_clip,
    )
    .expect("push_batched_relaxed_survivors should not error");

    // Child A: concretized lb = 1.0 > 0.5 for both rows → both clauses satisfied → verified
    assert_eq!(
        domains_verified_by_clip, 1,
        "child A should be verified by post-clip grouped concretization"
    );
    assert_eq!(lifecycle.domains_verified, 1);
    // Child B: remains in queue
    assert_eq!(
        queue.len(),
        1,
        "child B should be pushed to queue (concretized lb < threshold)"
    );
}

/// Build contradictory linear bounds (row 0: x≤0.2, row 1: x≥0.8 → empty box).
fn lb_infeasible_4366() -> LinearBounds {
    LinearBounds::new(
        arr2(&[[1.0_f32], [-1.0_f32]]),
        arr1(&[-0.2_f32, 0.8_f32]),
        arr2(&[[1.0_f32], [-1.0_f32]]),
        arr1(&[-0.2_f32, 0.8_f32]),
    )
    .expect("valid linear bounds")
}

/// Build bounds with large positive coefficients (lb = 10*x + 5 >> threshold).
fn lb_verified_4366() -> LinearBounds {
    LinearBounds::new(
        arr2(&[[10.0_f32], [10.0_f32]]),
        arr1(&[5.0_f32, 5.0_f32]),
        arr2(&[[10.0_f32], [10.0_f32]]),
        arr1(&[5.0_f32, 5.0_f32]),
    )
    .expect("valid linear bounds")
}

/// Build mild bounds that don't verify (lb = 0.1*x - 10 < threshold).
fn lb_survive_4366() -> LinearBounds {
    LinearBounds::new(
        arr2(&[[0.1_f32], [0.1_f32]]),
        arr1(&[-10.0_f32, -10.0_f32]),
        arr2(&[[0.1_f32], [0.1_f32]]),
        arr1(&[-10.0_f32, -10.0_f32]),
    )
    .expect("valid linear bounds")
}

#[test]
fn test_flat_reorder_survivor_routes_preserve_parent_alpha_f8() {
    use super::super::push_survivors::push_batched_relaxed_survivors;

    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        enable_relaxed_clip: true,
        input_clip_type: InputClipType::Relaxed,
        relaxed_clip_iterations: 1,
        ..Default::default()
    });
    let seed = Arc::new(GraphAlphaState::new());
    let survivors = vec![
        // Direct no-linear-bounds route.
        FlatPendingChild {
            flat_lower: arr1(&[-1.0_f32]).into_dyn(),
            flat_upper: arr1(&[0.0_f32]).into_dyn(),
            obj_bounds: vec![(-1.0, 1.0), (-1.0, 1.0)],
            linear_bounds: None,
            depth: 11,
            priority: 1.0,
            inherited_alpha_state: Some(Arc::clone(&seed)),
        },
        // Batched relaxed-clip survivor route.
        FlatPendingChild {
            flat_lower: arr1(&[0.0_f32]).into_dyn(),
            flat_upper: arr1(&[1.0_f32]).into_dyn(),
            obj_bounds: vec![(-1.0, 1.0), (-1.0, 1.0)],
            linear_bounds: Some(lb_survive_4366()),
            depth: 12,
            priority: 2.0,
            inherited_alpha_state: Some(Arc::clone(&seed)),
        },
    ];
    let mut queue = BinaryHeap::new();
    let mut lifecycle = GraphBabLifecycle::new(Instant::now());
    let mut domains_verified_by_clip = 0usize;

    push_batched_relaxed_survivors(
        &verifier,
        survivors,
        &[1],
        &[0.0, 0.0],
        &[2],
        &mut queue,
        &mut lifecycle,
        &mut domains_verified_by_clip,
    )
    .expect("flat survivor routes should succeed");

    assert_eq!(domains_verified_by_clip, 0);
    assert_eq!(queue.len(), 2);
    let mut depths = Vec::new();
    while let Some(child) = queue.pop() {
        depths.push(child.depth);
        assert!(child.needs_bounding);
        let carried = child
            .inherited_alpha_state
            .as_ref()
            .expect("every flat survivor route must carry alpha");
        assert!(Arc::ptr_eq(carried, &seed));
    }
    depths.sort_unstable();
    assert_eq!(depths, vec![11, 12], "both survivor routes exercised");
}

#[test]
fn test_flat_fallback_survivors_preserve_parent_alpha_disabled_and_complete_f8() {
    use super::super::push_survivors::push_fallback_survivors;

    let graph = build_disjunctive_batch_graph_4353();
    let seed = Arc::new(GraphAlphaState::new());
    for (enable_relaxed_clip, input_clip_type) in [
        (false, InputClipType::Relaxed),
        (true, InputClipType::Complete),
    ] {
        let verifier = BetaCrownVerifier::new(BetaCrownConfig {
            enable_relaxed_clip,
            input_clip_type,
            relaxed_clip_iterations: 1,
            ..Default::default()
        });
        let survivors = vec![FlatPendingChild {
            flat_lower: arr1(&[0.0_f32]).into_dyn(),
            flat_upper: arr1(&[1.0_f32]).into_dyn(),
            obj_bounds: vec![(-1.0, 1.0), (-1.0, 1.0)],
            linear_bounds: Some(lb_survive_4366()),
            depth: 7,
            priority: 1.0,
            inherited_alpha_state: Some(Arc::clone(&seed)),
        }];
        let mut queue = BinaryHeap::new();
        let mut lifecycle = GraphBabLifecycle::new(Instant::now());
        let mut domains_verified_by_clip = 0usize;

        push_fallback_survivors(
            &verifier,
            &graph,
            survivors,
            &[1],
            &[0.0, 0.0],
            None,
            &mut queue,
            &mut lifecycle,
            &mut domains_verified_by_clip,
        )
        .expect("fallback survivor route should succeed");

        let child = queue.pop().expect("fallback survivor must be queued");
        assert!(queue.is_empty());
        assert!(child.needs_bounding);
        let carried = child
            .inherited_alpha_state
            .as_ref()
            .expect("fallback survivor must carry alpha");
        assert!(Arc::ptr_eq(carried, &seed));
    }
}

fn build_three_child_survivors_4366() -> Vec<FlatPendingChild> {
    vec![
        FlatPendingChild {
            flat_lower: arr1(&[0.0_f32]).into_dyn(),
            flat_upper: arr1(&[1.0_f32]).into_dyn(),
            obj_bounds: vec![(-1.0, 1.0), (-1.0, 1.0)],
            linear_bounds: Some(lb_infeasible_4366()),
            depth: 1,
            priority: 1.0,
            inherited_alpha_state: None,
        },
        FlatPendingChild {
            flat_lower: arr1(&[0.5_f32]).into_dyn(),
            flat_upper: arr1(&[1.0_f32]).into_dyn(),
            obj_bounds: vec![(-1.0, 1.0), (-1.0, 1.0)],
            linear_bounds: Some(lb_verified_4366()),
            depth: 2,
            priority: 2.0,
            inherited_alpha_state: None,
        },
        FlatPendingChild {
            flat_lower: arr1(&[0.0_f32]).into_dyn(),
            flat_upper: arr1(&[1.0_f32]).into_dyn(),
            obj_bounds: vec![(-1.0, 1.0), (-1.0, 1.0)],
            linear_bounds: Some(lb_survive_4366()),
            depth: 3,
            priority: 3.0,
            inherited_alpha_state: None,
        },
    ]
}

/// Regression for #4366 batched clip: three children with distinct dispositions.
/// Child A: infeasible by clip. Child B: verified by grouped check. Child C: queued.
///
/// The infeasibility of child A comes from COMBINING its two rows (x<=0.2 AND
/// x>=0.8 -> empty), i.e. a single conjunctive clause of two rows — so this uses
/// `clause_sizes = [2]`, not `[1, 1]`. Under the clause-aware clip
/// (#disj-cross-clause-clip-unsat) `[1, 1]` would be two INDEPENDENT OR clauses,
/// each individually feasible, and child A would correctly become a survivor
/// (the lsnc false-unsat pattern) — the disjunctive survivor path is covered by
/// `test_batched_clip_disjoint_or_clause_survives_disj`.
#[test]
fn test_batched_clip_three_children_mixed_dispositions_4366() {
    use super::super::push_survivors::push_batched_relaxed_survivors;

    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        enable_relaxed_clip: true,
        input_clip_type: InputClipType::Relaxed,
        relaxed_clip_iterations: 3,
        ..Default::default()
    });

    let thresholds = [0.0_f32, 0.0_f32];
    let clause_sizes = [2usize];
    let shape = &[1usize];
    let mut queue = BinaryHeap::new();
    let mut lifecycle = GraphBabLifecycle::new(Instant::now());
    let mut domains_verified_by_clip = 0usize;

    push_batched_relaxed_survivors(
        &verifier,
        build_three_child_survivors_4366(),
        shape,
        &thresholds,
        &clause_sizes,
        &mut queue,
        &mut lifecycle,
        &mut domains_verified_by_clip,
    )
    .expect("batched clip with 3 children should not error");

    assert_eq!(
        domains_verified_by_clip, 2,
        "children A (infeasible) and B (grouped-verified) should both be counted"
    );
    assert_eq!(lifecycle.domains_verified, 2);
    assert_eq!(queue.len(), 1, "only child C should remain in queue");
    let queued = queue.pop().expect("one child in queue");
    assert_eq!(queued.depth, 3, "queued child should be child C (depth=3)");
    assert!(queued.needs_bounding, "queued child needs bounding");
}

/// #disj-cross-clause-clip-unsat plumbing (flat path): the SAME two rows as
/// `lb_infeasible_4366` (x<=0.2, x>=0.8) but as TWO independent OR clauses
/// (`clause_sizes = [1, 1]`) are each individually feasible over [0, 1], so the
/// child must NOT be clip-verified (this is exactly the lsnc false-unsat that
/// the historical cross-clause clip produced). The child survives to the queue,
/// carrying the UNION box, which encloses both [0, 0.2] and [0.8, 1] (i.e. all
/// of [0, 1]) so no counterexample is discarded.
#[test]
fn test_batched_clip_disjoint_or_clause_survives_disj() {
    use super::super::push_survivors::push_batched_relaxed_survivors;

    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        enable_relaxed_clip: true,
        input_clip_type: InputClipType::Relaxed,
        relaxed_clip_iterations: 3,
        ..Default::default()
    });

    let thresholds = [0.0_f32, 0.0_f32];
    let clause_sizes = [1usize, 1usize]; // two OR clauses, one row each
    let shape = &[1usize];

    let survivors = vec![FlatPendingChild {
        flat_lower: arr1(&[0.0_f32]).into_dyn(),
        flat_upper: arr1(&[1.0_f32]).into_dyn(),
        obj_bounds: vec![(-1.0, 1.0), (-1.0, 1.0)],
        linear_bounds: Some(lb_infeasible_4366()),
        depth: 1,
        priority: 1.0,
        inherited_alpha_state: None,
    }];

    let mut queue = BinaryHeap::new();
    let mut lifecycle = GraphBabLifecycle::new(Instant::now());
    let mut domains_verified_by_clip = 0usize;

    push_batched_relaxed_survivors(
        &verifier,
        survivors,
        shape,
        &thresholds,
        &clause_sizes,
        &mut queue,
        &mut lifecycle,
        &mut domains_verified_by_clip,
    )
    .expect("push_batched_relaxed_survivors should not error");

    assert_eq!(
        domains_verified_by_clip, 0,
        "two individually-feasible OR clauses must not be clip-verified"
    );
    assert_eq!(lifecycle.domains_verified, 0);
    assert_eq!(queue.len(), 1, "the survivor must reach the queue");
    let queued = queue.pop().expect("one survivor queued");
    // Union box encloses [0, 0.2] ∪ [0.8, 1] -> essentially [0, 1].
    assert!(
        queued.input_bounds.lower()[[0]] <= 1e-6,
        "union lower must reach 0, got {}",
        queued.input_bounds.lower()[[0]]
    );
    assert!(
        queued.input_bounds.upper()[[0]] >= 1.0 - 1e-6,
        "union upper must reach 1, got {}",
        queued.input_bounds.upper()[[0]]
    );
}

/// Regression: the joint multi-spec clip can find infeasibility from combined
/// constraints that no single row can prove alone.
///
/// Two 1D constraints: row 0 says x ≤ 0.3 (lA=[1], lb=-0.3, thresh=0),
/// row 1 says x ≥ 0.7 (lA=[-1], lb=0.7, thresh=0).
/// Neither row alone makes [0, 1] infeasible, but together they prove the child
/// box is empty (must be ≤0.3 AND ≥0.7).
///
/// Part of #4367 acceptance criteria: "a regression proves the grouped path can
/// eliminate a child through combined multi-spec clipping that no single row
/// proves alone."
#[test]
fn test_joint_multispec_clip_finds_combined_infeasibility_4367() {
    use super::super::push_survivors::push_batched_relaxed_survivors;

    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        enable_relaxed_clip: true,
        input_clip_type: InputClipType::Relaxed,
        relaxed_clip_iterations: 3,
        ..Default::default()
    });

    // ONE conjunctive clause of two rows: BOTH rows must hold for a
    // counterexample, so combining them proves the child empty. (This is the
    // #4367 "combined infeasibility" capability; expressing it as `[1, 1]` would
    // be two INDEPENDENT OR clauses — each individually feasible — which under
    // the clause-aware clip (#disj-cross-clause-clip-unsat) is correctly NOT
    // infeasible.)
    let thresholds = [0.0_f32, 0.0_f32];
    let clause_sizes = [2usize];
    let shape = &[1usize];

    // Row 0: lA=1, lb=-0.3 → 1*x - 0.3 > 0 → x > 0.3 (clips upper to ~0.3)
    // Row 1: lA=-1, lb=0.7 → -1*x + 0.7 > 0 → x < 0.7 (clips lower to ~0.7)
    // Together (AND): x ≤ 0.3 AND x ≥ 0.7 → empty box.
    let linear_bounds = LinearBounds::new(
        arr2(&[[1.0_f32], [-1.0_f32]]),
        arr1(&[-0.3_f32, 0.7_f32]),
        arr2(&[[1.0_f32], [-1.0_f32]]),
        arr1(&[-0.3_f32, 0.7_f32]),
    )
    .expect("valid linear bounds");

    let survivors = vec![FlatPendingChild {
        flat_lower: arr1(&[0.0_f32]).into_dyn(),
        flat_upper: arr1(&[1.0_f32]).into_dyn(),
        obj_bounds: vec![(-1.0, 1.0), (-1.0, 1.0)],
        linear_bounds: Some(linear_bounds),
        depth: 1,
        priority: 1.0,
        inherited_alpha_state: None,
    }];

    let mut queue = BinaryHeap::new();
    let mut lifecycle = GraphBabLifecycle::new(Instant::now());
    let mut domains_verified_by_clip = 0usize;

    push_batched_relaxed_survivors(
        &verifier,
        survivors,
        shape,
        &thresholds,
        &clause_sizes,
        &mut queue,
        &mut lifecycle,
        &mut domains_verified_by_clip,
    )
    .expect("push_batched_relaxed_survivors should not error");

    // The joint multi-spec clip should find the child infeasible from combined
    // constraints. Either infeasible_after_clip is true, or the post-clip
    // grouped verification catches it.
    assert_eq!(
        domains_verified_by_clip, 1,
        "joint multi-spec clip should verify child from combined constraints"
    );
    assert_eq!(lifecycle.domains_verified, 1);
    assert_eq!(
        queue.len(),
        0,
        "no children should be queued — combined constraints prove the child infeasible"
    );
}
