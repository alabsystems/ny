// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #disj-odi tests. The driver-level tests run the REAL batched disjunctive
//! driver (`run_disj_vjp_batched`) on small Linear→ReLU→Linear graphs; the
//! only stand-in is the wide point-VJP pass, served on the CPU by
//! [`CpuChainVjp`] from the same weights the graph carries, which also records
//! every pass (lane count, spec rows, ReLU masks) so lane trajectories can be
//! compared call for call.

use std::sync::Mutex;

use ndarray::{arr1, Array2};
use ny_core::{GpuCrownLayer, GpuCrownResult, NaiveCpuGemmEngine, NyError};
use ny_propagate::layers::{LinearLayer, ReLULayer};
use ny_propagate::{Layer, Network, PgdConfig};

use super::super::attack_stall::AttackStallPolicy;
use super::super::graph_pgd_vjp_batched_disj::{run_disj_vjp_batched, DisjVjpBatchedOutcome};
use super::*;

/// One recorded wide point-VJP pass.
#[derive(Debug, Clone, PartialEq)]
struct VjpCall {
    lanes: usize,
    spec_rows: Vec<f32>,
    masks: Vec<Vec<Vec<f32>>>,
}

/// CPU stand-in for the wide point-VJP pass on a Linear→ReLU→Linear chain:
/// `grad_k = spec_row_k · W2 · diag(mask_k) · W1`, the exact gradient the GPU
/// pass computes for that template. GEMMs go to the naive CPU engine.
struct CpuChainVjp {
    /// `hidden × input`.
    w1: Vec<Vec<f32>>,
    /// `output × hidden`.
    w2: Vec<Vec<f32>>,
    calls: Mutex<Vec<VjpCall>>,
}

impl GemmEngine for CpuChainVjp {
    fn gemm_f32(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
    ) -> ny_core::Result<Vec<f32>> {
        NaiveCpuGemmEngine.gemm_f32(m, k, n, a, b)
    }

    fn as_gpu_crown_backward(&self) -> Option<&dyn GpuCrownBackward> {
        Some(self)
    }
}

impl GpuCrownBackward for CpuChainVjp {
    fn crown_backward_gpu(
        &self,
        _layers: &[GpuCrownLayer],
        _spec: &[f32],
        _num_specs: usize,
        _input_lower: &[f32],
        _input_upper: &[f32],
    ) -> ny_core::Result<GpuCrownResult> {
        Err(NyError::UnsupportedOp(
            "the #disj-odi test engine serves point-VJP only".into(),
        ))
    }

    fn crown_point_vjp_batched(
        &self,
        _layers_backward: &[GpuCrownLayer],
        _mask_positions: &[usize],
        masks: &[Vec<Vec<f32>>],
        spec_rows: &[f32],
        output_dim: usize,
        input_dim: usize,
    ) -> ny_core::Result<Vec<Vec<f32>>> {
        let lanes = masks.len();
        if spec_rows.len() != lanes * output_dim {
            return Err(NyError::UnsupportedOp(
                "spec rows do not match the lane count".into(),
            ));
        }
        self.calls.lock().unwrap().push(VjpCall {
            lanes,
            spec_rows: spec_rows.to_vec(),
            masks: masks.to_vec(),
        });
        let mut grads = Vec::with_capacity(lanes);
        for lane in 0..lanes {
            let row = &spec_rows[lane * output_dim..(lane + 1) * output_dim];
            let mask = &masks[lane][0];
            let mut grad = vec![0.0_f32; input_dim];
            for (hidden, w1_row) in self.w1.iter().enumerate() {
                let back: f32 = (0..output_dim)
                    .map(|out| row[out] * self.w2[out][hidden])
                    .sum::<f32>()
                    * mask[hidden];
                for (g, &w) in grad.iter_mut().zip(w1_row.iter()) {
                    *g += back * w;
                }
            }
            grads.push(grad);
        }
        Ok(grads)
    }
}

struct ChainModel {
    w1: Vec<Vec<f32>>,
    b1: Vec<f32>,
    w2: Vec<Vec<f32>>,
    b2: Vec<f32>,
}

fn matrix(rows: &[Vec<f32>]) -> Array2<f32> {
    Array2::from_shape_vec((rows.len(), rows[0].len()), rows.concat()).unwrap()
}

impl ChainModel {
    fn graph(&self) -> GraphNetwork {
        let mut network = Network::new();
        network.add_layer(Layer::Linear(
            LinearLayer::new(matrix(&self.w1), Some(arr1(&self.b1))).unwrap(),
        ));
        network.add_layer(Layer::ReLU(ReLULayer));
        network.add_layer(Layer::Linear(
            LinearLayer::new(matrix(&self.w2), Some(arr1(&self.b2))).unwrap(),
        ));
        GraphNetwork::from_sequential(&network).unwrap()
    }

    fn engine(&self) -> CpuChainVjp {
        CpuChainVjp {
            w1: self.w1.clone(),
            w2: self.w2.clone(),
            calls: Mutex::new(Vec::new()),
        }
    }
}

/// Two inputs in `[-1, 1]^2`, four hidden units whose ReLUs flip inside the
/// box (so the masks track each lane's point), three outputs. `Y_0` carries a
/// +20 bias and `|Y_1|, |Y_2| <= 7`, so neither clause can ever hold: the
/// driver runs every step and never produces a candidate.
fn safe_model() -> ChainModel {
    ChainModel {
        w1: vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![1.0, -1.0],
        ],
        b1: vec![0.05, 0.2, -0.3, 0.1],
        w2: vec![
            vec![0.0, 0.0, 0.0, 0.0],
            vec![1.0, -1.0, 0.5, 0.0],
            vec![0.0, 1.0, -0.5, 1.0],
        ],
        b2: vec![20.0, 0.0, 0.0],
    }
}

/// One input in `[-1, 1]`, an always-active hidden unit (`h = x + 2`), and
/// `Y_1 - Y_0 = 5e-5 * (x + 2) - 1.2e-4 in [-7e-5, 3e-5]`: clause 0 HOLDS on
/// the template forward for `x >= 0.4`, but its margin never reaches the exact
/// re-screen floor `max(1e-4, 1e-5 * max|y|) = 1e-4`. Clause 1 (`Y_2 = -1`)
/// never holds. The clean point (`x = 0`, margin -2e-5) is not a hit.
fn near_miss_model() -> ChainModel {
    ChainModel {
        w1: vec![vec![1.0]],
        b1: vec![2.0],
        w2: vec![vec![0.0], vec![5e-5], vec![0.0]],
        b2: vec![0.0, -1.2e-4, -1.0],
    }
}

fn unit_box(dims: usize) -> BoundedTensor {
    BoundedTensor::new(
        arr1(&vec![-1.0_f32; dims]).into_dyn(),
        arr1(&vec![1.0_f32; dims]).into_dyn(),
    )
    .unwrap()
}

/// `(or (and (>= Y_1 Y_0)) (and (>= Y_2 Y_0)))`: label 0, classes 1 and 2.
fn two_clauses() -> Vec<Vec<OutputConstraint>> {
    vec![
        vec![OutputConstraint::GreaterEq(1, 0)],
        vec![OutputConstraint::GreaterEq(2, 0)],
    ]
}

fn pgd(num_restarts: usize, num_steps: usize, exact_vjp: ExactVjpPgdConfig) -> PgdConfig {
    PgdConfig {
        num_restarts,
        num_steps,
        exact_vjp,
        ..PgdConfig::default()
    }
}

fn fresh_outcome() -> GraphDisjunctiveAttackOutcome {
    GraphDisjunctiveAttackOutcome {
        candidate: None,
        best_margin: f32::NEG_INFINITY,
        hit_deadline: false,
        stalled_out: false,
        restarts_started: 0,
        steps_taken: 0,
    }
}

struct Run {
    result: DisjVjpBatchedOutcome,
    outcome: GraphDisjunctiveAttackOutcome,
    receipt: DisjOdiReceipt,
    calls: Vec<VjpCall>,
}

fn run(model: &ChainModel, input: &BoundedTensor, config: &PgdConfig) -> Run {
    let graph = model.graph();
    let engine = model.engine();
    let clauses = two_clauses();
    let mut outcome = fresh_outcome();
    let mut receipt = DisjOdiReceipt::new(DisjOdiPolicy::from_config(config.exact_vjp));
    let result = run_disj_vjp_batched(
        &graph,
        input,
        &clauses,
        config,
        Some(&engine),
        true,
        AttackStallPolicy::disabled(),
        &mut outcome,
        &mut receipt,
    )
    .expect("the batched disjunctive driver runs on the test chain");
    let calls = engine.calls.into_inner().unwrap();
    Run {
        result,
        outcome,
        receipt,
        calls,
    }
}

fn first_coordinate(point: &ArrayD<f32>) -> f32 {
    point.iter().copied().next().expect("a non-empty point")
}

fn lanes_per_call(calls: &[VjpCall]) -> Vec<usize> {
    calls.iter().map(|call| call.lanes).collect()
}

// ---------------------------------------------------------------------------
// Named tests (FRONTIER_PLAN §7.F1)
// ---------------------------------------------------------------------------

/// Unarmed, the driver ignores every field #disj-odi composes with: moving
/// `odi_steps`, `odi_eta` and `hinge_tau` changes neither a single wide-VJP
/// pass (lane count, spec rows, masks) nor any outcome counter, and no ODI or
/// focus work is issued.
#[test]
fn disj_odi_dark_is_byte_identical() {
    let model = safe_model();
    let input = unit_box(2);
    let base = run(&model, &input, &pgd(4, 5, ExactVjpPgdConfig::default()));
    let moved = run(
        &model,
        &input,
        &pgd(
            4,
            5,
            ExactVjpPgdConfig {
                odi_steps: 0,
                odi_eta: 0.25,
                hinge_tau: 0.01,
                ..ExactVjpPgdConfig::default()
            },
        ),
    );
    for arm in [&base, &moved] {
        assert!(matches!(arm.result, DisjVjpBatchedOutcome::Completed));
        assert!(!arm.receipt.policy().armed());
        assert_eq!(
            (
                arm.receipt.odi_vjp_steps,
                arm.receipt.odi_move_count,
                arm.receipt.odi_lanes,
                arm.receipt.focus_steps,
                arm.receipt.near_misses,
            ),
            (0, 0, 0, 0, 0)
        );
        assert_eq!(arm.receipt.focus_reason, "off");
        assert!(arm.outcome.candidate.is_none());
    }
    assert_eq!(
        lanes_per_call(&base.calls),
        vec![4; 5],
        "one wide VJP per step of the single four-lane wave and no ODI pass"
    );
    assert_eq!(base.calls, moved.calls, "identical lane trajectories");
    assert_eq!(
        base.outcome.best_margin.to_bits(),
        moved.outcome.best_margin.to_bits()
    );
    assert_eq!(
        (
            base.outcome.steps_taken,
            base.outcome.restarts_started,
            base.outcome.hit_deadline,
            base.outcome.stalled_out,
        ),
        (
            moved.outcome.steps_taken,
            moved.outcome.restarts_started,
            moved.outcome.hit_deadline,
            moved.outcome.stalled_out,
        )
    );
    assert_eq!(
        (base.outcome.steps_taken, base.outcome.restarts_started),
        (20, 4)
    );
}

/// Armed, every wave's DYNAMIC lanes take exactly `odi_steps` ODI passes
/// (`odi_vjp_steps == odi_steps × waves`), the passes cover the dynamic lanes
/// only, the points move, and the TARGETED lanes' trajectories are the dark
/// driver's, pass for pass.
#[test]
fn disj_odi_moves_lanes_and_counts_vjp_steps() {
    let model = safe_model();
    let input = unit_box(2);
    let dark_policy = ExactVjpPgdConfig {
        batch_width: Some(4),
        ..ExactVjpPgdConfig::default()
    };
    let armed_policy = ExactVjpPgdConfig {
        disj_odi: true,
        ..dark_policy
    };
    // Six restarts in waves of four: wave 0 = {targeted 0, dynamic 1,
    // targeted 2, dynamic 3}; wave 1 = {dynamic 4, dynamic 5} (only two
    // clauses rank, so restart 4 has no fixed target).
    let armed = run(&model, &input, &pgd(6, 3, armed_policy));
    let dark = run(&model, &input, &pgd(6, 3, dark_policy));

    assert!(matches!(armed.result, DisjVjpBatchedOutcome::Completed));
    assert!(armed.outcome.candidate.is_none());
    let waves = 2;
    assert_eq!(armed.receipt.odi_vjp_steps, armed_policy.odi_steps * waves);
    assert_eq!(armed.receipt.odi_waves, waves);
    assert_eq!(armed.receipt.odi_lanes, 4, "two dynamic lanes per wave");
    assert!(
        armed.receipt.odi_move_count > 0,
        "an ODI step that moves nothing is a VOID arm"
    );
    let odi = armed_policy.odi_steps;
    assert_eq!(
        lanes_per_call(&armed.calls),
        [vec![2_usize; odi], vec![4; 3], vec![2; odi], vec![2; 3]].concat(),
        "per wave: ODI passes over the dynamic lanes, then the joint steps over every lane"
    );
    assert_eq!(
        lanes_per_call(&dark.calls),
        [vec![4_usize; 3], vec![2; 3]].concat()
    );

    // Targeted lanes 0 and 2 of wave 0 keep their clean init: their masks and
    // rows in every joint step equal the dark driver's.
    let out_dim = 3;
    for step in 0..3 {
        let armed_call = &armed.calls[odi + step];
        let dark_call = &dark.calls[step];
        for lane in [0usize, 2] {
            assert_eq!(armed_call.masks[lane], dark_call.masks[lane]);
            assert_eq!(
                armed_call.spec_rows[lane * out_dim..(lane + 1) * out_dim],
                dark_call.spec_rows[lane * out_dim..(lane + 1) * out_dim]
            );
        }
    }
    // The ODI passes ascend random output directions, not a clause row.
    let first = &armed.calls[0];
    assert!(first
        .spec_rows
        .iter()
        .all(|value| (-1.0..=1.0).contains(value)));
    assert!(first.spec_rows.iter().any(|&value| value != 0.0));
}

/// Focus lanes: every focus ascent pass carries, for lane `i`, exactly the
/// margin row of the `i % 2`-th ranked clause (label → class `t`, row
/// `e_t − e_0`), the ranking comes from the restart waves (`wave0`), the
/// restart waves themselves are untouched, and the tau hinge zeroes a lane's
/// row exactly when its clause margin reaches tau.
#[test]
fn disj_focus_lanes_pin_the_selected_class_and_respect_the_hinge() {
    // --- the hinge rule itself ---
    let output = arr1(&[0.0_f32, 0.3, -1.0]).into_dyn();
    let clause = vec![OutputConstraint::GreaterEq(1, 0)];
    let (row, live) = focus_spec_row(&output, &clause, 0.5, 3);
    assert!(live, "margin 0.3 < tau 0.5 keeps the gradient");
    assert_eq!(row, vec![-1.0, 1.0, 0.0]);
    for tau in [0.3_f32, 0.2, 0.0] {
        let (row, live) = focus_spec_row(&output, &clause, tau, 3);
        assert!(!live, "margin 0.3 >= tau {tau} stops contributing gradient");
        assert_eq!(row, vec![0.0; 3]);
    }
    // Either vnnlib spelling pins the same class.
    let spelled_le = vec![OutputConstraint::LessEq(0, 1)];
    assert_eq!(
        focus_spec_row(&output, &spelled_le, 0.5, 3).0,
        vec![-1.0, 1.0, 0.0]
    );
    assert_eq!(focus_clause_label(&[spelled_le], 0), "1");

    // --- pinned classes on the real driver ---
    let model = safe_model();
    let input = unit_box(2);
    let focus_policy = ExactVjpPgdConfig {
        disj_focus_lanes: 6,
        odi_steps: 2,
        ..ExactVjpPgdConfig::default()
    };
    let focus = run(&model, &input, &pgd(4, 3, focus_policy));
    let dark = run(
        &model,
        &input,
        &pgd(
            4,
            3,
            ExactVjpPgdConfig {
                odi_steps: 2,
                ..ExactVjpPgdConfig::default()
            },
        ),
    );
    assert!(matches!(focus.result, DisjVjpBatchedOutcome::Completed));
    assert!(focus.outcome.candidate.is_none());
    assert_eq!(focus.receipt.focus_rule, "wave0");
    assert_eq!(focus.receipt.focus_admitted, 6);
    assert_eq!(focus.receipt.focus_reason, "exhausted");
    assert_eq!(focus.receipt.focus_steps, FOCUS_SCHEDULE_STEPS);
    let mut classes = focus.receipt.focus_classes.clone();
    classes.sort();
    assert_eq!(classes, vec!["1".to_string(), "2".to_string()]);
    // `attack.pgd_disj_odi` is dark here: only the focus lanes took ODI.
    assert_eq!(focus.receipt.odi_vjp_steps, 2);
    assert_eq!(focus.receipt.odi_lanes, 6);
    // Restart waves: the dark driver's passes, unchanged.
    assert_eq!(focus.calls[..3], dark.calls[..]);
    assert_eq!(
        lanes_per_call(&focus.calls),
        [vec![4_usize; 3], vec![6; 2], vec![6; FOCUS_SCHEDULE_STEPS]].concat()
    );
    // Focus restarts never count as restarts (the attack extension continues
    // the restart seed sequence from this number).
    assert_eq!(
        focus.outcome.restarts_started,
        dark.outcome.restarts_started
    );
    assert!(!focus.outcome.hit_deadline);

    let out_dim = 3;
    let pinned: Vec<usize> = focus
        .receipt
        .focus_classes
        .iter()
        .map(|label| label.parse::<usize>().expect("relational label"))
        .collect();
    for call in &focus.calls[5..] {
        for lane in 0..call.lanes {
            let class = pinned[lane % pinned.len()];
            let mut expected = vec![0.0_f32; out_dim];
            expected[class] = 1.0;
            expected[0] = -1.0;
            assert_eq!(
                call.spec_rows[lane * out_dim..(lane + 1) * out_dim],
                expected[..],
                "focus lane {lane} must ascend class {class} (margins stay below tau = 0)"
            );
        }
    }
}

/// A template-forward hit whose exact margin sits below the re-screen floor
/// is a near miss: armed ODI and focus lanes reach it over and over, count it,
/// and the driver still returns NO candidate.
#[test]
fn disj_odi_near_miss_below_floor_never_returns_a_candidate() {
    let model = near_miss_model();
    let input = unit_box(1);
    let armed = run(
        &model,
        &input,
        &pgd(
            4,
            5,
            ExactVjpPgdConfig {
                disj_odi: true,
                disj_focus_lanes: 16,
                odi_steps: 2,
                ..ExactVjpPgdConfig::default()
            },
        ),
    );
    assert!(matches!(armed.result, DisjVjpBatchedOutcome::Completed));
    assert!(
        armed.outcome.candidate.is_none(),
        "a margin below noise_scaled_margin must never be handed up"
    );
    assert!(
        armed.outcome.best_margin >= 0.0 && armed.outcome.best_margin < 1e-4,
        "the template forward did reach the clause (best margin {})",
        armed.outcome.best_margin
    );
    assert!(armed.receipt.near_misses > 0, "the screen must have fired");
    assert_eq!(armed.receipt.focus_reason, "exhausted");
    assert_eq!(armed.receipt.focus_steps, FOCUS_SCHEDULE_STEPS);
}

// ---------------------------------------------------------------------------
// Pure rules
// ---------------------------------------------------------------------------

#[test]
fn disj_odi_step_rules_match_the_calibrated_recipe() {
    let lower = arr1(&[-1.0_f32, -1.0, -1.0, -1.0]).into_dyn();
    let upper = arr1(&[1.0_f32, 1.0, 1.0, 1.0]).into_dyn();
    let mut point = arr1(&[0.0_f32, 0.5, -1.0, 0.9]).into_dyn();
    // eta 0.5 → a step of 0.5 half-ranges = 0.5; zero and NaN gradients stay.
    let moves = odi_sign_step(&mut point, &[1.0, -1.0, 0.0, f32::NAN], &lower, &upper, 0.5);
    assert_eq!(moves, 2);
    assert_eq!(
        point.iter().copied().collect::<Vec<_>>(),
        vec![0.5, 0.0, -1.0, 0.9]
    );
    // eta 1.0 clamps at the box.
    let moves = odi_sign_step(&mut point, &[1.0, 1.0, -1.0, 1.0], &lower, &upper, 1.0);
    assert_eq!(
        point.iter().copied().collect::<Vec<_>>(),
        vec![1.0, 1.0, -1.0, 1.0]
    );
    assert_eq!(moves, 3, "a coordinate already at its bound does not move");

    // a_k / h = 0.15 (1 + cos(pi k / S)) + 0.003.
    assert!((focus_step_scale(0, 200) - 0.303).abs() < 1e-6);
    assert!((focus_step_scale(100, 200) - 0.153).abs() < 1e-6);
    assert!((focus_step_scale(200, 200) - 0.003).abs() < 1e-6);

    // Momentum-sign: a hinge-held lane (zero gradient) coasts on its momentum.
    let lower = arr1(&[-1.0_f32]).into_dyn();
    let upper = arr1(&[1.0_f32]).into_dyn();
    let mut point = arr1(&[0.0_f32]).into_dyn();
    let mut momentum = vec![0.0_f32];
    assert_eq!(
        focus_momentum_step(&mut point, &mut momentum, &[1.0], &lower, &upper, 0.25),
        1
    );
    assert!((first_coordinate(&point) - 0.25).abs() < 1e-7);
    assert_eq!(
        focus_momentum_step(&mut point, &mut momentum, &[0.0], &lower, &upper, 0.25),
        1
    );
    assert!((first_coordinate(&point) - 0.5).abs() < 1e-7);
    // A fresh lane with no gradient does not move.
    let mut still = arr1(&[0.0_f32]).into_dyn();
    let mut zero = vec![0.0_f32];
    assert_eq!(
        focus_momentum_step(&mut still, &mut zero, &[0.0], &lower, &upper, 0.25),
        0
    );
}

#[test]
fn disj_focus_ranking_and_admission() {
    let best = [-0.5_f32, f32::NEG_INFINITY, 0.1, -0.2, f32::NAN];
    assert_eq!(rank_focus_clauses(&best, 2), vec![2, 3]);
    assert_eq!(rank_focus_clauses(&best, 10), vec![2, 3, 0]);
    assert_eq!((rank_focus_clauses(&[f32::NEG_INFINITY], 2)).len(), 0);

    let clauses = vec![
        vec![
            OutputConstraint::GreaterEq(1, 0),
            OutputConstraint::GreaterEq(2, 0),
        ],
        vec![OutputConstraint::GreaterEqConst(2, 5.0)],
    ];
    let output = arr1(&[1.0_f32, 0.5, 3.0]).into_dyn();
    let (margin, target) = clause_bottleneck(&output, &clauses[0]).expect("modeled clause");
    assert!(
        (margin + 0.5).abs() < 1e-7,
        "the bottleneck is the smaller margin"
    );
    assert!(matches!(target, GraphPgdTarget::Relational(1, 0)));
    let mut seen = vec![f32::NEG_INFINITY; 2];
    observe_clause_margins(&mut seen, &output, &clauses);
    assert!((seen[0] + 0.5).abs() < 1e-7);
    assert!((seen[1] + 2.0).abs() < 1e-7);
    assert_eq!(focus_clause_label(&clauses, 0), "1");
    assert_eq!(focus_clause_label(&clauses, 1), "c1");

    assert_eq!(focus_lane_bytes(100, 10, 3), 100 * 16 + 10 * 24 + 3 * 16);
    assert_eq!(admit_focus_lanes(64, 1 << 20, Some(1 << 30)), 64);
    assert_eq!(
        admit_focus_lanes(10_000, 1, Some(u64::MAX)),
        FOCUS_MAX_LANES,
        "the hard cap binds"
    );
    assert_eq!(
        admit_focus_lanes(64, 1 << 30, Some(1 << 30)),
        0,
        "a lane larger than half the headroom is refused"
    );
    assert_eq!(admit_focus_lanes(64, 1 << 20, None), 64);
    assert_eq!(
        admit_focus_lanes(8192, 1 << 20, None),
        FOCUS_MAX_LANES.min(4096),
        "unobservable headroom admits at most 4 GiB"
    );
}

#[test]
fn disj_odi_receipt_schema_is_fixed() {
    let dark = DisjOdiPolicy::from_config(ExactVjpPgdConfig::default());
    assert!(!dark.armed());
    assert_eq!(
        dark.wave_odi_steps(),
        0,
        "the dark driver takes no ODI step"
    );
    assert!(!dark.tracks_focus());

    let armed = DisjOdiPolicy::from_config(ExactVjpPgdConfig {
        disj_odi: true,
        disj_focus_lanes: 32,
        ..ExactVjpPgdConfig::default()
    });
    assert!(armed.armed());
    assert_eq!(armed.wave_odi_steps(), 8);
    let line = DisjOdiReceipt::new(armed).line();
    for token in [
        "[pgd-vjp-disj] disj-odi schema=v1 ",
        " odi=1 ",
        " odi_steps=8 ",
        " odi_eta=1 ",
        " odi_waves=0 ",
        " odi_lanes=0 ",
        " odi_vjp_steps=0 ",
        " odi_move_count=0 ",
        " near_misses=0 ",
        " focus_lanes=32 ",
        " focus_admitted=0 ",
        " focus_rule=none ",
        " focus_class=none ",
        " focus_steps=0 ",
        " focus_retries=0 ",
        " focus_tau=0 ",
        " focus_best_margin=none ",
        " focus_reason=not_reached ",
        " outcome=internal_error",
    ] {
        assert!(line.contains(token), "missing {token:?} in {line}");
    }
    assert_eq!(DisjOdiReceipt::new(dark).focus_reason, "off");
}

// ---------------------------------------------------------------------------
// #attack-setup-regrant
// ---------------------------------------------------------------------------

fn expired_deadline() -> Instant {
    Instant::now()
        .checked_sub(std::time::Duration::from_secs(1))
        .expect("one second before now is representable")
}

/// A slice that already expired before the driver could start is refused
/// before plan construction exactly as before when the phase offers no
/// re-grant: zero steps, zero restarts, no wide-VJP pass.
#[test]
fn setup_regrant_absent_keeps_the_deadline_before_start_refusal() {
    let model = safe_model();
    let input = unit_box(2);
    let config = PgdConfig {
        deadline: Some(expired_deadline()),
        ..pgd(4, 5, ExactVjpPgdConfig::default())
    };
    let refused = run(&model, &input, &config);
    assert!(matches!(refused.result, DisjVjpBatchedOutcome::Completed));
    assert!(refused.outcome.hit_deadline);
    assert_eq!(
        (
            refused.outcome.steps_taken,
            refused.outcome.restarts_started
        ),
        (0, 0)
    );
    assert_eq!((refused.calls).len(), 0);
}

/// The same expired slice WITH a re-grant: the driver builds its plan and
/// searches for the re-granted slice — here long enough for the whole
/// configured schedule — so the run is call-for-call the unbounded run.
#[test]
fn setup_regrant_searches_an_expired_slice_once_the_plan_is_ready() {
    let model = safe_model();
    let input = unit_box(2);
    let now = Instant::now();
    let regranted = PgdConfig {
        deadline: Some(expired_deadline()),
        setup_regrant: Some(ny_propagate::PgdSetupRegrant {
            slice: std::time::Duration::from_secs(60),
            cap: now + std::time::Duration::from_secs(120),
        }),
        ..pgd(4, 5, ExactVjpPgdConfig::default())
    };
    let unbounded = run(&model, &input, &pgd(4, 5, ExactVjpPgdConfig::default()));
    let arm = run(&model, &input, &regranted);
    assert!(matches!(arm.result, DisjVjpBatchedOutcome::Completed));
    assert!(!arm.outcome.hit_deadline, "the re-granted slice is ample");
    assert_eq!(
        (arm.outcome.steps_taken, arm.outcome.restarts_started),
        (20, 4)
    );
    assert_eq!(arm.calls, unbounded.calls, "identical lane trajectories");
}

/// A spent cap re-grants nothing: the refusal stands.
#[test]
fn setup_regrant_with_a_spent_cap_refuses() {
    let model = safe_model();
    let input = unit_box(2);
    let spent = PgdConfig {
        deadline: Some(expired_deadline()),
        setup_regrant: Some(ny_propagate::PgdSetupRegrant {
            slice: std::time::Duration::from_secs(60),
            cap: expired_deadline(),
        }),
        ..pgd(4, 5, ExactVjpPgdConfig::default())
    };
    let arm = run(&model, &input, &spent);
    assert!(arm.outcome.hit_deadline);
    assert_eq!(
        (arm.outcome.steps_taken, arm.outcome.restarts_started),
        (0, 0)
    );
    assert_eq!((arm.calls).len(), 0);
}

/// A LIVE slice ignores the re-grant entirely: a tiny re-grant slice cannot
/// shorten it and the run is call-for-call the run without one.
#[test]
fn setup_regrant_is_inert_on_a_live_slice() {
    let model = safe_model();
    let input = unit_box(2);
    let live = Instant::now() + std::time::Duration::from_secs(120);
    let base = run(
        &model,
        &input,
        &PgdConfig {
            deadline: Some(live),
            ..pgd(4, 5, ExactVjpPgdConfig::default())
        },
    );
    let with_regrant = run(
        &model,
        &input,
        &PgdConfig {
            deadline: Some(live),
            setup_regrant: Some(ny_propagate::PgdSetupRegrant {
                slice: std::time::Duration::from_millis(1),
                cap: live,
            }),
            ..pgd(4, 5, ExactVjpPgdConfig::default())
        },
    );
    assert_eq!(base.calls, with_regrant.calls);
    assert_eq!(
        (
            base.outcome.steps_taken,
            base.outcome.restarts_started,
            base.outcome.hit_deadline
        ),
        (
            with_regrant.outcome.steps_taken,
            with_regrant.outcome.restarts_started,
            with_regrant.outcome.hit_deadline
        )
    );
}
