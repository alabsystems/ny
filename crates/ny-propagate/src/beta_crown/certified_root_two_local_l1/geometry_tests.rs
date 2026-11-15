// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::*;
use crate::certify::work::{WorkBudget, WorkRequest};
use crate::layers::{LinearLayer, ReLULayer};
use crate::network::GraphNode;
use ndarray::{Array1, Array2};
use std::time::Duration;

fn ledger(work: u64) -> (WorkBudget, WorkLease) {
    let budget = WorkBudget::new(None, SCRATCH, 0, work, 1, Duration::ZERO).unwrap();
    let lease = budget
        .lease(WorkRequest {
            scratch_bytes: SCRATCH,
            ..WorkRequest::default()
        })
        .unwrap();
    (budget, lease)
}
fn future() -> Instant {
    Instant::now() + Duration::from_secs(600)
}
fn ranked(count: usize) -> [Item; 16] {
    let mut r = [(0, 0., 0.); 16];
    // One perturbed target-map score could promote row8. Geometry advice must
    // recover source cycles without reusing those target-map endpoint values.
    r[0] = (8, 2., 2.);
    for i in 0..8 {
        r[i + 1] = (i, 1., 2.);
    }
    if count == 10 {
        r[9] = (9, 1., 2.);
    }
    r
}
fn mask_for(r: &[Item; 16], count: usize, ids: &[usize]) -> u16 {
    ids.iter().fold(0, |mask, id| {
        mask | (1u16 << r[..count].iter().position(|x| x.0 == *id).unwrap())
    })
}
fn cycle_weights(groups: &[usize]) -> Vec<f32> {
    let e: usize = groups.iter().sum();
    let mut start = vec![0; groups.len() + 1];
    for i in 0..groups.len() {
        start[i + 1] = start[i] + groups[i];
    }
    let mut w = vec![0.; groups.len() * e];
    for base in [0, 3, 6] {
        for (row, mean, sign) in [
            (base, base, 1.),
            (base, base + 1, 1.),
            (base + 1, base + 1, 1.),
            (base + 1, base + 2, 1.),
            (base + 2, base + 2, 1.),
            (base + 2, base, -1.),
        ] {
            for j in start[mean]..start[mean + 1] {
                w[row * e + j] = sign * 0.5 / groups[mean] as f32;
            }
        }
    }
    if groups.len() == 10 {
        for j in start[9]..start[10] {
            w[9 * e + j] = 1. / groups[9] as f32;
        }
    }
    w
}

#[test]
fn hybrid_gram_keeps_source_cycles_and_preferred_on_small_and_dyadic_inputs() {
    for groups in [
        vec![1; 9],
        vec![1024, 1024, 512, 512, 256, 256, 256, 128, 128],
        vec![1; 10],
    ] {
        let count = groups.len();
        let e: usize = groups.iter().sum();
        let w = cycle_weights(&groups);
        let lo = vec![-1.; e];
        // A legitimate outward, slightly asymmetric parent box. In contrast
        // to target-row slack, its values are part of the source geometry.
        let hi = vec![f32::from_bits(1f32.to_bits() + 2); e];
        let source = GeometrySource {
            weights: &w,
            bias: None,
            lower: &lo,
            upper: &hi,
            bn: None,
            input_dim: e,
            width: count,
        };
        let r = ranked(count);
        let preferred_mask = (1u16 << (count - 3)) | (1u16 << (count - 2)) | (1u16 << (count - 1));
        let preferred = partition(&r, count, preferred_mask).unwrap();
        let (_, mut lease) = ledger(u64::MAX);
        let end = future();
        let mut obs = AdmissionObservation::default();
        let mut gram = GramWorkspace::default();
        source
            .gram(&r, count, &mut gram, &lease, end, &mut obs)
            .unwrap();
        let (short, used) = shortlist(&gram, &r, count, &preferred, &lease, end, &mut obs).unwrap();
        assert_eq!(used, 12);
        assert!(short[..used].iter().any(|x| x.mask == preferred_mask));
        for ids in [[0, 1, 2], [3, 4, 5], [6, 7, 8]] {
            let mask = mask_for(&r, count, &ids);
            let choice = partition(&r, count, mask).unwrap();
            assert!(gram_residual(&gram, &r, count, &choice).unwrap() < 1e-10);
            assert!(short[..used].iter().any(|x| x.mask == mask));
        }
        // For the unequal4096 dyadic means the block norm ratio is1/8.
        // Every noncycle triple has an exact positive normalized gap >=1/12;
        // use a wide tolerance because Gram is ordinary checked f64 advice.
        for a in 0..count - 2 {
            for b in a + 1..count - 1 {
                for c in b + 1..count {
                    let choice = partition(&r, count, (1 << a) | (1 << b) | (1 << c)).unwrap();
                    let ids = &choice.neurons[1][..3];
                    if ![[0, 1, 2], [3, 4, 5], [6, 7, 8]]
                        .iter()
                        .any(|x| x.as_slice() == ids)
                    {
                        assert!(gram_residual(&gram, &r, count, &choice).unwrap() > 0.01);
                    }
                }
            }
        }
        let selected = source
            .evaluate(&r, count, &preferred, &mut lease, end, &mut obs)
            .unwrap();
        assert_eq!(selected.active, [count - 3, 3]);
        assert!([[0, 1, 2], [3, 4, 5], [6, 7, 8]]
            .iter()
            .any(|ids| ids.as_slice() == &selected.neurons[1][..3]));
        let mut all = Vec::new();
        for g in 0..2 {
            all.extend_from_slice(&selected.neurons[g][..selected.active[g]]);
            assert!(selected.weights[g][..selected.active[g]]
                .iter()
                .all(|x| x.to_bits() == 1f32.to_bits()));
            assert_eq!(selected.multipliers[g].to_bits(), 1f32.to_bits());
            assert!(selected.neurons[g][selected.active[g]..]
                .iter()
                .all(|&x| x == 0));
            assert!(selected.weights[g][selected.active[g]..]
                .iter()
                .all(|x| x.to_bits() == 0));
        }
        all.sort_unstable();
        assert_eq!(all, (0..count).collect::<Vec<_>>());
    }
}

#[test]
fn hybrid_direct_bn_score_keeps_negative_scale_shift_and_common_radius() {
    // One exact frustrated cycle, with post-BN rows -2*z+1. Its nominal
    // marginal sum is9 and signed-box maximum7. Source uncertainty widens
    // both by3*(1/8+1/4)=9/8; the advice deficit stays2.
    let w = [0.5, 0.5, 0., 0., 0.5, 0.5, -0.5, 0., 0.5];
    let lo = [-1.; 3];
    let hi = [1.; 3];
    let scale = [-2.; 3];
    let shift = [1.; 3];
    let es = [0.125; 3];
    let eb = [0.25; 3];
    let zero = [0.; 3];
    let group = normalize_group(&[(0, 1., 2.), (1, 1., 2.), (2, 1., 2.)]).unwrap();
    let (_, lease) = ledger(u64::MAX);
    let end = future();
    let mut obs = AdmissionObservation::default();
    for errors in [false, true] {
        let source = GeometrySource {
            weights: &w,
            bias: None,
            lower: &lo,
            upper: &hi,
            bn: Some(BnView {
                scale: &scale,
                bias: &shift,
                scale_error: if errors { &es } else { &zero },
                bias_error: if errors { &eb } else { &zero },
            }),
            input_dim: 3,
            width: 3,
        };
        let got = source.triple_score(&group, &lease, end, &mut obs).unwrap();
        let radius = if errors { 9. / 8. } else { 0. };
        assert_eq!(got.radius, radius);
        assert_eq!(got.unary, 9. + radius);
        assert_eq!(got.signed_max, 7. + radius);
        assert_eq!(got.score, 2.);
    }
}

fn source_graph(
    bn: Option<BatchNormLayer>,
) -> (GraphNetwork, BoundedTensor, HashMap<String, BoundedTensor>) {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "pre",
        Layer::Linear(LinearLayer::new(Array2::eye(9), Some(Array1::zeros(9))).unwrap()),
    ));
    let mut map = HashMap::new();
    let b = BoundedTensor::new(
        Array1::from_elem(9, -1.).into_dyn(),
        Array1::from_elem(9, 1.).into_dyn(),
    )
    .unwrap();
    map.insert("pre".into(), b.clone());
    let parent = if let Some(bn) = bn {
        graph.add_node(GraphNode::new(
            "bn",
            Layer::BatchNorm(bn),
            vec!["pre".into()],
        ));
        map.insert("bn".into(), b.clone());
        "bn"
    } else {
        "pre"
    };
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec![parent.into()],
    ));
    graph.set_output("relu");
    (graph, b, map)
}
#[test]
fn hybrid_source_resolution_is_prepaid_and_refuses_malformed_raw_bn() {
    for malformed in 0..8 {
        let mut bn = BatchNormLayer::from_scale_bias(
            Array1::ones(9).into_dyn(),
            Array1::zeros(9).into_dyn(),
        )
        .unwrap();
        match malformed {
            1 => bn.scale = Array1::ones(8).into_dyn(),
            2 => bn.bias_err[ndarray::IxDyn(&[0])] = -0.25,
            3 => bn.scale[ndarray::IxDyn(&[0])] = f32::NAN,
            4 => bn.bias[ndarray::IxDyn(&[0])] = f32::INFINITY,
            5 => bn.scale_err[ndarray::IxDyn(&[0])] = f32::NAN,
            6 => bn.bias = Array1::zeros(18).slice_move(ndarray::s![..;2]).into_dyn(),
            7 => bn.bias_err = Array1::zeros(8).into_dyn(),
            _ => {}
        }
        let (graph, input, map) = source_graph(Some(bn));
        let (budget, mut lease) = ledger(SOURCE_WORK);
        let mut obs = AdmissionObservation::default();
        let result = geometry_quote(&graph, &input, &map, "relu", &mut lease, future(), &mut obs);
        assert_eq!(result.is_some(), malformed == 0);
        assert_eq!(budget.receipt().unwrap().work_reserved, SOURCE_WORK);
    }
    let (graph, input, mut map) = source_graph(None);
    let (budget, mut lease) = ledger(SOURCE_WORK - 1);
    let mut obs = AdmissionObservation::default();
    assert!(geometry_quote(
        &graph,
        &input,
        &map,
        "missing",
        &mut lease,
        future(),
        &mut obs
    )
    .is_none());
    assert_eq!(obs.reason, "geometry-source-work");
    assert_eq!(budget.receipt().unwrap().work_reserved, 0);
    map.remove("pre");
    let (_, mut lease) = ledger(SOURCE_WORK);
    assert!(geometry_quote(&graph, &input, &map, "relu", &mut lease, future(), &mut obs).is_none());
}

#[test]
fn hybrid_scan_preserves_exact_work_boundary_finite_deadline_and_storage_guards() {
    assert!(scratch_bytes().unwrap() <= SCRATCH);
    assert_eq!(SCRATCH, 8192);
    let mut w = cycle_weights(&[1; 9]);
    let lo = [-1.; 9];
    let hi = [1.; 9];
    let r = ranked(9);
    let preferred = partition(&r, 9, (1 << 6) | (1 << 7) | (1 << 8)).unwrap();
    let source = GeometrySource {
        weights: &w,
        bias: None,
        lower: &lo,
        upper: &hi,
        bn: None,
        input_dim: 9,
        width: 9,
    };
    let needed = source.work(9).unwrap();
    assert_eq!(needed, 20480 * 10 + (1 << 20));
    assert_eq!(source.work(10), Some(needed));
    assert_eq!(source.work(8), None);
    let end = future();
    for work in [needed - 1, needed] {
        let (budget, mut lease) = ledger(work);
        let mut obs = AdmissionObservation::default();
        let answer = source.evaluate(&r, 9, &preferred, &mut lease, end, &mut obs);
        assert_eq!(answer.is_some(), work == needed);
        assert_eq!(
            budget.receipt().unwrap().work_reserved,
            if work == needed { needed } else { 0 }
        );
        if work < needed {
            assert_eq!(obs.reason, "geometry-work");
        }
    }
    let (budget, mut lease) = ledger(needed);
    let mut obs = AdmissionObservation::default();
    assert!(source
        .evaluate(
            &r,
            9,
            &preferred,
            &mut lease,
            Instant::now().checked_sub(Duration::from_secs(1)).unwrap(),
            &mut obs
        )
        .is_none());
    assert_eq!(obs.reason, "deadline");
    assert_eq!(budget.receipt().unwrap().work_reserved, needed);
    let (budget, mut lease) = ledger(needed);
    lease.complete().unwrap();
    assert!(source
        .evaluate(&r, 9, &preferred, &mut lease, end, &mut obs)
        .is_none());
    assert_eq!(budget.receipt().unwrap().work_reserved, 0);
    let mut duplicate = r;
    duplicate[1].0 = duplicate[0].0;
    let (_, mut lease) = ledger(needed);
    assert!(source
        .evaluate(&duplicate, 9, &preferred, &mut lease, end, &mut obs)
        .is_none());
    // A real selected original coefficient is nonfinite, rather than an
    // unselected sentinel. Rejection follows the paid scan and cannot refund it.
    w[0] = f32::NAN;
    let invalid = GeometrySource {
        weights: &w,
        bias: None,
        lower: &lo,
        upper: &hi,
        bn: None,
        input_dim: 9,
        width: 9,
    };
    let (budget, mut lease) = ledger(needed);
    assert!(invalid
        .evaluate(&r, 9, &preferred, &mut lease, end, &mut obs)
        .is_none());
    assert_eq!(budget.receipt().unwrap().work_reserved, needed);
}
