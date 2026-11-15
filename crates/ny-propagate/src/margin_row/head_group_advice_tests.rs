use super::*;
use ny_core::head_phase_advice::HeadPhaseLpHint;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::margin_row::bab::build_pack;
use crate::margin_row::head_group_source::HeadGroupSourceGuard;
use crate::margin_row::net::TwinNet;
use crate::margin_row::root::RootGates;
use crate::margin_row::spec::{TwinOpSpec, TwinSpec};

fn deadline() -> Instant {
    Instant::now() + Duration::from_secs(5)
}

fn spec(bias: f64) -> TwinSpec {
    let mut spec = super::super::tests::classifier_spec(2, 0, 0.5, 0.0, 0.0);
    if let TwinOpSpec::Gemm {
        weight, bias: bs, ..
    } = &mut spec.ops[5]
    {
        weight[1] = -3.0 / 8.0;
        bs[0] = bias;
    } else {
        panic!("authored fixture head must be Gemm");
    }
    spec
}

struct Continuous {
    calls: AtomicUsize,
    negate: bool,
}
impl HeadPhaseAdvice for Continuous {
    fn propose(&self, request: &HeadPhaseLpRequest, deadline: Instant) -> Option<HeadPhaseLpHint> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        if !live(deadline) || !request.validate() {
            return None;
        }
        assert_eq!(request.phase_rows, vec![2, 5]);
        assert_eq!(request.bounds.len(), 3);
        let active: Vec<_> = request
            .phase_rows
            .iter()
            .enumerate()
            .map(|(i, &r)| request.rows[r].coefficients[1 + i] > 0.0)
            .collect();
        let p = match active.as_slice() {
            [false, false] => 0.0,
            [true, false] => 1.0,
            [false, true] => 3.0 / 8.0,
            [true, true] => 5.0 / 8.0,
            _ => return None,
        };
        // Other row multipliers are deliberately nonsensical. The seam reads
        // only the phase row roles from its OWN immutable export.
        let mut row_duals = vec![1e90; request.rows.len()];
        row_duals[2] = if self.negate { -p } else { p };
        row_duals[5] = 0.0;
        Some(HeadPhaseLpHint { row_duals })
    }
}

fn run_advice<P: HeadPhaseAdvice + 'static>(
    provider: Arc<P>,
    bias: f64,
) -> (Vec<f64>, Vec<f64>, f64, AdviceRefinement) {
    let _source = HeadGroupSourceGuard::enter(true);
    let net = TwinNet::compile(&spec(bias)).unwrap();
    let root = RootGates::build(&net, &[-1.0], &[1.0], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let pack = build_pack(&eng, None, &[]).unwrap();
    let mut source = from_owned_pack(&eng, &pack, deadline()).unwrap();
    bind_margin(&mut source, &eng, 0, 1, deadline()).unwrap();
    let mut cover = propose_and_check(&source, &[0, 1], deadline(), MAX_PROPOSAL_ROUNDS).unwrap();
    let before = cover.phase_lower.clone();
    let provider: Arc<dyn HeadPhaseAdvice> = provider;
    let report = refine_with_provider(
        &eng,
        &pack,
        &source,
        &[0, 1],
        &mut cover,
        &mut AdviceBudget::new(),
        deadline(),
        || Some(Arc::clone(&provider)),
    )
    .unwrap();
    (before, cover.phase_lower, cover.lower, report)
}

#[test]
fn continuous_phase_hints_are_owned_replayed_in_both_dual_conventions() {
    // Exact integer enumeration of the finite grid establishes its ceiling.
    let a = [0_i64, 8, 16, 32, 64, 128, 256];
    let b = [0_i64, 3, 6, 12, 24, 48, 96];
    let max11 = a
        .iter()
        .flat_map(|&a| {
            b.iter()
                .map(move |&b| 13 - (12 + b) - 2 * (20 - a - b).abs())
        })
        .max()
        .unwrap();
    let max01 = a
        .iter()
        .flat_map(|&a| {
            b.iter()
                .map(move |&b| 13 - (12 + b) - 2 * (a - 12 - b).abs())
        })
        .max()
        .unwrap();
    assert_eq!((max11, max01), (-4, -4));
    for negate in [false, true] {
        let provider = Arc::new(Continuous {
            calls: AtomicUsize::new(0),
            negate,
        });
        let (before, after, lower, report) = run_advice(Arc::clone(&provider), 13.0 / 64.0);
        assert!(before.iter().copied().fold(f64::INFINITY, f64::min) < 0.0);
        assert!(lower > 1.0 / 128.0 && lower <= 1.0 / 64.0);
        assert_eq!(after.len(), 4);
        assert_eq!(report.calls, 1);
        assert_eq!(provider.calls.load(Ordering::Relaxed), 1);
        assert_eq!(report.improved_phases, 2);
        assert_eq!(before[0].to_bits(), after[0].to_bits());
        assert!(after.iter().zip(&before).all(|(a, b)| a >= b));
    }
    // Actual exact network minimum is 1/64, at x=0. This test exercises
    // source exporter and replay, not a claim about an AY-chosen optimum or
    // gain beyond the separate continuous native Gemm proposal portfolio.
}

enum Fault {
    WrongLength,
    Nonfinite,
    Oversized,
    Late,
    Panic,
}
struct FaultProvider {
    fault: Fault,
    calls: AtomicUsize,
}
impl HeadPhaseAdvice for FaultProvider {
    fn propose(&self, request: &HeadPhaseLpRequest, deadline: Instant) -> Option<HeadPhaseLpHint> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let value = match self.fault {
            Fault::WrongLength => return Some(HeadPhaseLpHint { row_duals: vec![] }),
            Fault::Nonfinite => f64::NAN,
            Fault::Oversized => MAX_MAG * 2.0,
            Fault::Late => {
                std::thread::sleep(
                    deadline.saturating_duration_since(Instant::now()) + Duration::from_millis(2),
                );
                3.0 / 8.0
            }
            Fault::Panic => panic!("deliberate untrusted provider fault"),
        };
        Some(HeadPhaseLpHint {
            row_duals: vec![value; request.rows.len()],
        })
    }
}

#[test]
fn malformed_late_and_panicking_hints_preserve_all_old_masks() {
    for fault in [
        Fault::WrongLength,
        Fault::Nonfinite,
        Fault::Oversized,
        Fault::Late,
        Fault::Panic,
    ] {
        let provider = Arc::new(FaultProvider {
            fault,
            calls: AtomicUsize::new(0),
        });
        let (before, after, _, report) = run_advice(provider, 13.0 / 64.0);
        assert_eq!(
            before.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            after.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(report.candidates, 0);
        assert_eq!(report.improved_phases, 0);
        assert!(report.calls <= 2);
    }
}

#[test]
fn source_mutation_incomplete_cover_and_expiry_refuse_before_provider_work() {
    let _source = HeadGroupSourceGuard::enter(true);
    let net = TwinNet::compile(&spec(13.0 / 64.0)).unwrap();
    let root = RootGates::build(&net, &[-1.0], &[1.0], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let mut pack = build_pack(&eng, None, &[]).unwrap();
    let mut source = from_owned_pack(&eng, &pack, deadline()).unwrap();
    bind_margin(&mut source, &eng, 0, 1, deadline()).unwrap();
    let mut cover = propose_and_check(&source, &[0, 1], deadline(), MAX_PROPOSAL_ROUNDS).unwrap();
    let before = cover.phase_lower.clone();
    let provider: Arc<dyn HeadPhaseAdvice> = Arc::new(Continuous {
        calls: AtomicUsize::new(0),
        negate: false,
    });
    let mut budget = AdviceBudget::new();
    let expired = refine_with_provider(
        &eng,
        &pack,
        &source,
        &[0, 1],
        &mut cover,
        &mut budget,
        Instant::now(),
        || panic!("expired advice must not construct its provider"),
    )
    .unwrap();
    assert_eq!(expired.calls, 0);
    assert_eq!(cover.phase_lower, before);
    cover.phase_lower.pop();
    assert!(refine_with_provider(
        &eng,
        &pack,
        &source,
        &[0, 1],
        &mut cover,
        &mut budget,
        deadline(),
        || Some(Arc::clone(&provider))
    )
    .is_none());
    cover.phase_lower = before;
    pack.al.b[0] += 0.25;
    assert!(refine_with_provider(
        &eng,
        &pack,
        &source,
        &[0, 1],
        &mut cover,
        &mut budget,
        deadline(),
        || Some(Arc::clone(&provider))
    )
    .is_none());
    assert_eq!(budget.calls, 0);
}

#[test]
fn advice_call_limit_is_shared_across_multiple_covers_and_absence_preserves_bounds() {
    let _source = HeadGroupSourceGuard::enter(true);
    let net = TwinNet::compile(&spec(13.0 / 64.0)).unwrap();
    let root = RootGates::build(&net, &[-1.0], &[1.0], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let pack = build_pack(&eng, None, &[]).unwrap();
    let mut source = from_owned_pack(&eng, &pack, deadline()).unwrap();
    bind_margin(&mut source, &eng, 0, 1, deadline()).unwrap();
    let mut cover = propose_and_check(&source, &[0, 1], deadline(), MAX_PROPOSAL_ROUNDS).unwrap();
    let before = cover.phase_lower.clone();
    let mut absent_budget = AdviceBudget::new();
    let absent = refine_with_provider(
        &eng,
        &pack,
        &source,
        &[0, 1],
        &mut cover,
        &mut absent_budget,
        deadline(),
        || None,
    )
    .unwrap();
    assert_eq!(absent.calls, 0);
    assert_eq!(cover.phase_lower, before);
    let provider = Arc::new(FaultProvider {
        fault: Fault::WrongLength,
        calls: AtomicUsize::new(0),
    });
    let backend: Arc<dyn HeadPhaseAdvice> = provider.clone();
    let mut budget = AdviceBudget::new();
    let first = refine_with_provider(
        &eng,
        &pack,
        &source,
        &[0, 1],
        &mut cover,
        &mut budget,
        deadline(),
        || Some(Arc::clone(&backend)),
    )
    .unwrap();
    let second = refine_with_provider(
        &eng,
        &pack,
        &source,
        &[0, 1],
        &mut cover,
        &mut budget,
        deadline(),
        || Some(Arc::clone(&backend)),
    )
    .unwrap();
    assert_eq!((first.calls, second.calls), (2, 0));
    assert_eq!(provider.calls.load(Ordering::Relaxed), 2);
    assert_eq!(cover.phase_lower, before);
    assert!(budget.charged < TOTAL_PROPOSAL_TIME);
}

#[test]
fn arbitrary_phase_hints_cannot_close_a_negative_true_margin() {
    let (_, after, lower, _) = run_advice(
        Arc::new(Continuous {
            calls: AtomicUsize::new(0),
            negate: true,
        }),
        -1.0 / 64.0,
    );
    // At x=0, y=(0,1/2), so the exact margin is -1/64-3/16=-13/64.
    assert!(lower <= -13.0 / 64.0);
    assert_eq!(after.len(), 4);
}

#[test]
fn export_retains_repaired_bias_endpoints_explicit_phase_rows_and_source_box() {
    let _source = HeadGroupSourceGuard::enter(true);
    let net = TwinNet::compile(&spec(13.0 / 64.0)).unwrap();
    let root = RootGates::build(&net, &[-1.0], &[1.0], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let pack = build_pack(&eng, None, &[]).unwrap();
    let mut source = from_owned_pack(&eng, &pack, deadline()).unwrap();
    bind_margin(&mut source, &eng, 0, 1, deadline()).unwrap();
    let rest = remainder(&source, &[0, 1], deadline()).unwrap();
    for mask in 0..4 {
        let request = export_phase_lp(&source, &[0, 1], &rest, mask, deadline()).unwrap();
        assert!(request.validate());
        assert_eq!(request.bounds[0], [-1.0, 1.0]);
        assert_eq!(request.phase_rows, vec![2, 5]);
        for i in 0..2 {
            assert_eq!(
                request.rows[3 * i].lower.to_bits(),
                source.head[i].lower.b.lo.to_bits()
            );
            assert_eq!(
                request.rows[3 * i + 1].lower.to_bits(),
                (-source.head[i].upper.b.hi).to_bits()
            );
            let eta = if mask & (1 << i) != 0 { 1.0 } else { -1.0 };
            assert_eq!(request.rows[3 * i + 2].coefficients[1 + i], eta);
            // Bounds keep the original Y box; phase is an explicit row and
            // remains visible in the solver's returned row-dual vector.
            assert_eq!(
                request.bounds[1 + i],
                [source.head[i].y.lo, source.head[i].y.hi]
            );
        }
    }
}

fn cancellation_ridge_spec(bias: f64) -> TwinSpec {
    TwinSpec {
        n_in: 2,
        ops: vec![
            TwinOpSpec::Conv {
                input: 0,
                weight: vec![1.0, 0.0, 0.0, 1.0],
                bias: vec![2.0, 2.0],
                bias_err: vec![0.0, 0.0],
                weight_rel_err: 0.0,
                kernel: (2, 2, 1, 1),
                stride: (1, 1),
                pads: (0, 0, 0, 0),
                ishape: (2, 1, 1),
                oshape: (2, 1, 1),
            },
            TwinOpSpec::Relu { input: 1 },
            TwinOpSpec::Flatten { input: 2 },
            TwinOpSpec::Gemm {
                input: 3,
                weight: vec![1.0, 1.0, -1.0, 1.0, 0.0, 1.0],
                bias: vec![-63.0 / 16.0, 1.0 / 16.0, 0.0],
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
                shape: (3, 2),
            },
            TwinOpSpec::Relu { input: 4 },
            TwinOpSpec::Gemm {
                input: 5,
                weight: vec![1.0, 1.0, -3.0 / 4.0, 0.0, 0.0, 0.0],
                bias: vec![bias, 0.0],
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
                shape: (2, 3),
            },
        ],
    }
}

struct RidgeAnchor {
    calls: AtomicUsize,
    negate: bool,
}
impl HeadPhaseAdvice for RidgeAnchor {
    fn propose(&self, r: &HeadPhaseLpRequest, d: Instant) -> Option<HeadPhaseLpHint> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        if !r.validate() || !live(d) {
            return None;
        }
        assert_eq!(r.bounds.len(), 4);
        let mut row_duals = vec![0.0; r.rows.len()];
        for (i, &row) in r.phase_rows.iter().enumerate() {
            let p = if r.rows[row].coefficients[2 + i] > 0.0 {
                5.0 / 8.0
            } else {
                3.0 / 8.0
            };
            row_duals[row] = if self.negate { -p } else { p };
        }
        Some(HeadPhaseLpHint { row_duals })
    }
}

#[test]
fn one_phase_anchor_closes_all_masks_beyond_the_continuous_native_portfolio() {
    let _source = HeadGroupSourceGuard::enter(true);
    // Exact source proof, independent of advice: ReLU(y1)+ReLU(y2) >=
    // (3/8)(y1+y2). Since y1+y2=2*x2+1/8 and y3=x2+2,
    // m >= 97/64 + 3/64 - 3/2 = 1/16; equality at (0,-1/16).
    assert_eq!(97.0 / 64.0 + 3.0 / 64.0 - 3.0 / 2.0, 1.0 / 16.0);
    let net = TwinNet::compile(&cancellation_ridge_spec(97.0 / 64.0)).unwrap();
    let root =
        RootGates::build(&net, &[-1.0, -1.0], &[1.0, 1.0], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let pack = build_pack(&eng, None, &[]).unwrap();
    let mut source = from_owned_pack(&eng, &pack, deadline()).unwrap();
    bind_margin(&mut source, &eng, 0, 1, deadline()).unwrap();
    assert_eq!(
        source_groups(&source, 1, deadline()).unwrap(),
        vec![vec![0, 1]]
    );
    for negate in [false, true] {
        let mut cover =
            propose_and_check(&source, &[0, 1], deadline(), MAX_PROPOSAL_ROUNDS).unwrap();
        native::refine_complete_cover(&eng, &pack, &source, &[0, 1], &mut cover, deadline())
            .unwrap();
        assert!(
            cover.lower < 0.0,
            "the existing continuous coordinate portfolio misses the joint ridge"
        );
        let before = cover.phase_lower.clone();
        let provider = Arc::new(RidgeAnchor {
            calls: AtomicUsize::new(0),
            negate,
        });
        let backend: Arc<dyn HeadPhaseAdvice> = provider.clone();
        let report = refine_with_provider(
            &eng,
            &pack,
            &source,
            &[0, 1],
            &mut cover,
            &mut AdviceBudget::new(),
            deadline(),
            || Some(Arc::clone(&backend)),
        )
        .unwrap();
        assert_eq!(report.calls, 1);
        assert_eq!(provider.calls.load(Ordering::Relaxed), 1);
        assert!(cover.lower > 1.0 / 32.0 && cover.lower <= 1.0 / 16.0);
        assert_eq!(cover.phases, 4);
        assert!(cover.phase_lower.iter().zip(&before).all(|(a, b)| a >= b));
        assert!(cover.phase_lower.iter().all(|b| *b > 0.0));
        assert!(report.improved_phases > 0);
    }
}

#[test]
fn transferred_anchor_keeps_authored_full_q_residual_and_negative_source_guard() {
    let _source = HeadGroupSourceGuard::enter(true);
    let net = TwinNet::compile(&cancellation_ridge_spec(87.0 / 64.0)).unwrap();
    let root =
        RootGates::build(&net, &[-1.0, -1.0], &[1.0, 1.0], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let pack = build_pack(&eng, None, &[]).unwrap();
    let mut source = from_owned_pack(&eng, &pack, deadline()).unwrap();
    bind_margin(&mut source, &eng, 0, 1, deadline()).unwrap();
    // Valid larger authored-Q intervals deliberately spoil nominal reduced
    // costs. Rebuild the complete cover against this very same source.
    source.q[0] = I {
        lo: 1.0 - f64::EPSILON,
        hi: 1.0 + f64::EPSILON,
    };
    let mut cover = propose_and_check(&source, &[0, 1], deadline(), MAX_PROPOSAL_ROUNDS).unwrap();
    let backend: Arc<dyn HeadPhaseAdvice> = Arc::new(RidgeAnchor {
        calls: AtomicUsize::new(0),
        negate: true,
    });
    refine_with_provider(
        &eng,
        &pack,
        &source,
        &[0, 1],
        &mut cover,
        &mut AdviceBudget::new(),
        deadline(),
        || Some(Arc::clone(&backend)),
    )
    .unwrap();
    // Exact margin at the shared kink is 87/64-93/64=-3/32.
    assert!(cover.lower <= -3.0 / 32.0);
    assert_eq!(cover.phase_lower.len(), 4);
}
