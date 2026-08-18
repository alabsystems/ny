// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Domain-batching tests (#margin-row-gpu-batch), in the order the risk runs:
//!
//! * (a) GATE-OFF — the dark arm never observes a device and the lane's prefill
//!   is a no-op, so the batched path cannot move a bound while dark.
//! * (b) THE SLOT MAP — a deliberately PERMUTED payload must produce different
//!   published bounds (host, no device) and, on a real device, slot `d` must
//!   carry domain `d`'s own bound and not a sibling's.
//! * (c) FAIL-CLOSED PINS — one per refusal reason the batched path adds.
//! * (d) DEVICE EQUIVALENCE — each domain's batched bound equals the bound the
//!   one-at-a-time seam produces for that same domain, within the documented
//!   tolerance, and the counters the integrator reads actually move.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ny_core::GpuCrownLayer;

use super::super::tests::{chain_spec, residual_spec};
use super::*;
use crate::margin_row::engine::domain_gates;
use crate::margin_row::net::TwinNet;
use crate::margin_row::root::RootGates;
use crate::margin_row::rounding::RoundMode;
use crate::margin_row::spec::TwinSpec;

/// The batched fixtures use a WIDER box than the per-pass seam's tests: the
/// mapping pins need several genuinely UNSTABLE trunk neurons so that four
/// distinct piece-fixed domains exist and give distinguishable bounds. A
/// fixture whose domains all agree cannot certify a slot map.
const BOX_HALF_WIDTH: f64 = 0.5;

fn compile_wide(spec: &TwinSpec, mode: RoundMode) -> (TwinNet, RootGates) {
    let net = TwinNet::compile(spec).expect("fixture compiles");
    let lo = vec![-BOX_HALF_WIDTH; spec.n_in];
    let hi = vec![BOX_HALF_WIDTH; spec.n_in];
    let (root, _) =
        RootGates::build_retaining(&net, &lo, &hi, mode, None, None, &[]).expect("root gates");
    (net, root)
}

fn compile(spec: &TwinSpec) -> (TwinNet, RootGates) {
    compile_wide(spec, RoundMode::Outward)
}

/// Relative tolerance between a domain's BATCHED bound and its ONE-AT-A-TIME
/// bound.
///
/// They are not required to be bit-equal and must not be asserted so: the wide
/// pass folds `N*rows` stacked rows through one GEMM, so f32 accumulation order
/// differs from the single-domain fold. Both are independently certified
/// enclosures — the certified error lane carries exactly this rounding — so the
/// difference is a quality wobble, not a soundness one. The tolerance is far
/// tighter than the gap between two DIFFERENT domains' bounds on the fixture
/// (asserted separately), so it still falsifies a slot error.
const MATCH_REL_TOL: f64 = 1e-3;
const MATCH_ABS_TOL: f64 = 1e-6;

#[allow(dead_code)]
fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= MATCH_ABS_TOL + MATCH_REL_TOL * a.abs().max(b.abs())
}

/// Four DISTINCT single-neuron piece fixes on the fixture, or fewer if the
/// fixture has fewer unstable trunk neurons.
fn distinct_domains(root: &RootGates, want: usize) -> Vec<DomainGates> {
    let mut splits: Vec<(usize, usize, i8)> = Vec::new();
    for (li, rec) in root.layers.iter().enumerate() {
        for pos in 0..rec.unst.len() {
            for dir in [1i8, -1i8] {
                if splits.len() < want {
                    splits.push((li, pos, dir));
                }
            }
        }
    }
    splits
        .into_iter()
        .map(|s| domain_gates(root, &[s]))
        .collect()
}

// ---------------------------------------------------------------------------
// (a) Gate-off
// ---------------------------------------------------------------------------

/// Exact-`"1"` arming, and SUBORDINATE: the batch gate can never be the thing
/// that first sends a verdict-bearing bound to a device, because [`enabled`]
/// ANDs it with the per-pass seam's own gate, which is dark in tests.
#[test]
fn arming_is_exact_and_subordinate_and_default_dark() {
    for rejected in [
        None,
        Some(""),
        Some("0"),
        Some("true"),
        Some(" 1 "),
        Some("2"),
    ] {
        assert!(!armed_from_raw(rejected), "raw {rejected:?} must stay dark");
    }
    assert!(armed_from_raw(Some("1")));
    // The composed predicate is dark because `NY_MARGIN_ROW_GPU` is dark here.
    assert!(
        !enabled(),
        "batching must be dark while the seam gate is dark"
    );
    assert!(
        !crate::margin_row::gpu_seam::enabled(),
        "this test's premise: the per-pass seam gate is dark in the test process"
    );
}

/// The production entry must refuse before it builds a plan or observes a
/// device while dark — including for a batch that would otherwise be admissible.
#[test]
fn run_batch_is_a_no_op_while_dark() {
    let (net, root) = compile(&residual_spec());
    let eng = BackwardEngine::new(&net, &root);
    let doms = distinct_domains(&root, 4);
    let refs: Vec<&DomainGates> = doms.iter().collect();
    assert!(
        run_batch(&eng, &refs, None).is_none(),
        "the dark batch entry must publish nothing"
    );
}

// ---------------------------------------------------------------------------
// (c) Fail-closed pins
// ---------------------------------------------------------------------------

/// A unary CHAIN net has no batched coefficient egress; it belongs on the
/// per-pass seam. Refuse rather than silently dropping to N serial calls.
#[test]
fn prepare_refuses_a_unary_chain_plan() {
    let (net, root) = compile(&chain_spec());
    let eng = BackwardEngine::new(&net, &root);
    assert_eq!(
        prepare(&eng, &SeamCtx::default()).err(),
        Some(Refusal::Unmappable(
            "unary chain plan has no batched coefficient egress"
        ))
    );
}

/// Only the certified-outward mode is batched, exactly as for the per-pass seam.
#[test]
fn prepare_refuses_a_parity_root() {
    let (net, root) = compile_wide(&residual_spec(), RoundMode::Parity);
    let eng = BackwardEngine::new(&net, &root);
    assert_eq!(
        prepare(&eng, &SeamCtx::default()).err(),
        Some(Refusal::NotOutward)
    );
}

/// An empty chunk is a caller bug, not a batch.
#[test]
fn run_chunk_refuses_an_empty_domain_list() {
    let (net, root) = compile(&residual_spec());
    let eng = BackwardEngine::new(&net, &root);
    let bp = prepare(&eng, &SeamCtx::default()).expect("residual fixture prepares");
    assert_eq!(
        run_chunk(&bp, &eng, &[], None).err(),
        Some(BatchError::Lane(Refusal::Rows))
    );
}

/// A backend WITHOUT the batched coefficient egress declines (`Ok(None)`), and
/// the seam must read that as a refusal, never as an empty publication.
#[test]
fn a_backend_without_the_batched_egress_refuses() {
    struct NoEgress;
    impl GpuCrownBackward for NoEgress {
        fn crown_backward_gpu(
            &self,
            _layers: &[GpuCrownLayer],
            _spec: &[f32],
            _num_specs: usize,
            _input_lower: &[f32],
            _input_upper: &[f32],
        ) -> ny_core::Result<ny_core::GpuCrownResult> {
            unreachable!("the batched coefficient path never calls the bounds entry")
        }
        // `crown_backward_gpu_resnet_sound_batched_coeffs` is left at the
        // trait DEFAULT — `Ok(None)`, i.e. "this backend declines". That is
        // exactly the shape every non-upgraded backend has, and the seam must
        // read it as a refusal.
    }
    let (net, root) = compile(&residual_spec());
    let eng = BackwardEngine::new(&net, &root);
    let bp = prepare(&eng, &SeamCtx::default()).expect("residual fixture prepares");
    let doms = distinct_domains(&root, 2);
    let plans: Vec<Plan> = doms
        .iter()
        .map(|d| retarget_plan(&bp.plan, &bp.relus, &bp.node_abs, &eng, Some(d)))
        .collect::<Result<_, Refusal>>()
        .expect("re-gate");
    let segs: Vec<&[ny_core::GpuResnetSegment]> = plans
        .iter()
        .map(|p| match p {
            Plan::Segments(s) => s.as_slice(),
            Plan::Chain(_) => unreachable!("residual fixture"),
        })
        .collect();
    let refs: Vec<GpuResnetBatchedDomainRef<'_>> = segs
        .iter()
        .map(|&s| GpuResnetBatchedDomainRef {
            segments: s,
            input_lower: &bp.lo,
            input_upper: &bp.hi,
            beta_signed: &[],
            frontier_abs: &[],
            node_abs: &bp.node_abs,
        })
        .collect();
    assert_eq!(
        dispatch_batch_on(&NoEgress, &refs, &bp.gseed).err(),
        Some(BatchError::Lane(Refusal::NoCoeffEgress))
    );
}

#[derive(Clone, Copy)]
enum BackendFailure {
    Capacity,
    Deadline,
    Device,
}

struct FailingBackend {
    failure: BackendFailure,
    calls: AtomicUsize,
}

impl FailingBackend {
    fn new(failure: BackendFailure) -> Self {
        Self {
            failure,
            calls: AtomicUsize::new(0),
        }
    }
}

impl GpuCrownBackward for FailingBackend {
    fn crown_backward_gpu(
        &self,
        _layers: &[GpuCrownLayer],
        _spec: &[f32],
        _num_specs: usize,
        _input_lower: &[f32],
        _input_upper: &[f32],
    ) -> ny_core::Result<ny_core::GpuCrownResult> {
        unreachable!("the coefficient path never calls the bounds entry")
    }

    fn crown_backward_gpu_resnet_sound_batched_coeffs(
        &self,
        _domains: &[GpuResnetBatchedDomainRef<'_>],
        _seed: &GpuCrownSeed,
    ) -> ny_core::Result<Option<Vec<CertifiedCoeffs>>> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Err(match self.failure {
            BackendFailure::Capacity => ny_core::NyError::GpuBatchCapacityExceeded {
                requested: 16,
                capacity: 8,
                unit: "domains",
                site: "batch-test",
            },
            BackendFailure::Deadline => {
                ny_core::NyError::DeadlineExceeded("batch-test deadline".into())
            }
            BackendFailure::Device => ny_core::NyError::InternalError("batch-test fault".into()),
        })
    }
}

/// Exercise the actual trait-error classifier through the width ladder. This
/// prevents a future catch-all `.map_err(Device)` (or retry-on-Device) from
/// silently laundering a late/faulted accepted request.
#[test]
fn backend_errors_retry_only_the_typed_capacity_variant() {
    let seed = GpuCrownSeed {
        lower_a: vec![1.0].into(),
        upper_a: vec![1.0].into(),
        lower_b: vec![0.0].into(),
        upper_b: vec![0.0].into(),
        num_specs: 1,
        current_dim: 1,
    };

    for (failure, expected, calls) in [
        (BackendFailure::Capacity, BatchError::Capacity, 4usize),
        (BackendFailure::Deadline, BatchError::DeadlineExpired, 1),
        (BackendFailure::Device, BatchError::Lane(Refusal::Device), 1),
    ] {
        let backend = FailingBackend::new(failure);
        let result = run_width_ladder_with_clock(
            16,
            None,
            |_| dispatch_batch_on(&backend, &[], &seed),
            Instant::now,
        );
        assert_eq!(result.err(), Some(expected));
        assert_eq!(
            backend.calls.load(Ordering::Relaxed),
            calls,
            "{expected:?} had the wrong retry count"
        );
    }
}

/// Only the typed PRE-DISPATCH capacity refusal may enter the width ladder.
/// Generic accepted-request failures, including a backend deadline, are hard
/// failures and must be observed exactly once.
#[test]
fn width_ladder_retries_only_typed_capacity() {
    let mut widths = Vec::new();
    let got = run_width_ladder_with_clock(
        16,
        None,
        |width| {
            widths.push(width);
            if width > 4 {
                Err(BatchError::Capacity)
            } else {
                Ok(width)
            }
        },
        Instant::now,
    )
    .expect("typed capacity narrows to an admissible width");
    assert_eq!(got, 4);
    assert_eq!(widths, [16, 8, 4]);

    for terminal in [
        BatchError::DeadlineExpired,
        BatchError::Lane(Refusal::Device),
        BatchError::Lane(Refusal::NoCoeffEgress),
        BatchError::Lane(Refusal::Payload),
    ] {
        let mut calls = 0usize;
        assert_eq!(
            run_width_ladder_with_clock(
                16,
                None,
                |_| {
                    calls += 1;
                    Err::<(), _>(terminal)
                },
                Instant::now,
            ),
            Err(terminal)
        );
        assert_eq!(calls, 1, "{terminal:?} must never be retried");
    }
}

/// Deadline checks bracket every attempt. This deterministic clock makes a
/// capacity refusal consume the remaining budget and proves no narrower call
/// is issued afterwards; it also proves a late successful payload is dropped.
#[test]
fn deadline_stops_before_retry_and_before_late_publication() {
    let base = Instant::now();
    let deadline = base + Duration::from_secs(1);

    let mut calls = 0usize;
    let mut times = [base, deadline].into_iter();
    let got = run_width_ladder_with_clock(
        16,
        Some(deadline),
        |_| {
            calls += 1;
            Err::<(), _>(BatchError::Capacity)
        },
        || times.next().expect("one precheck per prospective attempt"),
    );
    assert_eq!(got, Err(BatchError::DeadlineExpired));
    assert_eq!(calls, 1, "expired retry must not reach the backend");

    let mut calls = 0usize;
    let mut times = [base, deadline].into_iter();
    let got = run_width_ladder_with_clock(
        16,
        Some(deadline),
        |_| {
            calls += 1;
            Ok(7usize)
        },
        || times.next().expect("pre/post publication deadline checks"),
    );
    assert_eq!(got, Err(BatchError::DeadlineExpired));
    assert_eq!(calls, 1, "a late success is dropped, never reissued");
}

/// A terminal error can arrive only after an earlier chunk completed. Prove
/// that neither a generic device failure nor a backend deadline restarts the
/// wave at a narrower width and reissues chunk one.
#[test]
fn late_terminal_chunk_error_never_reissues_an_earlier_chunk() {
    let inputs = [0usize, 1, 2, 3, 4, 5];
    for terminal in [
        BatchError::Lane(Refusal::Device),
        BatchError::DeadlineExpired,
    ] {
        let mut chunks_seen = Vec::new();
        let result = run_width_ladder_with_clock(
            4,
            None,
            |width| {
                run_chunks_with(&inputs, width, None, |chunk| {
                    chunks_seen.push(chunk.to_vec());
                    if chunks_seen.len() == 1 {
                        Ok(chunk.to_vec())
                    } else {
                        Err(terminal)
                    }
                })
            },
            Instant::now,
        );
        assert_eq!(result, Err(terminal));
        assert_eq!(
            chunks_seen,
            [vec![0, 1, 2, 3], vec![4, 5]],
            "{terminal:?} must stop after chunk two without narrowing and reissuing chunk one"
        );
    }
}

// ---------------------------------------------------------------------------
// (b) The slot map — host-only falsifiers
// ---------------------------------------------------------------------------

/// A synthetic, ADMISSIBLE payload: coefficients small, biases far outside any
/// realizable value and errors generous, so it clears the certified-error floor
/// and the realization probe and the test can isolate the MAPPING.
fn synthetic(rows: usize, dim: usize, tag: f32) -> CertifiedCoeffs {
    CertifiedCoeffs {
        lower_a: vec![1e-4 * tag; rows * dim],
        upper_a: vec![1e-4 * tag; rows * dim],
        lower_a_err: vec![1e-2; rows * dim],
        upper_a_err: vec![1e-2; rows * dim],
        lower_b: vec![-100.0 - tag; rows],
        upper_b: vec![100.0 + tag; rows],
        lower_b_err: vec![1.0; rows],
        upper_b_err: vec![1.0; rows],
        num_specs: rows,
        dim,
    }
}

/// THE PERMUTATION PIN (host). Slot `d`'s published pass must be a function of
/// payload `d`. Swapping the two payloads must therefore swap the two published
/// bounds — if it did not, the mapping would be inert and a real device-side
/// permutation would be undetectable.
#[test]
fn batched_slot_permutation_is_detectable() {
    let (net, root) = compile(&residual_spec());
    let eng = BackwardEngine::new(&net, &root);
    let bp = prepare(&eng, &SeamCtx::default()).expect("residual fixture prepares");
    let doms = distinct_domains(&root, 2);
    assert_eq!(doms.len(), 2, "fixture must offer two piece-fixed domains");
    let refs: Vec<&DomainGates> = doms.iter().collect();

    let (a, b) = (
        synthetic(bp.rows, net.n_in, 1.0),
        synthetic(bp.rows, net.n_in, 7.0),
    );
    let straight = finish_chunk(&bp, &eng, &refs, &[a.clone(), b.clone()]).expect("straight");
    let permuted = finish_chunk(&bp, &eng, &refs, &[b, a]).expect("permuted");

    let s0 = eng.concretize_lower(&straight[0].0);
    let s1 = eng.concretize_lower(&straight[1].0);
    let p0 = eng.concretize_lower(&permuted[0].0);
    let p1 = eng.concretize_lower(&permuted[1].0);
    assert_ne!(
        s0, p0,
        "slot 0 did not follow its payload — the mapping is inert, so a device-side \
         permutation would publish one domain's bound for another and no test would see it"
    );
    // And the swap is exactly a swap: slot 0 under the permutation carries what
    // slot 1 carried before it.
    assert_eq!(p0, s1, "permuted slot 0 must carry the payload slot 1 had");
    assert_eq!(p1, s0, "permuted slot 1 must carry the payload slot 0 had");
}

/// A payload of the wrong length REFUSES. Truncating to fit would shift every
/// later domain's association by one — the killer defect in its quietest form.
#[test]
fn a_wrong_length_payload_refuses_instead_of_re_associating() {
    let (net, root) = compile(&residual_spec());
    let eng = BackwardEngine::new(&net, &root);
    let bp = prepare(&eng, &SeamCtx::default()).expect("residual fixture prepares");
    let doms = distinct_domains(&root, 3);
    let refs: Vec<&DomainGates> = doms.iter().collect();
    let one = synthetic(bp.rows, net.n_in, 1.0);
    assert_eq!(
        finish_chunk(&bp, &eng, &refs, &[one.clone()]).err(),
        Some(Refusal::Payload),
        "a short payload must refuse"
    );
    let many = vec![one.clone(), one.clone(), one.clone(), one];
    assert_eq!(
        finish_chunk(&bp, &eng, &refs, &many).err(),
        Some(Refusal::Payload),
        "a long payload must refuse"
    );
}

// ---------------------------------------------------------------------------
// The batch shape — what varies per domain
// ---------------------------------------------------------------------------

/// The claim the whole batch rests on: re-gating for a domain changes ONLY the
/// `Activation` values. Every weight-bearing layer must remain the SAME `Arc`
/// allocation, so the device's homogeneity gate passes on pointer identity and
/// "the domains share weights" is structural rather than coincidental.
#[test]
fn retargeting_moves_only_the_gates_and_shares_the_weight_arcs() {
    let (net, root) = compile(&residual_spec());
    let eng = BackwardEngine::new(&net, &root);
    let bp = prepare(&eng, &SeamCtx::default()).expect("residual fixture prepares");
    let doms = distinct_domains(&root, 2);
    assert_eq!(doms.len(), 2);
    let a =
        retarget_plan(&bp.plan, &bp.relus, &bp.node_abs, &eng, Some(&doms[0])).expect("re-gate a");
    let b =
        retarget_plan(&bp.plan, &bp.relus, &bp.node_abs, &eng, Some(&doms[1])).expect("re-gate b");

    let flat = |p: &Plan| -> Vec<GpuCrownLayer> {
        let Plan::Segments(segs) = p else {
            unreachable!("residual fixture")
        };
        let mut out = Vec::new();
        for s in segs {
            match s {
                ny_core::GpuResnetSegment::Chain(l) | ny_core::GpuResnetSegment::Residual(l) => {
                    out.extend(l.iter().cloned());
                }
                ny_core::GpuResnetSegment::ResidualProj(f, p) => {
                    out.extend(f.iter().cloned());
                    out.extend(p.iter().cloned());
                }
            }
        }
        out
    };
    let (base, fa, fb) = (flat(&bp.plan), flat(&a), flat(&b));
    assert_eq!(base.len(), fa.len());
    assert_eq!(base.len(), fb.len());
    let mut activations = 0usize;
    let mut moved = 0usize;
    for ((l0, la), lb) in base.iter().zip(&fa).zip(&fb) {
        match (l0, la, lb) {
            (
                GpuCrownLayer::Activation {
                    lower_slope: s0,
                    upper_slope: u0,
                    upper_intercept: c0,
                    num_neurons: n0,
                    ..
                },
                GpuCrownLayer::Activation {
                    lower_slope: sa,
                    upper_slope: ua,
                    upper_intercept: ca,
                    num_neurons: na,
                    ..
                },
                GpuCrownLayer::Activation {
                    lower_slope: sb,
                    upper_slope: ub,
                    upper_intercept: cb,
                    num_neurons: nb,
                    ..
                },
            ) => {
                activations += 1;
                assert_eq!((n0, n0), (na, nb), "re-gate must not change the width");
                if (sa, ua, ca) != (sb, ub, cb) || (sa, ua, ca) != (s0, u0, c0) {
                    moved += 1;
                }
            }
            (
                GpuCrownLayer::Conv2d {
                    weight_col: w0,
                    bias_expanded: b0,
                    cert_err: e0,
                    ..
                },
                GpuCrownLayer::Conv2d {
                    weight_col: wa,
                    bias_expanded: ba,
                    cert_err: ea,
                    ..
                },
                GpuCrownLayer::Conv2d {
                    weight_col: wb,
                    bias_expanded: bb,
                    cert_err: eb,
                    ..
                },
            ) => {
                assert!(
                    Arc::ptr_eq(w0, wa) && Arc::ptr_eq(w0, wb),
                    "conv weights must be the SAME allocation across domains"
                );
                assert_eq!((e0, e0), (ea, eb), "the BN-fold charge must not move");
                match (b0, ba, bb) {
                    (Some(x), Some(y), Some(z)) => {
                        assert!(Arc::ptr_eq(x, y) && Arc::ptr_eq(x, z));
                    }
                    (None, None, None) => {}
                    _ => panic!("conv bias presence moved across domains"),
                }
            }
            (
                GpuCrownLayer::Linear {
                    weight: w0,
                    cert_err: e0,
                    ..
                },
                GpuCrownLayer::Linear {
                    weight: wa,
                    cert_err: ea,
                    ..
                },
                GpuCrownLayer::Linear {
                    weight: wb,
                    cert_err: eb,
                    ..
                },
            ) => {
                assert!(
                    Arc::ptr_eq(w0, wa) && Arc::ptr_eq(w0, wb),
                    "gemm weights must be the SAME allocation across domains"
                );
                assert_eq!((e0, e0), (ea, eb), "the BN-fold charge must not move");
            }
            _ => panic!("re-gate changed a layer VARIANT"),
        }
    }
    assert!(activations > 0, "the fixture must carry ReLUs to re-gate");
    assert!(
        moved > 0,
        "the two domains must actually differ in their gates, or this pin proves nothing"
    );
}

// ---------------------------------------------------------------------------
// (d) Device equivalence
// ---------------------------------------------------------------------------

/// REAL DEVICE, 4 domains with DIFFERENT gate overrides.
///
/// Asserts, in order:
/// 0. the counters the integrator reads (`gpu_batch_ok`, `gpu_batch_domains`)
///    actually MOVE — a lane that never fires looks exactly like one that works;
/// 1. the 4 domains give DISTINGUISHABLE bounds, or nothing below proves
///    anything;
/// 2. each domain's BATCHED bound equals the bound the ONE-AT-A-TIME seam
///    publishes for that SAME domain, within [`MATCH_REL_TOL`];
/// 3. a deliberately PERMUTED slot mapping FAILS assertion 2 — the killer
///    defect is observable by this very test.
///
/// NOTE on the input box: this lane's domains all share the ROOT box (a piece
/// fix restricts the region through GATES, never through the box), so the
/// per-domain `input_lower`/`input_upper` channel carries the same values for
/// every slot here. That is a property of the lane, not a gap in the test — and
/// the coefficient egress ignores the box entirely by contract, because the
/// LANE concretizes.
#[cfg(feature = "gpu-tests")]
#[test]
fn gpu_batched_domains_match_the_one_at_a_time_seam_on_device() {
    use crate::margin_row::gpu_seam::run_pass_armed;

    let device = ny_gpu::WgpuDevice::new_for_verdict(ny_gpu::WgpuVerdictRequest::new())
        .expect("gpu-tests requires a WGPU device passing all five authority rungs");
    crate::sound_gpu_gate::set_sound_gpu_crown_required(true);
    let shared: Arc<dyn ny_core::GemmEngine> = Arc::new(device);
    crate::sound_gpu_gate::set_sound_gpu_crown_factory(move || Some(shared.clone()));
    assert!(
        crate::sound_gpu_gate::prewarm_sound_gpu_crown(),
        "the explicitly requested device must advertise sound GPU CROWN"
    );

    let (net, root) = compile(&residual_spec());
    let eng = BackwardEngine::new(&net, &root);
    let doms = distinct_domains(&root, 4);
    assert!(
        doms.len() >= 4,
        "the fixture must offer 4 distinct piece-fixed domains, got {}",
        doms.len()
    );
    let refs: Vec<&DomainGates> = doms.iter().collect();

    // ONE-AT-A-TIME reference, through the established per-pass seam.
    let bp = prepare(&eng, &SeamCtx::default()).expect("residual fixture prepares");
    // Review defect D1: the UPPER lane had no device validation anywhere in
    // this change, yet `convert_pair` publishes BOTH and `pack_from_rows` uses
    // `au` for `uy0`/`au_dots`, which feed verdicts. Capture both lanes.
    let serial: Vec<(Vec<f64>, Vec<f64>)> = refs
        .iter()
        .map(|d| {
            let pair = run_pass_armed(&eng, &bp.seed, Some(*d), &SeamCtx::default(), None)
                .expect("the per-pass seam must reach the prewarmed device");
            let lower = eng.concretize_lower(&pair.0.expect("lower lane"));
            let upper = eng.concretize_upper(&pair.1.expect("upper lane"));
            (lower, upper)
        })
        .collect();

    // (1) the domains must be distinguishable.
    let mut distinguishable = false;
    for i in 1..serial.len() {
        if serial[i]
            .0
            .iter()
            .zip(&serial[0].0)
            .chain(serial[i].1.iter().zip(&serial[0].1))
            .any(|(a, b)| !close(*a, *b))
        {
            distinguishable = true;
        }
    }
    assert!(
        distinguishable,
        "every domain produced the same bound — a slot error would be invisible, so this \
         fixture cannot certify the mapping"
    );

    // (0) the counters.
    crate::margin_row::prof::force_active_for_test(true);
    let ok_before = crate::margin_row::prof::counter(Counter::GpuBatchOk);
    let dom_before = crate::margin_row::prof::counter(Counter::GpuBatchDomains);
    let attempts_before = crate::margin_row::prof::counter(Counter::GpuBatchAttempts);
    let trips_before = crate::margin_row::prof::counter(Counter::GpuBatchGuardTrip);

    let batched = run_batch_armed_recorded(&eng, &refs, None)
        .expect("the batched dispatch must reach the prewarmed device on a residual net");
    assert_eq!(batched.len(), refs.len());
    assert!(
        crate::margin_row::prof::counter(Counter::GpuBatchOk) > ok_before,
        "gpu_batch_ok must increment (once per wide call; more than one only if the \
         device declined the full width and the lane halved) — this is the counter that \
         distinguishes a lane that fired from one that silently never ran"
    );
    assert_eq!(
        crate::margin_row::prof::counter(Counter::GpuBatchDomains),
        dom_before + refs.len() as u64,
        "gpu_batch_domains must count every domain the wide call served"
    );
    assert!(
        crate::margin_row::prof::counter(Counter::GpuBatchAttempts) > attempts_before,
        "gpu_batch_attempts must be this lane's source-specific denominator"
    );

    // (2) per-domain equality with the one-at-a-time seam.
    for (d, (pair, want)) in batched.iter().zip(&serial).enumerate() {
        let got = eng.concretize_lower(&pair.0);
        assert_eq!(got.len(), want.0.len(), "domain {d}: row count moved");
        for (r, (g, w)) in got.iter().zip(&want.0).enumerate() {
            assert!(
                close(*g, *w),
                "domain {d} row {r}: batched lower {g} != one-at-a-time {w}"
            );
        }
        // D1: the upper lane carries verdict weight through `au`.
        let got_u = eng.concretize_upper(&pair.1);
        assert_eq!(
            got_u.len(),
            want.1.len(),
            "domain {d}: upper row count moved"
        );
        for (r, (g, w)) in got_u.iter().zip(&want.1).enumerate() {
            assert!(
                close(*g, *w),
                "domain {d} row {r}: batched upper {g} != one-at-a-time {w}"
            );
        }
    }

    // (3) THE PERMUTATION FALSIFIER: with the domains handed over in a swapped
    // order, slot 0 must carry domain 1's bound. If the device (or this module)
    // ever re-associated slots, assertion 2 above would have passed anyway —
    // this is what makes that impossible.
    let swapped: Vec<&DomainGates> = vec![refs[1], refs[0], refs[3], refs[2]];
    let permuted = run_batch_armed_recorded(&eng, &swapped, None)
        .expect("the batched dispatch must reach the device for the permuted order");
    for (slot, src) in [(0usize, 1usize), (1, 0), (2, 3), (3, 2)] {
        let got = eng.concretize_lower(&permuted[slot].0);
        for (r, (g, w)) in got.iter().zip(&serial[src].0).enumerate() {
            assert!(
                close(*g, *w),
                "permuted slot {slot} row {r}: got {g}, but it must carry domain {src}'s \
                 bound {w} — the output slot follows the INPUT ORDER"
            );
        }
        let got_u = eng.concretize_upper(&permuted[slot].1);
        for (r, (g, w)) in got_u.iter().zip(&serial[src].1).enumerate() {
            assert!(
                close(*g, *w),
                "permuted slot {slot} upper row {r}: got {g}, but it must carry domain \
                 {src}'s bound {w} — the upper output slot follows the INPUT ORDER"
            );
        }
    }

    assert_eq!(
        crate::margin_row::prof::counter(Counter::GpuBatchGuardTrip),
        trips_before,
        "a healthy device must never trip a soundness guard in the batched lane"
    );
    crate::margin_row::prof::force_active_for_test(false);
}
