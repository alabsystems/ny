// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Exact, small descriptor interpreter. This is not a production accelerator
//! or a scalar oracle: it folds the supplied Linear/ReLU rows and carrier.
use super::*;
use crate::layers::{LinearLayer, ReLULayer};
use crate::network::GraphNode;
use ndarray::{arr1, arr2};
use num_rational::BigRational as Q;
use num_traits::ToPrimitive;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

fn rat(x: f32) -> Q {
    Q::from_float(f64::from(x)).unwrap()
}
fn fail() -> NyError {
    NyError::UnsupportedOp("exact toy supports only tiny exact Linear/ReLU chains".into())
}
fn rounded(x: &Q, lower: bool) -> Result<f32> {
    let mut v = x.to_f32().ok_or_else(fail)?;
    for _ in 0..3 {
        if !v.is_finite() {
            return Err(fail());
        }
        if (lower && rat(v) <= *x) || (!lower && rat(v) >= *x) {
            return Ok(v);
        }
        v = if lower {
            ny_tensor::next_down_f32(v)
        } else {
            next_up_f32(v)
        };
    }
    Err(fail())
}
fn guard(end: Instant) -> Result<()> {
    if Instant::now() >= end {
        Err(NyError::DeadlineExceeded("exact toy deadline".into()))
    } else {
        Ok(())
    }
}

#[derive(Default)]
pub(crate) struct ExactToyBackend {
    pub(crate) support_calls: AtomicUsize,
    pub(crate) cut_calls: AtomicUsize,
    /// A failed second fold must not expose the first successful proposal.
    pub(crate) refuse_cut: AtomicUsize,
    /// Typed backend refusal under an otherwise live caller clock.
    refuse_support_deadline: AtomicUsize,
}
impl GemmEngine for ExactToyBackend {
    fn gemm_f32(&self, _: usize, _: usize, _: usize, _: &[f32], _: &[f32]) -> Result<Vec<f32>> {
        panic!("ordinary GEMM permission must not be used")
    }
    fn as_gpu_crown_backward(&self) -> Option<&dyn GpuCrownBackward> {
        Some(self)
    }
}
fn chain(s: &[GpuResnetSegment]) -> Result<&[GpuCrownLayer]> {
    match s {
        [GpuResnetSegment::Chain(v)] if v.len() <= 8 => Ok(v),
        _ => Err(fail()),
    }
}
fn exact_objective(seed: &GpuCrownSeed, rows: usize) -> Result<()> {
    if seed.num_specs != rows
        || seed.current_dim == 0
        || seed.current_dim > 16
        || seed.lower_a.len() != rows * seed.current_dim
        || seed.upper_a.len() != seed.lower_a.len()
        || seed.lower_b.len() != rows
        || seed.upper_b.len() != rows
        || seed
            .lower_a
            .iter()
            .chain(seed.lower_b.iter())
            .any(|v| !v.is_finite())
        || seed
            .lower_a
            .iter()
            .zip(seed.upper_a.iter())
            .any(|(a, b)| a.to_bits() != b.to_bits())
        || seed
            .lower_b
            .iter()
            .zip(seed.upper_b.iter())
            .any(|(a, b)| a.to_bits() != b.to_bits())
    {
        return Err(fail());
    }
    Ok(())
}
fn fold(
    layers: &[GpuCrownLayer],
    mut a: Vec<Q>,
    mut b: Q,
    lower: bool,
    carrier: Option<&ResidentLowerCutCarrier>,
    end: Instant,
) -> Result<(Vec<Q>, Q)> {
    let mut activation = 0;
    let mut applied = false;
    for layer in layers {
        guard(end)?;
        match layer {
            GpuCrownLayer::Linear {
                weight,
                bias,
                out_features,
                in_features,
                cert_err,
            } => {
                if !cert_err.is_exact()
                    || a.len() != *out_features
                    || *in_features > 1024
                    || *out_features > 32
                    || weight.len() > 32 * 1024
                    || weight.len() != out_features * in_features
                    || bias.as_ref().is_some_and(|v| v.len() != *out_features)
                    || weight
                        .iter()
                        .chain(bias.iter().flat_map(|v| v.iter()))
                        .any(|x| !x.is_finite())
                {
                    return Err(fail());
                }
                if let Some(bias) = bias {
                    for i in 0..a.len() {
                        b += &a[i] * rat(bias[i]);
                    }
                }
                let mut next = vec![rat(0.); *in_features];
                for i in 0..*out_features {
                    for j in 0..*in_features {
                        if j % 32 == 0 {
                            guard(end)?;
                        }
                        next[j] += &a[i] * rat(weight[i * in_features + j]);
                    }
                }
                a = next;
            }
            GpuCrownLayer::Activation { .. } | GpuCrownLayer::ActivationReluDualAlpha { .. } => {
                let (lower_slope, upper_slope, third, upper_intercept, num_neurons, dual) =
                    match layer {
                        GpuCrownLayer::Activation {
                            lower_slope,
                            upper_slope,
                            lower_intercept,
                            upper_intercept,
                            num_neurons,
                        } => (
                            lower_slope,
                            upper_slope,
                            lower_intercept,
                            upper_intercept,
                            num_neurons,
                            false,
                        ),
                        GpuCrownLayer::ActivationReluDualAlpha {
                            lower_pos_slope,
                            cross_slope,
                            upper_neg_slope,
                            cross_intercept,
                            num_neurons,
                        } => (
                            lower_pos_slope,
                            cross_slope,
                            upper_neg_slope,
                            cross_intercept,
                            num_neurons,
                            true,
                        ),
                        _ => unreachable!(),
                    };
                if a.len() != *num_neurons
                    || [lower_slope, upper_slope, third, upper_intercept]
                        .iter()
                        .any(|v| v.len() != a.len() || v.iter().any(|x| !x.is_finite()))
                {
                    return Err(fail());
                }
                let hit = carrier.filter(|c| c.target_activation() == activation);
                if let Some(c) = hit {
                    if !lower {
                        return Err(fail());
                    }
                    let r = &c.rows()[0];
                    // Refuse unsupported enclosures, rather than ignoring them.
                    if r.pre()
                        .iter()
                        .chain(r.post())
                        .chain(std::iter::once(&r.bias()))
                        .any(|v| v.source_abs_error() != 0.)
                    {
                        return Err(fail());
                    }
                    for (p, &j) in c.ordered_neurons().iter().enumerate() {
                        a[j] += rat(r.post()[p].value());
                    }
                }
                for i in 0..a.len() {
                    let use_lower = (a[i] >= rat(0.)) == lower;
                    let (s, t) = if use_lower {
                        // Dual's third buffer is the upper-path MINORANT,
                        // never an intercept. Interpret the supplied source
                        // descriptor independently of production extraction.
                        (
                            if dual && !lower {
                                third[i]
                            } else {
                                lower_slope[i]
                            },
                            if dual { 0. } else { third[i] },
                        )
                    } else {
                        (upper_slope[i], upper_intercept[i])
                    };
                    b += &a[i] * rat(t);
                    a[i] *= rat(s);
                }
                if let Some(c) = hit {
                    let r = &c.rows()[0];
                    for (p, &j) in c.ordered_neurons().iter().enumerate() {
                        a[j] += rat(r.pre()[p].value());
                    }
                    b += rat(r.bias().value());
                    applied = true;
                }
                activation += 1;
            }
            _ => return Err(fail()),
        }
    }
    if carrier.is_some() && !applied {
        return Err(fail());
    }
    guard(end)?;
    Ok((a, b))
}
fn concrete(a: &[Q], mut b: Q, l: &[f32], u: &[f32], lower: bool, end: Instant) -> Result<f32> {
    guard(end)?;
    if a.len() != l.len()
        || l.len() != u.len()
        || l.iter()
            .zip(u)
            .any(|(&l, &u)| !l.is_finite() || !u.is_finite() || l > u)
    {
        return Err(fail());
    }
    for i in 0..a.len() {
        if i % 32 == 0 {
            guard(end)?;
        }
        b += &a[i]
            * rat(if (a[i] >= rat(0.)) == lower {
                l[i]
            } else {
                u[i]
            });
    }
    guard(end)?;
    rounded(&b, lower)
}
impl GpuCrownBackward for ExactToyBackend {
    fn crown_backward_gpu(
        &self,
        _: &[GpuCrownLayer],
        _: &[f32],
        _: usize,
        _: &[f32],
        _: &[f32],
    ) -> Result<ny_core::GpuCrownResult> {
        panic!("ordinary CROWN must not be used")
    }
    fn provides_bounded_support4(&self) -> bool {
        true
    }
    fn provides_bounded_lower_cut(&self) -> bool {
        true
    }
    fn crown_backward_gpu_resnet_support4_with_deadline(
        &self,
        s: &[GpuResnetSegment],
        q: &GpuCrownSeed,
        l: &[f32],
        u: &[f32],
        end: Instant,
    ) -> Result<ny_core::GpuCrownResult> {
        guard(end)?;
        exact_objective(q, 4)?;
        let layers = chain(s)?;
        self.support_calls.fetch_add(1, Ordering::SeqCst);
        if self.refuse_support_deadline.load(Ordering::SeqCst) != 0 {
            return Err(NyError::DeadlineExceeded(
                "exact toy support refusal".into(),
            ));
        }
        let mut result = ny_core::GpuCrownResult {
            lower_bounds: Vec::new(),
            upper_bounds: Vec::new(),
        };
        for r in 0..4 {
            for lower in [true, false] {
                let a = q.lower_a[r * q.current_dim..(r + 1) * q.current_dim]
                    .iter()
                    .map(|&x| rat(x))
                    .collect();
                let (a, b) = fold(layers, a, rat(q.lower_b[r]), lower, None, end)?;
                let v = concrete(&a, b, l, u, lower, end)?;
                if lower {
                    result.lower_bounds.push(v)
                } else {
                    result.upper_bounds.push(v)
                }
            }
        }
        guard(end)?;
        Ok(result)
    }
    fn crown_backward_gpu_resnet_lower_cut_with_deadline(
        &self,
        s: &[GpuResnetSegment],
        q: &GpuCrownSeed,
        l: &[f32],
        u: &[f32],
        c: &ResidentLowerCutCarrier,
        end: Instant,
    ) -> Result<f32> {
        guard(end)?;
        exact_objective(q, 1)?;
        let layers = chain(s)?;
        let widths: Vec<_> = layers
            .iter()
            .filter_map(|x| match x {
                GpuCrownLayer::Activation { num_neurons, .. }
                | GpuCrownLayer::ActivationReluDualAlpha { num_neurons, .. } => Some(*num_neurons),
                _ => None,
            })
            .collect();
        let width = *widths.get(c.target_activation()).ok_or_else(fail)?;
        c.validate_for_call(widths.len(), width, 1, end)?;
        let n = self.cut_calls.fetch_add(1, Ordering::SeqCst) + 1;
        if n == self.refuse_cut.load(Ordering::SeqCst) {
            return Err(fail());
        }
        let (a, b) = fold(
            layers,
            q.lower_a.iter().map(|&x| rat(x)).collect(),
            rat(q.lower_b[0]),
            true,
            Some(c),
            end,
        )?;
        let value = concrete(&a, b, l, u, true, end)?;
        guard(end)?;
        Ok(value)
    }
}

pub(crate) fn strict_fixture() -> (GraphNetwork, BoundedTensor, HashMap<String, BoundedTensor>) {
    let mut g = GraphNetwork::new();
    g.add_node(GraphNode::from_input(
        "pre",
        Layer::Linear(
            LinearLayer::new(
                arr2(&[[0.5, 0.5], [0.5, -0.5], [1., 0.]]),
                Some(arr1(&[0., 0., 2.])),
            )
            .unwrap(),
        ),
    ));
    g.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["pre".into()],
    ));
    g.add_node(GraphNode::new(
        "out",
        Layer::Linear(LinearLayer::new(arr2(&[[-2., -2., 1.]]), Some(arr1(&[-0.75]))).unwrap()),
        vec!["relu".into()],
    ));
    g.set_output("out");
    let input =
        BoundedTensor::new(arr1(&[-1., -1.]).into_dyn(), arr1(&[1., 1.]).into_dyn()).unwrap();
    let bounds = g.collect_node_bounds(&input).unwrap();
    (g, input, bounds)
}
fn limits() -> RawRootCutLimits {
    RawRootCutLimits {
        max_nodes: 16,
        max_scratch_bytes: 16 << 20,
        max_work: 100_000_000,
    }
}

#[test]
fn fresh_root_cut_interprets_original_chain_and_improves_exact_diamond() {
    let (g, x, b) = strict_fixture();
    let engine = ExactToyBackend::default();
    let end = Instant::now() + Duration::from_secs(30);
    let q = quote_raw_root_cut(
        &g,
        &x,
        &b,
        None,
        &[1.],
        (-0.75, 1.25),
        "relu",
        [0, 1],
        1.,
        limits(),
        end,
        &mut || true,
    )
    .unwrap();
    let result = evaluate_raw_root_cut(q, &engine, &mut || true).unwrap();
    // Independent all-domain identity: F=5/4-max(|u|,|v|).
    // Its minimum on the original box is1/4, attained at every edge midpoint.
    assert_eq!(result.bounds(), (0.25, 1.25));
    assert_eq!((result.support_calls(), result.cut_calls()), (1, 2));
    assert_eq!(engine.support_calls.load(Ordering::SeqCst), 1);
    assert_eq!(engine.cut_calls.load(Ordering::SeqCst), 2);
}
#[test]
fn fresh_root_cut_late_refusal_and_every_poll_cancel_publish_nothing() {
    let (g, x, b) = strict_fixture();
    let end = Instant::now() + Duration::from_secs(30);
    let quote = || {
        quote_raw_root_cut(
            &g,
            &x,
            &b,
            None,
            &[1.],
            (-0.75, 1.25),
            "relu",
            [0, 1],
            1.,
            limits(),
            end,
            &mut || true,
        )
        .unwrap()
    };
    let engine = ExactToyBackend::default();
    engine.refuse_cut.store(2, Ordering::SeqCst);
    assert!(evaluate_raw_root_cut(quote(), &engine, &mut || true).is_none());
    assert_eq!(engine.cut_calls.load(Ordering::SeqCst), 2);
    let mut polls = 0;
    evaluate_raw_root_cut(quote(), &ExactToyBackend::default(), &mut || {
        polls += 1;
        true
    })
    .unwrap();
    for fail_at in 1..=polls {
        let mut n = 0;
        let out = evaluate_raw_root_cut(quote(), &ExactToyBackend::default(), &mut || {
            n += 1;
            n != fail_at
        });
        assert!(out.is_none(), "poll {fail_at}/{polls}");
        assert_eq!(n, fail_at);
    }
}
#[test]
fn fresh_root_cut_quote_edges_and_unsupported_source_refuse() {
    let (g, x, b) = strict_fixture();
    let end = Instant::now() + Duration::from_secs(30);
    let q = quote_raw_root_cut(
        &g,
        &x,
        &b,
        None,
        &[1.],
        (-0.75, 1.25),
        "relu",
        [0, 1],
        1.,
        limits(),
        end,
        &mut || true,
    )
    .unwrap();
    let exact = RawRootCutLimits {
        max_work: q.work(),
        max_scratch_bytes: q.scratch_bytes(),
        ..limits()
    };
    assert!(quote_raw_root_cut(
        &g,
        &x,
        &b,
        None,
        &[1.],
        (-0.75, 1.25),
        "relu",
        [0, 1],
        1.,
        exact,
        end,
        &mut || true
    )
    .is_some());
    assert_eq!(q.candidate_count(), 2);
    let one_cap = RawRootCutLimits {
        max_work: exact.max_work - 1,
        ..exact
    };
    let one = quote_raw_root_cut(
        &g,
        &x,
        &b,
        None,
        &[1.],
        (-0.75, 1.25),
        "relu",
        [0, 1],
        1.,
        one_cap,
        end,
        &mut || true,
    )
    .unwrap();
    assert_eq!(one.candidate_count(), 1);
    assert!(one.work() < q.work());
    let one_exact = RawRootCutLimits {
        max_work: one.work(),
        ..exact
    };
    let one = quote_raw_root_cut(
        &g,
        &x,
        &b,
        None,
        &[1.],
        (-0.75, 1.25),
        "relu",
        [0, 1],
        1.,
        one_exact,
        end,
        &mut || true,
    )
    .unwrap();
    let engine = ExactToyBackend::default();
    assert_eq!(
        evaluate_raw_root_cut(one, &engine, &mut || true)
            .unwrap()
            .bounds(),
        (0.25, 1.25)
    );
    assert_eq!(engine.cut_calls.load(Ordering::SeqCst), 1);
    for lim in [
        RawRootCutLimits {
            max_work: one_exact.max_work - 1,
            ..exact
        },
        RawRootCutLimits {
            max_scratch_bytes: exact.max_scratch_bytes - 1,
            ..exact
        },
    ] {
        assert!(quote_raw_root_cut(
            &g,
            &x,
            &b,
            None,
            &[1.],
            (-0.75, 1.25),
            "relu",
            [0, 1],
            1.,
            lim,
            end,
            &mut || true
        )
        .is_none());
    }
    assert!(quote_raw_root_cut(
        &g,
        &x,
        &b,
        None,
        &[1.],
        (-0.75, 1.25),
        "relu",
        [0, 0],
        1.,
        limits(),
        end,
        &mut || true
    )
    .is_none());
    assert!(quote_raw_root_cut(
        &g,
        &x,
        &b,
        None,
        &[1.],
        (-0.75, 1.25),
        "relu",
        [0, 1],
        f32::NAN,
        limits(),
        end,
        &mut || true
    )
    .is_none());
    let mut missing = b.clone();
    missing.remove("pre");
    assert!(quote_raw_root_cut(
        &g,
        &x,
        &missing,
        None,
        &[1.],
        (-0.75, 1.25),
        "relu",
        [0, 1],
        1.,
        limits(),
        end,
        &mut || true
    )
    .is_none());
}

fn diagnostic_quote<'a>(
    graph: &'a GraphNetwork,
    input: &'a BoundedTensor,
    bounds: &'a HashMap<String, BoundedTensor>,
) -> RawRootCutQuote<'a> {
    quote_raw_root_cut(
        graph,
        input,
        bounds,
        None,
        &[1.],
        (-0.75, 1.25),
        "relu",
        [0, 1],
        1.,
        limits(),
        Instant::now() + Duration::from_secs(30),
        &mut || true,
    )
    .unwrap()
}

#[test]
fn root_cut_detail_preserves_exact_completion_and_partial_refusal() {
    let (graph, input, bounds) = strict_fixture();
    for refuse in [0, 2] {
        let engine = ExactToyBackend::default();
        engine.refuse_cut.store(refuse, Ordering::SeqCst);
        let mut observation = ExecutionObservation::default();
        let completed = evaluate_raw_root_cut_observed(
            diagnostic_quote(&graph, &input, &bounds),
            &engine,
            &mut || true,
            AdviceMode::Weighted,
            &mut observation,
        );
        assert_eq!(observation.support_api_entered, 1);
        assert_eq!(observation.support_api_ok, 1);
        assert_eq!(observation.lower_api_entered, 2);
        assert_eq!(engine.support_calls.load(Ordering::SeqCst), 1);
        assert_eq!(engine.cut_calls.load(Ordering::SeqCst), 2);
        assert!(observation.poll_alive_at_return);
        assert!(!observation.deadline_expired_at_return);
        if refuse == 0 {
            let completed = completed.unwrap();
            assert_eq!(completed.bounds().0.to_bits(), 0.25f32.to_bits());
            assert_eq!(completed.bounds().1.to_bits(), 1.25f32.to_bits());
            assert_eq!((completed.support_calls(), completed.cut_calls()), (1, 2));
            assert_eq!(observation.phase, "complete");
            assert_eq!(observation.error_class, "none");
            assert_eq!((observation.lower_api_ok, observation.merges_ok), (2, 2));
        } else {
            // An earlier successful fold remains private after the later refusal.
            assert!(completed.is_none());
            assert_eq!(observation.phase, "lower-api");
            assert_eq!(observation.error_class, "unsupported-op");
            assert_eq!((observation.lower_api_ok, observation.merges_ok), (1, 1));
        }
    }
}

#[test]
fn root_cut_detail_distinguishes_expiry_callback_and_typed_api_refusal() {
    let (graph, input, bounds) = strict_fixture();
    for refusal in ["clock", "callback", "api"] {
        let engine = ExactToyBackend::default();
        let mut quote = diagnostic_quote(&graph, &input, &bounds);
        if refusal == "clock" {
            quote.deadline = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
        }
        if refusal == "api" {
            engine.refuse_support_deadline.store(1, Ordering::SeqCst);
        }
        let mut callbacks = 0;
        let mut observation = ExecutionObservation::default();
        let completed = evaluate_raw_root_cut_observed(
            quote,
            &engine,
            &mut || {
                callbacks += 1;
                refusal != "callback"
            },
            AdviceMode::Weighted,
            &mut observation,
        );
        assert!(completed.is_none());
        assert_eq!(observation.lower_api_entered, 0);
        assert_eq!(observation.support_api_ok, 0);
        assert_eq!(observation.deadline_expired_at_return, refusal == "clock");
        assert_eq!(observation.poll_alive_at_return, refusal == "api");
        match refusal {
            "clock" => {
                assert_eq!(callbacks, 0);
                assert_eq!(observation.phase, "entry");
                assert_eq!(observation.error_class, "none");
            }
            "callback" => {
                assert_eq!(callbacks, 1); // The observer must not poll again.
                assert_eq!(observation.phase, "entry");
                assert_eq!(observation.error_class, "none");
            }
            "api" => {
                assert!(callbacks > 1);
                assert_eq!(observation.phase, "support-api");
                assert_eq!(observation.error_class, "deadline");
                assert_eq!(observation.support_api_entered, 1);
                assert_eq!(engine.support_calls.load(Ordering::SeqCst), 1);
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn root_cut_detail_cancellation_after_returned_fold_grants_no_authority() {
    let (graph, input, bounds) = strict_fixture();
    let engine = ExactToyBackend::default();
    let mut observation = ExecutionObservation::default();
    let mut refused_polls = 0;
    let completed = evaluate_raw_root_cut_observed(
        diagnostic_quote(&graph, &input, &bounds),
        &engine,
        &mut || {
            let live = engine.cut_calls.load(Ordering::SeqCst) == 0;
            if !live {
                refused_polls += 1;
            }
            live
        },
        AdviceMode::Weighted,
        &mut observation,
    );
    assert!(completed.is_none());
    assert_eq!(refused_polls, 1);
    assert_eq!(observation.phase, "lower-api");
    assert_eq!(observation.error_class, "none");
    assert_eq!(observation.support_api_ok, 1);
    assert_eq!(
        (observation.lower_api_entered, observation.lower_api_ok),
        (1, 1)
    );
    assert_eq!(observation.merges_ok, 0);
    assert!(!observation.poll_alive_at_return);
    assert!(!observation.deadline_expired_at_return);
}

// The duplicate outputs are correlated in the authored source. IBP loses that
// relation: both selected preactivations cross zero, while exact prefix folding
// gives z1=c and z2=d+span*y. No hand-built bound map or support is substituted.
pub(crate) fn fresh_stable_fixture(
    c: f32,
    d: f32,
    span: f32,
    margin: f32,
) -> (
    GraphNetwork,
    BoundedTensor,
    HashMap<String, BoundedTensor>,
    (f32, f32),
) {
    let mut g = GraphNetwork::new();
    g.add_node(GraphNode::from_input(
        "copies",
        Layer::Linear(
            LinearLayer::new(arr2(&[[1., 0.], [1., 0.], [0., 1.], [0., 1.]]), None).unwrap(),
        ),
    ));
    g.add_node(GraphNode::new(
        "pre",
        Layer::Linear(
            LinearLayer::new(
                arr2(&[[1., -1., 0., 0.], [0., 0., 1. + span, -1.]]),
                Some(arr1(&[c, d])),
            )
            .unwrap(),
        ),
        vec!["copies".into()],
    ));
    g.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["pre".into()],
    ));
    g.add_node(GraphNode::new(
        "out",
        Layer::Linear(
            LinearLayer::new(
                arr2(&[[-2., -2.]]),
                Some(arr1(&[margin + 2. * c.max(0.) + 2. * (d + span).max(0.)])),
            )
            .unwrap(),
        ),
        vec!["relu".into()],
    ));
    g.set_output("out");
    let input =
        BoundedTensor::new(arr1(&[-1., -1.]).into_dyn(), arr1(&[1., 1.]).into_dyn()).unwrap();
    let bounds = g.collect_node_bounds(&input).unwrap();
    let pre = &bounds["pre"];
    assert!(pre
        .lower()
        .iter()
        .zip(pre.upper())
        .all(|(&l, &u)| l < 0. && u > 0.));
    let end = Instant::now() + Duration::from_secs(30);
    let full = extract_charged_raw_bn_chain(
        &g,
        &input,
        "out",
        &bounds,
        &bounds,
        None,
        ExtractionLimits {
            max_nodes: 16,
            max_caller_bytes: 16 << 20,
        },
        end,
        &mut || true,
    )
    .unwrap();
    let ordinary = |lower| {
        let (a, b) = fold(
            chain(&full.segments).unwrap(),
            vec![rat(1.)],
            rat(0.),
            lower,
            None,
            end,
        )
        .unwrap();
        concrete(
            &a,
            b,
            input.lower().as_slice().unwrap(),
            input.upper().as_slice().unwrap(),
            lower,
            end,
        )
        .unwrap()
    };
    let old = (ordinary(true), ordinary(false));
    assert!(old.0 < 0. && old.0 <= margin && old.1 >= margin);
    (g, input, bounds, old)
}

#[test]
fn fresh_stable_support_keeps_exact_points_lines_and_zero_boundaries() {
    for (c, d, span) in [
        (0.5, 0.25, 0.),
        (-0.5, -0.25, 0.),
        (0.5, -0.25, 0.),
        (0., 0.25, 0.),
        (0., 0., 0.),
        (0.5, 0., 0.25),
        (0., 0.5, 0.25),
    ] {
        let (g, input, nodes, old) = fresh_stable_fixture(c, d, span, 0.25);
        let original_nodes = nodes.clone();
        let engine = ExactToyBackend::default();
        let end = Instant::now() + Duration::from_secs(30);
        let quote = quote_raw_root_cut(
            &g,
            &input,
            &nodes,
            None,
            &[1.],
            old,
            "relu",
            [0, 1],
            1.,
            limits(),
            end,
            &mut || true,
        )
        .unwrap();
        let mut observation = ExecutionObservation::default();
        let completed = evaluate_raw_root_cut_observed(
            quote,
            &engine,
            &mut || true,
            AdviceMode::Weighted,
            &mut observation,
        )
        .unwrap();
        // Authored function is constant margin for points, or minimized at y=1
        // for these lines. Its exact minimum is 1/4. The full-scale cut cancels
        // both negative ReLU coefficients and certifies that same minimum.
        assert_eq!(
            completed.bounds().0.to_bits(),
            0.25f32.to_bits(),
            "{c},{d},{span}"
        );
        assert_eq!(completed.bounds().1.to_bits(), old.1.to_bits());
        assert_eq!(completed.cut_normal(), [-1., -1., 2., 2.]);
        assert_eq!((completed.support_calls(), completed.cut_calls()), (1, 2));
        assert_eq!(engine.support_calls.load(Ordering::SeqCst), 1);
        assert_eq!(engine.cut_calls.load(Ordering::SeqCst), 2);
        assert_eq!(observation.phase, "complete");
        assert_eq!(observation.error_class, "none");
        for (name, before) in &original_nodes {
            assert_eq!(nodes[name].lower(), before.lower());
            assert_eq!(nodes[name].upper(), before.upper());
        }
    }
}

#[test]
fn fresh_stable_support_negative_property_and_empty_support_do_not_verify() {
    let (g, input, nodes, old) = fresh_stable_fixture(0.5, -0.25, 0., -0.25);
    let end = Instant::now() + Duration::from_secs(30);
    let quote = quote_raw_root_cut(
        &g,
        &input,
        &nodes,
        None,
        &[1.],
        old,
        "relu",
        [0, 1],
        1.,
        limits(),
        end,
        &mut || true,
    )
    .unwrap();
    let completed =
        evaluate_raw_root_cut(quote, &ExactToyBackend::default(), &mut || true).unwrap();
    assert_eq!(completed.bounds().0.to_bits(), (-0.25f32).to_bits());
    assert_eq!(completed.bounds().1.to_bits(), old.1.to_bits());
    // Closed orthants retain a stable point, but empty/inverted/nonfinite
    // supports must still refuse instead of treating emptiness as a proof.
    let point = Octahedron2::from_bounds(0.5, 0.5, -0.25, -0.25, 0.25, 0.25, 0.75, 0.75);
    assert!(!point.both_unstable());
    assert_eq!(
        ExactRelu2Support::new(&point)
            .unwrap()
            .certify_normal_certificate([-1., -1., 2., 2.])
            .unwrap()
            .facet()
            .b,
        0.75
    );
    for bad in [
        Octahedron2 {
            s_lo: 0.5,
            s_hi: 0.5,
            ..point.clone()
        },
        Octahedron2 {
            l1: 0.75,
            ..point.clone()
        },
        Octahedron2 {
            u1: f64::NAN,
            ..point
        },
    ] {
        assert!(ExactRelu2Support::new(&bad).is_none());
    }
}

#[test]
fn fresh_stable_support_every_poll_and_late_fold_failure_publish_nothing() {
    let (g, input, nodes, old) = fresh_stable_fixture(0., 0., 0., 0.25);
    let end = Instant::now() + Duration::from_secs(30);
    let quote = || {
        quote_raw_root_cut(
            &g,
            &input,
            &nodes,
            None,
            &[1.],
            old,
            "relu",
            [0, 1],
            1.,
            limits(),
            end,
            &mut || true,
        )
        .unwrap()
    };
    let engine = ExactToyBackend::default();
    engine.refuse_cut.store(2, Ordering::SeqCst);
    assert!(evaluate_raw_root_cut(quote(), &engine, &mut || true).is_none());
    assert_eq!(engine.cut_calls.load(Ordering::SeqCst), 2);
    let mut polls = 0;
    evaluate_raw_root_cut(quote(), &ExactToyBackend::default(), &mut || {
        polls += 1;
        true
    })
    .unwrap();
    for fail_at in 1..=polls {
        let mut seen = 0;
        assert!(
            evaluate_raw_root_cut(quote(), &ExactToyBackend::default(), &mut || {
                seen += 1;
                seen != fail_at
            })
            .is_none(),
            "stable poll {fail_at}/{polls}"
        );
        assert_eq!(seen, fail_at);
    }
}

#[test]
fn exact_support_oracle_wide_affine_values_caps_errors_and_deadline() {
    let end = Instant::now() + Duration::from_secs(30);
    // A is the mean of all 1024 coordinates, C the alternating-sign mean.
    // Rows are A+1/4, C-1/2, -A/2+1/8. Every coordinate contributes.
    let mut weight = vec![0.; 3 * 1024];
    for j in 0..1024 {
        weight[j] = 1. / 1024.;
        weight[1024 + j] = if j % 2 == 0 { 1. } else { -1. } / 1024.;
        weight[2048 + j] = -1. / 2048.;
    }
    let source = GpuCrownLayer::Linear {
        weight: weight.into(),
        bias: Some(vec![0.25, -0.5, 0.125].into()),
        out_features: 3,
        in_features: 1024,
        cert_err: ny_core::CertifiedWeightError::EXACT,
    };
    let segments = vec![GpuResnetSegment::Chain(vec![source])];
    let seed = exact_seed(
        4,
        3,
        vec![1., 1., 1., 1., -1., 1., 1., 1., -1., 1., -1., -1.],
    );
    let lo = vec![-1.; 1024];
    let hi = vec![1.; 1024];
    let engine = ExactToyBackend::default();
    let result = engine
        .crown_backward_gpu_resnet_support4_with_deadline(&segments, &seed, &lo, &hi, end)
        .unwrap();
    // The two parity blocks give radii [1,1,3/2,3/2] and biases
    // [-1/8,7/8,-3/8,5/8], independently of the interpreter loop.
    assert_eq!(result.lower_bounds, vec![-1.125, -0.125, -1.875, -0.875]);
    assert_eq!(result.upper_bounds, vec![0.875, 1.875, 1.125, 2.125]);
    assert_eq!(engine.support_calls.load(Ordering::SeqCst), 1);
    assert_eq!(engine.cut_calls.load(Ordering::SeqCst), 0);
    let oversized_seed = exact_seed(4, 17, vec![0.; 4 * 17]);
    assert!(matches!(
        engine.crown_backward_gpu_resnet_support4_with_deadline(
            &segments,
            &oversized_seed,
            &lo,
            &hi,
            end,
        ),
        Err(NyError::UnsupportedOp(_))
    ));

    // Exercise the inclusive matrix limit with nonzero source rows and a
    // hand-computed exact fold: 32 identical mean rows plus 1/8 biases.
    let affine = |inputs: usize, outputs: usize, error| GpuCrownLayer::Linear {
        weight: vec![1. / 1024.; inputs * outputs].into(),
        bias: Some(vec![0.125; outputs].into()),
        out_features: outputs,
        in_features: inputs,
        cert_err: error,
    };
    let (a, b) = fold(
        &[affine(1024, 32, ny_core::CertifiedWeightError::EXACT)],
        vec![rat(1.); 32],
        rat(0.),
        true,
        None,
        end,
    )
    .unwrap();
    assert!(a.iter().all(|v| *v == rat(1. / 32.)));
    assert_eq!(b, rat(4.));
    assert_eq!(concrete(&a, b.clone(), &lo, &hi, true, end).unwrap(), -28.);
    assert_eq!(concrete(&a, b, &lo, &hi, false, end).unwrap(), 36.);
    for (inputs, outputs) in [(1025, 1), (1, 33)] {
        assert!(matches!(
            fold(
                &[affine(
                    inputs,
                    outputs,
                    ny_core::CertifiedWeightError::EXACT
                )],
                vec![rat(1.); outputs],
                rat(0.),
                true,
                None,
                end,
            ),
            Err(NyError::UnsupportedOp(_))
        ));
    }
    for error in [
        ny_core::CertifiedWeightError {
            weight_rel_err: 0.125,
            bias_abs_err: 0.,
        },
        ny_core::CertifiedWeightError {
            weight_rel_err: 0.,
            bias_abs_err: 0.125,
        },
    ] {
        assert!(matches!(
            fold(
                &[affine(1024, 1, error)],
                vec![rat(1.)],
                rat(0.),
                true,
                None,
                end,
            ),
            Err(NyError::UnsupportedOp(_))
        ));
    }
    let expired = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
    assert!(matches!(
        fold(
            &[affine(1024, 1, ny_core::CertifiedWeightError::EXACT)],
            vec![rat(1.)],
            rat(0.),
            true,
            None,
            expired,
        ),
        Err(NyError::DeadlineExceeded(_))
    ));
    assert!(matches!(
        concrete(&a, rat(0.), &lo, &hi, true, expired),
        Err(NyError::DeadlineExceeded(_))
    ));
    assert!(matches!(
        concrete(&a, rat(0.), &lo[..1023], &hi, true, end),
        Err(NyError::UnsupportedOp(_))
    ));

    assert!(matches!(
        engine
            .crown_backward_gpu_resnet_support4_with_deadline(&segments, &seed, &lo, &hi, expired,),
        Err(NyError::DeadlineExceeded(_))
    ));
}
