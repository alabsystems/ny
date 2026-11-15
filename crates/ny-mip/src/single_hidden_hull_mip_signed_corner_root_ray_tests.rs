// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! UNRUN controls for an explicitly supplied Plan and genuine original Source.
//! A root-family separation does not imply whole-pipeline superiority: another
//! existing alpha choice, warm child proof or native LP may close the same box.
use super::super::super::tests::{end, rat};
use super::*;
use crate::single_hidden_hull::bab::{HullBabOptions, HullBabStop};
use crate::single_hidden_hull::HullLpStatus;
use crate::MipParts;
use num_rational::BigRational as Q;
use num_traits::Zero;
use ny_core::Bound;

struct Fixture {
    weights: Vec<Vec<f64>>,
    biases: Vec<Vec<f64>>,
    dimensions: Vec<usize>,
    bounds: Vec<Vec<Bound>>,
}
impl Fixture {
    const D: usize = 16;
    fn separator() -> Self {
        let mut first = vec![0.; 4 * Self::D];
        first[0] = 2.;
        first[Self::D] = 1.;
        first[2 * Self::D + 1] = 1.;
        first[3 * Self::D + 1] = 1.;
        Self {
            weights: vec![first, vec![1., -1., 1., -1., 0., 0., 0., 0.]],
            biases: vec![vec![0., 0., 2., 2.], vec![0.25, 0.]],
            dimensions: vec![Self::D, 4, 2],
            bounds: vec![vec![
                Bound::new(-2., 2.),
                Bound::new(-1., 1.),
                Bound::new(1., 3.),
                Bound::new(1., 3.),
            ]],
        }
    }
    fn offset_counterexample() -> Self {
        Self {
            weights: vec![vec![1., -1.], vec![-1., 2., 0., 0.]],
            biases: vec![vec![0., 1.], vec![0.25, 0.]],
            dimensions: vec![1, 2, 2],
            bounds: vec![vec![Bound::new(-1., 1.), Bound::new(0., 2.)]],
        }
    }
    fn source(&self) -> HullMarginProblem {
        HullMarginProblem::new(
            self.weights[0].clone(),
            self.biases[0].clone(),
            self.dimensions[0],
            &self.weights[1],
            &self.biases[1],
            2,
            &[(0, 1.), (1, -1.)],
            0.,
            vec![-1.; self.dimensions[0]],
            vec![1.; self.dimensions[0]],
        )
        .unwrap()
    }
    fn plan(&self) -> Plan {
        if self.dimensions[1] == 4 {
            Plan {
                components: vec![
                    Component {
                        negative: vec![1],
                        positive: vec![0, 2],
                        positive_units: vec![2, 2],
                        g: vec![0.; Self::D],
                        // Mask-major: inactive [0,1], active [1,1].
                        alpha: vec![0., 1., 1., 1.],
                    },
                    Component {
                        negative: vec![3],
                        positive: vec![],
                        positive_units: vec![],
                        g: vec![0.; Self::D],
                        alpha: vec![],
                    },
                ],
                leftover: vec![],
            }
        } else {
            Plan {
                components: vec![Component {
                    negative: vec![0],
                    positive: vec![],
                    positive_units: vec![],
                    g: vec![0.; self.dimensions[0]],
                    alpha: vec![],
                }],
                leftover: vec![1],
            }
        }
    }
    fn original(&self) -> MipParts {
        let mut e = crate::encode_feedforward(
            &self.weights,
            &self.biases,
            &self.dimensions,
            &vec![Bound::new(-1., 1.); self.dimensions[0]],
            &self.bounds,
        )
        .unwrap();
        e.constrain_output_leq_row(0, 1).unwrap();
        e.into_parts()
    }
    fn encoding(&self) -> HullMipEncoding<'_> {
        HullMipEncoding {
            weights: &self.weights,
            biases: &self.biases,
            dimensions: &self.dimensions,
            hidden_bounds: &self.bounds,
            first_output: 0,
            second_output: 1,
            reverse: false,
            unsafe_row: Row(self.original().problem.num_rows() - 1),
        }
    }
}

fn satisfies(parts: &MipParts, point: &[Q], rows: usize) -> bool {
    point.len() == parts.num_cols
        && parts.problem.cols().iter().zip(point).all(|(c, x)| {
            (!c.lb.is_finite() || *x >= rat(c.lb)) && (!c.ub.is_finite() || *x <= rat(c.ub))
        })
        && parts.problem.rows()[..rows].iter().all(|r| {
            let value: Q = r.coeffs.iter().map(|&(j, a)| rat(a) * &point[j]).sum();
            (!r.lb.is_finite() || value >= rat(r.lb)) && (!r.ub.is_finite() || value <= rat(r.ub))
        })
}

fn paid_owner<'p>(p: &'p HullMarginProblem, f: &Fixture, enabled: bool) -> HullRootMipCuts<'p> {
    let until = end();
    let mut attempt = primal_capture::Attempt::new_with_native_counts(p, until);
    let mut cuts = propose_root_mip_cuts_inner(p, until, Some(&mut attempt)).unwrap();
    assert_eq!(cuts.proposal_lp_calls, 1);
    if f.dimensions[1] == 2 {
        // This is legal proposal advice for a stable positive leftover, not a
        // replacement of Source coefficients or an invented checked receipt.
        cuts.multipliers.alpha[1] = 1.;
    }
    let mut b = Budget::new_with_portfolio_conditions(&cuts, until, Duration::ZERO).unwrap();
    b.phase_all_enabled = true;
    b.phase_cube_enabled = true;
    let plan = f.plan();
    let cost = plan.cost(p).unwrap();
    b.reserve(cost).unwrap();
    b.proof(cost, false).unwrap();
    let checked = certificate::check(&cuts, &plan, &b, b.search_end, &mut || true).unwrap();
    component_rows::admit(&cuts, &plan, &b).unwrap();
    native_publication::fund(&cuts, &b).unwrap();
    child::prepay(&cuts, &plan, &b).unwrap();
    super::super::prepay(&cuts, &plan, &b).unwrap();
    // Respect the preceding optional route; this K=1 fixture honestly declines
    // input-phase search. The new method never borrows any of its old credit.
    let _input_phase = phase::cube::cell::input_phase_resources::prepay(&cuts, &b);
    let old_holds = {
        let c = b.state.lock().unwrap();
        let grant = c.corner.grant.as_ref().unwrap();
        (
            b.admitted_bytes,
            c.reserved,
            c.component_reserved,
            c.prepaid_collection_work,
            c.prepaid_bind_work,
            grant.prices,
            grant.base_calls,
            grant.base_call_cap,
        )
    };
    if enabled {
        prepay(&cuts, &plan, &b).unwrap();
    }
    {
        let c = b.state.lock().unwrap();
        let grant = c.corner.grant.as_ref().unwrap();
        assert_eq!(
            old_holds,
            (
                b.admitted_bytes,
                c.reserved,
                c.component_reserved,
                c.prepaid_collection_work,
                c.prepaid_bind_work,
                grant.prices,
                grant.base_calls,
                grant.base_call_cap
            )
        );
        assert_eq!(grant.root_ray.is_some(), enabled);
    }
    assert_eq!(native_publication::funding(&b).funded_calls, 1);
    let seal = Arc::new(signature(&cuts).collect());
    cuts.signed = Some(row(&cuts, &plan, checked, cost, &b, &seal));
    cuts
}

#[test]
fn root_ray_genuine_encoder_unsafe_relaxation_and_closed_graph_oracle() {
    let f = Fixture::separator();
    let p = f.source();
    assert_eq!(p.hidden_dim, 4); // Stable original heads are not erased from Source.
    let original = f.original();
    let heads = binding::head_columns(&f.encoding(), &original).unwrap();
    assert_eq!(original.binary_vars.len(), 2);
    let mut fake = vec![Q::zero(); original.num_cols];
    fake[Fixture::D + 2] = rat(2.);
    fake[Fixture::D + 3] = rat(2.);
    fake[heads[1].0] = rat(0.5);
    fake[heads[2].0] = rat(2.);
    fake[heads[3].0] = rat(2.);
    fake[original.binary_vars[0].0] = rat(0.5);
    fake[original.binary_vars[1].0] = rat(0.5);
    fake[original.output_vars[0].0] = rat(-0.25);
    assert!(satisfies(&original, &fake, original.problem.num_rows()));
    // This exact unsafe point belongs to the original LP relaxation only.
    // No assertion is made that it survives every older appended cut family.

    // Literal all-domain identity: R(2x)=2R(x), and the two stable heads cancel.
    // Each x half-interval is affine; its endpoint minimum proves F>=1/4.
    for (lo, hi, slope) in [(-1., 0., 0.), (0., 1., 1.)] {
        let left = rat(0.25) + rat(slope) * rat(lo);
        let right = rat(0.25) + rat(slope) * rat(hi);
        assert_eq!(left.min(right), rat(0.25));
    }
    for x in [-1., -0.25, 0., 0.5, 1.] {
        for y in [-1., 0., 1.] {
            let pre = [rat(2.) * rat(x), rat(x), rat(y) + rat(2.), rat(y) + rat(2.)];
            let exact = rat(0.25) + pre[0].clone().max(Q::zero()) - pre[1].clone().max(Q::zero())
                + pre[2].clone()
                - pre[3].clone();
            assert_eq!(exact, rat(0.25) + rat(x).max(Q::zero()));
            let zero_count = if x == 0. { 2 } else { 0 };
            for ties in 0..1usize << zero_count {
                let mut point = vec![Q::zero(); original.num_cols];
                point[original.input_vars[0].0] = rat(x);
                point[original.input_vars[1].0] = rat(y);
                for j in 0..4 {
                    point[Fixture::D + j] = pre[j].clone();
                    point[heads[j].0] = pre[j].clone().max(Q::zero());
                }
                for j in 0..2 {
                    let active = if x == 0. {
                        (ties >> j) & 1
                    } else {
                        usize::from(x > 0.)
                    };
                    point[original.binary_vars[j].0] = rat(active as f64);
                }
                point[original.output_vars[0].0] = exact.clone();
                assert!(satisfies(
                    &original,
                    &point,
                    original.problem.num_rows() - 1
                ));
                assert!(!satisfies(&original, &point, original.problem.num_rows()));
            }
        }
    }
}

#[test]
fn root_ray_genuine_two_replay_cache_strengthens_only_the_fixed_plan_root() {
    let f = Fixture::separator();
    let p = f.source(); // Keep every rounded original p.c/p.d word.
    let mut baseline_bits = None;
    for enabled in [false, true] {
        let cuts = paid_owner(&p, &f, enabled);
        let row = cuts.signed.as_ref().unwrap();
        let original_tuple = (
            row.lower.to_bits(),
            row.constant.to_bits(),
            row.g.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        );
        assert!(row.lower < -2.74);
        if let Some(old) = &baseline_bits {
            assert_eq!(&original_tuple, old);
        } else {
            baseline_bits = Some(original_tuple.clone());
        }
        let mut parts = f.original();
        let old_ir = parts.problem.clone();
        cuts.append_to(&mut parts, &f.encoding(), end()).unwrap();
        assert_eq!(&parts.problem.rows()[..old_ir.num_rows()], old_ir.rows());
        assert_eq!(parts.problem.cols(), old_ir.cols());
        assert_eq!(row.stats().source_replays, 1);
        let first = row.stats().component_rows.phase.cube.cell;
        assert_eq!(first.component, Some(0));
        assert!(first.ray.admitted);
        assert_eq!(first.ray.masks, 2);
        let _proof = cuts.best_signed_root_proof(end());
        assert_eq!(row.stats().source_replays, 2);
        let cache = cuts
            .signed_child_cache
            .get()
            .expect("second original replay installed cache");
        assert_eq!(cache.stats().cached_masks, 4);
        let bound = cache.bind(&cuts, end(), &mut || true).unwrap();
        let root = bound.root_lower().unwrap();
        let report = cache.stats();
        let base = report.corner_base_root_lower.unwrap();
        assert!(base < -0.74 && base > -0.76);
        if enabled {
            assert!(report.corner_ray_axes > 0);
            assert_eq!(report.corner_ray_root_lower, report.corner_root_lower);
            assert!(root > 0.24);
            assert!(rat(root) <= rat(0.25));
            // An unconditional global root floor can be inherited by a child.
            let child = bound.lower_zero(&[], end(), &mut || true).unwrap();
            assert!(child > 0.24 && rat(child) <= rat(0.25));
        } else {
            assert_eq!(report.corner_ray_axes, 0);
            assert!(report.corner_ray_root_lower.is_none());
            assert_eq!(report.corner_base_root_lower, report.corner_root_lower);
            assert!(root < -0.74 && root > -0.76);
        }
        assert_eq!(
            original_tuple,
            (
                row.lower.to_bits(),
                row.constant.to_bits(),
                row.g.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            )
        );
        assert_eq!(cuts.proposal_lp_calls, 1);
        assert_eq!(native_publication::funding(&row.budget).funded_calls, 1);
        assert!(row.replay_with_cache(&cuts, end(), &mut || true).is_none());
    }
}

#[test]
fn root_ray_genuine_cache_allows_public_continuation_to_close_without_new_lp() {
    let f = Fixture::separator();
    let p = f.source();
    let cuts = paid_owner(&p, &f, true);
    let mut parts = f.original();
    cuts.append_to(&mut parts, &f.encoding(), end()).unwrap();
    let _proof = cuts.best_signed_root_proof(end());
    let cache = cuts.signed_child_cache.get().unwrap();
    assert_eq!(
        (cache.stats().source_replays, cache.stats().bind_attempts),
        (2, 0)
    );
    assert!(cache.stats().corner_root_lower.unwrap() > 0.24);
    let proof = cuts
        .continue_hull_bab_with_native_rows(
            end(),
            HullBabOptions {
                workers: 1,
                max_nodes: 1,
            },
        )
        .unwrap();
    let report = proof.report();
    assert!(proof.verified(), "{report:?}");
    assert_eq!(report.stop, HullBabStop::Proved);
    assert_eq!((report.nodes, report.closed, report.open), (1, 1, 0));
    assert_eq!((report.root_lp_calls, report.proposal_lp_calls), (0, 0));
    assert_eq!(report.root_lp_status, HullLpStatus::Skipped);
    assert!(report.root_lp_value.is_none());
    assert_eq!(cuts.proposal_lp_calls, 1);
    assert_eq!(cuts.signed.as_ref().unwrap().stats().source_replays, 2);
    // Do not assert the disabled continuation fails: its original warm-child
    // alpha search or another valid earlier route may also prove this fixture.
}

#[test]
fn root_ray_genuine_worse_legal_advice_retains_bit_identical_old_corner_root() {
    let f = Fixture::separator();
    let p = f.source();
    let mut baseline = None;
    for enabled in [false, true] {
        let cuts = paid_owner(&p, &f, enabled);
        let row = cuts.signed.as_ref().unwrap();
        assert!(row
            .replay(&cuts, end(), &mut || true)
            .unwrap()
            .cache
            .is_none());
        if enabled {
            let mut counts = row.budget.state.lock().unwrap();
            let advice = counts
                .corner
                .grant
                .as_mut()
                .unwrap()
                .root_ray
                .as_mut()
                .unwrap()
                .advice
                .as_mut()
                .expect("first actual replay captured all target masks");
            assert_eq!((advice.target, advice.entries.len()), (0, 2));
            for (mask, entry) in advice.entries.iter_mut().enumerate() {
                // Perturb only untrusted numerical advice. The captured target,
                // complete coverage, Source, Plan and replay authority stay real.
                // These signs are legal, but both endpoint penalties are poor.
                entry.head = 1;
                entry.beta = if mask == 0 { 100. } else { -100. };
            }
        }
        let cache = row
            .replay_with_cache(&cuts, end(), &mut || true)
            .unwrap()
            .cache
            .unwrap();
        let bound = cache.bind(&cuts, end(), &mut || true).unwrap();
        let value = bound.root_lower().unwrap();
        assert!(value > -0.76 && value < -0.74);
        if let Some(bits) = baseline {
            assert_eq!(value.to_bits(), bits);
        } else {
            baseline = Some(value.to_bits());
        }
        assert_eq!(row.stats().source_replays, 2);
        assert_eq!(cuts.proposal_lp_calls, 1);
    }
}

#[test]
fn root_ray_original_raw_offset_is_not_refunded_when_legal_penalty_worsens_a_mask() {
    let f = Fixture::offset_counterexample();
    let p = f.source();
    let until = end();
    let mut attempt = primal_capture::Attempt::new_with_native_counts(&p, until);
    let mut cuts = propose_root_mip_cuts_inner(&p, until, Some(&mut attempt)).unwrap();
    cuts.multipliers.alpha[1] = 1.; // Legal advice for the stable positive head.
    let b = Budget::new_with_portfolio_conditions(&cuts, until, Duration::ZERO).unwrap();
    let plan = f.plan();
    let cost = plan.cost(&p).unwrap();
    b.reserve(cost).unwrap();
    b.proof(cost, false).unwrap();
    let checked = certificate::check(&cuts, &plan, &b, b.search_end, &mut || true).unwrap();
    let c = &plan.components[0];
    // Reconstruct precisely the original raw affine arithmetic, then compare
    // its result with the actual certificate helper. No Source word changes.
    let mut raw = Vec::new();
    for mask in 0..2 {
        let mut q: Vec<_> = c.g.iter().map(|&v| I::point(-v).unwrap()).collect();
        let mut constant = I::zero();
        if mask == 1 {
            let k = I::point(p.c[0]).unwrap();
            constant = constant.add(k.scale(p.b1[0]).unwrap()).unwrap();
            for (i, qi) in q.iter_mut().enumerate() {
                *qi = qi.add(k.scale(p.w_row(0)[i]).unwrap()).unwrap();
            }
        }
        let mut lower = constant.lo;
        for (i, qi) in q.iter().enumerate() {
            lower = finite(add_down(lower, qi.minimum(p.lower[i], p.upper[i]).unwrap())).unwrap();
        }
        assert_eq!(
            lower.to_bits(),
            certificate::mask_lower(&cuts, c, mask, end(), &mut || true,)
                .unwrap()
                .to_bits()
        );
        raw.push((q, constant, lower));
    }
    let global: Vec<_> = checked.g.iter().map(|&v| I::point(v).unwrap()).collect();
    let build = |checked_constant: f64| {
        let mut rank = core::Ranker::new(&p.lower, &p.upper, &[1], 1).unwrap();
        for (mask, (q, _, lower)) in raw.iter().enumerate() {
            rank.push(0, mask, q, *lower, &mut || true).unwrap();
        }
        let axes = rank
            .finish(checked_constant, p.d, &checked.g, &mut || true)
            .unwrap();
        assert_eq!(axes.selected(), &[0]);
        let q = quote(p.input_dim, 1, 1, 1, 1, 2).unwrap();
        let mut capture = Capture::new(q).unwrap();
        // Deliberately non-optimal but legal inactive beta=2. This isolates the
        // numeric offset contract; it does NOT claim production Ray chose it.
        capture.push(0, 0, Some((0, 2.)));
        capture.push(0, 1, None);
        let mut alternate = Builder::new(capture.finish().unwrap(), &axes, q).unwrap();
        let mut original = core::Collector::new(axes).unwrap();
        for (mask, (coeff, constant, lower)) in raw.iter().enumerate() {
            original.push(0, mask, coeff, *lower, &mut || true).unwrap();
            alternate
                .push(&p, c, 0, mask, coeff, *constant, *lower, &mut || true)
                .unwrap();
        }
        (
            original
                .finish(checked_constant, p.d, &mut || true)
                .unwrap(),
            alternate.finish().unwrap(),
        )
    };
    let (data, alternate) = build(checked.constant);
    let old = data
        .lower(&global, I::zero(), &[0], &[0], &mut || true)
        .unwrap();
    let new = data
        .replacement_root(&global, alternate.view(), &mut || true)
        .unwrap();
    let exact_source_witness = rat(p.d) + rat(p.c[0]); // x=1, positive head=0.
    assert!(new < -0.74 && new > -0.76);
    assert!(rat(new) <= exact_source_witness && rat(old) <= exact_source_witness);

    // In the literal graph, x=1 is actually unsafe. The original encoder,
    // including its unsafe output row, admits this exact graph point.
    let original = f.original();
    let heads = binding::head_columns(&f.encoding(), &original).unwrap();
    let mut unsafe_graph = vec![Q::zero(); original.num_cols];
    unsafe_graph[original.input_vars[0].0] = rat(1.);
    unsafe_graph[p.input_dim] = rat(1.);
    unsafe_graph[heads[0].0] = rat(1.);
    unsafe_graph[original.binary_vars[0].0] = rat(1.);
    unsafe_graph[original.output_vars[0].0] = rat(-0.75);
    assert!(satisfies(
        &original,
        &unsafe_graph,
        original.problem.num_rows()
    ));

    // Counterfactual mutation only: replacing the raw minimum -1 with the
    // penalized minimum -2 would refund one to the offset and falsely prove
    // this unsafe box. This deliberately invalid numeric input has no replay,
    // owner, seal or publication authority; the actual Data above kept raw L.
    let raw_min = raw
        .iter()
        .map(|(_, _, lo)| *lo)
        .fold(f64::INFINITY, f64::min);
    let mut penalized = raw[0].1;
    penalized = penalized
        .add(I::point(2.).unwrap().scale(p.b1[0]).unwrap())
        .unwrap();
    let mut adjusted_min = penalized.lo;
    for (i, qi) in raw[0].0.iter().enumerate() {
        let adjusted = qi
            .add(I::point(2.).unwrap().scale(p.w_row(0)[i]).unwrap())
            .unwrap();
        adjusted_min = finite(add_down(
            adjusted_min,
            adjusted.minimum(p.lower[i], p.upper[i]).unwrap(),
        ))
        .unwrap();
    }
    assert!(adjusted_min < raw_min - 0.99);
    let wrong_constant = checked.constant + (raw_min - adjusted_min);
    let (wrong, same_alternate) = build(wrong_constant);
    let false_positive = wrong
        .replacement_root(&global, same_alternate.view(), &mut || true)
        .unwrap();
    assert!(false_positive > 0.24);
    assert!(rat(false_positive) > exact_source_witness);
}

fn protected_holds(b: &Budget) -> [usize; 8] {
    let c = b.state.lock().unwrap();
    let g = c.corner.grant.as_ref().unwrap();
    [
        b.admitted_bytes,
        c.reserved,
        c.component_reserved,
        c.prepaid_collection_work,
        c.prepaid_bind_work,
        g.prices,
        g.base_calls,
        g.base_call_cap,
    ]
}

fn root_ray_admission_edge(work_edge: bool, axes: usize, excess: usize) {
    let f = Fixture::separator();
    let p = f.source();
    let cuts = paid_owner(&p, &f, false);
    let row = cuts.signed.as_ref().unwrap();
    let b = &row.budget;
    let protected = protected_holds(b);
    let (parent_axes, bits) = {
        let c = b.state.lock().unwrap();
        (
            c.corner.grant.as_ref().unwrap().quote.k,
            phase::cube::paid_ray_bits(&c, b).unwrap(),
        )
    };
    assert_eq!((parent_axes, bits), (7, 1));
    let q = quote(Fixture::D, bits, axes, parent_axes, 2, 4).unwrap();
    let before = {
        let mut c = b.state.lock().unwrap();
        if work_edge {
            let target = MAX_WORK - c.reserved - c.component_reserved - q.work + excess;
            assert!(target >= c.work);
            c.work = target; // Explicit synthetic consumption, never extra credit.
        } else {
            let total = b.admitted_bytes
                + c.shared_bytes
                + c.direction.cache_bytes
                + c.component_rows.admitted_bytes
                + c.child.cache_bytes;
            let target = MAX_BYTES - q.bytes + excess;
            assert!(target >= total);
            c.component_rows.admitted_bytes += target - total;
        }
        (c.work, c.child.work_units, c.child.cache_bytes)
    };
    let expected = if excess == 0 {
        Some(axes)
    } else if axes > 1 {
        Some(axes - 1)
    } else {
        None
    };
    assert_eq!(prepay(&cuts, &f.plan(), b).is_some(), expected.is_some());
    assert_eq!(protected_holds(b), protected);
    let after = {
        let c = b.state.lock().unwrap();
        assert!(c.corner.ray_attempted);
        assert_eq!(
            c.corner
                .grant
                .as_ref()
                .unwrap()
                .root_ray
                .as_ref()
                .map(|g| g.quote.axes),
            expected
        );
        let (work, bytes) = expected.map_or((SETUP, 0), |k| {
            let q = quote(Fixture::D, bits, k, parent_axes, 2, 4).unwrap();
            (q.work, q.bytes)
        });
        assert_eq!(
            (c.work, c.child.work_units, c.child.cache_bytes),
            (before.0 + work, before.1 + work, before.2 + bytes)
        );
        if excess == 0 {
            assert_eq!(
                if work_edge {
                    c.work + c.reserved + c.component_reserved
                } else {
                    b.admitted_bytes
                        + c.shared_bytes
                        + c.direction.cache_bytes
                        + c.component_rows.admitted_bytes
                        + c.child.cache_bytes
                },
                if work_edge { MAX_WORK } else { MAX_BYTES }
            );
        }
        (c.work, c.child.work_units, c.child.cache_bytes)
    };
    assert!(prepay(&cuts, &f.plan(), b).is_none());
    {
        let mut c = b.state.lock().unwrap();
        assert_eq!((c.work, c.child.work_units, c.child.cache_bytes), after);
        c.work = MAX_WORK - c.reserved - c.component_reserved;
    }
    // Even after ordinary work is exhausted, both ORIGINAL paid replays and
    // the original cache bind remain available. All-mode refusal falls back.
    assert!(row
        .replay(&cuts, end(), &mut || true)
        .unwrap()
        .cache
        .is_none());
    let cache = row
        .replay_with_cache(&cuts, end(), &mut || true)
        .unwrap()
        .cache
        .unwrap();
    let bound = cache.bind(&cuts, end(), &mut || true).unwrap();
    let root = bound.root_lower().unwrap();
    if expected.is_some() {
        assert!(root > 0.24 && rat(root) <= rat(0.25));
    } else {
        assert!(root > -0.76 && root < -0.74);
    }
    assert_eq!(row.stats().source_replays, 2);
    assert_eq!(cuts.proposal_lp_calls, 1);
    assert_eq!(native_publication::funding(b).funded_calls, 1);
}

#[test]
fn root_ray_paid_work_exact_plus_one_fallback_and_all_mode_refusal_keep_replays() {
    for axes in [7, 1] {
        for excess in [0, 1] {
            root_ray_admission_edge(true, axes, excess);
        }
    }
}

#[test]
fn root_ray_paid_bytes_exact_plus_one_fallback_and_all_mode_refusal_keep_replays() {
    for axes in [7, 1] {
        for excess in [0, 1] {
            root_ray_admission_edge(false, axes, excess);
        }
    }
}

#[test]
fn root_ray_quote_includes_original_hooks_and_separate_complete_root_traversal() {
    let q = quote(30, 8, 6, 7, 16, 1308).unwrap();
    assert_eq!((q.work, q.bytes), (1_734_656, 16_432));
    assert!(quote(30, 8, 8, 7, 16, 1308).is_none());
    assert!(quote(30, 8, 6, 7, 16, 255).is_none());
}
