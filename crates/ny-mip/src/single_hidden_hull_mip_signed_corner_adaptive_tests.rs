// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Self-contained genuine Source and paid two-replay adaptive-cover controls.
//! The explicit checked Plan is a controlled fixture, not automatic-plan superiority.
use super::super as corners;
use super::super::super::tests::{end, rat};
use super::super::super::*;
use super::*;
use crate::single_hidden_hull::bab::{HullBabOptions, HullBabStop};
use crate::single_hidden_hull::HullLpStatus;
use crate::MipParts;
use num_rational::BigRational as Q;
use num_traits::Zero;
use ny_core::Bound;

struct Fixture {
    blocks: usize,
    weights: Vec<Vec<f64>>,
    biases: Vec<Vec<f64>>,
    dimensions: Vec<usize>,
    bounds: Vec<Vec<Bound>>,
}
impl Fixture {
    const D: usize = 16;
    fn new(blocks: usize, extra: f64) -> Self {
        assert!((1..=4).contains(&blocks));
        let h = 6 * blocks;
        let mut first = vec![0.; h * Self::D];
        let mut bias = Vec::new();
        let mut output = vec![0.; 2 * h];
        for block in 0..blocks {
            for (slot, [x, y]) in [
                [1., 1.],
                [-1., -1.],
                [1., -1.],
                [-1., 1.],
                [1., 1.],
                [-1., -1.],
            ]
            .into_iter()
            .enumerate()
            {
                let head = 6 * block + slot;
                first[head * Self::D + 2 * block] = x;
                first[head * Self::D + 2 * block + 1] = y;
                output[head] = if slot < 4 { -1. } else { -0.125 };
            }
            bias.extend([
                1. / 16.,
                -1. / 16.,
                -1. / 16.,
                1. / 16.,
                -1. / 16.,
                1. / 16.,
            ]);
        }
        let hidden = bias
            .iter()
            .map(|&b| Bound::new((b - 2.) as f32, (b + 2.) as f32))
            .collect();
        Self {
            blocks,
            weights: vec![first, output],
            biases: vec![bias, vec![2.5 * blocks as f64 + extra, 0.]],
            dimensions: vec![Self::D, h, 2],
            bounds: vec![hidden],
        }
    }
    fn source(&self) -> HullMarginProblem {
        HullMarginProblem::new(
            self.weights[0].clone(),
            self.biases[0].clone(),
            Self::D,
            &self.weights[1],
            &self.biases[1],
            2,
            &[(0, 1.), (1, -1.)],
            0.,
            vec![-1.; Self::D],
            vec![1.; Self::D],
        )
        .unwrap()
    }
    fn plan(&self) -> Plan {
        Plan {
            components: (0..3 * self.blocks)
                .map(|c| Component {
                    negative: vec![2 * c, 2 * c + 1],
                    positive: vec![],
                    positive_units: vec![],
                    g: vec![0.; Self::D],
                    alpha: vec![],
                })
                .collect(),
            leftover: vec![],
        }
    }
    fn original(&self) -> MipParts {
        let mut e = crate::encode_feedforward(
            &self.weights,
            &self.biases,
            &self.dimensions,
            &[Bound::new(-1., 1.); Self::D],
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
#[derive(Clone, Copy)]
enum Mode {
    None,
    Fixed,
    Adaptive,
}
fn owner<'p>(p: &'p HullMarginProblem, f: &Fixture, mode: Mode) -> HullRootMipCuts<'p> {
    let until = end();
    let mut attempt = primal_capture::Attempt::new_with_native_counts(p, until);
    let mut cuts = propose_root_mip_cuts_inner(p, until, Some(&mut attempt)).unwrap();
    assert_eq!(cuts.proposal_lp_calls, 1);
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
    match mode {
        Mode::None => {}
        Mode::Fixed => corners::prepay(&cuts, &plan, &b).unwrap(),
        Mode::Adaptive => corners::prepay_adaptive(&cuts, &plan, &b).unwrap(),
    }
    assert_eq!(native_publication::funding(&b).funded_calls, 1);
    let seal = Arc::new(signature(&cuts).collect());
    cuts.signed = Some(row(&cuts, &plan, checked, cost, &b, &seal));
    cuts
}
fn complete(cuts: &HullRootMipCuts<'_>) -> Arc<child::Cache> {
    let row = cuts.signed.as_ref().unwrap();
    assert!(row
        .replay(cuts, end(), &mut || true)
        .unwrap()
        .cache
        .is_none());
    let second = row.replay_with_cache(cuts, end(), &mut || true).unwrap();
    assert_eq!(row.stats().source_replays, 2);
    assert_eq!(second.root_lower.to_bits(), row.lower.to_bits());
    second.cache.unwrap()
}
fn holds(b: &Budget) -> (usize, usize, usize, usize) {
    let c = b.state.lock().unwrap();
    (
        c.reserved,
        c.component_reserved,
        c.prepaid_collection_work,
        c.prepaid_bind_work,
    )
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
// These original Source affines are independently formed for
// low-level kernel controls. Full owner tests below use certificate.rs itself.
fn raw_rows(p: &HullMarginProblem, plan: &Plan) -> (Vec<Vec<(Vec<I>, f64)>>, f64) {
    let mut rows = Vec::new();
    let mut checked_constant = 0.;
    for component in &plan.components {
        assert!(component.g.iter().all(|&g| g == 0.));
        let mut masks = Vec::new();
        let mut minimum = f64::INFINITY;
        for mask in 0..1usize << component.negative.len() {
            let mut q = vec![I::zero(); p.input_dim];
            let mut b = I::zero();
            for (slot, &j) in component.positive.iter().enumerate() {
                let a = component.alpha[mask * component.positive.len() + slot];
                let k = component.coefficient(p, slot).unwrap().scale(a).unwrap();
                b = b.add(k.scale(p.b1[j]).unwrap()).unwrap();
                for i in 0..p.input_dim {
                    q[i] = q[i].add(k.scale(p.w_row(j)[i]).unwrap()).unwrap();
                }
            }
            for (bit, &j) in component.negative.iter().enumerate() {
                if mask & (1 << bit) != 0 {
                    let k = I::point(p.c[j]).unwrap();
                    b = b.add(k.scale(p.b1[j]).unwrap()).unwrap();
                    for i in 0..p.input_dim {
                        q[i] = q[i].add(k.scale(p.w_row(j)[i]).unwrap()).unwrap();
                    }
                }
            }
            let mut lower = b.lo;
            for i in 0..p.input_dim {
                lower = add_down(lower, q[i].minimum(p.lower[i], p.upper[i]).unwrap());
            }
            minimum = minimum.min(lower);
            masks.push((q, lower));
        }
        checked_constant = add_down(checked_constant, minimum);
        rows.push(masks);
    }
    (rows, checked_constant)
}
fn captured(
    f: &Fixture,
    k: usize,
    poll: &mut dyn FnMut() -> bool,
) -> Option<(Cover, Vec<Vec<(Vec<I>, f64)>>, f64, f64)> {
    let p = f.source();
    let plan = f.plan();
    let bits = vec![2; plan.components.len()];
    let q = quote(Fixture::D, &bits, k)?;
    let (rows, constant) = raw_rows(&p, &plan);
    let mut capture = Capture::new(&p.lower, &p.upper, &bits, q)?;
    for (c, masks) in rows.iter().enumerate() {
        for (m, (coeff, raw)) in masks.iter().enumerate() {
            capture.push(c, m, coeff, *raw, poll)?;
        }
    }
    let cover = capture.finish(constant, p.d, &[0.; Fixture::D], poll)?;
    Some((cover, rows, constant, p.d))
}
fn kernel(f: &Fixture, k: usize, poll: &mut dyn FnMut() -> bool) -> Option<CheckedRoot> {
    let (cover, rows, constant, d) = captured(f, k, poll)?;
    let mut collector = Collector::new(cover)?;
    for (c, masks) in rows.iter().enumerate() {
        for (m, (coeff, raw)) in masks.iter().enumerate() {
            collector.push(c, m, coeff, *raw, poll)?;
        }
    }
    collector.finish(constant, d, &[0.; Fixture::D], poll)
}

#[test]
fn adaptive_original_encoder_and_exact_concave_vertex_oracle() {
    let f = Fixture::new(4, 1.);
    let parts = f.original();
    let h = f.dimensions[1];
    let heads = binding::head_columns(&f.encoding(), &parts).unwrap();
    assert_eq!(parts.binary_vars.len(), h);
    let mut relaxed = vec![Q::zero(); parts.num_cols];
    for j in 0..h {
        relaxed[Fixture::D + j] = rat(f.biases[0][j]);
        relaxed[heads[j].0] = rat(1.) + rat(f.biases[0][j]) / rat(2.);
        relaxed[parts.binary_vars[j].0] = rat(0.5);
    }
    relaxed[parts.output_vars[0].0] = rat(-6.);
    assert!(
        satisfies(&parts, &relaxed, parts.problem.num_rows()),
        "actual unsafe LP relaxation point"
    );
    let mut minimum = None::<Q>;
    for vertex in 0..256 {
        let mut point = vec![Q::zero(); parts.num_cols];
        for i in 0..8 {
            point[parts.input_vars[i].0] = rat(if vertex & (1 << i) != 0 { 1. } else { -1. });
        }
        let mut output = rat(11.);
        for j in 0..h {
            let pre = rat(f.biases[0][j])
                + (0..8)
                    .map(|i| rat(f.weights[0][j * Fixture::D + i]) * &point[parts.input_vars[i].0])
                    .sum::<Q>();
            let value = pre.clone().max(Q::zero());
            point[Fixture::D + j] = pre.clone();
            point[heads[j].0] = value.clone();
            point[parts.binary_vars[j].0] = rat(if pre > Q::zero() { 1. } else { 0. });
            output += rat(f.weights[1][j]) * value;
        }
        point[parts.output_vars[0].0] = output.clone();
        assert!(satisfies(&parts, &point, parts.problem.num_rows() - 1));
        minimum = Some(minimum.map_or(output.clone(), |v| v.min(output)));
    }
    // All output coefficients are negative: the true objective is concave.
    // These are every vertex of its eight live inputs; dummy inputs are absent.
    assert_eq!(minimum.unwrap(), rat(49. / 32.));
}

#[test]
fn adaptive_genuine_two_replays_improve_fixed_seven_axis_root() {
    // Advice stops at the first positive complete cover. Closure is required;
    // attaining the independent exact network minimum 49/32 is not.
    let f = Fixture::new(4, 1.);
    let p = f.source();
    let mut old_tuple = None;
    for mode in [Mode::Fixed, Mode::Adaptive] {
        let cuts = owner(&p, &f, mode);
        let row = cuts.signed.as_ref().unwrap();
        let tuple = (
            row.lower.to_bits(),
            row.constant.to_bits(),
            row.g.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        );
        if let Some(old) = &old_tuple {
            assert_eq!(&tuple, old);
        } else {
            old_tuple = Some(tuple.clone());
        }
        let mut parts = f.original();
        let original = parts.problem.clone();
        cuts.append_to(&mut parts, &f.encoding(), end()).unwrap();
        assert_eq!(
            &parts.problem.rows()[..original.num_rows()],
            original.rows()
        );
        assert_eq!(parts.problem.cols(), original.cols());
        let _ = cuts.best_signed_root_proof(end());
        assert_eq!(row.stats().source_replays, 2);
        let cache = cuts.signed_child_cache.get().unwrap();
        let bound = cache.bind(&cuts, end(), &mut || true).unwrap();
        let lower = bound.root_lower().unwrap();
        match mode {
            Mode::Fixed => assert!(lower < -0.35 && lower > -0.37, "{lower}"),
            Mode::Adaptive => {
                assert!(lower > 0. && rat(lower) <= rat(49. / 32.), "{lower}");
                let report = cache.stats();
                assert!(report.corner_cover_admitted);
                assert!(report.corner_cover_leaves > 128 && report.corner_cover_leaves <= 192);
                assert!(report.corner_cover_root_lower.unwrap() > 0.);
            }
            Mode::None => unreachable!(),
        }
        assert_eq!(
            tuple,
            (
                row.lower.to_bits(),
                row.constant.to_bits(),
                row.g.iter().map(|v| v.to_bits()).collect()
            )
        );
        assert_eq!(cuts.proposal_lp_calls, 1);
        assert_eq!(cuts.native_funding_report().funded_calls, 1);
    }
}

#[test]
fn adaptive_genuine_public_continuation_closes_without_native_lp() {
    let f = Fixture::new(4, 1.);
    let p = f.source();
    let cuts = owner(&p, &f, Mode::Adaptive);
    assert!(cuts.signed.as_ref().unwrap().lower < 0. && cuts.checked_lower < 0.);
    let mut parts = f.original();
    cuts.append_to(&mut parts, &f.encoding(), end()).unwrap();
    let _ = cuts.best_signed_root_proof(end());
    let cache = cuts.signed_child_cache.get().unwrap();
    assert_eq!(cache.stats().bind_attempts, 0);
    let continuation = cuts
        .continue_hull_bab_with_native_rows(
            end(),
            HullBabOptions {
                workers: 1,
                max_nodes: 1,
            },
        )
        .unwrap();
    let report = continuation.report();
    assert!(continuation.verified(), "{report:?}");
    assert_eq!(report.stop, HullBabStop::Proved);
    assert_eq!((report.nodes, report.closed, report.open), (1, 1, 0));
    assert_eq!((report.root_lp_calls, report.proposal_lp_calls), (0, 0));
    assert_eq!(report.root_lp_status, HullLpStatus::Skipped);
    assert!(report.root_before_native.unwrap() > 0.);
    assert!(report.root_certified.is_some_and(|v| v <= 0.));
    assert_eq!(
        (
            cuts.native_rows_report().attempted,
            cuts.native_rows_report().committed
        ),
        (0, 0)
    );
}

#[test]
fn adaptive_core_complete_cover_rejects_missing_duplicate_and_repeated_axis() {
    assert!(fixed_scratch_bytes() <= 8192);
    assert!(size_of::<Node>() <= 8 && size_of::<Face>() <= 32);
    let f = Fixture::new(1, 0.);
    let good = kernel(&f, 1, &mut || true).unwrap();
    let p = f.source();
    for field in 0..9 {
        let mut bad_quote = quote(Fixture::D, &[2, 2, 2], 1).unwrap();
        match field {
            0 => bad_quote.bytes += 1,
            1 => bad_quote.work += 1,
            2 => bad_quote.max_splits += 1,
            3 => bad_quote.max_depth -= 1,
            4 => bad_quote.advice_work -= 1,
            5 => bad_quote.d -= 1,
            6 => bad_quote.k += 1,
            7 => bad_quote.masks += 1,
            8 => bad_quote.components += 1,
            _ => unreachable!(),
        }
        assert!(
            Capture::new(&p.lower, &p.upper, &[2, 2, 2], bad_quote).is_none(),
            "quote field {field}"
        );
    }
    assert!(good.lower() > 0.13 && good.lower() <= 17. / 128.);
    assert!(good.splits() > 0);
    for corrupt in 0..7 {
        let (mut cover, _, _, _) = captured(&f, 1, &mut || true).unwrap();
        let branch = cover.nodes.iter().position(|n| n.axis != LEAF).unwrap();
        match corrupt {
            0 => {
                cover.nodes.pop();
            }
            1 => cover.nodes[branch].low = cover.nodes[branch].high,
            2 => cover.nodes[branch].axis = cover.axes[0] as u8,
            3 => cover.axes[0] = Fixture::D,
            4 => cover.leaves -= 1,
            5 => cover.advice_spent = cover.q.advice_work + 1,
            6 => cover.q.work += 1,
            _ => unreachable!(),
        }
        assert!(Collector::new(cover).is_none(), "corruption {corrupt}");
    }
}

#[test]
fn adaptive_core_complete_source_coverage_and_sticky_cancel_are_required() {
    let f = Fixture::new(1, 0.);
    let mut polls = 0;
    assert!(kernel(&f, 1, &mut || {
        polls += 1;
        true
    })
    .is_some());
    assert!(polls > 10);
    for cutoff in 1..=polls {
        let mut count = 0;
        assert!(
            kernel(&f, 1, &mut || {
                count += 1;
                count != cutoff
            })
            .is_none(),
            "transient false at {cutoff}/{polls}"
        );
    }
    for duplicate in [false, true] {
        let (cover, rows, constant, d) = captured(&f, 1, &mut || true).unwrap();
        let mut collector = Collector::new(cover).unwrap();
        let (coeff, raw) = &rows[0][0];
        collector.push(0, 0, coeff, *raw, &mut || true).unwrap();
        if duplicate {
            assert!(collector.push(0, 0, coeff, *raw, &mut || true).is_none());
        }
        assert!(collector
            .finish(constant, d, &[0.; Fixture::D], &mut || true)
            .is_none());
    }
}

#[test]
fn adaptive_original_raw_offset_cannot_be_refunded_into_false_positive() {
    // Actual Source F(x)=1/4-ReLU(x) is unsafe at x=1, where F=-3/4.
    let p = HullMarginProblem::new(
        vec![1.],
        vec![0.],
        1,
        &[-1., 0.],
        &[0.25, 0.],
        2,
        &[(0, 1.), (1, -1.)],
        0.,
        vec![-1.],
        vec![1.],
    )
    .unwrap();
    let plan = Plan {
        components: vec![Component {
            negative: vec![0],
            positive: vec![],
            positive_units: vec![],
            g: vec![0.],
            alpha: vec![],
        }],
        leftover: vec![],
    };
    let (rows, constant) = raw_rows(&p, &plan);
    let q = quote(1, &[1], 1).unwrap();
    let run = |checked_constant: f64| {
        let mut capture = Capture::new(&p.lower, &p.upper, &[1], q).unwrap();
        for (m, (c, v)) in rows[0].iter().enumerate() {
            capture.push(0, m, c, *v, &mut || true).unwrap();
        }
        let cover = capture.finish(constant, p.d, &[0.], &mut || true).unwrap();
        let mut collector = Collector::new(cover).unwrap();
        for (m, (c, v)) in rows[0].iter().enumerate() {
            collector.push(0, m, c, *v, &mut || true).unwrap();
        }
        collector
            .finish(checked_constant, p.d, &[0.], &mut || true)
            .unwrap()
            .lower()
    };
    let actual = run(constant);
    assert!(actual <= -0.75 && actual > -0.751);
    // Deliberately invalid standalone input demonstrates the failure this
    // Source-authenticated original-constant contract must prevent. It is not
    // an owner/certificate capability and cannot enter production publication.
    assert!(run(0.) > 0.24);
}

fn full_quote(f: &Fixture) -> (usize, usize) {
    let bits = vec![2; 3 * f.blocks];
    let q = quote(Fixture::D, &bits, 7).unwrap();
    let cap = corners::base_call_cap(Fixture::D, f.dimensions[1], q.masks, bits.len()).unwrap();
    (
        corners::ADAPTIVE_SETUP_WORK
            + q.work
            + 16_384
            + 32 * q.masks
            + 128 * bits.len()
            + corners::ADAPTIVE_BASE_CALLS * cap,
        q.bytes + corners::ADAPTIVE_WRAPPER_BYTES,
    )
}
fn admission_edge(bytes: bool, excess: usize) {
    let f = Fixture::new(4, 1.);
    let p = f.source();
    let cuts = owner(&p, &f, Mode::None);
    let b = &cuts.signed.as_ref().unwrap().budget;
    let prior_holds = holds(b);
    let base = b.admitted_bytes;
    let (work, payload) = full_quote(&f);
    let before = {
        let mut c = b.state.lock().unwrap();
        if bytes {
            let total = base
                + c.shared_bytes
                + c.direction.cache_bytes
                + c.component_rows.admitted_bytes
                + c.child.cache_bytes;
            let target = MAX_BYTES - payload + excess;
            assert!(target >= total);
            c.component_rows.admitted_bytes += target - total;
        } else {
            let target = MAX_WORK - c.reserved - c.component_reserved - work + excess;
            assert!(target >= c.work);
            c.work = target;
        }
        (c.work, c.child.work_units, c.child.cache_bytes)
    };
    assert_eq!(
        corners::prepay_adaptive(&cuts, &f.plan(), b).is_some(),
        excess == 0
    );
    assert_eq!(holds(b), prior_holds);
    assert_eq!(b.admitted_bytes, base);
    let after = {
        let c = b.state.lock().unwrap();
        let paid = if excess == 0 {
            work
        } else {
            corners::ADAPTIVE_SETUP_WORK
        };
        assert_eq!(
            (c.work, c.child.work_units),
            (before.0 + paid, before.1 + paid)
        );
        assert_eq!(
            c.child.cache_bytes,
            before.2 + if excess == 0 { payload } else { 0 }
        );
        assert_eq!(c.corner.adaptive.is_some(), excess == 0);
        if excess == 0 {
            assert_eq!(
                if bytes {
                    base + c.shared_bytes
                        + c.direction.cache_bytes
                        + c.component_rows.admitted_bytes
                        + c.child.cache_bytes
                } else {
                    c.work + c.reserved + c.component_reserved
                },
                if bytes { MAX_BYTES } else { MAX_WORK }
            );
        }
        (c.work, c.child.work_units, c.child.cache_bytes)
    };
    assert!(corners::prepay_adaptive(&cuts, &f.plan(), b).is_none());
    {
        let c = b.state.lock().unwrap();
        assert_eq!((c.work, c.child.work_units, c.child.cache_bytes), after);
    }
    // A refused adaptive mode leaves the original fixed-corner path available.
    // Work-exact admission also proves both paid replays and ordinary bind
    // remain reachable with no unreserved work left.
    let fixed = excess != 0 && corners::prepay(&cuts, &f.plan(), b).is_some();
    let cache = complete(&cuts);
    let bound = cache.bind(&cuts, end(), &mut || true).unwrap();
    let lower = bound.root_lower().unwrap();
    if excess == 0 {
        assert!(lower > 0.);
    } else {
        assert!(lower < 0. && lower >= cuts.signed.as_ref().unwrap().lower);
        if fixed {
            // The old policy may choose a shallower paid K at this boundary;
            // preserving its fallback does not promise the K7 fixture floor.
            let c = b.state.lock().unwrap();
            let k = c.corner.grant.as_ref().unwrap().quote.k;
            assert!((1..=7).contains(&k));
        }
    }
}

#[test]
fn adaptive_genuine_admission_exact_work_and_one_unit_refusal_preserve_holds() {
    admission_edge(false, 0);
    admission_edge(false, 1);
}
#[test]
fn adaptive_genuine_admission_exact_bytes_and_one_byte_refusal_preserve_holds() {
    admission_edge(true, 0);
    admission_edge(true, 1);
}

#[test]
fn adaptive_four_paid_ordinary_calls_and_one_bind_share_owner_credit() {
    let f = Fixture::new(4, 1.);
    let p = f.source();
    let cuts = owner(&p, &f, Mode::Adaptive);
    let cache = complete(&cuts);
    let b = &cuts.signed.as_ref().unwrap().budget;
    {
        let mut c = b.state.lock().unwrap();
        c.work = MAX_WORK - c.reserved - c.component_reserved;
    }
    let bound = cache.bind(&cuts, end(), &mut || true).unwrap();
    assert!(bound.root_lower().unwrap() > 0.);
    for _ in 0..corners::ADAPTIVE_BASE_CALLS {
        assert!(bound.lower_zero(&[], end(), &mut || true).unwrap() > 0.);
    }
    assert!(bound.lower_zero(&[], end(), &mut || true).is_none());
    assert!(bound.root_lower().unwrap() > 0.);
    let c = b.state.lock().unwrap();
    let grant = c.corner.adaptive.as_ref().unwrap();
    assert_eq!(grant.base_calls, 0);
    assert!(!grant.bind);
    drop(c);
    // A second bind has neither the ordinary prepaid bind nor fresh work.
    assert!(cache.bind(&cuts, end(), &mut || true).is_none());
}

#[test]
fn adaptive_positive_relu_endpoints_do_not_certify_unsafe_interior() {
    // F(x)=ReLU(x)+ReLU(-x)-1/2 has positive endpoint values but F(0)=-1/2.
    // A zero-preactivation negative dummy gives a legal nonempty component;
    // it contributes exactly zero to the actual original network everywhere.
    let p = HullMarginProblem::new(
        vec![1., -1., 0.],
        vec![0.; 3],
        1,
        &[1., 1., -1., 0., 0., 0.],
        &[-0.5, 0.],
        2,
        &[(0, 1.), (1, -1.)],
        0.,
        vec![-1.],
        vec![1.],
    )
    .unwrap();
    let plan = Plan {
        components: vec![Component {
            negative: vec![2],
            positive: vec![0, 1],
            positive_units: vec![2, 2],
            g: vec![0.],
            alpha: vec![0.5; 4],
        }],
        leftover: vec![],
    };
    let (rows, constant) = raw_rows(&p, &plan);
    let q = quote(1, &[1], 1).unwrap();
    let mut capture = Capture::new(&p.lower, &p.upper, &[1], q).unwrap();
    for (m, (c, v)) in rows[0].iter().enumerate() {
        capture.push(0, m, c, *v, &mut || true).unwrap();
    }
    let cover = capture.finish(constant, p.d, &[0.], &mut || true).unwrap();
    let mut collector = Collector::new(cover).unwrap();
    for (m, (c, v)) in rows[0].iter().enumerate() {
        collector.push(0, m, c, *v, &mut || true).unwrap();
    }
    let lower = collector
        .finish(constant, p.d, &[0.], &mut || true)
        .unwrap()
        .lower();
    assert!(lower <= -0.5 && lower > -0.501);
    assert_eq!(rat(1.) + rat(0.) - rat(0.5), rat(0.5));
    assert_eq!(rat(0.) + rat(0.) - rat(0.5), rat(-0.5));
}

#[test]
fn adaptive_foreign_owner_or_box_advice_retains_original_cache_fallback() {
    for change_owner in [false, true] {
        let f = Fixture::new(4, 1.);
        let p = f.source();
        let mut cuts = owner(&p, &f, Mode::Adaptive);
        cuts.signed
            .as_ref()
            .unwrap()
            .replay(&cuts, end(), &mut || true)
            .unwrap();
        if change_owner {
            let row = cuts.signed.as_mut().unwrap();
            row.witness = Arc::new(row.witness.as_ref().clone());
        } else {
            let b = &cuts.signed.as_ref().unwrap().budget;
            let mut c = b.state.lock().unwrap();
            // Deliberately perturb advisory box metadata, leaving the actual
            // Source, signature and old certificate byte-for-byte untouched.
            c.corner
                .adaptive
                .as_mut()
                .unwrap()
                .cover
                .as_mut()
                .unwrap()
                .bounds[0][0] = -0.5;
        }
        let row = cuts.signed.as_ref().unwrap();
        let cache = row
            .replay_with_cache(&cuts, end(), &mut || true)
            .unwrap()
            .cache
            .unwrap();
        let bound = cache.bind(&cuts, end(), &mut || true).unwrap();
        assert_eq!(bound.root_lower().unwrap().to_bits(), row.lower.to_bits());
        assert!(cache.stats().corner_cover_root_lower.is_none());
    }
}
