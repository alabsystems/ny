// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Five independent factors distinguish retained feedback from one four-cut pool.
use super::*;
use ny_core::head_phase_advice::{HeadPhaseLpHint, HeadPhaseLpRequest};
use std::sync::atomic::{AtomicUsize, Ordering};

pub(super) fn end() -> Instant {
    Instant::now() + Duration::from_secs(10)
}

pub(super) fn budget() -> Budget {
    Budget {
        deadline: end(),
        cells: 0,
        masks: 0,
    }
}

fn input(at: usize, permuted: bool) -> usize {
    if permuted {
        (12 * at + 11) % 35
    } else {
        at
    }
}

fn head(block: usize, edge: usize, permuted: bool) -> usize {
    if permuted {
        ((3 * block + 2) % 5) * 7 + (3 * edge + 1) % 7
    } else {
        7 * block + edge
    }
}

fn spec(bias: f32, permuted: bool) -> InputHeadSpec {
    let mut first = vec![0.0; 35 * 35];
    for block in 0..5 {
        for edge in 0..7 {
            let h = head(block, edge, permuted);
            first[input(7 * block + edge, permuted) * 35 + h] = 1.0;
            first[input(7 * block + (edge + 1) % 7, permuted) * 35 + h] = -1.0;
        }
    }
    InputHeadSpec::new(
        35,
        35,
        first,
        vec![0.0; 35],
        (0..35).flat_map(|_| [-1.0, 0.0]).collect(),
        vec![bias, 0.0],
        [101; 32],
    )
    .unwrap()
}

fn source(bias: f32, permuted: bool) -> Source {
    Source::build(&spec(bias, permuted), &[-1.0; 35], &[1.0; 35], 0, 1, end()).unwrap()
}

fn direction(block: usize, permuted: bool) -> Direction {
    let mut heads: Vec<_> = (0..7)
        .map(|edge| (head(block, edge, permuted), 1.0))
        .collect();
    heads.sort_by_key(|&(h, _)| h);
    Direction { heads }
}

fn cuts(
    source: &Source,
    blocks: std::ops::Range<usize>,
    permuted: bool,
) -> Vec<CertifiedDirection<'_>> {
    let mut work = budget();
    blocks
        .map(|block| certify(source, direction(block, permuted), &mut work).unwrap())
        .collect()
}

/// Independent multiplier proposal: first price retained geometry, then every
/// still-negative head through its mandatory scalar upper row. Source alone
/// decides whether these ordinary nonnegative multipliers prove anything.
pub(super) fn reference_hint(
    request: &HeadFunctionalLpRequest,
    scalar_rows: usize,
) -> HeadFunctionalLpHint {
    let mut residual = request.objective.clone();
    let mut duals = vec![0.0; request.rows.len()];
    let mut apply = |at: usize, mass: f64, residual: &mut Vec<f64>| {
        duals[at] = mass;
        for &(j, a) in &request.rows[at].coefficients {
            residual[j] -= mass * a;
        }
    };
    for at in scalar_rows..request.rows.len() {
        let mass = request.rows[at]
            .coefficients
            .iter()
            .filter(|&&(j, a)| j >= request.feature_count && a < 0.0)
            .map(|&(j, a)| (-residual[j]).max(0.0) / -a)
            .fold(f64::INFINITY, f64::min);
        if mass.is_finite() && mass > 0.0 {
            apply(at, mass, &mut residual);
        }
    }
    for (h, lower) in request.slope_rows.iter().enumerate() {
        let j = request.feature_count + h;
        if residual[j] >= 0.0 {
            continue;
        }
        let upper = lower.unwrap() + 1;
        if upper >= scalar_rows {
            continue;
        }
        if let Some(&(_, a)) = request.rows[upper]
            .coefficients
            .iter()
            .find(|&&(column, a)| column == j && a < 0.0)
        {
            apply(upper, residual[j] / a, &mut residual);
        }
    }
    let hint = HeadFunctionalLpHint { row_duals: duals };
    assert!(hint.validate(request));
    hint
}

#[test]
fn feedback_five_cycles_have_exact_factor_minimum_and_a_four_cut_gap() {
    // This exact factorization covers the continuous box: a negative sum of
    // ReLUs is concave, so each disjoint seven-input minimum is at a vertex.
    let maximum = (0..128)
        .map(|mask: usize| {
            (0..7)
                .map(|j| {
                    let x = if mask & (1 << j) == 0 { -1i32 } else { 1 };
                    let next = if mask & (1 << ((j + 1) % 7)) == 0 {
                        -1i32
                    } else {
                        1
                    };
                    (x - next).max(0)
                })
                .sum::<i32>()
        })
        .max()
        .unwrap();
    assert_eq!(maximum, 6);
    assert_eq!(61 - 2 * 5 * maximum, 1); // doubled genuine minimum +1/2
    assert_eq!(61 - 2 * (4 * maximum + 7), -1); // fifth scalar outer factor
    let s = source(30.5, false);
    assert_eq!(s.spec.inputs, 35); // genuine Source control, not a 30-input adapter
    let four = cuts(&s, 0..4, false);
    let model = triangle::Model::from_root_directions(&s, &mut None, &four, end()).unwrap();
    let floor = model
        .replay(&s, &reference_hint(model.request(), 70), 1.0, end())
        .unwrap();
    assert!(floor <= -0.5 && floor > -0.501, "four-cut floor {floor}");
    // A fifth independently certified row augments, rather than replaces, the
    // completed four rows and all seventy mandatory scalar triangles.
    let prior = model.request().rows.clone();
    let fifth = cuts(&s, 4..5, false);
    let model = triangle::Model::from_root_directions(
        &s,
        &mut Some(model.into_phase_prefix()),
        &fifth,
        end(),
    )
    .unwrap();
    assert_eq!(model.request().rows.len(), 75);
    for (a, b) in model.request().rows.iter().take(74).zip(prior) {
        assert_eq!(a.coefficients, b.coefficients);
        assert_eq!(a.lower.to_bits(), b.lower.to_bits());
    }
    let closed = model
        .replay(&s, &reference_hint(model.request(), 70), 1.0, end())
        .unwrap();
    assert!(closed > 0.49 && closed <= 0.5, "five-cut floor {closed}");
}

#[test]
fn feedback_geometry_debt_survives_zero_full_reduced_residual() {
    let s = source(30.5, false);
    let four = cuts(&s, 0..4, false);
    let model = triangle::Model::from_root_directions(&s, &mut None, &four, end()).unwrap();
    let hint = reference_hint(model.request(), 70);
    let mut full = model.request().objective.clone();
    for (row, &mass) in model.request().rows.iter().zip(&hint.row_duals) {
        for &(j, a) in &row.coefficients {
            full[j] -= mass * a;
        }
    }
    assert!(full.iter().all(|v| v.abs() < 1e-10), "{full:?}");
    let (debt, gradient) =
        geometry_debt(model.request(), 70, 70, &hint, 1.0, &mut budget()).unwrap();
    assert_eq!(debt.len(), 35);
    assert!(debt[..28].iter().all(|v| v.abs() < 1e-10));
    assert!(debt[28..].iter().all(|v| *v < -0.99 && *v >= -1.0000001));
    assert!(gradient.iter().all(|v| v.abs() < 1e-9));
    let negative = HeadFunctionalLpHint {
        row_duals: hint.row_duals.iter().map(|v| -v).collect(),
    };
    let mirrored = geometry_debt(model.request(), 70, 70, &negative, -1.0, &mut budget()).unwrap();
    assert_eq!(debt, mirrored.0);
    assert_eq!(gradient, mirrored.1);
}

#[test]
fn feedback_reopens_the_pool_and_retains_geometry_under_permutations() {
    for permuted in [false, true] {
        let s = source(30.5, permuted);
        let original_pool = proposals(&s, &mut budget()).unwrap();
        assert!(original_pool
            .iter()
            .all(|d| d.heads.iter().all(|&(h, _)| h < 32)));
        let (report, retained) =
            refine_with_prefix(&s, s.baseline(end()).unwrap(), None, end(), end(), &|| None);
        assert!(
            report.lower > 0.49 && report.lower <= 0.5,
            "{permuted}: {}",
            report.lower
        );
        assert!(report.rounds.len() >= 2 && report.rounds.len() <= 3);
        assert!(report.rounds[0].after <= -0.49, "one round already closed");
        assert!(report.cuts >= 5 && report.cuts <= 12);
        assert!(report.masks <= MAX_MASKS && report.work_cells <= MAX_CELLS);
        assert_eq!(report.calls, 0);
        let (_, rows) = retained.unwrap().into_phase_parts(&s, 1028, end()).unwrap();
        assert!(rows.len() >= 75);
        let geometric = &rows[70..];
        for block in 0..5 {
            let expected = direction(block, permuted).heads;
            assert!(geometric.iter().any(|row| row.coefficients.len() == 7
                && row
                    .coefficients
                    .iter()
                    .zip(&expected)
                    .all(|(&(j, a), &(h, _))| j == 35 + h && a.lo == -1.0 && a.hi == -1.0)));
        }
    }
}

struct BadAdvice {
    calls: AtomicUsize,
}
impl HeadPhaseAdvice for BadAdvice {
    fn propose(&self, _: &HeadPhaseLpRequest, _: Instant) -> Option<HeadPhaseLpHint> {
        None
    }
    fn supports_functional_advice(&self) -> bool {
        true
    }
    fn propose_functional(
        &self,
        _: &HeadFunctionalLpRequest,
        _: Instant,
    ) -> Option<HeadFunctionalLpHint> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Some(HeadFunctionalLpHint {
            row_duals: vec![f64::NAN],
        })
    }
}

#[test]
fn feedback_missing_or_bad_advice_keeps_native_progress_with_shared_caps() {
    let s = source(30.5, false);
    let bad = Arc::new(BadAdvice {
        calls: AtomicUsize::new(0),
    });
    let (r, _) = refine_with_prefix(&s, s.baseline(end()).unwrap(), None, end(), end(), &|| {
        Some(bad.clone())
    });
    assert!(r.lower > 0.49 && r.lower <= 0.5, "{}", r.lower);
    assert!(r.rounds.len() >= 2 && r.rounds.len() <= 3);
    assert_eq!(r.calls, bad.calls.load(Ordering::Relaxed));
    assert!(r.calls <= 3 && r.native_replayed <= 3);
    assert_eq!(r.ay_replayed, 0);
    assert!(r.masks <= MAX_MASKS && r.work_cells <= MAX_CELLS);
    assert!(r.rounds.iter().all(|x| x.cuts <= 4 && x.after >= x.before));
    assert_eq!(r.rounds.iter().map(|x| x.cuts).sum::<usize>(), r.cuts);
    assert_eq!(r.rounds.iter().map(|x| x.masks).sum::<usize>(), r.masks);
    assert_eq!(
        r.rounds.iter().map(|x| x.attempted).sum::<usize>(),
        r.attempted
    );
    assert!(r.attempted <= MAX_ATTEMPTS);
}

struct SignedAdvice {
    sign: f64,
    calls: AtomicUsize,
}
impl HeadPhaseAdvice for SignedAdvice {
    fn propose(&self, _: &HeadPhaseLpRequest, _: Instant) -> Option<HeadPhaseLpHint> {
        None
    }
    fn supports_functional_advice(&self) -> bool {
        true
    }
    fn propose_functional(
        &self,
        request: &HeadFunctionalLpRequest,
        _: Instant,
    ) -> Option<HeadFunctionalLpHint> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let mut hint = reference_hint(request, 70);
        for value in &mut hint.row_duals {
            *value *= self.sign;
        }
        Some(hint)
    }
}

#[test]
fn feedback_wrong_dual_sign_cannot_overwrite_the_useful_next_round_debt() {
    // Remain negative after the fifth cut so both rounds actually ask advice.
    // The unhelpful other global sign must not replace the useful native or
    // correctly signed geometry residual with the original top32 ranking.
    for sign in [1.0, -1.0] {
        let s = source(29.5, false);
        let advice = Arc::new(SignedAdvice {
            sign,
            calls: AtomicUsize::new(0),
        });
        let (r, _) =
            refine_with_prefix(&s, s.baseline(end()).unwrap(), None, end(), end(), &|| {
                Some(advice.clone())
            });
        assert!(
            r.lower <= -0.5 && r.lower > -0.501,
            "sign {sign}: {}",
            r.lower
        );
        assert!(r.cuts >= 5 && r.rounds.len() >= 2);
        assert!(r.calls >= 2 && r.calls <= 3);
        assert_eq!(r.calls, advice.calls.load(Ordering::Relaxed));
        assert!(r.ay_replayed >= 2 && r.ay_replayed <= 6);
    }
}

struct LateAdvice;
impl HeadPhaseAdvice for LateAdvice {
    fn propose(&self, _: &HeadPhaseLpRequest, _: Instant) -> Option<HeadPhaseLpHint> {
        None
    }
    fn supports_functional_advice(&self) -> bool {
        true
    }
    fn propose_functional(
        &self,
        request: &HeadFunctionalLpRequest,
        deadline: Instant,
    ) -> Option<HeadFunctionalLpHint> {
        let hint = reference_hint(request, 70);
        std::thread::sleep(
            deadline.saturating_duration_since(Instant::now()) + Duration::from_millis(1),
        );
        Some(hint)
    }
}

#[test]
fn feedback_late_advice_keeps_the_native_floor_and_completed_geometry() {
    let s = source(30.5, false);
    let before = s.baseline(end()).unwrap();
    let stage = Instant::now() + Duration::from_millis(200);
    let (r, prefix) = refine_with_prefix(&s, before, None, stage, end(), &|| {
        Some(Arc::new(LateAdvice))
    });
    assert_eq!(r.calls, 1);
    assert_eq!(r.ay_replayed, 0);
    assert!(r.native_replayed >= 1 && r.cuts >= 4);
    assert!(r.lower <= -0.5 && r.lower > -0.501, "{}", r.lower);
    assert_eq!(r.lower.to_bits(), r.rounds[0].after.to_bits());
    assert!(prefix.is_some());
}

struct ResetFaults;
impl Drop for ResetFaults {
    fn drop(&mut self) {
        EXPIRE_AFTER_MASK.with(|f| f.set(None));
        EXPIRE_AFTER_ROUND.with(|f| f.set(None));
    }
}

#[test]
fn feedback_partial_later_mask_keeps_the_completed_first_round() {
    let _reset = ResetFaults;
    let s = source(30.5, false);
    // Four complete128-mask factors; expiry on the last mask of the fifth
    // must not mint its row, even though its partial maximum was finite.
    EXPIRE_AFTER_MASK.with(|f| f.set(Some(640)));
    let (r, prefix) =
        refine_with_prefix(&s, s.baseline(end()).unwrap(), None, end(), end(), &|| None);
    assert_eq!(r.masks, 640);
    assert_eq!(r.cuts, 4);
    assert!(r.lower <= -0.5 && r.lower > -0.501, "{}", r.lower);
    assert_eq!(r.lower.to_bits(), r.rounds[0].after.to_bits());
    assert!(prefix.is_some());
}

#[test]
fn feedback_expiry_between_rounds_never_renews_authority_or_drops_completed_floor() {
    let _reset = ResetFaults;
    let s = source(30.5, false);
    EXPIRE_AFTER_ROUND.with(|f| f.set(Some(1)));
    let (r, prefix) =
        refine_with_prefix(&s, s.baseline(end()).unwrap(), None, end(), end(), &|| None);
    assert_eq!(r.cuts, 4);
    assert!(r.lower <= -0.5 && r.lower > -0.501, "{}", r.lower);
    assert_eq!(r.lower.to_bits(), r.rounds[0].after.to_bits());
    assert!(prefix.is_some());
    let past = Instant::now()
        .checked_sub(Duration::from_millis(1))
        .unwrap();
    let before = r.lower;
    let (expired, prefix) = refine_with_prefix(&s, before, prefix, past, end(), &|| None);
    assert_eq!(expired.lower.to_bits(), before.to_bits());
    assert_eq!(expired.calls, 0);
    assert!(prefix.is_some());
}

#[test]
fn feedback_unsafe_sibling_and_original_authority_expiry_never_authorize_unsat() {
    let model = spec(29.5, false);
    let result = verify_with_provider(
        &model,
        &[-1.0; 35],
        &[1.0; 35],
        0,
        1,
        HeadGroupPolicy::new(3.0, 2, 4).unwrap(),
        end(),
        || None,
    );
    let InputHeadOutcome::Unknown(report) = result else {
        panic!("unsafe five-cycle Source closed")
    };
    assert!(report.lower.unwrap() <= -0.5);
    let s = source(30.5, false);
    let before = s.baseline(end()).unwrap();
    let (expired, _) = refine_with_prefix(
        &s,
        before,
        None,
        end(),
        Instant::now()
            .checked_sub(Duration::from_millis(1))
            .unwrap(),
        &|| None,
    );
    assert_eq!(expired.lower.to_bits(), before.to_bits());
    assert_eq!(expired.calls, 0);
}

#[test]
fn feedback_cover_allocation_is_inside_the_one_original_root_authority() {
    let now = Instant::now();
    let root = now + Duration::from_millis(750);
    assert_eq!(directional_stage_deadline(now, root, false), root);
    let allocated = directional_stage_deadline(now, root, true);
    assert_eq!(allocated.duration_since(now), Duration::from_millis(450));
    assert_eq!(root.duration_since(allocated), Duration::from_millis(300));
    let expired = now.checked_sub(Duration::from_millis(1)).unwrap();
    assert_eq!(directional_stage_deadline(now, expired, true), expired);
    // A later invocation cannot use an expired allocation to start new work.
    let s = source(30.5, false);
    let floor = s.baseline(end()).unwrap();
    let (r, _) = refine_with_prefix(&s, floor, None, expired, root, &|| None);
    assert_eq!(r.lower.to_bits(), floor.to_bits());
    assert_eq!(r.masks, 0);
}

#[path = "input_head_directional_capacity_tests.rs"]
mod capacity_tests;
