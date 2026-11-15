// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::*;
use ny_core::head_phase_advice::{HeadPhaseLpHint, HeadPhaseLpRequest};

fn end() -> Instant {
    Instant::now() + Duration::from_secs(10)
}

fn limits() -> Limits {
    Limits {
        nodes: 511,
        frontier: 256,
        depth: 16,
        calls: 64,
        work: 10_000_000,
    }
}

fn cycle(bias: f32) -> Source {
    let mut w = vec![0.0; 81];
    for j in 0..9 {
        w[j * 9 + j] = 1.0;
        w[((j + 1) % 9) * 9 + j] = -1.0;
    }
    let spec = InputHeadSpec::new(
        9,
        9,
        w,
        vec![0.0; 9],
        (0..9).flat_map(|_| [-1.0, 0.0]).collect(),
        vec![bias, 0.0],
        [181; 32],
    )
    .unwrap();
    Source::build(&spec, &[-1.0; 9], &[1.0; 9], 0, 1, end()).unwrap()
}

fn before(s: &Source) -> f64 {
    s.baseline(end()).unwrap()
}

#[test]
fn phase_tree_native_complete_cover_closes_nine_cycle_without_four_or_seven_depth_cap() {
    let s = cycle(8.5);
    let r = refine_with_prefix(&s, before(&s), None, end(), end(), limits(), &|| None);
    assert!(r.complete && r.open == 0);
    assert_eq!(r.closed, 18);
    assert_eq!(r.splits, 17);
    assert_eq!(r.nodes, 35);
    assert_eq!(r.depth, 9);
    assert_eq!(r.calls, 0);
    assert!(r.lower > 0.49 && r.lower <= 0.5, "{}", r.lower);
}

#[test]
fn phase_tree_missing_open_siblings_never_grant_completion_or_positive_global_floor() {
    let s = cycle(8.5);
    let l = Limits {
        nodes: 3,
        ..limits()
    };
    let r = refine_with_prefix(&s, before(&s), None, end(), end(), l, &|| None);
    assert!(!r.complete && r.open > 0);
    assert!(r.lower <= 0.0);
    assert_eq!(r.nodes, 3);
    assert!(r.lower >= before(&s));
}

#[test]
fn phase_tree_foreign_source_foreign_path_and_repeated_head_are_refused() {
    let a = cycle(8.5);
    let b = cycle(8.5);
    let model = Model::build(&a, None, limits(), end()).unwrap();
    let other = Model::build(&b, None, limits(), end()).unwrap();
    let path = model.path_rows(&[(0, false)], end()).unwrap();
    let foreign = other.path_rows(&[(0, false)], end()).unwrap();
    let duals = vec![0.0; model.rows.len() + path.rows.len()];
    assert!(model.replay(&b, &path, &duals, 1.0, false, end()).is_none());
    assert!(model
        .replay(&a, &foreign, &duals, 1.0, false, end())
        .is_none());
    assert!(model.request(&foreign, end()).is_none());
    assert!(model.path_rows(&[(0, false), (0, true)], end()).is_none());
    assert!(model.path_rows(&[(a.spec.heads, false)], end()).is_none());
}

#[test]
fn phase_tree_closed_zero_belongs_to_both_children_and_inactive_cap_keeps_root_e1() {
    let s = cycle(8.5);
    let model = Model::build(&s, None, limits(), end()).unwrap();
    assert!(s.first_error[0] > 0.0);
    let mut point = vec![I::p(0.0); model.boxes.len()];
    point[s.spec.inputs] = I::p(s.first_error[0] * 0.5);
    for active in [false, true] {
        let path = model.path_rows(&[(0, active)], end()).unwrap();
        assert!(path.rows[1].lower.lo <= -s.first_error[0]);
        assert!(path.rows[1].lower.hi >= -s.first_error[0]);
        for row in &path.rows {
            let lhs = row.coefficients.iter().fold(I::p(0.0), |v, &(j, a)| {
                v.add(a.mul(point[j]).unwrap()).unwrap()
            });
            assert!(lhs.lo >= row.lower.hi);
        }
    }
}

struct Fault(u8);
impl HeadPhaseAdvice for Fault {
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
        match self.0 {
            0 => None,
            1 => Some(HeadFunctionalLpHint {
                row_duals: vec![f64::NAN; request.rows.len()],
            }),
            2 => Some(HeadFunctionalLpHint {
                row_duals: vec![-1.0; request.rows.len()],
            }),
            3 => {
                std::thread::sleep(Duration::from_millis(120));
                Some(HeadFunctionalLpHint {
                    row_duals: vec![1.0; request.rows.len()],
                })
            }
            _ => panic!("untrusted provider failure"),
        }
    }
}

#[test]
fn phase_tree_bad_negative_panicking_or_late_advice_cannot_close_unsafe_source() {
    for fault in 0..5 {
        let s = cycle(7.5);
        let provider = Arc::new(Fault(fault));
        let stop = if fault == 3 {
            Instant::now() + Duration::from_millis(180)
        } else {
            end()
        };
        let r = refine_with_prefix(&s, before(&s), None, stop, end(), limits(), &|| {
            Some(Arc::clone(&provider) as Arc<dyn HeadPhaseAdvice>)
        });
        assert!(!r.complete);
        assert!(r.lower <= -0.49, "fault={fault} lower={}", r.lower);
        assert!(r.lower >= before(&s));
        assert!(r.calls <= limits().calls && r.work <= limits().work);
    }
}

#[test]
fn phase_tree_expired_authority_and_small_shared_work_budget_preserve_the_old_floor() {
    let s = cycle(8.5);
    let old = before(&s);
    let short = Instant::now() + Duration::from_millis(1);
    for (stop, authority, l) in [
        (short, end(), limits()),
        (end(), short, limits()),
        (
            end(),
            end(),
            Limits {
                work: 1,
                ..limits()
            },
        ),
    ] {
        let r = refine_with_prefix(&s, old, None, stop, authority, l, &|| None);
        assert!(!r.complete && r.calls == 0);
        assert_eq!(r.lower.to_bits(), old.to_bits());
    }
}

fn contradictory_source() -> Source {
    let spec = InputHeadSpec::new(
        1,
        2,
        vec![1.0, 1.0],
        vec![0.0, -0.5],
        vec![-1.0, 0.0, -1.0, 0.0],
        vec![2.0, 0.0],
        [182; 32],
    )
    .unwrap();
    Source::build(&spec, &[-1.0], &[1.0], 0, 1, end()).unwrap()
}

#[test]
fn phase_tree_zero_objective_farkas_checks_all_input_residuals_and_feasible_siblings() {
    let s = contradictory_source();
    let model = Model::build(&s, None, limits(), end()).unwrap();
    let impossible = model.path_rows(&[(0, false), (1, true)], end()).unwrap();
    let mut dual = vec![0.0; model.rows.len() + 4];
    dual[model.rows.len()] = 1.0;
    dual[model.rows.len() + 2] = 1.0;
    assert!(
        model
            .replay(&s, &impossible, &dual, 1.0, true, end())
            .unwrap()
            > 0.49
    );
    let sibling = model.path_rows(&[(0, true), (1, false)], end()).unwrap();
    assert!(model.replay(&s, &sibling, &dual, 1.0, true, end()).unwrap() <= 0.0);
    let single = model.path_rows(&[(1, true)], end()).unwrap();
    dual.truncate(model.rows.len() + 2);
    assert!(model.replay(&s, &single, &dual, 1.0, true, end()).unwrap() <= -0.49);
}

#[test]
fn phase_tree_phase_one_is_only_bounded_advice_and_preserves_original_row_order() {
    let s = contradictory_source();
    let model = Model::build(&s, None, limits(), end()).unwrap();
    let path = model.path_rows(&[(0, false), (1, true)], end()).unwrap();
    let original = model.request(&path, end()).unwrap();
    let trial = phase_one(&original, end()).unwrap();
    assert!(original.validate() && trial.validate());
    assert_eq!(trial.feature_count, original.feature_count + 1);
    assert_eq!(trial.rows.len(), original.rows.len());
    assert_eq!(trial.slope_rows, original.slope_rows);
    assert_eq!((trial.phase_rows).len(), 0);
    assert_eq!(trial.objective[original.feature_count], 1.0);
    for (a, b) in original.rows.iter().zip(&trial.rows) {
        assert_eq!(a.lower.to_bits(), b.lower.to_bits());
        let expected: Vec<_> = a
            .coefficients
            .iter()
            .map(|&(j, v)| (j + usize::from(j >= original.feature_count), v))
            .collect();
        let actual: Vec<_> = b
            .coefficients
            .iter()
            .copied()
            .filter(|v| v.0 != original.feature_count)
            .collect();
        assert_eq!(expected, actual);
    }
}
