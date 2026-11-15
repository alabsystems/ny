// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use ny_core::head_functional_advice::{HeadFunctionalLpHint, HeadFunctionalLpRequest};
use ny_core::head_phase_advice::{HeadPhaseLpHint, HeadPhaseLpRequest};
use ny_core::head_phase_cover_advice::PhaseCoverLeafHint;
use std::sync::atomic::{AtomicUsize, Ordering};

fn end() -> Instant {
    Instant::now() + Duration::from_secs(10)
}

fn cycle(bias: f32, duplicates: bool) -> InputHeadSpec {
    let heads = if duplicates { 14 } else { 7 };
    let mut first = vec![0.0; 7 * heads];
    let mut final_weight = vec![0.0; heads * 2];
    for head in 0..heads {
        let edge = head % 7;
        first[edge * heads + head] = 1.0;
        first[((edge + 1) % 7) * heads + head] = -1.0;
        final_weight[head * 2] = if head < 7 { -1.0 } else { 0.25 };
    }
    InputHeadSpec::new(
        7,
        heads,
        first,
        vec![0.0; heads],
        final_weight,
        vec![bias, 0.0],
        [71; 32],
    )
    .unwrap()
}

fn source(spec: &InputHeadSpec) -> Source {
    Source::build(spec, &[-1.0; 7], &[1.0; 7], 0, 1, end()).unwrap()
}

fn policy() -> HeadGroupPolicy {
    HeadGroupPolicy::new(3.0, 2, 4)
        .unwrap()
        .with_directional_cuts(false)
        .with_phase_cover(PhaseCoverPolicy::new(7).unwrap())
}

fn old_floor(source: &Source) -> f64 {
    let mut lower = source.baseline(end()).unwrap();
    if let Some(slopes) = source.slopes() {
        lower = lower.max(source.replay_retained(&slopes, end()).unwrap().0);
    }
    lower
}

fn native_hint(model: &Model<'_>) -> PhaseCoverLpHint {
    PhaseCoverLpHint {
        leaves: (0..model.leaves.len())
            .map(|mask| PhaseCoverLeafHint {
                mask,
                row_duals: model.native_duals(mask).unwrap(),
            })
            .collect(),
    }
}

fn jensen_spec() -> InputHeadSpec {
    InputHeadSpec::new(
        2,
        3,
        vec![1.0, 0.0, 0.5, 0.0, 1.0, 0.5],
        vec![0.0; 3],
        vec![0.5, 0.0, 0.5, 0.0, -1.0, 0.0],
        vec![1.0 / 16.0, 0.0],
        [72; 32],
    )
    .unwrap()
}

fn jensen_source() -> Source {
    Source::build(&jensen_spec(), &[-1.0; 2], &[1.0; 2], 0, 1, end()).unwrap()
}

fn jensen_row(rows: &[HeadFunctionalLpRow]) -> usize {
    rows.iter()
        .enumerate()
        .skip(6)
        .find(|(_, row)| {
            row.coefficients
                .iter()
                .any(|&(index, value)| index == 4 && value < 0.0)
                && row
                    .coefficients
                    .iter()
                    .any(|&(index, value)| index == 2 && value > 0.0)
                && row
                    .coefficients
                    .iter()
                    .any(|&(index, value)| index == 3 && value > 0.0)
        })
        .expect("a completed genuine Source Jensen row must be present")
        .0
}

fn jensen_cover_hint(request: &PhaseCoverLpRequest) -> PhaseCoverLpHint {
    let cut = jensen_row(&request.rows);
    PhaseCoverLpHint {
        leaves: request
            .leaves
            .iter()
            .map(|leaf| {
                let mut row_duals = vec![0.0; request.rows.len() + leaf.rows.len()];
                row_duals[cut] = 1.0;
                PhaseCoverLeafHint {
                    mask: leaf.mask,
                    row_duals,
                }
            })
            .collect(),
    }
}

#[test]
fn completed_optional_source_rows_reach_and_reprice_every_phase() {
    let s = jensen_source();
    let prior = triangle::Model::build(&s, end()).unwrap();
    assert!(prior.cuts() > 0);
    let expected_rows = prior.request().rows.clone();
    let mut old_duals = vec![0.0; expected_rows.len()];
    old_duals[jensen_row(&expected_rows)] = 1.0;
    let expected = prior
        .replay(
            &s,
            &HeadFunctionalLpHint {
                row_duals: old_duals,
            },
            1.0,
            end(),
        )
        .unwrap();
    let phase = Model::build_with_prefix(
        &s,
        PhaseCoverPolicy::new(1).unwrap(),
        Some(prior.into_phase_prefix()),
        end(),
    )
    .unwrap();
    assert_eq!(phase.request.rows.len(), expected_rows.len());
    for (actual, expected) in phase.request.rows.iter().zip(expected_rows) {
        assert_eq!(actual.coefficients, expected.coefficients);
        assert_eq!(actual.lower.to_bits(), expected.lower.to_bits());
    }
    let hint = jensen_cover_hint(phase.request());
    let mut floors = vec![old_floor(&s); phase.leaves.len()];
    assert_eq!(
        phase
            .replay_hint(&s, &hint, &mut floors, end())
            .unwrap()
            .len(),
        2
    );
    for lower in floors {
        assert_eq!(
            lower.to_bits(),
            expected.to_bits(),
            "conditional rows with zero dual cannot change the owned cut proof"
        );
        assert!(lower > 1.0 / 32.0 && lower <= 1.0 / 16.0);
    }
}

#[test]
fn production_moves_completed_prefix_even_when_old_advice_declines() {
    struct PrefixAdvice {
        functional_calls: AtomicUsize,
        phase_calls: AtomicUsize,
    }
    impl HeadPhaseAdvice for PrefixAdvice {
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
            self.functional_calls.fetch_add(1, Ordering::Relaxed);
            None
        }
        fn supports_phase_cover_advice(&self) -> bool {
            true
        }
        fn propose_phase_cover(
            &self,
            request: &PhaseCoverLpRequest,
            deadline: Instant,
        ) -> Option<PhaseCoverLpHint> {
            self.phase_calls.fetch_add(1, Ordering::Relaxed);
            assert!(live(deadline) && request.validate());
            assert_eq!(request.bounds.len(), 5);
            assert!(
                request.rows.len() > 6,
                "the already-checked Source prefix must reach the real production callback"
            );
            Some(jensen_cover_hint(request))
        }
    }
    let provider = Arc::new(PrefixAdvice {
        functional_calls: AtomicUsize::new(0),
        phase_calls: AtomicUsize::new(0),
    });
    let handle: Arc<dyn HeadPhaseAdvice> = provider.clone();
    let outcome = verify_with_provider(
        &jensen_spec(),
        &[-1.0; 2],
        &[1.0; 2],
        0,
        1,
        HeadGroupPolicy::new(3.0, 2, 4)
            .unwrap()
            // Exercise prefix transfer to the cover without earlier closure.
            .with_directional_cuts(false)
            .with_phase_cover(PhaseCoverPolicy::new(1).unwrap()),
        end(),
        || Some(Arc::clone(&handle)),
    );
    let InputHeadOutcome::Unsat(report, _) = outcome else {
        panic!("the real Source-owned optional row must close through the complete cover")
    };
    assert_eq!(provider.functional_calls.load(Ordering::Relaxed), 2);
    assert_eq!(provider.phase_calls.load(Ordering::Relaxed), 1);
    assert_eq!(
        report.triangle_replayed_candidates, 0,
        "declined old advice grants no numerical bound"
    );
    assert_eq!(report.phase_cover_ay_replayed_leaves, 2);
    assert!(report.phase_cover_before_lower.unwrap() <= 0.0);
    assert!(report.lower.unwrap() > 1.0 / 32.0 && report.lower.unwrap() <= 1.0 / 16.0);
}

#[test]
fn cached_prefix_checks_identity_expiry_and_preserves_every_scalar_row_at_capacity() {
    let s = jensen_source();
    let foreign = s.clone();
    let model = Model::build_with_prefix(
        &foreign,
        PhaseCoverPolicy::new(1).unwrap(),
        Some(
            triangle::Model::build(&s, end())
                .unwrap()
                .into_phase_prefix(),
        ),
        end(),
    )
    .unwrap();
    assert_eq!(
        model.rows.len(),
        6,
        "a foreign prefix must fall back to freshly checked bare Source rows"
    );
    assert!(triangle::Model::build(&s, end())
        .unwrap()
        .into_phase_prefix()
        .into_phase_parts(&s, PHASE_COVER_MAX_ROWS, Instant::now())
        .is_none());
    assert!(Model::build_with_prefix(
        &s,
        PhaseCoverPolicy::new(1).unwrap(),
        Some(
            triangle::Model::build(&s, end())
                .unwrap()
                .into_phase_prefix()
        ),
        Instant::now(),
    )
    .is_none());
    let prior = triangle::Model::build(&s, end()).unwrap();
    assert!(prior.request().rows.len() > 6);
    let (_, kept) = prior
        .into_phase_prefix()
        .into_phase_parts(&s, 6, end())
        .unwrap();
    assert_eq!(kept.len(), 6, "only optional suffix rows may be truncated");
    assert!(
        triangle::Model::build(&s, end())
            .unwrap()
            .into_phase_prefix()
            .into_phase_parts(&s, 5, end())
            .is_none(),
        "a row cap cannot delete a mandatory scalar row"
    );

    let spec = InputHeadSpec::new(
        1,
        512,
        vec![1.0; 512],
        vec![0.0; 512],
        (0..512).flat_map(|_| [-1.0, 0.0]).collect(),
        vec![0.0; 2],
        [73; 32],
    )
    .unwrap();
    let wide = Source::build(&spec, &[-1.0], &[1.0], 0, 1, end()).unwrap();
    let phase = Model::build_with_prefix(
        &wide,
        PhaseCoverPolicy::new(7).unwrap(),
        Some(
            triangle::Model::build(&wide, end())
                .unwrap()
                .into_phase_prefix(),
        ),
        end(),
    )
    .unwrap();
    assert!(
        phase.rows.len() >= 1024,
        "all512 genuine head triangles must survive"
    );
    assert_eq!(phase.leaves.len(), 128);
    assert!(phase.request.validate());
    assert!(phase
        .leaves
        .iter()
        .all(|leaf| phase.rows.len() + leaf.len() <= PHASE_COVER_MAX_ROWS));
}

#[test]
fn policy_is_default_off_and_seven_is_a_split_count_not_a_leaf_count() {
    assert!(PhaseCoverPolicy::new(0).is_none());
    assert!(PhaseCoverPolicy::new(8).is_none());
    assert!(HeadGroupPolicy::new(3.0, 2, 4)
        .unwrap()
        .phase_cover()
        .is_none());
    assert_eq!(policy().phase_cover().unwrap().max_heads(), 7);
    let s = source(&cycle(6.5, false));
    let model = Model::build(&s, policy().phase_cover().unwrap(), end()).unwrap();
    assert_eq!(model.selected.len(), 7);
    assert_eq!(model.request.leaves.len(), 128);
    assert!(model.request.validate());
}

#[test]
fn seven_cycle_has_exact_vertex_maximum_six_and_old_outer_point_stays_feasible() {
    let mut maximum = 0;
    for mask in 0..128 {
        let x: Vec<i32> = (0..7)
            .map(|bit| if mask & (1 << bit) != 0 { 1 } else { -1 })
            .collect();
        let value: i32 = (0..7)
            .map(|edge| (x[edge] - x[(edge + 1) % 7]).max(0))
            .sum();
        maximum = maximum.max(value);
    }
    assert_eq!(maximum, 6);
    let s = source(&cycle(6.5, false));
    assert!(s.first_error.iter().all(|&value| value > 0.0));
    assert!(s.output_error > 0.0);
    let old = triangle::Model::build(&s, end()).unwrap();
    let request = old.request();
    let point: Vec<_> = vec![0.0; 7].into_iter().chain(vec![1.0; 7]).collect();
    assert_eq!(point.len(), request.bounds.len());
    for row in &request.rows {
        let lhs: f64 = row
            .coefficients
            .iter()
            .map(|&(index, a)| a * point[index])
            .sum();
        assert!(
            lhs >= row.lower,
            "old supported row must admit the cycle point: {lhs} < {}",
            row.lower
        );
    }
    let margin = 6.5 - 7.0 - s.output_error;
    assert!(margin < 0.0 && old_floor(&s) <= 0.0);
}

#[test]
fn production_native_complete_cover_closes_real_source_with_all_error_debt() {
    let spec = cycle(6.5, false);
    let off = verify_with_provider(
        &spec,
        &[-1.0; 7],
        &[1.0; 7],
        0,
        1,
        HeadGroupPolicy::new(3.0, 2, 4)
            .unwrap()
            .with_directional_cuts(false),
        end(),
        || None,
    );
    let InputHeadOutcome::Unknown(old) = off else {
        panic!("old scalar portfolio must stay open")
    };
    assert!(old.lower.unwrap() <= 0.0);
    assert_eq!(old.phase_cover_leaves, 0);
    let on = verify_with_provider(&spec, &[-1.0; 7], &[1.0; 7], 0, 1, policy(), end(), || None);
    let InputHeadOutcome::Unsat(report, _) = on else {
        panic!("real Source/G2 cover must close")
    };
    let lower = report.lower.unwrap();
    assert!(
        lower > 0.0 && lower <= 0.5,
        "real outward replay lower={lower}"
    );
    assert_eq!(report.phase_cover_leaves, 128);
    assert_eq!(report.phase_cover_replayed_leaves, 128);
    assert_eq!(report.phase_cover_native_replayed_leaves, 128);
    assert_eq!(report.phase_cover_ay_replayed_leaves, 0);
    assert_eq!(
        report.phase_cover_advice_calls, 0,
        "a completed native proof skips redundant AY work"
    );
    assert!(report.phase_cover_before_lower.unwrap() <= 0.0);
}

#[test]
fn unsafe_original_cycle_sibling_cannot_acquire_a_root_token() {
    let spec = cycle(5.5, false);
    // Alternating endpoints have six directed cut units, so this original
    // authored network has margin -1/2 at a genuine point.
    let x = [1, -1, 1, -1, 1, -1, 1];
    let sum: i32 = (0..7)
        .map(|edge| (x[edge] - x[(edge + 1) % 7]).max(0))
        .sum();
    assert_eq!(5.5 - f64::from(sum), -0.5);
    let outcome =
        verify_with_provider(&spec, &[-1.0; 7], &[1.0; 7], 0, 1, policy(), end(), || None);
    let InputHeadOutcome::Unknown(report) = outcome else {
        panic!("unsafe original sibling must remain open")
    };
    assert!(report.lower.unwrap() <= -0.5);
    assert_eq!(report.phase_cover_native_replayed_leaves, 128);
}

struct DuplicateAdvice;

impl HeadPhaseAdvice for DuplicateAdvice {
    fn propose(&self, _: &HeadPhaseLpRequest, _: Instant) -> Option<HeadPhaseLpHint> {
        None
    }

    fn supports_phase_cover_advice(&self) -> bool {
        true
    }

    fn propose_phase_cover(
        &self,
        request: &PhaseCoverLpRequest,
        deadline: Instant,
    ) -> Option<PhaseCoverLpHint> {
        assert!(live(deadline) && request.validate());
        assert_eq!(request.selected_heads, 7);
        assert_eq!(request.bounds.len(), 21);
        assert!(request.rows.len() >= 28);
        Some(PhaseCoverLpHint {
            leaves: request
                .leaves
                .iter()
                .map(|leaf| {
                    let mut row_duals = vec![0.0; request.rows.len() + leaf.rows.len()];
                    for bit in 0..7 {
                        // All seven negative caps, plus genuine positive duplicate
                        // epigraphs in ACTIVE nominal phases. Residuals on every
                        // other authored variable are still priced by production.
                        row_duals[request.rows.len() + 2 * bit + 1] = 1.0;
                        if leaf.mask & (1 << bit) != 0 {
                            row_duals[2 * (7 + bit)] = 0.25;
                        }
                    }
                    PhaseCoverLeafHint {
                        mask: leaf.mask,
                        row_duals,
                    }
                })
                .collect(),
        })
    }
}

#[test]
fn full_authored_positive_remainder_requires_and_accepts_locally_replayed_advice() {
    let spec = cycle(4.75, true);
    let handle: Arc<dyn HeadPhaseAdvice> = Arc::new(DuplicateAdvice);
    let outcome = verify_with_provider(&spec, &[-1.0; 7], &[1.0; 7], 0, 1, policy(), end(), || {
        Some(Arc::clone(&handle))
    });
    let InputHeadOutcome::Unsat(report, _) = outcome else {
        panic!("complete source-owned LP dual cover must close")
    };
    let lower = report.lower.unwrap();
    assert!(
        lower > 0.0 && lower <= 0.25,
        "all-head replay lower={lower}"
    );
    assert_eq!(report.phase_cover_advice_calls, 1);
    assert_eq!(report.phase_cover_native_replayed_leaves, 128);
    assert_eq!(report.phase_cover_ay_replayed_leaves, 128);
    assert_eq!(report.phase_cover_replayed_leaves, 128);
}

#[test]
fn missing_leaf_inherits_old_floor_and_duplicate_or_nan_hint_is_rejected() {
    let s = source(&cycle(6.5, false));
    let model = Model::build(&s, PhaseCoverPolicy::new(7).unwrap(), end()).unwrap();
    let before = old_floor(&s);
    let complete = native_hint(&model);
    let mut missing = complete.clone();
    missing.leaves.pop();
    let mut floors = vec![before; 128];
    assert_eq!(
        model
            .replay_hint(&s, &missing, &mut floors, end())
            .map(|masks| masks.len()),
        Some(127)
    );
    assert_eq!(floors[127].to_bits(), before.to_bits());
    assert_eq!(cover_lower(&floors).unwrap().to_bits(), before.to_bits());
    let mut duplicate = complete.clone();
    duplicate.leaves[1].mask = duplicate.leaves[0].mask;
    let mut untouched = vec![before; 128];
    assert!(model
        .replay_hint(&s, &duplicate, &mut untouched, end())
        .is_none());
    assert!(untouched
        .iter()
        .all(|value| value.to_bits() == before.to_bits()));
    let mut nan = complete;
    nan.leaves[0].row_duals[0] = f64::NAN;
    assert!(model.replay_hint(&s, &nan, &mut untouched, end()).is_none());
    assert!(untouched
        .iter()
        .all(|value| value.to_bits() == before.to_bits()));
}

#[test]
fn foreign_source_identity_expired_replay_and_expired_commit_fail_closed() {
    let s = source(&cycle(6.5, false));
    let foreign = s.clone();
    assert!(
        Arc::ptr_eq(&s.spec, &foreign.spec),
        "even shared authored bytes do not transfer Source identity"
    );
    let model = Model::build(&s, PhaseCoverPolicy::new(7).unwrap(), end()).unwrap();
    let duals = model.native_duals(1).unwrap();
    assert!(model.replay_leaf(&foreign, 1, &duals, 1.0, end()).is_none());
    let expired = Instant::now();
    assert!(model.replay_leaf(&s, 1, &duals, 1.0, expired).is_none());
    assert!(Model::build(&s, PhaseCoverPolicy::new(7).unwrap(), expired).is_none());
    let lower = model.replay_leaf(&s, 1, &duals, 1.0, end()).unwrap();
    let before = old_floor(&s);
    assert_eq!(
        commit_triangle_best(before, lower, Instant::now(), end()).to_bits(),
        before.to_bits()
    );
    assert_eq!(
        commit_triangle_best(before, lower, end(), Instant::now()).to_bits(),
        before.to_bits()
    );
}

#[test]
fn inactive_nominal_phase_retains_error_and_all_output_debt_is_priced() {
    let s = source(&cycle(6.5, false));
    let model = Model::build(&s, PhaseCoverPolicy::new(7).unwrap(), end()).unwrap();
    for (bit, &head) in model.selected.iter().enumerate() {
        let cap = &model.leaves[0][2 * bit + 1];
        assert_eq!(cap.coefficients.len(), 1);
        assert_eq!(cap.lower.lo, -s.first_error[head]);
        assert!(
            cap.lower.lo < 0.0,
            "nominal inactive does not mean actual FP ReLU zero"
        );
    }
    let mut debt = source(&cycle(6.5, false));
    debt.first_error.fill(1.0);
    debt.output_error += 2.0;
    let charged = Model::build(&debt, PhaseCoverPolicy::new(7).unwrap(), end()).unwrap();
    let mask = 0b0101010;
    let lower = charged
        .replay_leaf(
            &debt,
            mask,
            &charged.native_duals(mask).unwrap(),
            1.0,
            end(),
        )
        .unwrap();
    assert!(
        lower <= 0.5 - 7.0 - debt.output_error,
        "no ROOT E1 or Eout may cancel: {lower}"
    );
}

#[test]
fn arbitrary_signed_advice_cannot_raise_an_unsafe_leaf_cover_to_positive() {
    let s = source(&cycle(5.5, false));
    let model = Model::build(&s, PhaseCoverPolicy::new(7).unwrap(), end()).unwrap();
    let before = old_floor(&s);
    for value in [-1e50, -1.0, 0.0, 0.5, 1.0, 1e50] {
        let hint = PhaseCoverLpHint {
            leaves: model
                .leaves
                .iter()
                .enumerate()
                .map(|(mask, leaf)| PhaseCoverLeafHint {
                    mask,
                    row_duals: vec![value; model.rows.len() + leaf.len()],
                })
                .collect(),
        };
        let mut floors = vec![before; 128];
        let _ = model.replay_hint(&s, &hint, &mut floors, end());
        assert!(
            cover_lower(&floors).unwrap() <= -0.5,
            "untrusted dual value {value} forged an unsafe cover"
        );
    }
}
