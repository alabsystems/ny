// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::*;
use ny_core::head_functional_advice::HeadFunctionalLpHint;

fn deadline() -> Instant {
    Instant::now() + Duration::from_secs(3)
}
fn expired() -> Instant {
    Instant::now().checked_sub(Duration::from_secs(1)).unwrap()
}
fn cycle_spec(bias: f32, last_input: bool) -> InputHeadSpec {
    let d = if last_input { 6 } else { 5 };
    let mut first = vec![0.0; d * 6];
    for j in 0..5 {
        first[j * 6 + j] = 1.0;
        first[((j + 1) % 5) * 6 + j] = -1.0;
    }
    if last_input {
        first[5 * 6] = 8.0;
    }
    let mut first_bias = vec![0.0; 6];
    first_bias[5] = 1.0;
    let mut weight = vec![0.0; 12];
    for j in 0..5 {
        weight[j * 2] = -1.0;
    }
    weight[10] = 1.0;
    InputHeadSpec::new(d, 6, first, first_bias, weight, vec![bias, 0.0], [107; 32]).unwrap()
}
fn source(bias: f32, radius: f64) -> Source {
    Source::build(
        &cycle_spec(bias, false),
        &[-radius; 5],
        &[radius; 5],
        0,
        1,
        deadline(),
    )
    .unwrap()
}
fn group() -> Vec<(usize, f64)> {
    (0..5).map(|j| (j, 1.0)).collect()
}
fn recipe_for(s: &Source, group: Vec<(usize, f64)>, p: f64) -> Recipe {
    let coefficients = group
        .iter()
        .map(|&(j, t)| (s.spec.inputs + j, I::p(-t)))
        .collect();
    Recipe::Dual {
        rows: vec![RowRecipe {
            coefficients,
            mass: None,
            group: Some(group),
        }],
        points: vec![p],
    }
}
fn lower(s: &Source, f: &ResidualForm) -> f64 {
    let x =
        s.lo.iter()
            .zip(&s.hi)
            .map(|(&lo, &hi)| I { lo, hi })
            .collect::<Vec<_>>();
    let r = s.relu_boxes_on(&x, deadline()).unwrap();
    f.support(s, &x, &r, deadline()).unwrap()
}
fn cycle_value(bits: usize, bias: f64) -> f64 {
    let x = (0..5)
        .map(|j| if bits & (1 << j) == 0 { -1.0f64 } else { 1.0 })
        .collect::<Vec<_>>();
    bias + 1.0
        - (0..5)
            .map(|j| (x[j] - x[(j + 1) % 5]).max(0.0))
            .sum::<f64>()
}

#[test]
fn grown_model_tags_mint_same_source_form_and_real_original_instance() {
    let s = source(49.0 / 16.0, 1.0);
    let model = triangle::Model::build(&s, deadline()).unwrap();
    assert_eq!(model.large_negative_groups(), 1);
    assert_eq!(model.large_negative_group_masks(), 32);
    let at = model
        .request()
        .rows
        .iter()
        .position(|row| {
            row.coefficients.len() == 5
                && row
                    .coefficients
                    .iter()
                    .enumerate()
                    .all(|(at, &(j, a))| j == 5 + at && a == -1.0)
        })
        .unwrap();
    let mut dual = vec![0.0; model.request().rows.len()];
    dual[at] = 1.0;
    let (bound, form, recipe) = model
        .replay_recorded(
            &s,
            &HeadFunctionalLpHint { row_duals: dual },
            1.0,
            deadline(),
            true,
        )
        .unwrap();
    assert!(bound > 1.0 / 32.0 && bound <= 1.0 / 16.0);
    let form = form.unwrap();
    let recipe = recipe.unwrap();
    let carry =
        InputHeadContinuation::build(&s, &[form], &[recipe], &s.lo, &s.hi, deadline(), deadline())
            .unwrap();
    let certified = carry
        .try_support_box(&[-1.0; 5], &[1.0; 5], deadline())
        .unwrap();
    assert!(certified > 1.0 / 32.0 && certified <= 1.0 / 16.0);
    assert!(carry.has_complete_source_cover(deadline()));
    assert_eq!(
        carry.try_finish(deadline()).unwrap().scope(),
        TokenScope::Instance
    );
}

#[test]
fn wider_five_head_source_recomputes_all32_rhs_and_owns_new_generation() {
    let narrow = source(49.0 / 16.0, 0.5);
    let wide = source(49.0 / 16.0, 1.0);
    let recipe = recipe_for(&narrow, group(), 1.0);
    let narrow_form = recipe.replay(&narrow, deadline()).unwrap();
    assert!(lower(&narrow, &narrow_form) > 2.0);
    let form = recipe.replay(&wide, deadline()).unwrap();
    let bound = lower(&wide, &form);
    assert!(bound > 1.0 / 32.0 && bound <= 1.0 / 16.0);
    let carry = InputHeadContinuation::build(
        &narrow,
        &[narrow_form],
        &[recipe],
        &[-1.0; 5],
        &[1.0; 5],
        deadline(),
        deadline(),
    )
    .unwrap();
    assert!(!Arc::ptr_eq(&narrow.spec, &carry.source.spec));
    assert!(
        carry
            .try_support_box(&[-1.0; 5], &[1.0; 5], deadline())
            .unwrap()
            > 1.0 / 32.0
    );
    assert_eq!(
        carry.try_finish(deadline()).unwrap().scope(),
        TokenScope::Instance
    );
}

#[test]
fn wider_rhs_retains_the_last_authored_input_and_refuses_unsafe_instance() {
    let spec = cycle_spec(49.0 / 16.0, true);
    let narrow = Source::build(
        &spec,
        &[-0.5, -0.5, -0.5, -0.5, -0.5, 0.0],
        &[0.5, 0.5, 0.5, 0.5, 0.5, 0.0],
        0,
        1,
        deadline(),
    )
    .unwrap();
    let lo = [-1.0, -1.0, -1.0, -1.0, -1.0, 0.0];
    let hi = [1.0; 6];
    let wide = Source::build(&spec, &lo, &hi, 0, 1, deadline()).unwrap();
    let recipe = recipe_for(&narrow, group(), 1.0);
    let form = recipe.replay(&wide, deadline()).unwrap();
    // x0..4=0, last x5=1 makes y0=8 and true M=-63/16.
    assert!(lower(&wide, &form) <= -63.0 / 16.0);
    let carry =
        InputHeadContinuation::build(&narrow, &[], &[recipe], &lo, &hi, deadline(), deadline())
            .unwrap();
    assert!(
        carry
            .try_support_box(&[0.0; 6], &[0.0, 0.0, 0.0, 0.0, 0.0, 1.0], deadline())
            .unwrap()
            < 0.0
    );
    assert!(carry.try_finish(deadline()).is_none());
}

#[test]
fn arbitrary_unequal_overdrawn_masses_reprice_full_q_and_bound_all_cycle_corners() {
    let s = source(49.0 / 16.0, 1.0);
    let g = vec![(0, 0.5), (1, 2.0), (2, 1.0), (3, 1.0), (4, 3.0)];
    for p in [0.0, 0.25, 1.0, 2.0] {
        let f = recipe_for(&s, g.clone(), p).replay(&s, deadline()).unwrap();
        let bound = lower(&s, &f);
        for bits in 0..32 {
            assert!(bound <= cycle_value(bits, 49.0 / 16.0));
        }
    }
    let mut uncertain = source(49.0 / 16.0, 1.0);
    uncertain.q[4] = I { lo: -1.5, hi: -0.5 };
    let bound = lower(
        &uncertain,
        &recipe_for(&uncertain, group(), 1.0)
            .replay(&uncertain, deadline())
            .unwrap(),
    );
    assert!(
        bound < 0.0,
        "crossing full-Q residual cannot be nominally canceled"
    );
}

#[test]
fn independent_first_bias_output_and_remaining_head_debts_survive_repricing() {
    let base = source(49.0 / 16.0, 1.0);
    let baseline = lower(
        &base,
        &recipe_for(&base, group(), 1.0)
            .replay(&base, deadline())
            .unwrap(),
    );
    for kind in 0..4 {
        let mut s = source(49.0 / 16.0, 1.0);
        match kind {
            0 => s.first_error[4] += 2.0,
            1 => {
                s.bias = I {
                    lo: -2.0,
                    hi: s.bias.hi,
                }
            }
            2 => s.output_error += 2.0,
            _ => s.q[5] = I::p(-1.0),
        }
        let bound = lower(
            &s,
            &recipe_for(&s, group(), 1.0).replay(&s, deadline()).unwrap(),
        );
        assert!(bound < baseline && bound < 0.0);
    }
}

#[test]
fn subnormal_group_error_is_charged_before_large_dual_amplification() {
    let spec = InputHeadSpec::new(
        1,
        5,
        vec![0.0; 5],
        vec![0.0; 5],
        vec![0.0; 10],
        vec![0.0; 2],
        [108; 32],
    )
    .unwrap();
    let s = Source::build(&spec, &[-1.0], &[1.0], 0, 1, deadline()).unwrap();
    let eta = f64::from_bits(1);
    let big = 2f64.powi(256);
    let g = (0..5).map(|j| (j, eta)).collect();
    let bound = lower(&s, &recipe_for(&s, g, big).replay(&s, deadline()).unwrap());
    assert!(bound < -(eta * big) * 0.25);
}

#[test]
fn tie_and_unsafe_odd_cycle_cannot_complete_the_original_cover() {
    for b in [3.0, 47.0 / 16.0] {
        let s = source(b, 1.0);
        let recipe = recipe_for(&s, group(), 1.0);
        let carry = InputHeadContinuation::build(
            &s,
            &[recipe.replay(&s, deadline()).unwrap()],
            &[recipe],
            &s.lo,
            &s.hi,
            deadline(),
            deadline(),
        )
        .unwrap();
        assert!(
            carry
                .try_support_box(&[-1.0; 5], &[1.0; 5], deadline())
                .unwrap()
                <= 0.0
        );
        assert!(!carry.has_complete_source_cover(deadline()));
        assert!(carry.try_finish(deadline()).is_none());
        assert!(cycle_value(0b01010, f64::from(b)) <= 0.0);
    }
}

#[test]
fn self_duplicate_unordered_or_incomplete_five_head_lhs_refuses_replay() {
    let s = source(49.0 / 16.0, 1.0);
    for g in [
        vec![(0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0), (3, 1.0)],
        vec![(1, 1.0), (0, 1.0), (2, 1.0), (3, 1.0), (4, 1.0)],
        vec![(0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0), (4, 0.0)],
        vec![(0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0), (99, 1.0)],
    ] {
        assert!(recipe_for(&s, g, 1.0).replay(&s, deadline()).is_none());
    }
    let mut bad = recipe_for(&s, group(), 1.0);
    if let Recipe::Dual { rows, .. } = &mut bad {
        rows[0].coefficients.pop();
    }
    assert!(bad.replay(&s, deadline()).is_none());
}

#[test]
fn nonzero_mask_budget_is_shared_across_forms_and_zero_duals_cost_no_masks() {
    let s = source(49.0 / 16.0, 1.0);
    let mut budget = GroupRepriceBudget::new(deadline());
    let recipe = recipe_for(&s, group(), 1.0);
    for _ in 0..8 {
        assert!(recipe
            .replay_with_budget(&s, deadline(), &mut budget)
            .is_some());
    }
    assert_eq!(budget.masks, 256);
    assert!(budget.cells <= MAX_REPRICE_CELLS);
    assert!(recipe
        .replay_with_budget(&s, deadline(), &mut budget)
        .is_none());
    assert_eq!(budget.masks, 256);
    assert!(recipe_for(&s, group(), 0.0)
        .replay_with_budget(&s, deadline(), &mut budget)
        .is_some());
    assert_eq!(budget.masks, 256);
}

#[test]
fn small_and_five_head_forms_share_one_cell_cap_and_no_new_deadline() {
    let s = source(49.0 / 16.0, 1.0);
    let mut budget = GroupRepriceBudget::new(deadline());
    assert!(recipe_for(&s, vec![(0, 1.0), (1, 1.0)], 1.0)
        .replay_with_budget(&s, deadline(), &mut budget)
        .is_some());
    assert_eq!(budget.masks, 4);
    assert!(recipe_for(&s, group(), 1.0)
        .replay_with_budget(&s, deadline(), &mut budget)
        .is_some());
    assert_eq!(budget.masks, 36);
    budget.cells = MAX_REPRICE_CELLS;
    assert!(recipe_for(&s, group(), 1.0)
        .replay_with_budget(&s, deadline(), &mut budget)
        .is_none());
    assert_eq!(budget.cells, MAX_REPRICE_CELLS);
    let original_cap = expired();
    let mut expired_budget = GroupRepriceBudget::new(original_cap);
    assert!(recipe_for(&s, group(), 1.0)
        .replay_with_budget(&s, deadline(), &mut expired_budget)
        .is_none());
    assert_eq!(expired_budget.deadline, original_cap);
    assert_eq!(expired_budget.masks, 0);
}

#[test]
fn partial32mask_refusal_never_becomes_a_residual_form() {
    let s = source(49.0 / 16.0, 1.0);
    let recipe = recipe_for(&s, group(), 1.0);
    let mut budget = GroupRepriceBudget::new(deadline());
    triangle::expire_five_reprice_after_mask(1);
    assert!(recipe
        .replay_with_budget(&s, deadline(), &mut budget)
        .is_none());
    assert_eq!(budget.masks, 1);
    assert!(budget.cells > 0);
    assert!(recipe.replay(&s, expired()).is_none());
}

#[test]
fn optional_unfinished_wider_form_keeps_the_earlier_complete_floor() {
    let narrow = source(49.0 / 16.0, 0.5);
    let first = recipe_for(&narrow, group(), 1.0);
    let Recipe::Dual { rows, points } = first.clone() else {
        unreachable!()
    };
    let unfinished = Recipe::Dual {
        rows: vec![rows[0].clone(); 9],
        points: vec![points[0]; 9],
    };
    let carry = InputHeadContinuation::build(
        &narrow,
        &[],
        &[first, unfinished],
        &[-1.0; 5],
        &[1.0; 5],
        deadline(),
        deadline(),
    )
    .unwrap();
    assert_eq!(
        carry.forms.len(),
        2,
        "baseline plus the one actually completed form"
    );
    assert!(
        carry
            .try_support_box(&[-1.0; 5], &[1.0; 5], deadline())
            .unwrap()
            > 1.0 / 32.0
    );
    assert_eq!(
        carry.try_finish(deadline()).unwrap().scope(),
        TokenScope::Instance
    );
}

#[test]
fn foreign_source_form_or_late_authority_never_authorizes_five_head_cover() {
    let s = source(49.0 / 16.0, 1.0);
    let foreign = source(49.0 / 16.0, 1.0);
    let f = recipe_for(&s, group(), 1.0).replay(&s, deadline()).unwrap();
    let x = vec![I { lo: -1.0, hi: 1.0 }; 5];
    let r = foreign.relu_boxes_on(&x, deadline()).unwrap();
    assert!(f.support(&foreign, &x, &r, deadline()).is_none());
    assert!(InputHeadContinuation::build(
        &foreign,
        &[f.clone()],
        &[],
        &foreign.lo,
        &foreign.hi,
        deadline(),
        deadline()
    )
    .is_none());
    assert!(InputHeadContinuation::build(
        &s,
        &[f.clone()],
        &[],
        &s.lo,
        &s.hi,
        deadline(),
        expired()
    )
    .is_none());
    let carry =
        InputHeadContinuation::build(&s, &[f], &[], &s.lo, &s.hi, deadline(), deadline()).unwrap();
    assert!(
        carry
            .try_support_box(&[-1.0; 5], &[1.0; 5], deadline())
            .unwrap()
            > 0.0
    );
    assert!(carry.has_complete_source_cover(deadline()));
    assert!(carry.try_finish(expired()).is_none());
    assert!(!carry.has_complete_source_cover(expired()));
}
