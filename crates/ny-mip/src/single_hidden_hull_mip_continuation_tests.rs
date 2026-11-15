// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::super::bab::{HullBabOptions, HullBabStop};
use super::*;

fn end() -> Instant {
    Instant::now() + Duration::from_secs(30)
}
fn problem(c: f64) -> HullMarginProblem {
    HullMarginProblem::new(
        vec![1.0, 2.0, -1.0, -2.0],
        vec![0.0; 4],
        1,
        &[-1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        &[c, 0.0],
        2,
        &[(0, 1.0), (1, -1.0)],
        0.0,
        vec![-1.0],
        vec![1.0],
    )
    .unwrap()
}
fn manual(p: &HullMarginProblem) -> HullRootMipCuts<'_> {
    let bounds = preactivation_bounds(p);
    let roles = classify(p, &bounds);
    let m = triangle_multipliers(p, &roles, &bounds);
    let value = certify_hull_lower_bound(p, &m).unwrap();
    prepare(p, m, value, end(), &mut || true).unwrap()
}

#[test]
fn hull_portfolio_opaque_seed_refuses_foreign_source_bits_and_bad_vector_seals() {
    let p = problem(0.25);
    let foreign = problem(-0.25);
    for change in 0..6 {
        let mut cuts = manual(&p);
        match change {
            0 => cuts.problem = &foreign,
            1 => cuts.checked_lower = cuts.checked_lower.next_up(),
            2 => cuts.multipliers.g[0] = f64::NAN,
            3 => {
                cuts.multipliers.alpha.pop();
            }
            4 => {
                cuts.multipliers.g.pop();
            }
            _ => cuts.planes[0].g[0] = cuts.planes[0].g[0].next_up(),
        }
        assert!(
            cuts.continue_hull_bab(end(), Default::default()).is_none(),
            "{change}"
        );
    }
}

#[test]
fn hull_portfolio_later_caller_end_never_renews_stored_root_authority() {
    let p = problem(0.25);
    let mut cuts = manual(&p);
    cuts.deadline = Instant::now()
        .checked_sub(Duration::from_secs(1))
        .expect("overflow when subtracting duration from instant");
    assert!(cuts.continue_hull_bab(end(), Default::default()).is_none());
    let cuts = manual(&p);
    assert!(cuts
        .continue_hull_bab(
            Instant::now()
                .checked_sub(Duration::from_secs(1))
                .expect("overflow when subtracting duration from instant"),
            Default::default()
        )
        .is_none());
}

#[test]
fn hull_portfolio_completed_proof_rechecks_its_own_publication_clock() {
    let p = problem(0.25);
    let cuts = manual(&p);
    let mut proof = cuts.continue_hull_bab(end(), Default::default()).unwrap();
    assert!(proof.verified());
    assert!(proof.report().verified());
    proof.deadline = Instant::now()
        .checked_sub(Duration::from_secs(1))
        .expect("overflow when subtracting duration from instant");
    assert!(
        !proof.verified(),
        "old public report cannot renew native proof publication"
    );
    assert!(
        proof.report().verified(),
        "numerical cover remains diagnostic evidence"
    );
}

#[test]
fn hull_portfolio_open_native_tree_keeps_banked_global_ir_and_original_bundle() {
    let p = problem(-0.25);
    let cuts = manual(&p);
    let m = cuts.multipliers.clone();
    let checked = cuts.checked_lower;
    let weights = vec![p.w1.clone(), vec![-1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0]];
    let biases = vec![vec![0.0; 4], vec![-0.25, 0.0]];
    let dims = vec![1, 4, 2];
    let hidden = vec![vec![
        Bound::new(-1.0, 1.0),
        Bound::new(-2.0, 2.0),
        Bound::new(-1.0, 1.0),
        Bound::new(-2.0, 2.0),
    ]];
    let mut encoder =
        crate::encode_feedforward(&weights, &biases, &dims, &[Bound::new(-1.0, 1.0)], &hidden)
            .unwrap();
    let unsafe_row = encoder.constrain_output_leq_row(0, 1).unwrap();
    let mut parts = encoder.into_parts();
    let binding = HullMipEncoding {
        weights: &weights,
        biases: &biases,
        dimensions: &dims,
        hidden_bounds: &hidden,
        first_output: 0,
        second_output: 1,
        reverse: false,
        unsafe_row,
    };
    cuts.append_to(&mut parts, &binding, end()).unwrap();
    let banked = parts.problem.clone();
    let vars = parts.binary_vars.clone();
    let widths = parts.binary_widths.clone();
    let proof = cuts
        .continue_hull_bab(
            end(),
            HullBabOptions {
                max_nodes: 1,
                workers: 1,
            },
        )
        .unwrap();
    assert!(!proof.verified());
    assert_eq!(proof.report().stop, HullBabStop::NodeCap);
    assert_eq!(proof.report().open, 1);
    assert_eq!(proof.report().proposal_lp_calls, 0);
    assert_eq!(parts.problem, banked);
    assert_eq!(parts.binary_vars, vars);
    assert_eq!(parts.binary_widths, widths);
    assert_eq!(cuts.multipliers, m);
    assert_eq!(cuts.checked_lower.to_bits(), checked.to_bits());
    assert_eq!(parts.problem.margin_row(), banked.margin_row());
    assert!(
        p.d < 0.0 && p.b1.iter().all(|v| *v == 0.0),
        "original x=0 is unsafe"
    );
}
