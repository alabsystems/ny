// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn missing_recipe_provenance_preserves_bound_only_replay_but_refuses_retention() {
    let deadline = Instant::now() + Duration::from_secs(3);
    let spec = InputHeadSpec::new(
        1,
        1,
        vec![1.0],
        vec![0.0],
        vec![1.0, 0.0],
        vec![1.0, 0.0],
        [134; 32],
    )
    .unwrap();
    let source = Source::build(&spec, &[-1.0], &[1.0], 0, 1, deadline).unwrap();
    let mut model = Model::build(&source, deadline).unwrap();
    let hint = HeadFunctionalLpHint {
        row_duals: vec![1.0; model.rows.len()],
    };
    assert!(hint.validate(model.request()));
    let before = model.replay(&source, &hint, 1.0, deadline).unwrap();
    let (_, form, recipe) = model
        .replay_recorded(&source, &hint, 1.0, deadline, true)
        .unwrap();
    assert!(form.is_some() && recipe.is_some());

    // Numerical rows and the request remain literal. Metadata alone cannot
    // erase their checked floor, and incomplete metadata cannot be exported
    // as a recipe for another domain.
    model.row_tags.clear();
    assert_eq!(
        model
            .replay(&source, &hint, 1.0, deadline)
            .unwrap()
            .to_bits(),
        before.to_bits()
    );
    assert!(model
        .replay_recorded(&source, &hint, 1.0, deadline, true)
        .is_none());
    let foreign = Source::build(&spec, &[-1.0], &[1.0], 0, 1, deadline).unwrap();
    assert!(model.replay(&foreign, &hint, 1.0, deadline).is_none());
    let malformed = HeadFunctionalLpHint { row_duals: vec![] };
    assert!(model.replay(&source, &malformed, 1.0, deadline).is_none());
    let expired = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
    assert!(model.replay(&source, &hint, 1.0, expired).is_none());
}
