// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Admission must retain old optional rows until the complete V2 request validates.
use super::*;

#[test]
fn feedback_full_row_capacity_refuses_new_geometry_without_deleting_accepted_rows() {
    let small = spec(30.5, false);
    let first = (0..35)
        .flat_map(|input| {
            let first = &small.first;
            (0..512).map(move |head| first[input * 35 + head % 35])
        })
        .collect();
    let large = InputHeadSpec::new(
        35,
        512,
        first,
        vec![0.0; 512],
        (0..512).flat_map(|_| [-1.0, 0.0]).collect(),
        vec![1024.0, 0.0],
        [102; 32],
    )
    .unwrap();
    let s = Source::build(&large, &[-1.0; 35], &[1.0; 35], 0, 1, end()).unwrap();
    let mut prefix = None;
    assert_eq!(
        triangle::Model::prepare_root_prefix(&s, &mut prefix, 12, end()),
        Some(4)
    );
    let four = cuts(&s, 0..4, false);
    let model = triangle::Model::from_root_portfolio(&s, &mut prefix, &four, end()).unwrap();
    let expected = model.request().rows.clone();
    assert_eq!(expected.len(), 1028);
    prefix = Some(model.into_phase_prefix());
    // Planning, no certified candidate, and an empty first batch must not
    // remove the old optional suffix before a complete admission succeeds.
    assert_eq!(
        triangle::Model::prepare_root_prefix(&s, &mut prefix, 12, end()),
        Some(4)
    );
    let invalid = Direction {
        heads: vec![(0, 1.0); 7],
    };
    assert!(certify(&s, invalid, &mut budget()).is_none());
    assert!(
        triangle::Model::from_root_portfolio_reserved(&s, &mut prefix, &[], 4, end()).is_none()
    );
    let fifth = cuts(&s, 4..5, false);
    assert!(triangle::Model::from_root_portfolio(&s, &mut prefix, &fifth, end()).is_none());
    let (_, retained) = prefix.unwrap().into_phase_parts(&s, 1028, end()).unwrap();
    assert_eq!(retained.len(), 1028);
    for (row, exported) in retained.iter().zip(&expected) {
        assert_eq!(row.lower.lo.to_bits(), exported.lower.to_bits());
        assert_eq!(
            row.coefficients
                .iter()
                .map(|&(j, a)| (j, 0.5 * a.lo + 0.5 * a.hi))
                .filter(|&(_, a)| a != 0.0)
                .collect::<Vec<_>>(),
            exported.coefficients
        );
    }
}

#[test]
fn feedback_first_wire_refusal_preserves_the_complete_old_optional_prefix() {
    // Valid scalar request just below V2's nonzero cap. Four new dense eta
    // rows exceed that independent wire cap even after reserving row slots.
    let mut first = vec![0.0; 65 * 512];
    for j in 0..64 {
        for h in 0..512 {
            first[j * 512 + h] = 1.0;
        }
    }
    for h in 0..128 {
        first[64 * 512 + h] = 1.0;
    }
    let spec = InputHeadSpec::new(
        65,
        512,
        first,
        vec![0.0; 512],
        (0..512).flat_map(|_| [-1.0, 0.0]).collect(),
        vec![100000.0, 0.0],
        [105; 32],
    )
    .unwrap();
    let s = Source::build(&spec, &[-1.0; 65], &[1.0; 65], 0, 1, end()).unwrap();
    let make = |start| Direction {
        heads: (start..start + 7).map(|h| (h, 1.0)).collect(),
    };
    let mut work = budget();
    let old = certify_affine(&s, make(0), vec![(0, 7.0)], &mut work).unwrap();
    let model = triangle::Model::from_root_portfolio(&s, &mut None, &[old], end()).unwrap();
    let expected = model.request().clone();
    assert!(expected.validate());
    assert_eq!(expected.rows.len(), 1025);
    let mut prefix = Some(model.into_phase_prefix());
    assert_eq!(
        triangle::Model::prepare_root_prefix(&s, &mut prefix, 4, end()),
        Some(4)
    );
    let rows: Vec<_> = (1..=4)
        .map(|block| {
            certify_affine(
                &s,
                make(7 * block),
                (0..65).map(|j| (j, 7.0)).collect(),
                &mut work,
            )
            .unwrap()
        })
        .collect();
    let mut prospective = expected.clone();
    prospective.rows.truncate(1024);
    for cut in &rows {
        let row = cut.owned_row(&s, end()).unwrap();
        prospective.rows.push(HeadFunctionalLpRow {
            coefficients: row.coefficients.iter().map(|&(j, a)| (j, a.lo)).collect(),
            lower: row.lower.lo,
        });
    }
    assert_eq!(prospective.rows.len(), 1028);
    assert_eq!(
        prospective
            .rows
            .iter()
            .map(|r| r.coefficients.len())
            .sum::<usize>(),
        67104
    );
    assert!(!prospective.validate());
    assert!(
        triangle::Model::from_root_portfolio_reserved(&s, &mut prefix, &rows, 4, end()).is_none()
    );
    let (boxes, retained) = prefix.unwrap().into_phase_parts(&s, 1028, end()).unwrap();
    assert_eq!(
        boxes.iter().map(|b| [b.lo, b.hi]).collect::<Vec<_>>(),
        expected.bounds
    );
    assert_eq!(retained.len(), expected.rows.len());
    for (row, exported) in retained.iter().zip(&expected.rows) {
        assert_eq!(row.lower.lo.to_bits(), exported.lower.to_bits());
        assert_eq!(
            row.coefficients
                .iter()
                .map(|&(j, a)| (j, 0.5 * a.lo + 0.5 * a.hi))
                .filter(|&(_, a)| a != 0.0)
                .collect::<Vec<_>>(),
            exported.coefficients
        );
    }
}
