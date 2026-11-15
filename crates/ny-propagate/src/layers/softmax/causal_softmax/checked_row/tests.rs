// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use num_rational::BigRational;

#[test]
fn uniform_coordinate_bounds_enclose_exact_rational_probabilities() {
    for count in [1, 2, 3, 7, 257] {
        let result = bounds(&vec![17.0; count], &vec![17.0; count], &mut || Ok(())).unwrap();
        let truth = BigRational::new(1.into(), count.into());
        for (lo, hi) in result {
            assert!(BigRational::from_float(lo).unwrap() <= truth);
            assert!(truth <= BigRational::from_float(hi).unwrap());
            assert!((hi - lo) as f64 <= 1e-6 / count as f64);
        }
    }
}

#[test]
fn checked_prefix_suffix_coordinates_enclose_all_small_box_vertices_and_corner_optima() {
    for (lower, upper) in [
        ([-1.0, -0.3, -2.0], [0.7, 0.1, 0.4]),
        ([-1000.0, -1000.0, -1000.0], [1000.0, -1000.0, -999.0]),
        ([0.0, -1000.0, -1000.0], [0.0, 1000.0, 1000.0]),
        ([-80.0, -80.0, -80.0], [0.0, -0.0001, -80.0]),
    ] {
        let result = bounds(&lower, &upper, &mut || Ok(())).unwrap();
        for vertex in 0..8 {
            let score: Vec<f64> = (0..3)
                .map(|i| {
                    if vertex & (1 << i) == 0 {
                        lower[i]
                    } else {
                        upper[i]
                    }
                })
                .collect();
            let shift = score.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weight: Vec<_> = score.iter().map(|s| (s - shift).exp()).collect();
            let mass: f64 = weight.iter().sum();
            for i in 0..3 {
                let truth = weight[i] / mass;
                assert!(
                    result[i].0 as f64 <= truth && truth <= result[i].1 as f64,
                    "vertex={vertex},i={i}: {:?} excludes {truth}",
                    result[i]
                );
            }
        }
    }
    // Excluding the unique upper maximum must locally rebase its remaining
    // mass. Otherwise all terms under a global +1000 shift underflow, hiding
    // the reachable exact 1/2 minimum at [-1000,-1000].
    let result = bounds(&[-1000.0, -1000.0], &[1000.0, -1000.0], &mut || Ok(())).unwrap();
    let half = BigRational::new(1.into(), 2.into());
    assert!(BigRational::from_float(result[0].0).unwrap() <= half);
    assert!(result[0].0 > 0.499999);
    assert!(BigRational::from_float(result[1].1).unwrap() >= half);
    assert!(result[1].1 < 0.500001);
}

#[test]
fn excluded_mass_avoids_total_minus_dominant_cancellation() {
    let result = bounds(&[0.0, -80.0, -80.0], &[0.0, -80.0, -80.0], &mut || Ok(())).unwrap();
    assert!(result[1].0 > 0.0 && result[1].1 > 0.0);
    assert!(result[1].1 < 1e-33);
    let shifted = bounds(
        &[1000.0, 920.0, 920.0],
        &[1000.0, 920.0, 920.0],
        &mut || Ok(()),
    )
    .unwrap();
    assert_eq!(result, shifted);
}

#[test]
fn malformed_nonfinite_and_deadline_controls_never_publish_unchecked_exp() {
    assert!(bounds(&[0.0, 0.0], &[0.0], &mut || Ok(())).is_err());
    assert_eq!(
        bounds(
            &[f64::NEG_INFINITY, 0.0],
            &[1.0, f64::INFINITY],
            &mut || Ok(())
        )
        .unwrap(),
        vec![(0.0, 1.0); 2]
    );
    assert_eq!(
        bounds(&[1.0], &[0.0], &mut || Ok(())).unwrap(),
        vec![(0.0, 1.0)]
    );
    let mut polls = 0;
    assert!(matches!(
        bounds(&[0.0; 257], &[1.0; 257], &mut || {
            polls += 1;
            if polls == 8 {
                Err(NyError::DeadlineExceeded("oracle expired".into()))
            } else {
                Ok(())
            }
        }),
        Err(NyError::DeadlineExceeded(_))
    ));
    assert_eq!(polls, 8);
}
