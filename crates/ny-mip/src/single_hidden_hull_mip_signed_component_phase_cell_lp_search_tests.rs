// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Exact limited-pivot controls and the measured v35 cache-admission boundary.
use super::*;
use lp::{simplex::Simplex, SignedTableSupportReport};

fn checked_score(values: &[f64], a: &[f64; 8]) -> certificate::I {
    let k = values.len().trailing_zeros() as usize;
    let lower = repair(values, &a[..k], end(), &mut || true).unwrap();
    for (mask, &value) in values.iter().enumerate() {
        let mut rhs = rat(2.) * rat(value);
        for (j, &coefficient) in a[..k].iter().enumerate() {
            if mask & (1 << j) != 0 {
                rhs += rat(coefficient);
            }
        }
        assert!(rat(lower) <= rhs, "mask={mask}");
    }
    support::score(lower, &a[..k], Some(&[0.5; 8][..k])).unwrap()
}

#[test]
fn table_lp_largest_violation_reaches_exact_optimum_in_one_pivot() {
    let values = [0., 0., -1., 0., -3., 0., 0., 0.];
    let mut s = Simplex::new(&values, &[true; 3], &[0.5; 3], &mut || true).unwrap();
    let mut r = SignedTableSupportReport::default();
    let a = s.run(1, &mut r, &mut || true).unwrap();
    let score = checked_score(&values, &a);
    // Independent exact primal: half mass each on masks 3 and 4 has all
    // three marginals 1/2 and doubled cost -3. The checked dual matches it.
    assert!(rat(score.lo) <= rat(-3.));
    assert!(score.lo > -3. - 1e-12);
    let upper = lp::upper(&s, &mut || true).unwrap();
    assert!(upper >= -3. && upper - score.lo < 1e-10);
    assert_eq!(r.pivots, 1);
    assert_eq!(r.stop_reason, "nominal-optimum");
    assert!(s.basis[..=3].contains(&4));
}

#[test]
fn table_lp_retains_earlier_support_when_terminal_dual_is_worse() {
    let values = [0., 0., -3., 0., -1., 0., 0., 0.];
    let mut s = Simplex::new(&values, &[true; 3], &[0.5; 3], &mut || true).unwrap();
    let mut r = SignedTableSupportReport::default();
    let kept = s.run(1, &mut r, &mut || true).unwrap();
    assert_eq!(r.pivots, 1);
    assert_eq!(r.stop_reason, "pivot-limit");
    assert!(s.basis[..=3].contains(&2) && !s.basis[..=3].contains(&3));
    // A separate zero-pivot extraction from the same terminal basis exposes
    // the discarded proposal; it makes no additional production LP call.
    let terminal = s
        .run(0, &mut SignedTableSupportReport::default(), &mut || true)
        .unwrap();
    assert_eq!(kept, [0.; 8]);
    assert_eq!(&terminal[..3], &[0., 6., -6.]);
    let kept = checked_score(&values, &kept);
    let terminal = checked_score(&values, &terminal);
    assert!(kept.lo > -6. - 1e-12 && kept.lo <= -6.);
    assert!(terminal.lo <= -8. && terminal.hi < -8. + 1e-12);
    assert!(kept.lo > terminal.hi + 1.9);
}

#[test]
fn table_lp_degenerate_fallback_stays_bland_after_a_positive_step() {
    let mut values = [0.; 16];
    values[2] = -3.;
    values[5] = -1.5;
    values[9] = -1.;
    values[11] = -2.5;
    let mut s = Simplex::new(&values, &[true; 4], &[0.5; 4], &mut || true).unwrap();
    let mut r = SignedTableSupportReport::default();
    let a = s.run(4, &mut r, &mut || true).unwrap();
    // Two zero steps enter 2 and 5. Bland then enters 8 with step 1/4,
    // followed by 9 with step zero. Dantzig would choose 9, then 11.
    // The exact terminal basis distinguishes both the trigger and its latch.
    assert_eq!(&s.basis[..5], &[8, 9, 2, 5, 15]);
    assert_eq!(r.pivots, 4);
    assert_eq!(r.stop_reason, "pivot-limit");
    checked_score(&values, &a);
}

#[test]
fn table_lp_compact_scratch_preserves_v35_shaped_child_admission() {
    let (d, h) = (30, 128);
    let p = source(vec![1.; d * h], vec![0.; h], &vec![-1.; h], d);
    let m = HullMultipliers::zeros(&p);
    let floor = certify_hull_lower_bound(&p, &m).unwrap();
    let cuts = root::prepare(&p, m, floor, end(), &mut || true).unwrap();
    // Accounting fixture with the observed component/head/mask census. It is
    // not the benchmark network, a checked signed plan, or a minted cache.
    let mut first = 0;
    let plan = Plan {
        components: [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 8, 8, 8, 8, 8]
            .into_iter()
            .map(|k| {
                let negative = (first..first + k).collect();
                first += k;
                Component {
                    negative,
                    positive: vec![],
                    positive_units: vec![],
                    g: vec![0.; d],
                    alpha: vec![],
                }
            })
            .collect(),
        leftover: vec![],
    };
    assert_eq!(first, 54);
    assert_eq!(plan.masks(), Some(1308));
    let quote = lp::quote(d, h, 8, 62).unwrap();
    assert_eq!(quote.work, 1_280_384);
    let revised = 8_349_904 - 20_352 + quote.bytes;
    assert!(revised + 44_512 <= MAX_BYTES);
    for (bytes, accepted) in [
        (8_349_904, false),
        (revised, true),
        (MAX_BYTES - 44_512, true),
        (MAX_BYTES - 44_512 + 1, false),
    ] {
        let b = Budget::new(&cuts, end(), Duration::ZERO).unwrap();
        b.state.lock().unwrap().shared_bytes = bytes - b.admitted_bytes;
        let collection = child::admit(&cuts, &plan, &b, end());
        assert_eq!(collection.is_some(), accepted, "bytes={bytes}");
        let c = b.state.lock().unwrap();
        assert_eq!(c.child.collection_attempts, 1);
        assert_eq!(c.child.cache_bytes, if accepted { 44_512 } else { 0 });
        assert_eq!(c.child.work_units, if accepted { 24_592 } else { 0 });
        if !accepted {
            assert_eq!(c.child.stop_reason, "collection-admission");
        }
    }
}
