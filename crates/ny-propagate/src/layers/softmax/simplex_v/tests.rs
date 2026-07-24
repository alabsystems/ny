// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use ndarray::{Array2, ArrayD, IxDyn};
use proptest::prelude::*;

/// Brute-force `max sum_k p_k a_k` over `{pl<=p<=ph, sum p=1}` by enumerating the
/// vertices (fix all-but-one coord to a box endpoint; the free coord absorbs the
/// residual to make sum=1, if feasible).
fn brute_lp_max(pl: &[f32], ph: &[f32], a: &[f32]) -> Option<f64> {
    let n = pl.len();
    if n == 0 || n > 8 {
        return None;
    }
    let s0: f64 = pl.iter().map(|&x| x as f64).sum();
    let smax: f64 = ph.iter().map(|&x| x as f64).sum();
    if s0 > 1.0 + 1e-6 || smax < 1.0 - 1e-6 {
        return None;
    }
    let mut best = f64::NEG_INFINITY;
    for free in 0..n {
        let others: Vec<usize> = (0..n).filter(|&i| i != free).collect();
        for mask in 0u32..(1u32 << others.len()) {
            let mut p = vec![0.0f64; n];
            let mut sum_others = 0.0f64;
            for (bit, &idx) in others.iter().enumerate() {
                let hi = (mask >> bit) & 1 == 1;
                p[idx] = if hi { ph[idx] as f64 } else { pl[idx] as f64 };
                sum_others += p[idx];
            }
            let pf = 1.0 - sum_others;
            if pf < pl[free] as f64 - 1e-9 || pf > ph[free] as f64 + 1e-9 {
                continue;
            }
            p[free] = pf;
            best = best.max((0..n).map(|k| p[k] * a[k] as f64).sum());
        }
    }
    best.is_finite().then_some(best)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(400))]

    /// Water-filling LP matches the brute-force vertex LP (exactness + the
    /// soundness direction `wf >= brute`).
    #[test]
    fn simplex_lp_matches_brute_force(
        pls in proptest::collection::vec(0.0f32..0.5, 2..7),
        widths in proptest::collection::vec(0.0f32..0.5, 2..7),
        coeffs in proptest::collection::vec(-3.0f32..3.0, 2..7),
    ) {
        let n = pls.len().min(widths.len()).min(coeffs.len());
        let pl: Vec<f32> = pls[..n].to_vec();
        let ph: Vec<f32> = (0..n).map(|i| (pl[i] + widths[i]).min(1.0)).collect();
        let a: Vec<f32> = coeffs[..n].to_vec();
        if let (Some(wf), Some(bf)) = (simplex_lp_max(&pl, &ph, &a), brute_lp_max(&pl, &ph, &a)) {
            prop_assert!((wf - bf).abs() <= 1e-4 + 1e-4 * bf.abs());
            prop_assert!(wf >= bf - 1e-4, "water-fill {wf} below brute max {bf}");
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(150))]

    /// End-to-end soundness vs Monte-Carlo: the tightened `P @ V` bound encloses
    /// every true `softmax(logits) @ V` for logits/V sampled from their boxes.
    #[test]
    fn tighten_softmax_v_is_sound_vs_monte_carlo(
        seed in 0u64..100_000,
        seq in 2usize..5,
        kdim in 2usize..5,
        ndim in 1usize..4,
        logit_r in 0.1f32..3.0,
        v_lo in -2.0f32..0.0,
        v_w in 0.1f32..2.0,
        batch in 1usize..3,
    ) {
        use crate::layers::common::BoundPropagation;
        use crate::layers::{MatMulLayer, SoftmaxLayer};
        use rand::{RngExt, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        // Logits box (batch, seq, kdim).
        let mut llo = ArrayD::zeros(IxDyn(&[batch, seq, kdim]));
        let mut lhi = ArrayD::zeros(IxDyn(&[batch, seq, kdim]));
        for b in 0..batch { for i in 0..seq { for k in 0..kdim {
            let c: f32 = rng.random_range(-1.0..1.0);
            llo[[b, i, k]] = c - logit_r;
            lhi[[b, i, k]] = c + logit_r;
        }}}
        let logits = BoundedTensor::new(llo.clone(), lhi.clone()).unwrap();

        // V box (batch, kdim, ndim).
        let mut vlo = ArrayD::zeros(IxDyn(&[batch, kdim, ndim]));
        let mut vhi = ArrayD::zeros(IxDyn(&[batch, kdim, ndim]));
        for b in 0..batch { for a in 0..kdim { for c in 0..ndim {
            let lo = v_lo + rng.random_range(0.0..1.0);
            vlo[[b, a, c]] = lo;
            vhi[[b, a, c]] = lo + v_w;
        }}}
        let vval = BoundedTensor::new(vlo.clone(), vhi.clone()).unwrap();

        let probs = SoftmaxLayer::new(-1).propagate_ibp(&logits).unwrap();
        let out_ibp = MatMulLayer::new(false, None)
            .propagate_ibp_binary(&probs, &vval)
            .unwrap();
        let tightened = tighten_softmax_v_ibp(&probs, &vval, &out_ibp, false);

        let (tl, tu) = tightened.lower_upper();
        for _ in 0..120 {
            // sample
            let mut p_all = Vec::new();
            let mut v_all = Vec::new();
            for b in 0..batch {
                let mut ls = Array2::<f32>::zeros((seq, kdim));
                for i in 0..seq { for k in 0..kdim {
                    ls[[i,k]] = rng.random_range(llo[[b,i,k]]..=lhi[[b,i,k]]);
                }}
                // softmax rows
                for mut row in ls.rows_mut() {
                    let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut s = 0.0f32;
                    for x in row.iter_mut() { *x = (*x - mx).exp(); s += *x; }
                    for x in row.iter_mut() { *x /= s; }
                }
                let mut vs = Array2::<f32>::zeros((kdim, ndim));
                for a in 0..kdim { for c in 0..ndim {
                    vs[[a,c]] = rng.random_range(vlo[[b,a,c]]..=vhi[[b,a,c]]);
                }}
                p_all.push(ls);
                v_all.push(vs);
            }
            for b in 0..batch {
                let y = p_all[b].dot(&v_all[b]);
                for ((i,j), &val) in y.indexed_iter() {
                    let lo = tl[[b,i,j]];
                    let hi = tu[[b,i,j]];
                    prop_assert!(
                        val >= lo - 1e-3 && val <= hi + 1e-3,
                        "UNSOUND: true {val} outside [{lo},{hi}] at b={b} ({i},{j})"
                    );
                }
            }
        }
    }
}

/// Tighten-or-equal: the tightened bound is never wider than the term-wise IBP,
/// and strictly tighter somewhere for wide logits/V.
#[test]
fn tighten_softmax_v_never_widens_and_fires() {
    use crate::layers::common::BoundPropagation;
    use crate::layers::{MatMulLayer, SoftmaxLayer};
    let (seq, kdim, ndim) = (4, 4, 3);
    let logits = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[seq, kdim]), -2.0f32),
        ArrayD::from_elem(IxDyn(&[seq, kdim]), 2.0f32),
    )
    .unwrap();
    let probs = SoftmaxLayer::new(-1).propagate_ibp(&logits).unwrap();
    let vval = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[kdim, ndim]), -1.0f32),
        ArrayD::from_elem(IxDyn(&[kdim, ndim]), 1.0f32),
    )
    .unwrap();
    let out_ibp = MatMulLayer::new(false, None)
        .propagate_ibp_binary(&probs, &vval)
        .unwrap();
    let tightened = tighten_softmax_v_ibp(&probs, &vval, &out_ibp, false);

    let (il, iu) = out_ibp.lower_upper();
    let (tl, tu) = tightened.lower_upper();
    let mut fired = false;
    for idx in 0..il.len() {
        let (il, iu) = (il.as_slice().unwrap()[idx], iu.as_slice().unwrap()[idx]);
        let (tl, tu) = (tl.as_slice().unwrap()[idx], tu.as_slice().unwrap()[idx]);
        assert!(tl >= il - 1e-4, "lower widened: {tl} < {il}");
        assert!(tu <= iu + 1e-4, "upper widened: {tu} > {iu}");
        if (tu - tl) < (iu - il) - 1e-4 {
            fired = true;
        }
    }
    assert!(fired, "simplex tightening must fire for wide logits/V");
}

/// Shape-mismatch / unsupported inputs return a clone (safe no-op).
#[test]
fn tighten_softmax_v_noop_on_mismatch() {
    let probs = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[3, 4]), 0.25f32),
        ArrayD::from_elem(IxDyn(&[3, 4]), 0.25f32),
    )
    .unwrap();
    // V with wrong contraction dim.
    let vval = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[5, 2]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[5, 2]), 1.0f32),
    )
    .unwrap();
    let out = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[3, 2]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[3, 2]), 1.0f32),
    )
    .unwrap();
    let r = tighten_softmax_v_ibp(&probs, &vval, &out, false);
    // Unchanged.
    assert_eq!(r.lower(), out.lower());
    assert_eq!(r.upper(), out.upper());
}
