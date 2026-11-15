// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::time::Duration;

fn root() -> BoxDomain {
    BoxDomain {
        lo: vec![-1.0],
        hi: vec![1.0],
    }
}

fn bindings(chain: &PairedChain) -> Vec<(Vec<usize>, Vec<usize>)> {
    let mut result = vec![(vec![0], vec![0])];
    let mut next = 1usize;
    for stage in &chain.stages {
        let width = match stage {
            Stage::Affine { rows, .. } => *rows,
            Stage::Relu { width } => *width,
        };
        let f = (next..next + width).collect();
        next += width;
        let g = (next..next + width).collect();
        next += width;
        result.push((f, g));
    }
    result
}

// Fixtures use dyadic weights and grid inputs, so these source-node traces
// are exact. The cancellation test below explicitly authors its real trace.
fn trace(chain: &PairedChain, map: &[(Vec<usize>, Vec<usize>)], x: f64) -> Vec<f64> {
    let variables = map
        .iter()
        .flat_map(|(f, g)| f.iter().chain(g))
        .copied()
        .max()
        .unwrap()
        + 1;
    let mut values = vec![0.0; variables];
    values[0] = x;
    for (s, stage) in chain.stages.iter().enumerate() {
        for g in [false, true] {
            let from = if g { &map[s].1 } else { &map[s].0 };
            let to = if g { &map[s + 1].1 } else { &map[s + 1].0 };
            match stage {
                Stage::Affine {
                    rows,
                    cols,
                    wf,
                    wg,
                    bf,
                    bg,
                } => {
                    let weights = if g { wg } else { wf };
                    let bias = if g { bg } else { bf };
                    for i in 0..*rows {
                        let mut value = bias[i];
                        for j in 0..*cols {
                            value += weights[i * cols + j] * values[from[j]];
                        }
                        values[to[i]] = value;
                    }
                }
                Stage::Relu { .. } => {
                    for (&input, &output) in from.iter().zip(to) {
                        values[output] = values[input].max(0.0);
                    }
                }
            }
        }
    }
    values
}

// Exact expansion sum of the stored binary64 row evaluated at these dyadic
// traces. It avoids granting a tolerance to a tight GEQ or to duplicate terms.
fn row_sign(row: &LinearRow, values: &[f64]) -> i8 {
    fn append(expansion: &mut Vec<f64>, value: f64) {
        let mut q = value;
        let mut next = Vec::with_capacity(expansion.len() + 1);
        for &part in expansion.iter() {
            let sum = q + part;
            let b = sum - q;
            let error = (q - (sum - b)) + (part - b);
            if error != 0.0 {
                next.push(error);
            }
            q = sum;
        }
        if q != 0.0 {
            next.push(q);
        }
        *expansion = next;
    }
    let mut expansion = Vec::new();
    for &(column, coefficient) in &row.terms {
        let product = coefficient * values[column];
        let error = coefficient.mul_add(values[column], -product);
        append(&mut expansion, error);
        append(&mut expansion, product);
    }
    append(&mut expansion, -row.rhs);
    match expansion.last().copied().unwrap_or(0.0).partial_cmp(&0.0) {
        Some(std::cmp::Ordering::Less) => -1,
        Some(std::cmp::Ordering::Equal) => 0,
        Some(std::cmp::Ordering::Greater) => 1,
        None => panic!("nonfinite test row"),
    }
}

fn causal_chain() -> PairedChain {
    PairedChain {
        input_dim: 1,
        output_dim: 1,
        stages: vec![
            Stage::Affine {
                rows: 3,
                cols: 1,
                wf: vec![1.0, -2.0, 1.0],
                wg: vec![2.0, -1.0, 1.0],
                bf: vec![0.5, 0.5, 2.0],
                bg: vec![0.5, 0.5, 2.0],
            },
            Stage::Relu { width: 3 },
            Stage::Affine {
                rows: 1,
                cols: 3,
                wf: vec![2.0, 1.0, 1.0],
                wg: vec![1.0, 2.0, 1.0],
                bf: vec![-3.875],
                bg: vec![-3.875],
            },
            Stage::Relu { width: 1 },
        ],
    }
}

fn causal_masks() -> Vec<Mask> {
    vec![
        Mask {
            stage: 1,
            f: vec![1; 3],
            g: vec![1; 3],
        },
        Mask {
            stage: 3,
            f: vec![0],
            g: vec![0],
        },
    ]
}

#[test]
fn causal_template_excludes_an_exact_local_product_hull_lift() {
    let chain = causal_chain();
    let map = bindings(&chain);
    let cuts = rows_with_masks(&chain, &root(), &map, &causal_masks(), None).unwrap();
    let mut lifted = trace(&chain, &map, 0.0);
    assert_eq!(
        map[2].0.iter().map(|&i| lifted[i]).collect::<Vec<_>>(),
        vec![0.5, 0.5, 2.0]
    );
    assert_eq!(lifted[map[3].0[0]], -0.375);
    assert_eq!(lifted[map[3].1[0]], -0.375);

    // The complete LOCAL affine-input product hull admits this false lift.
    // Equal-weight predecessor vectors (.75,.5,2) and (.25,.5,2), in BOTH
    // towers, have delta zero in every predecessor polygon. Their final
    // pre pairs are (.125,-.125) and (-.875,-.625), inside the exact source
    // final boxes F[-.875,2.125], G[-.625,1.625], delta[-.5,.5]. Their exact
    // ReLU graphs mix to the actual center prefix plus final post (1/16,0).
    // This proves local-hull membership, not feasibility of every old LP row.
    for first in [0.25, 0.75] {
        let f = 2.0 * first + 0.5 + 2.0 - 3.875;
        let g = first + 2.0 * 0.5 + 2.0 - 3.875;
        assert!((-0.875..=2.125).contains(&f));
        assert!((-0.625..=1.625).contains(&g));
        assert!((-0.5..=0.5).contains(&(f - g)));
    }
    assert_eq!((0.125 - 0.875) * 0.5, -0.375);
    assert_eq!((-0.125 - 0.625) * 0.5, -0.375);
    lifted[map[4].0[0]] = 0.0625;
    lifted[map[4].1[0]] = 0.0;
    let upper = &cuts[6]; // Three first-layer neurons, then final upper/lower.
    assert!(upper.rhs > -0.001);
    assert_eq!(row_sign(upper, &lifted), -1);
    assert!(upper.terms.iter().any(|&(i, _)| map[1].0.contains(&i)));
    assert!(upper.terms.iter().any(|&(i, _)| map[2].0.contains(&i)));
}

#[test]
fn causal_rows_contain_traces_on_both_sides_of_the_chosen_masks() {
    let chain = causal_chain();
    let map = bindings(&chain);
    let cuts = rows_with_masks(&chain, &root(), &map, &causal_masks(), None).unwrap();
    for i in -32..=32 {
        let x = i as f64 / 32.0;
        let values = trace(&chain, &map, x);
        for row in &cuts {
            assert!(row_sign(row, &values) >= 0, "source x={x}: {row:?}");
        }
    }
    // At these endpoints, different prior preactivations are negative while
    // their chosen template bits remain1. Validity cannot assume the pattern.
    assert!(trace(&chain, &map, -1.0)[map[1].0[0]] < 0.0);
    assert!(trace(&chain, &map, 1.0)[map[1].0[1]] < 0.0);
}

#[test]
fn shared_mask_ignores_a_negative_mismatched_g_slack_unconditionally() {
    let chain = PairedChain {
        input_dim: 1,
        output_dim: 1,
        stages: vec![
            Stage::Affine {
                rows: 1,
                cols: 1,
                wf: vec![1.0],
                wg: vec![1.0],
                bf: vec![0.25],
                bg: vec![-0.25],
            },
            Stage::Relu { width: 1 },
            Stage::Affine {
                rows: 1,
                cols: 1,
                wf: vec![1.0],
                wg: vec![1.0],
                bf: vec![-0.25],
                bg: vec![0.25],
            },
            Stage::Relu { width: 1 },
        ],
    };
    let map = bindings(&chain);
    let mut masks = vec![
        Mask {
            stage: 1,
            f: vec![1],
            g: vec![1],
        },
        Mask {
            stage: 3,
            f: vec![0],
            g: vec![1],
        },
    ];
    let shared = rows_with_masks(&chain, &root(), &map, &masks, None).unwrap();
    masks[0].g[0] = 0;
    let own = rows_with_masks(&chain, &root(), &map, &masks, None).unwrap();
    assert!(shared[2].rhs > -0.001);
    assert!(own[2].rhs < -0.7);
    assert!(!shared[2]
        .terms
        .iter()
        .any(|&(i, _)| i == map[2].1[0] || i == map[1].1[0]));
    let center = trace(&chain, &map, 0.0);
    assert_eq!(center[map[2].1[0]] - center[map[1].1[0]], 0.25);
    for i in -32..=32 {
        let values = trace(&chain, &map, i as f64 / 32.0);
        for row in shared.iter().chain(&own) {
            assert!(row_sign(row, &values) >= 0);
        }
    }
}

#[test]
fn template_interval_fold_retains_a_lost_unit_coefficient() {
    let large = 9_007_199_254_740_992.0;
    assert_eq!((large + 1.0) - large, 0.0);
    let chain = PairedChain {
        input_dim: 1,
        output_dim: 1,
        stages: vec![
            Stage::Affine {
                rows: 3,
                cols: 1,
                wf: vec![1.0; 3],
                wg: vec![1.0; 3],
                bf: vec![2.0; 3],
                bg: vec![2.0; 3],
            },
            Stage::Relu { width: 3 },
            Stage::Affine {
                rows: 1,
                cols: 3,
                wf: vec![large, 1.0, -large],
                wg: vec![0.0; 3],
                bf: vec![0.0],
                bg: vec![0.0],
            },
            Stage::Relu { width: 1 },
        ],
    };
    let map = bindings(&chain);
    let cuts = rows_with_masks(&chain, &root(), &map, &causal_masks(), None).unwrap();
    let mut real = trace(&chain, &map, 0.0);
    // Real authored arithmetic is (2^53+1-2^53)*(x+2)=x+2.
    real[map[3].0[0]] = 2.0;
    real[map[4].0[0]] = 2.0;
    assert!(cuts[6].rhs < -1.0);
    for row in &cuts {
        assert!(row_sign(row, &real) >= 0);
    }
}

#[test]
fn source_mutation_masks_and_expiry_do_not_mint_a_zero_band() {
    let mut chain = causal_chain();
    let Stage::Affine { bf, .. } = &mut chain.stages[2] else {
        unreachable!()
    };
    bf[0] += 1.0;
    let map = bindings(&chain);
    let masks = causal_masks();
    let cuts = rows_with_masks(&chain, &root(), &map, &masks, None).unwrap();
    let values = trace(&chain, &map, 0.0);
    assert_eq!(values[map[4].0[0]] - values[map[4].1[0]], 0.625);
    for row in &cuts {
        assert!(row_sign(row, &values) >= 0);
    }
    assert!(cuts[6].rhs < -0.99);
    let mut malformed = causal_masks();
    malformed[0].g[0] = 2;
    assert!(rows_with_masks(&chain, &root(), &map, &malformed, None).is_none());
    let expired = Instant::now()
        .checked_sub(Duration::from_millis(1))
        .unwrap();
    assert!(rows_with_masks(&chain, &root(), &map, &masks, Some(expired)).is_none());
    assert!(rows(&chain, &root(), &map, Some(expired)).is_none());
}

fn inactive_root_template_chain() -> PairedChain {
    PairedChain {
        input_dim: 1,
        output_dim: 1,
        stages: vec![
            Stage::Affine {
                rows: 2,
                cols: 1,
                wf: vec![1.0, 1.0],
                wg: vec![1.0, 1.0],
                bf: vec![0.125, 2.0],
                bg: vec![0.125, 2.0],
            },
            Stage::Relu { width: 2 },
            Stage::Affine {
                rows: 1,
                cols: 2,
                wf: vec![1.0, -0.75],
                wg: vec![1.0, -1.25],
                bf: vec![0.875],
                bg: vec![1.875],
            },
            Stage::Relu { width: 1 },
        ],
    }
}

fn inactive_root_template_masks() -> Vec<Mask> {
    vec![
        Mask {
            stage: 1,
            f: vec![1; 2],
            g: vec![1; 2],
        },
        Mask {
            stage: 3,
            f: vec![0],
            g: vec![0],
        },
    ]
}

#[test]
fn paired_root_inactivity_strictly_strengthens_both_causal_intercepts() {
    let chain = inactive_root_template_chain();
    let map = bindings(&chain);
    let masks = inactive_root_template_masks();
    let cuts = rows_with_masks(&chain, &root(), &map, &masks, None).unwrap();
    assert_eq!(cuts.len(), 6);
    // Own/shared-F/shared-G source-center masks coincide for this chain.
    // Production portfolio deduplication preserves exactly this one template.
    let production = rows(&chain, &root(), &map, None).unwrap();
    assert_eq!(production.len(), cuts.len());
    for (a, b) in production.iter().zip(&cuts) {
        assert_eq!(a.terms, b.terms);
        assert_eq!(a.rhs.to_bits(), b.rhs.to_bits());
    }

    // Under either fixed active first-layer mask:
    // HF=.25*x-.5 and HG=-.25*x-.5 each range[-.75,-.25], while
    // HF-HG=.5*x ranges[-.5,.5]. The earlier delta-root clipping can
    // therefore give only +/- .5 before slack penalties. Independent
    // paired-root ReLU boxes certify rho(HF)-rho(HG)=0 globally.
    let upper = &cuts[4];
    let lower = &cuts[5];
    assert_eq!(upper.rhs, 0.0);
    assert_eq!(lower.rhs, 0.0);
    let old_upper = LinearRow {
        terms: upper.terms.clone(),
        rhs: -0.5,
    };
    let old_lower = LinearRow {
        terms: lower.terms.clone(),
        rhs: -0.5,
    };
    assert!(upper.rhs > old_upper.rhs && lower.rhs > old_lower.rhs);
    let center = trace(&chain, &map, 0.0);
    assert_eq!(center[map[3].0[0]], -0.5);
    assert_eq!(center[map[3].1[0]], -0.5);
    for f_lift in [true, false] {
        let mut lifted = center.clone();
        lifted[if f_lift { map[4].0[0] } else { map[4].1[0] }] = 0.125;
        assert!(row_sign(&old_upper, &lifted) >= 0);
        assert!(row_sign(&old_lower, &lifted) >= 0);
        assert_eq!(row_sign(if f_lift { upper } else { lower }, &lifted), -1);
    }
    // This establishes strict scalar-template-row dominance, without a
    // claim that either lift satisfies every other old production LP row.
}

#[test]
fn paired_root_rows_compensate_out_of_mask_traces_and_reject_stale_authority() {
    let chain = inactive_root_template_chain();
    let map = bindings(&chain);
    let masks = inactive_root_template_masks();
    let cuts = rows_with_masks(&chain, &root(), &map, &masks, None).unwrap();
    for i in -32..=32 {
        let x = i as f64 / 32.0;
        let values = trace(&chain, &map, x);
        for row in &cuts {
            assert!(row_sign(row, &values) >= 0, "x={x}: {row:?}");
        }
    }
    let outside = trace(&chain, &map, -1.0);
    assert_eq!(outside[map[1].0[0]], -0.875);
    assert_eq!(outside[map[2].0[0]], 0.0);
    assert_eq!(outside[map[4].0[0]], 0.125);
    assert_eq!(outside[map[4].1[0]], 0.625);
    // The positive G slack penalty in the lower row is essential; merely
    // treating the chosen active template as a phase would wrongly assert0.
    assert!(cuts[5]
        .terms
        .iter()
        .any(|&(column, coefficient)| column == map[2].1[0] && coefficient > 0.0));

    for g in [false, true] {
        let mut mutated = inactive_root_template_chain();
        let Stage::Affine { bf, bg, .. } = &mut mutated.stages[2] else {
            unreachable!()
        };
        if g {
            bg[0] += 0.75;
        } else {
            bf[0] += 0.75;
        }
        let rows = rows_with_masks(&mutated, &root(), &map, &masks, None).unwrap();
        let values = trace(&mutated, &map, 0.0);
        assert_eq!(
            values[map[4].0[0]] - values[map[4].1[0]],
            if g { -0.25 } else { 0.25 }
        );
        for row in &rows {
            assert!(row_sign(row, &values) >= 0);
        }
        assert!(rows[if g { 5 } else { 4 }].rhs < -0.24);
    }
    let mut wrong_binding = map.clone();
    wrong_binding[0].1[0] = 1;
    assert!(rows_with_masks(&chain, &root(), &wrong_binding, &masks, None).is_none());
    let mut incomplete_binding = map.clone();
    incomplete_binding.pop();
    assert!(rows_with_masks(&chain, &root(), &incomplete_binding, &masks, None).is_none());
    let expired = Instant::now()
        .checked_sub(Duration::from_millis(1))
        .unwrap();
    assert!(rows_with_masks(&chain, &root(), &map, &masks, Some(expired)).is_none());
}
