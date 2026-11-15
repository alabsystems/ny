// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::time::Duration;

fn domain() -> BoxDomain {
    BoxDomain {
        lo: vec![-1.0],
        hi: vec![1.0],
    }
}

fn expr(root: &[f64], bias: f64) -> Expr {
    Expr {
        root: root.iter().copied().map(Interval::point).collect(),
        bias: Interval::point(bias),
        slack: Vec::new(),
    }
}

fn bindings(chain: &PairedChain) -> Vec<(Vec<usize>, Vec<usize>)> {
    let input: Vec<_> = (0..chain.input_dim).collect();
    let mut result = vec![(input.clone(), input)];
    let mut next = chain.input_dim;
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

fn trace(chain: &PairedChain, map: &[(Vec<usize>, Vec<usize>)], x: f64) -> Vec<f64> {
    let n = map
        .iter()
        .flat_map(|(f, g)| f.iter().chain(g))
        .copied()
        .max()
        .unwrap()
        + 1;
    let mut values = vec![0.0; n];
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
                    for (&from, &to) in from.iter().zip(to) {
                        values[to] = values[from].max(0.0);
                    }
                }
            }
        }
    }
    values
}

// Exact expansion evaluation of the stored row at dyadic source traces.
// No tolerance is granted to an inequality or to sparse duplicate terms.
fn row_sign(row: &LinearRow, values: &[f64]) -> i8 {
    fn append(e: &mut Vec<f64>, mut q: f64) {
        let mut next = Vec::new();
        for &part in e.iter() {
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
        *e = next;
    }
    let mut e = Vec::new();
    for &(j, a) in &row.terms {
        let product = a * values[j];
        append(&mut e, a.mul_add(values[j], -product));
        append(&mut e, product);
    }
    append(&mut e, -row.rhs);
    if e.last().copied().unwrap_or(0.0) < 0.0 {
        -1
    } else {
        1
    }
}

fn same_rows(a: &[LinearRow], b: &[LinearRow]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert_eq!(a.rhs.to_bits(), b.rhs.to_bits());
        assert_eq!(a.terms.len(), b.terms.len());
        for (&(aj, av), &(bj, bv)) in a.terms.iter().zip(&b.terms) {
            assert_eq!(aj, bj);
            assert_eq!(av.to_bits(), bv.to_bits());
        }
    }
}

fn causal_head() -> PairedChain {
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
                rows: 2,
                cols: 2,
                wf: vec![1.0, -0.75, -0.5, 0.25],
                wg: vec![1.0, -1.25, -0.25, 0.5],
                bf: vec![0.875, -0.25],
                bg: vec![1.875, -0.75],
            },
            Stage::Relu { width: 2 },
            Stage::Affine {
                rows: 1,
                cols: 2,
                wf: vec![1.0, -2.0],
                wg: vec![-1.0, 0.5],
                bf: vec![0.125],
                bg: vec![-0.25],
            },
        ],
    }
}

#[test]
fn output_complete_closed_five_dimensional_cover_retains_the_capped_frontier() {
    let root = BoxDomain {
        lo: vec![-1.0; 5],
        hi: vec![1.0; 5],
    };
    let f = vec![
        expr(&[1.0, 0.0, 0.0, 0.0, 0.0], 0.0),
        expr(&[-1.0, 0.0, 0.0, 0.0, 0.0], 0.0),
    ];
    let g = vec![expr(&[0.0; 5], 0.0), expr(&[0.0; 5], 0.0)];
    let wf = vec![Interval::point(1.0); 2];
    let wg = vec![Interval::point(0.0); 2];
    let (old, _, old_cells) =
        complete_cover(&f, &g, &wf, &wg, Interval::point(0.0), &root, 1, 0.0, None).unwrap();
    let (upper, nodes, cells) =
        complete_cover(&f, &g, &wf, &wg, Interval::point(0.0), &root, 3, 0.0, None).unwrap();
    assert_eq!(old_cells.len(), 1);
    assert_eq!(nodes, 3);
    assert_eq!(cells.len(), 2);
    assert!(upper >= 1.0 && upper < old);
    assert_eq!(cells[0].lo[0], -1.0);
    assert_eq!(cells[0].hi[0], 0.0);
    assert_eq!(cells[1].lo[0], 0.0);
    assert_eq!(cells[1].hi[0], 1.0);
    for n in 0..243usize {
        let mut code = n;
        let x: Vec<_> = (0..5)
            .map(|_| {
                let v = [-1.0, 0.0, 1.0][code % 3];
                code /= 3;
                v
            })
            .collect();
        assert!(cells
            .iter()
            .any(|b| (0..5).all(|j| b.lo[j] <= x[j] && x[j] <= b.hi[j])));
        assert!(x[0].abs() <= upper);
    }
}

#[test]
fn output_stable_terms_are_aggregated_before_box_maximization() {
    // Both minima are strictly positive even after outward rounding. A
    // template touching zero can correctly remain mixed in the enclosure.
    let f = vec![expr(&[1.0], 2.0), expr(&[1.0], 3.0)];
    let g = vec![expr(&[0.0], 0.0), expr(&[0.0], 0.0)];
    let wf = vec![Interval::point(1.0), Interval::point(-1.0)];
    let wg = vec![Interval::point(0.0); 2];
    let upper = cell_upper(&f, &g, &wf, &wg, Interval::point(0.0), &domain(), None).unwrap();
    // The exact whole sum is -1. Independent positive/negative term ranges
    // give +1 on this box, so affine aggregation is a material improvement.
    assert!((-1.0..-0.99).contains(&upper));
    for n in 0..65 {
        let x = -1.0 + n as f64 / 32.0;
        assert!((x + 2.0) - (x + 3.0) <= upper);
    }
}

#[test]
fn output_signed_tail_penalties_cover_traces_outside_every_fixed_mask() {
    let chain = causal_head();
    let map = bindings(&chain);
    for bit in [0, 1] {
        let masks = vec![
            Mask {
                stage: 1,
                f: vec![bit; 2],
                g: vec![1 - bit; 2],
            },
            Mask {
                stage: 3,
                f: vec![bit; 2],
                g: vec![1 - bit; 2],
            },
        ];
        let (_, cuts, covers) =
            rows_with_masks_impl(&chain, &domain(), &map, &masks, true, 0, None).unwrap();
        assert_eq!(cuts.len(), 2);
        assert_eq!(covers.len(), 2);
        for n in 0..65 {
            let values = trace(&chain, &map, -1.0 + n as f64 / 32.0);
            for row in &cuts {
                assert!(row_sign(row, &values) >= 0);
            }
        }
        assert!(cuts.iter().any(|row| row.terms.len() > 2));
    }
}

#[test]
fn output_tail_composition_repairs_lost_unit_and_underflow() {
    let mut chain = PairedChain {
        input_dim: 1,
        output_dim: 1,
        stages: vec![Stage::Relu { width: 1 }],
    };
    for (w, b) in [
        (9007199254740992.0, 0.0),
        (1.0, 1.0),
        (1.0, -9007199254740992.0),
    ] {
        chain.stages.push(Stage::Affine {
            rows: 1,
            cols: 1,
            wf: vec![w],
            wg: vec![0.0],
            bf: vec![b],
            bg: vec![0.0],
        });
    }
    let composed = tail(&chain, 0, 1, false, None).unwrap();
    let value = root_value(
        &composed[0],
        &BoxDomain {
            lo: vec![1.0],
            hi: vec![1.0],
        },
        None,
    )
    .unwrap();
    assert!(value.lo <= 1.0 && value.hi >= 1.0);
    // Nominal sequential floating evaluation loses this authored real unit.
    assert_eq!((9007199254740992.0f64 + 1.0) - 9007199254740992.0, 0.0);
    let tiny = f64::from_bits(1);
    chain.stages.truncate(1);
    for w in [tiny, 0.5] {
        chain.stages.push(Stage::Affine {
            rows: 1,
            cols: 1,
            wf: vec![w],
            wg: vec![0.0],
            bf: vec![0.0],
            bg: vec![0.0],
        });
    }
    let composed = tail(&chain, 0, 1, false, None).unwrap();
    assert!(composed[0].root[0].lo <= 0.0);
    assert!(composed[0].root[0].hi >= tiny);
}

#[test]
fn output_source_mutation_rebuilds_the_row_and_retains_a_real_violation() {
    let mut chain = PairedChain {
        input_dim: 1,
        output_dim: 1,
        stages: vec![
            Stage::Affine {
                rows: 1,
                cols: 1,
                wf: vec![1.0],
                wg: vec![1.0],
                bf: vec![0.0],
                bg: vec![0.0],
            },
            Stage::Relu { width: 1 },
            Stage::Affine {
                rows: 1,
                cols: 1,
                wf: vec![1.0],
                wg: vec![1.0],
                bf: vec![0.0],
                bg: vec![0.0],
            },
        ],
    };
    let map = bindings(&chain);
    let (_, old, _) = rows_with_output(&chain, &domain(), &map, None).unwrap();
    let Stage::Affine { bf, .. } = &mut chain.stages[2] else {
        unreachable!()
    };
    bf[0] = 0.125;
    let (_, new, _) = rows_with_output(&chain, &domain(), &map, None).unwrap();
    let values = trace(&chain, &map, 0.5);
    assert_eq!(values[map[3].0[0]] - values[map[3].1[0]], 0.125);
    assert!(old.iter().any(|row| row_sign(row, &values) < 0));
    assert!(new.iter().all(|row| row_sign(row, &values) >= 0));
}

#[test]
fn output_rejects_expiry_invalid_binding_and_nonfinite_source() {
    let mut chain = causal_head();
    let mut map = bindings(&chain);
    let expired = Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap());
    assert!(rows_with_output(&chain, &domain(), &map, expired).is_none());
    map[0].1[0] += 1;
    assert!(rows_with_output(&chain, &domain(), &map, None).is_none());
    map = bindings(&chain);
    let Stage::Affine { wf, .. } = &mut chain.stages[4] else {
        unreachable!()
    };
    wf[0] = f64::INFINITY;
    assert!(rows_with_output(&chain, &domain(), &map, None).is_none());
}

#[test]
fn output_extension_preserves_the_old_causal_prefix_bit_for_bit() {
    let chain = causal_head();
    let map = bindings(&chain);
    let old = super::super::rows(&chain, &domain(), &map, None).unwrap();
    let (prefix, cuts, covers) = rows_with_output(&chain, &domain(), &map, None).unwrap();
    same_rows(&old, &prefix);
    assert!(!cuts.is_empty() && cuts.len() <= 6);
    assert_eq!(cuts.len(), covers.len());
    for cover in covers {
        assert!(cover.nodes <= MAX_NODES);
        assert_eq!(cover.nodes, 2 * cover.leaves.len() - 1);
        assert!(cover.output == 0 && [-1, 1].contains(&cover.sign) && cover.portfolio < 3);
        assert!(cover.upper.is_finite());
    }
}
