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

fn expr(root: f64, bias: f64, slack: &[f64]) -> Expr {
    Expr {
        root: vec![Interval::point(root)],
        bias: Interval::point(bias),
        slack: slack.iter().copied().map(Interval::point).collect(),
    }
}

fn chain(g: bool, bias: f64) -> PairedChain {
    PairedChain {
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
                wf: vec![if g { 0.0 } else { -1.0 }],
                wg: vec![if g { -1.0 } else { 0.0 }],
                bf: vec![if g { 0.0 } else { bias }],
                bg: vec![if g { bias } else { 0.0 }],
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
    let mut value = vec![0.0; n];
    value[0] = x;
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
                    for i in 0..*rows {
                        value[to[i]] = if g { bg[i] } else { bf[i] };
                        for j in 0..*cols {
                            value[to[i]] += if g {
                                wg[i * cols + j]
                            } else {
                                wf[i * cols + j]
                            } * value[from[j]];
                        }
                    }
                }
                Stage::Relu { .. } => {
                    for (&a, &b) in from.iter().zip(to) {
                        value[b] = value[a].max(0.0);
                    }
                }
            }
        }
    }
    value
}

// Exact expansion sign, including duplicate sparse columns and product
// residuals. Test source traces are dyadic and all source arithmetic exact.
fn row_sign(row: &LinearRow, values: &[f64]) -> i8 {
    fn append(parts: &mut Vec<f64>, mut q: f64) {
        let mut next = Vec::new();
        for &part in parts.iter() {
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
        *parts = next;
    }
    let mut parts = Vec::new();
    for &(j, a) in &row.terms {
        let product = a * values[j];
        append(&mut parts, a.mul_add(values[j], -product));
        append(&mut parts, product);
    }
    append(&mut parts, -row.rhs);
    if parts.last().copied().unwrap_or(0.0) < 0.0 {
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
        for (&(j, x), &(k, y)) in a.terms.iter().zip(&b.terms) {
            assert_eq!(j, k);
            assert_eq!(x.to_bits(), y.to_bits());
        }
    }
}

#[test]
fn both_mask_guards_cover_actual_opposite_phases_and_negative_coefficients() {
    for mask in [0u8, 1] {
        let source = vec![expr(2.0, -0.25, &[-2.0])];
        let mut guards = Vec::new();
        collect_guards(&source, &[mask], false, &mut guards, &mut 0usize, None).unwrap();
        let guard = &guards[0];
        assert_eq!(guard.positive[0], if mask == 0 { 2.0 } else { 0.0 });
        assert_eq!(guard.positive[1], 1.0);
        let eta = if mask == 1 { 1.0 } else { -1.0 };
        for i in -16..=16 {
            for k in 0..=8 {
                let x = i as f64 / 16.0;
                let previous = k as f64 / 8.0;
                let h = 2.0 * x - 0.25;
                let z = h - 2.0 * previous;
                let own = z.max(0.0) - mask as f64 * z;
                assert!(own >= 0.0);
                let identity = eta * h - 2.0 * eta * previous + own;
                assert!(identity >= 0.0);
                let positive = guard.positive[0] * previous + own;
                assert!(positive >= -eta * h);
            }
        }
    }
    let mut guards = Vec::new();
    assert!(collect_guards(
        &[expr(1.0, 0.0, &[])],
        &[2],
        false,
        &mut guards,
        &mut 0,
        None
    )
    .is_none());
}

#[test]
fn beta_is_replayed_downward_and_subnormal_payment_cannot_disappear() {
    assert!(!pays(1.0, 0.0, -1.0, 1.0_f64.next_down()));
    assert!(!pays(1.0, 0.0, -1.0, -1.0));
    assert!(!pays(1.0, 0.0, 0.0, 2.0));
    assert!(!pays(1.0, 0.0, -1.0, f64::NAN));
    let beta = repaired_beta(1.0, 0.0, -1.0).unwrap();
    assert!(pays(1.0, 0.0, -1.0, beta));
    let tiny = f64::from_bits(1);
    assert!(!pays(tiny, 0.0, -1.0, tiny));
    let repaired = repaired_beta(tiny, 0.0, -1.0).unwrap();
    assert!(pays(tiny, 0.0, -1.0, repaired));
    assert!(repaired_beta(1.0, 0.0, -tiny).is_none());
    assert!(repaired_beta(1.0, 1.0, -1.0).is_none());
}

fn toy_cover(
    max_nodes: usize,
    deadline: Option<Instant>,
) -> Option<(Vec<f64>, usize, Vec<SoftGuardLeaf>)> {
    let f = vec![expr(-1.0, 0.0, &[-1.0])];
    let g = vec![expr(0.0, 0.0, &[])];
    let mut guards = Vec::new();
    collect_guards(
        &[expr(1.0, 0.0, &[])],
        &[1],
        false,
        &mut guards,
        &mut 0,
        None,
    )?;
    charged_cover(
        &f,
        &g,
        &[Interval::point(1.0)],
        &[Interval::point(0.0)],
        Interval::point(0.0),
        &guards,
        1,
        0,
        &domain(),
        max_nodes,
        TARGET,
        deadline,
    )
}

#[test]
fn complete_closed_cover_charges_every_leaf_and_maxes_all_penalties() {
    let (extra, nodes, leaves) = toy_cover(MAX_NODES, None).unwrap();
    assert_eq!(nodes, 2 * leaves.len() - 1);
    assert!(leaves.iter().any(|c| c.guard.is_some()));
    assert!(leaves.iter().any(|c| c.guard.is_none()));
    let mut boxes: Vec<_> = leaves
        .iter()
        .map(|c| (c.domain.lo[0], c.domain.hi[0]))
        .collect();
    boxes.sort_by(|a, b| a.0.total_cmp(&b.0));
    assert_eq!(boxes.first().unwrap().0, -1.0);
    assert_eq!(boxes.last().unwrap().1, 1.0);
    for pair in boxes.windows(2) {
        assert_eq!(pair[0].1, pair[1].0);
    }
    for leaf in &leaves {
        if let Some(i) = leaf.guard {
            assert_eq!(i, 0);
            let gu = leaf.guard_root_upper.unwrap();
            assert!(pays(leaf.root_upper, TARGET, gu, leaf.beta));
            assert!(extra[0] >= mul_up(leaf.beta, 1.0));
            assert!(!pays(leaf.root_upper, TARGET, gu, 0.0));
        } else {
            assert!(leaf.root_upper <= TARGET);
            assert_eq!(leaf.beta, 0.0);
            assert!(leaf.guard_root_upper.is_none());
        }
    }
    for i in -64..=64 {
        let x = i as f64 / 64.0;
        let e = x.max(0.0) - x;
        assert!((-x).max(0.0) <= TARGET + extra[0] * e);
    }
}

#[test]
fn unpaid_cap_deadline_singleton_and_bad_width_never_return_partial_cover() {
    assert!(toy_cover(0, None).is_none());
    assert!(toy_cover(1, None).is_none());
    assert!(toy_cover(MAX_NODES + 1, None).is_none());
    assert!(toy_cover(
        MAX_NODES,
        Some(
            Instant::now()
                .checked_sub(Duration::from_millis(1))
                .unwrap()
        )
    )
    .is_none());
    let root = BoxDomain {
        lo: vec![0.0],
        hi: vec![0.0],
    };
    let f = vec![expr(0.0, 1.0, &[])];
    let g = vec![expr(0.0, 0.0, &[])];
    assert!(charged_cover(
        &f,
        &g,
        &[Interval::point(1.0)],
        &[Interval::point(0.0)],
        Interval::point(0.0),
        &[],
        0,
        0,
        &root,
        MAX_NODES,
        TARGET,
        None
    )
    .is_none());
    let bad = Guard {
        root: expr(0.0, -1.0, &[]),
        g: false,
        positive: vec![1.0, 1.0],
        magnitude: 2.0,
    };
    assert!(charged_cover(
        &f,
        &g,
        &[Interval::point(1.0)],
        &[Interval::point(0.0)],
        Interval::point(0.0),
        &[bad],
        1,
        0,
        &domain(),
        MAX_NODES,
        TARGET,
        None
    )
    .is_none());
}

#[test]
fn appended_own_guard_rows_preserve_old_bits_and_cover_both_objective_signs() {
    for g in [false, true] {
        let chain = chain(g, 0.0);
        let map = bindings(&chain);
        let old = rows_with_output(&chain, &domain(), &map, None).unwrap();
        let new = rows_with_output_and_guards(&chain, &domain(), &map, None).unwrap();
        same_rows(&old.0, &new.0);
        same_rows(&old.1, &new.1);
        assert_eq!(old.2.len(), new.3.len());
        for (a, b) in old.2.iter().zip(&new.3) {
            assert_eq!(
                (a.output, a.sign, a.portfolio, a.nodes),
                (b.output, b.sign, b.portfolio, b.nodes)
            );
            assert_eq!(a.upper.to_bits(), b.upper.to_bits());
            assert_eq!(a.leaves.len(), b.leaves.len());
            for (a, b) in a.leaves.iter().zip(&b.leaves) {
                assert_eq!(a.lo, b.lo);
                assert_eq!(a.hi, b.hi);
            }
        }
        assert_eq!(new.2.len(), 1);
        assert_eq!(new.4.len(), 1);
        let cover = &new.4[0];
        assert_eq!(cover.sign, if g { -1 } else { 1 });
        assert_eq!(cover.portfolio, 2);
        assert_eq!(cover.upper.to_bits(), TARGET.to_bits());
        assert!(cover.leaves.iter().all(|c| c.guard.is_none_or(|j| j < 2)));
        for i in -64..=64 {
            let values = trace(&chain, &map, i as f64 / 64.0);
            for row in new.0.iter().chain(&new.1).chain(&new.2) {
                assert_eq!(row_sign(row, &values), 1);
            }
        }
        let mut lift = trace(&chain, &map, 0.0);
        let output = map.last().unwrap();
        lift[if g { output.1[0] } else { output.0[0] }] = 0.125;
        assert_eq!(row_sign(&new.2[0], &lift), -1);
    }
}

#[test]
fn source_mutation_invalid_mask_binding_and_expiry_cannot_reuse_a_guard_proof() {
    let original = chain(false, 0.0);
    let map = bindings(&original);
    let certified = rows_with_output_and_guards(&original, &domain(), &map, None).unwrap();
    let changed = chain(false, 0.25);
    let point = trace(&changed, &map, 0.0);
    assert_eq!(row_sign(&certified.2[0], &point), -1);
    let replay = rows_with_output_and_guards(&changed, &domain(), &map, None).unwrap();
    assert!(replay.2.is_empty());
    for i in -32..=32 {
        let values = trace(&changed, &map, i as f64 / 32.0);
        for row in replay.0.iter().chain(&replay.1).chain(&replay.2) {
            assert_eq!(row_sign(row, &values), 1);
        }
    }
    let mut masks = center_masks(&original, &domain(), None).unwrap();
    masks[0].f[0] = 2;
    assert!(rows(&original, &domain(), &map, &masks, 0, &certified.3, None).is_none());
    let masks = center_masks(&original, &domain(), None).unwrap();
    let mut wrong = map.clone();
    wrong[0].1[0] += 1;
    assert!(rows(&original, &domain(), &wrong, &masks, 0, &certified.3, None).is_none());
    assert!(rows_with_output_and_guards(
        &original,
        &domain(),
        &map,
        Some(
            Instant::now()
                .checked_sub(Duration::from_millis(1))
                .unwrap()
        )
    )
    .is_none());
}

#[test]
fn guard_unit_and_prefix_intervals_survive_lost_coefficient_composition() {
    let prior = vec![
        expr(1.0, 0.0, &[1.0]),
        expr(1.0, 0.0, &[1.0]),
        expr(1.0, 0.0, &[1.0]),
    ];
    let composed = authored_affine(
        &prior,
        &[9_007_199_254_740_992.0, 1.0, -9_007_199_254_740_992.0],
        &[0.0],
        1,
        3,
        None,
    )
    .unwrap();
    assert!(composed[0].root[0].lo <= 1.0 && composed[0].root[0].hi >= 1.0);
    assert!(composed[0].slack[0].lo <= 1.0 && composed[0].slack[0].hi >= 1.0);
    let mut guards = Vec::new();
    collect_guards(&composed, &[1], false, &mut guards, &mut 0, None).unwrap();
    assert!(guards[0].positive[0] >= 1.0);
    assert_eq!(guards[0].positive[1], 1.0);
    let value = root_value(
        &guards[0].root,
        &BoxDomain {
            lo: vec![0.5],
            hi: vec![0.5],
        },
        None,
    )
    .unwrap();
    assert!(value.lo <= 0.5 && value.hi >= 0.5);
}
