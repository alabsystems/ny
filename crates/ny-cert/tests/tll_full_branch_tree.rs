// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Durable regression: the FULL `tllverifybench` `instance_6_0` UNSAT verdict as
//! a kernel-checked branch-tree certificate over the exact `nx=8` (64-cell)
//! partition of `[-2,2]^2`. Mirrors `examples/tll_instance_6_0_full.rs` and locks
//! in the composed `global_LB > threshold` fact plus checker acceptance.

use ny_cert::{
    check_branch_tree, AxisPartition, BranchLeaf, BranchTreeCertificate, ConstraintKind,
    EntailmentCertificate, LinearConstraint, Rat, ThreshDir,
};

include!("../examples/data/tll_instance_6_0_full_data.rs");

fn r(t: (i128, i128)) -> Rat {
    Rat::new(t.0, t.1).unwrap()
}

fn axis_edges(lo: Rat, hi: Rat, nx: usize) -> Vec<Rat> {
    let span = hi.sub(lo).unwrap();
    (0..=nx)
        .map(|k| {
            lo.add(span.mul(Rat::new(k as i128, nx as i128).unwrap()).unwrap())
                .unwrap()
        })
        .collect()
}

fn corner_min(a0: Rat, a1: Rat, b: Rat, x0lo: Rat, x0hi: Rat, x1lo: Rat, x1hi: Rat) -> Rat {
    let cx0 = if !a0.is_negative() { x0lo } else { x0hi };
    let cx1 = if !a1.is_negative() { x1lo } else { x1hi };
    a0.mul(cx0)
        .unwrap()
        .add(a1.mul(cx1).unwrap())
        .unwrap()
        .add(b)
        .unwrap()
}

fn member_ent(
    a0: Rat,
    a1: Rat,
    b: Rat,
    m: Rat,
    x0lo: Rat,
    x0hi: Rat,
    x1lo: Rat,
    x1hi: Rat,
) -> EntailmentCertificate {
    let face = |v: &str, k: ConstraintKind, bnd: Rat| {
        LinearConstraint::with_kind(k, &[(v, Rat::ONE)], bnd)
    };
    let (p0, mu0) = if !a0.is_negative() {
        (face("x0", ConstraintKind::Ge, x0lo), a0)
    } else {
        (face("x0", ConstraintKind::Le, x0hi), a0.neg())
    };
    let (p1, mu1) = if !a1.is_negative() {
        (face("x1", ConstraintKind::Ge, x1lo), a1)
    } else {
        (face("x1", ConstraintKind::Le, x1hi), a1.neg())
    };
    EntailmentCertificate {
        premises: vec![p0, p1],
        multipliers: vec![mu0, mu1],
        conclusion: LinearConstraint::with_kind(
            ConstraintKind::Ge,
            &[("x0", a0), ("x1", a1)],
            m.sub(b).unwrap(),
        ),
    }
}

fn build_full_cert() -> (BranchTreeCertificate, Rat) {
    let edges = axis_edges(r(BOX_LO), r(BOX_HI), NX);
    let aff: Vec<(Rat, Rat, Rat)> = AFFINE
        .iter()
        .map(|&(a0, a1, b)| (r(a0), r(a1), r(b)))
        .collect();

    let mut leaves = Vec::new();
    let mut global: Option<Rat> = None;
    for ix in 0..NX {
        for iy in 0..NX {
            let (x0lo, x0hi) = (edges[ix], edges[ix + 1]);
            let (x1lo, x1hi) = (edges[iy], edges[iy + 1]);
            let c: Vec<Rat> = aff
                .iter()
                .map(|&(a0, a1, b)| corner_min(a0, a1, b, x0lo, x0hi, x1lo, x1hi))
                .collect();
            let mut best: Option<(Rat, usize)> = None;
            for (j, g) in GROUPS.iter().enumerate() {
                let mut gmin: Option<Rat> = None;
                for &i in g.iter() {
                    gmin = Some(match gmin {
                        Some(cur) if cur <= c[i] => cur,
                        _ => c[i],
                    });
                }
                let gmin = gmin.unwrap();
                best = Some(match best {
                    Some((cur, cj)) if cur >= gmin => (cur, cj),
                    _ => (gmin, j),
                });
            }
            let (m_cell, jstar) = best.unwrap();
            let mut mes = Vec::new();
            let mut mbs = Vec::new();
            for &i in GROUPS[jstar].iter() {
                let (a0, a1, b) = aff[i];
                mes.push(member_ent(a0, a1, b, m_cell, x0lo, x0hi, x1lo, x1hi));
                mbs.push(b);
            }
            leaves.push(BranchLeaf {
                lo: vec![x0lo, x1lo],
                hi: vec![x0hi, x1hi],
                bound: m_cell,
                member_entailments: mes,
                member_biases: mbs,
            });
            global = Some(match global {
                Some(g) if g <= m_cell => g,
                _ => m_cell,
            });
        }
    }
    let cert = BranchTreeCertificate {
        axes: vec![
            AxisPartition {
                var: "x0".to_owned(),
                edges: edges.clone(),
            },
            AxisPartition {
                var: "x1".to_owned(),
                edges,
            },
        ],
        leaves,
        threshold: r(THRESH),
        dir: ThreshDir::Le,
    };
    (cert, global.unwrap())
}

#[test]
fn full_instance_6_0_branch_tree_accepts_and_clears() {
    let (cert, producer_global) = build_full_cert();
    assert_eq!(cert.leaves.len(), (NX * NX), "64-cell partition");

    let (g, t) = check_branch_tree(&cert).expect("full branch-tree cert must be accepted");
    // Composed global bound == the exact slice global_LB, and clears threshold.
    assert_eq!(g, producer_global);
    assert_eq!(g, Rat::new(-39840843, 16777216).unwrap());
    assert_eq!(t, r(THRESH));
    assert!(
        g > t,
        "global_LB must strictly clear the Le threshold => UNSAT"
    );
}
