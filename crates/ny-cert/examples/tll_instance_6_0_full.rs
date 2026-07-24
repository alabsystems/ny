// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! FULL kernel-checked branch-tree certificate for `tllverifybench` `instance_6_0`.
//!
//! Where `examples/tll_instance_6_0.rs` proof-carries the single BINDING cell
//! (the slice that achieves the global min), THIS example certifies the WHOLE
//! UNSAT verdict: `min_x y(x) > threshold` over the entire box `[-2,2]^2`.
//!
//! `y(x) = max_j min_{i∈S_j} L_i(x)` is not bounded below by one flat Farkas
//! combination — the verdict is a MIN over the `nx×nx` cell partition of the
//! per-cell lower bounds. That case split is the machine-checked β-CROWN rule
//! `branch_split_min` (`Crownproof.Branch` in the exact pinned Clean dependency,
//! kernel re-typechecked). This example:
//!
//!   1. builds the exact `nx=8` (64-cell) product partition of the box as exact
//!      rationals (cell edges `-2, -3/2, …, 3/2, 2`);
//!   2. for EACH cell `C`, picks the witness group `j*(C) = argmax_j
//!      min_{i∈S_j} c_i(C)`, sets `m_C = LB_cell(C)`, and emits one entailment
//!      per member of `S_{j*}` proving `L_i(x) ≥ m_C` over `C` (a Farkas
//!      combination of `C`'s two corner-attaining box faces with multipliers
//!      `|a_i,k|`);
//!   3. composes all 64 leaves into a [`BranchTreeCertificate`] and runs
//!      [`check_branch_tree`], which verifies the partition is an EXACT cover,
//!      every leaf entailment is valid over its own cell, and
//!      `min_C m_C > threshold` — backed by `branch_split_min`, no new axiom.
//!
//! Every leaf entailment is also emitted as Clean-canonical JSON (a batch array
//! for `verify_entailment_certificate`), plus a self-describing branch-tree
//! envelope. Exact rational end to end; `global_LB = min_C m_C =
//! -39840843/16777216 ≈ -2.3747 > -3.0427` (threshold).
//!
//! Run: `cargo run -p ny-cert --example tll_instance_6_0_full -- [out_dir]`

use ny_cert::{
    branch_tree_leaf_batch_json, branch_tree_to_json, check_branch_tree, AxisPartition, BranchLeaf,
    BranchTreeCertificate, ConstraintKind, EntailmentCertificate, LinearConstraint, Rat, ThreshDir,
};

include!("data/tll_instance_6_0_full_data.rs");

fn r(t: (i128, i128)) -> Rat {
    Rat::new(t.0, t.1).expect("exact rational literal")
}

/// Exact cell edges: `edge_k = lo + (hi-lo)*k/nx` for `k = 0..=nx`.
fn axis_edges(lo: Rat, hi: Rat, nx: usize) -> Vec<Rat> {
    let span = hi.sub(lo).unwrap();
    (0..=nx)
        .map(|k| {
            let frac = Rat::new(k as i128, nx as i128).unwrap();
            lo.add(span.mul(frac).unwrap()).unwrap()
        })
        .collect()
}

/// Corner-min of `L = a0*x0 + a1*x1 + b` over the cell `[x0lo,x0hi]×[x1lo,x1hi]`.
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

/// Entailment `L_i(x) ≥ m` over the cell: the two corner-attaining box faces as
/// premises with multipliers `|a_i,k|`, conclusion `a·x ≥ m − b`.
fn member_entailment(
    a0: Rat,
    a1: Rat,
    b: Rat,
    m: Rat,
    x0lo: Rat,
    x0hi: Rat,
    x1lo: Rat,
    x1hi: Rat,
) -> EntailmentCertificate {
    let face = |var: &str, kind: ConstraintKind, bound: Rat| {
        LinearConstraint::with_kind(kind, &[(var, Rat::ONE)], bound)
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
    let concl = LinearConstraint::with_kind(
        ConstraintKind::Ge,
        &[("x0", a0), ("x1", a1)],
        m.sub(b).unwrap(),
    );
    EntailmentCertificate {
        premises: vec![p0, p1],
        multipliers: vec![mu0, mu1],
        conclusion: concl,
    }
}

fn main() {
    let box_lo = r(BOX_LO);
    let box_hi = r(BOX_HI);
    let thresh = r(THRESH);
    let edges = axis_edges(box_lo, box_hi, NX);

    // Decode affine functions once.
    let aff: Vec<(Rat, Rat, Rat)> = AFFINE
        .iter()
        .map(|&(a0, a1, b)| (r(a0), r(a1), r(b)))
        .collect();

    let mut leaves: Vec<BranchLeaf> = Vec::new();
    let mut global: Option<Rat> = None;

    for ix in 0..NX {
        for iy in 0..NX {
            let (x0lo, x0hi) = (edges[ix], edges[ix + 1]);
            let (x1lo, x1hi) = (edges[iy], edges[iy + 1]);

            // Per-affine cell corner-min (exact).
            let c: Vec<Rat> = aff
                .iter()
                .map(|&(a0, a1, b)| corner_min(a0, a1, b, x0lo, x0hi, x1lo, x1hi))
                .collect();

            // Witness group j*(C) = argmax_j min_{i∈S_j} c_i ; m_C = that value.
            let mut best: Option<(Rat, usize)> = None;
            for (j, g) in GROUPS.iter().enumerate() {
                let mut gmin: Option<Rat> = None;
                for &i in g.iter() {
                    gmin = Some(match gmin {
                        Some(cur) if cur <= c[i] => cur,
                        _ => c[i],
                    });
                }
                let gmin = gmin.expect("group is non-empty (decode guarantees)");
                best = Some(match best {
                    Some((cur, cj)) if cur >= gmin => (cur, cj),
                    _ => (gmin, j),
                });
            }
            let (m_cell, jstar) = best.expect("at least one group");

            // Leaf: one entailment per witness-group member proving L_i ≥ m_cell.
            let mut member_entailments = Vec::new();
            let mut member_biases = Vec::new();
            for &i in GROUPS[jstar].iter() {
                let (a0, a1, b) = aff[i];
                assert!(
                    c[i] >= m_cell,
                    "witness member corner must be >= cell bound"
                );
                member_entailments
                    .push(member_entailment(a0, a1, b, m_cell, x0lo, x0hi, x1lo, x1hi));
                member_biases.push(b);
            }

            leaves.push(BranchLeaf {
                lo: vec![x0lo, x1lo],
                hi: vec![x0hi, x1hi],
                bound: m_cell,
                member_entailments,
                member_biases,
            });

            global = Some(match global {
                Some(g) if g <= m_cell => g,
                _ => m_cell,
            });
        }
    }

    let global = global.unwrap();
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
        threshold: thresh,
        dir: ThreshDir::Le,
    };

    let n_leaves = cert.leaves.len();
    let n_ents: usize = cert.leaves.iter().map(|l| l.member_entailments.len()).sum();

    // --- Verify the whole composed certificate ---
    let (g, t) = check_branch_tree(&cert).expect("FULL branch-tree certificate must be accepted");
    assert_eq!(
        g, global,
        "checker global bound must match the producer min"
    );
    assert!(g > t, "global bound must strictly clear the Le threshold");

    println!("FULL branch-tree certificate ACCEPTED (check_branch_tree):");
    println!("  cells (leaves)      : {n_leaves}  (nx={NX} => {NX}x{NX})");
    println!("  leaf entailments    : {n_ents}  (per-cell witness-group members)");
    println!(
        "  global_LB = min_C m_C = {}/{}  (~{:.6})",
        g.num(),
        g.den(),
        ratio_f64(&g)
    );
    println!(
        "  threshold (Le)        = {}/{}  (~{:.6})",
        t.num(),
        t.den(),
        ratio_f64(&t)
    );
    println!("  => min_x y(x) >= global_LB > threshold  =>  Y_0 <= threshold is UNSAT");
    println!("  composition rule    : crownproof::branch_split_min (kernel re-typechecked)");

    // --- Emit Clean-canonical JSON artifacts ---
    if let Some(dir) = std::env::args().nth(1) {
        std::fs::create_dir_all(&dir).unwrap();
        let env = branch_tree_to_json(&cert).unwrap();
        std::fs::write(
            format!("{dir}/tll_instance_6_0_branch_tree.json"),
            serde_json::to_string_pretty(&env).unwrap(),
        )
        .unwrap();
        let batch = branch_tree_leaf_batch_json(&cert).unwrap();
        std::fs::write(
            format!("{dir}/tll_instance_6_0_leaf_batch.json"),
            serde_json::to_string_pretty(&batch).unwrap(),
        )
        .unwrap();
        println!("wrote branch-tree envelope + Clean batch ({n_ents} entailments) to {dir}/");
    }
}

fn ratio_f64(x: &Rat) -> f64 {
    use std::str::FromStr;
    let n = f64::from_str(&x.num().to_string()).unwrap_or(f64::NAN);
    let d = f64::from_str(&x.den().to_string()).unwrap_or(f64::NAN);
    n / d
}
