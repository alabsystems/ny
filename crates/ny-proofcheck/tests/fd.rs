// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! B2.3: the frozen-dual (`fd`) rung's named tests — acceptance of a correct
//! prune bundle, the four mutations (dropped prune, wrong side, multiplier
//! perturbed beyond rounding, wrong class), garbage hints, the checker's own
//! state containment, the per-node sample of a selected fd-confirmed node
//! (every shallow node, a hashed one in 64 of the deep ones, deterministic)
//! and its exact-rational soundness. Every fixture is synthetic (no
//! `benchmarks/`, no producer crate).
//!
//! # The fixture's frozen-dual tree (exact arithmetic)
//!
//! The base MLP (`support::mlp_onnx`): `z0 = x0 + x1`, `z1 = x0 - x1` on
//! `x ∈ [-1, 1]^2`, margin `y0 - y1 = relu(z0) + relu(z1) + δ`. With the
//! slope `λ = 1/2` on both unstable neurons (`μ = 1/2`), the state is
//! `margin = e_x0 + 1 + e_z0/2 + e_z1/2 + δ` (root `δ - 1 < 0`), and the
//! split on `(z0, z1)` closes with
//!
//! | sides (z0, z1) | L0 | `rc` | bound |
//! |---|---|---|---|
//! | Off, Off | (0, 0) | 0 | δ |
//! | Off, On  | (0, 1) | 0 | δ |
//! | On,  Off | (1, 0) | 0 | δ |
//! | On,  On  | (1, 1) | 0 | δ |
//!
//! (`L1 = 0` everywhere), so every prune is `δ` minus rounding. Without a
//! β the CROWN ladder cannot use the split constraints, so the three prunes
//! with an `On` side are confirmed by the fd rung or not at all.

mod support;

use ny_cert::json::Json;
use ny_cert::Rat;
use ny_proofcheck::bundle::sha256_hex;
use ny_proofcheck::check::{
    fd_node_sample, FdNodeSample, FD_NODE_SAMPLE_ALL_DEPTH, FD_NODE_SAMPLE_ONE_IN,
};
use ny_proofcheck::fd::{Form, Frame, Relu};
use ny_proofcheck::testkit::{fd_node_bound, fd_state, net_from_onnx};
use ny_proofcheck::{FaultInjection, Verdict};
use support::*;

/// `δ = 2^-20`.
const D: f64 = 1.0 / 1_048_576.0;

/// The four prunes of the fixture's tree: literals (site id 0 = tensor `z`)
/// and the sign-cut multipliers `L0`.
fn prunes() -> Vec<(Vec<L>, Vec<f64>)> {
    vec![
        (vec![L::R(0, 0, false), L::R(0, 1, false)], vec![0.0, 0.0]),
        (vec![L::R(0, 0, false), L::R(0, 1, true)], vec![0.0, 1.0]),
        (vec![L::R(0, 0, true), L::R(0, 1, false)], vec![1.0, 0.0]),
        (vec![L::R(0, 0, true), L::R(0, 1, true)], vec![1.0, 1.0]),
    ]
}

/// A bundle: header, sites, `clauses`, one `hint.fd_state` and its
/// `hint.fd_lambda` per class in `states`, the prune nodes `(lits, L0,
/// class, state)`, the verdict requiring `required`.
fn fd_bundle(
    fx: &Fixture,
    clause_list: &[(u64, Vec<String>)],
    states: &[(u64, u64)],
    nodes: &[(Vec<L>, Vec<f64>, u64, u64)],
    required: &[u64],
) -> Vec<u8> {
    let mut lines = vec![
        fx.header(),
        sites(&[(0, "z", 3, 1)]),
        clauses(0, clause_list),
    ];
    for &(state, class) in states {
        lines.push(fd_state_line(state, class, &[(0, 0), (0, 1)], 4));
        lines.push(fd_lambda_line(state, 0, &[0, 1], &[0.5, 0.5]));
    }
    for (k, (lits, l0, class, state)) in nodes.iter().enumerate() {
        let l1 = vec![0.0; l0.len()];
        lines.push(fd_node_line(k as u64, lits, &[*class], *state, l0, &l1, D));
    }
    lines.push(verdict(required, nodes.len() as u64));
    lines.push(published("unsat"));
    finish(&lines)
}

fn base_nodes(class: u64, state: u64) -> Vec<(Vec<L>, Vec<f64>, u64, u64)> {
    prunes()
        .into_iter()
        .map(|(lits, l0)| (lits, l0, class, state))
        .collect()
}

fn base_fixture(name: &str) -> Fixture {
    Fixture::new(name, mlp_onnx(D, 0.0), &mlp_vnnlib())
}

fn fd_section(o: &ny_proofcheck::CheckOutcome) -> Json {
    report(o)["fd"].clone()
}

// ---------------------------------------------------------------------------
// Acceptance
// ---------------------------------------------------------------------------

/// A correct frozen-dual prune bundle is VERIFIED, and the fd rung is what
/// confirmed every prune — with L3 / clip off too, where the CROWN ladder
/// has no β and could not use the split constraints at all.
#[test]
fn fd_accepts_a_correct_frozen_dual_prune_bundle() {
    let fx = base_fixture("fd_accept");
    let bytes = fd_bundle(
        &fx,
        &[(1, vec![clause01()])],
        &[(7, 1)],
        &base_nodes(1, 7),
        &[1],
    );
    for refine in [true, false] {
        let o = fx.check_with(&format!("accept_{refine}"), &bytes, |a| a.refine = refine);
        assert_eq!(o.verdict, Verdict::Verified, "{}", o.report_json);
        let r = report(&o);
        assert_eq!(r["ladder"]["fd"].as_u64(), Some(4), "{}", o.report_json);
        let fd = fd_section(&o);
        assert_eq!(fd["states_hinted"].as_u64(), Some(1));
        assert_eq!(fd["states_built"].as_u64(), Some(1));
        assert_eq!(fd["nodes_hinted"].as_u64(), Some(4));
        assert_eq!(fd["discharges_confirmed"].as_u64(), Some(4));
        assert_eq!(fd["states"][0]["hinted_lambda"].as_u64(), Some(2));
        assert_eq!(fd["states"][0]["unstable"].as_u64(), Some(2));
        assert_eq!(fd["states"][0]["in_gens"].as_u64(), Some(2));
        // The checker's certified margins are δ minus rounding.
        let min = match &r["margins"]["min_chk"] {
            Json::F64(v) => *v,
            other => panic!("min_chk {other:?}"),
        };
        assert!(min > 0.99 * D && min < D, "min_chk {min}");
    }
    // The multipliers sit within rounding of the exact dual point: a
    // one-ulp move of a multiplier still verifies.
    let mut nodes = base_nodes(1, 7);
    nodes[3].1[0] = 1.0f64.next_up();
    let bytes = fd_bundle(&fx, &[(1, vec![clause01()])], &[(7, 1)], &nodes, &[1]);
    let o = fx.check("ulp", &bytes);
    assert_eq!(o.verdict, Verdict::Verified, "{}", o.report_json);
}

/// Without the `hint.fd_lambda` record the checker uses its own chord slope
/// (`u / (u - l) = 1/2` here): the same tree still verifies — the slopes are
/// hints, the state is the checker's.
#[test]
fn fd_state_is_rebuilt_without_the_producers_slopes() {
    let fx = base_fixture("fd_no_lambda");
    let mut lines = vec![
        fx.header(),
        sites(&[(0, "z", 3, 1)]),
        clauses(0, &[(1, vec![clause01()])]),
    ];
    for (k, (lits, l0)) in prunes().into_iter().enumerate() {
        lines.push(fd_node_line(k as u64, &lits, &[1], 3, &l0, &[0.0, 0.0], D));
    }
    lines.push(verdict(&[1], 4));
    lines.push(published("unsat"));
    let o = fx.check("no_lambda", &finish(&lines));
    assert_eq!(o.verdict, Verdict::Verified, "{}", o.report_json);
    let fd = fd_section(&o);
    assert_eq!(fd["states_hinted"].as_u64(), Some(0));
    assert_eq!(fd["states_built"].as_u64(), Some(1));
    assert_eq!(fd["states"][0]["hinted_lambda"].as_u64(), Some(0));
}

// ---------------------------------------------------------------------------
// Mutations
// ---------------------------------------------------------------------------

/// A dropped prune leaves its cell uncovered: INVALID (a claimed-cover gap).
#[test]
fn fd_rejects_a_dropped_prune() {
    let fx = base_fixture("fd_dropped");
    for drop in 0..4 {
        let mut nodes = base_nodes(1, 7);
        nodes.remove(drop);
        let bytes = fd_bundle(&fx, &[(1, vec![clause01()])], &[(7, 1)], &nodes, &[1]);
        let o = fx.check(&format!("drop{drop}"), &bytes);
        assert_eq!(
            o.verdict,
            Verdict::Invalid,
            "drop {drop}: {}",
            o.report_json
        );
        assert!(reason_codes(&o).contains(&"cover_gap".to_string()));
    }
}

/// A flipped literal side duplicates one cell and leaves its sibling
/// uncovered: never verified (the cover has a gap).
#[test]
fn fd_rejects_a_wrong_side() {
    let fx = base_fixture("fd_side");
    for (node, lit) in [(3usize, 1usize), (1, 0), (2, 1)] {
        let mut nodes = base_nodes(1, 7);
        if let L::R(t, i, p) = nodes[node].0[lit] {
            nodes[node].0[lit] = L::R(t, i, !p);
        }
        let bytes = fd_bundle(&fx, &[(1, vec![clause01()])], &[(7, 1)], &nodes, &[1]);
        let o = fx.check(&format!("side{node}{lit}"), &bytes);
        assert_eq!(o.verdict, Verdict::Invalid, "{}", o.report_json);
    }
}

/// A multiplier moved BEYOND rounding (here by `4δ`, which makes the exact
/// prune bound `δ - 8δ < 0`) no longer certifies: the node is unconfirmed
/// and the bundle is never verified, with or without L3 / clip. A
/// `δ / 1024` move is still beyond the rounding slack but keeps the bound
/// positive: weak duality makes ANY dual point valid, so that one verifies.
#[test]
fn fd_rejects_a_multiplier_perturbed_beyond_rounding() {
    let fx = base_fixture("fd_mult");
    for (node, which) in [(3usize, 0usize), (3, 1), (1, 1), (2, 0)] {
        let mut nodes = base_nodes(1, 7);
        nodes[node].1[which] += 4.0 * D;
        let bytes = fd_bundle(&fx, &[(1, vec![clause01()])], &[(7, 1)], &nodes, &[1]);
        for refine in [true, false] {
            let o = fx.check_with(&format!("mult{node}{which}{refine}"), &bytes, |a| {
                a.refine = refine;
            });
            assert_eq!(o.verdict, Verdict::Unconfirmed, "{}", o.report_json);
            let r = report(&o);
            assert_eq!(r["ladder"]["fd"].as_u64(), Some(3));
            assert!(
                o.report_json.contains("fd_weaker_state"),
                "{}",
                o.report_json
            );
        }
    }
    let mut nodes = base_nodes(1, 7);
    nodes[3].1[0] += D / 1024.0;
    let bytes = fd_bundle(&fx, &[(1, vec![clause01()])], &[(7, 1)], &nodes, &[1]);
    assert_eq!(fx.check("mult_small", &bytes).verdict, Verdict::Verified);
}

/// Prunes relabelled to the WRONG class leave the right class uncovered:
/// INVALID. A two-clause property (class 0 true against classes 1 and 2,
/// identical margins) with each class's tree streamed verifies; moving one
/// class-1 prune to class 2 does not.
#[test]
fn fd_rejects_a_wrong_class() {
    let onnx = model(
        "x",
        &[1, 2],
        &[
            node("Gemm", &["x", "W1", "B1"], &["z"], vec![A::I("transB", 1)]),
            node("Relu", &["z"], &["h"], vec![]),
            node("Gemm", &["h", "W2", "B2"], &["y"], vec![A::I("transB", 1)]),
        ],
        &[
            tf("W1", &[3, 2], &[1.0, 1.0, 1.0, -1.0, 1.0, 1.0]),
            tf("B1", &[3], &[0.0, 0.0, 5.0]),
            tf(
                "W2",
                &[3, 3],
                &[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ),
            tf("B2", &[3], &[D, 0.0, 0.0]),
        ],
        "y",
        &[1, 3],
    );
    let vnn = robustness_vnnlib(&["-1", "-1"], &["1", "1"], 3, 0, &[1, 2]);
    let fx = Fixture::new("fd_class", onnx, &vnn);
    let cl = [
        (1, vec!["Y0:1,Y1:-1<=0".to_string()]),
        (2, vec!["Y0:1,Y2:-1<=0".to_string()]),
    ];
    let mut nodes = base_nodes(1, 7);
    nodes.extend(base_nodes(2, 8));
    let good = fd_bundle(&fx, &cl, &[(7, 1), (8, 2)], &nodes, &[1, 2]);
    let o = fx.check("both", &good);
    assert_eq!(o.verdict, Verdict::Verified, "{}", o.report_json);
    assert_eq!(fd_section(&o)["states_built"].as_u64(), Some(2));
    for k in 0..4 {
        let mut bad = nodes.clone();
        bad[k].2 = 2;
        let bytes = fd_bundle(&fx, &cl, &[(7, 1), (8, 2)], &bad, &[1, 2]);
        let o = fx.check(&format!("wrong{k}"), &bytes);
        assert_eq!(o.verdict, Verdict::Invalid, "{}", o.report_json);
    }
}

/// A misaligned dual point (fewer multipliers than literals) is an invalid
/// hint: counted, dropped, and the prune is then unconfirmed (never a crash,
/// never a confirmation).
#[test]
fn fd_misaligned_hint_is_dropped_and_never_confirms() {
    let fx = base_fixture("fd_misaligned");
    let mut lines = vec![
        fx.header(),
        sites(&[(0, "z", 3, 1)]),
        clauses(0, &[(1, vec![clause01()])]),
        fd_state_line(7, 1, &[(0, 0), (0, 1)], 4),
        fd_lambda_line(7, 0, &[0, 1], &[0.5, 0.5]),
    ];
    for (k, (lits, l0)) in prunes().into_iter().enumerate() {
        let (l0, l1) = if k == 3 {
            (vec![1.0], vec![0.0])
        } else {
            (l0, vec![0.0, 0.0])
        };
        lines.push(fd_node_line(k as u64, &lits, &[1], 7, &l0, &l1, D));
    }
    lines.push(verdict(&[1], 4));
    lines.push(published("unsat"));
    let o = fx.check("misaligned", &finish(&lines));
    assert_eq!(o.verdict, Verdict::Unconfirmed, "{}", o.report_json);
    let r = report(&o);
    assert!(r["nodes"]["hints_invalid"].as_u64().unwrap_or(0) >= 1);
    assert_eq!(fd_section(&o)["nodes_hinted"].as_u64(), Some(3));
}

/// On a FALSE property (`b0 < b1`: class 1 wins somewhere) no frozen-dual
/// hint — exact, huge, negative, NaN multipliers, garbage slopes — ever
/// confirms a discharge.
#[test]
fn fd_garbage_hints_never_confirm_a_false_discharge() {
    let fx = Fixture::new("fd_false", mlp_onnx(0.0, D), &mlp_vnnlib());
    let duals: [&[f64]; 5] = [
        &[0.0, 0.0],
        &[1.0, 1.0],
        &[1e300, 1e300],
        &[-5.0, f64::NAN],
        &[f64::NAN, 2.0],
    ];
    for (g, lam) in [(0usize, 0.5f64), (1, 0.0), (2, 1.0), (3, 7.0), (4, -3.0)] {
        let mut lines = vec![
            fx.header(),
            sites(&[(0, "z", 3, 1)]),
            clauses(0, &[(1, vec![clause01()])]),
            fd_state_line(7, 1, &[(0, 0), (0, 1)], 4),
            fd_lambda_line(7, 0, &[0, 1], &[lam, lam]),
        ];
        for (k, (lits, _)) in prunes().into_iter().enumerate() {
            let l0 = duals[(g + k) % duals.len()].to_vec();
            let l1 = duals[(g + 2 * k + 1) % duals.len()].to_vec();
            lines.push(fd_node_line(k as u64, &lits, &[1], 7, &l0, &l1, D));
        }
        lines.push(verdict(&[1], 4));
        lines.push(published("unsat"));
        let o = fx.check(&format!("false{g}"), &finish(&lines));
        assert_ne!(o.verdict, Verdict::Verified, "{}", o.report_json);
        let r = report(&o);
        assert_eq!(
            r["nodes"]["confirmed"].as_u64(),
            Some(0),
            "a garbage fd hint confirmed a false discharge: {}",
            o.report_json
        );
    }
}

/// The FD state's own containment sample catches a broken state: with every
/// objective form's constant raised by 1000 the checker reports a fault,
/// never a verification.
#[test]
fn fd_state_containment_trips_on_an_injected_checker_fault() {
    let fx = base_fixture("fd_fault");
    let bytes = fd_bundle(
        &fx,
        &[(1, vec![clause01()])],
        &[(7, 1)],
        &base_nodes(1, 7),
        &[1],
    );
    let o = fx.check_with("fault", &bytes, |a| {
        a.fault = Some(FaultInjection::InflateFdMargin);
    });
    assert_eq!(o.verdict, Verdict::CheckerFault, "{}", o.report_json);
    assert!(
        o.report_json.contains("fd objective form"),
        "{}",
        o.report_json
    );
}

// ---------------------------------------------------------------------------
// The per-node sample of a fd-confirmed node (self-containment (3))
// ---------------------------------------------------------------------------

fn sampling(o: &ny_proofcheck::CheckOutcome) -> Json {
    fd_section(o)["node_sampling"].clone()
}

fn count(j: &Json, key: &str) -> u64 {
    j[key]
        .as_u64()
        .unwrap_or_else(|| panic!("node_sampling.{key} missing: {j:?}"))
}

/// The per-node sample catches a wrong NODE bound on a selected node: with
/// every certified prune bound raised by 1000 over a CORRECT state (which the
/// per-state containment cannot see), the four depth-2 prunes of the base
/// tree (shallow, so every one is sampled) report a checker fault at a point
/// inside a node region. The honest bundle reports the sample's counts: four
/// shallow nodes, the centre plus 14 seeded points each, some inside.
#[test]
fn fd_node_sample_catches_an_inflated_bound_on_a_selected_node() {
    let fx = base_fixture("fd_node_fault");
    let bytes = fd_bundle(
        &fx,
        &[(1, vec![clause01()])],
        &[(7, 1)],
        &base_nodes(1, 7),
        &[1],
    );
    let honest = fx.check("honest", &bytes);
    assert_eq!(honest.verdict, Verdict::Verified, "{}", honest.report_json);
    let s = sampling(&honest);
    assert_eq!(count(&s, "all_depth"), FD_NODE_SAMPLE_ALL_DEPTH as u64);
    assert_eq!(count(&s, "one_in"), FD_NODE_SAMPLE_ONE_IN);
    assert_eq!(count(&s, "nodes"), 4, "{s:?}");
    assert_eq!(count(&s, "shallow"), 4, "{s:?}");
    assert_eq!(count(&s, "hashed"), 0, "{s:?}");
    assert_eq!(count(&s, "skipped"), 0, "{s:?}");
    assert_eq!(
        count(&s, "points"),
        4 * 15,
        "centre + refute_points - 2 per node: {s:?}"
    );
    assert!(
        count(&s, "in_region") > 0,
        "no sampled point inside any node region: {s:?}"
    );
    let o = fx.check_with("inflated", &bytes, |a| {
        a.fault = Some(FaultInjection::InflateFdNodeBound);
    });
    assert_eq!(o.verdict, Verdict::CheckerFault, "{}", o.report_json);
    assert!(
        o.report_json.contains("certified fd bound"),
        "the per-node sample must be what caught it: {}",
        o.report_json
    );
    assert!(
        !o.report_json.contains("fd objective form"),
        "the per-state containment cannot see a node-bound fault: {}",
        o.report_json
    );
}

/// The DEEP fixture: `z0 = x0 + x1`, `z1 = x0 - x1` (unstable) and
/// `z2..z6 = x0 + x1 + 5` (active on the whole box), `y0 = h0 + h1 + 2`,
/// `y1 = 0`. The margin is `>= 2` everywhere and the checker's FD root bound
/// is `2 - 1 = 1 > 0`, so a node whose literals are the five always-true
/// `z_k >= 0` (Relu depth 5, past [`FD_NODE_SAMPLE_ALL_DEPTH`]) has the WHOLE
/// box as its region — every sampled point applies — and the fd rung
/// confirms it honestly.
fn deep_onnx() -> Vec<u8> {
    deep_onnx_b0(2.0)
}

/// [`deep_onnx`] with `y0 = h0 + h1 + b0` (the margin's minimum is `b0`, at
/// any `x` with `x0 <= -|x1|`): `b0 < 0` makes the deep property FALSE.
fn deep_onnx_b0(b0: f64) -> Vec<u8> {
    // Rows z0, z1, then five rows `x0 + x1` with bias 5.
    let mut w1 = vec![1.0, 1.0, 1.0, -1.0];
    w1.extend_from_slice(&[1.0; 10]);
    let mut b1 = vec![0.0, 0.0];
    b1.extend_from_slice(&[5.0; 5]);
    let mut w2 = vec![0.0; 14];
    w2[0] = 1.0;
    w2[1] = 1.0;
    model(
        "x",
        &[1, 2],
        &[
            node("Gemm", &["x", "W1", "B1"], &["z"], vec![A::I("transB", 1)]),
            node("Relu", &["z"], &["h"], vec![]),
            node("Gemm", &["h", "W2", "B2"], &["y"], vec![A::I("transB", 1)]),
        ],
        &[
            tf("W1", &[7, 2], &w1),
            tf("B1", &[7], &b1),
            tf("W2", &[2, 7], &w2),
            tf("B2", &[2], &[b0, 0.0]),
        ],
        "y",
        &[1, 2],
    )
}

/// The deep bundle: the five complement cells `z2+..z(j-1)+, zj-` (empty:
/// the checker's root box puts every `z_j` in `[3, 7]`) and one fd prune per
/// id in `fd_ids` on `z2+..z6+` (overlapping copies, which the cover
/// accepts). Every fd prune has Relu depth 5.
fn deep_bundle(fx: &Fixture, fd_ids: &[u64]) -> Vec<u8> {
    deep_bundle_duals(fx, fd_ids, |_| (vec![0.0; 5], vec![0.0; 5]))
}

/// [`deep_bundle`] with the `j`-th fd prune certified at the dual point
/// `duals(j)` (`(L0, L1)`, five each).
fn deep_bundle_duals(
    fx: &Fixture,
    fd_ids: &[u64],
    duals: impl Fn(usize) -> (Vec<f64>, Vec<f64>),
) -> Vec<u8> {
    let mut lines = vec![
        fx.header(),
        sites(&[(0, "z", 7, 1)]),
        clauses(0, &[(1, vec![clause01()])]),
    ];
    for j in 2..7u64 {
        let mut lits: Vec<L> = (2..j).map(|i| L::R(0, i, true)).collect();
        lits.push(L::R(0, j, false));
        lines.push(node_line(
            j,
            &lits,
            "bound",
            Some(&[1]),
            None,
            None,
            false,
            1.0,
        ));
    }
    let prefix: Vec<L> = (2..7u64).map(|i| L::R(0, i, true)).collect();
    for (j, &id) in fd_ids.iter().enumerate() {
        let (l0, l1) = duals(j);
        lines.push(fd_node_line(id, &prefix, &[1], 9, &l0, &l1, 1.0));
    }
    lines.push(verdict(&[1], 5 + fd_ids.len() as u64));
    lines.push(published("unsat"));
    finish(&lines)
}

/// The first id set `[base, base + 4)` (`base = 100 + 10 s`) whose deep
/// bundle has exactly `want` nodes the hash selects, with its bytes and the
/// selected count as the checker will compute it (the selection is keyed by
/// the bundle's own sha256, which the fixture paths make host-specific).
fn deep_ids_with(fx: &Fixture, want: usize) -> (Vec<u64>, Vec<u8>) {
    deep_ids_where(want, |ids| deep_bundle(fx, ids))
}

/// As [`deep_ids_with`] for any deep bundle `build(ids)` (four depth-5 fd
/// prunes with ids `ids`): the GRIND a producer that controls the bundle can
/// run, since the selection is keyed by the bundle's own content.
fn deep_ids_where(want: usize, build: impl Fn(&[u64]) -> Vec<u8>) -> (Vec<u64>, Vec<u8>) {
    for salt in 0..4000u64 {
        let ids: Vec<u64> = (0..4).map(|j| 100 + 10 * salt + j).collect();
        let bytes = build(&ids);
        let sha = sha256_hex(&bytes);
        let picked = ids
            .iter()
            .filter(|&&id| fd_node_sample(5, id, &sha) == FdNodeSample::Hashed)
            .count();
        if picked == want {
            return (ids, bytes);
        }
    }
    panic!("no id set with {want} hashed node(s) in 4000 salts");
}

/// Past [`FD_NODE_SAMPLE_ALL_DEPTH`] only the HASHED nodes are sampled, and
/// the report says so. On the deep fixture (whole-box regions, so a sampled
/// node always applies its check): an honest bundle VERIFIES with every
/// deep node counted as hashed or skipped, 15 forward passes per hashed node
/// and none for a skipped one; with every prune bound inflated, a bundle
/// whose hash selects one node reports a checker fault, and one whose hash
/// selects none does NOT — the documented trade-off: an unselected deep node
/// is not audited (the per-state check still is, on every state). The
/// counts are the same at 1 and 4 threads.
#[test]
fn fd_node_sample_checks_a_hashed_fraction_of_deep_nodes() {
    let fx = Fixture::new("fd_node_deep", deep_onnx(), &mlp_vnnlib());
    // Honest: one hashed node (the bundle the hash selects exactly one of).
    let (ids, one) = deep_ids_with(&fx, 1);
    for threads in [1usize, 4] {
        let o = fx.check_with("deep_one", &one, |a| a.threads = Some(threads));
        assert_eq!(o.verdict, Verdict::Verified, "{}", o.report_json);
        assert_eq!(
            report(&o)["ladder"]["fd"].as_u64(),
            Some(4),
            "{}",
            o.report_json
        );
        let s = sampling(&o);
        assert_eq!(count(&s, "nodes"), 4, "{s:?}");
        assert_eq!(count(&s, "shallow"), 0, "{s:?}");
        assert_eq!(count(&s, "hashed"), 1, "ids {ids:?}: {s:?}");
        assert_eq!(count(&s, "skipped"), 3, "ids {ids:?}: {s:?}");
        assert_eq!(count(&s, "points"), 15, "{s:?}");
        assert_eq!(count(&s, "in_region"), 15, "a whole-box node: {s:?}");
    }
    // Inflated bounds: the selected node catches it.
    let o = fx.check_with("deep_one_inflated", &one, |a| {
        a.fault = Some(FaultInjection::InflateFdNodeBound);
    });
    assert_eq!(o.verdict, Verdict::CheckerFault, "{}", o.report_json);
    assert!(
        o.report_json.contains("certified fd bound"),
        "{}",
        o.report_json
    );
    // No selected node: nothing is sampled, so an inflated node bound is not
    // seen (the per-state check passes: the state is correct).
    let (ids, none) = deep_ids_with(&fx, 0);
    let o = fx.check_with("deep_none_inflated", &none, |a| {
        a.fault = Some(FaultInjection::InflateFdNodeBound);
    });
    assert_eq!(
        o.verdict,
        Verdict::Verified,
        "ids {ids:?}: {}",
        o.report_json
    );
    let s = sampling(&o);
    assert_eq!(count(&s, "skipped"), 4, "{s:?}");
    assert_eq!(count(&s, "hashed"), 0, "{s:?}");
    assert_eq!(
        count(&s, "points"),
        0,
        "a skipped node costs no forward pass: {s:?}"
    );
}

/// The per-node selection is a pure function of (Relu depth, node id, bundle
/// sha256): every node of depth `<= 4` is sampled; past it, a node is
/// sampled iff `sha256("ny-proofcheck/fd-node-sample/v1\0" || bundle hash ||
/// id_le)`'s first 64 bits are `0 mod 64`, whatever the depth. Pinned by a
/// golden list computed INDEPENDENTLY (Python `hashlib`), so a replay of an
/// old bundle samples the same nodes; keyed by the bundle hash (two hashes
/// pick disjoint sets here); and ~1 in 64 over 64000 ids.
#[test]
fn fd_node_sample_is_deterministic_and_keyed_by_the_bundle_hash() {
    assert_eq!(
        FD_NODE_SAMPLE_ALL_DEPTH, 4,
        "owner decision: depth <= 4 always"
    );
    assert_eq!(
        FD_NODE_SAMPLE_ONE_IN, 64,
        "owner decision: 1/64 of the deeper nodes"
    );
    let z = "0".repeat(64);
    let a = "a".repeat(64);
    for h in [&z, &a] {
        for id in 0..2000u64 {
            for d in 0..=FD_NODE_SAMPLE_ALL_DEPTH {
                assert_eq!(fd_node_sample(d, id, h), FdNodeSample::Shallow);
            }
            let s = fd_node_sample(FD_NODE_SAMPLE_ALL_DEPTH + 1, id, h);
            assert_ne!(s, FdNodeSample::Shallow, "depth 5 is deep");
            assert_eq!(s, fd_node_sample(FD_NODE_SAMPLE_ALL_DEPTH + 1, id, h));
            assert_eq!(s, fd_node_sample(999, id, h), "depth past 4 is not hashed");
            assert_eq!(s.sampled(), s == FdNodeSample::Hashed);
        }
    }
    let picked = |h: &str| -> Vec<u64> {
        (0..1000u64)
            .filter(|&id| fd_node_sample(5, id, h) == FdNodeSample::Hashed)
            .collect()
    };
    assert_eq!(
        picked(z.as_str()),
        vec![
            7, 60, 67, 194, 238, 251, 257, 395, 421, 426, 430, 439, 450, 509, 514, 533, 535, 576,
            671, 711, 725, 760, 762, 768, 814, 907, 931, 941
        ]
    );
    assert_eq!(
        picked(a.as_str()),
        vec![309, 344, 443, 455, 556, 738, 813, 842, 938, 985]
    );
    let n = (0..64_000u64)
        .filter(|&id| fd_node_sample(5, id, &z).sampled())
        .count();
    assert!(
        (800..=1200).contains(&n),
        "{n} of 64000 sampled, want ~1000"
    );
}

/// review(fd-sampling): the REAL guarantee past [`FD_NODE_SAMPLE_ALL_DEPTH`],
/// where the per-node sample mostly does not run. The selection is keyed by
/// the bundle's own content, so a hostile producer can GRIND it (this test
/// grinds it exactly that way); nothing may therefore depend on it. On a
/// FALSE deep property (`y0 - y1 = h0 + h1 - 1/2`, which is `-1/2` at the box
/// centre) no depth-5 fd prune is confirmed at any hinted dual point — zero,
/// one, huge, negative, NaN — whether the hash would select none or one of
/// the four nodes: the bundle is REFUTED (the checker probes the unconfirmed
/// nodes) and no node even reaches the sample decision. Positive control:
/// the TRUE deep property verifies with the same four fd confirmations
/// whether the hash selects none or one of them.
#[test]
fn fd_deep_discharge_never_depends_on_the_node_sample() {
    let duals: [&[f64]; 5] = [
        &[0.0; 5],
        &[1.0; 5],
        &[1e300; 5],
        &[-5.0, f64::NAN, 1.0, 0.0, 2.0],
        &[f64::NAN; 5],
    ];
    let fx = Fixture::new("fd_node_deep_false", deep_onnx_b0(-0.5), &mlp_vnnlib());
    let hostile = |ids: &[u64]| {
        deep_bundle_duals(&fx, ids, |j| {
            (duals[j % 5].to_vec(), duals[(j + 2) % 5].to_vec())
        })
    };
    for want in [0usize, 1] {
        let (ids, bytes) = deep_ids_where(want, &hostile);
        let o = fx.check(&format!("deep_false_{want}"), &bytes);
        assert_eq!(
            o.verdict,
            Verdict::Refuted,
            "{want} selected, ids {ids:?}: {}",
            o.report_json
        );
        let r = report(&o);
        assert_eq!(
            r["ladder"]["fd"].as_u64().unwrap_or(0),
            0,
            "a hostile dual point confirmed a false deep discharge: {}",
            o.report_json
        );
        assert_eq!(
            fd_section(&o)["discharges_confirmed"].as_u64(),
            Some(0),
            "{}",
            o.report_json
        );
        assert_eq!(
            count(&sampling(&o), "nodes"),
            0,
            "an unconfirmed node reached the sample decision: {}",
            o.report_json
        );
    }
    let fx = Fixture::new("fd_node_deep_true", deep_onnx(), &mlp_vnnlib());
    for want in [0usize, 1] {
        let (ids, bytes) = deep_ids_with(&fx, want);
        let o = fx.check(&format!("deep_true_{want}"), &bytes);
        assert_eq!(
            o.verdict,
            Verdict::Verified,
            "{want} selected, ids {ids:?}: {}",
            o.report_json
        );
        assert_eq!(
            report(&o)["ladder"]["fd"].as_u64(),
            Some(4),
            "{}",
            o.report_json
        );
        let s = sampling(&o);
        assert_eq!(count(&s, "hashed"), want as u64, "{s:?}");
        assert_eq!(count(&s, "skipped"), 4 - want as u64, "{s:?}");
    }
}

/// review(fd-sampling): the sample's depth is the count of DISTINCT Relu
/// sites, not of literals: a repeated literal does not thin the node region
/// (the cover dedups it the same way), so it must not push a shallow node
/// out of the always-sampled set into the 1-in-64 hash. On the deep fixture
/// (`z2..z6` active on the whole box) a node `z2+ x5` (five literals, ONE
/// site) and a node `z2+ z3+ z2+ z3+ z4+` (five literals, THREE sites) are
/// both shallow: sampled at the centre plus 14 points each, every point
/// inside (whole-box regions).
#[test]
fn fd_node_sample_depth_counts_distinct_relu_sites() {
    let fx = Fixture::new("fd_node_distinct", deep_onnx(), &mlp_vnnlib());
    let mut lines = vec![
        fx.header(),
        sites(&[(0, "z", 7, 1)]),
        clauses(0, &[(1, vec![clause01()])]),
    ];
    // The complement cells of `z2+ z3+ z4+` (empty: every `z_j` is in [3, 7]).
    for j in 2..5u64 {
        let mut lits: Vec<L> = (2..j).map(|i| L::R(0, i, true)).collect();
        lits.push(L::R(0, j, false));
        lines.push(node_line(
            j,
            &lits,
            "bound",
            Some(&[1]),
            None,
            None,
            false,
            1.0,
        ));
    }
    let three = [
        L::R(0, 2, true),
        L::R(0, 3, true),
        L::R(0, 2, true),
        L::R(0, 3, true),
        L::R(0, 4, true),
    ];
    let one = [L::R(0, 2, true); 5];
    for (id, lits) in [(20u64, &three[..]), (21, &one[..])] {
        assert!(lits.len() > FD_NODE_SAMPLE_ALL_DEPTH, "five literals");
        lines.push(fd_node_line(id, lits, &[1], 9, &[0.0; 5], &[0.0; 5], 1.0));
    }
    lines.push(verdict(&[1], 5));
    lines.push(published("unsat"));
    let o = fx.check("distinct", &finish(&lines));
    assert_eq!(o.verdict, Verdict::Verified, "{}", o.report_json);
    assert_eq!(
        report(&o)["ladder"]["fd"].as_u64(),
        Some(2),
        "{}",
        o.report_json
    );
    let s = sampling(&o);
    assert_eq!(count(&s, "nodes"), 2, "{s:?}");
    assert_eq!(
        count(&s, "shallow"),
        2,
        "a repeated literal pushed a shallow node into the hash: {s:?}"
    );
    assert_eq!(count(&s, "hashed") + count(&s, "skipped"), 0, "{s:?}");
    assert_eq!(count(&s, "points"), 2 * 15, "{s:?}");
    assert_eq!(count(&s, "in_region"), 2 * 15, "whole-box regions: {s:?}");
}

// ---------------------------------------------------------------------------
// Exact-rational soundness of the checker's own state and bound
// ---------------------------------------------------------------------------

fn rat(x: f64) -> Rat {
    Rat::from_f64_exact(x).expect("finite")
}

fn radd(a: Rat, b: Rat) -> Rat {
    a.add(b).expect("rat add")
}

fn rsub(a: Rat, b: Rat) -> Rat {
    a.sub(b).expect("rat sub")
}

fn rmul(a: Rat, b: Rat) -> Rat {
    a.mul(b).expect("rat mul")
}

fn rdiv(a: Rat, b: Rat) -> Rat {
    a.mul(b.inv().expect("nonzero")).expect("rat mul")
}

fn rmin(a: Rat, b: Rat) -> Rat {
    if a < b {
        a
    } else {
        b
    }
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 11
    }

    /// A dyadic in `[-1, 1]` with 7 fractional bits.
    fn dy(&mut self) -> f64 {
        (self.next() % 257) as f64 / 128.0 - 1.0
    }
}

/// A random Gemm/Relu/Gemm net with `n_in` inputs, `n_h` hidden neurons and
/// two outputs (dyadic f32-exact weights).
fn random_mlp(rng: &mut Lcg, n_in: usize, n_h: usize) -> Vec<u8> {
    let w1: Vec<f64> = (0..n_h * n_in).map(|_| rng.dy()).collect();
    let b1: Vec<f64> = (0..n_h).map(|_| rng.dy() * 0.5).collect();
    let w2: Vec<f64> = (0..2 * n_h).map(|_| rng.dy()).collect();
    let b2: Vec<f64> = (0..2).map(|_| rng.dy()).collect();
    model(
        "x",
        &[1, n_in as i64],
        &[
            node("Gemm", &["x", "W1", "B1"], &["z"], vec![A::I("transB", 1)]),
            node("Relu", &["z"], &["h"], vec![]),
            node("Gemm", &["h", "W2", "B2"], &["y"], vec![A::I("transB", 1)]),
        ],
        &[
            tf("W1", &[n_h as i64, n_in as i64], &w1),
            tf("B1", &[n_h as i64], &b1),
            tf("W2", &[2, n_h as i64], &w2),
            tf("B2", &[2], &b2),
        ],
        "y",
        &[1, 2],
    )
}

/// Exact `(z, y)` of the lowered net at `x`.
fn exact_forward(net: &ny_proofcheck::lower::Net, x: &[f64]) -> (Vec<Rat>, Vec<Rat>) {
    use ny_proofcheck::lower::Op;
    let mut vals: Vec<Vec<Rat>> = vec![Vec::new(); net.tensors.len()];
    vals[net.input] = x.iter().map(|&v| rat(v)).collect();
    for nd in &net.nodes {
        let out = match &nd.op {
            Op::Gemm(g) => (0..g.n_out)
                .map(|j| {
                    let mut s = rat(g.b[j]);
                    for i in 0..g.n_in {
                        s = radd(s, rmul(rat(g.w[j * g.n_in + i]), vals[g.x][i]));
                    }
                    s
                })
                .collect(),
            Op::Relu { x } => vals[*x]
                .iter()
                .map(|v| if v.is_negative() { Rat::ZERO } else { *v })
                .collect(),
            other => panic!("unexpected op {other:?}"),
        };
        vals[nd.out] = out;
    }
    (vals[net.by_name["z"]].clone(), vals[net.output].clone())
}

/// The exact generator vector `e(x)` of the checker's frame at `x`.
fn exact_symbols(net: &ny_proofcheck::lower::Net, frame: &Frame, x: &[f64], z: &[Rat]) -> Vec<Rat> {
    let mut e = vec![Rat::ZERO; frame.n_gens];
    for (i, c) in frame.in_col.iter().enumerate() {
        if let Some(col) = *c {
            e[col] = rdiv(rsub(rat(x[i]), rat(frame.mid[i])), rat(frame.rad[i]));
        }
    }
    let zt = net.by_name["z"];
    for (i, r) in frame.relax[zt].iter().enumerate() {
        if let Relu::Sym { col, lam, mu } = *r {
            let relu = if z[i].is_negative() { Rat::ZERO } else { z[i] };
            let v = rsub(rsub(relu, rmul(rat(lam), z[i])), rat(mu));
            e[col] = rdiv(v, rat(mu));
        }
    }
    e
}

fn exact_form(f: &Form, e: &[Rat]) -> Rat {
    let mut v = Rat::ZERO;
    for (g, x) in f.g.iter().zip(e) {
        v = radd(v, rmul(rat(*g), *x));
    }
    v
}

/// The SAME Lagrangian [`ny_proofcheck::fd::node_bound`] evaluates, in exact
/// rationals on the same stored floats (an independent re-derivation of the
/// formula in the module docs): the certified value must never exceed it.
fn exact_node_bound(margin: &Form, rows: &[(&Form, Relu, bool)], l0: &[f64], l1: &[f64]) -> Rat {
    let dual = |v: f64| if v.is_nan() { 0.0 } else { v.max(0.0) };
    let mut k = rsub(rat(margin.c_lo), rat(margin.err));
    let n = margin.g.len();
    let mut rc: Vec<Rat> = margin.g.iter().map(|&v| rat(v)).collect();
    let mut lag = Rat::ZERO;
    let mut subst = Vec::new();
    let mut mults = Vec::new();
    for (i, &(f, relax, on)) in rows.iter().enumerate() {
        let (m0, m1) = (rat(dual(l0[i])), rat(dual(l1[i])));
        let s = if on { Rat::ONE.neg() } else { Rat::ONE };
        let c = rat(f.c_lo);
        let r = radd(rat(f.err), rsub(rat(f.c_hi), rat(f.c_lo)));
        let mut row_abs = Rat::ZERO;
        for &a in &f.g {
            row_abs = radd(row_abs, rat(a).abs());
        }
        let (rho, q) = match relax {
            Relu::Sym { col, lam, mu } => {
                subst.push(col);
                let dg = rat(margin.g[col]);
                let slope = if on {
                    rsub(Rat::ONE, rat(lam))
                } else {
                    rat(lam).neg()
                };
                k = rsub(k, dg);
                let q = if on {
                    (lam < 1.0).then(|| rdiv(rmul(rat(2.0), rat(mu)), rsub(Rat::ONE, rat(lam))))
                } else {
                    (lam > 0.0).then(|| rdiv(rmul(rat(2.0), rat(mu)), rat(lam)))
                };
                (rdiv(rmul(dg, slope), rat(mu)), q)
            }
            _ => (Rat::ZERO, None),
        };
        k = radd(k, rmul(rho, c));
        k = rsub(k, rmul(rho.abs(), r));
        let b0 = radd(rmul(s.neg(), c), r);
        let b1 = match q {
            Some(q) => rmin(radd(radd(q, rmul(s, c)), r), row_abs),
            None => row_abs,
        };
        lag = radd(lag, radd(rmul(m0, b0), rmul(m1, b1)));
        mults.push(radd(rho, rmul(s, rsub(m0, m1))));
    }
    for col in subst {
        rc[col] = Rat::ZERO;
    }
    for ((f, _, _), m) in rows.iter().zip(&mults) {
        for j in 0..n {
            rc[j] = radd(rc[j], rmul(*m, rat(f.g[j])));
        }
    }
    let mut sumabs = Rat::ZERO;
    for v in rc {
        sumabs = radd(sumabs, v.abs());
    }
    rsub(rsub(k, sumabs), lag)
}

/// THE soundness property, exactly: on random nets, random slopes, random
/// nodes (sides chosen so a sampled point lies inside) and random — also
/// hostile — dual points, (1) the checker's state CONTAINS the network at
/// every sampled point (objective above `c_lo + g.e - err`, every row within
/// its remainder), (2) its certified prune bound never exceeds the exact
/// value of its own Lagrangian on the same floats, and (3) never exceeds the
/// exact objective at any sampled point of the node region.
#[test]
fn fd_certified_bound_never_exceeds_exact_rational_value() {
    let mut rng = Lcg(0x000B_23FD);
    let mut bounds_checked = 0usize;
    let mut points_inside = 0usize;
    for seed in 0..48u64 {
        let n_in = 2 + (seed % 2) as usize;
        let n_h = 3 + (seed % 3) as usize;
        let onnx = random_mlp(&mut rng, n_in, n_h);
        let net = net_from_onnx(&onnx).expect("lowers");
        let lo = vec![-1.0; n_in];
        let hi = vec![1.0; n_in];
        let lam: Vec<(&str, usize, f64)> = (0..n_h)
            .map(|i| ("z", i, (rng.next() % 9) as f64 / 8.0))
            .collect();
        let sites: Vec<(&str, usize)> = (0..n_h).map(|i| ("z", i)).collect();
        let (frame, margin, rows, _) = fd_state(
            &net,
            &lo,
            &hi,
            &lam,
            &[(0, 1.0), (1, -1.0)],
            0.0,
            0.0,
            &sites,
        )
        .expect("state");
        assert!(margin.ok && rows.iter().all(|r| r.ok));
        // (1) containment at sampled points of the box.
        let mut pts: Vec<Vec<f64>> = Vec::new();
        for _ in 0..24 {
            pts.push((0..n_in).map(|_| rng.dy()).collect());
        }
        for x in &pts {
            let (z, y) = exact_forward(&net, x);
            let e = exact_symbols(&net, &frame, x, &z);
            // review(b2-fd): THE invariant `node_bound` rests on — every
            // generator of the checker's frame lies in [-1, 1] on the box
            // (input symbols by their radius, relaxation symbols by mu's
            // height), pinned directly. The form checks below see a short mu
            // only INDIRECTLY (the substitution is an identity; what breaks
            // is the stored-coefficient rounding charge, which assumes
            // |e| <= 1): with mu's -lam*l half dropped they fired one seed
            // later than this pin (review mutation transcript).
            for (j, ej) in e.iter().enumerate() {
                assert!(
                    ej.abs() <= Rat::ONE,
                    "seed {seed}: generator {j} leaves [-1, 1] at a sampled point"
                );
            }
            let m = rsub(y[0], y[1]);
            let lower = rsub(
                radd(rat(margin.c_lo), exact_form(&margin, &e)),
                rat(margin.err),
            );
            assert!(lower <= m, "seed {seed}: objective form above the network");
            for (i, row) in rows.iter().enumerate() {
                let c = radd(rat(row.c_lo), exact_form(row, &e));
                let r = radd(rat(row.err), rsub(rat(row.c_hi), rat(row.c_lo)));
                assert!(
                    rsub(z[i], c).abs() <= r,
                    "seed {seed}: row {i} escapes its remainder"
                );
            }
        }
        // (2) and (3): nodes through each point, hostile duals included.
        for (p, x) in pts.iter().enumerate().take(8) {
            let (z, _) = exact_forward(&net, x);
            let depth = 1 + (rng.next() as usize % n_h);
            let mut chosen: Vec<usize> = (0..n_h).collect();
            for i in 0..n_h {
                let j = i + (rng.next() as usize % (n_h - i));
                chosen.swap(i, j);
            }
            chosen.truncate(depth);
            let sides: Vec<bool> = chosen.iter().map(|&i| !z[i].is_negative()).collect();
            let pool = [0.0, 0.25, 0.5, 1.0, 2.0, 1024.0, f64::NAN, -1.0];
            let l0: Vec<f64> = (0..depth).map(|_| pool[rng.next() as usize % 8]).collect();
            let l1: Vec<f64> = (0..depth).map(|_| pool[rng.next() as usize % 8]).collect();
            let named: Vec<(&str, usize, bool)> = chosen
                .iter()
                .zip(&sides)
                .map(|(&i, &on)| ("z", i, on))
                .collect();
            let node_rows: Vec<Form> = chosen.iter().map(|&i| rows[i].clone()).collect();
            let g = fd_node_bound(&net, &frame, &margin, &node_rows, &named, &l0, &l1);
            if !g.is_finite() {
                continue;
            }
            let zt = net.by_name["z"];
            let exact_rows: Vec<(&Form, Relu, bool)> = node_rows
                .iter()
                .zip(chosen.iter().zip(&sides))
                .map(|(f, (&i, &on))| (f, frame.relu(zt, i), on))
                .collect();
            let gx = exact_node_bound(&margin, &exact_rows, &l0, &l1);
            assert!(
                rat(g) <= gx,
                "seed {seed} point {p}: certified {g:e} above its exact Lagrangian"
            );
            bounds_checked += 1;
            for q in &pts {
                let (zq, yq) = exact_forward(&net, q);
                let inside = chosen.iter().zip(&sides).all(|(&i, &on)| {
                    if on {
                        !zq[i].is_negative()
                    } else {
                        !zq[i].is_positive()
                    }
                });
                if inside {
                    points_inside += 1;
                    assert!(
                        rat(g) <= rsub(yq[0], yq[1]),
                        "seed {seed}: certified {g:e} above the exact objective inside the node"
                    );
                }
            }
        }
    }
    assert!(bounds_checked > 200, "only {bounds_checked} bounds: VOID");
    assert!(
        points_inside > 400,
        "only {points_inside} inside points: VOID"
    );
}

/// review(fd-sampling): the exact-rational pin above draws node depths of at
/// most five, so the regime where the RUNTIME per-node sample now mostly
/// does not run (depth past [`FD_NODE_SAMPLE_ALL_DEPTH`], one node in
/// [`FD_NODE_SAMPLE_ONE_IN`] sampled, and a sampled point almost never
/// inside) had no exact pin past depth five. The same two properties, on
/// wider random nets (8 to 12 hidden neurons) and nodes of depth 5 to 12
/// through a sampled point, with hostile dual points: (2) the certified
/// prune bound never exceeds the exact value of its own Lagrangian on the
/// same floats, and (3) never exceeds the exact objective at any sampled
/// point inside the node region.
#[test]
fn fd_deep_certified_bound_never_exceeds_exact_rational_value() {
    let mut rng = Lcg(0xD33_9FD);
    let (mut checked, mut inside, mut deepest) = (0usize, 0usize, 0usize);
    for seed in 0..24u64 {
        let n_in = 2 + (seed % 2) as usize;
        let n_h = 8 + 2 * (seed % 3) as usize;
        let onnx = random_mlp(&mut rng, n_in, n_h);
        let net = net_from_onnx(&onnx).expect("lowers");
        let lo = vec![-1.0; n_in];
        let hi = vec![1.0; n_in];
        let lam: Vec<(&str, usize, f64)> = (0..n_h)
            .map(|i| ("z", i, (rng.next() % 9) as f64 / 8.0))
            .collect();
        let sites: Vec<(&str, usize)> = (0..n_h).map(|i| ("z", i)).collect();
        let (frame, margin, rows, _) = fd_state(
            &net,
            &lo,
            &hi,
            &lam,
            &[(0, 1.0), (1, -1.0)],
            0.0,
            0.0,
            &sites,
        )
        .expect("state");
        assert!(margin.ok && rows.iter().all(|r| r.ok), "seed {seed}");
        let pts: Vec<Vec<f64>> = (0..32)
            .map(|_| (0..n_in).map(|_| rng.dy()).collect())
            .collect();
        let exact: Vec<(Vec<Rat>, Vec<Rat>)> = pts.iter().map(|x| exact_forward(&net, x)).collect();
        let zt = net.by_name["z"];
        for (p, (z, _)) in exact.iter().enumerate().take(12) {
            let span = n_h - FD_NODE_SAMPLE_ALL_DEPTH;
            let depth = FD_NODE_SAMPLE_ALL_DEPTH + 1 + (rng.next() as usize % span);
            let mut chosen: Vec<usize> = (0..n_h).collect();
            for i in 0..n_h {
                let j = i + (rng.next() as usize % (n_h - i));
                chosen.swap(i, j);
            }
            chosen.truncate(depth);
            let sides: Vec<bool> = chosen.iter().map(|&i| !z[i].is_negative()).collect();
            let pool = [0.0, 0.25, 0.5, 1.0, 2.0, 1024.0, f64::NAN, -1.0];
            let l0: Vec<f64> = (0..depth).map(|_| pool[rng.next() as usize % 8]).collect();
            let l1: Vec<f64> = (0..depth).map(|_| pool[rng.next() as usize % 8]).collect();
            let named: Vec<(&str, usize, bool)> = chosen
                .iter()
                .zip(&sides)
                .map(|(&i, &on)| ("z", i, on))
                .collect();
            let node_rows: Vec<Form> = chosen.iter().map(|&i| rows[i].clone()).collect();
            let g = fd_node_bound(&net, &frame, &margin, &node_rows, &named, &l0, &l1);
            if !g.is_finite() {
                continue;
            }
            let exact_rows: Vec<(&Form, Relu, bool)> = node_rows
                .iter()
                .zip(chosen.iter().zip(&sides))
                .map(|(f, (&i, &on))| (f, frame.relu(zt, i), on))
                .collect();
            let gx = exact_node_bound(&margin, &exact_rows, &l0, &l1);
            assert!(
                rat(g) <= gx,
                "seed {seed} point {p} depth {depth}: certified {g:e} above its exact Lagrangian"
            );
            checked += 1;
            deepest = deepest.max(depth);
            for (zq, yq) in &exact {
                let within = chosen.iter().zip(&sides).all(|(&i, &on)| {
                    if on {
                        !zq[i].is_negative()
                    } else {
                        !zq[i].is_positive()
                    }
                });
                if within {
                    inside += 1;
                    assert!(
                        rat(g) <= rsub(yq[0], yq[1]),
                        "seed {seed} depth {depth}: certified {g:e} above the exact objective \
                         inside the node"
                    );
                }
            }
        }
    }
    assert!(checked > 150, "only {checked} deep bounds: VOID");
    assert!(
        inside >= checked,
        "{inside} inside points for {checked} nodes (each holds its own point): VOID"
    );
    assert!(deepest >= 10, "deepest node {deepest}: VOID");
}

/// The certified bound is TIGHT where the exact one is: on the fixture's
/// tree each prune certifies `δ` up to a rounding slack far below `δ`, and
/// the mutation's multiplier (`+4δ`) drops the EXACT value below zero —
/// the mutation test above rests on that, not on a lucky rounding.
#[test]
fn fd_fixture_bounds_match_the_exact_tree() {
    let net = net_from_onnx(&mlp_onnx(D, 0.0)).expect("lowers");
    let (frame, margin, rows, _) = fd_state(
        &net,
        &[-1.0, -1.0],
        &[1.0, 1.0],
        &[("z", 0, 0.5), ("z", 1, 0.5)],
        &[(0, 1.0), (1, -1.0)],
        0.0,
        0.0,
        &[("z", 0), ("z", 1)],
    )
    .expect("state");
    let zt = net.by_name["z"];
    for (lits, l0) in prunes() {
        let sides: Vec<bool> = lits
            .iter()
            .map(|l| match l {
                L::R(_, _, p) => *p,
                L::X(..) => unreachable!(),
            })
            .collect();
        let named = [("z", 0usize, sides[0]), ("z", 1usize, sides[1])];
        let g = fd_node_bound(&net, &frame, &margin, &rows, &named, &l0, &[0.0, 0.0]);
        assert!(g > 0.999 * D && g < D, "prune {sides:?}: {g:e}");
        let rows3: Vec<(&Form, Relu, bool)> = rows
            .iter()
            .zip(&sides)
            .enumerate()
            .map(|(i, (f, &on))| (f, frame.relu(zt, i), on))
            .collect();
        let mut bumped = l0.clone();
        if let Some(i) = l0.iter().position(|&v| v > 0.0) {
            bumped[i] += 4.0 * D;
            let gx = exact_node_bound(&margin, &rows3, &bumped, &[0.0, 0.0]);
            assert!(
                gx < rat(0.0),
                "prune {sides:?}: the mutation is not beyond rounding"
            );
            let g2 = fd_node_bound(&net, &frame, &margin, &rows, &named, &bumped, &[0.0, 0.0]);
            assert!(
                g2.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater),
                "prune {sides:?}: {g2:e}"
            );
        }
    }
    // The checker's state root is the fixture's `δ - 1`, rounded down.
    let root = fd_node_bound(&net, &frame, &margin, &[], &[], &[], &[]);
    assert!(root < D - 1.0 && root > D - 1.0 - 1e-12, "root {root:e}");
}

// ---------------------------------------------------------------------------
// #fd-compact-checkable: a producer's ranked COMPACT state
// ---------------------------------------------------------------------------
//
// A producer's #breakpoint-dual RANKED compact state keeps rows only for its
// retained split candidates, while every omitted unstable neuron keeps its
// relaxation symbol at the class slope. Its bundle therefore carries the
// slope of EVERY unstable neuron (`hint.fd_lambda`, `hint.fd_state.relaxed`
// counting them), and the checker's own state must be rebuilt from that
// whole table: the split rows' slopes alone describe a different
// computation.
//
// # The compact fixture (exact arithmetic)
//
// The base MLP plus one more unstable neuron `z3 = x0` that feeds the margin
// but is never split (the OMITTED row):
// `y0 = relu(z0) + relu(z1) + relu(z3) + δ`, `y1 = 0` (and, for the
// wrong-class pin, `y2 = 0`). The margin's exact minimum is `δ`. With the
// producer's slopes `λ0 = λ1 = 1/2` and `λ3 = 0` (`μ = 1/2` all three) the
// state is `margin = e_x0 + 3/2 + (e_z0 + e_z1 + e_z3)/2 + δ`, and the four
// prunes of the base tree, at the base dual points, are each exactly `δ`:
// the omitted symbol contributes `1/2 - |1/2| = 0`. At the CHORD slope
// `λ3 = 1/2` (`μ3 = 1/4`) the margin gains `x0/2` that no split row's
// multiplier cancels and every prune is `δ - 1/2 < 0`; at `λ3 = 1/4`
// (`μ3 = 3/8`) every prune is `δ - 1/4 < 0`; at `λ3 = 1` (`μ3 = 1/2`)
// `δ - 1 < 0`.

/// The compact fixture's network over `classes` outputs (2 or 3).
fn compact_onnx(classes: usize) -> Vec<u8> {
    let mut w2 = vec![1.0, 1.0, 0.0, 1.0];
    w2.extend(std::iter::repeat_n(0.0, 4 * (classes - 1)));
    let mut b2 = vec![D];
    b2.extend(std::iter::repeat_n(0.0, classes - 1));
    model(
        "x",
        &[1, 2],
        &[
            node("Gemm", &["x", "W1", "B1"], &["z"], vec![A::I("transB", 1)]),
            node("Relu", &["z"], &["h"], vec![]),
            node("Gemm", &["h", "W2", "B2"], &["y"], vec![A::I("transB", 1)]),
        ],
        &[
            tf("W1", &[4, 2], &[1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 0.0]),
            tf("B1", &[4], &[0.0, 0.0, 5.0, 0.0]),
            tf("W2", &[classes as i64, 4], &w2),
            tf("B2", &[classes as i64], &b2),
        ],
        "y",
        &[1, classes as i64],
    )
}

/// A compact state's `hint.fd_state`: two split rows (`unstable`) and, when
/// declared, the relaxation symbols its table covers (`relaxed`).
fn compact_state_line(id: u64, class: u64, prunes: u64, relaxed: Option<u64>) -> String {
    let mut s = fd_state_line(id, class, &[(0, 0), (0, 1)], prunes);
    if let Some(r) = relaxed {
        s.pop();
        s.push_str(&format!(",\"relaxed\":{r}}}"));
    }
    s
}

/// The compact bundle: one compact state per `(state, class)` declaring
/// `relaxed` with the slope table `(idx, val)`, the base prunes per state,
/// verdict requiring `required`.
fn compact_bundle(
    fx: &Fixture,
    clause_list: &[(u64, Vec<String>)],
    states: &[(u64, u64)],
    table: (&[u64], &[f64], Option<u64>),
    nodes: &[(Vec<L>, Vec<f64>, u64, u64)],
    required: &[u64],
) -> Vec<u8> {
    let mut lines = vec![
        fx.header(),
        sites(&[(0, "z", 4, 1)]),
        clauses(0, clause_list),
    ];
    for &(state, class) in states {
        lines.push(compact_state_line(state, class, 4, table.2));
        lines.push(fd_lambda_line(state, 0, table.0, table.1));
    }
    for (k, (lits, l0, class, state)) in nodes.iter().enumerate() {
        let l1 = vec![0.0; l0.len()];
        lines.push(fd_node_line(k as u64, lits, &[*class], *state, l0, &l1, D));
    }
    lines.push(verdict(required, nodes.len() as u64));
    lines.push(published("unsat"));
    finish(&lines)
}

/// The producer's complete table: both split rows and the omitted `z3`.
const COMPACT_IDX: [u64; 3] = [0, 1, 3];
const COMPACT_VAL: [f64; 3] = [0.5, 0.5, 0.0];

/// A compact state streamed with its whole class relaxation is VERIFIED,
/// every prune confirmed by the fd rung, and the report shows the checker
/// relaxed every unstable neuron at the producer's slope (`chord_lambda`
/// 0) and echoes the producer's `relaxed` count. An extra entry for a
/// neuron that is stable in the checker's box (`z2`) relaxes nothing
/// (`hint_unused` 1) and changes nothing.
#[test]
fn fd_accepts_a_compact_state_with_its_class_relaxation() {
    let fx = Fixture::new("fd_compact_accept", compact_onnx(2), &mlp_vnnlib());
    let bytes = compact_bundle(
        &fx,
        &[(1, vec![clause01()])],
        &[(7, 1)],
        (&COMPACT_IDX, &COMPACT_VAL, Some(3)),
        &base_nodes(1, 7),
        &[1],
    );
    for refine in [true, false] {
        let o = fx.check_with(&format!("compact_{refine}"), &bytes, |a| a.refine = refine);
        assert_eq!(o.verdict, Verdict::Verified, "{}", o.report_json);
        let fd = fd_section(&o);
        assert_eq!(
            fd["discharges_confirmed"].as_u64(),
            Some(4),
            "{}",
            o.report_json
        );
        let st = &fd["states"][0];
        assert_eq!(st["unstable"].as_u64(), Some(3));
        assert_eq!(st["hinted_lambda"].as_u64(), Some(3));
        assert_eq!(st["chord_lambda"].as_u64(), Some(0));
        assert_eq!(st["hint_unused"].as_u64(), Some(0));
        assert_eq!(st["lane_unstable"].as_u64(), Some(2));
        assert_eq!(st["lane_relaxed"].as_u64(), Some(3));
        assert_eq!(st["table_entries"].as_u64(), Some(3));
        assert_eq!(report(&o)["ladder"]["fd"].as_u64(), Some(4));
    }
    let bytes = compact_bundle(
        &fx,
        &[(1, vec![clause01()])],
        &[(7, 1)],
        (&[0, 1, 2, 3], &[0.5, 0.5, 0.7, 0.0], Some(4)),
        &base_nodes(1, 7),
        &[1],
    );
    let o = fx.check("compact_extra", &bytes);
    assert_eq!(o.verdict, Verdict::Verified, "{}", o.report_json);
    let st = &fd_section(&o)["states"][0];
    assert_eq!(st["hinted_lambda"].as_u64(), Some(3));
    assert_eq!(st["hint_unused"].as_u64(), Some(1));
}

/// The compact state WITHOUT its omitted row's slope. Declared complete
/// (`relaxed` 3) with a two-entry table it is REFUSED (`fd_table_count`):
/// the checker never fills a declared table's hole. Undeclared — the
/// producer's historical table, its split rows only — the checker rebuilds
/// it at its chord default (`λ3 = 1/2`, `chord_lambda` 1): a DIFFERENT
/// computation, in which no prune is `> 0`. Either way the fd rung confirms
/// nothing and the bundle is never verified.
#[test]
fn fd_compact_state_without_its_omitted_slope_is_not_confirmed() {
    let fx = Fixture::new("fd_compact_drop", compact_onnx(2), &mlp_vnnlib());
    for declared in [Some(3), None] {
        let bytes = compact_bundle(
            &fx,
            &[(1, vec![clause01()])],
            &[(7, 1)],
            (&[0, 1], &[0.5, 0.5], declared),
            &base_nodes(1, 7),
            &[1],
        );
        let o = fx.check(&format!("compact_drop_{}", declared.is_some()), &bytes);
        assert_eq!(o.verdict, Verdict::Unconfirmed, "{}", o.report_json);
        let fd = fd_section(&o);
        assert_eq!(
            fd["discharges_confirmed"].as_u64(),
            Some(0),
            "{}",
            o.report_json
        );
        let st = &fd["states"][0];
        assert_eq!(st["table_entries"].as_u64(), Some(2));
        if declared.is_some() {
            assert_eq!(st["built"], Json::Bool(false), "{}", o.report_json);
            assert_eq!(st["refused"].as_str(), Some("fd_table_count"));
        } else {
            assert_eq!(st["chord_lambda"].as_u64(), Some(1));
            assert_eq!(st["hinted_lambda"].as_u64(), Some(2));
            assert!(
                o.report_json.contains("fd_weaker_state"),
                "{}",
                o.report_json
            );
        }
    }
}

/// The omitted row's slope PERTURBED (`λ3 = 1/4` or `1`, away from the
/// producer's `0`): every prune's exact value is negative, so the fd rung
/// confirms nothing and the bundle is never verified. The checker evaluates
/// what the table says; it never assumes the table is the producer's.
#[test]
fn fd_compact_state_with_a_perturbed_omitted_slope_is_not_confirmed() {
    let fx = Fixture::new("fd_compact_perturb", compact_onnx(2), &mlp_vnnlib());
    for lam in [0.25, 1.0] {
        let bytes = compact_bundle(
            &fx,
            &[(1, vec![clause01()])],
            &[(7, 1)],
            (&COMPACT_IDX, &[0.5, 0.5, lam], Some(3)),
            &base_nodes(1, 7),
            &[1],
        );
        let o = fx.check(&format!("compact_perturb_{lam}"), &bytes);
        assert_eq!(o.verdict, Verdict::Unconfirmed, "{}", o.report_json);
        let fd = fd_section(&o);
        assert_eq!(
            fd["discharges_confirmed"].as_u64(),
            Some(0),
            "{}",
            o.report_json
        );
        assert_eq!(fd["states"][0]["hinted_lambda"].as_u64(), Some(3));
    }
}

/// Compact prunes relabelled to the WRONG class leave the right class's
/// cover open: INVALID. Two clauses with identical margins (classes 1 and 2
/// against class 0), each class's compact state streamed with its whole
/// table, verify; moving any class-1 prune to class 2 does not.
#[test]
fn fd_compact_state_rejects_a_wrong_class() {
    let vnn = robustness_vnnlib(&["-1", "-1"], &["1", "1"], 3, 0, &[1, 2]);
    let fx = Fixture::new("fd_compact_class", compact_onnx(3), &vnn);
    let cl = [
        (1, vec!["Y0:1,Y1:-1<=0".to_string()]),
        (2, vec!["Y0:1,Y2:-1<=0".to_string()]),
    ];
    let mut nodes = base_nodes(1, 7);
    nodes.extend(base_nodes(2, 8));
    let good = compact_bundle(
        &fx,
        &cl,
        &[(7, 1), (8, 2)],
        (&COMPACT_IDX, &COMPACT_VAL, Some(3)),
        &nodes,
        &[1, 2],
    );
    let o = fx.check("compact_both", &good);
    assert_eq!(o.verdict, Verdict::Verified, "{}", o.report_json);
    assert_eq!(fd_section(&o)["discharges_confirmed"].as_u64(), Some(8));
    for k in 0..4 {
        let mut bad = nodes.clone();
        bad[k].2 = 2;
        let bytes = compact_bundle(
            &fx,
            &cl,
            &[(7, 1), (8, 2)],
            (&COMPACT_IDX, &COMPACT_VAL, Some(3)),
            &bad,
            &[1, 2],
        );
        let o = fx.check(&format!("compact_wrong{k}"), &bytes);
        assert_eq!(o.verdict, Verdict::Invalid, "{}", o.report_json);
    }
}
