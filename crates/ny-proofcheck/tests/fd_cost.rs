// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! B2.3 unit-level MEASUREMENT (behind the `benchmarks` feature, so absent
//! from the default lane): the checker's wall on
//! frozen-dual prune trees over the ResNet-shaped synthetic fixture
//! (`support::conv_onnx`: Conv, BN, Relu, Conv, residual Add, Relu, Gemm,
//! Relu, Gemm), one complete binary split tree per depth, every prune a
//! `hint.fd` node at the zero dual point. It asserts only that the measured
//! path IS the fd rung (every prune confirmed there, the bundle verified),
//! so a timing from a bundle the rung did not confirm is VOID, not a number.
//!
//! The report's `timing.nodes_s` includes the join of the checker's progress
//! heartbeat thread, which sleeps in 200 ms steps, so a node phase shorter
//! than ~0.2 s reads as ~0.2 s: the depth-14 tree is there to resolve the
//! wall, and `FD_UNIT` prints the two unit costs the per-node phase is made
//! of on this network (one interval forward pass at a point = one sampled
//! point; one fd node bound = the rung itself), so the forward-pass counts
//! in `node_sampling.points` convert to time at any depth.
//!
//! It uses the checker's public API alone, so the same file runs against an
//! older checker (`node_sampling` then prints `absent`; that checker samples
//! the centre plus 14 points of EVERY fd-confirmed node, i.e. `15 n`):
//!
//! ```text
//! targo --unverified test -p ny-proofcheck --features benchmarks --test fd_cost -- --nocapture --test-threads=1
//! ```

mod support;

use std::hint::black_box;
use std::time::Instant;

use ny_cert::json::Json;
use ny_proofcheck::ibp::eval_point;
use ny_proofcheck::testkit::{fd_node_bound, fd_state, net_from_onnx};
use ny_proofcheck::Verdict;
use support::*;

/// Depths of the measured trees.
const DEPTHS: [usize; 4] = [4, 7, 10, 14];

/// The fixture's Relu sites `(site id, size)`: `b1` 32, `a1` 16, `g1` 4.
const SITES: [(u64, u64); 3] = [(0, 32), (1, 16), (2, 4)];

/// A complete binary tree of fd prunes over the split sites `order`
/// (`(site id, neuron)`, depth = `order.len()`), all discharging class 1
/// under one state; returns the bytes and the node count.
fn tree_bundle(fx: &Fixture, order: &[(u64, u64)]) -> (Vec<u8>, u64) {
    let depth = order.len();
    let mut lines = vec![
        fx.header(),
        sites(&[(0, "b1", 32, 2), (1, "a1", 16, 5), (2, "g1", 4, 8)]),
        clauses(0, &[(1, vec![clause01()])]),
    ];
    let n = 1u64 << depth;
    lines.push(fd_state_line(7, 1, order, n));
    let zeros = vec![0.0; depth];
    for leaf in 0..n {
        let lits: Vec<L> = order
            .iter()
            .enumerate()
            .map(|(i, &(t, j))| L::R(t, j, (leaf >> i) & 1 == 1))
            .collect();
        lines.push(fd_node_line(leaf, &lits, &[1], 7, &zeros, &zeros, 1.0));
    }
    lines.push(verdict(&[1], n));
    lines.push(published("unsat"));
    (finish(&lines), n)
}

/// The neurons the checker's OWN root bounds leave unstable, found through
/// the public report: a depth-1 split on a neuron its root box already fixes
/// has one child the literal clamp empties before any rung runs
/// (`nodes.empty > 0`), which would measure nothing. Kept: both children
/// reach the fd rung and it confirms both.
fn unstable_sites(fx: &Fixture) -> Vec<(u64, u64)> {
    let mut out = Vec::new();
    for &(t, size) in &SITES {
        for j in 0..size {
            let (bytes, _) = tree_bundle(fx, &[(t, j)]);
            let o = fx.check_with("cost_probe", &bytes, |a| a.threads = Some(1));
            let r = report(&o);
            if r["nodes"]["empty"].as_u64() == Some(0) && r["ladder"]["fd"].as_u64() == Some(2) {
                out.push((t, j));
            }
        }
    }
    out
}

/// Fourteen split sites spread over the three layers (up to 8 of `b1`, 4
/// of `a1`, 2 of `g1`, the rest from whatever unstable neuron is left).
fn split_order(unstable: &[(u64, u64)]) -> Vec<(u64, u64)> {
    let mut order: Vec<(u64, u64)> = Vec::new();
    for (site, want) in [(0u64, 8usize), (1, 4), (2, 2)] {
        order.extend(unstable.iter().filter(|s| s.0 == site).take(want));
    }
    for s in unstable {
        if order.len() >= 14 {
            break;
        }
        if !order.contains(s) {
            order.push(*s);
        }
    }
    order.truncate(14);
    order
}

/// Unit costs on the fixture, in microseconds (min over 5 batches of 2000):
/// one interval forward pass at a point (`ibp::eval_point`, what each point
/// of the per-node sample costs) and, per depth, one fd node bound over the
/// first `depth` split sites (`fd::node_bound` through the testkit, on a
/// pure-IBP state of the same network: what the rung costs per prune).
fn unit_costs(order: &[(u64, u64)]) -> (f64, Vec<(usize, f64)>) {
    const N: usize = 2000;
    let net = net_from_onnx(&conv_onnx()).expect("lowers");
    let (lo, hi) = (vec![0.0; 16], vec![1.0; 16]);
    let names = ["b1", "a1", "g1"];
    let sites: Vec<(&str, usize)> = order
        .iter()
        .map(|&(t, j)| (names[t as usize], j as usize))
        .collect();
    let (frame, margin, rows, _) = fd_state(
        &net,
        &lo,
        &hi,
        &[],
        &[(0, 1.0), (1, -1.0)],
        0.0,
        0.0,
        &sites,
    )
    .expect("state");
    let mut eval_us = f64::INFINITY;
    for _ in 0..5 {
        let t = Instant::now();
        for i in 0..N {
            let x: Vec<f64> = (0..16)
                .map(|k| ((i * 7 + k * 13) % 97) as f64 / 97.0)
                .collect();
            black_box(eval_point(&net, black_box(&x)));
        }
        eval_us = eval_us.min(t.elapsed().as_secs_f64() * 1e6 / N as f64);
    }
    let mut bound_us = Vec::new();
    for depth in DEPTHS {
        let named: Vec<(&str, usize, bool)> = sites[..depth]
            .iter()
            .enumerate()
            .map(|(i, &(n, j))| (n, j, i % 2 == 0))
            .collect();
        let zeros = vec![0.0; depth];
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let t = Instant::now();
            for _ in 0..N {
                black_box(fd_node_bound(
                    &net,
                    &frame,
                    &margin,
                    &rows[..depth],
                    black_box(&named),
                    &zeros,
                    &zeros,
                ));
            }
            best = best.min(t.elapsed().as_secs_f64() * 1e6 / N as f64);
        }
        bound_us.push((depth, best));
    }
    (eval_us, bound_us)
}

fn secs(j: &Json) -> f64 {
    match j {
        Json::F64(v) => *v,
        Json::U64(v) => *v as f64,
        Json::I64(v) => *v as f64,
        other => panic!("not a number: {other:?}"),
    }
}

#[test]
fn fd_checker_cost_on_the_resnet_shaped_fixture() {
    let lo = vec!["0"; 16];
    let hi = vec!["1"; 16];
    let fx = Fixture::new(
        "fd_cost",
        conv_onnx(),
        &robustness_vnnlib(&lo, &hi, 2, 0, &[1]),
    );
    const REPS: usize = 5;
    let unstable = unstable_sites(&fx);
    let order = split_order(&unstable);
    println!(
        "FD_COST_SITES unstable={} of 52, split order (site id, neuron) = {order:?}",
        unstable.len()
    );
    assert_eq!(order.len(), 14, "VOID: fewer than 14 unstable split sites");
    let (eval_us, bound_us) = unit_costs(&order);
    println!(
        "FD_UNIT eval_point_us={eval_us:.3} node_bound_us(depth)={:?}",
        bound_us
            .iter()
            .map(|(d, us)| format!("{d}:{us:.3}"))
            .collect::<Vec<_>>()
    );
    let mut void = Vec::new();
    for depth in DEPTHS {
        let (bytes, n) = tree_bundle(&fx, &order[..depth]);
        let mut nodes_s: Vec<f64> = Vec::with_capacity(REPS);
        let mut total_s: Vec<f64> = Vec::with_capacity(REPS);
        let mut sampling = String::from("absent");
        // Forward passes of the per-node sample: the report's count, or the
        // older checker's 15 per fd-confirmed node when it reports none.
        let mut passes = 15 * n;
        for rep in 0..REPS {
            let o = fx.check_with(&format!("cost_d{depth}"), &bytes, |a| {
                a.threads = Some(1);
            });
            let r = report(&o);
            if o.verdict != Verdict::Verified || r["ladder"]["fd"].as_u64() != Some(n) {
                // VOID: the measured path is not the fd rung on every prune.
                void.push(format!(
                    "depth {depth}: verdict {:?} ladder.fd {:?} of {n} empty {:?}",
                    o.verdict,
                    r["ladder"]["fd"].as_u64(),
                    r["nodes"]["empty"].as_u64()
                ));
                break;
            }
            nodes_s.push(secs(&r["timing"]["nodes_s"]));
            total_s.push(secs(&r["timing"]["total_s"]));
            let s = &r["fd"]["node_sampling"];
            if rep == 0 && !matches!(s, Json::Null) {
                passes = s["points"].as_u64().expect("node_sampling.points");
                let g = |k: &str| s[k].as_u64().map_or("-".to_string(), |v| v.to_string());
                sampling = format!(
                    "nodes={} shallow={} hashed={} skipped={} points={} in_region={}",
                    g("nodes"),
                    g("shallow"),
                    g("hashed"),
                    g("skipped"),
                    g("points"),
                    g("in_region")
                );
            }
        }
        if nodes_s.len() < REPS {
            println!("FD_COST depth={depth} fd_nodes={n} VOID");
            continue;
        }
        nodes_s.sort_by(f64::total_cmp);
        total_s.sort_by(f64::total_cmp);
        let bound = bound_us
            .iter()
            .find(|(d, _)| *d == depth)
            .map_or(f64::NAN, |(_, us)| *us);
        // The per-node phase as the two unit costs: the sample's forward
        // passes plus one bound per prune.
        let model_ms = (passes as f64 * eval_us + n as f64 * bound) / 1e3;
        println!(
            "FD_COST depth={depth} fd_nodes={n} reps={REPS} threads=1 nodes_s_min={:.5} \
             nodes_s_med={:.5} us_per_node_min={:.2} total_s_min={:.5} fwd_passes={passes} \
             model_ms={model_ms:.2} (sample {:.2} + rung {:.2}) sampling: {sampling}",
            nodes_s[0],
            nodes_s[REPS / 2],
            1e6 * nodes_s[0] / n as f64,
            total_s[0],
            passes as f64 * eval_us / 1e3,
            n as f64 * bound / 1e3
        );
    }
    assert!(void.is_empty(), "VOID measurement: {void:?}");
}
