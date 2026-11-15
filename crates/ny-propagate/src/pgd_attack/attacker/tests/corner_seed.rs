// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Box-corner restart seeds (#attack-vertex-geometry, `attack.corner_seed_restarts`).
//!
//! Two properties are pinned here. SOUNDNESS-NEUTRALITY: a corner seed is an
//! exact member of the input box (every coordinate is a box endpoint copied
//! verbatim), so the seed cannot move a candidate outside the property's input
//! set. DARK BY DEFAULT: with `corner_seed_restarts == 0` the seed-indexed
//! initializer is the historical one — same point, same RNG stream position.

use ndarray::{arr1, arr2};
use ny_tensor::BoundedTensor;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use super::super::init::corner_slot_for_seed;
use crate::layers::*;
use crate::pgd_attack::{corner_seed_kind, corner_seed_point, PgdAttacker, PgdConfig};
use crate::Network;

/// y = x0 + x1: the only points reaching 2 - 1e-3 on the unit square sit
/// within 1e-3 of the upper corner.
fn sum_network() -> Network {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0_f32, 1.0]]), Some(arr1(&[0.0_f32]))).unwrap(),
    ));
    network
}

fn unit_box_2d() -> BoundedTensor {
    BoundedTensor::new(
        arr1(&[0.0_f32, 0.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .unwrap()
}

/// SAMPLED CONTAINMENT: over random boxes (including degenerate and negative
/// intervals) every corner slot is an exact box member whose every coordinate
/// IS an endpoint, slot 0 is the lower corner and slot 1 the upper corner.
#[test]
fn corner_seed_points_are_exact_box_members() {
    let mut rng = StdRng::seed_from_u64(0xC0_4E_45);
    for case in 0..200 {
        let dim = 1 + case % 37;
        let mut lower = Vec::with_capacity(dim);
        let mut upper = Vec::with_capacity(dim);
        for d in 0..dim {
            let a = rng.random_range(-1.0e3_f32..=1.0e3_f32);
            let width = if d % 5 == 0 {
                0.0
            } else {
                rng.random_range(0.0_f32..=10.0_f32)
            };
            lower.push(a);
            upper.push(a + width);
        }
        for slot in 0..8 {
            let point = corner_seed_point(slot, &lower, &upper, || rng.random::<bool>());
            assert_eq!(point.len(), dim, "case {case} slot {slot}: arity");
            for d in 0..dim {
                let value = point[d];
                assert!(
                    value.to_bits() == lower[d].to_bits() || value.to_bits() == upper[d].to_bits(),
                    "case {case} slot {slot} dim {d}: {value} is not an endpoint of [{}, {}]",
                    lower[d],
                    upper[d]
                );
                assert!(
                    lower[d] <= value && value <= upper[d],
                    "case {case} slot {slot} dim {d}: {value} escapes [{}, {}]",
                    lower[d],
                    upper[d]
                );
            }
            match slot {
                0 => assert_eq!(point, lower, "slot 0 is the lower corner"),
                1 => assert_eq!(point, upper, "slot 1 is the upper corner"),
                _ => {}
            }
        }
    }
    assert_eq!(corner_seed_kind(0), "lo");
    assert_eq!(corner_seed_kind(1), "hi");
    assert_eq!(corner_seed_kind(2), "vertex");
    assert_eq!(corner_seed_kind(9), "vertex");
}

/// Slots 0 and 1 are deterministic and consume no coin; a vertex slot draws
/// exactly one coin per dimension.
#[test]
fn corner_seed_coin_draws_are_exactly_one_per_dimension_on_vertex_slots() {
    let lower = [0.0_f32, -1.0, 2.0];
    let upper = [1.0_f32, 1.0, 2.5];
    for slot in 0..2 {
        let mut draws = 0usize;
        let _ = corner_seed_point(slot, &lower, &upper, || {
            draws += 1;
            true
        });
        assert_eq!(draws, 0, "slot {slot} must not consume the coin");
    }
    let mut draws = 0usize;
    let point = corner_seed_point(2, &lower, &upper, || {
        draws += 1;
        draws % 2 == 1
    });
    assert_eq!(draws, 3);
    assert_eq!(
        point,
        vec![0.0, 1.0, 2.0],
        "coin true -> lower, false -> upper"
    );
}

/// The seed -> restart-index recovery every restart loop relies on, including
/// a base seed next to `u64::MAX` (the loops use `wrapping_add`).
#[test]
fn corner_slot_for_seed_recovers_the_restart_index() {
    for base in [0_u64, 42, u64::MAX - 1] {
        for restart in 0..6_u64 {
            let seed = base.wrapping_add(restart);
            let slot = corner_slot_for_seed(base, seed, 3);
            if restart < 3 {
                assert_eq!(
                    slot,
                    Some(restart as usize),
                    "base {base} restart {restart}"
                );
            } else {
                assert_eq!(slot, None, "base {base} restart {restart}");
            }
            assert_eq!(
                corner_slot_for_seed(base, seed, 0),
                None,
                "corner_seed_restarts == 0 never yields a slot"
            );
        }
    }
}

/// DARK BY DEFAULT: with the default `corner_seed_restarts == 0` the
/// seed-indexed initializer returns the historical point AND leaves the RNG at
/// the historical stream position, for every restart index.
#[test]
fn corner_seed_restarts_zero_is_byte_identical_to_historical_init() {
    assert_eq!(PgdConfig::default().corner_seed_restarts, 0);
    let network = sum_network();
    let bounds = unit_box_2d();
    let attacker = PgdAttacker::new(PgdConfig {
        seed: 42,
        ..Default::default()
    });
    for restart in 0..6_u64 {
        let seed = 42_u64.wrapping_add(restart);
        let mut indexed_rng = attacker.seeded_rng(seed);
        let mut historical_rng = attacker.seeded_rng(seed);
        let indexed = attacker
            .initialize_restart_for_seed(&network, &bounds, &mut indexed_rng, seed)
            .unwrap();
        let historical = attacker
            .initialize_restart(&network, &bounds, &mut historical_rng)
            .unwrap();
        assert_eq!(indexed, historical, "restart {restart}: seed point");
        assert_eq!(
            indexed_rng.random::<u64>(),
            historical_rng.random::<u64>(),
            "restart {restart}: RNG stream position"
        );
    }
}

/// ARMED: restart 0 is the lower corner, restart 1 the upper corner, restart 2
/// a vertex, and restart 3 (past N) keeps its historical seed exactly.
#[test]
fn corner_seed_restarts_seed_lo_hi_vertex_then_the_historical_schedule() {
    let network = sum_network();
    let bounds = unit_box_2d();
    let armed = PgdAttacker::new(PgdConfig {
        seed: 7,
        corner_seed_restarts: 3,
        ..Default::default()
    });
    let dark = PgdAttacker::new(PgdConfig {
        seed: 7,
        ..Default::default()
    });
    let init = |attacker: &PgdAttacker<'static>, restart: u64| {
        init_point(attacker, &network, &bounds, 7_u64.wrapping_add(restart))
    };
    assert_eq!(init(&armed, 0), vec![0.0, 0.0], "restart 0 = lower corner");
    assert_eq!(init(&armed, 1), vec![1.0, 1.0], "restart 1 = upper corner");
    let vertex = init(&armed, 2);
    assert!(
        vertex.iter().all(|&v| v == 0.0 || v == 1.0),
        "restart 2 = a box vertex, got {vertex:?}"
    );
    assert_eq!(
        init(&armed, 3),
        init(&dark, 3),
        "restart 3 is past N and keeps the historical seed"
    );
}

fn init_point(
    attacker: &PgdAttacker<'static>,
    network: &Network,
    bounds: &BoundedTensor,
    seed: u64,
) -> Vec<f32> {
    let mut rng = attacker.seeded_rng(seed);
    attacker
        .initialize_restart_for_seed(network, bounds, &mut rng, seed)
        .unwrap()
        .iter()
        .copied()
        .collect()
}

/// END TO END through `attack`: a violation that exists only within 1e-3 of the
/// upper corner. With zero PGD steps the attack reports its seeds as they are,
/// so the armed attacker must return the upper corner itself, and the dark
/// attacker's two uniform seeds (seeds 42/43) must miss it.
#[test]
fn corner_seeds_find_a_vertex_only_violation_the_uniform_seeds_miss() {
    let network = sum_network();
    let bounds = unit_box_2d();
    let config = PgdConfig {
        num_restarts: 2,
        num_steps: 0,
        seed: 42,
        parallel: false,
        ..Default::default()
    };
    let dark = PgdAttacker::new(config.clone())
        .attack(&network, &bounds, 0, 1.999, true)
        .unwrap();
    assert!(
        !dark.found_counterexample,
        "two uniform seeds cannot land within 1e-3 of the corner"
    );
    let armed = PgdAttacker::new(PgdConfig {
        corner_seed_restarts: 2,
        ..config
    })
    .attack(&network, &bounds, 0, 1.999, true)
    .unwrap();
    assert!(
        armed.found_counterexample,
        "the upper-corner seed is the witness"
    );
    let witness: Vec<f32> = armed.counterexample.unwrap().iter().copied().collect();
    assert_eq!(witness, vec![1.0, 1.0]);
}
