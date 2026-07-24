// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//
// Probe the (HISTORICAL) i64-EMISSION frontier of exact-rational CROWN
// certificates as a function of (a) weight fractional precision (bits after
// the point) and (b) network depth, on real-shaped dyadic-weight ReLU networks
// of width W.
//
// NOTE (2026-07): the i64 wall this bin was written to measure is GONE —
// `rational.rs` is BigRational end to end and `to_clean_string` always emits
// full bignum `n/d` strings, so every cell now reports OK. The bin is kept as
// a regression probe: any reappearance of `i64_EMIT_OVERFLOW`/`i128_OVERFLOW`
// in its table would signal a bignum regression.
//
// For each (precision p, depth k) we build a width-W k-hidden-layer net whose
// weights are p-bit dyadic rationals (denominator 2^p, like a real network
// quantized to 2^-p), certify exactly, and report whether the arithmetic and
// the emission both succeeded (they now always do).

use ny_cert::crown_deep::DeepReluProblem;
use ny_cert::rational::Rat;
use ny_cert::schema::{
    entailment_to_json, farkas_to_json, ConstraintKind, FarkasCertificate, LinearConstraint,
};

fn det_weight(seed: &mut u64, p: u32) -> Rat {
    // xorshift PRNG -> dyadic rational in [-1,1] with denominator 2^p.
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    let den = 1i128 << p;
    let num = (*seed % (2 * den as u64 + 1)) as i128 - den;
    Rat::new(num, den).unwrap()
}

fn build(p: u32, k: usize, w: usize) -> DeepReluProblem {
    let mut seed = 0x1234_5678_9abc_def0u64 ^ ((p as u64) << 8) ^ ((k as u64) << 1);
    let in_dim = 5usize;
    let mut weights = Vec::new();
    let mut biases = Vec::new();
    let mut prev = in_dim;
    for _ in 0..k {
        let mut layer = Vec::with_capacity(w);
        for _ in 0..w {
            layer.push(
                (0..prev)
                    .map(|_| det_weight(&mut seed, p))
                    .collect::<Vec<_>>(),
            );
        }
        biases.push((0..w).map(|_| det_weight(&mut seed, p)).collect());
        weights.push(layer);
        prev = w;
    }
    let out_weight = (0..prev).map(|_| det_weight(&mut seed, p)).collect();
    DeepReluProblem {
        weights,
        biases,
        out_weight,
        out_bias: Rat::ZERO,
        input_lower: vec![Rat::new(-1, 4).unwrap(); in_dim],
        input_upper: vec![Rat::new(1, 4).unwrap(); in_dim],
        alpha: None,
        interm_round: false,
    }
}

fn try_one(p: u32, k: usize, w: usize) -> &'static str {
    let net = build(p, k, w);
    let certified = match net.certify(Rat::from_int(-1_000_000_000)) {
        Ok(c) => c,
        Err(_) => return "i128_OVERFLOW",
    };
    // Build a real Farkas like the ONNX binary does and try to emit it.
    let m = certified.lower_bound;
    let mut f_constraints = certified.entailment.premises.clone();
    let mut f_mult = certified.entailment.multipliers.clone();
    f_constraints.push(LinearConstraint::with_kind(
        ConstraintKind::Le,
        &[("y", Rat::ONE)],
        m.sub(Rat::ONE).unwrap(),
    ));
    f_mult.push(Rat::ONE);
    let farkas = FarkasCertificate {
        constraints: f_constraints,
        multipliers: f_mult,
    };
    if entailment_to_json(&certified.entailment).is_err() || farkas_to_json(&farkas).is_err() {
        return "i64_EMIT_OVERFLOW";
    }
    "OK_i64_EMITTED"
}

fn main() {
    let w = 8usize;
    println!("# i64-emission frontier: width={w}, input_dim=5, dyadic weights (den=2^p)");
    println!("# columns: precision p (bits), then depth k = 1..6");
    print!("p\\k ");
    for k in 1..=6 {
        print!("{:>16}", k);
    }
    println!();
    for p in [1u32, 2, 4, 6, 8, 10, 12, 16, 23] {
        print!("{p:>3} ");
        for k in 1..=6 {
            print!("{:>16}", try_one(p, k, w));
        }
        println!();
    }
    println!();
    println!("# Historical note: real ACAS-Xu (p up to 40, k=6, width=50) was far past the");
    println!("# old i64 wall; with bignum emission every cell above should now be OK.");
}
