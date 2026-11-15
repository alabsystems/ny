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

#![forbid(unsafe_code)]

// weights are p-bit dyadic rationals (denominator 2^p, like a real network
// quantized to 2^-p), certify exactly, and report whether the arithmetic and
// the emission both succeeded (they now always do).

use ny_cert::crown_deep::DeepReluProblem;
use ny_cert::rational::Rat;
use ny_cert::schema::{
    entailment_to_json, farkas_to_json, ConstraintKind, FarkasCertificate, LinearConstraint,
};

/// Hidden-layer width of every probed network.
const WIDTH: usize = 8;

/// Fractional precisions probed (bits after the point; denominator 2^p).
const PRECISIONS: [u32; 9] = [1, 2, 4, 6, 8, 10, 12, 16, 23];

/// Input dimension of every probed network.
const IN_DIM: usize = 5;

/// Largest precision `det_weight` accepts: `2 * 2^p + 1` must fit the u64
/// modulus. Every `PRECISIONS` entry is far below it.
const MAX_PRECISION_BITS: u32 = 62;

/// xorshift PRNG -> dyadic rational in [-1,1] with denominator 2^p, or `None`
/// for a precision past [`MAX_PRECISION_BITS`] (the old `1i128 << p` reached
/// `i128::MIN` at p = 127 and `num - den` then overflowed).
fn det_weight(seed: &mut u64, p: u32) -> Option<Rat> {
    if p > MAX_PRECISION_BITS {
        return None;
    }
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    // den = 2^p <= 2^62, so 2*den + 1 fits u64 and both i128 operands below
    // are in [0, 2^64): the (checked) subtraction never fails.
    let den = 1u64.checked_shl(p)?;
    let modulus = den.checked_mul(2)?.checked_add(1)?;
    let num = i128::from(seed.checked_rem(modulus)?).checked_sub(i128::from(den))?;
    Rat::new(num, i128::from(den)).ok()
}

fn build(p: u32, k: usize) -> Option<DeepReluProblem> {
    let mut seed = 0x1234_5678_9abc_def0u64 ^ ((p as u64) << 8) ^ ((k as u64) << 1);
    let mut weights = Vec::new();
    let mut biases = Vec::new();
    for li in 0..k {
        // Fan-in: the input for the first layer, the previous hidden layer
        // after that (both constants, so every allocation below is constant).
        let fan_in = if li == 0 { IN_DIM } else { WIDTH };
        let mut layer = Vec::new();
        for _ in 0..WIDTH {
            layer.push(
                (0..fan_in)
                    .map(|_| det_weight(&mut seed, p))
                    .collect::<Option<Vec<_>>>()?,
            );
        }
        biases.push(
            (0..WIDTH)
                .map(|_| det_weight(&mut seed, p))
                .collect::<Option<Vec<_>>>()?,
        );
        weights.push(layer);
    }
    let readout_fan_in = if k == 0 { IN_DIM } else { WIDTH };
    let out_weight = (0..readout_fan_in)
        .map(|_| det_weight(&mut seed, p))
        .collect::<Option<Vec<_>>>()?;
    Some(DeepReluProblem {
        weights,
        biases,
        out_weight,
        out_bias: Rat::ZERO,
        input_lower: vec![Rat::new(-1, 4).ok()?; IN_DIM],
        input_upper: vec![Rat::new(1, 4).ok()?; IN_DIM],
        alpha: None,
        interm_round: false,
    })
}

fn try_one(p: u32, k: usize) -> &'static str {
    let Some(net) = build(p, k) else {
        return "PRECISION_UNSUPPORTED";
    };
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
    let w = WIDTH;
    println!("# i64-emission frontier: width={w}, input_dim=5, dyadic weights (den=2^p)");
    println!("# columns: precision p (bits), then depth k = 1..6");
    print!("p\\k ");
    for k in 1..=6 {
        print!("{:>16}", k);
    }
    println!();
    for p in PRECISIONS {
        print!("{p:>3} ");
        for k in 1..=6 {
            print!("{:>16}", try_one(p, k));
        }
        println!();
    }
    println!();
    println!("# Historical note: real ACAS-Xu (p up to 40, k=6, width=50) was far past the");
    println!("# old i64 wall; with bignum emission every cell above should now be OK.");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn det_weight_is_a_dyadic_in_the_unit_interval() {
        let mut seed = 0x1234_5678_9abc_def0u64;
        for p in PRECISIONS {
            let w = det_weight(&mut seed, p).expect("in-range precision");
            assert!(w >= Rat::ONE.neg() && w <= Rat::ONE, "p={p}: {w:?}");
        }
    }

    #[test]
    fn det_weight_past_the_precision_cap_is_none_not_an_overflow() {
        // The old `1i128 << 127` is i128::MIN, and `num - den` then overflowed.
        let mut seed = 1u64;
        assert!(det_weight(&mut seed, 127).is_none());
        assert!(det_weight(&mut seed, MAX_PRECISION_BITS + 1).is_none());
        assert!(det_weight(&mut seed, MAX_PRECISION_BITS).is_some());
    }

    #[test]
    fn built_nets_have_the_probed_shape() {
        let net = build(4, 3).expect("in-range precision");
        assert_eq!(net.weights.len(), 3);
        assert_eq!(net.weights[0].len(), WIDTH);
        assert!(net.weights[0].iter().all(|row| row.len() == IN_DIM));
        assert!(net.weights[1].iter().all(|row| row.len() == WIDTH));
        assert_eq!(net.out_weight.len(), WIDTH);
        assert_eq!(build(4, 0).expect("depth 0").out_weight.len(), IN_DIM);
        assert!(build(127, 1).is_none());
    }
}
