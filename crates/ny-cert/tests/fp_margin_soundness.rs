// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Randomized soundness harness for the deployed-FP error margin
//! (`ny_cert::fp_margin`).
//!
//! For deterministically-generated small deep ReLU networks (seeded LCG, the
//! crate's `generate::Lcg` convention — no `rand`, no wall clock) we sample
//! inputs in the box and check the module's ONE claim directly:
//!
//! ```text
//!   |fl32(net(x)) − net(x)| ≤ delta
//! ```
//!
//! The left side is computed as (a) an actual f32 forward pass (sequential
//! dot + bias + ReLU, round-to-nearest — the deployed semantics) minus (b)
//! the exact rational forward pass (`DeepReluProblem::eval`, the ideal
//! semantics), the difference taken exactly in `Rat` via the lossless
//! `Rat::from_f32_exact` embedding. All network constants are dyadic
//! (`i/16`), so they are exact in BOTH semantics — any violation would be a
//! genuine unsoundness of the error analysis, not test noise.

use ny_cert::crown_deep::DeepReluProblem;
use ny_cert::fp_margin::deployed_fp_margin;
use ny_cert::generate::Lcg;
use ny_cert::Rat;

/// A random small network in parallel f32 / exact-Rat form.
struct TestNet {
    problem: DeepReluProblem,
    weights_f32: Vec<Vec<Vec<f32>>>,
    biases_f32: Vec<Vec<f32>>,
    out_weight_f32: Vec<f32>,
    out_bias_f32: f32,
}

/// Draw a dyadic value `i/16` with `i ∈ [-32, 32]` — exactly representable
/// in f32 and as a `Rat`, so both semantics see the same constant.
fn dyadic(g: &mut Lcg) -> (f32, Rat) {
    let i = g.range_i128(-32, 32);
    #[allow(clippy::cast_precision_loss)] // |i| ≤ 32: exact in f32
    let f = (i as f32) / 16.0;
    (f, Rat::new(i, 16).expect("nonzero denominator"))
}

/// Random deep ReLU net: 1–2 hidden layers (so 2–3 affine layers with the
/// read-out), width ≤ 8, over the box [−1, 1]ⁿ with n ≤ 4.
fn random_deep_net(seed: u64) -> TestNet {
    let mut g = Lcg::new(seed);
    let n_in = usize::try_from(g.range_i128(2, 4)).expect("small");
    let n_hidden_layers = usize::try_from(g.range_i128(1, 2)).expect("small");

    let mut dims = vec![n_in];
    for _ in 0..n_hidden_layers {
        dims.push(usize::try_from(g.range_i128(1, 8)).expect("small"));
    }

    let mut weights_f32 = Vec::new();
    let mut biases_f32 = Vec::new();
    let mut weights = Vec::new();
    let mut biases = Vec::new();
    for li in 0..n_hidden_layers {
        let (rows, cols) = (dims[li + 1], dims[li]);
        let mut wf = Vec::new();
        let mut wr = Vec::new();
        let mut bf = Vec::new();
        let mut br = Vec::new();
        for _ in 0..rows {
            let mut row_f = Vec::new();
            let mut row_r = Vec::new();
            for _ in 0..cols {
                let (f, r) = dyadic(&mut g);
                row_f.push(f);
                row_r.push(r);
            }
            wf.push(row_f);
            wr.push(row_r);
            let (f, r) = dyadic(&mut g);
            bf.push(f);
            br.push(r);
        }
        weights_f32.push(wf);
        weights.push(wr);
        biases_f32.push(bf);
        biases.push(br);
    }
    let last = dims[n_hidden_layers];
    let mut out_weight_f32 = Vec::new();
    let mut out_weight = Vec::new();
    for _ in 0..last {
        let (f, r) = dyadic(&mut g);
        out_weight_f32.push(f);
        out_weight.push(r);
    }
    let (out_bias_f32, out_bias) = dyadic(&mut g);

    let minus_one = Rat::new(-1, 1).expect("exact");
    let problem = DeepReluProblem {
        weights,
        biases,
        out_weight,
        out_bias,
        input_lower: vec![minus_one; n_in],
        input_upper: vec![Rat::ONE; n_in],
        alpha: None,
        interm_round: false,
    };
    TestNet {
        problem,
        weights_f32,
        biases_f32,
        out_weight_f32,
        out_bias_f32,
    }
}

/// The DEPLOYED semantics: sequential f32 dot products (one rounding per
/// multiply and per add), bias added last, `max(z, 0)` ReLU.
fn f32_forward(net: &TestNet, x: &[f32]) -> f32 {
    let mut act: Vec<f32> = x.to_vec();
    for (w, b) in net.weights_f32.iter().zip(&net.biases_f32) {
        let mut next = Vec::new();
        for (row, bias) in w.iter().zip(b) {
            let mut acc = 0.0f32;
            for (wj, aj) in row.iter().zip(&act) {
                acc += wj * aj;
            }
            acc += bias;
            next.push(acc.max(0.0));
        }
        act = next;
    }
    let mut y = 0.0f32;
    for (wj, aj) in net.out_weight_f32.iter().zip(&act) {
        y += wj * aj;
    }
    y + net.out_bias_f32
}

#[test]
fn test_fp_margin_random_networks_f32_forward_pass_within_delta() {
    const NETWORKS: u64 = 50;
    const SAMPLES_PER_NET: u32 = 20;

    let mut checked = 0u32;
    for seed in 0..NETWORKS {
        let net = random_deep_net(seed);
        let margin = deployed_fp_margin(&net.problem).expect("margin computes");
        let delta = margin.output;
        assert!(
            !delta.is_negative(),
            "seed {seed}: delta must be non-negative"
        );

        let mut sampler = Lcg::new(seed.wrapping_mul(0x5DEE_CE66).wrapping_add(11));
        let n_in = net.problem.input_lower.len();
        for sample in 0..SAMPLES_PER_NET {
            // Dyadic inputs i/16 ∈ [−1, 1]: exact in f32 and Rat, in the box.
            let mut x_f32 = Vec::new();
            let mut x_rat = Vec::new();
            for _ in 0..n_in {
                let i = sampler.range_i128(-16, 16);
                #[allow(clippy::cast_precision_loss)] // |i| ≤ 16: exact
                x_f32.push((i as f32) / 16.0);
                x_rat.push(Rat::new(i, 16).expect("nonzero denominator"));
            }

            let deployed = f32_forward(&net, &x_f32);
            assert!(
                deployed.is_finite(),
                "seed {seed} sample {sample}: deployed output must be finite"
            );
            let deployed_rat = Rat::from_f32_exact(deployed).expect("finite f32 embeds exactly");
            let ideal = net.problem.eval(&x_rat).expect("exact forward pass");
            let diff = deployed_rat.sub(ideal).expect("exact difference").abs();
            assert!(
                diff <= delta,
                "seed {seed} sample {sample}: |fl32(net(x)) - net(x)| = {}/{} \
                 exceeds delta = {}/{}",
                diff.num(),
                diff.den(),
                delta.num(),
                delta.den()
            );
            checked += 1;
        }
    }
    assert_eq!(
        checked,
        u32::try_from(NETWORKS).expect("small") * SAMPLES_PER_NET,
        "harness must exercise every planned sample"
    );
}
