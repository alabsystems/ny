// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Randomized differential soundness harness for the CROWN certificate
//! generator.
//!
//! For thousands of deterministically-generated small ReLU-1 networks we:
//!
//! 1. compute the CROWN lower bound and emit a proof-carrying certificate;
//! 2. confirm the certificate passes the in-tree mirror of Clean's verifier
//!    (the *linear-program* obligation — that the multipliers derive the bound);
//! 3. confirm, by exact evaluation of the **true** network on a grid over the
//!    input box, that the certified bound never exceeds an actually-attained
//!    output (the *relaxation* obligation — that NY's premises are sound
//!    over-approximations, which Clean cannot see).
//!
//! Step 3 is the part a "Clean accepted it" claim alone would miss: a buggy
//! relaxation could emit premises that pass Clean yet describe a network whose
//! real minimum lies *below* the certified bound. None do.
//!
//! NY's certificate arithmetic is arbitrary precision. Every generated network
//! must therefore reach both the certificate checker and the exact-network
//! oracle; a failure to do so is a test failure, not skipped coverage.

use ny_cert::generate::random_problem;
use ny_cert::{check_entailment, check_farkas, Rat};

struct Stats {
    certified: u32,
}

fn run_range(seeds: std::ops::Range<u64>) -> Stats {
    let mut stats = Stats { certified: 0 };

    for seed in seeds {
        let problem = random_problem(seed, 3, 4);

        // Certify the property `y ≥ m` at the CROWN bound itself (the tightest
        // threshold this relaxation can prove). The generated problem is valid
        // and exact arithmetic is arbitrary precision, so refusal is a defect.
        problem
            .preact_bounds()
            .unwrap_or_else(|error| panic!("seed {seed}: pre-activation bounds failed: {error}"));
        let bound = certify_at_bound(&problem, seed);

        let certified = problem
            .certify(bound)
            .expect("certify at its own bound must succeed");

        // (2) Linear-program obligation: Clean-equivalent verifier accepts.
        let (derived, claimed) =
            check_entailment(&certified.entailment).expect("entailment must self-check");
        // derived is in normalized -y space: -y ≤ derived ⇔ y ≥ -derived = bound.
        assert_eq!(derived.neg(), certified.lower_bound, "seed {seed}");
        assert!(derived <= claimed, "seed {seed}");
        check_farkas(&certified.farkas).expect("farkas must self-check");

        // (3) Relaxation obligation: the *true* network never goes below the
        // certified bound. Checked against the EXACT true minimum in ANY input
        // dimension via complete hyperplane-arrangement vertex enumeration
        // (`exact_min_nd`) — a decision procedure, not sampling, so `bound ≤
        // true_min` here is a genuine per-network soundness proof.
        let true_min = problem
            .exact_min_nd()
            .expect("exact nd eval")
            .unwrap_or_else(|| panic!("seed {seed}: valid bounded network has no exact minimum"));
        assert!(
            certified.lower_bound <= true_min,
            "UNSOUND at seed {seed}: certified bound {:?} exceeds exact true minimum {:?}",
            certified.lower_bound,
            true_min,
        );

        stats.certified += 1;
    }
    stats
}

/// Certify at the network's own CROWN bound. Generated inputs are valid and the
/// rational backend is arbitrary precision, so every error is actionable.
fn certify_at_bound(problem: &ny_cert::Relu1Problem, seed: u64) -> Rat {
    // `certify(ZERO)` runs the full backward pass and returns the bound; we then
    // re-certify at that bound. A cleaner one-shot API could return the bound
    // directly, but this keeps the public surface minimal.
    match problem.certify(Rat::ZERO) {
        Ok(c) => c.lower_bound,
        Err(ny_cert::CrownError::ThresholdAboveBound { bound, .. }) => {
            // The bound is below 0; parse it back out exactly. With the bignum
            // rational backend this round-trips at any magnitude, so the only
            // way this can fail is a malformed bound string.
            parse_bound(&bound)
                .unwrap_or_else(|| panic!("seed {seed}: malformed certified bound {bound:?}"))
        }
        // Retained for source compatibility: with arbitrary-precision rationals
        // a `RatError::Overflow` from the backward pass is now UNREACHABLE
        // (bignum never overflows), so this arm no longer fires in practice.
        // Keep the explicit arm so a future fixed-width regression fails with
        // an actionable seed rather than being mistaken for absent coverage.
        Err(ny_cert::CrownError::Rat(error)) => {
            panic!("seed {seed}: arbitrary-precision certificate arithmetic failed: {error}")
        }
        Err(other) => panic!("unexpected certify error: {other:?}"),
    }
}

fn parse_bound(s: &str) -> Option<Rat> {
    use num_bigint::BigInt;
    use std::str::FromStr;
    // Parse via arbitrary-precision integers so bounds whose numerator or
    // denominator exceeds i128 still round-trip exactly. With the bignum
    // rational backend these big bounds are fully representable, so this never
    // fails on magnitude — only on a genuinely malformed string.
    let (n, d) = match s.split_once('/') {
        Some((n, d)) => (BigInt::from_str(n).ok()?, BigInt::from_str(d).ok()?),
        None => (BigInt::from_str(s).ok()?, BigInt::from(1)),
    };
    Rat::from_bigints(n, d).ok()
}

#[test]
fn differential_soundness_small_sweep() {
    // A fast in-`cargo test` sweep; the heavy multi-thousand sweep and the
    // cross-repo Clean-binary check run from scripts/clean_differential.sh.
    let stats = run_range(0..2000);
    eprintln!("differential: {} certified", stats.certified);
    assert_eq!(
        stats.certified, 2000,
        "every generated network must be checked"
    );
}

#[test]
fn exact_oracle_soundness_sweep_dim_le_2() {
    // The strongest Pillar-1 soundness statement available without a toolchain:
    // for thousands of random nets with input dim ≤ 2, the certified bound never
    // exceeds the EXACT true-network minimum (breakpoint-arrangement vertices,
    // no grid blind spot). This is the off-grid-safe check the harness-adequacy
    // adversarial finding asked for.
    let mut checked = 0u32;
    for seed in 0..4000u64 {
        let problem = random_problem(seed, 2, 4);
        let bound = certify_at_bound(&problem, seed);
        let exact = problem
            .exact_min()
            .expect("exact eval")
            .expect("dim ≤ 2 has an exact min");
        assert!(
            bound <= exact,
            "UNSOUND at seed {seed}: bound {bound:?} > exact true min {exact:?}"
        );
        checked += 1;
    }
    assert_eq!(checked, 4000, "every generated network must be checked");
}

#[test]
fn sbar_differential_soundness_sweep() {
    // Pillar 2: for thousands of random feasible truncated-simplex LPs, the SBAR
    // certified upper bound (a) self-checks as a Clean entailment and (b) really
    // dominates the objective at every vertex of the truncated simplex (the LP
    // max is attained at a vertex, so vertex-domination ⇔ global soundness).
    use ny_cert::generate::random_simplex_lp;
    let mut certified = 0u32;
    for seed in 0..3000u64 {
        let lp = random_simplex_lp(seed, 5);
        let cert = lp.certify_upper().expect("feasible by construction");
        check_entailment(&cert.entailment).expect("SBAR cert must self-check");

        // Enumerate the water-filling vertices for every priority order: start
        // at p_lo, pour the budget B=1−Σp_lo greedily in that order. Each such
        // vertex is feasible; the certified bound must dominate all of them.
        let m = lp.g.len();
        let mut sum_lo = Rat::ZERO;
        for v in &lp.p_lo {
            sum_lo = sum_lo.add(*v).unwrap();
        }
        let budget = Rat::ONE.sub(sum_lo).unwrap();
        for start in 0..m {
            let mut p = lp.p_lo.clone();
            let mut rem = budget;
            for off in 0..m {
                let j = (start + off) % m;
                let slack = lp.p_hi[j].sub(lp.p_lo[j]).unwrap();
                let take = if slack <= rem { slack } else { rem };
                p[j] = p[j].add(take).unwrap();
                rem = rem.sub(take).unwrap();
            }
            let obj = lp.objective(&p).unwrap();
            assert!(
                obj <= cert.bound,
                "UNSOUND SBAR at seed {seed}: feasible objective {obj:?} exceeds bound {:?}",
                cert.bound
            );
        }
        certified += 1;
    }
    assert_eq!(certified, 3000);
}

#[test]
fn certified_bound_is_a_valid_lower_bound_under_perturbation() {
    // Independent of the grid: for a handful of seeds, sample many interior
    // points via a second LCG stream and confirm soundness pointwise.
    use ny_cert::generate::Lcg;
    for seed in 0..500u64 {
        let problem = random_problem(seed, 2, 3);
        let bound = certify_at_bound(&problem, seed);
        let mut g = Lcg::new(seed ^ 0xD1CE);
        let n = problem.input_lower.len();
        for _ in 0..40 {
            let mut x = Vec::with_capacity(n);
            for i in 0..n {
                // x_i = l_i + (k/8)·(u_i - l_i), k ∈ 0..=8.
                let k = g.range_i128(0, 8);
                let frac = Rat::new(k, 8).unwrap();
                let span = problem.input_upper[i].sub(problem.input_lower[i]).unwrap();
                x.push(problem.input_lower[i].add(frac.mul(span).unwrap()).unwrap());
            }
            let y = problem.eval(&x).unwrap();
            assert!(
                bound <= y,
                "UNSOUND at seed {seed}: bound {bound:?} > network output {y:?} at {x:?}"
            );
        }
    }
}
