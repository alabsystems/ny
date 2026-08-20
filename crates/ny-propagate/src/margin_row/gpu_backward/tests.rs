// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GPU-free unit tests for the certified GPU backward skeleton:
//! gate spec, fail-closed admission, the ds32 host twin's EFT identities and
//! enclosure (moat phase M0), the M1 bit-comparison harness core, and source
//! pins on the WGSL kernels (the barrier idiom must not silently regress into
//! the compiler-destroyed plain forms).

use super::ds::{
    bit_compare_streams, ds_add, ds_add_f32, ds_dot, ds_mul_f32, fast_two_sum, gamma_ds, Ds, U_DS,
};
use super::{
    admission_shape, armed_from_raw, Refusal, CONV_BACKWARD_WGSL, DS_PRIMITIVES_WGSL,
    GATE_TRANSFORM_WGSL,
};
use ny_core::eft::{two_prod_f32, two_sum_f32};

// ---------------------------------------------------------------------------
// Gate + admission (fail-closed spec)
// ---------------------------------------------------------------------------

#[test]
fn only_exact_one_arms_the_gate() {
    assert!(armed_from_raw(Some("1")));
    for raw in [
        None,
        Some(""),
        Some("0"),
        Some("true"),
        Some("on"),
        Some("yes"),
        Some(" 1"),
        Some("1 "),
        Some("2"),
    ] {
        assert!(!armed_from_raw(raw), "must stay dark for {raw:?}");
    }
}

#[test]
fn refusal_tags_are_distinct_bits() {
    // The once-per-reason telemetry rate-limits by `1 << tag()`; two variants
    // sharing a tag would silently swallow one reason — the exact failure
    // ("refused=2, no reason") that cost a measurement cycle on gpu_seam.
    let all = [
        Refusal::Disabled,
        Refusal::Unimplemented,
        Refusal::ChannelDead,
        Refusal::Busy,
        Refusal::NotOutward,
        Refusal::Rows,
        Refusal::Unmappable("x"),
        Refusal::Deadline,
        Refusal::Device,
        Refusal::Payload,
        Refusal::ErrorFloor,
        Refusal::Probe,
        Refusal::NonFinite,
    ];
    let mut seen = 0u32;
    for r in all {
        let bit = 1u32 << r.tag();
        assert_eq!(seen & bit, 0, "duplicate tag for {r:?}");
        seen |= bit;
    }
}

#[test]
fn admission_shape_is_fail_closed() {
    assert_eq!(admission_shape(true, 4, 10, 10), Ok(()));
    assert_eq!(admission_shape(false, 4, 10, 10), Err(Refusal::NotOutward));
    assert_eq!(admission_shape(true, 0, 10, 10), Err(Refusal::Rows));
    assert_eq!(admission_shape(true, 4, 9, 10), Err(Refusal::Rows));
    assert_eq!(admission_shape(true, 4, 0, 0), Err(Refusal::Rows));
}

// ---------------------------------------------------------------------------
// EFT identities (executable copies of the ds.rs doc examples — that module
// is pub(crate), so rustdoc does not RUN its examples; these do).
// ---------------------------------------------------------------------------

/// `a + b == s + t` as reals, so one f64 rounding of each side must agree.
#[test]
fn two_sum_identity_holds_in_f64() {
    let cases: &[(f32, f32)] = &[
        (1.0, 1e-8),
        (1e30, -1e30),
        (1e30, 1.0),
        (3.0, 1.0 / 3.0),
        (f32::MIN_POSITIVE, -f32::from_bits(1)),
        (-7.25, 7.25 + f32::from_bits(0x3380_0000)),
    ];
    for &(a, b) in cases {
        let (s, t) = two_sum_f32(a, b);
        assert_eq!(
            f64::from(a) + f64::from(b),
            f64::from(s) + f64::from(t),
            "two_sum identity failed: a={a:e} b={b:e}"
        );
    }
}

/// `a * b` is exact in f64 (48 < 53 bits) and equals `p + e` exactly.
#[test]
fn two_prod_identity_holds_in_f64_away_from_underflow() {
    let cases: &[(f32, f32)] = &[
        (3.0, 1.0 / 3.0),
        (1e10, 1e-10),
        (-7.0, 0.142_857_15),
        (
            1.0 + f32::from_bits(0x3980_0000),
            1.0 - f32::from_bits(0x3980_0000),
        ),
    ];
    for &(a, b) in cases {
        let (p, e) = two_prod_f32(a, b);
        assert_eq!(
            f64::from(a) * f64::from(b),
            f64::from(p) + f64::from(e),
            "two_prod identity failed: a={a:e} b={b:e}"
        );
    }
}

#[test]
fn fast_two_sum_is_exact_under_its_precondition() {
    let cases: &[(f32, f32)] = &[
        (1.0, f32::from_bits(0x3380_0000)), // 1 + 2^-24
        (1e10, -3.5),
        (-2.0, 1.0),
        (0.0, 42.0), // a == 0 is admitted
    ];
    for &(a, b) in cases {
        let (s, t) = fast_two_sum(a, b);
        assert_eq!(
            f64::from(a) + f64::from(b),
            f64::from(s) + f64::from(t),
            "fast_two_sum identity failed: a={a:e} b={b:e}"
        );
    }
}

#[test]
fn ds_to_f64_is_exact_on_the_invariant() {
    let x = Ds {
        hi: 1.0,
        lo: f32::from_bits(0x3380_0000), // 2^-24 = ulp(1)/2
    };
    assert_eq!(x.to_f64(), 1.0 + 2f64.powi(-24));
    assert_eq!(Ds::from_f32(-3.5).to_f64(), -3.5);
}

// ---------------------------------------------------------------------------
// ds algebra enclosure (moat phase M0)
// ---------------------------------------------------------------------------

/// Deterministic xorshift so failures reproduce (the eft.rs test idiom).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// Magnitude ~U[0.5, 2) with a pseudo-random sign — the CROWN regime
    /// (large |terms| mass, cancelling running sums).
    #[allow(clippy::cast_precision_loss)]
    fn coeff(&mut self) -> f32 {
        let mag = 0.5 + (self.next() % 1_000_000) as f32 / 666_667.0;
        if self.next() & 1 == 0 {
            mag
        } else {
            -mag
        }
    }
}

/// |ds_dot − exact| ≤ gamma_ds(n)·Σ|w·a| (plus the f64 reference's own tiny
/// gamma) — the design 4.2 envelope the implementation session will charge.
#[test]
fn ds_dot_is_enclosed_by_the_u_ds_envelope() {
    let mut rng = Lcg(0x2545_F491_4F6C_DD1D);
    for &n in &[1usize, 7, 128, 1024, 4608] {
        let w: Vec<f32> = (0..n).map(|_| rng.coeff()).collect();
        let a: Vec<f32> = (0..n).map(|_| rng.coeff()).collect();
        let got = ds_dot(&w, &a).expect("finite").to_f64();
        let mut exact = 0.0f64;
        let mut mass = 0.0f64;
        for (&wi, &ai) in w.iter().zip(&a) {
            let t = f64::from(wi) * f64::from(ai); // exact in f64
            exact += t;
            mass += t.abs();
        }
        // The f64 reference itself rounds: charge its own Higham gamma on the
        // same mass so the assert tests the ds envelope, not the reference.
        #[allow(clippy::cast_precision_loss)]
        let g64 = 2.0 * (n as f64) * f64::EPSILON;
        let envelope = (gamma_ds(n) + g64) * mass;
        let diff = (got - exact).abs();
        assert!(
            diff <= envelope,
            "n={n}: |ds - exact| = {diff:e} > envelope {envelope:e}"
        );
    }
}

/// The whole point of ds: the value path must be far more accurate than the
/// plain f32 fold on the cancellation-heavy CROWN regime.
#[test]
fn ds_dot_beats_the_plain_f32_fold_by_orders() {
    let mut rng = Lcg(0x9E37_79B9_7F4A_7C15);
    let n = 4096usize;
    let w: Vec<f32> = (0..n).map(|_| rng.coeff()).collect();
    let a: Vec<f32> = (0..n).map(|_| rng.coeff()).collect();

    let mut plain = 0.0f32;
    let mut exact = 0.0f64;
    for (&wi, &ai) in w.iter().zip(&a) {
        plain += wi * ai;
        exact += f64::from(wi) * f64::from(ai);
    }
    let ds_err = (ds_dot(&w, &a).expect("finite").to_f64() - exact).abs();
    let plain_err = (f64::from(plain) - exact).abs();
    assert!(
        ds_err * 1e4 < plain_err,
        "expected >= 1e4x accuracy gain, got ds={ds_err:e} plain={plain_err:e}"
    );
}

#[test]
fn ds_elementwise_ops_stay_within_u_ds() {
    // ds ops are NOT error-free compositions; U_DS is the per-op relative
    // bound the design charges. Verify on adversarial operand mixes.
    // -E rather than a truncated literal: the point of the mix is an
    // irrational operand, and the exact constant is a stronger one than a
    // 10-digit approximation of it.
    let vals: &[f64] = &[1.0, -1.0, 3.5e7, 1.0 / 3.0, -std::f64::consts::E, 1e-12];
    for &x in vals {
        for &y in vals {
            #[allow(clippy::cast_possible_truncation)]
            let dx = {
                let hi = x as f32;
                let (h, l) = two_sum_f32(hi, (x - f64::from(hi)) as f32);
                Ds { hi: h, lo: l }
            };
            #[allow(clippy::cast_possible_truncation)]
            let wy = y as f32;

            let sum = ds_add_f32(dx, wy).to_f64();
            let sum_exact = dx.to_f64() + f64::from(wy);
            assert!(
                (sum - sum_exact).abs() <= U_DS * sum_exact.abs().max(1e-300),
                "ds_add_f32 out of envelope: x={x} y={y}"
            );

            let prod = ds_mul_f32(dx, wy).to_f64();
            let prod_exact = dx.to_f64() * f64::from(wy);
            assert!(
                (prod - prod_exact).abs() <= U_DS * prod_exact.abs().max(1e-300),
                "ds_mul_f32 out of envelope: x={x} y={y}"
            );

            #[allow(clippy::cast_possible_truncation)]
            let dy = {
                let hi = y as f32;
                let (h, l) = two_sum_f32(hi, (y - f64::from(hi)) as f32);
                Ds { hi: h, lo: l }
            };
            let both = ds_add(dx, dy).to_f64();
            let both_exact = dx.to_f64() + dy.to_f64();
            assert!(
                (both - both_exact).abs() <= U_DS * both_exact.abs().max(1e-300),
                "ds_add out of envelope: x={x} y={y}"
            );
        }
    }
}

#[test]
fn gamma_ds_dominates_and_saturates() {
    for &n in &[1usize, 100, 4608, 100_000] {
        #[allow(clippy::cast_precision_loss)]
        let raw = 2.0 * (n as f64) * U_DS;
        assert!(
            gamma_ds(n) >= raw,
            "gamma_ds must dominate 2n*U_DS at n={n}"
        );
        // SELF-SUFFICIENCY (review minor note): in this range `nu` and
        // `1 - nu` are exact (power-of-two unit, small n), so fl(nu/(1-nu))
        // is ONE rounding from the true ratio — any value >= 2 ulps above
        // the fl result strictly dominates the true value with NO caller
        // inflation. The executable copy of the ds.rs doc example.
        #[allow(clippy::cast_precision_loss)]
        let nu = (2 * n) as f64 * U_DS;
        let fl = nu / (1.0 - nu);
        assert!(
            gamma_ds(n) >= f64::from_bits(fl.to_bits() + 2),
            "gamma_ds must dominate its own roundings at n={n}"
        );
    }
    // Degenerate widths saturate to 1.0 (degrade-to-useless, never undercharge).
    assert_eq!(gamma_ds(usize::MAX / 4), 1.0);
    // And at conv scale the envelope is far below margin scale (design R1):
    // the whole reason ds replaces the f32 Higham charge.
    assert!(gamma_ds(4608) < 1e-9);
}

// ---------------------------------------------------------------------------
// M1 bit-comparison harness core
// ---------------------------------------------------------------------------

#[test]
fn bit_compare_scaffold_detects_single_bit_divergence() {
    let host: Vec<Ds> = (0..64)
        .map(|i| {
            #[allow(clippy::cast_precision_loss)]
            let hi = (i as f32).mul_add(0.5, -7.0);
            Ds {
                hi,
                lo: f32::from_bits(i as u32), // exercises subnormal lo lanes
            }
        })
        .collect();
    let mut device: Vec<(f32, f32)> = host.iter().map(|d| (d.hi, d.lo)).collect();
    assert_eq!(bit_compare_streams(&host, &device), Ok(()));

    // One flipped LSB in one lo lane must be found at its exact index.
    device[41].1 = f32::from_bits(device[41].1.to_bits() ^ 1);
    assert_eq!(bit_compare_streams(&host, &device), Err(41));
    device[41].1 = host[41].lo;

    // Sign-of-zero divergence must be caught: `to_bits`, not `==`.
    let zeros = [Ds { hi: 0.0, lo: 0.0 }];
    assert_eq!(bit_compare_streams(&zeros, &[(0.0, 0.0)]), Ok(()));
    assert_eq!(bit_compare_streams(&zeros, &[(-0.0, 0.0)]), Err(0));

    // Length mismatch diverges at the shorter length.
    device.pop();
    assert_eq!(bit_compare_streams(&host, &device), Err(63));
}

// ---------------------------------------------------------------------------
// Verdict-authority capability chain (adversarial-review items 2/3)
// ---------------------------------------------------------------------------

/// Deterministic full-coverage host stream for the qualification tests.
fn parity_host_stream(n: usize) -> Vec<Ds> {
    (0..n)
        .map(|i| {
            #[allow(clippy::cast_precision_loss)]
            let hi = (i as f32).mul_add(0.125, -13.0);
            Ds {
                hi,
                lo: f32::from_bits((i as u32).rotate_left(7) & 0x0000_FFFF),
            }
        })
        .collect()
}

#[test]
fn parity_proof_is_the_only_gate_into_authority() {
    use super::authority::{DeviceParityProof, VerdictAuthority, PARITY_MIN_LANES};

    let host = parity_host_stream(PARITY_MIN_LANES);
    let device: Vec<(f32, f32)> = host.iter().map(|d| (d.hi, d.lo)).collect();

    // Denorm-preserve resolved OFF refuses regardless of parity (item 3:
    // Auto => passthrough_supported, i.e. silently OFF on an unsupported
    // adapter — shader_loading.rs:100 — and FTZ voids the residual identity).
    assert!(matches!(
        DeviceParityProof::qualify(false, &host, &device),
        Err(Refusal::Unmappable(_))
    ));

    // An under-covered probe refuses: a token self-check cannot mint
    // authority (item 2).
    assert!(matches!(
        DeviceParityProof::qualify(true, &host[..PARITY_MIN_LANES - 1], &device),
        Err(Refusal::Unmappable(_))
    ));

    // A full bit-identical readback qualifies, and the proof is the ONLY
    // input `VerdictAuthority::grant` accepts — the compile-time shape of
    // the M1-is-blocking contract (run_transaction requires the authority).
    let proof = DeviceParityProof::qualify(true, &host, &device).expect("bit-identical stream");
    assert_eq!(proof.lanes(), PARITY_MIN_LANES);
    let auth = VerdictAuthority::grant(proof);
    assert_eq!(auth.parity_lanes(), PARITY_MIN_LANES);
}

#[test]
fn parity_mismatch_latches_the_channel_dead() {
    use super::authority::{DeviceParityProof, PARITY_MIN_LANES};

    // NOTE: this test intentionally trips the PROCESS-WIDE channel-dead
    // latch (OnceLock; deliberately no reset — design R3). Tests share one
    // process, so it must remain the only test that relies on post-latch
    // state, and no test in this suite may assert `channel_dead()` is None.
    let host = parity_host_stream(PARITY_MIN_LANES);
    let mut device: Vec<(f32, f32)> = host.iter().map(|d| (d.hi, d.lo)).collect();
    device[500].1 = f32::from_bits(device[500].1.to_bits() ^ 1);

    assert!(matches!(
        DeviceParityProof::qualify(true, &host, &device),
        Err(Refusal::ChannelDead)
    ));
    assert!(
        super::channel_dead().is_some(),
        "an M1 bit mismatch must latch the lane channel dead (item 2)"
    );
}

// ---------------------------------------------------------------------------
// Pre-registered sweep kill line (adversarial-review item 1)
// ---------------------------------------------------------------------------

#[test]
fn sweep_kill_line_matches_the_sweep_source() {
    // The M2/M3 harness greps run logs for this exact substring (any hit =
    // KILL: the measured post-submit poisoning recurring under this lane's
    // added contention). Pin the constant against the actual sweep source so
    // a reworded log message cannot silently blind the criterion.
    assert_eq!(
        super::SWEEP_POST_SUBMIT_KILL_LINE,
        "exited after submission without a final drain"
    );
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../ny-gpu/src/wgpu_device/ops/intermediate_sweep.rs"
    );
    let src = std::fs::read_to_string(path).expect("intermediate_sweep.rs readable in-tree");
    assert!(
        src.contains(super::SWEEP_POST_SUBMIT_KILL_LINE),
        "the sweep's poison log line drifted from SWEEP_POST_SUBMIT_KILL_LINE; \
         update the constant AND the design's section-7 kill clauses together"
    );
}

// ---------------------------------------------------------------------------
// WGSL source pins (the barrier idiom must not regress)
// ---------------------------------------------------------------------------

#[test]
fn ds_primitives_wgsl_uses_the_barrier_forms() {
    let src = DS_PRIMITIVES_WGSL;
    for f in [
        "fn eft_two_prod",
        "fn eft_two_sum",
        "fn eft_fast_two_sum",
        "fn ds_renorm",
        "fn ds_add_f32",
        "fn ds_add",
        "fn ds_mul_f32",
    ] {
        assert!(src.contains(f), "missing primitive: {f}");
    }
    // The TwoProduct residual is the fma form, and the TwoSum subtractions
    // are fma-barriered (the plain Knuth form is compiler-destroyed on the
    // GB10 — banked measurement; regression here is a channel killer).
    assert!(src.contains("fma(a, b, -p)"));
    assert!(
        src.matches("fma(-1.0,").count() >= 5,
        "expected the Knuth/Dekker subtractions to be routed through \
         fma(-1.0, x, y) barriers"
    );
    // No naked Knuth compensation `s - a` may appear as code (comment
    // mentions are fine; code lines are `let x = ...;`).
    for line in src.lines() {
        let code = line.split("//").next().unwrap_or("");
        assert!(
            !code.contains("= s - a"),
            "plain Knuth subtraction leaked into ds_primitives.wgsl: {line}"
        );
    }
}

#[test]
fn consumer_kernels_rely_on_the_shared_primitives() {
    for (name, src, entry) in [
        ("gate_transform", GATE_TRANSFORM_WGSL, "fn gate_transform"),
        ("conv_backward", CONV_BACKWARD_WGSL, "fn conv_backward"),
    ] {
        assert!(src.contains("@compute"), "{name}: missing @compute");
        assert!(
            src.contains("@workgroup_size(256)"),
            "{name}: the chunk-256 workgroup recipe is the measured moat fix"
        );
        assert!(src.contains(entry), "{name}: missing entry point");
        // Primitives come ONLY from the concatenated ds_primitives.wgsl —
        // a private redefinition could silently fork the algebra from the
        // host twin and void the M1 bit-compare.
        assert!(
            !src.contains("fn eft_two_sum") && !src.contains("fn eft_two_prod"),
            "{name}: must not redefine the EFT primitives"
        );
        assert!(
            src.contains("ds_mul_f32") || src.contains("ds_renorm"),
            "{name}: expected use of the shared ds algebra"
        );
    }
}

#[test]
fn gate_transform_selects_the_intercept_side_per_element() {
    // Re-derivation against engine.rs:653-735 (adversarial review, 2026-08-19):
    // the intercept side is SIGN-OF-V dependent and differs between lanes
    // (lower: v.min(0)*c, engine.rs:667; upper: v.max(0)*c), so the host
    // cannot bake it into one slot — the gate must carry an intercept PAIR
    // selected exactly like the slopes. A single-slot regression here drops
    // the upper lane's nonnegative c*vp bias mass: the unsound direction.
    let src = GATE_TRANSFORM_WGSL;
    assert!(
        src.contains("select(g.w, g.z, nonneg)"),
        "gate_transform must select the intercept from the (z, w) pair on the sign of v"
    );
    // The intercept partial goes through the SHARED ds multiply so it stays
    // bit-comparable with the host twin (and U_DS-charged like the value
    // multiply — review item 4; it is a ds composition, not an EFT identity).
    assert!(
        src.contains("ds_mul_f32(v, icept)"),
        "the intercept partial must reuse the shared ds multiply"
    );
    // The retracted claim must stay dead in the kernel's contract comments.
    assert!(
        !src.contains("EFT-EXACT") && !src.contains("no widening"),
        "the false 'EFT-exact / no widening' intercept claim must not return"
    );
}

#[test]
fn provenance_marker_matches_the_module() {
    // R8: the marker the runbook greps for must exist and stay versioned.
    assert!(super::PROVENANCE_MARKER.starts_with("margin-row-gpu-eft-"));
}
