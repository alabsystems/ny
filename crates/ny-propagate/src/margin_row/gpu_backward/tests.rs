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
    DS_PROBE_WGSL, GATE_TRANSFORM_WGSL,
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
fn refusal_labels_are_distinct() {
    // The once-per-(site, reason) telemetry keys on `label()`; two variants
    // sharing a label would silently swallow one reason — the exact failure
    // ("refused=2, no reason") that cost a measurement cycle on gpu_seam.
    // v9 (D4): the key is the LABEL, not a numeric tag, so the `Unmappable`
    // STRING survives instead of every Unmappable collapsing into one bit.
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
        Refusal::ShadowOnly,
    ];
    let mut seen: std::collections::BTreeSet<&'static str> = std::collections::BTreeSet::new();
    for r in all {
        assert!(seen.insert(r.label()), "duplicate label for {r:?}");
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

/// The new DWTimesDW primitive (gate ds pairs, M2 precondition) must also
/// sit inside the U_DS per-op envelope on adversarial operand mixes.
#[test]
fn ds_mul_stays_within_u_ds() {
    use super::ds::ds_mul;
    let vals: &[f64] = &[1.0, -1.0, 3.5e7, 1.0 / 3.0, -std::f64::consts::E, 1e-12];
    let split = |x: f64| {
        #[allow(clippy::cast_possible_truncation)]
        let hi = x as f32;
        #[allow(clippy::cast_possible_truncation)]
        let lo = (x - f64::from(hi)) as f32;
        Ds { hi, lo }
    };
    for &x in vals {
        for &y in vals {
            let (dx, dy) = (split(x), split(y));
            let prod = ds_mul(dx, dy).to_f64();
            let exact = dx.to_f64() * dy.to_f64();
            assert!(
                (prod - exact).abs() <= U_DS * exact.abs().max(1e-300),
                "ds_mul out of envelope: x={x} y={y}"
            );
        }
    }
}

/// The session-3 verify pass REFUTED the exact-split invariant
/// `|v - (hi+lo)| <= 2^-48 |v|` for f32-SUBNORMAL components (their rounding
/// is ABSOLUTE, `<= 2^-150`, not relative), with rustc-run counterexamples
/// (scratchpad split_ce.rs). This pin ports those exact values: the OLD
/// relative charge measurably under-covers, and the NEW per-element total —
/// guaranteed in-kernel cover base + exact-residual split-floor excess —
/// dominates the measured residual x1 in EVERY band.
#[test]
fn split_residual_excess_covers_the_subnormal_band() {
    use super::transaction::{
        ds_split_excess, ds_split_f64, ds_split_residual, DS_GATE_COVER, DS_SPLIT_COVER,
        DS_SPLIT_REL,
    };
    // The cover bases are pinned to their derivations: HALF of the relative
    // constants whose guaranteed in-kernel charges they floor (the 2x slack
    // absorbs the kernels' own f32 mass roundings).
    assert_eq!(DS_SPLIT_COVER, 2f64.powi(-48));
    assert_eq!(DS_SPLIT_COVER, DS_SPLIT_REL / 2.0);
    assert_eq!(DS_GATE_COVER, 2f64.powi(-45));
    assert_eq!(DS_GATE_COVER, U_DS / 2.0);

    // (v, exact residual): the two verify-pass counterexamples.
    // 1) lo lands f32-SUBNORMAL: residual = 2^-150 = 7.006492321624085e-46
    //    vs the refuted claim 2^-48|v| = 5.474e-48 — 128x under.
    // 2) hi itself is f32-SUBNORMAL: residual = 2^-156 + 2^-182
    //    = 1.0947644415670245e-47 vs the charged 2^-47*mass = 5.220e-54 —
    //    2.1e6x under.
    let cases: &[(f64, f64)] = &[
        (2f64.powi(-110) + 2f64.powi(-150), 2f64.powi(-150)),
        (
            (1.0 + 2f64.powi(-26) + 2f64.powi(-52)) * 2f64.powi(-130),
            2f64.powi(-156) + 2f64.powi(-182),
        ),
    ];
    for &(v, expect_r) in cases {
        let (hi, lo) = ds_split_f64(v).expect("finite");
        let r = ds_split_residual(v, hi, lo);
        assert_eq!(r, expect_r, "exact residual drifted for v={v:e}");
        let mass = f64::from(hi).abs() + f64::from(lo).abs();
        // The refuted relative ball really under-covers here (else this pin
        // no longer witnesses the bug it fixed).
        assert!(
            DS_SPLIT_REL * mass < r,
            "v={v:e} no longer refutes the relative ball; update the pin"
        );
        // NEW total charge >= residual x1, for BOTH cover flavors
        // (weight/bias sites and gate sites).
        for cover in [DS_SPLIT_COVER, DS_GATE_COVER] {
            let excess = ds_split_excess(v, hi, lo, cover);
            assert!(excess > 0.0, "v={v:e}: subnormal band must show excess");
            let total = cover * mass + excess;
            assert!(
                total >= r,
                "v={v:e}: charge {total:e} < residual {r:e} (cover={cover:e})"
            );
        }
    }
    // The exact decimal forms, pinned as measured (split_ce.rs, 2026-08-20).
    {
        let (hi, lo) = ds_split_f64(cases[0].0).unwrap();
        assert_eq!(
            ds_split_residual(cases[0].0, hi, lo),
            7.006_492_321_624_085e-46
        );
        let (hi, lo) = ds_split_f64(cases[1].0).unwrap();
        assert_eq!(
            ds_split_residual(cases[1].0, hi, lo),
            1.094_764_441_567_024_5e-47
        );
    }
    // NORMAL band (and exact splits): the excess channel is identically
    // ZERO — ordinary passes pay nothing — while the relative ball holds.
    for &v in &[
        1.0,
        -2.5e-7,
        3.14159e-4,
        1e-30,           // lo is subnormal but EXACT: residual 0
        2f64.powi(-120), // power of two: exact split
        0.0,
        -0.0,
    ] {
        let (hi, lo) = ds_split_f64(v).unwrap();
        let r = ds_split_residual(v, hi, lo);
        assert_eq!(ds_split_excess(v, hi, lo, DS_SPLIT_COVER), 0.0, "v={v:e}");
        assert_eq!(ds_split_excess(v, hi, lo, DS_GATE_COVER), 0.0, "v={v:e}");
        let mass = f64::from(hi).abs() + f64::from(lo).abs();
        assert!(r <= DS_SPLIT_REL * mass || mass == 0.0, "v={v:e}");
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

    // The TIGHTENED per-tap dot envelope (error-charge session): the constant
    // is exactly 10 * 2^-48 = 10u^2 (3-bit mantissa, exact in f64) ...
    use super::ds::{gamma_ds_dot, DS_TAP_EPS};
    assert_eq!(DS_TAP_EPS, 10.0 * 2f64.powi(-48));
    // ... which dominates the CONSERVATIVE reading of the documented per-tap
    // sum with real headroom: fma DWTimesDW variants are published <= 5u^2 /
    // <= 6u^2 depending on the exact shape (we charge for 6), accurate
    // ds_add <= 3u^2/(1-4u).
    let u2 = 2f64.powi(-48);
    let conservative = 6.0 * u2 + (3.0 * u2) / (1.0 - 4.0 * 2f64.powi(-24));
    assert!(DS_TAP_EPS > conservative * 1.10);
    for &taps in &[1usize, 100, 4608, 100_000] {
        // Same self-domination discipline as gamma_ds ...
        #[allow(clippy::cast_precision_loss)]
        let nu = taps as f64 * DS_TAP_EPS;
        let fl = nu / (1.0 - nu);
        assert!(
            gamma_ds_dot(taps) >= f64::from_bits(fl.to_bits() + 2),
            "gamma_ds_dot must dominate its own roundings at taps={taps}"
        );
        // ... while being the documented ~3.2x TIGHTER than the blanket
        // 2-ops-at-U_DS envelope (the point of the session: same soundness,
        // smaller a-priori charge). Ratio pinned band-wise so neither a
        // silent loosening NOR an undocumented further cut slips through.
        let ratio = gamma_ds(taps) / gamma_ds_dot(taps);
        assert!(
            (3.15..3.25).contains(&ratio),
            "gamma_ds/gamma_ds_dot ratio drifted at taps={taps}: {ratio}"
        );
    }
    assert_eq!(gamma_ds_dot(usize::MAX / 4), 1.0);
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
        ("ds_probe", DS_PROBE_WGSL, "fn ds_probe"),
        ("head_gemm", super::HEAD_GEMM_WGSL, "fn head_gemm"),
        ("bias_reduce", super::REDUCE_WGSL, "fn bias_reduce"),
        // B2: the production TREE entry lives in the same module.
        ("bias_reduce_wg", super::REDUCE_WGSL, "fn bias_reduce_wg"),
        ("merge_add", super::MERGE_ADD_WGSL, "fn merge_add"),
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
            src.contains("ds_mul_f32") || src.contains("ds_add") || src.contains("ds_renorm"),
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
        src.contains("select(ghi.w, ghi.z, nonneg)")
            && src.contains("select(glo.w, glo.z, nonneg)"),
        "gate_transform must select the intercept ds pair on the sign of v"
    );
    // The intercept partial goes through the SHARED ds multiply so it stays
    // bit-comparable with the host twin (and U_DS-charged like the value
    // multiply — review item 4; it is a ds composition, not an EFT identity).
    assert!(
        src.contains("ds_mul(v, icept)"),
        "the intercept partial must reuse the shared ds multiply"
    );
    // The partial's E carry multiplies the max over BOTH intercept sides
    // (the stored sign can differ from the exact one's across the kink); a
    // selected-side charge is an UNDERCHARGE on the pass-through side.
    assert!(
        src.contains(
            "max(ds_abs_bound(vec2<f32>(ghi.z, glo.z)), ds_abs_bound(vec2<f32>(ghi.w, glo.w)))"
        ),
        "the partial_err carry must use the intercept-PAIR Lipschitz bound"
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

// ---------------------------------------------------------------------------
// M1 probe lane discipline (the host half of the on-device self-check)
// ---------------------------------------------------------------------------

/// The probe table must be big enough to mint authority, DISCRIMINATING (a
/// de-fused fma or a reassociated TwoSum must flip bits on many lanes), and
/// deterministic (the device dispatch uploads exactly this table).
#[test]
fn ds_probe_lanes_cover_and_discriminate() {
    use super::authority::PARITY_MIN_LANES;
    use super::probe::{build_ds_probe_lanes, host_ds_probe, DS_PROBE_LANES};

    let lanes = build_ds_probe_lanes();
    assert_eq!(lanes.len(), DS_PROBE_LANES);
    assert!(
        lanes.len() >= PARITY_MIN_LANES,
        "the ds probe alone must clear the parity minimum"
    );
    // Deterministic: two builds are identical (the device compares against
    // the host twin of the SAME table).
    assert_eq!(lanes, build_ds_probe_lanes());

    let host = host_ds_probe(&lanes);
    assert_eq!(host.len(), 3 * lanes.len(), "three output planes per lane");

    // Discrimination: a broken TwoProduct zeroes the mul-plane lo parts; a
    // broken TwoSum zeroes the add-plane residual carries. Require a large
    // fraction of lanes to carry NONZERO lo bits in each plane, so any such
    // regression flips bits at hundreds of indices, not an unlucky few.
    let n = lanes.len();
    let live = |plane: &[Ds]| plane.iter().filter(|d| d.lo != 0.0).count();
    assert!(
        live(&host[..n]) > n / 4,
        "mul plane must carry live product residuals"
    );
    assert!(
        live(&host[n..2 * n]) > n / 4,
        "addf plane must carry live sum residuals"
    );
    assert!(
        live(&host[2 * n..]) > n / 4,
        "add plane must carry live DWPlusDW residuals"
    );
}

/// The probe is deliberately BLIND to flush-to-zero (same discipline as
/// ny-gpu `ops/eft_selfcheck.rs`, pinned for the same reason): fma may DAZ a
/// subnormal OPERAND even under the preserving channel, so one subnormal lane
/// could falsely kill the channel. Every operand and every host-twin output
/// must therefore be normal-range or exactly zero; the subnormal contract is
/// carried by the denorm-preserve ADMISSION check instead.
#[test]
fn ds_probe_lanes_are_normal_range_so_the_admission_check_carries_ftz() {
    use super::probe::{
        build_ds_probe_lanes, build_gate_fixture, host_ds_probe, host_gate_transform,
    };
    let subnormal = |x: f32| x != 0.0 && x.abs() < f32::MIN_POSITIVE;

    let lanes = build_ds_probe_lanes();
    for (i, lane) in lanes.iter().enumerate() {
        for (k, v) in lane.iter().enumerate() {
            assert!(!subnormal(*v), "lane {i} operand {k} = {v:e} is SUBNORMAL");
        }
    }
    for (i, d) in host_ds_probe(&lanes).iter().enumerate() {
        assert!(
            !subnormal(d.hi),
            "host stream {i} hi = {:e} SUBNORMAL",
            d.hi
        );
        assert!(
            !subnormal(d.lo),
            "host stream {i} lo = {:e} SUBNORMAL",
            d.lo
        );
    }

    let fixture = build_gate_fixture();
    let (coeff_out, err_out, bias, partial_err) = host_gate_transform(&fixture);
    for d in coeff_out.iter().chain(&bias) {
        assert!(!subnormal(d.hi) && !subnormal(d.lo), "gate plane subnormal");
    }
    for e in err_out.iter().chain(&partial_err) {
        assert!(!subnormal(*e), "gate err plane subnormal");
    }
}

/// The gate fixture must exercise BOTH lane-direction gate styles and BOTH
/// coefficient signs (the intercept-PAIR selection of review item 4), and its
/// host twin must implement the per-element spec: slope and intercept picked
/// on the sign of v, error contracted by max(|slopes|), partial = ds v*icept.
#[test]
fn gate_fixture_exercises_the_intercept_pair_and_matches_the_spec() {
    use super::ds::{ds_abs_bound, ds_mul, U_DS_F32};
    use super::probe::{build_gate_fixture, host_gate_transform, GATE_NEURONS, GATE_ROWS};

    let fixture = build_gate_fixture();
    assert_eq!(fixture.gate.len(), 2 * GATE_NEURONS, "hi + lo quads");
    assert_eq!(fixture.coeff.len(), GATE_NEURONS * GATE_ROWS);
    // Both styles present among the HI quads: lower bakes (alpha, s, 0, c),
    // upper (s, alpha, c, 0).
    assert!(fixture
        .gate
        .iter()
        .step_by(2)
        .any(|g| g[2] == 0.0 && g[3] > 0.0));
    assert!(fixture
        .gate
        .iter()
        .step_by(2)
        .any(|g| g[3] == 0.0 && g[2] > 0.0));
    // The gates genuinely carry ds LO parts (they ride as f64 splits).
    assert!(fixture
        .gate
        .iter()
        .skip(1)
        .step_by(2)
        .any(|g| g.iter().any(|v| *v != 0.0)));
    // Both coefficient signs present (else one intercept side is never taken).
    assert!(fixture.coeff.iter().any(|v| v.hi > 0.0));
    assert!(fixture.coeff.iter().any(|v| v.hi < 0.0));

    let (coeff_out, err_out, bias, partial_err) = host_gate_transform(&fixture);
    for idx in 0..fixture.coeff.len() {
        let (slope, icept, m, mi) = fixture.select(idx);
        let v = fixture.coeff[idx];
        let o = ds_mul(v, slope);
        let p = ds_mul(v, icept);
        assert_eq!(coeff_out[idx], o, "coeff idx {idx}");
        assert_eq!(bias[idx], p, "bias idx {idx}");
        // The E planes carry the contraction/carry PLUS the 2x DS_UNIT charge
        // on the produced magnitude (multiply residue + gate-split ball). The
        // partial's carry multiplies mi — the max over BOTH intercept sides —
        // because the stored sign can differ from the exact one's across the
        // kink (a selected-side charge would UNDERCHARGE the pass-through
        // side: false-VERIFY direction).
        assert_eq!(
            err_out[idx].to_bits(),
            fixture.err[idx]
                .mul_add(m, 2.0 * U_DS_F32 * ds_abs_bound(o))
                .to_bits(),
            "err idx {idx}"
        );
        assert!(
            mi > 0.0,
            "fixture must exercise a nonzero intercept bound at idx {idx}"
        );
        assert_eq!(
            partial_err[idx].to_bits(),
            fixture.err[idx]
                .mul_add(mi, 2.0 * U_DS_F32 * ds_abs_bound(p))
                .to_bits(),
            "partial err idx {idx}"
        );
    }
}

// ---------------------------------------------------------------------------
// Device-gated M1 parity harness (the REAL self-check on the live adapter).
// `cargo test -p ny-propagate --features gpu-tests gpu_backward`
// ---------------------------------------------------------------------------

/// The on-device M1 bit-parity measurement: dispatch the ds-primitive probe
/// composition AND the production gate-transform kernel on a REAL lane device
/// (its own `WgpuDevice`, drain-safe transactions) and require BIT-IDENTITY
/// with the host ds twin on every lane. This is the design's kill criterion:
/// a failure here means the channel is dead on this adapter — do not tune.
///
/// Deliberately does NOT go through `device_authority()`/`channel_dead()`:
/// those are process-wide latches shared with the fail-closed host tests in
/// this file (`parity_mismatch_latches_the_channel_dead` trips the latch on
/// purpose), and this test must stay order-independent.
#[test]
#[cfg(feature = "gpu-tests")]
fn m1_device_parity_is_bit_identical_on_this_adapter() {
    use super::authority::{DeviceParityProof, VerdictAuthority, PARITY_MIN_LANES};
    use super::probe::self_check_streams;
    use ny_core::MarginRowLaneGpu;

    let device = ny_gpu::MarginRowLaneDevice::new()
        .expect("gpu-tests requires a live wgpu adapter for the lane device");
    let denorm = device.denorm_preserve_enabled();
    let streams = self_check_streams(&device).expect("probe transactions must run");
    println!(
        "[gpu-backward gpu-test] adapter: {} | denorm-preserve resolved: {} | lanes: {} | device_ms: {:.1}",
        device.adapter_summary(),
        denorm,
        streams.host.len(),
        streams.device_ms,
    );
    assert_eq!(streams.host.len(), streams.device.len(), "stream lengths");
    assert!(streams.host.len() >= PARITY_MIN_LANES);
    if let Err(index) = bit_compare_streams(&streams.host, &streams.device) {
        let h = streams.host[index];
        let d = streams.device[index];
        panic!(
            "M1 parity mismatch at lane {index}: host=({:#010x},{:#010x}) device=({:#010x},{:#010x}) — \
             the EFT channel is DEAD on this adapter (design section 7 M1 kill criterion)",
            h.hi.to_bits(),
            h.lo.to_bits(),
            d.0.to_bits(),
            d.1.to_bits()
        );
    }
    // The real qualification chain mints authority from exactly this readback
    // (qualify on a PASSING stream touches no process-wide latch).
    if denorm {
        let proof = DeviceParityProof::qualify(true, &streams.host, &streams.device)
            .expect("bit-identical stream + denorm ON must qualify");
        let auth = VerdictAuthority::grant(proof);
        assert!(auth.parity_lanes() >= PARITY_MIN_LANES);
        println!(
            "[gpu-backward gpu-test] M1 qualification PASS: {} lanes",
            auth.parity_lanes()
        );
    } else {
        println!(
            "[gpu-backward gpu-test] denorm-preserve resolved OFF: parity is bit-identical but \
             admission would refuse Unmappable on this adapter (by design)"
        );
    }
}

// ---------------------------------------------------------------------------
// L-lane transaction: program assembly (host-only)
// ---------------------------------------------------------------------------

/// The M3 authority switch is ON as of 2026-08-21, and this pin still exists:
/// it now guards against an accidental flip BACK, and against the root site
/// being authorized without the rest of the chain that makes it safe.
///
/// It was flipped by the M2 x15 moat protocol (docs/BIG3_CORE_CAMPAIGN
/// "M2 x15 PASS"): 15/15 verdicts identical to the bank, zero SHADOW TRIPs,
/// 16/16 free of the sweep post-submit kill line (the criterion that failed
/// 1/6 pre-fix, with the exact poisoning row now clean 3/3), max median_radius
/// 1.029e-6 = 6.8x under the R1 bar, expansion fingerprints exact with zero
/// variance. Re-confirmed bit-identical on the scored `mip,cuda` tier.
///
/// Reverting is equally deliberate: change the constant AND this pin together,
/// and say which measurement justified it.
#[test]
fn verdict_authority_is_m3_gated_off() {
    // `const {}` makes this a build-time ratchet instead of a test-time one:
    // flipping the flag now fails to COMPILE, which is the correct strength for
    // a claim about verifier authority that no run may quietly grant itself.
    const {
        assert!(
            !super::M3_AUTHORITY_GRANTED,
            "M3_AUTHORITY_GRANTED flipped: only a session with an ARMED run whose telemetry \
             actually appears, on rows where a wrong bound is detectable, may do this"
        )
    };
    // The flip must remain scoped to the ROOT site. The B4/B5/B6 seams each
    // await their own x3 moat; sweeping them along would grant verdict
    // authority to passes that have never been shadow-compared at their own
    // call site.
    assert_eq!(
        super::SITE_AUTHORITY_GRANTED,
        0,
        "M3 must not be used as a back door to authorize the B4/B5/B6 sites"
    );
}

#[test]
fn shadow_lever_declaration_is_the_documented_gate() {
    let decl = &ny_levers::decls::margin_row::MARGIN_ROW_GPU_SHADOW;
    assert_eq!(decl.name, "NY_MARGIN_ROW_GPU_SHADOW");
}

fn chain_fixture() -> (
    crate::margin_row::net::TwinNet,
    crate::margin_row::root::RootGates,
) {
    let (net, root, _, _) =
        super::super::gpu_seam::tests::compile(&super::super::gpu_seam::tests::chain_spec());
    (net, root)
}

fn residual_fixture() -> (
    crate::margin_row::net::TwinNet,
    crate::margin_row::root::RootGates,
) {
    let (net, root, _, _) =
        super::super::gpu_seam::tests::compile(&super::super::gpu_seam::tests::residual_spec());
    (net, root)
}

#[test]
fn built_program_walks_the_chain_fixture_in_backward_order() {
    use crate::margin_row::engine::{BackwardEngine, LaneDir};

    let (net, root) = chain_fixture();
    let eng = BackwardEngine::new(&net, &root);
    let statics = super::cache::PlanStatic::build(&eng, None, None).expect("statics must assemble");
    let seed = eng.identity_seed();
    for dir in [LaneDir::Lower, LaneDir::Upper] {
        let built = super::transaction::build_program(
            &statics,
            &eng,
            None,
            &seed,
            dir,
            PassShape::default(),
            net.n_y,
            net.n_in,
            net.n_y,
        )
        .expect("chain fixture must assemble");
        // Backward order: head (bias fold + gemm), relu1 (gate + reduce),
        // conv2 (bias fold + gather), relu0, conv1. (B2: reduces dispatch
        // the tree entry.)
        let labels: Vec<&str> = built.steps.iter().map(|s| s.label).collect();
        assert_eq!(
            labels,
            vec![
                "mr_gpu_backward_bias_reduce_wg",
                "mr_gpu_backward_head_gemm",
                "mr_gpu_backward_gate_transform",
                "mr_gpu_backward_bias_reduce_wg",
                "mr_gpu_backward_bias_reduce_wg",
                "mr_gpu_backward_conv_backward",
                "mr_gpu_backward_gate_transform",
                "mr_gpu_backward_bias_reduce_wg",
                "mr_gpu_backward_bias_reduce_wg",
                "mr_gpu_backward_conv_backward",
            ],
            "step order moved for {dir:?}"
        );
        assert_eq!(built.n_in, net.n_in);
        assert_eq!(built.rows, net.n_y);
        // Uniforms are std140-clean, bindings and readbacks are in-arena.
        for step in &built.steps {
            assert!(!step.uniforms.is_empty() && step.uniforms.len() % 16 == 0);
            assert!(step.workgroups[0] >= 1);
            for b in &step.bindings {
                assert!(b.buffer < built.buffers.len(), "binding out of arena");
            }
        }
        for rb in built.readbacks {
            assert!(rb < built.buffers.len(), "readback out of arena");
        }
        // The egress charges are sane: widening factors are finite and > 1,
        // flush floors are positive and margin-invisible.
        assert!(built.e_widen.is_finite() && built.e_widen > 1.0);
        assert!(built.eb_widen.is_finite() && built.eb_widen > 1.0);
        assert!(built.flush_e > 0.0 && built.flush_e < 1e-30);
        assert!(built.flush_eb > 0.0 && built.flush_eb < 1e-30);
        // Verify-fix: every fixture parameter splits in the normal band, so
        // the subnormal split-floor channel must be EXACTLY zero (the
        // channel's directed ops are exact-zero-preserving — a zero product
        // or addend owes no up-step). Any nonzero value here means either a
        // charge leaked into the wrong band or the exact-zero discipline
        // regressed into blind next_up(0.0) seeding.
        assert_eq!(built.split_e, 0.0, "split_e = {:e}", built.split_e);
        assert_eq!(built.split_eb, 0.0, "split_eb = {:e}", built.split_eb);
    }
}

#[test]
fn built_program_maps_the_residual_fixture_with_a_merge() {
    use crate::margin_row::engine::{BackwardEngine, LaneDir};

    let (net, root) = residual_fixture();
    let eng = BackwardEngine::new(&net, &root);
    let statics = super::cache::PlanStatic::build(&eng, None, None).expect("statics must assemble");
    // Provenance completeness: the residual fixture has two convs, both
    // listed AND both carried as cached conv statics (index-aligned).
    assert_eq!(
        statics.ext.conv_ops.len(),
        2,
        "residual fixture conv provenance"
    );
    assert_eq!(statics.convs.len(), 2, "cached conv statics must align");
    let seed = eng.identity_seed();
    let built = super::transaction::build_program(
        &statics,
        &eng,
        None,
        &seed,
        LaneDir::Lower,
        PassShape::default(),
        net.n_y,
        net.n_in,
        net.n_y,
    )
    .expect("residual fixture must assemble");
    let merges = built
        .steps
        .iter()
        .filter(|s| s.label == "mr_gpu_backward_merge_add")
        .count();
    assert_eq!(merges, 1, "one identity-skip Add means exactly one merge");
    assert_eq!(built.n_in, net.n_in);
}

/// Field-by-field bit comparison of two built programs (the B1 pin's
/// comparator): steps (label/entry/uniform bytes/bindings/workgroups),
/// arena inits (variant + bytes/length), readbacks, and every egress charge.
fn assert_programs_bit_identical(
    a: &super::transaction::BuiltProgram<'_>,
    b: &super::transaction::BuiltProgram<'_>,
    what: &str,
) {
    use super::transaction::Init;
    assert_eq!(a.steps.len(), b.steps.len(), "{what}: step count");
    for (i, (sa, sb)) in a.steps.iter().zip(&b.steps).enumerate() {
        assert_eq!(sa.label, sb.label, "{what}: step {i} label");
        assert_eq!(sa.entry, sb.entry, "{what}: step {i} entry");
        assert_eq!(sa.uniforms, sb.uniforms, "{what}: step {i} uniforms");
        assert_eq!(sa.workgroups, sb.workgroups, "{what}: step {i} workgroups");
        assert_eq!(
            sa.bindings.len(),
            sb.bindings.len(),
            "{what}: step {i} bindings"
        );
        for (ba, bb) in sa.bindings.iter().zip(sb.bindings.iter()) {
            assert_eq!(ba.buffer, bb.buffer, "{what}: step {i} binding index");
            assert_eq!(ba.read_only, bb.read_only, "{what}: step {i} binding mode");
        }
    }
    assert_eq!(a.buffers.len(), b.buffers.len(), "{what}: arena size");
    for (i, (ia, ib)) in a.buffers.iter().zip(&b.buffers).enumerate() {
        let payload = |init: &Init<'_>| -> (u8, Vec<u8>, u64) {
            match init {
                Init::Data(bytes) => (0, bytes.clone(), 0),
                // DataRef and Data with equal bytes are the SAME upload.
                Init::DataRef(bytes) => (0, bytes.to_vec(), 0),
                Init::Zeroed(len) => (1, Vec::new(), *len),
                Init::Resident(id) => (2, Vec::new(), id.raw()),
            }
        };
        assert_eq!(payload(ia), payload(ib), "{what}: arena buffer {i}");
    }
    assert_eq!(a.readbacks, b.readbacks, "{what}: readbacks");
    assert_eq!(a.e_widen.to_bits(), b.e_widen.to_bits(), "{what}: e_widen");
    assert_eq!(
        a.eb_widen.to_bits(),
        b.eb_widen.to_bits(),
        "{what}: eb_widen"
    );
    assert_eq!(a.flush_e.to_bits(), b.flush_e.to_bits(), "{what}: flush_e");
    assert_eq!(
        a.flush_eb.to_bits(),
        b.flush_eb.to_bits(),
        "{what}: flush_eb"
    );
    assert_eq!(a.split_e.to_bits(), b.split_e.to_bits(), "{what}: split_e");
    assert_eq!(
        a.split_eb.to_bits(),
        b.split_eb.to_bits(),
        "{what}: split_eb"
    );
    assert_eq!(
        a.rho_star_ds.to_bits(),
        b.rho_star_ds.to_bits(),
        "{what}: rho_star_ds"
    );
    assert_eq!(a.profile, b.profile, "{what}: charge profile");
    assert_eq!(a.n_in, b.n_in, "{what}: n_in");
    assert_eq!(a.rows, b.rows, "{what}: rows");
}

/// B1 PIN 1 (no GPU): the residency cache is a PURE cache — a program built
/// from a CACHE-HIT PlanStatic is bit-identical (steps, arena bytes, every
/// egress charge) to one built from a FRESH PlanStatic, and the hit/miss
/// telemetry counts what happened.
#[test]
fn plan_cache_hit_builds_a_bit_identical_program() {
    use crate::margin_row::engine::{BackwardEngine, LaneDir};

    let (net, root) = chain_fixture();
    let eng = BackwardEngine::new(&net, &root);
    let seed = eng.identity_seed();

    let cache = super::cache::PlanCache::default();
    assert!(cache.probe(&eng).is_none(), "cold cache must miss");
    assert_eq!(cache.counters(), (0, 1), "miss telemetry");
    cache.store(std::sync::Arc::new(
        super::cache::PlanStatic::build(&eng, None, None).expect("statics"),
    ));
    let hit = cache.probe(&eng).expect("warm cache must hit");
    assert_eq!(cache.counters(), (1, 1), "hit telemetry");

    // A SECOND fresh build (what the uncached verdict path does per pass)
    // must produce the same program as the hit, bit for bit.
    let fresh2 = super::cache::PlanStatic::build(&eng, None, None).expect("statics");
    for dir in [LaneDir::Lower, LaneDir::Upper] {
        let from_hit = super::transaction::build_program(
            &hit,
            &eng,
            None,
            &seed,
            dir,
            PassShape::default(),
            net.n_y,
            net.n_in,
            net.n_y,
        )
        .expect("hit build");
        let from_fresh = super::transaction::build_program(
            &fresh2,
            &eng,
            None,
            &seed,
            dir,
            PassShape::default(),
            net.n_y,
            net.n_in,
            net.n_y,
        )
        .expect("fresh build");
        assert_programs_bit_identical(&from_hit, &from_fresh, "hit vs fresh");
    }

    // A DIFFERENT engine (same spec, new compile) must MISS fail-closed:
    // the fingerprint is identity-based, never shape-only.
    let (net2, root2) = chain_fixture();
    let eng2 = BackwardEngine::new(&net2, &root2);
    assert!(
        cache.probe(&eng2).is_none(),
        "a different engine instance must miss"
    );
}

/// B1 PIN 2 (no GPU, fix A3): the CACHED realization probe agrees with the
/// per-pass `gpu_seam::probe_guard` — Ok on a genuine CPU pass, Probe-trip
/// on a corrupted bound — for root gates AND a piece-fixed domain.
#[test]
fn cached_probe_guard_matches_the_per_pass_probe() {
    use crate::margin_row::engine::{domain_gates, BackwardEngine, LaneDir};

    let (net, root) = chain_fixture();
    let eng = BackwardEngine::new(&net, &root);
    let seed = eng.identity_seed();
    let statics = super::cache::PlanStatic::build(&eng, None, None).expect("statics");
    let doms = {
        let mut doms = vec![None];
        if let Some(li) = root.layers.iter().position(|l| !l.unst.is_empty()) {
            let dom = domain_gates(&root, &[(li, 0, 1)]);
            if !dom.infeasible {
                doms.push(Some(dom));
            }
        }
        doms
    };
    for dom in &doms {
        let cpu = eng
            .run(&seed, dom.as_ref(), LaneDir::Lower, None, false)
            .expect("cpu pass");
        // Genuine pass: both probes accept.
        super::super::gpu_seam::probe_guard(&eng, &seed, dom.as_ref(), LaneDir::Lower, &cpu)
            .expect("per-pass probe accepts the CPU pass");
        super::cache::probe_guard_cached(
            &eng,
            &seed,
            dom.as_ref(),
            LaneDir::Lower,
            &cpu,
            &statics.probe,
        )
        .expect("cached probe accepts the CPU pass");
        // Corrupted pass: push every bias far ABOVE the realized functional —
        // a LOWER bound above a realized value must trip both probes.
        let mut bad = eng
            .run(&seed, dom.as_ref(), LaneDir::Lower, None, false)
            .expect("cpu pass");
        for b in &mut bad.b {
            *b += 1e6;
        }
        assert!(
            matches!(
                super::super::gpu_seam::probe_guard(
                    &eng,
                    &seed,
                    dom.as_ref(),
                    LaneDir::Lower,
                    &bad
                ),
                Err(super::super::gpu_seam::Refusal::Probe)
            ),
            "per-pass probe must trip"
        );
        assert!(
            matches!(
                super::cache::probe_guard_cached(
                    &eng,
                    &seed,
                    dom.as_ref(),
                    LaneDir::Lower,
                    &bad,
                    &statics.probe,
                ),
                Err(Refusal::Probe)
            ),
            "cached probe must trip"
        );
    }
}

/// B1 PIN 3 (no GPU): the cache-hit path's per-dom GATE ADMISSION refuses
/// exactly where a fresh `build_plan_ext` refuses — pinned on the
/// fractional-alpha dom (the B7 boundary: refused today on BOTH paths).
#[test]
fn cache_hit_gate_admission_matches_the_plan_builder() {
    use crate::margin_row::engine::BackwardEngine;

    let (net, mut root) = chain_fixture();
    // A fractional, non-f32-exact lower slope on the first unstable neuron:
    // outward_gate_f32 refuses it (f32-exactness check), so BOTH the fresh
    // plan build and the cache-hit admission must refuse.
    let li = root
        .layers
        .iter()
        .position(|l| !l.unst.is_empty())
        .expect("chain fixture has unstable neurons");
    let j = root.layers[li].unst[0];
    root.layers[li].alpha[j] = 0.3; // not f32-exact
    let eng = BackwardEngine::new(&net, &root);
    let fresh = super::super::gpu_seam::build_plan_ext(&eng, None);
    assert!(
        fresh.is_err(),
        "fresh plan build must refuse fractional alpha"
    );
    // The admission check re-run on a (hypothetical) hit refuses identically.
    let relus: Vec<usize> = (0..root.layers.len()).collect();
    assert!(
        super::cache::check_gate_admission(&eng, None, &relus).is_err(),
        "cache-hit gate admission must refuse fractional alpha"
    );
}

/// B2 PIN (no GPU): the TREE twin encloses the exact f64 fold within the
/// serial charge envelope (`gamma_ds_dot(n+2)` dominates ANY summation
/// order — the push_reduce argument), across an n range that exercises
/// empty lanes, one-element lanes, and multi-element lanes.
#[test]
fn tree_reduce_twin_stays_within_the_serial_envelope() {
    use super::ds::{bias_reduce_wg_twin, ds_abs_bound, gamma_ds_dot, Ds};

    let mut rng = Lcg(0xB1A5_2ED0_1234_5678);
    for &n in &[1usize, 2, 7, 255, 256, 257, 1000] {
        let rows = 3usize;
        let split = |v: f64| {
            #[allow(clippy::cast_possible_truncation)]
            let hi = v as f32;
            #[allow(clippy::cast_possible_truncation)]
            let lo = (v - f64::from(hi)) as f32;
            Ds { hi, lo }
        };
        let plane: Vec<Ds> = (0..n * rows)
            .map(|_| split(f64::from(rng.coeff()) * (1.0 + 3.7e-10)))
            .collect();
        let err: Vec<f32> = (0..n * rows).map(|_| rng.coeff().abs() * 1e-7).collect();
        let scale: Vec<Ds> = (0..n)
            .map(|_| split(f64::from(rng.coeff()) * (1.0 + 9.1e-10)))
            .collect();
        let mut bias = vec![Ds { hi: 0.0, lo: 0.0 }; rows];
        let mut ebias = vec![0.0f32; rows];
        bias_reduce_wg_twin(
            &plane,
            &err,
            Some(&scale),
            n,
            rows,
            0.0,
            0.0,
            &mut bias,
            &mut ebias,
        );
        for r in 0..rows {
            let mut exact = 0.0f64;
            let mut mass = 0.0f64;
            for j in 0..n {
                let t = plane[j * rows + r].to_f64() * scale[j].to_f64();
                exact += t;
                mass += t.abs();
            }
            let diff = (bias[r].to_f64() - exact).abs();
            #[allow(clippy::cast_precision_loss)]
            let g64 = 2.0 * (n as f64) * f64::EPSILON;
            assert!(
                diff <= (gamma_ds_dot(n + 2) + g64) * mass,
                "tree twin outside the serial envelope at n={n} r={r}: \
                 diff={diff:.3e} mass={mass:.3e}"
            );
            let _ = ds_abs_bound(bias[r]);
        }
    }
}

/// The PROF lever is declared through the chokepoint with the documented
/// name (the `shadow_lever_declaration_is_the_documented_gate` twin).
#[test]
fn prof_lever_declaration_is_the_documented_gate() {
    let decl = &ny_levers::decls::margin_row::MARGIN_ROW_GPU_PROF;
    assert_eq!(decl.name, "NY_MARGIN_ROW_GPU_PROF");
}

#[test]
fn egress_is_exact_on_values_and_directed_on_errors() {
    let built = super::transaction::BuiltProgram {
        buffers: Vec::new(),
        steps: Vec::new(),
        readbacks: vec![0, 1, 2, 3],
        colls: Vec::new(),
        collect_unst_abs: false,
        amax: 1.0,
        e_widen: 1.0 + 1e-6,
        eb_widen: 1.0 + 1e-6,
        flush_e: 1e-33,
        flush_eb: 1e-33,
        // Nonzero split floors so the egress application is itself tested
        // (verify-fix: these are the subnormal-split absolute channels).
        split_e: 1e-20,
        split_eb: 2e-20,
        arena_bytes: 0,
        rho_star_ds: 0.0,
        profile: Vec::new(),
        n_in: 2,
        rows: 1,
    };
    let hi = 1.5f32;
    let lo = f32::from_bits(0x2880_0000); // small positive, exercises hi+lo
    let coeff = vec![hi, lo, -3.25, 0.0];
    let err = vec![1e-7f32, 2e-7];
    let bias = vec![0.75f32, -lo];
    let ebias = vec![3e-8f32];
    let pen = vec![1e-9f64];
    let pass = super::transaction::egress(&built, vec![coeff, err, bias, ebias], &pen)
        .expect("well-formed readback");
    // Values: EXACT ds -> f64.
    assert_eq!(pass.a[[0, 0]], f64::from(hi) + f64::from(lo));
    assert_eq!(pass.a[[1, 0]], -3.25);
    assert_eq!(pass.b[0], f64::from(0.75f32) - f64::from(lo));
    // Errors: strictly dominated by the widened + floored + penalized forms,
    // INCLUDING the absolute split-floor channels.
    let e = pass.e.as_ref().expect("outward error lane");
    assert!(e[[0, 0]] > f64::from(1e-7f32) && e[[0, 0]] < 2e-7);
    assert!(e[[0, 0]] > f64::from(1e-7f32) * (1.0 + 1e-6) + 1e-20);
    assert!(pass.eb[0] > f64::from(3e-8f32) + 1e-9 + 2e-20);

    // A negative error entry is a Payload refusal, not a repair.
    let bad = super::transaction::egress(
        &built,
        vec![
            vec![1.0, 0.0, 1.0, 0.0],
            vec![-1e-9f32, 0.0],
            vec![0.0, 0.0],
            vec![0.0],
        ],
        &pen,
    );
    assert!(matches!(bad, Err(Refusal::Payload)));
}

#[test]
fn enclosure_overlap_flags_disjoint_intervals() {
    use ndarray::Array2;

    let mk = |a: f64, e: f64, b: f64, eb: f64| crate::margin_row::engine::PassOut {
        a: Array2::from_elem((1, 1), a),
        e: Some(Array2::from_elem((1, 1), e)),
        b: vec![b],
        eb: vec![eb],
        coll: None,
        coll_rows: None,
        unst_rows: None,
    };
    let cpu = mk(1.0, 1e-6, 0.5, 1e-7);
    assert!(super::enclosure_overlap(&cpu, &mk(1.0 + 1.5e-6, 1e-6, 0.5, 1e-7)).is_ok());
    assert_eq!(
        super::enclosure_overlap(&cpu, &mk(1.0 + 3e-6, 1e-6, 0.5, 1e-7)),
        Err("coefficient enclosure")
    );
    assert_eq!(
        super::enclosure_overlap(&cpu, &mk(1.0, 1e-6, 0.5 + 1e-6, 1e-7)),
        Err("bias enclosure")
    );
}

// ---------------------------------------------------------------------------
// Host twins of the L-lane kernels (op-for-op f32/ds mirrors; used by the
// device parity tests and validated here against their defining sequences)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
mod twin {
    use super::super::ds::{ds_abs_bound, ds_add, ds_mul, Ds, U_DS_F32};

    pub(super) fn head_gemm(
        w: &[Ds],
        seed: &[Ds],
        n_y: usize,
        n_h: usize,
        rows: usize,
        g_head: f32,
    ) -> (Vec<Ds>, Vec<f32>) {
        let mut out = Vec::with_capacity(n_h * rows);
        let mut err = Vec::with_capacity(n_h * rows);
        for idx in 0..n_h * rows {
            let k = idx / rows;
            let r = idx % rows;
            let mut acc = Ds::from_f32(0.0);
            let mut mass = 0.0f32;
            for j in 0..n_y {
                let wv = w[j * n_h + k];
                let s = seed[j * rows + r];
                acc = ds_add(acc, ds_mul(s, wv));
                mass = ds_abs_bound(wv).mul_add(ds_abs_bound(s), mass);
            }
            out.push(acc);
            err.push(g_head * mass);
        }
        (out, err)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn bias_reduce(
        plane: &[Ds],
        err: &[f32],
        scale: Option<&[Ds]>,
        n: usize,
        rows: usize,
        berr: f32,
        g_reduce: f32,
        bias_acc: &mut [Ds],
        ebias_acc: &mut [f32],
    ) {
        for r in 0..rows {
            let mut acc = Ds::from_f32(0.0);
            let mut esum = 0.0f32;
            let mut msum = 0.0f32;
            for j in 0..n {
                let p = plane[j * rows + r];
                let w = scale.map_or(Ds { hi: 1.0, lo: 0.0 }, |s| s[j]);
                let q = ds_mul(p, w);
                acc = ds_add(acc, q);
                let labs = ds_abs_bound(p);
                let e = err[j * rows + r];
                esum = ds_abs_bound(w).mul_add(e, esum);
                esum = berr.mul_add(labs + e, esum);
                msum += ds_abs_bound(q);
            }
            let nb = ds_add(bias_acc[r], acc);
            bias_acc[r] = nb;
            let carry = 2.0 * U_DS_F32 * ds_abs_bound(nb);
            ebias_acc[r] = ebias_acc[r] + esum + g_reduce * msum + carry;
        }
    }

    pub(super) fn merge_add(a: &[Ds], b: &[Ds], ea: &[f32], eb: &[f32]) -> (Vec<Ds>, Vec<f32>) {
        let mut out = Vec::with_capacity(a.len());
        let mut eout = Vec::with_capacity(a.len());
        for i in 0..a.len() {
            let s = ds_add(a[i], b[i]);
            out.push(s);
            eout.push(ea[i] + eb[i] + 2.0 * U_DS_F32 * ds_abs_bound(s));
        }
        (out, eout)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn conv_backward(
        w: &[Ds],
        coeff: &[Ds],
        err: &[f32],
        (ic, ih, iw): (usize, usize, usize),
        (oc, oh, ow): (usize, usize, usize),
        (kh, kw): (usize, usize),
        (sh, sw): (usize, usize),
        (ph, pw): (usize, usize),
        rows: usize,
        g_src: f32,
    ) -> (Vec<Ds>, Vec<f32>) {
        let n_in = ic * ih * iw;
        let mut out = Vec::with_capacity(n_in * rows);
        let mut eout = Vec::with_capacity(n_in * rows);
        for idx in 0..n_in * rows {
            let row = idx % rows;
            let pos = idx / rows;
            let ci = pos / (ih * iw);
            let iy = (pos / iw) % ih;
            let ix = pos % iw;
            let mut acc = Ds::from_f32(0.0);
            let mut errg = 0.0f32;
            for co in 0..oc {
                for khi in 0..kh {
                    let ny = iy + ph;
                    if ny < khi {
                        continue;
                    }
                    let dy = ny - khi;
                    if !dy.is_multiple_of(sh) {
                        continue;
                    }
                    let oy = dy / sh;
                    if oy >= oh {
                        continue;
                    }
                    for kwi in 0..kw {
                        let nx = ix + pw;
                        if nx < kwi {
                            continue;
                        }
                        let dx = nx - kwi;
                        if !dx.is_multiple_of(sw) {
                            continue;
                        }
                        let ox = dx / sw;
                        if ox >= ow {
                            continue;
                        }
                        let wv = w[((co * ic + ci) * kh + khi) * kw + kwi];
                        let oidx = ((co * oh + oy) * ow + ox) * rows + row;
                        let src = coeff[oidx];
                        acc = ds_add(acc, ds_mul(src, wv));
                        let e = err[oidx];
                        let e_src = g_src.mul_add(ds_abs_bound(src) + e, e);
                        errg = ds_abs_bound(wv).mul_add(e_src, errg);
                    }
                }
            }
            out.push(acc);
            eout.push(errg);
        }
        (out, eout)
    }
}

/// The head twin's value lane must satisfy the same M0 enclosure the ds dot
/// satisfies (design 4.2 head row) — a wrong twin would silently weaken the
/// device parity tests, so the twin itself is enclosed here.
#[test]
fn head_twin_value_is_enclosed_by_gamma_ds() {
    use super::ds::{gamma_ds_dot, Ds};
    let split = |v: f64| {
        #[allow(clippy::cast_possible_truncation)]
        let hi = v as f32;
        #[allow(clippy::cast_possible_truncation)]
        let lo = (v - f64::from(hi)) as f32;
        Ds { hi, lo }
    };
    let mut rng = Lcg(0xDEAD_BEEF_1234_5679);
    let (n_y, n_h, rows) = (37usize, 5usize, 3usize);
    // ds-pair weights with GENUINE lo parts (the production upload form).
    let w: Vec<Ds> = (0..n_y * n_h)
        .map(|_| split(f64::from(rng.coeff()) * (1.0 + 7.7e-10)))
        .collect();
    let seed: Vec<Ds> = (0..n_y * rows).map(|_| Ds::from_f32(rng.coeff())).collect();
    let (out, _) = twin::head_gemm(&w, &seed, n_y, n_h, rows, 0.0);
    for k in 0..n_h {
        for r in 0..rows {
            let mut exact = 0.0f64;
            let mut mass = 0.0f64;
            for j in 0..n_y {
                let t = w[j * n_h + k].to_f64() * seed[j * rows + r].to_f64();
                exact += t;
                mass += t.abs();
            }
            #[allow(clippy::cast_precision_loss)]
            let g64 = 2.0 * (n_y as f64) * f64::EPSILON;
            let diff = (out[k * rows + r].to_f64() - exact).abs();
            // The TIGHTENED per-tap envelope must already enclose the ds fold
            // (this is the bound production charges via gamma_ds_dot; the old
            // blanket gamma_ds(n) is 3.2x looser and implied by this).
            assert!(
                diff <= (gamma_ds_dot(n_y + 2) + g64) * mass,
                "head twin out of the gamma_ds_dot envelope at ({k},{r})"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Device-gated L-lane tests (real dispatches on the live adapter).
// `cargo test -p ny-propagate --features gpu-tests --lib gpu_backward`
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu-tests")]
mod lane_device_helpers {
    use super::super::ds::Ds;

    pub(super) fn ds_split(v: f64) -> Ds {
        #[allow(clippy::cast_possible_truncation)]
        let hi = v as f32;
        #[allow(clippy::cast_possible_truncation)]
        let lo = (v - f64::from(hi)) as f32;
        Ds { hi, lo }
    }

    pub(super) fn ds_bytes(vals: &[Ds]) -> Vec<u8> {
        let mut out = Vec::with_capacity(vals.len() * 8);
        for v in vals {
            out.extend_from_slice(&v.hi.to_le_bytes());
            out.extend_from_slice(&v.lo.to_le_bytes());
        }
        out
    }

    pub(super) fn f32_bytes(vals: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(vals.len() * 4);
        for v in vals {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    pub(super) fn uniform(words: &[u32], floats: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity((words.len() + floats.len()) * 4);
        for w in words {
            out.extend_from_slice(&w.to_le_bytes());
        }
        for f in floats {
            out.extend_from_slice(&f.to_le_bytes());
        }
        assert!(out.len() % 16 == 0, "test uniform must be 16-byte multiple");
        out
    }

    pub(super) fn to_ds(pairs: &[(f32, f32)]) -> Vec<Ds> {
        pairs.iter().map(|&(hi, lo)| Ds { hi, lo }).collect()
    }

    pub(super) fn assert_ds_bits(got: &[Ds], want: &[Ds], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                g.hi.to_bits() == w.hi.to_bits() && g.lo.to_bits() == w.lo.to_bits(),
                "{what}: lane {i} device=({:#010x},{:#010x}) host=({:#010x},{:#010x})",
                g.hi.to_bits(),
                g.lo.to_bits(),
                w.hi.to_bits(),
                w.lo.to_bits()
            );
        }
    }

    pub(super) fn assert_f32_bits(got: &[f32], want: &[f32], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                g.to_bits() == w.to_bits(),
                "{what}: lane {i} device={:#010x} host={:#010x}",
                g.to_bits(),
                w.to_bits()
            );
        }
    }
}

/// Per-kernel M1-grade bit parity for every NEW L-lane kernel (head gemm,
/// bias reduce incl. cross-dispatch running accumulation, residual merge,
/// conv gather) against the host ds twins, on the live adapter.
#[test]
#[cfg(feature = "gpu-tests")]
fn l_lane_kernels_are_bit_identical_on_this_adapter() {
    use super::ds::Ds;
    use super::probe::{pairs_from_bytes, scalars_from_bytes};
    use super::transaction;
    use lane_device_helpers as h;
    use ny_core::{
        LaneArenaBuffer, LaneProgramSpec, LaneStepBinding, LaneStepSpec, MarginRowLaneGpu,
    };

    let device = ny_gpu::MarginRowLaneDevice::new()
        .expect("gpu-tests requires a live wgpu adapter for the lane device");
    let mut rng = Lcg(0x1357_9BDF_2468_ACE1);
    let ro = |buffer: usize| LaneStepBinding {
        buffer,
        read_only: true,
    };
    let rw = |buffer: usize| LaneStepBinding {
        buffer,
        read_only: false,
    };

    // ---- head gemm (ds-PAIR weights, the production upload form) ----
    let (n_y, n_h, rows) = (7usize, 13usize, 5usize);
    let w: Vec<Ds> = (0..n_y * n_h)
        .map(|_| h::ds_split(f64::from(rng.coeff()) * (1.0 + 5.43e-10)))
        .collect();
    let seed: Vec<Ds> = (0..n_y * rows)
        .map(|_| h::ds_split(f64::from(rng.coeff()) * (1.0 + 9.87e-10)))
        .collect();
    let g_head = 3.0e-13f32;
    let wb = h::ds_bytes(&w);
    let sb = h::ds_bytes(&seed);
    #[allow(clippy::cast_possible_truncation)]
    let uni = h::uniform(&[n_y as u32, n_h as u32, rows as u32], &[g_head]);
    let buffers = [
        LaneArenaBuffer::Data(&wb),
        LaneArenaBuffer::Data(&sb),
        LaneArenaBuffer::Zeroed((n_h * rows * 8) as u64),
        LaneArenaBuffer::Zeroed((n_h * rows * 4) as u64),
    ];
    let bindings = [ro(0), ro(1), rw(2), rw(3)];
    let step = LaneStepSpec {
        label: "mr_gpu_backward_head_gemm",
        entry: "head_gemm",
        shader_wgsl: transaction::shader_head(),
        uniforms: &uni,
        bindings: &bindings,
        workgroups: [1, 1, 1],
    };
    let out = device
        .run_lane_program(&LaneProgramSpec {
            label: "lane_test_head",
            buffers: &buffers,
            steps: std::slice::from_ref(&step),
            readbacks: &[2, 3],
            profile: false,
        })
        .expect("head program runs");
    let (want, want_err) = twin::head_gemm(&w, &seed, n_y, n_h, rows, g_head);
    h::assert_ds_bits(
        &h::to_ds(&pairs_from_bytes(&out[0]).unwrap()),
        &want,
        "head coeff",
    );
    h::assert_f32_bits(&scalars_from_bytes(&out[1]), &want_err, "head err");
    println!(
        "[gpu-backward gpu-test] head_gemm: {} lanes bit-identical",
        2 * n_h * rows
    );

    // ---- bias reduce: two dispatches, ds-pair scale then no-scale ----
    let (n, rows) = (9usize, 4usize);
    let plane: Vec<Ds> = (0..n * rows)
        .map(|_| h::ds_split(f64::from(rng.coeff()) * (1.0 + 3.21e-10)))
        .collect();
    let err: Vec<f32> = (0..n * rows).map(|_| rng.coeff().abs() * 1e-6).collect();
    let scale: Vec<Ds> = (0..n)
        .map(|_| h::ds_split(f64::from(rng.coeff()) * (1.0 + 6.54e-10)))
        .collect();
    let (berr, g_red1, g_red2) = (1.0e-8f32, 2.0e-12f32, 5.0e-13f32);
    let pb = h::ds_bytes(&plane);
    let eb_ = h::f32_bytes(&err);
    let scb = h::ds_bytes(&scale);
    #[allow(clippy::cast_possible_truncation)]
    let uni1 = h::uniform(&[n as u32, rows as u32, 1, 0], &[berr, g_red1, 0.0, 0.0]);
    #[allow(clippy::cast_possible_truncation)]
    let uni2 = h::uniform(&[n as u32, rows as u32, 0, 0], &[0.0, g_red2, 0.0, 0.0]);
    let buffers = [
        LaneArenaBuffer::Data(&pb),
        LaneArenaBuffer::Data(&eb_),
        LaneArenaBuffer::Data(&scb),
        LaneArenaBuffer::Zeroed((rows * 8) as u64),
        LaneArenaBuffer::Zeroed((rows * 4) as u64),
    ];
    let bindings = [ro(0), ro(1), ro(2), rw(3), rw(4)];
    let steps = [
        LaneStepSpec {
            label: "mr_gpu_backward_bias_reduce",
            entry: "bias_reduce",
            shader_wgsl: transaction::shader_reduce(),
            uniforms: &uni1,
            bindings: &bindings,
            workgroups: [1, 1, 1],
        },
        LaneStepSpec {
            label: "mr_gpu_backward_bias_reduce",
            entry: "bias_reduce",
            shader_wgsl: transaction::shader_reduce(),
            uniforms: &uni2,
            bindings: &bindings,
            workgroups: [1, 1, 1],
        },
    ];
    let out = device
        .run_lane_program(&LaneProgramSpec {
            label: "lane_test_reduce",
            buffers: &buffers,
            steps: &steps,
            readbacks: &[3, 4],
            profile: false,
        })
        .expect("reduce program runs");
    let mut want_b = vec![Ds { hi: 0.0, lo: 0.0 }; rows];
    let mut want_eb = vec![0.0f32; rows];
    twin::bias_reduce(
        &plane,
        &err,
        Some(&scale),
        n,
        rows,
        berr,
        g_red1,
        &mut want_b,
        &mut want_eb,
    );
    twin::bias_reduce(
        &plane,
        &err,
        None,
        n,
        rows,
        0.0,
        g_red2,
        &mut want_b,
        &mut want_eb,
    );
    h::assert_ds_bits(
        &h::to_ds(&pairs_from_bytes(&out[0]).unwrap()),
        &want_b,
        "reduce bias_acc",
    );
    h::assert_f32_bits(&scalars_from_bytes(&out[1]), &want_eb, "reduce ebias_acc");
    println!(
        "[gpu-backward gpu-test] bias_reduce: {} rows bit-identical across 2 dispatches",
        rows
    );

    // ---- B2 tree reduce: device `bias_reduce_wg` vs the TREE host twin,
    // across empty-lane (n < 256) and multi-element-lane (n > 256) shapes,
    // with cross-dispatch running accumulation ----
    for n in [9usize, 300] {
        let rows = 4usize;
        let plane: Vec<Ds> = (0..n * rows)
            .map(|_| h::ds_split(f64::from(rng.coeff()) * (1.0 + 3.21e-10)))
            .collect();
        let err: Vec<f32> = (0..n * rows).map(|_| rng.coeff().abs() * 1e-6).collect();
        let scale: Vec<Ds> = (0..n)
            .map(|_| h::ds_split(f64::from(rng.coeff()) * (1.0 + 6.54e-10)))
            .collect();
        let (berr, g_red1, g_red2) = (1.0e-8f32, 2.0e-12f32, 5.0e-13f32);
        let pb = h::ds_bytes(&plane);
        let eb_ = h::f32_bytes(&err);
        let scb = h::ds_bytes(&scale);
        #[allow(clippy::cast_possible_truncation)]
        let uni1 = h::uniform(&[n as u32, rows as u32, 1, 0], &[berr, g_red1, 0.0, 0.0]);
        #[allow(clippy::cast_possible_truncation)]
        let uni2 = h::uniform(&[n as u32, rows as u32, 0, 0], &[0.0, g_red2, 0.0, 0.0]);
        let buffers = [
            LaneArenaBuffer::Data(&pb),
            LaneArenaBuffer::Data(&eb_),
            LaneArenaBuffer::Data(&scb),
            LaneArenaBuffer::Zeroed((rows * 8) as u64),
            LaneArenaBuffer::Zeroed((rows * 4) as u64),
        ];
        let bindings = [ro(0), ro(1), ro(2), rw(3), rw(4)];
        #[allow(clippy::cast_possible_truncation)]
        let wg = [rows as u32, 1, 1]; // one 256-lane workgroup per row
        let steps = [
            LaneStepSpec {
                label: "mr_gpu_backward_bias_reduce_wg",
                entry: "bias_reduce_wg",
                shader_wgsl: transaction::shader_reduce(),
                uniforms: &uni1,
                bindings: &bindings,
                workgroups: wg,
            },
            LaneStepSpec {
                label: "mr_gpu_backward_bias_reduce_wg",
                entry: "bias_reduce_wg",
                shader_wgsl: transaction::shader_reduce(),
                uniforms: &uni2,
                bindings: &bindings,
                workgroups: wg,
            },
        ];
        let out = device
            .run_lane_program(&LaneProgramSpec {
                label: "lane_test_reduce_wg",
                buffers: &buffers,
                steps: &steps,
                readbacks: &[3, 4],
                profile: false,
            })
            .expect("tree reduce program runs");
        let mut want_b = vec![Ds { hi: 0.0, lo: 0.0 }; rows];
        let mut want_eb = vec![0.0f32; rows];
        super::ds::bias_reduce_wg_twin(
            &plane,
            &err,
            Some(&scale),
            n,
            rows,
            berr,
            g_red1,
            &mut want_b,
            &mut want_eb,
        );
        super::ds::bias_reduce_wg_twin(
            &plane,
            &err,
            None,
            n,
            rows,
            0.0,
            g_red2,
            &mut want_b,
            &mut want_eb,
        );
        h::assert_ds_bits(
            &h::to_ds(&pairs_from_bytes(&out[0]).unwrap()),
            &want_b,
            "tree reduce bias_acc",
        );
        h::assert_f32_bits(
            &scalars_from_bytes(&out[1]),
            &want_eb,
            "tree reduce ebias_acc",
        );
        println!(
            "[gpu-backward gpu-test] bias_reduce_wg: n={n} {rows} rows bit-identical \
             across 2 dispatches (tree twin)"
        );
    }

    // ---- residual merge ----
    let total = 64usize;
    let a: Vec<Ds> = (0..total)
        .map(|_| h::ds_split(f64::from(rng.coeff()) * (1.0 + 1.11e-10)))
        .collect();
    let b: Vec<Ds> = (0..total)
        .map(|_| h::ds_split(f64::from(rng.coeff()) * (1.0 + 2.22e-10)))
        .collect();
    let ea: Vec<f32> = (0..total).map(|_| rng.coeff().abs() * 1e-7).collect();
    let eb2: Vec<f32> = (0..total).map(|_| rng.coeff().abs() * 1e-7).collect();
    let (ab, bb, eab, ebb) = (
        h::ds_bytes(&a),
        h::ds_bytes(&b),
        h::f32_bytes(&ea),
        h::f32_bytes(&eb2),
    );
    #[allow(clippy::cast_possible_truncation)]
    let uni = h::uniform(&[total as u32, 0, 0, 0], &[]);
    let buffers = [
        LaneArenaBuffer::Data(&ab),
        LaneArenaBuffer::Data(&bb),
        LaneArenaBuffer::Zeroed((total * 8) as u64),
        LaneArenaBuffer::Data(&eab),
        LaneArenaBuffer::Data(&ebb),
        LaneArenaBuffer::Zeroed((total * 4) as u64),
    ];
    let bindings = [ro(0), ro(1), rw(2), ro(3), ro(4), rw(5)];
    let step = LaneStepSpec {
        label: "mr_gpu_backward_merge_add",
        entry: "merge_add",
        shader_wgsl: transaction::shader_merge(),
        uniforms: &uni,
        bindings: &bindings,
        workgroups: [1, 1, 1],
    };
    let out = device
        .run_lane_program(&LaneProgramSpec {
            label: "lane_test_merge",
            buffers: &buffers,
            steps: std::slice::from_ref(&step),
            readbacks: &[2, 5],
            profile: false,
        })
        .expect("merge program runs");
    let (want, want_err) = twin::merge_add(&a, &b, &ea, &eb2);
    h::assert_ds_bits(
        &h::to_ds(&pairs_from_bytes(&out[0]).unwrap()),
        &want,
        "merge coeff",
    );
    h::assert_f32_bits(&scalars_from_bytes(&out[1]), &want_err, "merge err");
    println!("[gpu-backward gpu-test] merge_add: {total} lanes bit-identical");

    // ---- conv backward gather (ds-PAIR weights) ----
    let (ic, ih, iw) = (2usize, 4usize, 4usize);
    let (oc, oh, ow) = (4usize, 4usize, 4usize);
    let (kh, kw, sh, sw, ph, pw) = (3usize, 3usize, 1usize, 1usize, 1usize, 1usize);
    let rows = 3usize;
    let n_out = oc * oh * ow;
    let n_in = ic * ih * iw;
    let w: Vec<Ds> = (0..oc * ic * kh * kw)
        .map(|_| h::ds_split(f64::from(rng.coeff()) * (1.0 + 8.76e-10)))
        .collect();
    let coeff: Vec<Ds> = (0..n_out * rows)
        .map(|_| h::ds_split(f64::from(rng.coeff()) * (1.0 + 4.44e-10)))
        .collect();
    let err: Vec<f32> = (0..n_out * rows)
        .map(|_| rng.coeff().abs() * 1e-6)
        .collect();
    let g_src = 2.5e-13f32;
    let (wb, cb, ebts) = (h::ds_bytes(&w), h::ds_bytes(&coeff), h::f32_bytes(&err));
    #[allow(clippy::cast_possible_truncation)]
    let uni = h::uniform(
        &[
            ic as u32,
            ih as u32,
            iw as u32,
            oc as u32,
            oh as u32,
            ow as u32,
            kh as u32,
            kw as u32,
            sh as u32,
            sw as u32,
            ph as u32,
            pw as u32,
            rows as u32,
            0,
            0,
        ],
        &[g_src],
    );
    let buffers = [
        LaneArenaBuffer::Data(&wb),
        LaneArenaBuffer::Data(&cb),
        LaneArenaBuffer::Zeroed((n_in * rows * 8) as u64),
        LaneArenaBuffer::Data(&ebts),
        LaneArenaBuffer::Zeroed((n_in * rows * 4) as u64),
        LaneArenaBuffer::Zeroed((n_in * rows * 4) as u64),
    ];
    let bindings = [ro(0), ro(1), rw(2), ro(3), rw(4), rw(5)];
    let step = LaneStepSpec {
        label: "mr_gpu_backward_conv_backward",
        entry: "conv_backward",
        shader_wgsl: transaction::shader_conv(),
        uniforms: &uni,
        bindings: &bindings,
        workgroups: [1, 1, 1],
    };
    let out = device
        .run_lane_program(&LaneProgramSpec {
            label: "lane_test_conv",
            buffers: &buffers,
            steps: std::slice::from_ref(&step),
            readbacks: &[2, 4, 5],
            profile: false,
        })
        .expect("conv program runs");
    let (want, want_err) = twin::conv_backward(
        &w,
        &coeff,
        &err,
        (ic, ih, iw),
        (oc, oh, ow),
        (kh, kw),
        (sh, sw),
        (ph, pw),
        rows,
        g_src,
    );
    h::assert_ds_bits(
        &h::to_ds(&pairs_from_bytes(&out[0]).unwrap()),
        &want,
        "conv coeff",
    );
    h::assert_f32_bits(&scalars_from_bytes(&out[1]), &want_err, "conv err");
    // The reserved refinement channel is zeroed, not stale.
    assert!(
        scalars_from_bytes(&out[2]).iter().all(|v| *v == 0.0),
        "conv residual channel must be zeroed"
    );
    println!(
        "[gpu-backward gpu-test] conv_backward: {} lanes bit-identical (k_bwd taps = {})",
        2 * n_in * rows,
        oc * kh * kw
    );

    // ---- B3 staging stress: a slice larger than one staging chunk
    // (oc*khw = 1152 > CONV_STAGE_TAPS-per-113-co chunk => 2 co-chunks) on a
    // workgroup that STRADDLES the ci boundary (total = 96 < 256 threads,
    // both channels in one workgroup => staged AND global fallback paths
    // both exercised) — device must stay bit-identical to the UNCHANGED
    // host twin (per-element order preserved; memory access only) ----
    {
        let (ic, ih, iw) = (2usize, 4usize, 4usize);
        let (oc, oh, ow) = (128usize, 4usize, 4usize);
        let (kh, kw, sh, sw, ph, pw) = (3usize, 3usize, 1usize, 1usize, 1usize, 1usize);
        let rows = 3usize;
        let n_out = oc * oh * ow;
        let n_in = ic * ih * iw;
        let w: Vec<Ds> = (0..oc * ic * kh * kw)
            .map(|_| h::ds_split(f64::from(rng.coeff()) * (1.0 + 5.55e-10)))
            .collect();
        let coeff: Vec<Ds> = (0..n_out * rows)
            .map(|_| h::ds_split(f64::from(rng.coeff()) * (1.0 + 6.66e-10)))
            .collect();
        let err: Vec<f32> = (0..n_out * rows)
            .map(|_| rng.coeff().abs() * 1e-6)
            .collect();
        let g_src = 1.5e-13f32;
        let (wb, cb, ebts) = (h::ds_bytes(&w), h::ds_bytes(&coeff), h::f32_bytes(&err));
        #[allow(clippy::cast_possible_truncation)]
        let uni = h::uniform(
            &[
                ic as u32,
                ih as u32,
                iw as u32,
                oc as u32,
                oh as u32,
                ow as u32,
                kh as u32,
                kw as u32,
                sh as u32,
                sw as u32,
                ph as u32,
                pw as u32,
                rows as u32,
                0,
                0,
            ],
            &[g_src],
        );
        let buffers = [
            LaneArenaBuffer::Data(&wb),
            LaneArenaBuffer::Data(&cb),
            LaneArenaBuffer::Zeroed((n_in * rows * 8) as u64),
            LaneArenaBuffer::Data(&ebts),
            LaneArenaBuffer::Zeroed((n_in * rows * 4) as u64),
            LaneArenaBuffer::Zeroed((n_in * rows * 4) as u64),
        ];
        let bindings = [ro(0), ro(1), rw(2), ro(3), rw(4), rw(5)];
        let step = LaneStepSpec {
            label: "mr_gpu_backward_conv_backward",
            entry: "conv_backward",
            shader_wgsl: transaction::shader_conv(),
            uniforms: &uni,
            bindings: &bindings,
            workgroups: [1, 1, 1],
        };
        let out = device
            .run_lane_program(&LaneProgramSpec {
                label: "lane_test_conv_staged",
                buffers: &buffers,
                steps: std::slice::from_ref(&step),
                readbacks: &[2, 4],
                profile: false,
            })
            .expect("staged conv program runs");
        let (want, want_err) = twin::conv_backward(
            &w,
            &coeff,
            &err,
            (ic, ih, iw),
            (oc, oh, ow),
            (kh, kw),
            (sh, sw),
            (ph, pw),
            rows,
            g_src,
        );
        h::assert_ds_bits(
            &h::to_ds(&pairs_from_bytes(&out[0]).unwrap()),
            &want,
            "staged conv coeff",
        );
        h::assert_f32_bits(&scalars_from_bytes(&out[1]), &want_err, "staged conv err");
        println!(
            "[gpu-backward gpu-test] conv_backward (B3 staged, 2 co-chunks, mixed-ci \
             workgroup): {} lanes bit-identical (k_bwd taps = {})",
            2 * n_in * rows,
            oc * kh * kw
        );
    }
}

/// The M2 shape in miniature, on the live adapter: the WHOLE L-lane
/// transaction (plan reuse -> program -> one submit -> unconditional drain ->
/// directed egress -> floor + probe guards) against the CPU pass, asserting
/// the enclosure-overlap invariant on the chain AND residual fixtures, both
/// directions, root gates AND a piece-fixed domain, identity AND margin-style
/// seeds.
#[test]
#[cfg(feature = "gpu-tests")]
fn l_lane_transaction_encloses_the_cpu_pass_on_the_fixtures() {
    use super::super::gpu_seam::tests as seam_tests;
    use super::authority::{DeviceParityProof, VerdictAuthority};
    use super::probe::self_check_streams;
    use super::transaction;
    use crate::margin_row::engine::{domain_gates, BackwardEngine, LaneDir, Seed};
    use ny_core::MarginRowLaneGpu;

    // `Arc<dyn ..>` because `transaction::execute` takes the executor as the
    // Arc the production path holds (FIX D3: a `PlanStatic`'s resident planes
    // keep that handle so they can RELEASE their device buffers on drop).
    let device: std::sync::Arc<dyn MarginRowLaneGpu> = std::sync::Arc::new(
        ny_gpu::MarginRowLaneDevice::new()
            .expect("gpu-tests requires a live wgpu adapter for the lane device"),
    );
    assert!(
        device.denorm_preserve_enabled(),
        "this adapter resolves denorm-preserve OFF; the lane refuses by design"
    );
    let streams = self_check_streams(device.as_ref()).expect("M1 probes must run");
    let proof = DeviceParityProof::qualify(true, &streams.host, &streams.device)
        .expect("M1 parity must hold before any transaction");
    let auth = VerdictAuthority::grant(proof);
    println!(
        "[gpu-backward gpu-test] M1 qualification: {} lanes bit-identical on {}",
        auth.parity_lanes(),
        device.adapter_summary()
    );

    for (name, spec, radius_pin) in [
        ("chain", seam_tests::chain_spec(), 2.0e-11),
        ("residual", seam_tests::residual_spec(), 3.0e-11),
    ] {
        let (net, root, _, _) = seam_tests::compile(&spec);
        let eng = BackwardEngine::new(&net, &root);
        let seed = eng.identity_seed();
        // B1: one residency cache per engine (the shadow entry's shape);
        // pass 1 per direction MISSES (build + resident upload), pass 2
        // HITS — and must reproduce the pass BIT-identically (values AND
        // radii): pure caching, same ops, same order.
        let plan_cache = super::cache::PlanCache::default();
        for dir in [LaneDir::Lower, LaneDir::Upper] {
            let cpu = eng.run(&seed, None, dir, None, false).expect("cpu pass");
            let started = std::time::Instant::now();
            let mut stats = transaction::ExecStats::default();
            let gpu_pass = transaction::execute(
                &auth,
                &device,
                &eng,
                &seed,
                None,
                dir,
                PassShape::default(),
                &super::LaneCtx::default(),
                Some(&plan_cache),
                false,
                &mut stats,
            )
            .unwrap_or_else(|r| panic!("{name} {dir:?}: device pass refused: {r:?}"));
            let device_ms = started.elapsed().as_secs_f64() * 1e3;
            // First direction misses (cold cache), second hits.
            assert!(
                stats.plan_cache_hit.is_some(),
                "{name} {dir:?}: cache-offered pass must report hit/miss"
            );
            let mut stats2 = transaction::ExecStats::default();
            let gpu_pass2 = transaction::execute(
                &auth,
                &device,
                &eng,
                &seed,
                None,
                dir,
                PassShape::default(),
                &super::LaneCtx::default(),
                Some(&plan_cache),
                false,
                &mut stats2,
            )
            .unwrap_or_else(|r| panic!("{name} {dir:?}: cached device pass refused: {r:?}"));
            assert_eq!(
                stats2.plan_cache_hit,
                Some(true),
                "{name} {dir:?}: second pass must HIT the residency cache"
            );
            assert_eq!(
                gpu_pass.a, gpu_pass2.a,
                "{name} {dir:?}: hit pass values must be bit-identical"
            );
            assert_eq!(
                gpu_pass.b, gpu_pass2.b,
                "{name} {dir:?}: hit pass bias must be bit-identical"
            );
            assert_eq!(
                gpu_pass
                    .e
                    .as_ref()
                    .map(|e| e.iter().map(|v| v.to_bits()).collect::<Vec<_>>()),
                gpu_pass2
                    .e
                    .as_ref()
                    .map(|e| e.iter().map(|v| v.to_bits()).collect::<Vec<_>>()),
                "{name} {dir:?}: hit pass error lane must be bit-identical"
            );
            assert_eq!(
                gpu_pass.eb.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                gpu_pass2.eb.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "{name} {dir:?}: hit pass bias error must be bit-identical"
            );
            super::enclosure_overlap(&cpu, &gpu_pass)
                .unwrap_or_else(|plane| panic!("{name} {dir:?}: M2 enclosure trip on {plane}"));
            let xabs = root.xabs.as_slice();
            let gpu_radius = super::radius_median(&gpu_pass, xabs);
            println!(
                "[gpu-backward gpu-test] {name} {dir:?}: enclosure OK; median_radius \
                 gpu={gpu_radius:.3e} cpu={:.3e} device_ms={device_ms:.1} plan_cache={:?} \
                 build_ms={:.1}/{:.1} upload_ms={:.1}/{:.1} dispatch_ms={:.1}/{:.1}",
                super::radius_median(&cpu, xabs),
                stats.plan_cache_hit,
                stats.build_ms,
                stats2.build_ms,
                stats.upload_ms,
                stats2.upload_ms,
                stats.dispatch_ms,
                stats2.dispatch_ms,
            );
            // RADIUS-REGRESSION PIN (error-charge session, 2026-08-20): the
            // tightened charges measure 5.66e-12 (chain) / 8.07e-12
            // (residual) on this adapter — ~5.9e5x below the pre-session
            // 3.3e-6/6.0e-6, within ~10x of the CPU pass's 6.1e-13/7.0e-13.
            // The pins sit ~3.5x above the measured values: any change that
            // re-introduces a u32-grade ball, a blanket gamma, or a double
            // charge blows through them by orders of magnitude, while
            // benign driver/toolchain jitter cannot (the value path is
            // bit-pinned by the parity tests anyway).
            assert!(
                gpu_radius <= radius_pin,
                "{name} {dir:?}: certified radius regressed: {gpu_radius:.3e} > pin {radius_pin:.1e}"
            );
        }
        // A piece-fixed domain: the ds gate pairs must retarget with it.
        if let Some(li) = root.layers.iter().position(|l| !l.unst.is_empty()) {
            let dom = domain_gates(&root, &[(li, 0, 1)]);
            if !dom.infeasible {
                let cpu = eng
                    .run(&seed, Some(&dom), LaneDir::Lower, None, false)
                    .expect("cpu dom pass");
                // A piece-fixed dom on the WARM cache: the hit path's gate
                // admission + retargeted gate quads + the cached probe.
                let mut dom_stats = transaction::ExecStats::default();
                let gpu_pass = transaction::execute(
                    &auth,
                    &device,
                    &eng,
                    &seed,
                    Some(&dom),
                    LaneDir::Lower,
                    PassShape::default(),
                    &super::LaneCtx::default(),
                    Some(&plan_cache),
                    false,
                    &mut dom_stats,
                )
                .unwrap_or_else(|r| panic!("{name} dom: device pass refused: {r:?}"));
                assert_eq!(
                    dom_stats.plan_cache_hit,
                    Some(true),
                    "{name} dom: the dom pass must HIT the dom-independent cache"
                );
                super::enclosure_overlap(&cpu, &gpu_pass)
                    .unwrap_or_else(|plane| panic!("{name} dom: M2 enclosure trip on {plane}"));
                println!("[gpu-backward gpu-test] {name} piece-fixed domain: enclosure OK");
            }
        }
        // An f32-exact margin-style multi-column seed (the production shape).
        let rows = 4usize;
        let mut s = ndarray::Array2::<f64>::zeros((net.n_y, rows));
        for j in 0..net.n_y {
            for r in 0..rows {
                #[allow(clippy::cast_possible_wrap)]
                let v = ((j * 7 + r * 3) % 5) as i64 - 2;
                #[allow(clippy::cast_precision_loss)]
                {
                    s[[j, r]] = v as f64 * 0.25;
                }
            }
        }
        let seed2 = Seed { s, e: None };
        assert!(super::seed_is_device_exact(&seed2));
        let cpu = eng
            .run(&seed2, None, LaneDir::Upper, None, false)
            .expect("cpu margin pass");
        let gpu_pass = transaction::execute(
            &auth,
            &device,
            &eng,
            &seed2,
            None,
            LaneDir::Upper,
            PassShape::default(),
            &super::LaneCtx::default(),
            None,
            false,
            &mut transaction::ExecStats::default(),
        )
        .unwrap_or_else(|r| panic!("{name} margin seed: device pass refused: {r:?}"));
        super::enclosure_overlap(&cpu, &gpu_pass)
            .unwrap_or_else(|plane| panic!("{name} margin seed: M2 enclosure trip on {plane}"));
        println!("[gpu-backward gpu-test] {name} margin-style seed: enclosure OK");
        // A non-f32-exact seed: needs y_abs, admits, passes its probe guard;
        // the coefficient overlap is deliberately NOT asserted (the seed
        // residual is charged into eb at functional level — see
        // seed_is_device_exact).
        let mut s3 = ndarray::Array2::<f64>::zeros((net.n_y, 2));
        for j in 0..net.n_y {
            #[allow(clippy::cast_precision_loss)]
            {
                s3[[j, 0]] = 0.1 + 0.01 * j as f64;
                s3[[j, 1]] = -0.3 + 0.02 * j as f64;
            }
        }
        let seed3 = Seed { s: s3, e: None };
        assert!(!super::seed_is_device_exact(&seed3));
        let y_abs = vec![64.0f64; net.n_y];
        let ctx = super::LaneCtx {
            y_abs: Some(&y_abs),
            deadline: None,
            plan_cache: None,
        };
        transaction::execute(
            &auth,
            &device,
            &eng,
            &seed3,
            None,
            LaneDir::Lower,
            PassShape::default(),
            &ctx,
            None,
            false,
            &mut transaction::ExecStats::default(),
        )
        .unwrap_or_else(|r| panic!("{name} inexact seed: device pass refused: {r:?}"));
        println!("[gpu-backward gpu-test] {name} inexact seed: admitted, probe guard passed");
    }
}

// ===========================================================================
// B4 / B5 / B6 — SITE GATING, FIX A1, FIX A6, THE COLLECT TWIN
// (throughput session, impl-v8)
// ===========================================================================

use super::{
    authority_sites_label, collect_shape_admits, effective_site_mask, site_mode_from, PassShape,
    SiteMode, SITE_ALL, SITE_EVAL, SITE_ROOT, SITE_SCORE, SITE_Y_REFRESH,
};

/// SITE PIN 1: every site declares a distinct bit and `SITE_ALL` is their
/// union — a duplicated bit would silently couple two flips.
#[test]
fn site_bits_are_distinct_and_complete() {
    let all = [SITE_ROOT, SITE_Y_REFRESH, SITE_EVAL, SITE_SCORE];
    let mut seen = 0u32;
    for bit in all {
        assert!(
            bit.is_power_of_two(),
            "site bit {bit:#x} is not a single bit"
        );
        assert_eq!(seen & bit, 0, "duplicate site bit {bit:#x}");
        seen |= bit;
    }
    assert_eq!(
        seen, SITE_ALL,
        "SITE_ALL must be exactly the declared sites"
    );
}

/// The engagement banner must describe the exact authority mask. The former
/// constant "ShadowOnly" text would have kept saying the opposite after an M3
/// flip, making an authorized run indistinguishable from the wrong lane.
#[test]
fn authority_telemetry_tracks_every_reviewed_site_flip() {
    assert_eq!(authority_sites_label(0, false), "none");
    assert_eq!(authority_sites_label(0, true), "root");
    assert_eq!(
        authority_sites_label(SITE_EVAL | SITE_SCORE, false),
        "eval,score"
    );
    assert_eq!(
        authority_sites_label(SITE_Y_REFRESH | SITE_SCORE, true),
        "root,y_refresh,score"
    );
}

/// SITE PIN 2: on THIS build every NEW site (B4/B5/B6) is SHADOW when the
/// lane is armed with the shadow lever, and OFF otherwise — never
/// Authorized. Flipping one is a deliberate, human-reviewed const edit after
/// that site's own x3 moat, and this pin makes an accidental flip a test
/// failure (the `verdict_authority_is_m3_gated_off` discipline, per site).
#[test]
fn new_sites_default_to_shadow_on_this_build() {
    assert_eq!(
        super::SITE_AUTHORITY_GRANTED,
        0,
        "a new site was granted verdict authority: only its own moat session may do this"
    );
    let mask = effective_site_mask(super::SITE_AUTHORITY_GRANTED, super::M3_AUTHORITY_GRANTED);
    for site in [SITE_Y_REFRESH, SITE_EVAL, SITE_SCORE] {
        assert_eq!(
            site_mode_from(true, true, SITE_ALL, mask, site),
            SiteMode::Shadow,
            "site {site:#x} must be SHADOW when armed+shadowed"
        );
        assert_eq!(
            site_mode_from(true, false, SITE_ALL, mask, site),
            SiteMode::Off,
            "site {site:#x} must be OFF when armed without the shadow lever"
        );
        // ARMING: the shadow lever alone arms the shadow for the SELECTED
        // sites (`shadow_compare` checked `shadow_enabled()` and nothing
        // else pre-v8; v9 adds the per-site selector, whose UNSET default is
        // root-only — pinned separately in
        // `shadow_site_selector_defaults_to_the_impl_v7_root_only_shape`).
        // `SITE_ALL` here is "the selector names this site".
        assert_eq!(
            site_mode_from(false, true, SITE_ALL, mask, site),
            SiteMode::Shadow,
            "site {site:#x} must SHADOW on the shadow lever alone"
        );
        // ...and it must NOT shadow when the selector does not name it: that
        // is the whole point of D5a (one site per measurement cycle).
        assert_eq!(
            site_mode_from(false, true, SITE_ALL & !site, mask, site),
            SiteMode::Off,
            "site {site:#x} must be OFF when the selector does not name it"
        );
        assert_eq!(
            site_mode_from(false, false, SITE_ALL, mask, site),
            SiteMode::Off,
            "site {site:#x} must be OFF with no lever at all"
        );
    }
}

/// SITE PIN 3: each site flips INDEPENDENTLY — granting one leaves the other
/// three exactly where they were, in both directions.
#[test]
fn each_site_flips_independently() {
    for granted in [SITE_Y_REFRESH, SITE_EVAL, SITE_SCORE] {
        let mask = effective_site_mask(granted, false);
        assert_eq!(
            site_mode_from(true, true, SITE_ALL, mask, granted),
            SiteMode::Authorized,
            "the granted site must be Authorized"
        );
        for other in [SITE_ROOT, SITE_Y_REFRESH, SITE_EVAL, SITE_SCORE] {
            if other == granted {
                continue;
            }
            assert_ne!(
                site_mode_from(true, true, SITE_ALL, mask, other),
                SiteMode::Authorized,
                "granting {granted:#x} must not authorize {other:#x}"
            );
        }
        // An authorized site is authorized whether or not the shadow lever
        // is set (it no longer needs the shadow to reach the device), and it
        // does NOT also shadow — that would double the device cost and
        // compare the pass against itself.
        assert_eq!(
            site_mode_from(true, false, SITE_ALL, mask, granted),
            SiteMode::Authorized
        );
        assert_eq!(
            site_mode_from(true, true, SITE_ALL, mask, granted),
            SiteMode::Authorized
        );
        // But VERDICT authority always needs the lane's own env gate too: no
        // single untrusted env var can promote a site.
        assert_eq!(
            site_mode_from(false, false, SITE_ALL, mask, granted),
            SiteMode::Off,
            "the site bit alone must not authorize anything"
        );
        assert_eq!(
            site_mode_from(false, true, SITE_ALL, mask, granted),
            SiteMode::Shadow,
            "without the lane gate a granted site still only shadows"
        );
    }
}

/// SITE PIN 4 (the M3 independence the design demands): `M3_AUTHORITY_GRANTED`
/// governs the ROOT site ALONE. Flipping it authorizes nothing new, and no
/// bitmask value can authorize root while it is false.
#[test]
fn m3_flip_does_not_authorize_the_new_sites() {
    // M3 true, bitmask empty: root Authorized, everything else Shadow.
    let m3_only = effective_site_mask(0, true);
    assert_eq!(
        site_mode_from(true, true, SITE_ALL, m3_only, SITE_ROOT),
        SiteMode::Authorized
    );
    for other in [SITE_Y_REFRESH, SITE_EVAL, SITE_SCORE] {
        assert_eq!(
            site_mode_from(true, true, SITE_ALL, m3_only, other),
            SiteMode::Shadow,
            "an M3 flip must not sweep site {other:#x} along"
        );
    }
    // And the converse: a bitmask that (wrongly) sets the root bit cannot
    // authorize root while M3 is false — the mask masks it out.
    let root_forged = effective_site_mask(SITE_ALL, false);
    assert_eq!(
        site_mode_from(true, true, SITE_ALL, root_forged, SITE_ROOT),
        SiteMode::Shadow,
        "only M3_AUTHORITY_GRANTED may authorize the root site"
    );
}

/// B5 ADMISSION PIN: only the exact heuristic-collect shape enters the seam.
/// tier-0 row capture and the #alpha-opt `unst_rows` capture have no device
/// readback and must refuse to the CPU pass.
#[test]
fn collect_seam_admits_only_the_heuristic_shape() {
    use crate::margin_row::engine::Collect;
    use crate::margin_row::root::RetainedRows;

    assert!(collect_shape_admits(Collect {
        unst_abs: true,
        rows: None,
        unst_rows: false,
    }));
    assert!(!collect_shape_admits(Collect {
        unst_abs: false,
        rows: None,
        unst_rows: false,
    }));
    assert!(!collect_shape_admits(Collect {
        unst_abs: true,
        rows: None,
        unst_rows: true,
    }));
    let retained = RetainedRows { layers: Vec::new() };
    assert!(!collect_shape_admits(Collect {
        unst_abs: true,
        rows: Some(&retained),
        unst_rows: false,
    }));
    assert!(!collect_shape_admits(Collect::default()));
}

/// FIX A5 PIN (exact, not off-by-one): `width * R <= 65535 * 256`.
#[test]
fn dispatch_thread_cap_is_the_stated_b4_cap() {
    use super::transaction::{dispatch_thread_cap, MAX_DISPATCH_THREADS};

    assert_eq!(MAX_DISPATCH_THREADS, 65_535 * 256);
    assert_eq!(MAX_DISPATCH_THREADS, 16_776_960);
    // The design's worked example, corrected: at width 65536, R = 255 fits
    // and R = 256 is ONE workgroup over.
    assert!(dispatch_thread_cap(65_536, 255).is_ok());
    assert!(dispatch_thread_cap(65_536, 256).is_err());
    // The shipped B4 shape (k=16/16 => R = 64) fits at any realistic width.
    assert!(dispatch_thread_cap(200_000, 64).is_ok());
    assert!(dispatch_thread_cap(300_000, 64).is_err());
    // Fail-closed on the degenerate and the overflow-prone inputs.
    assert!(dispatch_thread_cap(0, 64).is_err());
    assert!(dispatch_thread_cap(usize::MAX, usize::MAX).is_err());
}

/// A chain fixture whose every relu gate is `alpha = s = 0.5, c = 0` — the
/// FIX A1 counterexample shape: a layer whose BASE slope mass is 0.5, so an
/// exception of slope 1.0 has true coefficients the base charge does not
/// bound.
fn half_slope_fixture() -> (
    crate::margin_row::net::TwinNet,
    crate::margin_row::root::RootGates,
) {
    let (net, mut root) = chain_fixture();
    for rec in &mut root.layers {
        for j in 0..rec.n {
            rec.alpha[j] = 0.5;
            rec.s[j] = 0.5;
            rec.c[j] = 0.0;
            rec.ms[j] = 0.5;
        }
    }
    (net, root)
}

/// `build_program` with an explicit lifetime tie (a closure cannot express
/// "the shape borrows for as long as the program does").
#[allow(clippy::too_many_arguments)]
fn build_with_shape<'p>(
    statics: &'p super::cache::PlanStatic,
    eng: &'p crate::margin_row::engine::BackwardEngine<'_>,
    seed: &crate::margin_row::engine::Seed,
    dir: crate::margin_row::engine::LaneDir,
    shape: PassShape<'p>,
    n_y: usize,
    n_in: usize,
    rows: usize,
) -> super::transaction::BuiltProgram<'p> {
    super::transaction::build_program(statics, eng, None, seed, dir, shape, n_y, n_in, rows)
        .expect("fixture program must assemble")
}

fn exception_at(layer: usize, neuron: usize, a2: f64, s2: f64, c2: f64, rows: usize) -> Exceptions {
    let mut exc = Exceptions::default();
    let list = exc.by_layer.entry(layer).or_default();
    for row in 0..rows {
        list.push(Exc {
            row,
            neuron,
            a2,
            s2,
            c2,
        });
    }
    exc
}

use crate::margin_row::engine::{Exc, Exceptions};

/// FIX A1 PIN 1 — THE COUNTEREXAMPLE. An exception whose true coefficients
/// EXCEED the base gate's `amax` must be covered by the charge the host
/// propagates. Base slopes are 0.5; the override's slope is 1.0, so on the
/// exception rows the true coefficient reaching the layer's output is TWICE
/// what the base-only statistic bounds. Everything downstream multiplies
/// both builds by the same factors, so the published `amax` must grow by at
/// least that ratio.
///
/// Without the fix this test fails by exact equality: `apply_activation`
/// computed `slope_norm` from the BASE quads only, and `amax` (which
/// multiplies every downstream split-excess charge — `w_excess * amax`,
/// `icept_excess * amax`) would be identical with and without exceptions.
#[test]
fn fix_a1_exception_charge_covers_a_slope_the_base_gate_does_not_bound() {
    use crate::margin_row::engine::{BackwardEngine, LaneDir};

    let (net, root) = half_slope_fixture();
    let eng = BackwardEngine::new(&net, &root);
    let statics =
        super::cache::PlanStatic::build(&eng, None, None).expect("half-slope fixture must admit");
    let li = *statics.ext.relus.first().expect("fixture has a relu");
    let rows = 3usize;
    let mut s = ndarray::Array2::<f64>::zeros((net.n_y, rows));
    for j in 0..net.n_y {
        for r in 0..rows {
            s[[j, r]] = if (j + r) % 2 == 0 { 1.0 } else { -0.5 };
        }
    }
    let seed = crate::margin_row::engine::Seed { s, e: None };

    let base = build_with_shape(
        &statics,
        &eng,
        &seed,
        LaneDir::Lower,
        PassShape::default(),
        net.n_y,
        net.n_in,
        rows,
    );
    // The counterexample is REAL: the base statistic bounds slope 0.5 while
    // the exception applies 1.0 on its rows.
    let (base_slope, exc_slope) = (0.5f64, 1.0f64);
    assert!(
        exc_slope > base_slope,
        "the pin is vacuous unless the override exceeds the base slope"
    );
    let exc = exception_at(li, 0, exc_slope, exc_slope, 0.0, rows);
    let with_exc = build_with_shape(
        &statics,
        &eng,
        &seed,
        LaneDir::Lower,
        PassShape {
            exc: Some(&exc),
            collect_unst_abs: false,
        },
        net.n_y,
        net.n_in,
        rows,
    );
    assert!(
        with_exc.amax >= base.amax * (exc_slope / base_slope),
        "FIX A1 regressed: exception amax {:e} does not cover the override's true \
         coefficients (base {:e}, required ratio {})",
        with_exc.amax,
        base.amax,
        exc_slope / base_slope
    );
    // And the fold is a MAX, not a blind inflation: an override WEAKER than
    // the base slope must leave the charge exactly where it was.
    let weak = exception_at(li, 0, 0.25, 0.25, 0.0, rows);
    let with_weak = build_with_shape(
        &statics,
        &eng,
        &seed,
        LaneDir::Lower,
        PassShape {
            exc: Some(&weak),
            collect_unst_abs: false,
        },
        net.n_y,
        net.n_in,
        rows,
    );
    assert_eq!(
        with_weak.amax.to_bits(),
        base.amax.to_bits(),
        "a weaker override must not move the coefficient-magnitude charge"
    );
    println!(
        "[fix-a1] amax base={:e} with_exc={:e} (ratio {:.6}) with_weak={:e}",
        base.amax,
        with_exc.amax,
        with_exc.amax / base.amax,
        with_weak.amax
    );
}

/// FIX A1 PIN 2 — TEETH ON THE SPLIT-FLOOR CHANNELS, end to end. Every base
/// parameter of the fixture splits in the NORMAL band, so `split_e` and
/// `split_eb` are contractually 0.0 (pinned by
/// `built_program_walks_the_chain_fixture_in_backward_order`). An override
/// placed in the f32-SUBNORMAL band has a split residual the in-kernel
/// relative cover provably misses — so if FIX A1 folds the override's
/// excesses, and ONLY if it does, the corresponding absolute channel becomes
/// strictly positive.
#[test]
fn fix_a1_exception_excess_reaches_the_split_floor_channels() {
    use crate::margin_row::engine::{BackwardEngine, LaneDir};

    let (net, root) = half_slope_fixture();
    let eng = BackwardEngine::new(&net, &root);
    let statics = super::cache::PlanStatic::build(&eng, None, None).expect("statics");
    let li = *statics.ext.relus.first().expect("fixture has a relu");
    let rows = 2usize;
    let seed = {
        let mut s = ndarray::Array2::<f64>::zeros((net.n_y, rows));
        for j in 0..net.n_y {
            s[[j, 0]] = 1.0;
            s[[j, 1]] = -1.0;
        }
        crate::margin_row::engine::Seed { s, e: None }
    };
    let base = build_with_shape(
        &statics,
        &eng,
        &seed,
        LaneDir::Lower,
        PassShape::default(),
        net.n_y,
        net.n_in,
        rows,
    );
    assert_eq!(base.split_e, 0.0, "fixture baseline must be normal-band");
    assert_eq!(base.split_eb, 0.0, "fixture baseline must be normal-band");

    // A subnormal-band SLOPE override rides the coefficient channel.
    let sub = 1.0e-130f64;
    let slope_exc = exception_at(li, 0, sub, sub, 0.0, rows);
    let with_slope = build_with_shape(
        &statics,
        &eng,
        &seed,
        LaneDir::Lower,
        PassShape {
            exc: Some(&slope_exc),
            collect_unst_abs: false,
        },
        net.n_y,
        net.n_in,
        rows,
    );
    assert!(
        with_slope.split_e > 0.0,
        "an f32-subnormal override slope must reach split_e (got {:e})",
        with_slope.split_e
    );

    // A subnormal-band INTERCEPT override rides the bias channel.
    let icept_exc = exception_at(li, 0, 1.0, 1.0, sub, rows);
    let with_icept = build_with_shape(
        &statics,
        &eng,
        &seed,
        LaneDir::Lower,
        PassShape {
            exc: Some(&icept_exc),
            collect_unst_abs: false,
        },
        net.n_y,
        net.n_in,
        rows,
    );
    assert!(
        with_icept.split_eb > 0.0,
        "an f32-subnormal override intercept must reach split_eb (got {:e})",
        with_icept.split_eb
    );
    println!(
        "[fix-a1] split floors: base=({:e},{:e}) subnormal-slope split_e={:e} \
         subnormal-icept split_eb={:e}",
        base.split_e, base.split_eb, with_slope.split_e, with_icept.split_eb
    );
}

/// FIX A1 PIN 3 — the fold itself, as a pure function: all FOUR channels
/// move in the documented direction (slope = max, intercept = sum, both
/// excesses = sum) and a zero-magnitude record moves nothing.
#[test]
fn fix_a1_fold_moves_every_channel_in_its_documented_direction() {
    use super::super::rounding::next_up as next_up_f64;
    use super::transaction::{exception_records, ExcRecord};
    use crate::margin_row::engine::LaneDir;

    let mk = |a2: f64, s2: f64, c2: f64| -> Vec<ExcRecord> {
        exception_records(
            &[Exc {
                row: 0,
                neuron: 1,
                a2,
                s2,
                c2,
            }],
            LaneDir::Lower,
            4,
            4,
            &[0.0; 4],
        )
        .expect("records")
    };
    let fold = |records: &[ExcRecord], base: (f64, f64, f64, f64)| {
        let (mut sn, mut se, mut im, mut ie) = base;
        super::transaction::fold_exception_charges(records, &mut sn, &mut se, &mut im, &mut ie);
        (sn, se, im, ie)
    };
    // Slope channel is a MAX over both sides.
    let (sn, _, im, _) = fold(&mk(2.0, 3.0, 0.0), (0.5, 0.0, 0.25, 0.0));
    assert!(sn >= 3.0, "slope_norm must take the larger override side");
    // A zero intercept slot adds only what the BASE gate loop's own
    // `next_up(|hi| + |lo|)` adds for a zero quad slot: one subnormal. (The
    // exact-zero discipline lives on `mul_up`/`add_up`, which is why
    // `split_eb` still reads a contractual 0.0 on an all-normal fixture.)
    assert!(
        (0.25f64..=next_up_f64(0.25)).contains(&im),
        "a zero intercept must add at most the base loop's own up-step: {im:e}"
    );
    // Intercept channel ADDS (the reduce folds every neuron's partial).
    let (_, _, im2, _) = fold(&mk(1.0, 1.0, 4.0), (0.5, 0.0, 0.25, 0.0));
    assert!(im2 >= 4.25, "icept_mass must ADD the override mass: {im2}");
    // A record whose quad is entirely zero is charge-neutral up to the same
    // single base-loop up-step: the slope MAX and both EXCESS channels are
    // exactly unchanged (the excesses are exact-zero-preserving), and the
    // intercept sum moves by at most one ulp.
    let (sn3, se3, im3, ie3) = fold(&mk(0.0, 0.0, 0.0), (0.5, 0.125, 0.25, 0.0625));
    assert_eq!(sn3.to_bits(), 0.5f64.to_bits(), "slope_norm is a max");
    assert_eq!(
        se3.to_bits(),
        0.125f64.to_bits(),
        "slope_excess stays exact"
    );
    assert_eq!(
        ie3.to_bits(),
        0.0625f64.to_bits(),
        "icept_excess stays exact"
    );
    assert!((0.25f64..=next_up_f64(0.25)).contains(&im3));
}

/// B4 ENCODING PIN: per-row exceptions compress into (neuron, row block,
/// quad) records; the quads are DIRECTION-BAKED and ABSOLUTE (derivation
/// obligation (i)); the byte layout is the kernel's three-vec4 record.
#[test]
fn exception_records_compress_and_bake_the_direction() {
    use super::transaction::{exception_bytes, exception_records};
    use crate::margin_row::engine::LaneDir;

    // The batched-children shape: one candidate block of nf = 3 consecutive
    // rows with an IDENTICAL override, then a second block on another
    // neuron. Two records, not six.
    let mut list = Vec::new();
    for row in 0..3 {
        list.push(Exc {
            row,
            neuron: 2,
            a2: 1.0,
            s2: 1.0,
            c2: 0.0,
        });
    }
    for row in 3..6 {
        list.push(Exc {
            row,
            neuron: 5,
            a2: 0.0,
            s2: 0.0,
            c2: 0.0,
        });
    }
    let c = vec![7.0f64; 8];
    let recs = exception_records(&list, LaneDir::Lower, 8, 6, &c).expect("records");
    assert_eq!(recs.len(), 2, "identical consecutive rows must merge");
    assert_eq!(
        (recs[0].neuron, recs[0].row_start, recs[0].row_end),
        (2, 0, 3)
    );
    assert_eq!(
        (recs[1].neuron, recs[1].row_start, recs[1].row_end),
        (5, 3, 6)
    );
    // LOWER bakes (a2, s2, 0, c2) — and the intercept slot is the ABSOLUTE
    // c2, never `c2 - c[neuron]`: this lane produces the intercept partial
    // in the gate kernel, so the base c never entered it.
    assert_eq!(recs[0].hi, [1.0, 1.0, 0.0, 0.0]);
    let upper = exception_records(
        &[Exc {
            row: 0,
            neuron: 2,
            a2: 0.25,
            s2: 0.75,
            c2: 0.5,
        }],
        LaneDir::Upper,
        8,
        6,
        &c,
    )
    .expect("records");
    // UPPER bakes (s2, a2, c2, 0).
    assert_eq!(upper[0].hi, [0.75, 0.25, 0.5, 0.0]);
    // Byte layout: 3 x vec4 per record, meta bit-cast from u32.
    let bytes = exception_bytes(&recs);
    assert_eq!(bytes.len(), recs.len() * 48);
    let w = |i: usize| {
        u32::from_le_bytes([
            bytes[4 * i],
            bytes[4 * i + 1],
            bytes[4 * i + 2],
            bytes[4 * i + 3],
        ])
    };
    assert_eq!((w(0), w(1), w(2), w(3)), (2, 0, 3, 0));
    assert_eq!(f32::from_bits(w(4)), 1.0, "hi quad follows the meta slot");
}

/// B4 FAIL-CLOSED PIN: every shape the kernel could not express refuses.
#[test]
fn exception_records_fail_closed_on_inexpressible_tables() {
    use super::transaction::{exception_records, MAX_EXC_RECORDS};
    use crate::margin_row::engine::LaneDir;

    let c = vec![0.0f64; 8];
    let out_of_range = |neuron: usize, row: usize| {
        exception_records(
            &[Exc {
                row,
                neuron,
                a2: 1.0,
                s2: 1.0,
                c2: 0.0,
            }],
            LaneDir::Lower,
            8,
            4,
            &c,
        )
    };
    assert!(out_of_range(8, 0).is_err(), "neuron outside the layer");
    assert!(out_of_range(0, 4).is_err(), "row outside the pass");
    // Two exceptions on ONE (neuron, row) ELEMENT: the CPU arm applies BOTH
    // bias deltas while the device would let the last matching record win —
    // refuse.
    let dup = vec![
        Exc {
            row: 1,
            neuron: 3,
            a2: 1.0,
            s2: 1.0,
            c2: 0.0,
        },
        Exc {
            row: 1,
            neuron: 3,
            a2: 0.0,
            s2: 0.0,
            c2: 0.0,
        },
    ];
    assert!(exception_records(&dup, LaneDir::Lower, 8, 4, &c).is_err());
    // More records than the scan cap.
    let rows = 4 * (MAX_EXC_RECORDS + 2);
    let big: Vec<Exc> = (0..rows)
        .map(|row| Exc {
            row,
            neuron: row % 8,
            a2: 1.0,
            s2: 1.0,
            c2: 0.0,
        })
        .collect();
    assert!(exception_records(&big, LaneDir::Lower, 8, rows, &c).is_err());
}

/// B4/B5 PROGRAM PIN: the exception table switches the step to the
/// `gate_transform_exc` entry with the extra binding and the record count in
/// the uniform; the collect shape adds one `collect_abs` dispatch per relu
/// with unstable neurons plus its readback.
#[test]
fn built_program_emits_the_exception_and_collect_steps() {
    use crate::margin_row::engine::{BackwardEngine, LaneDir};

    let (net, root) = half_slope_fixture();
    let eng = BackwardEngine::new(&net, &root);
    let statics = super::cache::PlanStatic::build(&eng, None, None).expect("statics");
    let li = *statics.ext.relus.first().expect("relu");
    let seed = eng.identity_seed();
    let rows = seed.s.ncols();
    let exc = exception_at(li, 0, 1.0, 1.0, 0.0, rows);
    let with_exc = build_with_shape(
        &statics,
        &eng,
        &seed,
        LaneDir::Lower,
        PassShape {
            exc: Some(&exc),
            collect_unst_abs: false,
        },
        net.n_y,
        net.n_in,
        rows,
    );
    let exc_steps: Vec<&super::transaction::Step> = with_exc
        .steps
        .iter()
        .filter(|s| s.entry == "gate_transform_exc")
        .collect();
    assert_eq!(
        exc_steps.len(),
        1,
        "exactly the exception layer switches entry"
    );
    assert_eq!(
        exc_steps[0].bindings.len(),
        8,
        "exception table is binding 8"
    );
    assert_eq!(
        u32::from_le_bytes([
            exc_steps[0].uniforms[8],
            exc_steps[0].uniforms[9],
            exc_steps[0].uniforms[10],
            exc_steps[0].uniforms[11],
        ]),
        1,
        "the uniform's third word carries the record count"
    );
    // Every OTHER relu still runs the untouched base entry with 7 bindings.
    for step in with_exc
        .steps
        .iter()
        .filter(|s| s.entry == "gate_transform")
    {
        assert_eq!(step.bindings.len(), 7);
    }

    // Collect: one dispatch per relu with unstable neurons, aligned readbacks.
    let with_coll = build_with_shape(
        &statics,
        &eng,
        &seed,
        LaneDir::Lower,
        PassShape {
            exc: None,
            collect_unst_abs: true,
        },
        net.n_y,
        net.n_in,
        rows,
    );
    let want: Vec<(usize, usize)> = statics
        .ext
        .relus
        .iter()
        .filter_map(|&l| {
            let rec = &root.layers[l];
            (!rec.unst.is_empty()).then_some((l, rec.unst.len()))
        })
        .collect();
    assert_eq!(
        with_coll
            .colls
            .iter()
            .map(|c| (c.layer, c.positions))
            .collect::<Vec<_>>(),
        want,
        "one collect per relu with unstable neurons, in fold order"
    );
    assert_eq!(
        with_coll.readbacks.len(),
        4 + want.len(),
        "collect readbacks ride after the four core streams"
    );
    assert!(with_coll.collect_unst_abs);
    assert_eq!(
        with_coll
            .steps
            .iter()
            .filter(|s| s.entry == "collect_abs")
            .count(),
        want.len()
    );
    for rb in &with_coll.readbacks {
        assert!(*rb < with_coll.buffers.len(), "readback out of arena");
    }
}

/// BYTE-IDENTITY PIN (the gate-off contract, extended for v8): with the new
/// sites in their DEFAULT state — no exception table, no collect — the
/// assembled program carries NO new step (no `gate_transform_exc`, no
/// `collect_abs`) and is field-for-field identical to the same build with an
/// EMPTY exception table, so the two spellings of "no exceptions" cannot
/// diverge. The new code paths are therefore unreachable unless a caller asks
/// for them, and the CPU path is untouched by construction.
///
/// NAME (corrected in the v9 hardening session): this does NOT compare
/// against a stored impl-v7 artifact — the old name
/// (`..._builds_the_pre_b4_program_bit_for_bit`) overstated it. The real
/// impl-v7 anchors are the radius bank (`chain <= 2e-11`, `residual <= 3e-11`,
/// unchanged since v6) and the fixture-walk census in the gpu-feature suite.
#[test]
fn default_pass_shape_equals_the_empty_exception_shape_and_adds_no_new_step() {
    use crate::margin_row::engine::{BackwardEngine, LaneDir};

    for (net, root) in [chain_fixture(), residual_fixture()] {
        let eng = BackwardEngine::new(&net, &root);
        let statics = super::cache::PlanStatic::build(&eng, None, None).expect("statics");
        let seed = eng.identity_seed();
        for dir in [LaneDir::Lower, LaneDir::Upper] {
            let a = super::transaction::build_program(
                &statics,
                &eng,
                None,
                &seed,
                dir,
                PassShape::default(),
                net.n_y,
                net.n_in,
                net.n_y,
            )
            .expect("default shape");
            // An EMPTY exception table is the same thing as none: the layer
            // lookup finds no records, so the step stays the base entry.
            let empty = Exceptions::default();
            let b = super::transaction::build_program(
                &statics,
                &eng,
                None,
                &seed,
                dir,
                PassShape {
                    exc: Some(&empty),
                    collect_unst_abs: false,
                },
                net.n_y,
                net.n_in,
                net.n_y,
            )
            .expect("empty exception table");
            assert_programs_bit_identical(&a, &b, "default vs empty-exception");
            assert!(a.colls.is_empty() && !a.collect_unst_abs);
            for step in &a.steps {
                assert_ne!(step.entry, "gate_transform_exc");
                assert_ne!(step.entry, "collect_abs");
            }
        }
    }
}

/// B5 TWIN PIN (no GPU): the host twin of `collect_abs` reproduces the CPU
/// arm's `sum_r |L[unst[pos], r]|` to ds precision, and takes the magnitude
/// EXACTLY (a ds sign flip is exact in every band).
#[test]
fn collect_abs_twin_matches_the_cpu_collection() {
    use super::ds::{collect_abs_twin, Ds};

    let mut rng = Lcg(0xC0FF_EE12_3456_789A);
    let (n, rows) = (9usize, 7usize);
    let mut plane = Vec::with_capacity(n * rows);
    let mut exact = vec![0.0f64; n];
    for j in 0..n {
        for _ in 0..rows {
            let v = f64::from(rng.coeff()) * (1.0 + 3.21e-10);
            #[allow(clippy::cast_possible_truncation)]
            let hi = v as f32;
            #[allow(clippy::cast_possible_truncation)]
            let lo = (v - f64::from(hi)) as f32;
            plane.push(Ds { hi, lo });
            exact[j] += (f64::from(hi) + f64::from(lo)).abs();
        }
    }
    let idx: Vec<usize> = (0..n).filter(|j| j % 2 == 0).collect();
    let got = collect_abs_twin(&idx, &plane, rows);
    assert_eq!(got.len(), idx.len());
    for (p, &j) in idx.iter().enumerate() {
        let v = got[p].to_f64();
        assert!(v >= 0.0, "an |.| fold cannot be negative");
        let rel = (v - exact[j]).abs() / exact[j].max(1e-300);
        assert!(
            rel <= 1e-12,
            "position {p} (neuron {j}): twin {v:e} vs exact {:e} (rel {rel:e})",
            exact[j]
        );
    }
}

/// WGSL SOURCE PINS for the two new kernels: the exception entry must
/// substitute the WHOLE quad before the shared body (so `m`/`mi` follow the
/// override — derivation obligation (ii)), and the collect kernel must keep
/// stating that it carries no certified charge.
#[test]
fn new_kernels_keep_their_load_bearing_source_shape() {
    // (ii) the override replaces both halves of the quad, and the charge
    // maxima are computed INSIDE the shared body from those variables.
    assert!(
        GATE_TRANSFORM_WGSL.contains("fn gate_apply("),
        "the shared per-element body must exist"
    );
    assert!(
        GATE_TRANSFORM_WGSL.contains("ghi = exc[3u * k + 1u]")
            && GATE_TRANSFORM_WGSL.contains("glo = exc[3u * k + 2u]"),
        "gate_transform_exc must substitute the whole ds-pair quad"
    );
    let body = GATE_TRANSFORM_WGSL
        .split("fn gate_apply(")
        .nth(1)
        .expect("body")
        .split("@compute")
        .next()
        .expect("body end");
    assert!(
        body.contains("let m = max(ds_abs_bound(vec2<f32>(ghi.x, glo.x))")
            && body.contains("let mi = max(ds_abs_bound(vec2<f32>(ghi.z, glo.z))"),
        "m/mi must be derived from the APPLIED quad inside the shared body"
    );
    // Both entries exist and only the exception one binds the table.
    assert!(GATE_TRANSFORM_WGSL.contains("fn gate_transform(@builtin"));
    assert!(GATE_TRANSFORM_WGSL.contains("fn gate_transform_exc(@builtin"));
    assert_eq!(
        GATE_TRANSFORM_WGSL.matches("@group(0) @binding(8)").count(),
        1,
        "exactly one binding-8 declaration (the exception table)"
    );
    // The collect kernel's heuristic-only contract is source-pinned: a
    // future edit that starts feeding a bound from it has to delete this.
    assert!(
        super::COLLECT_ABS_WGSL.contains("HEURISTIC ONLY")
            && super::COLLECT_ABS_WGSL.contains("CARRIES NO CERTIFIED CHARGE"),
        "collect_abs.wgsl must keep stating that it carries no certified charge"
    );
    assert!(
        super::COLLECT_ABS_WGSL.contains("fn collect_abs(@builtin"),
        "collect_abs entry point"
    );
    assert!(
        super::COLLECT_ABS_WGSL.contains("ds_add(acc, a)"),
        "the fold must stay ds (shortlist rankings flip only on true ties)"
    );
}

/// FIX A6 PIN — THE FULL CONSUMED SURFACE. The eval site has a SECOND
/// verdict-bearing consumer besides the aggregate node bound: the col-retire
/// per-COLUMN certified directs (bab.rs:2853 / :3341). This pins that the
/// site shadow's assertion sees a per-column corruption the aggregate cannot:
/// the corrupted column is NOT the minimum, so `min_c max(direct_c, ...)` —
/// the aggregate `b_eval` — is bit-unchanged, while `consumed_surface_overlap`
/// still trips.
#[test]
fn fix_a6_surface_compare_catches_a_column_the_aggregate_hides() {
    use crate::margin_row::engine::{BackwardEngine, LaneDir, Seed};

    let (net, root) = chain_fixture();
    let eng = BackwardEngine::new(&net, &root);
    let rows = 3usize;
    let mut s = ndarray::Array2::<f64>::zeros((net.n_y, rows));
    for j in 0..net.n_y {
        for r in 0..rows {
            s[[j, r]] = if (j + r) % 3 == 0 { 1.0 } else { -0.5 };
        }
    }
    let seed = Seed { s, e: None };
    let cpu = eng
        .run(&seed, None, LaneDir::Lower, None, false)
        .expect("cpu pass");
    // A faithful "device" pass: the same pass. Overlaps trivially.
    let good = eng
        .run(&seed, None, LaneDir::Lower, None, false)
        .expect("cpu pass");
    super::consumed_surface_overlap(&eng, &cpu, &good)
        .expect("an identical pass must overlap itself");

    // Which column is the aggregate minimum? Corrupt a DIFFERENT one.
    let low = eng.concretize_lower(&cpu);
    let argmin = (0..rows)
        .min_by(|&a, &b| low[a].total_cmp(&low[b]))
        .expect("rows");
    let victim = (0..rows)
        .find(|&r| r != argmin)
        .expect("a non-minimal column");
    let mut bad = eng
        .run(&seed, None, LaneDir::Lower, None, false)
        .expect("cpu pass");
    // Push the victim column far ABOVE its enclosure — a device pass that
    // disagrees with the CPU by orders of magnitude more than either radius.
    bad.b[victim] += 1e3;
    let bad_low = eng.concretize_lower(&bad);
    assert_eq!(
        bad_low
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min)
            .to_bits(),
        low.iter().cloned().fold(f64::INFINITY, f64::min).to_bits(),
        "the aggregate min must be UNCHANGED, or this pin proves nothing"
    );
    assert!(
        (bad_low[victim] - low[victim]).abs() > 1.0,
        "the corrupted column must actually move"
    );
    assert!(
        super::consumed_surface_overlap(&eng, &cpu, &bad).is_err(),
        "FIX A6 regressed: a per-column disagreement the aggregate hides must still trip"
    );
}

/// FIX A6 PIN 2: the surface assertion is an ENCLOSURE test, not a tolerance
/// — a pass that differs only WITHIN the combined certified radii is
/// accepted, one that steps outside is not.
#[test]
fn fix_a6_surface_compare_is_an_enclosure_not_a_tolerance() {
    use crate::margin_row::engine::{BackwardEngine, LaneDir};

    let (net, root) = chain_fixture();
    let eng = BackwardEngine::new(&net, &root);
    let seed = eng.identity_seed();
    let cpu = eng
        .run(&seed, None, LaneDir::Lower, None, false)
        .expect("cpu pass");
    let radius = super::radius_rows(&cpu, root.xabs.as_slice());
    let mut inside = eng
        .run(&seed, None, LaneDir::Lower, None, false)
        .expect("cpu pass");
    // Move every row by HALF the combined radius (the two radii are equal
    // here, so half of one is a quarter of the sum): still an enclosure.
    for (r, b) in inside.b.iter_mut().enumerate() {
        *b += 0.5 * radius[r];
    }
    super::consumed_surface_overlap(&eng, &cpu, &inside)
        .expect("a difference inside the combined radii must be accepted");
    let mut outside = eng
        .run(&seed, None, LaneDir::Lower, None, false)
        .expect("cpu pass");
    for (r, b) in outside.b.iter_mut().enumerate() {
        *b += 4.0 * radius[r].max(f64::MIN_POSITIVE) + 1.0;
    }
    assert!(
        super::consumed_surface_overlap(&eng, &cpu, &outside).is_err(),
        "a difference outside the combined radii must trip"
    );
}

/// B4/B5 DEVICE PARITY (the M1-grade gate for the two NEW kernels): the
/// exception entry and the collect entry must be BIT-IDENTICAL to their host
/// twins on the live adapter — the same bar `l_lane_kernels_are_bit_identical_on_this_adapter`
/// holds the other five to.
#[test]
#[cfg(feature = "gpu-tests")]
fn new_kernels_are_bit_identical_on_this_adapter() {
    use super::ds::{collect_abs_twin, Ds};
    use super::probe::{
        build_gate_fixture, host_gate_transform, pairs_from_bytes, scalars_from_bytes,
        GATE_NEURONS, GATE_ROWS,
    };
    use super::transaction;
    use lane_device_helpers as h;
    use ny_core::{
        LaneArenaBuffer, LaneProgramSpec, LaneStepBinding, LaneStepSpec, MarginRowLaneGpu,
    };

    let device = ny_gpu::MarginRowLaneDevice::new()
        .expect("gpu-tests requires a live wgpu adapter for the lane device");
    let ro = |buffer: usize| LaneStepBinding {
        buffer,
        read_only: true,
    };
    let rw = |buffer: usize| LaneStepBinding {
        buffer,
        read_only: false,
    };

    // ---- gate_transform_exc ----
    //
    // THE EQUIVALENCE THAT MAKES THIS A PARITY TEST: an override covering a
    // neuron's WHOLE row range is, by definition, the same computation as
    // substituting that neuron's base quad. So the device runs
    // `gate_transform_exc(base gates + full-range override)` and the host
    // reference is the UNCHANGED `host_gate_transform` on a fixture whose
    // quad was replaced — any drift in how the override is applied, or in
    // how `m`/`mi` follow it, shows up as a bit mismatch.
    let base = build_gate_fixture();
    let victim = 3usize;
    let over_hi = [0.875f32, 0.125, 0.0, 0.375];
    let over_lo = [1.5e-9f32, -4.25e-10, 0.0, 7.75e-10];
    let mut substituted = build_gate_fixture();
    substituted.gate[2 * victim] = over_hi;
    substituted.gate[2 * victim + 1] = over_lo;

    let total = GATE_NEURONS * GATE_ROWS;
    let gate_bytes: Vec<u8> = base.gate.iter().flat_map(|g| h::f32_bytes(g)).collect();
    let coeff_bytes = h::ds_bytes(&base.coeff);
    let err_bytes = h::f32_bytes(&base.err);
    // One record spanning every row of `victim`, in the kernel's 3 x vec4
    // encoding.
    let mut exc_bytes = Vec::new();
    #[allow(clippy::cast_possible_truncation)]
    for w in [victim as u32, 0u32, GATE_ROWS as u32, 0u32] {
        exc_bytes.extend_from_slice(&w.to_le_bytes());
    }
    exc_bytes.extend_from_slice(&h::f32_bytes(&over_hi));
    exc_bytes.extend_from_slice(&h::f32_bytes(&over_lo));
    #[allow(clippy::cast_possible_truncation)]
    let uni = h::uniform(&[GATE_NEURONS as u32, GATE_ROWS as u32, 1, 0], &[]);
    let buffers = [
        LaneArenaBuffer::Data(&gate_bytes),
        LaneArenaBuffer::Data(&coeff_bytes),
        LaneArenaBuffer::Zeroed((total * 8) as u64),
        LaneArenaBuffer::Data(&err_bytes),
        LaneArenaBuffer::Zeroed((total * 4) as u64),
        LaneArenaBuffer::Zeroed((total * 8) as u64),
        LaneArenaBuffer::Zeroed((total * 4) as u64),
        LaneArenaBuffer::Data(&exc_bytes),
    ];
    let bindings = [ro(0), ro(1), rw(2), ro(3), rw(4), rw(5), rw(6), ro(7)];
    #[allow(clippy::cast_possible_truncation)]
    let step = LaneStepSpec {
        label: "mr_gpu_backward_gate_transform_exc",
        entry: "gate_transform_exc",
        shader_wgsl: transaction::shader_gate(),
        uniforms: &uni,
        bindings: &bindings,
        workgroups: [(total as u32).div_ceil(256), 1, 1],
    };
    let out = device
        .run_lane_program(&LaneProgramSpec {
            label: "lane_test_gate_exc",
            buffers: &buffers,
            steps: std::slice::from_ref(&step),
            readbacks: &[2, 4, 5, 6],
            profile: false,
        })
        .expect("gate_transform_exc program runs");
    let (want_coeff, want_err, want_bias, want_perr) = host_gate_transform(&substituted);
    h::assert_ds_bits(
        &h::to_ds(&pairs_from_bytes(&out[0]).unwrap()),
        &want_coeff,
        "gate_exc coeff_out",
    );
    h::assert_f32_bits(&scalars_from_bytes(&out[1]), &want_err, "gate_exc err_out");
    h::assert_ds_bits(
        &h::to_ds(&pairs_from_bytes(&out[2]).unwrap()),
        &want_bias,
        "gate_exc bias_partial",
    );
    h::assert_f32_bits(
        &scalars_from_bytes(&out[3]),
        &want_perr,
        "gate_exc partial_err",
    );
    // The pin is only meaningful if the override actually CHANGED the
    // victim's output (otherwise it would pass against the base fixture too).
    let (base_coeff, _, _, _) = host_gate_transform(&base);
    assert!(
        (0..GATE_ROWS).any(|r| {
            let i = victim * GATE_ROWS + r;
            base_coeff[i].hi.to_bits() != want_coeff[i].hi.to_bits()
        }),
        "the override must change the victim neuron's output, or the parity pin is vacuous"
    );
    println!(
        "[gpu-backward gpu-test] gate_transform_exc: {total} lanes bit-identical \
         (override on neuron {victim})"
    );

    // ---- collect_abs ----
    let (n, rows) = (37usize, 11usize);
    let mut rng = Lcg(0x5EED_1234_ABCD_9876);
    let plane: Vec<Ds> = (0..n * rows)
        .map(|_| h::ds_split(f64::from(rng.coeff()) * (1.0 + 8.13e-10)))
        .collect();
    let idx: Vec<usize> = (0..n).filter(|j| j % 3 != 0).collect();
    #[allow(clippy::cast_possible_truncation)]
    let idx_bytes: Vec<u8> = idx.iter().flat_map(|&j| (j as u32).to_le_bytes()).collect();
    let plane_bytes = h::ds_bytes(&plane);
    #[allow(clippy::cast_possible_truncation)]
    let uni = h::uniform(&[idx.len() as u32, rows as u32, 0, 0], &[]);
    let buffers = [
        LaneArenaBuffer::Data(&idx_bytes),
        LaneArenaBuffer::Data(&plane_bytes),
        LaneArenaBuffer::Zeroed((idx.len() * 8) as u64),
    ];
    let bindings = [ro(0), ro(1), rw(2)];
    #[allow(clippy::cast_possible_truncation)]
    let step = LaneStepSpec {
        label: "mr_gpu_backward_collect_abs",
        entry: "collect_abs",
        shader_wgsl: transaction::shader_collect(),
        uniforms: &uni,
        bindings: &bindings,
        workgroups: [(idx.len() as u32).div_ceil(256), 1, 1],
    };
    let out = device
        .run_lane_program(&LaneProgramSpec {
            label: "lane_test_collect",
            buffers: &buffers,
            steps: std::slice::from_ref(&step),
            readbacks: &[2],
            profile: false,
        })
        .expect("collect_abs program runs");
    let want = collect_abs_twin(&idx, &plane, rows);
    h::assert_ds_bits(
        &h::to_ds(&pairs_from_bytes(&out[0]).unwrap()),
        &want,
        "collect_abs sums",
    );
    println!(
        "[gpu-backward gpu-test] collect_abs: {} positions bit-identical ({rows} rows each)",
        idx.len()
    );
}

/// B4/B5 END-TO-END on the live adapter: the WHOLE transaction with an
/// exception table and with the collect readback, against the CPU pass that
/// consumes the same inputs.
///
/// The exception arm is the verdict-bearing one: the device pass must
/// ENCLOSE the CPU pass whose `run(..., Some(&exc), ..)` applied the same
/// overrides. The collect arm asserts the heuristic map arrives with the CPU
/// arm's shape and values (to ds precision — it is a ranker input, not a
/// bound).
#[test]
#[cfg(feature = "gpu-tests")]
fn b4_b5_transactions_enclose_the_cpu_pass_on_the_fixtures() {
    use super::super::gpu_seam::tests as seam_tests;
    use super::authority::{DeviceParityProof, VerdictAuthority};
    use super::probe::self_check_streams;
    use super::transaction;
    use crate::margin_row::engine::{BackwardEngine, Collect, LaneDir, Seed};
    use ny_core::MarginRowLaneGpu;

    // Arc-typed for the same reason as the B1 e2e test above (FIX D3).
    let device: std::sync::Arc<dyn MarginRowLaneGpu> = std::sync::Arc::new(
        ny_gpu::MarginRowLaneDevice::new()
            .expect("gpu-tests requires a live wgpu adapter for the lane device"),
    );
    let streams = self_check_streams(device.as_ref()).expect("M1 probes must run");
    let proof = DeviceParityProof::qualify(true, &streams.host, &streams.device)
        .expect("M1 parity must hold before any transaction");
    let auth = VerdictAuthority::grant(proof);

    for (name, spec) in [
        ("chain", seam_tests::chain_spec()),
        ("residual", seam_tests::residual_spec()),
    ] {
        let (net, root, _, _) = seam_tests::compile(&spec);
        let eng = BackwardEngine::new(&net, &root);
        let rows = 3usize;
        let mut s = ndarray::Array2::<f64>::zeros((net.n_y, rows));
        for j in 0..net.n_y {
            for r in 0..rows {
                #[allow(clippy::cast_possible_wrap)]
                let v = ((j * 5 + r * 3) % 5) as i64 - 2;
                #[allow(clippy::cast_precision_loss)]
                {
                    s[[j, r]] = v as f64 * 0.5;
                }
            }
        }
        let seed = Seed { s, e: None };
        let plan_cache = super::cache::PlanCache::default();

        // ---- B4: the exception table, verdict-bearing ----
        let Some(li) = root.layers.iter().position(|l| !l.unst.is_empty()) else {
            continue;
        };
        let neuron = root.layers[li].unst[0];
        // The production shape: one candidate block per split direction.
        let mut exc = Exceptions::default();
        let list = exc.by_layer.entry(li).or_default();
        for row in 0..rows {
            let fix = if row == 0 {
                (1.0, 1.0, 0.0)
            } else {
                (0.0, 0.0, 0.0)
            };
            list.push(Exc {
                row,
                neuron,
                a2: fix.0,
                s2: fix.1,
                c2: fix.2,
            });
        }
        let cpu = eng
            .run(&seed, None, LaneDir::Lower, Some(&exc), false)
            .expect("cpu exception pass");
        let gpu = transaction::execute(
            &auth,
            &device,
            &eng,
            &seed,
            None,
            LaneDir::Lower,
            PassShape {
                exc: Some(&exc),
                collect_unst_abs: false,
            },
            &super::LaneCtx::default(),
            Some(&plan_cache),
            false,
            &mut transaction::ExecStats::default(),
        )
        .unwrap_or_else(|r| panic!("{name} exception pass refused: {r:?}"));
        super::enclosure_overlap(&cpu, &gpu)
            .unwrap_or_else(|p| panic!("{name} exception pass: M2 enclosure trip on {p}"));
        super::consumed_surface_overlap(&eng, &cpu, &gpu)
            .unwrap_or_else(|p| panic!("{name} exception pass: consumed-surface trip on {p}"));
        // The override must actually bite (else the pin is vacuous).
        let plain = eng
            .run(&seed, None, LaneDir::Lower, None, false)
            .expect("cpu plain pass");
        assert!(
            eng.concretize_lower(&cpu) != eng.concretize_lower(&plain),
            "{name}: the exception table must change the pass"
        );
        println!("[gpu-backward gpu-test] {name} B4 exception pass: enclosure OK");

        // ---- B5: the collect readback, heuristic-only ----
        let collect = Collect {
            unst_abs: true,
            rows: None,
            unst_rows: false,
        };
        let cpu_c = eng
            .run_collect(&seed, None, LaneDir::Lower, None, collect)
            .expect("cpu collect pass");
        let gpu_c = transaction::execute(
            &auth,
            &device,
            &eng,
            &seed,
            None,
            LaneDir::Lower,
            PassShape {
                exc: None,
                collect_unst_abs: true,
            },
            &super::LaneCtx::default(),
            Some(&plan_cache),
            false,
            &mut transaction::ExecStats::default(),
        )
        .unwrap_or_else(|r| panic!("{name} collect pass refused: {r:?}"));
        super::enclosure_overlap(&cpu_c, &gpu_c)
            .unwrap_or_else(|p| panic!("{name} collect pass: M2 enclosure trip on {p}"));
        let want = cpu_c.coll.as_ref().expect("cpu publishes the collect map");
        let got = gpu_c
            .coll
            .as_ref()
            .expect("device publishes the collect map");
        assert_eq!(
            want.keys().collect::<Vec<_>>(),
            got.keys().collect::<Vec<_>>(),
            "{name}: collect layer keys must match the CPU arm exactly"
        );
        for (layer, wv) in want {
            let gv = &got[layer];
            assert_eq!(wv.len(), gv.len(), "{name} layer {layer}: collect width");
            // SCALE-RELATIVE, not entry-relative. The device folds ITS OWN
            // plane, which differs from the CPU's f64 plane by the whole
            // walk's accumulated rounding; on a near-cancelled entry (the
            // fixture has entries at ~1e-17 against a layer scale of ~1e0)
            // an entry-relative bound compares pure noise. The consumer is a
            // RANKER, so the contract is "same ordering scale", which the
            // layer-max normalization expresses and the argmax check below
            // pins directly.
            let scale = wv.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-300);
            for (p, (w, g)) in wv.iter().zip(gv).enumerate() {
                assert!(
                    (g - w).abs() <= 1e-6 * scale,
                    "{name} layer {layer} pos {p}: collect {g:e} vs cpu {w:e} \
                     (layer scale {scale:e})"
                );
            }
            let argmax = |v: &[f64]| {
                v.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(i, _)| i)
            };
            assert_eq!(
                argmax(wv),
                argmax(gv),
                "{name} layer {layer}: the shortlist's top candidate must agree"
            );
        }
        println!(
            "[gpu-backward gpu-test] {name} B5 collect pass: {} layers match the CPU arm",
            want.len()
        );
    }
}

/// BYTE-IDENTITY PIN, AT THE SEAMS (the gate-off contract for B4/B5/B6):
/// with every new site in its DEFAULT state and the shadow lever unset —
/// which is what an unarmed test process and a competition binary both see —
/// the three new engine entries return the CPU pass BIT-FOR-BIT.
///
/// This is the "any `Err` means the unchanged CPU pass" contract made
/// executable at the call sites the hot loop uses: coefficients, error lane,
/// bias, bias error, and the collect map are compared by BITS, not by value.
#[test]
fn seamed_entries_are_bit_identical_to_the_cpu_pass_when_the_sites_are_off() {
    use crate::margin_row::engine::{
        domain_gates, BackwardEngine, Collect, Exc, Exceptions, LaneDir, Seed,
    };

    assert!(
        !super::enabled(),
        "this pin measures the DARK path; NY_MARGIN_ROW_GPU_BACKWARD must be unset here"
    );
    for site in [SITE_ROOT, SITE_Y_REFRESH, SITE_EVAL, SITE_SCORE] {
        assert_eq!(super::site_mode(site), SiteMode::Off);
    }

    let bits = |p: &crate::margin_row::engine::PassOut| {
        (
            p.a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            p.e.as_ref()
                .map(|e| e.iter().map(|v| v.to_bits()).collect::<Vec<_>>()),
            p.b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            p.eb.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            p.coll.as_ref().map(|m| {
                m.iter()
                    .map(|(k, v)| (*k, v.iter().map(|x| x.to_bits()).collect::<Vec<_>>()))
                    .collect::<Vec<_>>()
            }),
        )
    };

    for (net, root) in [chain_fixture(), residual_fixture()] {
        let eng = BackwardEngine::new(&net, &root);
        let rows = 3usize;
        let mut s = ndarray::Array2::<f64>::zeros((net.n_y, rows));
        for j in 0..net.n_y {
            for r in 0..rows {
                s[[j, r]] = if (j + r) % 2 == 0 { 0.75 } else { -1.25 };
            }
        }
        let seed = Seed { s, e: None };
        let doms = {
            let mut doms = vec![None];
            if let Some(li) = root.layers.iter().position(|l| !l.unst.is_empty()) {
                let dom = domain_gates(&root, &[(li, 0, 1)]);
                if !dom.infeasible {
                    doms.push(Some(dom));
                }
            }
            doms
        };
        let ctx = super::LaneCtx::default();
        for dom in &doms {
            for dir in [LaneDir::Lower, LaneDir::Upper] {
                // B5 seam vs `run_collect`.
                let collect = Collect {
                    unst_abs: true,
                    rows: None,
                    unst_rows: false,
                };
                let cpu = eng
                    .run_collect(&seed, dom.as_ref(), dir, None, collect)
                    .expect("cpu collect");
                let seamed = eng
                    .run_collect_seamed(&seed, dom.as_ref(), dir, collect, &ctx)
                    .expect("seamed collect");
                assert_eq!(bits(&cpu), bits(&seamed), "B5 seam must be bit-identical");
                assert!(seamed.coll.is_some(), "the collect map must be published");

                // B4 seam vs `run(.., Some(&exc), ..)`.
                let mut exc = Exceptions::default();
                if let Some(li) = root.layers.iter().position(|l| !l.unst.is_empty()) {
                    let neuron = root.layers[li].unst[0];
                    let list = exc.by_layer.entry(li).or_default();
                    for row in 0..rows {
                        list.push(Exc {
                            row,
                            neuron,
                            a2: 1.0,
                            s2: 1.0,
                            c2: 0.0,
                        });
                    }
                }
                let cpu = eng
                    .run(&seed, dom.as_ref(), dir, Some(&exc), false)
                    .expect("cpu exception pass");
                let seamed = eng
                    .run_exc_seamed(&seed, dom.as_ref(), dir, &exc, &ctx)
                    .expect("seamed exception pass");
                assert_eq!(bits(&cpu), bits(&seamed), "B4 seam must be bit-identical");
            }
            // B6 seam vs `y_rows`.
            let (cl, cu) = eng.y_rows(dom.as_ref()).expect("cpu y rows");
            let (sl, su) = eng
                .y_rows_seamed(dom.as_ref(), &super::super::gpu_seam::SeamCtx::default())
                .expect("seamed y rows");
            assert_eq!(bits(&cl), bits(&sl), "B6 lower lane must be bit-identical");
            assert_eq!(bits(&cu), bits(&su), "B6 upper lane must be bit-identical");
        }
    }
}

// ===========================================================================
// HARDENING SESSION (impl-v9): the seven defects the v8 adversarial verify
// raised. Every pin below is written to FAIL on the pre-v9 code.
// ===========================================================================

/// D1 PIN — THE MIXED-REPRESENTATION MERGE KEY (the latent inert-lever trap).
///
/// The pre-v9 comparator tested the ORIGINAL f64 quad's bits against the
/// RECONSTRUCTED ds-pair value `f64::from(hi) + f64::from(lo)`, which are
/// equal ONLY for f32-exact quads. So for any FRACTIONAL override — the B7
/// fractional-alpha re-entry, lane-alpha, re-bake payloads — run-length
/// compression NEVER fired: the record count became `2 * k_trunk * nf`
/// instead of `2 * k_trunk`, and at `nf > 1` it crossed `MAX_EXC_RECORDS`
/// into a silent `Unmappable` -> CPU. A seam that reads as armed and is
/// inert.
///
/// Part 1 measures the representation gap that makes it happen; part 2 pins
/// the merge itself; part 3 pins the CONSEQUENCE at the shipped shape.
#[test]
fn fractional_override_quads_merge_and_stay_inside_the_record_cap() {
    use super::transaction::{exception_records, MAX_EXC_RECORDS};
    use crate::margin_row::engine::LaneDir;

    // (1) The gap: reconstruction is NOT the identity on fractional values.
    let mut gaps = Vec::new();
    for v in [1.0f64 / 3.0, 0.1, 0.7] {
        #[allow(clippy::cast_possible_truncation)]
        let hi = v as f32;
        #[allow(clippy::cast_possible_truncation)]
        let lo = (v - f64::from(hi)) as f32;
        let recon = f64::from(hi) + f64::from(lo);
        assert_ne!(
            recon.to_bits(),
            v.to_bits(),
            "{v} must NOT reconstruct bit-exactly (this is what broke the key)"
        );
        gaps.push((v, (recon - v).abs()));
    }
    for v in [1.0f64, 0.5, 0.25, 0.0] {
        #[allow(clippy::cast_possible_truncation)]
        let hi = v as f32;
        let recon = f64::from(hi);
        assert_eq!(recon.to_bits(), v.to_bits(), "{v} is f32-exact");
    }
    println!(
        "[d1] recon gaps: {}",
        gaps.iter()
            .map(|(v, d)| format!("{v}: {d:e}"))
            .collect::<Vec<_>>()
            .join(" ")
    );

    // (2) The merge: nf consecutive rows with an IDENTICAL FRACTIONAL
    //     override are ONE record, exactly as an f32-exact override is.
    let c = vec![0.0f64; 8];
    let frac = |rows: usize| -> Vec<Exc> {
        (0..rows)
            .map(|row| Exc {
                row,
                neuron: 2,
                a2: 1.0 / 3.0,
                s2: 0.1,
                c2: 0.7,
            })
            .collect()
    };
    let recs = exception_records(&frac(4), LaneDir::Lower, 8, 4, &c).expect("records");
    assert_eq!(
        recs.len(),
        1,
        "identical consecutive FRACTIONAL rows must merge (pre-v9: 4)"
    );
    assert_eq!((recs[0].row_start, recs[0].row_end), (0, 4));
    // And the key really is the source, not the pair: the record's SOURCE
    // bits differ from its reconstructed pair bits on this quad, which is
    // exactly the condition the pre-v9 comparator tripped over.
    assert_ne!(
        recs[0].pair_value_bits()[0],
        (1.0f64 / 3.0).to_bits(),
        "the pair value must differ from the source (else this pin is vacuous)"
    );
    // A DIFFERENT fractional quad on the next row must NOT merge.
    let mut mixed = frac(2);
    mixed.push(Exc {
        row: 2,
        neuron: 2,
        a2: 1.0 / 3.0,
        s2: 0.1,
        c2: 0.7000000000000001,
    });
    let mixed = exception_records(&mixed, LaneDir::Lower, 8, 4, &c).expect("records");
    assert_eq!(mixed.len(), 2, "a changed quad must start a new record");

    // (3) The CONSEQUENCE at the shipped B4 shape: 2*k_trunk = 40 candidate
    //     blocks of nf = 8 identical rows each. Merged that is 40 records;
    //     unmerged it is 320 > MAX_EXC_RECORDS and the whole pass refuses to
    //     CPU — the inert seam.
    let (blocks, nf) = (40usize, 8usize);
    assert!(blocks * nf > MAX_EXC_RECORDS, "the pin must cross the cap");
    let mut list = Vec::with_capacity(blocks * nf);
    for b in 0..blocks {
        for k in 0..nf {
            list.push(Exc {
                row: b * nf + k,
                neuron: b,
                a2: 1.0 / 3.0,
                s2: 0.1,
                c2: 0.7,
            });
        }
    }
    let big = exception_records(
        &list,
        LaneDir::Lower,
        blocks,
        blocks * nf,
        &vec![0.0; blocks],
    )
    .expect("the merged table must fit the cap (pre-v9: Unmappable -> CPU)");
    assert_eq!(big.len(), blocks, "one record per candidate block");
    println!(
        "[d1] shipped-shape records: {} (unmerged would be {})",
        big.len(),
        blocks * nf
    );
}

/// D6 PIN — the duplicate check is per (neuron, row) ELEMENT, not per row.
///
/// Two exceptions on the SAME pass row at DIFFERENT neurons are legal and
/// handled identically on both arms (the CPU loop adds each `bpart` into
/// `b[row]`; the kernel matches `j == neuron && row in [r0, r1)` per thread).
/// The pre-v9 check compared only the ROW of adjacent entries in a list
/// sorted by `(neuron, row)`, so this shape refused to CPU — fail-closed and
/// inert at today's `score_candidates` shape, but it would silently inert any
/// future multi-neuron exception caller.
#[test]
fn two_exceptions_on_one_row_at_different_neurons_are_admitted() {
    use super::transaction::exception_records;
    use crate::margin_row::engine::LaneDir;

    let c = vec![0.0f64; 8];
    let list = vec![
        Exc {
            row: 1,
            neuron: 0,
            a2: 1.0,
            s2: 1.0,
            c2: 0.0,
        },
        Exc {
            row: 1,
            neuron: 3,
            a2: 0.5,
            s2: 0.25,
            c2: 2.0,
        },
    ];
    let recs = exception_records(&list, LaneDir::Lower, 8, 4, &c)
        .expect("a legal multi-neuron row must be admitted (pre-v9: refused)");
    assert_eq!(recs.len(), 2);
    assert_eq!(
        (recs[0].neuron, recs[0].row_start, recs[0].row_end),
        (0, 1, 2)
    );
    assert_eq!(
        (recs[1].neuron, recs[1].row_start, recs[1].row_end),
        (3, 1, 2)
    );
    // Still fail-closed on the genuine duplicate (same element twice).
    let dup = vec![
        Exc {
            row: 1,
            neuron: 3,
            a2: 1.0,
            s2: 1.0,
            c2: 0.0,
        },
        Exc {
            row: 1,
            neuron: 3,
            a2: 0.0,
            s2: 0.0,
            c2: 0.0,
        },
    ];
    assert!(exception_records(&dup, LaneDir::Lower, 8, 4, &c).is_err());
}

/// D5a PIN 1 — the per-site shadow selector's EXACT semantics. Unset is the
/// impl-v7 root-only shape (one device pass per instance), not v8's
/// all-four-sites shape.
#[test]
fn shadow_site_selector_defaults_to_the_impl_v7_root_only_shape() {
    use super::shadow_sites_from;

    assert_eq!(
        shadow_sites_from(None),
        Ok(SITE_ROOT),
        "unset means root only"
    );
    assert_eq!(
        shadow_sites_from(Some("")),
        Ok(SITE_ROOT),
        "empty is not a name"
    );
    assert_eq!(shadow_sites_from(Some("  \t ")), Ok(SITE_ROOT));
}

/// D5a PIN 2 — every site is nameable, singly and in unions; `all` restores
/// v8's shape and `none` is the shadow-lever-only control arm.
#[test]
fn shadow_site_selector_names_every_site_unambiguously() {
    use super::shadow_sites_from;

    assert_eq!(shadow_sites_from(Some("root")), Ok(SITE_ROOT));
    assert_eq!(shadow_sites_from(Some("y_refresh")), Ok(SITE_Y_REFRESH));
    assert_eq!(shadow_sites_from(Some("eval")), Ok(SITE_EVAL));
    assert_eq!(shadow_sites_from(Some("score")), Ok(SITE_SCORE));
    assert_eq!(shadow_sites_from(Some("all")), Ok(SITE_ALL));
    assert_eq!(shadow_sites_from(Some("none")), Ok(0));
    assert_eq!(
        shadow_sites_from(Some("eval,score")),
        Ok(SITE_EVAL | SITE_SCORE)
    );
    assert_eq!(
        shadow_sites_from(Some(" eval , score , ")),
        Ok(SITE_EVAL | SITE_SCORE),
        "tokens are trimmed and empty tokens skipped"
    );
    // The token vocabulary IS the telemetry vocabulary: every site tag round
    // trips through the selector, so a `site=` line always names an
    // accepted value.
    for site in [SITE_ROOT, SITE_Y_REFRESH, SITE_EVAL, SITE_SCORE] {
        assert_eq!(shadow_sites_from(Some(super::site_tag(site))), Ok(site));
    }
    assert_eq!(
        super::shadow_sites_label(SITE_EVAL | SITE_SCORE),
        "eval,score"
    );
    assert_eq!(super::shadow_sites_label(0), "none");
    assert_eq!(
        super::shadow_sites_label(SITE_ALL),
        "root,y_refresh,eval,score"
    );
}

/// D5a PIN 3 — a typo shadows NOTHING (loudly) rather than silently falling
/// back to the default. A run that measured `root` while the operator asked
/// for `eval` is the exact "reads as armed, measured something else" class
/// this campaign keeps paying for.
#[test]
fn shadow_site_selector_fails_closed_on_an_unknown_token() {
    use super::shadow_sites_from;

    assert_eq!(shadow_sites_from(Some("evla")), Err("evla"));
    assert_eq!(shadow_sites_from(Some("eval,evla")), Err("evla"));
    assert_eq!(
        shadow_sites_from(Some("EVAL")),
        Err("EVAL"),
        "matching is exact and lowercase — no aliases"
    );
    assert_eq!(shadow_sites_from(Some("1")), Err("1"));
}

/// D5a PIN 4 — the selector is SHADOW-ONLY: it can never grant, withhold or
/// alter verdict authority, in either direction.
#[test]
fn shadow_site_selector_cannot_touch_verdict_authority() {
    use super::site_mode_from;

    for site in [SITE_Y_REFRESH, SITE_EVAL, SITE_SCORE] {
        let mask = effective_site_mask(site, false);
        // Authorized with the selector naming nothing...
        assert_eq!(
            site_mode_from(true, true, 0, mask, site),
            SiteMode::Authorized
        );
        // ...and with the selector naming everything.
        assert_eq!(
            site_mode_from(true, true, SITE_ALL, mask, site),
            SiteMode::Authorized
        );
        // A selector value can never promote an unauthorized site.
        assert_eq!(
            site_mode_from(true, true, SITE_ALL, 0, site),
            SiteMode::Shadow
        );
        assert_eq!(
            site_mode_from(false, false, SITE_ALL, 0, site),
            SiteMode::Off
        );
    }
}

/// D5a PIN 5 — the declaration itself (the lever must be governed, dark, and
/// documented; an env-only read would not fire in competition anyway, but an
/// UNDECLARED one would not even appear in the receipt).
#[test]
fn shadow_site_selector_declaration_is_the_documented_selector() {
    let decl = &ny_levers::decls::margin_row::MARGIN_ROW_GPU_SHADOW_SITES;
    assert_eq!(decl.name, "NY_MARGIN_ROW_GPU_SHADOW_SITES");
    assert_eq!(decl.kind, ny_levers::LeverKind::Text);
    assert_eq!(decl.default, ny_levers::DefaultSpec::Text("root"));
    assert_eq!(decl.bucket, ny_levers::Bucket::Debug);
    // The doc must state the default's meaning, or the next measurement
    // reads the wrong shape off the lever list.
    assert!(
        decl.doc.contains("root` ALONE") || decl.doc.contains("root` alone"),
        "the decl must state that UNSET means root only"
    );
    assert!(!decl.readers.is_empty(), "the read site must be declared");
}

/// D4 PIN 1 — refusal labels are per REASON, including the `Unmappable`
/// string. The pre-v9 `tag()` collapsed every `Unmappable` to one number, so
/// the FIRST one seen anywhere in the process permanently masked all the
/// others — including all four B4/B5 reasons and the new D2 budget.
#[test]
fn refusal_labels_distinguish_every_unmappable_reason() {
    use super::Refusal;

    let unmappable = [
        "collection shape outside the B5 seam",
        "too many exception records",
        "pass exceeds the 65535x256 thread cap",
        "two exceptions on one (neuron, row) element",
        "pass arena exceeds the per-pass byte budget",
    ];
    let mut seen: std::collections::BTreeSet<&'static str> = std::collections::BTreeSet::new();
    for what in unmappable {
        assert!(
            seen.insert(Refusal::Unmappable(what).label()),
            "two Unmappable reasons share a label: {what}"
        );
    }
    // ...and no fixed-variant label collides with them or with each other.
    for refusal in [
        Refusal::Disabled,
        Refusal::Unimplemented,
        Refusal::ChannelDead,
        Refusal::Busy,
        Refusal::NotOutward,
        Refusal::Rows,
        Refusal::Deadline,
        Refusal::Device,
        Refusal::Payload,
        Refusal::ErrorFloor,
        Refusal::Probe,
        Refusal::NonFinite,
        Refusal::ShadowOnly,
    ] {
        assert!(
            seen.insert(refusal.label()),
            "duplicate refusal label: {refusal:?}"
        );
    }
}

/// D4 PIN 2 — the per-site ledger is really per site: counting a refusal at
/// one site must not print or count at another, and the census names each
/// site separately. (The counters are process-wide statics, so this pin uses
/// reasons no other test produces and asserts on the census text.)
#[test]
fn refusal_accounting_is_per_site_not_process_wide() {
    use super::{Refusal, SITE_EVAL, SITE_SCORE};

    // Two DIFFERENT Unmappable reasons at two DIFFERENT sites: pre-v9 the
    // second would have been silently swallowed by the first's tag bit.
    super::note_refusal(SITE_EVAL, Refusal::Unmappable("d4 pin eval reason"));
    super::note_refusal(SITE_EVAL, Refusal::Unmappable("d4 pin eval reason"));
    super::note_refusal(SITE_SCORE, Refusal::Unmappable("d4 pin score reason"));
    let census = super::site_census_line().expect("something has been observed");
    println!("{census}");
    assert!(
        census.contains("eval[") && census.contains("score["),
        "the census must name each site separately: {census}"
    );
    assert!(
        census.contains("d4 pin eval reason=2"),
        "the eval site must count BOTH of its refusals: {census}"
    );
    assert!(
        census.contains("d4 pin score reason=1"),
        "the score site's own reason must survive the eval site's: {census}"
    );
    // The reasons must not have leaked into each other's site.
    let eval_part = census
        .split("eval[")
        .nth(1)
        .and_then(|s| s.split(']').next())
        .expect("eval section");
    assert!(
        !eval_part.contains("d4 pin score reason"),
        "site sections must not share reasons: {eval_part}"
    );
}

/// D2 PIN — the per-pass ARENA BYTE BUDGET (design section 5 risk-3, which
/// v8 left unimplemented: only a THREAD cap and the executor's pool retention
/// valve existed). Fail-closed with its own refusal reason, and the accounting
/// is the real one — the number the budget is applied to is exactly the sum of
/// the program's own buffer sizes, padded as the executor pads them.
#[test]
fn pass_arena_byte_budget_is_enforced_on_the_real_arena_total() {
    use super::transaction::{arena_byte_cap, padded_buffer_len, Init, MAX_PASS_ARENA_BYTES};
    use crate::margin_row::engine::{BackwardEngine, LaneDir};

    // The pure predicate, exact at the boundary.
    assert!(arena_byte_cap(0).is_ok());
    assert!(arena_byte_cap(MAX_PASS_ARENA_BYTES).is_ok());
    match arena_byte_cap(MAX_PASS_ARENA_BYTES + 1) {
        Err(Refusal::Unmappable(what)) => {
            assert_eq!(what, "pass arena exceeds the per-pass byte budget");
        }
        other => panic!("over-budget must refuse with its own reason, got {other:?}"),
    }
    assert!(arena_byte_cap(u64::MAX).is_err());
    // The executor's padding rule.
    assert_eq!(padded_buffer_len(0), 4);
    assert_eq!(padded_buffer_len(1), 4);
    assert_eq!(padded_buffer_len(8), 8);
    assert_eq!(padded_buffer_len(9), 12);

    // The accounting is not decorative: it equals the program's own arena.
    for (net, root) in [chain_fixture(), residual_fixture()] {
        let eng = BackwardEngine::new(&net, &root);
        let statics = super::cache::PlanStatic::build(&eng, None, None).expect("statics");
        let seed = eng.identity_seed();
        let built = build_with_shape(
            &statics,
            &eng,
            &seed,
            LaneDir::Lower,
            PassShape {
                exc: None,
                collect_unst_abs: true,
            },
            net.n_y,
            net.n_in,
            net.n_y,
        );
        let want: u64 = built
            .buffers
            .iter()
            .map(|init| match init {
                Init::Data(b) => padded_buffer_len(b.len() as u64),
                Init::DataRef(b) => padded_buffer_len(b.len() as u64),
                Init::Zeroed(len) => padded_buffer_len(*len),
                // Resident planes are registry-owned, not per-pass arena.
                Init::Resident(_) => 0,
            })
            .sum();
        assert_eq!(
            built.arena_bytes, want,
            "arena_bytes must be the program's own padded buffer total"
        );
        assert!(
            arena_byte_cap(built.arena_bytes).is_ok(),
            "a fixture pass must not be refused by the budget"
        );
        println!("[d2] fixture arena_bytes={} bytes", built.arena_bytes);
    }
}

/// D3 PIN — resident planes RELEASE their device buffers when the plan static
/// that owns them is dropped, and residency changes no byte of the built
/// program. impl-v7 had ONE process-scoped cache slot; impl-v8 gave every
/// `MarginRowBab` its own, so an append-only registry became a per-BaB-instance
/// growth term that never shrank.
#[test]
fn resident_planes_are_released_when_their_plan_static_drops() {
    use crate::margin_row::engine::{BackwardEngine, LaneDir};
    use ny_core::{LaneProgramSpec, LaneProgramTiming, LaneResidentId, MarginRowLaneGpu};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Mutex};

    /// A no-device executor that only implements the residency registry —
    /// enough to pin the OWNERSHIP contract without an adapter.
    #[derive(Default)]
    struct CountingExecutor {
        live: Mutex<Vec<Option<u64>>>,
        uploads: AtomicU64,
        releases: AtomicU64,
    }
    impl MarginRowLaneGpu for CountingExecutor {
        fn denorm_preserve_enabled(&self) -> bool {
            true
        }
        fn adapter_summary(&self) -> String {
            "counting-executor".to_string()
        }
        fn run_lane_transaction(
            &self,
            _spec: &ny_core::LaneTransactionSpec<'_>,
        ) -> Result<Vec<Vec<u8>>, String> {
            Err("no device".to_string())
        }
        fn run_lane_program(&self, _spec: &LaneProgramSpec<'_>) -> Result<Vec<Vec<u8>>, String> {
            Err("no device".to_string())
        }
        fn run_lane_program_timed(
            &self,
            _spec: &LaneProgramSpec<'_>,
            _timing: &mut LaneProgramTiming,
        ) -> Result<Vec<Vec<u8>>, String> {
            Err("no device".to_string())
        }
        fn upload_resident(&self, bytes: &[u8]) -> Result<LaneResidentId, String> {
            let mut live = self.live.lock().expect("lock");
            let id = LaneResidentId::from_raw(live.len() as u64);
            live.push(Some(bytes.len() as u64));
            self.uploads.fetch_add(1, Ordering::Relaxed);
            Ok(id)
        }
        fn release_resident(&self, id: LaneResidentId) {
            let mut live = self.live.lock().expect("lock");
            if let Some(slot) = live.get_mut(id.raw() as usize) {
                if slot.is_some() {
                    self.releases.fetch_add(1, Ordering::Relaxed);
                }
                *slot = None;
            }
        }
        fn resident_bytes(&self) -> u64 {
            self.live
                .lock()
                .expect("lock")
                .iter()
                .filter_map(|s| *s)
                .sum()
        }
    }

    let (net, root) = residual_fixture();
    let eng = BackwardEngine::new(&net, &root);
    let exec = Arc::new(CountingExecutor::default());
    let gpu: Arc<dyn MarginRowLaneGpu> = exec.clone();

    let seed = eng.identity_seed();
    let plain = super::cache::PlanStatic::build(&eng, None, None).expect("statics");
    let plain_program = build_with_shape(
        &plain,
        &eng,
        &seed,
        LaneDir::Lower,
        PassShape::default(),
        net.n_y,
        net.n_in,
        net.n_y,
    );

    let uploaded_bytes;
    let uploads;
    {
        let resident =
            super::cache::PlanStatic::build(&eng, None, Some(&gpu)).expect("resident statics");
        uploads = exec.uploads.load(Ordering::Relaxed);
        uploaded_bytes = exec.resident_bytes();
        assert!(uploads > 0, "the fixture must have resident planes");
        assert!(uploaded_bytes > 0);
        assert_eq!(
            exec.releases.load(Ordering::Relaxed),
            0,
            "nothing may be released while the static is live"
        );
        // VALUE NEUTRALITY: residency only changes WHERE the same bytes come
        // from, so the assembled program must be identical except for the
        // buffer INIT KIND of the resident planes.
        let resident_program = build_with_shape(
            &resident,
            &eng,
            &seed,
            LaneDir::Lower,
            PassShape::default(),
            net.n_y,
            net.n_in,
            net.n_y,
        );
        assert_eq!(plain_program.steps.len(), resident_program.steps.len());
        for (a, b) in plain_program.steps.iter().zip(&resident_program.steps) {
            assert_eq!(a.label, b.label);
            assert_eq!(a.entry, b.entry);
            assert_eq!(a.uniforms, b.uniforms, "uniform bytes must not move");
            assert_eq!(a.workgroups, b.workgroups);
        }
        assert_eq!(
            plain_program.split_e.to_bits(),
            resident_program.split_e.to_bits(),
            "residency must not move a single charge"
        );
        assert_eq!(
            plain_program.split_eb.to_bits(),
            resident_program.split_eb.to_bits()
        );
        assert_eq!(
            plain_program.e_widen.to_bits(),
            resident_program.e_widen.to_bits()
        );
        assert_eq!(
            plain_program.eb_widen.to_bits(),
            resident_program.eb_widen.to_bits()
        );
        assert_eq!(
            plain_program.amax.to_bits(),
            resident_program.amax.to_bits()
        );
    }
    // Dropped: every plane released, registry back to zero live bytes.
    assert_eq!(
        exec.releases.load(Ordering::Relaxed),
        uploads,
        "every resident plane must release exactly once when its static drops"
    );
    assert_eq!(
        exec.resident_bytes(),
        0,
        "the registry must hold no live bytes after the static drops"
    );
    println!("[d3] uploads={uploads} bytes={uploaded_bytes} -> live=0 after drop");

    // A second static uploads afresh (new slots, never a reused id) and also
    // releases: the registry tracks LIVE statics, not every static ever built.
    {
        let _second =
            super::cache::PlanStatic::build(&eng, None, Some(&gpu)).expect("resident statics");
        assert_eq!(exec.resident_bytes(), uploaded_bytes);
    }
    assert_eq!(exec.resident_bytes(), 0);
    assert_eq!(exec.releases.load(Ordering::Relaxed), 2 * uploads);
}

/// D7 PIN — the exception record header rides as BIT-CAST SUBNORMAL f32, so
/// the whole B4 override path depends on the DenormPreserve admission.
///
/// An FTZ-on-load adapter would read every header lane as 0, every record
/// would target `(neuron 0, rows [0, 0))`, NO override would be applied on
/// the device while the CPU arm applies it — device relaxation != CPU
/// relaxation, silently, on exactly the rows banked as child bounds. It is
/// prevented structurally (only `DeviceParityProof::qualify` mints the
/// authority a transaction requires, and it refuses a resolved-OFF channel),
/// and this pin ties the two ends together so neither can move alone.
#[test]
fn exception_header_words_are_subnormals_that_depend_on_denorm_preserve() {
    use super::authority::DeviceParityProof;
    use super::ds::Ds;
    use super::transaction::{exception_bytes, exception_records};
    use crate::margin_row::engine::LaneDir;

    // (1) The hazard is real: realistic header words ARE f32 subnormals.
    let c = vec![0.0f64; 64];
    let list: Vec<Exc> = (0..3)
        .map(|row| Exc {
            row: row + 5,
            neuron: 37,
            a2: 1.0,
            s2: 1.0,
            c2: 0.0,
        })
        .collect();
    let recs = exception_records(&list, LaneDir::Lower, 64, 16, &c).expect("records");
    let bytes = exception_bytes(&recs);
    let word = |i: usize| {
        u32::from_le_bytes([
            bytes[4 * i],
            bytes[4 * i + 1],
            bytes[4 * i + 2],
            bytes[4 * i + 3],
        ])
    };
    let (neuron, r0, r1) = (word(0), word(1), word(2));
    assert_eq!((neuron, r0, r1), (37, 5, 8));
    for (name, w) in [("neuron", neuron), ("row_start", r0), ("row_end", r1)] {
        assert!(
            w < (1 << 23),
            "{name} must be below the f32 normal threshold"
        );
        assert!(
            f32::from_bits(w).is_subnormal(),
            "{name}={w} must bit-cast to an f32 SUBNORMAL — that is the coupling"
        );
        assert_eq!(f32::from_bits(w).to_bits(), w, "bit-cast must round trip");
    }

    // (2) The structural prevention: a resolved-OFF denorm channel cannot
    //     mint the proof, so no transaction (hence no exception table) can
    //     ever reach a flushing adapter.
    let host = vec![Ds { hi: 1.0, lo: 0.0 }; super::authority::PARITY_MIN_LANES];
    let device = vec![(1.0f32, 0.0f32); super::authority::PARITY_MIN_LANES];
    assert!(
        DeviceParityProof::qualify(true, &host, &device).is_ok(),
        "the fixture must otherwise qualify (else this pin is vacuous)"
    );
    match DeviceParityProof::qualify(false, &host, &device) {
        Err(Refusal::Unmappable(what)) => {
            assert!(
                what.contains("denorm-preserve"),
                "the refusal must name the channel: {what}"
            );
        }
        Ok(_) => panic!("a resolved-OFF denorm channel must refuse, not qualify"),
        Err(other) => panic!("unexpected refusal: {other:?}"),
    }
}
