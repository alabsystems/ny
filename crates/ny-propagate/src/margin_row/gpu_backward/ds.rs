// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Host-side double-single (ds32) twin of `kernels/ds_primitives.wgsl`.
//!
//! Every function here mirrors its WGSL namesake OP-FOR-OP — same EFT
//! sequence, same order of operations — so that the M1 device-vs-host
//! bit-comparison (design §7) is meaningful: on an adapter where the
//! primitives are bit-exact (measured on this GB10: fma TwoProduct and the
//! fma-barrier TwoSum, 0 ULP over the adversarial probe lanes), the device
//! stream and this twin must agree BIT-IDENTICALLY, and any divergence kills
//! the channel.
//!
//! The scalar EFT primitives are REUSED from [`ny_core::eft`] (exact-rational
//! oracle-validated), not re-derived. This module adds only the double-single
//! composition layer and its certified envelope constant.
//!
//! # Soundness role
//!
//! Nothing in this file feeds a verdict. It is (a) the M0/M1 comparison
//! reference and (b) the executable specification the implementation session
//! ports into the device transaction. The EFT-identity doc examples below
//! are also duplicated as executable unit tests in `super::tests` — this
//! module is `pub(crate)`, and rustdoc only RUNS examples on externally
//! reachable items, so the copies in `tests.rs` are the ones CI executes.
//! The certified-error accounting that
//! WILL feed verdicts is specified in design §4.2; the envelope constant
//! [`U_DS`] below is the per-op relative bound that section charges for ds
//! algebra (a-priori, tiny) — a WRONG value here could eventually cost proofs
//! (loosening is safe, tightening is not), so it is set with 8x headroom over
//! the literature bounds and pinned by the enclosure tests.

// Production callers exist as of v4 (probe + transaction); the allow stays
// only for the test-facing reference pieces (`ds_dot`, doc-twin helpers)
// that deliberately live here beside the algebra they specify.
#![allow(dead_code)]

use ny_core::eft::{two_prod_f32, two_sum_f32};

/// Certified per-op RELATIVE error bound for the ds32 algebra: `2^-44`.
///
/// Joldes–Muller–Popescu ("Tight and rigorous error bounds for basic building
/// blocks of double-word arithmetic", ACM TOMS 2017) prove, for `u = 2^-24`:
/// DWPlusFP (Algorithm 4, the shape of [`ds_add_f32`]) has relative error
/// `<= 2u^2 = 2^-47`; DWTimesFP with fma (Algorithm 9, the shape of
/// [`ds_mul_f32`]) `<= 2u^2`; ACCURATE DWPlusDW (Algorithm 6, the shape of
/// [`ds_add`]) `<= 3u^2/(1-4u) < 2^-45.4`; one-fma DWTimesDW in the
/// `xl*yl`-DROPPING shape of [`ds_mul`] (DWTimesDW2) `<= 6u^2 < 2^-45.4`
/// (the tighter `5u^2` belongs to the `xl*yl`-CARRYING Algorithm 12, which
/// is NOT the implemented sequence — see the [`ds_mul`] note) — all under
/// round-to-nearest with no underflow, all UNCONDITIONAL in the inputs
/// (cancellation included). `2^-44 = 16u^2` covers every one with >= 2.6x
/// headroom on the loosest.
///
/// SOUNDNESS (load-bearing): this constant is only ever used to WIDEN an
/// error term (design §4.2 "ds algebra's own O(u^2) residue"), so an
/// overestimate loosens bounds (costs proofs, never manufactures one). The
/// underflow exclusion is discharged separately by the absolute flush floors
/// (design §4.1), exactly as `ny_core::eft` separates its `2^-126` floor from
/// the residual channel.
pub(crate) const U_DS: f64 = 5.684_341_886_080_802e-14; // 2^-44, exact

/// [`U_DS`] as f32 (2^-44 is a power of two, so the downcast is EXACT). The
/// host twin of the WGSL `DS_UNIT` constant in `ds_primitives.wgsl` — kernels
/// charge per-op ds residues with it IN f32, so the twin must too.
// The f32 literal is the shortest decimal that round-trips to 2^-44; it is the
// SAME f32 value (0x2980_0000) as the f64 `U_DS` above, which needs more digits
// only because f64 resolves finer.
pub(crate) const U_DS_F32: f32 = 5.684_342e-14; // 2^-44, exact

/// Host twin of the WGSL `ds_abs_bound`: `|hi| + |lo| >= |hi + lo|`, one f32
/// add of nonnegative terms (the E-lane egress widening covers its rounding).
#[inline]
pub(crate) fn ds_abs_bound(x: Ds) -> f32 {
    x.hi.abs() + x.lo.abs()
}

/// A double-single f32 value: the unevaluated sum `hi + lo`.
///
/// Invariant (re-established by every constructor via renormalization):
/// `|lo| <= ulp(hi) / 2`, so the pair carries ~49 effective significand bits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Ds {
    /// Leading component.
    pub(crate) hi: f32,
    /// Trailing component, `|lo| <= ulp(hi)/2`.
    pub(crate) lo: f32,
}

impl Ds {
    /// Exact injection of a single f32.
    #[inline]
    pub(crate) const fn from_f32(x: f32) -> Self {
        Self { hi: x, lo: 0.0 }
    }

    /// The value as f64 — EXACT, not an approximation.
    ///
    /// Both components convert to f64 exactly (24 < 53 significand bits) and
    /// the renormalization invariant bounds the exponent gap so the true sum
    /// `hi + lo` needs at most ~49 significand bits: one f64 addition of the
    /// two exact conversions commits zero rounding.
    ///
    /// ```
    /// // Doc-test of the exactness claim on a worst-spread pair.
    /// let hi = 1.0f32;
    /// let lo = f32::from_bits(0x3380_0000); // 2^-24 = ulp(1.0)/2
    /// let v = f64::from(hi) + f64::from(lo);
    /// assert_eq!(v, 1.0 + 2f64.powi(-24)); // representable, exact
    /// ```
    #[inline]
    pub(crate) fn to_f64(self) -> f64 {
        f64::from(self.hi) + f64::from(self.lo)
    }
}

/// Fast two-sum (Dekker): for `|a| >= |b|` (or `a == 0`), `s = fl(a+b)` and
/// `t` is the EXACT residual `a + b - s`.
///
/// Used unconditionally inside the DWPlusFP/DWTimesFP shapes below, exactly
/// as Joldes et al. specify them: the composite algorithms' error bounds are
/// proved for THIS sequence, including the inputs where the magnitude
/// precondition is marginally violated — so no runtime ordering check is
/// performed (and the WGSL twin performs none either; a divergent branch
/// would also break the M1 bit-compare).
///
/// WGSL twin: `fast_two_sum` in `ds_primitives.wgsl` — there the two
/// subtractions are routed through `fma(-1.0, x, y)` barriers; here plain ops
/// are exact under Rust's strict f32 semantics, and the results are
/// bit-identical because each step performs the same single rounding of the
/// same real value.
#[inline]
pub(crate) fn fast_two_sum(a: f32, b: f32) -> (f32, f32) {
    let s = a + b;
    let bb = s - a; // exact under the precondition
    let t = b - bb;
    (s, t)
}

/// Renormalize a (possibly overlapping) pair into the ds invariant.
#[inline]
pub(crate) fn ds_renorm(hi: f32, lo: f32) -> Ds {
    let (h, l) = fast_two_sum(hi, lo);
    Ds { hi: h, lo: l }
}

/// ds + f32 (DWPlusFP, Joldes et al. Algorithm 4). Relative error `<= 2u^2`.
///
/// The EFT identity backing it — `a + b = s + t` EXACTLY for
/// [`two_sum_f32`] — is doc-tested here. Note the check is rigorous: both
/// sides of the assert round THE SAME real value once into f64, so equality
/// of the f64 sums is implied by exactness of the transformation.
///
/// ```
/// use ny_core::eft::two_sum_f32;
/// for &(a, b) in &[(1.0f32, 1e-8f32), (1e30, -1e30), (3.0, 1.0 / 3.0)] {
///     let (s, t) = two_sum_f32(a, b);
///     // a + b == s + t as REALS => their f64 roundings agree.
///     assert_eq!(f64::from(a) + f64::from(b), f64::from(s) + f64::from(t));
/// }
/// ```
#[inline]
pub(crate) fn ds_add_f32(x: Ds, b: f32) -> Ds {
    let (s, t) = two_sum_f32(x.hi, b);
    ds_renorm(s, t + x.lo)
}

/// ds + ds (ACCURATE DWPlusDW, Joldes et al. Algorithm 6). Relative error
/// `<= 3u^2/(1-4u) < 2^-45.4` — unconditional, cancellation included.
///
/// Deliberately NOT the cheaper "sloppy" Algorithm 5 (one `two_sum` + a plain
/// lo add): its relative error is UNBOUNDED under near-cancellation of the hi
/// parts — and near-cancellation is exactly the CROWN backward regime. The
/// mass-relative envelope would still hold, but the result-relative [`U_DS`]
/// contract this module states would be false. Three extra flops buy a
/// theorem.
#[inline]
pub(crate) fn ds_add(x: Ds, y: Ds) -> Ds {
    let (sh, sl) = two_sum_f32(x.hi, y.hi);
    let (th, tl) = two_sum_f32(x.lo, y.lo);
    let c = sl + th;
    let (vh, vl) = fast_two_sum(sh, c);
    let w = tl + vl;
    ds_renorm(vh, w)
}

/// ds * ds: one-fma DWTimesDW in the `xl*yl`-DROPPING shape (Joldes et al.'s
/// DWTimesDW2 — Algorithm 12 WITHOUT its `xl*yl` term). Published relative
/// error for THIS shape: `<= 6u^2`. TRAP: the tighter `5u^2` bound
/// sometimes quoted belongs to the
/// `xl*yl`-CARRYING Algorithm 12 — a DIFFERENT sequence — and must not be
/// imported here: a "tighten 10u^2 -> 9u^2/tap" edit derived from `5u^2 + 3u^2`
/// would UNDERCHARGE [`DS_TAP_EPS`], which correctly charges the `6u^2`
/// reading. (`U_DS = 16u^2` dominates every published DWTimesDW variant,
/// `<= 7u^2` worst, with >= 2.2x headroom.)
///
/// Exists for the GATE parameters (design M2): the gate slopes/intercepts are
/// f64 on the CPU lane, and certifying the SAME relaxation on device — the
/// precondition of the M2 coefficient-overlap assertion — requires carrying
/// them as ds pairs (exact-split residual `<= 2^-48` relative in the normal
/// band, charged; the subnormal band rides the transaction's split-floor
/// channel) and
/// multiplying in ds, not substituting a downcast relaxation.
///
/// WGSL twin: `ds_mul` in `ds_primitives.wgsl`, op-for-op.
#[inline]
pub(crate) fn ds_mul(x: Ds, y: Ds) -> Ds {
    let (ph, e1) = two_prod_f32(x.hi, y.hi);
    let tl = x.hi * y.lo;
    let cl2 = x.lo.mul_add(y.hi, tl);
    let cl3 = e1 + cl2;
    let (h, l) = fast_two_sum(ph, cl3);
    Ds { hi: h, lo: l }
}

/// ds * f32 (DWTimesFP with fma, Joldes et al. Algorithm 9). Relative error
/// `<= 2u^2`.
///
/// The EFT identity backing it — `a * b = p + e` EXACTLY for
/// [`two_prod_f32`] away from underflow — is doc-tested: f32*f32 is exact in
/// f64 (48 < 53 bits), and `p + e` is that same exactly-representable value.
///
/// ```
/// use ny_core::eft::two_prod_f32;
/// for &(a, b) in &[(3.0f32, 1.0f32 / 3.0), (1e10, 1e-10), (-7.0, 0.142_857_15)] {
///     let (p, e) = two_prod_f32(a, b);
///     assert_eq!(f64::from(a) * f64::from(b), f64::from(p) + f64::from(e));
/// }
/// ```
///
/// UNDERFLOW CAVEAT (design §4.1): in the band guarded by
/// `ny_core::eft::TWO_PROD_EXACT_FLOOR_F32` (`|p| < 2^-101`) the residual is
/// itself rounded and the identity fails; the device path charges the
/// absolute `2^-126` floor there. Margin-row coefficients at that magnitude
/// contribute nothing to any bound at margin scale, so the floor is
/// verdict-invisible — but it must be CHARGED, not assumed away.
#[inline]
pub(crate) fn ds_mul_f32(x: Ds, w: f32) -> Ds {
    let (p, e) = two_prod_f32(x.hi, w);
    ds_renorm(p, x.lo.mul_add(w, e))
}

/// Reference ds dot product `sum_j w_j * a_j` with SCALAR f32 weights — the
/// M0 enclosure reference for the ds fold shape. (Since the error-charge
/// session the PRODUCTION conv/gemm accumulate with ds-PAIR weights via
/// [`ds_mul`]; the per-tap envelope both shapes obey is [`gamma_ds_dot`],
/// and the pair form is enclosure-tested through the kernel twins.)
///
/// Returns `None` on any non-finite intermediate (fail-closed, mirroring
/// `ny_core::eft::eft_dot_f32`).
pub(crate) fn ds_dot(w: &[f32], a: &[f32]) -> Option<Ds> {
    debug_assert_eq!(w.len(), a.len());
    let mut acc = Ds::from_f32(0.0);
    for (&wi, &ai) in w.iter().zip(a) {
        let term = ds_mul_f32(Ds::from_f32(ai), wi);
        if !term.hi.is_finite() {
            return None;
        }
        acc = ds_add(acc, term);
        if !acc.hi.is_finite() {
            return None;
        }
    }
    Some(acc)
}

/// Host twin of the WGSL `bias_reduce_wg` TREE reduce (B2), OP-FOR-OP:
/// lane `l` folds the contiguous block `[l*chunk, min((l+1)*chunk, n))`
/// serially (same per-element body as the serial twin), then the 256 lane
/// partials combine by the fixed binary tree (offset 128, 64, .., 1). This
/// order is the bit-parity contract of the device kernel — any edit here
/// must move in lock-step with `kernels/bias_reduce.wgsl::bias_reduce_wg`.
///
/// Soundness role: comparison reference only (nothing here feeds a
/// verdict); the charge argument for why `gamma_ds_dot(n+2)` dominates the
/// TREE fold too lives at `transaction.rs::push_reduce`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn bias_reduce_wg_twin(
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
    const LANES: usize = 256;
    for r in 0..rows {
        let chunk = n.div_ceil(LANES);
        let mut lane_acc = [Ds { hi: 0.0, lo: 0.0 }; LANES];
        let mut lane_esum = [0.0f32; LANES];
        let mut lane_msum = [0.0f32; LANES];
        for (lane, ((acc_slot, esum_slot), msum_slot)) in lane_acc
            .iter_mut()
            .zip(lane_esum.iter_mut())
            .zip(lane_msum.iter_mut())
            .enumerate()
        {
            let j0 = lane * chunk;
            let j1 = (j0 + chunk).min(n);
            let mut acc = Ds { hi: 0.0, lo: 0.0 };
            let mut esum = 0.0f32;
            let mut msum = 0.0f32;
            for j in j0..j1 {
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
            *acc_slot = acc;
            *esum_slot = esum;
            *msum_slot = msum;
        }
        let mut off = LANES / 2;
        while off > 0 {
            for lane in 0..off {
                lane_acc[lane] = ds_add(lane_acc[lane], lane_acc[lane + off]);
                lane_esum[lane] += lane_esum[lane + off];
                lane_msum[lane] += lane_msum[lane + off];
            }
            off /= 2;
        }
        let nb = ds_add(bias_acc[r], lane_acc[0]);
        bias_acc[r] = nb;
        let carry = 2.0 * U_DS_F32 * ds_abs_bound(nb);
        ebias_acc[r] = ebias_acc[r] + lane_esum[0] + g_reduce * lane_msum[0] + carry;
    }
}

/// Host twin of the WGSL `collect_abs` kernel (B5), OP-FOR-OP: per position
/// `p`, fold `|coeff[idx[p] * rows + r]|` over `r = 0..rows` in ds, taking
/// the magnitude by negating BOTH ds components (exact in every band).
///
/// HEURISTIC ONLY — like the kernel it mirrors, this feeds no verdict: the
/// published `PassOut::coll` is read only by the trunk shortlist and the
/// tier-0 pool (both rankers). It exists so the kernel has a bit-parity
/// reference on the running adapter, exactly as `bias_reduce_wg_twin` does.
pub(crate) fn collect_abs_twin(idx: &[usize], coeff: &[Ds], rows: usize) -> Vec<Ds> {
    let mut out = Vec::with_capacity(idx.len());
    for &j in idx {
        let mut acc = Ds { hi: 0.0, lo: 0.0 };
        for r in 0..rows {
            let v = coeff[j * rows + r];
            let a = if v.hi >= 0.0 {
                v
            } else {
                Ds {
                    hi: -v.hi,
                    lo: -v.lo,
                }
            };
            acc = ds_add(acc, a);
        }
        out.push(acc);
    }
    out
}

/// M1 harness core: bit-compare a host ds stream against a device `(hi, lo)`
/// readback. `Ok(())` iff every pair agrees BIT-FOR-BIT (NaN payloads and
/// signed zeros included — `to_bits`, not `==`); otherwise `Err(i)` with the
/// first diverging index, where a length mismatch diverges at the shorter
/// length. There is deliberately no tolerance parameter: on an adapter where
/// the primitives are bit-exact, ANY divergence is a channel-killing signal,
/// and a tolerance here would be a soundness hole (design section 7, M1).
#[allow(dead_code)] // fed by device readbacks in the implementation session
pub(crate) fn bit_compare_streams(host: &[Ds], device: &[(f32, f32)]) -> Result<(), usize> {
    let n = host.len().min(device.len());
    for (i, (h, d)) in host.iter().zip(device).enumerate().take(n) {
        if h.hi.to_bits() != d.0.to_bits() || h.lo.to_bits() != d.1.to_bits() {
            return Err(i);
        }
    }
    if host.len() != device.len() {
        return Err(n);
    }
    Ok(())
}

/// A-priori relative envelope for a length-`n` [`ds_dot`]: `gamma`-shaped with
/// the unit swapped for [`U_DS`] (design §4.2, "ds algebra's own residue").
///
/// `2n` ds ops (n multiplies, n adds), each `<= U_DS` relative, composed the
/// standard Higham way. SELF-SUFFICIENT domination (adversarial-review minor
/// note resolved — the old one-ulp-up relied on an UNENFORCED caller-side
/// publication inflation): in the non-saturated range `nu = 2n·2^-44` and
/// `1 - nu` are computed EXACTLY (power-of-two unit; the subtraction aligns
/// within 53 bits for every `n` below the saturation cut), so only the
/// division and the inflation multiply round — the `(1 + 2^-50)` factor
/// (4 f64 units) dominates those two roundings with >= 2x slack, and the
/// final one-ulp-up absorbs the residue, so the returned value is `>=` the
/// exact `nu/(1-nu)` with NO help from any caller. Callers apply no further
/// inflation. Saturates to 1.0 (degrade-to-useless, the NaN/Inf firewall
/// then refuses) for absurd `n` rather than undercharging.
///
/// ```
/// // Domination is self-contained: nu and 1-nu are exact, so fl(nu/(1-nu))
/// // is ONE rounding from the true ratio and anything >= 2 ulps above the
/// // fl value strictly dominates the true value. (Executable copy in
/// // tests.rs — this module is pub(crate), rustdoc does not run this.)
/// let n = 4608usize;
/// let nu = (2 * n) as f64 * 2f64.powi(-44);
/// let fl = nu / (1.0 - nu);
/// // assert!(gamma_ds(n) >= f64::from_bits(fl.to_bits() + 2));
/// ```
#[inline]
pub(crate) fn gamma_ds(n: usize) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    let nu = (2 * n) as f64 * U_DS;
    gamma_from_nu(nu)
}

/// Per-TAP (one [`ds_mul`] + one accurate [`ds_add`]) relative bound for the
/// production dot shape: `10 * 2^-48 = 10 u^2`, `u = 2^-24`. Exact in f64
/// (3-bit mantissa: `5 * 2^-47`).
///
/// DOCUMENTED constituents (design §4):
/// ACCURATE DWPlusDW (the [`ds_add`] shape) `<= 3u^2/(1-4u) < 3.0000008u^2`
/// unconditional; one-fma DWTimesDW in the `xl*yl`-DROPPING DWTimesDW2
/// shape (the [`ds_mul`] sequence) has published bound `6u^2` — the `5u^2`
/// sometimes quoted is the `xl*yl`-CARRYING Algorithm 12's, a different
/// sequence (see the [`ds_mul`] trap note) — so the per-tap charge is
/// `6u^2 + 3.0000008u^2 < 9.0000008u^2`,
/// charged at `10u^2` (>= 11% headroom). This replaces the blanket
/// 2-ops-at-`U_DS` accounting (`32u^2`/tap) of [`gamma_ds`]: a 3.2x
/// tightening of the a-priori dot envelope at identical soundness (the
/// constituent bounds are unconditional, cancellation included; underflow is
/// discharged by the §4.1 flush floors exactly as for [`U_DS`]).
pub(crate) const DS_TAP_EPS: f64 = 3.552_713_678_800_501e-14; // 10 * 2^-48, exact

/// A-priori relative envelope for a `taps`-tap ds dot in the PRODUCTION shape
/// (per tap: one [`ds_mul`] of a ds coefficient by a ds weight, one accurate
/// [`ds_add`] into the running sum): `nu/(1-nu)` with `nu = taps * DS_TAP_EPS`.
///
/// Derivation (mass-relative, the standard Higham composition): each tap's
/// multiply rounds once relative to ITS term (`<= 6u^2 |t_i|`, the
/// conservative DWTimesDW reading) and each add rounds relative to the
/// RESULT partial sum (`<= 3.0000008u^2 |s_i|`, unconditional —
/// result-relative is STRONGER than mass-relative), and both `|t_i|` and
/// `|s_i|` are `<=` the absolute mass `sum |t_j|`, so the first-order total
/// is `<= taps * DS_TAP_EPS * mass`; the `nu/(1-nu)` form absorbs the cross
/// terms exactly as `gamma_n` does. Callers add `+2` taps of slack for the
/// zero-init and any final renormalize, mirroring the CPU arms'
/// `gamma_n(k + 2)`.
///
/// Self-domination is inherited from [`gamma_from_nu`]: `nu` is EXACT
/// (`DS_TAP_EPS` has a 5-bit mantissa, so the product is exact for every
/// `taps < 2^48`), `1 - nu` is exact in the non-saturated range, and the
/// `(1 + 2^-50)` inflation plus the final ulp-up dominate the division's and
/// the inflation's own roundings with no caller help.
#[inline]
pub(crate) fn gamma_ds_dot(taps: usize) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    let nu = taps as f64 * DS_TAP_EPS;
    gamma_from_nu(nu)
}

/// Shared self-dominating `nu -> up(nu/(1-nu))` core of [`gamma_ds`] /
/// [`gamma_ds_dot`] (see their doc comments for the exactness argument).
/// Saturates to 1.0 (degrade-to-useless; the NaN/Inf firewall then refuses)
/// rather than undercharging.
#[inline]
fn gamma_from_nu(nu: f64) -> f64 {
    if nu >= 0.5 {
        return 1.0;
    }
    // Inflated up-rounding of nu/(1-nu); same shape as `rounding::gamma_n`
    // but self-dominating (see the doc comments).
    let g = (nu / (1.0 - nu)) * (1.0 + 2f64.powi(-50));
    f64::from_bits(g.to_bits() + 1)
}
