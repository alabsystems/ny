//! EAGER discharge of the certified patches coefficient error over a box
//! (#patches-eager-err), the Patches analogue of
//! [`LinearBounds::fold_coeff_err_over_box_eager`](crate::bounds::LinearBounds::fold_coeff_err_over_box_eager).
//!
//! # Why this exists
//!
//! The dense CROWN path discharges its carried `coeff_err` right after every
//! elementwise activation backward step, against the activation's
//! (CROWN-tightened) PRE-ACTIVATION cut — the tightest box that error will ever
//! see. The Patches path had no such fold: a conv stack carried `coeff_err`
//! all the way to the network input, where it is discharged against the input
//! box after ABS-composition through every remaining layer. On a deep conv
//! stack that abs-composition grows at IBP scale, so the penalty that lands in
//! the bias is exponentially larger than the one the dense path pays for the
//! very same network — the Patches representation was being charged for its own
//! depth. Discharging early costs `err[i] · Σ_j mag_j` over the row's own
//! receptive window and nothing downstream.
//!
//! # Enclosure
//!
//! `coeff_err` is a per-logical-ROW bound: every stored coefficient in row `i`
//! satisfies `|stored − true| ≤ err[i]` (`PatchesData::coeff_err`). For true
//! coefficients `Ã` and any `y` in the box the columns multiply,
//!
//! ```text
//!   |Ã_i·y − A_i·y| = |Σ_j (Ã_ij − A_ij)·y_j| ≤ err[i] · Σ_j mag_j
//! ```
//!
//! with `mag_j = max(|l_j|, |u_j|)` and `j` ranging over exactly the row's
//! stored taps (every other column is structurally zero in BOTH `Ã` and `A`, so
//! it contributes nothing). Folding that penalty OUTWARD into the bias
//! (`lower_b −= p`, `upper_b += p`) therefore preserves the enclosure for every
//! admissible `Ã`, and the row's coefficients become exact — the same fold
//! identity the dense side uses, restricted to the patch support.
//!
//! Soundness details:
//! - the window sum is accumulated in f64 and inflated by `γ_n^f64` before use,
//!   so the accumulation's own rounding is covered;
//! - the fold into the f32 bias rounds outward (`next_down_f32`/`next_up_f32`);
//! - rows whose penalty is NON-FINITE (unbounded box, or an `+INF`-poisoned
//!   error entry) keep their error and continue to carry — byte-identical to
//!   the prior behavior for those rows, never a new degrade;
//! - the 7D explicit-rows layout is modeled behind its own opt-in policy
//!   (`NY_PATCHES_EAGER_ERR_7D=1`, see [`eager_err_7d_enabled`]); with the
//!   policy off it keeps carrying, byte-identical to before;
//! - shapes this routine does not model exactly (sparse 4D/5D, identity,
//!   non-contiguous box) are left untouched and keep carrying.

use ndarray::ArrayD;
use ny_core::dd::gamma_n_f64;
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use super::scatter::{build_unfold_plan, compute_unfold_index_map};
use super::types::PatchesData;
use super::PatchesLinearBounds;

/// DEFAULT-ON; disable with `NY_PATCHES_EAGER_ERR=0` (repo convention:
/// default-on features carry a kill switch).
///
/// The original blocker was that the BENEFIT was unpriced: on the benchmark it
/// targets (CIFAR100_resnet_medium) the root pipeline never reaches the point
/// where the bound is consumed, so neither a verdict A/B nor the
/// `NY_CROWN_GAIN` width probe could price it. That is no longer true for
/// yolo_2023 — see the measurement below.
///
/// # MEASURED ON CIFAR100 (2026-08-06): converts nothing, and here is why
///
/// This gate is default-ON now, so the obvious question is whether it helps the
/// benchmark it was written for. It does not. Four `CIFAR100_resnet_medium` rows
/// at the official 100 s budget, same binary, gate toggled by env only:
///
/// ```text
///   row  eager ON   eager OFF
///   1    sat        sat
///   2    timeout    timeout
///   3    timeout    timeout
///   4    unsat      unsat
/// ```
///
/// Byte-identical: zero differing WARN lines between arms on every row.
///
/// The mechanism, and it is decisive: **cifar100 does not route through the
/// patches path at all** — `patches` appears zero times in a full `-v` run log.
/// This gate is checked at the patches ReLU backward call sites, so on a
/// benchmark that never takes that route it cannot engage, whatever its value
/// elsewhere. The original docstring's instinct was right and the reason is
/// sharper than "the bound is never consumed": the lane is never entered.
///
/// So this is NOT the tighter-intermediate-bound lever cifar100 needs. Whatever
/// closes that deficit has to act on the route cifar100 actually takes.
///
/// # PRICED (#patches-eager-err-pricing, 2026-07-31)
///
/// `yolo_2023 / TinyYOLO_prop_000001_eps_1_255` prices it decisively. Same
/// binary, same host, gate toggled by env — measured, not projected:
///
/// ```text
///   NY_PATCHES_EAGER_ERR unset/1  ->  unsat in 62 s
///   NY_PATCHES_EAGER_ERR=0        ->  timeout (300 s budget)
/// ```
///
/// The deciding row is output 269, which needs `lower(Y[269]) > -1`; carrying
/// leaves it at −2684.44, discharging eagerly reaches +0.613 (truth +1.675).
/// The mechanism is the one the module docs describe: the carried per-row
/// scalar is tiny (3.7e-5 at `Add_15`) but is multiplied ×2 per ReLU row-lift
/// and ×‖k‖₁ per conv carry, so by the input box it is ~93% of the node's
/// width — the node's CROWN ends up WIDER than its own IBP.
///
/// # What the default flip moved, and how each was handled
///
/// Seven tests, and they were not all the same kind of thing:
///
///   * FIVE were exact-value / structural pins that moved in the SOUND
///     direction — at the fold site the bias is folded OUTWARD, so each pinned
///     lower bound got LOWER (e.g. `-8.375001 -> -8.375008`,
///     `-1.7500006 -> -1.7500013`) and `coeff_err` became `None` because the
///     carry has been discharged. Locally wider, globally tighter, which is the
///     whole point. Re-pinned, each with the direction checked.
///   * TWO are NOT value pins.
///     `test_patches_alpha_optimized_matches_indexed_reference_bit_identical`
///     and `test_patches_alpha_optimized_identity_input_matches_reference`
///     compare the optimized alpha kernel against a hand-written REFERENCE
///     implementation in the test file, bit for bit. That reference models the
///     KERNEL; the fold is a policy applied at the call sites wrapped around it.
///     Teaching the reference to fold would make the moat circular — it would
///     then assert only that the implementation equals itself. They now pin the
///     unfolded kernel explicitly via
///     [`test_override::with_eager_err`], which preserves exactly what they were
///     written to assert.
///
/// The folded configuration is covered instead by the property that actually
/// matters once this is default-on:
/// `eager_fold_never_tightens_against_the_unfolded_path` asserts that folding
/// never produces a tighter bound than not folding, on the same fixtures. A
/// tightening there would be a false proof; a widening is merely weaker.
///
/// What IS established: the fold is sound (see the module docs and
/// `eager_fold_preserves_enclosure_against_perturbed_coefficients`), and
/// `NY_ERR_SPLIT` measured 93-100% of the coefficient error emitted at a conv
/// node to be CARRY — exactly the term this retires.
///
/// Checked at the POLICY call sites (the patches ReLU backward steps), not
/// inside the fold itself, so the fold's own unit tests exercise it directly
/// regardless of how the test process was launched.
///
/// `NY_PATCHES_EAGER_ERR_7D=1` (see [`eager_err_7d_enabled`]) is a SUPERSET
/// switch: it also enables the fold call here, so the 7D discharge engages
/// without requiring both variables. With only `NY_PATCHES_EAGER_ERR=1` the
/// behavior is byte-identical to before the 7D extension existed (the 7D
/// layout keeps carrying).
pub(crate) fn eager_err_enabled() -> bool {
    #[cfg(test)]
    {
        // #eager-err-test-override: the process-wide `OnceLock` below caches the
        // first read, so an env var cannot be toggled per-test. Tests that must
        // pin the UNFOLDED kernel -- the optimized-vs-indexed-reference
        // equivalence moats -- use `with_eager_err(false, ...)` instead.
        if let Some(forced) = test_override::current() {
            return forced;
        }
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("NY_PATCHES_EAGER_ERR").ok().as_deref() != Some("0"))
        || eager_err_7d_enabled()
}

/// Test-only, thread-local override for [`eager_err_enabled`]
/// (#eager-err-test-override).
///
/// This exists for ONE purpose: the two equivalence moats
/// (`test_patches_alpha_optimized_matches_indexed_reference_bit_identical`,
/// `test_patches_alpha_optimized_identity_input_matches_reference`) compare the
/// optimized alpha kernel against a hand-written reference implementation in the
/// test file. That reference deliberately does NOT model the eager fold, because
/// the fold is a POLICY applied at the patches ReLU backward call sites, wrapped
/// around the kernel the moat is about.
///
/// Teaching the reference to fold would make the moat circular -- it would then
/// assert only that the implementation equals itself. Running the moat with the
/// fold off keeps it asserting exactly what it was written to assert. The folded
/// configuration is covered separately by
/// `eager_fold_never_tightens_against_the_unfolded_path`, which checks the
/// property that actually matters once the fold is default-on: that folding only
/// ever widens.
#[cfg(test)]
pub(crate) mod test_override {
    use std::cell::Cell;

    thread_local! {
        static FORCED: Cell<Option<bool>> = const { Cell::new(None) };
    }

    pub(crate) fn current() -> Option<bool> {
        FORCED.with(Cell::get)
    }

    /// Run `body` with [`super::eager_err_enabled`] forced to `enabled`.
    ///
    /// Restores the previous value on the way out, including on unwind, so one
    /// failing test cannot silently reconfigure the rest of the binary.
    pub(crate) fn with_eager_err<T>(enabled: bool, body: impl FnOnce() -> T) -> T {
        struct Restore(Option<bool>);
        impl Drop for Restore {
            fn drop(&mut self) {
                FORCED.with(|c| c.set(self.0));
            }
        }
        let _restore = Restore(FORCED.with(Cell::get));
        FORCED.with(|c| c.set(Some(enabled)));
        body()
    }
}

/// OPT-IN (`NY_PATCHES_EAGER_ERR_7D=1`) — admits the 7D explicit-rows layout
/// into the eager fold (#patches-eager-err-7d), default-off.
///
/// Why a separate gate: the 7D explicit-rows layout is exactly the carrier the
/// cifar100/tinyimagenet ResNet pool runs (Dense→Patches re-entry spec rows),
/// and it is the layout whose carried error compounds worst — ×2 per ReLU
/// row-lift, ×‖k‖₁ per conv carry, ×(taps per spec row, ~1e6 at Add_28-scale
/// junctions) at densification
/// (docs/ADD28_COEFF_ERR_AND_PATCHES_SENTINEL_DIAGNOSIS_2026-07-30.md §2).
/// Discharging at the post-activation compose point retires the dominant
/// (93-100% carry) component before those multipliers see it. Kept behind its
/// own default-off flag until the tightness gain is priced on the scored
/// pools, mirroring the `NY_PATCHES_EAGER_ERR` gate rationale above.
///
/// Like the parent gate this is read at the policy boundary
/// ([`PatchesLinearBounds::fold_coeff_err_over_box_eager`] delegating to
/// `..._with_policy`), so unit tests drive the policy explicitly.
pub(crate) fn eager_err_7d_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("NY_PATCHES_EAGER_ERR_7D").ok().as_deref() == Some("1"))
}

/// Per-row receptive-window magnitude sums for a 6D-broadcast `PatchesData`,
/// or (behind `allow_7d`) per-spec-row slab magnitude sums for the 7D
/// explicit-rows layout.
///
/// `window[out_flat]` over-bounds `Σ_j mag_j` across exactly the taps stored in
/// logical row `out_flat`. Returns `None` when the layout is not a shape this
/// fold models exactly (or the 7D policy is off) — those keep carrying.
fn window_mag_sums(
    data: &PatchesData,
    mag: &[f32],
    row_count: usize,
    allow_7d: bool,
) -> Option<Vec<f64>> {
    if data.identity || data.unstable_idx.is_some() {
        return None;
    }
    let patches: &ArrayD<f32> = data.patches.as_ref()?;
    let shape = patches.shape();
    if shape.len() != 6 {
        if shape.len() == 7 && allow_7d {
            return window_mag_sums_7d(data, mag, row_count, shape);
        }
        // 7D with the policy off, or sparse: not folded here, keep carrying.
        return None;
    }
    let (out_c, out_h, out_w) = data.output_shape;
    let (in_c, in_h, in_w) = data.input_shape;
    if out_c.checked_mul(out_h)?.checked_mul(out_w)? != row_count {
        return None;
    }
    let in_dim = in_c.checked_mul(in_h)?.checked_mul(in_w)?;
    if in_dim != mag.len() {
        return None; // box does not match the columns these patches multiply
    }
    let (kh, kw) = (shape[4], shape[5]);
    let index_map = compute_unfold_index_map(data, kh, kw).ok()?;
    let plan = build_unfold_plan(&index_map);

    let positions = out_h.checked_mul(out_w)?;
    let mut sums = vec![0.0f64; row_count];
    for (out_flat, slot) in sums.iter_mut().enumerate() {
        let pos = out_flat % positions;
        let taps = plan.taps_for(pos / out_w, pos % out_w);
        let mut acc = 0.0f64;
        for &(_, in_flat) in taps {
            // `in_flat` is produced by the same unfold geometry that built the
            // patch block, so it indexes the box the columns multiply.
            acc += f64::from(mag.get(in_flat).copied().unwrap_or(f32::INFINITY));
        }
        // Cover the f64 accumulation itself (Higham γ_n·Σ|terms|; all terms ≥ 0
        // so Σ|terms| = acc). Non-finite acc stays non-finite → row keeps carrying.
        *slot = acc * (1.0 + gamma_n_f64(taps.len().max(1)));
    }
    Some(sums)
}

/// Per-spec-row occurrence-weighted magnitude sums for the 7D explicit-rows
/// layout `patches[r, oc, oh, ow, ic, ki, kj]` (#patches-eager-err-7d).
///
/// # Enclosure (the inequality this fold rests on)
///
/// Spec row `r` owns EVERY occurrence `q = (oc, oh, ow, ic, ki, kj)`. Only
/// in-range occurrences — the unfold plan's taps, the identical geometry the
/// densification uses (`scatter_rows_err_accumulators`) — ever materialize
/// into the dense row: an out-of-range tap is dropped by BOTH the stored walk
/// and the exact walk, so it contributes nothing to either `A_r·y` or `Ã_r·y`.
///
/// Contract (`PatchesData::coeff_err`): `|Ã[r,q] − A[r,q]| ≤ err[r]` for every
/// stored occurrence `q`. For any `y` in the box (`|y_j| ≤ mag_j =
/// max(|l_j|, |u_j|)`), with `j(q)` the input column occurrence `q` multiplies:
///
/// ```text
///   |Ã_r·y − A_r·y| = |Σ_{q in-range} (Ã[r,q] − A[r,q]) · y_{j(q)}|
///                   ≤  Σ_{q in-range} |Ã[r,q] − A[r,q]| · |y_{j(q)}|
///                   ≤  err[r] · Σ_{q in-range} mag_{j(q)}  =:  err[r] · S_r
/// ```
///
/// `S_r` is an OCCURRENCE sum, with multiplicity: overlapping taps repeat
/// `mag_{j(q)}` exactly as the densification's `count[r,j]` repeats `err[r]` —
/// never a per-unique-column sum. Hence for every admissible `Ã` and every `y`
/// in the box:
///
/// ```text
///   A_r·y + (b_lo[r] − err[r]·S_r)  ≤  Ã_r·y + b_lo[r]
///   Ã_r·y + b_up[r]                 ≤  A_r·y + (b_up[r] + err[r]·S_r)
/// ```
///
/// so widening the bias OUTWARD by `err[r]·S_r` and zeroing `err[r]` preserves
/// the enclosure `true_bound(y) ∈ [A·y + b_lo', A·y + b_up']` — the 6D fold
/// identity (module docs) extended from one spatial window to the whole
/// spec-row slab. Error created AFTER this fold (conv contraction intrinsic,
/// densification `γ·absacc`) is untouched and still certified by its own
/// producers.
///
/// # Row-independence and rounding
///
/// Every spec row stores the same full `out_c × out_h × out_w` grid, so `S_r`
/// is identical for all rows: one f64 spatial pass (`base`, T occurrences per
/// output channel), one multiply by `out_c` (an exact f64 integer). Rounding
/// cover: the T-term same-sign f64 sum has relative error ≤ γ_{T−1}; the single
/// multiply adds ≤ γ_1 (exact when `out_c == 1`); `(1+γ_{T−1})(1+γ_1) ≤ 1+γ_T
/// ≤ 1+γ_N` with `N = out_c·T ≥ T` occurrences, so inflating by `γ_N` (below)
/// over-bounds both. Non-finite sums poison the row and it keeps carrying
/// (`fold_side` checks the penalty).
fn window_mag_sums_7d(
    data: &PatchesData,
    mag: &[f32],
    row_count: usize,
    shape: &[usize],
) -> Option<Vec<f64>> {
    if data.identity || data.unstable_idx.is_some() {
        return None;
    }
    let (out_c, out_h, out_w) = data.output_shape;
    let (in_c, in_h, in_w) = data.input_shape;
    if data.coeff_err.as_ref()?.len() != row_count {
        return None; // a row-wide certificate must have exactly one entry per spec row
    }
    if shape[0] != row_count
        || shape[1] != out_c
        || shape[2] != out_h
        || shape[3] != out_w
        || shape[4] != in_c
    {
        return None; // metadata disagrees with the tensor: keep carrying
    }
    let in_dim = in_c.checked_mul(in_h)?.checked_mul(in_w)?;
    if in_dim != mag.len() {
        return None; // box does not match the columns these patches multiply
    }
    let (kh, kw) = (shape[5], shape[6]);
    let index_map = compute_unfold_index_map(data, kh, kw).ok()?;
    if index_map.shape() != [out_h, out_w, in_c, kh, kw] {
        return None; // patch geometry metadata does not produce the declared output grid
    }
    let plan = build_unfold_plan(&index_map);

    let mut base = 0.0f64;
    let mut spatial_occurrences = 0usize;
    for oh in 0..out_h {
        for ow in 0..out_w {
            let taps = plan.taps_for(oh, ow);
            spatial_occurrences = spatial_occurrences.checked_add(taps.len())?;
            for &(_, in_flat) in taps {
                // Same unfold geometry as the 7D scatter, so `in_flat` indexes
                // the box the columns multiply; a missing entry poisons the sum
                // (row keeps carrying), never silently under-counts.
                base += f64::from(mag.get(in_flat).copied().unwrap_or(f32::INFINITY));
            }
        }
    }
    let occurrences = out_c.checked_mul(spatial_occurrences)?;
    // `base·out_c` then the γ_N inflation per the header comment. A non-finite
    // base stays non-finite (INF, or NaN via `INF·0` when out_c == 0) → every
    // row keeps carrying, the fail-closed direction.
    let sum = base * out_c as f64 * (1.0 + gamma_n_f64(occurrences.max(1)));
    Some(vec![sum; row_count])
}

/// Fold one side's carried error into its bias, outward, and clear the entries
/// that were discharged. Returns `true` if every entry was discharged.
fn fold_side(data: &mut PatchesData, bias: &mut [f32], sums: &[f64], subtract: bool) -> bool {
    let Some(err) = data.coeff_err.as_mut() else {
        return true;
    };
    let mut all_clear = true;
    for (i, b) in bias.iter_mut().enumerate() {
        let Some(e) = err.get_mut(i) else {
            all_clear = false;
            continue;
        };
        let ev = f64::from(*e);
        if ev.is_nan() || ev <= 0.0 {
            // Zero (or a sanitized non-positive) entry: nothing to discharge.
            // A negative entry is never legal; leave it for the consumer's
            // sanitize rather than silently reinterpreting it here.
            if ev.is_nan() || ev < 0.0 {
                all_clear = false;
            } else {
                *e = 0.0;
            }
            continue;
        }
        let p = ev * sums.get(i).copied().unwrap_or(f64::INFINITY);
        if !p.is_finite() {
            all_clear = false; // keep carrying: prior behavior for this row
            continue;
        }
        *b = if subtract {
            next_down_f32((f64::from(*b) - p) as f32)
        } else {
            next_up_f32((f64::from(*b) + p) as f32)
        };
        *e = 0.0;
    }
    all_clear
}

impl PatchesLinearBounds {
    /// EAGERLY discharge the certified per-row coefficient error over
    /// `input_box` — the box this object's COLUMNS multiply (i.e. the
    /// pre-activation cut, when called right after an activation backward
    /// step).
    ///
    /// No-op when nothing is carried, and a no-op (still sound: the error keeps
    /// carrying, exactly as before) for any layout this fold does not model
    /// exactly. See the module docs for the enclosure argument.
    ///
    /// The 7D explicit-rows layout is admitted per the
    /// [`eager_err_7d_enabled`] policy; tests drive the policy explicitly via
    /// [`fold_coeff_err_over_box_eager_with_policy`](Self::fold_coeff_err_over_box_eager_with_policy)
    /// (same idiom as `compose_*_with_policy`).
    pub(crate) fn fold_coeff_err_over_box_eager(&mut self, input_box: &BoundedTensor) {
        self.fold_coeff_err_over_box_eager_with_policy(input_box, eager_err_7d_enabled());
    }

    /// Policy-explicit body of
    /// [`fold_coeff_err_over_box_eager`](Self::fold_coeff_err_over_box_eager):
    /// `allow_7d` admits the 7D explicit-rows layout
    /// ([`window_mag_sums_7d`]; `false` keeps 7D carrying, byte-identical to
    /// the pre-extension behavior). The 6D broadcast layout folds regardless,
    /// exactly as before.
    pub(crate) fn fold_coeff_err_over_box_eager_with_policy(
        &mut self,
        input_box: &BoundedTensor,
        allow_7d: bool,
    ) {
        if self.lower_a.coeff_err.is_none() && self.upper_a.coeff_err.is_none() {
            return;
        }
        // Both sides are folded in one infallible transaction. If their exact
        // mapping differs (including anchored-origin arrays), refuse before
        // touching either bias or certificate.
        if self
            .lower_a
            .validate_common_geometry(&self.upper_a)
            .is_err()
        {
            return;
        }
        let row_count = self.row_count;
        // This API is deliberately infallible, so malformed row certificates
        // refuse atomically: do not discharge either side unless both bias
        // carriers are writable and every present certificate has exactly one
        // entry per logical row.  In particular, never fold a valid prefix of
        // a short `coeff_err` and leave the relation half-mutated.
        for (data, bias) in [
            (&self.lower_a, &self.lower_b),
            (&self.upper_a, &self.upper_b),
        ] {
            if bias.len() != row_count
                || bias.as_slice().is_none()
                || data
                    .coeff_err
                    .as_ref()
                    .is_some_and(|err| err.len() != row_count)
            {
                return;
            }
        }
        let flat = input_box.flatten();
        let (Some(l), Some(u)) = (flat.lower().as_slice(), flat.upper().as_slice()) else {
            return; // non-contiguous box: keep carrying (sound, prior behavior)
        };
        if l.len() != u.len() {
            return;
        }
        let mag: Vec<f32> = l
            .iter()
            .zip(u.iter())
            .map(|(&lo, &hi)| {
                let m = lo.abs().max(hi.abs());
                // NaN/inf in the box ⇒ non-finite penalty ⇒ the row keeps carrying.
                if m.is_nan() {
                    f32::INFINITY
                } else {
                    m
                }
            })
            .collect();

        // Preflight BOTH explicit-row sides before changing either one. This is
        // stronger than merely guarding each call to `fold_side`: a malformed
        // upper carrier must not leave a valid lower carrier half-discharged
        // (or vice versa). Six-dimensional and unsupported layouts retain
        // their historical independent-side behavior.
        let lower_is_7d = self
            .lower_a
            .patches
            .as_ref()
            .is_some_and(|patches| patches.ndim() == 7);
        let upper_is_7d = self
            .upper_a
            .patches
            .as_ref()
            .is_some_and(|patches| patches.ndim() == 7);
        if allow_7d && lower_is_7d {
            if self.lower_b.len() != row_count
                || self.lower_b.as_slice().is_none()
                || self
                    .lower_a
                    .coeff_err
                    .as_ref()
                    .is_some_and(|err| err.len() != row_count)
            {
                return;
            }
        }
        if allow_7d && upper_is_7d {
            if self.upper_b.len() != row_count
                || self.upper_b.as_slice().is_none()
                || self
                    .upper_a
                    .coeff_err
                    .as_ref()
                    .is_some_and(|err| err.len() != row_count)
            {
                return;
            }
        }

        let lower_7d_sums = if allow_7d && lower_is_7d && self.lower_a.coeff_err.is_some() {
            let Some(shape) = self.lower_a.patches.as_ref().map(|patches| patches.shape()) else {
                return;
            };
            let Some(sums) = window_mag_sums_7d(&self.lower_a, &mag, row_count, shape) else {
                return;
            };
            Some(sums)
        } else {
            None
        };
        let upper_7d_sums = if allow_7d && upper_is_7d && self.upper_a.coeff_err.is_some() {
            let Some(shape) = self.upper_a.patches.as_ref().map(|patches| patches.shape()) else {
                return;
            };
            let Some(sums) = window_mag_sums_7d(&self.upper_a, &mag, row_count, shape) else {
                return;
            };
            Some(sums)
        } else {
            None
        };

        if self.lower_a.coeff_err.is_some() {
            let sums =
                lower_7d_sums.or_else(|| window_mag_sums(&self.lower_a, &mag, row_count, allow_7d));
            if let Some(sums) = sums {
                if let Some(bias) = self.lower_b.as_slice_mut() {
                    if fold_side(&mut self.lower_a, bias, &sums, true) {
                        self.lower_a.coeff_err = None;
                    }
                }
            }
        }
        if self.upper_a.coeff_err.is_some() {
            let sums =
                upper_7d_sums.or_else(|| window_mag_sums(&self.upper_a, &mag, row_count, allow_7d));
            if let Some(sums) = sums {
                if let Some(bias) = self.upper_b.as_slice_mut() {
                    if fold_side(&mut self.upper_a, bias, &sums, false) {
                        self.upper_a.coeff_err = None;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bounds::patches::{PatchGeometry, PatchesLinearBounds, UnstableIdx};
    use ndarray::{Array1, ArrayD, IxDyn};
    use proptest::prelude::*;

    /// Deterministic LCG so the enclosure sweep is reproducible.
    fn lcg(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        ((*state >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    }

    /// in_c=1, 4x4 input, 3x3 kernel, stride 1, no padding => 2x2 spatial,
    /// out_c=2 => 8 logical rows, 9 taps each (all interior).
    fn build(err: f32, seed: u64) -> (PatchesLinearBounds, BoundedTensor) {
        let (out_c, out_h, out_w) = (2usize, 2usize, 2usize);
        let (in_c, kh, kw) = (1usize, 3usize, 3usize);
        let rows = out_c * out_h * out_w;
        let mut s = seed;
        let patches =
            ArrayD::from_shape_fn(IxDyn(&[out_c, out_h, out_w, in_c, kh, kw]), |_| lcg(&mut s));
        let data = PatchesData {
            coeff_err: Some(Array1::from_elem(rows, err)),
            patches: Some(patches),
            geometry: PatchGeometry::affine((1, 1), (0, 0, 0, 0)),
            identity: false,
            output_shape: (out_c, out_h, out_w),
            input_shape: (in_c, 4, 4),
            unstable_idx: None,
        };
        let plb = PatchesLinearBounds {
            row_count: rows,
            lower_a: data.clone(),
            lower_b: Array1::zeros(rows),
            upper_a: data,
            upper_b: Array1::zeros(rows),
        };
        let lo = ArrayD::from_elem(IxDyn(&[in_c, 4, 4]), -1.0f32);
        let hi = ArrayD::from_elem(IxDyn(&[in_c, 4, 4]), 1.0f32);
        (plb, BoundedTensor::new(lo, hi).unwrap())
    }

    #[derive(Clone, Copy, Debug)]
    struct ExplicitGeometry {
        rows: usize,
        out_c: usize,
        out_h: usize,
        out_w: usize,
        in_c: usize,
        in_h: usize,
        in_w: usize,
        kh: usize,
        kw: usize,
        stride: (usize, usize),
        padding: (usize, usize, usize, usize),
    }

    fn explicit_out_dims(
        in_h: usize,
        in_w: usize,
        kh: usize,
        kw: usize,
        stride: (usize, usize),
        padding: (usize, usize, usize, usize),
    ) -> Option<(usize, usize)> {
        let padded_h = in_h.checked_add(padding.2)?.checked_add(padding.3)?;
        let padded_w = in_w.checked_add(padding.0)?.checked_add(padding.1)?;
        if kh == 0 || kw == 0 || stride.0 == 0 || stride.1 == 0 || padded_h < kh || padded_w < kw {
            return None;
        }
        Some((
            (padded_h - kh) / stride.0 + 1,
            (padded_w - kw) / stride.1 + 1,
        ))
    }

    fn build_explicit(
        geometry: ExplicitGeometry,
        errors: &[f32],
        seed: u64,
    ) -> (PatchesLinearBounds, BoundedTensor, Array1<f32>, Array1<f32>) {
        assert_eq!(errors.len(), geometry.rows);
        let mut state = seed;
        let shape = [
            geometry.rows,
            geometry.out_c,
            geometry.out_h,
            geometry.out_w,
            geometry.in_c,
            geometry.kh,
            geometry.kw,
        ];
        let patches = ArrayD::from_shape_fn(IxDyn(&shape), |_| lcg(&mut state));
        let data = PatchesData {
            coeff_err: Some(Array1::from_vec(errors.to_vec())),
            patches: Some(patches),
            geometry: PatchGeometry::affine(geometry.stride, geometry.padding),
            identity: false,
            output_shape: (geometry.out_c, geometry.out_h, geometry.out_w),
            input_shape: (geometry.in_c, geometry.in_h, geometry.in_w),
            unstable_idx: None,
        };
        let lower_b = Array1::from_shape_fn(geometry.rows, |r| -0.125 * ((r + 1) as f32));
        let upper_b = Array1::from_shape_fn(geometry.rows, |r| 0.2 * ((r + 1) as f32));
        let bounds = PatchesLinearBounds {
            row_count: geometry.rows,
            lower_a: data.clone(),
            lower_b: lower_b.clone(),
            upper_a: data,
            upper_b: upper_b.clone(),
        };

        let box_shape = IxDyn(&[geometry.in_c, geometry.in_h, geometry.in_w]);
        let lower = ArrayD::from_shape_fn(box_shape.clone(), |idx| {
            let flat = (idx[0] * geometry.in_h + idx[1]) * geometry.in_w + idx[2];
            -0.2 - 0.017 * ((flat % 11) as f32)
        });
        let upper = ArrayD::from_shape_fn(box_shape, |idx| {
            let flat = (idx[0] * geometry.in_h + idx[1]) * geometry.in_w + idx[2];
            0.3 + 0.013 * ((flat % 13) as f32)
        });
        (
            bounds,
            BoundedTensor::new(lower, upper).unwrap(),
            lower_b,
            upper_b,
        )
    }

    /// Independent coordinate-arithmetic oracle for the 7D unfold support.
    /// Deliberately does not call the production unfold helpers.
    fn reference_occurrence_mass(geometry: ExplicitGeometry, mag: &[f32]) -> (f64, usize) {
        let mut sum = 0.0f64;
        let mut count = 0usize;
        for _oc in 0..geometry.out_c {
            for oh in 0..geometry.out_h {
                for ow in 0..geometry.out_w {
                    for ic in 0..geometry.in_c {
                        for ki in 0..geometry.kh {
                            for kj in 0..geometry.kw {
                                let ih_padded = oh * geometry.stride.0 + ki;
                                let iw_padded = ow * geometry.stride.1 + kj;
                                let Some(ih) = ih_padded.checked_sub(geometry.padding.2) else {
                                    continue;
                                };
                                let Some(iw) = iw_padded.checked_sub(geometry.padding.0) else {
                                    continue;
                                };
                                if ih >= geometry.in_h || iw >= geometry.in_w {
                                    continue;
                                }
                                let in_flat = (ic * geometry.in_h + ih) * geometry.in_w + iw;
                                sum += f64::from(mag[in_flat]);
                                count += 1;
                            }
                        }
                    }
                }
            }
        }
        (sum, count)
    }

    /// Evaluate one stored 7D row and an adversarial admissible true row.
    fn reference_explicit_eval(
        data: &PatchesData,
        geometry: ExplicitGeometry,
        row: usize,
        y: &[f32],
        err: f32,
        direction: f64,
    ) -> (f64, f64) {
        let patches = data.patches.as_ref().unwrap();
        let mut stored = 0.0f64;
        let mut perturbed = 0.0f64;
        for oc in 0..geometry.out_c {
            for oh in 0..geometry.out_h {
                for ow in 0..geometry.out_w {
                    for ic in 0..geometry.in_c {
                        for ki in 0..geometry.kh {
                            for kj in 0..geometry.kw {
                                let ih_padded = oh * geometry.stride.0 + ki;
                                let iw_padded = ow * geometry.stride.1 + kj;
                                let Some(ih) = ih_padded.checked_sub(geometry.padding.2) else {
                                    continue;
                                };
                                let Some(iw) = iw_padded.checked_sub(geometry.padding.0) else {
                                    continue;
                                };
                                if ih >= geometry.in_h || iw >= geometry.in_w {
                                    continue;
                                }
                                let in_flat = (ic * geometry.in_h + ih) * geometry.in_w + iw;
                                let yv = f64::from(y[in_flat]);
                                let stored_coeff =
                                    f64::from(patches[[row, oc, oh, ow, ic, ki, kj]]);
                                let perturbation = direction * f64::from(err) * yv.signum();
                                stored += stored_coeff * yv;
                                perturbed += (stored_coeff + perturbation) * yv;
                            }
                        }
                    }
                }
            }
        }
        (stored, perturbed)
    }

    /// The discharged penalty is exactly `err * Σ_taps mag` (= err*9 on a
    /// unit box with 9 interior taps), folded OUTWARD, and the error is cleared.
    #[test]
    fn eager_fold_discharges_err_times_window_mass() {
        let (mut plb, boxb) = build(1e-3, 12345);
        plb.fold_coeff_err_over_box_eager(&boxb);
        assert!(plb.lower_a.coeff_err.is_none(), "lower err not cleared");
        assert!(plb.upper_a.coeff_err.is_none(), "upper err not cleared");
        for i in 0..plb.row_count {
            let expect = 1e-3f32 * 9.0;
            assert!(
                (plb.lower_b[i] + expect).abs() <= 1e-6,
                "row {i}: lower_b {} != -{expect}",
                plb.lower_b[i]
            );
            assert!(
                (plb.upper_b[i] - expect).abs() <= 1e-6,
                "row {i}: upper_b {} != {expect}",
                plb.upper_b[i]
            );
        }
    }

    /// ENCLOSURE: for every admissible true coefficient matrix `Ã` (within
    /// `err` of the stored coefficients on the patch support) and every `y` in
    /// the box, the folded bias must dominate what the carried error covered:
    ///   `A·y + lower_b_folded ≤ Ã·y`  and  `A·y + upper_b_folded ≥ Ã·y`
    /// (original biases are zero). This is the fold identity under test.
    #[test]
    fn eager_fold_preserves_enclosure_against_perturbed_coefficients() {
        let err = 2e-3f32;
        let (plb0, boxb) = build(err, 999);
        let mut plb = plb0.clone();
        plb.fold_coeff_err_over_box_eager(&boxb);

        let a = plb0.lower_a.patches.as_ref().unwrap().clone();
        let shape = a.shape().to_vec();
        let (out_c, out_h, out_w) = (shape[0], shape[1], shape[2]);
        let (in_c, kh, kw) = (shape[3], shape[4], shape[5]);
        let mut s = 4242u64;
        for _trial in 0..200 {
            // y in the box, and a perturbation of every coefficient within err.
            let y = ArrayD::from_shape_fn(IxDyn(&[in_c, 4, 4]), |_| lcg(&mut s));
            for oc in 0..out_c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let row = (oc * out_h + oh) * out_w + ow;
                        let (mut exact, mut perturbed) = (0.0f64, 0.0f64);
                        for ic in 0..in_c {
                            for ki in 0..kh {
                                for kj in 0..kw {
                                    let c = a[[oc, oh, ow, ic, ki, kj]];
                                    let yv = y[[ic, oh + ki, ow + kj]];
                                    exact += f64::from(c) * f64::from(yv);
                                    perturbed += f64::from(c + err * lcg(&mut s)) * f64::from(yv);
                                }
                            }
                        }
                        let lo = exact + f64::from(plb.lower_b[row]);
                        let hi = exact + f64::from(plb.upper_b[row]);
                        assert!(
                            lo <= perturbed + 1e-9 && perturbed <= hi + 1e-9,
                            "row {row}: perturbed {perturbed} escaped [{lo}, {hi}]"
                        );
                    }
                }
            }
        }
    }

    /// A row with a non-finite or illegal negative error keeps carrying — never
    /// a new degrade, and never a silently-dropped error.
    #[test]
    fn eager_fold_keeps_invalid_rows_carrying() {
        let (mut plb, boxb) = build(1e-3, 77);
        plb.lower_a.coeff_err.as_mut().unwrap()[3] = f32::INFINITY;
        plb.lower_a.coeff_err.as_mut().unwrap()[4] = f32::NAN;
        plb.lower_a.coeff_err.as_mut().unwrap()[5] = -1.0;
        plb.fold_coeff_err_over_box_eager(&boxb);
        let err = plb
            .lower_a
            .coeff_err
            .as_ref()
            .expect("lower err must still be carried");
        assert!(err[3].is_infinite(), "poisoned row must keep its INF");
        assert!(err[4].is_nan(), "poisoned row must keep its NaN");
        assert_eq!(err[5], -1.0, "illegal negative error must not be cleared");
        assert_eq!(err[0], 0.0, "discharged row must be cleared to 0");
        assert!(plb.upper_a.coeff_err.is_none(), "upper side fully cleared");
    }

    /// The infallible eager API cannot surface `ShapeMismatch`, so a malformed
    /// row certificate refuses atomically without discharging the valid side
    /// or a valid prefix of the malformed side.
    #[test]
    fn eager_fold_6d_malformed_coeff_err_length_is_atomic_noop() {
        let (mut plb, boxb) = build(1e-3, 0x6dba_d1e0);
        plb.upper_a.coeff_err = Some(Array1::from_elem(plb.row_count - 1, 2e-3));
        let before = plb.clone();
        plb.fold_coeff_err_over_box_eager(&boxb);
        assert_explicit_fold_noop(&before, &plb, "malformed 6D coeff_err length");
    }

    #[test]
    fn eager_fold_6d_supports_anchored_windows_and_mismatch_is_atomic() {
        let patches = ArrayD::zeros(IxDyn(&[1, 1, 2, 1, 1, 1]));
        let data = PatchesData {
            coeff_err: Some(Array1::from_elem(2, 0.5)),
            patches: Some(patches),
            geometry: PatchGeometry::anchored(vec![0], vec![0, 3]).unwrap(),
            identity: false,
            output_shape: (1, 1, 2),
            input_shape: (1, 1, 4),
            unstable_idx: None,
        };
        let bounds = PatchesLinearBounds {
            row_count: 2,
            lower_a: data.clone(),
            lower_b: Array1::zeros(2),
            upper_a: data,
            upper_b: Array1::zeros(2),
        };
        let magnitudes =
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let boxb = BoundedTensor::new(-&magnitudes, magnitudes).unwrap();

        let mut folded = bounds.clone();
        folded.fold_coeff_err_over_box_eager(&boxb);
        assert!(folded.lower_a.coeff_err.is_none());
        assert!(folded.upper_a.coeff_err.is_none());
        assert!(folded.lower_b[0] <= -0.5 && folded.upper_b[0] >= 0.5);
        assert!(folded.lower_b[1] <= -2.0 && folded.upper_b[1] >= 2.0);

        let mut mismatched = bounds;
        mismatched.upper_a.geometry = PatchGeometry::anchored(vec![0], vec![0, 2]).unwrap();
        let before = mismatched.clone();
        mismatched.fold_coeff_err_over_box_eager(&boxb);
        assert_explicit_fold_noop(&before, &mismatched, "anchored lower/upper mismatch");
    }

    // ---- 7D explicit-rows extension (#patches-eager-err-7d) ----

    /// 7D explicit-rows fixture: `rows` spec rows over grid out_c=2, 2x2
    /// spatial; in_c=1, 4x4 input, 3x3 kernel, stride 1, no padding. Every
    /// tap is in range, so each spec row owns 2*(2*2)*(1*3*3) = 72 occurrences.
    fn build_7d(
        err: f32,
        seed: u64,
        lo: &dyn Fn(usize) -> f32,
        hi: &dyn Fn(usize) -> f32,
    ) -> (PatchesLinearBounds, BoundedTensor) {
        let rows = 3usize;
        let (out_c, out_h, out_w) = (2usize, 2usize, 2usize);
        let (in_c, kh, kw) = (1usize, 3usize, 3usize);
        let mut s = seed;
        let patches =
            ArrayD::from_shape_fn(IxDyn(&[rows, out_c, out_h, out_w, in_c, kh, kw]), |_| {
                lcg(&mut s)
            });
        let data = PatchesData {
            coeff_err: Some(Array1::from_elem(rows, err)),
            patches: Some(patches),
            geometry: PatchGeometry::affine((1, 1), (0, 0, 0, 0)),
            identity: false,
            output_shape: (out_c, out_h, out_w),
            input_shape: (in_c, 4, 4),
            unstable_idx: None,
        };
        let plb = PatchesLinearBounds {
            row_count: rows,
            lower_a: data.clone(),
            lower_b: Array1::zeros(rows),
            upper_a: data,
            upper_b: Array1::zeros(rows),
        };
        let lo_t = ArrayD::from_shape_fn(IxDyn(&[in_c, 4, 4]), |ix| {
            lo(ix[0] * 16 + ix[1] * 4 + ix[2])
        });
        let hi_t = ArrayD::from_shape_fn(IxDyn(&[in_c, 4, 4]), |ix| {
            hi(ix[0] * 16 + ix[1] * 4 + ix[2])
        });
        (plb, BoundedTensor::new(lo_t, hi_t).unwrap())
    }

    /// 7D + policy ON: the discharged penalty is `err * (occurrence sum of box
    /// magnitudes over the WHOLE spec-row slab)` (= err*72 on a unit box: 2
    /// output channels x 4 positions x 9 taps), folded OUTWARD, error cleared.
    #[test]
    fn eager_fold_7d_discharges_err_times_row_slab_mass() {
        let (mut plb, boxb) = build_7d(1e-3, 2024, &|_| -1.0, &|_| 1.0);
        plb.fold_coeff_err_over_box_eager_with_policy(&boxb, true);
        assert!(plb.lower_a.coeff_err.is_none(), "lower err not cleared");
        assert!(plb.upper_a.coeff_err.is_none(), "upper err not cleared");
        for i in 0..plb.row_count {
            let expect = 1e-3f32 * 72.0;
            assert!(
                (plb.lower_b[i] + expect).abs() <= 1e-5,
                "row {i}: lower_b {} != -{expect}",
                plb.lower_b[i]
            );
            assert!(
                (plb.upper_b[i] - expect).abs() <= 1e-5,
                "row {i}: upper_b {} != {expect}",
                plb.upper_b[i]
            );
        }
    }

    #[test]
    fn eager_fold_7d_supports_anchored_occurrence_geometry() {
        let patches = ArrayD::zeros(IxDyn(&[1, 1, 1, 2, 1, 1, 1]));
        let data = PatchesData {
            coeff_err: Some(Array1::from_elem(1, 0.25)),
            patches: Some(patches),
            geometry: PatchGeometry::anchored(vec![0], vec![0, 3]).unwrap(),
            identity: false,
            output_shape: (1, 1, 2),
            input_shape: (1, 1, 4),
            unstable_idx: None,
        };
        let mut bounds = PatchesLinearBounds {
            row_count: 1,
            lower_a: data.clone(),
            lower_b: Array1::zeros(1),
            upper_a: data,
            upper_b: Array1::zeros(1),
        };
        let magnitudes =
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let boxb = BoundedTensor::new(-&magnitudes, magnitudes).unwrap();

        bounds.fold_coeff_err_over_box_eager_with_policy(&boxb, true);
        assert!(bounds.lower_a.coeff_err.is_none());
        assert!(bounds.upper_a.coeff_err.is_none());
        assert!(bounds.lower_b[0] <= -1.25);
        assert!(bounds.upper_b[0] >= 1.25);
    }

    /// 7D + policy OFF: bit-exact no-op — the biases and the carried error are
    /// byte-identical to the input (the pre-extension behavior, and the pin
    /// that `NY_PATCHES_EAGER_ERR=1` alone does not change 7D carriers).
    #[test]
    fn eager_fold_7d_policy_off_is_bitwise_noop() {
        let (mut plb, boxb) = build_7d(1e-3, 5150, &|_| -1.0, &|_| 1.0);
        let lower_b0: Vec<u32> = plb.lower_b.iter().map(|v| v.to_bits()).collect();
        let upper_b0: Vec<u32> = plb.upper_b.iter().map(|v| v.to_bits()).collect();
        let err0: Vec<u32> = plb
            .lower_a
            .coeff_err
            .as_ref()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        plb.fold_coeff_err_over_box_eager_with_policy(&boxb, false);
        let lower_b1: Vec<u32> = plb.lower_b.iter().map(|v| v.to_bits()).collect();
        let upper_b1: Vec<u32> = plb.upper_b.iter().map(|v| v.to_bits()).collect();
        assert_eq!(lower_b0, lower_b1, "lower bias must be untouched");
        assert_eq!(upper_b0, upper_b1, "upper bias must be untouched");
        let err_l: Vec<u32> = plb
            .lower_a
            .coeff_err
            .as_ref()
            .expect("lower err must still be carried")
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let err_u: Vec<u32> = plb
            .upper_a
            .coeff_err
            .as_ref()
            .expect("upper err must still be carried")
            .iter()
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(err0, err_l, "lower err must be untouched");
        assert_eq!(err0, err_u, "upper err must be untouched");
    }

    /// Overlapping taps must be counted WITH MULTIPLICITY (the trap named in
    /// the module docs): on a non-uniform box the penalty must equal the brute
    /// force occurrence sum over the full `oc, oh, ow, taps` slab — the same
    /// geometry `scatter_rows_err_accumulators` counts — not a unique-column
    /// or single-window sum.
    #[test]
    fn eager_fold_7d_counts_overlapping_taps_with_multiplicity() {
        let err = 2e-3f32;
        let lo = |j: usize| -(0.2 + 0.01 * j as f32);
        let hi = |j: usize| 0.15 + 0.012 * j as f32;
        let (mut plb, boxb) = build_7d(err, 31337, &lo, &hi);
        plb.fold_coeff_err_over_box_eager_with_policy(&boxb, true);
        assert!(plb.lower_a.coeff_err.is_none(), "lower err not cleared");

        // Brute-force occurrence sum: 2 output channels x every 2x2 window
        // position x 3x3 taps into the 4x4 input (stride 1, pad 0). Interior
        // columns are visited by up to 4 windows x 2 channels = multiplicity 8.
        let mut s_true = 0.0f64;
        let mut max_col_multiplicity = [0u32; 16];
        for _oc in 0..2 {
            for oh in 0..2 {
                for ow in 0..2 {
                    for ki in 0..3 {
                        for kj in 0..3 {
                            let j = (oh + ki) * 4 + (ow + kj);
                            let m = f64::from(lo(j).abs().max(hi(j).abs()));
                            s_true += m;
                            max_col_multiplicity[j] += 1;
                        }
                    }
                }
            }
        }
        assert!(
            max_col_multiplicity.iter().any(|&c| c >= 8),
            "fixture must actually overlap (interior column multiplicity 8)"
        );
        let expect = f64::from(err) * s_true;
        for i in 0..plb.row_count {
            let got = -f64::from(plb.lower_b[i]);
            assert!(
                got >= expect * (1.0 - 1e-6) && got <= expect * (1.0 + 1e-6),
                "row {i}: penalty {got} != occurrence-sum expectation {expect}"
            );
            assert!(
                f64::from(plb.upper_b[i]) >= expect * (1.0 - 1e-6),
                "row {i}: upper penalty under the occurrence sum"
            );
        }
    }

    /// ENCLOSURE (7D): for every admissible true coefficient slab (within `err`
    /// of the stored occurrences) and every `y` in the box, the folded bias
    /// dominates what the carried error covered — the fold identity of
    /// [`window_mag_sums_7d`]'s header comment under test.
    #[test]
    fn eager_fold_7d_preserves_enclosure_against_perturbed_coefficients() {
        let err = 2e-3f32;
        let (plb0, boxb) = build_7d(err, 9_997, &|_| -1.0, &|_| 1.0);
        let mut plb = plb0.clone();
        plb.fold_coeff_err_over_box_eager_with_policy(&boxb, true);

        let a = plb0.lower_a.patches.as_ref().unwrap().clone();
        let shape = a.shape().to_vec();
        let (rows, out_c, out_h, out_w) = (shape[0], shape[1], shape[2], shape[3]);
        let (in_c, kh, kw) = (shape[4], shape[5], shape[6]);
        let mut s = 424_242u64;
        for _trial in 0..100 {
            let y = ArrayD::from_shape_fn(IxDyn(&[in_c, 4, 4]), |_| lcg(&mut s));
            for row in 0..rows {
                let (mut exact, mut perturbed) = (0.0f64, 0.0f64);
                for oc in 0..out_c {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            for ic in 0..in_c {
                                for ki in 0..kh {
                                    for kj in 0..kw {
                                        let c = a[[row, oc, oh, ow, ic, ki, kj]];
                                        let yv = y[[ic, oh + ki, ow + kj]];
                                        exact += f64::from(c) * f64::from(yv);
                                        perturbed +=
                                            f64::from(c + err * lcg(&mut s)) * f64::from(yv);
                                    }
                                }
                            }
                        }
                    }
                }
                let lo = exact + f64::from(plb.lower_b[row]);
                let hi = exact + f64::from(plb.upper_b[row]);
                assert!(
                    lo <= perturbed + 1e-9 && perturbed <= hi + 1e-9,
                    "row {row}: perturbed {perturbed} escaped [{lo}, {hi}]"
                );
            }
        }
    }

    /// 7D rows with non-finite or illegal negative error keep carrying, valid
    /// rows in the same tensor are still discharged — same partial-clear
    /// semantics as the 6D fold.
    #[test]
    fn eager_fold_7d_keeps_invalid_rows_carrying() {
        let (mut plb, boxb) = build_7d(1e-3, 88, &|_| -1.0, &|_| 1.0);
        plb.lower_a.coeff_err.as_mut().unwrap()[0] = f32::INFINITY;
        plb.lower_a.coeff_err.as_mut().unwrap()[1] = f32::NAN;
        plb.fold_coeff_err_over_box_eager_with_policy(&boxb, true);
        let err = plb
            .lower_a
            .coeff_err
            .as_ref()
            .expect("lower err must still be carried");
        assert!(err[0].is_infinite(), "poisoned row must keep its INF");
        assert!(err[1].is_nan(), "poisoned row must keep its NaN");
        assert_eq!(err[2], 0.0, "valid row must be discharged to 0");
        assert!(plb.lower_b[2] < 0.0, "discharged row must widen its bias");
        assert!(plb.upper_a.coeff_err.is_none(), "upper side fully cleared");
    }

    #[test]
    fn eager_fold_7d_asymmetric_geometry_matches_strict_occurrence_oracle() {
        let (out_h, out_w) = explicit_out_dims(5, 6, 5, 3, (2, 2), (1, 0, 2, 0)).unwrap();
        let geometry = ExplicitGeometry {
            rows: 3,
            out_c: 2,
            out_h,
            out_w,
            in_c: 2,
            in_h: 5,
            in_w: 6,
            kh: 5,
            kw: 3,
            stride: (2, 2),
            padding: (1, 0, 2, 0),
        };
        let errors = [1.0e-4f32, 3.0e-4, 7.0e-4];
        let (mut bounds, input_box, initial_lower, initial_upper) =
            build_explicit(geometry, &errors, 0x71d_5eed);

        // Dilation-expanded storage contains exact zeros, but the row-wide
        // certificate still charges those stored occurrences.
        for side in [&mut bounds.lower_a, &mut bounds.upper_a] {
            let patches = side.patches.as_mut().unwrap();
            for r in 0..geometry.rows {
                for oc in 0..geometry.out_c {
                    for oh in 0..geometry.out_h {
                        for ow in 0..geometry.out_w {
                            for ic in 0..geometry.in_c {
                                for ki in 0..geometry.kh {
                                    for kj in 0..geometry.kw {
                                        if ki % 2 == 1 || kj % 2 == 1 {
                                            patches[[r, oc, oh, ow, ic, ki, kj]] = 0.0;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let flat = input_box.flatten();
        let mag: Vec<f32> = flat
            .lower()
            .as_slice()
            .unwrap()
            .iter()
            .zip(flat.upper().as_slice().unwrap())
            .map(|(&lo, &hi)| lo.abs().max(hi.abs()))
            .collect();
        let (raw_mass, occurrences) = reference_occurrence_mass(geometry, &mag);
        let inflated_mass = raw_mass * (1.0 + gamma_n_f64(occurrences.max(1)));
        assert!(occurrences > geometry.in_c * geometry.in_h * geometry.in_w);

        bounds.fold_coeff_err_over_box_eager_with_policy(&input_box, true);
        assert!(bounds.lower_a.coeff_err.is_none());
        assert!(bounds.upper_a.coeff_err.is_none());
        for r in 0..geometry.rows {
            let penalty = f64::from(errors[r]) * inflated_mass;
            let expected_lower = next_down_f32((f64::from(initial_lower[r]) - penalty) as f32);
            let expected_upper = next_up_f32((f64::from(initial_upper[r]) + penalty) as f32);
            assert_eq!(bounds.lower_b[r].to_bits(), expected_lower.to_bits());
            assert_eq!(bounds.upper_b[r].to_bits(), expected_upper.to_bits());
        }
    }

    #[test]
    fn eager_fold_7d_keeps_later_scatter_rounding_certified() {
        let geometry = ExplicitGeometry {
            rows: 1,
            out_c: 1,
            out_h: 2,
            out_w: 2,
            in_c: 1,
            in_h: 3,
            in_w: 3,
            kh: 2,
            kw: 2,
            stride: (1, 1),
            padding: (0, 0, 0, 0),
        };
        let (mut bounds, input_box, _, _) = build_explicit(geometry, &[2.0e-3], 99);
        bounds.fold_coeff_err_over_box_eager_with_policy(&input_box, true);
        assert!(bounds.lower_a.coeff_err.is_none());
        assert!(bounds.upper_a.coeff_err.is_none());

        let dense = bounds.to_dense().unwrap();
        let lower_err = dense
            .lower_a_err()
            .expect("explicit-row overlap scatter must publish intrinsic error");
        let upper_err = dense
            .upper_a_err()
            .expect("explicit-row overlap scatter must publish intrinsic error");
        assert!(
            lower_err.iter().any(|&v| v > 0.0) && upper_err.iter().any(|&v| v > 0.0),
            "eager clearing must not clear the later gamma*absacc scatter term"
        );
    }

    #[test]
    fn eager_fold_7d_rounds_bias_outward_at_f32_boundary() {
        let geometry = ExplicitGeometry {
            rows: 1,
            out_c: 1,
            out_h: 1,
            out_w: 1,
            in_c: 1,
            in_h: 1,
            in_w: 1,
            kh: 1,
            kw: 1,
            stride: (1, 1),
            padding: (0, 0, 0, 0),
        };
        let err = f32::from_bits(0x3300_0000); // 2^-25: midpoint at bias 1
        let (mut bounds, _, _, _) = build_explicit(geometry, &[err], 7);
        bounds.lower_b[0] = 1.0;
        bounds.upper_b[0] = 1.0;
        let unit = ArrayD::from_elem(IxDyn(&[1, 1, 1]), 1.0f32);
        let unit_box = BoundedTensor::new(unit.clone(), unit).unwrap();
        let exact_penalty = f64::from(err) * (1.0 + gamma_n_f64(1));

        bounds.fold_coeff_err_over_box_eager_with_policy(&unit_box, true);
        assert!(f64::from(bounds.lower_b[0]) <= 1.0f64 - exact_penalty);
        assert!(f64::from(bounds.upper_b[0]) >= 1.0f64 + exact_penalty);
    }

    fn assert_explicit_fold_noop(
        before: &PatchesLinearBounds,
        after: &PatchesLinearBounds,
        context: &str,
    ) {
        assert_eq!(
            before
                .lower_b
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            after
                .lower_b
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            "{context}: lower bias changed"
        );
        assert_eq!(
            before
                .upper_b
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            after
                .upper_b
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            "{context}: upper bias changed"
        );
        for (name, before_err, after_err) in [
            (
                "lower",
                before.lower_a.coeff_err.as_ref(),
                after.lower_a.coeff_err.as_ref(),
            ),
            (
                "upper",
                before.upper_a.coeff_err.as_ref(),
                after.upper_a.coeff_err.as_ref(),
            ),
        ] {
            let before_bits = before_err.map(|e| e.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
            let after_bits = after_err.map(|e| e.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
            assert_eq!(
                before_bits, after_bits,
                "{context}: {name} coeff_err changed"
            );
        }
    }

    #[test]
    fn eager_fold_7d_malformed_carriers_fail_closed_before_either_side_mutates() {
        let geometry = ExplicitGeometry {
            rows: 2,
            out_c: 1,
            out_h: 2,
            out_w: 2,
            in_c: 1,
            in_h: 3,
            in_w: 3,
            kh: 2,
            kw: 2,
            stride: (1, 1),
            padding: (0, 0, 0, 0),
        };
        let (base, input_box, _, _) = build_explicit(geometry, &[1e-3, 2e-3], 13);
        let mut cases: Vec<(&str, PatchesLinearBounds)> = Vec::new();

        let mut malformed_shape = base.clone();
        malformed_shape.lower_a.patches = Some(ArrayD::zeros(IxDyn(&[
            geometry.rows + 1,
            geometry.out_c,
            geometry.out_h,
            geometry.out_w,
            geometry.in_c,
            geometry.kh,
            geometry.kw,
        ])));
        cases.push(("lower shape[0] != row_count", malformed_shape));

        let mut bad_err_len = base.clone();
        bad_err_len.upper_a.coeff_err = Some(Array1::from_vec(vec![1e-3]));
        cases.push(("upper coeff_err length mismatch", bad_err_len));

        let mut bad_lower_bias_len = base.clone();
        bad_lower_bias_len.lower_b = Array1::from_vec(vec![0.0]);
        cases.push(("lower bias length mismatch", bad_lower_bias_len));

        let mut bad_upper_bias_len = base.clone();
        bad_upper_bias_len.upper_b = Array1::from_vec(vec![0.0]);
        cases.push(("upper bias length mismatch", bad_upper_bias_len));

        let mut bad_metadata = base.clone();
        bad_metadata.upper_a.output_shape.2 += 1;
        cases.push(("upper unfold/output shape mismatch", bad_metadata));

        let mut sparse = base.clone();
        sparse.upper_a.unstable_idx = Some(UnstableIdx {
            channels: vec![0],
            heights: vec![0],
            widths: vec![0],
        });
        cases.push(("upper sparse marker", sparse));

        let mut identity = base.clone();
        identity.lower_a.identity = true;
        cases.push(("lower identity marker", identity));

        let mut overflow = base;
        overflow.upper_a.geometry = PatchGeometry::affine((1, 1), (usize::MAX, 0, 0, 0));
        cases.push(("upper padding overflow", overflow));

        for (context, mut candidate) in cases {
            let before = candidate.clone();
            candidate.fold_coeff_err_over_box_eager_with_policy(&input_box, true);
            assert_explicit_fold_noop(&before, &candidate, context);
        }
    }

    #[test]
    fn eager_fold_7d_preflights_opposite_none_side_bias_atomically() {
        let geometry = ExplicitGeometry {
            rows: 2,
            out_c: 1,
            out_h: 2,
            out_w: 2,
            in_c: 1,
            in_h: 3,
            in_w: 3,
            kh: 2,
            kw: 2,
            stride: (1, 1),
            padding: (0, 0, 0, 0),
        };
        let (base, input_box, _, _) = build_explicit(geometry, &[1e-3, 2e-3], 0xA70C);

        let mut live_control = base.clone();
        live_control.fold_coeff_err_over_box_eager_with_policy(&input_box, true);
        assert!(
            live_control.lower_a.coeff_err.is_none() && live_control.upper_a.coeff_err.is_none(),
            "the valid control must actually discharge both sides"
        );

        let mut lower_active = base.clone();
        lower_active.upper_a.coeff_err = None;
        lower_active.upper_b = Array1::from_vec(vec![0.0]);
        let before = lower_active.clone();
        lower_active.fold_coeff_err_over_box_eager_with_policy(&input_box, true);
        assert_explicit_fold_noop(
            &before,
            &lower_active,
            "upper err None plus malformed upper bias must protect active lower",
        );

        let mut upper_active = base;
        upper_active.lower_a.coeff_err = None;
        upper_active.lower_b = Array1::from_vec(vec![0.0]);
        let before = upper_active.clone();
        upper_active.fold_coeff_err_over_box_eager_with_policy(&input_box, true);
        assert_explicit_fold_noop(
            &before,
            &upper_active,
            "lower err None plus malformed lower bias must protect active upper",
        );
    }

    #[test]
    fn eager_fold_7d_noncontiguous_bias_is_atomic_noop_in_both_directions() {
        let geometry = ExplicitGeometry {
            rows: 2,
            out_c: 1,
            out_h: 2,
            out_w: 2,
            in_c: 1,
            in_h: 3,
            in_w: 3,
            kh: 2,
            kw: 2,
            stride: (1, 1),
            padding: (0, 0, 0, 0),
        };
        let (base, input_box, _, _) = build_explicit(geometry, &[1e-3, 2e-3], 0xB1A5);
        let strided = |first: f32, second: f32| {
            Array1::from_vec(vec![first, 123.0, second]).slice_move(ndarray::s![..;2])
        };

        let mut bad_lower = base.clone();
        bad_lower.lower_b = strided(bad_lower.lower_b[0], bad_lower.lower_b[1]);
        assert_eq!(bad_lower.lower_b.len(), geometry.rows);
        assert!(bad_lower.lower_b.as_slice().is_none());
        let before = bad_lower.clone();
        bad_lower.fold_coeff_err_over_box_eager_with_policy(&input_box, true);
        assert_explicit_fold_noop(
            &before,
            &bad_lower,
            "non-contiguous lower bias must protect upper from partial mutation",
        );

        let mut bad_upper = base;
        bad_upper.upper_b = strided(bad_upper.upper_b[0], bad_upper.upper_b[1]);
        assert_eq!(bad_upper.upper_b.len(), geometry.rows);
        assert!(bad_upper.upper_b.as_slice().is_none());
        let before = bad_upper.clone();
        bad_upper.fold_coeff_err_over_box_eager_with_policy(&input_box, true);
        assert_explicit_fold_noop(
            &before,
            &bad_upper,
            "non-contiguous upper bias must protect lower from partial mutation",
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(96))]

        #[test]
        fn eager_fold_7d_preserves_strict_randomized_f64_oracle(
            rows in 1usize..4,
            out_c in 1usize..4,
            in_c in 1usize..3,
            in_h in 2usize..7,
            in_w in 2usize..7,
            kh in 1usize..5,
            kw in 1usize..5,
            sh in 1usize..3,
            sw in 1usize..3,
            pl in 0usize..3,
            pr in 0usize..3,
            pt in 0usize..3,
            pb in 0usize..3,
            seed in any::<u64>(),
        ) {
            let padding = (pl, pr, pt, pb);
            let Some((out_h, out_w)) =
                explicit_out_dims(in_h, in_w, kh, kw, (sh, sw), padding)
            else {
                prop_assume!(false);
                unreachable!();
            };
            prop_assume!(out_h <= 8 && out_w <= 8);
            let geometry = ExplicitGeometry {
                rows,
                out_c,
                out_h,
                out_w,
                in_c,
                in_h,
                in_w,
                kh,
                kw,
                stride: (sh, sw),
                padding,
            };
            let errors: Vec<f32> = (0..rows)
                .map(|r| 1.0e-5 * ((r + 1) as f32))
                .collect();
            let (mut bounds, input_box, initial_lower, initial_upper) =
                build_explicit(geometry, &errors, seed);
            let original = bounds.clone();

            let flat_box = input_box.flatten();
            let lower = flat_box.lower().as_slice().unwrap();
            let upper = flat_box.upper().as_slice().unwrap();
            let mut sample_state = seed ^ 0xa5a5_5a5a_1357_2468;
            let y: Vec<f32> = lower
                .iter()
                .zip(upper.iter())
                .map(|(&lo, &hi)| {
                    let t = (lcg(&mut sample_state) + 1.0) * 0.5;
                    lo + t * (hi - lo)
                })
                .collect();

            bounds.fold_coeff_err_over_box_eager_with_policy(&input_box, true);
            prop_assert!(bounds.lower_a.coeff_err.is_none());
            prop_assert!(bounds.upper_a.coeff_err.is_none());
            for r in 0..rows {
                let (stored_lower, true_lower) = reference_explicit_eval(
                    &original.lower_a,
                    geometry,
                    r,
                    &y,
                    errors[r],
                    -1.0,
                );
                let (stored_upper, true_upper) = reference_explicit_eval(
                    &original.upper_a,
                    geometry,
                    r,
                    &y,
                    errors[r],
                    1.0,
                );
                let published_lower = stored_lower + f64::from(bounds.lower_b[r]);
                let oracle_lower = true_lower + f64::from(initial_lower[r]);
                let published_upper = stored_upper + f64::from(bounds.upper_b[r]);
                let oracle_upper = true_upper + f64::from(initial_upper[r]);
                prop_assert!(
                    published_lower <= oracle_lower,
                    "row {r}: lower {published_lower} > oracle {oracle_lower}; geometry={geometry:?}"
                );
                prop_assert!(
                    published_upper >= oracle_upper,
                    "row {r}: upper {published_upper} < oracle {oracle_upper}; geometry={geometry:?}"
                );
            }
        }
    }

    /// Miniature Add_28 cascade
    /// (docs/ADD28_COEFF_ERR_AND_PATCHES_SENTINEL_DIAGNOSIS_2026-07-30.md §2),
    /// driven through the REAL pipeline: a 7D explicit-rows spec carrier with
    /// seeded certified error `e0` arrives at a ReLU, then walks backward
    /// through relu -> conv -> relu -> conv and densifies (the residual
    /// junction event). OFF is today's walk (row-lift x MSS per ReLU, x‖k‖₁
    /// per conv carry, x count at densification). ON additionally applies the
    /// eager 7D discharge after each ReLU backward step — exactly what the
    /// `NY_PATCHES_EAGER_ERR_7D=1` call-site gating does.
    ///
    /// Asserts, on f64-evaluated forward passes at sampled box points:
    ///  - BOTH paths' final concrete bounds enclose the true outputs, for the
    ///    stored coefficients AND for e0-perturbed admissible coefficients;
    ///  - ON's final interval is never wider than OFF's;
    ///  - ON's final max-row-L1 certified error is at least 100x smaller —
    ///    MEASURED 585.8x on this fixture (off 6.815e-2, on 1.163e-4), floor
    ///    set at ~1/5 of measured; the intrinsic conv contraction +
    ///    scatter-rounding certificates must remain nonzero.
    #[test]
    fn eager_fold_7d_add28_miniature_cascade() {
        use crate::bounds::patches::CrownBounds;
        use crate::layers::common::crown_elementwise_backward_patches;
        use crate::layers::Conv2dLayer;

        // Forward net: x(2,8,8) -conv1 3x3 s1 p0-> h(2,6,6) -relu1-> g
        //              -conv2 3x3 s1 p0-> y(2,4,4) -relu2-> z; C0 reads z.
        // Zero conv biases keep every box zero-centered: the discharge then
        // sees the CONTRACTED intermediate cut (the deep-stack physics the
        // fold exploits) instead of an activation offset.
        let rows = 3usize;
        let (xc, xh, xw) = (2usize, 8usize, 8usize);
        let (hc, hh, hw) = (2usize, 6usize, 6usize);
        let (yc, yh, yw) = (2usize, 4usize, 4usize);
        let e0 = 1e-4f32;
        let mut s = 0x7A55_2026u64;

        let c0_patches =
            ArrayD::from_shape_fn(IxDyn(&[rows, 2, 2, 2, yc, 3, 3]), |_| 0.5 * lcg(&mut s));
        let c0_data = PatchesData {
            coeff_err: Some(Array1::from_elem(rows, e0)),
            patches: Some(c0_patches.clone()),
            geometry: PatchGeometry::affine((1, 1), (0, 0, 0, 0)),
            identity: false,
            output_shape: (2, 2, 2),
            input_shape: (yc, yh, yw),
            unstable_idx: None,
        };
        let c0 = PatchesLinearBounds {
            row_count: rows,
            lower_a: c0_data.clone(),
            lower_b: Array1::zeros(rows),
            upper_a: c0_data,
            upper_b: Array1::zeros(rows),
        };

        let k1 = ArrayD::from_shape_fn(IxDyn(&[hc, xc, 3, 3]), |_| 0.06 * lcg(&mut s));
        let k2 = ArrayD::from_shape_fn(IxDyn(&[yc, hc, 3, 3]), |_| 0.06 * lcg(&mut s));
        let mut conv1 = Conv2dLayer::new(k1.clone(), None, (1, 1), (0, 0)).unwrap();
        conv1.input_shape = Some((xh, xw));
        let mut conv2 = Conv2dLayer::new(k2.clone(), None, (1, 1), (0, 0)).unwrap();
        conv2.input_shape = Some((hh, hw));

        let x_lo = ArrayD::from_shape_fn(IxDyn(&[xc, xh, xw]), |_| -0.4 + 0.05 * lcg(&mut s));
        let x_hi = ArrayD::from_shape_fn(IxDyn(&[xc, xh, xw]), |_| 0.35 + 0.05 * lcg(&mut s));
        let x_box = BoundedTensor::new(x_lo.clone(), x_hi.clone()).unwrap();

        // f64 interval conv (kernel taps distribute over the box), outward to
        // f32 — the verifier's collected pre-activation cuts.
        let ibp_conv = |kern: &ArrayD<f32>,
                        lo: &ArrayD<f32>,
                        hi: &ArrayD<f32>,
                        oc_n: usize,
                        oh_n: usize,
                        ow_n: usize,
                        ic_n: usize|
         -> (ArrayD<f32>, ArrayD<f32>) {
            let mut out_lo = ArrayD::<f32>::zeros(IxDyn(&[oc_n, oh_n, ow_n]));
            let mut out_hi = ArrayD::<f32>::zeros(IxDyn(&[oc_n, oh_n, ow_n]));
            for oc in 0..oc_n {
                for oh in 0..oh_n {
                    for ow in 0..ow_n {
                        let (mut alo, mut ahi) = (0.0f64, 0.0f64);
                        for ic in 0..ic_n {
                            for ki in 0..3 {
                                for kj in 0..3 {
                                    let k = f64::from(kern[[oc, ic, ki, kj]]);
                                    let a = k * f64::from(lo[[ic, oh + ki, ow + kj]]);
                                    let b = k * f64::from(hi[[ic, oh + ki, ow + kj]]);
                                    alo += a.min(b);
                                    ahi += a.max(b);
                                }
                            }
                        }
                        out_lo[[oc, oh, ow]] = next_down_f32(alo as f32);
                        out_hi[[oc, oh, ow]] = next_up_f32(ahi as f32);
                    }
                }
            }
            (out_lo, out_hi)
        };
        let (h_lo, h_hi) = ibp_conv(&k1, &x_lo, &x_hi, hc, hh, hw, xc);
        let h_box = BoundedTensor::new(h_lo.clone(), h_hi.clone()).unwrap();
        let g_lo = h_lo.mapv(|v| v.max(0.0));
        let g_hi = h_hi.mapv(|v| v.max(0.0));
        let (y_lo, y_hi) = ibp_conv(&k2, &g_lo, &g_hi, yc, yh, yw, hc);
        let y_box = BoundedTensor::new(y_lo, y_hi).unwrap();

        // Backward walk, OFF (today) and ON (+ eager discharge after each
        // ReLU backward, against that step's pre-activation cut).
        let relu = crate::layers::activations::relu::relu_linear_relaxation;
        let unwrap_patches = |cb: CrownBounds| -> PatchesLinearBounds {
            match cb {
                CrownBounds::Patches(p) => *p,
                CrownBounds::Dense(_) => panic!("cascade must stay in patches mode"),
            }
        };

        let after_relu2 =
            unwrap_patches(crown_elementwise_backward_patches(&c0, &y_box, relu).unwrap());
        let carried = after_relu2.lower_a.coeff_err.as_ref().expect("carry")[0];
        assert!(
            carried > e0 * 0.5 && carried.is_finite(),
            "post-ReLU carried err must exist (got {carried})"
        );

        let walk_tail = |start: &PatchesLinearBounds, discharge: bool| -> crate::LinearBounds {
            let mut cur = start.clone();
            if discharge {
                cur.fold_coeff_err_over_box_eager_with_policy(&y_box, true);
                assert!(
                    cur.lower_a.coeff_err.is_none() && cur.upper_a.coeff_err.is_none(),
                    "7D discharge must clear the carried error"
                );
            }
            cur = unwrap_patches(conv2.propagate_patches_engine(&cur, None).unwrap());
            cur = unwrap_patches(crown_elementwise_backward_patches(&cur, &h_box, relu).unwrap());
            if discharge {
                cur.fold_coeff_err_over_box_eager_with_policy(&h_box, true);
            }
            cur = unwrap_patches(conv1.propagate_patches_engine(&cur, None).unwrap());
            cur.to_dense().unwrap()
        };
        let off_dense = walk_tail(&after_relu2, false);
        let on_dense = walk_tail(&after_relu2, true);

        // (c) The junction metric of the diagnosis: max-row-L1 of the
        // materialized certified-error matrix.
        let row_l1 = |m: Option<&ndarray::Array2<f32>>| -> f64 {
            m.map(|e| {
                (0..e.nrows())
                    .map(|r| e.row(r).iter().map(|&v| f64::from(v)).sum::<f64>())
                    .fold(0.0f64, f64::max)
            })
            .unwrap_or(0.0)
        };
        let off_l1 =
            row_l1(off_dense.lower_a_err.as_ref()).max(row_l1(off_dense.upper_a_err.as_ref()));
        let on_l1 =
            row_l1(on_dense.lower_a_err.as_ref()).max(row_l1(on_dense.upper_a_err.as_ref()));
        assert!(
            on_l1 > 0.0,
            "intrinsic conv/scatter certificates must survive the discharge"
        );
        let improvement = off_l1 / on_l1;
        assert!(
            improvement >= 100.0,
            "measured cascade row-L1 improvement {improvement:.1}x < 100x \
             (off {off_l1:.3e}, on {on_l1:.3e}; MEASURED 585.8x at authoring)"
        );

        // (a) End-bound comparison + enclosure of concrete forward passes.
        let off_conc = off_dense.concretize_sound(&x_box);
        let on_conc = on_dense.concretize_sound(&x_box);
        for r in 0..rows {
            let woff = f64::from(off_conc.upper()[[r]]) - f64::from(off_conc.lower()[[r]]);
            let won = f64::from(on_conc.upper()[[r]]) - f64::from(on_conc.lower()[[r]]);
            assert!(
                won <= woff + 1e-6,
                "row {r}: ON width {won} looser than OFF width {woff}"
            );
        }

        let mut s2 = 0x0BAD_5EEDu64;
        let xf = |lo: f32, hi: f32, t: f32| f64::from(lo) + f64::from(t) * f64::from(hi - lo);
        for _sample in 0..25 {
            let x: Vec<f64> = x_lo
                .iter()
                .zip(x_hi.iter())
                .map(|(&l, &h)| xf(l, h, (lcg(&mut s2) + 1.0) * 0.5))
                .collect();
            let conv_fwd = |kern: &ArrayD<f32>,
                            inp: &[f64],
                            oc_n: usize,
                            oh_n: usize,
                            ow_n: usize,
                            ic_n: usize,
                            ih_n: usize,
                            iw_n: usize|
             -> Vec<f64> {
                let mut out = vec![0.0f64; oc_n * oh_n * ow_n];
                for oc in 0..oc_n {
                    for oh in 0..oh_n {
                        for ow in 0..ow_n {
                            let mut acc = 0.0f64;
                            for ic in 0..ic_n {
                                for ki in 0..3 {
                                    for kj in 0..3 {
                                        acc += f64::from(kern[[oc, ic, ki, kj]])
                                            * inp[(ic * ih_n + oh + ki) * iw_n + ow + kj];
                                    }
                                }
                            }
                            out[(oc * oh_n + oh) * ow_n + ow] = acc;
                        }
                    }
                }
                out
            };
            let h = conv_fwd(&k1, &x, hc, hh, hw, xc, xh, xw);
            let g: Vec<f64> = h.iter().map(|&v| v.max(0.0)).collect();
            let y = conv_fwd(&k2, &g, yc, yh, yw, hc, hh, hw);
            let z: Vec<f64> = y.iter().map(|&v| v.max(0.0)).collect();
            for r in 0..rows {
                let (mut out_stored, mut out_pert) = (0.0f64, 0.0f64);
                for oc in 0..2 {
                    for oh in 0..2 {
                        for ow in 0..2 {
                            for ic in 0..yc {
                                for ki in 0..3 {
                                    for kj in 0..3 {
                                        let c = f64::from(c0_patches[[r, oc, oh, ow, ic, ki, kj]]);
                                        let zv = z[(ic * yh + oh + ki) * yw + ow + kj];
                                        out_stored += c * zv;
                                        out_pert +=
                                            (c + f64::from(e0) * f64::from(lcg(&mut s2))) * zv;
                                    }
                                }
                            }
                        }
                    }
                }
                for out in [out_stored, out_pert] {
                    assert!(
                        f64::from(off_conc.lower()[[r]]) - 1e-6 <= out
                            && out <= f64::from(off_conc.upper()[[r]]) + 1e-6,
                        "row {r}: OFF bounds lost the true output {out}"
                    );
                    assert!(
                        f64::from(on_conc.lower()[[r]]) - 1e-6 <= out
                            && out <= f64::from(on_conc.upper()[[r]]) + 1e-6,
                        "row {r}: ON bounds lost the true output {out}"
                    );
                }
            }
        }
    }
}
