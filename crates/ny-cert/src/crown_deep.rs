// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Exact-rational CROWN certificate generation for a *deep* ReLU network with
//! an arbitrary number `k` of hidden layers.
//!
//! This generalizes [`crate::crown`] (which handles a single hidden layer) to
//! the multi-layer case. The construction, and the reason it is a *proof* and
//! not merely a heuristic, is unchanged: a CROWN backward pass over a ReLU
//! network is literally the choice of non-negative Farkas / dual multipliers on
//! the relaxed-network constraint system. We assemble that system, run the
//! backward pass while accumulating each multiplier, and emit an
//! entailment + Farkas certificate that Clean's kernel-side external-certificate
//! verifier checks.
//!
//! ## The network
//!
//! For `k` hidden ReLU layers with weight matrices `W⁽¹⁾ … W⁽ᵏ⁾` and a final
//! scalar affine read-out `W⁽ᵏ⁺¹⁾`:
//!
//! ```text
//!   z⁽¹⁾ = W⁽¹⁾·x       + b⁽¹⁾
//!   a⁽¹⁾ = ReLU(z⁽¹⁾)
//!   z⁽²⁾ = W⁽²⁾·a⁽¹⁾    + b⁽²⁾
//!   a⁽²⁾ = ReLU(z⁽²⁾)
//!     ⋮
//!   z⁽ᵏ⁾ = W⁽ᵏ⁾·a⁽ᵏ⁻¹⁾ + b⁽ᵏ⁾
//!   a⁽ᵏ⁾ = ReLU(z⁽ᵏ⁾)
//!   y     = W⁽ᵏ⁺¹⁾·a⁽ᵏ⁾ + b⁽ᵏ⁺¹⁾     (scalar)
//! ```
//! over an input box `x ∈ [l, u]`. We verify a safety property `y ≥ threshold`.
//!
//! ## Pre-activation bounds (IBP, layer by layer)
//!
//! Interval Bound Propagation: given a box on the previous layer's *activations*
//! `a⁽ᴸ⁻¹⁾ ∈ [aₗ, aᵤ]` (for `L = 1`, that is the input box `x ∈ [l, u]`), the
//! affine layer gives `z⁽ᴸ⁾` interval bounds, and ReLU then gives the activation
//! box `a⁽ᴸ⁾ ∈ [max(0,zₗ), max(0,zᵤ)]` for the next layer. These exact-rational
//! pre-activation bounds drive the per-unit ReLU envelope selection.
//!
//! ## The certificate
//!
//! Every relaxation fact is a linear inequality over the variables
//! `xᵢ, z⁽ᴸ⁾ⱼ, a⁽ᴸ⁾ⱼ, y`:
//!
//! * box bounds `lᵢ ≤ xᵢ ≤ uᵢ`,
//! * each affine layer, supplied as a `≤`/`≥` pair (Clean scales an `eq`
//!   premise by a single multiplier, which cancels; so equalities must be split
//!   to obtain a signed effective weight from two non-negative multipliers),
//! * per-unit ReLU envelopes: a lower envelope `a⁽ᴸ⁾ⱼ ≥ pⱼ·z⁽ᴸ⁾ⱼ + qⱼ` and an
//!   upper envelope `a⁽ᴸ⁾ⱼ ≤ rⱼ·z⁽ᴸ⁾ⱼ + tⱼ`.
//!
//! The CROWN backward pass starts from `−y + Σ W⁽ᵏ⁺¹⁾ⱼ·a⁽ᵏ⁾ⱼ ≤ −b⁽ᵏ⁺¹⁾`, then,
//! layer by layer from `k` down to `1`, eliminates each `a⁽ᴸ⁾ⱼ` by its
//! sign-appropriate ReLU envelope (lower envelope when the running coefficient
//! on `a⁽ᴸ⁾ⱼ` is positive, upper when negative) and each `z⁽ᴸ⁾ⱼ` through the
//! affine layer (which re-expresses it in the previous layer's activations,
//! i.e. `a⁽ᴸ⁻¹⁾` or, at `L = 1`, the inputs `x`). Finally each `xᵢ` is
//! eliminated through the box. Every elimination *is* the choice of a
//! non-negative multiplier on the corresponding inequality. The accumulated
//! combination is exactly `−y ≤ −m`, i.e. `y ≥ m`, the CROWN lower bound.

use crate::rational::{Rat, RatError};
use crate::schema::{ConstraintKind, EntailmentCertificate, FarkasCertificate, LinearConstraint};
// Contracts are written as the BARE `#[ensures]` (see `selfcheck.rs` for the full
// rationale): under tRustc contract verification (`--cfg trust_verify`) it is the
// first-class builtin that emits a static postcondition VC, so the NY-owned
// compatibility macro must NOT be imported then or it shadows the builtin and
// degrades the contract to a runtime-checked closure. Under stable rustc the macro
// provides the no-op `#[ensures]`. The path-qualified `#[trust::requires(...)]`
// and `#[trust::cite(...)]` attributes resolve through the extern-prelude `trust`
// crate in both builds (this `use` also pulls it in, silencing the unused-dep lint).
#[cfg(trust_verify)]
use core::contracts::ensures;
#[cfg(not(trust_verify))]
use trust::ensures;

/// Errors that can arise while building a deep certificate.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DeepCrownError {
    /// A matrix/vector dimension did not line up.
    #[error("dimension mismatch: {0}")]
    Dimension(String),
    /// The network has no hidden layers.
    #[error("a deep network needs at least one hidden layer")]
    NoHiddenLayers,
    /// An α slope was outside `[0, 1]` (would make the lower envelope unsound).
    #[error("alpha[layer {0}][unit {1}] must lie in [0, 1]")]
    AlphaOutOfRange(usize, usize),
    /// The requested safety threshold exceeds the certified lower bound.
    #[error("threshold {threshold} exceeds certified lower bound {bound}")]
    ThresholdAboveBound {
        /// Requested threshold.
        threshold: String,
        /// Best certified lower bound.
        bound: String,
    },
    /// Exact arithmetic failure.
    #[error(transparent)]
    Rat(#[from] RatError),
    /// A CROWN f64 intermediate bound was non-finite (NaN/±inf). We fail closed
    /// rather than convert it via a saturating `as i128` cast, which could clamp
    /// a lower bound up / an upper bound down and yield an unsound (too-tight)
    /// snapped enclosure.
    #[error("non-finite CROWN intermediate bound: {0}")]
    NonFiniteBound(String),
    /// A 2-neuron joint cut was requested on a network whose input dimension is
    /// too large to enumerate `2ⁿ` box corners. We fail closed: the corner sweep
    /// uses a `u32` mask (`1u32 << n`), which would *wrap* for `n ≥ 32` and yield
    /// a silently WRONG joint upper bound `B_ij` — an UNSOUND cut premise. Rather
    /// than emit such a premise we reject the request. (`2ⁿ` is in any case far
    /// too expensive for `n` anywhere near this bound.)
    #[error("cut box-corner enumeration needs input dim < 32, got {0}")]
    CutDimTooLarge(usize),
    /// The caller-installed wall-clock budget ([`crate::budget`]) expired while
    /// building the certificate. Certification is post-verdict, optional work,
    /// so this fails OPEN to "no certificate" — the verdict is unaffected.
    #[error("certificate construction exceeded its wall-clock budget")]
    BudgetExceeded,
    /// The thread's rational interning arena is poisoned: one of the
    /// unreachable-by-construction fallback arms in [`crate::rational`] was
    /// reached and silently substituted `0/1` for an arbitrary value. Every
    /// `Rat` touched on this thread since then is suspect, so certificate
    /// construction fails CLOSED (checked at entry and again before returning
    /// a built certificate — a poison that appears mid-build must also refuse).
    #[error("rational arena poisoned (fallback arm was reached); certificate must not be trusted")]
    ArenaPoisoned,
}

/// Return `Err(BudgetExceeded)` iff the thread's certificate budget
/// ([`crate::budget`]) has expired. TOTAL; a `()` result means "keep going".
/// Free fn (not a closure) so loop-boundary polls stay in verified code.
fn budget_check() -> Result<(), DeepCrownError> {
    if crate::budget::expired() {
        return Err(crate::err_barrier(DeepCrownError::BudgetExceeded));
    }
    Ok(())
}

/// A deep `k`-hidden-layer ReLU network plus an input box.
///
/// Layer `L` (1-indexed, `L = 1 … k`) maps activations of dimension
/// `dim(L−1)` to pre-activations of dimension `dim(L)`, where `dim(0)` is the
/// input dimension. `weights[L−1]` has shape `[dim(L)][dim(L−1)]`, and
/// `biases[L−1]` has length `dim(L)`. The final read-out `out_weight` /
/// `out_bias` maps the last activation `a⁽ᵏ⁾` (dimension `dim(k)`) to the scalar
/// output.
#[derive(Debug, Clone)]
pub struct DeepReluProblem {
    /// Hidden weight matrices `W⁽¹⁾ … W⁽ᵏ⁾`; `weights[L-1]` is `[dim(L)][dim(L-1)]`.
    pub weights: Vec<Vec<Vec<Rat>>>,
    /// Hidden biases `b⁽¹⁾ … b⁽ᵏ⁾`; `biases[L-1]` has length `dim(L)`.
    pub biases: Vec<Vec<Rat>>,
    /// Scalar read-out weight `W⁽ᵏ⁺¹⁾`, length `dim(k)`.
    pub out_weight: Vec<Rat>,
    /// Scalar read-out bias `b⁽ᵏ⁺¹⁾`.
    pub out_bias: Rat,
    /// Input lower bounds, length `dim(0)`.
    pub input_lower: Vec<Rat>,
    /// Input upper bounds, length `dim(0)`.
    pub input_upper: Vec<Rat>,
    /// Optional per-layer CROWN lower-envelope slopes `α⁽ᴸ⁾ⱼ ∈ [0, 1]`. When
    /// `None`, the adaptive default (`1` if `uⱼ ≥ −lⱼ`, else `0`) is used per
    /// unstable unit.
    pub alpha: Option<Vec<Vec<Rat>>>,
    /// When `true`, [`preact_bounds_crown`] outward-rounds each layer's exact
    /// intermediate bounds to 53-significant-bit dyadic rationals before they
    /// feed the next layer (see that method). Sound either way (outward rounding
    /// only widens, so every envelope stays a valid ReLU relaxation and the cert
    /// stays exactly Clean-checkable); the trade is tightness (relative loss
    /// ≤ 2⁻⁵²) for bounded bit-length on deep nets. Explicit config — NOT read
    /// from the environment — so a certificate's content is a pure function of
    /// its `DeepReluProblem`. Default `false` (fully exact).
    ///
    /// [`preact_bounds_crown`]: DeepReluProblem::preact_bounds_crown
    pub interm_round: bool,
}

/// Per-layer pre-activation bounds produced by IBP.
#[derive(Debug, Clone)]
pub struct PreactBounds {
    /// `lower[L-1][j]` = IBP lower bound on `z⁽ᴸ⁾ⱼ`.
    pub lower: Vec<Vec<Rat>>,
    /// `upper[L-1][j]` = IBP upper bound on `z⁽ᴸ⁾ⱼ`.
    pub upper: Vec<Vec<Rat>>,
}

/// A certified verification result for a [`DeepReluProblem`].
#[derive(Debug, Clone)]
pub struct CertifiedDeep {
    /// Entailment certificate proving `y ≥ threshold`.
    pub entailment: EntailmentCertificate,
    /// Farkas certificate proving the unsafe region `y < threshold` is empty.
    pub farkas: FarkasCertificate,
    /// The CROWN lower bound `m` on the output (`y ≥ m`).
    pub lower_bound: Rat,
    /// Per-layer IBP pre-activation bounds.
    pub preact: PreactBounds,
}

/// One squared term `q · (a·x + b)²` of a QUADRATIC ground-truth side (see
/// [`DeepReluProblem::certify_difference_quadratic`]): `coeff` is the exact
/// rational multiplier `q` on the square (any sign), and `lin`/`offset` are the
/// exact rational affine pre-square `t(x) = a·x + b` over the network inputs.
#[derive(Debug, Clone)]
pub struct QuadTerm {
    /// Multiplier `q` on the square (any sign).
    pub coeff: Rat,
    /// Affine coefficients `a` of the pre-square `t = a·x + b` (input dim).
    pub lin: Vec<Rat>,
    /// Affine offset `b` of the pre-square.
    pub offset: Rat,
}

/// Per-term bookkeeping for the quadratic-side premises assembled by
/// `certify_impl`: the premise indices of the definitional `t` pair, the
/// secant, and the tangent, plus the exact interval `[l, u]` of the affine
/// pre-square over the input box and the chosen tangency point `c`.
struct QuadPremises {
    coeff: Rat,
    lin: Vec<Rat>,
    offset: Rat,
    t_le: usize,
    t_ge: usize,
    secant: usize,
    tangent: usize,
    lo: Rat,
    hi: Rat,
    tangency: Rat,
}

/// Lossless `f64 -> Rat` (every finite f64 is the dyadic `m · 2^e`).
fn f64_to_rat_exact(v: f64) -> Option<Rat> {
    Rat::from_f64_exact(v)
}

/// Next `f64` strictly above `v` (finite `v`; bit-level `nextafter`, avoiding
/// `f64::next_up` which is stable only since Rust 1.86 — crate MSRV is 1.85).
fn f64_next_up_compat(v: f64) -> f64 {
    if v == 0.0 {
        return f64::from_bits(1); // smallest positive subnormal
    }
    let bits = v.to_bits();
    if v > 0.0 {
        // behavior-identical: `v > 0.0` ⇒ the sign bit is clear ⇒ `bits < 2^63`,
        // so `bits + 1` can never overflow u64; `wrapping_add(1)` equals `+ 1`.
        f64::from_bits(bits.wrapping_add(1))
    } else {
        // behavior-identical: reached only when `v < 0.0` (sign bit set ⇒
        // `bits ≥ 2^63`) or `v` is NaN (`bits ≥ 0x7FF0…0001`); the `v == 0.0`
        // case (bits 0 or 2^63) was returned above, so `bits ≥ 1` always and
        // `bits - 1` can never underflow; `wrapping_sub(1)` equals `- 1`.
        f64::from_bits(bits.wrapping_sub(1))
    }
}

/// Next `f64` strictly below `v` (finite `v`).
fn f64_next_down_compat(v: f64) -> f64 {
    -f64_next_up_compat(-v)
}

/// Outward-round an exact rational bound to a (≤ 53-significant-bit) dyadic:
/// `lower = true` rounds DOWN (result ≤ `r`), else UP (result ≥ `r`).
///
/// SOUND by construction: the returned value is verified to lie on the outward
/// side of `r` before it is used, so an interval `[round(l,↓), round(u,↑)]`
/// always CONTAINS `[l, u]` and every ReLU envelope built from it remains
/// valid. Falls back to the exact `r` (no rounding) when `r` overflows `f64` —
/// never returns an inward-rounded value.
fn dyadic_round_out(r: Rat, lower: bool) -> Rat {
    let mut v = r.to_f64_approx();
    if !v.is_finite() {
        return r;
    }
    // The nearest-f64 is within 1 ulp; nudge outward until on the correct side.
    for _ in 0..3 {
        if let Some(d) = f64_to_rat_exact(v) {
            if (lower && d <= r) || (!lower && d >= r) {
                return d;
            }
        } else {
            return r;
        }
        v = if lower {
            f64_next_down_compat(v)
        } else {
            f64_next_up_compat(v)
        };
    }
    r
}

/// Exact `min`/`max` of `Σ wᵢ·vᵢ` over `vᵢ ∈ [loᵢ, hiᵢ]` (IBP for one row).
fn dot_extreme(weights: &[Rat], lo: &[Rat], hi: &[Rat], want_min: bool) -> Result<Rat, RatError> {
    let mut acc = Rat::ZERO;
    for ((w, l), u) in weights.iter().zip(lo).zip(hi) {
        let pick = if (w.is_positive() && want_min) || (w.is_negative() && !want_min) {
            *l
        } else {
            *u
        };
        acc = acc.add(w.mul(pick)?)?;
    }
    Ok(acc)
}

/// Append the decimal digits of `v` to `s` — byte-identical to
/// `format!("{v}")` for every `usize`, with no `core::fmt` dispatch (the
/// extern formatting machinery may run arbitrary `Display` code, so it is a
/// trusted-assumption row under the strict verifier; this manual divmod
/// digit-push construction is fully statically verified instead).
fn push_usize_ascii(s: &mut String, v: usize) {
    if v == 0 {
        s.push('0');
        return;
    }
    // Least-significant-first digit bytes. `x % 10 <= 9`, so `wrapping_add`
    // equals the exact `b'0' + digit` (never wraps) with no overflow VC; the
    // `% 10` / `/ 10` are by a nonzero constant (no div-by-zero VC).
    let mut x = v;
    let mut rev: Vec<u8> = Vec::new();
    while x > 0 {
        // checked forms by the constant 10 (fallbacks unreachable): no MIR
        // zero-divisor assert, unlike bare `%`/`/` (see sbar::pvar).
        rev.push(b'0'.wrapping_add((x.checked_rem(10).unwrap_or(0)) as u8));
        x = x.checked_div(10).unwrap_or(0);
    }
    // Emit most-significant first. `i` starts at `rev.len()` and is
    // decremented BEFORE each read, so `i < rev.len()` and the `b'0'` fallback
    // is unreachable — a total read, no `[]` obligation. `saturating_sub`
    // never saturates (`i > 0` inside the loop), so it equals `i - 1` exactly.
    let mut i = rev.len();
    while i > 0 {
        i = i.saturating_sub(1);
        s.push(rev.get(i).copied().unwrap_or(b'0') as char);
    }
}

impl DeepReluProblem {
    fn input_dim(&self) -> usize {
        self.input_lower.len()
    }

    /// Number of hidden layers `k`.
    #[must_use]
    pub fn depth(&self) -> usize {
        self.weights.len()
    }

    /// Width of hidden layer `L` (1-indexed): `dim(L)`.
    ///
    /// `layer1` is a **1-indexed** layer tag, always `>= 1` at every call site
    /// (callers pass `li + 1` or a value already guarded `>= 1`). To make the
    /// `layer1 - 1` index provably free of usize underflow *without* imposing a
    /// call-site precondition the callers cannot statically discharge, fail
    /// closed locally: `checked_sub(1)` yields the 0-indexed layer, and the
    /// `unwrap_or(0)` branch is only reachable for the (never-passed) `layer1 ==
    /// 0`. The verifier discharges the subtraction obligation from the
    /// `checked_sub` instead of a raw `- 1` that could wrap.
    fn layer_dim(&self, layer1: usize) -> usize {
        #[allow(clippy::manual_saturating_arithmetic)]
        let layer0 = layer1.checked_sub(1).unwrap_or(0);
        // Total read (fail-safe 0): callers pass `layer1 ∈ [1, k]`, so
        // `layer0 ∈ [0, k-1]` indexes `weights` (len `k`) — the `0` fallback is
        // unreachable. Keeps the width read free of a slice-bounds obligation.
        match self.weights.get(layer0) {
            Some(w) => w.len(),
            None => 0,
        }
    }

    fn validate(&self) -> Result<(), DeepCrownError> {
        let k = self.depth();
        if k == 0 {
            return Err(DeepCrownError::NoHiddenLayers);
        }
        if self.biases.len() != k {
            return Err(DeepCrownError::Dimension(format!(
                "weights has {k} layers but biases has {}",
                self.biases.len()
            )));
        }
        if self.input_upper.len() != self.input_dim() {
            return Err(DeepCrownError::Dimension(
                "input bounds length differ".into(),
            ));
        }
        let mut prev_dim = self.input_dim();
        for (li, (w, b)) in self.weights.iter().zip(&self.biases).enumerate() {
            let this_dim = w.len();
            if b.len() != this_dim {
                return Err(DeepCrownError::Dimension(format!(
                    "layer {} bias len {} != layer width {this_dim}",
                    // Display-only 1-indexing; saturating form keeps the
                    // error path free of an (unreachable) overflow VC on the
                    // havoced enumerate counter.
                    li.saturating_add(1),
                    b.len()
                )));
            }
            for (j, row) in w.iter().enumerate() {
                if row.len() != prev_dim {
                    return Err(DeepCrownError::Dimension(format!(
                        "layer {} row {j} width {} != prev dim {prev_dim}",
                        li.saturating_add(1),
                        row.len()
                    )));
                }
            }
            prev_dim = this_dim;
        }
        if self.out_weight.len() != prev_dim {
            return Err(DeepCrownError::Dimension(format!(
                "out_weight len {} != last hidden width {prev_dim}",
                self.out_weight.len()
            )));
        }
        Ok(())
    }

    /// Layer-by-layer IBP pre-activation bounds `l⁽ᴸ⁾, u⁽ᴸ⁾` for every hidden
    /// layer.
    ///
    /// # Errors
    /// Propagates exact-rational arena failures.
    pub fn preact_bounds(&self) -> Result<PreactBounds, DeepCrownError> {
        crate::rational::ensure_healthy()?;
        self.validate()?;
        let k = self.depth();
        // `Vec::new()` (not `with_capacity(k)`): the capacity hint on the
        // unbounded `&self` depth carries a hardened allocation obligation the
        // model cannot bound; amortized growth is noise next to the Rat math.
        let mut lower = Vec::new();
        let mut upper = Vec::new();

        // Activation box feeding the current layer; starts as the input box.
        let mut act_lo = self.input_lower.clone();
        let mut act_hi = self.input_upper.clone();

        for li in 0..k {
            budget_check()?;
            // `li < k`; `weights`/`biases` have `k` entries (`validate`), so the
            // empty-slice fallbacks are unreachable (a no-op empty iteration).
            let w = match self.weights.get(li) {
                Some(v) => v.as_slice(),
                None => &[],
            };
            let b = match self.biases.get(li) {
                Some(v) => v.as_slice(),
                None => &[],
            };
            // Same rationale as `lower`/`upper` above: the unbounded layer
            // width makes the capacity hint an unboundable allocation VC.
            let mut zl = Vec::new();
            let mut zu = Vec::new();
            for (row, bias) in w.iter().zip(b) {
                let lo = dot_extreme(row, &act_lo, &act_hi, true)?.add(*bias)?;
                let hi = dot_extreme(row, &act_lo, &act_hi, false)?.add(*bias)?;
                zl.push(lo);
                zu.push(hi);
            }
            // ReLU the pre-activation box to get the next layer's activation box.
            // Explicit loops (not `.map(..).collect()`): the `|v| ..` closure
            // would be invoked through an absent `<{closure} as Fn>::call` shim,
            // and the bulk `.collect()` carries an unbounded-count allocation
            // obligation; identical elements and order.
            let mut next_lo: Vec<Rat> = Vec::new();
            for v in &zl {
                next_lo.push(if v.is_negative() { Rat::ZERO } else { *v });
            }
            act_lo = next_lo;
            let mut next_hi: Vec<Rat> = Vec::new();
            for v in &zu {
                next_hi.push(if v.is_negative() { Rat::ZERO } else { *v });
            }
            act_hi = next_hi;
            lower.push(zl);
            upper.push(zu);
        }
        crate::rational::ensure_healthy()?;
        Ok(PreactBounds { lower, upper })
    }

    /// EXACT CROWN bound of a linear functional `const0 + Σ z_coeff0[j]·z⁽ᵘᵖ⁾ⱼ`
    /// over the input box, substituting back through the affine layers and ReLU
    /// envelopes (using already-computed preact bounds for layers `< up_to`).
    /// `want_lower` selects min vs max. Mirrors the f64 `crown_bound_z`; every
    /// envelope used is a valid over/under-approximation, so the returned bound is
    /// a sound bound on the functional. Used to compute tight intermediate bounds.
    #[allow(clippy::too_many_arguments)]
    fn crown_bound_z_exact(
        &self,
        pre_lo: &[Vec<Rat>],
        pre_hi: &[Vec<Rat>],
        up_to: usize,
        z_coeff0: &[Rat],
        const0: Rat,
        want_lower: bool,
    ) -> Result<Rat, DeepCrownError> {
        budget_check()?;
        let n = self.input_dim();
        let li0 = up_to;
        let mut const_acc = const0;
        // Total accumulator update + reads (fail-safe `Rat::ZERO` / skipped row):
        // every index is a loop counter bounded by the container it indexes
        // (`weights[·].len()`, `row.len()`, `n`) or an `li0`/`li` guarded
        // `< weights.len()`, so each fallback is UNREACHABLE — identical bound,
        // no slice-bounds obligations.
        // Free (nested) `fn`, not a closure: called directly rather than through
        // an absent `<{closure} as Fn>::call` shim, so the accumulator update
        // stays in verified code. Captures nothing (only its params + `Rat::ZERO`).
        fn add_at(v: &mut [Rat], idx: usize, delta: Rat) -> Result<(), RatError> {
            let cur = v.get(idx).copied().unwrap_or(Rat::ZERO);
            let next = cur.add(delta)?;
            if let Some(slot) = v.get_mut(idx) {
                *slot = next;
            }
            Ok(())
        }
        // Free (nested) `fn`s, not closures: called directly rather than through
        // an absent `<{closure} as Fn>::call` shim (the captured `self.biases`
        // becomes the explicit `biases` parameter), and each nested lookup is a
        // plain `match` rather than a closure passed to `Option::and_then` —
        // identical value on every input.
        fn bias_at(biases: &[Vec<Rat>], li: usize, j: usize) -> Rat {
            match biases.get(li) {
                Some(b) => b.get(j).copied().unwrap_or(Rat::ZERO),
                None => Rat::ZERO,
            }
        }
        fn pre_at(m: &[Vec<Rat>], li: usize, jj: usize) -> Rat {
            match m.get(li) {
                Some(r) => r.get(jj).copied().unwrap_or(Rat::ZERO),
                None => Rat::ZERO,
            }
        }
        let prev_dim = if li0 == 0 { n } else { self.layer_dim(li0) };
        // `Vec::new()` + push (not `vec![_; prev_dim]`): the bulk fill carries a
        // hardened allocation obligation unbounded on the network dimension; the
        // push loop yields the identical `prev_dim`-zero coefficient vector.
        let mut a_coeff = Vec::new();
        for _ in 0..prev_dim {
            a_coeff.push(Rat::ZERO);
        }
        for j in 0..match self.weights.get(li0) {
            Some(w) => w.len(),
            None => 0,
        } {
            let c = z_coeff0.get(j).copied().unwrap_or(Rat::ZERO);
            if c.is_zero() {
                continue;
            }
            // Plain `match` flattening (not `.and_then(|l| ..)`): the closure
            // would be invoked through an absent `Fn::call` shim; identical row
            // selection.
            let row_opt = match self.weights.get(li0) {
                Some(l) => l.get(j),
                None => None,
            };
            if let Some(row) = row_opt {
                for (i, wji) in row.iter().enumerate() {
                    add_at(&mut a_coeff, i, c.mul(*wji)?)?;
                }
            }
            const_acc = const_acc.add(c.mul(bias_at(&self.biases, li0, j))?)?;
        }
        if li0 == 0 {
            for i in 0..n {
                let d = a_coeff.get(i).copied().unwrap_or(Rat::ZERO);
                if d.is_positive() {
                    let pick = if want_lower {
                        self.input_lower.get(i).copied().unwrap_or(Rat::ZERO)
                    } else {
                        self.input_upper.get(i).copied().unwrap_or(Rat::ZERO)
                    };
                    const_acc = const_acc.add(d.mul(pick)?)?;
                } else if d.is_negative() {
                    let pick = if want_lower {
                        self.input_upper.get(i).copied().unwrap_or(Rat::ZERO)
                    } else {
                        self.input_lower.get(i).copied().unwrap_or(Rat::ZERO)
                    };
                    const_acc = const_acc.add(d.mul(pick)?)?;
                }
            }
            return Ok(const_acc);
        }
        // Forward index (not `(0..li0).rev()`): the `Rev<Range>` adapter is an
        // absent-callee for the panic-freedom checker; `li = li0-1-idx` reverses
        // the walk exactly. Saturating subs match the file idiom and are exact
        // here (li0 >= 1 in-body, idx <= li0-1).
        for idx in 0..li0 {
            let li = li0.saturating_sub(1).saturating_sub(idx);
            budget_check()?;
            let width = match self.weights.get(li) {
                Some(w) => w.len(),
                None => 0,
            };
            // `Vec::new()` + push (not `vec![_; width]`): unbounded-count bulk
            // fill → hardened allocation obligation; push loop is identical.
            let mut z_coeff = Vec::new();
            for _ in 0..width {
                z_coeff.push(Rat::ZERO);
            }
            for jj in 0..width {
                let d = a_coeff.get(jj).copied().unwrap_or(Rat::ZERO);
                if d.is_zero() {
                    continue;
                }
                let (l, u) = (pre_at(pre_lo, li, jj), pre_at(pre_hi, li, jj));
                let (p, q, r, t) = if !l.is_negative() {
                    (Rat::ONE, Rat::ZERO, Rat::ONE, Rat::ZERO)
                } else if !u.is_positive() {
                    (Rat::ZERO, Rat::ZERO, Rat::ZERO, Rat::ZERO)
                } else {
                    let s = u.mul(u.sub(l)?.inv()?)?;
                    let alpha = if u >= l.neg() { Rat::ONE } else { Rat::ZERO };
                    (alpha, Rat::ZERO, s, s.mul(l.neg())?)
                };
                let use_lower_env = d.is_positive() == want_lower;
                if use_lower_env {
                    add_at(&mut z_coeff, jj, d.mul(p)?)?;
                    const_acc = const_acc.add(d.mul(q)?)?;
                } else {
                    add_at(&mut z_coeff, jj, d.mul(r)?)?;
                    const_acc = const_acc.add(d.mul(t)?)?;
                }
            }
            let prev = if li == 0 { n } else { self.layer_dim(li) };
            // `Vec::new()` + push (not `vec![_; prev]`): unbounded-count bulk fill
            // → hardened allocation obligation; push loop is identical.
            let mut prev_coeff = Vec::new();
            for _ in 0..prev {
                prev_coeff.push(Rat::ZERO);
            }
            for jj in 0..width {
                let c = z_coeff.get(jj).copied().unwrap_or(Rat::ZERO);
                if c.is_zero() {
                    continue;
                }
                // Plain `match` flattening (not `.and_then(|l| ..)`): same
                // absent `Fn::call` shim rationale as above.
                let row_opt = match self.weights.get(li) {
                    Some(l) => l.get(jj),
                    None => None,
                };
                if let Some(row) = row_opt {
                    for (i, wji) in row.iter().enumerate() {
                        add_at(&mut prev_coeff, i, c.mul(*wji)?)?;
                    }
                }
                const_acc = const_acc.add(c.mul(bias_at(&self.biases, li, jj))?)?;
            }
            a_coeff = prev_coeff;
        }
        for i in 0..n {
            let d = a_coeff.get(i).copied().unwrap_or(Rat::ZERO);
            if d.is_positive() {
                let pick = if want_lower {
                    self.input_lower.get(i).copied().unwrap_or(Rat::ZERO)
                } else {
                    self.input_upper.get(i).copied().unwrap_or(Rat::ZERO)
                };
                const_acc = const_acc.add(d.mul(pick)?)?;
            } else if d.is_negative() {
                let pick = if want_lower {
                    self.input_upper.get(i).copied().unwrap_or(Rat::ZERO)
                } else {
                    self.input_lower.get(i).copied().unwrap_or(Rat::ZERO)
                };
                const_acc = const_acc.add(d.mul(pick)?)?;
            }
        }
        Ok(const_acc)
    }

    /// EXACT CROWN intermediate pre-activation bounds (tighter than IBP on deep
    /// nets). For each hidden layer L and unit j, bound `z⁽ᴸ⁾ⱼ` by a CROWN
    /// backward pass using the preact bounds of layers `< L`. The bounds are valid
    /// (contain the true preact range), so the ReLU envelopes built from them stay
    /// sound; the Clean kernel checks the resulting Farkas combination unchanged.
    ///
    /// This is the fully-exact (rational) variant, and the one the pipeline uses.
    /// A former f64 "snapped" fast variant (f64 CROWN + outward snap) was removed:
    /// it was unused and its heuristic `|x|·1e-9 + 1e-9` inflation was not a
    /// certified bound on the f64 rounding error, so its premise could be
    /// unsoundly too tight — the exact variant here is both sound and
    /// equal-or-tighter.
    ///
    /// The [`interm_round`] config (opt-in, default off) additionally
    /// OUTWARD-ROUNDS each layer's exact bounds to 53-significant-bit dyadic
    /// rationals before they feed the next layer's envelopes
    /// ([`dyadic_round_out`]). Rounding a bound outward only WIDENS the interval,
    /// so every envelope built from it remains a valid ReLU relaxation and the
    /// emitted certificate stays exactly checkable — this is a certified-sound
    /// tightness-for-bit-length trade (relative loss ≤ 2⁻⁵²) that stops the
    /// multiplicative bit-length blow-up of the envelope slopes `s = u/(u−l)` on
    /// deep (6+ layer) nets, where the fully-exact variant's per-unit backward
    /// passes otherwise cost minutes. It is EXPLICIT config on the problem, not
    /// an environment read: the certificate content is a pure function of the
    /// `DeepReluProblem`, so two runs on the same input are bit-identical.
    ///
    /// [`interm_round`]: DeepReluProblem::interm_round
    ///
    /// # Errors
    /// Propagates exact-rational arena failures.
    pub fn preact_bounds_crown(&self) -> Result<PreactBounds, DeepCrownError> {
        crate::rational::ensure_healthy()?;
        self.validate()?;
        let round = self.interm_round;
        let k = self.depth();
        // Allocation caps (`.min(1 << 20)`): syntactic `min(count, C) <= C <
        // 2^28` bounds the strict allocation checker consumes. `k` is the layer
        // count and `width` a layer width — both far below `1 << 20` for any
        // real network — so every `.min` here is the identity
        // (behavior-preserving). Same convention as `exact::solve_system`.
        // `Vec::new()` + push (not `with_capacity`/`vec![;n]`): the allocation
        // checker fail-closes the capped bulk-alloc forms here; incremental push
        // growth carries no bulk-alloc obligation (same durable fix as
        // `exact::solve_system`).
        let mut lower: Vec<Vec<Rat>> = Vec::new();
        let mut upper: Vec<Vec<Rat>> = Vec::new();
        // Running EXACT IBP activation bounds, layered from the (intersected)
        // preact bounds below. Each per-unit bound is INTERSECTED with the IBP
        // bound: both are valid enclosures of the true preact range, so their
        // intersection is too — and IBP is strictly better exactly where CROWN's
        // lower envelope loses the `a >= 0` fact (e.g. an identity "passthrough"
        // unit reading a previous activation: IBP proves its preact >= 0, so it
        // stays stable-active/exact instead of picking up a spurious triangle
        // relaxation). Cost: O(k·w²) exact ops — noise next to the CROWN passes.
        let mut act_lo: Vec<Rat> = self.input_lower.clone();
        let mut act_hi: Vec<Rat> = self.input_upper.clone();
        for li in 0..k {
            budget_check()?;
            let width = match self.weights.get(li) {
                Some(w) => w.len(),
                None => 0,
            };
            let w = match self.weights.get(li) {
                Some(v) => v.as_slice(),
                None => &[],
            };
            let b = match self.biases.get(li) {
                Some(v) => v.as_slice(),
                None => &[],
            };
            let mut zl = Vec::new();
            let mut zu = Vec::new();
            for _ in 0..width {
                zl.push(Rat::ZERO);
                zu.push(Rat::ZERO);
            }
            // `j < width`, and `e`/`zl`/`zu` all have `width` entries, so the
            // `.get_mut(j)` writes always hit — total, identical bounds.
            for j in 0..width {
                // `e` keeps `width` entries for every real net, so the
                // `.get_mut(j)` write below always hits.
                let mut e = Vec::new();
                for _ in 0..width {
                    e.push(Rat::ZERO);
                }
                if let Some(s) = e.get_mut(j) {
                    *s = Rat::ONE;
                }
                let mut lo_b = self.crown_bound_z_exact(&lower, &upper, li, &e, Rat::ZERO, true)?;
                let mut hi_b =
                    self.crown_bound_z_exact(&lower, &upper, li, &e, Rat::ZERO, false)?;
                // Exact IBP bound for the same unit; keep the tighter side.
                if let (Some(row), Some(bj)) = (w.get(j), b.get(j)) {
                    let ibp_lo = dot_extreme(row, &act_lo, &act_hi, true)?.add(*bj)?;
                    let ibp_hi = dot_extreme(row, &act_lo, &act_hi, false)?.add(*bj)?;
                    if ibp_lo > lo_b {
                        lo_b = ibp_lo;
                    }
                    if ibp_hi < hi_b {
                        hi_b = ibp_hi;
                    }
                }
                if round {
                    lo_b = dyadic_round_out(lo_b, true);
                    hi_b = dyadic_round_out(hi_b, false);
                }
                if let Some(s) = zl.get_mut(j) {
                    *s = lo_b;
                }
                if let Some(s) = zu.get_mut(j) {
                    *s = hi_b;
                }
            }
            // Next layer's IBP activations from this layer's intersected preacts.
            // Explicit loops (not `.map(..).collect()`): the `|l| ..` closure
            // would be invoked through an absent `<{closure} as Fn>::call` shim,
            // and the bulk `.collect()` carries an unbounded-count allocation
            // obligation; identical elements and order.
            let mut next_lo: Vec<Rat> = Vec::new();
            for l in &zl {
                next_lo.push(if l.is_negative() { Rat::ZERO } else { *l });
            }
            act_lo = next_lo;
            let mut next_hi: Vec<Rat> = Vec::new();
            for u in &zu {
                next_hi.push(if u.is_negative() { Rat::ZERO } else { *u });
            }
            act_hi = next_hi;
            lower.push(zl);
            upper.push(zu);
        }
        crate::rational::ensure_healthy()?;
        Ok(PreactBounds { lower, upper })
    }

    fn alpha_for(&self, layer0: usize, lo: &[Rat], hi: &[Rat]) -> Result<Vec<Rat>, DeepCrownError> {
        if let Some(all) = &self.alpha {
            if all.len() != self.depth() {
                return Err(DeepCrownError::Dimension(
                    "alpha layer count mismatch".into(),
                ));
            }
            // `all.len() == depth()` (checked above) and `layer0 = li < k =
            // depth()`, so `all.get(layer0)` is always `Some` — the `else` arm is
            // an unreachable, fail-safe Dimension error (keeps the read total).
            let Some(a) = all.get(layer0) else {
                return Err(DeepCrownError::Dimension(
                    "alpha layer index out of range".into(),
                ));
            };
            if a.len() != lo.len() {
                return Err(DeepCrownError::Dimension(format!(
                    "alpha[layer {}] length mismatch",
                    layer0.saturating_add(1)
                )));
            }
            for (j, v) in a.iter().enumerate() {
                if v.is_negative() || *v > Rat::ONE {
                    return Err(DeepCrownError::AlphaOutOfRange(layer0.saturating_add(1), j));
                }
            }
            return Ok(a.clone());
        }
        // Explicit Vec::new()+push (not `.collect()`): the length is `lo.len()`,
        // an input-derived count the intraprocedural verifier cannot bound, so a
        // bulk `.collect()` raises an UnboundedAllocation obligation. The loop
        // has no bulk-alloc obligation at all — identical elements and order.
        let mut slopes: Vec<Rat> = Vec::new();
        for (l, u) in lo.iter().zip(hi) {
            slopes.push(if *u >= l.neg() { Rat::ONE } else { Rat::ZERO });
        }
        Ok(slopes)
    }

    /// Build the proof-carrying certificate for the property `y ≥ threshold`.
    ///
    /// # Errors
    /// Returns [`DeepCrownError`] on a dimension mismatch, an out-of-range α, an
    /// infeasible threshold, an expired certificate budget, or an
    /// exact-rational arena failure.
    ///
    /// The `#[ensures]` states the locally-provable producer well-formedness
    /// invariant (on `Ok` the emitted multi-layer entailment certificate has one
    /// non-negative multiplier per premise — `premises.len() == multipliers.len()`);
    /// it is result-only because the builtin `#[ensures]` closure must be
    /// `Copy + 'static` (cannot capture `threshold`/`&self`). The `#[trust::cite]`
    /// grounds the deep-network completeness claim — that the emitted multi-layer
    /// Farkas certificate proves `y ≥ threshold` — in the kernel-checked
    /// `crown_bridge_deepK` theorem.
    #[ensures(|r: &Result<CertifiedDeep, DeepCrownError>| match r { Ok(c) => c.entailment.premises.len() == c.entailment.multipliers.len(), Err(_) => true })]
    #[trust::cite(crownproof::crown_bridge_deepK)]
    #[allow(clippy::too_many_lines)]
    // `?` here would desugar to `from_residual` return paths the verifier's
    // len-witness grounding cannot aggregate over — the explicit match/if-let
    // returns ARE the proof shape (see the extract-then-guard comment).
    #[allow(clippy::question_mark)]
    pub fn certify(&self, threshold: Rat) -> Result<CertifiedDeep, DeepCrownError> {
        // Extract-then-guard: makes the `#[ensures]` locally provable. The
        // match only EXTRACTS (the Err arm returns early), the arity guard is
        // straight-line, and the tail is a plain `Ok(c)` — so every return
        // path constructs its `Ok`/`Err` in the direct predecessor of the
        // return block and the guard's equality edge dominates the `Ok`
        // (the verifier's len-witness grounding window; a guard INSIDE the
        // match arm splits the arm join from the return block and the
        // construction falls outside it). The guard is unreachable by
        // construction — the delegate upholds the same invariant — so this is
        // behavior-identical, fail-closed hardening.
        let c = match self.certify_with_interm(threshold, false) {
            Ok(c) => c,
            // `crate::err_barrier` (identity, `#[inline(never)]`): a fresh in-body
            // `Err` aggregate, not a whole-`Result` forward the return-grounding
            // lane cannot see (nor a const-promoted+merged unit variant).
            Err(e) => return Err(crate::err_barrier(e)),
        };
        if c.entailment.premises.len() != c.entailment.multipliers.len() {
            return Err(crate::err_barrier(DeepCrownError::Dimension(
                "certificate premise/multiplier arity mismatch".into(),
            )));
        }
        Ok(c)
    }

    /// Like [`certify`], but selects the intermediate pre-activation bounds:
    /// `crown_interm = false` uses IBP (original behaviour); `true` uses exact
    /// CROWN intermediate bounds (much tighter on deep nets). Both produce SOUND
    /// premises — the only difference is the `[l,u]` used to build the ReLU
    /// envelopes — so the emitted Farkas/entailment certs are Clean-checkable
    /// identically. The CROWN-tightened bounds let a far smaller bisection tree
    /// close each leaf, matching the f64 tree-discovery screen.
    ///
    /// # Errors
    /// See [`certify`].
    ///
    /// `#[ensures]` (result-only well-formedness; see [`certify`]) + `#[trust::cite]`
    /// grounding in `crown_bridge_deepK`.
    #[ensures(|r: &Result<CertifiedDeep, DeepCrownError>| match r { Ok(c) => c.entailment.premises.len() == c.entailment.multipliers.len(), Err(_) => true })]
    #[trust::cite(crownproof::crown_bridge_deepK)]
    #[allow(clippy::too_many_lines)]
    // `?` here would desugar to `from_residual` return paths the verifier's
    // len-witness grounding cannot aggregate over — the explicit match/if-let
    // returns ARE the proof shape (see the extract-then-guard comment).
    #[allow(clippy::question_mark)]
    pub fn certify_with_interm(
        &self,
        threshold: Rat,
        crown_interm: bool,
    ) -> Result<CertifiedDeep, DeepCrownError> {
        // Extract-then-guard (see `certify` — the verifier's len-witness
        // grounding window): behavior-identical, fail-closed hardening.
        let c = match self.certify_with_interm_cuts(threshold, crown_interm, &[]) {
            Ok(c) => c,
            // `crate::err_barrier` (identity, `#[inline(never)]`): fresh in-body
            // `Err` aggregate (see `certify`).
            Err(e) => return Err(crate::err_barrier(e)),
        };
        if c.entailment.premises.len() != c.entailment.multipliers.len() {
            return Err(crate::err_barrier(DeepCrownError::Dimension(
                "certificate premise/multiplier arity mismatch".into(),
            )));
        }
        Ok(c)
    }

    /// `1u32 << shift`, the number of box corners `2^shift`.
    ///
    /// The `#[trust::requires(shift < 32)]` precondition is the soundness gate on
    /// the shift: for `shift < 32` it is in range, so the result is the exact
    /// `2^shift` with no wrap. Isolating the shift in this tiny function lets the
    /// verifier discharge the shift-overflow obligation cleanly (PROVED), and the
    /// single caller establishes `shift < 32` before calling.
    #[trust::requires(shift < 32)]
    fn corner_count(shift: u32) -> u32 {
        1u32 << shift
    }

    /// EXACT box-corner upper bound `B` on `relu(z1_i x) + relu(z1_j x)` over the
    /// input box, where `z1_* = W⁽¹⁾[*]·x + b⁽¹⁾[*]` is the FIRST-layer pre-activation
    /// (affine in `x`).  `relu∘affine` is convex, so the sum's max over the box is
    /// attained at a corner; we enumerate all `2^n` corners exactly in rationals.
    /// This is exactly the Lean `multiReluCut_box_le` corner derivation, so the
    /// emitted premise `a1_i + a1_j ≤ B` is a VALID joint cut Clean can check.
    fn cut2_box_b_exact(&self, i: usize, j: usize) -> Result<Rat, DeepCrownError> {
        let n = self.input_dim();
        // Fail closed before the corner sweep: a `1u32 << n` mask would WRAP for
        // n >= 32 and produce a wrong (unsound) joint bound. After this guard
        // `n < 32`, so the corner count and every per-dimension bit selector
        // `1 << d` (d < n < 32) are computed through `corner_count`, whose
        // `#[trust::requires(shift < 32)]` proves each shift in-range.
        if n >= 32 {
            return Err(DeepCrownError::CutDimTooLarge(n));
        }
        // `n < 32` (guarded above), so `n as u32 <= 31` and `.min(31)` is the
        // identity — but it makes `n32 <= 31 < 32` PROVABLE, discharging
        // `corner_count`'s `#[trust::requires(shift < 32)]` precondition (the
        // verifier does not thread the `if n >= 32` guard through the `as u32`
        // cast, but it models `min(_, 31) <= 31`). Behavior-preserving.
        let n32 = (n as u32).min(31); // n < 32, so the cast+clamp is exact.
        let corners = Self::corner_count(n32);
        // Layer-0 lookup totalized (index 0; `k >= 1` by `validate`, fallback
        // unreachable). The per-corner box reads `input_*[d]` are loop-bounded by
        // `d < n = input dim`, also total. NOTE: `b[i]`/`b[j]`/`w[i][d]`/`w[j][d]`
        // are indexed by the CALLER-SUPPLIED cut units `i,j`, which this crate
        // never validates against the first-layer width — so they are left as raw
        // `[]` on purpose: a malformed cut must fail LOUD here, not silently read
        // a `Rat::ZERO` and emit an unsound joint bound `B_ij`.
        #[allow(clippy::get_first)]
        let w = match self.weights.get(0) {
            Some(v) => v.as_slice(),
            None => &[],
        };
        #[allow(clippy::get_first)]
        let b = match self.biases.get(0) {
            Some(v) => v.as_slice(),
            None => &[],
        };
        // Fail-CLOSED cut-unit resolution. `i`/`j` are caller-supplied first-layer
        // unit indices this crate never validates against the layer width, so a
        // raw `b[i]`/`w[i][d]` is a potential panic (and the strict verifier
        // fail-closes it). `.get().ok_or(Dimension)?` makes every read TOTAL (no
        // bounds obligation) AND sound: a malformed cut returns Err — it never
        // silently reads `Rat::ZERO`, which would emit an unsound joint bound
        // `B_ij`. For a well-formed cut (`i,j < width`, each row `>= n` wide) the
        // Err arm is unreachable, so behavior is unchanged (ny tests stay green).
        // Free (nested) `fn`s, not an `oor` closure fed to `ok_or_else` (which
        // invokes the closure through an absent `<{closure} as Fn>::call` shim).
        // Same fail-CLOSED semantics, same error value, same evaluation order.
        fn oor() -> DeepCrownError {
            DeepCrownError::Dimension("cut2: unit/column index out of range".into())
        }
        fn rat_at(v: &[Rat], i: usize) -> Result<Rat, DeepCrownError> {
            match v.get(i) {
                Some(x) => Ok(*x),
                None => Err(oor()),
            }
        }
        fn row_at(w: &[Vec<Rat>], i: usize) -> Result<&Vec<Rat>, DeepCrownError> {
            match w.get(i) {
                Some(r) => Ok(r),
                None => Err(oor()),
            }
        }
        let bi = rat_at(b, i)?;
        let bj = rat_at(b, j)?;
        let wi = row_at(w, i)?;
        let wj = row_at(w, j)?;
        let mut best: Option<Rat> = None;
        for mask in 0u32..corners {
            let mut zi = bi;
            let mut zj = bj;
            for d in 0..n {
                // `wrapping_shl` is TOTAL (no Shl-overflow obligation); `d < n <
                // 32` (guarded above) so it equals `1u32 << d` exactly.
                let x = if mask & 1u32.wrapping_shl(d as u32) != 0 {
                    self.input_upper.get(d).copied().unwrap_or(Rat::ZERO)
                } else {
                    self.input_lower.get(d).copied().unwrap_or(Rat::ZERO)
                };
                let wid = rat_at(wi, d)?;
                let wjd = rat_at(wj, d)?;
                zi = zi.add(wid.mul(x)?)?;
                zj = zj.add(wjd.mul(x)?)?;
            }
            let ri = if zi.is_positive() { zi } else { Rat::ZERO };
            let rj = if zj.is_positive() { zj } else { Rat::ZERO };
            let v = ri.add(rj)?;
            best = Some(match best {
                Some(bb) if bb >= v => bb,
                _ => v,
            });
        }
        Ok(best.unwrap_or(Rat::ZERO))
    }

    /// Like [`certify_with_interm`], but additionally emits FIRST-LAYER 2-neuron
    /// joint cuts `a1_i + a1_j ≤ B_ij` as extra `≥0`-multiplier Farkas premises
    /// (the verified `multiReluCut` lever).  `cuts` is a list of `(i, j)` first-layer
    /// unit pairs.  For each cut, `B_ij` is DERIVED exactly from the leaf box corners
    /// (`cut2_box_b_exact`), so the premise is a valid joint upper bound on
    /// `relu(z1_i)+relu(z1_j)` — Clean checks it with the others.  In the backward
    /// pass, for a pair both reduced via the UPPER envelope (running coeff `< 0`),
    /// a `μ = min(|c_i|,|c_j|)` share is DIVERTED from the per-neuron envelopes to
    /// the joint cut (no `z`-term, joint const `μ·B`), TIGHTENING the bound when
    /// `B < t_i + t_j`.  Every other multiplier is unchanged, so the result is a
    /// valid Farkas combination with the cut as one extra premise.
    ///
    /// The `#[ensures]` states the locally-provable producer well-formedness
    /// invariant (on `Ok` the emitted entailment certificate has one non-negative
    /// multiplier per premise, including the joint-cut premise); result-only
    /// because the builtin closure must be `Copy + 'static`. The `#[trust::cite]`
    /// grounds the deep-network-plus-cuts completeness claim in the kernel-checked
    /// `crown_bridge_deepK` theorem (the joint ReLU cut is the `multiReluCut`
    /// lever, itself a sound extra `≥0`-multiplier premise).
    #[ensures(|r: &Result<CertifiedDeep, DeepCrownError>| match r { Ok(c) => c.entailment.premises.len() == c.entailment.multipliers.len(), Err(_) => true })]
    #[trust::cite(crownproof::crown_bridge_deepK)]
    // `?` here would desugar to `from_residual` return paths the verifier's
    // len-witness grounding cannot aggregate over — the explicit match/if-let
    // returns ARE the proof shape (see the extract-then-guard comment).
    #[allow(clippy::question_mark)]
    pub fn certify_with_interm_cuts(
        &self,
        threshold: Rat,
        crown_interm: bool,
        cuts: &[(usize, usize)],
    ) -> Result<CertifiedDeep, DeepCrownError> {
        // Extract-then-guard (see `certify` — the verifier's len-witness
        // grounding window): behavior-identical, fail-closed hardening.
        let c = match self.certify_impl(threshold, crown_interm, cuts, None, &[]) {
            Ok(c) => c,
            // `crate::err_barrier` (identity, `#[inline(never)]`): fresh in-body
            // `Err` aggregate (see `certify`).
            Err(e) => return Err(crate::err_barrier(e)),
        };
        if c.entailment.premises.len() != c.entailment.multipliers.len() {
            return Err(crate::err_barrier(DeepCrownError::Dimension(
                "certificate premise/multiplier arity mismatch".into(),
            )));
        }
        Ok(c)
    }

    /// Certify `y ≥ threshold` for the LINEAR-DIFFERENCE read-out
    /// `y = W⁽ᵏ⁺¹⁾·a⁽ᵏ⁾ + b⁽ᵏ⁺¹⁾ − (g_coeffs·x + g_offset)` — the network's
    /// scalar output minus an exact rational affine function of the *inputs*.
    ///
    /// This is the ny-cert half of the ground-truth dominance certificate
    /// (`docs/GEOMETRIC_GROUND_TRUTH_PLAN.md` §4): for a difference network
    /// `h = f − g` whose subtracted side `g` is a *pure Linear* ground-truth
    /// graph (the PLANE builder), `h` is exactly this read-out over `f`'s
    /// FC-ReLU stack. The g-side enters the certificate as EXACT RATIONAL ROWS:
    /// the read-out premise pair gains the terms `+g_coeffs[i]·xᵢ` and the
    /// constant `b⁽ᵏ⁺¹⁾ − g_offset`, and the backward pass eliminates those
    /// input terms through the very same box premises as the network's own
    /// input coefficients. Every premise is still a linear constraint over
    /// named variables and every multiplier is still non-negative, so the
    /// emitted certificate is checked by the UNCHANGED
    /// [`crate::selfcheck::check_entailment`] / [`crate::selfcheck::check_farkas`]
    /// and remains grounded in the same kernel-checked
    /// `farkas_premise_combination` theorem — no new checker surface, no new
    /// lemma.
    ///
    /// ## Scope
    ///
    /// This entry point covers the pure-*linear* ground-truth side. A
    /// QUADRATIC side (the M1 quadric builders use `PowConstant(2)`) needs the
    /// pow2 envelope premises and their kernel-checked grounding — that is
    /// [`Self::certify_difference_quadratic`].
    ///
    /// # Errors
    /// [`DeepCrownError::Dimension`] when `g_coeffs.len()` differs from the
    /// input dimension; otherwise as [`Self::certify`].
    #[ensures(|r: &Result<CertifiedDeep, DeepCrownError>| match r { Ok(c) => c.entailment.premises.len() == c.entailment.multipliers.len(), Err(_) => true })]
    #[trust::cite(crownproof::farkas_premise_combination)]
    // `?` here would desugar to `from_residual` return paths the verifier's
    // len-witness grounding cannot aggregate over — the explicit match/if-let
    // returns ARE the proof shape (see the extract-then-guard comment).
    #[allow(clippy::question_mark)]
    pub fn certify_difference_linear(
        &self,
        g_coeffs: &[Rat],
        g_offset: Rat,
        threshold: Rat,
    ) -> Result<CertifiedDeep, DeepCrownError> {
        // `if let Err` instead of `?`: keeps every return path an IN-BODY
        // aggregate (no `from_residual` desugar), which the `#[ensures]`
        // proof discipline requires. Behavior-identical (same error type).
        // `crate::err_barrier` (identity, `#[inline(never)]`) forces a fresh
        // in-body `Err` aggregate rather than a whole-`Result` forward.
        if let Err(e) = self.validate() {
            return Err(crate::err_barrier(e));
        }
        if g_coeffs.len() != self.input_dim() {
            // Manual ASCII construction (not `format!`): the fmt machinery's
            // dynamic dispatch is a trusted-assumption row under the strict
            // verifier; `push_usize_ascii` is byte-identical to the `format!`
            // output for `usize` arguments.
            let mut msg = String::new();
            msg.push_str("ground-truth coefficient count ");
            push_usize_ascii(&mut msg, g_coeffs.len());
            msg.push_str(" != input dim ");
            push_usize_ascii(&mut msg, self.input_dim());
            return Err(crate::err_barrier(DeepCrownError::Dimension(msg)));
        }
        // Extract-then-guard (see `certify` — the verifier's len-witness
        // grounding window): behavior-identical, fail-closed hardening.
        let c = match self.certify_impl(threshold, false, &[], Some((g_coeffs, g_offset)), &[]) {
            Ok(c) => c,
            Err(e) => return Err(crate::err_barrier(e)),
        };
        if c.entailment.premises.len() != c.entailment.multipliers.len() {
            return Err(crate::err_barrier(DeepCrownError::Dimension(
                "certificate premise/multiplier arity mismatch".into(),
            )));
        }
        Ok(c)
    }

    /// Certify `y ≥ threshold` for the QUADRATIC-DIFFERENCE read-out
    /// `y = W⁽ᵏ⁺¹⁾·a⁽ᵏ⁾ + b⁽ᵏ⁺¹⁾ − (g_coeffs·x + g_offset + Σⱼ qⱼ·tⱼ(x)²)`
    /// — the network's scalar output minus an exact rational *quadratic*
    /// function of the inputs, given as an affine part plus a weighted sum of
    /// squares of affine pre-squares `tⱼ(x) = aⱼ·x + bⱼ` (the [`QuadTerm`]s).
    ///
    /// This is the quadratic half of the ground-truth dominance certificate
    /// (`docs/GEOMETRIC_GROUND_TRUTH_PLAN.md` §4): the M1 quadric residual
    /// builders (sphere / cylinder / cone, via `PowConstant(2)`) all fold to
    /// exactly this shape. Each square introduces two fresh certificate
    /// variables — `tⱼ` (the pre-square) and `sⱼ` (the square) — and three
    /// premise classes, every one a plain linear constraint over named
    /// variables for the UNCHANGED checker:
    ///
    /// * the definitional affine pair `tⱼ − aⱼ·x ⋛ bⱼ` (same class as the
    ///   network's own affine-layer rows: definitional, split ≤/≥);
    /// * the **secant upper envelope** `sⱼ ≤ (lⱼ+uⱼ)·tⱼ − lⱼ·uⱼ`, where
    ///   `[lⱼ, uⱼ]` is the exact interval of `tⱼ` over the input box — valid
    ///   because `sⱼ = tⱼ²` and `tⱼ ∈ [lⱼ, uⱼ]`, grounded in the
    ///   kernel-checked corpus theorem `pow2_secant`
    ///   (`Crownproof.Pow2Envelope` in the exact pinned Clean dependency);
    /// * the **tangent lower envelope** `sⱼ ≥ 2c·tⱼ − c²` at the tangency
    ///   point `c = median(lⱼ, 0, uⱼ)` — valid for EVERY `tⱼ` (a supporting
    ///   line of the parabola), grounded in the kernel-checked corpus theorem
    ///   `pow2_tangent`.
    ///
    /// The read-out premise pair gains `+qⱼ·sⱼ` terms, and the backward pass
    /// eliminates each `sⱼ` through its sign-appropriate envelope (secant when
    /// the running coefficient is negative, i.e. `qⱼ > 0`; tangent when
    /// positive) and each `tⱼ` through its definitional pair into the box —
    /// each elimination is one more non-negative Farkas multiplier. The
    /// emitted certificate is checked by the UNCHANGED
    /// [`crate::selfcheck::check_entailment`] / [`crate::selfcheck::check_farkas`]
    /// and rests on `farkas_premise_combination` for the combination plus the
    /// two pow2 envelope theorems for the new premise class — no new checker
    /// surface.
    ///
    /// # Errors
    /// [`DeepCrownError::Dimension`] when `g_coeffs.len()` or any
    /// `QuadTerm::lin` length differs from the input dimension; otherwise as
    /// [`Self::certify`].
    #[ensures(|r: &Result<CertifiedDeep, DeepCrownError>| match r { Ok(c) => c.entailment.premises.len() == c.entailment.multipliers.len(), Err(_) => true })]
    #[trust::cite(crownproof::farkas_premise_combination)]
    #[trust::cite(crownproof::pow2_tangent)]
    #[trust::cite(crownproof::pow2_secant)]
    // `?` here would desugar to `from_residual` return paths the verifier's
    // len-witness grounding cannot aggregate over — the explicit match/if-let
    // returns ARE the proof shape (see the extract-then-guard comment).
    #[allow(clippy::question_mark)]
    pub fn certify_difference_quadratic(
        &self,
        g_coeffs: &[Rat],
        g_offset: Rat,
        quad: &[QuadTerm],
        threshold: Rat,
    ) -> Result<CertifiedDeep, DeepCrownError> {
        // `if let Err` instead of `?`: keeps every return path an IN-BODY
        // aggregate (no `from_residual` desugar), which the `#[ensures]`
        // proof discipline requires. Behavior-identical (same error type).
        // `crate::err_barrier` (identity, `#[inline(never)]`) forces a fresh
        // in-body `Err` aggregate rather than a whole-`Result` forward.
        if let Err(e) = self.validate() {
            return Err(crate::err_barrier(e));
        }
        if g_coeffs.len() != self.input_dim() {
            // Manual ASCII construction (not `format!`): the fmt machinery's
            // dynamic dispatch is a trusted-assumption row under the strict
            // verifier; `push_usize_ascii` is byte-identical to the `format!`
            // output for `usize` arguments.
            let mut msg = String::new();
            msg.push_str("ground-truth coefficient count ");
            push_usize_ascii(&mut msg, g_coeffs.len());
            msg.push_str(" != input dim ");
            push_usize_ascii(&mut msg, self.input_dim());
            return Err(crate::err_barrier(DeepCrownError::Dimension(msg)));
        }
        for (j, term) in quad.iter().enumerate() {
            if term.lin.len() != self.input_dim() {
                // Manual ASCII construction (not `format!`): same rationale as
                // the `g_coeffs` guard above — byte-identical message.
                let mut msg = String::new();
                msg.push_str("quad term ");
                push_usize_ascii(&mut msg, j);
                msg.push_str(" pre-square coefficient count ");
                push_usize_ascii(&mut msg, term.lin.len());
                msg.push_str(" != input dim ");
                push_usize_ascii(&mut msg, self.input_dim());
                return Err(crate::err_barrier(DeepCrownError::Dimension(msg)));
            }
        }
        // Extract-then-guard (see `certify` — the verifier's len-witness
        // grounding window): behavior-identical, fail-closed hardening.
        let c = match self.certify_impl(threshold, false, &[], Some((g_coeffs, g_offset)), quad) {
            Ok(c) => c,
            Err(e) => return Err(crate::err_barrier(e)),
        };
        if c.entailment.premises.len() != c.entailment.multipliers.len() {
            return Err(crate::err_barrier(DeepCrownError::Dimension(
                "certificate premise/multiplier arity mismatch".into(),
            )));
        }
        Ok(c)
    }

    /// Shared certificate construction. `subtract`, when present, is the
    /// `(coeffs, offset)` of an exact rational affine input functional
    /// subtracted from the scalar read-out (see [`Self::certify_difference_linear`]);
    /// `quad` lists the additionally subtracted squared terms `qⱼ·tⱼ(x)²`
    /// (see [`Self::certify_difference_quadratic`]).
    #[allow(clippy::too_many_lines)]
    fn certify_impl(
        &self,
        threshold: Rat,
        crown_interm: bool,
        cuts: &[(usize, usize)],
        subtract: Option<(&[Rat], Rat)>,
        quad: &[QuadTerm],
    ) -> Result<CertifiedDeep, DeepCrownError> {
        // Fail-CLOSED entry gate: a poisoned arena means an earlier `Rat`
        // operation on this thread silently substituted `0/1` for an arbitrary
        // value — nothing built from those handles can be trusted.
        if crate::rational::poisoned() {
            return Err(crate::err_barrier(DeepCrownError::ArenaPoisoned));
        }
        self.validate()?;
        budget_check()?;
        let k = self.depth();
        let n = self.input_dim();
        // crown_interm => EXACT-rational CROWN intermediate bounds. These premises
        // gate the verdict, so they must be certified via `preact_bounds_crown`
        // (exact rational, via `crown_bound_z_exact`), which is both sound and
        // equal-or-tighter — matching this function's documented `true` => exact
        // contract. (A former f64 "snapped" fast path made its bound "sound" only
        // with a heuristic `|x|·1e-9 + 1e-9` inflation that was NOT a certified
        // error bound — unsoundly too tight if the real error exceeded it — so it
        // was removed.)
        let preact = if crown_interm {
            self.preact_bounds_crown()?
        } else {
            self.preact_bounds()?
        };

        // Variable name helpers. Free (nested) `fn`s, not closures: called
        // directly rather than through an absent `<{closure} as Fn>::call` shim,
        // so the builders stay in verified code. They capture nothing; `format!`
        // over `usize`s proves panic-free.
        fn xv(i: usize) -> String {
            format!("x{i}")
        }
        // z and a variables carry a 1-indexed layer tag and unit index.
        fn zv(layer1: usize, j: usize) -> String {
            format!("z{layer1}_{j}")
        }
        fn av(layer1: usize, j: usize) -> String {
            format!("a{layer1}_{j}")
        }

        // --- Assemble the relaxed-network constraint system (premises) ---
        let mut premises: Vec<LinearConstraint> = Vec::new();
        let mut mult: Vec<Rat> = Vec::new();
        // Free (nested) `fn`, not a closure: called directly (no absent
        // `<{closure} as Fn>::call` shim); it captures nothing — the premise and
        // multiplier vectors are explicit `&mut` parameters.
        fn push(
            c: LinearConstraint,
            premises: &mut Vec<LinearConstraint>,
            mult: &mut Vec<Rat>,
        ) -> usize {
            // The new premise's index is the length BEFORE the push, which
            // avoids the `len() - 1` subtraction (provably free of usize
            // underflow regardless of the vector's prior length).
            let idx = premises.len();
            premises.push(c);
            mult.push(Rat::ZERO);
            idx
        }
        // Total read-modify-add on an accumulator `Vec<Rat>` (`mult` by premise
        // index, or a coefficient vector by loop index): every `idx` used with
        // this helper is either a premise index returned by `push`
        // (`< premises.len() == mult.len()`) or a loop counter `< the Vec's
        // constructed length`, so both `.get`/`.get_mut` fallbacks are
        // UNREACHABLE. Replaces `v[idx] = v[idx].add(delta)?` with no `[]` panic /
        // slice-bounds obligation, identical result for every valid input.
        // Free (nested) `fn`, not a closure: called directly rather than through
        // an absent `<{closure} as Fn>::call` shim, so the accumulator update
        // stays in verified code. Captures nothing (only its params + `Rat::ZERO`).
        fn add_at(v: &mut [Rat], idx: usize, delta: Rat) -> Result<(), RatError> {
            let cur = v.get(idx).copied().unwrap_or(Rat::ZERO);
            let next = cur.add(delta)?;
            if let Some(slot) = v.get_mut(idx) {
                *slot = next;
            }
            Ok(())
        }
        // Total nested bias read (fail-safe `Rat::ZERO`): `validate` pinned
        // `biases.len() == k` and each `biases[li].len()` to the layer width, so
        // for `li < k`, `j < width` the `.get`s always match — fallback unreachable.
        // Free (nested) `fn`, not a closure: called directly rather than through
        // an absent `<{closure} as Fn>::call` shim (the captured `self.biases`
        // becomes the explicit `biases` parameter), and the nested lookup is a
        // plain `match` rather than a closure passed to `Option::and_then`.
        fn bias_at(biases: &[Vec<Rat>], li: usize, j: usize) -> Rat {
            match biases.get(li) {
                Some(b) => b.get(j).copied().unwrap_or(Rat::ZERO),
                None => Rat::ZERO,
            }
        }

        // Box: xᵢ ≤ uᵢ and xᵢ ≥ lᵢ.
        // `Vec::new()` (not `with_capacity(n)`): the capacity hint on the
        // unbounded input dimension carries a hardened allocation obligation the
        // model cannot bound; amortized growth is noise next to the Rat math.
        let mut box_u = Vec::new();
        let mut box_l = Vec::new();
        // `i < n`; `input_upper`/`input_lower` are length `n` (`validate`), so the
        // `Rat::ZERO` fallbacks are unreachable — total reads, no value change.
        for i in 0..n {
            box_u.push(push(
                LinearConstraint::with_kind(
                    ConstraintKind::Le,
                    &[(&xv(i), Rat::ONE)],
                    self.input_upper.get(i).copied().unwrap_or(Rat::ZERO),
                ),
                &mut premises,
                &mut mult,
            ));
            box_l.push(push(
                LinearConstraint::with_kind(
                    ConstraintKind::Ge,
                    &[(&xv(i), Rat::ONE)],
                    self.input_lower.get(i).copied().unwrap_or(Rat::ZERO),
                ),
                &mut premises,
                &mut mult,
            ));
        }

        // Affine layers as ≤/≥ pairs. For layer L (1-indexed), the input
        // variables are x (L==1) or a⁽ᴸ⁻¹⁾ (L>1).
        // z_le[L-1][j], z_ge[L-1][j] = premise indices.
        // `Vec::new()` (not `with_capacity(k)`): unbounded-depth capacity hint →
        // hardened allocation obligation; amortized growth is noise.
        let mut z_le: Vec<Vec<usize>> = Vec::new();
        let mut z_ge: Vec<Vec<usize>> = Vec::new();
        for li in 0..k {
            budget_check()?;
            // total: `saturating_add` (not `+ 1`): `li < k = weights.len() <=
            // isize::MAX`, so the add never saturates — identical 1-indexed
            // layer tag, no overflow VC on the havoced counter.
            let layer1 = li.saturating_add(1);
            // `li < k`; `weights`/`biases` have `k` entries (`validate`), so the
            // empty-slice fallbacks are unreachable (a no-op empty iteration).
            let w = match self.weights.get(li) {
                Some(v) => v.as_slice(),
                None => &[],
            };
            let b = match self.biases.get(li) {
                Some(v) => v.as_slice(),
                None => &[],
            };
            // Free (nested) `fn`, not a closure: called directly (no absent
            // `<{closure} as Fn>::call` shim); the captured `layer1` becomes an
            // explicit parameter.
            // `layer1 == li + 1 >= 1`; use `checked_sub` so the verifier sees
            // no usize underflow (the `0` fallback is unreachable here since
            // `layer1 >= 1`, and the `== 1` arm handles the first layer).
            fn prev_in(layer1: usize, i: usize) -> String {
                match layer1.checked_sub(1) {
                    Some(0) | None => xv(i),
                    Some(prev) => av(prev, i),
                }
            }
            // `Vec::new()` (not `with_capacity(w.len())`): unbounded-width
            // capacity hint → hardened allocation obligation.
            let mut le_row = Vec::new();
            let mut ge_row = Vec::new();
            for (j, row) in w.iter().enumerate() {
                // Per-row poll: on wide layers (up to the 8192-unit eligibility
                // cap) a per-layer poll alone leaves minutes between checks.
                budget_check()?;
                // z⁽ᴸ⁾ⱼ − Σ W⁽ᴸ⁾[j][i]·in_i = b⁽ᴸ⁾ⱼ.
                // `Vec::new()` + push (not a `vec![…]` literal): the macro's
                // boxed-slice `into_vec` inlines hardened alloc/arith
                // obligations; identical single seed element.
                let mut terms: Vec<(String, Rat)> = Vec::new();
                terms.push((zv(layer1, j), Rat::ONE));
                for (i, wji) in row.iter().enumerate() {
                    terms.push((prev_in(layer1, i), wji.neg()));
                }
                // Explicit loop (not `.map(..).collect()`): the `|(s, v)| ..`
                // closure would be invoked through an absent `Fn::call` shim;
                // identical elements and order, no bulk-alloc obligation.
                let mut refs: Vec<(&str, Rat)> = Vec::new();
                for (s, v) in &terms {
                    refs.push((s.as_str(), *v));
                }
                // `j < w.len()` and `b.len() == w.len()` (`validate`), so `b.get(j)`
                // always matches — fallback unreachable.
                let bj = b.get(j).copied().unwrap_or(Rat::ZERO);
                le_row.push(push(
                    LinearConstraint::with_kind(ConstraintKind::Le, &refs, bj),
                    &mut premises,
                    &mut mult,
                ));
                ge_row.push(push(
                    LinearConstraint::with_kind(ConstraintKind::Ge, &refs, bj),
                    &mut premises,
                    &mut mult,
                ));
            }
            z_le.push(le_row);
            z_ge.push(ge_row);
        }

        // ReLU envelopes per layer/unit. Lower: a⁽ᴸ⁾ⱼ ≥ pⱼ·z⁽ᴸ⁾ⱼ + qⱼ (Ge).
        // Upper: a⁽ᴸ⁾ⱼ ≤ rⱼ·z⁽ᴸ⁾ⱼ + tⱼ (Le).
        // `Vec::new()` (not `with_capacity(k)`): unbounded-depth capacity hint →
        // hardened allocation obligation; amortized growth is noise.
        let mut env_lower: Vec<Vec<(Rat, Rat, usize)>> = Vec::new();
        let mut env_upper: Vec<Vec<(Rat, Rat, usize)>> = Vec::new();
        for li in 0..k {
            budget_check()?;
            // total: `saturating_add` (not `+ 1`): `li < k <= isize::MAX`, so
            // the add never saturates — identical 1-indexed layer tag.
            let layer1 = li.saturating_add(1);
            // `li < k`; `preact.lower`/`upper` have `k` rows (one per layer), so
            // the empty-slice fallbacks are unreachable.
            // Plain `match` (not `.map(Vec::as_slice)`): the fn-item value would
            // be invoked through the `Fn` shim inside `Option::map`; identical
            // slice selection (same idiom as the affine-layer loop above).
            let lo = match preact.lower.get(li) {
                Some(v) => v.as_slice(),
                None => &[],
            };
            let hi = match preact.upper.get(li) {
                Some(v) => v.as_slice(),
                None => &[],
            };
            let alpha = self.alpha_for(li, lo, hi)?;
            // `Vec::new()` (not `with_capacity(lo.len())`): unbounded-width
            // capacity hint → hardened allocation obligation.
            let mut low_row = Vec::new();
            let mut up_row = Vec::new();
            // `j < lo.len()`, and `hi`/`alpha` are the same length as `lo` (built
            // together / validated in `alpha_for`), so the `Rat::ZERO` fallbacks
            // are unreachable — identical per-unit envelope selection.
            for j in 0..lo.len() {
                // Per-row poll (see the premise-assembly loop above).
                budget_check()?;
                let l = lo.get(j).copied().unwrap_or(Rat::ZERO);
                let u = hi.get(j).copied().unwrap_or(Rat::ZERO);
                let (p, q, r, t) = if !l.is_negative() {
                    // Always active: a = z.
                    (Rat::ONE, Rat::ZERO, Rat::ONE, Rat::ZERO)
                } else if !u.is_positive() {
                    // Always inactive: a = 0.
                    (Rat::ZERO, Rat::ZERO, Rat::ZERO, Rat::ZERO)
                } else {
                    // Unstable. Lower a ≥ α·z. Upper a ≤ s·(z − l), s = u/(u−l).
                    let s = u.mul(u.sub(l)?.inv()?)?;
                    (
                        alpha.get(j).copied().unwrap_or(Rat::ZERO),
                        Rat::ZERO,
                        s,
                        s.mul(l.neg())?,
                    )
                };
                let le_idx = push(
                    LinearConstraint::with_kind(
                        ConstraintKind::Ge,
                        &[(&av(layer1, j), Rat::ONE), (&zv(layer1, j), p.neg())],
                        q,
                    ),
                    &mut premises,
                    &mut mult,
                );
                let ue_idx = push(
                    LinearConstraint::with_kind(
                        ConstraintKind::Le,
                        &[(&av(layer1, j), Rat::ONE), (&zv(layer1, j), r.neg())],
                        t,
                    ),
                    &mut premises,
                    &mut mult,
                );
                low_row.push((p, q, le_idx));
                up_row.push((r, t, ue_idx));
            }
            env_lower.push(low_row);
            env_upper.push(up_row);
        }

        // QUADRATIC ground-truth squares (certify_difference_quadratic): for
        // each subtracted term qⱼ·tⱼ(x)², introduce the fresh variables tⱼ
        // (pre-square) and sⱼ (square) and emit
        //   * the definitional affine pair  tⱼ − aⱼ·x ⋛ bⱼ           (Le/Ge),
        //   * the secant upper envelope     sⱼ − (l+u)·tⱼ ≤ −l·u     (pow2_secant,
        //     valid because tⱼ ∈ [l, u] over the box),
        //   * the tangent lower envelope    sⱼ − 2c·tⱼ ≥ −c²         (pow2_tangent,
        //     valid for every tⱼ; c = median(l, 0, u)).
        // All three are plain linear constraints over named variables — the
        // checker is unchanged; the envelope premises' validity is grounded in
        // the kernel-checked pow2 theorems (`Crownproof.Pow2Envelope` in Clean).
        // Free (nested) `fn`s, not closures: called directly (no absent
        // `<{closure} as Fn>::call` shim) — same rationale as `xv`/`zv`/`av`.
        fn tv(j: usize) -> String {
            format!("t{j}")
        }
        fn sv(j: usize) -> String {
            format!("s{j}")
        }
        // `Vec::new()` (not `with_capacity(quad.len())`): input-derived capacity
        // hint → hardened allocation obligation; identical contents.
        let mut quad_prem: Vec<QuadPremises> = Vec::new();
        for (j, term) in quad.iter().enumerate() {
            let lo = dot_extreme(&term.lin, &self.input_lower, &self.input_upper, true)?
                .add(term.offset)?;
            let hi = dot_extreme(&term.lin, &self.input_lower, &self.input_upper, false)?
                .add(term.offset)?;
            // Definitional pair: tⱼ − Σ aⱼᵢ·xᵢ ⋛ bⱼ.
            // `Vec::new()` + push (not a `vec![…]` literal): macro-internal
            // alloc/arith obligations; identical single seed element.
            let mut t_terms: Vec<(String, Rat)> = Vec::new();
            t_terms.push((tv(j), Rat::ONE));
            for (i, a_i) in term.lin.iter().enumerate() {
                t_terms.push((xv(i), a_i.neg()));
            }
            // Explicit loop (not `.map(..).collect()`): same absent `Fn::call`
            // shim rationale as `refs` above — identical elements and order.
            let mut t_refs: Vec<(&str, Rat)> = Vec::new();
            for (s, v) in &t_terms {
                t_refs.push((s.as_str(), *v));
            }
            let t_le = push(
                LinearConstraint::with_kind(ConstraintKind::Le, &t_refs, term.offset),
                &mut premises,
                &mut mult,
            );
            let t_ge = push(
                LinearConstraint::with_kind(ConstraintKind::Ge, &t_refs, term.offset),
                &mut premises,
                &mut mult,
            );
            // Secant: sⱼ − (l+u)·tⱼ ≤ −l·u  [pow2_secant on tⱼ ∈ [l, u]].
            let secant = push(
                LinearConstraint::with_kind(
                    ConstraintKind::Le,
                    &[(&sv(j), Rat::ONE), (&tv(j), lo.add(hi)?.neg())],
                    lo.mul(hi)?.neg(),
                ),
                &mut premises,
                &mut mult,
            );
            // Tangent at c = median(l, 0, u): sⱼ − 2c·tⱼ ≥ −c²  [pow2_tangent].
            let tangency = if lo.is_positive() {
                lo
            } else if hi.is_negative() {
                hi
            } else {
                Rat::ZERO
            };
            let tangent = push(
                LinearConstraint::with_kind(
                    ConstraintKind::Ge,
                    &[(&sv(j), Rat::ONE), (&tv(j), tangency.add(tangency)?.neg())],
                    tangency.mul(tangency)?.neg(),
                ),
                &mut premises,
                &mut mult,
            );
            quad_prem.push(QuadPremises {
                coeff: term.coeff,
                lin: term.lin.clone(),
                offset: term.offset,
                t_le,
                t_ge,
                secant,
                tangent,
                lo,
                hi,
                tangency,
            });
        }

        // Output read-out y = Σ W⁽ᵏ⁺¹⁾ⱼ·a⁽ᵏ⁾ⱼ + b⁽ᵏ⁺¹⁾ [− (g·x + g₀ + Σ qⱼ·sⱼ)],
        // split into ≤ / ≥. With a subtracted affine input functional (the linear
        // ground-truth side), the premise pair is
        //   y − Σ W⁽ᵏ⁺¹⁾ⱼ·a⁽ᵏ⁾ⱼ + Σ gᵢ·xᵢ [+ Σ qⱼ·sⱼ]  ⋛  b⁽ᵏ⁺¹⁾ − g₀,
        // i.e. the g-side contributes exact rational rows on the INPUT
        // variables (and the square variables sⱼ for the quadratic side) —
        // still plain linear constraints for the unchanged checker.
        let last_dim = self.layer_dim(k);
        // `Vec::new()` + push (not a `vec![…]` literal): macro-internal
        // alloc/arith obligations; identical single seed element.
        let mut y_terms: Vec<(String, Rat)> = Vec::new();
        y_terms.push(("y".to_string(), Rat::ONE));
        // `j < last_dim = dim(k)` and `out_weight.len() == dim(k)` (`validate`),
        // so the `Rat::ZERO` fallback is unreachable.
        for j in 0..last_dim {
            y_terms.push((
                av(k, j),
                self.out_weight.get(j).copied().unwrap_or(Rat::ZERO).neg(),
            ));
        }
        let mut y_rhs = self.out_bias;
        if let Some((g_coeffs, g_offset)) = subtract {
            for (i, g_i) in g_coeffs.iter().enumerate() {
                y_terms.push((xv(i), *g_i));
            }
            y_rhs = y_rhs.sub(g_offset)?;
        }
        for (j, qp) in quad_prem.iter().enumerate() {
            y_terms.push((sv(j), qp.coeff));
        }
        // Explicit loop (not `.map(..).collect()`): same absent `Fn::call` shim
        // rationale as `refs` above — identical elements and order.
        let mut y_refs: Vec<(&str, Rat)> = Vec::new();
        for (s, v) in &y_terms {
            y_refs.push((s.as_str(), *v));
        }
        let _y_le = push(
            LinearConstraint::with_kind(ConstraintKind::Le, &y_refs, y_rhs),
            &mut premises,
            &mut mult,
        );
        let y_ge = push(
            LinearConstraint::with_kind(ConstraintKind::Ge, &y_refs, y_rhs),
            &mut premises,
            &mut mult,
        );

        // --- FIRST-LAYER 2-NEURON JOINT CUTS (the verified multiReluCut lever) ---
        // For each (i,j), emit  a1_i + a1_j ≤ B_ij  with B_ij the exact box-corner
        // max of relu(z1_i)+relu(z1_j).  Premise index recorded so the backward pass
        // can give it a ≥0 multiplier μ.  Only first-layer cuts (z1 affine in x) so
        // B_ij is exactly the Lean corner derivation.
        let mut cut_prem: Vec<(usize, usize, usize, Rat)> = Vec::new(); // (i,j,prem_idx,B)
        for &(ci, cj) in cuts {
            let b_ij = self.cut2_box_b_exact(ci, cj)?;
            let idx = push(
                LinearConstraint::with_kind(
                    ConstraintKind::Le,
                    &[(&av(1, ci), Rat::ONE), (&av(1, cj), Rat::ONE)],
                    b_ij,
                ),
                &mut premises,
                &mut mult,
            );
            cut_prem.push((ci, cj, idx, b_ij));
        }

        // --- CROWN backward pass = choosing the non-negative multipliers ---
        // Start from the normalized form of y_ge:
        //   −y + Σ W⁽ᵏ⁺¹⁾ⱼ·a⁽ᵏ⁾ⱼ [− Σ gᵢ·xᵢ] ≤ −(b [− g₀]).
        // The running x coefficients therefore start at −gᵢ (folded in after the
        // layer sweep, where the network's own input coefficients join them).
        add_at(&mut mult, y_ge, Rat::ONE)?;
        let mut const_acc = y_rhs.neg();

        // Eliminate each square variable sⱼ (running coefficient −qⱼ from the
        // normalized read-out) through its sign-appropriate pow2 envelope, then
        // each pre-square tⱼ through its definitional pair. The resulting input
        // coefficients are collected in `quad_x` and joined with the network's
        // own input coefficients after the layer sweep.
        // `qp.secant`/`tangent`/`t_ge`/`t_le` are premise indices returned by
        // `push`, and `quad_x` has `n` entries with `qp.lin.len() == n` (validated
        // in `certify_difference_quadratic`), so every `add_at` index is in range.
        // `Vec::new()` + push (not `vec![_; n]`): the bulk fill carries a
        // hardened allocation obligation unbounded on the input dimension; the
        // push loop yields the identical `n`-zero coefficient vector.
        let mut quad_x = Vec::new();
        for _ in 0..n {
            quad_x.push(Rat::ZERO);
        }
        for qp in &quad_prem {
            let cs = qp.coeff.neg();
            let ct = if cs.is_negative() {
                // qⱼ > 0: need an UPPER bound on sⱼ — the secant, scaled by qⱼ.
                let mag = cs.neg();
                add_at(&mut mult, qp.secant, mag)?;
                const_acc = const_acc.add(mag.mul(qp.lo.mul(qp.hi)?.neg())?)?; // −mag·l·u
                mag.mul(qp.lo.add(qp.hi)?)?.neg() // t coefficient −mag·(l+u)
            } else if cs.is_positive() {
                // qⱼ < 0: need a LOWER bound on sⱼ — the tangent at c, scaled by |qⱼ|.
                add_at(&mut mult, qp.tangent, cs)?;
                const_acc = const_acc.add(cs.mul(qp.tangency.mul(qp.tangency)?)?)?; // +cs·c²
                cs.mul(qp.tangency.add(qp.tangency)?)? // t coefficient +2c·cs
            } else {
                Rat::ZERO
            };
            if ct.is_zero() {
                continue;
            }
            // Eliminate tⱼ through the definitional pair (same pattern as the
            // affine z-rows): input coefficients gain ct·aⱼ, constant −ct·bⱼ.
            if ct.is_positive() {
                add_at(&mut mult, qp.t_ge, ct)?;
            } else {
                add_at(&mut mult, qp.t_le, ct.neg())?;
            }
            for (i, a_i) in qp.lin.iter().enumerate() {
                add_at(&mut quad_x, i, ct.mul(*a_i)?)?;
            }
            const_acc = const_acc.add(ct.mul(qp.offset)?.neg())?;
        }

        // `a_coeff` holds the running coefficient on the *current* layer's
        // activations a⁽ᴸ⁾. Initialized to the read-out weights on a⁽ᵏ⁾.
        let mut a_coeff: Vec<Rat> = self.out_weight.clone();

        // Forward index (not `(0..k).rev()`): the `Rev<Range>` adapter is an
        // absent-callee for the panic-freedom checker; `li = k-1-idx` reverses the
        // walk exactly. Saturating subs match the file idiom and are exact here
        // (k >= 1 in-body, idx <= k-1).
        for idx in 0..k {
            let li = k.saturating_sub(1).saturating_sub(idx);
            // total: `saturating_add` (not `+ 1`): `li < k <= isize::MAX`, so
            // the add never saturates — identical 1-indexed layer tag.
            let layer1 = li.saturating_add(1);
            let width = self.layer_dim(layer1);

            // Step 0 (FIRST LAYER ONLY): divert a μ-share of negative-coefficient
            // cut pairs into the joint cut premise.  For a pair (i,j) with
            // c_i,c_j < 0 and joint const B < t_i+t_j, take μ = min(|c_i|,|c_j|),
            // give the cut premise multiplier μ (joint const μ·B), and SUBTRACT μ
            // from the magnitude each neuron sends to its per-neuron upper envelope.
            // The cut contributes NO z-term, so the diverted share is a flat μ·B
            // instead of μ·(t_i+t_j) — a strictly smaller const_acc when B<t_i+t_j.
            // `Vec::new()` + push (not `vec![_; width]`): unbounded-count bulk
            // fill → hardened allocation obligation; push loop is identical.
            let mut cut_share = Vec::new();
            for _ in 0..width {
                cut_share.push(Rat::ZERO);
            }
            if li == 0 {
                // Fail-CLOSED cut-unit reads (same pattern as
                // `cut2_box_b_exact`): `ci`/`cj` are caller-supplied first-layer
                // unit indices. Every entry of `cut_prem` already passed
                // `cut2_box_b_exact`, which rejects units outside the first-layer
                // width, so here `ci, cj < width` and the `Err` arms are
                // unreachable — while `.get().ok_or(Dimension)?` keeps a
                // malformed cut failing LOUD (an error, never a silent
                // `Rat::ZERO` read that would emit an unsound combination) with
                // no bounds/panic obligation. This block only runs for non-empty
                // `cuts`; the certified paths (`certify`, `certify_difference_*`)
                // pass none.
                // Free (nested) `fn`s, not an `oor` closure fed to
                // `ok_or_else`/`and_then` chains (those invoke the closure
                // through an absent `<{closure} as Fn>::call` shim). Same
                // fail-CLOSED semantics, same error value, same evaluation
                // order: a malformed cut still returns Err, never a silent read.
                fn cut_oor() -> DeepCrownError {
                    DeepCrownError::Dimension("cut unit index out of range".into())
                }
                fn cut_rat_at(v: &[Rat], i: usize) -> Result<Rat, DeepCrownError> {
                    match v.get(i) {
                        Some(x) => Ok(*x),
                        None => Err(cut_oor()),
                    }
                }
                // First-layer envelope read (`env_upper[0][i]`), fail-closed.
                // `.get(0)` (not `.first()`) keeps the flattening a plain
                // `match` pair; identical lookup.
                #[allow(clippy::get_first)]
                fn cut_env0_at(
                    env: &[Vec<(Rat, Rat, usize)>],
                    i: usize,
                ) -> Result<(Rat, Rat, usize), DeepCrownError> {
                    let e = match env.get(0) {
                        Some(row) => row.get(i),
                        None => None,
                    };
                    match e {
                        Some(x) => Ok(*x),
                        None => Err(cut_oor()),
                    }
                }
                for &(ci, cj, idx, b_ij) in &cut_prem {
                    let cco_i = cut_rat_at(&a_coeff, ci)?;
                    let cco_j = cut_rat_at(&a_coeff, cj)?;
                    if cco_i.is_negative() && cco_j.is_negative() {
                        let avail_i = cco_i.neg().sub(cut_rat_at(&cut_share, ci)?)?;
                        let avail_j = cco_j.neg().sub(cut_rat_at(&cut_share, cj)?)?;
                        let mu = if avail_i <= avail_j { avail_i } else { avail_j };
                        if !mu.is_positive() {
                            continue;
                        }
                        let (_, t_i, _) = cut_env0_at(&env_upper, ci)?;
                        let (_, t_j, _) = cut_env0_at(&env_upper, cj)?;
                        // gain condition: joint const beats the two per-neuron consts.
                        if b_ij < t_i.add(t_j)? {
                            // `idx` is a `push` premise index (`< mult.len()`) and
                            // `ci, cj < width == cut_share.len()` (established
                            // above), so `add_at`'s skip arms are unreachable —
                            // identical read-add-write, no `[]` obligation.
                            add_at(&mut mult, idx, mu)?; // ≥0 cut multiplier
                            const_acc = const_acc.add(mu.mul(b_ij)?)?; // +μ·B
                            add_at(&mut cut_share, ci, mu)?;
                            add_at(&mut cut_share, cj, mu)?;
                        }
                    }
                }
            }

            // Step 1: eliminate each a⁽ᴸ⁾ⱼ via its sign-appropriate envelope,
            // producing a coefficient on z⁽ᴸ⁾ⱼ.  At the first layer, the per-neuron
            // share already absorbed by cuts (`cut_share[j]`) is removed from |c|.
            // `j < width`, and `a_coeff`/`z_coeff`/`cut_share` all have `width`
            // entries while `env_lower[li]`/`env_upper[li]` have one entry per
            // unit (len `width`), so every `.get()` guard matches — the skip arms
            // are unreachable and the accumulation is identical. (`idx` is a
            // premise index from `push`, in range for `add_at`.)
            // `Vec::new()` + push (not `vec![_; width]`): unbounded-count bulk
            // fill → hardened allocation obligation; push loop is identical.
            let mut z_coeff = Vec::new();
            for _ in 0..width {
                z_coeff.push(Rat::ZERO);
            }
            for j in 0..width {
                let c = a_coeff.get(j).copied().unwrap_or(Rat::ZERO);
                if c.is_positive() {
                    // Plain `match` flattening (not `.and_then(|row| ..)`): the
                    // closure would be invoked through an absent `Fn::call`
                    // shim; identical row selection.
                    let low_opt = match env_lower.get(li) {
                        Some(row) => row.get(j),
                        None => None,
                    };
                    if let Some(&(p, q, idx)) = low_opt {
                        add_at(&mut mult, idx, c)?; // scale lower envelope by c
                        add_at(&mut z_coeff, j, c.mul(p)?)?;
                        const_acc = const_acc.add(c.mul(q.neg())?)?; // −c·q
                    }
                } else if c.is_negative() {
                    let up_opt = match env_upper.get(li) {
                        Some(row) => row.get(j),
                        None => None,
                    };
                    if let Some(&(r, t, idx)) = up_opt {
                        // remaining magnitude after the cut diversion (0 at layers > 0).
                        let mag = c
                            .neg()
                            .sub(cut_share.get(j).copied().unwrap_or(Rat::ZERO))?;
                        if mag.is_positive() {
                            add_at(&mut mult, idx, mag)?; // scale upper envelope by remaining |c|
                            add_at(&mut z_coeff, j, mag.neg().mul(r)?)?; // (−mag)·r
                            const_acc = const_acc.add(mag.mul(t)?)?; // +mag·t
                        }
                    }
                }
            }

            // Step 2: eliminate each z⁽ᴸ⁾ⱼ through the affine layer, producing
            // coefficients on the previous activations (a⁽ᴸ⁻¹⁾, or x at L==1).
            // `checked_sub` makes the previous-layer index free of usize underflow
            // (`layer1 >= 1`, and `layer1 == 1` is the `x`-input case).
            let prev_dim = match layer1.checked_sub(1) {
                Some(0) | None => n,
                Some(prev) => self.layer_dim(prev),
            };
            // `j < width`; `z_coeff` has `width` entries, `z_ge[li]`/`z_le[li]`
            // have one premise index per unit (len `width`), `weights[li]` has
            // `width` rows and each row's width equals `prev_dim` (`validate`), so
            // every `.get()` guard matches — skip arms unreachable, identical
            // result. (`z_ge[li][j]`/`z_le[li][j]` are `push` premise indices.)
            // `Vec::new()` + push (not `vec![_; prev_dim]`): unbounded-count
            // bulk fill → hardened allocation obligation; push loop is identical.
            let mut prev_coeff = Vec::new();
            for _ in 0..prev_dim {
                prev_coeff.push(Rat::ZERO);
            }
            for j in 0..width {
                let c = z_coeff.get(j).copied().unwrap_or(Rat::ZERO);
                if c.is_zero() {
                    continue;
                }
                // Plain `match` flattenings (not `.and_then(|row| ..)`): the
                // closures would be invoked through absent `Fn::call` shims;
                // identical premise-index / row selection.
                if c.is_positive() {
                    let ge_opt = match z_ge.get(li) {
                        Some(row) => row.get(j),
                        None => None,
                    };
                    if let Some(&pidx) = ge_opt {
                        add_at(&mut mult, pidx, c)?;
                    }
                } else {
                    let le_opt = match z_le.get(li) {
                        Some(row) => row.get(j),
                        None => None,
                    };
                    if let Some(&pidx) = le_opt {
                        add_at(&mut mult, pidx, c.neg())?;
                    }
                }
                let w_row_opt = match self.weights.get(li) {
                    Some(l) => l.get(j),
                    None => None,
                };
                if let Some(row) = w_row_opt {
                    for (i, wji) in row.iter().enumerate() {
                        add_at(&mut prev_coeff, i, c.mul(*wji)?)?;
                    }
                }
                const_acc = const_acc.add(c.mul(bias_at(&self.biases, li, j))?.neg())?;
                // −c·b
            }

            // The previous layer's activation coefficients become `a_coeff` for
            // the next backward iteration (or the input coefficients at L==1).
            a_coeff = prev_coeff;
        }

        // After the loop, `a_coeff` holds the coefficients on the inputs x that
        // came through the network; the subtracted affine functional's initial
        // −gᵢ coefficients (from the normalized read-out) join them here.
        // `a_coeff` now has `n` entries (input coefficients); `g_coeffs.len() == n`
        // (validated) and `quad_x.len() == n`, so every `i` below indexes it in
        // range — total read-modify-write, fallbacks unreachable.
        if let Some((g_coeffs, _)) = subtract {
            for (i, g_i) in g_coeffs.iter().enumerate() {
                let cur = a_coeff.get(i).copied().unwrap_or(Rat::ZERO);
                let next = cur.sub(*g_i)?;
                if let Some(slot) = a_coeff.get_mut(i) {
                    *slot = next;
                }
            }
        }
        // The quadratic side's tⱼ eliminations contributed input coefficients too.
        for (i, qx) in quad_x.iter().enumerate() {
            add_at(&mut a_coeff, i, *qx)?;
        }
        // Eliminate each xᵢ through the box.
        // `i < n`; `a_coeff`/`box_l`/`box_u`/`input_lower`/`input_upper` are all
        // length `n`, so every `.get()` guard matches — skip arms unreachable,
        // identical result. (`box_l[i]`/`box_u[i]` are `push` premise indices.)
        for i in 0..n {
            let d = a_coeff.get(i).copied().unwrap_or(Rat::ZERO);
            if d.is_zero() {
                continue;
            }
            if d.is_positive() {
                if let Some(&pidx) = box_l.get(i) {
                    add_at(&mut mult, pidx, d)?;
                }
                let li_v = self.input_lower.get(i).copied().unwrap_or(Rat::ZERO);
                const_acc = const_acc.add(d.mul(li_v.neg())?)?; // −d·l
            } else {
                let mag = d.neg();
                if let Some(&pidx) = box_u.get(i) {
                    add_at(&mut mult, pidx, mag)?;
                }
                let ui_v = self.input_upper.get(i).copied().unwrap_or(Rat::ZERO);
                const_acc = const_acc.add(mag.mul(ui_v)?)?; // |d|·u
            }
        }

        // The combination is exactly  −y ≤ const_acc, i.e. y ≥ −const_acc.
        let lower_bound = const_acc.neg();
        if threshold > lower_bound {
            return Err(DeepCrownError::ThresholdAboveBound {
                threshold: format!("{}/{}", threshold.num(), threshold.den()),
                bound: format!("{}/{}", lower_bound.num(), lower_bound.den()),
            });
        }

        // `.minimized()`: drop the dead premise rows (multiplier exactly zero —
        // `push` seeds every premise with `Rat::ZERO` and the backward pass only
        // touches the rows it uses), so the emitted certificate carries no rows
        // that contribute nothing to the Farkas combination. Fail-closed inside
        // `minimized`: the smaller cert is kept only when `check_entailment` /
        // `check_farkas` accept it with the IDENTICAL bounds/residual as the
        // full cert; otherwise the full cert is returned unchanged.
        let entailment = EntailmentCertificate {
            premises: premises.clone(),
            multipliers: mult.clone(),
            conclusion: LinearConstraint::with_kind(
                ConstraintKind::Ge,
                &[("y", Rat::ONE)],
                threshold,
            ),
        }
        .minimized();

        // Farkas: append the negated property y < threshold with multiplier 1.
        let mut f_constraints = premises;
        let mut f_mult = mult;
        f_constraints.push(LinearConstraint::with_kind(
            ConstraintKind::Lt,
            &[("y", Rat::ONE)],
            threshold,
        ));
        f_mult.push(Rat::ONE);
        let farkas = FarkasCertificate {
            constraints: f_constraints,
            multipliers: f_mult,
        }
        .minimized();

        // Fail-CLOSED exit gate: re-check the poison flag immediately before
        // returning `Ok` — a fallback arm reached DURING this build (after the
        // entry gate) must also refuse the certificate.
        if crate::rational::poisoned() {
            return Err(crate::err_barrier(DeepCrownError::ArenaPoisoned));
        }
        Ok(CertifiedDeep {
            entailment,
            farkas,
            lower_bound,
            preact,
        })
    }

    /// Evaluate the true (un-relaxed) deep network at an exact rational point.
    ///
    /// `x` must have exactly `input_dim()` (= `input_lower.len()`) entries;
    /// every in-repo caller builds `x` with exactly that many pushes. A mis-sized
    /// point is rejected fail-CLOSED with [`RatError::Dimension`] (was a fail-loud
    /// `assert_eq!`, which the strict verifier could not discharge — the length
    /// equality is an interprocedural precondition, and a `#[trust::requires]`
    /// method-call predicate is currently unparseable by the contract lowering).
    ///
    /// # Errors
    /// [`RatError::Dimension`] for a wrong-length point; propagates
    /// exact-rational arena failures.
    pub fn eval(&self, x: &[Rat]) -> Result<Rat, RatError> {
        crate::rational::ensure_healthy()?;
        // Fail-CLOSED dimension guard (was a fail-loud `assert_eq!`): returns a
        // sound `Err` on a mis-sized point instead of panicking, so the strict
        // verifier sees a total function (no unprovable `x.len() == input_dim`
        // assert). Unreachable for in-repo callers (they size `x` exactly).
        if x.len() != self.input_dim() {
            return Err(RatError::Dimension {
                expected: self.input_dim(),
                got: x.len(),
            });
        }
        let mut act: Vec<Rat> = x.to_vec();
        for (w, b) in self.weights.iter().zip(&self.biases) {
            // `Vec::new()`: the `with_capacity(w.len())` hint on an unbounded
            // `&self` width carries a hardened allocation obligation the model
            // cannot bound; amortized growth is noise next to the Rat math.
            let mut next = Vec::new();
            for (row, bias) in w.iter().zip(b) {
                let mut z = *bias;
                for (wji, ai) in row.iter().zip(&act) {
                    z = z.add(wji.mul(*ai)?)?;
                }
                next.push(if z.is_positive() { z } else { Rat::ZERO });
            }
            act = next;
        }
        let mut y = self.out_bias;
        for (wj, aj) in self.out_weight.iter().zip(&act) {
            y = y.add(wj.mul(*aj)?)?;
        }
        crate::rational::ensure_healthy()?;
        Ok(y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{entailment_to_json, farkas_to_json};
    use crate::selfcheck::{check_entailment, check_farkas};

    struct PoisonReset;

    impl Drop for PoisonReset {
        fn drop(&mut self) {
            crate::rational::set_poisoned_for_test(false);
        }
    }

    fn r(n: i128, d: i128) -> Rat {
        Rat::new(n, d).unwrap()
    }

    #[test]
    fn dyadic_round_out_is_outward_and_tight() {
        // Non-dyadic values: the rounded bound must lie on the OUTWARD side and
        // within one f64 ulp of the exact value.
        let cases = [
            r(1, 3),
            r(-1, 3),
            r(22, 7),
            r(-355, 113),
            r(1, 1_000_000_007),
        ];
        for v in cases {
            let dn = dyadic_round_out(v, true);
            let up = dyadic_round_out(v, false);
            assert!(dn <= v, "round-down went inward");
            assert!(up >= v, "round-up went inward");
            // Tightness: |v - rounded| <= 2^-40 * (|v| + 1), far looser than the
            // actual 1-ulp guarantee but immune to representation edge cases.
            let tol = v.abs().add(Rat::ONE).unwrap().mul(r(1, 1 << 40)).unwrap();
            assert!(v.sub(dn).unwrap() <= tol);
            assert!(up.sub(v).unwrap() <= tol);
        }
        // Exact dyadics round to themselves in both directions.
        for v in [Rat::ZERO, Rat::ONE, r(-3, 8), r(5, 4)] {
            assert_eq!(dyadic_round_out(v, true), v);
            assert_eq!(dyadic_round_out(v, false), v);
        }
    }

    /// A concrete 4 -> 5 -> 5 -> 3 -> scalar ReLU network (k = 3 hidden layers).
    fn deep_4_5_5_3() -> DeepReluProblem {
        let w1 = vec![
            vec![r(1, 2), r(-1, 2), r(1, 1), r(1, 2)],
            vec![r(-1, 1), r(1, 2), r(1, 2), r(-1, 2)],
            vec![r(1, 2), r(1, 1), r(-1, 2), r(1, 2)],
            vec![r(1, 1), r(-1, 2), r(1, 2), r(-1, 1)],
            vec![r(-1, 2), r(1, 1), r(1, 2), r(1, 2)],
        ];
        let b1 = vec![r(1, 2), r(-1, 2), r(0, 1), r(1, 2), r(-1, 2)];
        let w2 = vec![
            vec![r(1, 2), r(-1, 2), r(1, 2), r(1, 1), r(-1, 2)],
            vec![r(1, 1), r(1, 2), r(-1, 2), r(-1, 2), r(1, 2)],
            vec![r(-1, 2), r(1, 2), r(1, 1), r(1, 2), r(-1, 2)],
            vec![r(1, 2), r(-1, 2), r(1, 2), r(-1, 1), r(1, 2)],
            vec![r(-1, 1), r(1, 2), r(-1, 2), r(1, 2), r(1, 1)],
        ];
        let b2 = vec![r(1, 2), r(-1, 2), r(1, 2), r(0, 1), r(1, 2)];
        let w3 = vec![
            vec![r(1, 2), r(-1, 2), r(1, 2), r(-1, 1), r(1, 2)],
            vec![r(-1, 2), r(1, 1), r(1, 2), r(1, 2), r(-1, 2)],
            vec![r(1, 2), r(1, 2), r(-1, 1), r(-1, 2), r(1, 2)],
        ];
        let b3 = vec![r(1, 2), r(-1, 2), r(1, 2)];
        DeepReluProblem {
            weights: vec![w1, w2, w3],
            biases: vec![b1, b2, b3],
            out_weight: vec![r(1, 1), r(-1, 2), r(1, 2)],
            out_bias: r(2, 1),
            input_lower: vec![r(-1, 4), r(-1, 4), r(-1, 4), r(-1, 4)],
            input_upper: vec![r(1, 4), r(1, 4), r(1, 4), r(1, 4)],
            alpha: None,
            interm_round: false,
        }
    }

    #[test]
    fn three_hidden_layers_certificate_is_self_consistent() -> Result<(), String> {
        let net = deep_4_5_5_3();
        assert_eq!(net.depth(), 3);
        // A certification failure is an `Err` that FAILS the test (fail-closed).
        let cert = net
            .certify(Rat::ZERO)
            .map_err(|e| format!("certify y >= 0: {e}"))?;
        // CROWN bound is positive (property y >= 0 is provable for this box).
        assert!(cert.lower_bound >= Rat::ZERO);
        // The entailment and farkas certificates pass NY's mirror of Clean.
        let (derived, claimed) = check_entailment(&cert.entailment).unwrap();
        assert!(derived <= claimed);
        check_farkas(&cert.farkas).unwrap();
        // Dead-row minimization: no premise with multiplier exactly zero
        // survives construction (the seeded-zero rows are dropped).
        for m in &cert.entailment.multipliers {
            assert!(!m.is_zero(), "dead premise row survived minimization");
        }
        for m in &cert.farkas.multipliers {
            assert!(!m.is_zero(), "dead constraint row survived minimization");
        }
        // The cert mentions all multi-layer variables.
        let mut vars = std::collections::BTreeSet::new();
        for p in &cert.entailment.premises {
            for k in p.coefficients.keys() {
                vars.insert(k.clone());
            }
        }
        for tag in ["x0", "z1_0", "a1_0", "z2_0", "a2_0", "z3_0", "a3_0", "y"] {
            assert!(vars.contains(tag), "missing variable {tag}");
        }
        // JSON serialization succeeds (every rational fits Clean's i64 encoding).
        let ent = entailment_to_json(&cert.entailment).unwrap();
        let far = farkas_to_json(&cert.farkas).unwrap();
        // Emit for the cross-repo round-trip against Clean's real binary.
        if let Ok(dir) = std::env::var("NY_CERT_OUT_DIR") {
            std::fs::write(
                format!("{dir}/ny_deep_entailment.json"),
                serde_json::to_string_pretty(&ent).unwrap(),
            )
            .unwrap();
            std::fs::write(
                format!("{dir}/ny_deep_farkas.json"),
                serde_json::to_string_pretty(&far).unwrap(),
            )
            .unwrap();
        }
        Ok(())
    }

    #[test]
    fn certified_bound_is_sound_against_true_network() {
        let net = deep_4_5_5_3();
        let cert = net.certify(Rat::ZERO).unwrap();
        let lb = cert.lower_bound;
        // Exact-eval the TRUE network on a dense grid; the certified bound must
        // never exceed any true output.
        let lows = &net.input_lower;
        let highs = &net.input_upper;
        let dim = lows.len();
        let steps = 6u64; // 7^4 = 2401 points
        let total = (steps + 1).pow(dim as u32);
        for idx in 0..total {
            let mut x = Vec::with_capacity(dim);
            let mut rem = idx;
            for a in 0..dim {
                let s = (rem % (steps + 1)) as i128;
                rem /= steps + 1;
                let span = highs[a].sub(lows[a]).unwrap();
                let frac = Rat::new(s, steps as i128).unwrap();
                x.push(lows[a].add(span.mul(frac).unwrap()).unwrap());
            }
            let y = net.eval(&x).unwrap();
            assert!(
                lb <= y,
                "UNSOUND: true output {y:?} < certified bound {lb:?}"
            );
        }
    }

    #[test]
    fn threshold_above_bound_is_rejected() {
        let net = deep_4_5_5_3();
        let m = net.certify(Rat::from_int(-1_000_000)).unwrap().lower_bound;
        // Anything strictly above the CROWN bound must be refused.
        let too_high = m.add(Rat::ONE).unwrap();
        assert!(matches!(
            net.certify(too_high),
            Err(DeepCrownError::ThresholdAboveBound { .. })
        ));
    }

    // --- Linear-difference read-out (ground-truth dominance, plan §4) --------

    /// f(x) = relu(x0) + relu(-x0) + 3 = |x0| + 3 as a 1->2->scalar FC-ReLU net.
    fn abs_plus_3() -> DeepReluProblem {
        DeepReluProblem {
            weights: vec![vec![vec![r(1, 1)], vec![r(-1, 1)]]],
            biases: vec![vec![Rat::ZERO, Rat::ZERO]],
            out_weight: vec![r(1, 1), r(1, 1)],
            out_bias: r(3, 1),
            input_lower: vec![r(-1, 1)],
            input_upper: vec![r(1, 1)],
            alpha: None,
            interm_round: false,
        }
    }

    #[test]
    fn certify_fails_closed_on_poisoned_arena() {
        let net = abs_plus_3();
        // Poison the (thread-local) arena via the test-only setter: certify
        // must refuse at the entry gate with ArenaPoisoned, not emit.
        crate::rational::set_poisoned_for_test(true);
        let _reset = PoisonReset;
        assert!(matches!(
            net.preact_bounds(),
            Err(DeepCrownError::Rat(RatError::Poisoned))
        ));
        assert!(matches!(
            net.preact_bounds_crown(),
            Err(DeepCrownError::Rat(RatError::Poisoned))
        ));
        assert_eq!(net.eval(&[Rat::ZERO]), Err(RatError::Poisoned));
        assert!(
            matches!(net.certify(Rat::ZERO), Err(DeepCrownError::ArenaPoisoned)),
            "certify must fail CLOSED while the arena is poisoned"
        );
        // Clearing the flag restores normal fail-open/fail-closed behaviour:
        // the same tiny problem certifies again.
        crate::rational::set_poisoned_for_test(false);
        let cert = net
            .certify(Rat::ZERO)
            .expect("certify must succeed once the poison flag is cleared");
        assert!(cert.lower_bound >= Rat::ZERO);
    }

    #[test]
    fn difference_linear_dominance_certificate_is_valid_and_sound() -> Result<(), String> {
        // Ground truth: the signed plane residual g(x) = 2·x0 − 1/2.
        // h(x) = f(x) − g(x) = |x0| − 2·x0 + 7/2 ≥ 3/2 on [−1, 1] (min at x0=1).
        let net = abs_plus_3();
        let g_coeffs = vec![r(2, 1)];
        let g_offset = r(-1, 2);
        let cert = net
            .certify_difference_linear(&g_coeffs, g_offset, Rat::ZERO)
            .map_err(|e| format!("certify h >= 0: {e}"))?;

        // Both ReLU units are unstable on [−1,1]; the CROWN relaxation of |x0|
        // loses at most the envelope slack, and the exact certified bound must
        // still be non-negative and never exceed the true minimum 3/2.
        assert!(!cert.lower_bound.is_negative(), "dominance closes");
        assert!(
            cert.lower_bound <= r(3, 2),
            "bound cannot beat the true min"
        );

        // The UNCHANGED self-checkers accept the certificate (the g-side rows
        // are plain linear premises for the same Farkas combination check).
        let (derived, claimed) = check_entailment(&cert.entailment).unwrap();
        assert!(derived <= claimed);
        check_farkas(&cert.farkas).unwrap();

        // The read-out premises mention the input variable with g's coefficient
        // (the "exact rational rows" the ground-truth side contributes). Only
        // the direction the backward pass actually uses survives dead-row
        // minimization (the other half of the <=/>= pair carried multiplier 0).
        let y_rows: Vec<_> = cert
            .entailment
            .premises
            .iter()
            .filter(|p| p.coefficients.contains_key("y"))
            .collect();
        assert_eq!(y_rows.len(), 1, "the live direction of the read-out pair");
        for row in y_rows {
            assert_eq!(row.coefficients.get("x0"), Some(&r(2, 1)));
        }

        // SOUNDNESS: h(x) = f(x) − g(x) must dominate the certified bound at
        // every grid point (exact rational evaluation of the TRUE network).
        for step in 0..=40i128 {
            let x = r(-1, 1).add(r(step, 20)).unwrap();
            let f_val = net.eval(&[x]).unwrap();
            let g_val = g_coeffs[0].mul(x).unwrap().add(g_offset).unwrap();
            let h_val = f_val.sub(g_val).unwrap();
            assert!(
                cert.lower_bound <= h_val,
                "UNSOUND: h({x:?}) = {h_val:?} < certified {:?}",
                cert.lower_bound
            );
        }

        // JSON serialization stays within Clean's encoding.
        entailment_to_json(&cert.entailment).unwrap();
        farkas_to_json(&cert.farkas).unwrap();
        Ok(())
    }

    #[test]
    fn difference_linear_matches_plain_certify_for_zero_g() {
        // Subtracting the zero functional must reproduce the plain certificate
        // bound exactly.
        let net = deep_4_5_5_3();
        let plain = net.certify(Rat::ZERO).unwrap();
        let zero_g = vec![Rat::ZERO; 4];
        let diff = net
            .certify_difference_linear(&zero_g, Rat::ZERO, Rat::ZERO)
            .unwrap();
        assert_eq!(plain.lower_bound, diff.lower_bound);
        check_entailment(&diff.entailment).unwrap();
        check_farkas(&diff.farkas).unwrap();
    }

    #[test]
    fn difference_linear_rejects_mismatched_coeffs_and_high_threshold() {
        let net = abs_plus_3();
        assert!(matches!(
            net.certify_difference_linear(&[Rat::ZERO, Rat::ZERO], Rat::ZERO, Rat::ZERO),
            Err(DeepCrownError::Dimension(_))
        ));
        // h ≥ 3/2 is the true min; a threshold of 100 must be refused.
        assert!(matches!(
            net.certify_difference_linear(&[r(2, 1)], r(-1, 2), Rat::from_int(100)),
            Err(DeepCrownError::ThresholdAboveBound { .. })
        ));
    }

    // --- Quadratic-difference read-out (pow2 envelopes, plan §4) -------------

    #[test]
    fn difference_quadratic_dominance_certificate_is_valid_and_sound() -> Result<(), String> {
        // Ground truth: the 1-D "sphere" residual g(x) = x0² − 1/4.
        // h(x) = f(x) − g(x) = |x0| − x0² + 13/4 ≥ 13/4 on [−1, 1]
        // (|t| − t² ≥ 0 for |t| ≤ 1, with equality at t ∈ {0, ±1}).
        let net = abs_plus_3();
        let g_coeffs = vec![Rat::ZERO];
        let g_offset = r(-1, 4);
        let quad = vec![QuadTerm {
            coeff: Rat::ONE,
            lin: vec![Rat::ONE],
            offset: Rat::ZERO,
        }];
        let cert = net
            .certify_difference_quadratic(&g_coeffs, g_offset, &quad, Rat::ZERO)
            .map_err(|e| format!("certify h >= 0: {e}"))?;
        assert!(!cert.lower_bound.is_negative(), "dominance closes");
        assert!(
            cert.lower_bound <= r(13, 4),
            "bound cannot beat the true min"
        );

        // The UNCHANGED self-checkers accept the certificate: the pow2
        // envelope rows are plain linear premises over (s0, t0, x0) for the
        // same non-negative Farkas combination check.
        let (derived, claimed) = check_entailment(&cert.entailment).unwrap();
        assert!(derived <= claimed);
        check_farkas(&cert.farkas).unwrap();

        // The square variable's LIVE premises survive minimization: on [−1, 1]
        // the secant is s0 − 0·t0 ≤ 1 (used to upper-bound the subtracted
        // square) and the read-out row mentions s0 — 2 rows. The tangent
        // (c = 0: s0 ≥ 0) and the definitional <=/>= pair on t0 carry
        // multiplier 0 here (the secant's t-coefficient vanishes on the
        // symmetric box, so t0 is never eliminated through them) — dead rows,
        // dropped, and with them the variable t0 vanishes entirely.
        let s_rows: Vec<_> = cert
            .entailment
            .premises
            .iter()
            .filter(|p| p.coefficients.contains_key("s0"))
            .collect();
        assert_eq!(s_rows.len(), 2, "secant + read-out mention s0");
        let t_rows = cert
            .entailment
            .premises
            .iter()
            .filter(|p| p.coefficients.contains_key("t0"))
            .count();
        assert_eq!(
            t_rows, 0,
            "t0 only occurred in dead rows and vanishes with them"
        );

        // SOUNDNESS: h(x) must dominate the certified bound at every grid
        // point (exact rational evaluation of the TRUE network minus g).
        for step in 0..=40i128 {
            let x = r(-1, 1).add(r(step, 20)).unwrap();
            let f_val = net.eval(&[x]).unwrap();
            let g_val = x.mul(x).unwrap().add(g_offset).unwrap();
            let h_val = f_val.sub(g_val).unwrap();
            assert!(
                cert.lower_bound <= h_val,
                "UNSOUND: h({x:?}) = {h_val:?} < certified {:?}",
                cert.lower_bound
            );
        }

        entailment_to_json(&cert.entailment).unwrap();
        farkas_to_json(&cert.farkas).unwrap();
        Ok(())
    }

    #[test]
    fn difference_quadratic_negative_square_uses_tangent_exactly() -> Result<(), String> {
        // g(x) = −x0² on the STABLE box [1/4, 1]: f = x0 + 3 exactly (both
        // units stable), and the tangent at c = l = 1/4 gives
        //   h = f + x0² ≥ (x0 + 3) + (x0/2 − 1/16) ≥ 3 + 1/4 + 1/8 − 1/16 = 53/16,
        // which is exactly the true minimum h(1/4) = 1/4 + 3 + 1/16. The
        // certified bound must be EXACT here.
        let net = DeepReluProblem {
            weights: vec![vec![vec![r(1, 1)], vec![r(-1, 1)]]],
            biases: vec![vec![Rat::ZERO, Rat::ZERO]],
            out_weight: vec![r(1, 1), r(1, 1)],
            out_bias: r(3, 1),
            input_lower: vec![r(1, 4)],
            input_upper: vec![r(1, 1)],
            alpha: None,
            interm_round: false,
        };
        let quad = vec![QuadTerm {
            coeff: r(-1, 1),
            lin: vec![Rat::ONE],
            offset: Rat::ZERO,
        }];
        let cert = net
            .certify_difference_quadratic(&[Rat::ZERO], Rat::ZERO, &quad, Rat::ZERO)
            .map_err(|e| format!("certify h >= 0: {e}"))?;
        assert_eq!(cert.lower_bound, r(53, 16), "tangent bound is exact");
        check_entailment(&cert.entailment).unwrap();
        check_farkas(&cert.farkas).unwrap();
        for step in 0..=30i128 {
            let x = r(1, 4).add(r(step, 40)).unwrap();
            let f_val = net.eval(&[x]).unwrap();
            let h_val = f_val.add(x.mul(x).unwrap()).unwrap();
            assert!(cert.lower_bound <= h_val, "UNSOUND at {x:?}");
        }
        Ok(())
    }

    #[test]
    fn difference_quadratic_with_no_squares_matches_linear() {
        let net = abs_plus_3();
        let lin = net
            .certify_difference_linear(&[r(2, 1)], r(-1, 2), Rat::ZERO)
            .unwrap();
        let quadless = net
            .certify_difference_quadratic(&[r(2, 1)], r(-1, 2), &[], Rat::ZERO)
            .unwrap();
        assert_eq!(lin.lower_bound, quadless.lower_bound);
    }

    #[test]
    fn difference_quadratic_rejects_mismatched_presquare_dim() {
        let net = abs_plus_3();
        let quad = vec![QuadTerm {
            coeff: Rat::ONE,
            lin: vec![Rat::ONE, Rat::ONE],
            offset: Rat::ZERO,
        }];
        assert!(matches!(
            net.certify_difference_quadratic(&[Rat::ZERO], Rat::ZERO, &quad, Rat::ZERO),
            Err(DeepCrownError::Dimension(_))
        ));
    }
}
