// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Fast f32 point back-propagation (reverse-mode autodiff at a single concrete
//! input point) for the PGD attack.
//!
//! # What this computes
//!
//! [`GraphNetwork::attack_point_gradient`] returns
//! `d(spec_row · network_output) / d(input)` evaluated **at** the concrete point
//! `x`, reshaped to `x.shape()`. This is an ATTACK gradient: it is the exact
//! local point-Jacobian, and it does **not** need certified soundness (any
//! counterexample the attack finds is re-checked concretely elsewhere).
//!
//! # Why the CROWN backward machinery gives the exact point-Jacobian here
//!
//! At a concrete point the input "box" is degenerate (`[x, x]`), so every node's
//! forward interval is degenerate too. In particular every ReLU's pre-activation
//! interval is `[v, v]`, so the CROWN ReLU relaxation collapses to the EXACT mask
//! (slope `1` if `v > 0`, `0` if `v < 0`, zero intercept) — no crossing /
//! relaxation case ever fires. Consequently a single-row cotangent (the
//! `spec_row`) propagated backward through the network with `err = None` stays
//! exact (`lower_a == upper_a` up to floating-point roundoff), and the row that
//! lands on the network input IS the exact gradient.
//!
//! # Implementation
//!
//! This mirrors the reverse loop in [`super::propagation`]
//! (`crown_backward_with_relaxation_and_deadline_and_truncation`) but:
//! - seeds the OUTPUT node with the `spec_row` (a `1 × num_outputs`
//!   [`LinearBounds`] with `lower_a == upper_a == spec_row`, zero bias,
//!   `*_err = None`) instead of `identity(output_dim)`, so we track a single row;
//! - stays entirely in Dense mode (no Patches);
//! - reuses the exact same per-node backward dispatch
//!   ([`dispatch_backward_layer`] for the linear/graph ops,
//!   [`dispatch_relu_backward`] for ReLU) and the exact same accumulation
//!   frontier ([`CrownMergeAccumulator`] +
//!   [`apply_dense_backward_dispatch_result_with_deadline`]) so residual fan-in summation and
//!   the bias-to-input routing are byte-for-byte the same as the certified path.
//!
//! The Dense-only invariant is enforced at both accumulator extraction
//! boundaries. An unexpected [`CrownBounds::Patches`] carrier declines this
//! attack route (`Ok(None)`) without materializing it; no safe reason exists to
//! open an unpolled Patches allocation in a non-certified optimization helper.
//!
//! # Correctness-first note (PERF markers below)
//!
//! Both the Linear backward (`aw_f64_with_abssum`, invoked inside
//! `dispatch_backward_layer`'s `Layer::Linear` arm) and the sound conv backward
//! (`Conv2d` arm) compute in f64 for soundness. For an ATTACK gradient that is
//! unnecessary — see the `// PERF:` markers for exactly where a plain-f32 GEMM
//! (`crate::fast_f32_gemm::with_engine`) and a direct `conv2d_transpose` could be
//! swapped in once this passes its gradient-check. Correctness against the exact
//! oracle comes first; the f32 swap is a follow-up.

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_core::{GemmEngine, NyError};
use ny_tensor::BoundedTensor;
use std::cell::Cell;
use std::collections::HashMap;
use std::time::Instant;

use crate::bounds::patches::CrownBounds;
use crate::bounds::LinearBounds;
use crate::layers::Layer;
use crate::network::backward_dispatch::{
    dispatch_backward_layer, BackwardDispatchResult, DispatchContext,
};
use crate::network::core::{
    apply_dense_backward_dispatch_result_with_deadline, GraphNetwork, NETWORK_INPUT,
};
use crate::network::CrownMergeAccumulator;
use crate::MulBinaryRelaxationMode;

use super::backward_node_dispatch::{dispatch_relu_backward, NodeDispatchResult};

/// Consume a carrier only when the point-VJP Dense-only invariant holds.
///
/// This route seeds Dense and calls only Dense accumulation helpers. Keeping
/// the invariant as a typed boundary makes any future accidental Patches
/// ingress fail closed without an opaque `into_dense()` allocation.
fn take_attack_dense_carrier(carrier: CrownBounds) -> Option<LinearBounds> {
    match carrier {
        CrownBounds::Dense(bounds) => Some(bounds),
        CrownBounds::Patches(_) => None,
    }
}

/// Map finite-resource authority to this attack helper's documented `None`
/// fallback while preserving structural errors.
fn attack_step_or_decline<T>(result: ny_core::Result<T>) -> anyhow::Result<Option<T>> {
    match result {
        Ok(value) => Ok(Some(value)),
        Err(error) if error.is_deadline_exceeded() || error.is_cpu_memory_exceeded() => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn attack_publication_completed(result: ny_core::Result<()>) -> anyhow::Result<bool> {
    Ok(attack_step_or_decline(result)?.is_some())
}

fn anyhow_attack_resource_refusal(error: &anyhow::Error) -> bool {
    error
        .downcast_ref::<NyError>()
        .is_some_and(|error| error.is_deadline_exceeded() || error.is_cpu_memory_exceeded())
}

// ---------------------------------------------------------------------------
// ATTACK-only soft-sign surrogate sharpness (β) — thread-local ramp control
// ---------------------------------------------------------------------------

/// Default soft-sign surrogate sharpness β for the ATTACK point-gradient's
/// [`Layer::Sign`] arm. Historically a hard-coded constant; kept as the default
/// so every caller that does NOT opt into the ramp behaves exactly as before
/// (byte-identical). Matches `pgd_attack`'s `SMOOTH_SIGN_BETA`.
pub const DEFAULT_ATTACK_SIGN_BETA: f32 = 10.0;

/// Reachability window for the masked straight-through Sign surrogate
/// (#traffic-masked-ste). `Some(t)` selects mode (C): slope `1` where
/// `|z| <= t`, `0` elsewhere. `None` keeps the tanh modes (A)/(B).
///
/// Default `Some(6.0)`: with ±1 weights and no biases these BNN pre-activations
/// are EVEN integers, so `t = 6` is "at most 3 sign flips away" — the setting
/// measured to crack both traffic net-1 hard-tail rows (margins +20 and +28,
/// where unmasked STE stalls at -420 and the tanh modes never close).
///
/// `NY_MASKED_STE_WINDOW=0` restores the tanh surrogate exactly; any other
/// positive value overrides `t` (the prototype sweep was `t in {2,4,6,8}`).
///
/// ATTACK DIRECTION ONLY: selects a PGD search direction. Acceptance is the
/// unchanged `gate_sat_with_trusted_oracle`, so this cannot affect soundness.
fn masked_ste_window() -> Option<f32> {
    static W: std::sync::OnceLock<Option<f32>> = std::sync::OnceLock::new();
    *W.get_or_init(|| match std::env::var("NY_MASKED_STE_WINDOW") {
        Ok(v) => v.parse::<f32>().ok().filter(|t| *t > 0.0),
        Err(_) => Some(6.0),
    })
}

// ---------------------------------------------------------------------------
// ATTACK-only PER-SIGN-NODE reachability window (#traffic-input-linear-window)
// ---------------------------------------------------------------------------

thread_local! {
    /// Per-`Sign`-node masked-STE reachability window, one entry per unit, in the
    /// node's pre-activation flat order. Installed by the attack lane through
    /// [`AttackSteWindowsGuard`]; EMPTY by default, so every caller that does not
    /// install one is byte-identical to the flat [`masked_ste_window`] behaviour.
    ///
    /// ATTACK DIRECTION ONLY (see the `Layer::Sign` arm of
    /// [`GraphNetwork::attack_point_gradient`]): this picks which units carry a
    /// surrogate slope. Acceptance is always the unchanged trusted-oracle gate.
    static ATTACK_STE_WINDOWS: std::cell::RefCell<HashMap<String, Vec<f32>>> =
        std::cell::RefCell::new(HashMap::new());
}

/// RAII guard that installs a per-`Sign`-node masked-STE window map for the
/// current thread and restores the previous map on drop, so an attack lane's
/// per-instance windows cannot leak into later work on the same thread.
///
/// Build the map with [`GraphNetwork::input_linear_sign_windows`].
pub struct AttackSteWindowsGuard {
    prev: HashMap<String, Vec<f32>>,
}

impl AttackSteWindowsGuard {
    /// Install `windows`, remembering the previous thread-local map.
    pub fn install(windows: HashMap<String, Vec<f32>>) -> Self {
        let prev = ATTACK_STE_WINDOWS.with(|w| std::mem::replace(&mut *w.borrow_mut(), windows));
        Self { prev }
    }
}

impl Drop for AttackSteWindowsGuard {
    fn drop(&mut self) {
        let prev = std::mem::take(&mut self.prev);
        ATTACK_STE_WINDOWS.with(|w| *w.borrow_mut() = prev);
    }
}

/// Minimum surrogate β. The surrogate slope stays in `(0, β]`, so β must remain
/// a sane positive sharpness; [`set_attack_sign_beta`] clamps into `[MIN, MAX]`.
const ATTACK_SIGN_BETA_MIN: f32 = 2.0;
/// Maximum surrogate β (see [`ATTACK_SIGN_BETA_MIN`]).
const ATTACK_SIGN_BETA_MAX: f32 = 20.0;

thread_local! {
    // Thread-local so an attack loop can VARY β without threading a new
    // parameter through `attack_point_gradient`'s signature (which would force a
    // change to the certified-path caller `graph_pgd_exact.rs`). ATTACK
    // DIRECTION ONLY — read solely by the non-certified `Layer::Sign` surrogate.
    static ATTACK_SIGN_BETA: Cell<f32> = const { Cell::new(DEFAULT_ATTACK_SIGN_BETA) };
}

/// Current thread-local soft-sign β read by
/// [`GraphNetwork::attack_point_gradient`]'s `Layer::Sign` surrogate. Defaults
/// to [`DEFAULT_ATTACK_SIGN_BETA`].
pub fn attack_sign_beta() -> f32 {
    ATTACK_SIGN_BETA.with(|b| b.get())
}

/// Set the thread-local soft-sign β (clamped to `[2, 20]`).
///
/// ATTACK DIRECTION ONLY: β scales the *non-certified* Sign surrogate slope, so
/// it can only make the search sharper/smoother — it can NEVER change a verdict.
/// Every candidate the attack proposes is still concretely re-checked by the
/// unchanged trusted-oracle gate.
pub fn set_attack_sign_beta(beta: f32) {
    let clamped = beta.clamp(ATTACK_SIGN_BETA_MIN, ATTACK_SIGN_BETA_MAX);
    ATTACK_SIGN_BETA.with(|b| b.set(clamped));
}

/// RAII guard that restores the thread-local β to its prior value on drop, so a
/// ramp installed inside an attack loop cannot leak into later work on the same
/// thread. Construct with the value to install; the previous value is captured
/// and reinstated when the guard is dropped.
pub struct AttackSignBetaGuard {
    prev: f32,
}

impl AttackSignBetaGuard {
    /// Install `beta` (clamped) and remember the previous thread-local value.
    pub fn new(beta: f32) -> Self {
        let prev = attack_sign_beta();
        set_attack_sign_beta(beta);
        Self { prev }
    }
}

impl Drop for AttackSignBetaGuard {
    fn drop(&mut self) {
        // Restore the captured value verbatim (it was already valid/clamped).
        ATTACK_SIGN_BETA.with(|b| b.set(self.prev));
    }
}

/// ATTACK-DIRECTION-ONLY, ON BY DEFAULT for binarized nets: the
/// [`Layer::Sign`] surrogate slope in [`GraphNetwork::attack_point_gradient`] is
/// evaluated at a SMOOTH forward's pre-activations (a parallel forward pass where
/// every upstream Sign is replaced by `tanh(β·z/rms)` — see
/// [`GraphNetwork::smooth_sign_preactivations`]) with the UNFLOORED, FOLDED
/// `(β/rms)` normalisation, exactly matching the external `bnn_falsifier`
/// prototype's fully-smooth forward that cracks the hard-tail traffic_signs BNN
/// instances the default fixed hard-Sign forward times out on. It also collapses
/// each `Sign(Sign(x)+c)` pair to ONE surrogate (see
/// [`GraphNetwork::sign_is_collapsed_pair_tail`]).
///
/// SCOPING (batteries-included: no enable-flag, a disable-flag instead). The
/// caller ANDs this with [`GraphNetwork::has_sign_layer`], so a net with no
/// `Sign` node never pays for the extra forward and is BYTE-IDENTICAL to its
/// historical behaviour. `NY_SIGN_SMOOTH_FORWARD=0` restores the old hard-Sign
/// pre-activations (rms floored at 1.0, leading β, every Sign a surrogate) as a
/// diagnostic escape hatch.
///
/// Read ONLY inside the attack lane, so it can never touch any certified /
/// bound-propagation path — and, like β, it only reshapes a NON-certified ascent
/// DIRECTION: every candidate is still concretely re-checked by the UNCHANGED
/// trusted-oracle gate, so it can never change a verdict.
/// #attack-fast-kernels gate (default OFF => byte-identical).
///
/// When set, [`GraphNetwork::attack_point_gradient`] collects its point
/// activations through [`GraphNetwork::collect_node_bounds_attack_point`], which
/// keeps the layer kernels on their im2col/GEMM route instead of the per-MAC
/// pollable scalar contraction a finite deadline forces. Same certified
/// enclosure; ~91x cheaper on conv resnets. Attack-only: this gradient steers a
/// search and every candidate still passes the trusted-ORT + true-f64 admission
/// gates, so a late or wrong gradient can only waste steps.
pub fn attack_fast_kernels_enabled() -> bool {
    std::env::var("NY_ATTACK_POINT_FAST_KERNELS")
        .ok()
        .as_deref()
        == Some("1")
}

pub fn smooth_sign_forward_enabled() -> bool {
    std::env::var("NY_SIGN_SMOOTH_FORWARD").ok().as_deref() != Some("0")
}

/// Is `layer` a UNARY affine map (one live input, output affine in it)?
///
/// Used by [`GraphNetwork::input_linear_sign_windows`] to decide where interval
/// propagation from the input box is EXACT. Deliberately CONSERVATIVE: anything
/// not listed here (every activation, every pooling op, and every BINARY join —
/// a reconverging DAG breaks interval exactness even when each op is affine)
/// answers `false`, which only means the downstream `Sign` keeps today's flat
/// masked-STE window. Attack-only classification; never used for a bound.
fn layer_is_unary_affine(layer: &Layer) -> bool {
    matches!(
        layer,
        Layer::Linear(_)
            | Layer::Conv1d(_)
            | Layer::Conv2d(_)
            | Layer::ConvTranspose1d(_)
            | Layer::ConvTranspose2d(_)
            | Layer::BatchNorm(_)
            | Layer::AveragePool(_)
            | Layer::AddConstant(_)
            | Layer::SubConstant(_)
            | Layer::MulConstant(_)
            | Layer::DivConstant(_)
            | Layer::Transpose(_)
            | Layer::Reshape(_)
            | Layer::Flatten(_)
            | Layer::Squeeze(_)
            | Layer::Unsqueeze(_)
            | Layer::Slice(_)
            | Layer::Pad(_)
    )
}

/// `NY_STE_POOL_WINDOW=0` disables the [`Layer::MaxPool2d`] extension of the
/// per-`Sign` reachability window (#traffic-pool-window) and restores the
/// unary-affine-only classification exactly. Batteries-included: absent ⇒ on.
fn ste_pool_window_enabled() -> bool {
    std::env::var("NY_STE_POOL_WINDOW").ok().as_deref() != Some("0")
}

/// Scale factor `c` for the DEPTH-relative masked-STE window
/// (#traffic-depth-scale-window): on a `Sign` node that has no per-unit
/// reachability window AND whose flat window holds EVERY unit, the window
/// becomes `t_node = c · median_j |z_j|` instead. Default `Some(0.25)`;
/// `NY_STE_DEPTH_SCALE=0` (or any non-positive / unparseable value) restores the
/// flat [`masked_ste_window`] everywhere. Batteries-included: a disable-flag,
/// never an enable-flag.
///
/// # Why a flat `t` cannot be right on BOTH families
///
/// The flat `t = 6` is calibrated in `±1` (Sign-output) units: a BNN's `±1`
/// weights make the pre-activation an even integer, so `t = 6` reads "at most 3
/// flips away". That calibration survives depth on traffic net-1, whose Sign
/// layers feed each other directly. It does NOT survive a `BatchNorm`: net-2 and
/// net-3 put a BN in front of EVERY `Sign`, and BN rescales the pre-activation by
/// its per-channel `γ/σ` to `O(1)`. Measured at the box centre, eps 1, first
/// backward pass (`NY_STE_MASK_DIAG=1`):
///
/// | net / row | Sign node | units | mean\|z\| | held by flat `t=6` |
/// |---|---|---|---|---|
/// | net-1 `30_7573`  | conv2d_21 | 12544 | 268.8 | 1752 (14.0%, per-unit window) |
/// | net-1 `30_7573`  | dense_10  | 23328 | 7.64  | 10571 (45.3%) |
/// | net-2 `48_10645` | conv2d_4  | 15488 | 0.60  | 1794 (11.6%, per-unit window) |
/// | net-2 `48_10645` | conv2d_5  | 5184  | 0.77  | **5184 (100%)** |
/// | net-2 `48_10645` | dense_1   | 3136  | 0.71  | **3136 (100%)** |
/// | net-2 `48_10645` | dense_2   | 256   | 0.82  | **256 (100%)** |
/// | net-3 `64_9022`  | conv2d_13 | 28800 | 0.41  | 5109 (17.7%, per-unit window) |
/// | net-3 `64_9022`  | conv2d_14 | 10816 | 0.73  | **10816 (100%)** |
/// | net-3 `64_9022`  | dense_7   | 1600  | 0.86  | **1600 (100%)** |
/// | net-3 `64_9022`  | dense_8   | 1024  | 0.66  | **1024 (100%)** |
///
/// `#traffic-pool-window` repaired the FIRST `Sign` of each net (the three
/// per-unit rows above). Every deeper `Sign` of net-2/net-3 still holds EVERY
/// unit — which is not a mask at all, it is the plain unmasked STE that was
/// measured to plateau ~1200 margin units short on net-1. The scale gap is ~10x
/// (net-1 dense mean\|z\| 7.64 vs net-2/net-3 ~0.7), i.e. the same single global
/// constant is 0.79x the layer scale on one family and 8.6x on the other.
///
/// # Why the trigger is the REALISED MASK, not the layer type
///
/// "The flat window holds every unit" is a decidable, per-point statement that
/// the mask has degenerated to the identity — exactly the condition this window
/// exists to prevent. Gating on it makes the fallback IMPOSSIBLE to reach on any
/// node whose flat window still discriminates, so net-1 (dense layer 45.3% held,
/// first layer 14.0% held) is byte-identical, and no benchmark-specific
/// classification is needed. The three eps-1 rows measured at the official 480 s
/// budget, ungated (`c` applied to every non-per-unit node):
///
/// | `c` | net-2 `48_10645` | net-3 `64_10645` | net-3 `64_9022` |
/// |---|---|---|---|
/// | flat `t=6` | timeout 476 s | timeout 476 s | timeout 476 s |
/// | 2.0  | timeout 475 s | — | — |
/// | 1.3  | timeout 475 s | — | — |
/// | 0.9  | timeout 475 s | **sat 15.4 s** | **sat 110.6 s** |
/// | 0.5  | timeout 475 s | — | — |
/// | **0.25** | **sat 130.5 s** | **sat 30.4 s** | **sat 62.4 s** |
///
/// The payoff in `c` is a STEP, not a slope: only `c = 0.25` crosses on all
/// three, which is why the shipped value is the measured one rather than the
/// value (`6 / 6.5992 = 0.909`) that would reproduce net-1's own calibration.
///
/// ATTACK DIRECTION ONLY: this chooses which units carry a NON-certified
/// surrogate slope. Every candidate is still re-checked by the unchanged
/// `gate_sat_with_trusted_oracle`, so no value here can make ny unsound.
fn ste_depth_scale() -> Option<f32> {
    static C: std::sync::OnceLock<Option<f32>> = std::sync::OnceLock::new();
    *C.get_or_init(|| {
        match std::env::var("NY_STE_DEPTH_SCALE") {
            Ok(v) => v.parse::<f32>().ok(),
            Err(_) => Some(DEFAULT_STE_DEPTH_SCALE),
        }
        .filter(|c| *c > 0.0 && c.is_finite())
    })
}

/// Shipped [`ste_depth_scale`] factor — the only value measured to cross on all
/// three traffic_signs eps-1 rows (see the table there).
const DEFAULT_STE_DEPTH_SCALE: f32 = 0.25;

/// The masked straight-through surrogate slopes for ONE `Sign` node: `1` where
/// the unit is inside its reachability window, `0` outside.
///
/// Three window sources, in priority order:
/// 1. `per_unit` — the EXACT (or `MaxPool`-enclosing) per-unit radius from
///    [`GraphNetwork::input_linear_sign_windows`], when the attack lane installed
///    one for this node and its length matches.
/// 2. `depth_scale = Some(c)` AND the flat window holds EVERY unit — the flat
///    window is provably INERT here, so fall back to `c · median|z|`
///    (#traffic-depth-scale-window).
/// 3. the flat `±1`-calibrated `window` otherwise.
///
/// Pure, total, and independent of every environment read, so the priority order
/// is unit-testable. ATTACK DIRECTION ONLY.
fn masked_ste_slopes(
    z: &[f32],
    window: f32,
    per_unit: Option<&[f32]>,
    depth_scale: Option<f32>,
) -> Vec<f32> {
    if let Some(t) = per_unit.filter(|t| t.len() == z.len()) {
        return z
            .iter()
            .zip(t.iter())
            .map(|(&zj, &tj)| if zj.abs() <= tj { 1.0 } else { 0.0 })
            .collect();
    }
    let t = match depth_scale {
        // "Holds every unit" == the mask is the identity == the surrogate has
        // silently degenerated to the plain UNMASKED STE that this window exists
        // to replace. Deciding it from the REALISED mask rather than the layer
        // type keeps every node whose flat window still discriminates
        // byte-identical, with no benchmark-specific classification.
        Some(c) if z.iter().all(|zj| zj.abs() <= window) => {
            let med = median_abs(z);
            if med > 0.0 {
                c * med
            } else {
                window
            }
        }
        _ => window,
    };
    z.iter()
        .map(|&zj| if zj.abs() <= t { 1.0 } else { 0.0 })
        .collect()
}

/// Median of `|z_j|`, via `select_nth_unstable` (O(n), no full sort) so the
/// depth-relative window costs one linear pass per `Sign` node per backward.
/// Returns `0.0` for an empty slice, which the caller treats as "no scale
/// information" and falls back to the flat window.
fn median_abs(z: &[f32]) -> f32 {
    if z.is_empty() {
        return 0.0;
    }
    let mut abs: Vec<f32> = z.iter().map(|zj| zj.abs()).collect();
    let mid = abs.len() / 2;
    let (_, med, _) = abs.select_nth_unstable_by(mid, f32::total_cmp);
    *med
}

/// `NY_STE_MASK_DIAG=1`: print, ONCE per `Sign` node, how many units that node's
/// masked-STE window actually HOLDS (non-zero surrogate slope) on the first
/// backward pass, together with the node's `mean|z|` and `rms`.
///
/// This is the diagnostic that distinguishes the two failure modes of a flat
/// window: "holds ~0 units" (window far too small — the gradient is identically
/// zero) and "holds ALL units" (window far too large — the mask is inert and the
/// surrogate degenerates to a plain unmasked STE). Pure `println!`; absent flag ⇒
/// no work beyond one `OnceLock` read, and never consulted for a bound.
fn ste_mask_diag(node_name: &str, z: &[f32], slopes: &[f32], rms: f32) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var("NY_STE_MASK_DIAG").ok().as_deref() == Some("1")) {
        return;
    }
    static SEEN: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<String>>> =
        std::sync::OnceLock::new();
    let seen = SEEN.get_or_init(|| std::sync::Mutex::new(std::collections::HashSet::new()));
    let Ok(mut seen) = seen.lock() else { return };
    if !seen.insert(node_name.to_string()) {
        return;
    }
    let held = slopes.iter().filter(|s| **s != 0.0).count();
    let mean_abs = z.iter().map(|zj| zj.abs() as f64).sum::<f64>() / z.len().max(1) as f64;
    let mut abs: Vec<f32> = z.iter().map(|zj| zj.abs()).collect();
    abs.sort_by(f32::total_cmp);
    let med = abs.get(abs.len() / 2).copied().unwrap_or(0.0);
    println!(
        "#ste-mask-diag node={node_name} units={} held={held} ({:.1}%) mean|z|={mean_abs:.4} \
         median|z|={med:.4} rms={rms:.4}",
        z.len(),
        100.0 * held as f64 / z.len().max(1) as f64,
    );
}

/// Does interval propagation to this layer's output stay a VALID per-unit
/// ENCLOSURE of what the input box can reach — exactly (unary affine) or as an
/// over-approximation (`MaxPool2d`)?
///
/// `MaxPool2d` is monotone, so `[max_i l_i, max_i h_i]` always CONTAINS the true
/// range of `max_i z_i`; it stops being tight only because the pooled `z_i` share
/// input pixels, so the componentwise minimum need not be jointly attainable.
/// For an attack MASK that error has exactly one direction: the window can only
/// grow, so the masked set is a SUPERSET of the truly flippable units and no
/// reachable unit is ever dropped from the ascent direction. That is the safe
/// side (dropping reachable units is what kills the gradient — see
/// [`GraphNetwork::input_linear_sign_windows`]).
///
/// Attack-only classification; never used for a bound.
fn layer_is_interval_enclosed(layer: &Layer) -> bool {
    layer_is_unary_affine(layer)
        || (ste_pool_window_enabled() && matches!(layer, Layer::MaxPool2d(_)))
}

impl GraphNetwork {
    /// Whitelist gate: the point back-prop only runs when EVERY node is in the
    /// exact-gradient fragment `{Conv2d, Linear, ReLU, Add, AveragePool, Flatten,
    /// Reshape}` plus the affine constant-arithmetic ops below. Mirrors
    /// `layer_supports_exact_gradient` in `ny-cli graph_pgd_exact.rs`. Any other
    /// layer → the caller gets `Ok(None)` and falls back to its slower gradient
    /// (certified CROWN / SPSA).
    /// Public, ZERO-COST view of the [`Self::point_vjp_supported_fragment`] gate
    /// below (#deadlane): `false` means [`Self::attack_point_gradient`] will
    /// return `Ok(None)` for EVERY point on this net, no matter the budget.
    ///
    /// Purely static — a scan of node layer types, reachable at graph-load time
    /// with no bounds, no forward pass and no wall clock. Callers use it to
    /// decline a gradient-driven attack lane BEFORE it is handed budget, instead
    /// of discovering the same answer inside the innermost step loop after the
    /// graph has been re-loaded and a trusted-ORT forward already run (measured
    /// on vit_2023: 87.8 s granted, ~0.05 s used, "exact gradient unavailable for
    /// this net", then `unknown`).
    pub fn supports_attack_point_gradient(&self) -> bool {
        !self.nodes.is_empty() && self.point_vjp_supported_fragment()
    }

    /// Does this graph contain a [`Layer::Sign`] node — i.e. is it a binarized
    /// (BNN) network?
    ///
    /// This is the SCOPING predicate for every BNN-specific attack behaviour, so
    /// those behaviours need no environment flag and apply to exactly the nets
    /// they were measured on (see [`smooth_sign_forward_enabled`] and the
    /// caller-side attack-budget policy). A `Sign` net is qualitatively different
    /// from every other benchmark family: `Sign` is piecewise constant, so BaB
    /// cannot make progress and the surrogate-gradient attack is the ONLY lane
    /// that can produce a verdict. Cheap structural scan; NON-certified (it only
    /// selects an attack policy — acceptance is always the unchanged trusted gate).
    pub fn has_sign_layer(&self) -> bool {
        self.nodes
            .values()
            .any(|node| matches!(node.layer, Layer::Sign(_)))
    }

    /// ATTACK-ONLY (#traffic-input-linear-window): per-unit reachability windows
    /// for every `Sign` node reached from the network input through a chain of
    /// interval-ENCLOSING ops (unary affine, plus monotone `MaxPool2d` —
    /// [`layer_is_interval_enclosed`]), computed over the real search box
    /// `input_bounds`. `Sign` nodes that do not qualify are ABSENT from the map
    /// and keep the flat [`masked_ste_window`] default, so a net without such a
    /// node gets an empty map and is byte-identical to today.
    ///
    /// # Why the window must be per-layer, and why this value is the right one
    ///
    /// The masked straight-through estimator zeroes a `Sign`'s surrogate slope
    /// wherever `|z| > t`, because a `Sign` that far from its threshold cannot be
    /// flipped by any admissible input move and keeping it in the gradient only
    /// dilutes the direction. The default `t = 6` is calibrated in `±1` units:
    /// with `±1` weights and `±1`-valued (Sign-output) inputs a single input flip
    /// moves `z` by `2`, so `t = 2k` means "at most `k` flips away".
    ///
    /// The FIRST `Sign` of a BNN does not see `±1` inputs — it sees a convolution
    /// of RAW PIXELS. Measured on traffic_signs net-1 (`model_30_idx_12375_eps_3`,
    /// 30x30x3 in `0..255`, eps 3): `mean|z1| = 1356` while the per-unit
    /// reachability radius is `40.5 .. 81`. A flat `t = 6` therefore holds a MEDIAN
    /// OF 3 of 12544 units (empty in 157 of 2743 backward passes) and the attack
    /// dies of an identically-zero gradient after ~10 of its 120 steps.
    ///
    /// For a `Sign` whose pre-activation is an affine function of the input along a
    /// UNARY chain, plain interval propagation over a box is EXACT (each input
    /// coordinate enters once and independently), so `(u_j - l_j)/2` is the exact
    /// per-unit reachability radius: unit `j` can flip inside the box iff its
    /// pre-activation is within that radius of `0`. That is the window's own
    /// definition, with no calibration constant at all. Binary joins (`Add`,
    /// `Concat`, `MulBinary`, …) are EXCLUDED because a reconverging DAG makes
    /// interval propagation inexact and the node keeps the `±1`-calibrated
    /// default.
    ///
    /// Measured (external deterministic prototype, box centre, both hard-tail
    /// net-1 rows; a counterexample needs margin `>= 0`):
    ///
    /// | layer-1 window | `11379_eps_3` | `12375_eps_3` |
    /// |---|---|---|
    /// | `6` (flat default) | miss, best `-628` | ZEROGRAD at it 6, best `-368` |
    /// | `12`               | hit at it 10  | miss, best `-68` |
    /// | per-unit radius    | **hit at it 3** | **hit at it 3** |
    ///
    /// # Why `MaxPool2d` is in the chain too (#traffic-pool-window)
    ///
    /// traffic net-2/net-3 put a `MaxPool` **and a `BatchNorm`** between the raw
    /// pixels and their first `Sign` (`Conv -> MaxPool -> BN -> Sign`), so
    /// excluding pooling left those nets with the flat `t = 6` — and the BN
    /// rescales the pre-activation to `O(1)`, which makes the flat window the
    /// OPPOSITE failure from net-1. Measured at the box centre, eps 1:
    ///
    /// | net / row | units | mean abs z | per-unit radius (med) | held by `t=6` | held by radius |
    /// |---|---|---|---|---|---|
    /// | net-1 `30_7573`   | 12544 | 268.8 | 27.00 | 420 | 1752 |
    /// | net-2 `48_10645`  | 15488 | 0.6   | 0.10  | **15488 (ALL)** | 1794 |
    /// | net-3 `64_9022`   | 28800 | 0.4   | 0.10  | **28800 (ALL)** | 5109 |
    ///
    /// A window that holds EVERY unit is not a mask at all — it degenerates to the
    /// plain unmasked STE the table above measures plateauing ~1200 margin units
    /// short. The same probe at eps 5 holds 20081/28800 by radius, which is why
    /// only the eps-1 rows of these two nets are the hard tail.
    ///
    /// `MaxPool2d` is monotone, so `[max_i l_i, max_i h_i]` always CONTAINS the
    /// true range of `max_i z_i`. It is no longer TIGHT (the pooled units share
    /// pixels, so the componentwise minimum need not be jointly attainable), but
    /// the looseness has one direction only: the window can only GROW, so the mask
    /// stays a SUPERSET of the truly flippable units. Dropping reachable units is
    /// the failure mode that kills the gradient; adding unreachable ones only
    /// dilutes it, and even then never past the flat default's "hold everything".
    /// `NY_STE_POOL_WINDOW=0` restores the unary-affine-only classification.
    ///
    /// ATTACK DIRECTION ONLY. This selects which units contribute to a
    /// NON-certified ascent direction; every candidate is still concretely
    /// re-checked by the unchanged `gate_sat_with_trusted_oracle`, so no window
    /// here can make ny unsound.
    pub fn input_linear_sign_windows(
        &self,
        input_bounds: &BoundedTensor,
    ) -> HashMap<String, Vec<f32>> {
        use crate::network::core::graph::ibp::dispatch::{
            dispatch_ibp_resolved, resolve_node_inputs, ResolvedInputs,
        };

        let mut windows: HashMap<String, Vec<f32>> = HashMap::new();
        let exec_order: Vec<String> = match self.exec_order() {
            Ok(order) => order.to_vec(),
            Err(_) => return windows,
        };
        let mut cache: HashMap<String, BoundedTensor> = HashMap::new();
        // `true` == interval propagation to this node is a VALID per-unit
        // enclosure of what the input box reaches: exact along a unary-affine
        // chain, an over-approximation once a monotone `MaxPool2d` joins several
        // affine units (see [`layer_is_interval_enclosed`]). The network input
        // itself is exact by definition, which is why the per-input probe below
        // short-circuits on `NETWORK_INPUT`; anything not yet classified answers
        // `false`.
        let mut enclosed: HashMap<String, bool> = HashMap::new();

        for node_name in &exec_order {
            let Some(node) = self.nodes.get(node_name) else {
                return windows;
            };
            // Enclosing-interval iff EVERY input is itself enclosing AND there is
            // at most one graph input (a binary join can reconverge, which makes
            // interval propagation inexact even for affine layers).
            let inputs_enclosed = node.inputs.len() <= 1
                && node.inputs.iter().all(|name| {
                    name.as_str() == NETWORK_INPUT
                        || enclosed.get(name.as_str()).copied().unwrap_or(false)
                });

            let resolved = match resolve_node_inputs(node, node_name, &mut |name| {
                Ok(self.bounds_ref(name, input_bounds, &cache)?.clone())
            }) {
                Ok(r) => r,
                Err(_) => return windows, // partial map: callers fall back per node
            };

            if inputs_enclosed && matches!(&node.layer, Layer::Sign(_)) {
                if let ResolvedInputs::Unary(z) = &resolved {
                    let lo = z.lower();
                    let hi = z.upper();
                    windows.insert(
                        node_name.clone(),
                        lo.iter()
                            .zip(hi.iter())
                            .map(|(&l, &h)| ((h - l) * 0.5).max(0.0))
                            .collect(),
                    );
                }
            }

            let out = match dispatch_ibp_resolved(node, node_name, resolved) {
                Ok(b) => b,
                Err(_) => return windows,
            };
            enclosed.insert(
                node_name.clone(),
                inputs_enclosed && layer_is_interval_enclosed(&node.layer),
            );
            cache.insert(node_name.clone(), out);
        }
        windows
    }

    /// ATTACK-ONLY: the PRE-`Softmax` logits at the concrete point `x`, or `None`
    /// when the output node is not a `Softmax` (or the forward is unsupported).
    ///
    /// # Why the attack needs these
    ///
    /// The traffic_signs BNNs end in `Softmax`, and their logit margins are large
    /// integers, so the f32 output SATURATES: measured on
    /// `model_30_idx_12375_eps_3`, ORT returns `p_true = 1` and all 42 other
    /// outputs EXACTLY `0`, at every one of 697 attack iterates. Every disjunct's
    /// margin is then exactly `-1`, the guidance signal is a two-valued step
    /// function, and the caller's `max_by` tie-break degenerates to "the LAST
    /// clause" — targeting a class of logit rank 31/42 instead of the argmax.
    ///
    /// `Softmax` is strictly monotone, so ranking disjuncts by logits is the SAME
    /// ranking whenever the output space carries any information, and is defined
    /// when it does not. Callers use this only to break EXACT ties (see
    /// `margin_subgradient_row`), so nothing changes on an unsaturated net.
    ///
    /// Non-certified: a plain (non-sound) per-node forward, used only to pick an
    /// attack target. Acceptance remains the unchanged trusted-oracle gate.
    /// When supplied, `deadline` is polled before and after every node dispatch;
    /// an expired walk returns `None` and can never extend its caller's authority.
    pub fn attack_pre_softmax_logits(
        &self,
        x: &ArrayD<f32>,
        deadline: Option<Instant>,
    ) -> Option<Vec<f64>> {
        use crate::network::core::graph::ibp::dispatch::{
            dispatch_ibp_resolved, resolve_node_inputs,
        };

        if self.nodes.is_empty() || deadline.is_some_and(|limit| Instant::now() >= limit) {
            return None;
        }
        let output_name = self.output_name().to_string();
        let output_node = self.nodes.get(&output_name)?;
        if !matches!(output_node.layer, Layer::Softmax(_)) {
            return None;
        }
        let logits_name = output_node.inputs.first()?.clone();

        let exec_order: Vec<String> = self.exec_order().ok()?.to_vec();
        let input_bounds = BoundedTensor::concrete(x.clone()).ok()?;
        let mut cache: HashMap<String, BoundedTensor> = HashMap::new();
        for node_name in &exec_order {
            if deadline.is_some_and(|limit| Instant::now() >= limit) {
                return None;
            }
            let node = self.nodes.get(node_name)?;
            let resolved = resolve_node_inputs(node, node_name, &mut |name| {
                Ok(self.bounds_ref(name, &input_bounds, &cache)?.clone())
            })
            .ok()?;
            let out = dispatch_ibp_resolved(node, node_name, resolved).ok()?;
            if deadline.is_some_and(|limit| Instant::now() >= limit) {
                return None;
            }
            // Collapse to the interval CENTER (mirrors `smooth_sign_preactivations`)
            // so per-node widening cannot accumulate through the DAG.
            let centered: ArrayD<f32> = (out.lower() + out.upper()) * 0.5;
            if *node_name == logits_name {
                return Some(centered.iter().map(|&v| v as f64).collect());
            }
            cache.insert(node_name.clone(), BoundedTensor::concrete(centered).ok()?);
        }
        None
    }

    fn point_vjp_supported_fragment(&self) -> bool {
        self.node_names().iter().all(|name| {
            self.node(name).is_some_and(|node| {
                matches!(
                    node.layer(),
                    Layer::Conv2d(_)
                        | Layer::Linear(_)
                        | Layer::ReLU(_)
                        | Layer::Add(_)
                        | Layer::AveragePool(_)
                        // MaxPool2d: EXACT at a point. The degenerate box [x, x]
                        // makes every pooling window have a definite winner (the
                        // argmax input, l == u), so dispatch_backward_layer's
                        // generic unary arm routes propagate_crown_backward →
                        // MaxPool2dLayer::propagate_linear_with_bounds, which
                        // routes the gradient through that winner (exact
                        // route-to-max — see max.rs "If one input definitely
                        // dominates … route gradient through it"). Needed for the
                        // deeper traffic_signs BNNs (net-2/net-3) which stack
                        // MaxPool + BatchNorm between the Sign conv blocks.
                        | Layer::MaxPool2d(_)
                        | Layer::Flatten(_)
                        | Layer::Reshape(_)
                        // Shape/linear plumbing exact at a point: Transpose is a
                        // pure permutation and MatMul(live, const-weight) is affine
                        // (the dense head of the traffic_signs BNNs).
                        // dispatch_backward_layer routes both through the exact
                        // linear transpose.
                        | Layer::Transpose(_)
                        | Layer::MatMul(_)
                        // GAN/deconv fragment (cgan): ConvTranspose + BatchNorm.
                        // dispatch_backward_layer handles all three; attack-only.
                        | Layer::ConvTranspose1d(_)
                        | Layer::ConvTranspose2d(_)
                        | Layer::BatchNorm(_)
                        // Affine constant arithmetic (cora_2024 MLP fragment:
                        // unfused Gemm = MatMul + AddConstant bias, and the
                        // mnist Div-by-constant normalization; d/dx (x/c) = 1/c).
                        // All exact at a point — dispatch_backward_layer routes
                        // them through the plain unary CROWN backward, which is
                        // the exact affine transpose for these layers.
                        | Layer::AddConstant(_)
                        | Layer::SubConstant(_)
                        | Layer::MulConstant(_)
                        | Layer::DivConstant(_)
                        // Sign / binarized nets (traffic_signs_recognition BNNs)
                        // and the trailing Softmax classifier head.
                        // ATTACK-ONLY, NON-EXACT: unlike every other layer above
                        // (each exact at a point), Sign's true point-Jacobian is
                        // 0 a.e. and Softmax's certified relaxation carries a
                        // vanishing gradient. The backward loop special-cases BOTH
                        // — Sign with a soft-sign surrogate slope, Softmax as a
                        // monotone identity pass-through (the logit-margin
                        // direction, matching the external bnn_falsifier prototype)
                        // — so PGD can descend. Each direction is a heuristic, never
                        // a certified bound: every candidate it yields is still
                        // concretely re-checked by the unchanged trusted-oracle
                        // gate. See the Sign / Softmax arms in attack_point_gradient.
                        | Layer::Sign(_)
                        | Layer::Softmax(_)
                )
            })
        })
    }

    /// ATTACK-ONLY (smooth mode): is `name` the TAIL Sign of a collapsed
    /// `Sign(Sign(x) + c)` pair (`|c| < 1`, so the hard net satisfies
    /// `sign(sign(x)+c) == sign(x)` and the pair is ONE logical sign)? The
    /// traffic_signs BNNs store every activation as such a pair
    /// (`Sign → Add(0.1) → Sign`: 8 ONNX Sign nodes = 4 logical signs). The
    /// external `bnn_falsifier` prototype models each pair as a SINGLE soft-sign on
    /// the FIRST sign's real pre-activation `x`; applying the smooth surrogate to
    /// BOTH signs instead stacks two saturating `tanh` Jacobians per pair and badly
    /// distorts the attack gradient (measured: the hard-tail target stays
    /// uncracked). So in smooth mode the tail sign is treated as an identity
    /// pass-through (forward AND backward), leaving exactly one surrogate per pair.
    ///
    /// Detection: walk the input cone back through the constant-affine / shape ops
    /// that can sit between the two signs of a pair (`Add`, `Add/Sub/Mul/DivConstant`,
    /// `Transpose`, `Reshape`, `Flatten`); if a [`Layer::Sign`] is reached before any
    /// weight/normalisation layer, `name` is a pair tail. A non-tail (primary) sign
    /// reaches a Conv/Linear/MatMul/BatchNorm/network-input first and returns
    /// `false`. NON-certified, attack-direction only.
    fn sign_is_collapsed_pair_tail(&self, name: &str) -> bool {
        self.collapsed_pair_primary(name).is_some()
    }

    /// ATTACK-ONLY (smooth mode): if `name` is the TAIL Sign of a collapsed
    /// `Sign(Sign(x) + c)` pair, return the node name of that pair's PRIMARY (first)
    /// Sign; otherwise `None`. See [`Self::sign_is_collapsed_pair_tail`] for what a
    /// collapsed pair is and why the pair must carry exactly one surrogate.
    ///
    /// Naming the primary — not just detecting the tail — is what lets the smooth
    /// forward drop the pair's INTERIOR constant shift. The traffic_signs BNNs
    /// encode each activation as `Sign → Add(0.1) → Sign`. Treating the tail as a
    /// plain identity pass-through leaves that `+0.1` on the wire, so the next conv
    /// sees `tanh(β·z/rms) + 0.1` where the prototype's collapsed `dsign` yields
    /// `tanh(β·z/rms)`. That bias is NOT benign: with ±1 binary conv weights it adds
    /// `0.1·Σw` per output — a per-channel offset comparable to the signal itself.
    /// Measured directly in the prototype (`plus01_ablation`, 4-way A/B): with the
    /// shift the hard-tail target `model_48_idx_10645_eps_1` stalls at margin
    /// −0.9999 in both f64 and f32, and without it cracks at iteration 118 in both —
    /// i.e. the shift alone is decisive and f32 rounding is irrelevant. So the smooth
    /// forward emits the primary's surrogate value AT the tail, collapsing the whole
    /// chain to one soft sign. NON-certified, attack-direction only.
    fn collapsed_pair_primary(&self, name: &str) -> Option<String> {
        let node = self.nodes.get(name)?;
        if !matches!(node.layer, Layer::Sign(_)) {
            return None;
        }
        let mut frontier: Vec<String> = node.inputs.clone();
        // A pair has only Add(+ maybe a Transpose) between its two signs, so a
        // shallow bounded walk (depth 4) is ample and keeps this cheap per sign.
        for _ in 0..4 {
            let mut next: Vec<String> = Vec::new();
            for inp in frontier {
                if inp == NETWORK_INPUT {
                    continue;
                }
                let Some(n) = self.nodes.get(&inp) else {
                    continue; // constant / initializer input (not a graph node) — skip
                };
                match &n.layer {
                    Layer::Sign(_) => return Some(inp),
                    // Constant-affine + shape ops that can sit between paired signs.
                    Layer::Add(_)
                    | Layer::AddConstant(_)
                    | Layer::SubConstant(_)
                    | Layer::MulConstant(_)
                    | Layer::DivConstant(_)
                    | Layer::Transpose(_)
                    | Layer::Reshape(_)
                    | Layer::Flatten(_) => next.extend(n.inputs.iter().cloned()),
                    // Any weight / normalisation / other op ⇒ this is a PRIMARY sign.
                    _ => {}
                }
            }
            if next.is_empty() {
                break;
            }
            frontier = next;
        }
        None
    }

    /// ATTACK-ONLY smooth forward (opt-in via [`smooth_sign_forward_enabled`]):
    /// evaluate the whole graph at the concrete point `x`, but replace each
    /// PRIMARY [`Layer::Sign`] with the smooth surrogate value `tanh(β·z/rms)` (rms
    /// UNFLOORED, `= sqrt(mean_j z_j²) + 1e-9`) and each collapsed-pair TAIL Sign
    /// with the PRIMARY sign's surrogate value, which also drops the pair's interior
    /// constant shift (see [`Self::collapsed_pair_primary`]).
    /// Returns, for every PRIMARY Sign node, the pre-activation vector `z` (its
    /// input value BEFORE the surrogate, flattened row-major) under THIS smooth
    /// forward, keyed by node name.
    ///
    /// This is the exact analogue of the external `bnn_falsifier` prototype's
    /// `forward_soft`: each primary Sign emits a value in `(-1, 1)` (not `±1`), so
    /// every downstream Sign's pre-activation differs from the hard-forward's, and
    /// evaluating the surrogate slope THERE is what makes PGD descend on the
    /// hard-tail instances. Every node output is collapsed to its interval CENTER
    /// before caching so the non-sound per-node widening (esp. BatchNorm — net-2/3
    /// are not DAG-lowerable) cannot be amplified by the deep DAG.
    ///
    /// NON-CERTIFIED / NON-soundness-critical: reuses the plain (non-sound, CPU)
    /// per-node forward [`dispatch_ibp_resolved`] for every non-Sign node purely to
    /// produce an ATTACK gradient direction. Structural failures let the caller
    /// fall back to hard-forward pre-activations. A finite resource refusal is
    /// terminal for this gradient attempt.
    fn smooth_sign_preactivations(
        &self,
        input_bounds: &BoundedTensor,
        beta: f32,
        exec_order: &[String],
        deadline: Option<Instant>,
    ) -> anyhow::Result<HashMap<String, Vec<f32>>> {
        use crate::network::core::graph::ibp::dispatch::{
            dispatch_ibp_resolved, resolve_node_inputs, ResolvedInputs,
        };

        // Concrete (degenerate) value of every node under the smooth forward.
        let mut cache: HashMap<String, BoundedTensor> = HashMap::new();
        // Pre-activation z feeding each PRIMARY Sign node under the smooth forward.
        let mut pre: HashMap<String, Vec<f32>> = HashMap::new();

        for node_name in exec_order {
            if deadline.is_some_and(|d| Instant::now() >= d) {
                return Err(NyError::DeadlineExceeded(
                    "attack smooth-Sign forward deadline exceeded".to_string(),
                )
                .into());
            }
            let node = self.nodes.get(node_name).ok_or_else(|| {
                anyhow::anyhow!("smooth_sign_preactivations: node '{node_name}' not found")
            })?;

            let resolved = resolve_node_inputs(node, node_name, &mut |name| {
                Ok(self.bounds_ref(name, input_bounds, &cache)?.clone())
            })?;

            let out = match (&node.layer, resolved) {
                (Layer::Sign(_), ResolvedInputs::Unary(z_bounds)) => {
                    if let Some(primary) = self.collapsed_pair_primary(node_name) {
                        // Collapsed-pair TAIL sign: emit the PRIMARY sign's surrogate
                        // value directly. The pair's single surrogate lives upstream,
                        // and routing its value through here DROPS the pair's interior
                        // constant shift (`Sign → Add(0.1) → Sign`) instead of leaving
                        // `+0.1` on the wire — the shift is decisive for the hard-tail
                        // instances (see [`Self::collapsed_pair_primary`]). Falls back
                        // to a plain identity pass-through if the primary is missing or
                        // reshaped between the two signs, so this can never error.
                        match cache.get(&primary) {
                            Some(v) if v.lower().shape() == z_bounds.lower().shape() => v.clone(),
                            _ => z_bounds,
                        }
                    } else {
                        // PRIMARY sign: smooth surrogate value tanh(β·z/rms), rms
                        // UNFLOORED (prototype), and record z as its pre-activation.
                        let z_arr: ArrayD<f32> = (z_bounds.lower() + z_bounds.upper()) * 0.5;
                        let z_vec: Vec<f32> = z_arr.iter().copied().collect();
                        let n = z_vec.len().max(1);
                        let mean_sq =
                            z_vec.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / n as f64;
                        let rms = (mean_sq.sqrt() as f32) + 1e-9;
                        let smoothed = z_arr.mapv(|v| (beta * v / rms).tanh());
                        pre.insert(node_name.clone(), z_vec);
                        BoundedTensor::concrete(smoothed)?
                    }
                }
                // Everything else: the plain (non-sound, CPU) per-node forward.
                (_, resolved) => dispatch_ibp_resolved(node, node_name, resolved)?,
            };
            // Collapse each node output to its interval CENTER before caching, so
            // non-sound per-node widening (esp. BatchNorm) cannot amplify through
            // the deep DAG — mirrors `propagate_concrete_point_core`. The Sign arms
            // above are already degenerate, so this is a no-op there.
            let centered: ArrayD<f32> = (out.lower() + out.upper()) * 0.5;
            cache.insert(node_name.clone(), BoundedTensor::concrete(centered)?);
        }
        if deadline.is_some_and(|limit| Instant::now() >= limit) {
            return Err(NyError::DeadlineExceeded(
                "attack smooth-Sign forward deadline exceeded before publication".to_string(),
            )
            .into());
        }
        Ok(pre)
    }

    /// Fast f32 point back-prop: `d(spec_row · network_output) / d(input)` at the
    /// concrete point `x`, reshaped to `x.shape()`.
    ///
    /// - `spec_row` must have shape `(1, num_outputs)` where `num_outputs` is the
    ///   flattened element count of the network output node.
    /// - `engine` is an optional GEMM engine (e.g. GPU); `None` uses CPU.
    /// - `deadline` aborts the pass (returning `Ok(None)`) if exceeded.
    ///
    /// Returns:
    /// - `Ok(Some(grad))` with `grad.shape() == x.shape()` on success,
    /// - `Ok(None)` when the graph is outside the supported fragment, the graph
    ///   is empty, a layer reports `Unsupported`, or the deadline is hit,
    /// - `Err(..)` only on an internal/structural failure.
    ///
    /// This is an ATTACK gradient — it is the exact point-Jacobian but carries no
    /// certified error interval (see the module docs for why it is exact at a
    /// point, and why that is sufficient for an attack).
    pub fn attack_point_gradient(
        &self,
        x: &ArrayD<f32>,
        spec_row: &Array2<f32>,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> anyhow::Result<Option<ArrayD<f32>>> {
        // --- Gate: supported fragment + non-empty graph. -------------------
        if self.nodes.is_empty() {
            return Ok(None);
        }
        if !self.point_vjp_supported_fragment() {
            return Ok(None);
        }
        if deadline.is_some_and(|limit| Instant::now() >= limit) {
            return Ok(None);
        }

        // --- Degenerate input box [x, x]: this is what makes every ReLU mask
        //     exact (see module docs). -----------------------------------------
        let input_bounds = BoundedTensor::concrete(x.clone())?;
        let input_dim = input_bounds.len();

        // --- Node values / pre-activations. At a point every entry is
        //     degenerate (lower == upper == node value); this is BOTH the ReLU
        //     pre-activations AND the concrete node values. --------------------
        // PERF: this reuses the certified (Higham-sound, f64-widened) forward IBP
        // collection for correctness. A plain-f32 forward evaluation at the point
        // would be faster and is a valid later optimization; the sound forward is
        // used here only so the pre-activations/masks match the certified path.
        // #attack-fast-kernels (NY_ATTACK_POINT_FAST_KERNELS=1, default OFF =>
        // byte-identical). A finite deadline forces every Conv2d off im2col+GEMM
        // onto the per-MAC pollable scalar contraction and makes conv2d/bound.rs:308
        // discard the engine, which costs ~91x here: this call measures 8.5 s/step
        // on CIFAR100_resnet_large against the ~93 ms/step this crate's own
        // `point_vjp_batched_resnet.rs:9-12` records for the same call on the same
        // model family. At 8.5 s/step the upfront falsification lane completes ZERO
        // gradient steps in its 4 s slice, and raising that slice is separately
        // refuted (it costs 15 wall-hugger proofs to win 9 counterexamples).
        //
        // The gate keeps the LOOP deadline and drops only the LAYER deadline, so the
        // enclosure is unchanged (the two conv routes are documented as
        // "mathematically identical") and abort granularity coarsens from per-MAC to
        // per-node. Sound here because this function has no verdict authority — see
        // `collect_node_bounds_attack_point`.
        let node_bounds = if attack_fast_kernels_enabled() {
            self.collect_node_bounds_attack_point(&input_bounds, engine, deadline)
        } else {
            self.collect_node_bounds_with_engine_and_deadline(&input_bounds, engine, deadline)
        };
        let Some(node_bounds) = attack_step_or_decline(node_bounds)? else {
            return Ok(None);
        };

        // --- Output node + its flattened dimension. -------------------------
        let output_node_name = self.output_name().to_string();
        let output_bounds = node_bounds.get(&output_node_name).ok_or_else(|| {
            anyhow::anyhow!("attack_point_gradient: output node '{output_node_name}' not found")
        })?;
        let output_dim = output_bounds.len();

        // Spec row must be a single row selecting a linear combination of outputs.
        if spec_row.nrows() != 1 || spec_row.ncols() != output_dim {
            return Err(anyhow::anyhow!(
                "attack_point_gradient: spec_row must be (1, {output_dim}), got {:?}",
                spec_row.shape()
            ));
        }

        // We track exactly ONE cotangent row (the spec row) through the whole
        // backward pass, so the "output dimension" for every accumulation helper
        // (used only to size the zero bias-coefficient matrices routed to the
        // network input) is 1, NOT the network output dimension.
        const SPEC_ROWS: usize = 1;

        // --- Accumulator frontier (indexed, Dense-only). --------------------
        // `new_indexed` keys every exec-order node name + NETWORK_INPUT; the
        // string-keyed API (insert / take / merge via accumulate_*) uses the
        // O(1) indexed storage underneath.
        let exec_order: Vec<String> = self.exec_order()?.to_vec();
        let mut acc = CrownMergeAccumulator::new_indexed(&exec_order);

        // Seed the OUTPUT node with the spec row: lower_a == upper_a == spec_row,
        // zero biases, err = None (default). err = None means the coeff-error
        // carrier second pass inside `dispatch_backward_layer` is skipped.
        let seed = LinearBounds::new(
            spec_row.clone(),
            Array1::zeros(SPEC_ROWS),
            spec_row.clone(),
            Array1::zeros(SPEC_ROWS),
        )?;
        acc.insert(output_node_name.clone(), CrownBounds::Dense(seed));

        let mut input_accumulated = false;

        // --- Smooth-Sign forward, BNN-only (attack direction only). -----------
        // On a net that HAS a `Sign` node, precompute — ONCE per gradient step, at
        // the CURRENT β — the pre-activation z feeding every PRIMARY Sign under a
        // parallel forward in which every primary Sign is the smooth surrogate
        // `tanh(β·z/rms)` and each `Sign(Sign(x)+c)` pair collapses to one sign.
        // The Sign backward arm below reads these instead of the hard-forward's
        // `node_bounds` z (with the matching UNFLOORED `(β/rms)` slope). A net with
        // no Sign node skips this entirely ⇒ empty map ⇒ the whole function is
        // byte-identical to the historical path and pays nothing.
        // NON-certified: a bad/partial map only weakens the ATTACK direction; every
        // candidate is still concretely re-checked by the unchanged trusted gate.
        let smooth_sign = smooth_sign_forward_enabled() && self.has_sign_layer();
        let smooth_pre: HashMap<String, Vec<f32>> = if smooth_sign {
            match self.smooth_sign_preactivations(
                &input_bounds,
                attack_sign_beta(),
                &exec_order,
                deadline,
            ) {
                Ok(map) => map,
                Err(err) if anyhow_attack_resource_refusal(&err) => return Ok(None),
                Err(err) => {
                    tracing::debug!(
                        "attack_point_gradient: smooth-Sign forward failed ({err}); \
                         falling back to hard-Sign pre-activations"
                    );
                    HashMap::new()
                }
            }
        } else {
            HashMap::new()
        };

        // --- Reverse (reverse-topological) walk. Every consumer of a node is
        //     processed before the node itself, so all cotangent contributions
        //     to a node have already landed when we take it. This is what makes
        //     residual Add fan-in summation correct (see the note below). ------
        for node_name in exec_order.iter().rev() {
            let node_name: &str = node_name.as_str();
            if deadline.is_some_and(|d| Instant::now() >= d) {
                return Ok(None);
            }

            // Move out this node's accumulated cotangent. `None` = the node has
            // no consumers on the spec's cone (its gradient is zero) — skip it.
            let Some(node_cb) =
                attack_step_or_decline(acc.take_with_deadline(node_name, deadline))?
            else {
                return Ok(None);
            };
            let node_cb = match node_cb {
                Some(bounds) => bounds,
                None => continue,
            };

            let node = self.nodes.get(node_name).ok_or_else(|| {
                anyhow::anyhow!("attack_point_gradient: node '{node_name}' not found")
            })?;

            // Dense throughout: the seed and every accumulation helper are
            // Dense-only. Refuse an unexpected Patches carrier without opening
            // an unpolled materialization. We then DROP any accumulated coefficient
            // error: this is an attack gradient, so the small f64-merge roundoff
            // carrier is irrelevant and dropping it keeps `dispatch_backward_layer`
            // on its plain (single-pass) path.
            let Some(mut node_lb) = take_attack_dense_carrier(node_cb) else {
                tracing::debug!(
                    "attack_point_gradient: unexpected Patches frontier at '{node_name}'; \
                     declining Dense-only point-VJP route"
                );
                return Ok(None);
            };
            node_lb.lower_a_err = None;
            node_lb.upper_a_err = None;

            // First input drives the shared pre-activation lookup (matches
            // propagation.rs). For network-input-fed nodes, the pre-activation is
            // the concrete input box.
            let first_input = node
                .inputs
                .first()
                .map(String::as_str)
                .unwrap_or(NETWORK_INPUT);
            let pre_activation: &BoundedTensor = if first_input == NETWORK_INPUT {
                &input_bounds
            } else {
                node_bounds.get(first_input).ok_or_else(|| {
                    anyhow::anyhow!(
                        "attack_point_gradient: pre-activation for '{first_input}' not found"
                    )
                })?
            };

            // === ReLU: site-specific (handled by dispatch_relu_backward, NOT by
            //     dispatch_backward_layer). At a point the pre-activation box is
            //     degenerate, so `propagate_crown_backward` produces the EXACT
            //     0/1 mask with zero intercept. ===
            if matches!(&node.layer, Layer::ReLU(_)) {
                let Some(relu_result) = attack_step_or_decline(dispatch_relu_backward(
                    node,
                    &node_lb,
                    pre_activation,
                    node_name,
                    "attack_point_gradient",
                    None, // alpha_lower: heuristic (exact-at-point) mask
                    None, // alpha_upper
                ))?
                else {
                    return Ok(None);
                };
                match relu_result {
                    NodeDispatchResult::SingleDense(bounds) => {
                        if !attack_publication_completed(
                            self.accumulate_dense_bounds_to_input_with_deadline(
                                first_input,
                                *bounds,
                                &mut acc,
                                SPEC_ROWS,
                                input_dim,
                                &mut input_accumulated,
                                deadline,
                            ),
                        )? {
                            return Ok(None);
                        }
                    }
                    // A ReLU that cannot dispatch exactly at the point (should not
                    // happen for the whitelist) → bail to the caller's fallback.
                    NodeDispatchResult::IbpFallback(_) => return Ok(None),
                }
                continue;
            }

            // === Sign: ATTACK-ONLY soft-sign surrogate (NON-EXACT). ===
            // Unlike ReLU and every other whitelist layer — which are EXACT at a
            // point — Sign's true point-Jacobian is 0 almost everywhere (the
            // certified relaxation `propagate_crown_backward` gives slope 0), so
            // it carries NO gradient signal and PGD cannot descend on a BNN. To
            // crack binarized nets we instead apply the smooth surrogate
            // `tanh(β·z)`'s local diagonal Jacobian at the pre-activation point z:
            // `slope_j = β·(1 − tanh²(β·z_j))` with β = the thread-local
            // `attack_sign_beta()` (default DEFAULT_ATTACK_SIGN_BETA = 10.0).
            // Sign is element-wise, so input-dim == output-dim and this is a pure
            // diagonal column-scale of the incoming cotangent — the same
            // SingleDense accumulation as the ReLU arm.
            //
            // SOUNDNESS: this is a surrogate ATTACK direction, NOT a certified
            // bound. Every candidate the attack proposes is still validated by
            // the UNCHANGED trusted-oracle gate (real ORT + sound f64 + zero-tol
            // in-box), so a wrong/surrogate direction can only make the attack
            // weaker or stronger — never a wrong verdict. Do NOT route Sign
            // through `propagate_crown_backward` (that is the certified slope-0
            // relaxation, not this attack surrogate).
            if matches!(&node.layer, Layer::Sign(_)) {
                // Collapsed `Sign(Sign(x)+c)` pair TAIL (smooth mode only): identity
                // pass-through (slope 1) — the pair's single surrogate lives on the
                // PRIMARY sign upstream. Applying the surrogate to BOTH signs stacks
                // two tanh Jacobians per pair and distorts the gradient. Default OFF
                // ⇒ every Sign is a surrogate, byte-identical to today.
                if smooth_sign && self.sign_is_collapsed_pair_tail(node_name) {
                    if !attack_publication_completed(
                        self.accumulate_dense_bounds_to_input_with_deadline(
                            first_input,
                            node_lb,
                            &mut acc,
                            SPEC_ROWS,
                            input_dim,
                            &mut input_accumulated,
                            deadline,
                        ),
                    )? {
                        return Ok(None);
                    }
                    continue;
                }
                // β = thread-local soft-sign sharpness (default
                // DEFAULT_ATTACK_SIGN_BETA = 10.0, matching pgd_attack's
                // SMOOTH_SIGN_BETA). An attack loop MAY ramp this across
                // restarts/steps (smooth/exploratory ≈2 → sharp/decisive ≈20) to
                // crack tight boxes a fixed β gets stuck on; when unset it is the
                // historical constant, so every non-ramping caller is
                // byte-identical. ATTACK-ONLY, soundness-neutral (see below).
                let beta = attack_sign_beta();
                // Pre-activation point values z_j. On a BNN (default): the SMOOTH
                // forward's pre-activations for THIS Sign, precomputed in
                // `smooth_pre` (each upstream primary Sign replaced by
                // tanh(β·z/rms)). Otherwise, or under `NY_SIGN_SMOOTH_FORWARD=0`,
                // or on a missing/short entry (deadline/partial map): the HARD-Sign
                // forward's pre-activations (node_bounds via `pre_activation`).
                let z: Vec<f32> = match smooth_pre.get(node_name) {
                    Some(zs) if smooth_sign && zs.len() == node_lb.num_inputs() => zs.clone(),
                    _ => pre_activation.flatten().lower().iter().copied().collect(),
                };
                // Element-wise: the incoming cotangent's columns index the Sign
                // outputs, which are 1:1 with its inputs, so z must have exactly
                // one entry per coefficient column.
                if z.len() != node_lb.num_inputs() {
                    return Ok(None);
                }
                // ATTACK-ONLY depth normalisation (net-2/net-3): the deeper BNNs
                // stack 3–4 Sign layers, so |z| GROWS with depth. A raw Sign layer
                // that operates on a large-magnitude pre-activation (measured here:
                // rms up to ~370 on net-1's dense head) has β·z ≫ 1 for almost
                // every unit, so tanh(β·z) saturates and the surrogate slope
                // underflows to ~0 — the gradient dies before it reaches the input.
                // Normalise the tanh ARGUMENT by the layer's RMS
                // `rms = sqrt(mean_j(z_j²))`. TWO recipes, selected by the flag
                // (both are ATTACK-ONLY ascent DIRECTIONS; every candidate is still
                // ORT + zero-tol re-checked, so no verdict can change):
                //
                // (A) DEFAULT (flag OFF) — rms FLOORED AT 1.0, LEADING (unfolded) β:
                //     slope_j = β·(1 − tanh²(β·z_j / max(rms, 1))).
                //     The 1.0 floor is one-sided — it only DE-saturates large-|z|
                //     layers (rms > 1) and never sharpens small-|z| ones; folding
                //     1/rms into β (or a 1e-6 floor) over-saturates small layers and
                //     measurably REGRESSES net-1. Evaluated at the HARD forward's z.
                //
                // (B) OPT-IN (flag ON) — rms UNFLOORED (`+1e-9`), FOLDED (β/rms):
                //     slope_j = (β/rms)·(1 − tanh²(β·z_j / rms)),
                //     evaluated at the SMOOTH forward's z (see `smooth_pre`). This is
                //     the EXACT external `bnn_falsifier` prototype recipe (its
                //     `softsign` returns `deriv = (α/c)·(1−t²)`, `c = rms(z)+1e-9`),
                //     which cracks the hard-tail traffic_signs instances that recipe
                //     (A) times out on.
                let (rms, lead) = {
                    let mean_sq =
                        z.iter().map(|&zj| (zj as f64) * (zj as f64)).sum::<f64>() / z.len() as f64;
                    let raw_rms = mean_sq.sqrt() as f32;
                    if smooth_sign {
                        let r = raw_rms + 1e-9; // (B) UNFLOORED
                        (r, beta / r) // (B) FOLDED (β/rms)
                    } else {
                        let r = raw_rms.max(1.0); // (A) FLOORED at 1.0
                        (r, beta) // (A) LEADING β
                    }
                };
                // (C) MASKED STRAIGHT-THROUGH ESTIMATOR (#traffic-masked-ste).
                //
                // The decisive mode for the traffic net-1 hard tail. Under a plain
                // STE (`d sign/dz := 1`) a BNN's Jacobian is CONSTANT in x, so one
                // corner move along it already beats hundreds of greedy-ascent
                // restarts. What makes it CLOSE is a REACHABILITY WINDOW: zero the
                // slope wherever `|z| > t`, because a Sign that far from its
                // threshold cannot be flipped by any admissible input move, and
                // keeping it in the gradient just dilutes the direction.
                //
                // `t` is matched to the INTEGER structure of these nets: with ±1
                // weights and no biases the second-layer pre-activation is an EVEN
                // integer, so `t = 6` means "at most 3 sign flips away".
                //
                // Measured on `model_30_idx_11379_eps_3` (best margin reached; a
                // counterexample needs >= 0):
                //   box center            -1228
                //   random lattice         -932
                //   212-restart greedy     -464
                //   one-shot UNMASKED STE  -420   <- stalls, mode (A)/(B) territory
                //   MASKED STE (t=6)        +20   <- COUNTEREXAMPLE
                // and `idx_12375_eps_3` reaches +28. Both were validated
                // GENUINE-IN-BOX-CE by `scripts/extended_bank/vnnlib_ce.py`.
                //
                // The window is the whole trick: unmasked STE and the tanh modes
                // both plateau ~1200 margin units short.
                //
                // ATTACK DIRECTION ONLY — this scales a NON-CERTIFIED surrogate
                // slope used to pick the next PGD step. Acceptance remains the
                // unchanged `gate_sat_with_trusted_oracle` (ORT + f64 + zero-tol
                // closed box), so no slope choice here can make ny unsound.
                //
                // PER-NODE WINDOW (#traffic-input-linear-window): the flat `t` is
                // calibrated in `±1` (Sign-output) input units, which is WRONG for
                // the first Sign of a BNN — it sees a convolution of raw pixels
                // (measured `mean|z| = 1356` against a reachability radius of
                // 40.5–81), so a flat `t = 6` holds ~3 of 12544 units and the
                // gradient dies. When the attack lane installed an EXACT per-unit
                // reachability window for this node
                // ([`GraphNetwork::input_linear_sign_windows`]) use it; otherwise
                // the flat default, byte-identical to today.
                let slopes: Vec<f32> = if let Some(window) = masked_ste_window() {
                    ATTACK_STE_WINDOWS.with(|w| {
                        let per_node = w.borrow();
                        let per_unit = per_node
                            .get(node_name)
                            .map(Vec::as_slice)
                            .filter(|t| t.len() == z.len());
                        masked_ste_slopes(&z, window, per_unit, ste_depth_scale())
                    })
                } else {
                    z.iter()
                        .map(|&zj| {
                            let t = (beta * zj / rms).tanh();
                            lead * (1.0 - t * t) // (A) β·sech² ∈ (0, β] | (B) (β/rms)·sech²
                        })
                        .collect()
                };
                ste_mask_diag(node_name, &z, &slopes, rms);
                // Scale coefficient column j by the surrogate slope_j (diagonal
                // Jacobian). Biases carry through unchanged: they route to the
                // network input as a constant channel and do not affect the
                // extracted gradient row. Error carriers are already None here.
                let mut surrogate = node_lb;
                surrogate
                    .lower_a_mut()
                    .indexed_iter_mut()
                    .for_each(|((_, j), v)| *v *= slopes[j]);
                surrogate
                    .upper_a_mut()
                    .indexed_iter_mut()
                    .for_each(|((_, j), v)| *v *= slopes[j]);
                if !attack_publication_completed(
                    self.accumulate_dense_bounds_to_input_with_deadline(
                        first_input,
                        surrogate,
                        &mut acc,
                        SPEC_ROWS,
                        input_dim,
                        &mut input_accumulated,
                        deadline,
                    ),
                )? {
                    return Ok(None);
                }
                continue;
            }

            // === Softmax: ATTACK-ONLY monotone identity pass-through. ===
            // The traffic_signs BNNs end in Softmax, but the property is an
            // argmax/margin over the outputs and softmax is strictly monotone, so
            // the logit-space margin gradient is a valid (and non-vanishing) ascent
            // direction. Routing softmax's certified relaxation here would instead
            // hand back a saturated/near-zero cotangent. We therefore skip the
            // nonlinearity for the ATTACK and pass the incoming cotangent straight
            // to the logits input — exactly what the external bnn_falsifier
            // prototype does (it seeds the backward on the logits and ignores the
            // softmax). Softmax is element-wise-shaped (in-dim == out-dim), so the
            // cotangent flows to the input unchanged. Attack-only: never a bound;
            // every candidate is still concretely re-checked by the trusted gate.
            if matches!(&node.layer, Layer::Softmax(_)) {
                if !attack_publication_completed(
                    self.accumulate_dense_bounds_to_input_with_deadline(
                        first_input,
                        node_lb,
                        &mut acc,
                        SPEC_ROWS,
                        input_dim,
                        &mut input_accumulated,
                        deadline,
                    ),
                )? {
                    return Ok(None);
                }
                continue;
            }

            // === All other whitelist layers: shared canonical dispatch. ===
            // PERF: for `Layer::Linear` this enters `aw_f64_with_abssum` (f64 A·W
            // with abssum error) and for `Layer::Conv2d` the sound conv-transpose
            // backward — both inside `dispatch_backward_layer`. For an ATTACK
            // gradient these could be swapped for `fast_f32_gemm::with_engine(..)`
            // (Linear) and a direct `conv2d_transpose` (Conv2d) to get the f32
            // speedup once this pass is validated against the exact oracle. The
            // dispatch result shape (Single / Binary / PassThrough) is unchanged.
            let ctx = DispatchContext {
                node_name,
                layer: &node.layer,
                inputs: &node.inputs,
                pre_activation,
                network_input: &input_bounds,
                node_bounds: (&node_bounds).into(),
                engine,
                deadline,
                bilinear_alphas: None,
                mul_binary_relaxation: MulBinaryRelaxationMode::default(),
                mul_binary_alphas: None,
                norm_inv_rms_override: None,
            };

            let Some(result) = attack_step_or_decline(dispatch_backward_layer(&ctx, &node_lb))?
            else {
                return Ok(None);
            };
            if let BackwardDispatchResult::Unsupported(_reason) = &result {
                // Outside the exact-linear fragment at dispatch time → fall back.
                return Ok(None);
            }

            // Distribute the result to the node's input(s). For a binary `Add`
            // (residual), `bounds_a` is routed to inputs[0] and `bounds_b` to
            // inputs[1]; the separate bias channel is folded onto the network
            // input. When two distinct paths reach the SAME node (the classic
            // residual, where one tensor feeds both a conv and the skip Add), the
            // SECOND `accumulate_dense_bounds_to_input_with_deadline` on that node
            // name is a MERGE: the deadline-aware accumulator SUMS the two cotangent
            // matrices (in f64) instead of overwriting. That summation is the
            // reverse-mode "fan-in adds" rule, and it is why reverse-topological
            // order (all consumers before the node) is required.
            if !attack_publication_completed(apply_dense_backward_dispatch_result_with_deadline(
                self,
                node,
                first_input,
                &node_lb,
                result,
                &mut acc,
                SPEC_ROWS,
                input_dim,
                &mut input_accumulated,
                "attack_point_gradient",
                deadline,
            ))? {
                return Ok(None);
            }
        }

        // --- Extract the gradient row at the network input. -----------------
        let Some(final_cb) =
            attack_step_or_decline(acc.take_with_deadline(NETWORK_INPUT, deadline))?
        else {
            return Ok(None);
        };
        let final_cb = match final_cb {
            Some(bounds) => bounds,
            None => return Ok(None), // no path reached the input (zero gradient region)
        };
        let Some(final_lb) = take_attack_dense_carrier(final_cb) else {
            tracing::debug!(
                "attack_point_gradient: unexpected Patches carrier at network input; \
                 declining Dense-only point-VJP route"
            );
            return Ok(None);
        };
        if final_lb.num_outputs() != SPEC_ROWS || final_lb.num_inputs() != input_dim {
            return Ok(None);
        }

        // At a point lower_a == upper_a in exact arithmetic; the only asymmetry is
        // the ±1 ULP directed-rounding gap introduced by the f64 merge downcast
        // (next_down for lower, next_up for upper). Averaging recovers the
        // symmetric gradient — mirrors graph_pgd_exact.rs:159.
        let grad_row = (final_lb.lower_a() + final_lb.upper_a()) * 0.5;
        let grad_flat: Vec<f32> = grad_row.row(0).to_vec();
        if grad_flat.len() != x.len() {
            return Ok(None);
        }
        let grad = ArrayD::from_shape_vec(IxDyn(x.shape()), grad_flat)?;
        Ok(Some(grad))
    }
}

#[cfg(test)]
#[path = "point_vjp_tests.rs"]
mod tests;
