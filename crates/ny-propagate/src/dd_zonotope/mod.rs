// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Certified sparse-input double-double zonotope forward pass (`#dd-zonotope`).
//!
//! # What this is for
//!
//! The `vggnet16_2022` VNN-COMP category fixes almost every input pixel: only
//! `k` pixels carry a non-degenerate interval (MEASURED k = 1, 5, 10, 20, 100
//! for specs 0-14; the full 150528 for specs 15-17). ny's graph engine routes
//! that model to the `large_conv_graph` arm, i.e. plain IBP intermediates,
//! whose VGG16 boxes explode to ~1e13 and produce a `-1.8e18` root bound. The
//! true margin varies by ~1e-6 across the whole box.
//!
//! A DeepZ zonotope over a `k`-column input basis closes that gap — but ONLY
//! with a working precision above f64. The reference probe
//! (`scripts/vggnet16_zonotope_rounding_probe.py`, commit 49737525) MEASURED
//! the certified rounding half-width as a function of the unit roundoff and
//! found a stable amplification of `~2^66`: at f64 the certified margin is
//! `[-29298, +29045]` on a true margin of `1.6375` (vacuous, and it
//! manufactures 31121 spurious generator columns); at double-double it is
//! `1.1e-12` wide. Hence `ny_core::dd`.
//!
//! # Soundness posture
//!
//! This module produces a NEW certified lower bound, so every `-150` risk in
//! the lane lives here. Four independent guards, in order of strength:
//!
//! 1. **Self-policing precision gate.** The certified rounding half-width is
//!    COMPUTED, not assumed. [`DdZonoMargin::precision_ok`] refuses to
//!    contribute a margin unless that half-width is below
//!    `precision_ratio * |margin center|`. So "is double-double enough for
//!    THIS network?" is answered at runtime, not by a static claim about
//!    `vgg16-7`.
//! 2. **`dd_selfcheck`.** The double-double error-free transformations are
//!    algebraically trivial and a reassociating backend silently degrades them
//!    to f64 — which would publish a bound that is too TIGHT. The path refuses
//!    unless `ny_core::dd_selfcheck::dd_selfcheck_ok()` passes.
//! 3. **Fail-closed everywhere.** Any unsupported layer, non-finite value,
//!    generator-count overflow, byte-budget overflow, or deadline expiry
//!    returns `None`. There is no degraded bound.
//! 4. **Intersect, never replace.** The caller keeps the tighter side of two
//!    certified enclosures; a refusal is byte-identical to today.
//!
//! # Deliberate scope limits
//!
//! * `Conv2d` (dilation 1, groups 1), `Linear`, `ReLU`, `MaxPool2d` (no
//!   padding), `Flatten`/`Reshape`/`Squeeze`/`Unsqueeze`/`Identity`, and
//!   node-to-node `Add`. **Everything else refuses.**
//! * The pass is admitted only for large-input (`> 50000` element) graphs with
//!   `k <= 128` perturbed inputs, so no other VNN-COMP category can be
//!   affected — see [`DdZonoPlan::detect`].

mod affine;
pub mod certified_box;
mod maxpool;
mod relu;
mod state;

#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::time::Instant;

use ny_core::dd::{next_down_f64, next_up_f64, Dd, U_F64};
use ny_core::{NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use crate::layers::Layer;
use crate::GraphNetwork;
use crate::{GraphNode, NETWORK_INPUT};

use affine::{apply_affine, AffineOp, ConvPlan};
use maxpool::{apply_maxpool, PoolPlan};
use relu::{apply_relu, prune_zero_generators};
use state::{err_up, DdZono};

/// Runtime configuration of the certified double-double zonotope pass.
#[derive(Debug, Clone, Copy)]
pub struct DdZonoConfig {
    /// Maximum number of perturbed input coordinates admitted.
    pub max_k: usize,
    /// Hard cap on live generator columns; exceeding it FAILS CLOSED.
    pub max_generators: usize,
    /// Hard cap on the bytes a single state may hold; exceeding it fails closed.
    pub max_bytes: usize,
    /// Minimum input volume for admission (keeps every non-image category out).
    pub min_input_numel: usize,
    /// Self-policing precision gate: refuse to publish a margin whose certified
    /// ROUNDING half-width exceeds this fraction of `|margin center|`.
    ///
    /// Default `0.1`. MEASURED on `vgg16-7` spec1: `2.04e-2` rounding
    /// half-width on a margin of `1.6376`, i.e. a ratio of `1.25e-2` — an order
    /// of magnitude inside the gate. A network for which double-double is
    /// genuinely insufficient sits far on the other side: the same spec seeded
    /// from plain-f64 exact endpoints MEASURED `53.4`, a ratio of `33`.
    pub precision_ratio: f64,
    /// Safety factor applied to the CERTIFIED ROUNDING CHANNEL before the
    /// verdict is taken.
    ///
    /// The verdict uses `center - relax - safety_factor * rounding > threshold`
    /// rather than the published `lower`. The rounding channel is the novel
    /// part of this certificate — a Higham `gamma_n` composition over the
    /// double-double envelope derived in `ny_core::dd` — so it is the term that
    /// deserves an explicit margin. Default `2.0`: the pass must still prove
    /// the property with DOUBLE the rounding error it computed for itself.
    pub safety_factor: f64,
    /// Width at or below which an input coordinate is folded into the interval
    /// channel `ec` instead of buying a generator column.
    ///
    /// Default `0.0`: EVERY non-degenerate coordinate buys a column, so a box
    /// with 150528 non-degenerate coordinates simply refuses. Raising it is the
    /// only way to admit a `CertifiedInputBox` whose "fixed" pixels carry the
    /// one-f64-ulp residue of a non-dyadic decimal — and the fold is sound (an
    /// interval over-approximates a `+-w` symbol) but is transported by `|W|`,
    /// i.e. by IBP. The self-policing precision gate is what decides whether
    /// the result survives. See `certified_box` for the measurement.
    pub point_tol: f64,
    /// **MEASUREMENT ONLY — changes what is verified. Default `false`.**
    ///
    /// When set (`NY_DD_ZONOTOPE_DECLARED_POINT_EXACT=1`), a sub-`point_tol`
    /// input coordinate is seeded as an EXACT POINT at the midpoint of its
    /// exact-decimal enclosure instead of carrying its residual width in the
    /// interval channel.
    ///
    /// That residue is the gap between a non-dyadic VNN-LIB decimal (e.g.
    /// `2.6399001`) and its nearest f64 — one f64 ulp, `~4.4e-16` at vggnet16
    /// pixel magnitudes. Dropping it verifies the property at the FLOAT point
    /// rather than over the exact-rational point. That is how every tool which
    /// parses the decimal into a float behaves, but it is strictly weaker than
    /// what the rest of ny guarantees (`split_input_bounds_f32` rounds OUTWARD
    /// to f32, a `~2.4e-7`-wide interval). It exists so the cost of the
    /// rigorous treatment can be MEASURED rather than argued about, and every
    /// telemetry line it produces is tagged `semantics=float-point` so no
    /// number obtained under it can be mistaken for the shipped one.
    pub declared_point_exact: bool,
}

impl Default for DdZonoConfig {
    fn default() -> Self {
        DdZonoConfig {
            max_k: 128,
            max_generators: 512,
            // ~48 GiB. The measured peak for k=100 on VGG16 is the first conv
            // output: 3211264 elements x ~200 columns x 8 B = 5.1 GB.
            max_bytes: 48 << 30,
            min_input_numel: 50_000,
            precision_ratio: 0.1,
            safety_factor: 2.0,
            point_tol: 0.0,
            declared_point_exact: false,
        }
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(default)
}

impl DdZonoConfig {
    /// Read the config from the environment (all knobs optional).
    #[must_use]
    pub fn from_env() -> Self {
        let d = DdZonoConfig::default();
        DdZonoConfig {
            max_k: env_usize("NY_DD_ZONOTOPE_MAX_K", d.max_k),
            max_generators: env_usize("NY_DD_ZONOTOPE_MAX_GENS", d.max_generators),
            max_bytes: env_usize("NY_DD_ZONOTOPE_MAX_GB", d.max_bytes >> 30) << 30,
            min_input_numel: env_usize("NY_DD_ZONOTOPE_MIN_INPUT", d.min_input_numel),
            precision_ratio: env_f64("NY_DD_ZONOTOPE_PRECISION_RATIO", d.precision_ratio),
            safety_factor: env_f64("NY_DD_ZONOTOPE_SAFETY", d.safety_factor),
            point_tol: std::env::var("NY_DD_ZONOTOPE_POINT_TOL")
                .ok()
                .and_then(|v| v.trim().parse::<f64>().ok())
                .filter(|v| v.is_finite() && *v >= 0.0)
                .unwrap_or(d.point_tol),
            declared_point_exact: std::env::var("NY_DD_ZONOTOPE_DECLARED_POINT_EXACT")
                .ok()
                .as_deref()
                == Some("1"),
        }
    }
}

/// Is the dark `#dd-zonotope` gate on?
///
/// Read once per process so a mid-run environment change cannot make two
/// callers disagree about whether the path is armed.
#[must_use]
pub fn dd_zonotope_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        // DEFAULT-ON (`NY_DD_ZONOTOPE=0` is the kill switch). The 2026-07-23
        // adversarial audit (5 attack agents + a real-model oracle run) returned
        // GO-AFTER-FIXES; the one residual gap (two_prod underflow) is closed by
        // the absolute floor in `affine.rs`. Blast radius is structurally
        // bounded: `DdZonoPlan::detect` is fail-closed and self-limiting
        // (k <= 128 perturbed inputs, exact-decimal box required, full op-surface
        // check), so every non-sparse-input instance declines before any
        // allocation and is byte-identical to gate-off. MEASURED with the gate
        // on: soundnessbench sat/sat/sat, acasxu unsat, vggnet16 spec0 stays
        // sat, spec1/spec2 convert timeout -> unsat in ~252 s (root-caused to
        // this path via NY_DD_ZONOTOPE_VERBOSE telemetry; without it spec1 is
        // still unconverged at 6 h).
        std::env::var("NY_DD_ZONOTOPE")
            .ok()
            .is_none_or(|v| v != "0")
    })
}

/// Print-only telemetry gate (`NY_DD_ZONOTOPE_VERBOSE=1`).
fn verbose() -> bool {
    std::env::var("NY_DD_ZONOTOPE_VERBOSE").ok().as_deref() == Some("1")
}

/// Print-only refusal reason. Every decline is silent in production; without
/// this, "the detector declined" is unactionable.
fn decline(reason: &str) {
    if verbose() {
        eprintln!("[dd-zonotope] detect declined: {reason}");
    }
}

/// Admission decision for one (graph, input) pair.
#[derive(Debug, Clone)]
pub struct DdZonoPlan {
    /// Indices of the perturbed input coordinates (`lower < upper` in the
    /// EXACT-decimal box).
    perturbed: Vec<usize>,
    /// Element shape of the network input with leading unit axes stripped.
    input_shape: Vec<usize>,
    /// Exact-decimal input box at DOUBLE-DOUBLE center precision, from
    /// [`certified_box`]. Required: the engine's f32 box is 2 f32 ULPs wide on
    /// every "fixed" pixel, which this method cannot afford to carry.
    exact: certified_box::ExactBox,
    /// See [`DdZonoConfig::declared_point_exact`]. MEASUREMENT ONLY.
    declared_point_exact: bool,
}

impl DdZonoPlan {
    /// Number of perturbed input coordinates.
    #[must_use]
    pub fn k(&self) -> usize {
        self.perturbed.len()
    }

    /// Decide whether the certified pass may run.
    ///
    /// CONJUNCTIVE and narrow by design (risk R6): a misfiring detector could
    /// turn a currently-correct instance in another category into a timeout.
    /// Every clause must hold.
    #[must_use]
    pub fn detect(
        graph: &GraphNetwork,
        input: &BoundedTensor,
        cfg: &DdZonoConfig,
    ) -> Option<DdZonoPlan> {
        // The double-double EFTs must have survived the compiler. A silent
        // degradation to f64 would publish a bound that is too TIGHT.
        if !ny_core::dd_selfcheck::dd_selfcheck_ok() {
            decline("dd_selfcheck failed: the double-double EFTs did not survive the compiler");
            return None;
        }
        if input.len() < cfg.min_input_numel {
            decline(&format!(
                "input volume {} < min_input_numel {}",
                input.len(),
                cfg.min_input_numel
            ));
            return None;
        }
        // The EXACT-decimal box is mandatory, not an optimisation: the
        // engine's f32 box is 2 f32 ULPs wide on every declared-point pixel
        // (`split_input_bounds_f32`, #2658), which would make `k = 150528` and,
        // if carried in the interval channel instead, IBP-explode. See
        // `certified_box` for the measurement behind that.
        let Some(exact) = certified_box::lookup(input) else {
            decline(
                "no exact-decimal input box was published for this instance \
                 (see dd_zonotope::certified_box)",
            );
            return None;
        };
        if exact.lower.len() != input.len() {
            decline("exact-decimal box arity mismatch");
            return None;
        }
        let mut perturbed = Vec::new();
        let mut max_folded_width = 0.0_f64;
        // A coordinate buys a generator iff its EXACT half-width is positive.
        // That is derived from the rationals, so a declared point whose decimal
        // is non-dyadic (`2.6399001`) is still exactly `0.0` here — unlike
        // `upper - lower`, which carries the outward-f64 bracket.
        for i in 0..exact.lower.len() {
            let (l, u) = (exact.lower[i], exact.upper[i]);
            if !l.is_finite() || !u.is_finite() || l > u {
                decline(&format!("non-finite or inverted exact bound at input {i}"));
                return None;
            }
            let w = 2.0 * exact.half_width[i];
            if w > cfg.point_tol {
                perturbed.push(i);
                if perturbed.len() > cfg.max_k {
                    decline(&format!(
                        "k exceeds max_k={} (widths > point_tol={:.3e}); the exact box has \
                         too many non-degenerate coordinates for a per-coordinate basis",
                        cfg.max_k, cfg.point_tol
                    ));
                    return None;
                }
            } else if w > max_folded_width {
                max_folded_width = w;
            }
        }
        if perturbed.is_empty() {
            decline("every input coordinate is degenerate");
            return None;
        }
        if verbose() {
            eprintln!(
                "[dd-zonotope] detect: k={} point_tol={:.3e} max_folded_input_width={:.3e} \
                 semantics={}",
                perturbed.len(),
                cfg.point_tol,
                max_folded_width,
                if cfg.declared_point_exact {
                    "float-point (MEASUREMENT ONLY, weaker than ny's outward f32 box)"
                } else {
                    "exact-decimal (rigorous)"
                }
            );
        }
        // Full op-surface support is required up front: a mid-pass refusal
        // would have already burnt the budget.
        let order = graph.exec_order().ok()?;
        for name in order {
            let node = graph.node(name)?;
            if !layer_supported(node.layer()) {
                decline(&format!(
                    "unsupported layer '{}' ({})",
                    name,
                    node.layer().layer_type()
                ));
                return None;
            }
        }
        let Some(shape) = input_chw(graph, input) else {
            decline("could not recover a (C, H, W) input shape");
            return None;
        };
        Some(DdZonoPlan {
            perturbed,
            input_shape: shape,
            exact,
            declared_point_exact: cfg.declared_point_exact,
        })
    }
}

/// Recover the `(C, H, W)` element shape of the network input.
///
/// `Verifier::bounds_to_tensor` builds the engine box from a flat `Vec<Bound>`
/// and only carries a shape when the caller supplies one, so the box can arrive
/// as `[150528]`, `[3, 224, 224]`, or `[1, 3, 224, 224]`. All three are handled:
/// leading unit axes are stripped, and a flat box is reconstructed from the
/// FIRST Conv2d node's declared `input_shape` plus its kernel's input-channel
/// count. The product is always re-checked against `input.len()`, so a wrong
/// guess refuses instead of mis-indexing every convolution (which would be a
/// silent unsoundness, scoping risk R2).
fn input_chw(graph: &GraphNetwork, input: &BoundedTensor) -> Option<Vec<usize>> {
    let mut shape: Vec<usize> = input.shape().to_vec();
    while shape.len() > 3 && shape[0] == 1 {
        shape.remove(0);
    }
    if shape.len() == 3 && shape.iter().product::<usize>() == input.len() {
        return Some(shape);
    }
    let order = graph.exec_order().ok()?;
    for name in order {
        if let Some(Layer::Conv2d(conv)) = graph.node(name).map(GraphNode::layer) {
            let (h, w) = conv.input_shape?;
            let c = *conv.kernel.shape().get(1)?;
            if c.checked_mul(h)?.checked_mul(w)? == input.len() {
                return Some(vec![c, h, w]);
            }
            return None;
        }
    }
    None
}

fn layer_supported(layer: &Layer) -> bool {
    matches!(
        layer,
        Layer::Conv2d(_)
            | Layer::Linear(_)
            | Layer::ReLU(_)
            | Layer::MaxPool2d(_)
            | Layer::Flatten(_)
            | Layer::Reshape(_)
            | Layer::Squeeze(_)
            | Layer::Unsqueeze(_)
            | Layer::Add(_)
    )
}

/// Certified per-objective margin produced by the double-double zonotope.
#[derive(Debug, Clone)]
pub struct DdZonoMargin {
    /// Certified lower bound of `objective . output`, per objective.
    pub lower: Vec<f64>,
    /// Certified upper bound, per objective.
    pub upper: Vec<f64>,
    /// The part of the half-width attributable to the certified ROUNDING
    /// channel (`ec + eg`), per objective. This is the self-policing quantity.
    pub rounding_half_width: Vec<f64>,
    /// The part attributable to the zonotope RELAXATION (generator span).
    pub relax_half_width: Vec<f64>,
    /// Certified enclosure of the graph output node.
    pub output_lower: Vec<f32>,
    /// Certified enclosure of the graph output node.
    pub output_upper: Vec<f32>,
    /// Output element shape.
    pub output_shape: Vec<usize>,
    /// Live generator columns at the output.
    pub n_generators: usize,
    /// Wall time of the pass.
    pub wall: std::time::Duration,
}

impl DdZonoMargin {
    /// Self-policing precision gate: is the CERTIFIED rounding channel small
    /// enough, on every objective, for this margin to be worth publishing?
    ///
    /// The point is that this is measured, not assumed. The probe's `~2^66`
    /// amplification was observed on `vgg16-7` only (risk R4); a deeper or
    /// larger-weight network could exceed double-double. This turns that from
    /// an unsound assumption into a refusal.
    /// Certified lower bound with the ROUNDING channel scaled by
    /// `safety_factor`, i.e. `center - relax - factor * rounding`.
    ///
    /// The published [`Self::lower`] already subtracts the rounding channel
    /// once; this subtracts `factor - 1` more of it. Used for the VERDICT, so
    /// the property must survive an error term this pass overestimated by the
    /// safety factor.
    #[must_use]
    pub fn lower_with_safety(&self, index: usize, factor: f64) -> f64 {
        let (Some(&l), Some(&hw)) = (self.lower.get(index), self.rounding_half_width.get(index))
        else {
            return f64::NEG_INFINITY;
        };
        if !l.is_finite() || !hw.is_finite() || !factor.is_finite() || factor < 1.0 {
            return f64::NEG_INFINITY;
        }
        next_down_f64(l - (factor - 1.0) * hw)
    }

    #[must_use]
    pub fn precision_ok(&self, ratio: f64) -> bool {
        self.lower
            .iter()
            .zip(self.upper.iter())
            .zip(self.rounding_half_width.iter())
            .all(|((&l, &u), &hw)| {
                if !l.is_finite() || !u.is_finite() || !hw.is_finite() || l > u {
                    return false;
                }
                let center = 0.5 * (l + u);
                hw <= ratio * center.abs()
            })
    }
}

/// Run the certified double-double zonotope forward pass and evaluate every
/// objective row on the resulting zonotope.
///
/// Returns `Ok(None)` for every refusal (unsupported op, cap exceeded,
/// non-finite value, deadline expiry). It NEVER returns a degraded bound.
pub fn dd_zonotope_margins(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    objectives: &[Vec<f32>],
    plan: &DdZonoPlan,
    cfg: &DdZonoConfig,
    deadline: Option<Instant>,
) -> Result<Option<DdZonoMargin>> {
    let t0 = Instant::now();
    if !ny_core::dd_selfcheck::dd_selfcheck_ok() {
        return Ok(None);
    }
    // The plan's exact box must describe THIS input; `certified_box::lookup`
    // already checked byte-exact identity and containment, but the plan can be
    // carried by a caller, so re-check the arity before any allocation.
    if plan.exact.lower.len() != input.len() {
        return Ok(None);
    }
    let poll = || -> Result<()> {
        match deadline {
            Some(d) if Instant::now() >= d => Err(NyError::DeadlineExceeded(
                "#dd-zonotope certified forward pass".to_string(),
            )),
            _ => Ok(()),
        }
    };

    let seeded = match seed(plan) {
        Some(z) => z,
        None => return Ok(None),
    };

    let order = graph.exec_order()?.to_vec();
    // Only the values still needed downstream are kept live.
    let mut live: HashMap<String, DdZono> = HashMap::new();
    let mut last_name = NETWORK_INPUT.to_string();
    live.insert(last_name.clone(), seeded);

    for name in &order {
        if poll().is_err() {
            return Ok(None);
        }
        let node = match graph.node(name) {
            Some(n) => n,
            None => return Ok(None),
        };
        let out = match step(
            node.layer(),
            node.inputs(),
            &live,
            cfg.max_generators,
            &poll,
        ) {
            Some(out) => out,
            None => {
                if verbose() {
                    eprintln!(
                        "[dd-zonotope] refusing at node '{}' ({})",
                        name,
                        node.layer().layer_type()
                    );
                }
                return Ok(None);
            }
        };
        let mut out = out;

        // Relaxation spends: cap FAILS CLOSED.
        if out.n_gens() > cfg.max_generators {
            if verbose() {
                eprintln!(
                    "[dd-zonotope] generator cap exceeded at '{}': {} > {}",
                    name,
                    out.n_gens(),
                    cfg.max_generators
                );
            }
            return Ok(None);
        }
        if out.bytes() > cfg.max_bytes {
            if verbose() {
                eprintln!(
                    "[dd-zonotope] byte cap exceeded at '{}': {} > {}",
                    name,
                    out.bytes(),
                    cfg.max_bytes
                );
            }
            return Ok(None);
        }
        if !out.all_finite() {
            if verbose() {
                eprintln!("[dd-zonotope] non-finite state at '{name}'; refusing");
            }
            return Ok(None);
        }
        prune_zero_generators(&mut out);

        if verbose() {
            let (lo, up) = out.concretize();
            let maxw = lo
                .iter()
                .zip(up.iter())
                .map(|(a, b)| b - a)
                .fold(0.0_f64, f64::max);
            let max_e = out
                .ec
                .iter()
                .zip(out.eg.iter())
                .map(|(a, b)| a + b)
                .fold(0.0_f64, f64::max);
            eprintln!(
                "[dd-zonotope] {:<24} {:<12} n={:<9} gens={:<4} maxwidth={:.4e} maxErr={:.4e}",
                name,
                node.layer().layer_type(),
                out.numel(),
                out.n_gens(),
                maxw,
                max_e
            );
        }

        // Free every producer whose last consumer is this node.
        live.insert(name.clone(), out);
        retire_dead(&live_names_needed(&order, name, graph), &mut live);
        last_name = name.clone();
    }

    let out_name = if graph.output_name().is_empty() {
        last_name
    } else {
        graph.output_name().to_string()
    };
    let z = match live.get(&out_name) {
        Some(z) => z,
        None => return Ok(None),
    };

    let margin = evaluate_objectives(z, objectives, t0.elapsed());
    Ok(margin)
}

/// Nodes whose values are still needed after `current` has been produced.
fn live_names_needed(order: &[String], current: &str, graph: &GraphNetwork) -> Vec<String> {
    let pos = order.iter().position(|n| n == current).unwrap_or(0);
    let mut needed = vec![current.to_string()];
    if !graph.output_name().is_empty() {
        needed.push(graph.output_name().to_string());
    }
    for name in &order[pos + 1..] {
        if let Some(node) = graph.node(name) {
            for input in node.inputs() {
                needed.push(input.clone());
            }
        }
    }
    needed
}

fn retire_dead(needed: &[String], live: &mut HashMap<String, DdZono>) {
    live.retain(|k, _| needed.iter().any(|n| n == k));
}

/// Seed the zonotope from the plan's exact-decimal input box, at DOUBLE-DOUBLE
/// center precision.
///
/// # The f32 gap (scoping risk R7), and why it forced a new seam
///
/// `ny_core::Bound` is f32 and `VnnLibSpec::split_input_bounds_f32`
/// (`ny-onnx/src/vnnlib/spec.rs:414`, #2658) widens every endpoint outward by
/// one f32 ULP, so the engine's box is a strict SUPERSET of the VNN-LIB
/// region — sound, and unusable here: it makes all 150528 vggnet16 pixels
/// non-degenerate. Carrying that as generators is unaffordable and carrying it
/// in `ec` is IBP, which the probe measured as vacuous on VGG16. The pass
/// therefore seeds from `crate::dd_zonotope::certified_box`, which the CLI
/// populates from `ny_onnx::vnnlib::CertifiedInputBox` (exact rationals,
/// endpoints rounded OUTWARD to f64). Without it, `DdZonoPlan::detect`
/// refuses.
///
/// Consequences for the certificate:
///
/// * `two_sum(l, u)` is exact and halving is exact, so the double-double
///   center is the EXACT midpoint of the exact box and `ec` starts at `0` —
///   which is mandatory: even one f64 ULP of center uncertainty
///   (`~2.9e-16`) reaches the logits at ~1e4 under the measured amplification.
/// * The generator half-width is rounded OUTWARD by one ulp so the zonotope
///   encloses the exact box; `eg` also starts at `0`.
fn seed(plan: &DdZonoPlan) -> Option<DdZono> {
    let e = &plan.exact;
    let n = e.lower.len();
    let mut center = Vec::with_capacity(n);
    let mut ec = vec![0.0_f64; n];
    let mut is_gen = vec![false; n];
    for &i in &plan.perturbed {
        is_gen[i] = true;
    }
    for i in 0..n {
        let (hi, lo) = ny_core::dd::two_sum(e.center_hi[i], e.center_lo[i]);
        let c = Dd { hi, lo };
        if !c.is_finite() {
            return None;
        }
        center.push(c);
        // Residual of the double-double decomposition of the EXACT rational
        // center. `~|x| * 2^-106 ~ 3e-32` at vggnet16 pixel magnitudes — the
        // whole reason a second word is carried; see `certified_box`.
        let mut e_i = e.center_err[i];
        if !is_gen[i] && !plan.declared_point_exact {
            // A sub-tolerance coordinate keeps its exact half-width in the
            // INTERVAL channel (sound: an interval over-approximates a `+-w`
            // symbol). For a declared point this term is exactly zero.
            e_i += e.half_width[i];
        }
        ec[i] = relu::up_nonneg(e_i);
    }
    let mut gens = Vec::with_capacity(plan.perturbed.len());
    for &i in &plan.perturbed {
        let half = e.half_width[i];
        if !half.is_finite() || half < 0.0 {
            return None;
        }
        let mut col = vec![0.0_f64; n];
        col[i] = half;
        gens.push(col);
    }
    Some(DdZono {
        shape: plan.input_shape.clone(),
        center,
        gens,
        ec,
        eg: vec![0.0; n],
    })
}

/// Apply one graph node. `None` is a refusal.
fn step(
    layer: &Layer,
    inputs: &[String],
    live: &HashMap<String, DdZono>,
    max_generators: usize,
    poll: &impl Fn() -> Result<()>,
) -> Option<DdZono> {
    match layer {
        Layer::Conv2d(conv) => {
            let z = live.get(inputs.first()?)?;
            let shape = as_chw(&z.shape)?;
            let kernel = conv.kernel.shape();
            if kernel.len() != 4 {
                return None;
            }
            let plan = ConvPlan::build(
                shape,
                kernel[0],
                kernel[2],
                kernel[3],
                conv.stride,
                conv.padding,
                conv.dilation,
                conv.groups,
            )?;
            if kernel[1] != plan.in_c {
                return None;
            }
            let w: Vec<f64> = conv.kernel.iter().map(|&v| f64::from(v)).collect();
            let wabs: Vec<f64> = w.iter().map(|v| v.abs()).collect();
            let bias: Option<Vec<f64>> = conv
                .bias
                .as_ref()
                .map(|b| b.iter().map(|&v| f64::from(v)).collect());
            let op = AffineOp::Conv {
                plan,
                w: &w,
                wabs: &wabs,
                bias: bias.as_deref(),
            };
            apply_affine(z, &op, poll).ok()
        }
        Layer::Linear(lin) => {
            let z = live.get(inputs.first()?)?;
            let (out_features, in_features) = (lin.weight.nrows(), lin.weight.ncols());
            if z.numel() != in_features {
                return None;
            }
            let w: Vec<f64> = lin
                .weight
                .iter()
                .copied()
                .map(f64::from)
                .collect::<Vec<f64>>();
            let wabs: Vec<f64> = w.iter().map(|v| v.abs()).collect();
            let bias: Option<Vec<f64>> = lin
                .bias
                .as_ref()
                .map(|b| b.iter().map(|&v| f64::from(v)).collect());
            let op = AffineOp::Linear {
                out_features,
                in_features,
                w: &w,
                wabs: &wabs,
                bias: bias.as_deref(),
            };
            apply_affine(z, &op, poll).ok()
        }
        Layer::ReLU(_) => {
            let mut z = live.get(inputs.first()?)?.clone();
            let outcome = apply_relu(&mut z)?;
            if verbose() {
                eprintln!(
                    "[dd-zonotope]   relu: crossing={} spend={} fold={}",
                    outcome.relaxed,
                    outcome.spent.len(),
                    outcome.folded
                );
            }
            // Check the cap BEFORE allocating the new columns: a relaxation
            // that wants 10^5 generators on a 3.2M-element tensor would OOM
            // long before a post-hoc check could refuse (scoping risk R5).
            if z.n_gens() + outcome.spent.len() > max_generators {
                return None;
            }
            for (i, mu) in outcome.spent {
                z.push_sparse_generator(&[(i, mu)]);
            }
            Some(z)
        }
        Layer::MaxPool2d(pool) => {
            let z = live.get(inputs.first()?)?;
            let shape = as_chw(&z.shape)?;
            let plan = PoolPlan::build(shape, pool.kernel_size, pool.stride, pool.padding)?;
            let (mut out, outcome) = apply_maxpool(z, &plan)?;
            if verbose() {
                eprintln!(
                    "[dd-zonotope]   maxpool: tied={} spend={} fold={}",
                    outcome.relaxed,
                    outcome.spent.len(),
                    outcome.folded
                );
            }
            if out.n_gens() + outcome.spent.len() > max_generators {
                return None;
            }
            for (i, half) in outcome.spent {
                out.push_sparse_generator(&[(i, half)]);
            }
            Some(out)
        }
        Layer::Flatten(_) | Layer::Reshape(_) | Layer::Squeeze(_) | Layer::Unsqueeze(_) => {
            let mut z = live.get(inputs.first()?)?.clone();
            let n = z.numel();
            z.reshape(vec![n]);
            Some(z)
        }
        Layer::Add(_) => {
            if inputs.len() != 2 {
                return None;
            }
            let a = live.get(&inputs[0])?;
            let b = live.get(&inputs[1])?;
            add_states(a, b)
        }
        _ => None,
    }
}

/// Sound zonotope addition: centers add in double-double, error channels add,
/// and the two operands' generator columns are **concatenated** as independent
/// symbols.
///
/// # Why concatenate rather than sum column-by-column
///
/// A zonotope column `j` is a specific noise symbol `eps_j in [-1, 1]`. Summing
/// `a.gens[j] + b.gens[j]` is only correct when the two operands' column `j`
/// denote the SAME symbol. That holds for the shared input-seed prefix, but a
/// DAG that branches and then spends a relaxation generator on each branch
/// (a crossing ReLU or a tied max-pool window) appends DIFFERENT new symbols at
/// the same index on each side. Summing those treats two independent
/// uncertainties as perfectly correlated, which lets them cancel and yields a
/// set SMALLER than the true one — an unsound (too-tight) bound, i.e. a -150.
///
/// Concatenation keeps every column as its own symbol. For a genuinely shared
/// symbol that is merely looser (two independent copies enclose the correlated
/// one); for an independent symbol it is exact. Either way it is always a
/// superset of the true reachable set, so it is unconditionally sound on ANY
/// graph shape. VGG16 has no `Add`, so on the targeted category this branch is
/// never taken; the concatenation exists so a future residual admission cannot
/// silently produce a wrong `unsat`. The generator cap upstream bounds the
/// column growth.
fn add_states(a: &DdZono, b: &DdZono) -> Option<DdZono> {
    if a.numel() != b.numel() {
        return None;
    }
    let n = a.numel();
    let center: Vec<Dd> = (0..n)
        .map(|i| ny_core::dd::dd_add(a.center[i], b.center[i]))
        .collect();
    // Centers add in double-double; the only new center error is the one
    // dd_add rounding per element, `<= U_DD * (|a| + |b|)`.
    let ec: Vec<f64> = (0..n)
        .map(|i| {
            err_up(
                a.ec[i]
                    + b.ec[i]
                    + ny_core::dd::U_DD * (a.center[i].abs_upper() + b.center[i].abs_upper()),
            )
        })
        .collect();
    // Columns are carried through unchanged (no arithmetic on them), so the
    // generator error channel simply adds with no new rounding term.
    let eg: Vec<f64> = (0..n).map(|i| err_up(a.eg[i] + b.eg[i])).collect();
    let mut gens = Vec::with_capacity(a.n_gens() + b.n_gens());
    gens.extend(a.gens.iter().cloned());
    gens.extend(b.gens.iter().cloned());
    Some(DdZono {
        shape: a.shape.clone(),
        center,
        gens,
        ec,
        eg,
    })
}

/// Interpret an element shape as `(C, H, W)`, stripping leading unit axes.
fn as_chw(shape: &[usize]) -> Option<(usize, usize, usize)> {
    let mut s: Vec<usize> = shape.to_vec();
    while s.len() > 3 && s[0] == 1 {
        s.remove(0);
    }
    match s.len() {
        3 => Some((s[0], s[1], s[2])),
        _ => None,
    }
}

/// Evaluate each objective row on the final zonotope.
///
/// For a row `r`, the certified interval is
///
/// ```text
/// [ r.c - sum_j |r . g_j| - sum_i |r_i| (ec_i + eg_i) - eps,
///   r.c + sum_j |r . g_j| + sum_i |r_i| (ec_i + eg_i) + eps ]
/// ```
///
/// `r . c` is accumulated in double-double (the rows are `+-1` spec rows over
/// at most a few thousand outputs, so `eps` here is far below every other
/// term, but it is carried anyway).
fn evaluate_objectives(
    z: &DdZono,
    objectives: &[Vec<f32>],
    wall: std::time::Duration,
) -> Option<DdZonoMargin> {
    let n = z.numel();
    let (out_lo, out_hi) = z.concretize();
    if out_lo.iter().chain(out_hi.iter()).any(|v| !v.is_finite()) {
        return None;
    }

    let mut lower = Vec::with_capacity(objectives.len());
    let mut upper = Vec::with_capacity(objectives.len());
    let mut rounding = Vec::with_capacity(objectives.len());
    let mut relax = Vec::with_capacity(objectives.len());

    for obj in objectives {
        if obj.len() != n {
            return None;
        }
        let mut mc = Dd::ZERO;
        let mut s_abs = 0.0_f64;
        for (i, &r) in obj.iter().enumerate() {
            let r = f64::from(r);
            if r == 0.0 {
                continue;
            }
            let c = z.center[i];
            mc = ny_core::dd::dd_fma(mc, r, c.hi);
            mc = ny_core::dd::dd_fma(mc, r, c.lo);
            s_abs += r.abs() * c.abs_upper();
        }
        // Relaxation half-width: |r . g_j| summed over columns.
        //
        // `g_abs` accumulates `sum_j sum_i |r_i| |g_ji|`, NOT `sum_j |r . g_j|`:
        // the Higham bound for the f64 row-dot is `gamma_nnz * sum_i |r_i g_ji|`,
        // and that sum can be far larger than the (cancelling) dot itself. Using
        // the dot would under-widen.
        let mut mg = 0.0_f64;
        let mut g_abs = 0.0_f64;
        for g in &z.gens {
            let mut d = 0.0_f64;
            let mut a = 0.0_f64;
            for (i, &r) in obj.iter().enumerate() {
                if r != 0.0 {
                    let r = f64::from(r);
                    d += r * g[i];
                    a += r.abs() * g[i].abs();
                }
            }
            mg += d.abs();
            g_abs += a;
        }
        // Rounding half-width: |r| . (ec + eg), plus this reduction's own
        // rounding and the double-double collapse of `mc`.
        let mut me = 0.0_f64;
        for (i, &r) in obj.iter().enumerate() {
            if r != 0.0 {
                me += f64::from(r).abs() * (z.ec[i] + z.eg[i]);
            }
        }
        let nnz = obj.iter().filter(|v| **v != 0.0).count().max(1);
        let me = err_up(
            me + ny_core::dd::gamma_n_dd(2 * nnz + 2) * s_abs
                + 2.0 * U_F64 * mc.abs_upper()
                + ny_core::dd::gamma_n_f64(nnz + 1) * g_abs,
        );
        let mg = err_up(mg);
        let c = mc.to_f64();
        if !c.is_finite() || !mg.is_finite() || !me.is_finite() {
            return None;
        }
        lower.push(next_down_f64(c - mg - me));
        upper.push(next_up_f64(c + mg + me));
        rounding.push(me);
        relax.push(mg);
    }

    Some(DdZonoMargin {
        lower,
        upper,
        rounding_half_width: rounding,
        relax_half_width: relax,
        output_lower: out_lo.iter().map(|&v| next_down_f32(v as f32)).collect(),
        output_upper: out_hi.iter().map(|&v| next_up_f32(v as f32)).collect(),
        output_shape: z.shape.clone(),
        n_generators: z.n_gens(),
        wall,
    })
}
