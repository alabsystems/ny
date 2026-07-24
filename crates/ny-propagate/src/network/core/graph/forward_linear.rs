// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Forward-linear intermediate bounds for graph/DAG networks.
//! Collect per-node `LinearBounds` relative to the original input, then concretize them
//! back to `BoundedTensor` node bounds. The first packet stays intentionally narrow:
//! support the nn4sys-style DAG operator surface and fail closed instead of degrading to IBP.

pub(crate) mod alpha_opt;
mod binary;
mod concat;
mod image;

use std::borrow::Cow;
use std::collections::HashMap;
use std::time::Instant;

use ndarray::{Array1, Array2, IxDyn};
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::info;

use crate::bounds::LinearBounds;
use crate::layers::{BoundPropagation, Layer};

use super::{GraphNetwork, NETWORK_INPUT};

/// Minimum remaining wall time required to start a cold image forward-linear
/// reference build.
///
/// A CIFAR-sized cold build contains f64 GEMMs that cannot be interrupted once
/// submitted.  The full pass is measured at roughly 22--25 seconds, so starting
/// it inside a 10-second verifier slice can hold a scoped cache warmer until the
/// competition watchdog fires.  Cached hits remain admissible at any deadline;
/// this floor only refuses optional cold work, whose callers already fail closed
/// to IBP/CROWN.  Thirty seconds retains five seconds of safety margin over the
/// slow end of the measured pass.
const FORWARD_LINEAR_COLD_BUILD_MIN_HEADROOM: std::time::Duration =
    std::time::Duration::from_secs(30);

fn forward_linear_cold_build_admitted_at(deadline: Option<Instant>, now: Instant) -> bool {
    deadline
        .is_none_or(|d| d.saturating_duration_since(now) >= FORWARD_LINEAR_COLD_BUILD_MIN_HEADROOM)
}

impl GraphNetwork {
    /// Collect forward-linear intermediate bounds for supported DAG operators.
    ///
    /// Unlike `collect_node_bounds_with_engine`, this preserves affine
    /// correlations with the original input box instead of repeatedly
    /// concretizing to IBP at each node.
    pub fn collect_forward_linear_bounds_dag_with_engine(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<HashMap<String, BoundedTensor>> {
        let (node_bounds, _) = collect_forward_linear_state_dag(
            self,
            input,
            engine,
            None,
            None,
            Self::forward_linear_conv_transpose_reference_enabled(),
        )?;
        Ok(node_bounds)
    }

    /// Collect forward-linear intermediate bounds for supported DAG operators,
    /// aborting when the deadline is exceeded.
    pub fn collect_forward_linear_bounds_dag_with_engine_and_deadline(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<HashMap<String, BoundedTensor>> {
        let (node_bounds, _) = collect_forward_linear_state_dag(
            self,
            input,
            engine,
            deadline,
            None,
            Self::forward_linear_conv_transpose_reference_enabled(),
        )?;
        Ok(node_bounds)
    }

    /// Alpha-fed variant of
    /// [`Self::collect_forward_linear_bounds_dag_with_engine`]
    /// (#w4-root-alpha): image-mode ReLU nodes present in `relu_alphas` use
    /// the given per-neuron LOWER slopes (clamped to [0, 1], sound intercept
    /// 0 on crossing neurons — see `image::compose_relu_diag_forward`);
    /// absent nodes keep the adaptive rule. Uncached.
    #[cfg(test)]
    pub(crate) fn collect_forward_linear_bounds_dag_with_alphas(
        &self,
        input: &BoundedTensor,
        relu_alphas: &std::collections::BTreeMap<String, Array1<f32>>,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<HashMap<String, BoundedTensor>> {
        let (node_bounds, _) = collect_forward_linear_state_dag(
            self,
            input,
            engine,
            None,
            Some(relu_alphas),
            Self::forward_linear_conv_transpose_reference_enabled(),
        )?;
        Ok(node_bounds)
    }

    /// Test-only direct entry for the dark ConvTranspose image surface.  This
    /// bypasses the production enable flag without mutating process-global env
    /// state (cargo tests run concurrently).
    #[cfg(test)]
    pub(crate) fn collect_forward_linear_bounds_dag_with_conv_transpose_for_test(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<HashMap<String, BoundedTensor>> {
        let (node_bounds, _) =
            collect_forward_linear_state_dag(self, input, engine, None, None, true)?;
        Ok(node_bounds)
    }

    /// Test-only default-compat entry. Forces the dark surface OFF regardless
    /// of the process environment so parallel tests can prove legacy routing.
    #[cfg(test)]
    pub(crate) fn collect_forward_linear_bounds_dag_without_conv_transpose_for_test(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<HashMap<String, BoundedTensor>> {
        let (node_bounds, _) =
            collect_forward_linear_state_dag(self, input, engine, None, None, false)?;
        Ok(node_bounds)
    }

    /// Batteries-included gate for the conv-DAG forward-linear reference-bounds
    /// source (#vnncomp-image-forward-linear): ON by default, opt out with
    /// `NY_NO_FORWARD_LINEAR_REF=1` (disable-flag principle). Shared by the
    /// alpha reference collection, the spec-propagation setup, and the CLI
    /// attack-phase cache warmer (#w5-bab-throughput) so all consult ONE policy.
    pub fn forward_linear_reference_enabled() -> bool {
        !matches!(
            std::env::var("NY_NO_FORWARD_LINEAR_REF").ok().as_deref(),
            Some("1")
        )
    }

    /// Dark enable gate for ConvTranspose2d/BatchNorm image forward-linear
    /// references.  The operator implementation is certified and directly
    /// testable, but exact cGAN row-7 currently improves the final box ~40x
    /// without making the root decisive.  Keep automatic warmers/reference
    /// paths byte-identical until an end-to-end solver A/B proves a gain.
    ///
    /// Set `NY_FORWARD_LINEAR_CONV_TRANSPOSE_REF=1` to opt into the candidate.
    pub fn forward_linear_conv_transpose_reference_enabled() -> bool {
        matches!(
            std::env::var("NY_FORWARD_LINEAR_CONV_TRANSPOSE_REF")
                .ok()
                .as_deref(),
            Some("1")
        )
    }

    /// Whether a cold forward-linear reference build has enough wall-clock
    /// headroom to start.  This is public so the CLI can avoid spawning a
    /// scoped optional warmer that the cache implementation would immediately
    /// refuse.  A warm cache is checked before this admission gate.
    pub fn forward_linear_cold_build_admitted(deadline: Option<Instant>) -> bool {
        forward_linear_cold_build_admitted_at(deadline, Instant::now())
    }

    /// Cached variant of
    /// [`Self::collect_forward_linear_bounds_dag_with_engine_and_deadline`]:
    /// single-entry cache keyed by a bit-exact hash of the input bounds (see
    /// [`super::ForwardLinearMapCache`]). The root input recurs across the PGD
    /// spec-CROWN prechecks, the alpha reference collection, and the
    /// spec-propagation setup — each paid the full O(L) certified pass (~22s
    /// on cifar100 release) before this cache.
    ///
    /// Errors (deadline/unsupported/mem-cap) are NOT cached: a later call with
    /// more budget may succeed.
    pub fn collect_forward_linear_bounds_dag_cached(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<std::sync::Arc<HashMap<String, BoundedTensor>>> {
        Ok(self
            .collect_forward_linear_state_cached(input, engine, deadline)?
            .0)
    }

    /// Shared cached forward-linear state: the concretized per-node bounds map
    /// plus the OUTPUT node's certified `LinearBounds` (#w4-root-margin) when
    /// retained by the pass. One O(L) certified computation per root input.
    #[allow(clippy::type_complexity)]
    pub(crate) fn collect_forward_linear_state_cached(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<(
        std::sync::Arc<HashMap<String, BoundedTensor>>,
        Option<std::sync::Arc<LinearBounds>>,
    )> {
        let allow_conv_transpose = Self::forward_linear_conv_transpose_reference_enabled();
        let key = input_bits_hash(input, None)
            ^ if allow_conv_transpose {
                0xC6A4_2025_C0DE_0001
            } else {
                0
            };

        if let Ok(guard) = self.cached_forward_linear_map.fixed.read() {
            if let Some((cached_key, map, output_lb, _)) = guard.as_ref() {
                if *cached_key == key {
                    return Ok((std::sync::Arc::clone(map), output_lb.clone()));
                }
            }
        }

        if !Self::forward_linear_cold_build_admitted(deadline) {
            return Err(NyError::DeadlineExceeded(format!(
                "forward-linear cold build requires at least {}s headroom",
                FORWARD_LINEAR_COLD_BUILD_MIN_HEADROOM.as_secs()
            )));
        }

        let build_start = Instant::now();
        let (map, output_lb) =
            self.collect_forward_linear_state_fresh(input, engine, deadline, None)?;
        let build_cost = build_start.elapsed();
        if let Ok(mut guard) = self.cached_forward_linear_map.fixed.write() {
            *guard = Some((
                key,
                std::sync::Arc::clone(&map),
                output_lb.clone(),
                build_cost,
            ));
        }
        Ok((map, output_lb))
    }

    /// Alpha-fed variant of [`Self::collect_forward_linear_state_cached`]
    /// (#w4-root-alpha): the image-mode diagonal ReLU compositions use the
    /// given per-neuron lower slopes (the warmup's optimized alphas). Cached
    /// in a SEPARATE single-entry slot whose key includes a bit-exact
    /// fingerprint of the alpha map, so the fixed-slope entry is never
    /// clobbered and a stale alpha map can never be served.
    #[allow(clippy::type_complexity)]
    pub(crate) fn collect_forward_linear_state_cached_with_alphas(
        &self,
        input: &BoundedTensor,
        relu_alphas: &std::collections::BTreeMap<String, Array1<f32>>,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<(
        std::sync::Arc<HashMap<String, BoundedTensor>>,
        Option<std::sync::Arc<LinearBounds>>,
    )> {
        let allow_conv_transpose = Self::forward_linear_conv_transpose_reference_enabled();
        let key = input_bits_hash(input, Some(relu_alphas))
            ^ if allow_conv_transpose {
                0xC6A4_2025_C0DE_0001
            } else {
                0
            };

        if let Ok(guard) = self.cached_forward_linear_map.alpha.read() {
            if let Some((cached_key, map, output_lb, _)) = guard.as_ref() {
                if *cached_key == key {
                    return Ok((std::sync::Arc::clone(map), output_lb.clone()));
                }
            }
        }

        let build_start = Instant::now();
        let (map, output_lb) =
            self.collect_forward_linear_state_fresh(input, engine, deadline, Some(relu_alphas))?;
        let build_cost = build_start.elapsed();
        if let Ok(mut guard) = self.cached_forward_linear_map.alpha.write() {
            *guard = Some((
                key,
                std::sync::Arc::clone(&map),
                output_lb.clone(),
                build_cost,
            ));
        }
        Ok((map, output_lb))
    }

    /// Run one full forward-linear pass and split off the OUTPUT node's
    /// retained `LinearBounds` (the margin composition seed).
    #[allow(clippy::type_complexity)]
    fn collect_forward_linear_state_fresh(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
        relu_alphas: Option<&std::collections::BTreeMap<String, Array1<f32>>>,
    ) -> Result<(
        std::sync::Arc<HashMap<String, BoundedTensor>>,
        Option<std::sync::Arc<LinearBounds>>,
    )> {
        let (node_bounds, mut linear_map) = collect_forward_linear_state_dag(
            self,
            input,
            engine,
            deadline,
            relu_alphas,
            Self::forward_linear_conv_transpose_reference_enabled(),
        )?;
        // The output node's affine map w.r.t. the original input — the margin
        // composition seed. Retained by the pass (the output has no consumer,
        // so image-mode liveness never frees it).
        let output_name = if self.output_node.is_empty() {
            self.topological_sort()?.last().cloned().unwrap_or_default()
        } else {
            self.output_node.clone()
        };
        let output_lb = linear_map.remove(&output_name).map(std::sync::Arc::new);
        Ok((std::sync::Arc::new(node_bounds), output_lb))
    }

    /// Certified spec-margin bounds from the forward-linear output map
    /// (#w4-root-margin): compose the spec matrix `C` (an exact affine map, no
    /// bias) with the output node's certified forward-linear `LinearBounds`
    /// using the SAME certified dense-affine composition the pass uses for
    /// Gemm layers (f64 GEMM + outward coefficient-cast gap + γ·S discharge),
    /// then sound-concretize on the input box.
    ///
    /// This keeps the CROSS-OUTPUT correlation that the per-logit projection
    /// destroys: a margin row `e_i − e_j` composes to `w_i − w_j` coefficient
    /// CANCELLATION before concretization, instead of `lower_i − upper_j`
    /// interval subtraction after. Measured on cifar100 prop_idx_7641 this is
    /// the difference between obj[0] = −23.85 (projection) and a decidable
    /// root bound.
    ///
    /// Errors mirror the forward-linear reference collection: refusal classes
    /// (unsupported op / deadline / memory cap) surface as their `NyError`s so
    /// the caller can fail closed to the CPU spec loop.
    pub(crate) fn forward_linear_spec_margin_bounds(
        &self,
        input: &BoundedTensor,
        spec_matrix: &Array2<f32>,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<BoundedTensor> {
        let (_, output_lb) = self.collect_forward_linear_state_cached(input, engine, deadline)?;
        compose_spec_margin(input, spec_matrix, output_lb.as_deref(), engine)
    }

    /// Alpha-fed variant of [`Self::forward_linear_spec_margin_bounds`]
    /// (#w4-root-alpha): the forward-linear map is rebuilt with the given
    /// per-neuron lower ReLU slopes (sound for any α ∈ [0, 1] —
    /// see `image::compose_relu_diag_forward`), then composed with `C`
    /// through the same certified dense-affine composition. The result is a
    /// sound enclosure of the same spec values as the fixed-slope route, so
    /// callers may intersect the two element-wise. Production traffic goes
    /// through [`Self::forward_linear_alpha_optimized_spec_margin_bounds`];
    /// this direct variant remains for the soundness test suite.
    #[cfg(test)]
    pub(crate) fn forward_linear_spec_margin_bounds_with_alphas(
        &self,
        input: &BoundedTensor,
        spec_matrix: &Array2<f32>,
        relu_alphas: &std::collections::BTreeMap<String, Array1<f32>>,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<BoundedTensor> {
        let (_, output_lb) = self.collect_forward_linear_state_cached_with_alphas(
            input,
            relu_alphas,
            engine,
            deadline,
        )?;
        compose_spec_margin(input, spec_matrix, output_lb.as_deref(), engine)
    }

    /// Measured wall cost of the cached fixed-slope forward-linear pass for
    /// THIS input (`None` when the cache is cold). The alpha-fed rebuild
    /// costs the same O(L) pass, so this is the budget quantum the root
    /// warmup cap and the optimizer's self-budgeting both consult
    /// (#w4-root-alpha-opt).
    pub(crate) fn forward_linear_fixed_pass_cost(
        &self,
        input: &BoundedTensor,
    ) -> Option<std::time::Duration> {
        self.forward_linear_fixed_state_if_cached(input)
            .map(|(.., cost)| cost)
    }

    /// Cached fixed-slope forward-linear state for THIS input — `None` when
    /// the cache is cold (#w4-root-alpha-opt: the optimizer must never pay
    /// the fresh O(L) pass itself; it only runs where the fixed pass already
    /// did, i.e. on the root input).
    #[allow(clippy::type_complexity)]
    fn forward_linear_fixed_state_if_cached(
        &self,
        input: &BoundedTensor,
    ) -> Option<(
        std::sync::Arc<HashMap<String, BoundedTensor>>,
        Option<std::sync::Arc<LinearBounds>>,
        std::time::Duration,
    )> {
        let key = input_bits_hash(input, None);
        let guard = self.cached_forward_linear_map.fixed.read().ok()?;
        guard
            .as_ref()
            .filter(|(cached_key, ..)| *cached_key == key)
            .map(|(_, map, output_lb, cost)| (std::sync::Arc::clone(map), output_lb.clone(), *cost))
    }

    /// Forward-map ALPHA OPTIMIZER + certified rebuild (#w4-root-alpha-opt):
    /// optimize per-neuron lower ReLU slopes against the C-margin objective of
    /// the unverified spec rows (see [`alpha_opt`] module docs), then rebuild
    /// the forward-linear map ONCE with the optimized alphas through the
    /// certified machinery and compose the margin. Returns `Ok(None)` when the
    /// fixed cache is cold for this input, the headroom cannot fit the
    /// rebuild, or the optimizer finds no predicted improvement (in which case
    /// the ~`fixed_cost` rebuild is skipped entirely and the budget returns to
    /// BaB).
    ///
    /// Soundness: the returned bounds come from the same certified alpha-fed
    /// pass as any other alpha map (sound for any α ∈ [0, 1]); the optimizer
    /// itself never touches the verdict path.
    pub(crate) fn forward_linear_alpha_optimized_spec_margin_bounds(
        &self,
        input: &BoundedTensor,
        spec_matrix: &Array2<f32>,
        current_lower: Option<&BoundedTensor>,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<Option<(BoundedTensor, alpha_opt::AlphaOptStats)>> {
        use std::time::Duration;

        // Root-class requests only: the 1-row spec calls on the root input are
        // the PGD margin PRECHECKS (many per instance, each with a distinct C
        // row, so the memo below cannot amortize them) — running sweeps plus a
        // ~20s certified rebuild per precheck would eat the attack phase. The
        // multi-row root C-matrix is the single call this lever exists for.
        if spec_matrix.nrows() < 2 {
            return Ok(None);
        }

        let Some((map, output_lb, fixed_cost)) = self.forward_linear_fixed_state_if_cached(input)
        else {
            tracing::debug!(
                "forward-linear alpha-opt: fixed cache cold for this input, skipping (#w4-root-alpha-opt)"
            );
            return Ok(None);
        };
        let Some(output_lb) = output_lb else {
            return Ok(None);
        };

        // Memo: one optimizer run per (input, spec matrix). Root re-bounds of
        // the same request must not re-pay the sweeps or the rebuild.
        let memo_key = margin_opt_memo_key(input, spec_matrix);
        if let Ok(guard) = self.cached_forward_linear_map.alpha_opt.read() {
            if let Some((cached_key, memo)) = guard.as_ref() {
                if *cached_key == memo_key {
                    return match memo {
                        None => Ok(None),
                        Some((alphas, stats)) => {
                            // The alpha cache slot is keyed by (input, alphas) —
                            // warm after the first run, so this is cheap.
                            let (_, alpha_lb) = self
                                .collect_forward_linear_state_cached_with_alphas(
                                    input, alphas, engine, deadline,
                                )?;
                            let bounds = compose_spec_margin(
                                input,
                                spec_matrix,
                                alpha_lb.as_deref(),
                                engine,
                            )?;
                            Ok(Some((bounds, *stats)))
                        }
                    };
                }
            }
        }
        let memoize = |value: Option<(
            std::sync::Arc<std::collections::BTreeMap<String, Array1<f32>>>,
            alpha_opt::AlphaOptStats,
        )>| {
            if let Ok(mut guard) = self.cached_forward_linear_map.alpha_opt.write() {
                *guard = Some((memo_key, value));
            }
        };

        // Self-budgeting (#w4-root-alpha): the certified rebuild costs the
        // same O(L) pass as the fixed map (measured via the cache entry).
        // Reserve it with margin; give the optimizer a bounded slice of the
        // rest. When even the rebuild does not fit, skip everything (the
        // fixed-slope candidates stand).
        const OPT_FLOOR: Duration = Duration::from_millis(1500);
        const OPT_CAP: Duration = Duration::from_secs(12);
        let now = Instant::now();
        let rebuild_reserve = fixed_cost.mul_f64(1.15);
        let opt_budget = match deadline.map(|d| d.saturating_duration_since(now)) {
            Some(remaining) => {
                if remaining < rebuild_reserve + OPT_FLOOR {
                    tracing::info!(
                        headroom_ms = remaining.as_millis() as u64,
                        rebuild_reserve_ms = rebuild_reserve.as_millis() as u64,
                        "forward-linear alpha-opt: skipping (insufficient headroom, #w4-root-alpha-opt)"
                    );
                    return Ok(None);
                }
                // Cannot underflow: the guard above ensures
                // `remaining >= rebuild_reserve + OPT_FLOOR`.
                remaining
                    .saturating_sub(rebuild_reserve)
                    .min(remaining.mul_f32(0.35))
                    .min(OPT_CAP)
                    .max(OPT_FLOOR)
            }
            None => OPT_CAP,
        };
        let opt_deadline = Some(now + opt_budget);

        match alpha_opt::optimize_margin_alphas(
            self,
            input,
            spec_matrix,
            current_lower,
            &map,
            &output_lb,
            engine,
            opt_deadline,
        )? {
            None => {
                memoize(None);
                Ok(None)
            }
            Some((alphas, stats)) => {
                let alphas = std::sync::Arc::new(alphas);
                let (_, alpha_lb) = self.collect_forward_linear_state_cached_with_alphas(
                    input, &alphas, engine, deadline,
                )?;
                let bounds = compose_spec_margin(input, spec_matrix, alpha_lb.as_deref(), engine)?;
                memoize(Some((alphas, stats)));
                Ok(Some((bounds, stats)))
            }
        }
    }
}

/// Compose the spec matrix `C` with the OUTPUT node's certified
/// forward-linear map and sound-concretize on the input box (shared by the
/// fixed-slope and alpha-fed margin routes).
fn compose_spec_margin(
    input: &BoundedTensor,
    spec_matrix: &Array2<f32>,
    output_lb: Option<&LinearBounds>,
    engine: Option<&dyn GemmEngine>,
) -> Result<BoundedTensor> {
    let Some(output_lb) = output_lb else {
        return Err(NyError::UnsupportedConfiguration(
            "forward-linear spec margin: output linear map not retained".to_string(),
        ));
    };
    if spec_matrix.ncols() != output_lb.num_outputs() {
        return Err(NyError::shape_mismatch(
            vec![output_lb.num_outputs()],
            vec![spec_matrix.ncols()],
        ));
    }
    // Worst-case input magnitude per coordinate (the certified-error
    // discharge weights), exactly as the forward pass computes them.
    let input_flat = input.flatten();
    let input_mag: Vec<f64> = input_flat
        .lower()
        .iter()
        .zip(input_flat.upper().iter())
        .map(|(&l, &u)| f64::from(l).abs().max(f64::from(u).abs()))
        .collect();
    let composed = image::compose_dense_affine_forward(
        "spec-margin",
        spec_matrix,
        None,
        output_lb,
        &input_mag,
        engine,
        None,
    )?;
    composed
        .concretize_checked(input)?
        .reshape(&[spec_matrix.nrows()])
}

/// Bit-exact cache key: the input box bits plus (when present) a fingerprint
/// of the per-node alpha vectors (#w4-root-alpha). f32 bits are hashed
/// directly, so any numeric change — including −0.0 vs 0.0 or NaN payloads —
/// produces a different key.
fn input_bits_hash(
    input: &BoundedTensor,
    relu_alphas: Option<&std::collections::BTreeMap<String, Array1<f32>>>,
) -> u64 {
    use std::hash::Hasher;
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for v in input.lower().iter().chain(input.upper().iter()) {
        hasher.write_u32(v.to_bits());
    }
    hasher.write_usize(input.lower().len());
    if let Some(alphas) = relu_alphas {
        hasher.write_usize(alphas.len());
        for (name, alpha) in alphas {
            hasher.write(name.as_bytes());
            hasher.write_usize(alpha.len());
            for v in alpha.iter() {
                hasher.write_u32(v.to_bits());
            }
        }
    }
    hasher.finish()
}

/// Memo key for the forward-map alpha optimizer (#w4-root-alpha-opt): the
/// input box bits plus the spec-matrix bits.
fn margin_opt_memo_key(input: &BoundedTensor, spec_matrix: &Array2<f32>) -> u64 {
    use std::hash::Hasher;
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    hasher.write_u64(input_bits_hash(input, None));
    hasher.write_usize(spec_matrix.nrows());
    hasher.write_usize(spec_matrix.ncols());
    for v in spec_matrix.iter() {
        hasher.write_u32(v.to_bits());
    }
    hasher.finish()
}

fn collect_forward_linear_state_dag(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
    relu_alphas: Option<&std::collections::BTreeMap<String, Array1<f32>>>,
    allow_conv_transpose: bool,
) -> Result<(
    HashMap<String, BoundedTensor>,
    HashMap<String, LinearBounds>,
)> {
    let exec_order = graph.topological_sort()?;

    // Image mode (#vnncomp-image-forward-linear): conv DAGs route through the
    // certified compositions in `image.rs` (Conv2d / ConvTranspose2d /
    // BatchNorm / diagonal ReLU / Add / Linear / shape pass-through). The
    // generic identity-trick path below is
    // O(N²) memory per activation node — infeasible at image scale — and
    // never supported Conv2d, so conv graphs previously always failed closed.
    // Graphs WITHOUT a 2-D convolution keep the legacy path byte-identical.
    let has_conv2d = exec_order.iter().any(|name| {
        graph
            .nodes
            .get(name)
            .is_some_and(|node| matches!(node.layer, Layer::Conv2d(_)))
    });
    let has_conv_transpose = exec_order.iter().any(|name| {
        graph
            .nodes
            .get(name)
            .is_some_and(|node| matches!(node.layer, Layer::ConvTranspose2d(_)))
    });
    let image_mode = has_conv2d || (allow_conv_transpose && has_conv_transpose);
    if image_mode {
        // Fail closed BEFORE any expensive work if the graph leaves the
        // certified image op surface: the caller falls back to plain IBP.
        for name in &exec_order {
            if let Some(node) = graph.nodes.get(name) {
                if !image_mode_supported(&node.layer, allow_conv_transpose) {
                    return Err(unsupported_forward_linear_node(
                        name,
                        &node.layer,
                        "operator is outside the certified image forward-linear surface",
                    ));
                }
            }
        }
    }

    // #w4-root-alpha-opt profile: which parts of this pass are alpha-
    // independent (cacheable across alpha-fed rebuilds) vs alpha-dependent.
    // The IBP prepass never depends on alpha; every coefficient composition
    // downstream of the first crossing ReLU does (the ReLU diagonal feeds the
    // conv im2col+GEMM inputs), so it must be re-done per alpha map.
    let pass_start = Instant::now();
    let ibp_node_bounds =
        graph.collect_node_bounds_with_engine_and_deadline(input, engine, deadline)?;
    let ibp_elapsed = pass_start.elapsed();
    let mut profile = image_mode.then(ForwardLinearProfile::default);
    let input_dim = input.len();

    if image_mode {
        // Dense-coefficient memory guard: each node carries two f32 matrices
        // of `node_numel × input_dim`. Refuse (fail closed to IBP) when the
        // largest exceeds ~128M entries (512 MB per matrix) — cifar100-scale
        // (16384×3072 ≈ 50M) passes; tinyimagenet-scale (12288-dim inputs)
        // stays on its existing IBP gate until column-block streaming lands.
        const MAX_COEFF_ENTRIES: usize = 1 << 27;
        let max_numel = ibp_node_bounds.values().map(|b| b.len()).max().unwrap_or(0);
        if max_numel.saturating_mul(input_dim) > MAX_COEFF_ENTRIES {
            return Err(NyError::UnsupportedConfiguration(format!(
                "forward-linear image bounds: coefficient state {max_numel}x{input_dim} exceeds \
                 the dense memory cap ({MAX_COEFF_ENTRIES} entries)"
            )));
        }
    }

    // max(|x_l|, |x_u|) per input coordinate: the worst-case input magnitude
    // used to discharge certified coefficient errors into the bias (image mode).
    let input_flat = input.flatten();
    let input_mag: Vec<f64> = input_flat
        .lower()
        .iter()
        .zip(input_flat.upper().iter())
        .map(|(&l, &u)| (l as f64).abs().max((u as f64).abs()))
        .collect();

    // Liveness: last consumer index per node, so image mode can drop dense
    // coefficient matrices as soon as no downstream node needs them.
    let mut last_use: HashMap<&str, usize> = HashMap::new();
    for (t, name) in exec_order.iter().enumerate() {
        if let Some(node) = graph.nodes.get(name) {
            for input_name in &node.inputs {
                last_use.insert(input_name.as_str(), t);
            }
        }
    }

    let mut node_bounds = HashMap::with_capacity(exec_order.len());
    let mut linear_bounds = HashMap::with_capacity(exec_order.len());

    for (exec_idx, node_name) in exec_order.iter().enumerate() {
        if deadline.is_some_and(|d| Instant::now() >= d) {
            return Err(NyError::DeadlineExceeded(format!(
                "Graph forward-linear: deadline exceeded before node '{node_name}'"
            )));
        }
        let node = graph.nodes.get(node_name).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "forward-linear bounds: unknown node '{node_name}' in execution order"
            ))
        })?;
        let output_shape = ibp_node_bounds
            .get(node_name)
            .ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "forward-linear bounds: missing IBP output shape for node '{node_name}'"
                ))
            })?
            .shape()
            .to_vec();
        let output_dim = ibp_node_bounds[node_name].len();

        let compose_started = Instant::now();
        let node_linear = if image_mode {
            compose_image_node(
                node_name,
                &node.layer,
                &node.inputs,
                output_dim,
                &linear_bounds,
                &node_bounds,
                &ibp_node_bounds,
                input,
                input_dim,
                &input_mag,
                engine,
                deadline,
                relu_alphas,
            )?
        } else {
            match &node.layer {
                Layer::Concat(layer) => concat::compose_concat_forward(
                    node_name,
                    layer,
                    &node.inputs,
                    output_dim,
                    &linear_bounds,
                    &ibp_node_bounds,
                    input,
                    input_dim,
                )?,
                // Match auto_LiRPA's forward-mode BoundMul middle relaxation for
                // graph warmup so forward+crown uses the same fixed bilinear
                // interpolation on non-optimized MulBinary nodes.
                // Source: auto_LiRPA/operators/bivariate.py::MulHelper.get_forward_relaxation.
                Layer::MulBinary(_) => binary::compose_binary_forward(
                    node_name,
                    &node.layer,
                    &node.inputs,
                    output_dim,
                    &linear_bounds,
                    &ibp_node_bounds,
                    input,
                    input_dim,
                    |identity, input_a_bounds, input_b_bounds| {
                        crate::layers::MulBinaryLayer.propagate_linear_binary(
                            identity,
                            input_a_bounds,
                            input_b_bounds,
                            crate::MulBinaryRelaxationMode::Middle,
                        )
                    },
                )?,
                Layer::Sub(layer) => binary::compose_binary_forward(
                    node_name,
                    &node.layer,
                    &node.inputs,
                    output_dim,
                    &linear_bounds,
                    &ibp_node_bounds,
                    input,
                    input_dim,
                    |identity, _, _| layer.propagate_linear_binary(identity),
                )?,
                Layer::Div(_) => binary::compose_div_forward(
                    node_name,
                    &node.layer,
                    &node.inputs,
                    output_dim,
                    &linear_bounds,
                    &ibp_node_bounds,
                    input,
                    input_dim,
                )?,
                _ => {
                    if node.inputs.len() != 1 {
                        return Err(unsupported_forward_linear_node(
                            node_name,
                            &node.layer,
                            "only unary nodes and Concat are supported in this packet",
                        ));
                    }

                    let pred_name = node.inputs.first().ok_or_else(|| {
                        NyError::InternalError(
                            "validated unary node must have exactly one input".into(),
                        )
                    })?;
                    let upstream = resolve_upstream_linear_bounds(
                        pred_name,
                        None,
                        &linear_bounds,
                        input_dim,
                        node_name,
                    )?;
                    let layer_name = layer_debug_name(&node.layer);
                    let pre_activation = resolve_pre_activation_bounds(
                        pred_name,
                        &ibp_node_bounds,
                        input,
                        node_name,
                        &layer_name,
                    )?;
                    let local = local_forward_relaxation(
                        node_name,
                        &node.layer,
                        output_dim,
                        Some(pre_activation),
                    )?;
                    compose_forward_relaxation(&local, &upstream)?
                }
            }
        };

        if let Some(profile) = profile.as_mut() {
            profile.record(&node.layer, compose_started.elapsed());
        }

        let concretize_started = Instant::now();
        let concretized = concretize_to_node_shape(&node_linear, input, &output_shape, node_name)?;
        // Intersect element-wise with IBP: forward-linear preserves correlations but may be
        // looser per element after nonlinear relaxations (e.g., ReLU triangle).
        let ibp_bounds = &ibp_node_bounds[node_name];
        let tightened = if concretized.shape() == ibp_bounds.shape() {
            tighten_with_ibp(&concretized, ibp_bounds)
        } else {
            concretized
        };
        if let Some(profile) = profile.as_mut() {
            profile.concretize += concretize_started.elapsed();
        }
        node_bounds.insert(node_name.clone(), tightened);
        linear_bounds.insert(node_name.clone(), node_linear);

        // Image mode: free dense coefficient matrices whose consumers have all
        // executed (the returned linear map is unused by both public wrappers;
        // node_bounds keeps the concretized per-node boxes).
        if image_mode {
            for input_name in &node.inputs {
                if last_use.get(input_name.as_str()) == Some(&exec_idx) {
                    linear_bounds.remove(input_name);
                }
            }
        }
    }

    if let Some(profile) = profile {
        info!(
            total_ms = pass_start.elapsed().as_millis() as u64,
            ibp_prepass_ms = ibp_elapsed.as_millis() as u64,
            conv_ms = profile.conv.as_millis() as u64,
            relu_ms = profile.relu.as_millis() as u64,
            add_ms = profile.add.as_millis() as u64,
            linear_ms = profile.linear.as_millis() as u64,
            shape_ms = profile.shape.as_millis() as u64,
            concretize_ms = profile.concretize.as_millis() as u64,
            alpha_fed = relu_alphas.is_some(),
            "forward-linear image pass profile (#w4-root-alpha-opt): only the IBP prepass is alpha-independent"
        );
    }

    Ok((node_bounds, linear_bounds))
}

/// Per-op-class wall-time accumulator for the image pass (#w4-root-alpha-opt
/// profile): answers which fraction of an alpha-fed rebuild re-does
/// alpha-independent work.
#[derive(Default)]
struct ForwardLinearProfile {
    conv: std::time::Duration,
    relu: std::time::Duration,
    add: std::time::Duration,
    linear: std::time::Duration,
    shape: std::time::Duration,
    concretize: std::time::Duration,
}

impl ForwardLinearProfile {
    fn record(&mut self, layer: &Layer, elapsed: std::time::Duration) {
        match layer {
            Layer::Conv2d(_) | Layer::ConvTranspose2d(_) => self.conv += elapsed,
            Layer::ReLU(_) => self.relu += elapsed,
            Layer::Add(_) => self.add += elapsed,
            Layer::Linear(_) => self.linear += elapsed,
            _ => self.shape += elapsed,
        }
    }
}

/// Certified image op surface (#vnncomp-image-forward-linear): the conv-DAG
/// allowlist. Anything else fails closed (caller falls back to plain IBP).
fn image_mode_supported(layer: &Layer, allow_conv_transpose: bool) -> bool {
    matches!(
        layer,
        Layer::Conv2d(_)
            | Layer::ReLU(_)
            | Layer::Add(_)
            | Layer::Linear(_)
            | Layer::Flatten(_)
            | Layer::Reshape(_)
            | Layer::Squeeze(_)
            | Layer::Unsqueeze(_)
    ) || (allow_conv_transpose && matches!(layer, Layer::ConvTranspose2d(_) | Layer::BatchNorm(_)))
}

/// Resolve an upstream forward-linear map without cloning stored matrices
/// (image-scale coefficient state is 100s of MB per node).
fn resolve_upstream_linear_ref<'a>(
    input_name: &str,
    forward_bounds: &'a HashMap<String, LinearBounds>,
    input_dim: usize,
    node_name: &str,
) -> Result<Cow<'a, LinearBounds>> {
    if input_name == NETWORK_INPUT {
        return Ok(Cow::Owned(LinearBounds::identity(input_dim)));
    }
    forward_bounds
        .get(input_name)
        .map(Cow::Borrowed)
        .ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "forward-linear bounds: node '{node_name}' references unknown upstream input '{input_name}'"
            ))
        })
}

/// Resolve the tightened running bounds (forward∩IBP) for a predecessor —
/// the pre-activation source for image-mode relaxations. Falls back to the
/// IBP prepass map only when the running map has no entry (never happens in
/// topological order, kept as a sound fallback).
fn resolve_running_bounds<'a>(
    pred_name: &str,
    running_bounds: &'a HashMap<String, BoundedTensor>,
    ibp_node_bounds: &'a HashMap<String, BoundedTensor>,
    input: &'a BoundedTensor,
    node_name: &str,
) -> Result<&'a BoundedTensor> {
    if pred_name == NETWORK_INPUT {
        return Ok(input);
    }
    running_bounds
        .get(pred_name)
        .or_else(|| ibp_node_bounds.get(pred_name))
        .ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "forward-linear bounds: node '{node_name}' is missing predecessor bounds for '{pred_name}'"
            ))
        })
}

/// Route one node through the certified image compositions (#vnncomp-image-
/// forward-linear). Every concretization downstream goes through
/// `concretize_sound`; every rounding inside these compositions is certified
/// and discharged outward (see `image.rs` module docs).
#[allow(clippy::too_many_arguments)]
fn compose_image_node(
    node_name: &str,
    layer: &Layer,
    inputs: &[String],
    output_dim: usize,
    linear_bounds: &HashMap<String, LinearBounds>,
    node_bounds: &HashMap<String, BoundedTensor>,
    ibp_node_bounds: &HashMap<String, BoundedTensor>,
    input: &BoundedTensor,
    input_dim: usize,
    input_mag: &[f64],
    engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
    relu_alphas: Option<&std::collections::BTreeMap<String, Array1<f32>>>,
) -> Result<LinearBounds> {
    let single_input = |layer: &Layer| -> Result<&str> {
        if inputs.len() == 1 {
            Ok(inputs[0].as_str())
        } else {
            Err(unsupported_forward_linear_node(
                node_name,
                layer,
                "expected exactly one input",
            ))
        }
    };

    match layer {
        Layer::Conv2d(conv) => {
            let pred = single_input(layer)?;
            let upstream = resolve_upstream_linear_ref(pred, linear_bounds, input_dim, node_name)?;
            let pred_shape =
                resolve_input_shape(pred, None, None, ibp_node_bounds, input, node_name)?;
            image::compose_conv2d_forward(
                node_name,
                conv,
                &upstream,
                &pred_shape,
                output_dim,
                input_mag,
                engine,
                deadline,
                None,
            )
        }
        Layer::ConvTranspose2d(conv) => {
            let pred = single_input(layer)?;
            let upstream = resolve_upstream_linear_ref(pred, linear_bounds, input_dim, node_name)?;
            let pred_shape =
                resolve_input_shape(pred, None, None, ibp_node_bounds, input, node_name)?;
            image::compose_conv_transpose2d_forward(
                node_name,
                conv,
                &upstream,
                &pred_shape,
                output_dim,
                input_mag,
                engine,
            )
        }
        Layer::BatchNorm(batch_norm) => {
            let pred = single_input(layer)?;
            let upstream = resolve_upstream_linear_ref(pred, linear_bounds, input_dim, node_name)?;
            let pre_activation =
                resolve_running_bounds(pred, node_bounds, ibp_node_bounds, input, node_name)?;
            image::compose_batch_norm_forward(
                node_name,
                batch_norm,
                &upstream,
                pre_activation,
                output_dim,
                input_mag,
            )
        }
        Layer::ReLU(_) => {
            let pred = single_input(layer)?;
            let upstream = resolve_upstream_linear_ref(pred, linear_bounds, input_dim, node_name)?;
            // Pre-activation from the RUNNING tightened map (forward∩IBP), not
            // the exploding raw-IBP prepass — this is what keeps relaxation
            // slopes sane on deep conv stacks (design step 1d).
            let pre_activation =
                resolve_running_bounds(pred, node_bounds, ibp_node_bounds, input, node_name)?;
            // #w4-root-alpha: optimized per-neuron lower slopes when supplied.
            // Length mismatches fail OPEN to the adaptive rule (sound — the
            // adaptive relaxation is always valid); contiguity is guaranteed
            // for freshly-built Array1 but checked defensively.
            let alpha_lower = relu_alphas
                .and_then(|m| m.get(node_name))
                .filter(|a| a.len() == output_dim)
                .and_then(|a| a.as_slice());
            image::compose_relu_diag_forward(
                node_name,
                &upstream,
                pre_activation,
                input_mag,
                alpha_lower,
            )
        }
        Layer::Add(_) => {
            if inputs.len() != 2 {
                return Err(unsupported_forward_linear_node(
                    node_name,
                    layer,
                    "binary Add must have exactly 2 inputs",
                ));
            }
            let a = resolve_upstream_linear_ref(&inputs[0], linear_bounds, input_dim, node_name)?;
            let b = resolve_upstream_linear_ref(&inputs[1], linear_bounds, input_dim, node_name)?;
            image::compose_add_forward(node_name, &a, &b, input_mag)
        }
        Layer::Linear(linear) => {
            let pred = single_input(layer)?;
            let upstream = resolve_upstream_linear_ref(pred, linear_bounds, input_dim, node_name)?;
            image::compose_dense_affine_forward(
                node_name,
                &linear.weight,
                linear.bias.as_ref(),
                &upstream,
                input_mag,
                engine,
                None,
            )
        }
        Layer::Flatten(_) | Layer::Reshape(_) | Layer::Squeeze(_) | Layer::Unsqueeze(_) => {
            // Pure C-order shape ops: the flattened coefficient layout is
            // unchanged, so the linear map passes through exactly.
            let pred = single_input(layer)?;
            let upstream = resolve_upstream_linear_ref(pred, linear_bounds, input_dim, node_name)?;
            if upstream.num_outputs() != output_dim {
                return Err(NyError::ShapeMismatch {
                    expected: vec![output_dim],
                    got: vec![upstream.num_outputs()],
                });
            }
            Ok(upstream.into_owned())
        }
        _ => Err(unsupported_forward_linear_node(
            node_name,
            layer,
            "operator is outside the certified image forward-linear surface",
        )),
    }
}

fn local_forward_relaxation(
    node_name: &str,
    layer: &Layer,
    output_dim: usize,
    pre_activation: Option<&BoundedTensor>,
) -> Result<LinearBounds> {
    let identity = LinearBounds::identity(output_dim);
    let pre_activation = pre_activation.ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "forward-linear bounds: node '{node_name}' ({}) is missing pre-activation bounds",
            layer_debug_name(layer),
        ))
    })?;

    let result = match layer {
        Layer::Linear(layer) => layer
            .propagate_linear(&identity)
            .map(|bounds| bounds.into_owned()),
        Layer::Conv1d(layer) => layer
            .propagate_linear(&identity)
            .map(|bounds| bounds.into_owned()),
        Layer::AddConstant(layer) => layer
            .propagate_linear(&identity)
            .map(|bounds| bounds.into_owned()),
        Layer::MulConstant(layer) => layer
            .propagate_linear(&identity)
            .map(|bounds| bounds.into_owned()),
        Layer::DivConstant(layer) => layer
            .propagate_linear(&identity)
            .map(|bounds| bounds.into_owned()),
        Layer::SubConstant(layer) => layer
            .propagate_linear(&identity)
            .map(|bounds| bounds.into_owned()),
        Layer::Reshape(layer) => layer
            .propagate_linear(&identity)
            .map(|bounds| bounds.into_owned()),
        Layer::Flatten(layer) => layer
            .propagate_linear(&identity)
            .map(|bounds| bounds.into_owned()),
        Layer::Transpose(layer) => {
            let mut layer = layer.clone();
            layer.set_input_shape(pre_activation.shape().to_vec());
            layer
                .propagate_linear(&identity)
                .map(|bounds| bounds.into_owned())
        }
        Layer::Squeeze(layer) => layer
            .propagate_linear(&identity)
            .map(|bounds| bounds.into_owned()),
        Layer::Unsqueeze(layer) => layer
            .propagate_linear(&identity)
            .map(|bounds| bounds.into_owned()),
        Layer::Slice(layer) => layer.propagate_linear_with_bounds(&identity, pre_activation),
        Layer::Gather(layer) => {
            let mut layer = layer.clone();
            layer.set_input_shape(pre_activation.shape().to_vec());
            layer
                .propagate_linear(&identity)
                .map(|bounds| bounds.into_owned())
        }
        Layer::ReLU(layer) => layer.propagate_linear_with_bounds(&identity, pre_activation),
        Layer::Sigmoid(layer) => layer.propagate_linear_with_bounds(&identity, pre_activation),
        Layer::PowConstant(layer) => layer.propagate_linear_with_bounds(&identity, pre_activation),
        Layer::ReduceSum(layer) => layer.propagate_linear_with_bounds(&identity, pre_activation),
        _ => Err(unsupported_forward_linear_node(
            node_name,
            layer,
            "operator is outside the forward-linear packet surface",
        )),
    };

    result.map_err(|error| wrap_forward_linear_error(node_name, layer, error))
}

fn compose_forward_relaxation(
    local: &LinearBounds,
    upstream: &LinearBounds,
) -> Result<LinearBounds> {
    if local.num_inputs() != upstream.num_outputs() {
        return Err(NyError::ShapeMismatch {
            expected: vec![upstream.num_outputs()],
            got: vec![local.num_inputs()],
        });
    }

    let local_lower_pos = local.lower_a().mapv(|value| value.max(0.0));
    let local_lower_neg = local.lower_a().mapv(|value| value.min(0.0));
    let local_upper_pos = local.upper_a().mapv(|value| value.max(0.0));
    let local_upper_neg = local.upper_a().mapv(|value| value.min(0.0));

    let lower_a = local_lower_pos.dot(upstream.lower_a()) + local_lower_neg.dot(upstream.upper_a());
    let upper_a = local_upper_pos.dot(upstream.upper_a()) + local_upper_neg.dot(upstream.lower_a());

    let lower_b = local_lower_pos.dot(upstream.lower_b())
        + local_lower_neg.dot(upstream.upper_b())
        + local.lower_b();
    let upper_b = local_upper_pos.dot(upstream.upper_b())
        + local_upper_neg.dot(upstream.lower_b())
        + local.upper_b();

    LinearBounds::new_or_conservative(lower_a, lower_b, upper_a, upper_b)
}

fn sum_linear_bounds(parts: &[LinearBounds]) -> Result<LinearBounds> {
    let first = parts.first().ok_or_else(|| {
        NyError::InvalidSpec("forward-linear bounds: empty linear-bounds sum".to_string())
    })?;
    let num_outputs = first.num_outputs();
    let num_inputs = first.num_inputs();

    let mut lower_a = Array2::zeros((num_outputs, num_inputs));
    let mut lower_b = Array1::zeros(num_outputs);
    let mut upper_a = Array2::zeros((num_outputs, num_inputs));
    let mut upper_b = Array1::zeros(num_outputs);

    for part in parts {
        if part.num_outputs() != num_outputs || part.num_inputs() != num_inputs {
            return Err(NyError::ShapeMismatch {
                expected: vec![num_outputs, num_inputs],
                got: vec![part.num_outputs(), part.num_inputs()],
            });
        }
        lower_a += part.lower_a();
        lower_b += part.lower_b();
        upper_a += part.upper_a();
        upper_b += part.upper_b();
    }

    LinearBounds::new_or_conservative(lower_a, lower_b, upper_a, upper_b)
}

fn resolve_upstream_linear_bounds(
    input_name: &str,
    constant_input: Option<&BoundedTensor>,
    forward_bounds: &HashMap<String, LinearBounds>,
    input_dim: usize,
    node_name: &str,
) -> Result<LinearBounds> {
    if input_name == NETWORK_INPUT {
        return Ok(LinearBounds::identity(input_dim));
    }
    if let Some(bounds) = forward_bounds.get(input_name) {
        return Ok(bounds.clone());
    }
    if let Some(constant_input) = constant_input {
        return constant_linear_bounds(constant_input, input_dim);
    }

    Err(NyError::InvalidSpec(format!(
        "forward-linear bounds: node '{node_name}' references unknown upstream input '{input_name}'"
    )))
}

fn resolve_input_shape(
    input_name: &str,
    constant_input: Option<&BoundedTensor>,
    stored_shape: Option<&[usize]>,
    ibp_node_bounds: &HashMap<String, BoundedTensor>,
    input: &BoundedTensor,
    node_name: &str,
) -> Result<Vec<usize>> {
    if input_name == NETWORK_INPUT {
        return Ok(input.shape().to_vec());
    }
    if let Some(bounds) = ibp_node_bounds.get(input_name) {
        return Ok(bounds.shape().to_vec());
    }
    if let Some(constant_input) = constant_input {
        return Ok(constant_input.shape().to_vec());
    }
    if let Some(stored_shape) = stored_shape {
        return Ok(stored_shape.to_vec());
    }

    Err(NyError::InvalidSpec(format!(
        "forward-linear bounds: node '{node_name}' is missing shape metadata for input '{input_name}'"
    )))
}

fn resolve_pre_activation_bounds<'a>(
    pred_name: &str,
    ibp_node_bounds: &'a HashMap<String, BoundedTensor>,
    input: &'a BoundedTensor,
    node_name: &str,
    layer_name: &str,
) -> Result<&'a BoundedTensor> {
    if pred_name == NETWORK_INPUT {
        return Ok(input);
    }
    ibp_node_bounds.get(pred_name).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "forward-linear bounds: node '{node_name}' ({layer_name}) is missing predecessor bounds for '{pred_name}'"
        ))
    })
}

fn constant_linear_bounds(bounds: &BoundedTensor, input_dim: usize) -> Result<LinearBounds> {
    let flat = bounds.flatten();
    LinearBounds::new_or_conservative(
        Array2::zeros((flat.len(), input_dim)),
        Array1::from_iter(flat.lower().iter().copied()),
        Array2::zeros((flat.len(), input_dim)),
        Array1::from_iter(flat.upper().iter().copied()),
    )
}

fn concretize_to_node_shape(
    bounds: &LinearBounds,
    input: &BoundedTensor,
    output_shape: &[usize],
    node_name: &str,
) -> Result<BoundedTensor> {
    // SOUNDNESS (#concretize-soundness-hardening): use the directed-rounding
    // `concretize_sound` (lower rounds toward -∞, upper toward +∞) rather than the
    // plain round-to-nearest `concretize`. These concretized forward-linear node
    // bounds are *intermediate* bounds: they are intersected with IBP via
    // `tighten_with_ibp` (max(lower)/min(upper)) and used to constrain downstream
    // relaxations (e.g. pre-activation bounds feeding ReLU/activation planes), which
    // in turn feed the certified verdict. A round-to-nearest f64→f32 cast can land up
    // to 0.5 ULP *inside* the true range, producing an optimistically narrow
    // intermediate bound; `tighten_with_ibp`'s comment ("both sets are sound")
    // depends on this being a sound over-approximation. `concretize_sound` guarantees
    // it. The forward-linear path is not the hot per-domain tightening loop, so the
    // 1-ULP directed cast has no measurable cost here.
    let flat = bounds.concretize_sound(input);
    let lower = flat
        .lower()
        .clone()
        .into_shape_with_order(IxDyn(output_shape))
        .map_err(|error| {
            NyError::InvalidSpec(format!(
                "forward-linear bounds: reshape lower failed for node '{node_name}': {error}"
            ))
        })?;
    let upper = flat
        .upper()
        .clone()
        .into_shape_with_order(IxDyn(output_shape))
        .map_err(|error| {
            NyError::InvalidSpec(format!(
                "forward-linear bounds: reshape upper failed for node '{node_name}': {error}"
            ))
        })?;

    if lower.iter().all(|value| value.is_finite()) && upper.iter().all(|value| value.is_finite()) {
        BoundedTensor::new(lower, upper)
    } else {
        BoundedTensor::new_allow_infinite(lower, upper)
    }
}

/// Tighten forward-linear bounds by intersecting element-wise with IBP bounds.
/// Both sets are sound, so `max(lower)` / `min(upper)` per element is also sound.
fn tighten_with_ibp(forward: &BoundedTensor, ibp: &BoundedTensor) -> BoundedTensor {
    let mut lower = forward.lower().clone();
    let mut upper = forward.upper().clone();
    for (fl, il) in lower.iter_mut().zip(ibp.lower().iter()) {
        *fl = fl.max(*il);
    }
    for (fu, iu) in upper.iter_mut().zip(ibp.upper().iter()) {
        *fu = fu.min(*iu);
    }
    // Clamp: if intersection is empty on any element, use IBP (always sound).
    for ((l, u), (il, iu)) in lower
        .iter_mut()
        .zip(upper.iter_mut())
        .zip(ibp.lower().iter().zip(ibp.upper().iter()))
    {
        if *l > *u {
            *l = *il;
            *u = *iu;
        }
    }
    if lower.iter().all(|v| v.is_finite()) && upper.iter().all(|v| v.is_finite()) {
        BoundedTensor::new(lower, upper).unwrap_or_else(|_| ibp.clone())
    } else {
        BoundedTensor::new_allow_infinite(lower, upper).unwrap_or_else(|_| ibp.clone())
    }
}

fn wrap_forward_linear_error(node_name: &str, layer: &Layer, error: NyError) -> NyError {
    match error {
        NyError::UnsupportedOp(_) | NyError::UnsupportedConfiguration(_) => {
            unsupported_forward_linear_node(node_name, layer, &error.to_string())
        }
        other => other,
    }
}

fn unsupported_forward_linear_node(node_name: &str, layer: &Layer, reason: &str) -> NyError {
    NyError::UnsupportedConfiguration(format!(
        "forward-linear bounds do not support node '{node_name}' ({}){separator}{reason}",
        layer_debug_name(layer),
        separator = if reason.is_empty() { "" } else { ": " },
    ))
}

fn layer_debug_name(layer: &Layer) -> String {
    let debug = format!("{layer:?}");
    debug.split('(').next().unwrap_or("Unknown").to_string()
}

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_image;
