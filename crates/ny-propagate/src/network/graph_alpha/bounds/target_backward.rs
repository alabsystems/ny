// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::gpu_suffix::{try_finish_target_gpu_suffix_with_pending_input, GpuSuffixPlan};
use super::target_backward_patches::{
    initial_target_crown_bounds_with_override, resolve_preactivation, target_allows_patches_start,
    try_patches_target_step_core,
};
use super::*;
use crate::bounds::LinearBounds;
use crate::network::core::apply_dense_backward_dispatch_result;
use crate::network::CrownMergeAccumulator;
use ndarray::Array2;

/// Environment knob for PRIMARY objective-chunking streaming (#patches-obj-chunk).
///
/// `NY_CROWN_OBJ_CHUNK = C`:
/// - unset or `0` (DEFAULT): DISABLED — the target backward runs in a single pass,
///   byte-for-byte identical to the pre-chunking behavior.
/// - `C > 0`: seed the backward pass in row-chunks of at most `C` objective rows,
///   concretize each chunk independently, and scatter into a pre-sized output.
///   Bound-equivalent to the single pass by row-independence (conv col2im scatter,
///   per-row error bounds, and per-row concretize are all row-local).
const CROWN_OBJ_CHUNK_ENV: &str = "NY_CROWN_OBJ_CHUNK";

/// Hard objective-row cap for deadline-bearing backward walks that cross a
/// ConvTranspose layer.
///
/// ConvTranspose2d's certified coefficient path currently contains several
/// whole-objective GEMMs (including the f64 recomputation) that cannot poll a
/// deadline internally.  Streaming at most 32 independent objective rows per
/// pass gives the existing between-pass deadline poll bounded granularity on
/// cGAN-class generators.  The exact sealed row-7 first target has 4,608 rows
/// and its unchunked backward exceeded 250 s; the cap reduces each unpolled
/// pass's objective-row workload by 144x.  This cap is used only when a caller
/// supplied a deadline and the target ancestry contains ConvTranspose;
/// no-deadline/default execution is byte-identical.
const DEADLINE_CONV_TRANSPOSE_OBJ_CHUNK_ROWS: usize = 32;

/// Hard aggregate heap cap for the opt-in target-to-input linear-certificate
/// API's owned dense payload.
///
/// The normal concrete target backward is intentionally unaffected. The
/// certificate API charges the simultaneously resident full assembly plus a
/// conservative upper bound on one row-chunk's relations/transients. Identity
/// rows stream until that SUM fits; a request is refused before allocation when
/// the assembly plus even one row cannot fit.
const TARGET_INPUT_LINEAR_AGGREGATE_MAX_BYTES: usize = 512 * 1024 * 1024;

#[derive(Clone, Copy)]
struct TargetInputLinearLimits {
    aggregate_bytes: usize,
}

impl TargetInputLinearLimits {
    const PRODUCTION: Self = Self {
        aggregate_bytes: TARGET_INPUT_LINEAR_AGGREGATE_MAX_BYTES,
    };
}

// Both payloads are existing by-value return types. Boxing the concrete arm
// would add a heap allocation to the default path solely for this opt-in API.
#[allow(clippy::large_enum_variant)]
enum TargetBackwardPassResult {
    NoInputContribution,
    Concrete(BoundedTensor),
    InputLinear(LinearBounds),
}

/// Checked worst-case heap payload for the raw assembled input-linear
/// certificate:
/// `{lower,upper}_a`, `{lower,upper}_a_err`, and `{lower,upper}_b`.
fn target_input_linear_raw_bytes(target_dim: usize, input_dim: usize) -> Option<usize> {
    let coeff_entries = target_dim.checked_mul(input_dim)?;
    let coeff_bytes = coeff_entries
        .checked_mul(size_of::<f32>())?
        .checked_mul(4)?;
    let bias_bytes = target_dim.checked_mul(size_of::<f32>())?.checked_mul(2)?;
    coeff_bytes.checked_add(bias_bytes)
}

/// Full raw assembly plus the workspace that remains live while the public
/// wrapper discharges coefficient error over the supplied input box.
fn target_input_linear_fixed_bytes(target_dim: usize, input_dim: usize) -> Option<usize> {
    let raw_bytes = target_input_linear_raw_bytes(target_dim, input_dim)?;
    // `BoundedTensor::flatten` allocates both f32 endpoints. The error fold
    // additionally holds one f64 magnitude per input column.
    let flat_input_bytes = input_dim.checked_mul(size_of::<f32>())?.checked_mul(2)?;
    let fold_magnitude_bytes = input_dim.checked_mul(size_of::<f64>())?;
    raw_bytes
        .checked_add(flat_input_bytes)?
        .checked_add(fold_magnitude_bytes)
}

/// Conservative per-row charge for all relations owned by one capture chunk.
///
/// A relation can carry lower/upper A plus lower/upper coefficient error
/// (`4*f32` per column) and lower/upper bias (`2*f32` per row). Summing every
/// relevant node width bounds a DAG frontier even if every relation coexists.
///
/// The factor six covers the worst merge/conversion phase observed in
/// `CrownMergeAccumulator`: existing and incoming f32 relations, an f64
/// relation (including f64 coefficient-error matrices), two f64 roundoff
/// matrices, and one relation-equivalent reserve for downcast rows or Patches
/// conversion. The dense identity seed (`2*f32*target_dim`) is included
/// explicitly and the larger charge wins.
fn target_input_linear_chunk_row_bytes(
    target_dim: usize,
    relation_cols_sum: usize,
    relation_count: usize,
) -> Option<usize> {
    let relation_coeff = relation_cols_sum
        .checked_mul(size_of::<f32>())?
        .checked_mul(4)?;
    let relation_bias = relation_count
        .checked_mul(size_of::<f32>())?
        .checked_mul(2)?;
    let relation_peak = relation_coeff.checked_add(relation_bias)?.checked_mul(6)?;
    let identity_seed = crate::network::crown_memory::dense_pair_bytes(1, target_dim)?;
    Some(relation_peak.max(identity_seed))
}

/// Choose a deterministic identity-row chunk whose assembly + chunk payload
/// satisfies the hard aggregate cap and the existing deadline-bearing
/// ConvTranspose cap.
fn target_input_linear_chunk_rows(
    target_dim: usize,
    assembly_bytes: usize,
    chunk_row_bytes: usize,
    has_deadline: bool,
    has_conv_transpose: bool,
    aggregate_cap_bytes: usize,
) -> Result<usize> {
    if target_dim == 0 {
        return Err(NyError::InvalidSpec(
            "target input-linear CROWN requires a non-empty target".to_string(),
        ));
    }
    let one_row_peak = assembly_bytes.saturating_add(chunk_row_bytes);
    if one_row_peak > aggregate_cap_bytes {
        return Err(NyError::CpuMemoryExceeded {
            required_bytes: one_row_peak,
            budget_bytes: aggregate_cap_bytes,
            site: "graph-alpha target input-linear aggregate workspace",
        });
    }
    let available = aggregate_cap_bytes - assembly_bytes;
    let rows_by_memory = (available / chunk_row_bytes).max(1).min(target_dim);
    let requested = if rows_by_memory < target_dim {
        rows_by_memory
    } else {
        0
    };
    let effective = effective_target_chunk_size(requested, has_deadline, has_conv_transpose);
    Ok(if effective == 0 {
        target_dim
    } else {
        effective.min(target_dim)
    })
}

/// Parse the objective row-chunk size. Returns 0 (disabled) when unset, empty,
/// non-numeric, or explicitly 0 — preserving the single-pass default.
fn crown_obj_chunk_size() -> usize {
    std::env::var(CROWN_OBJ_CHUNK_ENV)
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .unwrap_or(0)
}

/// Combine the caller/env chunk request with the deadline-safety cap.
/// A zero requested size means "no explicit chunking", but cannot disable the
/// safety cap when a deadline-bearing ConvTranspose walk needs it.
fn effective_target_chunk_size(
    requested: usize,
    has_deadline: bool,
    has_conv_transpose: bool,
) -> usize {
    if !has_deadline || !has_conv_transpose {
        return requested;
    }
    if requested == 0 {
        DEADLINE_CONV_TRANSPOSE_OBJ_CHUNK_ROWS
    } else {
        requested.min(DEADLINE_CONV_TRANSPOSE_OBJ_CHUNK_ROWS)
    }
}

/// Environment knob for the CROWN-IBP sweep's backward-to-nearest-bounded-cut
/// (#crown-cut-segment).
///
/// `NY_CROWN_CUT_SEGMENT = N`:
/// - unset or `0` (DEFAULT): DISABLED — every per-target backward expands its
///   full prefix to the network input, byte-for-byte the prior behavior.
/// - `N > 0`: the CROWN-IBP collector designates every node whose topological
///   index is a multiple of `N` as a CUT node. A per-target backward that
///   reaches a cut node whose bounds THIS sweep already finalized concretizes
///   the accumulated linear relation against that node's bound-box — the same
///   directed-rounding concretization the input-box path uses
///   (`LinearBounds::concretize_sound`) — instead of expanding through the
///   node's prefix. SOUND: the swept box is a valid enclosure of the cut
///   node's reachable set, so a linear relation concretized over it encloses
///   the relation's value for every reachable input; the input box is just
///   the trivial cut. The result is generally LOOSER than the full-prefix
///   backward (the box drops inter-node correlations), never tighter than a
///   valid enclosure. The sweep cost drops from O(n²) prefix steps to
///   ~O(n·N). Only the CROWN-IBP collection sweep passes a cut context; the
///   verdict-shaped α-CROWN output backward always runs full depth.
pub(in crate::network::graph_alpha) const CROWN_CUT_SEGMENT_ENV: &str = "NY_CROWN_CUT_SEGMENT";

/// Parse the cut segment length from the environment. Returns 0 (disabled)
/// when unset, empty, non-numeric, or explicitly 0 — preserving the
/// full-prefix default.
pub(in crate::network::graph_alpha) fn crown_cut_segment_from_env() -> usize {
    parse_crown_cut_segment(std::env::var(CROWN_CUT_SEGMENT_ENV).ok().as_deref())
}

/// Cap on the per-block materialization budget for the blockwise Patches
/// final concretization (#patches-row-range). The effective block budget is
/// `min(cpu_crown_dense_budget_bytes(), this)`: even a user-raised
/// `NY_DENSE_BUDGET_MB` never materializes more than 1 GiB of dense rows at a
/// time on that path (the peak also carries the transient err accumulators,
/// see `PatchesLinearBounds::concretize_sound_chunked`).
const PATCHES_CONCRETIZE_MAX_BLOCK_BYTES: usize = 1 << 30;

/// #patches-row-range: mid-walk densify budget guard. Returns the structured
/// `CpuMemoryExceeded` when a Patches-carried relation's dense pair would
/// exceed `budget_bytes` (or overflows the byte estimate, which saturates to
/// `usize::MAX`); `None` when it fits — or when the estimate itself errors, so
/// the subsequent `into_dense()` surfaces the original (shape) error
/// unchanged. `CpuMemoryExceeded` is mapped to a sound fallback by every CROWN
/// caller (CROWN-IBP collector -> IBP for the target; alpha paths likewise),
/// which is strictly better than aborting the process on a TB-scale
/// allocation (VGG16: 3.2M x 150K rows ~ 1.9 TB per matrix).
fn patches_densify_over_budget(pb: &PatchesLinearBounds, budget_bytes: usize) -> Option<NyError> {
    match pb.dense_pair_bytes() {
        Ok(required) if required > budget_bytes => Some(NyError::CpuMemoryExceeded {
            required_bytes: required,
            budget_bytes,
            site: "graph-alpha target-backward mid-walk patches densify",
        }),
        _ => None,
    }
}

/// Pure parser for [`CROWN_CUT_SEGMENT_ENV`] (unit-testable without touching
/// process env). `None`/empty/non-numeric/`0` all mean disabled.
fn parse_crown_cut_segment(raw: Option<&str>) -> usize {
    raw.and_then(|r| r.trim().parse::<usize>().ok())
        .unwrap_or(0)
}

/// Cut-node context for the CROWN-IBP sweep (#crown-cut-segment,
/// `NY_CROWN_CUT_SEGMENT`). Built once per collection when the gate is on;
/// `None` everywhere else (α-CROWN backward, gate off) keeps the backward walk
/// byte-identical to the pre-cut behavior.
pub(crate) struct CrownCutContext {
    /// Nodes designated as cuts (topological index ≡ 0 mod segment length).
    /// Coverage of every path is NOT required for soundness — the input box
    /// remains the ultimate cut; membership only short-circuits the walk.
    cut_nodes: std::collections::HashSet<String>,
    /// Number of cut concretizations performed (for the sweep info line).
    /// `Cell` suffices: the sweep and its backward walks are single-threaded.
    cuts_used: std::cell::Cell<usize>,
}

impl CrownCutContext {
    pub(in crate::network::graph_alpha) fn new(
        cut_nodes: std::collections::HashSet<String>,
    ) -> Self {
        Self {
            cut_nodes,
            cuts_used: std::cell::Cell::new(0),
        }
    }

    fn is_cut(&self, node_name: &str) -> bool {
        self.cut_nodes.contains(node_name)
    }

    fn record_cut(&self) {
        self.cuts_used.set(self.cuts_used.get() + 1);
    }

    pub(in crate::network::graph_alpha) fn cuts_used(&self) -> usize {
        self.cuts_used.get()
    }
}

/// Whether every endpoint of the box is finite. A cut concretization over a
/// non-finite box would be vacuous (±inf rows); the walk expands through the
/// node instead (fail-open to the exact, tighter behavior).
fn box_is_finite(b: &BoundedTensor) -> bool {
    b.lower().iter().all(|v| v.is_finite()) && b.upper().iter().all(|v| v.is_finite())
}

impl GraphNetwork {
    /// Shared backward CROWN core for per-target CROWN-IBP and α-CROWN.
    /// `collector_patches_override=true` enables patches mode for spatial Conv2d
    /// even in matrix conv_mode — used by the CROWN-IBP collector (#3813).
    ///
    /// `chunk_override` (#cgan-bn11-chunk): explicit objective row-chunk size
    /// that takes precedence over the `NY_CROWN_OBJ_CHUNK` env knob. `None`
    /// preserves the env-driven behavior byte-for-byte (single pass when the
    /// env is unset/0). The CROWN-IBP collector passes `Some(C)` for targets
    /// whose dense identity pair exceeds the CPU dense budget, so they stream
    /// through the bound-equivalent chunked backward instead of degrading to
    /// IBP.
    ///
    /// `cut_ctx` (#crown-cut-segment): optional backward-to-nearest-bounded-cut
    /// context for the CROWN-IBP sweep. `None` (every non-sweep caller and the
    /// default-OFF gate) keeps the backward walk byte-identical; see
    /// [`CROWN_CUT_SEGMENT_ENV`].
    #[allow(clippy::too_many_arguments)] // Backward CROWN dispatch requires all parameters; bundling into a struct would obscure the per-call-site engine threading that #3549 fixes.
    pub(in crate::network::graph_alpha) fn propagate_crown_to_node_core(
        &self,
        input: &BoundedTensor,
        target_node: &str,
        crown_bounds: &std::collections::HashMap<String, BoundedTensor>,
        ibp_bounds: &std::collections::HashMap<String, BoundedTensor>,
        alpha_state: Option<&GraphAlphaState>,
        engine: Option<&dyn ny_core::GemmEngine>,
        label: &str,
        per_node_deadline: Option<std::time::Instant>,
        collector_patches_override: bool,
        chunk_override: Option<usize>,
        cut_ctx: Option<&CrownCutContext>,
    ) -> Result<BoundedTensor> {
        let relevant_nodes = self.ancestors(target_node)?;

        if relevant_nodes.is_empty() {
            return Ok(input.clone());
        }

        // #w4-refresh-deadline: scope the cooperative GPU deadline over this whole
        // per-target backward (the GPU suffix and resnet dispatches below run wide
        // spec-batched / deep resident walks that can only stop BETWEEN batches or
        // layers). Set on the routed backend, always cleared on scope exit; an
        // expired check surfaces as DeadlineExceeded, which every caller handles
        // with a sound reference/IBP fallback.
        let _gpu_deadline_scope =
            crate::sound_gpu_gate::GpuCrownDeadlineScope::set(engine, per_node_deadline);

        let target_bounds = ibp_bounds.get(target_node).ok_or_else(|| {
            NyError::InvalidSpec(format!("Target node {} not in IBP bounds", target_node))
        })?;
        let target_contract = GraphTargetShapeContract::from_bounds(target_node, target_bounds);
        let target_dim = target_contract.flat_dim();
        let input_dim = input.len();

        // PRIMARY (#patches-obj-chunk): objective-chunking streaming, gated OFF
        // by default. When the effective chunk size C > 0 and the seed spans
        // more than one chunk, stream the objective rows in C-row slices and
        // reuse the (objective-independent) GpuSuffixPlan across chunks. When
        // C == 0 we fall through to the single-pass path, which is byte-for-byte
        // the prior behavior. Both paths share the same backward-loop body via
        // `run_target_backward_pass`, so the per-row math is identical.
        //
        // The effective chunk size is the explicit `chunk_override` when given
        // (#cgan-bn11-chunk: the CROWN-IBP collector's memory-budget reroute),
        // otherwise the `NY_CROWN_OBJ_CHUNK` env knob. The chunk decision runs
        // BEFORE seed construction: chunking exists precisely so an over-budget
        // target never materializes its full `[dim x dim]` dense identity pair
        // (6.6 GB for cgan_2023's 28,800-dim BatchNormalization_11).
        let requested_chunk_size = chunk_override.unwrap_or_else(crown_obj_chunk_size);
        let has_conv_transpose = relevant_nodes.iter().any(|name| {
            self.nodes.get(name).is_some_and(|node| {
                matches!(
                    node.layer,
                    Layer::ConvTranspose1d(_) | Layer::ConvTranspose2d(_)
                )
            })
        });
        let chunk_size = effective_target_chunk_size(
            requested_chunk_size,
            per_node_deadline.is_some(),
            has_conv_transpose,
        );
        if chunk_size > 0 && target_dim > chunk_size {
            let allow_patches = target_allows_patches_start(
                self,
                target_node,
                alpha_state,
                &relevant_nodes,
                target_bounds,
                collector_patches_override,
            );
            let gpu_suffix_plan = GpuSuffixPlan::build(
                &relevant_nodes,
                self,
                input,
                crown_bounds,
                ibp_bounds,
                alpha_state,
            );
            return self.propagate_crown_to_node_chunked(
                input,
                target_node,
                crown_bounds,
                ibp_bounds,
                alpha_state,
                engine,
                label,
                per_node_deadline,
                collector_patches_override,
                &relevant_nodes,
                target_bounds,
                &target_contract,
                target_dim,
                input_dim,
                allow_patches,
                &gpu_suffix_plan,
                chunk_size,
                cut_ctx,
            );
        }

        let (allow_patches, initial_bounds) = initial_target_crown_bounds_with_override(
            self,
            target_node,
            alpha_state,
            &relevant_nodes,
            target_bounds,
            &target_contract,
            collector_patches_override,
        );
        let gpu_suffix_plan = // O(N) plan replaces O(N*K) rescans (#4340)
            GpuSuffixPlan::build(&relevant_nodes, self, input, crown_bounds, ibp_bounds, alpha_state);

        // Single-pass: seed the full objective and run one backward pass. The
        // `None` result means no input contribution accumulated, in which case
        // the target bounds pass through unchanged (byte-for-byte the prior
        // `target_bounds.clone()` terminal branch).
        let produced = self.run_target_backward_pass(
            input,
            target_node,
            crown_bounds,
            ibp_bounds,
            alpha_state,
            engine,
            label,
            per_node_deadline,
            collector_patches_override,
            relevant_nodes.as_slice(),
            &target_contract,
            target_dim,
            input_dim,
            allow_patches,
            &gpu_suffix_plan,
            initial_bounds,
            cut_ctx,
        )?;
        if per_node_deadline.is_some_and(|deadline| std::time::Instant::now() >= deadline) {
            return Err(NyError::DeadlineExceeded(format!(
                "{label}: per-node deadline exceeded after backward pass for target '{target_node}'"
            )));
        }
        match produced {
            Some(bounds) => Ok(bounds),
            None => Ok(target_bounds.clone()),
        }
    }

    /// Opt-in target-to-input CROWN certificate.
    ///
    /// Returns identity-seeded target rows in flat target-output order as
    /// `lower_a * input + lower_b <= target <= upper_a * input + upper_b`.
    /// Any certified coefficient error produced by the backward pass is folded
    /// outward into the biases over the exact supplied `input` box before this
    /// public method returns; a residual error channel is rejected fail-closed.
    /// Consequently the returned raw coefficients are valid only on that exact
    /// box and its subsets. They must not be reused on a wider or unrelated box.
    ///
    /// This path is deliberately separate from `propagate_crown_to_node`: the
    /// existing concrete API keeps its bounds-only GPU suffixes and concrete
    /// finalization unchanged. Linear capture runs the same CPU/Patches backward
    /// steps in deterministic identity-row chunks, with a checked hard cap on
    /// the aggregate full assembly plus one conservative chunk payload.
    #[allow(clippy::too_many_arguments)]
    pub fn propagate_crown_input_linear_to_node(
        &self,
        input: &BoundedTensor,
        target_node: &str,
        crown_bounds: &std::collections::HashMap<String, BoundedTensor>,
        ibp_bounds: &std::collections::HashMap<String, BoundedTensor>,
        engine: Option<&dyn ny_core::GemmEngine>,
        deadline: Option<std::time::Instant>,
    ) -> Result<LinearBounds> {
        self.propagate_crown_input_linear_to_node_with_limits(
            input,
            target_node,
            crown_bounds,
            ibp_bounds,
            engine,
            deadline,
            TargetInputLinearLimits::PRODUCTION,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn propagate_crown_input_linear_to_node_with_limits(
        &self,
        input: &BoundedTensor,
        target_node: &str,
        crown_bounds: &std::collections::HashMap<String, BoundedTensor>,
        ibp_bounds: &std::collections::HashMap<String, BoundedTensor>,
        engine: Option<&dyn ny_core::GemmEngine>,
        deadline: Option<std::time::Instant>,
        limits: TargetInputLinearLimits,
    ) -> Result<LinearBounds> {
        let mut linear = self.capture_crown_input_linear_to_node_raw_with_limits(
            input,
            target_node,
            crown_bounds,
            ibp_bounds,
            engine,
            deadline,
            limits,
        )?;
        let flat_input = input.flatten();
        let input_lower = flat_input.lower().as_slice().ok_or_else(|| {
            NyError::InternalError(
                "target input-linear CROWN flattened lower input is not contiguous".to_string(),
            )
        })?;
        let input_upper = flat_input.upper().as_slice().ok_or_else(|| {
            NyError::InternalError(
                "target input-linear CROWN flattened upper input is not contiguous".to_string(),
            )
        })?;
        linear.fold_coeff_err_into_bias(input_lower, input_upper);
        if linear.has_coeff_err() {
            return Err(NyError::InternalError(format!(
                "target input-linear CROWN retained coefficient error after exact-box fold \
                 for target '{target_node}'"
            )));
        }
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return Err(NyError::DeadlineExceeded(format!(
                "target input-linear CROWN deadline exceeded while folding target \
                 '{target_node}'"
            )));
        }
        Ok(linear)
    }

    /// Private raw capture. The coefficient-error channel is intentionally
    /// inaccessible outside this module's implementation/tests; the public API
    /// above discharges it over the exact supplied input box.
    #[allow(clippy::too_many_arguments)]
    fn capture_crown_input_linear_to_node_raw_with_limits(
        &self,
        input: &BoundedTensor,
        target_node: &str,
        crown_bounds: &std::collections::HashMap<String, BoundedTensor>,
        ibp_bounds: &std::collections::HashMap<String, BoundedTensor>,
        engine: Option<&dyn ny_core::GemmEngine>,
        deadline: Option<std::time::Instant>,
        limits: TargetInputLinearLimits,
    ) -> Result<LinearBounds> {
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return Err(NyError::DeadlineExceeded(format!(
                "target input-linear CROWN deadline expired before target '{target_node}'"
            )));
        }
        if target_node != NETWORK_INPUT && !self.nodes.contains_key(target_node) {
            return Err(NyError::InvalidSpec(format!(
                "target input-linear CROWN target '{target_node}' is not a graph node"
            )));
        }

        let input_dim = input.len();
        let target_dim = if target_node == NETWORK_INPUT {
            input_dim
        } else {
            ibp_bounds
                .get(target_node)
                .ok_or_else(|| {
                    NyError::InvalidSpec(format!("Target node {target_node} not in IBP bounds"))
                })?
                .len()
        };
        let fixed_bytes =
            target_input_linear_fixed_bytes(target_dim, input_dim).unwrap_or(usize::MAX);
        if fixed_bytes > limits.aggregate_bytes {
            return Err(NyError::CpuMemoryExceeded {
                required_bytes: fixed_bytes,
                budget_bytes: limits.aggregate_bytes,
                site: "graph-alpha target input-linear aggregate workspace",
            });
        }
        if target_node == NETWORK_INPUT {
            return Ok(LinearBounds::identity(input_dim));
        }

        let target_bounds = ibp_bounds.get(target_node).ok_or_else(|| {
            NyError::InvalidSpec(format!("Target node {target_node} not in IBP bounds"))
        })?;
        let relevant_nodes = self.ancestors(target_node)?;
        if relevant_nodes.is_empty() {
            return Err(NyError::InvalidSpec(format!(
                "target input-linear CROWN graph node '{target_node}' has no backward ancestry"
            )));
        }
        let mut relation_cols_sum = target_dim.saturating_add(input_dim);
        let mut relation_count = 2usize;
        for name in &relevant_nodes {
            let width = crown_bounds
                .get(name)
                .or_else(|| ibp_bounds.get(name))
                .ok_or_else(|| {
                    NyError::InvalidSpec(format!(
                        "target input-linear CROWN relation node '{name}' has no bound box"
                    ))
                })?
                .len();
            relation_cols_sum = relation_cols_sum.saturating_add(width);
            relation_count = relation_count.saturating_add(1);
        }
        let chunk_row_bytes =
            target_input_linear_chunk_row_bytes(target_dim, relation_cols_sum, relation_count)
                .unwrap_or(usize::MAX);
        let has_conv_transpose = relevant_nodes.iter().any(|name| {
            self.nodes.get(name).is_some_and(|node| {
                matches!(
                    node.layer,
                    Layer::ConvTranspose1d(_) | Layer::ConvTranspose2d(_)
                )
            })
        });
        let chunk_rows = target_input_linear_chunk_rows(
            target_dim,
            fixed_bytes,
            chunk_row_bytes,
            deadline.is_some(),
            has_conv_transpose,
            limits.aggregate_bytes,
        )?;

        let _gpu_deadline_scope =
            crate::sound_gpu_gate::GpuCrownDeadlineScope::set(engine, deadline);
        let allow_patches = target_allows_patches_start(
            self,
            target_node,
            None,
            &relevant_nodes,
            target_bounds,
            false,
        );
        let target_shape = target_bounds.shape();
        let spatial = if allow_patches && target_shape.len() == 3 {
            Some((target_shape[0], target_shape[1], target_shape[2]))
        } else {
            None
        };
        let gpu_suffix_plan =
            GpuSuffixPlan::build(&relevant_nodes, self, input, crown_bounds, ibp_bounds, None);

        // Allocate the complete, preflighted return payload once. Chunks are
        // copied directly into their deterministic row ranges, so we never hold
        // a second full certificate during concatenation.
        let mut lower_a = Array2::<f32>::zeros((target_dim, input_dim));
        let mut upper_a = Array2::<f32>::zeros((target_dim, input_dim));
        let mut lower_b = Array1::<f32>::zeros(target_dim);
        let mut upper_b = Array1::<f32>::zeros(target_dim);
        let mut lower_err = Array2::<f32>::zeros((target_dim, input_dim));
        let mut upper_err = Array2::<f32>::zeros((target_dim, input_dim));
        let mut any_err = false;

        let mut r0 = 0usize;
        while r0 < target_dim {
            if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
                return Err(NyError::DeadlineExceeded(format!(
                    "target input-linear CROWN deadline exceeded before rows {r0}.. for \
                     target '{target_node}'"
                )));
            }
            let r1 = (r0 + chunk_rows).min(target_dim);
            let rows = r1 - r0;
            let chunk_bytes = chunk_row_bytes
                .checked_mul(rows)
                .and_then(|bytes| fixed_bytes.checked_add(bytes))
                .unwrap_or(usize::MAX);
            if chunk_bytes > limits.aggregate_bytes {
                return Err(NyError::CpuMemoryExceeded {
                    required_bytes: chunk_bytes,
                    budget_bytes: limits.aggregate_bytes,
                    site: "graph-alpha target input-linear aggregate workspace",
                });
            }
            let seed = build_chunk_seed(target_dim, r0, r1, spatial)?;
            let chunk_contract =
                GraphTargetShapeContract::from_bounds(target_node, &flat_bounds_view(rows)?);
            let produced = self.run_target_backward_pass_linear(
                input,
                target_node,
                crown_bounds,
                ibp_bounds,
                None,
                engine,
                "CROWN target input-linear",
                deadline,
                false,
                relevant_nodes.as_slice(),
                &chunk_contract,
                rows,
                input_dim,
                allow_patches,
                &gpu_suffix_plan,
                seed,
                None,
            )?;
            if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
                return Err(NyError::DeadlineExceeded(format!(
                    "target input-linear CROWN deadline exceeded after rows {r0}..{r1} \
                     for target '{target_node}'"
                )));
            }

            let chunk = match produced {
                Some(linear) => linear,
                None => {
                    // Mirror the concrete API's no-input-contribution fallback:
                    // the certified target box is a zero-coefficient affine
                    // enclosure over the original input.
                    let lo = Array1::from_iter(
                        target_bounds.lower().iter().copied().skip(r0).take(rows),
                    );
                    let up = Array1::from_iter(
                        target_bounds.upper().iter().copied().skip(r0).take(rows),
                    );
                    LinearBounds::new(
                        Array2::zeros((rows, input_dim)),
                        lo,
                        Array2::zeros((rows, input_dim)),
                        up,
                    )?
                }
            };
            if chunk.num_outputs() != rows || chunk.num_inputs() != input_dim {
                return Err(NyError::InvalidSpec(format!(
                    "target input-linear CROWN rows {r0}..{r1} produced shape {}x{}, \
                     expected {rows}x{input_dim}",
                    chunk.num_outputs(),
                    chunk.num_inputs()
                )));
            }
            lower_a
                .slice_mut(ndarray::s![r0..r1, ..])
                .assign(chunk.lower_a());
            upper_a
                .slice_mut(ndarray::s![r0..r1, ..])
                .assign(chunk.upper_a());
            lower_b
                .slice_mut(ndarray::s![r0..r1])
                .assign(chunk.lower_b());
            upper_b
                .slice_mut(ndarray::s![r0..r1])
                .assign(chunk.upper_b());
            if let Some(err) = chunk.lower_a_err() {
                if err.iter().any(|v| !v.is_finite() || *v < 0.0) {
                    return Err(NyError::NumericalInstability(format!(
                        "target input-linear CROWN lower coefficient error is invalid in \
                         rows {r0}..{r1}"
                    )));
                }
                any_err = true;
                lower_err.slice_mut(ndarray::s![r0..r1, ..]).assign(err);
            }
            if let Some(err) = chunk.upper_a_err() {
                if err.iter().any(|v| !v.is_finite() || *v < 0.0) {
                    return Err(NyError::NumericalInstability(format!(
                        "target input-linear CROWN upper coefficient error is invalid in \
                         rows {r0}..{r1}"
                    )));
                }
                any_err = true;
                upper_err.slice_mut(ndarray::s![r0..r1, ..]).assign(err);
            }
            r0 = r1;
        }

        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return Err(NyError::DeadlineExceeded(format!(
                "target input-linear CROWN deadline exceeded while assembling target \
                 '{target_node}'"
            )));
        }
        // Strict construction fails closed without allocating a second
        // conservative A/b payload while the four assembled A/error matrices
        // are still resident under the aggregate cap.
        let mut assembled = LinearBounds::new(lower_a, lower_b, upper_a, upper_b)?;
        if any_err {
            assembled.set_coeff_err(lower_err, upper_err);
        }
        Ok(assembled)
    }

    /// Run one backward CROWN pass for a single seed (`initial_bounds`) over
    /// `relevant_nodes`, returning the concretized + shape-restored bounds for
    /// the seed's rows, or `None` when no input contribution accumulated.
    ///
    /// This is the verbatim per-target backward loop. Both the single-pass and
    /// the objective-chunking (#patches-obj-chunk) paths drive it; chunking only
    /// changes the seed (a C-row slice of the objective) and the restore shape
    /// (a flat `[chunk_rows]` contract), never the per-step relaxation math.
    #[allow(clippy::too_many_arguments)]
    fn run_target_backward_pass(
        &self,
        input: &BoundedTensor,
        target_node: &str,
        crown_bounds: &std::collections::HashMap<String, BoundedTensor>,
        ibp_bounds: &std::collections::HashMap<String, BoundedTensor>,
        alpha_state: Option<&GraphAlphaState>,
        engine: Option<&dyn ny_core::GemmEngine>,
        label: &str,
        per_node_deadline: Option<std::time::Instant>,
        collector_patches_override: bool,
        relevant_nodes: &[String],
        // Restore contract: full target shape for single-pass, flat `[chunk_rows]`
        // for objective-chunking. Used by both the GPU-suffix and final concretize.
        target_contract: &GraphTargetShapeContract,
        // Number of objective rows in THIS seed (full target_dim, or chunk size).
        target_dim: usize,
        input_dim: usize,
        allow_patches: bool,
        gpu_suffix_plan: &GpuSuffixPlan,
        initial_bounds: CrownBounds,
        cut_ctx: Option<&CrownCutContext>,
    ) -> Result<Option<BoundedTensor>> {
        match self.run_target_backward_pass_core(
            input,
            target_node,
            crown_bounds,
            ibp_bounds,
            alpha_state,
            engine,
            label,
            per_node_deadline,
            collector_patches_override,
            relevant_nodes,
            target_contract,
            target_dim,
            input_dim,
            allow_patches,
            gpu_suffix_plan,
            initial_bounds,
            cut_ctx,
            false,
        )? {
            TargetBackwardPassResult::NoInputContribution => Ok(None),
            TargetBackwardPassResult::Concrete(bounds) => Ok(Some(bounds)),
            TargetBackwardPassResult::InputLinear(_) => Err(NyError::InternalError(
                "concrete target backward unexpectedly returned an input-linear certificate"
                    .to_string(),
            )),
        }
    }

    /// Linear-capture sibling of [`run_target_backward_pass`]. It deliberately
    /// skips bounds-only GPU suffixes and returns the final relation at
    /// `NETWORK_INPUT` with certified coefficient errors still attached.
    #[allow(clippy::too_many_arguments)]
    fn run_target_backward_pass_linear(
        &self,
        input: &BoundedTensor,
        target_node: &str,
        crown_bounds: &std::collections::HashMap<String, BoundedTensor>,
        ibp_bounds: &std::collections::HashMap<String, BoundedTensor>,
        alpha_state: Option<&GraphAlphaState>,
        engine: Option<&dyn ny_core::GemmEngine>,
        label: &str,
        per_node_deadline: Option<std::time::Instant>,
        collector_patches_override: bool,
        relevant_nodes: &[String],
        target_contract: &GraphTargetShapeContract,
        target_dim: usize,
        input_dim: usize,
        allow_patches: bool,
        gpu_suffix_plan: &GpuSuffixPlan,
        initial_bounds: CrownBounds,
        cut_ctx: Option<&CrownCutContext>,
    ) -> Result<Option<LinearBounds>> {
        match self.run_target_backward_pass_core(
            input,
            target_node,
            crown_bounds,
            ibp_bounds,
            alpha_state,
            engine,
            label,
            per_node_deadline,
            collector_patches_override,
            relevant_nodes,
            target_contract,
            target_dim,
            input_dim,
            allow_patches,
            gpu_suffix_plan,
            initial_bounds,
            cut_ctx,
            true,
        )? {
            TargetBackwardPassResult::NoInputContribution => Ok(None),
            TargetBackwardPassResult::InputLinear(linear) => Ok(Some(linear)),
            TargetBackwardPassResult::Concrete(_) => Err(NyError::InternalError(
                "input-linear target backward unexpectedly returned concrete bounds".to_string(),
            )),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_target_backward_pass_core(
        &self,
        input: &BoundedTensor,
        target_node: &str,
        crown_bounds: &std::collections::HashMap<String, BoundedTensor>,
        ibp_bounds: &std::collections::HashMap<String, BoundedTensor>,
        alpha_state: Option<&GraphAlphaState>,
        engine: Option<&dyn ny_core::GemmEngine>,
        label: &str,
        per_node_deadline: Option<std::time::Instant>,
        collector_patches_override: bool,
        relevant_nodes: &[String],
        target_contract: &GraphTargetShapeContract,
        target_dim: usize,
        input_dim: usize,
        allow_patches: bool,
        gpu_suffix_plan: &GpuSuffixPlan,
        initial_bounds: CrownBounds,
        cut_ctx: Option<&CrownCutContext>,
        capture_input_linear: bool,
    ) -> Result<TargetBackwardPassResult> {
        // Resnet-aware GPU-resident suffix (#vnncomp-resnet): if the whole ancestor
        // suffix decomposes into clean chains + identity/projection residual blocks,
        // run it on the proven sound GPU-resident resnet backward in one shot —
        // avoiding the CPU dense fork that materializes `[num_objectives × conv_dim]`
        // host matrices and OOMs/times out on cifar100/tinyimagenet ResNets. Every
        // bail (non-decomposable suffix, no sound GPU engine, GPU error, NaN) falls
        // through to the proven-sound CPU dense backward below, so the 0-wrong moat
        // holds. Only attempted from a fresh frontier (the loop has not run yet).
        //
        // The cheap `target_dim` (objective count) gate runs FIRST: a full dense GPU
        // resnet backward is only worthwhile for the verdict-shaped backward (few
        // objectives). Crucially this also avoids densifying a wide intermediate
        // node's identity seed (a `[dim × dim]` blow-up) — the gate must precede the
        // `into_dense()` below.
        if !capture_input_linear
            && super::super::resnet_decompose::resnet_gpu_enabled()
            && target_dim <= super::super::resnet_decompose::resnet_gpu_max_objectives()
        {
            if let Ok(seed_lb) = initial_bounds.clone().into_dense() {
                if let Some(bounds) = super::super::resnet_decompose::try_resnet_gpu_suffix(
                    self,
                    input,
                    target_node,
                    crown_bounds,
                    ibp_bounds,
                    alpha_state,
                    engine,
                    per_node_deadline,
                    &seed_lb,
                )? {
                    return Ok(TargetBackwardPassResult::Concrete(
                        target_contract.restore_concrete(
                            bounds,
                            "Graph alpha-CROWN GPU resnet suffix restore",
                        )?,
                    ));
                }
            }
        }

        let mut node_crown_bounds = CrownMergeAccumulator::new();
        node_crown_bounds.insert(target_node.to_string(), initial_bounds);
        let mut input_accumulated = false;
        for node_name in relevant_nodes.iter().rev() {
            if let Some(deadline) = per_node_deadline {
                if std::time::Instant::now() >= deadline {
                    return Err(NyError::DeadlineExceeded(format!(
                        "{}: per-node deadline exceeded at backward step '{}' for target '{}'",
                        label, node_name, target_node
                    )));
                }
            }

            let node = match self.nodes.get(node_name) {
                Some(n) => n,
                None => continue,
            };
            let mut node_cb = match node_crown_bounds.take(node_name)? {
                Some(cb) => cb,
                None => continue,
            };
            if tracing::enabled!(tracing::Level::TRACE) {
                if let CrownBounds::Dense(ref lb) = node_cb {
                    let bad = lb
                        .lower_a()
                        .iter()
                        .chain(lb.upper_a().iter())
                        .filter(|v| !v.is_finite())
                        .count();
                    let bad_b = lb
                        .lower_b()
                        .iter()
                        .chain(lb.upper_b().iter())
                        .filter(|v| !v.is_finite())
                        .count();
                    tracing::trace!(
                        "target-backward step '{}' ({}) seed dense a_bad={} b_bad={} rows={} cols={}",
                        node_name,
                        node.layer.layer_type(),
                        bad,
                        bad_b,
                        lb.num_outputs(),
                        lb.num_inputs()
                    );
                } else {
                    tracing::trace!(
                        "target-backward step '{}' ({}) seed patches",
                        node_name,
                        node.layer.layer_type()
                    );
                }
            }

            // #crown-cut-segment (NY_CROWN_CUT_SEGMENT): backward-to-nearest-
            // bounded-cut. When the walk reaches a designated cut node — an
            // EARLIER node whose bounds this sweep already finalized — the
            // accumulated linear relation is concretized against that node's
            // bound-box with the SAME directed-rounding concretization the
            // input-box path uses (`concretize_sound`, which also folds any
            // carried certified coefficient error over the box), and this path
            // of the walk stops instead of expanding the node's prefix.
            //
            // SOUNDNESS: the swept box is a valid enclosure of the cut node's
            // reachable set, so the concretized interval encloses the
            // relation's value for every reachable input — this is exactly the
            // input-box concretization applied at an intermediate cut (the
            // input box is the trivial cut). The result is only ever LOOSER
            // than full-prefix expansion (the box drops inter-node
            // correlations). Per-node relaxations of expanded nodes are
            // untouched — only the DEPTH of the walk changes.
            //
            // FAIL-OPEN: a missing, non-finite, or shape-mismatched box means
            // the node is expanded exactly as with the gate off (slower exact
            // behavior, never a wrong bound).
            if let Some(ctx) = cut_ctx {
                if node_name != target_node && ctx.is_cut(node_name) {
                    let cut_box = crown_bounds.get(node_name).filter(|b| box_is_finite(b));
                    // A Patches-carried relation must densify to concretize
                    // here; only cut when its dense pair fits the CPU dense
                    // budget (a Dense relation already paid that memory, so it
                    // always cuts). Over-budget patches relations keep the
                    // gate-off patches walk (fail-open: exact, no new
                    // allocation cliff).
                    let densify_fits = matches!(node_cb, CrownBounds::Dense(_))
                        || cut_box.is_some_and(|b| {
                            crate::network::crown_memory::dense_pair_bytes(target_dim, b.len())
                                .is_some_and(|bytes| bytes <= cpu_crown_dense_budget_bytes())
                        });
                    if let (Some(cut_box), true) = (cut_box, densify_fits) {
                        let node_lb = node_cb.into_dense()?;
                        if node_lb.num_inputs() == cut_box.len() {
                            let concrete = node_lb.concretize_sound(cut_box);
                            let (lower, upper) = concrete
                                .flatten_to_ix1("graph-alpha crown cut-segment concretize")?;
                            Self::accumulate_bias_to_network_input_crown(
                                &lower,
                                &upper,
                                &mut node_crown_bounds,
                                target_dim,
                                input_dim,
                                &mut input_accumulated,
                            );
                            ctx.record_cut();
                            continue;
                        }
                        // Defensive shape drift: keep the (already densified)
                        // relation and expand through the node as with the
                        // gate off.
                        node_cb = CrownBounds::Dense(node_lb);
                    }
                }
            }

            // Resolve first input; multi-input nodes skip to dedicated branches (#4112).
            let (first_input_name, pre_activation) = if node.inputs.len() == 1 {
                let name = node.require_unary_input().map_err(|_| {
                    NyError::InvalidSpec(format!(
                        "{label} failed at '{node_name}' ({}): node has no inputs",
                        node.layer.layer_type()
                    ))
                })?;
                (
                    name,
                    resolve_preactivation(input, name, crown_bounds, ibp_bounds)?,
                )
            } else {
                let fallback = node
                    .inputs
                    .first()
                    .map(String::as_str)
                    .unwrap_or(NETWORK_INPUT);
                (fallback, input)
            };

            if allow_patches
                && node.inputs.len() == 1
                && try_patches_target_step_core(
                    self,
                    label,
                    node_name,
                    node,
                    &mut node_cb,
                    first_input_name,
                    pre_activation,
                    ibp_bounds,
                    &mut node_crown_bounds,
                    target_dim,
                    input_dim,
                    &mut input_accumulated,
                    engine,
                    per_node_deadline,
                    collector_patches_override,
                    alpha_state,
                )?
            {
                continue;
            }

            // #patches-row-range: a Patches relation the patches-capable step
            // could not consume must densify here. On VGG-scale conv targets
            // that dense pair reaches TB scale (3.2M x 150K rows) and the
            // allocation aborts the process. Refuse over-budget densification
            // with the structured CpuMemoryExceeded that every caller maps to
            // a sound fallback (IBP / another strategy); under-budget
            // relations keep the existing path byte-for-byte.
            if let CrownBounds::Patches(ref pb) = node_cb {
                if let Some(err) = patches_densify_over_budget(pb, cpu_crown_dense_budget_bytes()) {
                    let (rows, cols) = pb.dense_pair_shape().unwrap_or((0, 0));
                    debug!(
                        "{}: backward step '{}' for target '{}' needs a {}x{} dense pair \
                         over the CPU dense budget; degrading ({})",
                        label, node_name, target_node, rows, cols, err
                    );
                    return Err(err);
                }
            }
            let node_lb = node_cb.into_dense()?;
            if !capture_input_linear {
                if let Some(bounds) = try_finish_target_gpu_suffix_with_pending_input(
                    input,
                    node_name,
                    &node_lb,
                    gpu_suffix_plan,
                    engine,
                    target_contract,
                    &mut node_crown_bounds,
                )? {
                    return Ok(TargetBackwardPassResult::Concrete(bounds));
                }
            }
            if let Layer::ReLU(r) = &node.layer {
                let mut new_lb = if let Some(alpha) = alpha_state.and_then(|s| s.alpha(node_name)) {
                    let alpha_upper = alpha_state.and_then(|s| s.alpha_upper(node_name));
                    let alpha_expanded = alpha_state
                        .map(|s| s.expand_alpha(node_name, alpha))
                        .unwrap_or_else(|| alpha.clone());
                    let alpha_upper_expanded = alpha_state
                        .and_then(|s| alpha_upper.map(|upper| s.expand_alpha(node_name, upper)));
                    let (bounds, _grad, _grad_upper) = r.propagate_linear_with_alpha(
                        &node_lb,
                        pre_activation,
                        &alpha_expanded,
                        alpha_upper_expanded.as_ref(),
                    )?;
                    bounds
                } else {
                    r.propagate_linear_with_bounds(&node_lb, pre_activation)
                        .map_err(|e| {
                            NyError::InvalidSpec(format!(
                                "{} failed at '{}' (ReLU): {}",
                                label, node_name, e
                            ))
                        })?
                };
                // Eager per-row discharge of the carried coefficient error over
                // the (CROWN-tightened) pre-activation cut — the tightest box the
                // error will ever see; carrying it further pays IBP-scale
                // magnitudes (#cgan-conv-err-compose, see
                // LinearBounds::fold_coeff_err_over_box_eager).
                new_lb.fold_coeff_err_over_box_eager(pre_activation);
                self.accumulate_dense_bounds_to_input(
                    first_input_name,
                    new_lb,
                    &mut node_crown_bounds,
                    target_dim,
                    input_dim,
                    &mut input_accumulated,
                )?;
            } else if let Layer::Sigmoid(sigmoid) = &node.layer {
                let mut new_lb = match alpha_state
                    .and_then(|s| s.monotone_s_shaped_alpha(node_name))
                {
                    Some(alpha) => {
                        match sigmoid.propagate_linear_with_alpha(&node_lb, pre_activation, alpha) {
                            Ok(bounds) => bounds,
                            // #4118: a monotone alpha bundle whose width disagrees with the
                            // pre-activation (e.g. a stale warm-started bundle reused across a
                            // reshape) surfaces as ShapeMismatch. Fixed-slope CROWN is always a
                            // sound relaxation, so retry this node locally instead of failing the
                            // whole backward pass and triggering a graph-wide fallback.
                            Err(NyError::ShapeMismatch { expected, got }) => {
                                warn!(
                                    "{} monotone alpha shape mismatch at '{}' (Sigmoid): expected {:?}, got {:?}; retrying fixed-slope locally",
                                    label, node_name, expected, got
                                );
                                sigmoid
                                    .propagate_linear_with_bounds(&node_lb, pre_activation)
                                    .map_err(|e| {
                                        NyError::InvalidSpec(format!(
                                            "{} failed at '{}' (Sigmoid fixed-slope retry): {}",
                                            label, node_name, e
                                        ))
                                    })?
                            }
                            Err(e) => return Err(e),
                        }
                    }
                    None => sigmoid
                        .propagate_linear_with_bounds(&node_lb, pre_activation)
                        .map_err(|e| {
                            NyError::InvalidSpec(format!(
                                "{} failed at '{}' (Sigmoid): {}",
                                label, node_name, e
                            ))
                        })?,
                };
                new_lb.fold_coeff_err_over_box_eager(pre_activation); // #cgan-conv-err-compose
                self.accumulate_dense_bounds_to_input(
                    first_input_name,
                    new_lb,
                    &mut node_crown_bounds,
                    target_dim,
                    input_dim,
                    &mut input_accumulated,
                )?;
            } else if let Layer::Tanh(tanh) = &node.layer {
                let mut new_lb = match alpha_state
                    .and_then(|s| s.monotone_s_shaped_alpha(node_name))
                {
                    Some(alpha) => {
                        match tanh.propagate_linear_with_alpha(&node_lb, pre_activation, alpha) {
                            Ok(bounds) => bounds,
                            // #4118: see the Sigmoid branch — a mismatched monotone alpha bundle
                            // falls back to the sound fixed-slope relaxation for this node only.
                            Err(NyError::ShapeMismatch { expected, got }) => {
                                warn!(
                                    "{} monotone alpha shape mismatch at '{}' (Tanh): expected {:?}, got {:?}; retrying fixed-slope locally",
                                    label, node_name, expected, got
                                );
                                tanh.propagate_linear_with_bounds(&node_lb, pre_activation)
                                    .map_err(|e| {
                                        NyError::InvalidSpec(format!(
                                            "{} failed at '{}' (Tanh fixed-slope retry): {}",
                                            label, node_name, e
                                        ))
                                    })?
                            }
                            Err(e) => return Err(e),
                        }
                    }
                    None => tanh
                        .propagate_linear_with_bounds(&node_lb, pre_activation)
                        .map_err(|e| {
                            NyError::InvalidSpec(format!(
                                "{} failed at '{}' (Tanh): {}",
                                label, node_name, e
                            ))
                        })?,
                };
                new_lb.fold_coeff_err_over_box_eager(pre_activation); // #cgan-conv-err-compose
                self.accumulate_dense_bounds_to_input(
                    first_input_name,
                    new_lb,
                    &mut node_crown_bounds,
                    target_dim,
                    input_dim,
                    &mut input_accumulated,
                )?;
            } else if let Layer::Sqrt(sqrt) = &node.layer {
                let mut new_lb = sqrt_support::backward_sqrt_node(
                    sqrt,
                    alpha_state,
                    node_name,
                    label,
                    &node_lb,
                    pre_activation,
                )?;
                new_lb.fold_coeff_err_over_box_eager(pre_activation); // #cgan-conv-err-compose
                self.accumulate_dense_bounds_to_input(
                    first_input_name,
                    new_lb,
                    &mut node_crown_bounds,
                    target_dim,
                    input_dim,
                    &mut input_accumulated,
                )?;
            } else if let Layer::Reciprocal(reciprocal) = &node.layer {
                let mut new_lb = reciprocal_support::backward_reciprocal_node(
                    reciprocal,
                    alpha_state,
                    node_name,
                    label,
                    &node_lb,
                    pre_activation,
                )?;
                new_lb.fold_coeff_err_over_box_eager(pre_activation); // #cgan-conv-err-compose
                self.accumulate_dense_bounds_to_input(
                    first_input_name,
                    new_lb,
                    &mut node_crown_bounds,
                    target_dim,
                    input_dim,
                    &mut input_accumulated,
                )?;
            } else if let Layer::MulBinary(mul) = &node.layer {
                let (input_a_name, input_b_name) = node.require_binary_inputs()?;
                let input_a_bounds = if input_a_name == NETWORK_INPUT {
                    input
                } else {
                    crown_bounds
                        .get(input_a_name)
                        .or_else(|| ibp_bounds.get(input_a_name))
                        .ok_or_else(|| {
                            NyError::InvalidSpec(format!(
                                "MulBinary input A '{}' not found",
                                input_a_name
                            ))
                        })?
                };
                let input_b_bounds = if input_b_name == NETWORK_INPUT {
                    input
                } else {
                    crown_bounds
                        .get(input_b_name)
                        .or_else(|| ibp_bounds.get(input_b_name))
                        .ok_or_else(|| {
                            NyError::InvalidSpec(format!(
                                "MulBinary input B '{}' not found",
                                input_b_name
                            ))
                        })?
                };

                match mul.propagate_linear_binary(
                    &node_lb,
                    input_a_bounds,
                    input_b_bounds,
                    MulBinaryRelaxationMode::default(),
                ) {
                    Ok((mut lb_a, mut lb_b)) => {
                        debug!("{}: MulBinary '{}' CROWN succeeded", label, node_name);
                        let bias_lower = lb_a.lower_b() + lb_b.lower_b();
                        let bias_upper = lb_a.upper_b() + lb_b.upper_b();
                        lb_a.lower_b_mut().fill(0.0);
                        lb_a.upper_b_mut().fill(0.0);
                        lb_b.lower_b_mut().fill(0.0);
                        lb_b.upper_b_mut().fill(0.0);
                        Self::verify_split_path_bias_zero(
                            &lb_a,
                            &format!("{} MulBinary lhs split path", label),
                        )?;
                        Self::verify_split_path_bias_zero(
                            &lb_b,
                            &format!("{} MulBinary rhs split path", label),
                        )?;
                        Self::accumulate_bias_to_network_input_crown(
                            &bias_lower,
                            &bias_upper,
                            &mut node_crown_bounds,
                            target_dim,
                            input_dim,
                            &mut input_accumulated,
                        );
                        self.accumulate_dense_bounds_to_input(
                            input_a_name,
                            lb_a,
                            &mut node_crown_bounds,
                            target_dim,
                            input_dim,
                            &mut input_accumulated,
                        )?;
                        self.accumulate_dense_bounds_to_input(
                            input_b_name,
                            lb_b,
                            &mut node_crown_bounds,
                            target_dim,
                            input_dim,
                            &mut input_accumulated,
                        )?;
                    }
                    Err(
                        e @ NyError::UnsupportedOp(_)
                        | e @ NyError::UnsupportedConfiguration(_)
                        | e @ NyError::NumericalInstability(_),
                    ) => {
                        debug!(
                            "{}: MulBinary '{}' failed ({}), returning UnsupportedOp for IBP fallback",
                            label, node_name, e
                        );
                        return Err(NyError::UnsupportedOp(format!(
                            "{}: MulBinary '{}' CROWN failed: {}",
                            label, node_name, e
                        )));
                    }
                    Err(e) => return Err(e),
                }
            } else if matches!(&node.layer, Layer::Div(_)) {
                let (input_a_name, input_b_name) = node.require_binary_inputs()?;
                let input_a_bounds = if input_a_name == NETWORK_INPUT {
                    input
                } else {
                    crown_bounds
                        .get(input_a_name)
                        .or_else(|| ibp_bounds.get(input_a_name))
                        .ok_or_else(|| {
                            NyError::InvalidSpec(format!(
                                "Div numerator '{}' not found",
                                input_a_name
                            ))
                        })?
                };
                let input_b_bounds = if input_b_name == NETWORK_INPUT {
                    input
                } else {
                    crown_bounds
                        .get(input_b_name)
                        .or_else(|| ibp_bounds.get(input_b_name))
                        .ok_or_else(|| {
                            NyError::InvalidSpec(format!(
                                "Div denominator '{}' not found",
                                input_b_name
                            ))
                        })?
                };
                let node_output_bounds = crown_bounds
                    .get(node_name)
                    .or_else(|| ibp_bounds.get(node_name))
                    .ok_or_else(|| {
                        NyError::InvalidSpec(format!(
                            "Div output '{}' not found in bound maps",
                            node_name
                        ))
                    })?;
                match div::backward_div_to_numerator(
                    node_name,
                    &node_lb,
                    input_a_bounds,
                    input_b_bounds,
                    node_output_bounds,
                )? {
                    div::DivBackwardResult::PropagateNumerator(bounds) => {
                        self.accumulate_dense_bounds_to_input(
                            input_a_name,
                            *bounds,
                            &mut node_crown_bounds,
                            target_dim,
                            input_dim,
                            &mut input_accumulated,
                        )?;
                    }
                    div::DivBackwardResult::ConcretizeCurrentNode { lower, upper } => {
                        Self::accumulate_bias_to_network_input_crown(
                            &lower,
                            &upper,
                            &mut node_crown_bounds,
                            target_dim,
                            input_dim,
                            &mut input_accumulated,
                        );
                    }
                }
            } else if let Layer::Where(where_layer) = &node.layer {
                // === Embedded-constant Where (single `cond` input; both branches
                // constants). Output is a constant vector w.r.t. the network input;
                // fold it into the bias and route zero to `cond`. Exact per-element
                // select when `cond` is constant, sound IBP union otherwise.
                // require_ternary_inputs would error: the node has only 1 input.
                if where_layer.has_embedded_constants() {
                    let cond_input = node.require_unary_input()?;
                    let cond_bounds = if cond_input == NETWORK_INPUT {
                        input
                    } else {
                        crown_bounds
                            .get(cond_input)
                            .or_else(|| ibp_bounds.get(cond_input))
                            .ok_or_else(|| {
                                NyError::InvalidSpec(format!(
                                    "Where condition '{}' not found",
                                    cond_input
                                ))
                            })?
                    };
                    let select = where_layer.embedded_constant_select_output(cond_bounds)?;
                    let concrete = node_lb.concretize_checked(&select)?;
                    let (lower, upper) =
                        concrete.flatten_to_ix1("graph-alpha embedded-constant Where")?;
                    Self::accumulate_bias_to_network_input_crown(
                        &lower,
                        &upper,
                        &mut node_crown_bounds,
                        target_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                    continue;
                }

                let (cond_input, true_input, false_input) = node.require_ternary_inputs()?;
                let cond_bounds = if cond_input == NETWORK_INPUT {
                    input
                } else {
                    crown_bounds
                        .get(cond_input)
                        .or_else(|| ibp_bounds.get(cond_input))
                        .ok_or_else(|| {
                            NyError::InvalidSpec(format!(
                                "Where condition '{}' not found",
                                cond_input
                            ))
                        })?
                };
                let where_bounds = crown_bounds
                    .get(node_name)
                    .or_else(|| ibp_bounds.get(node_name))
                    .ok_or_else(|| {
                        NyError::InvalidSpec(format!(
                            "Where output '{}' not found in bound maps",
                            node_name
                        ))
                    })?;
                let cond_all_true = cond_bounds.lower().iter().all(|&v| v >= 0.5);
                let cond_all_false = cond_bounds.upper().iter().all(|&v| v <= 0.5);

                if cond_all_true {
                    self.accumulate_dense_bounds_to_input(
                        true_input,
                        node_lb,
                        &mut node_crown_bounds,
                        target_dim,
                        input_dim,
                        &mut input_accumulated,
                    )?;
                } else if cond_all_false {
                    self.accumulate_dense_bounds_to_input(
                        false_input,
                        node_lb,
                        &mut node_crown_bounds,
                        target_dim,
                        input_dim,
                        &mut input_accumulated,
                    )?;
                } else if let Some(mask) =
                    crate::network::core::graph::backward_helpers::where_constant_mask(cond_bounds)
                {
                    // === Exact per-element select for a bound-independent (constant)
                    // condition mask. When `cond` is fixed (lower == upper), Where is
                    // a fixed 0/1 select: out[i] = true_input[i] if mask[i] else
                    // false_input[i] — an EXACT linear transform. Route each output
                    // column to the correct branch by zeroing the other branch's
                    // columns. Mirrors the graph-CROWN path; tighter than the loose
                    // concretize fallback below. Falls back to concretize on a shape
                    // mismatch (defensive; mask length == node_lb columns by
                    // construction).
                    if mask.len() == node_lb.num_inputs() {
                        let true_lb =
                            crate::network::core::graph::backward_helpers::mask_linear_bounds_columns(
                                &node_lb, &mask, true,
                            );
                        let false_lb =
                            crate::network::core::graph::backward_helpers::mask_linear_bounds_columns(
                                &node_lb, &mask, false,
                            );
                        self.accumulate_dense_bounds_to_input(
                            true_input,
                            true_lb,
                            &mut node_crown_bounds,
                            target_dim,
                            input_dim,
                            &mut input_accumulated,
                        )?;
                        self.accumulate_dense_bounds_to_input(
                            false_input,
                            false_lb,
                            &mut node_crown_bounds,
                            target_dim,
                            input_dim,
                            &mut input_accumulated,
                        )?;
                    } else {
                        let concrete = node_lb.concretize_checked(where_bounds)?;
                        let (lower, upper) =
                            concrete.flatten_to_ix1("graph-alpha Where mixed fallback")?;
                        Self::accumulate_bias_to_network_input_crown(
                            &lower,
                            &upper,
                            &mut node_crown_bounds,
                            target_dim,
                            input_dim,
                            &mut input_accumulated,
                        );
                    }
                } else {
                    let concrete = node_lb.concretize_checked(where_bounds)?;
                    let (lower, upper) =
                        concrete.flatten_to_ix1("graph-alpha Where mixed fallback")?;
                    Self::accumulate_bias_to_network_input_crown(
                        &lower,
                        &upper,
                        &mut node_crown_bounds,
                        target_dim,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
            } else {
                let mut combined_bounds = std::collections::HashMap::new();
                for inp_name in &node.inputs {
                    if inp_name == NETWORK_INPUT {
                        continue;
                    }
                    if let Some(b) = crown_bounds.get(inp_name) {
                        combined_bounds.insert(inp_name.clone(), b.clone());
                    } else if let Some(b) = ibp_bounds.get(inp_name) {
                        combined_bounds.insert(inp_name.clone(), b.clone());
                    }
                }
                // #cgan-coeff-err-fold: also expose THIS node's own output box.
                // `dispatch_backward_layer` folds any incoming certified
                // coefficient error into the bias via
                // `ctx.node_bounds.get(ctx.node_name)` when the layer is not an
                // err-carrier (BatchNorm, pooling, ...). Without the node's own
                // bounds that fold degraded EVERY err-carrying row to
                // `[-inf, +inf]` (`discharge_coeff_err_to_conservative`), which
                // collapsed the whole per-target CROWN backward to IBP on
                // conv->BatchNorm stacks (cGAN generators: conv backward attaches
                // a fresh gamma_n*S err, the following BatchNorm found no output
                // box, and the target bound became vacuous before the IBP
                // intersection masked it as "Crown"). Sound: the fold is the
                // established precise discharge over the node's certified output
                // enclosure; providing the box only replaces the +/-inf degrade.
                if let Some(b) = crown_bounds
                    .get(node_name)
                    .or_else(|| ibp_bounds.get(node_name))
                {
                    combined_bounds.insert(node_name.clone(), b.clone());
                }

                let ctx = DispatchContext {
                    node_name,
                    layer: &node.layer,
                    inputs: &node.inputs,
                    pre_activation,
                    network_input: input,
                    node_bounds: (&combined_bounds).into(),
                    engine,
                    deadline: per_node_deadline,
                    bilinear_alphas: None,
                    mul_binary_relaxation: MulBinaryRelaxationMode::default(),
                    mul_binary_alphas: None,
                    norm_inv_rms_override: None,
                };
                let result = dispatch_backward_layer(&ctx, &node_lb)?;
                match apply_dense_backward_dispatch_result(
                    self,
                    node,
                    first_input_name,
                    &node_lb,
                    result,
                    &mut node_crown_bounds,
                    target_dim,
                    input_dim,
                    &mut input_accumulated,
                    "Alpha-CROWN/IBP",
                ) {
                    Ok(()) => {}
                    Err(NyError::UnsupportedOp(reason)) => {
                        return Err(NyError::UnsupportedOp(format!(
                            "{}: unsupported layer '{}' ({}): {}",
                            label,
                            node_name,
                            node.layer.layer_type(),
                            reason,
                        )));
                    }
                    Err(e) => return Err(e),
                }
            }
        }
        if input_accumulated {
            let final_lb = node_crown_bounds
                .take(NETWORK_INPUT)?
                .ok_or_else(|| NyError::InvalidSpec("No linear bounds at input".to_string()))?;
            if capture_input_linear {
                if per_node_deadline.is_some_and(|deadline| std::time::Instant::now() >= deadline) {
                    return Err(NyError::DeadlineExceeded(format!(
                        "{label}: per-node deadline exceeded before linear capture for \
                         target '{target_node}'"
                    )));
                }
                // The caller preflighted the complete dense A+bias+error
                // payload. `into_dense` preserves Patches coefficient-error
                // provenance; no concretization or error discharge happens
                // here.
                return Ok(TargetBackwardPassResult::InputLinear(
                    final_lb.into_dense()?,
                ));
            }
            // #patches-row-range: a Patches-carried final relation whose dense
            // pair exceeds the CPU dense budget must never densify in one shot
            // — VGG16 conv targets reach 3.2M x 150K rows (~1.9 TB per matrix)
            // and abort the process. Per-row independence makes the blockwise
            // materialize-and-concretize bit-identical to the single-shot
            // path with memory bounded by the block budget; the deadline is
            // re-checked between blocks so a slow target degrades to the same
            // sound DeadlineExceeded fallback as the walk itself. The
            // under-budget path (including its DEBUG err diagnostic) is
            // unchanged byte-for-byte.
            let bounds = match final_lb {
                CrownBounds::Patches(pb)
                    if pb
                        .dense_pair_bytes()
                        .is_ok_and(|bytes| bytes > cpu_crown_dense_budget_bytes()) =>
                {
                    // #patches-sparse-concretize: prefer the patches-native sparse
                    // concretize — it visits only each row's receptive-field taps
                    // (~27 for a VGG16 conv1 target) instead of all 150528 input
                    // columns, turning the ~4.8e11-element dense-chunked traversal
                    // (a timeout) into ~Σ receptive-field work while staying
                    // BIT-IDENTICAL to `to_dense()?.concretize_sound(input)`. On an
                    // unsupported layout it returns `UnsupportedOp`; only then do we
                    // fall back to the certified (sound, memory-bounded) dense-chunked
                    // path. `DeadlineExceeded` and malformed-layout errors propagate.
                    match pb.concretize_sound_sparse(input, per_node_deadline) {
                        Ok(bounds) => bounds,
                        Err(NyError::UnsupportedOp(_)) => {
                            let block_bytes = cpu_crown_dense_budget_bytes()
                                .min(PATCHES_CONCRETIZE_MAX_BLOCK_BYTES);
                            pb.concretize_sound_chunked(input, block_bytes, per_node_deadline)?
                        }
                        Err(e) => return Err(e),
                    }
                }
                final_lb => {
                    let final_dense = final_lb.into_dense()?;
                    if tracing::enabled!(tracing::Level::DEBUG) {
                        // Diagnostic (#cgan-conv-err-compose): report the certified-error
                        // share of the final concretized width for this target.
                        let flat = input.flatten();
                        let xl = flat.lower();
                        let xu = flat.upper();
                        let mut max_pen = 0.0f64;
                        for err in [final_dense.lower_a_err(), final_dense.upper_a_err()]
                            .into_iter()
                            .flatten()
                        {
                            for i in 0..err.nrows() {
                                let mut pen = 0.0f64;
                                for j in 0..err.ncols() {
                                    let mag = (xl[j].abs()).max(xu[j].abs()) as f64;
                                    pen += err[[i, j]] as f64 * mag;
                                }
                                max_pen = max_pen.max(pen);
                            }
                        }
                        debug!(
                            "target '{}': final concretize max per-row err penalty {:.3e}",
                            target_node, max_pen
                        );
                    }
                    final_dense.concretize_sound(input)
                }
            };
            Ok(TargetBackwardPassResult::Concrete(
                target_contract.restore_concrete(bounds, "Graph alpha-CROWN target restore")?,
            ))
        } else {
            // No input contribution accumulated; the caller substitutes the
            // pass-through target bounds (full target for single-pass, or the
            // chunk slice for objective-chunking).
            Ok(TargetBackwardPassResult::NoInputContribution)
        }
    }

    /// PRIMARY (#patches-obj-chunk): objective-chunking streaming driver.
    ///
    /// Streams the objective dimension (`target_dim` rows) in chunks of at most
    /// `chunk_size`. For each chunk `r0..r1`:
    ///   1. seed a `chunk_rows`-row slice of the objective identity (Dense rows
    ///      of `LinearBounds::identity`, or a sparse-identity Patches seed whose
    ///      `unstable_idx` enumerates exactly those flat output positions),
    ///   2. run the EXISTING backward loop (`run_target_backward_pass`) with a
    ///      per-chunk accumulator and a flat `[chunk_rows]` restore contract,
    ///   3. scatter the concretized `chunk_rows` values into the pre-sized
    ///      output, then drop the chunk coefficients before the next chunk.
    ///
    /// Bound-equivalent to the single pass by row-independence: the conv col2im
    /// scatter, the per-row CROWN error term, and the per-row concretize are all
    /// row-local, so concretizing rows `r0..r1` in isolation yields the same
    /// values they would take in a full-objective pass. The `GpuSuffixPlan` is
    /// objective-independent and is built once by the caller and reused here.
    #[allow(clippy::too_many_arguments)]
    fn propagate_crown_to_node_chunked(
        &self,
        input: &BoundedTensor,
        target_node: &str,
        crown_bounds: &std::collections::HashMap<String, BoundedTensor>,
        ibp_bounds: &std::collections::HashMap<String, BoundedTensor>,
        alpha_state: Option<&GraphAlphaState>,
        engine: Option<&dyn ny_core::GemmEngine>,
        label: &str,
        per_node_deadline: Option<std::time::Instant>,
        collector_patches_override: bool,
        relevant_nodes: &[String],
        target_bounds: &BoundedTensor,
        target_contract: &GraphTargetShapeContract,
        target_dim: usize,
        input_dim: usize,
        allow_patches: bool,
        gpu_suffix_plan: &GpuSuffixPlan,
        chunk_size: usize,
        cut_ctx: Option<&CrownCutContext>,
    ) -> Result<BoundedTensor> {
        // The Patches seed is only valid for a 3D spatial objective; otherwise
        // (and whenever `allow_patches` is false) seed Dense identity rows.
        let target_shape = target_bounds.shape();
        let patches_seed = allow_patches && target_shape.len() == 3;
        let spatial = if patches_seed {
            Some((target_shape[0], target_shape[1], target_shape[2]))
        } else {
            None
        };

        // Pre-sized flat output; scatter each chunk's concrete rows into place.
        let target_flat = target_bounds.flatten();
        let target_lower_flat = target_flat.lower();
        let target_upper_flat = target_flat.upper();
        let mut out_lower = vec![0.0_f32; target_dim];
        let mut out_upper = vec![0.0_f32; target_dim];

        // The chunk ranges `[r0, r1)` partition the objective rows.
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        let mut r0 = 0usize;
        while r0 < target_dim {
            let r1 = (r0 + chunk_size).min(target_dim);
            ranges.push((r0, r1));
            r0 = r1;
        }

        // Bound one chunk `[r0, r1)`. ROW-INDEPENDENT (see this fn's docstring): the
        // col2im scatter, the per-row CROWN error term, and the per-row concretize are
        // all row-local, so a chunk's rows take the same values in isolation as in the
        // full-objective pass. Returns `(r0, lower_rows, upper_rows)`.
        let bound_chunk = |&(r0, r1): &(usize, usize),
                           ctx: Option<&CrownCutContext>|
         -> Result<(usize, Vec<f32>, Vec<f32>)> {
            let chunk_rows = r1 - r0;
            let seed = build_chunk_seed(target_dim, r0, r1, spatial)?;
            // Flat restore contract for this chunk: 1D `[chunk_rows]`, so the
            // GPU-suffix / final concretize restore is an identity reshape and
            // the produced bounds line up 1:1 with the chunk's output rows.
            let chunk_contract =
                GraphTargetShapeContract::from_bounds(target_node, &flat_bounds_view(chunk_rows)?);

            let produced = self.run_target_backward_pass(
                input,
                target_node,
                crown_bounds,
                ibp_bounds,
                alpha_state,
                engine,
                label,
                per_node_deadline,
                collector_patches_override,
                relevant_nodes,
                &chunk_contract,
                chunk_rows,
                input_dim,
                allow_patches,
                gpu_suffix_plan,
                seed,
                ctx,
            )?;

            // A layer kernel may cross the deadline before it can return to
            // `run_target_backward_pass`'s per-node poll.  Check immediately
            // after every bounded objective chunk, including the final chunk,
            // so a late last chunk cannot be accepted as an on-time result.
            if per_node_deadline.is_some_and(|deadline| std::time::Instant::now() >= deadline) {
                return Err(NyError::DeadlineExceeded(format!(
                    "{label}: per-node deadline exceeded after objective chunk \
                     {r0}..{r1} for target '{target_node}'"
                )));
            }

            match produced {
                Some(bounds) => {
                    let lo = bounds.lower();
                    let up = bounds.upper();
                    let lo = lo.as_slice().ok_or_else(|| {
                        NyError::InvalidSpec(
                            "objective-chunk concrete lower not contiguous".to_string(),
                        )
                    })?;
                    let up = up.as_slice().ok_or_else(|| {
                        NyError::InvalidSpec(
                            "objective-chunk concrete upper not contiguous".to_string(),
                        )
                    })?;
                    Ok((r0, lo.to_vec(), up.to_vec()))
                }
                None => {
                    // No input contribution accumulated for this chunk: pass the
                    // target bounds slice through unchanged (mirrors the single-
                    // pass `target_bounds.clone()` terminal branch, restricted to
                    // these rows).
                    let lo: Vec<f32> = (r0..r1).map(|row| target_lower_flat[[row]]).collect();
                    let up: Vec<f32> = (r0..r1).map(|row| target_upper_flat[[row]]).collect();
                    Ok((r0, lo, up))
                }
            }
        };

        // PARALLEL chunk driver — opt-in via the IMB anchor scope
        // (`crate::imb::anchor_chunk_parallel`). The chunks are row-independent, so a
        // parallel evaluation is bound-equivalent AND order-independent (each writes a
        // DISJOINT `out_*[r0..r1]` range) ⇒ deterministic. Faer stays Seq-guarded
        // inside the rayon workers (`current_par`), so the parallelism is across
        // chunks, not nested. Every OTHER caller (the collector's over-budget chunking
        // included) keeps the sequential loop below → byte-identical.
        // The cut-segment context (`NY_CROWN_CUT_SEGMENT`) carries interior
        // mutability (`Cell` hit counters) and is not `Sync`; every parallel
        // caller (the IMB anchor scope) passes `cut_ctx: None`, so ctx-carrying
        // sweeps simply keep the sequential branch. Deadline-bearing chunks
        // are also sequential: launching all ranges at once would defeat the
        // bounded-work guarantee and the between-chunk deadline poll.
        let chunk_out: Vec<(usize, Vec<f32>, Vec<f32>)> = if crate::imb::anchor_chunk_parallel()
            && cut_ctx.is_none()
            && per_node_deadline.is_none()
        {
            use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
            // `ctx = None` at compile level: the cut context is `Cell`-bearing
            // (not `Sync`), and every parallel caller passes `None` anyway
            // (enforced by the `cut_ctx.is_none()` gate above).
            ranges
                .par_iter()
                .map(|r| bound_chunk(r, None))
                .collect::<Result<Vec<_>>>()?
        } else {
            ranges
                .iter()
                .map(|r| bound_chunk(r, cut_ctx))
                .collect::<Result<Vec<_>>>()?
        };
        for (r0, lo, up) in chunk_out {
            let r1 = r0 + lo.len();
            out_lower[r0..r1].copy_from_slice(&lo);
            out_upper[r0..r1].copy_from_slice(&up);
        }

        let lower = ArrayD::from_shape_vec(ndarray::IxDyn(&[target_dim]), out_lower)
            .map_err(|e| NyError::InvalidSpec(format!("objective-chunk lower reshape: {e}")))?;
        let upper = ArrayD::from_shape_vec(ndarray::IxDyn(&[target_dim]), out_upper)
            .map_err(|e| NyError::InvalidSpec(format!("objective-chunk upper reshape: {e}")))?;
        let assembled = BoundedTensor::new_allow_infinite(lower, upper)?;
        target_contract.restore_concrete(assembled, "Graph alpha-CROWN objective-chunk restore")
    }

    /// #margin-subset-seed: selector-seeded k-row variant of the full-width
    /// target backward.
    ///
    /// Seeds the backward walk with the `rows.len()` identity rows named by
    /// `rows` (arbitrary flat output positions, not necessarily contiguous)
    /// instead of the full `[target_dim x target_dim]` identity, and returns
    /// the concretized `(lower, upper)` values for exactly those rows, in
    /// `rows` order. On vggnet16's 1000-wide OUTPUT node with a 2-index margin
    /// spec this turns the `[1000 x 401408]` conv coefficient buffers
    /// (~1.6 GiB each) into `[2 x 401408]` (~3.2 MiB).
    ///
    /// SOUNDNESS / ROW-EQUIVALENCE: each identity seed row is an independent
    /// linear objective — the conv col2im scatter, the per-row CROWN error
    /// term, and the per-row concretize are all row-local (the same
    /// row-independence the #patches-obj-chunk objective chunking relies on),
    /// so every returned row is bit-identical in semantics to the same row of
    /// the full-width backward under the same walk configuration. The seed
    /// builder (`build_subset_seed`) is shared with the chunk driver.
    ///
    /// The intended caller is the CROWN-IBP collector's OUTPUT-node consume
    /// (crown_tighten.rs), which scatters these k rows over the node's sound
    /// IBP bounds and intersects with IBP exactly as for full maps; on ANY
    /// error from this method it falls back to the existing full-width path
    /// (fail-open to the old behavior). `alpha_state` is always `None` here,
    /// matching both collector entry points.
    #[allow(clippy::too_many_arguments)] // Mirrors propagate_crown_to_node_core's threading (#3549).
    pub(in crate::network::graph_alpha) fn propagate_crown_to_node_subset(
        &self,
        input: &BoundedTensor,
        target_node: &str,
        crown_bounds: &std::collections::HashMap<String, BoundedTensor>,
        ibp_bounds: &std::collections::HashMap<String, BoundedTensor>,
        engine: Option<&dyn ny_core::GemmEngine>,
        label: &str,
        per_node_deadline: Option<std::time::Instant>,
        collector_patches_override: bool,
        rows: &[usize],
        cut_ctx: Option<&CrownCutContext>,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let target_bounds = ibp_bounds.get(target_node).ok_or_else(|| {
            NyError::InvalidSpec(format!("Target node {} not in IBP bounds", target_node))
        })?;
        let target_contract = GraphTargetShapeContract::from_bounds(target_node, target_bounds);
        let target_dim = target_contract.flat_dim();
        if rows.is_empty() {
            return Err(NyError::InvalidSpec(
                "margin-subset backward requires at least one seed row".to_string(),
            ));
        }
        if let Some(&bad) = rows.iter().find(|&&row| row >= target_dim) {
            return Err(NyError::InvalidSpec(format!(
                "margin-subset seed row {bad} out of range for target '{target_node}' \
                 (flat dim {target_dim})"
            )));
        }

        let relevant_nodes = self.ancestors(target_node)?;
        if relevant_nodes.is_empty() {
            // Full-width counterpart returns `input.clone()`; mirror it for
            // the requested rows (only meaningful when the target IS the
            // network input passthrough, so the dims must agree).
            let flat = input.flatten();
            if flat.len() != target_dim {
                return Err(NyError::InvalidSpec(format!(
                    "margin-subset target '{target_node}' has no ancestors and input dim {} \
                     != target dim {target_dim}",
                    flat.len()
                )));
            }
            let lower_flat = flat.lower();
            let upper_flat = flat.upper();
            let lower = rows.iter().map(|&row| lower_flat[[row]]).collect();
            let upper = rows.iter().map(|&row| upper_flat[[row]]).collect();
            return Ok((lower, upper));
        }

        // Same cooperative GPU deadline scope as the full-width core
        // (#w4-refresh-deadline).
        let _gpu_deadline_scope =
            crate::sound_gpu_gate::GpuCrownDeadlineScope::set(engine, per_node_deadline);

        let allow_patches = target_allows_patches_start(
            self,
            target_node,
            None,
            &relevant_nodes,
            target_bounds,
            collector_patches_override,
        );
        let target_shape = target_bounds.shape();
        let spatial = if allow_patches && target_shape.len() == 3 {
            Some((target_shape[0], target_shape[1], target_shape[2]))
        } else {
            None
        };
        let seed = build_subset_seed(target_dim, rows, spatial)?;
        let gpu_suffix_plan =
            GpuSuffixPlan::build(&relevant_nodes, self, input, crown_bounds, ibp_bounds, None);
        let k = rows.len();
        // Flat [k] restore contract: the produced bounds line up 1:1 with `rows`.
        let subset_contract =
            GraphTargetShapeContract::from_bounds(target_node, &flat_bounds_view(k)?);

        let produced = self.run_target_backward_pass(
            input,
            target_node,
            crown_bounds,
            ibp_bounds,
            None,
            engine,
            label,
            per_node_deadline,
            collector_patches_override,
            relevant_nodes.as_slice(),
            &subset_contract,
            k,
            input.len(),
            allow_patches,
            &gpu_suffix_plan,
            seed,
            cut_ctx,
        )?;
        // A layer kernel may cross the deadline before returning to the
        // per-node poll; a late result must not be accepted as on-time.
        if per_node_deadline.is_some_and(|deadline| std::time::Instant::now() >= deadline) {
            return Err(NyError::DeadlineExceeded(format!(
                "{label}: per-node deadline exceeded after margin-subset backward for \
                 target '{target_node}'"
            )));
        }
        match produced {
            Some(bounds) => {
                let lower = bounds.lower();
                let upper = bounds.upper();
                let lower = lower.as_slice().ok_or_else(|| {
                    NyError::InvalidSpec("margin-subset concrete lower not contiguous".to_string())
                })?;
                let upper = upper.as_slice().ok_or_else(|| {
                    NyError::InvalidSpec("margin-subset concrete upper not contiguous".to_string())
                })?;
                if lower.len() != k || upper.len() != k {
                    return Err(NyError::InvalidSpec(format!(
                        "margin-subset backward produced {} rows, expected {k}",
                        lower.len()
                    )));
                }
                Ok((lower.to_vec(), upper.to_vec()))
            }
            None => {
                // No input contribution accumulated: pass the target rows
                // through unchanged (mirrors the single-pass
                // `target_bounds.clone()` terminal branch, restricted to
                // these rows).
                let flat = target_bounds.flatten();
                let lower_flat = flat.lower();
                let upper_flat = flat.upper();
                let lower = rows.iter().map(|&row| lower_flat[[row]]).collect();
                let upper = rows.iter().map(|&row| upper_flat[[row]]).collect();
                Ok((lower, upper))
            }
        }
    }
}

/// Build a minimal placeholder `[n]`-shaped BoundedTensor used only to derive a
/// flat `GraphTargetShapeContract` (1D restore) for an objective chunk.
fn flat_bounds_view(n: usize) -> Result<BoundedTensor> {
    let lower = ArrayD::from_elem(ndarray::IxDyn(&[n]), 0.0_f32);
    let upper = ArrayD::from_elem(ndarray::IxDyn(&[n]), 0.0_f32);
    BoundedTensor::new(lower, upper)
}

/// Build the seed `CrownBounds` for objective rows `r0..r1` (count
/// `chunk_rows = r1 - r0`) of a `target_dim`-row identity.
///
/// - When `spatial` is `Some((c, h, w))`, returns a sparse-identity Patches seed
///   whose `unstable_idx` enumerates exactly the flat positions `r0..r1` of the
///   `(c, h, w)` output grid. The sparse patches `to_dense()` scatter reproduces,
///   per row, the same dense row as the full identity at that flat index
///   (proven by crown_patches.rs Test 6), so the chunk seed is row-equivalent.
/// - Otherwise returns a Dense `[chunk_rows x target_dim]` slice of the identity.
fn build_chunk_seed(
    target_dim: usize,
    r0: usize,
    r1: usize,
    spatial: Option<(usize, usize, usize)>,
) -> Result<CrownBounds> {
    let rows: Vec<usize> = (r0..r1).collect();
    build_subset_seed(target_dim, &rows, spatial)
}

/// Build the seed `CrownBounds` for an ARBITRARY set of objective `rows` of a
/// `target_dim`-row identity. The contiguous chunk seed above is the special
/// case `rows = r0..r1`; the margin-subset backward (#margin-subset-seed)
/// passes the (sorted, possibly non-contiguous) spec-referenced rows.
///
/// - When `spatial` is `Some((c, h, w))`, returns a sparse-identity Patches
///   seed whose `unstable_idx` enumerates exactly the flat positions in `rows`
///   of the `(c, h, w)` output grid. The sparse patches `to_dense()` scatter
///   reproduces, per row, the same dense row as the full identity at that flat
///   index (proven by crown_patches.rs Test 6), so the seed is row-equivalent.
/// - Otherwise returns a Dense `[rows.len() x target_dim]` selection of
///   identity rows.
fn build_subset_seed(
    target_dim: usize,
    rows: &[usize],
    spatial: Option<(usize, usize, usize)>,
) -> Result<CrownBounds> {
    let num_rows = rows.len();
    if let Some((out_c, out_h, out_w)) = spatial {
        // Decompose each flat output index into (channel, height, width).
        let plane = out_h * out_w;
        let mut channels = Vec::with_capacity(num_rows);
        let mut heights = Vec::with_capacity(num_rows);
        let mut widths = Vec::with_capacity(num_rows);
        for &flat in rows {
            let c = flat / plane;
            let rem = flat % plane;
            let h = rem / out_w;
            let w = rem % out_w;
            channels.push(c);
            heights.push(h);
            widths.push(w);
        }
        let idx = crate::bounds::patches::UnstableIdx {
            channels,
            heights,
            widths,
        };
        idx.validate(out_c, out_h, out_w, Some(num_rows))?;
        Ok(CrownBounds::Patches(Box::new(
            PatchesLinearBounds::sparse_identity((out_c, out_h, out_w), (out_c, out_h, out_w), idx),
        )))
    } else {
        // Dense identity rows at the requested positions.
        let mut lower_a = Array2::<f32>::zeros((num_rows, target_dim));
        let mut upper_a = Array2::<f32>::zeros((num_rows, target_dim));
        for (i, &col) in rows.iter().enumerate() {
            lower_a[[i, col]] = 1.0;
            upper_a[[i, col]] = 1.0;
        }
        let lb = LinearBounds::new(
            lower_a,
            Array1::zeros(num_rows),
            upper_a,
            Array1::zeros(num_rows),
        )?;
        Ok(CrownBounds::Dense(lb))
    }
}

#[cfg(test)]
mod patches_densify_budget_tests {
    use super::patches_densify_over_budget;
    use crate::bounds::patches::PatchesLinearBounds;

    /// #patches-row-range: the mid-walk densify guard refuses a Patches
    /// relation whose dense pair exceeds the budget with the structured
    /// `CpuMemoryExceeded` (mapped to a sound fallback by every CROWN caller),
    /// and passes under-budget relations through untouched.
    #[test]
    fn midwalk_densify_budget_guard() {
        // 1x2x2 identity: dense pair is 4x4x4x2 = 128 bytes — under budget.
        let small = PatchesLinearBounds::identity((1, 2, 2), (1, 2, 2));
        assert!(patches_densify_over_budget(&small, 1024).is_none());
        // Exactly at the budget is NOT over budget (strict >).
        let small_bytes = small.dense_pair_bytes().expect("estimable");
        assert!(patches_densify_over_budget(&small, small_bytes).is_none());
        assert!(patches_densify_over_budget(&small, small_bytes - 1).is_some());

        // VGG16 conv1 scale: 64x224x224 rows against a 3x224x224 input is a
        // ~3.9 TB dense pair — over any real budget, including the 2 GiB
        // default. (No dense allocation happens; identity patches are virtual.)
        let huge = PatchesLinearBounds::identity((64, 224, 224), (3, 224, 224));
        let err = patches_densify_over_budget(&huge, 2048 * 1024 * 1024)
            .expect("VGG-scale dense pair must trip the budget guard");
        assert!(err.is_cpu_memory_exceeded());
        // A large-enough budget lets it through (fits => None).
        assert!(patches_densify_over_budget(&huge, usize::MAX).is_none());
    }
}

#[cfg(test)]
mod margin_subset_backward_tests {
    use crate::layers::{Conv2dLayer, Layer, LinearLayer, ReLULayer};
    use crate::network::core::{GraphNetwork, GraphNode};
    use ndarray::{arr1, arr2, ArrayD, IxDyn};
    use ny_tensor::BoundedTensor;
    use std::collections::HashMap;

    /// input(2) -> Linear(2->3) -> ReLU -> Linear(3->6): unstable ReLUs make
    /// the CROWN relaxation non-trivial, so row equality is a real check.
    fn dense_chain() -> (GraphNetwork, BoundedTensor) {
        let l1 = LinearLayer::new(
            arr2(&[[1.0_f32, -0.5], [0.25, 0.75], [-0.6, 0.4]]),
            Some(arr1(&[0.05_f32, -0.1, 0.02])),
        )
        .expect("l1");
        let l2 = LinearLayer::new(
            arr2(&[
                [0.9_f32, -0.3, 0.2],
                [-0.7, 0.6, -0.1],
                [0.4, 0.4, 0.4],
                [-0.2, -0.8, 0.5],
                [0.3, -0.6, -0.9],
                [0.8, 0.1, -0.4],
            ]),
            Some(arr1(&[0.01_f32, -0.02, 0.03, 0.0, -0.05, 0.04])),
        )
        .expect("l2");
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("l1", Layer::Linear(l1)));
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["l1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "out",
            Layer::Linear(l2),
            vec!["relu".to_string()],
        ));
        graph.set_output("out");
        let input = BoundedTensor::new(
            arr1(&[-1.0_f32, -0.5]).into_dyn(),
            arr1(&[1.0_f32, 0.75]).into_dyn(),
        )
        .expect("input");
        (graph, input)
    }

    /// #margin-subset-seed: the subset backward's rows are BIT-IDENTICAL to
    /// the same rows of the full-width backward (dense identity seed path).
    #[test]
    fn subset_rows_match_full_backward_bit_identical_dense() {
        let (graph, input) = dense_chain();
        let forward = graph.collect_node_bounds(&input).expect("forward bounds");

        let full = graph
            .propagate_crown_to_node(
                &input,
                "out",
                &HashMap::new(),
                &forward,
                None,
                None,
                Some(0),
                None,
            )
            .expect("full-width backward");
        let full_flat = full.flatten();
        let rows = [1_usize, 4];
        let (lower, upper) = graph
            .propagate_crown_to_node_subset(
                &input,
                "out",
                &HashMap::new(),
                &forward,
                None,
                "margin-subset-test",
                None,
                false,
                &rows,
                None,
            )
            .expect("subset backward");
        for (i, &row) in rows.iter().enumerate() {
            assert_eq!(
                lower[i],
                full_flat.lower()[[row]],
                "lower row {row} must be bit-identical"
            );
            assert_eq!(
                upper[i],
                full_flat.upper()[[row]],
                "upper row {row} must be bit-identical"
            );
        }
        // Meaningfulness guard: the CROWN rows must actually tighten the IBP
        // rows somewhere among the chosen indices, or this test proves nothing.
        let ibp_out = forward.get("out").expect("out IBP").flatten();
        assert!(
            rows.iter().any(|&row| {
                full_flat.lower()[[row]] > ibp_out.lower()[[row]]
                    || full_flat.upper()[[row]] < ibp_out.upper()[[row]]
            }),
            "CROWN must beat IBP on at least one tested row"
        );
    }

    /// Spatial conv target via the collector (patches-override) walk: subset
    /// rows must match the full collector backward bit-for-bit as well.
    #[test]
    fn subset_rows_match_full_backward_bit_identical_conv_collector() {
        let kernel = ArrayD::from_shape_vec(IxDyn(&[2, 2, 1, 1]), vec![0.9_f32, -0.35, -0.45, 0.8])
            .expect("kernel");
        let conv = Conv2dLayer::with_input_shape(
            kernel,
            Some(arr1(&[0.05_f32, -0.1])),
            (1, 1),
            (0, 0),
            2,
            2,
        )
        .expect("conv");
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("conv", Layer::Conv2d(conv)));
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["conv".to_string()],
        ));
        graph.set_output("relu");
        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(
                IxDyn(&[2, 2, 2]),
                vec![-1.0_f32, -0.6, 0.1, -0.3, -0.5, -0.2, 0.0, -0.4],
            )
            .expect("lower"),
            ArrayD::from_shape_vec(
                IxDyn(&[2, 2, 2]),
                vec![1.2_f32, 0.7, 0.9, 0.6, 0.8, 0.5, 1.0, 0.4],
            )
            .expect("upper"),
        )
        .expect("input");
        let forward = graph.collect_node_bounds(&input).expect("forward bounds");

        let full = graph
            .propagate_crown_to_node_for_collector(
                &input,
                "relu",
                &HashMap::new(),
                &forward,
                None,
                None,
                Some(0),
                None,
            )
            .expect("full collector backward");
        let full_flat = full.flatten();
        let rows = [0_usize, 3, 7];
        let (lower, upper) = graph
            .propagate_crown_to_node_subset(
                &input,
                "relu",
                &HashMap::new(),
                &forward,
                None,
                "margin-subset-test",
                None,
                true,
                &rows,
                None,
            )
            .expect("subset collector backward");
        for (i, &row) in rows.iter().enumerate() {
            assert_eq!(
                lower[i],
                full_flat.lower()[[row]],
                "lower row {row} must be bit-identical"
            );
            assert_eq!(
                upper[i],
                full_flat.upper()[[row]],
                "upper row {row} must be bit-identical"
            );
        }
    }

    /// Fail-closed request validation: empty and out-of-range row sets error
    /// (the consume site maps any error to the full-width fallback).
    #[test]
    fn subset_rejects_empty_and_out_of_range_rows() {
        let (graph, input) = dense_chain();
        let forward = graph.collect_node_bounds(&input).expect("forward bounds");
        assert!(graph
            .propagate_crown_to_node_subset(
                &input,
                "out",
                &HashMap::new(),
                &forward,
                None,
                "margin-subset-test",
                None,
                false,
                &[],
                None,
            )
            .is_err());
        assert!(graph
            .propagate_crown_to_node_subset(
                &input,
                "out",
                &HashMap::new(),
                &forward,
                None,
                "margin-subset-test",
                None,
                false,
                &[6],
                None,
            )
            .is_err());
    }
}

#[cfg(test)]
mod target_input_linear_tests {
    use super::{
        target_allows_patches_start, target_input_linear_chunk_row_bytes,
        target_input_linear_chunk_rows, target_input_linear_fixed_bytes, TargetInputLinearLimits,
    };
    use crate::layers::{
        AddLayer, Conv2dLayer, ConvTranspose2dLayer, Layer, LinearLayer, ReLULayer,
    };
    use crate::network::core::{GraphNetwork, GraphNode, NETWORK_INPUT};
    use crate::tests::with_serialized_env_vars;
    use ndarray::{arr1, arr2, ArrayD, IxDyn};
    use ny_core::NyError;
    use ny_tensor::BoundedTensor;
    use std::collections::HashMap;
    use std::time::{Duration, Instant};

    fn affine_relu() -> (GraphNetwork, BoundedTensor) {
        let affine = LinearLayer::new(
            arr2(&[[0.7_f32, -0.2], [-0.35, 0.8], [0.45, 0.55]]),
            Some(arr1(&[0.1_f32, -0.05, 0.02])),
        )
        .expect("affine");
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("affine", Layer::Linear(affine)));
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["affine".to_string()],
        ));
        graph.set_output("relu");
        let input = BoundedTensor::new(
            arr1(&[-1.0_f32, -0.6]).into_dyn(),
            arr1(&[0.9_f32, 1.1]).into_dyn(),
        )
        .expect("input");
        (graph, input)
    }

    fn aggregate_plan(
        graph: &GraphNetwork,
        input: &BoundedTensor,
        target: &str,
        crown: &HashMap<String, BoundedTensor>,
        ibp: &HashMap<String, BoundedTensor>,
    ) -> (usize, usize) {
        let input_dim = input.len();
        let target_dim = if target == NETWORK_INPUT {
            input_dim
        } else {
            ibp.get(target).expect("target bound").len()
        };
        let fixed =
            target_input_linear_fixed_bytes(target_dim, input_dim).expect("fixed workspace bytes");
        if target == NETWORK_INPUT {
            return (fixed, 0);
        }
        let relevant = graph.ancestors(target).expect("target ancestors");
        let mut relation_cols_sum = target_dim + input_dim;
        let mut relation_count = 2usize;
        for name in &relevant {
            relation_cols_sum += crown
                .get(name)
                .or_else(|| ibp.get(name))
                .expect("relation bound")
                .len();
            relation_count += 1;
        }
        let chunk_row =
            target_input_linear_chunk_row_bytes(target_dim, relation_cols_sum, relation_count)
                .expect("chunk-row bytes");
        (fixed, chunk_row)
    }

    fn assert_certificate_encloses_direct(
        certificate: &crate::bounds::LinearBounds,
        input: &BoundedTensor,
        direct: &BoundedTensor,
    ) {
        let from_certificate = certificate.concretize_sound(input);
        let direct = direct.flatten();
        assert_eq!(from_certificate.len(), direct.len());
        for row in 0..direct.len() {
            assert!(
                from_certificate.lower()[[row]] <= direct.lower()[[row]],
                "certificate lower row {row}={} exceeds direct CROWN lower={}",
                from_certificate.lower()[[row]],
                direct.lower()[[row]]
            );
            assert!(
                from_certificate.upper()[[row]] >= direct.upper()[[row]],
                "certificate upper row {row}={} is below direct CROWN upper={}",
                from_certificate.upper()[[row]],
                direct.upper()[[row]]
            );
        }
    }

    #[test]
    fn input_linear_forced_chunks_stitch_raw_error_then_publicly_fold_it() {
        let (graph, input) = affine_relu();
        let ibp = graph.collect_node_bounds(&input).expect("IBP");
        let direct = graph
            .propagate_crown_to_node(
                &input,
                "relu",
                &HashMap::new(),
                &ibp,
                None,
                None,
                Some(0),
                None,
            )
            .expect("direct CROWN");
        let full = graph
            .propagate_crown_input_linear_to_node(&input, "relu", &HashMap::new(), &ibp, None, None)
            .expect("input-linear CROWN");
        assert_eq!((full.num_outputs(), full.num_inputs()), (3, 2));
        assert!(!full.has_coeff_err(), "public error must be discharged");

        let raw_full = graph
            .capture_crown_input_linear_to_node_raw_with_limits(
                &input,
                "relu",
                &HashMap::new(),
                &ibp,
                None,
                None,
                TargetInputLinearLimits::PRODUCTION,
            )
            .expect("raw input-linear CROWN");
        assert!(
            raw_full.has_coeff_err(),
            "affine backward should produce a private certified error channel"
        );
        let flat_input = input.flatten();
        let mut expected_public = raw_full.clone();
        expected_public.fold_coeff_err_into_bias(
            flat_input.lower().as_slice().expect("lower slice"),
            flat_input.upper().as_slice().expect("upper slice"),
        );
        assert!(!expected_public.has_coeff_err());
        assert_eq!(full.lower_a(), expected_public.lower_a());
        assert_eq!(full.upper_a(), expected_public.upper_a());
        assert_eq!(full.lower_b(), expected_public.lower_b());
        assert_eq!(full.upper_b(), expected_public.upper_b());

        // Set the aggregate cap to exactly assembly + one conservative chunk
        // row. This deterministically forces one identity row per pass.
        let (fixed, chunk_row) = aggregate_plan(&graph, &input, "relu", &HashMap::new(), &ibp);
        let one_row_cap = fixed + chunk_row;
        assert_eq!(
            target_input_linear_chunk_rows(3, fixed, chunk_row, false, false, one_row_cap)
                .expect("one-row plan"),
            1
        );
        let raw_chunked = graph
            .capture_crown_input_linear_to_node_raw_with_limits(
                &input,
                "relu",
                &HashMap::new(),
                &ibp,
                None,
                None,
                TargetInputLinearLimits {
                    aggregate_bytes: one_row_cap,
                },
            )
            .expect("one-row raw input-linear CROWN");
        assert_eq!(raw_chunked.lower_a(), raw_full.lower_a());
        assert_eq!(raw_chunked.upper_a(), raw_full.upper_a());
        assert_eq!(raw_chunked.lower_b(), raw_full.lower_b());
        assert_eq!(raw_chunked.upper_b(), raw_full.upper_b());
        assert_eq!(raw_chunked.lower_a_err(), raw_full.lower_a_err());
        assert_eq!(raw_chunked.upper_a_err(), raw_full.upper_a_err());

        let chunked = graph
            .propagate_crown_input_linear_to_node_with_limits(
                &input,
                "relu",
                &HashMap::new(),
                &ibp,
                None,
                None,
                TargetInputLinearLimits {
                    aggregate_bytes: one_row_cap,
                },
            )
            .expect("one-row input-linear CROWN");
        assert_eq!(chunked.lower_a(), full.lower_a());
        assert_eq!(chunked.upper_a(), full.upper_a());
        assert_eq!(chunked.lower_b(), full.lower_b());
        assert_eq!(chunked.upper_b(), full.upper_b());
        assert!(!chunked.has_coeff_err());
        assert_certificate_encloses_direct(&full, &input, &direct);
    }

    #[test]
    fn input_linear_preflight_rejects_fixed_and_one_row_aggregate_peaks() {
        let (graph, input) = affine_relu();
        let ibp = graph.collect_node_bounds(&input).expect("IBP");
        let (fixed, chunk_row) = aggregate_plan(&graph, &input, "relu", &HashMap::new(), &ibp);

        let fixed_err = graph
            .propagate_crown_input_linear_to_node_with_limits(
                &input,
                "relu",
                &HashMap::new(),
                &ibp,
                None,
                None,
                TargetInputLinearLimits {
                    aggregate_bytes: fixed - 1,
                },
            )
            .expect_err("fixed assembly must not exceed the aggregate cap");
        assert!(matches!(
            fixed_err,
            NyError::CpuMemoryExceeded {
                site: "graph-alpha target input-linear aggregate workspace",
                ..
            }
        ));

        let row_err = graph
            .propagate_crown_input_linear_to_node_with_limits(
                &input,
                "relu",
                &HashMap::new(),
                &ibp,
                None,
                None,
                TargetInputLinearLimits {
                    aggregate_bytes: fixed + chunk_row - 1,
                },
            )
            .expect_err("assembly plus one chunk row must not exceed the aggregate cap");
        assert!(matches!(
            row_err,
            NyError::CpuMemoryExceeded {
                site: "graph-alpha target input-linear aggregate workspace",
                ..
            }
        ));
    }

    #[test]
    fn input_linear_rejects_unknown_target_even_with_spoofed_bounds() {
        let (graph, input) = affine_relu();
        let mut ibp = graph.collect_node_bounds(&input).expect("IBP");
        ibp.insert(
            "ghost".to_string(),
            ibp.get("relu").expect("relu bound").clone(),
        );
        let err = graph
            .propagate_crown_input_linear_to_node(
                &input,
                "ghost",
                &HashMap::new(),
                &ibp,
                None,
                None,
            )
            .expect_err("unknown target must fail closed");
        assert!(
            matches!(err, NyError::InvalidSpec(message) if message.contains("not a graph node"))
        );
    }

    #[test]
    fn input_linear_accepts_only_the_network_input_sentinel_without_a_graph_node() {
        let (graph, input) = affine_relu();
        let certificate = graph
            .propagate_crown_input_linear_to_node(
                &input,
                NETWORK_INPUT,
                &HashMap::new(),
                &HashMap::new(),
                None,
                None,
            )
            .expect("network-input identity");
        assert_eq!(
            (certificate.num_outputs(), certificate.num_inputs()),
            (2, 2)
        );
        assert!(!certificate.has_coeff_err());
        assert_certificate_encloses_direct(&certificate, &input, &input);
    }

    #[test]
    fn input_linear_captures_dag_merge_and_input_skip() {
        let first = LinearLayer::new(
            arr2(&[[0.8_f32, -0.3], [0.4, 0.9]]),
            Some(arr1(&[0.1_f32, -0.05])),
        )
        .expect("first affine");
        let second = LinearLayer::new(
            arr2(&[[0.6_f32, -0.2], [-0.4, 0.7]]),
            Some(arr1(&[0.0_f32, 0.0])),
        )
        .expect("second affine");
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("first", Layer::Linear(first)));
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["first".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "second",
            Layer::Linear(second),
            vec!["relu".to_string()],
        ));
        graph.add_node(GraphNode::binary(
            "residual",
            Layer::Add(AddLayer),
            NETWORK_INPUT,
            "second",
        ));
        graph.set_output("residual");
        let input = BoundedTensor::new(
            arr1(&[-0.5_f32, -0.5]).into_dyn(),
            arr1(&[0.5_f32, 0.5]).into_dyn(),
        )
        .expect("input");
        let ibp = graph.collect_node_bounds(&input).expect("IBP");
        let direct = graph
            .propagate_crown_to_node(
                &input,
                "residual",
                &HashMap::new(),
                &ibp,
                None,
                None,
                Some(0),
                None,
            )
            .expect("direct DAG CROWN");
        let certificate = graph
            .propagate_crown_input_linear_to_node(
                &input,
                "residual",
                &HashMap::new(),
                &ibp,
                None,
                None,
            )
            .expect("DAG input-linear CROWN");
        assert_eq!(
            (certificate.num_outputs(), certificate.num_inputs()),
            (2, 2)
        );
        assert!(!certificate.has_coeff_err());
        assert_certificate_encloses_direct(&certificate, &input, &direct);
    }

    #[test]
    fn input_linear_captures_patches_conv_chain() {
        let in_c = 4usize;
        let in_h = 20usize;
        let in_w = 25usize;
        let input_dim = in_c * in_h * in_w;
        let conv1 = Conv2dLayer::with_input_shape(
            ArrayD::from_elem(IxDyn(&[2, in_c, 2, 2]), 0.125_f32),
            Some(arr1(&[0.03_f32, -0.02])),
            (2, 2),
            (0, 0),
            in_h,
            in_w,
        )
        .expect("conv1");
        let conv2 = Conv2dLayer::with_input_shape(
            ArrayD::from_elem(IxDyn(&[1, 2, 3, 3]), -0.075_f32),
            Some(arr1(&[0.01_f32])),
            (1, 1),
            (0, 0),
            10,
            12,
        )
        .expect("conv2");
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("conv1", Layer::Conv2d(conv1)));
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["conv1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "conv2",
            Layer::Conv2d(conv2),
            vec!["relu".to_string()],
        ));
        graph.set_output("conv2");
        graph.set_use_patches_mode(true);
        let input = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[in_c, in_h, in_w]), -0.5_f32),
            ArrayD::from_elem(IxDyn(&[in_c, in_h, in_w]), 0.5_f32),
        )
        .expect("input");
        assert_eq!(input.len(), input_dim);
        let ibp = graph.collect_node_bounds(&input).expect("IBP");
        let target = ibp.get("conv2").expect("target bound");
        let relevant = graph.ancestors("conv2").expect("ancestors");

        let (certificate, direct) =
            with_serialized_env_vars(&[("NY_DENSE_BUDGET_MB", "1")], || {
                assert!(
                    target_allows_patches_start(&graph, "conv2", None, &relevant, target, false,),
                    "fixture must start capture in Patches form"
                );
                let certificate = graph
                    .propagate_crown_input_linear_to_node(
                        &input,
                        "conv2",
                        &HashMap::new(),
                        &ibp,
                        None,
                        None,
                    )
                    .expect("Patches input-linear CROWN");
                let direct = graph
                    .propagate_crown_to_node(
                        &input,
                        "conv2",
                        &HashMap::new(),
                        &ibp,
                        None,
                        None,
                        Some(0),
                        None,
                    )
                    .expect("direct Patches CROWN");
                (certificate, direct)
            });
        assert_eq!(
            (certificate.num_outputs(), certificate.num_inputs()),
            (80, input_dim)
        );
        assert!(!certificate.has_coeff_err());
        assert_certificate_encloses_direct(&certificate, &input, &direct);
    }

    #[test]
    fn input_linear_captures_deadline_chunked_conv_transpose() {
        let conv = ConvTranspose2dLayer::with_input_shape(
            ArrayD::from_elem(IxDyn(&[1, 1, 1, 1]), -0.75_f32),
            Some(arr1(&[0.125_f32])),
            (1, 1),
            (0, 0),
            1,
            33,
        )
        .expect("ConvTranspose2d");
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("convt", Layer::ConvTranspose2d(conv)));
        graph.set_output("convt");
        let input = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[1, 1, 33]), -0.4_f32),
            ArrayD::from_elem(IxDyn(&[1, 1, 33]), 0.6_f32),
        )
        .expect("input");
        let ibp = graph.collect_node_bounds(&input).expect("IBP");
        let direct = graph
            .propagate_crown_to_node(
                &input,
                "convt",
                &HashMap::new(),
                &ibp,
                None,
                None,
                Some(0),
                None,
            )
            .expect("direct ConvTranspose CROWN");
        let certificate = graph
            .propagate_crown_input_linear_to_node(
                &input,
                "convt",
                &HashMap::new(),
                &ibp,
                None,
                Some(Instant::now() + Duration::from_secs(10)),
            )
            .expect("deadline-chunked ConvTranspose input-linear CROWN");
        assert_eq!(
            (certificate.num_outputs(), certificate.num_inputs()),
            (33, 33)
        );
        assert!(!certificate.has_coeff_err());
        assert_certificate_encloses_direct(&certificate, &input, &direct);
    }

    #[test]
    fn input_linear_rejects_expired_deadline_before_allocation() {
        let (graph, input) = affine_relu();
        let ibp = graph.collect_node_bounds(&input).expect("IBP");
        let expired = Instant::now()
            .checked_sub(Duration::from_millis(1))
            .expect("monotonic subtraction");
        let err = graph
            .propagate_crown_input_linear_to_node(
                &input,
                "relu",
                &HashMap::new(),
                &ibp,
                None,
                Some(expired),
            )
            .expect_err("expired deadline must fail");
        assert!(matches!(err, NyError::DeadlineExceeded(_)));
    }
}

#[cfg(test)]
mod cut_segment_env_tests {
    use super::{
        effective_target_chunk_size, parse_crown_cut_segment,
        DEADLINE_CONV_TRANSPOSE_OBJ_CHUNK_ROWS,
    };
    use crate::layers::{ConvTranspose2dLayer, Layer};
    use crate::network::core::{GraphNetwork, GraphNode};
    use ndarray::{arr1, ArrayD, IxDyn};
    use ny_core::{GemmEngine, NaiveCpuGemmEngine, Result};
    use ny_tensor::BoundedTensor;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    #[derive(Default)]
    struct CountingGemmEngine {
        calls: AtomicUsize,
    }

    impl GemmEngine for CountingGemmEngine {
        fn gemm_f32(&self, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            NaiveCpuGemmEngine.gemm_f32(m, k, n, a, b)
        }
    }

    fn direct_conv_transpose_33() -> (GraphNetwork, BoundedTensor) {
        let conv = ConvTranspose2dLayer::with_input_shape(
            ArrayD::from_elem(IxDyn(&[1, 1, 1, 1]), -0.75_f32),
            Some(arr1(&[0.125_f32])),
            (1, 1),
            (0, 0),
            1,
            33,
        )
        .expect("1x1 ConvTranspose2d");
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("convt", Layer::ConvTranspose2d(conv)));
        graph.set_output("convt");
        let input = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[1, 1, 33]), -0.4_f32),
            ArrayD::from_elem(IxDyn(&[1, 1, 33]), 0.6_f32),
        )
        .expect("input box");
        (graph, input)
    }

    /// #crown-cut-segment gate parsing: unset/empty/non-numeric/0 all mean
    /// DISABLED (byte-identical full-prefix backward); only a positive integer
    /// enables cuts.
    #[test]
    fn parse_crown_cut_segment_defaults_off() {
        assert_eq!(parse_crown_cut_segment(None), 0);
        assert_eq!(parse_crown_cut_segment(Some("")), 0);
        assert_eq!(parse_crown_cut_segment(Some("0")), 0);
        assert_eq!(parse_crown_cut_segment(Some("abc")), 0);
        assert_eq!(parse_crown_cut_segment(Some("-4")), 0);
        assert_eq!(parse_crown_cut_segment(Some("1.5")), 0);
    }

    #[test]
    fn parse_crown_cut_segment_accepts_positive() {
        assert_eq!(parse_crown_cut_segment(Some("1")), 1);
        assert_eq!(parse_crown_cut_segment(Some("4")), 4);
        assert_eq!(parse_crown_cut_segment(Some(" 8 ")), 8);
    }

    #[test]
    fn deadline_conv_transpose_chunk_cap_preserves_no_deadline_policy() {
        let cap = DEADLINE_CONV_TRANSPOSE_OBJ_CHUNK_ROWS;
        assert_eq!(effective_target_chunk_size(0, false, true), 0);
        assert_eq!(effective_target_chunk_size(64, false, true), 64);
        assert_eq!(effective_target_chunk_size(0, true, false), 0);
        assert_eq!(effective_target_chunk_size(0, true, true), cap);
        assert_eq!(effective_target_chunk_size(cap * 2, true, true), cap);
        assert_eq!(effective_target_chunk_size(7, true, true), 7);
    }

    /// A 33-row direct ConvTranspose target takes one coefficient GEMM pair
    /// without a deadline, but two pairs under the 32-row deadline cap.  The
    /// assembled rows remain bit-identical, proving the safety route really
    /// streams the objective and preserves row-independent arithmetic.
    ///
    /// Serialized on the shared env lock: the ConvTranspose dead-work skip
    /// (#wall-deadwork port, default-on) bypasses the engine f32 pair the GEMM
    /// call counts below assert on — pin the kill-switch ("0") to keep
    /// exercising the pair path.
    #[test]
    fn deadline_conv_transpose_route_chunks_and_is_row_identical() {
        crate::tests::with_serialized_env_vars(&[("NY_CONV_SKIP_DEAD_F32", "0")], || {
            deadline_conv_transpose_route_chunks_and_is_row_identical_body();
        });
    }

    fn deadline_conv_transpose_route_chunks_and_is_row_identical_body() {
        let (graph, input) = direct_conv_transpose_33();
        let forward = graph.collect_node_bounds(&input).expect("forward bounds");

        let flat_engine = CountingGemmEngine::default();
        let flat = graph
            .propagate_crown_to_node(
                &input,
                "convt",
                &HashMap::new(),
                &forward,
                Some(&flat_engine),
                None,
                Some(0),
                None,
            )
            .expect("no-deadline flat ConvTranspose backward");
        assert_eq!(flat_engine.calls.load(Ordering::Relaxed), 2);

        let chunk_engine = CountingGemmEngine::default();
        let chunked = graph
            .propagate_crown_to_node(
                &input,
                "convt",
                &HashMap::new(),
                &forward,
                Some(&chunk_engine),
                Some(Instant::now() + Duration::from_secs(30)),
                Some(0),
                None,
            )
            .expect("deadline-bearing ConvTranspose backward");
        assert_eq!(
            chunk_engine.calls.load(Ordering::Relaxed),
            4,
            "33 rows must stream as 32+1, with one lower/upper GEMM pair per chunk"
        );
        assert_eq!(chunked.lower(), flat.lower());
        assert_eq!(chunked.upper(), flat.upper());
    }
}
