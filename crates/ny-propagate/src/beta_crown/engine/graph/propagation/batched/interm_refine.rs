// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Per-subdomain INTERMEDIATE-BOUND refinement for the resident resnet lane
//! (#interm-refine, dark gate `NY_INTERM_REFINE=1`, default OFF = byte-identical).
//!
//! WHY: with frozen intermediate bounds,
//! β-CROWN provably caps at the split-constrained LP optimum; SOTA exceeds it
//! ONLY by re-optimizing per-subdomain intermediate pre-activation bounds
//! (auto_LiRPA `fix_intermediate_layer_bounds=False` / Clip-and-Verify complete
//! clipping). Measured on cifar100_2024 resnet_medium prop885: 100.0% of split
//! premises live on `Relu_57` (the LAST ReLU, 512-d, feeding the output Gemm),
//! whose per-subdomain pre-activation bounds stay frozen at ROOT values — so the
//! final-layer relaxation never tightens as splits accumulate and children never
//! prune (frontier explosion at depth ≥ 9).
//!
//! WHAT (first increment, narrowly targeted where the tree lives): for each BaB
//! subdomain, ONE extra backward with `num_specs = pre_dim` identity rows seeded
//! at the last ReLU's INPUT node, over the TRUNCATED resnet stack (that node down
//! to the network input), folding the subdomain's α exactly like the margin
//! backward (the same `prep_resnet_domain` machinery) — concretized over the
//! subdomain's input box by the SAME sound resident GPU path (certified error,
//! directed rounding). The per-neuron `[l', u']` is INTERSECTED with the
//! inherited entry (`l = max`, `u = min`).
//!
//! SOUNDNESS:
//! * The refinement backward is a plain sound α-CROWN backward from the seed
//!   node: β terms for splits AT the seed layer (the last ReLU) do NOT appear —
//!   they constrain the seed node itself, and their pre-activation clamps are
//!   already present in the inherited cache entry (`apply_pre_constraints`).
//!   β entries for ReLUs BELOW the seed (none measured on cifar100) fold via the
//!   truncated `beta_signed` — a valid Lagrangian dual for the same subdomain.
//! * Both the inherited entry and the refined bounds are sound enclosures of the
//!   SAME subdomain's reachable pre-activations, so their intersection is sound.
//!   An empty per-neuron intersection (possible only for an infeasible domain or
//!   f32 edge noise) conservatively keeps the inherited pair.
//! * The ReLU's own POST-activation entry is tightened to
//!   `[relu(l'), relu(u')] ∩ inherited` (relu is exact and monotone).
//!
//! COST CONTROL: runs ONCE per subdomain (the dense-spec driver bounds each
//! child exactly once); one batched GPU call for the whole domain batch through
//! the SAME wide machinery as the margin backward (serial per-domain fallback on
//! error, keep-inherited on any per-domain refusal — sound).
//!
//! INFEASIBILITY PRUNING (`NY_INTERM_REFINE_PRUNE=1`, dark, default OFF): each
//! of the domain's β entries at the seed ReLU IS one of its split premises
//! (`z_j ≥ s` active / `z_j ≤ s` inactive). The refined `[l', u']` is a sound
//! enclosure of `z_j` over the subdomain WITHOUT the seed-layer premises (they
//! are excluded from the truncated stack by construction), so a strict
//! contradiction — active premise with `u' < s − tol`, inactive with
//! `l' > s + tol` — proves the subdomain's constraint set EMPTY: it verifies
//! vacuously. The refined bounds carry certified directed-rounding error
//! outward already; `tol` (`NY_INTERM_REFINE_PRUNE_TOL`, default `1e-4`) is
//! defense-in-depth on top. Because premise-clamped neurons are STABLE in the
//! inherited entry (the clamp), the prune lane force-includes premise neurons
//! in the row selection — the `unstable` arm alone would exclude exactly the
//! rows that can prove infeasibility (measured: v3 unstable-rows logs had
//! crossings_kept=0 while the all-rows arms saw up to 8/batch).
//!
//! MULTI-LAYER CASCADE (`NY_INTERM_REFINE_LAYERS=2`, dark, default 1): also
//! refine the SECOND-to-last ReLU's pre-activation entry (on resnet_medium:
//! `Relu_51`, seed `Conv_49` — a unary conv off the last residual trunk `Add_48`,
//! decomposable by the same segment extraction). Passes run DEEPEST-FIRST so
//! the last-ReLU pass (and then the margin backward) consumes the improved
//! upstream entry through the ReLU relaxations the extraction derives from the
//! caches. Deep seed layers are conv-wide (2048 on resnet_medium), so the deep
//! pass caps identity rows at `NY_INTERM_REFINE_DEEP_MAX_ROWS` (default 256):
//! premise rows always kept, the rest top-K by inherited width (row selection
//! is cost-only, never soundness).

use std::collections::HashMap;
use std::mem::size_of;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use ndarray::{Array1, Array2};
use ny_core::{
    nan_propagating_max, nan_propagating_min, GemmEngine, GpuResidentCoeffBatched, NyError,
};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use crate::batched_domain::CachedLinearBounds;
use crate::beta_crown::bab_cuts::CutFoldScope;
use crate::beta_crown::branching::{GraphNeuronConstraint, GraphSplitHistory};
use crate::beta_crown::domain::{MultiObjectiveGraphBabDomain, NodeBoundsView};
use crate::beta_crown::state::{GraphBetaState, GraphDomainAlphaState};
use crate::clip_interm_domain::{
    build_split_constraints_with_deadline_check, sort_out_constraints_with_deadline_check,
    tighten_with_constraints_with_deadline,
};
use crate::complete_clip::{
    bind_root_sound_crown_rows_to_history, capture_exact_root_input_rows,
    capture_sound_crown_root_rows_at_node, mint_certified_affine_enclosure,
    CertifiedAffineEnclosure, CrownPassStamp, SoundCrownRootAffineRows, ValidatedAffineEnclosure,
};
#[cfg(test)]
use crate::complete_clip::{
    capture_host_sound_crown_root_rows, check_root_affine_dominance_and_seal,
    UntrustedCrownAffineRows,
};
#[cfg(test)]
use crate::LinearBounds;
use crate::{GraphNetwork, Layer, NETWORK_INPUT};

use super::super::super::super::BetaCrownVerifier;
use super::{
    build_call_skeleton, prep_resnet_domain_ext, prep_resnet_domain_with, ResnetDomainPrep,
};

const ROOT_INTERM_MAX_HOST_SEED_BYTES: usize = 128 * 1024 * 1024;

const SELECTIVE_CLIP_MAX_HOST_BYTES: usize = 64 * 1024 * 1024;
const COMPLETE_CLIP_ROOT_AFFINE_MAX_BYTES: usize = 64 * 1024 * 1024;
const COMPLETE_CLIP_ROOT_AFFINE_MAX_SELECTIONS: usize = 64;
const COMPLETE_CLIP_DECISION_MAX_BYTES: usize = 8 * 1024 * 1024;
const COMPLETE_CLIP_DECISION_MAX_HISTORIES: usize = 4096;
const COMPLETE_CLIP_GPU_MEAN_LA_MAX_DOMAINS: usize = 256;
// Reconstructing missing full-spec lA is an extra parent-only CROWN pass. On
// CIFAR-100/Metal, one parent costs ~0.72s and materially improves the first
// frontier, while 13 parents cost ~8.3s without increasing the established
// clipping lift. Larger waves therefore keep exact inherited host caches when
// available but fail closed to the local selector when their caches are
// missing. This is an admission limit, never a proof or clipping authority.
const COMPLETE_CLIP_GPU_MEAN_LA_MAX_RECONSTRUCT_DOMAINS: usize = 4;
const COMPLETE_CLIP_GPU_MEAN_LA_MAX_SPECS: usize = 512;
const COMPLETE_CLIP_GPU_MEAN_LA_MAX_SEED_VALUES: usize = 1 << 24;
const COMPLETE_CLIP_GPU_MEAN_LA_MAX_GATHER_VALUES: usize = 4 * 1024 * 1024;
const COMPLETE_CLIP_GPU_MEAN_LA_MAX_WIDE_ROWS: usize = 4096;
const COMPLETE_CLIP_GPU_MEAN_LA_MAX_WIDE_COEFFICIENTS: usize = 16 * 1024 * 1024;
const COMPLETE_CLIP_GPU_MEAN_LA_MAX_HOST_MEAN_VALUES: usize = 4 * 1024 * 1024;
/// Minimum authority budget required before starting an optional Complete Clip
/// CROWN pass. The pass polls the exact deadline, while this admission guard
/// avoids launching an expensive device operation in the final few seconds.
const COMPLETE_CLIP_OPTIONAL_MIN_START_HEADROOM: std::time::Duration =
    std::time::Duration::from_secs(3);
const SELECTIVE_CLIP_MAX_WORK: usize = 500_000_000;
const SELECTIVE_DEADLINE_POLL_STRIDE: usize = 1024;
const MARGIN_WEIGHT_SITE: &str = "interm_refine_margin_weights";
const COMPLETE_CLIP_ALL_HISTORY_ROUNDS: usize = 2;
// Up to four in-place heapsorts can occur across winner selection and the
// deep-row cap; each sort is bounded by 4*n*ceil(log2 n) comparisons.
const SELECTIVE_SORT_OP_WEIGHT: usize = 16;

#[inline]
fn complete_clip_optional_start_budget_available(now: Instant, deadline: Option<Instant>) -> bool {
    deadline.is_none_or(|value| {
        now.checked_add(COMPLETE_CLIP_OPTIONAL_MIN_START_HEADROOM)
            .is_some_and(|start_limit| start_limit < value)
    })
}

/// Active BaB deadline and advisory-suppression scopes for optional Complete
/// Clipping work.
///
/// A verifier can be borrowed concurrently, so scopes form a multiset rather
/// than overwriting one global slot. Readers take the earliest active boundary;
/// a concurrent stricter call or advisory suppression can only make clipping
/// decline optional work.
#[derive(Default)]
pub(crate) struct CompleteClipDeadlineOverrides {
    active: Mutex<Vec<(Arc<()>, Instant)>>,
    suppressed: Mutex<Vec<Arc<()>>>,
    /// #layer-deadline-suppression: scopes in which the constrained forward may
    /// hand its LAYER kernels `None` while still polling the deadline per node.
    ///
    /// A multiset of tokens rather than a bool because the verifier is borrowed
    /// concurrently by a 64-wide rayon fan-out; nested/overlapping scopes must
    /// not clear each other's suppression.
    layer_deadline_suppressed: Mutex<Vec<Arc<()>>>,
}

#[must_use = "the Complete Clipping deadline scope ends when this guard is dropped"]
pub(crate) struct CompleteClipDeadlineGuard<'a> {
    state: &'a CompleteClipDeadlineOverrides,
    token: Option<Arc<()>>,
}

#[must_use = "Complete Clipping is suppressed only while this guard is alive"]
pub(crate) struct CompleteClipSuppressionGuard<'a> {
    state: &'a CompleteClipDeadlineOverrides,
    token: Option<Arc<()>>,
}

#[must_use = "the layer-deadline suppression ends when this guard is dropped"]
pub(crate) struct LayerDeadlineSuppressionGuard<'a> {
    state: &'a CompleteClipDeadlineOverrides,
    token: Option<Arc<()>>,
}

impl CompleteClipDeadlineOverrides {
    pub(crate) fn scoped(&self, deadline: Option<Instant>) -> CompleteClipDeadlineGuard<'_> {
        let token = deadline.map(|deadline| {
            let token = Arc::new(());
            self.active
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .push((Arc::clone(&token), deadline));
            token
        });
        CompleteClipDeadlineGuard { state: self, token }
    }

    pub(crate) fn effective(&self, configured: Option<Instant>) -> Option<Instant> {
        self.active
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .iter()
            .map(|(_, deadline)| *deadline)
            .fold(configured, |current, deadline| {
                Some(current.map_or(deadline, |value| value.min(deadline)))
            })
    }

    /// Suppress optional Complete Clipping inside an advisory propagation.
    ///
    /// Branch-scoring children are discarded after their scalar score is read,
    /// so clipping them cannot contribute a reusable proof enclosure. A
    /// verifier may be borrowed by overlapping calls; token membership makes
    /// nested and out-of-order scope drops deterministic and underflow-free.
    pub(crate) fn suppress_complete_clip_scoped(&self) -> CompleteClipSuppressionGuard<'_> {
        let token = Arc::new(());
        self.suppressed
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .push(Arc::clone(&token));
        CompleteClipSuppressionGuard {
            state: self,
            token: Some(token),
        }
    }

    /// Suppress the LAYER deadline for the duration of the guard.
    ///
    /// WHY THIS EXISTS. `compute_constrained_forward_bounds_view_inner` passes one
    /// `deadline` to two different authorities: the per-node loop poll, and the
    /// layer kernels. A finite LAYER deadline routes every Conv2d to
    /// `propagate_ibp_sound_certified_f64_with_deadline` — a five-deep scalar
    /// `oc/oh/ow/ic/ki/kj` loop with `IxDyn` dynamic-stride indexing and an
    /// `Instant::now()` poll every 4096 taps — instead of im2col + faer GEMM.
    /// This repository has already measured that trade at ~91x on this model
    /// family (`graph_alpha/bounds/ibp.rs`).
    ///
    /// SOUNDNESS. `None` does NOT drop the certificate: the sound IBP then takes
    /// the f32 GEMM plus the `|W|·max(|l|,|u|)` abssum pass and the
    /// `γ_{K+2}^{f32}·S_safe + 2u·|y|` outward widening — the A-PRIORI γ
    /// certificate rather than the measured-f64 one. It is therefore SOUND and
    /// LOOSER-OR-EQUAL, never tighter. Looser is the safe direction here in both
    /// roles a simulated bound can play: as a ranking score it can only misrank,
    /// and as a certified LOWER bound it remains valid.
    ///
    /// The loop authority is untouched, so the pass still refuses cooperatively;
    /// overrun is bounded by one layer rather than one tap.
    pub(crate) fn suppress_layer_deadline_scoped(&self) -> LayerDeadlineSuppressionGuard<'_> {
        let token = Arc::new(());
        self.layer_deadline_suppressed
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .push(Arc::clone(&token));
        LayerDeadlineSuppressionGuard {
            state: self,
            token: Some(token),
        }
    }

    pub(crate) fn layer_deadline_suppressed(&self) -> bool {
        !self
            .layer_deadline_suppressed
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .is_empty()
    }

    pub(crate) fn complete_clip_suppressed(&self) -> bool {
        !self
            .suppressed
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .is_empty()
    }
}

impl Drop for CompleteClipDeadlineGuard<'_> {
    fn drop(&mut self) {
        let Some(token) = self.token.take() else {
            return;
        };
        let mut active = self
            .state
            .active
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some(position) = active
            .iter()
            .position(|(entry_token, _)| Arc::ptr_eq(entry_token, &token))
        {
            active.swap_remove(position);
        }
    }
}

impl Drop for LayerDeadlineSuppressionGuard<'_> {
    fn drop(&mut self) {
        let Some(token) = self.token.take() else {
            return;
        };
        let mut suppressed = self
            .state
            .layer_deadline_suppressed
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some(position) = suppressed
            .iter()
            .position(|entry_token| Arc::ptr_eq(entry_token, &token))
        {
            suppressed.swap_remove(position);
        }
    }
}

impl Drop for CompleteClipSuppressionGuard<'_> {
    fn drop(&mut self) {
        let Some(token) = self.token.take() else {
            return;
        };
        let mut suppressed = self
            .state
            .suppressed
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some(position) = suppressed
            .iter()
            .position(|entry_token| Arc::ptr_eq(entry_token, &token))
        {
            suppressed.swap_remove(position);
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
struct CompleteClipRootBoundsKey {
    graph_scope: CutFoldScope,
    shape: Box<[usize]>,
    lower_bits: Box<[u32]>,
    upper_bits: Box<[u32]>,
}

struct CompleteClipRootBoundsEntry {
    key: CompleteClipRootBoundsKey,
    bounds: Arc<HashMap<String, Arc<BoundedTensor>>>,
    affine_rows: HashMap<CompleteClipRootAffineKey, CompleteClipCachedAffineRows>,
    affine_rows_bytes: usize,
    affine_access_tick: u64,
    decisions: HashMap<Box<[u8]>, CompleteClipCachedDecisions>,
    decision_bytes: usize,
    decision_access_tick: u64,
    bab_iteration: Option<NonZeroUsize>,
}

struct CompleteClipCachedAffineRows {
    rows: Arc<SoundCrownRootAffineRows>,
    resident_bytes: usize,
    last_access: u64,
}

/// Compact DomainClipper objective identities for one prospective child.
///
/// The dense per-layer lA matrices live only until the kFSB winner-capture
/// returns.  Persisting neuron indices mirrors αβ-CROWN's one-generation
/// `SubDomainClipDecisionsDB` without retaining an additional CROWN graph.
pub(crate) type CompleteClipDecisionIndices = HashMap<String, Arc<[usize]>>;

struct CompleteClipCachedDecisions {
    indices: Arc<CompleteClipDecisionIndices>,
    resident_bytes: usize,
    last_access: u64,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct CompleteClipRootAffineKey {
    node_name: String,
    selected_neurons: Box<[usize]>,
    row_of_len: usize,
}

/// One exact root-box map per verifier. DomainClipper's affine rows are global
/// input-relative enclosures, so they must be computed from an unconstrained
/// root cache—not from one child's premise-clamped cache. The one-entry design
/// bounds memory and naturally replaces the map when a verifier is reused for
/// another graph/input pair.
#[derive(Default)]
pub(crate) struct CompleteClipRootBoundsCache {
    inner: Mutex<Option<CompleteClipRootBoundsEntry>>,
}

impl CompleteClipRootBoundsCache {
    fn key(graph: &GraphNetwork, input: &BoundedTensor) -> Option<CompleteClipRootBoundsKey> {
        Self::key_with_deadline(graph, input, None)
    }

    /// Pointer-and-content identity image for speculative-side-effect tests.
    ///
    /// This deliberately includes cache container identities, payload byte
    /// counts, access clocks, and every resident affine/decision Arc identity.
    /// A rejected advisory evaluation must leave the complete image unchanged.
    #[cfg(test)]
    pub(crate) fn test_identity_image(&self) -> Vec<u64> {
        use std::hash::{Hash, Hasher};

        let guard = self.inner.lock().unwrap_or_else(|error| error.into_inner());
        let Some(entry) = guard.as_ref() else {
            return Vec::new();
        };
        let mut image = vec![
            Arc::as_ptr(&entry.bounds) as usize as u64,
            entry.affine_rows.len() as u64,
            entry.affine_rows_bytes as u64,
            entry.affine_access_tick,
            entry.decisions.len() as u64,
            entry.decision_bytes as u64,
            entry.decision_access_tick,
            entry.bab_iteration.map_or(0, NonZeroUsize::get) as u64,
        ];
        let mut affine = Vec::with_capacity(entry.affine_rows.len());
        for (key, cached) in &entry.affine_rows {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            key.hash(&mut hasher);
            affine.push((
                hasher.finish(),
                Arc::as_ptr(&cached.rows) as usize as u64,
                cached.resident_bytes as u64,
                cached.last_access,
            ));
        }
        affine.sort_unstable();
        for row in affine {
            image.extend([row.0, row.1, row.2, row.3]);
        }
        let mut decisions = Vec::with_capacity(entry.decisions.len());
        for (history, cached) in &entry.decisions {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            history.hash(&mut hasher);
            decisions.push((
                hasher.finish(),
                Arc::as_ptr(&cached.indices) as usize as u64,
                cached.resident_bytes as u64,
                cached.last_access,
            ));
        }
        decisions.sort_unstable();
        for row in decisions {
            image.extend([row.0, row.1, row.2, row.3]);
        }
        image
    }

    fn key_with_deadline(
        graph: &GraphNetwork,
        input: &BoundedTensor,
        deadline: Option<Instant>,
    ) -> Option<CompleteClipRootBoundsKey> {
        // The ReLU-split VNN-COMP lane uses a rectangular input box. A future
        // root cache may encode the optional L2 annotation exactly; until then,
        // refusing it prevents a box-only identity from aliasing a stronger
        // non-box domain.
        if input.l2_constraint().is_some() || deadline.is_some_and(|d| Instant::now() >= d) {
            return None;
        }
        let mut lower_bits = Vec::new();
        let mut upper_bits = Vec::new();
        lower_bits.try_reserve_exact(input.len()).ok()?;
        upper_bits.try_reserve_exact(input.len()).ok()?;
        for (index, value) in input.lower().iter().enumerate() {
            if index.is_multiple_of(SELECTIVE_DEADLINE_POLL_STRIDE)
                && deadline.is_some_and(|d| Instant::now() >= d)
            {
                return None;
            }
            lower_bits.push(value.to_bits());
        }
        for (index, value) in input.upper().iter().enumerate() {
            if index.is_multiple_of(SELECTIVE_DEADLINE_POLL_STRIDE)
                && deadline.is_some_and(|d| Instant::now() >= d)
            {
                return None;
            }
            upper_bits.push(value.to_bits());
        }
        if deadline.is_some_and(|d| Instant::now() >= d) {
            return None;
        }
        Some(CompleteClipRootBoundsKey {
            graph_scope: graph.cut_fold_scope(),
            shape: input.shape().into(),
            lower_bits: lower_bits.into_boxed_slice(),
            upper_bits: upper_bits.into_boxed_slice(),
        })
    }

    /// Borrow the finalized root map only when every current child has the
    /// exact same input box. ReLU splitting preserves that box; hybrid/input
    /// splitting does not, and must not replace the one root entry with child
    /// zero's map.
    fn get_finalized_for_batch(
        &self,
        graph: &GraphNetwork,
        inputs: &[BoundedTensor],
        deadline: Option<Instant>,
    ) -> Option<(Arc<HashMap<String, Arc<BoundedTensor>>>, usize)> {
        let first = inputs.first()?;
        let key = Self::key_with_deadline(graph, first, deadline)?;
        for input in &inputs[1..] {
            if Self::key_with_deadline(graph, input, deadline)? != key {
                return None;
            }
        }
        if deadline.is_some_and(|d| Instant::now() >= d) {
            return None;
        }
        let guard = self.inner.lock().unwrap_or_else(|error| error.into_inner());
        let entry = guard.as_ref().filter(|entry| entry.key == key)?;
        let iteration = entry.bab_iteration?.get();
        let batch = (Arc::clone(&entry.bounds), iteration);
        deadline.is_none_or(|d| Instant::now() < d).then_some(batch)
    }

    /// Publish the current outer BaB round for this exact graph/root box.
    ///
    /// One outer wave can invoke several speculative and final CROWN passes
    /// (notably kFSB scoring). Those calls must all observe the same round:
    /// αβ-CROWN advances DomainClipper's all-history warm-up from the BaB
    /// driver's `total_round`, not from the number of bound-propagation calls.
    pub(crate) fn set_bab_iteration(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        iteration: usize,
    ) -> bool {
        let Some(iteration) = NonZeroUsize::new(iteration) else {
            return false;
        };
        let Some(key) = Self::key(graph, input) else {
            return false;
        };
        let mut guard = self.inner.lock().unwrap_or_else(|error| error.into_inner());
        let Some(entry) = guard.as_mut().filter(|entry| entry.key == key) else {
            return false;
        };
        entry.bab_iteration = Some(iteration);
        true
    }

    /// Seed the cache from the finalized root BaB map (after root intermediate
    /// tightening and alpha-state reconstruction). Arc values make this a
    /// shallow snapshot. This is the preferred quality path; `get_or_build`
    /// retains a sound fresh-root fallback for callers without a root seam.
    pub(crate) fn store_finalized(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        bounds: &HashMap<String, Arc<BoundedTensor>>,
    ) -> bool {
        let Some(key) = Self::key(graph, input) else {
            return false;
        };
        if bounds.is_empty() {
            return false;
        }
        let mut guard = self.inner.lock().unwrap_or_else(|error| error.into_inner());
        *guard = Some(CompleteClipRootBoundsEntry {
            key,
            bounds: Arc::new(bounds.clone()),
            affine_rows: HashMap::new(),
            affine_rows_bytes: 0,
            affine_access_tick: 0,
            decisions: HashMap::new(),
            decision_bytes: 0,
            decision_access_tick: 0,
            bab_iteration: None,
        });
        true
    }

    fn get_affine_rows(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        node_name: &str,
        selected_neurons: &[usize],
        row_of_len: usize,
    ) -> Option<Arc<SoundCrownRootAffineRows>> {
        let key = Self::key(graph, input)?;
        let affine_key = CompleteClipRootAffineKey {
            node_name: node_name.to_string(),
            selected_neurons: selected_neurons.into(),
            row_of_len,
        };
        let mut guard = self.inner.lock().unwrap_or_else(|error| error.into_inner());
        let entry = guard.as_mut().filter(|entry| entry.key == key)?;
        entry.affine_access_tick = entry.affine_access_tick.saturating_add(1);
        let access_tick = entry.affine_access_tick;
        entry.affine_rows.get_mut(&affine_key).map(|cached| {
            cached.last_access = access_tick;
            Arc::clone(&cached.rows)
        })
    }

    fn store_affine_rows(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        node_name: &str,
        selected_neurons: &[usize],
        row_of_len: usize,
        rows: Arc<SoundCrownRootAffineRows>,
    ) {
        self.store_affine_rows_with_limits(
            graph,
            input,
            node_name,
            selected_neurons,
            row_of_len,
            rows,
            COMPLETE_CLIP_ROOT_AFFINE_MAX_BYTES,
            COMPLETE_CLIP_ROOT_AFFINE_MAX_SELECTIONS,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn store_affine_rows_with_limits(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        node_name: &str,
        selected_neurons: &[usize],
        row_of_len: usize,
        rows: Arc<SoundCrownRootAffineRows>,
        max_bytes: usize,
        max_selections: usize,
    ) {
        let Some(resident_bytes) = rows.resident_payload_bytes() else {
            return;
        };
        if resident_bytes > max_bytes || max_selections == 0 {
            return;
        }
        let Some(key) = Self::key(graph, input) else {
            return;
        };
        let affine_key = CompleteClipRootAffineKey {
            node_name: node_name.to_string(),
            selected_neurons: selected_neurons.into(),
            row_of_len,
        };
        let mut guard = self.inner.lock().unwrap_or_else(|error| error.into_inner());
        let Some(entry) = guard.as_mut().filter(|entry| entry.key == key) else {
            return;
        };

        if let Some(previous) = entry.affine_rows.remove(&affine_key) {
            entry.affine_rows_bytes = entry
                .affine_rows_bytes
                .saturating_sub(previous.resident_bytes);
        }
        while !entry.affine_rows.is_empty()
            && (entry.affine_rows.len() >= max_selections
                || entry
                    .affine_rows_bytes
                    .checked_add(resident_bytes)
                    .is_none_or(|bytes| bytes > max_bytes))
        {
            let Some(evict_key) = entry
                .affine_rows
                .iter()
                .min_by_key(|(_, cached)| cached.last_access)
                .map(|(key, _)| key.clone())
            else {
                break;
            };
            if let Some(evicted) = entry.affine_rows.remove(&evict_key) {
                entry.affine_rows_bytes = entry
                    .affine_rows_bytes
                    .saturating_sub(evicted.resident_bytes);
            }
        }
        let Some(new_bytes) = entry.affine_rows_bytes.checked_add(resident_bytes) else {
            return;
        };
        if new_bytes > max_bytes || entry.affine_rows.len() >= max_selections {
            return;
        }
        entry.affine_access_tick = entry.affine_access_tick.saturating_add(1);
        entry.affine_rows.insert(
            affine_key,
            CompleteClipCachedAffineRows {
                rows,
                resident_bytes,
                last_access: entry.affine_access_tick,
            },
        );
        entry.affine_rows_bytes = new_bytes;
    }

    fn store_decisions(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        history: &GraphSplitHistory,
        indices: CompleteClipDecisionIndices,
    ) -> bool {
        if indices.is_empty() {
            return false;
        }
        let Some(history_id) = history.exact_provenance_identity() else {
            return false;
        };
        let mut resident_bytes = history_id.len();
        for (layer, neurons) in &indices {
            if neurons.is_empty() || neurons.windows(2).any(|pair| pair[0] >= pair[1]) {
                return false;
            }
            resident_bytes = match resident_bytes.checked_add(layer.len()).and_then(|bytes| {
                neurons
                    .len()
                    .checked_mul(size_of::<usize>())
                    .and_then(|indices_bytes| bytes.checked_add(indices_bytes))
            }) {
                Some(bytes) => bytes,
                None => return false,
            };
        }
        if resident_bytes > COMPLETE_CLIP_DECISION_MAX_BYTES {
            return false;
        }
        let Some(key) = Self::key(graph, input) else {
            return false;
        };
        let history_id: Box<[u8]> = history_id.into_boxed_slice();
        let mut guard = self.inner.lock().unwrap_or_else(|error| error.into_inner());
        let Some(entry) = guard.as_mut().filter(|entry| entry.key == key) else {
            return false;
        };

        if let Some(previous) = entry.decisions.remove(history_id.as_ref()) {
            entry.decision_bytes = entry.decision_bytes.saturating_sub(previous.resident_bytes);
        }
        while !entry.decisions.is_empty()
            && (entry.decisions.len() >= COMPLETE_CLIP_DECISION_MAX_HISTORIES
                || entry
                    .decision_bytes
                    .checked_add(resident_bytes)
                    .is_none_or(|bytes| bytes > COMPLETE_CLIP_DECISION_MAX_BYTES))
        {
            let Some(evict_key) = entry
                .decisions
                .iter()
                .min_by_key(|(_, cached)| cached.last_access)
                .map(|(key, _)| key.clone())
            else {
                break;
            };
            if let Some(evicted) = entry.decisions.remove(evict_key.as_ref()) {
                entry.decision_bytes = entry.decision_bytes.saturating_sub(evicted.resident_bytes);
            }
        }
        let Some(new_bytes) = entry.decision_bytes.checked_add(resident_bytes) else {
            return false;
        };
        if new_bytes > COMPLETE_CLIP_DECISION_MAX_BYTES
            || entry.decisions.len() >= COMPLETE_CLIP_DECISION_MAX_HISTORIES
        {
            return false;
        }
        entry.decision_access_tick = entry.decision_access_tick.saturating_add(1);
        entry.decisions.insert(
            history_id,
            CompleteClipCachedDecisions {
                indices: Arc::new(indices),
                resident_bytes,
                last_access: entry.decision_access_tick,
            },
        );
        entry.decision_bytes = new_bytes;
        true
    }

    /// Consume the prospective decision exactly once when the committed child
    /// reaches its verdict-path clipping pass. A miss keeps the historical
    /// local selector; it never changes proof soundness.
    fn take_decisions(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        history: &GraphSplitHistory,
    ) -> Option<Arc<CompleteClipDecisionIndices>> {
        let key = Self::key(graph, input)?;
        let history_id = history.exact_provenance_identity()?;
        let mut guard = self.inner.lock().unwrap_or_else(|error| error.into_inner());
        let entry = guard.as_mut().filter(|entry| entry.key == key)?;
        let cached = entry.decisions.remove(history_id.as_slice())?;
        entry.decision_bytes = entry.decision_bytes.saturating_sub(cached.resident_bytes);
        Some(cached.indices)
    }
}

#[derive(Clone, Copy)]
struct MarginWeightLimits {
    max_host_bytes: usize,
    max_work: usize,
}

const MARGIN_WEIGHT_LIMITS: MarginWeightLimits = MarginWeightLimits {
    max_host_bytes: SELECTIVE_CLIP_MAX_HOST_BYTES,
    max_work: SELECTIVE_CLIP_MAX_WORK,
};

/// Checked peak-owned allocation and arithmetic envelope for
/// `spec_matrix.dot(tail_weight)` plus the reduction into one weight per seed
/// neuron. Borrowed graph/spec storage is excluded; both the dense dot result
/// and the returned weight vector (including its `Vec` -> `Arc` transition) are
/// included in the peak.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MarginWeightPlan {
    dot_elements: usize,
    weight_elements: usize,
    peak_owned_bytes: usize,
    arithmetic_ops: usize,
}

impl MarginWeightPlan {
    fn checked(
        spec_rows: usize,
        output_dim: usize,
        seed_dim: usize,
        limits: MarginWeightLimits,
    ) -> ny_core::Result<Self> {
        let dot_elements = spec_rows.checked_mul(seed_dim).ok_or_else(|| {
            NyError::InvalidSpec("margin-weight dot shape product overflow".into())
        })?;
        let dot_bytes = dot_elements
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| NyError::InvalidSpec("margin-weight dot byte overflow".into()))?;
        let weight_bytes = seed_dim
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| NyError::InvalidSpec("margin-weight vector byte overflow".into()))?;
        let dot_plus_weights = dot_bytes
            .checked_add(weight_bytes)
            .ok_or_else(|| NyError::InvalidSpec("margin-weight peak byte overflow".into()))?;
        let arc_transition = weight_bytes.checked_mul(2).ok_or_else(|| {
            NyError::InvalidSpec("margin-weight Arc transition byte overflow".into())
        })?;
        let peak_owned_bytes = dot_plus_weights.max(arc_transition);
        if peak_owned_bytes > limits.max_host_bytes {
            return Err(NyError::CpuMemoryExceeded {
                required_bytes: peak_owned_bytes,
                budget_bytes: limits.max_host_bytes,
                site: MARGIN_WEIGHT_SITE,
            });
        }

        // Charge a multiply and add for every GEMM term, plus comparison,
        // negation, and accumulation for every reduced dot cell.
        let multiply_adds = dot_elements.checked_mul(output_dim).ok_or_else(|| {
            NyError::InvalidSpec("margin-weight multiply-add count overflow".into())
        })?;
        let gemm_ops = multiply_adds
            .checked_mul(2)
            .ok_or_else(|| NyError::InvalidSpec("margin-weight GEMM work overflow".into()))?;
        let reduction_ops = dot_elements
            .checked_mul(3)
            .ok_or_else(|| NyError::InvalidSpec("margin-weight reduction work overflow".into()))?;
        let arithmetic_ops = gemm_ops
            .checked_add(reduction_ops)
            .ok_or_else(|| NyError::InvalidSpec("margin-weight total work overflow".into()))?;
        if arithmetic_ops > limits.max_work {
            return Err(NyError::InvalidSpec(format!(
                "margin-weight arithmetic budget exceeded: {arithmetic_ops} > {}",
                limits.max_work
            )));
        }

        Ok(Self {
            dot_elements,
            weight_elements: seed_dim,
            peak_owned_bytes,
            arithmetic_ops,
        })
    }
}
// The 7-arg legacy face is the tests' reference oracle (`use super::*` in the
// test modules below); production lanes call the `_with`/`_ext` faces directly.
#[cfg(test)]
use super::prep_resnet_domain;

/// Env gate for per-subdomain intermediate-bound refinement (dark, default off).
/// The proposed `NY_CLIP_INTERM` umbrella is authority-quarantined and cannot
/// imply this lane. Only `NY_INTERM_REFINE=1` enables production refinement.
pub(in crate::beta_crown::engine::graph) fn interm_refine_enabled() -> bool {
    matches!(std::env::var("NY_INTERM_REFINE").ok().as_deref(), Some("1"))
        || clip_interm_umbrella_enabled()
}

/// Env gate for infeasibility pruning (dark, default OFF — see module docs).
fn interm_refine_prune_enabled() -> bool {
    matches!(
        std::env::var("NY_INTERM_REFINE_PRUNE").ok().as_deref(),
        Some("1")
    )
}

/// How many ReLU layers (from the output inward) get per-subdomain refinement
/// (`NY_INTERM_REFINE_LAYERS`, default 1 = the last ReLU only).
fn interm_refine_layers() -> usize {
    std::env::var("NY_INTERM_REFINE_LAYERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(1)
        .max(1)
}

/// Extra defense-in-depth margin the premise contradiction must exceed before
/// a domain is pruned (`NY_INTERM_REFINE_PRUNE_TOL`, default `1e-4`). The
/// refined bounds already carry certified outward rounding; this only makes
/// the prune MORE conservative.
fn interm_refine_prune_tol() -> f32 {
    std::env::var("NY_INTERM_REFINE_PRUNE_TOL")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|t| t.is_finite() && *t >= 0.0)
        .unwrap_or(1e-4)
}

/// Row cap for DEEP (non-last-ReLU) seed layers
/// (`NY_INTERM_REFINE_DEEP_MAX_ROWS`, default 256): conv-wide seeds would
/// otherwise carry thousands of identity rows. Premise rows are always kept;
/// the remainder is top-K by inherited width. Cost-only, never soundness.
fn interm_refine_deep_max_rows() -> usize {
    std::env::var("NY_INTERM_REFINE_DEEP_MAX_ROWS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(256)
}

/// Winner-style per-layer objective cap
/// (`NY_INTERM_REFINE_SELECTIVE_TOPK`, default 0 = OFF/unlimited).
///
/// αβ-CROWN's VNN-COMP 2025 TinyImageNet configuration clips only the top 20
/// intermediate objectives per layer.  NY historically seeded every unstable
/// neuron at each selected layer, making the identity backward and per-domain clip scale
/// with the full layer width.  A positive value keeps that many highest-impact
/// unstable objectives plus every split-premise source row.  Omitted rows retain
/// their inherited sound bounds, so this is cost/selection only.
fn interm_refine_selective_topk() -> usize {
    std::env::var("NY_INTERM_REFINE_SELECTIVE_TOPK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0)
}

/// Probe (`NY_INTERM_REFINE_PROBE=1` or the lane-wide `NY_BETA_GPU_PROBE=1`):
/// per-batch refinement stats on stderr.
fn probe_enabled() -> bool {
    std::env::var("NY_INTERM_REFINE_PROBE").ok().as_deref() == Some("1")
        || crate::beta_gpu_probe_armed()
}

/// Cap on the seed-layer width (the refinement backward carries `pre_dim`
/// identity rows — cost scales linearly). `Relu_57` is 512-d; the default keeps
/// room for tinyimagenet-class nets while refusing conv-wide layers.
/// Override with `NY_INTERM_REFINE_MAX_DIM`.
fn interm_refine_max_dim() -> usize {
    std::env::var("NY_INTERM_REFINE_MAX_DIM")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(2048)
}

/// Depth gate (`NY_INTERM_REFINE_MIN_DEPTH=d`, default 0 = refine every
/// domain, byte-identical): refine only domains with at least `d` split
/// premises (β entries — ReLU-split BaB adds exactly one per split, so the
/// count IS the domain depth). Shallow domains keep their inherited caches.
/// Cost-only lever (#fit-100s): row selection of WHICH domains get a refined
/// enclosure — excluded domains keep the inherited sound pair, exactly like
/// an unselected neuron row.
fn interm_refine_min_depth() -> usize {
    std::env::var("NY_INTERM_REFINE_MIN_DEPTH")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0)
}

/// Explicit seed list (`NY_INTERM_REFINE_SEEDS=Relu_13,Relu_19,Relu_31,last`,
/// dark, default unset = the `NY_INTERM_REFINE_LAYERS` walk, byte-identical):
/// comma-separated ReLU node names (exact graph names) plus the token `last`
/// for the last-ReLU chain. Purpose (#midref): under the LA brancher the split
/// premises accumulate on MID-network ReLUs (`Relu_13/19/31` on resnet_medium)
/// whose downstream effect is otherwise priced only by β — naming them refines
/// their pre-activation entries per subdomain, CASCADED in exec order
/// (earliest first), each pass consuming the previous passes' refined caches
/// (β folds of below-seed premises + clamped/refined below-seed relaxations),
/// then the margin backward consumes everything. Named seeds are exempt from
/// `NY_INTERM_REFINE_MAX_DIM` (naming a conv-wide layer is the deliberate ask;
/// the deep row cap + the hard `n_rows·pre_dim` guard still bound cost).
/// Unknown/non-ReLU/fully-stable names are skipped (probe-logged).
fn interm_refine_seed_names() -> Option<Vec<String>> {
    let raw = std::env::var("NY_INTERM_REFINE_SEEDS").ok()?;
    let names: Vec<String> = raw
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    (!names.is_empty()).then_some(names)
}

/// α′-FOR-REFINEMENT gate (`NY_INTERM_REFINE_ALPHA=1`, dark, default OFF =
/// byte-identical — #alpha-prime). auto_LiRPA "finding 4": each intermediate
/// bound is its own optimization with its OWN α, distinct from the margin
/// objective's α. The refinement backward today folds the WARMUP/MARGIN α
/// (optimized for the margin row, not for tightening each seed-layer
/// pre-activation). With the gate on, the last-ReLU pass runs a small Adam
/// ascent of a DEDICATED α′ against the refinement objective
/// `Σ_rows l′_row` — per-row gradients from the oracle-validated TRUE
/// chain rule (`max(ν,0)·ĥ(x*)`, `wide_alpha_true::true_alpha_grads_for_row`)
/// replayed on the truncated stack, summed over the identity rows — and
/// element-wise best-merges every iterate's per-row bounds (each iterate is a
/// sound enclosure of the SAME subdomain for its α′ ∈ [0,1], so `max l / min u`
/// across iterates is sound). The best-objective α′ is stored once per process
/// and REUSED by later batches (applied to each domain's own unstable neurons
/// before the single refinement backward — any α ∈ [0,1] is a valid lower
/// ReLU relaxation slope, so reuse is sound for every subdomain).
fn interm_refine_alpha_enabled() -> bool {
    matches!(
        std::env::var("NY_INTERM_REFINE_ALPHA").ok().as_deref(),
        Some("1")
    )
}

/// JOINT margin-directed intermediate-α gate (`NY_JOINT_INTERM_ALPHA=1`, dark,
/// default OFF = byte-identical — #joint-interm-alpha). The SOTA blueprint's #1
/// αβ-CROWN advantage: optimize the intermediate pre-activation bounds against
/// the FINAL margin, not each against its own objective. The base α′ lane
/// ([`interm_refine_alpha_enabled`]) ascends the refinement backward's α against
/// the UNIFORM refinement objective `Σ_rows l′_row` — every seed neuron weighted
/// equally. With this gate on, each refinement row is instead weighted by the
/// MARGIN's sensitivity to that seed neuron's refined LOWER bound
/// `m_j = Σ_specs max(−(spec·W_tail)[j], 0)` (only the upper-branch neurons the
/// margin's chord relaxation actually reads carry weight, ∝ the tail-coefficient
/// magnitude), so the α′ ascent tightens the refined bounds the margin uses.
/// Turning this on implies the α′ lane (auto-enabled in `from_env`). Sound: any
/// α ∈ [0,1] is a valid lower ReLU slope; the weights only STEER the ascent, and
/// only element-wise-tightest sound iterates are kept (exactly as the base lane).
fn joint_interm_alpha_enabled() -> bool {
    matches!(
        std::env::var("NY_JOINT_INTERM_ALPHA").ok().as_deref(),
        Some("1")
    )
}

/// PER-TARGET (per-intermediate-neuron) α′ gate (`NY_AB_PARITY_INTERM=1`, dark,
/// default OFF = byte-identical — #ab-parity-interm). The auto_LiRPA per-target
/// decoupling (STUDY-1): auto_LiRPA keeps a SEPARATE optimizable lower-slope per
/// (target-spec × source-ReLU-neuron) pair, so the sum-of-bounds objective
/// DECOUPLES and every target reaches its OWN slope optimum. The base α′ lane
/// ([`interm_refine_alpha_enabled`]) instead optimizes ONE shared slope set
/// against a SCALARIZED objective (`Σ_rows w_r·l′_r`), so a single α must serve
/// every seed-row target at once. With this gate on, the last-ReLU refinement
/// gives EACH target row (each seed-layer neuron whose pre-activation box we are
/// tightening) its OWN α, optimized against ITS OWN bound `l′_ri` via that row's
/// per-target gradient, and the sound per-row bound is computed with that row's
/// slopes (element-wise best-lo kept). Implies the α′ lane (auto-enabled in
/// `from_env`) and, under the gate, the once-per-process shared-α reuse is
/// DROPPED — a fresh per-target ascent runs each batch. Sound: any α ∈ [0,1] is
/// a valid lower ReLU slope; per-target α only steers which slope each target
/// picks, and only element-wise-tightest sound GPU-fold iterates are merged.
fn ab_parity_interm_enabled() -> bool {
    matches!(
        std::env::var("NY_AB_PARITY_INTERM").ok().as_deref(),
        Some("1")
    )
}

/// FC-head BOX-WIDTH PROBE (`NY_AB_INTERM_PROBE=1`, dark, stderr-only,
/// read-only — #ab-parity-interm). After the last-ReLU refinement pass, log the
/// seed node's (the FC-head `Gemm_56` pre-activation on cifar100 resnet_medium)
/// TOTAL refined box width `Σ_j (u_j − l_j)` over ALL pre-activation neurons —
/// the primary diagnostic for whether per-target α collapses the measured
/// 50×-too-wide head box (CROWN 419.27) toward the 8.35 true-sampled range,
/// independent of the verdict. Printed once at the ROOT batch and once at the
/// first mid-depth (≥ 5 split premises) batch.
fn ab_interm_probe_enabled() -> bool {
    matches!(
        std::env::var("NY_AB_INTERM_PROBE").ok().as_deref(),
        Some("1")
    )
}

/// Ascent iterations for the α′ lane (`NY_INTERM_REFINE_ALPHA_ITERS`,
/// default 3): each iteration is one gradient replay + one extra batched
/// refinement backward.
fn interm_refine_alpha_iters() -> usize {
    std::env::var("NY_INTERM_REFINE_ALPHA_ITERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(3)
        .max(1)
}

/// Adam learning rate for the α′ ascent (`NY_INTERM_REFINE_ALPHA_LR`,
/// default 0.05 — larger than the margin ascent's 0.01: only 2-4 iterations
/// run, and the keep-best merge makes overshoot harmless).
fn interm_refine_alpha_lr() -> f32 {
    std::env::var("NY_INTERM_REFINE_ALPHA_LR")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|t| t.is_finite() && *t > 0.0)
        .unwrap_or(0.05)
}

/// Replay-row cap for the α′ gradient (`NY_INTERM_REFINE_ALPHA_MAX_ROWS`,
/// default 64): the per-row host replays dominate the ascent wall; rows are
/// picked widest-first (most refinement headroom). Cost-only.
fn interm_refine_alpha_max_rows() -> usize {
    std::env::var("NY_INTERM_REFINE_ALPHA_MAX_ROWS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(64)
        .max(1)
}

/// Re-optimize α′ at EVERY batch instead of optimize-once-reuse
/// (`NY_INTERM_REFINE_ALPHA_REOPT=1`, dark — the expensive per-subdomain arm
/// for A/B; never touches the store).
fn interm_refine_alpha_reopt() -> bool {
    matches!(
        std::env::var("NY_INTERM_REFINE_ALPHA_REOPT")
            .ok()
            .as_deref(),
        Some("1")
    )
}

/// The optimized α′ snapshot the root-batch ascent stores and later batches
/// reuse (#alpha-prime). `slopes[r]` is the FULL lower-slope vector of the
/// truncated stack's `r`-th Activation (fold order = `relu_names` order);
/// `stepped[r]` marks the neurons the ascent actually stepped (unstable at
/// optimization time) — reuse writes ONLY stepped neurons that are ALSO
/// unstable in the consuming domain's own extraction.
#[derive(Clone)]
pub(in crate::beta_crown::engine::graph) struct AlphaPrime {
    pub seed_node: String,
    pub pre_dim: usize,
    pub relu_names: Vec<String>,
    pub slopes: Vec<Vec<f32>>,
    pub stepped: Vec<Vec<bool>>,
    /// Did any ascent iterate beat the borrowed-α objective? `false` ⇒ the
    /// entry only suppresses re-ascending (reuse never applies it).
    pub improved: bool,
}

/// Shared handle to the per-process α′ store (`None` until the first
/// successful ascent). Tests construct their own store; production
/// (`from_env`) hands out this global — one verification instance per
/// process, and the key check in [`alpha_prime_matches`] refuses a stale
/// entry from a different net/seed anyway.
pub(in crate::beta_crown::engine::graph) type AlphaPrimeStore = Arc<Mutex<Option<AlphaPrime>>>;

fn global_alpha_prime_store() -> AlphaPrimeStore {
    static STORE: OnceLock<AlphaPrimeStore> = OnceLock::new();
    STORE.get_or_init(|| Arc::new(Mutex::new(None))).clone()
}

/// ADAPTIVE REFINE SCHEDULE gate (`NY_INTERM_REFINE_ADAPTIVE=1`, dark, default
/// OFF = byte-identical — #adaptive-refine). Measured (2026-07-11, prop8945
/// @300s): the refine pass eats ~47.5% of the BaB window at the TAIL producing
/// ZERO (newly_stable=0, width −0.03%/chunk) while being ESSENTIAL early
/// (objective collapse 25→3 on prop54). The schedule keeps refining while it
/// PRODUCES (newly_stable > 0 or infeasible-pruned > 0 per batch); the first
/// time a completed batch at depth ≥ [`interm_refine_adaptive_floor`] yields
/// zero, a per-process LATCH records that depth and every LATER domain at
/// depth ≥ latch keeps its inherited caches (skip = identical to a per-domain
/// refusal — sound; cost-only lever). Shallower domains (new subtrees) still
/// refine.
fn interm_refine_adaptive_enabled() -> bool {
    matches!(
        std::env::var("NY_INTERM_REFINE_ADAPTIVE").ok().as_deref(),
        Some("1")
    )
}

/// Minimum batch depth (min split-premise count over the batch) at which a
/// zero-yield batch may trip the adaptive latch
/// (`NY_INTERM_REFINE_ADAPTIVE_FLOOR`, default 4): protects the measured
/// early-depth value (objective collapse at depths 0-2, newly_stable=48 at
/// depth 1) from a transiently unproductive shallow batch.
fn interm_refine_adaptive_floor() -> usize {
    std::env::var("NY_INTERM_REFINE_ADAPTIVE_FLOOR")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(4)
}

/// The adaptive latch: the smallest depth at which a completed refine batch
/// yielded zero (`usize::MAX` = unlatched). Per-process like the α′ store —
/// one verification instance per process in production; tests construct their
/// own handle.
pub(in crate::beta_crown::engine::graph) type AdaptiveLatch = Arc<std::sync::atomic::AtomicUsize>;

fn global_adaptive_latch() -> AdaptiveLatch {
    static LATCH: OnceLock<AdaptiveLatch> = OnceLock::new();
    LATCH
        .get_or_init(|| Arc::new(std::sync::atomic::AtomicUsize::new(usize::MAX)))
        .clone()
}

/// Should a completed zero-yield batch at `batch_depth` trip the latch?
/// (Pure — unit-tested.) Production is only ever a latch-DOWN: refinement
/// stops for deeper domains, bounds are never touched.
fn adaptive_should_latch(
    newly_stable: usize,
    n_infeasible: usize,
    batch_depth: usize,
    floor: usize,
) -> bool {
    newly_stable == 0 && n_infeasible == 0 && batch_depth >= floor
}

/// PER-CHILD RE-REFINEMENT (`NY_INTERM_REFINE_REDO=d`, dark, default 0 = OFF
/// = byte-identical — #interm-refine-redo). Under the adaptive latch, deep
/// domains inherit their ancestors' refined caches and never re-derive them —
/// even as split premises keep accumulating past the latch depth. Where the
/// accumulating splits are UPSTREAM (stem-directed branching), every refined
/// downstream box is derived against a stale premise set. With `d > 0`, a
/// latched-out domain whose depth is an exact multiple of `d` re-runs the
/// refinement anyway (its full premise set folds — β + clamps), and its
/// children inherit the re-derived caches; every other deep domain keeps the
/// latch skip. Pure modifier on the adaptive skip: with the latch untripped
/// (or the adaptive lane off) every domain already refines on every batch and
/// the knob is inert. Sound either way — refinement only ever intersects
/// tighter, skip ≡ per-domain refusal.
fn interm_refine_redo_every() -> usize {
    std::env::var("NY_INTERM_REFINE_REDO")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0)
}

/// Latched-batch inclusion predicate (pure — unit-tested): a domain refines
/// despite a tripped latch iff it is below the latch depth, or the redo lane
/// re-includes it (depth an exact multiple of `redo_every`).
fn latched_domain_refines(depth: usize, latched: usize, redo_every: usize) -> bool {
    depth < latched || (redo_every > 0 && depth.is_multiple_of(redo_every))
}

/// Row selection (`NY_INTERM_REFINE_ROWS=all|unstable`, default `unstable` —
/// the measured-better arm on prop885 2026-07-11): `unstable` restricts
/// identity rows to the union of inherited-UNSTABLE neurons — a stable
/// neuron's relaxation slope is already exact, so only the `node_abs` error
/// concretization / output-side forward-IBP side-channels could differ, and
/// the A/B measured IDENTICAL per-depth bounds (to 4 decimals through depth 6)
/// at ~3x less GPU per batch (111s vs 191s refine wall at 300s; 468 vs 212
/// domains explored; 5 vs 3 pruned). `all` keeps every neuron's row for A/B.
/// Both arms are sound (row selection only chooses WHICH neurons get a
/// refined enclosure; unselected neurons keep the inherited sound pair).
fn interm_refine_unstable_rows_only() -> bool {
    !matches!(
        std::env::var("NY_INTERM_REFINE_ROWS").ok().as_deref(),
        Some("all")
    )
}

/// #wide-chunk cap on wide rows per batched refinement call
/// (`NY_INTERM_REFINE_WIDE_MAX_N`, default 0 = OFF — one call, byte-identical).
/// See [`IntermRefineOptions::wide_max_n`].
fn interm_refine_wide_max_n() -> usize {
    std::env::var("NY_INTERM_REFINE_WIDE_MAX_N")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0)
}

/// #wide-chunk sizing: domains per batched refinement call so the wide pass
/// stays under `wide_max_n` total rows (`0` = OFF — the whole batch in one
/// call). Always ≥ 1 (a single domain's rows may exceed the cap; the wide
/// lane's own gates and the serial fallback still bound that case).
fn wide_chunk_domains(wide_max_n: usize, n_rows: usize, n_idxs: usize) -> usize {
    if wide_max_n == 0 || n_rows == 0 {
        n_idxs.max(1)
    } else {
        (wide_max_n / n_rows).max(1)
    }
}

/// Snapshot of the env-tunable knobs, taken ONCE per refinement call so unit
/// tests can drive the lane without racing on process-global env vars.
#[derive(Clone)]
pub(in crate::beta_crown::engine::graph) struct IntermRefineOptions {
    /// Earliest absolute boundary authorized for this optional refinement.
    pub deadline: Option<Instant>,
    pub unstable_rows_only: bool,
    pub prune: bool,
    pub layers: usize,
    pub max_dim: usize,
    pub deep_max_rows: usize,
    /// Winner-style cap for each seed layer's objectives (`0` = unlimited). Split
    /// premise rows are additive and never removed; see
    /// [`interm_refine_selective_topk`].
    pub selective_topk: usize,
    pub prune_tol: f32,
    pub probe: bool,
    pub min_depth: usize,
    /// Explicit seed list (`NY_INTERM_REFINE_SEEDS`, see
    /// [`interm_refine_seed_names`]); `None` = the `layers` walk.
    pub seeds: Option<Vec<String>>,
    /// α′-for-refinement ascent iterations (0 = lane OFF, byte-identical —
    /// see [`interm_refine_alpha_enabled`]).
    pub alpha_iters: usize,
    /// Adam lr for the α′ ascent.
    pub alpha_lr: f32,
    /// Replay-row cap for the α′ gradient (widest-first).
    pub alpha_max_rows: usize,
    /// Re-optimize α′ every batch (A/B arm) instead of optimize-once-reuse.
    pub alpha_reopt: bool,
    /// α′ store handle (`None` ⇔ lane OFF).
    pub alpha_store: Option<AlphaPrimeStore>,
    /// Adaptive-schedule latch handle (`None` ⇔ lane OFF — see
    /// [`interm_refine_adaptive_enabled`]).
    pub adaptive_latch: Option<AdaptiveLatch>,
    /// Min batch depth for a zero-yield batch to trip the adaptive latch.
    pub adaptive_floor: usize,
    /// Per-child re-refinement stride under the latch (0 = OFF — see
    /// [`interm_refine_redo_every`]).
    pub redo_every: usize,
    /// #wide-chunk (dark, `NY_INTERM_REFINE_WIDE_MAX_N=k`, default 0 = OFF =
    /// one batched call for the whole domain batch, byte-identical): cap the
    /// WIDE row count (`n_domains_in_call × n_rows`) of each batched
    /// refinement backward by splitting the domain batch into chunks of
    /// `max(1, k / n_rows)` domains. The wide resident lane fails device
    /// validation past ~2048 wide rows on the widest im2col conv dims
    /// (measured: binding cap under default limits; dispatch-dimension cap
    /// under `NY_GPU_BIG_BINDINGS=1`), and every failed batch falls to the
    /// serial per-domain stacker (~2× slower measured). Chunking is cost-only:
    /// each domain's bound is computed by the same wide kernel on its own
    /// domain block (per-domain independence — no cross-domain reduction);
    /// a chunk error falls back to the serial path for that chunk's domains.
    pub wide_max_n: usize,
    /// JOINT margin-directed intermediate-α (`NY_JOINT_INTERM_ALPHA=1`, dark —
    /// see [`joint_interm_alpha_enabled`]): weight each α′ refinement row by the
    /// margin's sensitivity to that seed neuron's refined lower bound instead of
    /// the uniform `Σ l′`. Implies the α′ lane (`from_env` auto-enables it).
    pub joint_margin: bool,
    /// Per-seed-neuron margin weights (len = seed `pre_dim`), computed once at
    /// the batched call site from the tail linear map + spec rows (see
    /// [`compute_margin_weights`]). `None` ⇒ uniform (the base α′ objective);
    /// only consulted when `joint_margin` is set. Sound: advisory ascent steering
    /// only — a wrong weight degrades the ascent, never the returned bound.
    pub margin_weights: Option<Arc<[f32]>>,
    /// #clip-interm-resnet-batched research option: after the batched seed
    /// backward, run the split-constraint clip (αβ `domain_clipper`) on the
    /// already-batched ResidentCoeff. Production environment authority is
    /// quarantined; explicit unit tests may still set this field to exercise the
    /// implementation and its enclosure oracles.
    pub clip_resnet: bool,
    /// Production `clip_interm_domain` must consume the finalized, iteration-
    /// stamped root affine bank.  When true, a missing bank is a fail-closed
    /// no-op instead of falling through to the legacy child-specific refiner.
    /// `from_env` leaves this false so explicit research/test options retain
    /// their historical behavior; only verifier configuration sets it true.
    pub require_complete_clip_root_bank: bool,
    /// #ab-parity-interm (dark, `NY_AB_PARITY_INTERM=1`): give EACH last-ReLU
    /// refinement target row its OWN α, optimized against its OWN bound (the
    /// auto_LiRPA per-target decoupling), instead of the ONE shared slope set the
    /// base α′ lane ascends against the scalarized objective. Implies the α′ lane
    /// (`from_env` auto-enables it) and DROPS the once-per-process shared-α reuse
    /// (fresh per-target ascent each batch). See [`ab_parity_interm_enabled`].
    /// `false` ⇒ byte-identical.
    pub per_target: bool,
    /// #ab-parity-interm FC-head box-width probe (`NY_AB_INTERM_PROBE=1`, dark,
    /// stderr-only). See [`ab_interm_probe_enabled`].
    pub interm_box_probe: bool,
    /// #clip-interm-guard FAIL-CLOSED runtime guard (`NY_CLIP_INTERM=1`, dark,
    /// default false = byte-identical). When set (and the clip is active), every
    /// clip-tightened row is checked against a bounded directed adversarial sample
    /// ([`clip_guard_verify_domain`]) BEFORE it is merged; on any feasible point
    /// outside the tightened box, or a non-finite folded coefficient/penalty, the
    /// seed node reverts to the inherited PARENT bound for that child. See
    /// [`clip_interm_umbrella_enabled`]. Never touches the OFF path.
    #[allow(dead_code)]
    pub clip_guard: bool,
}

impl IntermRefineOptions {
    /// Deterministic scored configuration for certificate-backed Complete
    /// Clipping. Research environment variables must not alter which layers,
    /// rows, α schedules, caches, or redo policy feed verdict-bearing bounds.
    fn production_complete_clip(deadline: Option<Instant>, selective_topk: usize) -> Self {
        Self {
            deadline,
            unstable_rows_only: true,
            // Infeasibility remains the responsibility of established
            // proof-producing paths; Complete Clip only tightens enclosures.
            prune: false,
            layers: 1,
            max_dim: 2048,
            deep_max_rows: 256,
            selective_topk,
            prune_tol: 1e-4,
            // Probe output is observational and cannot alter arithmetic.
            probe: probe_enabled(),
            min_depth: 0,
            seeds: None,
            alpha_iters: 0,
            alpha_lr: 0.05,
            alpha_max_rows: 64,
            alpha_reopt: false,
            alpha_store: None,
            adaptive_latch: None,
            adaptive_floor: 4,
            redo_every: 0,
            wide_max_n: 0,
            joint_margin: false,
            margin_weights: None,
            clip_resnet: true,
            require_complete_clip_root_bank: true,
            per_target: false,
            interm_box_probe: false,
            clip_guard: false,
        }
    }

    pub(in crate::beta_crown::engine::graph) fn from_env() -> Self {
        let joint_margin = joint_interm_alpha_enabled();
        // #ab-parity-interm per-target mode is itself a (per-target) α′ ascent.
        let per_target = ab_parity_interm_enabled();
        // Joint margin-directed mode IS an α′ ascent (with reweighted objective),
        // so it implies the α′ lane even if NY_INTERM_REFINE_ALPHA is unset.
        let alpha_on = interm_refine_alpha_enabled() || joint_margin || per_target;
        Self {
            deadline: None,
            unstable_rows_only: interm_refine_unstable_rows_only(),
            prune: interm_refine_prune_enabled(),
            layers: interm_refine_layers(),
            max_dim: interm_refine_max_dim(),
            deep_max_rows: interm_refine_deep_max_rows(),
            selective_topk: interm_refine_selective_topk(),
            prune_tol: interm_refine_prune_tol(),
            probe: probe_enabled(),
            min_depth: interm_refine_min_depth(),
            seeds: interm_refine_seed_names(),
            alpha_iters: if alpha_on {
                interm_refine_alpha_iters()
            } else {
                0
            },
            alpha_lr: interm_refine_alpha_lr(),
            alpha_max_rows: interm_refine_alpha_max_rows(),
            alpha_reopt: interm_refine_alpha_reopt(),
            alpha_store: alpha_on.then(global_alpha_prime_store),
            adaptive_latch: interm_refine_adaptive_enabled().then(global_adaptive_latch),
            adaptive_floor: interm_refine_adaptive_floor(),
            redo_every: interm_refine_redo_every(),
            wide_max_n: interm_refine_wide_max_n(),
            joint_margin,
            // Computed at the batched call site (needs the tail map + spec rows);
            // `from_env` cannot see the graph, so the env-default entry leaves it
            // None (uniform) and the joint path fills it in `refine_last_relu_*`.
            margin_weights: None,
            // Both proposed environment entry points are authority-quarantined.
            // Explicitly constructed unit-test options can still exercise the
            // clip and guard without granting production verdict authority.
            clip_resnet: clip_interm_resnet_batched_enabled() || clip_interm_umbrella_enabled(),
            require_complete_clip_root_bank: false,
            per_target,
            interm_box_probe: ab_interm_probe_enabled(),
            clip_guard: clip_interm_umbrella_enabled(),
        }
    }
}

/// Authority gate for the legacy `NY_CLIP_INTERM_RESNET` request.
///
/// Quarantined for the same reason as [`clip_interm_umbrella_enabled`]: these
/// refined caches feed verdict-authoritative CROWN bounds. The environment cannot
/// enable this path; direct options remain available to unit tests.
fn clip_interm_resnet_batched_enabled() -> bool {
    false
}

/// Authority gate for the certified batched intermediate clip
/// (`NY_CLIP_INTERM_CERTIFIED=1`, governed by `ny_levers`).
///
/// HISTORY. This was hard-`false` with the rationale "a bounded random/directed
/// sample … is not a proof". That rationale described the LEGACY guard; the
/// live bank path it now arms is certified end-to-end, reviewed 2026-08-13
/// (docs/CLIP_APPLY_ENCLOSURE_PROOF_DESIGN_2026-08-12.md): every applied row is
/// minted (`CertifiedLayerCapture`), scope-validated
/// (`validate_for_clip_in_scope` — constraint tokens once per source, objective
/// tokens per target, the constraint callback reads only from `validated`),
/// split half-spaces use the sound necessary-condition side in both directions,
/// the coordinate solve carries an independent outward checker, and the armed
/// path cannot reach the unminted legacy lane (`clip_seed_domain` has zero
/// production callers). The runtime sample guard stays armed as defense in
/// depth, no longer as the authority argument.
///
/// The lever ships dark (default OFF) until the armed 220-row moat sweep lands
/// its receipts in-tree; the legacy `NY_CLIP_INTERM`/`NY_CLIP_INTERM_RESNET`
/// spellings remain permanently inert.
fn clip_interm_umbrella_enabled() -> bool {
    ny_levers::read(&ny_levers::decls::collection::CLIP_INTERM_CERTIFIED)
        .value
        .as_bool()
}

#[cfg(test)]
#[test]
fn clip_interm_umbrella_is_dark_by_default_and_armed_only_by_the_governed_lever() {
    ny_test_utils::env::with_serialized_env_vars(
        &[("NY_CLIP_INTERM_RESNET", "1"), ("NY_CLIP_INTERM", "1")],
        || {
            let _unset = ny_test_utils::env::ScopedEnvVar::unset("NY_CLIP_INTERM_CERTIFIED");
            // Legacy spellings stay inert; absent lever means OFF.
            assert!(!clip_interm_umbrella_enabled());
        },
    );
    ny_test_utils::env::with_serialized_env_vars(&[("NY_CLIP_INTERM_CERTIFIED", "1")], || {
        assert!(clip_interm_umbrella_enabled());
    });
    ny_test_utils::env::with_serialized_env_vars(&[("NY_CLIP_INTERM_CERTIFIED", "true")], || {
        // Malformed spellings must never arm a verdict-adjacent lane.
        assert!(!clip_interm_umbrella_enabled());
    });
}

#[cfg(test)]
#[test]
fn legacy_batched_clip_env_gate_is_quarantined_but_explicit_test_options_remain() {
    ny_test_utils::env::with_serialized_env_vars(
        &[("NY_CLIP_INTERM_RESNET", "1"), ("NY_CLIP_INTERM", "1")],
        || {
            let _refine_unset = ny_test_utils::env::ScopedEnvVar::unset("NY_INTERM_REFINE");
            let _cert_unset = ny_test_utils::env::ScopedEnvVar::unset("NY_CLIP_INTERM_CERTIFIED");
            assert!(!clip_interm_resnet_batched_enabled());
            assert!(!clip_interm_umbrella_enabled());
            assert!(!interm_refine_enabled());

            let mut options = IntermRefineOptions::from_env();
            assert!(!options.clip_resnet);
            assert!(!options.clip_guard);

            // Research and soundness tests opt in structurally, without an env
            // path that can affect production verdicts.
            options.clip_resnet = true;
            options.clip_guard = true;
            assert!(options.clip_resnet);
            assert!(options.clip_guard);
        },
    );
}

#[cfg(test)]
#[test]
fn production_complete_clip_options_ignore_research_schedules() {
    let deadline = Some(Instant::now() + std::time::Duration::from_secs(30));
    let opts = IntermRefineOptions::production_complete_clip(deadline, 20);
    assert_eq!(opts.deadline, deadline);
    assert!(opts.unstable_rows_only);
    assert!(!opts.prune);
    assert_eq!(opts.layers, 1);
    assert_eq!(opts.selective_topk, 20);
    assert_eq!(opts.min_depth, 0);
    assert!(opts.seeds.is_none());
    assert_eq!(opts.alpha_iters, 0);
    assert!(opts.alpha_store.is_none());
    assert!(opts.adaptive_latch.is_none());
    assert_eq!(opts.redo_every, 0);
    assert!(!opts.joint_margin);
    assert!(opts.clip_resnet);
    assert!(opts.require_complete_clip_root_bank);
    assert!(!opts.per_target);
    assert!(!opts.clip_guard);
}

/// K-restart budget for the fail-closed clip guard (`NY_CLIP_INTERM_GUARD_K`,
/// default 24). Each restart is one directed sample + one true forward through
/// the graph, so this bounds the guard's per-domain cost. Production clip
/// authority is quarantined; explicit research tests can still exercise it.
#[allow(dead_code)]
fn clip_interm_guard_restarts() -> usize {
    std::env::var("NY_CLIP_INTERM_GUARD_K")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(24)
        .max(1)
}

/// Result of a refinement call: the refined per-domain caches plus, when the
/// prune lane is on, which domains were PROVEN infeasible (empty constraint
/// set — verified vacuously). `infeasible[i]` is only ever set under
/// `NY_INTERM_REFINE_PRUNE=1`.
pub(in crate::beta_crown::engine::graph) struct IntermRefineOutcome {
    pub caches: Vec<HashMap<String, Arc<BoundedTensor>>>,
    pub infeasible: Vec<bool>,
}

/// Walk the unary chain UP from `output_node` to the nearest ReLU (the
/// network's LAST ReLU) and return `(relu_name, seed_node)` where `seed_node`
/// is that ReLU's input node — the refinement backward's start node. `None` on
/// any non-unary node before a ReLU (no clean last-ReLU chain), on reaching the
/// network input, or on a ReLU fed directly by the network input (nothing below
/// the seed to refine with).
pub(in crate::beta_crown::engine::graph) fn find_last_relu_seed(
    graph: &GraphNetwork,
    output_node: &str,
) -> Option<(String, String)> {
    let mut current = output_node.to_string();
    for _ in 0..=graph.nodes.len() {
        if current == NETWORK_INPUT {
            return None;
        }
        let node = graph.nodes.get(&current)?;
        if matches!(node.layer, Layer::ReLU(_)) {
            let seed = node.require_unary_input().ok()?.to_string();
            if seed == NETWORK_INPUT {
                return None;
            }
            return Some((current, seed));
        }
        if node.inputs.len() != 1 {
            return None;
        }
        current = node.inputs[0].clone();
    }
    None
}

/// Per-seed-neuron MARGIN WEIGHTS for the joint α′ objective (#joint-interm-alpha,
/// [`joint_interm_alpha_enabled`]). The seed layer feeds the last ReLU
/// (`relu_name`), whose output is consumed by the tail linear map `W_tail`
/// (`Relu_57 → Gemm_58 → output` on resnet_medium). For spec row `s` the margin's
/// coefficient at the ReLU output is `a_s = spec_s · W_tail`; a seed neuron `j`
/// enters the margin's LOWER bound only through the chord (upper) relaxation when
/// `a_s[j] < 0`, with magnitude `|a_s[j]|`. So the margin's sensitivity to
/// tightening neuron `j`'s refined lower bound is `m_j = Σ_s max(−a_s[j], 0) ≥ 0`
/// — the reweighting the joint lane applies to the α′ refinement objective.
/// `None` (⇒ uniform fallback, sound) when the tail is not a single linear map
/// directly consuming the last ReLU or the dims disagree.
pub(in crate::beta_crown::engine::graph) fn compute_margin_weights(
    graph: &GraphNetwork,
    output_node: &str,
    spec_matrix: &Array2<f32>,
) -> Option<Arc<[f32]>> {
    let mut never_past_deadline = || false;
    let mut dot = |spec: &Array2<f32>, weight: &Array2<f32>| spec.dot(weight);
    compute_margin_weights_with_deadline_check_and_dot(
        graph,
        output_node,
        spec_matrix,
        MARGIN_WEIGHT_LIMITS,
        &mut never_past_deadline,
        &mut dot,
    )
    .ok()
    .flatten()
}

fn compute_margin_weights_with_deadline(
    graph: &GraphNetwork,
    output_node: &str,
    spec_matrix: &Array2<f32>,
    deadline: Option<Instant>,
) -> ny_core::Result<Option<Arc<[f32]>>> {
    let mut past_deadline = || deadline.is_some_and(|d| Instant::now() >= d);
    let mut dot = |spec: &Array2<f32>, weight: &Array2<f32>| spec.dot(weight);
    compute_margin_weights_with_deadline_check_and_dot(
        graph,
        output_node,
        spec_matrix,
        MARGIN_WEIGHT_LIMITS,
        &mut past_deadline,
        &mut dot,
    )
}

fn compute_margin_weights_with_deadline_check_and_dot<F, D>(
    graph: &GraphNetwork,
    output_node: &str,
    spec_matrix: &Array2<f32>,
    limits: MarginWeightLimits,
    past_deadline: &mut F,
    dot: &mut D,
) -> ny_core::Result<Option<Arc<[f32]>>>
where
    F: FnMut() -> bool,
    D: FnMut(&Array2<f32>, &Array2<f32>) -> Array2<f32>,
{
    let deadline_error = |phase: &str| {
        NyError::DeadlineExceeded(format!(
            "intermediate margin-weight planning exceeded deadline during {phase}"
        ))
    };
    if past_deadline() {
        return Err(deadline_error("entry"));
    }
    let Some((relu_name, _seed)) = find_last_relu_seed(graph, output_node) else {
        return Ok(None);
    };
    if past_deadline() {
        return Err(deadline_error("last-ReLU discovery"));
    }
    // The tail linear map: the UNIQUE Linear node directly consuming the last
    // ReLU's output. Any non-linear consumer, or fan-out to >1 consumer, bails
    // to the uniform objective (sound — the weights are advisory only).
    let mut tail: Option<&Array2<f32>> = None;
    for (node_index, node) in graph.nodes.values().enumerate() {
        if node_index.is_multiple_of(SELECTIVE_DEADLINE_POLL_STRIDE) && past_deadline() {
            return Err(deadline_error("tail discovery"));
        }
        if node.inputs.iter().any(|i| i == &relu_name) {
            match &node.layer {
                Layer::Linear(lin) if tail.is_none() => tail = Some(&lin.weight),
                _ => return Ok(None),
            }
        }
    }
    let Some(w) = tail else {
        return Ok(None);
    }; // (out_features × in_features)
    if w.nrows() != spec_matrix.ncols() || w.ncols() == 0 {
        return Ok(None); // the spec's output dim must match the tail's out dim
    }
    // Do not let the opaque matrixmultiply path materialize layout repairs that
    // are absent from the checked owned-byte plan. Advisory weights can safely
    // fall back to uniform for unusual layouts.
    if !spec_matrix.is_standard_layout() || !w.is_standard_layout() {
        return Ok(None);
    }
    let plan = MarginWeightPlan::checked(spec_matrix.nrows(), w.nrows(), w.ncols(), limits)?;
    if past_deadline() {
        return Err(deadline_error("pre-allocation plan"));
    }

    let mut m = Vec::new();
    m.try_reserve_exact(plan.weight_elements).map_err(|_| {
        NyError::InvalidSpec("margin-weight vector allocation refused after checked plan".into())
    })?;
    m.resize(plan.weight_elements, 0.0f32);
    if past_deadline() {
        return Err(deadline_error("before dense dot"));
    }

    // a = spec · W_tail → (num_specs × in_features). Margin weight of seed
    // neuron j = Σ_s max(−a[s][j], 0) (upper-branch tail-coefficient mass).
    let a = dot(spec_matrix, w);
    if a.nrows() != spec_matrix.nrows() || a.ncols() != w.ncols() || a.len() != plan.dot_elements {
        return Err(NyError::shape_mismatch(
            vec![spec_matrix.nrows(), w.ncols()],
            a.shape().to_vec(),
        ));
    }
    if past_deadline() {
        return Err(deadline_error("dense dot"));
    }
    let mut fold_cells = 0usize;
    for row in a.rows() {
        for (j, &v) in row.iter().enumerate() {
            if fold_cells.is_multiple_of(SELECTIVE_DEADLINE_POLL_STRIDE) && past_deadline() {
                return Err(deadline_error("dot reduction"));
            }
            fold_cells = fold_cells.saturating_add(1);
            if v < 0.0 {
                m[j] += -v;
            }
        }
    }
    drop(a);
    if !m.iter().any(|&v| v > 0.0) {
        return Ok(None); // no upper-branch mass — nothing to steer; keep uniform
    }
    if past_deadline() {
        return Err(deadline_error("return allocation"));
    }
    debug_assert!(plan.peak_owned_bytes <= limits.max_host_bytes);
    debug_assert!(plan.arithmetic_ops <= limits.max_work);
    Ok(Some(Arc::from(m)))
}

/// Select a shared top-K objective set for one intermediate seed/domain batch.
///
/// The score mirrors αβ-CROWN's `DomainClipScorer`: triangle intercept
/// `(-l*u)/(u-l)` times downstream margin sensitivity when the last-ReLU tail
/// map provides one, or intercept alone for deeper seeds. Since NY uses one
/// identity seed shared by the whole wide batch, a neuron's score is the
/// maximum over domains (rather than a separate top-K per domain). This
/// hard-bounds the GPU row count. Selection is advisory: every omitted neuron
/// keeps its inherited enclosure. Premise rows are retained separately because
/// they are needed as necessary-condition constraint sources.
#[cfg(test)]
fn select_intermediate_objective_rows(
    caches: &[HashMap<String, Arc<BoundedTensor>>],
    seed_node: &str,
    candidates: &[usize],
    premise: &[bool],
    topk: usize,
    margin_weights: Option<&[f32]>,
) -> Vec<usize> {
    select_intermediate_objective_rows_with_deadline(
        caches,
        seed_node,
        candidates,
        premise,
        topk,
        margin_weights,
        None,
    )
}

fn selective_row_work_bytes(candidate_len: usize, premise_len: usize) -> Option<usize> {
    // Conservatively count the already-materialized candidate and premise
    // inputs because the selector keeps them live, plus the maximum scored and
    // output vectors. The in-place heapsorts below are allocation-free, so
    // their scratch charge is zero rather than an undocumented library-sort
    // allocation.
    let candidate_input = candidate_len.checked_mul(size_of::<usize>())?;
    let premise_input = premise_len.checked_mul(size_of::<bool>())?;
    let scored = candidate_len.checked_mul(size_of::<(usize, f64)>())?;
    let output = candidate_len.checked_mul(size_of::<usize>())?;
    [candidate_input, premise_input, scored, output]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
}

fn validate_selective_row_budget(
    candidate_len: usize,
    premise_len: usize,
    cache_len: usize,
    score: bool,
) -> Option<()> {
    if selective_row_work_bytes(candidate_len, premise_len)? > SELECTIVE_CLIP_MAX_HOST_BYTES {
        return None;
    }
    let levels = if candidate_len <= 1 {
        1
    } else {
        (usize::BITS - (candidate_len - 1).leading_zeros()) as usize
    };
    let scoring = if score {
        candidate_len.checked_mul(cache_len)?
    } else {
        0
    };
    let sorting = candidate_len
        .checked_mul(levels)?
        .checked_mul(SELECTIVE_SORT_OP_WEIGHT)?;
    let total = scoring.checked_add(sorting)?.checked_add(candidate_len)?;
    (total <= SELECTIVE_CLIP_MAX_WORK).then_some(())
}

#[allow(clippy::too_many_arguments)]
fn select_intermediate_objective_rows_with_deadline(
    caches: &[HashMap<String, Arc<BoundedTensor>>],
    seed_node: &str,
    candidates: &[usize],
    premise: &[bool],
    topk: usize,
    margin_weights: Option<&[f32]>,
    deadline: Option<Instant>,
) -> Vec<usize> {
    let mut past_deadline = || deadline.is_some_and(|d| Instant::now() >= d);
    select_intermediate_objective_rows_with_deadline_check(
        caches,
        seed_node,
        candidates,
        premise,
        topk,
        margin_weights,
        &mut past_deadline,
    )
    .unwrap_or_default()
}

#[allow(clippy::too_many_arguments)]
fn select_intermediate_objective_rows_with_deadline_check<F>(
    caches: &[HashMap<String, Arc<BoundedTensor>>],
    seed_node: &str,
    candidates: &[usize],
    premise: &[bool],
    topk: usize,
    margin_weights: Option<&[f32]>,
    past_deadline: &mut F,
) -> Option<Vec<usize>>
where
    F: FnMut() -> bool,
{
    if past_deadline()
        || validate_selective_row_budget(candidates.len(), premise.len(), caches.len(), topk > 0)
            .is_none()
    {
        return None;
    }

    let mut keep = Vec::new();
    keep.try_reserve_exact(candidates.len()).ok()?;
    let mut scored = Vec::new();
    if topk > 0 {
        scored.try_reserve_exact(candidates.len()).ok()?;
    }
    for (position, &j) in candidates.iter().enumerate() {
        if position.is_multiple_of(SELECTIVE_DEADLINE_POLL_STRIDE) && past_deadline() {
            return None;
        }
        let is_premise = *premise.get(j)?;
        if topk == 0 || is_premise {
            keep.push(j);
        } else {
            scored.push((j, f64::NEG_INFINITY));
        }
    }
    if topk == 0 {
        return (!past_deadline()).then_some(keep);
    }

    let mut score_cells = 0usize;
    for cache in caches {
        if past_deadline() {
            return None;
        }
        let Some(entry) = cache.get(seed_node) else {
            continue;
        };
        // Resident bound tensors are contiguous in this lane. A non-standard
        // layout is an advisory-selector refusal for that cache, never authority.
        let (Some(lower), Some(upper)) = (
            entry.lower().as_slice_memory_order(),
            entry.upper().as_slice_memory_order(),
        ) else {
            continue;
        };
        for (j, score) in &mut scored {
            if score_cells.is_multiple_of(SELECTIVE_DEADLINE_POLL_STRIDE) && past_deadline() {
                return None;
            }
            score_cells = score_cells.saturating_add(1);
            let (Some(&l), Some(&u)) = (lower.get(*j), upper.get(*j)) else {
                continue;
            };
            if !l.is_finite() || !u.is_finite() || l >= 0.0 || u <= 0.0 || u <= l {
                continue;
            }
            let sensitivity = margin_weights
                .and_then(|m| m.get(*j))
                .copied()
                .filter(|v| v.is_finite() && *v >= 0.0)
                .unwrap_or(1.0);
            let intercept = (-f64::from(l) * f64::from(u)) / (f64::from(u) - f64::from(l));
            *score = score.max(intercept * f64::from(sensitivity));
        }
    }

    crate::complete_clip::deadline_heapsort_by(
        &mut scored,
        past_deadline,
        "selective objective score sort",
        |(ja, sa), (jb, sb)| {
            sb.partial_cmp(sa)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(ja.cmp(jb))
        },
    )
    .ok()?;
    keep.extend(scored.into_iter().take(topk).map(|(j, _)| j));

    crate::complete_clip::deadline_heapsort_by(
        &mut keep,
        past_deadline,
        "selective objective index sort",
        usize::cmp,
    )
    .ok()?;
    Some(keep)
}

/// Exact production correspondence between a pure ReLU split path and the
/// β-state consumed by the same child CROWN pass.  Provenance binds the affine
/// rows to the history; this check prevents accidentally computing those rows
/// with stale or foreign multipliers while labelling them with the live path.
fn beta_state_matches_pure_relu_history(
    history: &GraphSplitHistory,
    beta: &GraphBetaState,
) -> bool {
    if !history.is_pure_relu_at_zero()
        || history.constraints.len() != beta.entries.len()
        || history.depth() != beta.entries.len()
    {
        return false;
    }
    history
        .constraints
        .iter()
        .zip(&beta.entries)
        .all(|(constraint, entry)| {
            constraint.node_name() == entry.node_name()
                && constraint.neuron_idx() == entry.neuron_idx()
                && entry.split_point() == 0.0
                && entry.sign() == if constraint.is_active() { 1.0 } else { -1.0 }
                && entry.value().is_finite()
                && entry.value() >= 0.0
        })
}

/// ReLU premises scheduled for Complete Clipping in the current outer BaB
/// round. The full history remains attached to every provenance token; this
/// set controls only which half-spaces are concretized and which source rows
/// need capturing. After the two warm-up rounds, αβ-CROWN walks its fixed
/// split-node dictionary in network order, takes the first nonempty layer, and
/// uses that layer's most recent split. This is intentionally not the globally
/// newest history entry.
fn scheduled_relu_constraints<'a>(
    graph: &GraphNetwork,
    history: &'a GraphSplitHistory,
    use_all_history: bool,
) -> Option<Vec<&'a GraphNeuronConstraint>> {
    if use_all_history {
        return Some(history.constraints.iter().collect());
    }
    for node_name in graph.exec_order().ok()? {
        let Some(node) = graph.nodes.get(node_name) else {
            continue;
        };
        if !matches!(node.layer, Layer::ReLU(_)) {
            continue;
        }
        if let Some(constraint) = history
            .constraints
            .iter()
            .rev()
            .find(|constraint| constraint.node_name() == node_name)
        {
            return Some(vec![constraint]);
        }
    }
    Some(Vec::new())
}

/// Translate selected objective neurons into captured-row indices without
/// allowing a future selector/map drift to silently shrink the top-K set.
fn exact_objective_rows(
    objective_neurons: &[usize],
    row_of_neuron: &[usize],
) -> Option<Vec<usize>> {
    let mut rows = Vec::new();
    rows.try_reserve_exact(objective_neurons.len()).ok()?;
    for &neuron in objective_neurons {
        let row = *row_of_neuron.get(neuron)?;
        if row == usize::MAX {
            return None;
        }
        rows.push(row);
    }
    rows.sort_unstable();
    rows.dedup();
    (rows.len() == objective_neurons.len()).then_some(rows)
}

/// αβ-CROWN DomainClipper's per-domain, per-layer objective score:
///
/// `((-l).clamp_min(0) * u.clamp_min(0) / (u-l))
///    * (-lA.mean(spec_rows)).clamp_min(0)`.
///
/// The clamp is deliberately applied *after* the specification mean. Summing
/// per-row negative parts is a different ranking when lA signs disagree.
fn select_complete_clip_rows_from_mean_la(
    root_entry: &BoundedTensor,
    child_entry: &BoundedTensor,
    child_history: &GraphSplitHistory,
    relu_name: &str,
    mean_lower_a: &Array1<f32>,
    topk: usize,
    deadline: Option<Instant>,
) -> Option<Vec<usize>> {
    if root_entry.is_empty()
        || root_entry.len() != child_entry.len()
        || mean_lower_a.len() != child_entry.len()
    {
        return None;
    }
    let (root_lower, root_upper) = (
        root_entry.lower().as_slice_memory_order()?,
        root_entry.upper().as_slice_memory_order()?,
    );
    let (child_lower, child_upper) = (
        child_entry.lower().as_slice_memory_order()?,
        child_entry.upper().as_slice_memory_order()?,
    );
    let mut active_clamp = vec![false; child_entry.len()];
    let mut inactive_clamp = vec![false; child_entry.len()];
    for constraint in &child_history.constraints {
        if constraint.node_name() != relu_name {
            continue;
        }
        let neuron = constraint.neuron_idx();
        if neuron >= child_entry.len() {
            return None;
        }
        if constraint.is_active() {
            active_clamp[neuron] = true;
        } else {
            inactive_clamp[neuron] = true;
        }
    }

    let mut scored = Vec::new();
    scored.try_reserve_exact(root_entry.len()).ok()?;
    for neuron in 0..root_entry.len() {
        if neuron.is_multiple_of(SELECTIVE_DEADLINE_POLL_STRIDE)
            && deadline.is_some_and(|value| Instant::now() >= value)
        {
            return None;
        }
        let (root_l, root_u) = (root_lower[neuron], root_upper[neuron]);
        if !root_l.is_finite() || !root_u.is_finite() || root_l >= 0.0 || root_u <= 0.0 {
            continue;
        }
        // αβ-CROWN's prospective depth expansion repeats the parent lA while
        // clamping each final leaf's split bounds. NY children retain the
        // parent's node map until their verdict pass, so apply the exact same
        // history clamps virtually for this score.
        let mut l = child_lower[neuron];
        let mut u = child_upper[neuron];
        if active_clamp[neuron] {
            l = l.max(0.0);
        }
        if inactive_clamp[neuron] {
            u = u.min(0.0);
        }
        if !l.is_finite() || !u.is_finite() || l > u {
            return None;
        }

        let mean = mean_lower_a[neuron];
        if !mean.is_finite() {
            return None;
        }
        let sensitivity = (-mean).max(0.0);
        let denominator = u - l;
        let intercept = if denominator > 0.0 {
            ((-l).max(0.0) * u.max(0.0)) / denominator
        } else {
            0.0
        };
        let score = intercept * sensitivity;
        if !score.is_finite() {
            return None;
        }
        scored.push((neuron, score));
    }
    if deadline.is_some_and(|value| Instant::now() >= value) {
        return None;
    }
    scored.sort_by(|(left_neuron, left_score), (right_neuron, right_score)| {
        right_score
            .total_cmp(left_score)
            .then_with(|| left_neuron.cmp(right_neuron))
    });
    if topk > 0 {
        scored.truncate(topk.min(scored.len()));
    }
    let mut selected: Vec<usize> = scored.into_iter().map(|(neuron, _)| neuron).collect();
    selected.sort_unstable();
    Some(selected)
}

fn select_complete_clip_rows_from_cached_las(
    root_entry: &BoundedTensor,
    child_entry: &BoundedTensor,
    child_history: &GraphSplitHistory,
    relu_name: &str,
    lower_as: &[&Array2<f32>],
    topk: usize,
    deadline: Option<Instant>,
) -> Option<Vec<usize>> {
    if lower_as.is_empty()
        || lower_as
            .iter()
            .any(|lower_a| lower_a.nrows() != 1 || lower_a.ncols() != child_entry.len())
    {
        return None;
    }
    let mut mean_lower_a = Array1::<f32>::zeros(child_entry.len());
    for neuron in 0..child_entry.len() {
        if neuron.is_multiple_of(SELECTIVE_DEADLINE_POLL_STRIDE)
            && deadline.is_some_and(|value| Instant::now() >= value)
        {
            return None;
        }
        let mut sum = 0.0f32;
        for (position, lower_a) in lower_as.iter().enumerate() {
            if position.is_multiple_of(SELECTIVE_DEADLINE_POLL_STRIDE)
                && deadline.is_some_and(|value| Instant::now() >= value)
            {
                return None;
            }
            let coefficient = lower_a[[0, neuron]];
            if !coefficient.is_finite() {
                return None;
            }
            sum += coefficient;
        }
        mean_lower_a[neuron] = sum / lower_as.len() as f32;
    }
    select_complete_clip_rows_from_mean_la(
        root_entry,
        child_entry,
        child_history,
        relu_name,
        &mean_lower_a,
        topk,
        deadline,
    )
}

/// PURE-CHAIN SELECTOR GATHER gate (`NY_CLIP_CHAIN_GATHER=1`, DARK, default OFF
/// = byte-identical — #clip-chain-gather). It currently clears the measured
/// extraction refusal but still stops at `gather_no_sound_gpu_backend`; keep it
/// DARK until a sound backend and a scored delivery path are demonstrated.
///
/// WHAT IT OPENS. [`complete_clip_mean_las_from_gpu_probed`] reconstructs a
/// wave's per-spec lower-A frontier through `prep_resnet_domain_ext`, whose
/// segment extraction ends at the terminal gate `!saw_residual &&
/// !allow_pure_chain` (`resnet_decompose.rs`). A feed-forward network has no
/// `Layer::Add`, so `saw_residual` is structurally always false and the gate
/// fires on 100% of reconstruct attempts — measured 2026-08-10 on relusplitter
/// `oval21-benchmark_cifar_base_kw` img3365 eps0.0407 @60s: 710 of 878 waves
/// refused there, 81% of the whole search, every one at `wall_ms=0`. The
/// decomposition itself SUCCEEDS (all 15 other extraction exits measured zero);
/// a complete `[Chain(..)]` is built and then discarded on a residual-COUNT
/// predicate alone.
///
/// WHY A SECOND GATE RATHER THAN REUSING `NY_BAB_CHAIN_WIDE`. That key already
/// sets the same `allow_pure_chain` boolean, but it is not scoped to this
/// consumer: it simultaneously routes the BaB batched β lane's VERDICT-BEARING
/// bound onto the wide replacement, which
/// `chain_wide_replacement_oracle_body` measured LOOSER on 17 of 26 rows. So
/// the only existing key that lifts this refusal is coupled to a bound-quality
/// regression, which is precisely what makes the refusal overbroad rather than
/// merely closed. This gate decouples them.
///
/// WHY IT CANNOT MOVE A BOUND. The reconstructed frontier is consumed ONLY as a
/// ranking score: [`select_complete_clip_rows_from_mean_la`] turns each mean
/// coefficient into `sensitivity = (-mean).max(0.0)` for a top-K sort, and what
/// [`BetaCrownVerifier::publish_complete_clip_decisions`] stores is a
/// `HashMap<layer, Arc<[usize]>>` of NEURON INDICES — no bound value crosses
/// this path. Downstream those indices only ever SET bits in `candidate_mask`,
/// and every captured row is independently minted by
/// `bind_root_sound_crown_rows_to_history` + `mint_certified_affine_enclosure`,
/// which fail closed per domain. A different (or adversarial) score can
/// therefore change WHICH neurons are certified, never the arithmetic that
/// certifies them.
///
/// OFF ⇒ the gather's `allow_pure_chain` is exactly the historical
/// `bab_chain_wide_enabled()` and routing is byte-identical.
pub(crate) fn complete_clip_chain_gather_enabled() -> bool {
    matches!(
        std::env::var("NY_CLIP_CHAIN_GATHER").ok().as_deref(),
        Some("1")
    )
}

/// Compact DomainClipper scoring snapshot. The GPU gather is reduced across
/// the full specification dimension immediately, so prospective depth leaves
/// share O(sum ReLU widths) data rather than cloning O(specs × widths) caches.
#[derive(Debug)]
pub(crate) struct CompleteClipMeanLowerA {
    spec_rows: usize,
    by_relu: HashMap<String, Array1<f32>>,
}

fn complete_clip_spec_matrix_is_row_major(spec_matrix: &Array2<f32>) -> bool {
    spec_matrix.is_standard_layout() && spec_matrix.as_slice_memory_order().is_some()
}

fn complete_clip_gpu_mean_parent_count_admitted(parent_count: usize) -> bool {
    (1..=COMPLETE_CLIP_GPU_MEAN_LA_MAX_RECONSTRUCT_DOMAINS).contains(&parent_count)
}

/// WHICH admission predicate refuses [`complete_clip_mean_las_from_gpu_probed`], or
/// `None` when the wave is admitted.
///
/// This is the single source of truth for that decision: the function itself
/// early-returns on `self.is_some()`, so the reported reason cannot drift from
/// the behaviour. It is diagnostic only and decides nothing on its own.
///
/// Why it exists. The predicates were previously one `||` cascade whose failure
/// collapsed into a single `gpu_full_spec_la_refused` bucket at the call site.
/// Measured 2026-08-10 on relusplitter `oval21 base_kw`, that bucket absorbed 801
/// of 877 waves — the whole reason per-domain tightening stops publishing past
/// depth 4 — with no way to tell WHICH predicate fired, and the obvious suspect
/// (the reconstruct domain cap) was disproved by an A/B that changed nothing. One
/// reason per predicate turns that into a single named cause in one run.
///
/// The strings are stable identifiers consumed by the
/// `[complete-clip-decision-precompute]` probe line; keep them snake_case and do
/// not reuse one for two predicates.
/// Sound GPU CROWN backends admissible for a wide Complete Clipping gather.
///
/// Extracted so the gather and its diagnostic cannot disagree: the gather calls
/// this and refuses on empty, and the caller calls the same function to label
/// WHY it refused. Do not inline a second copy of these filters.
pub(crate) fn complete_clip_gpu_candidates(
    engine: &dyn GemmEngine,
    deadline: Option<Instant>,
) -> Vec<&dyn ny_core::GpuCrownBackward> {
    let local_gpu = engine
        .as_gpu_crown_backward()
        .filter(|gpu| gpu.provides_sound_gpu_crown())
        .filter(|gpu| deadline.is_none() || gpu.honors_crown_backward_deadline());
    let global_gpu = crate::sound_gpu_gate::sound_gpu_crown_for_wide_with_deadline(deadline)
        .filter(|gpu| deadline.is_none() || gpu.honors_crown_backward_deadline());
    let mut gpu_candidates: Vec<&dyn ny_core::GpuCrownBackward> = Vec::with_capacity(2);
    if let Some(gpu) = global_gpu {
        gpu_candidates.push(gpu);
    }
    if let Some(gpu) = local_gpu {
        if gpu_candidates
            .iter()
            .all(|candidate| !std::ptr::eq(*candidate, gpu))
        {
            gpu_candidates.push(gpu);
        }
    }
    gpu_candidates
}

/// Diagnostic: is ANY sound GPU CROWN backend admissible right now?
///
/// `false` means the wide gather cannot run at all on this host, whatever the
/// wave looks like. Ordinary WGPU devices never qualify here, and an explicit
/// typed request may have failed one of its five live rungs (in which case the
/// CLI reports the refusal and uses CPU). That is an expected capability result
/// on an unsupported host, not a fault.
///
/// Superseded as a call-site probe by the `gather_no_sound_gpu_backend` exit of
/// [`complete_clip_mean_las_from_gpu_probed`], which reports the same fact AT the
/// exit rather than re-deriving it afterwards. Kept because it answers the
/// question standalone, without running a wave.
#[allow(dead_code)]
pub(crate) fn complete_clip_sound_gpu_backend_available(
    engine: &dyn GemmEngine,
    deadline: Option<Instant>,
) -> bool {
    !complete_clip_gpu_candidates(engine, deadline).is_empty()
}

/// Host reconstruct-wave ceiling. Deliberately separate from the GPU constant:
/// the GPU number was a CIFAR-100 cost/benefit for a device dispatch, while the
/// host cost is one spec-seeded CROWN pass per parent — the same unit of work
/// BaB already performs per domain, so the observed 8-domain waves are
/// affordable when the deadline admits them.
const COMPLETE_CLIP_HOST_MEAN_LA_MAX_RECONSTRUCT_DOMAINS: usize = 8;

/// Opt-in switch for the host mean-lA reconstruct (`NY_CLIP_HOST_MEAN_LA=1`).
///
/// Default OFF. A contemporaneous relusplitter oval21 `base_kw` pilot
/// (2026-08-12) reported that the lane worked as designed — refusals fell
/// 712→0 and published decisions rose 20→3,622 — without a conversion: the
/// 220-row summary was 43/220 in each arm, the
/// zero-lift split rate did not improve (42%→47%), the best bound regressed
/// (−5.057→−5.115), and the host passes consumed ~29% of branch throughput
/// (3,453→2,467 events in the same budget). Those machine-readable A/B receipts
/// were not committed, so the observation is diagnosis rather than admissible
/// provenance. Keep the lane dark until a retained current-path A/B qualifies
/// it.
///
/// Kept opt-in rather than deleted because it is the fallback way to exercise
/// the per-domain clip lane when no sound GPU CROWN backend is admitted for the
/// current run. The WGPU source gate is open, but only a typed explicit request
/// plus all five live per-device rungs can admit a device; ordinary or refused
/// WGPU devices remain ineligible. Anyone investigating the inert consumer on
/// such a run needs this switch.
fn complete_clip_host_mean_la_enabled() -> bool {
    ny_levers::read(&ny_levers::decls::collection::CLIP_HOST_MEAN_LA)
        .value
        .as_bool()
}

/// HOST sibling of [`complete_clip_mean_las_from_gpu`]: reconstruct the
/// DomainClipper scoring snapshot with one spec-seeded host CROWN pass per
/// parent, for hosts with no admissible sound GPU CROWN backend.
///
/// Historical motivation (2026-08-10/12, relusplitter oval21 `base_kw`): a
/// contemporaneous narrative reported that, with the then-current WGPU source
/// gate closed, 712 of 880 scoring waves refused with `no_sound_gpu_backend`,
/// the whole 60 s run published 20 tightened bounds, intermediate bounds froze
/// from depth 5, and 42% of BaB splits lifted the bound by exactly zero. This
/// lane restores the reconstruct on the CPU, whose verdict authority is not in
/// question.
///
/// SOUNDNESS SCOPE. The snapshot is selection-only: it ranks which neurons the
/// DomainClipper should consider, and every published decision gates bounds
/// that are independently certified by the Complete Clipping root bank
/// (`affine_provenance`). A wrong mean here can waste effort, never narrow an
/// uncertified bound. The pass itself runs on the same host engine that decides
/// verdicts, so the inputs are not accelerator suggestions either.
///
/// Cost control mirrors the GPU lane: start-headroom admission, a parent-count
/// ceiling, a per-parent deadline poll here, and the exact BaB deadline polled
/// inside the pass by `ensure_constrained_propagation_deadline`.
pub(crate) fn complete_clip_mean_las_from_host(
    verifier: &BetaCrownVerifier,
    graph: &GraphNetwork,
    parents: &[&MultiObjectiveGraphBabDomain],
    spec_matrix: &Array2<f32>,
    engine: &dyn GemmEngine,
    deadline: Option<Instant>,
) -> Option<Vec<Option<CompleteClipMeanLowerA>>> {
    if !complete_clip_host_mean_la_enabled()
        || parents.is_empty()
        || parents.len() > COMPLETE_CLIP_HOST_MEAN_LA_MAX_RECONSTRUCT_DOMAINS
        || spec_matrix.nrows() == 0
        || spec_matrix.nrows() > COMPLETE_CLIP_GPU_MEAN_LA_MAX_SPECS
        || !complete_clip_optional_start_budget_available(Instant::now(), deadline)
    {
        return None;
    }
    let deadline_live = || deadline.is_none_or(|value| Instant::now() < value);
    let exec_order = graph.exec_order().ok()?;
    let mut snapshots = Vec::new();
    snapshots.try_reserve_exact(parents.len()).ok()?;
    for parent in parents {
        if !deadline_live() {
            // Mirror the GPU lane: a mid-wave deadline abandons the wave rather
            // than publishing a half-scored one.
            return None;
        }
        let context = crate::beta_crown::domain::GraphCrownContext::new_with_node_bounds_map(
            parent.history(),
            None,
            Some(parent.node_bounds()),
            Some(engine),
        )
        .with_alpha(parent.alpha_state());
        let captured = match verifier.propagate_crown_with_graph_constraints_with_spec_matrix(
            graph,
            parent.input_bounds(),
            &context,
            Some(parent.beta_state()),
            spec_matrix,
            None,
            true,
        ) {
            Ok((_, _, captured)) => captured,
            // A per-parent failure leaves that parent on the historical
            // selector; a deadline failure abandons the wave above.
            Err(_) => None,
        };
        let Some(captured) = captured else {
            snapshots.push(None);
            continue;
        };
        let mut by_relu = HashMap::new();
        for relu_name in exec_order.iter() {
            let Some(relu) = graph.nodes.get(relu_name) else {
                continue;
            };
            if !matches!(relu.layer, Layer::ReLU(_)) {
                continue;
            }
            let Ok(seed_node) = relu.require_unary_input() else {
                continue;
            };
            if seed_node == NETWORK_INPUT {
                continue;
            }
            // Same key discipline as the cached path: the captured lower-A at
            // the ReLU key is DomainClipper's A_key for this layer. Spec-matrix
            // mode must yield exactly one row per specification; anything else
            // is a shape drift we refuse rather than average.
            let Some(matrix) = captured.lower_a.get(relu_name) else {
                continue;
            };
            if matrix.nrows() != spec_matrix.nrows() {
                continue;
            }
            let Some(mean) = matrix.mean_axis(ndarray::Axis(0)) else {
                continue;
            };
            by_relu.insert(relu_name.clone(), mean);
        }
        snapshots.push((!by_relu.is_empty()).then_some(CompleteClipMeanLowerA {
            spec_rows: spec_matrix.nrows(),
            by_relu,
        }));
    }
    Some(snapshots)
}

pub(crate) fn complete_clip_gpu_mean_refusal_reason(
    parents: &[&MultiObjectiveGraphBabDomain],
    spec_matrix: &Array2<f32>,
    deadline: Option<Instant>,
) -> Option<&'static str> {
    if !complete_clip_gpu_mean_parent_count_admitted(parents.len()) {
        return Some("reconstruct_domain_cap");
    }
    if parents.len() > COMPLETE_CLIP_GPU_MEAN_LA_MAX_DOMAINS {
        return Some("domain_cap");
    }
    if spec_matrix.nrows() == 0 {
        return Some("spec_rows_zero");
    }
    if spec_matrix.nrows() > COMPLETE_CLIP_GPU_MEAN_LA_MAX_SPECS {
        return Some("spec_rows_cap");
    }
    if !complete_clip_spec_matrix_is_row_major(spec_matrix) {
        return Some("spec_not_row_major");
    }
    // The original cascade used `?` here, so an overflow refused the wave exactly
    // as a cap breach did. Kept distinguishable rather than merged: an overflow is
    // a bug signal, a cap breach is a policy decision.
    match spec_matrix.nrows().checked_mul(spec_matrix.ncols()) {
        None => return Some("seed_values_overflow"),
        Some(values) if values > COMPLETE_CLIP_GPU_MEAN_LA_MAX_SEED_VALUES => {
            return Some("seed_values_cap");
        }
        Some(_) => {}
    }
    if !complete_clip_optional_start_budget_available(Instant::now(), deadline) {
        return Some("start_headroom");
    }
    if parents.iter().any(|parent| {
        parent
            .per_disjunct_alphas()
            .is_some_and(|alphas| !alphas.is_empty())
    }) {
        return Some("per_disjunct_alphas");
    }
    None
}

fn reduce_complete_clip_gpu_gather(
    gathered: &[f32],
    domain_count: usize,
    spec_rows: usize,
    gathered_indices: &[u32],
    full_width: usize,
    deadline: Option<Instant>,
) -> Option<Vec<Array1<f32>>> {
    let gathered_width = gathered_indices.len();
    let expected = domain_count
        .checked_mul(spec_rows)?
        .checked_mul(gathered_width)?;
    if spec_rows == 0
        || gathered.len() != expected
        || gathered_indices
            .iter()
            .any(|&index| index as usize >= full_width)
    {
        return None;
    }
    let mut reduced = Vec::new();
    reduced.try_reserve_exact(domain_count).ok()?;
    for domain in 0..domain_count {
        let mut mean = Array1::<f32>::zeros(full_width);
        for (position, &index) in gathered_indices.iter().enumerate() {
            if position.is_multiple_of(SELECTIVE_DEADLINE_POLL_STRIDE)
                && deadline.is_some_and(|value| Instant::now() >= value)
            {
                return None;
            }
            let mut sum = 0.0f32;
            for spec in 0..spec_rows {
                let row = domain.checked_mul(spec_rows)?.checked_add(spec)?;
                let offset = row.checked_mul(gathered_width)?.checked_add(position)?;
                let coefficient = *gathered.get(offset)?;
                if !coefficient.is_finite() {
                    return None;
                }
                sum += coefficient;
            }
            let value = sum / spec_rows as f32;
            if !value.is_finite() {
                return None;
            }
            mean[index as usize] = value;
        }
        reduced.push(mean);
    }
    Some(reduced)
}

/// Map the refusal recorded by `prep_resnet_domain_with` onto the distinct
/// `gather_prep_*` identifier reported for the per-parent prep exit.
///
/// Measured 2026-08-10 on relusplitter `oval21 base_kw`, that ONE exit is 664 of
/// 789 waves, so the flat `gather_prep_resnet_domain` label is the new dominant
/// bucket and splitting it is the next question, not a later one.
fn complete_clip_prep_refusal_label(reason: &str) -> &'static str {
    match reason {
        "prep_extract_segments" => "gather_prep_extract_segments",
        // #clip-gather-probe L3: the extraction's own exit labels, forwarded by
        // `prep_resnet_domain_with`. Measured 2026-08-10 on relusplitter `oval21
        // base_kw`, `gather_prep_extract_segments` was 613 of 616 waves that reached
        // the gather, so splitting THAT bucket is the next question.
        "extract_cycle_guard" => "gather_extract_cycle_guard",
        "extract_bn_at_input" => "gather_extract_bn_at_input",
        "extract_step_non_relu_alpha" => "gather_extract_step_non_relu_alpha",
        "extract_step_node_missing" => "gather_extract_step_node_missing",
        "extract_step_not_unary" => "gather_extract_step_not_unary",
        "extract_step_pre_unresolved" => "gather_extract_step_pre_unresolved",
        "extract_step_layer_unsupported" => "gather_extract_step_layer_unsupported",
        "extract_step_recorder" => "gather_extract_step_recorder",
        "extract_step_binary_not_add" => "gather_extract_step_binary_not_add",
        "extract_step_chain_frontier_abs" => "gather_extract_step_chain_frontier_abs",
        "extract_step_block_decompose" => "gather_extract_step_block_decompose",
        "extract_step_block_frontier_abs" => "gather_extract_step_block_frontier_abs",
        "extract_step_arity" => "gather_extract_step_arity",
        "extract_step_unclassified" => "gather_extract_step_unclassified",
        "extract_final_frontier_abs" => "gather_extract_final_frontier_abs",
        "extract_segments_empty" => "gather_extract_segments_empty",
        "extract_pure_chain_disallowed" => "gather_extract_pure_chain_disallowed",
        "extract_not_recorded" => "gather_extract_not_recorded",
        "prep_relu_bounds_missing" => "gather_prep_relu_bounds_missing",
        "prep_stop_node_is_input" => "gather_prep_stop_node_is_input",
        "prep_stop_bounds_missing" => "gather_prep_stop_bounds_missing",
        "prep_stop_box_invalid" => "gather_prep_stop_box_invalid",
        _ => "gather_prep_resnet_domain",
    }
}

/// Map an admission reason from [`complete_clip_gpu_mean_refusal_reason`] onto
/// the distinct `gather_recheck_*` identifier reported when the *internal*
/// re-check refuses a wave the caller had already classified as admitted.
///
/// The caller classifies admission before the call, so reaching one of these
/// means the two evaluations disagreed — almost always the time-sensitive
/// `start_headroom` predicate flipping between them. Keeping the sub-reason
/// makes that hypothesis falsifiable in one run instead of assumed.
fn complete_clip_gpu_mean_recheck_reason(reason: &str) -> &'static str {
    match reason {
        "reconstruct_domain_cap" => "gather_recheck_reconstruct_domain_cap",
        "domain_cap" => "gather_recheck_domain_cap",
        "spec_rows_zero" => "gather_recheck_spec_rows_zero",
        "spec_rows_cap" => "gather_recheck_spec_rows_cap",
        "spec_not_row_major" => "gather_recheck_spec_not_row_major",
        "seed_values_overflow" => "gather_recheck_seed_values_overflow",
        "seed_values_cap" => "gather_recheck_seed_values_cap",
        "start_headroom" => "gather_recheck_start_headroom",
        "per_disjunct_alphas" => "gather_recheck_per_disjunct_alphas",
        _ => "gather_recheck_other",
    }
}

/// Recover the exact full-spec DomainClipper signal from one wave-wide,
/// parent-only GPU backward, reporting WHICH early exit refused rather than
/// merely THAT one did.
///
/// The gather point is the resident backend's pre-ReLU lower-A buffer, the same
/// point represented by `CachedLinearBounds::lower_a[relu]`. Every root-unstable
/// neuron is requested, all objective rows (including already verified rows)
/// remain in the seed, and the result is reduced immediately to the post-mean
/// vector consumed by DomainClipper. This is advisory-only: any missing backend,
/// structural mismatch, resource-cap refusal, or deadline expiry returns `Err`,
/// preserving the historical local selector.
///
/// Every `Err` return of the historical function collapsed into a single
/// `gather_failed` bucket at the call site. Measured 2026-08-10 on relusplitter
/// `oval21 base_kw`, that bucket absorbed 710 of 878 waves — every one of them at
/// `wall_ms=0`, i.e. failing on an early `?` before any measurable work — with no
/// way to tell the output-node resolution from the deadline poll from the backend
/// lookup. One identifier per exit turns that into a single named cause.
///
/// This is diagnostic only. The control flow, the order of every predicate, and
/// the returned snapshots are identical to the historical function; the `Err`
/// payload is a label on a path that already returned `None`. The identifiers are
/// consumed by the `[complete-clip-decision-precompute]` probe line; keep them
/// snake_case, `gather_`-prefixed (so they stay greppable as "inside the gather",
/// where the flat `gather_failed` used to be), and never reuse one for two exits.
#[allow(clippy::too_many_lines)]
pub(crate) fn complete_clip_mean_las_from_gpu_probed(
    graph: &GraphNetwork,
    root_bounds: &HashMap<String, Arc<BoundedTensor>>,
    parents: &[&MultiObjectiveGraphBabDomain],
    spec_matrix: &Array2<f32>,
    engine: &dyn GemmEngine,
    deadline: Option<Instant>,
    // #clip-chain-gather: OR'd into the extraction's `allow_pure_chain`. The
    // production caller passes [`complete_clip_chain_gather_enabled`]; tests pass
    // the boolean directly so the routing under test is the gate's own without
    // mutating process-global env (racy under the parallel harness — the same
    // `graft_pure_chain` convention the chain-wide oracle uses). `false` is
    // byte-identical to the historical `bab_chain_wide_enabled()`-only argument.
    allow_pure_chain_gather: bool,
) -> Result<Vec<Option<CompleteClipMeanLowerA>>, &'static str> {
    let deadline_live = || deadline.is_none_or(|value| Instant::now() < value);
    let deadline_has_start_headroom =
        || complete_clip_optional_start_budget_available(Instant::now(), deadline);
    if let Some(reason) = complete_clip_gpu_mean_refusal_reason(parents, spec_matrix, deadline) {
        return Err(complete_clip_gpu_mean_recheck_reason(reason));
    }
    let output_node = if graph.output_name().is_empty() {
        let exec_order = graph
            .exec_order()
            .map_err(|_| "gather_exec_order_unavailable")?;
        exec_order.last().ok_or("gather_exec_order_empty")?.clone()
    } else {
        graph.output_name().to_string()
    };
    let allow_pure_chain = crate::network::bab_chain_wide_enabled() || allow_pure_chain_gather;
    let mut preps = Vec::new();
    preps
        .try_reserve_exact(parents.len())
        .map_err(|_| "gather_preps_reserve")?;
    for parent in parents {
        if !deadline_live() {
            return Err("gather_prep_deadline");
        }
        let parent_node_bounds = parent.node_bounds().to_shared_hash_map();
        preps.push(
            prep_resnet_domain_ext(
                graph,
                &output_node,
                &parent_node_bounds,
                parent.input_bounds(),
                Some(parent.beta_state()),
                Some(parent.alpha_state()),
                allow_pure_chain,
                false,
                false,
            )
            .ok_or_else(|| {
                complete_clip_prep_refusal_label(
                    crate::beta_crown::engine::graph::propagation::batched::prep_resnet_domain_last_refusal(),
                )
            })?,
        );
    }
    let first = preps.first().ok_or("gather_preps_empty")?;
    if first.relu_names.is_empty()
        || first.relu_names.len() != first.beta_signed.len()
        || preps.iter().skip(1).any(|prep| {
            prep.relu_names != first.relu_names
                || prep.beta_signed.len() != first.beta_signed.len()
                || prep
                    .beta_signed
                    .iter()
                    .zip(&first.beta_signed)
                    .any(|(actual, reference)| actual.len() != reference.len())
        })
    {
        return Err("gather_relu_shape_mismatch");
    }
    let full_relu_width = first
        .beta_signed
        .iter()
        .try_fold(0usize, |sum, values| sum.checked_add(values.len()))
        .ok_or("gather_full_relu_width_overflow")?;
    let max_relu_width = first.beta_signed.iter().map(Vec::len).max().unwrap_or(0);
    if parents
        .len()
        .checked_mul(full_relu_width)
        .ok_or("gather_host_mean_values_overflow")?
        > COMPLETE_CLIP_GPU_MEAN_LA_MAX_HOST_MEAN_VALUES
    {
        return Err("gather_host_mean_values_cap");
    }

    // The candidate mask is fixed at root, matching DomainClipper's
    // `unstable_idx` contract. The same union applies to every parent in the
    // wave and is downloaded only once.
    let mut gather_indices = Vec::<Vec<u32>>::new();
    gather_indices
        .try_reserve_exact(first.relu_names.len())
        .map_err(|_| "gather_indices_reserve")?;
    let mut gathered_columns = 0usize;
    for (relu_name, beta_signed) in first.relu_names.iter().zip(&first.beta_signed) {
        let relu = graph
            .nodes
            .get(relu_name)
            .ok_or("gather_relu_node_missing")?;
        let seed_node = relu
            .require_unary_input()
            .map_err(|_| "gather_relu_not_unary")?;
        if seed_node == NETWORK_INPUT {
            gather_indices.push(Vec::new());
            continue;
        }
        let root_entry = root_bounds
            .get(seed_node)
            .ok_or("gather_root_bounds_missing")?;
        if root_entry.len() != beta_signed.len() {
            return Err("gather_root_width_mismatch");
        }
        let mut indices = Vec::new();
        for (neuron, (&lower, &upper)) in root_entry
            .lower()
            .iter()
            .zip(root_entry.upper().iter())
            .enumerate()
        {
            if lower < 0.0 && upper > 0.0 {
                indices.push(u32::try_from(neuron).map_err(|_| "gather_neuron_index_overflow")?);
            }
        }
        gathered_columns = gathered_columns
            .checked_add(indices.len())
            .ok_or("gather_columns_overflow")?;
        gather_indices.push(indices);
    }
    if gathered_columns == 0 {
        return Err("gather_no_root_unstable_columns");
    }
    if !deadline_live() {
        return Err("gather_setup_deadline");
    }

    // Prefer an explicitly registered wide backend when requested. The local
    // sound WGPU backend is also admissible: its large gather path is one
    // strided compute dispatch plus one contiguous readback per ReLU, while
    // established small beta gathers retain the byte-identical copy path.
    let gpu_candidates = complete_clip_gpu_candidates(engine, deadline);
    if gpu_candidates.is_empty() {
        return Err("gather_no_sound_gpu_backend");
    }
    let seed_rows = spec_matrix
        .as_slice_memory_order()
        .ok_or("gather_seed_not_contiguous")?;
    let seed = ny_core::GpuCrownSeed {
        lower_a: seed_rows.to_vec().into(),
        upper_a: seed_rows.to_vec().into(),
        lower_b: vec![0.0f32; spec_matrix.nrows()].into(),
        upper_b: vec![0.0f32; spec_matrix.nrows()].into(),
        num_specs: spec_matrix.nrows(),
        current_dim: spec_matrix.ncols(),
    };
    let gather_refs: Vec<&[u32]> = gather_indices.iter().map(Vec::as_slice).collect();
    // Bound every single GPU dispatch, then stitch successful deterministic
    // parent slices. A late deadline/backend refusal leaves earlier snapshots
    // usable and missing parents on the historical selector.
    let per_domain_gather = spec_matrix
        .nrows()
        .checked_mul(gathered_columns)
        .ok_or("gather_per_domain_gather_overflow")?;
    let per_domain_coeff = spec_matrix
        .nrows()
        .checked_mul(max_relu_width)
        .ok_or("gather_per_domain_coeff_overflow")?;
    let chunk_domains = [
        COMPLETE_CLIP_GPU_MEAN_LA_MAX_DOMAINS,
        COMPLETE_CLIP_GPU_MEAN_LA_MAX_WIDE_ROWS / spec_matrix.nrows(),
        COMPLETE_CLIP_GPU_MEAN_LA_MAX_GATHER_VALUES / per_domain_gather.max(1),
        COMPLETE_CLIP_GPU_MEAN_LA_MAX_WIDE_COEFFICIENTS / per_domain_coeff.max(1),
        COMPLETE_CLIP_GPU_MEAN_LA_MAX_HOST_MEAN_VALUES / full_relu_width.max(1),
        8,
    ]
    .into_iter()
    .min()
    .ok_or("gather_chunk_domains_unbounded")?;
    if chunk_domains == 0 {
        return Err("gather_chunk_domains_zero");
    }
    // WHY the chunk loop stopped, carried to the single `all(is_none)` exit
    // below. Each `break` already discarded its cause; naming it costs nothing
    // and is the difference between "the gather failed" and a located defect.
    let mut break_reason = "gather_no_chunks";
    let mut snapshots: Vec<Option<CompleteClipMeanLowerA>> =
        (0..parents.len()).map(|_| None).collect();
    for chunk_start in (0..preps.len()).step_by(chunk_domains) {
        if !deadline_has_start_headroom() {
            break_reason = "gather_chunk_start_headroom";
            break;
        }
        let chunk_end = (chunk_start + chunk_domains).min(preps.len());
        let chunk = &preps[chunk_start..chunk_end];
        let domains: Vec<ny_core::GpuResnetBatchedDomainRef<'_>> = chunk
            .iter()
            .map(|prep| ny_core::GpuResnetBatchedDomainRef {
                segments: &prep.segments,
                input_lower: &prep.in_lo,
                input_upper: &prep.in_hi,
                beta_signed: &prep.beta_signed,
                frontier_abs: &prep.frontier_abs,
                node_abs: &prep.node_abs,
            })
            .collect();
        let mut dispatch = None;
        let mut dispatch_reason = "gather_dispatch_backend_refused";
        for gpu in &gpu_candidates {
            // A failed preferred backend may consume most of the optional-pass
            // budget. Re-apply the admission margin before every retry, and
            // scope the deadline only over the exact backend being called.
            // This avoids cross-talk between the global and engine-local
            // candidates while retaining the sound local fallback.
            if !deadline_has_start_headroom() {
                dispatch_reason = "gather_dispatch_retry_headroom";
                break;
            }
            let _gpu_deadline_scope =
                crate::sound_gpu_gate::GpuCrownBackendDeadlineScope::set(*gpu, deadline);
            if let Ok(result) = gpu.crown_backward_gpu_resnet_sound_beta_batched_grad(
                &domains,
                &seed,
                &gather_refs,
                &[],
            ) {
                dispatch = Some(result);
                break;
            }
        }
        let Some((bounds, _alpha_grads, gathered)) = dispatch else {
            break_reason = dispatch_reason;
            break;
        };
        if !deadline_live() {
            break_reason = "gather_dispatch_deadline";
            break;
        }
        if bounds.len() != chunk.len()
            || bounds.iter().any(|result| {
                result.lower_bounds.len() != spec_matrix.nrows()
                    || result.upper_bounds.len() != spec_matrix.nrows()
            })
        {
            break_reason = "gather_bounds_shape_mismatch";
            break;
        }
        if gathered.len() != first.relu_names.len() {
            break_reason = "gather_relu_count_mismatch";
            break;
        }
        let mut chunk_snapshots: Vec<CompleteClipMeanLowerA> = (0..chunk.len())
            .map(|_| CompleteClipMeanLowerA {
                spec_rows: spec_matrix.nrows(),
                by_relu: HashMap::with_capacity(first.relu_names.len()),
            })
            .collect();
        let mut chunk_valid = true;
        for (((relu_name, indices), values), beta_signed) in first
            .relu_names
            .iter()
            .zip(&gather_indices)
            .zip(&gathered)
            .zip(&first.beta_signed)
        {
            if indices.is_empty() {
                continue;
            }
            let Some(means) = reduce_complete_clip_gpu_gather(
                values,
                chunk.len(),
                spec_matrix.nrows(),
                indices,
                beta_signed.len(),
                deadline,
            ) else {
                chunk_valid = false;
                break;
            };
            for (snapshot, mean) in chunk_snapshots.iter_mut().zip(means) {
                snapshot.by_relu.insert(relu_name.clone(), mean);
            }
        }
        if !chunk_valid {
            break_reason = "gather_reduce_failed";
            break;
        }
        if chunk_snapshots
            .iter()
            .any(|snapshot| snapshot.by_relu.is_empty())
        {
            break_reason = "gather_empty_snapshot";
            break;
        }
        for (slot, snapshot) in snapshots[chunk_start..chunk_end]
            .iter_mut()
            .zip(chunk_snapshots)
        {
            *slot = Some(snapshot);
        }
    }
    if snapshots.iter().all(Option::is_none) {
        return Err(break_reason);
    }
    Ok(snapshots)
}

/// Convert the parent pass's per-objective cached lAs plus one prospective
/// final leaf's split-clamped bounds into compact DomainClipper decisions.
///
/// This matches αβ-CROWN's `decision_precompute`: depth expansion repeats the
/// just-finished parent lAs; it does not run CROWN on each prospective child.
pub(crate) fn complete_clip_decisions_from_cached_las<'a, 'b>(
    graph: &GraphNetwork,
    root_bounds: impl Into<NodeBoundsView<'a>>,
    child_bounds: impl Into<NodeBoundsView<'b>>,
    child_history: &GraphSplitHistory,
    all_cached_las: &[&CachedLinearBounds],
    topk: usize,
    deadline: Option<Instant>,
) -> Option<HashMap<String, Vec<usize>>> {
    let root_bounds = root_bounds.into();
    let child_bounds = child_bounds.into();
    if all_cached_las.is_empty() || deadline.is_some_and(|value| Instant::now() >= value) {
        return None;
    }
    let mut decisions = HashMap::new();
    for (position, relu_name) in graph.exec_order().ok()?.iter().enumerate() {
        if position.is_multiple_of(SELECTIVE_DEADLINE_POLL_STRIDE)
            && deadline.is_some_and(|value| Instant::now() >= value)
        {
            return None;
        }
        let Some(relu) = graph.nodes.get(relu_name) else {
            continue;
        };
        if !matches!(relu.layer, Layer::ReLU(_)) {
            continue;
        }
        let Ok(seed_node) = relu.require_unary_input() else {
            continue;
        };
        if seed_node == NETWORK_INPUT {
            continue;
        }
        let root_entry = root_bounds.get(seed_node)?;
        let child_entry = child_bounds.get(seed_node)?;
        let has_root_candidate = root_entry
            .lower()
            .iter()
            .zip(root_entry.upper().iter())
            .any(|(&lower, &upper)| lower < 0.0 && upper > 0.0);
        if !has_root_candidate {
            continue;
        }
        // A cache is per objective, so each matrix has one row. At the ReLU
        // key that row is DomainClipper's A_key for this layer.
        let lower_as: Option<Vec<&Array2<f32>>> = all_cached_las
            .iter()
            .map(|cache| cache.lower_a.get(relu_name))
            .collect();
        // Inherited caches may have different per-layer coverage. A decision
        // is valid only when this layer has every objective row, but one
        // uncovered layer must not erase fully-covered layers in the same
        // prospective child.
        let Some(lower_as) = lower_as else {
            continue;
        };
        let selected = select_complete_clip_rows_from_cached_las(
            root_entry,
            child_entry,
            child_history,
            relu_name,
            &lower_as,
            topk,
            deadline,
        )?;
        if !selected.is_empty() {
            decisions.insert(relu_name.clone(), selected);
        }
    }
    if deadline.is_some_and(|value| Instant::now() >= value) {
        return None;
    }
    (!decisions.is_empty()).then_some(decisions)
}

/// Compact-mean sibling of [`complete_clip_decisions_from_cached_las`], used
/// when the image-DAG fast path produced concrete bounds without host lA
/// caches. All prospective children of a parent share one snapshot.
pub(crate) fn complete_clip_decisions_from_mean_las<'a, 'b>(
    graph: &GraphNetwork,
    root_bounds: impl Into<NodeBoundsView<'a>>,
    child_bounds: impl Into<NodeBoundsView<'b>>,
    child_history: &GraphSplitHistory,
    mean_las: &CompleteClipMeanLowerA,
    topk: usize,
    deadline: Option<Instant>,
) -> Option<HashMap<String, Vec<usize>>> {
    let root_bounds = root_bounds.into();
    let child_bounds = child_bounds.into();
    if mean_las.spec_rows == 0 || deadline.is_some_and(|value| Instant::now() >= value) {
        return None;
    }
    let mut decisions = HashMap::new();
    for (position, relu_name) in graph.exec_order().ok()?.iter().enumerate() {
        if position.is_multiple_of(SELECTIVE_DEADLINE_POLL_STRIDE)
            && deadline.is_some_and(|value| Instant::now() >= value)
        {
            return None;
        }
        let Some(relu) = graph.nodes.get(relu_name) else {
            continue;
        };
        if !matches!(relu.layer, Layer::ReLU(_)) {
            continue;
        }
        let Ok(seed_node) = relu.require_unary_input() else {
            continue;
        };
        if seed_node == NETWORK_INPUT {
            continue;
        }
        let root_entry = root_bounds.get(seed_node)?;
        let child_entry = child_bounds.get(seed_node)?;
        if !root_entry
            .lower()
            .iter()
            .zip(root_entry.upper().iter())
            .any(|(&lower, &upper)| lower < 0.0 && upper > 0.0)
        {
            continue;
        }
        let Some(mean_lower_a) = mean_las.by_relu.get(relu_name) else {
            continue;
        };
        let selected = select_complete_clip_rows_from_mean_la(
            root_entry,
            child_entry,
            child_history,
            relu_name,
            mean_lower_a,
            topk,
            deadline,
        )?;
        if !selected.is_empty() {
            decisions.insert(relu_name.clone(), selected);
        }
    }
    if deadline.is_some_and(|value| Instant::now() >= value) {
        return None;
    }
    (!decisions.is_empty()).then_some(decisions)
}

impl BetaCrownVerifier {
    pub(crate) fn complete_clip_root_bounds_for_decision_precompute(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        deadline: Option<Instant>,
    ) -> Option<Arc<HashMap<String, Arc<BoundedTensor>>>> {
        self.complete_clip_root_bounds_cache
            .get_finalized_for_batch(graph, std::slice::from_ref(input), deadline)
            .map(|(bounds, _)| bounds)
    }

    pub(crate) fn publish_complete_clip_decisions(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        history: &GraphSplitHistory,
        decisions: HashMap<String, Vec<usize>>,
    ) -> bool {
        let mut compact = HashMap::new();
        for (layer, mut neurons) in decisions {
            neurons.sort_unstable();
            neurons.dedup();
            if neurons.is_empty() {
                continue;
            }
            compact.insert(layer, Arc::from(neurons.into_boxed_slice()));
        }
        self.complete_clip_root_bounds_cache
            .store_decisions(graph, input, history, compact)
    }
}

/// Winner-style objective selection is per child and per layer.  The GPU seed
/// remains batch-shared, so this returns the sorted union to capture once plus
/// each child's own objective-neuron list.  Split-premise rows are added to the
/// capture union as constraint sources but do not consume the child's top-K
/// objective budget.
#[allow(clippy::too_many_arguments)]
fn select_complete_clip_rows(
    graph: &GraphNetwork,
    caches: &[HashMap<String, Arc<BoundedTensor>>],
    histories: &[&GraphSplitHistory],
    seed_node: &str,
    relu_name: &str,
    candidates: &[usize],
    pre_dim: usize,
    topk: usize,
    margin_weights: Option<&[f32]>,
    precomputed_decisions: Option<&[Option<Arc<CompleteClipDecisionIndices>>]>,
    deadline: Option<Instant>,
    use_all_history: bool,
) -> Option<(Vec<usize>, Vec<Vec<usize>>, Vec<bool>)> {
    if caches.len() != histories.len()
        || precomputed_decisions.is_some_and(|items| items.len() != caches.len())
        || pre_dim == 0
    {
        return None;
    }
    let mut union = vec![false; pre_dim];
    let mut premise_union = vec![false; pre_dim];
    let mut per_domain = Vec::new();
    per_domain.try_reserve_exact(caches.len()).ok()?;

    for (domain_index, (cache, history)) in caches.iter().zip(histories).enumerate() {
        if deadline.is_some_and(|d| Instant::now() >= d) {
            return None;
        }
        let entry = cache.get(seed_node)?;
        let (lower, upper) = (
            entry.lower().as_slice_memory_order()?,
            entry.upper().as_slice_memory_order()?,
        );
        if lower.len() != pre_dim || upper.len() != pre_dim {
            return None;
        }

        let mut domain_candidates = Vec::new();
        domain_candidates.try_reserve_exact(candidates.len()).ok()?;
        for (position, &neuron) in candidates.iter().enumerate() {
            if position.is_multiple_of(SELECTIVE_DEADLINE_POLL_STRIDE)
                && deadline.is_some_and(|d| Instant::now() >= d)
            {
                return None;
            }
            if neuron >= pre_dim {
                return None;
            }
            if topk == 0
                || (lower[neuron].is_finite()
                    && upper[neuron].is_finite()
                    && lower[neuron] < 0.0
                    && upper[neuron] > 0.0)
            {
                domain_candidates.push(neuron);
            }
        }
        let precomputed = precomputed_decisions
            .and_then(|items| items.get(domain_index))
            .and_then(Option::as_ref)
            .and_then(|snapshot| snapshot.get(relu_name));
        let mut objectives = if let Some(indices) = precomputed {
            let objectives = indices.to_vec();
            if objectives.iter().any(|&neuron| neuron >= pre_dim)
                || objectives.windows(2).any(|pair| pair[0] >= pair[1])
            {
                return None;
            }
            objectives
        } else {
            let no_premises = vec![false; pre_dim];
            let objectives = select_intermediate_objective_rows_with_deadline(
                std::slice::from_ref(cache),
                seed_node,
                &domain_candidates,
                &no_premises,
                topk,
                margin_weights,
                deadline,
            );
            if topk == 0 && objectives.is_empty() && !domain_candidates.is_empty() {
                return None;
            }
            objectives
        };
        for &neuron in &objectives {
            union[neuron] = true;
        }

        // Exact histories, rather than reconstructed β entries, own the split
        // source selection.  A source row may be inherited-stable because the
        // child premise already clamped it; it must still be captured.
        for (constraint_index, constraint) in
            scheduled_relu_constraints(graph, history, use_all_history)?
                .into_iter()
                .enumerate()
        {
            if constraint_index.is_multiple_of(SELECTIVE_DEADLINE_POLL_STRIDE)
                && deadline.is_some_and(|d| Instant::now() >= d)
            {
                return None;
            }
            if constraint.node_name() == relu_name {
                let neuron = constraint.neuron_idx();
                if neuron >= pre_dim {
                    return None;
                }
                union[neuron] = true;
                premise_union[neuron] = true;
            }
        }
        // The selector already index-sorts, but pin the invariant explicitly
        // for the row-map/provenance identity.
        objectives.sort_unstable();
        objectives.dedup();
        per_domain.push(objectives);
    }

    let selected: Vec<usize> = union
        .into_iter()
        .enumerate()
        .filter_map(|(neuron, keep)| keep.then_some(neuron))
        .collect();
    (!selected.is_empty()).then_some((selected, per_domain, premise_union))
}

/// Find the NEXT ReLU strictly BELOW `node` for the multi-layer cascade: among
/// all ancestor ReLUs (reachable through `inputs`, residual branches included),
/// the one LATEST in execution order — on resnet_medium from `Gemm_56` that is
/// `Relu_51` (F-branch of the last residual block). Returns
/// `(relu_name, seed_node)` where `seed_node` is that ReLU's input; `None` when
/// no ancestor ReLU exists, exec order is unavailable, or the ReLU is fed by
/// the network input (nothing below the seed to refine with).
pub(in crate::beta_crown::engine::graph) fn find_next_relu_seed_below(
    graph: &GraphNetwork,
    node: &str,
) -> Option<(String, String)> {
    let start = graph.nodes.get(node)?;
    let mut stack: Vec<String> = start.inputs.clone();
    let mut seen: std::collections::HashSet<String> = stack.iter().cloned().collect();
    let mut relu_ancestors: Vec<String> = Vec::new();
    while let Some(current) = stack.pop() {
        if current == NETWORK_INPUT {
            continue;
        }
        let Some(n) = graph.nodes.get(&current) else {
            continue;
        };
        if matches!(n.layer, Layer::ReLU(_)) {
            relu_ancestors.push(current.clone());
        }
        for inp in &n.inputs {
            if seen.insert(inp.clone()) {
                stack.push(inp.clone());
            }
        }
    }
    let exec = graph.exec_order().ok()?;
    let pos: HashMap<&str, usize> = exec
        .iter()
        .enumerate()
        .map(|(i, name)| (name.as_str(), i))
        .collect();
    let relu = relu_ancestors
        .into_iter()
        .filter(|name| pos.contains_key(name.as_str()))
        .max_by_key(|name| pos[name.as_str()])?;
    let seed = graph
        .nodes
        .get(&relu)?
        .require_unary_input()
        .ok()?
        .to_string();
    if seed == NETWORK_INPUT {
        return None;
    }
    Some((relu, seed))
}

/// C3-at-Relu_57 LEDGER PROBE gate (dark, `NY_C3_57_PROBE=1`, stderr-only,
/// read-only — `docs/CERTIFIED_CUT_CROWN_DESIGN.md` §C3-at-split-layer). The
/// probe dumps, per deep subdomain of the wide lane, exactly the raw data the
/// first-order cut ledger needs: the split premises at the last ReLU, the
/// REFINED pre-activation intervals of its post-refinement-unstable neurons,
/// and the per-spec-row optimized lower bounds.
pub(in crate::beta_crown::engine::graph) fn c3_57_probe_enabled() -> bool {
    matches!(std::env::var("NY_C3_57_PROBE").ok().as_deref(), Some("1"))
}

/// Minimum premise count (= BaB depth for ReLU-split BaB) for a domain to be
/// dumped (`NY_C3_57_PROBE_DEPTH`, default 8 — the prop885 frontier band).
fn c3_57_probe_min_depth() -> usize {
    std::env::var("NY_C3_57_PROBE_DEPTH")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(8)
}

/// Dump the C3-at-Relu_57 ledger data for one wide-lane batch (see
/// [`c3_57_probe_enabled`]). `bounds_caches` must be the REFINED caches (the
/// same ones the margin backward consumed); `bounds` the per-domain per-spec
/// optimized output bounds. One `dom` line per deep-enough domain: premises
/// (`idx+`/`idx-`), per-row lower bounds, and `idx:l:u` for every unstable
/// neuron of the last ReLU's pre-activation entry. A `spec_rows` line
/// (`pos:neg` per row) is emitted once per process.
pub(in crate::beta_crown::engine::graph) fn c3_57_probe_dump(
    graph: &GraphNetwork,
    output_node: &str,
    bounds_caches: &[HashMap<String, Arc<BoundedTensor>>],
    beta_states: &[Option<&GraphBetaState>],
    spec_matrix: &Array2<f32>,
    bounds: &[BoundedTensor],
) {
    let Some((relu_name, seed_node)) = find_last_relu_seed(graph, output_node) else {
        return;
    };
    let min_depth = c3_57_probe_min_depth();
    static SPEC_ONCE: std::sync::Once = std::sync::Once::new();
    SPEC_ONCE.call_once(|| {
        let rows: Vec<String> = spec_matrix
            .rows()
            .into_iter()
            .map(|r| {
                let pos = r.iter().position(|&v| v > 0.5).map_or(-1, |p| p as i64);
                let neg = r.iter().position(|&v| v < -0.5).map_or(-1, |n| n as i64);
                format!("{pos}:{neg}")
            })
            .collect();
        eprintln!(
            "[c3-57] spec_rows relu={relu_name} seed={seed_node} rows={}",
            rows.join(",")
        );
    });
    // Per-batch summary (diagnosis aid): the per-domain premise count at the
    // last ReLU as seen through the beta states this backward consumed, plus
    // domain 0's full split-layer histogram (the LA brancher may split OFF the
    // last ReLU — unlike the pre-LA measurement's 100%-Relu_57).
    let counts: Vec<String> = (0..bounds_caches.len())
        .map(|i| {
            beta_states.get(i).copied().flatten().map_or_else(
                || "-".to_string(),
                |bs| bs.entries_for_node(&relu_name).count().to_string(),
            )
        })
        .collect();
    let hist0: String = beta_states
        .first()
        .copied()
        .flatten()
        .map(|bs| {
            let mut h: std::collections::BTreeMap<&str, usize> = Default::default();
            for e in &bs.entries {
                *h.entry(e.node_name()).or_default() += 1;
            }
            h.iter()
                .map(|(n, c)| format!("{n}={c}"))
                .collect::<Vec<_>>()
                .join(" ")
        })
        .unwrap_or_default();
    eprintln!(
        "[c3-57] batch n_domains={} prem_counts=[{}] hist0: {hist0}",
        bounds_caches.len(),
        counts.join(",")
    );
    // Suffix-MIP dump: the spec matrix NARROWS across batches (verified rows
    // drop out), so the once-per-process `spec_rows` line cannot attribute a
    // later batch's `lbs` — emit the batch's own rows in the sfx stream.
    if suffix_mip_dump_enabled() {
        let rows: Vec<String> = spec_matrix
            .rows()
            .into_iter()
            .map(|r| {
                let pos = r.iter().position(|&v| v > 0.5).map_or(-1, |p| p as i64);
                let neg = r.iter().position(|&v| v < -0.5).map_or(-1, |n| n as i64);
                format!("{pos}:{neg}")
            })
            .collect();
        eprintln!("[sfx] spec rows={}", rows.join(","));
    }
    for (i, cache) in bounds_caches.iter().enumerate() {
        let Some(bs) = beta_states.get(i).copied().flatten() else {
            continue;
        };
        let prem: Vec<String> = bs
            .entries_for_node(&relu_name)
            .map(|e| {
                format!(
                    "{}{}",
                    e.neuron_idx(),
                    if e.sign() > 0.0 { "+" } else { "-" }
                )
            })
            .collect();
        // Depth gate on TOTAL split count (the BaB depth): under the LA
        // brancher only the first ~3 splits sit on the last ReLU, so gating
        // on `prem.len()` would never fire at the deep frontier.
        if bs.entries.len() < min_depth {
            continue;
        }
        let Some(entry) = cache.get(&seed_node) else {
            continue;
        };
        let unstable: Vec<String> = entry
            .lower()
            .iter()
            .zip(entry.upper().iter())
            .enumerate()
            .filter(|(_, (&l, &u))| l < 0.0 && u > 0.0)
            .map(|(j, (&l, &u))| format!("{j}:{l:.6}:{u:.6}"))
            .collect();
        let lbs: Vec<String> = bounds
            .get(i)
            .map(|b| b.lower().iter().map(|v| format!("{v:.5}")).collect())
            .unwrap_or_default();
        eprintln!(
            "[c3-57] dom={i} depth={} prem=[{}] lbs=[{}] unstable=[{}]",
            prem.len(),
            prem.join(","),
            lbs.join(","),
            unstable.join(","),
        );
        if suffix_mip_dump_enabled() {
            suffix_mip_dump_domain(graph, &relu_name, &seed_node, cache, bs, i, bounds.get(i));
        }
    }
}

/// Suffix-MIP dump extension (dark, `NY_SUFFIX_MIP_DUMP=1`, stderr-only,
/// read-only; rides the `NY_C3_57_PROBE=1` site — both envs must be set). Dumps
/// what the OFFLINE exact-suffix gate (docs/CERTIFIED_CUT_CROWN_DESIGN.md,
/// suffix-MIP escalation hypothesis) needs beyond the ledger probe:
/// * `[sfx] meta` — the last ReLU, its seed node (the GEMM producing its
///   pre-activation) and the seed's own input node (once per process);
/// * `[sfx] inbox` — the per-domain cache enclosure at the seed's INPUT node
///   (the z-box of the suffix MIP), hash-deduped: MEASURED ~14 distinct boxes
///   per 300s run (premise-conditioned per-domain recomputation, coords ≤0.05
///   apart), so each distinct box prints once and doms reference it by id;
/// * `[sfx] dom` — per deep domain: premises at the last ReLU with split
///   points, per-spec-row optimized lower bounds, and the FULL refined seed
///   `[l',u']` (all dims — the exact suffix min needs stable neurons too).
///
/// Floats print with Rust's shortest-round-trip `Display` (bit-exact reload).
fn suffix_mip_dump_enabled() -> bool {
    matches!(
        std::env::var("NY_SUFFIX_MIP_DUMP").ok().as_deref(),
        Some("1")
    )
}

fn suffix_mip_dump_domain(
    graph: &GraphNetwork,
    relu_name: &str,
    seed_node: &str,
    cache: &HashMap<String, Arc<BoundedTensor>>,
    bs: &GraphBetaState,
    dom_idx: usize,
    bounds: Option<&BoundedTensor>,
) {
    static META_ONCE: std::sync::Once = std::sync::Once::new();
    let seed_inputs: Vec<String> = graph
        .nodes
        .get(seed_node)
        .map(|n| n.inputs.clone())
        .unwrap_or_default();
    META_ONCE.call_once(|| {
        eprintln!(
            "[sfx] meta relu={relu_name} seed={seed_node} seed_inputs=[{}]",
            seed_inputs.join(",")
        );
    });
    // The z-box: enclosure at the seed GEMM's (first) input node, deduped by
    // content hash so the ~2048-wide line prints once per distinct box.
    let inbox_id: i64 = seed_inputs
        .first()
        .and_then(|inp| cache.get(inp).map(|bt| (inp, bt)))
        .map_or(-1, |(inp, bt)| {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            for v in bt.lower().iter().chain(bt.upper().iter()) {
                v.to_bits().hash(&mut h);
            }
            let key = h.finish();
            static SEEN: Mutex<Vec<u64>> = Mutex::new(Vec::new());
            let mut seen = SEEN.lock().unwrap();
            let id = seen.iter().position(|&k| k == key).unwrap_or_else(|| {
                seen.push(key);
                let vals: Vec<String> = bt
                    .lower()
                    .iter()
                    .zip(bt.upper().iter())
                    .map(|(l, u)| format!("{l}:{u}"))
                    .collect();
                eprintln!(
                    "[sfx] inbox id={} node={inp} n={} vals=[{}]",
                    seen.len() - 1,
                    vals.len(),
                    vals.join(",")
                );
                seen.len() - 1
            });
            id as i64
        });
    // Upstream-window extension (dark, additive): `NY_SUFFIX_MIP_DUMP_NODES` =
    // comma-separated cache node names whose per-domain enclosures the offline
    // 2-block window gate needs (e.g. `Add_48,BatchNormalization_50`). Also
    // prints the cache's available keys once so the offline side can discover
    // what is materialized.
    static KEYS_ONCE: std::sync::Once = std::sync::Once::new();
    KEYS_ONCE.call_once(|| {
        let mut keys: Vec<String> = cache
            .iter()
            .map(|(k, bt)| format!("{k}:{}", bt.lower().len()))
            .collect();
        keys.sort();
        eprintln!("[sfx] keys=[{}]", keys.join(","));
    });
    if let Ok(extra) = std::env::var("NY_SUFFIX_MIP_DUMP_NODES") {
        for name in extra.split(',').filter(|s| !s.is_empty()) {
            let Some(bt) = cache.get(name) else {
                continue;
            };
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            name.hash(&mut h);
            for v in bt.lower().iter().chain(bt.upper().iter()) {
                v.to_bits().hash(&mut h);
            }
            let key = h.finish();
            static XSEEN: Mutex<Vec<u64>> = Mutex::new(Vec::new());
            let mut seen = XSEEN.lock().unwrap();
            let id = seen.iter().position(|&k| k == key).unwrap_or_else(|| {
                seen.push(key);
                let vals: Vec<String> = bt
                    .lower()
                    .iter()
                    .zip(bt.upper().iter())
                    .map(|(l, u)| format!("{l}:{u}"))
                    .collect();
                eprintln!(
                    "[sfx] xbox id={} node={name} n={} vals=[{}]",
                    seen.len() - 1,
                    vals.len(),
                    vals.join(",")
                );
                seen.len() - 1
            });
            eprintln!("[sfx] xref dom={dom_idx} node={name} id={id}");
        }
    }
    let prem: Vec<String> = bs
        .entries_for_node(relu_name)
        .map(|e| {
            format!(
                "{}{}@{}",
                e.neuron_idx(),
                if e.sign() > 0.0 { "+" } else { "-" },
                e.split_point()
            )
        })
        .collect();
    let lbs: Vec<String> = bounds
        .map(|b| b.lower().iter().map(|v| format!("{v}")).collect())
        .unwrap_or_default();
    let seed_box: Vec<String> = cache.get(seed_node).map_or_else(Vec::new, |entry| {
        entry
            .lower()
            .iter()
            .zip(entry.upper().iter())
            .map(|(l, u)| format!("{l}:{u}"))
            .collect()
    });
    eprintln!(
        "[sfx] dom={dom_idx} depth={} inbox={inbox_id} prem=[{}] lbs=[{}] seed=[{}]",
        bs.entries.len(),
        prem.join(","),
        lbs.join(","),
        seed_box.join(","),
    );
}

/// Split-premise contradiction test (the prune lane's core; pure for unit
/// tests). Each of the domain's β entries at the seed ReLU IS one of its split
/// premises. `refined_l/refined_u` (indexed by `row_of[neuron]`,
/// `usize::MAX` = unselected) are sound enclosures of the neuron's
/// pre-activation over the subdomain WITHOUT the seed-layer premises, so an
/// ACTIVE premise `z_j ≥ s` with `u' < s − tol`, or an INACTIVE premise
/// `z_j ≤ s` with `l' > s + tol`, proves the constraint set empty. Returns the
/// first contradiction as `(neuron_idx, sign, violation_margin)`; `None` = no
/// contradiction (never prune). Invalid/non-finite refined pairs are skipped
/// (conservative).
fn premise_contradiction(
    beta: &GraphBetaState,
    relu_name: &str,
    row_of: &[usize],
    refined_l: &[f32],
    refined_u: &[f32],
    tol: f32,
) -> Option<(usize, f32, f32)> {
    for e in beta.entries_for_node(relu_name) {
        let j = e.neuron_idx();
        let Some(&row) = row_of.get(j) else {
            continue;
        };
        if row == usize::MAX || row >= refined_l.len() || row >= refined_u.len() {
            continue;
        }
        let (rl, ru) = (refined_l[row], refined_u[row]);
        if !(rl.is_finite() && ru.is_finite() && rl <= ru) {
            continue;
        }
        let s = e.split_point();
        if e.sign() > 0.0 && ru < s - tol {
            return Some((j, 1.0, s - ru));
        }
        if e.sign() < 0.0 && rl > s + tol {
            return Some((j, -1.0, rl - s));
        }
    }
    None
}

/// Per-neuron intersection of the inherited `[ol, ou]` with the refined
/// `[rl, ru]`: `l = max`, `u = min` (NaN-propagating, so a NaN inherited entry
/// is never silently repaired). A non-finite/inverted refined pair, or an empty
/// intersection, conservatively keeps the inherited pair (sound: the inherited
/// entry already encloses the subdomain). Returns the new pair plus whether it
/// strictly tightened.
fn intersect_pair(ol: f32, ou: f32, rl: f32, ru: f32) -> (f32, f32, bool) {
    if !(rl.is_finite() && ru.is_finite() && rl <= ru) {
        return (ol, ou, false);
    }
    let cl = nan_propagating_max(ol, rl);
    let cu = nan_propagating_min(ou, ru);
    // NaN-aware "not (cl <= cu)": TRUE for NaN — `cl > cu` would treat a NaN
    // pair as a valid interval, so the negated comparison is load-bearing.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(cl <= cu) {
        // NaN or empty intersection — keep inherited.
        return (ol, ou, false);
    }
    let tightened = cl > ol || cu < ou;
    (cl, cu, tightened)
}

/// Fold-order walk over the Activation layers of a segment stack (mutable) —
/// the same order as `relu_names` / `beta_signed` (segments in vec order;
/// per segment the branch layer vec in order; ResidualProj F then P), matching
/// `write_back_alpha` in `batched.rs` and the GPU fold.
fn visit_activations_mut(
    segs: &mut [ny_core::GpuResnetSegment],
    f: &mut dyn FnMut(usize, &mut ny_core::GpuCrownLayer),
) {
    let mut r = 0usize;
    let mut walk = |layers: &mut [ny_core::GpuCrownLayer], r: &mut usize| {
        for l in layers.iter_mut() {
            if matches!(l, ny_core::GpuCrownLayer::Activation { .. }) {
                f(*r, l);
                *r += 1;
            }
        }
    };
    for seg in segs.iter_mut() {
        match seg {
            ny_core::GpuResnetSegment::Chain(l) | ny_core::GpuResnetSegment::Residual(l) => {
                walk(l, &mut r)
            }
            ny_core::GpuResnetSegment::ResidualProj(fb, pb) => {
                walk(fb, &mut r);
                walk(pb, &mut r);
            }
        }
    }
}

/// Immutable fold-order Activation walk (same order as
/// [`visit_activations_mut`]).
fn visit_activations(
    segs: &[ny_core::GpuResnetSegment],
    f: &mut dyn FnMut(usize, &ny_core::GpuCrownLayer),
) {
    let mut r = 0usize;
    let mut walk = |layers: &[ny_core::GpuCrownLayer], r: &mut usize| {
        for l in layers.iter() {
            if matches!(l, ny_core::GpuCrownLayer::Activation { .. }) {
                f(*r, l);
                *r += 1;
            }
        }
    };
    for seg in segs.iter() {
        match seg {
            ny_core::GpuResnetSegment::Chain(l) | ny_core::GpuResnetSegment::Residual(l) => {
                walk(l, &mut r)
            }
            ny_core::GpuResnetSegment::ResidualProj(fb, pb) => {
                walk(fb, &mut r);
                walk(pb, &mut r);
            }
        }
    }
}

/// Number of Activation layers in the stack (fold-order count) — the guard
/// that the Activation walk aligns with `relu_names` (the extraction only
/// emits ReLU Activations on this lane; a mismatch fail-closes the α′ lane).
fn count_activations(segs: &[ny_core::GpuResnetSegment]) -> usize {
    let mut n = 0usize;
    visit_activations(segs, &mut |_, _| n += 1);
    n
}

/// Per-Activation UNSTABLE mask of a segment stack: for the ReLU relaxation
/// the chord intercept `−u·l/(u−l)` is > 0 iff `l < 0 < u`, and the
/// extraction pins stable neurons' slopes exactly (1/0 with 0 intercepts) —
/// so `upper_intercept > 0` selects exactly the neurons whose lower slope is
/// a free α ∈ [0,1].
fn unstable_masks(segs: &[ny_core::GpuResnetSegment], n_relu: usize) -> Vec<Vec<bool>> {
    let mut masks: Vec<Vec<bool>> = vec![Vec::new(); n_relu];
    visit_activations(segs, &mut |r, layer| {
        if let ny_core::GpuCrownLayer::Activation {
            upper_intercept, ..
        } = layer
        {
            if r < n_relu {
                masks[r] = upper_intercept.iter().map(|&t| t > 0.0).collect();
            }
        }
    });
    masks
}

/// Snapshot the per-Activation lower slopes (fold order).
fn collect_lower_slopes(segs: &[ny_core::GpuResnetSegment], n_relu: usize) -> Vec<Vec<f32>> {
    let mut out: Vec<Vec<f32>> = vec![Vec::new(); n_relu];
    visit_activations(segs, &mut |r, layer| {
        if let ny_core::GpuCrownLayer::Activation { lower_slope, .. } = layer {
            if r < n_relu {
                out[r] = lower_slope.clone();
            }
        }
    });
    out
}

/// Write α′ into a domain's segments: only neurons that were STEPPED by the
/// ascent AND are unstable in THIS domain's own extraction
/// (`upper_intercept > 0` — stable slopes are exact and must never change).
/// Any α ∈ [0,1] with zero lower intercept is a valid ReLU lower relaxation
/// slope, so this write is sound for every subdomain.
fn write_alpha_prime(
    segs: &mut [ny_core::GpuResnetSegment],
    slopes: &[Vec<f32>],
    stepped: &[Vec<bool>],
) {
    visit_activations_mut(segs, &mut |r, layer| {
        if let ny_core::GpuCrownLayer::Activation {
            lower_slope,
            upper_intercept,
            ..
        } = layer
        {
            let (Some(sl), Some(st)) = (slopes.get(r), stepped.get(r)) else {
                return;
            };
            if sl.len() != lower_slope.len() || st.len() != lower_slope.len() {
                return;
            }
            for (i, s) in lower_slope.iter_mut().enumerate() {
                if st[i] && upper_intercept[i] > 0.0 && sl[i].is_finite() {
                    *s = sl[i].clamp(0.0, 1.0);
                }
            }
        }
    });
}

/// Does a stored α′ match this pass's target (same seed node, width, and
/// ReLU layout)? Refuses reuse across nets/seeds.
fn alpha_prime_matches(
    ap: &AlphaPrime,
    seed_node: &str,
    pre_dim: usize,
    relu_names: &[String],
) -> bool {
    ap.seed_node == seed_node
        && ap.pre_dim == pre_dim
        && ap.relu_names.len() == relu_names.len()
        && ap.relu_names.iter().zip(relu_names).all(|(a, b)| a == b)
}

/// The α′ gradient of the REFINEMENT objective `Σ_{r ∈ rows} w_r · l′_r` for ONE
/// domain: per identity row, the TRUE chain-rule gradient
/// `max(ν,0)·ĥ(x*)` via the host replay of that row's backward over the
/// truncated segments (the same oracle-validated machinery as the wide-α
/// margin ascent — each row is its own CROWN objective with its own ν
/// selection and its own concretization corner), scaled by the per-row weight
/// `w_r` and summed over rows. `row_weights` (aligned with `rows`) is the
/// #joint-interm-alpha margin reweighting; `None` ⇒ every `w_r = 1` (the base
/// uniform `Σ l′` objective, byte-identical). Since `d(w·l′)/dα = w · dl′/dα`
/// the weight is applied to the finished per-row gradient — the fail-closed lb
/// validation still runs on the UNIT row (precision-safe). Rows whose replay
/// fails the validation contribute nothing (gradients only steer α′; the bound
/// is always the sound GPU fold). Returns `(grads, rows_ok)`.
#[allow(clippy::too_many_arguments)]
fn refine_alpha_objective_grads(
    segments: &[ny_core::GpuResnetSegment],
    beta_signed: &[Vec<f32>],
    in_lo: &[f32],
    in_hi: &[f32],
    n_relu: usize,
    pre_dim: usize,
    sel: &[usize],
    rows: &[usize],
    lbs: &[f32],
    row_weights: Option<&[f32]>,
    probe: bool,
) -> (Vec<Vec<f32>>, usize) {
    use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
    let per_row: Vec<Option<Vec<Vec<f32>>>> = rows
        .par_iter()
        .enumerate()
        .map(|(k, &ri)| {
            let neuron = *sel.get(ri)?;
            let lb = *lbs.get(ri)?;
            if neuron >= pre_dim || !lb.is_finite() {
                return None;
            }
            let mut row = vec![0.0f32; pre_dim];
            row[neuron] = 1.0;
            let mut g = super::wide_alpha_true::true_alpha_grads_for_row(
                segments,
                &row,
                beta_signed,
                in_lo,
                in_hi,
                n_relu,
                lb,
                false,
            )?;
            // #joint-interm-alpha: d(w·l′)/dα = w · dl′/dα — scale the finished
            // per-row gradient (uniform lane leaves w = 1 ⇒ byte-identical).
            if let Some(w) = row_weights.and_then(|ws| ws.get(k).copied()) {
                if w != 1.0 {
                    for layer in g.iter_mut() {
                        for v in layer.iter_mut() {
                            *v *= w;
                        }
                    }
                }
            }
            Some(g)
        })
        .collect();
    let mut grads: Vec<Vec<f32>> = Vec::with_capacity(n_relu);
    let mut rows_ok = 0usize;
    for g in per_row.into_iter().flatten() {
        if g.len() != n_relu {
            continue;
        }
        rows_ok += 1;
        if grads.is_empty() {
            grads = g;
            continue;
        }
        for (acc, gr) in grads.iter_mut().zip(g) {
            if acc.len() == gr.len() {
                for (a, v) in acc.iter_mut().zip(gr) {
                    *a += v;
                }
            }
        }
    }
    if grads.is_empty() {
        grads = vec![Vec::new(); n_relu];
    }
    if probe {
        let max_g = grads
            .iter()
            .flatten()
            .fold(0.0f32, |m, &g| nan_propagating_max(m, g.abs()));
        eprintln!(
            "[interm-refine-alpha] grads rows_ok={rows_ok}/{} max|g|={max_g:.3e}",
            rows.len()
        );
    }
    (grads, rows_ok)
}

/// Minimal Adam ascent state for α′ (bias-corrected, [0,1]-projected). Local
/// on purpose: the parameter is the raw lower-slope vector, not a
/// `GraphDomainAlphaState` (nothing to track — the mask pins the stepped set).
struct AlphaAdam {
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

impl AlphaAdam {
    fn new(shape: &[Vec<f32>]) -> Self {
        Self {
            m: shape.iter().map(|s| vec![0.0; s.len()]).collect(),
            v: shape.iter().map(|s| vec![0.0; s.len()]).collect(),
        }
    }

    /// One ascent step (maximize): `α += lr·m̂/(√v̂+ε)`, clamped to [0,1],
    /// masked neurons only. Returns max |gradient| over stepped neurons.
    fn step(
        &mut self,
        slopes: &mut [Vec<f32>],
        grads: &[Vec<f32>],
        stepped: &[Vec<bool>],
        lr: f32,
        t: usize,
    ) -> f32 {
        const B1: f32 = 0.9;
        const B2: f32 = 0.999;
        const EPS: f32 = 1e-8;
        let bc1 = 1.0 - B1.powi(t as i32);
        let bc2 = 1.0 - B2.powi(t as i32);
        let mut max_g = 0.0f32;
        for r in 0..slopes.len() {
            let (Some(gr), Some(st)) = (grads.get(r), stepped.get(r)) else {
                continue;
            };
            if gr.len() != slopes[r].len() || st.len() != slopes[r].len() {
                continue;
            }
            for i in 0..slopes[r].len() {
                if !st[i] {
                    continue;
                }
                let g = gr[i];
                if !g.is_finite() {
                    continue;
                }
                max_g = nan_propagating_max(max_g, g.abs());
                let m = &mut self.m[r][i];
                let v = &mut self.v[r][i];
                *m = B1 * *m + (1.0 - B1) * g;
                *v = B2 * *v + (1.0 - B2) * g * g;
                let mhat = *m / bc1;
                let vhat = *v / bc2;
                slopes[r][i] = (slopes[r][i] + lr * mhat / (vhat.sqrt() + EPS)).clamp(0.0, 1.0);
            }
        }
        max_g
    }
}

/// Element-wise best-merge of two sound per-row enclosures of the SAME
/// domain's rows: `l = max`, `u = min` (finite-guarded). Every α′ iterate is
/// a sound enclosure, so the merge is sound and never looser than iterate 0.
fn merge_refine_result(dst: &mut ny_core::GpuCrownResult, src: &ny_core::GpuCrownResult) {
    if dst.lower_bounds.len() != src.lower_bounds.len()
        || dst.upper_bounds.len() != src.upper_bounds.len()
    {
        return;
    }
    for (d, &s) in dst.lower_bounds.iter_mut().zip(&src.lower_bounds) {
        if s.is_finite() && (!d.is_finite() || s > *d) {
            *d = s;
        }
    }
    for (d, &s) in dst.upper_bounds.iter_mut().zip(&src.upper_bounds) {
        if s.is_finite() && (!d.is_finite() || s < *d) {
            *d = s;
        }
    }
}

/// The α′ ASCENT (#alpha-prime, see [`interm_refine_alpha_enabled`]): a small
/// Adam ascent of the truncated stack's lower slopes against the refinement
/// objective `Σ_rows l′_row`, replayed on ONE domain (the batch's first
/// successful one) and applied to EVERY domain's segments; each iterate
/// re-runs the batched refinement backward and best-merges the per-row bounds
/// into `per_domain` (sound: every iterate is a sound enclosure for its
/// α′ ∈ [0,1]). Returns the α′ snapshot to store (with `improved` marking
/// whether any iterate beat the borrowed-α objective on the replay domain).
///
/// SOUNDNESS: gradients only steer α′ (a wrong gradient degrades the ascent,
/// never the bound); every consumed bound comes from the sound GPU fold with
/// projected slopes, and only element-wise-tightest sound iterates are kept.
#[allow(clippy::too_many_arguments)]
fn alpha_prime_ascent(
    gpu: &dyn ny_core::GpuCrownBackward,
    preps: &mut [Option<ResnetDomainPrep>],
    idxs: &[usize],
    seed: &ny_core::GpuCrownSeed,
    per_domain: &mut [Option<ny_core::GpuCrownResult>],
    seed_node: &str,
    pre_dim: usize,
    sel: &[usize],
    row_widths: &[f32],
    opts: &IntermRefineOptions,
    past_deadline: &dyn Fn() -> bool,
) -> Option<AlphaPrime> {
    let probe = opts.probe;
    let n_rows = sel.len();
    let replay = idxs
        .iter()
        .copied()
        .find(|&i| preps[i].is_some() && per_domain[i].is_some())?;
    let (n_relu, relu_names, stepped, mut slopes) = {
        let p = preps[replay].as_ref()?;
        let n_relu = p.relu_names.len();
        if count_activations(&p.segments) != n_relu {
            if probe {
                eprintln!(
                    "[interm-refine-alpha] activation walk misaligned with relu_names — lane skipped"
                );
            }
            return None;
        }
        (
            n_relu,
            p.relu_names.clone(),
            unstable_masks(&p.segments, n_relu),
            collect_lower_slopes(&p.segments, n_relu),
        )
    };
    if !stepped.iter().any(|m| m.contains(&true)) {
        if probe {
            eprintln!("[interm-refine-alpha] no unstable below-seed neurons — nothing to step");
        }
        return None;
    }
    // Replay rows: widest inherited interval first (most refinement headroom),
    // capped (`NY_INTERM_REFINE_ALPHA_MAX_ROWS`) — the host replays dominate
    // the ascent wall. Cost/quality only, never soundness.
    let mut rows: Vec<usize> = (0..n_rows).collect();
    rows.sort_by(|&a, &b| {
        let wa = row_widths.get(a).copied().unwrap_or(0.0);
        let wb = row_widths.get(b).copied().unwrap_or(0.0);
        wb.partial_cmp(&wa)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    rows.truncate(opts.alpha_max_rows);
    // #joint-interm-alpha: per-refinement-row margin weight `w_all[ri] =
    // margin_weights[sel[ri]]` (≥0). When the joint gate is off (or the tail map
    // was not resolvable) `w_all` is None ⇒ the uniform `Σ l′` objective,
    // byte-identical. `picked_weights` is `w_all` gathered over the (sorted,
    // truncated) replay `rows`, aligned with `refine_alpha_objective_grads`.
    let w_all: Option<Vec<f32>> = opts
        .margin_weights
        .as_ref()
        .filter(|_| opts.joint_margin)
        .map(|mw| {
            (0..n_rows)
                .map(|ri| sel.get(ri).and_then(|&j| mw.get(j)).copied().unwrap_or(0.0))
                .collect()
        });
    let picked_weights: Option<Vec<f32>> = w_all.as_ref().map(|wa| {
        rows.iter()
            .map(|&ri| wa.get(ri).copied().unwrap_or(0.0))
            .collect()
    });
    let obj_of = |lbs: &[f32]| -> f64 {
        match &w_all {
            // Weighted margin objective Σ_ri w_all[ri]·l′_ri (joint lane).
            Some(wa) => lbs
                .iter()
                .zip(wa.iter())
                .filter(|(v, _)| v.is_finite())
                .map(|(&v, &w)| v as f64 * w as f64)
                .sum(),
            None => lbs
                .iter()
                .filter(|v| v.is_finite())
                .map(|&v| v as f64)
                .sum(),
        }
    };
    let mut cur_lbs: Vec<f32> = per_domain[replay].as_ref()?.lower_bounds.clone();
    if cur_lbs.len() != n_rows {
        return None;
    }
    let obj0 = obj_of(&cur_lbs);
    let mut best_obj = obj0;
    let mut best_slopes: Option<Vec<Vec<f32>>> = None;
    let mut best_iter = 0usize;
    let mut adam = AlphaAdam::new(&slopes);
    let t0 = Instant::now();
    for t in 1..=opts.alpha_iters {
        if past_deadline() {
            break;
        }
        let (grads, rows_ok) = {
            let p = preps[replay].as_ref()?;
            refine_alpha_objective_grads(
                &p.segments,
                &p.beta_signed,
                &p.in_lo,
                &p.in_hi,
                n_relu,
                pre_dim,
                sel,
                &rows,
                &cur_lbs,
                picked_weights.as_deref(),
                probe,
            )
        };
        if rows_ok == 0 {
            if probe {
                eprintln!("[interm-refine-alpha] iter={t} every row replay failed — stop");
            }
            break;
        }
        let max_g = adam.step(&mut slopes, &grads, &stepped, opts.alpha_lr, t);
        if max_g == 0.0 || !max_g.is_finite() {
            break;
        }
        for &i in idxs {
            if let Some(p) = preps[i].as_mut() {
                write_alpha_prime(&mut p.segments, &slopes, &stepped);
            }
        }
        let results = {
            let refs: Vec<ny_core::GpuResnetBatchedDomainRef> = idxs
                .iter()
                .map(|&i| {
                    let p = preps[i].as_ref().expect("idxs holds Some preps only");
                    ny_core::GpuResnetBatchedDomainRef {
                        segments: &p.segments,
                        input_lower: &p.in_lo,
                        input_upper: &p.in_hi,
                        beta_signed: &p.beta_signed,
                        frontier_abs: &p.frontier_abs,
                        node_abs: &p.node_abs,
                    }
                })
                .collect();
            match gpu.crown_backward_gpu_resnet_sound_beta_batched(&refs, seed) {
                Ok(r) if r.len() == idxs.len() => r,
                _ => break,
            }
        };
        let mut obj_t = f64::NEG_INFINITY;
        for (&i, r) in idxs.iter().zip(&results) {
            if r.lower_bounds.len() != n_rows || r.upper_bounds.len() != n_rows {
                continue;
            }
            if i == replay {
                cur_lbs.clone_from(&r.lower_bounds);
                obj_t = obj_of(&r.lower_bounds);
            }
            match per_domain[i].as_mut() {
                Some(dst) => merge_refine_result(dst, r),
                None => {
                    per_domain[i] = Some(ny_core::GpuCrownResult {
                        lower_bounds: r.lower_bounds.clone(),
                        upper_bounds: r.upper_bounds.clone(),
                    })
                }
            }
        }
        if probe {
            eprintln!(
                "[interm-refine-alpha] iter={t} rows_ok={rows_ok} max|g|={max_g:.3e} \
                 obj={obj_t:.4} (obj0={obj0:.4} best={best_obj:.4})"
            );
        }
        if obj_t > best_obj {
            best_obj = obj_t;
            best_slopes = Some(slopes.clone());
            best_iter = t;
        }
    }
    let improved = best_iter > 0;
    if probe {
        eprintln!(
            "[interm-refine-alpha] ascent done iters={} improved={improved} dObj={:+.4} ({}ms)",
            opts.alpha_iters,
            best_obj - obj0,
            t0.elapsed().as_millis()
        );
    }
    Some(AlphaPrime {
        seed_node: seed_node.to_string(),
        pre_dim,
        relu_names,
        slopes: best_slopes.unwrap_or(slopes),
        stepped,
        improved,
    })
}

/// PER-TARGET α′ ASCENT (#ab-parity-interm, [`ab_parity_interm_enabled`]): the
/// auto_LiRPA per-target decoupling. Where [`alpha_prime_ascent`] optimizes ONE
/// shared lower-slope set against the SCALARIZED objective `Σ_rows w_r·l′_r`,
/// this gives EACH target seed-row `ri` its OWN slope vector `α^(ri)`, ascended
/// against ITS OWN bound `l′_ri` (that row's per-target gradient), and computes
/// the SOUND per-row bound with `α^(ri)` from the GPU fold. The optimum for one
/// target no longer has to compromise with every other target's optimum — each
/// picks the below-seed slopes that tighten ITS pre-activation box.
///
/// STRUCTURE: the per-target ascent optimizes each `α^(ri)` on the batch's
/// representative (replay) domain via the oracle-validated host-replay gradient
/// (`refine_alpha_objective_grads` on a single row) + a single-domain sound GPU
/// fold to score each iterate; then APPLIES each converged `α^(ri)` to EVERY
/// domain in the batch (one batched backward per target, writing `α^(ri)` into
/// all domains' segments and best-merging ONLY that target's column into
/// `per_domain`). Cost scales with `n_targets` (capped by `alpha_max_rows`) —
/// this is the deliberate extra α cost of the dark parity lane.
///
/// SOUNDNESS: identical discipline to the base lane — any `α ∈ [0,1]` is a valid
/// lower ReLU slope, gradients only STEER which α each target picks, every
/// consumed bound is the sound GPU fold (`crown_backward_gpu_resnet_sound_beta`
/// / `_batched`) with projected slopes, and the merge keeps the element-wise
/// tightest sound value per row (`l = max`, `u = min`). Applying a target's α to
/// a NON-replay domain is sound because [`write_alpha_prime`] only touches
/// neurons unstable in THAT domain's own extraction (stable slopes stay exact).
/// Returns the number of targets whose per-target bound beat the borrowed-α one.
#[allow(clippy::too_many_arguments)]
fn alpha_prime_ascent_per_target(
    gpu: &dyn ny_core::GpuCrownBackward,
    preps: &mut [Option<ResnetDomainPrep>],
    idxs: &[usize],
    seed: &ny_core::GpuCrownSeed,
    per_domain: &mut [Option<ny_core::GpuCrownResult>],
    pre_dim: usize,
    sel: &[usize],
    row_widths: &[f32],
    opts: &IntermRefineOptions,
    past_deadline: &dyn Fn() -> bool,
) -> usize {
    let probe = opts.probe;
    let n_rows = sel.len();
    let replay = idxs
        .iter()
        .copied()
        .find(|&i| preps[i].is_some() && per_domain[i].is_some());
    let Some(replay) = replay else {
        return 0;
    };
    // Snapshot the replay domain (owned copies — the application phase below
    // mutates every domain's segments through `preps`, so the ascent phase must
    // not borrow `preps[replay]`).
    let (
        n_relu,
        stepped,
        base_slopes,
        in_lo,
        in_hi,
        beta_signed,
        frontier_abs,
        node_abs,
        base_segs,
    ) = {
        let Some(p) = preps[replay].as_ref() else {
            return 0;
        };
        let n_relu = p.relu_names.len();
        if count_activations(&p.segments) != n_relu {
            if probe {
                eprintln!(
                    "[ab-parity-interm] activation walk misaligned with relu_names — lane skipped"
                );
            }
            return 0;
        }
        (
            n_relu,
            unstable_masks(&p.segments, n_relu),
            collect_lower_slopes(&p.segments, n_relu),
            p.in_lo.clone(),
            p.in_hi.clone(),
            p.beta_signed.clone(),
            p.frontier_abs.clone(),
            p.node_abs.clone(),
            p.segments.clone(),
        )
    };
    if !stepped.iter().any(|m| m.contains(&true)) {
        if probe {
            eprintln!("[ab-parity-interm] no unstable below-seed neurons — nothing to step");
        }
        return 0;
    }
    let base_lbs = match per_domain[replay].as_ref() {
        Some(r) if r.lower_bounds.len() == n_rows => r.lower_bounds.clone(),
        _ => return 0,
    };
    // Target rows: widest inherited interval first (most refinement headroom),
    // capped by `alpha_max_rows` (cost only — untargeted rows keep the sound
    // borrowed-α bound).
    let mut targets: Vec<usize> = (0..n_rows).collect();
    targets.sort_by(|&a, &b| {
        let wa = row_widths.get(a).copied().unwrap_or(0.0);
        let wb = row_widths.get(b).copied().unwrap_or(0.0);
        wb.partial_cmp(&wa)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    targets.truncate(opts.alpha_max_rows);

    // PHASE 1 — per-target ascent on the replay domain. `α^(ri)` lives in
    // `per_target_slopes[ti]` (ti indexes `targets`).
    let mut per_target_slopes: Vec<Vec<Vec<f32>>> = Vec::with_capacity(targets.len());
    let mut work_segs = base_segs;
    let mut lbs = base_lbs.clone();
    let mut improved = 0usize;
    let t0 = Instant::now();
    for &ri in &targets {
        let base_lb = base_lbs.get(ri).copied().unwrap_or(f32::NEG_INFINITY);
        if past_deadline() {
            per_target_slopes.push(base_slopes.clone());
            continue;
        }
        let mut slopes = base_slopes.clone();
        let mut adam = AlphaAdam::new(&base_slopes);
        let mut cur_lb = base_lb;
        let mut best_lb = base_lb;
        let mut best_slopes = base_slopes.clone();
        for t in 1..=opts.alpha_iters {
            if past_deadline() {
                break;
            }
            // Gradient of THIS row's l′ at the current per-target slopes.
            write_alpha_prime(&mut work_segs, &slopes, &stepped);
            lbs[ri] = cur_lb;
            let (grads, rows_ok) = refine_alpha_objective_grads(
                &work_segs,
                &beta_signed,
                &in_lo,
                &in_hi,
                n_relu,
                pre_dim,
                sel,
                std::slice::from_ref(&ri),
                &lbs,
                None,
                false,
            );
            if rows_ok == 0 {
                break;
            }
            let max_g = adam.step(&mut slopes, &grads, &stepped, opts.alpha_lr, t);
            if max_g == 0.0 || !max_g.is_finite() {
                break;
            }
            // Sound bound for row ri with the stepped per-target slopes.
            write_alpha_prime(&mut work_segs, &slopes, &stepped);
            let r = match gpu.crown_backward_gpu_resnet_sound_beta(
                &work_segs,
                seed,
                &in_lo,
                &in_hi,
                &beta_signed,
                &frontier_abs,
                &node_abs,
            ) {
                Ok(r) if r.lower_bounds.len() == n_rows => r,
                _ => break,
            };
            cur_lb = r.lower_bounds[ri];
            if cur_lb.is_finite() && cur_lb > best_lb {
                best_lb = cur_lb;
                best_slopes = slopes.clone();
            }
        }
        if best_lb > base_lb {
            improved += 1;
        }
        per_target_slopes.push(best_slopes);
    }
    if probe {
        eprintln!(
            "[ab-parity-interm] ascent targets={} improved={improved} ({}ms)",
            targets.len(),
            t0.elapsed().as_millis()
        );
    }

    // PHASE 2 — application: for each target, write its α into EVERY domain's
    // segments, ONE batched backward, best-merge ONLY that target's column.
    for (ti, &ri) in targets.iter().enumerate() {
        if past_deadline() {
            break;
        }
        let slopes = &per_target_slopes[ti];
        for &i in idxs {
            if let Some(p) = preps[i].as_mut() {
                write_alpha_prime(&mut p.segments, slopes, &stepped);
            }
        }
        let refs: Vec<ny_core::GpuResnetBatchedDomainRef> = idxs
            .iter()
            .map(|&i| {
                let p = preps[i].as_ref().expect("idxs holds Some preps only");
                ny_core::GpuResnetBatchedDomainRef {
                    segments: &p.segments,
                    input_lower: &p.in_lo,
                    input_upper: &p.in_hi,
                    beta_signed: &p.beta_signed,
                    frontier_abs: &p.frontier_abs,
                    node_abs: &p.node_abs,
                }
            })
            .collect();
        let results = match gpu.crown_backward_gpu_resnet_sound_beta_batched(&refs, seed) {
            Ok(r) if r.len() == idxs.len() => r,
            _ => continue,
        };
        for (&i, r) in idxs.iter().zip(&results) {
            if r.lower_bounds.len() != n_rows || r.upper_bounds.len() != n_rows {
                continue;
            }
            if let Some(dst) = per_domain[i].as_mut() {
                if ri < dst.lower_bounds.len() && ri < dst.upper_bounds.len() {
                    let lo = r.lower_bounds[ri];
                    let hi = r.upper_bounds[ri];
                    if lo.is_finite()
                        && (!dst.lower_bounds[ri].is_finite() || lo > dst.lower_bounds[ri])
                    {
                        dst.lower_bounds[ri] = lo;
                    }
                    if hi.is_finite()
                        && (!dst.upper_bounds[ri].is_finite() || hi < dst.upper_bounds[ri])
                    {
                        dst.upper_bounds[ri] = hi;
                    }
                }
            }
        }
    }
    improved
}

/// Batch stats for the probe line.
#[derive(Default)]
struct RefineStats {
    domains_refined: usize,
    neurons_tightened: usize,
    newly_stable: usize,
    crossings_kept: usize,
    infeasible: usize,
    width_before: f64,
    width_after: f64,
    /// #clip-interm-resnet-batched: domains whose seed-layer bounds were tightened
    /// by the batched split-constraint clip (constrained concretization over the
    /// batched ResidentCoeff, zero extra backward).
    clip_domains: usize,
    /// #clip-interm-resnet-batched: seed rows refused (non-finite coeff-error fold —
    /// kept inherited, sound).
    clip_refused_rows: usize,
    /// #clip-interm-resnet-batched: max per-row tightening (|Δl| or |Δu|) from the clip.
    clip_max_tighten: f32,
    /// #clip-interm-guard: domains whose clip was REVERTED by the fail-closed runtime
    /// guard (a feasible directed sample fell outside a tightened row, or a non-finite
    /// clip bound) — the seed node kept its inherited parent bound. Nonzero ⇒ a
    /// (near-)unsound clip was caught at runtime.
    clip_guard_reverts: usize,
}

/// One layer of root-valid affine rows specialized to one exact child history.
/// Histories name the ReLU, while the provenance subject is its pre-activation
/// input node (`seed_node`).
struct CertifiedLayerCapture {
    seed_node: Arc<str>,
    selected_neurons: Arc<[usize]>,
    row_of_neuron: Arc<[usize]>,
    objective_rows: Vec<usize>,
    pass: CrownPassStamp,
    token: CertifiedAffineEnclosure,
}

#[derive(Default)]
struct DomainClipBank {
    layers: HashMap<String, CertifiedLayerCapture>,
    target_order: Vec<String>,
    disabled: bool,
}

impl BetaCrownVerifier {
    /// Refine each domain's LAST-ReLU pre-activation bounds (see module docs;
    /// with `NY_INTERM_REFINE_LAYERS=2` also the second-to-last ReLU's,
    /// deepest-first so the last-ReLU pass consumes the improved upstream
    /// entry) and return the refined per-domain caches plus per-domain
    /// infeasibility flags (prune lane), or `None` when the lane does not
    /// apply (no sound GPU, no clean last-ReLU chain, oversized seed layer,
    /// deadline passed, or every domain refused) — the caller then keeps the
    /// inherited caches byte-identically.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::beta_crown::engine::graph) fn refine_last_relu_interm_bounds(
        &self,
        graph: &GraphNetwork,
        output_node: &str,
        n_domains: usize,
        bounds_caches: &[HashMap<String, Arc<BoundedTensor>>],
        constrained_inputs: &[BoundedTensor],
        histories: &[&GraphSplitHistory],
        beta_states: &[Option<&GraphBetaState>],
        alpha_states: &[Option<&GraphDomainAlphaState>],
        engine: &dyn GemmEngine,
        spec_matrix: &Array2<f32>,
    ) -> Option<IntermRefineOutcome> {
        // Candidate-side CROWN calls use this same propagation primitive but
        // discard its node cache after reading an advisory branch score.
        // αβ-CROWN clips only the committed children, so decline the optional
        // work before even planning margin weights for a suppressed call.
        if self
            .complete_clip_deadline_overrides
            .complete_clip_suppressed()
        {
            return None;
        }
        let effective_deadline = self.effective_graph_bab_deadline();
        let mut opts = if self.config.enable_clip_interm_domain {
            IntermRefineOptions::production_complete_clip(
                effective_deadline,
                self.config.clip_interm_topk,
            )
        } else {
            let mut research = IntermRefineOptions::from_env();
            research.deadline = effective_deadline;
            research
        };
        // #joint-interm-alpha: compute the per-seed-neuron margin weights here
        // (the tail linear map + spec rows are only available at the batched call
        // site, not in `from_env`). `None` ⇒ the α′ lane keeps its uniform
        // objective (sound fallback).
        if opts.joint_margin || opts.selective_topk > 0 {
            opts.margin_weights = match compute_margin_weights_with_deadline(
                graph,
                output_node,
                spec_matrix,
                opts.deadline,
            ) {
                Ok(weights) => weights,
                Err(error) => {
                    if opts.probe {
                        eprintln!("[interm-refine] margin weights REFUSED: {error}");
                    }
                    None
                }
            };
            if opts.probe && opts.joint_margin {
                eprintln!(
                    "[joint-interm-alpha] margin weights: {}",
                    match &opts.margin_weights {
                        Some(m) => format!(
                            "n={} nonzero={} max={:.4}",
                            m.len(),
                            m.iter().filter(|&&v| v > 0.0).count(),
                            m.iter().cloned().fold(0.0f32, f32::max)
                        ),
                        None => "unavailable (uniform fallback)".to_string(),
                    }
                );
            }
        }
        self.refine_interm_bounds_with_opts(
            graph,
            output_node,
            n_domains,
            bounds_caches,
            constrained_inputs,
            histories,
            beta_states,
            alpha_states,
            engine,
            &opts,
        )
    }

    /// Options-explicit body of [`Self::refine_last_relu_interm_bounds`]
    /// (unit tests pass `IntermRefineOptions` directly — no env races).
    #[allow(clippy::too_many_arguments)]
    pub(in crate::beta_crown::engine::graph) fn refine_interm_bounds_with_opts(
        &self,
        graph: &GraphNetwork,
        output_node: &str,
        n_domains: usize,
        bounds_caches: &[HashMap<String, Arc<BoundedTensor>>],
        constrained_inputs: &[BoundedTensor],
        histories: &[&GraphSplitHistory],
        beta_states: &[Option<&GraphBetaState>],
        alpha_states: &[Option<&GraphDomainAlphaState>],
        engine: &dyn GemmEngine,
        opts: &IntermRefineOptions,
    ) -> Option<IntermRefineOutcome> {
        // Keep the options-explicit face suppression-safe as well; tests and
        // future propagation adapters may call it without going through the
        // environment/config wrapper above.
        if self
            .complete_clip_deadline_overrides
            .complete_clip_suppressed()
        {
            return None;
        }
        let configured_deadline = match (opts.deadline, self.effective_graph_bab_deadline()) {
            (Some(requested), Some(configured)) => Some(requested.min(configured)),
            (Some(requested), None) => Some(requested),
            (None, configured) => configured,
        };
        let mut effective_opts = opts.clone();
        effective_opts.deadline = self
            .complete_clip_deadline_overrides
            .effective(configured_deadline);
        let opts = &effective_opts;
        if n_domains == 0
            || bounds_caches.len() != n_domains
            || constrained_inputs.len() != n_domains
            || histories.len() != n_domains
            || beta_states.len() != n_domains
            || alpha_states.len() != n_domains
        {
            return None;
        }
        if !crate::network::resnet_beta_gpu_enabled() {
            return None;
        }
        // Deadline firewall: the refinement is an EXTRA GPU pass; never start it
        // past the BaB deadline (the margin backward still runs — unchanged).
        if opts
            .deadline
            .is_some_and(|deadline| Instant::now() >= deadline)
        {
            return None;
        }
        let gpu = engine
            .as_gpu_crown_backward()
            .filter(|g| g.provides_sound_gpu_crown())
            .filter(|g| opts.deadline.is_none() || g.honors_crown_backward_deadline())?;

        // ADAPTIVE SCHEDULE (#adaptive-refine, `NY_INTERM_REFINE_ADAPTIVE=1`,
        // dark): once the per-process latch has recorded a zero-yield depth,
        // domains at depth ≥ latch keep their inherited caches (skip =
        // identical to a per-domain refusal — sound, cost-only). A proper
        // subset recurses on the still-refined shallow sub-batch and scatters
        // back (same shape as the min_depth gate below); the recursed call
        // re-runs this filter as a no-op and owns the latch update.
        if let Some(latch) = &opts.adaptive_latch {
            let latched = latch.load(std::sync::atomic::Ordering::Relaxed);
            if latched != usize::MAX {
                let depth_of =
                    |i: usize| -> usize { beta_states[i].map_or(0, |b| b.entries.len()) };
                // #interm-refine-redo: depth-multiple domains re-refine past
                // the latch (stale-inheritance repair under upstream splits).
                let included: Vec<usize> = (0..n_domains)
                    .filter(|&i| latched_domain_refines(depth_of(i), latched, opts.redo_every))
                    .collect();
                if included.is_empty() {
                    if opts.probe {
                        eprintln!(
                            "[interm-refine-adaptive] batch SKIPPED ({n_domains} domains, all \
                             at depth >= latch {latched}, no redo multiple of {})",
                            opts.redo_every
                        );
                    }
                    return None;
                }
                if included.len() < n_domains {
                    let sub_caches: Vec<HashMap<String, Arc<BoundedTensor>>> =
                        included.iter().map(|&i| bounds_caches[i].clone()).collect();
                    let sub_inputs: Vec<BoundedTensor> = included
                        .iter()
                        .map(|&i| constrained_inputs[i].clone())
                        .collect();
                    let sub_histories: Vec<&GraphSplitHistory> =
                        included.iter().map(|&i| histories[i]).collect();
                    let sub_betas: Vec<Option<&GraphBetaState>> =
                        included.iter().map(|&i| beta_states[i]).collect();
                    let sub_alphas: Vec<Option<&GraphDomainAlphaState>> =
                        included.iter().map(|&i| alpha_states[i]).collect();
                    if opts.probe {
                        eprintln!(
                            "[interm-refine-adaptive] batch FILTERED to {}/{n_domains} domains \
                             below latch {latched} (redo stride {})",
                            included.len(),
                            opts.redo_every
                        );
                    }
                    let outcome = self.refine_interm_bounds_with_opts(
                        graph,
                        output_node,
                        included.len(),
                        &sub_caches,
                        &sub_inputs,
                        &sub_histories,
                        &sub_betas,
                        &sub_alphas,
                        engine,
                        opts,
                    )?;
                    let mut caches = bounds_caches.to_vec();
                    let mut infeasible = vec![false; n_domains];
                    for (k, &i) in included.iter().enumerate() {
                        caches[i] = outcome.caches[k].clone();
                        infeasible[i] = outcome.infeasible[k];
                    }
                    return Some(IntermRefineOutcome { caches, infeasible });
                }
                // Every domain is below the latch — fall through.
            }
        }

        // Depth gate (#fit-100s, `NY_INTERM_REFINE_MIN_DEPTH=d`, default 0 =
        // byte-identical): refine only domains with ≥ d split premises. A
        // proper subset recurses on the filtered sub-batch (min_depth=0) and
        // scatters the refined caches/flags back; excluded domains keep their
        // inherited caches — sound (identical to a per-domain refusal).
        if opts.min_depth > 0 {
            let included: Vec<usize> = (0..n_domains)
                .filter(|&i| beta_states[i].map_or(0, |b| b.entries.len()) >= opts.min_depth)
                .collect();
            if included.is_empty() {
                return None;
            }
            if included.len() < n_domains {
                let sub_caches: Vec<HashMap<String, Arc<BoundedTensor>>> =
                    included.iter().map(|&i| bounds_caches[i].clone()).collect();
                let sub_inputs: Vec<BoundedTensor> = included
                    .iter()
                    .map(|&i| constrained_inputs[i].clone())
                    .collect();
                let sub_histories: Vec<&GraphSplitHistory> =
                    included.iter().map(|&i| histories[i]).collect();
                let sub_betas: Vec<Option<&GraphBetaState>> =
                    included.iter().map(|&i| beta_states[i]).collect();
                let sub_alphas: Vec<Option<&GraphDomainAlphaState>> =
                    included.iter().map(|&i| alpha_states[i]).collect();
                let sub_opts = IntermRefineOptions {
                    min_depth: 0,
                    ..opts.clone()
                };
                let outcome = self.refine_interm_bounds_with_opts(
                    graph,
                    output_node,
                    included.len(),
                    &sub_caches,
                    &sub_inputs,
                    &sub_histories,
                    &sub_betas,
                    &sub_alphas,
                    engine,
                    &sub_opts,
                )?;
                let mut caches = bounds_caches.to_vec();
                let mut infeasible = vec![false; n_domains];
                for (k, &i) in included.iter().enumerate() {
                    caches[i] = outcome.caches[k].clone();
                    infeasible[i] = outcome.infeasible[k];
                }
                return Some(IntermRefineOutcome { caches, infeasible });
            }
            // Every domain is deep enough — fall through to the whole-batch path.
        }

        // Install the cooperative deadline only after the filtering recursions
        // above. A recursive sub-batch owns its own exact-backend scope; the
        // outer frame returns before reaching this point. This guard covers the
        // complete root-bank lookup and the full resident backward routine.
        let _gpu_deadline_scope =
            crate::sound_gpu_gate::GpuCrownBackendDeadlineScope::set(gpu, opts.deadline);

        let complete_clip_root_batch = if opts.clip_resnet {
            self.complete_clip_root_bounds_cache
                .get_finalized_for_batch(graph, constrained_inputs, opts.deadline)
        } else {
            None
        };
        if opts.require_complete_clip_root_bank && complete_clip_root_batch.is_none() {
            if opts.probe {
                eprintln!(
                    "[complete-clip] finalized iteration-stamped root bank unavailable; \
                     production clipping skipped"
                );
            }
            return None;
        }
        let complete_clip_root_bounds = complete_clip_root_batch
            .as_ref()
            .map(|(bounds, _)| Arc::clone(bounds));
        // αβ-CROWN's DomainClipper uses every split premise for its first two
        // BaB iterations. Later it walks the fixed split-node dictionary in
        // network order and uses the newest premise from the first nonempty
        // layer. This caps the coordinate solver as histories deepen while
        // retaining the full history in the provenance identity and ordinary
        // β-CROWN backward.
        let complete_clip_use_all_history = complete_clip_root_batch
            .as_ref()
            .is_some_and(|(_, iteration)| *iteration <= COMPLETE_CLIP_ALL_HISTORY_ROUNDS);
        if opts.clip_resnet && complete_clip_root_bounds.is_none() && opts.probe {
            eprintln!(
                "[complete-clip] root bound cache unavailable; ordinary intermediate \
                 refinement continues without clipping"
            );
        }
        let precomputed_decisions: Vec<Option<Arc<CompleteClipDecisionIndices>>> =
            if self.config.enable_clip_interm_domain && complete_clip_root_bounds.is_some() {
                constrained_inputs
                    .iter()
                    .zip(histories)
                    .map(|(input, history)| {
                        self.complete_clip_root_bounds_cache
                            .take_decisions(graph, input, history)
                    })
                    .collect()
            } else {
                (0..n_domains).map(|_| None).collect()
            };
        if opts.probe || std::env::var("NY_MO_KFSB_PROBE").ok().as_deref() == Some("1") {
            let hits = precomputed_decisions
                .iter()
                .filter(|snapshot| snapshot.is_some())
                .count();
            let layers = precomputed_decisions
                .iter()
                .flatten()
                .map(|snapshot| snapshot.len())
                .sum::<usize>();
            eprintln!(
                "[complete-clip-decision-consume] domains={} hits={} layers={} \
                 local_fallbacks={}",
                n_domains,
                hits,
                layers,
                n_domains.saturating_sub(hits),
            );
        }
        let has_precomputed_target = |relu_name: &str| {
            precomputed_decisions.iter().any(|snapshot| {
                snapshot
                    .as_ref()
                    .is_some_and(|indices| indices.contains_key(relu_name))
            })
        };

        // Any domain with an unstable neuron at a seed ⇒ useful target (a
        // fully-stable seed's slopes are already exact — refining its input
        // can never tighten anything downstream).
        let has_unstable = |seed: &str| {
            bounds_caches.iter().any(|cache| {
                cache.get(seed).is_some_and(|entry| {
                    entry
                        .lower()
                        .iter()
                        .zip(entry.upper().iter())
                        .any(|(&l, &u)| l < 0.0 && u > 0.0)
                })
            })
        };

        // Seed chain, built in EXEC ORDER (earliest first) — the cascade: each
        // pass (and afterwards the margin backward) consumes the earlier
        // passes' improved upstream entries through the ReLU relaxations the
        // extraction derives from the caches.
        let (last_relu_name, last_seed) = find_last_relu_seed(graph, output_node)?;
        let mut seeds: Vec<(String, String)> = if let Some(names) = &opts.seeds {
            // NAMED-SEED LANE (`NY_INTERM_REFINE_SEEDS`, #midref): resolve each
            // token (exact ReLU node name, or `last` = the last-ReLU chain),
            // skip unknown/non-ReLU/input-fed/fully-stable names (probe-logged
            // — row selection of WHICH layers get refined; never soundness),
            // dedupe, sort by exec order.
            let exec = graph.exec_order().ok()?;
            let pos: HashMap<&str, usize> = exec
                .iter()
                .enumerate()
                .map(|(i, name)| (name.as_str(), i))
                .collect();
            let mut resolved: Vec<(String, String)> = Vec::new();
            for tok in names {
                let pair = if tok.eq_ignore_ascii_case("last") {
                    (last_relu_name.clone(), last_seed.clone())
                } else {
                    let Some(node) = graph.nodes.get(tok) else {
                        if opts.probe {
                            eprintln!("[interm-refine] seed token {tok:?} not in graph — skipped");
                        }
                        continue;
                    };
                    if !matches!(node.layer, Layer::ReLU(_)) {
                        if opts.probe {
                            eprintln!("[interm-refine] seed token {tok:?} is not a ReLU — skipped");
                        }
                        continue;
                    }
                    let Ok(seed) = node.require_unary_input().map(str::to_string) else {
                        continue;
                    };
                    if seed == NETWORK_INPUT {
                        if opts.probe {
                            eprintln!(
                                "[interm-refine] seed token {tok:?} fed by the network input — skipped"
                            );
                        }
                        continue;
                    }
                    (tok.clone(), seed)
                };
                if resolved.iter().any(|(r, _)| *r == pair.0) {
                    continue; // dedupe
                }
                if pair.0 != last_relu_name
                    && !has_unstable(&pair.1)
                    && !has_precomputed_target(&pair.0)
                {
                    if opts.probe {
                        eprintln!(
                            "[interm-refine] named seed relu={} seed={} fully stable — skipped",
                            pair.0, pair.1
                        );
                    }
                    continue;
                }
                resolved.push(pair);
            }
            resolved.sort_by_key(|(r, _)| pos.get(r.as_str()).copied().unwrap_or(usize::MAX));
            if resolved.is_empty() {
                return None;
            }
            resolved
        } else {
            // LAYERS walk: layer 1 = the last ReLU; deeper layers walk the
            // ancestor ReLUs (latest-in-exec-order first), SKIPPING
            // fully-stable ones (measured on resnet_medium: `Relu_51`, the
            // last residual F-branch, is DEAD at the root box — all 2048
            // `Conv_49` uppers < 0 — so the useful second target is the next
            // unstable ancestor). Reversed into exec order afterwards.
            let mut walk: Vec<(String, String)> = vec![(last_relu_name.clone(), last_seed)];
            let mut cursor = walk[0].1.clone();
            while walk.len() < opts.layers {
                let Some((relu, seed)) = find_next_relu_seed_below(graph, &cursor) else {
                    break;
                };
                cursor = seed.clone();
                if has_unstable(&seed) || has_precomputed_target(&relu) {
                    walk.push((relu, seed));
                } else if opts.probe {
                    eprintln!(
                        "[interm-refine] deep seed candidate relu={relu} seed={seed} fully \
                         stable — walking deeper"
                    );
                }
            }
            if opts.probe && walk.len() < opts.layers {
                eprintln!(
                    "[interm-refine] seed chain stopped at {}/{} layers (no unstable ancestor ReLU seed)",
                    walk.len(),
                    opts.layers
                );
            }
            walk.reverse();
            walk
        };

        if opts.clip_resnet && complete_clip_root_bounds.is_some() {
            // Complete Clipping needs two sets of affine rows:
            //   (1) objective layers (root-unstable hidden ReLU inputs), and
            //   (2) every layer named by any live split premise, even when the
            //       premise clamp made that source row inherited-stable.
            // Build their union in execution order. This mirrors DomainClipper's
            // root bank and, crucially, prevents a history constraint on a layer
            // outside the legacy `NY_INTERM_REFINE_LAYERS` walk from vanishing.
            let mut source_relus = std::collections::HashSet::new();
            for history in histories {
                for constraint in
                    scheduled_relu_constraints(graph, history, complete_clip_use_all_history)?
                {
                    source_relus.insert(constraint.node_name().to_string());
                }
            }
            let exec = graph.exec_order().ok()?;
            let positions: HashMap<&str, usize> = exec
                .iter()
                .enumerate()
                .map(|(index, name)| (name.as_str(), index))
                .collect();
            for relu_name in exec {
                let Some(relu) = graph.nodes.get(relu_name) else {
                    continue;
                };
                if !matches!(relu.layer, Layer::ReLU(_)) {
                    continue;
                }
                let Ok(seed_node) = relu.require_unary_input() else {
                    continue;
                };
                if seed_node == NETWORK_INPUT {
                    // Direct-input ReLU premises are handled as identity rows by
                    // the bank adapter; there is no graph node at which to seed
                    // the truncated backward.
                    continue;
                }
                if !source_relus.contains(relu_name.as_str())
                    && !has_unstable(seed_node)
                    && !has_precomputed_target(relu_name)
                {
                    continue;
                }
                if !seeds.iter().any(|(name, _)| name == relu_name) {
                    seeds.push((relu_name.clone(), seed_node.to_string()));
                }
            }
            seeds.sort_by_key(|(relu, _)| {
                positions.get(relu.as_str()).copied().unwrap_or(usize::MAX)
            });
        }

        let mut caches: Vec<HashMap<String, Arc<BoundedTensor>>> = bounds_caches.to_vec();
        let mut infeasible = vec![false; n_domains];
        let mut any_refined = false;
        let mut newly_stable_total = 0usize;
        let mut passes_completed = 0usize;
        let mut deadline_break = false;
        let mut clip_banks: Vec<DomainClipBank> =
            (0..n_domains).map(|_| DomainClipBank::default()).collect();
        if opts.clip_resnet && complete_clip_root_bounds.is_some() {
            initialize_direct_input_sources(
                graph,
                &mut clip_banks,
                histories,
                beta_states,
                constrained_inputs,
                opts.deadline,
                complete_clip_use_all_history,
            );
        }
        let production_root_bank = self.config.enable_clip_interm_domain
            && opts.clip_resnet
            && complete_clip_root_bounds.is_some();
        let root_captures_expected = if production_root_bank { seeds.len() } else { 0 };
        let mut root_captures_refused = 0usize;
        let mut root_reclip_outcome = None;
        let mut root_bank_tightened = false;
        for (relu_name, seed_node) in &seeds {
            // Deadline check between every root-bank or legacy GPU pass.
            if opts
                .deadline
                .is_some_and(|deadline| Instant::now() >= deadline)
            {
                deadline_break = true;
                break;
            }
            let is_last_relu = *relu_name == last_relu_name;
            if production_root_bank {
                let captured = self
                    .capture_complete_clip_root_bank_layer(
                        graph,
                        relu_name,
                        seed_node,
                        &caches,
                        constrained_inputs,
                        histories,
                        beta_states,
                        alpha_states,
                        complete_clip_root_bounds
                            .as_deref()
                            .expect("production_root_bank checked root map"),
                        &mut clip_banks,
                        opts,
                        is_last_relu,
                        complete_clip_use_all_history,
                        &precomputed_decisions,
                    )
                    .is_some();
                if captured {
                    passes_completed += 1;
                } else {
                    root_captures_refused = root_captures_refused.saturating_add(1);
                    if opts
                        .deadline
                        .is_some_and(|deadline| Instant::now() >= deadline)
                    {
                        deadline_break = true;
                        break;
                    }
                }
            } else {
                let pass = self.refine_one_seed_pass(
                    graph,
                    relu_name,
                    seed_node,
                    n_domains,
                    &mut caches,
                    constrained_inputs,
                    histories,
                    beta_states,
                    alpha_states,
                    gpu,
                    opts,
                    is_last_relu,
                    &mut infeasible,
                    complete_clip_root_bounds.as_deref(),
                    &mut clip_banks,
                );
                if let Some((refined, newly_stable)) = pass {
                    any_refined |= refined;
                    newly_stable_total += newly_stable;
                    passes_completed += 1;
                }
            }
        }

        // Root rows do not depend on clip-tightened child caches. Capture the
        // complete layer bank first, then solve every target once against the
        // now-complete scheduled source set. Besides matching DomainClipper's
        // root-bank semantics, this removes the former O(layers²) progressive
        // re-solve/revalidation cost.
        if production_root_bank && passes_completed > 0 && !deadline_break {
            let solve_started = Instant::now();
            let mut stats = RefineStats::default();
            let outcome = reclip_all_captured_targets(
                graph,
                &clip_banks,
                histories,
                constrained_inputs,
                &mut caches,
                opts.deadline,
                &mut stats,
                complete_clip_use_all_history,
            );
            any_refined |= outcome.tightened;
            root_bank_tightened |= outcome.tightened;
            newly_stable_total += stats.newly_stable;
            let batch_complete = complete_clip_batch_completed(
                root_captures_expected,
                passes_completed,
                root_captures_refused,
                Some(outcome),
            );
            if !batch_complete {
                deadline_break = true;
            }
            if opts.probe {
                eprintln!(
                    "[complete-clip-root-bank-solve] history={} targets={} domains_changed={} \
                     targets_completed={} targets_refused={} tightened={} newly_stable={} \
                     max_tighten={:.4} completed={} deadline_interrupted={} elapsed_ms={}",
                    if complete_clip_use_all_history {
                        "all"
                    } else {
                        "first-layer-latest"
                    },
                    outcome.targets_expected,
                    stats.clip_domains,
                    outcome.targets_completed,
                    outcome.targets_refused,
                    stats.neurons_tightened,
                    stats.newly_stable,
                    stats.clip_max_tighten,
                    batch_complete,
                    outcome.deadline_interrupted,
                    solve_started.elapsed().as_millis(),
                );
            }
            root_reclip_outcome = Some(outcome);
        }
        if production_root_bank && opts.probe {
            let (targets_expected, targets_completed, targets_refused, interrupted) =
                root_reclip_outcome.map_or((0, 0, 0, deadline_break), |outcome| {
                    (
                        outcome.targets_expected,
                        outcome.targets_completed,
                        outcome.targets_refused,
                        outcome.deadline_interrupted,
                    )
                });
            eprintln!(
                "[complete-clip-root-bank-summary] history={} captures_expected={} \
                 captures_succeeded={} captures_refused={} solve_invoked={} \
                 targets_expected={} targets_completed={} targets_refused={} \
                 deadline_interrupted={} completed={}",
                if complete_clip_use_all_history {
                    "all"
                } else {
                    "first-layer-latest"
                },
                root_captures_expected,
                passes_completed,
                root_captures_refused,
                root_reclip_outcome.is_some(),
                targets_expected,
                targets_completed,
                targets_refused,
                interrupted,
                complete_clip_batch_completed(
                    root_captures_expected,
                    passes_completed,
                    root_captures_refused,
                    root_reclip_outcome,
                ),
            );
        }
        // ADAPTIVE latch update (#adaptive-refine): a COMPLETED batch (at
        // least one pass ran, no deadline break) that PRODUCED nothing
        // (newly_stable = 0, infeasible-pruned = 0, and no root-bank width
        // tightening) at depth ≥ floor stops refinement for all deeper domains
        // for the rest of the run. The batch's depth band = the MIN premise
        // count over its domains.
        if let Some(latch) = &opts.adaptive_latch {
            if passes_completed > 0 && !deadline_break {
                let n_inf = infeasible.iter().filter(|&&b| b).count();
                let batch_depth = (0..n_domains)
                    .map(|i| beta_states[i].map_or(0, |b| b.entries.len()))
                    .min()
                    .unwrap_or(0);
                if adaptive_should_latch(
                    newly_stable_total,
                    n_inf,
                    batch_depth,
                    opts.adaptive_floor,
                ) && !root_bank_tightened
                {
                    let prev = latch.fetch_min(batch_depth, std::sync::atomic::Ordering::Relaxed);
                    if opts.probe && batch_depth < prev {
                        eprintln!(
                            "[interm-refine-adaptive] LATCHED at depth {batch_depth} \
                             (zero-yield batch of {n_domains}; refinement now skipped for \
                             domains at depth >= {batch_depth})"
                        );
                    }
                }
            }
        }
        if any_refined || infeasible.iter().any(|&b| b) {
            Some(IntermRefineOutcome { caches, infeasible })
        } else {
            None
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn get_or_build_complete_clip_root_template(
        &self,
        graph: &GraphNetwork,
        seed_node: &str,
        selected: &[usize],
        row_of_neuron: &[usize],
        root_bounds: &HashMap<String, Arc<BoundedTensor>>,
        root_input: &BoundedTensor,
        alpha_state: Option<&GraphDomainAlphaState>,
        deadline: Option<Instant>,
    ) -> Option<Arc<SoundCrownRootAffineRows>> {
        if let Some(hit) = self.complete_clip_root_bounds_cache.get_affine_rows(
            graph,
            root_input,
            seed_node,
            selected,
            row_of_neuron.len(),
        ) {
            return Some(hit);
        }
        // A cache miss starts a fresh optional host/device CROWN capture. Its
        // internals poll the exact deadline, and this reserve prevents a new
        // long-running GEMM from being admitted at the edge of the caller's
        // post-BaB budget.
        if !complete_clip_optional_start_budget_available(Instant::now(), deadline) {
            return None;
        }
        let rows = Arc::new(
            capture_sound_crown_root_rows_at_node(
                graph,
                seed_node,
                selected,
                row_of_neuron,
                root_bounds,
                root_input,
                alpha_state,
                deadline,
            )
            .ok()?,
        );
        self.complete_clip_root_bounds_cache.store_affine_rows(
            graph,
            root_input,
            seed_node,
            selected,
            row_of_neuron.len(),
            Arc::clone(&rows),
        );
        Some(rows)
    }

    /// Capture one hidden ReLU into the production DomainClipper root bank.
    /// Unlike the legacy
    /// intermediate-refinement experiment, this does not run a child-specific
    /// identity-seeded CROWN backward.  It selects the child's objectives from
    /// the current cache, binds the reusable root affine template to each exact
    /// history, and defers constrained concretization until every scheduled
    /// source layer has been captured. This is the amortized path used by the
    /// scored preset.
    #[allow(clippy::too_many_arguments)]
    fn capture_complete_clip_root_bank_layer(
        &self,
        graph: &GraphNetwork,
        relu_name: &str,
        seed_node: &str,
        caches: &[HashMap<String, Arc<BoundedTensor>>],
        constrained_inputs: &[BoundedTensor],
        histories: &[&GraphSplitHistory],
        beta_states: &[Option<&GraphBetaState>],
        alpha_states: &[Option<&GraphDomainAlphaState>],
        root_bounds: &HashMap<String, Arc<BoundedTensor>>,
        clip_banks: &mut [DomainClipBank],
        opts: &IntermRefineOptions,
        is_last_relu: bool,
        use_all_history: bool,
        precomputed_decisions: &[Option<Arc<CompleteClipDecisionIndices>>],
    ) -> Option<()> {
        let n_domains = caches.len();
        if n_domains == 0
            || constrained_inputs.len() != n_domains
            || histories.len() != n_domains
            || beta_states.len() != n_domains
            || alpha_states.len() != n_domains
            || clip_banks.len() != n_domains
            || precomputed_decisions.len() != n_domains
            || opts
                .deadline
                .is_some_and(|deadline| Instant::now() >= deadline)
        {
            return None;
        }
        let pre_dim = caches.first()?.get(seed_node)?.len();
        if pre_dim == 0
            || validate_selective_row_budget(pre_dim, pre_dim, n_domains, true).is_none()
        {
            return None;
        }

        let mut candidate_mask = vec![false; pre_dim];
        for cache in caches.iter() {
            let entry = cache.get(seed_node)?;
            if entry.len() != pre_dim {
                return None;
            }
            for (index, (&lower, &upper)) in
                entry.lower().iter().zip(entry.upper().iter()).enumerate()
            {
                if index.is_multiple_of(SELECTIVE_DEADLINE_POLL_STRIDE)
                    && opts
                        .deadline
                        .is_some_and(|deadline| Instant::now() >= deadline)
                {
                    return None;
                }
                if lower < 0.0 && upper > 0.0 {
                    candidate_mask[index] = true;
                }
            }
        }
        for history in histories {
            for constraint in scheduled_relu_constraints(graph, history, use_all_history)? {
                if constraint.node_name() == relu_name {
                    let neuron = constraint.neuron_idx();
                    if neuron >= pre_dim {
                        return None;
                    }
                    candidate_mask[neuron] = true;
                }
            }
        }
        for snapshot in precomputed_decisions.iter().flatten() {
            if let Some(neurons) = snapshot.get(relu_name) {
                for &neuron in neurons.iter() {
                    if neuron >= pre_dim {
                        return None;
                    }
                    candidate_mask[neuron] = true;
                }
            }
        }
        let candidates: Vec<usize> = candidate_mask
            .into_iter()
            .enumerate()
            .filter_map(|(neuron, selected)| selected.then_some(neuron))
            .collect();
        if candidates.is_empty() {
            return None;
        }
        let (selected, domain_objective_neurons, _) = select_complete_clip_rows(
            graph,
            caches,
            histories,
            seed_node,
            relu_name,
            &candidates,
            pre_dim,
            opts.selective_topk,
            is_last_relu
                .then_some(opts.margin_weights.as_deref())
                .flatten(),
            Some(precomputed_decisions),
            opts.deadline,
            use_all_history,
        )?;
        let mut row_of = vec![usize::MAX; pre_dim];
        for (row, &neuron) in selected.iter().enumerate() {
            row_of[neuron] = row;
        }
        let capture_started = Instant::now();
        let root_rows = self.get_or_build_complete_clip_root_template(
            graph,
            seed_node,
            &selected,
            &row_of,
            root_bounds,
            constrained_inputs.first()?,
            alpha_states.first().copied().flatten(),
            opts.deadline,
        )?;
        let shared_seed_node: Arc<str> = Arc::from(seed_node);
        let shared_selected: Arc<[usize]> = Arc::from(selected.as_slice());
        let shared_row_of: Arc<[usize]> = Arc::from(row_of.as_slice());

        for domain in 0..n_domains {
            let Some(beta) = beta_states[domain] else {
                clip_banks[domain].disabled = true;
                continue;
            };
            if !beta_state_matches_pure_relu_history(histories[domain], beta) {
                clip_banks[domain].disabled = true;
                continue;
            }
            let flat = constrained_inputs[domain].flatten();
            let (Some(in_lo), Some(in_hi)) = (
                flat.lower().as_slice_memory_order(),
                flat.upper().as_slice_memory_order(),
            ) else {
                clip_banks[domain].disabled = true;
                continue;
            };
            let Some((pass, token)) = bind_root_sound_crown_rows_to_history(
                graph.cut_fold_scope(),
                in_lo,
                in_hi,
                histories[domain],
                seed_node,
                &selected,
                &row_of,
                &root_rows,
                opts.deadline,
            )
            .ok()
            .and_then(|rows| mint_certified_affine_enclosure(rows, opts.deadline).ok()) else {
                clip_banks[domain].disabled = true;
                continue;
            };
            let Some(objective_rows) =
                exact_objective_rows(&domain_objective_neurons[domain], &row_of)
            else {
                clip_banks[domain].disabled = true;
                continue;
            };
            let is_target = !objective_rows.is_empty();
            clip_banks[domain].layers.insert(
                relu_name.to_string(),
                CertifiedLayerCapture {
                    seed_node: Arc::clone(&shared_seed_node),
                    selected_neurons: Arc::clone(&shared_selected),
                    row_of_neuron: Arc::clone(&shared_row_of),
                    objective_rows,
                    pass,
                    token,
                },
            );
            if is_target
                && !clip_banks[domain]
                    .target_order
                    .iter()
                    .any(|name| name == relu_name)
            {
                clip_banks[domain].target_order.push(relu_name.to_string());
            }
        }

        if opts.probe {
            eprintln!(
                "[complete-clip-root-bank-capture] relu={relu_name} history={} rows={} \
                 elapsed_ms={}",
                if use_all_history {
                    "all"
                } else {
                    "first-layer-latest"
                },
                selected.len(),
                capture_started.elapsed().as_millis(),
            );
        }
        Some(())
    }

    /// ONE seed layer's refinement pass over the whole domain batch, applied
    /// in place into `caches` (a per-layer refusal returns `None` and leaves
    /// the caches untouched — sound). Returns `(any_domain_refined,
    /// newly_stable_count)` — the count feeds the adaptive-schedule latch
    /// (#adaptive-refine).
    #[allow(clippy::too_many_arguments)]
    fn refine_one_seed_pass(
        &self,
        graph: &GraphNetwork,
        relu_name: &str,
        seed_node: &str,
        n_domains: usize,
        caches: &mut [HashMap<String, Arc<BoundedTensor>>],
        constrained_inputs: &[BoundedTensor],
        histories: &[&GraphSplitHistory],
        beta_states: &[Option<&GraphBetaState>],
        alpha_states: &[Option<&GraphDomainAlphaState>],
        gpu: &dyn ny_core::GpuCrownBackward,
        opts: &IntermRefineOptions,
        is_last_relu: bool,
        infeasible: &mut [bool],
        complete_clip_root_bounds: Option<&HashMap<String, Arc<BoundedTensor>>>,
        clip_banks: &mut [DomainClipBank],
    ) -> Option<(bool, usize)> {
        if n_domains == 0
            || caches.len() != n_domains
            || constrained_inputs.len() != n_domains
            || histories.len() != n_domains
            || beta_states.len() != n_domains
            || alpha_states.len() != n_domains
            || infeasible.len() != n_domains
            || clip_banks.len() != n_domains
        {
            return None;
        }
        let probe = opts.probe;
        let refuse = |reason: &str| -> Option<(bool, usize)> {
            if probe {
                eprintln!(
                    "[interm-refine] pass REFUSED seed={seed_node} relu={relu_name}: {reason}"
                );
            }
            None
        };
        let Some(seed_entry) = caches[0].get(seed_node) else {
            return refuse("no seed entry in cache");
        };
        let pre_dim = seed_entry.len();
        // Named seeds (`NY_INTERM_REFINE_SEEDS`) are exempt from the dim cap —
        // naming a conv-wide layer is the deliberate ask; the deep row cap and
        // the hard `n_rows·pre_dim` guard below still bound cost.
        if pre_dim == 0
            || (opts.seeds.is_none()
                && pre_dim > opts.max_dim
                && !(opts.clip_resnet && opts.selective_topk > 0))
        {
            return refuse(&format!("pre_dim={pre_dim} outside (0, {}]", opts.max_dim));
        }
        // Account the simultaneously-live selected/premise/candidate/scored/
        // output vectors before the first row-table allocation. The same hard
        // cap also covers the deep width/rest/keep representation.
        if opts
            .deadline
            .is_some_and(|deadline| Instant::now() >= deadline)
            || validate_selective_row_budget(
                pre_dim,
                pre_dim,
                caches.len(),
                opts.selective_topk > 0,
            )
            .is_none()
        {
            return refuse("selective row work exceeds deadline/resource cap");
        }
        let t0 = Instant::now();

        // ROW SELECTION (cost, not soundness — see
        // `interm_refine_unstable_rows_only`): default = the union of
        // inherited-UNSTABLE neurons (stable slopes are already exact; measured
        // bound-identical to all-rows at ~3x less GPU). The seed is shared
        // across the batch, so union over domains (per-domain intersection
        // keeps each domain's own inherited values elsewhere). The prune lane
        // force-includes every domain's split-premise neurons: clamped premises
        // are STABLE in the inherited entry, so the unstable arm alone would
        // exclude exactly the rows that can prove infeasibility.
        let mut selected = Vec::new();
        selected.try_reserve_exact(pre_dim).ok()?;
        selected.resize(pre_dim, !opts.unstable_rows_only);
        if opts.unstable_rows_only {
            for cache in caches.iter() {
                if opts
                    .deadline
                    .is_some_and(|deadline| Instant::now() >= deadline)
                {
                    return refuse("deadline during unstable-row selection");
                }
                let Some(entry) = cache.get(seed_node) else {
                    continue;
                };
                if entry.len() != pre_dim {
                    // Heterogeneous seed entry — refuse the whole batch.
                    return refuse("heterogeneous seed entry");
                }
                for (j, (&l, &u)) in entry.lower().iter().zip(entry.upper().iter()).enumerate() {
                    if j.is_multiple_of(SELECTIVE_DEADLINE_POLL_STRIDE)
                        && opts
                            .deadline
                            .is_some_and(|deadline| Instant::now() >= deadline)
                    {
                        return refuse("deadline during unstable-row scan");
                    }
                    if l < 0.0 && u > 0.0 {
                        selected[j] = true;
                    }
                }
            }
        }
        // Force-include every domain's split-premise neurons: the prune lane needs them
        // (clamped premises are STABLE ⇒ excluded by the unstable arm), and the
        // #clip-interm-resnet-batched clip needs them as CONSTRAINT SOURCES (a missing
        // premise row ⇒ that split's half-space is dropped ⇒ the clip loses its
        // tightening). Both are seed-layer splits at `relu_name`.
        let mut premise = Vec::new();
        premise.try_reserve_exact(pre_dim).ok()?;
        premise.resize(pre_dim, false);
        if opts.prune || opts.clip_resnet {
            for history in histories {
                if opts
                    .deadline
                    .is_some_and(|deadline| Instant::now() >= deadline)
                {
                    return refuse("deadline during premise-row selection");
                }
                for (entry_index, constraint) in history
                    .constraints
                    .iter()
                    .filter(|constraint| constraint.node_name() == relu_name)
                    .enumerate()
                {
                    if entry_index.is_multiple_of(SELECTIVE_DEADLINE_POLL_STRIDE)
                        && opts
                            .deadline
                            .is_some_and(|deadline| Instant::now() >= deadline)
                    {
                        return refuse("deadline during premise-row scan");
                    }
                    if constraint.neuron_idx() < pre_dim {
                        premise[constraint.neuron_idx()] = true;
                        selected[constraint.neuron_idx()] = true;
                    }
                }
            }
        }
        let mut sel = Vec::new();
        sel.try_reserve_exact(pre_dim).ok()?;
        for (j, &is_selected) in selected.iter().enumerate() {
            if j.is_multiple_of(SELECTIVE_DEADLINE_POLL_STRIDE)
                && opts
                    .deadline
                    .is_some_and(|deadline| Instant::now() >= deadline)
            {
                return refuse("deadline during candidate materialization");
            }
            if is_selected {
                sel.push(j);
            }
        }
        // The candidate vector now owns the selection result; release the
        // dense mask before allocating scored/output vectors so the checked
        // live-byte model remains conservative.
        drop(selected);

        // Winner-style selective clip: αβ-CROWN's TinyImageNet configuration
        // uses top-20 objectives per layer.  NY's wide seed is
        // shared across domains, so select one batch-wide top-K (max score over
        // domains) and keep split-premise source rows additively.  Unselected
        // rows retain their inherited enclosures.
        let mut domain_objective_neurons = vec![sel.clone(); n_domains];
        if opts.clip_resnet {
            let (complete_sel, objectives, exact_premise) = select_complete_clip_rows(
                graph,
                caches,
                histories,
                seed_node,
                relu_name,
                &sel,
                pre_dim,
                opts.selective_topk,
                is_last_relu
                    .then_some(opts.margin_weights.as_deref())
                    .flatten(),
                None,
                opts.deadline,
                true,
            )?;
            sel = complete_sel;
            premise = exact_premise;
            domain_objective_neurons = objectives;
        } else if opts.selective_topk > 0 {
            sel = select_intermediate_objective_rows_with_deadline(
                caches,
                seed_node,
                &sel,
                &premise,
                opts.selective_topk,
                is_last_relu
                    .then_some(opts.margin_weights.as_deref())
                    .flatten(),
                opts.deadline,
            );
            domain_objective_neurons = vec![sel.clone(); n_domains];
        }

        // DEEP-layer row cap (cost-only): keep premise rows, then top-K by
        // inherited width (widest = most refinement headroom), index tie-break.
        if !is_last_relu
            && sel.len() > opts.deep_max_rows
            && !(opts.clip_resnet && opts.selective_topk > 0)
        {
            let mut width = Vec::new();
            width.try_reserve_exact(pre_dim).ok()?;
            width.resize(pre_dim, 0.0f32);
            for cache in caches.iter() {
                if opts
                    .deadline
                    .is_some_and(|deadline| Instant::now() >= deadline)
                {
                    return refuse("deadline during deep-row width scan");
                }
                let Some(entry) = cache.get(seed_node) else {
                    continue;
                };
                if entry.len() != pre_dim {
                    continue;
                }
                for (j, (&l, &u)) in entry.lower().iter().zip(entry.upper().iter()).enumerate() {
                    if j.is_multiple_of(SELECTIVE_DEADLINE_POLL_STRIDE)
                        && opts
                            .deadline
                            .is_some_and(|deadline| Instant::now() >= deadline)
                    {
                        return refuse("deadline during deep-row width fold");
                    }
                    if l.is_finite() && u.is_finite() {
                        width[j] = width[j].max(u - l);
                    }
                }
            }
            let mut keep = Vec::new();
            let mut rest = Vec::new();
            keep.try_reserve_exact(sel.len()).ok()?;
            rest.try_reserve_exact(sel.len()).ok()?;
            for (position, &j) in sel.iter().enumerate() {
                if position.is_multiple_of(SELECTIVE_DEADLINE_POLL_STRIDE)
                    && opts
                        .deadline
                        .is_some_and(|deadline| Instant::now() >= deadline)
                {
                    return refuse("deadline during deep-row partition");
                }
                if premise[j] {
                    keep.push(j);
                } else {
                    rest.push(j);
                }
            }
            let mut past_deadline = || {
                opts.deadline
                    .is_some_and(|deadline| Instant::now() >= deadline)
            };
            if crate::complete_clip::deadline_heapsort_by(
                &mut rest,
                &mut past_deadline,
                "deep selective width sort",
                |&a, &b| {
                    width[b]
                        .partial_cmp(&width[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(a.cmp(&b))
                },
            )
            .is_err()
            {
                return refuse("deadline during deep-row score sort");
            }
            let room = opts.deep_max_rows.saturating_sub(keep.len());
            keep.extend(rest.into_iter().take(room));
            let mut past_deadline = || {
                opts.deadline
                    .is_some_and(|deadline| Instant::now() >= deadline)
            };
            if crate::complete_clip::deadline_heapsort_by(
                &mut keep,
                &mut past_deadline,
                "deep selective index sort",
                usize::cmp,
            )
            .is_err()
            {
                return refuse("deadline during deep-row index sort");
            }
            sel = keep;
        }
        let n_rows = sel.len();
        if n_rows == 0 || n_rows.saturating_mul(pre_dim) > (1 << 24) {
            // Diagnostic detail for the zero-row case: what does the seed
            // entry actually look like?
            let (mut n_l_neg, mut n_u_pos, mut n_nan) = (0usize, 0usize, 0usize);
            let (mut min_l, mut max_u) = (f32::INFINITY, f32::NEG_INFINITY);
            if let Some(entry) = caches[0].get(seed_node) {
                for (&l, &u) in entry.lower().iter().zip(entry.upper().iter()) {
                    if l.is_nan() || u.is_nan() {
                        n_nan += 1;
                    }
                    if l < 0.0 {
                        n_l_neg += 1;
                    }
                    if u > 0.0 {
                        n_u_pos += 1;
                    }
                    min_l = min_l.min(l);
                    max_u = max_u.max(u);
                }
            }
            return refuse(&format!(
                "n_rows={n_rows} (zero or seed too large; entry: l<0 {n_l_neg}/{pre_dim}, \
                 u>0 {n_u_pos}/{pre_dim}, nan {n_nan}, min_l={min_l:.4}, max_u={max_u:.4})"
            ));
        }

        // Identity seed: row r bounds pre-activation `sel[r]` of the seed node.
        let mut rows = vec![0.0f32; n_rows * pre_dim];
        for (r, &j) in sel.iter().enumerate() {
            rows[r * pre_dim + j] = 1.0;
        }
        let seed = ny_core::GpuCrownSeed {
            lower_a: rows.clone().into(),
            upper_a: rows.into(),
            lower_b: vec![0.0f32; n_rows].into(),
            upper_b: vec![0.0f32; n_rows].into(),
            num_specs: n_rows,
            current_dim: pre_dim,
        };

        // Truncated per-domain preps: same alpha-bridge/β/extraction machinery
        // as the margin backward, start node = the seed node. β for the SEED
        // ReLU is automatically EXCLUDED (its Activation is not in the
        // truncated stack — the seed-layer splits live in the inherited clamp).
        // The preps are independent across domains — rayon fan-out (the
        // extraction's conv weight_col conversion dominates the refinement
        // wall; GPU calls still serialize on the device mutex). Preps read the
        // CURRENT caches: a deeper pass's refinement cascades into this one.
        let mut preps: Vec<Option<ResnetDomainPrep>> = {
            use rayon::iter::{IntoParallelIterator, ParallelIterator};
            let caches_ref: &[HashMap<String, Arc<BoundedTensor>>] = caches;
            let allow_pure_chain = crate::network::bab_chain_wide_enabled();
            // #extract-skeleton increment 3: the truncated-stack skeleton for
            // THIS seed node comes from the verifier-level cross-batch cache
            // (`self.skeleton_cache`, keyed `(seed_node, allow_pure_chain)`) —
            // a hit re-validates `matches_graph` and shares the same `Arc`
            // across every pass and batch; a miss/stale entry rebuilds the
            // increment-2 per-pass skeleton (exemplar = domain 0). The "preps
            // read CURRENT caches" contract above is preserved: only
            // graph-static weights are shared through the skeleton; every
            // bounds-derived slot/table is re-folded per domain from
            // `caches_ref[i]` inside `prep_resnet_domain_with`. `None`
            // (kill-switch / build refusal) keeps every domain on the legacy
            // extraction, byte-identically (fail closed).
            let skeleton = (n_domains > 0)
                .then(|| {
                    self.skeleton_cache
                        .get_or_build(graph, seed_node, allow_pure_chain, || {
                            build_call_skeleton(
                                graph,
                                seed_node,
                                &caches_ref[0],
                                &constrained_inputs[0],
                                alpha_states[0],
                                allow_pure_chain,
                            )
                        })
                })
                .flatten();
            let skeleton = skeleton.as_deref();
            (0..n_domains)
                .into_par_iter()
                .map(|i| {
                    let _rayon_task_guard = crate::faer_parallelism::RayonTaskGuard::new();
                    prep_resnet_domain_with(
                        skeleton,
                        graph,
                        seed_node,
                        &caches_ref[i],
                        &constrained_inputs[i],
                        beta_states[i],
                        alpha_states[i],
                        allow_pure_chain,
                        // #image-node-crown flags: legacy #interm-refine lane —
                        // never BN, never a frozen stop (byte-identical).
                        false,
                        false,
                    )
                })
                .collect()
        };
        let prep_ms = t0.elapsed().as_millis();
        let idxs: Vec<usize> = (0..n_domains).filter(|&i| preps[i].is_some()).collect();
        if idxs.is_empty() {
            return refuse("every domain's truncated-stack prep refused");
        }

        // #alpha-prime (dark, `NY_INTERM_REFINE_ALPHA=1`, last-ReLU pass only):
        // REUSE a stored α′ (optimize-once at the first batch) by writing it
        // into every domain's truncated segments BEFORE the refinement
        // backward — sound (any α ∈ [0,1] is a valid lower ReLU slope; only
        // this domain's own unstable neurons are touched). When no usable α′
        // is stored yet (or under `NY_INTERM_REFINE_ALPHA_REOPT=1`), mark the
        // ascent pending — it runs after the iterate-0 backward below.
        let mut alpha_pending = false;
        // #ab-parity-interm: the per-target lane runs a FRESH ascent every batch
        // (no once-per-process shared-α reuse — the shared α′ store is bypassed).
        if opts.per_target && opts.alpha_iters > 0 && is_last_relu {
            alpha_pending = true;
        } else if opts.alpha_iters > 0 && is_last_relu {
            if let Some(store) = &opts.alpha_store {
                let mut reuse: Option<(Vec<Vec<f32>>, Vec<Vec<bool>>)> = None;
                {
                    let stored = store.lock().unwrap_or_else(|e| e.into_inner());
                    match stored.as_ref() {
                        Some(ap) if !opts.alpha_reopt => {
                            let key_ok = preps[idxs[0]].as_ref().is_some_and(|p| {
                                alpha_prime_matches(ap, seed_node, pre_dim, &p.relu_names)
                            });
                            if ap.improved && key_ok {
                                reuse = Some((ap.slopes.clone(), ap.stepped.clone()));
                            } else if probe && !key_ok {
                                eprintln!(
                                    "[interm-refine-alpha] stored α′ key mismatch — reuse skipped"
                                );
                            }
                        }
                        Some(_) => alpha_pending = true, // REOPT: ascend every batch
                        None => alpha_pending = true,
                    }
                }
                if let Some((slopes, stepped)) = reuse {
                    let mut applied = 0usize;
                    for &i in &idxs {
                        if let Some(p) = preps[i].as_mut() {
                            write_alpha_prime(&mut p.segments, &slopes, &stepped);
                            applied += 1;
                        }
                    }
                    if probe {
                        eprintln!(
                            "[interm-refine-alpha] reuse applied doms={applied}/{}",
                            idxs.len()
                        );
                    }
                }
            }
        }
        let t_gpu = Instant::now();

        // ONE batched call through the same wide machinery as the margin
        // backward (#wide-chunk: under `NY_INTERM_REFINE_WIDE_MAX_N=k`, split
        // into domain chunks of `max(1, k / n_rows)` so each wide pass stays
        // under the device's binding/dispatch caps instead of failing whole and
        // falling serial — see `IntermRefineOptions::wide_max_n`); on any
        // batched error, serial per-domain fallback for that chunk's domains;
        // per-domain failures keep that domain's inherited bounds (sound).
        let mut per_domain: Vec<Option<ny_core::GpuCrownResult>> =
            (0..n_domains).map(|_| None).collect();
        // Complete Clipping uses independently checked root templates.  The
        // child backward below is only the ordinary scalar intermediate
        // refinement, so do not copy its full input coefficient frontier back
        // to the host.  The old resident-coefficient adapter remains covered by
        // differential tests, but it is not clipping authority.
        let do_clip = opts.clip_resnet;
        let chunk_doms = wide_chunk_domains(opts.wide_max_n, n_rows, idxs.len());
        let mut batched_ok = true;
        for chunk in idxs.chunks(chunk_doms) {
            let refs: Vec<ny_core::GpuResnetBatchedDomainRef> = chunk
                .iter()
                .map(|&i| {
                    let p = preps[i].as_ref().expect("idxs holds Some preps only");
                    ny_core::GpuResnetBatchedDomainRef {
                        segments: &p.segments,
                        input_lower: &p.in_lo,
                        input_upper: &p.in_hi,
                        beta_signed: &p.beta_signed,
                        frontier_abs: &p.frontier_abs,
                        node_abs: &p.node_abs,
                    }
                })
                .collect();
            match gpu.crown_backward_gpu_resnet_sound_beta_batched(&refs, &seed) {
                Ok(results) if results.len() == chunk.len() => {
                    for (&i, r) in chunk.iter().zip(results) {
                        per_domain[i] = Some(r);
                    }
                }
                _ => {
                    batched_ok = false;
                    for &i in chunk {
                        let p = preps[i].as_ref().expect("idxs holds Some preps only");
                        if let Ok(r) = gpu.crown_backward_gpu_resnet_sound_beta(
                            &p.segments,
                            &seed,
                            &p.in_lo,
                            &p.in_hi,
                            &p.beta_signed,
                            &p.frontier_abs,
                            &p.node_abs,
                        ) {
                            per_domain[i] = Some(r);
                        }
                    }
                }
            }
        }

        // #alpha-prime ASCENT (dark): optimize a DEDICATED α′ against the
        // refinement objective on top of the iterate-0 (borrowed-α) backward
        // above, best-merging every sound iterate into `per_domain`; the
        // resulting α′ is stored for reuse by later batches (unless REOPT).
        // Skipped when the batched call fell back to serial (rare error arm).
        if alpha_pending
            && batched_ok
            && opts
                .deadline
                .is_none_or(|deadline| Instant::now() < deadline)
        {
            let row_widths: Vec<f32> = {
                let entry = idxs
                    .iter()
                    .find_map(|&i| caches[i].get(seed_node))
                    .filter(|e| e.len() == pre_dim);
                match entry {
                    Some(e) => {
                        let lo: Vec<f32> = e.lower().iter().copied().collect();
                        let hi: Vec<f32> = e.upper().iter().copied().collect();
                        sel.iter()
                            .map(|&j| {
                                let (l, u) = (lo[j], hi[j]);
                                if l.is_finite() && u.is_finite() {
                                    u - l
                                } else {
                                    0.0
                                }
                            })
                            .collect()
                    }
                    None => vec![0.0; sel.len()],
                }
            };
            if opts.per_target {
                // #ab-parity-interm: per-target decoupled ascent (each target row
                // its own α + own bound). Effect is merged directly into
                // `per_domain`; nothing is stored (reuse is dropped under the gate).
                alpha_prime_ascent_per_target(
                    gpu,
                    &mut preps,
                    &idxs,
                    &seed,
                    &mut per_domain,
                    pre_dim,
                    &sel,
                    &row_widths,
                    opts,
                    &|| {
                        opts.deadline
                            .is_some_and(|deadline| Instant::now() >= deadline)
                    },
                );
            } else if let Some(ap) = alpha_prime_ascent(
                gpu,
                &mut preps,
                &idxs,
                &seed,
                &mut per_domain,
                seed_node,
                pre_dim,
                &sel,
                &row_widths,
                opts,
                &|| {
                    opts.deadline
                        .is_some_and(|deadline| Instant::now() >= deadline)
                },
            ) {
                if !opts.alpha_reopt {
                    if let Some(store) = &opts.alpha_store {
                        *store.lock().unwrap_or_else(|e| e.into_inner()) = Some(ap);
                    }
                }
            }
        }

        let gpu_ms = t_gpu.elapsed().as_millis();

        // Row index of each selected neuron (usize::MAX = unselected) — the
        // prune test's map from a premise neuron to its refined row.
        let mut row_of = vec![usize::MAX; pre_dim];
        for (row, &j) in sel.iter().enumerate() {
            row_of[j] = row;
        }

        // Build ONE root-valid, host-checked affine template for this selected
        // layer. Root rows are globally valid on the full input box and can be
        // rebound to every ReLU child; this is DomainClipper's central
        // amortization and avoids the old per-child CPU backward wall.
        let root_template = if do_clip {
            match (complete_clip_root_bounds, constrained_inputs.first()) {
                (Some(root_bounds), Some(root_input)) => self
                    .get_or_build_complete_clip_root_template(
                        graph,
                        seed_node,
                        &sel,
                        &row_of,
                        root_bounds,
                        root_input,
                        alpha_states.first().copied().flatten(),
                        opts.deadline,
                    ),
                _ => None,
            }
        } else {
            None
        };

        if let Some(root_rows) = root_template.as_ref() {
            let shared_seed_node: Arc<str> = Arc::from(seed_node);
            let shared_selected: Arc<[usize]> = Arc::from(sel.as_slice());
            let shared_row_of: Arc<[usize]> = Arc::from(row_of.as_slice());
            for domain in 0..n_domains {
                let Some(beta) = beta_states[domain] else {
                    clip_banks[domain].disabled = true;
                    continue;
                };
                if !beta_state_matches_pure_relu_history(histories[domain], beta) {
                    clip_banks[domain].disabled = true;
                    continue;
                }
                let flat = constrained_inputs[domain].flatten();
                let (Some(in_lo), Some(in_hi)) = (
                    flat.lower().as_slice_memory_order(),
                    flat.upper().as_slice_memory_order(),
                ) else {
                    clip_banks[domain].disabled = true;
                    continue;
                };
                let sealed = match bind_root_sound_crown_rows_to_history(
                    graph.cut_fold_scope(),
                    in_lo,
                    in_hi,
                    histories[domain],
                    seed_node,
                    &sel,
                    &row_of,
                    root_rows,
                    opts.deadline,
                ) {
                    Ok(rows) => rows,
                    Err(_) => {
                        clip_banks[domain].disabled = true;
                        continue;
                    }
                };
                let (pass, token) = match mint_certified_affine_enclosure(sealed, opts.deadline) {
                    Ok(authority) => authority,
                    Err(_) => {
                        clip_banks[domain].disabled = true;
                        continue;
                    }
                };
                let Some(objective_rows) =
                    exact_objective_rows(&domain_objective_neurons[domain], &row_of)
                else {
                    clip_banks[domain].disabled = true;
                    continue;
                };
                let is_target = !objective_rows.is_empty();
                clip_banks[domain].layers.insert(
                    relu_name.to_string(),
                    CertifiedLayerCapture {
                        seed_node: Arc::clone(&shared_seed_node),
                        selected_neurons: Arc::clone(&shared_selected),
                        row_of_neuron: Arc::clone(&shared_row_of),
                        objective_rows,
                        pass,
                        token,
                    },
                );
                if is_target
                    && !clip_banks[domain]
                        .target_order
                        .iter()
                        .any(|name| name == relu_name)
                {
                    clip_banks[domain].target_order.push(relu_name.to_string());
                }
            }
        }

        // Prune test + intersect into the caches (pre entry + the ReLU's post
        // entry), in place.
        let mut stats = RefineStats::default();
        let mut any_refined = false;
        for i in 0..n_domains {
            if per_domain[i].is_none() {
                continue;
            }
            // INFEASIBILITY PRUNE (dark, `NY_INTERM_REFINE_PRUNE=1` — see
            // module docs): the RAW refined enclosure (pre-intersection, no
            // seed-layer premises folded) contradicting one of the domain's
            // own split premises beyond tol proves the subdomain empty. Uses the
            // scalar backward bound (pre-clip) so its behaviour is byte-identical
            // whether or not the clip runs.
            if opts.prune && !infeasible[i] {
                let r = per_domain[i].as_ref().expect("checked Some");
                if r.lower_bounds.len() == sel.len() && r.upper_bounds.len() == sel.len() {
                    if let Some(bs) = beta_states[i] {
                        if let Some((j, sign, margin)) = premise_contradiction(
                            bs,
                            relu_name,
                            &row_of,
                            &r.lower_bounds,
                            &r.upper_bounds,
                            opts.prune_tol,
                        ) {
                            infeasible[i] = true;
                            stats.infeasible += 1;
                            if probe {
                                eprintln!(
                                    "[interm-refine-prune] relu={relu_name} dom={i} neuron={j} \
                                     premise={} margin={margin:.6}",
                                    if sign > 0.0 { "active" } else { "inactive" },
                                );
                            }
                        }
                    }
                }
            }
            let r = per_domain[i].as_ref().expect("checked Some");
            if apply_refinement(
                &mut caches[i],
                seed_node,
                relu_name,
                r,
                pre_dim,
                &sel,
                opts.deadline,
                &mut stats,
            ) {
                any_refined = true;
                stats.domains_refined += 1;
            }
        }

        if do_clip
            && reclip_all_captured_targets(
                graph,
                clip_banks,
                histories,
                constrained_inputs,
                caches,
                opts.deadline,
                &mut stats,
                true,
            )
            .tightened
        {
            any_refined = true;
        }

        // #ab-parity-interm FC-head BOX-WIDTH PROBE (`NY_AB_INTERM_PROBE=1`,
        // stderr-only, read-only): the seed node's TOTAL refined box width
        // Σ_j (u_j − l_j) over ALL pre-activation neurons — directly comparable
        // to the measured CROWN 419.27 head width. Read from the replay domain's
        // REFINED cache (post-apply). Runs for both the shared (OFF) and
        // per-target (ON) modes; prints once at the ROOT batch and once at the
        // first mid-depth (≥ 5 split premises) batch.
        if opts.interm_box_probe && is_last_relu {
            if let Some(&ridx) = idxs.first() {
                if let Some(entry) = caches[ridx].get(seed_node) {
                    let width: f64 = entry
                        .lower()
                        .iter()
                        .zip(entry.upper().iter())
                        .filter(|(l, u)| l.is_finite() && u.is_finite())
                        .map(|(&l, &u)| (u - l) as f64)
                        .sum();
                    let batch_depth = beta_states
                        .iter()
                        .flatten()
                        .map(|bs| bs.entries.len())
                        .max()
                        .unwrap_or(0);
                    let mode = if opts.per_target {
                        "per-target"
                    } else {
                        "shared"
                    };
                    static ROOT_ONCE: std::sync::Once = std::sync::Once::new();
                    let did_root = !ROOT_ONCE.is_completed();
                    ROOT_ONCE.call_once(|| {
                        eprintln!(
                            "[ab-interm-probe] ROOT seed={seed_node} mode={mode} pre_dim={pre_dim} \
                             depth={batch_depth} fc_head_box_width={width:.4}"
                        );
                    });
                    if !did_root && batch_depth >= 5 {
                        static MID_ONCE: std::sync::Once = std::sync::Once::new();
                        MID_ONCE.call_once(|| {
                            eprintln!(
                                "[ab-interm-probe] MID seed={seed_node} mode={mode} pre_dim={pre_dim} \
                                 depth={batch_depth} fc_head_box_width={width:.4}"
                            );
                        });
                    }
                }
            }
        }
        if probe && do_clip {
            eprintln!(
                "[clip-resnet-batched] relu={relu_name} batched_backward=1 \
                 clip_domains={} nodes_changed={} max_tighten={:.4} refused_rows={} \
                 guard_reverts={}",
                stats.clip_domains,
                stats.clip_domains,
                stats.clip_max_tighten,
                stats.clip_refused_rows,
                stats.clip_guard_reverts,
            );
        }
        if probe {
            eprintln!(
                "[interm-refine] seed={seed_node} relu={relu_name} pre_dim={pre_dim} rows={n_rows} \
                 domains={}/{n_domains} chunk_doms={chunk_doms} batched_ok={batched_ok} \
                 tightened={} newly_stable={} crossings_kept={} \
                 infeasible={} width {:.3}->{:.3} ({}ms: prep={prep_ms} gpu={gpu_ms})",
                stats.domains_refined,
                stats.neurons_tightened,
                stats.newly_stable,
                stats.crossings_kept,
                stats.infeasible,
                stats.width_before,
                stats.width_after,
                t0.elapsed().as_millis()
            );
        }
        Some((any_refined, stats.newly_stable))
    }
}

/// #clip-interm-resnet-batched: one domain's seed-node input-relative rows, with the
/// certified per-coefficient error already folded OUTWARD into the bias over that
/// domain's own input box. Row `r` (in `sel` order) is the enclosure
/// `lower_a[r]·x + lower_b[r] ≤ z_{sel[r]}(x) ≤ upper_a[r]·x + upper_b[r]` for every
/// `x` in the domain box; `refused[r]` marks a row whose fold produced a non-finite
/// penalty (kept inherited — sound, no clip for that neuron).
//
// This proposal path stays compiled for differential/provenance tests while
// production authority remains quarantined.
#[allow(dead_code)]
struct FoldedSeedRows {
    lower_a: Vec<f32>,
    upper_a: Vec<f32>,
    lower_b: Vec<f32>,
    upper_b: Vec<f32>,
    refused: Vec<bool>,
    dim: usize,
    n_rows: usize,
}

/// Fold the certified per-coefficient error of one domain's block of the batched
/// [`GpuResidentCoeffBatched`] OUTWARD into the bias over that domain's input box.
///
/// SOUNDNESS: the captured rows carry (a) a bias center split into `b ± b_err`
/// (`b_err` already folds the per-ReLU concretized error on the force-fine pass) and
/// (b) a residual per-coefficient error `err_a`. A raw-coefficient enclosure that
/// dropped `err_a` would be UNSOUND (a too-tight bound → false UNSAT). Here each row's
/// residual coefficient error is bounded by `Σ_j |err_a[j]|·max(|x_l[j]|,|x_u[j]|)`
/// over the box and folded outward: `lower_b ← next_down(b_lo − b_err_lo − penalty)`,
/// `upper_b ← next_up(b_hi + b_err_hi + penalty)`. Any row with a non-finite penalty
/// or non-finite coefficient is REFUSED (kept inherited). The resulting affine form is
/// a guaranteed enclosure of `z(x)` over the box (mirrors the serial path's
/// `fold_coeff_err_over_box_eager` + `concretize_resident_coeff_batched`).
#[allow(dead_code)]
fn fold_seed_rows_for_domain(
    coeff: &GpuResidentCoeffBatched,
    local_block: usize,
    n_rows: usize,
    in_lo: &[f32],
    in_hi: &[f32],
    deadline: Option<Instant>,
) -> Option<FoldedSeedRows> {
    let mut past_deadline = || deadline.is_some_and(|d| Instant::now() >= d);
    fold_seed_rows_for_domain_with_deadline_check(
        coeff,
        local_block,
        n_rows,
        in_lo,
        in_hi,
        &mut past_deadline,
    )
}

/// Callback-explicit fold body for deterministic expiry tests. Production uses
/// the `Instant` wrapper above. Polling is bounded by 1024 coefficient cells so
/// a deadline cannot expire during a full resident error fold and remain hidden
/// until the later CPU clip proposal.
#[allow(dead_code)]
fn fold_seed_rows_for_domain_with_deadline_check<F>(
    coeff: &GpuResidentCoeffBatched,
    local_block: usize,
    n_rows: usize,
    in_lo: &[f32],
    in_hi: &[f32],
    past_deadline: &mut F,
) -> Option<FoldedSeedRows>
where
    F: FnMut() -> bool,
{
    if past_deadline() {
        return None;
    }
    let dim = coeff.dim;
    let coeff_cells = coeff.num_specs.checked_mul(dim)?;
    let row_cells = n_rows.checked_mul(dim)?;
    if dim == 0
        || n_rows == 0
        || coeff.num_specs_per_dom != n_rows
        || !coeff.num_specs.is_multiple_of(n_rows)
        || coeff.lower_a.len() != coeff_cells
        || coeff.upper_a.len() != coeff_cells
        || coeff.lower_err.len() != coeff_cells
        || coeff.upper_err.len() != coeff_cells
        || coeff.lower_b.len() != coeff.num_specs
        || coeff.upper_b.len() != coeff.num_specs
        || coeff.lower_b_err.len() != coeff.num_specs
        || coeff.upper_b_err.len() != coeff.num_specs
        || in_lo.len() != dim
        || in_hi.len() != dim
    {
        return None;
    }
    for j in 0..dim {
        if j.is_multiple_of(1024) && past_deadline() {
            return None;
        }
        let (l, u) = (in_lo[j], in_hi[j]);
        if !l.is_finite() || !u.is_finite() || l > u {
            return None;
        }
    }
    if past_deadline() {
        return None;
    }
    // Includes conservative multiplicities for both coefficient directions,
    // their error sources, per-row biases, and the certificate that may consume
    // this proposal. This check precedes `abs_x` and every output Vec allocation.
    crate::complete_clip::validate_clip_work_budget(1, n_rows, 0, dim).ok()?;
    if past_deadline() {
        return None;
    }
    // Per-input-dim magnitude bound max(|x_l|, |x_u|).
    let mut abs_x = Vec::with_capacity(dim);
    for j in 0..dim {
        if j.is_multiple_of(1024) && past_deadline() {
            return None;
        }
        abs_x.push(f64::from(in_lo[j].abs().max(in_hi[j].abs())));
    }
    if past_deadline() {
        return None;
    }
    let mut out = FoldedSeedRows {
        lower_a: vec![0.0; row_cells],
        upper_a: vec![0.0; row_cells],
        lower_b: vec![0.0; n_rows],
        upper_b: vec![0.0; n_rows],
        refused: vec![false; n_rows],
        dim,
        n_rows,
    };
    for r in 0..n_rows {
        if past_deadline() {
            return None;
        }
        let s = local_block.checked_mul(n_rows)?.checked_add(r)?;
        if s >= coeff.num_specs {
            return None;
        }
        let base = s * dim;
        if base + dim > coeff.lower_a.len() || base + dim > coeff.upper_a.len() {
            return None;
        }
        let mut penalty_lo = 0.0f64;
        let mut penalty_hi = 0.0f64;
        let mut coeff_ok = true;
        for j in 0..dim {
            if j.is_multiple_of(1024) && past_deadline() {
                return None;
            }
            let la = coeff.lower_a[base + j];
            let ua = coeff.upper_a[base + j];
            if !la.is_finite() || !ua.is_finite() {
                coeff_ok = false;
                break;
            }
            out.lower_a[r * dim + j] = la;
            out.upper_a[r * dim + j] = ua;
            // Residual per-coeff error (near-zero after the force-fine fold; folded
            // outward as a backstop so the enclosure is guaranteed even if a row
            // still carries error).
            let le = coeff.lower_err[base + j];
            let ue = coeff.upper_err[base + j];
            penalty_lo = fold_add_up(penalty_lo, fold_mul_up(f64::from(le.abs()), abs_x[j]));
            penalty_hi = fold_add_up(penalty_hi, fold_mul_up(f64::from(ue.abs()), abs_x[j]));
        }
        if past_deadline() {
            return None;
        }
        let lb_c = f64::from(coeff.lower_b[s]);
        let lb_e = f64::from(coeff.lower_b_err[s].abs());
        let ub_c = f64::from(coeff.upper_b[s]);
        let ub_e = f64::from(coeff.upper_b_err[s].abs());
        let lbias = fold_f64_to_f32_down(fold_sub_down(fold_sub_down(lb_c, lb_e), penalty_lo));
        let ubias = fold_f64_to_f32_up(fold_add_up(fold_add_up(ub_c, ub_e), penalty_hi));
        if !coeff_ok
            || !penalty_lo.is_finite()
            || !penalty_hi.is_finite()
            || !lbias.is_finite()
            || !ubias.is_finite()
        {
            out.refused[r] = true;
            // Non-tightening placeholders so a refused row is a no-op in the clip.
            out.lower_b[r] = f32::NEG_INFINITY;
            out.upper_b[r] = f32::INFINITY;
            for j in 0..dim {
                out.lower_a[r * dim + j] = 0.0;
                out.upper_a[r * dim + j] = 0.0;
            }
            continue;
        }
        out.lower_b[r] = lbias;
        out.upper_b[r] = ubias;
    }
    Some(out)
}

/// Directed arithmetic used only by the resident-coefficient certificate fold.
/// Every source is an exact finite f32 dyadic. Widening each f64 operation (not
/// just the final f32 cast) remains sound even when a large error penalty nearly
/// cancels the stored bias center.
#[allow(dead_code)]
fn fold_add_up(a: f64, b: f64) -> f64 {
    fold_next_up_f64(a + b)
}

#[allow(dead_code)]
fn fold_sub_down(a: f64, b: f64) -> f64 {
    fold_next_down_f64(a - b)
}

#[allow(dead_code)]
fn fold_mul_up(a: f64, b: f64) -> f64 {
    fold_next_up_f64(a * b)
}

#[allow(dead_code)]
fn fold_next_down_f64(value: f64) -> f64 {
    if value.is_nan() || value == f64::NEG_INFINITY {
        return value;
    }
    if value == 0.0 {
        return -f64::from_bits(1);
    }
    if value > 0.0 {
        f64::from_bits(value.to_bits() - 1)
    } else {
        f64::from_bits(value.to_bits() + 1)
    }
}

#[allow(dead_code)]
fn fold_next_up_f64(value: f64) -> f64 {
    if value.is_nan() || value == f64::INFINITY {
        return value;
    }
    if value == 0.0 {
        return f64::from_bits(1);
    }
    if value > 0.0 {
        f64::from_bits(value.to_bits() + 1)
    } else {
        f64::from_bits(value.to_bits() - 1)
    }
}

#[allow(dead_code)]
fn fold_f64_to_f32_down(value: f64) -> f32 {
    let candidate = value as f32;
    if f64::from(candidate) <= value {
        candidate
    } else {
        next_down_f32(candidate)
    }
}

#[allow(dead_code)]
fn fold_f64_to_f32_up(value: f64) -> f32 {
    let candidate = value as f32;
    if f64::from(candidate) >= value {
        candidate
    } else {
        next_up_f32(candidate)
    }
}

/// Bit-exact compatibility guard between the legacy raw CUDA proposal and the
/// independently context-validated token rows.  The proposal never supplies an
/// authoritative number; any disagreement disables clipping for the domain.
#[allow(dead_code)]
fn folded_proposal_matches_certificate<F>(
    proposal: &FoldedSeedRows,
    certified: &ValidatedAffineEnclosure,
    past_deadline: &mut F,
) -> bool
where
    F: FnMut() -> bool,
{
    if proposal.dim != certified.dim()
        || proposal.n_rows != certified.rows()
        || proposal.refused.len() != proposal.n_rows
        || proposal.refused.iter().any(|&refused| refused)
        || proposal.lower_a.len() != proposal.n_rows.saturating_mul(proposal.dim)
        || proposal.upper_a.len() != proposal.n_rows.saturating_mul(proposal.dim)
        || proposal.lower_b.len() != proposal.n_rows
        || proposal.upper_b.len() != proposal.n_rows
    {
        return false;
    }
    let mut cells = 0usize;
    for row in 0..proposal.n_rows {
        if proposal.lower_b[row].to_bits() != certified.lower_b()[row].to_bits()
            || proposal.upper_b[row].to_bits() != certified.upper_b()[row].to_bits()
        {
            return false;
        }
        for column in 0..proposal.dim {
            if cells.is_multiple_of(SELECTIVE_DEADLINE_POLL_STRIDE) && past_deadline() {
                return false;
            }
            let index = row * proposal.dim + column;
            if proposal.lower_a[index].to_bits() != certified.lower_a()[[row, column]].to_bits()
                || proposal.upper_a[index].to_bits() != certified.upper_a()[[row, column]].to_bits()
            {
                return false;
            }
            cells = cells.saturating_add(1);
        }
    }
    !past_deadline()
}

/// Capture premise sources for ReLUs whose pre-activation is the network input
/// itself. There is no graph node to seed in `refine_one_seed_pass`; the exact
/// selected identity is already the required root affine enclosure.
#[allow(clippy::too_many_arguments)]
fn initialize_direct_input_sources(
    graph: &GraphNetwork,
    banks: &mut [DomainClipBank],
    histories: &[&GraphSplitHistory],
    beta_states: &[Option<&GraphBetaState>],
    constrained_inputs: &[BoundedTensor],
    deadline: Option<Instant>,
    use_all_history: bool,
) {
    if banks.len() != histories.len()
        || banks.len() != beta_states.len()
        || banks.len() != constrained_inputs.len()
        || banks.is_empty()
    {
        return;
    }
    let root_input = &constrained_inputs[0];
    let root_flat = root_input.flatten();
    let input_dim = root_flat.lower().len();
    let mut direct_relus = std::collections::BTreeSet::new();
    for history in histories {
        let Some(scheduled) = scheduled_relu_constraints(graph, history, use_all_history) else {
            return;
        };
        for constraint in scheduled {
            let Some(relu) = graph.nodes.get(constraint.node_name()) else {
                continue;
            };
            if matches!(relu.layer, Layer::ReLU(_))
                && relu
                    .require_unary_input()
                    .is_ok_and(|input| input == NETWORK_INPUT)
            {
                direct_relus.insert(constraint.node_name().to_string());
            }
        }
    }

    for relu_name in direct_relus {
        if deadline.is_some_and(|d| Instant::now() >= d) {
            return;
        }
        let mut selected = Vec::new();
        for history in histories {
            let Some(scheduled) = scheduled_relu_constraints(graph, history, use_all_history)
            else {
                return;
            };
            selected.extend(
                scheduled
                    .into_iter()
                    .filter(|constraint| constraint.node_name() == relu_name)
                    .map(|constraint| constraint.neuron_idx()),
            );
        }
        selected.sort_unstable();
        selected.dedup();
        if selected.is_empty() || selected.iter().any(|&neuron| neuron >= input_dim) {
            continue;
        }
        let mut row_of = vec![usize::MAX; input_dim];
        for (row, &neuron) in selected.iter().enumerate() {
            row_of[neuron] = row;
        }
        let Some(root_rows) =
            capture_exact_root_input_rows(graph, &selected, &row_of, root_input, deadline).ok()
        else {
            continue;
        };
        let shared_selected: Arc<[usize]> = Arc::from(selected.as_slice());
        let shared_row_of: Arc<[usize]> = Arc::from(row_of.as_slice());

        for domain in 0..banks.len() {
            let Some(beta) = beta_states[domain] else {
                banks[domain].disabled = true;
                continue;
            };
            if !beta_state_matches_pure_relu_history(histories[domain], beta) {
                banks[domain].disabled = true;
                continue;
            }
            let flat = constrained_inputs[domain].flatten();
            let (Some(in_lo), Some(in_hi)) = (
                flat.lower().as_slice_memory_order(),
                flat.upper().as_slice_memory_order(),
            ) else {
                banks[domain].disabled = true;
                continue;
            };
            let Some((pass, token)) = bind_root_sound_crown_rows_to_history(
                graph.cut_fold_scope(),
                in_lo,
                in_hi,
                histories[domain],
                NETWORK_INPUT,
                &selected,
                &row_of,
                &root_rows,
                deadline,
            )
            .ok()
            .and_then(|rows| mint_certified_affine_enclosure(rows, deadline).ok()) else {
                banks[domain].disabled = true;
                continue;
            };
            banks[domain].layers.insert(
                relu_name.clone(),
                CertifiedLayerCapture {
                    seed_node: Arc::from(NETWORK_INPUT),
                    selected_neurons: Arc::clone(&shared_selected),
                    row_of_neuron: Arc::clone(&shared_row_of),
                    objective_rows: Vec::new(),
                    pass,
                    token,
                },
            );
        }
    }
}

type ClipTargetProposal = (String, Vec<usize>, Vec<f32>, Vec<f32>);

/// A root-bank target is either solved, a legitimate no-op because every
/// scheduled premise is already decided by the input box, or incomplete.
/// Keeping `Noop` distinct from `Incomplete` prevents resource, provenance,
/// structural, and deadline refusals from masquerading as completed zero-yield
/// batches in the adaptive scheduler.
//
// Boxing `Proposal` would add allocation and deadline behavior to the
// certificate path. Keep the intentionally stack-resident representation.
#[allow(clippy::large_enum_variant)]
enum ClipTargetOutcome {
    Proposal(ClipTargetProposal),
    Noop,
    Infeasible,
    Incomplete,
    Deadline,
}

/// Result of attempting the amortized multi-target solve for one domain.
///
/// `Fallback` is reserved for a combined resource envelope or an unexpected
/// solver refusal: the caller then runs the already-audited scalar target path.
/// Structural/provenance outcomes remain explicit and aligned with
/// `DomainClipBank::target_order`.
enum ClipDomainBatchOutcome {
    Outcomes(Vec<ClipTargetOutcome>),
    Fallback,
}

/// Clip one already-captured target against either every ReLU premise or the
/// newest premise in the first nonempty split layer in network order, following
/// αβ-CROWN's first-two-iterations/all-then-layer-latest policy. The certified
/// rows remain bound to the exact full child history in both modes. Missing
/// rows for the selected constraint set refuse the whole proposal rather than
/// silently solving a partial set.
#[cfg(test)]
fn clip_target_from_bank_checked(
    graph: &GraphNetwork,
    bank: &DomainClipBank,
    target_relu: &str,
    history: &GraphSplitHistory,
    in_lo: &[f32],
    in_hi: &[f32],
    deadline: Option<Instant>,
    use_all_history: bool,
) -> ClipTargetOutcome {
    let mut past_deadline = || deadline.is_some_and(|d| Instant::now() >= d);
    clip_target_from_bank_checked_with_deadline_check(
        graph,
        bank,
        target_relu,
        history,
        in_lo,
        in_hi,
        deadline,
        use_all_history,
        &mut past_deadline,
    )
}

#[allow(clippy::too_many_arguments)]
fn clip_target_from_bank_checked_with_deadline_check<F>(
    graph: &GraphNetwork,
    bank: &DomainClipBank,
    target_relu: &str,
    history: &GraphSplitHistory,
    in_lo: &[f32],
    in_hi: &[f32],
    deadline: Option<Instant>,
    use_all_history: bool,
    past_deadline: &mut F,
) -> ClipTargetOutcome
where
    F: FnMut() -> bool,
{
    if bank.disabled
        || !history.is_pure_relu_at_zero()
        || history.constraints.len() != history.depth()
        || in_lo.len() != in_hi.len()
    {
        return ClipTargetOutcome::Incomplete;
    }
    let Some(target) = bank.layers.get(target_relu) else {
        return ClipTargetOutcome::Incomplete;
    };
    if target.objective_rows.is_empty() {
        return ClipTargetOutcome::Incomplete;
    }
    if history.depth() == 0 {
        return ClipTargetOutcome::Noop;
    }
    let scheduled_history = if use_all_history {
        None
    } else {
        let Some(scheduled) = scheduled_relu_constraints(graph, history, false) else {
            return ClipTargetOutcome::Incomplete;
        };
        let Some(constraint) = scheduled.into_iter().next() else {
            return ClipTargetOutcome::Incomplete;
        };
        Some(GraphSplitHistory::new().with_constraint(constraint.clone()))
    };
    let constraint_history = scheduled_history.as_ref().unwrap_or(history);
    let x_dim = in_lo.len();
    if past_deadline() {
        return ClipTargetOutcome::Deadline;
    }
    if crate::complete_clip::validate_clip_work_budget(
        1,
        target.objective_rows.len(),
        constraint_history.depth(),
        x_dim,
    )
    .is_err()
    {
        return ClipTargetOutcome::Incomplete;
    }
    if past_deadline() {
        return ClipTargetOutcome::Deadline;
    }

    // Preflight the scheduled constraint set before allocating solver matrices.
    // Tokens below are still validated against `history`, never this subset.
    let mut needed_layers = std::collections::BTreeSet::new();
    needed_layers.insert(target_relu.to_string());
    for (index, constraint) in constraint_history.constraints.iter().enumerate() {
        if index.is_multiple_of(SELECTIVE_DEADLINE_POLL_STRIDE) && past_deadline() {
            return ClipTargetOutcome::Deadline;
        }
        let Some(source) = bank.layers.get(constraint.node_name()) else {
            return ClipTargetOutcome::Incomplete;
        };
        let Some(&row) = source.row_of_neuron.get(constraint.neuron_idx()) else {
            return ClipTargetOutcome::Incomplete;
        };
        if row == usize::MAX || row >= source.selected_neurons.len() {
            return ClipTargetOutcome::Incomplete;
        }
        needed_layers.insert(constraint.node_name().to_string());
    }

    let mut validated: HashMap<String, ValidatedAffineEnclosure> = HashMap::new();
    if validated.try_reserve(needed_layers.len()).is_err() {
        return ClipTargetOutcome::Incomplete;
    }
    for relu_name in needed_layers {
        if past_deadline() {
            return ClipTargetOutcome::Deadline;
        }
        let Some(capture) = bank.layers.get(&relu_name) else {
            return ClipTargetOutcome::Incomplete;
        };
        let rows = match capture.token.validate_for_clip_in_scope(
            graph.cut_fold_scope(),
            &capture.pass,
            in_lo,
            in_hi,
            history,
            &capture.seed_node,
            &capture.selected_neurons,
            &capture.row_of_neuron,
            deadline,
        ) {
            Ok(rows) => rows,
            Err(_) if past_deadline() => return ClipTargetOutcome::Deadline,
            Err(_) => return ClipTargetOutcome::Incomplete,
        };
        validated.insert(relu_name, rows);
    }

    let constraints = match build_split_constraints_with_deadline_check(
        constraint_history,
        |node_name: &str, neuron_idx: usize, deadline_check| {
            let capture = bank.layers.get(node_name)?;
            let rows = validated.get(node_name)?;
            let row = *capture.row_of_neuron.get(neuron_idx)?;
            if row == usize::MAX || row >= rows.rows() {
                return None;
            }
            let mut lower_a = Array1::zeros(x_dim);
            let mut upper_a = Array1::zeros(x_dim);
            for column in 0..x_dim {
                if column.is_multiple_of(SELECTIVE_DEADLINE_POLL_STRIDE) && deadline_check() {
                    return None;
                }
                lower_a[column] = rows.lower_a()[[row, column]];
                upper_a[column] = rows.upper_a()[[row, column]];
            }
            Some((lower_a, rows.lower_b()[row], upper_a, rows.upper_b()[row]))
        },
        x_dim,
        past_deadline,
    ) {
        Ok(constraints) => constraints,
        Err(_) if past_deadline() => return ClipTargetOutcome::Deadline,
        Err(_) => return ClipTargetOutcome::Incomplete,
    };
    if constraints.num_constraints != constraint_history.constraints.len() || constraints.is_empty()
    {
        return ClipTargetOutcome::Incomplete;
    }

    let input_lower = Array1::from_vec(in_lo.to_vec());
    let input_upper = Array1::from_vec(in_hi.to_vec());
    let preprocessed = match sort_out_constraints_with_deadline_check(
        &constraints,
        &input_lower,
        &input_upper,
        past_deadline,
    ) {
        Ok(preprocessed) => preprocessed,
        Err(_) if past_deadline() => return ClipTargetOutcome::Deadline,
        Err(_) => return ClipTargetOutcome::Incomplete,
    };
    if past_deadline() {
        return ClipTargetOutcome::Deadline;
    }
    if preprocessed
        .infeasible_mask
        .iter()
        .any(|&is_infeasible| is_infeasible)
    {
        return ClipTargetOutcome::Infeasible;
    }
    if preprocessed.a_active.nrows() == 0 {
        return ClipTargetOutcome::Noop;
    }

    let Some(target_rows) = validated.get(target_relu) else {
        return ClipTargetOutcome::Incomplete;
    };
    let n_obj = target.objective_rows.len();
    let mut objective_lower_a = Array2::zeros((n_obj, x_dim));
    let mut objective_upper_a = Array2::zeros((n_obj, x_dim));
    let mut objective_lower_b = Array1::zeros(n_obj);
    let mut objective_upper_b = Array1::zeros(n_obj);
    let mut neurons = Vec::new();
    if neurons.try_reserve_exact(n_obj).is_err() {
        return ClipTargetOutcome::Incomplete;
    }
    for (objective, &source_row) in target.objective_rows.iter().enumerate() {
        if source_row >= target_rows.rows() || source_row >= target.selected_neurons.len() {
            return ClipTargetOutcome::Incomplete;
        }
        neurons.push(target.selected_neurons[source_row]);
        objective_lower_b[objective] = target_rows.lower_b()[source_row];
        objective_upper_b[objective] = target_rows.upper_b()[source_row];
        for column in 0..x_dim {
            if column.is_multiple_of(SELECTIVE_DEADLINE_POLL_STRIDE) && past_deadline() {
                return ClipTargetOutcome::Deadline;
            }
            objective_lower_a[[objective, column]] = target_rows.lower_a()[[source_row, column]];
            objective_upper_a[[objective, column]] = target_rows.upper_a()[[source_row, column]];
        }
    }
    let (lower, upper) = match tighten_with_constraints_with_deadline(
        &preprocessed,
        &objective_lower_a,
        &objective_lower_b,
        &objective_upper_a,
        &objective_upper_b,
        &input_lower,
        &input_upper,
        deadline,
    ) {
        Ok(bounds) => bounds,
        Err(_) if past_deadline() => return ClipTargetOutcome::Deadline,
        Err(_) => return ClipTargetOutcome::Incomplete,
    };
    if past_deadline() {
        return ClipTargetOutcome::Deadline;
    }
    ClipTargetOutcome::Proposal((
        target.seed_node.to_string(),
        neurons,
        lower.to_vec(),
        upper.to_vec(),
    ))
}

/// Amortize the Complete Clipping compatibility seam across every distinct
/// target of one domain.
///
/// Constraint-source tokens, split-row construction, box preprocessing, and
/// the certified coordinate solve are shared. Objective rows are concatenated
/// in target order; the solver and independent checker operate independently
/// per objective row, so splitting the returned rows is bit-identical to
/// solving the same rows separately. No proposal escapes an interrupted
/// combined solve, and cache mutation remains in the caller's serial,
/// target-ordered `apply_selected_bounds` stage.
#[allow(clippy::too_many_arguments)]
fn clip_targets_from_bank_batched_checked_with_deadline_check<F>(
    graph: &GraphNetwork,
    bank: &DomainClipBank,
    history: &GraphSplitHistory,
    in_lo: &[f32],
    in_hi: &[f32],
    deadline: Option<Instant>,
    use_all_history: bool,
    past_deadline: &mut F,
) -> ClipDomainBatchOutcome
where
    F: FnMut() -> bool,
{
    let n_targets = bank.target_order.len();
    let incomplete = || {
        ClipDomainBatchOutcome::Outcomes(
            std::iter::repeat_with(|| ClipTargetOutcome::Incomplete)
                .take(n_targets)
                .collect(),
        )
    };
    let deadline_outcome = || {
        ClipDomainBatchOutcome::Outcomes(
            std::iter::repeat_with(|| ClipTargetOutcome::Deadline)
                .take(n_targets)
                .collect(),
        )
    };
    if n_targets < 2
        || bank.disabled
        || !history.is_pure_relu_at_zero()
        || history.constraints.len() != history.depth()
        || in_lo.len() != in_hi.len()
    {
        return incomplete();
    }
    let mut unique_targets = std::collections::BTreeSet::new();
    if bank
        .target_order
        .iter()
        .any(|target| !unique_targets.insert(target.as_str()))
    {
        // Repeated target names are not emitted by production capture. Keep
        // the scalar path for compatibility tests that deliberately construct
        // them to observe per-target deadline interruption.
        return ClipDomainBatchOutcome::Fallback;
    }
    if past_deadline() {
        return deadline_outcome();
    }

    let scheduled_history = if use_all_history {
        None
    } else {
        let Some(scheduled) = scheduled_relu_constraints(graph, history, false) else {
            return incomplete();
        };
        let Some(constraint) = scheduled.into_iter().next() else {
            return incomplete();
        };
        Some(GraphSplitHistory::new().with_constraint(constraint.clone()))
    };
    let constraint_history = scheduled_history.as_ref().unwrap_or(history);
    let x_dim = in_lo.len();

    // Preserve the scalar target's ordering: an absent/empty target is
    // incomplete even for the depth-zero no-op case.
    let mut eligible = vec![false; n_targets];
    for (target_index, target_relu) in bank.target_order.iter().enumerate() {
        let Some(target) = bank.layers.get(target_relu) else {
            continue;
        };
        if target.objective_rows.is_empty() {
            continue;
        }
        if history.depth() == 0 {
            eligible[target_index] = true;
            continue;
        }
        if crate::complete_clip::validate_clip_work_budget(
            1,
            target.objective_rows.len(),
            constraint_history.depth(),
            x_dim,
        )
        .is_ok()
        {
            eligible[target_index] = true;
        }
    }
    if history.depth() == 0 {
        return ClipDomainBatchOutcome::Outcomes(
            eligible
                .into_iter()
                .map(|is_eligible| {
                    if is_eligible {
                        ClipTargetOutcome::Noop
                    } else {
                        ClipTargetOutcome::Incomplete
                    }
                })
                .collect(),
        );
    }
    if !eligible.iter().any(|&is_eligible| is_eligible) {
        return incomplete();
    }
    if past_deadline() {
        return deadline_outcome();
    }

    // Every target consumes the same scheduled split rows. Validate each
    // unique source token once, then retain the resulting dense enclosure for
    // target reuse when a source layer is also an objective layer.
    let mut constraint_layers = std::collections::BTreeSet::new();
    for (index, constraint) in constraint_history.constraints.iter().enumerate() {
        if index.is_multiple_of(SELECTIVE_DEADLINE_POLL_STRIDE) && past_deadline() {
            return deadline_outcome();
        }
        let Some(source) = bank.layers.get(constraint.node_name()) else {
            return incomplete();
        };
        let Some(&row) = source.row_of_neuron.get(constraint.neuron_idx()) else {
            return incomplete();
        };
        if row == usize::MAX || row >= source.selected_neurons.len() {
            return incomplete();
        }
        constraint_layers.insert(constraint.node_name().to_string());
    }

    let mut validated: HashMap<String, ValidatedAffineEnclosure> = HashMap::new();
    if validated
        .try_reserve(constraint_layers.len().saturating_add(n_targets))
        .is_err()
    {
        return ClipDomainBatchOutcome::Fallback;
    }
    for relu_name in constraint_layers {
        if past_deadline() {
            return deadline_outcome();
        }
        let Some(capture) = bank.layers.get(&relu_name) else {
            return incomplete();
        };
        let rows = match capture.token.validate_for_clip_in_scope(
            graph.cut_fold_scope(),
            &capture.pass,
            in_lo,
            in_hi,
            history,
            &capture.seed_node,
            &capture.selected_neurons,
            &capture.row_of_neuron,
            deadline,
        ) {
            Ok(rows) => rows,
            Err(_) if past_deadline() => return deadline_outcome(),
            Err(_) => return incomplete(),
        };
        validated.insert(relu_name, rows);
    }

    // Validate objective tokens at most once. A target-local refusal remains
    // local; the other targets can still share the certified domain solve.
    for (target_index, target_relu) in bank.target_order.iter().enumerate() {
        if !eligible[target_index] || validated.contains_key(target_relu) {
            continue;
        }
        if past_deadline() {
            return deadline_outcome();
        }
        let Some(capture) = bank.layers.get(target_relu) else {
            eligible[target_index] = false;
            continue;
        };
        match capture.token.validate_for_clip_in_scope(
            graph.cut_fold_scope(),
            &capture.pass,
            in_lo,
            in_hi,
            history,
            &capture.seed_node,
            &capture.selected_neurons,
            &capture.row_of_neuron,
            deadline,
        ) {
            Ok(rows) => {
                validated.insert(target_relu.clone(), rows);
            }
            Err(_) if past_deadline() => return deadline_outcome(),
            Err(_) => eligible[target_index] = false,
        }
    }

    let constraints = match build_split_constraints_with_deadline_check(
        constraint_history,
        |node_name: &str, neuron_idx: usize, deadline_check| {
            let capture = bank.layers.get(node_name)?;
            let rows = validated.get(node_name)?;
            let row = *capture.row_of_neuron.get(neuron_idx)?;
            if row == usize::MAX || row >= rows.rows() {
                return None;
            }
            let mut lower_a = Array1::zeros(x_dim);
            let mut upper_a = Array1::zeros(x_dim);
            for column in 0..x_dim {
                if column.is_multiple_of(SELECTIVE_DEADLINE_POLL_STRIDE) && deadline_check() {
                    return None;
                }
                lower_a[column] = rows.lower_a()[[row, column]];
                upper_a[column] = rows.upper_a()[[row, column]];
            }
            Some((lower_a, rows.lower_b()[row], upper_a, rows.upper_b()[row]))
        },
        x_dim,
        past_deadline,
    ) {
        Ok(constraints) => constraints,
        Err(_) if past_deadline() => return deadline_outcome(),
        Err(_) => return incomplete(),
    };
    if constraints.num_constraints != constraint_history.constraints.len() || constraints.is_empty()
    {
        return incomplete();
    }

    let input_lower = Array1::from_vec(in_lo.to_vec());
    let input_upper = Array1::from_vec(in_hi.to_vec());
    let preprocessed = match sort_out_constraints_with_deadline_check(
        &constraints,
        &input_lower,
        &input_upper,
        past_deadline,
    ) {
        Ok(preprocessed) => preprocessed,
        Err(_) if past_deadline() => return deadline_outcome(),
        Err(_) => return incomplete(),
    };
    if past_deadline() {
        return deadline_outcome();
    }

    let mut outcomes: Vec<ClipTargetOutcome> =
        std::iter::repeat_with(|| ClipTargetOutcome::Incomplete)
            .take(n_targets)
            .collect();
    if preprocessed
        .infeasible_mask
        .iter()
        .any(|&is_infeasible| is_infeasible)
    {
        for (outcome, &is_eligible) in outcomes.iter_mut().zip(&eligible) {
            if is_eligible {
                *outcome = ClipTargetOutcome::Infeasible;
            }
        }
        return ClipDomainBatchOutcome::Outcomes(outcomes);
    }
    if preprocessed.a_active.nrows() == 0 {
        for (outcome, &is_eligible) in outcomes.iter_mut().zip(&eligible) {
            if is_eligible {
                *outcome = ClipTargetOutcome::Noop;
            }
        }
        return ClipDomainBatchOutcome::Outcomes(outcomes);
    }

    struct PackedTarget {
        target_index: usize,
        seed_node: String,
        neurons: Vec<usize>,
        start: usize,
        len: usize,
    }

    let mut total_objectives = 0usize;
    for (target_index, target_relu) in bank.target_order.iter().enumerate() {
        if !eligible[target_index] {
            continue;
        }
        let Some(target) = bank.layers.get(target_relu) else {
            continue;
        };
        let Some(target_rows) = validated.get(target_relu) else {
            continue;
        };
        let rows_valid = target.objective_rows.iter().all(|&source_row| {
            source_row < target_rows.rows() && source_row < target.selected_neurons.len()
        });
        if !rows_valid {
            eligible[target_index] = false;
            continue;
        }
        let Some(next_total) = total_objectives.checked_add(target.objective_rows.len()) else {
            return ClipDomainBatchOutcome::Fallback;
        };
        total_objectives = next_total;
    }
    if total_objectives == 0
        || crate::complete_clip::validate_clip_work_budget(
            1,
            total_objectives,
            preprocessed.a_active.nrows(),
            x_dim,
        )
        .is_err()
    {
        return ClipDomainBatchOutcome::Fallback;
    }

    let mut objective_lower_a = Array2::zeros((total_objectives, x_dim));
    let mut objective_upper_a = Array2::zeros((total_objectives, x_dim));
    let mut objective_lower_b = Array1::zeros(total_objectives);
    let mut objective_upper_b = Array1::zeros(total_objectives);
    let mut packed = Vec::new();
    if packed.try_reserve_exact(n_targets).is_err() {
        return ClipDomainBatchOutcome::Fallback;
    }
    let mut cursor = 0usize;
    for (target_index, target_relu) in bank.target_order.iter().enumerate() {
        if !eligible[target_index] {
            continue;
        }
        let target = &bank.layers[target_relu];
        let target_rows = &validated[target_relu];
        let start = cursor;
        let mut neurons = Vec::new();
        if neurons
            .try_reserve_exact(target.objective_rows.len())
            .is_err()
        {
            return ClipDomainBatchOutcome::Fallback;
        }
        for &source_row in &target.objective_rows {
            neurons.push(target.selected_neurons[source_row]);
            objective_lower_b[cursor] = target_rows.lower_b()[source_row];
            objective_upper_b[cursor] = target_rows.upper_b()[source_row];
            for column in 0..x_dim {
                if column.is_multiple_of(SELECTIVE_DEADLINE_POLL_STRIDE) && past_deadline() {
                    return deadline_outcome();
                }
                objective_lower_a[[cursor, column]] = target_rows.lower_a()[[source_row, column]];
                objective_upper_a[[cursor, column]] = target_rows.upper_a()[[source_row, column]];
            }
            cursor += 1;
        }
        packed.push(PackedTarget {
            target_index,
            seed_node: target.seed_node.to_string(),
            neurons,
            start,
            len: cursor - start,
        });
    }
    debug_assert_eq!(cursor, total_objectives);

    let (lower, upper) = match tighten_with_constraints_with_deadline(
        &preprocessed,
        &objective_lower_a,
        &objective_lower_b,
        &objective_upper_a,
        &objective_upper_b,
        &input_lower,
        &input_upper,
        deadline,
    ) {
        Ok(bounds) => bounds,
        Err(_) if past_deadline() => return deadline_outcome(),
        // A combined envelope can be refused even when its individual targets
        // fit. Preserve the scalar fail-closed compatibility path in that rare
        // case rather than changing target-local completion semantics.
        Err(_) => return ClipDomainBatchOutcome::Fallback,
    };
    if past_deadline() {
        return deadline_outcome();
    }
    for target in packed {
        let end = target.start + target.len;
        outcomes[target.target_index] = ClipTargetOutcome::Proposal((
            target.seed_node,
            target.neurons,
            lower
                .slice(ndarray::s![target.start..end])
                .iter()
                .copied()
                .collect(),
            upper
                .slice(ndarray::s![target.start..end])
                .iter()
                .copied()
                .collect(),
        ));
    }
    ClipDomainBatchOutcome::Outcomes(outcomes)
}

#[cfg(test)]
fn clip_target_from_bank(
    graph: &GraphNetwork,
    bank: &DomainClipBank,
    target_relu: &str,
    history: &GraphSplitHistory,
    in_lo: &[f32],
    in_hi: &[f32],
    deadline: Option<Instant>,
    use_all_history: bool,
) -> Option<ClipTargetProposal> {
    match clip_target_from_bank_checked(
        graph,
        bank,
        target_relu,
        history,
        in_lo,
        in_hi,
        deadline,
        use_all_history,
    ) {
        ClipTargetOutcome::Proposal(proposal) => Some(proposal),
        ClipTargetOutcome::Noop
        | ClipTargetOutcome::Infeasible
        | ClipTargetOutcome::Incomplete
        | ClipTargetOutcome::Deadline => None,
    }
}

/// Re-clip every target captured so far after inserting a new source layer.
/// This is what makes a later-in-execution-order premise affect an earlier
/// target while retaining the immediate same-child cache cascade.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct ReclipOutcome {
    tightened: bool,
    /// False only when the batch was structurally invalid or its deadline
    /// interrupted the solve. Callers use this to suppress zero-yield latches.
    completed: bool,
    targets_expected: usize,
    targets_completed: usize,
    targets_refused: usize,
    deadline_interrupted: bool,
}

fn complete_clip_batch_completed(
    captures_expected: usize,
    captures_succeeded: usize,
    captures_refused: usize,
    reclip: Option<ReclipOutcome>,
) -> bool {
    captures_expected > 0
        && captures_refused == 0
        && captures_succeeded == captures_expected
        && reclip.is_some_and(|outcome| outcome.completed && !outcome.deadline_interrupted)
}

#[allow(clippy::too_many_arguments)]
fn reclip_all_captured_targets(
    graph: &GraphNetwork,
    banks: &[DomainClipBank],
    histories: &[&GraphSplitHistory],
    constrained_inputs: &[BoundedTensor],
    caches: &mut [HashMap<String, Arc<BoundedTensor>>],
    deadline: Option<Instant>,
    stats: &mut RefineStats,
    use_all_history: bool,
) -> ReclipOutcome {
    let mut past_deadline = || deadline.is_some_and(|d| Instant::now() >= d);
    reclip_all_captured_targets_with_deadline_check(
        graph,
        banks,
        histories,
        constrained_inputs,
        caches,
        deadline,
        stats,
        use_all_history,
        &mut past_deadline,
    )
}

#[allow(clippy::too_many_arguments)]
fn reclip_all_captured_targets_with_deadline_check<F>(
    graph: &GraphNetwork,
    banks: &[DomainClipBank],
    histories: &[&GraphSplitHistory],
    constrained_inputs: &[BoundedTensor],
    caches: &mut [HashMap<String, Arc<BoundedTensor>>],
    deadline: Option<Instant>,
    stats: &mut RefineStats,
    use_all_history: bool,
    past_deadline: &mut F,
) -> ReclipOutcome
where
    F: FnMut() -> bool,
{
    if banks.len() != histories.len()
        || banks.len() != constrained_inputs.len()
        || banks.len() != caches.len()
    {
        return ReclipOutcome::default();
    }
    let targets_expected = banks
        .iter()
        .map(|bank| bank.target_order.len())
        .sum::<usize>();
    if targets_expected == 0 {
        return ReclipOutcome::default();
    }
    let mut any_tightened = false;
    let mut structurally_complete = true;
    let mut targets_completed = 0usize;
    let mut targets_refused = 0usize;
    for domain in 0..banks.len() {
        if past_deadline() {
            return ReclipOutcome {
                tightened: any_tightened,
                completed: false,
                targets_expected,
                targets_completed,
                targets_refused,
                deadline_interrupted: true,
            };
        }
        let flat = constrained_inputs[domain].flatten();
        let (Some(in_lo), Some(in_hi)) = (
            flat.lower().as_slice_memory_order(),
            flat.upper().as_slice_memory_order(),
        ) else {
            structurally_complete = false;
            targets_refused = targets_refused.saturating_add(banks[domain].target_order.len());
            continue;
        };
        if banks[domain].disabled {
            structurally_complete = false;
            targets_refused = targets_refused.saturating_add(banks[domain].target_order.len());
            continue;
        }
        let mut batched_outcomes = if banks[domain].target_order.len() > 1 {
            match clip_targets_from_bank_batched_checked_with_deadline_check(
                graph,
                &banks[domain],
                histories[domain],
                in_lo,
                in_hi,
                deadline,
                use_all_history,
                past_deadline,
            ) {
                ClipDomainBatchOutcome::Outcomes(outcomes) => Some(outcomes.into_iter()),
                ClipDomainBatchOutcome::Fallback => None,
            }
        } else {
            None
        };
        for (target_index, target_relu) in banks[domain].target_order.iter().enumerate() {
            let clip_outcome = if let Some(outcomes) = batched_outcomes.as_mut() {
                outcomes.next().unwrap_or(ClipTargetOutcome::Incomplete)
            } else {
                clip_target_from_bank_checked_with_deadline_check(
                    graph,
                    &banks[domain],
                    target_relu,
                    histories[domain],
                    in_lo,
                    in_hi,
                    deadline,
                    use_all_history,
                    past_deadline,
                )
            };
            let (seed_node, neurons, lower, upper) = match clip_outcome {
                ClipTargetOutcome::Proposal(proposal) => proposal,
                ClipTargetOutcome::Noop => {
                    targets_completed = targets_completed.saturating_add(1);
                    continue;
                }
                ClipTargetOutcome::Infeasible => {
                    // The mask now comes from directed endpoint arithmetic,
                    // but this compatibility seam deliberately grants it no
                    // BaB-pruning authority. Refuse optional tightening and
                    // leave domain closure to the verifier's established
                    // proof-producing paths.
                    structurally_complete = false;
                    targets_refused = targets_refused.saturating_add(
                        banks[domain]
                            .target_order
                            .len()
                            .saturating_sub(target_index),
                    );
                    break;
                }
                ClipTargetOutcome::Incomplete => {
                    structurally_complete = false;
                    targets_refused = targets_refused.saturating_add(1);
                    continue;
                }
                ClipTargetOutcome::Deadline => {
                    return ReclipOutcome {
                        tightened: any_tightened,
                        completed: false,
                        targets_expected,
                        targets_completed,
                        targets_refused,
                        deadline_interrupted: true,
                    };
                }
            };
            if past_deadline() {
                return ReclipOutcome {
                    tightened: any_tightened,
                    completed: false,
                    targets_expected,
                    targets_completed,
                    targets_refused,
                    deadline_interrupted: true,
                };
            }
            let Some(pre_dim) = caches[domain].get(&seed_node).map(|entry| entry.len()) else {
                structurally_complete = false;
                targets_refused = targets_refused.saturating_add(1);
                continue;
            };
            let before_tightened = stats.neurons_tightened;
            let before = caches[domain].get(&seed_node).cloned();
            let applied = apply_selected_bounds(
                &mut caches[domain],
                &seed_node,
                target_relu,
                &lower,
                &upper,
                pre_dim,
                &neurons,
                deadline,
                stats,
            );
            if past_deadline() {
                return ReclipOutcome {
                    tightened: any_tightened,
                    completed: false,
                    targets_expected,
                    targets_completed,
                    targets_refused,
                    deadline_interrupted: true,
                };
            }
            if applied && stats.neurons_tightened > before_tightened {
                any_tightened = true;
                stats.clip_domains += 1;
                if let (Some(before), Some(after)) =
                    (before.as_ref(), caches[domain].get(&seed_node))
                {
                    let (
                        Some(before_lower),
                        Some(before_upper),
                        Some(after_lower),
                        Some(after_upper),
                    ) = (
                        before.lower().as_slice_memory_order(),
                        before.upper().as_slice_memory_order(),
                        after.lower().as_slice_memory_order(),
                        after.upper().as_slice_memory_order(),
                    )
                    else {
                        continue;
                    };
                    for &neuron in &neurons {
                        if neuron < before.len() && neuron < after.len() {
                            stats.clip_max_tighten = stats
                                .clip_max_tighten
                                .max((after_lower[neuron] - before_lower[neuron]).abs())
                                .max((after_upper[neuron] - before_upper[neuron]).abs());
                        }
                    }
                }
            }
            if applied {
                // `apply_selected_bounds` returns true for a valid unchanged
                // intersection as well as for a tightening.
                targets_completed = targets_completed.saturating_add(1);
            } else {
                structurally_complete = false;
                targets_refused = targets_refused.saturating_add(1);
            }
        }
    }
    ReclipOutcome {
        tightened: any_tightened,
        completed: structurally_complete
            && targets_completed == targets_expected
            && targets_refused == 0,
        targets_expected,
        targets_completed,
        targets_refused,
        deadline_interrupted: false,
    }
}

/// #clip-interm-resnet-batched: run the split-constraint clip for one domain from its
/// folded seed rows + the domain's original split history. Builds input-space
/// half-spaces from the split neurons' folded rows
/// ([`build_split_constraints`]), and constrained-concretizes every `sel` objective row
/// over `box ∩ half-spaces` ([`tighten_with_constraints_with_deadline`]). Returns
/// `(tightened_lower, tightened_upper)` over `sel` rows, or `None` when there are no
/// usable constraints (no-op — sound). All bound math is the existing clip solver;
/// this only sources the objective/constraint rows from the batched ResidentCoeff
/// instead of a per-child backward.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn clip_seed_domain(
    folded: &FoldedSeedRows,
    authority: Option<(&CertifiedAffineEnclosure, &CrownPassStamp)>,
    history: &GraphSplitHistory,
    beta_state: &GraphBetaState,
    relu_name: &str,
    row_of: &[usize],
    sel: &[usize],
    in_lo: &[f32],
    in_hi: &[f32],
    deadline: Option<Instant>,
) -> Option<(Vec<f32>, Vec<f32>)> {
    let mut past_deadline = || deadline.is_some_and(|d| Instant::now() >= d);
    clip_seed_domain_with_deadline_check(
        folded,
        authority,
        history,
        beta_state,
        relu_name,
        row_of,
        sel,
        in_lo,
        in_hi,
        deadline,
        &mut past_deadline,
    )
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn clip_seed_domain_with_deadline_check<F>(
    folded: &FoldedSeedRows,
    authority: Option<(&CertifiedAffineEnclosure, &CrownPassStamp)>,
    history: &GraphSplitHistory,
    beta_state: &GraphBetaState,
    relu_name: &str,
    row_of: &[usize],
    sel: &[usize],
    in_lo: &[f32],
    in_hi: &[f32],
    deadline: Option<Instant>,
    past_deadline: &mut F,
) -> Option<(Vec<f32>, Vec<f32>)>
where
    F: FnMut() -> bool,
{
    // A raw ResidentCoeff/FoldedSeedRows value is a proposal, never authority.
    // The scored gate is still hard-false, and the current sound-CROWN boundary
    // deliberately has no production token constructor, so this returns `None`
    // (inherit bounds) outside explicit provenance fixtures.
    let (token, pass) = authority?;
    let x_dim = folded.dim;
    // Fail before building histories, half-spaces, or objective matrices when
    // their conservative dense-work estimate exceeds the certified clip cap.
    crate::complete_clip::validate_clip_work_budget(1, sel.len(), history.depth(), x_dim).ok()?;
    if past_deadline() {
        return None;
    }
    let dbg = std::env::var("NY_CLIP_DBG").ok().as_deref() == Some("1");
    if dbg {
        let names: Vec<String> = beta_state
            .entries
            .iter()
            .take(3)
            .map(|e| format!("{}[{}]", e.node_name(), e.neuron_idx()))
            .collect();
        let rows: Vec<i64> = beta_state
            .entries
            .iter()
            .take(3)
            .map(|e| {
                if e.node_name() == relu_name && e.neuron_idx() < row_of.len() {
                    row_of[e.neuron_idx()] as i64
                } else {
                    -1
                }
            })
            .collect();
        eprintln!(
            "[clip-dbg] relu_name={relu_name} n_entries={} entry_nodes={names:?} rows={rows:?} sel_len={} row_of_len={}",
            beta_state.entries.len(),
            sel.len(),
            row_of.len()
        );
    }
    if history.depth() == 0 {
        if dbg {
            eprintln!("[clip-dbg] history.depth==0 (no beta entries) -> None");
        }
        return None;
    }

    let certified = token
        .validate_for_clip(
            pass, in_lo, in_hi, history, relu_name, sel, row_of, deadline,
        )
        .ok()?;
    if !folded_proposal_matches_certificate(folded, &certified, past_deadline) {
        return None;
    }

    // Split-constraint source rows: the seed IS the split pre-node on this lane, so a
    // split at `relu_name` neuron `j` maps to the folded row `row_of[j]` (Relu neuron j's
    // pre-activation is the seed neuron j). Any unmapped/refused premise → None → the
    // constraint is dropped (sound weakening).
    let split_closure = |node_name: &str,
                         neuron_idx: usize,
                         deadline_check: &mut F|
     -> Option<(Array1<f32>, f32, Array1<f32>, f32)> {
        if node_name != relu_name || neuron_idx >= row_of.len() {
            return None;
        }
        let row = row_of[neuron_idx];
        if row == usize::MAX || row >= certified.rows() {
            return None;
        }
        let d = certified.dim();
        let mut la = Array1::<f32>::zeros(d);
        let mut ua = Array1::<f32>::zeros(d);
        for j in 0..d {
            if j.is_multiple_of(SELECTIVE_DEADLINE_POLL_STRIDE) && deadline_check() {
                return None;
            }
            la[j] = certified.lower_a()[[row, j]];
            ua[j] = certified.upper_a()[[row, j]];
        }
        Some((la, certified.lower_b()[row], ua, certified.upper_b()[row]))
    };

    let constraints = match build_split_constraints_with_deadline_check(
        history,
        split_closure,
        x_dim,
        past_deadline,
    ) {
        Ok(c) => c,
        Err(e) => {
            if dbg {
                eprintln!(
                    "[clip-dbg] depth={} build_split_constraints ERR: {e}",
                    history.depth()
                );
            }
            return None;
        }
    };
    if constraints.is_empty() {
        if dbg {
            eprintln!("[clip-dbg] depth={} constraints EMPTY (split_closure mapped 0 premises; relu={relu_name})", history.depth());
        }
        return None;
    }
    if in_lo.len() != x_dim || in_hi.len() != x_dim || past_deadline() {
        return None;
    }
    let mut in_lo_arr = Array1::<f32>::zeros(x_dim);
    let mut in_hi_arr = Array1::<f32>::zeros(x_dim);
    for j in 0..x_dim {
        if j.is_multiple_of(SELECTIVE_DEADLINE_POLL_STRIDE) && past_deadline() {
            return None;
        }
        in_lo_arr[j] = in_lo[j];
        in_hi_arr[j] = in_hi[j];
    }
    let preprocessed = sort_out_constraints_with_deadline_check(
        &constraints,
        &in_lo_arr,
        &in_hi_arr,
        past_deadline,
    )
    .ok()?;
    if preprocessed.a_active.nrows() == 0 {
        if dbg {
            eprintln!(
                "[clip-dbg] depth={} a_active=0 (all {} constraints infeasible/fully-covered)",
                history.depth(),
                constraints.num_constraints
            );
        }
        return None;
    }
    if dbg {
        eprintln!(
            "[clip-dbg] depth={} constraints={} a_active={} -> RUNNING clip",
            history.depth(),
            constraints.num_constraints,
            preprocessed.a_active.nrows()
        );
    }

    // Objective rows = every `sel` row (the seed pre-activations to tighten).
    // Invalid/refused rows cannot enter the provenance token at all, so this arm
    // contains only finite outward enclosures.
    let n_obj = sel.len();
    if past_deadline() {
        return None;
    }
    let mut obj_la = Array2::<f32>::zeros((n_obj, x_dim));
    let mut obj_lb = Array1::<f32>::zeros(n_obj);
    let mut obj_ua = Array2::<f32>::zeros((n_obj, x_dim));
    let mut obj_ub = Array1::<f32>::zeros(n_obj);
    let mut objective_cells = 0usize;
    for r in 0..n_obj {
        if r.is_multiple_of(64) && past_deadline() {
            return None;
        }
        obj_lb[r] = certified.lower_b()[r];
        obj_ub[r] = certified.upper_b()[r];
        for j in 0..x_dim {
            if objective_cells.is_multiple_of(SELECTIVE_DEADLINE_POLL_STRIDE) && past_deadline() {
                return None;
            }
            obj_la[[r, j]] = certified.lower_a()[[r, j]];
            obj_ua[[r, j]] = certified.upper_a()[[r, j]];
            objective_cells = objective_cells.saturating_add(1);
        }
    }
    if past_deadline() {
        return None;
    }
    let (tl, tu) = tighten_with_constraints_with_deadline(
        &preprocessed,
        &obj_la,
        &obj_lb,
        &obj_ua,
        &obj_ub,
        &in_lo_arr,
        &in_hi_arr,
        deadline,
    )
    .ok()?;
    if past_deadline() {
        return None;
    }
    let mut lower = Vec::new();
    let mut upper = Vec::new();
    lower.try_reserve_exact(tl.len()).ok()?;
    upper.try_reserve_exact(tu.len()).ok()?;
    for (i, (&l, &u)) in tl.iter().zip(tu.iter()).enumerate() {
        if i.is_multiple_of(SELECTIVE_DEADLINE_POLL_STRIDE) && past_deadline() {
            return None;
        }
        lower.push(l);
        upper.push(u);
    }
    (!past_deadline()).then_some((lower, upper))
}

/// #clip-interm-guard: FAIL-CLOSED runtime enclosure check for ONE domain's clip.
/// Production authority is quarantined; explicit Leg-A tests call it directly.
/// Draws `restarts`
/// DIRECTED adversarial points in the child's input box — interior randoms plus
/// per-row extremal pushes toward the box CORNER that maximizes/minimizes a row's
/// folded objective (the vertex a too-tight bound would be exposed at) — keeps the
/// points that satisfy the child's ReLU split SIGNS at the seed layer (checked on
/// the TRUE forward through `graph`, exact on the degenerate point box), and
/// asserts every clip-tightened row `r` encloses the true seed value
/// `clip_lower[r] ≤ z_{sel[r]}(x) ≤ clip_upper[r]` (f32 forward tolerance).
///
/// Returns `Err((seed_neuron_idx, true_value, clip_lower, clip_upper))` on the
/// FIRST feasible point found outside its row's tightened box — proof the clip is
/// unsound for this child, so the caller REVERTS the whole seed node to the
/// inherited parent bound. `Ok(())` when every sampled feasible point is enclosed
/// (or no feasible sample was drawn — no evidence of unsoundness; the sound
/// scalar∩inherited intersect still bounds the merge). Refused rows carry no
/// tightening and are skipped.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn clip_guard_verify_domain(
    graph: &GraphNetwork,
    input_box: &BoundedTensor,
    seed_node: &str,
    relu_name: &str,
    beta: &GraphBetaState,
    sel: &[usize],
    folded: &FoldedSeedRows,
    clip_lower: &[f32],
    clip_upper: &[f32],
    restarts: usize,
    seed_rng: u64,
) -> Result<(), (usize, f32, f32, f32)> {
    let lo = input_box.lower();
    let hi = input_box.upper();
    let shape = lo.raw_dim();
    let lo_flat: Vec<f32> = lo.iter().copied().collect();
    let hi_flat: Vec<f32> = hi.iter().copied().collect();
    let dim = lo_flat.len();
    let n_rows = sel.len();
    if dim == 0 || clip_lower.len() != n_rows || clip_upper.len() != n_rows || folded.dim != dim {
        return Ok(()); // nothing checkable — the sound scalar∩inherited still guards
    }

    // Deterministic LCG in [0,1) (the tests_soundness sampling pattern).
    let mut state = seed_rng | 1;
    let mut u01 = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 40) as f32) / ((1u64 << 24) as f32)
    };

    // The premises the clip actually encoded: split SIGNS at the seed ReLU.
    let premises: Vec<(usize, f32, bool)> = beta
        .entries_for_node(relu_name)
        .map(|e| (e.neuron_idx(), e.split_point(), e.sign() > 0.0))
        .collect();

    for k in 0..restarts {
        // Half the restarts are DIRECTED corner pushes (rotate over row × dir),
        // the rest interior randoms.
        let mut x = vec![0.0f32; dim];
        if k % 2 == 0 && n_rows > 0 {
            let r = (k / 2) % n_rows;
            let test_upper = (k / 2 / n_rows).is_multiple_of(2);
            for d in 0..dim {
                let a = if test_upper {
                    folded.upper_a[r * dim + d]
                } else {
                    folded.lower_a[r * dim + d]
                };
                // Corner extremizing a·x over [lo,hi] in the violating direction.
                let corner = if (test_upper && a > 0.0) || (!test_upper && a < 0.0) {
                    hi_flat[d]
                } else {
                    lo_flat[d]
                };
                let span = hi_flat[d] - lo_flat[d];
                x[d] = (corner + (u01() - 0.5) * 0.15 * span).clamp(lo_flat[d], hi_flat[d]);
            }
        } else {
            for d in 0..dim {
                x[d] = lo_flat[d] + u01() * (hi_flat[d] - lo_flat[d]);
            }
        }

        let pt = match ndarray::ArrayD::from_shape_vec(shape.clone(), x)
            .ok()
            .and_then(|a| BoundedTensor::new(a.clone(), a).ok())
        {
            Some(p) => p,
            None => continue,
        };
        let Ok(vals) = graph.collect_node_bounds(&pt) else {
            continue;
        };
        let Some(seed_vals) = vals.get(seed_node) else {
            continue;
        };
        let seed_flat = seed_vals.flatten();
        let seed_len = seed_flat.len();
        let val_at =
            |idx: usize| -> Option<f32> { (idx < seed_len).then(|| seed_flat.lower()[[idx]]) };

        // Feasibility: the child's ReLU split signs at the seed layer.
        let feasible = premises
            .iter()
            .all(|&(j, s, active)| val_at(j).is_some_and(|v| if active { v >= s } else { v <= s }));
        if !feasible {
            continue;
        }

        // Enclosure: every clip-tightened row must contain the true seed value.
        for r in 0..n_rows {
            if folded.refused[r] {
                continue;
            }
            let Some(z) = val_at(sel[r]) else {
                continue;
            };
            let (cl, cu) = (clip_lower[r], clip_upper[r]);
            let tol = 1e-3 * (1.0 + z.abs());
            if (cl.is_finite() && z < cl - tol) || (cu.is_finite() && z > cu + tol) {
                return Err((sel[r], z, cl, cu));
            }
        }
    }
    Ok(())
}

/// Intersect one domain's refined `[l', u']` into its cache (the seed node's
/// pre-activation entry and the ReLU's post-activation entry). Row `r` of the
/// result refines neuron `sel[r]`; unselected neurons keep their inherited
/// pair. Returns whether the cache was updated (a length/shape/repair refusal
/// keeps the inherited entries and returns false — sound).
fn apply_refinement(
    cache: &mut HashMap<String, Arc<BoundedTensor>>,
    seed_node: &str,
    relu_name: &str,
    r: &ny_core::GpuCrownResult,
    pre_dim: usize,
    sel: &[usize],
    deadline: Option<Instant>,
    stats: &mut RefineStats,
) -> bool {
    apply_selected_bounds(
        cache,
        seed_node,
        relu_name,
        &r.lower_bounds,
        &r.upper_bounds,
        pre_dim,
        sel,
        deadline,
        stats,
    )
}

/// Intersect selected pre-activation rows into both the pre node and its exact
/// ReLU image.  Scalar intermediate refinement and Complete Clipping share this
/// one mutation boundary so neither path can accidentally replace (rather than
/// intersect) an inherited child enclosure.
#[allow(clippy::too_many_arguments)]
fn apply_selected_bounds(
    cache: &mut HashMap<String, Arc<BoundedTensor>>,
    seed_node: &str,
    relu_name: &str,
    lower_bounds: &[f32],
    upper_bounds: &[f32],
    pre_dim: usize,
    sel: &[usize],
    deadline: Option<Instant>,
    stats: &mut RefineStats,
) -> bool {
    let mut past_deadline = || deadline.is_some_and(|deadline| Instant::now() >= deadline);
    apply_selected_bounds_with_deadline_check(
        cache,
        seed_node,
        relu_name,
        lower_bounds,
        upper_bounds,
        pre_dim,
        sel,
        stats,
        &mut past_deadline,
    )
}

#[allow(clippy::too_many_arguments)]
fn apply_selected_bounds_with_deadline_check<F>(
    cache: &mut HashMap<String, Arc<BoundedTensor>>,
    seed_node: &str,
    relu_name: &str,
    lower_bounds: &[f32],
    upper_bounds: &[f32],
    pre_dim: usize,
    sel: &[usize],
    stats: &mut RefineStats,
    past_deadline: &mut F,
) -> bool
where
    F: FnMut() -> bool,
{
    if past_deadline() {
        return false;
    }
    if lower_bounds.len() != sel.len() || upper_bounds.len() != sel.len() {
        return false;
    }
    let Some(old) = cache.get(seed_node) else {
        return false;
    };
    if old.len() != pre_dim {
        return false;
    }
    let Some(old_post) = cache.get(relu_name) else {
        return false;
    };
    if old_post.len() != pre_dim {
        return false;
    }
    // Row index of each selected neuron (usize::MAX = unselected).
    let mut row_of = vec![usize::MAX; pre_dim];
    for (row, &j) in sel.iter().enumerate() {
        if row.is_multiple_of(256) && past_deadline() {
            return false;
        }
        if j >= pre_dim || row_of[j] != usize::MAX {
            return false;
        }
        let (refined_lower, refined_upper) = (lower_bounds[row], upper_bounds[row]);
        if !refined_lower.is_finite() || !refined_upper.is_finite() || refined_lower > refined_upper
        {
            return false;
        }
        row_of[j] = row;
    }
    let mut new_l = Vec::with_capacity(pre_dim);
    let mut new_u = Vec::with_capacity(pre_dim);
    let mut tightened = 0usize;
    let mut newly_stable = 0usize;
    let crossings = 0usize;
    let mut w_before = 0.0f64;
    let mut w_after = 0.0f64;
    for (j, (&ol, &ou)) in old.lower().iter().zip(old.upper().iter()).enumerate() {
        if j.is_multiple_of(256) && past_deadline() {
            return false;
        }
        if ol.is_nan() || ou.is_nan() || ol > ou {
            return false;
        }
        let (nl, nu, t) = if row_of[j] == usize::MAX {
            (ol, ou, false)
        } else {
            let (rl, ru) = (lower_bounds[row_of[j]], upper_bounds[row_of[j]]);
            // Two certified enclosures of one nonempty child must intersect.
            // A disjoint proposal is a numerical/provenance refusal for the
            // whole target, not a completed partial application.
            if ol.max(rl) > ou.min(ru) {
                return false;
            }
            intersect_pair(ol, ou, rl, ru)
        };
        if t {
            tightened += 1;
            // Stability flip: the pruning lever — the margin backward's
            // relaxation of this neuron becomes EXACT (slope 1 or 0).
            if ol < 0.0 && ou > 0.0 && (nl >= 0.0 || nu <= 0.0) {
                newly_stable += 1;
            }
        }
        if ol.is_finite() && ou.is_finite() {
            w_before += (ou - ol) as f64;
            w_after += (nu - nl) as f64;
        }
        new_l.push(nl);
        new_u.push(nu);
    }
    let shape = old.lower().raw_dim();
    let Ok(lower) = ndarray::ArrayD::from_shape_vec(shape.clone(), new_l.clone()) else {
        return false;
    };
    let Ok(upper) = ndarray::ArrayD::from_shape_vec(shape, new_u.clone()) else {
        return false;
    };
    let Ok(refined_pre) = BoundedTensor::new(lower, upper) else {
        return false;
    };

    // The ReLU's own POST-activation entry: relu is exact + monotone, so
    // [relu(l'), relu(u')] encloses the post-activations; intersect with the
    // inherited post entry. Refused (kept inherited) on any shape mismatch.
    let refined_post = {
        let post = old_post;
        let mut pl = Vec::with_capacity(pre_dim);
        let mut pu = Vec::with_capacity(pre_dim);
        for (j, (&ol, &ou)) in post.lower().iter().zip(post.upper().iter()).enumerate() {
            if j.is_multiple_of(256) && past_deadline() {
                return false;
            }
            if ol.is_nan() || ou.is_nan() || ol > ou {
                return false;
            }
            let mapped_lower = new_l[j].max(0.0);
            let mapped_upper = new_u[j].max(0.0);
            if ol.max(mapped_lower) > ou.min(mapped_upper) {
                return false;
            }
            let (nl, nu, _) = intersect_pair(ol, ou, mapped_lower, mapped_upper);
            pl.push(nl);
            pu.push(nu);
        }
        let shape = post.lower().raw_dim();
        let Ok(lower) = ndarray::ArrayD::from_shape_vec(shape.clone(), pl) else {
            return false;
        };
        let Ok(upper) = ndarray::ArrayD::from_shape_vec(shape, pu) else {
            return false;
        };
        let Ok(refined) = BoundedTensor::new(lower, upper) else {
            return false;
        };
        refined
    };

    if past_deadline() {
        return false;
    }
    // Fresh Arcs: refinement replaces entries wholesale, never mutates a
    // shared tensor in place (#cone-delta increment 2 aliasing rule).
    cache.insert(seed_node.to_string(), Arc::new(refined_pre));
    cache.insert(relu_name.to_string(), Arc::new(refined_post));
    stats.neurons_tightened += tightened;
    stats.newly_stable += newly_stable;
    stats.crossings_kept += crossings;
    stats.width_before += w_before;
    stats.width_after += w_after;
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::{AddLayer, LinearLayer, ReLULayer};
    use crate::GraphNode;
    use ndarray::{arr1, arr2, ArrayD, IxDyn, ShapeBuilder};

    fn lin(name: &str, input: &str) -> GraphNode {
        let w = arr2(&[[0.7_f32, -0.3], [0.2, 0.6]]);
        let b = arr1(&[0.05_f32, -0.04]);
        let layer = Layer::Linear(LinearLayer::new(w, Some(b)).expect("valid linear"));
        if input == NETWORK_INPUT {
            GraphNode::from_input(name, layer)
        } else {
            GraphNode::new(name, layer, vec![input.to_string()])
        }
    }
    fn relu(name: &str, input: &str) -> GraphNode {
        GraphNode::new(name, Layer::ReLU(ReLULayer), vec![input.to_string()])
    }
    fn selector_graph() -> GraphNetwork {
        let mut graph = GraphNetwork::new();
        graph.add_node(lin("seed", NETWORK_INPUT));
        graph.add_node(relu("relu", "seed"));
        graph.set_output("relu");
        graph
    }

    #[test]
    fn optional_complete_clip_work_requires_three_seconds_of_start_budget() {
        let now = Instant::now();
        assert!(complete_clip_optional_start_budget_available(now, None));
        assert!(!complete_clip_optional_start_budget_available(
            now,
            Some(now + std::time::Duration::from_secs(3))
        ));
        assert!(complete_clip_optional_start_budget_available(
            now,
            Some(now + std::time::Duration::from_secs(4))
        ));
    }

    #[test]
    fn complete_clip_deadline_scopes_choose_earliest_and_restore() {
        let scopes = CompleteClipDeadlineOverrides::default();
        let now = Instant::now();
        let configured = now + std::time::Duration::from_secs(30);
        let later = now + std::time::Duration::from_secs(20);
        let earlier = now + std::time::Duration::from_secs(10);

        assert_eq!(scopes.effective(Some(configured)), Some(configured));
        let inert = scopes.scoped(None);
        assert_eq!(scopes.effective(None), None);
        let later_guard = scopes.scoped(Some(later));
        assert_eq!(scopes.effective(Some(configured)), Some(later));
        let earlier_guard = scopes.scoped(Some(earlier));
        assert_eq!(scopes.effective(Some(configured)), Some(earlier));
        assert_eq!(scopes.effective(None), Some(earlier));

        // Scopes may finish out of nesting order when verifier calls overlap.
        drop(later_guard);
        assert_eq!(scopes.effective(Some(configured)), Some(earlier));
        drop(earlier_guard);
        assert_eq!(scopes.effective(Some(configured)), Some(configured));
        drop(inert);

        let unwind = std::panic::catch_unwind(|| {
            let _guard = scopes.scoped(Some(earlier));
            panic!("exercise deadline-guard unwind");
        });
        assert!(unwind.is_err());
        assert_eq!(scopes.effective(None), None);
    }

    #[test]
    fn complete_clip_suppression_scopes_overlap_and_restore() {
        let scopes = Arc::new(CompleteClipDeadlineOverrides::default());
        assert!(!scopes.complete_clip_suppressed());

        let outer = scopes.suppress_complete_clip_scoped();
        let inner = scopes.suppress_complete_clip_scoped();
        assert!(scopes.complete_clip_suppressed());

        // Overlapping verifier calls need not finish in stack order.
        drop(outer);
        assert!(scopes.complete_clip_suppressed());
        drop(inner);
        assert!(!scopes.complete_clip_suppressed());

        let unwind = std::panic::catch_unwind({
            let scopes = Arc::clone(&scopes);
            move || {
                let _guard = scopes.suppress_complete_clip_scoped();
                panic!("exercise Complete Clipping suppression unwind");
            }
        });
        assert!(unwind.is_err());
        assert!(!scopes.complete_clip_suppressed());

        // The state is verifier-wide, so an advisory scope remains visible
        // when propagation is entered by another worker.
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let worker_scopes = Arc::clone(&scopes);
        let worker_barrier = Arc::clone(&barrier);
        let worker = std::thread::spawn(move || {
            let _guard = worker_scopes.suppress_complete_clip_scoped();
            worker_barrier.wait();
            worker_barrier.wait();
        });
        barrier.wait();
        assert!(scopes.complete_clip_suppressed());
        barrier.wait();
        worker.join().expect("suppression worker joins");
        assert!(!scopes.complete_clip_suppressed());
    }

    /// input → l1 → relu1 → l2 → add(l2, l1) → gemm_pre → relu_last → out.
    /// The walk from `out` must find (relu_last, gemm_pre).
    #[test]
    fn find_last_relu_seed_walks_the_unary_chain() {
        let mut g = GraphNetwork::new();
        g.add_node(lin("l1", NETWORK_INPUT));
        g.add_node(relu("relu1", "l1"));
        g.add_node(lin("l2", "relu1"));
        g.add_node(GraphNode::new(
            "add",
            Layer::Add(AddLayer),
            vec!["l2".to_string(), "l1".to_string()],
        ));
        g.add_node(lin("gemm_pre", "add"));
        g.add_node(relu("relu_last", "gemm_pre"));
        g.add_node(lin("out", "relu_last"));
        g.set_output("out");

        let (relu_name, seed) = find_last_relu_seed(&g, "out").expect("chain must resolve");
        assert_eq!(relu_name, "relu_last");
        assert_eq!(seed, "gemm_pre");
    }

    /// A binary node between the output and the first ReLU refuses (no clean
    /// last-ReLU chain), and a ReLU fed by the network input refuses too.
    #[test]
    fn find_last_relu_seed_refuses_non_unary_and_input_fed() {
        let mut g = GraphNetwork::new();
        g.add_node(lin("l1", NETWORK_INPUT));
        g.add_node(lin("l2", NETWORK_INPUT));
        g.add_node(GraphNode::new(
            "add",
            Layer::Add(AddLayer),
            vec!["l1".to_string(), "l2".to_string()],
        ));
        g.set_output("add");
        assert!(
            find_last_relu_seed(&g, "add").is_none(),
            "binary node before any ReLU must refuse"
        );

        let mut g2 = GraphNetwork::new();
        g2.add_node(relu("r0", NETWORK_INPUT));
        g2.add_node(lin("out", "r0"));
        g2.set_output("out");
        assert!(
            find_last_relu_seed(&g2, "out").is_none(),
            "ReLU fed by the network input has nothing below the seed"
        );
    }

    #[test]
    fn complete_clip_root_cache_requires_exact_graph_and_homogeneous_box() {
        let graph = GraphNetwork::new();
        let root = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![-0.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(),
        )
        .unwrap();
        let marker = Arc::new(
            BoundedTensor::new(
                ArrayD::from_shape_vec(IxDyn(&[1]), vec![-2.0]).unwrap(),
                ArrayD::from_shape_vec(IxDyn(&[1]), vec![3.0]).unwrap(),
            )
            .unwrap(),
        );
        let bounds = HashMap::from([("marker".to_string(), Arc::clone(&marker))]);
        let cache = CompleteClipRootBoundsCache::default();
        assert!(cache.store_finalized(&graph, &root, &bounds));
        assert!(
            cache
                .get_finalized_for_batch(&graph, std::slice::from_ref(&root), None)
                .is_none(),
            "an unstamped outer BaB round must fail closed"
        );
        assert!(!cache.set_bab_iteration(&graph, &root, 0));
        assert!(cache.set_bab_iteration(&graph, &root, 1));

        let (hit, iteration) = cache
            .get_finalized_for_batch(&graph, &[root.clone(), root.clone()], None)
            .expect("exact homogeneous root batch must hit");
        assert_eq!(iteration, 1);
        assert!(Arc::ptr_eq(&hit["marker"], &marker));
        let (_, repeated_iteration) = cache
            .get_finalized_for_batch(&graph, std::slice::from_ref(&root), None)
            .expect("repeated CROWN call in the same wave must hit");
        assert_eq!(
            repeated_iteration, 1,
            "cache lookups must not advance the outer BaB round"
        );
        assert!(cache.set_bab_iteration(&graph, &root, 2));
        let (_, second_iteration) = cache
            .get_finalized_for_batch(&graph, std::slice::from_ref(&root), None)
            .expect("explicitly advanced outer round must hit");
        assert_eq!(second_iteration, 2);

        // Signed zero is part of the exact input-box identity.
        let positive_zero = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(),
        )
        .unwrap();
        assert!(cache
            .get_finalized_for_batch(&graph, std::slice::from_ref(&positive_zero), None)
            .is_none());
        assert!(cache
            .get_finalized_for_batch(&graph, &[root.clone(), positive_zero], None)
            .is_none());

        // A same-shaped but independently scoped graph cannot consume the row
        // authority, and an already-expired deadline cannot return a hit.
        let other_graph = GraphNetwork::new();
        assert!(!cache.set_bab_iteration(&other_graph, &root, 3));
        assert!(cache
            .get_finalized_for_batch(&other_graph, std::slice::from_ref(&root), None)
            .is_none());
        let (_, iteration_after_mismatch) = cache
            .get_finalized_for_batch(&graph, std::slice::from_ref(&root), None)
            .expect("foreign setter must leave the exact entry unchanged");
        assert_eq!(iteration_after_mismatch, 2);
        assert!(cache
            .get_finalized_for_batch(
                &graph,
                std::slice::from_ref(&root),
                Some(
                    Instant::now()
                        .checked_sub(std::time::Duration::from_millis(1))
                        .expect("current instant admits a one-millisecond expired fixture"),
                ),
            )
            .is_none());

        assert!(cache.store_finalized(&graph, &root, &bounds));
        assert!(
            cache
                .get_finalized_for_batch(&graph, std::slice::from_ref(&root), None)
                .is_none(),
            "a new finalized root must reset the outer-round stamp"
        );
    }

    #[test]
    fn complete_clip_root_affine_cache_enforces_payload_budget_with_lru_eviction() {
        let graph = GraphNetwork::new();
        let root = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0, -1.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.0]).unwrap(),
        )
        .unwrap();
        let marker = Arc::new(root.clone());
        let bounds = HashMap::from([("marker".to_string(), marker)]);
        let cache = CompleteClipRootBoundsCache::default();
        assert!(cache.store_finalized(&graph, &root, &bounds));

        let first_map = [0, usize::MAX];
        let first = Arc::new(
            capture_exact_root_input_rows(&graph, &[0], &first_map, &root, None)
                .expect("first exact root row"),
        );
        let second_map = [usize::MAX, 0];
        let second = Arc::new(
            capture_exact_root_input_rows(&graph, &[1], &second_map, &root, None)
                .expect("second exact root row"),
        );
        let budget = first
            .resident_payload_bytes()
            .expect("finite first payload")
            .max(
                second
                    .resident_payload_bytes()
                    .expect("finite second payload"),
            );

        cache.store_affine_rows_with_limits(
            &graph,
            &root,
            NETWORK_INPUT,
            &[0],
            first_map.len(),
            Arc::clone(&first),
            budget,
            64,
        );
        assert!(cache
            .get_affine_rows(&graph, &root, NETWORK_INPUT, &[0], first_map.len())
            .is_some());

        cache.store_affine_rows_with_limits(
            &graph,
            &root,
            NETWORK_INPUT,
            &[1],
            second_map.len(),
            Arc::clone(&second),
            budget,
            64,
        );
        assert!(
            cache
                .get_affine_rows(&graph, &root, NETWORK_INPUT, &[0], first_map.len())
                .is_none(),
            "the least-recent payload must be evicted before the byte cap is exceeded"
        );
        assert!(cache
            .get_affine_rows(&graph, &root, NETWORK_INPUT, &[1], second_map.len())
            .is_some());

        let guard = cache.inner.lock().unwrap();
        let entry = guard.as_ref().expect("root entry");
        assert_eq!(entry.affine_rows.len(), 1);
        assert!(entry.affine_rows_bytes <= budget);
    }

    /// The gate is dark: unset/other values are OFF, only "1" is ON. (Pure
    /// parse check via the env contract — matches!(…, Some("1")).)
    #[test]
    fn interm_refine_gate_semantics() {
        // The gate reads the env var directly; assert the default-off contract
        // by construction: matches!(None | Some("0"), Some("1")) is false.
        assert!(!matches!(None::<&str>, Some("1")));
        assert!(!matches!(Some("0"), Some("1")));
        assert!(matches!(Some("1"), Some("1")));
    }

    /// Adaptive-schedule latch decision (#adaptive-refine): only a zero-yield
    /// batch (no newly-stable neuron, no infeasibility prune) at depth ≥ floor
    /// may latch; production of either kind, or a shallow batch, never does.
    #[test]
    fn adaptive_latch_decision_is_yield_and_floor_gated() {
        // Zero yield at/above the floor ⇒ latch.
        assert!(adaptive_should_latch(0, 0, 4, 4));
        assert!(adaptive_should_latch(0, 0, 9, 4));
        // Any production ⇒ never latch, at any depth.
        assert!(!adaptive_should_latch(1, 0, 9, 4));
        assert!(!adaptive_should_latch(0, 1, 9, 4));
        // Below the floor ⇒ never latch (early refine is protected).
        assert!(!adaptive_should_latch(0, 0, 3, 4));
        assert!(!adaptive_should_latch(0, 0, 0, 4));
        // The gate itself is dark (same env contract as NY_INTERM_REFINE).
        assert!(!matches!(None::<&str>, Some("1")));
    }

    #[test]
    fn complete_clip_batch_requires_every_capture_and_target_before_latching() {
        let completed = ReclipOutcome {
            completed: true,
            targets_expected: 1,
            targets_completed: 1,
            ..ReclipOutcome::default()
        };
        assert!(complete_clip_batch_completed(2, 2, 0, Some(completed)));
        assert!(!complete_clip_batch_completed(2, 1, 1, Some(completed)));
        assert!(!complete_clip_batch_completed(2, 2, 0, None));
        assert!(!complete_clip_batch_completed(
            2,
            2,
            0,
            Some(ReclipOutcome {
                completed: false,
                targets_expected: 1,
                targets_refused: 1,
                ..ReclipOutcome::default()
            }),
        ));
        assert!(!complete_clip_batch_completed(
            2,
            2,
            0,
            Some(ReclipOutcome {
                completed: false,
                deadline_interrupted: true,
                ..ReclipOutcome::default()
            }),
        ));
    }

    /// Per-child re-refinement (#interm-refine-redo): under a tripped latch,
    /// only sub-latch depths refine when the stride is 0 (the dark default);
    /// with stride `d`, exact depth multiples of `d` re-refine past the latch
    /// and every other deep depth keeps the skip.
    #[test]
    fn redo_reincludes_exact_depth_multiples_under_latch() {
        // Stride 0 (default): pre-redo behavior exactly.
        assert!(latched_domain_refines(6, 7, 0));
        assert!(!latched_domain_refines(7, 7, 0));
        assert!(!latched_domain_refines(16, 7, 0));
        // Stride 8: depths 8/16/24 re-refine; 9..15 and 17 keep the skip.
        assert!(latched_domain_refines(8, 7, 8));
        assert!(latched_domain_refines(16, 7, 8));
        assert!(latched_domain_refines(24, 7, 8));
        assert!(!latched_domain_refines(9, 7, 8));
        assert!(!latched_domain_refines(15, 7, 8));
        assert!(!latched_domain_refines(17, 7, 8));
        // Stride 2: every even deep depth re-refines.
        assert!(latched_domain_refines(8, 7, 2));
        assert!(latched_domain_refines(10, 7, 2));
        assert!(!latched_domain_refines(9, 7, 2));
        // Sub-latch depths always refine regardless of stride.
        assert!(latched_domain_refines(3, 7, 8));
        assert!(latched_domain_refines(0, 7, 8));
    }

    /// #wide-chunk sizing: 0 = OFF (whole batch, one call); otherwise
    /// `max(1, wide_max_n / n_rows)` domains per call.
    #[test]
    fn wide_chunk_domains_sizing() {
        // Gate off: the whole batch in one call regardless of rows.
        assert_eq!(wide_chunk_domains(0, 69, 32), 32);
        assert_eq!(wide_chunk_domains(0, 0, 5), 5);
        // Measured prop54 deep pass: 69 rows/domain, cap 2048 wide rows
        // ⇒ 29 domains per call (2001 rows — under both device caps).
        assert_eq!(wide_chunk_domains(2048, 69, 32), 29);
        // Row-capped deep pass (16 rows): 128 domains per call.
        assert_eq!(wide_chunk_domains(2048, 16, 32), 128);
        // A single domain over the cap still runs (chunk of 1).
        assert_eq!(wide_chunk_domains(64, 100, 8), 1);
        // Zero selected rows: degenerate, whole batch (call refuses upstream).
        assert_eq!(wide_chunk_domains(2048, 0, 4), 4);
        // Empty idxs never yields 0 (chunks(0) would panic).
        assert_eq!(wide_chunk_domains(0, 3, 0), 1);
    }

    /// Intersection semantics: tightening applies, crossings/NaN/non-finite
    /// refined pairs keep the inherited pair (sound).
    #[test]
    fn intersect_pair_is_sound_and_conservative() {
        // Plain tightening on both sides.
        let (l, u, t) = intersect_pair(-2.0, 3.0, -1.0, 2.0);
        assert_eq!((l, u), (-1.0, 2.0));
        assert!(t);
        // Refined looser than inherited → unchanged, not tightened.
        let (l, u, t) = intersect_pair(-0.5, 0.5, -3.0, 3.0);
        assert_eq!((l, u), (-0.5, 0.5));
        assert!(!t);
        // Empty intersection (disjoint sound-in-exact-math edge) → keep inherited.
        let (l, u, t) = intersect_pair(1.0, 2.0, -3.0, 0.5);
        assert_eq!((l, u), (1.0, 2.0));
        assert!(!t);
        // Non-finite refined → keep inherited.
        let (l, u, t) = intersect_pair(-1.0, 1.0, f32::NEG_INFINITY, f32::NAN);
        assert_eq!((l, u), (-1.0, 1.0));
        assert!(!t);
        // Inverted refined → keep inherited.
        let (l, u, t) = intersect_pair(-1.0, 1.0, 0.5, -0.5);
        assert_eq!((l, u), (-1.0, 1.0));
        assert!(!t);
        // NaN inherited must NOT be silently repaired by a finite refined pair.
        let (l, _u, t) = intersect_pair(f32::NAN, 1.0, -0.5, 0.5);
        assert!(l.is_nan(), "NaN inherited lower must propagate");
        assert!(!t);
    }

    /// apply_refinement: pre entry intersected, post entry gets relu-mapped
    /// refinement, stability flip counted, and a wrong-length result refuses.
    #[test]
    fn apply_refinement_updates_pre_and_post_entries() {
        let mk = |l: &[f32], u: &[f32]| {
            BoundedTensor::new(
                ArrayD::from_shape_vec(IxDyn(&[l.len()]), l.to_vec()).unwrap(),
                ArrayD::from_shape_vec(IxDyn(&[u.len()]), u.to_vec()).unwrap(),
            )
            .unwrap()
        };
        let mut cache: HashMap<String, Arc<BoundedTensor>> = HashMap::new();
        // Neuron 0 unstable, neuron 1 unstable (will flip stable-active).
        cache.insert("pre".into(), Arc::new(mk(&[-2.0, -1.0], &[3.0, 4.0])));
        cache.insert("relu".into(), Arc::new(mk(&[0.0, 0.0], &[3.0, 4.0])));
        let r = ny_core::GpuCrownResult {
            lower_bounds: vec![-1.5, 0.25],
            upper_bounds: vec![2.0, 3.0],
        };
        let mut stats = RefineStats::default();
        assert!(apply_refinement(
            &mut cache,
            "pre",
            "relu",
            &r,
            2,
            &[0, 1],
            None,
            &mut stats
        ));
        let pre = &cache["pre"];
        assert_eq!(pre.lower().as_slice().unwrap(), &[-1.5, 0.25]);
        assert_eq!(pre.upper().as_slice().unwrap(), &[2.0, 3.0]);
        let post = &cache["relu"];
        // post = [relu(l'), relu(u')] ∩ inherited = [0,2] and [0.25,3].
        assert_eq!(post.lower().as_slice().unwrap(), &[0.0, 0.25]);
        assert_eq!(post.upper().as_slice().unwrap(), &[2.0, 3.0]);
        assert_eq!(stats.neurons_tightened, 2);
        assert_eq!(stats.newly_stable, 1, "neuron 1 flipped stable-active");

        // Selective rows: sel=[1] refines only neuron 1; neuron 0 keeps the
        // inherited pair (unstable-only row selection).
        let mut cache_sel: HashMap<String, Arc<BoundedTensor>> = HashMap::new();
        cache_sel.insert("pre".into(), Arc::new(mk(&[-2.0, -1.0], &[3.0, 4.0])));
        cache_sel.insert("relu".into(), Arc::new(mk(&[0.0, 0.0], &[3.0, 4.0])));
        let r_sel = ny_core::GpuCrownResult {
            lower_bounds: vec![-0.5],
            upper_bounds: vec![2.5],
        };
        let mut stats_sel = RefineStats::default();
        assert!(apply_refinement(
            &mut cache_sel,
            "pre",
            "relu",
            &r_sel,
            2,
            &[1],
            None,
            &mut stats_sel
        ));
        let pre_sel = &cache_sel["pre"];
        assert_eq!(pre_sel.lower().as_slice().unwrap(), &[-2.0, -0.5]);
        assert_eq!(pre_sel.upper().as_slice().unwrap(), &[3.0, 2.5]);
        assert_eq!(stats_sel.neurons_tightened, 1);

        // Wrong-length refinement refuses and leaves the cache untouched.
        let r_bad = ny_core::GpuCrownResult {
            lower_bounds: vec![0.0],
            upper_bounds: vec![0.0],
        };
        let before = cache["pre"].lower().to_owned();
        let mut stats2 = RefineStats::default();
        assert!(!apply_refinement(
            &mut cache,
            "pre",
            "relu",
            &r_bad,
            2,
            &[0, 1],
            None,
            &mut stats2
        ));
        assert_eq!(cache["pre"].lower(), &before);

        // The pre/post pair is one atomic cache update. A missing post entry
        // refuses without replacing the otherwise valid pre entry.
        let original_pre = Arc::new(mk(&[-2.0, -1.0], &[3.0, 4.0]));
        let mut missing_post = HashMap::from([("pre".to_string(), Arc::clone(&original_pre))]);
        let mut missing_post_stats = RefineStats::default();
        assert!(!apply_refinement(
            &mut missing_post,
            "pre",
            "relu",
            &r,
            2,
            &[0, 1],
            None,
            &mut missing_post_stats,
        ));
        assert!(Arc::ptr_eq(&missing_post["pre"], &original_pre));
        assert_eq!(missing_post_stats.neurons_tightened, 0);

        // Invalid, duplicate, or disjoint selected rows refuse the entire
        // pre/post update and retain both original Arcs.
        for (bad_lower, bad_upper, bad_sel) in [
            (vec![f32::NEG_INFINITY], vec![1.0], vec![0]),
            (vec![1.0], vec![-1.0], vec![0]),
            (vec![4.0], vec![5.0], vec![0]),
            (vec![-1.0, -1.0], vec![1.0, 1.0], vec![0, 0]),
        ] {
            let original_pre = Arc::new(mk(&[-2.0, -1.0], &[3.0, 4.0]));
            let original_post = Arc::new(mk(&[0.0, 0.0], &[3.0, 4.0]));
            let mut guarded = HashMap::from([
                ("pre".to_string(), Arc::clone(&original_pre)),
                ("relu".to_string(), Arc::clone(&original_post)),
            ]);
            let mut guarded_stats = RefineStats::default();
            assert!(!apply_selected_bounds(
                &mut guarded,
                "pre",
                "relu",
                &bad_lower,
                &bad_upper,
                2,
                &bad_sel,
                None,
                &mut guarded_stats,
            ));
            assert!(Arc::ptr_eq(&guarded["pre"], &original_pre));
            assert!(Arc::ptr_eq(&guarded["relu"], &original_post));
            assert_eq!(guarded_stats.neurons_tightened, 0);
        }

        // Deadline expiry during either full-width scan is an atomic refusal:
        // neither cache entry nor the stats may be partially updated.
        for expire_on_call in [4usize, 6usize] {
            let original_pre = Arc::new(mk(&vec![-1.0; 512], &vec![1.0; 512]));
            let original_post = Arc::new(mk(&vec![0.0; 512], &vec![1.0; 512]));
            let mut guarded = HashMap::from([
                ("pre".to_string(), Arc::clone(&original_pre)),
                ("relu".to_string(), Arc::clone(&original_post)),
            ]);
            let lower = vec![-0.5; 512];
            let upper = vec![0.5; 512];
            let selected: Vec<usize> = (0..512).collect();
            let mut calls = 0usize;
            let mut expire = || {
                calls += 1;
                calls >= expire_on_call
            };
            let mut guarded_stats = RefineStats::default();
            assert!(!apply_selected_bounds_with_deadline_check(
                &mut guarded,
                "pre",
                "relu",
                &lower,
                &upper,
                512,
                &selected,
                &mut guarded_stats,
                &mut expire,
            ));
            assert!(Arc::ptr_eq(&guarded["pre"], &original_pre));
            assert!(Arc::ptr_eq(&guarded["relu"], &original_post));
            assert_eq!(guarded_stats.neurons_tightened, 0);
        }
    }

    use crate::beta_crown::state::GraphBetaEntry;

    fn beta_with(entries: Vec<(usize, f32, f32)>) -> GraphBetaState {
        // (neuron_idx, split_point, sign) at node "relu_last".
        GraphBetaState::from_entries(
            entries
                .into_iter()
                .map(|(idx, s, sign)| {
                    GraphBetaEntry::new("relu_last".into(), idx, s, 0.0, sign).expect("valid entry")
                })
                .collect(),
        )
    }

    #[test]
    fn complete_clip_beta_state_requires_exact_ordered_relu_history() {
        use crate::beta_crown::branching::GraphNeuronConstraint;

        let mut history = GraphSplitHistory::new();
        history
            .add_constraint(GraphNeuronConstraint::new("relu_last".into(), 2, true, 0.0).unwrap());
        history.add_constraint(
            GraphNeuronConstraint::new("relu_other".into(), 7, false, 0.0).unwrap(),
        );
        let entry = |node: &str, neuron: usize, split: f32, value: f32, sign: f32| {
            GraphBetaEntry::new(node.into(), neuron, split, value, sign).unwrap()
        };

        let exact = GraphBetaState::from_entries(vec![
            entry("relu_last", 2, 0.0, 0.25, 1.0),
            entry("relu_other", 7, 0.0, 0.5, -1.0),
        ]);
        assert!(beta_state_matches_pure_relu_history(&history, &exact));

        for mismatched in [
            GraphBetaState::from_entries(vec![
                entry("relu_other", 7, 0.0, 0.5, -1.0),
                entry("relu_last", 2, 0.0, 0.25, 1.0),
            ]),
            GraphBetaState::from_entries(vec![
                entry("wrong", 2, 0.0, 0.25, 1.0),
                entry("relu_other", 7, 0.0, 0.5, -1.0),
            ]),
            GraphBetaState::from_entries(vec![
                entry("relu_last", 3, 0.0, 0.25, 1.0),
                entry("relu_other", 7, 0.0, 0.5, -1.0),
            ]),
            GraphBetaState::from_entries(vec![
                entry("relu_last", 2, 0.0, 0.25, -1.0),
                entry("relu_other", 7, 0.0, 0.5, -1.0),
            ]),
            GraphBetaState::from_entries(vec![
                entry("relu_last", 2, 0.125, 0.25, 1.0),
                entry("relu_other", 7, 0.0, 0.5, -1.0),
            ]),
            GraphBetaState::from_entries(vec![entry("relu_last", 2, 0.0, 0.25, 1.0)]),
        ] {
            assert!(!beta_state_matches_pure_relu_history(&history, &mismatched));
        }

        let mut invalid_value = exact;
        invalid_value.entries[0].value = f32::NAN;
        assert!(!beta_state_matches_pure_relu_history(
            &history,
            &invalid_value
        ));
        invalid_value.entries[0].value = -1.0;
        assert!(!beta_state_matches_pure_relu_history(
            &history,
            &invalid_value
        ));
    }

    /// The prune core: an ACTIVE premise `z ≥ s` contradicted by refined
    /// `u' < s − tol` (and the INACTIVE mirror) fires; anything within tol,
    /// feasible, unselected, or non-finite never fires.
    #[test]
    fn premise_contradiction_fires_only_on_strict_violation() {
        let row_of = vec![0usize, usize::MAX]; // neuron 0 → row 0; neuron 1 unselected
        let tol = 1e-4;

        // ACTIVE premise z0 ≥ 0, refined enclosure entirely below zero → fires.
        let beta = beta_with(vec![(0, 0.0, 1.0)]);
        let hit = premise_contradiction(&beta, "relu_last", &row_of, &[-1.0], &[-0.25], tol);
        assert_eq!(hit, Some((0, 1.0, 0.25)));
        // Wrong node name → no premises → never fires.
        assert!(
            premise_contradiction(&beta, "other_relu", &row_of, &[-1.0], &[-0.25], tol).is_none()
        );

        // INACTIVE premise z0 ≤ 0, refined entirely above zero → fires.
        let beta = beta_with(vec![(0, 0.0, -1.0)]);
        let hit = premise_contradiction(&beta, "relu_last", &row_of, &[0.5], &[2.0], tol);
        assert_eq!(hit, Some((0, -1.0, 0.5)));

        // Feasible: refined straddles the premise threshold → no prune.
        let beta = beta_with(vec![(0, 0.0, 1.0)]);
        assert!(premise_contradiction(&beta, "relu_last", &row_of, &[-0.5], &[1.5], tol).is_none());

        // Within tolerance: violation must EXCEED tol (u' = −tol/2 does not).
        assert!(
            premise_contradiction(&beta, "relu_last", &row_of, &[-1.0], &[-tol / 2.0], tol)
                .is_none()
        );

        // Non-zero split point s: active premise z ≥ 1 with u' = 0.5 fires.
        let beta = beta_with(vec![(0, 1.0, 1.0)]);
        let hit = premise_contradiction(&beta, "relu_last", &row_of, &[-1.0], &[0.5], tol);
        assert_eq!(hit, Some((0, 1.0, 0.5)));

        // Unselected premise neuron (row usize::MAX) → skipped, never fires.
        let beta = beta_with(vec![(1, 0.0, 1.0)]);
        assert!(
            premise_contradiction(&beta, "relu_last", &row_of, &[-1.0], &[-1.0], tol).is_none()
        );

        // Non-finite / inverted refined pair → skipped (conservative).
        let beta = beta_with(vec![(0, 0.0, 1.0)]);
        assert!(
            premise_contradiction(&beta, "relu_last", &row_of, &[f32::NAN], &[-1.0], tol).is_none()
        );
        assert!(
            premise_contradiction(&beta, "relu_last", &row_of, &[0.5], &[-0.5], tol).is_none(),
            "inverted refined pair must be skipped"
        );
    }

    /// α′ knob semantics: gate dark (only "1" is on), and write/step helpers
    /// respect masks + [0,1] projection and never touch stable neurons.
    #[test]
    fn alpha_prime_write_and_adam_respect_masks_and_projection() {
        use ny_core::{GpuCrownLayer, GpuResnetSegment};
        // One Activation: neuron 0 unstable (chord intercept > 0), neuron 1
        // stable-active (exact slopes, zero intercepts).
        let act = GpuCrownLayer::Activation {
            lower_slope: vec![0.5, 1.0],
            upper_slope: vec![0.6, 1.0],
            lower_intercept: vec![0.0, 0.0],
            upper_intercept: vec![0.3, 0.0],
            num_neurons: 2,
        };
        let mut segs = vec![GpuResnetSegment::Chain(vec![act])];
        assert_eq!(count_activations(&segs), 1);
        let masks = unstable_masks(&segs, 1);
        assert_eq!(masks, vec![vec![true, false]]);

        // Out-of-range α′ clamps; the stable neuron NEVER changes even when
        // marked stepped (its upper_intercept is 0 — the per-domain guard).
        write_alpha_prime(&mut segs, &[vec![5.0, 0.2]], &[vec![true, true]]);
        let slopes = collect_lower_slopes(&segs, 1);
        assert_eq!(slopes, vec![vec![1.0, 1.0]]);

        // Adam ascent: positive gradient pushes the masked slope up (clamped
        // at 1), the unmasked neuron never moves; negative gradient walks down.
        let mut sl = vec![vec![0.4f32, 1.0]];
        let stepped = vec![vec![true, false]];
        let mut adam = AlphaAdam::new(&sl);
        let g = adam.step(&mut sl, &[vec![2.0, 9.0]], &stepped, 0.1, 1);
        assert!((g - 2.0).abs() < 1e-6, "max|g| over STEPPED neurons only");
        assert!(sl[0][0] > 0.4 && sl[0][0] <= 1.0);
        assert_eq!(sl[0][1], 1.0, "unmasked neuron untouched");
        let mut sl2 = vec![vec![0.05f32, 1.0]];
        let mut adam2 = AlphaAdam::new(&sl2);
        for t in 1..=5 {
            adam2.step(&mut sl2, &[vec![-3.0, 0.0]], &stepped, 0.1, t);
        }
        assert!(sl2[0][0] >= 0.0, "projection keeps α′ in [0,1]");
        assert!(sl2[0][0] < 0.05);

        // Store key check.
        let ap = AlphaPrime {
            seed_node: "gemm_pre".into(),
            pre_dim: 2,
            relu_names: vec!["relu1".into()],
            slopes: vec![vec![0.5, 1.0]],
            stepped: vec![vec![true, false]],
            improved: true,
        };
        assert!(alpha_prime_matches(
            &ap,
            "gemm_pre",
            2,
            &["relu1".to_string()]
        ));
        assert!(!alpha_prime_matches(
            &ap,
            "other",
            2,
            &["relu1".to_string()]
        ));
        assert!(!alpha_prime_matches(
            &ap,
            "gemm_pre",
            3,
            &["relu1".to_string()]
        ));
        assert!(!alpha_prime_matches(
            &ap,
            "gemm_pre",
            2,
            &["reluX".to_string()]
        ));
    }

    /// Merge semantics: element-wise max-l/min-u, finite-guarded, length
    /// mismatch refuses.
    #[test]
    fn merge_refine_result_tightens_elementwise() {
        let mut dst = ny_core::GpuCrownResult {
            lower_bounds: vec![-1.0, 0.5, f32::NAN],
            upper_bounds: vec![2.0, 3.0, 1.0],
        };
        let src = ny_core::GpuCrownResult {
            lower_bounds: vec![-0.5, 0.2, 0.0],
            upper_bounds: vec![2.5, f32::INFINITY, 0.5],
        };
        merge_refine_result(&mut dst, &src);
        assert_eq!(dst.lower_bounds, vec![-0.5, 0.5, 0.0]);
        assert_eq!(dst.upper_bounds, vec![2.0, 3.0, 0.5]);
        let before = dst.lower_bounds.clone();
        let bad = ny_core::GpuCrownResult {
            lower_bounds: vec![9.0],
            upper_bounds: vec![9.0],
        };
        merge_refine_result(&mut dst, &bad);
        assert_eq!(dst.lower_bounds, before, "length mismatch refuses");
    }

    /// CPU FD ORACLE for the α′ REFINEMENT-objective gradient (#alpha-prime
    /// deliverable 1, formula tier): on a dense truncated stack with identity
    /// seed rows (the refinement backward's shape — each row its own CROWN
    /// objective with its own ν selection and its own argmin corner), the
    /// summed per-row TRUE chain-rule gradient returned by
    /// `refine_alpha_objective_grads` must match central finite differences
    /// of `Σ_rows lower_bound(row)` through the replayed backward when a
    /// lower slope is perturbed — including through a nonzero β fold.
    #[test]
    fn refine_alpha_grad_sum_matches_replay_fd() {
        use ny_core::{GpuCrownLayer, GpuResnetSegment};
        use std::sync::Arc;

        fn rngf(state: &mut u64) -> f32 {
            *state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((*state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }
        let mut st = 0xA11F_A5EEu64;
        let n = 3usize;
        let mk_lin = |st: &mut u64| GpuCrownLayer::Linear {
            weight: Arc::from(
                (0..n * n)
                    .map(|_| rngf(st) * 0.6)
                    .collect::<Vec<f32>>()
                    .into_boxed_slice(),
            ),
            bias: Some(Arc::from(
                (0..n)
                    .map(|_| rngf(st) * 0.1)
                    .collect::<Vec<f32>>()
                    .into_boxed_slice(),
            )),
            out_features: n,
            in_features: n,
            cert_err: Default::default(),
        };
        let mk_act = |st: &mut u64| {
            let ls: Vec<f32> = (0..n).map(|_| rngf(st) * 0.4 + 0.5).collect();
            let us: Vec<f32> = (0..n).map(|_| rngf(st) * 0.3 + 0.5).collect();
            let ui: Vec<f32> = (0..n).map(|_| rngf(st).abs() * 0.3 + 0.05).collect();
            GpuCrownLayer::Activation {
                lower_slope: ls,
                upper_slope: us,
                lower_intercept: vec![0.0; n],
                upper_intercept: ui,
                num_neurons: n,
            }
        };
        // Fold order (output → input): Chain(lin3, act2, lin2) then
        // Residual(act1, lin1) — two Activations, residual structure, dense
        // mixed-sign weights.
        let segments = vec![
            GpuResnetSegment::Chain(vec![mk_lin(&mut st), mk_act(&mut st), mk_lin(&mut st)]),
            GpuResnetSegment::Residual(vec![mk_act(&mut st), mk_lin(&mut st)]),
        ];
        let in_lo: Vec<f32> = (0..n).map(|_| -0.5 + 0.1 * rngf(&mut st)).collect();
        let in_hi: Vec<f32> = in_lo.iter().map(|&l| l + 0.8).collect();
        // Nonzero β on the SECOND fold Activation (act1).
        let beta = vec![
            Vec::new(),
            (0..n).map(|_| rngf(&mut st) * 0.04).collect::<Vec<f32>>(),
        ];
        let sel: Vec<usize> = (0..n).collect();
        let rows: Vec<usize> = (0..n).collect();

        let row_lb = |segs: &[GpuResnetSegment], r: usize| -> f32 {
            let mut row = vec![0.0f32; n];
            row[r] = 1.0;
            super::super::wide_alpha_true::replay_critical_row(segs, &row, &beta)
                .expect("replay")
                .lower_bound(&in_lo, &in_hi)
        };
        let obj =
            |segs: &[GpuResnetSegment]| -> f64 { (0..n).map(|r| row_lb(segs, r) as f64).sum() };
        let lbs: Vec<f32> = (0..n).map(|r| row_lb(&segments, r)).collect();
        let (grads, rows_ok) = refine_alpha_objective_grads(
            &segments, &beta, &in_lo, &in_hi, 2, n, &sel, &rows, &lbs, None, false,
        );
        assert_eq!(rows_ok, n, "every row replay must validate against itself");

        // Per-row ν (for the near-kink skip rule, mirroring the wide oracle).
        let nus: Vec<Vec<Vec<f32>>> = (0..n)
            .map(|r| {
                let mut row = vec![0.0f32; n];
                row[r] = 1.0;
                super::super::wide_alpha_true::replay_critical_row(&segments, &row, &beta)
                    .expect("replay")
                    .nu
            })
            .collect();

        let h = 2e-3f32;
        let mut checked = 0usize;
        for fold_r in 0..2usize {
            for i in 0..n {
                let perturb = |delta: f32| -> f64 {
                    let mut segs = segments.clone();
                    visit_activations_mut(&mut segs, &mut |r, layer| {
                        if r == fold_r {
                            if let GpuCrownLayer::Activation { lower_slope, .. } = layer {
                                lower_slope[i] += delta;
                            }
                        }
                    });
                    obj(&segs)
                };
                let fd = ((perturb(h) - perturb(-h)) / (2.0 * h as f64)) as f32;
                let g = grads[fold_r][i];
                let tol = 2e-3 + 0.03 * fd.abs();
                // Near-kink rows (ν ≈ 0 at this neuron) legitimately disagree.
                let near_kink = nus.iter().any(|nu| nu[fold_r][i].abs() < 5e-2);
                if (fd - g).abs() > tol && near_kink {
                    continue;
                }
                assert!(
                    (fd - g).abs() <= tol,
                    "fold {fold_r} neuron {i}: fd {fd} != analytic {g}"
                );
                checked += 1;
            }
        }
        assert!(
            checked >= 4,
            "oracle must exercise enough neurons ({checked})"
        );
        assert!(
            grads.iter().flatten().any(|&g| g.abs() > 1e-4),
            "gradient signal must exist"
        );
    }

    /// #joint-interm-alpha FD ORACLE: the MARGIN-WEIGHTED α′ gradient
    /// (`refine_alpha_objective_grads` with `row_weights = Some(w)`) must match
    /// central finite differences of the WEIGHTED objective `Σ_r w_r · l′_r` —
    /// the exact chain `d(w·l′)/dα = w · dl′/dα`. Non-uniform weights (one zero)
    /// so a mis-applied, mis-aligned, or dropped weight is caught. Same synthetic
    /// residual net + nonzero β as the uniform oracle.
    #[test]
    fn joint_margin_alpha_grad_matches_weighted_replay_fd() {
        use ny_core::{GpuCrownLayer, GpuResnetSegment};
        use std::sync::Arc;

        fn rngf(state: &mut u64) -> f32 {
            *state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((*state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }
        let mut st = 0x5EED_1234u64;
        let n = 3usize;
        let mk_lin = |st: &mut u64| GpuCrownLayer::Linear {
            weight: Arc::from(
                (0..n * n)
                    .map(|_| rngf(st) * 0.6)
                    .collect::<Vec<f32>>()
                    .into_boxed_slice(),
            ),
            bias: Some(Arc::from(
                (0..n)
                    .map(|_| rngf(st) * 0.1)
                    .collect::<Vec<f32>>()
                    .into_boxed_slice(),
            )),
            out_features: n,
            in_features: n,
            cert_err: Default::default(),
        };
        let mk_act = |st: &mut u64| {
            let ls: Vec<f32> = (0..n).map(|_| rngf(st) * 0.4 + 0.5).collect();
            let us: Vec<f32> = (0..n).map(|_| rngf(st) * 0.3 + 0.5).collect();
            let ui: Vec<f32> = (0..n).map(|_| rngf(st).abs() * 0.3 + 0.05).collect();
            GpuCrownLayer::Activation {
                lower_slope: ls,
                upper_slope: us,
                lower_intercept: vec![0.0; n],
                upper_intercept: ui,
                num_neurons: n,
            }
        };
        let segments = vec![
            GpuResnetSegment::Chain(vec![mk_lin(&mut st), mk_act(&mut st), mk_lin(&mut st)]),
            GpuResnetSegment::Residual(vec![mk_act(&mut st), mk_lin(&mut st)]),
        ];
        let in_lo: Vec<f32> = (0..n).map(|_| -0.5 + 0.1 * rngf(&mut st)).collect();
        let in_hi: Vec<f32> = in_lo.iter().map(|&l| l + 0.8).collect();
        let beta = vec![
            Vec::new(),
            (0..n).map(|_| rngf(&mut st) * 0.04).collect::<Vec<f32>>(),
        ];
        let sel: Vec<usize> = (0..n).collect();
        let rows: Vec<usize> = (0..n).collect();
        // Non-uniform margin weights, one exactly zero (must be dropped).
        let weights: Vec<f32> = vec![2.5, 0.0, 1.25];

        let row_lb = |segs: &[GpuResnetSegment], r: usize| -> f32 {
            let mut row = vec![0.0f32; n];
            row[r] = 1.0;
            super::super::wide_alpha_true::replay_critical_row(segs, &row, &beta)
                .expect("replay")
                .lower_bound(&in_lo, &in_hi)
        };
        let obj_w = |segs: &[GpuResnetSegment]| -> f64 {
            (0..n)
                .map(|r| weights[r] as f64 * row_lb(segs, r) as f64)
                .sum()
        };
        let lbs: Vec<f32> = (0..n).map(|r| row_lb(&segments, r)).collect();
        let (grads, rows_ok) = refine_alpha_objective_grads(
            &segments,
            &beta,
            &in_lo,
            &in_hi,
            2,
            n,
            &sel,
            &rows,
            &lbs,
            Some(&weights),
            false,
        );
        assert_eq!(rows_ok, n, "every row replay must validate against itself");

        let nus: Vec<Vec<Vec<f32>>> = (0..n)
            .map(|r| {
                let mut row = vec![0.0f32; n];
                row[r] = 1.0;
                super::super::wide_alpha_true::replay_critical_row(&segments, &row, &beta)
                    .expect("replay")
                    .nu
            })
            .collect();

        let h = 2e-3f32;
        let mut checked = 0usize;
        for fold_r in 0..2usize {
            for i in 0..n {
                let perturb = |delta: f32| -> f64 {
                    let mut segs = segments.clone();
                    visit_activations_mut(&mut segs, &mut |r, layer| {
                        if r == fold_r {
                            if let GpuCrownLayer::Activation { lower_slope, .. } = layer {
                                lower_slope[i] += delta;
                            }
                        }
                    });
                    obj_w(&segs)
                };
                let fd = ((perturb(h) - perturb(-h)) / (2.0 * h as f64)) as f32;
                let g = grads[fold_r][i];
                let tol = 2e-3 + 0.03 * fd.abs();
                let near_kink = nus.iter().any(|nu| nu[fold_r][i].abs() < 5e-2);
                if (fd - g).abs() > tol && near_kink {
                    continue;
                }
                assert!(
                    (fd - g).abs() <= tol,
                    "fold {fold_r} neuron {i}: weighted fd {fd} != analytic {g}"
                );
                checked += 1;
            }
        }
        assert!(
            checked >= 4,
            "weighted oracle must exercise enough neurons ({checked})"
        );
        assert!(
            grads.iter().flatten().any(|&g| g.abs() > 1e-4),
            "weighted gradient signal must exist"
        );
        // A row's weight must actually scale its contribution: doubling weights[0]
        // doubles the row-0 gradient mass reaching fold 1 (the seed-adjacent
        // Activation whose ν for row 0 is nonzero).
        let mut w2 = weights.clone();
        w2[0] *= 2.0;
        let (grads2, _) = refine_alpha_objective_grads(
            &segments,
            &beta,
            &in_lo,
            &in_hi,
            2,
            n,
            &sel,
            &rows,
            &lbs,
            Some(&w2),
            false,
        );
        // Zero-weight row 1 contributes nothing regardless.
        let (grads_uni, _) = refine_alpha_objective_grads(
            &segments, &beta, &in_lo, &in_hi, 2, n, &sel, &rows, &lbs, None, false,
        );
        assert!(
            grads2
                .iter()
                .flatten()
                .zip(grads.iter().flatten())
                .any(|(&a, &b)| (a - b).abs() > 1e-4),
            "reweighting must change the gradient"
        );
        assert!(
            grads_uni.iter().flatten().any(|&g| g.abs() > 1e-4),
            "uniform baseline gradient signal must exist"
        );
    }

    /// #joint-interm-alpha: `compute_margin_weights` builds
    /// `m_j = Σ_s max(−(spec·W_tail)[j], 0)` from the tail linear map directly
    /// consuming the last ReLU, and falls back to `None` (uniform) with no
    /// negative tail mass.
    #[test]
    fn compute_margin_weights_upper_branch_tail_mass() {
        // gemm_pre → relu_last → out; tail W = [[0.7,-0.3],[0.2,0.6]] (from `lin`).
        let mut g = GraphNetwork::new();
        g.add_node(lin("gemm_pre", NETWORK_INPUT));
        g.add_node(relu("relu_last", "gemm_pre"));
        g.add_node(lin("out", "relu_last"));
        g.set_output("out");
        // spec rows [-1,0], [1,0]: a0 = -row0 = [-0.7, 0.3] (0.7 mass on j0),
        // a1 = row0 = [0.7, -0.3] (0.3 mass on j1) ⇒ m = [0.7, 0.3].
        let spec = arr2(&[[-1.0_f32, 0.0], [1.0, 0.0]]);
        let m = compute_margin_weights(&g, "out", &spec).expect("weights resolve");
        assert_eq!(m.len(), 2);
        assert!((m[0] - 0.7).abs() < 1e-5, "m0={}", m[0]);
        assert!((m[1] - 0.3).abs() < 1e-5, "m1={}", m[1]);
        // a = [1,1]·W = [0.9, 0.3] — no negative tail mass ⇒ uniform fallback.
        let spec_pos = arr2(&[[1.0_f32, 1.0]]);
        assert!(compute_margin_weights(&g, "out", &spec_pos).is_none());
    }

    #[test]
    fn margin_weight_resource_and_deadline_refuse_before_dot() {
        let mut g = GraphNetwork::new();
        g.add_node(lin("gemm_pre", NETWORK_INPUT));
        g.add_node(relu("relu_last", "gemm_pre"));
        g.add_node(lin("out", "relu_last"));
        g.set_output("out");
        let spec = arr2(&[[-1.0_f32, 0.0], [1.0, 0.0]]);
        let dot_calls = std::cell::Cell::new(0usize);
        let mut counted_dot = |lhs: &Array2<f32>, rhs: &Array2<f32>| {
            dot_calls.set(dot_calls.get() + 1);
            lhs.dot(rhs)
        };

        let mut within_deadline = || false;
        let oversized = compute_margin_weights_with_deadline_check_and_dot(
            &g,
            "out",
            &spec,
            MarginWeightLimits {
                max_host_bytes: 0,
                max_work: SELECTIVE_CLIP_MAX_WORK,
            },
            &mut within_deadline,
            &mut counted_dot,
        )
        .unwrap_err();
        assert!(matches!(
            oversized,
            NyError::CpuMemoryExceeded {
                site: MARGIN_WEIGHT_SITE,
                ..
            }
        ));
        assert_eq!(dot_calls.get(), 0, "oversized work must refuse before dot");

        let mut expired = || true;
        let deadline = compute_margin_weights_with_deadline_check_and_dot(
            &g,
            "out",
            &spec,
            MARGIN_WEIGHT_LIMITS,
            &mut expired,
            &mut counted_dot,
        )
        .unwrap_err();
        assert!(matches!(deadline, NyError::DeadlineExceeded(_)));
        assert_eq!(dot_calls.get(), 0, "expired work must refuse before dot");
    }

    #[test]
    fn selective_last_rows_match_winner_score_and_retain_premises() {
        let bounds = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[4]), vec![-1.0, 0.2, -2.0, -4.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, 0.4, 2.0, 1.0]).unwrap(),
        )
        .unwrap();
        let cache = HashMap::from([("seed".to_string(), Arc::new(bounds))]);
        let candidates = vec![0, 1, 2, 3];
        let premise = vec![false, true, false, false];
        // Non-premise scores:
        //   j0 = 0.5 * 1 = 0.5
        //   j2 = 1.0 * 3 = 3.0
        //   j3 = 0.8 * 1 = 0.8
        // Keep top two (j2,j3) plus the stable premise source j1.
        let selected = select_intermediate_objective_rows(
            &[cache],
            "seed",
            &candidates,
            &premise,
            2,
            Some(&[1.0, 100.0, 3.0, 1.0]),
        );
        assert_eq!(selected, vec![1, 2, 3]);
        assert!(!selected.contains(&0), "top-K omission must be explicit");
    }

    #[test]
    fn selective_topk_zero_is_exact_selection_noop() {
        let candidates = vec![7usize, 2, 9, 1];
        let premise = vec![false; 10];
        let selected = select_intermediate_objective_rows(
            &[],
            "missing-seed",
            &candidates,
            &premise,
            0,
            Some(&[f32::NAN; 10]),
        );
        assert_eq!(
            selected, candidates,
            "gate-off must preserve every row and its order"
        );

        let refused = select_intermediate_objective_rows(
            &[],
            "missing-seed",
            &[usize::MAX],
            &[false],
            1,
            None,
        );
        assert!(
            refused.is_empty(),
            "overflowing score-table shape must refuse before allocation"
        );
    }

    #[test]
    fn complete_clip_top20_is_per_domain_and_stable_premises_are_additive() {
        let graph = selector_graph();
        let make = |lower: Vec<f32>, upper: Vec<f32>| {
            Arc::new(
                BoundedTensor::new(
                    ArrayD::from_shape_vec(IxDyn(&[25]), lower).unwrap(),
                    ArrayD::from_shape_vec(IxDyn(&[25]), upper).unwrap(),
                )
                .unwrap(),
            )
        };
        let mut lower0 = vec![-1.0f32; 25];
        let mut upper0 = vec![1.0f32; 25];
        for neuron in 0..20 {
            lower0[neuron] = -10.0;
            upper0[neuron] = 10.0;
        }
        // A premise-clamped row is stable in this child's inherited cache.
        lower0[24] = 0.0;
        upper0[24] = 1.0;
        let mut lower1 = vec![-1.0f32; 25];
        let mut upper1 = vec![1.0f32; 25];
        for neuron in 5..25 {
            lower1[neuron] = -10.0;
            upper1[neuron] = 10.0;
        }
        let caches = vec![
            HashMap::from([("seed".to_string(), make(lower0, upper0))]),
            HashMap::from([("seed".to_string(), make(lower1, upper1))]),
        ];
        let mut history0 = GraphSplitHistory::new();
        history0.add_constraint(GraphNeuronConstraint::new("relu".into(), 24, true, 0.0).unwrap());
        let history1 = GraphSplitHistory::new();
        let candidates: Vec<usize> = (0..25).collect();
        let (union, per_domain, premise) = select_complete_clip_rows(
            &graph,
            &caches,
            &[&history0, &history1],
            "seed",
            "relu",
            &candidates,
            25,
            20,
            None,
            None,
            None,
            true,
        )
        .expect("per-domain selection");
        assert_eq!(per_domain[0], (0..20).collect::<Vec<_>>());
        assert_eq!(per_domain[1], (5..25).collect::<Vec<_>>());
        assert_eq!(union, (0..25).collect::<Vec<_>>());
        assert!(premise[24], "stable split source must be force-captured");
        assert!(
            !per_domain[0].contains(&24),
            "source rows do not consume or silently expand the top-20 objective budget"
        );
    }

    #[test]
    fn complete_clip_cached_la_clamps_after_mean_per_domain_and_layer() {
        let bounds = |lower: Vec<f32>, upper: Vec<f32>| {
            BoundedTensor::new(
                ArrayD::from_shape_vec(IxDyn(&[lower.len()]), lower).unwrap(),
                ArrayD::from_shape_vec(IxDyn(&[upper.len()]), upper).unwrap(),
            )
            .unwrap()
        };
        let root_a = bounds(vec![-1.0; 3], vec![1.0; 3]);
        let child_a = bounds(vec![-1.0; 3], vec![1.0; 3]);
        let history = GraphSplitHistory::new();
        let la_a = [
            arr2(&[[-3.0f32, -2.0, -0.2]]),
            arr2(&[[1.0, 4.0, -0.2]]),
            arr2(&[[3.0, -4.0, -0.2]]),
        ];
        assert_eq!(
            select_complete_clip_rows_from_cached_las(
                &root_a,
                &child_a,
                &history,
                "relu",
                &[&la_a[0], &la_a[1]],
                1,
                None,
            ),
            Some(vec![0]),
            "rows [-2,+4] must average to +1 then clamp to zero; summing \
             per-row negative parts would incorrectly favor neuron 1"
        );
        assert_eq!(
            select_complete_clip_rows_from_cached_las(
                &root_a,
                &child_a,
                &history,
                "relu",
                &[&la_a[1], &la_a[2]],
                1,
                None,
            ),
            Some(vec![2]),
            "the same captured layer must score independently per domain's active rows"
        );
        let verified = [true, false, true];
        let full_spec_rows: Vec<&Array2<f32>> = la_a.iter().collect();
        let incorrectly_pruned_rows: Vec<&Array2<f32>> = la_a
            .iter()
            .zip(verified)
            .filter_map(|(row, is_verified)| (!is_verified).then_some(row))
            .collect();
        assert_eq!(
            select_complete_clip_rows_from_cached_las(
                &root_a,
                &child_a,
                &history,
                "relu",
                &full_spec_rows,
                1,
                None,
            ),
            Some(vec![1]),
            "DomainClipScorer averages the full spec dimension, including \
             individually verified objective caches"
        );
        assert_eq!(
            select_complete_clip_rows_from_cached_las(
                &root_a,
                &child_a,
                &history,
                "relu",
                &incorrectly_pruned_rows,
                1,
                None,
            ),
            Some(vec![2]),
            "the mixed verified/unverified fixture must detect per-spec pruning"
        );

        let root_b = bounds(vec![-2.0; 2], vec![2.0; 2]);
        let child_b = bounds(vec![-2.0; 2], vec![2.0; 2]);
        let la_b = [
            arr2(&[[-4.0f32, -1.0]]),
            arr2(&[[-4.0, 3.0]]),
            arr2(&[[4.0, -5.0]]),
        ];
        assert_eq!(
            select_complete_clip_rows_from_cached_las(
                &root_b,
                &child_b,
                &history,
                "relu",
                &[&la_b[0], &la_b[1]],
                1,
                None,
            ),
            Some(vec![0])
        );
        assert_eq!(
            select_complete_clip_rows_from_cached_las(
                &root_b,
                &child_b,
                &history,
                "relu",
                &[&la_b[1], &la_b[2]],
                1,
                None,
            ),
            Some(vec![1]),
            "a second layer gets its own lA mean; no final-layer weight is reused"
        );

        let mut active_leaf = GraphSplitHistory::new();
        active_leaf
            .add_constraint(GraphNeuronConstraint::new("relu".into(), 0, true, 0.0).unwrap());
        let clamp_la = arr2(&[[-10.0f32, -1.0]]);
        assert_eq!(
            select_complete_clip_rows_from_cached_las(
                &root_b,
                &child_b,
                &active_leaf,
                "relu",
                &[&clamp_la],
                1,
                None,
            ),
            Some(vec![1]),
            "prospective active split clamps neuron 0 to l>=0, making its \
             triangle intercept zero without a child CROWN pass"
        );
    }

    #[test]
    fn complete_clip_gpu_mean_reduction_matches_full_cached_la_scorer() {
        let bounds = |lower: Vec<f32>, upper: Vec<f32>| {
            BoundedTensor::new(
                ArrayD::from_shape_vec(IxDyn(&[lower.len()]), lower).unwrap(),
                ArrayD::from_shape_vec(IxDyn(&[upper.len()]), upper).unwrap(),
            )
            .unwrap()
        };
        let root = bounds(vec![-2.0; 3], vec![2.0; 3]);
        let child = bounds(vec![-2.0; 3], vec![2.0; 3]);
        // Row-major GPU gather: three full specification rows × three
        // root-unstable columns. Signs disagree deliberately, so only
        // clamp-after-mean produces the expected ranking.
        let rows = [
            arr2(&[[-3.0f32, -2.0, -0.2]]),
            arr2(&[[1.0, 4.0, -0.2]]),
            arr2(&[[3.0, -4.0, -0.2]]),
        ];
        let gathered: Vec<f32> = rows.iter().flat_map(|row| row.iter().copied()).collect();
        let reduced = reduce_complete_clip_gpu_gather(&gathered, 1, 3, &[0, 1, 2], 3, None)
            .expect("well-formed one-domain gather");
        assert_eq!(reduced.len(), 1);

        let cached_rows: Vec<&Array2<f32>> = rows.iter().collect();
        let mut history = GraphSplitHistory::new();
        history.add_constraint(GraphNeuronConstraint::new("relu".into(), 0, true, 0.0).unwrap());
        let cached_pick = select_complete_clip_rows_from_cached_las(
            &root,
            &child,
            &history,
            "relu",
            &cached_rows,
            2,
            None,
        );
        let gpu_mean_pick = select_complete_clip_rows_from_mean_la(
            &root,
            &child,
            &history,
            "relu",
            &reduced[0],
            2,
            None,
        );
        assert_eq!(
            gpu_mean_pick, cached_pick,
            "compact GPU reduction must preserve the full-spec cached-lA \
             ranking and prospective split clamps exactly"
        );
    }

    #[test]
    fn complete_clip_gpu_gather_reduction_keeps_domain_blocks_separate() {
        // Two domains, two specification rows, two gathered columns:
        // d0 mean = [-2, +2], d1 mean = [+3, -3].
        let gathered = vec![
            -4.0f32, 1.0, // d0 s0
            0.0, 3.0, // d0 s1
            2.0, -2.0, // d1 s0
            4.0, -4.0, // d1 s1
        ];
        let reduced = reduce_complete_clip_gpu_gather(&gathered, 2, 2, &[1, 3], 4, None)
            .expect("domain-major wide gather");
        assert_eq!(reduced[0], arr1(&[0.0, -2.0, 0.0, 2.0]));
        assert_eq!(reduced[1], arr1(&[0.0, 3.0, 0.0, -3.0]));

        assert!(
            reduce_complete_clip_gpu_gather(&gathered[..7], 2, 2, &[1, 3], 4, None).is_none(),
            "truncated domain/spec layout must refuse"
        );
        assert!(
            reduce_complete_clip_gpu_gather(&gathered, 2, 2, &[1, 4], 4, None).is_none(),
            "out-of-range gathered columns must refuse"
        );
        let mut non_finite = gathered;
        non_finite[3] = f32::NAN;
        assert!(
            reduce_complete_clip_gpu_gather(&non_finite, 2, 2, &[1, 3], 4, None).is_none(),
            "non-finite gathered coefficients must refuse"
        );
    }

    #[test]
    fn complete_clip_gpu_spec_seed_requires_standard_row_major_layout() {
        let row_major = Array2::from_shape_vec((2, 3), (0..6).map(|v| v as f32).collect())
            .expect("row-major fixture");
        let fortran_order = Array2::from_shape_vec((2, 3).f(), (0..6).map(|v| v as f32).collect())
            .expect("column-major fixture");
        assert!(complete_clip_spec_matrix_is_row_major(&row_major));
        assert!(!fortran_order.is_standard_layout());
        assert!(
            !complete_clip_spec_matrix_is_row_major(&fortran_order),
            "a memory-order slice of an F-order C matrix must never be \
             mislabeled as row-major specification rows"
        );
    }

    #[test]
    fn complete_clip_gpu_missing_parent_reconstruction_cap_is_exact() {
        assert!(!complete_clip_gpu_mean_parent_count_admitted(0));
        assert!(complete_clip_gpu_mean_parent_count_admitted(1));
        assert!(complete_clip_gpu_mean_parent_count_admitted(
            COMPLETE_CLIP_GPU_MEAN_LA_MAX_RECONSTRUCT_DOMAINS
        ));
        assert!(!complete_clip_gpu_mean_parent_count_admitted(
            COMPLETE_CLIP_GPU_MEAN_LA_MAX_RECONSTRUCT_DOMAINS + 1
        ));
    }

    #[test]
    fn complete_clip_missing_la_is_layer_local_and_keeps_full_spec_mean() {
        let mut graph = GraphNetwork::new();
        graph.add_node(lin("seed_a", NETWORK_INPUT));
        graph.add_node(relu("relu_a", "seed_a"));
        graph.add_node(lin("seed_b", "relu_a"));
        graph.add_node(relu("relu_b", "seed_b"));
        graph.set_output("relu_b");

        let bounds = || {
            Arc::new(
                BoundedTensor::new(
                    ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0, -1.0]).unwrap(),
                    ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.0]).unwrap(),
                )
                .unwrap(),
            )
        };
        let root_bounds = HashMap::from([
            ("seed_a".to_string(), bounds()),
            ("seed_b".to_string(), bounds()),
        ]);
        let child_bounds = HashMap::from([
            ("seed_a".to_string(), bounds()),
            ("seed_b".to_string(), bounds()),
        ]);

        let mut objective0 = CachedLinearBounds::default();
        objective0
            .lower_a
            .insert("relu_a".into(), arr2(&[[-2.0f32, -1.0]]));
        objective0
            .lower_a
            .insert("relu_b".into(), arr2(&[[-4.0f32, -1.0]]));
        let mut objective1 = CachedLinearBounds::default();
        objective1
            .lower_a
            .insert("relu_b".into(), arr2(&[[4.0f32, -5.0]]));

        let decisions = complete_clip_decisions_from_cached_las(
            &graph,
            &root_bounds,
            &child_bounds,
            &GraphSplitHistory::new(),
            &[&objective0, &objective1],
            1,
            None,
        )
        .expect("the fully-covered layer remains available");
        assert_eq!(decisions.len(), 1);
        assert!(!decisions.contains_key("relu_a"));
        assert_eq!(
            decisions.get("relu_b").map(Vec::as_slice),
            Some([1usize].as_slice()),
            "the emitted layer must average both objectives; objective0 alone \
             would select neuron 0"
        );
    }

    #[test]
    fn committed_child_precomputed_decisions_are_consumed_and_reused_once() {
        let graph = selector_graph();
        let verifier = BetaCrownVerifier::default();
        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0, -1.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.0]).unwrap(),
        )
        .unwrap();
        let root_seed = Arc::new(
            BoundedTensor::new(
                ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0, -1.0]).unwrap(),
                ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.0]).unwrap(),
            )
            .unwrap(),
        );
        let root_bounds = HashMap::from([("seed".to_string(), Arc::clone(&root_seed))]);
        assert!(verifier.complete_clip_root_bounds_cache.store_finalized(
            &graph,
            &input,
            &root_bounds
        ));
        let mut history = GraphSplitHistory::new();
        history
            .add_constraint(GraphNeuronConstraint::new("prior_relu".into(), 0, true, 0.0).unwrap());
        assert!(verifier.publish_complete_clip_decisions(
            &graph,
            &input,
            &history,
            HashMap::from([("relu".to_string(), vec![1])]),
        ));
        let snapshot = verifier
            .complete_clip_root_bounds_cache
            .take_decisions(&graph, &input, &history)
            .expect("committed child decision");
        assert!(
            verifier
                .complete_clip_root_bounds_cache
                .take_decisions(&graph, &input, &history)
                .is_none(),
            "decision persistence is one generation"
        );

        // Local scoring would select unstable neuron 0 and exclude child-stable
        // neuron 1. The committed-child snapshot must override that recompute.
        let child_seed = Arc::new(
            BoundedTensor::new(
                ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0, 0.0]).unwrap(),
                ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.0]).unwrap(),
            )
            .unwrap(),
        );
        let caches = [HashMap::from([("seed".to_string(), child_seed)])];
        let snapshots = [Some(snapshot)];
        let (selected, objectives, _) = select_complete_clip_rows(
            &graph,
            &caches,
            &[&history],
            "seed",
            "relu",
            &[0, 1],
            2,
            1,
            None,
            Some(&snapshots),
            None,
            true,
        )
        .expect("precomputed selector");
        assert_eq!(selected, vec![1]);
        assert_eq!(objectives, vec![vec![1]]);
    }

    #[test]
    fn complete_clip_latest_schedule_captures_only_newest_premise_source() {
        let graph = selector_graph();
        let bounds = Arc::new(
            BoundedTensor::new(
                ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0, -1.0]).unwrap(),
                ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.0]).unwrap(),
            )
            .unwrap(),
        );
        let caches = [HashMap::from([("seed".to_string(), bounds)])];
        let mut history = GraphSplitHistory::new();
        history.add_constraint(GraphNeuronConstraint::new("relu".into(), 0, true, 0.0).unwrap());
        history.add_constraint(GraphNeuronConstraint::new("relu".into(), 1, false, 0.0).unwrap());

        let (all_rows, _, all_premises) = select_complete_clip_rows(
            &graph,
            &caches,
            &[&history],
            "seed",
            "relu",
            &[],
            2,
            20,
            None,
            None,
            None,
            true,
        )
        .expect("all-history source selection");
        assert_eq!(all_rows, vec![0, 1]);
        assert_eq!(all_premises, vec![true, true]);

        let (latest_rows, _, latest_premises) = select_complete_clip_rows(
            &graph,
            &caches,
            &[&history],
            "seed",
            "relu",
            &[],
            2,
            20,
            None,
            None,
            None,
            false,
        )
        .expect("latest-only source selection");
        assert_eq!(latest_rows, vec![1]);
        assert_eq!(latest_premises, vec![false, true]);
    }

    #[test]
    fn complete_clip_late_round_uses_first_nonempty_network_layer_not_global_newest() {
        let mut graph = GraphNetwork::new();
        graph.add_node(lin("early_pre", NETWORK_INPUT));
        graph.add_node(relu("early_relu", "early_pre"));
        graph.add_node(lin("late_pre", "early_relu"));
        graph.add_node(relu("late_relu", "late_pre"));
        graph.set_output("late_relu");

        let mut history = GraphSplitHistory::new();
        history
            .add_constraint(GraphNeuronConstraint::new("late_relu".into(), 0, true, 0.0).unwrap());
        history
            .add_constraint(GraphNeuronConstraint::new("early_relu".into(), 0, true, 0.0).unwrap());
        history.add_constraint(
            GraphNeuronConstraint::new("early_relu".into(), 1, false, 0.0).unwrap(),
        );
        history
            .add_constraint(GraphNeuronConstraint::new("late_relu".into(), 1, false, 0.0).unwrap());

        let scheduled = scheduled_relu_constraints(&graph, &history, false).expect("network order");
        assert_eq!(scheduled.len(), 1);
        assert_eq!(scheduled[0].node_name(), "early_relu");
        assert_eq!(
            scheduled[0].neuron_idx(),
            1,
            "within the first nonempty layer, use that layer's newest split"
        );
    }

    #[test]
    fn complete_clip_objective_row_map_refuses_any_silent_shrink() {
        assert_eq!(
            exact_objective_rows(&[1, 3], &[usize::MAX, 0, usize::MAX, 1]),
            Some(vec![0, 1])
        );
        assert_eq!(
            exact_objective_rows(&[1, 2], &[usize::MAX, 0, usize::MAX]),
            None,
            "an unmapped objective must disable the proposal"
        );
        assert_eq!(
            exact_objective_rows(&[1, 1], &[usize::MAX, 0]),
            None,
            "duplicate objective identities must not silently collapse"
        );
        assert_eq!(
            exact_objective_rows(&[3], &[usize::MAX, 0]),
            None,
            "an out-of-range objective must disable the proposal"
        );
    }

    #[test]
    fn selective_row_budget_accounts_live_vectors_at_exact_boundary() {
        let bytes_per_candidate =
            size_of::<usize>() + size_of::<bool>() + size_of::<(usize, f64)>() + size_of::<usize>();
        let last_admitted = SELECTIVE_CLIP_MAX_HOST_BYTES / bytes_per_candidate;
        assert!(selective_row_work_bytes(last_admitted, last_admitted)
            .is_some_and(|n| n <= SELECTIVE_CLIP_MAX_HOST_BYTES));
        assert!(
            selective_row_work_bytes(last_admitted + 1, last_admitted + 1)
                .is_none_or(|n| n > SELECTIVE_CLIP_MAX_HOST_BYTES)
        );
        // The independent work cap can bind before the byte cap. Pin its exact
        // monotone boundary too without materializing any large vectors.
        let (mut lo, mut hi) = (0usize, last_admitted);
        while lo < hi {
            let mid = lo + (hi - lo).div_ceil(2);
            if validate_selective_row_budget(mid, mid, 1, true).is_some() {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        assert!(validate_selective_row_budget(lo, lo, 1, true).is_some());
        assert!(validate_selective_row_budget(lo + 1, lo + 1, 1, true).is_none());
        assert!(selective_row_work_bytes(usize::MAX, usize::MAX).is_none());
    }

    #[ntest::timeout(10000)]
    #[test]
    fn large_selective_candidate_set_is_bounded_deterministic_and_deadline_aware() {
        let n = 65_536usize;
        let bounds = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[n]), -1.0f32),
            ArrayD::from_elem(IxDyn(&[n]), 1.0f32),
        )
        .unwrap();
        let cache = HashMap::from([("seed".to_string(), Arc::new(bounds))]);
        let candidates: Vec<usize> = (0..n).collect();
        let premise = vec![false; n];
        let selected = select_intermediate_objective_rows(
            std::slice::from_ref(&cache),
            "seed",
            &candidates,
            &premise,
            20,
            None,
        );
        assert_eq!(selected, (0..20).collect::<Vec<_>>());

        let mut polls = 0usize;
        let mut expire_during_score = || {
            polls += 1;
            polls >= 70
        };
        let refused = select_intermediate_objective_rows_with_deadline_check(
            &[cache],
            "seed",
            &candidates,
            &premise,
            20,
            None,
            &mut expire_during_score,
        );
        assert!(
            refused.is_none(),
            "expired scoring must return no partial selection"
        );
        assert_eq!(polls, 70);
    }

    /// input → l1 → relu1 → l2 → add(l2, l1) → gemm_pre → relu_last → out:
    /// the next ReLU strictly below `gemm_pre` is `relu1` (seed `l1`).
    #[test]
    fn find_next_relu_seed_below_walks_ancestors() {
        let mut g = GraphNetwork::new();
        g.add_node(lin("l1", NETWORK_INPUT));
        g.add_node(relu("relu1", "l1"));
        g.add_node(lin("l2", "relu1"));
        g.add_node(GraphNode::new(
            "add",
            Layer::Add(AddLayer),
            vec!["l2".to_string(), "l1".to_string()],
        ));
        g.add_node(lin("gemm_pre", "add"));
        g.add_node(relu("relu_last", "gemm_pre"));
        g.add_node(lin("out", "relu_last"));
        g.set_output("out");

        let (relu_name, seed) =
            find_next_relu_seed_below(&g, "gemm_pre").expect("ancestor ReLU must resolve");
        assert_eq!(relu_name, "relu1");
        assert_eq!(seed, "l1");

        // A ReLU fed by the network input refuses (nothing below the seed).
        let mut g2 = GraphNetwork::new();
        g2.add_node(relu("r0", NETWORK_INPUT));
        g2.add_node(lin("out", "r0"));
        g2.set_output("out");
        assert!(find_next_relu_seed_below(&g2, "out").is_none());

        // No ancestor ReLU at all refuses.
        let mut g3 = GraphNetwork::new();
        g3.add_node(lin("l1", NETWORK_INPUT));
        g3.add_node(lin("out", "l1"));
        g3.set_output("out");
        assert!(find_next_relu_seed_below(&g3, "out").is_none());
    }
}

/// Deterministic, CPU-backed implementation of the sound-GPU trait used by
/// `ny-propagate`'s routing tests.
///
/// The intermediate-refinement algorithms consume a capability, not GPU
/// identity. Their former WGPU-only tests were hardware-dependent, which left
/// the algorithms untested in the portable default suite. This adapter
/// exercises the exact capability boundary while evaluating the extracted
/// ResNet fold in f64 and publishing it outward to f32. The optional
/// `gpu-tests` contract below separately exercises the public typed WGPU proof
/// constructor and fails fast when no device earns all five authority rungs.
#[cfg(test)]
#[derive(Default)]
pub(super) struct HermeticSoundGpuCrownEngine {
    bound_calls: std::sync::atomic::AtomicUsize,
    gradient_calls: std::sync::atomic::AtomicUsize,
    /// Cooperative-cancellation deadline, scoped by the caller exactly as a real
    /// backend's would be. See [`Self::deadline_expired`] for why this fixture
    /// carries one at all.
    crown_backward_deadline: Mutex<Option<Instant>>,
}

#[cfg(test)]
impl HermeticSoundGpuCrownEngine {
    /// Whether the scoped deadline has passed, polled between the two folds of
    /// [`Self::bounds`] — the only bounded work units this fixture has.
    ///
    /// WHY A TEST DOUBLE IMPLEMENTS COOPERATIVE CANCELLATION. It is not for the
    /// cancellation itself; these fixtures finish in microseconds. It is because
    /// `honors_crown_backward_deadline` is a CAPABILITY CLAIM that deadline-scored
    /// lanes filter on, and `BetaCrownVerifier::new` always installs a deadline
    /// (`alpha_config.deadline = now + timeout`). A fixture that left the default
    /// `false` was therefore refused at the wide entry for every test that used
    /// it — silently, as `WideLaneDecline::EntryNoSoundBackend` — which is exactly
    /// how the two conv-chain oracles went vacuous. Returning `true` while never
    /// polling would trade a vacuous test for a lying one, so the claim is made
    /// true here instead: the deadline is stored, polled, and honoured with
    /// `NyError::DeadlineExceeded`, which every CROWN caller already treats as a
    /// sound fallback.
    fn deadline_expired(&self) -> bool {
        self.crown_backward_deadline
            .lock()
            .expect("hermetic crown deadline mutex")
            .is_some_and(|deadline| Instant::now() >= deadline)
    }

    fn bounds(
        &self,
        segments: &[ny_core::GpuResnetSegment],
        seed: &ny_core::GpuCrownSeed,
        input_lower: &[f32],
        input_upper: &[f32],
    ) -> ny_core::Result<ny_core::GpuCrownResult> {
        use std::sync::atomic::Ordering;

        if self.deadline_expired() {
            return Err(NyError::DeadlineExceeded(
                "hermetic sound-CROWN fold observed an expired scoped deadline".into(),
            ));
        }
        self.bound_calls.fetch_add(1, Ordering::Relaxed);
        let lower = ny_core::joint_alpha_grad::joint_lower_bound_debug_f64(
            segments,
            seed.lower_a.as_ref(),
            seed.lower_b.as_ref(),
            seed.num_specs,
            seed.current_dim,
            input_lower,
            input_upper,
        )
        .ok_or_else(|| {
            NyError::InvalidSpec("hermetic sound-CROWN lower fold rejected its shape".into())
        })?;

        // max(Ax+b) = -min((-A)x+(-b)).  Negating the upper seed also makes
        // the ordinary lower-fold sign choice select the upper ReLU plane in
        // exactly the cases required by upper CROWN.
        let neg_upper_a: Vec<f32> = seed.upper_a.iter().map(|&value| -value).collect();
        let neg_upper_b: Vec<f32> = seed.upper_b.iter().map(|&value| -value).collect();
        if self.deadline_expired() {
            return Err(NyError::DeadlineExceeded(
                "hermetic sound-CROWN fold observed an expired scoped deadline between folds"
                    .into(),
            ));
        }
        let neg_upper = ny_core::joint_alpha_grad::joint_lower_bound_debug_f64(
            segments,
            &neg_upper_a,
            &neg_upper_b,
            seed.num_specs,
            seed.current_dim,
            input_lower,
            input_upper,
        )
        .ok_or_else(|| {
            NyError::InvalidSpec("hermetic sound-CROWN upper fold rejected its shape".into())
        })?;
        if self.deadline_expired() {
            return Err(NyError::DeadlineExceeded(
                "hermetic sound-CROWN fold expired before result publication".into(),
            ));
        }

        let lower_bounds = lower
            .into_iter()
            .map(ny_core::f64_to_f32_down)
            .collect::<Vec<_>>();
        let upper_bounds = neg_upper
            .into_iter()
            .map(|value| ny_core::f64_to_f32_up(-value))
            .collect::<Vec<_>>();
        if lower_bounds
            .iter()
            .zip(&upper_bounds)
            .any(|(&lower, &upper)| !lower.is_finite() || !upper.is_finite() || lower > upper)
        {
            return Err(NyError::InternalError(
                "hermetic sound-CROWN fold produced an invalid enclosure".into(),
            ));
        }
        Ok(ny_core::GpuCrownResult {
            lower_bounds,
            upper_bounds,
        })
    }

    pub(super) fn assert_exercised(&self) {
        use std::sync::atomic::Ordering;

        assert!(
            self.bound_calls.load(Ordering::Relaxed) > 0,
            "test never crossed the injected sound-GPU capability seam"
        );
    }
}

#[cfg(test)]
impl Drop for HermeticSoundGpuCrownEngine {
    fn drop(&mut self) {
        // A capability-adapter test that never invokes the capability is just
        // another form of a skipped test.  Enforce non-vacuity at the fixture
        // boundary while avoiding a double panic when a substantive assertion
        // has already failed.
        if !std::thread::panicking() {
            self.assert_exercised();
        }
    }
}

#[cfg(test)]
impl GemmEngine for HermeticSoundGpuCrownEngine {
    fn gemm_f32(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
    ) -> ny_core::Result<Vec<f32>> {
        ny_core::NaiveCpuGemmEngine.gemm_f32(m, k, n, a, b)
    }

    fn as_gpu_crown_backward(&self) -> Option<&dyn ny_core::GpuCrownBackward> {
        Some(self)
    }
}

#[cfg(test)]
impl ny_core::GpuCrownBackward for HermeticSoundGpuCrownEngine {
    fn crown_backward_gpu(
        &self,
        _layers: &[ny_core::GpuCrownLayer],
        _spec: &[f32],
        _num_specs: usize,
        _input_lower: &[f32],
        _input_upper: &[f32],
    ) -> ny_core::Result<ny_core::GpuCrownResult> {
        Err(NyError::UnsupportedOp(
            "hermetic adapter exposes only the extracted ResNet contract".into(),
        ))
    }

    fn provides_sound_gpu_crown(&self) -> bool {
        true
    }

    fn set_crown_backward_deadline(&self, deadline: Option<Instant>) {
        *self
            .crown_backward_deadline
            .lock()
            .expect("hermetic crown deadline mutex") = deadline;
    }

    /// True, and made true rather than merely asserted — see
    /// [`HermeticSoundGpuCrownEngine::deadline_expired`].
    fn honors_crown_backward_deadline(&self) -> bool {
        true
    }

    fn crown_backward_gpu_resnet_sound_beta(
        &self,
        segments: &[ny_core::GpuResnetSegment],
        seed: &ny_core::GpuCrownSeed,
        input_lower: &[f32],
        input_upper: &[f32],
        _beta_signed: &[Vec<f32>],
        _frontier_abs: &[Vec<f32>],
        _node_abs: &[Vec<f32>],
    ) -> ny_core::Result<ny_core::GpuCrownResult> {
        // Omitting beta is conservative: this is the ordinary alpha-CROWN
        // enclosure.  Tests of beta transport use dedicated recording seams;
        // these refinement tests need a sound authority source.
        self.bounds(segments, seed, input_lower, input_upper)
    }

    /// The SINGLE-DOMAIN β-ascent entry — same "sound bounds, stationary
    /// gradients" contract as
    /// [`Self::crown_backward_gpu_resnet_sound_beta_batched_grad`], which see for
    /// why the captures are zeros rather than fabricated coefficients.
    ///
    /// Implemented rather than inherited. The trait default is
    /// `UnsupportedOp`, and `gpu_beta_optimize_domain` treats an `Err` on
    /// iteration 0 as "this lane is unavailable" and returns `None` — so a
    /// fixture that left the default sent every β-opt-eligible domain down the
    /// dense-only path while its test went on asserting a property of the β lane.
    /// That is the same vacuity the batched entry above was added to close; the
    /// per-domain lane reaches this entry instead, and needs it closed too.
    fn crown_backward_gpu_resnet_sound_beta_grad(
        &self,
        segments: &[ny_core::GpuResnetSegment],
        seed: &ny_core::GpuCrownSeed,
        input_lower: &[f32],
        input_upper: &[f32],
        _beta_signed: &[Vec<f32>],
        beta_gather_idx: &[Vec<u32>],
        _frontier_abs: &[Vec<f32>],
        _node_abs: &[Vec<f32>],
    ) -> ny_core::Result<ny_core::GpuCrownBetaGradResult> {
        let bounds = self.bounds(segments, seed, input_lower, input_upper)?;
        Ok(ny_core::GpuCrownBetaGradResult {
            lower_bounds: bounds.lower_bounds,
            upper_bounds: bounds.upper_bounds,
            // Contracted shape per ReLU, zero-filled: `num_specs × idx.len()`,
            // in the same fold order as `beta_signed`. An empty index list
            // yields an empty vec, matching "nothing gathered for that ReLU".
            beta_gather: beta_gather_idx
                .iter()
                .map(|idx| vec![0.0f32; seed.num_specs * idx.len()])
                .collect(),
        })
    }

    fn crown_backward_gpu_resnet_sound_beta_batched(
        &self,
        domains: &[ny_core::GpuResnetBatchedDomainRef<'_>],
        seed: &ny_core::GpuCrownSeed,
    ) -> ny_core::Result<Vec<ny_core::GpuCrownResult>> {
        domains
            .iter()
            .map(|domain| {
                self.bounds(
                    domain.segments,
                    seed,
                    domain.input_lower,
                    domain.input_upper,
                )
            })
            .collect()
    }

    /// The gradient-capturing wide batched backward: SOUND BOUNDS, STATIONARY
    /// GRADIENTS.
    ///
    /// The bounds are the same f64 fold every other entry on this fixture uses,
    /// per domain — so the wide β-opt lane publishes exactly the enclosure the
    /// non-gather batched entry would. What this fixture does NOT reproduce is
    /// the coefficient stream: a real backend gathers `A_lower` at the union
    /// columns off the resident frontier, and `joint_alpha_grad` does not expose
    /// its per-ReLU `a_pre` checkpoints. Fabricating those values would make the
    /// β ascent LOOK exercised while stepping on numbers no backend produced.
    ///
    /// So the captures are returned at their exact contracted shapes, filled
    /// with zeros. A zero gradient is a valid ascent direction — it is the
    /// stationary one — so β holds at each domain's starting dual and every
    /// published iterate stays the sound wide fold. The consumer's own
    /// shape check (`vals.len() == n_domains * nsp * u_r`) is therefore
    /// satisfied rather than skipped, which is the point: the indexing path
    /// runs. The α side is unaffected either way — the caller computes the TRUE
    /// joint α-gradient on the CPU from the domain's segments (INC2) and only
    /// falls back to this entry's `alpha_grads` under `NY_WIDE_ALPHA_LOCAL=1`.
    ///
    /// Gradient TRANSPORT has its own dedicated recording seams; what this entry
    /// exists to make non-vacuous is the enclosure and composition contract of
    /// the lanes that cannot be reached without it.
    fn crown_backward_gpu_resnet_sound_beta_batched_grad(
        &self,
        domains: &[ny_core::GpuResnetBatchedDomainRef<'_>],
        seed: &ny_core::GpuCrownSeed,
        union_gather_idx: &[&[u32]],
        relu_pre_lower: &[&[Vec<f32>]],
    ) -> ny_core::Result<(Vec<ny_core::GpuCrownResult>, Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        let bounds = domains
            .iter()
            .map(|domain| {
                self.bounds(
                    domain.segments,
                    seed,
                    domain.input_lower,
                    domain.input_upper,
                )
            })
            .collect::<ny_core::Result<Vec<_>>>()?;

        let n_domains = domains.len();
        let gathers = union_gather_idx
            .iter()
            .map(|columns| vec![0.0f32; n_domains * seed.num_specs * columns.len()])
            .collect();
        // Empty `relu_pre_lower` is the contract's "no capture requested", and
        // its answer is an empty vec — not a zero-filled one of unknown width.
        let alpha_grads = relu_pre_lower
            .iter()
            .map(|per_domain| {
                let nn = per_domain.first().map_or(0, Vec::len);
                vec![0.0f32; n_domains * nn]
            })
            .collect();
        Ok((bounds, alpha_grads, gathers))
    }

    fn crown_joint_alpha_gradient_resident(
        &self,
        segments: &[ny_core::GpuResnetSegment],
        seed_lower_a: &[f32],
        num_specs: usize,
        output_dim: usize,
        input_lower: &[f32],
        input_upper: &[f32],
    ) -> ny_core::Result<Vec<Vec<f32>>> {
        use std::sync::atomic::Ordering;

        self.gradient_calls.fetch_add(1, Ordering::Relaxed);
        ny_core::joint_alpha_grad::joint_alpha_gradient(
            segments,
            seed_lower_a,
            &vec![0.0; num_specs],
            num_specs,
            output_dim,
            input_lower,
            input_upper,
            ny_core::joint_alpha_grad::JointGradConfig::default(),
        )
        .ok_or_else(|| NyError::InvalidSpec("hermetic joint-alpha fold rejected its shape".into()))
    }
}

/// End-to-end tests for the prune lane + the 2-layer cascade on a tiny
/// two-residual-block net (the smallest structure the ResNet segment
/// extraction accepts for both seed layers).  Default tests use the hermetic
/// sound-capability adapter above; `gpu-tests` adds a fail-fast raw-WGPU run.
#[cfg(test)]
mod gpu_tests {
    use super::*;
    use crate::beta_crown::branching::{GraphNeuronConstraint, GraphSplitHistory};
    use crate::beta_crown::state::GraphBetaEntry;
    use crate::layers::{AddLayer, LinearLayer, ReLULayer};
    use crate::GraphNode;
    use ndarray::{arr1, arr2, ArrayD, IxDyn};

    fn hermetic_engine() -> HermeticSoundGpuCrownEngine {
        HermeticSoundGpuCrownEngine::default()
    }

    fn lin_w(name: &str, input: &str, w: [[f32; 2]; 2], b: [f32; 2]) -> GraphNode {
        let layer =
            Layer::Linear(LinearLayer::new(arr2(&w), Some(arr1(&b))).expect("valid linear"));
        if input == NETWORK_INPUT {
            GraphNode::from_input(name, layer)
        } else {
            GraphNode::new(name, layer, vec![input.to_string()])
        }
    }

    /// Two stacked residual blocks, all 2-d, then the last-ReLU tail:
    /// input → l1(I) → relu1 → l2(0.5·I) → add1(l2, l1)
    ///       → g3(I) → relu3 → l4(0.5·I) → add2(l4, g3)
    ///       → gemm_pre(I) → relu_last → out(I).
    /// Per coordinate: add1 = f(x), add2 = f(f(x)) with f(t) = 0.5·relu(t) + t
    /// (t if t < 0, 1.5·t if t ≥ 0), and gemm_pre = add2 exactly.
    fn build_two_block_net() -> GraphNetwork {
        let ident = [[1.0f32, 0.0], [0.0, 1.0]];
        let half = [[0.5f32, 0.0], [0.0, 0.5]];
        let zero = [0.0f32, 0.0];
        let mut g = GraphNetwork::new();
        g.add_node(lin_w("l1", NETWORK_INPUT, ident, zero));
        g.add_node(GraphNode::new(
            "relu1",
            Layer::ReLU(ReLULayer),
            vec!["l1".to_string()],
        ));
        g.add_node(lin_w("l2", "relu1", half, zero));
        g.add_node(GraphNode::new(
            "add1",
            Layer::Add(AddLayer),
            vec!["l2".to_string(), "l1".to_string()],
        ));
        g.add_node(lin_w("g3", "add1", ident, zero));
        g.add_node(GraphNode::new(
            "relu3",
            Layer::ReLU(ReLULayer),
            vec!["g3".to_string()],
        ));
        g.add_node(lin_w("l4", "relu3", half, zero));
        g.add_node(GraphNode::new(
            "add2",
            Layer::Add(AddLayer),
            vec!["l4".to_string(), "g3".to_string()],
        ));
        g.add_node(lin_w("gemm_pre", "add2", ident, zero));
        g.add_node(GraphNode::new(
            "relu_last",
            Layer::ReLU(ReLULayer),
            vec!["gemm_pre".to_string()],
        ));
        g.add_node(lin_w("out", "relu_last", ident, zero));
        g.set_output("out");
        g
    }

    /// Exact forward of the tiny net: gemm_pre(x) per coordinate.
    fn gemm_pre_of(x: [f32; 2]) -> [f32; 2] {
        let f = |t: f32| 0.5 * t.max(0.0) + t;
        [f(f(x[0])), f(f(x[1]))]
    }

    fn box2(lo: [f32; 2], hi: [f32; 2]) -> BoundedTensor {
        BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2]), lo.to_vec()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2]), hi.to_vec()).unwrap(),
        )
        .unwrap()
    }

    fn opts(prune: bool, layers: usize) -> IntermRefineOptions {
        IntermRefineOptions {
            deadline: None,
            unstable_rows_only: true,
            prune,
            layers,
            max_dim: 2048,
            deep_max_rows: 256,
            selective_topk: 0,
            prune_tol: 1e-4,
            probe: false,
            min_depth: 0,
            seeds: None,
            alpha_iters: 0,
            alpha_lr: 0.05,
            alpha_max_rows: 64,
            alpha_reopt: false,
            alpha_store: None,
            adaptive_latch: None,
            adaptive_floor: 4,
            redo_every: 0,
            wide_max_n: 0,
            joint_margin: false,
            margin_weights: None,
            clip_resnet: false,
            require_complete_clip_root_bank: false,
            per_target: false,
            interm_box_probe: false,
            clip_guard: false,
        }
    }

    /// End-to-end prune + soundness on a real sound GPU backward:
    ///
    /// * Domain 0 (input x0 ∈ [−1, −0.9]) carries the split premise
    ///   `relu_last[0] ACTIVE` (z0 ≥ 0), but z0 = x0 ≤ −0.9 over its whole
    ///   box — contradictory through the net ⇒ MUST prune (infeasible). Even
    ///   with the ROOT-frozen chord relaxations the refinement inherits
    ///   (chord of [−1, 1.5] at both ReLUs), the refined upper is
    ///   1.625·x0 + 0.625 ≤ −0.8375 < 0 − tol.
    /// * Domain 1 (x0 ∈ [−0.5, 1]) carries the same premise, which is
    ///   satisfiable there (z0 ∈ [−0.5, 2.25]) ⇒ must NEVER prune; sampled
    ///   premise-satisfying points must stay inside the refined enclosure.
    /// * With layers=2, the deep pass (seed g3, one residual below) runs
    ///   first and the cascade must keep every entry sound (sample check on
    ///   BOTH seed entries) — and still never prune the feasible domain.
    #[test]
    fn interm_refine_prune_e2e_sound_gpu() {
        let device = hermetic_engine();
        let engine: &dyn GemmEngine = &device;
        let graph = build_two_block_net();
        let verifier = BetaCrownVerifier::default();

        // ROOT-frozen caches + the premise clamp, exactly like BaB children:
        // node bounds computed on the ROOT box, `apply_pre_constraints` clamps
        // the seed entry (relu_last[0] active ⇒ gemm_pre[0].l = 0).
        let root = box2([-1.0, -1.0], [1.0, 1.0]);
        let mut history = GraphSplitHistory::new();
        history.add_constraint(
            GraphNeuronConstraint::new("relu_last".to_string(), 0, true, 0.0)
                .expect("valid constraint"),
        );
        let (cache, _root_input) = verifier
            .compute_constrained_forward_bounds(&graph, &root, &history, None, None)
            .expect("constrained forward on the root box");
        // Root entry for gemm_pre[0] must be unstable-then-clamped (premise
        // admissible at root — the refinement, not the forward clamp, must be
        // what proves infeasibility).
        let pre = cache.get("gemm_pre").expect("seed entry");
        assert_eq!(pre.lower()[[0]], 0.0, "active clamp on the seed entry");
        assert!(pre.upper()[[0]] > 0.0);

        let beta = GraphBetaState::from_entries(vec![GraphBetaEntry::new(
            "relu_last".into(),
            0,
            0.0,
            0.0,
            1.0,
        )
        .expect("valid entry")]);

        let caches = vec![cache.clone(), cache];
        let inputs = vec![
            box2([-1.0, -1.0], [-0.9, 1.0]), // premise z0 ≥ 0 is IMPOSSIBLE here
            box2([-0.5, -1.0], [1.0, 1.0]),  // premise satisfiable here
        ];
        let histories = vec![&history, &history];
        let betas: Vec<Option<&GraphBetaState>> = vec![Some(&beta), Some(&beta)];
        let alphas: Vec<Option<&GraphDomainAlphaState>> = vec![None, None];

        for layers in [1usize, 2] {
            let outcome = verifier
                .refine_interm_bounds_with_opts(
                    &graph,
                    "out",
                    2,
                    &caches,
                    &inputs,
                    &histories,
                    &betas,
                    &alphas,
                    engine,
                    &opts(true, layers),
                )
                .expect("refinement must apply on the sound GPU lane");
            assert!(
                outcome.infeasible[0],
                "contradictory premise domain must prune (layers={layers})"
            );
            assert!(
                !outcome.infeasible[1],
                "feasible domain must NEVER prune (layers={layers})"
            );

            // Soundness: sampled premise-satisfying points of the FEASIBLE
            // domain stay inside the refined enclosures (gemm_pre always; g3
            // too — the layers=2 cascade must not corrupt the deep entry).
            let refined = &outcome.caches[1];
            let (in_lo, in_hi) = ([-0.5f32, -1.0], [1.0f32, 1.0]);
            for i in 0..=8 {
                for j in 0..=8 {
                    let x = [
                        in_lo[0] + (in_hi[0] - in_lo[0]) * (i as f32) / 8.0,
                        in_lo[1] + (in_hi[1] - in_lo[1]) * (j as f32) / 8.0,
                    ];
                    let z = gemm_pre_of(x);
                    if z[0] < 0.0 {
                        continue; // premise z0 ≥ 0 not satisfied — out of domain
                    }
                    let entry = refined.get("gemm_pre").expect("refined seed entry");
                    for (k, &zk) in z.iter().enumerate() {
                        assert!(
                            entry.lower()[[k]] <= zk + 1e-5 && zk - 1e-5 <= entry.upper()[[k]],
                            "gemm_pre[{k}]={zk} outside refined [{}, {}] at x={x:?} (layers={layers})",
                            entry.lower()[[k]],
                            entry.upper()[[k]],
                        );
                    }
                    if layers == 2 {
                        // g3 = f(x) per coordinate.
                        let f = |t: f32| 0.5 * t.max(0.0) + t;
                        let g = [f(x[0]), f(x[1])];
                        let entry = refined.get("g3").expect("deep seed entry");
                        for (k, &gk) in g.iter().enumerate() {
                            assert!(
                                entry.lower()[[k]] <= gk + 1e-5 && gk - 1e-5 <= entry.upper()[[k]],
                                "g3[{k}]={gk} outside refined [{}, {}] at x={x:?}",
                                entry.lower()[[k]],
                                entry.upper()[[k]],
                            );
                        }
                    }
                }
            }

            // With the prune lane OFF, the same batch must not flag anything
            // (dark default = today's behavior).
            if let Some(outcome_off) = verifier.refine_interm_bounds_with_opts(
                &graph,
                "out",
                2,
                &caches,
                &inputs,
                &histories,
                &betas,
                &alphas,
                engine,
                &opts(false, layers),
            ) {
                assert!(outcome_off.infeasible.iter().all(|&b| !b));
            }
        }

        // The deep-seed discovery itself (layers=2 walks add2's F-branch).
        let (relu_name, seed) = find_next_relu_seed_below(&graph, "gemm_pre").expect("deep seed");
        assert_eq!((relu_name.as_str(), seed.as_str()), ("relu3", "g3"));
    }

    /// #wide-chunk e2e on the sound GPU lane: forcing 1-domain chunks
    /// (`wide_max_n=1`) must reproduce the single-call refinement — same
    /// prune flags, same refined entries within the wide-vs-serial f32
    /// reorder tolerance (the chunked calls route each domain through the
    /// same batched entry, just in smaller groups).
    #[test]
    fn interm_refine_wide_chunk_matches_single_call_gpu() {
        let device = hermetic_engine();
        let engine: &dyn GemmEngine = &device;
        let graph = build_two_block_net();
        let verifier = BetaCrownVerifier::default();
        let root = box2([-1.0, -1.0], [1.0, 1.0]);
        let history = GraphSplitHistory::new();
        let (cache, _root_input) = verifier
            .compute_constrained_forward_bounds(&graph, &root, &history, None, None)
            .expect("constrained forward on the root box");
        let caches = vec![cache.clone(), cache.clone(), cache];
        let inputs = vec![
            box2([-1.0, -1.0], [0.0, 1.0]),
            box2([-0.5, -1.0], [1.0, 1.0]),
            box2([-1.0, 0.0], [1.0, 1.0]),
        ];
        let histories = vec![&history, &history, &history];
        let betas: Vec<Option<&GraphBetaState>> = vec![None, None, None];
        let alphas: Vec<Option<&GraphDomainAlphaState>> = vec![None, None, None];
        let run = |wide_max_n: usize| {
            verifier.refine_interm_bounds_with_opts(
                &graph,
                "out",
                3,
                &caches,
                &inputs,
                &histories,
                &betas,
                &alphas,
                engine,
                &IntermRefineOptions {
                    wide_max_n,
                    ..opts(false, 2)
                },
            )
        };
        let single = run(0).expect("single-call refinement applies");
        let chunked = run(1).expect("chunked refinement applies");
        assert_eq!(single.infeasible, chunked.infeasible);
        for (cs, cc) in single.caches.iter().zip(chunked.caches.iter()) {
            for node in ["gemm_pre", "relu_last", "g3", "relu3"] {
                let (es, ec) = (cs.get(node).expect(node), cc.get(node).expect(node));
                for k in 0..es.len() {
                    let (ls, us) = (es.lower()[[k]], es.upper()[[k]]);
                    let (lc, uc) = (ec.lower()[[k]], ec.upper()[[k]]);
                    assert!(
                        (ls - lc).abs() <= 1e-5 && (us - uc).abs() <= 1e-5,
                        "{node}[{k}]: single [{ls}, {us}] vs chunked [{lc}, {uc}]"
                    );
                }
            }
        }
    }

    /// ADAPTIVE SCHEDULE (#adaptive-refine) e2e on the sound GPU lane:
    ///
    /// * A zero-yield batch (no stability flip is POSSIBLE — the crossing
    ///   coordinate's true range spans 0, so any sound refinement keeps it
    ///   crossing — and prune is off) at depth 1 with floor 1 must TRIP the
    ///   latch to 1.
    /// * Once latched, a batch whose domains all sit at depth ≥ 1 must be
    ///   SKIPPED (returns `None` ⇒ caller keeps inherited caches).
    /// * With floor 2 the same zero-yield depth-1 batch must NOT latch
    ///   (early-depth protection).
    /// * `adaptive_latch: None` (the dark default) never touches anything.
    #[test]
    fn interm_refine_adaptive_latch_trips_and_skips_gpu() {
        use std::sync::atomic::AtomicUsize;
        let device = hermetic_engine();
        let engine: &dyn GemmEngine = &device;
        let graph = build_two_block_net();
        let verifier = BetaCrownVerifier::default();

        // Same fixture as the prune e2e: root-frozen caches + the ACTIVE
        // premise clamp on relu_last[0] (depth 1). Feasible domain only, prune
        // OFF ⇒ yield can only come from a stability flip; gemm_pre[1]'s true
        // range over the box spans 0 (f(f(−1)) = −1 < 0 < 2.25 = f(f(1))), and
        // gemm_pre[0] is already clamp-stable (l = 0), so a SOUND refinement
        // can never produce newly_stable > 0 here — zero yield by construction.
        let root = box2([-1.0, -1.0], [1.0, 1.0]);
        let mut history = GraphSplitHistory::new();
        history.add_constraint(
            GraphNeuronConstraint::new("relu_last".to_string(), 0, true, 0.0)
                .expect("valid constraint"),
        );
        let (cache, _root_input) = verifier
            .compute_constrained_forward_bounds(&graph, &root, &history, None, None)
            .expect("constrained forward on the root box");
        let beta = GraphBetaState::from_entries(vec![GraphBetaEntry::new(
            "relu_last".into(),
            0,
            0.0,
            0.0,
            1.0,
        )
        .expect("valid entry")]);
        let caches = vec![cache];
        let inputs = vec![box2([-0.5, -1.0], [1.0, 1.0])];
        let histories = vec![&history];
        let betas: Vec<Option<&GraphBetaState>> = vec![Some(&beta)];
        let alphas: Vec<Option<&GraphDomainAlphaState>> = vec![None];
        let run = |o: &IntermRefineOptions| {
            verifier.refine_interm_bounds_with_opts(
                &graph, "out", 1, &caches, &inputs, &histories, &betas, &alphas, engine, o,
            )
        };

        // Floor 1, fresh latch: the zero-yield depth-1 batch must trip it.
        let latch: AdaptiveLatch = Arc::new(AtomicUsize::new(usize::MAX));
        let o1 = IntermRefineOptions {
            adaptive_latch: Some(latch.clone()),
            adaptive_floor: 1,
            ..opts(false, 1)
        };
        let _ = run(&o1);
        assert_eq!(
            latch.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "zero-yield depth-1 batch at floor 1 must latch at depth 1"
        );

        // Latched: the same batch (all domains at depth ≥ 1) must be skipped.
        assert!(
            run(&o1).is_none(),
            "latched schedule must skip the whole deep batch"
        );

        // Floor 2: the same zero-yield depth-1 batch must NOT latch.
        let latch2: AdaptiveLatch = Arc::new(AtomicUsize::new(usize::MAX));
        let o2 = IntermRefineOptions {
            adaptive_latch: Some(latch2.clone()),
            adaptive_floor: 2,
            ..opts(false, 1)
        };
        let _ = run(&o2);
        assert_eq!(
            latch2.load(std::sync::atomic::Ordering::Relaxed),
            usize::MAX,
            "below-floor batch must never latch"
        );

        // #interm-refine-redo through the tripped latch: the depth-1 batch is
        // an exact multiple of stride 1 ⇒ it must NOT be skipped — the call
        // behaves exactly like the unlatched lane (same Some/None outcome).
        // Stride 2 leaves depth 1 (1 % 2 ≠ 0) latched out ⇒ skip (None).
        let unlatched = run(&IntermRefineOptions {
            adaptive_latch: None,
            ..opts(false, 1)
        });
        let pre_latched: AdaptiveLatch = Arc::new(AtomicUsize::new(1));
        let redo1 = run(&IntermRefineOptions {
            adaptive_latch: Some(pre_latched.clone()),
            adaptive_floor: 1,
            redo_every: 1,
            ..opts(false, 1)
        });
        assert_eq!(
            unlatched.is_some(),
            redo1.is_some(),
            "redo stride 1 must re-include the latched depth-1 batch"
        );
        let redo2 = run(&IntermRefineOptions {
            adaptive_latch: Some(pre_latched),
            adaptive_floor: 1,
            redo_every: 2,
            ..opts(false, 1)
        });
        assert!(
            redo2.is_none(),
            "depth 1 is not a multiple of stride 2 — the latched batch must stay skipped"
        );
    }

    // ==== α′-for-refinement (#alpha-prime) fixtures: a DENSE two-block net —
    // mixed-sign weights so ν rows exercise both relaxation branches. Same
    // topology as `build_two_block_net` (the smallest structure the resnet
    // extraction accepts), weights shared with the concrete forward below.
    const DW1: [[f32; 2]; 2] = [[0.9, -0.4], [0.3, 0.8]];
    const DB1: [f32; 2] = [0.05, -0.05];
    const DW2: [[f32; 2]; 2] = [[0.5, 0.3], [-0.4, 0.6]];
    const DB2: [f32; 2] = [0.0, 0.05];
    const DW3: [[f32; 2]; 2] = [[0.7, -0.5], [0.4, 0.9]];
    const DB3: [f32; 2] = [-0.05, 0.0];
    const DW4: [[f32; 2]; 2] = [[0.6, 0.2], [-0.3, 0.5]];
    const DB4: [f32; 2] = [0.05, 0.05];
    const DW5: [[f32; 2]; 2] = [[1.0, -0.6], [0.5, 0.8]];
    const DB5: [f32; 2] = [0.0, -0.05];

    fn build_two_block_net_dense() -> GraphNetwork {
        let ident = [[1.0f32, 0.0], [0.0, 1.0]];
        let mut g = GraphNetwork::new();
        g.add_node(lin_w("l1", NETWORK_INPUT, DW1, DB1));
        g.add_node(GraphNode::new(
            "relu1",
            Layer::ReLU(ReLULayer),
            vec!["l1".to_string()],
        ));
        g.add_node(lin_w("l2", "relu1", DW2, DB2));
        g.add_node(GraphNode::new(
            "add1",
            Layer::Add(AddLayer),
            vec!["l2".to_string(), "l1".to_string()],
        ));
        g.add_node(lin_w("g3", "add1", DW3, DB3));
        g.add_node(GraphNode::new(
            "relu3",
            Layer::ReLU(ReLULayer),
            vec!["g3".to_string()],
        ));
        g.add_node(lin_w("l4", "relu3", DW4, DB4));
        g.add_node(GraphNode::new(
            "add2",
            Layer::Add(AddLayer),
            vec!["l4".to_string(), "g3".to_string()],
        ));
        g.add_node(lin_w("gemm_pre", "add2", DW5, DB5));
        g.add_node(GraphNode::new(
            "relu_last",
            Layer::ReLU(ReLULayer),
            vec!["gemm_pre".to_string()],
        ));
        g.add_node(lin_w("out", "relu_last", ident, [0.0, 0.0]));
        g.set_output("out");
        g
    }

    /// Concrete `gemm_pre(x)` of the dense net (the soundness oracle).
    fn dense_gemm_pre_of(x: [f32; 2]) -> [f32; 2] {
        let mv = |w: [[f32; 2]; 2], b: [f32; 2], v: [f32; 2]| {
            [
                w[0][0] * v[0] + w[0][1] * v[1] + b[0],
                w[1][0] * v[0] + w[1][1] * v[1] + b[1],
            ]
        };
        let relu = |v: [f32; 2]| [v[0].max(0.0), v[1].max(0.0)];
        let l1 = mv(DW1, DB1, x);
        let l2 = mv(DW2, DB2, relu(l1));
        let a1 = [l2[0] + l1[0], l2[1] + l1[1]];
        let g3 = mv(DW3, DB3, a1);
        let l4 = mv(DW4, DB4, relu(g3));
        let a2 = [l4[0] + g3[0], l4[1] + g3[1]];
        mv(DW5, DB5, a2)
    }

    /// Explicit real-device counterpart to the hermetic capability tests.
    ///
    /// This is feature-selected because it is an infrastructure contract, not
    /// a portable unit test.  Once selected it has no skip path: adapter
    /// creation, the five-rung authority qualification, extraction, dispatch,
    /// and enclosure checking all fail loudly on debt or unsupported hardware.
    #[cfg(feature = "gpu-tests")]
    #[test]
    fn interm_refine_public_qualified_wgpu_sound_fold_is_non_vacuous() {
        let device = ny_gpu::WgpuDevice::new_for_verdict(ny_gpu::WgpuVerdictRequest::new())
            .expect("gpu-tests requires a WGPU device that passes all five authority rungs");
        let engine: &dyn GemmEngine = &device;
        let gpu = engine
            .as_gpu_crown_backward()
            .filter(|candidate| candidate.provides_sound_gpu_crown())
            .expect("the typed proof constructor must expose its qualified CROWN seam");

        let graph = build_two_block_net_dense();
        let verifier = BetaCrownVerifier::default();
        let input = box2([-1.0, -1.0], [1.0, 1.0]);
        let history = GraphSplitHistory::new();
        let (cache, root_input) = verifier
            .compute_constrained_forward_bounds(&graph, &input, &history, None, None)
            .expect("root forward bounds");
        let prep = prep_resnet_domain(&graph, "gemm_pre", &cache, &root_input, None, None, false)
            .expect("extract the real refinement stack");
        let seed = ny_core::GpuCrownSeed {
            lower_a: vec![1.0, 0.0, 0.0, 1.0].into(),
            upper_a: vec![1.0, 0.0, 0.0, 1.0].into(),
            lower_b: vec![0.0; 2].into(),
            upper_b: vec![0.0; 2].into(),
            num_specs: 2,
            current_dim: 2,
        };
        let result = gpu
            .crown_backward_gpu_resnet_sound_beta(
                &prep.segments,
                &seed,
                &prep.in_lo,
                &prep.in_hi,
                &prep.beta_signed,
                &prep.frontier_abs,
                &prep.node_abs,
            )
            .expect("qualified public WGPU sound fold must execute");
        assert_eq!(result.lower_bounds.len(), 2);
        assert_eq!(result.upper_bounds.len(), 2);

        let mut samples = 0usize;
        for i in 0..=8 {
            for j in 0..=8 {
                let x = [-1.0 + i as f32 / 4.0, -1.0 + j as f32 / 4.0];
                let z = dense_gemm_pre_of(x);
                for row in 0..2 {
                    assert!(
                        result.lower_bounds[row] <= z[row] + 1e-4
                            && z[row] - 1e-4 <= result.upper_bounds[row],
                        "qualified WGPU row {row} excludes {z:?} at {x:?}: [{}, {}]",
                        result.lower_bounds[row],
                        result.upper_bounds[row],
                    );
                }
                samples += 1;
            }
        }
        assert_eq!(samples, 81, "real-device enclosure oracle was vacuous");
    }

    /// THE FD ORACLE (#alpha-prime deliverable 1): finite differences of the
    /// refinement objective `Σ_rows l′_row` through the ACTUAL sound GPU
    /// refinement backward (the very
    /// `crown_backward_gpu_resnet_sound_beta_batched` call
    /// `refine_one_seed_pass` makes, on the truncated stack
    /// `prep_resnet_domain` builds at the seed node) must match the analytic
    /// per-row-summed TRUE chain-rule gradient `refine_alpha_objective_grads`
    /// returns — the gradient the α′ ascent steps on.
    #[test]
    fn interm_refine_alpha_fd_oracle_gpu() {
        let device = hermetic_engine();
        let engine: &dyn GemmEngine = &device;
        let Some(gpu) = engine
            .as_gpu_crown_backward()
            .filter(|g| g.provides_sound_gpu_crown())
        else {
            panic!("hermetic engine must expose sound GPU CROWN capability");
        };
        let graph = build_two_block_net_dense();
        let verifier = BetaCrownVerifier::default();
        let root = box2([-1.0, -1.0], [1.0, 1.0]);
        let history = crate::beta_crown::branching::GraphSplitHistory::new();
        let (cache, root_input) = verifier
            .compute_constrained_forward_bounds(&graph, &root, &history, None, None)
            .expect("root forward bounds");
        let mut prep =
            prep_resnet_domain(&graph, "gemm_pre", &cache, &root_input, None, None, false)
                .expect("truncated-stack prep at the seed node");
        let n_relu = prep.relu_names.len();
        assert_eq!(n_relu, 2, "relu1 + relu3 below the seed");
        assert_eq!(count_activations(&prep.segments), n_relu);

        // Interior α (off the 0/1 boundary and the CROWN adaptive kink) so
        // central differences stay inside [0,1].
        visit_activations_mut(&mut prep.segments, &mut |r, layer| {
            if let ny_core::GpuCrownLayer::Activation {
                lower_slope,
                upper_intercept,
                ..
            } = layer
            {
                for i in 0..lower_slope.len() {
                    if upper_intercept[i] > 0.0 {
                        lower_slope[i] = 0.35 + 0.3 * (((r + i) % 2) as f32);
                    }
                }
            }
        });
        let masks = unstable_masks(&prep.segments, n_relu);
        assert!(
            masks.iter().flatten().filter(|&&b| b).count() >= 2,
            "fixture must have unstable below-seed neurons"
        );

        let pre_dim = 2usize;
        let mut rows_seed = vec![0.0f32; pre_dim * pre_dim];
        rows_seed[0] = 1.0;
        rows_seed[3] = 1.0;
        let seed = ny_core::GpuCrownSeed {
            lower_a: rows_seed.clone().into(),
            upper_a: rows_seed.into(),
            lower_b: vec![0.0f32; pre_dim].into(),
            upper_b: vec![0.0f32; pre_dim].into(),
            num_specs: pre_dim,
            current_dim: pre_dim,
        };
        let run = |segs: &[ny_core::GpuResnetSegment]| -> Vec<f32> {
            let refs = vec![ny_core::GpuResnetBatchedDomainRef {
                segments: segs,
                input_lower: &prep.in_lo,
                input_upper: &prep.in_hi,
                beta_signed: &prep.beta_signed,
                frontier_abs: &prep.frontier_abs,
                node_abs: &prep.node_abs,
            }];
            gpu.crown_backward_gpu_resnet_sound_beta_batched(&refs, &seed)
                .expect("batched refinement backward")[0]
                .lower_bounds
                .clone()
        };
        let obj = |segs: &[ny_core::GpuResnetSegment]| -> f64 {
            run(segs).iter().map(|&v| v as f64).sum()
        };

        let base_lbs = run(&prep.segments);
        let sel: Vec<usize> = (0..pre_dim).collect();
        let rows: Vec<usize> = (0..pre_dim).collect();
        let (grads, rows_ok) = refine_alpha_objective_grads(
            &prep.segments,
            &prep.beta_signed,
            &prep.in_lo,
            &prep.in_hi,
            n_relu,
            pre_dim,
            &sel,
            &rows,
            &base_lbs,
            None,
            false,
        );
        assert_eq!(
            rows_ok, pre_dim,
            "every identity-row replay must validate against the GPU bound"
        );

        // Per-row ν for the near-kink skip rule (mirrors the wide-α oracle).
        let nus: Vec<Vec<Vec<f32>>> = (0..pre_dim)
            .map(|r| {
                let mut row = vec![0.0f32; pre_dim];
                row[r] = 1.0;
                super::super::wide_alpha_true::replay_critical_row(
                    &prep.segments,
                    &row,
                    &prep.beta_signed,
                )
                .expect("replay")
                .nu
            })
            .collect();

        let h = 4e-3f32;
        let mut checked = 0usize;
        for r in 0..n_relu {
            for i in 0..masks[r].len() {
                if !masks[r][i] {
                    continue;
                }
                let perturb = |delta: f32| -> f64 {
                    let mut segs = prep.segments.clone();
                    visit_activations_mut(&mut segs, &mut |rr, layer| {
                        if rr == r {
                            if let ny_core::GpuCrownLayer::Activation { lower_slope, .. } = layer {
                                lower_slope[i] += delta;
                            }
                        }
                    });
                    obj(&segs)
                };
                let fd = ((perturb(h) - perturb(-h)) / (2.0 * h as f64)) as f32;
                let g = grads[r][i];
                let tol = 4e-3 + 0.06 * fd.abs();
                let near_kink = nus.iter().any(|nu| nu[r][i].abs() < 5e-2);
                if (fd - g).abs() > tol && near_kink {
                    continue;
                }
                assert!(
                    (fd - g).abs() <= tol,
                    "relu {r} neuron {i}: GPU FD {fd} != analytic {g} (tol {tol})"
                );
                checked += 1;
            }
        }
        assert!(
            checked >= 2,
            "oracle must exercise enough neurons ({checked})"
        );
        assert!(
            grads.iter().flatten().any(|&g| g.abs() > 1e-4),
            "gradient signal must exist on the dense net"
        );
    }

    /// α′ ascent integration (#alpha-prime): with the lane ON the refined
    /// bounds are NEVER looser than the borrowed-α run (iterate 0 is the same
    /// backward; the merge only tightens), sampled concrete pre-activations
    /// stay enclosed (soundness), the α′ store fills, and a second (reuse)
    /// call stays sound.
    #[test]
    fn interm_refine_alpha_ascent_never_looser_sound_and_reuses() {
        let device = hermetic_engine();
        let engine: &dyn GemmEngine = &device;
        let graph = build_two_block_net_dense();
        let verifier = BetaCrownVerifier::default();
        let root = box2([-1.0, -1.0], [1.0, 1.0]);
        let history = crate::beta_crown::branching::GraphSplitHistory::new();
        let (cache, root_input) = verifier
            .compute_constrained_forward_bounds(&graph, &root, &history, None, None)
            .expect("root forward bounds");
        let caches = vec![cache];
        let inputs = vec![root_input];
        let histories = vec![&history];
        let betas: Vec<Option<&GraphBetaState>> = vec![None];
        let alphas: Vec<Option<&GraphDomainAlphaState>> = vec![None];
        let run = |o: &IntermRefineOptions| {
            verifier.refine_interm_bounds_with_opts(
                &graph, "out", 1, &caches, &inputs, &histories, &betas, &alphas, engine, o,
            )
        };

        let off = run(&opts(false, 1)).expect("borrowed-α refinement must apply");
        let store: AlphaPrimeStore = Arc::new(Mutex::new(None));
        let on_opts = IntermRefineOptions {
            alpha_iters: 4,
            alpha_lr: 0.15,
            alpha_max_rows: 8,
            alpha_store: Some(store.clone()),
            ..opts(false, 1)
        };
        let on = run(&on_opts).expect("α′ refinement must apply");

        let (o_pre, n_pre) = (&off.caches[0]["gemm_pre"], &on.caches[0]["gemm_pre"]);
        for k in 0..2 {
            assert!(
                n_pre.lower()[[k]] >= o_pre.lower()[[k]] - 1e-6
                    && n_pre.upper()[[k]] <= o_pre.upper()[[k]] + 1e-6,
                "α′ merge must never loosen: neuron {k} ON [{}, {}] vs OFF [{}, {}]",
                n_pre.lower()[[k]],
                n_pre.upper()[[k]],
                o_pre.lower()[[k]],
                o_pre.upper()[[k]],
            );
        }
        let stored = store.lock().expect("store lock");
        let improved = stored.as_ref().map(|ap| ap.improved);
        assert!(stored.is_some(), "ascent must record an α′ snapshot");
        drop(stored);
        eprintln!("[test] alpha-prime improved={improved:?}");

        // Soundness of BOTH the ascent run and the reuse run: concrete
        // gemm_pre(x) of every sampled root point inside the refined bounds.
        let on2 = run(&on_opts).expect("reuse refinement must apply");
        for outcome in [&on, &on2] {
            let entry = &outcome.caches[0]["gemm_pre"];
            for i in 0..=8 {
                for j in 0..=8 {
                    let x = [-1.0 + 0.25 * i as f32, -1.0 + 0.25 * j as f32];
                    let z = dense_gemm_pre_of(x);
                    for (k, &zk) in z.iter().enumerate() {
                        assert!(
                            entry.lower()[[k]] <= zk + 1e-4 && zk - 1e-4 <= entry.upper()[[k]],
                            "gemm_pre[{k}]={zk} outside refined [{}, {}] at x={x:?}",
                            entry.lower()[[k]],
                            entry.upper()[[k]],
                        );
                    }
                }
            }
        }
    }

    /// #ab-parity-interm PER-TARGET lane: with `per_target` ON the refined
    /// bounds are NEVER looser than the borrowed-α run (each target's iterate-0
    /// bound seeds the merge; only element-wise-tightest sound GPU folds are
    /// kept), and every sampled concrete pre-activation stays enclosed
    /// (soundness). Two domains so the application phase's batched write/read
    /// over all domains is exercised.
    #[test]
    fn interm_refine_per_target_never_looser_and_sound_gpu() {
        let device = hermetic_engine();
        let engine: &dyn GemmEngine = &device;
        let graph = build_two_block_net_dense();
        let verifier = BetaCrownVerifier::default();
        let root = box2([-1.0, -1.0], [1.0, 1.0]);
        let history = crate::beta_crown::branching::GraphSplitHistory::new();
        let (cache, _root_input) = verifier
            .compute_constrained_forward_bounds(&graph, &root, &history, None, None)
            .expect("root forward bounds");
        // Two domains sharing the root cache but different input sub-boxes: the
        // per-target application phase writes each α into BOTH and reads columns.
        let caches = vec![cache.clone(), cache];
        let inputs = vec![
            box2([-1.0, -1.0], [1.0, 1.0]),
            box2([-0.5, -0.75], [0.75, 1.0]),
        ];
        let histories = vec![&history, &history];
        let betas: Vec<Option<&GraphBetaState>> = vec![None, None];
        let alphas: Vec<Option<&GraphDomainAlphaState>> = vec![None, None];
        let run = |o: &IntermRefineOptions| {
            verifier.refine_interm_bounds_with_opts(
                &graph, "out", 2, &caches, &inputs, &histories, &betas, &alphas, engine, o,
            )
        };

        let off = run(&opts(false, 1)).expect("borrowed-α refinement must apply");
        let store: AlphaPrimeStore = Arc::new(Mutex::new(None));
        let on_opts = IntermRefineOptions {
            alpha_iters: 4,
            alpha_lr: 0.15,
            alpha_max_rows: 8,
            alpha_store: Some(store.clone()),
            per_target: true,
            ..opts(false, 1)
        };
        let on = run(&on_opts).expect("per-target refinement must apply");

        // Never looser than the borrowed-α run, in BOTH domains.
        for d in 0..2 {
            let (o_pre, n_pre) = (&off.caches[d]["gemm_pre"], &on.caches[d]["gemm_pre"]);
            for k in 0..2 {
                assert!(
                    n_pre.lower()[[k]] >= o_pre.lower()[[k]] - 1e-6
                        && n_pre.upper()[[k]] <= o_pre.upper()[[k]] + 1e-6,
                    "per-target merge must never loosen: dom {d} neuron {k} ON [{}, {}] vs OFF [{}, {}]",
                    n_pre.lower()[[k]],
                    n_pre.upper()[[k]],
                    o_pre.lower()[[k]],
                    o_pre.upper()[[k]],
                );
            }
        }
        // Per-target drops the shared-α store reuse — nothing is written to it.
        assert!(
            store.lock().expect("store lock").is_none(),
            "per-target lane must NOT populate the shared-α store (reuse dropped)"
        );

        // Soundness: concrete gemm_pre(x) of every sampled point of each domain's
        // OWN input box is inside that domain's refined bounds.
        for (d, dbox) in [
            ([-1.0f32, -1.0], [1.0f32, 1.0]),
            ([-0.5, -0.75], [0.75, 1.0]),
        ]
        .into_iter()
        .enumerate()
        {
            let (lo, hi) = dbox;
            let entry = &on.caches[d]["gemm_pre"];
            for i in 0..=8 {
                for j in 0..=8 {
                    let x = [
                        lo[0] + (hi[0] - lo[0]) * i as f32 / 8.0,
                        lo[1] + (hi[1] - lo[1]) * j as f32 / 8.0,
                    ];
                    let z = dense_gemm_pre_of(x);
                    for (k, &zk) in z.iter().enumerate() {
                        assert!(
                            entry.lower()[[k]] <= zk + 1e-4 && zk - 1e-4 <= entry.upper()[[k]],
                            "dom {d} gemm_pre[{k}]={zk} outside per-target refined [{}, {}] at x={x:?}",
                            entry.lower()[[k]],
                            entry.upper()[[k]],
                        );
                    }
                }
            }
        }
    }

    /// NAMED-SEED lane (`NY_INTERM_REFINE_SEEDS`, #midref): naming the same
    /// two ReLUs the layers=2 walk finds must produce BIT-IDENTICAL refined
    /// caches and the same prune flags, regardless of token order (exec-order
    /// sort) and casing of `last`; unknown/non-ReLU tokens are skipped; a list
    /// with no resolvable seed refuses (returns None).
    #[test]
    fn interm_refine_named_seeds_match_layers_walk() {
        let device = hermetic_engine();
        let engine: &dyn GemmEngine = &device;
        let graph = build_two_block_net();
        let verifier = BetaCrownVerifier::default();

        let root = box2([-1.0, -1.0], [1.0, 1.0]);
        let mut history = GraphSplitHistory::new();
        history.add_constraint(
            GraphNeuronConstraint::new("relu_last".to_string(), 0, true, 0.0)
                .expect("valid constraint"),
        );
        let (cache, _root_input) = verifier
            .compute_constrained_forward_bounds(&graph, &root, &history, None, None)
            .expect("constrained forward on the root box");
        let beta = GraphBetaState::from_entries(vec![GraphBetaEntry::new(
            "relu_last".into(),
            0,
            0.0,
            0.0,
            1.0,
        )
        .expect("valid entry")]);
        let caches = vec![cache.clone(), cache];
        let inputs = vec![
            box2([-1.0, -1.0], [-0.9, 1.0]),
            box2([-0.5, -1.0], [1.0, 1.0]),
        ];
        let histories = vec![&history, &history];
        let betas: Vec<Option<&GraphBetaState>> = vec![Some(&beta), Some(&beta)];
        let alphas: Vec<Option<&GraphDomainAlphaState>> = vec![None, None];

        let run = |seeds: Option<Vec<String>>, layers: usize| {
            verifier.refine_interm_bounds_with_opts(
                &graph,
                "out",
                2,
                &caches,
                &inputs,
                &histories,
                &betas,
                &alphas,
                engine,
                &IntermRefineOptions {
                    seeds,
                    ..opts(true, layers)
                },
            )
        };
        let walk = run(None, 2).expect("layers=2 walk must apply");
        for tokens in [
            vec!["relu3".to_string(), "last".to_string()],
            // Out of order + shouty `LAST` + junk tokens: same outcome.
            vec![
                "LAST".to_string(),
                "nope".to_string(),
                "l2".to_string(),
                "relu3".to_string(),
            ],
        ] {
            let named = run(Some(tokens.clone()), 1).expect("named seeds must apply");
            assert_eq!(named.infeasible, walk.infeasible, "tokens={tokens:?}");
            for node in ["g3", "relu3", "gemm_pre", "relu_last"] {
                let a = &walk.caches[1][node];
                let b = &named.caches[1][node];
                assert_eq!(a.lower(), b.lower(), "node={node} tokens={tokens:?}");
                assert_eq!(a.upper(), b.upper(), "node={node} tokens={tokens:?}");
            }
        }
        // No resolvable seed ⇒ the lane refuses (keeps inherited caches).
        assert!(
            run(Some(vec!["nope".to_string()]), 1).is_none(),
            "unresolvable seed list must refuse"
        );
    }

    /// LEG-A e2e on the direct research clip path (`clip_resnet + clip_guard`).
    /// Runs the full `refine_interm_bounds_with_opts` with explicit test options
    /// and the batched split-constraint clip ARMED on ≥2 heterogeneous last-ReLU
    /// split domains, then asserts every premise-satisfying sample's true forward
    /// stays inside the REFINED (clip-tightened) `gemm_pre` enclosure — the common-mode
    /// GPU coeff→fold→clip→merge path the CPU oracle cannot reach. The runtime guard
    /// is armed too (must not spuriously revert a sound clip). Also checks the clip is
    /// NEVER-LOOSER than the gate-OFF refinement.
    #[test]
    fn leg_a_clip_e2e_enclosure_and_guard_gpu() {
        let device = hermetic_engine();
        let engine: &dyn GemmEngine = &device;
        let graph = build_two_block_net();
        let verifier = BetaCrownVerifier::default();

        // Two heterogeneous last-ReLU split domains (different premise + box).
        let root = box2([-1.0, -1.0], [1.0, 1.0]);
        let mk = |neuron: usize, active: bool, lo: [f32; 2], hi: [f32; 2]| {
            let mut history = GraphSplitHistory::new();
            history.add_constraint(
                GraphNeuronConstraint::new("relu_last".to_string(), neuron, active, 0.0)
                    .expect("valid constraint"),
            );
            let (cache, _cin) = verifier
                .compute_constrained_forward_bounds(&graph, &root, &history, None, None)
                .expect("constrained forward");
            let beta = GraphBetaState::from_entries(vec![GraphBetaEntry::new(
                "relu_last".into(),
                neuron,
                0.0,
                0.0,
                if active { 1.0 } else { -1.0 },
            )
            .expect("entry")]);
            (cache, beta, box2(lo, hi), (neuron, active), history)
        };
        let d0 = mk(0, true, [-0.5, -1.0], [1.0, 1.0]); // gemm_pre[0] ≥ 0
        let d1 = mk(1, false, [-1.0, -1.0], [1.0, 0.6]); // gemm_pre[1] ≤ 0

        let caches = vec![d0.0.clone(), d1.0.clone()];
        let inputs = vec![d0.2.clone(), d1.2.clone()];
        let histories = vec![&d0.4, &d1.4];
        let betas: Vec<Option<&GraphBetaState>> = vec![Some(&d0.1), Some(&d1.1)];
        let alphas: Vec<Option<&GraphDomainAlphaState>> = vec![None, None];
        let prem = [d0.3, d1.3];
        let dom_boxes = [([-0.5f32, -1.0], [1.0f32, 1.0]), ([-1.0, -1.0], [1.0, 0.6])];

        let run = |clip: bool| {
            verifier
                .refine_interm_bounds_with_opts(
                    &graph,
                    "out",
                    2,
                    &caches,
                    &inputs,
                    &histories,
                    &betas,
                    &alphas,
                    engine,
                    &IntermRefineOptions {
                        clip_resnet: clip,
                        clip_guard: clip, // the NY_CLIP_INTERM umbrella arms both
                        ..opts(false, 1)
                    },
                )
                .expect("refinement must apply on the sound GPU lane")
        };
        let off = run(false);
        let on = run(true);

        let mut clip_tightened = false;
        for di in 0..2 {
            let (lo, hi) = dom_boxes[di];
            let (pn, pa) = prem[di];
            let refined = on.caches[di].get("gemm_pre").expect("refined seed");
            let base = off.caches[di].get("gemm_pre").expect("base seed");
            for k in 0..2 {
                // NEVER-LOOSER: clip-ON must be at least as tight as clip-OFF.
                assert!(
                    refined.lower()[[k]] >= base.lower()[[k]] - 1e-4
                        && refined.upper()[[k]] <= base.upper()[[k]] + 1e-4,
                    "dom {di} gemm_pre[{k}]: clip-ON [{}, {}] looser than clip-OFF [{}, {}]",
                    refined.lower()[[k]],
                    refined.upper()[[k]],
                    base.lower()[[k]],
                    base.upper()[[k]],
                );
                if refined.lower()[[k]] > base.lower()[[k]] + 1e-4
                    || refined.upper()[[k]] < base.upper()[[k]] - 1e-4
                {
                    clip_tightened = true;
                }
            }
            // ENCLOSURE: premise-satisfying grid samples stay inside the refined box.
            for i in 0..=16 {
                for j in 0..=16 {
                    let x = [
                        lo[0] + (hi[0] - lo[0]) * (i as f32) / 16.0,
                        lo[1] + (hi[1] - lo[1]) * (j as f32) / 16.0,
                    ];
                    let z = gemm_pre_of(x);
                    let sat = if pa { z[pn] >= 0.0 } else { z[pn] <= 0.0 };
                    if !sat {
                        continue;
                    }
                    for k in 0..2 {
                        let tol = 1e-4 * (1.0 + z[k].abs());
                        assert!(
                            refined.lower()[[k]] - tol <= z[k]
                                && z[k] <= refined.upper()[[k]] + tol,
                            "LEG-A(GPU) UNSOUND: dom {di} gemm_pre[{k}]={} escapes refined \
                             [{}, {}] at x={x:?}",
                            z[k],
                            refined.lower()[[k]],
                            refined.upper()[[k]],
                        );
                    }
                }
            }
        }
        eprintln!("[leg-a-gpu] clip engaged (tightened vs OFF)={clip_tightened}");
    }
}

/// #clip-interm SOUNDNESS harness (Leg-A enclosure oracle + Leg-B batched-vs-serial
/// differential + fold/shape guard + the fail-closed runtime guard). These drive the
/// PRODUCTION per-domain clip functions (`fold_seed_rows_for_domain`,
/// `clip_seed_domain`, `clip_guard_verify_domain`) directly on a hand-built
/// `GpuResidentCoeffBatched`, so they run WITHOUT a GPU and are fully deterministic —
/// the load-bearing soundness gate for the bound-TIGHTENING clip lever.
#[cfg(test)]
mod clip_interm_soundness {
    use super::*;
    use crate::beta_crown::branching::{GenBabConstraint, GraphNeuronConstraint};
    use crate::beta_crown::state::GraphBetaEntry;
    use crate::complete_clip::{test_certified_affine_fixture, TestSoundCrownAffineParts};
    use crate::layers::{LinearLayer, ReLULayer};
    use crate::GraphNode;
    use ndarray::{arr1, arr2, ArrayD, IxDyn};
    use num_rational::BigRational;
    use ny_core::GpuResidentCoeffBatched;

    // The seed net: input(3) → gemm_pre(Linear W,b) → relu_last(ReLU) → out(I).
    // `gemm_pre` is EXACTLY affine in the network input, so the folded seed rows are
    // an EXACT enclosure and the split constraints are EXACT — the HARDEST (no-slack)
    // adversarial case for the clip's enclosure DIRECTION and its dual concretization.
    const W: [[f32; 3]; 3] = [[1.0, 0.4, -0.2], [-0.3, 0.9, 0.5], [0.6, -0.7, 0.8]];
    const B: [f32; 3] = [0.1, -0.2, 0.15];

    fn oracle_net() -> GraphNetwork {
        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::from_input(
            "gemm_pre",
            Layer::Linear(LinearLayer::new(arr2(&W), Some(arr1(&B))).expect("linear")),
        ));
        g.add_node(GraphNode::new(
            "relu_last",
            Layer::ReLU(ReLULayer),
            vec!["gemm_pre".to_string()],
        ));
        let ident = [[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        g.add_node(GraphNode::new(
            "out",
            Layer::Linear(
                LinearLayer::new(arr2(&ident), Some(arr1(&[0.0, 0.0, 0.0]))).expect("id"),
            ),
            vec!["relu_last".to_string()],
        ));
        g.set_output("out");
        g
    }

    /// True forward of `gemm_pre` at a point (exact through the production forward on
    /// a degenerate box — the ground-truth sampler, not a re-implementation).
    fn forward_gemm_pre(g: &GraphNetwork, x: [f32; 3]) -> [f32; 3] {
        let arr = ArrayD::from_shape_vec(IxDyn(&[3]), x.to_vec()).unwrap();
        let pt = BoundedTensor::new(arr.clone(), arr).unwrap();
        let vals = g.collect_node_bounds(&pt).unwrap();
        let gp = vals.get("gemm_pre").unwrap().flatten();
        [gp.lower()[[0]], gp.lower()[[1]], gp.lower()[[2]]]
    }

    /// Batched coeff for `n_domains` blocks of the same seed W/b (the input-relative
    /// CROWN coeff of a Linear seed is W regardless of box); per-domain certified
    /// per-coeff error `err_lo[d]`/`err_hi[d]` (folded OUTWARD by
    /// `fold_seed_rows_for_domain`). Layout `= (d*n_rows + r)*dim + j`, exactly the
    /// invariant the fold's `local_block` indexing relies on.
    fn batched_coeff(n_domains: usize, err_lo: &[f32], err_hi: &[f32]) -> GpuResidentCoeffBatched {
        let (dim, n_rows) = (3usize, 3usize);
        let num_specs = n_domains * n_rows;
        let mut c = GpuResidentCoeffBatched {
            lower_a: vec![0.0; num_specs * dim],
            upper_a: vec![0.0; num_specs * dim],
            lower_err: vec![0.0; num_specs * dim],
            upper_err: vec![0.0; num_specs * dim],
            lower_b: vec![0.0; num_specs],
            upper_b: vec![0.0; num_specs],
            lower_b_err: vec![0.0; num_specs],
            upper_b_err: vec![0.0; num_specs],
            dim,
            num_specs,
            num_specs_per_dom: n_rows,
        };
        for d in 0..n_domains {
            for r in 0..n_rows {
                let s = d * n_rows + r;
                for j in 0..dim {
                    c.lower_a[s * dim + j] = W[r][j];
                    c.upper_a[s * dim + j] = W[r][j];
                    c.lower_err[s * dim + j] = err_lo[d];
                    c.upper_err[s * dim + j] = err_hi[d];
                }
                c.lower_b[s] = B[r];
                c.upper_b[s] = B[r];
            }
        }
        c
    }

    fn beta(prem: &[(usize, bool)]) -> GraphBetaState {
        GraphBetaState::from_entries(
            prem.iter()
                .map(|&(idx, active)| {
                    GraphBetaEntry::new(
                        "relu_last".into(),
                        idx,
                        0.0,
                        0.0,
                        if active { 1.0 } else { -1.0 },
                    )
                    .expect("valid entry")
                })
                .collect(),
        )
    }

    fn relu_history(prem: &[(usize, bool)]) -> GraphSplitHistory {
        let mut history = GraphSplitHistory::new();
        for &(idx, active) in prem {
            history.add_constraint(
                GraphNeuronConstraint::new("relu_last".into(), idx, active, 0.0)
                    .expect("valid ReLU split history"),
            );
        }
        history
    }

    /// Route legacy clip soundness fixtures through the same provenance-required
    /// compatibility seam as production.  The raw-parts constructor is compiled
    /// only for tests; non-test code has no way to mint the token/pass pair.
    #[allow(clippy::too_many_arguments)]
    fn provenance_fixture(
        folded: &FoldedSeedRows,
        history: &GraphSplitHistory,
        relu_name: &str,
        row_of: &[usize],
        sel: &[usize],
        in_lo: &[f32],
        in_hi: &[f32],
        pass_words: [u64; 2],
    ) -> Option<(CrownPassStamp, CertifiedAffineEnclosure)> {
        let parts = TestSoundCrownAffineParts {
            lower_a: Array2::from_shape_vec((folded.n_rows, folded.dim), folded.lower_a.clone())
                .ok()?,
            upper_a: Array2::from_shape_vec((folded.n_rows, folded.dim), folded.upper_a.clone())
                .ok()?,
            lower_a_error: Array2::zeros((folded.n_rows, folded.dim)),
            upper_a_error: Array2::zeros((folded.n_rows, folded.dim)),
            lower_bias_center: Array1::from_vec(folded.lower_b.clone()),
            upper_bias_center: Array1::from_vec(folded.upper_b.clone()),
            lower_bias_error: Array1::zeros(folded.n_rows),
            upper_bias_error: Array1::zeros(folded.n_rows),
        };
        test_certified_affine_fixture(
            pass_words,
            b"clip-soundness-oracle-v1",
            in_lo,
            in_hi,
            history,
            relu_name,
            sel,
            row_of,
            parts,
        )
        .ok()
    }

    #[allow(clippy::too_many_arguments)]
    fn clip_seed_domain(
        folded: &FoldedSeedRows,
        history: &GraphSplitHistory,
        beta_state: &GraphBetaState,
        relu_name: &str,
        row_of: &[usize],
        sel: &[usize],
        in_lo: &[f32],
        in_hi: &[f32],
        deadline: Option<Instant>,
    ) -> Option<(Vec<f32>, Vec<f32>)> {
        let (pass, token) = provenance_fixture(
            folded,
            history,
            relu_name,
            row_of,
            sel,
            in_lo,
            in_hi,
            [0x434c_4950, 0x5052_4f56],
        )?;
        super::clip_seed_domain(
            folded,
            Some((&token, &pass)),
            history,
            beta_state,
            relu_name,
            row_of,
            sel,
            in_lo,
            in_hi,
            deadline,
        )
    }

    /// The compatibility seam is authority-by-token, never authority-by-array.
    /// A valid test-only CROWN fixture reaches the existing independently checked
    /// Complete Clipping dual; every exact-context mismatch returns `None`, which
    /// is the caller's inherited-bound path.
    #[ntest::timeout(20000)]
    #[test]
    fn affine_provenance_token_is_required_and_exactly_context_bound() {
        let lo = [-1.0f32, -1.0, -1.0];
        let hi = [1.0f32, 1.0, 1.0];
        let sel = [0usize, 1, 2];
        let row_of = [0usize, 1, 2];
        let state = beta(&[(1, true)]);
        let history = relu_history(&[(1, true)]);
        let mut proposal =
            fold_seed_rows_for_domain(&batched_coeff(1, &[0.0], &[0.0]), 0, 3, &lo, &hi, None)
                .expect("finite proposal");
        let (pass, token) = provenance_fixture(
            &proposal,
            &history,
            "relu_last",
            &row_of,
            &sel,
            &lo,
            &hi,
            [41, 1],
        )
        .expect("test-only sound CROWN fixture");

        assert!(
            super::clip_seed_domain(
                &proposal,
                None,
                &history,
                &state,
                "relu_last",
                &row_of,
                &sel,
                &lo,
                &hi,
                None,
            )
            .is_none(),
            "raw affine arrays without a token must inherit bounds"
        );
        let valid = super::clip_seed_domain(
            &proposal,
            Some((&token, &pass)),
            &history,
            &state,
            "relu_last",
            &row_of,
            &sel,
            &lo,
            &hi,
            None,
        )
        .expect("valid token must replay through the checked clipping dual");
        assert!(valid.0.iter().chain(&valid.1).all(|v| v.is_finite()));

        let other_node = super::clip_seed_domain(
            &proposal,
            Some((&token, &pass)),
            &history,
            &state,
            "other_relu",
            &row_of,
            &sel,
            &lo,
            &hi,
            None,
        );
        assert!(other_node.is_none(), "mismatched node must inherit");

        let changed_hi = [1.0f32, 1.0, 0.75];
        let other_domain = super::clip_seed_domain(
            &proposal,
            Some((&token, &pass)),
            &history,
            &state,
            "relu_last",
            &row_of,
            &sel,
            &lo,
            &changed_hi,
            None,
        );
        assert!(
            other_domain.is_none(),
            "mismatched input domain must inherit"
        );

        let other_state = beta(&[(1, false)]);
        let other_history = relu_history(&[(1, false)]);
        let changed_history = super::clip_seed_domain(
            &proposal,
            Some((&token, &pass)),
            &other_history,
            &other_state,
            "relu_last",
            &row_of,
            &sel,
            &lo,
            &hi,
            None,
        );
        assert!(changed_history.is_none(), "mismatched history must inherit");

        let other_sel = [1usize, 0, 2];
        let other_row_of = [1usize, 0, 2];
        let changed_objective = super::clip_seed_domain(
            &proposal,
            Some((&token, &pass)),
            &history,
            &state,
            "relu_last",
            &other_row_of,
            &other_sel,
            &lo,
            &hi,
            None,
        );
        assert!(
            changed_objective.is_none(),
            "mismatched objective must inherit"
        );

        let (new_pass, _new_token) = provenance_fixture(
            &proposal,
            &history,
            "relu_last",
            &row_of,
            &sel,
            &lo,
            &hi,
            [41, 2],
        )
        .expect("second pass fixture");
        let stale = super::clip_seed_domain(
            &proposal,
            Some((&token, &new_pass)),
            &history,
            &state,
            "relu_last",
            &row_of,
            &sel,
            &lo,
            &hi,
            None,
        );
        assert!(stale.is_none(), "stale pass token must inherit");

        let original_lower_a = proposal.lower_a[0];
        proposal.lower_a[0] = next_up_f32(original_lower_a);
        let altered_raw = super::clip_seed_domain(
            &proposal,
            Some((&token, &pass)),
            &history,
            &state,
            "relu_last",
            &row_of,
            &sel,
            &lo,
            &hi,
            None,
        );
        assert!(
            altered_raw.is_none(),
            "a one-ULP tighter raw GPU proposal must inherit"
        );

        proposal.lower_a[0] = f32::NAN;
        assert!(
            super::clip_seed_domain(
                &proposal,
                Some((&token, &pass)),
                &history,
                &state,
                "relu_last",
                &row_of,
                &sel,
                &lo,
                &hi,
                None,
            )
            .is_none(),
            "a NaN raw GPU proposal must inherit"
        );

        proposal.lower_a[0] = original_lower_a;
        proposal.upper_a.pop();
        let rejected_shape = super::clip_seed_domain(
            &proposal,
            Some((&token, &pass)),
            &history,
            &state,
            "relu_last",
            &row_of,
            &sel,
            &lo,
            &hi,
            None,
        );
        assert!(
            rejected_shape.is_none(),
            "a wrong-shaped raw GPU proposal must inherit"
        );

        // Model the production publication boundary: only an authorized replay
        // result enters `apply_selected_bounds`. A failed replay leaves both
        // inherited cache Arcs byte-identically in place.
        let inherited_pre = Arc::new(
            BoundedTensor::new(
                arr1(&[-2.0, -1.0, -0.5]).into_dyn(),
                arr1(&[2.0, 1.0, 0.5]).into_dyn(),
            )
            .expect("inherited pre bounds"),
        );
        let inherited_post = Arc::new(
            BoundedTensor::new(
                arr1(&[0.0, 0.0, 0.0]).into_dyn(),
                arr1(&[2.0, 1.0, 0.5]).into_dyn(),
            )
            .expect("inherited post bounds"),
        );
        let mut inherited_cache = HashMap::from([
            ("seed".to_string(), Arc::clone(&inherited_pre)),
            ("relu_last".to_string(), Arc::clone(&inherited_post)),
        ]);
        if let Some((lower, upper)) = rejected_shape {
            let mut stats = RefineStats::default();
            let _ = apply_selected_bounds(
                &mut inherited_cache,
                "seed",
                "relu_last",
                &lower,
                &upper,
                3,
                &sel,
                None,
                &mut stats,
            );
        }
        assert!(Arc::ptr_eq(
            inherited_cache.get("seed").expect("seed inherited"),
            &inherited_pre
        ));
        assert!(Arc::ptr_eq(
            inherited_cache.get("relu_last").expect("ReLU inherited"),
            &inherited_post
        ));

        assert!(!clip_interm_resnet_batched_enabled());
        assert!(!clip_interm_umbrella_enabled());
    }

    fn premises_satisfied(prem: &[(usize, bool)], z: [f32; 3]) -> bool {
        prem.iter()
            .all(|&(j, active)| if active { z[j] >= 0.0 } else { z[j] <= 0.0 })
    }

    /// Box-only (unconstrained IBP) range of seed row r over [lo,hi] — the range the
    /// clip must TIGHTEN INSIDE OF to be non-vacuous, and stay OUTSIDE OF to be sound.
    fn box_range(r: usize, lo: [f32; 3], hi: [f32; 3]) -> (f32, f32) {
        // Bit-identical (a+b)*0.5 anchor (f32 center fixture).
        #[allow(clippy::manual_midpoint)]
        let (x0, eps): ([f32; 3], [f32; 3]) = (
            [
                (lo[0] + hi[0]) * 0.5,
                (lo[1] + hi[1]) * 0.5,
                (lo[2] + hi[2]) * 0.5,
            ],
            [
                (hi[0] - lo[0]) * 0.5,
                (hi[1] - lo[1]) * 0.5,
                (hi[2] - lo[2]) * 0.5,
            ],
        );
        let mut center = B[r];
        let mut rad = 0.0f32;
        for j in 0..3 {
            center += W[r][j] * x0[j];
            rad += W[r][j].abs() * eps[j];
        }
        (center - rad, center + rad)
    }

    /// LEG-A GROUND-TRUTH ENCLOSURE ORACLE (the load-bearing soundness gate).
    ///
    /// For ≥2 heterogeneous split domains: fold the batched coeff → run the PRODUCTION
    /// per-domain clip (`clip_seed_domain`) → then DIRECTED adversarial search (dense
    /// grid + per-row box-corner pushes toward crossing clip_lower/clip_upper) for
    /// FEASIBLE points (satisfying the child's ReLU split signs on the true forward),
    /// asserting `clip_lower[r] ≤ z_r(x) ≤ clip_upper[r]` for EVERY sampled feasible x.
    /// A feasible x outside the tightened box PROVES the clip unsound (false-VERIFY) →
    /// this test goes RED and names the exact domain/row/value. ≥200 feasible points
    /// across the domains; the clip must also genuinely tighten (non-vacuity).
    #[ntest::timeout(60000)]
    #[test]
    fn leg_a_clip_enclosure_oracle_cpu() {
        let g = oracle_net();
        let sel = [0usize, 1, 2];
        let row_of = [0usize, 1, 2];

        // Two heterogeneous domains: different premises, different boxes, and a
        // genuine-relaxation (err>0) block to exercise the OUTWARD error fold.
        struct Dom {
            lo: [f32; 3],
            hi: [f32; 3],
            prem: Vec<(usize, bool)>,
            block: usize,
        }
        let doms = [
            Dom {
                lo: [-1.0, -1.0, -1.0],
                hi: [1.0, 1.0, 1.0],
                prem: vec![(0, true), (2, false)], // z0 ≥ 0, z2 ≤ 0
                block: 0,
            },
            Dom {
                lo: [-1.0, -0.8, -1.0],
                hi: [0.8, 1.0, 0.9],
                prem: vec![(1, true)], // z1 ≥ 0
                block: 1,
            },
        ];
        // block 0: exact (err 0). block 1: err>0 (folded looser — still an enclosure).
        let coeff = batched_coeff(2, &[0.0, 0.02], &[0.0, 0.02]);

        let mut total_feasible = 0usize;
        let mut tightened_rows = 0usize;

        for (di, d) in doms.iter().enumerate() {
            let lo = d.lo;
            let hi = d.hi;
            let folded = fold_seed_rows_for_domain(&coeff, d.block, 3, &lo, &hi, None)
                .expect("fold must succeed on finite coeff");
            assert!(
                folded.refused.iter().all(|&r| !r),
                "dom {di}: finite coeff must not refuse any row"
            );
            let bs = beta(&d.prem);
            let split_history = relu_history(&d.prem);
            let (tl, tu) = clip_seed_domain(
                &folded,
                &split_history,
                &bs,
                "relu_last",
                &row_of,
                &sel,
                &lo,
                &hi,
                None,
            )
            .unwrap_or_else(|| {
                panic!("dom {di}: clip must run (constraints non-empty on a full-sign box)")
            });
            assert_eq!(tl.len(), 3);
            assert_eq!(tu.len(), 3);

            // Non-vacuity: the clip must TIGHTEN inside the box-only range on ≥1 row.
            for r in 0..3 {
                let (blo, bhi) = box_range(r, lo, hi);
                if tl[r] > blo + 1e-3 || tu[r] < bhi - 1e-3 {
                    tightened_rows += 1;
                }
            }

            let mut state: u64 = 0xD1CE_0000 ^ (di as u64);
            let mut u01 = || {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 40) as f32) / ((1u64 << 24) as f32)
            };

            let check = |x: [f32; 3], feasible_count: &mut usize| {
                let z = forward_gemm_pre(&g, x);
                if !premises_satisfied(&d.prem, z) {
                    return;
                }
                *feasible_count += 1;
                for r in 0..3 {
                    let tol = 1e-3 * (1.0 + z[r].abs());
                    assert!(
                        tl[r] - tol <= z[r] && z[r] <= tu[r] + tol,
                        "LEG-A UNSOUND: dom {di} row {r} true z={} escapes clip [{}, {}] at x={x:?} \
                         (premises {:?}) — a too-tight intermediate bound = false-VERIFY",
                        z[r],
                        tl[r],
                        tu[r],
                        d.prem,
                    );
                }
            };

            let mut feasible = 0usize;
            // Dense grid.
            let n = 16i32;
            for i0 in 0..=n {
                for i1 in 0..=n {
                    for i2 in 0..=n {
                        let x = [
                            lo[0] + (hi[0] - lo[0]) * (i0 as f32) / (n as f32),
                            lo[1] + (hi[1] - lo[1]) * (i1 as f32) / (n as f32),
                            lo[2] + (hi[2] - lo[2]) * (i2 as f32) / (n as f32),
                        ];
                        check(x, &mut feasible);
                    }
                }
            }
            // DIRECTED corner pushes: extremize each row's folded objective toward the
            // box corner a too-tight bound would be exposed at, then jitter.
            for k in 0..600usize {
                let r = (k / 2) % 3;
                let test_upper = (k / 2 / 3) % 2 == 0;
                let mut x = [0.0f32; 3];
                for j in 0..3 {
                    let a = if test_upper {
                        folded.upper_a[r * 3 + j]
                    } else {
                        folded.lower_a[r * 3 + j]
                    };
                    let corner = if (test_upper && a > 0.0) || (!test_upper && a < 0.0) {
                        hi[j]
                    } else {
                        lo[j]
                    };
                    x[j] = (corner + (u01() - 0.5) * 0.25 * (hi[j] - lo[j])).clamp(lo[j], hi[j]);
                }
                check(x, &mut feasible);
            }
            assert!(
                feasible >= 100,
                "dom {di}: only {feasible} feasible samples — oracle vacuous"
            );
            total_feasible += feasible;
        }
        assert!(
            total_feasible >= 200,
            "oracle must sample ≥200 feasible points, got {total_feasible}"
        );
        assert!(
            tightened_rows > 0,
            "the clip never tightened any row vs the box-only range — test would be vacuous"
        );
    }

    /// LEG-B BATCHED-vs-SERIAL DIFFERENTIAL (misalignment guard). The batched clip
    /// folds domain d's block at `local_block = d` of the STACKED coeff; the serial
    /// reference builds a fresh 1-block coeff carrying ONLY domain d's own error and
    /// folds it at `local_block = 0` — sharing NO batched indexing. Per row the two
    /// must agree within f32 slop (a `local_block` misalignment would hand domain d
    /// another block's coefficients ⇒ a divergent, possibly too-tight, clip).
    #[ntest::timeout(30000)]
    #[test]
    fn leg_b_batched_vs_serial_clip_differential_cpu() {
        let sel = [0usize, 1, 2];
        let row_of = [0usize, 1, 2];
        // Distinct per-domain error ⇒ blocks genuinely differ (so the differential
        // has discriminating power against a misalignment).
        let err = [0.0f32, 0.03, 0.07];
        let batched = batched_coeff(3, &err, &err);
        let boxes = [
            ([-1.0f32, -1.0, -1.0], [1.0f32, 1.0, 1.0]),
            ([-0.9f32, -1.0, -0.7], [1.0f32, 0.6, 1.0]),
            ([-1.0f32, -0.5, -1.0], [0.5f32, 1.0, 0.8]),
        ];
        let prems: [Vec<(usize, bool)>; 3] = [
            vec![(0, true)],
            vec![(1, false)],
            vec![(2, true), (0, false)],
        ];

        let mut distinct_folds = 0usize;
        let mut prev_ub: Option<f32> = None;
        for d in 0..3 {
            let (lo, hi) = boxes[d];
            let bs = beta(&prems[d]);
            let split_history = relu_history(&prems[d]);

            let bf =
                fold_seed_rows_for_domain(&batched, d, 3, &lo, &hi, None).expect("batched fold");
            // Serial reference: a private 1-block coeff with ONLY this domain's error.
            let serial_coeff = batched_coeff(1, &[err[d]], &[err[d]]);
            let sf = fold_seed_rows_for_domain(&serial_coeff, 0, 3, &lo, &hi, None)
                .expect("serial fold");

            // Folded rows must be numerically identical (same numbers, no shared index).
            for r in 0..3 {
                assert!(
                    (bf.lower_b[r] - sf.lower_b[r]).abs() <= 1e-4
                        && (bf.upper_b[r] - sf.upper_b[r]).abs() <= 1e-4,
                    "dom {d} row {r}: batched fold [{},{}] ≠ serial fold [{},{}] — MISALIGNMENT",
                    bf.lower_b[r],
                    bf.upper_b[r],
                    sf.lower_b[r],
                    sf.upper_b[r],
                );
            }

            let (btl, btu) = clip_seed_domain(
                &bf,
                &split_history,
                &bs,
                "relu_last",
                &row_of,
                &sel,
                &lo,
                &hi,
                None,
            )
            .expect("batched clip");
            let (stl, stu) = clip_seed_domain(
                &sf,
                &split_history,
                &bs,
                "relu_last",
                &row_of,
                &sel,
                &lo,
                &hi,
                None,
            )
            .expect("serial clip");
            for r in 0..3 {
                // Task form: batched ≤ serial + slop (not tighter beyond f32 reorder)
                // AND batched ≥ serial − slop (sound side / no drift).
                assert!(
                    btl[r] <= stl[r] + 1e-3 && btl[r] >= stl[r] - 1e-3,
                    "dom {d} row {r} LOWER: batched {} vs serial {} beyond slop",
                    btl[r],
                    stl[r]
                );
                assert!(
                    btu[r] <= stu[r] + 1e-3 && btu[r] >= stu[r] - 1e-3,
                    "dom {d} row {r} UPPER: batched {} vs serial {} beyond slop",
                    btu[r],
                    stu[r]
                );
            }
            // Non-tautology: the per-domain blocks must actually differ (else a
            // misalignment would be invisible).
            if prev_ub.is_some_and(|p| (p - bf.upper_b[0]).abs() > 1e-6) {
                distinct_folds += 1;
            }
            prev_ub = Some(bf.upper_b[0]);
        }
        assert!(
            distinct_folds > 0,
            "batched blocks were indistinguishable — the differential cannot catch a misalignment"
        );
    }

    /// `fold_seed_rows_for_domain` shape/indexing guard + OUTWARD error fold.
    #[ntest::timeout(20000)]
    #[test]
    fn fold_seed_rows_shape_and_outward_fold_cpu() {
        let lo = [-1.0f32, -1.0, -1.0];
        let hi = [1.0f32, 1.0, 1.0];
        let coeff = batched_coeff(2, &[0.05, 0.05], &[0.05, 0.05]); // num_specs=6

        assert!(fold_seed_rows_for_domain(&coeff, 0, 3, &lo, &hi, None).is_some());
        assert!(fold_seed_rows_for_domain(&coeff, 1, 3, &lo, &hi, None).is_some());
        // Out-of-range block (would read rows 6,7,8 ≥ num_specs) → None (fail-closed).
        assert!(
            fold_seed_rows_for_domain(&coeff, 2, 3, &lo, &hi, None).is_none(),
            "out-of-range block must refuse (the s>=num_specs bound)"
        );
        // Wrong input dim → None.
        assert!(fold_seed_rows_for_domain(&coeff, 0, 3, &[0.0, 0.0], &[1.0, 1.0], None).is_none());
        let mut malformed = batched_coeff(2, &[0.05, 0.05], &[0.05, 0.05]);
        malformed.lower_err.pop();
        assert!(
            fold_seed_rows_for_domain(&malformed, 0, 3, &lo, &hi, None).is_none(),
            "a truncated certified-error table must refuse"
        );
        let mut wrong_stride = batched_coeff(2, &[0.05, 0.05], &[0.05, 0.05]);
        wrong_stride.num_specs_per_dom = 2;
        assert!(
            fold_seed_rows_for_domain(&wrong_stride, 0, 3, &lo, &hi, None).is_none(),
            "domain-row stride mismatch must refuse"
        );
        let mut partial_block = batched_coeff(2, &[0.05, 0.05], &[0.05, 0.05]);
        partial_block.num_specs -= 1;
        partial_block
            .lower_a
            .truncate(partial_block.num_specs * partial_block.dim);
        partial_block
            .upper_a
            .truncate(partial_block.num_specs * partial_block.dim);
        partial_block
            .lower_err
            .truncate(partial_block.num_specs * partial_block.dim);
        partial_block
            .upper_err
            .truncate(partial_block.num_specs * partial_block.dim);
        partial_block.lower_b.truncate(partial_block.num_specs);
        partial_block.upper_b.truncate(partial_block.num_specs);
        partial_block.lower_b_err.truncate(partial_block.num_specs);
        partial_block.upper_b_err.truncate(partial_block.num_specs);
        assert!(
            fold_seed_rows_for_domain(&partial_block, 0, 3, &lo, &hi, None).is_none(),
            "a partial final domain block must refuse globally"
        );

        // OUTWARD fold: with err>0 the folded box strictly contains B, and the folded
        // affine form encloses the true z = W·x + b at every sampled point.
        let f = fold_seed_rows_for_domain(&coeff, 0, 3, &lo, &hi, None).unwrap();
        let mut state: u64 = 0x5EED;
        let mut u01 = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32) / ((1u64 << 24) as f32)
        };
        for r in 0..3 {
            assert!(
                f.lower_b[r] <= B[r] + 1e-6,
                "row {r} lower_b not folded outward"
            );
            assert!(
                f.upper_b[r] >= B[r] - 1e-6,
                "row {r} upper_b not folded outward"
            );
        }
        for _ in 0..2000 {
            let x = [
                lo[0] + u01() * 2.0,
                lo[1] + u01() * 2.0,
                lo[2] + u01() * 2.0,
            ];
            for r in 0..3 {
                let z: f32 = (0..3).map(|j| W[r][j] * x[j]).sum::<f32>() + B[r];
                let lo_aff: f32 =
                    (0..3).map(|j| f.lower_a[r * 3 + j] * x[j]).sum::<f32>() + f.lower_b[r];
                let hi_aff: f32 =
                    (0..3).map(|j| f.upper_a[r * 3 + j] * x[j]).sum::<f32>() + f.upper_b[r];
                assert!(
                    lo_aff <= z + 1e-5 && z <= hi_aff + 1e-5,
                    "row {r}: folded affine [{lo_aff},{hi_aff}] does NOT enclose true z={z} at x={x:?}"
                );
            }
        }
    }

    #[ntest::timeout(10000)]
    #[test]
    fn resident_error_fold_polls_deadline_inside_coefficient_loop() {
        let dim = 4096usize;
        let coeff = GpuResidentCoeffBatched {
            lower_a: vec![0.25f32; dim],
            upper_a: vec![0.25f32; dim],
            lower_err: vec![f32::from_bits(1); dim],
            upper_err: vec![f32::from_bits(1); dim],
            lower_b: vec![0.0],
            upper_b: vec![0.0],
            lower_b_err: vec![0.0],
            upper_b_err: vec![0.0],
            dim,
            num_specs: 1,
            num_specs_per_dom: 1,
        };
        let lo = vec![-1.0f32; dim];
        let hi = vec![1.0f32; dim];
        assert!(
            fold_seed_rows_for_domain(&coeff, 0, 1, &lo, &hi, None).is_some(),
            "fixture must be an otherwise-valid resident fold"
        );
        assert!(
            fold_seed_rows_for_domain(&coeff, 0, 1, &lo, &hi, Some(Instant::now()),).is_none(),
            "expired production Instant must refuse before fold allocation"
        );

        let mut polls = 0usize;
        let mut expires_in_error_fold = || {
            polls += 1;
            polls >= 16
        };
        let refused = fold_seed_rows_for_domain_with_deadline_check(
            &coeff,
            0,
            1,
            &lo,
            &hi,
            &mut expires_in_error_fold,
        );
        assert!(refused.is_none(), "in-fold expiry must refuse the proposal");
        assert_eq!(polls, 16, "fixture must expire inside the coefficient fold");
    }

    /// A nonzero GenBaB split must use the supplied original history rather
    /// than a lossy history reconstructed from `GraphBetaState`.
    #[ntest::timeout(10000)]
    #[test]
    fn nonzero_beta_split_points_use_original_history_for_both_signs() {
        let lo = [-1.0f32, -1.0, -1.0];
        let hi = [1.0f32, 1.0, 1.0];
        let folded =
            fold_seed_rows_for_domain(&batched_coeff(1, &[0.0], &[0.0]), 0, 3, &lo, &hi, None)
                .unwrap();
        for sign in [-1.0f32, 1.0] {
            let state = GraphBetaState::from_entries(vec![GraphBetaEntry::new(
                "relu_last".into(),
                0,
                0.25,
                0.0,
                sign,
            )
            .unwrap()]);
            let mut exact_history = GraphSplitHistory::new();
            exact_history.add_genbab_constraint(
                GenBabConstraint::new("relu_last".into(), 0, 0.25, sign == 1.0, 0.0).unwrap(),
            );
            assert!(
                clip_seed_domain(
                    &folded,
                    &exact_history,
                    &state,
                    "relu_last",
                    &[0, 1, 2],
                    &[0, 1, 2],
                    &lo,
                    &hi,
                    None,
                )
                .is_some(),
                "nonzero split with sign {sign} must retain its exact original history"
            );
        }

        let zero_relu = beta(&[(0, true)]);
        let zero_history = relu_history(&[(0, true)]);
        assert!(
            clip_seed_domain(
                &folded,
                &zero_history,
                &zero_relu,
                "relu_last",
                &[0, 1, 2],
                &[0, 1, 2],
                &lo,
                &hi,
                Some(Instant::now()),
            )
            .is_none(),
            "expired authority deadline must refuse before proposal allocation"
        );
    }

    /// Exact-real regression for the cancellation that a single final f32 ULP
    /// cannot cover: the first error product is 2^100 and the remaining 4095
    /// products are 1. An ordinary f64 sum loses all small terms, so subtracting
    /// it from a 2^100 bias incorrectly yields about zero instead of -4095.
    #[ntest::timeout(20000)]
    #[test]
    fn fold_bias_rounding_handles_large_small_cancellation_exactly() {
        let dim = 4096usize;
        let huge = 2.0f32.powi(50);
        let huge_sq = 2.0f32.powi(100);
        let mut err = vec![1.0f32; dim];
        err[0] = huge;
        let mut lo = vec![-1.0f32; dim];
        let mut hi = vec![1.0f32; dim];
        lo[0] = -huge;
        hi[0] = huge;
        let coeff = GpuResidentCoeffBatched {
            lower_a: vec![0.0; dim],
            upper_a: vec![0.0; dim],
            lower_err: err.clone(),
            upper_err: err,
            lower_b: vec![huge_sq],
            upper_b: vec![-huge_sq],
            lower_b_err: vec![0.0],
            upper_b_err: vec![0.0],
            dim,
            num_specs: 1,
            num_specs_per_dom: 1,
        };

        let naive_penalty: f64 = coeff
            .lower_err
            .iter()
            .zip(lo.iter().zip(&hi))
            .map(|(&e, (&l, &u))| f64::from(e.abs()) * f64::from(l.abs().max(u.abs())))
            .sum();
        assert_eq!(naive_penalty, f64::from(huge_sq));
        let naive_lower = next_down_f32((f64::from(huge_sq) - naive_penalty) as f32);

        let folded = fold_seed_rows_for_domain(&coeff, 0, 1, &lo, &hi, None).expect("valid fold");
        let exact = BigRational::from_integer((dim as i64 - 1).into());
        let stored_lower = BigRational::from_float(folded.lower_b[0]).expect("finite lower");
        let stored_upper = BigRational::from_float(folded.upper_b[0]).expect("finite upper");
        assert!(
            BigRational::from_float(naive_lower).unwrap() > -exact.clone(),
            "fixture must expose the old one-final-ULP failure"
        );
        assert!(stored_lower <= -exact.clone());
        assert!(stored_upper >= exact);
    }

    /// The FAIL-CLOSED runtime guard itself: it must PASS a sound clip and CATCH a
    /// deliberately too-tight bound (both directions). This directly exercises the
    /// guard retained for research while production clip authority is quarantined.
    #[ntest::timeout(30000)]
    #[test]
    fn clip_guard_catches_too_tight_and_passes_sound_cpu() {
        let g = oracle_net();
        let sel = [0usize, 1, 2];
        let row_of = [0usize, 1, 2];
        let lo = [-1.0f32, -1.0, -1.0];
        let hi = [1.0f32, 1.0, 1.0];
        let prem = vec![(1usize, true)]; // z1 ≥ 0
        let bs = beta(&prem);
        let split_history = relu_history(&prem);
        let coeff = batched_coeff(1, &[0.0], &[0.0]);
        let folded = fold_seed_rows_for_domain(&coeff, 0, 3, &lo, &hi, None).unwrap();
        let (tl, tu) = clip_seed_domain(
            &folded,
            &split_history,
            &bs,
            "relu_last",
            &row_of,
            &sel,
            &lo,
            &hi,
            None,
        )
        .expect("clip");
        let input_box = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[3]), lo.to_vec()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[3]), hi.to_vec()).unwrap(),
        )
        .unwrap();

        // SOUND clip → guard passes.
        assert!(
            clip_guard_verify_domain(
                &g,
                &input_box,
                "gemm_pre",
                "relu_last",
                &bs,
                &sel,
                &folded,
                &tl,
                &tu,
                128,
                0xABC,
            )
            .is_ok(),
            "guard must PASS the genuinely sound clip"
        );

        // Too-tight UPPER on row 0 (below the whole box range) → guard catches it.
        let (blo0, _bhi0) = box_range(0, lo, hi);
        let mut bad_tu = tu.clone();
        bad_tu[0] = blo0 - 1.0;
        assert!(
            clip_guard_verify_domain(
                &g,
                &input_box,
                "gemm_pre",
                "relu_last",
                &bs,
                &sel,
                &folded,
                &tl,
                &bad_tu,
                256,
                0xABC,
            )
            .is_err(),
            "guard must CATCH a too-tight UPPER bound (false-VERIFY)"
        );

        // Too-tight LOWER on row 0 (above the whole box range) → guard catches it.
        let (_blo0b, bhi0) = box_range(0, lo, hi);
        let mut bad_tl = tl;
        bad_tl[0] = bhi0 + 1.0;
        assert!(
            clip_guard_verify_domain(
                &g,
                &input_box,
                "gemm_pre",
                "relu_last",
                &bs,
                &sel,
                &folded,
                &bad_tl,
                &tu,
                256,
                0xABC,
            )
            .is_err(),
            "guard must CATCH a too-tight LOWER bound"
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn bank_capture(
        graph: &GraphNetwork,
        history: &GraphSplitHistory,
        _relu_name: &str,
        seed_node: &str,
        input_lower: &[f32],
        input_upper: &[f32],
        lower_a: Array2<f32>,
        lower_b: Array1<f32>,
        upper_a: Array2<f32>,
        upper_b: Array1<f32>,
        objective: bool,
    ) -> CertifiedLayerCapture {
        let selected: Vec<usize> = (0..lower_a.nrows()).collect();
        let row_of = selected.clone();
        let terminal = LinearBounds::new(
            lower_a.clone(),
            lower_b.clone(),
            upper_a.clone(),
            upper_b.clone(),
        )
        .unwrap();
        let host = capture_host_sound_crown_root_rows(
            graph.cut_fold_scope(),
            input_lower,
            input_upper,
            seed_node,
            &selected,
            &row_of,
            None,
            |_capability| Ok(terminal),
        )
        .unwrap();
        let suggestion = UntrustedCrownAffineRows::new(
            lower_a.clone(),
            upper_a.clone(),
            Array2::zeros(lower_a.raw_dim()),
            Array2::zeros(upper_a.raw_dim()),
            lower_b.clone(),
            upper_b.clone(),
            Array1::zeros(lower_b.len()),
            Array1::zeros(upper_b.len()),
        );
        let root = check_root_affine_dominance_and_seal(
            graph.cut_fold_scope(),
            input_lower,
            input_upper,
            seed_node,
            &selected,
            &row_of,
            &host,
            suggestion,
            None,
        )
        .unwrap();
        let sealed = bind_root_sound_crown_rows_to_history(
            graph.cut_fold_scope(),
            input_lower,
            input_upper,
            history,
            seed_node,
            &selected,
            &row_of,
            &root,
            None,
        )
        .unwrap();
        let (pass, token) = mint_certified_affine_enclosure(sealed, None).unwrap();
        CertifiedLayerCapture {
            seed_node: Arc::from(seed_node),
            selected_neurons: Arc::from(selected.as_slice()),
            row_of_neuron: Arc::from(row_of),
            objective_rows: if objective { selected } else { Vec::new() },
            pass,
            token,
        }
    }

    fn direct_input_graph(width: usize) -> GraphNetwork {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("direct_relu", Layer::ReLU(ReLULayer)));
        graph.add_node(GraphNode::new(
            "out",
            Layer::Linear(
                LinearLayer::new(Array2::eye(width), Some(Array1::zeros(width))).unwrap(),
            ),
            vec!["direct_relu".into()],
        ));
        graph.set_output("out");
        graph
    }

    #[test]
    fn direct_input_premises_capture_exact_identity_for_both_directions() {
        let graph = direct_input_graph(2);
        let mut active = GraphSplitHistory::new();
        active.add_constraint(
            GraphNeuronConstraint::new("direct_relu".into(), 0, true, 0.0).unwrap(),
        );
        let mut inactive = GraphSplitHistory::new();
        inactive.add_constraint(
            GraphNeuronConstraint::new("direct_relu".into(), 1, false, 0.0).unwrap(),
        );
        let active_beta = GraphBetaState::from_entries(vec![GraphBetaEntry::new(
            "direct_relu".into(),
            0,
            0.0,
            0.25,
            1.0,
        )
        .unwrap()]);
        let inactive_beta = GraphBetaState::from_entries(vec![GraphBetaEntry::new(
            "direct_relu".into(),
            1,
            0.0,
            0.5,
            -1.0,
        )
        .unwrap()]);
        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0, -1.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.0]).unwrap(),
        )
        .unwrap();
        let histories = [&active, &inactive];
        let betas = [Some(&active_beta), Some(&inactive_beta)];
        let inputs = [input.clone(), input];
        let mut banks = [DomainClipBank::default(), DomainClipBank::default()];

        initialize_direct_input_sources(
            &graph, &mut banks, &histories, &betas, &inputs, None, true,
        );
        for (domain, history) in histories.iter().enumerate() {
            assert!(!banks[domain].disabled);
            let capture = banks[domain]
                .layers
                .get("direct_relu")
                .expect("direct-input source must be captured");
            assert_eq!(capture.seed_node.as_ref(), NETWORK_INPUT);
            assert_eq!(capture.selected_neurons.as_ref(), [0, 1]);
            assert_eq!(capture.row_of_neuron.as_ref(), [0, 1]);
            let validated = capture
                .token
                .validate_for_clip_in_scope(
                    graph.cut_fold_scope(),
                    &capture.pass,
                    &[-1.0, -1.0],
                    &[1.0, 1.0],
                    history,
                    NETWORK_INPUT,
                    &[0, 1],
                    &[0, 1],
                    None,
                )
                .unwrap();
            assert_eq!(validated.lower_a(), &Array2::<f32>::eye(2));
            assert_eq!(validated.upper_a(), &Array2::<f32>::eye(2));
        }
    }

    #[test]
    fn direct_input_source_refuses_bad_index_and_expired_deadline() {
        let graph = direct_input_graph(1);
        let mut history = GraphSplitHistory::new();
        history.add_constraint(
            GraphNeuronConstraint::new("direct_relu".into(), 3, true, 0.0).unwrap(),
        );
        let beta = GraphBetaState::from_entries(vec![GraphBetaEntry::new(
            "direct_relu".into(),
            3,
            0.0,
            0.0,
            1.0,
        )
        .unwrap()]);
        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![-1.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(),
        )
        .unwrap();
        let mut banks = [DomainClipBank::default()];
        initialize_direct_input_sources(
            &graph,
            &mut banks,
            &[&history],
            &[Some(&beta)],
            std::slice::from_ref(&input),
            None,
            true,
        );
        assert!(banks[0].layers.is_empty());

        let mut expired_banks = [DomainClipBank::default()];
        initialize_direct_input_sources(
            &graph,
            &mut expired_banks,
            &[&history],
            &[Some(&beta)],
            std::slice::from_ref(&input),
            Some(
                Instant::now()
                    .checked_sub(std::time::Duration::from_millis(1))
                    .expect("current instant admits a one-millisecond expired fixture"),
            ),
            true,
        );
        assert!(expired_banks[0].layers.is_empty());
    }

    fn cross_layer_graph() -> GraphNetwork {
        let mut graph = GraphNetwork::new();
        let identity = arr2(&[[1.0f32]]);
        graph.add_node(GraphNode::from_input(
            "source_pre",
            Layer::Linear(LinearLayer::new(identity.clone(), Some(arr1(&[0.0]))).unwrap()),
        ));
        graph.add_node(GraphNode::new(
            "source_relu",
            Layer::ReLU(ReLULayer),
            vec!["source_pre".into()],
        ));
        graph.add_node(GraphNode::from_input(
            "target_pre",
            Layer::Linear(LinearLayer::new(identity, Some(arr1(&[0.0]))).unwrap()),
        ));
        graph.add_node(GraphNode::new(
            "target_relu",
            Layer::ReLU(ReLULayer),
            vec!["target_pre".into()],
        ));
        graph.set_output("target_relu");
        graph
    }

    fn multi_target_cross_layer_graph() -> GraphNetwork {
        let mut graph = cross_layer_graph();
        graph.add_node(GraphNode::from_input(
            "target2_pre",
            Layer::Linear(LinearLayer::new(arr2(&[[-1.0f32]]), Some(arr1(&[0.0]))).unwrap()),
        ));
        graph.add_node(GraphNode::new(
            "target2_relu",
            Layer::ReLU(ReLULayer),
            vec!["target2_pre".into()],
        ));
        graph.set_output("target2_relu");
        graph
    }

    fn multi_target_cross_layer_bank(
        graph: &GraphNetwork,
        history: &GraphSplitHistory,
        input_lower: &[f32],
        input_upper: &[f32],
    ) -> DomainClipBank {
        let source = bank_capture(
            graph,
            history,
            "source_relu",
            "source_pre",
            input_lower,
            input_upper,
            arr2(&[[1.0]]),
            arr1(&[0.0]),
            arr2(&[[1.0]]),
            arr1(&[0.0]),
            false,
        );
        let target = bank_capture(
            graph,
            history,
            "target_relu",
            "target_pre",
            input_lower,
            input_upper,
            arr2(&[[1.0]]),
            arr1(&[0.0]),
            arr2(&[[1.0]]),
            arr1(&[0.0]),
            true,
        );
        let target2 = bank_capture(
            graph,
            history,
            "target2_relu",
            "target2_pre",
            input_lower,
            input_upper,
            arr2(&[[-1.0]]),
            arr1(&[0.0]),
            arr2(&[[-1.0]]),
            arr1(&[0.0]),
            true,
        );
        DomainClipBank {
            layers: HashMap::from([
                ("source_relu".to_string(), source),
                ("target_relu".to_string(), target),
                ("target2_relu".to_string(), target2),
            ]),
            target_order: vec!["target_relu".into(), "target2_relu".into()],
            disabled: false,
        }
    }

    #[test]
    fn domain_batched_targets_are_bit_exact_to_scalar_targets() {
        let graph = multi_target_cross_layer_graph();
        let input_lower = [-1.0f32];
        let input_upper = [1.0f32];
        let mut history = GraphSplitHistory::new();
        history.add_constraint(
            GraphNeuronConstraint::new("source_relu".into(), 0, true, 0.0).unwrap(),
        );
        let bank = multi_target_cross_layer_bank(&graph, &history, &input_lower, &input_upper);

        let scalar: Vec<ClipTargetProposal> = bank
            .target_order
            .iter()
            .map(|target| {
                clip_target_from_bank(
                    &graph,
                    &bank,
                    target,
                    &history,
                    &input_lower,
                    &input_upper,
                    None,
                    true,
                )
                .expect("scalar target proposal")
            })
            .collect();
        let mut never = || false;
        let ClipDomainBatchOutcome::Outcomes(batched) =
            clip_targets_from_bank_batched_checked_with_deadline_check(
                &graph,
                &bank,
                &history,
                &input_lower,
                &input_upper,
                None,
                true,
                &mut never,
            )
        else {
            panic!("two distinct valid targets must use the amortized path");
        };
        assert_eq!(batched.len(), scalar.len());
        for (batched, scalar) in batched.into_iter().zip(scalar) {
            let ClipTargetOutcome::Proposal(batched) = batched else {
                panic!("valid batched target must produce a proposal");
            };
            assert_eq!(batched.0, scalar.0);
            assert_eq!(batched.1, scalar.1);
            assert_eq!(
                batched.2.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                scalar.2.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            assert_eq!(
                batched.3.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                scalar.3.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn domain_batched_targets_late_round_use_first_network_split_layer() {
        let graph = multi_target_cross_layer_graph();
        let input_lower = [-1.0f32];
        let input_upper = [1.0f32];
        let mut history = GraphSplitHistory::new();
        history.add_constraint(
            GraphNeuronConstraint::new("source_relu".into(), 0, true, 0.0).unwrap(),
        );
        history.add_constraint(
            GraphNeuronConstraint::new("target2_relu".into(), 0, true, 0.0).unwrap(),
        );
        let bank = multi_target_cross_layer_bank(&graph, &history, &input_lower, &input_upper);

        let mut never = || false;
        let ClipDomainBatchOutcome::Outcomes(outcomes) =
            clip_targets_from_bank_batched_checked_with_deadline_check(
                &graph,
                &bank,
                &history,
                &input_lower,
                &input_upper,
                None,
                false,
                &mut never,
            )
        else {
            panic!("two valid targets must use the amortized path");
        };
        assert_eq!(outcomes.len(), 2);
        let ClipTargetOutcome::Proposal(target) = &outcomes[0] else {
            panic!("first target must produce a proposal");
        };
        let ClipTargetOutcome::Proposal(target2) = &outcomes[1] else {
            panic!("second target must produce a proposal");
        };

        // source_relu is the first split layer in network order and constrains
        // x >= 0. The global history tail is target2_relu, whose active side
        // would instead constrain -x >= 0. These bounds distinguish the two
        // schedules while exercising the shared two-target solver.
        assert!(target.2[0] >= -1e-6);
        assert!(target.2[0] <= 1e-6);
        assert!(target2.3[0] >= -1e-6);
        assert!(target2.3[0] <= 1e-6);
    }

    #[test]
    fn cross_layer_source_tightens_target_in_both_directions() {
        let graph = cross_layer_graph();
        let input_lower = [-1.0f32];
        let input_upper = [1.0f32];
        for active in [true, false] {
            let mut history = GraphSplitHistory::new();
            history.add_constraint(
                GraphNeuronConstraint::new("source_relu".into(), 0, active, 0.0).unwrap(),
            );
            let source = bank_capture(
                &graph,
                &history,
                "source_relu",
                "source_pre",
                &input_lower,
                &input_upper,
                arr2(&[[1.0]]),
                arr1(&[0.0]),
                arr2(&[[1.0]]),
                arr1(&[0.0]),
                false,
            );
            let target = bank_capture(
                &graph,
                &history,
                "target_relu",
                "target_pre",
                &input_lower,
                &input_upper,
                arr2(&[[1.0]]),
                arr1(&[0.0]),
                arr2(&[[1.0]]),
                arr1(&[0.0]),
                true,
            );
            let bank = DomainClipBank {
                layers: HashMap::from([
                    ("source_relu".to_string(), source),
                    ("target_relu".to_string(), target),
                ]),
                target_order: vec!["target_relu".into()],
                disabled: false,
            };
            let (_, neurons, lower, upper) = clip_target_from_bank(
                &graph,
                &bank,
                "target_relu",
                &history,
                &input_lower,
                &input_upper,
                None,
                true,
            )
            .expect("cross-layer clip");
            assert_eq!(neurons, vec![0]);
            if active {
                assert!(lower[0] >= -1e-6, "active x>=0 must tighten lower");
                assert!(upper[0] <= 1.0 + 1e-6);
            } else {
                assert!(upper[0] <= 1e-6, "inactive x<=0 must tighten upper");
                assert!(lower[0] >= -1.0 - 1e-6);
            }
        }
    }

    #[test]
    fn root_bank_only_production_pass_tightens_without_child_backward() {
        let graph = cross_layer_graph();
        let verifier = BetaCrownVerifier::default();
        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![-1.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(),
        )
        .unwrap();
        let root_bounds: HashMap<String, Arc<BoundedTensor>> = graph
            .collect_node_bounds(&input)
            .unwrap()
            .into_iter()
            .map(|(name, bounds)| (name, Arc::new(bounds)))
            .collect();
        assert!(verifier.complete_clip_root_bounds_cache.store_finalized(
            &graph,
            &input,
            &root_bounds
        ));

        let mut history = GraphSplitHistory::new();
        history.add_constraint(
            GraphNeuronConstraint::new("source_relu".into(), 0, true, 0.0).unwrap(),
        );
        let beta = GraphBetaState::from_entries(vec![GraphBetaEntry::new(
            "source_relu".into(),
            0,
            0.0,
            0.25,
            1.0,
        )
        .unwrap()]);
        let mut caches = [root_bounds.clone()];
        let inputs = [input];
        let histories = [&history];
        let betas = [Some(&beta)];
        let alphas = [None];
        let mut banks = [DomainClipBank::default()];
        let mut options = IntermRefineOptions::from_env();
        options.clip_resnet = true;
        options.selective_topk = 20;
        options.probe = false;

        verifier
            .capture_complete_clip_root_bank_layer(
                &graph,
                "source_relu",
                "source_pre",
                &caches,
                &inputs,
                &histories,
                &betas,
                &alphas,
                &root_bounds,
                &mut banks,
                &options,
                false,
                true,
                &[None],
            )
            .expect("source capture");
        verifier
            .capture_complete_clip_root_bank_layer(
                &graph,
                "target_relu",
                "target_pre",
                &caches,
                &inputs,
                &histories,
                &betas,
                &alphas,
                &root_bounds,
                &mut banks,
                &options,
                true,
                true,
                &[None],
            )
            .expect("target capture");
        let mut stats = RefineStats::default();
        let outcome = reclip_all_captured_targets(
            &graph,
            &banks,
            &histories,
            &inputs,
            &mut caches,
            None,
            &mut stats,
            true,
        );
        assert!(outcome.completed);
        assert!(outcome.tightened);
        assert!(
            caches[0]["target_pre"].lower()[[0]] >= -1e-6,
            "source x>=0 must tighten the separate target lower bound"
        );
    }

    #[test]
    fn missing_cross_layer_source_refuses_instead_of_partial_clip() {
        let graph = cross_layer_graph();
        let input_lower = [-1.0f32];
        let input_upper = [1.0f32];
        let mut history = GraphSplitHistory::new();
        history.add_constraint(
            GraphNeuronConstraint::new("source_relu".into(), 0, true, 0.0).unwrap(),
        );
        let target = bank_capture(
            &graph,
            &history,
            "target_relu",
            "target_pre",
            &input_lower,
            &input_upper,
            arr2(&[[1.0]]),
            arr1(&[0.0]),
            arr2(&[[1.0]]),
            arr1(&[0.0]),
            true,
        );
        let bank = DomainClipBank {
            layers: HashMap::from([("target_relu".to_string(), target)]),
            target_order: vec!["target_relu".into()],
            disabled: false,
        };
        assert!(
            clip_target_from_bank(
                &graph,
                &bank,
                "target_relu",
                &history,
                &input_lower,
                &input_upper,
                None,
                true,
            )
            .is_none(),
            "an incomplete all-history bank must fail closed"
        );

        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1]), input_lower.to_vec()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[1]), input_upper.to_vec()).unwrap(),
        )
        .unwrap();
        let pre = Arc::new(
            BoundedTensor::new(
                ArrayD::from_shape_vec(IxDyn(&[1]), input_lower.to_vec()).unwrap(),
                ArrayD::from_shape_vec(IxDyn(&[1]), input_upper.to_vec()).unwrap(),
            )
            .unwrap(),
        );
        let post = Arc::new(
            BoundedTensor::new(
                ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap(),
                ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(),
            )
            .unwrap(),
        );
        let mut caches = [HashMap::from([
            ("target_pre".to_string(), Arc::clone(&pre)),
            ("target_relu".to_string(), Arc::clone(&post)),
        ])];
        let mut stats = RefineStats::default();
        let outcome = reclip_all_captured_targets(
            &graph,
            &[bank],
            &[&history],
            &[input],
            &mut caches,
            None,
            &mut stats,
            true,
        );
        assert!(!outcome.completed);
        assert_eq!(outcome.targets_expected, 1);
        assert_eq!(outcome.targets_completed, 0);
        assert_eq!(outcome.targets_refused, 1);
        assert!(!outcome.deadline_interrupted);
        assert!(Arc::ptr_eq(&caches[0]["target_pre"], &pre));
        assert!(Arc::ptr_eq(&caches[0]["target_relu"], &post));
    }

    #[test]
    fn present_source_layer_with_missing_neuron_row_refuses() {
        let mut graph = GraphNetwork::new();
        let identity = Array2::eye(2);
        for (pre, relu) in [("source_pre", "source_relu"), ("target_pre", "target_relu")] {
            graph.add_node(GraphNode::from_input(
                pre,
                Layer::Linear(LinearLayer::new(identity.clone(), Some(Array1::zeros(2))).unwrap()),
            ));
            graph.add_node(GraphNode::new(
                relu,
                Layer::ReLU(ReLULayer),
                vec![pre.into()],
            ));
        }
        graph.set_output("target_relu");
        let input_lower = [-1.0f32, -1.0];
        let input_upper = [1.0f32, 1.0];
        let mut history = GraphSplitHistory::new();
        history.add_constraint(
            GraphNeuronConstraint::new("source_relu".into(), 1, true, 0.0).unwrap(),
        );
        // Capture only source neuron 0: the layer exists, but the exact
        // history's neuron 1 must not be silently dropped.
        let source = bank_capture(
            &graph,
            &history,
            "source_relu",
            "source_pre",
            &input_lower,
            &input_upper,
            arr2(&[[1.0, 0.0]]),
            arr1(&[0.0]),
            arr2(&[[1.0, 0.0]]),
            arr1(&[0.0]),
            false,
        );
        let target = bank_capture(
            &graph,
            &history,
            "target_relu",
            "target_pre",
            &input_lower,
            &input_upper,
            identity.clone(),
            Array1::zeros(2),
            identity,
            Array1::zeros(2),
            true,
        );
        let bank = DomainClipBank {
            layers: HashMap::from([
                ("source_relu".to_string(), source),
                ("target_relu".to_string(), target),
            ]),
            target_order: vec!["target_relu".into()],
            disabled: false,
        };
        assert!(clip_target_from_bank(
            &graph,
            &bank,
            "target_relu",
            &history,
            &input_lower,
            &input_upper,
            None,
            true,
        )
        .is_none());

        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2]), input_lower.to_vec()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2]), input_upper.to_vec()).unwrap(),
        )
        .unwrap();
        let pre = Arc::new(
            BoundedTensor::new(
                ArrayD::from_shape_vec(IxDyn(&[2]), input_lower.to_vec()).unwrap(),
                ArrayD::from_shape_vec(IxDyn(&[2]), input_upper.to_vec()).unwrap(),
            )
            .unwrap(),
        );
        let post = Arc::new(
            BoundedTensor::new(
                ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 0.0]).unwrap(),
                ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.0]).unwrap(),
            )
            .unwrap(),
        );
        let mut caches = [HashMap::from([
            ("target_pre".to_string(), Arc::clone(&pre)),
            ("target_relu".to_string(), Arc::clone(&post)),
        ])];
        let mut stats = RefineStats::default();
        let outcome = reclip_all_captured_targets(
            &graph,
            &[bank],
            &[&history],
            &[input],
            &mut caches,
            None,
            &mut stats,
            true,
        );
        assert!(!outcome.completed);
        assert_eq!(outcome.targets_expected, 1);
        assert_eq!(outcome.targets_completed, 0);
        assert_eq!(outcome.targets_refused, 1);
        assert!(Arc::ptr_eq(&caches[0]["target_pre"], &pre));
        assert!(Arc::ptr_eq(&caches[0]["target_relu"], &post));
    }

    #[test]
    fn zero_target_and_disabled_banks_are_not_completed_zero_yield() {
        let graph = cross_layer_graph();
        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![-1.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(),
        )
        .unwrap();
        let empty_history = GraphSplitHistory::new();
        let mut empty_cache = [HashMap::new()];
        let mut stats = RefineStats::default();
        let empty = reclip_all_captured_targets(
            &graph,
            &[DomainClipBank::default()],
            &[&empty_history],
            std::slice::from_ref(&input),
            &mut empty_cache,
            None,
            &mut stats,
            true,
        );
        assert!(!empty.completed);
        assert_eq!(empty.targets_expected, 0);

        let mut history = GraphSplitHistory::new();
        history.add_constraint(
            GraphNeuronConstraint::new("source_relu".into(), 0, true, 0.0).unwrap(),
        );
        let disabled = DomainClipBank {
            layers: HashMap::new(),
            target_order: vec!["target_relu".into()],
            disabled: true,
        };
        let mut disabled_cache = [HashMap::new()];
        let disabled_outcome = reclip_all_captured_targets(
            &graph,
            &[disabled],
            &[&history],
            &[input],
            &mut disabled_cache,
            None,
            &mut stats,
            true,
        );
        assert!(!disabled_outcome.completed);
        assert_eq!(disabled_outcome.targets_expected, 1);
        assert_eq!(disabled_outcome.targets_completed, 0);
        assert_eq!(disabled_outcome.targets_refused, 1);
    }

    #[test]
    fn fully_covered_is_noop_but_numerical_infeasibility_is_refused() {
        let graph = cross_layer_graph();
        for (input_lower, input_upper, expect_completed) in
            [([0.5f32], [1.0f32], true), ([-1.0f32], [-0.5f32], false)]
        {
            let mut history = GraphSplitHistory::new();
            history.add_constraint(
                GraphNeuronConstraint::new("source_relu".into(), 0, true, 0.0).unwrap(),
            );
            let source = bank_capture(
                &graph,
                &history,
                "source_relu",
                "source_pre",
                &input_lower,
                &input_upper,
                arr2(&[[1.0]]),
                arr1(&[0.0]),
                arr2(&[[1.0]]),
                arr1(&[0.0]),
                false,
            );
            let target = bank_capture(
                &graph,
                &history,
                "target_relu",
                "target_pre",
                &input_lower,
                &input_upper,
                arr2(&[[1.0]]),
                arr1(&[0.0]),
                arr2(&[[1.0]]),
                arr1(&[0.0]),
                true,
            );
            let bank = DomainClipBank {
                layers: HashMap::from([
                    ("source_relu".to_string(), source),
                    ("target_relu".to_string(), target),
                ]),
                target_order: vec!["target_relu".into()],
                disabled: false,
            };
            let input = BoundedTensor::new(
                ArrayD::from_shape_vec(IxDyn(&[1]), input_lower.to_vec()).unwrap(),
                ArrayD::from_shape_vec(IxDyn(&[1]), input_upper.to_vec()).unwrap(),
            )
            .unwrap();
            let pre = Arc::new(
                BoundedTensor::new(
                    ArrayD::from_shape_vec(IxDyn(&[1]), input_lower.to_vec()).unwrap(),
                    ArrayD::from_shape_vec(IxDyn(&[1]), input_upper.to_vec()).unwrap(),
                )
                .unwrap(),
            );
            let post = Arc::new(
                BoundedTensor::new(
                    ArrayD::from_shape_vec(IxDyn(&[1]), vec![input_lower[0].max(0.0)]).unwrap(),
                    ArrayD::from_shape_vec(IxDyn(&[1]), vec![input_upper[0].max(0.0)]).unwrap(),
                )
                .unwrap(),
            );
            let mut caches = [HashMap::from([
                ("target_pre".to_string(), Arc::clone(&pre)),
                ("target_relu".to_string(), Arc::clone(&post)),
            ])];
            let mut stats = RefineStats::default();
            let outcome = reclip_all_captured_targets(
                &graph,
                &[bank],
                &[&history],
                &[input],
                &mut caches,
                None,
                &mut stats,
                true,
            );
            assert_eq!(outcome.completed, expect_completed);
            assert_eq!(outcome.targets_expected, 1);
            assert_eq!(outcome.targets_completed, usize::from(expect_completed));
            assert_eq!(outcome.targets_refused, usize::from(!expect_completed));
            assert_eq!(stats.infeasible, 0);
            assert!(Arc::ptr_eq(&caches[0]["target_pre"], &pre));
            assert!(Arc::ptr_eq(&caches[0]["target_relu"], &post));
        }
    }

    #[test]
    fn expired_bank_reclip_leaves_cache_byte_identical() {
        let graph = cross_layer_graph();
        let input_lower = [-1.0f32];
        let input_upper = [1.0f32];
        let mut history = GraphSplitHistory::new();
        history.add_constraint(
            GraphNeuronConstraint::new("source_relu".into(), 0, true, 0.0).unwrap(),
        );
        let source = bank_capture(
            &graph,
            &history,
            "source_relu",
            "source_pre",
            &input_lower,
            &input_upper,
            arr2(&[[1.0]]),
            arr1(&[0.0]),
            arr2(&[[1.0]]),
            arr1(&[0.0]),
            false,
        );
        let target = bank_capture(
            &graph,
            &history,
            "target_relu",
            "target_pre",
            &input_lower,
            &input_upper,
            arr2(&[[1.0]]),
            arr1(&[0.0]),
            arr2(&[[1.0]]),
            arr1(&[0.0]),
            true,
        );
        let bank = DomainClipBank {
            layers: HashMap::from([
                ("source_relu".to_string(), source),
                ("target_relu".to_string(), target),
            ]),
            target_order: vec!["target_relu".into()],
            disabled: false,
        };
        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1]), input_lower.to_vec()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[1]), input_upper.to_vec()).unwrap(),
        )
        .unwrap();
        let pre = Arc::new(
            BoundedTensor::new(
                ArrayD::from_shape_vec(IxDyn(&[1]), vec![-1.0]).unwrap(),
                ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(),
            )
            .unwrap(),
        );
        let post = Arc::new(
            BoundedTensor::new(
                ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap(),
                ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(),
            )
            .unwrap(),
        );
        let mut caches = [HashMap::from([
            ("target_pre".to_string(), Arc::clone(&pre)),
            ("target_relu".to_string(), Arc::clone(&post)),
        ])];
        let mut stats = RefineStats::default();
        let outcome = reclip_all_captured_targets(
            &graph,
            &[bank],
            &[&history],
            &[input],
            &mut caches,
            Some(
                Instant::now()
                    .checked_sub(std::time::Duration::from_millis(1))
                    .expect("current instant admits a one-millisecond expired fixture"),
            ),
            &mut stats,
            true,
        );
        assert!(!outcome.tightened);
        assert!(!outcome.completed);
        assert!(outcome.deadline_interrupted);
        assert_eq!(outcome.targets_expected, 1);
        assert_eq!(outcome.targets_completed, 0);
        assert!(Arc::ptr_eq(&caches[0]["target_pre"], &pre));
        assert!(Arc::ptr_eq(&caches[0]["target_relu"], &post));
        assert_eq!(stats.neurons_tightened, 0);
    }

    #[test]
    fn deadline_inside_last_target_keeps_partial_tightening_but_is_incomplete() {
        let graph = cross_layer_graph();
        let input_lower = [-1.0f32];
        let input_upper = [1.0f32];
        let mut history = GraphSplitHistory::new();
        history.add_constraint(
            GraphNeuronConstraint::new("source_relu".into(), 0, true, 0.0).unwrap(),
        );
        let make_bank = |target_count: usize| {
            let source = bank_capture(
                &graph,
                &history,
                "source_relu",
                "source_pre",
                &input_lower,
                &input_upper,
                arr2(&[[1.0]]),
                arr1(&[0.0]),
                arr2(&[[1.0]]),
                arr1(&[0.0]),
                false,
            );
            let target = bank_capture(
                &graph,
                &history,
                "target_relu",
                "target_pre",
                &input_lower,
                &input_upper,
                arr2(&[[1.0]]),
                arr1(&[0.0]),
                arr2(&[[1.0]]),
                arr1(&[0.0]),
                true,
            );
            DomainClipBank {
                layers: HashMap::from([
                    ("source_relu".to_string(), source),
                    ("target_relu".to_string(), target),
                ]),
                target_order: std::iter::repeat_n("target_relu".to_string(), target_count)
                    .collect(),
                disabled: false,
            }
        };
        let make_input = || {
            BoundedTensor::new(
                ArrayD::from_shape_vec(IxDyn(&[1]), input_lower.to_vec()).unwrap(),
                ArrayD::from_shape_vec(IxDyn(&[1]), input_upper.to_vec()).unwrap(),
            )
            .unwrap()
        };
        let make_cache = || {
            HashMap::from([
                (
                    "target_pre".to_string(),
                    Arc::new(
                        BoundedTensor::new(
                            ArrayD::from_shape_vec(IxDyn(&[1]), vec![-1.0]).unwrap(),
                            ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(),
                        )
                        .unwrap(),
                    ),
                ),
                (
                    "target_relu".to_string(),
                    Arc::new(
                        BoundedTensor::new(
                            ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap(),
                            ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(),
                        )
                        .unwrap(),
                    ),
                ),
            ])
        };

        // Measure the exact number of injected deadline polls through one
        // complete target, then expire on the first poll of a second target.
        let mut baseline_polls = 0usize;
        let mut never = || {
            baseline_polls += 1;
            false
        };
        let mut baseline_cache = [make_cache()];
        let mut baseline_stats = RefineStats::default();
        let baseline = reclip_all_captured_targets_with_deadline_check(
            &graph,
            &[make_bank(1)],
            &[&history],
            &[make_input()],
            &mut baseline_cache,
            None,
            &mut baseline_stats,
            true,
            &mut never,
        );
        assert!(baseline.completed);
        assert!(baseline.tightened);

        let mut polls = 0usize;
        let mut expire_after_first = || {
            polls += 1;
            polls > baseline_polls
        };
        let mut caches = [make_cache()];
        let mut stats = RefineStats::default();
        let outcome = reclip_all_captured_targets_with_deadline_check(
            &graph,
            &[make_bank(2)],
            &[&history],
            &[make_input()],
            &mut caches,
            None,
            &mut stats,
            true,
            &mut expire_after_first,
        );
        assert!(outcome.tightened);
        assert!(!outcome.completed);
        assert!(outcome.deadline_interrupted);
        assert_eq!(outcome.targets_expected, 2);
        assert_eq!(outcome.targets_completed, 1);
        assert_eq!(outcome.targets_refused, 0);
        assert!(caches[0]["target_pre"].lower()[[0]] >= -1e-6);
    }

    #[test]
    fn later_source_enables_reclip_of_earlier_target_with_full_history() {
        let graph = cross_layer_graph();
        let input_lower = [-1.0f32, -1.0];
        let input_upper = [1.0f32, 1.0];
        let mut history = GraphSplitHistory::new();
        history.add_constraint(
            GraphNeuronConstraint::new("source_relu".into(), 0, true, 0.0).unwrap(),
        );
        history
            .add_constraint(GraphNeuronConstraint::new("late_relu".into(), 0, true, 0.0).unwrap());
        let source = bank_capture(
            &graph,
            &history,
            "source_relu",
            "source_pre",
            &input_lower,
            &input_upper,
            arr2(&[[1.0, 0.0]]),
            arr1(&[0.0]),
            arr2(&[[1.0, 0.0]]),
            arr1(&[0.0]),
            false,
        );
        let late = bank_capture(
            &graph,
            &history,
            "late_relu",
            "late_pre",
            &input_lower,
            &input_upper,
            arr2(&[[0.0, 1.0]]),
            arr1(&[0.0]),
            arr2(&[[0.0, 1.0]]),
            arr1(&[0.0]),
            false,
        );
        let target = bank_capture(
            &graph,
            &history,
            "target_relu",
            "target_pre",
            &input_lower,
            &input_upper,
            arr2(&[[1.0, 0.0]]),
            arr1(&[0.0]),
            arr2(&[[1.0, 0.0]]),
            arr1(&[0.0]),
            true,
        );
        let mut bank = DomainClipBank {
            layers: HashMap::from([
                ("source_relu".to_string(), source),
                ("target_relu".to_string(), target),
            ]),
            target_order: vec!["target_relu".into()],
            disabled: false,
        };
        assert!(
            clip_target_from_bank(
                &graph,
                &bank,
                "target_relu",
                &history,
                &input_lower,
                &input_upper,
                None,
                true,
            )
            .is_none(),
            "the earlier target waits for the later history source"
        );
        bank.layers.insert("late_relu".into(), late);
        let (_, _, lower, upper) = clip_target_from_bank(
            &graph,
            &bank,
            "target_relu",
            &history,
            &input_lower,
            &input_upper,
            None,
            true,
        )
        .expect("full bank reclips earlier target");
        assert!(lower[0] >= -1e-6);
        assert!(upper[0] <= 1.0 + 1e-6);

        // Match DomainClipper's post-warmup schedule: the first nonempty split
        // layer in network order wins, even though its split is not the global
        // history tail. Every row token remains validated against the exact
        // full (x>=0, y>=0) child history.
        bank.layers.remove("late_relu");
        let (_, _, scheduled_lower, _) = clip_target_from_bank(
            &graph,
            &bank,
            "target_relu",
            &history,
            &input_lower,
            &input_upper,
            None,
            false,
        )
        .expect("late-round schedule needs the first split-layer source row");
        assert!(scheduled_lower[0] >= -1e-6);
        assert!(scheduled_lower[0] <= 1e-6);
    }
}

// (free fns at module scope — all private helpers are file-visible here).

/// Root JOINT per-target intermediate-bound α pass (#root-joint-interm-alpha).
///
/// auto_LiRPA's `fix_intermediate_layer_bounds=False` root pass, scoped: for each
/// target pre-activation node L in `targets` (deepest-first), build the truncated
/// L→input resnet stack, seed identity over L's CROSSING rows (`num_specs = n_sel`),
/// and Adam-ascend the below-L α against the SUMMED lower-bound objective
/// `Σ_r lb(L_r)` whose joint gradient flows THROUGH L's own bound computation via
/// the on-device adjoint (`crown_joint_alpha_gradient_resident`). Every iterate is
/// scored with the certified sound fold (`crown_backward_gpu_resnet_sound_beta`);
/// the element-wise best `[l',u']` across iterates is intersected SHRINK-ONLY into
/// `bounds[L]` (`intersection_per_element`, union fallback per element — the
/// alpha_explicit contract), so the frozen root tree can only tighten, never widen.
///
/// SOUNDNESS: any α∈[0,1] is a valid ReLU lower slope ⇒ every scored fold is a
/// valid enclosure of L's pre-activation over the root box; the element-wise best
/// across valid enclosures is a valid enclosure (intersection); the final
/// intersect-with-reference is shrink-only with per-element union fallback. On any
/// refusal (no sound GPU, prep failure, shape mismatch, deadline) the target keeps
/// its sound reference bound. Dark: only reachable under
/// `NY_ROOT_JOINT_INTERM_ALPHA=1` (root.rs gate); finite-slice ascent additionally
/// requires the typed `NY_ROOT_JOINT_INTERM_ALPHA_DEADLINE_ASCENT=1` policy.
/// Otherwise a bounded invocation remains base-fold-only. Default byte-identical.
///
/// Returns the number of targets whose bound strictly tightened.
#[allow(clippy::too_many_arguments)]
pub(in crate::beta_crown::engine::graph) fn root_joint_tighten_relu_preactivations(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    targets: &[String],
    engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
    iters: usize,
    lr: f32,
    bounds: &mut HashMap<String, BoundedTensor>,
) -> usize {
    root_joint_tighten_relu_preactivations_weighted(
        graph, input, targets, engine, deadline, iters, lr, None, bounds,
    )
}

/// #joint-interm-grad: the OBJECTIVE-WEIGHTED joint tightening entry.
///
/// # What the weights change
///
/// Unweighted (`objective_sensitivity == None`, the historical behaviour), this
/// lane seeds a unit diagonal and so ascends alpha to tighten each intermediate
/// target **for its own sake** — every neuron of every target counts equally,
/// including neurons the final bound barely reads.
///
/// With weights it ascends alpha to tighten each intermediate **in proportion to
/// how much the final objective actually depends on it**. That is the indirect
/// gradient term
///
/// ```text
///     df/dalpha_k  +=  SUM_i  df/dl_m[i] * dl_m[i]/dalpha_k
/// ```
///
/// obtained in one existing device call rather than a new kernel: the per-row
/// harvest is degree-1 positive-homogeneous (branch selections depend only on
/// coefficient SIGNS), so seeding row `r` with `w[j] >= 0` makes the device's
/// sum-over-rows return `SUM_j w_j * dl_j/dalpha_k` exactly. The row reduction
/// that normally blocks extracting per-row sensitivities is precisely what
/// performs the contraction.
///
/// # Contract on the weights
///
/// `objective_sensitivity` maps target node name -> per-neuron `df/dl`, as
/// produced by `GraphNetwork::interm_sensitivity_weights`, which returns
/// non-negative values (verified against central finite differences, along with
/// the fact that the naive slope-only form is SIGN-FLIPPED on the `l` axis and
/// would steer the ascent backwards). Non-negativity is what licenses the
/// homogeneity argument; entries that are negative or non-finite fail closed to
/// `1.0` at the seed site, degrading to the historical unit behaviour for that
/// row rather than emitting a meaningless sensitivity.
///
/// A target with no entry keeps the unit diagonal, so a partial map is safe.
///
/// # Soundness
///
/// None at risk. The weights steer WHICH alpha the ascent lands on; every alpha
/// it can land on is clamped to `[0,1]` and is a certified-sound relaxation
/// (`alpha_sound_regardless`), and the bounds this lane publishes still go
/// through the same shrink-only intersect. Wrong weights cost tightness, never
/// validity.
#[allow(clippy::too_many_arguments)]
pub(in crate::beta_crown::engine::graph) fn root_joint_tighten_relu_preactivations_weighted(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    targets: &[String],
    engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
    iters: usize,
    lr: f32,
    objective_sensitivity: Option<&HashMap<String, Vec<f32>>>,
    bounds: &mut HashMap<String, BoundedTensor>,
) -> usize {
    let gpu = engine
        .and_then(|engine| engine.as_gpu_crown_backward())
        .filter(|gpu| gpu.provides_sound_gpu_crown());
    let mut options = root_joint_tighten_options(iters, lr);
    options.objective_sensitivity = objective_sensitivity;
    root_tighten_relu_preactivations_with_options(
        graph,
        input,
        targets.iter().cloned().map(|target| (target, None)),
        gpu,
        deadline,
        options,
        bounds,
    )
}

/// Call-local finite-deadline root-joint entry for the sanctioned CUDA engine.
///
/// Unlike the legacy research entry above, this accepts only the typed
/// capability constructed by `sound_gpu_gate` from NY's deadline-safe sound-f64
/// factory. It therefore cannot consult or fall back to the ordinary WGPU
/// propagation engine. Requests wider than the engine's audited bounded sound-
/// fold capacity retain the existing per-target refusal behavior.
#[allow(clippy::too_many_arguments)]
pub(in crate::beta_crown::engine::graph) fn root_joint_tighten_relu_preactivations_with_deadline_gpu(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    targets: &[String],
    gpu: crate::sound_gpu_gate::RootJointDeadlineGpu<'_>,
    deadline: Instant,
    iters: usize,
    lr: f32,
    // #joint-interm-grad: per-target `df/dl` weights. `None` = the historical
    // unit seed, byte for byte. See `root_joint_tighten_relu_preactivations_weighted`.
    objective_sensitivity: Option<&HashMap<String, Vec<f32>>>,
    bounds: &mut HashMap<String, BoundedTensor>,
) -> usize {
    debug_assert!(
        (1..=ny_core::DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS).contains(&gpu.sound_fold_max_rows())
    );
    root_tighten_relu_preactivations_with_options(
        graph,
        input,
        targets.iter().cloned().map(|target| (target, None)),
        Some(gpu.backend()),
        Some(deadline),
        {
            let mut options = root_joint_tighten_options(iters, lr);
            options.objective_sensitivity = objective_sensitivity;
            options
        },
        bounds,
    )
}

fn root_joint_tighten_options(iters: usize, lr: f32) -> RootIntermTightenOptions<'static> {
    RootIntermTightenOptions {
        objective_sensitivity: None,
        iters,
        lr,
        max_sel: std::env::var("NY_ROOT_JOINT_INTERM_ALPHA_MAX_SEL")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(512),
        probe: std::env::var("NY_ROOT_JOINT_INTERM_ALPHA_PROBE")
            .ok()
            .as_deref()
            == Some("1"),
        frozen_stop: std::env::var("NY_ROOT_JOINT_INTERM_ALPHA_FROZEN_STOP")
            .ok()
            .as_deref()
            == Some("1"),
        allow_bn: std::env::var("NY_BN_GPU_EXTRACT").ok().as_deref() == Some("1"),
        allow_pure_chain: std::env::var("NY_ROOT_JOINT_INTERM_ALPHA_PURE_CHAIN")
            .ok()
            .as_deref()
            == Some("1"),
        deadline_ascent: RootJointDeadlineAscentPolicy::from_env(),
        log_tag: "root-joint-interm-alpha",
    }
}

/// Production-shaped base CROWN fold for sparse crossing rows at the root.
///
/// This deliberately performs ZERO gradient/ascent iterations and admits none
/// of the experimental frozen-stop, BatchNorm-extraction, or pure-chain seams.
/// Every option is supplied by the typed caller, so unrelated research
/// environment variables cannot expand this pass's authority or resource use.
#[allow(clippy::too_many_arguments)]
pub(in crate::beta_crown::engine::graph) fn root_sparse_tighten_relu_preactivations(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    targets: &[String],
    engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
    max_sel: usize,
    bounds: &mut HashMap<String, BoundedTensor>,
) -> usize {
    root_base_tighten_relu_preactivations(
        graph,
        input,
        targets.iter().cloned().map(|target| (target, None)),
        engine,
        deadline,
        max_sel,
        false,
        "root-sparse-interm-crown",
        bounds,
    )
}

/// One ranked wide target together with the exact already-prepared resident
/// suffix. Preparation is consumed by execution, so the synchronous extractor
/// runs at most once per candidate and cannot silently double the phase wall.
pub(in crate::beta_crown::engine::graph) struct PreparedWideDemandedTarget {
    pub name: String,
    prep: ResnetDomainPrep,
}

/// Diagnostic-first face of the same zero-iteration certified base fold used
/// by [`root_sparse_tighten_relu_preactivations`]. It changes no authority or
/// arithmetic: the only difference is retaining actionable refusal receipts
/// for the default-dark one-target wide lane.
#[allow(clippy::too_many_arguments)]
pub(in crate::beta_crown::engine::graph) fn root_wide_demanded_tighten_relu_preactivations(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    targets: Vec<PreparedWideDemandedTarget>,
    engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
    max_sel: usize,
    bounds: &mut HashMap<String, BoundedTensor>,
) -> usize {
    root_base_tighten_relu_preactivations(
        graph,
        input,
        targets
            .into_iter()
            .map(|target| (target.name, Some(target.prep))),
        engine,
        deadline,
        max_sel,
        true,
        "root-wide-demanded-interm-crown",
        bounds,
    )
}

#[allow(clippy::too_many_arguments)]
fn root_base_tighten_relu_preactivations(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    targets: impl IntoIterator<Item = (String, Option<ResnetDomainPrep>)>,
    engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
    max_sel: usize,
    probe: bool,
    log_tag: &'static str,
    bounds: &mut HashMap<String, BoundedTensor>,
) -> usize {
    let gpu = engine
        .and_then(|engine| engine.as_gpu_crown_backward())
        .filter(|gpu| gpu.provides_sound_gpu_crown());
    let (effective_max_sel, deadline_route) =
        gpu.map_or((max_sel, RootSparseDeadlineRoute::Unavailable), |gpu| {
            root_sparse_effective_max_sel(
                max_sel,
                deadline.is_some(),
                gpu.honors_crown_backward_deadline(),
                gpu.deadline_bounded_resnet_sound_max_rows(),
            )
        });
    if gpu.is_some() {
        eprintln!(
            "[{log_tag}] requested_max_rows={max_sel} \
             effective_max_rows={effective_max_sel} deadline_route={}",
            deadline_route.label()
        );
    }
    let options = RootIntermTightenOptions {
        objective_sensitivity: None,
        iters: 0,
        lr: 0.0,
        max_sel: effective_max_sel,
        probe,
        frozen_stop: false,
        allow_bn: false,
        allow_pure_chain: false,
        deadline_ascent: RootJointDeadlineAscentPolicy::BaseFoldOnly,
        log_tag,
    };
    root_tighten_relu_preactivations_with_options(
        graph, input, targets, gpu, deadline, options, bounds,
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RootSparseDeadlineRoute {
    Unbounded,
    BackendScoped,
    CallLocal,
    Unavailable,
}

impl RootSparseDeadlineRoute {
    fn label(self) -> &'static str {
        match self {
            Self::Unbounded => "unbounded",
            Self::BackendScoped => "backend-scoped",
            Self::CallLocal => "call-local",
            Self::Unavailable => "unavailable",
        }
    }
}

/// Select the largest row batch that the sound sparse root fold may construct.
///
/// Backends advertising the broad CROWN deadline contract retain the caller's
/// requested cap. A call-local-only backend (currently CUDA) is narrowed before
/// seed allocation to its audited deadline-bounded capacity. Invalid or absent
/// finite-deadline capacity fails closed.
fn root_sparse_effective_max_sel(
    requested: usize,
    has_deadline: bool,
    honors_backend_deadline: bool,
    call_local_capacity: usize,
) -> (usize, RootSparseDeadlineRoute) {
    if !has_deadline {
        return (requested, RootSparseDeadlineRoute::Unbounded);
    }
    if honors_backend_deadline {
        return (requested, RootSparseDeadlineRoute::BackendScoped);
    }
    if (1..=ny_core::DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS).contains(&call_local_capacity) {
        return (
            requested.min(call_local_capacity),
            RootSparseDeadlineRoute::CallLocal,
        );
    }
    (0, RootSparseDeadlineRoute::Unavailable)
}

#[derive(Debug, Clone, Copy)]
struct RootIntermTightenOptions<'a> {
    /// #joint-interm-grad: optional per-neuron weights for the seed diagonal.
    ///
    /// `None` = the historical unit diagonal, byte for byte. `Some(w)` makes the
    /// device's sum-over-rows return `SUM_j w_j * d(l_j)/d(alpha_k)` exactly,
    /// because the per-row harvest is degree-1 positive-homogeneous (the forward
    /// branch selections depend only on coefficient SIGNS). A borrowed slice
    /// keeps this struct `Copy`.
    objective_sensitivity: Option<&'a HashMap<String, Vec<f32>>>,
    iters: usize,
    lr: f32,
    max_sel: usize,
    probe: bool,
    frozen_stop: bool,
    allow_bn: bool,
    allow_pure_chain: bool,
    deadline_ascent: RootJointDeadlineAscentPolicy,
    log_tag: &'static str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RootJointDeadlineAscentPolicy {
    BaseFoldOnly,
    ExactBounded,
}

impl RootJointDeadlineAscentPolicy {
    fn from_env() -> Self {
        if std::env::var("NY_ROOT_JOINT_INTERM_ALPHA_DEADLINE_ASCENT")
            .ok()
            .as_deref()
            == Some("1")
        {
            Self::ExactBounded
        } else {
            Self::BaseFoldOnly
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RootJointSoundFoldRoute {
    Unbounded,
    BackendScopedDeadline(Instant),
    CallLocalDeadline(Instant),
}

fn root_joint_sound_fold_route(
    deadline: Option<Instant>,
    honors_backend_deadline: bool,
    call_local_capacity: usize,
    n_rows: usize,
) -> Option<RootJointSoundFoldRoute> {
    match deadline {
        None => Some(RootJointSoundFoldRoute::Unbounded),
        Some(deadline)
            if (1..=ny_core::DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS)
                .contains(&call_local_capacity)
                && n_rows <= call_local_capacity =>
        {
            Some(RootJointSoundFoldRoute::CallLocalDeadline(deadline))
        }
        Some(deadline) if honors_backend_deadline => {
            Some(RootJointSoundFoldRoute::BackendScopedDeadline(deadline))
        }
        Some(_) => None,
    }
}

/// Dispatch the root intermediate BASE fold with the canonical zero-beta
/// representation.
///
/// An empty OUTER table means "no beta constraints" in the GPU contract. A
/// table containing one empty inner slice per ReLU is not equivalent: it means
/// beta was supplied and each slice must have that activation's neuron count.
/// The latter malformed shape was masked by the <=8-row call-local non-beta API
/// and made WGPU's backend-scoped >8-row route refuse before dispatch.
fn root_base_sound_fold_by_route(
    route: RootJointSoundFoldRoute,
    backend_beta_fold: impl FnOnce(&[Vec<f32>]) -> ny_core::Result<ny_core::GpuCrownResult>,
    call_local_fold: impl FnOnce(Instant) -> ny_core::Result<ny_core::GpuCrownResult>,
) -> ny_core::Result<ny_core::GpuCrownResult> {
    match route {
        RootJointSoundFoldRoute::Unbounded | RootJointSoundFoldRoute::BackendScopedDeadline(_) => {
            backend_beta_fold(&[])
        }
        RootJointSoundFoldRoute::CallLocalDeadline(deadline) => call_local_fold(deadline),
    }
}

fn root_joint_effective_iters(
    requested_iters: usize,
    deadline: Option<Instant>,
    provides_deadline_bounded_joint: bool,
    policy: RootJointDeadlineAscentPolicy,
) -> usize {
    if deadline.is_some()
        && (policy != RootJointDeadlineAscentPolicy::ExactBounded
            || !provides_deadline_bounded_joint)
    {
        0
    } else {
        requested_iters
    }
}

fn root_joint_accept_sound_fold(
    result: ny_core::Result<ny_core::GpuCrownResult>,
    expected_rows: usize,
    deadline: Option<Instant>,
) -> Option<ny_core::GpuCrownResult> {
    if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
        return None;
    }
    let result = result.ok()?;
    if result.lower_bounds.len() != expected_rows
        || result.upper_bounds.len() != expected_rows
        || !result
            .lower_bounds
            .iter()
            .zip(&result.upper_bounds)
            .all(|(&lower, &upper)| lower.is_finite() && upper.is_finite() && lower <= upper)
    {
        return None;
    }
    Some(result)
}

fn root_joint_accept_gradient(
    result: ny_core::Result<Vec<Vec<f32>>>,
    stepped: &[Vec<bool>],
    deadline: Option<Instant>,
) -> Option<Vec<Vec<f32>>> {
    if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
        return None;
    }
    let gradients = result.ok()?;
    if gradients.len() != stepped.len()
        || !gradients.iter().zip(stepped).all(|(gradient, mask)| {
            gradient.len() == mask.len() && gradient.iter().all(|value| value.is_finite())
        })
    {
        return None;
    }
    Some(gradients)
}

fn root_joint_publish_tightened_bound(
    bounds: &mut HashMap<String, BoundedTensor>,
    target: &str,
    tightened: BoundedTensor,
    deadline: Option<Instant>,
) -> bool {
    if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
        return false;
    }
    let Some(current) = bounds.get(target) else {
        return false;
    };
    let Some((commit_value, disjoint)) = current.intersection_per_element(&tightened) else {
        return false;
    };
    // `intersection_per_element` conservatively unions a disjoint element.
    // That is sound in isolation, but this publication seam promises a strict
    // lattice meet against the CURRENT map. A stale/corrupt disjoint proposal
    // must therefore abort rather than widen one live endpoint.
    if disjoint != 0 {
        return false;
    }
    if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
        return false;
    }
    bounds.insert(target.to_owned(), commit_value);
    true
}

/// Consume an already-validated preparation when the selector produced one;
/// only legacy target-name callers invoke the extractor closure. Keeping this
/// tiny ownership seam explicit prevents the wide preflight from accidentally
/// becoming a second full graph extraction during execution.
fn prepared_or_extract<T>(prepared: Option<T>, extract: impl FnOnce() -> Option<T>) -> Option<T> {
    match prepared {
        Some(value) => Some(value),
        None => extract(),
    }
}

fn root_interm_host_seed_peak_bytes(n_rows: usize, pre_dim: usize) -> Option<usize> {
    n_rows
        .checked_mul(pre_dim)?
        .checked_mul(3)?
        .checked_mul(size_of::<f32>())?
        .checked_add(pre_dim.checked_mul(2 * size_of::<f32>() + size_of::<usize>())?)?
        .checked_add(n_rows.checked_mul(8 * size_of::<f32>())?)
}

/// #clip-chain-gather: the regression lock for the measured 81%-of-waves refusal
/// and the exact-rational enclosure oracle for what opening it admits.
///
/// The fixture is a PURE FEED-FORWARD CHAIN with ZERO `Layer::Add` — the
/// oval21 `base_kw` shape class (`Conv → Relu → Conv → Relu → Flatten → Gemm →
/// Relu → Gemm`), and the shape class of all 155 relusplitter ONNX models. On
/// such a graph `saw_residual` is structurally false forever, so the terminal
/// gate `!saw_residual && !allow_pure_chain` fires on 100% of reconstruct
/// attempts.
#[cfg(test)]
mod clip_chain_gather_gate {
    use super::*;
    use crate::layers::{LinearLayer, ReLULayer};
    use crate::GraphNode;
    use ndarray::{arr1, arr2};
    use num_rational::BigRational;
    use num_traits::Zero;

    const W1: [[f32; 4]; 4] = [
        [0.9, -0.4, 0.3, 0.2],
        [-0.5, 0.8, -0.2, 0.4],
        [0.35, 0.25, -0.75, -0.3],
        [-0.2, -0.6, 0.45, 0.7],
    ];
    const B1: [f32; 4] = [0.05, -0.1, 0.2, -0.15];
    const W2: [[f32; 4]; 3] = [
        [0.6, -0.3, 0.5, 0.2],
        [-0.4, 0.7, -0.25, 0.35],
        [0.3, 0.45, -0.55, -0.6],
    ];
    const B2: [f32; 3] = [-0.05, 0.12, 0.08];
    const W3: [[f32; 3]; 2] = [[0.8, -0.5, 0.3], [-0.35, 0.65, -0.45]];
    const B3: [f32; 2] = [0.02, -0.07];

    /// Env that would silently reroute one leg and void the differential: the
    /// closed leg must be the historical refusal, and the open leg's exit must be
    /// decided by the new gate alone.
    fn guard_env() -> std::sync::RwLockWriteGuard<'static, ()> {
        let env_lock = ny_test_utils::env::lock_env();
        for (var, why) in [
            ("NY_BAB_CHAIN_WIDE", "the closed leg would route wide too"),
            ("NY_CLIP_CHAIN_GATHER", "the closed leg would open itself"),
            (
                "NY_CUDA_WIDE",
                "a global wide backend would change the open exit",
            ),
            (
                "NY_HYDRA_CROWN",
                "a global wide backend would change the open exit",
            ),
        ] {
            assert!(
                std::env::var(var).is_err(),
                "clip-chain-gather tests require {var} to be unset ({why})"
            );
        }
        env_lock
    }

    fn pure_chain_net() -> GraphNetwork {
        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::from_input(
            "g1",
            Layer::Linear(LinearLayer::new(arr2(&W1), Some(arr1(&B1))).expect("linear g1")),
        ));
        g.add_node(GraphNode::new(
            "r1",
            Layer::ReLU(ReLULayer),
            vec!["g1".to_string()],
        ));
        g.add_node(GraphNode::new(
            "g2",
            Layer::Linear(LinearLayer::new(arr2(&W2), Some(arr1(&B2))).expect("linear g2")),
            vec!["r1".to_string()],
        ));
        g.add_node(GraphNode::new(
            "r2",
            Layer::ReLU(ReLULayer),
            vec!["g2".to_string()],
        ));
        g.add_node(GraphNode::new(
            "out",
            Layer::Linear(LinearLayer::new(arr2(&W3), Some(arr1(&B3))).expect("linear out")),
            vec!["r2".to_string()],
        ));
        g.set_output("out");
        g
    }

    /// The WIDE box: every ReLU is unstable, so the fold is a genuine relaxation
    /// and the gather has root-unstable columns to gather.
    const WIDE_BOX: ([f32; 4], [f32; 4]) = ([-1.0, -1.0, -1.0, -1.0], [1.0, 1.0, 1.0, 1.0]);
    /// The NO-SLACK box: a 1e-3 half-width box around a point whose every
    /// pre-activation is >= 0.17 away from zero, so ALL SEVEN ReLUs are stable and
    /// the network is EXACTLY AFFINE on the box. CROWN's relaxation is then exact,
    /// the fold's concretized minimum is the true minimum, and the enclosure
    /// oracle becomes sharp to rounding instead of resting on relaxation slack —
    /// the adversarial no-slack case, the same discipline `clip_interm_soundness`
    /// uses for the clip's own enclosure direction.
    const TIGHT_BOX: ([f32; 4], [f32; 4]) = (
        [0.799, -0.601, 0.899, -0.401],
        [0.801, -0.599, 0.901, -0.399],
    );

    fn boxed(bounds: ([f32; 4], [f32; 4])) -> BoundedTensor {
        BoundedTensor::new(arr1(&bounds.0).into_dyn(), arr1(&bounds.1).into_dyn())
            .expect("valid input box")
    }

    fn spec_matrix() -> Array2<f32> {
        Array2::from_shape_vec((2, 2), vec![1.0, -1.0, -1.0, 1.0]).expect("row-major spec")
    }

    struct Fixture {
        graph: GraphNetwork,
        root_bounds: HashMap<String, Arc<BoundedTensor>>,
        parent: MultiObjectiveGraphBabDomain,
    }

    fn fixture() -> Fixture {
        fixture_in(WIDE_BOX)
    }

    fn fixture_in(bounds: ([f32; 4], [f32; 4])) -> Fixture {
        let graph = pure_chain_net();
        let input = boxed(bounds);
        let node_bounds = graph.collect_node_bounds(&input).expect("node bounds");
        let root_bounds: HashMap<String, Arc<BoundedTensor>> = node_bounds
            .iter()
            .map(|(name, bounds)| (name.clone(), Arc::new(bounds.clone())))
            .collect();
        let parent = MultiObjectiveGraphBabDomain::root(
            node_bounds,
            vec![(-8.0, 8.0), (-8.0, 8.0)],
            &input,
            &[0.0, 0.0],
            false,
        )
        .expect("root domain");
        Fixture {
            graph,
            root_bounds,
            parent,
        }
    }

    /// The fixture is genuinely the class the gate discards: no `Layer::Add`
    /// anywhere, so `saw_residual` can never be set.
    #[ntest::timeout(20000)]
    #[test]
    fn fixture_is_add_free_so_saw_residual_is_structurally_false() {
        let graph = pure_chain_net();
        assert!(
            graph
                .nodes
                .values()
                .all(|node| !matches!(node.layer, Layer::Add(_))),
            "fixture must be a pure chain for the >=1-residual gate to be the only cause"
        );
        assert!(
            graph.nodes.values().all(|node| node.inputs.len() <= 1),
            "a 2-input node is the only thing that can set saw_residual"
        );
    }

    #[ntest::timeout(20000)]
    #[test]
    fn chain_gather_gate_defaults_off() {
        let _env_lock = guard_env();
        assert!(
            !complete_clip_chain_gather_enabled(),
            "NY_CLIP_CHAIN_GATHER must default OFF"
        );
    }

    #[ntest::timeout(20000)]
    #[test]
    fn chain_gather_gate_accepts_exactly_one() {
        ny_test_utils::env::with_serialized_env_vars_removed(&["NY_CLIP_CHAIN_GATHER"], || {
            assert!(!complete_clip_chain_gather_enabled());
        });
        for (value, enabled) in [("0", false), ("true", false), ("1", true)] {
            ny_test_utils::env::with_serialized_env_vars(
                &[("NY_CLIP_CHAIN_GATHER", value)],
                || assert_eq!(complete_clip_chain_gather_enabled(), enabled),
            );
        }
    }

    /// LEVEL 3 named directly: with the gate closed, the per-parent prep refuses
    /// at `extract_pure_chain_disallowed` — the second disjunct of
    /// `resnet_decompose.rs`'s terminal gate — and NOT at any other extraction
    /// exit. This is the exact expression measured as 710 of 878 waves.
    #[ntest::timeout(20000)]
    #[test]
    fn closed_gate_refuses_the_prep_at_the_residual_count_predicate() {
        let _env_lock = guard_env();
        let f = fixture();
        let parent_node_bounds = f.parent.node_bounds().to_shared_hash_map();
        let closed = prep_resnet_domain_ext(
            &f.graph,
            "out",
            &parent_node_bounds,
            f.parent.input_bounds(),
            Some(f.parent.beta_state()),
            Some(f.parent.alpha_state()),
            false,
            false,
            false,
        );
        assert!(closed.is_none(), "closed gate must refuse the pure chain");
        assert_eq!(
            crate::beta_crown::engine::graph::propagation::batched::prep_resnet_domain_last_refusal(
            ),
            "extract_pure_chain_disallowed",
            "the refusal must be the residual-COUNT predicate, not a structural one"
        );
    }

    /// THE EXTRACTION SUCCEEDS AND WAS BEING DISCARDED: with the gate open the
    /// SAME walk over the SAME graph yields a complete decomposition. Nothing
    /// about the graph changed, so the closed leg's refusal was a policy verdict
    /// on a valid object.
    #[ntest::timeout(20000)]
    #[test]
    fn open_gate_yields_a_complete_decomposition_of_the_same_graph() {
        let _env_lock = guard_env();
        let f = fixture();
        let parent_node_bounds = f.parent.node_bounds().to_shared_hash_map();
        let prep = prep_resnet_domain_ext(
            &f.graph,
            "out",
            &parent_node_bounds,
            f.parent.input_bounds(),
            Some(f.parent.beta_state()),
            Some(f.parent.alpha_state()),
            true,
            false,
            false,
        )
        .expect("open gate must decompose the pure chain");
        assert_eq!(
            prep.relu_names,
            vec!["r2".to_string(), "r1".to_string()],
            "both ReLUs must be captured in backward fold order"
        );
        assert!(
            prep.stop_node.is_none(),
            "the walk must reach NETWORK_INPUT"
        );
        assert_eq!(prep.in_lo.len(), 4);
        assert_eq!(prep.in_hi.len(), 4);
        assert_eq!(
            prep.segments.len(),
            1,
            "a pure chain decomposes to exactly one Chain segment"
        );
    }

    /// The wave-level regression lock. Both legs reproduce a MEASURED production
    /// state on relusplitter oval21 (2026-08-10, `NY_CLIP_INTERM_RESNET_PROBE=1`):
    ///
    /// * closed  -> `gather_extract_pure_chain_disallowed` (the baseline run's
    ///   dominant bucket, 710/878 waves at `wall_ms=0`);
    /// * open    -> `gather_no_sound_gpu_backend` (Run D's dominant bucket, 605
    ///   waves, once the pure-chain gate stopped being the blocker).
    ///
    /// The open leg's exit sits AFTER the prep loop, the cross-parent ReLU-shape
    /// check, the full-width caps, and the root-unstable column build, so reaching
    /// it proves every one of those passed on this graph.
    #[ntest::timeout(20000)]
    #[test]
    fn gate_moves_the_gather_exit_from_the_residual_gate_to_the_backend_lookup() {
        let _env_lock = guard_env();
        let f = fixture();
        let spec = spec_matrix();
        let engine = ny_core::NaiveCpuGemmEngine;
        let parents = [&f.parent];

        assert!(
            complete_clip_gpu_mean_refusal_reason(&parents, &spec, None).is_none(),
            "the wave must be admitted so the exit under test is the only variable"
        );

        let closed = complete_clip_mean_las_from_gpu_probed(
            &f.graph,
            &f.root_bounds,
            &parents,
            &spec,
            &engine,
            None,
            false,
        );
        assert_eq!(
            closed.err(),
            Some("gather_extract_pure_chain_disallowed"),
            "gate OFF must be byte-identical to the historical refusal"
        );

        let open = complete_clip_mean_las_from_gpu_probed(
            &f.graph,
            &f.root_bounds,
            &parents,
            &spec,
            &engine,
            None,
            true,
        );
        assert_eq!(
            open.err(),
            Some("gather_no_sound_gpu_backend"),
            "gate ON must clear the pure-chain gate and reach the backend lookup"
        );
    }

    // ---- exact-rational enclosure oracle -------------------------------------

    fn rat(value: f32) -> BigRational {
        BigRational::from_float(value).expect("finite f32 is exactly a rational")
    }

    /// The EXACT network value of each spec row at a point, in `BigRational`.
    /// Every f32 weight/bias/coordinate converts EXACTLY, and ReLU is exact on
    /// rationals, so this reference carries ZERO floating-point slop.
    fn exact_spec_values(x: &[f32; 4], spec: &Array2<f32>) -> Vec<BigRational> {
        let zero = BigRational::zero();
        let xs: Vec<BigRational> = x.iter().map(|&v| rat(v)).collect();

        let mut h1 = Vec::with_capacity(4);
        for (row, bias) in W1.iter().zip(B1) {
            let mut acc = rat(bias);
            for (w, xv) in row.iter().zip(&xs) {
                acc += rat(*w) * xv;
            }
            h1.push(if acc > zero { acc } else { zero.clone() });
        }
        let mut h2 = Vec::with_capacity(3);
        for (row, bias) in W2.iter().zip(B2) {
            let mut acc = rat(bias);
            for (w, hv) in row.iter().zip(&h1) {
                acc += rat(*w) * hv;
            }
            h2.push(if acc > zero { acc } else { zero.clone() });
        }
        let mut y = Vec::with_capacity(2);
        for (row, bias) in W3.iter().zip(B3) {
            let mut acc = rat(bias);
            for (w, hv) in row.iter().zip(&h2) {
                acc += rat(*w) * hv;
            }
            y.push(acc);
        }
        (0..spec.nrows())
            .map(|s| {
                let mut acc = zero.clone();
                for (j, yv) in y.iter().enumerate() {
                    acc += rat(spec[[s, j]]) * yv;
                }
                acc
            })
            .collect()
    }

    /// SOUNDNESS ORACLE for the repair. The decomposition the gate was discarding
    /// is folded through the SAME arithmetic the sound capability seam uses
    /// (`joint_lower_bound_debug_f64` + `f64_to_f32_down`, exactly
    /// `HermeticSoundGpuCrownEngine::bounds`), and the published f32 bound must
    /// ENCLOSE — be `<=` — the EXACT rational network value at every sampled point
    /// of the domain box, on every spec row.
    ///
    /// Direction matters: Complete Clipping TIGHTENS, so the failure mode this
    /// guards is a bound that cut below the truth. The reference is exact
    /// rational, and the outward `f64_to_f32_down` publication step is ~1e-7 at
    /// unit magnitude against ~1e-15 of f64 fold drift, so the comparison is not
    /// resting on the reference's own rounding.
    ///
    /// Two legs, because an enclosure oracle is only as sharp as its slack. On
    /// [`WIDE_BOX`] the ReLUs are unstable and the relaxation gap is large
    /// (measured: the assertion first fires at a +2.0 bound perturbation, holds at
    /// +1.5), so that leg proves DIRECTION only. On [`TIGHT_BOX`] every ReLU is
    /// stable, the chain is exactly affine, and the fold's concretization IS the
    /// true minimum — measured minimum slack 3.99e-8, i.e. ONE f32 ULP at that
    /// magnitude, and a mutation that publishes `next_up_f32` of the sound value
    /// fails this leg. The `max_slack` assertion pins that sharpness so the leg
    /// cannot silently decay into another direction-only check.
    #[ntest::timeout(60000)]
    #[test]
    fn newly_admitted_pure_chain_fold_encloses_the_exact_rational_network() {
        let _env_lock = guard_env();
        let spec = spec_matrix();
        let mut checked = 0usize;

        for (label, bounds, max_slack) in [
            ("wide/relaxed", WIDE_BOX, None),
            ("tight/no-slack", TIGHT_BOX, Some(1.0e-6f64)),
        ] {
            let f = fixture_in(bounds);
            let parent_node_bounds = f.parent.node_bounds().to_shared_hash_map();
            let prep = prep_resnet_domain_ext(
                &f.graph,
                "out",
                &parent_node_bounds,
                f.parent.input_bounds(),
                Some(f.parent.beta_state()),
                Some(f.parent.alpha_state()),
                true,
                false,
                false,
            )
            .expect("open gate must decompose the pure chain");

            let seed_rows = spec.as_slice_memory_order().expect("row-major spec");
            let folded = ny_core::joint_alpha_grad::joint_lower_bound_debug_f64(
                &prep.segments,
                seed_rows,
                &vec![0.0f32; spec.nrows()],
                spec.nrows(),
                spec.ncols(),
                &prep.in_lo,
                &prep.in_hi,
            )
            .expect("the admitted decomposition must fold to the input box");
            let published: Vec<f32> = folded.into_iter().map(ny_core::f64_to_f32_down).collect();
            assert_eq!(published.len(), spec.nrows());
            assert!(published.iter().all(|v| v.is_finite()));

            // Every corner of the box (the affine minimum lives at one) plus a
            // deterministic interior sweep.
            let mut points: Vec<[f32; 4]> = Vec::new();
            for mask in 0u8..16 {
                let mut p = [0.0f32; 4];
                for (j, coord) in p.iter_mut().enumerate() {
                    *coord = if mask & (1 << j) == 0 {
                        bounds.0[j]
                    } else {
                        bounds.1[j]
                    };
                }
                points.push(p);
            }
            let mut state: u64 = 0x5EED_C1A1_9726;
            let mut next = move || {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 40) as f32 / f64::from(1u32 << 24) as f32) * 0.5 + 0.5
            };
            for _ in 0..96 {
                let mut p = [0.0f32; 4];
                for (j, coord) in p.iter_mut().enumerate() {
                    let t = next().clamp(0.0, 1.0);
                    *coord = bounds.0[j] + t * (bounds.1[j] - bounds.0[j]);
                }
                points.push(p);
            }

            let mut min_slack = f64::INFINITY;
            for point in &points {
                let exact = exact_spec_values(point, &spec);
                for (row, exact_value) in exact.iter().enumerate() {
                    let bound = rat(published[row]);
                    assert!(
                        bound <= *exact_value,
                        "[{label}] newly-admitted fold cut BELOW the exact value: row {row}, \
                         point {point:?}, bound {} > exact {}",
                        published[row],
                        exact_value,
                    );
                    let slack = exact_to_f64(exact_value) - f64::from(published[row]);
                    min_slack = min_slack.min(slack);
                    checked += 1;
                }
            }
            if let Some(limit) = max_slack {
                assert!(
                    min_slack >= 0.0 && min_slack < limit,
                    "[{label}] leg must be SHARP for the enclosure claim to bite: observed \
                     minimum slack {min_slack}"
                );
            }
        }
        assert_eq!(
            checked,
            2 * 112 * spec.nrows(),
            "oracle must be non-vacuous"
        );
    }

    /// Rational → f64 for the slack DIAGNOSTIC only. Never used in the enclosure
    /// assertion itself, which stays entirely in exact rationals.
    fn exact_to_f64(value: &BigRational) -> f64 {
        use num_traits::ToPrimitive;
        value.to_f64().expect("finite rational")
    }

    /// THE SCORE CANNOT BECOME A BOUND. The gather's only consumer turns each
    /// mean coefficient into a top-K rank, and what leaves is a set of NEURON
    /// INDICES. Feed the selector an ADVERSARIAL score (sign-flipped and scaled by
    /// 1e6) and the published payload must still be indices that are in range AND
    /// root-unstable — never a value, and never a neuron the root pass did not
    /// already list as a candidate.
    #[ntest::timeout(20000)]
    #[test]
    fn an_adversarial_mean_score_can_only_nominate_root_unstable_indices() {
        let _env_lock = guard_env();
        let f = fixture();
        let honest: HashMap<String, Array1<f32>> = ["r1", "r2"]
            .into_iter()
            .map(|relu| {
                let seed = f.graph.nodes[relu].inputs[0].clone();
                let width = f.root_bounds[&seed].len();
                let values: Vec<f32> = (0..width).map(|i| -0.25 - 0.1 * i as f32).collect();
                (relu.to_string(), Array1::from_vec(values))
            })
            .collect();
        let adversarial: HashMap<String, Array1<f32>> = honest
            .iter()
            .map(|(relu, values)| (relu.clone(), values.mapv(|v| -v * 1.0e6)))
            .collect();

        let history = GraphSplitHistory::new();
        let mut nominated = 0usize;
        for by_relu in [honest, adversarial] {
            let mean_las = CompleteClipMeanLowerA {
                spec_rows: 2,
                by_relu,
            };
            let Some(decisions) = complete_clip_decisions_from_mean_las(
                &f.graph,
                &f.root_bounds,
                f.parent.node_bounds(),
                &history,
                &mean_las,
                2,
                None,
            ) else {
                continue;
            };
            for (relu, neurons) in &decisions {
                let seed = f.graph.nodes[relu].inputs[0].clone();
                let root = &f.root_bounds[&seed];
                for &neuron in neurons {
                    assert!(neuron < root.len(), "index escaped its layer width");
                    assert!(
                        root.lower()[neuron] < 0.0 && root.upper()[neuron] > 0.0,
                        "a score nominated a neuron the root pass did not list as unstable"
                    );
                    nominated += 1;
                }
            }
        }
        assert!(
            nominated > 0,
            "oracle must be non-vacuous: the fixture must produce nominations"
        );
    }
}

#[cfg(test)]
mod root_joint_deadline_route_tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn deadline_ascent_gate_accepts_exactly_one() {
        for value in ["0", "true", "yes"] {
            ny_test_utils::env::with_serialized_env_vars(
                &[("NY_ROOT_JOINT_INTERM_ALPHA_DEADLINE_ASCENT", value)],
                || {
                    assert_eq!(
                        RootJointDeadlineAscentPolicy::from_env(),
                        RootJointDeadlineAscentPolicy::BaseFoldOnly
                    );
                },
            );
        }
        ny_test_utils::env::with_serialized_env_vars(
            &[("NY_ROOT_JOINT_INTERM_ALPHA_DEADLINE_ASCENT", "1")],
            || {
                assert_eq!(
                    RootJointDeadlineAscentPolicy::from_env(),
                    RootJointDeadlineAscentPolicy::ExactBounded
                );
            },
        );
    }

    #[test]
    fn finite_joint_ascent_requires_its_exact_deadline_capability() {
        let deadline = Instant::now() + Duration::from_secs(1);
        assert_eq!(
            root_joint_effective_iters(
                7,
                Some(deadline),
                false,
                RootJointDeadlineAscentPolicy::ExactBounded,
            ),
            0
        );
        assert_eq!(
            root_joint_effective_iters(
                7,
                Some(deadline),
                true,
                RootJointDeadlineAscentPolicy::BaseFoldOnly,
            ),
            0
        );
        assert_eq!(
            root_joint_effective_iters(
                7,
                Some(deadline),
                true,
                RootJointDeadlineAscentPolicy::ExactBounded,
            ),
            7
        );
        assert_eq!(
            root_joint_effective_iters(7, None, false, RootJointDeadlineAscentPolicy::BaseFoldOnly,),
            7
        );
    }

    #[test]
    fn cuda_sized_rows_use_call_local_deadline_route() {
        let deadline = Instant::now() + Duration::from_secs(1);
        assert_eq!(
            root_joint_sound_fold_route(
                Some(deadline),
                false,
                ny_core::DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS,
                ny_core::DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS,
            ),
            Some(RootJointSoundFoldRoute::CallLocalDeadline(deadline))
        );
    }

    #[test]
    fn finite_fold_refuses_unbounded_backend_and_oversized_call_local_batch() {
        let deadline = Instant::now() + Duration::from_secs(1);
        assert_eq!(
            root_joint_sound_fold_route(
                Some(deadline),
                false,
                ny_core::DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS,
                ny_core::DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS + 1,
            ),
            None
        );
        assert_eq!(
            root_joint_sound_fold_route(
                Some(deadline),
                false,
                ny_core::DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS + 1,
                1,
            ),
            None
        );
    }

    #[test]
    fn backend_scoped_and_unbounded_routes_remain_available() {
        let deadline = Instant::now() + Duration::from_secs(1);
        assert_eq!(
            root_joint_sound_fold_route(Some(deadline), true, 0, 512),
            Some(RootJointSoundFoldRoute::BackendScopedDeadline(deadline))
        );
        assert_eq!(
            root_joint_sound_fold_route(None, false, 0, 512),
            Some(RootJointSoundFoldRoute::Unbounded)
        );
    }

    #[test]
    fn backend_scoped_base_fold_above_eight_rows_uses_absent_beta_and_succeeds() {
        let deadline = Instant::now() + Duration::from_secs(1);
        let route = root_joint_sound_fold_route(Some(deadline), true, 8, 9)
            .expect("a deadline-authoritative backend owns rows above the call-local cap");
        assert_eq!(
            route,
            RootJointSoundFoldRoute::BackendScopedDeadline(deadline)
        );

        let backend_called = std::cell::Cell::new(false);
        let result = root_base_sound_fold_by_route(
            route,
            |beta_signed| {
                backend_called.set(true);
                assert_eq!(
                    beta_signed.len(),
                    0,
                    "zero beta is an empty outer table, never N empty inner slices"
                );
                Ok(ny_core::GpuCrownResult {
                    lower_bounds: vec![-1.0; 9],
                    upper_bounds: vec![1.0; 9],
                })
            },
            |_| panic!("nine rows must not use the <=8 call-local non-beta API"),
        )
        .expect("the well-formed zero-beta backend-scoped fold succeeds");
        assert!(backend_called.get());
        assert_eq!(result.lower_bounds.len(), 9);
        assert_eq!(result.upper_bounds.len(), 9);
    }

    #[test]
    fn sparse_cuda_call_local_route_caps_rows_before_seed_construction() {
        assert_eq!(
            root_sparse_effective_max_sel(
                512,
                true,
                false,
                ny_core::DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS,
            ),
            (
                ny_core::DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS,
                RootSparseDeadlineRoute::CallLocal,
            )
        );
        assert_eq!(
            root_sparse_effective_max_sel(
                4,
                true,
                false,
                ny_core::DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS,
            ),
            (4, RootSparseDeadlineRoute::CallLocal),
        );
    }

    #[test]
    fn sparse_broad_deadline_backend_keeps_requested_rows() {
        assert_eq!(
            root_sparse_effective_max_sel(512, true, true, 0),
            (512, RootSparseDeadlineRoute::BackendScoped),
        );
        assert_eq!(
            root_sparse_effective_max_sel(512, false, false, 0),
            (512, RootSparseDeadlineRoute::Unbounded),
        );
    }

    #[test]
    fn sparse_invalid_call_local_capacity_fails_closed() {
        assert_eq!(
            root_sparse_effective_max_sel(512, true, false, 0),
            (0, RootSparseDeadlineRoute::Unavailable),
        );
        assert_eq!(
            root_sparse_effective_max_sel(
                512,
                true,
                false,
                ny_core::DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS + 1,
            ),
            (0, RootSparseDeadlineRoute::Unavailable),
        );
    }

    #[test]
    fn fold_acceptance_rejects_malformed_or_nonfinite_results() {
        let result = |lower_bounds, upper_bounds| {
            Ok(ny_core::GpuCrownResult {
                lower_bounds,
                upper_bounds,
            })
        };
        assert!(
            root_joint_accept_sound_fold(result(vec![-1.0, 0.0], vec![1.0, 2.0]), 2, None,)
                .is_some()
        );
        assert!(
            root_joint_accept_sound_fold(result(vec![-1.0], vec![1.0, 2.0]), 2, None).is_none()
        );
        assert!(
            root_joint_accept_sound_fold(result(vec![f32::NAN, 0.0], vec![1.0, 2.0]), 2, None,)
                .is_none()
        );
        assert!(root_joint_accept_sound_fold(
            result(vec![-1.0, 0.0], vec![1.0, f32::INFINITY]),
            2,
            None,
        )
        .is_none());
        assert!(
            root_joint_accept_sound_fold(result(vec![2.0, 0.0], vec![1.0, 2.0]), 2, None).is_none()
        );
    }

    #[test]
    fn late_gradient_or_replay_cannot_publish_partial_bound() {
        let expired = Some(
            Instant::now()
                .checked_sub(Duration::from_millis(1))
                .expect("system uptime exceeds one millisecond"),
        );
        let stepped = vec![vec![true, false]];
        assert!(root_joint_accept_gradient(Ok(vec![vec![1.0, 0.0]]), &stepped, expired).is_none());
        assert!(root_joint_accept_sound_fold(
            Ok(ny_core::GpuCrownResult {
                lower_bounds: vec![-0.5],
                upper_bounds: vec![0.5],
            }),
            1,
            expired,
        )
        .is_none());

        let original = BoundedTensor::new(
            ndarray::arr1(&[-1.0f32]).into_dyn(),
            ndarray::arr1(&[1.0f32]).into_dyn(),
        )
        .unwrap();
        let candidate = BoundedTensor::new(
            ndarray::arr1(&[-0.5f32]).into_dyn(),
            ndarray::arr1(&[0.5f32]).into_dyn(),
        )
        .unwrap();
        let mut bounds = HashMap::from([("target".to_owned(), original.clone())]);
        assert!(!root_joint_publish_tightened_bound(
            &mut bounds,
            "target",
            candidate,
            expired,
        ));
        let retained = bounds.get("target").unwrap();
        assert!(retained
            .lower()
            .iter()
            .zip(original.lower())
            .all(|(&actual, &expected)| actual.to_bits() == expected.to_bits()));
        assert!(retained
            .upper()
            .iter()
            .zip(original.upper())
            .all(|(&actual, &expected)| actual.to_bits() == expected.to_bits()));
    }

    #[test]
    fn publication_reintersects_the_live_map_instead_of_overwriting_a_newer_shrink() {
        let live = BoundedTensor::new(
            ndarray::arr1(&[-0.25f32]).into_dyn(),
            ndarray::arr1(&[0.25f32]).into_dyn(),
        )
        .unwrap();
        let stale_candidate = BoundedTensor::new(
            ndarray::arr1(&[-0.5f32]).into_dyn(),
            ndarray::arr1(&[0.5f32]).into_dyn(),
        )
        .unwrap();
        let mut bounds = HashMap::from([("target".to_owned(), live.clone())]);
        assert!(root_joint_publish_tightened_bound(
            &mut bounds,
            "target",
            stale_candidate,
            None,
        ));
        let retained = bounds.get("target").unwrap();
        assert_eq!(
            retained
                .lower()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            live.lower().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            retained
                .upper()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            live.upper().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn publication_rejects_a_disjoint_candidate_instead_of_union_widening() {
        let live = BoundedTensor::new(
            ndarray::arr1(&[-0.25f32]).into_dyn(),
            ndarray::arr1(&[0.25f32]).into_dyn(),
        )
        .unwrap();
        let disjoint = BoundedTensor::new(
            ndarray::arr1(&[1.0f32]).into_dyn(),
            ndarray::arr1(&[2.0f32]).into_dyn(),
        )
        .unwrap();
        let mut bounds = HashMap::from([("target".to_owned(), live.clone())]);
        assert!(!root_joint_publish_tightened_bound(
            &mut bounds,
            "target",
            disjoint,
            None,
        ));
        let retained = bounds.get("target").unwrap();
        assert!(retained
            .lower()
            .iter()
            .zip(live.lower())
            .all(|(&actual, &expected)| actual.to_bits() == expected.to_bits()));
        assert!(retained
            .upper()
            .iter()
            .zip(live.upper())
            .all(|(&actual, &expected)| actual.to_bits() == expected.to_bits()));
    }
}

#[allow(clippy::too_many_arguments)]
fn root_tighten_relu_preactivations_with_options(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    targets: impl IntoIterator<Item = (String, Option<ResnetDomainPrep>)>,
    gpu: Option<&dyn ny_core::GpuCrownBackward>,
    deadline: Option<Instant>,
    options: RootIntermTightenOptions<'_>,
    bounds: &mut HashMap<String, BoundedTensor>,
) -> usize {
    let RootIntermTightenOptions {
        objective_sensitivity,
        iters: requested_iters,
        lr,
        max_sel,
        probe,
        frozen_stop,
        allow_bn,
        allow_pure_chain,
        deadline_ascent,
        log_tag,
    } = options;
    if max_sel == 0 {
        return 0;
    }
    // Sanctioned sound-GPU access (same filter as refine_interm_bounds_with_opts):
    // the joint gradient only STEERS Adam (heuristic — cannot affect soundness);
    // every kept bound comes from the certified sound fold.
    let Some(gpu) = gpu.filter(|gpu| gpu.provides_sound_gpu_crown()) else {
        eprintln!("[{log_tag}] no sound GPU crown backward; skipping (sound no-op)");
        return 0;
    };
    // Since #critical-alpha-deadline, both CUDA and wgpu expose a method-specific,
    // cooperatively cancellable joint adjoint. Keep finite-slice ascent enabled
    // only for that exact capability; older backends retain the certified
    // base-fold-only behavior.
    let iters = root_joint_effective_iters(
        requested_iters,
        deadline,
        gpu.provides_deadline_bounded_joint_alpha_gradient_resident(),
        deadline_ascent,
    );
    let past_deadline = || deadline.is_some_and(|d| Instant::now() >= d);

    let mut n_tightened = 0usize;
    for (target, prepared) in targets {
        if past_deadline() {
            break;
        }
        let Some(ref_bt) = bounds.get(&target).cloned() else {
            continue;
        };
        let pre_dim = ref_bt.lower().len();
        if pre_dim == 0 {
            continue;
        }
        let ref_l: Vec<f32> = ref_bt.lower().iter().copied().collect();
        let ref_u: Vec<f32> = ref_bt.upper().iter().copied().collect();
        // Crossing rows only (stable rows' relaxations are already exact),
        // widest-first, capped — the interm_refine sel convention.
        let mut sel: Vec<usize> = (0..pre_dim)
            .filter(|&j| ref_l[j] < 0.0 && ref_u[j] > 0.0)
            .collect();
        sel.sort_by(|&a, &b| {
            let wa = ref_u[a] - ref_l[a];
            let wb = ref_u[b] - ref_l[b];
            wb.partial_cmp(&wa).unwrap_or(std::cmp::Ordering::Equal)
        });
        sel.truncate(max_sel);
        let n_rows = sel.len();
        if n_rows == 0 {
            continue;
        }
        let Some(sound_fold_route) = root_joint_sound_fold_route(
            deadline,
            gpu.honors_crown_backward_deadline(),
            gpu.deadline_bounded_resnet_sound_max_rows(),
            n_rows,
        ) else {
            if probe {
                eprintln!(
                    "[{log_tag}] '{target}': no deadline-bounded sound fold for {n_rows} row(s); \
                     keeping reference"
                );
            }
            continue;
        };

        // Truncated stack from L down to the network input — or, under the
        // #frozen-stop opt-in, down to the deepest frozen-bounded node M below
        // which the extraction cannot continue (cgan: the generator boundary
        // ConvTranspose). SOUND: reachable(M) ⊆ box(M), so the certified fold of
        // graph[M→L] concretized against box(M) is a valid enclosure of L's true
        // pre-activation set; every KEPT bound still comes from the certified
        // sound fold, shrink-only intersected below. Dark-within-dark.
        // #cgan-bn-gpu-extract: BN-as-1x1-conv extraction, scoped to THIS lane
        // (legacy callers hard-pass false, so other lanes never see BN even with
        // the env set).
        // frozen-stop IMPLIES pure-chain: a truncated discriminator stack has no
        // residual Add, so without [Chain(..)] the stop lane could never fire.
        let allow_pure_chain = frozen_stop || allow_pure_chain;
        let prep = prepared_or_extract(prepared, || {
            let Some(prep) = prep_resnet_domain_ext(
                graph,
                &target,
                bounds,
                input,
                None,
                None,
                allow_pure_chain,
                allow_bn,
                frozen_stop,
            ) else {
                if probe {
                    eprintln!(
                        "[{log_tag}] '{target}': prep refused reason={}; keeping reference",
                        super::prep_resnet_domain_last_refusal(),
                    );
                }
                return None;
            };
            Some(prep)
        });
        let Some(prep) = prep else {
            continue;
        };
        // Preparation can include graph extraction and host-side materialization.
        // Never launch or publish a fold after that work consumed the slice.
        if past_deadline() {
            break;
        }
        // Stop-box belt check (fail-closed): when the stack stopped at M, the
        // concretization box must be bit-identical to the frozen map's entry for
        // M — the very tensor the extraction resolved against.
        if let Some(m) = prep.stop_node.as_deref() {
            let box_ok = bounds.get(m).is_some_and(|bt| {
                let lo = bt.lower();
                let hi = bt.upper();
                lo.len() == prep.in_lo.len()
                    && hi.len() == prep.in_hi.len()
                    && lo
                        .iter()
                        .zip(prep.in_lo.iter())
                        .all(|(&a, &b)| a.to_bits() == b.to_bits())
                    && hi
                        .iter()
                        .zip(prep.in_hi.iter())
                        .all(|(&a, &b)| a.to_bits() == b.to_bits())
                    && prep
                        .in_lo
                        .iter()
                        .zip(prep.in_hi.iter())
                        .all(|(&l, &u)| l.is_finite() && u.is_finite() && l <= u)
            });
            if !box_ok {
                if probe {
                    eprintln!(
                        "[{log_tag}] '{target}': stop-box mismatch at '{m}'; keeping reference"
                    );
                }
                continue;
            }
            if probe {
                eprintln!(
                    "[{log_tag}] '{target}': truncated stack stop_at='{m}' box_dim={} segs={} relus={}",
                    prep.in_lo.len(),
                    prep.segments.len(),
                    prep.relu_names.len()
                );
            }
        }
        let n_relu = prep.relu_names.len();
        if n_relu == 0 || count_activations(&prep.segments) != n_relu {
            continue;
        }
        let stepped = unstable_masks(&prep.segments, n_relu);
        if !stepped.iter().any(|m| m.iter().any(|&s| s)) {
            continue; // nothing below L to optimize
        }
        let base_slopes = collect_lower_slopes(&prep.segments, n_relu);
        // Identity seed over sel rows (both sides: the sound fold then returns
        // per-row [l,u] of L's pre-activation in one call).
        let Some(seed_elems) = n_rows.checked_mul(pre_dim) else {
            continue;
        };
        // `Vec<f32> -> Arc<[f32]>` may transiently keep the source allocation
        // while filling the Arc allocation. At the second conversion the first
        // Arc is already live, so charge three full coefficient matrices, plus
        // the row-selection/reference vectors and conservative bias staging.
        // This is a host peak admission bound, not merely the two-array logical
        // payload size.
        let Some(seed_peak_bytes) = root_interm_host_seed_peak_bytes(n_rows, pre_dim) else {
            continue;
        };
        if seed_peak_bytes > ROOT_INTERM_MAX_HOST_SEED_BYTES {
            if probe {
                eprintln!(
                    "[{log_tag}] '{target}': host seed peak bytes {seed_peak_bytes} exceed cap \
                     {ROOT_INTERM_MAX_HOST_SEED_BYTES}; keeping reference"
                );
            }
            continue;
        }
        let mut rows = Vec::new();
        if rows.try_reserve_exact(seed_elems).is_err() {
            continue;
        }
        rows.resize(seed_elems, 0.0f32);
        // #joint-interm-grad: the seed diagonal may carry WEIGHTS instead of 1.0.
        //
        // Why this is sound and exact, not an approximation. The joint adjoint's
        // forward branch selections and its adjoint seed depend only on the SIGNS
        // of the coefficient matrix, so the harvested adjoint is invariant under
        // positive scaling of a seed row while the pre-activation coefficients
        // scale linearly. The per-row harvest is therefore exactly degree-1
        // positive-homogeneous, and seeding row `r` with `w[j] >= 0` instead of
        // `1.0` makes the device's sum-over-rows return
        //
        //     SUM_j w_j * d(l_j)/d(alpha_k)
        //
        // which IS the indirect gradient term. The reduction over rows — normally
        // the obstacle to extracting per-row sensitivities — is what delivers the
        // contraction for free. No new kernel, no unreduction, no negated seed.
        //
        // The non-negativity precondition is guaranteed, not assumed:
        // `interm_sensitivity_weights` returns `w_l >= 0` because it is
        // `A_neg * d * (hhat - u) / D` with `A_neg <= 0` and `hhat` clamped into
        // `[l, u]` — verified against central finite differences, including that
        // the naive slope-only form is SIGN-FLIPPED on the l axis and would steer
        // the ascent backwards.
        //
        // `None` is the historical unit diagonal, byte for byte.
        // Per-target lookup: a target absent from the map keeps the unit diagonal.
        let seed_row_weights = objective_sensitivity
            .and_then(|m| m.get(target.as_str()))
            .map(Vec::as_slice);
        match seed_row_weights {
            Some(w) => {
                for (r, &j) in sel.iter().enumerate() {
                    // A weight that is non-finite or negative would break the
                    // homogeneity argument above, so fail closed to the unit seed
                    // for that row rather than emit a meaningless sensitivity.
                    let wj = w.get(j).copied().unwrap_or(1.0);
                    rows[r * pre_dim + j] = if wj.is_finite() && wj >= 0.0 { wj } else { 1.0 };
                }
            }
            None => {
                for (r, &j) in sel.iter().enumerate() {
                    rows[r * pre_dim + j] = 1.0;
                }
            }
        }
        let mut upper_rows = Vec::new();
        if upper_rows.try_reserve_exact(seed_elems).is_err() {
            continue;
        }
        upper_rows.extend_from_slice(&rows);
        let seed = ny_core::GpuCrownSeed {
            lower_a: rows.into(),
            upper_a: upper_rows.into(),
            lower_b: vec![0.0f32; n_rows].into(),
            upper_b: vec![0.0f32; n_rows].into(),
            num_specs: n_rows,
            current_dim: pre_dim,
        };
        // Host-side masks, seed construction, Vec→Arc conversion, and the
        // preparation's segment clone are bounded by explicit memory/work caps
        // but are not preemptible midway. Reassert the authority immediately
        // before any backend call so none of that work can launch a late GPU
        // transaction.
        if past_deadline() {
            break;
        }

        // The legacy wgpu route uses its exact backend deadline lease. CUDA does
        // not advertise that broad mutable contract: it takes the call-local
        // bounded-row and joint-adjoint methods below instead.
        let _gpu_deadline_scope = match sound_fold_route {
            RootJointSoundFoldRoute::BackendScopedDeadline(deadline) => Some(
                crate::sound_gpu_gate::GpuCrownBackendDeadlineScope::set(gpu, Some(deadline)),
            ),
            RootJointSoundFoldRoute::Unbounded | RootJointSoundFoldRoute::CallLocalDeadline(_) => {
                None
            }
        };
        let sound_fold = |segments: &[ny_core::GpuResnetSegment]| {
            root_base_sound_fold_by_route(
                sound_fold_route,
                |beta_signed| {
                    gpu.crown_backward_gpu_resnet_sound_beta(
                        segments,
                        &seed,
                        &prep.in_lo,
                        &prep.in_hi,
                        beta_signed,
                        &prep.frontier_abs,
                        &prep.node_abs,
                    )
                },
                |deadline| {
                    gpu.crown_backward_gpu_resnet_sound_bounded_rows_with_deadline(
                        segments,
                        &seed,
                        &prep.in_lo,
                        &prep.in_hi,
                        &prep.frontier_abs,
                        &prep.node_abs,
                        deadline,
                    )
                },
            )
        };

        // Base sound fold = the ascent's starting enclosure (and best-so-far).
        let mut work_segs = prep.segments.clone();
        let base_fold_result = match sound_fold(&work_segs) {
            Ok(result) => Ok(result),
            Err(error) => {
                if probe {
                    eprintln!(
                        "[{log_tag}] '{target}': base sound fold refused error={error:?}; \
                         keeping reference"
                    );
                }
                continue;
            }
        };
        let base = match root_joint_accept_sound_fold(base_fold_result, n_rows, deadline) {
            Some(result) => result,
            None => {
                if probe {
                    eprintln!(
                        "[{log_tag}] '{target}': base sound fold result rejected \
                         expected_rows={n_rows} deadline_live={}; keeping reference",
                        !past_deadline(),
                    );
                }
                continue;
            }
        };
        let mut best_l = base.lower_bounds.clone();
        let mut best_u = base.upper_bounds.clone();
        let obj0: f32 = best_l.iter().filter(|v| v.is_finite()).sum();
        let mut best_obj = obj0;

        // Joint Adam ascent on the below-L α (summed lower objective).
        let mut slopes = base_slopes.clone();
        let mut adam = AlphaAdam::new(&base_slopes);
        for t in 1..=iters {
            if past_deadline() {
                break;
            }
            write_alpha_prime(&mut work_segs, &slopes, &stepped);
            let gradient_result = match deadline {
                Some(deadline) => gpu.crown_joint_alpha_gradient_resident_with_deadline(
                    &work_segs,
                    seed.lower_a.as_ref(),
                    n_rows,
                    pre_dim,
                    &prep.in_lo,
                    &prep.in_hi,
                    deadline,
                ),
                None => gpu.crown_joint_alpha_gradient_resident(
                    &work_segs,
                    seed.lower_a.as_ref(),
                    n_rows,
                    pre_dim,
                    &prep.in_lo,
                    &prep.in_hi,
                ),
            };
            let grads = match root_joint_accept_gradient(gradient_result, &stepped, deadline) {
                Some(gradients) => gradients,
                None => break, // fail-closed: keep best-so-far (sound)
            };
            let lr_t = lr * 0.98f32.powi((t - 1) as i32);
            let max_g = adam.step(&mut slopes, &grads, &stepped, lr_t, t);
            if max_g == 0.0 || !max_g.is_finite() {
                break;
            }
            write_alpha_prime(&mut work_segs, &slopes, &stepped);
            let r = match root_joint_accept_sound_fold(sound_fold(&work_segs), n_rows, deadline) {
                Some(result) => result,
                None => break,
            };
            // Element-wise best across iterates: each fold is a valid enclosure,
            // so per-element max-l / min-u is a valid enclosure (intersection).
            let mut obj_t = 0.0f32;
            for i in 0..n_rows {
                let (l, u) = (r.lower_bounds[i], r.upper_bounds[i]);
                if l.is_finite() && l > best_l[i] {
                    best_l[i] = l;
                }
                if u.is_finite() && u < best_u[i] {
                    best_u[i] = u;
                }
                if l.is_finite() {
                    obj_t += l;
                }
            }
            if obj_t > best_obj {
                best_obj = obj_t;
            }
            if probe && (t % 10 == 0 || t == 1) {
                eprintln!(
                    "[{log_tag}] '{target}' iter={t} max|g|={max_g:.3e} \
                     obj={obj_t:.4} (obj0={obj0:.4} best={best_obj:.4})"
                );
            }
        }
        if past_deadline() {
            break;
        }

        // SHRINK-ONLY writeback via the alpha_explicit contract: build the refined
        // enclosure on the reference SHAPE, then intersection_per_element (union
        // fallback per disjoint element — still sound).
        let mut new_l = ref_l.clone();
        let mut new_u = ref_u.clone();
        for (r, &j) in sel.iter().enumerate() {
            if best_l[r].is_finite() {
                new_l[j] = new_l[j].max(best_l[r]);
            }
            if best_u[r].is_finite() {
                new_u[j] = new_u[j].min(best_u[r]);
            }
        }
        let shape = ref_bt.lower().raw_dim();
        let (Ok(la), Ok(ua)) = (
            ndarray::ArrayD::from_shape_vec(shape.clone(), new_l),
            ndarray::ArrayD::from_shape_vec(shape, new_u),
        ) else {
            continue;
        };
        let Ok(refined) = BoundedTensor::new(la, ua) else {
            continue; // crossed interval anywhere ⇒ keep the reference (sound)
        };
        let (tightened, disjoint) = ref_bt
            .intersection_per_element(&refined)
            .unwrap_or_else(|| (ref_bt.clone(), 0));
        if past_deadline() {
            break;
        }
        let ref_w: f32 =
            ref_l.iter().zip(&ref_u).map(|(&l, &u)| u - l).sum::<f32>() / pre_dim as f32;
        let new_w: f32 = tightened
            .lower()
            .iter()
            .zip(tightened.upper().iter())
            .map(|(&l, &u)| u - l)
            .sum::<f32>()
            / pre_dim as f32;
        let stop_tag = prep
            .stop_node
            .as_deref()
            .map(|m| format!(" stop_at='{m}' box_dim={}", prep.in_lo.len()))
            .unwrap_or_default();
        eprintln!(
            "[{log_tag}] '{target}': sel={n_rows}/{pre_dim} relus_below={n_relu} \
             meanw {ref_w:.5} -> {new_w:.5} (disjoint={disjoint}){stop_tag}"
        );
        // The return value tells the caller whether warmup α was computed from
        // now-stale boxes. Detect ANY strict endpoint shrink, rather than using
        // a mean-width display threshold that can hide a few changed rows in an
        // 8192-wide tensor.
        let strictly_tightened = tightened
            .lower()
            .iter()
            .zip(tightened.upper().iter())
            .zip(ref_l.iter().zip(ref_u.iter()))
            .any(|((&new_l, &new_u), (&old_l, &old_u))| new_l > old_l || new_u < old_u);
        // Keep the reference and do not report a stale-alpha signal if even the
        // bounded publication bookkeeping exhausts the slice.
        if past_deadline() {
            break;
        }
        if !root_joint_publish_tightened_bound(bounds, &target, tightened, deadline) {
            break;
        }
        if strictly_tightened {
            n_tightened += 1;
        }
    }
    n_tightened
}

/// Scoped target pre-activation (seed) node names for the root JOINT interm-α
/// pass (#root-joint-interm-alpha). Walks every ReLU in execution order and
/// selects its PRE-ACTIVATION node when (a) it is not the network input, (b) a
/// reference bound exists, (c) `dim ≤ NY_ROOT_JOINT_INTERM_ALPHA_MAX_DIM`
/// (default 2048 ⇒ the FC head + the last residual block on cifar100), and
/// (d) it has at least one crossing neuron (a fully-stable seed's relaxations
/// are already exact). `NY_ROOT_JOINT_INTERM_ALPHA_LAYERS` (comma-list of ReLU
/// node names, or "all") overrides the auto scope. Selection-only — never
/// affects soundness.
pub(in crate::beta_crown::engine::graph) fn scoped_joint_alpha_targets(
    graph: &GraphNetwork,
    bounds: &HashMap<String, BoundedTensor>,
) -> Vec<String> {
    let max_dim = root_joint_interm_alpha_max_dim(2048);
    let mut targets = joint_alpha_candidates_in_exec_order(graph, bounds, max_dim);
    // Deepest-first: later exec-order targets first, so downstream layers are
    // tightened before shallower ones are attempted within the grace slice.
    targets.reverse();
    targets
}

/// Default width admitted by the DEMAND-RANKED selector when the sound-GPU
/// deadline lane is armed (#root-joint-demand-rank).
///
/// The legacy 2048 default was sized for the CPU-affordable tail (the FC head
/// plus the last residual block on cifar100) and structurally EXCLUDES every
/// node the CROWN-IBP collector reports as "DEMANDED targets did not complete
/// CROWN" — the wide conv/BatchNorm pre-activations (28,800 dims on the
/// cgan/cifar100 exhibits) whose fallback-grade boxes poison the whole BaB
/// tree at −29 scale. 32,768 covers that measured class; the armed GPU lane's
/// per-target deadline loop (checked before each target and after prep) is
/// what bounds the actual work, so widening the SCOPE never widens the SLICE.
const ARMED_JOINT_ALPHA_DEFAULT_MAX_DIM: usize = 32_768;

/// Demand-ranked variant of [`scoped_joint_alpha_targets`] for the ARMED
/// sound-GPU deadline lane (#root-joint-demand-rank).
///
/// Same structural eligibility, two differences, both selection-only:
/// 1. the default `max_dim` is [`ARMED_JOINT_ALPHA_DEFAULT_MAX_DIM`] (an
///    explicit `NY_ROOT_JOINT_INTERM_ALPHA_MAX_DIM` still wins), so the
///    demanded wide targets are IN SCOPE at all;
/// 2. targets are ranked demanded-first via the collector's own demand
///    selector (`nodes_requiring_crown_tightening` — the SAME list whose
///    "6/7 DEMANDED targets did not complete CROWN" degradation this lane
///    exists to repair), deepest-first within each class. The runner consumes
///    targets in order under the pass deadline, so ranking IS the budget
///    policy: decisive nodes get the GPU slice first and the tail is cut off
///    by the deadline exactly as before.
///
/// Never affects soundness: a target that is selected but not tightened keeps
/// its reference bound bit-for-bit.
pub(in crate::beta_crown::engine::graph) fn scoped_joint_alpha_targets_demand_ranked(
    graph: &GraphNetwork,
    bounds: &HashMap<String, BoundedTensor>,
) -> Vec<String> {
    let max_dim = root_joint_interm_alpha_max_dim(ARMED_JOINT_ALPHA_DEFAULT_MAX_DIM);
    let candidates = joint_alpha_candidates_in_exec_order(graph, bounds, max_dim);
    let demanded: std::collections::HashSet<String> = graph.exec_order().map_or_else(
        |_| std::collections::HashSet::new(),
        |order| {
            let order: Vec<String> = order.to_vec();
            crate::network::nodes_requiring_crown_tightening(graph, &order, bounds)
        },
    );
    rank_joint_alpha_targets(candidates, |target| demanded.contains(target))
}

/// Resolve the shared explicit max-dimension override, retaining each
/// selector's own contextual fallback when the override is absent or rejected.
fn root_joint_interm_alpha_max_dim(contextual_default: usize) -> usize {
    ny_levers::read(&ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_MAX_DIM)
        .value
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or(contextual_default)
}

/// Pure ranking: demanded targets first, deepest-first within each class
/// (#root-joint-demand-rank). `candidates_in_exec_order` must be in execution
/// order; the stable sort preserves the deepest-first order inside both the
/// demanded and the non-demanded class.
fn rank_joint_alpha_targets(
    candidates_in_exec_order: Vec<String>,
    is_demanded: impl Fn(&str) -> bool,
) -> Vec<String> {
    let mut ranked = candidates_in_exec_order;
    ranked.reverse();
    ranked.sort_by_key(|target| !is_demanded(target));
    ranked
}

/// Shared structural eligibility walk for the joint interm-α selectors:
/// every ReLU pre-activation, in EXECUTION order, that (a) is not the network
/// input, (b) has a reference bound, (c) is no wider than `max_dim`, and
/// (d) has at least one crossing neuron. Extracted verbatim from
/// `scoped_joint_alpha_targets` so both selectors share one definition of
/// eligibility and differ only in scope default and ordering.
fn joint_alpha_candidates_in_exec_order(
    graph: &GraphNetwork,
    bounds: &HashMap<String, BoundedTensor>,
    max_dim: usize,
) -> Vec<String> {
    let Ok(order) = graph.exec_order() else {
        return Vec::new();
    };
    let order: Vec<String> = order.to_vec();
    let layers_env = std::env::var("NY_ROOT_JOINT_INTERM_ALPHA_LAYERS")
        .ok()
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    let select_all = layers_env.is_empty() || layers_env.eq_ignore_ascii_case("all");
    let explicit: std::collections::HashSet<String> = if select_all {
        std::collections::HashSet::new()
    } else {
        layers_env
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    };
    let has_crossing = |bt: &BoundedTensor| -> bool {
        bt.lower()
            .iter()
            .zip(bt.upper().iter())
            .any(|(&l, &u)| l < 0.0 && u > 0.0)
    };
    let mut targets: Vec<String> = Vec::new();
    for name in &order {
        let Some(node) = graph.nodes.get(name) else {
            continue;
        };
        if !matches!(node.layer(), Layer::ReLU(_)) {
            continue;
        }
        if !select_all && !explicit.contains(name.as_str()) {
            continue;
        }
        let Some(pre) = node.inputs().first() else {
            continue;
        };
        if pre == NETWORK_INPUT {
            continue;
        }
        let Some(ref_bt) = bounds.get(pre) else {
            continue;
        };
        let pre_dim = ref_bt.lower().len();
        if pre_dim == 0 || pre_dim > max_dim {
            continue;
        }
        if !has_crossing(ref_bt) {
            continue;
        }
        if !targets.iter().any(|t| t == pre) {
            targets.push(pre.clone());
        }
    }
    targets
}

/// Ranking record for the default-dark one-target wide demanded root fold.
///
/// `crossing_mass` is the number of finite rows whose current interval crosses
/// zero. `downstream_relu_leverage` is the number of reachable ReLU layers that
/// are themselves currently unstable. Their saturating product is the primary
/// selection key; the remaining fields make ties deterministic and prefer the
/// candidate that exposes more executable work, then the deeper target.
#[derive(Debug, Clone, PartialEq, Eq)]
struct WideDemandedTargetImpact {
    name: String,
    crossing_mass: usize,
    downstream_relu_leverage: usize,
    pre_dim: usize,
    exec_index: usize,
}

impl WideDemandedTargetImpact {
    fn score(&self) -> usize {
        self.crossing_mass
            .saturating_mul(self.downstream_relu_leverage)
    }
}

#[cfg(test)]
fn rank_wide_demanded_target_impacts(
    mut candidates: Vec<WideDemandedTargetImpact>,
    max_targets: usize,
) -> Vec<WideDemandedTargetImpact> {
    candidates.sort_by(wide_demanded_target_impact_order);
    candidates.truncate(max_targets);
    candidates
}

fn wide_demanded_target_impact_order(
    a: &WideDemandedTargetImpact,
    b: &WideDemandedTargetImpact,
) -> std::cmp::Ordering {
    b.score()
        .cmp(&a.score())
        .then_with(|| b.crossing_mass.cmp(&a.crossing_mass))
        .then_with(|| b.downstream_relu_leverage.cmp(&a.downstream_relu_leverage))
        .then_with(|| b.pre_dim.cmp(&a.pre_dim))
        .then_with(|| b.exec_index.cmp(&a.exec_index))
        .then_with(|| a.name.cmp(&b.name))
}

fn retain_wide_demanded_candidate(
    candidates: &mut Vec<WideDemandedTargetImpact>,
    candidate: WideDemandedTargetImpact,
    capacity: usize,
) -> Option<()> {
    if capacity == 0 {
        return Some(());
    }
    if candidates.len() < capacity {
        candidates.try_reserve(1).ok()?;
        candidates.push(candidate);
    } else if wide_demanded_target_impact_order(&candidate, candidates.last()?).is_lt() {
        candidates.pop();
        candidates.push(candidate);
    } else {
        return Some(());
    }
    // `capacity` is the fixed max-preflight count (currently eight), so this
    // sort is a bounded work unit even on a malformed enormous graph.
    candidates.sort_by(wide_demanded_target_impact_order);
    Some(())
}

fn finite_crossing_mass_before(bounds: &BoundedTensor, deadline: Option<Instant>) -> Option<usize> {
    let mut crossings = 0usize;
    for (index, (&lower, &upper)) in bounds.lower().iter().zip(bounds.upper().iter()).enumerate() {
        if index.is_multiple_of(4096) && deadline.is_some_and(|limit| Instant::now() >= limit) {
            return None;
        }
        crossings +=
            usize::from(lower.is_finite() && upper.is_finite() && lower < 0.0 && upper > 0.0);
    }
    deadline
        .is_none_or(|limit| Instant::now() < limit)
        .then_some(crossings)
}

fn bounds_are_concrete_before(bounds: &BoundedTensor, deadline: Option<Instant>) -> Option<bool> {
    for (index, (&lower, &upper)) in bounds.lower().iter().zip(bounds.upper().iter()).enumerate() {
        if index.is_multiple_of(4096) && deadline.is_some_and(|limit| Instant::now() >= limit) {
            return None;
        }
        if lower != upper {
            return Some(false);
        }
    }
    deadline
        .is_none_or(|limit| Instant::now() < limit)
        .then_some(true)
}

fn demanded_intermediate_nodes_before(
    graph: &GraphNetwork,
    order: &[String],
    bounds: &HashMap<String, BoundedTensor>,
    deadline: Option<Instant>,
) -> Option<std::collections::HashSet<String>> {
    let mut demanded = std::collections::HashSet::new();
    demanded.try_reserve(order.len()).ok()?;
    let output = if graph.output_name().is_empty() {
        order.last().map(String::as_str)
    } else {
        Some(graph.output_name())
    };
    if let Some(output) = output.filter(|name| *name != NETWORK_INPUT) {
        demanded.insert(output.to_string());
    }
    for (index, name) in order.iter().enumerate() {
        if index.is_multiple_of(64) && deadline.is_some_and(|limit| Instant::now() >= limit) {
            return None;
        }
        let Some(node) = graph.nodes.get(name) else {
            continue;
        };
        for &input_index in node.layer().required_input_bound_indices() {
            let Some(input_name) = node.inputs().get(input_index) else {
                continue;
            };
            if input_name == NETWORK_INPUT {
                continue;
            }
            if let Some(input_bounds) = bounds.get(input_name) {
                if bounds_are_concrete_before(input_bounds, deadline)? {
                    continue;
                }
            }
            demanded.insert(input_name.clone());
        }
    }
    deadline
        .is_none_or(|limit| Instant::now() < limit)
        .then_some(demanded)
}

fn downstream_unstable_relu_leverage(
    graph: &GraphNetwork,
    order: &[String],
    bounds: &HashMap<String, BoundedTensor>,
    target: &str,
    deadline: Option<Instant>,
) -> Option<usize> {
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        return None;
    }
    let mut descendants = std::collections::HashSet::new();
    descendants.try_reserve(order.len()).ok()?;
    let mut leverage = 0usize;
    for (index, name) in order.iter().enumerate() {
        if index.is_multiple_of(64) && deadline.is_some_and(|limit| Instant::now() >= limit) {
            return None;
        }
        let Some(node) = graph.nodes.get(name) else {
            continue;
        };
        let affected = name == target
            || node
                .inputs()
                .iter()
                .any(|input| descendants.contains(input.as_str()));
        if !affected {
            continue;
        }
        descendants.insert(name.clone());
        if !matches!(node.layer(), Layer::ReLU(_)) {
            continue;
        }
        let Some(pre_bounds) = node.inputs().first().and_then(|pre| bounds.get(pre)) else {
            continue;
        };
        leverage += usize::from(finite_crossing_mass_before(pre_bounds, deadline)? > 0);
    }
    deadline
        .is_none_or(|limit| Instant::now() < limit)
        .then_some(leverage)
}

/// Ranking record for the typed intermediate sweep. Unlike the legacy sparse
/// and crossing-only wide selectors, this measures all finite non-point width:
/// a sign-stable box can still have substantial CROWN headroom and can be
/// consumed downstream. Crossing mass remains a ranking component, not an
/// eligibility gate.
struct WideDemandedSweepTargetImpact {
    name: String,
    total_width: f64,
    crossing_mass: usize,
    nonpoint_rows: usize,
    downstream_relu_leverage: usize,
    pre_dim: usize,
    exec_index: usize,
}

impl WideDemandedSweepTargetImpact {
    fn width_leverage_score(&self) -> f64 {
        self.total_width * (1.0 + self.downstream_relu_leverage as f64)
    }
}

fn finite_width_summary_before(
    bounds: &BoundedTensor,
    deadline: Option<Instant>,
) -> Option<(f64, usize, usize)> {
    let mut total_width = 0.0f64;
    let mut crossing_mass = 0usize;
    let mut nonpoint_rows = 0usize;
    for (index, (&lower, &upper)) in bounds.lower().iter().zip(bounds.upper()).enumerate() {
        if index.is_multiple_of(4096) && deadline.is_some_and(|limit| Instant::now() >= limit) {
            return None;
        }
        if !lower.is_finite() || !upper.is_finite() || lower > upper {
            return None;
        }
        if lower < upper {
            total_width += f64::from(upper) - f64::from(lower);
            nonpoint_rows += 1;
            crossing_mass += usize::from(lower < 0.0 && upper > 0.0);
        }
    }
    (total_width.is_finite() && deadline.is_none_or(|limit| Instant::now() < limit)).then_some((
        total_width,
        crossing_mass,
        nonpoint_rows,
    ))
}

fn wide_demanded_sweep_target_impact_order(
    a: &WideDemandedSweepTargetImpact,
    b: &WideDemandedSweepTargetImpact,
) -> std::cmp::Ordering {
    b.width_leverage_score()
        .total_cmp(&a.width_leverage_score())
        .then_with(|| b.crossing_mass.cmp(&a.crossing_mass))
        .then_with(|| b.nonpoint_rows.cmp(&a.nonpoint_rows))
        .then_with(|| b.total_width.total_cmp(&a.total_width))
        .then_with(|| b.downstream_relu_leverage.cmp(&a.downstream_relu_leverage))
        .then_with(|| b.pre_dim.cmp(&a.pre_dim))
        .then_with(|| b.exec_index.cmp(&a.exec_index))
        .then_with(|| a.name.cmp(&b.name))
}

fn retain_wide_demanded_sweep_candidate(
    candidates: &mut Vec<WideDemandedSweepTargetImpact>,
    candidate: WideDemandedSweepTargetImpact,
    capacity: usize,
) -> Option<()> {
    if capacity == 0 {
        return Some(());
    }
    if candidates.len() < capacity {
        candidates.try_reserve(1).ok()?;
        candidates.push(candidate);
    } else if wide_demanded_sweep_target_impact_order(&candidate, candidates.last()?).is_lt() {
        candidates.pop();
        candidates.push(candidate);
    } else {
        return Some(());
    }
    candidates.sort_by(wide_demanded_sweep_target_impact_order);
    Some(())
}

/// Rank demanded wide pre-activations for the typed GPU sweep.
///
/// Eligibility deliberately includes finite sign-stable non-point tensors. The
/// measured CIFAR root payoff includes Conv_37/Conv_43 even though their frozen
/// boxes have zero crossing rows, so inheriting the sparse selector's crossing
/// gate would repeat its known inertness.
pub(super) fn scoped_wide_demanded_sweep_targets_before(
    graph: &GraphNetwork,
    bounds: &HashMap<String, BoundedTensor>,
    min_dim: usize,
    max_dim: usize,
    max_targets: usize,
    deadline: Option<Instant>,
) -> Option<Vec<String>> {
    if min_dim == 0 || min_dim > max_dim || max_targets == 0 {
        return Some(Vec::new());
    }
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        return None;
    }
    let order = match deadline {
        Some(_) => graph.cached_exec_order.get().map(Vec::as_slice)?,
        None => graph.exec_order().ok()?,
    };
    let demanded = demanded_intermediate_nodes_before(graph, order, bounds, deadline)?;
    let mut seen = std::collections::HashSet::new();
    let mut candidates = Vec::new();
    candidates.try_reserve_exact(max_targets).ok()?;
    for (exec_index, name) in order.iter().enumerate() {
        if exec_index.is_multiple_of(64) && deadline.is_some_and(|limit| Instant::now() >= limit) {
            return None;
        }
        let Some(node) = graph.nodes.get(name) else {
            continue;
        };
        if !matches!(node.layer(), Layer::ReLU(_)) {
            continue;
        }
        let Some(pre) = node.inputs().first() else {
            continue;
        };
        if pre == NETWORK_INPUT
            || !demanded.contains(pre)
            || !seen.insert(pre.clone())
            || graph
                .nodes
                .get(pre)
                .is_some_and(|producer| matches!(producer.layer(), Layer::Linear(_)))
        {
            continue;
        }
        let Some(ref_bt) = bounds.get(pre) else {
            continue;
        };
        let pre_dim = ref_bt.len();
        if pre_dim < min_dim || pre_dim > max_dim {
            continue;
        }
        let (total_width, crossing_mass, nonpoint_rows) =
            finite_width_summary_before(ref_bt, deadline)?;
        if nonpoint_rows == 0 {
            continue;
        }
        let downstream_relu_leverage =
            downstream_unstable_relu_leverage(graph, order, bounds, pre, deadline)?;
        retain_wide_demanded_sweep_candidate(
            &mut candidates,
            WideDemandedSweepTargetImpact {
                name: pre.clone(),
                total_width,
                crossing_mass,
                nonpoint_rows,
                downstream_relu_leverage,
                pre_dim,
                exec_index,
            },
            max_targets,
        )?;
    }
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        return None;
    }
    Some(
        candidates
            .into_iter()
            .map(|candidate| candidate.name)
            .collect(),
    )
}

/// Select at most `max_targets` high-impact wide demanded pre-activations.
///
/// Eligibility is the existing sparse root fold's structural scope (a ReLU
/// pre-activation with a frozen box, excluding the separately owned Linear
/// head), narrowed to the collector's own demanded set and the explicit
/// dimension window. Ranking spends the bounded slice on crossing unstable
/// mass × downstream unstable-ReLU leverage. This function only chooses work;
/// the certified fold and shrink-only publication remain unchanged.
#[cfg(test)]
pub(in crate::beta_crown::engine::graph) fn scoped_wide_demanded_crown_targets(
    graph: &GraphNetwork,
    bounds: &HashMap<String, BoundedTensor>,
    min_dim: usize,
    max_dim: usize,
    max_targets: usize,
) -> Vec<String> {
    scoped_wide_demanded_crown_targets_before(graph, bounds, min_dim, max_dim, max_targets, None)
        .unwrap_or_default()
}

fn scoped_wide_demanded_crown_targets_before(
    graph: &GraphNetwork,
    bounds: &HashMap<String, BoundedTensor>,
    min_dim: usize,
    max_dim: usize,
    max_targets: usize,
    deadline: Option<Instant>,
) -> Option<Vec<String>> {
    if min_dim == 0 || min_dim > max_dim || max_targets == 0 {
        return Some(Vec::new());
    }
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        return None;
    }
    let order = match deadline {
        Some(_) => graph.cached_exec_order.get().map(Vec::as_slice)?,
        None => graph.exec_order().ok()?,
    };
    let demanded = demanded_intermediate_nodes_before(graph, order, bounds, deadline)?;
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        return None;
    }
    let mut seen = std::collections::HashSet::new();
    let mut candidates = Vec::new();
    candidates.try_reserve_exact(max_targets).ok()?;
    for (exec_index, name) in order.iter().enumerate() {
        if exec_index.is_multiple_of(64) && deadline.is_some_and(|limit| Instant::now() >= limit) {
            return None;
        }
        let Some(node) = graph.nodes.get(name) else {
            continue;
        };
        if !matches!(node.layer(), Layer::ReLU(_)) {
            continue;
        }
        let Some(pre) = node.inputs().first() else {
            continue;
        };
        if pre == NETWORK_INPUT
            || !demanded.contains(pre)
            || !seen.insert(pre.clone())
            || graph
                .nodes
                .get(pre)
                .is_some_and(|producer| matches!(producer.layer(), Layer::Linear(_)))
        {
            continue;
        }
        let Some(ref_bt) = bounds.get(pre) else {
            continue;
        };
        let pre_dim = ref_bt.lower().len();
        if pre_dim < min_dim || pre_dim > max_dim {
            continue;
        }
        let crossing_mass = finite_crossing_mass_before(ref_bt, deadline)?;
        if crossing_mass == 0 {
            continue;
        }
        let downstream_relu_leverage =
            downstream_unstable_relu_leverage(graph, order, bounds, pre, deadline)?;
        if downstream_relu_leverage == 0 {
            continue;
        }
        retain_wide_demanded_candidate(
            &mut candidates,
            WideDemandedTargetImpact {
                name: pre.clone(),
                crossing_mass,
                downstream_relu_leverage,
                pre_dim,
                exec_index,
            },
            max_targets,
        )?;
    }
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        return None;
    }
    Some(
        candidates
            .into_iter()
            .map(|candidate| candidate.name)
            .collect(),
    )
}

#[cfg(test)]
fn select_preparable_wide_target_names(
    ranked: Vec<String>,
    max_targets: usize,
    max_preflights: usize,
    mut is_preparable: impl FnMut(&str) -> bool,
) -> Vec<String> {
    let mut selected = Vec::new();
    for target in ranked.into_iter().take(max_preflights) {
        if is_preparable(&target) {
            selected.push(target);
            if selected.len() == max_targets {
                break;
            }
        }
    }
    selected
}

/// Prepare ranked wide-demanded candidates through the exact extraction used by
/// the sound fold, so the one execution slot is not wasted on an unrepresentable
/// target. The resulting preparation is moved into execution rather than
/// recomputed. Refusals are selection-only and no preparation grants proof
/// authority: the backend and deadline are checked again before dispatch.
#[allow(clippy::too_many_arguments)]
pub(in crate::beta_crown::engine::graph) fn scoped_preparable_wide_demanded_crown_targets(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    bounds: &HashMap<String, BoundedTensor>,
    min_dim: usize,
    max_dim: usize,
    max_targets: usize,
    max_preflights: usize,
    deadline: Instant,
) -> Vec<PreparedWideDemandedTarget> {
    if max_targets == 0 || max_preflights == 0 {
        return Vec::new();
    }
    let Some(ranked) = scoped_wide_demanded_crown_targets_before(
        graph,
        bounds,
        min_dim,
        max_dim,
        max_preflights,
        Some(deadline),
    ) else {
        eprintln!("[root-wide-demanded-interm-crown] ranking deadline expired; no target selected");
        return Vec::new();
    };
    let mut selected = Vec::new();
    if selected.try_reserve_exact(max_targets).is_err() {
        return Vec::new();
    }
    for target in ranked.into_iter().take(max_preflights) {
        if Instant::now() >= deadline {
            eprintln!(
                "[root-wide-demanded-interm-crown] preflight deadline expired before \
                 target='{target}'"
            );
            return Vec::new();
        }
        let Some(prep) = prep_resnet_domain_ext(
            graph, &target, bounds, input, None, None, false, false, false,
        ) else {
            eprintln!(
                "[root-wide-demanded-interm-crown] preflight target='{target}' \
                 refused reason={}",
                super::prep_resnet_domain_last_refusal(),
            );
            continue;
        };
        if Instant::now() >= deadline {
            return Vec::new();
        }
        selected.push(PreparedWideDemandedTarget { name: target, prep });
        if selected.len() == max_targets {
            break;
        }
    }
    selected
}

/// Structurally select convolutional/residual pre-activations for the bounded
/// root sparse-row CROWN pass.
///
/// Selection is deliberately independent of exporter node names: every ReLU
/// pre-activation with a frozen sound box, at least one crossing row, and width
/// no larger than `max_dim` is eligible except a `Linear` producer (the existing
/// dense-head pass owns that disjoint scope). Targets are processed
/// deepest-first and capped before any identity seed allocation. `max_rows == 0`
/// disables selection because the caller could seed no rows.
pub(in crate::beta_crown::engine::graph) fn scoped_sparse_crown_targets(
    graph: &GraphNetwork,
    bounds: &HashMap<String, BoundedTensor>,
    max_dim: usize,
    max_rows: usize,
    max_targets: usize,
) -> Vec<String> {
    if max_dim == 0 || max_rows == 0 || max_targets == 0 {
        return Vec::new();
    }
    let Ok(order) = graph.exec_order() else {
        return Vec::new();
    };
    let mut targets = Vec::new();
    // #target-census: this enumeration yields FIVE candidates on a ~20-ReLU
    // cifar100 resnet, and none of the knobs explain it — not `max_dim` (raising
    // it to 262_144 changed nothing), not `max_targets` (24 still yields 5), not
    // `max_rows` (512, unclamped). Two successive guesses at the cause were
    // wrong, so record WHICH filter rejects each ReLU instead of guessing a
    // third time. Print-only, behind the existing phase gate.
    let census = crate::phase_telemetry::phase_telemetry_enabled();
    let mut rej_not_relu = 0usize;
    let mut rej_no_input = 0usize;
    let mut rej_producer = 0usize;
    let mut rej_no_frozen_box = 0usize;
    let mut rej_dim = 0usize;
    let mut rej_no_crossing = 0usize;
    let mut relus_seen = 0usize;
    for name in order {
        let Some(node) = graph.nodes.get(name) else {
            continue;
        };
        if !matches!(node.layer(), Layer::ReLU(_)) {
            rej_not_relu += 1;
            continue;
        }
        relus_seen += 1;
        let Some(pre) = node.inputs().first() else {
            rej_no_input += 1;
            continue;
        };
        if pre == NETWORK_INPUT
            || graph
                .nodes
                .get(pre)
                .is_some_and(|producer| matches!(producer.layer(), Layer::Linear(_)))
        {
            rej_producer += 1;
            continue;
        }
        if bounds.get(pre).is_none() {
            rej_no_frozen_box += 1;
            if census {
                eprintln!("[target-census] '{pre}': REJECT no-frozen-box");
            }
        }
        let Some(ref_bt) = bounds.get(pre) else {
            continue;
        };
        let pre_dim = ref_bt.lower().len();
        if pre_dim == 0 || pre_dim > max_dim {
            rej_dim += 1;
            if census {
                eprintln!("[target-census] '{pre}': REJECT dim {pre_dim} > max_dim {max_dim}");
            }
            continue;
        }
        let has_crossing = ref_bt
            .lower()
            .iter()
            .zip(ref_bt.upper().iter())
            .any(|(&l, &u)| l.is_finite() && u.is_finite() && l < 0.0 && u > 0.0);
        if has_crossing && !targets.iter().any(|target| target == pre) {
            targets.push(pre.clone());
        } else if !has_crossing {
            rej_no_crossing += 1;
            if census {
                let crossing = ref_bt
                    .lower()
                    .iter()
                    .zip(ref_bt.upper().iter())
                    .filter(|(&l, &u)| l.is_finite() && u.is_finite() && l < 0.0 && u > 0.0)
                    .count();
                eprintln!(
                    "[target-census] '{pre}': REJECT no-crossing (dim {pre_dim}, crossing {crossing})"
                );
            }
        }
    }
    if census {
        eprintln!(
            "[target-census] relus={relus_seen} eligible={} | rejects: no_input={rej_no_input} \
             producer(linear/input)={rej_producer} no_frozen_box={rej_no_frozen_box} \
             dim>{max_dim}={rej_dim} no_crossing={rej_no_crossing} (non-relu nodes={rej_not_relu})",
            targets.len(),
        );
    }
    targets.reverse();
    targets.truncate(max_targets);
    targets
}

#[cfg(test)]
mod root_sparse_target_tests {
    use super::*;
    use crate::layers::{AddLayer, LinearLayer, ReLULayer};
    use crate::GraphNode;
    use ndarray::{arr1, arr2, ArrayD, IxDyn};

    fn crossing_box(dim: usize) -> BoundedTensor {
        BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[dim]), -1.0),
            ArrayD::from_elem(IxDyn(&[dim]), 1.0),
        )
        .unwrap()
    }

    fn linear(name: &str, input: &str) -> GraphNode {
        let layer = Layer::Linear(
            LinearLayer::new(
                arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]),
                Some(arr1(&[0.0_f32, 0.0])),
            )
            .unwrap(),
        );
        if input == NETWORK_INPUT {
            GraphNode::from_input(name, layer)
        } else {
            GraphNode::new(name, layer, vec![input.to_string()])
        }
    }

    fn relu(name: &str, input: &str) -> GraphNode {
        GraphNode::new(name, Layer::ReLU(ReLULayer), vec![input.to_string()])
    }

    #[test]
    fn sparse_targets_are_deepest_first_capped_and_exclude_dense_heads() {
        let mut graph = GraphNetwork::new();
        graph.add_node(linear("stem", NETWORK_INPUT));
        graph.add_node(GraphNode::new(
            "res0",
            Layer::Add(AddLayer),
            vec!["stem".into(), "stem".into()],
        ));
        graph.add_node(relu("relu0", "res0"));
        graph.add_node(GraphNode::new(
            "res1",
            Layer::Add(AddLayer),
            vec!["relu0".into(), "stem".into()],
        ));
        graph.add_node(relu("relu1", "res1"));
        graph.add_node(linear("head", "relu1"));
        graph.add_node(relu("relu_head", "head"));
        graph.set_output("relu_head");

        let bounds = HashMap::from([
            ("res0".to_string(), crossing_box(2)),
            ("res1".to_string(), crossing_box(2)),
            ("head".to_string(), crossing_box(2)),
        ]);
        assert_eq!(
            scoped_sparse_crown_targets(&graph, &bounds, 2, 2, 4),
            vec!["res1".to_string(), "res0".to_string()]
        );
        assert_eq!(
            scoped_sparse_crown_targets(&graph, &bounds, 2, 2, 1),
            vec!["res1".to_string()]
        );
        assert!(scoped_sparse_crown_targets(&graph, &bounds, 1, 2, 4).is_empty());
        assert!(scoped_sparse_crown_targets(&graph, &bounds, 2, 0, 4).is_empty());
    }
}

#[cfg(test)]
mod wide_demanded_target_tests {
    use super::*;
    use crate::layers::{AddLayer, LinearLayer, ReLULayer};
    use crate::GraphNode;
    use ndarray::{arr1, arr2, ArrayD, IxDyn};

    fn box_with_crossings(dim: usize, crossings: usize) -> BoundedTensor {
        let mut lower = vec![1.0; dim];
        let mut upper = vec![2.0; dim];
        for index in 0..crossings.min(dim) {
            lower[index] = -1.0;
            upper[index] = 1.0;
        }
        BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[dim]), lower).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[dim]), upper).unwrap(),
        )
        .unwrap()
    }

    fn linear(name: &str, input: &str) -> GraphNode {
        let layer = Layer::Linear(
            LinearLayer::new(
                arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]),
                Some(arr1(&[0.0_f32, 0.0])),
            )
            .unwrap(),
        );
        if input == NETWORK_INPUT {
            GraphNode::from_input(name, layer)
        } else {
            GraphNode::new(name, layer, vec![input.to_string()])
        }
    }

    fn relu(name: &str, input: &str) -> GraphNode {
        GraphNode::new(name, Layer::ReLU(ReLULayer), vec![input.to_string()])
    }

    #[test]
    fn impact_ranking_uses_product_then_deterministic_ties() {
        let candidates = vec![
            WideDemandedTargetImpact {
                name: "deep".into(),
                crossing_mass: 5,
                downstream_relu_leverage: 2,
                pre_dim: 9,
                exec_index: 9,
            },
            WideDemandedTargetImpact {
                name: "leveraged".into(),
                crossing_mass: 4,
                downstream_relu_leverage: 3,
                pre_dim: 8,
                exec_index: 2,
            },
            WideDemandedTargetImpact {
                name: "tie".into(),
                crossing_mass: 6,
                downstream_relu_leverage: 2,
                pre_dim: 7,
                exec_index: 7,
            },
        ];
        let ranked = rank_wide_demanded_target_impacts(candidates, 2);
        assert_eq!(
            ranked
                .iter()
                .map(|candidate| candidate.name.as_str())
                .collect::<Vec<_>>(),
            vec!["tie", "leveraged"],
            "equal products prefer more crossing mass; the lower product is capped"
        );
    }

    #[test]
    fn production_streaming_top_k_is_identical_to_the_full_reference_sort() {
        let candidates: Vec<_> = (0..37)
            .map(|index| WideDemandedTargetImpact {
                name: format!("candidate_{index:02}"),
                crossing_mass: 1 + (index * 17) % 23,
                downstream_relu_leverage: 1 + (index * 11) % 13,
                pre_dim: 2_049 + (index * 997) % 30_720,
                exec_index: index,
            })
            .collect();
        let expected = rank_wide_demanded_target_impacts(candidates.clone(), 8);
        let mut streamed = Vec::new();
        for candidate in candidates {
            retain_wide_demanded_candidate(&mut streamed, candidate, 8)
                .expect("bounded top-k retention must not allocate beyond its fixed capacity");
        }
        assert_eq!(streamed, expected);
    }

    #[test]
    fn preparability_filter_skips_refusals_and_keeps_one_execution_target() {
        let ranked = ["unpreppable", "winner", "tail"]
            .into_iter()
            .map(str::to_string)
            .collect();
        let selected =
            select_preparable_wide_target_names(ranked, 1, 3, |target| target != "unpreppable");
        assert_eq!(selected, vec!["winner".to_string()]);

        let ranked = ["a", "b", "winner"]
            .into_iter()
            .map(str::to_string)
            .collect();
        assert!(
            select_preparable_wide_target_names(ranked, 1, 2, |target| target == "winner")
                .is_empty(),
            "preflight count is a hard bound even when a later target could prepare"
        );
    }

    #[test]
    fn owned_preparation_handoff_never_reinvokes_the_extractor() {
        let extraction_called = std::cell::Cell::new(false);
        let value = prepared_or_extract(Some(7usize), || {
            extraction_called.set(true);
            Some(9)
        });
        assert_eq!(value, Some(7));
        assert!(!extraction_called.get());

        let value = prepared_or_extract(None, || {
            extraction_called.set(true);
            Some(9)
        });
        assert_eq!(value, Some(9));
        assert!(extraction_called.get());
    }

    #[test]
    fn host_seed_admission_counts_arc_conversion_peak_and_overflow() {
        assert!(
            root_interm_host_seed_peak_bytes(512, 8_192).unwrap() < ROOT_INTERM_MAX_HOST_SEED_BYTES,
            "the measured CIFAR 8,192-wide slice remains admissible"
        );
        assert!(
            root_interm_host_seed_peak_bytes(512, 32_768).unwrap()
                > ROOT_INTERM_MAX_HOST_SEED_BYTES,
            "the dimension/row maxima may not combine into an uncharged 192 MiB transient"
        );
        assert!(root_interm_host_seed_peak_bytes(usize::MAX, usize::MAX).is_none());
    }

    #[test]
    fn selector_chooses_one_demanded_target_by_crossing_mass_times_leverage() {
        let mut graph = GraphNetwork::new();
        graph.add_node(linear("stem", NETWORK_INPUT));
        graph.add_node(GraphNode::new(
            "wide0",
            Layer::Add(AddLayer),
            vec!["stem".into(), "stem".into()],
        ));
        graph.add_node(relu("relu0", "wide0"));
        graph.add_node(GraphNode::new(
            "wide1",
            Layer::Add(AddLayer),
            vec!["relu0".into(), "stem".into()],
        ));
        graph.add_node(relu("relu1", "wide1"));
        graph.add_node(GraphNode::new(
            "wide2",
            Layer::Add(AddLayer),
            vec!["relu1".into(), "stem".into()],
        ));
        graph.add_node(relu("relu2", "wide2"));
        graph.set_output("relu2");

        let bounds = HashMap::from([
            ("wide0".to_string(), box_with_crossings(8, 4)),
            ("wide1".to_string(), box_with_crossings(8, 3)),
            ("wide2".to_string(), box_with_crossings(8, 5)),
        ]);
        assert_eq!(
            scoped_wide_demanded_crown_targets(&graph, &bounds, 3, 8, 1),
            vec!["wide0".to_string()],
            "wide0 wins: 4 crossings × 3 unstable ReLU descendants = 12"
        );
        assert!(
            scoped_wide_demanded_crown_targets(&graph, &bounds, 9, 32, 1).is_empty(),
            "the dimension window is a hard selector bound"
        );
        assert!(
            scoped_wide_demanded_crown_targets(&graph, &bounds, 3, 8, 0).is_empty(),
            "zero target capacity must fail closed"
        );
        assert!(
            scoped_wide_demanded_crown_targets_before(
                &graph,
                &bounds,
                3,
                8,
                1,
                Some(Instant::now()),
            )
            .is_none(),
            "an expired ranking slice must return no partial candidate set"
        );
    }

    #[test]
    fn typed_sweep_selector_keeps_zero_crossing_wide_nonpoint_targets() {
        let mut graph = GraphNetwork::new();
        graph.add_node(linear("stem", NETWORK_INPUT));
        graph.add_node(GraphNode::new(
            "stable_wide",
            Layer::Add(AddLayer),
            vec!["stem".into(), "stem".into()],
        ));
        graph.add_node(relu("relu0", "stable_wide"));
        graph.set_output("relu0");
        let bounds = HashMap::from([("stable_wide".to_string(), box_with_crossings(8, 0))]);
        graph
            .exec_order()
            .expect("cache deterministic execution order");

        assert!(
            scoped_wide_demanded_crown_targets(&graph, &bounds, 3, 8, 1).is_empty(),
            "the legacy crossing-only route remains unchanged"
        );
        assert_eq!(
            scoped_wide_demanded_sweep_targets_before(
                &graph,
                &bounds,
                3,
                8,
                1,
                Some(Instant::now() + std::time::Duration::from_secs(2)),
            )
            .expect("live typed-sweep ranking"),
            vec!["stable_wide".to_string()],
            "a demanded finite non-point box remains eligible even with zero crossings"
        );
    }
}

#[cfg(test)]
mod joint_target_rank_tests {
    use super::*;
    use crate::layers::{LinearLayer, ReLULayer};
    use crate::GraphNode;
    use ndarray::{arr1, arr2, ArrayD, IxDyn};

    fn crossing_box(dim: usize) -> BoundedTensor {
        BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[dim]), -1.0),
            ArrayD::from_elem(IxDyn(&[dim]), 1.0),
        )
        .unwrap()
    }

    fn linear(name: &str, input: &str) -> GraphNode {
        let layer = Layer::Linear(
            LinearLayer::new(
                arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]),
                Some(arr1(&[0.0_f32, 0.0])),
            )
            .unwrap(),
        );
        if input == NETWORK_INPUT {
            GraphNode::from_input(name, layer)
        } else {
            GraphNode::new(name, layer, vec![input.to_string()])
        }
    }

    fn relu(name: &str, input: &str) -> GraphNode {
        GraphNode::new(name, Layer::ReLU(ReLULayer), vec![input.to_string()])
    }

    /// PIN (#root-joint-demand-rank): demanded targets come first, and the
    /// deepest-first order is preserved inside BOTH classes (stable sort).
    #[test]
    fn ranking_is_demanded_first_then_deepest_first_within_each_class() {
        let candidates: Vec<String> = ["a", "b", "c", "d"]
            .iter()
            .map(ToString::to_string)
            .collect();
        let demanded = |t: &str| t == "b" || t == "d";
        assert_eq!(
            rank_joint_alpha_targets(candidates, demanded),
            vec!["d".to_string(), "b".into(), "c".into(), "a".into()]
        );
    }

    /// PIN: with an empty demand set the ranking degenerates to EXACTLY the
    /// legacy deepest-first order — the fallback identity for the ranking.
    #[test]
    fn empty_demand_set_is_the_legacy_deepest_first_order() {
        let candidates: Vec<String> = ["a", "b", "c"].iter().map(ToString::to_string).collect();
        assert_eq!(
            rank_joint_alpha_targets(candidates, |_| false),
            vec!["c".to_string(), "b".into(), "a".into()]
        );
        let candidates: Vec<String> = ["a", "b", "c"].iter().map(ToString::to_string).collect();
        assert_eq!(
            rank_joint_alpha_targets(candidates, |_| true),
            vec!["c".to_string(), "b".into(), "a".into()],
            "all-demanded must equal none-demanded (class order only reorders ACROSS classes)"
        );
    }

    /// PIN: the armed selector's widened default admits the wide demanded
    /// class the legacy 2048 default structurally excludes, while the legacy
    /// selector is unchanged. Uses a >2048-dim pre-activation standing in for
    /// the measured 28,800-dim cgan/cifar100 exhibits.
    #[test]
    fn armed_selector_covers_the_wide_demanded_class_legacy_stays_narrow() {
        let mut graph = GraphNetwork::new();
        graph.add_node(linear("stem", NETWORK_INPUT));
        graph.add_node(relu("relu0", "stem"));
        graph.add_node(linear("wide", "relu0"));
        graph.add_node(relu("relu_wide", "wide"));
        graph.add_node(linear("head", "relu_wide"));
        graph.add_node(relu("relu_head", "head"));
        graph.set_output("relu_head");

        let bounds = HashMap::from([
            ("stem".to_string(), crossing_box(2)),
            ("wide".to_string(), crossing_box(4096)),
            ("head".to_string(), crossing_box(2)),
        ]);

        let legacy = scoped_joint_alpha_targets(&graph, &bounds);
        assert_eq!(
            legacy,
            vec!["head".to_string(), "stem".into()],
            "legacy scope must exclude the wide target and stay deepest-first"
        );

        let ranked = scoped_joint_alpha_targets_demand_ranked(&graph, &bounds);
        assert!(
            ranked.contains(&"wide".to_string()),
            "armed default must cover the wide demanded class, got {ranked:?}"
        );
        // Every eligible pre-activation here is demanded (ReLU consumers
        // require pre-activation bounds), so the ranked order is exactly the
        // deepest-first order over the widened scope.
        assert_eq!(
            ranked,
            vec!["head".to_string(), "wide".into(), "stem".into()]
        );
    }
}
