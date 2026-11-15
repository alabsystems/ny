// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #bbe-joint-root (ENGINE_PROGRAM §2 M5a): the margin-row side of the
//! UNTRUSTED joint-alpha root search.
//!
//! # What
//!
//! `margin_row.root_search: joint_f32` replaces the search half of
//! `#root-certify-once` — the Parity-scratch ascent (plus its optional
//! untrusted refresh rounds) — with abc's joint-alpha root recipe run by an
//! injected [`SearchBackend`] (`ny-bbe`'s CPU f32 backend in production): CROWN
//! init, spec prune, per-start-node alpha with an in-loop refresh of the
//! unstable rows, early exits. The certify half is UNCHANGED: the backend's
//! final slopes go through the existing ingest gate
//! (`root_certify_once::install_candidate_alpha`: finite, clamped to `[0, 1]`,
//! only on neurons unstable in the CERTIFIED boxes, `ms = max(alpha, s)`
//! re-derived), ONE certified Outward `root_eval` scores the candidate, and it
//! is kept only if it strictly beats the certified incumbent. `ascent` (the
//! declaration default) is today's phase, bit for bit: this module is not
//! entered and no backend is consulted.
//!
//! # The trust boundary
//!
//! * The backend reads [`lower_twin_net`]'s plain-data COPY of the lane's
//!   `TwinNet` and the certified boxes as REFERENCE boxes; nothing it returns
//!   is read by a verdict path except slopes, and those only through the
//!   ingest gate. The certified root, its boxes, chords and clip rows are never
//!   written from here.
//! * The backend is `&dyn SearchBackend` from `ny-core`. This crate never
//!   depends on `ny-bbe` (or on any CUDA crate); ny-cli registers the backend
//!   with [`set_root_search_backend_factory`]. Pinned by
//!   `crates/ny-bbe/tests/crate_dag_ratchet.rs`.
//! * A backend refusal, panic or missing registration is a REFUSAL: no
//!   candidate, the incumbent stands, `[bbe] ... refused=<why>`.
//!
//! # Observable
//!
//! Armed, every return path of the phase prints exactly one
//! `[bbe] phase=joint_root iters= search_s= certify_s= root_lb= ...` line
//! (after the `[root-certify-once] armed=1 ...` terminal line). `root_lb` is the
//! search's own UNTRUSTED f32 root bound (the program's S3 metric), never a
//! published number; the certified numbers are `incumbent=`, `certified=` and
//! `published=`. An armed `joint_f32` run without the line is VOID.

use std::sync::{Arc, OnceLock};
use std::time::Instant;

use ny_core::search::{
    ReluBox, RootAlphaHint, RootSearchRequest, SearchBackend, SearchConv, SearchNet, SearchOp,
    GATHER_PAD,
};

use super::net::{TwinNet, TwinOp};
use super::root::RootGates;

/// Which search runs inside `#root-certify-once` (`margin_row.root_search`).
///
/// The serde tokens are the typed preset key's values and the lever
/// declaration's `Enum` members ([`Self::TOKENS`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum RootSearch {
    /// Today's Parity-scratch ascent (`ascent`, the default): bit-identical.
    #[default]
    #[serde(rename = "ascent")]
    Ascent,
    /// The injected untrusted joint-alpha root search (`joint_f32`, M5a).
    #[serde(rename = "joint_f32")]
    JointF32,
}

impl RootSearch {
    /// The preset / environment vocabulary, in declaration order.
    pub const TOKENS: [&'static str; 2] = ["ascent", "joint_f32"];

    /// Parse one canonical token (exact spelling).
    #[must_use]
    pub fn from_token(token: &str) -> Option<Self> {
        match token {
            "ascent" => Some(Self::Ascent),
            "joint_f32" => Some(Self::JointF32),
            _ => None,
        }
    }

    /// The canonical token.
    #[must_use]
    pub const fn token(self) -> &'static str {
        match self {
            Self::Ascent => "ascent",
            Self::JointF32 => "joint_f32",
        }
    }
}

// ---------------------------------------------------------------------------
//  The injected backend
// ---------------------------------------------------------------------------

type BackendFactory = Box<dyn Fn() -> Option<Arc<dyn SearchBackend>> + Send + Sync>;

static FACTORY: OnceLock<BackendFactory> = OnceLock::new();

/// Register the process's root-search backend factory (ny-cli, once, at
/// startup). First install wins. The closure runs at most once, on the first
/// armed `joint_f32` root or breakpoint dual phase, so a dark lane constructs
/// nothing.
pub fn set_root_search_backend_factory<F>(factory: F)
where
    F: Fn() -> Option<Arc<dyn SearchBackend>> + Send + Sync + 'static,
{
    let _ = FACTORY.set(Box::new(factory));
}

/// The registered backend, built once per process on first use. `None` when
/// no factory is installed or it declined; an absent backend is a refusal of
/// the phase, never a verdict.
pub(crate) fn registered_backend() -> Option<Arc<dyn SearchBackend>> {
    static HANDLE: OnceLock<Arc<dyn SearchBackend>> = OnceLock::new();
    if let Some(backend) = HANDLE.get() {
        return Some(Arc::clone(backend));
    }
    let built = FACTORY.get().and_then(|factory| factory())?;
    Some(Arc::clone(HANDLE.get_or_init(|| built)))
}

// ---------------------------------------------------------------------------
//  The search copy of the lane IR
// ---------------------------------------------------------------------------

/// Lower the compiled twin-wall net into the backend's plain-data IR, op for
/// op (the same op indices, tensor ids, f64 weights and gather tables).
///
/// A LOSSLESS copy for SEARCH: the certified fold-error budgets are dropped
/// because nothing computed from this copy is published. Fails closed when a
/// gather index does not fit `u32` or the lowered sizes disagree with the net.
pub(crate) fn lower_twin_net(net: &TwinNet) -> Result<SearchNet, String> {
    let mut ops = Vec::with_capacity(net.ops.len());
    for (k, op) in net.ops.iter().enumerate() {
        ops.push(match op {
            TwinOp::Conv(c) => {
                let gather = c
                    .gather
                    .iter()
                    .map(|&g| {
                        if g == usize::MAX {
                            Ok(GATHER_PAD)
                        } else {
                            u32::try_from(g)
                                .ok()
                                .filter(|&v| v != GATHER_PAD)
                                .ok_or_else(|| format!("conv {k}: gather index {g} exceeds u32"))
                        }
                    })
                    .collect::<Result<Vec<u32>, String>>()?;
                SearchOp::Conv(Box::new(SearchConv {
                    input: c.input,
                    cout: c.kernel.0,
                    k: c.k_fwd,
                    p: c.oshape.1 * c.oshape.2,
                    gather,
                    weight: c.wmat.clone(),
                    bias: c.bias.clone(),
                }))
            }
            TwinOp::ChannelAffine {
                input,
                scale,
                shift,
                ..
            } => SearchOp::Affine {
                input: *input,
                scale: scale.clone(),
                shift: shift.clone(),
            },
            TwinOp::Relu { input, layer } => SearchOp::Relu {
                input: *input,
                head: *layer == usize::MAX,
            },
            TwinOp::Add { lhs, rhs } => SearchOp::Add {
                lhs: *lhs,
                rhs: *rhs,
            },
            TwinOp::Flatten { input } => SearchOp::Flatten { input: *input },
            TwinOp::Gemm {
                input,
                weight,
                bias,
                shape,
                ..
            } => SearchOp::Gemm {
                input: *input,
                n_out: shape.0,
                n_in: shape.1,
                weight: weight.clone(),
                bias: bias.clone(),
            },
        });
    }
    let lowered = SearchNet {
        n_in: net.n_in,
        ops,
    };
    let sizes = lowered.tensor_sizes()?;
    if sizes != net.tsize {
        return Err("lowered tensor sizes disagree with the twin net".into());
    }
    Ok(lowered)
}

/// Adam step of the joint search's slopes: abc's root alpha-CROWN `lr 0.25`
/// (cifar100 / tinyimagenet configs). Deliberately NOT the lane's
/// `margin_row.alpha_lr`, which steers the certified ascent's own optimizer.
pub(crate) const JOINT_ROOT_LR: f64 = 0.25;

/// Bytes of [`lower_twin_net`]'s copy (RSS admission; computed without
/// building it).
pub(crate) fn search_copy_bytes(net: &TwinNet) -> u64 {
    const F: u64 = 8;
    net.ops
        .iter()
        .map(|op| match op {
            TwinOp::Conv(c) => {
                (c.gather.len() as u64) * 4 + ((c.wmat.len() + c.bias.len()) as u64) * F
            }
            TwinOp::ChannelAffine { scale, shift, .. } => ((scale.len() + shift.len()) as u64) * F,
            TwinOp::Gemm { weight, bias, .. } => ((weight.len() + bias.len()) as u64) * F,
            TwinOp::Relu { .. } | TwinOp::Add { .. } | TwinOp::Flatten { .. } => 0,
        })
        .fold(0u64, u64::saturating_add)
}

/// The certified root's trunk boxes, as the search's REFERENCE boxes.
pub(crate) fn reference_boxes(root: &RootGates) -> Vec<ReluBox> {
    root.layers
        .iter()
        .map(|layer| ReluBox {
            op: layer.op,
            l: layer.l.clone(),
            u: layer.u.clone(),
        })
        .collect()
}

/// Write a hint's slopes into the UNTRUSTED scratch the ingest gate reads.
///
/// A trunk vector lands only on the layer with the same op index and width; a
/// head vector only at the head's width. Values are copied verbatim (widened
/// to f64): VALIDATING them is the ingest gate's job
/// (`install_candidate_alpha`: finite, clamped, unstable-in-the-certified-box
/// only). Returns `(trunk layers written, head written)`.
pub(crate) fn hint_into_scratch(
    scratch: &mut RootGates,
    hint: &RootAlphaHint,
    n_y: usize,
) -> (usize, bool) {
    let mut layers = 0usize;
    for (op, slopes) in &hint.trunk {
        if let Some(layer) = scratch.layers.iter_mut().find(|layer| layer.op == *op) {
            if slopes.len() == layer.n && layer.alpha.len() == layer.n {
                layer.alpha = slopes.iter().map(|&a| f64::from(a)).collect();
                layers += 1;
            }
        }
    }
    let head = hint.head.len() == n_y && n_y > 0;
    if head {
        scratch.head_alpha = Some(hint.head.iter().map(|&a| f64::from(a)).collect());
    }
    (layers, head)
}

// ---------------------------------------------------------------------------
//  Running the search and its observable
// ---------------------------------------------------------------------------

/// What one `joint_f32` phase did (telemetry; nothing here is read by a
/// verdict path). Search bounds are kept as FORMATTED text so no untrusted
/// number can be compared with zero by accident.
#[derive(Debug, Clone)]
pub struct BbeReport {
    /// Backend name (`none` when no backend was registered).
    pub backend: &'static str,
    /// Why the search produced no candidate, if it did not.
    pub refused: Option<String>,
    /// Joint iterations after the CROWN init.
    pub iters: usize,
    /// Start-node passes.
    pub passes: usize,
    /// Wall seconds of lowering + search.
    pub search_s: f64,
    /// Wall seconds of the candidate's certification (ingest, certified
    /// evaluations and the optional certified refresh).
    pub certify_s: f64,
    /// The search's untrusted root bound (S3), formatted; `nan` when none.
    pub root_lb: String,
    /// The search's untrusted CROWN-init root bound, formatted.
    pub crown_lb: String,
    /// Why the search stopped.
    pub search_stop: &'static str,
    /// Adversarial classes / classes pruned by the CROWN init.
    pub specs: usize,
    pub pruned: usize,
    /// Intermediate rows at the init; slopes shared per start node.
    pub rows: usize,
    pub share_rows: bool,
    /// Worker threads.
    pub threads: usize,
    /// Reference box sides tightened / reference-unstable neurons stabilized
    /// by the search (search-side counts).
    pub tightened: usize,
    pub stabilized: usize,
    /// Trunk layers / head the hint wrote into the scratch.
    pub hint_layers: usize,
    pub hint_head: bool,
}

impl BbeReport {
    pub(crate) fn new(backend: &'static str) -> Self {
        Self {
            backend,
            refused: None,
            iters: 0,
            passes: 0,
            search_s: 0.0,
            certify_s: 0.0,
            root_lb: "nan".into(),
            crown_lb: "nan".into(),
            search_stop: "not_run",
            specs: 0,
            pruned: 0,
            rows: 0,
            share_rows: false,
            threads: 0,
            tightened: 0,
            stabilized: 0,
            hint_layers: 0,
            hint_head: false,
        }
    }

    /// The fixed-schema `[bbe]` line. `moved`, `kept`, the certified numbers
    /// and `stop` come from the phase's own report.
    #[must_use]
    pub fn line(&self, phase: &super::root_certify_once::RootCertifyOnceReport) -> String {
        format!(
            "[bbe] phase=joint_root iters={} search_s={:.3} certify_s={:.3} root_lb={} \
backend={} search_stop={} crown_lb={} specs={} pruned={} rows={} share_rows={} threads={} \
tightened={} stabilized={} hint_layers={} hint_head={} moved={} kept={} incumbent={:.6} \
certified={:.6} published={:.6} stop={} refused={}",
            self.iters,
            self.search_s,
            self.certify_s,
            self.root_lb,
            self.backend,
            self.search_stop,
            self.crown_lb,
            self.specs,
            self.pruned,
            self.rows,
            u8::from(self.share_rows),
            self.threads,
            self.tightened,
            self.stabilized,
            self.hint_layers,
            u8::from(self.hint_head),
            phase.moved,
            phase.kept.token(),
            phase.incumbent,
            phase.certified,
            phase.published,
            phase.stop,
            self.refused.as_deref().unwrap_or("none"),
        )
    }
}

/// Run the injected search for `(t, adv)` over the certified root's box and
/// reference boxes, until `deadline`, within `mem_cap` bytes. Returns the
/// hint (`None` on any refusal) and the telemetry. Never panics: a backend
/// panic is caught and reported as a refusal.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_joint_search(
    net: &TwinNet,
    root: &RootGates,
    t: usize,
    adv: &[usize],
    backend: Option<&dyn SearchBackend>,
    iters: usize,
    lr: f64,
    deadline: Option<Instant>,
    mem_cap: u64,
) -> (Option<RootAlphaHint>, BbeReport) {
    let t0 = Instant::now();
    let Some(backend) = backend else {
        let mut report = BbeReport::new("none");
        report.refused = Some("no search backend registered".into());
        return (None, report);
    };
    let mut report = BbeReport::new(backend.name());
    let lowered = match lower_twin_net(net) {
        Ok(lowered) => lowered,
        Err(error) => {
            report.refused = Some(format!("lowering: {error}"));
            report.search_s = t0.elapsed().as_secs_f64();
            return (None, report);
        }
    };
    let request = RootSearchRequest {
        lo: root.lo.clone(),
        hi: root.hi.clone(),
        ref_boxes: reference_boxes(root),
        target: t,
        adv: adv.to_vec(),
        iters,
        lr,
        deadline,
        threads: 0,
        mem_cap_bytes: mem_cap.saturating_sub(lowered.approx_bytes()),
    };
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        backend.root_joint_alpha(&lowered, &request)
    }));
    report.search_s = t0.elapsed().as_secs_f64();
    match outcome {
        Err(_) => {
            report.refused = Some("the search backend panicked".into());
            (None, report)
        }
        Ok(Err(refusal)) => {
            report.refused = Some(refusal.to_string());
            (None, report)
        }
        Ok(Ok(outcome)) => {
            let stats = &outcome.stats;
            report.iters = stats.iters;
            report.passes = stats.passes;
            report.root_lb = format!("{:.6}", stats.root_lb);
            report.crown_lb = format!("{:.6}", stats.crown_lb);
            report.search_stop = stats.stop;
            report.specs = stats.specs;
            report.pruned = stats.pruned;
            report.rows = stats.rows;
            report.share_rows = stats.share_rows;
            report.threads = stats.threads;
            report.tightened = stats.tightened;
            report.stabilized = stats.stabilized;
            (Some(outcome.alpha.into_hint().into_ingest()), report)
        }
    }
}
