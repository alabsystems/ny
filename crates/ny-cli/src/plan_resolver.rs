// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Plan resolver v1 (#plan-resolver, design I2/I6/I10).
//!
//! "ny needs to be smart enough to choose the correct settings" — the
//! resolver is where that choice lives. v1 owns exactly two things:
//!
//! 1. **The layering contract** (scored-adoption form): resolved defaults
//!    first, preset keys OVERRIDE. A key the category preset sets explicitly
//!    is NEVER displaced, so every shipped category with explicit keys
//!    behaves byte-identically with the resolver wired in. A key the preset
//!    leaves absent takes the resolver's measured rule. Every final value
//!    carries its true [`SettingSource`]; when one explicit preset key blocks
//!    the other half of an evidence-inseparable pair, the printed source SAYS
//!    so. The earlier printer-only draft let measured rules shadow explicit
//!    preset keys, and wiring that contract into the scored path would have
//!    changed shipped-category behavior without a sealed A/B.
//!
//! 2. **The measured decision table** (each rule cites its evidence inline
//!    below). Rules the resolver cannot yet derive stay preset-driven and are
//!    printed as pass-throughs — v1 is not omniscient, it is explicit.
//!
//! `resolve_plan` is deliberately pure: model facts + scored budget + backend
//! report + optional preset + typed runtime facts in, `ResolvedPlan` out. No
//! filesystem, no env, no globals — the tests pin the whole decision table on
//! synthetic facts (real competition ONNX files are gitignored downloads).
//! Command boundaries decode process-global runtime inputs before calling it.
//! Around it sit the two consumers:
//! - `ny vnncomp plan` (commands/vnncomp_plan.rs), the thin I2 printer over
//!   `render_settings()`/`iter()`.
//! - the scored path (commands/vnncomp.rs), which calls
//!   [`resolve_and_materialize_with_runtime`]: model facts come from a
//!   single-pass raw protobuf scan (no second model load — weight payloads are
//!   skipped by offset), and resolved values are APPLIED by materializing a
//!   merged temp preset (the original YAML plus ONLY the absent resolver-owned
//!   keys) so every downstream preset consumer (β-CROWN handler, margin-row lanes,
//!   upfront attack) reads ONE consistent plan. Any materialization failure
//!   degrades to the original preset path — preset-only behavior, recorded
//!   in the `plan_resolved` flight note. The resolver can never lose a
//!   preset or cost a verdict.
//!
//! Docs: docs/PLAN_RESOLVER_V1_2026-08-01.md.

use std::ffi::OsStr;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use crate::compute_backend::BackendReport;
use crate::preset::PresetConfig;
use ny_core::LayerType;

/// Facts derived from the MODEL — never from its filename.
///
/// The 2026-07-31 attribution work replaced a seven-filename CIFAR100
/// allowlist with structural predicates for exactly this reason: filename
/// matching is a per-instance bet, model facts travel with the artifact.
///
/// Two constructors, one contract:
/// - [`ModelFacts::from_loaded_model`] — the printer path, which loads the
///   model anyway.
/// - [`ModelFacts::from_onnx_file`] — the scored path, which must NOT pay a
///   second model load: a single-pass raw protobuf scan of the graph
///   skeleton (initializer dims + Conv node wiring) yields exact
///   `param_count`/`conv_layers`/`max_conv_out_channels` in milliseconds for
///   the 10–15MB competition files. Output rows and the full layer-type
///   multiset are deliberately omitted in v1: no rule consumes them.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub(crate) struct ModelFacts {
    /// Total parameter count (summed initializer element counts).
    pub(crate) param_count: usize,
    /// Number of convolution layers (Conv1d/Conv2d/ConvTranspose1d/ConvTranspose2d).
    pub(crate) conv_layers: usize,
    /// Widest convolution output-channel count (0 when `conv_layers == 0`).
    pub(crate) max_conv_out_channels: usize,
    /// On-disk ONNX size in bytes (coarse cross-check on the two above).
    pub(crate) file_size_bytes: u64,
}

/// Conv-model scale class for the attack-slice rule (rule 1).
///
/// MEASURED anchors (parsed from the shipped 2025 benchmark ONNX files,
/// 2026-08-01):
/// - `CIFAR100_resnet_medium.onnx`: 2,536,344 params, max conv width 128,
///   10,156,168 bytes.
/// - `CIFAR100_resnet_large.onnx`: 3,808,152 params, max conv width 256,
///   15,243,961 bytes.
/// - `TinyImageNet_resnet_medium.onnx`: 3,616,144 params, max conv width 128
///   — the tie-breaker for the discriminator choice: its tuned preset pins
///   the 0.40 slice, and the param-count predicate (>= 3.2M) classifies it
///   LARGE in agreement, while a width-only discriminator would have pushed
///   it into the medium pair against its own preset. That is why BOTH
///   predicates admit LARGE.
///
/// CONFIG: thresholds sit at the midpoints between the cifar100 anchors
/// (channels 192, params 3.2M), so both anchor models classify with wide
/// margin and a future model must genuinely move toward the other anchor to
/// flip class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ConvScale {
    /// Large conv net: attacks need tens of seconds (21–31s measured).
    Large,
    /// Medium conv net: attacks land in seconds; budget belongs to BaB.
    Medium,
    /// Small conv net or no convolutions: no measured slice rule applies.
    SmallOrNone,
}

impl std::fmt::Display for ConvScale {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ConvScale::Large => "conv-large",
            ConvScale::Medium => "conv-medium",
            ConvScale::SmallOrNone => "conv-small-or-none",
        })
    }
}

/// CONFIG: LARGE lower bounds — midpoints between the measured medium/large
/// anchors above. Either predicate admits (channel width is the sharper
/// discriminator; param count catches wide-but-shallow nets and the
/// TinyImageNet-medium 3.62M anchor).
const LARGE_CONV_MIN_CHANNELS: usize = 192;
const LARGE_CONV_MIN_PARAMS: usize = 3_200_000;

/// CONFIG: MEDIUM lower bound. Below ~1M params the cifar100 evidence claims
/// nothing; small conv nets keep engine defaults rather than inheriting a
/// rule measured two size classes away.
const MEDIUM_CONV_MIN_PARAMS: usize = 1_000_000;

/// CONFIG: "short budget" ceiling for the medium PAIR rule, in SCORED
/// seconds. The conversion evidence is at the official 100s budget
/// (prop_idx_9694, bd6a9eff) and the preset commentary covers scored budgets
/// up to 300s; beyond that the pair has no measurement, so the resolver
/// declines rather than extrapolates.
const MEDIUM_PAIR_MAX_BUDGET_SECS: u64 = 300;

/// Skip the raw graph scan above this file size. CONFIG: the 500MB-class
/// models (vggnet16/vit) all ship explicit presets, so the resolver has
/// nothing to decide there and should not pay a half-gigabyte read at
/// instance start.
const SCAN_MAX_BYTES: u64 = 256 * 1024 * 1024;

impl ModelFacts {
    /// Derive facts from a loaded model plus the on-disk size.
    ///
    /// Conv weight tensors are looked up in the model's `WeightStore` by the
    /// layer's second input name — the ONNX graph loader keeps weights there
    /// and leaves `LayerSpec::weights` unset (only the native builders fill
    /// it), so a spec-only walk would report width 0 for every real conv
    /// net (MEASURED on CIFAR100_resnet_medium during this module's smoke).
    /// The `WeightRef` path is kept as the primary source for loaders that
    /// do fill it.
    pub(crate) fn from_loaded_model(model: &ny_onnx::OnnxModel, file_size_bytes: u64) -> Self {
        let network = &model.network;
        let mut conv_layers = 0usize;
        let mut max_conv_out_channels = 0usize;
        for layer in &network.layers {
            let is_conv = matches!(
                layer.layer_type,
                LayerType::Conv1d
                    | LayerType::Conv2d
                    | LayerType::ConvTranspose1d
                    | LayerType::ConvTranspose2d
            );
            if !is_conv {
                continue;
            }
            conv_layers += 1;
            let ref_width = layer
                .weights
                .as_ref()
                .map(|weights| leading_pair_max(&weights.shape));
            let store_width = layer
                .inputs
                .get(1)
                .and_then(|name| model.weights.get(name))
                .map(|tensor| leading_pair_max(tensor.shape()));
            let width = ref_width.or(store_width).unwrap_or(0);
            max_conv_out_channels = max_conv_out_channels.max(width);
        }
        Self {
            param_count: network.param_count,
            conv_layers,
            max_conv_out_channels,
            file_size_bytes,
        }
    }

    /// Scored-path facts WITHOUT a model load: single-pass raw protobuf scan
    /// of the graph skeleton (weight payloads are skipped by offset).
    ///
    /// Returns `None` (every model rule declines, sources stay `Default`)
    /// when the file is missing, not a plain `.onnx`, larger than
    /// [`SCAN_MAX_BYTES`], or does not parse cleanly — wrong facts are worse
    /// than no facts.
    pub(crate) fn from_onnx_file(onnx: &Path) -> Option<Self> {
        let len = std::fs::metadata(onnx).ok()?.len();
        let is_plain_onnx = onnx
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("onnx"));
        if !is_plain_onnx || len == 0 || len > SCAN_MAX_BYTES {
            return None;
        }
        let bytes = std::fs::read(onnx).ok()?;
        let scan = scan_onnx_graph(&bytes).ok()?;
        Some(Self {
            param_count: scan.param_count,
            conv_layers: scan.conv_layers,
            max_conv_out_channels: scan.max_conv_out_channels,
            file_size_bytes: len,
        })
    }

    /// Classify the conv scale for rule 1. Pure function of the facts.
    ///
    /// Note: the param-count predicate alone classifies both measured
    /// anchors correctly, so a loader that cannot resolve conv widths
    /// (width 0) still discriminates medium from large on the anchors.
    pub(crate) fn conv_scale(&self) -> ConvScale {
        if self.conv_layers == 0 {
            return ConvScale::SmallOrNone;
        }
        if self.max_conv_out_channels >= LARGE_CONV_MIN_CHANNELS
            || self.param_count >= LARGE_CONV_MIN_PARAMS
        {
            return ConvScale::Large;
        }
        if self.param_count >= MEDIUM_CONV_MIN_PARAMS {
            return ConvScale::Medium;
        }
        ConvScale::SmallOrNone
    }

    fn summary(&self) -> String {
        format!(
            "facts{{params={}, convs={}, max_conv={}, bytes={}, scale={}}}",
            self.param_count,
            self.conv_layers,
            self.max_conv_out_channels,
            self.file_size_bytes,
            self.conv_scale(),
        )
    }
}

/// Max of a weight shape's two leading dims — the conv width witness used by
/// [`ModelFacts::from_loaded_model`] for both Conv ([out, in, k...]) and
/// ConvTranspose ([in, out, k...]) layouts.
fn leading_pair_max(shape: &[usize]) -> usize {
    shape
        .first()
        .copied()
        .unwrap_or(0)
        .max(shape.get(1).copied().unwrap_or(0))
}

// ---------------------------------------------------------------------------
// Raw protobuf graph scan (no ny-onnx load, no ORT, no weight materialization)
// ---------------------------------------------------------------------------

struct GraphScan {
    param_count: usize,
    conv_layers: usize,
    max_conv_out_channels: usize,
}

type ScanResult<T> = Result<T, &'static str>;

fn read_varint(buf: &[u8], i: &mut usize) -> ScanResult<u64> {
    let mut out: u64 = 0;
    let mut shift = 0u32;
    loop {
        let byte = *buf.get(*i).ok_or("varint past end")?;
        *i += 1;
        if shift >= 64 {
            return Err("varint too long");
        }
        out |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok(out);
        }
        shift += 7;
    }
}

/// One protobuf field: varints by value, length-delimited by span, fixed
/// widths skipped. Group wire types (3/4, long-deprecated) are a parse error.
enum Field<'a> {
    Varint(u32, u64),
    Bytes(u32, &'a [u8]),
    Skipped,
}

fn read_field<'a>(buf: &'a [u8], i: &mut usize) -> ScanResult<Field<'a>> {
    let key = read_varint(buf, i)?;
    let field = u32::try_from(key >> 3).map_err(|_| "field number overflow")?;
    match key & 7 {
        0 => Ok(Field::Varint(field, read_varint(buf, i)?)),
        1 => {
            *i = i
                .checked_add(8)
                .filter(|&n| n <= buf.len())
                .ok_or("fixed64 past end")?;
            Ok(Field::Skipped)
        }
        2 => {
            let len = usize::try_from(read_varint(buf, i)?).map_err(|_| "length overflow")?;
            let end = i
                .checked_add(len)
                .filter(|&n| n <= buf.len())
                .ok_or("bytes past end")?;
            let span = &buf[*i..end];
            *i = end;
            Ok(Field::Bytes(field, span))
        }
        5 => {
            *i = i
                .checked_add(4)
                .filter(|&n| n <= buf.len())
                .ok_or("fixed32 past end")?;
            Ok(Field::Skipped)
        }
        _ => Err("unsupported wire type"),
    }
}

/// Scan an ONNX `ModelProto` for the graph-skeleton facts. Wire layout:
/// `ModelProto.graph` = field 7; `GraphProto.node` = field 1 (repeated
/// `NodeProto`), `.initializer` = field 5 (repeated `TensorProto`);
/// `NodeProto.input` = field 1, `.op_type` = field 4; `TensorProto.dims` =
/// field 1 (varint, unpacked or packed), `.name` = field 8. Weight payloads
/// (`raw_data` etc.) are skipped by offset, never copied.
fn scan_onnx_graph(bytes: &[u8]) -> ScanResult<GraphScan> {
    // ModelProto: last `graph` occurrence wins (proto merge semantics).
    let mut graph: Option<&[u8]> = None;
    let mut i = 0usize;
    while i < bytes.len() {
        if let Field::Bytes(7, span) = read_field(bytes, &mut i)? {
            graph = Some(span);
        }
    }
    let graph = graph.ok_or("no graph field")?;

    let mut init_dims: std::collections::BTreeMap<&str, Vec<u64>> =
        std::collections::BTreeMap::new();
    let mut conv_weight_names: Vec<&str> = Vec::new();
    let mut conv_layers = 0usize;
    let mut param_count: u64 = 0;
    let mut i = 0usize;
    while i < graph.len() {
        match read_field(graph, &mut i)? {
            Field::Bytes(5, tensor) => {
                let (name, dims) = scan_tensor(tensor)?;
                let mut elements: u64 = 1;
                for &d in &dims {
                    // int64 dims are plain varints (not zigzag); a negative
                    // or absurd dim decodes huge. Fail the scan — wrong facts
                    // are worse than no facts.
                    if d > 1_000_000_000 {
                        return Err("implausible initializer dim");
                    }
                    elements = elements.checked_mul(d).ok_or("param count overflow")?;
                }
                param_count = param_count
                    .checked_add(elements)
                    .ok_or("param count overflow")?;
                if let Some(name) = name {
                    init_dims.insert(name, dims);
                }
            }
            Field::Bytes(1, node) => {
                let (op, inputs) = scan_node(node)?;
                if op == "Conv" || op == "ConvTranspose" {
                    conv_layers += 1;
                    if let Some(weight) = inputs.get(1) {
                        conv_weight_names.push(weight);
                    }
                }
            }
            _ => {}
        }
    }

    let mut max_conv_out_channels = 0usize;
    for weight in conv_weight_names {
        if let Some(dims) = init_dims.get(weight) {
            // Same width witness as `from_loaded_model`: max of the two
            // leading dims covers Conv ([out, in, k...]) and ConvTranspose
            // ([in, out, k...]) without per-type special-casing.
            if dims.len() >= 3 {
                let leading = usize::try_from(dims[0]).unwrap_or(0);
                let second = usize::try_from(dims[1]).unwrap_or(0);
                max_conv_out_channels = max_conv_out_channels.max(leading.max(second));
            }
        }
    }

    Ok(GraphScan {
        param_count: usize::try_from(param_count).map_err(|_| "param count overflow")?,
        conv_layers,
        max_conv_out_channels,
    })
}

/// TensorProto: `(name, dims)`. Handles both unpacked (wire 0) and packed
/// (wire 2) encodings of the repeated int64 `dims` field.
fn scan_tensor(tensor: &[u8]) -> ScanResult<(Option<&str>, Vec<u64>)> {
    let mut dims = Vec::new();
    let mut name = None;
    let mut i = 0usize;
    while i < tensor.len() {
        match read_field(tensor, &mut i)? {
            Field::Varint(1, d) => dims.push(d),
            Field::Bytes(1, packed) => {
                let mut j = 0usize;
                while j < packed.len() {
                    dims.push(read_varint(packed, &mut j)?);
                }
            }
            Field::Bytes(8, raw) => name = std::str::from_utf8(raw).ok(),
            _ => {}
        }
    }
    Ok((name, dims))
}

/// NodeProto: `(op_type, inputs)`.
fn scan_node(node: &[u8]) -> ScanResult<(&str, Vec<&str>)> {
    let mut op = "";
    let mut inputs = Vec::new();
    let mut i = 0usize;
    while i < node.len() {
        match read_field(node, &mut i)? {
            Field::Bytes(1, raw) => inputs.push(std::str::from_utf8(raw).unwrap_or("")),
            Field::Bytes(4, raw) => op = std::str::from_utf8(raw).unwrap_or(""),
            _ => {}
        }
    }
    Ok((op, inputs))
}

// ---------------------------------------------------------------------------
// Sources and settings
// ---------------------------------------------------------------------------

/// Where a final plan value truly came from — the layering contract, typed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SettingSource {
    /// A model-fact rule produced the value (rule id, evidence cited).
    ResolvedModel(&'static str),
    /// A budget rule produced the value (rule id, evidence cited). Carries a
    /// `String` so per-run measured arithmetic (#fl-phase-budget: the probed
    /// rate, the predicted build seconds, the tier) can ride in the reason —
    /// provenance with the numbers, not just the rule name.
    ResolvedBudget(String),
    /// The category preset set the key explicitly; carries the yaml key
    /// path. Preset keys OVERRIDE resolved rules — shipped categories with
    /// explicit keys are byte-identical with the resolver wired in.
    PresetOverride(String),
    /// (There is deliberately no `ResolvedBackend`. Both settings that used it —
    /// `accelerator_role` and `attack_steering` — were host observations, not
    /// per-category choices, so once they were retagged the variant had no
    /// constructor left. A backend-derived RULE would still need evidence
    /// scoping under #rule-contract, so it would not resurrect it as-is.)
    /// An observed property of the HOST, recorded for provenance — never a
    /// choice the resolver made for a category (#rule-contract).
    ///
    /// `is_resolved()` deliberately does NOT match this: host facts are not
    /// evidence-scoped (they are true of the machine, not of a benchmark) and
    /// nothing materializes them into the effective preset. Keeping them out of
    /// `Resolved*` is what makes "every `Resolved*` is evidence-scoped AND
    /// materializable" a total statement rather than an aspiration — so a
    /// future revision that loops over resolved settings cannot silently flip
    /// all 48 categories on a property of the developer's laptop.
    HostFact(&'static str),
    /// A process-level diagnostic input captured exactly once at the command
    /// boundary. Like [`Self::HostFact`], this is observed provenance rather
    /// than a resolver-owned rule and is never materialized into the preset.
    RuntimeOverride(&'static str),
    /// Preset scheduling could not be resolved. Execution rejects the same
    /// preset; the nominal reporter fails closed instead of inventing an
    /// enabled schedule and attributing it to a malformed key.
    InvalidPreset(String),
    /// No rule fired and no preset key exists: the engine default stands.
    Default,
}

impl std::fmt::Display for SettingSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SettingSource::ResolvedModel(rule) => write!(f, "resolved(model-facts): {rule}"),
            SettingSource::ResolvedBudget(rule) => write!(f, "resolved(budget): {rule}"),
            SettingSource::HostFact(rule) => write!(f, "host fact: {rule}"),
            SettingSource::RuntimeOverride(rule) => write!(f, "runtime override: {rule}"),
            SettingSource::InvalidPreset(error) => write!(f, "invalid preset: {error}"),
            SettingSource::PresetOverride(key) => write!(f, "preset override: {key}"),
            SettingSource::Default => f.write_str("default"),
        }
    }
}

/// One resolved setting: the value the run will use, and why.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ResolvedSetting<T> {
    pub(crate) value: T,
    pub(crate) source: SettingSource,
}

impl<T> ResolvedSetting<T> {
    fn new(value: T, source: SettingSource) -> Self {
        Self { value, source }
    }

    /// True when a resolver rule owns the value — and, since #rule-contract,
    /// EXACTLY the values materialization may apply.
    ///
    /// That second half used to be false: `accelerator_role` and
    /// `attack_steering` emitted `ResolvedBackend` for all 48 categories while
    /// being materialized by nothing. They are `HostFact` now, while command
    /// diagnostics are `RuntimeOverride`, so the two properties finally
    /// coincide.
    fn is_resolved(&self) -> bool {
        matches!(
            self.source,
            SettingSource::ResolvedModel(_) | SettingSource::ResolvedBudget(_)
        )
    }
}

/// Margin-row reserve policy (rule 2). The resolver records the preset's
/// explicit choice; with no key it reports the fixed engine default and
/// materializes nothing. Admission (lane armed + twin-spec structural match)
/// stays downstream in `margin_row_bab`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MarginRowPolicy {
    /// Adaptive release of the reserve (cifar100-measured).
    Adaptive,
    /// A fixed reserve of N seconds. This is the ENGINE DEFAULT (45 s, see
    /// `margin_row_bab::margin_row_reserve_secs`), not a preset-only value.
    ///
    /// The previous comment here called it "measured harmful as a blanket
    /// default (b61b5f10)". That inverts the citation: b61b5f10 found holding
    /// the reserve is what the two near-wall unsat rows NEEDED, and that
    /// releasing it (`Adaptive`) is what cost them.
    Fixed(u64),
    /// Reserve explicitly zeroed by the preset.
    NoReserve,
}

impl std::fmt::Display for MarginRowPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MarginRowPolicy::Adaptive => f.write_str("adaptive-release"),
            MarginRowPolicy::Fixed(secs) => write!(f, "fixed({secs}s)"),
            MarginRowPolicy::NoReserve => f.write_str("no-reserve"),
        }
    }
}

/// One rendered plan line for the I2 printer (`name = value  [source]`).
#[derive(Debug, Clone)]
pub(crate) struct SettingLine {
    pub(crate) name: &'static str,
    pub(crate) value: String,
    pub(crate) source: String,
}

/// The I6 nominal budget snapshot the plan prints.
///
/// Dynamic wrapper deductions and elapsed wall time do not exist at plan
/// resolution, so these fields describe policy allocation before execution,
/// not the phase's eventual remaining-time grant.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub(crate) struct BudgetLedgerSnapshot {
    /// The competition budget as scored (protocol TIMEOUT).
    pub(crate) scored_budget_secs: u64,
    /// Nominal internal verifier tier: `internal_timeout_secs` before dynamic
    /// wrapper deductions (grace = max(5, budget/20)).
    pub(crate) nominal_internal_tier_secs: u64,
    /// Nominal disjunctive attack deadline offset in seconds, before dynamic
    /// wrapper deductions and time elapsed between ledger construction and the
    /// phase. `None` means effective disjunctive PGD is disabled. When present,
    /// the value mirrors the runtime's fraction/tiny-cap/ceiling/floor policy.
    pub(crate) nominal_attack_slice_secs: Option<f64>,
    /// Hard wall cap on the root alpha-CROWN warmup; `None` = uncapped
    /// (the initial-bounds fraction alone governs).
    pub(crate) root_alpha_cap_secs: Option<f64>,
}

/// The resolved plan: typed v1 settings (each value + true source), the
/// rendered print lines, and the I6 nominal budget snapshot.
#[derive(Debug, Clone)]
pub(crate) struct ResolvedPlan {
    /// `cuda`, `metal`, `gpu`, or `cpu-only` (BackendReport::kind).
    pub(crate) backend_kind: &'static str,
    /// The one-line host summary from detection.
    pub(crate) backend_summary: String,
    /// Derived conv scale class (printed with the model facts).
    pub(crate) conv_scale: ConvScale,
    /// Rule 1 slice: `None` = engine default (0.50) stands.
    pub(crate) disjunctive_pgd_fraction: ResolvedSetting<Option<f32>>,
    /// Effective global disjunctive-PGD admission after preset + env policy.
    pub(crate) disjunctive_pgd_enabled: ResolvedSetting<bool>,
    /// Diagnostic override of the runtime's <=30s 15% attack cap.
    pub(crate) pgd_time_cap_disabled: ResolvedSetting<bool>,
    /// Preset-only absolute ceiling on the disjunctive PGD slice.
    pub(crate) disjunctive_pgd_max_secs: ResolvedSetting<Option<u64>>,
    /// Preset-only absolute floor on the disjunctive PGD slice.
    pub(crate) disjunctive_pgd_min_secs: ResolvedSetting<Option<u64>>,
    /// Whether the disjunctive slice is anchored at phase start.
    pub(crate) disjunctive_pgd_from_phase_start: ResolvedSetting<bool>,
    /// Rule 1 pair leg: `None` = uncapped root alpha warmup.
    pub(crate) root_alpha_cap_secs: ResolvedSetting<Option<f64>>,
    /// Rule 2 margin-row reserve policy.
    pub(crate) margin_row: ResolvedSetting<MarginRowPolicy>,
    /// Rule 3, RECORDED: async attack-steering arming exists on this host.
    pub(crate) attack_steering_armed: ResolvedSetting<bool>,
    /// Rule 5, pass-through: preset value forwarded, source recorded.
    pub(crate) alpha_spec_slots: ResolvedSetting<Option<usize>>,
    /// Rule 7 (#fl-alpha-composition): forward-map alpha surrogate arming.
    /// `Some(true)` resolved ONLY on rule 6's own widen events; a preset that
    /// sets either surrogate key wins and is forwarded as `PresetOverride`.
    pub(crate) forward_alpha_surrogate: ResolvedSetting<Option<bool>>,
    /// Rule 6 (#fl-phase-budget) decision note, `Some` whenever a measured FL
    /// rate was injected and the rule's scope applied — "widened root window
    /// to Ns" or the decline arithmetic. `None` = scope never applied, so no
    /// FL-specific segment is emitted. Recorded in the flight event EITHER
    /// WAY.
    pub(crate) fl_phase_budget: Option<String>,
    /// Resolved settings in stable print order (the I2 surface).
    pub(crate) settings: Vec<SettingLine>,
    /// The I6 nominal budget snapshot.
    pub(crate) ledger: BudgetLedgerSnapshot,
}

impl ResolvedPlan {
    /// Iterate `(name, value, source)` triples in print order. The source is
    /// rendered, so printers and JSON writers share one spelling per layer.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (&'static str, &str, String)> + '_ {
        self.settings
            .iter()
            .map(|s| (s.name, s.value.as_str(), s.source.clone()))
    }

    /// Render the settings block exactly as the human printer shows it —
    /// also the snapshot-test surface, so a routing change is a reviewed
    /// diff, never a discovery three weeks into a scorecard (I2).
    pub(crate) fn render_settings(&self) -> String {
        self.iter()
            .map(|(name, value, source)| format!("{name} = {value}  [{source}]"))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Engine default for the disjunctive PGD fraction
/// (`PhaseBudgetConfig::default().disjunctive_pgd_fraction`).
const DEFAULT_DISJUNCTIVE_PGD_FRACTION: f32 = 0.50;

pub(crate) const PGD_TIME_CAP_DISABLE_ENV: &str = "NY_NO_PGD_TIME_CAP";
pub(crate) const DISJUNCTIVE_PGD_SKIP_ENV: &str = "NY_SKIP_DISJ_PGD";

const RULE_PGD_TIME_CAP_ENV: &str =
    "NY_NO_PGD_TIME_CAP=1 — diagnostic override of the <=30s attack cap";
const RULE_DISJUNCTIVE_PGD_SKIP_ENV: &str =
    "NY_SKIP_DISJ_PGD=1 — diagnostic skip of global disjunctive PGD";

/// Process-global runtime decisions captured at a command boundary and then
/// passed as immutable data into the pure resolver.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct PlanRuntimeOverrides {
    pgd_time_cap_disabled: bool,
    disjunctive_pgd_skipped: bool,
}

impl PlanRuntimeOverrides {
    /// Exact-string decoding shared by the scored and printer boundaries.
    /// Injected values keep resolver tests environment-free and deterministic.
    pub(crate) fn from_env_values(
        pgd_time_cap: Option<&OsStr>,
        disjunctive_pgd_skip: Option<&OsStr>,
    ) -> Self {
        Self {
            pgd_time_cap_disabled: env_value_is_exact_one(pgd_time_cap),
            disjunctive_pgd_skipped: env_value_is_exact_one(disjunctive_pgd_skip),
        }
    }
}

/// Shared exact-value decoder for runtime and plan command boundaries.
pub(crate) fn env_value_is_exact_one(value: Option<&OsStr>) -> bool {
    value == Some(OsStr::new("1"))
}

/// Runtime's tiny-budget boundary and attack ceiling. These intentionally
/// mirror `commands/beta_crown/verify/phase_budget.rs::{SMALL_BUDGET_TOTAL,
/// SMALL_BUDGET_ATTACK_FRACTION}`. Keep the source citation beside the mirror
/// so a runtime policy edit cannot masquerade as independent planner math.
const PLAN_SMALL_BUDGET_TOTAL_SECS: u64 = 30;
const PLAN_SMALL_BUDGET_ATTACK_FRACTION: f32 = 0.15;

/// Nominal disjunctive-PGD slice reported by the pure plan ledger.
///
/// This mirrors, in order, runtime
/// `PhaseBudgetLedger::{attack_phase_deadline,disjunctive_pgd_deadline}`:
/// finite legacy fractions clamp to `[0, 1]`; the competition-default tiny
/// tier caps them at 0.15; the absolute ceiling applies next; and the
/// half-total-clamped floor applies last. `Duration::mul_f32` is used rather
/// than widened `f64` multiplication so the snapshot has the same rounding as
/// the deadline actually constructed by the engine.
///
/// `NY_NO_PGD_TIME_CAP=1` is captured once by the command boundary and arrives
/// as `pgd_time_cap_disabled`; this helper never reads process-global state.
pub(crate) fn planned_disjunctive_pgd_slice_secs(
    internal_tier_secs: u64,
    fraction: f32,
    max_secs: Option<u64>,
    min_secs: Option<u64>,
    pgd_time_cap_disabled: bool,
) -> f64 {
    let total = std::time::Duration::from_secs(internal_tier_secs);
    // Applied engine configs reject non-finite fractions before the runtime
    // ledger is constructed. Keep this pure reporter total for a raw,
    // unvalidated PresetConfig too; the valid-runtime path below is exact.
    let fraction = if fraction.is_finite() {
        fraction.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let fraction = if internal_tier_secs <= PLAN_SMALL_BUDGET_TOTAL_SECS && !pgd_time_cap_disabled {
        fraction.min(PLAN_SMALL_BUDGET_ATTACK_FRACTION)
    } else {
        fraction
    };
    let mut slice = total.mul_f32(fraction);
    if let Some(max_secs) = max_secs {
        slice = slice.min(std::time::Duration::from_secs(max_secs));
    }
    if let Some(min_secs) = min_secs.filter(|&secs| secs > 0) {
        let floor = std::time::Duration::from_secs(min_secs).min(total.mul_f32(0.5));
        slice = slice.max(floor);
    }
    // Runtime makes the overall deadline the final authority after the floor.
    slice.min(total).as_secs_f64()
}

/// Rule 1 values: the MEDIUM short-budget PAIR. Only ever applied together;
/// LARGE classification deliberately has no slice rule without a sealed A/B.
const MEDIUM_PAIR_PGD_FRACTION: f32 = 0.05;
const MEDIUM_PAIR_ROOT_ALPHA_CAP_SECS: f64 = 40.0;

// ---------------------------------------------------------------------------
// FL-aware phase budgeting (#fl-phase-budget, I10) — rule 6 constants
//
// Evidence chain (docs/FL_FIRST_MEASUREMENT_2026-08-02.md +
// docs/CONVWALL_PANEL_VERDICT_2026-08-01.md addendum): the forward-linear
// reference build tightens CIFAR100_resnet_medium deficit rows by +34..+54
// margin per row, but the FL affordability gate sees the ALPHA deadline
// (init.rs min-composes `root_alpha_cap_secs` into it), and the shipped 40s
// cap cannot host a measured 24-48s FL build at ANY tier — "the cap and
// forward-linear are mutually exclusive". Flight witness at the official
// 100s tier (off100_prop_idx_7500_sidx_40): `forward_linear_admission:
// skipped — rate=11.57 GMAC/s (probe, 0.084s) predicted=48s remaining=38s`.
// The resolver owns this trade: when the MEASURED rate predicts the build
// plus an α slice plus a preserved BaB floor fits the internal tier, widen
// the root window to exactly what that phase plan needs; otherwise the
// banked cap-40 recipe stands untouched.
// ---------------------------------------------------------------------------

/// Cold FL build cost anchor for the MEDIUM conv class, in GMACs. MEASURED:
/// `CIFAR100_resnet_medium` is 559.4 G f64 MACs (19 convs + 2 Gemms, center +
/// radius passes — the same figure `forward_linear_cold_build_macs` computes
/// exactly at runtime and the flight event records as `build_gmacs=559`).
/// The rule's evidence scope is the medium band this anchor defines
/// (TinyImageNet_resnet_medium classifies LARGE and is out of scope).
const FL_MEDIUM_BUILD_GMACS: f64 = 559.4;

/// Admission safety margin — MUST mirror the gate's
/// `FORWARD_LINEAR_ADMISSION_MARGIN_NUM/DEN` (5/4): the resolver sizes the
/// window with the same pad the gate demands, so a window this rule opens is
/// one the gate's own `remaining >= predicted x 5/4` check can accept.
const FL_ADMISSION_MARGIN: f64 = 1.25;

/// Minimum α-CROWN ascent slice reserved AFTER the FL build inside the root
/// window. FL replaces the reference-bounds collection, not the α ascent that
/// consumes it; the shipped 100s recipe runs ~7 warmup iterations at ~1.5s
/// each plus collection overhead inside its window, so 15s keeps the ascent
/// whole rather than admitting FL into a window it fills completely.
const FL_ALPHA_SLICE_MIN_SECS: f64 = 15.0;

/// Minimum BaB time preserved BELOW the root window. Justified by the
/// banked-convert recipe (bd6a9eff): cap-40 at the 95s internal tier left
/// ~50s post-α, of which ~26s of BaB converted prop_idx_9694 — the two banked
/// rows (9694/8762) are the regression guard for any window change. 40s keeps
/// BaB at least the window class that recipe delivered (26s used + margin);
/// the widened root window may never eat into it.
const FL_BAB_FLOOR_SECS: f64 = 40.0;

// ---------------------------------------------------------------------------
// RULE CONTRACT (#rule-contract, 2026-08-01)
//
// Two rules shipped in `ed57f912` violated this and each flipped 34-47 of the
// 48 shipped categories on evidence measured in one. Read this before adding a
// rule or widening one.
//
// I.   EVIDENCE SCOPE IS PART OF THE RULE. A rule may produce a `Resolved*`
//      source only for a category its cited evidence actually covers.
//      Everywhere else it declines and the source stays
//      `SettingSource::Default` — `is_resolved()` false, so materialization
//      inserts nothing and the category is byte-identical to pre-resolver.
//
// II.  ABSENCE OF A PRESET KEY IS NOT CONSENT. Most categories omit any given
//      key. "The preset didn't say" is silence, not an opt-in. Firing into
//      silence is how one category's measurement becomes a board-wide default
//      one layer below where preset review can see it — which is exactly how
//      `8f5c7299`'s revert got undone.
//
// III. MODEL SHAPE IS NOT EVIDENCE OF TRANSFER. b61b5f10: "Op-identity of the
//      NETWORK says nothing about which lane is productive on the PROPERTY
//      distribution." `conv_scale()` may SELECT among rules already licensed
//      by I; it may not LICENSE one.
//
// IV.  THE CITATION MUST BE READ, NOT NAMED. The margin-row rule cited
//      b61b5f10 for the opposite of what b61b5f10 measured. The large-slice
//      rule cited a figure ("24 banked sats") that appears nowhere in the
//      document it named, which separately retracts the attribution for two of
//      the three rows it rested on. Quote the sentence that licenses the value.
//
// V.   THE BASELINE MUST BE THE ONE THE RULE FACES. An A/B of 0.05-vs-0.40
//      does not license 0.40 where the standing value is the 0.50 engine
//      default. If no A/B exists against the default, the rule has no evidence.
//
// A rule that can only fire where its measured category left the key absent is
// usually a sign the measurement belongs in THAT CATEGORY'S yaml, not here.
// ---------------------------------------------------------------------------

/// Rule ids — stable, evidence-citing strings for sources and flight notes.
pub(crate) const RULE_MEDIUM_PAIR: &str = "medium-conv-short-budget-pair — 0.05 slice + 40s \
     root-alpha cap converted CIFAR100_resnet_medium prop_idx_9694 at the official 100s \
     budget (bd6a9eff); neither half converts alone, applied as a PAIR or not at all";
pub(crate) const RULE_MARGIN_ADAPTIVE: &str = "margin-row-adaptive-release — cifar100-measured \
     ONLY, and NOT a board-wide default: b61b5f10 isolates margin_row.adaptive_reserve as \
     independently harmful on tinyimagenet (costs both near-wall GT-unsat rows at 80.8s/84.1s, \
     which need the reserve HELD). Fires only where a preset opts in explicitly";
pub(crate) const RULE_ATTACK_STEERING: &str = "attack-steering-always-armed-async — b030e2a8 + \
     beta_crown/attack_arming.rs";
pub(crate) const RULE_BACKEND_DETECT: &str = "compute_backend::detect() — accelerator roles \
     (I3: proposals vs certified)";
pub(crate) const RULE_CHARGED_METAL_GATE: &str = "charged-metal-authority (#flush-charge) — \
     RECORDED build/host fact, never a resolver rule: \
     ny_gpu::wgpu_charged_proof_authority() (the reviewed source gate \
     PRODUCTION_WGPU_CHARGED_VERDICT_AUTHORITY_ENABLED) narrates whether the fail-closed \
     proof chain commands::backend::resolve_proof_backend (new_for_proof -> \
     new_for_proof_flush_charged -> CPU) can even attempt charged qualification";
pub(crate) const RULE_FL_PHASE_BUDGET: &str = "fl-phase-budget (I10) — FL tightens \
     CIFAR100_resnet_medium deficit rows +34..+54/row but the 40s root window cannot host the \
     measured 24-48s build (docs/FL_FIRST_MEASUREMENT_2026-08-02.md); widen only when \
     pred + α-slice + BaB floor fits the tier, measured rate only, never below the banked \
     cap-40 recipe";
pub(crate) const RULE_FL_ALPHA_COMPOSITION: &str = "fl-alpha-composition (I10) — consult #8 \
     Days 1-3 (docs/ALGO_FRONTIER_CONSULT8_2026-08-02.md): where FL is admitted, arm the \
     forward-map alpha surrogate + ONE certified rebuild (#w4-root-alpha-opt) so binding-row \
     alpha optimizes ON the FL-fixed intermediates; spec propagation intersects with the \
     fixed FL C-margin candidate, so the fixed FL bound is the monotone floor (never weaker); \
     scope = EXACTLY this run's rule-6 widening (the window rule 6 opened IS the alpha \
     slice's budget)";

/// A measured forward-linear build rate, injected into plan resolution
/// (#fl-phase-budget). `resolve_plan` stays pure: the PROBE runs in the
/// callers (scored path / printer), only where the rule's scope applies, and
/// the observation travels in as data — tests inject fixed rates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct FlRateObservation {
    /// Derated MACs/second (the gate's admission units).
    pub(crate) macs_per_sec: u64,
    /// "probe" | "env" | "fallback" — only measured sources ("probe", or the
    /// manual "env" override) may widen; the stale "fallback" constant never
    /// does.
    pub(crate) source: &'static str,
    /// Best probe rep duration (0.0 unless source == "probe").
    pub(crate) probe_secs: f64,
}

/// Whether the FL widening rule's SCOPE applies: MEDIUM conv class (the
/// evidence anchor's band) with BOTH pair keys preset-absent (presets-win —
/// an explicit `root_alpha_cap_secs` or slice key pins the category and the
/// probe is never even paid for it).
pub(crate) fn fl_rate_scope_applies(facts: &ModelFacts, preset: Option<&PresetConfig>) -> bool {
    facts.conv_scale() == ConvScale::Medium
        && preset
            .and_then(|p| p.bab.phase_budget.disjunctive_pgd_fraction)
            .is_none()
        && preset.and_then(|p| p.bab.root_alpha_cap_secs).is_none()
}

/// Production probe: the SAME per-process calibration the FL admission gate
/// reads (env override > measured probe > fallback), so the resolver's window
/// arithmetic and the gate's admission arithmetic use one rate. ~0.1s quiet,
/// bounded by the calibration's 5s rep deadline; the gate later reuses the
/// cached result for free.
pub(crate) fn probe_fl_rate() -> Option<FlRateObservation> {
    let obs = ny_propagate::forward_linear_measured_rate();
    Some(FlRateObservation {
        macs_per_sec: obs.macs_per_sec,
        source: obs.source,
        probe_secs: obs.probe_secs,
    })
}

/// The FL phase-budget decision for one run (#fl-phase-budget), pure.
enum FlPhaseBudget {
    /// Widen the root window to `window_secs`; `reason` is the full
    /// provenance string (rule + this run's numbers).
    Widen { window_secs: f64, reason: String },
    /// Keep the standing cap; `note` says why, for the flight event.
    Decline { note: String },
}

/// Rule 6 arithmetic (#fl-phase-budget, I10):
///
/// ```text
/// pred   = FL_MEDIUM_BUILD_GMACS / measured_rate x FL_ADMISSION_MARGIN
/// fits   = pred + FL_ALPHA_SLICE_MIN_SECS + FL_BAB_FLOOR_SECS <= internal_tier
/// window = ceil(clamp(pred + FL_ALPHA_SLICE_MIN_SECS,
///                     MEDIUM_PAIR_ROOT_ALPHA_CAP_SECS,   // never narrower than the banked recipe
///                     internal_tier - FL_BAB_FLOOR_SECS))
/// ```
///
/// Declines when: the rate is the unmeasured fallback constant; the phase
/// plan does not fit the tier; or the clamped window would not actually be
/// wider than the 40s recipe (FL already fits the standing window — widening
/// must only ever RAISE the cap, so a `<= 40` result applies nothing).
fn fl_phase_budget(rate: &FlRateObservation, internal_tier_secs: u64) -> FlPhaseBudget {
    let rate_gmacs = rate.macs_per_sec as f64 / 1e9;
    if rate.source == "fallback" || !rate_gmacs.is_finite() || rate_gmacs <= 0.0 {
        return FlPhaseBudget::Decline {
            note: format!(
                "declined: rate source '{}' is not a measurement — the stale-constant lockout \
                 is the failure mode this rule exists to end, not to act on",
                rate.source
            ),
        };
    }
    let pred_secs = (FL_MEDIUM_BUILD_GMACS / rate_gmacs) * FL_ADMISSION_MARGIN;
    let tier = internal_tier_secs as f64;
    if pred_secs + FL_ALPHA_SLICE_MIN_SECS + FL_BAB_FLOOR_SECS > tier {
        return FlPhaseBudget::Decline {
            note: format!(
                "declined: pred {pred_secs:.1}s ({FL_MEDIUM_BUILD_GMACS} GMAC / {rate_gmacs:.2} \
                 GMAC/s x {FL_ADMISSION_MARGIN}) + {FL_ALPHA_SLICE_MIN_SECS:.0}s α-slice + \
                 {FL_BAB_FLOOR_SECS:.0}s BaB floor > {tier:.0}s tier; cap unchanged \
                 (banked cap-40 recipe stands)"
            ),
        };
    }
    let window_secs = (pred_secs + FL_ALPHA_SLICE_MIN_SECS)
        .min(tier - FL_BAB_FLOOR_SECS)
        .max(MEDIUM_PAIR_ROOT_ALPHA_CAP_SECS)
        .ceil();
    if window_secs <= MEDIUM_PAIR_ROOT_ALPHA_CAP_SECS {
        return FlPhaseBudget::Decline {
            note: format!(
                "declined: pred {pred_secs:.1}s fits inside the standing 40s window \
                 (rate {rate_gmacs:.2} GMAC/s); no widening needed and narrowing is never applied"
            ),
        };
    }
    FlPhaseBudget::Widen {
        window_secs,
        reason: format!(
            "{RULE_FL_PHASE_BUDGET}; this run: measured rate {rate_gmacs:.2} GMAC/s \
             ({}, probe {:.3}s) predicts {pred_secs:.1}s cold FL build ({FL_MEDIUM_BUILD_GMACS} \
             GMAC x {FL_ADMISSION_MARGIN} admission margin); {pred_secs:.1}s + \
             {FL_ALPHA_SLICE_MIN_SECS:.0}s α-slice + {FL_BAB_FLOOR_SECS:.0}s BaB floor <= \
             {tier:.0}s tier => root window {window_secs:.0}s (clamped to [40s banked recipe, \
             tier - BaB floor])",
            rate.source, rate.probe_secs
        ),
    }
}

/// Resolve the run plan for one instance.
///
/// Pure: no filesystem, no env, no process globals. Callers supply the
/// backend report from `compute_backend::detect()` and the preset from the
/// same category->yaml resolution the scored path uses.
///
/// LAYERING CONTRACT: a key the preset sets explicitly ALWAYS wins and is
/// tagged `PresetOverride`; a measured rule fills only absent keys. When an
/// explicit key blocks a rule that would have fired, the rendered source
/// line says so — visibility without displacement.
// Retained as the zero-override entry point onto the decision table; the
// shipped callers all pass typed runtime facts, so only the in-file tests
// reach this arity today.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn resolve_plan(
    model_facts: &ModelFacts,
    budget_secs: u64,
    backend: &BackendReport,
    preset: Option<&PresetConfig>,
) -> ResolvedPlan {
    resolve_plan_with_fl_rate_and_runtime(
        model_facts,
        budget_secs,
        backend,
        preset,
        None,
        PlanRuntimeOverrides::default(),
    )
}

/// [`resolve_plan`] with an optional measured forward-linear rate
/// (#fl-phase-budget, I10). Still pure: the rate is DATA measured by the
/// caller (scored path / printer probe only where [`fl_rate_scope_applies`]),
/// so tests pin the widening rule on injected rates, not on what this host
/// happens to measure.
// Same story as `resolve_plan`: the rate-only arity is how the tests pin the
// widening rule on injected rates; shipped callers use the runtime form.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn resolve_plan_with_fl_rate(
    model_facts: &ModelFacts,
    budget_secs: u64,
    backend: &BackendReport,
    preset: Option<&PresetConfig>,
    fl_rate: Option<&FlRateObservation>,
) -> ResolvedPlan {
    resolve_plan_with_fl_rate_and_runtime(
        model_facts,
        budget_secs,
        backend,
        preset,
        fl_rate,
        PlanRuntimeOverrides::default(),
    )
}

/// Pure resolver with command-boundary runtime overrides supplied as data.
pub(crate) fn resolve_plan_with_fl_rate_and_runtime(
    model_facts: &ModelFacts,
    budget_secs: u64,
    backend: &BackendReport,
    preset: Option<&PresetConfig>,
    fl_rate: Option<&FlRateObservation>,
    runtime: PlanRuntimeOverrides,
) -> ResolvedPlan {
    let mut settings: Vec<SettingLine> = Vec::new();
    let conv_scale = model_facts.conv_scale();
    // Nominal tier before dynamic wrapper deductions (upfront attack elapsed,
    // margin-row/post-BaB reserves, and special deadline routes).
    let internal_tier_secs = crate::commands::vnncomp::internal_timeout_secs(budget_secs);

    // --- Backend (rule 4): detection decides accelerator roles (I3). ---
    let backend_source = SettingSource::HostFact(RULE_BACKEND_DETECT);
    settings.push(SettingLine {
        name: "backend",
        value: backend.kind.to_string(),
        source: backend_source.to_string(),
    });
    // I3 contract: accelerators PROPOSE, the sound fold CERTIFIES. CUDA f64
    // GEMM seams are inside the sound boundary; Metal/wgpu f32 is proposals
    // only; cpu-only is the proven-sound f64 path itself.
    let accelerator_role = match backend.kind {
        "cuda" => "proposals + certified f64 GEMM seams",
        "metal" | "gpu" => "proposals only (f32; outside the sound boundary)",
        _ => "none (CPU f64 sound path only)",
    };
    settings.push(SettingLine {
        name: "accelerator_role",
        value: accelerator_role.to_string(),
        source: backend_source.to_string(),
    });

    // --- #flush-charge (RECORDED, like attack_steering: a fact, not a rule):
    // the charged-Metal proof route's build-time gate state, so `ny vnncomp
    // plan` shows whether the scored chain can even attempt charged
    // qualification here. Emitted only where a WGPU adapter regime exists;
    // cuda/cpu-only plan output stays byte-identical. The per-run
    // qualified/refused outcome belongs to the ProofBackendReceipt, not the
    // plan.
    if matches!(backend.kind, "metal" | "gpu") {
        settings.push(SettingLine {
            name: "wgpu_charged_authority",
            value: if ny_gpu::wgpu_charged_proof_authority() {
                "armed (source gate OPEN; live pure-flush ladder decides per device)".to_string()
            } else {
                "dark (source gate closed; charged constructor refuses)".to_string()
            },
            source: SettingSource::HostFact(RULE_CHARGED_METAL_GATE).to_string(),
        });
    }

    // --- Attack steering (rule 3): always armed, async — RECORDED. The
    // engine's arming route (`AttackSteering`) exists wherever the host is
    // not cpu-only; on the CPU route it is permanently disarmed. ---
    let attack_steering_armed = ResolvedSetting::new(
        backend.kind != "cpu-only",
        // A HOST fact, not a per-category choice: the arming route exists
        // wherever the host is not cpu-only, and nothing materializes it.
        SettingSource::HostFact(RULE_ATTACK_STEERING),
    );
    settings.push(SettingLine {
        name: "attack_steering",
        value: if attack_steering_armed.value {
            "armed (async)".to_string()
        } else {
            "disarmed (cpu-only route)".to_string()
        },
        source: attack_steering_armed.source.to_string(),
    });

    // Runtime falsifier admission is resolved from the same preset snapshot as
    // β-CROWN, plus the exact command-boundary diagnostic skip. Preserve a
    // scheduling error as typed provenance: actual execution rejects it, so a
    // reporter that silently substitutes default-on would be misleading.
    let preset_pgd_schedule = preset.map(crate::preset::resolve_initial_pgd_schedule);
    let disjunctive_pgd_enabled = if let Some(Err(error)) = &preset_pgd_schedule {
        ResolvedSetting::new(false, SettingSource::InvalidPreset(error.to_string()))
    } else if runtime.disjunctive_pgd_skipped {
        ResolvedSetting::new(
            false,
            SettingSource::RuntimeOverride(RULE_DISJUNCTIVE_PGD_SKIP_ENV),
        )
    } else if let Some(Ok(Some(schedule))) = preset_pgd_schedule {
        ResolvedSetting::new(
            !matches!(
                schedule,
                crate::preset::ResolvedInitialPgdSchedule::Disabled
            ),
            SettingSource::PresetOverride("attack.pgd_order".into()),
        )
    } else {
        ResolvedSetting::new(true, SettingSource::Default)
    };
    settings.push(SettingLine {
        name: "disjunctive_pgd_attack_enabled",
        value: disjunctive_pgd_enabled.value.to_string(),
        source: disjunctive_pgd_enabled.source.to_string(),
    });

    let pgd_time_cap_disabled = ResolvedSetting::new(
        runtime.pgd_time_cap_disabled,
        if runtime.pgd_time_cap_disabled {
            SettingSource::RuntimeOverride(RULE_PGD_TIME_CAP_ENV)
        } else {
            SettingSource::Default
        },
    );
    settings.push(SettingLine {
        name: "pgd_time_cap_disabled",
        value: pgd_time_cap_disabled.value.to_string(),
        source: pgd_time_cap_disabled.source.to_string(),
    });

    // --- Rule 1: disjunctive PGD slice + root alpha cap, from MODEL FACTS,
    // never filenames — under the preset. ---
    let preset_fraction = preset.and_then(|p| p.bab.phase_budget.disjunctive_pgd_fraction);
    let preset_alpha_cap = preset.and_then(|p| p.bab.root_alpha_cap_secs);
    let medium_short =
        conv_scale == ConvScale::Medium && budget_secs <= MEDIUM_PAIR_MAX_BUDGET_SECS;
    // The PAIR fires only when the preset owns NEITHER key (bd6a9eff
    // measured the two levers inseparable — each alone stayed timeout).
    let pair_fires = medium_short && preset_fraction.is_none() && preset_alpha_cap.is_none();

    let disjunctive_pgd_fraction = if let Some(v) = preset_fraction {
        ResolvedSetting::new(
            Some(v),
            SettingSource::PresetOverride("bab.phase_budget.disjunctive_pgd_fraction".into()),
        )
    // #rule-contract (2026-08-01): the Large-conv arm that used to sit here is
    // REMOVED. It cut `disjunctive_pgd_fraction` 0.50 -> 0.40 for every category
    // whose preset omits the key — 34 of 48 — citing
    // `CIFAR100_REGRESSION_ATTRIBUTION_2026-07-31.md` for "24 banked sats lost".
    // That figure appears NOWHERE in that document, and the document retracts
    // the slice-starvation attribution for two of the three rows it rested on
    // (§ "the 'PGD slice starvation' attribution for L1063 and L5308 is
    // retracted"); only L3585 survives. No 0.40-vs-0.50 A/B exists in the tree.
    //
    // Decisively: `configs/vnncomp25/cifar100_2024.yaml:128` already sets
    // `disjunctive_pgd_fraction: 0.05`, so on the ONE category where the
    // evidence lives the preset owns the key and this arm could never fire
    // there. Its entire blast radius was categories the evidence does not
    // cover — and in the completeness-losing direction, since shortening
    // falsification can turn a sat into an unknown.
    } else if pair_fires {
        ResolvedSetting::new(
            Some(MEDIUM_PAIR_PGD_FRACTION),
            SettingSource::ResolvedModel(RULE_MEDIUM_PAIR),
        )
    } else {
        ResolvedSetting::new(None, SettingSource::Default)
    };
    // Visibility without displacement: name the rule an explicit preset key
    // blocks, or the pair leg withheld because the OTHER key is preset-owned.
    let slice_note = if preset_fraction.is_none() && medium_short && preset_alpha_cap.is_some() {
        " (pair withheld: root_alpha_cap_secs is preset-owned; the bd6a9eff pair applies \
         together or not at all)"
            .to_string()
    } else {
        String::new()
    };
    settings.push(SettingLine {
        name: "disjunctive_pgd_fraction",
        value: format!(
            "{:.2}",
            disjunctive_pgd_fraction
                .value
                .unwrap_or(DEFAULT_DISJUNCTIVE_PGD_FRACTION)
        ),
        source: format!("{}{slice_note}", disjunctive_pgd_fraction.source),
    });

    // --- Rule 6 (#fl-phase-budget, I10): FL-aware root-window sizing. Scope
    // re-checked here even though callers gate the probe on
    // `fl_rate_scope_applies` (defense in depth: an explicit preset key or a
    // non-medium class must pin the outcome regardless of what a caller
    // injected — presets-win invariant). The rule only ever RE-SIZES a window
    // the resolver itself owns, UPWARD: the widened cap replaces the pair's
    // 40s (or establishes the phase-budgeted window at longer tiers where the
    // pair declines), and the `fl_phase_budget` clamp guarantees it is never
    // below the banked 40s recipe. Window sizing stays independent of the FL
    // gate's own admission: the gate still measures `remaining >= pred x 5/4`
    // against the alpha deadline this cap min-composes into.
    let fl_decision = fl_rate
        .filter(|_| {
            conv_scale == ConvScale::Medium
                && preset_fraction.is_none()
                && preset_alpha_cap.is_none()
        })
        .map(|rate| fl_phase_budget(rate, internal_tier_secs));
    let (fl_widened, fl_note): (Option<(f64, String)>, Option<String>) = match fl_decision {
        Some(FlPhaseBudget::Widen {
            window_secs,
            reason,
        }) => (
            Some((window_secs, reason)),
            Some(format!("widened root window to {window_secs:.0}s")),
        ),
        Some(FlPhaseBudget::Decline { note }) => (None, Some(note)),
        None => (None, None),
    };
    let fl_applied = fl_widened.is_some();

    let root_alpha_cap_secs = if let Some(v) = preset_alpha_cap {
        ResolvedSetting::new(
            Some(v),
            SettingSource::PresetOverride("bab.root_alpha_cap_secs".into()),
        )
    } else if let Some((window_secs, reason)) = fl_widened {
        ResolvedSetting::new(Some(window_secs), SettingSource::ResolvedBudget(reason))
    } else if pair_fires {
        // The budget leg of the pair: the cap exists because a short budget
        // cannot afford an uncapped root warmup.
        ResolvedSetting::new(
            Some(MEDIUM_PAIR_ROOT_ALPHA_CAP_SECS),
            SettingSource::ResolvedBudget(RULE_MEDIUM_PAIR.to_string()),
        )
    } else {
        ResolvedSetting::new(None, SettingSource::Default)
    };
    let cap_note = if preset_alpha_cap.is_none() && medium_short && preset_fraction.is_some() {
        " (pair withheld: disjunctive_pgd_fraction is preset-owned; the bd6a9eff pair applies \
         together or not at all)"
    } else {
        ""
    };
    // Visibility either way (#fl-phase-budget): a DECLINED widening is
    // printed on the cap line (a widened one already carries its numbers in
    // the ResolvedBudget reason itself).
    let fl_cap_note = match (&fl_note, fl_applied) {
        (Some(note), false) => format!(" (fl-phase-budget {note})"),
        _ => String::new(),
    };
    settings.push(SettingLine {
        name: "root_alpha_cap_secs",
        value: root_alpha_cap_secs
            .value
            .map_or_else(|| "none".to_string(), |secs| format!("{secs:.0}")),
        source: format!("{}{cap_note}{fl_cap_note}", root_alpha_cap_secs.source),
    });

    // --- Rule 7 (#fl-alpha-composition, I10): FL → margin-row α as one
    // authoritative path. Fires ONLY where rule 6 actually widened this run —
    // rule 6's provenance chain (medium conv class, measured rate, phase plan
    // fits the tier, both pair keys preset-absent) is reused wholesale rather
    // than re-derived, so the composition can never arm where FL itself was
    // refused. Presets-win: either surrogate key set explicitly (true OR
    // false) pins the outcome and is forwarded as PresetOverride. The armed
    // lever is intersect-only at the root (the fixed FL C-margin candidate is
    // the floor), so the worst case of a wrong arming is unused grace, never
    // a weaker bound. ---
    let preset_surrogate = preset.and_then(|p| {
        p.model
            .forward_alpha_surrogate
            .map(|v| (v, "model.forward_alpha_surrogate"))
            .or_else(|| {
                p.model
                    .forward_linear_spec_alpha
                    .map(|v| (v, "model.forward_linear_spec_alpha"))
            })
    });
    let forward_alpha_surrogate = match preset_surrogate {
        Some((v, key)) => ResolvedSetting::new(Some(v), SettingSource::PresetOverride(key.into())),
        None if fl_applied => ResolvedSetting::new(
            Some(true),
            SettingSource::ResolvedBudget(RULE_FL_ALPHA_COMPOSITION.to_string()),
        ),
        None => ResolvedSetting::new(None, SettingSource::Default),
    };
    settings.push(SettingLine {
        name: "forward_alpha_surrogate",
        value: forward_alpha_surrogate
            .value
            .map_or_else(|| "unset".to_string(), |v| v.to_string()),
        source: forward_alpha_surrogate.source.to_string(),
    });

    // Absolute slice cap/floor stay preset-driven knobs (the resolver has no
    // model-facts rule for them yet); print their effective values/sources so
    // the nominal math below is auditable, including an explicit default none.
    let preset_slice_cap = preset.and_then(|p| p.bab.phase_budget.disjunctive_pgd_max_secs);
    let disjunctive_pgd_max_secs = ResolvedSetting::new(
        preset_slice_cap,
        match preset_slice_cap {
            Some(_) => {
                SettingSource::PresetOverride("bab.phase_budget.disjunctive_pgd_max_secs".into())
            }
            None => SettingSource::Default,
        },
    );
    settings.push(SettingLine {
        name: "disjunctive_pgd_max_secs",
        value: preset_slice_cap.map_or_else(|| "none".to_string(), |cap| cap.to_string()),
        source: disjunctive_pgd_max_secs.source.to_string(),
    });
    // #attack-anchor: retain both the effective value and its source for every
    // plan surface and the scored flight record.
    let preset_phase_anchor =
        preset.and_then(|p| p.bab.phase_budget.disjunctive_pgd_from_phase_start);
    let disjunctive_pgd_from_phase_start = ResolvedSetting::new(
        preset_phase_anchor.unwrap_or(false),
        match preset_phase_anchor {
            Some(_) => SettingSource::PresetOverride(
                "bab.phase_budget.disjunctive_pgd_from_phase_start".into(),
            ),
            None => SettingSource::Default,
        },
    );
    settings.push(SettingLine {
        name: "disjunctive_pgd_from_phase_start",
        value: disjunctive_pgd_from_phase_start.value.to_string(),
        source: disjunctive_pgd_from_phase_start.source.to_string(),
    });
    let preset_slice_floor = preset.and_then(|p| p.bab.phase_budget.disjunctive_pgd_min_secs);
    let disjunctive_pgd_min_secs = ResolvedSetting::new(
        preset_slice_floor,
        match preset_slice_floor {
            Some(_) => {
                SettingSource::PresetOverride("bab.phase_budget.disjunctive_pgd_min_secs".into())
            }
            None => SettingSource::Default,
        },
    );
    settings.push(SettingLine {
        name: "disjunctive_pgd_min_secs",
        value: preset_slice_floor.map_or_else(|| "none".to_string(), |floor| floor.to_string()),
        source: disjunctive_pgd_min_secs.source.to_string(),
    });

    // --- Rule 2: margin-row reserve — preset-owned, with the fixed engine
    // default retained when absent. An explicit preset policy wins
    // (layering), including an explicit `adaptive_reserve: false`. ---
    let margin_preset = preset.map(|p| &p.margin_row);
    let margin_row = match margin_preset {
        Some(mr) if mr.adaptive_reserve == Some(true) => ResolvedSetting::new(
            MarginRowPolicy::Adaptive,
            SettingSource::PresetOverride("margin_row.adaptive_reserve".into()),
        ),
        Some(mr) if mr.adaptive_reserve == Some(false) => ResolvedSetting::new(
            match mr.reserve_secs {
                Some(0) => MarginRowPolicy::NoReserve,
                Some(secs) => MarginRowPolicy::Fixed(secs),
                // 45 mirrors `margin_row_bab::margin_row_reserve_secs`'s
                // shipped default when the preset pins only the policy bit.
                None => MarginRowPolicy::Fixed(45),
            },
            SettingSource::PresetOverride("margin_row.adaptive_reserve".into()),
        ),
        Some(mr) if mr.reserve_secs == Some(0) => ResolvedSetting::new(
            MarginRowPolicy::NoReserve,
            SettingSource::PresetOverride("margin_row.reserve_secs".into()),
        ),
        // #rule2-sign-inversion (2026-08-01): this arm USED to resolve
        // `Adaptive` board-wide, citing b61b5f10. b61b5f10 says the opposite:
        //
        //   "the third lever, margin_row.adaptive_reserve, is independently
        //    harmful -- alone it costs both near-wall GT-unsat rows, which
        //    finish at 80.8s/84.1s and need the reserve."
        //
        // Adaptive RELEASES the reserve; those rows need it HELD. The 8/15 ->
        // 3/15 A/B the rule quoted was the whole allocation-trio port, not the
        // fixed fallback, and `8f5c7299` had already reverted that port at the
        // preset layer -- this arm re-landed the same behaviour one layer
        // below, where preset review does not see it. Exactly one of 48 presets
        // sets the key, so 47 categories were flipped on evidence measured in
        // one (cifar100) and contradicted in another (tinyimagenet).
        //
        // A rule may not fire board-wide on single-category evidence. With no
        // preset key the ENGINE DEFAULT stands (`SettingSource::Default` =>
        // `is_resolved()` false => materialization inserts nothing), so a
        // category that has never been A/B'd is byte-identical to pre-resolver.
        // The displayed value mirrors `margin_row_bab::margin_row_reserve_secs`'s
        // documented order (`NY_MARGIN_ROW_RESERVE_SECS` > preset > 45); it is
        // informational only and deliberately not a second source of truth.
        _ => ResolvedSetting::new(MarginRowPolicy::Fixed(45), SettingSource::Default),
    };
    let margin_note = if margin_preset.is_some_and(|mr| mr.adaptive_reserve == Some(false)) {
        format!(" (preset declines measured rule: {RULE_MARGIN_ADAPTIVE})")
    } else {
        String::new()
    };
    settings.push(SettingLine {
        name: "margin_row.policy",
        value: margin_row.value.to_string(),
        source: format!("{}{margin_note}", margin_row.source),
    });
    if let Some(reserve) = margin_preset
        .and_then(|mr| mr.reserve_secs)
        .filter(|&secs| secs > 0)
    {
        settings.push(SettingLine {
            name: "margin_row.reserve_secs",
            value: reserve.to_string(),
            source: SettingSource::PresetOverride("margin_row.reserve_secs".into()).to_string(),
        });
    }

    // --- Rule 5: alpha_spec_slots stays preset/experimental (acceptance
    // open, #spec-axis-alpha); the resolver forwards and records. ---
    let spec_slots = preset.and_then(|p| {
        p.solver
            .alpha_crown
            .spec_slots
            .map(|v| (v, "solver.alpha_crown.spec_slots"))
            .or_else(|| {
                p.bab
                    .alpha_crown
                    .spec_slots
                    .map(|v| (v, "bab.alpha_crown.spec_slots"))
            })
    });
    let alpha_spec_slots = match spec_slots {
        Some((v, key)) => ResolvedSetting::new(Some(v), SettingSource::PresetOverride(key.into())),
        None => ResolvedSetting::new(None, SettingSource::Default),
    };
    settings.push(SettingLine {
        name: "alpha_spec_slots",
        value: alpha_spec_slots.value.unwrap_or(0).to_string(),
        source: format!(
            "{}{}",
            alpha_spec_slots.source,
            if alpha_spec_slots.value.is_some() {
                " (pass-through: #spec-axis-alpha acceptance still open)"
            } else {
                ""
            }
        ),
    });

    // --- Preset-driven settings the resolver has no rule for (printed so
    // the plan is the whole picture, sourced so the layering is honest). ---
    let complete_verifier = preset.and_then(|p| p.general.complete_verifier.clone());
    settings.push(SettingLine {
        name: "complete_verifier",
        value: complete_verifier
            .clone()
            .unwrap_or_else(|| "auto".to_string()),
        source: match complete_verifier {
            Some(_) => {
                SettingSource::PresetOverride("general.complete_verifier".into()).to_string()
            }
            None => SettingSource::Default.to_string(),
        },
    });
    let batch_size = preset.and_then(|p| p.bab.batch_size.or(p.solver.batch_size));
    settings.push(SettingLine {
        name: "bab.batch_size",
        value: batch_size.unwrap_or(64).to_string(),
        source: match batch_size {
            Some(_) => SettingSource::PresetOverride("bab.batch_size".into()).to_string(),
            None => SettingSource::Default.to_string(),
        },
    });
    let branching_method = preset.and_then(|p| p.bab.branching.method.clone());
    settings.push(SettingLine {
        name: "bab.branching.method",
        value: branching_method
            .clone()
            .unwrap_or_else(|| "auto (model-class selection)".to_string()),
        source: match branching_method {
            Some(_) => SettingSource::PresetOverride("bab.branching.method".into()).to_string(),
            None => SettingSource::Default.to_string(),
        },
    });

    // --- The I6 nominal ledger snapshot. Dynamic wrapper deductions and
    // elapsed wall time do not exist yet at plan resolution; say "nominal"
    // explicitly rather than claiming to predict that future remaining time.
    // Within that scope the pure helper mirrors PhaseBudgetLedger's policy. ---
    let effective_fraction = disjunctive_pgd_fraction
        .value
        .unwrap_or(DEFAULT_DISJUNCTIVE_PGD_FRACTION);
    let nominal_attack_slice_secs = disjunctive_pgd_enabled.value.then(|| {
        planned_disjunctive_pgd_slice_secs(
            internal_tier_secs,
            effective_fraction,
            preset_slice_cap,
            preset_slice_floor,
            pgd_time_cap_disabled.value,
        )
    });

    ResolvedPlan {
        backend_kind: backend.kind,
        backend_summary: backend.summary.clone(),
        conv_scale,
        ledger: BudgetLedgerSnapshot {
            scored_budget_secs: budget_secs,
            nominal_internal_tier_secs: internal_tier_secs,
            nominal_attack_slice_secs,
            root_alpha_cap_secs: root_alpha_cap_secs.value,
        },
        disjunctive_pgd_fraction,
        disjunctive_pgd_enabled,
        pgd_time_cap_disabled,
        disjunctive_pgd_max_secs,
        disjunctive_pgd_min_secs,
        disjunctive_pgd_from_phase_start,
        root_alpha_cap_secs,
        margin_row,
        attack_steering_armed,
        alpha_spec_slots,
        forward_alpha_surrogate,
        fl_phase_budget: fl_note,
        settings,
    }
}

// ---------------------------------------------------------------------------
// Materialization — apply resolved values through the one preset channel
// ---------------------------------------------------------------------------

/// The plan plus the preset path the scored instance should actually run
/// with, and the keep-alive guard for the merged temp preset.
pub(crate) struct MaterializedPlan {
    pub(crate) plan: ResolvedPlan,
    /// Facts the plan was resolved from; `None` = scan unavailable, every
    /// model rule declined.
    pub(crate) facts: Option<ModelFacts>,
    /// The preset to hand downstream: the merged temp file when the resolver
    /// applied anything, else the original path (or `None`, unchanged).
    ///
    /// PRIVATE on purpose: this path is DATA and `_temp_guard` below controls
    /// the merged file's lifetime. A public field permits an accidental
    /// partial move of the `PathBuf`, separating normal path use from the value
    /// that owns the guard. [`MaterializedPlan::effective_preset`] instead
    /// makes the safe current call chain borrow by default. A caller that
    /// deliberately copies the path must still retain this plan until all
    /// readers finish.
    effective_preset: Option<PathBuf>,
    /// Human-readable degradation note (`None` on the clean path).
    pub(crate) note: Option<String>,
    /// Dropping this struct at the end of the instance deletes the merged
    /// file. Every downstream consumer re-loads the preset before the
    /// verdict is published, so a guard scoped to the whole handler covers
    /// all current readers. The borrowed accessor prevents an accidental field
    /// move; it does not make an independently copied path keep this guard
    /// alive.
    _temp_guard: Option<tempfile::NamedTempFile>,
}

impl MaterializedPlan {
    /// The preset path the scored instance must actually run with, borrowed
    /// from the value that also owns the merged file's keep-alive guard.
    ///
    /// The returned reference cannot outlive `self`, which prevents direct
    /// borrowed use after the plan drops and rules out folding the resolver
    /// call into a temporary. This is guidance, not an owning-path capability:
    /// `Path::to_path_buf` can copy the name, so any such caller must also keep
    /// the `MaterializedPlan` alive.
    pub(crate) fn effective_preset(&self) -> Option<&Path> {
        self.effective_preset.as_deref()
    }

    /// The one-line `plan_resolved` flight payload: value:source pairs, the
    /// model facts, and the effective preset path.
    pub(crate) fn flight_summary(&self) -> String {
        let p = &self.plan;
        let nominal_attack_slice = p
            .ledger
            .nominal_attack_slice_secs
            .map_or_else(|| "disabled".to_string(), |secs| format!("{secs:.6}"));
        let mut out = format!(
            "disjunctive_pgd_attack_enabled={}:[{}]; pgd_time_cap_disabled={}:[{}]; \
             disjunctive_pgd_fraction={}:[{}]; disjunctive_pgd_max_secs={}:[{}]; \
             disjunctive_pgd_min_secs={}:[{}]; disjunctive_pgd_from_phase_start={}:[{}]; \
             nominal_internal_tier_secs={}; nominal_attack_slice_secs={}; \
             root_alpha_cap_secs={}:[{}]; \
             margin_row={}:[{}]; attack_steering={}:[{}]; alpha_spec_slots={}:[{}]; \
             backend={}:[{}]; {}; effective_preset={}",
            p.disjunctive_pgd_enabled.value,
            p.disjunctive_pgd_enabled.source,
            p.pgd_time_cap_disabled.value,
            p.pgd_time_cap_disabled.source,
            p.disjunctive_pgd_fraction
                .value
                .map_or_else(|| "default".to_string(), |v| format!("{v}")),
            p.disjunctive_pgd_fraction.source,
            p.disjunctive_pgd_max_secs
                .value
                .map_or_else(|| "none".to_string(), |v| v.to_string()),
            p.disjunctive_pgd_max_secs.source,
            p.disjunctive_pgd_min_secs
                .value
                .map_or_else(|| "none".to_string(), |v| v.to_string()),
            p.disjunctive_pgd_min_secs.source,
            p.disjunctive_pgd_from_phase_start.value,
            p.disjunctive_pgd_from_phase_start.source,
            p.ledger.nominal_internal_tier_secs,
            nominal_attack_slice,
            p.root_alpha_cap_secs
                .value
                .map_or_else(|| "none".to_string(), |v| format!("{v}")),
            p.root_alpha_cap_secs.source,
            p.margin_row.value,
            p.margin_row.source,
            if p.attack_steering_armed.value {
                "armed-async"
            } else {
                "disarmed(cpu-only)"
            },
            p.attack_steering_armed.source,
            p.alpha_spec_slots
                .value
                .map_or_else(|| "unset".to_string(), |v| v.to_string()),
            p.alpha_spec_slots.source,
            p.backend_kind,
            // #rule-contract: a HOST fact. Constructed inline here rather than
            // read off a field, which is exactly how it escaped the retag —
            // guarded now by `every_resolved_source_is_materializable`.
            SettingSource::HostFact(RULE_BACKEND_DETECT),
            self.facts
                .as_ref()
                .map_or_else(|| "facts{unavailable}".to_string(), ModelFacts::summary),
            self.effective_preset
                .as_ref()
                .map_or_else(|| "none".to_string(), |p| p.display().to_string()),
        );
        // #fl-phase-budget: the decision rides in the flight event EITHER WAY
        // (widened or declined); absent scope adds no FL-specific segment.
        if let Some(fl) = &self.plan.fl_phase_budget {
            out.push_str("; fl_phase_budget=");
            out.push_str(fl);
        }
        // Rule 7 (#fl-alpha-composition): recorded only when the key exists
        // (preset-forwarded or resolved), so unaffected categories' flight
        // lines stay byte-identical.
        if let Some(armed) = self.plan.forward_alpha_surrogate.value {
            out.push_str(&format!(
                "; forward_alpha_surrogate={armed}:[{}]",
                self.plan.forward_alpha_surrogate.source
            ));
        }
        if let Some(note) = &self.note {
            out.push_str("; note=");
            out.push_str(note);
        }
        if matches!(
            self.plan.disjunctive_pgd_enabled.source,
            SettingSource::RuntimeOverride(RULE_DISJUNCTIVE_PGD_SKIP_ENV)
        ) {
            // This warmer is specific to the env-skip branch in
            // `verify/disjunctive.rs`: a preset `pgd_order: skip` makes
            // `pgd_attack` false and never enters that branch.
            out.push_str(
                "; skipped_pgd_forward_linear_warmer=conditional-conv-route-synchronous-overall-deadline; \
                 nominal_attack_slice_excludes_warmer=true",
            );
        }
        out
    }
}

/// Resolve and materialize with default (no diagnostic) runtime inputs.
///
/// Total function: every failure path degrades to the ORIGINAL
/// preset (preset-only behavior) and says so in the note — the resolver must
/// never be able to lose a preset or kill an instance.
/// `fl_rate_probe` (#fl-phase-budget, I10): measures the forward-linear build
/// rate — scored callers pass [`probe_fl_rate`]. Called AT MOST ONCE, and
/// ONLY when [`fl_rate_scope_applies`] (medium conv class, both pair keys
/// preset-absent): categories a preset pins never pay the probe, and tests
/// inject fixed rates (or `|| None`) so nothing here depends on the build
/// host's throughput.
// The scored path calls `resolve_and_materialize_with_runtime` directly (it
// has already decoded its process-global runtime inputs); this default-runtime
// arity is exercised only by the in-file tests.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn resolve_and_materialize(
    onnx: &Path,
    preset_path: Option<&Path>,
    scored_budget_secs: u64,
    backend: &BackendReport,
    fl_rate_probe: impl FnOnce() -> Option<FlRateObservation>,
) -> MaterializedPlan {
    resolve_and_materialize_with_runtime(
        onnx,
        preset_path,
        scored_budget_secs,
        backend,
        PlanRuntimeOverrides::default(),
        fl_rate_probe,
    )
}

/// Scored/printer boundary variant with process-global runtime decisions
/// already captured as typed data.
pub(crate) fn resolve_and_materialize_with_runtime(
    onnx: &Path,
    preset_path: Option<&Path>,
    scored_budget_secs: u64,
    backend: &BackendReport,
    runtime: PlanRuntimeOverrides,
    fl_rate_probe: impl FnOnce() -> Option<FlRateObservation>,
) -> MaterializedPlan {
    let facts = ModelFacts::from_onnx_file(onnx);
    // Scan unavailable => zero facts: `conv_layers == 0` classifies
    // SmallOrNone and every model-scale rule declines.
    let unknown = ModelFacts {
        param_count: 0,
        conv_layers: 0,
        max_conv_out_channels: 0,
        file_size_bytes: 0,
    };

    let preset_cfg = match preset_path {
        Some(path) => match crate::preset::load_preset(path) {
            Ok(cfg) => Some(cfg),
            Err(err) => {
                // Unreadable preset: resolve nothing, change nothing. The
                // downstream loader surfaces the same error on its own
                // authority; synthesizing a replacement here would mask a
                // broken category config. The plan still records facts,
                // backend, and steering for the flight note.
                let plan = resolve_plan_with_fl_rate_and_runtime(
                    facts.as_ref().unwrap_or(&unknown),
                    scored_budget_secs,
                    backend,
                    None,
                    None,
                    runtime,
                );
                return MaterializedPlan {
                    plan,
                    facts,
                    effective_preset: preset_path.map(Path::to_path_buf),
                    note: Some(format!("preset unreadable, resolver declined: {err}")),
                    _temp_guard: None,
                };
            }
        },
        None => None,
    };

    // #fl-phase-budget: pay the rate probe only where the rule could fire —
    // scope declined (or facts unavailable) means no probe, no rate, and a
    // byte-identical plan for every preset-pinned category.
    let fl_rate = facts
        .as_ref()
        .filter(|f| fl_rate_scope_applies(f, preset_cfg.as_ref()))
        .and_then(|_| fl_rate_probe());

    let plan = resolve_plan_with_fl_rate_and_runtime(
        facts.as_ref().unwrap_or(&unknown),
        scored_budget_secs,
        backend,
        preset_cfg.as_ref(),
        fl_rate.as_ref(),
        runtime,
    );

    // Only these three are APPLIED in v1; steering/backend are recorded and
    // alpha_spec_slots is a pass-through.
    let mut insertions: Vec<(&[&str], serde_yaml::Value)> = Vec::new();
    if plan.disjunctive_pgd_fraction.is_resolved() {
        if let Some(v) = plan.disjunctive_pgd_fraction.value {
            // Serialize via the f32's shortest decimal form so the merged
            // yaml reads `0.05`, not the f64-widened `0.05000000074...`.
            let as_f64 = v
                .to_string()
                .parse::<f64>()
                .unwrap_or_else(|_| f64::from(v));
            insertions.push((
                &["bab", "phase_budget", "disjunctive_pgd_fraction"],
                serde_yaml::Value::from(as_f64),
            ));
        }
    }
    if plan.root_alpha_cap_secs.is_resolved() {
        if let Some(v) = plan.root_alpha_cap_secs.value {
            insertions.push((&["bab", "root_alpha_cap_secs"], serde_yaml::Value::from(v)));
        }
    }
    if plan.margin_row.is_resolved() && plan.margin_row.value == MarginRowPolicy::Adaptive {
        insertions.push((
            &["margin_row", "adaptive_reserve"],
            serde_yaml::Value::from(true),
        ));
    }
    // Rule 7 (#fl-alpha-composition): a resolved arming is applied through the
    // same one preset channel — `insert_absent` guarantees an authored
    // surrogate key is never overwritten (and the rule already declined then).
    if plan.forward_alpha_surrogate.is_resolved()
        && plan.forward_alpha_surrogate.value == Some(true)
    {
        insertions.push((
            &["model", "forward_alpha_surrogate"],
            serde_yaml::Value::from(true),
        ));
    }

    if insertions.is_empty() {
        return MaterializedPlan {
            plan,
            facts,
            effective_preset: preset_path.map(Path::to_path_buf),
            note: None,
            _temp_guard: None,
        };
    }

    match materialize_merged_preset(preset_path, &insertions) {
        Ok(temp) => MaterializedPlan {
            plan,
            facts,
            effective_preset: Some(temp.path().to_path_buf()),
            note: None,
            _temp_guard: Some(temp),
        },
        Err(err) => MaterializedPlan {
            plan,
            facts,
            effective_preset: preset_path.map(Path::to_path_buf),
            note: Some(format!(
                "materialization failed, preset-only behavior kept: {err}"
            )),
            _temp_guard: None,
        },
    }
}

/// Write the merged preset: the original YAML document with ONLY absent keys
/// inserted. Untouched keys keep their authored values (raw-document edit,
/// no typed roundtrip); an existing key is NEVER overwritten, even if the
/// typed view disagreed (defense against yaml-null pathologies).
fn materialize_merged_preset(
    preset_path: Option<&Path>,
    insertions: &[(&[&str], serde_yaml::Value)],
) -> Result<tempfile::NamedTempFile, String> {
    let mut root: serde_yaml::Value = match preset_path {
        Some(path) => {
            let raw = std::fs::read_to_string(path)
                .map_err(|e| format!("read {}: {e}", path.display()))?;
            serde_yaml::from_str(&raw).map_err(|e| format!("parse {}: {e}", path.display()))?
        }
        None => serde_yaml::Value::Mapping(serde_yaml::Mapping::new()),
    };
    let serde_yaml::Value::Mapping(map) = &mut root else {
        return Err("preset root is not a mapping".to_string());
    };
    for (path, value) in insertions {
        insert_absent(map, path, value.clone())?;
    }

    let body = serde_yaml::to_string(&root).map_err(|e| format!("serialize: {e}"))?;
    let base = preset_path.map_or_else(
        || "(no category preset; resolver-only)".to_string(),
        |p| p.display().to_string(),
    );
    let mut file = tempfile::Builder::new()
        .prefix("ny-plan-")
        .suffix(".yaml")
        .tempfile()
        .map_err(|e| format!("tempfile: {e}"))?;
    file.write_all(format!("# plan-resolver v1 merged preset — base: {base}\n{body}").as_bytes())
        .and_then(|()| file.flush())
        .map_err(|e| format!("write: {e}"))?;
    Ok(file)
}

/// Insert `leaf` at `path`, creating intermediate mappings, but ONLY if the
/// leaf key is absent. Returns whether an insert happened.
fn insert_absent(
    root: &mut serde_yaml::Mapping,
    path: &[&str],
    leaf: serde_yaml::Value,
) -> Result<bool, String> {
    let (head, rest) = path.split_first().ok_or("empty key path")?;
    let key = serde_yaml::Value::String((*head).to_string());
    if rest.is_empty() {
        if root.contains_key(&key) {
            return Ok(false);
        }
        root.insert(key, leaf);
        return Ok(true);
    }
    if !root.contains_key(&key) {
        root.insert(
            key.clone(),
            serde_yaml::Value::Mapping(serde_yaml::Mapping::new()),
        );
    }
    match root.get_mut(&key) {
        Some(serde_yaml::Value::Mapping(child)) => insert_absent(child, rest, leaf),
        Some(_) => Err(format!("preset key '{head}' is not a mapping")),
        None => Err("mapping insert lost the key".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic MEDIUM-like facts: the measured CIFAR100_resnet_medium
    /// anchors (2,536,344 params / max width 128 / 10,156,168 bytes) — the
    /// real ONNX files are gitignored downloads, so tests pin the resolver
    /// on the facts, not the files.
    fn medium_facts() -> ModelFacts {
        ModelFacts {
            param_count: 2_536_344,
            conv_layers: 19,
            max_conv_out_channels: 128,
            file_size_bytes: 10_156_168,
        }
    }

    /// Synthetic LARGE-like facts: the measured CIFAR100_resnet_large
    /// anchors (3,808,152 params / max width 256 / 15,243,961 bytes).
    fn large_facts() -> ModelFacts {
        ModelFacts {
            param_count: 3_808_152,
            conv_layers: 20,
            max_conv_out_channels: 256,
            file_size_bytes: 15_243_961,
        }
    }

    fn cpu_backend() -> BackendReport {
        BackendReport {
            kind: "cpu-only",
            summary: "cpu-only [test fixture]".to_string(),
            wgpu_adapter: None,
            wgpu_probe_skipped: false,
            cuda_engine_candidate: false,
        }
    }

    fn cuda_backend() -> BackendReport {
        BackendReport {
            kind: "cuda",
            summary: "cuda [test fixture]".to_string(),
            wgpu_adapter: None,
            wgpu_probe_skipped: true,
            cuda_engine_candidate: true,
        }
    }

    fn metal_backend() -> BackendReport {
        BackendReport {
            kind: "metal",
            summary: "metal [test fixture]".to_string(),
            wgpu_adapter: Some(ny_gpu::AdapterProbe {
                backend: "Metal".to_string(),
                name: "test-metal-adapter".to_string(),
                device_type: "IntegratedGpu".to_string(),
            }),
            wgpu_probe_skipped: false,
            cuda_engine_candidate: false,
        }
    }

    fn preset_from(yaml: &str) -> PresetConfig {
        serde_yaml::from_str(yaml).expect("test preset parses")
    }

    // -- facts classification ------------------------------------------------

    #[test]
    fn model_facts_classify_the_measured_anchors() {
        assert_eq!(medium_facts().conv_scale(), ConvScale::Medium);
        assert_eq!(large_facts().conv_scale(), ConvScale::Large);
        // TinyImageNet-medium (3.62M / width 128): the param predicate
        // classes it LARGE, agreeing with its tuned preset's 0.40 slice.
        let tiny = ModelFacts {
            param_count: 3_616_144,
            conv_layers: 19,
            max_conv_out_channels: 128,
            file_size_bytes: 14_475_376,
        };
        assert_eq!(tiny.conv_scale(), ConvScale::Large);
        // No convolutions => no conv rule, whatever the param count.
        let dense = ModelFacts {
            param_count: 50_000_000,
            conv_layers: 0,
            max_conv_out_channels: 0,
            file_size_bytes: 1,
        };
        assert_eq!(dense.conv_scale(), ConvScale::SmallOrNone);
    }

    // -- layering: preset key beats resolved rule -----------------------------

    #[test]
    fn preset_keys_own_the_slice_and_cap_for_large_conv_facts() {
        // cifar100 shape: the preset pins the medium-tuned 0.05 while the
        // facts say LARGE. LAYERING: the preset wins (shipped categories
        // with explicit keys are byte-identical); the blocked rule is
        // printed, not silently applied.
        let preset = preset_from(
            "bab:\n  root_alpha_cap_secs: 40\n  phase_budget:\n    disjunctive_pgd_fraction: 0.05\n",
        );
        let plan = resolve_plan(&large_facts(), 100, &cpu_backend(), Some(&preset));
        assert_eq!(plan.disjunctive_pgd_fraction.value, Some(0.05));
        assert_eq!(
            plan.disjunctive_pgd_fraction.source,
            SettingSource::PresetOverride("bab.phase_budget.disjunctive_pgd_fraction".into())
        );
        assert_eq!(plan.root_alpha_cap_secs.value, Some(40.0));
        assert_eq!(
            plan.root_alpha_cap_secs.source,
            SettingSource::PresetOverride("bab.root_alpha_cap_secs".into())
        );
        // #rule-contract: there is no longer a Large-conv rule to block, so the
        // "(blocks measured rule: ...)" note is gone with it. What still matters
        // — and is asserted above — is that the explicit preset keys own both
        // values outright.
        let rendered = plan.render_settings();
        assert!(
            !rendered.contains("blocks measured rule"),
            "no rule should claim to have been blocked once the Large-conv arm is \
             removed:\n{rendered}"
        );
    }

    #[test]
    fn preset_slice_beats_medium_pair_and_breaks_it() {
        // TinyImageNet-yaml shape at medium-band facts: the preset pins the
        // slice; the preset value wins AND the cap must NOT be applied alone
        // (pair inseparability, bd6a9eff).
        let preset = preset_from("bab:\n  phase_budget:\n    disjunctive_pgd_fraction: 0.40\n");
        let plan = resolve_plan(&medium_facts(), 100, &cpu_backend(), Some(&preset));
        assert_eq!(plan.disjunctive_pgd_fraction.value, Some(0.40));
        assert_eq!(
            plan.disjunctive_pgd_fraction.source,
            SettingSource::PresetOverride("bab.phase_budget.disjunctive_pgd_fraction".into())
        );
        assert_eq!(plan.root_alpha_cap_secs.value, None);
        assert_eq!(plan.root_alpha_cap_secs.source, SettingSource::Default);
        assert!(plan.render_settings().contains("pair withheld"));
    }

    #[test]
    fn preset_cap_alone_blocks_the_slice_leg_too() {
        let preset = preset_from("bab:\n  root_alpha_cap_secs: 60\n");
        let plan = resolve_plan(&medium_facts(), 100, &cpu_backend(), Some(&preset));
        assert_eq!(plan.root_alpha_cap_secs.value, Some(60.0));
        assert_eq!(
            plan.root_alpha_cap_secs.source,
            SettingSource::PresetOverride("bab.root_alpha_cap_secs".into())
        );
        // No solo 0.05: the pair fires together or not at all.
        assert_eq!(plan.disjunctive_pgd_fraction.value, None);
        assert_eq!(plan.disjunctive_pgd_fraction.source, SettingSource::Default);
        assert!(plan.render_settings().contains("pair withheld"));
    }

    // -- absent key takes the rule ---------------------------------------------

    #[test]
    fn medium_short_budget_gets_the_pair_together() {
        let plan = resolve_plan(&medium_facts(), 100, &cpu_backend(), None);
        assert_eq!(
            plan.disjunctive_pgd_fraction.value,
            Some(MEDIUM_PAIR_PGD_FRACTION)
        );
        assert_eq!(
            plan.disjunctive_pgd_fraction.source,
            SettingSource::ResolvedModel(RULE_MEDIUM_PAIR)
        );
        assert_eq!(
            plan.root_alpha_cap_secs.value,
            Some(MEDIUM_PAIR_ROOT_ALPHA_CAP_SECS)
        );
        assert_eq!(
            plan.root_alpha_cap_secs.source,
            SettingSource::ResolvedBudget(RULE_MEDIUM_PAIR.to_string())
        );
        // Ledger: internal tier 95s (grace max(5, 100/20)=5); slice 0.05*95
        // (computed through the f32 fraction, so compare at f32 widening
        // tolerance, not f64 exactness).
        assert_eq!(plan.ledger.scored_budget_secs, 100);
        assert_eq!(plan.ledger.nominal_internal_tier_secs, 95);
        assert!((plan.ledger.nominal_attack_slice_secs.unwrap() - 4.75).abs() < 1e-6);
        assert_eq!(plan.ledger.root_alpha_cap_secs, Some(40.0));
    }

    /// Collins' cap must remain absolute across every competition tier. These
    /// scored budgets nominally materialize to 25/285/570/1140 seconds; the
    /// runtime tiny-tier policy stays smaller at the first tier.
    #[test]
    fn collins_cap_is_stable_at_scored_budget_tiers() {
        let preset = preset_from("bab:\n  phase_budget:\n    disjunctive_pgd_max_secs: 15\n");
        for (scored, internal, expected_slice) in [
            (30, 25, 3.75),
            (300, 285, 15.0),
            (600, 570, 15.0),
            (1200, 1140, 15.0),
        ] {
            let plan = resolve_plan(&large_facts(), scored, &cpu_backend(), Some(&preset));
            assert_eq!(
                plan.ledger.nominal_internal_tier_secs, internal,
                "scored={scored}"
            );
            assert!(
                (plan.ledger.nominal_attack_slice_secs.unwrap() - expected_slice).abs() < 1e-6,
                "scored={scored}: expected {expected_slice}, got {}",
                plan.ledger.nominal_attack_slice_secs.unwrap()
            );
        }
    }

    #[test]
    fn collins_cap_respects_the_exact_thirty_second_tiny_tier_boundary() {
        let preset = preset_from(
            "bab:\n  phase_budget:\n    disjunctive_pgd_fraction: 0.50\n    \
             disjunctive_pgd_max_secs: 15\n",
        );
        // At 30 internal seconds the inclusive tiny-tier rule wins (4.5s).
        // One second later that rule releases and Collins' absolute 15s cap
        // wins over the nominal 15.5s fraction slice.
        for (scored, internal, expected_slice) in [(35, 30, 4.5), (36, 31, 15.0)] {
            let plan = resolve_plan(&large_facts(), scored, &cpu_backend(), Some(&preset));
            assert_eq!(plan.ledger.nominal_internal_tier_secs, internal);
            assert!(
                (plan.ledger.nominal_attack_slice_secs.unwrap() - expected_slice).abs() < 1e-6,
                "scored={scored}: {:?}",
                plan.ledger.nominal_attack_slice_secs
            );
        }
    }

    #[test]
    fn runtime_override_decoder_matches_exact_runtime_environment_contract() {
        for rejected in [
            None,
            Some(OsStr::new("")),
            Some(OsStr::new("0")),
            Some(OsStr::new("01")),
            Some(OsStr::new("true")),
        ] {
            let runtime = PlanRuntimeOverrides::from_env_values(rejected, rejected);
            assert!(!runtime.pgd_time_cap_disabled, "value={rejected:?}");
            assert!(!runtime.disjunctive_pgd_skipped, "value={rejected:?}");
        }

        let cap_only = PlanRuntimeOverrides::from_env_values(Some(OsStr::new("1")), None);
        assert!(cap_only.pgd_time_cap_disabled);
        assert!(!cap_only.disjunctive_pgd_skipped);

        let skip_only = PlanRuntimeOverrides::from_env_values(None, Some(OsStr::new("1")));
        assert!(!skip_only.pgd_time_cap_disabled);
        assert!(skip_only.disjunctive_pgd_skipped);
    }

    #[test]
    fn plan_settings_always_expose_the_effective_attack_schedule() {
        let rendered = resolve_plan(&large_facts(), 300, &cpu_backend(), None).render_settings();
        for expected in [
            "disjunctive_pgd_attack_enabled = true  [default]",
            "pgd_time_cap_disabled = false  [default]",
            "disjunctive_pgd_max_secs = none  [default]",
            "disjunctive_pgd_min_secs = none  [default]",
            "disjunctive_pgd_from_phase_start = false  [default]",
        ] {
            assert!(
                rendered.contains(expected),
                "missing {expected:?}: {rendered}"
            );
        }
    }

    #[test]
    fn typed_pgd_time_cap_override_matches_runtime_without_reading_environment() {
        let preset = preset_from("bab:\n  phase_budget:\n    disjunctive_pgd_fraction: 0.50\n");
        let runtime = PlanRuntimeOverrides::from_env_values(Some(OsStr::new("1")), None);
        let plan = resolve_plan_with_fl_rate_and_runtime(
            &large_facts(),
            30,
            &cpu_backend(),
            Some(&preset),
            None,
            runtime,
        );
        assert_eq!(plan.ledger.nominal_internal_tier_secs, 25);
        assert_eq!(plan.ledger.nominal_attack_slice_secs, Some(12.5));
        assert!(plan.pgd_time_cap_disabled.value);
        assert_eq!(
            plan.pgd_time_cap_disabled.source,
            SettingSource::RuntimeOverride(RULE_PGD_TIME_CAP_ENV)
        );
    }

    #[test]
    fn disabled_disjunctive_pgd_has_no_nominal_slice_with_typed_provenance() {
        let preset_skip = preset_from("attack:\n  pgd_order: skip\n");
        let by_preset = resolve_plan(&large_facts(), 300, &cpu_backend(), Some(&preset_skip));
        assert!(!by_preset.disjunctive_pgd_enabled.value);
        assert_eq!(
            by_preset.disjunctive_pgd_enabled.source,
            SettingSource::PresetOverride("attack.pgd_order".into())
        );
        assert_eq!(by_preset.ledger.nominal_attack_slice_secs, None);

        let runtime = PlanRuntimeOverrides::from_env_values(None, Some(OsStr::new("1")));
        let by_env = resolve_plan_with_fl_rate_and_runtime(
            &large_facts(),
            300,
            &cpu_backend(),
            None,
            None,
            runtime,
        );
        assert!(!by_env.disjunctive_pgd_enabled.value);
        assert_eq!(
            by_env.disjunctive_pgd_enabled.source,
            SettingSource::RuntimeOverride(RULE_DISJUNCTIVE_PGD_SKIP_ENV)
        );
        assert_eq!(by_env.ledger.nominal_attack_slice_secs, None);
    }

    #[test]
    fn malformed_pgd_schedule_is_reported_invalid_instead_of_default_on() {
        let invalid = preset_from("attack:\n  pgd_order: middle\n");
        let plan = resolve_plan(&large_facts(), 300, &cpu_backend(), Some(&invalid));
        assert!(!plan.disjunctive_pgd_enabled.value);
        assert_eq!(plan.ledger.nominal_attack_slice_secs, None);
        let SettingSource::InvalidPreset(error) = &plan.disjunctive_pgd_enabled.source else {
            panic!("expected invalid-preset provenance");
        };
        assert!(error.contains("not implemented"), "{error}");
    }

    /// PhaseBudgetConfig deliberately preserves finite legacy values outside
    /// `[0, 1]`; the runtime clamps them locally. The reporter must do the same
    /// rather than advertising negative time or more than the whole tier.
    #[test]
    fn attack_slice_ledger_clamps_finite_legacy_fractions() {
        for (fraction, expected_slice) in [(-2.0f32, 0.0), (2.0, 570.0)] {
            let preset = preset_from(&format!(
                "bab:\n  phase_budget:\n    disjunctive_pgd_fraction: {fraction}\n"
            ));
            let plan = resolve_plan(&large_facts(), 600, &cpu_backend(), Some(&preset));
            assert!(
                (plan.ledger.nominal_attack_slice_secs.unwrap() - expected_slice).abs() < 1e-6,
                "fraction={fraction}: expected {expected_slice}, got {}",
                plan.ledger.nominal_attack_slice_secs.unwrap()
            );
        }
    }

    /// Ordering trip-wire: clamp 2.0 -> 1.0, tiny tier -> 0.15 (3.75s),
    /// ceiling -> 2s, then floor -> 5s. Reordering any two controls changes
    /// this result.
    #[test]
    fn attack_slice_ledger_applies_tiny_cap_then_ceiling_then_floor() {
        let preset = preset_from(
            "bab:\n  phase_budget:\n    disjunctive_pgd_fraction: 2.0\n    \
             disjunctive_pgd_max_secs: 2\n    disjunctive_pgd_min_secs: 5\n",
        );
        let plan = resolve_plan(&large_facts(), 30, &cpu_backend(), Some(&preset));
        assert_eq!(plan.ledger.nominal_internal_tier_secs, 25);
        assert_eq!(plan.ledger.nominal_attack_slice_secs, Some(5.0));
    }

    #[test]
    fn attack_slice_ledger_floor_is_half_tier_clamped_and_never_shrinks() {
        assert_eq!(
            planned_disjunctive_pgd_slice_secs(25, 0.10, None, Some(999), false),
            12.5,
            "an oversized floor keeps half the tier for proof"
        );
        assert_eq!(
            planned_disjunctive_pgd_slice_secs(200, 0.50, None, Some(5), false),
            100.0,
            "a floor below the fraction slice cannot shrink it"
        );
        assert_eq!(
            planned_disjunctive_pgd_slice_secs(25, 0.10, None, Some(0), false),
            2.5,
            "an explicit zero floor is inert"
        );
    }

    /// The PAIR is budget-gated: a medium net at a long budget gets NEITHER
    /// half (the evidence stops at short budgets; the resolver declines to
    /// extrapolate rather than half-applying a pair).
    #[test]
    fn medium_pair_rule_declines_beyond_short_budgets() {
        let plan = resolve_plan(&medium_facts(), 900, &cpu_backend(), None);
        assert_eq!(plan.disjunctive_pgd_fraction.value, None);
        assert_eq!(plan.disjunctive_pgd_fraction.source, SettingSource::Default);
        assert_eq!(plan.root_alpha_cap_secs.value, None);
        let rendered = plan.render_settings();
        assert!(rendered.contains("disjunctive_pgd_fraction = 0.50  [default]"));
        assert!(rendered.contains("root_alpha_cap_secs = none  [default]"));
    }

    // -- the removed large-conv rule stays removed --------------------------------

    /// #rule-contract: Large-conv facts must NOT resolve a slice.
    ///
    /// The removed rule cut `disjunctive_pgd_fraction` 0.50 -> 0.40 for the 34
    /// of 48 categories omitting the key, citing a "24 banked sats" figure that
    /// appears nowhere in the document it named, whose slice-starvation
    /// attribution is retracted for two of its three rows. And
    /// `cifar100_2024.yaml:128` sets the key itself, so the rule could never
    /// fire on the one category its evidence covers — its whole blast radius
    /// was un-evidenced categories, in the completeness-losing direction.
    #[test]
    fn large_conv_facts_resolve_no_slice_without_evidence() {
        for budget in [100u64, 300, 900] {
            let plan = resolve_plan(&large_facts(), budget, &cpu_backend(), None);
            assert_eq!(
                plan.disjunctive_pgd_fraction.value, None,
                "model shape alone must not license a slice change (invariant III)"
            );
            assert_eq!(plan.disjunctive_pgd_fraction.source, SettingSource::Default);
            assert!(!plan.disjunctive_pgd_fraction.is_resolved());
            assert_eq!(plan.root_alpha_cap_secs.value, None);
        }
    }

    #[test]
    fn dense_facts_fire_no_slice_rule() {
        let dense = ModelFacts {
            param_count: 5_000_000,
            conv_layers: 0,
            max_conv_out_channels: 0,
            file_size_bytes: 20_000_000,
        };
        let plan = resolve_plan(&dense, 100, &cpu_backend(), None);
        assert_eq!(plan.disjunctive_pgd_fraction.value, None);
        assert_eq!(plan.root_alpha_cap_secs.value, None);
    }

    // -- margin row, steering, spec slots, backend --------------------------------

    /// #rule2-sign-inversion: a category that never opted in must keep the
    /// ENGINE default, not inherit cifar100's measurement.
    ///
    /// b61b5f10 isolates `margin_row.adaptive_reserve` as independently harmful
    /// on tinyimagenet — it costs both near-wall GT-unsat rows, which need the
    /// reserve HELD. Exactly 1 of 48 presets sets the key, so resolving
    /// `Adaptive` here flipped 47 categories on one category's data.
    #[test]
    fn absent_preset_keeps_the_engine_default_not_the_cifar100_rule() {
        let plan = resolve_plan(&medium_facts(), 100, &cpu_backend(), None);
        assert_eq!(
            plan.margin_row.source,
            SettingSource::Default,
            "a category with no margin_row key must not have a rule resolve one for it"
        );
        assert!(
            !plan.margin_row.is_resolved(),
            "Default must not be materialized into the effective preset"
        );
        assert_ne!(
            plan.margin_row.value,
            MarginRowPolicy::Adaptive,
            "adaptive release is cifar100-measured and contradicted on tinyimagenet"
        );
    }

    /// The materialization half of the same invariant: an un-opted-in category
    /// must come out of the resolver byte-identical to pre-resolver behaviour.
    #[test]
    fn absent_preset_materializes_no_margin_row_key() {
        let plan = resolve_plan(&medium_facts(), 100, &cpu_backend(), None);
        assert!(
            !(plan.margin_row.is_resolved() && plan.margin_row.value == MarginRowPolicy::Adaptive),
            "this is the exact condition guarding the `margin_row.adaptive_reserve = true` \
             insertion; if it holds for an absent preset, 47 categories get flipped"
        );
    }

    #[test]
    fn margin_row_sources_are_tagged_right() {
        // Absent -> ENGINE default stands; the resolver does not own it.
        let plan = resolve_plan(&medium_facts(), 100, &cpu_backend(), None);
        assert_eq!(plan.margin_row.value, MarginRowPolicy::Fixed(45));
        assert_eq!(plan.margin_row.source, SettingSource::Default);
        // Explicit true -> preset override, adaptive.
        let preset = preset_from("margin_row:\n  adaptive_reserve: true\n");
        let plan = resolve_plan(&medium_facts(), 100, &cpu_backend(), Some(&preset));
        assert_eq!(plan.margin_row.value, MarginRowPolicy::Adaptive);
        assert_eq!(
            plan.margin_row.source,
            SettingSource::PresetOverride("margin_row.adaptive_reserve".into())
        );
        // Explicit false + secs -> the preset keeps its fixed reserve
        // (layering: explicit keys win), and the declined rule is printed.
        let preset = preset_from("margin_row:\n  adaptive_reserve: false\n  reserve_secs: 82\n");
        let plan = resolve_plan(&medium_facts(), 100, &cpu_backend(), Some(&preset));
        assert_eq!(plan.margin_row.value, MarginRowPolicy::Fixed(82));
        assert!(plan
            .render_settings()
            .contains("preset declines measured rule"));
        // Explicit zero reserve -> none.
        let preset = preset_from("margin_row:\n  reserve_secs: 0\n");
        let plan = resolve_plan(&medium_facts(), 100, &cpu_backend(), Some(&preset));
        assert_eq!(plan.margin_row.value, MarginRowPolicy::NoReserve);
        assert_eq!(
            plan.margin_row.source,
            SettingSource::PresetOverride("margin_row.reserve_secs".into())
        );
    }

    #[test]
    fn steering_and_backend_are_recorded_per_host() {
        let plan = resolve_plan(&medium_facts(), 100, &cuda_backend(), None);
        assert!(plan.attack_steering_armed.value);
        assert_eq!(
            plan.attack_steering_armed.source,
            SettingSource::HostFact(RULE_ATTACK_STEERING)
        );
        assert_eq!(plan.backend_kind, "cuda");
        let plan = resolve_plan(&medium_facts(), 100, &cpu_backend(), None);
        assert!(!plan.attack_steering_armed.value);
        assert_eq!(plan.backend_kind, "cpu-only");
        assert!(plan
            .render_settings()
            .contains("attack_steering = disarmed (cpu-only route)"));
    }

    #[test]
    fn charged_metal_gate_state_is_recorded_on_wgpu_adapter_hosts_only() {
        // #flush-charge: a RECORDED host/build fact in the plan surface —
        // present exactly where a WGPU adapter regime exists, narrating the
        // source-gate state; cuda/cpu-only plan output stays byte-identical.
        let plan = resolve_plan(&medium_facts(), 100, &metal_backend(), None);
        let line = plan
            .iter()
            .find(|(name, _, _)| *name == "wgpu_charged_authority")
            .expect("metal plans record the charged-authority gate state");
        if ny_gpu::wgpu_charged_proof_authority() {
            assert!(
                line.1.starts_with("armed"),
                "open gate must read armed: {}",
                line.1
            );
        } else {
            assert!(
                line.1.starts_with("dark"),
                "closed gate must read dark: {}",
                line.1
            );
        }
        assert!(
            line.2.starts_with("host fact"),
            "the charged gate is recorded, never resolved: {}",
            line.2
        );

        for backend in [cpu_backend(), cuda_backend()] {
            let plan = resolve_plan(&medium_facts(), 100, &backend, None);
            assert!(
                plan.iter()
                    .all(|(name, _, _)| name != "wgpu_charged_authority"),
                "{} plans must not grow a charged-authority line",
                backend.kind
            );
        }
    }

    #[test]
    fn alpha_spec_slots_pass_through_records_source() {
        let preset = preset_from("solver:\n  alpha_crown:\n    spec_slots: 4\n");
        let plan = resolve_plan(&medium_facts(), 100, &cpu_backend(), Some(&preset));
        assert_eq!(plan.alpha_spec_slots.value, Some(4));
        assert_eq!(
            plan.alpha_spec_slots.source,
            SettingSource::PresetOverride("solver.alpha_crown.spec_slots".into())
        );
        assert!(plan
            .render_settings()
            .contains("#spec-axis-alpha acceptance still open"));
        let plan = resolve_plan(&medium_facts(), 100, &cpu_backend(), None);
        assert_eq!(plan.alpha_spec_slots.value, None);
        assert_eq!(plan.alpha_spec_slots.source, SettingSource::Default);
    }

    /// The iter() contract the printer and JSON writer share: every setting
    /// yields a (name, value, source) triple, in print order.
    #[test]
    fn iter_yields_name_value_source_triples() {
        let plan = resolve_plan(&medium_facts(), 100, &cpu_backend(), None);
        let triples: Vec<_> = plan.iter().collect();
        assert_eq!(triples.len(), plan.settings.len());
        assert_eq!(triples[0].0, "backend");
        assert_eq!(triples[0].1, "cpu-only");
        // #rule-contract: the backend is a HOST FACT, not a resolved rule.
        assert!(triples[0].2.starts_with("host fact"));
    }

    // -- raw protobuf scan (hermetic: hand-encoded ONNX bytes) --------------------

    fn varint_enc(mut v: u64, out: &mut Vec<u8>) {
        loop {
            let byte = (v & 0x7f) as u8;
            v >>= 7;
            if v == 0 {
                out.push(byte);
                return;
            }
            out.push(byte | 0x80);
        }
    }

    fn field_varint(field: u32, v: u64, out: &mut Vec<u8>) {
        varint_enc(u64::from(field) << 3, out);
        varint_enc(v, out);
    }

    fn field_bytes(field: u32, payload: &[u8], out: &mut Vec<u8>) {
        varint_enc((u64::from(field) << 3) | 2, out);
        varint_enc(payload.len() as u64, out);
        out.extend_from_slice(payload);
    }

    fn enc_tensor(name: &str, dims: &[u64]) -> Vec<u8> {
        let mut t = Vec::new();
        for &d in dims {
            field_varint(1, d, &mut t);
        }
        field_bytes(8, name.as_bytes(), &mut t);
        t
    }

    fn enc_node(op: &str, inputs: &[&str]) -> Vec<u8> {
        let mut n = Vec::new();
        for input in inputs {
            field_bytes(1, input.as_bytes(), &mut n);
        }
        field_bytes(4, op.as_bytes(), &mut n);
        n
    }

    fn enc_model(initializers: &[Vec<u8>], nodes: &[Vec<u8>]) -> Vec<u8> {
        let mut graph = Vec::new();
        for n in nodes {
            field_bytes(1, n, &mut graph);
        }
        for t in initializers {
            field_bytes(5, t, &mut graph);
        }
        let mut m = Vec::new();
        field_bytes(7, &graph, &mut m);
        m
    }

    #[test]
    fn scan_reads_params_convs_and_widths() {
        let bytes = enc_model(
            &[
                enc_tensor("w0", &[64, 3, 3, 3]),
                enc_tensor("w1", &[256, 64, 3, 3]),
                enc_tensor("fc", &[100, 4096]),
            ],
            &[
                enc_node("Conv", &["x", "w0"]),
                enc_node("Conv", &["h", "w1"]),
                enc_node("Relu", &["h"]),
                enc_node("Gemm", &["h", "fc"]),
            ],
        );
        let scan = scan_onnx_graph(&bytes).expect("scan");
        assert_eq!(
            scan.param_count,
            64 * 3 * 3 * 3 + 256 * 64 * 3 * 3 + 100 * 4096
        );
        assert_eq!(scan.conv_layers, 2);
        assert_eq!(scan.max_conv_out_channels, 256);
    }

    #[test]
    fn scan_handles_packed_dims() {
        // Same tensor with dims packed into one length-delimited field 1
        // (proto3 packs repeated scalars by default; some exporters do not).
        let mut packed = Vec::new();
        for &d in &[256u64, 64, 3, 3] {
            varint_enc(d, &mut packed);
        }
        let mut t = Vec::new();
        field_bytes(1, &packed, &mut t);
        field_bytes(8, b"w", &mut t);
        let bytes = enc_model(&[t], &[enc_node("Conv", &["x", "w"])]);
        let scan = scan_onnx_graph(&bytes).expect("scan");
        assert_eq!(scan.param_count, 256 * 64 * 3 * 3);
        assert_eq!(scan.max_conv_out_channels, 256);
    }

    fn write_temp(suffix: &str, bytes: &[u8]) -> tempfile::NamedTempFile {
        let mut f = tempfile::Builder::new()
            .suffix(suffix)
            .tempfile()
            .expect("temp file");
        f.write_all(bytes).expect("write");
        f.flush().expect("flush");
        f
    }

    /// Medium-band conv onnx (facts land in ConvScale::Medium) as real bytes.
    fn medium_conv_onnx() -> tempfile::NamedTempFile {
        let mut initializers = vec![enc_tensor("w_big", &[128, 64, 3, 3])];
        let mut nodes = vec![enc_node("Conv", &["x", "w_big"])];
        for k in 0..8u64 {
            let name = format!("w{k}");
            initializers.push(enc_tensor(&name, &[64, 64, 3, 3]));
            nodes.push(enc_node("Conv", &["x", &name]));
        }
        initializers.push(enc_tensor("fc", &[2_000_000]));
        nodes.push(enc_node("Gemm", &["y", "fc"]));
        write_temp(".onnx", &enc_model(&initializers, &nodes))
    }

    #[test]
    fn facts_from_onnx_file_match_the_scan_and_fail_closed() {
        let onnx = medium_conv_onnx();
        let facts = ModelFacts::from_onnx_file(onnx.path()).expect("facts");
        assert_eq!(facts.conv_layers, 9);
        assert_eq!(facts.max_conv_out_channels, 128);
        assert_eq!(facts.conv_scale(), ConvScale::Medium);
        // Missing file and non-.onnx extension both decline.
        assert!(ModelFacts::from_onnx_file(Path::new("/nonexistent/x.onnx")).is_none());
        let gz = write_temp(".onnx.gz", b"not a model");
        assert!(ModelFacts::from_onnx_file(gz.path()).is_none());
        // Garbage bytes decline rather than inventing facts.
        let junk = write_temp(".onnx", &[0xff, 0xff, 0xff, 0xff]);
        assert!(ModelFacts::from_onnx_file(junk.path()).is_none());
    }

    // -- materialization -----------------------------------------------------------

    #[test]
    fn materialize_merges_only_absent_keys_and_preserves_the_rest() {
        let onnx = medium_conv_onnx();
        let preset = write_temp(
            ".yaml",
            b"bab:\n  batch_size: 256\n  phase_budget:\n    attack_extension_fraction: 0.0\n",
        );
        let out = resolve_and_materialize(
            onnx.path(),
            Some(preset.path()),
            100,
            &cuda_backend(),
            || None,
        );
        assert!(out.note.is_none(), "clean path expected: {:?}", out.note);
        let effective = out.effective_preset.as_deref().expect("effective preset");
        assert_ne!(
            effective,
            preset.path(),
            "pair rule must materialize a merge"
        );
        let merged = crate::preset::load_preset(effective).expect("merged preset parses");
        // Inserted by the pair rule + the margin-row default:
        assert_eq!(
            merged.bab.phase_budget.disjunctive_pgd_fraction,
            Some(MEDIUM_PAIR_PGD_FRACTION)
        );
        assert_eq!(
            merged.bab.root_alpha_cap_secs,
            Some(MEDIUM_PAIR_ROOT_ALPHA_CAP_SECS)
        );
        // #rule2-sign-inversion: an un-opted-in category must NOT have
        // `margin_row.adaptive_reserve` synthesised for it — that insertion
        // flipped 47 of 48 categories on cifar100-only evidence that
        // b61b5f10 contradicts for tinyimagenet.
        assert_eq!(merged.margin_row.adaptive_reserve, None);
        // Authored keys preserved:
        assert_eq!(merged.bab.batch_size, Some(256));
        assert_eq!(merged.bab.phase_budget.attack_extension_fraction, Some(0.0));
    }

    #[test]
    fn materialize_never_overwrites_an_explicit_key() {
        let onnx = medium_conv_onnx();
        // Preset owns BOTH pair keys and the margin-row policy (cifar100
        // shape): nothing to apply, the ORIGINAL path is used untouched.
        let preset = write_temp(
            ".yaml",
            b"margin_row:\n  adaptive_reserve: true\nbab:\n  root_alpha_cap_secs: 40\n  \
              phase_budget:\n    disjunctive_pgd_fraction: 0.05\n",
        );
        let out = resolve_and_materialize(
            onnx.path(),
            Some(preset.path()),
            100,
            &cuda_backend(),
            || None,
        );
        assert_eq!(
            out.effective_preset.as_deref(),
            Some(preset.path()),
            "explicit keys everywhere: no merge, byte-identical shipped behavior"
        );
        assert!(out._temp_guard.is_none());
    }

    // -- shipped-category layering proof (real yamls, applied config) ---------

    /// The β-CROWN config a run applies from a preset path, serialized —
    /// "byte-identical" is literal: same defaults, same `apply_preset`, same
    /// bytes. This is the exact `BetaCrownConfig::default()` + `apply_preset`
    /// sequence `invoke_beta_crown` performs before CLI-flag overrides.
    fn applied_beta_crown_config(preset_path: &Path) -> String {
        let preset = crate::preset::load_preset(preset_path).expect("preset loads");
        let mut config = ny_propagate::BetaCrownConfig::default();
        crate::preset::apply_preset(&mut config, &preset).expect("preset applies");
        serde_json::to_string(&config).expect("config serializes")
    }

    fn shipped_preset(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../configs/vnncomp25")
            .join(name)
    }

    /// Large-band conv ONNX bytes (width 256 >= the 192 LARGE threshold).
    fn large_conv_onnx() -> tempfile::NamedTempFile {
        let bytes = enc_model(
            &[enc_tensor("w", &[256, 128, 3, 3])],
            &[enc_node("Conv", &["x", "w"])],
        );
        write_temp(".onnx", &bytes)
    }

    /// Dense (no-conv) ONNX bytes: the nn4sys model class as the resolver
    /// sees it — `conv_layers == 0`, so no slice rule can fire.
    fn dense_onnx() -> tempfile::NamedTempFile {
        let bytes = enc_model(
            &[enc_tensor("fc", &[512, 512])],
            &[enc_node("Gemm", &["x", "fc"])],
        );
        write_temp(".onnx", &bytes)
    }

    /// LAYERING PROOF on the shipped cifar100_2024.yaml: it owns BOTH pair
    /// keys and the margin-row policy, so every model class it serves must
    /// reuse the original preset without paying the FL probe or arming rule 7.
    /// This is deliberate: delegating the pair regressed the banked large
    /// ResNet rows even though the medium resolver can reproduce the values.
    #[test]
    fn shipped_cifar100_2024_resolves_byte_identical() {
        let yaml = shipped_preset("cifar100_2024.yaml");
        let before = applied_beta_crown_config(&yaml);
        for model in [medium_conv_onnx(), large_conv_onnx()] {
            let out =
                resolve_and_materialize(model.path(), Some(&yaml), 100, &cuda_backend(), || {
                    panic!("preset owns the pair keys: the FL probe must never be paid here")
                });
            assert_eq!(
                out.effective_preset.as_deref(),
                Some(yaml.as_path()),
                "cifar100_2024 owns every resolver-eligible key: no merge"
            );
            assert!(out._temp_guard.is_none());
            assert_eq!(
                applied_beta_crown_config(out.effective_preset.as_deref().unwrap()),
                before,
                "applied BetaCrownConfig must remain byte-identical"
            );
            assert_eq!(
                out.plan.disjunctive_pgd_fraction.value,
                Some(MEDIUM_PAIR_PGD_FRACTION)
            );
            assert_eq!(
                out.plan.disjunctive_pgd_fraction.source,
                SettingSource::PresetOverride("bab.phase_budget.disjunctive_pgd_fraction".into())
            );
            assert_eq!(
                out.plan.root_alpha_cap_secs.value,
                Some(MEDIUM_PAIR_ROOT_ALPHA_CAP_SECS)
            );
            assert_eq!(
                out.plan.root_alpha_cap_secs.source,
                SettingSource::PresetOverride("bab.root_alpha_cap_secs".into())
            );
            assert_eq!(out.plan.fl_phase_budget, None);
            assert_eq!(out.plan.forward_alpha_surrogate.value, None);
            assert_eq!(
                out.plan.forward_alpha_surrogate.source,
                SettingSource::Default
            );
        }
    }

    /// LAYERING PROOF on the shipped nn4sys.yaml: its model class fires no
    /// slice rule, so the ONLY insertion is rule 2's margin-row adaptive
    /// default (inert until the opt-in twin-wall lane admits an instance).
    /// The merge must preserve every explicit nn4sys key: the applied
    /// `BetaCrownConfig` from the merged preset is byte-identical to the one
    /// from the original yaml.
    #[test]
    fn shipped_nn4sys_resolves_byte_identical_through_the_merge() {
        let yaml = shipped_preset("nn4sys.yaml");
        let before = applied_beta_crown_config(&yaml);
        let onnx = dense_onnx();
        let out = resolve_and_materialize(onnx.path(), Some(&yaml), 116, &cuda_backend(), || None);
        let effective = out.effective_preset.as_deref().expect("effective preset");
        // #rule2-sign-inversion: nn4sys sets no margin_row key, so after the fix
        // NOTHING is materialised for it and the shipped preset is used verbatim
        // — which is exactly what this test's name asks for. It previously
        // asserted the opposite, encoding the defect as the expectation.
        assert_eq!(
            effective,
            yaml.as_path(),
            "an un-opted-in category must resolve to its shipped preset with no merge"
        );
        assert_eq!(
            applied_beta_crown_config(effective),
            before,
            "every explicit nn4sys key must survive the merge byte-identically"
        );
        let merged = crate::preset::load_preset(effective).expect("merged parses");
        // #rule2-sign-inversion: an un-opted-in category must NOT have
        // `margin_row.adaptive_reserve` synthesised for it — that insertion
        // flipped 47 of 48 categories on cifar100-only evidence that
        // b61b5f10 contradicts for tinyimagenet.
        assert_eq!(merged.margin_row.adaptive_reserve, None);
        // No slice rule fired: β-CROWN-visible keys were not inserted.
        assert_eq!(merged.bab.phase_budget.disjunctive_pgd_fraction, None);
        assert_eq!(merged.bab.root_alpha_cap_secs, None);
    }

    #[test]
    fn materialize_without_a_preset_synthesizes_resolved_keys_only() {
        let onnx = medium_conv_onnx();
        let out = resolve_and_materialize(onnx.path(), None, 100, &cuda_backend(), || None);
        let effective = out.effective_preset.as_deref().expect("synthesized preset");
        let merged = crate::preset::load_preset(effective).expect("parses");
        assert_eq!(
            merged.bab.phase_budget.disjunctive_pgd_fraction,
            Some(MEDIUM_PAIR_PGD_FRACTION)
        );
        assert_eq!(
            merged.bab.root_alpha_cap_secs,
            Some(MEDIUM_PAIR_ROOT_ALPHA_CAP_SECS)
        );
        // #rule2-sign-inversion: an un-opted-in category must NOT have
        // `margin_row.adaptive_reserve` synthesised for it — that insertion
        // flipped 47 of 48 categories on cifar100-only evidence that
        // b61b5f10 contradicts for tinyimagenet.
        assert_eq!(merged.margin_row.adaptive_reserve, None);
        // Nothing else appears: resolved keys only.
        assert_eq!(merged.bab.batch_size, None);
        assert!(merged.general.device.is_none());
    }

    #[test]
    fn unreadable_preset_declines_resolution_and_keeps_the_path() {
        let onnx = medium_conv_onnx();
        let preset = write_temp(".yaml", b"bab: [this is not a mapping\n");
        let out = resolve_and_materialize(
            onnx.path(),
            Some(preset.path()),
            100,
            &cuda_backend(),
            || None,
        );
        assert_eq!(out.effective_preset.as_deref(), Some(preset.path()));
        assert!(out
            .note
            .as_deref()
            .is_some_and(|n| n.contains("resolver declined")));
        // Declined resolution runs preset-less against the facts, but nothing
        // is materialized: the broken preset error stays downstream's to
        // raise.
        assert!(out._temp_guard.is_none());
    }

    /// The merged preset is readable while the plan owns its guard and is
    /// deleted when that plan drops. The copied path below intentionally proves
    /// deletion; it also documents why callers that copy the name must retain
    /// the plan themselves.
    #[test]
    fn merged_preset_is_readable_while_plan_lives_and_deleted_on_drop() {
        let onnx = medium_conv_onnx();
        let path = {
            let out = resolve_and_materialize(onnx.path(), None, 100, &cuda_backend(), || None);
            assert!(
                out._temp_guard.is_some(),
                "fixture must materialize a merged temp preset for this to prove anything"
            );
            let borrowed = out.effective_preset().expect("materialized preset path");
            assert!(
                borrowed.is_file(),
                "readable for as long as the plan is alive: {}",
                borrowed.display()
            );
            borrowed.to_path_buf()
        };
        // The plan — guard included — died at the brace above.
        assert!(
            !path.exists(),
            "the guard deletes the merged preset at the plan's scope exit: {}",
            path.display()
        );
    }

    /// #rule-contract, back-door closer. `is_resolved()` claims to be "exactly
    /// the values materialization may apply". Assert that literally, over the
    /// rendered plan, so a future inline `SettingSource::ResolvedBackend(..)`
    /// like the one `flight_summary` used to carry cannot reintroduce a
    /// resolved-but-unmaterializable source.
    #[test]
    fn every_resolved_source_is_materializable() {
        for backend in [cpu_backend(), cuda_backend()] {
            for onnx in [medium_conv_onnx(), large_conv_onnx(), dense_onnx()] {
                let out = resolve_and_materialize(onnx.path(), None, 100, &backend, || None);
                let line = out.flight_summary();
                for host_only in ["backend=", "attack_steering="] {
                    let seg = line
                        .split(host_only)
                        .nth(1)
                        .and_then(|t| t.split(';').next())
                        .unwrap_or("");
                    assert!(
                        !seg.contains("resolved("),
                        "{host_only} is a host fact and must not render as resolved(): {seg}"
                    );
                }
            }
        }
    }

    #[test]
    fn flight_summary_names_every_value_and_source() {
        let onnx = medium_conv_onnx();
        let out = resolve_and_materialize(onnx.path(), None, 100, &cuda_backend(), || None);
        let line = out.flight_summary();
        assert!(line.contains("disjunctive_pgd_attack_enabled=true:[default]"));
        assert!(line.contains("pgd_time_cap_disabled=false:[default]"));
        assert!(line.contains("disjunctive_pgd_fraction=0.05:[resolved(model-facts)"));
        assert!(line.contains("disjunctive_pgd_max_secs=none:[default]"));
        assert!(line.contains("disjunctive_pgd_min_secs=none:[default]"));
        assert!(line.contains("disjunctive_pgd_from_phase_start=false:[default]"));
        assert!(line.contains("nominal_internal_tier_secs=95"));
        assert!(line.contains("nominal_attack_slice_secs=4.750000"));
        assert!(line.contains("root_alpha_cap_secs=40:[resolved(budget)"));
        // Engine default, not a resolver-owned value (#rule2-sign-inversion).
        assert!(line.contains("margin_row=fixed(45s):[default]"));
        // Host facts, recorded for provenance; never materialized, never
        // evidence-scoped to a category (#rule-contract).
        assert!(line.contains("attack_steering=armed-async:[host fact"));
        assert!(line.contains("alpha_spec_slots=unset:[default]"));
        assert!(line.contains("backend=cuda:[host fact"));
        assert!(line.contains("facts{params="));
        assert!(line.contains("effective_preset="));
        // No measured rate injected => no FL-specific segment.
        assert!(!line.contains("fl_phase_budget="));
    }

    #[test]
    fn flight_summary_records_attack_ceiling_and_floor_values_and_sources() {
        let onnx = medium_conv_onnx();
        let preset = write_temp(
            ".yaml",
            b"attack:\n  pgd_order: before\nbab:\n  phase_budget:\n    disjunctive_pgd_fraction: 0.5\n    disjunctive_pgd_max_secs: 30\n    disjunctive_pgd_min_secs: 5\n    disjunctive_pgd_from_phase_start: true\n",
        );
        let runtime = PlanRuntimeOverrides::from_env_values(Some(OsStr::new("1")), None);
        let out = resolve_and_materialize_with_runtime(
            onnx.path(),
            Some(preset.path()),
            300,
            &cuda_backend(),
            runtime,
            || None,
        );
        let line = out.flight_summary();
        assert!(
            line.contains(
                "disjunctive_pgd_attack_enabled=true:[preset override: attack.pgd_order]"
            ),
            "{line}"
        );
        assert!(
            line.contains("pgd_time_cap_disabled=true:[runtime override: NY_NO_PGD_TIME_CAP=1"),
            "{line}"
        );
        assert!(
            line.contains(
                "disjunctive_pgd_max_secs=30:[preset override: \
                 bab.phase_budget.disjunctive_pgd_max_secs]"
            ),
            "{line}"
        );
        assert!(
            line.contains(
                "disjunctive_pgd_min_secs=5:[preset override: \
                 bab.phase_budget.disjunctive_pgd_min_secs]"
            ),
            "{line}"
        );
        assert!(
            line.contains(
                "disjunctive_pgd_from_phase_start=true:[preset override: \
                 bab.phase_budget.disjunctive_pgd_from_phase_start]"
            ),
            "{line}"
        );
        assert!(line.contains("nominal_internal_tier_secs=285"), "{line}");
        assert!(
            line.contains("nominal_attack_slice_secs=30.000000"),
            "{line}"
        );
    }

    #[test]
    fn flight_summary_cannot_claim_a_slice_when_runtime_skips_pgd() {
        let onnx = medium_conv_onnx();
        let runtime = PlanRuntimeOverrides::from_env_values(None, Some(OsStr::new("1")));
        let out = resolve_and_materialize_with_runtime(
            onnx.path(),
            None,
            100,
            &cuda_backend(),
            runtime,
            || None,
        );
        let line = out.flight_summary();
        assert!(
            line.contains(
                "disjunctive_pgd_attack_enabled=false:[runtime override: NY_SKIP_DISJ_PGD=1"
            ),
            "{line}"
        );
        assert!(
            line.contains("nominal_attack_slice_secs=disabled"),
            "{line}"
        );
        assert!(
            line.contains(
                "skipped_pgd_forward_linear_warmer=conditional-conv-route-synchronous-overall-deadline"
            ),
            "{line}"
        );
        assert!(
            line.contains("nominal_attack_slice_excludes_warmer=true"),
            "{line}"
        );

        // Preset skip follows a different runtime branch: it disables
        // `pgd_attack` itself, so the env-skip-only synchronous warmer does
        // not run and must not be advertised.
        let preset = write_temp(".yaml", b"attack:\n  pgd_order: skip\n");
        let preset_out = resolve_and_materialize(
            onnx.path(),
            Some(preset.path()),
            100,
            &cuda_backend(),
            || None,
        );
        let preset_line = preset_out.flight_summary();
        assert!(
            preset_line.contains(
                "disjunctive_pgd_attack_enabled=false:[preset override: attack.pgd_order]"
            ),
            "{preset_line}"
        );
        assert!(
            !preset_line.contains("skipped_pgd_forward_linear_warmer="),
            "{preset_line}"
        );
    }

    // -- rule 6: FL-aware phase budgeting (#fl-phase-budget, I10) -----------------

    /// Fixed injected rates; nothing in these tests depends on the build
    /// host's throughput. 11.57 GMAC/s is the MEASURED loaded-host rate from
    /// the official-100s refusal flight (off100_prop_idx_7500_sidx_40);
    /// 23.0 GMAC/s is the quiet-host end of the measured span (~24s build).
    fn fl_probe(gmacs: f64) -> FlRateObservation {
        FlRateObservation {
            macs_per_sec: (gmacs * 1e9) as u64,
            source: "probe",
            probe_secs: 0.084,
        }
    }

    /// Fast-rate host at the official 100s tier: pred = 559.4/23 x 1.25 =
    /// 30.4s; 30.4 + 15 + 40 = 85.4 <= 95 => window = ceil(30.4 + 15) = 46s,
    /// within the tier - BaB-floor bound (55s). The pair's slice leg is
    /// untouched; only the cap leg is re-sized, upward.
    #[test]
    fn fl_fast_rate_widens_the_root_window_at_100s() {
        let rate = fl_probe(23.0);
        let plan =
            resolve_plan_with_fl_rate(&medium_facts(), 100, &cpu_backend(), None, Some(&rate));
        assert_eq!(plan.root_alpha_cap_secs.value, Some(46.0));
        let SettingSource::ResolvedBudget(reason) = &plan.root_alpha_cap_secs.source else {
            panic!(
                "widened cap must be budget-resolved: {:?}",
                plan.root_alpha_cap_secs.source
            );
        };
        // Provenance carries THIS RUN's numbers, not just the rule name.
        assert!(reason.contains("fl-phase-budget"), "{reason}");
        assert!(reason.contains("23.00 GMAC/s"), "{reason}");
        assert!(reason.contains("30.4s cold FL build"), "{reason}");
        assert!(reason.contains("95s tier"), "{reason}");
        assert!(reason.contains("root window 46s"), "{reason}");
        // Window respects the BaB floor bound: 46 <= 95 - 40.
        assert!(plan.root_alpha_cap_secs.value.unwrap() <= 95.0 - FL_BAB_FLOOR_SECS);
        // Slice leg of the pair still fires, unchanged.
        assert_eq!(
            plan.disjunctive_pgd_fraction.value,
            Some(MEDIUM_PAIR_PGD_FRACTION)
        );
        // Ledger mirrors the widened window.
        assert_eq!(plan.ledger.root_alpha_cap_secs, Some(46.0));
        assert_eq!(
            plan.fl_phase_budget.as_deref(),
            Some("widened root window to 46s")
        );
    }

    /// Slow-rate host (the measured loaded-host 11.57 GMAC/s) at 100s: pred =
    /// 60.4s; 60.4 + 15 + 40 = 115.4 > 95 => REFUSE. The banked cap-40 recipe
    /// stands byte-identically and the decline is visible in plan + flight.
    #[test]
    fn fl_slow_rate_refuses_and_keeps_the_banked_cap() {
        let rate = fl_probe(11.57);
        let plan =
            resolve_plan_with_fl_rate(&medium_facts(), 100, &cpu_backend(), None, Some(&rate));
        assert_eq!(plan.root_alpha_cap_secs.value, Some(40.0));
        assert_eq!(
            plan.root_alpha_cap_secs.source,
            SettingSource::ResolvedBudget(RULE_MEDIUM_PAIR.to_string()),
            "declined widening must leave the pair's cap leg untouched"
        );
        let note = plan
            .fl_phase_budget
            .as_deref()
            .expect("decline is recorded");
        assert!(note.contains("declined"), "{note}");
        assert!(note.contains("60.4s"), "{note}");
        assert!(note.contains("95s tier"), "{note}");
        assert!(plan.render_settings().contains("(fl-phase-budget declined"));
    }

    /// A 900s budget (tier 855s) widens generously: the SAME 11.57 GMAC/s
    /// rate that was refused at 100s now fits — window = ceil(60.4 + 15) =
    /// 76s, well past the 100s tier's 55s bound, with 855 - 76 >> 40s of BaB
    /// preserved. The window is sized to the PREDICTION, not to the budget.
    #[test]
    fn fl_widens_generously_at_900s() {
        let rate = fl_probe(11.57);
        let plan =
            resolve_plan_with_fl_rate(&medium_facts(), 900, &cpu_backend(), None, Some(&rate));
        assert_eq!(plan.root_alpha_cap_secs.value, Some(76.0));
        assert!(plan.root_alpha_cap_secs.is_resolved());
        const { assert!(76.0 > 95.0 - FL_BAB_FLOOR_SECS, "past the 100s-tier bound") };
        let SettingSource::ResolvedBudget(reason) = &plan.root_alpha_cap_secs.source else {
            panic!("widened cap must be budget-resolved");
        };
        assert!(reason.contains("855s tier"), "{reason}");
        assert!(reason.contains("root window 76s"), "{reason}");
    }

    /// Presets-win invariant: an explicit `root_alpha_cap_secs` pins the
    /// window regardless of any injected rate (defense in depth — the scoped
    /// callers never even probe for preset-owned categories, and the pure
    /// resolver re-checks scope besides).
    #[test]
    fn fl_preset_override_pins_regardless() {
        let preset = preset_from(
            "bab:\n  root_alpha_cap_secs: 40\n  phase_budget:\n    disjunctive_pgd_fraction: 0.05\n",
        );
        let rate = fl_probe(23.0);
        let plan = resolve_plan_with_fl_rate(
            &medium_facts(),
            100,
            &cpu_backend(),
            Some(&preset),
            Some(&rate),
        );
        assert_eq!(plan.root_alpha_cap_secs.value, Some(40.0));
        assert_eq!(
            plan.root_alpha_cap_secs.source,
            SettingSource::PresetOverride("bab.root_alpha_cap_secs".into())
        );
        assert_eq!(plan.fl_phase_budget, None, "scope refused: no fl decision");
    }

    /// The unmeasured fallback constant may never widen (the stale-rate
    /// lockout is what this rule ends, not what it acts on), and a rate so
    /// fast FL fits inside 40s applies nothing — widening only ever RAISES.
    #[test]
    fn fl_fallback_and_already_fitting_rates_change_nothing() {
        let fallback = FlRateObservation {
            macs_per_sec: 5_500_000_000,
            source: "fallback",
            probe_secs: 0.0,
        };
        let plan =
            resolve_plan_with_fl_rate(&medium_facts(), 900, &cpu_backend(), None, Some(&fallback));
        assert_eq!(
            plan.root_alpha_cap_secs.value, None,
            "900s: pair declines, no cap"
        );
        assert!(plan
            .fl_phase_budget
            .as_deref()
            .is_some_and(|n| n.contains("not a measurement")));
        // 100 GMAC/s: pred = 7.0s, window would clamp to 40 => decline, pair cap stands.
        let fast = fl_probe(100.0);
        let plan =
            resolve_plan_with_fl_rate(&medium_facts(), 100, &cpu_backend(), None, Some(&fast));
        assert_eq!(plan.root_alpha_cap_secs.value, Some(40.0));
        assert_eq!(
            plan.root_alpha_cap_secs.source,
            SettingSource::ResolvedBudget(RULE_MEDIUM_PAIR.to_string())
        );
        assert!(plan
            .fl_phase_budget
            .as_deref()
            .is_some_and(|n| n.contains("no widening needed")));
    }

    /// Scope is part of the rule (#rule-contract): LARGE and dense facts must
    /// not produce an fl decision even when a caller injects a rate.
    #[test]
    fn fl_scope_declines_outside_the_medium_band() {
        let rate = fl_probe(23.0);
        let plan =
            resolve_plan_with_fl_rate(&large_facts(), 100, &cpu_backend(), None, Some(&rate));
        assert_eq!(plan.fl_phase_budget, None);
        assert_eq!(plan.root_alpha_cap_secs.value, None);
        // And the probe-scope helper mirrors the same predicate for callers.
        assert!(!fl_rate_scope_applies(&large_facts(), None));
        assert!(fl_rate_scope_applies(&medium_facts(), None));
        let preset = preset_from("bab:\n  root_alpha_cap_secs: 40\n");
        assert!(
            !fl_rate_scope_applies(&medium_facts(), Some(&preset)),
            "preset-owned cap: the probe must never be paid"
        );
    }

    /// WIRING: the widened window reaches the applied `BetaCrownConfig` via
    /// the materialized merged preset, and RAISES the init.rs min-composition
    /// (`alpha_config.deadline = d.min(now + cap)` — the rule changes what
    /// `capped` is, so the composed deadline moves later, never earlier).
    #[test]
    fn fl_widened_window_reaches_the_alpha_deadline_cap() {
        let onnx = medium_conv_onnx();
        // Preset with neither pair key: scope applies, probe closure runs.
        let preset = write_temp(".yaml", b"bab:\n  batch_size: 256\n");
        let out = resolve_and_materialize(
            onnx.path(),
            Some(preset.path()),
            100,
            &cuda_backend(),
            || Some(fl_probe(23.0)),
        );
        assert!(out.note.is_none(), "clean path expected: {:?}", out.note);
        let effective = out.effective_preset.as_deref().expect("merged preset");
        let merged = crate::preset::load_preset(effective).expect("parses");
        assert_eq!(merged.bab.root_alpha_cap_secs, Some(46.0));
        // Through the exact apply path a scored run performs:
        let mut config = ny_propagate::BetaCrownConfig::default();
        crate::preset::apply_preset(&mut config, &merged).expect("applies");
        let widened_cap = config.root_alpha_cap_secs.expect("cap applied");
        assert_eq!(widened_cap, 46.0);
        assert!(
            widened_cap > MEDIUM_PAIR_ROOT_ALPHA_CAP_SECS,
            "widening must RAISE the cap for this run"
        );
        // The init.rs composition direction: deadline = d.min(now + cap_secs).
        // With the same instance deadline d (95s tier), a larger cap_secs
        // yields a LATER effective alpha deadline, still bounded by d.
        let now = std::time::Instant::now();
        let d = now + std::time::Duration::from_secs(95);
        let composed_widened = d.min(now + std::time::Duration::from_secs_f64(widened_cap));
        let composed_recipe =
            d.min(now + std::time::Duration::from_secs_f64(MEDIUM_PAIR_ROOT_ALPHA_CAP_SECS));
        assert!(composed_widened > composed_recipe);
        assert!(composed_widened <= d);
        // Flight event carries the decision.
        assert!(out
            .flight_summary()
            .contains("fl_phase_budget=widened root window to 46s"));
    }

    // -- adversarial-verify boundary probes (#fl-phase-budget) ---------------

    /// Exact widening threshold at the 100s budget (tier 95): rate =
    /// 559.4 x 1.25 / 40 = 17.48125 GMAC/s gives pred = 40.0s exactly;
    /// 40 + 15 + 40 = 95 <= 95 fits, window = ceil(min(55, 55)) = 55 and BaB
    /// keeps EXACTLY the 40s floor. A hair slower must decline.
    #[test]
    fn av_fl_threshold_boundary_at_100s() {
        let at = fl_probe(17.481_25);
        let plan = resolve_plan_with_fl_rate(&medium_facts(), 100, &cpu_backend(), None, Some(&at));
        assert_eq!(plan.root_alpha_cap_secs.value, Some(55.0));
        const { assert!(95.0 - 55.0 >= FL_BAB_FLOOR_SECS) };
        let below = fl_probe(17.4);
        let plan =
            resolve_plan_with_fl_rate(&medium_facts(), 100, &cpu_backend(), None, Some(&below));
        assert_eq!(
            plan.root_alpha_cap_secs.value,
            Some(40.0),
            "must decline below threshold"
        );
        assert!(plan
            .fl_phase_budget
            .as_deref()
            .unwrap()
            .contains("declined"));
    }

    /// Property sweep: for every measured rate and tier, a widened window
    /// never eats the BaB floor and never narrows below the banked 40s.
    #[test]
    fn av_fl_window_never_eats_bab_floor_property() {
        for budget in [100u64, 116, 300, 900] {
            let tier = crate::commands::vnncomp::internal_timeout_secs(budget) as f64;
            for tenth in 1..=2000u64 {
                let rate = fl_probe(tenth as f64 / 10.0);
                let plan = resolve_plan_with_fl_rate(
                    &medium_facts(),
                    budget,
                    &cpu_backend(),
                    None,
                    Some(&rate),
                );
                if let SettingSource::ResolvedBudget(reason) = &plan.root_alpha_cap_secs.source {
                    if reason.contains("fl-phase-budget") {
                        let w = plan.root_alpha_cap_secs.value.unwrap();
                        assert!(w > 40.0, "widening must strictly raise: {w} at {budget}");
                        assert!(
                            w <= tier - FL_BAB_FLOOR_SECS,
                            "BaB floor violated: window {w}, tier {tier}, budget {budget}"
                        );
                    }
                }
            }
        }
    }

    /// The 23 GMAC/s widened window still satisfies the GATE's own integer
    /// admission arithmetic at the widened deadline (no widened-but-refused
    /// from margin rounding): remaining 46s vs padded 559/23*5/4 = 30s.
    #[test]
    fn av_fl_widened_window_passes_gate_integer_margin() {
        let macs: u128 = 559_400_000_000;
        let rate: u128 = 23_000_000_000;
        let predicted = macs / rate; // 24s (integer)
        let padded = predicted * 5 / 4; // 30s
        assert!(46 >= padded, "gate would refuse inside the widened window");
        // And at the 900s widening (76s window, 11.57 GMAC/s):
        let rate2: u128 = 11_570_000_000;
        let padded2 = (macs / rate2) * 5 / 4; // 48*5/4 = 60
        assert!(76 >= padded2);
    }

    /// Flight event either way: the DECLINED decision also rides the
    /// `plan_resolved` line.
    #[test]
    fn fl_decline_rides_the_flight_summary() {
        let onnx = medium_conv_onnx();
        let out = resolve_and_materialize(onnx.path(), None, 100, &cuda_backend(), || {
            Some(fl_probe(11.57))
        });
        let line = out.flight_summary();
        assert!(line.contains("fl_phase_budget=declined"), "{line}");
        assert!(
            line.contains("root_alpha_cap_secs=40:[resolved(budget)"),
            "{line}"
        );
    }

    /// Rule 7 (#fl-alpha-composition) arms EXACTLY on rule 6's widen events:
    /// a widening rate resolves `forward_alpha_surrogate=true` with the rule's
    /// provenance; a declined rate and an absent rate both leave it Default —
    /// the composition can never arm where FL itself was refused.
    #[test]
    fn fl_alpha_composition_arms_only_on_rule6_widening() {
        // Widen: 23 GMAC/s at 100s (same arithmetic as the rule-6 widen test).
        let rate = fl_probe(23.0);
        let plan =
            resolve_plan_with_fl_rate(&medium_facts(), 100, &cpu_backend(), None, Some(&rate));
        assert_eq!(plan.forward_alpha_surrogate.value, Some(true));
        assert!(plan.forward_alpha_surrogate.is_resolved());
        let SettingSource::ResolvedBudget(reason) = &plan.forward_alpha_surrogate.source else {
            panic!("rule-7 arming must be budget-resolved");
        };
        assert!(reason.contains("fl-alpha-composition"), "{reason}");
        assert!(reason.contains("rule-6 widening"), "{reason}");

        // Decline: the same rate rule 6 refuses at 100s must NOT arm rule 7.
        let slow = fl_probe(11.57);
        let plan =
            resolve_plan_with_fl_rate(&medium_facts(), 100, &cpu_backend(), None, Some(&slow));
        assert_eq!(plan.forward_alpha_surrogate.value, None);
        assert!(!plan.forward_alpha_surrogate.is_resolved());

        // No rate injected (scope never applied): Default, value unset.
        let plan = resolve_plan_with_fl_rate(&medium_facts(), 100, &cpu_backend(), None, None);
        assert_eq!(plan.forward_alpha_surrogate.value, None);
        assert_eq!(plan.forward_alpha_surrogate.source, SettingSource::Default);
    }

    /// Presets-win: an explicit surrogate key (either spelling, either value)
    /// pins rule 7 as a PresetOverride pass-through, even on a widening rate.
    #[test]
    fn fl_alpha_composition_preset_key_wins() {
        let rate = fl_probe(23.0);
        // Note: the preset also pins nothing else — but an explicit surrogate
        // key alone must not stop rule 6's window widening; only rule 7 pins.
        let preset = preset_from("model:\n  forward_alpha_surrogate: false\n");
        let plan = resolve_plan_with_fl_rate(
            &medium_facts(),
            100,
            &cpu_backend(),
            Some(&preset),
            Some(&rate),
        );
        assert_eq!(plan.forward_alpha_surrogate.value, Some(false));
        assert_eq!(
            plan.forward_alpha_surrogate.source,
            SettingSource::PresetOverride("model.forward_alpha_surrogate".into())
        );
        // Rule 6's own widening is unaffected by the surrogate pin.
        assert_eq!(plan.root_alpha_cap_secs.value, Some(46.0));

        let preset = preset_from(
            "model:\n  cgan_forward_alpha_surrogate: true\n  batch_norm_folding: preserve_raw\n  require_authored_float32_initializers: true\n",
        );
        let plan = resolve_plan_with_fl_rate(
            &medium_facts(),
            100,
            &cpu_backend(),
            Some(&preset),
            Some(&rate),
        );
        assert_eq!(plan.forward_alpha_surrogate.value, Some(true));
        assert_eq!(
            plan.forward_alpha_surrogate.source,
            SettingSource::PresetOverride("model.forward_linear_spec_alpha".into())
        );
    }

    /// Rule 7 materializes through the one preset channel: the merged temp
    /// preset carries `model.forward_alpha_surrogate: true`, it round-trips
    /// through `load_preset`, and — unlike the cGAN key — it passes the ONNX
    /// load-config admission WITHOUT `preserve_raw` (the loaded-graph
    /// authority contract, see the ModelPreset field docs).
    #[test]
    fn fl_alpha_composition_materializes_model_key() {
        let onnx = medium_conv_onnx();
        let out = resolve_and_materialize(onnx.path(), None, 100, &cuda_backend(), || {
            Some(fl_probe(23.0))
        });
        let line = out.flight_summary();
        assert!(
            line.contains("forward_alpha_surrogate=true:[resolved(budget)"),
            "{line}"
        );
        let merged = out
            .effective_preset
            .as_ref()
            .expect("resolved values must materialize a merged preset");
        let cfg = crate::preset::load_preset(merged).expect("merged preset must load");
        assert_eq!(cfg.model.forward_alpha_surrogate, Some(true));
        assert_eq!(cfg.model.forward_linear_spec_alpha, None);
        // δ=0 admission trip-wire: no preserve_raw demanded for this key.
        crate::preset::build_onnx_load_config(&cfg)
            .expect("forward_alpha_surrogate must not require preserve_raw");
    }
}
