// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Static row facts and the host envelope for the router (ROUTER_PROGRAM
//! §2.2 FACTS, §8).
//!
//! Facts come ONLY from reads the entry point hands over — the plan
//! resolver's single-pass protobuf scan (`plan_resolver::ModelFacts`), the
//! relational route's declared-network count and, since #admission-precision,
//! the property's routing shape (`beta_crown::spec_shape::read_spec`: the
//! lanes' own parser and the one extractor the lanes route on). Nothing here
//! re-parses the model or the property, and nothing reads a name or a path:
//! the scan is `None` above `SCAN_MAX_BYTES` (256 MiB) or on a parse failure,
//! and the facts are then `Unknown`, which every facts-dependent predicate
//! treats as a refusal; an unread shape is `None`, which defers every
//! specification check to the consumer's own call site.
//!
//! The host envelope is read from the probes `ny vnncomp` already cached at
//! entry (`compute_backend::host`, `compute_backend::detect`) plus the
//! process's own cgroup v2 `memory.max` — a STATIC limit, never the live
//! headroom. Width is `available_parallelism()` as the host probe cached it,
//! and never `rayon::current_num_threads()`: the global pool is configured
//! later, from the frozen margin-row policy, and must not be initialised here.

use ny_core::router::decl::{OpKind, OpSet};
use ny_core::router::facts::{DeviceQual, HostClass, HostEnvelope, InstanceFacts};

use crate::commands::beta_crown::spec_shape::SpecRead;
use crate::plan_resolver::ModelFacts;

/// The plan resolver's scan ceiling (`plan_resolver::SCAN_MAX_BYTES`).
const SCAN_MAX_BYTES: u64 = 256 * 1024 * 1024;

/// Bytes the envelope reserves for the process and the OS (§8: 16 GiB).
pub(crate) const HOST_RESERVE_BYTES: u64 = 16 << 30;

/// Why the scan produced no facts, from the on-disk size alone.
#[must_use]
pub(crate) fn unavailable_reason(model_bytes: Option<u64>) -> &'static str {
    match model_bytes {
        Some(bytes) if bytes > SCAN_MAX_BYTES => "scan>256MiB",
        Some(0) => "empty-model",
        Some(_) => "scan-unavailable",
        None => "model-unreadable",
    }
}

/// #admission-precision: the property's facts — its archetype, its routing
/// shape and the counts the shape carries — on `facts`.
fn record_spec(facts: &mut InstanceFacts, spec: &SpecRead) {
    facts.spec = spec.archetype();
    facts.spec_shape = spec.shape();
    if let Some(shape) = spec.shape() {
        facts.input_dims = Some(shape.inputs);
        facts.p_eff = Some(shape.perturbed_inputs);
        facts.classes = Some(shape.outputs);
        facts.clauses = Some(shape.effective_clauses());
    }
}

/// Instance facts from the entry point's existing scan results.
#[must_use]
pub(crate) fn instance_facts(
    model: Option<&ModelFacts>,
    model_bytes: Option<u64>,
    declared_networks: Option<u64>,
    reason: Option<&'static str>,
    spec: &SpecRead,
) -> InstanceFacts {
    let Some(model) = model else {
        let mut facts =
            InstanceFacts::unknown(reason.unwrap_or_else(|| unavailable_reason(model_bytes)));
        facts.model_bytes = model_bytes;
        facts.declared_networks = declared_networks;
        record_spec(&mut facts, spec);
        return facts;
    };
    let mut facts = InstanceFacts::known_empty();
    facts.params = u64::try_from(model.param_count).ok();
    facts.conv_layers = u64::try_from(model.conv_layers).ok();
    facts.max_conv_channels = u64::try_from(model.max_conv_out_channels).ok();
    facts.model_bytes = Some(model.file_size_bytes);
    if model.conv_layers > 0 {
        facts.ops = facts.ops.with(OpKind::Conv);
    }
    // #router-r5: the scan's operator tokens give the COMPLETE signature
    // (every token maps to a kind, `other`, or a pure reshape). Without them
    // (the printer path loads the model and keeps no tokens) the signature is
    // a LOWER bound and every envelope check refuses (`ops-incomplete`).
    facts.ops_complete = !model.op_tokens.is_empty();
    for token in &model.op_tokens {
        facts.ops = OpSet(facts.ops.0 | op_kinds(token).0);
    }
    facts.declared_networks = declared_networks;
    record_spec(&mut facts, spec);
    facts
}

/// #router-r5: the operator kinds one scan token stands for. Conservative by
/// construction: an ambiguous operator maps to the kind that is HARDER to
/// cover (an `Add` whose operands are both computed is a residual join; a
/// `Mul` / `MatMul` / `Gemm` of two computed factors, and a `Div` whose
/// divisor is computed, are bilinear; a convolution reading a computed
/// kernel is a convolution AND a bilinear product), so the row's signature
/// is a superset of what it computes and an envelope that covers it covers
/// the row. An unrecognised operator is `other`; pure reshapes carry no kind.
///
/// Review fix (#router-r5): `Gemm` was mapped to `linear` whatever its
/// factors, `Div` was `linear` whenever ANY operand was an initializer (so
/// `c / x`, a reciprocal, read as linear), a computed-kernel convolution read
/// as a plain convolution, and `Cast` carried no kind although a cast to an
/// integer type rounds (the scan does not read its `to` attribute). The scan
/// now records the operand position (`plan_resolver::scan_onnx_graph`) and
/// `Cast` is `other`.
#[must_use]
pub(crate) fn op_kinds(token: &str) -> OpSet {
    let one = |kind: OpKind| OpSet::EMPTY.with(kind);
    match token {
        "Conv" => one(OpKind::Conv),
        "ConvTranspose" => one(OpKind::ConvTranspose),
        "Conv:dynamic" => one(OpKind::Conv).with(OpKind::Bilinear),
        "ConvTranspose:dynamic" => one(OpKind::ConvTranspose).with(OpKind::Bilinear),
        "Gemm:const" | "MatMul:const" | "Add:const" | "Sub:const" | "Mul:const" | "Div:const" => {
            one(OpKind::Linear)
        }
        "Gemm" | "MatMul" | "Mul" | "Div" => one(OpKind::Bilinear),
        "Add" | "Sub" => one(OpKind::Residual),
        "Relu" => one(OpKind::Relu),
        "Sigmoid" => one(OpKind::Sigmoid),
        "Tanh" => one(OpKind::Tanh),
        "Softmax" | "LogSoftmax" => one(OpKind::Softmax),
        "BatchNormalization" => one(OpKind::BatchNorm),
        "Concat" => one(OpKind::Concat),
        "MaxPool" | "AveragePool" | "GlobalAveragePool" | "GlobalMaxPool" => one(OpKind::Pool),
        "Sign" => one(OpKind::Sign),
        "Resize" | "Upsample" => one(OpKind::Resize),
        "LayerNormalization" | "GroupNormalization" | "InstanceNormalization" => one(OpKind::Norm),
        "Flatten" | "Reshape" | "Transpose" | "Squeeze" | "Unsqueeze" | "Identity" | "Constant"
        | "Shape" | "Dropout" => OpSet::EMPTY,
        _ => one(OpKind::Other),
    }
}

/// The process's own cgroup v2 `memory.max`, when it is a number.
#[cfg(target_os = "linux")]
fn cgroup_v2_limit_bytes() -> Option<u64> {
    let membership = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let path = membership
        .lines()
        .find_map(|line| line.strip_prefix("0::"))?
        .trim();
    let limit = std::fs::read_to_string(format!("/sys/fs/cgroup{path}/memory.max")).ok()?;
    limit.trim().parse::<u64>().ok()
}

#[cfg(not(target_os = "linux"))]
fn cgroup_v2_limit_bytes() -> Option<u64> {
    None
}

/// The host envelope: a declared target (measurement only), else the running
/// host as the entry point already probed it.
#[must_use]
pub(crate) fn host_envelope(target: Option<HostClass>) -> HostEnvelope {
    if let Some(class) = target {
        return HostEnvelope::declared_target(class);
    }
    let probe = crate::compute_backend::host();
    let threads = u32::try_from(probe.logical_cores).unwrap_or(u32::MAX);
    let mem_total = probe.ram_bytes;
    let mem_bytes = match cgroup_v2_limit_bytes() {
        Some(limit) if mem_total == 0 => limit,
        Some(limit) => mem_total.min(limit),
        None => mem_total,
    };
    let device = if crate::compute_backend::detect().kind == "cpu-only" {
        DeviceQual::None
    } else {
        DeviceQual::Search
    };
    HostEnvelope {
        class: HostClass::classify(cfg!(target_arch = "aarch64"), threads, mem_bytes),
        threads,
        mem_bytes,
        reserve_bytes: HOST_RESERVE_BYTES,
        device,
        declared: false,
    }
}
