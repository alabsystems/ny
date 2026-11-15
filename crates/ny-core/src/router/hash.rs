// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The static-input plan hash (ROUTER_PROGRAM §2.4, GC K3.2).
//!
//! sha256 over a canonical, length-prefixed encoding of the STATIC inputs
//! only: facts, the budget view, the host envelope, the capabilities with
//! their admission answers, pins, the priors version and the registry
//! version. Progress samples, wall rates and amendments are logged separately
//! and never enter it, so two runs of one row on one host hash identically
//! whatever their timings, and both proof profiles hash identically because
//! the input has no profile field.
//!
//! Every struct is destructured exhaustively, so adding an input field fails
//! to compile here until the hash covers it.

use sha2::{Digest, Sha256};

use super::decl::{
    AdmitSite, AdmitWhen, AmendmentSet, DeviceUse, KnobRisk, LaneRole, ProgressKind, SafePoints,
    SpecArchetype, SrLevel, WorkUnit,
};
use super::facts::{DeviceQual, FactsStatus, HostClass, HostEnvelope, InstanceFacts};
use super::plan::{
    AdmissionState, BudgetView, CapClass, CapInput, LedgerState, PolicyInput, RouteSpec,
};
use super::priors::{PhasePrediction, PricedCap, PriorView, UnpricedCap};
use super::spec::SpecShape;

const DOMAIN: &[u8] = b"ny-router/static-hash/v1\0";

struct Encoder {
    hash: Sha256,
}

impl Encoder {
    fn tag(&mut self, tag: u8) {
        self.hash.update([tag]);
    }

    fn u64(&mut self, value: u64) {
        self.hash.update(value.to_le_bytes());
    }

    fn bool(&mut self, value: bool) {
        self.tag(u8::from(value));
    }

    fn str(&mut self, value: &str) {
        self.u64(value.len() as u64);
        self.hash.update(value.as_bytes());
    }

    fn opt_u64(&mut self, value: Option<u64>) {
        match value {
            Some(v) => {
                self.tag(1);
                self.u64(v);
            }
            None => self.tag(0),
        }
    }
}

const fn spec_tag(spec: Option<SpecArchetype>) -> u8 {
    match spec {
        None => 0,
        Some(SpecArchetype::Robustness) => 1,
        Some(SpecArchetype::Halfspace) => 2,
        Some(SpecArchetype::Disjunctive) => 3,
        Some(SpecArchetype::Relational) => 4,
        Some(SpecArchetype::Nonlinear) => 5,
    }
}

const fn progress_tag(kind: ProgressKind) -> u8 {
    match kind {
        ProgressKind::FrontierBinary => 0,
        ProgressKind::FrontierKary => 1,
        ProgressKind::Bfs => 2,
        ProgressKind::Volume => 3,
        ProgressKind::Margin => 4,
        ProgressKind::Screen => 5,
        ProgressKind::Dual => 6,
        ProgressKind::Yield => 7,
        ProgressKind::Opaque => 8,
    }
}

const fn unit_tag(unit: WorkUnit) -> u8 {
    match unit {
        WorkUnit::Expansion => 0,
        WorkUnit::Node => 1,
        WorkUnit::Batch => 2,
        WorkUnit::Level => 3,
        WorkUnit::Restart => 4,
        WorkUnit::Iteration => 5,
        WorkUnit::Wave => 6,
        WorkUnit::Root => 7,
        WorkUnit::Pass => 8,
        WorkUnit::Solve => 9,
        WorkUnit::Step => 10,
    }
}

const fn host_tag(class: HostClass) -> u8 {
    match class {
        HostClass::Gb10 => 0,
        HostClass::G5 => 1,
        HostClass::M5 => 2,
        HostClass::Other => 3,
    }
}

const fn device_tag(device: DeviceQual) -> u8 {
    match device {
        DeviceQual::None => 0,
        DeviceQual::Search => 1,
        DeviceQual::VerdictQualified => 2,
    }
}

/// The device a capability needs (#router-r5: hashed with its routing data).
const fn device_use_tag(device: DeviceUse) -> u8 {
    match device {
        DeviceUse::None => 0,
        DeviceUse::Search => 1,
        DeviceUse::VerdictQualified => 2,
    }
}

fn facts(enc: &mut Encoder, facts: &InstanceFacts) {
    let InstanceFacts {
        status,
        ops,
        ops_complete,
        params,
        nonlinear,
        relu,
        conv_layers,
        max_conv_channels,
        p_eff,
        input_dims,
        spec,
        classes,
        clauses,
        model_bytes,
        declared_networks,
        spec_shape,
    } = facts;
    match status {
        FactsStatus::Known => enc.tag(0),
        FactsStatus::Unknown(why) => {
            enc.tag(1);
            enc.str(why);
        }
    }
    enc.u64(ops.0);
    enc.bool(*ops_complete);
    for value in [
        *params,
        *nonlinear,
        *relu,
        *conv_layers,
        *max_conv_channels,
        *p_eff,
        *input_dims,
    ] {
        enc.opt_u64(value);
    }
    enc.tag(spec_tag(*spec));
    for value in [*classes, *clauses, *model_bytes, *declared_networks] {
        enc.opt_u64(value);
    }
    match spec_shape {
        None => enc.tag(0),
        Some(shape) => {
            enc.tag(1);
            spec_shape_fields(enc, shape);
        }
    }
}

/// #admission-precision: the specification's routing shape.
fn spec_shape_fields(enc: &mut Encoder, shape: &SpecShape) {
    let SpecShape {
        atoms,
        clauses,
        max_clause_atoms,
        boxed_clauses,
        disjunction,
        relational_atoms,
        argmax_margin,
        multi_clause_disjunction,
        boxed_clause_disjunction,
        same_lhs_reducible,
        inputs,
        perturbed_inputs,
        outputs,
    } = shape;
    for value in [
        *atoms,
        *clauses,
        *max_clause_atoms,
        *boxed_clauses,
        *inputs,
        *perturbed_inputs,
        *outputs,
    ] {
        enc.u64(value);
    }
    for flag in [
        *disjunction,
        *relational_atoms,
        *argmax_margin,
        *multi_clause_disjunction,
        *boxed_clause_disjunction,
        *same_lhs_reducible,
    ] {
        enc.bool(flag);
    }
}

/// #admission-precision: a consumer's site inside its parent lane.
const fn site_tag(site: Option<AdmitSite>) -> u8 {
    match site {
        None => 0,
        Some(AdmitSite::DisjunctiveLaneGraph) => 1,
        Some(AdmitSite::DisjunctiveMultiObjective) => 2,
        Some(AdmitSite::GraphInputSplit) => 3,
        Some(AdmitSite::DagGraph) => 4,
        Some(AdmitSite::GraphReluSplit) => 5,
    }
}

fn host(enc: &mut Encoder, host: &HostEnvelope) {
    let HostEnvelope {
        class,
        threads,
        mem_bytes,
        reserve_bytes,
        device,
        declared,
    } = host;
    enc.tag(host_tag(*class));
    enc.u64(u64::from(*threads));
    enc.u64(*mem_bytes);
    enc.u64(*reserve_bytes);
    enc.tag(device_tag(*device));
    enc.bool(*declared);
}

fn budget(enc: &mut Encoder, budget: &BudgetView) {
    let BudgetView {
        official_ms,
        internal_ms,
        join_margin_ms,
    } = budget;
    enc.u64(*official_ms);
    enc.u64(*internal_ms);
    enc.u64(*join_margin_ms);
}

fn capability(enc: &mut Encoder, cap: &CapInput) {
    let CapInput {
        id,
        class,
        ledger,
        progress,
        unit,
        first_rung_ms,
        safe_points,
        accepts,
        admission,
        route,
    } = cap;
    enc.str(id);
    match class {
        CapClass::Lane(role) => {
            enc.tag(0);
            match role {
                LaneRole::Attack => enc.tag(0),
                LaneRole::Prover { terminal_ok } => {
                    enc.tag(1);
                    enc.bool(*terminal_ok);
                }
                LaneRole::Screen => enc.tag(2),
            }
        }
        CapClass::Tier { parent } => {
            enc.tag(1);
            enc.str(parent);
        }
        CapClass::Phase { lane } => {
            enc.tag(2);
            enc.str(lane);
        }
        CapClass::Variant { slot } => {
            enc.tag(3);
            enc.str(slot);
        }
        CapClass::Rule => enc.tag(4),
    }
    match ledger {
        LedgerState::Open(why) => {
            enc.tag(0);
            enc.str(why);
        }
        LedgerState::Shadow => enc.tag(1),
        LedgerState::SealedAsShipped => enc.tag(2),
        LedgerState::Sealed => enc.tag(3),
    }
    enc.tag(progress_tag(*progress));
    enc.tag(unit_tag(*unit));
    enc.opt_u64(*first_rung_ms);
    enc.tag(match safe_points {
        SafePoints::None => 0,
        SafePoints::BetweenUnits => 1,
    });
    let AmendmentSet {
        deadline,
        hand_on_source,
        hand_on_sink,
        defer,
    } = accepts;
    for flag in [*deadline, *hand_on_source, *hand_on_sink, *defer] {
        enc.bool(flag);
    }
    match admission {
        AdmissionState::Admit => enc.tag(0),
        AdmissionState::Decline(why) => {
            enc.tag(1);
            enc.str(why);
        }
        AdmissionState::Unknown(why) => {
            enc.tag(2);
            enc.str(why);
        }
    }
    match route {
        None => enc.tag(0),
        Some(spec) => {
            enc.tag(1);
            route_spec(enc, spec);
        }
    }
}

/// #router-r5: a routable capability's routing data.
fn route_spec(enc: &mut Encoder, spec: &RouteSpec) {
    let RouteSpec {
        sr,
        parent,
        admit_when,
        envelope_ops,
        envelope_specs,
        risk,
        settings,
        slot,
    } = spec;
    enc.tag(match sr {
        SrLevel::Decl => 0,
        SrLevel::Obs => 1,
        SrLevel::Seal => 2,
    });
    opt_str(enc, parent.as_deref());
    let AdmitWhen {
        ops_any,
        specs,
        max_params,
        max_p_eff,
        device,
        min_threads,
        site,
    } = admit_when;
    enc.u64(ops_any.0);
    enc.u64(u64::from(specs.0));
    enc.opt_u64(*max_params);
    enc.opt_u64(*max_p_eff);
    enc.tag(device_use_tag(*device));
    enc.u64(u64::from(*min_threads));
    enc.tag(site_tag(*site));
    enc.u64(envelope_ops.0);
    enc.u64(u64::from(envelope_specs.0));
    enc.tag(match risk {
        KnobRisk::Low => 0,
        KnobRisk::High => 1,
    });
    enc.u64(settings.len() as u64);
    for (key, value) in settings {
        enc.str(key);
        enc.str(value);
    }
    opt_str(enc, slot.as_deref());
}

fn opt_str(enc: &mut Encoder, value: Option<&str>) {
    match value {
        Some(text) => {
            enc.tag(1);
            enc.str(text);
        }
        None => enc.tag(0),
    }
}

/// sha256 of the policy's static input.
#[must_use]
pub fn static_hash(input: &PolicyInput) -> [u8; 32] {
    let PolicyInput {
        schema,
        facts: row_facts,
        host: row_host,
        budget: row_budget,
        capabilities,
        pins,
        priors_version,
        registry_version,
        priors,
    } = input;
    let mut enc = Encoder {
        hash: Sha256::new(),
    };
    enc.hash.update(DOMAIN);
    enc.str(schema);
    facts(&mut enc, row_facts);
    host(&mut enc, row_host);
    budget(&mut enc, row_budget);
    enc.u64(capabilities.len() as u64);
    for cap in capabilities {
        capability(&mut enc, cap);
    }
    enc.u64(pins.len() as u64);
    for pin in pins {
        enc.str(pin);
    }
    enc.str(priors_version);
    enc.str(registry_version);
    // #router-r7: the prior's view is static (compiled table x static facts x
    // the declared host), so it is hashed — but ONLY when present: an input
    // without a prior encodes exactly R5's bytes.
    if let Some(view) = priors {
        prior_view(&mut enc, view);
    }
    enc.hash.finalize().into()
}

fn prior_view(enc: &mut Encoder, view: &PriorView) {
    let PriorView {
        version,
        host,
        cell,
        predictions,
        priced,
        unpriced,
    } = view;
    enc.tag(0x7a);
    enc.str(version);
    enc.str(host);
    opt_str(enc, cell.as_deref());
    enc.u64(predictions.len() as u64);
    for PhasePrediction {
        phase,
        cell,
        median_ms,
        upper_ms,
    } in predictions
    {
        enc.str(phase);
        enc.str(cell);
        enc.u64(*median_ms);
        enc.u64(*upper_ms);
    }
    enc.u64(priced.len() as u64);
    for PricedCap { cap, ms } in priced {
        enc.str(cap);
        enc.u64(*ms);
    }
    enc.u64(unpriced.len() as u64);
    for UnpricedCap { cap, reason } in unpriced {
        enc.str(cap);
        enc.str(reason);
    }
}

/// Lowercase hex.
#[must_use]
pub fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(char::from(DIGITS[usize::from(byte >> 4)]));
        out.push(char::from(DIGITS[usize::from(byte & 0x0f)]));
    }
    out
}

/// The first 8 hex digits of a digest.
#[must_use]
pub fn sha8(digest: &[u8; 32]) -> String {
    let mut full = hex(digest);
    full.truncate(8);
    full
}
