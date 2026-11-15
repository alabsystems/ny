// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Instance facts and the host envelope (ROUTER_PROGRAM §2.2). DATA ONLY.
//!
//! Facts are STATIC properties of the row, computed in milliseconds inside the
//! timed run from scans the entry point already did (no prepare-phase cache,
//! no second model load). Every field is optional: a fact that was not
//! measured is `None`, and a whole-row failure is [`FactsStatus::Unknown`].
//! Neither is ever guessed, and a capability whose admission depends on an
//! unknown fact reports `Unknown`, which is a refusal (today's route runs).
//!
//! There is no name and no path here: the policy is name-blind by type.

use serde::{Deserialize, Serialize};

use super::decl::{OpSet, SpecArchetype};
use super::spec::SpecShape;

/// Whether the facts could be extracted at all.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FactsStatus {
    /// Extracted (individual fields may still be `None`).
    Known,
    /// Not extracted; the string says why (`scan>256MiB`, `scan-failed`, ...).
    Unknown(String),
}

/// Static facts of one row.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct InstanceFacts {
    /// Extraction status.
    pub status: FactsStatus,
    /// Operator kinds the scan saw.
    pub ops: OpSet,
    /// Whether `ops` is the COMPLETE operator signature (false = a lower bound).
    pub ops_complete: bool,
    /// Parameter count.
    pub params: Option<u64>,
    /// Nonlinear activation count.
    pub nonlinear: Option<u64>,
    /// ReLU count.
    pub relu: Option<u64>,
    /// Convolution layers.
    pub conv_layers: Option<u64>,
    /// Widest convolution output-channel count.
    pub max_conv_channels: Option<u64>,
    /// Perturbed input dimensions.
    pub p_eff: Option<u64>,
    /// Input dimensions.
    pub input_dims: Option<u64>,
    /// Specification archetype.
    pub spec: Option<SpecArchetype>,
    /// Output classes.
    pub classes: Option<u64>,
    /// Specification clauses.
    pub clauses: Option<u64>,
    /// Model size in bytes.
    pub model_bytes: Option<u64>,
    /// Networks the property declares (2+ = relational).
    pub declared_networks: Option<u64>,
    /// #admission-precision: the routing shape of an affine property, read
    /// at router start by the extractor the lanes route on (`None`: not an
    /// affine single-network property, or not read).
    #[serde(default)]
    pub spec_shape: Option<SpecShape>,
}

impl InstanceFacts {
    /// Facts that could not be extracted, for `reason`.
    #[must_use]
    pub fn unknown(reason: &str) -> Self {
        Self {
            status: FactsStatus::Unknown(reason.to_owned()),
            ops: OpSet::EMPTY,
            ops_complete: false,
            params: None,
            nonlinear: None,
            relu: None,
            conv_layers: None,
            max_conv_channels: None,
            p_eff: None,
            input_dims: None,
            spec: None,
            classes: None,
            clauses: None,
            model_bytes: None,
            declared_networks: None,
            spec_shape: None,
        }
    }

    /// Known facts with every field unmeasured (callers fill what they have).
    #[must_use]
    pub const fn known_empty() -> Self {
        Self {
            status: FactsStatus::Known,
            ops: OpSet::EMPTY,
            ops_complete: false,
            params: None,
            nonlinear: None,
            relu: None,
            conv_layers: None,
            max_conv_channels: None,
            p_eff: None,
            input_dims: None,
            spec: None,
            classes: None,
            clauses: None,
            model_bytes: None,
            declared_networks: None,
            spec_shape: None,
        }
    }

    /// Whether extraction succeeded.
    #[must_use]
    pub const fn is_known(&self) -> bool {
        matches!(self.status, FactsStatus::Known)
    }
}

/// A declared host class (§8).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HostClass {
    /// The GB10 development host (aarch64, 121 GiB unified memory).
    Gb10,
    /// An AWS g5.8xlarge (32 vCPU, 128 GiB, A10G).
    G5,
    /// An AWS m5.16xlarge (64 vCPU, 256 GiB, no GPU): the default competition host.
    M5,
    /// Anything else.
    Other,
}

impl HostClass {
    /// Stable token.
    #[must_use]
    pub const fn token(self) -> &'static str {
        match self {
            Self::Gb10 => "gb10",
            Self::G5 => "g5",
            Self::M5 => "m5",
            Self::Other => "other",
        }
    }

    /// Parse a token.
    #[must_use]
    pub fn from_token(token: &str) -> Option<Self> {
        match token {
            "gb10" => Some(Self::Gb10),
            "g5" => Some(Self::G5),
            "m5" => Some(Self::M5),
            "other" => Some(Self::Other),
            _ => None,
        }
    }

    /// Classify the RUNNING host from coarse static facts. Deliberately
    /// coarse: a wrong class only changes logged plan values.
    #[must_use]
    pub const fn classify(aarch64: bool, threads: u32, mem_bytes: u64) -> Self {
        const GIB: u64 = 1 << 30;
        if aarch64 && mem_bytes >= 100 * GIB && mem_bytes <= 140 * GIB {
            Self::Gb10
        } else if !aarch64 && threads == 64 && mem_bytes >= 224 * GIB {
            Self::M5
        } else if !aarch64 && threads == 32 && mem_bytes >= 112 * GIB && mem_bytes <= 144 * GIB {
            Self::G5
        } else {
            Self::Other
        }
    }
}

/// Device qualification (§8).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceQual {
    /// No device.
    None,
    /// A device usable for untrusted search.
    Search,
    /// A device qualified for verdict-feeding bounds.
    VerdictQualified,
}

impl DeviceQual {
    /// Stable token.
    #[must_use]
    pub const fn token(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Search => "search",
            Self::VerdictQualified => "verdict_qualified",
        }
    }
}

/// The host the plan is made against (§8). Width and memory are LOGGED plan
/// values, never a silent `available_parallelism()` inside a decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HostEnvelope {
    /// Host class.
    pub class: HostClass,
    /// Worker threads.
    pub threads: u32,
    /// `min(MemTotal, cgroup limit)` in bytes (0 = unknown).
    pub mem_bytes: u64,
    /// Bytes reserved for the process and the OS.
    pub reserve_bytes: u64,
    /// Device qualification.
    pub device: DeviceQual,
    /// True when declared (a measurement-only target host), false when read
    /// from the running host.
    pub declared: bool,
}

impl HostEnvelope {
    /// The static memory envelope `E_static = mem − reserve` (saturating).
    #[must_use]
    pub const fn envelope_bytes(&self) -> u64 {
        self.mem_bytes.saturating_sub(self.reserve_bytes)
    }

    /// The declared envelope of a target host class (§8 table), for replaying
    /// the plan a different host would make. `Other` declares nothing known.
    #[must_use]
    pub const fn declared_target(class: HostClass) -> Self {
        const GIB: u64 = 1 << 30;
        match class {
            HostClass::Gb10 => Self {
                class,
                threads: 20,
                mem_bytes: 121 * GIB,
                reserve_bytes: 16 * GIB,
                device: DeviceQual::Search,
                declared: true,
            },
            HostClass::G5 => Self {
                class,
                threads: 32,
                mem_bytes: 128 * GIB,
                reserve_bytes: 8 * GIB,
                device: DeviceQual::Search,
                declared: true,
            },
            HostClass::M5 => Self {
                class,
                threads: 64,
                mem_bytes: 256 * GIB,
                reserve_bytes: 16 * GIB,
                device: DeviceQual::None,
                declared: true,
            },
            HostClass::Other => Self {
                class,
                threads: 0,
                mem_bytes: 0,
                reserve_bytes: 0,
                device: DeviceQual::None,
                declared: true,
            },
        }
    }
}
