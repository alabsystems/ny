// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Version 1 retained-BaB topology/layout schema.
//!
//! The format is explicit little-endian data. Unknown tags, nonzero reserved
//! fields, noncanonical ranges, and trailing bytes are rejected. It never uses
//! Rust layout, serde, bincode, or debug text as an ABI.

// The crate-private encoder is consumed only by the next default-dark runtime
// checkpoint; the decoder is already the shared compatibility surface.
#![allow(dead_code)]

use ny_core::{
    NyError, Result, GPU_BAB_BOUND_MAX_ARENA_VALUES, GPU_BAB_BOUND_MAX_SPLIT_HISTORY_WORDS,
};
use std::collections::HashSet;
use std::mem::size_of;

pub const RESIDENT_BAB_TOPOLOGY_SCHEMA_V1: u32 = 1;
pub const RESIDENT_BAB_TOPOLOGY_MAGIC_V1: [u8; 8] = *b"NYRBWV1\0";
pub const RESIDENT_BAB_TOPOLOGY_HEADER_BYTES_V1: u32 = 168;
pub const RESIDENT_BAB_TOPOLOGY_ENDIAN_MARKER_V1: u32 = 0x0102_0304;

pub const RESIDENT_BAB_NETWORK_INPUT_ID_V1: u32 = u32::MAX;
const MAX_TOPOLOGY_BYTES: usize = 64 * 1024 * 1024;
const MAX_RECORDS: usize = 1 << 18;
const MAX_RANK: usize = 8;
const MAX_NODE_INPUTS: usize = 2;
const MAX_NODE_NAME_BYTES: usize = 4096;
const WIRE_POLL_STRIDE: usize = 1024;
const NODE_FIXED_BYTES: usize = 32;
const SEGMENT_BYTES: usize = 48;
const LAYER_BYTES: usize = 152;
const HASH_BUCKET_OVERHEAD_BYTES: usize = 16;

pub(crate) const RESIDENT_BAB_MAX_NODE_NAME_BYTES_V1: usize = MAX_NODE_NAME_BYTES;
pub(crate) const RESIDENT_BAB_MAX_RECORDS_V1: usize = MAX_RECORDS;
pub(crate) const RESIDENT_BAB_MAX_RANK_V1: usize = MAX_RANK;

/// Crate-private eligibility result for the exact v1 wire-length equations.
///
/// A coherent producer topology can satisfy every per-record cap yet exceed
/// the aggregate 64 MiB schema bound. The pre-open static composer uses this
/// typed result to classify that case as an ordinary unsupported eligibility
/// miss, while the standalone encoder continues to reject it as invalid input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ResidentBabTopologyWireLengthPreflightV1 {
    Encodable { encoded_bytes: usize },
    ExceedsV1ByteCap,
}

/// Closed interpretation of one retained-BaB v1 ReLU row.
///
/// This is a bit-level wire tag, not an arithmetic scalar. Only positive-zero
/// (`0x00000000`) and positive-one (`0x3f800000`) are canonical. In
/// particular, negative zero is rejected.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidentBabActivationVariantV1 {
    /// The four executed sections are `lower_slope`, `upper_slope`,
    /// `lower_intercept`, and `upper_intercept`.
    Ordinary,
    /// The four executed sections are `lower_pos_slope`, `cross_slope`,
    /// `upper_neg_slope`, and `cross_intercept`.
    DualAlpha,
}

/// Allocation-free error returned for a noncanonical v1 Activation tag.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidentBabActivationTagErrorV1;

impl ResidentBabActivationVariantV1 {
    pub const ORDINARY_TAG_BITS: u32 = 0x0000_0000;
    pub const DUAL_ALPHA_TAG_BITS: u32 = 0x3f80_0000;

    /// Return the exact in-band f32 tag bits for this variant.
    #[must_use]
    pub const fn tag_bits(self) -> u32 {
        match self {
            Self::Ordinary => Self::ORDINARY_TAG_BITS,
            Self::DualAlpha => Self::DUAL_ALPHA_TAG_BITS,
        }
    }

    /// Decode the closed v1 tag without allocating or formatting diagnostics.
    pub const fn from_tag_bits(
        bits: u32,
    ) -> std::result::Result<Self, ResidentBabActivationTagErrorV1> {
        match bits {
            Self::ORDINARY_TAG_BITS => Ok(Self::Ordinary),
            Self::DUAL_ALPHA_TAG_BITS => Ok(Self::DualAlpha),
            _ => Err(ResidentBabActivationTagErrorV1),
        }
    }
}

fn invalid(message: impl Into<String>) -> NyError {
    NyError::InvalidSpec(message.into())
}

fn poll_wire(
    check: &mut dyn FnMut(&'static str) -> Result<()>,
    label: &'static str,
    index: usize,
) -> Result<()> {
    if index.is_multiple_of(WIRE_POLL_STRIDE) {
        check(label)?;
    }
    Ok(())
}

fn initialize_coverage_v1(
    coverage: &mut Vec<u8>,
    count: usize,
    check: &mut dyn FnMut(&'static str) -> Result<()>,
    label: &'static str,
    final_label: &'static str,
) -> Result<()> {
    if !coverage.is_empty() || coverage.capacity() < count {
        return Err(invalid(
            "retained-BaB coverage storage was not reserved exactly before initialization",
        ));
    }
    for index in 0..count {
        poll_wire(check, label, index)?;
        coverage.push(0);
    }
    check(final_label)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResidentBabWireRangeV1 {
    pub start: u64,
    pub len: u64,
}

impl ResidentBabWireRangeV1 {
    fn end(self, label: &str) -> Result<u64> {
        self.start
            .checked_add(self.len)
            .ok_or_else(|| invalid(format!("retained-BaB {label} range overflows u64")))
    }
}

/// Checked subranges of one tagged v1 Activation row.
///
/// The six numeric sections are always contiguous and width-sized. The
/// meaning of the final four is selected only by
/// [`ResidentBabActivationVariantV1`].
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidentBabActivationSectionsV1 {
    pub tag_index: u64,
    pub pre_lower: ResidentBabWireRangeV1,
    pub pre_upper: ResidentBabWireRangeV1,
    pub section_0: ResidentBabWireRangeV1,
    pub section_1: ResidentBabWireRangeV1,
    pub section_2: ResidentBabWireRangeV1,
    pub section_3: ResidentBabWireRangeV1,
}

/// Allocation-free failure for malformed Activation row arithmetic/layout.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidentBabActivationLayoutErrorV1;

impl ResidentBabActivationSectionsV1 {
    /// Split an exact `tag + 6 * width` row without allocation.
    pub const fn from_row(
        row: ResidentBabWireRangeV1,
        width: u64,
    ) -> std::result::Result<Self, ResidentBabActivationLayoutErrorV1> {
        if width == 0 {
            return Err(ResidentBabActivationLayoutErrorV1);
        }
        let Some(numeric_len) = width.checked_mul(6) else {
            return Err(ResidentBabActivationLayoutErrorV1);
        };
        let Some(expected_len) = numeric_len.checked_add(1) else {
            return Err(ResidentBabActivationLayoutErrorV1);
        };
        if row.len != expected_len {
            return Err(ResidentBabActivationLayoutErrorV1);
        }
        let tag_index = row.start;
        let Some(first) = row.start.checked_add(1) else {
            return Err(ResidentBabActivationLayoutErrorV1);
        };
        let Some(second) = first.checked_add(width) else {
            return Err(ResidentBabActivationLayoutErrorV1);
        };
        let Some(third) = second.checked_add(width) else {
            return Err(ResidentBabActivationLayoutErrorV1);
        };
        let Some(fourth) = third.checked_add(width) else {
            return Err(ResidentBabActivationLayoutErrorV1);
        };
        let Some(fifth) = fourth.checked_add(width) else {
            return Err(ResidentBabActivationLayoutErrorV1);
        };
        let Some(sixth) = fifth.checked_add(width) else {
            return Err(ResidentBabActivationLayoutErrorV1);
        };
        let Some(end) = sixth.checked_add(width) else {
            return Err(ResidentBabActivationLayoutErrorV1);
        };
        let Some(row_end) = row.start.checked_add(row.len) else {
            return Err(ResidentBabActivationLayoutErrorV1);
        };
        if end != row_end {
            return Err(ResidentBabActivationLayoutErrorV1);
        }
        Ok(Self {
            tag_index,
            pre_lower: ResidentBabWireRangeV1 {
                start: first,
                len: width,
            },
            pre_upper: ResidentBabWireRangeV1 {
                start: second,
                len: width,
            },
            section_0: ResidentBabWireRangeV1 {
                start: third,
                len: width,
            },
            section_1: ResidentBabWireRangeV1 {
                start: fourth,
                len: width,
            },
            section_2: ResidentBabWireRangeV1 {
                start: fifth,
                len: width,
            },
            section_3: ResidentBabWireRangeV1 {
                start: sixth,
                len: width,
            },
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ResidentBabNodeKindV1 {
    Linear = 1,
    Conv2d = 2,
    Relu = 3,
    /// Row-major structural no-op in the normalized resident layout. V1 binds
    /// the exact source/output shapes and equal element count, not the raw
    /// ONNX `axis` spelling; a producer may normalize directly to rank one.
    Flatten = 4,
    /// Row-major structural no-op with an exact normalized target shape.
    /// Raw framework reshape operands are deliberately outside V1 identity.
    Reshape = 5,
    Add = 6,
}

impl ResidentBabNodeKindV1 {
    fn from_wire(tag: u8) -> Result<Self> {
        match tag {
            1 => Ok(Self::Linear),
            2 => Ok(Self::Conv2d),
            3 => Ok(Self::Relu),
            4 => Ok(Self::Flatten),
            5 => Ok(Self::Reshape),
            6 => Ok(Self::Add),
            _ => Err(invalid(format!(
                "retained-BaB topology has unknown node tag {tag}"
            ))),
        }
    }

    fn input_count(self) -> usize {
        match self {
            Self::Add => 2,
            _ => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ResidentBabSegmentKindV1 {
    Chain = 1,
    Residual = 2,
    ResidualProjection = 3,
}

impl ResidentBabSegmentKindV1 {
    fn from_wire(tag: u8) -> Result<Self> {
        match tag {
            1 => Ok(Self::Chain),
            2 => Ok(Self::Residual),
            3 => Ok(Self::ResidualProjection),
            _ => Err(invalid(format!(
                "retained-BaB topology has unknown segment tag {tag}"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ResidentBabLayerKindV1 {
    Linear = 1,
    Conv2d = 2,
    Relu = 3,
}

impl ResidentBabLayerKindV1 {
    fn from_wire(tag: u8) -> Result<Self> {
        match tag {
            1 => Ok(Self::Linear),
            2 => Ok(Self::Conv2d),
            3 => Ok(Self::Relu),
            _ => Err(invalid(format!(
                "retained-BaB topology has unknown layer tag {tag}"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ResidentBabLayerBranchV1 {
    Main = 1,
    Projection = 2,
}

/// Provenance class of the compact input-side Abs row for one segment.
///
/// A chain row is the main branch input. Residual rows are the exact shared
/// input from which both branches originate; they are never attributed to one
/// branch's internal state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ResidentBabFrontierBranchV1 {
    Main = 1,
    SharedResidualInput = 2,
}

impl ResidentBabFrontierBranchV1 {
    fn from_wire(tag: u8) -> Result<Self> {
        match tag {
            1 => Ok(Self::Main),
            2 => Ok(Self::SharedResidualInput),
            _ => Err(invalid(format!(
                "retained-BaB topology has unknown frontier branch tag {tag}"
            ))),
        }
    }
}

impl ResidentBabLayerBranchV1 {
    fn from_wire(tag: u8) -> Result<Self> {
        match tag {
            1 => Ok(Self::Main),
            2 => Ok(Self::Projection),
            _ => Err(invalid(format!(
                "retained-BaB topology has unknown branch tag {tag}"
            ))),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentBabNodeV1 {
    pub id: u32,
    /// Exact nonempty UTF-8 graph node name. Names are unique within a wire and
    /// are identity-bearing: an isomorphic graph with renamed nodes has a
    /// different topology transcript.
    pub name: String,
    pub kind: ResidentBabNodeKindV1,
    /// `u32::MAX` denotes the network input. Every other input must be an
    /// earlier execution-order node ID.
    pub inputs: Vec<u32>,
    /// Exact preactivation execution-order ID for a ReLU. Every non-ReLU must
    /// carry `None`; a ReLU must carry its unique unary input ID.
    pub relu_preactivation_node_id: Option<u32>,
    pub output_shape: Vec<u64>,
    pub output_values: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentBabSegmentV1 {
    pub id: u32,
    pub kind: ResidentBabSegmentKindV1,
    pub first_layer: u32,
    pub main_layer_count: u32,
    pub projection_layer_count: u32,
    pub frontier_node_id: u32,
    pub merge_node_id: Option<u32>,
    /// Exact graph-owner source class for `frontier_abs`. Together with
    /// `frontier_node_id`, the referenced node shape, and this segment ID it
    /// binds the compact row to its signed-endpoint provenance.
    pub frontier_branch: ResidentBabFrontierBranchV1,
    pub frontier_abs: ResidentBabWireRangeV1,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentBabLayerV1 {
    pub ordinal: u32,
    pub kind: ResidentBabLayerKindV1,
    pub branch: ResidentBabLayerBranchV1,
    pub segment_id: u32,
    pub node_id: u32,
    pub parameters: ResidentBabWireRangeV1,
    pub certified_errors: ResidentBabWireRangeV1,
    /// Exact in-band row layout for a ReLU: one canonical f32 variant tag,
    /// followed by six contiguous width-sized sections. Sections zero and one
    /// are always signed `pre_lower` and `pre_upper`. For tag `0x00000000`,
    /// sections two through five are `lower_slope`, `upper_slope`,
    /// `lower_intercept`, `upper_intercept`. For tag `0x3f800000`, they are
    /// `lower_pos_slope`, `cross_slope`, `upper_neg_slope`,
    /// `cross_intercept`. [`ResidentBabActivationVariantV1`] is the shared
    /// allocation-free tag parser; no other bit pattern is authorized.
    pub activation: ResidentBabWireRangeV1,
    pub beta: ResidentBabWireRangeV1,
    pub node_abs: ResidentBabWireRangeV1,
    /// Closed geometry slots. Linear uses `[out,in,has_bias,0..]`. Conv2d
    /// uses `[out_c,in_c,kh,kw,sh,sw,ph,pw,oh,ow,ih,iw,has_bias]`.
    /// Padding may be zero; `has_bias` is exactly zero or one. ReLU uses
    /// `[width,0..]`.
    pub geometry: [u32; 13],
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResidentBabFamilyLengthsV1 {
    pub parameters: u64,
    pub certified_errors: u64,
    pub activation: u64,
    pub beta: u64,
    pub abs: u64,
    pub box_values: u64,
    /// V1 deliberately admits only the canonical empty cached-lA family.
    pub cached_la: u64,
    /// V1 binds all layout in topology bytes and allocates no separate u32
    /// topology arena.
    pub topology_metadata: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentBabTopologyV1 {
    pub input_shape: Vec<u64>,
    pub output_shape: Vec<u64>,
    pub output_node_id: u32,
    pub nodes: Vec<ResidentBabNodeV1>,
    pub segments: Vec<ResidentBabSegmentV1>,
    pub layers: Vec<ResidentBabLayerV1>,
    pub relu_count: u32,
    pub families: ResidentBabFamilyLengthsV1,
}

/// Accounted result of independently decoding one v1 topology wire.
///
/// Both byte counts are absolute and include the caller-supplied baseline.
/// That baseline must already charge every caller-live input wire and source
/// object that overlaps decoding. The peak additionally includes decoder-only
/// name/coverage indexes. The retained count includes every observed nested
/// `Vec`/`String` capacity in the returned model.
#[derive(Debug, PartialEq, Eq)]
pub struct ResidentBabDecodedTopologyV1 {
    topology: ResidentBabTopologyV1,
    pub adapter_host_peak_bytes: usize,
    pub adapter_host_retained_bytes_after: usize,
}

impl ResidentBabDecodedTopologyV1 {
    #[must_use]
    pub fn topology(&self) -> &ResidentBabTopologyV1 {
        &self.topology
    }

    #[must_use]
    pub fn into_topology(self) -> ResidentBabTopologyV1 {
        self.topology
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ResidentBabEncodedTopologyV1 {
    pub bytes: Vec<u8>,
    /// Absolute simultaneous encoder peak including the caller-supplied
    /// baseline, fixed result header, observed output capacity, and temporary
    /// unique-name index. Callers must not add the baseline a second time.
    pub adapter_host_peak_bytes: usize,
    /// Absolute retained charge after return, including the caller-supplied
    /// baseline, fixed result header, and observed output capacity.
    pub adapter_host_retained_bytes: usize,
}

impl ResidentBabTopologyV1 {
    /// Decode and independently validate one exact v1 topology under a
    /// caller-owned adapter-host cap. `resident_bytes_before` must include the
    /// still-live input wire and source storage. The prospective scan performs
    /// no request-scaled allocation; every observed capacity is charged and
    /// deadline-checked before it is filled.
    pub fn decode(
        bytes: &[u8],
        max_adapter_host_bytes: usize,
        resident_bytes_before: usize,
        check: &mut dyn FnMut(&'static str) -> Result<()>,
    ) -> Result<ResidentBabDecodedTopologyV1> {
        decode_topology_v1(bytes, max_adapter_host_bytes, resident_bytes_before, check)
    }

    /// Encode one producer-owned topology after producer-side validation.
    pub(crate) fn encode(
        &self,
        max_adapter_host_bytes: usize,
        check: &mut dyn FnMut(&'static str) -> Result<()>,
    ) -> Result<ResidentBabEncodedTopologyV1> {
        self.encode_with_baseline(max_adapter_host_bytes, 0, check)
    }

    /// Encode while charging bytes already retained by the graph/phase
    /// composer. This makes the wire reserve part of the same absolute adapter
    /// host ledger instead of checking a misleading isolated allocation cap.
    pub(crate) fn encode_with_baseline(
        &self,
        max_adapter_host_bytes: usize,
        resident_bytes_before: usize,
        check: &mut dyn FnMut(&'static str) -> Result<()>,
    ) -> Result<ResidentBabEncodedTopologyV1> {
        let result_header_bytes = size_of::<ResidentBabEncodedTopologyV1>();
        let minimum_peak = resident_bytes_before
            .checked_add(result_header_bytes)
            .ok_or_else(|| invalid("retained-BaB topology encoder fixed peak overflows"))?;
        if max_adapter_host_bytes == 0 || minimum_peak > max_adapter_host_bytes {
            return Err(NyError::CpuMemoryExceeded {
                required_bytes: minimum_peak,
                budget_bytes: max_adapter_host_bytes,
                site: "retained-BaB topology wire fixed header",
            });
        }
        validate_encoder_model(self, check)?;
        let lengths = wire_lengths(self, check)?;
        let name_bucket_bytes = size_of::<&str>()
            .checked_add(HASH_BUCKET_OVERHEAD_BYTES)
            .ok_or_else(|| invalid("retained-BaB topology name-index bucket overflows"))?;
        let nominal_name_index_bytes = self
            .nodes
            .len()
            .checked_mul(name_bucket_bytes)
            .ok_or_else(|| invalid("retained-BaB topology name-index bytes overflow"))?;
        let nominal_coverage_bytes = self.nodes.len();
        let scratch_headers = size_of::<HashSet<&str>>()
            .checked_add(size_of::<Vec<u8>>())
            .ok_or_else(|| invalid("retained-BaB topology scratch headers overflow"))?;
        let nominal_peak = resident_bytes_before
            .checked_add(result_header_bytes)
            .and_then(|bytes| bytes.checked_add(lengths.total))
            .and_then(|bytes| bytes.checked_add(nominal_name_index_bytes))
            .and_then(|bytes| bytes.checked_add(nominal_coverage_bytes))
            .and_then(|bytes| bytes.checked_add(scratch_headers))
            .ok_or_else(|| invalid("retained-BaB topology encoder peak overflows"))?;
        if nominal_peak > max_adapter_host_bytes {
            return Err(NyError::CpuMemoryExceeded {
                required_bytes: nominal_peak,
                budget_bytes: max_adapter_host_bytes,
                site: "retained-BaB topology wire",
            });
        }
        let mut out = Vec::new();
        out.try_reserve_exact(lengths.total)
            .map_err(|_| invalid("retained-BaB topology encoder allocation was refused"))?;
        let output_and_nominal_scratch = resident_bytes_before
            .checked_add(result_header_bytes)
            .and_then(|bytes| bytes.checked_add(out.capacity()))
            .and_then(|bytes| bytes.checked_add(nominal_name_index_bytes))
            .and_then(|bytes| bytes.checked_add(nominal_coverage_bytes))
            .and_then(|bytes| bytes.checked_add(scratch_headers))
            .ok_or_else(|| invalid("retained-BaB topology observed output peak overflows"))?;
        if output_and_nominal_scratch > max_adapter_host_bytes {
            return Err(NyError::CpuMemoryExceeded {
                required_bytes: output_and_nominal_scratch,
                budget_bytes: max_adapter_host_bytes,
                site: "retained-BaB topology wire observed capacity",
            });
        }
        check("resident topology wire reserve")?;
        let mut names = HashSet::new();
        names
            .try_reserve(self.nodes.len())
            .map_err(|_| invalid("retained-BaB topology name-index allocation was refused"))?;
        let observed_name_index_bytes = names
            .capacity()
            .checked_mul(name_bucket_bytes)
            .ok_or_else(|| invalid("retained-BaB observed name-index bytes overflow"))?;
        let adapter_host_peak_bytes = resident_bytes_before
            .checked_add(result_header_bytes)
            .and_then(|bytes| bytes.checked_add(out.capacity()))
            .and_then(|bytes| bytes.checked_add(observed_name_index_bytes))
            .and_then(|bytes| bytes.checked_add(nominal_coverage_bytes))
            .and_then(|bytes| bytes.checked_add(scratch_headers))
            .ok_or_else(|| invalid("retained-BaB topology observed peak overflows"))?;
        if adapter_host_peak_bytes > max_adapter_host_bytes {
            return Err(NyError::CpuMemoryExceeded {
                required_bytes: adapter_host_peak_bytes,
                budget_bytes: max_adapter_host_bytes,
                site: "retained-BaB topology wire observed name-index capacity",
            });
        }
        check("resident topology name-index reserve")?;
        let mut coverage = Vec::new();
        coverage
            .try_reserve_exact(self.nodes.len())
            .map_err(|_| invalid("retained-BaB topology coverage allocation was refused"))?;
        let coverage_excess = coverage.capacity().saturating_sub(self.nodes.len());
        let adapter_host_peak_bytes = adapter_host_peak_bytes
            .checked_add(coverage_excess)
            .ok_or_else(|| invalid("retained-BaB topology coverage excess overflows"))?;
        if adapter_host_peak_bytes > max_adapter_host_bytes {
            return Err(NyError::CpuMemoryExceeded {
                required_bytes: adapter_host_peak_bytes,
                budget_bytes: max_adapter_host_bytes,
                site: "retained-BaB topology wire observed coverage capacity",
            });
        }
        check("resident topology coverage reserve")?;
        initialize_coverage_v1(
            &mut coverage,
            self.nodes.len(),
            check,
            "resident topology coverage initialization",
            "resident topology coverage initialization final",
        )?;
        validate_encoder_connectivity(self, &mut coverage, check)?;

        out.extend_from_slice(&RESIDENT_BAB_TOPOLOGY_MAGIC_V1);
        push_u32(&mut out, RESIDENT_BAB_TOPOLOGY_SCHEMA_V1);
        push_u32(&mut out, RESIDENT_BAB_TOPOLOGY_HEADER_BYTES_V1);
        push_u32(&mut out, RESIDENT_BAB_TOPOLOGY_ENDIAN_MARKER_V1);
        push_u32(&mut out, 0);
        for value in [
            lengths.total,
            lengths.shapes,
            lengths.nodes,
            lengths.segments,
            lengths.layers,
        ] {
            push_u64(
                &mut out,
                u64::try_from(value)
                    .map_err(|_| invalid("retained-BaB topology section length exceeds u64"))?,
            );
        }
        push_u32(&mut out, count_u32(self.nodes.len(), "nodes")?);
        push_u32(&mut out, count_u32(self.segments.len(), "segments")?);
        push_u32(&mut out, count_u32(self.layers.len(), "layers")?);
        push_u32(&mut out, self.relu_count);
        push_u32(&mut out, self.output_node_id);
        push_u32(&mut out, count_u32(self.input_shape.len(), "input rank")?);
        push_u32(&mut out, count_u32(self.output_shape.len(), "output rank")?);
        push_u32(&mut out, 0);
        for value in [
            self.families.parameters,
            self.families.certified_errors,
            self.families.activation,
            self.families.beta,
            self.families.abs,
            self.families.box_values,
            self.families.cached_la,
            self.families.topology_metadata,
            0,
        ] {
            push_u64(&mut out, value);
        }
        for &dim in &self.input_shape {
            push_u64(&mut out, dim);
        }
        for &dim in &self.output_shape {
            push_u64(&mut out, dim);
        }
        for (index, node) in self.nodes.iter().enumerate() {
            poll_wire(check, "resident topology node encoding", index)?;
            check("resident topology node-name encoding")?;
            if !names.insert(node.name.as_str()) {
                return Err(invalid("retained-BaB topology repeats a graph node name"));
            }
            push_u32(&mut out, node.id);
            out.push(node.kind as u8);
            out.push(count_u8(node.inputs.len(), "node inputs")?);
            out.push(count_u8(node.output_shape.len(), "node rank")?);
            out.push(0);
            push_u32(&mut out, count_u32(node.name.len(), "node name bytes")?);
            push_u32(
                &mut out,
                node.relu_preactivation_node_id
                    .unwrap_or(RESIDENT_BAB_NETWORK_INPUT_ID_V1),
            );
            push_u32(&mut out, 0);
            push_u32(&mut out, 0);
            push_u64(&mut out, node.output_values);
            out.extend_from_slice(node.name.as_bytes());
            for &input in &node.inputs {
                push_u32(&mut out, input);
            }
            for &dim in &node.output_shape {
                push_u64(&mut out, dim);
            }
        }
        check("resident topology node encoding final")?;
        for (index, segment) in self.segments.iter().enumerate() {
            poll_wire(check, "resident topology segment encoding", index)?;
            push_u32(&mut out, segment.id);
            out.push(segment.kind as u8);
            out.push(segment.frontier_branch as u8);
            out.extend_from_slice(&[0; 2]);
            push_u32(&mut out, segment.first_layer);
            push_u32(&mut out, segment.main_layer_count);
            push_u32(&mut out, segment.projection_layer_count);
            push_u32(&mut out, segment.frontier_node_id);
            push_u32(
                &mut out,
                segment
                    .merge_node_id
                    .unwrap_or(RESIDENT_BAB_NETWORK_INPUT_ID_V1),
            );
            push_u32(&mut out, 0);
            push_range(&mut out, segment.frontier_abs);
        }
        check("resident topology segment encoding final")?;
        for (index, layer) in self.layers.iter().enumerate() {
            poll_wire(check, "resident topology layer encoding", index)?;
            push_u32(&mut out, layer.ordinal);
            out.push(layer.kind as u8);
            out.push(layer.branch as u8);
            push_u16(&mut out, 0);
            push_u32(&mut out, layer.segment_id);
            push_u32(&mut out, layer.node_id);
            push_u32(&mut out, 0);
            push_range(&mut out, layer.parameters);
            push_range(&mut out, layer.certified_errors);
            push_range(&mut out, layer.activation);
            push_range(&mut out, layer.beta);
            push_range(&mut out, layer.node_abs);
            for value in layer.geometry {
                push_u32(&mut out, value);
            }
        }
        check("resident topology layer encoding final")?;
        if out.len() != lengths.total {
            return Err(invalid("retained-BaB topology encoder length drift"));
        }
        let adapter_host_retained_bytes = resident_bytes_before
            .checked_add(result_header_bytes)
            .and_then(|bytes| bytes.checked_add(out.capacity()))
            .ok_or_else(|| invalid("retained-BaB topology retained bytes overflow"))?;
        Ok(ResidentBabEncodedTopologyV1 {
            bytes: out,
            adapter_host_peak_bytes,
            adapter_host_retained_bytes,
        })
    }
}

fn count_u32(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| invalid(format!("retained-BaB {label} count does not fit u32")))
}

fn count_u8(value: usize, label: &str) -> Result<u8> {
    u8::try_from(value).map_err(|_| invalid(format!("retained-BaB {label} count does not fit u8")))
}

fn checked_shape(shape: &[u64], label: &str) -> Result<u64> {
    if shape.is_empty() || shape.len() > MAX_RANK || shape.contains(&0) {
        return Err(invalid(format!(
            "retained-BaB {label} shape must have rank 1..={MAX_RANK} and nonzero dimensions"
        )));
    }
    let values = shape.iter().try_fold(1u64, |product, &dim| {
        product
            .checked_mul(dim)
            .ok_or_else(|| invalid(format!("retained-BaB {label} shape product overflows u64")))
    })?;
    if values > GPU_BAB_BOUND_MAX_ARENA_VALUES as u64 {
        return Err(invalid(format!(
            "retained-BaB {label} shape exceeds the core arena-value cap"
        )));
    }
    Ok(values)
}

fn validate_encoder_model(
    topology: &ResidentBabTopologyV1,
    check: &mut dyn FnMut(&'static str) -> Result<()>,
) -> Result<()> {
    if topology.nodes.is_empty()
        || topology.nodes.len() > MAX_RECORDS
        || topology.segments.is_empty()
        || topology.segments.len() > MAX_RECORDS
        || topology.layers.is_empty()
        || topology.layers.len() > MAX_RECORDS
    {
        return Err(invalid(
            "retained-BaB topology record counts are empty or exceed the v1 cap",
        ));
    }
    let input_values = checked_shape(&topology.input_shape, "input")?;
    let output_values = checked_shape(&topology.output_shape, "output")?;
    if topology.families.parameters == 0
        || topology.families.box_values != input_values
        || topology.families.cached_la != 0
        || topology.families.topology_metadata != 0
        || !families_fit_core_cap(topology.families)
    {
        return Err(invalid(
            "retained-BaB v1 box/cached-lA/topology-metadata family lengths are noncanonical",
        ));
    }
    validate_encoder_nodes(topology, output_values, check)?;
    validate_encoder_layout(topology, check)
}

/// Allocation-free bounded-work precondition for composing dynamic resident
/// operands from a borrowed topology. Full wire encoding/decoding remains the
/// semantic topology validation boundary; this check only ensures that a bare
/// safe Rust value cannot make the composer exceed v1/core work bounds before
/// that sealed artifact exists.
pub(crate) fn validate_composition_caps_v1(
    topology: &ResidentBabTopologyV1,
    history_word_count: usize,
    check: &mut dyn FnMut(&'static str) -> Result<()>,
) -> Result<()> {
    if topology.nodes.is_empty()
        || topology.nodes.len() > MAX_RECORDS
        || topology.segments.is_empty()
        || topology.segments.len() > MAX_RECORDS
        || topology.layers.is_empty()
        || topology.layers.len() > MAX_RECORDS
        || usize::try_from(topology.relu_count).map_or(true, |count| count > MAX_RECORDS)
        || history_word_count > GPU_BAB_BOUND_MAX_SPLIT_HISTORY_WORDS
    {
        return Err(invalid(
            "retained-BaB composition record/history counts exceed v1/core bounds",
        ));
    }
    let input_values = checked_shape(&topology.input_shape, "composition input")?;
    let _output_values = checked_shape(&topology.output_shape, "composition output")?;
    if topology.families.parameters == 0
        || topology.families.box_values != input_values
        || topology.families.cached_la != 0
        || topology.families.topology_metadata != 0
        || !families_fit_core_cap(topology.families)
    {
        return Err(invalid(
            "retained-BaB composition families exceed v1/core bounds",
        ));
    }

    let mut total_name_bytes = 0usize;
    for (index, node) in topology.nodes.iter().enumerate() {
        poll_wire(check, "resident composition node cap validation", index)?;
        if node.name.is_empty()
            || node.name.len() > MAX_NODE_NAME_BYTES
            || node.inputs.len() > MAX_NODE_INPUTS
            || checked_shape(&node.output_shape, "composition node output")? != node.output_values
        {
            return Err(invalid(format!(
                "retained-BaB composition node {index} exceeds v1/core bounds"
            )));
        }
        total_name_bytes = total_name_bytes
            .checked_add(node.name.len())
            .ok_or_else(|| invalid("retained-BaB composition name bytes overflow usize"))?;
        if total_name_bytes > MAX_TOPOLOGY_BYTES {
            return Err(invalid(
                "retained-BaB composition graph names exceed the topology byte cap",
            ));
        }
    }
    check("resident composition node cap validation final")?;
    for index in 0..topology.segments.len() {
        poll_wire(check, "resident composition segment cap validation", index)?;
    }
    check("resident composition segment cap validation final")?;
    for index in 0..topology.layers.len() {
        poll_wire(check, "resident composition layer cap validation", index)?;
    }
    check("resident composition layer cap validation final")
}

fn validate_encoder_nodes(
    topology: &ResidentBabTopologyV1,
    output_values: u64,
    check: &mut dyn FnMut(&'static str) -> Result<()>,
) -> Result<()> {
    for (index, node) in topology.nodes.iter().enumerate() {
        poll_wire(check, "resident topology node validation", index)?;
        if usize::try_from(node.id).ok() != Some(index)
            || node.name.is_empty()
            || node.name.len() > MAX_NODE_NAME_BYTES
            || node.inputs.len() != node.kind.input_count()
            || checked_shape(&node.output_shape, "node output")? != node.output_values
        {
            return Err(invalid(format!(
                "retained-BaB encoder node {index} is noncanonical"
            )));
        }
        for &input in &node.inputs {
            if input != RESIDENT_BAB_NETWORK_INPUT_ID_V1
                && usize::try_from(input).map_or(true, |id| id >= index)
            {
                return Err(invalid(format!(
                    "retained-BaB encoder node {index} input is not earlier in execution order"
                )));
            }
        }
        let expected_preactivation = (node.kind == ResidentBabNodeKindV1::Relu)
            .then(|| node.inputs.first().copied())
            .flatten();
        if node.relu_preactivation_node_id != expected_preactivation {
            return Err(invalid(format!(
                "retained-BaB encoder node {index} has noncanonical ReLU preactivation evidence"
            )));
        }
    }
    check("resident topology node validation final")?;
    let output = topology
        .nodes
        .get(usize::try_from(topology.output_node_id).unwrap_or(usize::MAX))
        .ok_or_else(|| invalid("retained-BaB output node ID is out of range"))?;
    if output.output_shape != topology.output_shape || output.output_values != output_values {
        return Err(invalid(
            "retained-BaB output shape does not match the selected output node",
        ));
    }
    Ok(())
}

fn validate_encoder_layout(
    topology: &ResidentBabTopologyV1,
    check: &mut dyn FnMut(&'static str) -> Result<()>,
) -> Result<()> {
    let mut next_layer = 0u64;
    let mut next_frontier_abs = 0u64;
    for (index, segment) in topology.segments.iter().enumerate() {
        poll_wire(check, "resident topology segment validation", index)?;
        let layer_count = u64::from(segment.main_layer_count)
            .checked_add(u64::from(segment.projection_layer_count))
            .ok_or_else(|| invalid("retained-BaB segment layer count overflows"))?;
        if usize::try_from(segment.id).ok() != Some(index)
            || u64::from(segment.first_layer) != next_layer
            || segment.frontier_abs.start != next_frontier_abs
            || segment.frontier_abs.len == 0
            || match segment.kind {
                ResidentBabSegmentKindV1::Chain => {
                    segment.projection_layer_count != 0
                        || segment.merge_node_id.is_some()
                        || segment.frontier_branch != ResidentBabFrontierBranchV1::Main
                }
                ResidentBabSegmentKindV1::Residual => {
                    segment.projection_layer_count != 0
                        || segment.merge_node_id.is_none()
                        || segment.frontier_branch
                            != ResidentBabFrontierBranchV1::SharedResidualInput
                }
                ResidentBabSegmentKindV1::ResidualProjection => {
                    segment.projection_layer_count == 0
                        || segment.merge_node_id.is_none()
                        || segment.frontier_branch
                            != ResidentBabFrontierBranchV1::SharedResidualInput
                }
            }
        {
            return Err(invalid(format!(
                "retained-BaB encoder segment {index} is noncanonical"
            )));
        }
        if segment.main_layer_count == 0
            || (segment.frontier_node_id != RESIDENT_BAB_NETWORK_INPUT_ID_V1
                && usize::try_from(segment.frontier_node_id)
                    .map_or(true, |id| id >= topology.nodes.len()))
            || segment
                .merge_node_id
                .is_some_and(|id| usize::try_from(id).map_or(true, |i| i >= topology.nodes.len()))
        {
            return Err(invalid(format!(
                "retained-BaB encoder segment {index} has invalid frontier/merge/layer fields"
            )));
        }
        next_layer = next_layer
            .checked_add(layer_count)
            .ok_or_else(|| invalid("retained-BaB layer partition overflows"))?;
        next_frontier_abs = segment.frontier_abs.end("frontier abs")?;
    }
    check("resident topology segment validation final")?;
    if usize::try_from(next_layer).ok() != Some(topology.layers.len()) {
        return Err(invalid(
            "retained-BaB segments do not exactly partition the layer records",
        ));
    }

    let mut parameters = 0u64;
    let mut errors = 0u64;
    let mut activation = 0u64;
    let mut beta = 0u64;
    let mut node_abs = next_frontier_abs;
    let mut relus = 0u32;
    for (index, layer) in topology.layers.iter().enumerate() {
        poll_wire(check, "resident topology layer validation", index)?;
        if usize::try_from(layer.ordinal).ok() != Some(index)
            || usize::try_from(layer.segment_id).map_or(true, |id| id >= topology.segments.len())
            || usize::try_from(layer.node_id).map_or(true, |id| id >= topology.nodes.len())
        {
            return Err(invalid(format!(
                "retained-BaB encoder layer {index} has invalid IDs"
            )));
        }
        let segment = &topology.segments[layer.segment_id as usize];
        let local = u32::try_from(index)
            .ok()
            .and_then(|ordinal| ordinal.checked_sub(segment.first_layer))
            .ok_or_else(|| invalid("retained-BaB layer lies before its segment"))?;
        let expected_branch = if local < segment.main_layer_count {
            ResidentBabLayerBranchV1::Main
        } else {
            ResidentBabLayerBranchV1::Projection
        };
        let segment_total = segment
            .main_layer_count
            .checked_add(segment.projection_layer_count)
            .ok_or_else(|| invalid("retained-BaB encoder segment count overflows"))?;
        if local >= segment_total || layer.branch != expected_branch {
            return Err(invalid(format!(
                "retained-BaB encoder layer {index} branch is noncanonical"
            )));
        }
        let node = &topology.nodes[layer.node_id as usize];
        match layer.kind {
            ResidentBabLayerKindV1::Linear | ResidentBabLayerKindV1::Conv2d => {
                let expected_node = if layer.kind == ResidentBabLayerKindV1::Linear {
                    ResidentBabNodeKindV1::Linear
                } else {
                    ResidentBabNodeKindV1::Conv2d
                };
                if node.kind != expected_node
                    || layer.parameters.start != parameters
                    || layer.parameters.len == 0
                    || layer.certified_errors.start != errors
                    || layer.certified_errors.len != 2
                    || layer.activation.len != 0
                    || layer.beta.len != 0
                    || layer.node_abs.len != 0
                    || layer.activation.start != activation
                    || layer.beta.start != beta
                    || layer.node_abs.start != node_abs
                {
                    return Err(invalid(format!(
                        "retained-BaB encoder static layer {index} ranges are noncanonical"
                    )));
                }
                validate_static_geometry(layer)?;
                parameters = layer.parameters.end("parameters")?;
                errors = layer.certified_errors.end("certified errors")?;
            }
            ResidentBabLayerKindV1::Relu => {
                let width = u64::from(layer.geometry[0]);
                if node.kind != ResidentBabNodeKindV1::Relu
                    || width == 0
                    || layer.geometry[1..].iter().any(|&value| value != 0)
                    || layer.parameters.len != 0
                    || layer.certified_errors.len != 0
                    || layer.parameters.start != parameters
                    || layer.certified_errors.start != errors
                    || layer.activation.start != activation
                    || ResidentBabActivationSectionsV1::from_row(layer.activation, width).is_err()
                    || layer.beta.start != beta
                    || layer.beta.len != width
                    || layer.node_abs.start != node_abs
                    || layer.node_abs.len != width
                {
                    return Err(invalid(format!(
                        "retained-BaB encoder ReLU layer {index} ranges are noncanonical"
                    )));
                }
                activation = layer.activation.end("activation")?;
                beta = layer.beta.end("beta")?;
                node_abs = layer.node_abs.end("node abs")?;
                relus = relus
                    .checked_add(1)
                    .ok_or_else(|| invalid("retained-BaB ReLU count overflows"))?;
            }
        }
    }
    check("resident topology layer validation final")?;
    if parameters != topology.families.parameters
        || errors != topology.families.certified_errors
        || activation != topology.families.activation
        || beta != topology.families.beta
        || node_abs != topology.families.abs
        || relus != topology.relu_count
    {
        return Err(invalid(
            "retained-BaB family lengths do not equal their canonical layer partitions",
        ));
    }
    Ok(())
}

fn encoder_source_shape(topology: &ResidentBabTopologyV1, source: u32) -> Result<&[u64]> {
    if source == RESIDENT_BAB_NETWORK_INPUT_ID_V1 {
        return Ok(&topology.input_shape);
    }
    topology
        .nodes
        .get(
            usize::try_from(source)
                .map_err(|_| invalid("retained-BaB encoder source ID does not fit usize"))?,
        )
        .map(|node| node.output_shape.as_slice())
        .ok_or_else(|| invalid("retained-BaB encoder source ID is outside topology"))
}

fn encoder_mark_node(coverage: &mut [u8], node_id: u32, label: &str) -> Result<()> {
    let slot = coverage
        .get_mut(
            usize::try_from(node_id)
                .map_err(|_| invalid(format!("retained-BaB encoder {label} ID overflows")))?,
        )
        .ok_or_else(|| {
            invalid(format!(
                "retained-BaB encoder {label} ID is outside topology"
            ))
        })?;
    if *slot != 0 {
        return Err(invalid(format!(
            "retained-BaB encoder {label} is covered more than once"
        )));
    }
    *slot = 1;
    Ok(())
}

fn encoder_coverage_is_exact(
    coverage: &[u8],
    expected: u8,
    check: &mut dyn FnMut(&'static str) -> Result<()>,
) -> Result<bool> {
    for (index, &mark) in coverage.iter().enumerate() {
        poll_wire(check, "resident encoder coverage audit", index)?;
        if mark != expected {
            return Ok(false);
        }
    }
    check("resident encoder coverage audit final")?;
    Ok(true)
}

fn validate_encoder_node_shapes(
    topology: &ResidentBabTopologyV1,
    check: &mut dyn FnMut(&'static str) -> Result<()>,
) -> Result<()> {
    if usize::try_from(topology.output_node_id).ok() != topology.nodes.len().checked_sub(1) {
        return Err(invalid(
            "retained-BaB encoder output must be the final execution-order node",
        ));
    }
    for (index, node) in topology.nodes.iter().enumerate() {
        poll_wire(check, "resident encoder graph-shape validation", index)?;
        match node.kind {
            ResidentBabNodeKindV1::Relu => {
                let source = encoder_source_shape(topology, node.inputs[0])?;
                if node.output_shape != source {
                    return Err(invalid(format!(
                        "retained-BaB encoder ReLU node {index} changes shape"
                    )));
                }
            }
            ResidentBabNodeKindV1::Flatten | ResidentBabNodeKindV1::Reshape => {
                let source = encoder_source_shape(topology, node.inputs[0])?;
                if checked_shape(source, "structural source")? != node.output_values {
                    return Err(invalid(format!(
                        "retained-BaB encoder structural node {index} changes element count"
                    )));
                }
            }
            ResidentBabNodeKindV1::Add => {
                let left = encoder_source_shape(topology, node.inputs[0])?;
                let right = encoder_source_shape(topology, node.inputs[1])?;
                if left != right || left != node.output_shape {
                    return Err(invalid(format!(
                        "retained-BaB encoder Add node {index} is not exact equal-shape addition"
                    )));
                }
            }
            ResidentBabNodeKindV1::Linear | ResidentBabNodeKindV1::Conv2d => {}
        }
    }
    check("resident encoder graph-shape validation final")
}

fn validate_encoder_layer_shape(
    topology: &ResidentBabTopologyV1,
    layer: &ResidentBabLayerV1,
) -> Result<()> {
    let node = topology
        .nodes
        .get(layer.node_id as usize)
        .ok_or_else(|| invalid("retained-BaB encoder layer node is outside topology"))?;
    let source = encoder_source_shape(topology, node.inputs[0])?;
    match layer.kind {
        ResidentBabLayerKindV1::Linear => {
            validate_static_geometry(layer)?;
            let input = [u64::from(layer.geometry[1])];
            let output = [u64::from(layer.geometry[0])];
            if source != input || node.output_shape != output {
                return Err(invalid(
                    "retained-BaB encoder V1 Linear is not exact rank-1 [in] -> [out]",
                ));
            }
        }
        ResidentBabLayerKindV1::Conv2d => {
            validate_static_geometry(layer)?;
            let input = [
                u64::from(layer.geometry[1]),
                u64::from(layer.geometry[10]),
                u64::from(layer.geometry[11]),
            ];
            let output = [
                u64::from(layer.geometry[0]),
                u64::from(layer.geometry[8]),
                u64::from(layer.geometry[9]),
            ];
            if source != input || node.output_shape != output {
                return Err(invalid(
                    "retained-BaB encoder Conv2d geometry disagrees with [C,H,W] graph shapes",
                ));
            }
        }
        ResidentBabLayerKindV1::Relu => {
            let width = u64::from(layer.geometry[0]);
            if node.output_shape != source
                || node.output_values != width
                || checked_shape(source, "ReLU source")? != width
                || node.relu_preactivation_node_id != Some(node.inputs[0])
            {
                return Err(invalid(
                    "retained-BaB encoder ReLU geometry disagrees with preactivation/output shape",
                ));
            }
        }
    }
    Ok(())
}

fn trace_encoder_branch(
    topology: &ResidentBabTopologyV1,
    layers: &[ResidentBabLayerV1],
    mut cursor: u32,
    frontier: u32,
    coverage: &mut [u8],
    check: &mut dyn FnMut(&'static str) -> Result<()>,
) -> Result<()> {
    let mut structural_steps = 0usize;
    for (position, layer) in layers.iter().enumerate() {
        poll_wire(check, "resident encoder branch connectivity", position)?;
        while cursor != frontier {
            let node = topology
                .nodes
                .get(cursor as usize)
                .ok_or_else(|| invalid("retained-BaB encoder branch ended before a layer"))?;
            if !matches!(
                node.kind,
                ResidentBabNodeKindV1::Flatten | ResidentBabNodeKindV1::Reshape
            ) {
                break;
            }
            poll_wire(
                check,
                "resident encoder structural connectivity",
                structural_steps,
            )?;
            structural_steps += 1;
            encoder_mark_node(coverage, cursor, "structural node")?;
            cursor = node.inputs[0];
        }
        if cursor == frontier || cursor != layer.node_id {
            return Err(invalid(
                "retained-BaB encoder layer order does not follow exact unary ancestry",
            ));
        }
        validate_encoder_layer_shape(topology, layer)?;
        encoder_mark_node(coverage, cursor, "executable layer node")?;
        cursor = topology.nodes[cursor as usize].inputs[0];
    }
    while cursor != frontier {
        let node = topology
            .nodes
            .get(cursor as usize)
            .ok_or_else(|| invalid("retained-BaB encoder branch does not reach its frontier"))?;
        if !matches!(
            node.kind,
            ResidentBabNodeKindV1::Flatten | ResidentBabNodeKindV1::Reshape
        ) {
            return Err(invalid(
                "retained-BaB encoder branch has an uncovered executable or merge node",
            ));
        }
        poll_wire(
            check,
            "resident encoder structural connectivity",
            structural_steps,
        )?;
        structural_steps += 1;
        encoder_mark_node(coverage, cursor, "structural node")?;
        cursor = node.inputs[0];
    }
    check("resident encoder branch connectivity final")
}

fn drain_encoder_structural_seam(
    topology: &ResidentBabTopologyV1,
    mut cursor: u32,
    target: u32,
    coverage: &mut [u8],
    check: &mut dyn FnMut(&'static str) -> Result<()>,
) -> Result<u32> {
    let mut steps = 0usize;
    while cursor != target {
        let Some(node) = usize::try_from(cursor)
            .ok()
            .and_then(|index| topology.nodes.get(index))
        else {
            break;
        };
        if !matches!(
            node.kind,
            ResidentBabNodeKindV1::Flatten | ResidentBabNodeKindV1::Reshape
        ) {
            break;
        }
        poll_wire(check, "resident encoder structural seam", steps)?;
        steps += 1;
        encoder_mark_node(coverage, cursor, "structural seam node")?;
        cursor = node.inputs[0];
    }
    check("resident encoder structural seam final")?;
    Ok(cursor)
}

fn validate_encoder_connectivity(
    topology: &ResidentBabTopologyV1,
    coverage: &mut [u8],
    check: &mut dyn FnMut(&'static str) -> Result<()>,
) -> Result<()> {
    if coverage.len() != topology.nodes.len() || !encoder_coverage_is_exact(coverage, 0, check)? {
        return Err(invalid(
            "retained-BaB encoder coverage storage is not an exact zeroed node table",
        ));
    }
    validate_encoder_node_shapes(topology, check)?;
    let mut cursor = topology.output_node_id;
    for (segment_index, segment) in topology.segments.iter().enumerate() {
        poll_wire(
            check,
            "resident encoder segment connectivity",
            segment_index,
        )?;
        let first = usize::try_from(segment.first_layer)
            .map_err(|_| invalid("retained-BaB encoder segment start does not fit usize"))?;
        let main_end = first
            .checked_add(segment.main_layer_count as usize)
            .ok_or_else(|| invalid("retained-BaB encoder main branch end overflows"))?;
        let projection_end = main_end
            .checked_add(segment.projection_layer_count as usize)
            .ok_or_else(|| invalid("retained-BaB encoder projection branch end overflows"))?;
        let main = topology
            .layers
            .get(first..main_end)
            .ok_or_else(|| invalid("retained-BaB encoder main branch range is invalid"))?;
        let projection = topology
            .layers
            .get(main_end..projection_end)
            .ok_or_else(|| invalid("retained-BaB encoder projection branch range is invalid"))?;
        let frontier_width = checked_shape(
            encoder_source_shape(topology, segment.frontier_node_id)?,
            "segment frontier",
        )?;
        if segment.frontier_abs.len != frontier_width {
            return Err(invalid(
                "retained-BaB encoder frontier Abs length disagrees with its source shape",
            ));
        }
        if segment.kind == ResidentBabSegmentKindV1::Chain
            && segment.frontier_node_id != RESIDENT_BAB_NETWORK_INPUT_ID_V1
        {
            let next = topology.segments.get(segment_index + 1).ok_or_else(|| {
                invalid("retained-BaB encoder Chain has a non-input terminal frontier")
            })?;
            if !matches!(
                next.kind,
                ResidentBabSegmentKindV1::Residual | ResidentBabSegmentKindV1::ResidualProjection
            ) || next.merge_node_id != Some(segment.frontier_node_id)
            {
                return Err(invalid(
                    "retained-BaB encoder Chain frontier is not the next residual merge",
                ));
            }
        }
        if matches!(
            segment.kind,
            ResidentBabSegmentKindV1::Residual | ResidentBabSegmentKindV1::ResidualProjection
        ) {
            let merge_id = segment
                .merge_node_id
                .ok_or_else(|| invalid("retained-BaB encoder residual has no merge"))?;
            cursor = drain_encoder_structural_seam(topology, cursor, merge_id, coverage, check)?;
        }
        match segment.kind {
            ResidentBabSegmentKindV1::Chain => {
                trace_encoder_branch(
                    topology,
                    main,
                    cursor,
                    segment.frontier_node_id,
                    coverage,
                    check,
                )?;
            }
            ResidentBabSegmentKindV1::Residual => {
                let merge_id = segment
                    .merge_node_id
                    .ok_or_else(|| invalid("retained-BaB encoder residual has no merge"))?;
                if cursor != merge_id {
                    return Err(invalid(
                        "retained-BaB encoder residual merge does not equal the fold cursor",
                    ));
                }
                let merge = topology
                    .nodes
                    .get(merge_id as usize)
                    .ok_or_else(|| invalid("retained-BaB encoder residual merge is invalid"))?;
                if merge.kind != ResidentBabNodeKindV1::Add {
                    return Err(invalid("retained-BaB encoder residual merge is not Add"));
                }
                encoder_mark_node(coverage, merge_id, "residual Add merge")?;
                let left_is_frontier = merge.inputs[0] == segment.frontier_node_id;
                let right_is_frontier = merge.inputs[1] == segment.frontier_node_id;
                if left_is_frontier == right_is_frontier {
                    return Err(invalid(
                        "retained-BaB encoder identity residual must have exactly one frontier input",
                    ));
                }
                let main_start = if left_is_frontier {
                    merge.inputs[1]
                } else {
                    merge.inputs[0]
                };
                trace_encoder_branch(
                    topology,
                    main,
                    main_start,
                    segment.frontier_node_id,
                    coverage,
                    check,
                )?;
            }
            ResidentBabSegmentKindV1::ResidualProjection => {
                let merge_id = segment.merge_node_id.ok_or_else(|| {
                    invalid("retained-BaB encoder projection residual has no merge")
                })?;
                if cursor != merge_id {
                    return Err(invalid(
                        "retained-BaB encoder projection merge does not equal the fold cursor",
                    ));
                }
                let merge = topology
                    .nodes
                    .get(merge_id as usize)
                    .ok_or_else(|| invalid("retained-BaB encoder projection merge is invalid"))?;
                if merge.kind != ResidentBabNodeKindV1::Add {
                    return Err(invalid("retained-BaB encoder projection merge is not Add"));
                }
                encoder_mark_node(coverage, merge_id, "residual Add merge")?;
                trace_encoder_branch(
                    topology,
                    main,
                    merge.inputs[0],
                    segment.frontier_node_id,
                    coverage,
                    check,
                )?;
                trace_encoder_branch(
                    topology,
                    projection,
                    merge.inputs[1],
                    segment.frontier_node_id,
                    coverage,
                    check,
                )?;
            }
        }
        cursor = segment.frontier_node_id;
    }
    cursor = drain_encoder_structural_seam(
        topology,
        cursor,
        RESIDENT_BAB_NETWORK_INPUT_ID_V1,
        coverage,
        check,
    )?;
    check("resident encoder segment connectivity final")?;
    if cursor != RESIDENT_BAB_NETWORK_INPUT_ID_V1 || !encoder_coverage_is_exact(coverage, 1, check)?
    {
        return Err(invalid(
            "retained-BaB encoder segments do not cover every exec-order node exactly once to network input",
        ));
    }
    Ok(())
}

fn validate_static_geometry(layer: &ResidentBabLayerV1) -> Result<()> {
    match layer.kind {
        ResidentBabLayerKindV1::Linear => {
            let [out, input, has_bias, rest @ ..] = &layer.geometry;
            let expected_parameters = u64::from(*out)
                .checked_mul(u64::from(*input))
                .and_then(|weights| {
                    u64::from(*out)
                        .checked_mul(u64::from(*has_bias))
                        .and_then(|bias| weights.checked_add(bias))
                })
                .ok_or_else(|| invalid("retained-BaB Linear parameter length overflows"))?;
            if *out == 0
                || *input == 0
                || *has_bias > 1
                || rest.iter().any(|&value| value != 0)
                || layer.parameters.len != expected_parameters
            {
                return Err(invalid("retained-BaB Linear geometry is noncanonical"));
            }
        }
        ResidentBabLayerKindV1::Conv2d => {
            let [out_c, in_c, kh, kw, sh, sw, _ph, _pw, oh, ow, ih, iw, has_bias] = layer.geometry;
            if [out_c, in_c, kh, kw, sh, sw, oh, ow, ih, iw].contains(&0) || has_bias > 1 {
                return Err(invalid(
                    "retained-BaB Conv2d geometry has a zero required field or invalid bias tag",
                ));
            }
            let expected_oh = u64::from(ih)
                .checked_add(
                    u64::from(_ph)
                        .checked_mul(2)
                        .ok_or_else(|| invalid("retained-BaB Conv2d padded height overflows"))?,
                )
                .and_then(|padded| padded.checked_sub(u64::from(kh)))
                .map(|span| span / u64::from(sh) + 1)
                .ok_or_else(|| invalid("retained-BaB Conv2d output height is invalid"))?;
            let expected_ow = u64::from(iw)
                .checked_add(
                    u64::from(_pw)
                        .checked_mul(2)
                        .ok_or_else(|| invalid("retained-BaB Conv2d padded width overflows"))?,
                )
                .and_then(|padded| padded.checked_sub(u64::from(kw)))
                .map(|span| span / u64::from(sw) + 1)
                .ok_or_else(|| invalid("retained-BaB Conv2d output width is invalid"))?;
            if expected_oh != u64::from(oh) || expected_ow != u64::from(ow) {
                return Err(invalid(
                    "retained-BaB Conv2d output geometry does not match its checked formula",
                ));
            }
            let weights = [out_c, in_c, kh, kw]
                .iter()
                .try_fold(1u64, |product, &value| {
                    product.checked_mul(u64::from(value))
                })
                .ok_or_else(|| invalid("retained-BaB Conv2d weight length overflows"))?;
            let bias = u64::from(out_c)
                .checked_mul(u64::from(oh))
                .and_then(|value| value.checked_mul(u64::from(ow)))
                .and_then(|value| value.checked_mul(u64::from(has_bias)))
                .ok_or_else(|| invalid("retained-BaB Conv2d bias length overflows"))?;
            let expected_parameters = weights
                .checked_add(bias)
                .ok_or_else(|| invalid("retained-BaB Conv2d parameter length overflows"))?;
            if layer.parameters.len != expected_parameters {
                return Err(invalid(
                    "retained-BaB Conv2d parameter range does not match geometry",
                ));
            }
        }
        ResidentBabLayerKindV1::Relu => unreachable!("caller excludes ReLU"),
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct ResidentBabWireLengthsV1 {
    shapes: usize,
    nodes: usize,
    segments: usize,
    layers: usize,
    total: usize,
}

fn wire_lengths_preflight(
    topology: &ResidentBabTopologyV1,
    check: &mut dyn FnMut(&'static str) -> Result<()>,
) -> Result<Option<ResidentBabWireLengthsV1>> {
    let Some(shapes) = topology
        .input_shape
        .len()
        .checked_add(topology.output_shape.len())
        .and_then(|dims| dims.checked_mul(8))
    else {
        return Ok(None);
    };
    let mut nodes = 0usize;
    for (index, node) in topology.nodes.iter().enumerate() {
        poll_wire(check, "resident topology length validation", index)?;
        let Some(next) = nodes
            .checked_add(NODE_FIXED_BYTES)
            .and_then(|value| value.checked_add(node.name.len()))
            .and_then(|value| value.checked_add(node.inputs.len().checked_mul(4)?))
            .and_then(|value| value.checked_add(node.output_shape.len().checked_mul(8)?))
        else {
            return Ok(None);
        };
        nodes = next;
    }
    check("resident topology length validation final")?;
    let Some(segments) = topology.segments.len().checked_mul(SEGMENT_BYTES) else {
        return Ok(None);
    };
    let Some(layers) = topology.layers.len().checked_mul(LAYER_BYTES) else {
        return Ok(None);
    };
    let Ok(header) = usize::try_from(RESIDENT_BAB_TOPOLOGY_HEADER_BYTES_V1) else {
        return Ok(None);
    };
    let Some(total) = header
        .checked_add(shapes)
        .and_then(|value| value.checked_add(nodes))
        .and_then(|value| value.checked_add(segments))
        .and_then(|value| value.checked_add(layers))
    else {
        return Ok(None);
    };
    if total > MAX_TOPOLOGY_BYTES {
        return Ok(None);
    }
    Ok(Some(ResidentBabWireLengthsV1 {
        shapes,
        nodes,
        segments,
        layers,
        total,
    }))
}

pub(crate) fn topology_wire_length_preflight_v1(
    topology: &ResidentBabTopologyV1,
    check: &mut dyn FnMut(&'static str) -> Result<()>,
) -> Result<ResidentBabTopologyWireLengthPreflightV1> {
    Ok(match wire_lengths_preflight(topology, check)? {
        Some(lengths) => ResidentBabTopologyWireLengthPreflightV1::Encodable {
            encoded_bytes: lengths.total,
        },
        None => ResidentBabTopologyWireLengthPreflightV1::ExceedsV1ByteCap,
    })
}

fn wire_lengths(
    topology: &ResidentBabTopologyV1,
    check: &mut dyn FnMut(&'static str) -> Result<()>,
) -> Result<ResidentBabWireLengthsV1> {
    wire_lengths_preflight(topology, check)?
        .ok_or_else(|| invalid("retained-BaB topology exceeds the v1 byte cap"))
}

fn encoded_len(
    topology: &ResidentBabTopologyV1,
    check: &mut dyn FnMut(&'static str) -> Result<()>,
) -> Result<usize> {
    Ok(wire_lengths(topology, check)?.total)
}

#[derive(Clone, Copy)]
struct ResidentBabDecodePlanV1 {
    node_count: usize,
    segment_count: usize,
    layer_count: usize,
    input_rank: usize,
    output_rank: usize,
    nominal_host_bytes: usize,
}

struct ResidentBabDecodeBudgetV1 {
    limit_bytes: usize,
    peak_bytes: usize,
}

impl ResidentBabDecodeBudgetV1 {
    fn begin(
        limit_bytes: usize,
        resident_bytes_before: usize,
        nominal_extra: usize,
    ) -> Result<Self> {
        let required = resident_bytes_before
            .checked_add(nominal_extra)
            .ok_or_else(|| invalid("retained-BaB decoder host charge overflows usize"))?;
        if limit_bytes == 0 || required > limit_bytes {
            return Err(NyError::CpuMemoryExceeded {
                required_bytes: required,
                budget_bytes: limit_bytes,
                site: "retained-BaB topology decode",
            });
        }
        Ok(Self {
            limit_bytes,
            peak_bytes: required,
        })
    }

    fn charge_excess(&mut self, bytes: usize) -> Result<()> {
        let required = self
            .peak_bytes
            .checked_add(bytes)
            .ok_or_else(|| invalid("retained-BaB decoder observed charge overflows usize"))?;
        if required > self.limit_bytes {
            return Err(NyError::CpuMemoryExceeded {
                required_bytes: required,
                budget_bytes: self.limit_bytes,
                site: "retained-BaB topology decode observed capacity",
            });
        }
        self.peak_bytes = required;
        Ok(())
    }

    fn reserve_vec<T>(
        &mut self,
        values: &mut Vec<T>,
        nominal_count: usize,
        label: &'static str,
        check: &mut dyn FnMut(&'static str) -> Result<()>,
    ) -> Result<()> {
        values
            .try_reserve_exact(nominal_count)
            .map_err(|_| invalid(format!("retained-BaB {label} allocation was refused")))?;
        let excess = values.capacity().saturating_sub(nominal_count);
        self.charge_excess(
            excess
                .checked_mul(size_of::<T>())
                .ok_or_else(|| invalid("retained-BaB decoder Vec excess overflows usize"))?,
        )?;
        check(label)
    }

    fn charge_hash_capacity<K, V>(
        &mut self,
        nominal_count: usize,
        observed_capacity: usize,
    ) -> Result<()> {
        let excess = observed_capacity.saturating_sub(nominal_count);
        let bucket = size_of::<(K, V)>()
            .checked_add(HASH_BUCKET_OVERHEAD_BYTES)
            .ok_or_else(|| invalid("retained-BaB decoder hash bucket charge overflows"))?;
        self.charge_excess(
            excess
                .checked_mul(bucket)
                .ok_or_else(|| invalid("retained-BaB decoder hash excess overflows"))?,
        )
    }
}

fn checked_host_elements<T>(count: usize, label: &str) -> Result<usize> {
    count
        .checked_mul(size_of::<T>())
        .ok_or_else(|| invalid(format!("retained-BaB {label} host bytes overflow")))
}

fn checked_host_add(total: &mut usize, bytes: usize, label: &str) -> Result<()> {
    *total = total
        .checked_add(bytes)
        .ok_or_else(|| invalid(format!("retained-BaB {label} host total overflows")))?;
    Ok(())
}

fn families_fit_core_cap(families: ResidentBabFamilyLengthsV1) -> bool {
    let cap = GPU_BAB_BOUND_MAX_ARENA_VALUES as u64;
    [
        families.parameters,
        families.certified_errors,
        families.activation,
        families.beta,
        families.abs,
        families.box_values,
        families.cached_la,
        families.topology_metadata,
    ]
    .into_iter()
    .all(|length| length <= cap)
}

/// Allocation-free header/section/nested-length scan used before any decoder
/// reserve. It deliberately re-parses the wire rather than trusting the later
/// materializing decoder.
fn scan_decode_plan_v1(
    bytes: &[u8],
    check: &mut dyn FnMut(&'static str) -> Result<()>,
) -> Result<ResidentBabDecodePlanV1> {
    if bytes.len() < RESIDENT_BAB_TOPOLOGY_HEADER_BYTES_V1 as usize
        || bytes.len() > MAX_TOPOLOGY_BYTES
    {
        return Err(invalid(
            "retained-BaB topology byte length is outside the v1 bounds",
        ));
    }
    let mut reader = Reader::new(bytes);
    if reader.array8()? != RESIDENT_BAB_TOPOLOGY_MAGIC_V1 {
        return Err(invalid("retained-BaB topology magic mismatch"));
    }
    let version = reader.u32()?;
    let header_bytes = reader.u32()?;
    let endian_marker = reader.u32()?;
    let flags = reader.u32()?;
    let total_bytes = reader.usize_u64("total bytes")?;
    let shape_section_bytes = reader.usize_u64("shape section bytes")?;
    let node_section_bytes = reader.usize_u64("node section bytes")?;
    let segment_section_bytes = reader.usize_u64("segment section bytes")?;
    let layer_section_bytes = reader.usize_u64("layer section bytes")?;
    let node_count = reader.count("nodes", MAX_RECORDS)?;
    let segment_count = reader.count("segments", MAX_RECORDS)?;
    let layer_count = reader.count("layers", MAX_RECORDS)?;
    let _relu_count = reader.u32()?;
    let _output_node_id = reader.u32()?;
    let input_rank = reader.count("input rank", MAX_RANK)?;
    let output_rank = reader.count("output rank", MAX_RANK)?;
    let reserved1 = reader.u32()?;
    let families = ResidentBabFamilyLengthsV1 {
        parameters: reader.u64()?,
        certified_errors: reader.u64()?,
        activation: reader.u64()?,
        beta: reader.u64()?,
        abs: reader.u64()?,
        box_values: reader.u64()?,
        cached_la: reader.u64()?,
        topology_metadata: reader.u64()?,
    };
    let reserved2 = reader.u64()?;
    if version != RESIDENT_BAB_TOPOLOGY_SCHEMA_V1
        || header_bytes != RESIDENT_BAB_TOPOLOGY_HEADER_BYTES_V1
        || endian_marker != RESIDENT_BAB_TOPOLOGY_ENDIAN_MARKER_V1
        || flags != 0
        || reserved1 != 0
        || reserved2 != 0
        || node_count == 0
        || segment_count == 0
        || layer_count == 0
        || input_rank == 0
        || output_rank == 0
        || families.parameters == 0
        || families.cached_la != 0
        || families.topology_metadata != 0
        || !families_fit_core_cap(families)
    {
        return Err(invalid("retained-BaB topology header is noncanonical"));
    }
    let expected_shape_bytes = input_rank
        .checked_add(output_rank)
        .and_then(|count| count.checked_mul(size_of::<u64>()))
        .ok_or_else(|| invalid("retained-BaB shape section bytes overflow"))?;
    let expected_segment_bytes = segment_count
        .checked_mul(SEGMENT_BYTES)
        .ok_or_else(|| invalid("retained-BaB segment section bytes overflow"))?;
    let expected_layer_bytes = layer_count
        .checked_mul(LAYER_BYTES)
        .ok_or_else(|| invalid("retained-BaB layer section bytes overflow"))?;
    let minimum_node_bytes = node_count
        .checked_mul(NODE_FIXED_BYTES + 1 + size_of::<u32>() + size_of::<u64>())
        .ok_or_else(|| invalid("retained-BaB minimum node section bytes overflow"))?;
    let declared_total = usize::try_from(RESIDENT_BAB_TOPOLOGY_HEADER_BYTES_V1)
        .ok()
        .and_then(|value| value.checked_add(shape_section_bytes))
        .and_then(|value| value.checked_add(node_section_bytes))
        .and_then(|value| value.checked_add(segment_section_bytes))
        .and_then(|value| value.checked_add(layer_section_bytes));
    if total_bytes != bytes.len()
        || declared_total != Some(total_bytes)
        || shape_section_bytes != expected_shape_bytes
        || segment_section_bytes != expected_segment_bytes
        || layer_section_bytes != expected_layer_bytes
        || node_section_bytes < minimum_node_bytes
    {
        return Err(invalid(
            "retained-BaB topology fixed section/count equations are inconsistent",
        ));
    }
    check("resident topology prospective header")?;

    reader.bytes(shape_section_bytes)?;
    let node_start = reader.position();
    let mut nested_host_bytes = 0usize;
    for index in 0..node_count {
        // A single record may carry a 4-KiB name. Poll every record rather
        // than allowing the generic 1024-record stride to skip megabytes of
        // variable-width prospective work.
        check("resident topology prospective node scan")?;
        let id = reader.u32()?;
        let _kind = reader.u8()?;
        let input_count = usize::from(reader.u8()?);
        let rank = usize::from(reader.u8()?);
        let reserved = reader.u8()?;
        let name_len = reader.count("node name bytes", MAX_NODE_NAME_BYTES)?;
        let _preactivation = reader.u32()?;
        let reserved_more = reader.u32()?;
        let reserved_more2 = reader.u32()?;
        let _output_values = reader.u64()?;
        if id as usize != index
            || input_count == 0
            || input_count > MAX_NODE_INPUTS
            || rank == 0
            || rank > MAX_RANK
            || name_len == 0
            || reserved != 0
            || reserved_more != 0
            || reserved_more2 != 0
        {
            return Err(invalid(format!(
                "retained-BaB prospective node {index} is noncanonical"
            )));
        }
        let payload_bytes = name_len
            .checked_add(
                input_count
                    .checked_mul(size_of::<u32>())
                    .ok_or_else(|| invalid("retained-BaB prospective input bytes overflow"))?,
            )
            .and_then(|value| value.checked_add(rank.checked_mul(size_of::<u64>())?))
            .ok_or_else(|| invalid("retained-BaB prospective node payload overflows"))?;
        reader.bytes(payload_bytes)?;
        checked_host_add(&mut nested_host_bytes, name_len, "prospective node names")?;
        checked_host_add(
            &mut nested_host_bytes,
            checked_host_elements::<u32>(input_count, "prospective node inputs")?,
            "prospective nested node",
        )?;
        checked_host_add(
            &mut nested_host_bytes,
            checked_host_elements::<u64>(rank, "prospective node shapes")?,
            "prospective nested node",
        )?;
    }
    if reader.position().checked_sub(node_start) != Some(node_section_bytes) {
        return Err(invalid(
            "retained-BaB prospective node section length is inconsistent",
        ));
    }
    reader.bytes(segment_section_bytes)?;
    reader.bytes(layer_section_bytes)?;
    if !reader.is_empty() {
        return Err(invalid("retained-BaB topology has trailing bytes"));
    }
    check("resident topology prospective scan final")?;

    let mut nominal_host_bytes = size_of::<ResidentBabDecodedTopologyV1>();
    for charge in [
        checked_host_elements::<u64>(input_rank, "prospective input shape")?,
        checked_host_elements::<u64>(output_rank, "prospective output shape")?,
        checked_host_elements::<ResidentBabNodeV1>(node_count, "prospective nodes")?,
        checked_host_elements::<ResidentBabSegmentV1>(segment_count, "prospective segments")?,
        checked_host_elements::<ResidentBabLayerV1>(layer_count, "prospective layers")?,
        nested_host_bytes,
        size_of::<Vec<u8>>(),
        checked_host_elements::<u8>(node_count, "prospective coverage bitmap")?,
        size_of::<HashSet<&str>>(),
        checked_host_elements::<&str>(node_count, "prospective name index")?,
        node_count
            .checked_mul(HASH_BUCKET_OVERHEAD_BYTES)
            .ok_or_else(|| invalid("retained-BaB prospective name controls overflow"))?,
    ] {
        checked_host_add(
            &mut nominal_host_bytes,
            charge,
            "prospective topology decode",
        )?;
    }
    Ok(ResidentBabDecodePlanV1 {
        node_count,
        segment_count,
        layer_count,
        input_rank,
        output_rank,
        nominal_host_bytes,
    })
}

fn decode_topology_v1(
    bytes: &[u8],
    max_adapter_host_bytes: usize,
    resident_bytes_before: usize,
    check: &mut dyn FnMut(&'static str) -> Result<()>,
) -> Result<ResidentBabDecodedTopologyV1> {
    let minimum_required = resident_bytes_before
        .checked_add(size_of::<ResidentBabDecodedTopologyV1>())
        .ok_or_else(|| invalid("retained-BaB topology decode baseline overflows usize"))?;
    if max_adapter_host_bytes == 0 || minimum_required > max_adapter_host_bytes {
        return Err(NyError::CpuMemoryExceeded {
            required_bytes: minimum_required,
            budget_bytes: max_adapter_host_bytes,
            site: "retained-BaB topology decode baseline",
        });
    }
    let decode_plan = scan_decode_plan_v1(bytes, check)?;
    let mut budget = ResidentBabDecodeBudgetV1::begin(
        max_adapter_host_bytes,
        resident_bytes_before,
        decode_plan.nominal_host_bytes,
    )?;
    check("resident topology prospective decode")?;
    if bytes.len() < RESIDENT_BAB_TOPOLOGY_HEADER_BYTES_V1 as usize
        || bytes.len() > MAX_TOPOLOGY_BYTES
    {
        return Err(invalid(
            "retained-BaB topology byte length is outside the v1 bounds",
        ));
    }
    let mut reader = Reader::new(bytes);
    if reader.array8()? != RESIDENT_BAB_TOPOLOGY_MAGIC_V1 {
        return Err(invalid("retained-BaB topology magic mismatch"));
    }
    let version = reader.u32()?;
    let header_bytes = reader.u32()?;
    let endian_marker = reader.u32()?;
    let flags = reader.u32()?;
    let total_bytes = reader.usize_u64("total bytes")?;
    let shape_section_bytes = reader.usize_u64("shape section bytes")?;
    let node_section_bytes = reader.usize_u64("node section bytes")?;
    let segment_section_bytes = reader.usize_u64("segment section bytes")?;
    let layer_section_bytes = reader.usize_u64("layer section bytes")?;
    let node_count = reader.count("nodes", MAX_RECORDS)?;
    let segment_count = reader.count("segments", MAX_RECORDS)?;
    let layer_count = reader.count("layers", MAX_RECORDS)?;
    let relu_count = reader.u32()?;
    let output_node_id = reader.u32()?;
    let input_rank = reader.count("input rank", MAX_RANK)?;
    let output_rank = reader.count("output rank", MAX_RANK)?;
    let reserved1 = reader.u32()?;
    let families = ResidentBabFamilyLengthsV1 {
        parameters: reader.u64()?,
        certified_errors: reader.u64()?,
        activation: reader.u64()?,
        beta: reader.u64()?,
        abs: reader.u64()?,
        box_values: reader.u64()?,
        cached_la: reader.u64()?,
        topology_metadata: reader.u64()?,
    };
    let reserved2 = reader.u64()?;
    if version != RESIDENT_BAB_TOPOLOGY_SCHEMA_V1
        || header_bytes != RESIDENT_BAB_TOPOLOGY_HEADER_BYTES_V1
        || endian_marker != RESIDENT_BAB_TOPOLOGY_ENDIAN_MARKER_V1
        || flags != 0
        || reserved1 != 0
        || reserved2 != 0
        || node_count == 0
        || segment_count == 0
        || layer_count == 0
        || input_rank == 0
        || output_rank == 0
        || families.parameters == 0
        || families.cached_la != 0
        || families.topology_metadata != 0
        || !families_fit_core_cap(families)
        || node_count != decode_plan.node_count
        || segment_count != decode_plan.segment_count
        || layer_count != decode_plan.layer_count
        || input_rank != decode_plan.input_rank
        || output_rank != decode_plan.output_rank
    {
        return Err(invalid("retained-BaB topology header is noncanonical"));
    }
    let declared_total = usize::try_from(RESIDENT_BAB_TOPOLOGY_HEADER_BYTES_V1)
        .ok()
        .and_then(|bytes| bytes.checked_add(shape_section_bytes))
        .and_then(|bytes| bytes.checked_add(node_section_bytes))
        .and_then(|bytes| bytes.checked_add(segment_section_bytes))
        .and_then(|bytes| bytes.checked_add(layer_section_bytes));
    if total_bytes != bytes.len() || declared_total != Some(total_bytes) {
        return Err(invalid(
            "retained-BaB topology total/section byte lengths are inconsistent",
        ));
    }

    let input_shape = reader.u64_vec_accounted(
        input_rank,
        "input shape",
        &mut budget,
        check,
        "resident decoded input-shape reserve",
    )?;
    let output_shape = reader.u64_vec_accounted(
        output_rank,
        "output shape",
        &mut budget,
        check,
        "resident decoded output-shape reserve",
    )?;
    let header_len = usize::try_from(RESIDENT_BAB_TOPOLOGY_HEADER_BYTES_V1)
        .map_err(|_| invalid("retained-BaB header length does not fit usize"))?;
    let shape_end = header_len
        .checked_add(shape_section_bytes)
        .ok_or_else(|| invalid("retained-BaB shape section end overflows"))?;
    if reader.position() != shape_end {
        return Err(invalid(
            "retained-BaB decoded shape section length does not match the header",
        ));
    }
    let input_values = decode_shape_product(&input_shape, "input")?;
    let output_values = decode_shape_product(&output_shape, "output")?;
    if families.box_values != input_values {
        return Err(invalid(
            "retained-BaB box family does not match the decoded input shape",
        ));
    }

    let mut nodes = Vec::new();
    budget.reserve_vec(
        &mut nodes,
        node_count,
        "resident decoded node-table reserve",
        check,
    )?;
    for index in 0..node_count {
        poll_wire(check, "resident topology node decoding", index)?;
        let id = reader.u32()?;
        let kind = ResidentBabNodeKindV1::from_wire(reader.u8()?)?;
        let input_count = usize::from(reader.u8()?);
        let rank = usize::from(reader.u8()?);
        let reserved = reader.u8()?;
        let name_len = reader.count("node name bytes", MAX_NODE_NAME_BYTES)?;
        let preactivation_wire = reader.u32()?;
        let reserved_more = reader.u32()?;
        let reserved_more2 = reader.u32()?;
        let output_values_field = reader.u64()?;
        if id as usize != index
            || input_count != kind.input_count()
            || input_count > MAX_NODE_INPUTS
            || rank == 0
            || rank > MAX_RANK
            || name_len == 0
            || reserved != 0
            || reserved_more != 0
            || reserved_more2 != 0
        {
            return Err(invalid(format!(
                "retained-BaB decoded node {index} is noncanonical"
            )));
        }
        let name = reader.utf8_string_accounted(
            name_len,
            "node name",
            &mut budget,
            check,
            "resident decoded node-name reserve",
        )?;
        let inputs = reader.u32_vec_accounted(
            input_count,
            "node inputs",
            &mut budget,
            check,
            "resident decoded node-input reserve",
        )?;
        for &input in &inputs {
            if input != RESIDENT_BAB_NETWORK_INPUT_ID_V1 && input as usize >= index {
                return Err(invalid(format!(
                    "retained-BaB decoded node {index} input is not earlier in execution order"
                )));
            }
        }
        let relu_preactivation_node_id = if kind == ResidentBabNodeKindV1::Relu {
            Some(preactivation_wire)
        } else {
            if preactivation_wire != RESIDENT_BAB_NETWORK_INPUT_ID_V1 {
                return Err(invalid(format!(
                    "retained-BaB decoded non-ReLU node {index} carries preactivation evidence"
                )));
            }
            None
        };
        let expected_preactivation = (kind == ResidentBabNodeKindV1::Relu)
            .then(|| inputs.first().copied())
            .flatten();
        if relu_preactivation_node_id != expected_preactivation {
            return Err(invalid(format!(
                "retained-BaB decoded node {index} has noncanonical ReLU preactivation evidence"
            )));
        }
        let shape = reader.u64_vec_accounted(
            rank,
            "node output shape",
            &mut budget,
            check,
            "resident decoded node-shape reserve",
        )?;
        if decode_shape_product(&shape, "node output")? != output_values_field {
            return Err(invalid(format!(
                "retained-BaB decoded node {index} shape product mismatch"
            )));
        }
        nodes.push(ResidentBabNodeV1 {
            id,
            name,
            kind,
            inputs,
            relu_preactivation_node_id,
            output_shape: shape,
            output_values: output_values_field,
        });
    }
    check("resident topology node decoding final")?;
    let mut node_names = HashSet::new();
    node_names
        .try_reserve(node_count)
        .map_err(|_| invalid("retained-BaB decoded name-index allocation was refused"))?;
    budget.charge_hash_capacity::<&str, ()>(node_count, node_names.capacity())?;
    check("resident decoded name-index reserve")?;
    for node in &nodes {
        check("resident decoded node-name validation")?;
        if !node_names.insert(node.name.as_str()) {
            return Err(invalid("retained-BaB topology repeats a graph node name"));
        }
    }
    check("resident decoded node-name validation final")?;
    let mut coverage = Vec::new();
    budget.reserve_vec(
        &mut coverage,
        nodes.len(),
        "resident decoded coverage reserve",
        check,
    )?;
    initialize_coverage_v1(
        &mut coverage,
        nodes.len(),
        check,
        "resident decoded coverage initialization",
        "resident decoded coverage initialization final",
    )?;
    let node_end = shape_end
        .checked_add(node_section_bytes)
        .ok_or_else(|| invalid("retained-BaB node section end overflows"))?;
    if reader.position() != node_end {
        return Err(invalid(
            "retained-BaB decoded node section length does not match the header",
        ));
    }

    let mut segments = Vec::new();
    budget.reserve_vec(
        &mut segments,
        segment_count,
        "resident decoded segment-table reserve",
        check,
    )?;
    let mut next_layer = 0u64;
    let mut next_frontier_abs = 0u64;
    for index in 0..segment_count {
        poll_wire(check, "resident topology segment decoding", index)?;
        let id = reader.u32()?;
        let kind = ResidentBabSegmentKindV1::from_wire(reader.u8()?)?;
        let frontier_branch = ResidentBabFrontierBranchV1::from_wire(reader.u8()?)?;
        if reader.bytes(2)? != [0, 0] {
            return Err(invalid("retained-BaB segment reserved bytes are nonzero"));
        }
        let first_layer = reader.u32()?;
        let main_layer_count = reader.u32()?;
        let projection_layer_count = reader.u32()?;
        let frontier_node_id = reader.u32()?;
        let merge_wire = reader.u32()?;
        let reserved = reader.u32()?;
        let frontier_abs = reader.range()?;
        let merge_node_id = (merge_wire != RESIDENT_BAB_NETWORK_INPUT_ID_V1).then_some(merge_wire);
        let layer_total = u64::from(main_layer_count)
            .checked_add(u64::from(projection_layer_count))
            .ok_or_else(|| invalid("retained-BaB decoded segment layer count overflows"))?;
        let kind_invalid = match kind {
            ResidentBabSegmentKindV1::Chain => {
                projection_layer_count != 0
                    || merge_node_id.is_some()
                    || frontier_branch != ResidentBabFrontierBranchV1::Main
            }
            ResidentBabSegmentKindV1::Residual => {
                projection_layer_count != 0
                    || merge_node_id.is_none()
                    || frontier_branch != ResidentBabFrontierBranchV1::SharedResidualInput
            }
            ResidentBabSegmentKindV1::ResidualProjection => {
                projection_layer_count == 0
                    || merge_node_id.is_none()
                    || frontier_branch != ResidentBabFrontierBranchV1::SharedResidualInput
            }
        };
        if id as usize != index
            || reserved != 0
            || main_layer_count == 0
            || u64::from(first_layer) != next_layer
            || frontier_abs.start != next_frontier_abs
            || frontier_abs.len == 0
            || kind_invalid
            || (frontier_node_id != RESIDENT_BAB_NETWORK_INPUT_ID_V1
                && frontier_node_id as usize >= nodes.len())
            || merge_node_id.is_some_and(|node| node as usize >= nodes.len())
        {
            return Err(invalid(format!(
                "retained-BaB decoded segment {index} is noncanonical"
            )));
        }
        next_layer = next_layer
            .checked_add(layer_total)
            .ok_or_else(|| invalid("retained-BaB decoded layer partition overflows"))?;
        next_frontier_abs = frontier_abs.end("decoded frontier abs")?;
        segments.push(ResidentBabSegmentV1 {
            id,
            kind,
            first_layer,
            main_layer_count,
            projection_layer_count,
            frontier_node_id,
            merge_node_id,
            frontier_branch,
            frontier_abs,
        });
    }
    check("resident topology segment decoding final")?;
    let segment_end = node_end
        .checked_add(segment_section_bytes)
        .ok_or_else(|| invalid("retained-BaB segment section end overflows"))?;
    if reader.position() != segment_end {
        return Err(invalid(
            "retained-BaB decoded segment section length does not match the header",
        ));
    }
    if next_layer != layer_count as u64 {
        return Err(invalid(
            "retained-BaB decoded segments do not partition the layers",
        ));
    }

    let mut layers = Vec::new();
    budget.reserve_vec(
        &mut layers,
        layer_count,
        "resident decoded layer-table reserve",
        check,
    )?;
    for index in 0..layer_count {
        poll_wire(check, "resident topology layer decoding", index)?;
        let ordinal = reader.u32()?;
        let kind = ResidentBabLayerKindV1::from_wire(reader.u8()?)?;
        let branch = ResidentBabLayerBranchV1::from_wire(reader.u8()?)?;
        let reserved = reader.u16()?;
        let segment_id = reader.u32()?;
        let node_id = reader.u32()?;
        let reserved_more = reader.u32()?;
        let parameters = reader.range()?;
        let certified_errors = reader.range()?;
        let activation = reader.range()?;
        let beta = reader.range()?;
        let node_abs = reader.range()?;
        let mut geometry = [0u32; 13];
        for value in &mut geometry {
            *value = reader.u32()?;
        }
        if ordinal as usize != index
            || reserved != 0
            || reserved_more != 0
            || segment_id as usize >= segments.len()
            || node_id as usize >= nodes.len()
        {
            return Err(invalid(format!(
                "retained-BaB decoded layer {index} has invalid IDs"
            )));
        }
        layers.push(ResidentBabLayerV1 {
            ordinal,
            kind,
            branch,
            segment_id,
            node_id,
            parameters,
            certified_errors,
            activation,
            beta,
            node_abs,
            geometry,
        });
    }
    check("resident topology layer decoding final")?;
    let layer_end = segment_end
        .checked_add(layer_section_bytes)
        .ok_or_else(|| invalid("retained-BaB layer section end overflows"))?;
    if reader.position() != layer_end {
        return Err(invalid(
            "retained-BaB decoded layer section length does not match the header",
        ));
    }
    if !reader.is_empty() {
        return Err(invalid("retained-BaB topology has trailing bytes"));
    }

    // Keep the exact-name index live through every other request-scaled table
    // reserve so the recorded decoder peak is a real simultaneous peak. The
    // borrow must end only immediately before `nodes` moves into the result.
    drop(node_names);
    let topology = ResidentBabTopologyV1 {
        input_shape,
        output_shape,
        output_node_id,
        nodes,
        segments,
        layers,
        relu_count,
        families,
    };
    validate_decoded_layout(&topology, output_values, &mut coverage, check)?;
    drop(coverage);
    let adapter_host_retained_bytes_after =
        decoded_topology_retained_bytes(&topology, resident_bytes_before, check)?;
    if adapter_host_retained_bytes_after > budget.peak_bytes {
        return Err(invalid(
            "retained-BaB decoded retained bytes exceed admitted peak",
        ));
    }
    Ok(ResidentBabDecodedTopologyV1 {
        topology,
        adapter_host_peak_bytes: budget.peak_bytes,
        adapter_host_retained_bytes_after,
    })
}

fn decoded_topology_retained_bytes(
    topology: &ResidentBabTopologyV1,
    resident_bytes_before: usize,
    check: &mut dyn FnMut(&'static str) -> Result<()>,
) -> Result<usize> {
    let mut total = resident_bytes_before
        .checked_add(size_of::<ResidentBabDecodedTopologyV1>())
        .ok_or_else(|| invalid("retained-BaB decoded retained header overflows"))?;
    for charge in [
        checked_host_elements::<u64>(topology.input_shape.capacity(), "retained input shape")?,
        checked_host_elements::<u64>(topology.output_shape.capacity(), "retained output shape")?,
        checked_host_elements::<ResidentBabNodeV1>(topology.nodes.capacity(), "retained nodes")?,
        checked_host_elements::<ResidentBabSegmentV1>(
            topology.segments.capacity(),
            "retained segments",
        )?,
        checked_host_elements::<ResidentBabLayerV1>(topology.layers.capacity(), "retained layers")?,
    ] {
        checked_host_add(&mut total, charge, "decoded retained topology")?;
    }
    for (index, node) in topology.nodes.iter().enumerate() {
        poll_wire(check, "resident decoded retained accounting", index)?;
        checked_host_add(&mut total, node.name.capacity(), "retained node name")?;
        checked_host_add(
            &mut total,
            checked_host_elements::<u32>(node.inputs.capacity(), "retained node inputs")?,
            "decoded retained topology",
        )?;
        checked_host_add(
            &mut total,
            checked_host_elements::<u64>(node.output_shape.capacity(), "retained node shape")?,
            "decoded retained topology",
        )?;
    }
    check("resident decoded retained accounting final")?;
    Ok(total)
}

/// Decoder-side semantic validation. Kept independent from producer-side
/// validation so malformed-wire tests do not merely exercise shared encoder
/// assumptions.
fn validate_decoded_layout(
    topology: &ResidentBabTopologyV1,
    output_values: u64,
    coverage: &mut [u8],
    check: &mut dyn FnMut(&'static str) -> Result<()>,
) -> Result<()> {
    validate_decoded_node_shapes(topology, check)?;
    let output = topology
        .nodes
        .get(topology.output_node_id as usize)
        .ok_or_else(|| invalid("retained-BaB decoded output node ID is invalid"))?;
    if output.output_shape != topology.output_shape || output.output_values != output_values {
        return Err(invalid(
            "retained-BaB decoded output declaration is inconsistent",
        ));
    }

    let mut p = 0u64;
    let mut e = 0u64;
    let mut a = 0u64;
    let mut b = 0u64;
    let mut abs = topology
        .segments
        .last()
        .ok_or_else(|| invalid("retained-BaB decoded topology has no segments"))?
        .frontier_abs
        .end("decoded frontier abs")?;
    let mut relus = 0u32;
    for (index, layer) in topology.layers.iter().enumerate() {
        poll_wire(check, "resident decoded layout validation", index)?;
        let segment = &topology.segments[layer.segment_id as usize];
        let local = layer
            .ordinal
            .checked_sub(segment.first_layer)
            .ok_or_else(|| invalid("retained-BaB decoded layer precedes its segment"))?;
        let total = segment
            .main_layer_count
            .checked_add(segment.projection_layer_count)
            .ok_or_else(|| invalid("retained-BaB decoded segment count overflows"))?;
        let expected_branch = if local < segment.main_layer_count {
            ResidentBabLayerBranchV1::Main
        } else {
            ResidentBabLayerBranchV1::Projection
        };
        if local >= total || layer.branch != expected_branch {
            return Err(invalid(format!(
                "retained-BaB decoded layer {index} has wrong branch"
            )));
        }
        let node_kind = topology.nodes[layer.node_id as usize].kind;
        match layer.kind {
            ResidentBabLayerKindV1::Linear | ResidentBabLayerKindV1::Conv2d => {
                let expected = if layer.kind == ResidentBabLayerKindV1::Linear {
                    ResidentBabNodeKindV1::Linear
                } else {
                    ResidentBabNodeKindV1::Conv2d
                };
                if node_kind != expected
                    || layer.parameters.start != p
                    || layer.parameters.len == 0
                    || layer.certified_errors.start != e
                    || layer.certified_errors.len != 2
                    || layer.activation != (ResidentBabWireRangeV1 { start: a, len: 0 })
                    || layer.beta != (ResidentBabWireRangeV1 { start: b, len: 0 })
                    || layer.node_abs != (ResidentBabWireRangeV1 { start: abs, len: 0 })
                {
                    return Err(invalid(format!(
                        "retained-BaB decoded static layer {index} ranges are invalid"
                    )));
                }
                validate_decoded_static_geometry(layer)?;
                p = layer.parameters.end("decoded parameters")?;
                e = layer.certified_errors.end("decoded errors")?;
            }
            ResidentBabLayerKindV1::Relu => {
                let width = u64::from(layer.geometry[0]);
                let node = &topology.nodes[layer.node_id as usize];
                let source_shape = decoded_source_shape(topology, node.inputs[0])?;
                if node_kind != ResidentBabNodeKindV1::Relu
                    || width == 0
                    || node.output_shape != source_shape
                    || node.output_values != width
                    || decode_shape_product(source_shape, "ReLU preactivation")? != width
                    || layer.geometry[1..].iter().any(|&value| value != 0)
                    || layer.parameters != (ResidentBabWireRangeV1 { start: p, len: 0 })
                    || layer.certified_errors != (ResidentBabWireRangeV1 { start: e, len: 0 })
                    || layer.activation.start != a
                    || ResidentBabActivationSectionsV1::from_row(layer.activation, width).is_err()
                    || layer.beta
                        != (ResidentBabWireRangeV1 {
                            start: b,
                            len: width,
                        })
                    || layer.node_abs
                        != (ResidentBabWireRangeV1 {
                            start: abs,
                            len: width,
                        })
                {
                    return Err(invalid(format!(
                        "retained-BaB decoded ReLU layer {index} ranges are invalid"
                    )));
                }
                a = layer.activation.end("decoded activation")?;
                b = layer.beta.end("decoded beta")?;
                abs = layer.node_abs.end("decoded node abs")?;
                relus = relus
                    .checked_add(1)
                    .ok_or_else(|| invalid("retained-BaB decoded ReLU count overflows"))?;
            }
        }
    }
    check("resident decoded layout validation final")?;
    if p != topology.families.parameters
        || e != topology.families.certified_errors
        || a != topology.families.activation
        || b != topology.families.beta
        || abs != topology.families.abs
        || relus != topology.relu_count
    {
        return Err(invalid(
            "retained-BaB decoded family partitions are incomplete",
        ));
    }
    validate_decoded_connectivity(topology, coverage, check)?;
    Ok(())
}

fn decoded_source_shape(topology: &ResidentBabTopologyV1, source: u32) -> Result<&[u64]> {
    if source == RESIDENT_BAB_NETWORK_INPUT_ID_V1 {
        return Ok(&topology.input_shape);
    }
    topology
        .nodes
        .get(
            usize::try_from(source)
                .map_err(|_| invalid("retained-BaB decoded source ID does not fit usize"))?,
        )
        .map(|node| node.output_shape.as_slice())
        .ok_or_else(|| invalid("retained-BaB decoded source ID is outside topology"))
}

/// Decoder-owned geometry arithmetic. This deliberately does not call the
/// producer's `validate_static_geometry` helper: a common formula bug must be
/// observable by handwritten-wire tests.
fn validate_decoded_static_geometry(layer: &ResidentBabLayerV1) -> Result<()> {
    match layer.kind {
        ResidentBabLayerKindV1::Linear => {
            let [out, input, has_bias, rest @ ..] = &layer.geometry;
            if *out == 0 || *input == 0 || *has_bias > 1 || rest.iter().any(|&value| value != 0) {
                return Err(invalid(
                    "retained-BaB decoded Linear geometry is noncanonical",
                ));
            }
            let weights = u64::from(*out)
                .checked_mul(u64::from(*input))
                .ok_or_else(|| invalid("retained-BaB decoded Linear weights overflow"))?;
            let bias = u64::from(*out)
                .checked_mul(u64::from(*has_bias))
                .ok_or_else(|| invalid("retained-BaB decoded Linear bias overflows"))?;
            let expected = weights
                .checked_add(bias)
                .ok_or_else(|| invalid("retained-BaB decoded Linear parameters overflow"))?;
            if layer.parameters.len != expected {
                return Err(invalid(
                    "retained-BaB decoded Linear parameter range disagrees with geometry",
                ));
            }
        }
        ResidentBabLayerKindV1::Conv2d => {
            let [out_c, in_c, kh, kw, sh, sw, ph, pw, oh, ow, ih, iw, has_bias] = layer.geometry;
            if [out_c, in_c, kh, kw, sh, sw, oh, ow, ih, iw].contains(&0) || has_bias > 1 {
                return Err(invalid(
                    "retained-BaB decoded Conv2d has a zero required field or invalid bias tag",
                ));
            }
            let padded_h =
                u64::from(ih)
                    .checked_add(u64::from(ph).checked_mul(2).ok_or_else(|| {
                        invalid("retained-BaB decoded Conv2d pad height overflows")
                    })?)
                    .ok_or_else(|| invalid("retained-BaB decoded Conv2d input height overflows"))?;
            let padded_w =
                u64::from(iw)
                    .checked_add(u64::from(pw).checked_mul(2).ok_or_else(|| {
                        invalid("retained-BaB decoded Conv2d pad width overflows")
                    })?)
                    .ok_or_else(|| invalid("retained-BaB decoded Conv2d input width overflows"))?;
            let expected_oh = padded_h
                .checked_sub(u64::from(kh))
                .and_then(|span| span.checked_div(u64::from(sh)))
                .and_then(|value| value.checked_add(1))
                .ok_or_else(|| invalid("retained-BaB decoded Conv2d output height is invalid"))?;
            let expected_ow = padded_w
                .checked_sub(u64::from(kw))
                .and_then(|span| span.checked_div(u64::from(sw)))
                .and_then(|value| value.checked_add(1))
                .ok_or_else(|| invalid("retained-BaB decoded Conv2d output width is invalid"))?;
            if expected_oh != u64::from(oh) || expected_ow != u64::from(ow) {
                return Err(invalid(
                    "retained-BaB decoded Conv2d output formula is inconsistent",
                ));
            }
            let weights = [out_c, in_c, kh, kw]
                .into_iter()
                .try_fold(1u64, |product, value| product.checked_mul(u64::from(value)))
                .ok_or_else(|| invalid("retained-BaB decoded Conv2d weights overflow"))?;
            let bias = u64::from(out_c)
                .checked_mul(u64::from(oh))
                .and_then(|value| value.checked_mul(u64::from(ow)))
                .and_then(|value| value.checked_mul(u64::from(has_bias)))
                .ok_or_else(|| invalid("retained-BaB decoded Conv2d bias overflows"))?;
            let expected = weights
                .checked_add(bias)
                .ok_or_else(|| invalid("retained-BaB decoded Conv2d parameters overflow"))?;
            if layer.parameters.len != expected {
                return Err(invalid(
                    "retained-BaB decoded Conv2d parameter range disagrees with geometry",
                ));
            }
        }
        ResidentBabLayerKindV1::Relu => {
            return Err(invalid(
                "retained-BaB decoded static geometry received a ReLU",
            ));
        }
    }
    Ok(())
}

fn validate_decoded_node_shapes(
    topology: &ResidentBabTopologyV1,
    check: &mut dyn FnMut(&'static str) -> Result<()>,
) -> Result<()> {
    if usize::try_from(topology.output_node_id).ok() != topology.nodes.len().checked_sub(1) {
        return Err(invalid(
            "retained-BaB decoded output must be the final execution-order node",
        ));
    }
    for (index, node) in topology.nodes.iter().enumerate() {
        poll_wire(check, "resident decoded graph-shape validation", index)?;
        match node.kind {
            ResidentBabNodeKindV1::Relu => {
                let source = decoded_source_shape(topology, node.inputs[0])?;
                if node.output_shape != source {
                    return Err(invalid(format!(
                        "retained-BaB decoded ReLU node {index} changes shape"
                    )));
                }
            }
            ResidentBabNodeKindV1::Flatten | ResidentBabNodeKindV1::Reshape => {
                let source = decoded_source_shape(topology, node.inputs[0])?;
                if decode_shape_product(source, "structural source")? != node.output_values {
                    return Err(invalid(format!(
                        "retained-BaB decoded structural node {index} changes element count"
                    )));
                }
            }
            ResidentBabNodeKindV1::Add => {
                let left = decoded_source_shape(topology, node.inputs[0])?;
                let right = decoded_source_shape(topology, node.inputs[1])?;
                if left != right || left != node.output_shape {
                    return Err(invalid(format!(
                        "retained-BaB decoded Add node {index} is not exact equal-shape addition"
                    )));
                }
            }
            ResidentBabNodeKindV1::Linear | ResidentBabNodeKindV1::Conv2d => {}
        }
    }
    check("resident decoded graph-shape validation final")
}

fn validate_decoded_layer_shape(
    topology: &ResidentBabTopologyV1,
    layer: &ResidentBabLayerV1,
) -> Result<()> {
    let node = topology
        .nodes
        .get(layer.node_id as usize)
        .ok_or_else(|| invalid("retained-BaB decoded layer node is outside topology"))?;
    let source = decoded_source_shape(topology, node.inputs[0])?;
    match layer.kind {
        ResidentBabLayerKindV1::Linear => {
            validate_decoded_static_geometry(layer)?;
            let input = [u64::from(layer.geometry[1])];
            let output = [u64::from(layer.geometry[0])];
            if source != input || node.output_shape != output {
                return Err(invalid(
                    "retained-BaB decoded V1 Linear is not exact rank-1 [in] -> [out]",
                ));
            }
        }
        ResidentBabLayerKindV1::Conv2d => {
            validate_decoded_static_geometry(layer)?;
            let input = [
                u64::from(layer.geometry[1]),
                u64::from(layer.geometry[10]),
                u64::from(layer.geometry[11]),
            ];
            let output = [
                u64::from(layer.geometry[0]),
                u64::from(layer.geometry[8]),
                u64::from(layer.geometry[9]),
            ];
            if source != input || node.output_shape != output {
                return Err(invalid(
                    "retained-BaB decoded Conv2d geometry disagrees with [C,H,W] graph shapes",
                ));
            }
        }
        ResidentBabLayerKindV1::Relu => {
            let width = u64::from(layer.geometry[0]);
            if node.output_shape != source
                || node.output_values != width
                || decode_shape_product(source, "ReLU source")? != width
                || node.relu_preactivation_node_id != Some(node.inputs[0])
            {
                return Err(invalid(
                    "retained-BaB decoded ReLU geometry disagrees with preactivation/output shape",
                ));
            }
        }
    }
    Ok(())
}

fn decoded_mark_node(coverage: &mut [u8], node_id: u32, label: &str) -> Result<()> {
    let slot = coverage
        .get_mut(
            usize::try_from(node_id)
                .map_err(|_| invalid(format!("retained-BaB decoded {label} ID overflows")))?,
        )
        .ok_or_else(|| {
            invalid(format!(
                "retained-BaB decoded {label} ID is outside topology"
            ))
        })?;
    if *slot != 0 {
        return Err(invalid(format!(
            "retained-BaB decoded {label} is covered more than once"
        )));
    }
    *slot = 1;
    Ok(())
}

fn decoded_coverage_is_exact(
    coverage: &[u8],
    expected: u8,
    check: &mut dyn FnMut(&'static str) -> Result<()>,
) -> Result<bool> {
    for (index, &mark) in coverage.iter().enumerate() {
        poll_wire(check, "resident decoded coverage audit", index)?;
        if mark != expected {
            return Ok(false);
        }
    }
    check("resident decoded coverage audit final")?;
    Ok(true)
}

fn trace_decoded_branch(
    topology: &ResidentBabTopologyV1,
    layers: &[ResidentBabLayerV1],
    mut cursor: u32,
    frontier: u32,
    coverage: &mut [u8],
    check: &mut dyn FnMut(&'static str) -> Result<()>,
) -> Result<()> {
    let mut structural_steps = 0usize;
    for (position, layer) in layers.iter().enumerate() {
        poll_wire(check, "resident decoded branch connectivity", position)?;
        while cursor != frontier {
            let node = topology
                .nodes
                .get(cursor as usize)
                .ok_or_else(|| invalid("retained-BaB decoded branch ended before a layer"))?;
            if !matches!(
                node.kind,
                ResidentBabNodeKindV1::Flatten | ResidentBabNodeKindV1::Reshape
            ) {
                break;
            }
            poll_wire(
                check,
                "resident decoded structural connectivity",
                structural_steps,
            )?;
            structural_steps += 1;
            decoded_mark_node(coverage, cursor, "structural node")?;
            cursor = node.inputs[0];
        }
        if cursor == frontier || cursor != layer.node_id {
            return Err(invalid(
                "retained-BaB decoded layer order does not follow exact unary ancestry",
            ));
        }
        validate_decoded_layer_shape(topology, layer)?;
        decoded_mark_node(coverage, cursor, "executable layer node")?;
        cursor = topology.nodes[cursor as usize].inputs[0];
    }
    while cursor != frontier {
        let node = topology
            .nodes
            .get(cursor as usize)
            .ok_or_else(|| invalid("retained-BaB decoded branch does not reach its frontier"))?;
        if !matches!(
            node.kind,
            ResidentBabNodeKindV1::Flatten | ResidentBabNodeKindV1::Reshape
        ) {
            return Err(invalid(
                "retained-BaB decoded branch has an uncovered executable or merge node",
            ));
        }
        poll_wire(
            check,
            "resident decoded structural connectivity",
            structural_steps,
        )?;
        structural_steps += 1;
        decoded_mark_node(coverage, cursor, "structural node")?;
        cursor = node.inputs[0];
    }
    check("resident decoded branch connectivity final")
}

fn drain_decoded_structural_seam(
    topology: &ResidentBabTopologyV1,
    mut cursor: u32,
    target: u32,
    coverage: &mut [u8],
    check: &mut dyn FnMut(&'static str) -> Result<()>,
) -> Result<u32> {
    let mut steps = 0usize;
    while cursor != target {
        let Some(node) = usize::try_from(cursor)
            .ok()
            .and_then(|index| topology.nodes.get(index))
        else {
            break;
        };
        if !matches!(
            node.kind,
            ResidentBabNodeKindV1::Flatten | ResidentBabNodeKindV1::Reshape
        ) {
            break;
        }
        poll_wire(check, "resident decoded structural seam", steps)?;
        steps += 1;
        decoded_mark_node(coverage, cursor, "structural seam node")?;
        cursor = node.inputs[0];
    }
    check("resident decoded structural seam final")?;
    Ok(cursor)
}

fn validate_decoded_connectivity(
    topology: &ResidentBabTopologyV1,
    coverage: &mut [u8],
    check: &mut dyn FnMut(&'static str) -> Result<()>,
) -> Result<()> {
    if coverage.len() != topology.nodes.len() || !decoded_coverage_is_exact(coverage, 0, check)? {
        return Err(invalid(
            "retained-BaB decoded coverage storage is not an exact zeroed node table",
        ));
    }
    let mut cursor = topology.output_node_id;
    for (segment_index, segment) in topology.segments.iter().enumerate() {
        poll_wire(
            check,
            "resident decoded segment connectivity",
            segment_index,
        )?;
        let first = usize::try_from(segment.first_layer)
            .map_err(|_| invalid("retained-BaB decoded segment start does not fit usize"))?;
        let main_end = first
            .checked_add(segment.main_layer_count as usize)
            .ok_or_else(|| invalid("retained-BaB decoded main branch end overflows"))?;
        let projection_end = main_end
            .checked_add(segment.projection_layer_count as usize)
            .ok_or_else(|| invalid("retained-BaB decoded projection branch end overflows"))?;
        let main = topology
            .layers
            .get(first..main_end)
            .ok_or_else(|| invalid("retained-BaB decoded main branch range is invalid"))?;
        let projection = topology
            .layers
            .get(main_end..projection_end)
            .ok_or_else(|| invalid("retained-BaB decoded projection branch range is invalid"))?;
        let frontier_width = decode_shape_product(
            decoded_source_shape(topology, segment.frontier_node_id)?,
            "segment frontier",
        )?;
        if segment.frontier_abs.len != frontier_width {
            return Err(invalid(
                "retained-BaB decoded frontier Abs length disagrees with its source shape",
            ));
        }
        if segment.kind == ResidentBabSegmentKindV1::Chain
            && segment.frontier_node_id != RESIDENT_BAB_NETWORK_INPUT_ID_V1
        {
            let next = topology.segments.get(segment_index + 1).ok_or_else(|| {
                invalid("retained-BaB decoded Chain has a non-input terminal frontier")
            })?;
            if !matches!(
                next.kind,
                ResidentBabSegmentKindV1::Residual | ResidentBabSegmentKindV1::ResidualProjection
            ) || next.merge_node_id != Some(segment.frontier_node_id)
            {
                return Err(invalid(
                    "retained-BaB decoded Chain frontier is not the next residual merge",
                ));
            }
        }
        if matches!(
            segment.kind,
            ResidentBabSegmentKindV1::Residual | ResidentBabSegmentKindV1::ResidualProjection
        ) {
            let merge_id = segment
                .merge_node_id
                .ok_or_else(|| invalid("retained-BaB decoded residual has no merge"))?;
            cursor = drain_decoded_structural_seam(topology, cursor, merge_id, coverage, check)?;
        }
        match segment.kind {
            ResidentBabSegmentKindV1::Chain => {
                trace_decoded_branch(
                    topology,
                    main,
                    cursor,
                    segment.frontier_node_id,
                    coverage,
                    check,
                )?;
            }
            ResidentBabSegmentKindV1::Residual => {
                let merge_id = segment
                    .merge_node_id
                    .ok_or_else(|| invalid("retained-BaB decoded residual has no merge"))?;
                if cursor != merge_id {
                    return Err(invalid(
                        "retained-BaB decoded residual merge does not equal the fold cursor",
                    ));
                }
                let merge = topology
                    .nodes
                    .get(merge_id as usize)
                    .ok_or_else(|| invalid("retained-BaB decoded residual merge is invalid"))?;
                if merge.kind != ResidentBabNodeKindV1::Add {
                    return Err(invalid("retained-BaB decoded residual merge is not Add"));
                }
                decoded_mark_node(coverage, merge_id, "residual Add merge")?;
                let left_is_frontier = merge.inputs[0] == segment.frontier_node_id;
                let right_is_frontier = merge.inputs[1] == segment.frontier_node_id;
                if left_is_frontier == right_is_frontier {
                    return Err(invalid(
                        "retained-BaB decoded identity residual must have exactly one frontier input",
                    ));
                }
                let main_start = if left_is_frontier {
                    merge.inputs[1]
                } else {
                    merge.inputs[0]
                };
                trace_decoded_branch(
                    topology,
                    main,
                    main_start,
                    segment.frontier_node_id,
                    coverage,
                    check,
                )?;
            }
            ResidentBabSegmentKindV1::ResidualProjection => {
                let merge_id = segment.merge_node_id.ok_or_else(|| {
                    invalid("retained-BaB decoded projection residual has no merge")
                })?;
                if cursor != merge_id {
                    return Err(invalid(
                        "retained-BaB decoded projection merge does not equal the fold cursor",
                    ));
                }
                let merge = topology
                    .nodes
                    .get(merge_id as usize)
                    .ok_or_else(|| invalid("retained-BaB decoded projection merge is invalid"))?;
                if merge.kind != ResidentBabNodeKindV1::Add {
                    return Err(invalid("retained-BaB decoded projection merge is not Add"));
                }
                decoded_mark_node(coverage, merge_id, "residual Add merge")?;
                trace_decoded_branch(
                    topology,
                    main,
                    merge.inputs[0],
                    segment.frontier_node_id,
                    coverage,
                    check,
                )?;
                trace_decoded_branch(
                    topology,
                    projection,
                    merge.inputs[1],
                    segment.frontier_node_id,
                    coverage,
                    check,
                )?;
            }
        }
        cursor = segment.frontier_node_id;
    }
    cursor = drain_decoded_structural_seam(
        topology,
        cursor,
        RESIDENT_BAB_NETWORK_INPUT_ID_V1,
        coverage,
        check,
    )?;
    check("resident decoded segment connectivity final")?;
    if cursor != RESIDENT_BAB_NETWORK_INPUT_ID_V1 || !decoded_coverage_is_exact(coverage, 1, check)?
    {
        return Err(invalid(
            "retained-BaB decoded segments do not cover every exec-order node exactly once to network input",
        ));
    }
    Ok(())
}

fn decode_shape_product(shape: &[u64], label: &str) -> Result<u64> {
    if shape.is_empty() || shape.len() > MAX_RANK || shape.contains(&0) {
        return Err(invalid(format!(
            "retained-BaB decoded {label} shape is invalid"
        )));
    }
    let values = shape.iter().try_fold(1u64, |product, &dim| {
        product
            .checked_mul(dim)
            .ok_or_else(|| invalid(format!("retained-BaB decoded {label} shape overflows")))
    })?;
    if values > GPU_BAB_BOUND_MAX_ARENA_VALUES as u64 {
        return Err(invalid(format!(
            "retained-BaB decoded {label} shape exceeds the core arena-value cap"
        )));
    }
    Ok(values)
}

fn push_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_range(out: &mut Vec<u8>, range: ResidentBabWireRangeV1) {
    push_u64(out, range.start);
    push_u64(out, range.len);
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn is_empty(&self) -> bool {
        self.offset == self.bytes.len()
    }

    fn position(&self) -> usize {
        self.offset
    }

    fn bytes(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or_else(|| invalid("retained-BaB decoder offset overflows"))?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or_else(|| invalid("retained-BaB topology is truncated"))?;
        self.offset = end;
        Ok(value)
    }

    fn array8(&mut self) -> Result<[u8; 8]> {
        self.bytes(8)?
            .try_into()
            .map_err(|_| invalid("retained-BaB topology is truncated"))
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.bytes(1)?[0])
    }

    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(
            self.bytes(2)?
                .try_into()
                .map_err(|_| invalid("retained-BaB u16 is truncated"))?,
        ))
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(
            self.bytes(4)?
                .try_into()
                .map_err(|_| invalid("retained-BaB u32 is truncated"))?,
        ))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(
            self.bytes(8)?
                .try_into()
                .map_err(|_| invalid("retained-BaB u64 is truncated"))?,
        ))
    }

    fn usize_u64(&mut self, label: &str) -> Result<usize> {
        usize::try_from(self.u64()?)
            .map_err(|_| invalid(format!("retained-BaB {label} does not fit usize")))
    }

    fn count(&mut self, label: &str, cap: usize) -> Result<usize> {
        let count = usize::try_from(self.u32()?)
            .map_err(|_| invalid(format!("retained-BaB {label} count does not fit usize")))?;
        if count > cap {
            return Err(invalid(format!(
                "retained-BaB {label} count exceeds the v1 cap"
            )));
        }
        Ok(count)
    }

    fn u32_vec_accounted(
        &mut self,
        count: usize,
        label: &str,
        budget: &mut ResidentBabDecodeBudgetV1,
        check: &mut dyn FnMut(&'static str) -> Result<()>,
        reserve_label: &'static str,
    ) -> Result<Vec<u32>> {
        let byte_count = count
            .checked_mul(4)
            .ok_or_else(|| invalid(format!("retained-BaB {label} byte count overflows")))?;
        let raw = self.bytes(byte_count)?;
        let mut values = Vec::new();
        budget.reserve_vec(&mut values, count, reserve_label, check)?;
        for chunk in raw.chunks_exact(4) {
            values.push(u32::from_le_bytes(chunk.try_into().map_err(|_| {
                invalid(format!("retained-BaB {label} value is truncated"))
            })?));
        }
        Ok(values)
    }

    fn utf8_string_accounted(
        &mut self,
        len: usize,
        label: &str,
        budget: &mut ResidentBabDecodeBudgetV1,
        check: &mut dyn FnMut(&'static str) -> Result<()>,
        reserve_label: &'static str,
    ) -> Result<String> {
        let raw = self.bytes(len)?;
        let mut owned = Vec::new();
        budget.reserve_vec(&mut owned, len, reserve_label, check)?;
        owned.extend_from_slice(raw);
        String::from_utf8(owned)
            .map_err(|_| invalid(format!("retained-BaB {label} is not canonical UTF-8")))
    }

    fn u64_vec_accounted(
        &mut self,
        count: usize,
        label: &str,
        budget: &mut ResidentBabDecodeBudgetV1,
        check: &mut dyn FnMut(&'static str) -> Result<()>,
        reserve_label: &'static str,
    ) -> Result<Vec<u64>> {
        let byte_count = count
            .checked_mul(8)
            .ok_or_else(|| invalid(format!("retained-BaB {label} byte count overflows")))?;
        let raw = self.bytes(byte_count)?;
        let mut values = Vec::new();
        budget.reserve_vec(&mut values, count, reserve_label, check)?;
        for chunk in raw.chunks_exact(8) {
            values.push(u64::from_le_bytes(chunk.try_into().map_err(|_| {
                invalid(format!("retained-BaB {label} value is truncated"))
            })?));
        }
        Ok(values)
    }

    fn range(&mut self) -> Result<ResidentBabWireRangeV1> {
        Ok(ResidentBabWireRangeV1 {
            start: self.u64()?,
            len: self.u64()?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode(topology: &ResidentBabTopologyV1) -> Result<Vec<u8>> {
        let mut check = |_| Ok(());
        Ok(topology.encode(1 << 20, &mut check)?.bytes)
    }

    fn decode(bytes: &[u8]) -> Result<ResidentBabTopologyV1> {
        let mut check = |_| Ok(());
        Ok(ResidentBabTopologyV1::decode(bytes, 1 << 24, 0, &mut check)?.into_topology())
    }

    fn length(topology: &ResidentBabTopologyV1) -> Result<usize> {
        let mut check = |_| Ok(());
        encoded_len(topology, &mut check)
    }

    fn range(start: u64, len: u64) -> ResidentBabWireRangeV1 {
        ResidentBabWireRangeV1 { start, len }
    }

    fn fixture() -> ResidentBabTopologyV1 {
        ResidentBabTopologyV1 {
            input_shape: vec![2],
            output_shape: vec![2],
            output_node_id: 2,
            nodes: vec![
                ResidentBabNodeV1 {
                    id: 0,
                    name: "linear_in".to_string(),
                    kind: ResidentBabNodeKindV1::Linear,
                    inputs: vec![RESIDENT_BAB_NETWORK_INPUT_ID_V1],
                    relu_preactivation_node_id: None,
                    output_shape: vec![2],
                    output_values: 2,
                },
                ResidentBabNodeV1 {
                    id: 1,
                    name: "relu".to_string(),
                    kind: ResidentBabNodeKindV1::Relu,
                    inputs: vec![0],
                    relu_preactivation_node_id: Some(0),
                    output_shape: vec![2],
                    output_values: 2,
                },
                ResidentBabNodeV1 {
                    id: 2,
                    name: "linear_out".to_string(),
                    kind: ResidentBabNodeKindV1::Linear,
                    inputs: vec![1],
                    relu_preactivation_node_id: None,
                    output_shape: vec![2],
                    output_values: 2,
                },
            ],
            segments: vec![ResidentBabSegmentV1 {
                id: 0,
                kind: ResidentBabSegmentKindV1::Chain,
                first_layer: 0,
                main_layer_count: 3,
                projection_layer_count: 0,
                frontier_node_id: RESIDENT_BAB_NETWORK_INPUT_ID_V1,
                merge_node_id: None,
                frontier_branch: ResidentBabFrontierBranchV1::Main,
                frontier_abs: range(0, 2),
            }],
            // Backward fold order: output Linear, ReLU, input Linear.
            layers: vec![
                ResidentBabLayerV1 {
                    ordinal: 0,
                    kind: ResidentBabLayerKindV1::Linear,
                    branch: ResidentBabLayerBranchV1::Main,
                    segment_id: 0,
                    node_id: 2,
                    parameters: range(0, 6),
                    certified_errors: range(0, 2),
                    activation: range(0, 0),
                    beta: range(0, 0),
                    node_abs: range(2, 0),
                    geometry: [2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                },
                ResidentBabLayerV1 {
                    ordinal: 1,
                    kind: ResidentBabLayerKindV1::Relu,
                    branch: ResidentBabLayerBranchV1::Main,
                    segment_id: 0,
                    node_id: 1,
                    parameters: range(6, 0),
                    certified_errors: range(2, 0),
                    activation: range(0, 13),
                    beta: range(0, 2),
                    node_abs: range(2, 2),
                    geometry: [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                },
                ResidentBabLayerV1 {
                    ordinal: 2,
                    kind: ResidentBabLayerKindV1::Linear,
                    branch: ResidentBabLayerBranchV1::Main,
                    segment_id: 0,
                    node_id: 0,
                    parameters: range(6, 6),
                    certified_errors: range(2, 2),
                    activation: range(13, 0),
                    beta: range(2, 0),
                    node_abs: range(4, 0),
                    geometry: [2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                },
            ],
            relu_count: 1,
            families: ResidentBabFamilyLengthsV1 {
                parameters: 12,
                certified_errors: 4,
                activation: 13,
                beta: 2,
                abs: 4,
                box_values: 2,
                cached_la: 0,
                topology_metadata: 0,
            },
        }
    }

    fn residual_fixture() -> ResidentBabTopologyV1 {
        ResidentBabTopologyV1 {
            input_shape: vec![2],
            output_shape: vec![2],
            output_node_id: 3,
            nodes: vec![
                ResidentBabNodeV1 {
                    id: 0,
                    name: "residual_main_linear".to_string(),
                    kind: ResidentBabNodeKindV1::Linear,
                    inputs: vec![RESIDENT_BAB_NETWORK_INPUT_ID_V1],
                    relu_preactivation_node_id: None,
                    output_shape: vec![2],
                    output_values: 2,
                },
                ResidentBabNodeV1 {
                    id: 1,
                    name: "residual_relu".to_string(),
                    kind: ResidentBabNodeKindV1::Relu,
                    inputs: vec![0],
                    relu_preactivation_node_id: Some(0),
                    output_shape: vec![2],
                    output_values: 2,
                },
                ResidentBabNodeV1 {
                    id: 2,
                    name: "residual_add".to_string(),
                    kind: ResidentBabNodeKindV1::Add,
                    inputs: vec![1, RESIDENT_BAB_NETWORK_INPUT_ID_V1],
                    relu_preactivation_node_id: None,
                    output_shape: vec![2],
                    output_values: 2,
                },
                ResidentBabNodeV1 {
                    id: 3,
                    name: "residual_output".to_string(),
                    kind: ResidentBabNodeKindV1::Linear,
                    inputs: vec![2],
                    relu_preactivation_node_id: None,
                    output_shape: vec![2],
                    output_values: 2,
                },
            ],
            segments: vec![
                ResidentBabSegmentV1 {
                    id: 0,
                    kind: ResidentBabSegmentKindV1::Chain,
                    first_layer: 0,
                    main_layer_count: 1,
                    projection_layer_count: 0,
                    frontier_node_id: 2,
                    merge_node_id: None,
                    frontier_branch: ResidentBabFrontierBranchV1::Main,
                    frontier_abs: range(0, 2),
                },
                ResidentBabSegmentV1 {
                    id: 1,
                    kind: ResidentBabSegmentKindV1::Residual,
                    first_layer: 1,
                    main_layer_count: 2,
                    projection_layer_count: 0,
                    frontier_node_id: RESIDENT_BAB_NETWORK_INPUT_ID_V1,
                    merge_node_id: Some(2),
                    frontier_branch: ResidentBabFrontierBranchV1::SharedResidualInput,
                    frontier_abs: range(2, 2),
                },
            ],
            layers: vec![
                ResidentBabLayerV1 {
                    ordinal: 0,
                    kind: ResidentBabLayerKindV1::Linear,
                    branch: ResidentBabLayerBranchV1::Main,
                    segment_id: 0,
                    node_id: 3,
                    parameters: range(0, 6),
                    certified_errors: range(0, 2),
                    activation: range(0, 0),
                    beta: range(0, 0),
                    node_abs: range(4, 0),
                    geometry: [2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                },
                ResidentBabLayerV1 {
                    ordinal: 1,
                    kind: ResidentBabLayerKindV1::Relu,
                    branch: ResidentBabLayerBranchV1::Main,
                    segment_id: 1,
                    node_id: 1,
                    parameters: range(6, 0),
                    certified_errors: range(2, 0),
                    activation: range(0, 13),
                    beta: range(0, 2),
                    node_abs: range(4, 2),
                    geometry: [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                },
                ResidentBabLayerV1 {
                    ordinal: 2,
                    kind: ResidentBabLayerKindV1::Linear,
                    branch: ResidentBabLayerBranchV1::Main,
                    segment_id: 1,
                    node_id: 0,
                    parameters: range(6, 6),
                    certified_errors: range(2, 2),
                    activation: range(13, 0),
                    beta: range(2, 0),
                    node_abs: range(6, 0),
                    geometry: [2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                },
            ],
            relu_count: 1,
            families: ResidentBabFamilyLengthsV1 {
                parameters: 12,
                certified_errors: 4,
                activation: 13,
                beta: 2,
                abs: 6,
                box_values: 2,
                cached_la: 0,
                topology_metadata: 0,
            },
        }
    }

    fn projection_fixture() -> ResidentBabTopologyV1 {
        let mut topology = residual_fixture();
        topology.nodes[1] = ResidentBabNodeV1 {
            id: 1,
            name: "residual_projection".to_string(),
            kind: ResidentBabNodeKindV1::Linear,
            inputs: vec![RESIDENT_BAB_NETWORK_INPUT_ID_V1],
            relu_preactivation_node_id: None,
            output_shape: vec![2],
            output_values: 2,
        };
        topology.nodes[2].inputs = vec![0, 1];
        topology.segments[1].kind = ResidentBabSegmentKindV1::ResidualProjection;
        topology.segments[1].main_layer_count = 1;
        topology.segments[1].projection_layer_count = 1;
        topology.layers[1] = ResidentBabLayerV1 {
            ordinal: 1,
            kind: ResidentBabLayerKindV1::Linear,
            branch: ResidentBabLayerBranchV1::Main,
            segment_id: 1,
            node_id: 0,
            parameters: range(6, 6),
            certified_errors: range(2, 2),
            activation: range(0, 0),
            beta: range(0, 0),
            node_abs: range(4, 0),
            geometry: [2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        };
        topology.layers[2] = ResidentBabLayerV1 {
            ordinal: 2,
            kind: ResidentBabLayerKindV1::Linear,
            branch: ResidentBabLayerBranchV1::Projection,
            segment_id: 1,
            node_id: 1,
            parameters: range(12, 6),
            certified_errors: range(4, 2),
            activation: range(0, 0),
            beta: range(0, 0),
            node_abs: range(4, 0),
            geometry: [2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        };
        topology.relu_count = 0;
        topology.families.parameters = 18;
        topology.families.certified_errors = 6;
        topology.families.activation = 0;
        topology.families.beta = 0;
        topology.families.abs = 4;
        topology
    }

    fn conv_fixture() -> ResidentBabTopologyV1 {
        ResidentBabTopologyV1 {
            input_shape: vec![1, 3, 3],
            output_shape: vec![2],
            output_node_id: 3,
            nodes: vec![
                ResidentBabNodeV1 {
                    id: 0,
                    name: "conv".to_string(),
                    kind: ResidentBabNodeKindV1::Conv2d,
                    inputs: vec![RESIDENT_BAB_NETWORK_INPUT_ID_V1],
                    relu_preactivation_node_id: None,
                    output_shape: vec![1, 3, 3],
                    output_values: 9,
                },
                ResidentBabNodeV1 {
                    id: 1,
                    name: "conv_relu".to_string(),
                    kind: ResidentBabNodeKindV1::Relu,
                    inputs: vec![0],
                    relu_preactivation_node_id: Some(0),
                    output_shape: vec![1, 3, 3],
                    output_values: 9,
                },
                ResidentBabNodeV1 {
                    id: 2,
                    name: "flatten".to_string(),
                    kind: ResidentBabNodeKindV1::Flatten,
                    inputs: vec![1],
                    relu_preactivation_node_id: None,
                    output_shape: vec![9],
                    output_values: 9,
                },
                ResidentBabNodeV1 {
                    id: 3,
                    name: "conv_output".to_string(),
                    kind: ResidentBabNodeKindV1::Linear,
                    inputs: vec![2],
                    relu_preactivation_node_id: None,
                    output_shape: vec![2],
                    output_values: 2,
                },
            ],
            segments: vec![ResidentBabSegmentV1 {
                id: 0,
                kind: ResidentBabSegmentKindV1::Chain,
                first_layer: 0,
                main_layer_count: 3,
                projection_layer_count: 0,
                frontier_node_id: RESIDENT_BAB_NETWORK_INPUT_ID_V1,
                merge_node_id: None,
                frontier_branch: ResidentBabFrontierBranchV1::Main,
                frontier_abs: range(0, 9),
            }],
            layers: vec![
                ResidentBabLayerV1 {
                    ordinal: 0,
                    kind: ResidentBabLayerKindV1::Linear,
                    branch: ResidentBabLayerBranchV1::Main,
                    segment_id: 0,
                    node_id: 3,
                    parameters: range(0, 20),
                    certified_errors: range(0, 2),
                    activation: range(0, 0),
                    beta: range(0, 0),
                    node_abs: range(9, 0),
                    geometry: [2, 9, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                },
                ResidentBabLayerV1 {
                    ordinal: 1,
                    kind: ResidentBabLayerKindV1::Relu,
                    branch: ResidentBabLayerBranchV1::Main,
                    segment_id: 0,
                    node_id: 1,
                    parameters: range(20, 0),
                    certified_errors: range(2, 0),
                    activation: range(0, 55),
                    beta: range(0, 9),
                    node_abs: range(9, 9),
                    geometry: [9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                },
                ResidentBabLayerV1 {
                    ordinal: 2,
                    kind: ResidentBabLayerKindV1::Conv2d,
                    branch: ResidentBabLayerBranchV1::Main,
                    segment_id: 0,
                    node_id: 0,
                    parameters: range(20, 1),
                    certified_errors: range(2, 2),
                    activation: range(55, 0),
                    beta: range(9, 0),
                    node_abs: range(18, 0),
                    geometry: [1, 1, 1, 1, 1, 1, 0, 0, 3, 3, 3, 3, 0],
                },
            ],
            relu_count: 1,
            families: ResidentBabFamilyLengthsV1 {
                parameters: 21,
                certified_errors: 4,
                activation: 55,
                beta: 9,
                abs: 18,
                box_values: 9,
                cached_la: 0,
                topology_metadata: 0,
            },
        }
    }

    fn nontrivial_conv_fixture() -> ResidentBabTopologyV1 {
        let mut topology = conv_fixture();
        topology.input_shape = vec![1, 5, 5];
        topology.nodes[0].output_shape = vec![2, 3, 3];
        topology.nodes[0].output_values = 18;
        topology.nodes[1].output_shape = vec![2, 3, 3];
        topology.nodes[1].output_values = 18;
        topology.nodes[2].output_shape = vec![18];
        topology.nodes[2].output_values = 18;
        topology.segments[0].frontier_abs = range(0, 25);
        topology.layers[0].parameters = range(0, 38);
        topology.layers[0].node_abs.start = 25;
        topology.layers[0].geometry = [2, 18, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        topology.layers[1].parameters.start = 38;
        topology.layers[1].activation = range(0, 109);
        topology.layers[1].beta = range(0, 18);
        topology.layers[1].node_abs = range(25, 18);
        topology.layers[1].geometry[0] = 18;
        topology.layers[2].parameters = range(38, 36);
        topology.layers[2].activation.start = 109;
        topology.layers[2].beta.start = 18;
        topology.layers[2].node_abs.start = 43;
        topology.layers[2].geometry = [2, 1, 3, 3, 2, 2, 1, 1, 3, 3, 5, 5, 1];
        topology.families.parameters = 74;
        topology.families.activation = 109;
        topology.families.beta = 18;
        topology.families.abs = 43;
        topology.families.box_values = 25;
        topology
    }

    fn structural_seam_fixture() -> ResidentBabTopologyV1 {
        ResidentBabTopologyV1 {
            input_shape: vec![2],
            output_shape: vec![2],
            output_node_id: 4,
            nodes: vec![
                ResidentBabNodeV1 {
                    id: 0,
                    name: "seam_input_reshape".to_string(),
                    kind: ResidentBabNodeKindV1::Reshape,
                    inputs: vec![RESIDENT_BAB_NETWORK_INPUT_ID_V1],
                    relu_preactivation_node_id: None,
                    output_shape: vec![2],
                    output_values: 2,
                },
                ResidentBabNodeV1 {
                    id: 1,
                    name: "seam_main_linear".to_string(),
                    kind: ResidentBabNodeKindV1::Linear,
                    inputs: vec![0],
                    relu_preactivation_node_id: None,
                    output_shape: vec![2],
                    output_values: 2,
                },
                ResidentBabNodeV1 {
                    id: 2,
                    name: "seam_relu".to_string(),
                    kind: ResidentBabNodeKindV1::Relu,
                    inputs: vec![1],
                    relu_preactivation_node_id: Some(1),
                    output_shape: vec![2],
                    output_values: 2,
                },
                ResidentBabNodeV1 {
                    id: 3,
                    name: "seam_add".to_string(),
                    kind: ResidentBabNodeKindV1::Add,
                    inputs: vec![2, 0],
                    relu_preactivation_node_id: None,
                    output_shape: vec![2],
                    output_values: 2,
                },
                ResidentBabNodeV1 {
                    id: 4,
                    name: "seam_output_flatten".to_string(),
                    kind: ResidentBabNodeKindV1::Flatten,
                    inputs: vec![3],
                    relu_preactivation_node_id: None,
                    output_shape: vec![2],
                    output_values: 2,
                },
            ],
            segments: vec![ResidentBabSegmentV1 {
                id: 0,
                kind: ResidentBabSegmentKindV1::Residual,
                first_layer: 0,
                main_layer_count: 2,
                projection_layer_count: 0,
                frontier_node_id: 0,
                merge_node_id: Some(3),
                frontier_branch: ResidentBabFrontierBranchV1::SharedResidualInput,
                frontier_abs: range(0, 2),
            }],
            layers: vec![
                ResidentBabLayerV1 {
                    ordinal: 0,
                    kind: ResidentBabLayerKindV1::Relu,
                    branch: ResidentBabLayerBranchV1::Main,
                    segment_id: 0,
                    node_id: 2,
                    parameters: range(0, 0),
                    certified_errors: range(0, 0),
                    activation: range(0, 13),
                    beta: range(0, 2),
                    node_abs: range(2, 2),
                    geometry: [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                },
                ResidentBabLayerV1 {
                    ordinal: 1,
                    kind: ResidentBabLayerKindV1::Linear,
                    branch: ResidentBabLayerBranchV1::Main,
                    segment_id: 0,
                    node_id: 1,
                    parameters: range(0, 6),
                    certified_errors: range(0, 2),
                    activation: range(13, 0),
                    beta: range(2, 0),
                    node_abs: range(4, 0),
                    geometry: [2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                },
            ],
            relu_count: 1,
            families: ResidentBabFamilyLengthsV1 {
                parameters: 6,
                certified_errors: 2,
                activation: 13,
                beta: 2,
                abs: 4,
                box_values: 2,
                cached_la: 0,
                topology_metadata: 0,
            },
        }
    }

    fn manual_u16(bytes: &mut Vec<u8>, value: u16) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn manual_u32(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn manual_u64(bytes: &mut Vec<u8>, value: u64) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    /// A test-only schema writer that does not call the production encoder or
    /// either production validation pass. It is intentionally duplicated so a
    /// decoder/encoder bug cannot be hidden by shared validation logic.
    fn manual_topology_bytes(model: &ResidentBabTopologyV1) -> Vec<u8> {
        let shape_bytes = model
            .input_shape
            .len()
            .checked_add(model.output_shape.len())
            .and_then(|count| count.checked_mul(size_of::<u64>()))
            .unwrap();
        let node_bytes = model
            .nodes
            .iter()
            .map(|node| {
                NODE_FIXED_BYTES
                    + node.name.len()
                    + node.inputs.len() * 4
                    + node.output_shape.len() * 8
            })
            .sum::<usize>();
        let segment_bytes = model.segments.len() * SEGMENT_BYTES;
        let layer_bytes = model.layers.len() * LAYER_BYTES;
        let total_bytes = RESIDENT_BAB_TOPOLOGY_HEADER_BYTES_V1 as usize
            + shape_bytes
            + node_bytes
            + segment_bytes
            + layer_bytes;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"NYRBWV1\0");
        for value in [1, 168, RESIDENT_BAB_TOPOLOGY_ENDIAN_MARKER_V1, 0] {
            manual_u32(&mut bytes, value);
        }
        for value in [
            total_bytes,
            shape_bytes,
            node_bytes,
            segment_bytes,
            layer_bytes,
        ] {
            manual_u64(&mut bytes, u64::try_from(value).unwrap());
        }
        for value in [
            u32::try_from(model.nodes.len()).unwrap(),
            u32::try_from(model.segments.len()).unwrap(),
            u32::try_from(model.layers.len()).unwrap(),
            model.relu_count,
            model.output_node_id,
            u32::try_from(model.input_shape.len()).unwrap(),
            u32::try_from(model.output_shape.len()).unwrap(),
            0,
        ] {
            manual_u32(&mut bytes, value);
        }
        for value in [
            model.families.parameters,
            model.families.certified_errors,
            model.families.activation,
            model.families.beta,
            model.families.abs,
            model.families.box_values,
            model.families.cached_la,
            model.families.topology_metadata,
            0,
        ] {
            manual_u64(&mut bytes, value);
        }
        for &dim in &model.input_shape {
            manual_u64(&mut bytes, dim);
        }
        for &dim in &model.output_shape {
            manual_u64(&mut bytes, dim);
        }

        for node in &model.nodes {
            manual_u32(&mut bytes, node.id);
            bytes.push(node.kind as u8);
            bytes.push(u8::try_from(node.inputs.len()).unwrap());
            bytes.push(u8::try_from(node.output_shape.len()).unwrap());
            bytes.push(0);
            manual_u32(&mut bytes, u32::try_from(node.name.len()).unwrap());
            manual_u32(
                &mut bytes,
                node.relu_preactivation_node_id
                    .unwrap_or(RESIDENT_BAB_NETWORK_INPUT_ID_V1),
            );
            manual_u32(&mut bytes, 0);
            manual_u32(&mut bytes, 0);
            manual_u64(&mut bytes, node.output_values);
            bytes.extend_from_slice(node.name.as_bytes());
            for &input in &node.inputs {
                manual_u32(&mut bytes, input);
            }
            for &dim in &node.output_shape {
                manual_u64(&mut bytes, dim);
            }
        }

        for segment in &model.segments {
            manual_u32(&mut bytes, segment.id);
            bytes.push(segment.kind as u8);
            bytes.push(segment.frontier_branch as u8);
            manual_u16(&mut bytes, 0);
            for value in [
                segment.first_layer,
                segment.main_layer_count,
                segment.projection_layer_count,
                segment.frontier_node_id,
                segment
                    .merge_node_id
                    .unwrap_or(RESIDENT_BAB_NETWORK_INPUT_ID_V1),
                0,
            ] {
                manual_u32(&mut bytes, value);
            }
            manual_u64(&mut bytes, segment.frontier_abs.start);
            manual_u64(&mut bytes, segment.frontier_abs.len);
        }

        for layer in &model.layers {
            manual_u32(&mut bytes, layer.ordinal);
            bytes.push(layer.kind as u8);
            bytes.push(layer.branch as u8);
            manual_u16(&mut bytes, 0);
            manual_u32(&mut bytes, layer.segment_id);
            manual_u32(&mut bytes, layer.node_id);
            manual_u32(&mut bytes, 0);
            for range in [
                layer.parameters,
                layer.certified_errors,
                layer.activation,
                layer.beta,
                layer.node_abs,
            ] {
                manual_u64(&mut bytes, range.start);
                manual_u64(&mut bytes, range.len);
            }
            for value in layer.geometry {
                manual_u32(&mut bytes, value);
            }
        }
        bytes
    }

    fn manual_fixture_bytes() -> Vec<u8> {
        manual_topology_bytes(&fixture())
    }

    fn assert_producer_and_decoder_reject(topology: &ResidentBabTopologyV1) {
        assert!(
            encode(topology).is_err(),
            "producer accepted malformed topology"
        );
        assert!(
            decode(&manual_topology_bytes(topology)).is_err(),
            "independent decoder accepted malformed wire"
        );
    }

    #[test]
    fn conv_geometry_accepts_zero_padding_and_exact_optional_bias() {
        let mut layer = fixture().layers[0].clone();
        layer.kind = ResidentBabLayerKindV1::Conv2d;
        layer.geometry = [1, 1, 1, 1, 1, 1, 0, 0, 2, 2, 2, 2, 0];
        layer.parameters = range(0, 1);
        validate_static_geometry(&layer).unwrap();

        layer.geometry[12] = 1;
        layer.parameters = range(0, 5);
        validate_static_geometry(&layer).unwrap();

        layer.parameters = range(0, 4);
        assert!(validate_static_geometry(&layer).is_err());
    }

    #[test]
    fn encoder_and_independent_decoder_round_trip_exact_layout() {
        let expected = fixture();
        let encoded = encode(&expected).unwrap();
        assert_eq!(encoded.len(), length(&expected).unwrap());
        assert_eq!(decode(&encoded).unwrap(), expected);
    }

    #[test]
    fn residual_projection_conv_and_structural_seams_round_trip() {
        for expected in [
            residual_fixture(),
            projection_fixture(),
            conv_fixture(),
            structural_seam_fixture(),
        ] {
            let encoded = encode(&expected).unwrap();
            assert_eq!(decode(&encoded).unwrap(), expected);
        }
    }

    #[test]
    fn handwritten_nontrivial_conv_with_expanded_bias_is_exact() {
        let expected = nontrivial_conv_fixture();
        let manual = manual_topology_bytes(&expected);
        assert_eq!(decode(&manual).unwrap(), expected);
        assert_eq!(encode(&expected).unwrap(), manual);

        let mut bad_bias_charge = expected;
        bad_bias_charge.layers[2].parameters.len -= 1;
        bad_bias_charge.families.parameters -= 1;
        assert_producer_and_decoder_reject(&bad_bias_charge);
    }

    #[test]
    fn independent_decoder_accepts_handwritten_nonempty_input_records() {
        let expected = fixture();
        let manual = manual_fixture_bytes();
        assert_eq!(manual.len(), 843);
        assert_eq!(decode(&manual).unwrap(), expected);
        assert_eq!(encode(&expected).unwrap(), manual);
    }

    #[test]
    fn decoder_rejects_reserved_unknown_trailing_and_truncated_data() {
        let encoded = encode(&fixture()).unwrap();
        let first_node = 168 + 16;
        for mutation in [20usize, first_node + 7] {
            let mut bad = encoded.clone();
            bad[mutation] = 1;
            assert!(decode(&bad).is_err());
        }
        let mut trailing = encoded.clone();
        trailing.push(0);
        assert!(decode(&trailing).is_err());
        assert!(decode(&encoded[..encoded.len() - 1]).is_err());
    }

    #[test]
    fn decoder_rejects_endian_total_and_each_section_length_mismatch() {
        let encoded = encode(&fixture()).unwrap();
        let header_u64 =
            |offset: usize| u64::from_le_bytes(encoded[offset..offset + 8].try_into().unwrap());
        assert_eq!(
            header_u64(24),
            u64::from(RESIDENT_BAB_TOPOLOGY_HEADER_BYTES_V1)
                + header_u64(32)
                + header_u64(40)
                + header_u64(48)
                + header_u64(56)
        );
        for offset in [16usize, 24, 32, 40, 48, 56] {
            let mut bad = encoded.clone();
            bad[offset] ^= 1;
            assert!(
                decode(&bad).is_err(),
                "offset {offset} must be identity-bearing"
            );
        }
    }

    #[test]
    fn graph_names_are_unique_and_identity_bearing() {
        let original = fixture();
        let original_bytes = encode(&original).unwrap();
        let mut renamed = original.clone();
        renamed.nodes[1].name = "relu_renamed".to_string();
        let renamed_bytes = encode(&renamed).unwrap();
        assert_ne!(renamed_bytes, original_bytes);
        assert_eq!(decode(&renamed_bytes).unwrap(), renamed);

        let mut duplicate = original;
        duplicate.nodes[1].name = duplicate.nodes[0].name.clone();
        assert_producer_and_decoder_reject(&duplicate);
    }

    #[test]
    fn producer_and_decoder_reject_coverage_connectivity_and_shape_drift() {
        let mut duplicate_layer = fixture();
        duplicate_layer.layers[0].node_id = 0;
        assert_producer_and_decoder_reject(&duplicate_layer);

        let mut broken_chain = fixture();
        broken_chain.nodes[2].inputs[0] = 0;
        assert_producer_and_decoder_reject(&broken_chain);

        let mut unused = fixture();
        let mut output = unused.nodes.pop().unwrap();
        unused.nodes.push(ResidentBabNodeV1 {
            id: 2,
            name: "unused_linear".to_string(),
            kind: ResidentBabNodeKindV1::Linear,
            inputs: vec![1],
            relu_preactivation_node_id: None,
            output_shape: vec![2],
            output_values: 2,
        });
        output.id = 3;
        unused.nodes.push(output);
        unused.output_node_id = 3;
        unused.layers[0].node_id = 3;
        assert_producer_and_decoder_reject(&unused);

        let mut bad_frontier = fixture();
        bad_frontier.segments[0].frontier_abs.len = 1;
        bad_frontier.layers[0].node_abs.start = 1;
        bad_frontier.layers[1].node_abs = range(1, 2);
        bad_frontier.layers[2].node_abs.start = 3;
        bad_frontier.families.abs = 3;
        assert_producer_and_decoder_reject(&bad_frontier);

        let mut bad_relu_width = fixture();
        bad_relu_width.layers[1].activation = range(0, 7);
        bad_relu_width.layers[1].beta = range(0, 1);
        bad_relu_width.layers[1].node_abs = range(2, 1);
        bad_relu_width.layers[1].geometry[0] = 1;
        bad_relu_width.layers[2].activation.start = 7;
        bad_relu_width.layers[2].beta.start = 1;
        bad_relu_width.layers[2].node_abs.start = 3;
        bad_relu_width.families.activation = 7;
        bad_relu_width.families.beta = 1;
        bad_relu_width.families.abs = 3;
        assert_producer_and_decoder_reject(&bad_relu_width);

        let mut bad_linear_width = fixture();
        bad_linear_width.layers[0].parameters.len = 4;
        bad_linear_width.layers[0].geometry[1] = 1;
        bad_linear_width.layers[1].parameters.start = 4;
        bad_linear_width.layers[2].parameters = range(4, 6);
        bad_linear_width.families.parameters = 10;
        assert_producer_and_decoder_reject(&bad_linear_width);

        // Graph Linear contracts the last axis and preserves any prefix. V1's
        // flat descriptor deliberately admits only exact rank-1 [in] -> [out],
        // so matching total products cannot substitute for exact shapes.
        let nd_linear = ResidentBabTopologyV1 {
            input_shape: vec![2, 3],
            output_shape: vec![2, 2],
            output_node_id: 0,
            nodes: vec![ResidentBabNodeV1 {
                id: 0,
                name: "nd_linear_counterexample".to_string(),
                kind: ResidentBabNodeKindV1::Linear,
                inputs: vec![RESIDENT_BAB_NETWORK_INPUT_ID_V1],
                relu_preactivation_node_id: None,
                output_shape: vec![2, 2],
                output_values: 4,
            }],
            segments: vec![ResidentBabSegmentV1 {
                id: 0,
                kind: ResidentBabSegmentKindV1::Chain,
                first_layer: 0,
                main_layer_count: 1,
                projection_layer_count: 0,
                frontier_node_id: RESIDENT_BAB_NETWORK_INPUT_ID_V1,
                merge_node_id: None,
                frontier_branch: ResidentBabFrontierBranchV1::Main,
                frontier_abs: range(0, 6),
            }],
            layers: vec![ResidentBabLayerV1 {
                ordinal: 0,
                kind: ResidentBabLayerKindV1::Linear,
                branch: ResidentBabLayerBranchV1::Main,
                segment_id: 0,
                node_id: 0,
                parameters: range(0, 24),
                certified_errors: range(0, 2),
                activation: range(0, 0),
                beta: range(0, 0),
                node_abs: range(6, 0),
                geometry: [4, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            }],
            relu_count: 0,
            families: ResidentBabFamilyLengthsV1 {
                parameters: 24,
                certified_errors: 2,
                activation: 0,
                beta: 0,
                abs: 6,
                box_values: 6,
                cached_la: 0,
                topology_metadata: 0,
            },
        };
        assert_producer_and_decoder_reject(&nd_linear);

        let mut bad_conv_formula = conv_fixture();
        bad_conv_formula.layers[2].geometry[8] = 2;
        assert_producer_and_decoder_reject(&bad_conv_formula);

        let mut bad_conv_shape = conv_fixture();
        bad_conv_shape.layers[2].geometry[8] = 2;
        bad_conv_shape.layers[2].geometry[10] = 2;
        assert_producer_and_decoder_reject(&bad_conv_shape);

        let mut bad_chain_boundary = residual_fixture();
        bad_chain_boundary.segments[0].frontier_node_id = 1;
        assert_producer_and_decoder_reject(&bad_chain_boundary);

        let mut neither_identity_input = residual_fixture();
        neither_identity_input.nodes[2].inputs = vec![1, 0];
        assert_producer_and_decoder_reject(&neither_identity_input);

        let mut both_identity_inputs = residual_fixture();
        both_identity_inputs.nodes[2].inputs = vec![
            RESIDENT_BAB_NETWORK_INPUT_ID_V1,
            RESIDENT_BAB_NETWORK_INPUT_ID_V1,
        ];
        assert_producer_and_decoder_reject(&both_identity_inputs);

        let mut swapped_projection = projection_fixture();
        swapped_projection.nodes[2].inputs.swap(0, 1);
        assert_producer_and_decoder_reject(&swapped_projection);

        let mut broken_structural_seam = structural_seam_fixture();
        broken_structural_seam.nodes[4].inputs[0] = 2;
        assert_producer_and_decoder_reject(&broken_structural_seam);
    }

    #[test]
    fn independent_decoder_rejects_raw_tags_reserved_utf8_caps_and_overflow() {
        let encoded = manual_fixture_bytes();
        let first_node = RESIDENT_BAB_TOPOLOGY_HEADER_BYTES_V1 as usize + 16;
        let node_bytes = 155usize;
        let first_segment = first_node + node_bytes;
        let first_layer = first_segment + SEGMENT_BYTES;
        for offset in [
            first_node + 4,
            first_segment + 4,
            first_segment + 5,
            first_layer + 4,
            first_layer + 5,
        ] {
            let mut bad = encoded.clone();
            bad[offset] = u8::MAX;
            assert!(decode(&bad).is_err(), "unknown tag at byte {offset}");
        }
        for offset in [
            20usize,
            92,
            160,
            first_node + 7,
            first_node + 16,
            first_node + 20,
            first_segment + 6,
            first_segment + 28,
            first_layer + 6,
            first_layer + 16,
        ] {
            let mut bad = encoded.clone();
            bad[offset] = 1;
            assert!(decode(&bad).is_err(), "reserved byte at {offset}");
        }
        let mut invalid_utf8 = encoded;
        invalid_utf8[first_node + NODE_FIXED_BYTES] = u8::MAX;
        assert!(decode(&invalid_utf8).is_err());

        assert!(checked_shape(&[GPU_BAB_BOUND_MAX_ARENA_VALUES as u64 + 1], "test shape").is_err());
        assert!(
            decode_shape_product(&[GPU_BAB_BOUND_MAX_ARENA_VALUES as u64 + 1], "test shape")
                .is_err()
        );

        let mut over_cap = manual_fixture_bytes();
        over_cap[96..104]
            .copy_from_slice(&(GPU_BAB_BOUND_MAX_ARENA_VALUES as u64 + 1).to_le_bytes());
        assert!(decode(&over_cap).is_err());

        let mut overflow_layer = conv_fixture();
        overflow_layer.layers[2].geometry = [
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            1,
            1,
            0,
            0,
            1,
            1,
            u32::MAX,
            u32::MAX,
            0,
        ];
        assert_producer_and_decoder_reject(&overflow_layer);
    }

    #[test]
    fn direct_network_input_relu_round_trips_with_present_sentinel() {
        let topology = ResidentBabTopologyV1 {
            input_shape: vec![2],
            output_shape: vec![2],
            output_node_id: 1,
            nodes: vec![
                ResidentBabNodeV1 {
                    id: 0,
                    name: "relu_input".to_string(),
                    kind: ResidentBabNodeKindV1::Relu,
                    inputs: vec![RESIDENT_BAB_NETWORK_INPUT_ID_V1],
                    relu_preactivation_node_id: Some(RESIDENT_BAB_NETWORK_INPUT_ID_V1),
                    output_shape: vec![2],
                    output_values: 2,
                },
                ResidentBabNodeV1 {
                    id: 1,
                    name: "linear_output".to_string(),
                    kind: ResidentBabNodeKindV1::Linear,
                    inputs: vec![0],
                    relu_preactivation_node_id: None,
                    output_shape: vec![2],
                    output_values: 2,
                },
            ],
            segments: vec![ResidentBabSegmentV1 {
                id: 0,
                kind: ResidentBabSegmentKindV1::Chain,
                first_layer: 0,
                main_layer_count: 2,
                projection_layer_count: 0,
                frontier_node_id: RESIDENT_BAB_NETWORK_INPUT_ID_V1,
                merge_node_id: None,
                frontier_branch: ResidentBabFrontierBranchV1::Main,
                frontier_abs: range(0, 2),
            }],
            layers: vec![
                ResidentBabLayerV1 {
                    ordinal: 0,
                    kind: ResidentBabLayerKindV1::Linear,
                    branch: ResidentBabLayerBranchV1::Main,
                    segment_id: 0,
                    node_id: 1,
                    parameters: range(0, 6),
                    certified_errors: range(0, 2),
                    activation: range(0, 0),
                    beta: range(0, 0),
                    node_abs: range(2, 0),
                    geometry: [2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                },
                ResidentBabLayerV1 {
                    ordinal: 1,
                    kind: ResidentBabLayerKindV1::Relu,
                    branch: ResidentBabLayerBranchV1::Main,
                    segment_id: 0,
                    node_id: 0,
                    parameters: range(6, 0),
                    certified_errors: range(2, 0),
                    activation: range(0, 13),
                    beta: range(0, 2),
                    node_abs: range(2, 2),
                    geometry: [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                },
            ],
            relu_count: 1,
            families: ResidentBabFamilyLengthsV1 {
                parameters: 6,
                certified_errors: 2,
                activation: 13,
                beta: 2,
                abs: 4,
                box_values: 2,
                cached_la: 0,
                topology_metadata: 0,
            },
        };
        let bytes = encode(&topology).unwrap();
        assert_eq!(decode(&bytes).unwrap(), topology);
    }

    #[test]
    fn decoder_does_not_accept_noncanonical_family_or_range_partition() {
        let encoded = encode(&fixture()).unwrap();
        let mut bad_total = encoded.clone();
        // Header activation family length starts at byte 112.
        bad_total[112..120].copy_from_slice(&9u64.to_le_bytes());
        assert!(decode(&bad_total).is_err());

        let mut bad_range = encoded;
        let shape_bytes = 16;
        let node_bytes = 155;
        let layer0 = 168 + shape_bytes + node_bytes + SEGMENT_BYTES;
        // layer0 parameter start follows its 20-byte fixed prefix.
        bad_range[layer0 + 20..layer0 + 28].copy_from_slice(&1u64.to_le_bytes());
        assert!(decode(&bad_range).is_err());
    }

    #[test]
    fn encoder_rejects_cached_la_and_range_drift_before_allocating_wire() {
        let mut topology = fixture();
        topology.families.cached_la = 1;
        assert!(encode(&topology).is_err());
        topology.families.cached_la = 0;
        topology.layers[1].beta.start = 1;
        assert!(encode(&topology).is_err());
    }

    #[test]
    fn encoder_charges_observed_capacity_and_checks_deadline_after_reserve() {
        let topology = fixture();
        let exact_len = length(&topology).unwrap();
        let mut check = |_| Ok(());
        let encoded = topology.encode(1 << 20, &mut check).unwrap();
        assert_eq!(encoded.bytes.len(), exact_len);
        assert!(encoded.adapter_host_retained_bytes >= encoded.bytes.capacity());
        assert!(encoded.adapter_host_peak_bytes >= encoded.adapter_host_retained_bytes);
        assert!(encoded.adapter_host_peak_bytes <= 1 << 20);

        let mut check = |_| Ok(());
        assert!(matches!(
            topology.encode(exact_len - 1, &mut check),
            Err(NyError::CpuMemoryExceeded { .. })
        ));

        let mut reserve_checks = 0usize;
        let mut check = |label: &'static str| {
            if label == "resident topology wire reserve" {
                reserve_checks += 1;
                return Err(NyError::DeadlineExceeded(
                    "injected topology reserve deadline".to_string(),
                ));
            }
            Ok(())
        };
        assert!(matches!(
            topology.encode(1 << 20, &mut check),
            Err(NyError::DeadlineExceeded(_))
        ));
        assert_eq!(reserve_checks, 1);
    }

    #[test]
    fn encoder_rejects_impossible_fixed_cap_without_callbacks() {
        let topology = fixture();
        let baseline = 4_096usize;
        let minimum = baseline + size_of::<ResidentBabEncodedTopologyV1>();
        for limit in [0, minimum - 1] {
            let mut callbacks = 0usize;
            let mut check = |_| {
                callbacks += 1;
                Ok(())
            };
            assert!(matches!(
                topology.encode_with_baseline(limit, baseline, &mut check),
                Err(NyError::CpuMemoryExceeded {
                    required_bytes,
                    budget_bytes,
                    ..
                }) if required_bytes == minimum && budget_bytes == limit
            ));
            assert_eq!(callbacks, 0);
        }

        let mut callbacks = 0usize;
        let mut check = |_| {
            callbacks += 1;
            Ok(())
        };
        assert!(matches!(
            topology.encode_with_baseline(usize::MAX, usize::MAX, &mut check),
            Err(NyError::InvalidSpec(_))
        ));
        assert_eq!(callbacks, 0);
    }

    #[test]
    fn decoder_accounts_baseline_cap_and_every_post_reserve_deadline() {
        let bytes = encode(&structural_seam_fixture()).unwrap();
        let mut check = |_| Ok(());
        let without = ResidentBabTopologyV1::decode(&bytes, 1 << 24, 0, &mut check).unwrap();
        let baseline = bytes.capacity() + 4_096;
        let mut labels = Vec::new();
        let mut check = |label: &'static str| {
            labels.push(label);
            Ok(())
        };
        let with = ResidentBabTopologyV1::decode(&bytes, 1 << 24, baseline, &mut check).unwrap();
        assert_eq!(
            with.adapter_host_peak_bytes,
            without.adapter_host_peak_bytes + baseline
        );
        assert_eq!(
            with.adapter_host_retained_bytes_after,
            without.adapter_host_retained_bytes_after + baseline
        );
        assert!(with.adapter_host_retained_bytes_after <= with.adapter_host_peak_bytes);
        assert!(with.adapter_host_peak_bytes <= 1 << 24);

        let reserve_labels = [
            "resident decoded input-shape reserve",
            "resident decoded output-shape reserve",
            "resident decoded node-table reserve",
            "resident decoded node-name reserve",
            "resident decoded node-input reserve",
            "resident decoded node-shape reserve",
            "resident decoded name-index reserve",
            "resident decoded coverage reserve",
            "resident decoded segment-table reserve",
            "resident decoded layer-table reserve",
        ];
        for &target in &reserve_labels {
            assert!(
                labels.contains(&target),
                "missing reserve checkpoint {target}"
            );
            let mut fired = false;
            let mut check = |label: &'static str| {
                if label == target && !fired {
                    fired = true;
                    return Err(NyError::DeadlineExceeded(format!(
                        "injected after {target}"
                    )));
                }
                Ok(())
            };
            assert!(matches!(
                ResidentBabTopologyV1::decode(&bytes, 1 << 24, baseline, &mut check),
                Err(NyError::DeadlineExceeded(_))
            ));
            assert!(fired);
        }

        let mut scan_hits = 0usize;
        let mut reserve_hits = 0usize;
        let mut check = |label: &'static str| {
            if label.contains("reserve") {
                reserve_hits += 1;
            }
            if label == "resident topology prospective node scan" {
                scan_hits += 1;
                if scan_hits == 2 {
                    return Err(NyError::DeadlineExceeded(
                        "injected prospective scan deadline".to_string(),
                    ));
                }
            }
            Ok(())
        };
        assert!(matches!(
            ResidentBabTopologyV1::decode(&bytes, 1 << 24, baseline, &mut check),
            Err(NyError::DeadlineExceeded(_))
        ));
        assert_eq!(scan_hits, 2);
        assert_eq!(reserve_hits, 0);

        let fixed = size_of::<ResidentBabDecodedTopologyV1>();
        let mut callbacks = 0usize;
        let mut check = |_| {
            callbacks += 1;
            Ok(())
        };
        assert!(matches!(
            ResidentBabTopologyV1::decode(&bytes, baseline + fixed - 1, baseline, &mut check),
            Err(NyError::CpuMemoryExceeded { .. })
        ));
        assert_eq!(callbacks, 0);

        let mut malformed = bytes.clone();
        malformed[32] ^= 1;
        let mut reserve_hits = 0usize;
        let mut check = |label: &'static str| {
            if label.contains("reserve") {
                reserve_hits += 1;
            }
            Ok(())
        };
        assert!(ResidentBabTopologyV1::decode(&malformed, 1 << 24, baseline, &mut check).is_err());
        assert_eq!(reserve_hits, 0);

        let mut check = |_| Ok(());
        let plan = scan_decode_plan_v1(&bytes, &mut check).unwrap();
        let mut observed = ResidentBabDecodeBudgetV1::begin(1 << 24, 0, 64).unwrap();
        observed.charge_hash_capacity::<&str, ()>(3, 7).unwrap();
        assert!(observed.peak_bytes > 64);
        let admitted_peak = without.adapter_host_peak_bytes;
        let mut check = |_| Ok(());
        assert!(matches!(
            ResidentBabTopologyV1::decode(&bytes, admitted_peak - 1, 0, &mut check),
            Err(NyError::CpuMemoryExceeded { .. })
        ));
        assert!(admitted_peak >= plan.nominal_host_bytes);

        let mut scaled = fixture();
        let template = scaled.nodes[0].clone();
        scaled.nodes.resize(1_025, template);
        let mut accounting_hits = 0usize;
        let mut check = |label: &'static str| {
            if label == "resident decoded retained accounting" {
                accounting_hits += 1;
                if accounting_hits == 2 {
                    return Err(NyError::DeadlineExceeded(
                        "injected retained-accounting deadline".to_string(),
                    ));
                }
            }
            Ok(())
        };
        assert!(matches!(
            decoded_topology_retained_bytes(&scaled, 0, &mut check),
            Err(NyError::DeadlineExceeded(_))
        ));
        assert_eq!(accounting_hits, 2);

        let marks = vec![1u8; 1_025];
        for audit in [
            encoder_coverage_is_exact
                as fn(&[u8], u8, &mut dyn FnMut(&'static str) -> Result<()>) -> Result<bool>,
            decoded_coverage_is_exact,
        ] {
            let mut hits = 0usize;
            let mut check = |label: &'static str| {
                if label.ends_with("coverage audit") {
                    hits += 1;
                    if hits == 2 {
                        return Err(NyError::DeadlineExceeded(
                            "injected coverage deadline".to_string(),
                        ));
                    }
                }
                Ok(())
            };
            assert!(matches!(
                audit(&marks, 1, &mut check),
                Err(NyError::DeadlineExceeded(_))
            ));
            assert_eq!(hits, 2);
        }
    }

    #[test]
    fn coverage_initialization_is_polled_without_reallocation() {
        for (label, final_label) in [
            (
                "resident topology coverage initialization",
                "resident topology coverage initialization final",
            ),
            (
                "resident decoded coverage initialization",
                "resident decoded coverage initialization final",
            ),
        ] {
            let count = WIRE_POLL_STRIDE + 1;
            let mut coverage = Vec::new();
            coverage.try_reserve_exact(count).unwrap();
            let capacity = coverage.capacity();
            let mut hits = 0usize;
            let mut check = |observed| {
                if observed == label {
                    hits += 1;
                    if hits == 2 {
                        return Err(NyError::DeadlineExceeded(
                            "injected coverage initialization deadline".to_string(),
                        ));
                    }
                }
                Ok(())
            };
            assert!(matches!(
                initialize_coverage_v1(&mut coverage, count, &mut check, label, final_label),
                Err(NyError::DeadlineExceeded(_))
            ));
            assert_eq!(hits, 2);
            assert_eq!(coverage.len(), WIRE_POLL_STRIDE);
            assert_eq!(coverage.capacity(), capacity);
        }
    }

    #[test]
    fn composition_cap_validation_is_bounded_and_polled() {
        let topology = fixture();
        let mut check = |_| Ok(());
        validate_composition_caps_v1(&topology, 4, &mut check).unwrap();

        let mut callbacks = 0usize;
        let mut check = |_| {
            callbacks += 1;
            Ok(())
        };
        assert!(validate_composition_caps_v1(
            &topology,
            GPU_BAB_BOUND_MAX_SPLIT_HISTORY_WORDS + 1,
            &mut check,
        )
        .is_err());
        assert_eq!(callbacks, 0);

        let mut scaled = topology;
        scaled.nodes.resize(1_025, scaled.nodes[0].clone());
        let mut hits = 0usize;
        let mut check = |label| {
            if label == "resident composition node cap validation" {
                hits += 1;
                if hits == 2 {
                    return Err(NyError::DeadlineExceeded(
                        "injected composition-cap deadline".to_string(),
                    ));
                }
            }
            Ok(())
        };
        assert!(matches!(
            validate_composition_caps_v1(&scaled, 4, &mut check),
            Err(NyError::DeadlineExceeded(_))
        ));
        assert_eq!(hits, 2);
    }

    #[test]
    fn typed_wire_length_preflight_is_polled_past_one_record_stride() {
        let topology = fixture();
        let mut check = |_| Ok(());
        assert_eq!(
            topology_wire_length_preflight_v1(&topology, &mut check).unwrap(),
            ResidentBabTopologyWireLengthPreflightV1::Encodable {
                encoded_bytes: length(&topology).unwrap(),
            }
        );

        let mut scaled = topology;
        let template = scaled.nodes[0].clone();
        scaled.nodes.clear();
        for index in 0..=WIRE_POLL_STRIDE {
            let mut node = template.clone();
            node.id = index as u32;
            node.name = format!("wire_length_{index}");
            scaled.nodes.push(node);
        }
        let mut stride_polls = 0usize;
        let mut check = |label| {
            if label == "resident topology length validation" {
                stride_polls += 1;
                if stride_polls == 2 {
                    return Err(NyError::DeadlineExceeded(
                        "injected wire-length deadline".to_string(),
                    ));
                }
            }
            Ok(())
        };
        assert!(matches!(
            topology_wire_length_preflight_v1(&scaled, &mut check),
            Err(NyError::DeadlineExceeded(_))
        ));
        assert_eq!(stride_polls, 2);
    }

    #[test]
    fn public_activation_tag_and_section_helpers_are_closed_and_checked() {
        assert_eq!(
            ResidentBabActivationVariantV1::from_tag_bits(0.0f32.to_bits()).unwrap(),
            ResidentBabActivationVariantV1::Ordinary
        );
        assert_eq!(
            ResidentBabActivationVariantV1::from_tag_bits(1.0f32.to_bits()).unwrap(),
            ResidentBabActivationVariantV1::DualAlpha
        );
        for value in [-0.0f32, 2.0, f32::NAN] {
            assert!(ResidentBabActivationVariantV1::from_tag_bits(value.to_bits()).is_err());
        }

        let sections = ResidentBabActivationSectionsV1::from_row(range(7, 19), 3).unwrap();
        assert_eq!(sections.tag_index, 7);
        assert_eq!(sections.pre_lower, range(8, 3));
        assert_eq!(sections.pre_upper, range(11, 3));
        assert_eq!(sections.section_0, range(14, 3));
        assert_eq!(sections.section_1, range(17, 3));
        assert_eq!(sections.section_2, range(20, 3));
        assert_eq!(sections.section_3, range(23, 3));
        assert!(ResidentBabActivationSectionsV1::from_row(range(0, 1), 0).is_err());
        assert!(ResidentBabActivationSectionsV1::from_row(range(0, 18), 3).is_err());
        assert!(ResidentBabActivationSectionsV1::from_row(range(u64::MAX, 7), 1).is_err());
        assert!(ResidentBabActivationSectionsV1::from_row(range(0, u64::MAX), u64::MAX).is_err());
    }

    #[test]
    fn encoder_accounting_is_absolute_with_a_nonzero_baseline() {
        let topology = fixture();
        let mut check = |_| Ok(());
        let without = topology
            .encode_with_baseline(1 << 20, 0, &mut check)
            .unwrap();
        let baseline = 4_096usize;
        let mut check = |_| Ok(());
        let with = topology
            .encode_with_baseline(1 << 20, baseline, &mut check)
            .unwrap();
        assert_eq!(
            with.adapter_host_peak_bytes,
            without.adapter_host_peak_bytes + baseline
        );
        assert_eq!(
            with.adapter_host_retained_bytes,
            without.adapter_host_retained_bytes + baseline
        );
    }
}
