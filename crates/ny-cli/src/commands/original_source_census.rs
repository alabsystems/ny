// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Staged allocation admission for the original twin ONNX surface.
//!
//! The existing source issuer's blanket 512-byte envelope remains unchanged.
//! This separate entry first walks borrowed protobuf wire without allocating
//! a graph, then prices the actual recognized message/vector/string/payload
//! geometry. It reconciles visible decoded and extracted capacities, and
//! keeps the complete decode/extraction claim until the genuine compiler owns
//! the transferred parameters. It grants no property/publication authority.
//!
//! This bounded wire surface matches the heap-bearing fields in ny-onnx's
//! ModelProto and the inference-only source guard. Unknown ordinary scalar or
//! length-delimited fields allocate nothing in that decoder and are skipped.
//! Groups, deep recursion and unsupported heap-bearing fields decline this
//! entry; callers may select the separately admitted historical envelope.
//!
//! The Vec slack rule is a source qualification premise for the pinned
//! Trust/std and prost0.14.4 implementation, not a portable Rust API promise:
//! https://github.com/tokio-rs/prost/blob/v0.14.4/prost/src/encoding.rs
//! https://github.com/rust-lang/rust/blob/1.95.0/library/alloc/src/raw_vec/mod.rs
//! https://doc.rust-lang.org/std/vec/struct.Vec.html#guarantees
//! A changed decoder or allocator lane requires requalification or the old
//! envelope. Visible capacities are checked independently after decoding.

use std::{mem::size_of, time::Instant};

use anyhow::{anyhow, bail, ensure, Result};
use ny_onnx::onnx_proto::{
    tensor_shape_proto::{dimension::Value, Dimension},
    AttributeProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto,
    TensorShapeProto, TensorTypeProto, TypeProto, ValueInfoProto,
};
use ny_propagate::{
    certify::work::{LeaseClass, WorkBudget, WorkLease, WorkRequest},
    margin_row::{PreparedRootCompilation, TwinOpSpec, TwinSpec},
};
use prost::Message;
use sha2::{Digest, Sha256};

const META: usize = 8192;
const POLL: usize = 256;
const MAX_DEPTH: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct OriginalSourceAllocationReceipt {
    pub(super) source_bytes: usize,
    pub(super) decoded_capacity_bound: usize,
    pub(super) decoded_visible_bytes: usize,
    pub(super) extraction_capacity_bound: usize,
    pub(super) extracted_visible_bytes: usize,
    pub(super) decode_scratch_bytes: usize,
    pub(super) message_occurrences: usize,
    pub(super) node_occurrences: usize,
    pub(super) parameter_cells: usize,
}

/// The source was checked and extracted from these exact immutable bytes, but
/// it has not been attached to a VNN-LIB objective or published a result.
/// The CLI issuer must retain its source snapshot and authenticate that later
/// objective; the public compiled owner remains a conditional declaration.
pub(super) struct StagedOriginalSourceCompilation {
    compiled: PreparedRootCompilation,
    source_sha256: [u8; 32],
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "retain the staged allocation audit receipt and owner layout"
        )
    )]
    receipt: OriginalSourceAllocationReceipt,
    deadline: Instant,
}

impl StagedOriginalSourceCompilation {
    pub(super) fn source_sha256(&self) -> &[u8; 32] {
        &self.source_sha256
    }
    #[cfg(test)]
    pub(super) fn receipt(&self) -> OriginalSourceAllocationReceipt {
        self.receipt
    }
    pub(super) fn compiled(&self) -> &PreparedRootCompilation {
        &self.compiled
    }
    pub(super) fn into_compilation(
        self,
        budget: &WorkBudget,
        requested: Option<Instant>,
    ) -> Result<PreparedRootCompilation> {
        let at = requested.map_or(self.deadline, |d| d.min(self.deadline));
        Ok(self.compiled.restrict_deadline(budget, Some(at))?)
    }
}

/// Prepay and compile from one borrowed immutable original snapshot. All
/// completed decoded graphs and extractor-only metadata cease to be live
/// before the decoder claim releases. The returned model keeps its original
/// persistent claim; proof construction uses that owner without recompiling.
pub(super) fn compile_staged_original_source(
    source: &[u8],
    budget: &WorkBudget,
    requested: Instant,
) -> Result<StagedOriginalSourceCompilation> {
    compile_inner(source, budget, requested, None)
}

fn compile_inner(
    source: &[u8],
    budget: &WorkBudget,
    requested: Instant,
    mut after_decode: Option<&mut dyn FnMut()>,
) -> Result<StagedOriginalSourceCompilation> {
    checkpoint(budget, requested)?;
    ensure!(
        budget.memory_authority().is_some(),
        "staged source requires original shared memory authority"
    );
    let census = census_source(source, budget, requested)?;
    // Borrowed original bytes also overlap this transaction. Counting their
    // span conservatively here prevents an unpriced slice from creating a
    // decoder credit; an issuer's separate source claim stays independently
    // live on the same ledger.
    let scratch = add(
        source.len(),
        add(census.decoded_bytes, add(census.extraction_bytes()?, META)?)?,
    )?;
    let mut lease = budget.lease_for(
        LeaseClass::Fallback,
        WorkRequest {
            scratch_bytes: scratch,
            work: census.production_work(source.len())?,
            ..WorkRequest::default()
        },
    )?;
    let at = lease.deadline().map_or(requested, |d| d.min(requested));
    checkpoint(budget, at)?;
    let decoded = ModelProto::decode(source)?;
    checkpoint(budget, at)?;
    let visible_decoded = decoded_visible_bytes(&decoded, budget, at)?;
    ensure!(
        visible_decoded <= census.decoded_bytes,
        "actual original decoder capacity exceeds admitted wire census"
    );
    if let Some(observe) = after_decode.as_mut() {
        observe();
    }
    checkpoint(budget, at)?;
    drop(decoded);
    // The existing source guard and extractor are the sole mathematical
    // original-to-declaration path. They read these same bytes. Their graphs
    // run sequentially, under the still-held full decoded/workspace envelope.
    super::prepared_root_source::authenticate_original_twin_surface(source, budget, at)?;
    let (spec, names) =
        super::margin_row_bab::extract_original_twin_spec_named_bytes(source, budget, at)
            .ok_or_else(|| anyhow!("original source outside checked twin extractor"))?;
    checkpoint(budget, at)?;
    let visible_extracted = extracted_visible_bytes(&spec, &names, budget, at)?;
    ensure!(
        visible_extracted <= census.extraction_bytes()?,
        "actual original extraction capacity exceeds admitted wire census"
    );
    drop(names);
    let mut sha = Sha256::new();
    for block in source.chunks(4096) {
        checkpoint(budget, at)?;
        sha.update(block);
    }
    let source_sha256 = sha.finalize().into();
    // This is the actual moved original declaration, with the decoder/source
    // overlap still charged while its parameter ownership changes hands.
    let compiled = PreparedRootCompilation::compile_declared_twin(spec, budget, Some(at))?;
    compiled.checkpoint(budget, Some(at))?;
    let final_at = compiled.deadline(Some(at))?.unwrap_or(at).min(at);
    let receipt = OriginalSourceAllocationReceipt {
        source_bytes: source.len(),
        decoded_capacity_bound: census.decoded_bytes,
        decoded_visible_bytes: visible_decoded,
        extraction_capacity_bound: census.extraction_bytes()?,
        extracted_visible_bytes: visible_extracted,
        decode_scratch_bytes: scratch,
        message_occurrences: census.messages,
        node_occurrences: census.nodes,
        parameter_cells: census.parameter_cells,
    };
    lease.complete()?;
    drop(lease);
    compiled.checkpoint(budget, Some(final_at))?;
    Ok(StagedOriginalSourceCompilation {
        compiled,
        source_sha256,
        receipt,
        deadline: final_at,
    })
}

#[derive(Clone, Copy, Default)]
struct Census {
    decoded_bytes: usize,
    messages: usize,
    nodes: usize,
    tensors: usize,
    opsets: usize,
    string_bytes: usize,
    parameter_cells: usize,
    fields: usize,
    numeric_cells: usize,
    byte_payload_bytes: usize,
    node_inputs: usize,
    attribute_pairs: usize,
    materialization_cells: usize,
}
impl Census {
    fn extraction_bytes(self) -> Result<usize> {
        // Ten binary64 lanes per original FLOAT cell cover nominal/folded
        // weights, transpose/cancellation scratch, channel zeros/error balls
        // and their complete original/extracted overlap. Node metadata pays
        // the consumer maps, tensor IDs, shapes, op Vec growth and site names;
        // tensor metadata also prices initializer/shape maps. The production
        // compiler independently owns/prices all new compiled arrays.
        add(
            mul(self.parameter_cells.max(self.materialization_cells), 80)?,
            add(
                mul(self.nodes, 4096)?,
                add(
                    mul(self.tensors, 2048)?,
                    add(mul(self.string_bytes, 8)?, META)?,
                )?,
            )?,
        )
    }

    fn metadata_work(self) -> Result<usize> {
        // Hash-table probes are not assumed constant time. Sixty-four copies
        // of every source name at the maximum live topology-table population
        // cover the guard/extractor maps, consumers and this census's bounded
        // initializer lookup. Duplicate attribute and normalized opset-domain
        // detection are quadratic. Count every wire opset occurrence before
        // the original guard may compare an unused import with its predecessors.
        let population = add(
            add(self.tensors, self.nodes)?,
            add(add(self.node_inputs, self.opsets)?, 1)?,
        )?;
        add(
            mul(mul(self.string_bytes, population)?, 64)?,
            add(
                mul(
                    add(self.attribute_pairs, mul(self.opsets, self.opsets)?)?,
                    128,
                )?,
                add(mul(self.nodes, 4096)?, mul(self.tensors, 1024)?)?,
            )?,
        )
    }

    fn production_work(self, source_bytes: usize) -> Result<u64> {
        // The production transaction executes three complete prost decodes:
        // reconciliation, original-surface authentication, and the unchanged
        // extractor. Varints are <=10 bytes; repeated numeric pushes include
        // geometric copying. Vec-backed byte fields may copy twice. Strings
        // additionally undergo UTF8/name inspection. All occurrences count,
        // including overwritten protobuf fields and unused original tensors.
        let decode = add(
            add(mul(self.fields, 256)?, mul(self.messages, 128)?)?,
            add(
                mul(self.numeric_cells, 128)?,
                add(mul(self.byte_payload_bytes, 4)?, mul(self.string_bytes, 8)?)?,
            )?,
        )?;
        // Per parameter reference: exact f32 widening, optional transpose,
        // zero initialization, finite scans and the Conv/Gemm BN fold's
        // actual scalar operations. Reused weights/BN/shape constants count
        // each time. Compiler arrays/walks are paid separately by its owner.
        let extraction = mul(self.materialization_cells, 96)?;
        let work = add(
            mul(decode, 3)?,
            add(
                extraction,
                add(self.metadata_work()?, add(mul(source_bytes, 64)?, META)?)?,
            )?,
        )?;
        u64::try_from(work).map_err(Into::into)
    }
}

#[derive(Clone, Copy)]
enum Kind {
    Model,
    Graph,
    Node,
    Attribute,
    Tensor,
    Opset,
    ValueInfo,
    Type,
    TensorType,
    Shape,
    Dimension,
}
impl Kind {
    fn header(self) -> usize {
        match self {
            Self::Model => size_of::<ModelProto>(),
            Self::Graph => size_of::<GraphProto>(),
            Self::Node => size_of::<NodeProto>(),
            Self::Attribute => size_of::<AttributeProto>(),
            Self::Tensor => size_of::<TensorProto>(),
            Self::Opset => size_of::<OperatorSetIdProto>(),
            Self::ValueInfo => size_of::<ValueInfoProto>(),
            Self::Type => size_of::<TypeProto>(),
            Self::TensorType => size_of::<TensorTypeProto>(),
            Self::Shape => size_of::<TensorShapeProto>(),
            Self::Dimension => size_of::<Dimension>(),
        }
    }
}

enum Field {
    Scalar(u8),
    String,
    Bytes,
    Strings,
    Numeric { width: usize, wire: u8 },
    Message(Kind),
    Forbidden,
    Unknown,
}
fn field(kind: Kind, tag: u32) -> Field {
    use Field::{Bytes, Forbidden, Message, Numeric, Scalar, String, Strings, Unknown};
    match (kind, tag) {
        (Kind::Model, 1 | 5) => Scalar(0),
        (Kind::Model, 2 | 3 | 4 | 6) => String,
        (Kind::Model, 7) => Message(Kind::Graph),
        (Kind::Model, 8) => Message(Kind::Opset),
        (Kind::Graph, 1) => Message(Kind::Node),
        (Kind::Graph, 2) => String,
        (Kind::Graph, 5) => Message(Kind::Tensor),
        (Kind::Graph, 11..=13) => Message(Kind::ValueInfo),
        (Kind::Graph, 15) => Forbidden,
        (Kind::Node, 1 | 2) => Strings,
        (Kind::Node, 3 | 4 | 7) => String,
        (Kind::Node, 5) => Message(Kind::Attribute),
        (Kind::Attribute, 1 | 13 | 21) => String,
        (Kind::Attribute, 2) => Scalar(5),
        (Kind::Attribute, 3 | 20) => Scalar(0),
        (Kind::Attribute, 4) => Bytes,
        (Kind::Attribute, 5) => Message(Kind::Tensor),
        (Kind::Attribute, 8) => Numeric { width: 8, wire: 0 },
        (Kind::Attribute, 6 | 7 | 9 | 10 | 11 | 14 | 15 | 22 | 23) => Forbidden,
        (Kind::Tensor, 1 | 7) => Numeric { width: 8, wire: 0 },
        (Kind::Tensor, 2 | 14) => Scalar(0),
        (Kind::Tensor, 4) => Numeric { width: 4, wire: 5 },
        (Kind::Tensor, 8) => String,
        (Kind::Tensor, 9) => Bytes,
        (Kind::Tensor, 3 | 5 | 6 | 10 | 11 | 13) => Forbidden,
        (Kind::Opset, 1) => String,
        (Kind::Opset, 2) => Scalar(0),
        (Kind::ValueInfo, 1) => String,
        (Kind::ValueInfo, 2) => Message(Kind::Type),
        (Kind::Type, 1) => Message(Kind::TensorType),
        (Kind::TensorType, 1) => Scalar(0),
        (Kind::TensorType, 2) => Message(Kind::Shape),
        (Kind::Shape, 1) => Message(Kind::Dimension),
        (Kind::Dimension, 1) => Scalar(0),
        (Kind::Dimension, 2) => String,
        _ => Unknown,
    }
}

struct Scan<'a> {
    budget: &'a WorkBudget,
    at: Instant,
    steps: usize,
    census: Census,
    work: &'a mut WorkLease,
}
impl Scan<'_> {
    fn poll(&mut self) -> Result<()> {
        if self.steps.is_multiple_of(POLL) {
            checkpoint(self.budget, self.at)?;
            self.work.reserve_more_work(charged(POLL, 384)?)?;
            self.at = self.work.deadline().map_or(self.at, |d| d.min(self.at));
        }
        self.steps = add(self.steps, 1)?;
        Ok(())
    }
    fn walk(&mut self, kind: Kind, bytes: &[u8], depth: usize) -> Result<()> {
        ensure!(depth <= MAX_DEPTH, "original protobuf census depth refused");
        self.census.messages = add(self.census.messages, 1)?;
        // prost0.14 repeated messages/scalars use Vec::push. Four slots per
        // occurrence pay geometric growth and the four-element minimum; the
        // two-copy byte envelope additionally pays replacement overlap.
        self.census.decoded_bytes = add(self.census.decoded_bytes, mul(kind.header(), 4)?)?;
        if matches!(kind, Kind::Node) {
            self.census.nodes = add(self.census.nodes, 1)?;
        }
        if matches!(kind, Kind::Tensor) {
            self.census.tensors = add(self.census.tensors, 1)?;
        }
        if matches!(kind, Kind::Opset) {
            self.census.opsets = add(self.census.opsets, 1)?;
        }
        let mut cursor = WireCursor::new(bytes);
        let mut tensor_type = 0;
        let mut raw_bytes = 0usize;
        let mut float_cells = 0usize;
        let mut attributes = 0usize;
        loop {
            self.poll()?;
            let Some((tag, value)) = cursor.next()? else {
                break;
            };
            self.census.fields = add(self.census.fields, 1)?;
            if matches!(kind, Kind::Node) && tag == 1 {
                self.census.node_inputs = add(self.census.node_inputs, 1)?;
            }
            if matches!(kind, Kind::Node) && tag == 5 {
                attributes = add(attributes, 1)?;
            }
            match field(kind, tag) {
                Field::Scalar(wire) => {
                    ensure!(value.wire == wire, "original scalar protobuf wire mismatch");
                    if matches!(kind, Kind::Tensor) && tag == 2 {
                        tensor_type = value.scalar.unwrap_or(0) as i32;
                    }
                }
                Field::String | Field::Strings | Field::Bytes => {
                    ensure!(value.wire == 2, "original bytes protobuf wire mismatch");
                    let payload = value.payload;
                    if matches!(field(kind, tag), Field::String | Field::Strings) {
                        self.work.reserve_more_work(charged(payload.len(), 8)?)?;
                        std::str::from_utf8(payload)?;
                        self.census.string_bytes = add(self.census.string_bytes, payload.len())?;
                    } else {
                        self.census.byte_payload_bytes =
                            add(self.census.byte_payload_bytes, payload.len())?;
                    }
                    if matches!(field(kind, tag), Field::Strings) {
                        self.census.decoded_bytes =
                            add(self.census.decoded_bytes, mul(size_of::<String>(), 4)?)?;
                    }
                    self.census.decoded_bytes =
                        add(self.census.decoded_bytes, add(mul(payload.len(), 2)?, 16)?)?;
                    if matches!(kind, Kind::Tensor) && tag == 9 {
                        raw_bytes = add(raw_bytes, payload.len())?;
                    }
                }
                Field::Numeric { width, wire } => {
                    let count = if value.wire == wire {
                        1
                    } else {
                        ensure!(value.wire == 2, "original packed protobuf wire mismatch");
                        if wire == 0 {
                            let mut position = 0;
                            let mut count = 0usize;
                            while position < value.payload.len() {
                                self.poll()?;
                                read_varint(value.payload, &mut position)?;
                                count = add(count, 1)?;
                            }
                            count
                        } else {
                            ensure!(
                                value.payload.len() % width == 0,
                                "partial packed original numeric payload"
                            );
                            value.payload.len() / width
                        }
                    };
                    if count != 0 {
                        self.census.numeric_cells = add(self.census.numeric_cells, count)?;
                        self.census.decoded_bytes =
                            add(self.census.decoded_bytes, mul(count, mul(width, 4)?)?)?;
                    }
                    if matches!(kind, Kind::Tensor) && tag == 4 {
                        float_cells = add(float_cells, count)?;
                    }
                }
                Field::Message(child) => {
                    ensure!(value.wire == 2, "original message protobuf wire mismatch");
                    self.walk(child, value.payload, add(depth, 1)?)?;
                }
                Field::Forbidden => {
                    bail!("original heap-bearing protobuf field outside bounded census")
                }
                Field::Unknown => (),
            }
        }
        self.census.attribute_pairs =
            add(self.census.attribute_pairs, mul(attributes, attributes)?)?;
        if matches!(kind, Kind::Tensor) {
            ensure!(
                matches!(tensor_type, 1 | 7),
                "original tensor type outside bounded source census"
            );
            if tensor_type == 1 {
                ensure!(
                    raw_bytes.is_multiple_of(4),
                    "partial original float raw payload"
                );
                self.census.parameter_cells = add(
                    self.census.parameter_cells,
                    add(float_cells, raw_bytes / 4)?,
                )?;
            }
        }
        Ok(())
    }
}

fn census_source(source: &[u8], budget: &WorkBudget, requested: Instant) -> Result<Census> {
    checkpoint(budget, requested)?;
    // Constant stack metadata is paid first. Before every block of keys or
    // packed varints, Scan reserves that block's work on this same ledger.
    // Raw payload spans are skipped, never charged as thousands of key reads.
    let mut lease = budget.lease_for(
        LeaseClass::Fallback,
        WorkRequest {
            scratch_bytes: META,
            work: u64::try_from(META)?,
            ..WorkRequest::default()
        },
    )?;
    let at = lease.deadline().map_or(requested, |d| d.min(requested));
    let mut scan = Scan {
        budget,
        at,
        steps: 0,
        census: Census::default(),
        work: &mut lease,
    };
    scan.walk(Kind::Model, source, 0)?;
    checkpoint(budget, scan.at)?;
    let at = scan.at;
    let mut census = scan.census;
    lease.complete()?;
    drop(lease);
    census.materialization_cells = materialization_cells(source, census, budget, at)?;
    Ok(census)
}

struct ParameterCells<'a> {
    name: &'a str,
    cells: usize,
}

fn tensor_materialization_cells(bytes: &[u8], budget: &WorkBudget, at: Instant) -> Result<usize> {
    let mut cursor = WireCursor::new(bytes);
    let mut cells = 0usize;
    let mut steps = 0usize;
    while let Some((tag, value)) = cursor.next()? {
        poll(budget, at, steps)?;
        steps = add(steps, 1)?;
        match tag {
            4 => {
                cells = add(
                    cells,
                    if value.wire == 2 {
                        value.payload.len() / 4
                    } else {
                        1
                    },
                )?
            }
            7 => {
                let mut position = 0;
                let mut count = 0;
                if value.wire == 2 {
                    while position < value.payload.len() {
                        poll(budget, at, steps)?;
                        steps = add(steps, 1)?;
                        read_varint(value.payload, &mut position)?;
                        count = add(count, 1)?;
                    }
                } else {
                    count = 1;
                }
                cells = add(cells, count)?;
            }
            // Divide by four even for INT64: this conservatively pays both
            // shape words and original FLOAT parameters on a common scale.
            9 => cells = add(cells, value.payload.len() / 4)?,
            _ => (),
        }
    }
    Ok(cells)
}

fn last_string<'a>(
    bytes: &'a [u8],
    wanted: u32,
    budget: &WorkBudget,
    at: Instant,
) -> Result<Option<&'a str>> {
    let mut cursor = WireCursor::new(bytes);
    let mut result = None;
    let mut steps = 0usize;
    while let Some((tag, value)) = cursor.next()? {
        poll(budget, at, steps)?;
        steps = add(steps, 1)?;
        if tag == wanted {
            result = Some(std::str::from_utf8(value.payload)?);
        }
    }
    Ok(result)
}

fn collect_parameter_cells<'a>(
    kind: Kind,
    bytes: &'a [u8],
    entries: &mut Vec<ParameterCells<'a>>,
    budget: &WorkBudget,
    at: Instant,
    depth: usize,
) -> Result<()> {
    ensure!(
        depth <= MAX_DEPTH,
        "original reference census depth refused"
    );
    if matches!(kind, Kind::Tensor) {
        if let Some(name) = last_string(bytes, 8, budget, at)? {
            ensure!(
                entries.len() < entries.capacity(),
                "original reference index exceeds admission"
            );
            entries.push(ParameterCells {
                name,
                cells: tensor_materialization_cells(bytes, budget, at)?,
            });
        }
    }
    let constant =
        matches!(kind, Kind::Node) && last_string(bytes, 4, budget, at)? == Some("Constant");
    let output = if constant {
        last_string(bytes, 2, budget, at)?
    } else {
        None
    };
    let mut cursor = WireCursor::new(bytes);
    let mut i = 0;
    while let Some((tag, value)) = cursor.next()? {
        poll(budget, at, i)?;
        i = add(i, 1)?;
        if let Field::Message(child) = field(kind, tag) {
            collect_parameter_cells(child, value.payload, entries, budget, at, add(depth, 1)?)?;
            if constant && tag == 5 {
                let mut attr = WireCursor::new(value.payload);
                let mut attribute_steps = 0usize;
                while let Some((tag, tensor)) = attr.next()? {
                    poll(budget, at, attribute_steps)?;
                    attribute_steps = add(attribute_steps, 1)?;
                    if tag == 5 {
                        if let Some(name) = output {
                            ensure!(
                                entries.len() < entries.capacity(),
                                "original constant index exceeds admission"
                            );
                            entries.push(ParameterCells {
                                name,
                                cells: tensor_materialization_cells(tensor.payload, budget, at)?,
                            });
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

fn count_parameter_references(
    kind: Kind,
    bytes: &[u8],
    entries: &[ParameterCells<'_>],
    budget: &WorkBudget,
    at: Instant,
    depth: usize,
) -> Result<usize> {
    ensure!(
        depth <= MAX_DEPTH,
        "original reference census depth refused"
    );
    let mut cursor = WireCursor::new(bytes);
    let mut cells = 0;
    let mut i = 0;
    while let Some((tag, value)) = cursor.next()? {
        poll(budget, at, i)?;
        i = add(i, 1)?;
        if matches!(kind, Kind::Node) && tag == 1 {
            let name = std::str::from_utf8(value.payload)?;
            for (j, entry) in entries.iter().enumerate() {
                poll(budget, at, j)?;
                if entry.name == name {
                    cells = add(cells, entry.cells)?;
                }
            }
        }
        if let Field::Message(child) = field(kind, tag) {
            cells = add(
                cells,
                count_parameter_references(
                    child,
                    value.payload,
                    entries,
                    budget,
                    at,
                    add(depth, 1)?,
                )?,
            )?;
        }
    }
    Ok(cells)
}

fn materialization_cells(
    source: &[u8],
    census: Census,
    budget: &WorkBudget,
    requested: Instant,
) -> Result<usize> {
    let slots = mul(census.tensors, 2)?;
    let mut lease = budget.lease_for(
        LeaseClass::Fallback,
        WorkRequest {
            scratch_bytes: add(mul(slots, size_of::<ParameterCells<'_>>())?, META)?,
            work: u64::try_from(add(
                census.metadata_work()?,
                add(mul(census.fields, 512)?, mul(census.numeric_cells, 256)?)?,
            )?)?,
            ..WorkRequest::default()
        },
    )?;
    let at = lease.deadline().map_or(requested, |d| d.min(requested));
    let mut entries = Vec::new();
    entries.try_reserve_exact(slots)?;
    ensure!(
        entries.capacity() <= slots,
        "original reference index allocator exceeds admission"
    );
    collect_parameter_cells(Kind::Model, source, &mut entries, budget, at, 0)?;
    let cells = count_parameter_references(Kind::Model, source, &entries, budget, at, 0)?;
    checkpoint(budget, at)?;
    drop(entries);
    lease.complete()?;
    Ok(cells)
}

struct WireValue<'a> {
    wire: u8,
    scalar: Option<u64>,
    payload: &'a [u8],
}
struct WireCursor<'a> {
    bytes: &'a [u8],
    position: usize,
}
impl<'a> WireCursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }
    fn next(&mut self) -> Result<Option<(u32, WireValue<'a>)>> {
        if self.position == self.bytes.len() {
            return Ok(None);
        }
        let key = read_varint(self.bytes, &mut self.position)?;
        let tag = u32::try_from(key >> 3)?;
        ensure!(
            tag != 0 && tag <= 0x1fff_ffff,
            "invalid original protobuf tag"
        );
        let wire = (key & 7) as u8;
        let scalar = if wire == 0 {
            Some(read_varint(self.bytes, &mut self.position)?)
        } else {
            None
        };
        let n = match wire {
            0 => 0,
            1 => 8,
            2 => usize::try_from(read_varint(self.bytes, &mut self.position)?)?,
            5 => 4,
            _ => bail!("original protobuf groups/reserved wire outside census"),
        };
        let end = add(self.position, n)?;
        let payload = self
            .bytes
            .get(self.position..end)
            .ok_or_else(|| anyhow!("truncated original protobuf payload"))?;
        self.position = end;
        Ok(Some((
            tag,
            WireValue {
                wire,
                scalar,
                payload,
            },
        )))
    }
}
fn read_varint(bytes: &[u8], position: &mut usize) -> Result<u64> {
    let mut value = 0u64;
    for shift in (0..70).step_by(7) {
        let byte = *bytes
            .get(*position)
            .ok_or_else(|| anyhow!("truncated original varint"))?;
        *position = add(*position, 1)?;
        ensure!(shift != 63 || byte <= 1, "overflowed original varint");
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
    }
    bail!("overflowed original varint")
}

fn decoded_visible_bytes(model: &ModelProto, budget: &WorkBudget, at: Instant) -> Result<usize> {
    let mut bytes = size_of::<ModelProto>();
    for value in [
        &model.producer_name,
        &model.producer_version,
        &model.domain,
        &model.doc_string,
    ] {
        bytes = add(bytes, value.capacity())?;
    }
    bytes = add(
        bytes,
        mul(
            model.opset_import.capacity(),
            size_of::<OperatorSetIdProto>(),
        )?,
    )?;
    for (i, opset) in model.opset_import.iter().enumerate() {
        poll(budget, at, i)?;
        bytes = add(bytes, opset.domain.capacity())?;
    }
    if let Some(graph) = model.graph.as_ref() {
        ensure!(
            graph.sparse_initializer.is_empty(),
            "unsupported original sparse storage"
        );
        bytes = add(bytes, add(size_of::<GraphProto>(), graph.name.capacity())?)?;
        for amount in [
            mul(graph.node.capacity(), size_of::<NodeProto>())?,
            mul(graph.initializer.capacity(), size_of::<TensorProto>())?,
            mul(graph.input.capacity(), size_of::<ValueInfoProto>())?,
            mul(graph.output.capacity(), size_of::<ValueInfoProto>())?,
            // ny-cli's ny-onnx dependency enables its default value-info
            // feature. Inspect the actual capacity, not just slice length.
            mul(graph.value_info.capacity(), size_of::<ValueInfoProto>())?,
        ] {
            bytes = add(bytes, amount)?;
        }
        for (i, tensor) in graph.initializer.iter().enumerate() {
            poll(budget, at, i)?;
            bytes = add(bytes, tensor_visible_bytes(tensor)?)?;
        }
        for (i, node) in graph.node.iter().enumerate() {
            poll(budget, at, i)?;
            for value in [&node.name, &node.op_type, &node.domain] {
                bytes = add(bytes, value.capacity())?;
            }
            for names in [&node.input, &node.output] {
                bytes = add(bytes, mul(names.capacity(), size_of::<String>())?)?;
                for (j, name) in names.iter().enumerate() {
                    poll(budget, at, j)?;
                    bytes = add(bytes, name.capacity())?;
                }
            }
            bytes = add(
                bytes,
                mul(node.attribute.capacity(), size_of::<AttributeProto>())?,
            )?;
            for (j, attribute) in node.attribute.iter().enumerate() {
                poll(budget, at, j)?;
                bytes = add(bytes, attribute_visible_bytes(attribute)?)?;
            }
        }
        for infos in [&graph.input, &graph.output, &graph.value_info] {
            for (i, info) in infos.iter().enumerate() {
                poll(budget, at, i)?;
                bytes = add(bytes, info_visible_bytes(info, budget, at)?)?;
            }
        }
    }
    checkpoint(budget, at)?;
    Ok(bytes)
}
fn tensor_visible_bytes(tensor: &TensorProto) -> Result<usize> {
    ensure!(
        tensor.segment.is_none()
            && tensor.external_data.is_empty()
            && tensor.string_data.is_empty()
            && tensor.int32_data.is_empty()
            && tensor.double_data.is_empty()
            && tensor.uint64_data.is_empty(),
        "actual original tensor outside bounded decoder surface"
    );
    // The tensor header is already included in its actual initializer Vec
    // capacity or the enclosing AttributeProto's inline optional field.
    let mut bytes = tensor.name.capacity();
    for amount in [
        mul(tensor.dims.capacity(), 8)?,
        tensor.raw_data.capacity(),
        mul(tensor.float_data.capacity(), 4)?,
        mul(tensor.int64_data.capacity(), 8)?,
    ] {
        bytes = add(bytes, amount)?;
    }
    Ok(bytes)
}
fn attribute_visible_bytes(attribute: &AttributeProto) -> Result<usize> {
    ensure!(
        attribute.g.is_none()
            && attribute.sparse_tensor.is_none()
            && attribute.tp.is_none()
            && attribute.floats.is_empty()
            && attribute.strings.is_empty()
            && attribute.tensors.is_empty()
            && attribute.graphs.is_empty()
            && attribute.sparse_tensors.is_empty()
            && attribute.type_protos.is_empty(),
        "actual original attribute outside bounded decoder surface"
    );
    let mut bytes = add(
        attribute.name.capacity(),
        attribute.ref_attr_name.capacity(),
    )?;
    bytes = add(bytes, attribute.doc_string.capacity())?;
    bytes = add(bytes, mul(attribute.ints.capacity(), 8)?)?;
    if let Some(s) = attribute.s.as_ref() {
        bytes = add(bytes, s.capacity())?;
    }
    if let Some(tensor) = attribute.t.as_ref() {
        bytes = add(bytes, tensor_visible_bytes(tensor)?)?;
    }
    Ok(bytes)
}
fn info_visible_bytes(info: &ValueInfoProto, budget: &WorkBudget, at: Instant) -> Result<usize> {
    let mut bytes = info.name.capacity();
    if let Some(typ) = info.r#type.as_ref() {
        bytes = add(bytes, size_of::<TypeProto>())?;
        if let Some(tensor) = typ.tensor_type.as_ref() {
            bytes = add(bytes, size_of::<TensorTypeProto>())?;
            if let Some(shape) = tensor.shape.as_ref() {
                bytes = add(
                    bytes,
                    add(
                        size_of::<TensorShapeProto>(),
                        mul(shape.dim.capacity(), size_of::<Dimension>())?,
                    )?,
                )?;
                for (i, dimension) in shape.dim.iter().enumerate() {
                    poll(budget, at, i)?;
                    if let Some(Value::DimParam(name)) = dimension.value.as_ref() {
                        bytes = add(bytes, name.capacity())?;
                    }
                }
            }
        }
    }
    Ok(bytes)
}
fn extracted_visible_bytes(
    spec: &TwinSpec,
    names: &super::margin_row_bab::TwinSiteNames,
    budget: &WorkBudget,
    at: Instant,
) -> Result<usize> {
    let mut bytes = add(
        size_of::<TwinSpec>(),
        mul(spec.ops.capacity(), size_of::<TwinOpSpec>())?,
    )?;
    for (i, op) in spec.ops.iter().enumerate() {
        poll(budget, at, i)?;
        match op {
            TwinOpSpec::Conv {
                weight,
                bias,
                bias_err,
                ..
            }
            | TwinOpSpec::ConvTranspose {
                weight,
                bias,
                bias_err,
                ..
            }
            | TwinOpSpec::Gemm {
                weight,
                bias,
                bias_err,
                ..
            } => {
                for vector in [weight, bias, bias_err] {
                    bytes = add(bytes, mul(vector.capacity(), 8)?)?;
                }
            }
            TwinOpSpec::ChannelAffine {
                scale,
                shift,
                shift_err,
                ..
            } => {
                for vector in [scale, shift, shift_err] {
                    bytes = add(bytes, mul(vector.capacity(), 8)?)?;
                }
            }
            _ => (),
        }
    }
    bytes = add(
        bytes,
        mul(names.capacity(), size_of::<(usize, String, usize)>())?,
    )?;
    for (i, (_, name, _)) in names.iter().enumerate() {
        poll(budget, at, i)?;
        bytes = add(bytes, name.capacity())?;
    }
    checkpoint(budget, at)?;
    Ok(bytes)
}
fn checkpoint(budget: &WorkBudget, at: Instant) -> Result<()> {
    budget.checkpoint_for(LeaseClass::Fallback)?;
    ensure!(
        Instant::now() < at,
        "staged original source deadline expired"
    );
    ny_core::require_f64_interval_proof_environment()?;
    Ok(())
}
fn poll(budget: &WorkBudget, at: Instant, i: usize) -> Result<()> {
    if i.is_multiple_of(POLL) {
        checkpoint(budget, at)?;
    }
    Ok(())
}
fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| anyhow!("original source capacity overflow"))
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| anyhow!("original source capacity overflow"))
}
fn charged(n: usize, factor: usize) -> Result<u64> {
    u64::try_from(add(mul(n, factor)?, META)?).map_err(Into::into)
}

#[cfg(test)]
#[path = "original_source_census_tests.rs"]
mod tests;
