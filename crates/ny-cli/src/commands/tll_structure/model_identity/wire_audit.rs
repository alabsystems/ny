// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Bounded raw-wire allowlist for the ONNX messages that carry TLL semantics.
//!
//! Prost intentionally discards unknown protobuf fields. That is useful for
//! ordinary compatibility, but source authentication must not silently accept
//! a future semantic extension it did not inspect. This scanner runs before
//! the typed decode, rejects unknown tags and wrong wire types, and recursively
//! audits every message that can affect this lane's graph, tensor, or endpoint
//! interpretation. Known documentation/metadata fields remain allowed because
//! they cannot affect execution.

const VARINT: u8 = 0;
const FIXED64: u8 = 1;
const LENGTH_DELIMITED: u8 = 2;
const FIXED32: u8 = 5;
const MAX_PROTOBUF_FIELD_NUMBER: u64 = (1_u64 << 29) - 1;

pub(super) fn audit_model_wire(bytes: &[u8]) -> Option<()> {
    audit_fields(bytes, |number, wire, payload| match number {
        1 | 5 => require_wire(wire, &[VARINT]),
        2 | 3 | 4 | 6 | 14 | 20 | 25 | 26 => require_wire(wire, &[LENGTH_DELIMITED]),
        7 => {
            require_wire(wire, &[LENGTH_DELIMITED])?;
            audit_graph(payload)
        }
        8 => {
            require_wire(wire, &[LENGTH_DELIMITED])?;
            audit_opset(payload)
        }
        _ => None,
    })
}

fn audit_graph(bytes: &[u8]) -> Option<()> {
    audit_fields(bytes, |number, wire, payload| match number {
        1 => {
            require_wire(wire, &[LENGTH_DELIMITED])?;
            audit_node(payload)
        }
        2 | 10 | 14 | 16 => require_wire(wire, &[LENGTH_DELIMITED]),
        5 => {
            require_wire(wire, &[LENGTH_DELIMITED])?;
            audit_tensor(payload)
        }
        11..=13 => {
            require_wire(wire, &[LENGTH_DELIMITED])?;
            audit_value_info(payload)
        }
        // Sparse initializers are rejected by the typed audit, so their
        // contents cannot become authority.
        15 => require_wire(wire, &[LENGTH_DELIMITED]),
        _ => None,
    })
}

fn audit_node(bytes: &[u8]) -> Option<()> {
    audit_fields(bytes, |number, wire, _| match number {
        1..=10 => require_wire(wire, &[LENGTH_DELIMITED]),
        _ => None,
    })
}

fn audit_tensor(bytes: &[u8]) -> Option<()> {
    audit_fields(bytes, |number, wire, _| match number {
        1 => require_wire(wire, &[VARINT, LENGTH_DELIMITED]),
        2 | 14 => require_wire(wire, &[VARINT]),
        3 | 6 | 8 | 9 | 12 | 13 | 16 => require_wire(wire, &[LENGTH_DELIMITED]),
        4 => require_wire(wire, &[FIXED32, LENGTH_DELIMITED]),
        5 | 7 | 11 => require_wire(wire, &[VARINT, LENGTH_DELIMITED]),
        10 => require_wire(wire, &[FIXED64, LENGTH_DELIMITED]),
        _ => None,
    })
}

fn audit_value_info(bytes: &[u8]) -> Option<()> {
    audit_fields(bytes, |number, wire, payload| match number {
        1 | 3 | 4 => require_wire(wire, &[LENGTH_DELIMITED]),
        2 => {
            require_wire(wire, &[LENGTH_DELIMITED])?;
            audit_type(payload)
        }
        _ => None,
    })
}

fn audit_type(bytes: &[u8]) -> Option<()> {
    audit_fields(bytes, |number, wire, payload| match number {
        1 => {
            require_wire(wire, &[LENGTH_DELIMITED])?;
            audit_tensor_type(payload)
        }
        // Whole-type denotation is documentation, not execution semantics.
        6 => require_wire(wire, &[LENGTH_DELIMITED]),
        // Sequence, map, sparse-tensor, and optional alternatives must never
        // coexist with or be silently discarded beside the required tensor.
        4 | 5 | 8 | 9 => None,
        _ => None,
    })
}

fn audit_tensor_type(bytes: &[u8]) -> Option<()> {
    audit_fields(bytes, |number, wire, payload| match number {
        1 => require_wire(wire, &[VARINT]),
        2 => {
            require_wire(wire, &[LENGTH_DELIMITED])?;
            audit_tensor_shape(payload)
        }
        _ => None,
    })
}

fn audit_tensor_shape(bytes: &[u8]) -> Option<()> {
    audit_fields(bytes, |number, wire, payload| match number {
        1 => {
            require_wire(wire, &[LENGTH_DELIMITED])?;
            audit_dimension(payload)
        }
        _ => None,
    })
}

fn audit_dimension(bytes: &[u8]) -> Option<()> {
    audit_fields(bytes, |number, wire, _| match number {
        1 => require_wire(wire, &[VARINT]),
        2 | 3 => require_wire(wire, &[LENGTH_DELIMITED]),
        _ => None,
    })
}

fn audit_opset(bytes: &[u8]) -> Option<()> {
    audit_fields(bytes, |number, wire, _| match number {
        1 => require_wire(wire, &[LENGTH_DELIMITED]),
        2 => require_wire(wire, &[VARINT]),
        _ => None,
    })
}

fn require_wire(wire: u8, allowed: &[u8]) -> Option<()> {
    allowed.contains(&wire).then_some(())
}

fn audit_fields(
    mut bytes: &[u8],
    mut audit: impl FnMut(u32, u8, &[u8]) -> Option<()>,
) -> Option<()> {
    while !bytes.is_empty() {
        let (key, key_len) = decode_varint(bytes)?;
        bytes = bytes.get(key_len..)?;
        let number_u64 = key >> 3;
        if number_u64 == 0 || number_u64 > MAX_PROTOBUF_FIELD_NUMBER {
            return None;
        }
        let number = u32::try_from(number_u64).ok()?;
        let wire = u8::try_from(key & 7).ok()?;
        let (payload, consumed) = match wire {
            VARINT => {
                let (_, len) = decode_varint(bytes)?;
                (bytes.get(..len)?, len)
            }
            FIXED64 => (bytes.get(..8)?, 8),
            LENGTH_DELIMITED => {
                let (len, prefix) = decode_varint(bytes)?;
                let len = usize::try_from(len).ok()?;
                let consumed = prefix.checked_add(len)?;
                (bytes.get(prefix..consumed)?, consumed)
            }
            FIXED32 => (bytes.get(..4)?, 4),
            _ => return None,
        };
        audit(number, wire, payload)?;
        bytes = bytes.get(consumed..)?;
    }
    Some(())
}

fn decode_varint(bytes: &[u8]) -> Option<(u64, usize)> {
    let mut value = 0_u64;
    for index in 0..10usize {
        let byte = *bytes.get(index)?;
        if index == 9 && byte > 1 {
            return None;
        }
        value |= u64::from(byte & 0x7f) << (index * 7);
        if byte & 0x80 == 0 {
            return Some((value, index + 1));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_wire_and_alternate_endpoint_types_fail_closed() {
        assert!(audit_model_wire(&[0]).is_none());
        assert!(audit_model_wire(&[0x0b]).is_none()); // deprecated group wire type
        assert!(audit_type(&[0x22, 0x00]).is_none()); // empty sequence_type
        assert!(audit_type(&[0x4a, 0x00]).is_none()); // empty optional_type
    }
}
