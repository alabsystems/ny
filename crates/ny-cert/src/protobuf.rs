// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `protobuf` — a total protobuf wire-format reader for the ONNX loaders.
//!
//! `certify_onnx` and `certify_onnx_diff` read untrusted `.onnx` files, which
//! are protobuf-encoded. Each binary used to carry its own copy of an indexing
//! reader (`self.buf[self.i]`, `&self.buf[self.i..self.i + len]`, an unbounded
//! `<< shift` varint loop, `panic!` on an unknown wire type), so a truncated or
//! corrupt model crashed the tool with an index, shift or arithmetic panic
//! instead of a diagnostic. This module is the single replacement:
//!
//! * every read is a slice split (`split_first`, `split_at_checked`,
//!   `split_first_chunk`) with the short case returned as an error — no
//!   indexing, no position arithmetic;
//! * a varint is at most [`MAX_VARINT_BYTES`] bytes (the protobuf limit for a
//!   64-bit value); a longer run of continuation bytes is an error rather than
//!   a shift past 63;
//! * a length prefix that does not fit `usize`, or that runs past the end of
//!   the buffer, is an error;
//! * the group wire types (3/4, deprecated and never emitted for ONNX) and the
//!   undefined ones (6/7) are errors.
//!
//! Decoding of well-formed input is unchanged: the same field values in the
//! same order.

use std::fmt;

/// The longest valid varint encoding: ten 7-bit groups cover 64 bits.
pub const MAX_VARINT_BYTES: usize = 10;

/// Why a protobuf buffer could not be decoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WireError {
    /// The buffer ended inside a varint, fixed-width value or tag.
    Truncated,
    /// A varint ran past [`MAX_VARINT_BYTES`] continuation bytes.
    VarintTooLong,
    /// A length-delimited field claimed `len` bytes but fewer remain (or `len`
    /// does not fit in `usize`).
    LengthOutOfRange {
        /// The declared length, as encoded.
        len: u64,
    },
    /// A tag carried a wire type this reader does not decode.
    UnsupportedWireType {
        /// The wire type (low three bits of the tag).
        wire_type: u8,
    },
}

impl fmt::Display for WireError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated => f.write_str("protobuf buffer truncated"),
            Self::VarintTooLong => f.write_str("protobuf varint longer than 10 bytes"),
            Self::LengthOutOfRange { len } => write!(
                f,
                "protobuf length-delimited field of {len} bytes runs past the end of the buffer"
            ),
            Self::UnsupportedWireType { wire_type } => {
                write!(f, "unsupported protobuf wire type {wire_type}")
            }
        }
    }
}

impl std::error::Error for WireError {}

/// A forward-only reader over one protobuf message.
#[derive(Debug, Clone)]
pub struct Reader<'a> {
    rest: &'a [u8],
}

impl<'a> Reader<'a> {
    /// A reader over the whole of `buf`.
    pub const fn new(buf: &'a [u8]) -> Self {
        Reader { rest: buf }
    }

    /// True once every byte has been consumed.
    pub const fn eof(&self) -> bool {
        self.rest.is_empty()
    }

    fn byte(&mut self) -> Result<u8, WireError> {
        let (&b, rest) = self.rest.split_first().ok_or(WireError::Truncated)?;
        self.rest = rest;
        Ok(b)
    }

    /// A base-128 varint. Bits beyond the 64th (only possible in the tenth
    /// byte) are discarded, as every protobuf decoder does.
    pub fn varint(&mut self) -> Result<u64, WireError> {
        let mut result = 0u64;
        // 0, 7, ..., 63: exactly MAX_VARINT_BYTES shifts, each < 64.
        for shift in (0u32..64).step_by(7) {
            let b = self.byte()?;
            result |= u64::from(b & 0x7f).checked_shl(shift).unwrap_or(0);
            if b & 0x80 == 0 {
                return Ok(result);
            }
        }
        Err(WireError::VarintTooLong)
    }

    /// A field tag, split into `(field_number, wire_type)`.
    pub fn tag(&mut self) -> Result<(u64, u8), WireError> {
        let t = self.varint()?;
        // `t & 7` is at most 7, so the narrowing is lossless.
        Ok((t >> 3, (t & 7) as u8))
    }

    /// A length-delimited payload (wire type 2).
    pub fn bytes(&mut self) -> Result<&'a [u8], WireError> {
        let len = self.varint()?;
        let n = usize::try_from(len).map_err(|_| WireError::LengthOutOfRange { len })?;
        let (s, rest) = self
            .rest
            .split_at_checked(n)
            .ok_or(WireError::LengthOutOfRange { len })?;
        self.rest = rest;
        Ok(s)
    }

    /// A little-endian fixed 32-bit payload (wire type 5).
    pub fn fixed32(&mut self) -> Result<[u8; 4], WireError> {
        let (s, rest) = self
            .rest
            .split_first_chunk::<4>()
            .ok_or(WireError::Truncated)?;
        self.rest = rest;
        Ok(*s)
    }

    /// A little-endian fixed 64-bit payload (wire type 1).
    pub fn fixed64(&mut self) -> Result<[u8; 8], WireError> {
        let (s, rest) = self
            .rest
            .split_first_chunk::<8>()
            .ok_or(WireError::Truncated)?;
        self.rest = rest;
        Ok(*s)
    }

    /// Skip one field payload of wire type `wt`.
    pub fn skip(&mut self, wt: u8) -> Result<(), WireError> {
        match wt {
            0 => self.varint().map(drop),
            1 => self.fixed64().map(drop),
            2 => self.bytes().map(drop),
            5 => self.fixed32().map(drop),
            wire_type => Err(WireError::UnsupportedWireType { wire_type }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varint_decodes_single_and_multi_byte_values() {
        assert_eq!(Reader::new(&[0x00]).varint(), Ok(0));
        assert_eq!(Reader::new(&[0x7f]).varint(), Ok(127));
        assert_eq!(Reader::new(&[0xac, 0x02]).varint(), Ok(300));
        let max = [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01];
        assert_eq!(Reader::new(&max).varint(), Ok(u64::MAX));
    }

    #[test]
    fn varint_longer_than_ten_bytes_is_an_error_not_a_shift_panic() {
        // Eleven continuation bytes: the old reader shifted by 70 and panicked.
        let bad = [0x80u8; 11];
        assert_eq!(Reader::new(&bad).varint(), Err(WireError::VarintTooLong));
    }

    #[test]
    fn truncated_varint_is_an_error_not_an_index_panic() {
        assert_eq!(Reader::new(&[]).varint(), Err(WireError::Truncated));
        assert_eq!(
            Reader::new(&[0x80, 0x80]).varint(),
            Err(WireError::Truncated)
        );
    }

    #[test]
    fn tag_splits_field_number_and_wire_type() {
        // field 7, wire type 2 -> (7 << 3) | 2 = 58.
        assert_eq!(Reader::new(&[58]).tag(), Ok((7, 2)));
    }

    #[test]
    fn bytes_returns_payload_and_advances() {
        let mut r = Reader::new(&[3, b'a', b'b', b'c', 0x05]);
        assert_eq!(r.bytes(), Ok(&b"abc"[..]));
        assert_eq!(r.varint(), Ok(5));
        assert!(r.eof());
    }

    #[test]
    fn bytes_past_end_is_an_error_not_a_slice_panic() {
        let mut r = Reader::new(&[10, 1, 2]);
        assert_eq!(r.bytes(), Err(WireError::LengthOutOfRange { len: 10 }));
        // A length near u64::MAX: the old `self.i + len` overflowed.
        let huge = [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01];
        assert_eq!(
            Reader::new(&huge).bytes(),
            Err(WireError::LengthOutOfRange { len: u64::MAX })
        );
    }

    #[test]
    fn fixed_width_reads_and_truncation() {
        let mut r = Reader::new(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        assert_eq!(r.fixed32(), Ok([1, 2, 3, 4]));
        assert_eq!(r.fixed64(), Ok([5, 6, 7, 8, 9, 10, 11, 12]));
        assert!(r.eof());
        assert_eq!(Reader::new(&[1, 2, 3]).fixed32(), Err(WireError::Truncated));
        assert_eq!(Reader::new(&[1; 7]).fixed64(), Err(WireError::Truncated));
    }

    #[test]
    fn skip_handles_every_supported_wire_type() {
        // varint(300), fixed64, bytes("ab"), fixed32, then a trailing 0x09.
        let buf = [
            0xac, 0x02, 0, 0, 0, 0, 0, 0, 0, 0, 2, b'a', b'b', 0, 0, 0, 0, 0x09,
        ];
        let mut r = Reader::new(&buf);
        for wt in [0, 1, 2, 5] {
            assert_eq!(r.skip(wt), Ok(()));
        }
        assert_eq!(r.varint(), Ok(9));
        assert!(r.eof());
    }

    #[test]
    fn skip_rejects_group_and_undefined_wire_types() {
        for wt in [3u8, 4, 6, 7] {
            assert_eq!(
                Reader::new(&[0; 16]).skip(wt),
                Err(WireError::UnsupportedWireType { wire_type: wt })
            );
        }
    }

    #[test]
    fn errors_display_a_diagnostic() {
        assert_eq!(
            WireError::Truncated.to_string(),
            "protobuf buffer truncated"
        );
        assert!(WireError::LengthOutOfRange { len: 9 }
            .to_string()
            .contains("9 bytes"));
    }
}
