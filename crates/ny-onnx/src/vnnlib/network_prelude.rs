// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! How many networks does a VNN-LIB property declare? (#name-blind-routing)
//!
//! A VNN-LIB 2.0 property states its own arity: every network it constrains is
//! introduced by a top-level `(declare-network NAME ...)` form, and a
//! relational property (isomorphism, monotonicity, ground-truth dominance)
//! declares two of them. That count is a fact about the INSTANCE, so a router
//! can choose the dual-network route from it instead of from the benchmark's
//! category name — which is what the VNN-COMP dispatch used to do.
//!
//! # Why a prelude probe and not [`super::load_vnnlib`]
//!
//! The router runs on every scored row before any lane, and some properties
//! are hundreds of megabytes (the 2026 nn4sys and collins files reach ~130 MB
//! each). A full parse there would be paid twice on exactly the rows whose
//! budgets are tightest. Declarations precede the assertions that use them,
//! so the probe reads only the DECLARATION PRELUDE — every top-level form up to
//! the first top-level `(assert ...)` — and never enters an assertion body.
//! Its cost is bounded by the prelude, which is a strict prefix of what the
//! ordinary parse reads anyway.
//!
//! # Lexical contract
//!
//! The probe uses the same lexical conventions as the ordinary tokenizer
//! (`vnnlib::syntax`): whitespace is space, tab, `\n` and `\r`; a `;` outside
//! a string starts a comment that runs to the end of the line; a `"` string
//! has no escapes and runs to the next `"`; an atom ends at whitespace, a
//! parenthesis, a quote or a `;`. A form's head is its first atom.
//!
//! # What the count does and does not mean
//!
//! It counts `declare-network` HEADS. It does not validate the declarations —
//! that is [`super::parse_vnnlib`]'s `dual_network` extraction, which the
//! relational route performs itself and which fails closed to `unknown` on any
//! shape it cannot validate. So `declared_networks >= 2` is the conservative
//! routing signal: every property that names two networks leaves the
//! single-network lanes (which ignore network metadata), and nothing else does.

use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use flate2::read::GzDecoder;
use ny_core::{NyError, Result};

/// Where the prelude probe stopped reading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreludeEnd {
    /// At the head of the first top-level `(assert ...)` form. Nothing after
    /// this point was read.
    FirstAssert,
    /// End of input before any top-level assertion.
    EndOfInput,
}

/// The network declarations found in a VNN-LIB property's prelude.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NetworkPrelude {
    /// Top-level `(declare-network ...)` forms before the first top-level
    /// `(assert ...)` form. `0` for a VNN-LIB 1.0 property, `1` for an ordinary
    /// VNN-LIB 2.0 property, `2` for a relational (dual-network) one.
    pub declared_networks: usize,
    /// Where the probe stopped.
    pub end: PreludeEnd,
}

impl NetworkPrelude {
    /// True when the property constrains two or more declared networks — the
    /// structural signal for the relational (dual-network) verification route.
    #[must_use]
    pub fn is_multi_network(&self) -> bool {
        self.declared_networks >= 2
    }
}

const fn is_space(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | b'\r')
}

const fn ends_atom(byte: u8) -> bool {
    is_space(byte) || matches!(byte, b'(' | b')' | b'"' | b';')
}

/// Byte cursor over a buffered reader. `peek` never consumes.
struct Cursor<R: BufRead> {
    inner: R,
}

impl<R: BufRead> Cursor<R> {
    fn peek(&mut self) -> std::io::Result<Option<u8>> {
        Ok(self.inner.fill_buf()?.first().copied())
    }

    fn bump(&mut self) {
        self.inner.consume(1);
    }

    fn next_byte(&mut self) -> std::io::Result<Option<u8>> {
        let byte = self.peek()?;
        if byte.is_some() {
            self.bump();
        }
        Ok(byte)
    }

    /// After a `;`: consume through the end of the line.
    fn skip_comment(&mut self) -> std::io::Result<()> {
        while let Some(byte) = self.next_byte()? {
            if byte == b'\n' {
                break;
            }
        }
        Ok(())
    }

    /// After an opening `"`: consume through the closing `"` (no escapes, as
    /// in the ordinary tokenizer).
    fn skip_string(&mut self) -> std::io::Result<()> {
        while let Some(byte) = self.next_byte()? {
            if byte == b'"' {
                break;
            }
        }
        Ok(())
    }

    /// Consume the rest of an atom whose first byte was already consumed.
    /// `head` collects at most `HEAD_CAPACITY` bytes; longer atoms are not
    /// either of the two heads the probe compares against, so truncation is
    /// harmless.
    fn read_atom(&mut self, first: u8, head: &mut Vec<u8>) -> std::io::Result<()> {
        head.clear();
        head.push(first);
        while let Some(byte) = self.peek()? {
            if ends_atom(byte) {
                break;
            }
            if head.len() < HEAD_CAPACITY {
                head.push(byte);
            }
            self.bump();
        }
        Ok(())
    }

    /// Consume the remainder of a form whose opening parenthesis (and possibly
    /// head) was already consumed, through its matching `)`.
    fn skip_form_body(&mut self) -> std::io::Result<()> {
        let mut depth = 1_usize;
        while depth > 0 {
            let Some(byte) = self.next_byte()? else {
                break;
            };
            match byte {
                b';' => self.skip_comment()?,
                b'"' => self.skip_string()?,
                b'(' => depth += 1,
                b')' => depth -= 1,
                _ => {}
            }
        }
        Ok(())
    }

    /// Skip whitespace and comments without consuming the next significant byte.
    fn skip_insignificant(&mut self) -> std::io::Result<()> {
        while let Some(byte) = self.peek()? {
            if is_space(byte) {
                self.bump();
            } else if byte == b';' {
                self.bump();
                self.skip_comment()?;
            } else {
                break;
            }
        }
        Ok(())
    }
}

/// Longer than either compared head (`declare-network` is 15 bytes), so a
/// truncated atom can never compare equal to one.
const HEAD_CAPACITY: usize = 32;
const ASSERT_HEAD: &[u8] = b"assert";
const DECLARE_NETWORK_HEAD: &[u8] = b"declare-network";

/// Count the top-level `(declare-network ...)` forms that precede the first
/// top-level `(assert ...)` form of a VNN-LIB property read from `reader`.
///
/// Reading stops at the head of the first assertion, so the cost is bounded
/// by the declaration prelude however large the property is.
///
/// # Errors
///
/// Propagates the reader's I/O errors.
pub fn scan_network_prelude<R: Read>(reader: R) -> std::io::Result<NetworkPrelude> {
    let mut cursor = Cursor {
        inner: BufReader::new(reader),
    };
    let mut declared_networks = 0_usize;
    let mut head = Vec::with_capacity(HEAD_CAPACITY);
    while let Some(byte) = cursor.next_byte()? {
        match byte {
            b if is_space(b) => {}
            b';' => cursor.skip_comment()?,
            b'"' => cursor.skip_string()?,
            // A stray closing parenthesis at top level carries no declaration.
            b')' => {}
            b'(' => {
                cursor.skip_insignificant()?;
                head.clear();
                if let Some(first) = cursor.peek()? {
                    if !ends_atom(first) {
                        cursor.bump();
                        cursor.read_atom(first, &mut head)?;
                    }
                }
                if head == ASSERT_HEAD {
                    return Ok(NetworkPrelude {
                        declared_networks,
                        end: PreludeEnd::FirstAssert,
                    });
                }
                if head == DECLARE_NETWORK_HEAD {
                    declared_networks += 1;
                }
                cursor.skip_form_body()?;
            }
            // A stray top-level atom carries no declaration.
            other => cursor.read_atom(other, &mut head)?,
        }
    }
    Ok(NetworkPrelude {
        declared_networks,
        end: PreludeEnd::EndOfInput,
    })
}

/// [`scan_network_prelude`] over a VNN-LIB file (`.gz` is decompressed, as in
/// [`super::load_vnnlib`]).
///
/// # Errors
///
/// Returns [`NyError::ModelLoad`] when the file cannot be opened or read.
pub fn load_vnnlib_network_prelude<P: AsRef<Path>>(path: P) -> Result<NetworkPrelude> {
    let path = path.as_ref();
    let file = File::open(path).map_err(|error| {
        NyError::ModelLoad(format!(
            "Failed to open VNN-LIB {}: {error}",
            path.display()
        ))
    })?;
    let scanned = if path.extension().and_then(|extension| extension.to_str()) == Some("gz") {
        scan_network_prelude(GzDecoder::new(file))
    } else {
        scan_network_prelude(file)
    };
    scanned.map_err(|error| {
        NyError::ModelLoad(format!(
            "Failed to read VNN-LIB {}: {error}",
            path.display()
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::{load_vnnlib_network_prelude, scan_network_prelude, NetworkPrelude, PreludeEnd};
    use crate::vnnlib::parse_vnnlib;
    use std::io::{Read, Write};

    fn scan(text: &str) -> NetworkPrelude {
        scan_network_prelude(text.as_bytes()).expect("in-memory scan cannot fail")
    }

    /// The canonical 2026 `isomorphic_acasxu_2026` shape (instance_0).
    const ISOMORPHIC: &str = r"; Epsilon equivalence (comment mentions (declare-network x) and (assert y))
(vnnlib-version <2.0>)

(declare-network f
    (declare-input X_f real [1, 1, 1, 5])
    (declare-output Y_f real [1, 5])
)
(declare-network g
    (isomorphic-to f)
    (declare-input X_g real [1, 1, 1, 5])
    (declare-output Y_g real [1, 5])
)
(assert (and (<= X_f[0,0,0,0] 0.679857769) (>= X_f[0,0,0,0] 0.6)))
(assert (and (<= X_f[0,0,0,1] 0.5) (>= X_f[0,0,0,1] -0.5)))
(assert (and (<= X_f[0,0,0,2] 0.5) (>= X_f[0,0,0,2] -0.5)))
(assert (and (<= X_f[0,0,0,3] 0.5) (>= X_f[0,0,0,3] 0.45)))
(assert (and (<= X_f[0,0,0,4] -0.45) (>= X_f[0,0,0,4] -0.5)))
(assert (== X_f[0,0,0,0] X_g[0,0,0,0]))
(assert (== X_f[0,0,0,1] X_g[0,0,0,1]))
(assert (== X_f[0,0,0,2] X_g[0,0,0,2]))
(assert (== X_f[0,0,0,3] X_g[0,0,0,3]))
(assert (== X_f[0,0,0,4] X_g[0,0,0,4]))
(assert (or
 (or (> Y_g[0,0] (+ Y_f[0,0] 0.05)) (< Y_g[0,0] (- Y_f[0,0] 0.05)))
 (or (> Y_g[0,1] (+ Y_f[0,1] 0.05)) (< Y_g[0,1] (- Y_f[0,1] 0.05)))
 (or (> Y_g[0,2] (+ Y_f[0,2] 0.05)) (< Y_g[0,2] (- Y_f[0,2] 0.05)))
 (or (> Y_g[0,3] (+ Y_f[0,3] 0.05)) (< Y_g[0,3] (- Y_f[0,3] 0.05)))
 (or (> Y_g[0,4] (+ Y_f[0,4] 0.05)) (< Y_g[0,4] (- Y_f[0,4] 0.05)))
))
";

    /// An ordinary single-network VNN-LIB 2.0 property (2026 acasxu shape).
    const SINGLE_V2: &str = r"(vnnlib-version <2.0>)

(declare-network N
    (declare-input  X float32 [1, 1, 1, 5])
    (declare-output Y float32 [1, 5])
)
; ACAS Xu property 1.0
(assert (<= X[0,0,0,0] 0.679857769))
(assert (>= X[0,0,0,0] 0.6))
(assert (<= Y[0,0] 3.991125645861615))
";

    /// A VNN-LIB 1.0 property.
    const V1: &str = r"; Property with label: 0.
(declare-const X_0 Real)
(declare-const Y_0 Real)
(assert (<= X_0 0.5))
(assert (>= X_0 -0.5))
(assert (<= Y_0 0.0))
";

    #[test]
    fn relational_spec_declares_two_networks_before_its_first_assertion() {
        let prelude = scan(ISOMORPHIC);
        assert_eq!(
            prelude,
            NetworkPrelude {
                declared_networks: 2,
                end: PreludeEnd::FirstAssert,
            }
        );
        assert!(prelude.is_multi_network());
        // The parser agrees that this is a validated dual-network property.
        let spec = parse_vnnlib(ISOMORPHIC).expect("canonical isomorphic spec parses");
        assert!(spec.dual_network.is_some());
    }

    #[test]
    fn single_network_and_v1_specs_are_not_multi_network() {
        let v2 = scan(SINGLE_V2);
        assert_eq!(v2.declared_networks, 1);
        assert_eq!(v2.end, PreludeEnd::FirstAssert);
        assert!(!v2.is_multi_network());
        assert!(parse_vnnlib(SINGLE_V2)
            .expect("single-network v2 parses")
            .dual_network
            .is_none());

        let v1 = scan(V1);
        assert_eq!(v1.declared_networks, 0);
        assert_eq!(v1.end, PreludeEnd::FirstAssert);
        assert!(!v1.is_multi_network());
    }

    #[test]
    fn comments_strings_and_nested_forms_never_count_as_declarations() {
        let text = r#"; (declare-network commented_out)
(vnnlib-version <2.0>)
(set-info :source "(declare-network in_a_string) ; not a comment")
(declare-network N
    (declare-input X real [2]) ; (declare-network trailing_comment)
    (declare-output Y real [1])
    (declare-network nested_is_not_top_level)
)
(assert (<= X[0] 1.0))
(declare-network after_the_first_assert_is_never_read)
"#;
        let prelude = scan(text);
        assert_eq!(prelude.declared_networks, 1);
        assert_eq!(prelude.end, PreludeEnd::FirstAssert);
    }

    #[test]
    fn probe_stops_at_the_first_assertion_without_reading_its_body() {
        // An unterminated assertion body would make a full reader fail or run
        // to the end; the probe must stop at the head.
        struct Guarded {
            data: &'static [u8],
            offset: usize,
            limit: usize,
        }
        impl Read for Guarded {
            fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
                if self.offset >= self.limit {
                    return Err(std::io::Error::other("probe read past its prelude"));
                }
                let end = self.limit.min(self.offset + out.len().min(1));
                let n = end - self.offset;
                out[..n].copy_from_slice(&self.data[self.offset..end]);
                self.offset = end;
                Ok(n)
            }
        }
        const TEXT: &str = "(declare-network f (declare-input X real [1]) (declare-output Y real [1]))\n\
                            (declare-network g (equal-to f) (declare-input X2 real [1]) (declare-output Y2 real [1]))\n\
                            (assert (or (and ";
        let head_end = TEXT.find("(assert").expect("fixture has an assertion") + "(assert".len();
        // One byte of lookahead after the head is allowed (the atom terminator).
        let reader = Guarded {
            data: TEXT.as_bytes(),
            offset: 0,
            limit: head_end + 1,
        };
        let prelude = scan_network_prelude(reader).expect("probe stops before the guard");
        assert_eq!(prelude.declared_networks, 2);
        assert_eq!(prelude.end, PreludeEnd::FirstAssert);
    }

    #[test]
    fn a_property_without_assertions_is_scanned_to_the_end() {
        let prelude =
            scan("(declare-network f (declare-input X real [1]) (declare-output Y real [1]))");
        assert_eq!(
            prelude,
            NetworkPrelude {
                declared_networks: 1,
                end: PreludeEnd::EndOfInput,
            }
        );
        assert_eq!(scan("").declared_networks, 0);
    }

    #[test]
    fn gzip_and_plain_files_probe_identically() {
        let dir = tempfile::tempdir().expect("tempdir");
        let plain = dir.path().join("instance_0.vnnlib");
        std::fs::write(&plain, ISOMORPHIC).expect("write plain");
        let gz = dir.path().join("instance_0.vnnlib.gz");
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        encoder
            .write_all(ISOMORPHIC.as_bytes())
            .expect("compress fixture");
        std::fs::write(&gz, encoder.finish().expect("finish gzip")).expect("write gz");
        let from_plain = load_vnnlib_network_prelude(&plain).expect("probe plain");
        let from_gz = load_vnnlib_network_prelude(&gz).expect("probe gz");
        assert_eq!(from_plain, from_gz);
        assert!(from_plain.is_multi_network());
        assert!(load_vnnlib_network_prelude(dir.path().join("missing.vnnlib")).is_err());
    }
}
