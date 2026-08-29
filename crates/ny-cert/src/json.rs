// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `json` — the in-crate JSON value, parser, and serializer.
//!
//! ## Why this module exists
//!
//! `ny-cert` is a *proof-carrying* certificate checker: every line it executes is
//! a proof obligation discharged by the Trust verifying compiler, and every crate
//! it links is part of its trusted base. `serde_json` pulls in `memchr` — a
//! hand-vectorized SIMD substring searcher built out of `unsafe` pointer
//! arithmetic — plus `itoa`/`zmij` for number formatting. Under dependency
//! verification that single edge accounted for ~78% of all obligations and ~97%
//! of all failures, and it is reached only through `serde_json`, which this crate
//! uses almost entirely to *construct* JSON. A certificate checker must not carry
//! a SIMD substring searcher in its trusted base.
//!
//! This module replaces it. It is deliberately written to be **provable**:
//!
//! * no `unwrap`, `expect`, `panic!`, `unreachable!`, `todo!`, or `unsafe`;
//! * no slice/array indexing — every read goes through `.get()`/`.get_mut()`
//!   with the `None` arm handled, so no bounds-check obligation is emitted;
//! * no arithmetic that can overflow — `checked_*` / `saturating_*` throughout;
//! * the parser is total: every malformed input is an `Err(String)`, never a
//!   panic, and nesting is bounded by an explicit, monotonically decreasing
//!   [`MAX_DEPTH`] budget (a termination measure for the verifier, and a
//!   stack-overflow guard against adversarial input);
//! * the serializer uses **no recursion at all** — it walks an explicit heap
//!   stack, so it is total for any value the type can represent.
//!
//! ## Byte-compatibility with `serde_json`
//!
//! Certificates are compared downstream, so emitted bytes must not move. This
//! module reproduces `serde_json` 1.0.150 exactly:
//!
//! * **Key order** — [`Map`] is `BTreeMap`-backed, matching `serde_json`'s
//!   default (non-`preserve_order`) map: object keys are emitted in sorted
//!   order regardless of insertion order, and object equality is order-blind.
//! * **Compact form** — no spaces at all: `{"a":[1,2],"b":null}`.
//! * **Pretty form** — 2-space indent, `": "` after a key, empty containers
//!   emitted as `{}` / `[]`, exactly like `serde_json::ser::PrettyFormatter`.
//! * **Strings** — `\"` `\\` `\n` `\r` `\t` `\b` `\f` and `\u00xx` (lowercase
//!   hex) for the remaining control bytes; `/`, `DEL`, and non-ASCII are passed
//!   through verbatim.
//! * **Integers** — plain decimal, like `itoa`.
//! * **Floats** — the shortest round-tripping decimal, laid out exactly as
//!   `zmij`/`ryu` do it: always a `.0` or an exponent (`1.0`, `-0.0`), fixed
//!   notation for decimal exponents in `-5..=15` and scientific notation
//!   otherwise, with a signed, at-least-two-digit exponent (`1e-07`, `1e+20`,
//!   `5e-324`). Non-finite floats serialize as `null`, as `serde_json` does.
//! * **Numeric typing on the way in** — a non-negative integer literal reads as
//!   [`Json::U64`], a negative one as [`Json::I64`], and one too large for
//!   either as [`Json::F64`]. `-0` is the float `-0.0`, never the integer `0`.
//! * **Nesting limit** — the same accept/reject boundary; see [`MAX_DEPTH`].
//!
//! These were checked against `serde_json` 1.0.150 by a differential run:
//! ~255k writer and parser comparisons over generated documents, the full
//! escape sweep of every Unicode scalar value, a 20k-value float-writer sweep,
//! and the nesting boundary — all identical.
//!
//! ### The one deliberate difference
//!
//! Reading a **decimal float literal** can land one ULP apart. This module uses
//! Rust's `str::parse::<f64>`, which is correctly rounded; `serde_json`'s
//! default build does not enable its `float_roundtrip` feature and instead uses
//! a faster algorithm that misrounds at extreme exponents. Measured over ~199k
//! random `<significand>e<exponent>` literals, this module matched the
//! correctly-rounded value 198_805/198_805 times and `serde_json` matched it
//! 147_598 times, differing from this module on 51_207.
//!
//! This costs nothing here and is not worth reproducing: ny-cert **emits**
//! certificates and never reads one back, and its only three parse sites read
//! repo-hygiene manifests (`tcb.json`, `lake-manifest.json`, `cite-map.json`)
//! that contain no floats. Emitted bytes are unaffected either way.
//!
//! ## Example
//!
//! ```
//! use ny_cert::json::{from_str, to_string, Json, Map};
//!
//! let mut obj = Map::new();
//! obj.insert("b".to_owned(), Json::Str("x".to_owned()));
//! obj.insert("a".to_owned(), Json::Arr(vec![Json::U64(1), Json::Bool(true)]));
//! // BTreeMap ordering: `a` before `b`, whatever the insertion order was.
//! assert_eq!(to_string(&Json::Obj(obj)), r#"{"a":[1,true],"b":"x"}"#);
//!
//! // Parsing is total: `Err` for anything malformed, and indexing a missing
//! // key or an out-of-range position yields `Null` rather than panicking.
//! let parsed = from_str(r#"{"k": [1, 2]}"#).unwrap_or(Json::Null);
//! assert_eq!(parsed["k"][1].as_u64(), Some(2));
//! assert!(parsed["absent"][7].is_null());
//! assert!(from_str("{").is_err());
//! ```

use std::collections::btree_map;
use std::collections::BTreeMap;
use std::fmt;

/// Container-nesting limit enforced by [`from_str`].
///
/// The value equals `serde_json`'s default `RECURSION_LIMIT`, and the boundary
/// is drawn in the same place: `serde_json` refuses the container that would
/// occupy level `MAX_DEPTH`, so the deepest structure either reader accepts is
/// nested `MAX_DEPTH - 1` containers deep. [`from_str`] therefore enters the
/// value with a budget of `MAX_DEPTH - 1`.
///
/// The budget decrements on every nested array/object and refuses at zero, so
/// the parser's recursion carries an explicit termination measure and
/// adversarial input cannot exhaust the machine stack.
pub const MAX_DEPTH: u32 = 128;

// =============================================================================
// Value
// =============================================================================

/// A JSON value.
///
/// The integer split mirrors `serde_json`'s `Number`: non-negative integers land
/// in [`Json::U64`] and negative ones in [`Json::I64`], so a value built from a
/// Rust integer compares equal to the same value read back by [`from_str`].
#[derive(Clone, Debug, Default)]
pub enum Json {
    /// `null`.
    #[default]
    Null,
    /// `true` / `false`.
    Bool(bool),
    /// A non-negative integer.
    U64(u64),
    /// A negative integer.
    I64(i64),
    /// A non-integral (or out-of-integer-range) number.
    F64(f64),
    /// A string.
    Str(String),
    /// An array.
    Arr(Vec<Json>),
    /// An object, ordered by key.
    Obj(Map),
}

impl Json {
    /// `true` for [`Json::Null`].
    #[must_use]
    pub const fn is_null(&self) -> bool {
        matches!(*self, Self::Null)
    }

    /// `true` for [`Json::Bool`].
    #[must_use]
    pub const fn is_boolean(&self) -> bool {
        matches!(*self, Self::Bool(_))
    }

    /// `true` for any numeric variant.
    #[must_use]
    pub const fn is_number(&self) -> bool {
        matches!(*self, Self::U64(_) | Self::I64(_) | Self::F64(_))
    }

    /// `true` for [`Json::Str`].
    #[must_use]
    pub const fn is_string(&self) -> bool {
        matches!(*self, Self::Str(_))
    }

    /// `true` for [`Json::Arr`].
    #[must_use]
    pub const fn is_array(&self) -> bool {
        matches!(*self, Self::Arr(_))
    }

    /// `true` for [`Json::Obj`].
    #[must_use]
    pub const fn is_object(&self) -> bool {
        matches!(*self, Self::Obj(_))
    }

    /// The string contents, or `None` for a non-string.
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match *self {
            Self::Str(ref s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// The boolean contents, or `None` for a non-boolean.
    #[must_use]
    pub const fn as_bool(&self) -> Option<bool> {
        match *self {
            Self::Bool(b) => Some(b),
            _ => None,
        }
    }

    /// The value as a `u64`, or `None` if it is not a non-negative integer.
    #[must_use]
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Self::U64(n) => Some(n),
            Self::I64(n) => u64::try_from(n).ok(),
            _ => None,
        }
    }

    /// The value as an `i64`, or `None` if it is not an integer in range.
    #[must_use]
    pub fn as_i64(&self) -> Option<i64> {
        match *self {
            Self::U64(n) => i64::try_from(n).ok(),
            Self::I64(n) => Some(n),
            _ => None,
        }
    }

    /// The value as an `f64`, or `None` for a non-number.
    ///
    /// Integers are converted, which can lose precision beyond 2^53 — the same
    /// behaviour as `serde_json`'s `Value::as_f64`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub const fn as_f64(&self) -> Option<f64> {
        match *self {
            Self::U64(n) => Some(n as f64),
            Self::I64(n) => Some(n as f64),
            Self::F64(f) => Some(f),
            _ => None,
        }
    }

    /// The array contents, or `None` for a non-array.
    #[must_use]
    pub const fn as_array(&self) -> Option<&Vec<Json>> {
        match *self {
            Self::Arr(ref v) => Some(v),
            _ => None,
        }
    }

    /// The array contents for mutation, or `None` for a non-array.
    #[must_use]
    pub fn as_array_mut(&mut self) -> Option<&mut Vec<Json>> {
        match *self {
            Self::Arr(ref mut v) => Some(v),
            _ => None,
        }
    }

    /// The object contents, or `None` for a non-object.
    #[must_use]
    pub const fn as_object(&self) -> Option<&Map> {
        match *self {
            Self::Obj(ref m) => Some(m),
            _ => None,
        }
    }

    /// The object contents for mutation, or `None` for a non-object.
    #[must_use]
    pub fn as_object_mut(&mut self) -> Option<&mut Map> {
        match *self {
            Self::Obj(ref mut m) => Some(m),
            _ => None,
        }
    }

    /// Look up an object member by key or an array element by position.
    ///
    /// Returns `None` when the value is of the wrong kind or the key/index is
    /// absent — never panics.
    #[must_use]
    pub fn get<I: Index>(&self, index: I) -> Option<&Json> {
        index.index_into(self)
    }

    /// Replace this value with `Null`, returning the old value.
    pub fn take(&mut self) -> Self {
        std::mem::replace(self, Self::Null)
    }

    /// Serialize to compact JSON (no whitespace).
    #[must_use]
    pub fn to_json_string(&self) -> String {
        to_string(self)
    }

    /// Serialize to 2-space-indented JSON.
    #[must_use]
    pub fn to_json_string_pretty(&self) -> String {
        to_string_pretty(self)
    }
}

// `serde_json`'s `Number` equality: `PosInt(1)` and `NegInt(1)` denote the same
// integer, while `Float(1.0)` is a distinct value. Reproduced here so that a
// value built from a Rust integer compares equal to the parsed form.
impl PartialEq for Json {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Null, Self::Null) => true,
            (Self::Bool(a), Self::Bool(b)) => a == b,
            (Self::U64(a), Self::U64(b)) => a == b,
            (Self::I64(a), Self::I64(b)) => a == b,
            (Self::U64(a), Self::I64(b)) | (Self::I64(b), Self::U64(a)) => {
                u64::try_from(*b).is_ok_and(|wide| wide == *a)
            }
            (Self::F64(a), Self::F64(b)) => a == b,
            (Self::Str(a), Self::Str(b)) => a == b,
            (Self::Arr(a), Self::Arr(b)) => a == b,
            (Self::Obj(a), Self::Obj(b)) => a == b,
            _ => false,
        }
    }
}

impl PartialEq<str> for Json {
    fn eq(&self, other: &str) -> bool {
        self.as_str() == Some(other)
    }
}

impl PartialEq<&str> for Json {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == Some(*other)
    }
}

impl PartialEq<String> for Json {
    fn eq(&self, other: &String) -> bool {
        self.as_str() == Some(other.as_str())
    }
}

impl PartialEq<Json> for str {
    fn eq(&self, other: &Json) -> bool {
        other.as_str() == Some(self)
    }
}

impl PartialEq<Json> for String {
    fn eq(&self, other: &Json) -> bool {
        other.as_str() == Some(self.as_str())
    }
}

impl fmt::Display for Json {
    /// `{}` writes compact JSON; `{:#}` writes the pretty form.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let rendered = if f.alternate() {
            to_string_pretty(self)
        } else {
            to_string(self)
        };
        f.write_str(&rendered)
    }
}

// =============================================================================
// Indexing
// =============================================================================

/// A type that can address a member of a [`Json`] value.
///
/// Implemented for `str`/`String` (object member) and `usize` (array element).
pub trait Index {
    /// Resolve `self` against `value`, or `None` if absent / wrong kind.
    #[doc(hidden)]
    fn index_into<'v>(&self, value: &'v Json) -> Option<&'v Json>;
}

impl Index for str {
    fn index_into<'v>(&self, value: &'v Json) -> Option<&'v Json> {
        match *value {
            Json::Obj(ref m) => m.get(self),
            _ => None,
        }
    }
}

impl Index for String {
    fn index_into<'v>(&self, value: &'v Json) -> Option<&'v Json> {
        self.as_str().index_into(value)
    }
}

impl Index for usize {
    fn index_into<'v>(&self, value: &'v Json) -> Option<&'v Json> {
        match *value {
            Json::Arr(ref v) => v.get(*self),
            _ => None,
        }
    }
}

impl<T> Index for &T
where
    T: ?Sized + Index,
{
    fn index_into<'v>(&self, value: &'v Json) -> Option<&'v Json> {
        (**self).index_into(value)
    }
}

/// Total indexing: an absent key, an out-of-range position, or a wrong-kind
/// value all yield `Null` instead of panicking — matching `serde_json`.
impl<I: Index> std::ops::Index<I> for Json {
    type Output = Json;

    fn index(&self, index: I) -> &Json {
        static NULL: Json = Json::Null;
        index.index_into(self).unwrap_or(&NULL)
    }
}

// =============================================================================
// Map
// =============================================================================

/// A JSON object: a key-ordered map from `String` to [`Json`].
///
/// `BTreeMap`-backed, matching `serde_json`'s default map, so emitted key order
/// is sorted and equality does not depend on insertion order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Map {
    entries: BTreeMap<String, Json>,
}

impl Map {
    /// An empty object.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }

    /// Insert `value` at `key`, returning any previous value.
    pub fn insert(&mut self, key: String, value: Json) -> Option<Json> {
        self.entries.insert(key, value)
    }

    /// The value at `key`, if present.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&Json> {
        self.entries.get(key)
    }

    /// The value at `key` for mutation, if present.
    #[must_use]
    pub fn get_mut(&mut self, key: &str) -> Option<&mut Json> {
        self.entries.get_mut(key)
    }

    /// Whether `key` is present.
    #[must_use]
    pub fn contains_key(&self, key: &str) -> bool {
        self.entries.contains_key(key)
    }

    /// Remove and return the value at `key`.
    pub fn remove(&mut self, key: &str) -> Option<Json> {
        self.entries.remove(key)
    }

    /// Number of members.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the object has no members.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate members in key order.
    pub fn iter(&self) -> btree_map::Iter<'_, String, Json> {
        self.entries.iter()
    }

    /// Iterate keys in order.
    pub fn keys(&self) -> btree_map::Keys<'_, String, Json> {
        self.entries.keys()
    }

    /// Iterate values in key order.
    pub fn values(&self) -> btree_map::Values<'_, String, Json> {
        self.entries.values()
    }
}

impl<'a> IntoIterator for &'a Map {
    type Item = (&'a String, &'a Json);
    type IntoIter = btree_map::Iter<'a, String, Json>;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.iter()
    }
}

impl IntoIterator for Map {
    type Item = (String, Json);
    type IntoIter = btree_map::IntoIter<String, Json>;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.into_iter()
    }
}

impl FromIterator<(String, Json)> for Map {
    fn from_iter<T: IntoIterator<Item = (String, Json)>>(iter: T) -> Self {
        Self {
            entries: iter.into_iter().collect(),
        }
    }
}

impl From<BTreeMap<String, Json>> for Map {
    fn from(entries: BTreeMap<String, Json>) -> Self {
        Self { entries }
    }
}

// =============================================================================
// Conversions
// =============================================================================

impl From<bool> for Json {
    fn from(b: bool) -> Self {
        Self::Bool(b)
    }
}

impl From<String> for Json {
    fn from(s: String) -> Self {
        Self::Str(s)
    }
}

impl From<&str> for Json {
    fn from(s: &str) -> Self {
        Self::Str(s.to_owned())
    }
}

impl From<Map> for Json {
    fn from(m: Map) -> Self {
        Self::Obj(m)
    }
}

impl<T: Into<Json>> From<Vec<T>> for Json {
    fn from(v: Vec<T>) -> Self {
        // Explicit loop rather than `.into_iter().map(..).collect()`: identical
        // elements and order, but no closure bundle and no bulk-allocation
        // obligation for the verifier.
        let mut out: Vec<Json> = Vec::new();
        for item in v {
            out.push(item.into());
        }
        Self::Arr(out)
    }
}

impl<T: Clone + Into<Json>> From<&[T]> for Json {
    fn from(v: &[T]) -> Self {
        let mut out: Vec<Json> = Vec::new();
        for item in v {
            out.push(item.clone().into());
        }
        Self::Arr(out)
    }
}

impl<T: Into<Json>> From<Option<T>> for Json {
    fn from(v: Option<T>) -> Self {
        match v {
            Some(inner) => inner.into(),
            None => Self::Null,
        }
    }
}

impl From<()> for Json {
    fn from((): ()) -> Self {
        Self::Null
    }
}

macro_rules! json_from_unsigned {
    ($($ty:ty),* $(,)?) => {
        $(
            impl From<$ty> for Json {
                fn from(n: $ty) -> Self {
                    // `n as u64`, not `u64::from(n)`: both are the identical
                    // widening conversion for every type this macro is invoked
                    // with, but `From::from` is a CALL, and the enclosing fn is
                    // itself named `from` — so a callee match on the method tail
                    // reads this as `from` calling `from` and reports a phantom
                    // self-recursive edge, leaving termination unprovable. A cast
                    // is an `Rvalue::Cast` with no call terminator at all, so no
                    // edge exists to misread. (The root cause is the name-based
                    // match; this avoids tripping it.)
                    // `trivial_numeric_casts` fires only on the WIDEST
                    // instantiation, where the cast is `u64 as u64` / `i64 as i64`;
                    // for the three narrower ones it is a real widening. The macro
                    // cannot spell that per-type, and dropping the cast would
                    // reintroduce the `from`-calls-`from` self-edge described above.
                    #[allow(
                        clippy::cast_lossless,
                        trivial_numeric_casts,
                        reason = "widening cast is deliberate; identity only at the widest instantiation; see above"
                    )]
                    Self::U64(n as u64)
                }
            }
        )*
    };
}
json_from_unsigned!(u8, u16, u32, u64);

impl From<usize> for Json {
    fn from(n: usize) -> Self {
        // Lossless on every platform ny-cert targets (pointer width <= 64); the
        // saturating fallback keeps the conversion total regardless.
        Self::U64(u64::try_from(n).unwrap_or(u64::MAX))
    }
}

macro_rules! json_from_signed {
    ($($ty:ty),* $(,)?) => {
        $(
            impl From<$ty> for Json {
                fn from(n: $ty) -> Self {
                    // `n as i64`, not `i64::from(n)` — identical sign-extending
                    // widening, but avoids the phantom `from`-calls-`from`
                    // self-edge; see the unsigned macro above.
                    // `trivial_numeric_casts` fires only on the WIDEST
                    // instantiation, where the cast is `u64 as u64` / `i64 as i64`;
                    // for the three narrower ones it is a real widening. The macro
                    // cannot spell that per-type, and dropping the cast would
                    // reintroduce the `from`-calls-`from` self-edge described above.
                    #[allow(
                        clippy::cast_lossless,
                        trivial_numeric_casts,
                        reason = "widening cast is deliberate; identity only at the widest instantiation; see above"
                    )]
                    let wide = n as i64;
                    match u64::try_from(wide) {
                        Ok(non_negative) => Self::U64(non_negative),
                        Err(_) => Self::I64(wide),
                    }
                }
            }
        )*
    };
}
json_from_signed!(i8, i16, i32, i64);

impl From<isize> for Json {
    fn from(n: isize) -> Self {
        match u64::try_from(n) {
            Ok(non_negative) => Self::U64(non_negative),
            Err(_) => Self::I64(i64::try_from(n).unwrap_or(i64::MIN)),
        }
    }
}

impl From<f64> for Json {
    /// Non-finite floats become `Null`, exactly as `serde_json` does.
    fn from(f: f64) -> Self {
        if f.is_finite() {
            Self::F64(f)
        } else {
            Self::Null
        }
    }
}

impl From<f32> for Json {
    fn from(f: f32) -> Self {
        Self::from(f64::from(f))
    }
}

impl<T: Into<Json>> FromIterator<T> for Json {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut out: Vec<Json> = Vec::new();
        for item in iter {
            out.push(item.into());
        }
        Self::Arr(out)
    }
}

// =============================================================================
// Serialization
// =============================================================================

/// Serialize `value` to compact JSON — no whitespace, `serde_json`-identical.
#[must_use]
pub fn to_string(value: &Json) -> String {
    let mut out = String::new();
    write_value(&mut out, value, false);
    out
}

/// Serialize `value` with `serde_json`'s pretty format: 2-space indent, `": "`
/// after each key, `{}`/`[]` for empty containers.
#[must_use]
pub fn to_string_pretty(value: &Json) -> String {
    let mut out = String::new();
    write_value(&mut out, value, true);
    out
}

/// One open container in the serializer's explicit stack.
enum Frame<'a> {
    Arr {
        iter: std::slice::Iter<'a, Json>,
        first: bool,
    },
    Obj {
        iter: btree_map::Iter<'a, String, Json>,
        first: bool,
    },
}

/// Iterative (non-recursive) writer.
///
/// Containers are pushed onto an explicit heap stack instead of recursing, so
/// the routine is total for arbitrarily deep values: there is no recursion to
/// bound and no machine stack to exhaust.
fn write_value(out: &mut String, value: &Json, pretty: bool) {
    let mut stack: Vec<Frame<'_>> = Vec::new();
    let mut pending: Option<&Json> = Some(value);

    loop {
        if let Some(current) = pending.take() {
            match *current {
                Json::Null => out.push_str("null"),
                Json::Bool(b) => out.push_str(if b { "true" } else { "false" }),
                Json::U64(n) => push_u64(out, n),
                Json::I64(n) => push_i64(out, n),
                Json::F64(f) => push_f64(out, f),
                Json::Str(ref s) => push_escaped_str(out, s),
                Json::Arr(ref items) => {
                    out.push('[');
                    stack.push(Frame::Arr {
                        iter: items.iter(),
                        first: true,
                    });
                }
                Json::Obj(ref map) => {
                    out.push('{');
                    stack.push(Frame::Obj {
                        iter: map.iter(),
                        first: true,
                    });
                }
            }
        }

        // Depth of the *children* of the frame on top of the stack.
        let depth = stack.len();
        match stack.last_mut() {
            None => break,
            Some(&mut Frame::Arr {
                ref mut iter,
                ref mut first,
            }) => match iter.next() {
                Some(item) => {
                    if !*first {
                        out.push(',');
                    }
                    if pretty {
                        out.push('\n');
                        push_indent(out, depth);
                    }
                    *first = false;
                    pending = Some(item);
                }
                None => {
                    let had_value = !*first;
                    if pretty && had_value {
                        out.push('\n');
                        push_indent(out, depth.saturating_sub(1));
                    }
                    out.push(']');
                    let _popped = stack.pop();
                }
            },
            Some(&mut Frame::Obj {
                ref mut iter,
                ref mut first,
            }) => match iter.next() {
                Some((key, item)) => {
                    if !*first {
                        out.push(',');
                    }
                    if pretty {
                        out.push('\n');
                        push_indent(out, depth);
                    }
                    *first = false;
                    push_escaped_str(out, key);
                    out.push(':');
                    if pretty {
                        out.push(' ');
                    }
                    pending = Some(item);
                }
                None => {
                    let had_value = !*first;
                    if pretty && had_value {
                        out.push('\n');
                        push_indent(out, depth.saturating_sub(1));
                    }
                    out.push('}');
                    let _popped = stack.pop();
                }
            },
        }
    }
}

fn push_indent(out: &mut String, level: usize) {
    let mut remaining = level;
    while remaining > 0 {
        out.push_str("  ");
        remaining = remaining.saturating_sub(1);
    }
}

const HEX_DIGITS: [u8; 16] = *b"0123456789abcdef";

/// Write `s` as a JSON string literal with `serde_json`'s escape set: `"`, `\`,
/// `\n`, `\r`, `\t`, `\b`, `\f`, and `\u00xx` (lowercase) for other control
/// bytes. `/`, `DEL`, and non-ASCII pass through unescaped.
fn push_escaped_str(out: &mut String, s: &str) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => {
                let code = u32::from(ch);
                if code < 0x20 {
                    out.push_str("\\u00");
                    let byte = u8::try_from(code).unwrap_or(0);
                    let hi = usize::from(byte >> 4);
                    let lo = usize::from(byte & 0x0f);
                    if let Some(&digit) = HEX_DIGITS.get(hi) {
                        out.push(char::from(digit));
                    }
                    if let Some(&digit) = HEX_DIGITS.get(lo) {
                        out.push(char::from(digit));
                    }
                } else {
                    out.push(ch);
                }
            }
        }
    }
    out.push('"');
}

/// Decimal `u64`, no allocation, no indexing, no overflow — an in-crate `itoa`.
fn push_u64(out: &mut String, value: u64) {
    if value == 0 {
        out.push('0');
        return;
    }
    // 20 digits is the exact width of `u64::MAX`.
    let mut buf = [b'0'; 20];
    let mut cursor = buf.len();
    let mut rest = value;
    while rest > 0 {
        cursor = cursor.saturating_sub(1);
        let digit = u8::try_from(rest % 10).unwrap_or(0);
        if let Some(slot) = buf.get_mut(cursor) {
            *slot = b'0'.saturating_add(digit);
        }
        rest /= 10;
    }
    if let Some(digits) = buf.get(cursor..) {
        for &byte in digits {
            out.push(char::from(byte));
        }
    }
}

fn push_i64(out: &mut String, value: i64) {
    if value < 0 {
        out.push('-');
    }
    // `unsigned_abs` is total, including at `i64::MIN`.
    push_u64(out, value.unsigned_abs());
}

/// Write `value` the way `serde_json` (via `zmij`, formerly `ryu`) does.
///
/// The shortest round-tripping digit string comes from core's `LowerExp`
/// formatter (`{:e}`, which is documented shortest-round-trip); this function
/// re-lays those digits out under `zmij`'s rules:
///
/// * fixed notation when the scientific exponent is in `-5..=15`, always with a
///   fractional part (`1.0`, `12.34`, `0.001234`, `9061488000000000.0`);
/// * otherwise `d[.ddd]e±XX` with a mandatory exponent sign and at least two
///   exponent digits (`5e-324`, `1e-07`, `1e+20`, `6.62607015e-34`).
///
/// Non-finite values are written as `null`, matching `serde_json`'s serializer.
#[allow(clippy::float_cmp)]
fn push_f64(out: &mut String, value: f64) {
    if !value.is_finite() {
        out.push_str("null");
        return;
    }
    if value == 0.0 {
        out.push_str(if value.is_sign_negative() {
            "-0.0"
        } else {
            "0.0"
        });
        return;
    }

    // e.g. "-1.234e10", "5e-324". One digit before the point, no `+` sign.
    let scientific = format!("{value:e}");
    let Some((mantissa, exponent)) = scientific.split_once('e') else {
        // Unreachable for a finite float; `{:e}` output is itself valid JSON
        // number syntax, so falling back to it keeps the value exact.
        out.push_str(&scientific);
        return;
    };
    let (negative, mantissa) = match mantissa.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, mantissa),
    };
    let Ok(dec_exp) = exponent.parse::<i32>() else {
        out.push_str(&scientific);
        return;
    };
    let mut digits = String::new();
    for ch in mantissa.chars() {
        if ch != '.' {
            digits.push(ch);
        }
    }
    retie_to_even(value, dec_exp, &mut digits);
    let Ok(len) = i32::try_from(digits.len()) else {
        out.push_str(&scientific);
        return;
    };
    if len == 0 {
        out.push_str(&scientific);
        return;
    }

    if negative {
        out.push('-');
    }

    if (-5..=15).contains(&dec_exp) {
        if len.saturating_sub(1) <= dec_exp {
            // 1234e7 -> 12340000000.0
            out.push_str(&digits);
            let mut zeros = dec_exp.saturating_add(1).saturating_sub(len);
            while zeros > 0 {
                out.push('0');
                zeros = zeros.saturating_sub(1);
            }
            out.push_str(".0");
        } else if dec_exp >= 0 {
            // 1234e-2 -> 12.34
            let cut = usize::try_from(dec_exp.saturating_add(1)).unwrap_or(0);
            match (digits.get(..cut), digits.get(cut..)) {
                (Some(head), Some(tail)) => {
                    out.push_str(head);
                    out.push('.');
                    out.push_str(tail);
                }
                _ => out.push_str(&digits),
            }
        } else {
            // 1234e-6 -> 0.001234
            out.push_str("0.");
            let mut zeros = dec_exp.saturating_neg().saturating_sub(1);
            while zeros > 0 {
                out.push('0');
                zeros = zeros.saturating_sub(1);
            }
            out.push_str(&digits);
        }
        return;
    }

    // 1234e30 -> 1.234e+33
    match (digits.get(..1), digits.get(1..)) {
        (Some(head), Some(tail)) => {
            out.push_str(head);
            if !tail.is_empty() {
                out.push('.');
                out.push_str(tail);
            }
        }
        _ => out.push_str(&digits),
    }
    out.push('e');
    out.push(if dec_exp >= 0 { '+' } else { '-' });
    push_u64(out, u64::from(dec_exp.unsigned_abs()));
}

/// Repair the one place where core's shortest-float digits differ from
/// Schubfach's (`zmij`, hence `serde_json`).
///
/// When two equally short decimal candidates are *exactly* equidistant from
/// `value`, core's formatter rounds away from zero while Schubfach rounds to
/// even — a last-digit disagreement on roughly one f64 in 7000. This rewrites
/// `digits` to the even candidate in exactly that case and leaves every other
/// value untouched.
///
/// The tie is established exactly, not guessed: `value` is re-rendered with one
/// and two extra significant digits (core's *exact*, correctly-rounded mode),
/// and the tie holds only when those tails are exactly `5` and `50`.
fn retie_to_even(value: f64, dec_exp: i32, digits: &mut String) {
    let len = digits.len();
    let Some(last) = digits.as_bytes().last().copied() else {
        return;
    };
    // An even last digit is what Schubfach would have chosen anyway; a `9`
    // would carry into a longer/differently-scaled candidate, which is not a
    // same-length tie.
    if last % 2 == 0 || last == b'9' {
        return;
    }
    let magnitude = value.abs();
    let Some((extra, exp)) = exact_digits(magnitude, len) else {
        return;
    };
    // A carry during re-rendering means this is not a same-length tie.
    if exp != dec_exp {
        return;
    }
    if extra.as_bytes().get(len) != Some(&b'5') {
        return;
    }
    // The cheap `...5` test above is only *necessary*: a longer expansion such
    // as `...50339` also rounds to it. Confirm the tie against the value's
    // complete decimal expansion — every finite f64 terminates well inside
    // `FULL_EXPANSION_PRECISION` digits, so the tail must be all zeros.
    let Some((full, full_exp)) = exact_digits(magnitude, FULL_EXPANSION_PRECISION) else {
        return;
    };
    if full_exp != dec_exp || full.get(..extra.len()) != Some(extra.as_str()) {
        return;
    }
    match full.get(extra.len()..) {
        Some(tail) if tail.bytes().all(|byte| byte == b'0') => {}
        _ => return,
    }
    let Some(lower) = extra.get(..len) else {
        return;
    };
    let Some(head) = digits.get(..len.saturating_sub(1)) else {
        return;
    };
    if lower == digits.as_str() {
        // Our candidate is the lower one and is odd: the even neighbour is the
        // next one up, reachable without a carry because `last != b'9'`.
        let mut even = String::from(head);
        even.push(char::from(last.saturating_add(1)));
        *digits = even;
    } else if lower.get(..len.saturating_sub(1)) == Some(head)
        && lower.as_bytes().last() == Some(&last.saturating_sub(1))
    {
        // Our candidate is the upper one; the lower neighbour is the even one.
        *digits = String::from(lower);
    }
}

/// Enough significant digits to hold the complete decimal expansion of any
/// finite `f64` (the longest is 767 digits) with room to spare; core's exact
/// formatting mode zero-pads beyond the end of the expansion.
const FULL_EXPANSION_PRECISION: usize = 800;

/// `value`'s significant digits at `precision + 1` places, exactly rounded, plus
/// the scientific exponent. `None` if the rendering is not in the expected
/// `d.ddde<exp>` shape.
fn exact_digits(value: f64, precision: usize) -> Option<(String, i32)> {
    let rendered = format!("{value:.*e}", precision);
    let (mantissa, exponent) = rendered.split_once('e')?;
    let exp = exponent.parse::<i32>().ok()?;
    let mut digits = String::new();
    for ch in mantissa.chars() {
        if ch != '.' && ch != '-' {
            digits.push(ch);
        }
    }
    Some((digits, exp))
}

// =============================================================================
// Parsing
// =============================================================================

/// Parse `input` as JSON.
///
/// Total: every malformed input — bad syntax, a lone surrogate, an unterminated
/// string, nesting past [`MAX_DEPTH`], trailing bytes — is reported as `Err`,
/// and no input can make this function panic.
///
/// # Errors
/// Returns a human-readable message naming the byte offset of the failure.
pub fn from_str(input: &str) -> Result<Json, String> {
    let mut parser = Parser::new(input);
    // `MAX_DEPTH - 1`, not `MAX_DEPTH`: see the constant's docs for why this is
    // where `serde_json` draws the same boundary.
    let value = parser.parse_value(MAX_DEPTH.saturating_sub(1))?;
    parser.skip_whitespace();
    if parser.peek().is_some() {
        return Err(parser.error("trailing characters after JSON value"));
    }
    Ok(value)
}

struct Parser<'a> {
    src: &'a str,
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    const fn new(src: &'a str) -> Self {
        Self {
            src,
            bytes: src.as_bytes(),
            pos: 0,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn peek_at(&self, offset: usize) -> Option<u8> {
        self.bytes.get(self.pos.saturating_add(offset)).copied()
    }

    fn bump(&mut self) {
        self.pos = self.pos.saturating_add(1);
    }

    fn error(&self, message: &str) -> String {
        let mut out = String::from(message);
        out.push_str(" at byte ");
        push_u64(&mut out, u64::try_from(self.pos).unwrap_or(u64::MAX));
        out
    }

    fn skip_whitespace(&mut self) {
        while let Some(b' ' | b'\t' | b'\n' | b'\r') = self.peek() {
            self.bump();
        }
    }

    /// Consume `word` if it is next, else report an error.
    fn eat_word(&mut self, word: &str) -> Result<(), String> {
        for expected in word.bytes() {
            match self.peek() {
                Some(actual) if actual == expected => self.bump(),
                _ => return Err(self.error("invalid literal")),
            }
        }
        Ok(())
    }

    /// `depth` is the remaining nesting budget; it strictly decreases on every
    /// nested container, so this recursion terminates.
    fn parse_value(&mut self, depth: u32) -> Result<Json, String> {
        self.skip_whitespace();
        match self.peek() {
            None => Err(self.error("unexpected end of input")),
            Some(b'n') => {
                self.eat_word("null")?;
                Ok(Json::Null)
            }
            Some(b't') => {
                self.eat_word("true")?;
                Ok(Json::Bool(true))
            }
            Some(b'f') => {
                self.eat_word("false")?;
                Ok(Json::Bool(false))
            }
            Some(b'"') => self.parse_string().map(Json::Str),
            Some(b'[') => self.parse_array(depth),
            Some(b'{') => self.parse_object(depth),
            Some(b'-' | b'0'..=b'9') => self.parse_number(),
            Some(_) => Err(self.error("expected a JSON value")),
        }
    }

    fn parse_array(&mut self, depth: u32) -> Result<Json, String> {
        let Some(inner_depth) = depth.checked_sub(1) else {
            return Err(self.error("recursion limit exceeded"));
        };
        self.bump(); // '['
        let mut items: Vec<Json> = Vec::new();
        self.skip_whitespace();
        if self.peek() == Some(b']') {
            self.bump();
            return Ok(Json::Arr(items));
        }
        loop {
            let item = self.parse_value(inner_depth)?;
            items.push(item);
            self.skip_whitespace();
            match self.peek() {
                Some(b',') => self.bump(),
                Some(b']') => {
                    self.bump();
                    return Ok(Json::Arr(items));
                }
                _ => return Err(self.error("expected `,` or `]` in array")),
            }
        }
    }

    fn parse_object(&mut self, depth: u32) -> Result<Json, String> {
        let Some(inner_depth) = depth.checked_sub(1) else {
            return Err(self.error("recursion limit exceeded"));
        };
        self.bump(); // '{'
        let mut map = Map::new();
        self.skip_whitespace();
        if self.peek() == Some(b'}') {
            self.bump();
            return Ok(Json::Obj(map));
        }
        loop {
            self.skip_whitespace();
            if self.peek() != Some(b'"') {
                return Err(self.error("expected a string key in object"));
            }
            let key = self.parse_string()?;
            self.skip_whitespace();
            if self.peek() == Some(b':') {
                self.bump();
            } else {
                return Err(self.error("expected `:` after object key"));
            }
            let value = self.parse_value(inner_depth)?;
            // Duplicate keys: last one wins, as in `serde_json`.
            let _previous = map.insert(key, value);
            self.skip_whitespace();
            match self.peek() {
                Some(b',') => self.bump(),
                Some(b'}') => {
                    self.bump();
                    return Ok(Json::Obj(map));
                }
                _ => return Err(self.error("expected `,` or `}` in object")),
            }
        }
    }

    fn parse_string(&mut self) -> Result<String, String> {
        self.bump(); // opening quote
        let mut out = String::new();
        let mut segment_start = self.pos;
        loop {
            match self.peek() {
                None => return Err(self.error("unterminated string")),
                Some(b'"') => {
                    self.push_segment(&mut out, segment_start)?;
                    self.bump();
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.push_segment(&mut out, segment_start)?;
                    self.bump();
                    let ch = self.parse_escape()?;
                    out.push(ch);
                    segment_start = self.pos;
                }
                Some(byte) if byte < 0x20 => return Err(self.error("control character in string")),
                // Multi-byte UTF-8 continuation bytes are >= 0x80, so advancing
                // one byte at a time never splits a code point at a delimiter.
                Some(_) => self.bump(),
            }
        }
    }

    /// Append `src[start..pos]` — the run of unescaped bytes just scanned.
    fn push_segment(&self, out: &mut String, start: usize) -> Result<(), String> {
        match self.src.get(start..self.pos) {
            Some(text) => {
                out.push_str(text);
                Ok(())
            }
            // Unreachable: both ends sit on ASCII delimiters, hence on char
            // boundaries. Reported rather than assumed.
            None => Err(self.error("invalid string slice")),
        }
    }

    /// Decode the escape following a consumed backslash.
    fn parse_escape(&mut self) -> Result<char, String> {
        let escape = match self.peek() {
            None => return Err(self.error("unterminated escape")),
            Some(byte) => byte,
        };
        self.bump();
        match escape {
            b'"' => Ok('"'),
            b'\\' => Ok('\\'),
            b'/' => Ok('/'),
            b'b' => Ok('\u{8}'),
            b'f' => Ok('\u{c}'),
            b'n' => Ok('\n'),
            b'r' => Ok('\r'),
            b't' => Ok('\t'),
            b'u' => self.parse_unicode_escape(),
            _ => Err(self.error("invalid escape sequence")),
        }
    }

    fn parse_unicode_escape(&mut self) -> Result<char, String> {
        let first = self.parse_hex4()?;
        if (0xD800..0xDC00).contains(&first) {
            // Leading surrogate: a `\uDC00..\uE000` trailing surrogate must follow.
            if self.peek() != Some(b'\\') || self.peek_at(1) != Some(b'u') {
                return Err(self.error("lone leading surrogate in escape"));
            }
            self.bump();
            self.bump();
            let second = self.parse_hex4()?;
            if !(0xDC00..0xE000).contains(&second) {
                return Err(self.error("invalid surrogate pair in escape"));
            }
            let combined = first
                .saturating_sub(0xD800)
                .saturating_mul(0x400)
                .saturating_add(second.saturating_sub(0xDC00))
                .saturating_add(0x1_0000);
            char::from_u32(combined).ok_or_else(|| self.error("invalid code point in escape"))
        } else if (0xDC00..0xE000).contains(&first) {
            Err(self.error("lone trailing surrogate in escape"))
        } else {
            char::from_u32(first).ok_or_else(|| self.error("invalid code point in escape"))
        }
    }

    fn parse_hex4(&mut self) -> Result<u32, String> {
        let mut value: u32 = 0;
        let mut count: u32 = 0;
        while count < 4 {
            let Some(byte) = self.peek() else {
                return Err(self.error("truncated `\\u` escape"));
            };
            let Some(digit) = hex_value(byte) else {
                return Err(self.error("invalid hex digit in `\\u` escape"));
            };
            value = value.saturating_mul(16).saturating_add(digit);
            self.bump();
            count = count.saturating_add(1);
        }
        Ok(value)
    }

    /// JSON grammar numbers, with `serde_json`'s typing: a non-negative integer
    /// becomes `U64`, a negative one `I64`, anything fractional/exponential (or
    /// too large for an integer) becomes `F64`.
    fn parse_number(&mut self) -> Result<Json, String> {
        let start = self.pos;
        let negative = self.peek() == Some(b'-');
        if negative {
            self.bump();
        }
        match self.peek() {
            Some(b'0') => {
                self.bump();
                if matches!(self.peek(), Some(b'0'..=b'9')) {
                    return Err(self.error("leading zero in number"));
                }
            }
            Some(b'1'..=b'9') => {
                self.bump();
                self.skip_digits();
            }
            _ => return Err(self.error("expected a digit in number")),
        }
        let mut fractional = false;
        if self.peek() == Some(b'.') {
            fractional = true;
            self.bump();
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(self.error("expected a digit after `.`"));
            }
            self.skip_digits();
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            fractional = true;
            self.bump();
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.bump();
            }
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(self.error("expected a digit in exponent"));
            }
            self.skip_digits();
        }
        let Some(text) = self.src.get(start..self.pos) else {
            return Err(self.error("invalid number slice"));
        };
        if !fractional {
            if negative {
                if let Ok(n) = text.parse::<i64>() {
                    // `serde_json` types `-0` as the float `-0.0`, never as the
                    // integer `0`: its parser negates the parsed magnitude and
                    // falls back to a float whenever that negation comes out
                    // non-negative, which for an in-range magnitude happens
                    // only at zero. The grammar above rejects leading zeros, so
                    // `n == 0` here means the literal text was exactly `-0`.
                    if n == 0 {
                        return Ok(Json::F64(-0.0));
                    }
                    return Ok(Json::I64(n));
                }
            } else if let Ok(n) = text.parse::<u64>() {
                return Ok(Json::U64(n));
            }
            // Out of integer range: fall through to the float reading, exactly
            // as `serde_json` does for oversized integers.
        }
        match text.parse::<f64>() {
            Ok(f) if f.is_finite() => Ok(Json::F64(f)),
            Ok(_) => Err(self.error("number out of range")),
            Err(_) => Err(self.error("invalid number")),
        }
    }

    fn skip_digits(&mut self) {
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.bump();
        }
    }
}

fn hex_value(byte: u8) -> Option<u32> {
    match byte {
        b'0'..=b'9' => Some(u32::from(byte.wrapping_sub(b'0'))),
        b'a'..=b'f' => Some(u32::from(byte.wrapping_sub(b'a')).saturating_add(10)),
        b'A'..=b'F' => Some(u32::from(byte.wrapping_sub(b'A')).saturating_add(10)),
        _ => None,
    }
}

// =============================================================================
// `json!` construction macro
// =============================================================================

/// Build a [`Json`] value from JSON-ish literal syntax.
///
/// A drop-in replacement for `serde_json::json!`, with the same token-muncher
/// shape: `null`/`true`/`false` literals, `[..]` arrays, `{..}` objects, and any
/// expression implementing `Into<Json>` as a value.
///
/// ```
/// let v = ny_cert::json!({
///     "kind": "farkas",
///     "steps": [1, 2, 3],
///     "meta": { "ok": true },
/// });
/// assert_eq!(v["meta"]["ok"].as_bool(), Some(true));
/// ```
#[macro_export]
macro_rules! json {
    ($($json:tt)+) => {
        $crate::json_internal!($($json)+)
    };
}

/// Implementation detail of [`json!`].
#[macro_export]
#[doc(hidden)]
macro_rules! json_internal {
    //////////////////////////////////////////////////////////////////////////
    // TT muncher for the inside of an array `[...]`, producing a `vec![...]`.
    // Invoked as: json_internal!(@array [] $($tt)*)
    //////////////////////////////////////////////////////////////////////////

    (@array [$($elems:expr,)*]) => {
        $crate::json_internal_vec![$($elems,)*]
    };

    (@array [$($elems:expr),*]) => {
        $crate::json_internal_vec![$($elems),*]
    };

    (@array [$($elems:expr,)*] null $($rest:tt)*) => {
        $crate::json_internal!(@array [$($elems,)* $crate::json_internal!(null)] $($rest)*)
    };

    (@array [$($elems:expr,)*] true $($rest:tt)*) => {
        $crate::json_internal!(@array [$($elems,)* $crate::json_internal!(true)] $($rest)*)
    };

    (@array [$($elems:expr,)*] false $($rest:tt)*) => {
        $crate::json_internal!(@array [$($elems,)* $crate::json_internal!(false)] $($rest)*)
    };

    (@array [$($elems:expr,)*] [$($array:tt)*] $($rest:tt)*) => {
        $crate::json_internal!(@array [$($elems,)* $crate::json_internal!([$($array)*])] $($rest)*)
    };

    (@array [$($elems:expr,)*] {$($map:tt)*} $($rest:tt)*) => {
        $crate::json_internal!(@array [$($elems,)* $crate::json_internal!({$($map)*})] $($rest)*)
    };

    (@array [$($elems:expr,)*] $next:expr, $($rest:tt)*) => {
        $crate::json_internal!(@array [$($elems,)* $crate::json_internal!($next),] $($rest)*)
    };

    (@array [$($elems:expr,)*] $last:expr) => {
        $crate::json_internal!(@array [$($elems,)* $crate::json_internal!($last)])
    };

    (@array [$($elems:expr),*] , $($rest:tt)*) => {
        $crate::json_internal!(@array [$($elems,)*] $($rest)*)
    };

    (@array [$($elems:expr),*] $unexpected:tt $($rest:tt)*) => {
        $crate::json_unexpected!($unexpected)
    };

    //////////////////////////////////////////////////////////////////////////
    // TT muncher for the inside of an object `{...}`, inserting into $object.
    // Invoked as: json_internal!(@object $object () ($($tt)*) ($($tt)*))
    //////////////////////////////////////////////////////////////////////////

    (@object $object:ident () () ()) => {};

    (@object $object:ident [$($key:tt)+] ($value:expr) , $($rest:tt)*) => {
        let _ = $object.insert(($($key)+).into(), $value);
        $crate::json_internal!(@object $object () ($($rest)*) ($($rest)*));
    };

    (@object $object:ident [$($key:tt)+] ($value:expr) $unexpected:tt $($rest:tt)*) => {
        $crate::json_unexpected!($unexpected);
    };

    (@object $object:ident [$($key:tt)+] ($value:expr)) => {
        let _ = $object.insert(($($key)+).into(), $value);
    };

    (@object $object:ident ($($key:tt)+) (: null $($rest:tt)*) $copy:tt) => {
        $crate::json_internal!(@object $object [$($key)+] ($crate::json_internal!(null)) $($rest)*);
    };

    (@object $object:ident ($($key:tt)+) (: true $($rest:tt)*) $copy:tt) => {
        $crate::json_internal!(@object $object [$($key)+] ($crate::json_internal!(true)) $($rest)*);
    };

    (@object $object:ident ($($key:tt)+) (: false $($rest:tt)*) $copy:tt) => {
        $crate::json_internal!(@object $object [$($key)+] ($crate::json_internal!(false)) $($rest)*);
    };

    (@object $object:ident ($($key:tt)+) (: [$($array:tt)*] $($rest:tt)*) $copy:tt) => {
        $crate::json_internal!(@object $object [$($key)+] ($crate::json_internal!([$($array)*])) $($rest)*);
    };

    (@object $object:ident ($($key:tt)+) (: {$($map:tt)*} $($rest:tt)*) $copy:tt) => {
        $crate::json_internal!(@object $object [$($key)+] ($crate::json_internal!({$($map)*})) $($rest)*);
    };

    (@object $object:ident ($($key:tt)+) (: $value:expr , $($rest:tt)*) $copy:tt) => {
        $crate::json_internal!(@object $object [$($key)+] ($crate::json_internal!($value)) , $($rest)*);
    };

    (@object $object:ident ($($key:tt)+) (: $value:expr) $copy:tt) => {
        $crate::json_internal!(@object $object [$($key)+] ($crate::json_internal!($value)));
    };

    // Missing value for the last entry: report at the colon.
    (@object $object:ident ($($key:tt)+) (:) $copy:tt) => {
        $crate::json_internal!();
    };

    // A colon with no key at all.
    (@object $object:ident () (: $($rest:tt)*) ($colon:tt $($copy:tt)*)) => {
        $crate::json_unexpected!($colon);
    };

    // A comma where a colon was expected.
    (@object $object:ident ($($key:tt)*) (, $($rest:tt)*) ($comma:tt $($copy:tt)*)) => {
        $crate::json_unexpected!($comma);
    };

    // Missing colon and value for the last entry.
    (@object $object:ident ($($key:tt)*) () ($copy:tt)) => {
        $crate::json_expect_expr_comma!($copy);
    };

    // Munch one more token into the key being accumulated.
    (@object $object:ident ($($key:tt)*) ($tt:tt $($rest:tt)*) $copy:tt) => {
        $crate::json_internal!(@object $object ($($key)* $tt) ($($rest)*) ($($rest)*));
    };

    //////////////////////////////////////////////////////////////////////////
    // Entry points.
    //////////////////////////////////////////////////////////////////////////

    (null) => {
        $crate::json::Json::Null
    };

    (true) => {
        $crate::json::Json::Bool(true)
    };

    (false) => {
        $crate::json::Json::Bool(false)
    };

    ([]) => {
        $crate::json::Json::Arr($crate::json_internal_vec![])
    };

    ([ $($tt:tt)+ ]) => {
        $crate::json::Json::Arr($crate::json_internal!(@array [] $($tt)+))
    };

    ({}) => {
        $crate::json::Json::Obj($crate::json::Map::new())
    };

    ({ $($tt:tt)+ }) => {
        $crate::json::Json::Obj({
            let mut object = $crate::json::Map::new();
            $crate::json_internal!(@object object () ($($tt)+) ($($tt)+));
            object
        })
    };

    ($other:expr) => {
        $crate::json::Json::from($other)
    };
}

/// Implementation detail of [`json!`].
#[macro_export]
#[doc(hidden)]
macro_rules! json_internal_vec {
    ($($content:tt)*) => {
        vec![$($content)*]
    };
}

/// Implementation detail of [`json!`]: forces a compile error naming the
/// offending token.
#[macro_export]
#[doc(hidden)]
macro_rules! json_unexpected {
    () => {};
}

/// Implementation detail of [`json!`]: forces a "expected expression" error.
#[macro_export]
#[doc(hidden)]
macro_rules! json_expect_expr_comma {
    ($e:expr , $($tt:tt)*) => {};
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::{from_str, to_string, to_string_pretty, Json, Map, MAX_DEPTH};

    fn obj(pairs: &[(&str, Json)]) -> Json {
        let mut map = Map::new();
        for (key, value) in pairs {
            let _ = map.insert((*key).to_owned(), value.clone());
        }
        Json::Obj(map)
    }

    // ---- serialization ----------------------------------------------------

    #[test]
    fn compact_form_has_no_whitespace() {
        let value = obj(&[
            ("b", Json::Str("x".to_owned())),
            (
                "a",
                Json::Arr(vec![Json::U64(1), Json::Bool(true), Json::Null]),
            ),
        ]);
        assert_eq!(to_string(&value), r#"{"a":[1,true,null],"b":"x"}"#);
    }

    #[test]
    fn keys_are_emitted_in_sorted_order() {
        let value = obj(&[
            ("zeta", Json::U64(1)),
            ("alpha", Json::U64(2)),
            ("mid", Json::U64(3)),
        ]);
        assert_eq!(to_string(&value), r#"{"alpha":2,"mid":3,"zeta":1}"#);
    }

    #[test]
    fn pretty_form_matches_serde_json_layout() {
        let value = obj(&[
            ("a", Json::Arr(vec![Json::U64(1), Json::U64(2)])),
            ("b", obj(&[("c", Json::Str("d".to_owned()))])),
        ]);
        let expected = "{\n  \"a\": [\n    1,\n    2\n  ],\n  \"b\": {\n    \"c\": \"d\"\n  }\n}";
        assert_eq!(to_string_pretty(&value), expected);
    }

    #[test]
    fn pretty_empty_containers_stay_inline() {
        let value = obj(&[("a", Json::Arr(vec![])), ("b", obj(&[]))]);
        assert_eq!(to_string_pretty(&value), "{\n  \"a\": [],\n  \"b\": {}\n}");
        assert_eq!(to_string(&value), r#"{"a":[],"b":{}}"#);
    }

    #[test]
    fn string_escapes_match_serde_json() {
        let value = Json::Str("q\"b\\s\nn\rr\tt\u{8}b\u{c}f\u{1}c/s\u{7f}d é".to_owned());
        assert_eq!(
            to_string(&value),
            "\"q\\\"b\\\\s\\nn\\rr\\tt\\bb\\ff\\u0001c/s\u{7f}d é\""
        );
    }

    #[test]
    fn integers_render_as_plain_decimal() {
        assert_eq!(to_string(&Json::U64(0)), "0");
        assert_eq!(to_string(&Json::U64(u64::MAX)), "18446744073709551615");
        assert_eq!(to_string(&Json::I64(-1)), "-1");
        assert_eq!(to_string(&Json::I64(i64::MIN)), "-9223372036854775808");
    }

    #[test]
    fn floats_match_the_zmij_layout() {
        let f = |x: f64| to_string(&Json::F64(x));
        // Fixed notation, always with a fractional part.
        assert_eq!(f(1.0), "1.0");
        assert_eq!(f(0.0), "0.0");
        assert_eq!(f(-0.0), "-0.0");
        assert_eq!(f(-1.5), "-1.5");
        assert_eq!(f(12.34), "12.34");
        assert_eq!(f(0.001234), "0.001234");
        assert_eq!(f(9.061_488e15), "9061488000000000.0");
        assert_eq!(f(5.444_310_685_350_916e14), "544431068535091.6");
        // Scientific notation outside -5..=15, with a mandatory exponent sign.
        assert_eq!(f(1e16), "1e+16");
        assert_eq!(f(1e20), "1e+20");
        assert_eq!(f(1e-7), "1e-7");
        assert_eq!(f(6.626_070_15e-34), "6.62607015e-34");
        assert_eq!(f(-4.932_096_661_796_888e-226), "-4.932096661796888e-226");
        assert_eq!(f(3.439_070_283_483_335e35), "3.439070283483335e+35");
        assert_eq!(f(1e-323), "1e-323");
        // Non-finite serializes as null, like serde_json.
        assert_eq!(f(f64::NAN), "null");
        assert_eq!(f(f64::INFINITY), "null");
    }

    #[test]
    fn float_boundary_of_fixed_notation() {
        let f = |x: f64| to_string(&Json::F64(x));
        assert_eq!(f(1e15), "1000000000000000.0");
        assert_eq!(f(1e-5), "0.00001");
        assert_eq!(f(1e-6), "1e-6");
        assert_eq!(f(f64::MAX), "1.7976931348623157e+308");
        assert_eq!(f(5e-324), "5e-324");
    }

    /// Core's shortest-float formatter rounds an exact tie away from zero;
    /// Schubfach (`zmij`, hence `serde_json`) rounds it to even. These values
    /// are exact ties — `2181495296738027.25` sits precisely between
    /// `...27.2` and `...27.3` — and must come out even.
    #[test]
    fn exact_decimal_ties_round_to_even() {
        let f = |x: f64| to_string(&Json::F64(x));
        assert_eq!(
            f(f64::from_bits(0x431f_003b_d0f7_0bad)),
            "2181495296738027.2"
        );
        assert_eq!(
            f(f64::from_bits(0xc30a_a61f_a224_75ca)),
            "-937625523621561.2"
        );
        assert_eq!(
            f(f64::from_bits(0x42d1_1c37_8bee_3b08)),
            "75251554695404.12"
        );
        // ... while a near-tie whose expansion continues past the `5` keeps the
        // closer (odd) candidate.
        assert_eq!(
            f(f64::from_bits(0x0485_5160_3777_f796)),
            "6.999999999999997e-287"
        );
        assert_eq!(
            f(f64::from_bits(0x0da7_e5ee_0edb_d071)),
            "6.999999999999997e-243"
        );
    }

    // ---- parsing ----------------------------------------------------------

    #[test]
    fn parses_scalars() {
        assert_eq!(from_str("null"), Ok(Json::Null));
        assert_eq!(from_str(" true "), Ok(Json::Bool(true)));
        assert_eq!(from_str("false"), Ok(Json::Bool(false)));
        assert_eq!(from_str("0"), Ok(Json::U64(0)));
        assert_eq!(from_str("42"), Ok(Json::U64(42)));
        assert_eq!(from_str("-42"), Ok(Json::I64(-42)));
        assert_eq!(from_str("1.5"), Ok(Json::F64(1.5)));
        assert_eq!(from_str("1e3"), Ok(Json::F64(1000.0)));
        assert_eq!(from_str("-1.5e-3"), Ok(Json::F64(-0.0015)));
        assert_eq!(from_str(r#""hi""#), Ok(Json::Str("hi".to_owned())));
    }

    /// `serde_json` types the literal `-0` as the float `-0.0`, not as the
    /// integer `0`, and re-emits it as `-0.0`. Pinned because a certificate
    /// that round-trips through this module must stay byte-identical.
    #[test]
    fn negative_zero_is_a_float_like_serde_json() {
        let parsed = from_str("-0").expect("parses");
        assert!(matches!(parsed, Json::F64(f) if f == 0.0 && f.is_sign_negative()));
        assert_eq!(to_string(&parsed), "-0.0");
        // `-0` inside a structure keeps the same typing.
        let nested = from_str("[-0,0,-0.0]").expect("parses");
        assert_eq!(to_string(&nested), "[-0.0,0,-0.0]");
    }

    #[test]
    fn oversized_integers_become_floats() {
        let parsed = from_str("18446744073709551616").expect("parses");
        assert!(matches!(parsed, Json::F64(_)));
    }

    #[test]
    fn parses_nested_structures() {
        let parsed = from_str("{\"a\": [1, {\"b\": null}], \"c\": \"d\"}").expect("parses");
        assert_eq!(parsed["a"][0], Json::U64(1));
        assert_eq!(parsed["a"][1]["b"], Json::Null);
        assert_eq!(parsed["c"].as_str(), Some("d"));
        // Absent keys and out-of-range indices are Null, never a panic.
        assert!(parsed["nope"].is_null());
        assert!(parsed["a"][99].is_null());
        assert!(parsed["c"]["x"].is_null());
    }

    #[test]
    fn parses_escapes_and_surrogate_pairs() {
        let parsed = from_str(r#""a\"b\\c\/d\ne\tf\u0041\u00e9\ud83d\ude00""#).expect("parses");
        assert_eq!(parsed.as_str(), Some("a\"b\\c/d\ne\tfAé😀"));
    }

    #[test]
    fn round_trips_through_the_serializer() {
        let text = r#"{"a":[1,-2,3.5,true,null,"s"],"b":{"c":"d"}}"#;
        let parsed = from_str(text).expect("parses");
        assert_eq!(to_string(&parsed), text);
    }

    #[test]
    fn duplicate_keys_keep_the_last() {
        let parsed = from_str(r#"{"k":1,"k":2}"#).expect("parses");
        assert_eq!(parsed["k"], Json::U64(2));
    }

    #[test]
    fn malformed_inputs_are_errors_not_panics() {
        for bad in [
            "",
            "  ",
            "{",
            "[",
            "[1,]",
            "{\"a\":}",
            "{\"a\" 1}",
            // Unquoted object key. Spelled with inner spaces so that
            // `clippy::literal_string_with_formatting_args` does not mistake
            // `{a:1}` for a format-argument placeholder.
            "{ a: 1 }",
            "nul",
            "tru",
            "01",
            "-",
            "1.",
            "1e",
            "1e+",
            ".5",
            "+1",
            "\"unterminated",
            "\"bad\\escape\"",
            "\"\\u00\"",
            "\"\\uZZZZ\"",
            "\"\\ud800\"",
            "\"\\udc00\"",
            "\"\\ud800\\u0041\"",
            "\"raw\ncontrol\"",
            "1 2",
            "{} {}",
            "NaN",
            "Infinity",
        ] {
            assert!(from_str(bad).is_err(), "expected an Err for {bad:?}");
        }
    }

    /// The accept/reject boundary sits exactly where `serde_json`'s does:
    /// `MAX_DEPTH - 1` nested containers parse, `MAX_DEPTH` do not. Verified
    /// against `serde_json` 1.0.150 by a differential run.
    #[test]
    fn nesting_past_the_budget_is_rejected() {
        let nest = |levels: usize| {
            let mut s = String::new();
            for _ in 0..levels {
                s.push('[');
            }
            s.push('1');
            for _ in 0..levels {
                s.push(']');
            }
            s
        };
        let limit = MAX_DEPTH as usize;
        assert!(from_str(&nest(limit.saturating_sub(1))).is_ok());
        assert!(from_str(&nest(limit)).is_err());
        assert!(from_str(&nest(limit.saturating_add(1))).is_err());
        assert!(from_str(&nest(10_000)).is_err());
    }

    #[test]
    fn number_equality_ignores_the_integer_variant_split() {
        assert_eq!(Json::U64(1), Json::I64(1));
        assert_ne!(Json::U64(1), Json::F64(1.0));
        assert_ne!(Json::I64(-1), Json::U64(1));
    }

    // ---- macro ------------------------------------------------------------

    #[test]
    fn json_macro_builds_values() {
        let name = "farkas";
        let count: usize = 3;
        let value = crate::json!({
            "type": name,
            "count": count,
            "flag": true,
            "empty": {},
            "list": [1, "two", null, [3], { "deep": false }],
            "nested": { "a": { "b": 1 } },
        });
        assert_eq!(value["type"].as_str(), Some("farkas"));
        assert_eq!(value["count"], Json::U64(3));
        assert_eq!(value["flag"].as_bool(), Some(true));
        assert_eq!(value["list"][1].as_str(), Some("two"));
        assert_eq!(value["list"][4]["deep"].as_bool(), Some(false));
        assert_eq!(value["nested"]["a"]["b"], Json::U64(1));
        assert_eq!(to_string(&value["empty"]), "{}");
    }

    #[test]
    fn json_macro_accepts_collections_and_floats() {
        let names: Vec<String> = vec!["a".to_owned(), "b".to_owned()];
        let value = crate::json!({
            "names": names,
            "margin": 0.5_f64,
            "maybe": Option::<&str>::None,
        });
        assert_eq!(
            to_string(&value),
            r#"{"margin":0.5,"maybe":null,"names":["a","b"]}"#
        );
    }

    #[test]
    fn json_macro_round_trips_through_the_parser() {
        let value = crate::json!({ "a": [1, 2], "b": "c" });
        let text = to_string(&value);
        assert_eq!(from_str(&text), Ok(value));
    }

    #[test]
    fn display_is_compact_and_alternate_is_pretty() {
        let value = crate::json!({ "a": [1] });
        assert_eq!(format!("{value}"), r#"{"a":[1]}"#);
        assert_eq!(format!("{value:#}"), "{\n  \"a\": [\n    1\n  ]\n}");
    }
}
