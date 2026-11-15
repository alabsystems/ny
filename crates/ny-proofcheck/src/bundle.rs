// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `ny-proof-bundle/v2`: parse, integrity and structural validation.
//!
//! # The syntactic trust split (brief §2)
//!
//! * **Structural** records define WHAT is claimed: `header`, `sites`,
//!   `clauses`, `node` (every field but `hint`), `verdict`, `published`,
//!   `end`. They are parsed strictly: an unknown kind, an unknown field, an
//!   unknown field value or an unknown literal kind is `malformed` (exit 30).
//! * **Hints** say HOW: every record whose kind starts with `hint.` and every
//!   node's `hint` object. They are untrusted and parsed leniently: an
//!   unknown hint kind is ignored and counted (`hints_ignored`); a known hint
//!   with an ill-formed value is dropped and counted (`hints_invalid`). A
//!   hint can only ever cost a confirmation.
//!
//! B2.3 adds three hint forms for the frozen-dual tier's prunes, all
//! lenient like every hint: `hint.fd_state` (the producer's identity of one
//! frozen-dual state: id, class, state sha256, sizes, root, prune count, and
//! since `#fd-compact-checkable` the count `relaxed` of relaxation symbols
//! its slope table covers), `hint.fd_lambda` (one site's relaxation slopes
//! of that state, like `hint.alpha`; for a producer's ranked compact state
//! the slopes of EVERY unstable neuron, its omitted rows' included), and a
//! node's `hint.fd` (`{"state", "l0", "l1"}`: the dual
//! point the prune was certified at, aligned with the node's literals). The
//! prune itself is an ordinary structural `node` (`close: bound`) — the claim
//! is its literals and its clause; HOW it closes is the hint.
//!
//! # v2.1: the explicit `unavailable` forms (legacy well-formedness)
//!
//! `ny-proof-bundle/v2.1` is v2 plus two forms a producer writes when it
//! CANNOT give the checker something checkable, so that a legacy lane's
//! published `unsat` is a well-formed, `unconfirmed` bundle rather than a
//! `malformed` one:
//!
//! * a header input `{"path": p, "unavailable": "<why>"}` (no `sha256`, no
//!   `bytes`): the producer could not hash one regular file at `p` (a
//!   VNN-LIB 2.0 dual-network ONNX field);
//! * a clause `{"id": k, "unavailable": "<why>"}` (no `atoms`): the producer
//!   could not render the property in the canonical form; it must be the ONE
//!   clause of the ONE `clauses` record, and stands for the whole property.
//!
//! Both are structural (strict): under a v2 header either one is an
//! `unknown_field`, and an unavailable clause in any other shape is
//! `unavailable_clause_shape`. Nothing about them is checkable, so the
//! checker reports such a bundle `unconfirmed` at best (`check::check_bundle`).
//!
//! # Integrity (brief §7 rule 2)
//!
//! The file ends with a newline; its LAST line is the `end` record, whose
//! `lines` is the file's line count and whose `sha256` is the digest of every
//! preceding byte (each line with its `\n`). The second-to-last record is
//! `published` with `result = "unsat"`. A `.partial` path, a missing or wrong
//! `end`, or any other published result is `malformed`.

use std::collections::{BTreeMap, BTreeSet};

use ny_cert::json::{self, Json};
use sha2::{Digest, Sha256};

/// The schema this checker reads.
pub const SCHEMA_V2: &str = "ny-proof-bundle/v2";
/// v2 plus the explicit `unavailable` forms (module docs).
pub const SCHEMA_V2_1: &str = "ny-proof-bundle/v2.1";
/// The node-site identity space the checker resolves literals in.
pub const SITE_SPACE: &str = "onnx-relu-input/v1";
/// The input-literal identity space.
pub const INPUT_SPACE: &str = "onnx-input0-flat/v1";

/// A header file reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileRef {
    /// Path as recorded by the producer.
    pub path: String,
    /// Lower-case hex sha256 (empty when `unavailable`).
    pub sha256: String,
    /// Byte count (0 when `unavailable`).
    pub bytes: u64,
    /// v2.1: why the producer could not hash one regular file at `path`.
    pub unavailable: Option<String>,
}

/// The header record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Header {
    /// Schema token.
    pub schema: String,
    /// Producer build line.
    pub build: String,
    /// Proof profile token.
    pub profile: String,
    /// Profile source token.
    pub profile_source: String,
    /// Producing lane.
    pub lane: String,
    /// The ONNX file.
    pub onnx: FileRef,
    /// The VNN-LIB file.
    pub vnnlib: FileRef,
}

/// One entry of the `sites` record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SiteTensor {
    /// Bundle-local id (literals name it).
    pub id: u64,
    /// ONNX name of a Relu node's `input[0]`.
    pub name: String,
    /// Flat size.
    pub size: u64,
    /// Producer op index (diagnostic).
    pub op: u64,
}

/// One producer clause.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Clause {
    /// Producer clause id.
    pub id: u64,
    /// Atom strings (canonical form expected; empty when `unavailable`).
    pub atoms: Vec<String>,
    /// v2.1: why the producer could not render the property (the clause
    /// stands for the whole property).
    pub unavailable: Option<String>,
}

/// A literal.
#[derive(Debug, Clone, PartialEq)]
pub enum Lit {
    /// `z_{tensor,index} >= 0` (`pos`) or `<= 0`.
    Relu {
        /// Site tensor id (bundle-local).
        tensor: u64,
        /// Flat index.
        index: u64,
        /// Side.
        pos: bool,
    },
    /// `x_i >= t` (`pos`) or `x_i <= t`.
    X {
        /// Input index.
        index: u64,
        /// Threshold (exact, parsed from the decimal string).
        t: f64,
        /// Side.
        pos: bool,
    },
}

/// How a node closes (diagnostic except `Empty`, `Subtree`, `Opaque`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Close {
    /// A certified bound.
    Bound,
    /// A certified empty region (claims every clause).
    Empty,
    /// A column retirement at an internal node.
    Retire,
    /// A root class discharge.
    Root,
    /// A frozen-dual class discharge (needs `fd-check`, B2.2).
    Fd,
    /// A nested frame's leaf (its cell is covered by the nested nodes).
    Subtree,
    /// A route that streams no checkable certificate.
    Opaque,
}

impl Close {
    fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "bound" => Self::Bound,
            "empty" => Self::Empty,
            "retire" => Self::Retire,
            "root" => Self::Root,
            "fd" => Self::Fd,
            "subtree" => Self::Subtree,
            "opaque" => Self::Opaque,
            _ => return None,
        })
    }

    /// Token.
    #[must_use]
    pub const fn token(self) -> &'static str {
        match self {
            Self::Bound => "bound",
            Self::Empty => "empty",
            Self::Retire => "retire",
            Self::Root => "root",
            Self::Fd => "fd",
            Self::Subtree => "subtree",
            Self::Opaque => "opaque",
        }
    }
}

/// The clauses a node claims.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Dis {
    /// Every clause (`"*"`, emptiness nodes).
    All,
    /// These producer clause ids.
    Ids(Vec<u64>),
}

/// B2.3: a node's frozen-dual hint (`hint.fd`): the FD state it was pruned
/// in and the dual point `(L0, L1)` it was certified at, aligned with the
/// node's literals. Untrusted: any `L >= 0` is a valid dual point, so a wrong
/// one can only fail to confirm.
#[derive(Debug, Clone, PartialEq)]
pub struct FdNodeHint {
    /// The `hint.fd_state` id.
    pub state: u64,
    /// Sign-cut multipliers (NaN where the producer wrote `null`).
    pub l0: Vec<f64>,
    /// Far-cut multipliers.
    pub l1: Vec<f64>,
}

/// A node's (untrusted) hint.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct NodeHint {
    /// β aligned with `lits` (already sanitized: NaN → 0, negative → 0).
    pub beta: Option<Vec<f64>>,
    /// α table id.
    pub alpha: Option<u64>,
    /// The producer's bound came from an ancestor region.
    pub lift: bool,
    /// The producer's certified margin.
    pub margin: Option<f64>,
    /// A site the producer found empty `(tensor id, index)`.
    pub empty_site: Option<(u64, u64)>,
    /// B2.3: the frozen-dual prune hint.
    pub fd: Option<FdNodeHint>,
}

/// A node record.
#[derive(Debug, Clone, PartialEq)]
pub struct NodeRec {
    /// Id.
    pub id: u64,
    /// Region.
    pub region: u64,
    /// Literals in branch order.
    pub lits: Vec<Lit>,
    /// Close kind.
    pub close: Close,
    /// Source (diagnostic).
    pub src: String,
    /// Claimed clauses.
    pub dis: Dis,
    /// Hint.
    pub hint: NodeHint,
}

/// A `hint.alpha` record.
#[derive(Debug, Clone, PartialEq)]
pub struct AlphaHint {
    /// Table id.
    pub id: u64,
    /// Site tensor id.
    pub tensor: u64,
    /// Neurons.
    pub idx: Vec<u64>,
    /// Slopes (sanitized into `[0, 1]`, NaN dropped).
    pub val: Vec<f64>,
}

/// B2.3: a `hint.fd_state` record — the producer's identity of one
/// frozen-dual state. Diagnostic only: the checker rebuilds its own state.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FdStateHint {
    /// State id (the nodes' `hint.fd.state`).
    pub id: u64,
    /// The producer clause id the state's margin discharges.
    pub class: Option<u64>,
    /// The producer's sha256 of its certified state.
    pub state_sha256: Option<String>,
    /// The producer's generator count.
    pub n_gens: Option<u64>,
    /// The producer's unstable rows.
    pub unstable: Option<u64>,
    /// `#fd-compact-checkable`: the relaxation symbols the producer's slope
    /// table covers (diagnostic; the checker counts its own coverage).
    pub relaxed: Option<u64>,
    /// The producer's certified state root.
    pub root: Option<f64>,
    /// Prunes the producer streamed.
    pub prunes: Option<u64>,
}

/// B2.3: a `hint.fd_lambda` record — one site's relaxation slopes of one
/// frozen-dual state (sanitized into `[0, 1]`, NaN dropped).
#[derive(Debug, Clone, PartialEq)]
pub struct FdLambdaHint {
    /// State id.
    pub id: u64,
    /// Site tensor id.
    pub tensor: u64,
    /// Neurons.
    pub idx: Vec<u64>,
    /// Slopes.
    pub val: Vec<f64>,
}

/// The verdict record.
#[derive(Debug, Clone, PartialEq)]
pub struct VerdictRec {
    /// Token source.
    pub source: String,
    /// Clause ids the token covers.
    pub required: Vec<u64>,
    /// Producer minimum margin (diagnostic).
    pub min_margin: Option<f64>,
    /// Nodes streamed.
    pub nodes: u64,
}

/// A parsed, integrity-checked bundle.
#[derive(Debug, Clone, PartialEq)]
pub struct Bundle {
    /// Header.
    pub header: Header,
    /// Sites.
    pub sites: Vec<SiteTensor>,
    /// Clauses per region.
    pub clauses: BTreeMap<u64, Vec<Clause>>,
    /// Nodes.
    pub nodes: Vec<NodeRec>,
    /// α hint records.
    pub alphas: Vec<AlphaHint>,
    /// B2.3: frozen-dual state hints.
    pub fd_states: Vec<FdStateHint>,
    /// B2.3: frozen-dual slope hints.
    pub fd_lambdas: Vec<FdLambdaHint>,
    /// Verdict.
    pub verdict: VerdictRec,
    /// Line count.
    pub lines: usize,
    /// Unknown hint records.
    pub hints_ignored: usize,
    /// Ill-formed hints dropped.
    pub hints_invalid: usize,
}

impl Bundle {
    /// v2.1: why nothing in this bundle is checkable — the property's
    /// clauses, or a header input, are `unavailable` — as `(code, detail)`
    /// reasons, or empty for a checkable bundle.
    #[must_use]
    pub fn unavailable(&self) -> Vec<(&'static str, String)> {
        let mut out = Vec::new();
        for (what, f) in [("onnx", &self.header.onnx), ("vnnlib", &self.header.vnnlib)] {
            if let Some(why) = &f.unavailable {
                out.push(("inputs_unavailable", format!("{what} `{}`: {why}", f.path)));
            }
        }
        for c in self.clauses.values().flatten() {
            if let Some(why) = &c.unavailable {
                out.push(("clauses_unavailable", why.clone()));
            }
        }
        out
    }
}

/// A malformed-bundle reason (report code + detail).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Malformed {
    /// Short code (`v1_positional_sites`, `truncated`, ...).
    pub code: String,
    /// Detail.
    pub detail: String,
}

fn bad<T>(code: &str, detail: impl Into<String>) -> Result<T, Malformed> {
    Err(Malformed {
        code: code.to_string(),
        detail: detail.into(),
    })
}

fn obj(v: &Json, line: usize) -> Result<&json::Map, Malformed> {
    match v {
        Json::Obj(m) => Ok(m),
        _ => bad("not_an_object", format!("line {line}")),
    }
}

fn only_keys(m: &json::Map, allowed: &[&str], what: &str, line: usize) -> Result<(), Malformed> {
    for k in m.keys() {
        if !allowed.contains(&k.as_str()) {
            return bad(
                "unknown_field",
                format!("line {line}: `{what}` has unknown field `{k}`"),
            );
        }
    }
    Ok(())
}

fn req<'a>(m: &'a json::Map, key: &str, line: usize) -> Result<&'a Json, Malformed> {
    m.get(key).ok_or_else(|| Malformed {
        code: "missing_field".into(),
        detail: format!("line {line}: missing `{key}`"),
    })
}

fn req_str(m: &json::Map, key: &str, line: usize) -> Result<String, Malformed> {
    req(m, key, line)?
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| Malformed {
            code: "bad_field".into(),
            detail: format!("line {line}: `{key}` is not a string"),
        })
}

fn req_u64(m: &json::Map, key: &str, line: usize) -> Result<u64, Malformed> {
    req(m, key, line)?.as_u64().ok_or_else(|| Malformed {
        code: "bad_field".into(),
        detail: format!("line {line}: `{key}` is not a non-negative integer"),
    })
}

fn num(v: &Json) -> Option<f64> {
    match v {
        Json::U64(u) => Some(*u as f64),
        Json::I64(i) => Some(*i as f64),
        Json::F64(f) => Some(*f),
        _ => None,
    }
}

fn file_ref(v: &Json, what: &str, line: usize, v2_1: bool) -> Result<FileRef, Malformed> {
    let m = obj(v, line)?;
    if v2_1 && m.contains_key("unavailable") {
        only_keys(m, &["path", "unavailable"], what, line)?;
        return Ok(FileRef {
            path: req_str(m, "path", line)?,
            sha256: String::new(),
            bytes: 0,
            unavailable: Some(req_str(m, "unavailable", line)?),
        });
    }
    only_keys(m, &["path", "sha256", "bytes"], what, line)?;
    let sha = req_str(m, "sha256", line)?;
    if sha.len() != 64 || !sha.bytes().all(|b| b.is_ascii_hexdigit()) {
        return bad(
            "bad_field",
            format!("line {line}: {what}.sha256 is not hex sha256"),
        );
    }
    Ok(FileRef {
        path: req_str(m, "path", line)?,
        sha256: sha.to_ascii_lowercase(),
        bytes: req_u64(m, "bytes", line)?,
        unavailable: None,
    })
}

fn parse_lit(v: &Json, line: usize) -> Result<Lit, Malformed> {
    let Json::Arr(a) = v else {
        return bad(
            "unknown_literal_kind",
            format!("line {line}: literal is not an array"),
        );
    };
    let side = |x: &Json| -> Result<bool, Malformed> {
        match x.as_u64() {
            Some(1) => Ok(true),
            Some(0) => Ok(false),
            _ => bad("bad_literal", format!("line {line}: side must be 0 or 1")),
        }
    };
    match a.as_slice() {
        [Json::Str(tag), i, t, s] if tag == "x" => {
            let index = i.as_u64().ok_or_else(|| Malformed {
                code: "bad_literal".into(),
                detail: format!("line {line}: input index"),
            })?;
            let Json::Str(text) = t else {
                return bad(
                    "bad_literal",
                    format!("line {line}: threshold must be a decimal string"),
                );
            };
            let tv: f64 = text.parse().map_err(|_| Malformed {
                code: "bad_literal".into(),
                detail: format!("line {line}: threshold `{text}`"),
            })?;
            if !tv.is_finite() {
                return bad("bad_literal", format!("line {line}: non-finite threshold"));
            }
            Ok(Lit::X {
                index,
                t: tv,
                pos: side(s)?,
            })
        }
        [t, i, s] if t.as_u64().is_some() => Ok(Lit::Relu {
            tensor: t.as_u64().unwrap_or(0),
            index: i.as_u64().ok_or_else(|| Malformed {
                code: "bad_literal".into(),
                detail: format!("line {line}: neuron index"),
            })?,
            pos: side(s)?,
        }),
        _ => bad(
            "unknown_literal_kind",
            format!("line {line}: literal shape"),
        ),
    }
}

fn parse_hint(v: Option<&Json>, invalid: &mut usize) -> NodeHint {
    let mut h = NodeHint::default();
    let Some(v) = v else {
        return h;
    };
    let Json::Obj(m) = v else {
        *invalid += 1;
        return h;
    };
    let mut h_fd: Option<FdNodeHint> = None;
    if let Some(b) = m.get("beta") {
        match b {
            Json::Arr(xs) => {
                let mut out = Vec::with_capacity(xs.len());
                let mut bad_values = false;
                for x in xs {
                    match num(x) {
                        Some(f) if f.is_finite() && f >= 0.0 => out.push(f),
                        Some(f) if f.is_finite() => {
                            bad_values = true;
                            out.push(0.0);
                        }
                        _ => {
                            bad_values = true;
                            out.push(0.0);
                        }
                    }
                }
                if bad_values {
                    *invalid += 1;
                }
                h.beta = Some(out);
            }
            Json::Null => {}
            _ => *invalid += 1,
        }
    }
    if let Some(a) = m.get("alpha") {
        match a {
            Json::Null => {}
            other => match other.as_u64() {
                Some(id) => h.alpha = Some(id),
                None => *invalid += 1,
            },
        }
    }
    if let Some(l) = m.get("lift") {
        match l {
            Json::Bool(b) => h.lift = *b,
            _ => *invalid += 1,
        }
    }
    if let Some(mg) = m.get("margin") {
        h.margin = num(mg);
    }
    if let Some(fd) = m.get("fd") {
        match fd {
            Json::Null => {}
            other => match parse_fd_hint(other) {
                Some(h) => h_fd = Some(h),
                None => *invalid += 1,
            },
        }
    }
    if let Some(e) = m.get("empty_site") {
        match e {
            Json::Null => {}
            Json::Arr(p) => match (
                p.first().and_then(Json::as_u64),
                p.get(1).and_then(Json::as_u64),
            ) {
                (Some(t), Some(i)) if p.len() == 2 => h.empty_site = Some((t, i)),
                _ => *invalid += 1,
            },
            _ => *invalid += 1,
        }
    }
    h.fd = h_fd;
    h
}

/// A multiplier list: numbers (any sign: the checker clamps) or `null`
/// (NaN). Anything else is ill-formed.
fn fd_multipliers(v: Option<&Json>) -> Option<Vec<f64>> {
    let Some(Json::Arr(xs)) = v else {
        return None;
    };
    xs.iter()
        .map(|x| match x {
            Json::Null => Some(f64::NAN),
            other => num(other),
        })
        .collect()
}

/// `hint.fd`: `{"state": id, "l0": [...], "l1": [...]}` with equal lengths.
fn parse_fd_hint(v: &Json) -> Option<FdNodeHint> {
    let Json::Obj(m) = v else {
        return None;
    };
    let state = m.get("state").and_then(Json::as_u64)?;
    let l0 = fd_multipliers(m.get("l0"))?;
    let l1 = fd_multipliers(m.get("l1"))?;
    (l0.len() == l1.len()).then_some(FdNodeHint { state, l0, l1 })
}

/// `hint.fd_state` (lenient: only the id is required).
fn parse_fd_state(m: &json::Map, invalid: &mut usize) -> Option<FdStateHint> {
    let Some(id) = m.get("id").and_then(Json::as_u64) else {
        *invalid += 1;
        return None;
    };
    Some(FdStateHint {
        id,
        class: m.get("class").and_then(Json::as_u64),
        state_sha256: m
            .get("state_sha256")
            .and_then(Json::as_str)
            .map(str::to_string),
        n_gens: m.get("n_gens").and_then(Json::as_u64),
        unstable: m.get("unstable").and_then(Json::as_u64),
        relaxed: m.get("relaxed").and_then(Json::as_u64),
        root: m.get("root").and_then(num),
        prunes: m.get("prunes").and_then(Json::as_u64),
    })
}

/// `hint.fd_lambda`: like `hint.alpha` (slopes sanitized into `[0, 1]`).
fn parse_fd_lambda(m: &json::Map, invalid: &mut usize) -> Option<FdLambdaHint> {
    let id = m.get("id").and_then(Json::as_u64);
    let t = m.get("t").and_then(Json::as_u64);
    let (Some(id), Some(tensor)) = (id, t) else {
        *invalid += 1;
        return None;
    };
    let (Some(Json::Arr(ix)), Some(Json::Arr(vx))) = (m.get("idx"), m.get("val")) else {
        *invalid += 1;
        return None;
    };
    if ix.len() != vx.len() {
        *invalid += 1;
        return None;
    }
    let mut idx = Vec::with_capacity(ix.len());
    let mut val = Vec::with_capacity(ix.len());
    let mut dropped = false;
    for (i, v) in ix.iter().zip(vx) {
        match (i.as_u64(), num(v)) {
            (Some(i), Some(v)) if !v.is_nan() => {
                if !(0.0..=1.0).contains(&v) {
                    dropped = true;
                }
                idx.push(i);
                val.push(v.clamp(0.0, 1.0));
            }
            _ => dropped = true,
        }
    }
    if dropped {
        *invalid += 1;
    }
    Some(FdLambdaHint {
        id,
        tensor,
        idx,
        val,
    })
}

fn parse_alpha(m: &json::Map, invalid: &mut usize) -> Option<AlphaHint> {
    let id = m.get("id").and_then(Json::as_u64);
    let t = m.get("t").and_then(Json::as_u64);
    let start_ok = matches!(m.get("start"), None | Some(Json::Null))
        || m.get("start").and_then(Json::as_str) == Some("*");
    let (Some(id), Some(tensor), true) = (id, t, start_ok) else {
        *invalid += 1;
        return None;
    };
    let (Some(Json::Arr(ix)), Some(Json::Arr(vx))) = (m.get("idx"), m.get("val")) else {
        *invalid += 1;
        return None;
    };
    if ix.len() != vx.len() {
        *invalid += 1;
        return None;
    }
    let mut idx = Vec::with_capacity(ix.len());
    let mut val = Vec::with_capacity(ix.len());
    let mut dropped = false;
    for (i, v) in ix.iter().zip(vx) {
        match (i.as_u64(), num(v)) {
            (Some(i), Some(v)) if !v.is_nan() => {
                if !(0.0..=1.0).contains(&v) {
                    dropped = true;
                }
                idx.push(i);
                val.push(v.clamp(0.0, 1.0));
            }
            _ => dropped = true,
        }
    }
    if dropped {
        *invalid += 1;
    }
    Some(AlphaHint {
        id,
        tensor,
        idx,
        val,
    })
}

/// Lower-case hex sha256 of `bytes`.
#[must_use]
pub fn sha256_hex(bytes: &[u8]) -> String {
    let d = Sha256::digest(bytes);
    let mut s = String::with_capacity(64);
    for b in d {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// Parse and integrity-check a bundle's bytes.
#[allow(clippy::too_many_lines)]
pub fn parse_bundle(bytes: &[u8]) -> Result<Bundle, Malformed> {
    if bytes.is_empty() {
        return bad("empty", "empty bundle");
    }
    if bytes.last() != Some(&b'\n') {
        return bad(
            "truncated",
            "the bundle does not end with a newline (a cut stream)",
        );
    }
    let body = &bytes[..bytes.len() - 1];
    let line_slices: Vec<&[u8]> = body.split(|&b| b == b'\n').collect();
    let n_lines = line_slices.len();
    if n_lines < 2 {
        return bad("truncated", "fewer than two lines");
    }
    let mut lines: Vec<Json> = Vec::with_capacity(n_lines);
    for (i, raw) in line_slices.iter().enumerate() {
        let text = std::str::from_utf8(raw).map_err(|_| Malformed {
            code: "not_utf8".into(),
            detail: format!("line {}", i + 1),
        })?;
        let v = json::from_str(text).map_err(|e| Malformed {
            code: "parse".into(),
            detail: format!("line {}: {e}", i + 1),
        })?;
        lines.push(v);
    }
    // Integrity first: `end` is the last line.
    let last = obj(&lines[n_lines - 1], n_lines)?;
    if last.get("kind").and_then(Json::as_str) != Some("end") {
        return bad("truncated", "the last record is not `end`");
    }
    only_keys(last, &["kind", "lines", "sha256"], "end", n_lines)?;
    let want_lines = req_u64(last, "lines", n_lines)?;
    if want_lines != n_lines as u64 {
        return bad(
            "integrity",
            format!("`end.lines` = {want_lines} but the file has {n_lines} lines"),
        );
    }
    let end_start = bytes.len() - 1 - line_slices[n_lines - 1].len();
    let digest = sha256_hex(&bytes[..end_start]);
    if req_str(last, "sha256", n_lines)?.to_ascii_lowercase() != digest {
        return bad(
            "integrity",
            "`end.sha256` does not match the preceding bytes",
        );
    }
    // Header.
    let h = obj(&lines[0], 1)?;
    if h.get("kind").and_then(Json::as_str) != Some("header") {
        return bad("no_header", "the first record is not `header`");
    }
    let schema = req_str(h, "schema", 1)?;
    let v2_1 = schema == SCHEMA_V2_1;
    if schema != SCHEMA_V2 && !v2_1 {
        if schema == "ny-proof-bundle/v1" {
            return bad(
                "v1_positional_sites",
                "a v1 bundle carries frame-relative positional sites the checker cannot resolve",
            );
        }
        return bad("unknown_schema", format!("schema `{schema}`"));
    }
    only_keys(
        h,
        &[
            "kind",
            "schema",
            "build",
            "profile",
            "profile_source",
            "lane",
            "site_space",
            "input_space",
            "onnx",
            "vnnlib",
        ],
        "header",
        1,
    )?;
    if req_str(h, "site_space", 1)? != SITE_SPACE {
        return bad("unknown_site_space", "site_space");
    }
    if req_str(h, "input_space", 1)? != INPUT_SPACE {
        return bad("unknown_input_space", "input_space");
    }
    let header = Header {
        schema,
        build: req_str(h, "build", 1)?,
        profile: req_str(h, "profile", 1)?,
        profile_source: req_str(h, "profile_source", 1)?,
        lane: req_str(h, "lane", 1)?,
        onnx: file_ref(req(h, "onnx", 1)?, "onnx", 1, v2_1)?,
        vnnlib: file_ref(req(h, "vnnlib", 1)?, "vnnlib", 1, v2_1)?,
    };
    // `published` is second to last and says unsat.
    let p = obj(&lines[n_lines - 2], n_lines - 1)?;
    if p.get("kind").and_then(Json::as_str) != Some("published") {
        return bad("unpublished", "the record before `end` is not `published`");
    }
    only_keys(p, &["kind", "result"], "published", n_lines - 1)?;
    let result = req_str(p, "result", n_lines - 1)?;
    if result != "unsat" {
        return bad(
            "unpublished",
            format!("published result is `{result}`, not unsat"),
        );
    }
    let mut sites: Option<Vec<SiteTensor>> = None;
    let mut clauses: BTreeMap<u64, Vec<Clause>> = BTreeMap::new();
    let mut nodes: Vec<NodeRec> = Vec::new();
    let mut alphas: Vec<AlphaHint> = Vec::new();
    let mut fd_states: Vec<FdStateHint> = Vec::new();
    let mut fd_lambdas: Vec<FdLambdaHint> = Vec::new();
    let mut verdict: Option<VerdictRec> = None;
    let mut hints_ignored = 0usize;
    let mut hints_invalid = 0usize;
    let mut node_ids: BTreeSet<u64> = BTreeSet::new();
    for (i, v) in lines.iter().enumerate().take(n_lines - 2).skip(1) {
        let line = i + 1;
        let m = obj(v, line)?;
        let kind = req_str(m, "kind", line)?;
        if kind.starts_with("hint.") {
            match kind.as_str() {
                "hint.alpha" => {
                    if let Some(a) = parse_alpha(m, &mut hints_invalid) {
                        alphas.push(a);
                    }
                }
                // B2.3: the frozen-dual tier's state identity and slopes.
                "hint.fd_state" => {
                    if let Some(f) = parse_fd_state(m, &mut hints_invalid) {
                        fd_states.push(f);
                    }
                }
                "hint.fd_lambda" => {
                    if let Some(f) = parse_fd_lambda(m, &mut hints_invalid) {
                        fd_lambdas.push(f);
                    }
                }
                _ => hints_ignored += 1,
            }
            continue;
        }
        match kind.as_str() {
            "header" => return bad("duplicate_header", format!("line {line}")),
            "sites" => {
                if sites.is_some() {
                    return bad("duplicate_sites", format!("line {line}"));
                }
                only_keys(m, &["kind", "tensors"], "sites", line)?;
                let Json::Arr(ts) = req(m, "tensors", line)? else {
                    return bad("bad_field", format!("line {line}: tensors"));
                };
                let mut out = Vec::with_capacity(ts.len());
                let mut ids = BTreeSet::new();
                for t in ts {
                    let tm = obj(t, line)?;
                    only_keys(tm, &["id", "name", "size", "op"], "sites.tensors", line)?;
                    let st = SiteTensor {
                        id: req_u64(tm, "id", line)?,
                        name: req_str(tm, "name", line)?,
                        size: req_u64(tm, "size", line)?,
                        op: req_u64(tm, "op", line)?,
                    };
                    if !ids.insert(st.id) {
                        return bad("duplicate_site_id", format!("line {line}: id {}", st.id));
                    }
                    out.push(st);
                }
                sites = Some(out);
            }
            "clauses" => {
                only_keys(m, &["kind", "region", "list"], "clauses", line)?;
                let region = req_u64(m, "region", line)?;
                if clauses.contains_key(&region) {
                    return bad("duplicate_clauses", format!("line {line}: region {region}"));
                }
                let Json::Arr(list) = req(m, "list", line)? else {
                    return bad("bad_field", format!("line {line}: list"));
                };
                let mut out = Vec::with_capacity(list.len());
                let mut ids = BTreeSet::new();
                for c in list {
                    let cm = obj(c, line)?;
                    if v2_1 && cm.contains_key("unavailable") {
                        only_keys(cm, &["id", "unavailable"], "clauses.list", line)?;
                        let id = req_u64(cm, "id", line)?;
                        if !ids.insert(id) {
                            return bad("duplicate_clause_id", format!("line {line}: id {id}"));
                        }
                        out.push(Clause {
                            id,
                            atoms: Vec::new(),
                            unavailable: Some(req_str(cm, "unavailable", line)?),
                        });
                        continue;
                    }
                    only_keys(cm, &["id", "atoms"], "clauses.list", line)?;
                    let id = req_u64(cm, "id", line)?;
                    if !ids.insert(id) {
                        return bad("duplicate_clause_id", format!("line {line}: id {id}"));
                    }
                    let Json::Arr(atoms) = req(cm, "atoms", line)? else {
                        return bad("bad_field", format!("line {line}: atoms"));
                    };
                    let mut a = Vec::with_capacity(atoms.len());
                    for x in atoms {
                        a.push(
                            x.as_str()
                                .ok_or_else(|| Malformed {
                                    code: "bad_field".into(),
                                    detail: format!("line {line}: atom is not a string"),
                                })?
                                .to_string(),
                        );
                    }
                    if a.is_empty() {
                        return bad(
                            "bad_field",
                            format!("line {line}: clause {id} has no atoms"),
                        );
                    }
                    out.push(Clause {
                        id,
                        atoms: a,
                        unavailable: None,
                    });
                }
                clauses.insert(region, out);
            }
            "node" => {
                only_keys(
                    m,
                    &[
                        "kind", "id", "region", "lits", "close", "src", "dis", "hint",
                    ],
                    "node",
                    line,
                )?;
                let id = req_u64(m, "id", line)?;
                if !node_ids.insert(id) {
                    return bad("duplicate_node_id", format!("line {line}: id {id}"));
                }
                let close_s = req_str(m, "close", line)?;
                let close = Close::parse(&close_s).ok_or_else(|| Malformed {
                    code: "unknown_close".into(),
                    detail: format!("line {line}: close `{close_s}`"),
                })?;
                let Json::Arr(ls) = req(m, "lits", line)? else {
                    return bad("bad_field", format!("line {line}: lits"));
                };
                let lits = ls
                    .iter()
                    .map(|l| parse_lit(l, line))
                    .collect::<Result<Vec<_>, _>>()?;
                let dis = match req(m, "dis", line)? {
                    Json::Str(s) if s == "*" => Dis::All,
                    Json::Arr(xs) => Dis::Ids(
                        xs.iter()
                            .map(|x| {
                                x.as_u64().ok_or_else(|| Malformed {
                                    code: "bad_field".into(),
                                    detail: format!("line {line}: dis entry"),
                                })
                            })
                            .collect::<Result<Vec<_>, _>>()?,
                    ),
                    _ => return bad("bad_field", format!("line {line}: dis")),
                };
                match (close, &dis) {
                    (Close::Empty, Dis::All) => {}
                    (Close::Empty, _) | (_, Dis::All) => {
                        return bad(
                            "bad_field",
                            format!("line {line}: dis \"*\" is exactly the emptiness claim"),
                        )
                    }
                    (Close::Subtree, Dis::Ids(v)) if !v.is_empty() => {
                        return bad(
                            "bad_field",
                            format!("line {line}: a subtree node claims nothing"),
                        )
                    }
                    _ => {}
                }
                let hint = parse_hint(m.get("hint"), &mut hints_invalid);
                nodes.push(NodeRec {
                    id,
                    region: req_u64(m, "region", line)?,
                    lits,
                    close,
                    src: req_str(m, "src", line)?,
                    dis,
                    hint,
                });
            }
            "verdict" => {
                if verdict.is_some() {
                    return bad("duplicate_verdict", format!("line {line}"));
                }
                only_keys(
                    m,
                    &["kind", "source", "required", "min_margin", "nodes"],
                    "verdict",
                    line,
                )?;
                let Json::Arr(rq) = req(m, "required", line)? else {
                    return bad("bad_field", format!("line {line}: required"));
                };
                let required = rq
                    .iter()
                    .map(|x| {
                        x.as_u64().ok_or_else(|| Malformed {
                            code: "bad_field".into(),
                            detail: format!("line {line}: required entry"),
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                verdict = Some(VerdictRec {
                    source: req_str(m, "source", line)?,
                    required,
                    min_margin: m.get("min_margin").and_then(num),
                    nodes: req_u64(m, "nodes", line)?,
                });
            }
            "published" => return bad("duplicate_published", format!("line {line}")),
            "end" => return bad("misplaced_end", format!("line {line}")),
            other => {
                return bad(
                    "unknown_record_kind",
                    format!("line {line}: structural kind `{other}`"),
                )
            }
        }
    }
    let sites = sites.ok_or_else(|| Malformed {
        code: "missing_sites".into(),
        detail: "no `sites` record".into(),
    })?;
    let verdict = verdict.ok_or_else(|| Malformed {
        code: "missing_verdict".into(),
        detail: "no `verdict` record".into(),
    })?;
    if clauses.is_empty() {
        return bad("missing_clauses", "no `clauses` record");
    }
    // v2.1: an unavailable clause stands for the WHOLE property, so it is the
    // one clause of the one `clauses` record.
    let all: Vec<&Clause> = clauses.values().flatten().collect();
    if all.iter().any(|c| c.unavailable.is_some()) && (clauses.len() != 1 || all.len() != 1) {
        return bad(
            "unavailable_clause_shape",
            "an unavailable clause must be the only clause of the only `clauses` record",
        );
    }
    Ok(Bundle {
        header,
        sites,
        clauses,
        nodes,
        alphas,
        fd_states,
        fd_lambdas,
        verdict,
        lines: n_lines,
        hints_ignored,
        hints_invalid,
    })
}
