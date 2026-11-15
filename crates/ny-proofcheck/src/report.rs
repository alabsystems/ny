// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The `ny-proof-check-report/v1` document and the DONE line.
//!
//! The report is deterministic: every collection is ordered, and
//! `result_digest` is the sha256 of the compact report with `timing` and
//! `result_digest` itself removed, so two runs over the same inputs agree on
//! it whatever the thread count.

use std::collections::BTreeMap;

use ny_cert::json::{self, Json, Map};

use crate::bundle::sha256_hex;

/// The report schema token.
pub const REPORT_SCHEMA: &str = "ny-proof-check-report/v1";

/// The checker's verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Verdict {
    /// Every discharge the cover needs is confirmed.
    Verified,
    /// The claims cover, but some needed discharge could not be confirmed.
    Unconfirmed,
    /// The claims do not cover, or a clause does not match the spec.
    Invalid,
    /// A rigorous counterexample exists in the input set.
    Refuted,
    /// Schema, integrity, hash, partial or unpublished bundle.
    Malformed,
    /// An op, spec form or memory need outside the checker's scope.
    Unsupported,
    /// The checker's own self-containment check tripped.
    CheckerFault,
}

impl Verdict {
    /// The process exit code (brief §6).
    #[must_use]
    pub const fn code(self) -> i32 {
        match self {
            Self::Verified => 0,
            Self::Unconfirmed => 10,
            Self::Invalid => 20,
            Self::Refuted => 21,
            Self::Malformed => 30,
            Self::Unsupported => 31,
            Self::CheckerFault => 40,
        }
    }

    /// The report token.
    #[must_use]
    pub const fn token(self) -> &'static str {
        match self {
            Self::Verified => "verified",
            Self::Unconfirmed => "unconfirmed",
            Self::Invalid => "invalid",
            Self::Refuted => "refuted",
            Self::Malformed => "malformed",
            Self::Unsupported => "unsupported",
            Self::CheckerFault => "checker_fault",
        }
    }

    /// Precedence rank: 30/31/40 > 21 > 20 > 10 > 0.
    #[must_use]
    pub const fn rank(self) -> u8 {
        match self {
            Self::Verified => 0,
            Self::Unconfirmed => 1,
            Self::Invalid => 2,
            Self::Refuted => 3,
            Self::CheckerFault => 4,
            Self::Unsupported => 5,
            Self::Malformed => 6,
        }
    }
}

/// Build a JSON object from `(key, value)` pairs.
#[must_use]
pub fn object(pairs: Vec<(&str, Json)>) -> Json {
    let mut m = Map::new();
    for (k, v) in pairs {
        m.insert(k.to_string(), v);
    }
    Json::Obj(m)
}

/// A float as JSON (non-finite → null).
#[must_use]
pub fn fnum(x: f64) -> Json {
    if x.is_finite() {
        Json::F64(x)
    } else {
        Json::Null
    }
}

/// An optional float as JSON.
#[must_use]
pub fn onum(x: Option<f64>) -> Json {
    x.map_or(Json::Null, fnum)
}

/// A decade histogram of positive values (`<1e-12`, `1e-12..1e-11`, ...,
/// `>=1e3`) plus a count of non-positive / non-finite values.
#[must_use]
pub fn decade_hist(values: &[f64]) -> Json {
    let mut bins: BTreeMap<String, u64> = BTreeMap::new();
    for &v in values {
        let key = if v.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) || !v.is_finite() {
            "nonpositive".to_string()
        } else {
            let e = v.log10().floor().clamp(-12.0, 3.0) as i32;
            format!("1e{e:+03}")
        };
        *bins.entry(key).or_insert(0) += 1;
    }
    let mut m = Map::new();
    for (k, c) in bins {
        m.insert(k, Json::U64(c));
    }
    Json::Obj(m)
}

/// The finished report: the document, its digest and its compact bytes.
pub struct Rendered {
    /// Compact JSON text (the report file's contents, without newline).
    pub text: String,
    /// `result_digest`.
    pub result_digest: String,
    /// sha256 of `text` plus its trailing newline (the file's sha256).
    pub file_sha256: String,
}

/// Insert `result_digest` into `report` (an object without `timing`) and
/// then add `timing`; return the rendered text.
#[must_use]
pub fn render(mut report: Map, timing: Json) -> Rendered {
    report.remove("result_digest");
    report.remove("timing");
    let digest = sha256_hex(json::to_string(&Json::Obj(report.clone())).as_bytes());
    report.insert("result_digest".to_string(), Json::Str(digest.clone()));
    report.insert("timing".to_string(), timing);
    let text = json::to_string(&Json::Obj(report));
    let mut with_nl = text.clone();
    with_nl.push('\n');
    let file_sha256 = sha256_hex(with_nl.as_bytes());
    Rendered {
        text,
        result_digest: digest,
        file_sha256,
    }
}

/// The DONE line. `rc=` is the REAL exit code and comes first; runners match
/// `*"PROOF_CHECK_DONE rc=0 "*`, never a bare `rc=`.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn done_line(
    verdict: Verdict,
    nodes: usize,
    confirmed: usize,
    unconfirmed: usize,
    cover: &str,
    refuted: bool,
    wall_s: f64,
    report: &str,
    report_sha256: &str,
) -> String {
    format!(
        "PROOF_CHECK_DONE rc={} verdict={} nodes={nodes} confirmed={confirmed} \
         unconfirmed={unconfirmed} cover={cover} refuted={} wall_s={wall_s:.3} report={report} \
         report_sha256={report_sha256}",
        verdict.code(),
        verdict.token(),
        u8::from(refuted),
    )
}
