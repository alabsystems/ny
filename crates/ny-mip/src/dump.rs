// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//
// MILP corpus capture: bit-exact serialization of `MilpProblem` for the
// backend differential harness (`mip-diff`) and the ay P0 baseline corpus.
//
// Set `NY_MIP_DUMP=<dir>` to write every solved IR instance to
// `<dir>/mip-<pid>-<counter>.milp`. The `.milp` text format stores every f64
// as its IEEE-754 bit pattern in hex (with a human-readable echo in a
// comment column), so a reloaded problem is the byte-identical IR — the
// harness compares backends on exactly the problem production solved.
//
// Format (line-oriented, `#`-prefixed comments ignored):
//   milp v1
//   cols <N>
//   <lb_hex> <ub_hex> <obj_hex> <0|1>      # one line per column
//   rows <M>
//   <lb_hex> <ub_hex> <k> <idx> <coef_hex> ... # one line per row, k pairs
//   margin <row_idx>                              # optional, caller-marked row
//
// Reference: docs/AY_MIP_P0.md (P0 corpus + differential harness).

use crate::error::MipError;
use crate::ir::MilpProblem;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

type Result<T> = std::result::Result<T, MipError>;

/// Serialize a problem to the `.milp` text format.
pub fn to_milp_text(problem: &MilpProblem) -> String {
    let mut s = String::with_capacity(64 * (problem.num_cols() + problem.num_rows()));
    s.push_str("milp v1\n");
    let _ = writeln!(s, "cols {}", problem.num_cols());
    for spec in problem.cols() {
        let _ = writeln!(
            s,
            "{:016x} {:016x} {:016x} {} # lb={} ub={} obj={}",
            spec.lb.to_bits(),
            spec.ub.to_bits(),
            spec.obj.to_bits(),
            u8::from(spec.integer),
            spec.lb,
            spec.ub,
            spec.obj,
        );
    }
    let _ = writeln!(s, "rows {}", problem.num_rows());
    for row in problem.rows() {
        let _ = write!(
            s,
            "{:016x} {:016x} {}",
            row.lb.to_bits(),
            row.ub.to_bits(),
            row.coeffs.len()
        );
        for &(c, w) in &row.coeffs {
            let _ = write!(s, " {c} {:016x}", w.to_bits());
        }
        let _ = writeln!(s, " # lb={} ub={}", row.lb, row.ub);
    }
    // Keep unmarked v1 dumps byte-identical.  A marked problem appends optional
    // metadata that old v1 readers already ignore after consuming `rows <M>`.
    if let Some(row) = problem.margin_row() {
        let _ = writeln!(s, "margin {}", row.0);
    }
    s
}

/// Parse the `.milp` text format back into a bit-identical problem.
pub fn from_milp_text(text: &str) -> Result<MilpProblem> {
    let bad = |msg: &str| MipError::Encoding(format!("milp parse: {msg}"));
    let mut lines = text
        .lines()
        .map(|l| l.split('#').next().unwrap_or("").trim())
        .filter(|l| !l.is_empty());

    if lines.next() != Some("milp v1") {
        return Err(bad("missing `milp v1` header"));
    }
    let ncols = parse_count(lines.next(), "cols").map_err(|m| bad(&m))?;
    let mut problem = MilpProblem::new();
    for i in 0..ncols {
        let line = lines.next().ok_or_else(|| bad("truncated cols"))?;
        let mut f = line.split_ascii_whitespace();
        let lb = parse_hex_f64(f.next()).map_err(|m| bad(&format!("col {i}: {m}")))?;
        let ub = parse_hex_f64(f.next()).map_err(|m| bad(&format!("col {i}: {m}")))?;
        let obj = parse_hex_f64(f.next()).map_err(|m| bad(&format!("col {i}: {m}")))?;
        let integer = match f.next() {
            Some("0") => false,
            Some("1") => true,
            other => return Err(bad(&format!("col {i}: bad integer flag {other:?}"))),
        };
        if integer {
            problem.add_integer_col(obj, lb, ub);
        } else {
            problem.add_col(obj, lb, ub);
        }
    }
    let nrows = parse_count(lines.next(), "rows").map_err(|m| bad(&m))?;
    for r in 0..nrows {
        let line = lines.next().ok_or_else(|| bad("truncated rows"))?;
        let mut f = line.split_ascii_whitespace();
        let lb = parse_hex_f64(f.next()).map_err(|m| bad(&format!("row {r}: {m}")))?;
        let ub = parse_hex_f64(f.next()).map_err(|m| bad(&format!("row {r}: {m}")))?;
        let k: usize = f
            .next()
            .and_then(|t| t.parse().ok())
            .ok_or_else(|| bad(&format!("row {r}: bad coeff count")))?;
        let mut coeffs = Vec::with_capacity(k);
        for j in 0..k {
            let idx: usize = f
                .next()
                .and_then(|t| t.parse().ok())
                .ok_or_else(|| bad(&format!("row {r} pair {j}: bad index")))?;
            if idx >= ncols {
                return Err(bad(&format!(
                    "row {r} pair {j}: index {idx} >= cols {ncols}"
                )));
            }
            let w = parse_hex_f64(f.next()).map_err(|m| bad(&format!("row {r} pair {j}: {m}")))?;
            coeffs.push((crate::ir::Col(idx), w));
        }
        problem.add_row(lb, ub, coeffs);
    }
    if let Some(line) = lines.next() {
        let mut f = line.split_ascii_whitespace();
        if f.next() == Some("margin") {
            let row: usize = f
                .next()
                .and_then(|t| t.parse().ok())
                .ok_or_else(|| bad("bad margin row index"))?;
            if f.next().is_some() {
                return Err(bad("unexpected fields after margin row index"));
            }
            problem
                .mark_margin_row(crate::ir::Row(row))
                .map_err(|e| bad(&format!("invalid margin marker: {e}")))?;
        }
    }
    Ok(problem)
}

fn parse_count(line: Option<&str>, keyword: &str) -> std::result::Result<usize, String> {
    let line = line.ok_or_else(|| format!("missing `{keyword} <n>` line"))?;
    let mut f = line.split_ascii_whitespace();
    if f.next() != Some(keyword) {
        return Err(format!("expected `{keyword} <n>`, got {line:?}"));
    }
    f.next()
        .and_then(|t| t.parse().ok())
        .ok_or_else(|| format!("bad count in {line:?}"))
}

fn parse_hex_f64(token: Option<&str>) -> std::result::Result<f64, String> {
    let token = token.ok_or_else(|| "missing f64 field".to_string())?;
    u64::from_str_radix(token, 16)
        .map(f64::from_bits)
        .map_err(|_| format!("bad f64 hex {token:?}"))
}

/// Whether bit-exact MILP capture was explicitly requested.
///
/// Large callers use this as a lazy guard so the default-off path does not
/// clone a model solely for diagnostics.
pub(crate) fn enabled() -> bool {
    std::env::var_os("NY_MIP_DUMP").is_some()
}

/// Write `problem` into the `NY_MIP_DUMP` directory if the env var is set.
///
/// Never fails the solve: capture problems are logged and dropped. Returns
/// the exact artifact path only after a successful write so callers and tests
/// do not have to infer ownership by scanning a process-wide dump directory.
pub(crate) fn maybe_dump(problem: &MilpProblem) -> Option<PathBuf> {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::var_os("NY_MIP_DUMP")?;
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = Path::new(&dir).join(format!("mip-{}-{n:06}.milp", std::process::id()));
    match std::fs::create_dir_all(&dir).and_then(|()| std::fs::write(&path, to_milp_text(problem)))
    {
        Ok(()) => Some(path),
        Err(e) => {
            tracing::warn!("NY_MIP_DUMP: failed to write {}: {e}", path.display());
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::MilpProblem;

    fn sample() -> MilpProblem {
        let mut p = MilpProblem::new();
        let x = p.add_col(0.0, -1.5, f64::INFINITY);
        let y = p.add_col(0.1, f64::NEG_INFINITY, 2.25);
        let z = p.add_integer_col(0.0, 0.0, 1.0);
        p.add_row(0.0, 0.0, [(x, 1.0), (y, -1.0)]);
        p.add_row(0.5, f64::INFINITY, [(x, 0.1), (z, 100.0)]);
        p
    }

    #[test]
    fn test_milp_roundtrip_is_bit_exact() {
        let p = sample();
        let text = to_milp_text(&p);
        let q = from_milp_text(&text).expect("roundtrip must parse");
        assert_eq!(p.num_cols(), q.num_cols());
        assert_eq!(p.num_rows(), q.num_rows());
        for (a, b) in p.cols().iter().zip(q.cols()) {
            assert_eq!(a.lb.to_bits(), b.lb.to_bits());
            assert_eq!(a.ub.to_bits(), b.ub.to_bits());
            assert_eq!(a.obj.to_bits(), b.obj.to_bits());
            assert_eq!(a.integer, b.integer);
        }
        for (a, b) in p.rows().iter().zip(q.rows()) {
            assert_eq!(a.lb.to_bits(), b.lb.to_bits());
            assert_eq!(a.ub.to_bits(), b.ub.to_bits());
            assert_eq!(a.coeffs.len(), b.coeffs.len());
            for ((ci, wi), (cj, wj)) in a.coeffs.iter().zip(&b.coeffs) {
                assert_eq!(ci, cj);
                assert_eq!(wi.to_bits(), wj.to_bits());
            }
        }
        assert_eq!(p.margin_row(), q.margin_row());
    }

    #[test]
    fn test_marked_margin_roundtrip_preserves_identity() {
        let mut p = sample();
        p.mark_margin_row(crate::ir::Row(1))
            .expect("sample row 1 is one-sided");
        let text = to_milp_text(&p);
        assert!(text.ends_with("margin 1\n"));
        let q = from_milp_text(&text).expect("marked roundtrip must parse");
        assert_eq!(q.margin_row(), Some(crate::ir::Row(1)));
    }

    #[test]
    fn test_milp_parse_rejects_garbage() {
        assert!(from_milp_text("not a milp file").is_err());
        assert!(
            from_milp_text("milp v1\ncols 1\n").is_err(),
            "truncated cols"
        );
    }
}
