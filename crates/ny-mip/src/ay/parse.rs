// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//
// Parse ay's SMT-LIB output: the check-sat verdict line, then the
// `(get-value ...)` response `((c0 VAL) (c1 VAL) ...)` where VAL is an
// integer/decimal numeral, `(- VAL)`, or `(/ VAL VAL)`. Values feed witness
// extraction only — the f64 conversion is best-effort by design, because
// every Sat witness is revalidated downstream by a concrete forward pass.

use crate::error::MipError;

type Result<T> = std::result::Result<T, MipError>;

/// The check-sat verdict found in solver output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Verdict {
    Sat,
    Unsat,
    Unknown,
    /// No verdict token in the output (crash, parse error upstream, ...).
    Missing,
}

/// Scan for the first standalone verdict line.
pub(super) fn parse_verdict(output: &str) -> Verdict {
    for line in output.lines() {
        match line.trim() {
            "sat" => return Verdict::Sat,
            "unsat" => return Verdict::Unsat,
            "unknown" => return Verdict::Unknown,
            _ => {}
        }
    }
    Verdict::Missing
}

/// Extract the model values for columns `c0..c{num_cols-1}` from the
/// `(get-value ...)` response. Every column must be present.
pub(super) fn parse_values(output: &str, num_cols: usize) -> Result<Vec<f64>> {
    let mut values = vec![f64::NAN; num_cols];
    let mut seen = 0usize;
    let tokens = tokenize(output);
    let mut i = 0;
    while i < tokens.len() {
        // Look for the shape: "(" "cN" VAL ")".
        if tokens[i] == "(" {
            if let Some(name) = tokens.get(i + 1) {
                if let Some(idx) = parse_col_name(name) {
                    let (value, next) = parse_value(&tokens, i + 2)?;
                    if idx < num_cols && values[idx].is_nan() {
                        values[idx] = value;
                        seen += 1;
                    }
                    // Expect the closing paren of the binding pair.
                    i = next;
                    continue;
                }
            }
        }
        i += 1;
    }
    if seen != num_cols {
        return Err(MipError::Solver(format!(
            "ay model incomplete: {seen}/{num_cols} column values parsed"
        )));
    }
    Ok(values)
}

fn parse_col_name(token: &str) -> Option<usize> {
    token.strip_prefix('c')?.parse::<usize>().ok()
}

/// Parse one VAL starting at `tokens[i]`; return (value, index after it).
fn parse_value(tokens: &[String], i: usize) -> Result<(f64, usize)> {
    let tok = tokens
        .get(i)
        .ok_or_else(|| MipError::Solver("ay model: truncated value".to_string()))?;
    if tok == "(" {
        let op = tokens
            .get(i + 1)
            .ok_or_else(|| MipError::Solver("ay model: truncated s-expr".to_string()))?;
        match op.as_str() {
            "-" => {
                let (v, next) = parse_value(tokens, i + 2)?;
                expect_close(tokens, next)?;
                Ok((-v, next + 1))
            }
            "/" => {
                let (num, after_num) = parse_value(tokens, i + 2)?;
                let (den, after_den) = parse_value(tokens, after_num)?;
                expect_close(tokens, after_den)?;
                if den == 0.0 {
                    return Err(MipError::Solver(
                        "ay model: division by zero in value".to_string(),
                    ));
                }
                Ok((num / den, after_den + 1))
            }
            other => Err(MipError::Solver(format!(
                "ay model: unsupported value operator {other:?}"
            ))),
        }
    } else {
        let v: f64 = tok
            .parse()
            .map_err(|_| MipError::Solver(format!("ay model: bad numeral {tok:?}")))?;
        Ok((v, i + 1))
    }
}

fn expect_close(tokens: &[String], i: usize) -> Result<()> {
    match tokens.get(i).map(String::as_str) {
        Some(")") => Ok(()),
        other => Err(MipError::Solver(format!(
            "ay model: expected ')', found {other:?}"
        ))),
    }
}

fn tokenize(s: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut cur = String::new();
    for ch in s.chars() {
        match ch {
            '(' | ')' => {
                if !cur.is_empty() {
                    tokens.push(std::mem::take(&mut cur));
                }
                tokens.push(ch.to_string());
            }
            c if c.is_whitespace() => {
                if !cur.is_empty() {
                    tokens.push(std::mem::take(&mut cur));
                }
            }
            c => cur.push(c),
        }
    }
    if !cur.is_empty() {
        tokens.push(cur);
    }
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_verdict_sat_unsat_unknown() {
        assert_eq!(parse_verdict("sat\n((c0 1.0))"), Verdict::Sat);
        assert_eq!(parse_verdict("unsat\n"), Verdict::Unsat);
        assert_eq!(parse_verdict("unknown\n"), Verdict::Unknown);
        assert_eq!(parse_verdict("error: boom\n"), Verdict::Missing);
    }

    #[test]
    fn test_parse_values_plain_and_rational() {
        let out = "sat\n((c0 1.0) (c1 (/ 1.0 2.0)) (c2 (- 3)) (c3 (- (/ 1 4))))";
        let v = parse_values(out, 4).expect("model must parse");
        assert_eq!(v, vec![1.0, 0.5, -3.0, -0.25]);
    }

    #[test]
    fn test_parse_values_multiline() {
        let out = "sat\n((c0 0.0)\n (c1 1.0))\n";
        let v = parse_values(out, 2).expect("model must parse");
        assert_eq!(v, vec![0.0, 1.0]);
    }

    #[test]
    fn test_parse_values_incomplete_model_is_error() {
        let out = "sat\n((c0 1.0))";
        let err = parse_values(out, 2).expect_err("missing column must be an error");
        assert!(matches!(err, MipError::Solver(_)), "{err:?}");
    }
}
