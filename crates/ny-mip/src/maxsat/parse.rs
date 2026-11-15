// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::io::{BufRead, Read};
use std::time::Instant;

use super::{
    before_deadline, Clause, Error, Wcnf, MAX_CLAUSES, MAX_EXACT_INTEGER, MAX_LITERALS,
    MAX_VARIABLES,
};

const MAX_LINE_BYTES: usize = 8 * 1024 * 1024;
const MAX_INPUT_BYTES: usize = 2 * 1024 * 1024 * 1024;

fn unsupported(reason: &str) -> Error {
    Error::Unsupported(reason.into())
}

fn number(token: &str) -> Result<u64, Error> {
    if token.is_empty() || !token.bytes().all(|b| b.is_ascii_digit()) {
        return Err(unsupported(
            "weight/header must be an unsigned decimal integer",
        ));
    }
    token.parse().map_err(|_| unsupported("integer overflow"))
}

pub(super) fn read(mut reader: impl BufRead, deadline: Instant) -> Result<Wcnf, Error> {
    let mut clauses = Vec::new();
    let mut pending: Option<Clause> = None;
    let mut declaration = None;
    let mut top = None;
    let mut variables = 0_usize;
    let mut soft_total = 0_u64;
    let mut literals = 0_usize;
    let mut input_bytes = 0_usize;
    let mut line = Vec::new();
    loop {
        before_deadline(deadline)?;
        line.clear();
        let bytes = (&mut reader)
            .take((MAX_LINE_BYTES + 1) as u64)
            .read_until(b'\n', &mut line)?;
        if bytes == 0 {
            break;
        }
        input_bytes = input_bytes
            .checked_add(bytes)
            .ok_or_else(|| unsupported("input size overflow"))?;
        if bytes > MAX_LINE_BYTES || input_bytes > MAX_INPUT_BYTES {
            return Err(unsupported("input/line size limit"));
        }
        let text = std::str::from_utf8(&line)
            .map_err(|_| unsupported("non-UTF8 WCNF"))?
            .trim();
        if text.is_empty() || text.starts_with('c') {
            continue;
        }
        let tokens: Vec<_> = text.split_whitespace().collect();
        if tokens[0] == "p" {
            if declaration.is_some()
                || !clauses.is_empty()
                || pending.is_some()
                || !(tokens.len() == 4 || tokens.len() == 5)
                || tokens[1] != "wcnf"
            {
                return Err(unsupported("misplaced or malformed p wcnf header"));
            }
            let n = number(tokens[2])?;
            let m = number(tokens[3])?;
            if n > MAX_VARIABLES as u64 || m > MAX_CLAUSES as u64 {
                return Err(unsupported("declared dimensions exceed limits"));
            }
            variables = n as usize;
            declaration = Some((variables, m as usize));
            if tokens.len() == 5 {
                let value = number(tokens[4])?;
                if value == 0 || value > MAX_EXACT_INTEGER {
                    return Err(unsupported("top outside exact integer range"));
                }
                top = Some(value);
            }
            continue;
        }
        for (position, token) in tokens.into_iter().enumerate() {
            if position % 4096 == 0 {
                before_deadline(deadline)?;
            }
            if pending.is_none() {
                if clauses.len() >= MAX_CLAUSES {
                    return Err(unsupported("clause count limit"));
                }
                let weight = if token == "h" {
                    None
                } else {
                    let value = number(token)?;
                    if value == 0 || value > MAX_EXACT_INTEGER {
                        return Err(unsupported("weight outside exact integer range"));
                    }
                    if top.is_some_and(|bound| value >= bound) {
                        None
                    } else {
                        Some(value)
                    }
                };
                pending = Some(Clause {
                    weight,
                    literals: Vec::new(),
                });
                continue;
            }
            let literal: i32 = token
                .parse()
                .map_err(|_| unsupported("invalid signed literal"))?;
            if literal == 0 {
                let clause = pending
                    .take()
                    .ok_or_else(|| unsupported("missing clause"))?;
                if let Some(weight) = clause.weight {
                    soft_total = soft_total
                        .checked_add(weight)
                        .filter(|&v| v <= MAX_EXACT_INTEGER)
                        .ok_or_else(|| {
                            unsupported("total objective outside exact integer range")
                        })?;
                }
                clauses.push(clause);
                continue;
            }
            let variable = literal.unsigned_abs() as usize;
            if variable > MAX_VARIABLES || declaration.is_some_and(|(n, _)| variable > n) {
                return Err(unsupported("literal outside declared/supported variables"));
            }
            if declaration.is_none() {
                variables = variables.max(variable);
            }
            literals += 1;
            if literals > MAX_LITERALS {
                return Err(unsupported("literal count limit"));
            }
            pending
                .as_mut()
                .ok_or_else(|| unsupported("missing clause"))?
                .literals
                .push(literal);
        }
    }
    if pending.is_some() {
        return Err(unsupported("unterminated clause"));
    }
    if declaration.is_none() && clauses.is_empty() {
        return Err(unsupported("no WCNF header or clauses"));
    }
    if declaration.is_some_and(|(_, count)| count != clauses.len()) {
        return Err(unsupported("declared clause count differs"));
    }
    Ok(Wcnf {
        variables,
        clauses,
        soft_total,
    })
}
