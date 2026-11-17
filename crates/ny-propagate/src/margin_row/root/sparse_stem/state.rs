// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! One owner per live root tensor, regardless of its physical row layout.

use super::*;

impl<T> State<T> {
    pub(in crate::margin_row::root) fn dense(
        m: Array2<T>,
        d: Array2<T>,
        gerr: Vec<f64>,
        support: Vec<u64>,
    ) -> Self {
        Self {
            rows: ReluRows::Dense(m, d),
            gerr,
            support,
        }
    }

    pub(in crate::margin_row::root) fn packed(
        rows: PackedRows<T>,
        gerr: Vec<f64>,
        support: Vec<u64>,
    ) -> Self {
        Self {
            rows: ReluRows::Packed(rows),
            gerr,
            support,
        }
    }
}

fn live<T>(state: &[Option<State<T>>], consumers: &[usize], id: usize) -> Result<()> {
    if consumers.get(id).copied().unwrap_or(0) == 0 || state.get(id).is_none_or(Option::is_none) {
        return Err(NyError::InvalidSpec(
            "margin_row: dead tensor or exhausted use count".into(),
        ));
    }
    Ok(())
}

pub(in crate::margin_row::root) fn borrow<'a, T>(
    state: &'a [Option<State<T>>],
    consumers: &[usize],
    id: usize,
) -> Result<&'a State<T>> {
    live(state, consumers, id)?;
    Ok(state[id].as_ref().expect("live tensor"))
}

/// Release a borrowed use only after every source-dependent result exists.
/// Add(x,x) calls this twice; the first call cannot drop the second borrow.
pub(in crate::margin_row::root) fn release<T>(
    state: &mut [Option<State<T>>],
    consumers: &mut [usize],
    id: usize,
) -> Result<()> {
    live(state, consumers, id)?;
    consumers[id] -= 1;
    if consumers[id] == 0 {
        state[id] = None;
    }
    Ok(())
}

/// ReLU/Flatten need exclusive rows. Move the final use and clone a shared
/// input. The caller may then mutate its owned copy without touching peers.
pub(in crate::margin_row::root) fn take_rows<T: TabElem>(
    state: &mut [Option<State<T>>],
    consumers: &mut [usize],
    id: usize,
) -> Result<State<T>> {
    live(state, consumers, id)?;
    let out = if consumers[id] == 1 {
        state[id].take().expect("live tensor")
    } else {
        state[id].as_ref().expect("live tensor").clone()
    };
    consumers[id] -= 1;
    Ok(out)
}

/// Materialize directly from a packed borrow: never clone the compact payload
/// just to expand it. The source remains available if expansion hits deadline.
pub(in crate::margin_row::root) fn take_dense<T: TabElem>(
    state: &mut [Option<State<T>>],
    consumers: &mut [usize],
    id: usize,
    deadline: Option<Instant>,
) -> Result<(Array2<T>, Array2<T>, Vec<f64>, Vec<u64>)> {
    check_deadline(deadline)?;
    live(state, consumers, id)?;
    let source = state[id].as_ref().expect("live tensor");
    if let ReluRows::Packed(rows) = &source.rows {
        let (m, d) = rows.materialize_checked(deadline)?;
        let (gerr, support) = if consumers[id] == 1 {
            let source = state[id].take().expect("live tensor");
            (source.gerr, source.support)
        } else {
            (source.gerr.clone(), source.support.clone())
        };
        consumers[id] -= 1;
        Ok((m, d, gerr, support))
    } else {
        let source = take_rows(state, consumers, id)?;
        let ReluRows::Dense(m, d) = source.rows else {
            unreachable!("checked dense source");
        };
        Ok((m, d, source.gerr, source.support))
    }
}

#[cfg(test)]
mod tests;
