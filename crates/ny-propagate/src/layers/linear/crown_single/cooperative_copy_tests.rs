// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::copy_col_major_block_into_row_major;
use crate::amortise::{NoPoll, WeightedPoll, POLL_CHUNK_UNITS};
use faer::{Mat, MatRef};
use ny_core::NyError;

const SENTINEL: u64 = 0x7ff8_1020_3040_5060;

#[test]
fn tiled_copy_preserves_raw_bits_ragged_tiles_offsets_and_strided_sources() {
    let bits = [
        0,
        1 << 63,
        1,
        (1 << 63) | 1,
        0x7ff0_0000_0000_0000,
        0xfff0_0000_0000_0000,
        0x7ff8_0000_0000_0097,
        0x7ff0_0000_0000_0001,
        0x3ff0_0000_0000_0000,
    ];
    for (rows, cols) in [(1, 1), (7, 33), (70, 45), (65, 120), (0, 45), (70, 0)] {
        let source = Mat::from_fn(rows, cols, |i, j| {
            f64::from_bits(bits[(i * 17 + j) % bits.len()])
        });
        for weighted in [false, true] {
            let mut destination = vec![f64::from_bits(SENTINEL); (rows + 4) * cols];
            let mut units = 0;
            if weighted {
                let mut poll = WeightedPoll::new(|n| {
                    units += n;
                    Ok(())
                });
                copy_col_major_block_into_row_major(
                    &mut destination,
                    source.as_ref(),
                    2,
                    cols,
                    &mut poll,
                )
                .unwrap();
                assert_eq!(units, 2 * rows * cols);
            } else {
                copy_col_major_block_into_row_major(
                    &mut destination,
                    source.as_ref(),
                    2,
                    cols,
                    &mut NoPoll,
                )
                .unwrap();
            }
            for (index, value) in destination.iter().enumerate() {
                let row = index / cols;
                let col = index % cols;
                let expected = if row >= 2 && row < rows + 2 {
                    source[(row - 2, col)].to_bits()
                } else {
                    SENTINEL
                };
                assert_eq!(
                    value.to_bits(),
                    expected,
                    "row={row} col={col} weighted={weighted}"
                );
            }
        }
    }
    let data: Vec<f64> = (0..(11 * 19))
        .map(|i| f64::from_bits(bits[i % bits.len()]))
        .collect();
    let source = MatRef::from_column_major_slice_with_stride(&data, 7, 19, 11);
    let mut destination = vec![0.0; 7 * 19];
    copy_col_major_block_into_row_major(&mut destination, source, 0, 19, &mut NoPoll).unwrap();
    for i in 0..7 {
        for j in 0..19 {
            assert_eq!(
                destination[i * 19 + j].to_bits(),
                data[i + j * 11].to_bits()
            );
        }
    }
}

#[test]
fn tiled_copy_refuses_after_2048_real_f64_copies_and_keeps_the_source() {
    let source = Mat::from_fn(70, 45, |i, j| (i * 45 + j + 1) as f64);
    let mut destination = vec![f64::from_bits(SENTINEL); 70 * 45];
    let mut units = 0;
    let mut poll = WeightedPoll::new(|n| {
        units += n;
        if units >= POLL_CHUNK_UNITS {
            Err(NyError::DeadlineExceeded("copy forced expiry".into()))
        } else {
            Ok(())
        }
    });
    let error =
        copy_col_major_block_into_row_major(&mut destination, source.as_ref(), 0, 45, &mut poll)
            .unwrap_err();
    assert!(error.is_deadline_exceeded());
    assert_eq!(units, POLL_CHUNK_UNITS);
    assert_eq!(
        destination
            .iter()
            .filter(|v| v.to_bits() != SENTINEL)
            .count(),
        2048
    );
    for i in 0..70 {
        for j in 0..45 {
            assert_eq!(source[(i, j)], (i * 45 + j + 1) as f64);
        }
    }
}

#[test]
fn tiled_copy_entry_and_final_publication_refusals_are_propagated() {
    let source = Mat::from_fn(3, 5, |_, _| 1.0);
    for fail_call in [1, 2] {
        let mut destination = vec![f64::from_bits(SENTINEL); 15];
        let mut calls = 0;
        let mut poll = WeightedPoll::new(|_| {
            calls += 1;
            if calls == fail_call {
                Err(NyError::DeadlineExceeded("copy boundary refused".into()))
            } else {
                Ok(())
            }
        });
        assert!(copy_col_major_block_into_row_major(
            &mut destination,
            source.as_ref(),
            0,
            5,
            &mut poll
        )
        .unwrap_err()
        .is_deadline_exceeded());
        assert_eq!(
            destination
                .iter()
                .filter(|v| v.to_bits() != SENTINEL)
                .count(),
            if fail_call == 1 { 0 } else { 15 }
        );
    }
}
