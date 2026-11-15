// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use crate::amortise::CooperativeWork;
use faer::MatRef;
use ny_core::Result;

/// Cache-tiled copy of one COLUMN-major faer product block into the ROW-major
/// `[m, dst_cols]` destination buffer, landing at destination row `row_start`.
///
/// The naive `for i { for j { dst[[row_start + i, j]] = src[(i, j)] } }` walks
/// the source DOWN a column stride of `src.nrows()` while the destination
/// advances one element at a time, so on a wide block every inner step touches
/// a fresh source cache line and that line is evicted long before the next row
/// wants it. Sweeping 32×32 tiles keeps one tile of source columns and one tile
/// of destination rows resident together, and writing through `as_slice_mut()`
/// hoists ndarray's per-element stride arithmetic (and its bounds check) out of
/// the inner loop, exactly as the scalar `A·W` reductions in this module do.
///
/// BIT-IDENTICAL, and trivially so: this MOVES f64 values, it never computes
/// with them. Every destination element is assigned exactly once from exactly
/// the same source element as before, so no float enters an arithmetic
/// operation and there is nothing to round, reassociate or reorder. (The
/// alternative — writing the GEMM straight into the destination via
/// `faer_parallelism::mat_mul_f64_into` — is deliberately NOT taken: that
/// helper's own notes record that faer selects kernel families whose
/// per-element `k`-accumulation order tracks the destination layout, which
/// would change the published bits.)
///
/// Cache-tiled bit copy from the original column-major GEMM destination into
/// a row-major result. Full-k reduction, GEMM layout and tile order are intact.
/// A fixed view also accepts leased aligned storage without transferring its
/// owner. Every f64 copy is priced at two f32-equivalent units; no batch spans
/// more than one 32-element tile row. Refusal invalidates the private output.
pub(super) fn copy_col_major_block_into_row_major<P: CooperativeWork>(
    dst: &mut [f64],
    src: MatRef<'_, f64>,
    row_start: usize,
    dst_cols: usize,
    poll: &mut P,
) -> Result<()> {
    const TILE: usize = 32;
    poll.checkpoint()?;
    let rows = src.nrows();
    debug_assert_eq!(src.ncols(), dst_cols);
    for col_start in (0..dst_cols).step_by(TILE) {
        let col_end = (col_start + TILE).min(dst_cols);
        for tile_start in (0..rows).step_by(TILE) {
            let tile_end = (tile_start + TILE).min(rows);
            for i in tile_start..tile_end {
                poll.before(2 * (col_end - col_start))?;
                let row_base = (row_start + i) * dst_cols;
                let tile = &mut dst[row_base + col_start..row_base + col_end];
                for (offset, value) in tile.iter_mut().enumerate() {
                    *value = src[(i, col_start + offset)];
                }
            }
        }
    }
    poll.checkpoint()
}

#[cfg(test)]
#[path = "cooperative_copy_tests.rs"]
mod tests;
