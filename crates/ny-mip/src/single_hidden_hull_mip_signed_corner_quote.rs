// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Payload/work quotes for the bounded shared-input corner modes.
//! The wrapper admits and debits these once against the original work and
//! byte ledger, including every existing reservation.
use super::core::{MAX_D, MAX_K};

pub(in crate::single_hidden_hull::mip_cuts::signed) const CHILD_PRICES: usize = 16;

#[derive(Debug, Clone, Copy)]
pub(in crate::single_hidden_hull::mip_cuts::signed) struct Quote {
    pub k: usize,
    pub work: usize,
    pub bytes: usize,
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Quote decomposition is checked by tests; production consumes total work."
        )
    )]
    pub bind_work: usize,
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Quote decomposition is checked by tests; production consumes total work."
        )
    )]
    pub price_work: usize,
    pub seal_words: usize,
}

pub(in crate::single_hidden_hull::mip_cuts::signed) fn quote(
    d: usize,
    bits: &[u8],
    k: usize,
) -> Option<Quote> {
    if d == 0
        || d > MAX_D
        || k == 0
        || k > MAX_K
        || k > d
        || bits.is_empty()
        || bits.len() > 128
        || bits.iter().any(|&n| n == 0 || n > 8)
    {
        return None;
    }
    let c = bits.len();
    let mut n = 0usize;
    let mut m = 0usize;
    let mut s = 0usize;
    for &b in bits {
        let masks = 1usize << b;
        n = n.checked_add(b as usize)?;
        m = m.checked_add(masks)?;
        s = s.checked_add(masks.checked_mul(b as usize)?)?;
    }
    if m > 4096 {
        return None;
    }
    let z = 1usize << k;
    let l = c.checked_add(2usize.checked_mul(n)?)?.checked_mul(z)?;
    let seal_words = 6usize
        .checked_add(2usize.checked_mul(d)?)?
        .checked_add(c)?
        .checked_add(k)?
        .checked_add(l)?;

    // Ranking: raw coordinate removal, endpoint interval products, complete
    // per-component reduction, and bounded D^2 global-g pricing/sort advice.
    let rank = 4096usize
        .checked_add(32usize.checked_mul(m)?.checked_mul(d.checked_add(1)?)?)?
        .checked_add(32usize.checked_mul(c)?.checked_mul(d)?)?
        .checked_add(32usize.checked_mul(d)?.checked_mul(d)?)?
        .checked_add(32usize.checked_mul(c.checked_add(d)?.checked_add(k)?.checked_add(1)?)?)?;
    // Second replay: coefficients validated; selected raw summands removed;
    // one Gray corner scratch generated; then contiguous min slices. Four
    // work units per per-bit min item replace eight for indexed scatter.
    let collect = m
        .checked_mul(
            8usize
                .checked_mul(d)?
                .checked_add(64usize.checked_mul(k)?)?
                .checked_add(64)?,
        )?
        .checked_add(16usize.checked_mul(s)?)?
        .checked_add(
            z.checked_mul(
                24usize
                    .checked_mul(m)?
                    .checked_add(4usize.checked_mul(s)?)?,
            )?,
        )?
        .checked_add(
            32usize.checked_mul(
                l.checked_add(c)?
                    .checked_add(d)?
                    .checked_add(k)?
                    .checked_add(1)?,
            )?,
        )?;
    // This is an independent optional seal/bind, never an enlargement of the
    // ordinary child bind quote. It is prepaid even if optional bind refuses.
    let bind_work = 64usize.checked_mul(seal_words)?;
    let price_work = 64usize
        .checked_mul(d.checked_add(k)?.checked_add(c)?.checked_add(1)?)?
        .checked_add(
            z.checked_mul(
                32usize
                    .checked_add(16usize.checked_mul(c)?)?
                    .checked_add(8usize.checked_mul(n)?)?,
            )?,
        )?;
    let work = rank
        .checked_add(collect)?
        .checked_add(bind_work)?
        .checked_add((1 + CHILD_PRICES).checked_mul(price_work)?)?;
    // Raw values + one u64 seal; boxes/counts/axes and their sealed copies;
    // 4096 transient bytes (including the 1024-byte Gray scratch) and 512
    // bytes per maximum eight child workers. All shared-state INLINE growth
    // is separately charged by the initial actual-size fixed metadata census.
    let bytes = 16usize
        .checked_mul(l)?
        .checked_add(32usize.checked_mul(d)?)?
        .checked_add(16usize.checked_mul(c)?)?
        .checked_add(16usize.checked_mul(k)?)?
        .checked_add(8192)?;
    Some(Quote {
        k,
        work,
        bytes,
        bind_work,
        price_work,
        seal_words,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn corner_quote_actual_shape_and_bounded_mode_fallback() {
        // Actual captured NEGATIVE topology only; not a runtime admission.
        let bits = [8, 8, 8, 8, 8, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1];
        for (k, work, bytes) in [
            (7, 14_933_760, 263_472),
            (6, 8_662_848, 136_480),
            (5, 5_484_928, 72_976),
            (4, 3_853_504, 41_216),
        ] {
            let q = quote(30, &bits, k).unwrap();
            assert_eq!((q.k, q.work, q.bytes), (k, work, bytes));
            assert_eq!(q.bind_work, 64 * q.seal_words);
            assert!(q.price_work > 0);
        }
        assert!(quote(30, &bits, 8).is_none());
        assert!(quote(0, &bits, 1).is_none());
        assert!(quote(2, &[9], 1).is_none());
        assert!(quote(32, &[8; 17], 7).is_none());
    }
}
