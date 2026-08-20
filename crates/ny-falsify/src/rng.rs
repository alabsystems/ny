// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! A small, in-crate, seeded PRNG.
//!
//! In-crate on purpose: the port-fidelity argument is "given this seed, these
//! are the points the search emits", and that is only checkable from this repo
//! if the generator lives here. It is `xoshiro256**` over a `SplitMix64` seed
//! expansion — not cryptographic, and it does not need to be. It is NOT
//! numpy's PCG64, so `square`'s point stream is not bit-identical to the
//! Python portfolio's; `special` is deterministic and has no stream at all,
//! which is why that one IS pinned bit-for-bit.

/// Seeded `xoshiro256**`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rng {
    state: [u64; 4],
}

impl Rng {
    /// Seed the generator. The same seed always yields the same stream.
    pub fn new(seed: u64) -> Self {
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        Self {
            state: [next(), next(), next(), next()],
        }
    }

    fn next_u64(&mut self) -> u64 {
        let result = self.state[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.state[1] << 17;
        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);
        result
    }

    /// Uniform in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        // 53 significant bits, the same construction numpy uses.
        ((self.next_u64() >> 11) as f64) * (1.0 / 9_007_199_254_740_992.0)
    }

    /// Uniform integer in `[0, bound)`. `bound` must be non-zero.
    pub fn next_below(&mut self, bound: usize) -> usize {
        assert!(bound > 0, "next_below needs a non-zero bound");
        (self.next_u64() % bound as u64) as usize
    }

    /// `size` distinct indices drawn from `[0, n)` without replacement, written
    /// into `scratch` (a permutation buffer the caller owns so a hot loop does
    /// not allocate). Partial Fisher-Yates: `O(size)`, not `O(n)`, which is
    /// what makes `square` affordable at the 6912 free inputs it won
    /// `traffic_signs` at.
    pub fn choose_without_replacement(
        &mut self,
        n: usize,
        size: usize,
        scratch: &mut Vec<usize>,
    ) -> Vec<usize> {
        assert!(size <= n, "cannot choose {size} of {n} without replacement");
        if scratch.len() != n {
            scratch.clear();
            scratch.extend(0..n);
        }
        let mut picks = Vec::with_capacity(size);
        for i in 0..size {
            let j = i + self.next_below(n - i);
            scratch.swap(i, j);
            picks.push(scratch[i]);
        }
        picks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_same_seed_gives_the_same_stream() {
        let mut a = Rng::new(20_260_808);
        let mut b = Rng::new(20_260_808);
        let mut c = Rng::new(20_260_809);
        let left: Vec<f64> = (0..32).map(|_| a.next_f64()).collect();
        let right: Vec<f64> = (0..32).map(|_| b.next_f64()).collect();
        let other: Vec<f64> = (0..32).map(|_| c.next_f64()).collect();
        assert_eq!(left, right);
        assert_ne!(left, other);
        assert!(left.iter().all(|&v| (0.0..1.0).contains(&v)));
    }

    #[test]
    fn a_block_pick_is_distinct_and_covers_the_range() {
        let mut rng = Rng::new(7);
        let mut scratch = Vec::new();
        let mut seen = vec![false; 6912];
        for _ in 0..64 {
            let picks = rng.choose_without_replacement(6912, 3456, &mut scratch);
            assert_eq!(picks.len(), 3456);
            let mut sorted = picks.clone();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(
                sorted.len(),
                3456,
                "a block flip must not repeat a coordinate"
            );
            for pick in picks {
                assert!(pick < 6912);
                seen[pick] = true;
            }
        }
        assert!(
            seen.iter().all(|&s| s),
            "every coordinate must be reachable"
        );
    }
}
