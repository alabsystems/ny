// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

/// Simple xorshift64 RNG (avoids `rand` dependency).
pub(super) struct SimpleRng(u64);

impl SimpleRng {
    pub(super) fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    pub(super) fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 & 0xFFFF_FFFF) as u32
    }

    pub(super) fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    pub(super) fn next_bool(&mut self) -> bool {
        self.next_u32() & 1 == 0
    }
}
