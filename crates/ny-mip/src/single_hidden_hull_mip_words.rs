// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Immutable serialized Source/witness words with their original Vec allocation.
//!
//! These words confer no authority. The existing model, geometry and Arc owner
//! checks still authenticate every retained row and grant. Preserve the Vec's
//! unused capacity: converting to an Arc slice would copy/reallocate its payload
//! while both buffers overlap, unlike moving this descriptor into the Arc.

#[repr(transparent)]
#[derive(Debug, Clone)]
pub(super) struct RetainedWords(Vec<u64>);

impl RetainedWords {
    pub(super) fn as_slice(&self) -> &[u64] {
        &self.0
    }

    // Existing payload controls inspect unused retained slots. No mutable Vec
    // or capacity-changing operation is exposed to production consumers.
    #[cfg(test)]
    pub(super) fn capacity(&self) -> usize {
        self.0.capacity()
    }
}

impl From<Vec<u64>> for RetainedWords {
    fn from(words: Vec<u64>) -> Self {
        Self(words)
    }
}

impl FromIterator<u64> for RetainedWords {
    fn from_iter<I: IntoIterator<Item = u64>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl std::ops::Deref for RetainedWords {
    type Target = [u64];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn retained_words_move_the_existing_payload_and_unused_capacity() {
        let mut input = Vec::with_capacity(37);
        input.extend_from_slice(&[7, 11, 13]);
        let pointer = input.as_ptr();
        let capacity = input.capacity();
        let owner = Arc::new(RetainedWords::from(input));
        assert_eq!(owner.as_ptr(), pointer);
        assert_eq!(owner.capacity(), capacity);
        assert_eq!(owner.as_slice(), &[7, 11, 13]);
    }
}
