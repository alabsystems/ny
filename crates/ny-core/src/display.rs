// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Display utilities shared across crates.

/// Truncate a name to fit within `width` characters, prepending "..." if too long.
///
/// Returns the name unchanged if it fits. Otherwise, returns the last
/// `width - 3` characters prefixed with "...".
///
/// Uses char-based (not byte-based) slicing to avoid panics on multi-byte UTF-8.
/// For `width < 4`, returns the first `width` characters without the "..." prefix
/// since there is not enough room for both the ellipsis and any content.
pub fn truncate_name(name: &str, width: usize) -> String {
    if width < 4 {
        return name.chars().take(width).collect();
    }
    let char_count = name.chars().count();
    if char_count <= width {
        name.to_string()
    } else {
        let suffix: String = name.chars().skip(char_count - (width - 3)).collect();
        format!("...{suffix}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_short_name_unchanged() {
        assert_eq!(truncate_name("relu", 10), "relu");
    }

    #[test]
    fn test_exact_width_unchanged() {
        assert_eq!(truncate_name("abcde", 5), "abcde");
    }

    #[test]
    fn test_long_name_truncated() {
        assert_eq!(truncate_name("very_long_layer_name", 10), "...er_name");
    }

    // Edge cases for width < 4 (#3042)

    #[test]
    fn test_width_zero() {
        assert_eq!(truncate_name("anything", 0), "");
    }

    #[test]
    fn test_width_one() {
        assert_eq!(truncate_name("hello", 1), "h");
    }

    #[test]
    fn test_width_two() {
        assert_eq!(truncate_name("hello", 2), "he");
    }

    #[test]
    fn test_width_three() {
        assert_eq!(truncate_name("hello", 3), "hel");
    }

    #[test]
    fn test_width_four_long_name() {
        assert_eq!(truncate_name("hello_world", 4), "...d");
    }

    #[test]
    fn test_empty_name() {
        assert_eq!(truncate_name("", 10), "");
    }

    #[test]
    fn test_empty_name_width_zero() {
        assert_eq!(truncate_name("", 0), "");
    }

    // Multi-byte UTF-8 tests (#3042)

    #[test]
    fn test_multibyte_utf8_short() {
        // 3 chars, each 3 bytes = 9 bytes but 3 chars
        assert_eq!(truncate_name("日本語", 5), "日本語");
    }

    #[test]
    fn test_multibyte_utf8_truncated() {
        // "日本語レイヤ名" = 7 chars. width=6 → "...イヤ名" (3 suffix chars + "...")
        assert_eq!(truncate_name("日本語レイヤ名", 6), "...イヤ名");
    }

    #[test]
    fn test_multibyte_utf8_exact() {
        assert_eq!(truncate_name("日本語", 3), "日本語");
    }

    #[test]
    fn test_multibyte_utf8_width_two() {
        // width < 4, so take first 2 chars
        assert_eq!(truncate_name("日本語", 2), "日本");
    }
}
