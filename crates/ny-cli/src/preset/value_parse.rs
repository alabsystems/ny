// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use serde::de::Deserializer;
use serde::Deserialize;

pub(super) fn option_string_or_number<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum StringOrNumber {
        String(String),
        Float(f32),
        Integer(i64),
    }

    Ok(match Option::<StringOrNumber>::deserialize(deserializer)? {
        Some(StringOrNumber::String(value)) => Some(value),
        Some(StringOrNumber::Float(value)) => Some(value.to_string()),
        Some(StringOrNumber::Integer(value)) => Some(value.to_string()),
        None => None,
    })
}
