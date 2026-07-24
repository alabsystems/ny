// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use serde_json::Value;

#[derive(Debug, Clone)]
pub(crate) struct JsonCliError {
    payload: Value,
}

impl JsonCliError {
    pub(crate) fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        let payload = serde_json::json!({
            "error": code.into(),
            "message": message.into(),
        });
        Self { payload }
    }

    pub(crate) fn from_value(payload: Value) -> Self {
        Self { payload }
    }

    pub(crate) fn payload(&self) -> &Value {
        &self.payload
    }

    pub(crate) fn message(&self) -> Option<&str> {
        self.payload.get("message").and_then(|m| m.as_str())
    }
}

impl std::fmt::Display for JsonCliError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(message) = self.message() {
            return write!(f, "{message}");
        }
        write!(f, "{}", self.payload)
    }
}

impl std::error::Error for JsonCliError {}

pub(crate) fn find_json_cli_error(err: &anyhow::Error) -> Option<&JsonCliError> {
    err.chain()
        .find_map(|cause| cause.downcast_ref::<JsonCliError>())
}
