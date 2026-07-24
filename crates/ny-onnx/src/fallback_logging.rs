// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_core::NyError;
use std::io::{self, Write};
use tracing::{subscriber::NoSubscriber, warn};

fn gpu_fallback_message(context: &str, err: &NyError) -> String {
    format!("WARN {context}; falling back to CPU: {err}")
}

fn has_no_tracing_subscriber() -> bool {
    tracing::dispatcher::get_default(|dispatch| dispatch.is::<NoSubscriber>())
}

fn write_gpu_fallback_line(mut writer: impl Write, context: &str, err: &NyError) -> io::Result<()> {
    writeln!(writer, "{}", gpu_fallback_message(context, err))
}

/// Surface recoverable GPU failures at warn level so test output shows CPU fallback.
pub(crate) fn warn_gpu_fallback(context: &str, err: &NyError) {
    warn!(error = %err, "{context}; falling back to CPU");
    if has_no_tracing_subscriber() {
        // Keep fallback visibility when library callers never install tracing.
        let _ = write_gpu_fallback_line(io::stderr().lock(), context, err);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;
    use std::sync::{Arc, Mutex};
    use tracing_subscriber::fmt::{self, MakeWriter};
    use tracing_subscriber::prelude::*;
    use tracing_subscriber::registry::Registry;

    #[derive(Clone, Default)]
    struct SharedBuffer(Arc<Mutex<Vec<u8>>>);

    struct SharedWriter(Arc<Mutex<Vec<u8>>>);

    impl SharedBuffer {
        fn snapshot(&self) -> String {
            String::from_utf8(self.0.lock().expect("buffer lock").clone())
                .expect("captured logs should be valid UTF-8")
        }
    }

    impl<'writer> MakeWriter<'writer> for SharedBuffer {
        type Writer = SharedWriter;

        fn make_writer(&'writer self) -> Self::Writer {
            SharedWriter(self.0.clone())
        }
    }

    impl Write for SharedWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0.lock().expect("buffer lock").extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn test_warn_gpu_fallback_emits_warn_level_message() {
        let buffer = SharedBuffer::default();
        let subscriber = Registry::default().with(
            fmt::layer()
                .with_ansi(false)
                .without_time()
                .with_target(false)
                .with_level(true)
                .with_writer(buffer.clone()),
        );

        tracing::subscriber::with_default(subscriber, || {
            warn_gpu_fallback(
                "GPU attention failed",
                &NyError::InvalidSpec("projection lookup failed".to_string()),
            );
        });

        let output = buffer.snapshot();
        assert!(
            output.contains("WARN"),
            "expected warn-level output, got: {output}"
        );
        assert!(
            output.contains("GPU attention failed; falling back to CPU"),
            "expected fallback context in output, got: {output}"
        );
        assert!(
            output.contains("projection lookup failed"),
            "expected error details in output, got: {output}"
        );
    }

    #[test]
    fn test_write_gpu_fallback_line_formats_stderr_message() {
        let mut buffer = Vec::new();
        write_gpu_fallback_line(
            &mut buffer,
            "GPU attention failed",
            &NyError::InvalidSpec("projection lookup failed".to_string()),
        )
        .expect("stderr fallback line should be writable");

        let output = String::from_utf8(buffer).expect("captured stderr should be valid UTF-8");
        assert!(
            output.contains("WARN GPU attention failed; falling back to CPU"),
            "expected warn-level stderr prefix, got: {output}"
        );
        assert!(
            output.contains("projection lookup failed"),
            "expected error details in stderr output, got: {output}"
        );
    }

    #[test]
    fn test_has_no_tracing_subscriber_detects_scoped_subscriber() {
        assert!(
            has_no_tracing_subscriber(),
            "expected bare unit test thread to start without a tracing subscriber"
        );

        let subscriber = Registry::default().with(
            fmt::layer()
                .with_ansi(false)
                .without_time()
                .with_target(false),
        );

        tracing::subscriber::with_default(subscriber, || {
            assert!(
                !has_no_tracing_subscriber(),
                "expected scoped tracing subscriber to suppress stderr fallback path"
            );
        });
    }
}
