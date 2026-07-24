// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::WgpuDevice;
use bytemuck::Pod;
use ny_core::{NyError, Result};
use std::time::Instant;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct ReadTwoBuffersTiming {
    pub(super) map_requests_seconds: f64,
    pub(super) poll_wait_seconds: f64,
    pub(super) copy_seconds: f64,
}

/// RAII guard that unmaps a wgpu buffer on drop.
///
/// Once `map_async` has been issued for a buffer slice and the mapping has
/// (or may have) taken effect, the buffer MUST be unmapped before it can be
/// mapped again. If a readback fails partway (map callback error, byte-layout
/// cast error, short buffer), an early `?` return would otherwise leave the
/// buffer mapped — and the *next* `map_async` on that pooled/cached buffer hits
/// wgpu's `assert_eq!(mapped_range, 0..0, "Buffer is already mapped")` panic
/// (`wgpu-28.0.0/src/api/buffer.rs:572`), which aborts the process under
/// panic=abort. This guard ensures unmap runs on every exit path so a failed
/// GPU readback degrades to a clean `Err` (caller falls back to CPU) instead.
struct UnmapOnDrop<'a> {
    buffer: &'a wgpu::Buffer,
    armed: bool,
}

impl<'a> UnmapOnDrop<'a> {
    fn new(buffer: &'a wgpu::Buffer) -> Self {
        Self {
            buffer,
            armed: true,
        }
    }

    /// Unmap now and disarm the guard (so Drop is a no-op).
    fn unmap(mut self) {
        if self.armed {
            self.buffer.unmap();
            self.armed = false;
        }
    }
}

impl Drop for UnmapOnDrop<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.buffer.unmap();
        }
    }
}

impl WgpuDevice {
    fn read_buffer_typed<T: Pod>(
        device: &wgpu::Device,
        buffer: &wgpu::Buffer,
        count: usize,
        label: &str,
    ) -> Result<Vec<T>> {
        let buffer_slice = buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        // `map_async` sets `mapped_range` synchronously regardless of whether
        // the callback later succeeds, so arm the unmap guard immediately: every
        // early return below (map error, cast error, short buffer) must unmap or
        // the next map_async on this pooled buffer aborts the process via wgpu's
        // `assert_eq!(mapped_range, 0..0, "Buffer is already mapped")`.
        let unmap_guard = UnmapOnDrop::new(buffer);
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        let map_result = receiver.recv().map_err(|err| {
            NyError::InternalError(format!("{label}: map callback channel disconnected: {err}"))
        })?;
        map_result
            .map_err(|err| NyError::InternalError(format!("{label}: map_async failed: {err:?}")))?;

        let data = buffer_slice.get_mapped_range();
        let typed: &[T] = bytemuck::try_cast_slice(&data).map_err(|err| {
            NyError::InternalError(format!(
                "{label}: invalid mapped byte layout for typed slice: {err:?}"
            ))
        })?;
        let result = typed
            .get(..count)
            .ok_or_else(|| {
                NyError::InternalError(format!(
                    "{label}: requested {count} values, mapped buffer has {}",
                    typed.len()
                ))
            })?
            .to_vec();
        drop(data);
        unmap_guard.unmap();

        Ok(result)
    }

    pub(super) fn read_buffer(
        device: &wgpu::Device,
        buffer: &wgpu::Buffer,
        count: usize,
    ) -> Result<Vec<f32>> {
        Self::read_buffer_typed::<f32>(device, buffer, count, "wgpu read_buffer")
    }

    pub(super) fn read_u64_buffer(
        device: &wgpu::Device,
        buffer: &wgpu::Buffer,
        count: usize,
    ) -> Result<Vec<u64>> {
        Self::read_buffer_typed::<u64>(device, buffer, count, "wgpu read_u64_buffer")
    }

    /// Read back an `array<u32>` storage buffer as raw bit patterns.
    ///
    /// Used by the IEEE-754 f32-model self-check (`ops/f32_selfcheck.rs`), which
    /// compares on-device bit patterns byte-for-byte against a CPU reference. A pure
    /// reinterpret (`bytemuck`) of the mapped bytes — no float load that could
    /// canonicalize a NaN — so the returned bits are EXACTLY what the shader stored.
    pub(super) fn read_u32_buffer(
        device: &wgpu::Device,
        buffer: &wgpu::Buffer,
        count: usize,
    ) -> Result<Vec<u32>> {
        Self::read_buffer_typed::<u32>(device, buffer, count, "wgpu read_u32_buffer")
    }

    /// Map N readback buffers concurrently with a single `device.poll`.
    ///
    /// Generalizes `read_two_buffers_profiled` to an arbitrary slice of
    /// `(buffer, count)` pairs. wgpu only fires a `map_async` callback during a
    /// `poll`/`submit`, so reading N buffers via N sequential `read_buffer` calls
    /// costs N blocking `device.poll(Wait)` round-trips. When every buffer was
    /// filled by the SAME prior submit (true for the resident-backward final
    /// download: all staging buffers are written in one `res_dl` encoder submit),
    /// they are all ready after the first poll — so issuing every `map_async`
    /// first and then a SINGLE blocking poll fires all callbacks at once.
    ///
    /// Bit-identical to calling `read_buffer` per buffer: each result is the exact
    /// same `get_mapped_range()[..count].to_vec()` of the same staging buffer;
    /// only the number of (otherwise no-op-after-the-first) polls changes. Order
    /// of returned vecs matches the input slice order.
    pub(super) fn read_buffers_batched(
        device: &wgpu::Device,
        buffers: &[(&wgpu::Buffer, usize)],
    ) -> Result<Vec<Vec<f32>>> {
        // Issue every map request before polling. Arm an unmap guard per buffer
        // immediately: `map_async` sets `mapped_range` synchronously, so every
        // early return below must unmap or the next `map_async` on a pooled
        // buffer aborts the process (see `read_two_buffers_profiled`).
        let mut slices = Vec::with_capacity(buffers.len());
        let mut receivers = Vec::with_capacity(buffers.len());
        let mut guards = Vec::with_capacity(buffers.len());
        for (buffer, _) in buffers {
            let slice = buffer.slice(..);
            let (sender, receiver) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = sender.send(r);
            });
            guards.push(UnmapOnDrop::new(buffer));
            slices.push(slice);
            receivers.push(receiver);
        }

        // Single poll completes every mapping.
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        let mut results = Vec::with_capacity(buffers.len());
        for (idx, ((_, count), slice)) in buffers.iter().zip(slices.iter()).enumerate() {
            let map_result = receivers[idx].recv().map_err(|err| {
                NyError::InternalError(format!(
                    "read_buffers_batched: map callback channel disconnected: {err}"
                ))
            })?;
            map_result.map_err(|err| {
                NyError::InternalError(format!("read_buffers_batched: map_async failed: {err:?}"))
            })?;
            let data = slice.get_mapped_range();
            let typed: &[f32] = bytemuck::try_cast_slice(&data).map_err(|err| {
                NyError::InternalError(format!(
                    "read_buffers_batched: invalid mapped byte layout: {err:?}"
                ))
            })?;
            let vec = typed
                .get(..*count)
                .ok_or_else(|| {
                    NyError::InternalError(format!(
                        "read_buffers_batched: requested {count} values, mapped buffer has {}",
                        typed.len()
                    ))
                })?
                .to_vec();
            drop(data);
            results.push(vec);
        }

        for guard in guards {
            guard.unmap();
        }
        Ok(results)
    }

    /// Map two readback buffers concurrently with a single `device.poll` (#3397).
    ///
    /// The standard `read_buffer` issues map_async + poll + read sequentially for
    /// each buffer. When both buffers were submitted in the same command encoder
    /// (as in CROWN backward), both are ready simultaneously after one poll. This
    /// method issues both map_async calls first, then does a single blocking poll,
    /// eliminating the second poll's overhead while returning a host-side timing
    /// breakdown for the mapping, wait, and copy phases.
    pub(super) fn read_two_buffers_profiled(
        device: &wgpu::Device,
        buf_a: &wgpu::Buffer,
        buf_b: &wgpu::Buffer,
        count_a: usize,
        count_b: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, ReadTwoBuffersTiming)> {
        let slice_a = buf_a.slice(..);
        let slice_b = buf_b.slice(..);

        let (sender_a, receiver_a) = std::sync::mpsc::channel();
        let (sender_b, receiver_b) = std::sync::mpsc::channel();

        // Issue both map requests before polling.
        //
        // NOTE: `map_async` sets the buffer's `mapped_range` SYNCHRONOUSLY (see
        // `wgpu-28.0.0/src/api/buffer.rs:574`), independent of whether the map
        // callback later reports success or failure. So once `map_async` is
        // called the buffer MUST be unmapped before it can be mapped again,
        // even on the error path — otherwise the next `map_async` on this
        // pooled buffer hits `assert_eq!(mapped_range, 0..0)` and aborts the
        // process. Arm the unmap guards immediately so all early returns below
        // (channel/map errors, cast errors, short buffers) still unmap.
        let map_request_start = Instant::now();
        slice_a.map_async(wgpu::MapMode::Read, move |r| {
            let _ = sender_a.send(r);
        });
        let unmap_a = UnmapOnDrop::new(buf_a);
        slice_b.map_async(wgpu::MapMode::Read, move |r| {
            let _ = sender_b.send(r);
        });
        let unmap_b = UnmapOnDrop::new(buf_b);
        let map_requests_seconds = map_request_start.elapsed().as_secs_f64();

        // Single poll completes both mappings.
        let poll_start = Instant::now();
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        let poll_wait_seconds = poll_start.elapsed().as_secs_f64();

        // Check both map results.
        let copy_start = Instant::now();
        let map_a = receiver_a.recv().map_err(|err| {
            NyError::InternalError(format!(
                "wgpu read_two_buffers: map_a channel disconnected: {err}"
            ))
        })?;
        map_a.map_err(|err| {
            NyError::InternalError(format!(
                "wgpu read_two_buffers: map_async buf_a failed: {err:?}"
            ))
        })?;

        let map_b = receiver_b.recv().map_err(|err| {
            NyError::InternalError(format!(
                "wgpu read_two_buffers: map_b channel disconnected: {err}"
            ))
        })?;
        map_b.map_err(|err| {
            NyError::InternalError(format!(
                "wgpu read_two_buffers: map_async buf_b failed: {err:?}"
            ))
        })?;

        // Read both mapped ranges.
        let data_a = slice_a.get_mapped_range();
        let f32_a: &[f32] = bytemuck::try_cast_slice(&data_a).map_err(|err| {
            NyError::InternalError(format!("wgpu read_two_buffers: buf_a cast failed: {err:?}"))
        })?;
        let result_a = f32_a
            .get(..count_a)
            .ok_or_else(|| {
                NyError::InternalError(format!(
                    "wgpu read_two_buffers: buf_a has {} values, requested {count_a}",
                    f32_a.len()
                ))
            })?
            .to_vec();

        let data_b = slice_b.get_mapped_range();
        let f32_b: &[f32] = bytemuck::try_cast_slice(&data_b).map_err(|err| {
            NyError::InternalError(format!("wgpu read_two_buffers: buf_b cast failed: {err:?}"))
        })?;
        let result_b = f32_b
            .get(..count_b)
            .ok_or_else(|| {
                NyError::InternalError(format!(
                    "wgpu read_two_buffers: buf_b has {} values, requested {count_b}",
                    f32_b.len()
                ))
            })?
            .to_vec();

        drop(data_a);
        drop(data_b);
        unmap_a.unmap();
        unmap_b.unmap();
        let copy_seconds = copy_start.elapsed().as_secs_f64();

        Ok((
            result_a,
            result_b,
            ReadTwoBuffersTiming {
                map_requests_seconds,
                poll_wait_seconds,
                copy_seconds,
            },
        ))
    }
}
