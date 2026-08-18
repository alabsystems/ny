// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::WgpuDevice;
use bytemuck::Pod;
use ny_core::{NyError, Result};
use std::mem::size_of;
use std::sync::mpsc::TryRecvError;
use std::time::Instant;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct ReadTwoBuffersTiming {
    pub(super) map_requests_seconds: f64,
    pub(super) poll_wait_seconds: f64,
    pub(super) copy_seconds: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReadbackWait {
    Unbounded,
    Expired,
    Bounded(std::time::Duration),
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
    fn readback_wait() -> ReadbackWait {
        match crate::wgpu_device::CALL_LOCAL_CROWN_DEADLINE.with(std::cell::Cell::get) {
            None => ReadbackWait::Unbounded,
            Some(deadline) => match deadline.checked_duration_since(Instant::now()) {
                Some(remaining) => ReadbackWait::Bounded(remaining),
                None => ReadbackWait::Expired,
            },
        }
    }

    fn poll_readback(device: &wgpu::Device, label: &str) -> Result<()> {
        let timeout = match Self::readback_wait() {
            ReadbackWait::Unbounded => None,
            ReadbackWait::Bounded(remaining) => Some(remaining),
            ReadbackWait::Expired => {
                super::ops::intermediate_sweep::note_post_submit_abort();
                return Err(NyError::DeadlineExceeded(format!(
                    "{label}: deadline expired before GPU readback"
                )));
            }
        };
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout,
            })
            .map_err(|error| match error {
                wgpu::PollError::Timeout => {
                    super::ops::intermediate_sweep::note_post_submit_abort();
                    NyError::DeadlineExceeded(format!(
                        "{label}: GPU readback did not complete before the call-local deadline"
                    ))
                }
                other => NyError::InternalError(format!(
                    "{label}: device poll failed during GPU readback: {other}"
                )),
            })?;
        Ok(())
    }

    /// Deadline-bounded completion fence for device-resident intermediate
    /// sweep units. Unlike a readback this transfers no bytes, but it is a real
    /// synchronization and ensures command-buffer-owned scratch allocations
    /// from the completed unit cannot overlap the next unit's admitted peak.
    pub(super) fn wait_for_intermediate_sweep_unit(&self, label: &str) -> Result<()> {
        Self::poll_readback(&self.device, label)?;
        super::ops::intermediate_sweep::note_device_to_host(0, 0, 1);
        Ok(())
    }

    fn receive_map_result(
        receiver: &std::sync::mpsc::Receiver<std::result::Result<(), wgpu::BufferAsyncError>>,
        label: &str,
    ) -> Result<std::result::Result<(), wgpu::BufferAsyncError>> {
        receiver.try_recv().map_err(|error| match error {
            TryRecvError::Empty => NyError::InternalError(format!(
                "{label}: device poll completed without delivering the map callback"
            )),
            TryRecvError::Disconnected => {
                NyError::InternalError(format!("{label}: map callback channel disconnected"))
            }
        })
    }

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
        Self::poll_readback(device, label)?;
        let map_result = Self::receive_map_result(&receiver, label)?;
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

        super::ops::intermediate_sweep::note_device_to_host(
            count.saturating_mul(size_of::<T>()),
            1,
            1,
        );

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
        Self::poll_readback(device, "read_buffers_batched")?;

        let mut results = Vec::with_capacity(buffers.len());
        for (idx, ((_, count), slice)) in buffers.iter().zip(slices.iter()).enumerate() {
            let map_result =
                Self::receive_map_result(&receivers[idx], &format!("read_buffers_batched[{idx}]"))?;
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
        let bytes = buffers.iter().fold(0usize, |total, (_, count)| {
            total.saturating_add(count.saturating_mul(size_of::<f32>()))
        });
        super::ops::intermediate_sweep::note_device_to_host(bytes, buffers.len(), 1);
        Ok(results)
    }

    /// Mixed-type sibling used by the worded intermediate-sweep carrier: map
    /// all f32 value/error staging buffers and the final u32 row accumulator
    /// before one bounded poll, preserving a single transaction-final
    /// synchronization and an exact nine-buffer readback receipt.
    pub(super) fn read_sweep_carrier_batched(
        device: &wgpu::Device,
        f32_buffers: &[(&wgpu::Buffer, usize)],
        row_words: (&wgpu::Buffer, usize),
    ) -> Result<(Vec<Vec<f32>>, Vec<u32>)> {
        let mut f32_slices = Vec::with_capacity(f32_buffers.len());
        let mut f32_receivers = Vec::with_capacity(f32_buffers.len());
        let mut guards = Vec::with_capacity(f32_buffers.len() + 1);
        for (buffer, _) in f32_buffers {
            let slice = buffer.slice(..);
            let (sender, receiver) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
            guards.push(UnmapOnDrop::new(buffer));
            f32_slices.push(slice);
            f32_receivers.push(receiver);
        }
        let row_slice = row_words.0.slice(..);
        let (row_sender, row_receiver) = std::sync::mpsc::channel();
        row_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = row_sender.send(result);
        });
        guards.push(UnmapOnDrop::new(row_words.0));

        Self::poll_readback(device, "read_sweep_carrier_batched")?;

        let mut values = Vec::with_capacity(f32_buffers.len());
        for (index, ((_, count), slice)) in f32_buffers.iter().zip(f32_slices.iter()).enumerate() {
            Self::receive_map_result(
                &f32_receivers[index],
                &format!("read_sweep_carrier_batched[{index}]"),
            )?
            .map_err(|error| {
                NyError::InternalError(format!(
                    "read_sweep_carrier_batched[{index}]: map_async failed: {error:?}"
                ))
            })?;
            let data = slice.get_mapped_range();
            let typed: &[f32] = bytemuck::try_cast_slice(&data).map_err(|error| {
                NyError::InternalError(format!(
                    "read_sweep_carrier_batched[{index}]: invalid f32 layout: {error:?}"
                ))
            })?;
            values.push(
                typed
                    .get(..*count)
                    .ok_or_else(|| {
                        NyError::InternalError(format!(
                            "read_sweep_carrier_batched[{index}]: requested {count}, mapped {}",
                            typed.len()
                        ))
                    })?
                    .to_vec(),
            );
            drop(data);
        }

        Self::receive_map_result(&row_receiver, "read_sweep_carrier_batched[rows]")?.map_err(
            |error| {
                NyError::InternalError(format!(
                    "read_sweep_carrier_batched[rows]: map_async failed: {error:?}"
                ))
            },
        )?;
        let row_data = row_slice.get_mapped_range();
        let typed_rows: &[u32] = bytemuck::try_cast_slice(&row_data).map_err(|error| {
            NyError::InternalError(format!(
                "read_sweep_carrier_batched[rows]: invalid u32 layout: {error:?}"
            ))
        })?;
        let rows = typed_rows
            .get(..row_words.1)
            .ok_or_else(|| {
                NyError::InternalError(format!(
                    "read_sweep_carrier_batched[rows]: requested {}, mapped {}",
                    row_words.1,
                    typed_rows.len()
                ))
            })?
            .to_vec();
        drop(row_data);

        for guard in guards {
            guard.unmap();
        }
        let value_bytes = f32_buffers.iter().try_fold(0usize, |total, (_, count)| {
            total.checked_add(count.checked_mul(size_of::<f32>())?)
        });
        let bytes = value_bytes
            .and_then(|total| total.checked_add(row_words.1.checked_mul(size_of::<u32>())?))
            .ok_or_else(|| NyError::InternalError("sweep readback byte count overflow".into()))?;
        super::ops::intermediate_sweep::note_device_to_host(bytes, f32_buffers.len() + 1, 1);
        Ok((values, rows))
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
        Self::poll_readback(device, "wgpu read_two_buffers")?;
        let poll_wait_seconds = poll_start.elapsed().as_secs_f64();

        // Check both map results.
        let copy_start = Instant::now();
        let map_a = Self::receive_map_result(&receiver_a, "wgpu read_two_buffers map_a")?;
        map_a.map_err(|err| {
            NyError::InternalError(format!(
                "wgpu read_two_buffers: map_async buf_a failed: {err:?}"
            ))
        })?;

        let map_b = Self::receive_map_result(&receiver_b, "wgpu read_two_buffers map_b")?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn readback_wait_tracks_call_local_deadline_and_restores_scope() {
        assert_eq!(WgpuDevice::readback_wait(), ReadbackWait::Unbounded);
        {
            let _scope = crate::wgpu_device::CallLocalCrownDeadlineScope::arm(
                Instant::now() + Duration::from_mins(1),
            );
            assert!(matches!(
                WgpuDevice::readback_wait(),
                ReadbackWait::Bounded(remaining) if !remaining.is_zero()
            ));
        }
        assert_eq!(WgpuDevice::readback_wait(), ReadbackWait::Unbounded);
        {
            let _scope = crate::wgpu_device::CallLocalCrownDeadlineScope::arm(
                Instant::now()
                    .checked_sub(Duration::from_millis(1))
                    .expect("one millisecond is representable"),
            );
            assert_eq!(WgpuDevice::readback_wait(), ReadbackWait::Expired);
        }
    }

    #[test]
    fn map_callback_receive_distinguishes_empty_and_disconnected() {
        let (sender, receiver) = std::sync::mpsc::channel();
        let empty = WgpuDevice::receive_map_result(&receiver, "scripted").unwrap_err();
        assert!(empty.to_string().contains("without delivering"));
        sender.send(Ok(())).unwrap();
        assert!(WgpuDevice::receive_map_result(&receiver, "scripted")
            .unwrap()
            .is_ok());
        drop(sender);
        let disconnected = WgpuDevice::receive_map_result(&receiver, "scripted").unwrap_err();
        assert!(disconnected.to_string().contains("disconnected"));
    }
}
