# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Cross-platform exclusive locking for the evidence tools.

WHY THIS EXISTS. Four production scripts locked their caches with a bare
``import fcntl`` at module scope. ``fcntl`` does not exist on Windows, so those
scripts could not be IMPORTED there at all — not merely "the lock degrades",
they raised ``ModuleNotFoundError`` on load and took eight test files down with
them at pytest collection time.

WHAT IS PRESERVED. The lock is a real cross-process exclusive lock on every
platform; nothing here silently becomes a no-op. A no-op would be worse than
the import error it replaces: these locks guard read-modify-write cycles on
measurement caches, and a lost lock corrupts evidence rather than failing loudly.

WHERE THE PLATFORMS GENUINELY DIFFER, and why each substitution is sound:

* ``flock`` locks the whole open file; ``msvcrt.locking`` locks a BYTE RANGE
  from the current offset. Windows therefore anchors at offset 0 and takes one
  byte. Every participant here uses this same module, so the convention is
  self-consistent — which is all a lock protocol requires.

* ``LK_LOCK`` is not ``LOCK_EX``: it retries for roughly ten seconds and then
  raises, where ``flock`` waits indefinitely. The blocking form loops so the
  contract ("return only once held") matches POSIX.

* Non-blocking failure surfaces as ``BlockingIOError`` on both platforms, so
  existing ``except BlockingIOError`` handlers keep working unchanged. Windows
  reports it as ``OSError(EDEADLOCK)``, which is translated here.

* A DIRECTORY descriptor cannot be locked on Windows and ``O_DIRECTORY`` does
  not exist there. ``directory_lock`` keeps the POSIX descriptor lock exactly as
  it was and, on Windows only, falls back to an exclusive lock on a sentinel
  file inside that directory. Mutual exclusion between participants is
  preserved; the object being locked differs.
"""

from __future__ import annotations

import errno
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

_WINDOWS = sys.platform == "win32"

if _WINDOWS:  # pragma: no cover - platform-selected
    import msvcrt
else:  # pragma: no cover - platform-selected
    import fcntl

#: Sentinel used only on Windows, where a directory handle cannot be locked.
WINDOWS_DIRECTORY_SENTINEL = ".ny-directory.lock"


def lock_exclusive(fileno: int, *, blocking: bool = True) -> None:
    """Take an exclusive lock on ``fileno``, or raise ``BlockingIOError``.

    With ``blocking=True`` this returns only once the lock is held, matching
    ``flock(LOCK_EX)``. With ``blocking=False`` it raises ``BlockingIOError``
    immediately when another process holds the lock, matching ``LOCK_NB``.
    """
    if not _WINDOWS:
        flags = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
        fcntl.flock(fileno, flags)
        return

    os.lseek(fileno, 0, os.SEEK_SET)
    if not blocking:
        try:
            msvcrt.locking(fileno, msvcrt.LK_NBLCK, 1)
        except OSError as error:
            if error.errno in (errno.EDEADLOCK, errno.EACCES, errno.EAGAIN):
                raise BlockingIOError(
                    errno.EWOULDBLOCK, "the exclusive lock is held by another process"
                ) from error
            raise
        return

    # LK_LOCK gives up after ~10s; flock does not. Loop so the blocking
    # contract is the POSIX one. Only contention is retried — any other OSError
    # propagates rather than spinning.
    while True:
        try:
            msvcrt.locking(fileno, msvcrt.LK_LOCK, 1)
            return
        except OSError as error:
            if error.errno not in (errno.EDEADLOCK, errno.EACCES, errno.EAGAIN):
                raise
            os.lseek(fileno, 0, os.SEEK_SET)


def unlock(fileno: int) -> None:
    """Release a lock taken by :func:`lock_exclusive`."""
    if not _WINDOWS:
        fcntl.flock(fileno, fcntl.LOCK_UN)
        return
    os.lseek(fileno, 0, os.SEEK_SET)
    try:
        msvcrt.locking(fileno, msvcrt.LK_UNLCK, 1)
    except OSError:
        # Closing the handle releases the range regardless; a failure here must
        # not mask the caller's own exception during unwinding.
        pass


@contextmanager
def directory_lock(directory: Path) -> Iterator[None]:
    """Serialize writers against ``directory``.

    POSIX keeps the original descriptor lock verbatim. Windows cannot lock a
    directory handle, so it locks a sentinel file inside the directory instead.
    """
    if not _WINDOWS:
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            lock_exclusive(descriptor)
            yield
        finally:
            os.close(descriptor)
        return

    sentinel = Path(directory) / WINDOWS_DIRECTORY_SENTINEL
    with sentinel.open("a+b") as handle:
        lock_exclusive(handle.fileno())
        try:
            yield
        finally:
            unlock(handle.fileno())
