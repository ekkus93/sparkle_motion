from contextlib import contextmanager
import os


"""Small portable file-lock helper used by ADK shim fixtures.

Provides an ``exclusive_lock(fileobj)`` context manager that acquires an
exclusive lock on the given open file object. On POSIX this uses ``fcntl``,
and on Windows it uses ``msvcrt`` locking. The helper keeps the API tiny so
fixture code can switch from platform-specific ``fcntl`` calls to this
cross-platform helper without adding external deps.
"""


if os.name == "nt":
    import msvcrt

    @contextmanager
    def exclusive_lock(fileobj):
        """Acquire an exclusive lock on *fileobj* (Windows).

        This locks a single byte at the start of the file which is sufficient
        for coordination between processes in our test fixtures.
        """
        try:
            # Move to start and lock one byte; prefer non-blocking then fallback
            fileobj.seek(0)
            try:
                msvcrt.locking(fileobj.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                msvcrt.locking(fileobj.fileno(), msvcrt.LK_LOCK, 1)
            yield
        finally:
            try:
                fileobj.seek(0)
                msvcrt.locking(fileobj.fileno(), msvcrt.LK_UNLCK, 1)
            except Exception:
                # Best-effort unlock; tests should not crash because of unlock
                pass

else:
    import fcntl

    @contextmanager
    def exclusive_lock(fileobj):
        """Acquire an exclusive lock on *fileobj* (POSIX).

        Uses ``fcntl.flock`` to block until the lock is acquired, and
        always attempts to release the lock in a finally block.
        """
        try:
            fcntl.flock(fileobj.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(fileobj.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
