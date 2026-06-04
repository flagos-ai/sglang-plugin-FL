# Tests for OpManager fork safety with real os.fork().

import os
import sys
import signal

import pytest

from sglang_fl.dispatch.types import BackendImplKind, BackendPriority, OpImpl
from sglang_fl.dispatch.registry import OpRegistry
from sglang_fl.dispatch.manager import OpManager


@pytest.fixture
def manager_with_cache():
    """Create a manager with pre-populated cache."""
    registry = OpRegistry()
    manager = OpManager(registry=registry)
    manager._state.initialized = True
    manager._state.init_pid = os.getpid()

    def impl_fn(*a, **kw):
        return "parent_result"

    registry.register_impl(OpImpl(
        op_name="test_op",
        impl_id="default.flagos",
        kind=BackendImplKind.DEFAULT,
        fn=impl_fn,
        priority=BackendPriority.DEFAULT,
    ))

    # Warm the cache
    fn = manager.resolve("test_op")
    assert fn() == "parent_result"
    assert len(manager._dispatch_cache) > 0

    return manager


class TestForkSafety:
    """Test that OpManager correctly resets state after fork."""

    @pytest.mark.skipif(
        sys.platform == "win32", reason="fork not available on Windows"
    )
    def test_fork_clears_cache(self, manager_with_cache):
        """After fork, child process should have empty cache."""
        manager = manager_with_cache

        r_fd, w_fd = os.pipe()
        pid = os.fork()

        if pid == 0:
            # Child process
            os.close(r_fd)
            try:
                # After fork, _reset_after_fork should have been called
                cache_empty = len(manager._dispatch_cache) == 0
                initialized = manager._state.initialized
                pid_reset = manager._state.init_pid == -1

                result = f"{int(cache_empty)},{int(not initialized)},{int(pid_reset)}"
                os.write(w_fd, result.encode())
            except Exception as e:
                os.write(w_fd, f"ERROR:{e}".encode())
            finally:
                os.close(w_fd)
                os._exit(0)
        else:
            # Parent process
            os.close(w_fd)
            _, status = os.waitpid(pid, 0)
            data = os.read(r_fd, 1024).decode()
            os.close(r_fd)

            assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0, (
                f"Child exited abnormally: {status}"
            )
            assert not data.startswith("ERROR"), f"Child error: {data}"
            cache_empty, not_initialized, pid_reset = data.split(",")
            assert cache_empty == "1", "Cache should be empty after fork"
            assert not_initialized == "1", "Should be uninitialized after fork"
            assert pid_reset == "1", "init_pid should be -1 after fork"

    @pytest.mark.skipif(
        sys.platform == "win32", reason="fork not available on Windows"
    )
    def test_fork_child_reinitializes_on_resolve(self, manager_with_cache):
        """After fork, child should re-initialize when resolve is called."""
        manager = manager_with_cache

        r_fd, w_fd = os.pipe()
        pid = os.fork()

        if pid == 0:
            os.close(r_fd)
            try:
                # After fork, state is reset. Calling resolve should trigger
                # ensure_initialized which re-registers builtins.
                # But since we manually set up the registry, we just verify
                # the PID tracking works.
                old_pid = manager._state.init_pid
                manager.ensure_initialized()
                new_pid = manager._state.init_pid

                # After ensure_initialized, pid should be current child pid
                result = f"{old_pid},{new_pid},{os.getpid()}"
                os.write(w_fd, result.encode())
            except Exception as e:
                os.write(w_fd, f"ERROR:{e}".encode())
            finally:
                os.close(w_fd)
                os._exit(0)
        else:
            os.close(w_fd)
            _, status = os.waitpid(pid, 0)
            data = os.read(r_fd, 1024).decode()
            os.close(r_fd)

            assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
            assert not data.startswith("ERROR"), f"Child error: {data}"
            old_pid, new_pid, child_pid = data.split(",")
            assert old_pid == "-1", "After fork, init_pid should be -1"
            assert new_pid == child_pid, "After ensure_initialized, pid should match child"

    @pytest.mark.skipif(
        sys.platform == "win32", reason="fork not available on Windows"
    )
    def test_fork_parent_unaffected(self, manager_with_cache):
        """Fork should not affect parent process state."""
        manager = manager_with_cache
        cache_before = len(manager._dispatch_cache)
        epoch_before = manager._state.policy_epoch

        pid = os.fork()
        if pid == 0:
            os._exit(0)
        else:
            os.waitpid(pid, 0)

        # Parent state should be unchanged
        assert len(manager._dispatch_cache) == cache_before
        assert manager._state.policy_epoch == epoch_before
        assert manager._state.initialized is True
        assert manager._state.init_pid == os.getpid()

    @pytest.mark.skipif(
        sys.platform == "win32", reason="fork not available on Windows"
    )
    def test_fork_epoch_increments(self, manager_with_cache):
        """After fork, policy epoch should increment in child."""
        manager = manager_with_cache
        parent_epoch = manager._state.policy_epoch

        r_fd, w_fd = os.pipe()
        pid = os.fork()

        if pid == 0:
            os.close(r_fd)
            child_epoch = manager._state.policy_epoch
            os.write(w_fd, str(child_epoch).encode())
            os.close(w_fd)
            os._exit(0)
        else:
            os.close(w_fd)
            os.waitpid(pid, 0)
            data = os.read(r_fd, 1024).decode()
            os.close(r_fd)

            child_epoch = int(data)
            assert child_epoch == parent_epoch + 1, (
                f"Child epoch ({child_epoch}) should be parent ({parent_epoch}) + 1"
            )
