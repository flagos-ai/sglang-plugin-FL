from sglang_fl.dispatch import logger_manager


def test_log_exec_without_path_keeps_no_state(monkeypatch, tmp_path):
    monkeypatch.delenv("SGLANG_FL_DISPATCH_LOG", raising=False)
    monkeypatch.setattr(logger_manager, "_exec_counts", {})

    logger_manager.log_exec("mhc_pre", "default.flagos")

    assert logger_manager._exec_counts == {}
    assert list(tmp_path.iterdir()) == []


def test_log_exec_samples_counts_at_powers_of_two(monkeypatch, tmp_path):
    path = tmp_path / "dispatch.log"
    monkeypatch.setenv("SGLANG_FL_DISPATCH_LOG", str(path))
    monkeypatch.setattr(logger_manager, "_exec_counts", {})

    for _ in range(5):
        logger_manager.log_exec("mhc_pre", "default.flagos")

    lines = path.read_text().splitlines()
    assert lines.count("[EXEC] mhc_pre → default.flagos") == 1
    assert [line for line in lines if line.startswith("[STAT]")] == [
        "[STAT] mhc_pre → default.flagos count=1",
        "[STAT] mhc_pre → default.flagos count=2",
        "[STAT] mhc_pre → default.flagos count=4",
    ]


def test_reset_exec_state_after_fork_clears_counts(monkeypatch):
    monkeypatch.setattr(logger_manager, "_exec_counts", {("op", "backend"): 3})

    logger_manager._reset_exec_state_after_fork()

    assert logger_manager._exec_counts == {}
