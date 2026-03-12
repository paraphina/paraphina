import importlib.util
import sys
from pathlib import Path


def _load_watch_module():
    module_path = Path(__file__).resolve().parents[1] / "tools" / "paraphina_watch.py"
    spec = importlib.util.spec_from_file_location("paraphina_watch_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_auto_target_candidates_include_systemd_and_globs(tmp_path, monkeypatch):
    watch = _load_watch_module()
    systemd_telemetry = tmp_path / "var" / "lib" / "paraphina" / "out" / "telemetry.jsonl"
    systemd_telemetry.parent.mkdir(parents=True)
    systemd_telemetry.write_text("{}\n", encoding="utf-8")

    tmp_root = tmp_path / "tmp"
    tmp_run = tmp_root / "shadow_run" / "telemetry.jsonl"
    tmp_run.parent.mkdir(parents=True)
    tmp_run.write_text("{}\n", encoding="utf-8")

    runs_root = tmp_path / "runs"
    repo_run = runs_root / "manual" / "telemetry.jsonl"
    repo_run.parent.mkdir(parents=True)
    repo_run.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(watch, "_SYSTEMD_TELEMETRY_PATH", systemd_telemetry)
    monkeypatch.setattr(
        watch,
        "_AUTO_TARGET_GLOB_ROOTS",
        (
            (tmp_root, ("*/telemetry.jsonl",)),
            (runs_root, ("*/telemetry.jsonl",)),
        ),
    )

    candidates = watch._auto_target_candidates()

    assert systemd_telemetry in candidates
    assert tmp_run in candidates
    assert repo_run in candidates


def test_load_runner_status_falls_back_to_running_process_probe(tmp_path, monkeypatch):
    watch = _load_watch_module()
    telemetry_path = tmp_path / "var" / "lib" / "paraphina" / "out" / "telemetry.jsonl"
    telemetry_path.parent.mkdir(parents=True)
    telemetry_path.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(watch, "_SYSTEMD_TELEMETRY_PATH", telemetry_path)
    monkeypatch.setattr(
        watch,
        "_iter_running_paraphina_live",
        lambda: iter([(42627, ["/opt/paraphina/paraphina_live"])]),
    )

    status = watch.load_runner_status(telemetry_path)

    assert status is not None
    assert status.alive is True
    assert status.runner_pid == 42627
    assert status.state == "running"


def test_resolve_latest_prefers_single_running_process_target(tmp_path, monkeypatch):
    watch = _load_watch_module()
    systemd_telemetry = tmp_path / "var" / "lib" / "paraphina" / "out" / "telemetry.jsonl"
    systemd_telemetry.parent.mkdir(parents=True)
    systemd_telemetry.write_text("{}\n", encoding="utf-8")

    tmp_root = tmp_path / "tmp"
    fresher_tmp = tmp_root / "candidate" / "telemetry.jsonl"
    fresher_tmp.parent.mkdir(parents=True)
    fresher_tmp.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(watch, "_SYSTEMD_TELEMETRY_PATH", systemd_telemetry)
    monkeypatch.setattr(
        watch,
        "_AUTO_TARGET_GLOB_ROOTS",
        ((tmp_root, ("*/telemetry.jsonl",)),),
    )
    monkeypatch.setattr(
        watch,
        "_iter_running_paraphina_live",
        lambda: iter([(42627, ["/opt/paraphina/paraphina_live"])]),
    )

    resolved = watch.resolve_latest_telemetry_path()

    assert resolved == systemd_telemetry.resolve(strict=False)


def test_resolve_current_run_prefers_valid_pointer_record(tmp_path, monkeypatch):
    watch = _load_watch_module()
    telemetry_path = tmp_path / "live_out" / "telemetry.jsonl"
    telemetry_path.parent.mkdir(parents=True)

    pointer_path = tmp_path / "paraphina_current_run.json"
    pointer_path.write_text(
        (
            "{"
            f"\"pid\":4242,"
            f"\"started_at\":\"1234567890\","
            f"\"started_at_unix_ms\":1234567890,"
            f"\"trade_mode\":\"live\","
            f"\"telemetry_path\":\"{telemetry_path}\""
            "}\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(watch, "_CURRENT_RUN_POINTER_PATH", pointer_path)
    monkeypatch.setattr(watch, "_CURRENT_RUNS_DIR", tmp_path / "registry")
    monkeypatch.setattr(watch, "_pid_is_paraphina_runner", lambda pid: pid == 4242)

    resolved = watch.resolve_current_run_telemetry_path()

    assert resolved == telemetry_path.resolve(strict=False)


def test_resolve_current_run_repairs_stale_pointer_from_registry(tmp_path, monkeypatch):
    watch = _load_watch_module()
    stale_telemetry = tmp_path / "stale" / "telemetry.jsonl"
    active_telemetry = tmp_path / "active" / "telemetry.jsonl"
    active_telemetry.parent.mkdir(parents=True)

    pointer_path = tmp_path / "paraphina_current_run.json"
    pointer_path.write_text(
        (
            "{"
            f"\"pid\":1111,"
            f"\"started_at\":\"1111\","
            f"\"started_at_unix_ms\":1111,"
            f"\"trade_mode\":\"shadow\","
            f"\"telemetry_path\":\"{stale_telemetry}\""
            "}\n"
        ),
        encoding="utf-8",
    )

    registry_dir = tmp_path / "paraphina_current_runs"
    registry_dir.mkdir()
    (registry_dir / "2222.json").write_text(
        (
            "{"
            f"\"pid\":2222,"
            f"\"started_at\":\"2222\","
            f"\"started_at_unix_ms\":2222,"
            f"\"trade_mode\":\"live\","
            f"\"telemetry_path\":\"{active_telemetry}\""
            "}\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(watch, "_CURRENT_RUN_POINTER_PATH", pointer_path)
    monkeypatch.setattr(watch, "_CURRENT_RUNS_DIR", registry_dir)
    monkeypatch.setattr(watch, "_pid_is_paraphina_runner", lambda pid: pid == 2222)

    resolved = watch.resolve_current_run_telemetry_path()

    assert resolved == active_telemetry.resolve(strict=False)


def test_resolve_current_run_uses_newest_active_registry_record(tmp_path, monkeypatch):
    watch = _load_watch_module()
    older_telemetry = tmp_path / "older" / "telemetry.jsonl"
    newer_telemetry = tmp_path / "newer" / "telemetry.jsonl"

    registry_dir = tmp_path / "paraphina_current_runs"
    registry_dir.mkdir()
    (registry_dir / "1010.json").write_text(
        (
            "{"
            f"\"pid\":1010,"
            f"\"started_at\":\"1010\","
            f"\"started_at_unix_ms\":1010,"
            f"\"trade_mode\":\"live\","
            f"\"telemetry_path\":\"{older_telemetry}\""
            "}\n"
        ),
        encoding="utf-8",
    )
    (registry_dir / "2020.json").write_text(
        (
            "{"
            f"\"pid\":2020,"
            f"\"started_at\":\"2020\","
            f"\"started_at_unix_ms\":2020,"
            f"\"trade_mode\":\"shadow\","
            f"\"telemetry_path\":\"{newer_telemetry}\""
            "}\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(watch, "_CURRENT_RUN_POINTER_PATH", tmp_path / "missing_pointer.json")
    monkeypatch.setattr(watch, "_CURRENT_RUNS_DIR", registry_dir)
    monkeypatch.setattr(watch, "_pid_is_paraphina_runner", lambda pid: pid in {1010, 2020})

    resolved = watch.resolve_current_run_telemetry_path()

    assert resolved == newer_telemetry.resolve(strict=False)


def test_resolve_current_shadow_uses_running_global_state(tmp_path, monkeypatch):
    watch = _load_watch_module()
    outdir = tmp_path / "shadow_run"
    telemetry_path = outdir / "telemetry.jsonl"
    telemetry_path.parent.mkdir(parents=True)
    telemetry_path.write_text("{}\n", encoding="utf-8")

    state_path = tmp_path / "paraphina_shadow_runner.state"
    state_path.write_text(
        "\n".join(
            (
                "state=running",
                "runner_pid=4242",
                f"outdir={outdir}",
                f"telemetry_path={telemetry_path}",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(watch, "_SHADOW_STATE_PATH", state_path)
    monkeypatch.setattr(watch, "_pid_is_paraphina_runner", lambda pid: pid == 4242)

    resolved = watch.resolve_current_shadow_telemetry_path()

    assert resolved == telemetry_path.resolve(strict=False)


def test_resolve_current_shadow_fails_closed_on_stale_global_state(tmp_path, monkeypatch):
    watch = _load_watch_module()
    outdir = tmp_path / "shadow_run"
    telemetry_path = outdir / "telemetry.jsonl"
    telemetry_path.parent.mkdir(parents=True)
    telemetry_path.write_text("{}\n", encoding="utf-8")

    state_path = tmp_path / "paraphina_shadow_runner.state"
    state_path.write_text(
        "\n".join(
            (
                "state=running",
                "runner_pid=4242",
                f"outdir={outdir}",
                f"telemetry_path={telemetry_path}",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    last_outdir_path = tmp_path / "paraphina_last_outdir.txt"
    last_outdir_path.write_text(f"{outdir}\n", encoding="utf-8")

    monkeypatch.setattr(watch, "_SHADOW_STATE_PATH", state_path)
    monkeypatch.setattr(watch, "_SHADOW_LAST_OUTDIR_PATH", last_outdir_path)
    monkeypatch.setattr(watch, "_pid_is_paraphina_runner", lambda _pid: False)
    monkeypatch.setattr(
        watch,
        "load_runner_status",
        lambda _telemetry_path: watch.RunnerStatus(
            state="running",
            runner_pid=9999,
            alive=True,
        ),
    )

    try:
        watch.resolve_current_shadow_telemetry_path()
    except FileNotFoundError as exc:
        assert "pid 4242 is not active" in str(exc)
    else:
        raise AssertionError("expected current shadow resolution to fail closed")


def test_resolve_current_shadow_falls_back_to_last_outdir(tmp_path, monkeypatch):
    watch = _load_watch_module()
    outdir = tmp_path / "shadow_run"
    telemetry_path = outdir / "telemetry.jsonl"
    telemetry_path.parent.mkdir(parents=True)
    telemetry_path.write_text("{}\n", encoding="utf-8")

    state_path = tmp_path / "missing_shadow_runner.state"
    last_outdir_path = tmp_path / "paraphina_last_outdir.txt"
    last_outdir_path.write_text(f"{outdir}\n", encoding="utf-8")

    monkeypatch.setattr(watch, "_SHADOW_STATE_PATH", state_path)
    monkeypatch.setattr(watch, "_SHADOW_LAST_OUTDIR_PATH", last_outdir_path)
    monkeypatch.setattr(
        watch,
        "load_runner_status",
        lambda candidate: watch.RunnerStatus(
            state="running",
            runner_pid=5150,
            alive=True,
            outdir=str(candidate.parent),
        ),
    )

    resolved = watch.resolve_current_shadow_telemetry_path()

    assert resolved == telemetry_path.resolve(strict=False)
