import importlib.util
import os
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


def test_pnl_formatters_show_sub_dollar_precision():
    watch = _load_watch_module()

    assert watch._format_signed_dollars(0.001) == "+0.0010"
    assert watch._format_pnl_short(0.001) == "+0.0010"


def test_watch_state_ingests_canonical_pnl_total():
    watch = _load_watch_module()
    state = watch.WatchState()

    state.update({"t": 1, "pnl_total": 0.001, "venue_status": []})

    assert list(state.pnl_history) == [0.001]


def test_balance_pnl_state_loads_completed_run_comparison(tmp_path):
    watch = _load_watch_module()
    comparison_path = tmp_path / "balance_snapshot_comparison.json"
    comparison_path.write_text(
        (
            "{"
            "\"generated_at_utc\":\"2026-05-01T00:00:00Z\","
            "\"venue_count\":5,"
            "\"total\":{"
            "\"pre_usd\":\"317.24306565\","
            "\"post_usd\":\"317.25048978\","
            "\"delta_usd\":\"0.00742413\""
            "}"
            "}\n"
        ),
        encoding="utf-8",
    )
    state = watch.WatchState()

    watch.refresh_balance_pnl_from_run_dir(state, tmp_path, force=True)

    assert state.balance_pnl.status == "available"
    assert state.balance_pnl.delta_usd == 0.00742413
    assert state.balance_pnl.pre_usd == 317.24306565
    assert state.balance_pnl.post_usd == 317.25048978
    assert state.balance_pnl.venue_count == 5
    assert state.balance_pnl.source_path == str(comparison_path)


def test_balance_pnl_state_marks_active_run_pending_post(tmp_path):
    watch = _load_watch_module()
    pre_path = tmp_path / "balance_pre_snapshot.json"
    pre_path.write_text(
        "{\"captured_at_utc\":\"2026-05-01T00:00:00Z\",\"venue_count\":5,\"total_balance_usd\":\"317.24306565\"}\n",
        encoding="utf-8",
    )
    state = watch.WatchState()

    watch.refresh_balance_pnl_from_run_dir(state, tmp_path, force=True)

    assert state.balance_pnl.status == "pending_post"
    assert state.balance_pnl.pre_usd == 317.24306565
    assert state.balance_pnl.venue_count == 5
    assert state.balance_pnl.source_path == str(pre_path)


def test_simple_frame_prefers_balance_pnl_over_telemetry_pnl():
    watch = _load_watch_module()
    if not watch._RICH_AVAILABLE:
        return

    state = watch.WatchState()
    state.update(
        {
            "t": 1,
            "trade_mode": "live",
            "pnl_total": 999.0,
            "venue_status": [],
        }
    )
    state.balance_pnl = watch.BalancePnlState(status="available", delta_usd=0.00742413)

    from rich.console import Console

    console = Console(record=True, width=140, no_color=True)
    console.print(
        watch.render_frame_simple(
            state,
            50,
            term_width=140,
            term_height=20,
        )
    )
    text = console.export_text()

    assert "bPNL +0.0074" in text
    assert "tPNL +999" not in text


def test_net_pos_base_prefers_q_global_over_usd_delta():
    watch = _load_watch_module()

    record = {
        "q_global_tao": 0.0125,
        "net_position_tao": 0.02,
        "dollar_delta_usd": 28.125,
    }

    assert watch._extract_net_pos_base(record) == 0.0125
    assert watch._extract_net_pos_usd(record) == 28.125


def test_frame_key_changes_on_sub_dollar_pnl_only_update():
    watch = _load_watch_module()
    state = watch.WatchState()
    state.update({"t": 1, "pnl_total": 0.001, "venue_status": []})
    key1 = watch._build_frame_key(state, 10.0)

    state.update({"t": 1, "pnl_total": 0.002, "venue_status": []})
    key2 = watch._build_frame_key(state, 10.0)

    assert key1 != key2


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


def test_default_auto_targets_include_source_owner_phase51_root():
    watch = _load_watch_module()

    roots = [root for root, _patterns in watch._AUTO_TARGET_GLOB_ROOTS]

    assert watch._SOURCE_OWNER_PHASE51_ROOT in roots


def test_resolve_latest_considers_source_owner_phase51_artifacts(tmp_path, monkeypatch):
    watch = _load_watch_module()
    tmp_root = tmp_path / "tmp"
    tmp_candidate = tmp_root / "old_run" / "telemetry.jsonl"
    tmp_candidate.parent.mkdir(parents=True)
    tmp_candidate.write_text("{}\n", encoding="utf-8")

    source_owner_root = tmp_path / "source_owner_inbox" / "phase51"
    source_owner_candidate = source_owner_root / "new_run" / "telemetry.jsonl"
    source_owner_candidate.parent.mkdir(parents=True)
    source_owner_candidate.write_text("{}\n", encoding="utf-8")

    os.utime(tmp_candidate, (100.0, 100.0))
    os.utime(source_owner_candidate, (200.0, 200.0))

    monkeypatch.setattr(watch, "_SYSTEMD_TELEMETRY_PATH", tmp_path / "missing_systemd.jsonl")
    monkeypatch.setattr(watch, "_SHADOW_LATEST_PATH", tmp_path / "missing_shadow_latest")
    monkeypatch.setattr(watch, "_SHADOW_LAST_OUTDIR_PATH", tmp_path / "missing_last_outdir.txt")
    monkeypatch.setattr(
        watch,
        "_AUTO_TARGET_GLOB_ROOTS",
        (
            (tmp_root, ("*/telemetry.jsonl",)),
            (source_owner_root, ("*/telemetry.jsonl",)),
        ),
    )
    monkeypatch.setattr(watch, "_iter_running_paraphina_live", lambda: iter(()))

    resolved = watch.resolve_latest_telemetry_path()

    assert resolved == source_owner_candidate.resolve(strict=False)


def test_resolve_current_run_fails_closed_without_active_runner(tmp_path, monkeypatch):
    watch = _load_watch_module()
    latest_candidate = tmp_path / "latest" / "telemetry.jsonl"
    latest_candidate.parent.mkdir(parents=True)
    latest_candidate.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(watch, "_CURRENT_RUN_POINTER_PATH", tmp_path / "missing_pointer.json")
    monkeypatch.setattr(watch, "_CURRENT_RUNS_DIR", tmp_path / "missing_registry")
    monkeypatch.setattr(
        watch,
        "_AUTO_TARGET_GLOB_ROOTS",
        ((latest_candidate.parent.parent, ("*/telemetry.jsonl",)),),
    )
    monkeypatch.setattr(watch, "_iter_running_paraphina_live", lambda: iter(()))

    try:
        watch.resolve_current_run_telemetry_path()
    except FileNotFoundError as exc:
        assert "no active current run found" in str(exc)
    else:
        raise AssertionError("expected current-run resolution to fail closed")


def test_live_record_without_active_runner_displays_snapshot_mode():
    watch = _load_watch_module()

    label, style = watch._mode_display_token({"trade_mode": "live"}, None)

    assert label == "SNAP"
    assert style == "bold yellow"


def test_runner_status_label_reports_no_active_current_run():
    watch = _load_watch_module()
    state = watch.WatchState()
    state.runner_status = watch.RunnerStatus(
        state="no_active",
        alive=False,
        status_path="current-run registry/process",
    )

    label, style = watch.runner_status_label(state)

    assert label == "no active run"
    assert style == "bold red"


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
