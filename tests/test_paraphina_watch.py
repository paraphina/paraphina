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
