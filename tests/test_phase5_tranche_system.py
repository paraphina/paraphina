import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import yaml


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "tools" / "phase5_tranche.py"
    spec = importlib.util.spec_from_file_location("phase5_tranche_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_yaml(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _minimal_queue():
    return {
        "schema_version": 1,
        "phase": 5,
        "updated_utc": "2026-03-29T00:00:00Z",
        "serialized_mainline": [
            {
                "id": "t_main_1",
                "track": "serialized_mainline",
                "status": "ready",
                "objective": "Main tranche",
                "hypothesis": "Current mainline hypothesis",
                "branch_class": "qualification",
                "hypothesis_blocker_family": "stale_restart",
                "support_gate": "none",
                "progress_credit": "minor",
                "control": {"description": "control"},
                "candidate": {"change_scope": {"files": ["runner.rs"]}},
                "next_if_pass": "t_main_2",
                "next_if_fail": "t_main_2",
                "next_if_fail_when_matched": "t_main_2",
                "matched_fail_routes": {"actual_residual_live_orders": "t_main_3"},
                "automation": {
                    "support_tracks": ["t_support_1"],
                },
            },
            {
                "id": "t_main_2",
                "track": "serialized_mainline",
                "status": "blocked",
                "objective": "Next tranche",
                "hypothesis": "Next hypothesis",
                "control": {"description": "control"},
                "candidate": {"change_scope": {"files": []}},
            },
            {
                "id": "t_main_3",
                "track": "serialized_mainline",
                "status": "blocked",
                "objective": "Matched fail tranche",
                "hypothesis": "Matched fail hypothesis",
                "control": {"description": "control"},
                "candidate": {"change_scope": {"files": []}},
            },
        ],
        "parallel_support_tracks": [
            {
                "id": "t_support_1",
                "track": "parallel_support",
                "status": "ready",
                "objective": "Support work",
                "hypothesis": "Support hypothesis",
                "control": {"description": "control"},
                "candidate": {"change_scope": {"files": ["telemetry.py"]}},
                "automation": {"autorun_policy": "validate_only"},
            }
        ],
    }


def _minimal_control_pack():
    return {
        "schema_version": 1,
        "phase": 5,
        "baseline_id": "phase5_test_baseline",
        "topology": {
            "connectors": ["hyperliquid", "aster", "lighter", "paradex"],
            "fv_disabled_venues": ["lighter"],
            "excluded_venues": ["extended"],
            "roles": {
                "hyperliquid": "primary_fill",
                "aster": "anchor_fv",
                "lighter": "connected_non_fv",
                "paradex": "anchor_fv",
                "extended": "excluded_pending_rescue",
            },
        },
        "execution_defaults": {
            "runtime_binary": "/opt/paraphina/paraphina_live",
            "cleanup_binary": "/tmp/live_cleanup",
            "stage_overlay_target": "/etc/paraphina/stage_overlay.env",
            "stage_overlay_source": "/tmp/stage_overlay_live.env",
            "live_guard_args": ["--pre-restore-cleanup-on-exit"],
            "live_exec_dropin_target": "/etc/systemd/system/paraphina_live.service.d/live_exec_flag.conf",
            "live_exec_dropin_source": "/tmp/live_exec_flag.conf",
            "service": "paraphina_live",
            "guard_script": "/tmp/unattended_live_guard.py",
            "analyzer_script": "/tmp/telemetry_analyzer.py",
            "telemetry_path": "/tmp/telemetry.jsonl",
            "stderr_path": "/tmp/paraphina_live.err",
            "promotion_runs_root": "/tmp/promotion_runs",
        },
        "automation_defaults": {
            "autonomy_mode": "full_auto",
            "subagent_model": "deterministic_handoff",
            "worktree_lifecycle": "ephemeral",
            "artifact_packaging": "full",
            "worktree_root": "/tmp/phase5_worktrees",
            "lane_bundle_root": "phase5/runs/{tranche_id}/lanes",
            "cleanup_policy": "ephemeral_on_verdict",
            "repo_headroom_bytes": 1024,
            "promotion_runs_headroom_bytes": 1024,
            "telemetry_headroom_bytes": 1024,
            "tempdir_headroom_bytes": 1024,
            "autorun_support_default": "manual",
            "max_parallel_support_lanes": 2,
            "support_lane_priority": ["forensics", "blocker_shadow", "topology_audit", "tooling"],
            "support_lane_capacity_gate": True,
            "stage_verdict_contract": [
                "stage_verdict.json",
                "venue_capability_matrix.json",
                "support_summary.json",
            ],
        },
    }


class TestPhase5TrancheSystem(unittest.TestCase):
    def setUp(self):
        self.mod = _load_module()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo_root = Path(self.temp_dir.name)
        Path("/tmp/live_cleanup").write_text("#!/bin/sh\n", encoding="utf-8")
        (self.repo_root / "tools").mkdir(parents=True, exist_ok=True)
        (self.repo_root / "tests").mkdir(parents=True, exist_ok=True)
        (self.repo_root / "docs").mkdir(parents=True, exist_ok=True)
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", _minimal_queue())
        _write_yaml(self.repo_root / "phase5" / "control_pack.yaml", _minimal_control_pack())
        _write_yaml(self.repo_root / "phase5" / "orchestration.yaml", {"schema_version": 1, "sessions": []})
        (self.repo_root / "phase5" / "status.md").write_text("# status\n", encoding="utf-8")
        (self.repo_root / "tools" / "phase5_tranche.py").write_text("# stub\n", encoding="utf-8")
        (self.repo_root / "tools" / "telemetry_analyzer.py").write_text("# stub\n", encoding="utf-8")
        (self.repo_root / "tests" / "test_phase5_tranche_system.py").write_text("# stub\n", encoding="utf-8")
        (self.repo_root / "docs" / "PHASE5_AUTONOMOUS_TRANCHE_SYSTEM.md").write_text("# stub\n", encoding="utf-8")
        subprocess.run(["git", "init"], cwd=self.repo_root, check=True, capture_output=True, text=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=self.repo_root, check=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=self.repo_root, check=True)
        subprocess.run(["git", "add", "."], cwd=self.repo_root, check=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=self.repo_root, check=True, capture_output=True, text=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_compare_rule_value_orders_numeric_strings(self):
        self.assertTrue(self.mod.compare_rule_value("0.00742413", ">=", 0.0))
        self.assertTrue(self.mod.compare_rule_value("-0.01210395", "<", 0.0))
        self.assertFalse(self.mod.compare_rule_value("not-a-number", ">=", 0.0))

    def _state_sync_pass_report(self, tranche_id: str = "t_main_1") -> dict:
        return {
            "schema_version": 1,
            "generated_utc": "2026-04-26T00:00:00Z",
            "repo_root": str(self.repo_root),
            "requested_tranche_id": tranche_id,
            "status": "pass",
            "critical_count": 0,
            "warning_count": 0,
            "tranches": [
                {
                    "tranche_id": tranche_id,
                    "surface_id": "surface-1",
                    "linked_child_ids": ["t_main_2"],
                    "status_summary": "pass",
                }
            ],
            "findings": [],
        }

    def _spawn_lanes_with_preflight(self, tranche_id: str = "t_main_1") -> dict:
        with mock.patch.object(
            self.mod,
            "ensure_shadow_health",
            return_value={"healthy": True, "ready": True, "trade_mode": "shadow"},
        ), mock.patch.object(
            self.mod,
            "ensure_runtime_storage_headroom",
            return_value={
                "repo": {"free_bytes": 10},
                "promotion_runs": {"free_bytes": 10},
                "telemetry": {"free_bytes": 10},
                "tempdir": {"free_bytes": 10},
                "current_runs": {"free_bytes": 10},
            },
        ), mock.patch.object(
            self.mod,
            "audit_state_sync",
            return_value=self._state_sync_pass_report(tranche_id),
        ):
            return self.mod.spawn_lanes(tranche_id, repo_root=self.repo_root)

    def _state_sync_warning_report(self, tranche_id: str = "t_main_1") -> dict:
        report = self._state_sync_pass_report(tranche_id)
        report["status"] = "warn"
        report["warning_count"] = 1
        report["findings"] = [
            {
                "severity": "warn",
                "tranche_id": tranche_id,
                "field": "ROADMAP.md",
                "message": "roadmap warning",
            }
        ]
        return report

    def _state_sync_critical_report(self, tranche_id: str = "t_main_1") -> dict:
        report = self._state_sync_pass_report(tranche_id)
        report["status"] = "fail"
        report["critical_count"] = 1
        report["findings"] = [
            {
                "severity": "fail",
                "tranche_id": tranche_id,
                "field": "ROADMAP.md",
                "message": "roadmap drift",
            }
        ]
        return report

    def _write_coherent_state_sync_layout(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][1]["env_diff"] = {
            "stage_overlay_source": str(self.repo_root / "phase5" / "runs" / "t_main_2" / "stage_overlay_live.env"),
        }
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)
        main_card_path = self.mod.prepare_tranche(repo_root=self.repo_root, tranche_id="t_main_1", mark_in_progress=False)
        child_card_path = self.mod.prepare_tranche(repo_root=self.repo_root, tranche_id="t_main_2", mark_in_progress=False)

        main_card = yaml.safe_load(main_card_path.read_text(encoding="utf-8"))
        child_card = yaml.safe_load(child_card_path.read_text(encoding="utf-8"))

        (self.repo_root / "phase5" / "status.md").write_text(
            "\n".join(
                [
                    "# Phase 5 Status",
                    "",
                    f"- `t_main_1`: `ready` surface_id `{main_card['surface_id']}`",
                    f"- `t_main_2`: `blocked` surface_id `{child_card['surface_id']}`",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        (self.repo_root / "ROADMAP.md").write_text(
            "\n".join(
                [
                    "# Roadmap",
                    "",
                    f"- `t_main_1` surface `{main_card['surface_id']}`",
                    f"- `t_main_2` surface `{child_card['surface_id']}`",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return queue, main_card, child_card

    def test_prepare_tranche_creates_card_and_marks_in_progress(self):
        card_path = self.mod.prepare_tranche(repo_root=self.repo_root)

        self.assertTrue(card_path.exists())
        card = yaml.safe_load(card_path.read_text(encoding="utf-8"))
        self.assertEqual(card["tranche"]["id"], "t_main_1")

        queue = yaml.safe_load((self.repo_root / "phase5" / "queue.yaml").read_text(encoding="utf-8"))
        self.assertEqual(queue["serialized_mainline"][0]["status"], "in_progress")

    def test_record_result_promote_advances_next_tranche(self):
        run_root = self.repo_root / "promotion_runs" / "parent_live"
        run_root.mkdir(parents=True, exist_ok=True)
        summary_path = run_root / "live_segment_summary.json"
        summary_path.write_text("{}", encoding="utf-8")
        state_sync_report = {
            "schema_version": 1,
            "generated_utc": "2026-04-26T00:00:00Z",
            "repo_root": str(self.repo_root),
            "requested_tranche_id": "t_main_1",
            "status": "pass",
            "critical_count": 0,
            "warning_count": 0,
            "tranches": [
                {
                    "tranche_id": "t_main_1",
                    "surface_id": "surface-1",
                    "linked_child_ids": ["t_main_2"],
                    "status_summary": "pass",
                }
            ],
            "findings": [],
        }

        with mock.patch.object(self.mod, "audit_state_sync", return_value=state_sync_report):
            self.mod.record_result("t_main_1", "promote", repo_root=self.repo_root, summary_path=str(summary_path))

        queue = yaml.safe_load((self.repo_root / "phase5" / "queue.yaml").read_text(encoding="utf-8"))
        self.assertEqual(queue["serialized_mainline"][0]["status"], "promoted")
        self.assertEqual(queue["serialized_mainline"][1]["status"], "ready")
        history = queue["serialized_mainline"][0]["history"]
        self.assertEqual(history[-1]["summary_path"], str(summary_path))
        self.assertEqual(history[-1]["requested_decision"], "promote")
        self.assertEqual(history[-1]["decision"], "promote")
        self.assertFalse(history[-1]["state_sync_blocked_promotion"])
        self.assertEqual(history[-1]["state_sync"]["status"], "pass")
        self.assertTrue((run_root / "state_sync_report.json").exists())

        stage_verdict = json.loads((self.repo_root / "phase5" / "runs" / "t_main_1" / "stage_verdict.json").read_text(encoding="utf-8"))
        support_summary = json.loads((self.repo_root / "phase5" / "runs" / "t_main_1" / "support_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(stage_verdict["verdict"], "PROMOTE")
        self.assertEqual(stage_verdict["state_sync"]["status"], "pass")
        self.assertEqual(support_summary["state_sync"]["status"], "pass")

    def test_record_result_promote_holds_on_state_sync_critical_failure(self):
        run_root = self.repo_root / "promotion_runs" / "parent_live_critical"
        run_root.mkdir(parents=True, exist_ok=True)
        summary_path = run_root / "live_segment_summary.json"
        summary_path.write_text("{}", encoding="utf-8")
        state_sync_report = {
            "schema_version": 1,
            "generated_utc": "2026-04-26T00:00:00Z",
            "repo_root": str(self.repo_root),
            "requested_tranche_id": "t_main_1",
            "status": "fail",
            "critical_count": 1,
            "warning_count": 0,
            "tranches": [
                {
                    "tranche_id": "t_main_1",
                    "surface_id": "surface-1",
                    "linked_child_ids": ["t_main_2"],
                    "status_summary": "fail",
                }
            ],
            "findings": [
                {
                    "severity": "fail",
                    "tranche_id": "t_main_1",
                    "field": "ROADMAP.md",
                    "message": "roadmap drift",
                }
            ],
        }

        with mock.patch.object(self.mod, "audit_state_sync", return_value=state_sync_report):
            self.mod.record_result("t_main_1", "promote", repo_root=self.repo_root, summary_path=str(summary_path))

        queue = yaml.safe_load((self.repo_root / "phase5" / "queue.yaml").read_text(encoding="utf-8"))
        self.assertEqual(queue["serialized_mainline"][0]["status"], "hold")
        self.assertEqual(queue["serialized_mainline"][1]["status"], "blocked")
        history = queue["serialized_mainline"][0]["history"][-1]
        self.assertEqual(history["requested_decision"], "promote")
        self.assertEqual(history["decision"], "hold")
        self.assertTrue(history["precondition_failed"])
        self.assertTrue(history["state_sync_blocked_promotion"])
        self.assertFalse(history["child_activation_allowed"])
        self.assertIsNone(history["activated_child"])
        self.assertEqual(history["state_sync"]["critical_count"], 1)
        self.assertEqual(history["state_sync"]["status"], "fail")

        stage_verdict = json.loads((self.repo_root / "phase5" / "runs" / "t_main_1" / "stage_verdict.json").read_text(encoding="utf-8"))
        support_summary = json.loads((self.repo_root / "phase5" / "runs" / "t_main_1" / "support_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(stage_verdict["verdict"], "HOLD")
        self.assertEqual(stage_verdict["state_sync"]["critical_count"], 1)
        self.assertEqual(support_summary["state_sync"]["critical_count"], 1)

    def test_record_result_promote_holds_on_state_sync_warning_only_report(self):
        run_root = self.repo_root / "promotion_runs" / "parent_live_warning"
        run_root.mkdir(parents=True, exist_ok=True)
        summary_path = run_root / "live_segment_summary.json"
        summary_path.write_text("{}", encoding="utf-8")

        with mock.patch.object(self.mod, "audit_state_sync", return_value=self._state_sync_warning_report()):
            self.mod.record_result("t_main_1", "promote", repo_root=self.repo_root, summary_path=str(summary_path))

        queue = yaml.safe_load((self.repo_root / "phase5" / "queue.yaml").read_text(encoding="utf-8"))
        parent = queue["serialized_mainline"][0]
        self.assertEqual(parent["status"], "hold")
        self.assertEqual(queue["serialized_mainline"][1]["status"], "blocked")
        history = parent["history"][-1]
        self.assertEqual(history["requested_decision"], "promote")
        self.assertEqual(history["decision"], "hold")
        self.assertTrue(history["precondition_failed"])
        self.assertTrue(history["state_sync_blocked_promotion"])
        self.assertEqual(history["state_sync"]["warning_count"], 1)

        stage_verdict = json.loads((self.repo_root / "phase5" / "runs" / "t_main_1" / "stage_verdict.json").read_text(encoding="utf-8"))
        self.assertEqual(stage_verdict["verdict"], "HOLD")
        self.assertEqual(stage_verdict["state_sync"]["warning_count"], 1)

    def test_record_result_state_sync_exception_is_fail_closed_and_atomic(self):
        run_root = self.repo_root / "promotion_runs" / "parent_live_state_sync_error"
        run_root.mkdir(parents=True, exist_ok=True)
        summary_path = run_root / "live_segment_summary.json"
        summary_path.write_text("{}", encoding="utf-8")

        with mock.patch.object(self.mod, "audit_state_sync", side_effect=RuntimeError("state-sync exploded")):
            self.mod.record_result("t_main_1", "promote", repo_root=self.repo_root, summary_path=str(summary_path))

        queue = yaml.safe_load((self.repo_root / "phase5" / "queue.yaml").read_text(encoding="utf-8"))
        parent = queue["serialized_mainline"][0]
        self.assertEqual(parent["status"], "hold")
        self.assertEqual(queue["serialized_mainline"][1]["status"], "blocked")
        history = parent["history"][-1]
        self.assertEqual(history["requested_decision"], "promote")
        self.assertEqual(history["decision"], "hold")
        self.assertTrue(history["precondition_failed"])
        self.assertTrue(history["state_sync_blocked_promotion"])
        self.assertEqual(history["state_sync"]["status"], "error")
        self.assertEqual(history["state_sync"]["critical_count"], 1)
        self.assertIn("state-sync exploded", history["state_sync"]["error"])
        self.assertIn("state-sync exploded", history["state_sync_error"])

        status_text = (self.repo_root / "phase5" / "status.md").read_text(encoding="utf-8")
        self.assertIn("`t_main_1`", status_text)
        self.assertIn("`hold`", status_text)
        stage_verdict = json.loads((self.repo_root / "phase5" / "runs" / "t_main_1" / "stage_verdict.json").read_text(encoding="utf-8"))
        self.assertEqual(stage_verdict["verdict"], "HOLD")
        self.assertEqual(stage_verdict["state_sync"]["status"], "error")

    def test_record_result_hold_suppresses_fail_child_on_blocker_mismatch(self):
        self.mod.record_result(
            "t_main_1",
            "hold",
            repo_root=self.repo_root,
            observed_blocker_family="restore_hygiene",
        )

        queue = yaml.safe_load((self.repo_root / "phase5" / "queue.yaml").read_text(encoding="utf-8"))
        self.assertEqual(queue["serialized_mainline"][0]["status"], "hold")
        self.assertEqual(queue["serialized_mainline"][1]["status"], "blocked")
        history = queue["serialized_mainline"][0]["history"][-1]
        self.assertEqual(history["observed_blocker_family"], "restore_hygiene")
        self.assertFalse(history["child_activation_allowed"])
        self.assertIsNone(history["activated_child"])

    def test_record_result_hold_activates_fail_child_on_blocker_match(self):
        self.mod.record_result(
            "t_main_1",
            "hold",
            repo_root=self.repo_root,
            observed_blocker_family="stale_restart",
        )

        queue = yaml.safe_load((self.repo_root / "phase5" / "queue.yaml").read_text(encoding="utf-8"))
        self.assertEqual(queue["serialized_mainline"][1]["status"], "ready")
        history = queue["serialized_mainline"][0]["history"][-1]
        self.assertTrue(history["child_activation_allowed"])
        self.assertEqual(history["activated_child"], "t_main_2")

    def test_record_result_hold_activates_matched_fail_route(self):
        self.mod.record_result(
            "t_main_1",
            "hold",
            repo_root=self.repo_root,
            observed_blocker_family="actual_residual_live_orders",
        )

        queue = yaml.safe_load((self.repo_root / "phase5" / "queue.yaml").read_text(encoding="utf-8"))
        self.assertEqual(queue["serialized_mainline"][2]["status"], "ready")
        history = queue["serialized_mainline"][0]["history"][-1]
        self.assertTrue(history["child_activation_allowed"])
        self.assertEqual(history["activated_child"], "t_main_3")

    def test_record_result_clean_hold_autoruns_selected_child_support_gate(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][1]["support_gate"] = "shadow_smoke_10m"
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        run_root = self.repo_root / "promotion_runs" / "parent_live"
        run_root.mkdir(parents=True, exist_ok=True)
        summary_path = run_root / "live_segment_summary.json"
        summary_path.write_text("{}", encoding="utf-8")
        (run_root / "autoscore_bundle.json").write_text(
            json.dumps(
                {
                    "clean": {"passed": True},
                    "mechanism": {"passed": True},
                    "promotion": {"passed": False},
                    "suggested_action": "hold",
                }
            ),
            encoding="utf-8",
        )
        (run_root / "live_closeout_bundle.json").write_text(
            json.dumps(
                {
                    "summary_exists": True,
                    "report_exists": True,
                    "metrics_exists": True,
                    "guard_intervened": False,
                    "guard_window_completed": True,
                    "pre_restore_venue_audit_clean": True,
                    "post_rollback_venue_audit_clean": True,
                    "healthy_post": True,
                    "ready_post": True,
                    "kill_events_present_post": False,
                    "trade_mode_post": "shadow",
                    "systemd_active_state_post": "active",
                    "systemd_sub_state_post": "running",
                    "systemd_nrestarts_post": "0",
                    "guard_exit_code": 0,
                }
            ),
            encoding="utf-8",
        )

        with mock.patch.object(self.mod, "ensure_shadow_health", return_value={"healthy": True, "ready": True, "trade_mode": "shadow"}), \
             mock.patch.object(self.mod, "prepare_tranche", return_value=self.repo_root / "phase5" / "runs" / "t_main_2" / "tranche_card.yaml") as prepare_mock, \
             mock.patch.object(self.mod, "run_shadow_smoke", return_value=self.repo_root / "promotion_runs" / "child_shadow") as shadow_mock:
            self.mod.record_result(
                "t_main_1",
                "hold",
                repo_root=self.repo_root,
                summary_path=str(summary_path),
                observed_blocker_family="stale_restart",
            )

        prepare_mock.assert_called_once_with(repo_root=self.repo_root, tranche_id="t_main_2", mark_in_progress=True)
        shadow_mock.assert_called_once_with("t_main_2", 600, self.repo_root)
        queue_after = yaml.safe_load((self.repo_root / "phase5" / "queue.yaml").read_text(encoding="utf-8"))
        history = queue_after["serialized_mainline"][0]["history"][-1]
        self.assertEqual(history["selected_child_support_gate"]["status"], "pass")
        self.assertEqual(history["selected_child_support_gate"]["child_tranche_id"], "t_main_2")

    def test_record_result_skips_child_support_gate_for_unclean_hold(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][1]["support_gate"] = "shadow_smoke_10m"
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        run_root = self.repo_root / "promotion_runs" / "parent_live"
        run_root.mkdir(parents=True, exist_ok=True)
        summary_path = run_root / "live_segment_summary.json"
        summary_path.write_text("{}", encoding="utf-8")
        (run_root / "autoscore_bundle.json").write_text(
            json.dumps(
                {
                    "clean": {"passed": False},
                    "mechanism": {"passed": True},
                    "promotion": {"passed": False},
                    "suggested_action": "hold",
                }
            ),
            encoding="utf-8",
        )
        (run_root / "live_closeout_bundle.json").write_text(
            json.dumps(
                {
                    "guard_intervened": True,
                    "post_rollback_venue_audit_clean": True,
                    "trade_mode_post": "shadow",
                }
            ),
            encoding="utf-8",
        )

        with mock.patch.object(self.mod, "prepare_tranche") as prepare_mock, \
             mock.patch.object(self.mod, "run_shadow_smoke") as shadow_mock:
            self.mod.record_result(
                "t_main_1",
                "hold",
                repo_root=self.repo_root,
                summary_path=str(summary_path),
                observed_blocker_family="stale_restart",
            )

        prepare_mock.assert_not_called()
        shadow_mock.assert_not_called()
        queue_after = yaml.safe_load((self.repo_root / "phase5" / "queue.yaml").read_text(encoding="utf-8"))
        history = queue_after["serialized_mainline"][0]["history"][-1]
        self.assertEqual(history["selected_child_support_gate"]["status"], "skipped")
        self.assertEqual(history["selected_child_support_gate"]["reason"], "parent_hold_not_clean")

    def test_record_result_hold_prefers_matched_fail_route_over_hypothesis_match(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["hypothesis_blocker_family"] = "microstructure_underconversion"
        queue["serialized_mainline"][0]["matched_fail_routes"] = {
            "microstructure_underconversion": "t_main_3"
        }
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        self.mod.record_result(
            "t_main_1",
            "hold",
            repo_root=self.repo_root,
            observed_blocker_family="microstructure_underconversion",
        )

        queue = yaml.safe_load((self.repo_root / "phase5" / "queue.yaml").read_text(encoding="utf-8"))
        self.assertEqual(queue["serialized_mainline"][2]["status"], "ready")
        history = queue["serialized_mainline"][0]["history"][-1]
        self.assertTrue(history["child_activation_allowed"])
        self.assertEqual(history["activated_child"], "t_main_3")

    def test_record_result_hold_suppresses_fail_child_on_precondition_failure(self):
        self.mod.record_result(
            "t_main_1",
            "hold",
            repo_root=self.repo_root,
            observed_blocker_family="stale_restart",
            precondition_failed=True,
        )

        queue = yaml.safe_load((self.repo_root / "phase5" / "queue.yaml").read_text(encoding="utf-8"))
        self.assertEqual(queue["serialized_mainline"][1]["status"], "blocked")
        history = queue["serialized_mainline"][0]["history"][-1]
        self.assertTrue(history["precondition_failed"])
        self.assertFalse(history["child_activation_allowed"])
        self.assertIsNone(history["activated_child"])

    def test_build_lane_specs_prepares_multiple_fail_children(self):
        queue = _minimal_queue()
        control_pack = _minimal_control_pack()

        specs = self.mod.build_lane_specs(queue, queue["serialized_mainline"][0], control_pack, self.repo_root)

        fail_specs = [spec for spec in specs if spec["kind"] == "child_prep" and spec["decision_preview"] == "hold"]
        child_ids = {spec["child_tranche_id"] for spec in fail_specs}
        self.assertEqual(child_ids, {"t_main_2", "t_main_3"})

    def test_validate_queue_accepts_extended_soak_defect_routes(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["matched_fail_routes"] = {
            "transport_gap_watchdog": "t_main_2",
            "no_data_transport_gap": "t_main_2",
            "data_seen_no_publish": "t_main_3",
            "runner_freeze_apply_gap": "t_main_2",
            "future_timestamp_deferral": "t_main_3",
            "paradex_underfill_with_interactive_profile": "t_main_2",
        }

        self.mod.validate_queue(queue)

    def test_validate_queue_accepts_manual_gate_marker(self):
        queue = _minimal_queue()
        tranche = queue["serialized_mainline"][0]
        tranche["promotion_gate"] = {"required": ["clean"]}
        tranche["manual_gate_required"] = True
        self.mod.validate_queue(queue)

    def test_validate_queue_accepts_mechanism_gate_mode_and_local_hygiene_child(self):
        queue = _minimal_queue()
        tranche = queue["serialized_mainline"][0]
        tranche["mechanism_gate_mode"] = "live_5m"
        tranche["mechanism_fail_blocker_family"] = "paradex_private_order_truth_gap"
        tranche["clean_final_hold_blocker_family"] = "microstructure_underconversion"
        tranche["surface_local_restore_hygiene_child"] = "t_main_3"
        self.mod.validate_queue(queue)

    def test_audit_gate_contract_passes_complete_promotion_contract(self):
        queue = _minimal_queue()
        tranche = queue["serialized_mainline"][0]
        tranche["support_gate"] = "shadow_smoke_10m"
        tranche["promotion_gate"] = {"required": ["clean final rung", "mechanism evidence"]}
        tranche["rollback_criteria"] = ["any kill event"]
        tranche["gate_contract"] = {
            "required_artifacts": [
                "closeout",
                "metrics",
                "autoscore",
                "direct_venue_audit",
                "cashflow",
                "state_sync",
                "balance_snapshot",
            ]
        }
        tranche["capital_budget"] = {"max_unexplained_equity_drift_usd": 0.5}
        tranche["automation"]["rung_plan"] = [
            {"duration_sec": 300, "continue_on": "clean"},
            {"duration_sec": 1200, "continue_on": "mechanism"},
            {"duration_sec": 7200, "continue_on": "promotion"},
        ]
        tranche["automation"]["autoscore"] = {
            "promotion": [
                {
                    "source": "metrics",
                    "path": "fills.total_count",
                    "op": ">",
                    "value": 0,
                    "severity": "fail",
                }
            ]
        }

        report = self.mod.audit_gate_contract(queue, _minimal_control_pack(), self.repo_root, "t_main_1")

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["critical_count"], 0)
        self.assertFalse(
            any("unknown required artifact" in finding["message"] for finding in report["findings"])
        )
        self.assertEqual(report["tranches"][0]["final_rung_sec"], 7200)

    def test_audit_gate_contract_fails_unroutable_latest_observed_blocker(self):
        queue = _minimal_queue()
        tranche = queue["serialized_mainline"][0]
        tranche["support_gate"] = "shadow_smoke_10m"
        tranche["promotion_gate"] = {"required": ["clean final rung", "mechanism evidence"]}
        tranche["rollback_criteria"] = ["any kill event"]
        tranche["gate_contract"] = {
            "required_artifacts": [
                "closeout",
                "metrics",
                "autoscore",
                "direct_venue_audit",
                "cashflow",
                "state_sync",
                "balance_snapshot",
            ]
        }
        tranche["capital_budget"] = {"max_unexplained_equity_drift_usd": 0.5}
        tranche["automation"]["rung_plan"] = [
            {"duration_sec": 300, "continue_on": "clean"},
            {"duration_sec": 1200, "continue_on": "clean"},
            {"duration_sec": 7200, "continue_on": "promotion"},
        ]
        tranche["automation"]["autoscore"] = {
            "promotion": [
                {
                    "source": "metrics",
                    "path": "fills.total_count",
                    "op": ">",
                    "value": 0,
                    "severity": "fail",
                }
            ]
        }
        tranche["history"] = [
            {
                "decision": "hold",
                "precondition_failed": False,
                "observed_blocker_family": "extended_pre_kill_degraded_rebootstrap_alignment_gap",
            }
        ]

        report = self.mod.audit_gate_contract(queue, _minimal_control_pack(), self.repo_root, "t_main_1")

        self.assertEqual(report["status"], "fail")
        self.assertIn(
            "matched_fail_routes",
            {finding["field"] for finding in report["findings"] if finding["severity"] == "fail"},
        )

        tranche["matched_fail_routes"][
            "extended_pre_kill_degraded_rebootstrap_alignment_gap"
        ] = "t_main_2"
        report_after_route = self.mod.audit_gate_contract(
            queue,
            _minimal_control_pack(),
            self.repo_root,
            "t_main_1",
        )

        self.assertEqual(report_after_route["status"], "pass")

    def test_audit_gate_contract_detects_short_promotion_rung_and_missing_promotion_rules(self):
        queue = _minimal_queue()
        tranche = queue["serialized_mainline"][0]
        tranche["support_gate"] = "shadow_smoke_10m"
        tranche["promotion_gate"] = {"required": ["clean final rung"]}
        tranche["rollback_criteria"] = ["any kill event"]
        tranche["automation"]["rung_plan"] = [
            {"duration_sec": 300, "continue_on": "clean"},
            {"duration_sec": 1200, "continue_on": "clean"},
            {"duration_sec": 3600, "continue_on": "promotion"},
        ]

        report = self.mod.audit_gate_contract(queue, _minimal_control_pack(), self.repo_root, "t_main_1")

        self.assertEqual(report["status"], "fail")
        fields = {finding["field"] for finding in report["findings"] if finding["severity"] == "fail"}
        self.assertIn("automation.rung_plan", fields)
        self.assertIn("automation.autoscore.promotion", fields)

    def test_audit_gate_contract_detects_missing_rollback_criteria_and_artifact_gaps(self):
        queue = _minimal_queue()
        tranche = queue["serialized_mainline"][0]
        tranche["support_gate"] = "shadow_smoke_10m"
        tranche["promotion_gate"] = {"required": ["clean final rung"]}
        tranche["rollback_criteria"] = []
        tranche["gate_contract"] = {"required_artifacts": ["closeout"]}
        tranche["automation"]["rung_plan"] = [
            {"duration_sec": 300, "continue_on": "clean"},
            {"duration_sec": 7200, "continue_on": "promotion"},
        ]
        tranche["automation"]["autoscore"] = {
            "promotion": [
                {
                    "source": "metrics",
                    "path": "fills.total_count",
                    "op": ">",
                    "value": 0,
                    "severity": "fail",
                }
            ]
        }

        report = self.mod.audit_gate_contract(queue, _minimal_control_pack(), self.repo_root, "t_main_1")

        fail_fields = {finding["field"] for finding in report["findings"] if finding["severity"] == "fail"}
        warn_fields = {finding["field"] for finding in report["findings"] if finding["severity"] == "warn"}
        self.assertIn("rollback_criteria", fail_fields)
        self.assertIn("gate_contract.required_artifacts", warn_fields)

    def test_audit_gate_contract_detects_unsupported_support_gate(self):
        queue = _minimal_queue()
        tranche = queue["serialized_mainline"][0]
        tranche["support_gate"] = "manual_review"
        tranche["promotion_gate"] = {"required": ["clean final rung"]}
        tranche["rollback_criteria"] = ["any kill event"]
        tranche["automation"]["rung_plan"] = [
            {"duration_sec": 300, "continue_on": "clean"},
            {"duration_sec": 7200, "continue_on": "promotion"},
        ]
        tranche["automation"]["autoscore"] = {
            "promotion": [
                {
                    "source": "metrics",
                    "path": "fills.total_count",
                    "op": ">",
                    "value": 0,
                    "severity": "fail",
                }
            ]
        }

        report = self.mod.audit_gate_contract(queue, _minimal_control_pack(), self.repo_root, "t_main_1")

        self.assertEqual(report["status"], "fail")
        self.assertIn(
            "support_gate",
            {finding["field"] for finding in report["findings"] if finding["severity"] == "fail"},
        )

    def test_audit_state_sync_passes_with_coherent_cards_status_and_roadmap(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][1]["env_diff"] = {
            "stage_overlay_source": str(self.repo_root / "phase5" / "runs" / "t_main_2" / "stage_overlay_live.env"),
        }
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)
        main_card_path = self.mod.prepare_tranche(repo_root=self.repo_root, tranche_id="t_main_1", mark_in_progress=False)
        child_card_path = self.mod.prepare_tranche(repo_root=self.repo_root, tranche_id="t_main_2", mark_in_progress=False)

        main_card = yaml.safe_load(main_card_path.read_text(encoding="utf-8"))
        child_card = yaml.safe_load(child_card_path.read_text(encoding="utf-8"))

        (self.repo_root / "phase5" / "status.md").write_text(
            "\n".join(
                [
                    "# Phase 5 Status",
                    "",
                    f"- `t_main_1`: `ready` surface_id `{main_card['surface_id']}`",
                    f"- `t_main_2`: `blocked` surface_id `{child_card['surface_id']}`",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        (self.repo_root / "ROADMAP.md").write_text(
            "\n".join(
                [
                    "# Roadmap",
                    "",
                    f"- `t_main_1` surface `{main_card['surface_id']}`",
                    f"- `t_main_2` surface `{child_card['surface_id']}`",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        report = self.mod.audit_state_sync(queue, _minimal_control_pack(), self.repo_root)

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["critical_count"], 0)
        self.assertEqual({tranche["tranche_id"] for tranche in report["tranches"]}, {"t_main_1", "t_main_2"})
        self.assertEqual(report["tranches"][0]["linked_child_ids"], ["t_main_2"])

    def test_state_sync_blocks_promotion_on_warning_only_report(self):
        self.assertTrue(
            self.mod.state_sync_blocks_promotion(
                {
                    "critical_count": 0,
                    "warning_count": 1,
                }
            )
        )
        self.assertFalse(
            self.mod.state_sync_blocks_promotion(
                {
                    "critical_count": 0,
                    "warning_count": 0,
                }
            )
        )

    def test_audit_state_sync_detects_roadmap_drift(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][1]["env_diff"] = {
            "stage_overlay_source": str(self.repo_root / "phase5" / "runs" / "t_main_2" / "stage_overlay_live.env"),
        }
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)
        main_card_path = self.mod.prepare_tranche(repo_root=self.repo_root, tranche_id="t_main_1", mark_in_progress=False)
        child_card_path = self.mod.prepare_tranche(repo_root=self.repo_root, tranche_id="t_main_2", mark_in_progress=False)

        main_card = yaml.safe_load(main_card_path.read_text(encoding="utf-8"))
        child_card = yaml.safe_load(child_card_path.read_text(encoding="utf-8"))

        (self.repo_root / "phase5" / "status.md").write_text(
            "\n".join(
                [
                    "# Phase 5 Status",
                    "",
                    f"- `t_main_1`: `ready` surface_id `{main_card['surface_id']}`",
                    f"- `t_main_2`: `blocked` surface_id `{child_card['surface_id']}`",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        (self.repo_root / "ROADMAP.md").write_text(
            "\n".join(
                [
                    "# Roadmap",
                    "",
                    f"- `t_main_1` surface `{main_card['surface_id']}`",
                    f"- `t_main_2` registered for later",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        report = self.mod.audit_state_sync(queue, _minimal_control_pack(), self.repo_root)

        self.assertEqual(report["status"], "fail")
        fields = {finding["field"] for finding in report["findings"] if finding["severity"] == "fail"}
        self.assertIn("ROADMAP.md", fields)

    def test_audit_state_sync_detects_status_board_status_drift(self):
        queue, main_card, child_card = self._write_coherent_state_sync_layout()
        (self.repo_root / "phase5" / "status.md").write_text(
            "\n".join(
                [
                    "# Phase 5 Status",
                    "",
                    f"- `t_main_1`: `promoted` surface_id `{main_card['surface_id']}`",
                    f"- `t_main_2`: `blocked` surface_id `{child_card['surface_id']}`",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        report = self.mod.audit_state_sync(queue, _minimal_control_pack(), self.repo_root)

        self.assertEqual(report["critical_count"], 0)
        self.assertEqual(report["warning_count"], 1)
        warning_fields = {finding["field"] for finding in report["findings"] if finding["severity"] == "warn"}
        self.assertIn("phase5/status.md.status", warning_fields)

    def test_audit_state_sync_detects_tranche_card_surface_mismatch(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][1]["env_diff"] = {
            "stage_overlay_source": str(self.repo_root / "phase5" / "runs" / "t_main_2" / "stage_overlay_live.env"),
        }
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)
        main_card_path = self.mod.prepare_tranche(repo_root=self.repo_root, tranche_id="t_main_1", mark_in_progress=False)
        child_card_path = self.mod.prepare_tranche(repo_root=self.repo_root, tranche_id="t_main_2", mark_in_progress=False)

        main_card = yaml.safe_load(main_card_path.read_text(encoding="utf-8"))
        child_card = yaml.safe_load(child_card_path.read_text(encoding="utf-8"))
        child_surface_id = child_card["surface_id"]
        child_card["surface_id"] = "deadbeefdeadbeef"
        _write_yaml(child_card_path, child_card)

        (self.repo_root / "phase5" / "status.md").write_text(
            "\n".join(
                [
                    "# Phase 5 Status",
                    "",
                    f"- `t_main_1`: `ready` surface_id `{main_card['surface_id']}`",
                    f"- `t_main_2`: `blocked` surface_id `{child_card['surface_id']}`",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        (self.repo_root / "ROADMAP.md").write_text(
            "\n".join(
                [
                    "# Roadmap",
                    "",
                    f"- `t_main_1` surface `{main_card['surface_id']}`",
                    f"- `t_main_2` surface `{child_surface_id}`",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        report = self.mod.audit_state_sync(queue, _minimal_control_pack(), self.repo_root)

        self.assertEqual(report["status"], "fail")
        fields = {finding["field"] for finding in report["findings"] if finding["severity"] == "fail"}
        self.assertIn("tranche_card.surface_id", fields)

    def test_ensure_orchestration_preflight_passes_with_coherent_state_sync(self):
        queue, main_card, child_card = self._write_coherent_state_sync_layout()
        tranche = queue["serialized_mainline"][0]
        control_pack = _minimal_control_pack()

        with mock.patch.object(
            self.mod,
            "ensure_shadow_health",
            return_value={"healthy": True, "ready": True, "trade_mode": "shadow"},
        ), mock.patch.object(
            self.mod,
            "ensure_runtime_storage_headroom",
            return_value={
                "repo": {"free_bytes": 10},
                "promotion_runs": {"free_bytes": 10},
                "telemetry": {"free_bytes": 10},
                "tempdir": {"free_bytes": 10},
                "current_runs": {"free_bytes": 10},
            },
        ), mock.patch.object(
            self.mod,
            "hyperliquid_user_rate_limit_preflight",
            return_value={"status": "pass", "blocked": False, "nRequestsUsed": 1, "nRequestsCap": 10},
        ):
            preflight = self.mod.ensure_orchestration_preflight(tranche, control_pack, self.repo_root)

        self.assertEqual(preflight["status"], "pass")
        self.assertEqual(preflight["decision"], "pass")
        self.assertFalse(preflight["blocked_by_state_sync"])
        self.assertFalse(preflight["blocked_by_hyperliquid_quota"])
        self.assertEqual(preflight["hyperliquid_quota_preflight"]["status"], "pass")
        self.assertEqual(preflight["state_sync"]["status"], "pass")
        self.assertEqual(preflight["state_sync"]["critical_count"], 0)
        self.assertTrue(Path(preflight["preflight_summary_path"]).exists())
        self.assertTrue(Path(preflight["state_sync_report_path"]).exists())
        self.assertEqual(preflight["surface_id"], main_card["surface_id"])
        report = json.loads(Path(preflight["state_sync_report_path"]).read_text(encoding="utf-8"))
        self.assertEqual(report["tranches"][0]["linked_child_ids"], ["t_main_2"])
        self.assertEqual(report["tranches"][1]["surface_id"], child_card["surface_id"])

    def test_ensure_orchestration_preflight_blocks_warning_only_state_sync(self):
        queue = _minimal_queue()
        tranche = queue["serialized_mainline"][0]
        control_pack = _minimal_control_pack()

        with mock.patch.object(
            self.mod,
            "ensure_shadow_health",
            return_value={"healthy": True, "ready": True, "trade_mode": "shadow"},
        ), mock.patch.object(
            self.mod,
            "ensure_runtime_storage_headroom",
            return_value={
                "repo": {"free_bytes": 10},
                "promotion_runs": {"free_bytes": 10},
                "telemetry": {"free_bytes": 10},
                "tempdir": {"free_bytes": 10},
                "current_runs": {"free_bytes": 10},
            },
        ), mock.patch.object(
            self.mod,
            "hyperliquid_user_rate_limit_preflight",
            return_value={"status": "pass", "blocked": False},
        ), mock.patch.object(
            self.mod,
            "audit_state_sync",
            return_value=self._state_sync_warning_report(),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                self.mod.ensure_orchestration_preflight(tranche, control_pack, self.repo_root)

        self.assertIn("orchestration preflight blocked by state-sync", str(ctx.exception))
        preflight_summary = self.repo_root / "phase5" / "runs" / "t_main_1" / "preflight_summary.json"
        state_sync_report = self.repo_root / "phase5" / "runs" / "t_main_1" / "state_sync_report.json"
        self.assertTrue(preflight_summary.exists())
        self.assertTrue(state_sync_report.exists())
        payload = json.loads(preflight_summary.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "fail")
        self.assertEqual(payload["decision"], "hold")
        self.assertTrue(payload["blocked_by_state_sync"])
        self.assertFalse(payload["blocked_by_hyperliquid_quota"])
        self.assertEqual(payload["state_sync"]["warning_count"], 1)

    def test_ensure_orchestration_preflight_blocks_critical_state_sync(self):
        queue = _minimal_queue()
        tranche = queue["serialized_mainline"][0]
        control_pack = _minimal_control_pack()

        with mock.patch.object(
            self.mod,
            "ensure_shadow_health",
            return_value={"healthy": True, "ready": True, "trade_mode": "shadow"},
        ), mock.patch.object(
            self.mod,
            "ensure_runtime_storage_headroom",
            return_value={
                "repo": {"free_bytes": 10},
                "promotion_runs": {"free_bytes": 10},
                "telemetry": {"free_bytes": 10},
                "tempdir": {"free_bytes": 10},
                "current_runs": {"free_bytes": 10},
            },
        ), mock.patch.object(
            self.mod,
            "hyperliquid_user_rate_limit_preflight",
            return_value={"status": "pass", "blocked": False},
        ), mock.patch.object(
            self.mod,
            "audit_state_sync",
            return_value=self._state_sync_critical_report(),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                self.mod.ensure_orchestration_preflight(tranche, control_pack, self.repo_root)

        self.assertIn("orchestration preflight blocked by state-sync", str(ctx.exception))
        preflight_summary = self.repo_root / "phase5" / "runs" / "t_main_1" / "preflight_summary.json"
        state_sync_report = self.repo_root / "phase5" / "runs" / "t_main_1" / "state_sync_report.json"
        self.assertTrue(preflight_summary.exists())
        self.assertTrue(state_sync_report.exists())
        payload = json.loads(preflight_summary.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "fail")
        self.assertEqual(payload["decision"], "hold")
        self.assertTrue(payload["blocked_by_state_sync"])
        self.assertFalse(payload["blocked_by_hyperliquid_quota"])
        self.assertEqual(payload["state_sync"]["critical_count"], 1)

    def test_ensure_orchestration_preflight_persists_hyperliquid_quota_block(self):
        queue, _, _ = self._write_coherent_state_sync_layout()
        tranche = queue["serialized_mainline"][0]
        control_pack = _minimal_control_pack()

        with mock.patch.object(
            self.mod,
            "ensure_shadow_health",
            return_value={"healthy": True, "ready": True, "trade_mode": "shadow"},
        ), mock.patch.object(
            self.mod,
            "ensure_runtime_storage_headroom",
            return_value={
                "repo": {"free_bytes": 10},
                "promotion_runs": {"free_bytes": 10},
                "telemetry": {"free_bytes": 10},
                "tempdir": {"free_bytes": 10},
                "current_runs": {"free_bytes": 10},
            },
        ), mock.patch.object(
            self.mod,
            "hyperliquid_user_rate_limit_preflight",
            side_effect=RuntimeError(
                "Hyperliquid action quota blocks live admission: nRequestsUsed=71983 nRequestsCap=67835 usage_pct=106.1148."
            ),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                self.mod.ensure_orchestration_preflight(tranche, control_pack, self.repo_root)

        self.assertIn("orchestration preflight blocked by Hyperliquid action quota", str(ctx.exception))
        preflight_summary = self.repo_root / "phase5" / "runs" / "t_main_1" / "preflight_summary.json"
        self.assertTrue(preflight_summary.exists())
        payload = json.loads(preflight_summary.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "fail")
        self.assertEqual(payload["decision"], "hold")
        self.assertFalse(payload["blocked_by_state_sync"])
        self.assertTrue(payload["blocked_by_hyperliquid_quota"])
        self.assertEqual(payload["hyperliquid_quota_preflight"]["status"], "fail")
        self.assertIn("nRequestsUsed=71983", payload["hyperliquid_quota_preflight"]["error"])

    def test_orchestration_preflight_requires_refresh_for_legacy_or_failed_payloads(self):
        self.assertTrue(self.mod.orchestration_preflight_requires_refresh(None))
        self.assertTrue(
            self.mod.orchestration_preflight_requires_refresh(
                {
                    "status": "pass",
                    "blocked_by_state_sync": False,
                }
            )
        )
        self.assertTrue(
            self.mod.orchestration_preflight_requires_refresh(
                {
                    "status": "fail",
                    "hyperliquid_quota_preflight": {"status": "fail", "blocked": True},
                }
            )
        )
        self.assertFalse(
            self.mod.orchestration_preflight_requires_refresh(
                {
                    "status": "pass",
                    "hyperliquid_quota_preflight": {"status": "pass", "blocked": False},
                }
            )
        )

    def test_lane_overlay_paths_include_roadmap_for_state_sync_truth(self):
        queue = _minimal_queue()
        control_pack = _minimal_control_pack()
        parent = queue["serialized_mainline"][0]
        lane_spec = {"lane_id": "child_prep_t_main_2", "child_tranche_id": "t_main_2"}

        paths = self.mod.lane_overlay_paths(self.repo_root, queue, control_pack, parent, lane_spec)

        self.assertIn("ROADMAP.md", paths)
        self.assertIn("phase5", paths)

    def test_tranche_fail_child_target_prefers_surface_local_restore_hygiene_child(self):
        tranche = _minimal_queue()["serialized_mainline"][0]
        tranche["surface_local_restore_hygiene_child"] = "t_main_3"
        tranche["matched_fail_routes"] = {"restore_hygiene": "t_main_2"}

        target = self.mod.tranche_fail_child_target(tranche, "restore_hygiene")

        self.assertEqual(target, "t_main_3")

    def test_tranche_fail_route_specs_include_surface_local_restore_hygiene_child(self):
        tranche = _minimal_queue()["serialized_mainline"][0]
        tranche["surface_local_restore_hygiene_child"] = "t_main_3"
        tranche["matched_fail_routes"] = {}

        routes = self.mod.tranche_fail_route_specs(tranche)

        self.assertIn(
            {
                "child_tranche_id": "t_main_3",
                "observed_blocker_family": "restore_hygiene",
                "route_kind": "surface_local_restore_hygiene_child",
            },
            routes,
        )

    def test_validate_queue_accepts_support_families(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["automation"]["support_families"] = [
            {
                "id": "tooling",
                "support_track_id": "t_support_1",
                "trigger_mode": "always",
                "autorun_policy": "validate_only",
                "max_parallel_runs": 2,
                "stop_on_mainline_promote": True,
            }
        ]

        self.mod.validate_queue(queue)

    def test_render_status_includes_topology_and_current_mainline(self):
        rendered = self.mod.render_status_markdown(_minimal_queue(), _minimal_control_pack(), self.repo_root)

        self.assertIn("Phase 5 Status", rendered)
        self.assertIn("aster,hyperliquid,lighter,paradex", rendered)
        self.assertIn("`t_main_1`: `ready` <- current", rendered)
        self.assertIn("`t_support_1`", rendered)
        self.assertIn("hypothesis blocker", rendered)

    def test_render_status_prefers_effective_overlay_topology_for_current_mainline(self):
        overlay = self.repo_root / "current_overlay.env"
        overlay.write_text(
            "\n".join(
                [
                    "PARAPHINA_LIVE_CONNECTORS=hyperliquid,lighter,extended,aster,paradex",
                    "PARAPHINA_FV_DISABLED_VENUES=lighter",
                    "PARAPHINA_MM_VENUE_ROLE_EXTENDED=fill",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["env_diff"] = {"stage_overlay_source": str(overlay)}

        rendered = self.mod.render_status_markdown(queue, _minimal_control_pack(), self.repo_root)

        self.assertIn("aster,extended,hyperliquid,lighter,paradex", rendered)

    def test_render_status_prefers_staged_overlay_target_when_present(self):
        target_overlay = self.repo_root / "stage_overlay_target.env"
        target_overlay.write_text(
            "\n".join(
                [
                    "PARAPHINA_LIVE_CONNECTORS=hyperliquid,lighter,extended,aster,paradex",
                    "PARAPHINA_FV_DISABLED_VENUES=",
                    "PARAPHINA_MM_VENUE_ROLE_EXTENDED=fill",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        source_overlay = self.repo_root / "current_overlay.env"
        source_overlay.write_text(
            "PARAPHINA_LIVE_CONNECTORS=hyperliquid,lighter,aster,paradex\n",
            encoding="utf-8",
        )
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["env_diff"] = {"stage_overlay_source": str(source_overlay)}
        control_pack = _minimal_control_pack()
        control_pack["execution_defaults"]["stage_overlay_target"] = str(target_overlay)

        rendered = self.mod.render_status_markdown(queue, control_pack, self.repo_root)

        self.assertIn("stage_overlay_target.env", rendered)
        self.assertIn("aster,extended,hyperliquid,lighter,paradex", rendered)

    def test_render_status_treats_hold_mainline_as_current(self):
        target_overlay = self.repo_root / "stage_overlay_target.env"
        target_overlay.write_text(
            "PARAPHINA_LIVE_CONNECTORS=hyperliquid,lighter,extended,aster,paradex\n",
            encoding="utf-8",
        )
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["status"] = "hold"
        control_pack = _minimal_control_pack()
        control_pack["execution_defaults"]["stage_overlay_target"] = str(target_overlay)

        rendered = self.mod.render_status_markdown(queue, control_pack, self.repo_root)

        self.assertIn("`t_main_1`: `hold` <- current", rendered)
        self.assertIn("aster,extended,hyperliquid,lighter,paradex", rendered)

    def test_render_status_treats_latest_promoted_mainline_as_current(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["status"] = "hold"
        queue["serialized_mainline"][0]["history"] = [
            {"decision": "hold", "timestamp_utc": "2026-03-29T00:00:00Z"}
        ]
        promoted = json.loads(json.dumps(queue["serialized_mainline"][0]))
        promoted["id"] = "t_main_promoted"
        promoted["status"] = "promoted"
        promoted["hypothesis"] = "Promoted hypothesis"
        promoted["history"] = [
            {"decision": "promote", "timestamp_utc": "2026-03-29T01:00:00Z"}
        ]
        queue["serialized_mainline"].append(promoted)

        rendered = self.mod.render_status_markdown(queue, _minimal_control_pack(), self.repo_root)

        self.assertIn("`t_main_promoted`: `promoted` <- current", rendered)
        self.assertNotIn("`t_main_1`: `hold` <- current", rendered)
        self.assertIn("`none`: latest serialized-mainline child `t_main_promoted` promoted", rendered)

    def test_duration_label_maps_expected_windows(self):
        self.assertEqual(self.mod.duration_label(300), "5m_canary")
        self.assertEqual(self.mod.duration_label(1200), "20m_soak")
        self.assertEqual(self.mod.duration_label(3600), "60m_qual")

    def test_tranche_autoscore_override_keeps_default_clean_rules(self):
        tranche = _minimal_queue()["serialized_mainline"][0]
        tranche["automation"]["autoscore"] = {
            "promotion": [
                {
                    "source": "metrics",
                    "path": "fills.total_count",
                    "op": ">=",
                    "value": 1,
                    "severity": "fail",
                }
            ]
        }

        automation = self.mod.tranche_automation(tranche, _minimal_control_pack(), self.repo_root)

        self.assertTrue(
            any(rule["path"] == "guard_intervened" for rule in automation["autoscore"]["clean"])
        )
        self.assertTrue(
            any(rule["path"] == "closeout_contract_complete" for rule in automation["autoscore"]["clean"])
        )
        self.assertTrue(
            any(rule["path"] == "fills.total_count" for rule in automation["autoscore"]["promotion"])
        )

    def test_autoscore_run_fails_guard_intervention_with_custom_promotion_only(self):
        tranche = _minimal_queue()["serialized_mainline"][0]
        tranche["automation"]["autoscore"] = {
            "promotion": [
                {
                    "source": "metrics",
                    "path": "fills.total_count",
                    "op": ">=",
                    "value": 1,
                    "severity": "fail",
                }
            ]
        }
        run_root = self.repo_root / "promotion_runs" / "guard_intervened"
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "live_closeout_bundle.json").write_text(
            json.dumps(
                {
                    "summary_exists": True,
                    "report_exists": True,
                    "guard_intervened": True,
                    "guard_window_completed": False,
                    "pre_restore_venue_audit_clean": False,
                    "post_rollback_venue_audit_clean": False,
                    "healthy_post": True,
                    "ready_post": True,
                    "kill_events_present_post": False,
                    "trade_mode_post": "shadow",
                    "systemd_nrestarts_post": "0",
                }
            ),
            encoding="utf-8",
        )
        (run_root / "live_segment_summary.json").write_text("{}", encoding="utf-8")
        (run_root / "live_metrics.json").write_text(
            json.dumps({"fills": {"total_count": 1}}),
            encoding="utf-8",
        )
        (run_root / "health_post.json").write_text(json.dumps({"healthy": True}), encoding="utf-8")
        (run_root / "systemd_post.txt").write_text("NRestarts=0\n", encoding="utf-8")

        autoscore = self.mod.autoscore_run(
            tranche,
            _minimal_control_pack(),
            run_root,
            3600,
            self.repo_root,
        )

        self.assertFalse(autoscore["clean"]["passed"])
        self.assertIn("guard_intervened", {rule["path"] for rule in autoscore["clean"]["failed_rules"]})

    def test_autoscore_blocks_clean_continue_when_restore_cleanup_was_required(self):
        tranche = _minimal_queue()["serialized_mainline"][0]
        run_root = self.repo_root / "promotion_runs" / "restore_cleanup_required"
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "live_closeout_bundle.json").write_text(
            json.dumps(
                {
                    "summary_exists": True,
                    "report_exists": True,
                    "metrics_exists": True,
                    "guard_intervened": False,
                    "guard_window_completed": True,
                    "first_pre_restore_venue_audit_clean": False,
                    "pre_restore_cleanup_required": True,
                    "pre_restore_cleanup_cost_usd": 0.707216,
                    "pre_restore_venue_audit_clean": True,
                    "post_rollback_venue_audit_clean": True,
                    "healthy_post": True,
                    "ready_post": True,
                    "kill_events_present_post": False,
                    "trade_mode_post": "shadow",
                    "systemd_active_state_post": "active",
                    "systemd_sub_state_post": "running",
                    "systemd_nrestarts_post": "0",
                }
            ),
            encoding="utf-8",
        )
        (run_root / "live_segment_summary.json").write_text("{}", encoding="utf-8")
        (run_root / "live_metrics.json").write_text("{}", encoding="utf-8")
        (run_root / "health_post.json").write_text(json.dumps({"healthy": True}), encoding="utf-8")
        (run_root / "systemd_post.txt").write_text("NRestarts=0\n", encoding="utf-8")

        autoscore = self.mod.autoscore_run(
            tranche,
            _minimal_control_pack(),
            run_root,
            300,
            self.repo_root,
        )

        failed_paths = {rule["path"] for rule in autoscore["clean"]["failed_rules"]}
        self.assertFalse(autoscore["clean"]["passed"])
        self.assertEqual(autoscore["suggested_action"], "hold")
        self.assertEqual(self.mod.rung_decision({"continue_on": "clean"}, autoscore), "hold")
        self.assertIn("first_pre_restore_venue_audit_clean", failed_paths)
        self.assertIn("pre_restore_cleanup_required", failed_paths)

    def test_evaluate_rule_group_separates_hold_only_from_hard_failure(self):
        result = self.mod.evaluate_rule_group(
            [
                {
                    "source": "metrics",
                    "path": "fills.total_count",
                    "op": ">",
                    "value": 0,
                    "severity": "hold_only",
                }
            ],
            {"metrics": {"fills": {"total_count": 0}}},
        )

        self.assertFalse(result["passed"])
        self.assertEqual(result["decision_effect"], "hold")
        self.assertEqual(len(result["hold_rules"]), 1)
        self.assertEqual(result["hard_failed_rules"], [])
        self.assertEqual(result["rollback_rules"], [])

    def test_autoscore_run_suggests_rollback_for_rollback_decision_effect(self):
        tranche = _minimal_queue()["serialized_mainline"][0]
        tranche["automation"]["autoscore"] = {
            "clean": [
                {
                    "source": "closeout",
                    "path": "reconcile_mismatch_count_post",
                    "op": "==",
                    "value": 0,
                    "severity": "fail",
                    "decision_effect": "rollback",
                }
            ]
        }
        run_root = self.repo_root / "promotion_runs" / "rollback_effect"
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "live_closeout_bundle.json").write_text(
            json.dumps(
                {
                    "summary_exists": True,
                    "report_exists": True,
                    "metrics_exists": True,
                    "guard_result_exists": True,
                    "health_post_complete": True,
                    "systemd_post_complete": True,
                    "closeout_contract_complete": True,
                    "guard_intervened": False,
                    "guard_window_completed": True,
                    "pre_restore_venue_audit_clean": True,
                    "post_rollback_venue_audit_clean": True,
                    "healthy_post": True,
                    "ready_post": True,
                    "kill_events_present_post": False,
                    "trade_mode_post": "shadow",
                    "systemd_nrestarts_post": "0",
                    "reconcile_mismatch_count_post": 1,
                }
            ),
            encoding="utf-8",
        )
        (run_root / "live_segment_summary.json").write_text("{}", encoding="utf-8")
        (run_root / "live_metrics.json").write_text("{}", encoding="utf-8")
        (run_root / "telemetry_report_live_segment.md").write_text("report\n", encoding="utf-8")
        (run_root / "guard_result.json").write_text(json.dumps({"exit_code": 0}), encoding="utf-8")
        (run_root / "health_post.json").write_text(json.dumps({"healthy": True}), encoding="utf-8")
        (run_root / "systemd_post.txt").write_text("NRestarts=0\n", encoding="utf-8")

        autoscore = self.mod.autoscore_run(
            tranche,
            _minimal_control_pack(),
            run_root,
            300,
            self.repo_root,
        )

        self.assertEqual(autoscore["suggested_action"], "rollback")
        self.assertEqual(autoscore["clean"]["decision_effect"], "rollback")
        self.assertEqual(len(autoscore["clean"]["rollback_rules"]), 1)
        self.assertEqual(self.mod.rung_decision({"continue_on": "clean"}, autoscore), "rollback")

    def test_autoscore_uses_direct_venue_audit_and_cashflow_sources(self):
        tranche = _minimal_queue()["serialized_mainline"][0]
        tranche["automation"]["rung_plan"] = [
            {"duration_sec": 300, "continue_on": "clean"},
            {"duration_sec": 7200, "continue_on": "promotion"},
        ]
        tranche["automation"]["autoscore"] = {
            "promotion": [
                {
                    "source": "direct_venue_audit",
                    "path": "ok",
                    "op": "==",
                    "value": True,
                    "severity": "fail",
                },
                {
                    "source": "cashflow",
                    "path": "max_unexplained_equity_drift_usd",
                    "op": "<=",
                    "value": 0.5,
                    "severity": "fail",
                },
            ]
        }
        run_root = self.repo_root / "promotion_runs" / "artifact_sources"
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "live_closeout_bundle.json").write_text(
            json.dumps(
                {
                    "summary_exists": True,
                    "report_exists": True,
                    "metrics_exists": True,
                    "guard_result_exists": True,
                    "health_post_complete": True,
                    "systemd_post_complete": True,
                    "closeout_contract_complete": True,
                    "guard_intervened": False,
                    "guard_window_completed": True,
                    "pre_restore_venue_audit_clean": True,
                    "post_rollback_venue_audit_clean": True,
                    "healthy_post": True,
                    "ready_post": True,
                    "kill_events_present_post": False,
                    "trade_mode_post": "shadow",
                    "systemd_nrestarts_post": "0",
                }
            ),
            encoding="utf-8",
        )
        (run_root / "live_segment_summary.json").write_text("{}", encoding="utf-8")
        (run_root / "live_metrics.json").write_text("{}", encoding="utf-8")
        (run_root / "telemetry_report_live_segment.md").write_text("report\n", encoding="utf-8")
        (run_root / "guard_result.json").write_text(json.dumps({"exit_code": 0}), encoding="utf-8")
        (run_root / "health_post.json").write_text(json.dumps({"healthy": True}), encoding="utf-8")
        (run_root / "systemd_post.txt").write_text("NRestarts=0\n", encoding="utf-8")
        (run_root / "direct_venue_audit_post_20260425T010000Z.json").write_text(
            json.dumps({"ok": True, "violations": []}),
            encoding="utf-8",
        )
        (run_root / "cashflow_attribution.json").write_text(
            json.dumps({"max_unexplained_equity_drift_usd": 0.0}),
            encoding="utf-8",
        )

        payloads = self.mod.autoscore_payloads(run_root)
        autoscore = self.mod.autoscore_run(
            tranche,
            _minimal_control_pack(),
            run_root,
            7200,
            self.repo_root,
        )

        self.assertTrue(payloads["direct_venue_audit"]["exists"])
        self.assertTrue(payloads["cashflow"]["exists"])
        self.assertEqual(autoscore["suggested_action"], "promote")

    def test_tranche_live_guard_args_inherits_defaults(self):
        defaults = _minimal_control_pack()["execution_defaults"]
        self.assertEqual(
            self.mod.tranche_live_guard_args({"id": "t_main_1"}, defaults),
            ["--pre-restore-cleanup-on-exit"],
        )

    def test_tranche_live_guard_args_validates_strings(self):
        defaults = _minimal_control_pack()["execution_defaults"]
        self.assertEqual(
            self.mod.tranche_live_guard_args({"execution": {"live_guard_args": ["--flag"]}}, defaults),
            ["--pre-restore-cleanup-on-exit", "--flag"],
        )
        with self.assertRaises(ValueError):
            self.mod.tranche_live_guard_args({"execution": {"live_guard_args": "--flag"}}, defaults)
        with self.assertRaises(ValueError):
            self.mod.tranche_live_guard_args({"id": "t_main_1"}, {"live_guard_args": "--flag"})

    def test_prepare_tranche_card_uses_effective_overlay_and_guard_args(self):
        overlay = self.repo_root / "effective_stage_overlay.env"
        overlay.write_text("PARAPHINA_LIVE_CONNECTORS=hyperliquid,lighter,extended,aster,paradex\n", encoding="utf-8")
        queue = _minimal_queue()
        queue["serialized_mainline"][2]["env_diff"] = {"stage_overlay_source": str(overlay)}
        queue["serialized_mainline"][2]["execution"] = {"live_guard_args": ["--pre-audit-cleanup-on-exit"]}
        queue["serialized_mainline"][2]["candidate"]["runtime_binary"] = "/tmp/candidate_live"
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        card_path = self.mod.prepare_tranche(
            repo_root=self.repo_root,
            tranche_id="t_main_3",
            mark_in_progress=False,
        )

        card = yaml.safe_load(card_path.read_text(encoding="utf-8"))
        self.assertEqual(card["execution_defaults"]["stage_overlay_source"], str(overlay))
        self.assertEqual(
            card["execution_defaults"]["live_guard_args"],
            ["--pre-restore-cleanup-on-exit", "--pre-audit-cleanup-on-exit"],
        )
        self.assertEqual(card["execution_defaults"]["runtime_binary"], "/tmp/candidate_live")

    def test_prepare_tranche_card_uses_effective_topology_snapshot(self):
        overlay = self.repo_root / "effective_stage_overlay.env"
        overlay.write_text(
            "\n".join(
                [
                    "PARAPHINA_LIVE_CONNECTORS=hyperliquid,lighter,extended,aster,paradex",
                    "PARAPHINA_FV_DISABLED_VENUES=lighter",
                    "PARAPHINA_MM_VENUE_ROLE_EXTENDED=fill",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        queue = _minimal_queue()
        queue["serialized_mainline"][2]["env_diff"] = {"stage_overlay_source": str(overlay)}
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        card_path = self.mod.prepare_tranche(
            repo_root=self.repo_root,
            tranche_id="t_main_3",
            mark_in_progress=False,
        )

        card = yaml.safe_load(card_path.read_text(encoding="utf-8"))
        self.assertEqual(
            card["effective_topology"]["connectors"],
            ["aster", "extended", "hyperliquid", "lighter", "paradex"],
        )
        self.assertEqual(card["topology"], card["effective_topology"])
        self.assertIn("excluded_pending_rescue", card["baseline_topology"]["roles"]["extended"])

    def test_record_result_promote_refreshes_reopened_long_soak_surface_from_parent(self):
        exact_overlay = self.repo_root / "exact_surface_stage_overlay.env"
        exact_overlay.write_text(
            "PARAPHINA_LIVE_CONNECTORS=hyperliquid,lighter,extended,aster,paradex\nPARAPHINA_FV_DISABLED_VENUES=\n",
            encoding="utf-8",
        )
        exact_runtime = self.repo_root / "target" / "release" / "paraphina_live"
        exact_runtime.parent.mkdir(parents=True, exist_ok=True)
        exact_runtime.write_text("bin", encoding="utf-8")

        queue = _minimal_queue()
        queue["serialized_mainline"][0]["next_if_pass"] = self.mod.REOPENED_LONG_SOAK_ID
        queue["serialized_mainline"][0]["env_diff"] = {"stage_overlay_source": str(exact_overlay)}
        queue["serialized_mainline"][0]["candidate"]["runtime_binary"] = str(exact_runtime)
        queue["serialized_mainline"].append(
            {
                "id": self.mod.REOPENED_LONG_SOAK_ID,
                "track": "serialized_mainline",
                "status": "blocked",
                "objective": "Long soak",
                "hypothesis": "Long soak hypothesis",
                "control": {"description": "stale"},
                "candidate": {"change_scope": {"files": []}, "runtime_binary": "/stale/runtime"},
                "env_diff": {"stage_overlay_source": "/stale/overlay.env"},
            }
        )
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        with mock.patch.object(
            self.mod,
            "audit_state_sync",
            return_value={
                "schema_version": 1,
                "generated_utc": "2026-04-26T00:00:00Z",
                "repo_root": str(self.repo_root),
                "requested_tranche_id": "t_main_1",
                "status": "pass",
                "critical_count": 0,
                "warning_count": 0,
                "tranches": [
                    {
                        "tranche_id": "t_main_1",
                        "surface_id": "surface-1",
                        "linked_child_ids": [self.mod.REOPENED_LONG_SOAK_ID],
                        "status_summary": "pass",
                    }
                ],
                "findings": [],
            },
        ):
            self.mod.record_result("t_main_1", "promote", repo_root=self.repo_root)

        queue_after = yaml.safe_load((self.repo_root / "phase5" / "queue.yaml").read_text(encoding="utf-8"))
        long_soak = next(
            tranche for tranche in queue_after["serialized_mainline"] if tranche["id"] == self.mod.REOPENED_LONG_SOAK_ID
        )
        self.assertEqual(long_soak["env_diff"]["stage_overlay_source"], str(exact_overlay))
        self.assertEqual(long_soak["candidate"]["runtime_binary"], str(exact_runtime))
        self.assertIn("t_main_1", long_soak["control"]["description"])

    def test_record_result_hold_does_not_loop_reopened_long_soak_to_promoted_child(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["id"] = self.mod.REOPENED_LONG_SOAK_ID
        queue["serialized_mainline"][0]["matched_fail_routes"] = {
            "all5_projected_mm_budget_distribution_gap": "t_main_2"
        }
        queue["serialized_mainline"][0]["hypothesis_blocker_family"] = "all5_projected_mm_budget_distribution_gap"
        queue["serialized_mainline"][1]["status"] = "promoted"
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        control_pack = yaml.safe_load((self.repo_root / "phase5" / "control_pack.yaml").read_text(encoding="utf-8"))
        surface_id = self.mod.safe_tranche_surface_id(queue["serialized_mainline"][0], control_pack, self.repo_root)
        queue["serialized_mainline"][1]["history"] = [
            {
                "decision": "promote",
                "timestamp_utc": "2026-04-22T11:00:00Z",
                "surface_id": surface_id,
            }
        ]
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        self.mod.record_result(
            self.mod.REOPENED_LONG_SOAK_ID,
            "hold",
            repo_root=self.repo_root,
            observed_blocker_family="all5_projected_mm_budget_distribution_gap",
        )

        queue_after = yaml.safe_load((self.repo_root / "phase5" / "queue.yaml").read_text(encoding="utf-8"))
        parent = queue_after["serialized_mainline"][0]
        child = queue_after["serialized_mainline"][1]
        history = parent["history"][-1]
        self.assertFalse(history["child_activation_allowed"])
        self.assertIsNone(history["activated_child"])
        self.assertEqual(child["status"], "promoted")

    def test_record_result_hold_does_not_loop_reopened_long_soak_to_promoted_paradex_child(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["id"] = self.mod.REOPENED_LONG_SOAK_ID
        queue["serialized_mainline"][0]["matched_fail_routes"] = {
            "paradex_edge_floor_queue_loss": "t_main_2"
        }
        queue["serialized_mainline"][0]["hypothesis_blocker_family"] = "all5_projected_mm_budget_distribution_gap"
        queue["serialized_mainline"][1]["status"] = "promoted"
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        control_pack = yaml.safe_load((self.repo_root / "phase5" / "control_pack.yaml").read_text(encoding="utf-8"))
        surface_id = self.mod.safe_tranche_surface_id(queue["serialized_mainline"][0], control_pack, self.repo_root)
        queue["serialized_mainline"][1]["history"] = [
            {
                "decision": "promote",
                "timestamp_utc": "2026-04-22T13:00:00Z",
                "surface_id": surface_id,
            }
        ]
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        self.mod.record_result(
            self.mod.REOPENED_LONG_SOAK_ID,
            "hold",
            repo_root=self.repo_root,
            observed_blocker_family="paradex_edge_floor_queue_loss",
        )

        queue_after = yaml.safe_load((self.repo_root / "phase5" / "queue.yaml").read_text(encoding="utf-8"))
        parent = queue_after["serialized_mainline"][0]
        child = queue_after["serialized_mainline"][1]
        history = parent["history"][-1]
        self.assertFalse(history["child_activation_allowed"])
        self.assertIsNone(history["activated_child"])
        self.assertEqual(child["status"], "promoted")

    def test_auto_progress_serialized_mainline_once_waits_for_closeout(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["status"] = "in_progress"
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        run_root = self.repo_root / "promotion_runs" / "active_parent" / "live_canary"
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "guard.log").write_text("guard", encoding="utf-8")
        _write_yaml(
            self.repo_root / "phase5" / "runs" / "t_main_1" / "latest_run.yaml",
            {
                "updated_utc": "2026-04-22T11:08:18Z",
                "surface_id": "surface-1",
                "run_root": str(run_root),
                "duration_sec": 7200,
                "run_state": "live_started",
            },
        )

        result = self.mod.auto_progress_serialized_mainline_once(repo_root=self.repo_root)

        self.assertEqual(result["state"], "waiting_for_closeout")
        self.assertEqual(result["tranche_id"], "t_main_1")
        self.assertEqual(result["run_root"], str(run_root))
        self.assertEqual(result["duration_sec"], 7200)

    def test_auto_progress_serialized_mainline_once_returns_retryable_infra_invalid_on_headroom_error(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["status"] = "ready"
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        with mock.patch.object(
            self.mod,
            "resume_orchestrate_tranche",
            side_effect=RuntimeError(
                "telemetry headroom below automation default: free=8577986560 required=8589934592 path=/var/lib/paraphina/out/telemetry.jsonl"
            ),
        ):
            result = self.mod.auto_progress_serialized_mainline_once(repo_root=self.repo_root)

        self.assertEqual(result["state"], "infra_invalid")
        self.assertEqual(result["reason"], "telemetry_headroom_below_default")
        self.assertEqual(result["tranche_id"], "t_main_1")
        self.assertTrue(result["retryable"])
        self.assertIn("free=8577986560", result["error"])

    def test_auto_progress_serialized_mainline_once_returns_control_plane_invalid_on_state_sync_preflight_block(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["status"] = "ready"
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        with mock.patch.object(
            self.mod,
            "resume_orchestrate_tranche",
            side_effect=RuntimeError("t_main_1: orchestration preflight blocked by state-sync (0 critical, 1 warning findings)"),
        ):
            result = self.mod.auto_progress_serialized_mainline_once(repo_root=self.repo_root)

        self.assertEqual(result["state"], "control_plane_invalid")
        self.assertEqual(result["reason"], "state_sync_preflight_block")
        self.assertEqual(result["tranche_id"], "t_main_1")
        self.assertFalse(result["retryable"])
        self.assertIn("state-sync", result["error"])

    def test_auto_progress_serialized_mainline_once_handoffs_completed_current_hold_to_selected_child(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["status"] = "hold"
        queue["serialized_mainline"][1]["status"] = "hold"
        queue["serialized_mainline"][2]["status"] = "blocked"
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        with mock.patch.object(
            self.mod,
            "resume_orchestrate_tranche",
            return_value={
                "session_id": "t_main_1-completed",
                "state": "completed",
                "final_decision": "hold",
                "selected_child": "t_main_2",
            },
        ), mock.patch.object(
            self.mod,
            "orchestrate_tranche",
            return_value={
                "session_id": "t_main_2-running",
                "state": "running",
                "final_decision": None,
                "selected_child": None,
            },
        ) as handoff_mock:
            result = self.mod.auto_progress_serialized_mainline_once(repo_root=self.repo_root)

        self.assertEqual(result["state"], "resumed")
        self.assertEqual(result["reason"], "handoff_selected_child")
        self.assertEqual(result["tranche_id"], "t_main_2")
        self.assertEqual(result["prior_tranche_id"], "t_main_1")
        handoff_mock.assert_called_once_with("t_main_2", repo_root=self.repo_root, resume=False)

    def test_auto_progress_serialized_mainline_once_idles_on_completed_current_hold(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["status"] = "hold"
        queue["serialized_mainline"][1]["status"] = "blocked"
        queue["serialized_mainline"][2]["status"] = "blocked"
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        with mock.patch.object(
            self.mod,
            "resume_orchestrate_tranche",
            return_value={
                "session_id": "t_main_1-completed",
                "state": "completed",
                "final_decision": "hold",
                "selected_child": "t_main_2",
            },
        ):
            result = self.mod.auto_progress_serialized_mainline_once(repo_root=self.repo_root)

        self.assertEqual(result["state"], "idle")
        self.assertEqual(result["reason"], "completed_current_mainline_hold")
        self.assertEqual(result["tranche_id"], "t_main_1")
        self.assertEqual(result["session_state"], "completed")

    def test_watch_serialized_mainline_loops_until_idle(self):
        with mock.patch.object(
            self.mod,
            "auto_progress_serialized_mainline_once",
            side_effect=[
                {"state": "waiting_for_closeout", "tranche_id": "t_main_1"},
                {"state": "resumed", "tranche_id": "t_main_1", "next_current_tranche_id": "t_main_2"},
                {"state": "idle", "reason": "no_current_mainline"},
            ],
        ) as step_mock, mock.patch.object(self.mod.time, "sleep") as sleep_mock:
            events = self.mod.watch_serialized_mainline(repo_root=self.repo_root, poll_sec=7.0)

        self.assertEqual([event["state"] for event in events], ["waiting_for_closeout", "resumed", "idle"])
        sleep_mock.assert_called_once_with(7.0)
        self.assertEqual(step_mock.call_count, 3)

    def test_watch_serialized_mainline_retries_after_infra_invalid_until_idle(self):
        with mock.patch.object(
            self.mod,
            "auto_progress_serialized_mainline_once",
            side_effect=[
                {
                    "state": "infra_invalid",
                    "reason": "telemetry_headroom_below_default",
                    "retryable": True,
                    "tranche_id": "t_main_1",
                },
                {"state": "idle", "reason": "no_current_mainline"},
            ],
        ) as step_mock, mock.patch.object(self.mod.time, "sleep") as sleep_mock:
            events = self.mod.watch_serialized_mainline(repo_root=self.repo_root, poll_sec=7.0)

        self.assertEqual([event["state"] for event in events], ["infra_invalid", "idle"])
        sleep_mock.assert_called_once_with(7.0)
        self.assertEqual(step_mock.call_count, 2)

    def test_record_result_promote_autorefreshes_reopened_final_closeout(self):
        exact_overlay = self.repo_root / "exact_surface_stage_overlay.env"
        exact_overlay.write_text(
            "\n".join(
                [
                    "PARAPHINA_LIVE_CONNECTORS=hyperliquid,lighter,extended,aster,paradex",
                    "PARAPHINA_FV_DISABLED_VENUES=",
                    "PARAPHINA_EXCLUDED_VENUES=",
                    "PARAPHINA_MM_VENUE_ROLE_HYPERLIQUID=fill",
                    "PARAPHINA_MM_VENUE_ROLE_LIGHTER=fill",
                    "PARAPHINA_MM_VENUE_ROLE_EXTENDED=fill",
                    "PARAPHINA_MM_VENUE_ROLE_ASTER=fill",
                    "PARAPHINA_MM_VENUE_ROLE_PARADEX=fill",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        exact_runtime = self.repo_root / "target" / "release" / "paraphina_live"
        exact_runtime.parent.mkdir(parents=True, exist_ok=True)
        exact_runtime.write_text("bin", encoding="utf-8")

        queue = _minimal_queue()
        queue["serialized_mainline"][0]["next_if_pass"] = self.mod.REOPENED_FINAL_CLOSEOUT_ID
        queue["serialized_mainline"][0]["env_diff"] = {"stage_overlay_source": str(exact_overlay)}
        queue["serialized_mainline"][0]["candidate"]["runtime_binary"] = str(exact_runtime)
        queue["serialized_mainline"].append(
            {
                "id": self.mod.REOPENED_FINAL_CLOSEOUT_ID,
                "track": "serialized_mainline",
                "status": "promoted",
                "objective": "Final closeout",
                "hypothesis": "Closeout",
                "control": {"description": "stale"},
                "candidate": {"change_scope": {"files": []}, "runtime_binary": "/stale/runtime"},
                "env_diff": {"stage_overlay_source": "/stale/overlay.env"},
            }
        )
        queue["serialized_mainline"].append(
            {
                "id": "t_lineage",
                "track": "serialized_mainline",
                "status": "promoted",
                "objective": "Lineage",
                "hypothesis": "Lineage",
                "control": {"description": "lineage"},
                "candidate": {"change_scope": {"files": []}},
                "history": [],
            }
        )
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        control_pack = yaml.safe_load((self.repo_root / "phase5" / "control_pack.yaml").read_text(encoding="utf-8"))
        surface_id = self.mod.safe_tranche_surface_id(queue["serialized_mainline"][0], control_pack, self.repo_root)

        run_root = self.repo_root / "promotion_runs" / "parent_live"
        run_root.mkdir(parents=True, exist_ok=True)
        summary_path = run_root / "live_segment_summary.json"
        summary_path.write_text("{}", encoding="utf-8")
        guard_path = run_root / "guard.log"
        guard_path.write_text("guard\n", encoding="utf-8")
        (run_root / "telemetry_report_live_segment.md").write_text("report\n", encoding="utf-8")
        (run_root / "autoscore_bundle.json").write_text(
            json.dumps(
                {
                    "clean": {"passed": True},
                    "mechanism": {"passed": True},
                    "promotion": {"passed": True},
                    "suggested_action": "promote",
                }
            ),
            encoding="utf-8",
        )
        (run_root / "live_closeout_bundle.json").write_text(
            json.dumps(
                {
                    "segment_start_utc": "2026-04-22T11:08:18.000000Z",
                    "segment_end_utc": "2026-04-22T13:08:18.000000Z",
                    "tick_count": 28800,
                    "fill_count_total": 20,
                    "fill_base_total": 0.2,
                    "final_pnl_total": 0.01,
                    "final_q_global": 0.0,
                    "guard_window_completed": True,
                    "guard_exit_code": 0,
                    "guard_intervened": False,
                    "closeout_contract_complete": True,
                    "pre_restore_venue_audit_clean": True,
                    "post_rollback_venue_audit_clean": True,
                    "healthy_post": True,
                    "ready_post": True,
                    "kill_events_present_post": False,
                    "reconcile_mismatch_count_post": 0,
                    "systemd_nrestarts_post": 0,
                    "trade_mode_post": "shadow",
                    "summary_exists": True,
                    "report_exists": True,
                    "metrics_exists": True,
                    "guard_result_exists": True,
                    "health_post_complete": True,
                    "systemd_post_complete": True,
                }
            ),
            encoding="utf-8",
        )
        (run_root / "live_metrics.json").write_text(
            json.dumps(
                {
                    "execution_scorecard": {
                        "hyperliquid": {"place_i": 1, "place_ack": 1, "cancel_i": 1, "cancel_ack": 1, "fills": 1, "fill_base": 0.01},
                        "lighter": {"place_i": 2, "place_ack": 2, "cancel_i": 3, "cancel_ack": 3, "fills": 5, "fill_base": 0.05},
                        "extended": {"place_i": 1, "place_ack": 1, "cancel_i": 1, "cancel_ack": 1, "fills": 2, "fill_base": 0.02},
                        "aster": {"place_i": 2, "place_ack": 2, "cancel_i": 2, "cancel_ack": 2, "fills": 6, "fill_base": 0.06},
                        "paradex": {"place_i": 2, "place_ack": 2, "cancel_i": 2, "cancel_ack": 2, "fills": 6, "fill_base": 0.06},
                    },
                    "risk": {"would_send_zero_pct": 42.5},
                }
            ),
            encoding="utf-8",
        )
        (run_root / "balance_snapshot_comparison.json").write_text(
            json.dumps(
                {
                    "exists": True,
                    "venue_count": 5,
                    "total": {
                        "pre_usd": "317.00000000",
                        "post_usd": "317.01000000",
                        "delta_usd": "0.01000000",
                        "abs_delta_usd_float": 0.01,
                    },
                }
            ),
            encoding="utf-8",
        )
        (run_root / "direct_venue_audit_post_20260422T1308Z.json").write_text("{}", encoding="utf-8")

        lineage_root = self.repo_root / "promotion_runs" / "lineage_live"
        lineage_root.mkdir(parents=True, exist_ok=True)
        (lineage_root / "live_segment_summary.json").write_text("{}", encoding="utf-8")
        (lineage_root / "live_closeout_bundle.json").write_text(
            json.dumps(
                {
                    "segment_start_utc": "2026-04-22T09:59:47.000000Z",
                    "segment_end_utc": "2026-04-22T10:59:50.000000Z",
                    "fill_count_total": 41,
                    "fill_base_total": 0.394,
                }
            ),
            encoding="utf-8",
        )
        (lineage_root / "live_metrics.json").write_text(
            json.dumps(
                {
                    "execution_scorecard": {
                        "hyperliquid": {"fills": 3, "fill_base": 0.03},
                        "lighter": {"fills": 17, "fill_base": 0.16},
                        "extended": {"fills": 3, "fill_base": 0.03},
                        "aster": {"fills": 11, "fill_base": 0.104},
                        "paradex": {"fills": 7, "fill_base": 0.07},
                    }
                }
            ),
            encoding="utf-8",
        )
        queue["serialized_mainline"][3]["history"] = [
            {
                "decision": "promote",
                "timestamp_utc": "2026-04-22T11:03:46Z",
                "surface_id": surface_id,
                "summary_path": str(lineage_root / "live_segment_summary.json"),
                "activated_child": "t_main_1",
            }
        ]
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        with mock.patch.object(
            self.mod,
            "audit_state_sync",
            return_value={
                "schema_version": 1,
                "generated_utc": "2026-04-26T00:00:00Z",
                "repo_root": str(self.repo_root),
                "requested_tranche_id": "t_main_1",
                "status": "pass",
                "critical_count": 0,
                "warning_count": 0,
                "tranches": [
                    {
                        "tranche_id": "t_main_1",
                        "surface_id": surface_id,
                        "linked_child_ids": [self.mod.REOPENED_FINAL_CLOSEOUT_ID],
                        "status_summary": "pass",
                    }
                ],
                "findings": [],
            },
        ):
            self.mod.record_result(
                "t_main_1",
                "promote",
                repo_root=self.repo_root,
                summary_path=str(summary_path),
                guard_path=str(guard_path),
            )

        final_run_dir = self.repo_root / "phase5" / "runs" / self.mod.REOPENED_FINAL_CLOSEOUT_ID
        final_spec = yaml.safe_load((final_run_dir / "final_topology_spec.yaml").read_text(encoding="utf-8"))
        final_closeout = (final_run_dir / "final_closeout.md").read_text(encoding="utf-8")
        final_stage_verdict = json.loads((final_run_dir / "stage_verdict.json").read_text(encoding="utf-8"))
        queue_after = yaml.safe_load((self.repo_root / "phase5" / "queue.yaml").read_text(encoding="utf-8"))
        final_tranche = next(
            tranche for tranche in queue_after["serialized_mainline"] if tranche["id"] == self.mod.REOPENED_FINAL_CLOSEOUT_ID
        )

        self.assertEqual(final_spec["surface_id"], surface_id)
        self.assertEqual(final_spec["stage_overlay_source"], str(exact_overlay))
        self.assertTrue(final_spec["completion_standard"]["passed"])
        self.assertEqual(final_spec["closeout_disposition"]["verdict"], "accepted")
        self.assertIn(str(run_root), final_closeout)
        self.assertEqual(final_stage_verdict["decision"], "promote")
        self.assertEqual(final_stage_verdict["run_root"], str(run_root))
        self.assertEqual(final_tranche["status"], "promoted")
        self.assertEqual(final_tranche["history"][-1]["decision"], "promote")

    def test_build_reopened_final_topology_spec_holds_when_completion_standard_incomplete(self):
        exact_overlay = self.repo_root / "exact_surface_stage_overlay.env"
        exact_overlay.write_text(
            "\n".join(
                [
                    "PARAPHINA_LIVE_CONNECTORS=hyperliquid,lighter,extended,aster,paradex",
                    "PARAPHINA_FV_DISABLED_VENUES=",
                    "PARAPHINA_EXCLUDED_VENUES=",
                    "PARAPHINA_MM_VENUE_ROLE_HYPERLIQUID=fill",
                    "PARAPHINA_MM_VENUE_ROLE_LIGHTER=fill",
                    "PARAPHINA_MM_VENUE_ROLE_EXTENDED=fill",
                    "PARAPHINA_MM_VENUE_ROLE_ASTER=fill",
                    "PARAPHINA_MM_VENUE_ROLE_PARADEX=fill",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        exact_runtime = self.repo_root / "target" / "release" / "paraphina_live"
        exact_runtime.parent.mkdir(parents=True, exist_ok=True)
        exact_runtime.write_text("bin", encoding="utf-8")
        control_pack = _minimal_control_pack()
        source_tranche = _minimal_queue()["serialized_mainline"][0]
        source_tranche["env_diff"] = {"stage_overlay_source": str(exact_overlay)}
        source_tranche["candidate"]["runtime_binary"] = str(exact_runtime)
        final_tranche = {"id": self.mod.REOPENED_FINAL_CLOSEOUT_ID}
        run_root = self.repo_root / "promotion_runs" / "incomplete_final"
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "live_closeout_bundle.json").write_text(
            json.dumps(
                {
                    "segment_start_utc": "2026-04-22T11:08:18.000000Z",
                    "segment_end_utc": "2026-04-22T13:08:18.000000Z",
                    "tick_count": 28800,
                    "fill_count_total": 12,
                    "fill_base_total": 0.12,
                    "guard_window_completed": True,
                    "guard_exit_code": 0,
                    "guard_intervened": False,
                    "closeout_contract_complete": True,
                    "pre_restore_venue_audit_clean": True,
                    "post_rollback_venue_audit_clean": True,
                    "kill_events_present_post": False,
                    "reconcile_mismatch_count_post": 0,
                    "systemd_nrestarts_post": 0,
                    "trade_mode_post": "shadow",
                }
            ),
            encoding="utf-8",
        )
        (run_root / "live_metrics.json").write_text(
            json.dumps(
                {
                    "execution_scorecard": {
                        "hyperliquid": {"place_i": 1, "place_ack": 1, "cancel_i": 1, "cancel_ack": 1, "fills": 1, "fill_base": 0.01},
                        "lighter": {"place_i": 1, "place_ack": 1, "cancel_i": 1, "cancel_ack": 1, "fills": 4, "fill_base": 0.04},
                        "extended": {"place_i": 1, "place_ack": 1, "cancel_i": 1, "cancel_ack": 1, "fills": 0, "fill_base": 0.0},
                        "aster": {"place_i": 1, "place_ack": 1, "cancel_i": 1, "cancel_ack": 1, "fills": 4, "fill_base": 0.04},
                        "paradex": {"place_i": 1, "place_ack": 1, "cancel_i": 1, "cancel_ack": 1, "fills": 3, "fill_base": 0.03},
                    }
                }
            ),
            encoding="utf-8",
        )
        (run_root / "balance_snapshot_comparison.json").write_text(
            json.dumps(
                {
                    "exists": True,
                    "venue_count": 5,
                    "total": {
                        "pre_usd": "317.00000000",
                        "post_usd": "317.01000000",
                        "delta_usd": "0.01000000",
                        "abs_delta_usd_float": 0.01,
                    },
                }
            ),
            encoding="utf-8",
        )
        (run_root / "direct_venue_audit_post_20260422T1308Z.json").write_text("{}", encoding="utf-8")

        spec = self.mod.build_reopened_final_topology_spec(
            final_tranche,
            source_tranche,
            control_pack,
            self.repo_root,
            run_root,
            lineage=None,
        )

        self.assertEqual(spec["status"], "hold_closeout")
        self.assertFalse(spec["completion_standard"]["passed"])
        self.assertEqual(spec["completion_standard"]["non_hyperliquid_fill_venues_in_final_soak"], 3)
        self.assertEqual(spec["closeout_disposition"]["verdict"], "hold")

    def test_write_manual_live_stage_contracts_emits_stage_contract_files(self):
        queue = _minimal_queue()
        tranche = queue["serialized_mainline"][0]
        run_root = self.repo_root / "promotion_runs" / "manual_live"
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "live_segment_summary.json").write_text(
            json.dumps({"tick_count": 1}),
            encoding="utf-8",
        )
        (run_root / "live_metrics.json").write_text(
            json.dumps(
                {
                    "paradex_profile_usage_summary": {"interactive_token_usage_observed": True},
                    "paradex_ui_book_truth_summary": {"observed": True},
                    "supported_replace_visibility": {"paradex": {"gap_grace": 1}},
                }
            ),
            encoding="utf-8",
        )
        (run_root / "live_closeout_bundle.json").write_text(
            json.dumps(
                {
                    "summary_exists": True,
                    "report_exists": True,
                    "guard_intervened": False,
                    "guard_window_completed": True,
                    "pre_restore_venue_audit_clean": True,
                    "post_rollback_venue_audit_clean": True,
                    "healthy_post": True,
                    "ready_post": True,
                    "kill_events_present_post": False,
                    "trade_mode_post": "shadow",
                    "systemd_nrestarts_post": "0",
                }
            ),
            encoding="utf-8",
        )
        (run_root / "health_post.json").write_text(json.dumps({"healthy": True}), encoding="utf-8")
        (run_root / "systemd_post.txt").write_text("NRestarts=0\n", encoding="utf-8")

        self.mod.write_manual_live_stage_contracts(
            tranche,
            _minimal_control_pack(),
            run_root,
            300,
            repo_root=self.repo_root,
        )

        run_dir = self.repo_root / "phase5" / "runs" / "t_main_1"
        self.assertTrue((run_dir / "stage_verdict.json").exists())
        self.assertTrue((run_dir / "support_summary.json").exists())
        self.assertTrue((run_dir / "venue_capability_matrix.json").exists())

    def test_tranche_stage_overlay_source_defaults_to_control_pack(self):
        defaults = _minimal_control_pack()["execution_defaults"]
        path = self.mod.tranche_stage_overlay_source({"id": "t_main_1"}, defaults)
        self.assertEqual(path, Path("/tmp/stage_overlay_live.env"))

    def test_tranche_stage_overlay_source_allows_override(self):
        defaults = _minimal_control_pack()["execution_defaults"]
        path = self.mod.tranche_stage_overlay_source(
            {"execution": {"stage_overlay_source": "/tmp/override.env"}},
            defaults,
        )
        self.assertEqual(path, Path("/tmp/override.env"))

    def test_tranche_stage_overlay_source_allows_env_diff_override(self):
        defaults = _minimal_control_pack()["execution_defaults"]
        path = self.mod.tranche_stage_overlay_source(
            {"env_diff": {"stage_overlay_source": "/tmp/env-diff-override.env"}},
            defaults,
        )
        self.assertEqual(path, Path("/tmp/env-diff-override.env"))

    def test_tranche_stage_overlay_source_validates_value(self):
        defaults = _minimal_control_pack()["execution_defaults"]
        with self.assertRaises(ValueError):
            self.mod.tranche_stage_overlay_source({"execution": {"stage_overlay_source": ""}}, defaults)
        with self.assertRaises(ValueError):
            self.mod.tranche_stage_overlay_source({"env_diff": {"stage_overlay_source": ""}}, defaults)

    def test_current_extended_rebootstrap_sleep_cap_overlay_preserves_stale_margin(self):
        repo_root = Path(__file__).resolve().parents[1]
        queue = yaml.safe_load((repo_root / "phase5" / "queue.yaml").read_text(encoding="utf-8"))
        tranches = queue.get("serialized_mainline", []) + queue.get("parallel_support_tracks", [])
        tranche = next(
            item
            for item in tranches
            if item.get("id") == "phase5_all5_current_surface_extended_degraded_rebootstrap_sleep_cap_requal"
        )
        env = {key: str(value) for key, value in tranche.get("env_diff", {}).items()}

        self.assertEqual(env["PARAPHINA_EXTENDED_DEGRADED_REBOOTSTRAP_MAX_SLEEP_MS"], "500")
        cap_ms = int(env["PARAPHINA_EXTENDED_DEGRADED_REBOOTSTRAP_MAX_SLEEP_MS"])
        state_stale_ms = int(env["PARAPHINA_EXTENDED_STATE_STALE_MS_OVERRIDE"])
        fallback_after_ms = int(env["PARAPHINA_EXTENDED_POST_PUBLISH_FALLBACK_AFTER_MS"])
        self.assertLessEqual(cap_ms, 750)
        self.assertLess(cap_ms, state_stale_ms)
        self.assertLess(fallback_after_ms + cap_ms, state_stale_ms)

    def test_runtime_install_required_false_for_same_binary(self):
        runtime = Path("/tmp/runtime")
        self.assertFalse(self.mod.runtime_install_required(runtime, runtime))

    def test_runtime_install_required_true_for_different_binary(self):
        self.assertTrue(
            self.mod.runtime_install_required(Path("/tmp/runtime_candidate"), Path("/tmp/runtime_live"))
        )

    def test_candidate_runtime_binary_path_defaults_to_live_runtime_for_non_code_tranche(self):
        defaults = _minimal_control_pack()["execution_defaults"]
        tranche = {"id": "t_main_1", "candidate": {"change_scope": {"files": ["tools/telemetry.py"]}}}
        path = self.mod.candidate_runtime_binary_path(tranche, defaults, self.repo_root)
        self.assertEqual(path, Path("/opt/paraphina/paraphina_live"))

    def test_candidate_runtime_binary_path_infers_repo_build_for_code_tranche(self):
        defaults = _minimal_control_pack()["execution_defaults"]
        src = self.repo_root / "paraphina" / "src" / "live" / "runner.rs"
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_text("// runner\n", encoding="utf-8")
        build = self.repo_root / "target" / "release" / "paraphina_live"
        build.parent.mkdir(parents=True, exist_ok=True)
        build.write_text("binary", encoding="utf-8")
        path = self.mod.candidate_runtime_binary_path(
            {"id": "t_main_1", "candidate": {"change_scope": {"files": [str(src)]}}},
            defaults,
            self.repo_root,
        )
        self.assertEqual(path, build)

    def test_candidate_runtime_binary_path_rejects_stale_repo_build_for_code_tranche(self):
        defaults = _minimal_control_pack()["execution_defaults"]
        src = self.repo_root / "paraphina" / "src" / "live" / "runner.rs"
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_text("// runner v1\n", encoding="utf-8")
        build = self.repo_root / "target" / "release" / "paraphina_live"
        build.parent.mkdir(parents=True, exist_ok=True)
        build.write_text("binary", encoding="utf-8")
        src.write_text("// runner v2\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.mod.candidate_runtime_binary_path(
                {"id": "t_main_1", "candidate": {"change_scope": {"files": [str(src)]}}},
                defaults,
                self.repo_root,
            )

    def test_candidate_runtime_binary_path_honors_explicit_override(self):
        defaults = _minimal_control_pack()["execution_defaults"]
        path = self.mod.candidate_runtime_binary_path(
            {"candidate": {"runtime_binary": "/tmp/custom_binary"}},
            defaults,
            self.repo_root,
        )
        self.assertEqual(path, Path("/tmp/custom_binary"))

    def test_curl_health_retries_through_transient_failure(self):
        responses = [
            subprocess.CalledProcessError(7, ["curl", "-fsS", "http://127.0.0.1:9898/health/detail"]),
            mock.Mock(stdout='{"healthy": true}\n'),
        ]

        def fake_run(*args, **kwargs):
            response = responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response

        with mock.patch.object(self.mod.subprocess, "run", side_effect=fake_run) as run_mock:
            with mock.patch.object(self.mod.time, "sleep") as sleep_mock:
                payload = self.mod.curl_health("http://127.0.0.1:9898/health/detail", attempts=2, delay_sec=0.5)

        self.assertEqual(payload, '{"healthy": true}\n')
        self.assertEqual(run_mock.call_count, 2)
        sleep_mock.assert_called_once_with(0.5)

    def test_curl_health_rejects_invalid_attempt_count(self):
        with self.assertRaises(ValueError):
            self.mod.curl_health("http://127.0.0.1:9898/health/detail", attempts=0)

    def test_ensure_shadow_health_retries_when_systemd_is_still_active(self):
        responses = [
            subprocess.CalledProcessError(7, ["curl", "-fsS", "http://127.0.0.1:9898/health/detail"]),
            '{"healthy": true, "ready": true, "trade_mode": "shadow"}',
        ]

        def fake_curl_health(url, attempts=1, delay_sec=0.0):
            response = responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response

        with mock.patch.object(self.mod, "curl_health", side_effect=fake_curl_health) as curl_mock, \
            mock.patch.object(
                self.mod,
                "systemd_show",
                return_value="ActiveState=active\nSubState=running\nNRestarts=0\n",
            ) as systemd_mock:
            payload = self.mod.ensure_shadow_health()

        self.assertEqual(payload["trade_mode"], "shadow")
        self.assertEqual(curl_mock.call_count, 2)
        systemd_mock.assert_called_once_with("paraphina_live")

    def test_recover_live_closeout_replays_analyzer_when_summary_missing(self):
        run_root = self.repo_root / "promotion_runs" / "t_main_1_5m_canary_20260401T000000Z" / "live_canary"
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "telemetry_bounded.jsonl").write_text('{"execution_mode":"live"}\n', encoding="utf-8")
        (run_root / "guard.log").write_text(
            "2026-04-01T00:05:00Z INFO guard_window_complete_restoring_shadow\n",
            encoding="utf-8",
        )
        (run_root / "health_post.json").write_text(
            json.dumps({"healthy": True, "ready": True, "trade_mode": "shadow"}) + "\n",
            encoding="utf-8",
        )
        (run_root / "live_metrics.json").write_text("{}\n", encoding="utf-8")
        (run_root / "systemd_post.txt").write_text(
            "ActiveState=active\nSubState=running\nNRestarts=0\n",
            encoding="utf-8",
        )

        def fake_run_logged_command(cmd):
            if cmd.label == "telemetry_analyzer_recovery":
                (run_root / "live_segment_summary.json").write_text(
                    json.dumps(
                        {
                            "tick_count": 10,
                            "first_ts_ms": 1775000000000,
                            "last_ts_ms": 1775000300000,
                            "pnl_validity": {
                                "final_pnl_total": 0.1,
                                "final_pnl_realised": 0.0,
                                "final_pnl_unrealised": 0.1,
                                "final_q_global_tao": 0.0,
                                "mm_place_total": 1,
                                "mm_keep_total": 2,
                                "mm_replace_total": 0,
                            },
                        }
                    ) + "\n",
                    encoding="utf-8",
                )
                (run_root / "live_metrics.json").write_text(
                    json.dumps({"fills": {"total_count": 0, "total_base": 0.0}}) + "\n",
                    encoding="utf-8",
                )
                (run_root / "telemetry_report_live_segment.md").write_text("report\n", encoding="utf-8")
                return None
            raise AssertionError(f"unexpected command label {cmd.label}")

        with mock.patch.object(self.mod, "run_logged_command", side_effect=fake_run_logged_command):
            closeout_path = self.mod.recover_live_closeout("t_main_1", repo_root=self.repo_root, run_root=run_root, duration_sec=300)

        self.assertTrue(closeout_path.exists())
        closeout = json.loads(closeout_path.read_text(encoding="utf-8"))
        self.assertTrue(closeout["summary_exists"])
        self.assertTrue(closeout["report_exists"])
        self.assertTrue(closeout["analyzer_recovery_attempted"])
        self.assertTrue(closeout["analyzer_recovery_succeeded"])
        self.assertEqual(closeout["final_pnl_total"], 0.1)
        manifest = yaml.safe_load((self.repo_root / "phase5" / "runs" / "t_main_1" / "latest_run.yaml").read_text(encoding="utf-8"))
        self.assertEqual(manifest["closeout_bundle_path"], str(closeout_path))

    def test_recover_live_closeout_marks_partial_when_summary_unavailable(self):
        run_root = self.repo_root / "promotion_runs" / "t_main_1_20m_soak_20260401T000000Z" / "live_canary"
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "guard.log").write_text(
            "2026-04-01T00:20:00Z INFO guard_window_complete_restoring_shadow\n",
            encoding="utf-8",
        )
        (run_root / "guard_result.json").write_text(
            json.dumps({"exit_code": 1}) + "\n",
            encoding="utf-8",
        )
        (run_root / "health_post.json").write_text(
            json.dumps({"healthy": True, "ready": True, "trade_mode": "live"}) + "\n",
            encoding="utf-8",
        )
        (run_root / "systemd_post.txt").write_text(
            "ActiveState=active\nSubState=running\nNRestarts=2\n",
            encoding="utf-8",
        )

        closeout_path = self.mod.recover_live_closeout("t_main_1", repo_root=self.repo_root, run_root=run_root, duration_sec=1200)

        closeout = json.loads(closeout_path.read_text(encoding="utf-8"))
        self.assertEqual(closeout["closeout_completeness"], "partial")
        self.assertFalse(closeout["closeout_contract_complete"])
        self.assertEqual(closeout["guard_exit_code"], 1)
        self.assertFalse(closeout["summary_exists"])
        self.assertEqual(closeout["trade_mode_post"], "live")
        self.assertTrue(closeout["restore_required"])
        self.assertIn("trade_mode_post_not_shadow", closeout["restore_required_reasons"])
        manifest = yaml.safe_load((self.repo_root / "phase5" / "runs" / "t_main_1" / "latest_run.yaml").read_text(encoding="utf-8"))
        self.assertEqual(manifest["run_state"], "restore_required")
        self.assertIn("post_restore_direct_venue_audit_not_clean", manifest["restore_required_reasons"])

    def test_recover_live_closeout_captures_guard_intervention(self):
        run_root = self.repo_root / "promotion_runs" / "t_main_1_60m_qual_20260401T000000Z" / "live_canary"
        run_root.mkdir(parents=True, exist_ok=True)
        self.mod.record_result(
            "t_main_1",
            "hold",
            repo_root=self.repo_root,
            observed_blocker_family="stale_restart",
            precondition_failed=True,
            credit_earned="none",
        )
        (run_root / "telemetry_bounded.jsonl").write_text('{"execution_mode":"live"}\n', encoding="utf-8")
        (run_root / "live_segment_summary.json").write_text(
            json.dumps(
                {
                    "tick_count": 100,
                    "first_ts_ms": 1775000000000,
                    "last_ts_ms": 1775003600000,
                    "pnl_validity": {"final_pnl_total": 0.0, "final_q_global_tao": 0.01},
                }
            ) + "\n",
            encoding="utf-8",
        )
        (run_root / "live_metrics.json").write_text(
            json.dumps({"fills": {"total_count": 0, "total_base": 0.0}}) + "\n",
            encoding="utf-8",
        )
        (run_root / "telemetry_report_live_segment.md").write_text("report\n", encoding="utf-8")
        (run_root / "guard.log").write_text(
            "2026-04-01T01:00:00Z CRITICAL triggered_intervention reason=kill_events_present\n"
            "2026-04-01T01:00:08Z INFO post_rollback_venue_audit_clean\n",
            encoding="utf-8",
        )
        (run_root / "health_post.json").write_text(
            json.dumps({"healthy": True, "ready": True, "trade_mode": "shadow", "kill_events_present": False}) + "\n",
            encoding="utf-8",
        )
        (run_root / "systemd_post.txt").write_text(
            "ActiveState=active\nSubState=running\nNRestarts=0\n",
            encoding="utf-8",
        )

        closeout_path = self.mod.recover_live_closeout("t_main_1", repo_root=self.repo_root, run_root=run_root, duration_sec=3600)
        closeout = json.loads(closeout_path.read_text(encoding="utf-8"))
        self.assertTrue(closeout["guard_intervened"])
        self.assertEqual(closeout["guard_intervention_reason"], "kill_events_present")
        self.assertTrue(closeout["post_rollback_venue_audit_clean"])
        self.assertEqual(closeout["segment_start_utc"], "2026-03-31T23:33:20Z")
        self.assertEqual(closeout["observed_primary_blocker_family"], "stale_restart")
        self.assertTrue(closeout["precondition_failed"])
        self.assertEqual(closeout["credit_earned"], "none")
        self.assertIn("surface_id", closeout)
        self.assertFalse(closeout["restore_required"])
        manifest = yaml.safe_load((self.repo_root / "phase5" / "runs" / "t_main_1" / "latest_run.yaml").read_text(encoding="utf-8"))
        self.assertEqual(manifest["run_state"], "recovered_closeout")

    def test_guard_closeout_info_captures_post_cleanup_clean_separately(self):
        guard_path = self.repo_root / "guard.log"
        guard_path.write_text(
            "2026-04-12T11:54:51Z INFO guard_window_complete_restoring_shadow\n"
            "2026-04-12T11:55:17Z INFO post_cleanup_venue_audit_clean\n",
            encoding="utf-8",
        )

        closeout = self.mod.guard_closeout_info(guard_path)

        self.assertTrue(closeout["guard_window_completed"])
        self.assertFalse(closeout["post_rollback_venue_audit_clean"])
        self.assertTrue(closeout["post_cleanup_venue_audit_clean"])

    def test_guard_closeout_info_tracks_first_dirty_audit_separately_from_cleanup_recovery(self):
        guard_path = self.repo_root / "guard.log"
        guard_path.write_text(
            "2026-04-12T20:31:53Z INFO guard_window_complete_restoring_shadow\n"
            "2026-04-12T20:32:01Z INFO pre_restore_cleanup_triggered payload={\"ok\":false}\n"
            "2026-04-12T20:32:05Z INFO live_cleanup_pass_0_stdout='{\"result\":\"success\",\"total_estimated_cleanup_cost_usd\":0.707216}'\n"
            "2026-04-12T20:32:09Z INFO pre_restore_cleanup_venue_audit_clean\n",
            encoding="utf-8",
        )

        closeout = self.mod.guard_closeout_info(guard_path)

        self.assertTrue(closeout["guard_window_completed"])
        self.assertFalse(closeout["first_pre_restore_venue_audit_clean"])
        self.assertTrue(closeout["pre_restore_cleanup_required"])
        self.assertAlmostEqual(closeout["pre_restore_cleanup_cost_usd"], 0.707216)
        self.assertTrue(closeout["pre_restore_venue_audit_clean"])
        self.assertTrue(closeout["pre_restore_cleanup_venue_audit_clean"])

    def test_guard_closeout_info_parses_text_cleanup_summary_cost(self):
        guard_path = self.repo_root / "guard.log"
        guard_path.write_text(
            "2026-04-12T20:31:53Z INFO guard_window_complete_restoring_shadow\n"
            "2026-04-12T20:32:01Z INFO pre_restore_cleanup_triggered payload={\"ok\":false}\n"
            "2026-04-12T20:32:05Z INFO live_cleanup_pass_0_stdout='cleanup venue=extended kind=reduce_only_ioc side=Some(Buy) size=0.02 price=2381.1255 order_id=Some(\"2048532487780954112\") est_cost_usd=0.47150999999999843\\ncleanup result=success venues_touched=extended total_estimated_cleanup_cost_usd=0.47150999999999843 settle_ms=2000'\n"
            "2026-04-12T20:32:09Z INFO pre_restore_cleanup_venue_audit_clean\n",
            encoding="utf-8",
        )

        closeout = self.mod.guard_closeout_info(guard_path)

        self.assertTrue(closeout["pre_restore_cleanup_required"])
        self.assertAlmostEqual(closeout["pre_restore_cleanup_cost_usd"], 0.47150999999999843)

    def test_recover_live_closeout_augments_metrics_with_paradex_stderr_audits(self):
        run_root = self.repo_root / "promotion_runs" / "t_main_1_20m_soak_20260401T000000Z" / "live_canary"
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "telemetry_bounded.jsonl").write_text('{"execution_mode":"live"}\n', encoding="utf-8")
        (run_root / "live_segment_summary.json").write_text(
            json.dumps(
                {
                    "tick_count": 100,
                    "first_ts_ms": 1775000000000,
                    "last_ts_ms": 1775001200000,
                    "pnl_validity": {"final_pnl_total": 0.0, "final_q_global_tao": 0.0},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (run_root / "live_metrics.json").write_text(
            json.dumps({"fills": {"total_count": 0, "total_base": 0.0}}) + "\n",
            encoding="utf-8",
        )
        (run_root / "telemetry_report_live_segment.md").write_text("report\n", encoding="utf-8")
        (run_root / "guard.log").write_text(
            "2026-04-01T00:20:00Z INFO guard_window_complete_restoring_shadow\n"
            "2026-04-01T00:20:08Z INFO pre_restore_venue_audit_clean\n"
            "2026-04-01T00:20:10Z INFO post_rollback_venue_audit_clean\n",
            encoding="utf-8",
        )
        (run_root / "paraphina_live.err.segment").write_text(
            "WS_AUDIT venue=paradex component=profile_usage action=fetched token_usage=interactive auth_source=jwt_cmd\n"
            "WS_AUDIT venue=paradex component=order_flags action=place token_usage=interactive instruction=POST_ONLY flags=none\n"
            "WS_AUDIT venue=paradex component=fill_flags token_usage=interactive flags=interactive,fastfill\n"
            "WS_AUDIT venue=paradex component=interactive_top count=1 feed_type=interactive seq_no=42 "
            "best_bid_api_price=3000.1 best_bid_api_size=1.2 "
            "best_bid_interactive_price=3000.2 best_bid_interactive_size=0.8 "
            "best_ask_api_price=3001.1 best_ask_api_size=1.5 "
            "best_ask_interactive_price=3001.2 best_ask_interactive_size=0.6\n"
            "WS_AUDIT venue=paradex component=ui_book_truth source=api status=ok token_usage=interactive error_class=none "
            "seq_no=100 last_updated_at_ms=1775315000123 bid_px=3000.1 bid_sz=1.2 ask_px=3001.1 ask_sz=1.5 "
            "best_bid_api_px=3000.1 best_bid_api_sz=1.2 best_bid_interactive_px=na best_bid_interactive_sz=na "
            "best_ask_api_px=3001.1 best_ask_api_sz=1.5 best_ask_interactive_px=na best_ask_interactive_sz=na\n"
            "WS_AUDIT venue=paradex component=ui_book_truth source=interactive status=ok token_usage=interactive error_class=none "
            "seq_no=101 last_updated_at_ms=1775315000223 bid_px=3000.2 bid_sz=0.8 ask_px=3001.0 ask_sz=0.6 "
            "best_bid_api_px=3000.1 best_bid_api_sz=1.2 best_bid_interactive_px=3000.2 best_bid_interactive_sz=0.8 "
            "best_ask_api_px=3001.1 best_ask_api_sz=1.5 best_ask_interactive_px=3001.0 best_ask_interactive_sz=0.6\n"
            "WS_AUDIT venue=paradex component=ui_touch_reference action=applied source_kind=split "
            "orig_bid=3000.1 orig_bid_sz=1.2 orig_ask=3001.1 orig_ask_sz=1.5 "
            "adj_bid=3000.2 adj_bid_sz=0.8 adj_ask=3001.0 adj_ask_sz=0.6\n",
            encoding="utf-8",
        )
        (run_root / "health_post.json").write_text(
            json.dumps({"healthy": True, "ready": True, "trade_mode": "shadow"}) + "\n",
            encoding="utf-8",
        )
        (run_root / "live_metrics.json").write_text("{}\n", encoding="utf-8")
        (run_root / "systemd_post.txt").write_text(
            "ActiveState=active\nSubState=running\nNRestarts=0\n",
            encoding="utf-8",
        )

        self.mod.recover_live_closeout("t_main_1", repo_root=self.repo_root, run_root=run_root, duration_sec=1200)

        metrics = json.loads((run_root / "live_metrics.json").read_text(encoding="utf-8"))
        self.assertTrue(metrics["paradex_profile_usage_summary"]["interactive_token_usage_observed"])
        self.assertEqual(metrics["paradex_order_flag_summary"]["instructions"]["POST_ONLY"], 1)
        self.assertEqual(metrics["paradex_fill_flag_summary"]["flags"]["fastfill"], 1)
        self.assertEqual(metrics["paradex_interactive_top_summary"]["last_seq_no"], 42)
        self.assertAlmostEqual(
            metrics["paradex_interactive_top_summary"]["last_top"]["best_bid_interactive_price"],
            3000.2,
        )
        self.assertTrue(metrics["paradex_ui_book_truth_summary"]["observed"])
        self.assertEqual(metrics["paradex_ui_book_truth_summary"]["api_records"], 1)
        self.assertEqual(metrics["paradex_ui_book_truth_summary"]["interactive_records"], 1)
        self.assertEqual(metrics["paradex_ui_book_truth_summary"]["last_seq_gap"], 1)
        self.assertEqual(metrics["paradex_ui_book_truth_summary"]["nonzero_gap_records"], 1)
        self.assertAlmostEqual(
            metrics["paradex_ui_book_truth_summary"]["last_split_top"]["best_bid_interactive_px"],
            3000.2,
        )
        self.assertTrue(metrics["paradex_ui_touch_reference_summary"]["observed"])
        self.assertEqual(metrics["paradex_ui_touch_reference_summary"]["applied_count"], 1)
        self.assertEqual(
            metrics["paradex_ui_touch_reference_summary"]["source_kind_counts"]["split"], 1
        )

    def test_recover_live_closeout_parses_paradex_top_level_fallback_touch_reference(self):
        control_pack = _minimal_control_pack()
        _write_yaml(self.repo_root / "phase5" / "control_pack.yaml", control_pack)

        run_root = self.repo_root / "promotion_runs" / "t_main_1_20m_soak_20260401T000000Z" / "live_canary"
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "guard.log").write_text(
            "2026-04-01T00:20:00Z INFO guard_window_complete_restoring_shadow\n",
            encoding="utf-8",
        )
        (run_root / "paraphina_live.err.segment").write_text(
            "PARADEX_INTERACTIVE_PUBLIC_TOP count=1 source=interactive_orderbook "
            "top_source=interactive_top_level_fallback bid=3000.2 bid_sz=0.8 ask=3001.0 ask_sz=0.6\n"
            "WS_AUDIT venue=paradex component=ui_book_truth source=api status=ok token_usage=interactive error_class=none "
            "seq_no=200 last_updated_at_ms=1775315001123 bid_px=3000.1 bid_sz=1.2 ask_px=3001.1 ask_sz=1.5 "
            "best_bid_api_px=3000.1 best_bid_api_sz=1.2 best_bid_interactive_px=na best_bid_interactive_sz=na "
            "best_ask_api_px=3001.1 best_ask_api_sz=1.5 best_ask_interactive_px=na best_ask_interactive_sz=na\n"
            "WS_AUDIT venue=paradex component=ui_book_truth source=interactive status=ok token_usage=interactive error_class=none "
            "seq_no=201 last_updated_at_ms=1775315001223 bid_px=3000.2 bid_sz=0.8 ask_px=3001.0 ask_sz=0.6 "
            "best_bid_api_px=3000.1 best_bid_api_sz=1.2 best_bid_interactive_px=na best_bid_interactive_sz=na "
            "best_ask_api_px=3001.1 best_ask_api_sz=1.5 best_ask_interactive_px=na best_ask_interactive_sz=na\n"
            "WS_AUDIT venue=paradex component=ui_touch_reference action=applied source_kind=top_level_fallback "
            "orig_bid=3000.1 orig_bid_sz=1.2 orig_ask=3001.1 orig_ask_sz=1.5 "
            "adj_bid=3000.2 adj_bid_sz=0.8 adj_ask=3001.0 adj_ask_sz=0.6\n",
            encoding="utf-8",
        )
        (run_root / "health_post.json").write_text(
            json.dumps({"healthy": True, "ready": True, "trade_mode": "shadow"}) + "\n",
            encoding="utf-8",
        )
        (run_root / "live_metrics.json").write_text("{}\n", encoding="utf-8")
        (run_root / "systemd_post.txt").write_text(
            "ActiveState=active\nSubState=running\nNRestarts=0\n",
            encoding="utf-8",
        )

        self.mod.recover_live_closeout("t_main_1", repo_root=self.repo_root, run_root=run_root, duration_sec=1200)

        metrics = json.loads((run_root / "live_metrics.json").read_text(encoding="utf-8"))
        self.assertFalse(metrics["paradex_ui_book_truth_summary"]["interactive_fields_present"])
        self.assertTrue(metrics["paradex_ui_book_truth_summary"]["interactive_top_level_present"])
        self.assertEqual(metrics["paradex_interactive_top_summary"]["records"], 1)
        self.assertEqual(metrics["paradex_interactive_top_summary"]["public_records"], 1)
        self.assertEqual(
            metrics["paradex_interactive_top_summary"]["public_top_source_counts"][
                "interactive_top_level_fallback"
            ],
            1,
        )
        self.assertTrue(metrics["paradex_interactive_top_summary"]["interactive_top_level_fallback_present"])
        self.assertAlmostEqual(
            metrics["paradex_interactive_top_summary"]["last_public_top"]["bid_px"],
            3000.2,
        )
        self.assertTrue(metrics["paradex_ui_touch_reference_summary"]["observed"])
        self.assertEqual(metrics["paradex_ui_touch_reference_summary"]["applied_count"], 1)
        self.assertEqual(
            metrics["paradex_ui_touch_reference_summary"]["source_kind_counts"]["top_level_fallback"],
            1,
        )

    def test_latest_run_requires_recovery_detects_live_started_without_closeout(self):
        run_root = self.repo_root / "promotion_runs" / "t_main_1_60m_qual_20260401T000000Z" / "live_canary"
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "guard.log").write_text("guard\n", encoding="utf-8")
        _write_yaml(
            self.repo_root / "phase5" / "runs" / "t_main_1" / "latest_run.yaml",
            {
                "run_root": str(run_root),
                "duration_sec": 3600,
                "run_state": "live_started",
            },
        )

        needs_recovery, recovered_root, duration_sec = self.mod.latest_run_requires_recovery(
            self.repo_root, "t_main_1"
        )

        self.assertTrue(needs_recovery)
        self.assertEqual(recovered_root, run_root)
        self.assertEqual(duration_sec, 3600)

    def test_admission_check_blocks_restore_required_latest_run(self):
        run_root = self.repo_root / "promotion_runs" / "t_main_1_60m_qual_20260401T000000Z" / "live_canary"
        run_root.mkdir(parents=True, exist_ok=True)
        _write_yaml(
            self.repo_root / "phase5" / "runs" / "t_main_1" / "latest_run.yaml",
            {
                "run_root": str(run_root),
                "duration_sec": 3600,
                "run_state": "restore_required",
                "restore_required_reasons": ["trade_mode_post_not_shadow"],
            },
        )

        with self.assertRaisesRegex(RuntimeError, "restore_required"):
            self.mod.admission_check("t_main_1", 300, repo_root=self.repo_root)

    def test_record_manual_recovery_verification_clears_restore_required_without_promotion_credit(self):
        run_root = self.repo_root / "promotion_runs" / "t_main_1_60m_qual_20260401T000000Z" / "live_canary"
        run_root.mkdir(parents=True, exist_ok=True)
        audit_path = run_root / "manual_recovery_venue_audit.json"
        audit_path.write_text(
            json.dumps(
                {
                    "ok": True,
                    "violations": [],
                    "results": [
                        {
                            "venue": "aster",
                            "ok": True,
                            "position_base": 0.0,
                            "open_order_count": 0,
                            "open_order_count_known": True,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        _write_yaml(
            self.repo_root / "phase5" / "runs" / "t_main_1" / "latest_run.yaml",
            {
                "run_root": str(run_root),
                "duration_sec": 3600,
                "run_state": "restore_required",
                "restore_required_reasons": ["post_restore_direct_venue_audit_not_clean"],
            },
        )

        verification_path = self.mod.record_manual_recovery_verification(
            "t_main_1",
            audit_path,
            self.repo_root,
            verify_host=False,
        )

        manifest = yaml.safe_load(
            (self.repo_root / "phase5" / "runs" / "t_main_1" / "latest_run.yaml").read_text(
                encoding="utf-8"
            )
        )
        verification = json.loads(verification_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["run_state"], "manual_recovery_verified")
        self.assertEqual(manifest["manual_recovery_promotional_credit"], "none")
        self.assertIsNone(self.mod.latest_run_restore_required_manifest(self.repo_root, "t_main_1"))
        self.assertEqual(verification["previous_run_state"], "restore_required")
        self.assertEqual(verification["promotion_credit"], "none")
        self.mod.ensure_latest_run_not_restore_required(self.repo_root, "t_main_1")

    def test_record_manual_recovery_verification_rejects_dirty_audit(self):
        run_root = self.repo_root / "promotion_runs" / "t_main_1_60m_qual_20260401T000000Z" / "live_canary"
        run_root.mkdir(parents=True, exist_ok=True)
        audit_path = run_root / "manual_recovery_venue_audit.json"
        audit_path.write_text(
            json.dumps(
                {
                    "ok": False,
                    "violations": ["aster dirty"],
                    "results": [
                        {
                            "venue": "aster",
                            "ok": False,
                            "position_base": 0.01,
                            "open_order_count": 0,
                            "open_order_count_known": True,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        _write_yaml(
            self.repo_root / "phase5" / "runs" / "t_main_1" / "latest_run.yaml",
            {
                "run_root": str(run_root),
                "duration_sec": 3600,
                "run_state": "restore_required",
                "restore_required_reasons": ["post_restore_direct_venue_audit_not_clean"],
            },
        )

        with self.assertRaisesRegex(RuntimeError, "manual recovery audit is not clean"):
            self.mod.record_manual_recovery_verification(
                "t_main_1",
                audit_path,
                self.repo_root,
                verify_host=False,
            )

        manifest = yaml.safe_load(
            (self.repo_root / "phase5" / "runs" / "t_main_1" / "latest_run.yaml").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(manifest["run_state"], "restore_required")

    def test_spawn_lanes_blocks_restore_required_latest_run(self):
        run_root = self.repo_root / "promotion_runs" / "t_main_1_60m_qual_20260401T000000Z" / "live_canary"
        run_root.mkdir(parents=True, exist_ok=True)
        _write_yaml(
            self.repo_root / "phase5" / "runs" / "t_main_1" / "latest_run.yaml",
            {
                "run_root": str(run_root),
                "duration_sec": 3600,
                "run_state": "restore_required",
                "restore_required_reasons": ["post_restore_direct_venue_audit_not_clean"],
            },
        )

        with self.assertRaisesRegex(RuntimeError, "restore_required"):
            self.mod.spawn_lanes("t_main_1", repo_root=self.repo_root)

    def test_record_result_auto_recovers_latest_run_before_writing_verdict(self):
        run_root = self.repo_root / "promotion_runs" / "t_main_1_60m_qual_20260401T000000Z" / "live_canary"
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "guard.log").write_text("guard\n", encoding="utf-8")
        _write_yaml(
            self.repo_root / "phase5" / "runs" / "t_main_1" / "latest_run.yaml",
            {
                "run_root": str(run_root),
                "duration_sec": 3600,
                "run_state": "live_started",
            },
        )

        with mock.patch.object(
            self.mod,
            "recover_live_closeout",
            return_value=run_root / "live_closeout_bundle.json",
        ) as recover_mock:
            self.mod.record_result(
                "t_main_1",
                "hold",
                repo_root=self.repo_root,
                observed_blocker_family="stale_restart",
            )

        recover_mock.assert_called_once_with(
            tranche_id="t_main_1",
            repo_root=self.repo_root,
            run_root=run_root,
            duration_sec=3600,
        )

    def test_cli_render_status_auto_recovers_pending_live_started_runs(self):
        run_root = self.repo_root / "promotion_runs" / "t_main_1_60m_qual_20260401T000000Z" / "live_canary"
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "guard.log").write_text("guard\n", encoding="utf-8")
        _write_yaml(
            self.repo_root / "phase5" / "runs" / "t_main_1" / "latest_run.yaml",
            {
                "run_root": str(run_root),
                "duration_sec": 3600,
                "run_state": "live_started",
            },
        )
        args = mock.Mock(repo_root=str(self.repo_root))

        with mock.patch.object(
            self.mod,
            "recover_live_closeout",
            return_value=run_root / "live_closeout_bundle.json",
        ) as recover_mock:
            rc = self.mod.cli_render_status(args)

        self.assertEqual(rc, 0)
        recover_mock.assert_called_once()

    def test_recover_live_closeout_rebuilds_bounded_slice_from_source_offsets(self):
        control_pack = _minimal_control_pack()
        telemetry_source = Path(control_pack["execution_defaults"]["telemetry_path"])
        telemetry_source.parent.mkdir(parents=True, exist_ok=True)
        telemetry_source.write_text(
            '{"execution_mode":"shadow"}\n'
            '{"execution_mode":"live","tick":1}\n'
            '{"execution_mode":"live","tick":2}\n'
            '{"execution_mode":"shadow"}\n',
            encoding="utf-8",
        )
        _write_yaml(self.repo_root / "phase5" / "control_pack.yaml", control_pack)

        run_root = self.repo_root / "promotion_runs" / "t_main_1_60m_qual_20260401T000000Z" / "live_canary"
        run_root.mkdir(parents=True, exist_ok=True)
        content = telemetry_source.read_bytes()
        live_start = content.index(b'{"execution_mode":"live","tick":1}\n')
        live_end = content.index(b'{"execution_mode":"shadow"}\n', live_start)
        (run_root / "telemetry_offset_pre.txt").write_text(str(live_start), encoding="utf-8")
        (run_root / "telemetry_offset_post.txt").write_text(str(live_end), encoding="utf-8")
        (run_root / "guard.log").write_text(
            "2026-04-01T00:05:00Z INFO guard_window_complete_restoring_shadow\n",
            encoding="utf-8",
        )
        (run_root / "health_post.json").write_text(
            json.dumps({"healthy": True, "ready": True, "trade_mode": "shadow"}) + "\n",
            encoding="utf-8",
        )
        (run_root / "systemd_post.txt").write_text(
            "ActiveState=active\nSubState=running\nNRestarts=0\n",
            encoding="utf-8",
        )

        def fake_run_logged_command(cmd):
            if cmd.label in {"telemetry_analyzer_recovery", "telemetry_analyzer_recovery_from_source_slice"}:
                (run_root / "live_segment_summary.json").write_text(
                    json.dumps(
                        {
                            "tick_count": 2,
                            "first_ts_ms": 1775000000000,
                            "last_ts_ms": 1775000100000,
                            "pnl_validity": {
                                "final_pnl_total": 0.0,
                                "final_pnl_realised": 0.0,
                                "final_pnl_unrealised": 0.0,
                                "final_q_global_tao": 0.0,
                            },
                        }
                    ) + "\n",
                    encoding="utf-8",
                )
                (run_root / "live_metrics.json").write_text(
                    json.dumps({"fills": {"total_count": 0, "total_base": 0.0}}) + "\n",
                    encoding="utf-8",
                )
                (run_root / "telemetry_report_live_segment.md").write_text("report\n", encoding="utf-8")
                return None
            raise AssertionError(f"unexpected command label {cmd.label}")

        with mock.patch.object(self.mod, "run_logged_command", side_effect=fake_run_logged_command):
            closeout_path = self.mod.recover_live_closeout("t_main_1", repo_root=self.repo_root, run_root=run_root, duration_sec=3600)

        closeout = json.loads(closeout_path.read_text(encoding="utf-8"))
        self.assertTrue((run_root / "telemetry_bounded.jsonl").exists())
        self.assertTrue(closeout["source_slice_recovery_attempted"])
        self.assertTrue(closeout["source_slice_recovery_succeeded"])
        self.assertTrue(closeout["analyzer_recovery_succeeded"])

    def test_recover_live_closeout_rebuilds_truncated_bounded_slice_from_source_offsets(self):
        control_pack = _minimal_control_pack()
        telemetry_source = Path(control_pack["execution_defaults"]["telemetry_path"])
        telemetry_source.parent.mkdir(parents=True, exist_ok=True)
        telemetry_source.write_text(
            '{"execution_mode":"shadow"}\n'
            '{"execution_mode":"live","tick":1}\n'
            '{"execution_mode":"live","tick":2}\n'
            '{"execution_mode":"shadow"}\n',
            encoding="utf-8",
        )
        _write_yaml(self.repo_root / "phase5" / "control_pack.yaml", control_pack)

        run_root = self.repo_root / "promotion_runs" / "t_main_1_60m_qual_20260401T000000Z" / "live_canary"
        run_root.mkdir(parents=True, exist_ok=True)
        content = telemetry_source.read_bytes()
        live_start = content.index(b'{"execution_mode":"live","tick":1}\n')
        live_end = content.index(b'{"execution_mode":"shadow"}\n', live_start)
        (run_root / "telemetry_offset_pre.txt").write_text(str(live_start), encoding="utf-8")
        (run_root / "telemetry_offset_post.txt").write_text(str(live_end), encoding="utf-8")
        (run_root / "telemetry_bounded.jsonl").write_text('{"execution_mode":"live","tick":2}\n', encoding="utf-8")
        (run_root / "guard.log").write_text(
            "2026-04-01T00:05:00Z INFO guard_window_complete_restoring_shadow\n",
            encoding="utf-8",
        )
        (run_root / "health_post.json").write_text(
            json.dumps({"healthy": True, "ready": True, "trade_mode": "shadow"}) + "\n",
            encoding="utf-8",
        )
        (run_root / "systemd_post.txt").write_text(
            "ActiveState=active\nSubState=running\nNRestarts=0\n",
            encoding="utf-8",
        )

        def fake_run_logged_command(cmd):
            if cmd.label in {"telemetry_analyzer_recovery", "telemetry_analyzer_recovery_from_source_slice"}:
                (run_root / "live_segment_summary.json").write_text(
                    json.dumps(
                        {
                            "tick_count": 2,
                            "first_ts_ms": 1775000000000,
                            "last_ts_ms": 1775000100000,
                            "pnl_validity": {
                                "final_pnl_total": 0.0,
                                "final_pnl_realised": 0.0,
                                "final_pnl_unrealised": 0.0,
                                "final_q_global_tao": 0.0,
                            },
                        }
                    ) + "\n",
                    encoding="utf-8",
                )
                (run_root / "live_metrics.json").write_text(
                    json.dumps({"fills": {"total_count": 0, "total_base": 0.0}}) + "\n",
                    encoding="utf-8",
                )
                (run_root / "telemetry_report_live_segment.md").write_text("report\n", encoding="utf-8")
                return None
            raise AssertionError(f"unexpected command label {cmd.label}")

        with mock.patch.object(self.mod, "run_logged_command", side_effect=fake_run_logged_command):
            closeout_path = self.mod.recover_live_closeout("t_main_1", repo_root=self.repo_root, run_root=run_root, duration_sec=3600)

        closeout = json.loads(closeout_path.read_text(encoding="utf-8"))
        rebuilt = (run_root / "telemetry_bounded.jsonl").read_text(encoding="utf-8")
        self.assertIn('"tick":1', rebuilt)
        self.assertIn('"tick":2', rebuilt)
        self.assertTrue(closeout["source_slice_recovery_attempted"])
        self.assertTrue(closeout["source_slice_recovery_succeeded"])

    def test_recover_live_closeout_rebuilds_bounded_slice_from_guard_time_window(self):
        control_pack = _minimal_control_pack()
        telemetry_source = Path(control_pack["execution_defaults"]["telemetry_path"])
        telemetry_source.parent.mkdir(parents=True, exist_ok=True)
        start_utc = "2026-04-01T00:00:00Z"
        end_utc = "2026-04-01T00:00:05Z"
        start_ms = self.mod.utc_to_epoch_ms(start_utc)
        end_ms = self.mod.utc_to_epoch_ms(end_utc)
        telemetry_source.write_text(
            "\n".join(
                [
                    json.dumps({"execution_mode": "shadow", "kf_last_update_ms": start_ms - 1000, "t": 0}),
                    json.dumps({"execution_mode": "live", "kf_last_update_ms": start_ms + 1000, "t": 1}),
                    json.dumps({"execution_mode": "live", "kf_last_update_ms": start_ms + 2000, "t": 2}),
                    json.dumps({"execution_mode": "shadow", "kf_last_update_ms": end_ms + 1000, "t": 3}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        _write_yaml(self.repo_root / "phase5" / "control_pack.yaml", control_pack)

        run_root = self.repo_root / "promotion_runs" / "t_main_1_60m_qual_20260401T000000Z" / "live_canary"
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "guard.log").write_text(
            f"{start_utc} INFO guard_started duration_sec=300\n"
            f"{end_utc} CRITICAL triggered_intervention reason=service_restarts=1\n",
            encoding="utf-8",
        )

        def fake_run_logged_command(cmd):
            if cmd.label in {"telemetry_analyzer_recovery", "telemetry_analyzer_recovery_from_source_slice"}:
                (run_root / "live_segment_summary.json").write_text(
                    json.dumps(
                        {
                            "tick_count": 2,
                            "first_ts_ms": start_ms + 1000,
                            "last_ts_ms": start_ms + 2000,
                            "pnl_validity": {
                                "final_pnl_total": 0.25,
                                "final_pnl_realised": 0.1,
                                "final_pnl_unrealised": 0.15,
                                "final_q_global_tao": 0.0,
                            },
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                (run_root / "live_metrics.json").write_text(
                    json.dumps(
                        {
                            "fills": {"total_count": 1, "total_base": 0.01},
                            "execution_scorecard": {"hyperliquid": {"fills": 1}},
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                (run_root / "telemetry_report_live_segment.md").write_text("report\n", encoding="utf-8")
                return None
            raise AssertionError(f"unexpected command label {cmd.label}")

        with mock.patch.object(self.mod, "run_logged_command", side_effect=fake_run_logged_command), \
            mock.patch.object(
                self.mod,
                "curl_health",
                return_value=json.dumps(
                    {
                        "healthy": True,
                        "ready": True,
                        "trade_mode": "shadow",
                        "kill_events_present": False,
                        "reconcile_mismatch_count": 0,
                    }
                ),
            ), \
            mock.patch.object(
                self.mod,
                "systemd_show",
                return_value="ActiveState=active\nSubState=running\nNRestarts=0\n",
            ):
            closeout_path = self.mod.recover_live_closeout(
                "t_main_1",
                repo_root=self.repo_root,
                run_root=run_root,
                duration_sec=300,
            )

        telemetry_bounded = (run_root / "telemetry_bounded.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(telemetry_bounded), 2)
        self.assertIn('"t": 1', telemetry_bounded[0])
        self.assertIn('"t": 2', telemetry_bounded[1])
        closeout = json.loads(closeout_path.read_text(encoding="utf-8"))
        self.assertTrue(closeout["time_window_recovery_attempted"])
        self.assertTrue(closeout["time_window_recovery_succeeded"])
        self.assertTrue(closeout["analyzer_recovery_succeeded"])
        self.assertEqual(closeout["guard_exit_code"], 1)
        self.assertEqual(closeout["trade_mode_post"], "shadow")
        self.assertTrue((run_root / "health_post.json").exists())
        self.assertTrue((run_root / "systemd_post.txt").exists())
        self.assertTrue((run_root / "guard_result.json").exists())

    def test_recover_live_closeout_rebuilds_stderr_segment_from_source_offsets(self):
        control_pack = _minimal_control_pack()
        stderr_source = Path(control_pack["execution_defaults"]["stderr_path"])
        stderr_source.parent.mkdir(parents=True, exist_ok=True)
        stderr_source.write_text(
            "2026-04-01T00:00:00Z INFO preface\n"
            "2026-04-01T00:00:01Z WS_AUDIT venue=paradex component=ui_book_truth source=api status=ok token_usage=interactive\n"
            "2026-04-01T00:00:02Z WS_AUDIT venue=paradex component=ui_book_truth source=interactive status=ok token_usage=interactive\n"
            "2026-04-01T00:00:03Z WS_AUDIT venue=paradex component=ui_touch_reference action=applied source_kind=split orig_bid=1 adj_bid=2 orig_ask=3 adj_ask=2\n"
            "2026-04-01T00:10:00Z INFO epilogue\n",
            encoding="utf-8",
        )
        _write_yaml(self.repo_root / "phase5" / "control_pack.yaml", control_pack)

        run_root = self.repo_root / "promotion_runs" / "t_main_1_5m_canary_20260401T000000Z" / "live_canary"
        run_root.mkdir(parents=True, exist_ok=True)
        content = stderr_source.read_bytes()
        slice_start = content.index(
            b"2026-04-01T00:00:01Z WS_AUDIT venue=paradex component=ui_book_truth source=api status=ok token_usage=interactive\n"
        )
        (run_root / "err_offset_pre.txt").write_text(str(slice_start), encoding="utf-8")
        (run_root / "live_segment_summary.json").write_text(
            json.dumps(
                {
                    "tick_count": 3,
                    "first_ts_ms": 1775000001000,
                    "last_ts_ms": 1775000003000,
                    "pnl_validity": {"final_pnl_total": 0.0, "final_q_global_tao": 0.0},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (run_root / "live_metrics.json").write_text("{}\n", encoding="utf-8")
        (run_root / "telemetry_report_live_segment.md").write_text("report\n", encoding="utf-8")
        (run_root / "guard.log").write_text(
            "2026-04-01T00:05:00Z INFO guard_window_complete_restoring_shadow\n",
            encoding="utf-8",
        )
        (run_root / "health_post.json").write_text(
            json.dumps({"healthy": True, "ready": True, "trade_mode": "shadow"}) + "\n",
            encoding="utf-8",
        )
        (run_root / "systemd_post.txt").write_text(
            "ActiveState=active\nSubState=running\nNRestarts=0\n",
            encoding="utf-8",
        )

        closeout_path = self.mod.recover_live_closeout(
            "t_main_1", repo_root=self.repo_root, run_root=run_root, duration_sec=300
        )

        closeout = json.loads(closeout_path.read_text(encoding="utf-8"))
        self.assertTrue((run_root / "paraphina_live.err.segment").exists())
        self.assertTrue(closeout["stderr_source_slice_recovery_attempted"])
        self.assertTrue(closeout["stderr_source_slice_recovery_succeeded"])
        metrics = json.loads((run_root / "live_metrics.json").read_text(encoding="utf-8"))
        self.assertTrue(metrics["paradex_ui_book_truth_summary"]["observed"])
        self.assertTrue(metrics["paradex_ui_touch_reference_summary"]["observed"])

    def test_support_gate_satisfied_uses_dedicated_shadow_smoke_manifest(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["support_gate"] = "shadow_smoke_10m"
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)
        latest_manifest = {
            "surface_id": "wrong",
            "duration_sec": 1200,
            "run_root": "/tmp/live_run",
        }
        shadow_manifest = {
            "surface_id": "surface-ok",
            "duration_sec": 600,
            "shadow_smoke_status": "pass",
            "shadow_smoke_run_dir": "/tmp/shadow_smoke",
        }
        _write_yaml(self.repo_root / "phase5" / "runs" / "t_main_1" / "latest_run.yaml", latest_manifest)
        _write_yaml(
            self.repo_root / "phase5" / "runs" / "t_main_1" / "shadow_smoke_10m.yaml",
            shadow_manifest,
        )

        ok, reason = self.mod.support_gate_satisfied(
            "t_main_1",
            queue["serialized_mainline"][0],
            self.repo_root,
            "surface-ok",
        )

        self.assertTrue(ok)
        self.assertEqual(reason, "shadow smoke passed")

    def test_support_gate_satisfied_accepts_shadow_smoke_30m(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["support_gate"] = "shadow_smoke_30m"
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)
        _write_yaml(
            self.repo_root / "phase5" / "runs" / "t_main_1" / "shadow_smoke_30m.yaml",
            {
                "surface_id": "surface-ok",
                "duration_sec": 1800,
                "shadow_smoke_status": "pass",
                "shadow_smoke_run_dir": "/tmp/shadow_smoke_30m",
            },
        )

        ok, reason = self.mod.support_gate_satisfied(
            "t_main_1",
            queue["serialized_mainline"][0],
            self.repo_root,
            "surface-ok",
        )

        self.assertTrue(ok)
        self.assertEqual(reason, "shadow smoke passed")

    def test_support_gate_satisfied_rejects_mechanism_not_exercised_manifest(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["support_gate"] = "shadow_smoke_10m"
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)
        _write_yaml(
            self.repo_root / "phase5" / "runs" / "t_main_1" / "shadow_smoke_10m.yaml",
            {
                "surface_id": "surface-ok",
                "duration_sec": 600,
                "shadow_smoke_status": "fail",
                "support_gate_evaluation": {
                    "gate_passed": False,
                    "failure_reason": "mechanism_not_exercised",
                },
            },
        )

        ok, reason = self.mod.support_gate_satisfied(
            "t_main_1",
            queue["serialized_mainline"][0],
            self.repo_root,
            "surface-ok",
        )

        self.assertFalse(ok)
        self.assertEqual(reason, "mechanism_not_exercised")

    def test_support_gate_satisfied_accepts_health_only_shadow_manifest(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["support_gate"] = "shadow_smoke_10m"
        queue["serialized_mainline"][0]["support_gate_require_mechanism"] = False
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)
        _write_yaml(
            self.repo_root / "phase5" / "runs" / "t_main_1" / "shadow_smoke_10m.yaml",
            {
                "surface_id": "surface-ok",
                "duration_sec": 600,
                "shadow_smoke_status": "pass",
                "mechanism_pass": False,
                "mechanism_required": False,
                "support_gate_evaluation": {
                    "gate_passed": True,
                    "health_pass": True,
                    "mechanism_pass": False,
                    "mechanism_required": False,
                    "failure_reason": None,
                },
            },
        )

        ok, reason = self.mod.support_gate_satisfied(
            "t_main_1",
            queue["serialized_mainline"][0],
            self.repo_root,
            "surface-ok",
        )

        self.assertTrue(ok)
        self.assertEqual(reason, "shadow smoke passed")

    def test_evaluate_shadow_mechanism_evidence_detects_missing_required_patterns(self):
        run_root = self.repo_root / "promotion_runs" / "shadow"
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "run.log").write_text("no relevant audit lines\n", encoding="utf-8")

        result = self.mod.evaluate_shadow_mechanism_evidence(
            {
                "mechanism_evidence": {
                    "required_log_patterns_all": [
                        {"pattern": "PARADEX_CANCEL_BATCH", "min_occurrences": 1},
                    ],
                }
            },
            run_root,
        )

        self.assertFalse(result["mechanism_pass"])
        self.assertEqual(
            result["failure_reason"],
            "missing_required_log_pattern:PARADEX_CANCEL_BATCH",
        )

    def test_run_shadow_smoke_fails_when_mechanism_not_exercised(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["candidate"] = {
            "runtime_binary": str(self.repo_root / "runtime_candidate")
        }
        queue["serialized_mainline"][0]["mechanism_evidence"] = {
            "required_log_patterns_all": [
                {"pattern": "PARADEX_CANCEL_BATCH", "min_occurrences": 1},
            ]
        }
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        control_pack = _minimal_control_pack()
        control_pack["execution_defaults"]["stage_overlay_source"] = str(self.repo_root / "stage_overlay.env")
        control_pack["execution_defaults"]["promotion_runs_root"] = str(self.repo_root / "promotion_runs")
        _write_yaml(self.repo_root / "phase5" / "control_pack.yaml", control_pack)

        (self.repo_root / "runtime_candidate").write_text("bin\n", encoding="utf-8")
        (self.repo_root / "stage_overlay.env").write_text("PARAPHINA_TRADE_MODE=shadow\n", encoding="utf-8")
        shadow_supervisor = self.repo_root / "tools" / "paraphina_shadow_supervisor.sh"
        shadow_supervisor.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        run_root = self.repo_root / "promotion_runs" / "shadow_run"
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "run.log").write_text("no batch audit lines\n", encoding="utf-8")

        with mock.patch.object(self.mod, "ensure_shadow_health", return_value={"healthy": True, "ready": True, "trade_mode": "shadow"}), \
             mock.patch.object(self.mod, "build_shadow_smoke_run_root", return_value=run_root), \
             mock.patch.object(
                 self.mod.subprocess,
                 "run",
                 return_value=subprocess.CompletedProcess(args=["timeout"], returncode=124, stdout="", stderr=""),
             ):
            with self.assertRaises(subprocess.CalledProcessError):
                self.mod.run_shadow_smoke("t_main_1", 600, repo_root=self.repo_root)

        manifest = yaml.safe_load(
            (self.repo_root / "phase5" / "runs" / "t_main_1" / "shadow_smoke_10m.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["shadow_smoke_status"], "fail")
        self.assertEqual(manifest["failure_reason"], "missing_required_log_pattern:PARADEX_CANCEL_BATCH")

    def test_run_shadow_smoke_passes_when_mechanism_is_not_required(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["candidate"] = {
            "runtime_binary": str(self.repo_root / "runtime_candidate")
        }
        queue["serialized_mainline"][0]["mechanism_evidence"] = {
            "required_log_patterns_all": [
                {"pattern": "PARADEX_CANCEL_BATCH", "min_occurrences": 1},
            ]
        }
        queue["serialized_mainline"][0]["support_gate_require_mechanism"] = False
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        control_pack = _minimal_control_pack()
        control_pack["execution_defaults"]["stage_overlay_source"] = str(self.repo_root / "stage_overlay.env")
        control_pack["execution_defaults"]["promotion_runs_root"] = str(self.repo_root / "promotion_runs")
        _write_yaml(self.repo_root / "phase5" / "control_pack.yaml", control_pack)

        (self.repo_root / "runtime_candidate").write_text("bin\n", encoding="utf-8")
        (self.repo_root / "stage_overlay.env").write_text("PARAPHINA_TRADE_MODE=shadow\n", encoding="utf-8")
        shadow_supervisor = self.repo_root / "tools" / "paraphina_shadow_supervisor.sh"
        shadow_supervisor.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        run_root = self.repo_root / "promotion_runs" / "shadow_run"
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "run.log").write_text("no batch audit lines\n", encoding="utf-8")

        with mock.patch.object(
            self.mod,
            "ensure_shadow_health",
            return_value={"healthy": True, "ready": True, "trade_mode": "shadow"},
        ), mock.patch.object(self.mod, "build_shadow_smoke_run_root", return_value=run_root), mock.patch.object(
            self.mod.subprocess,
            "run",
            return_value=subprocess.CompletedProcess(args=["timeout"], returncode=124, stdout="", stderr=""),
        ):
            returned_run_root = self.mod.run_shadow_smoke("t_main_1", 600, repo_root=self.repo_root)

        self.assertEqual(returned_run_root, run_root)
        manifest = yaml.safe_load(
            (self.repo_root / "phase5" / "runs" / "t_main_1" / "shadow_smoke_10m.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["shadow_smoke_status"], "pass")
        self.assertTrue(manifest["health_pass"])
        self.assertFalse(manifest["mechanism_pass"])
        self.assertFalse(manifest["mechanism_required"])
        self.assertTrue(manifest["support_gate_evaluation"]["gate_passed"])

    def test_run_shadow_smoke_fails_when_connector_unavailable_is_logged(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["candidate"] = {
            "runtime_binary": str(self.repo_root / "runtime_candidate")
        }
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        control_pack = _minimal_control_pack()
        control_pack["execution_defaults"]["stage_overlay_source"] = str(self.repo_root / "stage_overlay.env")
        control_pack["execution_defaults"]["promotion_runs_root"] = str(self.repo_root / "promotion_runs")
        _write_yaml(self.repo_root / "phase5" / "control_pack.yaml", control_pack)

        (self.repo_root / "runtime_candidate").write_text("bin\n", encoding="utf-8")
        (self.repo_root / "stage_overlay.env").write_text("PARAPHINA_TRADE_MODE=shadow\n", encoding="utf-8")
        shadow_supervisor = self.repo_root / "tools" / "paraphina_shadow_supervisor.sh"
        shadow_supervisor.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        run_root = self.repo_root / "promotion_runs" / "shadow_run"
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "run.log").write_text(
            "paraphina_live | error=connector_unavailable connector=hyperliquid\n",
            encoding="utf-8",
        )

        with mock.patch.object(self.mod, "ensure_shadow_health", return_value={"healthy": True, "ready": True, "trade_mode": "shadow"}), \
             mock.patch.object(self.mod, "build_shadow_smoke_run_root", return_value=run_root), \
             mock.patch.object(
                 self.mod.subprocess,
                 "run",
                 return_value=subprocess.CompletedProcess(args=["timeout"], returncode=124, stdout="", stderr=""),
             ):
            with self.assertRaises(subprocess.CalledProcessError):
                self.mod.run_shadow_smoke("t_main_1", 600, repo_root=self.repo_root)

        manifest = yaml.safe_load(
            (self.repo_root / "phase5" / "runs" / "t_main_1" / "shadow_smoke_10m.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["shadow_smoke_status"], "fail")
        self.assertEqual(manifest["failure_reason"], "connector_unavailable:hyperliquid")
        self.assertEqual(
            manifest["support_gate_evaluation"]["connector_availability"]["connector_unavailable"],
            ["hyperliquid"],
        )

    def test_run_shadow_smoke_merges_env_file_before_stage_overlay(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["candidate"] = {
            "runtime_binary": str(self.repo_root / "runtime_candidate")
        }
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        control_pack = _minimal_control_pack()
        control_pack["execution_defaults"]["env_file"] = str(self.repo_root / "current.env")
        control_pack["execution_defaults"]["stage_overlay_source"] = str(self.repo_root / "stage_overlay.env")
        control_pack["execution_defaults"]["promotion_runs_root"] = str(self.repo_root / "promotion_runs")
        _write_yaml(self.repo_root / "phase5" / "control_pack.yaml", control_pack)

        (self.repo_root / "runtime_candidate").write_text("bin\n", encoding="utf-8")
        (self.repo_root / "current.env").write_text(
            'PARADEX_JWT_CMD="/tmp/jwt.py --flag"\nSHARED=from_base\nBASE_ONLY=1\n',
            encoding="utf-8",
        )
        (self.repo_root / "stage_overlay.env").write_text(
            "PARAPHINA_PARADEX_PUBLIC_FEED=bbo\nSHARED=from_overlay\nOVERLAY_ONLY=1\n",
            encoding="utf-8",
        )
        shadow_supervisor = self.repo_root / "tools" / "paraphina_shadow_supervisor.sh"
        shadow_supervisor.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        run_root = self.repo_root / "promotion_runs" / "shadow_run"
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "run.log").write_text("PARADEX_OK\n", encoding="utf-8")

        seen_env = {}

        def fake_subprocess_run(*args, **kwargs):
            seen_env.update(kwargs["env"])
            return subprocess.CompletedProcess(args=["timeout"], returncode=124, stdout="", stderr="")

        with mock.patch.object(
            self.mod,
            "ensure_shadow_health",
            return_value={"healthy": True, "ready": True, "trade_mode": "shadow"},
        ), mock.patch.object(self.mod, "build_shadow_smoke_run_root", return_value=run_root), mock.patch.object(
            self.mod.subprocess,
            "run",
            side_effect=fake_subprocess_run,
        ):
            self.mod.run_shadow_smoke("t_main_1", 600, repo_root=self.repo_root)

        self.assertEqual(seen_env["PARADEX_JWT_CMD"], "/tmp/jwt.py --flag")
        self.assertEqual(seen_env["BASE_ONLY"], "1")
        self.assertEqual(seen_env["PARAPHINA_PARADEX_PUBLIC_FEED"], "bbo")
        self.assertEqual(seen_env["OVERLAY_ONLY"], "1")
        self.assertEqual(seen_env["SHARED"], "from_overlay")

    def test_run_live_guarded_recovers_closeout_after_guard_failure(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["candidate"] = {"runtime_binary": str(self.repo_root / "runtime_candidate")}
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)
        control_pack = _minimal_control_pack()
        control_pack["execution_defaults"].update(
            {
                "runtime_binary": str(self.repo_root / "runtime_live"),
                "stage_overlay_target": str(self.repo_root / "stage_overlay_target.env"),
                "stage_overlay_source": str(self.repo_root / "stage_overlay_source.env"),
                "live_exec_dropin_target": str(self.repo_root / "live_exec_flag.conf"),
                "live_exec_dropin_source": str(self.repo_root / "live_exec_flag.source.conf"),
                "guard_script": str(self.repo_root / "guard.py"),
                "telemetry_path": str(self.repo_root / "telemetry.jsonl"),
                "stderr_path": str(self.repo_root / "paraphina_live.err"),
                "promotion_runs_root": str(self.repo_root / "promotion_runs"),
            }
        )
        _write_yaml(self.repo_root / "phase5" / "control_pack.yaml", control_pack)
        for rel_path, content in (
            ("runtime_candidate", "candidate"),
            ("runtime_live", "live"),
            ("stage_overlay_target.env", "TARGET=1\n"),
            ("stage_overlay_source.env", "SOURCE=1\n"),
            ("live_exec_flag.source.conf", "[Service]\nEnvironment=LIVE=1\n"),
            ("guard.py", "#!/usr/bin/env python3\n"),
            ("telemetry.jsonl", '{"execution_mode":"live","tick":1}\n'),
            ("paraphina_live.err", "warn\n"),
        ):
            path = self.repo_root / rel_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

        def fake_run_logged_command(cmd):
            if cmd.label == "live_guard":
                if cmd.stdout_path:
                    cmd.stdout_path.write_text(
                        "2026-04-01T00:20:00Z INFO guard_window_complete_restoring_shadow\n",
                        encoding="utf-8",
                    )
                return subprocess.CompletedProcess(cmd.argv, 1, "", "")
            raise AssertionError(f"unexpected command label {cmd.label}")

        recovered: list[Path] = []

        def fake_recover(tranche_id, repo_root, run_root, duration_sec):
            recovered.append(run_root)
            (run_root / "live_closeout_bundle.json").write_text(json.dumps({"ok": True}) + "\n", encoding="utf-8")
            return run_root / "live_closeout_bundle.json"

        def fake_balance_snapshot(*, label, defaults, run_root, repo_root, pre_snapshot=None, check=True):
            path = run_root / f"balance_{label}_snapshot.json"
            path.write_text(
                json.dumps({"schema_version": 1, "label": label, "total_balance_usd": "1.0"}) + "\n",
                encoding="utf-8",
            )
            if label == "post":
                (run_root / "balance_snapshot_comparison.json").write_text(
                    json.dumps({"schema_version": 1, "total": {"delta_usd": "0.0"}}) + "\n",
                    encoding="utf-8",
                )
            return path

        def fake_pre_live_audit(*, defaults, run_root):
            path = run_root / "direct_venue_audit_pre.json"
            path.write_text(json.dumps({"ok": True, "results": []}) + "\n", encoding="utf-8")
            return path

        with mock.patch.object(self.mod, "admission_check", return_value={"surface_id": "surface-ok"}), mock.patch.object(
            self.mod, "runtime_install_required", return_value=False
        ), mock.patch.object(self.mod, "run_shell"), mock.patch.object(
            self.mod, "wait_for_live_health", return_value={"healthy": True, "ready": True, "trade_mode": "live"}
        ), mock.patch.object(
            self.mod, "wait_for_shadow_health", return_value={"healthy": True, "ready": True, "trade_mode": "shadow"}
        ), mock.patch.object(self.mod, "curl_health", return_value='{"healthy": true, "ready": true, "trade_mode": "live"}'), mock.patch.object(
            self.mod, "systemd_show", return_value="ActiveState=active\nSubState=running\nNRestarts=1\n"
        ), mock.patch.object(self.mod, "run_logged_command", side_effect=fake_run_logged_command), mock.patch.object(
            self.mod, "recover_live_closeout", side_effect=fake_recover
        ), mock.patch.object(
            self.mod, "run_balance_snapshot", side_effect=fake_balance_snapshot
        ), mock.patch.object(
            self.mod, "run_pre_live_direct_venue_audit", side_effect=fake_pre_live_audit
        ):
            run_root = self.mod.run_live_guarded("t_main_1", 1200, repo_root=self.repo_root)

        self.assertEqual(recovered, [run_root])
        guard_result = json.loads((run_root / "guard_result.json").read_text(encoding="utf-8"))
        self.assertEqual(guard_result["exit_code"], 1)
        self.assertTrue((run_root / "telemetry_bounded.jsonl").exists())
        manifest = yaml.safe_load((self.repo_root / "phase5" / "runs" / "t_main_1" / "latest_run.yaml").read_text(encoding="utf-8"))
        self.assertEqual(manifest["run_state"], "window_complete")
        self.assertEqual(manifest["guard_exit_code"], 1)

    def test_ensure_disk_headroom_rejects_when_free_space_too_low(self):
        with mock.patch.object(self.mod.shutil, "disk_usage", return_value=shutil._ntuple_diskusage(total=10, used=9, free=1)):
            with self.assertRaises(RuntimeError):
                self.mod.ensure_disk_headroom(self.repo_root, 3600)

    def test_admission_check_rejects_when_telemetry_headroom_is_low(self):
        queue = _minimal_queue()
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)
        control_pack = _minimal_control_pack()
        _write_yaml(self.repo_root / "phase5" / "control_pack.yaml", control_pack)
        Path(control_pack["execution_defaults"]["telemetry_path"]).write_text("", encoding="utf-8")

        healthy = shutil._ntuple_diskusage(total=20 * 1024 * 1024 * 1024, used=1, free=20 * 1024 * 1024 * 1024)
        starved = shutil._ntuple_diskusage(total=20 * 1024 * 1024 * 1024, used=(20 * 1024 * 1024 * 1024) - 1, free=1)

        def fake_disk_usage(path):
            if str(path) == "/tmp/telemetry.jsonl":
                return starved
            return healthy

        with mock.patch.object(self.mod.shutil, "disk_usage", side_effect=fake_disk_usage):
            with self.assertRaises(RuntimeError) as ctx:
                self.mod.admission_check("t_main_1", 1200, self.repo_root)

        self.assertIn("telemetry headroom below automation default", str(ctx.exception))

    def test_admission_check_rejects_missing_cleanup_binary(self):
        queue = _minimal_queue()
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)
        control_pack = _minimal_control_pack()
        runtime_binary = self.repo_root / "runtime_candidate"
        runtime_binary.write_text("bin", encoding="utf-8")
        control_pack["execution_defaults"]["runtime_binary"] = str(runtime_binary)
        control_pack["execution_defaults"]["cleanup_binary"] = str(self.repo_root / "missing_live_cleanup")
        _write_yaml(self.repo_root / "phase5" / "control_pack.yaml", control_pack)

        with mock.patch.object(
            self.mod,
            "curl_health",
            return_value='{"healthy": true, "ready": true, "trade_mode": "shadow"}\n',
        ), mock.patch.object(
            self.mod.shutil,
            "disk_usage",
            return_value=shutil._ntuple_diskusage(
                total=20 * 1024 * 1024 * 1024,
                used=1,
                free=20 * 1024 * 1024 * 1024,
            ),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                self.mod.admission_check("t_main_1", 1200, self.repo_root)

        self.assertIn("cleanup binary missing", str(ctx.exception))

    def test_hyperliquid_rate_limit_summary_blocks_over_cap(self):
        summary = self.mod.hyperliquid_rate_limit_summary(
            {
                "cumVlm": "57835.07",
                "nRequestsUsed": 71983,
                "nRequestsCap": 67835,
                "nRequestsSurplus": 0,
            }
        )

        self.assertEqual(summary["status"], "fail")
        self.assertTrue(summary["blocked"])
        self.assertEqual(summary["usage_pct"], 106.1148)
        self.assertEqual(summary["available_request_weight"], 0)
        self.assertEqual(summary["request_weight_to_clear"], 4149)
        self.assertEqual(summary["reserve_cost_to_clear_usdc"], 2.0745)

    def test_hyperliquid_quota_runway_thresholds_are_duration_scoped(self):
        self.assertEqual(self.mod.hyperliquid_quota_runway_threshold(None), 0)
        self.assertEqual(self.mod.hyperliquid_quota_runway_threshold(0), 0)
        self.assertEqual(self.mod.hyperliquid_quota_runway_threshold(300), 2000)
        self.assertEqual(self.mod.hyperliquid_quota_runway_threshold(1200), 5000)
        self.assertEqual(self.mod.hyperliquid_quota_runway_threshold(7200), 10000)

    def test_hyperliquid_user_rate_limit_preflight_rejects_over_cap(self):
        queue = _minimal_queue()
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)
        control_pack = _minimal_control_pack()
        overlay = self.repo_root / "stage_overlay.env"
        overlay.write_text(
            "\n".join(
                [
                    "PARAPHINA_LIVE_CONNECTORS=hyperliquid,lighter",
                    "HL_VAULT_ADDRESS=0x0000000000000000000000000000000000000001",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        control_pack["execution_defaults"]["stage_overlay_source"] = str(overlay)
        _write_yaml(self.repo_root / "phase5" / "control_pack.yaml", control_pack)

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b'{"nRequestsUsed":71983,"nRequestsCap":67835,"nRequestsSurplus":0,"cumVlm":"57835.07"}'

        with mock.patch.object(self.mod.urllib.request, "urlopen", return_value=FakeResponse()):
            with self.assertRaises(RuntimeError) as ctx:
                self.mod.hyperliquid_user_rate_limit_preflight(queue["serialized_mainline"][0], control_pack)

        self.assertIn("Hyperliquid action quota blocks live admission", str(ctx.exception))
        self.assertIn("request_weight_to_clear=4149", str(ctx.exception))

    def test_hyperliquid_user_rate_limit_preflight_rejects_insufficient_runway(self):
        queue = _minimal_queue()
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)
        control_pack = _minimal_control_pack()
        overlay = self.repo_root / "stage_overlay.env"
        overlay.write_text(
            "\n".join(
                [
                    "PARAPHINA_LIVE_CONNECTORS=hyperliquid,lighter",
                    "HL_VAULT_ADDRESS=0x0000000000000000000000000000000000000001",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        control_pack["execution_defaults"]["stage_overlay_source"] = str(overlay)
        _write_yaml(self.repo_root / "phase5" / "control_pack.yaml", control_pack)

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b'{"nRequestsUsed":60000,"nRequestsCap":67835,"nRequestsSurplus":0,"cumVlm":"57835.07"}'

        with mock.patch.object(self.mod.urllib.request, "urlopen", return_value=FakeResponse()):
            with self.assertRaises(RuntimeError) as ctx:
                self.mod.hyperliquid_user_rate_limit_preflight(
                    queue["serialized_mainline"][0],
                    control_pack,
                    7200,
                )

        self.assertIn("Hyperliquid quota runway insufficient for live admission", str(ctx.exception))
        self.assertIn("available_request_weight=7835", str(ctx.exception))
        self.assertIn("required_runway_request_weight=10000", str(ctx.exception))
        self.assertIn("runway_shortfall_request_weight=2165", str(ctx.exception))

    def test_hyperliquid_user_rate_limit_preflight_passes_with_sufficient_runway(self):
        queue = _minimal_queue()
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)
        control_pack = _minimal_control_pack()
        overlay = self.repo_root / "stage_overlay.env"
        overlay.write_text(
            "\n".join(
                [
                    "PARAPHINA_LIVE_CONNECTORS=hyperliquid,lighter",
                    "HL_VAULT_ADDRESS=0x0000000000000000000000000000000000000001",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        control_pack["execution_defaults"]["stage_overlay_source"] = str(overlay)
        _write_yaml(self.repo_root / "phase5" / "control_pack.yaml", control_pack)

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b'{"nRequestsUsed":50000,"nRequestsCap":67835,"nRequestsSurplus":0,"cumVlm":"57835.07"}'

        with mock.patch.object(self.mod.urllib.request, "urlopen", return_value=FakeResponse()):
            summary = self.mod.hyperliquid_user_rate_limit_preflight(
                queue["serialized_mainline"][0],
                control_pack,
                7200,
            )

        self.assertEqual(summary["status"], "pass")
        self.assertFalse(summary["blocked"])
        self.assertFalse(summary["runway_blocked"])
        self.assertEqual(summary["available_request_weight"], 17835)
        self.assertEqual(summary["required_runway_request_weight"], 10000)
        self.assertEqual(summary["runway_shortfall_request_weight"], 0)

    def test_run_shadow_smoke_rejects_when_tempdir_headroom_is_low(self):
        queue = _minimal_queue()
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)
        control_pack = _minimal_control_pack()
        control_pack["execution_defaults"]["telemetry_path"] = str(self.repo_root / "telemetry.jsonl")
        control_pack["execution_defaults"]["promotion_runs_root"] = str(self.repo_root / "promotion_runs")
        _write_yaml(self.repo_root / "phase5" / "control_pack.yaml", control_pack)

        healthy = shutil._ntuple_diskusage(total=20 * 1024 * 1024 * 1024, used=1, free=20 * 1024 * 1024 * 1024)
        starved = shutil._ntuple_diskusage(total=20 * 1024 * 1024 * 1024, used=(20 * 1024 * 1024 * 1024) - 1, free=1)

        def fake_disk_usage(path):
            if str(path) in {"/tmp", "/tmp/paraphina_current_runs"}:
                return starved
            return healthy

        with mock.patch.object(self.mod.shutil, "disk_usage", side_effect=fake_disk_usage), mock.patch.object(
            self.mod, "ensure_shadow_health"
        ) as health_mock:
            with self.assertRaises(RuntimeError) as ctx:
                self.mod.run_shadow_smoke("t_main_1", 600, repo_root=self.repo_root)

        self.assertIn("tempdir headroom below automation default", str(ctx.exception))
        health_mock.assert_not_called()

    def test_autoscore_final_rung_holds_without_explicit_promotion_rules(self):
        run_root = self.repo_root / "promotion_runs" / "t_main_1_60m_qual_20260401T000000Z" / "live_canary"
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "live_closeout_bundle.json").write_text(
            json.dumps(
                {
                    "summary_exists": True,
                    "report_exists": True,
                    "guard_intervened": False,
                    "guard_window_completed": True,
                    "pre_restore_venue_audit_clean": True,
                    "post_rollback_venue_audit_clean": True,
                    "healthy_post": True,
                    "ready_post": True,
                    "kill_events_present_post": False,
                    "trade_mode_post": "shadow",
                    "systemd_active_state_post": "active",
                    "systemd_sub_state_post": "running",
                    "systemd_nrestarts_post": "0",
                    "guard_exit_code": 0,
                }
            ) + "\n",
            encoding="utf-8",
        )
        (run_root / "live_segment_summary.json").write_text(
            json.dumps({"pnl_validity": {"final_pnl_total": 0.0}}) + "\n",
            encoding="utf-8",
        )
        (run_root / "live_metrics.json").write_text(
            json.dumps({"fills": {"total_count": 0, "total_base": 0.0}}) + "\n",
            encoding="utf-8",
        )
        queue, control_pack = self.mod.load_state(self.repo_root)
        tranche = queue["serialized_mainline"][0]

        result = self.mod.autoscore_run(tranche, control_pack, run_root, 3600, self.repo_root)

        self.assertTrue(result["clean"]["passed"])
        self.assertFalse(result["has_promotion_rules"])
        self.assertEqual(result["suggested_action"], "hold")

    def test_spawn_lanes_creates_child_and_support_worktrees(self):
        session = self._spawn_lanes_with_preflight("t_main_1")

        self.assertEqual(session["state"], "spawned")
        self.assertEqual(session["preflight"]["status"], "pass")
        self.assertEqual(session["preflight"]["state_sync"]["status"], "pass")
        self.assertTrue(Path(session["preflight_summary_path"]).exists())
        self.assertTrue(Path(session["state_sync_report_path"]).exists())
        lane_ids = {lane["lane_id"] for lane in session["lanes"]}
        self.assertIn(self.mod.LANE_ROLE_LIVE, lane_ids)
        self.assertIn(self.mod.LANE_ROLE_FORENSICS, lane_ids)
        self.assertIn(self.mod.LANE_ROLE_PASS_PREP, lane_ids)
        self.assertTrue(
            any(lane_id.startswith(self.mod.LANE_ROLE_FAIL_PREP) for lane_id in lane_ids)
        )
        self.assertIn(f"{self.mod.LANE_ROLE_SUPPORT_PREFIX}t_support_1", lane_ids)
        for lane in session["lanes"]:
            bundle_dir = self.repo_root / "phase5" / "runs" / "t_main_1" / "lanes" / lane["lane_id"]
            self.assertTrue((bundle_dir / "lane_manifest.yaml").exists())
            if lane["kind"] in {"child_prep", "support_track"}:
                self.assertTrue(Path(lane["worktree_path"]).exists())

    def test_run_support_lane_validate_only_reconstructs_bundle_snapshot_when_worktree_removed(self):
        session = self._spawn_lanes_with_preflight("t_main_1")
        lane_id = f"{self.mod.LANE_ROLE_SUPPORT_PREFIX}t_support_1"
        lane = next(lane for lane in session["lanes"] if lane["lane_id"] == lane_id)
        self.mod.remove_worktree(self.repo_root, Path(lane["worktree_path"]))

        result = self.mod.run_support_lane("t_main_1", lane, self.repo_root, stage_context="post_live")

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["workspace_source"], "bundle_snapshot")

    def test_spawn_lanes_includes_support_family_lane(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["automation"]["support_families"] = [
            {
                "id": "tooling",
                "support_track_id": "t_support_1",
                "trigger_mode": "always",
                "autorun_policy": "validate_only",
                "max_parallel_runs": 2,
                "stop_on_mainline_promote": True,
            }
        ]
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        session = self._spawn_lanes_with_preflight("t_main_1")

        lane_ids = {lane["lane_id"] for lane in session["lanes"]}
        family_lane_id = f"{self.mod.LANE_ROLE_SUPPORT_PREFIX}tooling__t_support_1"
        self.assertIn(family_lane_id, lane_ids)
        lane = next(lane for lane in session["lanes"] if lane["lane_id"] == family_lane_id)
        self.assertEqual(lane["family_id"], "tooling")
        self.assertEqual(lane["autorun_policy"], "validate_only")
        self.assertEqual(lane["max_parallel_runs"], 2)
        self.assertEqual(session["preflight"]["state_sync"]["status"], "pass")

    def test_spawn_lanes_blocks_when_state_sync_is_drifted(self):
        self._write_coherent_state_sync_layout()
        (self.repo_root / "ROADMAP.md").write_text(
            "\n".join(
                [
                    "# Roadmap",
                    "",
                    "- `t_main_1` surface `surface-1`",
                    "- `t_main_2` registered for later",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        with mock.patch.object(
            self.mod,
            "ensure_shadow_health",
            return_value={"healthy": True, "ready": True, "trade_mode": "shadow"},
        ), mock.patch.object(
            self.mod,
            "ensure_runtime_storage_headroom",
            return_value={
                "repo": {"free_bytes": 10},
                "promotion_runs": {"free_bytes": 10},
                "telemetry": {"free_bytes": 10},
                "tempdir": {"free_bytes": 10},
                "current_runs": {"free_bytes": 10},
            },
        ):
            with self.assertRaises(RuntimeError) as ctx:
                self.mod.spawn_lanes("t_main_1", repo_root=self.repo_root)

        self.assertIn("orchestration preflight blocked by state-sync", str(ctx.exception))
        orchestration = self.mod.load_orchestration(self.repo_root)
        self.assertEqual(orchestration["sessions"], [])
        lane_root = self.repo_root / "phase5" / "runs" / "t_main_1" / "lanes"
        self.assertFalse(lane_root.exists())
        preflight_summary = self.repo_root / "phase5" / "runs" / "t_main_1" / "preflight_summary.json"
        self.assertTrue(preflight_summary.exists())

    def test_build_venue_capability_matrix_prefers_stage_overlay(self):
        overlay = self.repo_root / "stage_overlay_live.env"
        overlay.write_text(
            "\n".join(
                [
                    "PARAPHINA_LIVE_CONNECTORS=hyperliquid,lighter,extended,aster,paradex",
                    "PARAPHINA_FV_DISABLED_VENUES=lighter",
                    "PARAPHINA_MM_VENUE_ROLE_HYPERLIQUID=probationary",
                    "PARAPHINA_MM_VENUE_ROLE_LIGHTER=probationary",
                    "PARAPHINA_MM_VENUE_ROLE_EXTENDED=fill",
                    "PARAPHINA_MM_VENUE_ROLE_ASTER=fill",
                    "PARAPHINA_MM_VENUE_ROLE_PARADEX=fill",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["env_diff"] = {"stage_overlay_source": str(overlay)}

        matrix = self.mod.build_venue_capability_matrix(
            queue["serialized_mainline"][0],
            _minimal_control_pack(),
            repo_root=self.repo_root,
        )

        self.assertTrue(matrix["venues"]["extended"]["connected"])
        self.assertTrue(matrix["venues"]["aster"]["fill_role_enabled"])
        self.assertFalse(matrix["venues"]["aster"]["fill_participation_observed"])
        self.assertFalse(matrix["venues"]["aster"]["primary_fill_candidate"])
        self.assertFalse(matrix["venues"]["lighter"]["fv_eligible"])

    def test_build_venue_capability_matrix_requires_fill_participation_evidence(self):
        overlay = self.repo_root / "stage_overlay_live.env"
        overlay.write_text(
            "\n".join(
                [
                    "PARAPHINA_LIVE_CONNECTORS=hyperliquid,lighter,extended,aster,paradex",
                    "PARAPHINA_FV_DISABLED_VENUES=",
                    "PARAPHINA_MM_VENUE_ROLE_HYPERLIQUID=fill",
                    "PARAPHINA_MM_VENUE_ROLE_LIGHTER=fill",
                    "PARAPHINA_MM_VENUE_ROLE_EXTENDED=fill",
                    "PARAPHINA_MM_VENUE_ROLE_ASTER=fill",
                    "PARAPHINA_MM_VENUE_ROLE_PARADEX=fill",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["env_diff"] = {"stage_overlay_source": str(overlay)}
        run_root = self.repo_root / "promotion_runs" / "t_main_1_60m_20260408T000000Z" / "live_canary"
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "live_closeout_bundle.json").write_text(
            json.dumps(
                {
                    "pre_restore_venue_audit_clean": True,
                    "post_rollback_venue_audit_clean": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (run_root / "live_metrics.json").write_text(
            json.dumps(
                {
                    "fills": {
                        "by_venue": {
                            "hyperliquid": {"fill_count": 0, "fill_base": 0.0},
                            "lighter": {"fill_count": 0, "fill_base": 0.0},
                            "extended": {"fill_count": 0, "fill_base": 0.0},
                            "aster": {"fill_count": 2, "fill_base": 0.02},
                            "paradex": {"fill_count": 0, "fill_base": 0.0},
                        }
                    },
                    "execution_scorecard": {
                        "hyperliquid": {"place_i": 0, "replace_i": 0, "cancel_i": 0, "place_ack": 0, "cancel_ack": 0, "fills": 0},
                        "lighter": {"place_i": 0, "replace_i": 0, "cancel_i": 0, "place_ack": 0, "cancel_ack": 0, "fills": 0},
                        "extended": {"place_i": 1, "replace_i": 0, "cancel_i": 1, "place_ack": 1, "cancel_ack": 1, "fills": 0},
                        "aster": {"place_i": 3, "replace_i": 0, "cancel_i": 1, "place_ack": 3, "cancel_ack": 1, "fills": 2},
                        "paradex": {"place_i": 5, "replace_i": 1, "cancel_i": 7, "place_ack": 5, "cancel_ack": 4, "fills": 0},
                    },
                    "orders_per_venue": {
                        "hyperliquid": 0,
                        "lighter": 0,
                        "extended": 10,
                        "aster": 25,
                        "paradex": 30,
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )

        matrix = self.mod.build_venue_capability_matrix(
            queue["serialized_mainline"][0],
            _minimal_control_pack(),
            repo_root=self.repo_root,
            run_root=run_root,
            autoscore={"clean": {"passed": True}, "mechanism": {"passed": True}},
            final_decision="hold",
        )

        self.assertTrue(matrix["venues"]["aster"]["fill_role_enabled"])
        self.assertTrue(matrix["venues"]["aster"]["fill_participation_observed"])
        self.assertTrue(matrix["venues"]["aster"]["primary_fill_candidate"])
        self.assertTrue(matrix["venues"]["extended"]["fill_role_enabled"])
        self.assertFalse(matrix["venues"]["extended"]["fill_participation_observed"])
        self.assertFalse(matrix["venues"]["extended"]["primary_fill_candidate"])
        self.assertTrue(matrix["venues"]["hyperliquid"]["fill_role_enabled"])
        self.assertFalse(matrix["venues"]["hyperliquid"]["fill_participation_observed"])
        self.assertFalse(matrix["venues"]["hyperliquid"]["primary_fill_candidate"])
        self.assertTrue(matrix["venues"]["paradex"]["fill_role_enabled"])
        self.assertFalse(matrix["venues"]["paradex"]["fill_participation_observed"])
        self.assertFalse(matrix["venues"]["paradex"]["primary_fill_candidate"])

    def test_inferred_hold_blocker_family_detects_restore_hygiene_from_clean_only_failure(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["hypothesis_blocker_family"] = "microstructure_underconversion"
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)
        run_root = self.repo_root / "promotion_runs" / "t_main_1_5m_canary_20260405T000000Z" / "live_canary"
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "live_closeout_bundle.json").write_text(
            json.dumps(
                {
                    "guard_intervened": False,
                    "guard_window_completed": True,
                    "pre_restore_venue_audit_clean": False,
                    "post_rollback_venue_audit_clean": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )

        blocker = self.mod.inferred_hold_blocker_family(
            queue["serialized_mainline"][0],
            {
                "clean": {
                    "passed": False,
                    "failed_rules": [
                        {
                            "source": "closeout",
                            "path": "pre_restore_venue_audit_clean",
                            "op": "==",
                            "expected": True,
                            "actual": False,
                            "severity": "fail",
                        }
                    ],
                },
                "mechanism": {"passed": True, "failed_rules": []},
            },
            run_root,
        )

        self.assertEqual(blocker, "restore_hygiene")

    def test_inferred_hold_blocker_family_detects_restore_hygiene_from_closeout_even_when_clean_passes(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["hypothesis_blocker_family"] = "microstructure_underconversion"
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)
        run_root = self.repo_root / "promotion_runs" / "t_main_1_60m_qual_20260406T000000Z" / "live_canary"
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "live_closeout_bundle.json").write_text(
            json.dumps(
                {
                    "guard_intervened": False,
                    "guard_window_completed": True,
                    "pre_restore_venue_audit_clean": False,
                    "post_rollback_venue_audit_clean": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )

        blocker = self.mod.inferred_hold_blocker_family(
            queue["serialized_mainline"][0],
            {
                "clean": {"passed": True, "failed_rules": []},
                "mechanism": {"passed": True, "failed_rules": []},
            },
            run_root,
        )

        self.assertEqual(blocker, "restore_hygiene")

    def test_inferred_hold_blocker_family_detects_restore_hygiene_from_first_audit_cleanup(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["hypothesis_blocker_family"] = "extended_pre_kill_degraded_rebootstrap_alignment_gap"
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)
        run_root = self.repo_root / "promotion_runs" / "t_main_1_7200s_20260429T000000Z" / "live_canary"
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "live_closeout_bundle.json").write_text(
            json.dumps(
                {
                    "guard_intervened": False,
                    "guard_window_completed": True,
                    "first_pre_restore_venue_audit_clean": False,
                    "pre_restore_cleanup_required": True,
                    "pre_restore_venue_audit_clean": True,
                    "post_rollback_venue_audit_clean": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )

        blocker = self.mod.inferred_hold_blocker_family(
            queue["serialized_mainline"][0],
            {
                "clean": {
                    "passed": False,
                    "failed_rules": [
                        {
                            "source": "closeout",
                            "path": "first_pre_restore_venue_audit_clean",
                            "op": "==",
                            "expected": True,
                            "actual": False,
                            "severity": "fail",
                        }
                    ],
                },
                "mechanism": {"passed": True, "failed_rules": []},
            },
            run_root,
        )

        self.assertEqual(blocker, "restore_hygiene")

    def test_inferred_hold_blocker_family_maps_restart_intervention_to_stale_restart(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["hypothesis_blocker_family"] = "restore_hygiene"
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)
        run_root = self.repo_root / "promotion_runs" / "t_main_1_restart_20260405T000000Z" / "live_canary"
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "live_closeout_bundle.json").write_text(
            json.dumps(
                {
                    "guard_intervened": True,
                    "guard_intervention_reason": "service_restarts=1",
                    "guard_window_completed": False,
                    "pre_restore_venue_audit_clean": False,
                    "post_rollback_venue_audit_clean": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )

        blocker = self.mod.inferred_hold_blocker_family(
            queue["serialized_mainline"][0],
            {
                "clean": {
                    "passed": False,
                    "failed_rules": [
                        {
                            "source": "closeout",
                            "path": "guard_intervened",
                            "op": "==",
                            "expected": False,
                            "actual": True,
                            "severity": "fail",
                        }
                    ],
                },
                "mechanism": {"passed": True, "failed_rules": []},
            },
            run_root,
        )

        self.assertEqual(blocker, "stale_restart")

    def test_inferred_hold_blocker_family_prefers_extended_degraded_stream_rebootstrap_gap(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["hypothesis_blocker_family"] = "restore_hygiene"
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)
        run_root = (
            self.repo_root
            / "promotion_runs"
            / "t_main_1_restart_extended_degraded_20260409T014616Z"
            / "live_canary"
        )
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "live_closeout_bundle.json").write_text(
            json.dumps(
                {
                    "guard_intervened": True,
                    "guard_intervention_reason": "systemd_state=activating/auto-restart",
                    "guard_window_completed": False,
                    "pre_restore_venue_audit_clean": False,
                    "post_rollback_venue_audit_clean": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (run_root / "paraphina_live.err.segment").write_text(
            "\n".join(
                [
                    "WS_AUDIT venue=extended component=post_publish_stream_fallback action=fallback_won attempt_index=1 stream_preference=depth1",
                    "WS_AUDIT venue=extended component=post_publish_stream_fallback action=preference_set attempt_index=1 stream_preference=full_orderbook_degraded",
                    "WS_AUDIT venue=extended component=ws_msg age_data_rx_ms=2997 age_book_event_ms=2997 age_published_ms=2997",
                    "[runner] tick=2917 stale_market_hygiene kill_triggered consecutive_stale_ticks=12 max_ticks=12 stale_market_count=1 stale_venues=extended",
                    "paraphina_live | error=unexpected_live_loop_exit kill_switch=true stale_market_count=1",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        blocker = self.mod.inferred_hold_blocker_family(
            queue["serialized_mainline"][0],
            {
                "clean": {
                    "passed": False,
                    "failed_rules": [
                        {
                            "source": "closeout",
                            "path": "guard_intervened",
                            "op": "==",
                            "expected": False,
                            "actual": True,
                            "severity": "fail",
                        }
                    ],
                },
                "mechanism": {"passed": True, "failed_rules": []},
            },
            run_root,
        )

        self.assertEqual(blocker, "extended_degraded_stream_rebootstrap_gap")

    def test_inferred_hold_blocker_family_prefers_runner_freeze_apply_gap_on_extended_evidence(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["hypothesis_blocker_family"] = "restore_hygiene"
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)
        run_root = self.repo_root / "promotion_runs" / "t_main_1_restart_freeze_20260405T000000Z" / "live_canary"
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "live_closeout_bundle.json").write_text(
            json.dumps(
                {
                    "guard_intervened": True,
                    "guard_intervention_reason": "service_restarts=1",
                    "guard_window_completed": False,
                    "pre_restore_venue_audit_clean": False,
                    "post_rollback_venue_audit_clean": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (run_root / "paraphina_live.err.segment").write_text(
            "\n".join(
                [
                    "FIRST_BOOK_UPDATE venue=extended symbol=extended mid=2047.95 spread=0.1 ts=1",
                    "APPLIED_BOOK venue=extended venue_index=0 mid=2047.95 spread=0.1 depth_usd=1000",
                    "WARN: Extended core book update frozen mid=2078.35 spread=61.5",
                    "[runner] tick=14 stale_market_hygiene kill_triggered consecutive_stale_ticks=12 max_ticks=12 stale_market_count=1 stale_venues=extended",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        blocker = self.mod.inferred_hold_blocker_family(
            queue["serialized_mainline"][0],
            {
                "clean": {
                    "passed": False,
                    "failed_rules": [
                        {
                            "source": "closeout",
                            "path": "guard_intervened",
                            "op": "==",
                            "expected": False,
                            "actual": True,
                            "severity": "fail",
                        }
                    ],
                },
                "mechanism": {"passed": True, "failed_rules": []},
            },
            run_root,
        )

        self.assertEqual(blocker, "runner_freeze_apply_gap")

    def test_inferred_hold_blocker_family_prefers_lighter_no_data_over_post_restore_extended_freeze(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["hypothesis_blocker_family"] = "restore_hygiene"
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)
        run_root = self.repo_root / "promotion_runs" / "t_main_1_lighter_no_data_20260412T000000Z" / "live_canary"
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "live_closeout_bundle.json").write_text(
            json.dumps(
                {
                    "guard_intervened": True,
                    "guard_intervention_reason": "kill_events_present",
                    "guard_window_completed": False,
                    "pre_restore_venue_audit_clean": False,
                    "post_rollback_venue_audit_clean": False,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (run_root / "paraphina_live.err.segment").write_text(
            "\n".join(
                [
                    "INFO: Lighter public WS connecting readonly=false url=wss://mainnet.zklighter.elliot.ai/stream",
                    "INFO: Lighter public WS error (consecutive_failures=1): Lighter public WS connect error: HTTP error: 503 Service Unavailable",
                    "WS_AUDIT venue=lighter component=freshness reason=stale_watchdog_trigger stale_ms=10000",
                    "[runner] tick=4846 stale_market_hygiene kill_triggered consecutive_stale_ticks=12 max_ticks=12 stale_market_count=1 stale_venues=lighter",
                    "FIRST_BOOK_UPDATE venue=extended symbol=extended mid=2047.95 spread=0.1 ts=1",
                    "WARN: Extended core book update frozen mid=2078.35 spread=61.5",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        blocker = self.mod.inferred_hold_blocker_family(
            queue["serialized_mainline"][0],
            {
                "clean": {
                    "passed": False,
                    "failed_rules": [
                        {
                            "source": "closeout",
                            "path": "guard_intervened",
                            "op": "==",
                            "expected": False,
                            "actual": True,
                            "severity": "fail",
                        }
                    ],
                },
                "mechanism": {"passed": True, "failed_rules": []},
            },
            run_root,
        )

        self.assertEqual(blocker, "no_data_transport_gap")

    def test_inferred_hold_blocker_family_prefers_extended_pre_kill_degraded_rebootstrap_alignment_gap(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["hypothesis_blocker_family"] = "restore_hygiene"
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)
        run_root = (
            self.repo_root
            / "promotion_runs"
            / "t_main_1_restart_extended_alignment_20260409T111038Z"
            / "live_canary"
        )
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "live_closeout_bundle.json").write_text(
            json.dumps(
                {
                    "guard_intervened": True,
                    "guard_intervention_reason": "systemd_state=activating/auto-restart",
                    "guard_window_completed": False,
                    "pre_restore_venue_audit_clean": False,
                    "post_rollback_venue_audit_clean": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (run_root / "paraphina_live.err.segment").write_text(
            "\n".join(
                [
                    "WS_AUDIT venue=extended component=post_publish_stream_fallback action=fallback_won attempt_index=1 stream_preference=depth1",
                    "WS_AUDIT venue=extended component=post_publish_stream_fallback action=preference_set attempt_index=1 stream_preference=full_orderbook_degraded",
                    "WS_AUDIT venue=extended component=post_publish_stream_fallback action=degraded_rebootstrap_started post_publish_fallback_after_ms=1200 age_data_rx_ms=5830 age_book_event_ms=5830 age_published_ms=5830",
                    "[runner] tick=979 stale_market_hygiene kill_triggered consecutive_stale_ticks=12 max_ticks=12 stale_market_count=1 stale_venues=extended",
                    "paraphina_live | error=unexpected_live_loop_exit kill_switch=true stale_market_count=1",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        blocker = self.mod.inferred_hold_blocker_family(
            queue["serialized_mainline"][0],
            {
                "clean": {
                    "passed": False,
                    "failed_rules": [
                        {
                            "source": "closeout",
                            "path": "guard_intervened",
                            "op": "==",
                            "expected": False,
                            "actual": True,
                            "severity": "fail",
                        }
                    ],
                },
                "mechanism": {"passed": True, "failed_rules": []},
            },
            run_root,
        )

        self.assertEqual(blocker, "extended_pre_kill_degraded_rebootstrap_alignment_gap")

    def test_inferred_hold_blocker_family_prefers_extended_pre_kill_alignment_on_kill_event_recovery(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["hypothesis_blocker_family"] = "paradex_ui_touch_reference_gap"
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)
        run_root = (
            self.repo_root
            / "promotion_runs"
            / "t_main_1_kill_extended_alignment_20260410T130700Z"
            / "live_canary"
        )
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "live_closeout_bundle.json").write_text(
            json.dumps(
                {
                    "guard_intervened": True,
                    "guard_intervention_reason": "kill_events_present",
                    "guard_window_completed": False,
                    "pre_restore_venue_audit_clean": False,
                    "post_rollback_venue_audit_clean": False,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (run_root / "telemetry_bounded.jsonl").write_text(
            json.dumps(
                {
                    "execution_mode": "live",
                    "stale_market_hygiene": {
                        "stale_venues": ["extended"],
                        "consecutive_stale_ticks": 11,
                        "kill_would_fire_next_tick": True,
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (run_root / "paraphina_live.err.segment").write_text(
            "\n".join(
                [
                    "2026-04-10T13:07:10Z WS_AUDIT venue=paradex component=ui_book_truth source=api status=ok token_usage=interactive",
                    "2026-04-10T13:07:11Z WS_AUDIT venue=paradex component=ui_book_truth source=interactive status=ok token_usage=interactive",
                    "2026-04-10T13:07:12Z WS_AUDIT venue=paradex component=ui_touch_reference action=applied source_kind=split",
                    "2026-04-10T13:07:45Z WS_AUDIT venue=extended component=degraded_stream_watchdog action=fired post_publish_fallback_after_ms=1200 age_data_rx_ms=3093 age_book_event_ms=3093 age_published_ms=3093",
                    "2026-04-10T13:07:46Z session_progress stage=first_publish venue=extended stream_kind=full_orderbook",
                    "2026-04-10T13:07:48Z [runner] tick=188 stale_market_hygiene kill_triggered consecutive_stale_ticks=12 max_ticks=12 stale_market_count=1 stale_venues=extended",
                    "2026-04-10T13:07:48Z paraphina_live | error=unexpected_live_loop_exit kill_switch=true stale_market_count=1",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        blocker = self.mod.inferred_hold_blocker_family(
            queue["serialized_mainline"][0],
            {
                "clean": {"passed": False, "failed_rules": []},
                "mechanism": {"passed": True, "failed_rules": []},
            },
            run_root,
        )

        self.assertEqual(blocker, "extended_pre_kill_degraded_rebootstrap_alignment_gap")

    def test_inferred_hold_blocker_family_prefers_extended_pre_kill_alignment_without_unexpected_exit_line(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["hypothesis_blocker_family"] = "paradex_ui_touch_reference_gap"
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)
        run_root = (
            self.repo_root
            / "promotion_runs"
            / "t_main_1_kill_extended_alignment_recovered_20260410T130700Z"
            / "live_canary"
        )
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "live_closeout_bundle.json").write_text(
            json.dumps(
                {
                    "guard_intervened": True,
                    "guard_intervention_reason": "kill_events_present",
                    "guard_window_completed": False,
                    "pre_restore_venue_audit_clean": False,
                    "post_rollback_venue_audit_clean": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (run_root / "telemetry_bounded.jsonl").write_text(
            json.dumps(
                {
                    "execution_mode": "live",
                    "stale_market_hygiene": {
                        "stale_venues": ["extended"],
                        "consecutive_stale_ticks": 11,
                        "kill_would_fire_next_tick": True,
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (run_root / "paraphina_live.err.segment").write_text(
            "\n".join(
                [
                    "2026-04-10T13:07:10Z WS_AUDIT venue=paradex component=ui_book_truth source=api status=ok token_usage=interactive",
                    "2026-04-10T13:07:11Z WS_AUDIT venue=paradex component=ui_book_truth source=interactive status=ok token_usage=interactive",
                    "2026-04-10T13:07:12Z WS_AUDIT venue=paradex component=ui_touch_reference action=applied source_kind=split",
                    "2026-04-10T13:07:45Z WS_AUDIT venue=extended component=degraded_stream_watchdog action=fired post_publish_fallback_after_ms=1200 age_data_rx_ms=3093 age_book_event_ms=3093 age_published_ms=3093",
                    "2026-04-10T13:07:46Z WS_AUDIT venue=extended component=session_progress stage=first_publish socket_role=primary stream_kind=full_orderbook ws_upgrade_completed=1 time_to_first_message_ms=5 time_to_first_book_ms=13 time_to_first_publish_ms=13",
                    "2026-04-10T13:07:48Z [runner] tick=188 stale_market_hygiene kill_triggered consecutive_stale_ticks=12 max_ticks=12 stale_market_count=1 stale_venues=extended",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        blocker = self.mod.inferred_hold_blocker_family(
            queue["serialized_mainline"][0],
            {
                "clean": {"passed": False, "failed_rules": []},
                "mechanism": {"passed": True, "failed_rules": []},
            },
            run_root,
        )

        self.assertEqual(blocker, "extended_pre_kill_degraded_rebootstrap_alignment_gap")

    def test_inferred_hold_blocker_family_prefers_transport_gap_watchdog_on_extended_collapse(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["hypothesis_blocker_family"] = "restore_hygiene"
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)
        run_root = self.repo_root / "promotion_runs" / "t_main_1_restart_transport_20260405T000000Z" / "live_canary"
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "live_closeout_bundle.json").write_text(
            json.dumps(
                {
                    "guard_intervened": True,
                    "guard_intervention_reason": "systemd_state=unknown/unknown",
                    "guard_window_completed": False,
                    "pre_restore_venue_audit_clean": False,
                    "post_rollback_venue_audit_clean": False,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (run_root / "paraphina_live.err.segment").write_text(
            "\n".join(
                [
                    "WS_AUDIT venue=extended reconnect_reason=stale_watchdog count=2",
                    "[runner] tick=6616 stale_market_hygiene kill_triggered consecutive_stale_ticks=12 max_ticks=12 stale_market_count=5 stale_venues=extended,hyperliquid,aster,lighter,paradex",
                    "paraphina_live | error=unexpected_live_loop_exit trade_mode=live ticks_run=6616 kill_switch=true ready_market_count=0 stale_market_count=5 fv_available=false",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        blocker = self.mod.inferred_hold_blocker_family(
            queue["serialized_mainline"][0],
            {
                "clean": {
                    "passed": False,
                    "failed_rules": [
                        {
                            "source": "closeout",
                            "path": "guard_intervened",
                            "op": "==",
                            "expected": False,
                            "actual": True,
                            "severity": "fail",
                        }
                    ],
                },
                "mechanism": {"passed": True, "failed_rules": []},
            },
            run_root,
        )

        self.assertEqual(blocker, "transport_gap_watchdog")

    def test_inferred_hold_blocker_family_prefers_aster_bridge_wait_timeout_on_bounded_pre_exit_evidence(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["hypothesis_blocker_family"] = "transport_gap_watchdog"
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)
        run_root = self.repo_root / "promotion_runs" / "t_main_1_restart_aster_bridge_20260405T000000Z" / "live_canary"
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "live_closeout_bundle.json").write_text(
            json.dumps(
                {
                    "guard_intervened": True,
                    "guard_intervention_reason": "service_restarts=1",
                    "guard_window_completed": False,
                    "pre_restore_venue_audit_clean": False,
                    "post_rollback_venue_audit_clean": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (run_root / "paraphina_live.err.segment").write_text(
            "\n".join(
                [
                    "WS_AUDIT venue=aster component=book_recovery stage=snapshot_wait_bridge phase=loop1 snap_id=1 buffered_before=0",
                    "WS_AUDIT venue=aster component=book_recovery stage=snapshot_wait_bridge phase=loop1 snap_id=2 buffered_before=0",
                    "WS_AUDIT venue=aster component=book_recovery stage=snapshot_wait_bridge phase=loop1 snap_id=3 buffered_before=0",
                    "WS_AUDIT venue=aster component=book_recovery stage=snapshot_wait_bridge phase=loop1 snap_id=4 buffered_before=0",
                    "WS_AUDIT venue=aster component=book_recovery stage=snapshot_wait_bridge phase=loop1 snap_id=5 buffered_before=0",
                    "WS_AUDIT venue=aster component=book_recovery stage=snapshot_fetch_failed phase=loop1 fail_streak=1",
                    "WS_AUDIT venue=aster reconnect_reason=stale_watchdog count=1",
                    "[runner] tick=14 stale_market_hygiene kill_triggered consecutive_stale_ticks=12 max_ticks=12 stale_market_count=1 stale_venues=aster",
                    "paraphina_live | error=unexpected_live_loop_exit trade_mode=live ticks_run=14 kill_switch=true ready_market_count=4 stale_market_count=1 fv_available=true",
                    "WARN: Extended core book update frozen mid=2055.15 spread=123.5",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        blocker = self.mod.inferred_hold_blocker_family(
            queue["serialized_mainline"][0],
            {
                "clean": {
                    "passed": False,
                    "failed_rules": [
                        {
                            "source": "closeout",
                            "path": "guard_intervened",
                            "op": "==",
                            "expected": False,
                            "actual": True,
                            "severity": "fail",
                        }
                    ],
                },
                "mechanism": {"passed": True, "failed_rules": []},
            },
            run_root,
        )

        self.assertEqual(blocker, "aster_bridge_wait_timeout")

    def test_inferred_hold_blocker_family_prefers_startup_readiness_gap_on_early_bootstrap_stale(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["hypothesis_blocker_family"] = "no_data_transport_gap"
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)
        run_root = self.repo_root / "promotion_runs" / "t_main_1_restart_startup_readiness_20260406T000000Z" / "live_canary"
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "live_closeout_bundle.json").write_text(
            json.dumps(
                {
                    "guard_intervened": True,
                    "guard_intervention_reason": "service_restarts=1",
                    "guard_window_completed": False,
                    "pre_restore_venue_audit_clean": False,
                    "post_rollback_venue_audit_clean": True,
                    "tick_count": 15,
                    "mm_place_total": 0,
                    "fill_count_total": 0,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (run_root / "paraphina_live.err.segment").write_text(
            "\n".join(
                [
                    "FIRST_BOOK_UPDATE venue=extended symbol=extended mid=2047.95 spread=0.1 ts=1",
                    "WS_AUDIT venue=extended component=runner_apply_truth publish_ok=1 age_apply_ms=2604 age_event_ms=2604",
                    "WS_AUDIT venue=paradex component=profile_usage token_usage=interactive observed=1",
                    "[runner] tick=11 stale_market_hygiene kill_triggered consecutive_stale_ticks=12 max_ticks=12 stale_market_count=2 stale_venues=extended,paradex",
                    "paraphina_live | error=unexpected_live_loop_exit trade_mode=live ticks_run=15 kill_switch=true ready_market_count=3 stale_market_count=2 fv_available=false",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        blocker = self.mod.inferred_hold_blocker_family(
            queue["serialized_mainline"][0],
            {
                "clean": {
                    "passed": False,
                    "failed_rules": [
                        {
                            "source": "closeout",
                            "path": "guard_intervened",
                            "op": "==",
                            "expected": False,
                            "actual": True,
                            "severity": "fail",
                        }
                    ],
                },
                "mechanism": {"passed": True, "failed_rules": []},
            },
            run_root,
        )

        self.assertEqual(blocker, "startup_readiness_gap")

    def test_inferred_hold_blocker_family_prefers_hyperliquid_post_publish_transport_gap(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["hypothesis_blocker_family"] = "paradex_edge_floor_queue_loss"
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)
        run_root = self.repo_root / "promotion_runs" / "t_main_1_restart_hl_post_publish_20260406T000000Z" / "live_canary"
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "live_closeout_bundle.json").write_text(
            json.dumps(
                {
                    "guard_intervened": True,
                    "guard_intervention_reason": "systemd_state=activating/auto-restart",
                    "guard_window_completed": False,
                    "pre_restore_venue_audit_clean": False,
                    "post_rollback_venue_audit_clean": True,
                    "tick_count": 8880,
                    "mm_place_total": 141,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (run_root / "paraphina_live.err.segment").write_text(
            "\n".join(
                [
                    "FIRST_BOOK_UPDATE venue=hyperliquid symbol=hyperliquid mid=2100.05 spread=0.1 ts=1",
                    "WS_AUDIT venue=hyperliquid component=hl_pubq reason=periodic interval_ms=1000 queue_cap=256 queued_len=1 queued_hiwater=1 pending_latest_present=0 pending_overwrite=0 pending_lock_fail=0 ts_zero_count=0 ws_rx_age_ms=0 data_rx_age_ms=0 pub_age_ms=1476 book_age_ms=0 pub_minus_book_age_ms=1476 send_block_max_ms=0 send_block_gt_5ms=0 send_block_gt_50ms=0 send_block_gt_250ms=0 forward_send_count=1 forward_send_err_count=0 coalesced_drop_count=0 pending_take_count=0 ts_missing_or_zero_count=0 ts_clamped_past_skew_count=0 ts_clamped_future_skew_count=0 ts_policy_enabled=0 ts_policy_applied_count=0 ts_kept_exchange_count=0 ts_past_skew_max_ms=1189 ts_future_skew_max_ms=0 try_send_ok=1 try_send_full=0 emit_since_ms=1476",
                    "[runner] tick=8879 stale_market_hygiene kill_triggered consecutive_stale_ticks=12 max_ticks=12 stale_market_count=1 stale_venues=hyperliquid",
                    "paraphina_live | error=unexpected_live_loop_exit trade_mode=live ticks_run=8879 kill_switch=true ready_market_count=4 stale_market_count=1 fv_available=true",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        blocker = self.mod.inferred_hold_blocker_family(
            queue["serialized_mainline"][0],
            {
                "clean": {
                    "passed": False,
                    "failed_rules": [
                        {
                            "source": "closeout",
                            "path": "guard_intervened",
                            "op": "==",
                            "expected": False,
                            "actual": True,
                            "severity": "fail",
                        }
                    ],
                },
                "mechanism": {"passed": True, "failed_rules": []},
            },
            run_root,
        )

        self.assertEqual(blocker, "hyperliquid_post_publish_transport_gap")

    def test_inferred_hold_blocker_family_prefers_paradex_edge_floor_queue_loss_after_clean_interactive_top_hold(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["hypothesis_blocker_family"] = "paradex_interactive_top_anchor_gap"
        queue["serialized_mainline"][0]["clean_final_hold_blocker_family"] = "microstructure_underconversion"
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)
        run_root = self.repo_root / "promotion_runs" / "t_main_1_paradex_edge_floor_hold_20260409T000000Z" / "live_canary"
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "live_closeout_bundle.json").write_text(
            json.dumps(
                {
                    "guard_intervened": False,
                    "guard_window_completed": True,
                    "pre_restore_venue_audit_clean": True,
                    "post_rollback_venue_audit_clean": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (run_root / "live_metrics.json").write_text(
            json.dumps(
                {
                    "execution_scorecard": {
                        "paradex": {
                            "fills": 0,
                            "place_ack": 1,
                            "place_i": 1,
                        }
                    },
                    "orders_per_venue": {"paradex": 500},
                    "paradex_profile_usage_summary": {
                        "interactive_token_usage_observed": True,
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (run_root / "paraphina_live.err.segment").write_text(
            "PARADEX_INTERACTIVE_PUBLIC_TOP count=1 source=interactive_orderbook top_source=api_best bid=2200 ask=2200.4\n",
            encoding="utf-8",
        )
        (run_root / "telemetry_bounded.jsonl").write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "execution_mode": "live",
                            "quote_levels": [
                                {
                                    "venue_id": "paradex",
                                    "quote_state": "active",
                                    "suppression_reason": "",
                                },
                                {
                                    "venue_id": "paradex",
                                    "quote_state": "suppressed",
                                    "suppression_reason": "edge_below_min",
                                },
                            ],
                        }
                    ),
                    json.dumps(
                        {
                            "execution_mode": "live",
                            "quote_levels": [
                                {
                                    "venue_id": "paradex",
                                    "quote_state": "suppressed",
                                    "suppression_reason": "edge_below_min",
                                }
                            ]
                            * 1200,
                        }
                    ),
                    json.dumps(
                        {
                            "execution_mode": "live",
                            "quote_levels": [
                                {
                                    "venue_id": "paradex",
                                    "quote_state": "suppressed",
                                    "suppression_reason": "generated_spread_cap",
                                }
                            ]
                            * 100,
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        blocker = self.mod.inferred_hold_blocker_family(
            queue["serialized_mainline"][0],
            {
                "clean": {"passed": True, "failed_rules": []},
                "mechanism": {"passed": True, "failed_rules": []},
                "promotion": {"passed": True, "failed_rules": []},
                "is_final_rung": True,
                "suggested_action": "hold",
            },
            run_root,
        )

        self.assertEqual(blocker, "paradex_edge_floor_queue_loss")

    def test_inferred_hold_blocker_family_prefers_paradex_same_side_persistence_gap_over_edge_floor_loss(self):
        queue = _minimal_queue()
        tranche = queue["serialized_mainline"][0]
        tranche["hypothesis_blocker_family"] = "paradex_edge_floor_shadow_mechanism_gate_gap"
        tranche["clean_final_hold_blocker_family"] = "microstructure_underconversion"
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)
        run_root = self.repo_root / "promotion_runs" / "t_main_1_paradex_same_side_hold_20260410T000000Z" / "live_canary"
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "live_closeout_bundle.json").write_text(
            json.dumps(
                {
                    "guard_intervened": False,
                    "guard_window_completed": True,
                    "pre_restore_venue_audit_clean": True,
                    "post_rollback_venue_audit_clean": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (run_root / "live_metrics.json").write_text(
            json.dumps(
                {
                    "execution_scorecard": {
                        "paradex": {
                            "fills": 0,
                            "place_ack": 8,
                            "place_i": 10,
                        }
                    },
                    "orders_per_venue": {"paradex": 13141},
                    "paradex_profile_usage_summary": {
                        "interactive_token_usage_observed": True,
                    },
                    "supported_replace_visibility": {
                        "paradex": {
                            "opportunities": 1,
                            "misses": 11,
                            "gap_grace": 2,
                            "actions": {"place": 10, "keep": 5},
                            "blockers": {
                                "no_current_same_side": 10,
                                "current_too_young": 1,
                            },
                        }
                    },
                    "mm_keep_replace_by_venue": {
                        "paradex": {"keep": 1, "replace": 0},
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (run_root / "paraphina_live.err.segment").write_text(
            "PARADEX_INTERACTIVE_PUBLIC_TOP count=172 source=interactive_orderbook top_source=api_best bid=2200 ask=2200.4\n",
            encoding="utf-8",
        )
        (run_root / "telemetry_bounded.jsonl").write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "execution_mode": "live",
                            "quote_levels": [
                                {
                                    "venue_id": "paradex",
                                    "quote_state": "active",
                                    "suppression_reason": "",
                                },
                                {
                                    "venue_id": "paradex",
                                    "quote_state": "suppressed",
                                    "suppression_reason": "edge_below_min",
                                },
                            ],
                        }
                    ),
                    json.dumps(
                        {
                            "execution_mode": "live",
                            "quote_levels": [
                                {
                                    "venue_id": "paradex",
                                    "quote_state": "suppressed",
                                    "suppression_reason": "edge_below_min",
                                }
                            ]
                            * 1200,
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        blocker = self.mod.inferred_hold_blocker_family(
            tranche,
            {
                "clean": {"passed": True, "failed_rules": []},
                "mechanism": {"passed": True, "failed_rules": []},
                "promotion": {"passed": True, "failed_rules": []},
                "is_final_rung": True,
                "suggested_action": "hold",
            },
            run_root,
        )

        self.assertEqual(blocker, "paradex_same_side_persistence_gap")

    def test_inferred_hold_blocker_family_uses_mechanism_fail_override_on_clean_mechanism_miss(self):
        queue = _minimal_queue()
        tranche = queue["serialized_mainline"][0]
        tranche["hypothesis_blocker_family"] = "restore_hygiene"
        tranche["mechanism_fail_blocker_family"] = "paradex_private_order_truth_gap"
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)
        run_root = self.repo_root / "promotion_runs" / "t_main_1_mechanism_miss_20260406T000000Z" / "live_canary"
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "live_closeout_bundle.json").write_text(
            json.dumps(
                {
                    "guard_intervened": False,
                    "guard_window_completed": True,
                    "pre_restore_venue_audit_clean": True,
                    "post_rollback_venue_audit_clean": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )

        blocker = self.mod.inferred_hold_blocker_family(
            tranche,
            {
                "clean": {"passed": True, "failed_rules": []},
                "mechanism": {
                    "passed": False,
                    "failed_rules": [
                        {
                            "source": "metrics",
                            "path": "supported_replace_visibility.paradex.gap_grace",
                            "op": ">",
                            "expected": 0,
                            "actual": 0,
                            "severity": "fail",
                        }
                    ],
                },
                "is_final_rung": False,
                "suggested_action": "hold",
            },
            run_root,
        )

        self.assertEqual(blocker, "paradex_private_order_truth_gap")

    def test_inferred_hold_blocker_family_uses_clean_final_hold_override(self):
        queue = _minimal_queue()
        tranche = queue["serialized_mainline"][0]
        tranche["hypothesis_blocker_family"] = "paradex_private_order_truth_gap"
        tranche["clean_final_hold_blocker_family"] = "microstructure_underconversion"
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)
        run_root = self.repo_root / "promotion_runs" / "t_main_1_final_hold_20260406T000000Z" / "live_canary"
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "live_closeout_bundle.json").write_text(
            json.dumps(
                {
                    "guard_intervened": False,
                    "guard_window_completed": True,
                    "pre_restore_venue_audit_clean": True,
                    "post_rollback_venue_audit_clean": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )

        blocker = self.mod.inferred_hold_blocker_family(
            tranche,
            {
                "clean": {"passed": True, "failed_rules": []},
                "mechanism": {"passed": True, "failed_rules": []},
                "is_final_rung": True,
                "suggested_action": "hold",
            },
            run_root,
        )

        self.assertEqual(blocker, "microstructure_underconversion")

    def test_inferred_hold_blocker_family_prefers_all5_projected_mm_budget_distribution_gap(self):
        queue = _minimal_queue()
        tranche = queue["serialized_mainline"][0]
        tranche["hypothesis_blocker_family"] = "restore_hygiene"
        tranche["clean_final_hold_blocker_family"] = "microstructure_underconversion"
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)
        run_root = self.repo_root / "promotion_runs" / "t_main_1_projected_budget_20260410T000000Z" / "live_canary"
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "live_closeout_bundle.json").write_text(
            json.dumps(
                {
                    "guard_intervened": False,
                    "guard_window_completed": True,
                    "pre_restore_venue_audit_clean": True,
                    "post_rollback_venue_audit_clean": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (run_root / "live_metrics.json").write_text(
            json.dumps(
                {
                    "fills": {
                        "by_venue": {
                            "extended": {"fill_count": 0},
                            "hyperliquid": {"fill_count": 0},
                            "aster": {"fill_count": 0},
                            "lighter": {"fill_count": 2},
                            "paradex": {"fill_count": 2},
                        }
                    },
                    "risk": {"would_send_zero_pct": 90.4},
                    "projected_mm_budget_summary": {
                        "configured_ticks": 14482,
                        "applied_ticks": 455,
                        "selected_counts": {
                            "extended": 263,
                            "hyperliquid": 153,
                            "aster": 463,
                            "lighter": 886,
                            "paradex": 877,
                        },
                        "suppressed_counts": {
                            "extended": 61,
                            "hyperliquid": 152,
                            "aster": 330,
                            "lighter": 49,
                            "paradex": 49,
                        },
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )

        blocker = self.mod.inferred_hold_blocker_family(
            tranche,
            {
                "clean": {"passed": True, "failed_rules": []},
                "mechanism": {"passed": True, "failed_rules": []},
                "is_final_rung": True,
                "suggested_action": "hold",
            },
            run_root,
        )

        self.assertEqual(blocker, "all5_projected_mm_budget_distribution_gap")

    def test_inferred_hold_blocker_family_prefers_all5_budget_over_paradex_edge_when_global_quote_selection_collapses(self):
        queue = _minimal_queue()
        tranche = queue["serialized_mainline"][0]
        tranche["hypothesis_blocker_family"] = "microstructure_underconversion"
        tranche["clean_final_hold_blocker_family"] = "microstructure_underconversion"
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)
        run_root = self.repo_root / "promotion_runs" / "t_main_1_all5_budget_plus_paradex_edge_20260412T000000Z" / "live_canary"
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "live_closeout_bundle.json").write_text(
            json.dumps(
                {
                    "guard_intervened": False,
                    "guard_window_completed": True,
                    "pre_restore_venue_audit_clean": True,
                    "post_rollback_venue_audit_clean": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (run_root / "live_metrics.json").write_text(
            json.dumps(
                {
                    "fills": {
                        "by_venue": {
                            "extended": {"fill_count": 0},
                            "hyperliquid": {"fill_count": 2},
                            "aster": {"fill_count": 0},
                            "lighter": {"fill_count": 0},
                            "paradex": {"fill_count": 0},
                        }
                    },
                    "risk": {"would_send_zero_pct": 99.7},
                    "projected_mm_budget_summary": {
                        "configured_ticks": 28961,
                        "applied_ticks": 19289,
                        "selected_counts": {
                            "extended": 34,
                            "hyperliquid": 60,
                            "aster": 34,
                            "lighter": 37,
                            "paradex": 34,
                        },
                        "suppressed_counts": {
                            "extended": 3414,
                            "hyperliquid": 5788,
                            "aster": 17971,
                            "lighter": 19285,
                            "paradex": 18824,
                        },
                    },
                    "execution_scorecard": {
                        "paradex": {
                            "fills": 0,
                            "place_ack": 1,
                            "place_i": 8,
                        }
                    },
                    "orders_per_venue": {"paradex": 109},
                    "paradex_profile_usage_summary": {
                        "interactive_token_usage_observed": True,
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (run_root / "paraphina_live.err.segment").write_text(
            "PARADEX_INTERACTIVE_PUBLIC_TOP count=1 source=interactive_orderbook top_source=api_best bid=2200 ask=2200.4\n",
            encoding="utf-8",
        )
        (run_root / "telemetry_bounded.jsonl").write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "execution_mode": "live",
                            "quote_levels": [
                                {
                                    "venue_id": "paradex",
                                    "quote_state": "active",
                                    "suppression_reason": "",
                                },
                                {
                                    "venue_id": "paradex",
                                    "quote_state": "suppressed",
                                    "suppression_reason": "edge_below_min",
                                },
                            ],
                        }
                    ),
                    json.dumps(
                        {
                            "execution_mode": "live",
                            "quote_levels": [
                                {
                                    "venue_id": "paradex",
                                    "quote_state": "suppressed",
                                    "suppression_reason": "edge_below_min",
                                }
                            ]
                            * 1200,
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        blocker = self.mod.inferred_hold_blocker_family(
            tranche,
            {
                "clean": {"passed": True, "failed_rules": []},
                "mechanism": {"passed": True, "failed_rules": []},
                "is_final_rung": True,
                "suggested_action": "hold",
            },
            run_root,
        )

        self.assertEqual(blocker, "all5_projected_mm_budget_distribution_gap")

    def test_inferred_hold_blocker_family_prefers_paradex_interactive_top_anchor_gap(self):
        queue = _minimal_queue()
        tranche = queue["serialized_mainline"][0]
        tranche["hypothesis_blocker_family"] = "extended_pre_kill_degraded_rebootstrap_alignment_gap"
        tranche["clean_final_hold_blocker_family"] = "microstructure_underconversion"
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)
        run_root = self.repo_root / "promotion_runs" / "t_main_1_final_hold_paradex_anchor_20260409T000000Z" / "live_canary"
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "live_closeout_bundle.json").write_text(
            json.dumps(
                {
                    "guard_intervened": False,
                    "guard_window_completed": True,
                    "pre_restore_venue_audit_clean": True,
                    "post_rollback_venue_audit_clean": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (run_root / "live_metrics.json").write_text(
            json.dumps(
                {
                    "execution_scorecard": {
                        "paradex": {"fills": 0, "place_ack": 12, "place_i": 13},
                    },
                    "orders_per_venue": {"paradex": 20930},
                    "paradex_profile_usage_summary": {"interactive_token_usage_observed": True},
                    "paradex_ui_book_truth_summary": {
                        "observed": True,
                        "last_api_top": {"bid_px": 2164.34, "ask_px": 2164.87},
                        "last_interactive_top": {"bid_px": 2164.56, "ask_px": 2164.72},
                    },
                    "paradex_ui_touch_reference_summary": {
                        "observed": True,
                        "applied_count": 13625,
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )

        blocker = self.mod.inferred_hold_blocker_family(
            tranche,
            {
                "clean": {"passed": True, "failed_rules": []},
                "mechanism": {"passed": True, "failed_rules": []},
                "is_final_rung": True,
                "suggested_action": "hold",
            },
            run_root,
        )

        self.assertEqual(blocker, "paradex_interactive_top_anchor_gap")

    def test_inferred_hold_blocker_family_does_not_reopen_labeled_paradex_interactive_top_anchor(self):
        queue = _minimal_queue()
        tranche = queue["serialized_mainline"][0]
        tranche["hypothesis_blocker_family"] = "extended_pre_kill_degraded_rebootstrap_alignment_gap"
        tranche["clean_final_hold_blocker_family"] = "microstructure_underconversion"
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)
        run_root = self.repo_root / "promotion_runs" / "t_main_1_final_hold_labeled_paradex_anchor_20260411T000000Z" / "live_canary"
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "live_closeout_bundle.json").write_text(
            json.dumps(
                {
                    "guard_intervened": False,
                    "guard_window_completed": True,
                    "pre_restore_venue_audit_clean": True,
                    "post_rollback_venue_audit_clean": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (run_root / "live_metrics.json").write_text(
            json.dumps(
                {
                    "execution_scorecard": {
                        "paradex": {"fills": 0, "place_ack": 7, "place_i": 9},
                    },
                    "orders_per_venue": {"paradex": 14365},
                    "paradex_profile_usage_summary": {"interactive_token_usage_observed": True},
                    "paradex_interactive_top_summary": {
                        "public_top_source_counts": {
                            "api_best": 6,
                            "interactive_top_level_fallback": 68,
                        },
                        "interactive_top_level_fallback_present": True,
                    },
                    "paradex_ui_book_truth_summary": {
                        "observed": True,
                        "last_api_top": {"bid_px": 2311.47, "ask_px": 2312.23},
                        "last_interactive_top": {"bid_px": 2311.55, "ask_px": 2311.79},
                    },
                    "paradex_ui_touch_reference_summary": {
                        "observed": True,
                        "applied_count": 17157,
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )

        blocker = self.mod.inferred_hold_blocker_family(
            tranche,
            {
                "clean": {"passed": True, "failed_rules": []},
                "mechanism": {"passed": True, "failed_rules": []},
                "is_final_rung": True,
                "suggested_action": "hold",
            },
            run_root,
        )

        self.assertEqual(blocker, "microstructure_underconversion")

    def test_inferred_hold_blocker_family_prefers_paradex_ui_touch_reference_gap(self):
        queue = _minimal_queue()
        tranche = queue["serialized_mainline"][0]
        tranche["hypothesis_blocker_family"] = "paradex_same_side_persistence_gap"
        tranche["clean_final_hold_blocker_family"] = "microstructure_underconversion"
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)
        run_root = self.repo_root / "promotion_runs" / "t_main_1_final_hold_paradex_touch_20260410T000000Z" / "live_canary"
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "live_closeout_bundle.json").write_text(
            json.dumps(
                {
                    "guard_intervened": False,
                    "guard_window_completed": True,
                    "pre_restore_venue_audit_clean": True,
                    "post_rollback_venue_audit_clean": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (run_root / "live_metrics.json").write_text(
            json.dumps(
                {
                    "execution_scorecard": {
                        "paradex": {"fills": 0, "place_ack": 12, "place_i": 13},
                    },
                    "orders_per_venue": {"paradex": 14324},
                    "supported_replace_visibility": {
                        "paradex": {
                            "opportunities": 3,
                            "actions": {"keep": 4, "replace": 1},
                        }
                    },
                    "mm_keep_replace_by_venue": {"paradex": {"keep": 4}},
                    "paradex_profile_usage_summary": {"interactive_token_usage_observed": True},
                    "paradex_ui_book_truth_summary": {"observed": False},
                    "paradex_interactive_top_summary": {"records": 0},
                    "paradex_ui_touch_reference_summary": {
                        "observed": False,
                        "applied_count": 0,
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (run_root / "paraphina_live.err.segment").write_text(
            "PARADEX_INTERACTIVE_PUBLIC_TOP count=1 source=interactive_orderbook top_source=api_best bid_px=2164.34 bid_sz=1 ask_px=2164.87 ask_sz=1\n",
            encoding="utf-8",
        )

        blocker = self.mod.inferred_hold_blocker_family(
            tranche,
            {
                "clean": {"passed": True, "failed_rules": []},
                "mechanism": {"passed": True, "failed_rules": []},
                "is_final_rung": True,
                "suggested_action": "hold",
            },
            run_root,
        )

        self.assertEqual(blocker, "paradex_ui_touch_reference_gap")

    def test_record_result_hold_activates_restore_hygiene_route(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["matched_fail_routes"]["restore_hygiene"] = "t_main_2"
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        self.mod.record_result(
            "t_main_1",
            "hold",
            repo_root=self.repo_root,
            observed_blocker_family="restore_hygiene",
        )

        queue_after = yaml.safe_load((self.repo_root / "phase5" / "queue.yaml").read_text(encoding="utf-8"))
        self.assertEqual(queue_after["serialized_mainline"][1]["status"], "ready")
        history = queue_after["serialized_mainline"][0]["history"][-1]
        self.assertTrue(history["child_activation_allowed"])
        self.assertEqual(history["activated_child"], "t_main_2")

    def test_record_result_hold_activates_runner_freeze_route(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["matched_fail_routes"]["runner_freeze_apply_gap"] = "t_main_2"
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        self.mod.record_result(
            "t_main_1",
            "hold",
            repo_root=self.repo_root,
            observed_blocker_family="runner_freeze_apply_gap",
        )

        queue_after = yaml.safe_load((self.repo_root / "phase5" / "queue.yaml").read_text(encoding="utf-8"))
        self.assertEqual(queue_after["serialized_mainline"][1]["status"], "ready")
        history = queue_after["serialized_mainline"][0]["history"][-1]
        self.assertTrue(history["child_activation_allowed"])
        self.assertEqual(history["activated_child"], "t_main_2")

    def test_record_result_hold_activates_transport_gap_route(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["matched_fail_routes"]["transport_gap_watchdog"] = "t_main_2"
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        self.mod.record_result(
            "t_main_1",
            "hold",
            repo_root=self.repo_root,
            observed_blocker_family="transport_gap_watchdog",
        )

        queue_after = yaml.safe_load((self.repo_root / "phase5" / "queue.yaml").read_text(encoding="utf-8"))
        self.assertEqual(queue_after["serialized_mainline"][1]["status"], "ready")
        history = queue_after["serialized_mainline"][0]["history"][-1]
        self.assertTrue(history["child_activation_allowed"])
        self.assertEqual(history["activated_child"], "t_main_2")

    def test_record_result_hold_activates_aster_bridge_wait_route(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["matched_fail_routes"]["aster_bridge_wait_timeout"] = "t_main_2"
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        self.mod.record_result(
            "t_main_1",
            "hold",
            repo_root=self.repo_root,
            observed_blocker_family="aster_bridge_wait_timeout",
        )

        queue_after = yaml.safe_load((self.repo_root / "phase5" / "queue.yaml").read_text(encoding="utf-8"))
        self.assertEqual(queue_after["serialized_mainline"][1]["status"], "ready")
        history = queue_after["serialized_mainline"][0]["history"][-1]
        self.assertTrue(history["child_activation_allowed"])
        self.assertEqual(history["activated_child"], "t_main_2")

    def test_record_result_hold_activates_paradex_replace_identity_route(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["matched_fail_routes"]["paradex_replace_identity_gap"] = "t_main_2"
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        self.mod.record_result(
            "t_main_1",
            "hold",
            repo_root=self.repo_root,
            observed_blocker_family="paradex_replace_identity_gap",
        )

        queue_after = yaml.safe_load((self.repo_root / "phase5" / "queue.yaml").read_text(encoding="utf-8"))
        self.assertEqual(queue_after["serialized_mainline"][1]["status"], "ready")
        history = queue_after["serialized_mainline"][0]["history"][-1]
        self.assertTrue(history["child_activation_allowed"])
        self.assertEqual(history["activated_child"], "t_main_2")

    def test_record_result_hold_activates_paradex_same_side_persistence_route(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["matched_fail_routes"]["paradex_same_side_persistence_gap"] = "t_main_2"
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        self.mod.record_result(
            "t_main_1",
            "hold",
            repo_root=self.repo_root,
            observed_blocker_family="paradex_same_side_persistence_gap",
        )

        queue_after = yaml.safe_load((self.repo_root / "phase5" / "queue.yaml").read_text(encoding="utf-8"))
        self.assertEqual(queue_after["serialized_mainline"][1]["status"], "ready")
        history = queue_after["serialized_mainline"][0]["history"][-1]
        self.assertTrue(history["child_activation_allowed"])
        self.assertEqual(history["activated_child"], "t_main_2")

    def test_record_result_hold_activates_paradex_batch_cancel_request_shape_route(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["matched_fail_routes"]["paradex_batch_cancel_request_shape_gap"] = "t_main_2"
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        self.mod.record_result(
            "t_main_1",
            "hold",
            repo_root=self.repo_root,
            observed_blocker_family="paradex_batch_cancel_request_shape_gap",
        )

        queue_after = yaml.safe_load((self.repo_root / "phase5" / "queue.yaml").read_text(encoding="utf-8"))
        self.assertEqual(queue_after["serialized_mainline"][1]["status"], "ready")
        history = queue_after["serialized_mainline"][0]["history"][-1]
        self.assertTrue(history["child_activation_allowed"])
        self.assertEqual(history["activated_child"], "t_main_2")

    def test_record_result_hold_activates_no_data_transport_gap_route(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["matched_fail_routes"]["no_data_transport_gap"] = "t_main_2"
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        self.mod.record_result(
            "t_main_1",
            "hold",
            repo_root=self.repo_root,
            observed_blocker_family="no_data_transport_gap",
        )

        queue_after = yaml.safe_load((self.repo_root / "phase5" / "queue.yaml").read_text(encoding="utf-8"))
        self.assertEqual(queue_after["serialized_mainline"][1]["status"], "ready")
        history = queue_after["serialized_mainline"][0]["history"][-1]
        self.assertTrue(history["child_activation_allowed"])
        self.assertEqual(history["activated_child"], "t_main_2")

    def test_record_result_hold_activates_topology_fv_reentry_gap_route(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["matched_fail_routes"]["topology_fv_reentry_gap"] = "t_main_2"
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        self.mod.record_result(
            "t_main_1",
            "hold",
            repo_root=self.repo_root,
            observed_blocker_family="topology_fv_reentry_gap",
        )

        queue_after = yaml.safe_load((self.repo_root / "phase5" / "queue.yaml").read_text(encoding="utf-8"))
        self.assertEqual(queue_after["serialized_mainline"][1]["status"], "ready")
        history = queue_after["serialized_mainline"][0]["history"][-1]
        self.assertTrue(history["child_activation_allowed"])
        self.assertEqual(history["activated_child"], "t_main_2")

    def test_record_result_hold_activates_startup_readiness_gap_route(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["matched_fail_routes"]["startup_readiness_gap"] = "t_main_2"
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        self.mod.record_result(
            "t_main_1",
            "hold",
            repo_root=self.repo_root,
            observed_blocker_family="startup_readiness_gap",
        )

        queue_after = yaml.safe_load((self.repo_root / "phase5" / "queue.yaml").read_text(encoding="utf-8"))
        self.assertEqual(queue_after["serialized_mainline"][1]["status"], "ready")
        history = queue_after["serialized_mainline"][0]["history"][-1]
        self.assertTrue(history["child_activation_allowed"])
        self.assertEqual(history["activated_child"], "t_main_2")

    def test_record_result_hold_activates_lighter_sequence_continuity_gap_route(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["matched_fail_routes"]["lighter_sequence_continuity_gap"] = "t_main_2"
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        self.mod.record_result(
            "t_main_1",
            "hold",
            repo_root=self.repo_root,
            observed_blocker_family="lighter_sequence_continuity_gap",
        )

        queue_after = yaml.safe_load((self.repo_root / "phase5" / "queue.yaml").read_text(encoding="utf-8"))
        self.assertEqual(queue_after["serialized_mainline"][1]["status"], "ready")
        history = queue_after["serialized_mainline"][0]["history"][-1]
        self.assertTrue(history["child_activation_allowed"])
        self.assertEqual(history["activated_child"], "t_main_2")

    def test_record_result_hold_activates_extended_post_publish_fallback_rearm_gap_route(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["matched_fail_routes"]["extended_post_publish_fallback_rearm_gap"] = "t_main_2"
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        self.mod.record_result(
            "t_main_1",
            "hold",
            repo_root=self.repo_root,
            observed_blocker_family="extended_post_publish_fallback_rearm_gap",
        )

        queue_after = yaml.safe_load((self.repo_root / "phase5" / "queue.yaml").read_text(encoding="utf-8"))
        self.assertEqual(queue_after["serialized_mainline"][1]["status"], "ready")
        history = queue_after["serialized_mainline"][0]["history"][-1]
        self.assertTrue(history["child_activation_allowed"])
        self.assertEqual(history["activated_child"], "t_main_2")

    def test_record_result_hold_activates_extended_degraded_stream_rebootstrap_gap_route(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["matched_fail_routes"]["extended_degraded_stream_rebootstrap_gap"] = "t_main_2"
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        self.mod.record_result(
            "t_main_1",
            "hold",
            repo_root=self.repo_root,
            observed_blocker_family="extended_degraded_stream_rebootstrap_gap",
        )

        queue_after = yaml.safe_load((self.repo_root / "phase5" / "queue.yaml").read_text(encoding="utf-8"))
        self.assertEqual(queue_after["serialized_mainline"][1]["status"], "ready")
        history = queue_after["serialized_mainline"][0]["history"][-1]
        self.assertTrue(history["child_activation_allowed"])
        self.assertEqual(history["activated_child"], "t_main_2")

    def test_record_result_hold_activates_extended_pre_kill_degraded_rebootstrap_alignment_gap_route(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["matched_fail_routes"][
            "extended_pre_kill_degraded_rebootstrap_alignment_gap"
        ] = "t_main_2"
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        self.mod.record_result(
            "t_main_1",
            "hold",
            repo_root=self.repo_root,
            observed_blocker_family="extended_pre_kill_degraded_rebootstrap_alignment_gap",
        )

        queue_after = yaml.safe_load((self.repo_root / "phase5" / "queue.yaml").read_text(encoding="utf-8"))
        self.assertEqual(queue_after["serialized_mainline"][1]["status"], "ready")
        history = queue_after["serialized_mainline"][0]["history"][-1]
        self.assertTrue(history["child_activation_allowed"])
        self.assertEqual(history["activated_child"], "t_main_2")

    def test_record_result_hold_activates_all_venue_market_frontier_backpressure_gap_route(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["matched_fail_routes"]["all_venue_market_frontier_backpressure_gap"] = "t_main_2"
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        self.mod.record_result(
            "t_main_1",
            "hold",
            repo_root=self.repo_root,
            observed_blocker_family="all_venue_market_frontier_backpressure_gap",
        )

        queue_after = yaml.safe_load((self.repo_root / "phase5" / "queue.yaml").read_text(encoding="utf-8"))
        self.assertEqual(queue_after["serialized_mainline"][1]["status"], "ready")
        history = queue_after["serialized_mainline"][0]["history"][-1]
        self.assertTrue(history["child_activation_allowed"])
        self.assertEqual(history["activated_child"], "t_main_2")

    def test_record_result_hold_activates_all5_projected_mm_budget_distribution_gap_route(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["matched_fail_routes"][
            "all5_projected_mm_budget_distribution_gap"
        ] = "t_main_2"
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        self.mod.record_result(
            "t_main_1",
            "hold",
            repo_root=self.repo_root,
            observed_blocker_family="all5_projected_mm_budget_distribution_gap",
        )

        queue_after = yaml.safe_load((self.repo_root / "phase5" / "queue.yaml").read_text(encoding="utf-8"))
        self.assertEqual(queue_after["serialized_mainline"][1]["status"], "ready")
        history = queue_after["serialized_mainline"][0]["history"][-1]
        self.assertTrue(history["child_activation_allowed"])
        self.assertEqual(history["activated_child"], "t_main_2")

    def test_record_result_hold_activates_paradex_interactive_top_anchor_gap_route(self):
        current_overlay = self.repo_root / "current_surface_stage_overlay.env"
        current_overlay.write_text(
            "SURFACE=current\nPARAPHINA_PARADEX_PUBLIC_FEED=interactive\nPARAPHINA_PARADEX_UI_TOUCH_REFERENCE_ENABLED=0\n",
            encoding="utf-8",
        )
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["matched_fail_routes"]["paradex_interactive_top_anchor_gap"] = "t_main_2"
        queue["serialized_mainline"][1]["hypothesis_blocker_family"] = "paradex_interactive_top_anchor_gap"
        queue["serialized_mainline"][1]["env_diff"] = {"stage_overlay_source": str(current_overlay)}
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        self.mod.record_result(
            "t_main_1",
            "hold",
            repo_root=self.repo_root,
            observed_blocker_family="paradex_interactive_top_anchor_gap",
        )

        queue_after = yaml.safe_load((self.repo_root / "phase5" / "queue.yaml").read_text(encoding="utf-8"))
        self.assertEqual(queue_after["serialized_mainline"][1]["status"], "ready")
        history = queue_after["serialized_mainline"][0]["history"][-1]
        self.assertTrue(history["child_activation_allowed"])
        self.assertEqual(history["activated_child"], "t_main_2")

        card_path = self.mod.prepare_tranche(
            repo_root=self.repo_root,
            tranche_id="t_main_2",
            mark_in_progress=False,
        )
        card = yaml.safe_load(card_path.read_text(encoding="utf-8"))
        self.assertEqual(card["execution_defaults"]["stage_overlay_source"], str(current_overlay))

    def test_record_result_microstructure_underconversion_route_prepares_current_surface_child_overlay(self):
        current_overlay = self.repo_root / "current_surface_stage_overlay.env"
        current_overlay.write_text(
            "SURFACE=current\nPARAPHINA_STARTUP_STALE_ARMING_MS=4000\n",
            encoding="utf-8",
        )
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["matched_fail_routes"]["microstructure_underconversion"] = "t_main_2"
        queue["serialized_mainline"][1]["hypothesis_blocker_family"] = "paradex_queue_position_loss"
        queue["serialized_mainline"][1]["env_diff"] = {"stage_overlay_source": str(current_overlay)}
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        self.mod.record_result(
            "t_main_1",
            "hold",
            repo_root=self.repo_root,
            observed_blocker_family="microstructure_underconversion",
        )

        queue_after = yaml.safe_load((self.repo_root / "phase5" / "queue.yaml").read_text(encoding="utf-8"))
        self.assertEqual(queue_after["serialized_mainline"][1]["status"], "ready")
        history = queue_after["serialized_mainline"][0]["history"][-1]
        self.assertTrue(history["child_activation_allowed"])
        self.assertEqual(history["activated_child"], "t_main_2")

        card_path = self.mod.prepare_tranche(
            repo_root=self.repo_root,
            tranche_id="t_main_2",
            mark_in_progress=False,
        )
        card = yaml.safe_load(card_path.read_text(encoding="utf-8"))
        self.assertEqual(card["execution_defaults"]["stage_overlay_source"], str(current_overlay))

    def test_record_result_edge_floor_queue_loss_route_prepares_current_surface_child_overlay(self):
        current_overlay = self.repo_root / "current_surface_stage_overlay.env"
        current_overlay.write_text(
            "SURFACE=current\nPARAPHINA_PARADEX_PRIVATE_ORDER_TRUTH_ENABLED=1\n",
            encoding="utf-8",
        )
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["matched_fail_routes"]["paradex_edge_floor_queue_loss"] = "t_main_2"
        queue["serialized_mainline"][1]["hypothesis_blocker_family"] = "paradex_edge_floor_queue_loss"
        queue["serialized_mainline"][1]["env_diff"] = {"stage_overlay_source": str(current_overlay)}
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        self.mod.record_result(
            "t_main_1",
            "hold",
            repo_root=self.repo_root,
            observed_blocker_family="paradex_edge_floor_queue_loss",
        )

        queue_after = yaml.safe_load((self.repo_root / "phase5" / "queue.yaml").read_text(encoding="utf-8"))
        self.assertEqual(queue_after["serialized_mainline"][1]["status"], "ready")
        history = queue_after["serialized_mainline"][0]["history"][-1]
        self.assertTrue(history["child_activation_allowed"])
        self.assertEqual(history["activated_child"], "t_main_2")

        card_path = self.mod.prepare_tranche(
            repo_root=self.repo_root,
            tranche_id="t_main_2",
            mark_in_progress=False,
        )
        card = yaml.safe_load(card_path.read_text(encoding="utf-8"))
        self.assertEqual(card["execution_defaults"]["stage_overlay_source"], str(current_overlay))

    def test_record_result_edge_floor_shadow_mechanism_gate_route_prepares_current_surface_child_overlay(self):
        current_overlay = self.repo_root / "current_surface_stage_overlay.env"
        current_overlay.write_text(
            "SURFACE=current\nPARAPHINA_PARADEX_EDGE_MIN_BPS=0.20\n",
            encoding="utf-8",
        )
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["matched_fail_routes"][
            "paradex_edge_floor_shadow_mechanism_gate_gap"
        ] = "t_main_2"
        queue["serialized_mainline"][1]["hypothesis_blocker_family"] = (
            "paradex_edge_floor_shadow_mechanism_gate_gap"
        )
        queue["serialized_mainline"][1]["env_diff"] = {"stage_overlay_source": str(current_overlay)}
        queue["serialized_mainline"][1]["support_gate"] = "shadow_smoke_10m"
        queue["serialized_mainline"][1]["support_gate_require_mechanism"] = False
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        self.mod.record_result(
            "t_main_1",
            "hold",
            repo_root=self.repo_root,
            observed_blocker_family="paradex_edge_floor_shadow_mechanism_gate_gap",
        )

        queue_after = yaml.safe_load((self.repo_root / "phase5" / "queue.yaml").read_text(encoding="utf-8"))
        self.assertEqual(queue_after["serialized_mainline"][1]["status"], "ready")
        history = queue_after["serialized_mainline"][0]["history"][-1]
        self.assertTrue(history["child_activation_allowed"])
        self.assertEqual(history["activated_child"], "t_main_2")

        card_path = self.mod.prepare_tranche(
            repo_root=self.repo_root,
            tranche_id="t_main_2",
            mark_in_progress=False,
        )
        card = yaml.safe_load(card_path.read_text(encoding="utf-8"))
        self.assertEqual(card["execution_defaults"]["stage_overlay_source"], str(current_overlay))

    def test_record_result_same_side_persistence_route_prepares_current_surface_child_overlay(self):
        current_overlay = self.repo_root / "current_surface_stage_overlay.env"
        current_overlay.write_text(
            "SURFACE=current\nPARAPHINA_PARADEX_POST_CONTROL_SUPPRESSION_GRACE_MS=1500\n",
            encoding="utf-8",
        )
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["matched_fail_routes"]["paradex_same_side_persistence_gap"] = "t_main_2"
        queue["serialized_mainline"][1]["hypothesis_blocker_family"] = "paradex_same_side_persistence_gap"
        queue["serialized_mainline"][1]["env_diff"] = {"stage_overlay_source": str(current_overlay)}
        queue["serialized_mainline"][1]["support_gate"] = "shadow_smoke_10m"
        queue["serialized_mainline"][1]["support_gate_require_mechanism"] = False
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        self.mod.record_result(
            "t_main_1",
            "hold",
            repo_root=self.repo_root,
            observed_blocker_family="paradex_same_side_persistence_gap",
        )

        queue_after = yaml.safe_load((self.repo_root / "phase5" / "queue.yaml").read_text(encoding="utf-8"))
        self.assertEqual(queue_after["serialized_mainline"][1]["status"], "ready")
        history = queue_after["serialized_mainline"][0]["history"][-1]
        self.assertTrue(history["child_activation_allowed"])
        self.assertEqual(history["activated_child"], "t_main_2")

        card_path = self.mod.prepare_tranche(
            repo_root=self.repo_root,
            tranche_id="t_main_2",
            mark_in_progress=False,
        )
        card = yaml.safe_load(card_path.read_text(encoding="utf-8"))
        self.assertEqual(card["execution_defaults"]["stage_overlay_source"], str(current_overlay))

    def test_record_result_open_snapshot_replace_identity_route_prepares_current_surface_child_overlay(self):
        current_overlay = self.repo_root / "current_surface_stage_overlay.env"
        current_overlay.write_text(
            "SURFACE=current\nPARAPHINA_PARADEX_PUBLIC_FEED=interactive\n",
            encoding="utf-8",
        )
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["matched_fail_routes"][
            "paradex_open_snapshot_replace_identity_gap"
        ] = "t_main_2"
        queue["serialized_mainline"][1]["hypothesis_blocker_family"] = (
            "paradex_open_snapshot_replace_identity_gap"
        )
        queue["serialized_mainline"][1]["env_diff"] = {"stage_overlay_source": str(current_overlay)}
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        self.mod.record_result(
            "t_main_1",
            "hold",
            repo_root=self.repo_root,
            observed_blocker_family="paradex_open_snapshot_replace_identity_gap",
        )

        queue_after = yaml.safe_load((self.repo_root / "phase5" / "queue.yaml").read_text(encoding="utf-8"))
        self.assertEqual(queue_after["serialized_mainline"][1]["status"], "ready")
        history = queue_after["serialized_mainline"][0]["history"][-1]
        self.assertTrue(history["child_activation_allowed"])
        self.assertEqual(history["activated_child"], "t_main_2")

        card_path = self.mod.prepare_tranche(
            repo_root=self.repo_root,
            tranche_id="t_main_2",
            mark_in_progress=False,
        )
        card = yaml.safe_load(card_path.read_text(encoding="utf-8"))
        self.assertEqual(card["execution_defaults"]["stage_overlay_source"], str(current_overlay))

    def test_record_result_microstructure_underconversion_route_prepares_interactive_public_top_child_overlay(self):
        current_overlay = self.repo_root / "current_surface_stage_overlay.env"
        current_overlay.write_text(
            "SURFACE=current\nPARAPHINA_PARADEX_PUBLIC_FEED=interactive\n",
            encoding="utf-8",
        )
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["matched_fail_routes"]["microstructure_underconversion"] = "t_main_2"
        queue["serialized_mainline"][1]["hypothesis_blocker_family"] = "paradex_interactive_top_anchor_gap"
        queue["serialized_mainline"][1]["env_diff"] = {"stage_overlay_source": str(current_overlay)}
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        self.mod.record_result(
            "t_main_1",
            "hold",
            repo_root=self.repo_root,
            observed_blocker_family="microstructure_underconversion",
        )

        queue_after = yaml.safe_load((self.repo_root / "phase5" / "queue.yaml").read_text(encoding="utf-8"))
        self.assertEqual(queue_after["serialized_mainline"][1]["status"], "ready")
        history = queue_after["serialized_mainline"][0]["history"][-1]
        self.assertTrue(history["child_activation_allowed"])
        self.assertEqual(history["activated_child"], "t_main_2")

        card_path = self.mod.prepare_tranche(
            repo_root=self.repo_root,
            tranche_id="t_main_2",
            mark_in_progress=False,
        )
        card = yaml.safe_load(card_path.read_text(encoding="utf-8"))
        self.assertEqual(card["execution_defaults"]["stage_overlay_source"], str(current_overlay))

    def test_record_result_paradex_underfill_with_ui_book_truth_route_prepares_current_surface_child_overlay(self):
        current_overlay = self.repo_root / "current_surface_stage_overlay.env"
        current_overlay.write_text(
            "SURFACE=current\nPARAPHINA_PARADEX_PUBLIC_FEED=bbo\nPARAPHINA_PARADEX_UI_TOUCH_REFERENCE_ENABLED=1\n",
            encoding="utf-8",
        )
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["matched_fail_routes"]["paradex_underfill_with_ui_book_truth"] = "t_main_2"
        queue["serialized_mainline"][1]["hypothesis_blocker_family"] = "paradex_underfill_with_ui_book_truth"
        queue["serialized_mainline"][1]["env_diff"] = {"stage_overlay_source": str(current_overlay)}
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        self.mod.record_result(
            "t_main_1",
            "hold",
            repo_root=self.repo_root,
            observed_blocker_family="paradex_underfill_with_ui_book_truth",
        )

        queue_after = yaml.safe_load((self.repo_root / "phase5" / "queue.yaml").read_text(encoding="utf-8"))
        self.assertEqual(queue_after["serialized_mainline"][1]["status"], "ready")
        history = queue_after["serialized_mainline"][0]["history"][-1]
        self.assertTrue(history["child_activation_allowed"])
        self.assertEqual(history["activated_child"], "t_main_2")

        card_path = self.mod.prepare_tranche(
            repo_root=self.repo_root,
            tranche_id="t_main_2",
            mark_in_progress=False,
        )
        card = yaml.safe_load(card_path.read_text(encoding="utf-8"))
        self.assertEqual(card["execution_defaults"]["stage_overlay_source"], str(current_overlay))

    def test_record_result_paradex_ui_touch_reference_gap_route_prepares_current_surface_child_overlay(self):
        current_overlay = self.repo_root / "current_surface_stage_overlay.env"
        current_overlay.write_text(
            "SURFACE=current\nPARAPHINA_PARADEX_PUBLIC_FEED=bbo\nPARAPHINA_PARADEX_UI_TOUCH_REFERENCE_ENABLED=1\n",
            encoding="utf-8",
        )
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["matched_fail_routes"]["paradex_ui_touch_reference_gap"] = "t_main_2"
        queue["serialized_mainline"][1]["hypothesis_blocker_family"] = "paradex_ui_touch_reference_gap"
        queue["serialized_mainline"][1]["env_diff"] = {"stage_overlay_source": str(current_overlay)}
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        self.mod.record_result(
            "t_main_1",
            "hold",
            repo_root=self.repo_root,
            observed_blocker_family="paradex_ui_touch_reference_gap",
        )

        queue_after = yaml.safe_load((self.repo_root / "phase5" / "queue.yaml").read_text(encoding="utf-8"))
        self.assertEqual(queue_after["serialized_mainline"][1]["status"], "ready")
        history = queue_after["serialized_mainline"][0]["history"][-1]
        self.assertTrue(history["child_activation_allowed"])
        self.assertEqual(history["activated_child"], "t_main_2")

        card_path = self.mod.prepare_tranche(
            repo_root=self.repo_root,
            tranche_id="t_main_2",
            mark_in_progress=False,
        )
        card = yaml.safe_load(card_path.read_text(encoding="utf-8"))
        self.assertEqual(card["execution_defaults"]["stage_overlay_source"], str(current_overlay))

    def test_record_result_hold_activates_hyperliquid_post_publish_transport_gap_route(self):
        current_overlay = self.repo_root / "current_surface_stage_overlay.env"
        current_overlay.write_text(
            "SURFACE=current\nPARAPHINA_HL_STATE_STALE_MS_OVERRIDE=5000\n",
            encoding="utf-8",
        )
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["matched_fail_routes"]["hyperliquid_post_publish_transport_gap"] = "t_main_2"
        queue["serialized_mainline"][1]["hypothesis_blocker_family"] = "hyperliquid_post_publish_transport_gap"
        queue["serialized_mainline"][1]["env_diff"] = {"stage_overlay_source": str(current_overlay)}
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        self.mod.record_result(
            "t_main_1",
            "hold",
            repo_root=self.repo_root,
            observed_blocker_family="hyperliquid_post_publish_transport_gap",
        )

        queue_after = yaml.safe_load((self.repo_root / "phase5" / "queue.yaml").read_text(encoding="utf-8"))
        self.assertEqual(queue_after["serialized_mainline"][1]["status"], "ready")
        history = queue_after["serialized_mainline"][0]["history"][-1]
        self.assertTrue(history["child_activation_allowed"])
        self.assertEqual(history["activated_child"], "t_main_2")

        card_path = self.mod.prepare_tranche(
            repo_root=self.repo_root,
            tranche_id="t_main_2",
            mark_in_progress=False,
        )
        card = yaml.safe_load(card_path.read_text(encoding="utf-8"))
        self.assertEqual(card["execution_defaults"]["stage_overlay_source"], str(current_overlay))

    def test_record_result_restore_hygiene_prefers_surface_local_child_for_hyperliquid_post_publish_route(self):
        current_overlay = self.repo_root / "current_surface_stage_overlay.env"
        current_overlay.write_text(
            "SURFACE=current\nPARAPHINA_HL_STATE_STALE_MS_OVERRIDE=5000\n",
            encoding="utf-8",
        )
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["hypothesis_blocker_family"] = "hyperliquid_post_publish_transport_gap"
        queue["serialized_mainline"][0]["surface_local_restore_hygiene_child"] = "t_main_2"
        queue["serialized_mainline"][1]["hypothesis_blocker_family"] = "restore_hygiene"
        queue["serialized_mainline"][1]["env_diff"] = {"stage_overlay_source": str(current_overlay)}
        queue["serialized_mainline"][1]["execution"] = {"live_guard_args": ["--pre-audit-cleanup-on-exit"]}
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        self.mod.record_result(
            "t_main_1",
            "hold",
            repo_root=self.repo_root,
            observed_blocker_family="restore_hygiene",
        )

        queue_after = yaml.safe_load((self.repo_root / "phase5" / "queue.yaml").read_text(encoding="utf-8"))
        self.assertEqual(queue_after["serialized_mainline"][1]["status"], "ready")
        history = queue_after["serialized_mainline"][0]["history"][-1]
        self.assertTrue(history["child_activation_allowed"])
        self.assertEqual(history["activated_child"], "t_main_2")

        card_path = self.mod.prepare_tranche(
            repo_root=self.repo_root,
            tranche_id="t_main_2",
            mark_in_progress=False,
        )
        card = yaml.safe_load(card_path.read_text(encoding="utf-8"))
        self.assertEqual(card["execution_defaults"]["stage_overlay_source"], str(current_overlay))
        self.assertEqual(
            card["execution_defaults"]["live_guard_args"],
            ["--pre-restore-cleanup-on-exit", "--pre-audit-cleanup-on-exit"],
        )

    def test_record_result_hold_activates_hyperliquid_pre_kill_recovery_alignment_route(self):
        current_overlay = self.repo_root / "current_surface_stage_overlay.env"
        current_overlay.write_text(
            "SURFACE=current\nPARAPHINA_HL_STATE_STALE_MS_OVERRIDE=5000\nPARAPHINA_HL_STALE_MS=3500\nPARAPHINA_HL_REST_FALLBACK_STALE_MS=4000\nPARAPHINA_HL_REST_FALLBACK_POLL_MS=500\n",
            encoding="utf-8",
        )
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["matched_fail_routes"]["hyperliquid_pre_kill_recovery_alignment_gap"] = "t_main_2"
        queue["serialized_mainline"][1]["hypothesis_blocker_family"] = "hyperliquid_pre_kill_recovery_alignment_gap"
        queue["serialized_mainline"][1]["env_diff"] = {"stage_overlay_source": str(current_overlay)}
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        self.mod.record_result(
            "t_main_1",
            "hold",
            repo_root=self.repo_root,
            observed_blocker_family="hyperliquid_pre_kill_recovery_alignment_gap",
        )

        queue_after = yaml.safe_load((self.repo_root / "phase5" / "queue.yaml").read_text(encoding="utf-8"))
        self.assertEqual(queue_after["serialized_mainline"][1]["status"], "ready")
        history = queue_after["serialized_mainline"][0]["history"][-1]
        self.assertTrue(history["child_activation_allowed"])
        self.assertEqual(history["activated_child"], "t_main_2")

        card_path = self.mod.prepare_tranche(
            repo_root=self.repo_root,
            tranche_id="t_main_2",
            mark_in_progress=False,
        )
        card = yaml.safe_load(card_path.read_text(encoding="utf-8"))
        self.assertEqual(card["execution_defaults"]["stage_overlay_source"], str(current_overlay))

    def test_record_result_restore_hygiene_prefers_surface_local_child_for_hyperliquid_pre_kill_route(self):
        current_overlay = self.repo_root / "current_surface_stage_overlay.env"
        current_overlay.write_text(
            "SURFACE=current\nPARAPHINA_HL_STATE_STALE_MS_OVERRIDE=5000\nPARAPHINA_HL_STALE_MS=3500\nPARAPHINA_HL_REST_FALLBACK_STALE_MS=4000\nPARAPHINA_HL_REST_FALLBACK_POLL_MS=500\n",
            encoding="utf-8",
        )
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["hypothesis_blocker_family"] = "hyperliquid_pre_kill_recovery_alignment_gap"
        queue["serialized_mainline"][0]["surface_local_restore_hygiene_child"] = "t_main_2"
        queue["serialized_mainline"][1]["hypothesis_blocker_family"] = "restore_hygiene"
        queue["serialized_mainline"][1]["env_diff"] = {"stage_overlay_source": str(current_overlay)}
        queue["serialized_mainline"][1]["execution"] = {"live_guard_args": ["--pre-audit-cleanup-on-exit"]}
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        self.mod.record_result(
            "t_main_1",
            "hold",
            repo_root=self.repo_root,
            observed_blocker_family="restore_hygiene",
        )

        queue_after = yaml.safe_load((self.repo_root / "phase5" / "queue.yaml").read_text(encoding="utf-8"))
        self.assertEqual(queue_after["serialized_mainline"][1]["status"], "ready")
        history = queue_after["serialized_mainline"][0]["history"][-1]
        self.assertTrue(history["child_activation_allowed"])
        self.assertEqual(history["activated_child"], "t_main_2")

        card_path = self.mod.prepare_tranche(
            repo_root=self.repo_root,
            tranche_id="t_main_2",
            mark_in_progress=False,
        )
        card = yaml.safe_load(card_path.read_text(encoding="utf-8"))
        self.assertEqual(card["execution_defaults"]["stage_overlay_source"], str(current_overlay))
        self.assertEqual(
            card["execution_defaults"]["live_guard_args"],
            ["--pre-restore-cleanup-on-exit", "--pre-audit-cleanup-on-exit"],
        )

    def test_record_result_restore_hygiene_prefers_surface_local_child_for_interactive_public_top_route(self):
        current_overlay = self.repo_root / "current_surface_stage_overlay.env"
        current_overlay.write_text(
            "SURFACE=current\nPARAPHINA_PARADEX_PUBLIC_FEED=interactive\nPARAPHINA_PARADEX_UI_TOUCH_REFERENCE_ENABLED=0\n",
            encoding="utf-8",
        )
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["hypothesis_blocker_family"] = "paradex_interactive_top_anchor_gap"
        queue["serialized_mainline"][0]["surface_local_restore_hygiene_child"] = "t_main_2"
        queue["serialized_mainline"][1]["hypothesis_blocker_family"] = "restore_hygiene"
        queue["serialized_mainline"][1]["env_diff"] = {"stage_overlay_source": str(current_overlay)}
        queue["serialized_mainline"][1]["execution"] = {"live_guard_args": ["--pre-audit-cleanup-on-exit"]}
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        self.mod.record_result(
            "t_main_1",
            "hold",
            repo_root=self.repo_root,
            observed_blocker_family="restore_hygiene",
        )

        queue_after = yaml.safe_load((self.repo_root / "phase5" / "queue.yaml").read_text(encoding="utf-8"))
        self.assertEqual(queue_after["serialized_mainline"][1]["status"], "ready")
        history = queue_after["serialized_mainline"][0]["history"][-1]
        self.assertTrue(history["child_activation_allowed"])
        self.assertEqual(history["activated_child"], "t_main_2")

        card_path = self.mod.prepare_tranche(
            repo_root=self.repo_root,
            tranche_id="t_main_2",
            mark_in_progress=False,
        )
        card = yaml.safe_load(card_path.read_text(encoding="utf-8"))
        self.assertEqual(card["execution_defaults"]["stage_overlay_source"], str(current_overlay))
        self.assertEqual(
            card["execution_defaults"]["live_guard_args"],
            ["--pre-restore-cleanup-on-exit", "--pre-audit-cleanup-on-exit"],
        )

    def test_record_result_restore_hygiene_prefers_surface_local_child_for_ui_book_truth_route(self):
        current_overlay = self.repo_root / "current_surface_stage_overlay.env"
        current_overlay.write_text(
            "SURFACE=current\nPARAPHINA_PARADEX_PUBLIC_FEED=bbo\nPARAPHINA_PARADEX_UI_BOOK_TRUTH_ENABLED=1\nPARAPHINA_PARADEX_UI_TOUCH_REFERENCE_ENABLED=1\n",
            encoding="utf-8",
        )
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["hypothesis_blocker_family"] = "paradex_underfill_with_ui_book_truth"
        queue["serialized_mainline"][0]["surface_local_restore_hygiene_child"] = "t_main_2"
        queue["serialized_mainline"][1]["hypothesis_blocker_family"] = "restore_hygiene"
        queue["serialized_mainline"][1]["env_diff"] = {"stage_overlay_source": str(current_overlay)}
        queue["serialized_mainline"][1]["execution"] = {"live_guard_args": ["--pre-audit-cleanup-on-exit"]}
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        self.mod.record_result(
            "t_main_1",
            "hold",
            repo_root=self.repo_root,
            observed_blocker_family="restore_hygiene",
        )

        queue_after = yaml.safe_load((self.repo_root / "phase5" / "queue.yaml").read_text(encoding="utf-8"))
        self.assertEqual(queue_after["serialized_mainline"][1]["status"], "ready")
        history = queue_after["serialized_mainline"][0]["history"][-1]
        self.assertTrue(history["child_activation_allowed"])
        self.assertEqual(history["activated_child"], "t_main_2")

        card_path = self.mod.prepare_tranche(
            repo_root=self.repo_root,
            tranche_id="t_main_2",
            mark_in_progress=False,
        )
        card = yaml.safe_load(card_path.read_text(encoding="utf-8"))
        self.assertEqual(card["execution_defaults"]["stage_overlay_source"], str(current_overlay))
        self.assertEqual(
            card["execution_defaults"]["live_guard_args"],
            ["--pre-restore-cleanup-on-exit", "--pre-audit-cleanup-on-exit"],
        )

    def test_record_result_restore_hygiene_route_prepares_current_surface_child_overlay(self):
        old_overlay = self.repo_root / "old_stage_overlay.env"
        old_overlay.write_text("SURFACE=old\n", encoding="utf-8")
        current_overlay = self.repo_root / "current_surface_stage_overlay.env"
        current_overlay.write_text("SURFACE=current\nPARAPHINA_EXTENDED_FREEZE_PROGRESS_AGE_ENABLED=1\n", encoding="utf-8")
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["matched_fail_routes"]["restore_hygiene"] = "t_main_2"
        queue["serialized_mainline"][1]["env_diff"] = {"stage_overlay_source": str(current_overlay)}
        queue["serialized_mainline"][1]["execution"] = {"live_guard_args": ["--pre-audit-cleanup-on-exit"]}
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        self.mod.record_result(
            "t_main_1",
            "hold",
            repo_root=self.repo_root,
            observed_blocker_family="restore_hygiene",
        )

        queue_after = yaml.safe_load((self.repo_root / "phase5" / "queue.yaml").read_text(encoding="utf-8"))
        self.assertEqual(queue_after["serialized_mainline"][1]["status"], "ready")

        card_path = self.mod.prepare_tranche(
            repo_root=self.repo_root,
            tranche_id="t_main_2",
            mark_in_progress=False,
        )
        card = yaml.safe_load(card_path.read_text(encoding="utf-8"))
        self.assertEqual(card["execution_defaults"]["stage_overlay_source"], str(current_overlay))
        self.assertEqual(
            card["execution_defaults"]["live_guard_args"],
            ["--pre-restore-cleanup-on-exit", "--pre-audit-cleanup-on-exit"],
        )

    def test_record_result_restore_hygiene_prefers_surface_local_child_overlay(self):
        current_overlay = self.repo_root / "current_surface_stage_overlay.env"
        current_overlay.write_text("SURFACE=current\nFEATURE=batch_cancel\n", encoding="utf-8")
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["surface_local_restore_hygiene_child"] = "t_main_2"
        queue["serialized_mainline"][1]["env_diff"] = {"stage_overlay_source": str(current_overlay)}
        queue["serialized_mainline"][1]["execution"] = {"live_guard_args": ["--pre-audit-cleanup-on-exit"]}
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        self.mod.record_result(
            "t_main_1",
            "hold",
            repo_root=self.repo_root,
            observed_blocker_family="restore_hygiene",
        )

        queue_after = yaml.safe_load((self.repo_root / "phase5" / "queue.yaml").read_text(encoding="utf-8"))
        self.assertEqual(queue_after["serialized_mainline"][1]["status"], "ready")
        history = queue_after["serialized_mainline"][0]["history"][-1]
        self.assertEqual(history["activated_child"], "t_main_2")

        card_path = self.mod.prepare_tranche(
            repo_root=self.repo_root,
            tranche_id="t_main_2",
            mark_in_progress=False,
        )
        card = yaml.safe_load(card_path.read_text(encoding="utf-8"))
        self.assertEqual(card["execution_defaults"]["stage_overlay_source"], str(current_overlay))
        self.assertEqual(
            card["execution_defaults"]["live_guard_args"],
            ["--pre-restore-cleanup-on-exit", "--pre-audit-cleanup-on-exit"],
        )

    def test_record_result_restore_hygiene_prefers_surface_local_child_for_fv_reentry_route(self):
        current_overlay = self.repo_root / "current_surface_stage_overlay.env"
        current_overlay.write_text(
            "SURFACE=current\nPARAPHINA_MM_VENUE_ROLE_HYPERLIQUID=fill\nPARAPHINA_MM_VENUE_ROLE_LIGHTER=fill\nPARAPHINA_FV_DISABLED_VENUES=\n",
            encoding="utf-8",
        )
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["hypothesis_blocker_family"] = "topology_fv_reentry_gap"
        queue["serialized_mainline"][0]["surface_local_restore_hygiene_child"] = "t_main_2"
        queue["serialized_mainline"][1]["hypothesis_blocker_family"] = "restore_hygiene"
        queue["serialized_mainline"][1]["env_diff"] = {"stage_overlay_source": str(current_overlay)}
        queue["serialized_mainline"][1]["execution"] = {"live_guard_args": ["--pre-audit-cleanup-on-exit"]}
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        self.mod.record_result(
            "t_main_1",
            "hold",
            repo_root=self.repo_root,
            observed_blocker_family="restore_hygiene",
        )

        queue_after = yaml.safe_load((self.repo_root / "phase5" / "queue.yaml").read_text(encoding="utf-8"))
        self.assertEqual(queue_after["serialized_mainline"][1]["status"], "ready")
        history = queue_after["serialized_mainline"][0]["history"][-1]
        self.assertTrue(history["child_activation_allowed"])
        self.assertEqual(history["activated_child"], "t_main_2")

        card_path = self.mod.prepare_tranche(
            repo_root=self.repo_root,
            tranche_id="t_main_2",
            mark_in_progress=False,
        )
        card = yaml.safe_load(card_path.read_text(encoding="utf-8"))
        self.assertEqual(card["execution_defaults"]["stage_overlay_source"], str(current_overlay))
        self.assertEqual(
            card["execution_defaults"]["live_guard_args"],
            ["--pre-restore-cleanup-on-exit", "--pre-audit-cleanup-on-exit"],
        )

    def test_orchestrate_promotes_with_explicit_promotion_rule(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["automation"] = {
            "support_tracks": ["t_support_1"],
            "rung_plan": [{"duration_sec": 300, "continue_on": "promotion"}],
            "autoscore": {
                "promotion": [
                    {
                        "source": "closeout",
                        "path": "summary_exists",
                        "op": "==",
                        "value": True,
                        "severity": "fail",
                    }
                ]
            },
        }
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)
        expected_run_root = self.repo_root / "promotion_runs" / "t_main_1_5m_canary_20260401T000000Z" / "live_canary"

        def fake_run_live_guarded(tranche_id, duration_sec, repo_root):
            run_root = repo_root / "promotion_runs" / f"{tranche_id}_5m_canary_20260401T000000Z" / "live_canary"
            run_root.mkdir(parents=True, exist_ok=True)
            (run_root / "live_closeout_bundle.json").write_text(
                json.dumps(
                    {
                        "summary_exists": True,
                        "report_exists": True,
                        "guard_intervened": False,
                        "guard_window_completed": True,
                    "pre_restore_venue_audit_clean": True,
                    "post_rollback_venue_audit_clean": True,
                    "healthy_post": True,
                    "ready_post": True,
                    "kill_events_present_post": False,
                    "trade_mode_post": "shadow",
                    "systemd_active_state_post": "active",
                    "systemd_sub_state_post": "running",
                    "systemd_nrestarts_post": "0",
                    "guard_exit_code": 0,
                }
            ) + "\n",
            encoding="utf-8",
        )
            (run_root / "live_segment_summary.json").write_text(
                json.dumps({"tick_count": 5, "pnl_validity": {"final_pnl_total": 0.0}}) + "\n",
                encoding="utf-8",
            )
            (run_root / "live_metrics.json").write_text(
                json.dumps({"fills": {"total_count": 0, "total_base": 0.0}}) + "\n",
                encoding="utf-8",
            )
            (run_root / "guard.log").write_text("guard complete\n", encoding="utf-8")
            return run_root

        with mock.patch.object(
            self.mod,
            "ensure_shadow_health",
            return_value={"healthy": True, "ready": True, "trade_mode": "shadow"},
        ), mock.patch.object(
            self.mod,
            "ensure_runtime_storage_headroom",
            return_value={
                "repo": {"free_bytes": 10},
                "promotion_runs": {"free_bytes": 10},
                "telemetry": {"free_bytes": 10},
                "tempdir": {"free_bytes": 10},
                "current_runs": {"free_bytes": 10},
            },
        ), mock.patch.object(
            self.mod,
            "audit_state_sync",
            return_value={
                "schema_version": 1,
                "generated_utc": "2026-04-26T00:00:00Z",
                "repo_root": str(self.repo_root),
                "requested_tranche_id": "t_main_1",
                "status": "pass",
                "critical_count": 0,
                "warning_count": 0,
                "tranches": [
                    {
                        "tranche_id": "t_main_1",
                        "surface_id": "surface-1",
                        "linked_child_ids": ["t_main_2"],
                        "status_summary": "pass",
                    }
                ],
                "findings": [],
            },
        ):
            with mock.patch.object(self.mod, "run_live_guarded", side_effect=fake_run_live_guarded):
                session = self.mod.orchestrate_tranche("t_main_1", repo_root=self.repo_root)

        self.assertEqual(session["final_decision"], "promote")
        self.assertEqual(session["selected_child"], "t_main_2")
        queue_after = yaml.safe_load((self.repo_root / "phase5" / "queue.yaml").read_text(encoding="utf-8"))
        self.assertEqual(queue_after["serialized_mainline"][0]["status"], "promoted")
        self.assertEqual(queue_after["serialized_mainline"][1]["status"], "ready")
        support_result = json.loads(
            (self.repo_root / "phase5" / "runs" / "t_main_1" / "lanes" / f"{self.mod.LANE_ROLE_SUPPORT_PREFIX}t_support_1" / "result_bundle.json").read_text(encoding="utf-8")
        )
        self.assertEqual(support_result["status"], "pass")
        stage_verdict = json.loads(
            (self.repo_root / "phase5" / "runs" / "t_main_1" / "stage_verdict.json").read_text(encoding="utf-8")
        )
        venue_matrix = json.loads(
            (self.repo_root / "phase5" / "runs" / "t_main_1" / "venue_capability_matrix.json").read_text(encoding="utf-8")
        )
        support_summary = json.loads(
            (self.repo_root / "phase5" / "runs" / "t_main_1" / "support_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(stage_verdict["verdict"], "PROMOTE")
        self.assertEqual(stage_verdict["selected_child"], "t_main_2")
        self.assertEqual(stage_verdict["state_sync"]["status"], "pass")
        self.assertIn("hyperliquid", venue_matrix["venues"])
        self.assertEqual(support_summary["state_sync"]["status"], "pass")
        self.assertEqual(support_summary["total_runs"], 1)
        self.assertTrue((expected_run_root / "state_sync_report.json").exists())

    def test_orchestrate_promote_downgraded_by_state_sync_keeps_session_and_final_contract_hold(self):
        queue = _minimal_queue()
        queue["serialized_mainline"][0]["automation"] = {
            "support_tracks": ["t_support_1"],
            "rung_plan": [{"duration_sec": 300, "continue_on": "promotion"}],
            "autoscore": {
                "promotion": [
                    {
                        "source": "closeout",
                        "path": "summary_exists",
                        "op": "==",
                        "value": True,
                        "severity": "fail",
                    }
                ]
            },
        }
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        def fake_run_live_guarded(tranche_id, duration_sec, repo_root):
            run_root = repo_root / "promotion_runs" / f"{tranche_id}_5m_canary_20260401T000000Z" / "live_canary"
            run_root.mkdir(parents=True, exist_ok=True)
            (run_root / "live_closeout_bundle.json").write_text(
                json.dumps(
                    {
                        "summary_exists": True,
                        "report_exists": True,
                        "guard_intervened": False,
                        "guard_window_completed": True,
                        "pre_restore_venue_audit_clean": True,
                        "post_rollback_venue_audit_clean": True,
                        "healthy_post": True,
                        "ready_post": True,
                        "kill_events_present_post": False,
                        "trade_mode_post": "shadow",
                        "systemd_active_state_post": "active",
                        "systemd_sub_state_post": "running",
                        "systemd_nrestarts_post": "0",
                        "guard_exit_code": 0,
                    }
                ) + "\n",
                encoding="utf-8",
            )
            (run_root / "live_segment_summary.json").write_text(
                json.dumps({"tick_count": 5, "pnl_validity": {"final_pnl_total": 0.0}}) + "\n",
                encoding="utf-8",
            )
            (run_root / "live_metrics.json").write_text(
                json.dumps({"fills": {"total_count": 0, "total_base": 0.0}}) + "\n",
                encoding="utf-8",
            )
            (run_root / "guard.log").write_text("guard complete\n", encoding="utf-8")
            return run_root

        with mock.patch.object(
            self.mod,
            "ensure_shadow_health",
            return_value={"healthy": True, "ready": True, "trade_mode": "shadow"},
        ), mock.patch.object(
            self.mod,
            "ensure_runtime_storage_headroom",
            return_value={
                "repo": {"free_bytes": 10},
                "promotion_runs": {"free_bytes": 10},
                "telemetry": {"free_bytes": 10},
                "tempdir": {"free_bytes": 10},
                "current_runs": {"free_bytes": 10},
            },
        ), mock.patch.object(
            self.mod,
            "audit_state_sync",
            side_effect=[self._state_sync_pass_report(), self._state_sync_warning_report()],
        ):
            with mock.patch.object(self.mod, "run_live_guarded", side_effect=fake_run_live_guarded):
                session = self.mod.orchestrate_tranche("t_main_1", repo_root=self.repo_root)

        self.assertEqual(session["final_decision"], "hold")
        self.assertIsNone(session["selected_child"])
        queue_after = yaml.safe_load((self.repo_root / "phase5" / "queue.yaml").read_text(encoding="utf-8"))
        self.assertEqual(queue_after["serialized_mainline"][0]["status"], "hold")
        self.assertEqual(queue_after["serialized_mainline"][1]["status"], "blocked")
        history = queue_after["serialized_mainline"][0]["history"][-1]
        self.assertEqual(history["requested_decision"], "promote")
        self.assertEqual(history["decision"], "hold")
        self.assertTrue(history["state_sync_blocked_promotion"])
        stage_verdict = json.loads(
            (self.repo_root / "phase5" / "runs" / "t_main_1" / "stage_verdict.json").read_text(encoding="utf-8")
        )
        self.assertEqual(stage_verdict["verdict"], "HOLD")
        self.assertEqual(stage_verdict["decision"], "hold")
        self.assertEqual(stage_verdict["state_sync"]["warning_count"], 1)

    def test_orchestrate_starts_new_session_after_archived_session(self):
        archived = self._spawn_lanes_with_preflight("t_main_1")
        archived["session_id"] = "t_main_1-archived"
        archived["state"] = "archived"
        archived["updated_utc"] = "2026-04-03T00:00:00Z"
        archived["cleanup_completed"] = True
        archived["cleanup_completed_utc"] = "2026-04-03T00:00:00Z"
        orchestration = self.mod.load_orchestration(self.repo_root)
        self.mod.upsert_orchestration_session(orchestration, archived)
        self.mod.save_orchestration(orchestration, self.repo_root)

        def fake_run_live_guarded(tranche_id, duration_sec, repo_root):
            run_root = repo_root / "promotion_runs" / f"{tranche_id}_5m_canary_20260401T000000Z" / "live_canary"
            run_root.mkdir(parents=True, exist_ok=True)
            (run_root / "live_closeout_bundle.json").write_text(
                json.dumps(
                    {
                        "summary_exists": True,
                        "report_exists": True,
                        "guard_intervened": False,
                        "guard_window_completed": True,
                        "pre_restore_venue_audit_clean": True,
                        "post_rollback_venue_audit_clean": True,
                        "healthy_post": True,
                        "ready_post": True,
                        "kill_events_present_post": False,
                        "trade_mode_post": "shadow",
                        "systemd_nrestarts_post": "0",
                    }
                ) + "\n",
                encoding="utf-8",
            )
            (run_root / "live_segment_summary.json").write_text(
                json.dumps({"tick_count": 5, "pnl_validity": {"final_pnl_total": 0.0}}) + "\n",
                encoding="utf-8",
            )
            (run_root / "live_metrics.json").write_text(
                json.dumps({"fills": {"total_count": 0, "total_base": 0.0}}) + "\n",
                encoding="utf-8",
            )
            (run_root / "guard.log").write_text("guard complete\n", encoding="utf-8")
            return run_root

        with mock.patch.object(
            self.mod,
            "ensure_shadow_health",
            return_value={"healthy": True, "ready": True, "trade_mode": "shadow"},
        ), mock.patch.object(
            self.mod,
            "ensure_runtime_storage_headroom",
            return_value={
                "repo": {"free_bytes": 10},
                "promotion_runs": {"free_bytes": 10},
                "telemetry": {"free_bytes": 10},
                "tempdir": {"free_bytes": 10},
                "current_runs": {"free_bytes": 10},
            },
        ), mock.patch.object(
            self.mod,
            "audit_state_sync",
            return_value=self._state_sync_pass_report(),
        ):
            with mock.patch.object(self.mod, "run_live_guarded", side_effect=fake_run_live_guarded):
                session = self.mod.orchestrate_tranche("t_main_1", repo_root=self.repo_root)

        self.assertNotEqual(session["session_id"], archived["session_id"])
        orchestration = self.mod.load_orchestration(self.repo_root)
        sessions = [item for item in orchestration["sessions"] if item["tranche_id"] == "t_main_1"]
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["session_id"], session["session_id"])
        self.assertEqual(sessions[0]["state"], "completed")

    def test_orchestrate_starts_new_session_after_completed_session(self):
        completed = self._spawn_lanes_with_preflight("t_main_1")
        completed["session_id"] = "t_main_1-completed"
        completed["state"] = "completed"
        completed["updated_utc"] = "2026-04-03T00:00:00Z"
        completed["cleanup_completed"] = True
        completed["cleanup_completed_utc"] = "2026-04-03T00:00:00Z"
        orchestration = self.mod.load_orchestration(self.repo_root)
        self.mod.upsert_orchestration_session(orchestration, completed)
        self.mod.save_orchestration(orchestration, self.repo_root)

        def fake_run_live_guarded(tranche_id, duration_sec, repo_root):
            run_root = repo_root / "promotion_runs" / f"{tranche_id}_5m_canary_20260401T000000Z" / "live_canary"
            run_root.mkdir(parents=True, exist_ok=True)
            (run_root / "live_closeout_bundle.json").write_text(
                json.dumps(
                    {
                        "summary_exists": True,
                        "report_exists": True,
                        "guard_intervened": False,
                        "guard_window_completed": True,
                        "pre_restore_venue_audit_clean": True,
                        "post_rollback_venue_audit_clean": True,
                        "healthy_post": True,
                        "ready_post": True,
                        "kill_events_present_post": False,
                        "trade_mode_post": "shadow",
                        "systemd_nrestarts_post": "0",
                    }
                ) + "\n",
                encoding="utf-8",
            )
            (run_root / "live_segment_summary.json").write_text(
                json.dumps({"tick_count": 5, "pnl_validity": {"final_pnl_total": 0.0}}) + "\n",
                encoding="utf-8",
            )
            (run_root / "live_metrics.json").write_text(
                json.dumps({"fills": {"total_count": 0, "total_base": 0.0}}) + "\n",
                encoding="utf-8",
            )
            (run_root / "guard.log").write_text("guard complete\n", encoding="utf-8")
            return run_root

        with mock.patch.object(
            self.mod,
            "ensure_shadow_health",
            return_value={"healthy": True, "ready": True, "trade_mode": "shadow"},
        ), mock.patch.object(
            self.mod,
            "ensure_runtime_storage_headroom",
            return_value={
                "repo": {"free_bytes": 10},
                "promotion_runs": {"free_bytes": 10},
                "telemetry": {"free_bytes": 10},
                "tempdir": {"free_bytes": 10},
                "current_runs": {"free_bytes": 10},
            },
        ), mock.patch.object(
            self.mod,
            "audit_state_sync",
            return_value=self._state_sync_pass_report(),
        ):
            with mock.patch.object(self.mod, "run_live_guarded", side_effect=fake_run_live_guarded):
                session = self.mod.orchestrate_tranche("t_main_1", repo_root=self.repo_root)

        self.assertNotEqual(session["session_id"], completed["session_id"])
        orchestration = self.mod.load_orchestration(self.repo_root)
        sessions = [item for item in orchestration["sessions"] if item["tranche_id"] == "t_main_1"]
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["session_id"], session["session_id"])
        self.assertEqual(sessions[0]["state"], "completed")

    def test_orchestrate_reduces_recovered_latest_run_without_rerun(self):
        session = self._spawn_lanes_with_preflight("t_main_1")
        orchestration = self.mod.load_orchestration(self.repo_root)
        self.mod.upsert_orchestration_session(orchestration, session)
        self.mod.save_orchestration(orchestration, self.repo_root)

        run_root = self.repo_root / "promotion_runs" / "t_main_1_5m_canary_20260401T000000Z" / "live_canary"
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "live_closeout_bundle.json").write_text(
            json.dumps(
                {
                    "summary_exists": True,
                    "report_exists": True,
                    "guard_intervened": True,
                    "guard_window_completed": False,
                    "pre_restore_venue_audit_clean": False,
                    "post_rollback_venue_audit_clean": True,
                    "healthy_post": True,
                    "ready_post": True,
                    "kill_events_present_post": False,
                    "trade_mode_post": "shadow",
                    "systemd_nrestarts_post": "0",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (run_root / "live_segment_summary.json").write_text(
            json.dumps({"tick_count": 5, "pnl_validity": {"final_pnl_total": 0.0}}) + "\n",
            encoding="utf-8",
        )
        (run_root / "live_metrics.json").write_text(
            json.dumps({"fills": {"total_count": 0, "total_base": 0.0}}) + "\n",
            encoding="utf-8",
        )
        (run_root / "guard.log").write_text(
            "2026-04-01T00:05:00Z CRITICAL triggered_intervention reason=kill_events_present\n",
            encoding="utf-8",
        )
        (run_root / "autoscore_bundle.json").write_text(
            json.dumps(
                {
                    "clean": {"passed": False, "failed_rules": []},
                    "mechanism": {"passed": True, "failed_rules": []},
                    "promotion": {"passed": True, "failed_rules": []},
                    "is_final_rung": False,
                    "suggested_action": "hold",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        _write_yaml(
            self.repo_root / "phase5" / "runs" / "t_main_1" / "latest_run.yaml",
            {
                "run_root": str(run_root),
                "duration_sec": 300,
                "run_state": "window_complete",
            },
        )

        with mock.patch.object(
            self.mod,
            "ensure_shadow_health",
            return_value={"healthy": True, "ready": True, "trade_mode": "shadow"},
        ), mock.patch.object(
            self.mod,
            "ensure_runtime_storage_headroom",
            return_value={
                "repo": {"free_bytes": 10},
                "promotion_runs": {"free_bytes": 10},
                "telemetry": {"free_bytes": 10},
                "tempdir": {"free_bytes": 10},
                "current_runs": {"free_bytes": 10},
            },
        ), mock.patch.object(
            self.mod,
            "run_live_guarded",
            side_effect=AssertionError("run_live_guarded should not be called for recovered latest run"),
        ):
            updated = self.mod.orchestrate_tranche("t_main_1", repo_root=self.repo_root)

        self.assertEqual(updated["final_decision"], "hold")
        self.assertEqual(len(updated["rung_results"]), 1)
        self.assertTrue(updated["rung_results"][0]["recovered"])
        self.assertEqual(updated["state"], "completed")

    def test_orchestrate_ignores_stale_latest_run_manifest_for_new_session(self):
        completed = self._spawn_lanes_with_preflight("t_main_1")
        completed["session_id"] = "t_main_1-completed"
        completed["state"] = "completed"
        completed["updated_utc"] = "2026-04-03T00:00:00Z"
        completed["cleanup_completed"] = True
        completed["cleanup_completed_utc"] = "2026-04-03T00:00:00Z"
        orchestration = self.mod.load_orchestration(self.repo_root)
        self.mod.upsert_orchestration_session(orchestration, completed)
        self.mod.save_orchestration(orchestration, self.repo_root)

        stale_run_root = self.repo_root / "promotion_runs" / "t_main_1_5m_canary_20260401T000000Z" / "live_canary"
        stale_run_root.mkdir(parents=True, exist_ok=True)
        (stale_run_root / "live_closeout_bundle.json").write_text(
            json.dumps(
                {
                    "summary_exists": True,
                    "report_exists": True,
                    "guard_intervened": True,
                    "guard_window_completed": False,
                    "pre_restore_venue_audit_clean": False,
                    "post_rollback_venue_audit_clean": True,
                    "healthy_post": True,
                    "ready_post": True,
                    "kill_events_present_post": False,
                    "trade_mode_post": "shadow",
                    "systemd_nrestarts_post": "0",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (stale_run_root / "live_segment_summary.json").write_text(
            json.dumps({"tick_count": 5, "pnl_validity": {"final_pnl_total": 0.0}}) + "\n",
            encoding="utf-8",
        )
        (stale_run_root / "live_metrics.json").write_text(
            json.dumps({"fills": {"total_count": 0, "total_base": 0.0}}) + "\n",
            encoding="utf-8",
        )
        (stale_run_root / "guard.log").write_text("stale guard\n", encoding="utf-8")
        (stale_run_root / "autoscore_bundle.json").write_text(
            json.dumps(
                {
                    "clean": {"passed": False, "failed_rules": []},
                    "mechanism": {"passed": True, "failed_rules": []},
                    "promotion": {"passed": True, "failed_rules": []},
                    "is_final_rung": False,
                    "suggested_action": "hold",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        _write_yaml(
            self.repo_root / "phase5" / "runs" / "t_main_1" / "latest_run.yaml",
            {
                "updated_utc": "2026-04-01T00:05:00Z",
                "run_root": str(stale_run_root),
                "duration_sec": 300,
                "run_state": "window_complete",
            },
        )

        fresh_run_root = self.repo_root / "promotion_runs" / "t_main_1_5m_canary_20260403T000000Z" / "live_canary"

        def fake_run_live_guarded(tranche_id, duration_sec, repo_root):
            self.assertEqual(tranche_id, "t_main_1")
            self.assertEqual(duration_sec, 300)
            fresh_run_root.mkdir(parents=True, exist_ok=True)
            (fresh_run_root / "live_closeout_bundle.json").write_text(
                json.dumps(
                    {
                        "summary_exists": True,
                        "report_exists": True,
                        "guard_intervened": False,
                        "guard_window_completed": True,
                        "pre_restore_venue_audit_clean": True,
                        "post_rollback_venue_audit_clean": True,
                        "healthy_post": True,
                        "ready_post": True,
                        "kill_events_present_post": False,
                        "trade_mode_post": "shadow",
                        "systemd_nrestarts_post": "0",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (fresh_run_root / "live_segment_summary.json").write_text(
                json.dumps({"tick_count": 5, "pnl_validity": {"final_pnl_total": 0.0}}) + "\n",
                encoding="utf-8",
            )
            (fresh_run_root / "live_metrics.json").write_text(
                json.dumps({"fills": {"total_count": 0, "total_base": 0.0}}) + "\n",
                encoding="utf-8",
            )
            (fresh_run_root / "guard.log").write_text("fresh guard\n", encoding="utf-8")
            return fresh_run_root

        with mock.patch.object(
            self.mod,
            "ensure_shadow_health",
            return_value={"healthy": True, "ready": True, "trade_mode": "shadow"},
        ), mock.patch.object(
            self.mod,
            "ensure_runtime_storage_headroom",
            return_value={
                "repo": {"free_bytes": 10},
                "promotion_runs": {"free_bytes": 10},
                "telemetry": {"free_bytes": 10},
                "tempdir": {"free_bytes": 10},
                "current_runs": {"free_bytes": 10},
            },
        ), mock.patch.object(
            self.mod,
            "audit_state_sync",
            return_value=self._state_sync_pass_report(),
        ), mock.patch.object(
            self.mod,
            "run_live_guarded",
            side_effect=fake_run_live_guarded,
        ):
            updated = self.mod.orchestrate_tranche("t_main_1", repo_root=self.repo_root)

        self.assertEqual(updated["rung_results"][0]["run_root"], str(fresh_run_root))
        self.assertFalse(updated["rung_results"][0].get("recovered", False))
        self.assertEqual(updated["state"], "completed")

    def test_admission_check_rejects_reopened_required_blocker_on_same_surface(self):
        runtime_binary = self.repo_root / "bin" / "paraphina_live"
        runtime_binary.parent.mkdir(parents=True, exist_ok=True)
        runtime_binary.write_text("binary", encoding="utf-8")
        overlay = self.repo_root / "stage_overlay_live.env"
        overlay.write_text("PARAPHINA_TEST=1\n", encoding="utf-8")

        control_pack = _minimal_control_pack()
        control_pack["execution_defaults"]["runtime_binary"] = str(runtime_binary)
        control_pack["execution_defaults"]["stage_overlay_source"] = str(overlay)
        _write_yaml(self.repo_root / "phase5" / "control_pack.yaml", control_pack)

        queue = _minimal_queue()
        queue["serialized_mainline"][0]["required_cleared_blockers"] = ["stale_restart"]
        queue["serialized_mainline"][0]["candidate"] = {"change_scope": {"files": ["tools/telemetry.py"]}}
        current_surface_id = self.mod.tranche_surface_id(
            queue["serialized_mainline"][0], control_pack, self.repo_root
        )
        queue["serialized_mainline"].append(
            {
                "id": "t_prev",
                "track": "serialized_mainline",
                "status": "hold",
                "objective": "Previous stale-restart branch reopened on the same surface.",
                "hypothesis_blocker_family": "stale_restart",
                "history": [
                    {
                        "timestamp_utc": "2026-04-08T11:00:00Z",
                        "surface_id": current_surface_id,
                        "decision": "hold",
                        "observed_blocker_family": "stale_restart",
                    }
                ],
            }
        )
        _write_yaml(self.repo_root / "phase5" / "queue.yaml", queue)

        with mock.patch.object(
            self.mod,
            "curl_health",
            return_value='{"healthy": true, "ready": true, "trade_mode": "shadow"}\n',
        ), mock.patch.object(
            self.mod.shutil,
            "disk_usage",
            return_value=shutil._ntuple_diskusage(
                total=20 * 1024 * 1024 * 1024,
                used=1,
                free=20 * 1024 * 1024 * 1024,
            ),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                self.mod.admission_check("t_main_1", 300, self.repo_root)

        self.assertIn("reopened", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
