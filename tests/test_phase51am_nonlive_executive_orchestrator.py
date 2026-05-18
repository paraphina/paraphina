import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


BASELINE_COMMIT = "18dd09512288a85e440d3977e32432c3aabc1190"


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def write_current_phase51ak_summary(path: Path, repo_root: Path) -> None:
    write_json(
        path,
        {
            "schema_version": 1,
            "run_id": "PHASE51AK-SOURCE-TRUTH-CAPTURE-HOLD-20260507T165132Z",
            "generated_at_utc": "2026-05-07T16:53:20+00:00",
            "baseline_commit": BASELINE_COMMIT,
            "gate_status": "HOLD",
            "gate_reason": "phase51ak_current_pack_incomplete_forward_refresh_required_nonlive_hold",
            "target_pack_mode": "current-pack",
            "target_run": str(repo_root / "target_run"),
            "request_pack": str(repo_root / "request_pack"),
            "native_role_capture_target_count": 287,
            "native_role_capture_target_ready_count": 73,
            "native_role_capture_target_missing_count": 214,
            "lighter_native_limit_capture_target_count": 3132,
            "lighter_native_limit_capture_target_ready_count": 0,
            "lighter_native_limit_capture_target_missing_count": 3132,
            "phase51v_downstream_chain_ready": False,
            "decision_status_counts": {
                "RECOVERED_CURRENT_PACK": 73,
                "UNRECOVERABLE_FROM_LOCAL_ARTIFACTS": 3346,
            },
            "decision_status_counts_by_target_type_venue": {
                "native_role:aster:UNRECOVERABLE_FROM_LOCAL_ARTIFACTS": 74,
                "native_role:extended:UNRECOVERABLE_FROM_LOCAL_ARTIFACTS": 7,
                "native_role:lighter:UNRECOVERABLE_FROM_LOCAL_ARTIFACTS": 125,
                "native_role:paradex:UNRECOVERABLE_FROM_LOCAL_ARTIFACTS": 8,
                "lighter_native_limit:lighter:UNRECOVERABLE_FROM_LOCAL_ARTIFACTS": 3132,
            },
            "next_required_action": "obtain_validated_mapping_or_forward_refresh_target_pack_with_event_time_sources",
            "clears_phase51_blockers": False,
            "no_live_flag": True,
            "approved_for_live": False,
            "approved_for_canary": False,
            "approved_for_model_training": False,
            "approved_for_capital_escalation": False,
            "admissible_for_financial_claim": False,
            "admissible_for_ev_admission": False,
            "live_orders_allowed": False,
            "capital_change_allowed": False,
            "risk_limit_relaxation_allowed": False,
            "raw_identifier_redaction_status": "PASS",
        },
    )


class TestPhase51amNonliveExecutiveOrchestrator(unittest.TestCase):
    def test_emits_subagent_packets_instead_of_dead_halt_when_no_route_is_ready(self):
        script_dir = Path(__file__).resolve().parents[1]
        tool_path = script_dir / "tools" / "phase51am_nonlive_executive_orchestrator.py"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            ak_summary = (
                tmp_path
                / "runs"
                / "phase51ak_blocker_resolution_runner"
                / "ak_hold"
                / "phase51ak_blocker_resolution_summary.json"
            )
            write_current_phase51ak_summary(ak_summary, tmp_path)

            fixture_run = (
                tmp_path
                / "runs"
                / "phase51al_forward_refresh_capture_gate"
                / "PHASE51AL-FORWARD-REFRESH-FIXTURE-HOLD"
            )
            (fixture_run / "target_run").mkdir(parents=True)
            (fixture_run / "phase51al_request_pack").mkdir()
            write_json(fixture_run / "candidate_manifest.forward_refresh.json", {"baseline_commit": BASELINE_COMMIT})
            write_json(
                fixture_run / "phase51al_forward_refresh_capture_summary.json",
                {
                    "schema_version": 1,
                    "run_id": "PHASE51AL-FORWARD-REFRESH-FIXTURE-HOLD",
                    "generated_at_utc": "2026-05-06T00:00:00+00:00",
                    "baseline_commit": BASELINE_COMMIT,
                    "gate_status": "HOLD",
                    "target_run": str(fixture_run / "target_run"),
                    "request_pack": str(fixture_run / "phase51al_request_pack"),
                    "candidate_manifest_path": str(fixture_run / "candidate_manifest.forward_refresh.json"),
                    "native_role_capture_target_count": 1,
                    "lighter_native_limit_capture_target_count": 1,
                    "source_row_count": 2,
                    "clears_phase51_blockers": False,
                    "no_live_flag": True,
                    "approved_for_live": False,
                    "live_orders_allowed": False,
                },
            )
            write_jsonl(
                tmp_path
                / "runs"
                / "phase51n_lighter_native_limit_time_alignment"
                / "pressure_hold"
                / "lighter_forward_native_limit_pressure_snapshot.jsonl",
                [],
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(tool_path),
                    "--repo-root",
                    str(tmp_path),
                    "--phase51ak-summary",
                    str(ak_summary),
                    "--output-root",
                    str(tmp_path / "phase51am_runs"),
                    "--run-id",
                    "phase51am_no_route",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"stdout={result.stdout}\nstderr={result.stderr}")

            run_dir = tmp_path / "phase51am_runs" / "phase51am_no_route"
            summary = json.loads((run_dir / "phase51am_nonlive_executive_orchestrator_summary.json").read_text())
            self.assertEqual(summary["gate_status"], "HOLD")
            self.assertEqual(summary["control_status"], "AWAITING_SOURCE_OWNER_INPUT")
            self.assertEqual(summary["selected_route"], "none")
            self.assertTrue(summary["implementation_route_blocked"])
            self.assertEqual(summary["subagent_work_packet_count"], 5)
            self.assertEqual(summary["subagent_prompt_count"], 5)
            self.assertEqual(summary["workflow_optimization_status"], "CONTINUOUS_ACTIVE")
            self.assertEqual(summary["workflow_optimization_action_count"], 6)
            self.assertFalse(summary["source_owner_intake_manifest_supplied"])
            self.assertEqual(summary["current_blocker"]["native_role_missing_by_venue"]["lighter"], 125)
            self.assertFalse(summary["live_orders_allowed"])
            self.assertTrue((run_dir / "source_owner_intake_manifest.template.json").exists())
            intake_status = json.loads((run_dir / "source_owner_intake_status.json").read_text())
            self.assertFalse(intake_status["manifest_supplied"])
            prompt_index = json.loads((run_dir / "subagent_prompt_pack" / "index.json").read_text())
            self.assertEqual(prompt_index["prompt_count"], 5)
            self.assertTrue((run_dir / "subagent_prompt_pack" / "01_phase51am_forward_refresh_source_owner.md").exists())

            ledger = [
                json.loads(line)
                for line in (run_dir / "phase51am_route_decision_ledger.jsonl").read_text().splitlines()
                if line.strip()
            ]
            self.assertEqual({row["route"] for row in ledger}, {"forward_refresh", "validated_mapping", "direct_private_rows", "lighter_pressure"})
            forward_route = next(row for row in ledger if row["route"] == "forward_refresh")
            self.assertEqual(forward_route["route_status"], "BLOCKED")
            self.assertEqual(forward_route["fixture_candidate_count"], 1)

            source_request = (run_dir / "source_owner_request.md").read_text()
            self.assertIn("native-role ready 73 / 287", source_request)
            self.assertIn("Complete sanitized Phase 5.1ab Lighter event-time pressure rows", source_request)

            optimization_records = [
                json.loads(line)
                for line in (run_dir / "workflow_optimization_ledger.jsonl").read_text().splitlines()
                if line.strip()
            ]
            self.assertEqual(optimization_records[0]["optimization_key"], "continuous_reclassification")
            self.assertEqual(optimization_records[-1]["optimization_key"], "no_route_handoff_compression")
            self.assertTrue(all(row["status"] == "ACTIVE" for row in optimization_records))

    def test_selects_real_forward_refresh_pack_for_phase51ak_validation(self):
        script_dir = Path(__file__).resolve().parents[1]
        tool_path = script_dir / "tools" / "phase51am_nonlive_executive_orchestrator.py"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            ak_summary = tmp_path / "phase51ak_blocker_resolution_summary.json"
            write_current_phase51ak_summary(ak_summary, tmp_path)

            phase51al_run = tmp_path / "real_forward_refresh"
            (phase51al_run / "target_run").mkdir(parents=True)
            (phase51al_run / "phase51al_request_pack").mkdir()
            write_json(phase51al_run / "candidate_manifest.forward_refresh.json", {"baseline_commit": BASELINE_COMMIT})
            al_summary = phase51al_run / "phase51al_forward_refresh_capture_summary.json"
            write_json(
                al_summary,
                {
                    "schema_version": 1,
                    "run_id": "PHASE51AL-REAL-FORWARD-REFRESH-HOLD",
                    "generated_at_utc": "2026-05-14T00:00:00+00:00",
                    "baseline_commit": BASELINE_COMMIT,
                    "gate_status": "HOLD",
                    "target_run": str(phase51al_run / "target_run"),
                    "request_pack": str(phase51al_run / "phase51al_request_pack"),
                    "candidate_manifest_path": str(phase51al_run / "candidate_manifest.forward_refresh.json"),
                    "native_role_capture_target_count": 1,
                    "lighter_native_limit_capture_target_count": 1,
                    "source_row_count": 2,
                    "clears_phase51_blockers": False,
                    "no_live_flag": True,
                    "approved_for_live": False,
                    "live_orders_allowed": False,
                },
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(tool_path),
                    "--repo-root",
                    str(tmp_path),
                    "--phase51ak-summary",
                    str(ak_summary),
                    "--phase51al-summary",
                    str(al_summary),
                    "--output-root",
                    str(tmp_path / "phase51am_runs"),
                    "--run-id",
                    "phase51am_forward_ready",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"stdout={result.stdout}\nstderr={result.stderr}")

            run_dir = tmp_path / "phase51am_runs" / "phase51am_forward_ready"
            summary = json.loads((run_dir / "phase51am_nonlive_executive_orchestrator_summary.json").read_text())
            self.assertEqual(summary["control_status"], "READY_TO_EXECUTE_SELECTED_ROUTE")
            self.assertEqual(summary["selected_route"], "forward_refresh")
            self.assertEqual(summary["selected_route_decision"]["route_status"], "READY_TO_VALIDATE")
            self.assertIn("--target-pack-mode forward-refresh", summary["selected_route_decision"]["command_template"])
            self.assertFalse(summary["implementation_route_blocked"])
            self.assertEqual(summary["subagent_prompt_count"], 2)
            self.assertEqual(summary["workflow_optimization_status"], "CONTINUOUS_ACTIVE")

            packets = [
                json.loads(line)
                for line in (run_dir / "subagent_work_packets.jsonl").read_text().splitlines()
                if line.strip()
            ]
            self.assertEqual(packets[0]["packet"], "phase51am_execute_selected_route")
            self.assertIn("phase51ak_blocker_resolution_runner.py", packets[0]["command_template"])

            optimization_records = [
                json.loads(line)
                for line in (run_dir / "workflow_optimization_ledger.jsonl").read_text().splitlines()
                if line.strip()
            ]
            self.assertEqual(optimization_records[-1]["optimization_key"], "ready_route_fast_path")

    def test_source_owner_intake_manifest_can_select_real_forward_refresh_pack(self):
        script_dir = Path(__file__).resolve().parents[1]
        tool_path = script_dir / "tools" / "phase51am_nonlive_executive_orchestrator.py"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            ak_summary = tmp_path / "phase51ak_blocker_resolution_summary.json"
            write_current_phase51ak_summary(ak_summary, tmp_path)

            phase51al_run = tmp_path / "real_forward_refresh"
            (phase51al_run / "target_run").mkdir(parents=True)
            (phase51al_run / "phase51al_request_pack").mkdir()
            write_json(phase51al_run / "candidate_manifest.forward_refresh.json", {"baseline_commit": BASELINE_COMMIT})
            al_summary = phase51al_run / "phase51al_forward_refresh_capture_summary.json"
            write_json(
                al_summary,
                {
                    "schema_version": 1,
                    "run_id": "PHASE51AL-REAL-FORWARD-REFRESH-HOLD",
                    "generated_at_utc": "2026-05-14T00:00:00+00:00",
                    "baseline_commit": BASELINE_COMMIT,
                    "gate_status": "HOLD",
                    "target_run": str(phase51al_run / "target_run"),
                    "request_pack": str(phase51al_run / "phase51al_request_pack"),
                    "candidate_manifest_path": str(phase51al_run / "candidate_manifest.forward_refresh.json"),
                    "native_role_capture_target_count": 1,
                    "lighter_native_limit_capture_target_count": 1,
                    "source_row_count": 2,
                    "clears_phase51_blockers": False,
                    "no_live_flag": True,
                    "approved_for_live": False,
                    "live_orders_allowed": False,
                },
            )
            intake_manifest = tmp_path / "source_owner_intake.json"
            write_json(
                intake_manifest,
                {
                    "schema_version": 1,
                    "material_change_reason": "real forward-refresh source-owner pack arrived",
                    "phase51al_summaries": [str(al_summary)],
                    "validated_mappings": [],
                    "phase51aj_source_json": [],
                    "phase51ab_pressure_jsonls": [],
                    "no_live_flag": True,
                    "approved_for_live": False,
                    "live_orders_allowed": False,
                },
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(tool_path),
                    "--repo-root",
                    str(tmp_path),
                    "--phase51ak-summary",
                    str(ak_summary),
                    "--source-owner-intake-manifest",
                    str(intake_manifest),
                    "--output-root",
                    str(tmp_path / "phase51am_runs"),
                    "--run-id",
                    "phase51am_intake_forward_ready",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"stdout={result.stdout}\nstderr={result.stderr}")

            run_dir = tmp_path / "phase51am_runs" / "phase51am_intake_forward_ready"
            summary = json.loads((run_dir / "phase51am_nonlive_executive_orchestrator_summary.json").read_text())
            self.assertTrue(summary["source_owner_intake_manifest_supplied"])
            self.assertEqual(summary["selected_route"], "forward_refresh")
            intake_status = json.loads((run_dir / "source_owner_intake_status.json").read_text())
            self.assertTrue(intake_status["manifest_supplied"])
            self.assertTrue(intake_status["material_change_reason_supplied"])
            self.assertEqual(intake_status["phase51al_summary_count"], 1)

    def test_surfaces_scoped_source_owner_readiness_without_route_or_global_clearance(self):
        script_dir = Path(__file__).resolve().parents[1]
        tool_path = script_dir / "tools" / "phase51am_nonlive_executive_orchestrator.py"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            ak_summary = tmp_path / "phase51ak_blocker_resolution_summary.json"
            write_json(
                ak_summary,
                {
                    "schema_version": 1,
                    "run_id": "PHASE51AK-FUTURE-LIGHTER-NATIVE-ROLE-VALIDATION",
                    "generated_at_utc": "2026-05-18T00:00:00+00:00",
                    "baseline_commit": BASELINE_COMMIT,
                    "gate_status": "HOLD",
                    "gate_reason": "phase51ak_source_owner_native_role_ready_hi_deferred_nonlive_hold",
                    "target_pack_mode": "forward-refresh",
                    "target_run": str(tmp_path / "target_run"),
                    "request_pack": str(tmp_path / "request_pack"),
                    "native_role_capture_target_count": 1,
                    "native_role_capture_target_ready_count": 1,
                    "native_role_capture_target_missing_count": 0,
                    "source_owner_native_role_evidence_ready": True,
                    "source_owner_native_role_ready_without_h_i": True,
                    "phase51_source_owner_blocker_status": "SOURCE_OWNER_NATIVE_ROLE_READY_HI_DEFERRED",
                    "lighter_pressure_unavailable_governance_accepted": True,
                    "h_i_feature_matrix_deferred": True,
                    "h_i_feature_matrix_deferred_reason": (
                        "source_owner_native_role_scope_does_not_require_pfill_feature_matrix"
                    ),
                    "source_owner_scope_next_required_action": (
                        "record_scoped_source_owner_native_role_acceptance_and_defer_h_i_calibration"
                    ),
                    "lighter_native_limit_capture_target_count": 1,
                    "lighter_native_limit_capture_target_ready_count": 0,
                    "lighter_native_limit_capture_target_missing_count": 1,
                    "lighter_native_limit_pressure_unavailable_target_count": 1,
                    "phase51v_downstream_chain_ready": False,
                    "decision_status_counts": {
                        "PRESSURE_UNAVAILABLE_GOVERNANCE_HOLD": 1,
                        "READY_FORWARD_REFRESH_PACK": 1,
                    },
                    "decision_status_counts_by_target_type_venue": {
                        "native_role:lighter:READY_FORWARD_REFRESH_PACK": 1,
                        "lighter_native_limit:lighter:PRESSURE_UNAVAILABLE_GOVERNANCE_HOLD": 1,
                    },
                    "next_required_action": "record_scoped_source_owner_native_role_acceptance_and_defer_h_i_calibration",
                    "clears_phase51_blockers": False,
                    "no_live_flag": True,
                    "approved_for_live": False,
                    "approved_for_canary": False,
                    "approved_for_model_training": False,
                    "approved_for_capital_escalation": False,
                    "admissible_for_financial_claim": False,
                    "admissible_for_ev_admission": False,
                    "live_orders_allowed": False,
                    "capital_change_allowed": False,
                    "risk_limit_relaxation_allowed": False,
                    "raw_identifier_redaction_status": "PASS",
                },
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(tool_path),
                    "--repo-root",
                    str(tmp_path),
                    "--phase51ak-summary",
                    str(ak_summary),
                    "--output-root",
                    str(tmp_path / "phase51am_runs"),
                    "--run-id",
                    "phase51am_source_owner_scope_ready",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"stdout={result.stdout}\nstderr={result.stderr}")

            run_dir = tmp_path / "phase51am_runs" / "phase51am_source_owner_scope_ready"
            summary = json.loads((run_dir / "phase51am_nonlive_executive_orchestrator_summary.json").read_text())
            self.assertEqual(summary["control_status"], "AWAITING_SOURCE_OWNER_INPUT")
            self.assertEqual(summary["selected_route"], "none")
            self.assertTrue(summary["source_owner_native_role_evidence_ready"])
            self.assertEqual(
                summary["phase51_source_owner_blocker_status"],
                "SOURCE_OWNER_NATIVE_ROLE_READY_HI_DEFERRED",
            )
            self.assertTrue(summary["lighter_pressure_unavailable_governance_accepted"])
            self.assertTrue(summary["h_i_feature_matrix_deferred"])
            self.assertFalse(summary["clears_phase51_blockers"])
            self.assertFalse(summary["live_orders_allowed"])
            ledger = [
                json.loads(line)
                for line in (run_dir / "phase51am_route_decision_ledger.jsonl").read_text().splitlines()
                if line.strip()
            ]
            self.assertNotIn("source_owner_native_role_scope", {row["route"] for row in ledger})

    def test_compares_previous_phase51am_summary_for_continuous_optimization(self):
        script_dir = Path(__file__).resolve().parents[1]
        tool_path = script_dir / "tools" / "phase51am_nonlive_executive_orchestrator.py"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            ak_summary = tmp_path / "phase51ak_blocker_resolution_summary.json"
            write_current_phase51ak_summary(ak_summary, tmp_path)
            previous_summary = tmp_path / "previous_phase51am_summary.json"
            write_json(
                previous_summary,
                {
                    "schema_version": 1,
                    "run_id": "phase51am_previous_no_route",
                    "generated_at_utc": "2026-05-13T00:00:00+00:00",
                    "baseline_commit": BASELINE_COMMIT,
                    "gate_status": "HOLD",
                    "control_status": "AWAITING_SOURCE_OWNER_INPUT",
                    "selected_route": "none",
                    "ready_route_count": 0,
                    "current_blocker": {
                        "native_role_capture_target_ready_count": 73,
                        "native_role_capture_target_missing_count": 214,
                        "lighter_native_limit_capture_target_ready_count": 0,
                        "lighter_native_limit_capture_target_missing_count": 3132,
                        "unrecoverable_from_local_artifacts_count": 3346,
                    },
                    "clears_phase51_blockers": False,
                    "no_live_flag": True,
                    "approved_for_live": False,
                    "live_orders_allowed": False,
                },
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(tool_path),
                    "--repo-root",
                    str(tmp_path),
                    "--phase51ak-summary",
                    str(ak_summary),
                    "--previous-phase51am-summary",
                    str(previous_summary),
                    "--output-root",
                    str(tmp_path / "phase51am_runs"),
                    "--run-id",
                    "phase51am_previous_compare",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"stdout={result.stdout}\nstderr={result.stderr}")

            run_dir = tmp_path / "phase51am_runs" / "phase51am_previous_compare"
            summary = json.loads((run_dir / "phase51am_nonlive_executive_orchestrator_summary.json").read_text())
            self.assertEqual(summary["phase51am_delta"]["staleness_status"], "UNCHANGED_NO_READY_ROUTE")
            self.assertEqual(
                summary["phase51am_delta"]["optimization_signal"],
                "avoid_duplicate_local_mining_and_keep_source_owner_handoff",
            )
            self.assertFalse(summary["phase51am_delta"]["blocker_counts_changed"])
            self.assertEqual(summary["workflow_optimization_action_count"], 7)

            optimization_records = [
                json.loads(line)
                for line in (run_dir / "workflow_optimization_ledger.jsonl").read_text().splitlines()
                if line.strip()
            ]
            self.assertEqual(optimization_records[-1]["optimization_key"], "previous_run_delta_monitor")
            self.assertEqual(optimization_records[-1]["previous_run_id"], "phase51am_previous_no_route")

    def test_rejects_env_mapping_paths_fail_closed(self):
        script_dir = Path(__file__).resolve().parents[1]
        tool_path = script_dir / "tools" / "phase51am_nonlive_executive_orchestrator.py"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            env_path = tmp_path / "mapping.env"
            env_path.write_text("not_used=true\n", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(tool_path),
                    "--repo-root",
                    str(tmp_path),
                    "--validated-mapping",
                    str(env_path),
                    "--output-root",
                    str(tmp_path / "phase51am_runs"),
                    "--run-id",
                    "phase51am_reject_env",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("must not reference .env", result.stderr)

    def test_rejects_network_paths_from_source_owner_intake_manifest(self):
        script_dir = Path(__file__).resolve().parents[1]
        tool_path = script_dir / "tools" / "phase51am_nonlive_executive_orchestrator.py"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            intake_manifest = tmp_path / "source_owner_intake.json"
            write_json(
                intake_manifest,
                {
                    "schema_version": 1,
                    "material_change_reason": "unsafe network path should fail closed",
                    "phase51al_summaries": [],
                    "validated_mappings": ["https://example.invalid/mapping.json"],
                    "phase51aj_source_json": [],
                    "phase51ab_pressure_jsonls": [],
                    "no_live_flag": True,
                    "approved_for_live": False,
                    "live_orders_allowed": False,
                },
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(tool_path),
                    "--repo-root",
                    str(tmp_path),
                    "--source-owner-intake-manifest",
                    str(intake_manifest),
                    "--output-root",
                    str(tmp_path / "phase51am_runs"),
                    "--run-id",
                    "phase51am_reject_network_intake",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("local filesystem path", result.stderr)

    def test_rejects_unsafe_mapping_content_before_route_ready(self):
        script_dir = Path(__file__).resolve().parents[1]
        tool_path = script_dir / "tools" / "phase51am_nonlive_executive_orchestrator.py"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            ak_summary = tmp_path / "phase51ak_blocker_resolution_summary.json"
            write_current_phase51ak_summary(ak_summary, tmp_path)
            mapping = tmp_path / "mapping.jsonl"
            write_jsonl(mapping, [{"source_record_sha256": "abc", "order_id": "raw-order"}])

            result = subprocess.run(
                [
                    sys.executable,
                    str(tool_path),
                    "--repo-root",
                    str(tmp_path),
                    "--phase51ak-summary",
                    str(ak_summary),
                    "--validated-mapping",
                    str(mapping),
                    "--output-root",
                    str(tmp_path / "phase51am_runs"),
                    "--run-id",
                    "phase51am_reject_raw_mapping",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("raw identifier", result.stderr)

    def test_rejects_secret_direct_private_source_before_route_ready(self):
        script_dir = Path(__file__).resolve().parents[1]
        tool_path = script_dir / "tools" / "phase51am_nonlive_executive_orchestrator.py"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            ak_summary = tmp_path / "phase51ak_blocker_resolution_summary.json"
            write_current_phase51ak_summary(ak_summary, tmp_path)
            direct_source = tmp_path / "direct_source.jsonl"
            write_jsonl(direct_source, [{"externalOrderId": "hash-ok", "api_key": "secret"}])

            result = subprocess.run(
                [
                    sys.executable,
                    str(tool_path),
                    "--repo-root",
                    str(tmp_path),
                    "--phase51ak-summary",
                    str(ak_summary),
                    "--phase51aj-source-json",
                    f"extended={direct_source}",
                    "--output-root",
                    str(tmp_path / "phase51am_runs"),
                    "--run-id",
                    "phase51am_reject_secret_direct",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("secret-shaped", result.stderr)

    def test_rejects_unsafe_lighter_pressure_content_before_route_ready(self):
        script_dir = Path(__file__).resolve().parents[1]
        tool_path = script_dir / "tools" / "phase51am_nonlive_executive_orchestrator.py"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            ak_summary = tmp_path / "phase51ak_blocker_resolution_summary.json"
            write_current_phase51ak_summary(ak_summary, tmp_path)
            pressure = tmp_path / "lighter_pressure.jsonl"
            write_jsonl(pressure, [{"approved_for_live": True, "active_order_headroom_account": 1}])

            result = subprocess.run(
                [
                    sys.executable,
                    str(tool_path),
                    "--repo-root",
                    str(tmp_path),
                    "--phase51ak-summary",
                    str(ak_summary),
                    "--phase51ab-pressure-jsonl",
                    str(pressure),
                    "--output-root",
                    str(tmp_path / "phase51am_runs"),
                    "--run-id",
                    "phase51am_reject_unsafe_pressure",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("approved_for_live=true", result.stderr)


if __name__ == "__main__":
    unittest.main()
