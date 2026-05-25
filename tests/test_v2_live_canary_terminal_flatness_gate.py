import json
import tempfile
import unittest
from pathlib import Path

from tools import v2_live_canary_terminal_flatness_gate as gate


EXPECTED = ["hyperliquid", "lighter", "extended", "aster", "paradex"]
RUN_TOKEN = "v2_live_ranked_admission_micro_canary_test"


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def venue_result(venue: str, *, position=0.0, open_orders=0, ok=True):
    return {
        "venue": venue,
        "market": "ETH",
        "position_base": position,
        "open_order_count": open_orders,
        "open_order_count_known": True,
        "ok": ok,
        "violations": [],
        "errors": [],
    }


def audit_doc(**overrides):
    data = {
        "ok": True,
        "position_tol_base": 0.0025,
        "max_open_orders": 0,
        "allow_unknown_open_orders": False,
        "results": [venue_result(venue) for venue in EXPECTED],
        "violations": [],
    }
    data.update(overrides)
    return data


def manifest_doc(**overrides):
    data = {
        "artifact_type": "v2_authority_decision_evidence_manifest",
        "decision_validation_status": "pass",
        "files": [
            {
                "path": f"{RUN_TOKEN}.jsonl",
                "sha256": "0" * 64,
                "bytes": 123,
            }
        ],
        "governance": {
            "approved_for_promotion": False,
            "blocker_cleared": False,
            "pressure_complete_claim": False,
        },
    }
    data.update(overrides)
    return data


def order_path_coverage_doc(venue: str, **coverage_overrides):
    coverage = {
        "bad_place_policy_count": 0,
        "observed_order_actions": [],
        "observed_order_statuses": [],
        "target_cancel_acks": 0,
        "target_cancel_intents": 0,
        "target_fills": 0,
        "target_place_acks": 0,
        "target_place_intents": 0,
        "target_replace_events": 0,
        "target_would_send_cancels": 0,
        "target_would_send_places": 0,
        "telemetry_rows": 10,
    }
    coverage.update(coverage_overrides)
    return {
        "artifact_type": "v2_telemetry_order_path_coverage_manifest",
        "coverage": coverage,
        "governance": {
            "blocker_cleared": False,
            "pressure_complete_claim": False,
        },
        "scope": {
            "target_venue": venue,
            "telemetry_only_order_path_coverage": True,
        },
    }


class TestV2LiveCanaryTerminalFlatnessGate(unittest.TestCase):
    def evaluate(
        self,
        data: dict,
        *,
        manifest: dict | None = None,
        audit_captured_after_run: bool = True,
        audit_captured_after_terminal_cleanup: bool = True,
        run_token: str = RUN_TOKEN,
        live_stderr: str | None = None,
        live_exit_code: str | None = None,
        live_end_utc: str | None = None,
        live_summary: dict | None = None,
        min_ticks_run: int | None = None,
        min_run_duration_ms: int | None = None,
        promotion_cleanup_strict: bool = False,
        order_path_coverages: list[dict] | None = None,
        position_tol_base: float = 0.0025,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / f"{RUN_TOKEN}_terminal_audit.json"
            canary_manifest = root / f"{RUN_TOKEN}_manifest.json"
            live_stderr_path = None
            live_exit_code_path = None
            live_end_utc_path = None
            live_summary_path = None
            order_path_coverage_paths = []
            write_json(audit, data)
            write_json(canary_manifest, manifest if manifest is not None else manifest_doc())
            if live_stderr is not None:
                live_stderr_path = root / f"{RUN_TOKEN}_live_stderr.log"
                live_stderr_path.write_text(live_stderr, encoding="utf-8")
            if live_exit_code is not None or promotion_cleanup_strict:
                live_exit_code_path = root / f"{RUN_TOKEN}_live_exit_code.txt"
                live_exit_code_path.write_text(
                    "0" if live_exit_code is None else live_exit_code,
                    encoding="utf-8",
                )
            if live_end_utc is not None or promotion_cleanup_strict:
                live_end_utc_path = root / f"{RUN_TOKEN}_live_end_utc.txt"
                live_end_utc_path.write_text(
                    "2026-05-25T00:10:00+00:00\n" if live_end_utc is None else live_end_utc,
                    encoding="utf-8",
                )
            if live_summary is not None or promotion_cleanup_strict:
                live_summary_path = root / f"{RUN_TOKEN}_summary.json"
                write_json(
                    live_summary_path,
                    live_summary
                    if live_summary is not None
                    else {
                        "execution_mode": "live",
                        "trade_mode": "live",
                        "ticks_run": 2400,
                        "run_duration_ms": 600000,
                    },
                )
            for idx, order_path_coverage in enumerate(order_path_coverages or []):
                path = root / f"{RUN_TOKEN}_order_path_{idx}.json"
                write_json(path, order_path_coverage)
                order_path_coverage_paths.append(path)
            return gate.evaluate_terminal_flatness(
                venue_audit_path=audit,
                expected_venues=EXPECTED,
                position_tol_base=position_tol_base,
                canary_manifest_path=canary_manifest,
                run_token=run_token,
                audit_captured_after_run=audit_captured_after_run,
                audit_captured_after_terminal_cleanup=audit_captured_after_terminal_cleanup,
                live_stderr_path=live_stderr_path,
                live_exit_code_path=live_exit_code_path,
                live_end_utc_path=live_end_utc_path,
                live_summary_path=live_summary_path,
                min_ticks_run=min_ticks_run,
                min_run_duration_ms=min_run_duration_ms,
                promotion_cleanup_strict=promotion_cleanup_strict,
                order_path_coverage_paths=order_path_coverage_paths,
            )

    def test_flat_all_venues_passes_without_clearance_claims(self):
        report = self.evaluate(audit_doc())

        self.assertEqual(report["terminal_flatness_gate_status"], "PASS")
        self.assertFalse(report["governance"]["approved_for_promotion"])
        self.assertFalse(report["governance"]["approved_for_live"])
        self.assertFalse(report["governance"]["approved_for_capital_escalation"])
        self.assertFalse(report["governance"]["blocker_cleared"])
        self.assertFalse(report["governance"]["pressure_complete_claim"])
        self.assertFalse(report["governance"]["v2_authority_admission"])
        self.assertFalse(report["governance"]["live_orders_allowed"])
        self.assertFalse(report["governance"]["capital_change_allowed"])

    def test_nonflat_position_holds(self):
        data = audit_doc(
            ok=False,
            results=[
                venue_result("hyperliquid"),
                venue_result("lighter", position=0.01, ok=False),
                venue_result("extended"),
                venue_result("aster"),
                venue_result("paradex"),
            ],
        )

        report = self.evaluate(data)

        self.assertEqual(report["terminal_flatness_gate_status"], "HOLD")
        self.assertIn("lighter:position_base_not_flat", report["terminal_flatness_gate_reasons"])

    def test_lighter_sub_lot_residual_is_terminal_dust_hold_not_promotion(self):
        data = audit_doc(
            ok=False,
            results=[
                venue_result("hyperliquid"),
                venue_result(
                    "lighter",
                    position=-0.004,
                    ok=False,
                )
                | {"violations": ["abs(position_base)=0.00400000 > 0.00250000"]},
                venue_result("extended"),
                venue_result("aster"),
                venue_result("paradex"),
            ],
            violations=["lighter: abs(position_base)=0.00400000 > 0.00250000"],
        )

        report = self.evaluate(data)

        self.assertEqual(report["terminal_flatness_gate_status"], "HOLD")
        self.assertEqual(report["terminal_dust"]["status"], "DUST_HOLD")
        self.assertFalse(report["terminal_dust"]["accepted_as_flat"])
        self.assertEqual(report["terminal_dust"]["candidate_venues"][0]["venue"], "lighter")
        self.assertEqual(
            report["terminal_dust"]["candidate_venues"][0]["terminal_dust_tolerance_base"],
            0.005,
        )
        self.assertFalse(report["closeout_status"]["promotion_ready"])
        self.assertFalse(report["governance"]["approved_for_promotion"])

    def test_lighter_dust_with_open_order_is_not_dust_candidate(self):
        data = audit_doc(
            ok=False,
            results=[
                venue_result("hyperliquid"),
                venue_result("lighter", position=-0.004, open_orders=1, ok=False),
                venue_result("extended"),
                venue_result("aster"),
                venue_result("paradex"),
            ],
        )

        report = self.evaluate(data)

        self.assertEqual(report["terminal_flatness_gate_status"], "HOLD")
        self.assertEqual(report["terminal_dust"]["status"], "NOT_APPLICABLE")
        self.assertIn("lighter:open_orders_present", report["terminal_flatness_gate_reasons"])

    def test_lighter_above_dust_tolerance_is_not_dust_candidate(self):
        data = audit_doc(
            ok=False,
            results=[
                venue_result("hyperliquid"),
                venue_result("lighter", position=-0.006, ok=False),
                venue_result("extended"),
                venue_result("aster"),
                venue_result("paradex"),
            ],
        )

        report = self.evaluate(data)

        self.assertEqual(report["terminal_flatness_gate_status"], "HOLD")
        self.assertEqual(report["terminal_dust"]["status"], "NOT_APPLICABLE")
        self.assertIn("lighter:position_base_not_flat", report["terminal_flatness_gate_reasons"])

    def test_non_lighter_residual_is_not_dust_candidate(self):
        data = audit_doc(
            ok=False,
            results=[
                venue_result("hyperliquid"),
                venue_result("lighter"),
                venue_result("extended", position=-0.004, ok=False),
                venue_result("aster"),
                venue_result("paradex"),
            ],
        )

        report = self.evaluate(data)

        self.assertEqual(report["terminal_flatness_gate_status"], "HOLD")
        self.assertEqual(report["terminal_dust"]["status"], "NOT_APPLICABLE")
        self.assertIn("extended:position_base_not_flat", report["terminal_flatness_gate_reasons"])

    def test_widened_position_tolerance_cannot_silently_clear_lighter_dust(self):
        data = audit_doc(
            position_tol_base=0.005,
            results=[
                venue_result("hyperliquid"),
                venue_result("lighter", position=-0.004, ok=True),
                venue_result("extended"),
                venue_result("aster"),
                venue_result("paradex"),
            ],
        )

        report = self.evaluate(data, position_tol_base=0.005)

        self.assertEqual(report["terminal_flatness_gate_status"], "HOLD")
        self.assertIn(
            "position_tolerance_exceeds_strict_terminal_flatness",
            report["terminal_flatness_gate_reasons"],
        )
        self.assertEqual(report["terminal_dust"]["status"], "DUST_HOLD")
        self.assertFalse(report["closeout_status"]["promotion_ready"])

    def test_open_order_holds(self):
        data = audit_doc(
            ok=False,
            results=[
                venue_result("hyperliquid"),
                venue_result("lighter", open_orders=1, ok=False),
                venue_result("extended"),
                venue_result("aster"),
                venue_result("paradex"),
            ],
        )

        report = self.evaluate(data)

        self.assertEqual(report["terminal_flatness_gate_status"], "HOLD")
        self.assertIn("lighter:open_orders_present", report["terminal_flatness_gate_reasons"])

    def test_unknown_open_orders_hold(self):
        data = audit_doc()
        data["results"][1]["open_order_count_known"] = False

        report = self.evaluate(data)

        self.assertEqual(report["terminal_flatness_gate_status"], "HOLD")
        self.assertIn("lighter:open_order_count_unknown", report["terminal_flatness_gate_reasons"])

    def test_missing_venue_holds(self):
        data = audit_doc(results=[venue_result(venue) for venue in EXPECTED if venue != "paradex"])

        report = self.evaluate(data)

        self.assertEqual(report["terminal_flatness_gate_status"], "HOLD")
        self.assertIn("paradex:missing_venue_result", report["terminal_flatness_gate_reasons"])

    def test_tolerance_mismatch_holds(self):
        report = self.evaluate(audit_doc(position_tol_base=0.01))

        self.assertEqual(report["terminal_flatness_gate_status"], "HOLD")
        self.assertIn("venue_audit_position_tolerance_mismatch", report["terminal_flatness_gate_reasons"])

    def test_nan_position_holds(self):
        data = audit_doc()
        data["results"][1]["position_base"] = float("nan")

        report = self.evaluate(data)

        self.assertEqual(report["terminal_flatness_gate_status"], "HOLD")
        self.assertIn("lighter:position_base_missing", report["terminal_flatness_gate_reasons"])

    def test_infinite_audit_tolerance_holds(self):
        report = self.evaluate(audit_doc(position_tol_base=float("inf")))

        self.assertEqual(report["terminal_flatness_gate_status"], "HOLD")
        self.assertIn("venue_audit_missing_position_tolerance", report["terminal_flatness_gate_reasons"])

    def test_missing_after_run_confirmation_holds(self):
        report = self.evaluate(audit_doc(), audit_captured_after_run=False)

        self.assertEqual(report["terminal_flatness_gate_status"], "HOLD")
        self.assertIn(
            "run_binding_after_run_confirmation_missing",
            report["terminal_flatness_gate_reasons"],
        )

    def test_missing_after_terminal_cleanup_confirmation_holds(self):
        report = self.evaluate(audit_doc(), audit_captured_after_terminal_cleanup=False)

        self.assertEqual(report["terminal_flatness_gate_status"], "HOLD")
        self.assertIn(
            "run_binding_after_terminal_cleanup_confirmation_missing",
            report["terminal_flatness_gate_reasons"],
        )

    def test_nonzero_audit_max_open_orders_holds(self):
        report = self.evaluate(audit_doc(max_open_orders=1))

        self.assertEqual(report["terminal_flatness_gate_status"], "HOLD")
        self.assertIn(
            "venue_audit_max_open_orders_not_zero",
            report["terminal_flatness_gate_reasons"],
        )

    def test_allow_unknown_open_orders_holds(self):
        report = self.evaluate(audit_doc(allow_unknown_open_orders=True))

        self.assertEqual(report["terminal_flatness_gate_status"], "HOLD")
        self.assertIn(
            "venue_audit_allows_unknown_open_orders",
            report["terminal_flatness_gate_reasons"],
        )

    def test_missing_run_token_holds(self):
        report = self.evaluate(audit_doc(), run_token="")

        self.assertEqual(report["terminal_flatness_gate_status"], "HOLD")
        self.assertIn("run_binding_missing_token", report["terminal_flatness_gate_reasons"])

    def test_manifest_with_promotion_claim_holds(self):
        manifest = manifest_doc(
            governance={
                "approved_for_promotion": True,
                "blocker_cleared": False,
                "pressure_complete_claim": False,
            }
        )

        report = self.evaluate(audit_doc(), manifest=manifest)

        self.assertEqual(report["terminal_flatness_gate_status"], "HOLD")
        self.assertIn(
            "run_binding_canary_manifest_already_promotion",
            report["terminal_flatness_gate_reasons"],
        )

    def test_promotion_cleanup_strict_holds_on_terminal_cancel_timeout(self):
        report = self.evaluate(
            audit_doc(),
            live_stderr="[runner] tick=1201 terminal_exit_cancel: timeout after 30000ms waiting for response\n",
            promotion_cleanup_strict=True,
        )

        self.assertEqual(report["terminal_flatness_gate_status"], "HOLD")
        self.assertIn(
            "terminal_cleanup_cancel_timeout_present",
            report["terminal_flatness_gate_reasons"],
        )
        self.assertEqual(report["terminal_cleanup"]["terminal_cancel_timeout_count"], 1)
        self.assertEqual(report["terminal_cleanup"]["terminal_cancel_timeout_ticks"], [1201])

    def test_non_strict_cleanup_log_preserves_operational_flatness_pass(self):
        report = self.evaluate(
            audit_doc(),
            live_stderr="[runner] tick=1201 terminal_exit_cancel: timeout after 30000ms waiting for response\n",
            promotion_cleanup_strict=False,
        )

        self.assertEqual(report["terminal_flatness_gate_status"], "PASS")
        self.assertEqual(report["terminal_cleanup"]["terminal_cancel_timeout_count"], 1)

    def test_promotion_cleanup_strict_holds_on_hyperliquid_cancel_all_backlog(self):
        report = self.evaluate(
            audit_doc(),
            live_stderr=(
                "HL_POST_SUBMIT submit_path=ws_post post_id=1 action_label=cancel_all "
                "batch_kind=cancel_all batch_size=1 post_inflight=1\n"
                "HL_POST_SUBMIT submit_path=ws_post post_id=2 action_label=cancel_all "
                "batch_kind=cancel_all batch_size=1 post_inflight=2\n"
            ),
            promotion_cleanup_strict=True,
        )

        self.assertEqual(report["terminal_flatness_gate_status"], "HOLD")
        self.assertIn(
            "terminal_cleanup_hyperliquid_cancel_all_post_inflight_backlog",
            report["terminal_flatness_gate_reasons"],
        )
        self.assertEqual(report["terminal_cleanup"]["hyperliquid_cancel_all_post_inflight_max"], 2)

    def test_promotion_cleanup_strict_allows_pre_clean_account_truth_miss_with_clean_direct_audit(self):
        report = self.evaluate(
            audit_doc(),
            live_stderr=(
                "[runner] canary_exit_position_flatten_cleanup account_refresh_not_fresh "
                "phase=pre_clean venue=paradex account_ok=true account_available=false\n"
                "[runner] canary_exit_position_flatten_cleanup blocked_pre_clean_account_truth "
                "attempt=1 requested_venues=extended,hyperliquid,aster,lighter,paradex "
                "fresh_venues=extended,hyperliquid,aster,lighter position_tol_tao=0.000100\n"
            ),
            promotion_cleanup_strict=True,
        )

        self.assertEqual(report["terminal_flatness_gate_status"], "PASS")
        self.assertEqual(report["terminal_cleanup"]["blocked_pre_clean_account_truth_count"], 1)
        self.assertTrue(
            report["terminal_cleanup"]["pre_clean_account_truth_direct_venue_audit_cleared"]
        )
        self.assertEqual(report["closeout_status"]["direct_venue_audit_status"], "PASS")
        self.assertEqual(report["closeout_status"]["promotion_cleanup_strict_status"], "PASS")
        self.assertTrue(report["closeout_status"]["promotion_ready"])

    def test_promotion_cleanup_strict_holds_pre_clean_account_truth_when_direct_audit_dirty(self):
        data = audit_doc(
            ok=False,
            results=[
                venue_result("hyperliquid"),
                venue_result("lighter"),
                venue_result("extended"),
                venue_result("aster"),
                venue_result("paradex", open_orders=1, ok=False),
            ],
        )
        report = self.evaluate(
            data,
            live_stderr=(
                "[runner] canary_exit_position_flatten_cleanup blocked_pre_clean_account_truth "
                "attempt=1 requested_venues=extended,hyperliquid,aster,lighter,paradex "
                "fresh_venues=extended,hyperliquid,aster,lighter position_tol_tao=0.000100\n"
            ),
            promotion_cleanup_strict=True,
        )

        self.assertEqual(report["terminal_flatness_gate_status"], "HOLD")
        self.assertIn("terminal_cleanup_account_truth_blocked", report["terminal_flatness_gate_reasons"])
        self.assertIn("paradex:open_orders_present", report["terminal_flatness_gate_reasons"])
        self.assertFalse(
            report["terminal_cleanup"]["pre_clean_account_truth_direct_venue_audit_cleared"]
        )

    def test_promotion_cleanup_strict_holds_on_post_dispatch_account_truth(self):
        report = self.evaluate(
            audit_doc(),
            live_stderr=(
                "[runner] canary_exit_position_flatten_cleanup blocked_post_dispatch_account_truth "
                "attempt=1 requested_venues=extended,hyperliquid,aster,lighter,paradex "
                "fresh_venues=extended,hyperliquid,aster,lighter position_tol_tao=0.000100\n"
            ),
            promotion_cleanup_strict=True,
        )

        self.assertEqual(report["terminal_flatness_gate_status"], "HOLD")
        self.assertIn("terminal_cleanup_account_truth_blocked", report["terminal_flatness_gate_reasons"])
        self.assertEqual(report["terminal_cleanup"]["blocked_post_dispatch_account_truth_count"], 1)
        self.assertEqual(report["closeout_status"]["promotion_cleanup_strict_status"], "HOLD")

    def test_promotion_cleanup_strict_allows_tracked_cancel_incomplete_with_clean_direct_audit(self):
        report = self.evaluate(
            audit_doc(),
            live_stderr=(
                "[runner] canary_exit_cancel_all_cleanup incomplete "
                "tracked_open_orders=1 venues=hyperliquid\n"
            ),
            promotion_cleanup_strict=True,
        )

        self.assertEqual(report["terminal_flatness_gate_status"], "PASS")
        self.assertEqual(report["terminal_cleanup"]["terminal_cancel_all_incomplete_count"], 1)
        self.assertEqual(
            report["terminal_cleanup"]["terminal_cancel_all_incomplete_venues"], ["hyperliquid"]
        )
        self.assertTrue(
            report["terminal_cleanup"]["cancel_all_incomplete_direct_venue_audit_cleared"]
        )

    def test_promotion_cleanup_strict_holds_tracked_cancel_incomplete_when_direct_audit_dirty(self):
        data = audit_doc(
            ok=False,
            results=[
                venue_result("hyperliquid", open_orders=1, ok=False),
                venue_result("lighter"),
                venue_result("extended"),
                venue_result("aster"),
                venue_result("paradex"),
            ],
        )
        report = self.evaluate(
            data,
            live_stderr=(
                "[runner] canary_exit_cancel_all_cleanup incomplete "
                "tracked_open_orders=1 venues=hyperliquid\n"
            ),
            promotion_cleanup_strict=True,
        )

        self.assertEqual(report["terminal_flatness_gate_status"], "HOLD")
        self.assertIn(
            "terminal_cleanup_cancel_all_incomplete",
            report["terminal_flatness_gate_reasons"],
        )
        self.assertFalse(
            report["terminal_cleanup"]["cancel_all_incomplete_direct_venue_audit_cleared"]
        )

    def test_promotion_cleanup_strict_splits_direct_audit_from_cleanup_hold(self):
        report = self.evaluate(
            audit_doc(),
            live_stderr=(
                "[runner] canary_exit_position_flatten_cleanup account_refresh_not_fresh "
                "phase=post_dispatch venue=paradex account_ok=true account_available=false\n"
            ),
            promotion_cleanup_strict=True,
        )

        self.assertEqual(report["terminal_flatness_gate_status"], "HOLD")
        self.assertEqual(report["closeout_status"]["direct_venue_audit_status"], "PASS")
        self.assertEqual(report["closeout_status"]["promotion_cleanup_strict_status"], "HOLD")
        self.assertEqual(report["closeout_status"]["run_binding_status"], "PASS")
        self.assertIn(
            "terminal_cleanup_account_refresh_not_fresh",
            report["closeout_status"]["promotion_cleanup_strict_reasons"],
        )

    def test_promotion_cleanup_strict_allows_transient_not_fresh_with_clean_closeout(self):
        report = self.evaluate(
            audit_doc(),
            live_stderr=(
                "[runner] canary_exit_position_flatten_cleanup account_refresh_not_fresh "
                "phase=post_dispatch venue=paradex account_ok=true account_available=false\n"
                "[runner] canary_exit_position_flatten_cleanup account_refresh_applied "
                "phase=final_check requested_venues=extended,hyperliquid,aster,lighter,paradex "
                "fresh_venues=extended,hyperliquid,aster,lighter,paradex position_changed=false "
                "remaining_venues=\n"
                "[runner] canary_exit_position_flatten_cleanup clean_after_final_account_refresh\n"
            ),
            promotion_cleanup_strict=True,
        )

        self.assertEqual(report["terminal_flatness_gate_status"], "PASS")
        self.assertEqual(report["terminal_cleanup"]["account_refresh_not_fresh_count"], 1)
        self.assertEqual(report["terminal_cleanup"]["clean_terminal_closeout_count"], 1)
        self.assertTrue(report["terminal_cleanup"]["final_check_account_truth"]["found"])
        self.assertTrue(
            report["terminal_cleanup"]["final_check_account_truth"]["fresh_covers_requested"]
        )
        self.assertTrue(
            report["terminal_cleanup"]["final_check_account_truth"][
                "after_latest_account_refresh_not_fresh"
            ]
        )
        self.assertEqual(report["closeout_status"]["promotion_cleanup_strict_status"], "PASS")

    def test_promotion_cleanup_strict_requires_final_truth_after_latest_not_fresh(self):
        report = self.evaluate(
            audit_doc(),
            live_stderr=(
                "[runner] canary_exit_position_flatten_cleanup account_refresh_applied "
                "phase=final_check requested_venues=extended,hyperliquid,aster,lighter,paradex "
                "fresh_venues=extended,hyperliquid,aster,lighter,paradex position_changed=false "
                "remaining_venues=\n"
                "[runner] canary_exit_position_flatten_cleanup clean_after_final_account_refresh\n"
                "[runner] canary_exit_position_flatten_cleanup account_refresh_not_fresh "
                "phase=post_dispatch venue=paradex account_ok=true account_available=false\n"
            ),
            promotion_cleanup_strict=True,
        )

        self.assertEqual(report["terminal_flatness_gate_status"], "HOLD")
        self.assertIn(
            "terminal_cleanup_account_refresh_not_fresh",
            report["terminal_flatness_gate_reasons"],
        )
        self.assertFalse(
            report["terminal_cleanup"]["final_check_account_truth"][
                "after_latest_account_refresh_not_fresh"
            ]
        )

    def test_promotion_cleanup_strict_requires_full_final_truth_to_clear_not_fresh(self):
        report = self.evaluate(
            audit_doc(),
            live_stderr=(
                "[runner] canary_exit_position_flatten_cleanup account_refresh_not_fresh "
                "phase=post_dispatch venue=paradex account_ok=true account_available=false\n"
                "[runner] canary_exit_position_flatten_cleanup account_refresh_applied "
                "phase=final_check requested_venues=extended,hyperliquid,aster,lighter,paradex "
                "fresh_venues=extended,hyperliquid,aster,lighter position_changed=false "
                "remaining_venues=\n"
                "[runner] canary_exit_position_flatten_cleanup clean_after_final_account_refresh\n"
            ),
            promotion_cleanup_strict=True,
        )

        self.assertEqual(report["terminal_flatness_gate_status"], "HOLD")
        self.assertIn(
            "terminal_cleanup_account_refresh_not_fresh",
            report["terminal_flatness_gate_reasons"],
        )
        self.assertFalse(
            report["terminal_cleanup"]["final_check_account_truth"]["fresh_covers_requested"]
        )

    def test_promotion_cleanup_strict_holds_on_blocked_final_account_truth(self):
        report = self.evaluate(
            audit_doc(),
            live_stderr=(
                "[runner] canary_exit_position_flatten_cleanup blocked_final_account_truth "
                "requested_venues=extended,hyperliquid,aster,lighter,paradex "
                "fresh_venues=extended,hyperliquid,aster,lighter position_tol_tao=0.002500\n"
            ),
            promotion_cleanup_strict=True,
        )

        self.assertEqual(report["terminal_flatness_gate_status"], "HOLD")
        self.assertIn(
            "terminal_cleanup_account_truth_blocked",
            report["terminal_flatness_gate_reasons"],
        )
        self.assertEqual(report["terminal_cleanup"]["blocked_final_account_truth_count"], 1)

    def test_promotion_cleanup_strict_allows_final_account_truth_missing_for_no_exposure_venue(self):
        report = self.evaluate(
            audit_doc(),
            live_stderr=(
                "[runner] canary_exit_position_flatten_cleanup account_refresh_not_fresh "
                "phase=final_check venue=paradex account_ok=true account_available=false\n"
                "[runner] canary_exit_position_flatten_cleanup account_refresh_applied "
                "phase=final_check requested_venues=extended,hyperliquid,aster,lighter,paradex "
                "fresh_venues=extended,hyperliquid,aster,lighter position_changed=false "
                "remaining_venues=\n"
                "[runner] canary_exit_position_flatten_cleanup blocked_final_account_truth "
                "requested_venues=extended,hyperliquid,aster,lighter,paradex "
                "fresh_venues=extended,hyperliquid,aster,lighter position_tol_tao=0.002500\n"
            ),
            promotion_cleanup_strict=True,
            order_path_coverages=[order_path_coverage_doc("paradex")],
        )

        self.assertEqual(report["terminal_flatness_gate_status"], "PASS")
        self.assertEqual(report["terminal_cleanup"]["blocked_final_account_truth_count"], 1)
        self.assertTrue(
            report["terminal_cleanup"][
                "final_account_truth_no_exposure_direct_venue_audit_cleared"
            ]
        )
        self.assertEqual(
            report["terminal_cleanup"]["final_account_truth_no_exposure_venues"], ["paradex"]
        )

    def test_promotion_cleanup_strict_allows_final_freshness_gap_without_order_path_coverage_when_bound_direct_audit_clean(self):
        report = self.evaluate(
            audit_doc(),
            live_stderr=(
                "[runner] canary_exit_position_flatten_cleanup account_refresh_not_fresh "
                "phase=final_check venue=paradex account_ok=true account_available=false\n"
                "[runner] canary_exit_position_flatten_cleanup account_refresh_applied "
                "phase=final_check requested_venues=extended,hyperliquid,aster,lighter,paradex "
                "fresh_venues=extended,hyperliquid,aster,lighter position_changed=false "
                "remaining_venues=\n"
                "[runner] canary_exit_position_flatten_cleanup blocked_final_account_truth "
                "requested_venues=extended,hyperliquid,aster,lighter,paradex "
                "fresh_venues=extended,hyperliquid,aster,lighter position_tol_tao=0.002500\n"
            ),
            promotion_cleanup_strict=True,
        )

        self.assertEqual(report["terminal_flatness_gate_status"], "PASS")
        self.assertFalse(
            report["terminal_cleanup"][
                "final_account_truth_no_exposure_direct_venue_audit_cleared"
            ]
        )
        self.assertTrue(
            report["terminal_cleanup"]["final_account_truth_direct_venue_audit_superseded"]
        )
        self.assertEqual(
            report["terminal_cleanup"]["final_account_truth_direct_venue_audit_venues"],
            ["paradex"],
        )
        self.assertFalse(report["closeout_status"]["promotion_ready"])

    def test_promotion_cleanup_strict_allows_final_freshness_gap_with_bound_direct_audit_when_venue_had_actions(self):
        report = self.evaluate(
            audit_doc(),
            live_stderr=(
                "[runner] canary_exit_position_flatten_cleanup account_refresh_not_fresh "
                "phase=final_check venue=paradex account_ok=true account_available=false\n"
                "[runner] canary_exit_position_flatten_cleanup account_refresh_applied "
                "phase=final_check requested_venues=extended,hyperliquid,aster,lighter,paradex "
                "fresh_venues=extended,hyperliquid,aster,lighter position_changed=false "
                "remaining_venues=\n"
                "[runner] canary_exit_position_flatten_cleanup blocked_final_account_truth "
                "requested_venues=extended,hyperliquid,aster,lighter,paradex "
                "fresh_venues=extended,hyperliquid,aster,lighter position_tol_tao=0.002500\n"
            ),
            promotion_cleanup_strict=True,
            order_path_coverages=[
                order_path_coverage_doc(
                    "paradex",
                    target_place_intents=1,
                    observed_order_actions=["place"],
                    observed_order_statuses=["intent"],
                )
            ],
        )

        self.assertEqual(report["terminal_flatness_gate_status"], "PASS")
        self.assertFalse(
            report["terminal_cleanup"][
                "final_account_truth_no_exposure_direct_venue_audit_cleared"
            ]
        )
        self.assertTrue(
            report["terminal_cleanup"]["final_account_truth_direct_venue_audit_superseded"]
        )
        self.assertEqual(
            report["terminal_cleanup"]["final_account_truth_direct_venue_audit_venues"],
            ["paradex"],
        )
        self.assertEqual(
            report["terminal_cleanup"]["final_account_truth_direct_venue_audit_reason"],
            "final_freshness_only_gap_cleared_by_bound_direct_venue_audit",
        )
        self.assertEqual(report["closeout_status"]["promotion_cleanup_strict_status"], "PASS")
        self.assertFalse(report["closeout_status"]["promotion_ready"])
        self.assertFalse(report["governance"]["blocker_cleared"])
        self.assertFalse(report["governance"]["approved_for_promotion"])

    def test_promotion_cleanup_strict_holds_final_direct_audit_supersession_when_remaining_nonempty(self):
        report = self.evaluate(
            audit_doc(),
            live_stderr=(
                "[runner] canary_exit_position_flatten_cleanup account_refresh_not_fresh "
                "phase=final_check venue=paradex account_ok=true account_available=false\n"
                "[runner] canary_exit_position_flatten_cleanup account_refresh_applied "
                "phase=final_check requested_venues=extended,hyperliquid,aster,lighter,paradex "
                "fresh_venues=extended,hyperliquid,aster,lighter position_changed=false "
                "remaining_venues=paradex\n"
                "[runner] canary_exit_position_flatten_cleanup blocked_final_account_truth "
                "requested_venues=extended,hyperliquid,aster,lighter,paradex "
                "fresh_venues=extended,hyperliquid,aster,lighter position_tol_tao=0.002500\n"
            ),
            promotion_cleanup_strict=True,
            order_path_coverages=[
                order_path_coverage_doc(
                    "paradex",
                    target_place_intents=1,
                    observed_order_actions=["place"],
                    observed_order_statuses=["intent"],
                )
            ],
        )

        self.assertEqual(report["terminal_flatness_gate_status"], "HOLD")
        self.assertIn(
            "terminal_cleanup_account_truth_blocked",
            report["terminal_flatness_gate_reasons"],
        )
        self.assertFalse(
            report["terminal_cleanup"]["final_account_truth_direct_venue_audit_superseded"]
        )

    def test_promotion_cleanup_strict_holds_final_direct_audit_supersession_when_incomplete_residual_present(self):
        report = self.evaluate(
            audit_doc(),
            live_stderr=(
                "[runner] canary_exit_position_flatten_cleanup account_refresh_not_fresh "
                "phase=final_check venue=paradex account_ok=true account_available=false\n"
                "[runner] canary_exit_position_flatten_cleanup account_refresh_applied "
                "phase=final_check requested_venues=extended,hyperliquid,aster,lighter,paradex "
                "fresh_venues=extended,hyperliquid,aster,lighter position_changed=false "
                "remaining_venues=\n"
                "[runner] canary_exit_position_flatten_cleanup blocked_final_account_truth "
                "requested_venues=extended,hyperliquid,aster,lighter,paradex "
                "fresh_venues=extended,hyperliquid,aster,lighter position_tol_tao=0.002500\n"
                "[runner] canary_exit_position_flatten_cleanup incomplete "
                "residual_venues=paradex position_tol_tao=0.002500\n"
            ),
            promotion_cleanup_strict=True,
            order_path_coverages=[
                order_path_coverage_doc(
                    "paradex",
                    target_place_intents=1,
                    observed_order_actions=["place"],
                    observed_order_statuses=["intent"],
                )
            ],
        )

        self.assertEqual(report["terminal_flatness_gate_status"], "HOLD")
        self.assertIn(
            "terminal_cleanup_incomplete_residual_present",
            report["terminal_flatness_gate_reasons"],
        )
        self.assertIn(
            "terminal_cleanup_account_truth_blocked",
            report["terminal_flatness_gate_reasons"],
        )
        self.assertFalse(
            report["terminal_cleanup"]["final_account_truth_direct_venue_audit_superseded"]
        )

    def test_promotion_cleanup_strict_holds_no_exposure_final_account_truth_when_direct_audit_dirty(self):
        data = audit_doc(
            ok=False,
            results=[
                venue_result("hyperliquid"),
                venue_result("lighter"),
                venue_result("extended"),
                venue_result("aster"),
                venue_result("paradex", open_orders=1, ok=False),
            ],
        )
        report = self.evaluate(
            data,
            live_stderr=(
                "[runner] canary_exit_position_flatten_cleanup account_refresh_not_fresh "
                "phase=final_check venue=paradex account_ok=true account_available=false\n"
                "[runner] canary_exit_position_flatten_cleanup account_refresh_applied "
                "phase=final_check requested_venues=extended,hyperliquid,aster,lighter,paradex "
                "fresh_venues=extended,hyperliquid,aster,lighter position_changed=false "
                "remaining_venues=\n"
                "[runner] canary_exit_position_flatten_cleanup blocked_final_account_truth "
                "requested_venues=extended,hyperliquid,aster,lighter,paradex "
                "fresh_venues=extended,hyperliquid,aster,lighter position_tol_tao=0.002500\n"
            ),
            promotion_cleanup_strict=True,
            order_path_coverages=[order_path_coverage_doc("paradex")],
        )

        self.assertEqual(report["terminal_flatness_gate_status"], "HOLD")
        self.assertIn("paradex:open_orders_present", report["terminal_flatness_gate_reasons"])
        self.assertIn(
            "terminal_cleanup_account_truth_blocked",
            report["terminal_flatness_gate_reasons"],
        )

    def test_promotion_cleanup_strict_holds_on_incomplete_residual(self):
        report = self.evaluate(
            audit_doc(),
            live_stderr=(
                "[runner] canary_exit_position_flatten_cleanup incomplete "
                "residual_venues=hyperliquid position_tol_tao=0.000100\n"
            ),
            promotion_cleanup_strict=True,
        )

        self.assertEqual(report["terminal_flatness_gate_status"], "HOLD")
        self.assertIn(
            "terminal_cleanup_incomplete_residual_present",
            report["terminal_flatness_gate_reasons"],
        )
        self.assertEqual(report["terminal_cleanup"]["incomplete_residual_count"], 1)

    def test_promotion_cleanup_strict_requires_cleanup_log(self):
        report = self.evaluate(audit_doc(), promotion_cleanup_strict=True)

        self.assertEqual(report["terminal_flatness_gate_status"], "HOLD")
        self.assertIn("terminal_cleanup_log_missing", report["terminal_flatness_gate_reasons"])

    def test_promotion_cleanup_strict_requires_same_run_exit_code_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / f"{RUN_TOKEN}_terminal_audit.json"
            canary_manifest = root / f"{RUN_TOKEN}_manifest.json"
            live_stderr = root / f"{RUN_TOKEN}_live_stderr.log"
            write_json(audit, audit_doc())
            write_json(canary_manifest, manifest_doc())
            live_stderr.write_text(
                "[runner] canary_exit_cancel_all_cleanup clean attempt=1 "
                "tracked_open_orders=0 clean_state_sweep_dispatched=true\n",
                encoding="utf-8",
            )

            report = gate.evaluate_terminal_flatness(
                venue_audit_path=audit,
                expected_venues=EXPECTED,
                position_tol_base=0.0025,
                canary_manifest_path=canary_manifest,
                run_token=RUN_TOKEN,
                audit_captured_after_run=True,
                audit_captured_after_terminal_cleanup=True,
                live_stderr_path=live_stderr,
                promotion_cleanup_strict=True,
            )

        self.assertEqual(report["terminal_flatness_gate_status"], "HOLD")
        self.assertIn("run_completion_exit_code_missing", report["terminal_flatness_gate_reasons"])
        self.assertEqual(report["closeout_status"]["run_completion_status"], "HOLD")
        self.assertFalse(report["closeout_status"]["promotion_ready"])

    def test_promotion_cleanup_strict_holds_on_nonzero_same_run_exit_code(self):
        report = self.evaluate(
            audit_doc(),
            live_stderr=(
                "[runner] canary_exit_cancel_all_cleanup clean attempt=1 "
                "tracked_open_orders=0 clean_state_sweep_dispatched=true\n"
            ),
            live_exit_code="-1",
            promotion_cleanup_strict=True,
        )

        self.assertEqual(report["terminal_flatness_gate_status"], "HOLD")
        self.assertIn("run_completion_exit_code_nonzero", report["terminal_flatness_gate_reasons"])
        self.assertEqual(report["run_completion"]["live_exit_code"], -1)
        self.assertFalse(report["run_completion"]["live_completed_cleanly"])
        self.assertFalse(report["closeout_status"]["promotion_ready"])

    def test_promotion_cleanup_strict_holds_without_same_run_end_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / f"{RUN_TOKEN}_terminal_audit.json"
            canary_manifest = root / f"{RUN_TOKEN}_manifest.json"
            live_stderr = root / f"{RUN_TOKEN}_live_stderr.log"
            live_exit_code = root / f"{RUN_TOKEN}_live_exit_code.txt"
            summary = root / f"{RUN_TOKEN}_summary.json"
            write_json(audit, audit_doc())
            write_json(canary_manifest, manifest_doc())
            live_stderr.write_text(
                "[runner] canary_exit_cancel_all_cleanup clean attempt=1 "
                "tracked_open_orders=0 clean_state_sweep_dispatched=true\n",
                encoding="utf-8",
            )
            live_exit_code.write_text("0\n", encoding="utf-8")
            write_json(
                summary,
                {
                    "execution_mode": "live",
                    "trade_mode": "live",
                    "ticks_run": 2400,
                    "run_duration_ms": 600000,
                },
            )

            report = gate.evaluate_terminal_flatness(
                venue_audit_path=audit,
                expected_venues=EXPECTED,
                position_tol_base=0.0025,
                canary_manifest_path=canary_manifest,
                run_token=RUN_TOKEN,
                audit_captured_after_run=True,
                audit_captured_after_terminal_cleanup=True,
                live_stderr_path=live_stderr,
                live_exit_code_path=live_exit_code,
                live_summary_path=summary,
                promotion_cleanup_strict=True,
            )

        self.assertEqual(report["terminal_flatness_gate_status"], "HOLD")
        self.assertIn("run_completion_live_end_missing", report["terminal_flatness_gate_reasons"])
        self.assertFalse(report["closeout_status"]["promotion_ready"])

    def test_promotion_cleanup_strict_holds_on_short_cleanup_only_summary(self):
        report = self.evaluate(
            audit_doc(),
            live_stderr=(
                "[runner] canary_exit_cancel_all_cleanup clean attempt=1 "
                "tracked_open_orders=0 clean_state_sweep_dispatched=true\n"
            ),
            live_summary={
                "execution_mode": "live",
                "trade_mode": "live",
                "ticks_run": 20,
                "run_duration_ms": 5000,
            },
            min_ticks_run=2400,
            min_run_duration_ms=600000,
            promotion_cleanup_strict=True,
        )

        self.assertEqual(report["terminal_flatness_gate_status"], "HOLD")
        self.assertIn("run_completion_ticks_short", report["terminal_flatness_gate_reasons"])
        self.assertIn("run_completion_duration_short", report["terminal_flatness_gate_reasons"])
        self.assertEqual(report["run_completion"]["ticks_run"], 20)
        self.assertFalse(report["closeout_status"]["promotion_ready"])

    def test_promotion_cleanup_strict_holds_on_ranked_execution_block(self):
        report = self.evaluate(
            audit_doc(),
            live_stderr=(
                "[runner] tick=752 v2_live_canary_ranked_execution_blocked_request "
                "reason=v2_ranked_non_mm_place\n"
                "[runner] canary_exit_cancel_all_cleanup clean attempt=1 "
                "tracked_open_orders=0 clean_state_sweep_dispatched=true\n"
            ),
            min_ticks_run=2400,
            min_run_duration_ms=600000,
            promotion_cleanup_strict=True,
        )

        self.assertEqual(report["terminal_flatness_gate_status"], "HOLD")
        self.assertIn(
            "run_validation_ranked_execution_blocked_request",
            report["terminal_flatness_gate_reasons"],
        )
        self.assertEqual(report["terminal_cleanup"]["ranked_execution_blocked_request_count"], 1)
        self.assertEqual(
            report["terminal_cleanup"]["ranked_execution_blocked_reasons"],
            ["v2_ranked_non_mm_place"],
        )
        self.assertFalse(report["closeout_status"]["promotion_ready"])

    def test_promotion_cleanup_strict_accepts_completed_same_run_shape(self):
        report = self.evaluate(
            audit_doc(),
            live_stderr=(
                "[runner] canary_exit_cancel_all_cleanup clean attempt=1 "
                "tracked_open_orders=0 clean_state_sweep_dispatched=true\n"
            ),
            min_ticks_run=2400,
            min_run_duration_ms=600000,
            promotion_cleanup_strict=True,
        )

        self.assertEqual(report["terminal_flatness_gate_status"], "PASS")
        self.assertEqual(report["closeout_status"]["run_completion_status"], "PASS")
        self.assertTrue(report["run_completion"]["live_completed_cleanly"])
        self.assertTrue(report["run_completion"]["live_end_utc_found"])
        self.assertTrue(report["closeout_status"]["promotion_ready"])


if __name__ == "__main__":
    unittest.main()
