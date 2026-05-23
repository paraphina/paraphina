import json
import tempfile
import unittest
from pathlib import Path

from tools import v2_telemetry_order_path_coverage_validator as validator


EXPECTED = ["hyperliquid", "lighter", "extended", "aster", "paradex"]
RUN_TOKEN = "v2_paradex_venue_coverage_probe_test"


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def preflight_text() -> str:
    return "\n".join(
        [
            "paraphina_live preflight:",
            "- PASS v2_live_canary_admission approved=true canary_mode=true "
            "exit_cancel_all=true exit_position_flatten=true "
            "venue_coverage_replacements_disabled=true "
            "venue_coverage_probe_approved=true "
            "venue_coverage_probe_venues_present=true "
            "ranked_execution_venues_present=true",
        ]
    )


def summary_doc(**overrides):
    data = {
        "execution_mode": "live",
        "ticks_run": 180,
        "run_duration_ms": 45000,
        "kill_events": 0,
        "would_place_by_purpose": {"Mm": 1},
        "would_cancel_by_purpose": {"unknown": 1},
        "would_replace_by_purpose": {},
    }
    data.update(overrides)
    return data


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
        "position_tol_base": 0.02,
        "max_open_orders": 0,
        "results": [venue_result(venue) for venue in EXPECTED],
        "violations": [],
    }
    data.update(overrides)
    return data


def telemetry_rows(**order_overrides):
    place = {
        "venue_id": "paradex",
        "action": "place",
        "status": "intent",
        "side": "Buy",
        "purpose": "Mm",
        "post_only": True,
        "reduce_only": False,
        "tif": "Gtc",
        "size": 0.01,
    }
    place.update(order_overrides)
    return [
        {"schema_version": 1, "t": 1, "orders": [place], "fills": [], "would_send_orders": []},
        {
            "schema_version": 1,
            "t": 2,
            "orders": [
                {"venue_id": "paradex", "action": "place", "status": "ack", "size": 0.01},
                {"venue_id": "paradex", "action": "cancel", "status": "intent"},
            ],
            "fills": [],
            "would_send_orders": [],
        },
        {
            "schema_version": 1,
            "t": 3,
            "orders": [{"venue_id": "paradex", "action": "cancel", "status": "ack"}],
            "fills": [],
            "would_send_orders": [],
        },
    ]


class TestV2TelemetryOrderPathCoverageValidator(unittest.TestCase):
    def evaluate(
        self,
        *,
        telemetry: list[dict] | None = None,
        preflight: str | None = None,
        summary: dict | None = None,
        audit: dict | None = None,
        audit_captured_after_run: bool = True,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / RUN_TOKEN
            root.mkdir()
            telemetry_path = root / "telemetry.jsonl"
            preflight_path = root / "preflight.out"
            summary_path = root / "summary.json"
            audit_path = root / "direct_venue_audit_post.json"
            write_jsonl(telemetry_path, telemetry if telemetry is not None else telemetry_rows())
            preflight_path.write_text(preflight if preflight is not None else preflight_text(), encoding="utf-8")
            write_json(summary_path, summary if summary is not None else summary_doc())
            write_json(audit_path, audit if audit is not None else audit_doc())
            return validator.validate_order_path_coverage(
                telemetry_path=telemetry_path,
                preflight_path=preflight_path,
                summary_path=summary_path,
                venue_audit_path=audit_path,
                target_venue="paradex",
                expected_venues=EXPECTED,
                position_tol_base=0.02,
                max_order_size=0.01,
                run_token=RUN_TOKEN,
                audit_captured_after_run=audit_captured_after_run,
            )

    def test_passes_telemetry_order_path_without_promotion_claims(self):
        report = self.evaluate()

        self.assertEqual(report["validation_status"], "PASS")
        self.assertEqual(report["coverage"]["target_place_intents"], 1)
        self.assertEqual(report["coverage"]["target_place_acks"], 1)
        self.assertEqual(report["coverage"]["target_cancel_intents"], 1)
        self.assertEqual(report["coverage"]["target_cancel_acks"], 1)
        self.assertFalse(report["governance"]["approved_for_promotion"])
        self.assertFalse(report["governance"]["approved_for_live"])
        self.assertFalse(report["governance"]["approved_for_capital_escalation"])
        self.assertFalse(report["governance"]["blocker_cleared"])
        self.assertFalse(report["governance"]["pressure_complete_claim"])
        self.assertFalse(report["governance"]["v2_authority_admission"])

    def test_replace_event_holds(self):
        rows = telemetry_rows()
        rows.append({"schema_version": 1, "t": 4, "orders": [{"venue_id": "paradex", "action": "replace", "status": "intent"}]})

        report = self.evaluate(telemetry=rows)

        self.assertEqual(report["validation_status"], "HOLD")
        self.assertIn("target_replace_events_present", report["validation_reasons"])

    def test_non_post_only_place_holds(self):
        report = self.evaluate(telemetry=telemetry_rows(post_only=False))

        self.assertEqual(report["validation_status"], "HOLD")
        self.assertIn("target_place_policy_not_strict_post_only_gtc", report["validation_reasons"])

    def test_missing_ack_holds(self):
        rows = [{"schema_version": 1, "t": 1, "orders": [telemetry_rows()[0]["orders"][0]], "fills": []}]

        report = self.evaluate(telemetry=rows)

        self.assertEqual(report["validation_status"], "HOLD")
        self.assertIn("target_place_ack_missing", report["validation_reasons"])
        self.assertIn("target_cancel_intent_missing", report["validation_reasons"])

    def test_nonflat_venue_audit_holds(self):
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

        report = self.evaluate(audit=data)

        self.assertEqual(report["validation_status"], "HOLD")
        self.assertIn("paradex:open_orders_present", report["validation_reasons"])

    def test_missing_no_replace_preflight_gate_holds(self):
        report = self.evaluate(preflight=preflight_text().replace("venue_coverage_replacements_disabled=true ", ""))

        self.assertEqual(report["validation_status"], "HOLD")
        self.assertTrue(
            any(reason.startswith("preflight_missing:venue_coverage_replacements_disabled=true") for reason in report["validation_reasons"])
        )

    def test_sensitive_key_holds_without_exposing_value(self):
        rows = telemetry_rows()
        rows[0]["orders"][0]["client_order_id"] = "raw-local-id"

        report = self.evaluate(telemetry=rows)

        self.assertEqual(report["validation_status"], "HOLD")
        self.assertIn("telemetry_sensitive_key_present_line:1", report["validation_reasons"])

    def test_after_run_confirmation_required(self):
        report = self.evaluate(audit_captured_after_run=False)

        self.assertEqual(report["validation_status"], "HOLD")
        self.assertIn("run_binding_after_run_confirmation_missing", report["validation_reasons"])


if __name__ == "__main__":
    unittest.main()
