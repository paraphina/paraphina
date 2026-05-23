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


class TestV2LiveCanaryTerminalFlatnessGate(unittest.TestCase):
    def evaluate(
        self,
        data: dict,
        *,
        manifest: dict | None = None,
        audit_captured_after_run: bool = True,
        audit_captured_after_terminal_cleanup: bool = True,
        run_token: str = RUN_TOKEN,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / f"{RUN_TOKEN}_terminal_audit.json"
            canary_manifest = root / f"{RUN_TOKEN}_manifest.json"
            write_json(audit, data)
            write_json(canary_manifest, manifest if manifest is not None else manifest_doc())
            return gate.evaluate_terminal_flatness(
                venue_audit_path=audit,
                expected_venues=EXPECTED,
                position_tol_base=0.0025,
                canary_manifest_path=canary_manifest,
                run_token=run_token,
                audit_captured_after_run=audit_captured_after_run,
                audit_captured_after_terminal_cleanup=audit_captured_after_terminal_cleanup,
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


if __name__ == "__main__":
    unittest.main()
