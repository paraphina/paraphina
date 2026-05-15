import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = REPO_ROOT / "tools" / "phase51ao_forward_refresh_remaining_merge.py"


def row(target_type="native_role", venue_id="extended", order_key="ok-1"):
    return {
        "target_type": target_type,
        "venue_id": venue_id,
        "canonical_group_id": "cg-1",
        "order_key": order_key,
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
        "isTaker": False,
    }


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")


class Phase51AoForwardRefreshRemainingMergeTests(unittest.TestCase):
    def run_tool(self, base, remaining, output):
        return subprocess.run(
            [
                sys.executable,
                str(TOOL),
                "--base",
                str(base),
                "--remaining",
                str(remaining),
                "--output",
                str(output),
            ],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_merges_without_mutating_base(self):
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            base = temp_path / "forward_refresh.jsonl"
            remaining = temp_path / "forward_refresh.remaining.jsonl"
            output = temp_path / "forward_refresh.merged.jsonl"
            base_row = row(order_key="ok-base")
            remaining_row = row(venue_id="paradex", order_key="ok-remaining")
            write_jsonl(base, [base_row])
            original_base = base.read_text(encoding="utf-8")
            write_jsonl(remaining, [remaining_row])

            result = self.run_tool(base, remaining, output)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(base.read_text(encoding="utf-8"), original_base)
            merged = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(merged), 2)

    def test_rejects_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            base = temp_path / "forward_refresh.jsonl"
            remaining = temp_path / "forward_refresh.remaining.jsonl"
            output = temp_path / "forward_refresh.merged.jsonl"
            duplicate = row(order_key="ok-dup")
            write_jsonl(base, [duplicate])
            write_jsonl(remaining, [duplicate])

            result = self.run_tool(base, remaining, output)

            self.assertEqual(result.returncode, 2)
            self.assertIn("duplicate row key", result.stderr)
            self.assertFalse(output.exists())

    def test_rejects_forbidden_raw_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            base = temp_path / "forward_refresh.jsonl"
            remaining = temp_path / "forward_refresh.remaining.jsonl"
            output = temp_path / "forward_refresh.merged.jsonl"
            unsafe = row(order_key="ok-raw")
            unsafe["order_id"] = "raw-order-id"
            write_jsonl(base, [])
            write_jsonl(remaining, [unsafe])

            result = self.run_tool(base, remaining, output)

            self.assertEqual(result.returncode, 2)
            self.assertIn("forbidden field order_id", result.stderr)
            self.assertFalse(output.exists())

    def test_rejects_unsafe_true_flags(self):
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            base = temp_path / "forward_refresh.jsonl"
            remaining = temp_path / "forward_refresh.remaining.jsonl"
            output = temp_path / "forward_refresh.merged.jsonl"
            unsafe = row(order_key="ok-unsafe")
            unsafe["approved_for_live"] = True
            write_jsonl(base, [])
            write_jsonl(remaining, [unsafe])

            result = self.run_tool(base, remaining, output)

            self.assertEqual(result.returncode, 2)
            self.assertIn("unsafe true flag approved_for_live", result.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
