import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


BASELINE_COMMIT = "18dd09512288a85e440d3977e32432c3aabc1190"


def stable_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class TestPhase51agLighterInactiveOrderBridgeAudit(unittest.TestCase):
    def test_materializes_hash_only_bridge_when_inactive_order_and_trade_hash_overlap(self):
        script_dir = Path(__file__).resolve().parents[1]
        tool_path = script_dir / "tools" / "phase51ag_lighter_inactive_order_bridge_audit.py"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            request_pack = tmp_path / "request_pack"
            request_pack.mkdir()
            output_root = tmp_path / "runs"

            shared_hash = stable_hash("shared-order-id")
            trade_row = {
                "account_index": 123,
                "ask_account_id": 123,
                "bid_account_id": 456,
                "is_maker_ask": True,
                "ask_id_sha256": shared_hash,
            }
            source_hash = stable_hash(trade_row)

            (request_pack / "manifest.json").write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
            (request_pack / "source_link_request_targets.jsonl").write_text(
                json.dumps({
                    "baseline_commit": BASELINE_COMMIT,
                    "venue_id": "lighter",
                    "canonical_group_id": "target-1",
                    "order_key": "order-key-1",
                    "order_id_hash": shared_hash,
                    "client_order_id_hash": None,
                    "no_live_flag": True,
                    "approved_for_live": False,
                    "live_orders_allowed": False,
                }) + "\n",
                encoding="utf-8",
            )
            (request_pack / "source_link_request_sources.jsonl").write_text(
                json.dumps({
                    "baseline_commit": BASELINE_COMMIT,
                    "venue_id": "lighter",
                    "source_record_sha256": source_hash,
                    "no_live_flag": True,
                    "approved_for_live": False,
                    "live_orders_allowed": False,
                }) + "\n",
                encoding="utf-8",
            )

            inactive_orders = tmp_path / "inactive_orders.sanitized.json"
            inactive_orders.write_text(
                json.dumps({
                    "orders": [{"order_id_sha256": shared_hash, "status": "filled"}],
                }),
                encoding="utf-8",
            )
            trade_source = tmp_path / "trades.sanitized.json"
            trade_source.write_text(json.dumps({"trades": [trade_row]}), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(tool_path),
                    "--request-pack",
                    str(request_pack),
                    "--inactive-orders-json",
                    str(inactive_orders),
                    "--trade-source-json",
                    str(trade_source),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51ag_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"stdout={result.stdout}\nstderr={result.stderr}")

            run_dir = output_root / "phase51ag_test"
            summary = json.loads(
                (run_dir / "phase51ag_lighter_inactive_order_bridge_audit_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["inactive_order_target_match_count"], 1)
            self.assertEqual(summary["bridge_source_link_count"], 1)
            sidecar = [
                json.loads(line)
                for line in (run_dir / "source_links.proposed.sanitized.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(sidecar, [{"canonical_group_id": "target-1", "source_record_sha256": source_hash}])


if __name__ == "__main__":
    unittest.main()
