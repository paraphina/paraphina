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


class TestPhase51ahLighterExplorerBridgeDiagnostic(unittest.TestCase):
    def test_materializes_source_link_only_for_request_source_and_unique_target_hash(self):
        script_dir = Path(__file__).resolve().parents[1]
        tool_path = script_dir / "tools" / "phase51ah_lighter_explorer_bridge_diagnostic.py"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            request_pack = tmp_path / "request_pack"
            request_pack.mkdir()
            output_root = tmp_path / "runs"

            raw_order_id = "281476612587355"
            target_hash = stable_hash(raw_order_id)
            explorer_trade_row = {
                "trade_id": 100,
                "ask_id": raw_order_id,
                "bid_id": "562948334068259",
                "ask_account_id": 718392,
                "bid_account_id": 317068,
                "is_maker_ask": False,
            }
            source_hash = stable_hash(explorer_trade_row)

            (request_pack / "source_link_request_targets.jsonl").write_text(
                json.dumps({
                    "baseline_commit": BASELINE_COMMIT,
                    "venue_id": "lighter",
                    "canonical_group_id": "target-1",
                    "order_key": "order-key-1",
                    "order_id_hash": target_hash,
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
            (request_pack / "candidate_manifest_with_empty_sidecar.json").write_text(
                json.dumps({"schema_version": 1, "sources": [], "source_links": []}),
                encoding="utf-8",
            )

            explorer_logs = tmp_path / "explorer_logs.json"
            explorer_logs.write_text(json.dumps({"logs": [{"trades": [explorer_trade_row]}]}), encoding="utf-8")
            tx_details = tmp_path / "tx_details.json"
            tx_details.write_text(
                json.dumps({
                    "info": json.dumps({
                        "ClientOrderIndex": raw_order_id,
                        "Sig": "secret-signature-value",
                    }),
                    "event_info": json.dumps({"to": {"i": raw_order_id}}),
                    "l1_address": "0xabc123",
                }),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(tool_path),
                    "--request-pack",
                    str(request_pack),
                    "--explorer-logs-json",
                    str(explorer_logs),
                    "--tx-json",
                    str(tx_details),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51ah_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"stdout={result.stdout}\nstderr={result.stderr}")

            run_dir = output_root / "phase51ah_test"
            summary = json.loads((run_dir / "phase51ah_lighter_explorer_bridge_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["materializable_source_link_count"], 1)
            self.assertEqual(summary["bridge_status_counts"]["SOURCE_LINK_PROPOSED"], 1)

            sidecar = [
                json.loads(line)
                for line in (run_dir / "source_links.proposed.sanitized.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(sidecar, [{"canonical_group_id": "target-1", "source_record_sha256": source_hash}])

            sanitized_logs = (run_dir / "source_snapshots" / "explorer_logs.sanitized.json").read_text(encoding="utf-8")
            self.assertNotIn('"ask_id"', sanitized_logs)
            self.assertIn('"ask_id_sha256"', sanitized_logs)
            sanitized_txs = (run_dir / "source_snapshots" / "tx_details.sanitized.json").read_text(encoding="utf-8")
            self.assertNotIn("secret-signature-value", sanitized_txs)
            self.assertNotIn("0xabc123", sanitized_txs)
            self.assertNotIn("ClientOrderIndex", sanitized_txs)
            self.assertIn('"info_sha256"', sanitized_txs)
            self.assertIn('"l1_address_sha256"', sanitized_txs)

            artifact_index = json.loads((run_dir / "evidence_pack" / "artifact_index.json").read_text(encoding="utf-8"))
            artifact_paths = {row["path"] for row in artifact_index["artifacts"]}
            self.assertIn("phase51ah_lighter_explorer_bridge_summary.json", artifact_paths)
            self.assertIn("source_links.proposed.sanitized.jsonl", artifact_paths)


if __name__ == "__main__":
    unittest.main()
