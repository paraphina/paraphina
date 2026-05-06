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


class TestPhase51ajForwardPrivateStreamSource(unittest.TestCase):
    def _write_request_pack(self, request_pack: Path, targets: list[dict], sources: list[dict] | None = None) -> None:
        request_pack.mkdir()
        (request_pack / "manifest.json").write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
        (request_pack / "source_link_request_targets.jsonl").write_text(
            "\n".join(json.dumps(row) for row in targets) + "\n",
            encoding="utf-8",
        )
        (request_pack / "source_link_request_sources.jsonl").write_text(
            "\n".join(json.dumps(row) for row in (sources or [])) + ("\n" if sources else ""),
            encoding="utf-8",
        )

    def test_materializes_direct_private_source_rows_and_request_hash_sidecar(self):
        script_dir = Path(__file__).resolve().parents[1]
        tool_path = script_dir / "tools" / "phase51aj_forward_private_stream_source.py"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            request_pack = tmp_path / "request_pack"
            output_root = tmp_path / "runs"

            extended_external_id = "extended-client-direct-001"
            paradex_client_id = "paradex-client-direct-001"
            lighter_client_id = "lighter-client-direct-001"
            aster_client_id = "aster-client-direct-001"
            extended_row = {
                "channel": "account",
                "type": "TRADE",
                "orderId": "9223372036854775808",
                "externalOrderId": extended_external_id,
                "isTaker": False,
                "market": "ETH-USD",
            }
            paradex_row = {
                "jsonrpc": "2.0",
                "method": "subscription",
                "params": {
                    "channel": "fills.ETH-USD-PERP",
                    "data": {
                        "client_id": paradex_client_id,
                        "order_id": "1681462103821101699438490000",
                        "liquidity": "MAKER",
                        "market": "ETH-USD-PERP",
                    },
                },
            }
            paradex_inner = paradex_row["params"]["data"]
            lighter_row = {
                "ask_client_id": lighter_client_id,
                "bid_client_id": "other-lighter-client",
                "account_index": 718392,
                "ask_account_id": 718392,
                "bid_account_id": 317068,
                "is_maker_ask": True,
            }
            aster_row = {
                "e": "ORDER_TRADE_UPDATE",
                "clientOrderId": aster_client_id,
                "o": {
                    "m": True,
                    "l": "0.010",
                },
            }

            targets = [
                {
                    "baseline_commit": BASELINE_COMMIT,
                    "venue_id": "extended",
                    "canonical_group_id": "extended-target",
                    "order_key": "extended-order-key",
                    "client_order_id_hash": stable_hash(extended_external_id),
                    "order_id_hash": None,
                    "no_live_flag": True,
                    "approved_for_live": False,
                    "live_orders_allowed": False,
                },
                {
                    "baseline_commit": BASELINE_COMMIT,
                    "venue_id": "paradex",
                    "canonical_group_id": "paradex-target",
                    "order_key": "paradex-order-key",
                    "client_order_id_hash": stable_hash(paradex_client_id),
                    "order_id_hash": None,
                    "no_live_flag": True,
                    "approved_for_live": False,
                    "live_orders_allowed": False,
                },
                {
                    "baseline_commit": BASELINE_COMMIT,
                    "venue_id": "lighter",
                    "canonical_group_id": "lighter-target",
                    "order_key": "lighter-order-key",
                    "client_order_id_hash": stable_hash(lighter_client_id),
                    "order_id_hash": None,
                    "no_live_flag": True,
                    "approved_for_live": False,
                    "live_orders_allowed": False,
                },
                {
                    "baseline_commit": BASELINE_COMMIT,
                    "venue_id": "aster",
                    "canonical_group_id": "aster-target",
                    "order_key": "aster-order-key",
                    "client_order_id_hash": stable_hash(aster_client_id),
                    "order_id_hash": None,
                    "no_live_flag": True,
                    "approved_for_live": False,
                    "live_orders_allowed": False,
                },
            ]
            sources = [
                {
                    "baseline_commit": BASELINE_COMMIT,
                    "venue_id": "extended",
                    "source_record_sha256": stable_hash(extended_row),
                    "no_live_flag": True,
                    "approved_for_live": False,
                    "live_orders_allowed": False,
                },
                {
                    "baseline_commit": BASELINE_COMMIT,
                    "venue_id": "paradex",
                    "source_record_sha256": stable_hash(paradex_inner),
                    "no_live_flag": True,
                    "approved_for_live": False,
                    "live_orders_allowed": False,
                },
            ]
            self._write_request_pack(request_pack, targets, sources)

            extended_path = tmp_path / "extended.json"
            extended_path.write_text(json.dumps({"data": [extended_row]}), encoding="utf-8")
            paradex_path = tmp_path / "paradex.jsonl"
            paradex_path.write_text(json.dumps(paradex_row) + "\n", encoding="utf-8")
            lighter_path = tmp_path / "lighter.json"
            lighter_path.write_text(json.dumps([lighter_row]), encoding="utf-8")
            aster_path = tmp_path / "aster.json"
            aster_path.write_text(json.dumps({"events": [aster_row]}), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(tool_path),
                    "--request-pack",
                    str(request_pack),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51aj_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                    "--source-json",
                    f"extended={extended_path}",
                    "--source-json",
                    f"paradex={paradex_path}",
                    "--source-json",
                    f"lighter={lighter_path}",
                    "--source-json",
                    f"aster={aster_path}",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"stdout={result.stdout}\nstderr={result.stderr}")

            run_dir = output_root / "phase51aj_test"
            summary = json.loads((run_dir / "phase51aj_forward_private_stream_source_summary.json").read_text())
            self.assertEqual(summary["direct_target_linked_source_row_count"], 4)
            self.assertEqual(summary["source_link_count"], 2)
            self.assertEqual(summary["request_source_hash_overlap_count"], 2)

            source_rows = [
                json.loads(line)
                for line in (run_dir / "source_snapshots" / "phase51aj_forward_private_stream_source_rows.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            self.assertEqual({row["canonical_group_id"] for row in source_rows}, {
                "extended-target",
                "paradex-target",
                "lighter-target",
                "aster-target",
            })
            self.assertIn({"isTaker": False}, [{k: row[k]} for row in source_rows for k in row if k == "isTaker"])
            self.assertIn({"liquidity": "MAKER"}, [{k: row[k]} for row in source_rows for k in row if k == "liquidity"])

            sidecar = [
                json.loads(line)
                for line in (run_dir / "source_links.proposed.sanitized.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual({row["canonical_group_id"] for row in sidecar}, {"extended-target", "paradex-target"})
            output_text = (run_dir / "source_snapshots" / "phase51aj_forward_private_stream_source_rows.jsonl").read_text(
                encoding="utf-8"
            )
            for raw_value in (extended_external_id, paradex_client_id, lighter_client_id, aster_client_id):
                self.assertNotIn(raw_value, output_text)

    def test_rejects_unsafe_or_secret_output_sources(self):
        script_dir = Path(__file__).resolve().parents[1]
        tool_path = script_dir / "tools" / "phase51aj_forward_private_stream_source.py"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            request_pack = tmp_path / "request_pack"
            self._write_request_pack(request_pack, [], [])
            source_path = tmp_path / "bad.json"
            source_path.write_text(
                json.dumps({"data": [{"externalOrderId": "raw", "isTaker": False, "api_key": "secret"}]}),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(tool_path),
                    "--request-pack",
                    str(request_pack),
                    "--output-root",
                    str(tmp_path / "runs"),
                    "--run-id",
                    "phase51aj_bad",
                    "--source-json",
                    f"extended={source_path}",
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("secret-shaped", result.stderr)


if __name__ == "__main__":
    unittest.main()
