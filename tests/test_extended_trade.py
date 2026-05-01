import importlib.util
import sys
import unittest
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "tools" / "extended_trade.py"
    tools_dir = module_path.parent
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    spec = importlib.util.spec_from_file_location("extended_trade_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestExtendedTradeBridge(unittest.TestCase):
    def setUp(self):
        self.mod = _load_module()

    def test_normalize_open_order_derives_remaining_size_from_qty_and_filled_qty(self):
        order = SimpleNamespace(
            id=1784963886257016832,
            external_id="d0_mm_v4_buy",
            side="BUY",
            price=Decimal("2500.10"),
            qty=Decimal("0.020"),
            filled_qty=Decimal("0.005"),
        )

        normalized = self.mod.normalize_open_order(order)

        self.assertEqual(normalized["order_id"], "1784963886257016832")
        self.assertEqual(normalized["client_order_id"], "d0_mm_v4_buy")
        self.assertEqual(normalized["side"], "BUY")
        self.assertAlmostEqual(normalized["price"], 2500.10)
        self.assertAlmostEqual(normalized["size"], 0.015)

    def test_normalize_open_order_prefers_remaining_size_when_present(self):
        order = SimpleNamespace(
            id=1784963886257016833,
            external_id="d1_mm_v4_sell",
            side="SELL",
            limit_price=Decimal("2501.20"),
            qty=Decimal("0.020"),
            filled_qty=Decimal("0.010"),
            remaining_size=Decimal("0.007"),
        )

        normalized = self.mod.normalize_open_order(order)

        self.assertEqual(normalized["side"], "SELL")
        self.assertAlmostEqual(normalized["price"], 2501.20)
        self.assertAlmostEqual(normalized["size"], 0.007)

    def test_normalize_open_order_clamps_overfilled_size_to_zero(self):
        order = SimpleNamespace(
            id=1784963886257016834,
            external_id="d2_mm_v4_buy",
            side="BUY",
            price=Decimal("2500.10"),
            qty=Decimal("0.010"),
            filled_qty=Decimal("0.011"),
        )

        normalized = self.mod.normalize_open_order(order)

        self.assertEqual(normalized["size"], 0.0)


if __name__ == "__main__":
    unittest.main()
