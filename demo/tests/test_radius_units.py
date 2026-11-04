"""Exercise the actual Python radius function without importing RL/GIS packages."""
import ast
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

SOURCE = Path(__file__).resolve().parents[2] / "custom_environment/helpers.py"
tree = ast.parse(SOURCE.read_text())
constants = {}
for statement in tree.body:
    if isinstance(statement, ast.Assign):
        for target in statement.targets:
            if isinstance(target, ast.Name) and target.id in {"RADIUS_MAX", "capacity_unit"}:
                constants[target.id] = ast.literal_eval(statement.value)
function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "influence_radius")
namespace = {"np": SimpleNamespace(exp=math.exp), **constants}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), namespace)


class RadiusUnitsTest(unittest.TestCase):
    def radius(self, power_mw):
        station = [None, None, {"capability": power_mw}]
        namespace["influence_radius"](station)
        self.assertEqual(station[2]["capability"], power_mw)
        return station[2]["radius"] * 1000

    def test_mw_to_kw_conversion(self):
        self.assertAlmostEqual(self.radius(.06), 514.9955016194, places=6)
        self.assertAlmostEqual(self.radius(.8), 689.9744811276, places=6)
        self.assertAlmostEqual(self.radius(1), 731.05857863, places=6)

    def test_current_capacity_and_bounds(self):
        self.assertGreater(self.radius(.063), self.radius(.06))
        self.assertGreater(self.radius(2), self.radius(1))
        self.assertLessEqual(self.radius(100), constants["RADIUS_MAX"] * 1000)


if __name__ == "__main__":
    unittest.main()
