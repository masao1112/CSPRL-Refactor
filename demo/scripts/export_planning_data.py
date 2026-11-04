"""Export data-only planning inputs. No model or RL dependency is imported."""
import ast
import csv
import json
import math
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "demo/assets"


class DataOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError(f"Only primitive plan data is supported: {module}.{name}")


def read_csv(name):
    with (ROOT / "custom_environment/data/hanoi_citywide" / name).open() as file:
        return list(csv.DictReader(file))


def export():
    folder = ROOT / "custom_environment/data/Graph/DongDa"
    nodes = ast.literal_eval((folder / "nodes_extended_DongDa.txt").read_text())
    with (folder / "new_existingplan_DongDa.pkl").open("rb") as file:
        plan = DataOnlyUnpickler(file).load()
    assert len(plan) == 5
    settings = {}
    names = {"CHARGING_POWER", "INSTALL_FEE", "BUDGET", "COST_UNIT", "K", "RADIUS_MAX", "BATTERY", "alpha", "DEMAND_ETA", "DEMAND_BETA", "RELOCATION_FACTOR", "capacity_unit", "evs_parking_area"}
    for statement in ast.parse((ROOT / "custom_environment/helpers.py").read_text()).body:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    value = statement.value.args[0] if isinstance(statement.value, ast.Call) else statement.value
                    settings[target.id] = ast.literal_eval(value)
    all_buses = [r for r in read_csv("bus.csv") if float(r["vn_kv"]) == 22]
    loads, lines, trafos = read_csv("load.csv"), read_csv("line.csv"), read_csv("trafo.csv")
    exported_nodes, selected = [], set()
    for node_id, attrs in nodes:
        bus = min(all_buses, key=lambda b: ((float(b["x"]) - attrs["x"]) * math.cos(math.radians(attrs["y"]))) ** 2 + (float(b["y"]) - attrs["y"]) ** 2)
        selected.add(bus["index"])
        exported_nodes.append({"id": str(node_id), "lon": attrs["x"], "lat": attrs["y"],
                               "demand": max(0, min(1, attrs["demand"])), "population": attrs["pop"],
                               "landPrice": attrs["land_price"], "busId": bus["index"]})
    buses = []
    for bus in all_buses:
        if bus["index"] not in selected:
            continue
        bus_id = bus["index"]
        connected_lines = [r for r in lines if bus_id in (r["from_bus"], r["to_bus"])]
        transformer_mva = sum(float(r["sn_mva"]) for r in trafos if bus_id in (r["lv_bus"], r["hv_bus"]))
        line_mva = min((1.732 * 22 * float(r["max_i_ka"]) for r in connected_lines), default=9999)
        rated_mva = min(line_mva, transformer_mva or 9999)
        fallback = rated_mva > 9000
        if fallback:
            rated_mva = 10
        base_load = sum(float(r["p_mw"]) * float(r.get("scaling") or 1) for r in loads if r["bus"] == bus_id)
        buses.append({"id": bus_id, "name": bus["name"], "lon": float(bus["x"]), "lat": float(bus["y"]),
                      "baseLoadMW": base_load, "ratedMVA": rated_mva,
                      "availableMW": max(0, rated_mva * .8 - base_load), "fallback": fallback})
    payload = {"district": "Đống Đa, Hà Nội", "nodes": exported_nodes, "buses": buses,
               "initialStations": [{"nodeId": str(s[0][0]), "counts": s[1]} for s in plan],
               "settings": settings,
               "provenance": {"nodes": "custom_environment/data/Graph/DongDa/nodes_extended_DongDa.txt",
                              "initialPlan": "custom_environment/data/Graph/DongDa/new_existingplan_DongDa.pkl",
                              "grid": "custom_environment/data/hanoi_citywide/*.csv",
                              "gridCapacity": "min(line rating, transformer rating) × 0.8 − CSV base load; no power-flow solver",
                              "runs": "illustrative rule-based trajectories; no model/checkpoint loaded"}}
    (ASSETS / "planning-data.json").write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    print(f"Exported {len(nodes)} demand nodes, {len(plan)} initial stations, {len(buses)} buses and {len(settings['CHARGING_POWER'])} charger levels.")


if __name__ == "__main__":
    export()
