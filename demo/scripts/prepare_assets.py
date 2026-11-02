"""Export road geometry and crop the supplied paper; never import a model.

Usage: python demo/scripts/prepare_assets.py /path/to/VNICT_0804_camera_ready.pdf
Requires pdftoppm and Pillow for paper crops. Road export uses the standard library.
"""

import json
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "demo/assets"


def export_roads():
    source = ROOT / "custom_environment/data/Graph/DongDa/DongDa.graphml"
    tree = ET.parse(source)
    ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
    keys = {k.attrib["id"]: k.attrib.get("attr.name") for k in tree.findall("g:key", ns)}

    def attrs(element):
        return {keys[d.attrib["key"]]: d.text for d in element.findall("g:data", ns)}

    nodes = []
    positions = {}
    for node in tree.findall(".//g:node", ns):
        a = attrs(node)
        point = [float(a["x"]), float(a["y"])]
        positions[node.attrib["id"]] = point
        nodes.append({"id": node.attrib["id"], "lon": point[0], "lat": point[1]})
    roads, seen = [], set()
    for edge in tree.findall(".//g:edge", ns):
        a = attrs(edge)
        coords = a.get("geometry", "")
        if coords.startswith("LINESTRING"):
            pairs = re.findall(r"(-?[\d.]+)\s+(-?[\d.]+)", coords)
            line = [[float(x), float(y)] for x, y in pairs]
        else:
            line = [positions[edge.attrib["source"]], positions[edge.attrib["target"]]]
        signature = min(str(line), str(line[::-1]))
        if signature not in seen:
            roads.append(line)
            seen.add(signature)
    payload = {
        "source": str(source.relative_to(ROOT)),
        "attribution": "Road network © OpenStreetMap contributors; extracted from repository GraphML.",
        "note": "Current repository graph has 670 nodes; paper reports 639. Not the exact paper experiment snapshot.",
        "nodes": nodes,
        "roads": roads,
    }
    (ASSETS / "roads.json").write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def export_paper(path):
    from PIL import Image

    shutil.copyfile(path, ASSETS / "paper.pdf")
    # Crop coordinates are fractions of rendered A4 page dimensions.
    regions = {4: ("architecture.png", (0.50, 0.080, 0.865, 0.225)),
               5: ("deployment-maps.png", (0.125, 0.250, 0.865, 0.375))}
    with tempfile.TemporaryDirectory() as work:
        for page, (filename, box) in regions.items():
            output = Path(work) / f"page-{page}"
            subprocess.run(["pdftoppm", "-f", str(page), "-l", str(page), "-r", "220", "-singlefile", "-png", str(path), str(output)], check=True)
            with Image.open(output.with_suffix(".png")) as image:
                w, h = image.size
                image.crop(tuple(round(v * (w if i % 2 == 0 else h)) for i, v in enumerate(box))).save(ASSETS / filename)


if __name__ == "__main__":
    ASSETS.mkdir(parents=True, exist_ok=True)
    export_roads()
    if len(sys.argv) > 1:
        export_paper(Path(sys.argv[1]))
    print("Demo assets prepared; no model loaded.")
