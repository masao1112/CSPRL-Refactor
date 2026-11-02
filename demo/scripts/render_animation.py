"""Render an illustrative scientific GIF offline; never load a model.

Requires Node.js, Matplotlib and Pillow. The webpage itself has no dependencies.
Copies the five source maps byte-for-byte from res/ into its static asset folder.
"""
import io
import json
import re
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "demo/assets"
FIGURES = ["map_DongDa_existingplan_DongDa.png", "map_DongDa_G_Benefit.png",
           "map_DongDa_G_Demand.png", "map_DongDa_GA.png", "map_DongDa_RL_102353.png"]


def main():
    target = ASSETS / "figures"
    target.mkdir(exist_ok=True)
    for filename in FIGURES:
        shutil.copyfile(ROOT / "res" / filename, target / filename)

    # Run the deterministic illustration generator once, at asset-build time.
    # No environment, training library, checkpoint or pickle is imported.
    script = """
      import {readFileSync} from 'node:fs';
      import {projectGraph,generateRun} from './demo/simulation.mjs';
      const graph=projectGraph(JSON.parse(readFileSync('./demo/assets/roads.json','utf8')));
      console.log(JSON.stringify({graph,run:generateRun(graph.nodes)}));
    """
    data = json.loads(subprocess.run(["node", "--input-type=module", "-e", script], cwd=ROOT,
                                    check=True, capture_output=True, text=True).stdout)
    paths = []
    for path in data["graph"]["paths"].split("M")[1:]:
        coords = list(map(float, re.findall(r"[-+]?\d*\.?\d+", path)))
        paths.append(list(zip(coords[::2], coords[1::2])))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig = plt.figure(figsize=(12, 6.6), dpi=100, facecolor="white")
    ax = fig.add_axes([.015, .15, .62, .72])
    ax.add_collection(LineCollection(paths, colors="#cdd0d3", linewidths=.65))
    ax.set(xlim=(35, 725), ylim=(475, 35))
    ax.set_aspect("equal")
    ax.axis("off")
    initial = ax.scatter([], [], s=[], c="#6c747c", edgecolors="white", linewidths=.8, zorder=4)
    added = ax.scatter([], [], s=[], c="#d62828", edgecolors="white", linewidths=.8, zorder=5)
    selected = ax.scatter([], [], s=330, facecolors="none", edgecolors="#366f9e", linewidths=1.8, zorder=6)
    fig.text(.045, .934, "Mở rộng kế hoạch trạm sạc", fontsize=20, color="#30353b")
    fig.text(.045, .895, "MINH HỌA GIẢ LẬP · KHÔNG NẠP MÔ HÌNH", fontsize=8.5, color="#7c838c")
    fig.text(.665, .80, "1   Quan sát toàn mạng", fontsize=13, color="#333a42")
    fig.text(.692, .758, "Nhu cầu · hạ tầng · ngân sách", fontsize=10, color="#7a828c")
    fig.text(.665, .65, "2   Chọn macro-action", fontsize=13, color="#333a42")
    action_text = fig.text(.692, .608, "", fontsize=11, color="#356b9b")
    fig.text(.665, .50, "3   Heuristic chọn nút", fontsize=13, color="#333a42")
    node_text = fig.text(.692, .458, "", fontsize=10, color="#7a828c")
    fig.text(.665, .35, "4   Cập nhật kế hoạch", fontsize=13, color="#333a42")
    count_text = fig.text(.692, .308, "", fontsize=10, color="#7a828c")
    for y in [.70, .55, .40]:
        fig.text(.67, y, "↓", color="#b7bec6", fontsize=15)
    step_text = fig.text(.045, .115, "", fontsize=10, color="#56616c")
    fig.text(.045, .055, "●  Trạm mẫu ban đầu", color="#6c747c", fontsize=9)
    fig.text(.245, .055, "●  Trạm bổ sung", color="#d62828", fontsize=9)
    fig.text(.425, .055, "○  Vị trí cập nhật", color="#356b9b", fontsize=9)
    fig.text(.665, .055, "Nền đường: © OpenStreetMap contributors", fontsize=7.5, color="#9298a0")
    progress = fig.add_axes([.045, .089, .545, .004])
    progress.set(xlim=(0, 48), ylim=(0, 1))
    progress.set_facecolor("#edf0f3")
    for spine in progress.spines.values():
        spine.set_visible(False)
    progress.set_xticks([])
    progress.set_yticks([])
    bar = progress.barh(.5, 0, height=1, color="#708fae")[0]

    actions = ["Tạo mới theo lợi ích", "Tạo mới theo nhu cầu", "Bổ sung theo lợi ích",
               "Bổ sung theo nhu cầu", "Di dời bộ sạc"]
    frames = []
    for index, snapshot in enumerate(data["run"]["snapshots"]):
        for artist, is_initial in [(initial, True), (added, False)]:
            stations = [s for s in snapshot["stations"] if s["initial"] == is_initial]
            artist.set_offsets([[s["node"]["x"], s["node"]["y"]] for s in stations] or [[-100, -100]])
            artist.set_sizes([15 + 12 * s["chargers"] for s in stations] or [0])
        active = next((s for s in snapshot["stations"] if s["id"] == snapshot["target"]), None)
        selected.set_offsets([[active["node"]["x"], active["node"]["y"]]] if active else [[-100, -100]])
        action_text.set_text(actions[snapshot["action"]] if snapshot["action"] is not None else "Khởi tạo hạ tầng mẫu")
        node_text.set_text(f"Nút {active['node']['id']}" if active else "5 vị trí mẫu trên đồ thị")
        count_text.set_text(f"{len(snapshot['stations'])} trạm trong kế hoạch minh họa")
        step_text.set_text(f"Bước {index:02d} / 48")
        bar.set_width(index)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=100)
        buffer.seek(0)
        frame = Image.open(buffer).convert("RGB")
        if index == 24:
            frame.save(ASSETS / "planning-poster.png")
        frames.append(frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=128))
    frames[0].save(ASSETS / "planning-process.gif", save_all=True, append_images=frames[1:],
                   duration=[1800] + [450] * (len(frames) - 2) + [2400], loop=0, optimize=True)
    plt.close(fig)
    print(f"Copied {len(FIGURES)} source figures unchanged; rendered {len(frames)} GIF frames.")


if __name__ == "__main__":
    main()
