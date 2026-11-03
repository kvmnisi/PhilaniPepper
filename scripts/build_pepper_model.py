#!/usr/bin/env python3
"""Convert Pepper's URDF into the small kinematic model the 3D viewer reads.

    python scripts/build_pepper_model.py path/to/pepper.urdf

The URDF comes from ros-naoqi/pepper_robot (pepper_description, Apache 2.0):
    pepper_description/urdf/pepper1.0_generated_urdf/pepper.urdf
Output: frontend/assets/pepper3d/pepper-model.json (committed). It holds only
joint frames and mesh file names, never mesh data; the meshes themselves are
installed separately by scripts/install_pepper_meshes.py.
"""

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "frontend" / "assets" / "pepper3d" / "pepper-model.json"


def floats(text, default):
    return [float(v) for v in text.split()] if text else list(default)


def origin(element):
    node = element.find("origin") if element is not None else None
    if node is None:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    return floats(node.get("xyz"), [0, 0, 0]), floats(node.get("rpy"), [0, 0, 0])


def build(urdf_path):
    robot = ET.parse(urdf_path).getroot()
    links = {}
    for link in robot.findall("link"):
        visuals = []
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                continue
            xyz, rpy = origin(visual)
            visuals.append({
                "mesh": Path(mesh.get("filename")).name,
                "xyz": xyz, "rpy": rpy,
                "scale": floats(mesh.get("scale"), [1, 1, 1]),
            })
        links[link.get("name")] = visuals

    joints = []
    for joint in robot.findall("joint"):
        xyz, rpy = origin(joint)
        axis = joint.find("axis")
        joints.append({
            "name": joint.get("name"), "type": joint.get("type"),
            "parent": joint.find("parent").get("link"), "child": joint.find("child").get("link"),
            "xyz": xyz, "rpy": rpy, "axis": floats(axis.get("xyz") if axis is not None else None, [1, 0, 0]),
        })

    # Keep only links that carry a mesh, plus the links on the way to them.
    parent_of = {j["child"]: j for j in joints}
    keep = set()
    for name, visuals in links.items():
        if visuals:
            while name and name not in keep:
                keep.add(name)
                name = parent_of[name]["parent"] if name in parent_of else None
    children = {j["child"] for j in joints}
    roots = [name for name in keep if name not in children]
    return {
        "source": "ros-naoqi/pepper_robot pepper_description pepper.urdf (Apache License 2.0)",
        "up_axis": "Z",
        "root": roots[0] if len(roots) == 1 else roots,
        "links": {name: links[name] for name in sorted(keep)},
        "joints": [j for j in joints if j["child"] in keep],
    }


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    model = build(sys.argv[1])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(model, indent=1) + "\n", encoding="utf-8")
    meshes = sorted({v["mesh"] for visuals in model["links"].values() for v in visuals})
    print("Wrote %s: %d links, %d joints, %d meshes" % (OUT.relative_to(ROOT), len(model["links"]), len(model["joints"]), len(meshes)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
