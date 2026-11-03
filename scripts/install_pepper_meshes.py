#!/usr/bin/env python3
"""Install the Pepper 3D meshes for the tablet's 3D view, after licence acceptance.

    python scripts/install_pepper_meshes.py                 # download (asks you to accept)
    python scripts/install_pepper_meshes.py --from DIR      # copy from a folder you already have

The meshes are (c) SoftBank Robotics / Aldebaran and licensed CC BY-NC-ND 4.0,
and Aldebaran allows redistribution only when the user explicitly accepts the
licence. So they are never committed to this repository: this script asks you
to type "yes", then copies the unmodified files into
frontend/assets/pepper3d/meshes/ (git-ignored).

Sources, in order of preference:
  * --from DIR: the "meshes/1.0" folder installed by the official ROS package
    (ros-naoqi/pepper_meshes, Linux installer) or any other official copy.
  * default: the copy published in github.com/Maelic/llm-agent-pepper-robot
    (assets/pepper_meshes/1.0), which works on Windows, macOS and Linux.
Without the meshes the website simply shows the 2D Pepper face.
"""

import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "frontend" / "assets" / "pepper3d" / "pepper-model.json"
DEST = ROOT / "frontend" / "assets" / "pepper3d" / "meshes"
DEFAULT_SOURCE = "https://raw.githubusercontent.com/Maelic/llm-agent-pepper-robot/main/assets/pepper_meshes/1.0/"
TEXTURES = ["juliette_ear.png", "juliette_eyes.png", "juliette_face.png", "juliette_head.png"]
LICENCE_URL = "https://github.com/ros-naoqi/pepper_meshes/blob/master/LICENSE"

NOTICE = """Pepper 3D meshes (c) SoftBank Robotics / Aldebaran.
Licensed under Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
International: https://creativecommons.org/licenses/by-nc-nd/4.0/
Full licence and redistribution conditions: %s
Installed unmodified for non-commercial use after explicit licence acceptance.
""" % LICENCE_URL


def needed_files():
    model = json.loads(MODEL.read_text(encoding="utf-8"))
    meshes = sorted({v["mesh"] for visuals in model["links"].values() for v in visuals})
    return meshes + TEXTURES


def accept_licence():
    print(__doc__.split("Sources,")[0].strip())
    print("\nLicence: Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International")
    print("Full text and Aldebaran's conditions: " + LICENCE_URL)
    print("In short: non-commercial use only, credit SoftBank Robotics / Aldebaran, no modified versions.\n")
    answer = input('Type "yes" to accept the licence and install the meshes: ').strip().lower()
    return answer == "yes"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="source_dir", help="folder containing HeadPitch.dae, Torso.dae, ...")
    parser.add_argument("--source-url", default=DEFAULT_SOURCE, help="base URL to download the files from")
    args = parser.parse_args(argv)

    if not accept_licence():
        print("Licence not accepted; nothing installed. The website will keep the 2D face.")
        return 1

    files = needed_files()
    DEST.mkdir(parents=True, exist_ok=True)
    missing = []
    for index, name in enumerate(files, 1):
        target = DEST / name
        try:
            if args.source_dir:
                shutil.copyfile(Path(args.source_dir) / name, target)
            else:
                with urllib.request.urlopen(args.source_url + name, timeout=60) as response:
                    target.write_bytes(response.read())
        except (OSError, ValueError) as error:
            missing.append("%s (%s)" % (name, error))
        print("\r%d/%d %s" % (index, len(files), name.ljust(28)), end="", flush=True)
    print()
    (DEST / "NOTICE.txt").write_text(NOTICE, encoding="utf-8")

    if missing:
        print("Could not install %d files:\n  %s" % (len(missing), "\n  ".join(missing)))
        return 1
    print("Installed %d files into %s" % (len(files), DEST.relative_to(ROOT)))
    print("Run `python scripts/sync_pepper_html.py` to copy them into the Pepper tablet build too.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
