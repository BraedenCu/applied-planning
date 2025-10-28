#!/usr/bin/env python3
from __future__ import annotations

"""Fetch the MuJoCo Menagerie uFactory Lite 6 assets into the project.

This downloads the mujoco_menagerie main.zip and extracts only the
`ufactory_lite6/` directory into:

    src/applied_planning/sim/assets/ufactory_lite6/

Requirements: standard library only (urllib, zipfile).
"""

import io
import os
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen

REPO_ZIP = (
    "https://github.com/google-deepmind/mujoco_menagerie/archive/refs/heads/main.zip"
)
SUBDIR = "mujoco_menagerie-main/ufactory_lite6/"
DEST = Path(__file__).resolve().parents[1] / "src/applied_planning/sim/assets/ufactory_lite6"


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Menagerie archive…\n  {REPO_ZIP}")
    with urlopen(REPO_ZIP) as resp:
        data = resp.read()
    print("Download complete. Extracting Lite6 assets…")

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        members = [m for m in zf.namelist() if m.startswith(SUBDIR)]
        if not members:
            print("Could not find ufactory_lite6 directory in archive.")
            sys.exit(1)
        for name in members:
            rel = name[len(SUBDIR) :]
            if not rel:  # directory itself
                continue
            target_path = DEST / rel
            if name.endswith("/"):
                target_path.mkdir(parents=True, exist_ok=True)
            else:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(name) as src, open(target_path, "wb") as dst:
                    dst.write(src.read())
    print(f"Assets extracted to: {DEST}")
    print("Examples:")
    print(f"  MJCF: {DEST / 'lite6.xml'}")
    print(f"  Scene: {DEST / 'scene.xml'}")


if __name__ == "__main__":
    main()
