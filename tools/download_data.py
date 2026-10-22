"""Download the 40 G1-retargeted LAFAN1 clips (30 fps CSV) into data/lafan1_g1.

Source: Hugging Face dataset lvhaidong/LAFAN1_Retargeting_Dataset, folder g1/.
LAFAN1 (Ubisoft La Forge) is CC BY-NC-ND 4.0; the data is not tracked by this repository.
The robot model comes from mjlab's asset zoo, so no URDF is downloaded.
"""

from __future__ import annotations

import json
import os
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF = "https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset"
API = "https://huggingface.co/api/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/tree/main/g1"


def main():
    out = os.path.join(REPO, "data", "lafan1_g1")
    os.makedirs(out, exist_ok=True)
    with urllib.request.urlopen(API) as r:
        files = [x["path"] for x in json.load(r) if x["path"].endswith(".csv")]
    for path in files:
        dst = os.path.join(out, os.path.basename(path))
        if os.path.exists(dst):
            continue
        urllib.request.urlretrieve(f"{HF}/resolve/main/{path}", dst + ".tmp")
        os.replace(dst + ".tmp", dst)
    print(f"motions: {len(os.listdir(out))} clips in {out}")


if __name__ == "__main__":
    main()
