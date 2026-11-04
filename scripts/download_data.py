"""下载 40 个重定向到 G1 的 LAFAN1 片段（30 fps CSV），保存到 data/lafan1_g1。

来源为 Hugging Face 的 lvhaidong/LAFAN1_Retargeting_Dataset 数据集 g1/ 目录。
LAFAN1（Ubisoft La Forge）采用 CC BY-NC-ND 4.0 许可，数据不纳入本仓库版本控制。
机器人模型由 mjlab 资源库提供，无需下载 URDF。"""

from __future__ import annotations

import json
import os
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF = "https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset"
API = "https://huggingface.co/api/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/tree/main/g1"


def main():
    """创建数据目录，跳过已有片段；下载到临时文件后原子替换目标文件。"""
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
