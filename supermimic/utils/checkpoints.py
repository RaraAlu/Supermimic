"""为滚动覆盖的检查点保留历史副本。"""

import os
import shutil


def rotate_states(path: str, keep: int) -> None:
    """在覆盖 ``path`` 之前轮换检查点；``keep`` 是包含当前文件在内的保留总数。

    当前文件保存为 ``<stem>_prev1<ext>``，更早的副本依次后移。文件系统支持时使用硬链接，
    否则复制文件；后续替换 ``path`` 不会改变硬链接指向的旧内容。这样训练崩溃前的完整状态
    可以在后续 ``keep - 1`` 次保存期间保留。
    """
    if keep < 2 or not os.path.exists(path):
        return
    stem, ext = os.path.splitext(path)

    def prev(i):
        """生成第 i 个历史检查点的路径。"""
        return f"{stem}_prev{i}{ext}"

    # 从最旧副本向前轮换，避免先覆盖仍待搬移的文件；当前检查点留给最后一步建立历史副本。
    for i in range(keep - 1, 1, -1):
        if os.path.exists(prev(i - 1)):
            os.replace(prev(i - 1), prev(i))
    if os.path.exists(prev(1)):
        os.remove(prev(1))
    try:
        os.link(path, prev(1))
    except OSError:
        shutil.copy2(path, prev(1))
