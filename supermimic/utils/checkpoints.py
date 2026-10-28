"""Keep earlier copies of a rolling checkpoint file."""

import os
import shutil


def rotate_states(path: str, keep: int) -> None:
    """Call before ``path`` is overwritten: ``keep`` is how many files to retain, ``path`` included.

    The current file becomes ``<stem>_prev1<ext>`` and the earlier copies shift to ``_prev2``, ... The copy is a
    hard link where the file system has them (nothing is rewritten, and replacing ``path`` afterwards leaves the
    link on the old contents), else a copy. A run that collapses keeps its pre-collapse full state for
    ``keep - 1`` more saves instead of losing it at the next one.
    """
    if keep < 2 or not os.path.exists(path):
        return
    stem, ext = os.path.splitext(path)

    def prev(i):
        return f"{stem}_prev{i}{ext}"

    for i in range(keep - 1, 1, -1):
        if os.path.exists(prev(i - 1)):
            os.replace(prev(i - 1), prev(i))
    if os.path.exists(prev(1)):
        os.remove(prev(1))
    try:
        os.link(path, prev(1))
    except OSError:
        shutil.copy2(path, prev(1))
