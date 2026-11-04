"""训练日志：每组一份CSV，可选同时写TensorBoard标量（标签为<组名>/<列名>，步数为迭代数）。"""

import os


class RunLog:
    """按组写入训练 CSV，并可同步记录 TensorBoard 标量。"""

    def __init__(self, log_dir: str, tables: dict, tensorboard: bool = False, purge_step: int | None = None):
        """tables：组名 →（CSV文件名, 列名），第一列是迭代数。追加写入；文件为空时先写表头，
        因此恢复训练时沿用原有的列。purge_step让TensorBoard隐藏上一次运行在该步及之后写下的事件。"""
        self.tables = tables
        self.files = {}
        for group, (name, columns) in tables.items():
            f = open(os.path.join(log_dir, name), "a", encoding="utf-8")
            if f.tell() == 0:
                f.write(",".join(columns) + "\n")
            self.files[group] = f
        self.tb = None
        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            self.tb = SummaryWriter(os.path.join(log_dir, "tensorboard"), purge_step=purge_step)

    def write(self, rows: dict):
        """rows：组名 → 一行数值（与该组的列一一对应）；浮点数保留6位有效数字。"""
        for group, row in rows.items():
            self.files[group].write(",".join(f"{x:.6g}" if isinstance(x, float) else str(x) for x in row) + "\n")
            self.files[group].flush()
        # CSV 的第一列作为 TensorBoard 步数，其余列逐一写标量；每轮刷新便于外部监控及时读取。
        if self.tb:
            for group, row in rows.items():
                for name, x in zip(self.tables[group][1][1:], row[1:]):
                    self.tb.add_scalar(f"{group}/{name}", x, row[0])
            self.tb.flush()

    def write_lines(self, group: str, rows: list) -> None:
        """一次写多行（如每次记录前若干个片段）；不写TensorBoard，行内可含文字列。"""
        for row in rows:
            self.files[group].write(",".join(f"{x:.6g}" if isinstance(x, float) else str(x) for x in row) + "\n")
        self.files[group].flush()

    def close(self):
        """关闭所有日志文件及可选的 TensorBoard 写入器。"""
        for f in self.files.values():
            f.close()
        if self.tb:
            self.tb.close()
