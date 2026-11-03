"""把检查点里的策略导出为ONNX，附带部署所需的元数据。

图的输入是未归一化的观测历史obs_hist（B, hist, 99，最旧在前）和目标goal（B, goal_frames*73）；训练结束时冻结的
观测与目标归一化统计量打包在图里。输出action（B, 29，训练时的动作语义）和joint_pos_target
（= default_joint_pos + action_scale * action，交给PD跟踪的关节位置目标）。
元数据写进ONNX的metadata_props（字符串，列表与字典为JSON），并另存同名.json。内容包括：关节顺序、默认姿态、动作尺度、
PD增益、力矩上限与电枢惯量、控制周期、观测与目标各段的排列和含义、检查点与参考数据的溯源，以及检查点旁的评估结果（若有）。
导出后用onnx.reference在随机输入上核对与PyTorch逐元素一致（需要onnx与onnxscript）。在仓库根目录运行：
    python scripts/export_onnx.py CKPT.pt OUT.onnx
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_ACTION_SCALE, G1_ARTICULATION, get_g1_robot_cfg
from mjlab.utils.lab_api.string import resolve_matching_names_values

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supermimic.tasks.tracking.constants import G1_JOINT_NAMES  # noqa: E402
from supermimic.tasks.tracking.env import NUM_SUBSTEPS, PHYSICS_DT  # noqa: E402
from supermimic.tasks.tracking.mdp import OBS_DIM  # noqa: E402
from supermimic.tasks.tracking.motion import CONTROL_DT, GOAL_FRAME_DIM, NUM_J  # noqa: E402
from supermimic.tasks.tracking.rl.networks import Policy  # noqa: E402

FORMAT, FORMAT_VERSION = "supermimic_policy_onnx", 1
# 与MDP.obs的拼接顺序一致；R为骨盆旋转矩阵（世界系），f为当前参考帧
OBS_LAYOUT = (
    ("joint_pos_rel", NUM_J, "q - default_joint_pos"),
    ("joint_vel", NUM_J, "关节速度qd（rad/s）"),
    ("base_ang_vel", 3, "骨盆角速度，骨盆坐标系（R^T w）"),
    ("projected_gravity", 3, "骨盆坐标系下的重力方向R^T (0, 0, -1)"),
    ("last_action", NUM_J, "上一步输出的action（未乘action_scale），回合第一步为0"),
    ("ref_root_rot_rel", 6, "R^T R_ref[f]的六维表示；R与R_ref须在同一世界系（部署时先对齐两者的偏航）"),
)
# 与MotionLib.goal的拼接顺序一致；H为当前参考帧根朝向R_ref[f]的偏航部分，第k帧取f+k
GOAL_LAYOUT = (
    ("root_pos", 3, "H^T (p_ref[f+k] - p_ref[f])"),
    ("root_rot", 6, "H^T R_ref[f+k]的六维表示"),
    ("root_lin_vel", 3, "参考根线速度，参考根坐标系（R_ref[f+k]^T v_ref[f+k]）"),
    ("root_ang_vel", 3, "参考根角速度，参考根坐标系"),
    ("joint_pos", NUM_J, "参考关节角（绝对值，不减默认姿态）"),
    ("joint_vel_scaled", NUM_J, "参考关节速度乘0.1"),
)


def layout(rows):
    """各段的名称、起点、长度与含义。"""
    out, offset = [], 0
    for name, dim, desc in rows:
        out.append({"name": name, "offset": offset, "dim": dim, "desc": desc})
        offset += dim
    return out, offset


class Exported(torch.nn.Module):
    """策略加上动作到PD目标的换算。"""

    def __init__(self, pi: Policy, default_q: torch.Tensor, scale: torch.Tensor):
        super().__init__()
        self.pi = pi
        self.register_buffer("default_q", default_q)
        self.register_buffer("scale", scale)

    def forward(self, obs_hist, goal):
        a = self.pi(obs_hist, goal)
        return a, self.default_q + self.scale * a


def per_joint(table: dict, fill: float | None = None) -> list[float]:
    """按G1_JOINT_NAMES展开{正则: 数值}；fill为None时要求每个关节都匹配到，否则未匹配的关节取fill。"""
    ids, _, values = resolve_matching_names_values(table, G1_JOINT_NAMES)
    if fill is None and sorted(ids) != list(range(NUM_J)):
        raise RuntimeError(f"joint table does not cover every joint: {table}")
    out = [0.0 if fill is None else float(fill)] * NUM_J
    for i, v in zip(ids, values):
        out[i] = float(v)
    return out


def robot_metadata() -> dict:
    """mjlab G1配置里的默认姿态、动作尺度与执行器参数（训练所用的同一份配置）。"""
    acts = {key: {} for key in ("stiffness", "damping", "effort_limit", "armature")}
    for a in G1_ARTICULATION.actuators:
        for expr in a.target_names_expr:
            for key, table in acts.items():
                table[expr] = getattr(a, key)
    return {
        "default_joint_pos": per_joint(get_g1_robot_cfg().init_state.joint_pos, fill=0.0),  # 同G1Kinematics.default_q
        "action_scale": per_joint(G1_ACTION_SCALE),
        "joint_stiffness": per_joint(acts["stiffness"]),
        "joint_damping": per_joint(acts["damping"]),
        "effort_limit": per_joint(acts["effort_limit"]),
        "armature": per_joint(acts["armature"]),
    }


def build_metadata(ck: dict, ckpt: Path, robot: dict, opset: int) -> dict:
    a = ck["args"]
    K, hist = a["goal_frames"], a["hist"]
    obs_layout, obs_dim = layout(OBS_LAYOUT)
    goal_layout, goal_dim = layout(GOAL_LAYOUT)
    assert (obs_dim, goal_dim) == (OBS_DIM, GOAL_FRAME_DIM)
    meta = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "inputs": {"obs_hist": ["batch", hist, OBS_DIM], "goal": ["batch", K * GOAL_FRAME_DIM]},
        "outputs": {"action": ["batch", NUM_J], "joint_pos_target": ["batch", NUM_J]},
        "joint_names": G1_JOINT_NAMES,
        **robot,
        "action_semantics": "joint_pos_target = default_joint_pos + action_scale * action，在一个控制周期内保持不变，"
                            "由位置PD（joint_stiffness、joint_damping，力矩限幅effort_limit）跟踪",
        "control_dt": CONTROL_DT,
        "physics_dt": PHYSICS_DT,
        "decimation": NUM_SUBSTEPS,
        "obs_history": hist,
        "obs_history_order": "最旧在前、最新在后；回合第一步用当前观测重复hist次填满，之后每步丢掉最旧的一帧",
        "obs_layout": obs_layout,
        "goal_frames": K,
        "goal_frame_dim": GOAL_FRAME_DIM,
        "goal_order": f"按帧拼接，goal[(k-1)*{GOAL_FRAME_DIM} + j]为第k帧（f+k，k=1..{K}）的第j维；"
                      "f+k超过片段末帧时取末帧（goal_tail=clamp）",
        "goal_layout": goal_layout,
        "goal_tail": a.get("goal_tail", "clamp"),
        "rotation_6d": "旋转矩阵前两列按行展平：[R00, R01, R10, R11, R20, R21]",
        "normalization": "观测与目标的归一化（训练冻结的均值与方差）已在图内，输入原始值",
        "training_noise": "训练时观测加均匀噪声、物理参数随机化；评估（附录F.1）与部署都不加",
        "checkpoint": {
            "file": ckpt.name,
            "sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
            "iteration": ck["iteration"],
            "saved_at": ck["saved_at"],
            "train_wall_h": round(ck["wall_time"] / 3600, 4),
            "sim_steps": ck.get("sim_steps"),
        },
        "reference": ck.get("reference"),
        "training_args": a,
        "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "torch_version": torch.__version__,
        "opset": opset,
    }
    result = ckpt.with_name(ckpt.stem + "_eval.json")
    if result.exists():
        res = json.loads(result.read_text(encoding="utf-8"))
        meta["evaluation"] = {k: v for k, v in res.items() if k != "per_clip"}
    return meta


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", help="ckpt_*.pt or last.pt")
    parser.add_argument("out", help="output .onnx; the metadata is also written next to it as .json")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--tol", type=float, default=1e-4, help="max abs difference from PyTorch on the check inputs")
    args = parser.parse_args(argv)
    import onnx
    from onnx.reference import ReferenceEvaluator

    ckpt, out = Path(args.ckpt), Path(args.out)
    if out.suffix != ".onnx":
        parser.error("out must end with .onnx")
    ck = torch.load(ckpt, map_location="cpu")
    K, hist = ck["args"]["goal_frames"], ck["args"]["hist"]
    pi = Policy(OBS_DIM, GOAL_FRAME_DIM * K, hist=hist)
    pi.load_state_dict(ck["pi"])
    robot = robot_metadata()
    model = Exported(pi, torch.tensor(robot["default_joint_pos"]), torch.tensor(robot["action_scale"])).eval()

    # 示例批量取2：批量为1时torch.export会把这一维特化成常数
    g = torch.Generator().manual_seed(0)
    def sample(norm, shape):
        return norm.mean + norm.std() * torch.randn(shape, generator=g)
    example = (sample(pi.obs_norm, (2, hist, OBS_DIM)), sample(pi.goal_norm, (2, K * GOAL_FRAME_DIM)))
    batch = torch.export.Dim("batch")
    with torch.no_grad():
        program = torch.onnx.export(model, example, input_names=["obs_hist", "goal"], output_names=["action", "joint_pos_target"],
                                    dynamic_shapes=({0: batch}, {0: batch}), opset_version=args.opset, dynamo=True)
    proto = program.model_proto

    meta = build_metadata(ck, ckpt, robot, args.opset)
    proto.doc_string = f"SuperMimic policy from {ckpt.name} (iteration {ck['iteration']}); see metadata_props"
    del proto.metadata_props[:]
    for key, value in meta.items():
        entry = proto.metadata_props.add()
        entry.key, entry.value = key, value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    onnx.checker.check_model(proto, full_check=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(proto, str(out))
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")

    # 核对：换一个批量（3）的随机输入，ONNX参考实现与PyTorch逐元素比较
    check = (sample(pi.obs_norm, (3, hist, OBS_DIM)), sample(pi.goal_norm, (3, K * GOAL_FRAME_DIM)))
    with torch.no_grad():
        want = [t.numpy() for t in model(*check)]
    got = ReferenceEvaluator(onnx.load(str(out))).run(None, {"obs_hist": check[0].numpy(), "goal": check[1].numpy()})
    diff = max(float(np.abs(x - y).max()) for x, y in zip(got, want))
    summary = {"onnx": str(out), "bytes": out.stat().st_size, "max_abs_diff": diff, "iteration": ck["iteration"],
               "inputs": meta["inputs"], "outputs": meta["outputs"]}
    print(json.dumps(summary, ensure_ascii=False))
    if not diff <= args.tol:
        sys.exit(f"ONNX output differs from PyTorch by {diff:.3g} (> --tol {args.tol})")
    return summary


if __name__ == "__main__":
    main()
