"""用导出的ONNX策略在CPU MuJoCo里闭环跟踪一段动作，也是部署端怎么构造输入的参考实现。

控制部分只用numpy、mujoco和onnxruntime，输入完全按ONNX的元数据与原始npz（root_pos、root_quat_wxyz、joint_pos，50 Hz）构造，
不读策略网络和MotionLib的代码；评分（--score_source，与附录F.1同一套奖励与终止）才用到supermimic。
--variants可以同时跑常见的部署偏差（输入约定、PD增益、物理设置），看各自把跟踪效果拖低多少，all为全部；
stress是逐级加大的压力测试（延迟、增益、电枢惯量、力矩上限、摩扰、质量、噪声、丢帧），看从哪一级开始摔、摔在第几秒，带随机性的项跑--seeds次。
模型默认用mjlab搭出训练时的场景（平地、位置执行器、电枢惯量、碰撞设置、求解器选项、足部摩擦1.0）；--save_model存成.mjb后，
没有mjlab的机器可以用--model载入同一份模型。注意mjlab自带的g1.xml不是这个模型：它步长2 ms、Euler积分、电枢惯量为0，没有执行器和地面，
这些都由mjlab在运行时加上（见raw_xml_*变体）。--golden把baseline前若干步的状态、输入与输出存下来，可与自己的推理程序逐段比对。
在仓库根目录运行：
    python scripts/sim2sim_onnx.py POLICY.onnx MOTION.npz --score_source MOTION.h5 --variants all --out sim2sim.csv
"""
import argparse
import copy
import csv
import json
import re
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

REPO = Path(__file__).resolve().parents[1]
TERM_HEIGHT, TERM_TILT = 0.25, 0.8  # RewardCfg的终止条件
FOOT_GEOMS = r"^(.*/)?(left|right)_foot[1-7]_collision$"
OBS_NOISE = {"joint_pos_rel": 0.01, "joint_vel": 0.5, "base_ang_vel": 0.2, "projected_gravity": 0.05, "ref_root_rot_rel": 0.05}  # 表A.1
# 常见的Unitree G1部署增益（量级示意）
HIGH_GAINS = ((r"hip", 100.0, 2.0), (r"knee", 150.0, 4.0), (r"ankle", 40.0, 2.0), (r"waist", 200.0, 5.0),
              (r"wrist", 20.0, 0.5), (r"shoulder|elbow", 40.0, 1.0))

# 每个变体相对baseline只改一处：sim_*是物理设置，其余是输入或控制上的偏差
VARIANTS = {
    "baseline": {},
    # ---- 物理设置 ----
    "sim_friction_0.6": {"friction": 0.6},
    "sim_friction_0.4": {"friction": 0.4},
    "sim_no_armature": {"no_armature": True},
    "sim_feet_only_collision": {"feet_only": True},
    "sim_default_solver": {"default_solver": True},
    "sim_dt2ms_decim10": {"timestep": 0.002, "decimation": 10},
    "sim_high_gains": {"high_gains": True},
    "sim_kp_x0.5": {"gain_scale": 0.5},
    "sim_kp_x2": {"gain_scale": 2.0},
    "sim_explicit_torque_pd": {"torque_pd": True},
    "sim_gains_in_actuator_order": {"gains_actuator_order": True},
    "sim_no_force_limit": {"no_force_limit": True},
    # ---- 直接载入mjlab自带的g1.xml（加一块地面），PD用力矩实现，增益取元数据 ----
    "raw_xml_dt2ms_decim4": {"raw_xml": True, "decimation": 4},  # 步长沿用XML、子步沿用元数据：控制8 ms，动作快2.5倍
    "raw_xml_dt2ms_decim10": {"raw_xml": True, "decimation": 10},
    "raw_xml_dt5ms_decim4": {"raw_xml": True, "timestep": 0.005},
    "raw_xml_armature_dt5ms_decim4": {"raw_xml": True, "timestep": 0.005, "add_armature": True},
    # ---- 控制频率与参考播放速度 ----
    "rate_dt2ms_decim4": {"timestep": 0.002, "decimation": 4},  # 控制8 ms，参考仍每步一帧，动作快2.5倍
    "rate_decim2": {"decimation": 2},  # 控制100 Hz，动作快2倍
    "rate_action_delay_1step": {"delay": 1},
    # ---- 起始状态 ----
    "start_zero_velocity": {"start": "zero_velocity"},
    "start_default_pose": {"start": "default_pose"},
    "start_yaw_offset_30deg": {"yaw_offset": 30.0},
    "start_yaw_offset_90deg": {"yaw_offset": 90.0},
    # ---- 输出 ----
    "act_ctrl_in_policy_order": {"ctrl_raw_order": True},
    "act_use_action_as_target": {"ctrl_action": True},
    "act_forget_default_pose": {"ctrl_no_default": True},
    # ---- 观测 ----
    "obs_noise_training": {"obs_noise": True},
    "obs_velocity_scaled": {"obs_vel_scaled": True},  # 常见写法：关节速度乘0.1、角速度乘0.25；这个策略不缩放
    "obs_robot_quat_as_xyzw": {"robot_quat_xyzw": True},  # MuJoCo的wxyz被当成xyzw（如scipy的from_quat）
    "obs_hist_newest_first": {"hist_newest_first": True},
    "obs_hist_zero_fill": {"hist_zero_fill": True},
    "obs_joint_pos_absolute": {"joint_pos_abs": True},
    "obs_ang_vel_world": {"ang_vel_world": True},
    "obs_gravity_sign_flipped": {"gravity_flip": True},
    "obs_last_action_scaled": {"last_action": "scaled"},
    "obs_last_action_target": {"last_action": "target"},
    "obs_last_action_zero": {"last_action": "zero"},
    "obs_rel_rot_inverted": {"rel_inverse": True},
    "rot6d_columns_flattened": {"rot6d_columns": True},
    # ---- 目标 ----
    "goal_joint_vel_unscaled": {"goal_vel_unscaled": True},
    "goal_frames_from_f": {"goal_offset": 0},
    "goal_world_frame": {"goal_world": True},
    "goal_root_vel_world": {"goal_vel_world": True},
    "goal_joint_pos_minus_default": {"goal_q_rel": True},
    "goal_ref_vel_zero": {"goal_vel_zero": True},
    "goal_ref_forward_diff": {"goal_forward_diff": True},
}


# 压力测试：训练时只随机化了摩擦（0.3～1.2）、关节零位（±0.01 rad）、躯干质心（±2.5～5 cm）、观测噪声和动作噪声（0.1）
STRESS = {}
for k in (1, 2, 3, 4, 5, 6):
    STRESS[f"stress_action_delay_{20 * k}ms"] = {"delay": k}
for k in (1, 2, 3, 4):
    STRESS[f"stress_obs_delay_{20 * k}ms"] = {"obs_delay": k}
for x in (0.7, 0.8, 0.9, 1.1, 1.2, 1.5):
    STRESS[f"stress_kp_x{x}"] = {"kp_scale": x}
for x in (0.5, 0.7, 0.85, 1.2, 1.5, 2.0):
    STRESS[f"stress_kd_x{x}"] = {"kd_scale": x}
for x in (0.25, 0.5, 0.75, 1.25, 1.5, 2.0):
    STRESS[f"stress_armature_x{x}"] = {"armature_scale": x}
for x in (0.9, 0.8, 0.7, 0.6, 0.5):
    STRESS[f"stress_effort_x{x}"] = {"effort_scale": x}
for x in (0.3, 0.25, 0.2, 0.15):
    STRESS[f"stress_friction_{x}"] = {"friction": x}
for x in (0.9, 1.1, 1.2):
    STRESS[f"stress_all_mass_x{x}"] = {"mass_scale": x}
for x in (0.05, 0.1):
    STRESS[f"stress_torso_com_forward_{x}m"] = {"com_offset": (x, 0.0, 0.0)}
for x in (1, 2, 3, 5):
    STRESS[f"stress_obs_noise_x{x}"] = {"obs_noise": x, "random": True}
for x in (0.1, 0.2, 0.4):
    STRESS[f"stress_action_noise_{x}"] = {"action_noise": x, "random": True}
for x in (0.01, 0.02, 0.05):
    STRESS[f"stress_joint_offset_{x}rad"] = {"joint_offset": x, "random": True}
for x in (0.02, 0.05, 0.1, 0.2):
    STRESS[f"stress_missed_cycles_{x}"] = {"drop_prob": x, "random": True}
for x in (1, 2):
    STRESS[f"stress_decimation_jitter_{x}"] = {"decim_jitter": x, "random": True}
STRESS["stress_no_armature_dt2ms_decim10"] = {"no_armature": True, "timestep": 0.002, "decimation": 10}
STRESS["stress_raw_xml_armature_dt2ms_decim10"] = {"raw_xml": True, "add_armature": True, "decimation": 10}
STRESS["stress_combo_delay20ms_noise"] = {"delay": 1, "obs_noise": 1, "random": True}
STRESS["stress_combo_delay40ms_noise"] = {"delay": 2, "obs_noise": 1, "random": True}
STRESS["stress_combo_delay20ms_noise_friction0.6"] = {"delay": 1, "obs_noise": 1, "friction": 0.6, "random": True}
VARIANTS.update(STRESS)


def quat_to_mat(q):
    """wxyz单位四元数转旋转矩阵。"""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                     2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                     2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), -1).reshape(q.shape[:-1] + (3, 3))


def so3_log(R):
    """旋转矩阵转旋转向量（转角明显小于π）。"""
    vee = np.stack((R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]), -1)
    s = np.linalg.norm(vee, axis=-1, keepdims=True) / 2
    c = (np.trace(R, axis1=-2, axis2=-1)[..., None] - 1) / 2
    return vee * np.where(s < 1e-9, 0.5, np.arctan2(s, c) / (2 * np.maximum(s, 1e-9)))


def yaw_mat(R):
    """把x轴投到水平面，取只含偏航的朝向（与训练相同）。"""
    fx, fy = R[..., 0, 0], R[..., 1, 0]
    n = np.sqrt(fx * fx + fy * fy + 1e-9)
    c, s = fx / n, fy / n
    o, one = np.zeros_like(c), np.ones_like(c)
    return np.stack((c, -s, o, s, c, o, o, o, one), -1).reshape(R.shape[:-2] + (3, 3))


def rot6d(R, columns=False):
    """元数据rotation_6d的约定：前两列按行展平。columns=True是常见的错误写法（按列展平）。"""
    two = R[..., :, :2]
    return (np.swapaxes(two, -1, -2) if columns else two).reshape(R.shape[:-2] + (6,))


def gradient(x, dt):
    """内部中心差分，两端单边差分（参考速度的算法与MotionLib相同）。"""
    g = np.zeros_like(x)
    g[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
    g[0], g[-1] = (x[1] - x[0]) / dt, (x[-1] - x[-2]) / dt
    return g


class Reference:
    """从原始npz算出目标用到的根速度与关节速度。"""

    def __init__(self, path, meta):
        data = np.load(path, allow_pickle=False)
        if "fps" in data and round(float(data["fps"]) * meta["control_dt"]) != 1:
            sys.exit(f"{path}: fps {data['fps']} does not match control_dt {meta['control_dt']}")
        if "joint_names" in data and [str(n) for n in data["joint_names"]] != meta["joint_names"]:
            sys.exit(f"{path}: joint order differs from the policy's joint_names")
        dt = meta["control_dt"]
        self.p, self.quat, self.q = (np.asarray(data[k], dtype=np.float64) for k in ("root_pos", "root_quat_wxyz", "joint_pos"))
        self.T = len(self.p)
        self.R = quat_to_mat(self.quat)
        self.v, self.qd = gradient(self.p, dt), gradient(self.q, dt)
        self.w = np.zeros_like(self.p)
        self.w[1:-1] = so3_log(self.R[2:] @ np.swapaxes(self.R[:-2], -1, -2)) / (2 * dt)
        self.w[0] = so3_log(self.R[1] @ self.R[0].T) / dt
        self.w[-1] = so3_log(self.R[-1] @ self.R[-2].T) / dt
        Rt = np.swapaxes(self.R, -1, -2)
        self.v_local = (Rt @ self.v[..., None])[..., 0]
        self.w_local = (Rt @ self.w[..., None])[..., 0]

    def goals(self, meta, flags=None):
        """整段动作每一帧的目标（T, goal_frames*73），只依赖参考动作，所以可以预先算好。"""
        flags = flags or {}
        K, default_q = meta["goal_frames"], np.array(meta["default_joint_pos"])
        fut = np.minimum(np.arange(self.T)[:, None] + np.arange(K)[None] + flags.get("goal_offset", 1), self.T - 1)
        H = np.broadcast_to(np.eye(3), (self.T, 3, 3)) if flags.get("goal_world") else yaw_mat(self.R)
        Ht = np.swapaxes(H, -1, -2)[:, None]
        origin = 0.0 if flags.get("goal_world") else self.p[:, None]
        dp = (Ht @ (self.p[fut] - origin)[..., None])[..., 0]
        rot = rot6d(Ht @ self.R[fut], flags.get("rot6d_columns", False))
        v, w = (self.v, self.w) if flags.get("goal_vel_world") else (self.v_local, self.w_local)
        qd = self.qd
        if flags.get("goal_forward_diff"):
            dt = meta["control_dt"]
            qd = np.concatenate(((self.q[1:] - self.q[:-1]) / dt, np.zeros((1, self.q.shape[1]))))
            v_world = np.concatenate(((self.p[1:] - self.p[:-1]) / dt, np.zeros((1, 3))))
            v = (np.swapaxes(self.R, -1, -2) @ v_world[..., None])[..., 0]
        if flags.get("goal_vel_zero"):
            v, w, qd = v * 0, w * 0, qd * 0
        q = self.q[fut] - default_q if flags.get("goal_q_rel") else self.q[fut]
        qd = qd[fut] * (1.0 if flags.get("goal_vel_unscaled") else 0.1)
        return np.concatenate((dp, rot, v[fut], w[fut], q, qd), -1).reshape(self.T, -1).astype(np.float32)


def training_model():
    """用mjlab搭出与TrackingSim相同的单环境场景；足部摩擦取评估时的1.0。"""
    sys.path.insert(0, str(REPO))
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_g1_robot_cfg
    from mjlab.scene import Scene, SceneCfg
    from mjlab.terrains import TerrainEntityCfg
    from supermimic.tasks.tracking.env import make_sim_cfg

    scene = Scene(SceneCfg(num_envs=1, env_spacing=0.0, terrain=TerrainEntityCfg(terrain_type="plane"),
                           entities={"robot": get_g1_robot_cfg()}), device="cpu")
    m = scene.spec.compile()
    make_sim_cfg().mujoco.apply(m)
    set_foot_friction(m, 1.0)
    return m


def raw_xml_model():
    """mjlab自带的g1.xml加一块地面：步长2 ms、Euler积分、电枢惯量为0、没有执行器。"""
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec

    spec = get_spec()
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05])
    return spec.compile()


def foot_geoms(m):
    return [i for i in range(m.ngeom) if re.match(FOOT_GEOMS, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "")]


def set_foot_friction(m, value):
    ids = foot_geoms(m)
    if not ids:
        sys.exit("no foot collision geoms found in the model")
    m.geom_friction[ids, 0] = value


class Indexing:
    """按名字找关节与执行器：编译后执行器的顺序与策略的关节顺序不同，不能直接写data.ctrl[:]。"""

    def __init__(self, m, joint_names):
        table = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j): j for j in range(m.njnt)}
        jids = []
        for name in joint_names:
            hits = [j for key, j in table.items() if key == name or key.endswith("/" + name)]
            if len(hits) != 1:
                sys.exit(f"joint {name}: {len(hits)} matches in the model")
            jids.append(hits[0])
        free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        if len(free) != 1:
            sys.exit(f"expected one free joint, found {len(free)}")
        self.root_q, self.root_v = m.jnt_qposadr[free[0]], m.jnt_dofadr[free[0]]
        self.jids = np.array(jids)
        self.q, self.v = m.jnt_qposadr[self.jids], m.jnt_dofadr[self.jids]
        by_joint = {int(m.actuator_trnid[i, 0]): i for i in range(m.nu)}
        self.act = np.array([by_joint[j] for j in jids]) if m.nu else None


def configure(m0, meta, flags):
    """按变体复制并改动模型，返回模型、索引和每个控制步的物理子步数。"""
    m = copy.copy(raw_xml_model() if flags.get("raw_xml") else m0)
    idx = Indexing(m, meta["joint_names"])
    if "friction" in flags:
        set_foot_friction(m, flags["friction"])
    if flags.get("no_armature") or flags.get("add_armature"):
        m.dof_armature[idx.v] = 0.0 if flags.get("no_armature") else np.array(meta["armature"])
        mujoco.mj_setConst(m, mujoco.MjData(m))  # 重算依赖惯量的常量
    if flags.get("feet_only"):
        feet = set(foot_geoms(m))
        for g in range(m.ngeom):
            if g not in feet and m.body_rootid[m.geom_bodyid[g]] == m.body_rootid[m.jnt_bodyid[0]]:  # 只动机器人，不动地面
                m.geom_contype[g] = m.geom_conaffinity[g] = 0
        m.geom_contype[list(feet)] = 0  # 足部只与地面接触
    if flags.get("default_solver"):
        d = mujoco.MjOption()
        m.opt.integrator, m.opt.cone, m.opt.iterations, m.opt.ls_iterations = d.integrator, d.cone, d.iterations, d.ls_iterations
    if "timestep" in flags:
        m.opt.timestep = flags["timestep"]
    kp, kd = np.array(meta["joint_stiffness"]), np.array(meta["joint_damping"])
    if flags.get("high_gains"):
        for i, name in enumerate(meta["joint_names"]):
            kp[i], kd[i] = next((p, d) for pat, p, d in HIGH_GAINS if re.search(pat, name))
    kp = kp * flags.get("gain_scale", 1.0) * flags.get("kp_scale", 1.0)
    kd = kd * flags.get("gain_scale", 1.0) * flags.get("kd_scale", 1.0)
    for pattern, x in flags.get("gain_by_name", ()):  # 按关节名同时缩放kp与kd
        hits = [i for i, name in enumerate(meta["joint_names"]) if re.search(pattern, name)]
        kp[hits], kd[hits] = kp[hits] * x, kd[hits] * x
    if "armature_scale" in flags:
        m.dof_armature[idx.v] *= flags["armature_scale"]
    if "mass_scale" in flags:
        m.body_mass *= flags["mass_scale"]
        m.body_inertia *= flags["mass_scale"]
    if "com_offset" in flags:
        torso = next(b for b in range(m.nbody) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").endswith("torso_link"))
        m.body_ipos[torso] += np.array(flags["com_offset"])
    if {"armature_scale", "mass_scale", "com_offset"} & set(flags):
        mujoco.mj_setConst(m, mujoco.MjData(m))
    if "effort_scale" in flags and m.nu:
        m.actuator_forcerange[idx.act] *= flags["effort_scale"]
    if m.nu:
        # 位置执行器：力 = kp*(ctrl - q) - kd*qd
        act = np.arange(m.nu) if flags.get("gains_actuator_order") else idx.act  # 错误示范：把关节顺序的增益按执行器顺序写入
        off = flags.get("torque_pd", False)
        m.actuator_gainprm[act, 0], m.actuator_biasprm[act, 1], m.actuator_biasprm[act, 2] = 0 if off else kp, 0 if off else -kp, 0 if off else -kd
        if flags.get("no_force_limit"):
            m.actuator_forcelimited[idx.act] = 0
    return m, idx, flags.get("decimation", meta["decimation"]), kp, kd


def initial_state(m, d, idx, ref, meta, flags):
    """默认与评估相同：完全放在参考首帧（含速度），执行器目标取首帧关节角。"""
    p, R, v, w, q, qd = ref.p[0].copy(), ref.R[0].copy(), ref.v[0].copy(), ref.w[0].copy(), ref.q[0].copy(), ref.qd[0].copy()
    if flags.get("start") == "zero_velocity":
        v, w, qd = v * 0, w * 0, qd * 0
    if flags.get("start") == "default_pose":
        v, w, qd, q = v * 0, w * 0, qd * 0, np.array(meta["default_joint_pos"])
        p[2], R = 0.76, yaw_mat(R)
    if "yaw_offset" in flags:
        a = np.deg2rad(flags["yaw_offset"])
        Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        R, v, w = Rz @ R, Rz @ v, Rz @ w
    lo, hi = m.jnt_range[idx.jids, 0], m.jnt_range[idx.jids, 1]
    q = np.clip(q, lo, hi)
    mujoco.mju_mat2Quat(quat := np.zeros(4), R.reshape(9))
    d.qpos[idx.root_q:idx.root_q + 3], d.qpos[idx.root_q + 3:idx.root_q + 7] = p, quat
    d.qvel[idx.root_v:idx.root_v + 3], d.qvel[idx.root_v + 3:idx.root_v + 6] = v, R.T @ w  # 自由关节的角速度在机体系
    d.qpos[idx.q], d.qvel[idx.v] = q, qd
    if m.nu:
        d.ctrl[idx.act] = q
    mujoco.mj_forward(m, d)
    return q


def rollout(session, meta, ref, m0, flags=None, seed=0, stop_on_fail=True, trace_steps=0):
    """闭环跑完整段动作，返回逐步的状态、动作、失败步，以及前trace_steps步的状态、输入与输出。"""
    flags = flags or {}
    m, idx, decimation, kp, kd = configure(m0, meta, flags)
    torque_pd = flags.get("torque_pd", False) or m.nu == 0
    d = mujoco.MjData(m)
    hist_len, obs_dim = meta["inputs"]["obs_hist"][1:]
    default_q, scale = np.array(meta["default_joint_pos"]), np.array(meta["action_scale"])
    effort = np.array(meta["effort_limit"]) * flags.get("effort_scale", 1.0)
    goals = ref.goals(meta, flags)
    rng = np.random.default_rng(seed)
    offset = rng.uniform(-1, 1, len(default_q)) * flags.get("joint_offset", 0.0)
    obs_buf = []
    noise = np.zeros(obs_dim)
    for row in meta["obs_layout"]:
        noise[row["offset"]:row["offset"] + row["dim"]] = OBS_NOISE.get(row["name"], 0.0)
    target = initial_state(m, d, idx, ref, meta, flags)
    a_prev, pending = np.zeros(len(default_q)), []
    hist = None
    out = {k: [] for k in ("p", "quat", "v", "w", "q", "qd", "a", "a_prev")}
    trace = {k: [] for k in ("qpos", "qvel", "obs", "goal", "action", "joint_pos_target")}
    fail_step = None
    for f in range(ref.T - 1):
        root = d.qpos[idx.root_q:idx.root_q + 7]
        R = quat_to_mat(root[[6, 3, 4, 5]] if flags.get("robot_quat_xyzw") else root[3:7])
        w_body = d.qvel[idx.root_v + 3:idx.root_v + 6].copy()
        q, qd = d.qpos[idx.q].copy(), d.qvel[idx.v].copy()
        M = ref.R[f].T @ R if flags.get("rel_inverse") else R.T @ ref.R[f]
        vel_scale, ang_scale = (0.1, 0.25) if flags.get("obs_vel_scaled") else (1.0, 1.0)
        obs = np.concatenate((q if flags.get("joint_pos_abs") else q - default_q - offset, qd * vel_scale,
                              (R @ w_body if flags.get("ang_vel_world") else w_body) * ang_scale,
                              (1 if flags.get("gravity_flip") else -1) * R[2], a_prev,
                              rot6d(M, flags.get("rot6d_columns", False))))
        if flags.get("obs_noise"):
            obs = obs + rng.uniform(-1, 1, obs_dim) * noise * float(flags["obs_noise"])
        obs = obs.astype(np.float32)
        if flags.get("obs_delay"):  # 传感器延迟：策略看到的是若干步之前的观测
            obs_buf.append(obs)
            obs = obs_buf[max(0, len(obs_buf) - 1 - flags["obs_delay"])]
        if hist is None:
            hist = np.zeros((hist_len, obs_dim), np.float32) if flags.get("hist_zero_fill") else np.tile(obs, (hist_len, 1))
            if flags.get("hist_zero_fill"):
                hist[-1] = obs
        else:
            hist = np.concatenate((hist[1:], obs[None]))
        feed = hist[::-1].copy() if flags.get("hist_newest_first") else hist
        a, tgt = session.run(None, {"obs_hist": feed[None], "goal": goals[f][None]})
        a, tgt = a[0].astype(np.float64), tgt[0].astype(np.float64)
        if f < trace_steps:
            for key, value in (("qpos", d.qpos), ("qvel", d.qvel), ("obs", obs), ("goal", goals[f]), ("action", a), ("joint_pos_target", tgt)):
                trace[key].append(np.array(value))
        if flags.get("action_noise"):
            a = a + rng.normal(0.0, flags["action_noise"], a.shape)
            tgt = default_q + scale * a
        tgt = tgt + offset
        if flags.get("drop_prob") and rng.random() < flags["drop_prob"]:  # 错过控制周期：执行器继续保持上一个目标
            tgt = target
        if flags.get("ctrl_action"):
            tgt = a
        elif flags.get("ctrl_no_default"):
            tgt = scale * a
        if flags.get("delay"):
            pending.append(tgt)
            tgt = pending.pop(0) if len(pending) > flags["delay"] else target
        target = tgt
        if flags.get("ctrl_raw_order"):
            d.ctrl[:] = target  # 错误示范：执行器的编译顺序不是关节顺序
        elif not torque_pd:
            d.ctrl[idx.act] = target
        jitter = flags.get("decim_jitter", 0)
        for _ in range(decimation + (int(rng.integers(-jitter, jitter + 1)) if jitter else 0)):
            if torque_pd:
                d.qfrc_applied[idx.v] = np.clip(kp * (target - d.qpos[idx.q]) - kd * d.qvel[idx.v], -effort, effort)
            mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos).all() or np.abs(d.qvel).max() > 1e4:  # 数值发散，不记这一步
            fail_step = f + 1
            break
        root = d.qpos[idx.root_q:idx.root_q + 7]
        R = quat_to_mat(root[3:7])
        for key, value in (("p", root[:3]), ("quat", root[3:7]), ("v", d.qvel[idx.root_v:idx.root_v + 3]),
                           ("w", R @ d.qvel[idx.root_v + 3:idx.root_v + 6]), ("q", d.qpos[idx.q]), ("qd", d.qvel[idx.v]),
                           ("a", a), ("a_prev", a_prev)):
            out[key].append(np.array(value, dtype=np.float64))
        last_action = {"scaled": scale * a, "target": target, "zero": a * 0}.get(flags.get("last_action"), a)
        a_prev = last_action
        e_z, e_tilt = abs(ref.p[f + 1, 2] - root[2]), abs(ref.R[f + 1, 2, 2] - R[2, 2])
        if e_z > TERM_HEIGHT or e_tilt > TERM_TILT:
            fail_step = f + 1
            if stop_on_fail:
                break
    return {k: np.array(v) for k, v in out.items()}, fail_step, {k: np.array(v) for k, v in trace.items()}


class Scorer:
    """附录F.1的奖励与终止（supermimic.MDP），同时报告不依赖奖励尺度的误差。"""

    def __init__(self, source):
        sys.path.insert(0, str(REPO))
        import torch

        from supermimic.tasks.tracking.mdp import MAX_STEP_REWARD, MDP
        from supermimic.tasks.tracking.motion import MotionLib
        from supermimic.utils.kinematics import G1Kinematics

        self.torch, self.max_r = torch, MAX_STEP_REWARD
        kin = G1Kinematics("cpu")
        self.motion = MotionLib.from_source(source, kin, "cpu")
        self.mdp = MDP(kin, self.motion, kin.default_q)

    def goal_gap(self, goals, K):
        """本实现算出的目标与训练用的MotionLib.goal的最大差。"""
        f = self.torch.arange(len(goals))
        return float((self.motion.goal(f, K).numpy() - goals).__abs__().max())

    def __call__(self, traj, fail_step):
        torch = self.torch
        t = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in traj.items()}
        n = len(t["p"])
        if n == 0:
            return {k: float("nan") for k in ("norm_return", "body_err_cm", "root_err_cm", "joint_err_deg", "action_rate")}
        f1 = torch.arange(1, n + 1)
        S = (t["p"], torch.as_tensor(quat_to_mat(traj["quat"]), dtype=torch.float32), t["v"], t["w"], t["q"], t["qd"])
        r, terms = self.mdp.reward(S, f1, t["a"], t["a_prev"], terms=True)
        T = int(self.motion.clip_len[0]) - 1
        return {"norm_return": float(r.sum()) / (T * self.max_r),
                "body_err_cm": 100 * float(terms["e_body_pos"].sqrt().mean()),
                "root_err_cm": 100 * float(terms["e_root_pos"].sqrt().mean()),
                "joint_err_deg": float((t["q"] - self.motion.q[f1]).abs().mean().rad2deg()),
                "action_rate": float(terms["action_rate"].mean())}


def parse_prop(value):
    """metadata_props都是字符串：列表、字典和数字是JSON，其余（如exported_at）是原文。"""
    try:
        return json.loads(value)
    except ValueError:
        return value


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx")
    parser.add_argument("motion", help="npz with root_pos, root_quat_wxyz, joint_pos at the policy's control rate")
    parser.add_argument("--model", default=None, help="compiled .mjb; default: build the training scene with mjlab")
    parser.add_argument("--save_model", default=None, help="save the training scene as .mjb and continue")
    parser.add_argument("--score_source", default=None, help="merged HDF5 of the same motion: score with the training reward")
    parser.add_argument("--variants", nargs="*", default=["baseline"], help="'all', 'stress', or names of the VARIANTS in this file")
    parser.add_argument("--seeds", type=int, default=5, help="runs of each variant that has randomness")
    parser.add_argument("--extra_variants", default=None, help="JSON file {name: flags}: more variants, selected with the group name 'extra'")
    parser.add_argument("--out", default=None, help="write the results as CSV")
    parser.add_argument("--record", default=None, help="save the first variant's trajectory as npz")
    parser.add_argument("--golden", default=None, help="save the baseline's states, inputs and outputs of the first steps as npz")
    parser.add_argument("--golden_steps", type=int, default=500)
    parser.add_argument("--threads", type=int, default=2, help="onnxruntime intra-op threads")
    args = parser.parse_args(argv)

    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    session = ort.InferenceSession(args.onnx, options, providers=["CPUExecutionProvider"])
    props = session.get_modelmeta().custom_metadata_map
    meta = {k: parse_prop(v) for k, v in props.items()}
    m0 = mujoco.MjModel.from_binary_path(args.model) if args.model else training_model()
    if args.save_model:
        mujoco.mj_saveModel(m0, args.save_model, None)
    ref = Reference(args.motion, meta)
    scorer = Scorer(args.score_source) if args.score_source else None
    if scorer:
        print(f"goal vs MotionLib.goal: max abs diff {scorer.goal_gap(ref.goals(meta), meta['goal_frames']):.2e}", flush=True)
    extra = json.loads(Path(args.extra_variants).read_text(encoding="utf-8")) if args.extra_variants else {}
    VARIANTS.update(extra)
    groups = {"all": [n for n in VARIANTS if n not in STRESS and n not in extra], "stress": ["baseline", *STRESS], "extra": ["baseline", *extra]}
    names = [n for v in args.variants for n in groups.get(v, [v])]
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        parser.error(f"unknown variants: {unknown}")
    if args.golden:
        _, _, trace = rollout(session, meta, ref, m0, trace_steps=args.golden_steps)
        np.savez_compressed(args.golden, control_dt=meta["control_dt"], joint_names=np.array(meta["joint_names"]), **trace)
        print(f"golden: {len(trace['obs'])} steps in {args.golden}", flush=True)
    rows = []
    runs = [(n, seed) for n in names for seed in range(args.seeds if VARIANTS[n].get("random") else 1)]
    for i, (name, seed) in enumerate(runs):
        t0 = time.time()
        traj, fail_step, _ = rollout(session, meta, ref, m0, VARIANTS[name], seed=seed)
        row = {"variant": name, "seed": seed, "failed": fail_step is not None,
               "survived_s": round((fail_step if fail_step else ref.T - 1) * meta["control_dt"], 2)}
        if scorer:
            row.update({k: round(v, 4) for k, v in scorer(traj, fail_step).items()})
        row["wall_s"] = round(time.time() - t0, 1)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if args.record and i == 0:
            np.savez_compressed(args.record, fail_step=-1 if fail_step is None else fail_step, **traj)
        if args.out:
            with open(args.out, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    return rows


if __name__ == "__main__":
    main()
