"""最小推理：用导出的ONNX策略在CPU MuJoCo里闭环跟踪一段动作。只依赖numpy、mujoco和onnxruntime。

PD增益、电枢惯量、步长、关节顺序都读ONNX元数据。模型用mjlab自带的g1.xml，但这个文件本身步长2 ms、Euler积分、电枢惯量为0，
没有执行器和地面，这些都是训练时由mjlab在运行时加上的，build_model逐项补齐。评分与各种部署偏差的对比见sim2sim_onnx.py。
    python infer_onnx_minimal.py POLICY.onnx MOTION.npz [--xml path/to/g1.xml] [--viewer] [--save traj.npz]
装了mjlab时可以省略--xml。
"""
import argparse
import importlib.util
import json
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import onnxruntime as ort

META_KEYS = ("joint_names", "default_joint_pos", "joint_stiffness", "joint_damping", "effort_limit", "armature",
             "control_dt", "physics_dt", "decimation", "obs_history", "goal_frames")
MAX_Z_ERR, MAX_TILT_ERR = 0.25, 0.8  # 训练的终止条件
FOOT_GEOM = re.compile(r"^(left|right)_foot[1-7]_collision$")
REPORT_STEPS = 250  # 每5秒打印一次误差


def quat_to_mat(q):
    """wxyz四元数转旋转矩阵。"""
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


def rot6d(R):
    """前两列按行展平：[R00, R01, R10, R11, R20, R21]。"""
    return R[..., :2].reshape(R.shape[:-2] + (6,))


def yaw_only(R):
    """只保留偏航的朝向：把x轴投到水平面。"""
    fx, fy = R[..., 0, 0], R[..., 1, 0]
    n = np.sqrt(fx * fx + fy * fy + 1e-9)
    c, s, zero, one = fx / n, fy / n, np.zeros_like(fx), np.ones_like(fx)
    return np.stack((c, -s, zero, s, c, zero, zero, zero, one), -1).reshape(R.shape[:-2] + (3, 3))


def load_motion(path, meta):
    """读npz（root_pos、root_quat_wxyz、joint_pos），用差分算参考速度：内部中心差分，两端单边差分。"""
    dt = meta["control_dt"]
    data = np.load(path, allow_pickle=False)
    if "fps" in data and abs(float(data["fps"]) * dt - 1) > 1e-3:
        sys.exit(f"{path}: fps {data['fps']} does not match control_dt {dt}")
    if "joint_names" in data and [str(n) for n in data["joint_names"]] != meta["joint_names"]:
        sys.exit(f"{path}: joint order differs from the policy's joint_names")
    p, quat, q = (np.asarray(data[k], dtype=np.float64) for k in ("root_pos", "root_quat_wxyz", "joint_pos"))
    R = quat_to_mat(quat)
    w = np.zeros_like(p)  # 世界系
    w[1:-1] = so3_log(R[2:] @ R[:-2].transpose(0, 2, 1)) / (2 * dt)
    w[0], w[-1] = so3_log(R[1] @ R[0].T) / dt, so3_log(R[-1] @ R[-2].T) / dt
    return SimpleNamespace(T=len(p), p=p, R=R, quat=quat, q=q, v=np.gradient(p, dt, axis=0), w=w, qd=np.gradient(q, dt, axis=0))


def make_goals(ref, K):
    """整段动作每一帧f的目标（T, K*73）：f+1..f+K按帧拼接，超过末帧取末帧。只依赖参考，可以预先算好。"""
    fut = np.minimum(np.arange(ref.T)[:, None] + np.arange(1, K + 1), ref.T - 1)
    Rt = ref.R.transpose(0, 2, 1)
    Ht = yaw_only(ref.R).transpose(0, 2, 1)[:, None]  # 当前参考帧的偏航
    return np.concatenate(((Ht @ (ref.p[fut] - ref.p[:, None])[..., None])[..., 0],  # 3
                           rot6d(Ht @ ref.R[fut]),                                   # 6
                           (Rt[fut] @ ref.v[fut][..., None])[..., 0],                # 3 参考根坐标系
                           (Rt[fut] @ ref.w[fut][..., None])[..., 0],                # 3 参考根坐标系
                           ref.q[fut],                                               # 29 绝对关节角
                           0.1 * ref.qd[fut]), -1).reshape(ref.T, -1).astype(np.float32)  # 29


def build_model(xml, meta):
    """g1.xml加上训练时mjlab在运行时加的东西。足部摩擦取评估用的1.0（训练时在0.3～1.2随机）。"""
    spec = mujoco.MjSpec.from_file(str(xml))
    spec.option.timestep = meta["physics_dt"]
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    spec.option.iterations, spec.option.ls_iterations, spec.option.ccd_iterations = 10, 20, 50
    for geom in spec.geoms:
        foot, collide = FOOT_GEOM.match(geom.name), geom.name.endswith("_collision")
        geom.contype = geom.conaffinity = 1 if collide else 0
        geom.condim, geom.priority = (3, 1) if foot else (1, 0)
        if foot:
            geom.friction[0] = 1.0
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05])
    for name, kp, kd, effort, armature in zip(meta["joint_names"], meta["joint_stiffness"], meta["joint_damping"],
                                              meta["effort_limit"], meta["armature"]):
        spec.joint(name).armature = armature
        act = spec.add_actuator(name=name, target=name)  # 按策略的关节顺序添加，所以data.ctrl的顺序就是策略的顺序
        act.trntype, act.biastype = mujoco.mjtTrn.mjTRN_JOINT, mujoco.mjtBias.mjBIAS_AFFINE
        act.gainprm[0], act.biasprm[1], act.biasprm[2] = kp, -kp, -kd  # 力 = kp*(ctrl - q) - kd*qd
        act.inheritrange, act.ctrllimited = 0.0, False  # 目标可以超出关节限位，力只由forcerange限幅
        act.forcelimited = True
        act.forcerange[:] = -effort, effort
    return spec.compile()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx")
    parser.add_argument("motion", help="npz with root_pos, root_quat_wxyz, joint_pos at the policy's control rate")
    parser.add_argument("--xml", default=None, help="mjlab's g1.xml; default: find it in the installed mjlab")
    parser.add_argument("--viewer", action="store_true", help="watch in real time; needs a display")
    parser.add_argument("--save", default=None, help="save qpos, qvel and action of every control step as npz")
    args = parser.parse_args(argv)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2  # 默认按整机核数开线程，在容器里或CPU紧张时单步可能超过20 ms
    session = ort.InferenceSession(args.onnx, options, providers=["CPUExecutionProvider"])
    props = session.get_modelmeta().custom_metadata_map
    meta = {k: json.loads(props[k]) for k in META_KEYS}
    xml = args.xml
    if xml is None:
        found = importlib.util.find_spec("mjlab")
        if found is None:
            sys.exit("mjlab is not installed: pass --xml path/to/g1.xml")
        xml = Path(found.origin).parent / "asset_zoo/robots/unitree_g1/xmls/g1.xml"
    m = build_model(xml, meta)
    d = mujoco.MjData(m)
    ref = load_motion(args.motion, meta)
    goals = make_goals(ref, meta["goal_frames"])
    names, default_q = meta["joint_names"], np.array(meta["default_joint_pos"])
    hist_len, decimation, dt = meta["obs_history"], meta["decimation"], meta["control_dt"]
    jnt = np.array([m.joint(n).id for n in names])
    qadr, vadr = m.jnt_qposadr[jnt], m.jnt_dofadr[jnt]

    # 初始状态：完全放在参考首帧（含速度），根在qpos[:7]、qvel[:6]；自由关节的角速度在机体系
    d.qpos[:3], d.qpos[3:7] = ref.p[0], ref.quat[0] / np.linalg.norm(ref.quat[0])
    d.qvel[:3], d.qvel[3:6] = ref.v[0], ref.R[0].T @ ref.w[0]
    d.qpos[qadr], d.qvel[vadr] = np.clip(ref.q[0], m.jnt_range[jnt, 0], m.jnt_range[jnt, 1]), ref.qd[0]
    d.ctrl[:] = d.qpos[qadr]
    mujoco.mj_forward(m, d)

    viewer = None
    if args.viewer:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(m, d)
    hist, a_prev = None, np.zeros(len(names))
    log, window, wall, fell_at = {"qpos": [], "qvel": [], "action": []}, [], 0.0, None
    for f in range(ref.T - 1):
        t0 = time.perf_counter()
        R = quat_to_mat(d.qpos[3:7])
        obs = np.concatenate((d.qpos[qadr] - default_q,    # 29
                              d.qvel[vadr],                # 29 不缩放
                              d.qvel[3:6],                 # 3 机体系角速度，不缩放
                              -R[2],                       # 3 机体系重力方向 R^T (0, 0, -1)
                              a_prev,                      # 29 上一步的action（未乘action_scale）
                              rot6d(R.T @ ref.R[f]))       # 6
                             ).astype(np.float32)
        # 最旧在前；第一步用当前观测填满
        hist = np.tile(obs, (hist_len, 1)) if hist is None else np.concatenate((hist[1:], obs[None]))
        action, target = session.run(None, {"obs_hist": hist[None], "goal": goals[f][None]})
        d.ctrl[:] = target[0]  # 一个控制周期内保持不变
        for _ in range(decimation):
            mujoco.mj_step(m, d)
        a_prev = action[0]
        wall += time.perf_counter() - t0

        R = quat_to_mat(d.qpos[3:7])
        z_err, tilt_err = abs(ref.p[f + 1, 2] - d.qpos[2]), abs(ref.R[f + 1, 2, 2] - R[2, 2])
        window.append((z_err, tilt_err, np.linalg.norm(ref.p[f + 1, :2] - d.qpos[:2]), np.abs(ref.q[f + 1] - d.qpos[qadr]).mean()))
        if args.save:
            for key, value in (("qpos", d.qpos), ("qvel", d.qvel), ("action", a_prev)):
                log[key].append(np.array(value))
        fell = not (z_err <= MAX_Z_ERR and tilt_err <= MAX_TILT_ERR)  # 写成这样，NaN也算摔倒
        if fell or len(window) == REPORT_STEPS or f == ref.T - 2:
            e = np.array(window)
            print(f"t={(f + 1) * dt:7.2f}s  z_err_max {e[:, 0].max():.3f} m  tilt_err_max {e[:, 1].max():.3f}  "
                  f"root_xy_err {e[:, 2].mean():.3f} m  joint_err {np.rad2deg(e[:, 3].mean()):.2f} deg", flush=True)
            window = []
        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync()
            time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
        if fell:
            fell_at = (f + 1) * dt
            break

    total = (ref.T - 1) * dt
    print(f"FELL at t={fell_at:.2f}s of {total:.2f}s" if fell_at else f"ran {(f + 1) * dt:.2f}s of {total:.2f}s without falling")
    print(f"mean control step {1000 * wall / (f + 1):.2f} ms (limit for real time: {1000 * dt:.0f} ms)")
    if args.save:
        np.savez_compressed(args.save, fell_at=-1.0 if fell_at is None else fell_at, joint_names=np.array(names),
                            **{k: np.array(v) for k, v in log.items()})
    if viewer is not None:
        while viewer.is_running():  # 摔倒后停在最后一帧，关窗口退出
            time.sleep(0.1)


if __name__ == "__main__":
    main()
