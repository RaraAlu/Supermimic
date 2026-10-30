"""把record_clips.py录下的轨迹离屏渲染成MP4：实体是策略在F.1条件下的执行，半透明青色虚影是参考动作，两者叠加对比。

每段一个文件，50 fps逐控制步输出；相机平滑跟随骨盆，每段末尾停留--hold秒。顶部字幕为片段名、类别、时间、检查点
与是否失败，需要能显示中文的字体（--font，默认依次尝试Windows、macOS与Linux的常见中文字体）。在仓库根目录运行：
    python scripts/render_clips.py traj.npz OUT_DIR [--width 1280 --height 720]
"""
import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supermimic.tasks.tracking.constants import G1_JOINT_NAMES

FONTS = (r"C:\Windows\Fonts\msyh.ttc", "/System/Library/Fonts/PingFang.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
         "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc")

parser = argparse.ArgumentParser()
parser.add_argument("npz", help="trajectories written by record_clips.py")
parser.add_argument("out_dir")
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--hold", type=float, default=1.0, help="seconds held on the last frame")
parser.add_argument("--prefix", default="", help="prefix of the output file names")
parser.add_argument("--limit", type=int, default=None, help="render only the first n clips (a trial render)")
parser.add_argument("--font", default=None, help="TrueType font for the captions; default: the first of FONTS that exists")
args = parser.parse_args()

DT = 0.02  # 控制步长（50 Hz），视频按50 fps逐帧输出
GHOST_RGBA = np.array([0.1, 0.85, 1.0, 0.35], dtype=np.float32)
data = np.load(args.npz, allow_pickle=False)
names = [str(n) for n in data["names"]][:args.limit]
iteration = int(data["iteration"]) if "iteration" in data else None

spec = get_spec()
spec.visual.global_.offwidth, spec.visual.global_.offheight = args.width, args.height
tex = spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                       rgb1=[0.32, 0.36, 0.40], rgb2=[0.22, 0.25, 0.28], width=512, height=512)
mat = spec.add_material(name="grid", texrepeat=[1, 1], texuniform=True)
mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
spec.worldbody.add_geom(name="floor_viz", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], material="grid")
spec.worldbody.add_light(pos=[0, 0, 5], dir=[0, 0, -1], diffuse=[0.7, 0.7, 0.7])
spec.visual.headlight.ambient, spec.visual.headlight.diffuse = [0.35] * 3, [0.6] * 3  # 默认偏暗
m = spec.compile()
jadr = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in G1_JOINT_NAMES])
d, d_ref = mujoco.MjData(m), mujoco.MjData(m)
vopt, pert = mujoco.MjvOption(), mujoco.MjvPerturb()
renderer = mujoco.Renderer(m, args.height, args.width, max_geom=4000)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.distance, cam.elevation, cam.azimuth = 4.0, -12.0, 135.0
font_path = args.font or next((p for p in FONTS if Path(p).exists()), None)
if font_path:
    font, small = ImageFont.truetype(font_path, 26), ImageFont.truetype(font_path, 20)
else:
    print("no CJK font found (pass --font); captions fall back to PIL's default font", flush=True)
    font = small = ImageFont.load_default()


def put(dd, p, quat, q):
    dd.qpos[:3], dd.qpos[3:7], dd.qpos[jadr] = p, quat, q
    mujoco.mj_kinematics(m, dd)


out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)
for i, name in enumerate(names):
    rob_p, rob_quat, rob_q = data[f"{i}_rob_p"], data[f"{i}_rob_quat"], data[f"{i}_rob_q"]
    ref_p, ref_quat, ref_q = data[f"{i}_ref_p"], data[f"{i}_ref_quat"], data[f"{i}_ref_q"]
    failed = bool(data[f"{i}_failed"])
    last = len(ref_p) - 1  # 参考在第last步结束，只播放到这里
    stem = name.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    path = out / f"{args.prefix}{i + 1:02d}_{stem}.mp4"
    w = imageio.get_writer(path, fps=round(1 / DT), codec="libx264", quality=8, macro_block_size=8,
                           ffmpeg_params=["-pix_fmt", "yuv420p"])
    look = rob_p[0].astype(np.float64).copy()
    for k in range(last + 1 + int(args.hold / DT)):
        kk = min(k, last)
        rk = min(kk, len(rob_p) - 1)
        put(d, rob_p[rk], rob_quat[rk], rob_q[rk])
        put(d_ref, ref_p[kk], ref_quat[kk], ref_q[kk])
        target = np.array([rob_p[rk][0], rob_p[rk][1], np.clip(rob_p[rk][2], 0.5, 0.9)])
        look += 0.08 * (target - look) if k else 0.0  # 平滑跟随，翻转时画面不抖
        cam.lookat[:] = look
        renderer.update_scene(d, camera=cam)
        scn = renderer.scene
        n0 = scn.ngeom
        mujoco.mjv_addGeoms(m, d_ref, vopt, pert, mujoco.mjtCatBit.mjCAT_DYNAMIC, scn)
        for g in range(n0, scn.ngeom):
            scn.geoms[g].rgba[:] = GHOST_RGBA
            scn.geoms[g].matid = -1
        img = Image.fromarray(renderer.render())
        dr = ImageDraw.Draw(img)
        dr.rectangle([0, 0, args.width, 78], fill=(0, 0, 0))
        dr.text((18, 8), stem, font=font, fill=(255, 255, 255))
        info = f"{name.split('/')[0]}   {kk * DT:5.2f} / {last * DT:.2f} s   实体：策略   青色虚影：参考"
        info += f"   检查点 {iteration // 1000}k" if iteration else ""
        info += "   （F.1失败）" if failed else ""
        dr.text((18, 44), info, font=small, fill=(190, 220, 235))
        w.append_data(np.asarray(img))
    w.close()
    print("wrote", path, f"{last * DT:.1f}s failed={failed}", flush=True)
