# Ported from zita-ch/supermimic_vibed (MIT, Copyright (c) 2026 ETH Zurich, Chong Zhang); see THIRD_PARTY_NOTICES.md.
"""Unitree G1 (29 DoF) from mjlab's asset zoo, with BeyondMimic's actuator model.

mjlab's G1 already carries BeyondMimic's drives (10 Hz natural frequency, damping ratio 2 on the
reflected motor armature) and action scale 0.25 * effort_limit / stiffness, so no URDF/USD
conversion is needed. The LAFAN1 retargets use the same 29-joint Unitree SDK order as its MJCF.
"""

from __future__ import annotations

import re

import mujoco
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
    FULL_COLLISION,
    FULL_COLLISION_WITHOUT_SELF,
    G1_ACTION_SCALE,
    KNEES_BENT_KEYFRAME,
    get_g1_robot_cfg,
)
from mjlab.entity import Entity, EntityCfg

# Joint order of the LAFAN1 retarget CSVs (Unitree SDK order), which is also mjlab's G1 order.
G1_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]  # fmt: skip

# BeyondMimic's tracked bodies; the anchor/root is the pelvis.
TRACKED_BODIES = [
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
]  # fmt: skip

FOOT_GEOMS = r"^(left|right)_foot[1-7]_collision$"


def robot_cfg(self_collision: bool = False) -> EntityCfg:
    """mjlab G1 config. Self-collisions are off by default, as in the reference implementation."""
    cfg = get_g1_robot_cfg()
    cfg.init_state = KNEES_BENT_KEYFRAME
    cfg.collisions = (FULL_COLLISION if self_collision else FULL_COLLISION_WITHOUT_SELF,)
    return cfg


def compile_robot_model() -> mujoco.MjModel:
    """Standalone MjModel of the G1 (entity-local names), used by the torch FK."""
    return Entity(robot_cfg()).compile()


def action_scale(joint_name: str) -> float:
    for pat, val in G1_ACTION_SCALE.items():
        if re.fullmatch(pat, joint_name):
            return val
    raise KeyError(joint_name)


def default_joint_pos_list() -> list[float]:
    out = []
    for name in G1_JOINT_NAMES:
        val = 0.0
        for pat, v in KNEES_BENT_KEYFRAME.joint_pos.items():
            if re.fullmatch(pat, name):
                val = v
        out.append(val)
    return out
