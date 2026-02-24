from typing import List, Optional, Tuple, Union
import numpy as np
import spatialmath as sm
import spatialmath.base as smb
import mujoco as mj
import pinocchio
from .rtb import make_tf


def set_body_pose(model: mj.MjModel, data: mj.MjData, body_name: Union[int, str], xpos: np.ndarray) -> None:
    body_id = (
        body_name if isinstance(body_name, int) else mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
    )
    data.body(body_id).xpos = xpos[:3]

def get_body_pose(model: mj.MjModel, data: mj.MjData, body_name: Union[int, str]) -> sm.SE3:
    body_id = (
        body_name if isinstance(body_name, int) else mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
    )
    t = data.body(body_id).xpos
    q = data.body(body_id).xquat
    return make_tf(pos=t, ori=q)

def mj_pose_body_or_site(model, data, name: str) -> sm.SE3:
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
    if bid != -1:
        return get_body_pose(model, data, bid)

    sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, name)
    if sid != -1:
        t = data.site(sid).xpos.copy()
        R = data.site(sid).xmat.reshape(3, 3).copy()
        return sm.SE3.Rt(R, t)

    raise RuntimeError(f"MuJoCo body/site '{name}' not found (mj_name2id returned -1).")

def world_pose_SE3(m: mj.MjModel, d: mj.MjData, name: str) -> pinocchio.SE3:
    bid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, name)
    if bid >= 0:
        t = np.asarray(d.body(bid).xpos).copy()
        R = np.asarray(d.body(bid).xmat).reshape(3, 3).copy()
        return pinocchio.SE3(R, t)

    sid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_SITE, name)
    if sid >= 0:
        t = np.asarray(d.site(sid).xpos).copy()
        R = np.asarray(d.site(sid).xmat).reshape(3, 3).copy()
        return pinocchio.SE3(R, t)

    raise RuntimeError(f"MuJoCo body/site '{name}' not found")

def pick_existing(model, names, objtype) -> str:
    for n in names:
        if mj.mj_name2id(model, objtype, n) != -1:
            return n
    return ""

def rot_angle(R: np.ndarray) -> float:
    # angle of rotation matrix (0..pi)
    tr = float(np.trace(R))
    c = max(-1.0, min(1.0, (tr - 1.0) * 0.5))
    return float(np.arccos(c))

def se3_err(A: sm.SE3, B: sm.SE3) -> tuple[float, float]:
    dT = A.inv() * B
    dt = float(np.linalg.norm(dT.t))
    ang = rot_angle(np.asarray(dT.R, dtype=np.float64))
    return dt, ang

def so3_angle(R: np.ndarray) -> float:
    return float(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))

def pin_frame_pose(model, q: np.ndarray, frame_name: str) -> sm.SE3:
    data = model.createData()
    pinocchio.framesForwardKinematics(model, data, q)
    fid = model.getFrameId(frame_name)
    oMf = data.oMf[fid]
    return sm.SE3.Rt(np.asarray(oMf.rotation), np.asarray(oMf.translation))

def mj_joint_qposadr(m, joint_name: str) -> int:
    jid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, joint_name)  # name->id :contentReference[oaicite:1]{index=1}
    if jid < 0:
        raise RuntimeError(f"[mj] joint not found: {joint_name}")
    return int(m.jnt_qposadr[jid])  # qpos address :contentReference[oaicite:2]{index=2}

def mj_get_arm_q(env, joint_names: list[str]) -> np.ndarray:
    return np.array([env.mj_data.qpos[mj_joint_qposadr(env.mj_model, jn)] for jn in joint_names], dtype=np.float64)

def mj_fk_pose_at_q(env, joint_names: list[str], q_arm: np.ndarray, body_or_site: str) -> sm.SE3:
    # temporary overwrite qpos -> mj_forward -> read pose -> restore
    qpos0 = env.mj_data.qpos.copy()
    for jn, qv in zip(joint_names, q_arm):
        env.mj_data.qpos[mj_joint_qposadr(env.mj_model, jn)] = float(qv)
    mj.mj_forward(env.mj_model, env.mj_data)  # forward kinematics/dynamics update :contentReference[oaicite:3]{index=3}
    Tw = mj.mj_pose_body_or_site(env.mj_model, env.mj_data, body_or_site)
    env.mj_data.qpos[:] = qpos0
    mj.mj_forward(env.mj_model, env.mj_data)
    return Tw

def set_joint_q(model: mj.MjModel, data: mj.MjData, joint_name: Union[int, str], q: Union[np.ndarray, float],
                unit: str = "rad") -> None:
    joint_id = (
        joint_name if isinstance(joint_name, int) else mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)
    )

    if unit == 'deg':
        q = np.deg2rad(q)

    q_inds = get_joint_qpos_inds(model, data, joint_id)
    data.qpos[q_inds] = q


def get_joint_q(model: mj.MjModel, data: mj.MjData, joint_name: Union[int, str]) -> np.ndarray:
    joint_id = (
        joint_name if isinstance(joint_name, int) else mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)
    )
    q_inds = get_joint_qpos_inds(model, data, joint_id)
    return data.qpos[q_inds]


def get_joint_qpos_inds(model: mj.MjModel, data: mj.MjData, joint_name: Union[int, str]) -> np.ndarray:
    joint_id = (
        joint_name if isinstance(joint_name, int) else mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)
    )
    addr = get_joint_qpos_addr(model, joint_id)
    joint_dim = get_joint_dim(model, data, joint_id)
    return np.array(range(addr, addr + joint_dim))


def get_joint_qpos_addr(model: mj.MjModel, joint_name: Union[int, str]) -> int:
    joint_id = (
        joint_name if isinstance(joint_name, int) else mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)
    )
    return model.jnt_qposadr[joint_id]


def get_joint_dim(model: mj.MjModel, data: mj.MjData, joint_name: Union[str, int]) -> int:
    joint_id = (
        joint_name if isinstance(joint_name, int) else mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)
    )
    return len(data.joint(joint_id).qpos)


def set_free_joint_pose(model: mj.MjModel, data: mj.MjData, joint_name: Union[int, str], T: sm.SE3) -> None:
    t = T.t
    q = sm.base.r2q(T.R).data
    T_new = np.append(t, q)
    set_joint_q(model, data, joint_name, T_new)


def attach(model: mj.MjModel, data: mj.MjData, equality_name: str, free_joint_name: str, T: sm.SE3,
           eq_data=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
           eq_solimp: np.ndarray = np.array([[0.99, 0.99, 0.001, 0.5, 1]]),
           eq_solref: np.ndarray = np.array([0.0001, 1])
           ) -> None:
    eq_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_EQUALITY, equality_name)

    if eq_id is None:
        raise ValueError(
            f"Equality constraint with name '{equality_name}' not found in the model."
        )

    data.eq_active[eq_id] = 0

    set_free_joint_pose(model, data, free_joint_name, T)

    model.equality(equality_name).data = eq_data

    model.equality(equality_name).solimp = eq_solimp
    model.equality(equality_name).solref = eq_solref

    data.eq_active[eq_id] = 1

# def attach(model: mj.MjModel, data: mj.MjData, equality_name: str, free_joint_name: str, T: sm.SE3,
#            eq_data=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=float),
#            eq_solimp: np.ndarray = np.array([0.99, 0.99, 0.001, 0.5, 1.0], dtype=float),
#            eq_solref: np.ndarray = np.array([0.0001, 1.0], dtype=float),
#            ) -> None:
#     eq_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_EQUALITY, equality_name)
#     if eq_id < 0:
#         raise ValueError(f"Equality constraint '{equality_name}' not found")

#     # disable first
#     data.eq_active[eq_id] = 0

#     # move gripper free joint to target pose
#     set_free_joint_pose(model, data, free_joint_name, T)

#     eq_data = np.asarray(eq_data, dtype=float).reshape(7)
#     eq_solimp = np.asarray(eq_solimp, dtype=float).reshape(5)
#     eq_solref = np.asarray(eq_solref, dtype=float).reshape(2)

#     if model.eq_data.ndim == 2:
#         model.eq_data[eq_id, :] = 0.0
#         model.eq_data[eq_id, :7] = eq_data
#     else:
#         model.eq_data[eq_id*model.eq_data.shape[0] : eq_id*model.eq_data.shape[0] + 7] = eq_data  # 兜底

#     if model.eq_solimp.ndim == 2:
#         model.eq_solimp[eq_id, :] = eq_solimp
#     else:
#         model.eq_solimp[eq_id*5 : eq_id*5 + 5] = eq_solimp

#     if model.eq_solref.ndim == 2:
#         model.eq_solref[eq_id, :] = eq_solref
#     else:
#         model.eq_solref[eq_id*2 : eq_id*2 + 2] = eq_solref

#     # enable
#     data.eq_active[eq_id] = 1  # runtime toggle is via mjData.eq_active :contentReference[oaicite:1]{index=1}

#     # after editing model constants, recompute dependent constants + update kinematics
#     mj.mj_setConst(model, data)  # common fix after changing model fields :contentReference[oaicite:2]{index=2}
#     mj.mj_forward(model, data)   # put mjData into valid state :contentReference[oaicite:3]{index=3}