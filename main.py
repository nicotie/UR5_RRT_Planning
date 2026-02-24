import os
import sys
from typing import Dict, Optional, Tuple, List

import numpy as np
import open3d as o3d
import torch
from PIL import Image
import spatialmath as sm
import mujoco
import pinocchio

from graspnetAPI import GraspGroup

from manipulator_grasp.planner.set_plan import getIk, get_traj
from pyroboplan.core.utils import (
    check_collisions_at_state,
    check_within_limits,
    get_random_collision_free_state,
)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, "graspnet_baseline", "models"))
sys.path.append(os.path.join(ROOT_DIR, "graspnet_baseline", "dataset"))
sys.path.append(os.path.join(ROOT_DIR, "graspnet_baseline", "utils"))
sys.path.append(os.path.join(ROOT_DIR, "manipulator_grasp"))

from graspnet import GraspNet, pred_decode  # noqa: E402
from collision_detector import ModelFreeCollisionDetector  # noqa: E402
from data_utils import CameraInfo, create_point_cloud_from_depth_image  # noqa: E402

from manipulator_grasp.env.grasp_env import GraspEnv
from manipulator_grasp.utils import mj

def get_net(
    checkpoint_path: str = "graspnet_baseline/ckp/checkpoint-rs.tar",
) -> GraspNet:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = GraspNet(
        input_feature_dim=0,
        num_view=300,
        num_angle=12,
        num_depth=4,
        cylinder_radius=0.05,
        hmin=-0.02,
        hmax_list=[0.01, 0.02, 0.03, 0.04],
        is_training=False,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()
    return net

def _normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    """Ensure rgb is float32 in [0, 1]."""
    if rgb.dtype == np.uint8:
        return rgb.astype(np.float32) / 255.0
    rgb = rgb.astype(np.float32)
    if rgb.max() > 1.5:
        rgb /= 255.0
    return rgb

def get_and_process_data(
    imgs: Dict[str, np.ndarray],
    workspace_mask: Optional[np.ndarray] = None,
    num_point: int = 6000,
    fovy: float = np.pi / 4,
) -> Tuple[Dict[str, np.ndarray], o3d.geometry.PointCloud]:

    if "img" not in imgs:
        raise ValueError("imgs must contain key 'img' (RGB).")
    if "depth" not in imgs:
        raise ValueError("imgs must contain key 'depth' (GraspNet baseline needs RGB-D/point cloud).")

    color = _normalize_rgb(imgs["img"])
    depth = imgs["depth"]

    height, width = color.shape[:2]

    # --- prefer env-provided intrinsics K ---
    K = imgs.get("K", None)
    if K is not None:
        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    else:
        focal = height / (2.0 * np.tan(fovy / 2.0))
        fx = fy = focal
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0

    factor_depth = 1.0
    camera = CameraInfo(width, height, fx, fy, cx, cy, factor_depth)  # CameraInfo(width,height,fx,fy,cx,cy,scale) :contentReference[oaicite:1]{index=1}
    cloud_organized = create_point_cloud_from_depth_image(depth, camera, organized=True)  # (H,W,3)

    # --- base valid mask ---
    valid = (depth > 1e-6) & (depth < 2.0)

    # --- semantic segmentation: keep ONLY target geom points (banana) ---
    seg = imgs.get("seg", None)
    gid = int(imgs.get("obj_gid", -1))
    if seg is not None and gid >= 0:
        seg = np.asarray(seg)
        if seg.shape[0] != height or seg.shape[1] != width:
            raise ValueError(f"seg shape {seg.shape} does not match depth {depth.shape}")

        obj_mask = None
        if seg.ndim == 3 and seg.shape[-1] >= 2:
            c0, c1 = seg[..., 0], seg[..., 1]

            m0 = valid & (c0 == gid)
            m1 = valid & (c1 == gid)

            # pick the channel that matches gid more often as "id channel"
            if np.count_nonzero(m0) >= np.count_nonzero(m1):
                obj_id, obj_type = c0, c1
            else:
                obj_id, obj_type = c1, c0

            mask_id = valid & (obj_id == gid)

            # type filter: only keep when it doesn't wipe out most pixels
            GEOM = int(mujoco.mjtObj.mjOBJ_GEOM)
            mask_type = mask_id & (obj_type == GEOM)
            obj_mask = mask_type if np.count_nonzero(mask_type) > 0.5 * np.count_nonzero(mask_id) else mask_id
        else:
            # degenerate: treat seg as id map
            obj_mask = valid & (seg == gid)

        if obj_mask is None or np.count_nonzero(obj_mask) <= 20:
            raise RuntimeError(f"Segmentation found too few target pixels for obj_gid={gid}")
        valid = obj_mask

    # optional workspace constraint (still applies after obj mask)
    if workspace_mask is not None:
        valid = valid & (workspace_mask > 0)

    cloud_masked = cloud_organized[valid]
    color_masked = color[valid]

    if len(cloud_masked) == 0:
        raise RuntimeError("No valid points after masking (seg/workspace/depth).")

    if len(cloud_masked) >= num_point:
        idxs = np.random.choice(len(cloud_masked), num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), num_point - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)

    cloud_sampled = cloud_masked[idxs].astype(np.float32)
    color_sampled = color_masked[idxs].astype(np.float32)

    cloud_o3d = o3d.geometry.PointCloud()
    cloud_o3d.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud_o3d.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    end_points: Dict[str, np.ndarray] = {}
    end_points["point_clouds"] = torch.from_numpy(cloud_sampled[np.newaxis]).to(device)
    end_points["cloud_colors"] = color_sampled

    return end_points, cloud_o3d

def get_grasps(net: GraspNet, end_points: Dict[str, np.ndarray]) -> GraspGroup:
    """Forward GraspNet and decode to GraspGroup."""
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg = GraspGroup(grasp_preds[0].detach().cpu().numpy())
    return gg

def collision_detection(
    gg: GraspGroup,
    cloud_xyz: np.ndarray,
    voxel_size: float = 0.01,
    approach_dist: float = 0.05,
    collision_thresh: float = 0.01,
) -> GraspGroup:
    """Single-view collision detection post-process."""
    mfcdetector = ModelFreeCollisionDetector(cloud_xyz, voxel_size=voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=approach_dist, collision_thresh=collision_thresh)
    return gg[~collision_mask]

def _select_and_rank_grasps(
    gg: GraspGroup,
    cloud_xyz: np.ndarray,
    vertical: np.ndarray = np.array([0.0, 0.0, 1.0], dtype=np.float32),
    angle_threshold_deg: float = 45.0,
) -> List:
    """
    Apply:
      1) vertical approach filtering
      2) composite scoring (grasp.score + distance to object center)
    Return: list of Grasp objects (sorted best-first).
    """
    all_grasps = list(gg)
    angle_threshold = np.deg2rad(angle_threshold_deg)

    filtered: List = []
    for g in all_grasps:
        approach_dir = g.rotation_matrix[:, 0]
        cos_angle = float(np.clip(np.dot(approach_dir, vertical), -1.0, 1.0))
        if np.arccos(cos_angle) < angle_threshold:
            filtered.append(g)

    if len(filtered) == 0:
        filtered = all_grasps

    object_center = cloud_xyz.mean(axis=0) if len(cloud_xyz) > 0 else np.zeros(3, dtype=np.float32)
    dists = np.array([np.linalg.norm(g.translation - object_center) for g in filtered], dtype=np.float32)
    max_dist = float(dists.max()) if len(dists) else 1.0

    # composite_score = g.score * 0.4 + (1 - d/max_dist) * 0.6
    ranked = []
    for g, d in zip(filtered, dists):
        distance_score = 1.0 - float(d / max_dist)
        composite_score = float(g.score) * 0.4 + distance_score * 0.6
        ranked.append((g, composite_score))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return [g for g, _ in ranked]

def vis_grasps(grasps: List, cloud: o3d.geometry.PointCloud) -> None:
    """Visualize candidate grasps."""
    gg_vis = GraspGroup()
    for g in grasps:
        gg_vis.add(g)
    grippers = gg_vis.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud, *grippers])

def generate_grasps(
    net: GraspNet,
    imgs: Dict[str, np.ndarray],
    workspace_mask: Optional[np.ndarray] = None,
    visual: bool = False,
) -> Tuple[List, o3d.geometry.PointCloud]:
    """preprocess -> predict -> collision -> nms/sort -> select/rank."""
    end_points, cloud = get_and_process_data(imgs, workspace_mask=workspace_mask)

    gg = get_grasps(net, end_points)

    # Optional: free tensors early
    del end_points
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gg = collision_detection(gg, np.asarray(cloud.points))
    gg.nms()
    gg.sort_by_score()

    grasps = _select_and_rank_grasps(gg, np.asarray(cloud.points))

    if visual:
        vis_grasps(grasps[:10], cloud)

    return grasps, cloud

def execute_grasp(env, grasps: List, cloud: o3d.geometry.PointCloud) -> None:
    # --- constants ---
    approach = 0.15
    lift = 0.15
    success = False
    q_traj1 = q_traj2 = q_traj3 = None
    q_goal1 = q_goal2 = q_goal3 = None

    # place pose
    n_wp = np.array([0.0, 1.0, 0.0])
    o_wp = np.array([1.0, 0.0, -0.5])
    t_wp = np.array([0.65, 0.2, 0.9])
    T_wp_g = sm.SE3.Trans(t_wp) * sm.SE3(sm.SO3.TwoVectors(x=n_wp, y=o_wp))

    T_wc = env.T_wc
    T_GT = env.T_TG.inv()  # gripper -> tool

    model_roboplan, collision_model = env.get_planning_models()
    arm_joint_names = env.joint_names

    def mj_to_pin_q(mj_q: np.ndarray) -> np.ndarray:
        q = pinocchio.neutral(model_roboplan).copy()
        for jn, qv in zip(arm_joint_names, mj_q):
            jid = model_roboplan.getJointId(jn)
            q[model_roboplan.joints[jid].idx_q] = float(qv)
        return np.ascontiguousarray(q, dtype=np.float64).ravel()

    def run_traj(action: np.ndarray, q_traj: np.ndarray):
        for i in range(q_traj.shape[1]):
            action[:6] = q_traj[:6, i]
            env.step(action)

    # --- start state check ---
    q_home = mj_to_pin_q(mj.mj_get_arm_q(env, arm_joint_names))
    if not check_within_limits(model_roboplan, q_home):
        print("[execute_grasp] q_home out of joint limits")
        return
    if check_collisions_at_state(model_roboplan, collision_model, q_home):
        print("[execute_grasp] q_home in collision")
        # ---- debug: dump first collisions ----
        data = model_roboplan.createData()
        gdata = collision_model.createData()
        pinocchio.updateGeometryPlacements(model_roboplan, data, collision_model, gdata, q_home)
        pinocchio.computeCollisions(model_roboplan, data, collision_model, gdata, q_home, False)
        n = 0
        for i, pair in enumerate(collision_model.collisionPairs):
            if gdata.collisionResults[i].isCollision():
                a = collision_model.geometryObjects[pair.first].name
                b = collision_model.geometryObjects[pair.second].name
                print(f"[COL@q_home] {a} <-> {b}")
                n += 1
                if n >= 10:
                    break
        return
    
    for g in grasps:
        R = np.asarray(g.rotation_matrix, dtype=np.float64)
        t = np.asarray(g.translation, dtype=np.float64)

        T_cg = sm.SE3.Trans(t) * sm.SE3(sm.SO3.TwoVectors(x=R[:, 0], y=R[:, 1]))
        T_wG = T_wc * T_cg
        print("[grasp] z_w=", float(T_wG.t[2]),"approach_w=", (T_wG.R @ np.array([1.0,0.0,0.0])).ravel())
        # pregrasp
        T_pre_g = T_wG * sm.SE3(-approach, 0.0, 0.0)
        # T_pre_g = T_wG * sm.SE3(0.0, 0.0, approach)
        # lift: move in world +z
        T_lift_g = sm.SE3(0.0, 0.0, lift) * T_wG

        # convert gripper pose -> tool pose for IK 
        # target frame = flange
        T_pre = T_pre_g * T_GT
        T_grasp = T_wG * T_GT
        T_lift = T_lift_g * T_GT

        q_goal1 = getIk(env, q_home, T_pre)
        if q_goal1 is None:
            continue
        q_goal1 = np.ascontiguousarray(np.asarray(q_goal1, dtype=np.float64)).ravel()
        if check_collisions_at_state(model_roboplan, collision_model, q_goal1):
            continue
        q_traj1 = get_traj(env, q_home, q_goal1)
        if q_traj1 is None:
            continue

        q_goal2 = getIk(env, q_goal1, T_grasp)
        if q_goal2 is None:
            continue
        q_goal2 = np.ascontiguousarray(np.asarray(q_goal2, dtype=np.float64)).ravel()
        if check_collisions_at_state(model_roboplan, collision_model, q_goal2):
            continue
        q_traj2 = get_traj(env, q_goal1, q_goal2)
        if q_traj2 is None:
            continue

        q_goal3 = getIk(env, q_goal2, T_lift)
        if q_goal3 is None:
            continue
        q_goal3 = np.ascontiguousarray(np.asarray(q_goal3, dtype=np.float64)).ravel()
        if check_collisions_at_state(model_roboplan, collision_model, q_goal3):
            continue
        q_traj3 = get_traj(env, q_goal2, q_goal3)
        if q_traj3 is None:
            continue

        success = True
        break

    if not success:
        print("Grasp Failed")
        return

    # --- execute ---
    action = np.zeros(7, dtype=np.float32)

    # 1) approach
    run_traj(action, q_traj1)

    # 2) move to grasp + close gripper
    run_traj(action, q_traj2)
    for _ in range(1500):
        action[-1] = np.float32(min(float(action[-1]) + 0.2, 255.0))
        env.step(action)

    # 3) lift
    run_traj(action, q_traj3)
    for _ in range(50):
        env.step(action)

    # 4) move to place, then open, then back home
    T_place = T_wp_g * T_GT  # place gripper pose -> tool pose
    q_start4 = q_goal3
    while True:
        q_goal4 = getIk(env, q_start4, T_place)
        if q_goal4 is None:
            continue
        q_goal4 = np.ascontiguousarray(np.asarray(q_goal4, dtype=np.float64)).ravel()
        if check_collisions_at_state(model_roboplan, collision_model, q_goal4):
            continue
        q_traj4 = get_traj(env, q_start4, q_goal4)
        if q_traj4 is None:
            continue
        q_traj5 = get_traj(env, q_goal4, q_home)
        if q_traj5 is None:
            continue
        break

    run_traj(action, q_traj4)

    for _ in range(1500):
        action[-1] = np.float32(max(float(action[-1]) - 0.2, 0.0))
        env.step(action)
    for _ in range(50):
        env.step(action)

    run_traj(action, q_traj5)
    for _ in range(50):
        env.step(action)

def main() -> None:
    net = get_net()
    env = GraspEnv()
    env.reset()
    for _ in range(1000):
        env.step()
    imgs = env.render()
    grasps, cloud = generate_grasps(net, imgs, visual=True)
    gg = grasps[:3]
    execute_grasp(env, gg, cloud)
    for _ in range(2000):
        env.step()
    env.close()

if __name__ == "__main__":
    main()
