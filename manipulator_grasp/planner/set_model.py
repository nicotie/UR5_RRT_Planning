import os
import numpy as np
import pinocchio
import hppfcl as coal

from pyroboplan.core.utils import set_collisions
from pyroboplan.ik.differential_ik import DifferentialIk, DifferentialIkOptions
from pyroboplan.planning.rrt import RRTPlannerOptions

OBSTACLES = [
    "ground_plane",
    "obstacle_box_1",
    # "obstacle_box_2",
    # "obstacle_box_3",
    "obstacle_sphere_1",
    "obstacle_sphere_2",
    "obstacle_sphere_3",
]

def load_models(model_path):
    package_dirs = os.path.dirname(model_path)
    model = pinocchio.buildModelFromUrdf(model_path)
    collision_model = pinocchio.buildGeomFromUrdf(
        model, model_path, pinocchio.GeometryType.COLLISION, package_dirs=package_dirs
    )
    visual_model = pinocchio.buildGeomFromUrdf(
        model, model_path, pinocchio.GeometryType.VISUAL, package_dirs=package_dirs
    )
    qref = pinocchio.neutral(model)
    lock = []
    for jn in model.names[1:]:
        if jn.startswith("robotiq_85_"):
            jid = model.getJointId(jn)
            if model.joints[jid].nq > 0:
                lock.append(jid)
    if lock:
        model, (collision_model, visual_model) = pinocchio.buildReducedModel(
            model, [collision_model, visual_model], lock, qref
        )
    return model, collision_model, visual_model

def add_self_collisions(model, collision_model, srdf_path):
    if os.path.exists(srdf_path):
        collision_model.addAllCollisionPairs()
        pinocchio.removeCollisionPairs(model, collision_model, srdf_path)
    else:
        print(f"警告: SRDF文件不存在 {srdf_path}")

def geom_names_with_prefix(collision_model, prefix: str) -> list[str]:
    return [go.name for go in collision_model.geometryObjects if go.name.startswith(prefix)]

def collect_ignore_q_indices(model: pinocchio.Model, joint_prefix: str) -> list[int]:
    ignore = []
    for jid in range(1, model.njoints):
        jname = model.names[jid]
        if not jname.startswith(joint_prefix):
            continue
        nq = model.joints[jid].nq
        if nq <= 0:
            continue
        q0 = model.idx_qs[jid]
        ignore.extend(range(q0, q0 + nq))
    return ignore

def add_object_collisions(model, collision_model, visual_model, inflation_radius=0.0):
    obs_vis = []
    obs_col = []
    # Add the collision objects
    ground_plane = pinocchio.GeometryObject(
        "ground_plane",
        0,
        coal.Box(1.6, 1.2, 0.1),
        pinocchio.SE3(np.eye(3), np.array([0.8, 0.6, 0.685])), # 0.69
    )
    ground_plane.meshColor = np.array([0.5, 0.5, 0.5, 0.5])
    obs_vis.append(visual_model.addGeometryObject(ground_plane))
    obs_col.append(collision_model.addGeometryObject(ground_plane))

    obstacle_sphere_1 = pinocchio.GeometryObject(
        "obstacle_sphere_1",
        0,
        coal.Sphere(0.1 + inflation_radius),
        pinocchio.SE3(np.eye(3), np.array([0.6, 0.6, 0.85])),
    )
    obstacle_sphere_1.meshColor = np.array([0.0, 1.0, 0.0, 0.5])
    obs_vis.append(visual_model.addGeometryObject(obstacle_sphere_1))
    obs_col.append(collision_model.addGeometryObject(obstacle_sphere_1))

    obstacle_sphere_2 = pinocchio.GeometryObject(
        "obstacle_sphere_2",
        0,
        coal.Sphere(0.15 + inflation_radius),
        pinocchio.SE3(np.eye(3), np.array([0.6, 0.6, 1.5])),
    )
    obstacle_sphere_2.meshColor = np.array([1.0, 1.0, 0.0, 0.5])
    obs_vis.append(visual_model.addGeometryObject(obstacle_sphere_2))
    obs_col.append(collision_model.addGeometryObject(obstacle_sphere_2))

    obstacle_sphere_3= pinocchio.GeometryObject(
        "obstacle_sphere_3",
        0,
        coal.Sphere(0.1 + inflation_radius),
        pinocchio.SE3(np.eye(3), np.array([0.8, 1.0, 1.0])),
    )
    obstacle_sphere_3.meshColor = np.array([1.0, 1.0, 0.0, 0.5])
    obs_vis.append(visual_model.addGeometryObject(obstacle_sphere_3))
    obs_col.append(collision_model.addGeometryObject(obstacle_sphere_3))

    obstacle_box_1 = pinocchio.GeometryObject(
        "obstacle_box_1",
        0,
        coal.Box(
            0.5 + 2.0 * inflation_radius,
            0.1 + 2.0 * inflation_radius,
            0.52 + 2.0 * inflation_radius,
        ),
        pinocchio.SE3(np.eye(3), np.array([1.35, 0.2, 1.0])),
    )
    obstacle_box_1.meshColor = np.array([1.0, 0.0, 0.0, 0.5])
    obs_vis.append(visual_model.addGeometryObject(obstacle_box_1))
    obs_col.append(collision_model.addGeometryObject(obstacle_box_1))

    obstacle_box_2 = pinocchio.GeometryObject(
        "obstacle_box_2",
        0,
        coal.Box(
            0.8 + 2.0 * inflation_radius,
            1.2 + 2.0 * inflation_radius,
            0.08 + 2.0 * inflation_radius,
        ),
        pinocchio.SE3(np.eye(3), np.array([0.4, 0.6, 1.3])),
    )
    obstacle_box_2.meshColor = np.array([0.0, 0.0, 1.0, 0.5])
    obs_vis.append(visual_model.addGeometryObject(obstacle_box_2))
    obs_col.append(collision_model.addGeometryObject(obstacle_box_2))

    obstacle_box_3 = pinocchio.GeometryObject(
        "obstacle_box_3",
        0,
        coal.Box(
            0.1 + 2.0 * inflation_radius,
            0.4 + 2.0 * inflation_radius,
            0.32 + 2.0 * inflation_radius,
        ),
        pinocchio.SE3(np.eye(3), np.array([1.15, 0.2, 0.9])),
    )
    obstacle_box_3.meshColor = np.array([0.0, 0.0, 1.0, 0.5])
    obs_vis.append(visual_model.addGeometryObject(obstacle_box_3))
    obs_col.append(collision_model.addGeometryObject(obstacle_box_3))

    # Collect robot "bodies" by frame names (robust to geometry object naming)
    ROBOT_FRAME_PREFIXES = (
        "base_link", "shoulder_link", "upper_arm_link", "forearm_link",
        "wrist_1_link", "wrist_2_link", "wrist_3_link",
        "ee_link", "tool0", "flange", "attachment",
        "robotiq_85",
    )

    obstacle_names = OBSTACLES

    robot_bodies = []
    for go in collision_model.geometryObjects:
        # skip obstacles themselves (they are attached to universe)
        if go.name in ("ground_plane", "obstacle_box_1", "obstacle_box_2", "obstacle_box_3",
                    "obstacle_sphere_1", "obstacle_sphere_2", "obstacle_sphere_3"):
            continue
        if go.parentFrame >= 0:
            fname = model.frames[go.parentFrame].name
            if fname.startswith(ROBOT_FRAME_PREFIXES):
                robot_bodies.append(fname)

    robot_bodies = sorted(set(robot_bodies))
    print(f"[add_object_collisions] robot_bodies={len(robot_bodies)} robotiq={sum(b.startswith('robotiq_85') for b in robot_bodies)}")

    for obstacle_name in obstacle_names:
        for body in robot_bodies:
            # skip gripper for now
            if obstacle_name == "ground_plane" and body.startswith("robotiq_85"):
                continue
            set_collisions(model, collision_model, obstacle_name, body, True)

    for base in ("base_link", "base_link_inertia", "base"):
        if model.existFrame(base):
            set_collisions(model, collision_model, "ground_plane", base, False)

    return obs_vis, obs_col

def load_path_planner(model_roboplan, data_roboplan, collision_model):
    for cand in ("flange", "tool0", "ee_link", "wrist_3_link"):
        if model_roboplan.existFrame(cand):
            target_frame = cand
            break
    # ignore non-arm joints
    arm = {
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    }
    ignore_joint_indices = []
    for jid in range(1, model_roboplan.njoints):
        jname = model_roboplan.names[jid]
        if jname in arm:
            continue
        nq = model_roboplan.joints[jid].nq
        if nq <= 0:
            continue
        q0 = model_roboplan.idx_qs[jid]
        ignore_joint_indices.extend(range(q0, q0 + nq))

    ik_options = DifferentialIkOptions(
        max_iters=800,
        max_retries=20,
        damping=0.00001,
        min_step_size=0.03,
        max_step_size=0.35,
        ignore_joint_indices=ignore_joint_indices,
        rng_seed=None,
    )
    ik = DifferentialIk(
        model_roboplan,
        data=data_roboplan,
        collision_model=collision_model,
        options=ik_options,
        visualizer=None,
    )

    rrt_options = RRTPlannerOptions(
            max_step_size=0.4,
            max_connection_dist=0.8,
            rrt_connect=False,
            bidirectional_rrt=False,
            rrt_star=True,
            max_rewire_dist=0.4,
            max_planning_time=5.0,
            fast_return=True,
            goal_biasing_probability=0.3,
            collision_distance_padding=0.0001,
        )

    return target_frame, ik, rrt_options