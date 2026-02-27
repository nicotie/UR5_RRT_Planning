import os
import numpy as np
import pinocchio
import hppfcl as coal

from pyroboplan.core.utils import set_collisions
from pyroboplan.ik.differential_ik import DifferentialIk, DifferentialIkOptions
from pyroboplan.planning.rrt import RRTPlannerOptions
from manipulator_grasp.utils import mj

OBSTACLES = [
    "ground_plane",
    "obstacle_box_1",
    # "obstacle_box_2",
    # "obstacle_box_3",
    "obstacle_sphere_1",
    "obstacle_sphere_2",
    "obstacle_sphere_3",
    "obstacle_sphere_4",
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
                iq = model.joints[jid].idx_q
                lo = float(model.lowerPositionLimit[iq])
                hi = float(model.upperPositionLimit[iq])
                if lo <= 0.0 <= hi:
                    qref[iq] = 0.0
                else:
                    qref[iq] = lo
                # print(f"[lock_gripper] {jn} idx_q={iq} lim=({lo:.4f},{hi:.4f}) use={qref[iq]:.4f}")
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

def add_object_collisions(model, collision_model, visual_model, inflation_radius=0.0,
                          mj_model=None, mj_data=None):
    obs_vis = []
    obs_col = []
    if mj_model is not None and mj_data is not None:
        import mujoco
        def _add_from_mj_geom(src_name: str, pin_name: str | None = None):
            if pin_name is None:
                pin_name = src_name
            gid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, src_name)
            if gid < 0:
                print(f"[obstacle_mjcf] geom not found: {src_name}")
                return

            gtype = int(mj_model.geom_type[gid])
            size = mj_model.geom_size[gid].copy()
            t = mj_data.geom_xpos[gid].copy()
            R = mj_data.geom_xmat[gid].reshape(3, 3).copy()

            if gtype == mujoco.mjtGeom.mjGEOM_BOX:
                # MuJoCo: half-sizes; FCL Box: full sizes
                geom = coal.Box(*(2.0 * size + 2.0 * inflation_radius))
            elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
                geom = coal.Sphere(float(size[0] + inflation_radius))
            else:
                print(f"[obstacle_mjcf] unsupported geom type {gtype} for {src_name}")
                return

            go = pinocchio.GeometryObject(pin_name, 0, geom, pinocchio.SE3(R, t))
            obs_vis.append(visual_model.addGeometryObject(go))
            obs_col.append(collision_model.addGeometryObject(go))
            # print(f"[obstacle_mjcf] {pin_name} <= {src_name} t={t} size={size} type={gtype}")

        for name in OBSTACLES:
            if name == "ground_plane":
                _add_from_mj_geom("simple_table", pin_name="ground_plane")
            else:
                _add_from_mj_geom(name)

        # --- enable obstacle <-> robot collision pairs ---
        ROBOT_FRAME_PREFIXES = (
            "base_link", "shoulder_link", "upper_arm_link", "forearm_link",
            "wrist_1_link", "wrist_2_link", "wrist_3_link",
            "ee_link", "tool0", "flange", "attachment",
            "robotiq_85",
        )

        geom_name_set = {go.name for go in collision_model.geometryObjects}

        robot_bodies = []
        for go in collision_model.geometryObjects:
            if go.name in geom_name_set and go.name in OBSTACLES:
                continue
            if go.parentFrame >= 0:
                fname = model.frames[go.parentFrame].name
                if fname.startswith(ROBOT_FRAME_PREFIXES):
                    robot_bodies.append(fname)

        robot_bodies = sorted(set(robot_bodies))
        # print(f"[add_object_collisions] robot_bodies={len(robot_bodies)}")

        for obstacle_name in OBSTACLES:
            if obstacle_name not in geom_name_set:
                print(f"[add_object_collisions] skip missing obstacle geom: {obstacle_name}")
                continue
            for body in robot_bodies:
                if obstacle_name == "ground_plane" and body.startswith("robotiq_85"):
                    continue
                set_collisions(model, collision_model, obstacle_name, body, True)

        for base in ("base_link", "base_link_inertia", "base"):
            if model.existFrame(base) and "ground_plane" in geom_name_set:
                set_collisions(model, collision_model, "ground_plane", base, False)

    else:
        # Hand write obstacles
        # Add the collision objects
        ground_plane = pinocchio.GeometryObject(
            "ground_plane",
            0,
            coal.Box(1.6, 1.2, 0.1),
            pinocchio.SE3(np.eye(3), np.array([0.8, 0.6, 0.69])), # 0.69
        )
        ground_plane.meshColor = np.array([0.5, 0.5, 0.5, 0.5])
        obs_vis.append(visual_model.addGeometryObject(ground_plane))
        obs_col.append(collision_model.addGeometryObject(ground_plane))

        obstacle_sphere_1 = pinocchio.GeometryObject(
            "obstacle_sphere_1",
            0,
            coal.Sphere(0.1 + inflation_radius),
            pinocchio.SE3(np.eye(3), np.array([0.3, 0.6, 0.85])),
        )
        obstacle_sphere_1.meshColor = np.array([0.0, 1.0, 0.0, 0.5])
        obs_vis.append(visual_model.addGeometryObject(obstacle_sphere_1))
        obs_col.append(collision_model.addGeometryObject(obstacle_sphere_1))

        obstacle_sphere_2 = pinocchio.GeometryObject(
            "obstacle_sphere_2",
            0,
            coal.Sphere(0.15 + inflation_radius),
            pinocchio.SE3(np.eye(3), np.array([0.3, 0.6, 1.5])),
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

        # Collect robot "bodies" by frame names
        ROBOT_FRAME_PREFIXES = (
            "base_link", "shoulder_link", "upper_arm_link", "forearm_link",
            "wrist_1_link", "wrist_2_link", "wrist_3_link",
            "ee_link", "tool0", "flange", "attachment",
            "robotiq_85",
        )

        obstacle_names = OBSTACLES

        robot_bodies = []
        for go in collision_model.geometryObjects:
            # skip obstacles themselves
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

def attach_grasped_object_mj_geom(
    model: pinocchio.Model,
    collision_model: pinocchio.GeometryModel,
    mj_model,
    mj_data,
    obj_gid: int,
    mj_tool_name: str = "flange",
    parent_frame: str = "flange",
    name: str = "grasped_object",
    inflation_radius: float = 0.0,
):
    import mujoco

    def _mj_pose_body_or_site(nm: str):
        bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, nm)
        if bid >= 0:
            t = mj_data.xpos[bid].copy()
            R = mj_data.xmat[bid].reshape(3, 3).copy()
            return R, t, False
        sid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, nm)
        if sid >= 0:
            t = mj_data.site_xpos[sid].copy()
            R = mj_data.site_xmat[sid].reshape(3, 3).copy()
            return R, t, True
        return None, None, None

    # --- tool pose in world ---
    R_wT, t_wT, is_site = _mj_pose_body_or_site(mj_tool_name)
    if R_wT is None:
        print(f"[attach_obj] mj tool not found (body/site): {mj_tool_name}")
        return False

    # --- object geom pose in world ---
    gtype = int(mj_model.geom_type[obj_gid])
    size = mj_model.geom_size[obj_gid].copy()
    t_wO = mj_data.geom_xpos[obj_gid].copy()
    R_wO = mj_data.geom_xmat[obj_gid].reshape(3, 3).copy()

    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
        geom = coal.Box(*(2.0 * size + 2.0 * inflation_radius))
    elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        geom = coal.Sphere(float(size[0] + inflation_radius))
    elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
        # MuJoCo: radius, half-height
        geom = coal.Cylinder(float(size[0] + inflation_radius), float(2.0 * (size[1] + inflation_radius)))
    elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
        # MuJoCo: radius, half-height (cylinder part)
        geom = coal.Capsule(float(size[0] + inflation_radius), float(2.0 * (size[1] + inflation_radius)))
    else:
        print(f"[attach_obj] unsupported geom type {gtype} for gid={obj_gid}")
        return False

    # --- compute T_T_O ---
    R_TW = R_wT.T
    R_TO = R_TW @ R_wO
    t_TO = R_TW @ (t_wO - t_wT)
    T_TO = pinocchio.SE3(R_TO, t_TO)

    # --- attach to Pinocchio frame ---
    fid = model.getFrameId(parent_frame)
    if fid >= model.nframes:
        print(f"[attach_obj] pin parent_frame not found: {parent_frame}")
        return False
    fr = model.frames[fid]
    pj = int(getattr(fr, "parentJoint", getattr(fr, "parent", -1)))
    if pj < 0:
        print(f"[attach_obj] frame has no parentJoint/parent: {fr}")
        return False
    print("[attach_obj] frame attrs has parent=", hasattr(fr,"parent"), "has parentJoint=", hasattr(fr,"parentJoint"))
    placement = fr.placement * T_TO  # joint->frame * frame->obj

    gid = collision_model.getGeometryId(name)
    is_new = gid >= collision_model.ngeoms
    if is_new:
        go = pinocchio.GeometryObject(name, pj, geom, placement)
        if hasattr(go, "parentFrame"):
            go.parentFrame = fid
        collision_model.addGeometryObject(go)
    else:
        go = collision_model.geometryObjects[gid]
        if hasattr(go, "parentJoint"):
            go.parentJoint = pj
        if hasattr(go, "parentFrame"):
            go.parentFrame = fid
        go.placement = placement
        go.geometry = geom

    if is_new:
        # enable obj <-> obstacles
        for obs in OBSTACLES:
            set_collisions(model, collision_model, name, obs, True)

        # disable obj <-> gripper/tool frames (avoid self-collision with hand)
        for fn in ("flange", "tool0", "ee_link", "attachment", "wrist_3_link"):
            if fn == parent_frame:
                continue  # avoid self-pair: grasped_object is attached to this frame
            if model.existFrame(fn):
                set_collisions(model, collision_model, name, fn, False)

        gripper_frames = set()
        for g in collision_model.geometryObjects:
            if 0 <= g.parentFrame < model.nframes:
                fname = model.frames[g.parentFrame].name
                if fname.startswith("robotiq_85"):
                    gripper_frames.add(fname)
        for fn in gripper_frames:
            set_collisions(model, collision_model, name, fn, False)

    print(f"[attach_obj] ok tool={mj_tool_name}({ 'site' if is_site else 'body'}) T_TO.t={t_TO} type={gtype}")
    return True

def disable_grasped_object_collisions(
    model: pinocchio.Model,
    collision_model: pinocchio.GeometryModel,
    name: str = "grasped_object",
):
    """Disable obj<->obstacles pairs after release (so retreat planning won't assume object is still attached)."""
    if collision_model.getGeometryId(name) >= collision_model.ngeoms:
        return
    for obs in OBSTACLES:
        set_collisions(model, collision_model, name, obs, False)

def load_path_planner(model_roboplan, data_roboplan, collision_model):
    target_frame = "flange"
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
            collision_distance_padding=0.0005,
        )

    return target_frame, ik, rrt_options