import os
import numpy as np
if not hasattr(np, "concat"):
    np.concat = np.concatenate
import pinocchio
# import hppfcl as coal
from pyroboplan.planning.utils import discretize_joint_space_path, has_collision_free_path
from pyroboplan.planning.rrt import RRTPlanner
from pyroboplan.planning.path_shortcutting import shortcut_path
from pyroboplan.trajectory.trajectory_optimization import (
    CubicTrajectoryOptimization,
    CubicTrajectoryOptimizationOptions,
)
from pyroboplan.core.utils import (
    check_collisions_at_state,
    check_within_limits,
    get_random_collision_free_state,
)
from manipulator_grasp.planner.set_model import OBSTACLES

def getIk(env, init_state, T_target):
    rotation_matrix = T_target.R
    translation_vector = T_target.t

    target_tform = pinocchio.SE3(rotation_matrix, np.array(translation_vector))
    q_sol = env.ik.solve(
        env.target_frame,
        target_tform,
        init_state=init_state.copy(),
        # verbose=True,
        verbose=False,
    )
    if q_sol is None:
        cm_saved = env.ik.collision_model
        env.ik.collision_model = None
        q_dbg = env.ik.solve(env.target_frame, target_tform, init_state=init_state.copy(), verbose=False)
        env.ik.collision_model = cm_saved

        if q_dbg is not None and cm_saved is not None:
            data = env.model_roboplan.createData()
            gdata = cm_saved.createData()
            pinocchio.updateGeometryPlacements(env.model_roboplan, data, cm_saved, gdata, q_dbg)
            pinocchio.computeCollisions(env.model_roboplan, data, cm_saved, gdata, q_dbg, False)
            n_obj = 0
            for i, pair in enumerate(cm_saved.collisionPairs):
                if not gdata.collisionResults[i].isCollision():
                    continue
                a = cm_saved.geometryObjects[pair.first].name
                b = cm_saved.geometryObjects[pair.second].name
                if ("grasped_object" in a) or ("grasped_object" in b):
                    print(f"[IK_FAIL_OBJ_COL] {a} <-> {b}")
                    n_obj += 1
                    if n_obj >= 10:
                        break
    return q_sol

def get_traj(env, q_start, q_goal):

    def _path_collision_free(q_list, step=0.03):
        pad = float(getattr(env.rrt_options, "collision_distance_padding", 0.0))
        for i, (qa, qb) in enumerate(zip(q_list[:-1], q_list[1:])):
            if not has_collision_free_path(qa, qb, step, env.model_roboplan, env.collision_model, distance_padding=pad):
                print(f"[PATH_COL] segment {i}->{i+1} step={step}")

                # --- debug: dump first collisions at qb ---
                data = env.model_roboplan.createData()
                gdata = env.collision_model.createData()
                pinocchio.updateGeometryPlacements(env.model_roboplan, data, env.collision_model, gdata, qb)
                pinocchio.computeCollisions(env.model_roboplan, data, env.collision_model, gdata, qb, False)
                n = 0
                for j, pair in enumerate(env.collision_model.collisionPairs):
                    if gdata.collisionResults[j].isCollision():
                        a = env.collision_model.geometryObjects[pair.first].name
                        b = env.collision_model.geometryObjects[pair.second].name
                        print(f"[PATH_COL@q] {a} <-> {b}")
                        n += 1
                        if n >= 6:
                            break

                return False
        return True

    # keep RRT input dimensions consistent with pinocchio model
    nq = env.model_roboplan.nq
    q_start = np.ascontiguousarray(np.asarray(q_start, dtype=np.float64).ravel())
    q_goal  = np.ascontiguousarray(np.asarray(q_goal,  dtype=np.float64).ravel())
    if q_start.size != nq:
        q_start = q_start[:nq] if q_start.size > nq else np.pad(q_start, (0, nq - q_start.size))
    if q_goal.size != nq:
        q_goal = q_goal[:nq] if q_goal.size > nq else np.pad(q_goal, (0, nq - q_goal.size))

    pad = float(getattr(env.rrt_options, "collision_distance_padding", 0.0))    # 0.0005
    chk = env.rrt_options.max_step_size # 0.02

    if has_collision_free_path(
        q_start, q_goal,
        0.02,
        env.model_roboplan, env.collision_model,
        distance_padding=pad,
    ):
        q_dense = discretize_joint_space_path([q_start, q_goal], max_angle_distance=chk)
        return np.asarray(q_dense, dtype=np.float64).T

    print("Planning a path...")
    planner = RRTPlanner(env.model_roboplan, env.collision_model, options=env.rrt_options)
    q_path = planner.plan(q_start, q_goal)
    if q_path is None or len(q_path) == 0:
        print("Failed to plan.")
        return None
    # short cut
    if pad <= 0.0:
        q_path = shortcut_path(
            env.model_roboplan,
            env.collision_model,
            q_path,
            max_iters=200,
            max_step_size=chk,
        )
    q_path = discretize_joint_space_path(q_path, max_angle_distance=0.15)

    if q_path is not None and len(q_path) > 0:
        print(f"Got a path with {len(q_path)} waypoints")
        if len(q_path) > 500:   # 100
            print("Path is too long, skipping...")
            return None
        else:
            # print(q_path)
            # Perform trajectory optimization.
            # dt = env.execute_dt
            traj_dt = 0.01  # 0.004
            traj_options = CubicTrajectoryOptimizationOptions(
                num_waypoints=len(q_path),
                samples_per_segment=5,
                min_segment_time=0.15,
                max_segment_time=3.0,
                min_vel=-1.0,
                max_vel=1.0,
                min_accel=-0.5,
                max_accel=0.5,
                min_jerk=-0.5,
                max_jerk=0.5,
                max_planning_time=10.0,
                check_collisions=False,
                collision_link_list=[],
            )         
            print("Optimizing the path...")
            optimizer = CubicTrajectoryOptimization(env.model_roboplan, env.collision_model, traj_options)
            # traj = optimizer.plan([q_path[0], q_path[-1]], init_path=q_path)
            traj = optimizer.plan(q_path, init_path=q_path)
            if traj is None:
                # fallback to discrete path
                print("Try to discretize the path...")
                q_dense = discretize_joint_space_path(q_path, max_angle_distance=chk)
                if not _path_collision_free(q_dense, step=chk): return None
                return np.asarray(q_dense, dtype=np.float64).T
            if traj is not None:
                traj_gen = traj.generate(traj_dt)
                q_vec = traj_gen[1]
                step = max(1, int(round(0.02 / traj_dt)))
                bad = False
                for i in range(0, q_vec.shape[1], step):
                    if check_collisions_at_state(env.model_roboplan, env.collision_model, q_vec[:, i], distance_padding=pad):
                        bad = True
                        break
                if bad:
                    q_dense = discretize_joint_space_path(q_path, max_angle_distance=chk)
                    if not _path_collision_free(q_dense, step=chk):
                        return None
                    return np.asarray(q_dense, dtype=np.float64).T
                return q_vec
            else:
                return None
    else:
        print("Failed to plan.")
        return None