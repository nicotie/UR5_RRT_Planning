import os.path
import sys

sys.path.append('../../manipulator_grasp')

import math
import numpy as np
import spatialmath as sm
import mujoco
import mujoco.viewer
import pinocchio

import glfw
import cv2

from manipulator_grasp.planner.set_model import (
    load_models,
    add_self_collisions,
    add_object_collisions,
    load_path_planner,
)

from manipulator_grasp.arm.robot import Robot, UR5e
from manipulator_grasp.arm.motion_planning import *
from manipulator_grasp.utils import mj

class GraspEnv:
    def __init__(self):
        self.sim_hz = 500
        self.mj_model: mujoco.MjModel = None
        self.mj_data: mujoco.MjData = None

        self.model_roboplan = None
        self.collision_model = None
        self.data_roboplan = None
        self.target_frame = None
        self.ik = None
        self.rrt_options = None

        self.mj_renderer: mujoco.Renderer = None
        self.mj_depth_renderer: mujoco.Renderer = None
        self.mj_viewer: mujoco.viewer.Handle = None
        self.mj_seg_renderer: mujoco.Renderer = None

        self.height = 640 # 256 640 720
        self.width = 640 # 256 640 1280
        self.fovy = np.pi / 4
        self.camera_matrix = np.eye(3)
        self.camera_matrix_inv = np.eye(3)

        self.cam_name = "cam"
        self.cam_id = -1
        self.obj_geom_id = -1

        self.offscreen_context = None
        self.offscreen_scene = None
        self.offscreen_camera = None
        self.offscreen_viewport = None
        self.glfw_window = None

    def reset(self):
        # ---- fixed alignment frames ----
        PIN_BASE_FRAME  = "base_link"
        MJ_BASE_BODY    = "ur5e_base"
        PIN_TOOL_FRAME  = "flange"                  # IK target frame
        PIN_GRASP_FRAME = "grasp_center"            # used for T_TG

        # --- for pin / roboplan ---
        urdf_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'ur5e_description', 'urdf', 'ur5e_2f85.urdf')
        srdf_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'ur5e_description', 'srdf', 'ur5e_2f85.srdf')
        self.model_roboplan, self.collision_model, visual_model = load_models(urdf_path)
        add_self_collisions(self.model_roboplan, self.collision_model, srdf_path)
        self.obs_vis_ids, self.obs_col_ids = add_object_collisions(
            self.model_roboplan, self.collision_model, visual_model, inflation_radius=0.01
        )
        self.obs_vis_ids = set(self.obs_vis_ids)
        self.obs_col_ids = set(self.obs_col_ids)

        # --- scene ---
        filename = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'scenes', 'grasp_env.xml')
        self.mj_model = mujoco.MjModel.from_xml_path(filename)
        self.mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # --- robot ---
        self.robot = UR5e()
        self.T_wb = mj.get_body_pose(self.mj_model, self.mj_data, MJ_BASE_BODY)

        def _apply_world_T(model, geom_models, T):
            for jid in range(1, model.njoints):
                if model.parents[jid] == 0:
                    model.jointPlacements[jid] = T * model.jointPlacements[jid]
            for fid in range(len(model.frames)):
                if model.frames[fid].parent == 0:
                    model.frames[fid].placement = T * model.frames[fid].placement

            for gm in geom_models:
                skip = self.obs_col_ids if gm is self.collision_model else self.obs_vis_ids
                for i, go in enumerate(gm.geometryObjects):
                    if go.parentJoint == 0 and i in skip:
                        continue
                    if go.parentJoint == 0:
                        go.placement = T * go.placement

        # --- pin world->base ---
        data0 = self.model_roboplan.createData()
        pinocchio.framesForwardKinematics(self.model_roboplan, data0, pinocchio.neutral(self.model_roboplan))
        fid_base = self.model_roboplan.getFrameId(PIN_BASE_FRAME)
        T_wB_pin = data0.oMf[fid_base]

        # mj world->base
        T_wB_mj = mj.world_pose_SE3(self.mj_model, self.mj_data, MJ_BASE_BODY)

        # apply correction to pin models (collision + visual, keep consistent)
        T_corr = T_wB_mj * T_wB_pin.inverse()
        _apply_world_T(self.model_roboplan, [self.collision_model, visual_model], T_corr)

        self.robot.set_base(self.T_wb.t)
        # self.robot_q = np.array([0.0, 0.0, np.pi / 2 * 0, 0.0, -np.pi / 2 * 0, 0.0])  # all zeros
        self.robot_q = np.array([0.0, 0.0, np.pi / 2, 0.0, -np.pi / 2, 0.0], dtype=float)
        self.robot.set_joint(self.robot_q)
        self.joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint",
                            "wrist_2_joint", "wrist_3_joint"]
        [mj.set_joint_q(self.mj_model, self.mj_data, jn, self.robot_q[i]) for i, jn in enumerate(self.joint_names)]

        # --- keep home pos ---
        if self.mj_model.nu >= 6:
            self.mj_data.ctrl[:6] = self.robot_q[:6]
        if self.mj_model.nu > 6:
            self.mj_data.ctrl[6] = 0.0 
        mujoco.mj_forward(self.mj_model, self.mj_data)

        robot_tool = sm.SE3.Trans(0.0, 0.0, 0.13) * sm.SE3.RPY(-np.pi / 2, -np.pi / 2, 0.0)
        self.robot.set_tool(robot_tool)
        self.robot_T = self.robot.fkine(self.robot_q)
        self.T0 = self.robot_T.copy()

        # --- planning objects ---
        self.data_roboplan = self.model_roboplan.createData()
        self.target_frame, self.ik, self.rrt_options = load_path_planner(
            self.model_roboplan, self.data_roboplan, self.collision_model
        )
        if self.target_frame != PIN_TOOL_FRAME:
            print(f"[env] WARN: target_frame is not {PIN_TOOL_FRAME}")
        else:
            print(f"[env] target_frame is {self.target_frame}")

        # --- constant tool->gripper transform in pin ---
        mj_q0 = np.array([mj.get_joint_q(self.mj_model, self.mj_data, jn) for jn in self.joint_names], dtype=float)
        q_fk = pinocchio.neutral(self.model_roboplan).copy()
        for jn, qv in zip(self.joint_names, mj_q0):
            jid = self.model_roboplan.getJointId(jn)
            q_fk[self.model_roboplan.joints[jid].idx_q] = float(qv)
        pinocchio.framesForwardKinematics(self.model_roboplan, self.data_roboplan, q_fk)
        fid_T = self.model_roboplan.getFrameId(PIN_TOOL_FRAME)
        fid_G = self.model_roboplan.getFrameId(PIN_GRASP_FRAME)
        T_TG_pin = self.data_roboplan.oMf[fid_T].inverse() * self.data_roboplan.oMf[fid_G]
        self.T_TG = sm.SE3.Rt(T_TG_pin.rotation, T_TG_pin.translation)

        # --- for segmentation / obj gid ---
        self.obj_geom_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "Banana")
        if self.obj_geom_id < 0:
            raise RuntimeError("Obj geom not found in XML")
        
        # --- camera ---
        self.cam_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, self.cam_name)
        if self.cam_id < 0:
            raise RuntimeError(f"Camera '{self.cam_name}' not found in XML")
        self.fovy = np.deg2rad(float(self.mj_model.cam_fovy[self.cam_id]))
        fy = self.height / (2.0 * np.tan(self.fovy / 2.0))
        fx = fy
        cx = (self.width - 1) / 2.0
        cy = (self.height - 1) / 2.0
        self.camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
        self.camera_matrix_inv = np.linalg.inv(self.camera_matrix)

        # --- renderers ---
        self.mj_renderer = mujoco.renderer.Renderer(self.mj_model, height=self.height, width=self.width)
        self.mj_depth_renderer = mujoco.renderer.Renderer(self.mj_model, height=self.height, width=self.width)
        self.mj_seg_renderer = mujoco.renderer.Renderer(self.mj_model, height=self.height, width=self.width)

        self.mj_depth_renderer.enable_depth_rendering()
        self.mj_seg_renderer.enable_segmentation_rendering()

        self.mj_renderer.update_scene(self.mj_data, camera=self.cam_id)
        self.mj_depth_renderer.update_scene(self.mj_data, camera=self.cam_id)
        self.mj_seg_renderer.update_scene(self.mj_data, camera=self.cam_id)

        # --- compute T_wc ---
        t_wc = self.mj_data.cam_xpos[self.cam_id].copy()
        R_wc_mj = self.mj_data.cam_xmat[self.cam_id].reshape(3, 3).copy()
        # MuJoCo camera: X right, Y up, look=-Z  ->  CV camera: X right, Y down, Z forward
        R_mj_to_cv = np.diag([1.0, -1.0, -1.0])
        R_wc_cv = R_wc_mj @ R_mj_to_cv
        self.T_wc = sm.SE3.Rt(R_wc_cv, t_wc)
        R = self.T_wc.R
        # print("[T_wc] [should be +1] det(R)=", np.linalg.det(R), "[should be small] orth_err=", np.linalg.norm(R.T @ R - np.eye(3)))
        # print("[T_wc] t=", self.T_wc.t)
        # print("[T_wc] cam forward(world)=", (self.T_wc.R @ np.array([0,0,1.0])).ravel())  # +Z (CV forward) in world

        # --- launch_passive viewer ---
        self.mj_viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
        self.mj_viewer.cam.lookat[:] = [1.8, 1.1, 1.7]
        self.mj_viewer.cam.azimuth = 210
        self.mj_viewer.cam.elevation = -35
        self.mj_viewer.cam.distance = 1.2
        self.mj_viewer.sync()

        print("[env] reset")

    def get_planning_models(self):
        if self.model_roboplan is None or self.collision_model is None:
            raise RuntimeError("Planning models not initialized. Call env.reset() first.")
        return self.model_roboplan, self.collision_model

    def close(self):
        if self.mj_viewer is not None:
            self.mj_viewer.close()
        if self.mj_renderer is not None:
            self.mj_renderer.close()
        if self.mj_depth_renderer is not None:
            self.mj_depth_renderer.close()
        if self.mj_seg_renderer is not None:
            self.mj_seg_renderer.close()
        cv2.destroyAllWindows()
        if self.glfw_window is not None:
            glfw.destroy_window(self.glfw_window)
        glfw.terminate()
        self.offscreen_context = None
        self.offscreen_scene = None

    def step(self, action=None):
        if action is not None:
            self.mj_data.ctrl[:] = action
        mujoco.mj_step(self.mj_model, self.mj_data)

        self.mj_viewer.sync()
                
    def render(self):
        self.mj_renderer.update_scene(self.mj_data, self.cam_id)
        self.mj_depth_renderer.update_scene(self.mj_data, self.cam_id)
        self.mj_seg_renderer.update_scene(self.mj_data, self.cam_id)
        return {
            'img': self.mj_renderer.render(),
            'depth': self.mj_depth_renderer.render(),
            'seg': self.mj_seg_renderer.render(),
            'obj_gid': int(self.obj_geom_id),
            'K': self.camera_matrix.copy(),
            'T_wc': self.T_wc.A.copy(),
        }

if __name__ == '__main__':
    env = GraspEnv()
    env.reset()
    for i in range(10000):
        env.step()
    imgs = env.render()
    env.close()
