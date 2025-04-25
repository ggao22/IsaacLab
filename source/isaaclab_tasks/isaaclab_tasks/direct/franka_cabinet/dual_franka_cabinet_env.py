# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform


@configclass
class DualFrankaCabinetEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 8.3333  # 500 timesteps
    decimation = 2
    action_space = 18  # 9 per robot
    observation_space = 41 # (9+9)*2 dof pos/vel + 3 target pos + 1 cabinet joint pos + 1 cabinet joint vel
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=True)

    # robot_left
    robot_left = ArticulationCfg(
        prim_path="/World/envs/env_.*/RobotLeft",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Franka/franka_instanceable.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 1.157,
                "panda_joint2": -1.066,
                "panda_joint3": -0.155,
                "panda_joint4": -2.239,
                "panda_joint5": -1.841,
                "panda_joint6": 1.003,
                "panda_joint7": 0.469,
                "panda_finger_joint.*": 0.035,
            },
            pos=(1.0, -0.3, 0.0), # Shifted left
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=12.0,
                velocity_limit=2.61,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit=200.0,
                velocity_limit=0.2,
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )

    # robot_right
    robot_right = ArticulationCfg(
        prim_path="/World/envs/env_.*/RobotRight",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Franka/franka_instanceable.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 1.157,
                "panda_joint2": -1.066,
                "panda_joint3": -0.155,
                "panda_joint4": -2.239,
                "panda_joint5": -1.841,
                "panda_joint6": 1.003,
                "panda_joint7": 0.469,
                "panda_finger_joint.*": 0.035,
            },
            pos=(1.0, 0.3, 0.0), # Shifted right
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=12.0,
                velocity_limit=2.61,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit=200.0,
                velocity_limit=0.2,
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )

    # cabinet
    cabinet = ArticulationCfg(
        prim_path="/World/envs/env_.*/Cabinet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd",
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0, 0.4),
            rot=(0.1, 0.0, 0.0, 0.0),
            joint_pos={
                "door_left_joint": 0.0,
                "door_right_joint": 0.0,
                "drawer_bottom_joint": 0.0,
                "drawer_top_joint": 0.0,
            },
        ),
        actuators={
            "drawers": ImplicitActuatorCfg(
                joint_names_expr=["drawer_top_joint", "drawer_bottom_joint"],
                effort_limit=87.0,
                velocity_limit=100.0,
                stiffness=10.0,
                damping=1.0,
            ),
            "doors": ImplicitActuatorCfg(
                joint_names_expr=["door_left_joint", "door_right_joint"],
                effort_limit=87.0,
                velocity_limit=100.0,
                stiffness=10.0,
                damping=2.5,
            ),
        },
    )

    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    action_scale = 7.5
    dof_velocity_scale = 0.1

    # reward scales
    dist_reward_scale = 1.5
    rot_reward_scale = 1.5
    open_reward_scale = 10.0
    action_penalty_scale = 0.05
    finger_reward_scale = 2.0


class DualFrankaCabinetEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: DualFrankaCabinetEnvCfg

    def __init__(self, cfg: DualFrankaCabinetEnvCfg, render_mode: str | None = None, **kwargs):
        # check config
        if cfg.action_space != 18:
            raise ValueError(f"Expected action space of 18, got: {cfg.action_space}")
        if cfg.observation_space != 41:
            raise ValueError(f"Expected observation space of 41, got: {cfg.observation_space}")

        super().__init__(cfg, render_mode, **kwargs)

        def get_env_local_pose(env_pos: torch.Tensor, xformable: UsdGeom.Xformable, device: torch.device):
            """Compute pose in env-local coordinates"""
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()

            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real

            return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # create auxiliary variables for computing applied action, observations and rewards
        # -- Robot Left
        self.robot_left_dof_lower_limits = self._robot_left.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_left_dof_upper_limits = self._robot_left.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_left_dof_speed_scales = torch.ones_like(self.robot_left_dof_lower_limits)
        self.robot_left_dof_speed_scales[self._robot_left.find_joints("panda_finger_joint1")[0]] = 0.1
        self.robot_left_dof_speed_scales[self._robot_left.find_joints("panda_finger_joint2")[0]] = 0.1
        self.robot_left_dof_targets = torch.zeros((self.num_envs, self._robot_left.num_joints), device=self.device)

        # -- Robot Right
        self.robot_right_dof_lower_limits = self._robot_right.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_right_dof_upper_limits = self._robot_right.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_right_dof_speed_scales = torch.ones_like(self.robot_right_dof_lower_limits)
        self.robot_right_dof_speed_scales[self._robot_right.find_joints("panda_finger_joint1")[0]] = 0.1
        self.robot_right_dof_speed_scales[self._robot_right.find_joints("panda_finger_joint2")[0]] = 0.1
        self.robot_right_dof_targets = torch.zeros((self.num_envs, self._robot_right.num_joints), device=self.device)

        stage = get_current_stage()
        # Use RobotLeft prim path for grasp pose calculation (assuming symmetry)
        hand_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/RobotLeft/panda_link7")),
            self.device,
        )
        lfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/RobotLeft/panda_leftfinger")),
            self.device,
        )
        rfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/RobotLeft/panda_rightfinger")),
            self.device,
        )

        finger_pose = torch.zeros(7, device=self.device)
        finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0
        finger_pose[3:7] = lfinger_pose[3:7]
        hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(hand_pose[3:7], hand_pose[0:3])

        robot_local_grasp_pose_rot, robot_local_pose_pos = tf_combine(
            hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3]
        )
        robot_local_pose_pos += torch.tensor([0, 0.04, 0], device=self.device)
        # Store local grasp poses (same for both robots relative to their base)
        self.robot_local_grasp_pos = robot_local_pose_pos.repeat((self.num_envs, 1))
        self.robot_local_grasp_rot = robot_local_grasp_pose_rot.repeat((self.num_envs, 1))

        drawer_local_grasp_pose = torch.tensor([0.3, 0.01, 0.0, 1.0, 0.0, 0.0, 0.0], device=self.device)
        self.drawer_local_grasp_pos = drawer_local_grasp_pose[0:3].repeat((self.num_envs, 1))
        self.drawer_local_grasp_rot = drawer_local_grasp_pose[3:7].repeat((self.num_envs, 1))

        self.gripper_forward_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.drawer_inward_axis = torch.tensor([-1, 0, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.gripper_up_axis = torch.tensor([0, 1, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.drawer_up_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )

        # Indices for Robot Left
        self.hand_link_idx_left = self._robot_left.find_bodies("panda_link7")[0][0]
        self.left_finger_link_idx_left = self._robot_left.find_bodies("panda_leftfinger")[0][0]
        self.right_finger_link_idx_left = self._robot_left.find_bodies("panda_rightfinger")[0][0]

        # Indices for Robot Right
        self.hand_link_idx_right = self._robot_right.find_bodies("panda_link7")[0][0]
        self.left_finger_link_idx_right = self._robot_right.find_bodies("panda_leftfinger")[0][0]
        self.right_finger_link_idx_right = self._robot_right.find_bodies("panda_rightfinger")[0][0]

        # Common index for cabinet
        self.drawer_link_idx = self._cabinet.find_bodies("drawer_top")[0][0]

        # Grasp poses for Robot Left
        self.robot_left_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_left_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)

        # Grasp poses for Robot Right
        self.robot_right_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_right_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)

        # Common grasp pose for the drawer (target)
        self.drawer_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.drawer_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)

    def _setup_scene(self):
        self._robot_left = Articulation(self.cfg.robot_left)
        self._robot_right = Articulation(self.cfg.robot_right)
        self._cabinet = Articulation(self.cfg.cabinet)
        self.scene.articulations["robot_left"] = self._robot_left
        self.scene.articulations["robot_right"] = self._robot_right
        self.scene.articulations["cabinet"] = self._cabinet

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # pre-physics step calls

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)
        # Split actions for left and right robots
        actions_left = self.actions[:, :9]
        actions_right = self.actions[:, 9:]

        # Update targets for robot left
        targets_left = self.robot_left_dof_targets + self.robot_left_dof_speed_scales * self.dt * actions_left * self.cfg.action_scale
        self.robot_left_dof_targets[:] = torch.clamp(targets_left, self.robot_left_dof_lower_limits, self.robot_left_dof_upper_limits)

        # Update targets for robot right
        targets_right = self.robot_right_dof_targets + self.robot_right_dof_speed_scales * self.dt * actions_right * self.cfg.action_scale
        self.robot_right_dof_targets[:] = torch.clamp(targets_right, self.robot_right_dof_lower_limits, self.robot_right_dof_upper_limits)

    def _apply_action(self):
        self._robot_left.set_joint_position_target(self.robot_left_dof_targets)
        self._robot_right.set_joint_position_target(self.robot_right_dof_targets)

    # post-physics step calls

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = self._cabinet.data.joint_pos[:, 3] > 0.39
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()
        robot_left_lfinger_pos = self._robot_left.data.body_pos_w[:, self.left_finger_link_idx_left]
        robot_left_rfinger_pos = self._robot_left.data.body_pos_w[:, self.right_finger_link_idx_left]
        robot_right_lfinger_pos = self._robot_right.data.body_pos_w[:, self.left_finger_link_idx_right]
        robot_right_rfinger_pos = self._robot_right.data.body_pos_w[:, self.right_finger_link_idx_right]

        # Combine actions for penalty calculation
        # actions_combined = torch.cat((self.actions[:, :9], self.actions[:, 9:]), dim=1) # self.actions is already combined

        return self._compute_rewards(
            self.actions, # Pass combined actions
            self._cabinet.data.joint_pos,
            # Left robot grasp info
            self.robot_left_grasp_pos,
            self.robot_left_grasp_rot,
            robot_left_lfinger_pos,
            robot_left_rfinger_pos,
             # Right robot grasp info
            self.robot_right_grasp_pos,
            self.robot_right_grasp_rot,
            robot_right_lfinger_pos,
            robot_right_rfinger_pos,
            # Common target info
            self.drawer_grasp_pos,
            self.drawer_grasp_rot,
            # Axes (assuming same for both grippers and target)
            self.gripper_forward_axis,
            self.drawer_inward_axis,
            self.gripper_up_axis,
            self.drawer_up_axis,
            # Env info
            self.num_envs,
            # Scales
            self.cfg.dist_reward_scale,
            self.cfg.rot_reward_scale,
            self.cfg.open_reward_scale,
            self.cfg.action_penalty_scale,
            self.cfg.finger_reward_scale,
        )

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids) # Call parent reset first

        # Robot Left state
        joint_pos_left = self._robot_left.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125,
            0.125,
            (len(env_ids), self._robot_left.num_joints),
            self.device,
        )
        joint_pos_left = torch.clamp(joint_pos_left, self.robot_left_dof_lower_limits, self.robot_left_dof_upper_limits)
        joint_vel_left = torch.zeros_like(joint_pos_left)
        self._robot_left.set_joint_position_target(joint_pos_left, env_ids=env_ids)
        self._robot_left.write_joint_state_to_sim(joint_pos_left, joint_vel_left, env_ids=env_ids)
        self.robot_left_dof_targets[env_ids] = joint_pos_left # Reset targets too


        # Robot Right state
        joint_pos_right = self._robot_right.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125,
            0.125,
            (len(env_ids), self._robot_right.num_joints),
            self.device,
        )
        joint_pos_right = torch.clamp(joint_pos_right, self.robot_right_dof_lower_limits, self.robot_right_dof_upper_limits)
        joint_vel_right = torch.zeros_like(joint_pos_right)
        self._robot_right.set_joint_position_target(joint_pos_right, env_ids=env_ids)
        self._robot_right.write_joint_state_to_sim(joint_pos_right, joint_vel_right, env_ids=env_ids)
        self.robot_right_dof_targets[env_ids] = joint_pos_right # Reset targets too

        # cabinet state
        zeros = torch.zeros((len(env_ids), self._cabinet.num_joints), device=self.device)
        self._cabinet.write_joint_state_to_sim(zeros, zeros, env_ids=env_ids)

        # Need to refresh the intermediate values so that _get_observations() can use the latest values
        self._compute_intermediate_values(env_ids)

    def _get_observations(self) -> dict:
        # Robot Left Observations
        dof_pos_scaled_left = (
            2.0
            * (self._robot_left.data.joint_pos - self.robot_left_dof_lower_limits)
            / (self.robot_left_dof_upper_limits - self.robot_left_dof_lower_limits)
            - 1.0
        )
        dof_vel_scaled_left = self._robot_left.data.joint_vel * self.cfg.dof_velocity_scale
        to_target_left = self.drawer_grasp_pos - self.robot_left_grasp_pos

        # Robot Right Observations
        dof_pos_scaled_right = (
            2.0
            * (self._robot_right.data.joint_pos - self.robot_right_dof_lower_limits)
            / (self.robot_right_dof_upper_limits - self.robot_right_dof_lower_limits)
            - 1.0
        )
        dof_vel_scaled_right = self._robot_right.data.joint_vel * self.cfg.dof_velocity_scale
        to_target_right = self.drawer_grasp_pos - self.robot_right_grasp_pos # Should this be different per robot? For now, same target


        # Cabinet observations (same as before)
        cabinet_dof_pos = self._cabinet.data.joint_pos[:, 3].unsqueeze(-1)
        cabinet_dof_vel = self._cabinet.data.joint_vel[:, 3].unsqueeze(-1)


        # Combine observations:
        # [dof_pos_left(9), dof_vel_left(9), to_target_left(3),
        #  dof_pos_right(9), dof_vel_right(9), to_target_right(3),
        #  cabinet_dof_pos(1), cabinet_dof_vel(1)]
        # Total: 9+9+3 + 9+9+3 + 1+1 = 18+3 + 18+3 + 2 = 21 + 21 + 2 = 44. Hmm, config says 41. Let's recheck.
        # Original was 23: dof_pos(9) + dof_vel(9) + to_target(3) + cabinet_pos(1) + cabinet_vel(1) = 9+9+3+1+1 = 23.
        # New: (dof_pos_left(9) + dof_vel_left(9)) + (dof_pos_right(9) + dof_vel_right(9)) + to_target_avg(3) ??? + cabinet_pos(1) + cabinet_vel(1)
        # Let's use the average grasp position for the target vector for now.
        avg_robot_grasp_pos = (self.robot_left_grasp_pos + self.robot_right_grasp_pos) / 2.0
        to_target_avg = self.drawer_grasp_pos - avg_robot_grasp_pos
        # Obs: [dof_pos_left(9), dof_vel_left(9), dof_pos_right(9), dof_vel_right(9), to_target_avg(3), cab_pos(1), cab_vel(1)]
        # Total: 9+9+9+9 + 3 + 1 + 1 = 36 + 3 + 1 + 1 = 41. Matches config.

        obs = torch.cat(
            (
                dof_pos_scaled_left,
                dof_vel_scaled_left,
                dof_pos_scaled_right,
                dof_vel_scaled_right,
                to_target_avg,
                cabinet_dof_pos,
                cabinet_dof_vel,
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    # auxiliary methods

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids_all = self._robot_left._ALL_INDICES # Use either robot, they share indices
            # Compute for Robot Left
            hand_pos_left = self._robot_left.data.body_pos_w[env_ids_all, self.hand_link_idx_left]
            hand_rot_left = self._robot_left.data.body_quat_w[env_ids_all, self.hand_link_idx_left]
            # Compute for Robot Right
            hand_pos_right = self._robot_right.data.body_pos_w[env_ids_all, self.hand_link_idx_right]
            hand_rot_right = self._robot_right.data.body_quat_w[env_ids_all, self.hand_link_idx_right]
            # Compute for Cabinet
            drawer_pos = self._cabinet.data.body_pos_w[env_ids_all, self.drawer_link_idx]
            drawer_rot = self._cabinet.data.body_quat_w[env_ids_all, self.drawer_link_idx]

            # Compute grasp transforms for Left Robot
            (
                self.robot_left_grasp_rot[env_ids_all],
                self.robot_left_grasp_pos[env_ids_all],
                self.drawer_grasp_rot[env_ids_all], # Drawer grasp is computed once
                self.drawer_grasp_pos[env_ids_all],
            ) = self._compute_grasp_transforms(
                hand_rot_left,
                hand_pos_left,
                self.robot_local_grasp_rot[env_ids_all],
                self.robot_local_grasp_pos[env_ids_all],
                drawer_rot,
                drawer_pos,
                self.drawer_local_grasp_rot[env_ids_all],
                self.drawer_local_grasp_pos[env_ids_all],
            )
             # Compute grasp transforms for Right Robot (only need robot part, drawer is same)
            (
                self.robot_right_grasp_rot[env_ids_all],
                self.robot_right_grasp_pos[env_ids_all],
                 _, # Discard drawer rot (already computed)
                 _, # Discard drawer pos (already computed)
            ) = self._compute_grasp_transforms(
                hand_rot_right,
                hand_pos_right,
                self.robot_local_grasp_rot[env_ids_all], # Use same local offset
                self.robot_local_grasp_pos[env_ids_all],
                drawer_rot, # Need these inputs for the function
                drawer_pos,
                self.drawer_local_grasp_rot[env_ids_all],
                self.drawer_local_grasp_pos[env_ids_all],
            )
        else:
             # Compute for Robot Left
            hand_pos_left = self._robot_left.data.body_pos_w[env_ids, self.hand_link_idx_left]
            hand_rot_left = self._robot_left.data.body_quat_w[env_ids, self.hand_link_idx_left]
            # Compute for Robot Right
            hand_pos_right = self._robot_right.data.body_pos_w[env_ids, self.hand_link_idx_right]
            hand_rot_right = self._robot_right.data.body_quat_w[env_ids, self.hand_link_idx_right]
            # Compute for Cabinet
            drawer_pos = self._cabinet.data.body_pos_w[env_ids, self.drawer_link_idx]
            drawer_rot = self._cabinet.data.body_quat_w[env_ids, self.drawer_link_idx]

            # Compute grasp transforms for Left Robot
            (
                self.robot_left_grasp_rot[env_ids],
                self.robot_left_grasp_pos[env_ids],
                self.drawer_grasp_rot[env_ids], # Drawer grasp is computed once
                self.drawer_grasp_pos[env_ids],
            ) = self._compute_grasp_transforms(
                hand_rot_left,
                hand_pos_left,
                self.robot_local_grasp_rot[env_ids],
                self.robot_local_grasp_pos[env_ids],
                drawer_rot,
                drawer_pos,
                self.drawer_local_grasp_rot[env_ids],
                self.drawer_local_grasp_pos[env_ids],
            )
             # Compute grasp transforms for Right Robot (only need robot part, drawer is same)
            (
                self.robot_right_grasp_rot[env_ids],
                self.robot_right_grasp_pos[env_ids],
                _, # Discard drawer rot (already computed)
                _, # Discard drawer pos (already computed)
            ) = self._compute_grasp_transforms(
                hand_rot_right,
                hand_pos_right,
                self.robot_local_grasp_rot[env_ids], # Use same local offset
                self.robot_local_grasp_pos[env_ids],
                drawer_rot, # Need these inputs for the function
                drawer_pos,
                self.drawer_local_grasp_rot[env_ids],
                self.drawer_local_grasp_pos[env_ids],
            )

    def _compute_rewards(
        self,
        actions, # Combined actions [N, 18]
        cabinet_dof_pos,
        # Left Robot
        franka_left_grasp_pos,
        franka_left_grasp_rot,
        franka_left_lfinger_pos,
        franka_left_rfinger_pos,
        # Right Robot
        franka_right_grasp_pos,
        franka_right_grasp_rot,
        franka_right_lfinger_pos,
        franka_right_rfinger_pos,
        # Target
        drawer_grasp_pos,
        drawer_grasp_rot,
        # Axes
        gripper_forward_axis,
        drawer_inward_axis,
        gripper_up_axis,
        drawer_up_axis,
        # Env info
        num_envs,
        # Scales
        dist_reward_scale,
        rot_reward_scale,
        open_reward_scale,
        action_penalty_scale,
        finger_reward_scale,
    ):
        # Calculate rewards for left robot
        d_left = torch.norm(franka_left_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
        dist_reward_left = 1.0 / (1.0 + d_left**2)
        dist_reward_left *= dist_reward_left
        dist_reward_left = torch.where(d_left <= 0.02, dist_reward_left * 2, dist_reward_left)

        axis1_left = tf_vector(franka_left_grasp_rot, gripper_forward_axis)
        axis2 = tf_vector(drawer_grasp_rot, drawer_inward_axis) # Target axis is the same
        axis3_left = tf_vector(franka_left_grasp_rot, gripper_up_axis)
        axis4 = tf_vector(drawer_grasp_rot, drawer_up_axis) # Target axis is the same

        dot1_left = torch.bmm(axis1_left.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        dot2_left = torch.bmm(axis3_left.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        rot_reward_left = 0.5 * (torch.sign(dot1_left) * dot1_left**2 + torch.sign(dot2_left) * dot2_left**2)

        lfinger_dist_left = franka_left_lfinger_pos[:, 2] - drawer_grasp_pos[:, 2]
        rfinger_dist_left = drawer_grasp_pos[:, 2] - franka_left_rfinger_pos[:, 2]
        finger_dist_penalty_left = torch.zeros_like(lfinger_dist_left)
        finger_dist_penalty_left += torch.where(lfinger_dist_left < 0, lfinger_dist_left, torch.zeros_like(lfinger_dist_left))
        finger_dist_penalty_left += torch.where(rfinger_dist_left < 0, rfinger_dist_left, torch.zeros_like(rfinger_dist_left))

        # Calculate rewards for right robot
        d_right = torch.norm(franka_right_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
        dist_reward_right = 1.0 / (1.0 + d_right**2)
        dist_reward_right *= dist_reward_right
        dist_reward_right = torch.where(d_right <= 0.02, dist_reward_right * 2, dist_reward_right)

        axis1_right = tf_vector(franka_right_grasp_rot, gripper_forward_axis)
        # axis2 is the same
        axis3_right = tf_vector(franka_right_grasp_rot, gripper_up_axis)
        # axis4 is the same

        dot1_right = torch.bmm(axis1_right.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        dot2_right = torch.bmm(axis3_right.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        rot_reward_right = 0.5 * (torch.sign(dot1_right) * dot1_right**2 + torch.sign(dot2_right) * dot2_right**2)

        lfinger_dist_right = franka_right_lfinger_pos[:, 2] - drawer_grasp_pos[:, 2]
        rfinger_dist_right = drawer_grasp_pos[:, 2] - franka_right_rfinger_pos[:, 2]
        finger_dist_penalty_right = torch.zeros_like(lfinger_dist_right)
        finger_dist_penalty_right += torch.where(lfinger_dist_right < 0, lfinger_dist_right, torch.zeros_like(lfinger_dist_right))
        finger_dist_penalty_right += torch.where(rfinger_dist_right < 0, rfinger_dist_right, torch.zeros_like(rfinger_dist_right))


        # Combine rewards (average the individual components)
        dist_reward = (dist_reward_left + dist_reward_right) / 2.0
        rot_reward = (rot_reward_left + rot_reward_right) / 2.0
        finger_dist_penalty = (finger_dist_penalty_left + finger_dist_penalty_right) / 2.0

        # regularization on the actions (summed for each environment across both robots)
        action_penalty = torch.sum(actions**2, dim=-1)

        # how far the cabinet has been opened out (same as before)
        open_reward = cabinet_dof_pos[:, 3]  # drawer_top_joint

        # Combined total reward
        rewards = (
            dist_reward_scale * dist_reward
            + rot_reward_scale * rot_reward
            + open_reward_scale * open_reward
            + finger_reward_scale * finger_dist_penalty # Note: this is a penalty, so it should be negative
            - action_penalty_scale * action_penalty
        )

        # Update logging extras
        self.extras["log"] = {
            "dist_reward_left": (dist_reward_scale * dist_reward_left).mean(),
            "rot_reward_left": (rot_reward_scale * rot_reward_left).mean(),
            "finger_dist_penalty_left": (finger_reward_scale * finger_dist_penalty_left).mean(),
            "dist_reward_right": (dist_reward_scale * dist_reward_right).mean(),
            "rot_reward_right": (rot_reward_scale * rot_reward_right).mean(),
            "finger_dist_penalty_right": (finger_reward_scale * finger_dist_penalty_right).mean(),
            "dist_reward_avg": (dist_reward_scale * dist_reward).mean(),
            "rot_reward_avg": (rot_reward_scale * rot_reward).mean(),
            "finger_dist_penalty_avg": (finger_reward_scale * finger_dist_penalty).mean(),
            "open_reward": (open_reward_scale * open_reward).mean(),
            "action_penalty": (-action_penalty_scale * action_penalty).mean(),
        }

        # bonus for opening drawer properly (same as before)
        rewards = torch.where(cabinet_dof_pos[:, 3] > 0.01, rewards + 0.25, rewards)
        rewards = torch.where(cabinet_dof_pos[:, 3] > 0.2, rewards + 0.25, rewards)
        rewards = torch.where(cabinet_dof_pos[:, 3] > 0.35, rewards + 0.25, rewards)

        return rewards

    def _compute_grasp_transforms(
        self,
        hand_rot,
        hand_pos,
        franka_local_grasp_rot,
        franka_local_grasp_pos,
        drawer_rot,
        drawer_pos,
        drawer_local_grasp_rot,
        drawer_local_grasp_pos,
    ):
        global_franka_rot, global_franka_pos = tf_combine(
            hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos
        )
        global_drawer_rot, global_drawer_pos = tf_combine(
            drawer_rot, drawer_pos, drawer_local_grasp_rot, drawer_local_grasp_pos
        )

        return global_franka_rot, global_franka_pos, global_drawer_rot, global_drawer_pos
