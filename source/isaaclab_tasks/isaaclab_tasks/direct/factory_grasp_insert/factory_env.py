# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch

import carb
import isaacsim.core.utils.torch as torch_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat

from . import factory_control as fc
from .factory_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG, FactoryEnvCfg


class FactoryEnv(DirectRLEnv):
    cfg: FactoryEnvCfg

    def __init__(self, cfg: FactoryEnvCfg, render_mode: str | None = None, **kwargs):
        # Update number of obs/states
        cfg.observation_space = sum([OBS_DIM_CFG[obs] for obs in cfg.obs_order])
        cfg.state_space = sum([STATE_DIM_CFG[state] for state in cfg.state_order])
        cfg.observation_space += cfg.action_space
        cfg.state_space += cfg.action_space
        self.cfg_task = cfg.task

        super().__init__(cfg, render_mode, **kwargs)

        v = getattr(self.cfg_task, "fixed_asset_speed", 0.005) 
        self._hole_linvel = torch.zeros((self.num_envs, 3), device=self.device)
        self._hole_linvel[:, 0] = v  

        if "fixed_linvel" in self.cfg.obs_order:
            self._fixed_linvel_obs = torch.zeros_like(self._hole_linvel)

        self._set_body_inertias()
        self._init_tensors()
        self._set_default_dynamics_parameters()
        self._compute_intermediate_values(dt=self.physics_dt)

    def _set_body_inertias(self):
        """Note: this is to account for the asset_options.armature parameter in IGE."""
        inertias = self._robot.root_physx_view.get_inertias()
        offset = torch.zeros_like(inertias)
        offset[:, :, [0, 4, 8]] += 0.01
        new_inertias = inertias + offset
        self._robot.root_physx_view.set_inertias(new_inertias, torch.arange(self.num_envs))

    def _set_default_dynamics_parameters(self):
        """Set parameters defining dynamic interactions."""
        self.default_gains = torch.tensor(self.cfg.ctrl.default_task_prop_gains, device=self.device).repeat(
            (self.num_envs, 1)
        )

        self.pos_threshold = torch.tensor(self.cfg.ctrl.pos_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.rot_threshold = torch.tensor(self.cfg.ctrl.rot_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )

        # Set masses and frictions.
        self._set_friction(self._held_asset, self.cfg_task.held_asset_cfg.friction)
        self._set_friction(self._fixed_asset, self.cfg_task.fixed_asset_cfg.friction)
        self._set_friction(self._robot, self.cfg_task.robot_cfg.friction)

    def _set_friction(self, asset, value):
        """Update material properties for a given asset."""
        materials = asset.root_physx_view.get_material_properties()
        materials[..., 0] = value  # Static friction.
        materials[..., 1] = value  # Dynamic friction.
        env_ids = torch.arange(self.scene.num_envs, device="cpu")
        asset.root_physx_view.set_material_properties(materials, env_ids)

    def _init_tensors(self):
        """Initialize tensors once."""
        self.identity_quat = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        )

        # Control targets.
        self.ctrl_target_joint_pos = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.ctrl_target_fingertip_midpoint_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.ctrl_target_fingertip_midpoint_quat = torch.zeros((self.num_envs, 4), device=self.device)

        # Fixed asset.
        self.fixed_pos_action_frame = torch.zeros((self.num_envs, 3), device=self.device)
        self.fixed_pos_obs_frame = torch.zeros((self.num_envs, 3), device=self.device)
        self.init_fixed_pos_obs_noise = torch.zeros((self.num_envs, 3), device=self.device)

        # Held asset
        held_base_x_offset = 0.0
        if self.cfg_task.name == "peg_insert":
            held_base_z_offset = 0.0
        elif self.cfg_task.name == "gear_mesh":
            gear_base_offset = self._get_target_gear_base_offset()
            held_base_x_offset = gear_base_offset[0]
            held_base_z_offset = gear_base_offset[2]
        elif self.cfg_task.name == "nut_thread":
            held_base_z_offset = self.cfg_task.fixed_asset_cfg.base_height
        else:
            raise NotImplementedError("Task not implemented")

        self.held_base_pos_local = torch.tensor([0.0, 0.0, 0.0], device=self.device).repeat((self.num_envs, 1))
        self.held_base_pos_local[:, 0] = held_base_x_offset
        self.held_base_pos_local[:, 2] = held_base_z_offset
        self.held_base_quat_local = self.identity_quat.clone().detach()

        self.held_base_pos = torch.zeros_like(self.held_base_pos_local)
        self.held_base_quat = self.identity_quat.clone().detach()

        # Computer body indices.
        self.left_finger_body_idx = self._robot.body_names.index("panda_leftfinger")
        self.right_finger_body_idx = self._robot.body_names.index("panda_rightfinger")
        self.fingertip_body_idx = self._robot.body_names.index("panda_fingertip_centered")

        # Tensors for finite-differencing.
        self.last_update_timestamp = 0.0  # Note: This is for finite differencing body velocities.
        self.prev_fingertip_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_fingertip_quat = self.identity_quat.clone()
        self.prev_joint_pos = torch.zeros((self.num_envs, 7), device=self.device)

        # Keypoint tensors.
        self.target_held_base_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_held_base_quat = self.identity_quat.clone().detach()

        offsets = self._get_keypoint_offsets(self.cfg_task.num_keypoints)
        self.keypoint_offsets = offsets * self.cfg_task.keypoint_scale
        self.keypoints_held = torch.zeros((self.num_envs, self.cfg_task.num_keypoints, 3), device=self.device)
        self.keypoints_fixed = torch.zeros_like(self.keypoints_held, device=self.device)

        # Used to compute target poses.
        self.fixed_success_pos_local = torch.zeros((self.num_envs, 3), device=self.device)
        if self.cfg_task.name == "peg_insert":
            self.fixed_success_pos_local[:, 2] = 0.0
        elif self.cfg_task.name == "gear_mesh":
            gear_base_offset = self._get_target_gear_base_offset()
            self.fixed_success_pos_local[:, 0] = gear_base_offset[0]
            self.fixed_success_pos_local[:, 2] = gear_base_offset[2]
        elif self.cfg_task.name == "nut_thread":
            head_height = self.cfg_task.fixed_asset_cfg.base_height
            shank_length = self.cfg_task.fixed_asset_cfg.height
            thread_pitch = self.cfg_task.fixed_asset_cfg.thread_pitch
            self.fixed_success_pos_local[:, 2] = head_height + shank_length - thread_pitch * 1.5
        else:
            raise NotImplementedError("Task not implemented")

        self.ep_succeeded = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.ep_success_times = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.stage = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.prev_held_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.align_hold_counter = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

    def _get_keypoint_offsets(self, num_keypoints):
        """Get uniformly-spaced keypoints along a line of unit length, centered at 0."""
        keypoint_offsets = torch.zeros((num_keypoints, 3), device=self.device)
        keypoint_offsets[:, -1] = torch.linspace(0.0, 1.0, num_keypoints, device=self.device) - 0.5

        return keypoint_offsets

    def _setup_scene(self):
        """Initialize simulation scene."""
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))

        # spawn a usd file of a table into the scene
        cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
        cfg.func(
            "/World/envs/env_.*/Table", cfg, translation=(0.55, 0.0, 0.0), orientation=(0.70711, 0.0, 0.0, 0.70711)
        )

        self._robot = Articulation(self.cfg.robot)
        self._fixed_asset = Articulation(self.cfg_task.fixed_asset)
        self._held_asset = Articulation(self.cfg_task.held_asset)
        if self.cfg_task.name == "gear_mesh":
            self._small_gear_asset = Articulation(self.cfg_task.small_gear_cfg)
            self._large_gear_asset = Articulation(self.cfg_task.large_gear_cfg)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            # we need to explicitly filter collisions for CPU simulation
            self.scene.filter_collisions()

        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["fixed_asset"] = self._fixed_asset
        self.scene.articulations["held_asset"] = self._held_asset
        if self.cfg_task.name == "gear_mesh":
            self.scene.articulations["small_gear"] = self._small_gear_asset
            self.scene.articulations["large_gear"] = self._large_gear_asset

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _compute_intermediate_values(self, dt):
        """Get values computed from raw tensors. This includes adding noise."""
        # TODO: A lot of these can probably only be set once?
        self.fixed_pos = self._fixed_asset.data.root_pos_w - self.scene.env_origins
        self.fixed_quat = self._fixed_asset.data.root_quat_w

        self.held_pos = self._held_asset.data.root_pos_w - self.scene.env_origins
        self.held_quat = self._held_asset.data.root_quat_w

        self.fingertip_midpoint_pos = self._robot.data.body_pos_w[:, self.fingertip_body_idx] - self.scene.env_origins
        self.fingertip_midpoint_quat = self._robot.data.body_quat_w[:, self.fingertip_body_idx]
        self.fingertip_midpoint_linvel = self._robot.data.body_lin_vel_w[:, self.fingertip_body_idx]
        self.fingertip_midpoint_angvel = self._robot.data.body_ang_vel_w[:, self.fingertip_body_idx]

        jacobians = self._robot.root_physx_view.get_jacobians()

        self.left_finger_jacobian = jacobians[:, self.left_finger_body_idx - 1, 0:6, 0:7]
        self.right_finger_jacobian = jacobians[:, self.right_finger_body_idx - 1, 0:6, 0:7]
        self.fingertip_midpoint_jacobian = (self.left_finger_jacobian + self.right_finger_jacobian) * 0.5
        self.arm_mass_matrix = self._robot.root_physx_view.get_generalized_mass_matrices()[:, 0:7, 0:7]
        self.joint_pos = self._robot.data.joint_pos.clone()
        self.joint_vel = self._robot.data.joint_vel.clone()

        # Finite-differencing results in more reliable velocity estimates.
        self.ee_linvel_fd = (self.fingertip_midpoint_pos - self.prev_fingertip_pos) / dt
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()

        # Add state differences if velocity isn't being added.
        rot_diff_quat = torch_utils.quat_mul(
            self.fingertip_midpoint_quat, torch_utils.quat_conjugate(self.prev_fingertip_quat)
        )
        rot_diff_quat *= torch.sign(rot_diff_quat[:, 0]).unsqueeze(-1)
        rot_diff_aa = axis_angle_from_quat(rot_diff_quat)
        self.ee_angvel_fd = rot_diff_aa / dt
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()

        joint_diff = self.joint_pos[:, 0:7] - self.prev_joint_pos
        self.joint_vel_fd = joint_diff / dt
        self.prev_joint_pos = self.joint_pos[:, 0:7].clone()

        # Keypoint tensors.
        self.held_base_quat[:], self.held_base_pos[:] = torch_utils.tf_combine(
            self.held_quat, self.held_pos, self.held_base_quat_local, self.held_base_pos_local
        )
        self.target_held_base_quat[:], self.target_held_base_pos[:] = torch_utils.tf_combine(
            self.fixed_quat, self.fixed_pos, self.identity_quat, self.fixed_success_pos_local
        )

        # Compute pos of keypoints on held asset, and fixed asset in world frame
        for idx, keypoint_offset in enumerate(self.keypoint_offsets):
            self.keypoints_held[:, idx] = torch_utils.tf_combine(
                self.held_base_quat, self.held_base_pos, self.identity_quat, keypoint_offset.repeat(self.num_envs, 1)
            )[1]
            self.keypoints_fixed[:, idx] = torch_utils.tf_combine(
                self.target_held_base_quat,
                self.target_held_base_pos,
                self.identity_quat,
                keypoint_offset.repeat(self.num_envs, 1),
            )[1]

        self.keypoint_dist = torch.norm(self.keypoints_held - self.keypoints_fixed, p=2, dim=-1).mean(-1)
        self.last_update_timestamp = self._robot._data._sim_timestamp

    def _get_observations(self):
        """Get actor/critic inputs using asymmetric critic."""
        prev_actions = self.actions.clone()
        obs_dict = {
            "fingertip_pos": self.fingertip_midpoint_pos,
            "fingertip_pos_rel_fixed": self.fingertip_midpoint_pos - self.fixed_pos,
            "fingertip_quat": self.fingertip_midpoint_quat,
            "ee_linvel": self.ee_linvel_fd,
            "ee_angvel": self.ee_angvel_fd,
            "fixed_linvel": self._hole_linvel,     
            "prev_actions": prev_actions,
            "gripper_gap": self.joint_pos[:, 7].unsqueeze(-1),
        }
        state_dict = {
            "fingertip_pos":self.fingertip_midpoint_pos,
            "fingertip_pos_rel_fixed": self.fingertip_midpoint_pos - self.fixed_pos,
            "fingertip_quat": self.fingertip_midpoint_quat,
            "ee_linvel": self.fingertip_midpoint_linvel,
            "ee_angvel": self.fingertip_midpoint_angvel,
            "joint_pos": self.joint_pos[:, 0:7],
            "held_pos": self.held_pos,
            "held_pos_rel_fixed": self.held_pos - self.fixed_pos,
            "held_quat": self.held_quat,
            "fixed_pos": self.fixed_pos,
            "fixed_quat": self.fixed_quat,
            "task_prop_gains": self.task_prop_gains,
            "pos_threshold": self.pos_threshold,
            "rot_threshold": self.rot_threshold,
            "fixed_linvel": self._hole_linvel,     
            "prev_actions": prev_actions,
            "gripper_gap": self.joint_pos[:, 7].unsqueeze(-1),
        }
        obs_tensors = [obs_dict[obs_name] for obs_name in self.cfg.obs_order + ["prev_actions"]]
        obs_tensors = torch.cat(obs_tensors, dim=-1)
        state_tensors = [state_dict[state_name] for state_name in self.cfg.state_order + ["prev_actions"]]
        state_tensors = torch.cat(state_tensors, dim=-1)
        return {"policy": obs_tensors, "critic": state_tensors}

    def _reset_buffers(self, env_ids):
        """Reset buffers."""
        self.ep_succeeded[env_ids] = 0
        self.stage[env_ids] = 0

    def _pre_physics_step(self, action):
        """Apply policy actions with smoothing."""
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self._reset_buffers(env_ids)

        self.actions = (
            self.cfg.ctrl.ema_factor * action.clone().to(self.device) + (1 - self.cfg.ctrl.ema_factor) * self.actions
        )

    def close_gripper_in_place(self):
        """Keep gripper in current position as gripper closes."""
        actions = torch.zeros((self.num_envs, 6), device=self.device)
        ctrl_target_gripper_dof_pos = 0.0

        # Interpret actions as target pos displacements and set pos target
        pos_actions = actions[:, 0:3] * self.pos_threshold
        self.ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = actions[:, 3:6]

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)

        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)

        rot_actions_quat = torch.where(
            angle.unsqueeze(-1).repeat(1, 4) > 1.0e-6,
            rot_actions_quat,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1),
        )
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)

        target_euler_xyz = torch.stack(torch_utils.get_euler_xyz(self.ctrl_target_fingertip_midpoint_quat), dim=1)
        target_euler_xyz[:, 0] = 3.14159
        target_euler_xyz[:, 1] = 0.0

        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            roll=target_euler_xyz[:, 0], pitch=target_euler_xyz[:, 1], yaw=target_euler_xyz[:, 2]
        )

        self.ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos
        self.generate_ctrl_signals()

    def _debug_trace(self, env_id: int = 0):
        # make sure cached tensors are up‑to‑date
        if self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values(dt=self.physics_dt)

        # ----- geometry for env_id -----
        xy_err = torch.linalg.norm(
            self.fingertip_midpoint_pos[env_id, :2] - self.held_pos[env_id, :2]
        ).item() * 1e3                       # [mm]

        peg_top_z = self.held_pos[env_id,2] + self.cfg_task.held_asset_cfg.height * self.cfg_task.held_asset_grasp_height_frac
        z_offset  = (self.fingertip_midpoint_pos[env_id,2] - peg_top_z).item()*1e3  # [mm]

        gap = (self.joint_pos[env_id, 7] * 1e3).item()  # [mm]

        print(
            f"stage={int(self.stage[env_id])}  "
            f"xy_err={xy_err:5.1f} mm  "
            f"z_off={z_offset:5.1f} mm  "
            f"gap={gap:5.1f} mm  "
            f"grip action range: {self.actions[:,6].min():+.2f} … {self.actions[:,6].max():+.2f}"
        )
        
    def _apply_action(self):
        """Apply actions for policy as delta targets from current position."""
        # self._debug_trace(env_id=17)

        new_pos = self._fixed_asset.data.root_pos_w + self._hole_linvel * self.physics_dt
        new_quat = self._fixed_asset.data.root_quat_w
        env_ids  = torch.arange(self.num_envs, device=self.device)
        self._fixed_asset.write_root_pose_to_sim(
            torch.cat((new_pos, new_quat), dim=-1), env_ids=env_ids
        )

        # Get current yaw for success checking.
        _, _, curr_yaw = torch_utils.get_euler_xyz(self.fingertip_midpoint_quat)
        self.curr_yaw = torch.where(curr_yaw > np.deg2rad(235), curr_yaw - 2 * np.pi, curr_yaw)

        # Note: We use finite-differenced velocities for control and observations.
        # Check if we need to re-compute velocities within the decimation loop.
        if self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values(dt=self.physics_dt)

        # Interpret actions as target pos displacements and set pos target
        pos_actions = self.actions[:, 0:3] * self.pos_threshold

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = self.actions[:, 3:6]
        if self.cfg_task.unidirectional_rot:
            rot_actions[:, 2] = -(rot_actions[:, 2] + 1.0) * 0.5  # [-1, 0]
        rot_actions = rot_actions * self.rot_threshold

        self.ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions

        reference_pos = torch.where(
            (self.stage == 0).unsqueeze(-1),   
            self.held_pos,                    
            self.fixed_pos                    
        )
        # To speed up learning, never allow the policy to move more than 5cm away from the base.
        delta_pos = pos_actions
        delta_pos = torch.clip(
            delta_pos,
            -self.cfg.ctrl.pos_action_bounds[0],
            self.cfg.ctrl.pos_action_bounds[1],
        )
        self.ctrl_target_fingertip_midpoint_pos = reference_pos + delta_pos

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)

        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        rot_actions_quat = torch.where(
            angle.unsqueeze(-1).repeat(1, 4) > 1e-6,
            rot_actions_quat,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1),
        )
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)

        target_euler_xyz = torch.stack(torch_utils.get_euler_xyz(self.ctrl_target_fingertip_midpoint_quat), dim=1)
        target_euler_xyz[:, 0] = 3.14159  # Restrict actions to be upright.
        target_euler_xyz[:, 1] = 0.0

        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            roll=target_euler_xyz[:, 0], pitch=target_euler_xyz[:, 1], yaw=target_euler_xyz[:, 2]
        )

        a = self.actions[:, 6]                # network out
        frac_closed = (torch.tanh(a) + 1.0) * 0.5      # 0 = open, 1 = closed

        MAX_GAP = 0.04                            
        desired_gap = (1.0 - frac_closed) * MAX_GAP    # 0 - 0.04 m gap
        self.ctrl_target_gripper_dof_pos = desired_gap.unsqueeze(-1).repeat(1, 2)
        
        # Gripper auto-close
        xy_err = torch.linalg.norm(
            self.fingertip_midpoint_pos[:, :2] - self.held_pos[:, :2], dim=1
        )
        peg_75_z   = self.held_pos[:, 2] + \
                    self.cfg_task.held_asset_grasp_height_frac * self.cfg_task.held_asset_cfg.height
        z_clear75  = self.fingertip_midpoint_pos[:, 2] - peg_75_z

        good_align = (
            (xy_err < 0.008) &                      
            (z_clear75 > -0.002) & (z_clear75 < 0.008)  
        )

        self.align_hold_counter[good_align] += 1
        self.align_hold_counter[~good_align] = 0

        auto_mask = (self.align_hold_counter >= 20) & (self.joint_pos[:, 7] > 0.006)
        if auto_mask.any():
            self.ctrl_target_gripper_dof_pos[auto_mask, :] = 0.0055

        self.generate_ctrl_signals()

    def _set_gains(self, prop_gains, rot_deriv_scale=1.0):
        """Set robot gains using critical damping."""
        self.task_prop_gains = prop_gains
        self.task_deriv_gains = 2 * torch.sqrt(prop_gains)
        self.task_deriv_gains[:, 3:6] /= rot_deriv_scale

    def generate_ctrl_signals(self):
        """Get Jacobian. Set Franka DOF position targets (fingers) or DOF torques (arm)."""
        self.joint_torque, self.applied_wrench = fc.compute_dof_torque(
            cfg=self.cfg,
            dof_pos=self.joint_pos,
            dof_vel=self.joint_vel,  # _fd,
            fingertip_midpoint_pos=self.fingertip_midpoint_pos,
            fingertip_midpoint_quat=self.fingertip_midpoint_quat,
            fingertip_midpoint_linvel=self.ee_linvel_fd,
            fingertip_midpoint_angvel=self.ee_angvel_fd,
            jacobian=self.fingertip_midpoint_jacobian,
            arm_mass_matrix=self.arm_mass_matrix,
            ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat,
            task_prop_gains=self.task_prop_gains,
            task_deriv_gains=self.task_deriv_gains,
            device=self.device,
        )

        # set target for gripper joints to use physx's PD controller
        self.ctrl_target_joint_pos[:, 7:9] = self.ctrl_target_gripper_dof_pos
        self.joint_torque[:, 7:9] = 0.0

        self._robot.set_joint_position_target(self.ctrl_target_joint_pos)
        self._robot.set_joint_effort_target(self.joint_torque)

    def _get_dones(self):
        """Update intermediate values used for rewards and observations."""
        self._compute_intermediate_values(dt=self.physics_dt)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return time_out, time_out

    def _get_curr_successes(self, success_threshold, check_rot=False):
        """Get success mask at current timestep."""
        curr_successes = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        xy_dist = torch.linalg.vector_norm(self.target_held_base_pos[:, 0:2] - self.held_base_pos[:, 0:2], dim=1)
        z_disp = self.held_base_pos[:, 2] - self.target_held_base_pos[:, 2]

        is_centered = torch.where(xy_dist < 0.0025, torch.ones_like(curr_successes), torch.zeros_like(curr_successes))
        # Height threshold to target
        fixed_cfg = self.cfg_task.fixed_asset_cfg
        if self.cfg_task.name == "peg_insert" or self.cfg_task.name == "gear_mesh":
            height_threshold = fixed_cfg.height * success_threshold
        elif self.cfg_task.name == "nut_thread":
            height_threshold = fixed_cfg.thread_pitch * success_threshold
        else:
            raise NotImplementedError("Task not implemented")
        is_close_or_below = torch.where(
            z_disp < height_threshold, torch.ones_like(curr_successes), torch.zeros_like(curr_successes)
        )
        curr_successes = torch.logical_and(is_centered, is_close_or_below)

        if check_rot:
            is_rotated = self.curr_yaw < self.cfg_task.ee_success_yaw
            curr_successes = torch.logical_and(curr_successes, is_rotated)
        
        fingers_closed = self.joint_pos[:, 7] < 0.006
        curr_successes = torch.logical_and(curr_successes, fingers_closed)
        return curr_successes

    def _compute_stage(self):
        """
        Finite-state machine: 0-GRASP 1-LIFT 2-INSERT.
        """
        ST_GRASP, ST_LIFT, ST_INSERT = 0, 1, 2

        h = self.cfg_task.held_asset_cfg.height           # peg height (m)
        min_gap_closed = 0.006                            # fingers ≤ 6 mm ⇒ closed
        target_clr = self.cfg_task.clearance_height       # lift goal above hole (m)

        # ------------------------------------------------------------------
        # 1)   Secure grasp?
        # ------------------------------------------------------------------
        xy_err_peg = torch.linalg.norm(
            self.fingertip_midpoint_pos[:, :2] - self.held_pos[:, :2], dim=1
        )
        peg_top_z   = self.held_pos[:, 2] + self.cfg_task.held_asset_grasp_height_frac * h
        z_clear_top = self.fingertip_midpoint_pos[:, 2] - peg_top_z
        finger_gap  = self.joint_pos[:, 7]

        grasp_ok = (
            (xy_err_peg < 0.008) &
            (torch.abs(z_clear_top) < 0.002) &
            (finger_gap < min_gap_closed)
        )

        # ------------------------------------------------------------------
        # 2)   Lifted high enough while *in* the LIFT stage?
        # ------------------------------------------------------------------
        lifted_now = (self.held_pos[:, 2] - self.fixed_pos[:, 2]) > target_clr
        lifted_ok  = (self.stage == ST_LIFT) & lifted_now

        # ------------------------------------------------------------------
        # 3)   Transitions
        # ------------------------------------------------------------------
        next_stage = self.stage.clone()
        next_stage[(self.stage == ST_GRASP) & grasp_ok] = ST_LIFT
        next_stage[lifted_ok]                           = ST_INSERT
        return next_stage


    def _get_rewards(self):
        """Update rewards and compute success statistics."""
        # Sync stage
        self.stage = self._compute_stage()
        ST_GRASP, ST_LIFT, ST_INSERT = 0, 1, 2

        rew = torch.zeros((self.num_envs,), device=self.device)

        # ------------------------------------------------------------------
        #  Shared geometry
        # ------------------------------------------------------------------
        h = self.cfg_task.held_asset_cfg.height
        target_clr = self.cfg_task.clearance_height

        # NB: most Stage‑0 variables were dropped – only one distance is used.
        grasp_target_pos = self.held_pos.clone()          # (N,3)
        grasp_target_pos[:, 2] += self.cfg_task.held_asset_grasp_height_frac * h                # 75 % up the peg

        pos_err = torch.linalg.norm(
            self.fingertip_midpoint_pos - grasp_target_pos, dim=1
        )

        # ===========================  STAGE 0 — GRASP  ==========================
        mask = self.stage == ST_GRASP
        if mask.any():
            # single dense reward:  0 → +2 as xyz error → 0
            rew[mask] += 2.0 * torch.exp(-200.0 * pos_err)[mask]

        # ===========================  STAGE 1 — LIFT   ==========================
        mask = self.stage == ST_LIFT
        if mask.any():
            height = (self.held_pos[:, 2] - self.fixed_pos[:, 2]).clamp(0.0, target_clr)
            lift_bonus = 2.0 * height / target_clr         # 0→2 as we reach clearance
            rew[mask] += lift_bonus[mask]

        # ===========================  STAGE 2 — INSERT ==========================
        mask = self.stage == ST_INSERT
        if mask.any():
            # --- key‑point shaping (unchanged)
            def _sq(x, a, b):
                return 1.0 / (torch.exp(a * x) + b + torch.exp(-a * x))
            a0, b0 = self.cfg_task.keypoint_coef_baseline
            a1, b1 = self.cfg_task.keypoint_coef_coarse
            a2, b2 = self.cfg_task.keypoint_coef_fine
            kp = (_sq(self.keypoint_dist, a0, b0) +
                  _sq(self.keypoint_dist, a1, b1) +
                  _sq(self.keypoint_dist, a2, b2))

            xy_err_hole = torch.linalg.norm(
                self.fingertip_midpoint_pos[:, :2] - self.fixed_pos[:, :2], dim=1
            )
            xy_align_hole = torch.exp(-50.0 * xy_err_hole)

            z_clear = self.fingertip_midpoint_pos[:, 2] - self.fixed_pos[:, 2]
            early_pen = torch.relu(0.01 - z_clear) * torch.relu(xy_err_hole - 0.01)

            engaged = self._get_curr_successes(
                self.cfg_task.engage_threshold, check_rot=False
            ).float()
            success = self._get_curr_successes(
                self.cfg_task.success_threshold, check_rot=False
            ).float()

            rew[mask] += (kp + xy_align_hole - 5.0 * early_pen + engaged + success)[mask]

        # ------------------------------------------------------------------
        #  Book‑keeping
        # ------------------------------------------------------------------
        self.reward_buf = rew
        return rew

    def _reset_idx(self, env_ids):
        """
        We assume all envs will always be reset at the same time.
        """
        super()._reset_idx(env_ids)

        self._set_assets_to_default_pose(env_ids)
        self._set_franka_to_default_pose(joints=self.cfg.ctrl.reset_joints, env_ids=env_ids)
        self.step_sim_no_action()

        self.randomize_initial_state(env_ids)

    def _get_target_gear_base_offset(self):
        """Get offset of target gear from the gear base asset."""
        target_gear = self.cfg_task.target_gear
        if target_gear == "gear_large":
            gear_base_offset = self.cfg_task.fixed_asset_cfg.large_gear_base_offset
        elif target_gear == "gear_medium":
            gear_base_offset = self.cfg_task.fixed_asset_cfg.medium_gear_base_offset
        elif target_gear == "gear_small":
            gear_base_offset = self.cfg_task.fixed_asset_cfg.small_gear_base_offset
        else:
            raise ValueError(f"{target_gear} not valid in this context!")
        return gear_base_offset

    def _set_assets_to_default_pose(self, env_ids):
        """Move assets to default pose before randomization."""
        held_state = self._held_asset.data.default_root_state.clone()[env_ids]
        held_state[:, 0:3] += self.scene.env_origins[env_ids]
        held_state[:, 7:] = 0.0
        self._held_asset.write_root_pose_to_sim(held_state[:, 0:7], env_ids=env_ids)
        self._held_asset.write_root_velocity_to_sim(held_state[:, 7:], env_ids=env_ids)
        self._held_asset.reset()

        fixed_state = self._fixed_asset.data.default_root_state.clone()[env_ids]
        fixed_state[:, 0:3] += self.scene.env_origins[env_ids]
        fixed_state[:, 7:] = 0.0
        self._fixed_asset.write_root_pose_to_sim(fixed_state[:, 0:7], env_ids=env_ids)
        self._fixed_asset.write_root_velocity_to_sim(fixed_state[:, 7:], env_ids=env_ids)
        self._fixed_asset.reset()

    def set_pos_inverse_kinematics(self, env_ids):
        """Set robot joint position using DLS IK."""
        ik_time = 0.0
        while ik_time < 0.25:
            # Compute error to target.
            pos_error, axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos[env_ids],
                fingertip_midpoint_quat=self.fingertip_midpoint_quat[env_ids],
                ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos[env_ids],
                ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat[env_ids],
                jacobian_type="geometric",
                rot_error_type="axis_angle",
            )

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)

            # Solve DLS problem.
            delta_dof_pos = fc._get_delta_dof_pos(
                delta_pose=delta_hand_pose,
                ik_method="dls",
                jacobian=self.fingertip_midpoint_jacobian[env_ids],
                device=self.device,
            )
            self.joint_pos[env_ids, 0:7] += delta_dof_pos[:, 0:7]
            self.joint_vel[env_ids, :] = torch.zeros_like(self.joint_pos[env_ids,])

            self.ctrl_target_joint_pos[env_ids, 0:7] = self.joint_pos[env_ids, 0:7]
            # Update dof state.
            self._robot.write_joint_state_to_sim(self.joint_pos, self.joint_vel)
            self._robot.set_joint_position_target(self.ctrl_target_joint_pos)

            # Simulate and update tensors.
            self.step_sim_no_action()
            ik_time += self.physics_dt

        return pos_error, axis_angle_error

    def get_handheld_asset_relative_pose(self):
        """Get default relative pose between help asset and fingertip."""
        if self.cfg_task.name == "peg_insert":
            held_asset_relative_pos = torch.zeros_like(self.held_base_pos_local)
            held_asset_relative_pos[:, 2] = self.cfg_task.held_asset_cfg.height
            held_asset_relative_pos[:, 2] -= self.cfg_task.robot_cfg.franka_fingerpad_length
        elif self.cfg_task.name == "gear_mesh":
            held_asset_relative_pos = torch.zeros_like(self.held_base_pos_local)
            gear_base_offset = self._get_target_gear_base_offset()
            held_asset_relative_pos[:, 0] += gear_base_offset[0]
            held_asset_relative_pos[:, 2] += gear_base_offset[2]
            held_asset_relative_pos[:, 2] += self.cfg_task.held_asset_cfg.height / 2.0 * 1.1
        elif self.cfg_task.name == "nut_thread":
            held_asset_relative_pos = self.held_base_pos_local
        else:
            raise NotImplementedError("Task not implemented")

        held_asset_relative_quat = self.identity_quat
        if self.cfg_task.name == "nut_thread":
            # Rotate along z-axis of frame for default position.
            initial_rot_deg = self.cfg_task.held_asset_rot_init
            rot_yaw_euler = torch.tensor([0.0, 0.0, initial_rot_deg * np.pi / 180.0], device=self.device).repeat(
                self.num_envs, 1
            )
            held_asset_relative_quat = torch_utils.quat_from_euler_xyz(
                roll=rot_yaw_euler[:, 0], pitch=rot_yaw_euler[:, 1], yaw=rot_yaw_euler[:, 2]
            )

        return held_asset_relative_pos, held_asset_relative_quat

    def _set_franka_to_default_pose(self, joints, env_ids):
        """Return Franka to its default joint position."""
        # gripper_width = self.cfg_task.held_asset_cfg.diameter / 2 * 1.25
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_pos[:, 7:] = 0.04
        joint_pos[:, :7] = torch.tensor(joints, device=self.device)[None, :]
        joint_vel = torch.zeros_like(joint_pos)
        joint_effort = torch.zeros_like(joint_pos)
        self.ctrl_target_joint_pos[env_ids, :] = joint_pos
        self._robot.set_joint_position_target(self.ctrl_target_joint_pos[env_ids], env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._robot.reset()
        self._robot.set_joint_effort_target(joint_effort, env_ids=env_ids)

        self.step_sim_no_action()

    def step_sim_no_action(self):
        """Step the simulation without an action. Used for resets."""
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=self.physics_dt)
        self._compute_intermediate_values(dt=self.physics_dt)

    def randomize_initial_state(self, env_ids):
        """Randomize initial state and perform any episode-level randomization."""
        # Disable gravity.
        physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view
        physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, 0.0))

        # (1.) Randomize fixed asset pose.
        fixed_state = self._fixed_asset.data.default_root_state.clone()[env_ids]
        # (1.a.) Position
        rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_pos_init_rand = 2 * (rand_sample - 0.5)  # [-1, 1]
        fixed_asset_init_pos_rand = torch.tensor(
            self.cfg_task.fixed_asset_init_pos_noise, dtype=torch.float32, device=self.device
        )
        fixed_pos_init_rand = fixed_pos_init_rand @ torch.diag(fixed_asset_init_pos_rand)
        fixed_state[:, 0:3] += fixed_pos_init_rand + self.scene.env_origins[env_ids]
        # (1.b.) Orientation
        fixed_orn_init_yaw = np.deg2rad(self.cfg_task.fixed_asset_init_orn_deg)
        fixed_orn_yaw_range = np.deg2rad(self.cfg_task.fixed_asset_init_orn_range_deg)
        rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_orn_euler = fixed_orn_init_yaw + fixed_orn_yaw_range * rand_sample
        fixed_orn_euler[:, 0:2] = 0.0  # Only change yaw.
        fixed_orn_quat = torch_utils.quat_from_euler_xyz(
            fixed_orn_euler[:, 0], fixed_orn_euler[:, 1], fixed_orn_euler[:, 2]
        )
        fixed_state[:, 3:7] = fixed_orn_quat
        # (1.c.) Velocity
        fixed_state[:, 7:] = 0.0  # vel
        # (1.d.) Update values.
        self._fixed_asset.write_root_pose_to_sim(fixed_state[:, 0:7], env_ids=env_ids)
        self._fixed_asset.write_root_velocity_to_sim(fixed_state[:, 7:], env_ids=env_ids)
        self._fixed_asset.reset()

        # (1.e.) Noisy position observation.
        fixed_asset_pos_noise = torch.randn((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_asset_pos_rand = torch.tensor(self.cfg.obs_rand.fixed_asset_pos, dtype=torch.float32, device=self.device)
        fixed_asset_pos_noise = fixed_asset_pos_noise @ torch.diag(fixed_asset_pos_rand)
        self.init_fixed_pos_obs_noise[:] = fixed_asset_pos_noise

        self.step_sim_no_action()

        # Compute the frame on the bolt that would be used as observation: fixed_pos_obs_frame
        # For example, the tip of the bolt can be used as the observation frame
        fixed_tip_pos_local = torch.zeros_like(self.fixed_pos)
        fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.height
        fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.base_height
        if self.cfg_task.name == "gear_mesh":
            fixed_tip_pos_local[:, 0] = self._get_target_gear_base_offset()[0]

        _, fixed_tip_pos = torch_utils.tf_combine(
            self.fixed_quat, self.fixed_pos, self.identity_quat, fixed_tip_pos_local
        )
        self.fixed_pos_obs_frame[:] = fixed_tip_pos

        # Spawn HELD asset on the table
        held_state = self._held_asset.data.default_root_state.clone()[env_ids]

        held_state[:, :3] += self.scene.env_origins[env_ids]        # world‑space
        pos_noise = torch.randn((len(env_ids), 3), device=self.device) * \
                torch.tensor(self.cfg_task.held_asset_pos_noise,
                             device=self.device)
        held_state[:, :3] += pos_noise
        held_state[:, 7:]  = 0.0

        self._held_asset.write_root_pose_to_sim(held_state[:, :7], env_ids=env_ids)
        self._held_asset.write_root_velocity_to_sim(held_state[:, 7:], env_ids=env_ids)
        self._held_asset.reset()

        # open fingers
        self.ctrl_target_joint_pos[env_ids, 7:] = 0.04                           # 4 cm opening
        self._robot.set_joint_position_target(self.ctrl_target_joint_pos[env_ids], env_ids=env_ids)

        origins = self.scene.env_origins[env_ids]       
        bad_envs = env_ids.clone()
        ik_attempt = 0
        POS_TOL = 1e-3     
        ANG_TOL = 1e-3   

        hand_down_quat = torch.zeros((self.num_envs, 4), device=self.device)
        while True:
            n_bad = bad_envs.numel()

            # target position
            above_pos_world = held_state[bad_envs, :3].clone()
            above_pos_world[:, 2] += 0.05                      # 5 cm above peg
            above_pos_local = above_pos_world - origins[bad_envs]

            # target orientation 
            base_eul = torch.tensor(
                self.cfg_task.hand_init_orn, device=self.device
            ).repeat(n_bad, 1)                 
            noise = (torch.rand((n_bad, 3), device=self.device) - 0.5) * 2.0
            noise *= torch.tensor(
                self.cfg_task.hand_init_orn_noise, device=self.device
            )                                   
            eul = base_eul + noise
            hand_down_quat[bad_envs] = torch_utils.quat_from_euler_xyz(eul[:,0], eul[:,1], eul[:,2])

            # IK targets and solve
            self.ctrl_target_fingertip_midpoint_pos[bad_envs]  = above_pos_local
            self.ctrl_target_fingertip_midpoint_quat[bad_envs] = hand_down_quat[bad_envs]

            pos_err, ang_err = self.set_pos_inverse_kinematics(bad_envs)

            bad_mask = torch.logical_or(
                torch.linalg.norm(pos_err, dim=1) > POS_TOL,
                torch.norm(ang_err, dim=1) > ANG_TOL
            )
            bad_envs = bad_envs[bad_mask.nonzero(as_tuple=False).squeeze(-1)]

            if not len(bad_envs) or ik_attempt >= 20:
                break

            # fallback: reset joints, re‑open fingers, step once
            self._set_franka_to_default_pose(
                joints=[0.00871, -0.10368, -0.00794, -1.49139, -0.00083, 1.38774, 0.0],
                env_ids=bad_envs
            )
            self.ctrl_target_joint_pos[bad_envs, 7:] = 0.04
            self._robot.set_joint_position_target(self.ctrl_target_joint_pos[bad_envs], env_ids=bad_envs)
            self.step_sim_no_action()
            ik_attempt += 1

        self.step_sim_no_action()

        self.actions = torch.zeros_like(self.actions)
        self.prev_actions = torch.zeros_like(self.actions)

        self.prev_joint_pos  = self.joint_pos[:, :7].clone()
        self.prev_fingertip_pos   = self.fingertip_midpoint_pos.clone()
        self.prev_fingertip_quat  = self.fingertip_midpoint_quat.clone()

        self.fixed_pos_action_frame[:] = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        pos_actions = ((self.fingertip_midpoint_pos - self.fixed_pos_action_frame) @ torch.diag(
            1 / torch.tensor(self.cfg.ctrl.pos_action_bounds, device=self.device)
        ))
        self.actions[:, :3] = self.prev_actions[:, :3] = pos_actions
        self.actions[:, 5]  = self.prev_actions[:, 5]  = 0.0                   # no yaw yet

        self.ee_angvel_fd.zero_()
        self.ee_linvel_fd.zero_()
        self._set_gains(self.default_gains)

        physics_sim_view.set_gravity(carb.Float3(*self.cfg.sim.gravity))
