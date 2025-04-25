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
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR # ISAACLAB_NUCLEUS_DIR removed as not used
from isaaclab.utils.math import sample_uniform

# Assuming the UrdfFileCfg is correctly imported or defined elsewhere
# If not, add: from isaaclab.sim.spawners.from_files import UrdfFileCfg
from isaaclab.sim.spawners.from_files import UrdfFileCfg # Make sure this import exists


@configclass
class DualFrankaPipetteEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 12.0 # Increased episode length
    decimation = 2
    action_space = 18 # 9 per robot
    observation_space = 58 # Updated observation space size (9+9+9+9 + 3 + 3 + 7 + 7 + 2)
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
    # scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=3.0, replicate_physics=True)

    # robot_left (Renamed from robot)
    robot_left = ArticulationCfg(
        prim_path="/World/envs/env_.*/RobotLeft", # Updated prim path
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
            pos=(0.7, -0.3, 0.0), # Adjusted initial position (closer and left)
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

    # robot_right (Added)
    robot_right = ArticulationCfg(
        prim_path="/World/envs/env_.*/RobotRight", # Updated prim path
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
            joint_pos={ # Slightly different initial pose for pressing arm? Maybe pointed downwards more. Let's keep same for now.
                "panda_joint1": 1.157,
                "panda_joint2": -1.066,
                "panda_joint3": -0.155,
                "panda_joint4": -2.239,
                "panda_joint5": -1.841,
                "panda_joint6": 1.003,
                "panda_joint7": 0.469,
                "panda_finger_joint.*": 0.005, # Keep fingers closed? Or open? Let's try closed.
            },
            pos=(0.7, 0.3, 0.0), # Adjusted initial position (closer and right)
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
                stiffness=2e3, # Lower stiffness/damping for pressing? Maybe not needed.
                damping=1e2,
            ),
        },
    )


    # cabinet (Removed)
    # cabinet = ArticulationCfg(...)

    # pipette (Keep as is, maybe adjust initial position)
    pipette = ArticulationCfg(
        prim_path="/World/envs/env_.*/Pipette",
        spawn=UrdfFileCfg(
            asset_path="~/IsaacLab/obj/pipe/pipe.urdf", # Make sure this path is correct or relative
            root_link_name="base_link",
            fix_base=False,
            force_usd_conversion=True,
            mass_props=sim_utils.MassPropertiesCfg(density=400.0),
            scale=(1.0,1.0,1.0),
            joint_drive=None,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                disable_gravity=False,
                kinematic_enabled=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.28), # Moved closer to robot origin
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={"slider_joint": 0.0},
        ),
        actuators={}, # No actuators needed for pipette itself
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

    # --- RL relevant parameters ---
    # Action scale (same for both arms)
    action_scale = 7.5
    # Observation scale
    dof_velocity_scale = 0.1

    # Termination condition
    success_height_thresh = 0.5 # meters (Lift pipette base_link to 50cm)

    # Reward scales (tune these based on training)
    # -- Arm 1 (Left Arm - Lifting)
    reach_reward_scale = 2.0
    lift_reward_scale = 10.0
    grasp_reward_scale = 3.0
    # -- Arm 2 (Right Arm - Pressing)
    press_reach_reward_scale = 2.0 # Reward reaching the press target
    press_align_reward_scale = 1.0 # Reward aligning the press approach
    press_joint_reward_scale = 5.0 # Reward pushing the slider joint
    # -- Common
    action_penalty_scale = 0.01


class DualFrankaPipetteEnv(DirectRLEnv):
    """Environment for two Franka robots to grasp, lift, and press a pipette."""

    cfg: DualFrankaPipetteEnvCfg

    def __init__(self, cfg: DualFrankaPipetteEnvCfg, render_mode: str | None = None, **kwargs):
        # Check config compatibility before super init
        if cfg.action_space != 18:
            raise ValueError(f"Expected action space of 18, got: {cfg.action_space}")
        # We will update obs space calculation later if needed, accept 58 for now
        # if cfg.observation_space != 58:
        #     raise ValueError(f"Expected observation space of 58, got: {cfg.observation_space}")

        # Overwrite config observation space size before calling super
        super().__init__(cfg, render_mode, **kwargs)

        # -- Define helper function (remains the same)
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

        # -- Robot Left DOF properties
        self.robot_left_dof_lower_limits = self._robot_left.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_left_dof_upper_limits = self._robot_left.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_left_dof_range = self.robot_left_dof_upper_limits - self.robot_left_dof_lower_limits
        self.robot_left_dof_range = torch.where(
             self.robot_left_dof_range < 1e-6, torch.ones_like(self.robot_left_dof_range), self.robot_left_dof_range
        )
        self.robot_left_dof_speed_scales = torch.ones_like(self.robot_left_dof_lower_limits)
        self.robot_left_dof_speed_scales[self._robot_left.find_joints("panda_finger_joint1")[0]] = 0.1
        self.robot_left_dof_speed_scales[self._robot_left.find_joints("panda_finger_joint2")[0]] = 0.1
        self.robot_left_dof_targets = torch.zeros((self.num_envs, self._robot_left.num_joints), device=self.device)

        # -- Robot Right DOF properties
        self.robot_right_dof_lower_limits = self._robot_right.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_right_dof_upper_limits = self._robot_right.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_right_dof_range = self.robot_right_dof_upper_limits - self.robot_right_dof_lower_limits
        self.robot_right_dof_range = torch.where(
             self.robot_right_dof_range < 1e-6, torch.ones_like(self.robot_right_dof_range), self.robot_right_dof_range
        )
        self.robot_right_dof_speed_scales = torch.ones_like(self.robot_right_dof_lower_limits)
        # No finger scaling needed for right arm if we aren't using fingers for pressing
        # self.robot_right_dof_speed_scales[self._robot_right.find_joints("panda_finger_joint1")[0]] = 0.1
        # self.robot_right_dof_speed_scales[self._robot_right.find_joints("panda_finger_joint2")[0]] = 0.1
        self.robot_right_dof_targets = torch.zeros((self.num_envs, self._robot_right.num_joints), device=self.device)


        # -- Calculate robot's local grasp frame offset (relative to hand link)
        # Use robot_left for calculation (assuming symmetry)
        stage = get_current_stage()
        hand_pose_left = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/RobotLeft/panda_link7")),
            self.device,
        )
        lfinger_pose_left = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/RobotLeft/panda_leftfinger")),
            self.device,
        )
        rfinger_pose_left = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/RobotLeft/panda_rightfinger")),
            self.device,
        )

        finger_pose_left = torch.zeros(7, device=self.device)
        finger_pose_left[0:3] = (lfinger_pose_left[0:3] + rfinger_pose_left[0:3]) / 2.0 # Midpoint between fingers
        finger_pose_left[3:7] = hand_pose_left[3:7] # Use hand orientation
        hand_pose_inv_rot_left, hand_pose_inv_pos_left = tf_inverse(hand_pose_left[3:7], hand_pose_left[0:3])

        # Combine transforms: Hand Frame -> World -> Finger Midpoint -> Local Grasp Frame
        robot_local_grasp_pose_rot_left, robot_local_grasp_pose_pos_left = tf_combine(
            hand_pose_inv_rot_left, hand_pose_inv_pos_left, finger_pose_left[3:7], finger_pose_left[0:3]
        )
        # Left arm grasp offset (for pipette base) - 5cm forward from finger midpoint seems reasonable for grasping
        grasp_offset_hand_frame_left = torch.tensor([0.0, 0.0, 0.05], device=self.device)
        robot_local_grasp_pose_pos_left += grasp_offset_hand_frame_left

        self.robot_left_local_grasp_pos = robot_local_grasp_pose_pos_left.repeat((self.num_envs, 1))
        self.robot_left_local_grasp_rot = robot_local_grasp_pose_rot_left.repeat((self.num_envs, 1))

        # Right arm grasp offset (for pressing pipette top) - use hand frame origin (panda_link7)
        # We can define the target on the pipette later. The 'grasp' point for the right arm is just its EEF origin.
        self.robot_right_local_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.robot_right_local_grasp_rot = torch.eye(4, device=self.device)[:3, :3].reshape(1, 4).repeat(self.num_envs, 1) # Identity rotation
        self.robot_right_local_grasp_rot[:, 0] = 1.0 # Ensure valid quaternion (w=1)


        # -- Find indices
        # Robot Left
        self.hand_body_idx_left = self._robot_left.find_bodies("panda_link7")[0][0]
        self.left_finger_body_idx_left = self._robot_left.find_bodies("panda_leftfinger")[0][0]
        self.right_finger_body_idx_left = self._robot_left.find_bodies("panda_rightfinger")[0][0]
        self.left_finger_joint_idx_left = self._robot_left.find_joints("panda_finger_joint1")[0][0]
        self.right_finger_joint_idx_left = self._robot_left.find_joints("panda_finger_joint2")[0][0]

        # Robot Right
        self.hand_body_idx_right = self._robot_right.find_bodies("panda_link7")[0][0]
        # Don't need finger indices for right arm if not used for reward/obs

        # Pipette
        self.pipette_base_body_idx = self._pipette.find_bodies("base_link")[0][0]
        self.pipette_movable_body_idx = self._pipette.find_bodies("Movable_Link")[0][0]
        self.pipette_slider_joint_idx = self._pipette.find_joints("slider_joint")[0][0]

        # -- Define target points in local frames of pipette links
        # Grasp target on pipette base_link (e.g., origin)
        self.pipette_base_local_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.pipette_base_local_grasp_rot = torch.eye(4, device=self.device)[:3, :3].reshape(1, 4).repeat(self.num_envs, 1)
        self.pipette_base_local_grasp_rot[:, 0] = 1.0

        # Press target on pipette Movable_Link (e.g., origin)
        self.pipette_movable_local_press_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.pipette_movable_local_press_rot = torch.eye(4, device=self.device)[:3, :3].reshape(1, 4).repeat(self.num_envs, 1)
        self.pipette_movable_local_press_rot[:, 0] = 1.0


        # -- Initialize state buffers
        # Robot Left grasp pose (world frame)
        self.robot_left_grasp_rot_w = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_left_grasp_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        # Robot Right grasp pose / EEF pose (world frame)
        self.robot_right_eef_rot_w = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_right_eef_pos_w = torch.zeros((self.num_envs, 3), device=self.device)

        # Pipette base link pose (world frame)
        self.pipette_base_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.pipette_base_rot_w = torch.zeros((self.num_envs, 4), device=self.device)
        # Pipette movable link pose (world frame)
        self.pipette_movable_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.pipette_movable_rot_w = torch.zeros((self.num_envs, 4), device=self.device)
        # Pipette grasp target pose (world frame) - calculated from base link pose
        self.pipette_grasp_target_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.pipette_grasp_target_rot_w = torch.zeros((self.num_envs, 4), device=self.device)
        # Pipette press target pose (world frame) - calculated from movable link pose
        self.pipette_press_target_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.pipette_press_target_rot_w = torch.zeros((self.num_envs, 4), device=self.device)


        # Pipette height (base link z coordinate)
        self.pipette_height = torch.zeros(self.num_envs, device=self.device)
        # Pipette slider joint state
        self.pipette_slider_joint_pos = torch.zeros(self.num_envs, device=self.device)
        self.pipette_slider_joint_vel = torch.zeros(self.num_envs, device=self.device)


        # Store initial pipette height for reward calculation
        self.pipette_initial_height = torch.tensor(self.cfg.pipette.init_state.pos[2], device=self.device).repeat(self.num_envs)

        self._in_reset_loop = False


    def _setup_scene(self):
        self._robot_left = Articulation(self.cfg.robot_left)
        self._robot_right = Articulation(self.cfg.robot_right) # Add right robot
        # self._cabinet = Articulation(self.cfg.cabinet) # Remove cabinet
        self._pipette = Articulation(self.cfg.pipette)

        # Add articulations to the scene
        self.scene.articulations["robot_left"] = self._robot_left
        self.scene.articulations["robot_right"] = self._robot_right # Add right robot
        # self.scene.articulations["cabinet"] = self._cabinet # Remove cabinet
        self.scene.articulations["pipette"] = self._pipette

        # Setup terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone and replicate environments
        self.scene.clone_environments(copy_from_source=False) # Must be False for non-cloned assets

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # pre-physics step calls

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)
        # Split actions for left and right robots
        actions_left = self.actions[:, :self._robot_left.num_actions]
        actions_right = self.actions[:, self._robot_left.num_actions:]

        # Update targets for robot left
        targets_left = self.robot_left_dof_targets + self.robot_left_dof_speed_scales * self.dt * actions_left * self.cfg.action_scale
        self.robot_left_dof_targets[:] = torch.clamp(targets_left, self.robot_left_dof_lower_limits, self.robot_left_dof_upper_limits)

        # Update targets for robot right
        targets_right = self.robot_right_dof_targets + self.robot_right_dof_speed_scales * self.dt * actions_right * self.cfg.action_scale
        self.robot_right_dof_targets[:] = torch.clamp(targets_right, self.robot_right_dof_lower_limits, self.robot_right_dof_upper_limits)


    def _apply_action(self):
        self._robot_left.set_joint_position_target(self.robot_left_dof_targets)
        self._robot_right.set_joint_position_target(self.robot_right_dof_targets) # Apply action to right robot

    # post-physics step calls

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:

        self._compute_intermediate_values() # Updates pipette_height

        # Terminate if pipette is lifted successfully above threshold by left arm
        terminated = self.pipette_height > self.cfg.success_height_thresh
        # Could add termination based on pressing success later if needed

        # Truncate if episode length is exceeded
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards for the current step."""
        # Values computed in _get_dones calling _compute_intermediate_values are up-to-date
        # self._compute_intermediate_values() # Already called in _get_dones

        # Retrieve necessary data
        # Left arm data
        robot_left_grasp_pos = self.robot_left_grasp_pos_w
        left_finger_pos = self._robot_left.data.joint_pos[:, self.left_finger_joint_idx_left]
        right_finger_pos = self._robot_left.data.joint_pos[:, self.right_finger_joint_idx_left]

        # Right arm data
        robot_right_eef_pos = self.robot_right_eef_pos_w
        robot_right_eef_rot = self.robot_right_eef_rot_w

        # Pipette data
        pipette_grasp_target_pos = self.pipette_grasp_target_pos_w
        pipette_press_target_pos = self.pipette_press_target_pos_w
        pipette_movable_rot = self.pipette_movable_rot_w
        pipette_height = self.pipette_height
        pipette_slider_joint_pos = self.pipette_slider_joint_pos

        # Split actions for penalties
        actions_left = self.actions[:, :self._robot_left.num_actions]
        actions_right = self.actions[:, self._robot_left.num_actions:]

        return self._compute_rewards(
            actions_left=actions_left,
            actions_right=actions_right,
            # Left Arm State (for lifting reward)
            robot_left_grasp_pos=robot_left_grasp_pos,
            pipette_grasp_target_pos=pipette_grasp_target_pos,
            pipette_height=pipette_height,
            initial_pipette_height=self.pipette_initial_height,
            left_finger_pos=left_finger_pos,
            right_finger_pos=right_finger_pos,
            # Right Arm State (for pressing reward)
            robot_right_eef_pos=robot_right_eef_pos,
            robot_right_eef_rot=robot_right_eef_rot,
            pipette_press_target_pos=pipette_press_target_pos,
            pipette_movable_rot=pipette_movable_rot,
            pipette_slider_joint_pos=pipette_slider_joint_pos,
            # Reward Scales
            reach_reward_scale=self.cfg.reach_reward_scale,
            lift_reward_scale=self.cfg.lift_reward_scale,
            grasp_reward_scale=self.cfg.grasp_reward_scale,
            press_reach_reward_scale=self.cfg.press_reach_reward_scale,
            press_align_reward_scale=self.cfg.press_align_reward_scale,
            press_joint_reward_scale=self.cfg.press_joint_reward_scale,
            action_penalty_scale=self.cfg.action_penalty_scale,
        )

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        super()._reset_idx(env_ids) # Resets episode length buf etc.

        # -- Reset robot left state
        joint_pos_left = self._robot_left.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125, 0.125, (len(env_ids), self._robot_left.num_joints), self.device,
        )
        joint_pos_left = torch.clamp(joint_pos_left, self.robot_left_dof_lower_limits, self.robot_left_dof_upper_limits)
        joint_vel_left = torch.zeros_like(joint_pos_left)
        self._robot_left.set_joint_position_target(joint_pos_left, env_ids=env_ids)
        self._robot_left.write_joint_state_to_sim(joint_pos_left, joint_vel_left, env_ids=env_ids)
        self.robot_left_dof_targets[env_ids] = joint_pos_left # Sync targets

        # -- Reset robot right state
        joint_pos_right = self._robot_right.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125, 0.125, (len(env_ids), self._robot_right.num_joints), self.device,
        )
        joint_pos_right = torch.clamp(joint_pos_right, self.robot_right_dof_lower_limits, self.robot_right_dof_upper_limits)
        joint_vel_right = torch.zeros_like(joint_pos_right)
        self._robot_right.set_joint_position_target(joint_pos_right, env_ids=env_ids)
        self._robot_right.write_joint_state_to_sim(joint_pos_right, joint_vel_right, env_ids=env_ids)
        self.robot_right_dof_targets[env_ids] = joint_pos_right # Sync targets


        # -- Reset pipette state
        # Get the default root state (position relative to env origin 0)
        default_pipette_state = self._pipette.data.default_root_state[env_ids].clone() # Clone to modify
        # Get the correct environment origins for the envs being reset
        env_origins = self.scene.env_origins[env_ids] # Shape: (num_resets, 3)
        # Add the origins to the position part of the default state
        default_pipette_state[:, 0:3] += env_origins
        pose = default_pipette_state[:, :7]                       # (x y z qw qx qy qz) - Check convention if different
        vel = torch.zeros((len(env_ids), 6), device=self.device)  # (vx vy vz wx wy wz)

        # Reset root state
        self._pipette.write_root_pose_to_sim(pose, env_ids=env_ids)
        self._pipette.write_root_velocity_to_sim(vel, env_ids=env_ids)

        # Reset joint state (slider joint)
        joint_pos_pipette = self._pipette.data.default_joint_pos[env_ids] # Should be [0.0]
        joint_vel_pipette = torch.zeros_like(joint_pos_pipette)
        self._pipette.write_joint_state_to_sim(joint_pos_pipette, joint_vel_pipette, env_ids=env_ids)


        # Need to compute intermediate values for observations after reset
        if not self._in_reset_loop:
            self._compute_intermediate_values(env_ids)


    def _get_observations(self) -> dict:
        # Compute observations for the RL policy.

        # === Robot States ===
        # Robot Left
        dof_pos_scaled_left = (
            2.0 * (self._robot_left.data.joint_pos - self.robot_left_dof_lower_limits) / self.robot_left_dof_range - 1.0
        )
        dof_vel_scaled_left = self._robot_left.data.joint_vel * self.cfg.dof_velocity_scale

        # Robot Right
        dof_pos_scaled_right = (
            2.0 * (self._robot_right.data.joint_pos - self.robot_right_dof_lower_limits) / self.robot_right_dof_range - 1.0
        )
        dof_vel_scaled_right = self._robot_right.data.joint_vel * self.cfg.dof_velocity_scale

        # === Relative Poses ===
        # Vector from robot left grasp point to pipette base grasp target
        vec_left_to_grasp_target = self.pipette_grasp_target_pos_w - self.robot_left_grasp_pos_w
        # Vector from robot right EEF point to pipette press target
        vec_right_to_press_target = self.pipette_press_target_pos_w - self.robot_right_eef_pos_w

        # === Pipette State ===
        pipette_base_pose_w = torch.cat((self.pipette_base_pos_w, self.pipette_base_rot_w), dim=-1) # 7D
        pipette_movable_pose_w = torch.cat((self.pipette_movable_pos_w, self.pipette_movable_rot_w), dim=-1) # 7D
        pipette_slider_joint_state = torch.cat((
            self.pipette_slider_joint_pos.unsqueeze(-1),
            self.pipette_slider_joint_vel.unsqueeze(-1)), dim=-1
        ) # 2D

        # === Concatenate observations ===
        # Order:
        # Robot Left: dof_pos_scaled(9), dof_vel_scaled(9)
        # Robot Right: dof_pos_scaled(9), dof_vel_scaled(9)
        # Relative Vecs: left_to_grasp(3), right_to_press(3)
        # Pipette: base_pose(7), movable_pose(7), slider_joint(2)
        # Total: 9+9 + 9+9 + 3+3 + 7+7 + 2 = 18 + 18 + 6 + 14 + 2 = 58. Matches config.
        obs = torch.cat(
            (
                dof_pos_scaled_left,        # 9
                dof_vel_scaled_left,        # 9
                dof_pos_scaled_right,       # 9
                dof_vel_scaled_right,       # 9
                vec_left_to_grasp_target,   # 3
                vec_right_to_press_target,  # 3
                pipette_base_pose_w,        # 7 (pos + quat)
                pipette_movable_pose_w,     # 7 (pos + quat)
                pipette_slider_joint_state, # 2 (pos + vel)
            ),
            dim=-1,
        )

        # Clamp observations for stability
        obs_clamped = torch.clamp(obs, -5.0, 5.0)

        return {"policy": obs_clamped}


    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        """Compute useful intermediate values derived from simulation state."""
        if env_ids is None:
            env_ids = self._robot_left._ALL_INDICES # Use left robot indices (should be same for all)

        # === Robot Left ===
        hand_pos_left_w = self._robot_left.data.body_pos_w[env_ids, self.hand_body_idx_left]
        hand_rot_left_w = self._robot_left.data.body_quat_w[env_ids, self.hand_body_idx_left]
        local_grasp_rot_left = self.robot_left_local_grasp_rot[env_ids]
        local_grasp_pos_left = self.robot_left_local_grasp_pos[env_ids]
        # Compute world grasp pose for left robot
        self.robot_left_grasp_rot_w[env_ids], self.robot_left_grasp_pos_w[env_ids] = tf_combine(
            hand_rot_left_w, hand_pos_left_w, local_grasp_rot_left, local_grasp_pos_left
        )

        # === Robot Right ===
        hand_pos_right_w = self._robot_right.data.body_pos_w[env_ids, self.hand_body_idx_right]
        hand_rot_right_w = self._robot_right.data.body_quat_w[env_ids, self.hand_body_idx_right]
        # For right robot, the 'grasp' point is just the EEF origin (panda_link7)
        self.robot_right_eef_rot_w[env_ids] = hand_rot_right_w
        self.robot_right_eef_pos_w[env_ids] = hand_pos_right_w


        # === Pipette ===
        # Base link pose (world frame)
        self.pipette_base_pos_w[env_ids] = self._pipette.data.body_pos_w[env_ids, self.pipette_base_body_idx]
        self.pipette_base_rot_w[env_ids] = self._pipette.data.body_quat_w[env_ids, self.pipette_base_body_idx]
        # Movable link pose (world frame)
        self.pipette_movable_pos_w[env_ids] = self._pipette.data.body_pos_w[env_ids, self.pipette_movable_body_idx]
        self.pipette_movable_rot_w[env_ids] = self._pipette.data.body_quat_w[env_ids, self.pipette_movable_body_idx]

        # Calculate grasp target point on base link (world frame)
        local_grasp_target_rot = self.pipette_base_local_grasp_rot[env_ids]
        local_grasp_target_pos = self.pipette_base_local_grasp_pos[env_ids]
        self.pipette_grasp_target_rot_w[env_ids], self.pipette_grasp_target_pos_w[env_ids] = tf_combine(
             self.pipette_base_rot_w[env_ids], self.pipette_base_pos_w[env_ids], local_grasp_target_rot, local_grasp_target_pos
        )

        # Calculate press target point on movable link (world frame)
        local_press_target_rot = self.pipette_movable_local_press_rot[env_ids]
        local_press_target_pos = self.pipette_movable_local_press_pos[env_ids]
        self.pipette_press_target_rot_w[env_ids], self.pipette_press_target_pos_w[env_ids] = tf_combine(
             self.pipette_movable_rot_w[env_ids], self.pipette_movable_pos_w[env_ids], local_press_target_rot, local_press_target_pos
        )

        # Pipette base Z position (height)
        self.pipette_height[env_ids] = self.pipette_base_pos_w[env_ids, 2]
        # Pipette slider joint state
        self.pipette_slider_joint_pos[env_ids] = self._pipette.data.joint_pos[env_ids, self.pipette_slider_joint_idx]
        self.pipette_slider_joint_vel[env_ids] = self._pipette.data.joint_vel[env_ids, self.pipette_slider_joint_idx]


        # === NaN / Reset Check ===
        # Check NaNs in relevant positions
        bad_mask = (torch.isnan(self.pipette_base_pos_w[env_ids]).any(dim=1) |
                    torch.isnan(self.robot_left_grasp_pos_w[env_ids]).any(dim=1) |
                    torch.isnan(self.robot_right_eef_pos_w[env_ids]).any(dim=1))
        if bad_mask.any():
            bad_ids = env_ids[bad_mask]
            print(f"WARN: NaN detected in positions for envs: {bad_ids.tolist()}. Resetting.")

            self.reset_buf[bad_ids] = True
            self.rew_buf[bad_ids]   = 0.0

            # Avoid recursion during reset
            if not self._in_reset_loop:
                self._in_reset_loop = True
                self._reset_idx(bad_ids)
                self._in_reset_loop = False


    def _compute_rewards(
        self,
        actions_left: torch.Tensor,
        actions_right: torch.Tensor,
        # Left Arm State
        robot_left_grasp_pos: torch.Tensor,
        pipette_grasp_target_pos: torch.Tensor,
        pipette_height: torch.Tensor,
        initial_pipette_height: torch.Tensor,
        left_finger_pos: torch.Tensor,
        right_finger_pos: torch.Tensor,
        # Right Arm State
        robot_right_eef_pos: torch.Tensor,
        robot_right_eef_rot: torch.Tensor,
        pipette_press_target_pos: torch.Tensor,
        pipette_movable_rot: torch.Tensor,
        pipette_slider_joint_pos: torch.Tensor,
        # Reward Scales
        reach_reward_scale: float,
        lift_reward_scale: float,
        grasp_reward_scale: float,
        press_reach_reward_scale: float,
        press_align_reward_scale: float,
        press_joint_reward_scale: float,
        action_penalty_scale: float,
    ) -> torch.Tensor:
        """Compute rewards for the dual-arm pipette task."""

        # === Arm 1 (Left) Rewards: Grasping and Lifting ===

        # 1. Reach reward: Encourage left gripper center to approach pipette grasp target
        reach_dist = torch.norm(robot_left_grasp_pos - pipette_grasp_target_pos, p=2, dim=-1)
        reach_reward = torch.exp(-5.0 * reach_dist) # Exponential decay

        # 2. Lift reward: Encourage lifting the pipette *above* its initial height
        height_diff = pipette_height - initial_pipette_height
        lift_reward = torch.tanh(10.0 * torch.clamp(height_diff, min=0.0))
        # Only give lift reward if gripper is close (likely grasping)
        lift_reward = torch.where(reach_dist < 0.05, lift_reward, torch.zeros_like(lift_reward))

        # 3. Grasp reward: Encourage left fingers to close *around* the pipette grasp target
        finger_closed_reward = 0.5 * (torch.exp(-10.0 * left_finger_pos) + torch.exp(-10.0 * right_finger_pos))
        # Only apply grasp reward when close to the object
        grasp_reward = torch.where(reach_dist < 0.08, finger_closed_reward, torch.zeros_like(finger_closed_reward))


        # === Arm 2 (Right) Rewards: Pressing ===
        # Gate for pressing rewards: only active when pipette is lifted
        lift_gate = (pipette_height > (initial_pipette_height + 0.05)).float() # Use 0.0 for false, 1.0 for true

        # 4. Press Reach reward: Encourage right EEF to approach pipette press target
        press_reach_dist = torch.norm(robot_right_eef_pos - pipette_press_target_pos, p=2, dim=-1)
        press_reach_reward = torch.exp(-10.0 * press_reach_dist) * lift_gate # Gated

        # 5. Press Align reward: Encourage right EEF Z-axis to align with Movable_Link -Y axis
        # Approach axis (EEF local Z) in world frame
        eef_approach_axis_local = torch.tensor([[0.0, 0.0, 1.0]], device=self.device).repeat(self.num_envs, 1)
        eef_approach_axis_world = tf_vector(robot_right_eef_rot, eef_approach_axis_local)
        # Target press axis (Movable_Link local -Y) in world frame
        press_axis_local = torch.tensor([[0.0, -1.0, 0.0]], device=self.device).repeat(self.num_envs, 1)
        press_axis_world = tf_vector(pipette_movable_rot, press_axis_local)
        # Dot product for alignment (-1 to 1)
        align_dot = torch.sum(eef_approach_axis_world * press_axis_world, dim=-1)
        # Scale alignment dot product to [0, 1] reward
        press_align_reward = (align_dot + 1.0) * 0.5 * lift_gate # Gated

        # 6. Press Joint reward: Encourage pushing the slider joint towards its upper limit (0.015)
        # Scale joint position to [0, 1] based on its limits [0, 0.015]
        slider_joint_max = 0.015
        press_joint_progress = torch.clamp(pipette_slider_joint_pos / slider_joint_max, 0.0, 1.0)
        # Gate press reward by proximity and alignment
        press_gate = (press_reach_dist < 0.03) & (align_dot > 0.9) # Stricter gate: closer and better aligned
        press_joint_reward = press_joint_progress * press_gate.float() * lift_gate # Gated


        # === Common Penalties ===
        # 7. Action penalty: Discourage excessive movements for both arms
        action_penalty_left = torch.sum(actions_left**2, dim=-1)
        action_penalty_right = torch.sum(actions_right**2, dim=-1)
        action_penalty = action_penalty_left + action_penalty_right


        # === Final reward composition ===
        rewards = (
            # Arm 1 Rewards
            reach_reward_scale * reach_reward
            + lift_reward_scale * lift_reward
            + grasp_reward_scale * grasp_reward
            # Arm 2 Rewards
            + press_reach_reward_scale * press_reach_reward
            + press_align_reward_scale * press_align_reward
            + press_joint_reward_scale * press_joint_reward
            # Penalties
            - action_penalty_scale * action_penalty
        )

        # Optional: Success bonus for lifting
        rewards = torch.where(pipette_height > self.cfg.success_height_thresh, rewards + 5.0, rewards) # Smaller bonus than single arm
        # Optional: Success bonus for pressing
        rewards = torch.where(press_joint_progress > 0.9, rewards + 5.0, rewards) # Bonus for full press


        # === Logging for TensorBoard/debugging ===
        self.extras["log"] = {
            # Arm 1
            "reward/reach": (reach_reward_scale * reach_reward).mean(),
            "reward/lift": (lift_reward_scale * lift_reward).mean(),
            "reward/grasp": (grasp_reward_scale * grasp_reward).mean(),
            # Arm 2
            "reward/press_reach": (press_reach_reward_scale * press_reach_reward).mean(),
            "reward/press_align": (press_align_reward_scale * press_align_reward).mean(),
            "reward/press_joint": (press_joint_reward_scale * press_joint_reward).mean(),
            # Common
            "reward/action_penalty": (-action_penalty_scale * action_penalty).mean(),
            "reward/total": rewards.mean(),
            # State - Arm 1
            "state/reach_distance": reach_dist.mean(),
            "state/left_finger_joint": left_finger_pos.mean(),
            "state/right_finger_joint": right_finger_pos.mean(),
            # State - Arm 2
            "state/press_reach_distance": press_reach_dist.mean(),
            "state/press_align_dot": align_dot.mean(),
            # State - Pipette
            "state/pipette_height": pipette_height.mean(),
            "state/pipette_slider_joint_pos": pipette_slider_joint_pos.mean(),
            "state/press_joint_progress": press_joint_progress.mean(),
            # Gates
            "gate/lift_active": lift_gate.mean(),
            "gate/press_active": press_gate.float().mean(),
        }


        # Safety check for NaN rewards
        if torch.isnan(rewards).any():
            print("WARN: NaN detected in rewards!")
            nan_mask = torch.isnan(rewards)
            print(f"NaN indices: {torch.where(nan_mask)[0].tolist()}")
            # Log components that might be causing NaNs
            if torch.isnan(reach_reward).any(): print("WARN: NaN in reach_reward")
            if torch.isnan(lift_reward).any(): print("WARN: NaN in lift_reward")
            if torch.isnan(grasp_reward).any(): print("WARN: NaN in grasp_reward")
            if torch.isnan(press_reach_reward).any(): print("WARN: NaN in press_reach_reward")
            if torch.isnan(press_align_reward).any(): print("WARN: NaN in press_align_reward")
            if torch.isnan(press_joint_reward).any(): print("WARN: NaN in press_joint_reward")
            if torch.isnan(action_penalty).any(): print("WARN: NaN in action_penalty")

            rewards = torch.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)
            self.extras["log"]["reward/nan_reward_detected"] = 1.0 # Log NaN detection
        else:
            self.extras["log"]["reward/nan_reward_detected"] = 0.0


        return rewards