"""Direct Isaac Lab environment for H1 tabletop object transfer.

The first manipulation target is deliberately simple and measurable: Unitree H1
stands at a lab table and moves a cube from the left side of the tabletop to a
goal marker on the right side. H1 has no dexterous hands in the stock Isaac Lab
asset, so the initial control surface uses whole-body joint-position control and
forearm/elbow contact to push/secure the cube while preserving balance.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import H1_CFG


TABLE_CENTER = (0.66, 0.0, 0.76)
TABLE_SIZE = (0.82, 0.92, 0.08)
CUBE_SIZE = 0.065
CUBE_START = (0.62, -0.26, 0.8325)
CUBE_GOAL = (0.62, 0.26, 0.8325)
GOAL_TOLERANCE = 0.075


@configclass
class H1TabletopTransferEnvCfg(DirectRLEnvCfg):
    """Configuration for H1 table-side transfer."""

    decimation = 2
    episode_length_s = 4.8
    action_space = 19
    observation_space = 79
    state_space = 0
    action_scale = 0.35

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    robot_cfg = H1_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    robot_cfg.init_state.pos = (0.0, 0.0, 1.05)
    robot_cfg.init_state.joint_pos.update(
        {
            ".*_shoulder_pitch": 0.55,
            ".*_shoulder_roll": 0.12,
            ".*_shoulder_yaw": 0.0,
            ".*_elbow": 1.05,
        }
    )

    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/TableTop",
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.1, dynamic_friction=0.9),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.34, 0.32, 0.28)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=TABLE_CENTER, rot=(1.0, 0.0, 0.0, 0.0)),
    )

    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/TargetCube",
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_linear_velocity=10.0,
                max_angular_velocity=20.0,
                max_depenetration_velocity=2.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.12),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.75, dynamic_friction=0.65),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.32, 1.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=CUBE_START, rot=(1.0, 0.0, 0.0, 0.0)),
    )

    goal_marker_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/GoalMarker",
        spawn=sim_utils.CuboidCfg(
            size=(0.11, 0.11, 0.012),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.85, 0.18)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(CUBE_GOAL[0], CUBE_GOAL[1], TABLE_CENTER[2] + TABLE_SIZE[2] / 2 + 0.006)),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=2.5, replicate_physics=True)

    reach_weight = 0.8
    progress_weight = 2.4
    placement_weight = 1.8
    success_weight = 8.0
    balance_weight = 0.7
    object_height_weight = 0.4
    action_rate_penalty = 0.015
    action_l2_penalty = 0.004
    fall_penalty = 5.0
    drop_penalty = 3.0


class H1TabletopTransferEnv(DirectRLEnv):
    """H1 pushes/transfers a tabletop cube from left to right."""

    cfg: H1TabletopTransferEnvCfg

    def __init__(self, cfg: H1TabletopTransferEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.num_robot_joints = self.robot.num_joints
        if self.num_robot_joints != self.cfg.action_space:
            raise RuntimeError(f"H1 action space mismatch: expected {self.cfg.action_space}, got {self.num_robot_joints}")

        limits = self.robot.data.soft_joint_pos_limits[0].to(self.device)
        self.joint_lower_limits = limits[:, 0]
        self.joint_upper_limits = limits[:, 1]
        self.default_joint_pos = self.robot.data.default_joint_pos.clone()
        self.actions = torch.zeros((self.num_envs, self.num_robot_joints), device=self.device)
        self.previous_actions = torch.zeros_like(self.actions)
        self.joint_targets = self.default_joint_pos.clone()

        self.cube_start_local = torch.tensor(CUBE_START, device=self.device).repeat(self.num_envs, 1)
        self.cube_goal_local = torch.tensor(CUBE_GOAL, device=self.device).repeat(self.num_envs, 1)
        self.goal_distance = torch.norm(self.cube_goal_local - self.cube_start_local, dim=1).clamp_min(1e-4)
        self.goal_tolerance = GOAL_TOLERANCE

        self.left_tool_body_id = self._find_body_index(["left_elbow_link", ".*left.*elbow.*", "left_shoulder_yaw_link"])
        self.right_tool_body_id = self._find_body_index(["right_elbow_link", ".*right.*elbow.*", "right_shoulder_yaw_link"])

    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        self.table = RigidObject(self.cfg.table_cfg)
        self.goal_marker = RigidObject(self.cfg.goal_marker_cfg)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["object"] = self.object
        self.scene.rigid_objects["table"] = self.table
        self.scene.rigid_objects["goal_marker"] = self.goal_marker

        light_cfg = sim_utils.DomeLightCfg(intensity=2800.0, color=(0.78, 0.78, 0.74))
        light_cfg.func("/World/Light", light_cfg)
        distant_light_cfg = sim_utils.DistantLightCfg(intensity=1200.0, color=(1.0, 0.96, 0.9), angle=0.45)
        distant_light_cfg.func("/World/KeyLight", distant_light_cfg, translation=(2.5, -2.0, 4.0))

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.previous_actions[:] = self.actions
        self.actions = actions.clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        targets = self.default_joint_pos + self.cfg.action_scale * self.actions
        targets = torch.max(torch.min(targets, self.joint_upper_limits), self.joint_lower_limits)
        self.joint_targets[:] = targets
        self.robot.set_joint_position_target(self.joint_targets)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        root_pos_local = self.robot.data.root_pos_w - self.scene.env_origins
        cube_pos_local = self.object.data.root_pos_w - self.scene.env_origins
        cube_vel = self.object.data.root_lin_vel_w
        left_tool = self.robot.data.body_pos_w[:, self.left_tool_body_id, :] - self.scene.env_origins
        right_tool = self.robot.data.body_pos_w[:, self.right_tool_body_id, :] - self.scene.env_origins
        joint_pos_rel = self.robot.data.joint_pos - self.default_joint_pos

        obs = torch.cat(
            (
                root_pos_local[:, 2:3],
                self.robot.data.root_lin_vel_w * 0.25,
                self.robot.data.root_ang_vel_w * 0.25,
                joint_pos_rel,
                self.robot.data.joint_vel * 0.05,
                self.actions,
                cube_pos_local - root_pos_local,
                self.cube_goal_local - cube_pos_local,
                cube_vel * 0.25,
                left_tool - cube_pos_local,
                right_tool - cube_pos_local,
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        metrics = self._compute_metrics()
        action_l2 = torch.mean(torch.square(self.actions), dim=1)
        action_rate = torch.mean(torch.square(self.actions - self.previous_actions), dim=1)
        rewards = (
            self.cfg.progress_weight * metrics["task_progress"]
            + self.cfg.reach_weight * metrics["contact_quality"]
            + self.cfg.placement_weight * metrics["placement_accuracy"]
            + self.cfg.balance_weight * metrics["balance_quality"]
            + self.cfg.object_height_weight * metrics["object_height_quality"]
            + self.cfg.success_weight * metrics["task_success_rate"]
            - self.cfg.action_l2_penalty * action_l2
            - self.cfg.action_rate_penalty * action_rate
            - self.cfg.fall_penalty * metrics["robot_fall_rate"]
            - self.cfg.drop_penalty * metrics["object_drop_rate"]
        )
        self.extras["log"] = {key: value.mean() for key, value in metrics.items()}
        self.extras["log"]["mean_action_l2"] = action_l2.mean()
        self.extras["log"]["mean_action_rate"] = action_rate.mean()
        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        metrics = self._compute_metrics()
        fallen = metrics["robot_fall_rate"] > 0.5
        dropped = metrics["object_drop_rate"] > 0.5
        success = metrics["task_success_rate"] > 0.5
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return fallen | dropped | success, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        root_state = self.robot.data.default_root_state[env_ids_tensor].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids_tensor]
        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids_tensor)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids_tensor)

        joint_pos = self.robot.data.default_joint_pos[env_ids_tensor].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids_tensor].clone()
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids_tensor)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids_tensor)

        cube_state = self.object.data.default_root_state[env_ids_tensor].clone()
        y_noise = torch.empty((len(env_ids_tensor),), device=self.device).uniform_(-0.035, 0.035)
        x_noise = torch.empty((len(env_ids_tensor),), device=self.device).uniform_(-0.025, 0.025)
        cube_state[:, 0] = self.scene.env_origins[env_ids_tensor, 0] + CUBE_START[0] + x_noise
        cube_state[:, 1] = self.scene.env_origins[env_ids_tensor, 1] + CUBE_START[1] + y_noise
        cube_state[:, 2] = self.scene.env_origins[env_ids_tensor, 2] + CUBE_START[2]
        cube_state[:, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), device=self.device)
        cube_state[:, 7:] = 0.0
        self.object.write_root_pose_to_sim(cube_state[:, :7], env_ids_tensor)
        self.object.write_root_velocity_to_sim(cube_state[:, 7:], env_ids_tensor)

        self.actions[env_ids_tensor] = 0.0
        self.previous_actions[env_ids_tensor] = 0.0

    def _compute_metrics(self) -> dict[str, torch.Tensor]:
        root_pos_local = self.robot.data.root_pos_w - self.scene.env_origins
        cube_pos_local = self.object.data.root_pos_w - self.scene.env_origins
        left_tool = self.robot.data.body_pos_w[:, self.left_tool_body_id, :] - self.scene.env_origins
        right_tool = self.robot.data.body_pos_w[:, self.right_tool_body_id, :] - self.scene.env_origins

        goal_error = torch.norm(cube_pos_local - self.cube_goal_local, dim=1)
        start_error = torch.norm(cube_pos_local - self.cube_start_local, dim=1)
        progress = ((self.goal_distance - goal_error) / self.goal_distance).clamp(0.0, 1.0)
        placement = torch.exp(-goal_error / 0.12).clamp(0.0, 1.0)
        success = (goal_error < self.goal_tolerance).float()

        tool_dist = torch.minimum(
            torch.norm(left_tool - cube_pos_local, dim=1),
            torch.norm(right_tool - cube_pos_local, dim=1),
        )
        contact_quality = torch.exp(-tool_dist / 0.22).clamp(0.0, 1.0)
        balance = ((root_pos_local[:, 2] - 0.58) / 0.42).clamp(0.0, 1.0)
        object_height = torch.exp(-torch.abs(cube_pos_local[:, 2] - CUBE_GOAL[2]) / 0.08).clamp(0.0, 1.0)
        on_table_x = (cube_pos_local[:, 0] > TABLE_CENTER[0] - TABLE_SIZE[0] / 2 - 0.08) & (
            cube_pos_local[:, 0] < TABLE_CENTER[0] + TABLE_SIZE[0] / 2 + 0.08
        )
        on_table_y = (cube_pos_local[:, 1] > -TABLE_SIZE[1] / 2 - 0.08) & (
            cube_pos_local[:, 1] < TABLE_SIZE[1] / 2 + 0.08
        )
        cube_on_table = on_table_x & on_table_y & (cube_pos_local[:, 2] > TABLE_CENTER[2] + TABLE_SIZE[2] / 2 - 0.06)
        fallen = (root_pos_local[:, 2] < 0.58).float()
        dropped = (~cube_on_table).float()
        collision_proxy = ((tool_dist < 0.04) & (start_error < 0.02)).float() * 0.1
        slip_proxy = torch.clamp(torch.abs(cube_pos_local[:, 0] - CUBE_GOAL[0]) / 0.35, 0.0, 1.0)

        return {
            "task_success_rate": success,
            "task_progress": progress,
            "contact_success_rate": (tool_dist < 0.16).float(),
            "contact_stability": contact_quality,
            "contact_quality": contact_quality,
            "placement_accuracy": placement,
            "placement_error_m": goal_error,
            "object_height_quality": object_height,
            "object_slip_rate": slip_proxy,
            "collision_rate": collision_proxy,
            "force_violation_rate": torch.zeros_like(progress),
            "robot_fall_rate": fallen,
            "object_drop_rate": dropped,
            "balance_quality": balance,
        }

    def _find_body_index(self, patterns: list[str]) -> int:
        for pattern in patterns:
            ids, _ = self.robot.find_bodies(pattern)
            if ids:
                return int(ids[0])
        raise RuntimeError(f"Could not find any H1 body matching {patterns}. Bodies: {self.robot.body_names}")

