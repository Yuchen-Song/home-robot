from typing import Any, Dict, Optional

import numpy as np
import cv2
from omegaconf import DictConfig

import rospy

import home_robot
from home_robot.core.interfaces import Action, DiscreteNavigationAction, Observations
from home_robot.utils.geometry import (
    sophus2xyt, xyt_base_to_global
)
from home_robot_hw.env.stretch_abstract_env import StretchEnv


class StretchImageNavEnv(StretchEnv):
    """Create a detic-based object nav environment"""

    def __init__(self, config: Optional[DictConfig] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if config:
            self.forward_step = config.habitat.simulator.forward_step_size  # in meters
            self.rotate_step = np.radians(config.habitat.simulator.turn_angle)
            self.image_goal = self._load_image_goal(config.stretch_goal_image_path)
        else:
            self.forward_step = 0.25
            self.rotate_step = np.radians(30)
            self.image_goal = None
        self.reset()

    def _load_image_goal(self, goal_img_path) -> np.ndarray:
        goal_image = cv2.imread(goal_img_path)
        # opencv loads as BGR, but we use RGB.
        goal_image = cv2.cvtColor(cv2.COLOR_BGR2RGB)
        assert goal_image.shape[0] == 512
        assert goal_image.shape[1] == 512
        return goal_image

    def reset(self):
        self._episode_start_pose = self.get_base_pose()
        if self.visualizer is not None:
            self.visualizer.reset()

    def apply_action(self, action: Action):
        continuous_action = np.zeros(3)
        if action == DiscreteNavigationAction.MOVE_FORWARD:
            print("FORWARD")
            continuous_action[0] = self.forward_step
        elif action == DiscreteNavigationAction.TURN_RIGHT:
            print("TURN RIGHT")
            continuous_action[2] = -self.rotate_step
        elif action == DiscreteNavigationAction.TURN_LEFT:
            print("TURN LEFT")
            continuous_action[2] = self.rotate_step
        else:
            # Do nothing if "stop"
            # continuous_action = None
            # if not self.in_manipulation_mode():
            #     self.switch_to_manipulation_mode()
            pass

        if continuous_action is not None:
            if not self.in_navigation_mode():
                self.switch_to_navigation_mode()
            self.navigate_to(continuous_action, relative=True)
        print("-------")
        print(action)
        print(continuous_action)
        rospy.sleep(5.0)

    def get_observation(self) -> Observations:
        """Get Detic and rgb/xyz/theta from this"""
        rgb, depth = self.get_images(compute_xyz=False, rotate_images=True)
        current_pose = self.get_base_pose()

        # use sophus to get the relative translation
        relative_pose = self._episode_start_pose.inverse() * current_pose
        euler_angles = relative_pose.so3().log()
        theta = euler_angles[-1]
        pos, vel, frc = self.get_joint_state()

        # Create the observation
        return home_robot.core.interfaces.Observations(
            rgb=rgb.copy(),
            depth=depth.copy(),
            gps=relative_pose.translation()[:2],
            compass=np.array([theta]),
            task_observations={"instance_imagegoal": self.image_goal},
            joint_positions=pos,
        )

    @property
    def episode_over(self) -> bool:
        pass

    def get_episode_metrics(self) -> Dict:
        pass

    def rotate(self, theta):
        """just rotate and keep trying"""
        init_pose = sophus2xyt(self.get_base_pose())
        xyt = [0, 0, theta]
        goal_pose = xyt_base_to_global(xyt, init_pose)
        rate = rospy.Rate(5)
        err = float("Inf"), float("Inf")
        pos_tol, ori_tol = 0.1, 0.1
        while not rospy.is_shutdown():
            curr_pose = sophus2xyt(self.get_base_pose())
            print("init =", init_pose)
            print("curr =", curr_pose)
            print("goal =", goal_pose)

            print("error =", err)
            if err[0] < pos_tol and err[1] < ori_tol:
                break
            rate.sleep()


if __name__ == "__main__":
    # Create the robot
    print("--------------")
    print("Start example - hardware using ROS")
    rospy.init_node("hello_stretch_ros_test")
    print("Create ROS interface")
    rob = StretchImageNavEnv(init_cameras=True)
    rob.switch_to_navigation_mode()

    # Debug the observation space
    import matplotlib.pyplot as plt

    while not rospy.is_shutdown():

        while not rospy.is_shutdown():
            cmd = None
            try:
                cmd = input("Enter a number 0-3:")
                cmd = DiscreteNavigationAction(int(cmd))
            except ValueError:
                cmd = None
            if cmd is not None:
                break
        rob.apply_action(cmd)

        obs = rob.get_observation()
        rgb, depth = obs.rgb, obs.depth

        # Add a visualiztion for debugging
        depth[depth > 5] = 0
        plt.subplot(121)
        plt.imshow(rgb)
        plt.subplot(122)
        plt.imshow(depth)

        print()
        print("----------------")
        print("values:")
        print("RGB =", np.unique(rgb))
        print("Depth =", np.unique(depth))
        print("Compass =", obs.compass)
        print("Gps =", obs.gps)
        plt.show()

