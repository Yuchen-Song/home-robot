# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from typing import TYPE_CHECKING

from habitat import make_dataset
from habitat.core.environments import get_env_class
from habitat.utils.gym_definitions import _get_env_name

from home_robot_sim.env.habitat_ovmm_env.habitat_ovmm_env import (
    HabitatOpenVocabManipEnv,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig


def create_ovmm_env_fn(config: "DictConfig") -> HabitatOpenVocabManipEnv:
    """
    Creates an environment for the OVMM task.

    Creates habitat environment from config and wraps it into HabitatOpenVocabManipEnv.

    :param config: configuration for the environment.
    :return: environment instance.
    """
    habitat_config = config.habitat
    dataset = make_dataset(habitat_config.dataset.type, config=habitat_config.dataset)
    env_class_name = _get_env_name(config)
    env_class = get_env_class(env_class_name)
    habitat_env = env_class(config=habitat_config, dataset=dataset)
    habitat_env.seed(habitat_config.seed)
    env = HabitatOpenVocabManipEnv(habitat_env, config, dataset=dataset)
    # for episode in dataset.episodes:
    #     if episode.episode_id in {
    #                                 # '193', '135', '117', '164', '179', '181', '114', '173', '194', '153'
    #                                 # '1071', '1076', '1056', '1079', '1082', '1083', '1034',
    #                                 # '1178', '1170', '1154', '1104', '1185', '1157', '1144'
    #                                 # '880', '816', '885', '899', '876', '849', '895',
    #                                 # '760', '752', '729',
    #                                 # '424', '423',
    #                                 '32'}:
    #         print(f'Scene: {episode.scene_id}')
    #         # print(f'Episode: {episode.episode_id}')
    #         print(f'goal_Cn({episode.episode_id}) = {episode.candidate_objects[0].position}')
    #         print(f'goal_Cw({episode.episode_id}) = '
    #               f'[{np.round(episode.candidate_objects[0].position[0]+21.8739236, 8)}, '
    #               f'{episode.candidate_objects[0].position[1]}, '
    #               f'{np.round(episode.candidate_objects[0].position[2]+5.4809293, 8)}]')
    return env
