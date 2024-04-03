# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import json
import os
import time
from collections import defaultdict
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np
import pandas as pd
from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer
from omegaconf import DictConfig
from tqdm import tqdm
from utils.env_utils import create_ovmm_env_fn
from utils.metrics_utils import get_stats_from_episode_metrics

if TYPE_CHECKING:
    from habitat.core.dataset import BaseEpisode
    from habitat.core.vector_env import VectorEnv

    from home_robot.agent.ovmm_agent.ovmm_agent import OpenVocabManipAgent
    from home_robot.core.abstract_agent import Agent


class EvaluationType(Enum):
    LOCAL = "local"
    LOCAL_VECTORIZED = "local_vectorized"
    REMOTE = "remote"


class OVMMEvaluator(PPOTrainer):
    """Class for creating vectorized environments, evaluating OpenVocabManipAgent on an episode dataset and returning metrics"""

    def __init__(self, eval_config: DictConfig) -> None:
        self.metrics_save_freq = eval_config.EVAL_VECTORIZED.metrics_save_freq
        self.results_dir = os.path.join(
            eval_config.DUMP_LOCATION, "results", eval_config.EXP_NAME
        )
        self.videos_dir = eval_config.habitat_baselines.video_dir
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.videos_dir, exist_ok=True)

        super().__init__(eval_config)

    def local_evaluate_vectorized(self, agent, num_episodes_per_env=10):
        self._init_envs(
            config=self.config, is_eval=True, make_env_fn=create_ovmm_env_fn
        )
        self._evaluate_vectorized(
            agent,
            self.envs,
            num_episodes_per_env=num_episodes_per_env,
        )

    def _summarize_metrics(self, episode_metrics: Dict) -> Dict:
        """Gets stats from episode metrics"""
        # convert to a dataframe
        episode_metrics_df = pd.DataFrame.from_dict(episode_metrics, orient="index")
        episode_metrics_df["start_idx"] = 0
        stats = get_stats_from_episode_metrics(episode_metrics_df)
        return stats

    def _print_summary(self, summary: dict):
        """Prints the summary of metrics"""
        print("=" * 50)
        print("Averaged metrics")
        print("=" * 50)
        for k, v in summary.items():
            print(f"{k}: {v}")
        print("=" * 50)

    def _check_set_planner_vis_dir(
        self, agent: "Agent", current_episode: "BaseEpisode"
    ):
        """
        Sets vis_dir for storing planner's debug visualisations if the agent has a planner.
        """
        if hasattr(agent, "planner"):
            agent.planner.set_vis_dir(
                current_episode.scene_id.split("/")[-1].split(".")[0],
                current_episode.episode_id,
            )

    def _evaluate_vectorized(
        self,
        agent: "OpenVocabManipAgent",
        envs: "VectorEnv",
        num_episodes_per_env=None,
    ):
        # The stopping condition is either specified through
        # num_episodes_per_env (stop after each environment
        # finishes a certain number of episodes)
        print(f"Running eval on {envs.number_of_episodes} episodes")

        if num_episodes_per_env is None:
            num_episodes_per_env = envs.number_of_episodes
        else:
            num_episodes_per_env = [num_episodes_per_env] * envs.num_envs

        episode_metrics = {}

        def stop():
            return all(
                [
                    episode_idxs[i] >= num_episodes_per_env[i]
                    for i in range(envs.num_envs)
                ]
            )

        start_time = time.time()
        episode_idxs = [0] * envs.num_envs
        obs = envs.call(["reset"] * envs.num_envs)

        agent.reset_vectorized()
        self._check_set_planner_vis_dir(agent, self.envs.current_episodes()[0])
        while not stop():
            current_episodes_info = self.envs.current_episodes()
            # TODO: Currently agent can work with only 1 env, Parallelize act across envs
            actions, infos, _ = zip(*[agent.act(ob) for ob in obs])

            outputs = envs.call(
                ["apply_action"] * envs.num_envs,
                [{"action": a, "info": i} for a, i in zip(actions, infos)],
            )

            obs, dones, hab_infos = [list(x) for x in zip(*outputs)]
            for e, (done, info, hab_info) in enumerate(zip(dones, infos, hab_infos)):
                episode_key = (
                    f"{current_episodes_info[e].scene_id.split('/')[-1].split('.')[0]}_"
                    f"{current_episodes_info[e].episode_id}"
                )
                if episode_key not in episode_metrics:
                    episode_metrics[episode_key] = {}
                # Record metrics after each skill finishes. This is useful for debugging.
                if "skill_done" in info and info["skill_done"] != "":
                    metrics = self._extract_scalars_from_info(hab_info)
                    metrics_at_skill_end = {
                        f"{info['skill_done']}." + k: v for k, v in metrics.items()
                    }
                    episode_metrics[episode_key] = {
                        **metrics_at_skill_end,
                        **episode_metrics[episode_key],
                    }
                    if "goal_name" in episode_metrics[episode_key]:
                        episode_metrics[episode_key]["goal_name"] = info["goal_name"]
                if done:  # environment times out
                    metrics = self._extract_scalars_from_info(hab_info)
                    if episode_idxs[e] < num_episodes_per_env[e]:
                        metrics_at_episode_end = {
                            f"END." + k: v for k, v in metrics.items()
                        }
                        episode_metrics[episode_key] = {
                            **metrics_at_episode_end,
                            **episode_metrics[episode_key],
                        }
                        if "goal_name" in episode_metrics[episode_key]:
                            episode_metrics[episode_key]["goal_name"] = info[
                                "goal_name"
                            ]
                        episode_idxs[e] += 1
                        print(
                            f"Episode indexes {episode_idxs[e]} / {num_episodes_per_env[e]} "
                            f"after {round(time.time() - start_time, 2)} seconds"
                        )
                    if len(episode_metrics) % self.metrics_save_freq == 0:
                        aggregated_metrics = self._aggregate_metrics(episode_metrics)
                        self._write_results(episode_metrics, aggregated_metrics)
                    if not stop():
                        obs[e] = envs.call_at(e, "reset")
                        agent.reset_vectorized_for_env(e)
                        self._check_set_planner_vis_dir(
                            envs, envs.current_episodes()[e]
                        )

        envs.close()

        aggregated_metrics = self._aggregate_metrics(episode_metrics)
        self._write_results(episode_metrics, aggregated_metrics)

        average_metrics = self._summarize_metrics(episode_metrics)
        self._print_summary(average_metrics)

        return average_metrics

    def _aggregate_metrics(self, episode_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Aggregates metrics tracked by environment."""
        aggregated_metrics = defaultdict(list)
        metrics = set(
            [
                k
                for metrics_per_episode in episode_metrics.values()
                for k in metrics_per_episode
                if k != "goal_name"
            ]
        )
        for v in episode_metrics.values():
            for k in metrics:
                if k in v:
                    aggregated_metrics[f"{k}/total"].append(v[k])

        aggregated_metrics = dict(
            sorted(
                {
                    k2: v2
                    for k1, v1 in aggregated_metrics.items()
                    for k2, v2 in {
                        f"{k1}/mean": np.mean(v1),
                        f"{k1}/min": np.min(v1),
                        f"{k1}/max": np.max(v1),
                    }.items()
                }.items()
            )
        )

        return aggregated_metrics

    def _write_results(
        self, episode_metrics: Dict[str, Dict], aggregated_metrics: Dict[str, float]
    ) -> None:
        """Writes metrics tracked by environment to a file."""
        with open(f"{self.results_dir}/aggregated_results.json", "w") as f:
            json.dump(aggregated_metrics, f, indent=4)
        with open(f"{self.results_dir}/episode_results.json", "w") as f:
            json.dump(episode_metrics, f, indent=4)

    def local_evaluate(
        self, agent: "Agent", num_episodes: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluates the agent in the local environment.

        :param agent: agent to be evaluated in environment.
        :param num_episodes: count of number of episodes for which the evaluation should be run.
        :return: dict containing metrics tracked by environment.
        """

        env_num_episodes = self._env.number_of_episodes
        if num_episodes is None:
            num_episodes = env_num_episodes
        else:
            assert num_episodes <= env_num_episodes, (
                "num_episodes({}) is larger than number of episodes "
                "in environment ({})".format(num_episodes, env_num_episodes)
            )

        assert num_episodes > 0, "num_episodes should be greater than 0"

        episode_metrics: Dict = {}

        count_episodes: int = 0

        pbar = tqdm(total=num_episodes)
        while count_episodes < num_episodes:
            observations, done = self._env.reset(), False
            current_episode = self._env.get_current_episode()
            # if current_episode.episode_id not in ['1071', '1178', '880', '760', '424', '32']:
            #     count_episodes += 1
            #     pbar.update(1)
            #     continue
            # if current_episode.episode_id in ['193', '131', '145', '135', '139',
            #                                   '116', '109', '160', '177', '141',
            #                                   '168', '117', '164', '151', '157',
            #                                   '113', '179', '197', '181', '156',
            #                                   '1036', '1061', '1068', '1094',
            #                                   '1015', '1046', '1085', '1096',
            #                                   '1035', '1026', '1065', '1037',
            #                                   '1042', '1032', '1070', '1071',
            #                                   '1012', '1008', '1058', '1062',
            #                                   '1076', '1006', '1081', '1087',
            #                                   '1028', '1025', '1122', '1152',
            #                                   '1115', '1105', '1134', '1109',
            #                                   '1118', '1131', '1146', '1100',
            #                                   '1182', '1178', '1170', '1124',
            #                                   '1111', '1168', '1126', '1172',
            #                                   '1141', '1174', '1149', '1114',
            #                                   '1143', '880', '891', '844',
            #                                   '824', '823', '816', '833', '861',
            #                                   '893', '864', '885', '847', '819',
            #                                   '868', '886', '899', '876', '849',
            #                                   '892', '857', '841', '895', '866',
            #                                   '882', '183', '178', '172', '180',
            #                                   '119', '114', '138', '196', '103',
            #                                   '188', '163', '170', '106', '190',
            #                                   '129', '173', '186', '184', '102',
            #                                   '111', '136', '1004', '1023',
            #                                   '1056', '1003', '1053', '1074',
            #                                   '1043', '1079', '1082', '1098',
            #                                   '1078', '1051', '1066', '1069',
            #                                   '1014', '1005', '1029', '1083',
            #                                   '1022', '1034', '1052', '1040',
            #                                   '1019', '1030', '1095', '1090',
            #                                   '1154', '1165', '1102', '1164',
            #                                   '1198', '1138', '1159', '1195',
            #                                   '1167', '1132', '1104', '1099',
            #                                   '1142', '1162', '1116', '1189',
            #                                   '1173', '1130', '1177', '1179',
            #                                   '1192', '871', '843', '898',
            #                                   '828', '806', '870', '805', '839',
            #                                   '829', '896', '855', '890', '894',
            #                                   '802', '852', '851', '853', '826',
            #                                   '787', '783', '705', '788', '763',
            #                                   '781', '784', '760', '780', '714',
            #                                   '769', '730', '764', '753', '752',
            #                                   '729', '732', '785', '786', '774',
            #                                   '791', '747', '766', '704', '480',
            #                                   '424', '440', '417', '408', '416',
            #                                   '482', '443', '404', '423', '420',
            #                                   '418', '460', '465', '447', '436',
            #                                   '496', '466', '73', '2', '69', '9',
            #                                   '83', '99', '45', '66', '30', '34',
            #                                   '32', '22', '53', '48', '84', '54',
            #                                   '13', '75', '59', '92', '49', '7',
            #                                   '36', '565', '532', '595', '511',
            #                                   '161', '189', '194', '112', '152',
            #                                   '132', '115', '166', '146', '125',
            #                                   '100', '195', '108', '123', '153',
            #                                   '187', '155', '142', '105', '134',
            #                                   '159', '1010', '1033', '1050', '1009',
            #                                   '1049', '1045', '1024', '1084', '1018',
            #                                   '1073', '1041', '1067', '1059', '1063',
            #                                   '1060', '1002', '1038', '1086', '1190',
            #                                   '1197', '1139', '1161', '1103', '1185',
            #                                   '1187', '1136', '1188', '1194', '1157',
            #                                   '1127', '1101', '1180', '1106', '1144',
            #                                   '1120', '1169', '1183', '1148', '1193',
            #                                   '1181', '1119', '1137', '1123', '1171',
            #                                   '1112', '1117', '887', '842', '840',
            #                                   '872', '810', '832', '879', '813', '831',
            #                                   '874', '835', '811', '877', '889', '838',
            #                                   '862', '804', '821']:
            #     count_episodes += 1
            #     pbar.update(1)
            #     continue

            agent.reset()
            self._check_set_planner_vis_dir(agent, current_episode)

            current_episode_key = (
                f"{current_episode.scene_id.split('/')[-1].split('.')[0]}_"
                f"{current_episode.episode_id}"
            )
            current_episode_metrics = {}

            while not done:
                action, info, obs = agent.act(observations)
                observations, done, hab_info = self._env.apply_action(action, info, obs)

                if "skill_done" in info and info["skill_done"] != "":
                    metrics = self._extract_scalars_from_info(hab_info)
                    metrics_at_skill_end = {
                        f"{info['skill_done']}." + k: v for k, v in metrics.items()
                    }
                    current_episode_metrics = {
                        **metrics_at_skill_end,
                        **current_episode_metrics,
                    }
                    if "goal_name" in info:
                        current_episode_metrics["goal_name"] = info["goal_name"]

            metrics = self._extract_scalars_from_info(hab_info)
            metrics_at_episode_end = {"END." + k: v for k, v in metrics.items()}
            current_episode_metrics = {
                **metrics_at_episode_end,
                **current_episode_metrics,
            }
            if "goal_name" in info:
                current_episode_metrics["goal_name"] = info["goal_name"]

            episode_metrics[current_episode_key] = current_episode_metrics
            if len(episode_metrics) % self.metrics_save_freq == 0:
                aggregated_metrics = self._aggregate_metrics(episode_metrics)
                self._write_results(episode_metrics, aggregated_metrics)

            count_episodes += 1
            pbar.update(1)

        self._env.close()

        aggregated_metrics = self._aggregate_metrics(episode_metrics)
        self._write_results(episode_metrics, aggregated_metrics)

        average_metrics = self._summarize_metrics(episode_metrics)
        self._print_summary(average_metrics)

        return average_metrics

    def remote_evaluate(
        self, agent: "Agent", num_episodes: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluates the agent in the remote environment.

        :param agent: agent to be evaluated in environment.
        :param num_episodes: count of number of episodes for which the evaluation should be run.
        :return: dict containing metrics tracked by environment.
        """

        # The modules imported below are specific to challenge remote evaluation.
        # These modules are not part of the home-robot repository.
        import pickle
        import time

        import evalai_environment_habitat  # noqa: F401
        import evaluation_pb2
        import evaluation_pb2_grpc
        import grpc

        # Wait for the remote environment to be up and running
        time.sleep(60)

        def grpc_dumps(entity):
            return pickle.dumps(entity)

        def grpc_loads(entity):
            return pickle.loads(entity)

        env_address_port = os.environ.get("EVALENV_ADDPORT", "localhost:8085")
        channel = grpc.insecure_channel(
            target=env_address_port,
            compression=grpc.Compression.Gzip,
            options=[
                (
                    "grpc.max_receive_message_length",
                    -1,
                )  # Unlimited message length that the channel can receive
            ],
        )
        stub = evaluation_pb2_grpc.EnvironmentStub(channel)

        stub.init_env(evaluation_pb2.Package())

        env_num_episodes = grpc_loads(
            stub.number_of_episodes(evaluation_pb2.Package()).SerializedEntity
        )
        if num_episodes is None:
            num_episodes = env_num_episodes
        else:
            assert num_episodes <= env_num_episodes, (
                "num_episodes({}) is larger than number of episodes "
                "in environment ({})".format(num_episodes, env_num_episodes)
            )

        assert num_episodes > 0, "num_episodes should be greater than 0"

        episode_metrics: Dict = {}

        count_episodes: int = 0

        pbar = tqdm(total=num_episodes)
        while count_episodes < num_episodes:
            observations, done = (
                grpc_loads(stub.reset(evaluation_pb2.Package()).SerializedEntity),
                False,
            )
            current_episode = grpc_loads(
                stub.get_current_episode(evaluation_pb2.Package()).SerializedEntity
            )
            agent.reset()
            self._check_set_planner_vis_dir(agent, current_episode)

            current_episode_key = (
                f"{current_episode.scene_id.split('/')[-1].split('.')[0]}_"
                f"{current_episode.episode_id}"
            )
            current_episode_metrics = {}

            while not done:
                action, info, _ = agent.act(observations)
                observations, done, hab_info = grpc_loads(
                    stub.apply_action(
                        evaluation_pb2.Package(
                            SerializedEntity=grpc_dumps((action, info))
                        )
                    ).SerializedEntity
                )

                # record metrics if the current skill finishes
                if "skill_done" in info and info["skill_done"] != "":
                    metrics = self._extract_scalars_from_info(hab_info)
                    metrics_at_skill_end = {
                        f"{info['skill_done']}." + k: v for k, v in metrics.items()
                    }
                    current_episode_metrics = {
                        **metrics_at_skill_end,
                        **current_episode_metrics,
                    }
                    if "goal_name" in info:
                        current_episode_metrics["goal_name"] = info["goal_name"]

            metrics = self._extract_scalars_from_info(hab_info)
            metrics_at_episode_end = {"END." + k: v for k, v in metrics.items()}
            current_episode_metrics = {
                **metrics_at_episode_end,
                **current_episode_metrics,
            }
            if "goal_name" in info:
                current_episode_metrics["goal_name"] = info["goal_name"]

            episode_metrics[current_episode_key] = current_episode_metrics
            if len(episode_metrics) % self.metrics_save_freq == 0:
                aggregated_metrics = self._aggregate_metrics(episode_metrics)
                self._write_results(episode_metrics, aggregated_metrics)

            count_episodes += 1
            pbar.update(1)

        stub.close(evaluation_pb2.Package())
        stub.evalai_update_submission(evaluation_pb2.Package())

        aggregated_metrics = self._aggregate_metrics(episode_metrics)
        self._write_results(episode_metrics, aggregated_metrics)

        average_metrics = self._summarize_metrics(episode_metrics)
        self._print_summary(average_metrics)

        return average_metrics

    def evaluate(
        self,
        agent: "Agent",
        num_episodes: Optional[int] = None,
        evaluation_type: str = "local",
    ) -> Dict[str, float]:
        r"""..

        :param agent: agent to be evaluated in environment.
        :param num_episodes: count of number of episodes for which the
            evaluation should be run.
        :return: dict containing metrics tracked by environment.
        """
        if evaluation_type == EvaluationType.LOCAL.value:
            self._env = create_ovmm_env_fn(self.config)
            return self.local_evaluate(agent, num_episodes)
        elif evaluation_type == EvaluationType.LOCAL_VECTORIZED.value:
            self._env = create_ovmm_env_fn(self.config)
            return self.local_evaluate_vectorized(agent, num_episodes)
        elif evaluation_type == EvaluationType.REMOTE.value:
            self._env = None
            return self.remote_evaluate(agent, num_episodes)
        else:
            raise ValueError(
                "Invalid evaluation type. Please choose from 'local', 'local_vectorized', 'remote'"
            )
