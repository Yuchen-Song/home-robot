# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import random
import sys

import numpy as np
import torch
from loguru import logger

try:
    sys.path.append(os.path.expanduser(os.environ["ACCEL_CORTEX"]))
    import grpc

    import src.rpc
    import src.rpc.cortex_rpc_pb2
    import src.rpc.proto_converter as proto_converter
    from src.rpc.cortex_rpc_pb2_grpc import AgentgRPCStub
    from src.utils.types.observations import Object, Observations

except Exception as e:
    ## Temporary hack until we make accel-cortex pip installable
    print(
        "Make sure path to accel-cortex base folder is set in the ACCEL_CORTEX environment variable."
    )
    print("If you do not know what that means, this code is not for you!")
    raise (e)


def parse_pick_and_place_plan(world_representation, plan: str):
    """Simple parser to pull out high level actions from a plan of the form:

        goto(obj1);pickup(obj1);goto(obj2);placeon(obj1,obj2)

    Args:
        plan(str): contains a plan
    """
    pick_instance_id, place_instance_id = None, None
    if plan == "explore":
        return None, None

    for current_high_level_action in plan.split("), "):
        current_high_level_action = current_high_level_action + ")"
        # addtional format checking of whether the current action is in the robot's skill set
        if not any(
            action in current_high_level_action
            for action in ["goto", "pickup", "placeon", "explore"]
        ):
            return None, None

        if "pickup" in current_high_level_action:
            img_id = (
                current_high_level_action.split("(")[1]
                .split(")")[0]
                .split("_")[1]
                .replace('"', "")
            )
            if img_id.isnumeric():
                pick_instance_id = int(
                    world_representation.object_images[int(img_id)].crop_id
                )
            else:
                pick_instance_id = None
        if "placeon" in current_high_level_action:
            img_id = (
                current_high_level_action.split("(")[1]
                .split(")")[0]
                .split(", ")[1]
                .split("_")[1]
            ).replace('"', "")
            if img_id.isnumeric():
                place_instance_id = int(
                    world_representation.object_images[int(img_id)].crop_id
                )
            else:
                place_instance_id = None
    return pick_instance_id, place_instance_id


def get_obj_centric_world_representation(
    instance_memory, max_context_length: int, sample_strategy: str
):
    """Get version that LLM can handle - convert images into torch if not already"""

    if sample_strategy == "all":
        # Send all the crop images so the agent can implement divide and conquer
        pass
    elif sample_strategy == "random_subsample":
        pass
    elif sample_strategy == "first":
        # Send the first images below the context length
        pass
    else:
        pass

    obs = Observations(object_images=[])
    for global_id, instance in enumerate(instance_memory):
        if global_id >= max_context_length:
            logger.warning(
                "\nWarning: this version of minigpt4 can only handle limited size of crops -- ignoring instance..."
            )
        else:
            instance_crops = instance.instance_views
            crop = random.sample(instance_crops, 1)[0].cropped_image
            if isinstance(crop, np.ndarray):
                crop = torch.from_numpy(crop)
            obs.object_images.append(
                Object(
                    crop_id=global_id,
                    image=(crop.contiguous() * 256).to(torch.uint8),
                )
            )

    # TODO: this code does not work as the global_ids have to be sequential and consecutive
    # # TODO: the model currenly can only handle 20 crops
    # if len(obs.object_images) > max_context_length:
    #     logger.warning(
    #         "\nWarning: this version of minigpt4 can only handle limited size of crops -- sampling a subset of crops from the instance memory..."
    #     )
    #     obs.object_images = random.sample(obs.object_images, max_context_length)

    return obs


def get_vlm_rpc_stub(vlm_server_addr: str, vlm_server_port: int):
    """Connect to a remote VLM server via RPC"""
    channel = grpc.insecure_channel(f"{vlm_server_addr}:{vlm_server_port}")
    stub = AgentgRPCStub(channel)
    return stub


def get_output_from_world_representation(stub, world_representation, goal: str):
    return stub.stream_act_on_observations(
        proto_converter.wrap_obs(
            obs=world_representation,
        )
    )
