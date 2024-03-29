import argparse
import importlib
import sys
import time
from typing import Callable

import torch
import pufferlib
import pufferlib.utils
import gymnasium as gym

from pokemonred_puffer.cleanrl_puffer import rollout
from pokemonred_puffer.train import (
    load_from_config,
    make_env_creator,
    update_args,
    init_wandb,
    train,
)

# These are used as the defaults in argparse
CUSTOM_REWARD_ENV = "environment.CustomRewardEnv"  # see environment.py
CUSTOM_POLICY = "policy.ConvolutionalPolicy"  # see policy.py


# Returns env_creator, agent_creator
def setup_agent(
    wrappers: list[str],
    reward_name: str,
    policy_name: str,
) -> Callable[[pufferlib.namespace, pufferlib.namespace], pufferlib.emulation.GymnasiumPufferEnv]:
    wrapper_classes = [
        (
            k,
            getattr(
                importlib.import_module(f"pokemonred_puffer.wrappers.{k.split('.')[0]}"),
                k.split(".")[1],
            ),
        )
        for wrapper_dicts in wrappers
        for k in wrapper_dicts.keys()
    ]
    reward_module, reward_class_name = reward_name.split(".")
    reward_class = getattr(
        importlib.import_module(reward_module), reward_class_name
    )
    # NOTE: This assumes reward_module has RewardWrapper(RedGymEnv) class
    env_creator = make_env_creator(wrapper_classes, reward_class)

    policy_module_name, policy_class_name = policy_name.split(".")
    policy_module = importlib.import_module(policy_module_name)
    policy_class = getattr(policy_module, policy_class_name)

    def agent_creator(env: gym.Env, args: pufferlib.namespace):
        policy = policy_class(env, **args.policies[policy_name]["policy"])
        if "recurrent" in args.policies[policy_name]:
            recurrent_args = args.policies[policy_name]["recurrent"]
            recurrent_class_name = recurrent_args["name"]
            del recurrent_args["name"]
            policy = getattr(policy_module, recurrent_class_name)(env, policy, **recurrent_args)
            policy = pufferlib.frameworks.cleanrl.RecurrentPolicy(policy)
        else:
            policy = pufferlib.frameworks.cleanrl.Policy(policy)

        if args.train.device == "cuda":
            torch.set_float32_matmul_precision(args.train.float32_matmul_precision)
            policy = policy.to(args.train.device, non_blocking=True)
            if args.train.compile:
                policy.get_value = torch.compile(policy.get_value, mode=args.train.compile_mode)
                policy.get_action_and_value = torch.compile(
                    policy.get_action_and_value, mode=args.train.compile_mode
                )

        return policy

    return env_creator, agent_creator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse environment argument", add_help=False)
    parser.add_argument("--yaml", default="config.yaml", help="Configuration file to use")
    parser.add_argument(
        "-p",
        "--policy-name",
        default=CUSTOM_POLICY,
        help="Policy module to use in policies",
    )
    parser.add_argument(
        "-r",
        "--reward-name",
        default=CUSTOM_REWARD_ENV,
        help="Reward module to use in rewards",
    )
    parser.add_argument(
        "-w",
        "--wrappers-name",
        type=str,
        default="baseline",
        help="Wrappers to use _in order of instantiation_.",
    )
    parser.add_argument("--mode", type=str, default="train", choices=["train", "evaluate"])
    parser.add_argument(
        "--eval-model-path", type=str, default=None, help="Path to model to evaluate"
    )
    parser.add_argument("--exp-name", type=str, default=None, help="Resume from experiment")
    parser.add_argument("--rom-path", default="red.gb")
    parser.add_argument("--track", action="store_true", help="Track on WandB")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument(
        "--vectorization",
        type=str,
        default="multiprocessing",
        choices=["serial", "multiprocessing"],
        help="Vectorization method (serial, multiprocessing)",
    )

    clean_parser = argparse.ArgumentParser(parents=[parser])
    args = parser.parse_known_args()[0].__dict__
    config = load_from_config(
        args["yaml"], args["wrappers_name"], args["policy_name"], args["reward_name"], args["debug"]
    )

    # Generate argparse menu from config
    # This is also a reason for Spock/Argbind/OmegaConf/pydantic-cli
    for name, sub_config in config.items():
        args[name] = {}
        for key, value in sub_config.items():
            data_key = f"{name}.{key}"
            cli_key = f"--{data_key}".replace("_", "-")
            if isinstance(value, bool) and value is False:
                action = "store_false"
                parser.add_argument(cli_key, default=value, action="store_true")
                clean_parser.add_argument(cli_key, default=value, action="store_true")
            elif isinstance(value, bool) and value is True:
                data_key = f"{name}.no_{key}"
                cli_key = f"--{data_key}".replace("_", "-")
                parser.add_argument(cli_key, default=value, action="store_false")
                clean_parser.add_argument(cli_key, default=value, action="store_false")
            else:
                parser.add_argument(cli_key, default=value, type=type(value))
                clean_parser.add_argument(cli_key, default=value, metavar="", type=type(value))

            args[name][key] = getattr(parser.parse_known_args()[0], data_key)
        args[name] = pufferlib.namespace(**args[name])
    clean_parser.parse_args(sys.argv[1:])
    args = update_args(args)

    env_creator, agent_creator = setup_agent(
        args.wrappers[args.wrappers_name], args.reward_name, args.policy_name
    )

    if args.track:
        args.exp_name = init_wandb(args).id
    else:
        args.exp_name = f"poke_{time.strftime('%Y%m%d_%H%M%S')}"
    args.env.session_path = args.exp_name

    if args.mode == "train":
        train(args, env_creator, agent_creator)

    elif args.mode == "evaluate":
        # TODO: check if this works
        rollout(
            env_creator=env_creator,
            env_creator_kwargs={"env_config": args.env, "wrappers_config": args.wrappers},
            agent_creator=agent_creator,
            agent_kwargs={"args": args},
            model_path=args.eval_model_path,
            device=args.train.device,
        )
    else:
        raise ValueError("Mode must be one of train or evaluate")
