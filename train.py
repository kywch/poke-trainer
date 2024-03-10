import argparse
import importlib
import inspect
import sys
import time

import pufferlib
import pufferlib.utils
import torch
import wandb
import yaml

from lib.clean_pufferl import CleanPuffeRL, rollout
from lib.environment import RedGymEnv
from lib.stream_wrapper import StreamWrapper

DEBUG = False


def load_from_config(yaml_path, reward_module, policy_module, debug=False):
    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    default_keys = ["env", "train", "reward", "policy", "recurrent", "wandb",
                    "sweep_metadata", "sweep_metric", "sweep"]
    defaults = {key: config.get(key, {}) for key in default_keys}

    debug_config = config.get('debug', {}) if debug else {}
    reward_config = config["reward_zoo"][reward_module]
    policy_config = config["policy_zoo"][policy_module]

    combined_config = {}
    for key in default_keys:
        policy_subconfig = policy_config.get(key, {})
        reward_subconfig = reward_config.get(key, {})
        debug_subconfig = debug_config.get(key, {})

        # Order of precedence: debug > reward > policy > defaults
        combined_config[key] = {**defaults[key], **policy_subconfig, **reward_subconfig, **debug_subconfig}

    return pufferlib.namespace(**combined_config)

def get_init_args(fn):
    if fn is None:
        return {}

    sig = inspect.signature(fn)
    args = {}
    for name, param in sig.parameters.items():
        if name in ("self", "env", "policy"):
            continue
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            continue
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            continue
        else:
            args[name] = param.default if param.default is not inspect.Parameter.empty else None
    return args

# Returns env_creator, agent_creator
def setup_agent(reward_name, policy_name, stream=False, stream_name=None):
    reward_module = importlib.import_module(f"reward_zoo.{reward_name}")
    # NOTE: This assumes reward_module has RewardWrapper(RedGymEnv) class
    env_creator = make_env_creator(reward_module.RewardWrapper,
                                   stream, stream_name)

    policy_module = importlib.import_module(f"policy_zoo.{policy_name}")
    def agent_creator(env, args):
        policy = policy_module.Policy(env, **args.policy)
        if not args.no_recurrence and policy_module.Recurrent is not None:
            policy = policy_module.Recurrent(env, policy, **args.recurrent)
            policy = pufferlib.frameworks.cleanrl.RecurrentPolicy(policy)
        else:
            policy = pufferlib.frameworks.cleanrl.Policy(policy)

        if args.train.device == "cuda":
            mode = "reduce-overhead"
            torch.set_float32_matmul_precision('high')  # to remove compile warning
            policy = policy.to(args.train.device, non_blocking=True)
            policy.get_value = torch.compile(policy.get_value, mode=mode)
            policy.get_action_and_value = torch.compile(policy.get_action_and_value, mode=mode)

        return policy

    return env_creator, agent_creator

def make_env_creator(env_cls: RedGymEnv, stream, stream_name):
    def env_creator(env_config, reward_config):
        env = env_cls(env_config, reward_config)
        if stream is True and stream_name is not None:
            env = StreamWrapper(env, stream_metadata={"user": stream_name})
        return pufferlib.emulation.GymnasiumPufferEnv(
            env=env, postprocessor_cls=pufferlib.emulation.BasicPostprocessor)

    return env_creator

def update_args(args):
    args = pufferlib.namespace(**args)

    args.track = not args.no_track
    args.stream = not args.no_stream
    if args.stream_name is None:
        args.stream_name = args.wandb.entity
    args.env.gb_path = args.rom_path

    if args.vectorization == "serial" or args.debug:
        args.vectorization = pufferlib.vectorization.Serial
    elif args.vectorization == "multiprocessing":
        args.vectorization = pufferlib.vectorization.Multiprocessing

    return args

def init_wandb(args, resume=True):
    if args.no_track:
        return None
    assert args.wandb.project is not None, "Please set the wandb project in config.yaml"
    assert args.wandb.entity is not None, "Please set the wandb entity in config.yaml"
    wandb_kwargs = {
        "id": args.exp_name or wandb.util.generate_id(),
        "project": args.wandb.project,
        "entity": args.wandb.entity,
        "group": args.wandb.group,
        "config": {
            "cleanrl": args.train,
            "env": args.env,
            "reward_module": args.reward_name,
            "policy_module": args.policy_name,
            "reward": args.reward,
            "policy": args.policy,
            "recurrent": args.recurrent,
        },
        'name': args.exp_name,
        'monitor_gym': True,
        'save_code': True,
        'resume': resume,
    }
    return wandb.init(**wandb_kwargs)

def train(args, env_creator, agent_creator):
    with CleanPuffeRL(
        config=args.train,
        agent_creator=agent_creator,
        agent_kwargs={"args": args},
        env_creator=env_creator,
        env_creator_kwargs={"env_config": args.env,
                            "reward_config": args.reward},
        vectorization=args.vectorization,
        exp_name=args.exp_name,
        track=args.track,
    ) as trainer:
        while not trainer.done_training():
            trainer.evaluate()
            trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse environment argument", add_help=False)
    parser.add_argument("-y", "--yaml", default="config.yaml", help="Configuration file to use")
    parser.add_argument("-p", "--policy-name", default="thatguy", help="Policy module to use in policy_zoo")
    parser.add_argument("-r", "--reward-name", default="thatguy_bet", help="Reward module to use in reward_zoo")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "evaluate"])
    parser.add_argument("--eval-model-path", type=str, default=None, help="Path to model to evaluate")
    parser.add_argument("--exp-name", type=str, default=None, help="Resume from experiment")
    parser.add_argument("--rom-path", default="red.gb", help="Path to ROM file")
    parser.add_argument("--no-recurrence", action="store_true", help="Do not use recurrence")
    parser.add_argument("--no-stream", action="store_true", help="Do not use StreamWrapper")
    parser.add_argument("--stream-name", default=None, help="Name to use in StreamWrapper")
    parser.add_argument(
        "--vectorization", type=str, default="multiprocessing", choices=["serial", "multiprocessing"],
        help="Vectorization method (serial, multiprocessing)",
    )
    if DEBUG:
        parser.add_argument("--no-track", default=True, help="Do not track on WandB")
        parser.add_argument("--debug", default=True, help='Debug mode')

    else:
        parser.add_argument("--no-track", action="store_true", help="Do not track on WandB")
        parser.add_argument("--debug", action="store_true", help='Debug mode')

    clean_parser = argparse.ArgumentParser(parents=[parser])
    args = parser.parse_known_args()[0].__dict__
    config = load_from_config(args["yaml"], args["reward_name"], args["policy_name"], args["debug"])

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

    env_creator, agent_creator = setup_agent(args.reward_name, args.policy_name,
                                             args.stream, args.stream_name)

    if args.track:
        args.exp_name = init_wandb(args).id
    else:
        args.exp_name = f"poke_{time.strftime('%Y%m%d_%H%M%S')}"
    args.env.session_path = args.exp_name

    if args.mode == "train":
        train(args, env_creator, agent_creator)

    # elif args.mode == "sweep":
    #     sweep(args, policy_module, env_creator)

    elif args.mode == "evaluate":
        # TODO: check if this works
        rollout(
            env_creator=env_creator,
            env_creator_kwargs={"env_config": args.env,
                                "reward_config": args.reward},
            agent_creator=agent_creator,
            agent_kwargs={"args": args},
            model_path=args.eval_model_path,
            device=args.train.device,
        )
    else:
        raise ValueError("Mode must be one of train or evaluate")
