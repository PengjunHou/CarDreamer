import datetime
import warnings

import embodied
import ruamel.yaml as yaml

import car_dreamer
import dreamerv3

warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")

import sys

sys.path.append("/home/peh324/carla_simulator/PythonAPI/")
sys.path.append("/home/peh324/carla_simulator/PythonAPI/carla")

import carla
from agents.navigation.basic_agent import BasicAgent

def wrap_env(env, config):
    args = config.wrapper
    env = embodied.wrappers.InfoWrapper(env)
    for name, space in env.act_space.items():
        if name == "reset":
            continue
        elif space.discrete:
            env = embodied.wrappers.OneHotAction(env, name)
        elif args.discretize:
            env = embodied.wrappers.DiscretizeAction(env, name, args.discretize)
        else:
            env = embodied.wrappers.NormalizeAction(env, name)
    env = embodied.wrappers.ExpandScalars(env)
    if args.length:
        env = embodied.wrappers.TimeLimit(env, args.length, args.reset)
    if args.checks:
        env = embodied.wrappers.CheckSpaces(env)
    for name, space in env.act_space.items():
        if not space.discrete:
            env = embodied.wrappers.ClipAction(env, name)
    return env


def main(argv=None):
    model_configs = yaml.YAML(typ="safe").load((embodied.Path(__file__).parent / "dreamerv3.yaml").read())
    config = embodied.Config({"dreamerv3": model_configs["defaults"]})
    config = config.update({"dreamerv3": model_configs["small"]})

    parsed, other = embodied.Flags(task=["carla_group_right_turn_auto"]).parse_known(argv)
    for name in parsed.task:
        print("Using task: ", name)
        env, env_config = car_dreamer.create_task(name, argv)
        config = config.update(env_config)
    config = embodied.Flags(config).parse(other)
    # print(config)

    logdir = embodied.Path(config.dreamerv3.logdir)
    step = embodied.Counter()
    dreamerv3_config = config.dreamerv3

    # --- Build logger outputs ---
    log_outputs = [
        embodied.logger.TerminalOutput(),
        embodied.logger.JSONLOutput(logdir, "metrics.jsonl"),
        # embodied.logger.TensorBoardOutput(logdir),
    ]

    wandb_cfg = getattr(dreamerv3_config, "wandb", None)
    if wandb_cfg is not None and getattr(wandb_cfg, "enable", False):
        run_name = getattr(wandb_cfg, "run_name", "") or logdir.name
        log_outputs.append(
            embodied.logger.WandBOutput(
                run_name=run_name,
                config=config,
                entity=getattr(wandb_cfg, "entity", ""),
                project=getattr(wandb_cfg, "project", "CarDreamer"),
                resume=getattr(wandb_cfg, "resume", False),
            )
        )
        print(f"[WandB] Logging to project '{getattr(wandb_cfg, 'project', 'CarDreamer')}', run '{run_name}'")

    logger = embodied.Logger(step, log_outputs)

    from embodied.envs import from_gym

    env = from_gym.FromGym(env)
    env = wrap_env(env, dreamerv3_config)
    env = embodied.BatchEnv([env], parallel=False)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    config_filename = f"config_{timestamp}.yaml"
    config.save(str(logdir / config_filename))
    print(f"[Train] Config saved to {logdir / config_filename}")

    # agent = dreamerv3.Agent(env.obs_space, env.act_space, step, dreamerv3_config)
    # agent = dreamerv3.CoopSACAgent(env.obs_space, env.act_space, step, dreamerv3_config)
    agent = dreamerv3.TestAgent(env.obs_space, env.act_space, step, dreamerv3_config)
    replay = embodied.replay.Uniform(dreamerv3_config.batch_length, dreamerv3_config.replay_size, logdir / "replay")
    args = embodied.Config(
        **dreamerv3_config.run,
        logdir=dreamerv3_config.logdir,
        batch_steps=dreamerv3_config.batch_size * dreamerv3_config.batch_length,
        actor_dist_disc=dreamerv3_config.actor_dist_disc,
    )
    embodied.run.train(agent, env, replay, logger, args)
    
    print(f"testing agent...")


if __name__ == "__main__":
    main()
