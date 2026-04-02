import datetime
import logging
import warnings

import embodied
import ruamel.yaml as yaml

import car_dreamer
import dreamerv3
from runtime_logging import (
    DEFAULT_RUNTIME_LOGGING_CONFIG,
    configure_runtime_logging,
    get_runtime_logger,
    log_key_event,
)
from train_config_merge import diff_explicit_config_overrides

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
    config = config.update({"runtime_logging": DEFAULT_RUNTIME_LOGGING_CONFIG})
    bootstrap_dreamerv3_base = dict(config.dreamerv3)
    bootstrap_runtime_logging_base = dict(config.runtime_logging)

    bootstrap_cfg, other = embodied.Flags(
        config,
        task=["carla_group_right_turn_auto"],
    ).parse_known(argv)
    cli_overrides = {
        "dreamerv3": diff_explicit_config_overrides(
            bootstrap_cfg.dreamerv3,
            bootstrap_dreamerv3_base,
        ),
        "runtime_logging": diff_explicit_config_overrides(
            bootstrap_cfg.runtime_logging,
            bootstrap_runtime_logging_base,
        ),
    }

    bootstrap_settings = configure_runtime_logging(
        bootstrap_cfg.runtime_logging,
        logdir=bootstrap_cfg.dreamerv3.logdir,
    )
    train_logger = get_runtime_logger("dreamerv3.train")
    log_key_event(
        train_logger,
        logging.INFO,
        "Bootstrapped runtime logging with level=%s console=%s file=%s logdir=%s",
        bootstrap_settings["level"],
        bootstrap_settings["console"],
        bootstrap_settings["file"],
        bootstrap_cfg.dreamerv3.logdir,
    )

    task_names = tuple(bootstrap_cfg.task)
    for name in task_names:
        log_key_event(train_logger, logging.INFO, "Using task '%s'", name)
        env, env_config = car_dreamer.create_task(name, argv)
        config = config.update(env_config)
    if cli_overrides["dreamerv3"]:
        config = config.update({"dreamerv3": cli_overrides["dreamerv3"]})
    if cli_overrides["runtime_logging"]:
        config = config.update({"runtime_logging": cli_overrides["runtime_logging"]})
    config = embodied.Flags(config).parse(other)

    logdir = embodied.Path(config.dreamerv3.logdir)
    logdir.mkdirs()
    step = embodied.Counter()
    dreamerv3_config = config.dreamerv3
    runtime_settings = configure_runtime_logging(config.runtime_logging, logdir=logdir)
    log_key_event(
        train_logger,
        logging.INFO,
        "Resolved runtime logging level=%s console_level=%s file_level=%s step_debug_interval=%s",
        runtime_settings["level"],
        runtime_settings["console_level"],
        runtime_settings["file_level"],
        runtime_settings["step_debug_interval"],
    )
    log_key_event(
        train_logger,
        logging.INFO,
        "Starting training run task=%s seed=%s logdir=%s runtime_log=%s",
        ",".join(task_names),
        dreamerv3_config.seed,
        logdir,
        logdir / "runtime.log",
    )

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
        train_logger.info(
            "WandB enabled project=%s run_name=%s entity=%s",
            getattr(wandb_cfg, "project", "CarDreamer"),
            run_name,
            getattr(wandb_cfg, "entity", ""),
        )

    logger = embodied.Logger(step, log_outputs)

    from embodied.envs import from_gym

    env = from_gym.FromGym(env)
    env = wrap_env(env, dreamerv3_config)
    env = embodied.BatchEnv([env], parallel=False)
    train_logger.info(
        "Environment ready obs_keys=%s act_keys=%s",
        sorted(env.obs_space.keys()),
        sorted(env.act_space.keys()),
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    config_filename = f"config_{timestamp}.yaml"
    config.save(str(logdir / config_filename))
    train_logger.info("Config saved to %s", logdir / config_filename)

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
    train_logger.info(
        "Agent and replay initialized agent=%s replay_size=%s batch_steps=%s runtime_level=%s",
        type(agent).__name__,
        dreamerv3_config.replay_size,
        args.batch_steps,
        runtime_settings["level"],
    )
    embodied.run.train(agent, env, replay, logger, args)
    log_key_event(train_logger, logging.INFO, "Training loop finished.")


if __name__ == "__main__":
    main()
