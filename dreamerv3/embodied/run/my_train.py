import re
import logging

import embodied
# import jax
import numpy as np

from runtime_logging import get_runtime_logger, log_key_event


TRAIN_LOGGER = get_runtime_logger("dreamerv3.train")


def train(agent, env, replay, logger, args):
    logdir = embodied.Path(args.logdir)
    logdir.mkdirs()
    TRAIN_LOGGER.info("Train loop initialized logdir=%s", logdir)
    should_expl = embodied.when.Until(args.expl_until)
    should_train = embodied.when.Ratio(args.train_ratio / args.batch_steps)
    should_log = embodied.when.Clock(args.log_every)
    should_save = embodied.when.Clock(args.save_every)
    should_sync = embodied.when.Every(args.sync_every)
    step = logger.step
    updates = embodied.Counter()
    metrics = embodied.Metrics()
    TRAIN_LOGGER.info("Observation space:\n%s", embodied.format(env.obs_space))
    TRAIN_LOGGER.info("Action space:\n%s", embodied.format(env.act_space))

    timer = embodied.Timer()
    # timer.wrap("agent", agent, ["policy", "train", "report", "save"])
    timer.wrap("env", env, ["step"])
    timer.wrap("replay", replay, ["add", "save"])
    timer.wrap("logger", logger, ["write"])

    nonzeros = set()

    def per_episode(ep, ep_info):
        length = len(ep["reward"]) - 1
        score = float(ep["reward"].astype(np.float64).sum())
        sum_abs_reward = float(np.abs(ep["reward"]).astype(np.float64).sum())
        logger.add(
            {
                "length": length,
                "score": score,
                "sum_abs_reward": sum_abs_reward,
                "reward_rate": (np.abs(ep["reward"]) >= 0.5).mean(),
                **ep_info,
            },
            prefix="episode",
        )
        log_key_event(
            TRAIN_LOGGER,
            logging.INFO,
            "Episode finished length=%d score=%.1f r_waypoints=%.1f r_speed=%.1f "
            "r_collision=%.1f r_out_of_lane=%.1f r_destination=%.1f time_penalty=%.1f",
            length,
            score,
            ep["r_waypoints"].sum(),
            ep["r_speed"].sum(),
            ep["r_collision"].sum(),
            ep["r_out_of_lane"].sum(),
            ep["r_destination"].sum(),
            ep["time_penalty"].sum(),
        )
        stats = {}
        for key in args.log_keys_video:
            if key in ep:
                stats[f"policy_{key}"] = ep[key]
        for key, value in ep.items():
            if not args.log_zeros and key not in nonzeros and (value == 0).all():
                continue
            nonzeros.add(key)
            if re.match(args.log_keys_sum, key):
                stats[f"sum_{key}"] = ep[key].sum()
            if re.match(args.log_keys_mean, key):
                stats[f"mean_{key}"] = ep[key].mean()
            if re.match(args.log_keys_max, key):
                stats[f"max_{key}"] = ep[key].max(0).mean()
        metrics.add(stats, prefix="stats")

    driver = embodied.Driver(env)
    driver.on_episode(lambda ep, ep_info, worker: per_episode(ep, ep_info))
    driver.on_step(lambda _, __, ___: step.increment())
    driver.on_step(lambda tran, _, worker: replay.add(tran, worker))

    log_key_event(
        TRAIN_LOGGER,
        logging.INFO,
        "Prefill train dataset start replay_length=%d target=%d",
        len(replay),
        max(args.batch_steps, args.train_fill),
    )
    random_agent = embodied.RandomAgent(env.act_space, args.actor_dist_disc)
    TRAIN_LOGGER.info(
        "Collecting experience with random policy batch_steps=%s train_fill=%s",
        args.batch_steps,
        args.train_fill,
    )
    while len(replay) < max(args.batch_steps, args.train_fill):
        driver(random_agent.policy, steps=100)
    log_key_event(TRAIN_LOGGER, logging.INFO, "Prefill complete replay_length=%d", len(replay))
    logger.add(metrics.result())
    logger.write()
    TRAIN_LOGGER.debug("Initial metrics written after prefill.")

    # dataset = agent.dataset(replay.dataset)
    state = [None]  # To be writable from train step function below.
    batch = [None]

    # def train_step(_, __, ___):
    #     for _ in range(should_train(step)):
    #         with timer.scope("dataset"):
    #             batch[0] = next(dataset)
    #         outs, state[0], mets = agent.train(batch[0], state[0])
    #         metrics.add(mets, prefix="train")

    #         if getattr(replay, "update_visit_count", False):
    #             replay.update_visit_count(jax.device_get(batch[0]["env_step"]))

    #         if "key" in outs:
    #             replay.prioritize(outs["key"], outs["env_step"], outs["model_loss"], outs["td_error"])

    #         updates.increment()
    #     if should_sync(updates):
    #         agent.sync()
    #     if should_log(step):
    #         agg = metrics.result()
    #         report = agent.report(batch[0])
    #         report = {k: v for k, v in report.items() if "train/" + k not in agg}
    #         logger.add(agg)
    #         logger.add(report, prefix="report")
    #         logger.add(replay.stats, prefix="replay")
    #         logger.add(timer.stats(), prefix="timer")
    #         logger.write(fps=True)
        # print(f"train_step called at step {step.value}, Not implemented yet.")

    # driver.on_step(train_step)

    checkpoint = embodied.Checkpoint(logdir / "checkpoint.ckpt")
    timer.wrap("checkpoint", checkpoint, ["save", "load"])
    checkpoint.step = step
    # checkpoint.agent = agent
    checkpoint.replay = replay
    if args.from_checkpoint:
        log_key_event(TRAIN_LOGGER, logging.INFO, "Loading checkpoint from %s", args.from_checkpoint)
        checkpoint.load(args.from_checkpoint)
    checkpoint.load_or_save()
    should_save(step)  # Register that we jused saved.
    TRAIN_LOGGER.info("Checkpoint initialized at %s", logdir / "checkpoint.ckpt")

    log_key_event(
        TRAIN_LOGGER,
        logging.INFO,
        "Start training loop total_steps=%s train_ratio=%s",
        args.steps,
        args.train_ratio,
    )
    driver._state = None
    # policy = lambda *args: agent.policy(*args, mode="explore" if should_expl(step) else "train")
    policy = lambda *args: random_agent.policy(*args)
    while step < args.steps:
        driver(policy, steps=100)
        if should_save(step):
            log_key_event(TRAIN_LOGGER, logging.INFO, "Saving checkpoint at env_step=%s", step.value)
            checkpoint.save()
        if should_log(step):
            TRAIN_LOGGER.debug("Periodic log tick env_step=%s updates=%s", step.value, updates.value)
    logger.write()
    log_key_event(TRAIN_LOGGER, logging.INFO, "Training loop complete final_step=%s", step.value)
