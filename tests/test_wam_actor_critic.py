import unittest

import torch

from car_dreamer.toolkit.wam import (
    ActorCriticConfig,
    Critic,
    FactorizedActor,
    RSSMConfig,
    WAMActorCritic,
    WAMWorldModel,
    WorldModelConfig,
    lambda_return,
)


def _world_model():
    cfg = WorldModelConfig(
        graph_embed_dim=32, scalar_dim=5, action_dims=(9, 4, 2), embed_dim=16, bev_size=16,
        head_hidden=16, rssm=RSSMConfig(deter=16, stoch=4, classes=4, hidden=16),
    )
    return WAMWorldModel(cfg), cfg


class ActorCriticTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.wm, self.wmcfg = _world_model()
        self.cfg = ActorCriticConfig(
            feat_dim=self.wmcfg.rssm.feat_dim, action_dims=(9, 4, 2),
            hidden=16, layers=2, imag_horizon=5,
        )
        self.ac = WAMActorCritic(self.wm, self.cfg, device=torch.device("cpu"))

    def _start(self, n=6):
        return {
            "deter": torch.randn(n, self.wmcfg.rssm.deter),
            "logit": torch.randn(n, self.wmcfg.rssm.stoch, self.wmcfg.rssm.classes),
            "stoch": torch.nn.functional.one_hot(
                torch.randint(0, self.wmcfg.rssm.classes, (n, self.wmcfg.rssm.stoch)),
                self.wmcfg.rssm.classes,
            ).float(),
        }

    def test_actor_dist_sample_logprob_entropy(self):
        actor = FactorizedActor(self.cfg)
        feat = torch.randn(4, self.cfg.feat_dim)
        dist = actor(feat)
        a = dist.sample()
        self.assertEqual(a.shape, (4, self.cfg.action_dim))
        self.assertTrue(torch.allclose(a.sum(-1), torch.full((4,), 3.0)))  # three one-hots
        self.assertEqual(dist.log_prob(a).shape, (4,))
        self.assertEqual(dist.entropy().shape, (4,))
        self.assertEqual(dist.mode().shape, (4, self.cfg.action_dim))

    def test_lambda_return_shape_and_bootstrap(self):
        reward = torch.zeros(3, 5)
        value = torch.ones(3, 5)
        disc = torch.full((3, 5), 0.9)
        ret = lambda_return(reward, value, disc, lam=0.95)
        self.assertEqual(ret.shape, (3, 5))
        # zero reward, constant value/disc -> returns positive (bootstrapped value)
        self.assertTrue((ret > 0).all())

    def test_rollout_shapes(self):
        traj = self.ac.rollout(self._start(6))
        H = self.cfg.imag_horizon
        self.assertEqual(traj["feat"].shape, (6, H, self.cfg.feat_dim))
        self.assertEqual(traj["action"].shape, (6, H, self.cfg.action_dim))
        self.assertEqual(traj["reward"].shape, (6, H))
        self.assertEqual(traj["cont"].shape, (6, H))

    def test_train_step_runs_and_updates_actor(self):
        before = [p.clone() for p in self.ac.actor.parameters()]
        metrics = self.ac.train_step(self._start(8))
        for k in ("actor_loss", "critic_loss", "return_mean", "entropy"):
            self.assertIn(k, metrics)
            self.assertTrue(torch.isfinite(torch.tensor(metrics[k])))
        after = list(self.ac.actor.parameters())
        self.assertTrue(any(not torch.equal(b, a) for b, a in zip(before, after)))

    def test_world_model_stays_frozen(self):
        wm_before = [p.clone() for p in self.wm.parameters()]
        self.ac.train_step(self._start(8))
        wm_after = list(self.wm.parameters())
        self.assertTrue(all(torch.equal(b, a) for b, a in zip(wm_before, wm_after)))


if __name__ == "__main__":
    unittest.main()
