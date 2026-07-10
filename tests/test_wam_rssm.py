import unittest

import torch

from car_dreamer.toolkit.wam.wam_rssm import RSSM, RSSMConfig


class RSSMTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.cfg = RSSMConfig(embed_dim=16, action_dim=7, deter=32, stoch=8, classes=6, hidden=32)
        self.rssm = RSSM(self.cfg)

    def test_initial_shapes(self):
        s = self.rssm.initial(4)
        self.assertEqual(s["deter"].shape, (4, 32))
        self.assertEqual(s["stoch"].shape, (4, 8, 6))
        self.assertEqual(self.rssm.get_feat(s).shape, (4, self.cfg.feat_dim))
        self.assertEqual(self.cfg.feat_dim, 32 + 8 * 6)

    def test_img_step_shapes_and_onehot(self):
        s = self.rssm.initial(3)
        a = torch.zeros(3, 7)
        p = self.rssm.img_step(s, a)
        self.assertEqual(p["deter"].shape, (3, 32))
        self.assertEqual(p["stoch"].shape, (3, 8, 6))
        # stoch is one-hot per categorical group
        self.assertTrue(torch.allclose(p["stoch"].sum(-1), torch.ones(3, 8)))

    def test_observe_and_imagine_sequence_shapes(self):
        B, T = 2, 5
        embed = torch.randn(B, T, 16)
        action = torch.zeros(B, T, 7)
        is_first = torch.zeros(B, T)
        is_first[:, 0] = 1.0
        post, prior = self.rssm.observe(embed, action, is_first)
        for st in (post, prior):
            self.assertEqual(st["deter"].shape, (B, T, 32))
            self.assertEqual(st["stoch"].shape, (B, T, 8, 6))
        # imagine from the first posterior state
        start = {k: v[:, 0] for k, v in post.items()}
        img = self.rssm.imagine(action, start)
        self.assertEqual(img["deter"].shape, (B, T, 32))

    def test_kl_loss_is_finite_and_nonneg(self):
        B, T = 2, 4
        embed = torch.randn(B, T, 16)
        action = torch.zeros(B, T, 7)
        is_first = torch.zeros(B, T)
        is_first[:, 0] = 1.0
        post, prior = self.rssm.observe(embed, action, is_first)
        out = self.rssm.kl_loss(post, prior)
        for k in ("kl", "dyn", "rep"):
            self.assertTrue(torch.isfinite(out[k]))
        self.assertGreaterEqual(float(out["dyn"].detach()), self.cfg.free_bits - 1e-4)  # free-bits floor

    def test_gradients_flow_through_rollout(self):
        B, T = 2, 4
        embed = torch.randn(B, T, 16, requires_grad=True)
        action = torch.zeros(B, T, 7)
        is_first = torch.zeros(B, T)
        is_first[:, 0] = 1.0
        post, prior = self.rssm.observe(embed, action, is_first)
        loss = self.rssm.kl_loss(post, prior)["kl"] + self.rssm.get_feat(
            {k: v[:, -1] for k, v in post.items()}
        ).pow(2).mean()
        loss.backward()
        # straight-through stoch + embed path should carry gradient
        self.assertIsNotNone(embed.grad)
        self.assertTrue(torch.isfinite(embed.grad).all())


if __name__ == "__main__":
    unittest.main()
