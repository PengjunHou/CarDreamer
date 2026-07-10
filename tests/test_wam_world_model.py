import unittest

import torch

from car_dreamer.toolkit.wam import RSSMConfig, WAMWorldModel, WorldModelConfig, symexp, symlog


def _batch(B=2, T=6, Dg=512, S=5, C=7, H=16):
    return {
        "obs_graph_embed": torch.randn(B, T, Dg),
        "obs_scalars": torch.randn(B, T, S),
        "actions": torch.stack([
            torch.randint(0, 9, (B, T)), torch.randint(0, 4, (B, T)), torch.randint(0, 2, (B, T))
        ], dim=-1),
        "rewards": -torch.rand(B, T) * 30.0,
        "is_first": torch.zeros(B, T).index_fill_(1, torch.tensor([0]), 1.0),
        "is_terminal": torch.zeros(B, T).index_fill_(1, torch.tensor([T - 1]), 1.0),
        "bev_target": (torch.rand(B, T, C, H, H) > 0.97).to(torch.uint8),
    }


class WorldModelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.cfg = WorldModelConfig(
            graph_embed_dim=512, scalar_dim=5, action_dims=(9, 4, 2), embed_dim=32, bev_size=16,
            head_hidden=32, rssm=RSSMConfig(deter=32, stoch=8, classes=6, hidden=32),
        )
        self.wm = WAMWorldModel(self.cfg)

    def test_symlog_roundtrip(self):
        x = torch.tensor([-100.0, -1.0, 0.0, 1.0, 100.0])
        self.assertTrue(torch.allclose(symexp(symlog(x)), x, atol=1e-4))

    def test_action_onehot_shape(self):
        a = torch.zeros(2, 6, 3, dtype=torch.long)
        oh = self.wm.action_onehot(a)
        self.assertEqual(oh.shape, (2, 6, 9 + 4 + 2))
        self.assertTrue(torch.allclose(oh.sum(-1), torch.full((2, 6), 3.0)))  # 3 one-hots

    def test_loss_runs_and_backprops(self):
        batch = _batch()
        total, metrics, post = self.wm.loss(batch)
        self.assertTrue(torch.isfinite(total))
        for k in ("kl", "reward", "cont", "bev", "bev_iou"):
            self.assertIn(k, metrics)
        self.assertEqual(post["deter"].shape, (2, 6, 32))
        total.backward()
        grads = [p.grad for p in self.wm.parameters() if p.grad is not None]
        self.assertTrue(len(grads) > 0)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))

    def test_imagine_shapes(self):
        batch = _batch()
        _, _, post = self.wm.loss(batch)
        start = {k: v[:, 0].detach() for k, v in post.items()}
        A = self.cfg.action_dim

        def actor(feat):  # random one-hot per factor, concatenated
            b = feat.shape[0]
            return torch.cat([
                torch.nn.functional.one_hot(torch.zeros(b, dtype=torch.long), n).float()
                for n in self.cfg.action_dims
            ], dim=-1)

        traj = self.wm.imagine(start, actor, horizon=5)
        self.assertEqual(traj["feat"].shape, (2, 5, self.cfg.rssm.feat_dim))
        self.assertEqual(traj["reward"].shape, (2, 5))
        self.assertEqual(traj["action"].shape, (2, 5, A))

    def test_loss_decreases_on_a_fixed_batch(self):
        batch = _batch()
        opt = torch.optim.Adam(self.wm.parameters(), lr=3e-3)
        first = None
        for _ in range(20):
            opt.zero_grad()
            total, _, _ = self.wm.loss(batch)
            total.backward()
            opt.step()
            if first is None:
                first = float(total)
        self.assertLess(float(total), first)  # overfits a single batch


if __name__ == "__main__":
    unittest.main()
