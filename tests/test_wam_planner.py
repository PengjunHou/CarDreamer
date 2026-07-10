import unittest

import torch

from car_dreamer.toolkit.wam import (
    ActionSpec,
    ActorCriticConfig,
    FactorizedActor,
    RSSMConfig,
    WAMDreamerPolicy,
    WAMWorldModel,
    WorldModelConfig,
)


class DreamerPolicyTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.wmcfg = WorldModelConfig(
            graph_embed_dim=32, scalar_dim=5, action_dims=(9, 4, 2), embed_dim=16, bev_size=16,
            head_hidden=16, rssm=RSSMConfig(deter=16, stoch=4, classes=4, hidden=16),
        )
        wm = WAMWorldModel(self.wmcfg)
        accfg = ActorCriticConfig(feat_dim=self.wmcfg.rssm.feat_dim, action_dims=(9, 4, 2), hidden=16, layers=2)
        actor = FactorizedActor(accfg)
        spec = ActionSpec(max_members=8, bandwidth_grid=(0.2, 0.5, 0.8, 1.0),
                          modalities=("objlist", "bev"), use_duration=False, step_slots=2)
        self.policy = WAMDreamerPolicy(wm, actor, spec, device=torch.device("cpu"))

    def _obs(self):
        return torch.randn(32), torch.randn(5)

    def test_act_returns_subaction(self):
        ge, sc = self._obs()
        sub = self.policy.act(ge, sc, candidate_ids=[7, 3, 9], is_first=True)
        # local-only or a cooperative sub-action over a valid candidate; duration = step_slots
        self.assertEqual(sub.duration_slots, 2)
        if not sub.is_local_only:
            self.assertIn(sub.selected[0], [7, 3, 9])
            self.assertIn(sub.modality_by_vehicle[sub.selected[0]], ("objlist", "bev"))

    def test_state_persists_across_epochs(self):
        ge, sc = self._obs()
        self.policy.act(ge, sc, [7, 3], is_first=True)
        s1 = {k: v.clone() for k, v in self.policy._state.items()}
        self.policy.act(*self._obs(), candidate_ids=[7, 3])
        s2 = self.policy._state
        self.assertFalse(torch.equal(s1["deter"], s2["deter"]))  # RSSM advanced

    def test_reset_clears_state(self):
        self.policy.act(*self._obs(), candidate_ids=[7], is_first=True)
        self.policy.reset()
        self.assertIsNone(self.policy._state)
        self.assertIsNone(self.policy._prev_action)

    def test_deterministic_given_same_obs_and_state(self):
        ge, sc = self._obs()
        a = self.policy.act(ge, sc, [7, 3, 9], is_first=True)
        self.policy.reset()
        b = self.policy.act(ge, sc, [7, 3, 9], is_first=True)
        self.assertEqual(a.selected, b.selected)  # mode() is deterministic


if __name__ == "__main__":
    unittest.main()
