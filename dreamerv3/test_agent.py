import numpy as np
import embodied

class TestAgent(embodied.Agent):
    def __init__(self, obs_space, act_space, step, config):
        self.obs_space = obs_space
        self.act_space = act_space
        self.step = step

        # embodied 通常是 dict action space
        self.act_dim = int(self.act_space["action"].shape[0])  # e.g., 15

        # 你的离散动作配置：n_acc=3, n_steer=5
        self.n_steer = 5
        self.acc_index = 2    # discrete_acc: [-2,0,2] -> pick +2
        self.steer_index = 2  # discrete_steer: [-0.6,-0.2,0,0.2,0.6] -> pick 0
        self.action_index = self.acc_index * self.n_steer + self.steer_index  # 12

    def policy(self, obs, state=None, mode="train"):
        # B = num_envs
        B = len(obs["is_first"]) if "is_first" in obs else 1

        # one-hot action
        action = np.zeros((B, self.act_dim), dtype=np.float32)
        action[:, self.action_index] = 1.0

        reset = np.zeros((B,), dtype=bool)
        return {"action": action, "reset": reset}, state
    
    # def policy(self, obs, state=None, mode="train"):
    #     B = len(obs["is_first"]) if "is_first" in obs else 1
    #     action = np.full((B, 1), self.action_index, dtype=np.int64)  # [B,1]
    #     return {"action": action}, state
    
    def dataset(self, generator_fn):
        return generator_fn()
    
    def train(self, data, state=None):
        print(f"Training on batch of data with keys: {list(data.keys())}")
        return {}, state, {}
    
    def report(self, data):
        return {}
    
    def save(self):
        return {}
    
    def load(self, data):
        pass
    
    def sync(self):
        pass