import torch
from typing import Tuple
from torch import Tensor

from elegantrl.train.config import Config
from elegantrl.agents.AgentBase import AgentBase
from elegantrl.agents.net import ActorPPO, CriticPPO
from elegantrl.agents.net import ActorDiscretePPO

from elegantrl.agents import AgentPPO, AgentA2C, AgentDDPG, AgentSAC


def kl_divergence(p, q, epsilon=1e-10):
    p = torch.clamp(p, epsilon, 1)
    q = torch.clamp(q, epsilon, 1)
    return torch.sum(p * torch.log(p / q), dim=-1)


class AgentPPOKL(AgentPPO):
    def __init__(
        self,
        net_dims: [int],
        state_dim: int,
        action_dim: int,
        gpu_id: int = 0,
        args: Config = Config(),
    ):
        super().__init__(net_dims, state_dim, action_dim, gpu_id, args)
        self.lambda_kl = 0.01

    def update_net(self, buffer, previous_actions) -> Tuple[float, ...]:
        with torch.no_grad():
            states, actions, logprobs, rewards, undones = buffer
            buffer_size = states.shape[0]
            buffer_num = states.shape[1]

            '''get advantages and reward_sums'''
            bs = 2 ** 10  # set a smaller 'batch_size' to avoiding out of GPU memory.
            values = torch.empty_like(rewards)  # values.shape == (buffer_size, buffer_num)
            for i in range(0, buffer_size, bs):
                for j in range(buffer_num):
                    values[i:i + bs, j] = self.cri(states[i:i + bs, j])

            advantages = self.get_advantages(rewards, undones, values)  # shape == (buffer_size, buffer_num)
            reward_sums = advantages + values  # shape == (buffer_size, buffer_num)
            del rewards, undones, values

            advantages = (advantages - advantages.mean()) / (advantages.std(dim=0) + 1e-4)

            self.update_avg_std_for_normalization(
                states=states.reshape((-1, self.state_dim)),
                returns=reward_sums.reshape((-1,))
            )
            """compute average KL divergence if previous_actions is not None"""
            if previous_actions:
                kl_divs = torch.stack(
                    [kl_divergence(actions, pa) for pa in previous_actions]
                )
                kl_loss = kl_divs.mean()  # Average over all previous actions
            else:
                kl_loss = 0.0

        """update network with KL divergence penalty"""
        obj_critics = 0.0
        obj_actors = 0.0
        sample_len = buffer_size - 1

        update_times = int(buffer_size * self.repeat_times / self.batch_size)
        assert update_times >= 1
        for _ in range(update_times):
            ids = torch.randint(sample_len * buffer_num, size=(self.batch_size,), requires_grad=False)
            ids0 = torch.fmod(ids, sample_len)  # ids % sample_len
            ids1 = torch.div(ids, sample_len, rounding_mode='floor')  # ids // sample_len

            state = states[ids0, ids1]
            action = actions[ids0, ids1]
            logprob = logprobs[ids0, ids1]
            advantage = advantages[ids0, ids1]
            reward_sum = reward_sums[ids0, ids1]

            value = self.cri(state)  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, reward_sum)
            self.optimizer_update(self.cri_optimizer, obj_critic)

            new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)
            ratio = (new_logprob - logprob.detach()).exp()
            surrogate1 = advantage * ratio
            surrogate2 = advantage * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = torch.min(surrogate1, surrogate2).mean()
            
            obj_actor = (
                obj_surrogate
                + obj_entropy.mean() * self.lambda_entropy
                - self.lambda_kl * kl_loss
            )
            self.optimizer_update(self.act_optimizer, -obj_actor)

            obj_critics += obj_critic.item()
            obj_actors += obj_actor.item()

        return obj_critics / update_times, obj_actors / update_times, kl_loss  # .item()
