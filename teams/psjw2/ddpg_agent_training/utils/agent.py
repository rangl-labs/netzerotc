from torch.optim.lr_scheduler import CosineAnnealingLR

from ddpg_agent_training.utils.models import actor, critic
import torch
import os
from ddpg_agent_training.utils.replay_buffer import replay_buffer
from ddpg_agent_training.utils.sampler import sampler

import numpy as np
from datetime import datetime
import csv

class agent:
    def __init__(self, args, env, env_params):
        self.args = args
        self.env = env

        self.env_params = env_params
        # create the network
        self.actor_network = actor(env_params)
        self.critic_network = critic(env_params)
        # build up the target network
        self.actor_target_network = actor(env_params)
        self.critic_target_network = critic(env_params)
        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)

        self.b_sampler = sampler()
        # create the replay buffer
        self.buffer = replay_buffer(self.env_params, self.args.buffer_size, self.b_sampler.sample_her_transitions)
        self.max_action_tensor = torch.tensor( self.env_params['action_max'], dtype=torch.float32).unsqueeze(0)
        self.seeds = self.read_seeds()[0::4]

        #TODO: Normalizers

    def learn(self):
        """
        Training the network

        """
        for epoch in range(self.args.n_epochs):
            noise_counters = 0
            for _ in range(self.args.n_cycles):
                mb_obs, mb_actions, mb_rewards = [], [], []
                for _ in range(self.args.num_rollouts_per_mpi):

                    # reset the rollouts
                    ep_obs, ep_actions, ep_rewards = [], [], []
                    # reset the environment
                    observation = self.env.reset()
                    # start to collect samples
                    for t in range(self.env_params['max_timesteps']):
                        with torch.no_grad():
                            input_tensor = self._preproc_inputs(observation)
                            pi = self.actor_network(input_tensor)
                            action = self._select_actions(pi) #pi.cpu().numpy().squeeze()


                        # feed the actions into the environment
                        observation_new, reward, done, _ = self.env.step(action)

                        # append rollouts
                        ep_obs.append(observation.copy())
                        ep_actions.append(action.copy())
                        ep_rewards.append(reward)
                        # re-assign the observation
                        observation = observation_new
                    ep_obs.append(observation.copy())
                    mb_obs.append(ep_obs)
                    mb_actions.append(ep_actions)
                    mb_rewards.append(ep_rewards)
                # convert them into arrays
                mb_obs = np.array(mb_obs)
                mb_actions = np.array(mb_actions)
                mb_rewards = np.array(mb_rewards)

                # store the episodes
                self.buffer.store_episode([mb_obs, mb_actions, mb_rewards])
                for _ in range(self.args.n_batches):
                    # train the network
                    self._update_network()
                # soft update
                self._soft_update_target_network(self.actor_target_network, self.actor_network)
                self._soft_update_target_network(self.critic_target_network, self.critic_network)

            # start to do the evaluation
            mean_reward = self._eval_agent()
            print(f"mean reward: {mean_reward}")
            print('[{}] epoch is: {}, mean reward: {:.3f}'.format(datetime.now(), epoch, mean_reward))

            torch.save(self.actor_network.state_dict(), "trained_models/model.pt")
            print(f"{noise_counters} / {self.env_params['max_timesteps']}")
            # torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std, self.actor_network.state_dict()], \
            #             self.model_path + '/model.pt')


    def _preproc_inputs(self, obs):
        inputs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        return inputs


    # this function will choose action for the agent and do the exploration
    def _select_actions(self, pi):

        action = pi.cpu().numpy().squeeze()

        # add the gaussian
        action += self.args.noise_eps * self.env_params['action_max'] * np.random.randn(*action.shape)
        action = np.clip(action, -self.env_params['action_max'], self.env_params['action_max'])
        # random actions...
        random_actions = np.random.uniform(low=-self.env_params['action_max'][0], high=self.env_params['action_max'][1], \
                                            size=self.env_params['action'])
        # choose if use the random actions
        action += np.random.binomial(1, self.args.random_eps, 1)[0] * (random_actions - action)
        return action

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    # update the network
    def _update_network(self):
        # sample the episodes
        transitions = self.buffer.sample(self.args.batch_size)

        # pre-process the observation and goal
        o, o_next = transitions['obs'], transitions['obs_next']
        transitions['obs'] = o #self._preproc_og(o, g)
        transitions['obs_next'] = o_next #self._preproc_og(o_next, g)
        # start to do the update
        obs_norm = transitions['obs']
        inputs_norm = obs_norm

        obs_next_norm = transitions['obs_next']
        inputs_next_norm = obs_next_norm

        # transfer them into the tensor
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32)

        if self.args.cuda:
            inputs_norm_tensor = inputs_norm_tensor.cuda()
            inputs_next_norm_tensor = inputs_next_norm_tensor.cuda()
            actions_tensor = actions_tensor.cuda()
            r_tensor = r_tensor.cuda()
        # calculate the target Q value function
        with torch.no_grad():
            # do the normalization
            actions_next = self.actor_target_network(inputs_next_norm_tensor)
            q_next_value = self.critic_target_network(inputs_next_norm_tensor, actions_next)
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + self.args.gamma * q_next_value
            target_q_value = target_q_value.detach()

        # the q loss
        real_q_value = self.critic_network(inputs_norm_tensor, actions_tensor)
        critic_loss = (target_q_value - real_q_value).pow(2).mean() #qf loss
        # the actor loss
        actions_real = self.actor_network(inputs_norm_tensor)

        actor_loss = -self.critic_network(inputs_norm_tensor, actions_real).mean()
        # actor_loss += self.args.action_l2 * (actions_real / self.max_action_tensor).pow(2).mean()
        actor_loss += self.args.action_l2 * (actions_real).pow(2).mean()

        # start to update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()

        self.actor_optim.step()
        # update the critic_network
        self.critic_optim.zero_grad()
        critic_loss.backward()

        self.critic_optim.step()

    def read_seeds(self, fname="seeds.csv"):
        file = open(fname)
        csv_file = csv.reader(file)
        seeds = []
        for row in csv_file:
            seeds.append(int(row[0]))
        self.seeds = seeds
        return seeds


    def _eval_agent(self):
        total_reward = []
        for seed in self.seeds:
            self.env.seed(seed)
            for _ in range(self.args.n_test_rollouts):
                per_reward = []
                observation = self.env.reset()
                obs = observation
                while not self.env.state.is_done():
                # for _ in range(self.env_params['max_timesteps']):
                    with torch.no_grad():
                        input_tensor = self._preproc_inputs(obs)
                        pi = self.actor_network(input_tensor)
                        # convert the actions
                        actions = pi.detach().cpu().numpy().squeeze()
                    observation_new, reward, done, _ = self.env.step(actions)
                    obs = observation_new
                total_reward.append(sum(self.env.state.rewards_all))
        return np.mean(total_reward)