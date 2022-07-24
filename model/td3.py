import os
import numpy as np
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim 
from model.common.replay_buffer import ReplayBuffer
from model.common.mlp import MLP
from utils.make_window import window_deq
import wandb 

class CriticNetwork(nn.Module):
    def __init__(self, lr, input_dim, output_dim = 1, num_nuerons : list =[8,8,8], 
                  name = None, chkpt_dir = None):
        super(CriticNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_nuerons = num_nuerons
        self.lr = lr 
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_td3')

        self.network = MLP(input_dim+output_dim, 1, num_nuerons)
        self.optimizer = optim.Adam(self.network.parameters())
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.network.to(self.device)

    def forward(self, state, action):
        x = T.cat([state,action], dim=1)
        q1 = self.network(x)
        return q1 

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.network.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.network.load_state_dict(T.load(self.checkpoint_file))



class ActorNetwork(nn.Module):
    def __init__(self, max_action, lr, input_dim, output_dim, num_nuerons : list =[128,128],
                 name = None, chkpt_dir = None):
        super(ActorNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_nuerons = num_nuerons
        self.lr = lr 
        self.max_action = max_action
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_td3')


        self.network = MLP(input_dim, output_dim, num_nuerons)
        self.optimizer = optim.Adam(self.network.parameters())
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.network.to(self.device)

    def forward(self, state):
        prob = self.network(state)
        mu = (T.tanh(prob))*self.max_action
        return mu

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

class TD3():
    def __init__(self, env, actor_lr, critic_lr, input_dim, output_dim, tau, chkpt_dir,
                       actor_num_nuerons = [8,8,8], critic_num_nuerons = [128,128], exp_noise=0.1,pol_noise=0.2,
                       noise_clip=0.5, gamma=1, update_actor_interval=2, warmup=1000, max_size=1000000, batch_size=100):

            self.env = env 
            self.actor_lr = actor_lr
            self.critic_lr = critic_lr 
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.tau = tau 
            self.gamma = gamma
            self.update_actor_iter = update_actor_interval
            self.warmup = warmup
            self.exp_noise = exp_noise
            self.pol_noise = pol_noise
            self.noise_clip = noise_clip
            self.memory  = ReplayBuffer(max_size, [input_dim], output_dim)
            self.batch_size = batch_size 
            self.learn_step_cntr =0 
            self.time_step = 0
            
            self.max_action = env.action_space.high[0]
            self.min_action = env.action_space.low[0]

            self.actor_num_nuerons = actor_num_nuerons
            self.critic_num_nuerons = critic_num_nuerons

            self.actor = ActorNetwork(self.max_action, actor_lr, input_dim, output_dim, actor_num_nuerons, 
                                          name='actor', chkpt_dir=chkpt_dir)
            self.critic_1 = CriticNetwork(critic_lr, input_dim, output_dim, critic_num_nuerons, 
                                          name='critic_1', chkpt_dir=chkpt_dir)
            self.critic_2 = CriticNetwork(critic_lr, input_dim, output_dim, critic_num_nuerons, 
                                          name='critic_2', chkpt_dir=chkpt_dir)
            
            self.target_actor = ActorNetwork(self.max_action, actor_lr, input_dim, output_dim, actor_num_nuerons, 
                                          name='target_actor', chkpt_dir=chkpt_dir)
            self.target_critic_1 = CriticNetwork(critic_lr, input_dim, output_dim, critic_num_nuerons, 
                                          name='critic_2', chkpt_dir=chkpt_dir)
            self.target_critic_2 = CriticNetwork(critic_lr, input_dim, output_dim, critic_num_nuerons, 
                                          name='critic_2', chkpt_dir=chkpt_dir)

            self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        if self.time_step < self.warmup:
            # exploration for early phase, hyperparmeter (default : uniform sampling )
            mu = T.tensor(np.random.uniform(self.min_action, self.max_action, size=(self.output_dim,)))
        else:
            state = T.tensor(observation, dtype=T.float).to(self.actor.device)
            mu = self.actor.forward(state).to(self.actor.device)

            # exploration for later phase, hyperparmeter 
        mu_prime = mu + T.tensor(np.random.normal(scale=self.exp_noise*self.max_action),
                                    dtype=T.float).to(self.actor.device)

        mu_prime = T.clamp(mu_prime, self.min_action, self.max_action)
        self.time_step += 1

        return mu_prime.cpu().detach().numpy()

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def train(self):
        if self.memory.mem_cntr < self.batch_size:
            return 

        # sample from replay buffer
        state, action, reward, new_state, done = \
                self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.critic_1.device)
        done = T.tensor(done).to(self.critic_1.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.critic_1.device)
        state = T.tensor(state, dtype=T.float).to(self.critic_1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic_1.device)

        # Select action according to policy and add clipped noise
        target_actions = self.target_actor.forward(state_)
        target_actions = target_actions + \
                T.clamp(T.tensor(np.random.normal(scale=self.pol_noise)), -self.noise_clip, self.noise_clip) # smoothing Q-value, hyperparameter
        target_actions = T.clamp(target_actions, self.min_action, 
                                self.max_action)
        
        # Compute the target Q value
        q1_ = self.target_critic_1.forward(state_, target_actions)
        q2_ = self.target_critic_2.forward(state_, target_actions)

        q1 = self.critic_1.forward(state, action)
        q2 = self.critic_2.forward(state, action)

        q1_[done] = 0.0
        q2_[done] = 0.0

        q1_ = q1_.view(-1)
        q2_ = q2_.view(-1)

        critic_value_ = T.min(q1_, q2_)

        target = reward + self.gamma*critic_value_
        target = target.view(self.batch_size, 1)

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)
        critic_loss = q1_loss + q2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()
        
        # Delayed policy updates
        self.learn_step_cntr += 1

        if self.learn_step_cntr % self.update_actor_iter != 0:
            return

        self.actor.optimizer.zero_grad()
        actor_q1_loss = self.critic_1.forward(state, self.actor.forward(state))
        actor_loss = -T.mean(actor_q1_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()

        critic_1 = dict(critic_1_params)
        critic_2 = dict(critic_2_params)
        actor = dict(actor_params)
        target_actor = dict(target_actor_params)
        target_critic_1 = dict(target_critic_1_params)
        target_critic_2 = dict(target_critic_2_params)

        # soft update 
        for name in critic_1:
            critic_1[name] = tau*critic_1[name].clone() + \
                    (1-tau)*target_critic_1[name].clone()

        for name in critic_2:
            critic_2[name] = tau*critic_2[name].clone() + \
                    (1-tau)*target_critic_2[name].clone()

        for name in actor:
            actor[name] = tau*actor[name].clone() + \
                    (1-tau)*target_actor[name].clone()

        self.target_critic_1.load_state_dict(critic_1)
        self.target_critic_2.load_state_dict(critic_2)
        self.target_actor.load_state_dict(actor)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.target_critic_2.load_checkpoint()


    def learn(self, total_timesteps=7e3, window=1, wandb_log=False):
        best_score = -1e7
        epi_rewards = [-1e7] 
        epi_states = [] 
        epi_actions = [] 

        rewards = []
        states = []
        actions = [] 
        
        window_deque = window_deq(state_dim=self.input_dim, window_size=window)
        state = self.env.reset()
        window_deque.put_info(state)
        done = False
        time_step = 0 
        episode_num = 0 
        episode_reward = 0
        for t in range(int(total_timesteps)):
            concate_state = window_deque.concat_state()
            time_step+=1 
            act = self.choose_action(concate_state)
            next_state, reward, done, _ = self.env.step(action = np.clip(a=act,a_min=self.min_action, a_max=self.max_action))
            window_deque.put_info(next_state)
            concate_state_ = window_deque.concat_state()
            self.remember(concate_state, act, reward, concate_state_, done)
            self.train()
            state = next_state
            episode_reward += reward

            # for saving 
            rewards.append(reward)
            states.append(state)
            actions.append(act)

            if done : 
                print(f'Avg_reward:{np.mean(epi_rewards[-50:]):.3f}, Episode Num: {episode_num+1}, Episode T: {time_step}, Reward: {np.sum(rewards):.3f}, action:{act}, state:{state}')
                if wandb_log :
                    wandb.log({'epi_reward':np.sum(rewards)})
                epi_rewards.append(np.sum(rewards))
                epi_actions.append(actions)
                epi_states.append(states)
                rewards = []
                states = []
                actions = []
                state = self.env.reset()
                done = False 
                episode_reward = 0 
                episode_num +=1  
                avg_reward = np.mean(epi_rewards[-30:])
               

                if t>self.warmup and avg_reward > best_score :
                    print(f"--------------- save model :{avg_reward :.3f} , prev best:{best_score :.3f} --------------------- ")
                    self.save_models()
                    best_score = avg_reward
                
        return {"states":epi_states, "actions":epi_actions, "rewards":epi_rewards}

                
    def test(self, total_timesteps=7e2, wandb_log=False, window=1):
        self.exp_noise = 0
        epi_rewards = [] 
        epi_states = [] 
        epi_actions = [] 

        rewards = []
        states = []
        actions = [] 
        
        window_deque = window_deq(state_dim=self.input_dim, window_size=window)
        state = self.env.reset()
        window_deque.put_info(state)
        done = False
        time_step = 0 
        episode_num = 0 
        episode_reward = 0
        for t in range(int(total_timesteps)):
            concate_state = window_deque.concat_state()
            time_step+=1 
            act = self.choose_action(concate_state)
            next_state, reward, done, _ = self.env.step(action = np.clip(a=act,a_min=self.min_action, a_max=self.max_action))
            window_deque.put_info(next_state)
            concate_state_ = window_deque.concat_state()
            state = next_state
            episode_reward += reward

            # for saving 
            rewards.append(reward)
            states.append(state)
            actions.append(act)

            if done : 
                print(f'Episode Num: {episode_num+1}, Episode T: {time_step}, Reward: {np.sum(rewards):.3f}, action:{act}, state:{state}')
                if wandb_log :
                    wandb.log({'test_epi_reward':np.sum(rewards)})
                epi_rewards.append(rewards)
                epi_actions.append(actions)
                epi_states.append(states)
                rewards = []
                states = []
                actions = []
                state = self.env.reset()
                done = False 
                episode_reward = 0 
                episode_num +=1  
                avg_reward = np.mean(epi_rewards[-50:])

                
        return {"states":epi_states, "actions":epi_actions, "rewards":epi_rewards}





            

    



