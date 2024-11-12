import random
from tqdm import tqdm

from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

from gymnasium import Env, spaces


# Использована статья https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/
class Environment(Env):
    def __init__(self, candidates_cnt):
        self.candidates_cnt = candidates_cnt
        self.observation_space = spaces.Discrete(candidates_cnt)  # выбор из всех кандидатов
        self.action_space = spaces.Discrete(2)  # действия - либо принять, либо отклонить
        self.reset()
    
    def _get_obs(self):
        return {
            'is_better': self.candidates[self.cur_id] >= self.max_candidate,
        }


    def _get_info(self):
        return {
            'cur_rank': self.ranks[self.cur_id],
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # create random data and rank it
        candidates = np.random.randint(1000, size=self.candidates_cnt, dtype='int')
        candidates[::-1].sort()
        ranks = np.array(list(range(self.candidates_cnt)))
        candidates_dict = {ranks[i]: candidates[i] for i in range(self.candidates_cnt)}
        cur_items = list(candidates_dict.items())
        random.shuffle(cur_items)
        candidates_data = dict(cur_items)  # rank -> candidate value

        self.candidates = np.array(list(candidates_data.values()))
        self.ranks = np.array(list(candidates_data.keys()))

        self.cur_id = 0
        self.cur_candidate_rank = self.candidates[self.cur_id]
        self.max_candidate = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def step(self, action):
        self.cur_candidate = self.candidates[self.cur_id]

        if self.candidates[self.cur_id] >= self.max_candidate:
            self.max_candidate = self.cur_candidate

        if action == 1:
            # выбираем кандидата и завершаем
            terminated = True
            reward = 1 if self.ranks[self.cur_id] == 0 else 0

        elif self.cur_id == self.candidates_cnt - 1:
            # остался последний - надо брать и завершать
            terminated = True
            reward = 1 if self.ranks[self.cur_id] == 0 else 0
        else:
            # не берем, ищем дальше
            terminated = False
            reward = 0
            self.cur_id += 1

        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, False, info


# Использована статья https://gymnasium.farama.org/main/introduction/train_agent/
class Agent:
    def __init__(self, env, lr, initial_epsilon, epsilon_decay, final_epsilon, discount_factor):
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = lr
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []
    
    def get_action(self, obs):
        if np.random.random() < self.epsilon:
            # exploration
            return obs['is_better']
        else:
            # exploitation
            return int(np.argmax(self.q_values[obs.values()]))

    def update(
            self, obs,
            action, reward,
            terminated, next_obs,
    ):
        future_q_value = (not terminated) * np.max(self.q_values[next_obs.values()])
        temporal_difference = (
                reward + self.discount_factor * future_q_value - self.q_values[obs.values()][action]
        )

        self.q_values[obs.values()][action] = (
                self.q_values[obs.values()][action] + self.lr * temporal_difference
        )

        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)



def train_agent(candidates_cnt, n_episodes):
    lr = 0.01
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (n_episodes / 2)
    final_epsilon = 0.1
    discount_factor = 0.95

    env = Environment(candidates_cnt)

    agent = Agent(
        env=env,
        lr=lr,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
        discount_factor=discount_factor,
    )

    ranks_of_chosen = []

    for _ in tqdm(range(n_episodes)):
        obs, info = env.reset()
        finished = False

        while not finished:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.update(obs, action, reward, terminated, next_obs)
            finished = terminated
            obs = next_obs

        ranks_of_chosen.append(info['cur_rank'])
        agent.decay_epsilon()

    return ranks_of_chosen[-int(n_episodes // 2):]


if __name__ == '__main__':
    candidates_num = 100
    n_episodes = 100000
    results = train_agent(candidates_num, n_episodes)

    bins = list(range(0, candidates_num + 1, 10))

    plt.figure(figsize=(10, 6))

    plt.title('RL results')
    plt.hist(results, bins=candidates_num)
    plt.xlabel('Chosen candidate')
    plt.show()
