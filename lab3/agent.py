from collections import defaultdict
import json
import numpy as np

class QLearningAgent:
    """Q-learning агент для игры в крестики-нолики"""
    
    def __init__(self, env, learning_rate=0.01, initial_epsilon=1.0, 
                 epsilon_decay=0.001, final_epsilon=0.1, gamma=0.99):
        self.env = env
        self.lr = learning_rate
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.gamma = gamma
        
        # Инициализация Q-таблицы
        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))
        
    def get_state_key(self, state):
        """Преобразует состояние в строковый ключ для Q-таблицы"""
        return ','.join(map(str, state))
    
    def get_action(self, state):
        """Выбирает действие на основе epsilon-greedy стратегии"""
        state_key = self.get_state_key(state)
        
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        return np.argmax(self.q_table[state_key])
    
    def update(self, state, action, reward, done, next_state):
        """Обновляет Q-значения на основе полученного опыта"""
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        target = reward if done else reward + self.gamma * np.max(self.q_table[next_state_key])
        
        # Обновление Q-значения
        self.q_table[state_key][action] += self.lr * (target - self.q_table[state_key][action])
    
    def decay_epsilon(self):
        """Уменьшает значение epsilon"""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
    
    def save_q_table(self, filename='q_table.json'):
        """Сохраняет Q-таблицу в файл"""
        q_dict = {state: values.tolist() for state, values in self.q_table.items()}
        with open(filename, 'w') as f:
            json.dump(q_dict, f)
    
    def load_q_table(self, filename='q_table.json'):
        """Загружает Q-таблицу из файла"""
        with open(filename, 'r') as f:
            q_dict = json.load(f)
        
        self.q_table = defaultdict(lambda: np.zeros(self.env.action_space.n))
        for state, values in q_dict.items():
            self.q_table[state] = np.array(values)