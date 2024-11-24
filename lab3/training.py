from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from environment import TicTacToeEnv
from agent import QLearningAgent

def plot_learning_curve(rewards, window_size=1000):
    """Отображает график обучения с использованием скользящего среднего"""
    smoothed_rewards = np.convolve(rewards, 
                                 np.ones(window_size) / window_size, 
                                 mode='valid')
    
    plt.figure(figsize=(10, 6))
    plt.plot(smoothed_rewards, label='Скользящее среднее наград')
    plt.xlabel('Эпизоды')
    plt.ylabel('Средняя награда')
    plt.title('Кривая обучения агента')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

def train_agent(board_size: int, n_episodes: int = 100_000, verbose=True):
    """Обучает агента игре в крестики-нолики"""
    env = TicTacToeEnv(board_size=board_size)
    
    # Настройка гиперпараметров
    learning_rate = 0.01
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (n_episodes / 2)
    final_epsilon = 0.1
    
    # Создание и настройка агента
    agent = QLearningAgent(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon
    )
    
    # Отслеживание прогресса обучения
    rewards_history = []
    moves_history = []
    
    # Основной цикл обучения
    progress_bar = tqdm(range(n_episodes), disable=not verbose)
    for episode in progress_bar:
        state, _ = env.reset()
        episode_reward = 0
        
        while True:
            # Выбор и выполнение действия
            action = agent.get_action(state)
            next_state, reward, done, _, info = env.step(action)
            
            # Обучение агента
            agent.update(state, action, reward, done, next_state)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # Сохранение статистики
        rewards_history.append(episode_reward)
        moves_history.append(info['move_count'])
        
        # Обновление epsilon
        agent.decay_epsilon()
        
        # Обновление прогресс-бара
        if verbose:
            progress_bar.set_description(
                f"Ср. награда: {np.mean(rewards_history[-100:]):.2f}, "
                f"Epsilon: {agent.epsilon:.3f}"
            )
    
    # Сохранение обученной модели
    agent.save_q_table()
    
    # Построение графика обучения
    if verbose:
        plot_learning_curve(rewards_history, window_size=100)
    
    return agent, rewards_history, moves_history