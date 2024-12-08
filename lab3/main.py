import numpy as np
from training import train_agent

def display_training_statistics(rewards: list, moves: list) -> None:
    """Отображает финальную статистику обучения."""
    avg_reward = np.mean(rewards[-1000:])
    avg_moves = np.mean(moves[-1000:])
    
    print("\nСтатистика обучения:")
    print(f"Средняя награда за последние 1000 эпизодов: {avg_reward:.2f}")
    print(f"Среднее количество ходов за последние 1000 эпизодов: {avg_moves:.2f}")

def main() -> None:
    """Основная функция для запуска обучения агента."""
    agent, rewards, moves = train_agent(board_size=3, n_episodes=20_000, verbose=True)
    display_training_statistics(rewards, moves)

if __name__ == "__main__":
    main()