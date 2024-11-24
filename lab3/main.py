import numpy as np
from training import train_agent

def main():
    # Запуск обучения
    agent, rewards, moves = train_agent(
        board_size=3,
        n_episodes=20_000,
        verbose=True
    )
    
    # Вывод финальной статистики
    print("\nСтатистика обучения:")
    print(f"Средняя награда за последние 1000 эпизодов: {np.mean(rewards[-1000:]):.2f}")
    print(f"Среднее количество ходов за последние 1000 эпизодов: {np.mean(moves[-1000:]):.2f}")

if __name__ == "__main__":
    main()