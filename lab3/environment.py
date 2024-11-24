from typing import Tuple, List
import numpy as np
from gymnasium import Env, spaces

class TicTacToeEnv(Env):
    PLAYER_X = 'X'
    PLAYER_O = 'O'
    EMPTY = '.'
    
    def __init__(self, board_size: int = 3):
        self.board_size = board_size
        self.observation_space = spaces.MultiDiscrete([3] * (board_size * board_size))
        self.action_space = spaces.Discrete(board_size * board_size)
        
        # Инициализация состояния игры
        self.board = None
        self.current_player = None
        self.move_count = None
        self.last_move = None
        
        # Сброс игры в начальное состояние
        self.reset()
    
    def _create_empty_board(self) -> np.ndarray:
        """Создает пустое игровое поле"""
        return np.full((self.board_size, self.board_size), self.EMPTY)
    
    def _encode_player(self, player: str) -> int:
        """Кодирует символ игрока в числовое представление"""
        encoding = {self.PLAYER_X: 0, self.PLAYER_O: 1, self.EMPTY: 2}
        return encoding[player]
    
    def _decode_player(self, code: int) -> str:
        """Декодирует числовое представление в символ игрока"""
        decoding = {0: self.PLAYER_X, 1: self.PLAYER_O, 2: self.EMPTY}
        return decoding[code]
    
    def _get_board_state(self) -> List[int]:
        """Возвращает закодированное состояние доски"""
        return [self._encode_player(cell) for row in self.board for cell in row]
    
    def reset(self, seed=None, options=None) -> Tuple[List[int], dict]:
        super().reset(seed=seed)
        
        self.board = self._create_empty_board()
        self.current_player = self.PLAYER_X
        self.move_count = 0
        self.last_move = (-1, -1)
        
        return self._get_board_state(), self._get_game_info()
    
    def _is_winning_move(self, player: str, row: int, col: int) -> bool:
        """Проверяет, является ли ход выигрышным"""
        # Проверка горизонтали
        for c in range(self.board_size - 2):
            if all(self.board[row, c+i] == player for i in range(3)):
                return True
                
        # Проверка вертикали
        for r in range(self.board_size - 2):
            if all(self.board[r+i, col] == player for i in range(3)):
                return True
                
        # Проверка диагоналей
        for r in range(self.board_size - 2):
            for c in range(self.board_size - 2):
                # Главная диагональ
                if all(self.board[r+i, c+i] == player for i in range(3)):
                    return True
                # Побочная диагональ
                if all(self.board[r+i, c+(2-i)] == player for i in range(3)):
                    return True
        
        return False
    
    def _get_game_info(self) -> dict:
        """Формирует информацию о текущем состоянии игры"""
        return {
            'board': self.board.copy(),
            'current_player': self.current_player,
            'move_count': self.move_count,
            'last_move': self.last_move
        }
    
    def step(self, action: int) -> Tuple[List[int], float, bool, bool, dict]:
        row, col = action // self.board_size, action % self.board_size
        self.last_move = (row, col)
        
        # Проверка валидности хода
        if self.board[row, col] != self.EMPTY:
            return (
                self._get_board_state(),
                -1000.0,  # Штраф за невалидный ход
                False,
                False,
                self._get_game_info()
            )
        
        # Выполнение хода
        self.board[row, col] = self.current_player
        self.move_count += 1
        
        # Проверка на победу
        if self._is_winning_move(self.current_player, row, col):
            return (
                self._get_board_state(),
                100.0,  # Награда за победу
                True,
                False,
                self._get_game_info()
            )
        
        # Проверка на ничью
        if self.move_count == self.board_size * self.board_size:
            return (
                self._get_board_state(),
                0.0,  # Ничья
                True,
                False,
                self._get_game_info()
            )
        
        # Смена игрока
        self.current_player = self.PLAYER_O if self.current_player == self.PLAYER_X else self.PLAYER_X
        
        return (
            self._get_board_state(),
            0.0,  # Нейтральная награда за обычный ход
            False,
            False,
            self._get_game_info()
        )
    
    def render(self) -> None:
        """Отображает текущее состояние игры"""
        print(f'\nХод #{self.move_count}')
        print(f'Ходит {"первый" if self.current_player == self.PLAYER_X else "второй"} игрок')
        if self.last_move != (-1, -1):
            print(f'Последний ход: {self.last_move}')
        
        print('\nТекущее состояние поля:')
        for row in self.board:
            print(' '.join(row))
        print()

def play_random_game():
    """Демонстрация игры с случайными ходами"""
    env = TicTacToeEnv()
    done = False
    
    _, info = env.reset()
    env.render()
    
    while not done:
        action = env.action_space.sample()
        _, reward, done, _, info = env.step(action)
        env.render()
        
        if done:
            if reward > 0:
                print(f'Игрок {info["current_player"]} победил!')
            elif reward == 0:
                print('Ничья!')

if __name__ == "__main__":
    play_random_game()