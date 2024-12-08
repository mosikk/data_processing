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
        
        self.reset()
    
    def _create_empty_board(self) -> np.ndarray:
        """Создает пустое игровое поле"""
        return np.full((self.board_size, self.board_size), self.EMPTY)
    
    def _encode_player(self, player: str) -> int:
        """Кодирует символ игрока в числовое представление"""
        return {'X': 0, 'O': 1, '.': 2}[player]
    
    def _decode_player(self, code: int) -> str:
        """Декодирует числовое представление в символ игрока"""
        return {0: self.PLAYER_X, 1: self.PLAYER_O, 2: self.EMPTY}[code]
    
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
        return (self._check_line(player, row, 0, 0, 1) or  # Горизонталь
                self._check_line(player, 0, col, 1, 0) or  # Вертикаль
                self._check_line(player, 0, 0, 1, 1) or    # Главная диагональ
                self._check_line(player, 0, 2, 1, -1))     # Побочная диагональ
    
    def _check_line(self, player: str, start_row: int, start_col: int, delta_row: int, delta_col: int) -> bool:
        """Проверяет линию на выигрыш"""
        return all(
            self.board[start_row + i * delta_row, start_col + i * delta_col] == player
            for i in range(3)
        )
    
    def _get_game_info(self) -> dict:
        """Формирует информацию о текущем состоянии игры"""
        return {
            'board': self.board.copy(),
            'current_player': self.current_player,
            'move_count': self.move_count,
            'last_move': self.last_move
        }
    
    def step(self, action: int) -> Tuple[List[int], float, bool, bool, dict]:
        row, col = divmod(action, self.board_size)
        self.last_move = (row, col)
        
        # Проверка валидности хода
        if self.board[row, col] != self.EMPTY:
            return self._get_board_state(), -1000.0, False, False, self._get_game_info()
        
        # Выполнение хода
        self.board[row, col] = self.current_player
        self.move_count += 1
        
        # Проверка на победу или ничью
        if self._is_winning_move(self.current_player, row, col):
            return self._get_board_state(), 100.0, True, False, self._get_game_info()
        
        if self.move_count == self.board_size * self.board_size:
            return self._get_board_state(), 0.0, True, False, self._get_game_info()
        
        # Смена игрока
        self.current_player = self.PLAYER_O if self.current_player == self.PLAYER_X else self.PLAYER_X
        
        return self._get_board_state(), 0.0, False, False, self._get_game_info()
    
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