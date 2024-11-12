import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random


def generate_candidate_dict(num_candidates: int, seed: int = None) -> dict:
    """Generate a dictionary of candidates with unique qualities and their ranks.

    :param num_candidates: Number of candidates
    :param seed: Optional seed for random number generation
    :return: Dictionary of candidates in the format {rank: quality}
    """
    random.seed(seed)
    random_candidates = set(random.randint(10 ** 3, 10 ** 4 - 1) for _ in range(num_candidates))
    while len(random_candidates) < num_candidates:
        random_candidates.add(random.randint(10 ** 3, 10 ** 4 - 1))
    sorted_candidates = sorted(random_candidates, reverse=True)
    candidates = {rank: quality for rank, quality in zip(range(1, num_candidates + 1), sorted_candidates)}
    shuffled_ranks = list(candidates.keys())
    random.shuffle(shuffled_ranks)
    shuffled_candidates = {rank: candidates[rank] for rank in shuffled_ranks}
    return shuffled_candidates


def find_key_by_value(dictionary: dict, value):
    """Find the key in a dictionary by its value."""
    for key, val in dictionary.items():
        if val == value:
            return key
    return -1


def select_candidate(candidates):
    """Select the best candidate from the remaining candidates."""
    stop_index = int(round(len(candidates) / 2.71828))
    best_from_rejected = max(candidates[:stop_index])
    rest_candidates = candidates[stop_index:]

    for candidate in rest_candidates:
        if candidate > best_from_rejected:
            return candidate

    return candidates[-1]  # Return the last candidate if none is better


def math_solution(num_candidates: int, n_episodes: int = 100_00):
    """
    Simulate the candidate selection process.

    :param num_candidates: Number of candidates
    :param n_episodes: Number of episodes
    :return: List of ranks of the chosen candidates
    """
    ranks_of_chosen = []

    for _ in tqdm(range(n_episodes)):
        candidates_dict = generate_candidate_dict(num_candidates=num_candidates)
        candidates = list(candidates_dict.values())

        chosen_candidate = select_candidate(candidates)

        chosen_rank = find_key_by_value(candidates_dict, chosen_candidate)
        ranks_of_chosen.append(chosen_rank)

    return ranks_of_chosen


num_candidates = 100
math_solution_results = math_solution(num_candidates=num_candidates)
bins = list(range(0, num_candidates + 1, 10))
plt.figure(figsize=(10, 6))
plt.hist(math_solution_results, bins=num_candidates, alpha=0.5, label='reinforcement learning')
plt.xlabel('Chosen candidate')
plt.ylabel('frequency')
plt.title('Math Solution')
plt.show()