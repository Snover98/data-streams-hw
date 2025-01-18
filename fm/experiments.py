from functools import cache
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm
import numpy as np

from full_fm import FullFm

from typing import Callable

@cache
def random_hash(value: int) -> float:
    return np.random.default_rng(seed=value).uniform()

def seeded_hash(hash_seed: int) -> Callable[[int], float]:
    def sub_hash(value: int) -> float:
        return random_hash(value + hash_seed * 53)
    return sub_hash


def run_experiment(values: np.ndarray, base_seed: int, num_hashes: int, num_betas: int) -> float:
    generator = np.random.default_rng(seed=base_seed)
    hash_funcs = [seeded_hash(generator.integers(1, 1000)) for _ in range(num_hashes)]
    ams = FullFm(hash_funcs, len(hash_funcs) // num_betas)
    return float(ams.update(values).estimate())


def run_experiments(values: np.ndarray, base_seeds: list[int], num_hashes: int, num_betas: int, num_procs: int = 6) -> list[float]:
    if not num_procs:
        return [run_experiment(values, s, num_hashes, num_betas) for s in tqdm(base_seeds)]

    with ProcessPoolExecutor(max_workers=num_procs) as executor:
        futures = [executor.submit(run_experiment, values, s, num_hashes, num_betas) for s in base_seeds]
        results =  [f.result() for f in tqdm(futures)]

    return results
