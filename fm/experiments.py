from functools import cache
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm
import numpy as np

from alpha_fm import AlphaFm
from beta_fm import BetaFm
from full_fm import FullFm

from typing import Callable

@cache
def random_hash(value: int) -> float:
    return np.random.default_rng(seed=value).uniform()

def seeded_hash(hash_seed: int) -> Callable[[int], float]:
    def sub_hash(value: int) -> float:
        return random_hash(value + 1153 * hash_seed)
    return sub_hash


def run_experiment(values: np.ndarray, base_seed: int, num_hashes: int, num_betas: int) -> float:
    hash_seeds = np.arange(0, num_hashes) + base_seed * num_hashes
    hash_funcs = [seeded_hash(seed) for seed in hash_seeds]

    if num_betas == 1 and num_hashes == 1:
        fm = AlphaFm(hash_funcs[0])
    elif num_betas == 1:
        fm = BetaFm(hash_funcs)
    else:
        fm = FullFm(hash_funcs, len(hash_funcs) // num_betas)
    return fm.update(values).estimate()


def run_experiments(values: np.ndarray, base_seeds: list[int], num_hashes: int, num_betas: int, num_procs: int = 0) -> list[float]:
    if not num_procs:
        return [run_experiment(values, s, num_hashes, num_betas) for s in tqdm(base_seeds)]

    with ProcessPoolExecutor(max_workers=num_procs) as executor:
        # TODO: remove unique
        # values = np.unique(values)

        futures = [executor.submit(run_experiment, values, s, num_hashes, num_betas) for s in base_seeds]
        results =  [
            f.result() for f in tqdm(as_completed(futures), total=len(futures), desc=f"{num_hashes=}, {num_betas=}")
        ]

    return results
