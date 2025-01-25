import numpy as np

from beta_fm import BetaFm

from typing import Callable, Iterable, Sequence, TypeVar

T = TypeVar("T")

def calculate_chunk_sizes(seq_len: int, num_chunks: int) -> list[int]:
    res = np.zeros(num_chunks).astype(int)
    cur_len = num_chunks
    num_left = seq_len

    while np.sum(res) < seq_len:
        addition = num_left // cur_len
        if np.sum(res) + addition * cur_len <= seq_len:
            res[np.arange(cur_len)] += addition
            num_left -= addition * cur_len
        cur_len -= 1

    return res.tolist()

def chunk(vals: Sequence[T], num_chunks: int) -> Iterable[Sequence[T]]:
    cur_start = 0
    for chunk_size in calculate_chunk_sizes(len(vals), num_chunks):
        yield vals[cur_start:cur_start+chunk_size]
        cur_start += chunk_size

class FullFm:
    def __init__(self, hash_funcs: list[Callable[[int], float]], num_betas: int):
        self.estimators: tuple[BetaFm, ...] = tuple(BetaFm(funcs) for funcs in chunk(hash_funcs, num_betas))
    
    def update(self, values: np.ndarray) -> 'FullFm':
        for est in self.estimators:
            est.update(values)
        return self
    
    def estimate(self) -> float:
        return float(np.median([est.estimate() for est in self.estimators]))