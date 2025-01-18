import numpy as np

from beta_fm import BetaFm

from typing import Callable, Iterable, Sequence, TypeVar

T = TypeVar("T")

def chunk(vals: Sequence[T], chunk_size: int) -> Iterable[Sequence[T]]:
    for idx in range(0, len(vals), chunk_size):
        yield vals[idx:idx+chunk_size]

class FullFm:
    def __init__(self, hash_funcs: list[Callable[[int], float]], alphas_per_beta: int):
        self.estimators: tuple[BetaFm, ...] = tuple(BetaFm(funcs) for funcs in chunk(hash_funcs, alphas_per_beta))
    
    def update(self, values: np.ndarray) -> 'FullFm':
        for est in self.estimators:
            est.update(values)
        return self
    
    def estimate(self) -> float:
        return float(np.median([est.estimate() for est in self.estimators]))