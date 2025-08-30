from typing import Tuple

import numpy as np


# Stationary bandits, the true value doesn't change
class Bandits:
    def __init__(self, number_of_bandits: int) -> None:
        self.number_of_bandits = number_of_bandits
        self.true_value = np.random.normal(size=number_of_bandits)

    def __len__(self) -> int:
        return self.number_of_bandits

    def get_reward(self, index: int) -> np.float64:
        return np.random.normal(loc=self.true_value[index])

    def get_best_reward(self) -> Tuple[int, np.float64]:
        return max(enumerate(self.true_value), key=lambda x: x[1])


# Non stationary bandits (the true value drifts)
class NSBandits(Bandits):
    def __init__(self, number_of_bandits: int, drift_scale: float = 0.01) -> None:
        super().__init__(number_of_bandits)
        self.drift_scale = drift_scale

    def drift_bandits(self) -> None:
        self.true_value += np.random.normal(
            scale=self.drift_scale, size=len(self.true_value)
        )
