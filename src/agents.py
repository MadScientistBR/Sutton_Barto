import random
from enum import Enum, auto
from typing import Tuple

import numpy as np

from .bandits import Bandits


class UpdatePolicy(Enum):
    WEIGHTED_AVERAGE = auto()
    SAMPLE_AVERAGE = auto()
    GRADIENT_BASED = auto()


class InitializationPolicy(Enum):
    REALISTIC = auto()
    OPTIMISTIC = auto()


class AgentParameters(Enum):
    OPTIMISTIC_ESTIMATIVE = 5.0
    STEP_SIZE = 0.10
    DEGREE_OF_EXPLORATION = 0.20


class Agent:
    def __init__(
        self,
        bandits: Bandits,
        update_policy_type: UpdatePolicy,
        initialization_policy_type: InitializationPolicy,
        step_size: float = AgentParameters.STEP_SIZE.value,
    ) -> None:
        self.environment = bandits
        self.update_policy_type = update_policy_type
        self.initialization_policy_type = initialization_policy_type

        if initialization_policy_type == InitializationPolicy.OPTIMISTIC:
            self.estimative = np.full(
                len(bandits), AgentParameters.OPTIMISTIC_ESTIMATIVE.value, dtype=float
            )

        if initialization_policy_type == InitializationPolicy.REALISTIC:
            self.estimative = np.random.normal(size=len(self.environment))

        if (
            update_policy_type == UpdatePolicy.WEIGHTED_AVERAGE
            or update_policy_type == UpdatePolicy.GRADIENT_BASED
        ):
            self.step_size = step_size

        self.number_of_tries = [0 for _ in range(len(bandits))]

    def update_policy(self, choosen_index: int, true_reward: float) -> None:
        self.number_of_tries[choosen_index] += 1

        old_estimative = self.estimative[choosen_index]

        if self.update_policy_type == UpdatePolicy.SAMPLE_AVERAGE:
            self.estimative[choosen_index] = old_estimative + (
                1 / self.number_of_tries[choosen_index]
            ) * (true_reward - old_estimative)

        if self.update_policy_type == UpdatePolicy.WEIGHTED_AVERAGE:
            old_estimative = self.estimative[choosen_index]
            self.estimative[choosen_index] = old_estimative + self.step_size * (
                true_reward - old_estimative
            )

    def _pick_best_action(self) -> Tuple[int, float]:
        return max(enumerate(self.estimative), key=lambda x: x[1])

    def choose_action(self) -> int:
        index, _ = self._pick_best_action()
        return index


class EGreedyAgent(Agent):
    def __init__(
        self,
        bandits: Bandits,
        update_policy_type: UpdatePolicy,
        initialization_policy_type: InitializationPolicy,
        epsilon: float = 0.1,
        step_size: float = AgentParameters.STEP_SIZE.value,
    ) -> None:
        super().__init__(
            bandits, update_policy_type, initialization_policy_type, step_size
        )
        self.epsilon = epsilon

    def _is_random_action(self) -> bool:
        return random.random() <= self.epsilon

    def choose_action(self) -> int:
        if self._is_random_action():
            index = random.randrange(len(self.environment))
        else:
            index, _ = self._pick_best_action()

        return index


class UCBAgent(Agent):
    def __init__(
        self,
        bandits: Bandits,
        update_policy_type: UpdatePolicy,
        initialization_policy_type: InitializationPolicy,
        step_size: float = AgentParameters.STEP_SIZE.value,
        degree_of_exploration: float = AgentParameters.DEGREE_OF_EXPLORATION.value,
    ) -> None:
        super().__init__(
            bandits, update_policy_type, initialization_policy_type, step_size
        )

        self.c = degree_of_exploration
        self.total_plays = 0

    def choose_action(self) -> int:
        self.total_plays += 1

        for index in range(len(self.number_of_tries)):
            if self.number_of_tries[index] == 0:
                return index

        ucb_estimates = self.estimative + self.c * np.sqrt(
            np.log(self.total_plays) / np.array(self.number_of_tries)
        )
        return int(np.argmax(ucb_estimates))


class GradientAgent(Agent):
    def __init__(
        self,
        bandits: Bandits,
        initialization_policy_type: InitializationPolicy,
        step_size: float = AgentParameters.STEP_SIZE.value,
        update_policy_type: UpdatePolicy = UpdatePolicy.GRADIENT_BASED,
    ) -> None:
        super().__init__(
            bandits, update_policy_type, initialization_policy_type, step_size
        )
        self.average_reward = 0.0
        self.total_plays = 0
        self._calculate_action_probabilities()

    def _calculate_action_probabilities(self) -> None:
        self.probabilities = np.exp(self.estimative) / np.sum(np.exp(self.estimative))

    def choose_action(self) -> int:
        self.total_plays += 1

        self._calculate_action_probabilities()
        index = np.random.choice(len(self.probabilities), p=self.probabilities)

        return index

    def update_policy(self, choosen_index: int, true_reward: float) -> None:
        self.number_of_tries[choosen_index] += 1

        self.average_reward += (true_reward - self.average_reward) / self.total_plays

        advantage = true_reward - self.average_reward

        for i in range(len(self.estimative)):
            if i == choosen_index:
                self.estimative[i] += (
                    self.step_size * advantage * (1.0 - self.probabilities[i])
                )
            else:
                self.estimative[i] -= self.step_size * advantage * self.probabilities[i]
