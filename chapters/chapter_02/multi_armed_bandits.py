import random
from enum import Enum, auto
from typing import Tuple

import matplotlib.pyplot as plt
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


class UpdatePolicy(Enum):
    WEIGHTED_AVERAGE = auto()
    SAMPLE_AVERAGE = auto()


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

        if update_policy_type == UpdatePolicy.WEIGHTED_AVERAGE:
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


agent_configs = {
    "UCB (c=2)": {
        "class": UCBAgent,
        "params": {
            "update_policy_type": UpdatePolicy.WEIGHTED_AVERAGE,
            "initialization_policy_type": InitializationPolicy.REALISTIC,
            "degree_of_exploration": 2.0,
            "step_size": 0.1,
        },
    },
    "EpsilonGreedy (e=0.1)": {
        "class": EGreedyAgent,
        "params": {
            "update_policy_type": UpdatePolicy.WEIGHTED_AVERAGE,
            "initialization_policy_type": InitializationPolicy.REALISTIC,
            "epsilon": 0.1,
            "step_size": 0.1,
        },
    },
    "EpsilonGreedy (e=0.01)": {
        "class": EGreedyAgent,
        "params": {
            "update_policy_type": UpdatePolicy.WEIGHTED_AVERAGE,
            "initialization_policy_type": InitializationPolicy.REALISTIC,
            "epsilon": 0.01,
            "step_size": 0.1,
        },
    },
}


# --- Simulação ---
NUMBER_OF_ITERATIONS = 10000
NUMBER_OF_RUNS = 2000
NUMBER_OF_BANDITS = 10
DRIFT_SCALE = 0.01

# PASSO 2: Preparar a estrutura para armazenar os resultados de TODOS os agentes
all_rewards = {
    name: np.zeros((NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)) for name in agent_configs
}

# Loop externo para cada simulação/run independente
for run_index in range(NUMBER_OF_RUNS):
    print(f"Executando Run {run_index + 1}/{NUMBER_OF_RUNS}...")

    # Para cada run, criamos um novo ambiente (para uma comparação justa)
    bandits = NSBandits(number_of_bandits=NUMBER_OF_BANDITS, drift_scale=DRIFT_SCALE)

    # E criamos um novo conjunto de agentes
    agents = {
        name: config["class"](bandits=bandits, **config["params"])
        for name, config in agent_configs.items()
    }

    # Loop interno para os passos de tempo
    for time_step in range(NUMBER_OF_ITERATIONS):
        # O mundo muda uma vez por passo de tempo
        bandits.drift_bandits()

        # PASSO 3: Executar um passo para CADA agente
        for agent_name, agent in agents.items():
            chosen_index = agent.choose_action()
            true_reward = bandits.get_reward(chosen_index)
            agent.update_policy(chosen_index, true_reward)

            # Armazena a recompensa para o agente e run específicos
            all_rewards[agent_name][run_index, time_step] = true_reward

# --- Análise e Plotagem ---

# PASSO 4: Calcular a média das recompensas para cada agente e plotar
plt.figure(figsize=(12, 8))
plt.title("Comparação de Agentes em um Problema Não Estacionário")
plt.xlabel("Passos de Tempo (Time Steps)")
plt.ylabel("Recompensa Média")

for agent_name, rewards_data in all_rewards.items():
    # Calcula a média através de todas as runs (média das colunas)
    average_rewards = np.mean(rewards_data, axis=0)
    plt.plot(average_rewards, label=agent_name, linewidth=1.0)

plt.legend()
plt.grid(True)
plt.show()
