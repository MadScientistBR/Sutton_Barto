import matplotlib.pyplot as plt
import numpy as np

from src.agents import (AgentParameters, EGreedyAgent, InitializationPolicy,
                        UCBAgent, UpdatePolicy)
from src.bandits import NSBandits

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
NUMBER_OF_ITERATIONS = 2000
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
