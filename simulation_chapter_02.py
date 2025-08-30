import matplotlib.pyplot as plt
import numpy as np

from src.agents import (
    EGreedyAgent,
    GradientAgent,
    InitializationPolicy,
    UCBAgent,
    UpdatePolicy,
)
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
    "Gradient (alpha=0.1)": {
        "class": GradientAgent,
        "params": {
            "initialization_policy_type": InitializationPolicy.REALISTIC,
            "step_size": 0.1,
            # opcional: "update_policy_type": UpdatePolicy.GRADIENT_BASED,
        },
    },
}


# --- Simulação ---
NUMBER_OF_ITERATIONS = 2000
NUMBER_OF_RUNS = 2000
NUMBER_OF_BANDITS = 10
DRIFT_SCALE = 0.01

# Armazenar recompensas E se escolheu a ação ótima (0/1)
all_rewards = {
    name: np.zeros((NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)) for name in agent_configs
}
all_optimal = {
    name: np.zeros((NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS), dtype=np.float32)
    for name in agent_configs
}

for run_index in range(NUMBER_OF_RUNS):
    print(f"Executando Run {run_index + 1}/{NUMBER_OF_RUNS}...")

    bandits = NSBandits(number_of_bandits=NUMBER_OF_BANDITS, drift_scale=DRIFT_SCALE)
    agents = {
        name: config["class"](bandits=bandits, **config["params"])
        for name, config in agent_configs.items()
    }

    for time_step in range(NUMBER_OF_ITERATIONS):
        # o mundo muda
        bandits.drift_bandits()

        # índice do braço ótimo NESTE passo (após o drift)
        best_index, _ = bandits.get_best_reward()

        # cada agente joga um passo
        for agent_name, agent in agents.items():
            chosen_index = agent.choose_action()
            true_reward = bandits.get_reward(chosen_index)
            agent.update_policy(chosen_index, true_reward)

            all_rewards[agent_name][run_index, time_step] = true_reward
            all_optimal[agent_name][run_index, time_step] = (
                1.0 if chosen_index == best_index else 0.0
            )

# --- Plots: recompensa média e % ação ótima ---
fig, (ax1, ax2) = plt.subplots(
    nrows=2, ncols=1, figsize=(12, 8), sharex=True, constrained_layout=True
)

# Plot 1: recompensa média por passo
for agent_name, rewards_data in all_rewards.items():
    avg_rewards = np.mean(rewards_data, axis=0)
    ax1.plot(avg_rewards, label=agent_name, linewidth=1.0)
ax1.set_title("Recompensa Média por Passo")
ax1.set_ylabel("Recompensa Média")
ax1.grid(True)
ax1.legend()

# Plot 2: % de ação ótima por passo
for agent_name, optimal_data in all_optimal.items():
    pct_optimal = np.mean(optimal_data, axis=0) * 100.0
    ax2.plot(pct_optimal, label=agent_name, linewidth=1.0)
ax2.set_title("Porcentagem de Ação Ótima por Passo (média sobre runs)")
ax2.set_xlabel("Passos de Tempo (Time Steps)")
ax2.set_ylabel("% de vezes que escolheu a ação ótima")
ax2.set_ylim(0, 100)
ax2.grid(True)
ax2.legend()

plt.show()
