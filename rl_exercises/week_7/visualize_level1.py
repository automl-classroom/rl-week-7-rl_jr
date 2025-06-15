import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from rliable import metrics
from rliable.library import get_interval_estimates
from rliable.plot_utils import plot_sample_efficiency_curve

seeds = [34645,  2055337, 13220621,  7247199, 13834783]
n_seeds = len(seeds)
bin_size = 1000

dfs_rnd = list()
dfs_dqn = list()

for s in seeds:
    df = pd.read_csv(f"outputs/2025-06-09/17-28-59/training_data_RND_seed_{s}.csv")
    df['seed'] = s
    df['algo'] = 'rnd'
    dfs_rnd.append(df)

    df = pd.read_csv(f"outputs/2025-06-09/17-39-59/training_data_DQN_seed_{s}.csv")
    df['seed'] = s
    df['algo'] = 'dqn'
    dfs_dqn.append(df)

# Make sure only one set of steps is attempted to be plotted
# Obviously the steps should match in such cases!
steps = dfs_rnd[0]["bin"].unique()

# The number of logged rewards differs for each recording
# -> Aggregate rewards by bins, whose number is the same for each recording (here 40)
# use the mean to aggregate
def get_rewards_per_bin(dfs):
    return np.array([
        df.groupby("bin")["rewards"].mean().sort_index().values
        for df in dfs
    ])

train_scores = {
    "dqn_epsilon_greedy": get_rewards_per_bin(dfs_dqn),
    "dqn_rnd": get_rewards_per_bin(dfs_rnd)
}

# This aggregates only IQM, but other options include mean and median
# Optimality gap exists, but you obviously need optimal scores for that
# If you want to use it, check their code
iqm = lambda scores: np.array(  # noqa: E731
    [metrics.aggregate_iqm(scores[:, eval_idx]) for eval_idx in range(scores.shape[-1])]
)
iqm_scores, iqm_cis = get_interval_estimates(
    train_scores,
    iqm,
    reps=2000,
)

# This is a utility function, but you can also just use a normal line plot with the IQM and CI scores
plot_sample_efficiency_curve(
    (steps + 1) * 1000,
    iqm_scores,
    iqm_cis,
    algorithms=["dqn_epsilon_greedy", "dqn_rnd"],
    xlabel="Steps",
    ylabel="IQM Normalized Score",
)
plt.gcf().canvas.manager.set_window_title(
    "IQM Normalized Score - Sample Efficiency Curve"
)
plt.legend()
plt.title("IQM Normalized Score - Sample Efficiency Curve - LunarLander-v3")
plt.tight_layout()
plt.savefig("rl_exercises/week_7/comparison.pdf")
plt.show()
