import matplotlib.pyplot as plt
import numpy as np

species = (
    "Ours",
    "Ours w/o CBF",
    "[36]",
    "MCTS + LQR",
    "End-to-end",
)
weight_counts = {
    "Ours": np.array([0,7,3,1,0]),
    "Ours w/o CBF": np.array([13,0,4,0,0]),
    "[36]": np.array([17,16,0,5,2]),
    "MCTS + LQR": np.array([19,20,15,0,0]),
    "End-to-end": np.array([20,20,18,19,1]),
}
width = 0.5

fig, ax = plt.subplots()
bottom = np.zeros(5)

for boolean, weight_count in weight_counts.items():
    p = ax.bar(species, weight_count, width, label=boolean, bottom=bottom)
    bottom += weight_count

# ax.set_title("No. of races won")
ax.legend(loc="upper right")
plt.ylabel('no of races won')
plt.xlabel('algorithm')
plt.savefig('race_wins.png')
plt.show()

n_steps = [0, 250000, 500000, 1000000, 1500000, 1750000, 2000000]
n_races_won_1 = [0, 5, 8, 6, 5, 6, 6]
n_races_won_2 = [0, 2, 4, 4, 4, 4, 4]
n_races_won_3 = [0, 0, 0, 2, 3, 2, 2]
plt.plot(n_steps,n_races_won_1,label='Ours')
plt.plot(n_steps,n_races_won_2,label='With model w/o CBF curriculum')
plt.plot(n_steps,n_races_won_3,label='Without curriculum')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.xlabel('steps')
plt.ylabel('no of wins')
plt.legend()
plt.savefig('n_wins_iter.png')
plt.show()