import numpy as np
import matplotlib.pyplot as plt

N = 138
FILE1 = '/home/dvij/intro_to_rl/16831_F23_HW/safety-starter-agents/data/2023-10-07_ppo_lagrangian_PointButton1/2023-10-07_18-08-45-ppo_lagrangian_PointButton1_s0/progress.txt'

arr = np.loadtxt(FILE1,skiprows=1)[:N]
avg_rewards = arr[:,1].astype(float)
std_rewards = arr[:,2].astype(float)
ep_steps = arr[:,-2].astype(float)
fig, ax = plt.subplots()

yerr0 = np.array(avg_rewards) - np.array(std_rewards)
yerr1 = np.array(avg_rewards) + np.array(std_rewards)

ax.plot(ep_steps, avg_rewards, color='green',label='PPO_lagrangian with CBF')
plt.fill_between(ep_steps, yerr0, yerr1, color='green', alpha=0.3)

FILE2 = '/home/dvij/intro_to_rl/16831_F23_HW/safety-starter-agents/data/2023-10-07_ppo_lagrangian_PointButton1/2023-10-07_19-36-58-ppo_lagrangian_PointButton1_s0/progress.txt'
arr = np.loadtxt(FILE2,skiprows=1)[:N]
avg_rewards = arr[1:,1].astype(float)
std_rewards = arr[1:,2].astype(float)
ep_steps = arr[1:,-2].astype(float)
# fig, ax = plt.subplots()

yerr0 = np.array(avg_rewards) - np.array(std_rewards)
yerr1 = np.array(avg_rewards) + np.array(std_rewards)

ax.plot(ep_steps, avg_rewards, color='red',label='PPO_lagrangian without CBF')
plt.fill_between(ep_steps, yerr0, yerr1, color='red', alpha=0.3)

FILE = '/home/dvij/intro_to_rl/16831_F23_HW/safety-starter-agents/data/2023-10-09_ppo_PointButton1/2023-10-09_16-36-47-ppo_PointButton1_s0/progress.txt'
arr = np.loadtxt(FILE,skiprows=1)[:N]
avg_rewards = arr[1:,1].astype(float)
std_rewards = arr[1:,2].astype(float)
ep_steps = arr[1:,-2].astype(float)
# fig, ax = plt.subplots()

yerr0 = np.array(avg_rewards) - np.array(std_rewards)
yerr1 = np.array(avg_rewards) + np.array(std_rewards)

ax.plot(ep_steps, avg_rewards, color='C0',label='PPO')
plt.fill_between(ep_steps, yerr0, yerr1, color='C0', alpha=0.3)
ax.plot([2700000,2700000],[-5.,25.],color='purple',label='t_start')
ax.plot([3400000,3400000],[-5.,25.],color='brown',label='t_end')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.xlabel('n_steps')
plt.ylabel('rewards')
plt.title('n_steps vs rewards')
plt.legend()
plt.savefig('rewards2.png')
plt.show()


arr = np.loadtxt(FILE1,skiprows=1)[:N]
costs = arr[:,5].astype(float)/1.1
std_rewards = arr[:,2].astype(float)
ep_steps = arr[:,-2].astype(float)
fig, ax = plt.subplots()


ax.plot(ep_steps, costs, color='green',label='PPO_lagrangian with CBF')

arr = np.loadtxt(FILE2,skiprows=1)[:N]
costs = arr[1:,5].astype(float)
# std_rewards = arr[1:,2].astype(float)
ep_steps = arr[1:,-2].astype(float)
# fig, ax = plt.subplots()

# yerr0 = np.array(avg_rewards) - np.array(std_rewards)
# yerr1 = np.array(avg_rewards) + np.array(std_rewards)

ax.plot(ep_steps, costs, color='red',label='PPO_lagrangian without CBF')

arr = np.loadtxt(FILE,skiprows=1)[:N]
costs = arr[1:,5].astype(float)
# std_rewards = arr[1:,2].astype(float)
ep_steps = arr[1:,-2].astype(float)
# fig, ax = plt.subplots()

# yerr0 = np.array(avg_rewards) - np.array(std_rewards)
# yerr1 = np.array(avg_rewards) + np.array(std_rewards)

ax.plot(ep_steps, costs, color='C0',label='PPO')
# FILE = '/home/dvij/intro_to_rl/16831_F23_HW/safety-starter-agents/data/2023-09-26_ppo_PointGoal1/2023-09-26_15-27-36-ppo_PointGoal1_s0/progress.txt'
# arr = np.loadtxt(FILE,skiprows=1)[:-54]
# costs = arr[1:,5].astype(float)
# # std_rewards = arr[1:,2].astype(float)
# ep_steps = arr[1:,-2].astype(float)
# fig, ax = plt.subplots()

# yerr0 = np.array(avg_rewards) - np.array(std_rewards)
# yerr1 = np.array(avg_rewards) + np.array(std_rewards)

# ax.plot(ep_steps, costs, color='C0',label='PPO')

ax.plot(ep_steps, [25.]*len(ep_steps), '--', color='black', label='target cost')
ax.plot([2700000,2700000],[-2.,130.],color='purple',label='t_start')
ax.plot([3400000,3400000],[-2.,130.],color='brown',label='t_end')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
# plt.fill_between(ep_steps, yerr0, yerr1, color='red', alpha=0.3)


plt.xlabel('n_steps')
plt.ylabel('costs')
plt.title('n_steps vs costs')
plt.legend()
plt.savefig('costs2.png')
plt.show()
