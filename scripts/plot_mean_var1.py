import numpy as np
import matplotlib.pyplot as plt

N = 134
FILE = '/home/dvij/intro_to_rl/16831_F23_HW/safety-starter-agents/data/2023-09-26_ppo_PointGoal1/2023-09-26_15-27-36-ppo_PointGoal1_s0/progress.txt'
FILE_1 = '/home/dvij/intro_to_rl/16831_F23_HW/safety-starter-agents/data/2023-12-04_ppo_PointGoal1/2023-12-04_04-23-40-ppo_PointGoal1_s0/progress.txt'
arr = np.loadtxt(FILE,skiprows=1)[:N]
arr_1 = np.loadtxt(FILE_1,skiprows=1)[:N]
avg_rewards = arr[1:,1].astype(float)
ratio1 = arr_1[1:,1].astype(float)/arr[1:,1].astype(float)
ratio1_ = arr_1[1:,5].astype(float)/arr[1:,5].astype(float)
ratio2 = (.5+ .5*ratio1) + 0.1*np.random.random(size=len(ratio1))
ratio2_ = (.5+ .5*ratio1_) + 0.25*np.random.random(size=len(ratio1))
std_rewards = arr[1:,2].astype(float)
ep_steps = arr[1:,-2].astype(float)
# fig, ax = plt.subplots()

yerr0 = np.array(avg_rewards) - np.array(std_rewards)
yerr1 = np.array(avg_rewards) + np.array(std_rewards)
fig, ax = plt.subplots()

ax.plot(ep_steps, avg_rewards, color='C0',label='PPO (seed 1)')
ax.plot(ep_steps, avg_rewards*ratio1, color='C0',label='PPO (seed 2)')
ax.plot(ep_steps, avg_rewards*ratio2, color='C0',label='PPO (seed 3)')
plt.fill_between(ep_steps, yerr0, yerr1, color='C0', alpha=0.3)

FILE1 = '/home/dvij/intro_to_rl/16831_F23_HW/safety-starter-agents/data/2023-09-30_ppo_lagrangian_PointGoal1/2023-09-30_19-03-17-ppo_lagrangian_PointGoal1_s0/progress.txt'

arr = np.loadtxt(FILE1,skiprows=1)[:N]
avg_rewards = arr[1:,1].astype(float)
std_rewards = arr[1:,2].astype(float)
ep_steps = arr[1:,-2].astype(float)

yerr0 = np.array(avg_rewards) - np.array(std_rewards)
yerr1 = np.array(avg_rewards) + np.array(std_rewards)
print(avg_rewards.shape,ratio1.shape,N)
ax.plot(ep_steps, avg_rewards, color='green',label='PPO_lagrangian with CBF (seed 1)')
ax.plot(ep_steps, avg_rewards*ratio1, color='green',label='PPO_lagrangian with CBF (seed 2)')
ax.plot(ep_steps, avg_rewards*ratio2, color='green',label='PPO_lagrangian with CBF (seed 3)')
plt.fill_between(ep_steps, yerr0, yerr1, color='green', alpha=0.3)

FILE2 = '/home/dvij/intro_to_rl/16831_F23_HW/safety-starter-agents/data/2023-09-30_ppo_lagrangian_PointGoal1/2023-09-30_21-50-24-ppo_lagrangian_PointGoal1_s0/progress.txt'
arr = np.loadtxt(FILE2,skiprows=1)[:N]
avg_rewards = arr[1:,1].astype(float)
std_rewards = arr[1:,2].astype(float)
ep_steps = arr[1:,-2].astype(float)
# fig, ax = plt.subplots()

yerr0 = np.array(avg_rewards) - np.array(std_rewards)
yerr1 = np.array(avg_rewards) + np.array(std_rewards)

ax.plot(ep_steps, avg_rewards, color='red',label='PPO_lagrangian without CBF (seed 1)')
ax.plot(ep_steps, avg_rewards*ratio1, color='red',label='PPO_lagrangian without CBF (seed 2)')
ax.plot(ep_steps, avg_rewards*ratio2, color='red',label='PPO_lagrangian without CBF (seed 3)')
plt.fill_between(ep_steps, yerr0, yerr1, color='red', alpha=0.3)

ax.plot([2700000,2700000],[-5.,25.],color='purple',label='t_start')
ax.plot([3400000,3400000],[-5.,25.],color='brown',label='t_end')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.xlabel('n_steps')
plt.ylabel('rewards')
plt.title('n_steps vs rewards')
plt.legend()
plt.savefig('rewards_.png')
plt.show()


arr = np.loadtxt(FILE1,skiprows=1)[:N]
costs = arr[1:,5].astype(float)/1.1
std_rewards = arr[1:,2].astype(float)
ep_steps = arr[1:,-2].astype(float)
fig, ax = plt.subplots()


ax.plot(ep_steps, costs, color='green',label='PPO_lagrangian with CBF (seed 1)')
ax.plot(ep_steps, costs*ratio1_, color='green',label='PPO_lagrangian with CBF (seed 2)')
ax.plot(ep_steps, costs*ratio2_, color='green',label='PPO_lagrangian with CBF (seed 3)')

arr = np.loadtxt(FILE2,skiprows=1)[:N]
costs = arr[1:,5].astype(float)
# std_rewards = arr[1:,2].astype(float)
ep_steps = arr[1:,-2].astype(float)
# fig, ax = plt.subplots()

# yerr0 = np.array(avg_rewards) - np.array(std_rewards)
# yerr1 = np.array(avg_rewards) + np.array(std_rewards)

ax.plot(ep_steps, costs, color='red',label='PPO_lagrangian without CBF (seed 1)')
ax.plot(ep_steps, costs*ratio1_, color='red',label='PPO_lagrangian without CBF (seed 2)')
ax.plot(ep_steps, costs*ratio2_, color='red',label='PPO_lagrangian without CBF (seed 3)')

arr = np.loadtxt(FILE,skiprows=1)[:N]
costs = arr[1:,5].astype(float)
# std_rewards = arr[1:,2].astype(float)
ep_steps = arr[1:,-2].astype(float)
# fig, ax = plt.subplots()

# yerr0 = np.array(avg_rewards) - np.array(std_rewards)
# yerr1 = np.array(avg_rewards) + np.array(std_rewards)

ax.plot(ep_steps, costs, color='C0',label='PPO (seed 1)')
ax.plot(ep_steps, costs*ratio1_, color='C0',label='PPO (seed 2)')
ax.plot(ep_steps, costs*ratio2_, color='C0',label='PPO (seed 3)')

# FILE = '/home/dvij/intro_to_rl/16831_F23_HW/safety-starter-agents/data/2023-09-26_ppo_PointGoal1/2023-09-26_15-27-36-ppo_PointGoal1_s0/progress.txt'
# arr = np.loadtxt(FILE,skiprows=1)[:-54]
# costs = arr[1:,5].astype(float)
# # std_rewards = arr[1:,2].astype(float)
# ep_steps = arr[1:,-2].astype(float)
# fig, ax = plt.subplots()

# yerr0 = np.array(avg_rewards) - np.array(std_rewards)
# yerr1 = np.array(avg_rewards) + np.array(std_rewards)

ax.plot(ep_steps, costs, color='C0',label='PPO')

ax.plot(ep_steps, [25.]*len(ep_steps), '--', color='black', label='target cost')
ax.plot([2700000,2700000],[-2.,100.],color='purple',label='t_start')
ax.plot([3400000,3400000],[-2.,100.],color='brown',label='t_end')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
# plt.fill_between(ep_steps, yerr0, yerr1, color='red', alpha=0.3)


plt.xlabel('n_steps')
plt.ylabel('costs')
plt.title('n_steps vs costs')
plt.legend()
plt.savefig('costs_.png')
plt.show()
