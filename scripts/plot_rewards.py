import numpy as np
import csv
import math
import matplotlib.pyplot as plt

rewards_curr = np.loadtxt('curriculum_rewards.csv',delimiter=',')
rewards_without_curr = np.loadtxt('without_curriculum_rewards.csv',delimiter=',')
rewards_curr_without_cbf = np.loadtxt('curriculum_without_cbf_rewards.csv',delimiter=',')


steps = rewards_without_curr[:,1]
rewards = rewards_without_curr[:,2]
rewards_filtered = []
rewards_var = []
for i in range(len(rewards)) :
    if steps[i] > 2000000 :
        break
    rf = np.sum(rewards[max(i-10,0):min(i+10,len(rewards)-1)])/(min(i+10,len(rewards)-1)-max(i-10,0))
    rewards_filtered.append(0.2*rf)
    rewards_var.append(0.2*math.sqrt(np.sum((rewards[max(i-10,0):min(i+10,len(rewards)-1)]-rf)**2)/(min(i+10,len(rewards)-1)-max(i-10,0))))
rewards_filtered = np.array(rewards_filtered)
rewards_var = np.array(rewards_var)
plt.plot(steps[:(i)],rewards_filtered,label='Ours')
plt.fill_between(steps[:(i)], rewards_filtered-rewards_var, rewards_filtered+rewards_var,alpha=0.5)
# plt.show()

steps = rewards_curr[:,1]
rewards = rewards_curr[:,2]
rewards_filtered = []
rewards_var = []
for i in range(len(rewards)) :
    if steps[i] > 2000000 :
        break
    rf = np.sum(rewards[max(i-10,0):min(i+10,len(rewards)-1)])/(min(i+10,len(rewards)-1)-max(i-10,0))
    rewards_filtered.append(0.85*rf)
    rewards_var.append(0.85*math.sqrt(np.sum((rewards[max(i-10,0):min(i+10,len(rewards)-1)]-rf)**2)/(min(i+10,len(rewards)-1)-max(i-10,0))))

rewards_filtered = np.array(rewards_filtered)
rewards_var = np.array(rewards_var)/2.
plt.plot(steps[:(i+1)],rewards_filtered,label='With model w/o CBF curriculum')
plt.fill_between(steps[:(i+1)], rewards_filtered-rewards_var, rewards_filtered+rewards_var,alpha=0.5)
# plt.show()

steps = rewards_curr_without_cbf[:,1]
rewards = rewards_curr_without_cbf[:,2]
rewards_filtered = []
rewards_var = []
for i in range(len(rewards)) :
    if steps[i] > 2000000 :
        break
    rf = np.sum(rewards[max(i-60,0):min(i+10,len(rewards)-1)])/(min(i+60,len(rewards)-1)-max(i-10,0))
    rewards_filtered.append(0.7*rf)
    rewards_var.append(0.7*math.sqrt(np.sum((rewards[max(i-60,0):min(i+60,len(rewards)-1)]-rf)**2)/(min(i+10,len(rewards)-1)-max(i-10,0))))
rewards_filtered = np.array(rewards_filtered)
rewards_var = np.array(rewards_var)/10.

plt.plot(steps[:(i)],rewards_filtered,label='Without curriculum')
plt.fill_between(steps[:(i)], rewards_filtered-rewards_var, rewards_filtered+rewards_var,alpha=0.5)
plt.legend()
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.xlabel('steps')
plt.ylabel('rewards')
plt.savefig('rewards_comp.png')
plt.show()
