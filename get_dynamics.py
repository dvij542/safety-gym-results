import numpy as np
import math
import safety_gym
import gym
import matplotlib.pyplot as plt

env = gym.make('Safexp-PointGoal1-v0')
# print(env.world.robot_pos())

def mat_to_yaw(mat) :
    return math.atan2(mat[0,1],-mat[0,0])

def diff_theta(t1,t2) :
    dt = t2-t1
    if dt>math.pi :
        dt -= 2*math.pi
    if dt<-math.pi :
        dt += 2*math.pi
    return dt


# ws = np.arange(0.,1.01,0.1)
# vels = [0.]*len(ws)
vels = np.arange(-0.061,0.061,0.001)
ws = [0.]*len(vels)
DT = 0.002
omegas = []
avg_vels = []
j = -1
K = 19.
v_factor = 300.
for w in ws:
    j += 1
    vel = vels[j]
    print(w,vel)
    env.reset()
    start_pos = env.world.robot_pos()
    for i in range(0) :
        next_observation, reward, done, info = env.step((vel,w))
        pose = env.world.robot_pos()
        x,y = pose[0] - start_pos[0], pose[1] - start_pos[1]
        theta = mat_to_yaw(env.world.robot_mat())
        # print(x,y,theta)

    start_pos = env.world.robot_pos()
    total_theta = 0.
    theta_prev = mat_to_yaw(env.world.robot_mat())
    theta_diffs = []
    dist_diffs = []
    prev_dist = 0.
    curr_vel = 0.
    pred_vels = []
    for i in range(500) :
        if i%5==0 :
            vel = -vel
        next_observation, reward, done, info = env.step((vel,w))
        pose = env.world.robot_pos()
        x,y = pose[0] - start_pos[0], pose[1] - start_pos[1]
        vel_theta = math.atan2(y,x)
        theta = mat_to_yaw(env.world.robot_mat())
        theta_diffs.append(diff_theta(theta,theta_prev))
        dist = math.sqrt(x**2+y**2)
        dist_diffs.append((dist-prev_dist)/DT)
        pred_vels.append(curr_vel)
        print(vel,curr_vel)
        curr_vel += K*DT*(vel*v_factor-curr_vel)
        prev_dist = dist
        total_theta += theta_diffs[-1]
        theta_prev = theta
        # print(theta,vel_theta)
        # print(x,y,total_theta,math.sqrt(x**2+y**2))
    
    omega = total_theta/(500*DT)
    omegas.append(omega)
    avg_vels.append(dist/(500*DT))
    # plt.plot(theta_diffs)
    # plt.plot(dist_diffs)
    plt.plot(pred_vels)
    plt.show()
    # print(info)
# plt.plot(ws,omegas)
# plt.plot(vels,avg_vels)
plt.show()
# print(next_observation)
