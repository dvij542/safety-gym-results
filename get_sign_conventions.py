import numpy as np
import math
import safety_gym
import gym
import matplotlib.pyplot as plt
from safety_gym.envs.engine import Engine

config = {
    'robot_base': 'xmls/car.xml',
    'task': 'push',
    'observe_goal_lidar': True,
    'observe_box_lidar': True,
    'observe_hazards': True,
    'observe_vases': True,
    'constrain_hazards': True,
    'lidar_max_dist': 3,
    'lidar_num_bins': 16,
    'hazards_num': 1,
    'vases_num': 0,
    'hazards_locations' : [(3.,2.)],
    'robot_locations' : [(2.,1.)],
    'robot_rot' : 0.1
}

radius = 0.4
lambda_1 = 0.02
lambda_2 = 0.02
v_factor = 300.
K = 19.
DT = 0.002

def mat_to_yaw(mat) :
    return math.atan2(-mat[0,1],mat[0,0])

def diff_theta(t1,t2) :
    dt = t2-t1
    dt -= (dt>math.pi)*2*math.pi 
    dt += (dt<-math.pi)*2*math.pi
    return dt

def wcmd_to_w(wcmd) :
    if wcmd > 0.1 :
        return (wcmd-0.1)*(30./0.895)
    elif wcmd < -0.1 :
        return (wcmd+0.1)*(30./0.895)
    else :
        return 0.

def w_to_wcmd(w) :
    if w > 0. :
        return (w)*(0.895/30.) + 0.1
    elif w < 0. :
        return (w)*(0.895/30.) - 0.1
    else :
        return 0.

def get_cbf_constrained_action(ac,hazards,state) :
    x,y,yaw,vx,vy,vyaw = state
    # print(ac)
    vt = v_factor*min(max(ac[0],-0.05),0.05)
    ws = np.arange(-30.,30.,.6)
    # print(wcmd_to_w(ac[1]))
    costs = (ws-wcmd_to_w(ac[1]))**2/1000.
    vel_yaw = math.atan2(vy,vx)
    # print(diff_theta(vel_yaw,yaw))
    if abs(diff_theta(vel_yaw,yaw)) < math.pi/2. :
        v = math.sqrt(vx**2+vy**2)
    else :
        v = -math.sqrt(vx**2+vy**2)
        vel_yaw += math.pi
    # print(v)
    # if v < 0.:
    #     return ac
    curr_min_vals = np.array([1000.]*len(ws))
    min_cost = np.zeros(len(ws))
    dtyaw = 0.0
    for hazard in hazards :
        xh,yh,_ = hazard
        print(xh,yh)
        yawh = math.atan2(yh-y,xh-x)
        print(yawh)
        h = math.sqrt((x-xh)**2+(y-yh)**2) - radius
        # print(math.sqrt((x-xh)**2+(y-yh)**2))
        yaw_diff = diff_theta(yaw,yawh)
        print(yaw_diff)
        h_dot = -v*np.cos(yaw_diff)
        h_dot_dot = -K*(vt-v)*np.cos(yaw_diff) - v*ws*np.sin(yaw_diff) + v**2*np.sin(yaw_diff)**2/((x-xh)**2+(y-yh)**2)
        cost = (lambda_1*lambda_2*h_dot_dot + (lambda_1+lambda_2)*h_dot + h)
        # cost = np.abs(yaw_diff) - np.arcsin(0.35/max(0.36,math.sqrt((x-xh)**2+(y-yh)**2)))
        # print(math.sqrt((x-xh)**2+(y-yh)**2))
        # print(np.abs(yaw_diff),np.arcsin(0.35/max(0.36,math.sqrt((x-xh)**2+(y-yh)**2))))
        cost = (cost<0.)*cost
        # print(cost)
        vals = h #+ lambda_1*h_dot
        min_cost = (vals<curr_min_vals)*cost + (vals>=curr_min_vals)*min_cost
        curr_min_vals = (vals<curr_min_vals)*vals + (vals>=curr_min_vals)*curr_min_vals
        # print(cost)
        costs += 1e6*(cost<0.)*(cost**2)
    # print(min_cost)
    # print(costs)
    # costs += 10000*min_cost**2
    # print(np.argmin(costs))
    w_target = wcmd_to_w(ac[1])
    w_new = -30.+.6*np.argmin(costs)
    w_new_cmd = w_to_wcmd(w_new)
    ac_new = [ac[0],ac[1]]
    ac_new[1] = w_new_cmd
    return ac_new, abs(w_target-w_new)/60.

# env = Engine(config)
# env.reset()

env = gym.make('Safexp-PointGoal1-v0')
# print(env.world.robot_pos())
# pose = env.world.robot_pos()
# yaw = mat_to_yaw(env.world.robot_mat())
# print(yaw,env.robot_rot)
# a_new, penalty_ = get_cbf_constrained_action((0.,0.),env.hazards_pos,[pose[0],pose[1],yaw,0.,0.,0.])

# exit(0)


ws = np.arange(0.,1.01,0.1)
vels = [0.]*len(ws)
# vels = np.arange(-0.061,0.061,0.001)
# ws = [0.]*len(vels)
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
    for i in range(500) :
        next_observation, reward, done, info = env.step((vel,w))
        pose = env.world.robot_pos()
        x,y = pose[0] - start_pos[0], pose[1] - start_pos[1]
        theta = mat_to_yaw(env.world.robot_mat())
        # print(i,theta)
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
        next_observation, reward, done, info = env.step((vel,w))
        pose = env.world.robot_pos()
        x,y = pose[0] - start_pos[0], pose[1] - start_pos[1]
        vel_theta = math.atan2(y,x)
        theta = mat_to_yaw(env.world.robot_mat())
        theta_diffs.append(diff_theta(theta,theta_prev))
        dist = math.sqrt(x**2+y**2)
        dist_diffs.append((dist-prev_dist)/DT)
        pred_vels.append(curr_vel)
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
    # plt.plot(pred_vels)
    # print(info)
plt.plot(ws,omegas)
# plt.plot(vels,avg_vels)
plt.show()
# print(next_observation)
