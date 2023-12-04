#!/usr/bin/env python
import gym 
import safety_gym
import safe_rl
from safe_rl.utils.run_utils import setup_logger_kwargs
from safe_rl.utils.mpi_tools import mpi_fork
import math
import numpy as np
# import safe_rl.
DT = 0.002
K = 19.
v_factor = 300.
radius = 0.4
lambda_1 = .04
lambda_2 = .04

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
        
        yawh = math.atan2(yh-y,xh-x)
        h = math.sqrt((x-xh)**2+(y-yh)**2) - radius
        # print(math.sqrt((x-xh)**2+(y-yh)**2))
        yaw_diff = diff_theta(yaw+ws*dtyaw,yawh)
        h_dot = -v*np.cos(yaw_diff)
        h_dot_dot = K*(vt-v)*np.cos(yaw_diff) - v*ws*np.sin(yaw_diff) + v**2*np.sin(yaw_diff)**2/((x-xh)**2+(y-yh)**2)
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
    w_new = -30.+.6*np.argmin(costs)
    w_new_cmd = w_to_wcmd(w_new)
    ac_new = [ac[0],ac[1]]
    ac_new[1] = w_new_cmd
    return ac_new

def main(robot, task, algo, seed, exp_name, cpu):

    # Verify experiment
    robot_list = ['point', 'car', 'doggo']
    task_list = ['goal1', 'goal2', 'button1', 'button2', 'push1', 'push2']
    algo_list = ['ppo', 'ppo_lagrangian', 'trpo', 'trpo_lagrangian', 'cpo']

    algo = algo.lower()
    task = task.capitalize()
    robot = robot.capitalize()
    assert algo in algo_list, "Invalid algo"
    assert task.lower() in task_list, "Invalid task"
    assert robot.lower() in robot_list, "Invalid robot"

    # Hyperparameters
    exp_name = algo + '_' + robot + task
    if robot=='Doggo':
        num_steps = 1e8
        steps_per_epoch = 60000
    else:
        num_steps = 1e7
        steps_per_epoch = 30000
    epochs = int(num_steps / steps_per_epoch)
    save_freq = 5
    target_kl = 0.01
    cost_lim = 25

    # Fork for parallelizing
    mpi_fork(cpu)

    # Prepare Logger
    exp_name = exp_name or (algo + '_' + robot.lower() + task.lower())
    logger_kwargs = setup_logger_kwargs(exp_name, seed)

    # Algo and Env
    algo = eval('safe_rl.'+algo)
    env_name = 'Safexp-'+robot+task+'-v0'

    algo(env_fn=lambda: gym.make(env_name),
         ac_kwargs=dict(
             hidden_sizes=(256, 256),
            ),
         epochs=epochs,
         steps_per_epoch=steps_per_epoch,
         save_freq=save_freq,
         target_kl=target_kl,
         cost_lim=cost_lim,
         seed=seed,
         logger_kwargs=logger_kwargs
         )



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', type=str, default='Point')
    parser.add_argument('--task', type=str, default='Goal1')
    parser.add_argument('--algo', type=str, default='ppo')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--cpu', type=int, default=8)
    args = parser.parse_args()
    exp_name = args.exp_name if not(args.exp_name=='') else None
    main(args.robot, args.task, args.algo, args.seed, exp_name, args.cpu)