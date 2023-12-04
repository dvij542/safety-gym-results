#!/usr/bin/env python

import time
import numpy as np
from safe_rl.utils.load_utils import load_policy
from safe_rl.utils.logx import EpochLogger
import math

def mat_to_yaw(mat) :
    return math.atan2(-mat[0,1],mat[0,0])

def diff_theta(t1,t2) :
    dt = t2-t1
    dt -= (dt>math.pi)*2*math.pi 
    dt += (dt<-math.pi)*2*math.pi
    return dt

DT = 0.002
K = 19.
v_factor = 300.
radius = 0.4
lambda_1 = .01
lambda_2 = .01

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

def get_cbf_constrained_action(ac,hazards,state,prev_greenlim_poses=None) :
    # print(ac,hazards,state)
    # ac[0]/=20.
    # print(len(hazards))
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
    # if abs(v) < 2.:
    #     return ac
    curr_min_vals = np.array([1000.]*len(ws))
    min_cost = np.zeros(len(ws))
    dtyaw = 0.0
    n_greenlims = 0
    if prev_greenlim_poses :
        n_greenlims = len(prev_greenlim_poses)
    for hazard in hazards :
        xh,yh,_ = hazard
        
        yawh = math.atan2(yh-y,xh-x)
        h = math.sqrt((x-xh)**2+(y-yh)**2) - radius
        # print(math.sqrt((x-xh)**2+(y-yh)**2))
        yaw_diff = diff_theta(yaw,yawh)
        h_dot = -v*np.cos(yaw_diff)
        h_dot_dot = -K*(vt-v)*np.cos(yaw_diff) - v*ws*np.sin(yaw_diff) + v**2*np.sin(yaw_diff)**2/math.sqrt((x-xh)**2+(y-yh)**2)
        # if (h + lambda_1*h_dot)> 0. :
        cost = (lambda_1*lambda_2*h_dot_dot + (lambda_1+lambda_2)*h_dot + h)
        # else :
        #     cost = (lambda_1*lambda_2*h_dot_dot/10. + (lambda_1+lambda_2/10.)*h_dot + h)
        # cost = np.abs(yaw_diff) - np.arcsin(0.35/max(0.36,math.sqrt((x-xh)**2+(y-yh)**2)))
        # print(math.sqrt((x-xh)**2+(y-yh)**2))
        # print(np.abs(yaw_diff),np.arcsin(0.35/max(0.36,math.sqrt((x-xh)**2+(y-yh)**2))))
        cost = (cost<0.)*cost*(abs(abs(yaw_diff)-math.pi/2.)>math.pi/10.)*(abs(yaw_diff)>math.pi/30.)
        # print(cost)
        vals = h + lambda_1*h_dot
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
    
    vts = np.arange(-15.,15.,.6)
    costs = (vts-v_factor*(ac[0]))**2/1000.
    vel_yaw = math.atan2(vy,vx)
    # print(diff_theta(vel_yaw,yaw))
    if abs(diff_theta(vel_yaw,yaw)) < math.pi/2. :
        v = math.sqrt(vx**2+vy**2)
    else :
        v = -math.sqrt(vx**2+vy**2)
        vel_yaw += math.pi
    curr_min_vals = np.array([1000.]*len(ws))
    min_cost = np.zeros(len(ws))
    dtyaw = 0.0
    for hazard in hazards :
        xh,yh,_ = hazard
        
        yawh = math.atan2(yh-y,xh-x)
        h = math.sqrt((x-xh)**2+(y-yh)**2) - radius
        yaw_diff = diff_theta(yaw+w_new*dtyaw,yawh)
        h_dot = -v*np.cos(yaw_diff)
        h_dot_dot = -K*(vts-v)*np.cos(yaw_diff) - v*w_new*np.sin(yaw_diff) + v**2*np.sin(yaw_diff)**2/math.sqrt((x-xh)**2+(y-yh)**2)
        cost = (lambda_1*lambda_2*h_dot_dot + (lambda_1+lambda_2)*h_dot + h)
        # cost = np.abs(yaw_diff) - np.arcsin(0.35/max(0.36,math.sqrt((x-xh)**2+(y-yh)**2)))
        # print(math.sqrt((x-xh)**2+(y-yh)**2))
        # print(np.abs(yaw_diff),np.arcsin(0.35/max(0.36,math.sqrt((x-xh)**2+(y-yh)**2))))
        cost = (h>0.)*(cost<0.)*cost*(abs(abs(yaw_diff)-math.pi/2.)>math.pi/4.)
        # print(cost)
        vals = h + lambda_1*h_dot
        # min_cost = (vals<curr_min_vals)*cost + (vals>=curr_min_vals)*min_cost
        # curr_min_vals = (vals<curr_min_vals)*vals + (vals>=curr_min_vals)*curr_min_vals
        # print(cost)
        costs += 1e6*(cost<0.)*(cost**2)
    
    a_new = -15.+.6*np.argmin(costs)
    # ac_new[0] = a_new/v_factor
    # print(v,ac,ac_new)
    return ac_new#, abs(w_new-wcmd_to_w(ac[1]))/60. + abs(ac_new[0]-ac[0])/60.

def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True):

    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :("

    logger = EpochLogger()
    o, r, d, ep_ret, ep_cost, ep_len, n = env.reset(seed=0), 0, False, 0, 0, 0, 0
    curr_pos = env.world.robot_pos()
    prevx,prevy = curr_pos[0], curr_pos[1]
    prevyaw = mat_to_yaw(env.world.robot_mat())
    while n < num_episodes:
        if render:
            env.render()#mode='rgb_array')
            # print(a.shape)
            time.sleep(1e-3)

        a = get_action(o)
        a = np.clip(a, env.action_space.low, env.action_space.high)
        # print(env.action_space.low,env.action_space.high)
        curr_pos = env.world.robot_pos()
        x,y = curr_pos[0], curr_pos[1]
        yaw = mat_to_yaw(env.world.robot_mat())
        vx,vy,vyaw = (x-prevx)/DT, (y-prevy)/DT, (yaw-prevyaw)/DT
        prevx = x
        prevy = y
        prevyaw = yaw
        # print(vx,vy,yaw)

        # a_new = get_cbf_constrained_action(a,env.vases_pos+env.hazards_pos,[x,y,yaw,vx,vy,vyaw])
        a_new = get_cbf_constrained_action(a,env.gremlins_obj_pos+env.hazards_pos,[x,y,yaw,vx,vy,vyaw])
        # print(ep_len,x,y,yaw)
        # print(len(env.vases_pos))
        
        # print(a,a_new)
        o, r, d, info = env.step(a_new)
        # o, r, d, info = env.step(a)
        ep_ret += r
        ep_cost += info.get('cost', 0)
        ep_len += 1

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpCost=ep_cost, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpCost %.3f \t EpLen %d'%(n, ep_ret, ep_cost, ep_len))
            o, r, d, ep_ret, ep_cost, ep_len = env.reset(seed=0), 0, False, 0, 0, 0
            prevx,prevy = curr_pos[0], curr_pos[1]
            prevyaw = mat_to_yaw(env.world.robot_mat())
            n += 1

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpCost', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=100)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    args = parser.parse_args()
    env, get_action, sess = load_policy(args.fpath,
                                        args.itr if args.itr >=0 else 'last',
                                        args.deterministic)
    run_policy(env, get_action, args.len, args.episodes, not(args.norender))
