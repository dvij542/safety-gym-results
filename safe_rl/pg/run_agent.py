import numpy as np
import tensorflow as tf
import gym
import time
import safe_rl.pg.trust_region as tro
from safe_rl.pg.agents import PPOAgent, TRPOAgent, CPOAgent
from safe_rl.pg.buffer import CPOBuffer
from safe_rl.pg.network import count_vars, \
                               get_vars, \
                               mlp_actor_critic,\
                               placeholders, \
                               placeholders_from_spaces
from safe_rl.pg.utils import values_as_sorted_list
from safe_rl.utils.logx import EpochLogger
from safe_rl.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from safe_rl.utils.mpi_tools import mpi_fork, proc_id, num_procs, mpi_sum
import math
import numpy as np

DT = 0.002
K = 19.
v_factor = 300.
radius = 0.4
iters_schedule = [0,590000,600000,2000000,2700000,3400000]
lambda_1_schedule = [0.02,0.02,0.02,0.02,0.02,0.01]
lambda_2_schedule = [0.02,0.02,0.02,0.02,0.02,0.01]
k_penalty = 0.#0.0022
USE_CBF = True

def mat_to_yaw(mat) :
    return math.atan2(-mat[0,1],mat[0,0])

def diff_theta(t1,t2) :
    dt = t2-t1
    dt -= (dt>math.pi)*2*math.pi 
    dt += (dt<-math.pi)*2*math.pi
    return dt

def wcmd_to_w(wcmd) :
    if wcmd > 0. :
        return (wcmd-0.)*(30./0.895)
    elif wcmd < -0. :
        return (wcmd+0.)*(30./0.895)
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
    ac_new[0] = a_new/v_factor
    # print(v,ac,ac_new)
    return ac_new, abs(w_new-wcmd_to_w(ac[1]))/60. + abs(ac_new[0]-ac[0])/60.


# Multi-purpose agent runner for policy optimization algos 
# (PPO, TRPO, their primal-dual equivalents, CPO)
def run_polopt_agent(env_fn, 
                     agent=PPOAgent(),
                     actor_critic=mlp_actor_critic, 
                     ac_kwargs=dict(), 
                     seed=0,
                     render=True,
                     # Experience collection:
                     steps_per_epoch=4000, 
                     epochs=50, 
                     max_ep_len=1000,
                     # Discount factors:
                     gamma=0.99, 
                     lam=0.97,
                     cost_gamma=0.99, 
                     cost_lam=0.97, 
                     # Policy learning:
                     ent_reg=0.,
                     # Cost constraints / penalties:
                     cost_lim=25,
                     penalty_init=1.,
                     penalty_lr=5e-2,
                     # KL divergence:
                     target_kl=0.01, 
                     # Value learning:
                     vf_lr=1e-3,
                     vf_iters=80, 
                     # Logging:
                     logger=None, 
                     logger_kwargs=dict(), 
                     save_freq=1
                     ):

    global lambda_1, lambda_2
    print("Save freq is ", save_freq)
    #=========================================================================#
    #  Prepare logger, seed, and environment in this process                  #
    #=========================================================================#

    logger = EpochLogger(**logger_kwargs) if logger is None else logger
    logger.save_config(locals())

    seed += 10000 * proc_id()
    tf.set_random_seed(seed)
    np.random.seed(seed)

    env = env_fn()

    agent.set_logger(logger)

    #=========================================================================#
    #  Create computation graph for actor and critic (not training routine)   #
    #=========================================================================#

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    # Inputs to computation graph from environment spaces
    x_ph, a_ph = placeholders_from_spaces(env.observation_space, env.action_space)

    # Inputs to computation graph for batch data
    adv_ph, cadv_ph, ret_ph, cret_ph, logp_old_ph = placeholders(*(None for _ in range(5)))

    # Inputs to computation graph for special purposes
    surr_cost_rescale_ph = tf.placeholder(tf.float32, shape=())
    cur_cost_ph = tf.placeholder(tf.float32, shape=())

    # Outputs from actor critic
    ac_outs = actor_critic(x_ph, a_ph, **ac_kwargs)
    pi, logp, logp_pi, pi_info, pi_info_phs, d_kl, ent, v, vc = ac_outs

    # Organize placeholders for zipping with data from buffer on updates
    buf_phs = [x_ph, a_ph, adv_ph, cadv_ph, ret_ph, cret_ph, logp_old_ph]
    buf_phs += values_as_sorted_list(pi_info_phs)

    # Organize symbols we have to compute at each step of acting in env
    get_action_ops = dict(pi=pi, 
                          v=v, 
                          logp_pi=logp_pi,
                          pi_info=pi_info)

    # If agent is reward penalized, it doesn't use a separate value function
    # for costs and we don't need to include it in get_action_ops; otherwise we do.
    if not(agent.reward_penalized):
        get_action_ops['vc'] = vc

    # Count variables
    var_counts = tuple(count_vars(scope) for scope in ['pi', 'vf', 'vc'])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d, \t vc: %d\n'%var_counts)

    # Make a sample estimate for entropy to use as sanity check
    approx_ent = tf.reduce_mean(-logp)


    #=========================================================================#
    #  Create replay buffer                                                   #
    #=========================================================================#

    # Obs/act shapes
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    # Experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    pi_info_shapes = {k: v.shape.as_list()[1:] for k,v in pi_info_phs.items()}
    buf = CPOBuffer(local_steps_per_epoch,
                    obs_shape, 
                    act_shape, 
                    pi_info_shapes, 
                    gamma, 
                    lam,
                    cost_gamma,
                    cost_lam)


    #=========================================================================#
    #  Create computation graph for penalty learning, if applicable           #
    #=========================================================================#

    if agent.use_penalty:
        with tf.variable_scope('penalty'):
            # param_init = np.log(penalty_init)
            param_init = np.log(max(np.exp(penalty_init)-1, 1e-8))
            penalty_param = tf.get_variable('penalty_param',
                                          initializer=float(param_init),
                                          trainable=agent.learn_penalty,
                                          dtype=tf.float32)
        # penalty = tf.exp(penalty_param)
        penalty = tf.nn.softplus(penalty_param)

    if agent.learn_penalty:
        if agent.penalty_param_loss:
            penalty_loss = -penalty_param * (cur_cost_ph - cost_lim)
        else:
            penalty_loss = -penalty * (cur_cost_ph - cost_lim)
        train_penalty = MpiAdamOptimizer(learning_rate=penalty_lr).minimize(penalty_loss)


    #=========================================================================#
    #  Create computation graph for policy learning                           #
    #=========================================================================#

    # Likelihood ratio
    ratio = tf.exp(logp - logp_old_ph)

    # Surrogate advantage / clipped surrogate advantage
    if agent.clipped_adv:
        min_adv = tf.where(adv_ph>0, 
                           (1+agent.clip_ratio)*adv_ph, 
                           (1-agent.clip_ratio)*adv_ph
                           )
        surr_adv = tf.reduce_mean(tf.minimum(ratio * adv_ph, min_adv))
    else:
        surr_adv = tf.reduce_mean(ratio * adv_ph)

    # Surrogate cost
    surr_cost = tf.reduce_mean(ratio * cadv_ph)

    # Create policy objective function, including entropy regularization
    pi_objective = surr_adv + ent_reg * ent

    # Possibly include surr_cost in pi_objective
    if agent.objective_penalized:
        pi_objective -= penalty * surr_cost
        pi_objective /= (1 + penalty)

    # Loss function for pi is negative of pi_objective
    pi_loss = -pi_objective

    # Optimizer-specific symbols
    if agent.trust_region:

        # Symbols needed for CG solver for any trust region method
        pi_params = get_vars('pi')
        flat_g = tro.flat_grad(pi_loss, pi_params)
        v_ph, hvp = tro.hessian_vector_product(d_kl, pi_params)
        if agent.damping_coeff > 0:
            hvp += agent.damping_coeff * v_ph

        # Symbols needed for CG solver for CPO only
        flat_b = tro.flat_grad(surr_cost, pi_params)

        # Symbols for getting and setting params
        get_pi_params = tro.flat_concat(pi_params)
        set_pi_params = tro.assign_params_from_flat(v_ph, pi_params)

        training_package = dict(flat_g=flat_g,
                                flat_b=flat_b,
                                v_ph=v_ph,
                                hvp=hvp,
                                get_pi_params=get_pi_params,
                                set_pi_params=set_pi_params)

    elif agent.first_order:

        # Optimizer for first-order policy optimization
        train_pi = MpiAdamOptimizer(learning_rate=agent.pi_lr).minimize(pi_loss)

        # Prepare training package for agent
        training_package = dict(train_pi=train_pi)

    else:
        raise NotImplementedError

    # Provide training package to agent
    training_package.update(dict(pi_loss=pi_loss, 
                                 surr_cost=surr_cost,
                                 d_kl=d_kl, 
                                 target_kl=target_kl,
                                 cost_lim=cost_lim))
    agent.prepare_update(training_package)

    #=========================================================================#
    #  Create computation graph for value learning                            #
    #=========================================================================#

    # Value losses
    v_loss = tf.reduce_mean((ret_ph - v)**2)
    vc_loss = tf.reduce_mean((cret_ph - vc)**2)

    # If agent uses penalty directly in reward function, don't train a separate
    # value function for predicting cost returns. (Only use one vf for r - p*c.)
    if agent.reward_penalized:
        total_value_loss = v_loss
    else:
        total_value_loss = v_loss + vc_loss

    # Optimizer for value learning
    train_vf = MpiAdamOptimizer(learning_rate=vf_lr).minimize(total_value_loss)


    #=========================================================================#
    #  Create session, sync across procs, and set up saver                    #
    #=========================================================================#

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Sync params across processes
    sess.run(sync_all_params())

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph}, outputs={'pi': pi, 'v': v, 'vc': vc})


    #=========================================================================#
    #  Provide session to agent                                               #
    #=========================================================================#
    agent.prepare_session(sess)


    #=========================================================================#
    #  Create function for running update (called at end of each epoch)       #
    #=========================================================================#

    def update():

        cur_cost = logger.get_stats('EpCost')[0]
        c = cur_cost - cost_lim
        if c > 0 and agent.cares_about_cost:
            logger.log('Warning! Safety constraint is already violated.', 'red')

        #=====================================================================#
        #  Prepare feed dict                                                  #
        #=====================================================================#

        inputs = {k:v for k,v in zip(buf_phs, buf.get())}
        inputs[surr_cost_rescale_ph] = logger.get_stats('EpLen')[0]
        inputs[cur_cost_ph] = cur_cost

        #=====================================================================#
        #  Make some measurements before updating                             #
        #=====================================================================#

        measures = dict(LossPi=pi_loss,
                        SurrCost=surr_cost,
                        LossV=v_loss,
                        Entropy=ent)
        if not(agent.reward_penalized):
            measures['LossVC'] = vc_loss
        if agent.use_penalty:
            measures['Penalty'] = penalty

        pre_update_measures = sess.run(measures, feed_dict=inputs)
        logger.store(**pre_update_measures)

        #=====================================================================#
        #  Update penalty if learning penalty                                 #
        #=====================================================================#
        if agent.learn_penalty:
            sess.run(train_penalty, feed_dict={cur_cost_ph: cur_cost})

        #=====================================================================#
        #  Update policy                                                      #
        #=====================================================================#
        agent.update_pi(inputs)

        #=====================================================================#
        #  Update value function                                              #
        #=====================================================================#
        for _ in range(vf_iters):
            sess.run(train_vf, feed_dict=inputs)

        #=====================================================================#
        #  Make some measurements after updating                              #
        #=====================================================================#

        del measures['Entropy']
        measures['KL'] = d_kl

        post_update_measures = sess.run(measures, feed_dict=inputs)
        deltas = dict()
        for k in post_update_measures:
            if k in pre_update_measures:
                deltas['Delta'+k] = post_update_measures[k] - pre_update_measures[k]
        logger.store(KL=post_update_measures['KL'], **deltas)




    #=========================================================================#
    #  Run main environment interaction loop                                  #
    #=========================================================================#

    start_time = time.time()
    o, r, d, c, ep_ret, ep_cost, ep_len = env.reset(), 0, False, 0, 0, 0, 0
    curr_pos = env.world.robot_pos()
    prevx,prevy = curr_pos[0], curr_pos[1]
    prevyaw = mat_to_yaw(env.world.robot_mat())
    cur_penalty = 0
    cum_cost = 0
    n_env_interactions = 0
    prev_gremlins_obj_pos = env.gremlins_obj_pos
    for epoch in range(epochs):

        if agent.use_penalty:
            cur_penalty = sess.run(penalty)

        for t in range(local_steps_per_epoch):
            # print("Save freq is ", save_freq)

            n_env_interactions += 8
            if n_env_interactions <= iters_schedule[0] :
                lambda_1 = lambda_1_schedule[0]
                lambda_2 = lambda_2_schedule[0]
            elif n_env_interactions >= iters_schedule[-1] :
                lambda_1 = lambda_1_schedule[-1]
                lambda_2 = lambda_2_schedule[-1]
            else :
                k = 0
                while iters_schedule[k] <= n_env_interactions :
                    k += 1
                k -= 1
                lambda_1 = (lambda_1_schedule[k+1]*(n_env_interactions-iters_schedule[k]) \
                        + lambda_1_schedule[k]*(iters_schedule[k+1]-n_env_interactions)) \
                        /(iters_schedule[k+1]-iters_schedule[k])
                lambda_2 = (lambda_2_schedule[k+1]*(n_env_interactions-iters_schedule[k]) \
                        + lambda_2_schedule[k]*(iters_schedule[k+1]-n_env_interactions)) \
                        /(iters_schedule[k+1]-iters_schedule[k])
            
            if t%1000==0 :
                print(lambda_1,lambda_2,ep_cost)
            # Possibly render
            # if render and proc_id()==0 and t < 1000:
            #     # print(t)
            #     env.render()
            
            # Get outputs from policy
            get_action_outs = sess.run(get_action_ops, 
                                       feed_dict={x_ph: o[np.newaxis]})
            a = get_action_outs['pi']
            # a[0]/=20.
            v_t = get_action_outs['v']
            vc_t = get_action_outs.get('vc', 0)  # Agent may not use cost value func
            logp_t = get_action_outs['logp_pi']
            pi_info_t = get_action_outs['pi_info']

            curr_pos = env.world.robot_pos()
            x,y = curr_pos[0], curr_pos[1]
            yaw = mat_to_yaw(env.world.robot_mat())
            vx,vy,vyaw = (x-prevx)/DT, (y-prevy)/DT, (yaw-prevyaw)/DT
            prevx = x
            prevy = y
            prevyaw = yaw
            # print(vx,vy,yaw)
            # print(a)
            # a[0,0] /= 20.
            # print(a)
            # if proc_id()==0 :
            #     print((env.gremlins_obj_pos))
            a_new, penalty_ = get_cbf_constrained_action(a[0],env.gremlins_obj_pos+env.hazards_pos,[x,y,yaw,vx,vy,vyaw],prev_gremlins_obj_pos)
            prev_gremlins_obj_pos = env.gremlins_obj_pos
            # print(penalty)
            # Step in environment
            if USE_CBF :
                # print("passing cbf command")
                o2, r, d, info = env.step(a_new)
                # print(cur_penalty)
                # cur_penalty += k_penalty*(penalty)
                # print(cur_penalty)
            else :
                o2, r, d, info = env.step(a)
            

            # Include penalty on cost
            c = info.get('cost', 0)
            # print("h ",c)
            # c += k_penalty*(penalty_)
            # print("h ",c)
            # Track cumulative cost over training
            cum_cost += c

            # save and log
            if agent.reward_penalized:
                print("reward is penalized")
                r_total = r - cur_penalty * c #- k_penalty*(penalty_)
                r_total = r_total / (1 + cur_penalty)
                buf.store(o, a, r_total, v_t, 0, 0, logp_t, pi_info_t)
            else:
                buf.store(o, a, r - k_penalty*(penalty_), v_t, c, vc_t, logp_t, pi_info_t)
            logger.store(VVals=v_t, CostVVals=vc_t)

            o = o2
            ep_ret += r
            ep_cost += c
            ep_len += 1

            terminal = d or (ep_len == max_ep_len)
            if terminal or (t==local_steps_per_epoch-1):

                # If trajectory didn't reach terminal state, bootstrap value target(s)
                if d and not(ep_len == max_ep_len):
                    # Note: we do not count env time out as true terminal state
                    last_val, last_cval = 0, 0
                else:
                    feed_dict={x_ph: o[np.newaxis]}
                    if agent.reward_penalized:
                        last_val = sess.run(v, feed_dict=feed_dict)
                        last_cval = 0
                    else:
                        last_val, last_cval = sess.run([v, vc], feed_dict=feed_dict)
                buf.finish_path(last_val, last_cval)

                # Only save EpRet / EpLen if trajectory finished
                if terminal:
                    print("Terminal reached ", ep_cost)
                    logger.store(EpRet=ep_ret, EpLen=ep_len, EpCost=ep_cost)
                else:
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len)

                # Reset environment
                o, r, d, c, ep_ret, ep_len, ep_cost = env.reset(), 0, False, 0, 0, 0, 0
                prev_gremlins_obj_pos = env.gremlins_obj_pos
                curr_pos = env.world.robot_pos()
                prevx,prevy = curr_pos[0], curr_pos[1]
                prevyaw = mat_to_yaw(env.world.robot_mat())
    
        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)

        #=====================================================================#
        #  Run RL update                                                      #
        #=====================================================================#
        update()

        #=====================================================================#
        #  Cumulative cost calculations                                       #
        #=====================================================================#
        cumulative_cost = mpi_sum(cum_cost)
        cost_rate = cumulative_cost / ((epoch+1)*steps_per_epoch)

        #=====================================================================#
        #  Log performance and stats                                          #
        #=====================================================================#

        logger.log_tabular('Epoch', epoch)

        # Performance stats
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpCost', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('CumulativeCost', cumulative_cost)
        logger.log_tabular('CostRate', cost_rate)

        # Value function values
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('CostVVals', with_min_and_max=True)

        # Pi loss and change
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)

        # Surr cost and change
        logger.log_tabular('SurrCost', average_only=True)
        logger.log_tabular('DeltaSurrCost', average_only=True)

        # V loss and change
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)

        # Vc loss and change, if applicable (reward_penalized agents don't use vc)
        if not(agent.reward_penalized):
            logger.log_tabular('LossVC', average_only=True)
            logger.log_tabular('DeltaLossVC', average_only=True)

        if agent.use_penalty or agent.save_penalty:
            logger.log_tabular('Penalty', average_only=True)
            logger.log_tabular('DeltaPenalty', average_only=True)
        else:
            logger.log_tabular('Penalty', 0)
            logger.log_tabular('DeltaPenalty', 0)

        # Anything from the agent?
        agent.log()

        # Policy stats
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)

        # Time and steps elapsed
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('Time', time.time()-start_time)
        # Show results!
        logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, default='ppo')
    parser.add_argument('--env', type=str, default='Safexp-PointGoal1-v0')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--cost_gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=4)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--len', type=int, default=1000)
    parser.add_argument('--cost_lim', type=float, default=10)
    parser.add_argument('--exp_name', type=str, default='runagent')
    parser.add_argument('--kl', type=float, default=0.01)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--reward_penalized', action='store_true')
    parser.add_argument('--objective_penalized', action='store_true')
    parser.add_argument('--learn_penalty', action='store_true')
    parser.add_argument('--penalty_param_loss', action='store_true')
    parser.add_argument('--entreg', type=float, default=0.)
    args = parser.parse_args()

    try:
        import safety_gym
    except:
        print('Make sure to install Safety Gym to use constrained RL environments.')

    mpi_fork(args.cpu)  # run parallel code with mpi

    # Prepare logger
    from safe_rl.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    # Prepare agent
    agent_kwargs = dict(reward_penalized=args.reward_penalized,
                        objective_penalized=args.objective_penalized,
                        learn_penalty=args.learn_penalty,
                        penalty_param_loss=args.penalty_param_loss)
    if args.agent=='ppo':
        agent = PPOAgent(**agent_kwargs)
    elif args.agent=='trpo':
        agent = TRPOAgent(**agent_kwargs)
    elif args.agent=='cpo':
        agent = CPOAgent(**agent_kwargs)

    run_polopt_agent(lambda : gym.make(args.env),
                     agent=agent,
                     actor_critic=mlp_actor_critic,
                     ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
                     seed=args.seed, 
                     render=args.render, 
                     # Experience collection:
                     steps_per_epoch=args.steps, 
                     epochs=args.epochs,
                     max_ep_len=args.len,
                     # Discount factors:
                     gamma=args.gamma,
                     cost_gamma=args.cost_gamma,
                     # Policy learning:
                     ent_reg=args.entreg,
                     # KL Divergence:
                     target_kl=args.kl,
                     cost_lim=args.cost_lim, 
                     # Logging:
                     logger_kwargs=logger_kwargs,
                     save_freq=1
                     )