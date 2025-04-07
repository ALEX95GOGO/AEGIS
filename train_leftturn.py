import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io as scio
import os
import pygame
import signal
import argparse

import torch
from torch.utils.tensorboard import SummaryWriter   
writer = SummaryWriter('./algo/checkpoints/log')
from human_model import Actor as h_Model
from env_leftturn import LeftTurn
from utils import set_seed, signal_handler
import matplotlib
matplotlib.use('TkAgg')  # Replace 'Agg' with your desired backend

import matplotlib.pyplot as plt
from algo.TD3_leftturn import DRL

def train_leftturn_task():
    
    set_seed(args.seed)
    env = LeftTurn(joystick_enabled = args.joystick_enabled, conservative_surrounding = args.simulator_conservative_surrounding, 
                   frame=args.simulator_render_frequency, port=args.simulator_port)

    s_dim = [env.observation_size_width, env.observation_size_height]
    a_dim = env.action_size
    
    DRL = DRL(a_dim, s_dim, device=args.device)
    
    exploration_rate = args.initial_exploration_rate 
    
    if args.resume and os.path.exists(log_dir):
        checkpoint = torch.load(log_dir)
        DRL.load(log_dir)
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0
    
    
    if args.human_model:
        from algo.SL import SL
        SL = SL(a_dim, s_dim)
        SL.load_model('./algo/models_leftturn/SL.pkl')
    
    if args.bc:
        #h_model = h_Model(nb_states=3, nb_actions=1)
        h_model = h_Model()
        h_model.cuda()
        if os.path.isfile('algo/checkpoints/human_model_epoch_5.tar'):
            print("=> loading checkpoint '{}'".format('algo/checkpoints/human_model_epoch_5.tar'))
            checkpoint = torch.load('algo/checkpoints/human_model_epoch_5.tar')
            #args.start_epoch = checkpoint['epoch']
            start_epoch = 1
            h_model.load_state_dict(checkpoint['state_dict'])
            #DRL.actor.load_state_dict(torch.load(args.bc)['state_dict'])
            DRL.actor.load_state_dict(torch.load('algo/checkpoints/human_model_epoch_5.tar')['state_dict'])
            #optimizer.load_state_dict(checkpoint['optim_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.bc, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format('algo/checkpoints/human_model_epoch_5.tar'))
    # initialize global variables
    total_step = 0
    a_loss,c_loss, dis_loss, kl_loss = 0,0,0,0
    q_target, cross_correlation, kl_ours, nss,mae, td_errors, sim = 0,0.,0,0,0,0,0
    count_sl = 0
    
    loss_critic, loss_actor = [], []
    episode_total_reward_list, episode_mean_reward_list = [], []
    global_reward_list, episode_duration_list = [], []
    
    drl_action = [[] for i in range(args.maximum_episode)] # drl_action: output by RL policy
    adopted_action = [[] for i in range(args.maximum_episode)]  # adopted_action: eventual action which may include human guidance action
     
    reward_i_record = [[] for i in range(args.maximum_episode)]  # reward_i: virtual reward (added shaping term);
    reward_e_record = [[] for i in range(args.maximum_episode)]  # reward_e: real reward
    
    
    start_time = time.perf_counter()
    
    for i in range(start_epoch, args.maximum_episode):
        
        # initial episode variables
        ep_reward = 0      
        step = 0
        step_intervened = 0
        done = False
        human_model_activated = False

        # initial environment
        observation = env.reset()
        state = np.repeat(np.expand_dims(observation,2), 3, axis=2)
        state_ = state.copy()
    
        while not done:
            ## Section DRL's actting ##
            #import pdb; pdb.set_trace()
            action, risk_pred = DRL.choose_action(state)
            action = np.clip( np.random.normal(action,  exploration_rate), -1, 1)
            
            if i < 8:
                action = np.array([0.5])
             
            drl_action[i].append(action)
            ## End of Section DRL's actting
    
       
            ## Section environment update ##
            observation_, action_fdbk, reward_e, _, done, scope, risk, travel_dis,_ = env.step(action)
            
            state_1 = state_[:,:,2].copy()
            state_[:,:,0] = state_[:,:,1].copy()
            state_[:,:,1] = state_1
            state_[:,:,2] = observation_.copy()
            
            ## End of Section environment update ##
    

            reward = reward_e 
            ## End of Section reward shaping ##


            ## Section DRL store ##
                       
            intervention = 0
            DRL.store_transition(state, action, action_fdbk, intervention, reward, state_,risk)
            ## End of DRL store ##
    
    
            ## Section DRL update ##
            learn_threshold = args.warmup_threshold if args.warmup else 256
            if total_step > learn_threshold:
                #c_loss, a_loss = DRL.learn(batch_size = 16, epoch=i)
                c_loss, a_loss, dis_loss, kl_loss, q_target, cross_correlation, kl_ours, nss, mae, td_errors, sim = DRL.learn(batch_size = 16, epoch=i,eye=args.eye)
                loss_critic.append(np.average(c_loss))
                loss_actor.append(np.average(a_loss))
                q_target = q_target.cpu()
                cross_correlation = cross_correlation.cpu()
                kl_ours = kl_ours.cpu()
                #nss = nss.cpu()
                mae = mae.cpu()
                
                # Decrease the exploration rate
                exploration_rate = exploration_rate * args.exploration_decay_rate if exploration_rate>args.cutoff_exploration_rate else args.cutoff_exploration_rate
            ## End of Section DRL update ##
            
            ep_reward += reward
            global_reward_list.append([reward_e,reward_i])
            reward_e_record[i].append(reward_e)
            reward_i_record[i].append(reward_i)
            
            adopted_action[i].append(action)

            observation = observation_.copy()
            state = state_.copy()
            
            dura = env.terminate_position
            total_step += 1
            step += 1
            
            signal.signal(signal.SIGINT, signal_handler)
        
        mean_reward =  ep_reward / step  
        episode_total_reward_list.append(ep_reward)
        episode_mean_reward_list.append(mean_reward)
        episode_duration_list.append(dura)

        print('\n episode is:',i)
        print('explore_rate:',round(exploration_rate,4))
        print('c_loss:',round(np.average(c_loss),4))
        print('a_loss',round(np.average(a_loss),4))
        print('total_step:',total_step)
        print('episode_step:',step)
        print('episode_cumu_reward',round(ep_reward,4))
        
        writer.add_scalar('reward/reward_episode', ep_reward, i)
        writer.add_scalar('reward/reward_episode_noshaping', np.mean(reward_e_record[i]), i)
        writer.add_scalar('reward/risk', risk, i)
        writer.add_scalar('reward/duration_episode', step, i)
        writer.add_scalar('reward/survival_distance', travel_dis, i)
        writer.add_scalar('rate_exploration', round(exploration_rate,4), i)
        writer.add_scalar('loss/loss_critic', round(np.average(c_loss),4), i)
        writer.add_scalar('loss/loss_actor', round(np.average(a_loss),4), i)
        writer.add_scalar('loss/loss_distance', dis_loss, i)
        writer.add_scalar('loss/loss_kl', kl_loss, i)
        writer.add_scalar('loss/loss_sim', sim, i)
        writer.add_scalar('loss/q_target', np.average(q_target), i)
        writer.add_scalar('loss/td_errors', np.average(td_errors), i)
        writer.add_scalar('loss/cross_correlation', np.average(cross_correlation), i)
        writer.add_scalar('loss/kl_ours', np.average(kl_ours), i)
        writer.add_scalar('loss/nss', np.average(nss), i)
        writer.add_scalar('loss/mae', np.average(mae), i)
        
    
    print('total time:',time.perf_counter()-start_time)        
    
    timeend = round(time.time())
    DRL.save_actor('./algo/models_leftturn', timeend)
    
    pygame.display.quit()
    pygame.quit()
    
    action_drl = drl_action[0:i]
    action_final = adopted_action[0:i]
    scio.savemat('dataleftturn_{}-{}-{}.mat'.format(args.seed,args.algorithm,timeend), mdict={'action_drl': action_drl,'action_final': action_final,
                                                    'stepreward':global_reward_list,'mreward':episode_mean_reward_list,
                                                    'step':episode_duration_list,'reward':episode_total_reward_list,
                                                    'r_i':reward_i_record,'r_e':reward_e_record})
    
    


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--bc', action="store_true", default=False)
    parser.add_argument('--algorithm', type=int, help='RL algorithm (0 for Proposed, 1 for IARL, 2 for HIRL, 3 for Vanilla TD3) (default: 0)', default=0)
    parser.add_argument('--human_model', action="store_true", help='whehther to use human behavior model (default: False)', default=False)
    parser.add_argument('--human_model_update', action="store_true", help='whehther to update human behavior model (default: False)', default=False)
    parser.add_argument('--maximum_episode', type=float, help='maximum training episode number (default:400)', default=1100)
    parser.add_argument('--seed', type=int, help='fix random seed', default=2)
    parser.add_argument("--initial_exploration_rate", type=float, help="initial explore policy variance (default: 0.5)", default=0.5)
    parser.add_argument("--cutoff_exploration_rate", type=float, help="minimum explore policy variance (default: 0.05)", default=0.05)
    parser.add_argument("--exploration_decay_rate", type=float, help="decay factor of explore policy variance (default: 0.99988)", default=0.99988)
    parser.add_argument('--resume', action="store_true", help='whether to resume trained agents (default: False)', default=False)
    parser.add_argument('--warmup', action="store_true", help='whether to start training until collecting enough data (default: False)', default=False)
    parser.add_argument('--warmup_threshold', type=int, help='warmup length by step (default: 1e4)', default=1e4)
    parser.add_argument('--reward_shaping', type=int, help='reward shaping scheme (0: none; 1:proposed) (default: 1)', default=1)
    parser.add_argument('--device', type=str, help='run on which device (default: cuda)', default='cuda')
    parser.add_argument('--simulator_port', type=int, help='Carla port value which needs specifize when using multiple CARLA clients (default: 2000)', default=2000)
    parser.add_argument('--simulator_render_frequency', type=int, help='Carla rendering frequenze, Hz (default: 12)', default=12)
    parser.add_argument('--simulator_conservative_surrounding', action="store_true", help='surrounding vehicles are conservative or not (default: False)', default=False)
    parser.add_argument('--joystick_enabled', action="store_true", help='whether use Logitech G29 joystick for human guidance (default: False)', default=False)
    parser.add_argument('--eye', action="store_true", default=False)
    args = parser.parse_args()

    # Run
    train_leftturn_task()
