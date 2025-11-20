import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io as scio
import os
import pygame
import signal
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import csv
from torch.utils.tensorboard import SummaryWriter   
writer = SummaryWriter('./algo/checkpoints/log')

from env_car_following_save import CarFollowing
#from utils import set_seed, signal_handler
from human_model import Actor as h_Model

import sys
#import pygame
def signal_handler(sig, frame):
    print('Procedure terminated!')
    pygame.display.quit()
    pygame.quit()
    sys.exit(0)

#import torch
import numpy as np
from datetime import datetime
import cv2

# Assuming images are in RGB format and need to be converted to BGR for OpenCV
def save_image(image, path):
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image_bgr)
    
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
directory = r'./ddpg_save/'
# Arguments
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--bc', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--algorithm', type=int, help='RL algorithm (0 for Proposed, 1 for IARL, 2 for HIRL, 3 for Vanilla TD3) (default: 0)', default=0)
parser.add_argument('--human_model', action="store_true", help='whehther to use human behavior model (default: False)', default=False)
parser.add_argument('--human_model_update', action="store_true", help='whehther to update human behavior model (default: False)', default=False)
parser.add_argument('--maximum_episode', type=float, help='maximum training episode number (default:400)', default=1)
#parser.add_argument('--seed', type=int, help='fix random seed', default=2)
parser.add_argument("--initial_exploration_rate", type=float, help="initial explore policy variance (default: 0.5)", default=0.5)
parser.add_argument("--cutoff_exploration_rate", type=float, help="minimum explore policy variance (default: 0.05)", default=0.05)
parser.add_argument("--exploration_decay_rate", type=float, help="decay factor of explore policy variance (default: 0.99988)", default=0.99988)
parser.add_argument('--resume', action="store_true", help='whether to resume trained agents (default: False)', default=False)
parser.add_argument('--warmup', action="store_true", help='whether to start training until collecting enough data (default: False)', default=False)
parser.add_argument('--warmup_threshold', type=int, help='warmup length by step (default: 1e4)', default=1e4)
parser.add_argument('--reward_shaping', type=int, help='reward shaping scheme (0: none; 1:proposed) (default: 1)', default=1)
parser.add_argument('--device', type=str, help='run on which device (default: cuda)', default='cuda:0')
parser.add_argument('--simulator_port', type=int, help='Carla port value which needs specifize when using multiple CARLA clients (default: 2000)', default=2000)
parser.add_argument('--simulator_render_frequency', type=int, help='Carla rendering frequenze, Hz (default: 12)', default=40)
parser.add_argument('--simulator_conservative_surrounding', action="store_true", help='surrounding vehicles are conservative or not (default: False)', default=False)
parser.add_argument('--joystick_enabled', action="store_true", help='whether use Logitech G29 joystick for human guidance (default: False)', default=False)

parser.add_argument('--tau',  default=0.005, type=float) # target smoothing coefficient
parser.add_argument('--target_update_interval', default=10, type=int)
parser.add_argument('--test_iteration', default=10, type=int)

parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--gamma', default=0.99, type=int) # discounted factor
parser.add_argument('--capacity', default=1000000, type=int) # replay buffer size
parser.add_argument('--batch_size', default=100, type=int) # mini batch size
parser.add_argument('--seed', default=False, type=bool)
parser.add_argument('--random_seed', default=9527, type=int)
# optional parameters

parser.add_argument('--sample_frequency', default=2000, type=int)
parser.add_argument('--render', default=False, type=bool) # show UI or not
parser.add_argument('--log_interval', default=50, type=int) #
parser.add_argument('--load', default=False, type=bool) # load model
parser.add_argument('--render_interval', default=100, type=int) # after render_interval, the env.render() will work
parser.add_argument('--exploration_noise', default=0.1, type=float)
parser.add_argument('--max_episode', default=100000, type=int) # num of games
parser.add_argument('--print_log', default=5, type=int)
parser.add_argument('--update_iteration', default=10, type=int)
parser.add_argument('--idm', action="store_true", default=False)

args = parser.parse_args()

class Replay_buffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''
    def __init__(self, max_size=args.capacity):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1) 



def train_leftturn_task():
    
    set_seed(args.seed)
    
    # construct the DRL agent
    from algo.TD3_car_following import DRL as DRL_base
    
    env = CarFollowing(joystick_enabled = args.joystick_enabled, conservative_surrounding = args.simulator_conservative_surrounding, 
                   frame=args.simulator_render_frequency, port=args.simulator_port)
     
    condition = ['aegis_checkpoints']
    
    for cond in condition:
        log_dir = r'algo/checkpoints/car_following/{}/'.format(cond)
        file_list = os.listdir(log_dir)
        now = datetime.now().strftime("-%d-%m-%Y-%H-%M-%S")
       
        figure_dir = r'{}/'.format(cond)
        
        if not os.path.exists(figure_dir):
            os.mkdir(figure_dir)
        for ii in range(0,len(file_list)):
        #for ii in range(1,2):
            print(file_list)
            log_dir_load = os.path.join(log_dir,file_list[ii])
            
            output_file = "output{}.csv".format(now)
            try:
                with open(output_file, 'w', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow(['frame','action','ego_speed','front_speed','ttc','predict_ttc','travel_distance','KL','CC','NSS','SIM','MAE'])
            except IOError as e:
                print("Error:", e)
            s_dim = [env.observation_size_width, env.observation_size_height]
            a_dim = env.action_size
            
            DRL = DRL_base(a_dim, s_dim, device=args.device)
        
            
            exploration_rate = args.initial_exploration_rate 
            
            if os.path.exists(log_dir_load):
                checkpoint = torch.load(log_dir_load)
                #DRL.load(log_dir)
                DRL.load_actor(log_dir_load)
              
                start_epoch = 0
            else:
                start_epoch = 0
            
 
              
            # initialize global variables
            total_step = 0
            a_loss,c_loss, dis_loss, kl_loss = 0,0,0,0
            q_target, cross_correlation, kl_ours, nss,mae, td_errors,sim = 0,0.,0,0,0,0,0
            count_sl = 0
            
            loss_critic, loss_actor = [], []
            episode_total_reward_list, episode_mean_reward_list = [], []
            global_reward_list, episode_duration_list = [], []
            ttc_list = []
            
            drl_action = [[] for i in range(args.maximum_episode)] # drl_action: output by RL policy
            adopted_action = [[] for i in range(args.maximum_episode)]  # adopted_action: eventual action which may include human guidance action
             
           
            dis_front = 0
            start_time = time.perf_counter()
            
            plt.ion()
            fig = plt.figure(figsize=(6, 8))
            for i in range(start_epoch, args.maximum_episode):
                set_seed(i)
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
                    action, dis = DRL.choose_action(state)
                    
                    if args.bc:
                        #action,_ = h_model(torch.from_numpy(state).cuda().transpose(0,2).float().unsqueeze(0))
                        with torch.no_grad():
                            action_h, _ = h_model(torch.from_numpy(state).cuda().transpose(0,2).float().unsqueeze(0))
                        #action_h = action.detach().cpu().numpy()
                        #import pdb; pdb.set_trace()
                        action = action_h.detach().cpu().numpy()
                        mae_h = np.abs(action-action_h.cpu().numpy())
                    
                    if args.idm:
                        if dis_front > 20:
                            action = np.array([0.8])
                        elif dis_front <=10:
                            action = np.array([-0.5])
                        else:
                            action = np.array([0.0])
                            
                    drl_action[i].append(action)
                    ## End of Section DRL's actting
            
               
                    ## Section environment update ##
                    observation_, action_fdbk, reward_e, _, done, scope, dis_front, travel_dis, _, rgb, ego_speed,front_speed, semantics = env.step(action)
                    
                    state_1 = state_[:,:,2].copy()
                    state_[:,:,0] = state_[:,:,1].copy()
                    state_[:,:,1] = state_1
                    state_[:,:,2] = observation_.copy()
                    
                    reward = reward_e
                    ## End of Section reward shaping ##
                    with torch.no_grad():
                        c_loss, a_loss, dis_loss, kl_loss, q_target, cross_correlation, kl_ours, nss, mae, td_errors, sim = DRL.learn(batch_size = 16, epoch=i, eye=True, train=False) 
                    
                    try:
                        with open(output_file, 'a', newline='') as csvfile:
                            csv_writer = csv.writer(csvfile)
                            csv_writer.writerow([total_step, action,ego_speed,front_speed,dis_front/(ego_speed-front_speed+0.0001),dis.cpu().numpy()[0][0]/(ego_speed-front_speed+0.0001),travel_dis])
                    except IOError as e:
                        print("Error:", e)
                
                    
                    machine_att = DRL.machine_att.cpu()
                                      
                    new_height, new_width = 28*16, 64*16
                   
                    if rgb is not None and step%10==0:
                    
                        # Calculate the coordinates to place the new image at the center
                        top = (rgb.shape[0] - new_height) // 2
                        left = (rgb.shape[1] - new_width) // 2
                        center_cropped_image = rgb[top:top + new_height, left:left + new_width,:]
                        center_cropped_state = semantics[top:top + new_height, left:left + new_width]
                        
                        plt.clf()   # clear previous frame

                        # ----- Subplot 1: image + attention map -----
                        ax1 = fig.add_subplot(2, 1, 1)
                        ax1.imshow(center_cropped_image)
                        ax1.imshow(machine_att[0, 0].detach().cpu(), 
                                   cmap='jet', alpha=0.5)
                        ax1.axis('off')

                        # ----- Subplot 2: image only -----
                        ax2 = fig.add_subplot(2, 1, 2)
                        ax2.imshow(center_cropped_image)
                        ax2.axis('off')

                        plt.pause(0.01)   # allow UI update    
                    
                    ep_reward += reward
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
   
            
    
    print('total time:',time.perf_counter()-start_time)        
    
    timeend = round(time.time())
    
    pygame.display.quit()
    pygame.quit()
    
    action_drl = drl_action[0:i]
    action_final = adopted_action[0:i]
   


if __name__ == "__main__":


    # Run
    train_leftturn_task()
