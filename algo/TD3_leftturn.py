import os
import numpy as np
import torch

import torch.nn.functional as F

from cpprb import PrioritizedReplayBuffer

from algo.network_model_leftturn import Actor,Critic
from algo.util import hard_update, soft_update
import torch.nn as nn
import matplotlib.pyplot as plt

MEMORY_CAPACITY = 38400
BATCH_SIZE = 128
GAMMA = 0.95
LR_C = 0.0005
LR_A = 0.0005
#LR_A = 0.0002
LR_I = 0.01
TAU = 0.001
#TAU = 0.01
POLICY_NOSIE = 0.2
POLICY_FREQ = 1
NOISE_CLIP = 0.5

def kl_loss(y_true, y_pred, eps=1e-7):
    """
    Kullback-Leiber divergence in PyTorch. Assumes shape (b, 1, h, w) for all tensors.

    :param y_true: groundtruth.
    :param y_pred: prediction.
    :param eps: regularization epsilon.
    :return: loss value (one value per batch element).
    """
    y_true = torch.tensor(y_true, dtype=torch.float32, requires_grad=True)
    y_pred = torch.tensor(y_pred, dtype=torch.float32, requires_grad=True)
    P = y_pred
    P = P / (eps + P.sum(dim=[1, 2, 3], keepdim=False))
    Q = y_true
    Q = Q / (eps + Q.sum(dim=[1, 2, 3], keepdim=False))

    kld = (Q * (((eps + Q) / (eps + P)).log())).sum()

    #kld = (Q * (Q / (eps + P)).log()).sum(dim=[1, 2, 3])
    return kld*28*64

def nss_loss(saliency_map, fixation_map):
    """
    Normalized Scanpath Saliency (NSS) loss implemented in PyTorch.
    Assumes the input tensors have shape (b, 1, h, w).

    :param y_true: Ground truth tensor.
    :param y_pred: Prediction tensor.
    :return: Loss value (one value per batch element).
    """
    
    # Normalize the saliency map
    saliency_map = (saliency_map - saliency_map.mean()) / saliency_map.std()
    
    eps = 1e-7
    fixation_map = (fixation_map - fixation_map.min()) / (eps + fixation_map.max()-fixation_map.min())
    # Create a boolean mask from fixation_map
    mask = fixation_map.bool()

    # Compute the mean of the normalized saliency map at fixation locations
    score = saliency_map[mask].sum().item()

    return score
    
def SIM(saliency_map1, saliency_map2):
    """
    Compute the similarity between two saliency maps using PyTorch.

    Parameters:
    saliency_map1 (array-like): First saliency map.
    saliency_map2 (array-like): Second saliency map.
    to_plot (bool): If True, display the maps and their intersection.

    Returns:
    float: The similarity score.
    """

    # Convert inputs to PyTorch tensors
    map1 = torch.tensor(saliency_map1, dtype=torch.float32)
    map2 = torch.tensor(saliency_map2, dtype=torch.float32)

    # Resize map1 to the size of map2
    #map1 = F.resize(map1, map2.shape[-2:])

    # Normalize the maps
    if torch.any(map1):
        map1 = (map1 - map1.min()) / (map1.max() - map1.min())
        map1 = map1 / map1.sum()

    if torch.any(map2):
        map2 = (map2 - map2.min()) / (map2.max() - map2.min())
        map2 = map2 / map2.sum()

    # Check for NaN values
    if torch.isnan(map1).all() or torch.isnan(map2).all():
        return torch.tensor(float('nan'))

    # Compute histogram intersection
    diff = torch.minimum(map1, map2)
    score = torch.sum(diff)

    return score.item()



def cc_loss(y_true, y_pred, eps=1e-7):
    # Normalize predictions and ground truth
    y_true = torch.tensor(y_true, dtype=torch.float32, requires_grad=True)
    y_pred = torch.tensor(y_pred, dtype=torch.float32, requires_grad=True)

    P = y_pred / (eps + torch.sum(y_pred, dim=[1, 2, 3], keepdim=True))
    Q = y_true / (eps + torch.sum(y_true, dim=[1, 2, 3], keepdim=True))

    # Calculate N (product of height and width dimensions)
    N = y_pred.size(2) * y_pred.size(3)

    # Compute expectations
    E_pq = torch.sum(Q * P, dim=[1, 2, 3], keepdim=True)
    E_q = torch.sum(Q, dim=[1, 2, 3], keepdim=True)
    E_p = torch.sum(P, dim=[1, 2, 3], keepdim=True)
    E_q2 = torch.sum(Q ** 2, dim=[1, 2, 3], keepdim=True) + eps
    E_p2 = torch.sum(P ** 2, dim=[1, 2, 3], keepdim=True) + eps

    # Compute numerator and denominator of the cc_loss
    num = E_pq - ((E_p * E_q) / N)
    den = torch.sqrt((E_q2 - E_q ** 2 / N) * (E_p2 - E_p ** 2 / N))

    # Final cc_loss calculation
    return torch.sum(- (num + eps) / (den + eps), dim=[1, 2, 3])

def information_gain(y_true, y_pred, y_base, eps=1e-7):
    """
    Information gain. Assumes shape (b, 1, h, w) for all tensors.

    :param y_true: groundtruth.
    :param y_pred: prediction.
    :param y_base: baseline.
    :param eps: regularization epsilon.
    :return: loss value (one value per batch element).
    """
    # Normalize the prediction
    P = y_pred / (eps + torch.max(y_pred, dim=[1, 2, 3], keepdim=True))

    # Discretize the ground truth at 0.5
    Qb = torch.round(y_true)

    # Calculate N
    N = torch.sum(Qb, dim=[1, 2, 3], keepdim=True)

    # Calculate Information Gain
    ig = torch.sum(Qb * (torch.log(eps + P) / torch.log(torch.tensor(2.)) -
                         torch.log(eps + y_base) / torch.log(torch.tensor(2.))),
                   dim=[1, 2, 3]) / (eps + N)

    return ig

class DRL:
    def __init__(self, action_dim, state_dim, device='cuda', LR_C = LR_C, LR_A = LR_A):

        self.device = device
        
        self.state_dim = state_dim[0] * state_dim[1] * 2
        self.state_dim_width = state_dim[0]
        self.state_dim_height = state_dim[1]
        self.action_dim = action_dim
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA
        self.tau = TAU
        self.policy_noise = POLICY_NOSIE
        self.noise_clip = NOISE_CLIP
        self.policy_freq = POLICY_FREQ
        self.itera = 0

        self.pointer = 0
        self.replay_buffer = PrioritizedReplayBuffer(MEMORY_CAPACITY,
                                                  {"obs": {"shape": (45,80,3)},
                                                   "act": {"shape":action_dim},
                                                   "acte": {"shape":action_dim},
                                                   "intervene": {},
                                                   "rew": {},
                                                   "next_obs": {"shape": (45,80,3)},
                                                   "done": {},
                                                   "distance": {},
                                                   "att_map": {"shape": (28*64,)},
                                                   "human_map": {"shape": (28*64,)}},
                                                  next_of=("obs"))
        
        self.actor = Actor(self.state_dim,self.action_dim).to(self.device)            
        self.actor_target = Actor(self.state_dim,self.action_dim).to(self.device)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),LR_A)
        self.actor_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.actor_optimizer, 0.996)
        self.previous_epoch = 0
        
        self.critic = Critic(self.state_dim,self.action_dim).to(self.device)
        self.critic_target = Critic(self.state_dim,self.action_dim).to(self.device)
        self.critic_optimizers = torch.optim.Adam(self.critic.parameters(),LR_C)
        
        hard_update(self.actor_target,self.actor)
        hard_update(self.critic_target,self.critic)
        self.mse_loss = nn.MSELoss()
        self.softmax = torch.nn.Softmax(dim=2)
        self.criterion = nn.KLDivLoss(reduction='batchmean')
        
        self.mae_loss = nn.L1Loss()
        #self.criterion = nn.CrossEntropyLoss()
        
            
    def learn(self, batch_size = BATCH_SIZE, epoch=0, train=True, eye=False):

        ## batched state, batched action, batched action from expert, batched intervention signal, batched reward, batched next state
        data = self.replay_buffer.sample(batch_size)
        #import pdb; pdb.set_trace()
        idxs = data['indexes']
        states, actions, actions_exp, att_maps, human_maps = data['obs'], data['act'], data['acte'], data['att_map'], data['human_map']
        rewards = data['rew']
        next_states, dones = data['next_obs'], data['done']
        distance = data['distance']
        
        states = torch.FloatTensor(states).permute(0,3,1,2).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        actions_exp = torch.FloatTensor(actions_exp).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).permute(0,3,1,2).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        distance = torch.FloatTensor(distance).to(self.device)

        # initialize the loss variables
        loss_c, loss_a, loss_d = 0, 0, 0

        ## calculate the predicted values of the critic
        with torch.no_grad():
            #noise1 = (torch.randn_like(actions) * self.policy_noise).clamp(0, 1)
            noise1 = (torch.randn_like(actions) * self.policy_noise).clamp(-1, 1)
            actor_target, _ = self.actor_target(next_states)
            actor_target = actor_target.detach()
            #next_actions = (actor_target + noise1).clamp(0, 1)
            next_actions = (actor_target + noise1).clamp(-1, 1)
            #print(next_actions)
            target_q1, target_q2 = self.critic_target([next_states, next_actions])
            target_q = torch.min(target_q1,target_q2)
            y_expected = rewards + (1-dones)*self.gamma * target_q    
        y_predicted1, y_predicted2 = self.critic.forward([states, actions]) 
        
        ## calculate td error
        td_errors = abs(y_expected - y_predicted1.detach())
        
        ## update the critic
        loss_critic = F.mse_loss(y_predicted1,y_expected) + F.mse_loss(y_predicted2,y_expected)
        self.critic_optimizers.zero_grad()
        if train: 
            loss_critic.backward()
        self.critic_optimizers.step()
        
        ##############
        #criterion = nn.BCELoss().cuda()
        #import pdb; pdb.set_trace()
        # Define the new size you want to crop to
        
        new_height, new_width = 28, 64
        
        # Calculate the coordinates to place the new image at the center
        top = (self.actor.human_map.size(2) - new_height) // 2
        left = (self.actor.human_map.size(3) - new_width) // 2
        
        # Perform the crop
        center_cropped_image = self.actor.human_map[:,:, top:top + new_height, left:left + new_width]
        # Since view cannot be used due to the tensor's memory layout, we will use reshape instead
        #import pdb; pdb.set_trace()
        #center_cropped_image= self.actor.human_map
        flattened = center_cropped_image.reshape(center_cropped_image.size(0), -1)
        
        # Apply softmax across the spatial dimension (which is now flattened)
        #prob_distribution = F.softmax(flattened, dim=1)
        
        # Reshape back to the original spatial dimensions [C, H, W]
        #prob_distribution = prob_distribution.view(center_cropped_image.size())

        # Since view cannot be used due to the tensor's memory layout, we will use reshape instead
        flattened_actor = self.actor.padded_saliency.cuda().unsqueeze(0).unsqueeze(0).reshape(center_cropped_image.size(0), -1)
        
        # Reshape back to the original spatial dimensions [C, H, W]
        #prob_distribution_actor = prob_distribution_actor.view(center_cropped_image.size())
        
        # Apply softmax to get probability distributions
        prob_distribution = F.softmax(flattened.view(1, -1), dim=1).view_as(flattened)
        prob_distribution_actor = F.softmax(flattened_actor.view(1, -1), dim=1).view_as(flattened_actor)
        #prob_distribution = (flattened.view(1, -1)).view_as(flattened)/(flattened.shape[1])
        #prob_distribution_actor = (flattened_actor.view(1, -1)).view_as(flattened_actor)/(flattened_actor.shape[1])
        
        #bce_loss = criterion(prob_distribution_actor, prob_distribution)
        #import pdb; pdb.set_trace()
        # Convert the first distribution to log-probabilities
        log_prob_distribution = F.log_softmax(flattened.view(1, -1), dim=1).view_as(flattened)
        log_prob_distribution_actor = F.log_softmax(flattened_actor.view(1, -1), dim=1).view_as(flattened_actor)
        
        #kl_divergence = kl_loss(prob_distribution_actor.view(1,1,28, 64), prob_distribution.view(1,1,28, 64))
        batch_size = 1
        self.human_att = prob_distribution.view(batch_size,1,28, 64)
        self.machine_att = prob_distribution_actor.view(batch_size,1,28, 64)
        kl_divergence = self.criterion(log_prob_distribution.view(batch_size,1,28, 64), prob_distribution_actor.view(batch_size,1,28, 64))
        #kl_divergence = F.kl_div(log_prob_distribution.view(1,1,28, 64), prob_distribution_actor.view(1,1,28, 64), None, None, 'sum')*28*64
        cross_correlation = cc_loss(prob_distribution.view(batch_size,1,28, 64), prob_distribution_actor.view(batch_size,1,28, 64))
        sim = SIM(prob_distribution.view(batch_size,1,28, 64), prob_distribution_actor.view(batch_size,1,28, 64))
        kl_ours = kl_loss(prob_distribution.view(batch_size,1,28, 64), prob_distribution_actor.view(batch_size,1,28, 64))
        nss = nss_loss(prob_distribution.view(batch_size,1,28, 64), prob_distribution_actor.view(batch_size,1,28, 64))
        mae = self.mae_loss(prob_distribution, prob_distribution_actor)
        #import pdb; pdb.set_trace()
        self.center_cropped_state = self.actor.observation[top:top + new_height, left:left + new_width]
        
        #######
        ## update the actor
        if self.itera % self.policy_freq == 0:
            
            pred_actions, pred_dis = self.actor.forward(states)
            
            loss_actor = -self.critic([states, pred_actions])[0] 
            #import pdb; pdb.set_trace()
            loss_distance = self.mse_loss(pred_dis, distance)
            #loss_actor = loss_actor + 0.1* loss_distance
            #loss_d = loss_distance.item()
            #loss_actor = kl_divergence.requires_grad_()*1000
            #loss_actor = loss_actor + loss_distance
            #loss_actor = loss_actor + loss_distance *(1/epoch)
            
            if eye == True:
                if self.itera<500:
                    loss_actor = loss_actor + 0.1* loss_distance + kl_divergence*0.05
                else:
                    loss_actor = loss_actor + 0.1* loss_distance
            else:
                loss_actor = loss_actor + 0.1* loss_distance
            
            loss_actor = loss_actor + loss_distance*(1/epoch)
            loss_actor = loss_actor.mean()
            
            #print(loss_distance)
            #print(loss_actor)

            self.actor_optimizer.zero_grad()
            if train:
                loss_actor.backward()
            self.actor_optimizer.step()

            if epoch != self.previous_epoch:
                self.actor_scheduler.step() 
            self.previous_epoch = epoch
            
            soft_update(self.actor_target,self.actor,self.tau)
            soft_update(self.critic_target,self.critic,self.tau)
            
            loss_a = loss_actor.item()
            
            
        loss_c = loss_critic.item()
        
        self.itera += 1
        
        # TDQA priority calculation
        #priorities = td_errors + qa_errors
        #import pdb; pdb.set_trace()
        #print(td_errors)
        priorities = td_errors
        #priorities = td_errors + cross_correlation*0.1
        #priorities = td_errors*kl_divergence
        priorities = priorities.cpu().numpy()
        self.replay_buffer.update_priorities(idxs, priorities)
        
        return loss_c, loss_a, loss_distance, kl_divergence, target_q, cross_correlation.detach(), kl_ours.detach(), nss, mae.detach(), td_errors.cpu(), sim
    
                
    def choose_action(self,state):

        state = torch.FloatTensor(state).float().unsqueeze(0).permute(0,3,1,2).to(self.device)
        #import pdb; pdb.set_trace()
        action, dis = self.actor.forward(state)
        action = action.detach().cpu()
        dis = dis.detach()
        action = action.squeeze(0).cpu().numpy()
        action = np.clip(action,-1, 1)
        
        new_height, new_width = 28, 64
        
        # Calculate the coordinates to place the new image at the center
        top = (self.actor.human_map.size(2) - new_height) // 2
        left = (self.actor.human_map.size(3) - new_width) // 2
        center_cropped_image = self.actor.human_map[:,:, top:top + new_height, left:left + new_width]
        # Since view cannot be used due to the tensor's memory layout, we will use reshape instead
        #import pdb; pdb.set_trace()
        #center_cropped_image= self.actor.human_map
        flattened = center_cropped_image.reshape(center_cropped_image.size(0), -1)
        
        prob_distribution = F.softmax(flattened.view(1, -1), dim=1).view_as(flattened)
        
        #import pdb; pdb.set_trace()
        self.human_att_i = prob_distribution.view(1,1,28, 64)
        
        return action, dis
    

    def store_transition(self,  s, a, ae, i, r, s_, dis, d=0):
        self.replay_buffer.add(obs=s,
                               act=a,
                               acte=ae,
                               intervene=i,
                               rew=r,
                               next_obs=s_,
                               done=d,
                               distance=dis,
                               att_map=self.actor.flattened_actor.detach().cpu(),
                               human_map=self.actor.flattened.detach().cpu())
    

    def save_transition(self, output, timeend=0):
        self.replay_buffer.save_transitions(file='{}/{}'.format(output, timeend))

    def load_transition(self, output):
        if output is None: return
        self.replay_buffer.load_transitions('{}.npz'.format(output))
        
    
    def load_model(self, output):
        if output is None: return
        self.actor.load_state_dict(torch.load('{}/actor.pkl'.format(output)))
        self.critic.load_state_dict(torch.load('{}/actor.pkl'.format(output)))
    
    def load_actor(self, output):
        self.actor.load_state_dict(torch.load(output))
        
    def save_model(self, output):
        torch.save(self.actor.state_dict(), '{}/actor.pkl'.format(output))
        torch.save(self.critic.state_dict(), '{}/critic.pkl'.format(output))
    
    def save_actor(self, output, no):
        torch.save(self.actor.state_dict(), '{}/actor{}.pkl'.format(output, no))
    
    def save(self, log_dir, epoch):
        state = {'actor':self.actor.state_dict(), 'actor_target':self.actor_target.state_dict(),
                 'actor_optimizer':self.actor_optimizer.state_dict(), 
                 'critic':self.critic.state_dict(), 'critic_target':self.critic_target.state_dict(),
                 'critic_optimizers':self.critic_optimizers.state_dict(),
                 'epoch':epoch}
        torch.save(state, log_dir)
        
    def load(self, log_dir):
        checkpoint = torch.load(log_dir)
        self.actor.load_state_dict(checkpoint['state_dict'])
        self.actor_target.load_state_dict(checkpoint['state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['optim_dict'])

