import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

seed = 2
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def conv3x3(in_planes, out_planes):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, 3, 1, 1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes, out_planes, 3, 1, 1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))


class Model(nn.Module):
    def __init__(self):
        n, m = 24, 3

        super(Model, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.maxpool = nn.MaxPool2d(2, 2)


        self.convd1 = conv3x3(1*m, 1*n)
        self.convd2 = conv3x3(1*n, 2*n)
        self.convd3 = conv3x3(2*n, 4*n)
        self.convd4 = conv3x3(4*n, 4*n)

        self.convu3 = conv3x3(8*n, 4*n)
        self.convu2 = conv3x3(6*n, 2*n)
        self.convu1 = conv3x3(3*n, 1*n)

        self.convu0 = nn.Conv2d(n, 1, 3, 1, 1)

    def forward(self, x):
        #import pdb; pdb.set_trace()
       
        x1 = x
        x1 = self.convd1(x1)
        # print(x1.size())

        x2 = self.maxpool(x1)
        x2 = self.convd2(x2)
        # print(x2.size())

        x3 = self.maxpool(x2)
        x3 = self.convd3(x3)
        # print(x3.size())

        x4 = self.maxpool(x3)
        x4 = self.convd4(x4)
        # print(x4.size())

        y3 = self.upsample(x4)
        y3 = torch.cat([x3, y3], 1)
        y3 = self.convu3(y3)
        # print(y3.size())

        y2 = self.upsample(y3)
        y2 = torch.cat([x2, y2], 1)
        y2 = self.convu2(y2)
        # print(y2.size())

        y1 = self.upsample(y2)
        y1 = torch.cat([x1, y1], 1)
        y1 = self.convu1(y1)
        # print(y1.size())

        y1 = self.convu0(y1)
        y1 = self.sigmoid(y1)
        # print(y1.size())
        # exit(0)
        #y1 = F.softmax(y1, dim=1)
        return y1

   
### This actor network is established on CNN structure for the image-style state with the shape of 80*45.
class Actor(nn.Module):
    def __init__(self, nb_states=3, nb_actions=1, hidden=32, init_w=3e-1):
        super(Actor,self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 6)
        self.conv2 = nn.Conv2d(6, 16, 6)
        
        #self.fc1 = nn.Linear(16*16*7, 256)
        self.fc1 = nn.Linear(2016, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, nb_actions)
        self.fc5 = nn.Linear(64, 1)
        self.sig = nn.Tanh()
        self.init_weights(init_w)
        
        self.node_size = 12
        self.proj_shape = (18,self.node_size)  
            #self.proj_shape = (34,36)
        self.k_proj = nn.Linear(*self.proj_shape)
        self.q_proj = nn.Linear(*self.proj_shape)
        self.v_proj = nn.Linear(*self.proj_shape)
        self.N = 112
        self.norm_shape = (self.N,self.node_size)
        self.k_norm = nn.LayerNorm(self.norm_shape, elementwise_affine=True)
        self.q_norm = nn.LayerNorm(self.norm_shape, elementwise_affine=True)
        self.v_norm = nn.LayerNorm(self.norm_shape, elementwise_affine=True)
        # Load the model only once during initialization
        self.model = Model().cuda()
        checkpoint = torch.load('/data/zhuzhuan/CDNN-traffic-saliency/ckpts/cdnn/model_epoch_26.tar')
        self.model.load_state_dict(checkpoint['state_dict'])

    
    def init_weights(self,init_w):
    
        torch.nn.init.kaiming_uniform_(self.fc1.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')    
        torch.nn.init.kaiming_uniform_(self.fc2.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.fc3.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
        self.fc4.weight.data.uniform_(-init_w,init_w)
        
    def forward(self,inp):

        rgb_image = torch.cat([inp[:,1:2],inp[:,1:2],inp[:,1:2]], axis=1)
        # Desired new size
        #new_size = (216//2, 384//2)
        new_size = (72,128)
        
        # Resize/Interpolate the tensor to the new size
        rgb_image = F.interpolate(rgb_image, size=new_size, mode='bilinear', align_corners=False)
        #import pdb; pdb.set_trace()
        #output = self.model(rgb_image)
        
        x = inp.unsqueeze(0) if len(inp.shape)==3 else inp
        N, Cin, H, W = x.shape
        #import pdb; pdb.set_trace()
        x = F.max_pool2d( self.conv1(x), 2)
        x = F.max_pool2d( self.conv2(x), 2)
        
        _,_,cH,cW = x.shape
        xcoords = torch.arange(cW).repeat(cH,1).float() / cW               
        ycoords = torch.arange(cH).repeat(cW,1).transpose(1,0).float() / cH
        spatial_coords = torch.stack([xcoords,ycoords],dim=0).cuda()
        spatial_coords = spatial_coords.unsqueeze(dim=0)
        spatial_coords = spatial_coords.repeat(N,1,1,1) 
        x = torch.cat([x,spatial_coords],dim=1)
        x = x.permute(0,2,3,1)
        x = x.flatten(1,2)
        #import pdb; pdb.set_trace()
        K = self.k_proj(x)                                                 
        K = self.k_norm(K) 
        
        Q = self.q_proj(x)
        Q = self.q_norm(Q) 
        
        V = self.v_proj(x)
        V = self.v_norm(V) 
        A = torch.einsum('bfe,bge->bfg',Q,K)                               
        A = A / np.sqrt(self.node_size)
        A = torch.nn.functional.softmax(A,dim=2) 
        #import pdb; pdb.set_trace()
#        with torch.no_grad():
#            self.att_map = A.clone()
#            self.human_map = F.interpolate(output, size=(45, 80), mode='bilinear',align_corners=False).clone()
#            max_saliency = A[0].detach().cpu().max(dim=0)[0].view(7,16)
#            interpolated_saliency_60 = F.interpolate(max_saliency.unsqueeze(0).unsqueeze(0), size=(28, 64), mode='bilinear',align_corners=False)
#            # Use padding to make it 80x80
#            padding_size_x = 8.5
#            padding_size_y = 8
#            #padded_saliency = F.pad(interpolated_saliency_60, (8, padding_size_y, 9, padding_size_y), mode='constant', value=0)
#            padded_saliency = interpolated_saliency_60
#            self.padded_saliency = padded_saliency.squeeze()
#            #plt.imshow(self.padded_saliency); plt.show(block=False); plt.pause(0.001);
        E = torch.einsum('bfc,bcd->bfd',A,V)                               
        #E = self.linear1(E)
        #E = torch.relu(E)
        #E = self.norm1(E)  
        #E = E.max(dim=1)[0]
                	
        x = x.reshape(E.size(0),-1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        fl = F.relu(self.fc3(x))
        x = self.sig(self.fc4(fl))
        dis = self.fc5(fl)
        
        return x, dis
'''        
### This actor network is established on CNN structure for the image-style state with the shape of 80*45.
class Actor(nn.Module):
    def __init__(self, nb_actions=1, hidden=32, init_w=3e-1):
        super(Actor,self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 6)
        self.conv2 = nn.Conv2d(6, 16, 6)
        self.fc1 = nn.Linear(16*16*7, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sig = nn.Tanh()
        self.init_weights(init_w)
        
    def init_weights(self,init_w):

        torch.nn.init.kaiming_uniform_(self.fc1.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')    
        torch.nn.init.kaiming_uniform_(self.fc2.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.fc3.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
        self.fc4.weight.data.uniform_(-init_w,init_w)
        
    def forward(self,inp):
        #import pdb; pdb.set_trace()
        #x = inp.unsqueeze(1)
        x = F.max_pool2d( self.conv1(inp), 2)
        x = F.max_pool2d( self.conv2(x), 2)
        x = x.view(x.size(0),-1)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        out = self.sig(x)
        return out
''' 
### This critic network is established on MLP structure.
class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden=256, init_w=3e-1):
        super(Critic,self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 6)
        self.conv2 = nn.Conv2d(6, 16, 6)
        self.fc1 = nn.Linear(16*16*7 + nb_actions, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, nb_actions)
        
        self.fc11 = nn.Linear(16*16*7 + nb_actions, 256)
        self.fc21 = nn.Linear(256, 256)
        self.fc31 = nn.Linear(256, 256)
        self.fc41 = nn.Linear(256, nb_actions)
        
        self.relu = nn.ReLU()
        self.sig = nn.Tanh()

        self.init_weights(init_w)
        
    def init_weights(self,init_w):

        torch.nn.init.kaiming_uniform_(self.fc1.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.fc2.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.fc3.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
        self.fc4.weight.data.uniform_(-init_w,init_w)
        
        torch.nn.init.kaiming_uniform_(self.fc11.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.fc21.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.fc31.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
        self.fc41.weight.data.uniform_(-init_w,init_w)
        
    def forward(self,inp):
        
        x, a = inp
        x = x.unsqueeze(1)
        x = F.max_pool2d( self.conv1(x), 2)
        x = F.max_pool2d( self.conv2(x), 2)
        x = x.view(x.size(0),-1)
        
        q1 = self.fc1(torch.cat([x,a],1))
        q1 = self.relu(q1)
        q1 = self.fc2(q1)
        q1 = self.relu(q1)
        q1 = self.fc3(q1)
        q1 = self.relu(q1)
        q1 = self.fc4(q1)
        
        q2 = self.fc11(torch.cat([x,a],1))
        q2 = self.relu(q2)
        q2 = self.fc21(q2)
        q2 = self.relu(q2)
        q2 = self.fc31(q2)
        q2 = self.relu(q2)
        q2 = self.fc41(q2)
        
        return q1, q2


    
    
    
    
