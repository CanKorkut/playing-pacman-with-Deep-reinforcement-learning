

import cv2
import gym
import time
import matplotlib.pyplot as plt

env = gym.make("MsPacman-v4")

frame = env.render('rgb_array')
plt.imshow(frame)
plt.show()


from PIL import Image
import torchvision.transforms as T

transform = T.Compose([
    T.ToPILImage(),
    T.Grayscale(),
    T.Resize((84,84), interpolation=Image.CUBIC),
    T.ToTensor()
    ])

import torch.nn as nn
import torch.nn.functional as F
import torch


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.head = nn.Linear(20736 ,256)
        self.out  = nn.Linear(256,4)
        self.dout = nn.Dropout(0.25)
        #self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(-1 , 20736)
        x = self.dout(x)
        x = self.head(x)
        x = torch.tanh(x)
        x = self.dout(x)
        x = self.out(x)
        return x


import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

BATCH_SIZE = 512
GAMMA = 0.99999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10


policy_net = DQN().to(device)
target_net = DQN().to(device)
#policy_net.load_state_dict(torch.load("./model_pacman.pth"))
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

def calc_Gamma(i):
	return GAMMA-0.00001

print("Model Olusturuldu")

import torch.nn as nn
import torch.optim as optim

#optimizer = optim.RMSprop(policy_net.parameters())
optimizer = optim.SGD(policy_net.parameters(),lr=0.0001)
memory = ReplayMemory(10000)

steps_done = 0
is_model = False
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            is_model = True
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        is_model = False
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

def get_screen():
    screen = env.render(mode='rgb_array')
    screen = screen[:172,:180].transpose((2, 0, 1))
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)

    return transform(screen).to(device)
running_loss = 0
def optimize_model():
    global running_loss
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))


    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.stack([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.stack(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    
    
    state_batch = state_batch.to(device)
    action_batch = action_batch.to(device)
    reward_batch = reward_batch.to(device)

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch


    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    running_loss+=loss.item()
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

import math
import random
import numpy as np
from itertools import count
from collections import namedtuple

Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))

total_reward = 0
best_rewar = 0

n_actions = 4
num_episodes = 10000

print("Egitim basliyor..")
print("Beni soguk bi yere koy")


save_index = 0
for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset()
    t1 = get_screen()
    t2 = t1
    t3 = t2
    t4 = t3
    #state = get_screen()
    next_state = get_screen()
    t_image = torch.stack([t1,t2,t3,t4],dim=0)
    t_image = t_image.reshape(4,84,84)
    for t in count():
        # Select and perform an action
        state = t_image
        action = select_action(t_image.unsqueeze(0))
        _, reward, done, _ = env.step(action.item() +1)
        reward = torch.tensor([reward], device=device)
        total_reward+=reward
        next_state = get_screen()
        t4 = t3
        t3 = t2 
        t2 = t1  
        t1 = next_state 
        t_image = torch.stack([t1,t2,t3,t4],dim=0)
        t_image = t_image.reshape(4,84,84)
        memory.push(state, action, t_image, reward)
        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            print("Episode: ", i_episode,"Curr Reward: ", total_reward.item(),"Loss: ",running_loss, "GAMMA: ", GAMMA)
            total_reward = 0
            running_loss=0
            GAMMA=calc_Gamma(i_episode)
            env.reset()
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
        torch.save(target_net.state_dict(),"./checkpoints_3/model_pacman_" + str(save_index)+".pth")
        save_index = save_index + 1
