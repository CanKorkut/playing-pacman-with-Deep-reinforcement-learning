#https://books.google.com.tr/books?id=JUC7DwAAQBAJ&pg=PA255&lpg=PA255&dq=Image.CUBIC+torch&source=bl&ots=FR7Ntydiiv&sig=ACfU3U3AOvw789YzftytQ1xP9LYPuDUCBg&hl=tr&sa=X&ved=2ahUKEwi55Z_drJ7nAhUOlIsKHaNbBVUQ6AEwCnoECAkQAQ#v=onepage&q=Image.CUBIC%20torch&f=false

import cv2
import gym
import time
import matplotlib.pyplot as plt
import time

env = gym.make("MsPacman-v4") # try for different environements

frame = env.render('rgb_array')

pathOut = 'video2.avi'
fps = 60

size = (640,480)
print(size)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('video.avi',fourcc,20.0,size)



from PIL import Image
import torchvision.transforms as T
import torch

transform = T.Compose([
    T.ToPILImage(),
    T.Grayscale(),
    T.Resize((84,84), interpolation=Image.CUBIC),
    T.ToTensor()
    ])


import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1,padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(12800 ,256)
        self.out  = nn.Linear(256,4)
        self.dout = nn.Dropout(0.25)
        #self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(-1 , 12800)
        x = self.dout(x)
        x = self.head(x)
        x = F.tanh(x)
        x = self.dout(x)
        x = self.out(x)
        return x


import torch
device = torch.device("cpu")

BATCH_SIZE = 512
GAMMA = 0.5
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 25

target_net = DQN()
target_net.load_state_dict(torch.load("./checkpoints/model_pacman_150.pth",map_location=torch.device('cpu')))
target_net.eval()



import torch.nn as nn
import torch.optim as optim


def get_screen():
    screen = env.render(mode='rgb_array')
    screen = screen[:172,:180].transpose((2, 0, 1))
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)

    return transform(screen).to(device)



import math
import random
import numpy as np
from itertools import count
from collections import namedtuple


yon = ['Yukari', 'Sol','Sag','Asagi']

soft = nn.Softmax()

def show_game():
    frames = []
    env.reset()
    t1 = get_screen()
    t2 = t1
    t3 = t2
    t4 = t3
    t_image = torch.stack([t1,t2,t3,t4],dim=0)
    t_image = t_image.reshape(4,84,84)
    for t in count():
        predict = target_net(t_image.unsqueeze(0))
        predict = soft(predict)
        action = predict[0].unsqueeze(0).max(1)[1].view(1, 1)
        print("%"+str(predict[0][action].item())+" "+yon[action.item()])
	#action = target_net(torch.Tensor(frame)).max(1)[1].view(1, 1)
        _, reward, done, _ = env.step(action.item()+1)
        next_state = get_screen()
        t4 = t3
        t3 = t2 
        t2 = t1  
        t1 = next_state 
        t_image = torch.stack([t1,t2,t3,t4],dim=0)
        t_image = t_image.reshape(4,84,84)
        frame=env.render('rgb_array')
        frame = cv2.resize(frame,(640,480))
        cv2.imshow('frames',frame)
        cv2.waitKey(10)
        out.write(frame)
        time.sleep(0.005)
        if done:
            env.reset()
            cv2.destroyAllWindows()
            break
    cv2.destroyAllWindows()


out.release()
show_game()
