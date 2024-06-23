import torch
from torch import nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(input_dim[0], 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.linear = nn.Linear(1152, 512)
        self.int_critic = nn.Linear(512, 1)
        self.ext_critic = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x/255.))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return self.actor_linear(x), self.ext_critic(x), self.int_critic(x)

class Feature_Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Feature_Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.linear1 = nn.Linear(1152, 512)
        self.linear2 = nn.Linear(512, 512)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class Target_Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Target_Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.linear = nn.Linear(1152, 512)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x