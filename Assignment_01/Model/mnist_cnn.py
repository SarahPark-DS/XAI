import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride = 1, padding = 1 )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # conv1 -> ReLU -> MaxPool
        x = self.pool(F.relu(self.conv2(x))) # conv2 -> ReLU -> MaxPool
        x = x.view(-1, 64 * 7 * 7) # Flatten
        x = F.relu(self.fc1(x)) # fc1 -> ReLU
        x = self.fc2(x) # fc2
        return x        

def get_mnist_model():
    return MNIST_CNN