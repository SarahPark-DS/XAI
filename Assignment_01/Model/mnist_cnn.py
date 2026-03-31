import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

class MNIST_CNN(nn.Module):
    def __init__(self):
        # super(MNIST_CNN, self).__init__()
        # self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride = 1, padding = 1)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride = 1, padding = 1 )
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # self.fc2 = nn.Linear(128, 10)

        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride = 1) # output: 26 * 26 * 32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride = 1) # output: 24 * 24 * 64
        self.dropout1 = nn.Dropout2d(p = 0.25)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.dropout2 = nn.Dropout(p = 0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim = 1)

        return output 

def get_mnist_model():
    return MNIST_CNN