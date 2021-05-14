import torch.nn as nn
import torch

class Noise(nn.Module):
    def __init__(self, std):
        super(Noise, self, ).__init__()
        self.std = std
        self.buffer = None

    def forward(self, x):
        out = x.clone()
        if self.training and self.std > 1.0e-6:
            if self.buffer is None:
                self.buffer = torch.Tensor(x.size()).normal_(0, self.std).cuda()
            else:
                self.buffer.resize_(x.size()).normal_(0, self.std)
            out += self.buffer
        if self.eval and self.std > 0:
            self.buffer = torch.Tensor(x.size()).normal_(0, self.std).cuda()
            out += self.buffer
        return out

class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()        
        self.main = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            )
        self.fc = nn.Sequential(
            nn.Linear(4*4*50, 500),    
            nn.ReLU(),            
            nn.Linear(500, 10),
            )
    def forward(self, x):
        x = self.main(x)
        x = x.view(-1, 4*4*50)
        x = self.fc(x)
        return x


class NoiseCNN(nn.Module):
    def __init__(self, std0, std):
        super(NoiseCNN, self).__init__()
        self.std = std
        self.main = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            Noise(std),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            Noise(std),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            )
        self.fc = nn.Sequential(
            nn.Linear(4*4*50, 500),    
            nn.ReLU(),            
            nn.Linear(500, 10),
            )
        self.noise_init = Noise(std0)

    def forward(self, x):
        x = self.noise_init(x)
        x = self.main(x)
        x = x.view(-1, 4*4*50)
        x = self.fc(x)
        return x