import torch.nn as nn

cfg = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}

class Encoder(nn.Module):
    def __init__(self, n_z, img_width=32):
        super(Encoder, self).__init__()
        """
        The encoder structure in OT_Classifier
        """
        self.img_width = img_width       
        
        self.feature = self._make_layers(cfg['D'])
        
        self.fc1 = nn.Linear(512, n_z)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        width = self.img_width
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                width = width // 2
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=width, stride=1)]
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x    

class SmallCNN(nn.Module):
    def __init__(self, n_z):
        super(SmallCNN, self).__init__()
        """
        The classifier structure in OT_Classifier
        """
        self.n_z = n_z
        
        self.main = nn.Sequential(
            nn.Linear(self.n_z, 512),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )
    def forward(self, x):
        x = self.main(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, dim_h, n_z):
        super(Discriminator, self).__init__()
        """
        The discriminator structure in OT_Classifier
        """
        self.dim_h = dim_h
        self.n_z = n_z
        
        self.main = nn.Sequential(
            nn.Linear(self.n_z, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, 1),
            nn.LogSigmoid()
        )

    def forward(self, x):
        x = self.main(x)
        return x   
    