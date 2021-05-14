import torch.nn as nn

class SmallCNN(nn.Module):
    def __init__(self, n_z):
        super(SmallCNN, self).__init__()
        """
        The classifier structure in OT_Classifier
        """
        self.n_z = n_z
        
        self.main = nn.Sequential(
            nn.Linear(self.n_z, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(True),
            nn.Linear(500, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.main(x)
        return x
    
class Encoder(nn.Module):
    def __init__(self, dim_h,n_z):
        super(Encoder, self).__init__()
        """
        The encoder structure in OT_Classifier
        """
        self.dim_h = dim_h
        self.n_z = n_z

        self.main = nn.Sequential(
            nn.Conv2d(1, self.dim_h, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(self.dim_h, self.dim_h * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),
            nn.Conv2d(self.dim_h * 2, self.dim_h * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(True),
            nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 8),
            nn.ReLU(True),
        )
        self.fc = nn.Linear(self.dim_h * (2 ** 3), self.n_z)

    def forward(self, x):
        x = self.main(x)
        x = x.squeeze()
        x = self.fc(x)
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