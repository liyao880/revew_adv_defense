import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.datasets as dset
import torch.utils.data as data
 
## For MNIST
class MNIST:
    def __init__(self, root):
        trans = transforms.Compose([transforms.ToTensor()])
        train_set = dset.MNIST(root=root+'/data', train=True, transform=trans, download=False)
        test_set = dset.MNIST(root=root+'/data', train=False, transform=trans, download=False)
        
        self.train_data = train_set
        self.test_data = test_set    

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

class AdvData(data.Dataset):
    def __init__(self, list_IDs, data, targets, labels):
        self.labels = labels
        self.list_IDs = list_IDs
        self.data = data
        self.targets = targets
        
    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self,index):
        ID = self.list_IDs[index]
                
#        x = self.data[ID].view(1,28,28)
        x = self.data[ID]
        x = x.view(x.shape[1],x.shape[2],x.shape[3])
        y = self.targets[ID]
        z = self.labels[ID]        
        return x, y, z

## For CIFAR10
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
        
class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, vgg_name, nclass=10, img_width=32):
        super(VGG, self).__init__()
        self.img_width = img_width
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, nclass)

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
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out 

def vgg16():
    return VGG('VGG16', nclass=10)

    
    
def loaddata(args):
    if args['dataset'] == 'mnist':
        train_loader = DataLoader(MNIST(args['root']).train_data, batch_size=args['batch_size'], shuffle=True)
        test_loader = DataLoader(MNIST(args['root']).test_data, batch_size=args['batch_size'], shuffle=False)
    elif args['dataset'] == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        
        trainset = datasets.CIFAR10(root=args['root']+"/data",
                                train=True,download=False,transform=transform_train)        
        train_loader = DataLoader(trainset, batch_size=args['batch_size'], shuffle=True)                
        transform_test = transforms.Compose([transforms.ToTensor()])
        testset = datasets.CIFAR10(root=args['root']+"/data",
                                train=False,download=False,transform=transform_test)
        test_loader = DataLoader(testset, batch_size=args['batch_size'], shuffle=False)    
    else:
        print("unknown dataset")
    return train_loader, test_loader


def loadmodel(args):
    if args['dataset'] == 'mnist':
        from setup_mnist import BasicCNN
        model = BasicCNN()           
        if args['init'] != None:
            model.load_state_dict(torch.load('./models/mnist'+args['init']))
    elif args['dataset'] == 'cifar10':       
        from setup_vgg import vgg16
        model = vgg16()
        if args['init'] != None:
            model.load_state_dict(torch.load("./models/cifar10"+args['init']))
    else:
        print("unknown model")
    return model