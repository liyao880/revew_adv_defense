import os,sys
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from setup.setup_pgd import LinfPGDAttack, attack_over_test_data


class MNIST:
    def __init__(self, root):
        trans = transforms.Compose([transforms.ToTensor()])
        train_set = datasets.MNIST(root=root+'/data', train=True, transform=trans, download=False)
        test_set = datasets.MNIST(root=root+'/data', train=False, transform=trans, download=False)
        
        self.train_data = train_set
        self.test_data = test_set   
    
    
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
        from setup.setup_mnist import BasicCNN, NoiseCNN
        if args['model'] == 'cnn':
            model = BasicCNN()
        else:
            model = NoiseCNN(args['std0'],args['std'])
            
        if args['init'] != None:
            model.load_state_dict(torch.load('./models/mnist'+args['init']))
    elif args['dataset'] == 'cifar10':       
        from setup.setup_vgg import vgg16, vgg16noise
        if args['model'] == 'cnn':    
            model = vgg16()
        else:
            model = vgg16noise(args['std0'],args['std'])
        if args['init'] != None:
            model.load_state_dict(torch.load("./models/cifar10"+args['init']))
    else:
        print("unknown model")
        return
    return model
    
        
def testattack(classifier, test_loader, epsilon, k, alpha, use_cuda=True):
    classifier.eval()
    adversary = LinfPGDAttack(classifier, epsilon=epsilon, k=k, a=alpha)
    param = {
    'test_batch_size': 100,
    'epsilon': epsilon,
    }            
    acc = attack_over_test_data(classifier, adversary, param, test_loader, use_cuda=use_cuda)
    return acc

def savefile(file_name, model, dataset):
    if file_name != None:
        root = os.path.abspath(os.path.dirname(sys.argv[0]))+"/models/"+dataset
        if not os.path.exists(root):
            os.mkdir(root)
        torch.save(model.state_dict(), root+file_name)
    return
