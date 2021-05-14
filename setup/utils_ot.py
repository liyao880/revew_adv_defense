import os,sys
import torch
import torch.nn as nn

from setup_ot_pgd import LinfPGDAttackOT, attack_over_test_data_ot

def loadmodel(args):
    if args['dataset'] == 'mnist':
        from setup_ot_mnist import Encoder, Discriminator, SmallCNN
        encoder = Encoder(args['dim_h'],args['n_z'])
        discriminator = Discriminator(args['dim_h'],args['n_z'])
        classifier = SmallCNN(args['n_z'])        
        if args['init'] != None:
            classifier.load_state_dict(torch.load('./models/mnist'+args['init']+'cla'))
            discriminator.load_state_dict(torch.load('./models/mnist'+args['init']+'dis'))
            encoder.load_state_dict(torch.load('./models/mnist'+args['init']+'enc'))
    elif args['dataset'] == 'cifar10':       
        from setup_ot_cifar10 import Encoder, Discriminator, SmallCNN        
        encoder = Encoder(args['n_z'])    
        discriminator = Discriminator(args['dim_h'],args['n_z'])
        classifier = SmallCNN(args['n_z'])
        if args['init'] != None:
            classifier.load_state_dict(torch.load('./models/cifar10'+args['init']+'cla'))
            encoder.load_state_dict(torch.load('./models/cifar10'+args['init']+'enc'))
            discriminator.load_state_dict(torch.load('./models/cifar10'+args['init']+'dis'))
    elif args['dataset'] == 'stl10':
        from setup_ot_stl10 import Encoder, Discriminator, SmallCNN
        encoder = Encoder(args['n_z'],args['dim_h1'])
        discriminator = Discriminator(args['dim_h'],args['n_z'])
        classifier = SmallCNN(args['n_z'])
        if args['init'] != None:
            classifier.load_state_dict(torch.load('./models/stl10'+args['init']+'cla'))
            encoder.load_state_dict(torch.load('./models/stl10'+args['init']+'enc'))
            discriminator.load_state_dict(torch.load('./models/stl10'+args['init']+'dis'))
    else:
        print("unknown model")
    return encoder, discriminator, classifier

def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True

def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False

def clip_params(module: nn.Module):
    for p in module.parameters():
        p.data.clamp_(-0.01, 0.01)     
        
def testattack(classifier, encoder, test_loader, epsilon, k, a, use_cuda=True):
    classifier.eval()
    encoder.eval()
    adversary = LinfPGDAttackOT(classifier, encoder, epsilon=epsilon, k=k, a=a)
    param = {
    'test_batch_size': 100,
    'epsilon': epsilon,
    }            
    acc = attack_over_test_data_ot(classifier, encoder, adversary, param, test_loader, use_cuda=use_cuda)
    return acc

def savefile(file_name, encoder, discriminator, classifier, dataset):
    if file_name != None:
        root = os.path.abspath(os.path.dirname(sys.argv[0]))+"/models/"+dataset
        if not os.path.exists(root):
            os.mkdir(root)
        torch.save(encoder.state_dict(), root+file_name+"enc")
        torch.save(discriminator.state_dict(), root+file_name+"dis")
        torch.save(classifier.state_dict(), root+file_name+"cla")
    return
