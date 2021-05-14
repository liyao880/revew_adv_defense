import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

from setup.utils import loaddata
from setup.utils_ot import loadmodel, free_params, frozen_params, clip_params, testattack, savefile
from setup.setup_ot_pgd import to_var, pred_batch_ot, adv_train_ot, LinfPGDAttackOT

        
def train_ot_classifier(train_loader, test_loader, encoder, discriminator, classifier,
                 use_cuda, n_z, sigma, num_epoch, lr, LAMBDA, LAMBDA0, LAMBDA1, file_name, 
                 epsilon, k, a, delay, print_every, dataset):
    criterion2 = nn.CrossEntropyLoss()
    adversary = LinfPGDAttackOT(epsilon=epsilon,k=k,a=a)
    
    encoder.train()
    discriminator.train()
    classifier.train()
    # Optimizers
    enc_optim = optim.Adam(encoder.parameters(), lr = lr)
    dis_optim = optim.Adam(discriminator.parameters(), lr = 0.5 * lr)
    cla_optim = optim.Adam(classifier.parameters(), lr = 0.05 * lr)

    enc_scheduler = StepLR(enc_optim, step_size=30, gamma=0.5)
    dis_scheduler = StepLR(dis_optim, step_size=30, gamma=0.5)
    cla_scheduler = StepLR(cla_optim, step_size=30, gamma=0.5)
    
    if use_cuda:
        encoder, discriminator, classifier = encoder.cuda(), discriminator.cuda(), classifier.cuda()
    
    one = torch.Tensor([1])
    mone = one * -1
    
    if use_cuda:
        one = one.cuda()
        mone = mone.cuda()
    for epoch in range(num_epoch):
        step = 0

        for images, labels in tqdm(train_loader):
    
            if use_cuda:
                images, labels = images.cuda(), labels.cuda()
            
            # ======== Training ======== #
            
            batch_size = images.size()[0]
                
            encoder.zero_grad()
            discriminator.zero_grad()
            classifier.zero_grad()

            # ======== Get Adversarial images ======== #
            if epoch >= delay:
                target_pred = pred_batch_ot(images, classifier, encoder)
                images_adv = adv_train_ot(images, target_pred, classifier, encoder,  
                                           adversary)
                images_adv = to_var(images_adv)                     
            # ======== Train Discriminator ======== #
            frozen_params(encoder)
            frozen_params(classifier)
            free_params(discriminator)

            z_fake = torch.randn(batch_size, n_z) * sigma
    
            if use_cuda:
                z_fake = z_fake.cuda()

            d_fake = discriminator(to_var(z_fake))
            z_real = encoder(images)
            d_real = discriminator(to_var(z_real))
            
            disc_fake = LAMBDA * d_fake.mean()
            disc_real = LAMBDA * d_real.mean()
            
            disc_fake.backward(one)
            disc_real.backward(mone)
            diss_loss = disc_fake - disc_real
    
            dis_optim.step()
            
            clip_params(discriminator)            
            
            if epoch >= delay:
                z_fake = torch.randn(batch_size, n_z) * sigma
        
                if use_cuda:
                    z_fake = z_fake.cuda()
    
                d_fake = discriminator(to_var(z_fake))
                z_real = encoder(images_adv)
                d_real = discriminator(to_var(z_real))
                
                disc_fake = LAMBDA * d_fake.mean()
                disc_real = LAMBDA * d_real.mean()
                
                disc_fake.backward(one)
                disc_real.backward(mone)
                diss_loss = disc_fake - disc_real
        
                dis_optim.step()
                
                clip_params(discriminator)
            # ======== Train Classifier and Encoder======== #
            free_params(encoder)
            free_params(classifier)
            frozen_params(discriminator)
            
            pred_labels = classifier(encoder(to_var(images)))
            class_loss = LAMBDA0 * criterion2(pred_labels,labels)
            
            if epoch >= delay:    
                pred_labels_adv = classifier(encoder(to_var(images_adv)))
                class_loss_adv =  LAMBDA0 * criterion2(pred_labels_adv,labels)
                class_loss = (class_loss + class_loss_adv)/2
                    
            class_loss.backward()
            
            cla_optim.step()
            enc_optim.step()         
            
#            # ======== Train Encoder ======== #
            free_params(encoder)
            frozen_params(classifier)
            frozen_params(discriminator)
            
            z_real = encoder(images)
            d_real = discriminator(encoder(Variable(images.data)))
            
            d_loss = LAMBDA1 * (d_real.mean())
            
            d_loss.backward(one)
            
            enc_optim.step()  
            
            if epoch >= delay:
                z_real = encoder(images_adv)
                d_real = discriminator(encoder(Variable(images_adv.data)))
        
                d_loss = LAMBDA1 * (d_real.mean())                    
                d_loss.backward(one)
                
                enc_optim.step()      
                            
            step += 1
    
            if (step + 1) % print_every == 0:
                print("Epoch: [%d/%d], Step: [%d/%d], Discriminative Loss: %.4f, Classification_Loss:%.4f" %
                      (epoch + 1, num_epoch, step + 1, len(train_loader), diss_loss.data.item(),class_loss.data.item()))
             
        if (epoch + 1) % 1 == 0:
            savefile(file_name, encoder, discriminator, classifier, dataset=dataset)
            test(test_loader, classifier, encoder=encoder, use_cuda=True)
            
    savefile(file_name, encoder, discriminator, classifier, dataset=dataset)
    return classifier, encoder

def test(test_loader, model, encoder, use_cuda=True):
    model.eval()
    encoder.eval()
    correct_cnt = 0
    total_cnt = 0
    for batch_idx, (x, target) in enumerate(test_loader):
        if use_cuda:
            x, target = x.cuda(), target.cuda()
        x, target = Variable(x), Variable(target)
        out = model(encoder(x))
        _, pred_label = torch.max(out.data, 1)
        total_cnt += x.data.size()[0]
        correct_cnt += (pred_label == target.data).sum()        
    acc = float(correct_cnt.double()/total_cnt)
    print("The prediction accuracy on testset is {}".format(acc))
    return acc
 
def main(args):
    use_cuda = torch.cuda.is_available()
    print('==> Loading data..')
    train_loader, test_loader = loaddata(args)
    
    print('==> Loading model..')
    encoder, discriminator, classifier = loadmodel(args)

    print('==> Training starts..')
    torch.manual_seed(123)
    classifier, encoder = train_ot_classifier(train_loader, test_loader, encoder, discriminator, classifier,
                 use_cuda=use_cuda, n_z=args['n_z'], sigma=args['sigma'],num_epoch=args['epochs'], 
                 lr=args['lr'], LAMBDA=args['LAMBDA'], LAMBDA0=args['LAMBDA0'],
                 LAMBDA1=args['LAMBDA1'],delay=args['delay'],
                 file_name=args['file_name'],epsilon=args['epsilon'],k=args['k'],a=args['a'],
                 print_every=args['print_every'], dataset=args['dataset'])
    
    test(test_loader, classifier, encoder=encoder, use_cuda=True)
    print('==> Testing the model against PGD attack..')
    testattack(classifier, encoder, test_loader, epsilon=args['epsilon'], 
               k=args['k'], a=args['a'], use_cuda=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Optimal Transport Classifier')
    parser.add_argument('-d', '--dataset', choices=["mnist", "cifar10"], default="mnist")    
    parser.add_argument('-epochs', type=int, default=30, help='number of epochs to train (default: 30)')
    parser.add_argument('-file_name', required=True)
    parser.add_argument('-init', default=None)
    parser.add_argument('-root', required=True)
    parser.add_argument('-delay', type=int, default=200)
    parser.add_argument('-LAMBDA', type=float, default=1.0)
    parser.add_argument('-LAMBDA0', type=float, default=1.0)    
    parser.add_argument('-LAMBDA1', type=float, default=0.1)
    parser.add_argument('-sigma', type=float, default=1, help='variance of hidden dimension (default: 1)')
    parser.add_argument('-dim_h', type=int, default=128, help='hidden dimension (default: 128)')
    parser.add_argument('-dim_h1', type=int, default=128, help='hidden dimension (default: 32) of encoder')
    parser.add_argument('-lr', type=float, default=5e-5, help='learning rate (default: 0.0001)')
    args = vars(parser.parse_args())
    if args['dataset'] == 'mnist':
        args['a'] = 0.02
        args['k'] = 40
        args['epsilon'] = 0.3
        args['n_z'] = 4
        args['batch_size'] = 100
        args['print_every'] = 300
        args['lr'] = 1e-4
        args['LAMBDA1'] = 0.0005
    elif args['dataset'] == 'cifar10':
        args['a'] = 0.01
        args['k'] = 20
        args['epsilon'] = 0.03
        args['n_z'] = 16
        args['batch_size'] = 100
        args['print_every'] = 250
    else:
        print('invalid dataset')
    print(args)
    main(args)

