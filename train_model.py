import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
from setup.trades import trades_loss
from setup.utils import loaddata, loadmodel, savefile
from setup.setup_pgd import to_var, pred_batch, adv_train, LinfPGDAttack, attack_over_test_data

    
def trainClassifier(model, train_loader, test_loader, args, use_cuda=True):
 
    adversary = LinfPGDAttack(epsilon=args['epsilon'], k=args['num_k'], a=args['alpha'])
    
    if use_cuda:
        model = model.cuda()

    optimizer = torch.optim.SGD(model.parameters(),lr=args['lr'],momentum=0.9, weight_decay=args['weight_decay'])
    
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(args['num_epoch']):
        # trainning
        ave_loss = 0
        step = 0
        for x, target in tqdm(train_loader):            
            x, target = to_var(x), to_var(target)
            loss = criterion(model(x),target)     
            
            # Adversarial training or no defense training

            if args['method'] == 'madry':
                target_pred = pred_batch(x, model)
                x_adv = adv_train(x, target_pred, model, criterion, adversary)
                x_adv = to_var(x_adv)
                loss = criterion(model(x_adv),target)
            elif args['method'] == 'trades':
                loss = trades_loss(model=model,
                                   x_natural=x,
                                   y=target,
                                   optimizer=optimizer,
                                   step_size=args['alpha'],
                                   epsilon=args['epsilon'],
                                   perturb_steps=args['num_k'],
                                   beta=args['beta'],
                                   distance='l_inf')
            else:
                loss = criterion(model(x),target)

                
            ave_loss = ave_loss * 0.9 + loss.item() * 0.1    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            if (step + 1) % args['print_every'] == 0:
                print("Epoch: [%d/%d], step: [%d/%d], Average Loss: %.4f" %
                      (epoch + 1, args['num_epoch'], step + 1, len(train_loader), ave_loss))
    savefile(args['file_name'], model, args['dataset'])
    return model


def testClassifier(test_loader, model, use_cuda=True, batch_size=100):
    model.eval()
    correct_cnt = 0
    total_cnt = 0
    for batch_idx, (x, target) in enumerate(test_loader):
        if use_cuda:
            x, target = x.cuda(), target.cuda()
        x, target = Variable(x), Variable(target)
        out = model(x)
        _, pred_label = torch.max(out.data, 1)
        total_cnt += x.data.size()[0]
        correct_cnt += (pred_label == target.data).sum()
    acc = float(correct_cnt.double()/total_cnt)
    print("The prediction accuracy on testset is {}".format(acc))
    return acc


def testattack(classifier, test_loader, epsilon, k, alpha, batch_size, use_cuda=True):
    classifier.eval()
    adversary = LinfPGDAttack(classifier, epsilon=epsilon, k=k, a=alpha)
    param = {
    'test_batch_size': batch_size,
    'epsilon': epsilon,
    }            
    acc = attack_over_test_data(classifier, adversary, param, test_loader, use_cuda=use_cuda)
    return acc


def main(args):
    use_cuda = torch.cuda.is_available()
    print('==> Loading data..')
    train_loader, test_loader = loaddata(args)
    
    print('==> Loading model..')
    model = loadmodel(args)

    print('==> Training starts..')            
    model = trainClassifier(model, train_loader, test_loader, args, use_cuda=use_cuda)
    testClassifier(test_loader,model,use_cuda=use_cuda,batch_size=args['batch_size'])
    testattack(model, test_loader, epsilon=args['epsilon'], k=args['num_k'], alpha=args['alpha'], 
               batch_size=args['batch_size'], use_cuda=use_cuda)
    

    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Training defense models')
    parser.add_argument("-d", '--dataset', choices=["mnist", "cifar10"], default="mnist")   
    parser.add_argument("-n", "--num_epoch", type=int, default=60)
    parser.add_argument("-f", "--file_name", required=True)
    parser.add_argument("-l", "--lr", type=float, default=1e-4)
    parser.add_argument("--model", default="cnn", choices=["cnn", "noise"])
    parser.add_argument("--method", default="no_defense", choices=["no_defense", "madry", "trades"])
    parser.add_argument("--init", default=None)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--root", required=True)
    args = vars(parser.parse_args())
    if args['dataset'] == 'mnist':
        args['alpha'] = 0.02
        args['num_k'] = 40
        args['epsilon'] = 0.3
        args['batch_size'] = 100
        args['std0'] = 0.8
        args['std'] = 0.4
        args['print_every'] = 300
    elif args['dataset'] == 'cifar10':
        args['alpha'] = 0.01
        args['num_k'] = 20
        args['epsilon'] = 0.03
        args['std0'] = 0.2
        args['std'] = 0.1
        args['batch_size'] = 100
        args['print_every'] = 250
    else:
        print('invalid dataset')
    print(args)
    main(args)
