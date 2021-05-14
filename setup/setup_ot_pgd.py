"""
Setup PGD attack

Modified from: https://github.com/wanglouis49/pytorch-adversarial_box
"""
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm

def to_var(x, requires_grad=False, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad, volatile=volatile)

def pred_batch_ot(x, model, encoder):
    """
    batch prediction helper
    """
    z = encoder(to_var(x))
    y_pred = np.argmax(model(z).data.cpu().numpy(), axis=1)
    return torch.from_numpy(y_pred)

class LinfPGDAttackOT(object):
    def __init__(self, model=None, encoder=None, epsilon=0.3, k=40, a=0.01, 
        random_start=True):
        """
        Attack parameter initialization. The attack performs k steps of
        size a, while always staying within epsilon from the initial
        point.
        https://github.com/MadryLab/mnist_challenge/blob/master/pgd_attack.py
        """
        self.model = model
        self.encoder = encoder
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.rand = random_start
        self.loss_fn = nn.CrossEntropyLoss()

    def perturb(self, X_nat, y):
        """
        Given examples (X_nat, y), returns adversarial
        examples within epsilon of X_nat in l_infinity norm.
        """
        if torch.cuda.is_available():
            X_nat, y = X_nat.cuda(), y.cuda()
            
        if self.rand:
            rand = torch.Tensor(X_nat.shape).uniform_(-self.epsilon, self.epsilon)
            if torch.cuda.is_available():
                rand = rand.cuda()
            X = X_nat + rand
        else:
            X = X_nat.clone()
        
        y_var = to_var(y)

        for i in range(self.k):
            X_var = to_var(X, requires_grad=True)
            scores = self.model(self.encoder(X_var))
            loss = self.loss_fn(scores, y_var)
            loss.backward()
            grad = X_var.grad

            X += self.a * torch.sign(grad)
            
            diff = X - X_nat
            diff.clamp_(-self.epsilon, self.epsilon)
            
            X.detach().copy_((diff + X_nat).clamp_(0, 1))                    
        return X
    
def attack_over_test_data_ot(model, encoder, adversary, param, loader_test,
                              use_cuda=True, oracle=None, verbose=True):
    """
    Given target model computes accuracy on perturbed data
    """
    adversary.model = model
    adversary.encoder = encoder
    adv_predictor = model if oracle is None else oracle
    total_correct = 0
    ntested = 0
    total_samples = len(loader_test.dataset)

    # For black-box
    if oracle is not None:
        total_samples -= param['hold_out_size']

    pbar = tqdm(loader_test)
    for X, y in pbar:
        y_pred = pred_batch_ot(X, model, encoder)
        X_adv = adversary.perturb(X, y_pred)
        y_pred_adv = pred_batch_ot(X_adv, adv_predictor, encoder)
        ntested += y.size()[0]
        total_correct += (y_pred_adv.numpy() == y.numpy()).sum()
        pbar.set_postfix(adv_acc="{0}/{1} {2:-6.2f}%".format(total_correct, ntested,
                                                             total_correct*100.0/ntested),
                         refresh=False)

    pbar.close()
    acc = total_correct/total_samples
    if verbose: print('Got %d/%d correct (%.2f%%) on the perturbed data' 
          % (total_correct, total_samples, 100 * acc), flush=True)
    return acc

def adv_train_ot(X, y, model, encoder, adversary):
    """
    Adversarial training. Returns pertubed mini batch.
    """
    model_cp = copy.deepcopy(model)
    for p in model_cp.parameters():
        p.requires_grad = False
    model_cp.eval()

    encoder_cp = copy.deepcopy(encoder)
    for p in encoder_cp.parameters():
        p.requires_grad = False
    encoder_cp.eval()
    
    adversary.model = model_cp
    adversary.encoder = encoder_cp
    
    X_adv = adversary.perturb(X, y)

    return X_adv
