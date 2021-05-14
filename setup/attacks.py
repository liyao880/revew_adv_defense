import time, torch
import numpy as np
import torch.nn as nn
from numpy import linalg as LA
from torch.autograd import Variable


class FGSM(object):
    def __init__(self,model):
        self.model = model

    def get_loss(self,xi,label_or_target,TARGETED):
        criterion = nn.CrossEntropyLoss()
        output = self.model.predict(xi)
        #print(output, label_or_target)
        loss = criterion(output, label_or_target)
        #print(loss)
        #print(c.size(),modifier.size())
        return loss

    def i_fgsm(self, input_xi, label_or_target, eta, TARGETED=False):
       
        yi = Variable(label_or_target.cuda())
        x_adv = Variable(input_xi.cuda(), requires_grad=True)
        for it in range(10):
            error = self.get_loss(x_adv,yi,TARGETED)
            if (it)%1==0:
                print(error.item()) 
            self.model.get_gradient(error)
            #print(gradient)
            x_adv.grad.sign_()
            if TARGETED:
                x_adv.data = x_adv.data - eta* x_adv.grad 
            else:
                x_adv.data = x_adv.data + eta* x_adv.grad
            #x_adv = Variable(x_adv.data, requires_grad=True)
            #error.backward()
        return x_adv

    def fgsm(self, input_xi, label_or_target, eta, TARGETED=False):
       
        yi = Variable(label_or_target.cuda())
        x_adv = Variable(input_xi.cuda(), requires_grad=True)

        error = self.get_loss(x_adv,yi,TARGETED)
        print(error.item()) 
        self.model.get_gradient(error)
        #print(gradient)
        x_adv.grad.sign_()
        if TARGETED:
            x_adv.data = x_adv.data - eta* x_adv.grad 
        else:
            x_adv.data = x_adv.data + eta* x_adv.grad
            #x_adv = Variable(x_adv.data, requires_grad=True)
            #error.backward()
        return x_adv 

    def __call__(self, input_xi, label_or_target, eta=0.01, TARGETED=False, ITERATIVE=False, epsilon=None):
        if ITERATIVE:
            adv = self.i_fgsm(input_xi, label_or_target, eta, TARGETED)
        else:
            eta = epsilon
            adv = self.fgsm(input_xi, label_or_target, eta, TARGETED)
        return adv   
    
    
class PGD(object):
    def __init__(self,model):
        self.model = model
    
    def random_start(self,x,eps):
        x+=torch.FloatTensor(x.size()).uniform_(-eps,eps).cuda()
        x.clamp_(0,1)
        return x

    def get_loss(self,xi,label_or_target,TARGETED):
        criterion = nn.CrossEntropyLoss()
        output = self.model.predict(xi)
        #print(output.shape)
        #print(output, label_or_target)
        loss = criterion(output, label_or_target)
        #print(loss)
        #print(c.size(),modifier.size())
        return loss
    
    def pgd(self, input_xi, label_or_target, epsilon, eta, TARGETED=False, random_start=True):
        #yi = Variable(label_or_target)
        #x_adv = Variable(input_xi.cuda(), requires_grad=True)
        yi = label_or_target
        x_adv = input_xi.clone()
        if random_start:
            x_adv = self.random_start(x_adv,epsilon)
        x_adv.requires_grad = True
        for it in range(100):
            error = self.get_loss(x_adv,yi, TARGETED)
            # if (it)%10==0:
            # print(error.data.item()) 
            #x_adv.grad.data.zero_()
            #error.backward(retain_graph=True)
            #print(error.requires_grad)
            self.model.get_gradient(error)
            x_adv.grad.sign_()
            if TARGETED:
                x_adv.data = x_adv.data - eta* epsilon * x_adv.grad.data
            else:
                x_adv.data = x_adv.data + eta* epsilon * x_adv.grad.data
            diff = x_adv.data - input_xi
            diff.clamp_(-epsilon,epsilon)
            x_adv.data=(diff + input_xi).clamp_(0, 1)
            x_adv.grad.data.zero_()
        return x_adv

    def __call__(self, input_xi, label_or_target, epsilon=0.01, eta=0.5, TARGETED=False):
        adv = self.pgd(input_xi, label_or_target, epsilon, eta, TARGETED)
        return adv  
    
    
MAX_ITER = 1000

class OPT_attack(object):
    def __init__(self,model):
        self.model = model
        self.log = torch.ones(MAX_ITER,2)
    
    def get_log(self):
        return self.log
        
    def attack_untargeted(self, x0, y0, alpha = 0.2, beta = 0.001, iterations = 1500, query_limit=80000):
        """ Attack the original image and return adversarial example
            model: (pytorch model)
            train_dataset: set of training data
            (x0, y0): original image
        """
        model = self.model
        if type(x0) is torch.Tensor:
            x0 = x0.cpu().numpy()
        if type(y0) is torch.Tensor:
            y0 = y0.item()
        if (model.predict_label(x0) != y0):
            print("Fail to classify the image. No need to attack.")
            return torch.tensor(x0).cuda()

        num_directions = 100
        best_theta, g_theta = None, float('inf')
        query_count = 0
        print("Searching for the initial direction on %d random directions: " % (num_directions))
        np.random.seed(0)
        timestart = time.time()
        for i in range(num_directions):
            query_count += 1
            theta = np.random.randn(*x0.shape)
            if model.predict_label(x0+theta)!=y0:
                initial_lbd = LA.norm(theta)
                theta /= initial_lbd
                lbd, count = self.fine_grained_binary_search(model, x0, y0, theta, initial_lbd, g_theta)
                query_count += count
                if lbd < g_theta:
                    best_theta, g_theta = theta, lbd
                    print("--------> Found distortion %.4f" % g_theta)
        if g_theta == float('inf'):
            num_directions = 500
            best_theta, g_theta = None, float('inf')
            print("Searching for the initial direction on %d random directions: " % (num_directions))
            timestart = time.time()
            for i in range(num_directions):
                query_count += 1
                theta = np.random.randn(*x0.shape)
                if model.predict_label(x0+theta)!=y0:
                    initial_lbd = LA.norm(theta)
                    theta /= initial_lbd
                    lbd, count = self.fine_grained_binary_search(model, x0, y0, theta, initial_lbd, g_theta)
                    query_count += count
                    if lbd < g_theta:
                        best_theta, g_theta = theta, lbd
                        print("--------> Found distortion %.4f" % g_theta)

        if g_theta == float('inf'):    
            print("Couldn't find valid initial, failed")
            return torch.tensor(x0).cuda()
        timeend = time.time()
        print("==========> Found best distortion %.4f in %.4f seconds using %d queries" % (g_theta, timeend-timestart, query_count))    
        self.log[0][0], self.log[0][1] = g_theta, query_count
        
        
        timestart = time.time()
        g1 = 1.0
        theta, g2 = best_theta, g_theta
        opt_count = 0
        stopping = 0.01
        prev_obj = 100000
        for i in range(iterations):
            gradient = np.zeros(theta.shape)
            q = 10
            min_g1 = float('inf')
            for _ in range(q):
                u = np.random.randn(*theta.shape)
                u /= LA.norm(u)
                ttt = theta+beta * u
                ttt /= LA.norm(ttt)
                g1, count = self.fine_grained_binary_search_local(model, x0, y0, ttt, initial_lbd = g2, tol=beta/500)
                opt_count += count
                gradient += (g1-g2)/beta * u
                if g1 < min_g1:
                    min_g1 = g1
                    min_ttt = ttt
            gradient = 1.0/q * gradient

            if opt_count > query_limit:
                break

            if (i+1)%10 == 0:
                print("Iteration %3d distortion %.4f num_queries %d" % (i+1, LA.norm(g2*theta), opt_count))
                #if g2 > prev_obj-stopping:
                #    break
                prev_obj = g2
            self.log[i+1][0], self.log[i+1][1] = g2, opt_count + query_count

            min_theta = theta
            min_g2 = g2
        
            for _ in range(15):
                new_theta = theta - alpha * gradient
                new_theta /= LA.norm(new_theta)
                new_g2, count = self.fine_grained_binary_search_local(model, x0, y0, new_theta, initial_lbd = min_g2, tol=beta/500)
                opt_count += count
                alpha = alpha * 2
                if new_g2 < min_g2:
                    min_theta = new_theta 
                    min_g2 = new_g2
                else:
                    break

            if min_g2 >= g2:
                for _ in range(15):
                    alpha = alpha * 0.25
                    new_theta = theta - alpha * gradient
                    new_theta /= LA.norm(new_theta)
                    new_g2, count = self.fine_grained_binary_search_local(model, x0, y0, new_theta, initial_lbd = min_g2, tol=beta/500)
                    opt_count += count
                    if new_g2 < g2:
                        min_theta = new_theta 
                        min_g2 = new_g2
                        break

            if min_g2 <= min_g1:
                theta, g2 = min_theta, min_g2
            else:
                theta, g2 = min_ttt, min_g1

            if g2 < g_theta:
                best_theta, g_theta = theta, g2
            
            #print(alpha)
            if alpha < 1e-4:
                alpha = 1.0
                print("Warning: not moving, g2 %lf gtheta %lf" % (g2, g_theta))
                beta = beta * 0.1
                if (beta < 1e-8):
                   break

        target = model.predict_label(x0 + g_theta*best_theta)
        timeend = time.time()
        print("\nAdversarial Example Found Successfully: distortion %.4f target %d queries %d \nTime: %.4f seconds" % (g_theta, target, query_count + opt_count, timeend-timestart))
        
        self.log[i+1:,0] = g_theta
        self.log[i+1:,1] = opt_count + query_count
        is_success = (target != y0)
        return torch.tensor(x0 + g_theta*best_theta, dtype=torch.float).cuda(), query_count + opt_count, is_success

    def fine_grained_binary_search_local(self, model, x0, y0, theta, initial_lbd = 1.0, tol=1e-5):
        nquery = 0
        lbd = initial_lbd
         
        if model.predict_label(x0+lbd*theta) == y0:
            lbd_lo = lbd
            lbd_hi = lbd*1.01
            nquery += 1
            while model.predict_label(x0+lbd_hi*theta) == y0:
                lbd_hi = lbd_hi*1.01
                nquery += 1
                if lbd_hi > 20:
                    return float('inf'), nquery
        else:
            lbd_hi = lbd
            lbd_lo = lbd*0.99
            nquery += 1
            while model.predict_label(x0+lbd_lo*theta) != y0 :
                lbd_lo = lbd_lo*0.99
                nquery += 1

        while (lbd_hi - lbd_lo) > tol:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if model.predict_label(x0 + lbd_mid*theta) != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        return lbd_hi, nquery

    def fine_grained_binary_search(self, model, x0, y0, theta, initial_lbd, current_best):
        nquery = 0
        if initial_lbd > current_best: 
            if model.predict_label(x0+current_best*theta) == y0:
                nquery += 1
                return float('inf'), nquery
            lbd = current_best
        else:
            lbd = initial_lbd
        
        lbd_hi = lbd
        lbd_lo = 0.0

        while (lbd_hi - lbd_lo) > 1e-5:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if model.predict_label(x0 + lbd_mid*theta) != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        return lbd_hi, nquery


    def __call__(self, input_xi, label_or_target, iterations=1500, TARGETED=False, epsilon=None):
        if TARGETED:
            print("Not Implemented.")
        else:
            adv = self.attack_untargeted(input_xi, label_or_target, iterations = iterations)
        return adv   


class ZOO(object):
    def __init__(self,model):
        self.model = model
        
    def get_loss(self,xi,label_onehot_v, c, modifier, TARGETED):
        #print(c.size(),modifier.size())
        loss1 = c*torch.sum(modifier*modifier)
        #output = net(torch.clamp(xi+modifier,0,1))
        output = self.model.predict(xi+modifier)
        real = torch.max(torch.mul(output, label_onehot_v), 1)[0]
        other = torch.max(torch.mul(output, (1-label_onehot_v))-label_onehot_v*10000,1)[0]
        #print(real,other)
        if TARGETED:
            loss2 = torch.sum(torch.clamp(other - real, min=0))
        else:
            loss2 = torch.sum(torch.clamp(real - other, min=0))
        error = loss2 + loss1 
        return error,loss1,loss2

    def zoo(self, input_xi, label_or_target, max_iter, c, TARGETED=False):
        step_size = 0.1
        modifier = Variable(torch.zeros(input_xi.size()).cuda())
        yi = label_or_target
        label_onehot = torch.FloatTensor(yi.size()[0],self.model.num_classes)
        label_onehot.zero_()
        label_onehot.scatter_(1,yi.view(-1,1),1)
        label_onehot_v = Variable(label_onehot, requires_grad=False).cuda()
        xi = Variable(input_xi.cuda())
        #optimizer = optim.Adam([modifier], lr = 0.1)
        best_loss1 = 1000
        best_adv = None
        num_coor = 1
        delta = 0.0001
        for it in range(max_iter):
            #optimizer.zero_grad()
            error1,loss11,loss12 = self.get_loss(xi,label_onehot_v,c,modifier, TARGETED)
            for j in range(num_coor):
                modifier = Variable(torch.zeros(xi.size()).cuda(), volatile=True)
                randx = np.random.randint(xi.size()[0])
                randy = np.random.randint(xi.size()[1])
                randz = np.random.randint(xi.size()[2])
                modifier[randx,randy,randz] = delta
                #print(modifier)
                new_xi = xi + modifier
                error2,loss21,loss22 = self.get_loss(new_xi,label_onehot_v,c,modifier, TARGETED)
                modifier_gradient = (error2 - error1) / delta * modifier
                modifier -= step_size*modifier_gradient
            xi = xi + modifier
            #self.model.get_gradient(error)
            #error.backward()
            #optimizer.step()
            if (it)%1000==0:
                print(error1.data[0],loss11.data[0],loss12.data[0]) 
        return xi
    
    def random_zoo(self, input_xi, label_or_target, max_iter, c, TARGETED=False):
        step_size = 5e-3    
        modifier = Variable(torch.zeros(input_xi.size()).cuda())
        yi = label_or_target
        label_onehot = torch.FloatTensor(yi.size()[0],self.model.num_classes).cuda()
        label_onehot.zero_()
        label_onehot.scatter_(1,yi.view(-1,1),1)
        label_onehot_v = Variable(label_onehot, requires_grad=False).cuda()
        xi = Variable(input_xi.cuda(),requires_grad=False)
        #optimizer = optim.Adam([modifier], lr = 0.1)
        best_loss1 = 1000
        best_adv = None
        num_coor = 1
        delta = 1e-6
        modifier = Variable(torch.zeros(xi.size()).cuda(), volatile=True)
        for it in range(max_iter):
            #optimizer.zero_grad()
            error1,loss11,loss12 = self.get_loss(xi,label_onehot_v,c,modifier, TARGETED)
            u=torch.randn(xi.size()).cuda()
            error2,loss21,loss22 = self.get_loss(xi,label_onehot_v,c,modifier+delta*u, TARGETED)
            modifier_gradient = (error2 - error1) / delta * u
            modifier.data = modifier.data - step_size*modifier_gradient
            #xi = xi + modifier
            #self.model.get_gradient(error)
            #error.backward()
            #optimizer.step()
            # if (it)%1000==0:
            #     print(it,error1.item(),loss11.item(),loss12.item()) 
        return xi        

    def __call__(self, input_xi, label_or_target, max_iter, c=0.1, TARGETED=False):
        adv = self.random_zoo(input_xi, label_or_target, max_iter, c, TARGETED)
        return adv



start_learning_rate = 1.0

def quad_solver(Q, b):
    """
    Solve min_a  0.5*aQa + b^T a s.t. a>=0
    """
    K = Q.shape[0]
    alpha = np.zeros((K,))
    g = b
    Qdiag = np.diag(Q)
    for i in range(20000):
        delta = np.maximum(alpha - g/Qdiag,0) - alpha
        idx = np.argmax(abs(delta))
        val = delta[idx]
        if abs(val) < 1e-7: 
            break
        g = g + val*Q[:,idx]
        alpha[idx] += val
    return alpha

def sign(y):
    """
    y -- numpy array of shape (m,)
    Returns an element-wise indication of the sign of a number.
    The sign function returns -1 if y < 0, 1 if x >= 0. nan is returned for nan inputs.
    """
    y_sign = np.sign(y)
    y_sign[y_sign==0] = 1
    return y_sign


class OPT_attack_sign_SGD(object):
    def __init__(self, model, k=200, train_dataset=None):
        self.model = model
        self.k = k
        self.train_dataset = train_dataset 
        self.log = torch.ones(MAX_ITER,2)

    def get_log(self):
        return self.log
    
    def attack_untargeted(self, x0, y0, alpha = 0.2, beta = 0.001, iterations = 1000, query_limit=20000,
                          distortion=None, svm=False, momentum=0.0, stopping=0.0001):
        """ Attack the original image and return adversarial example
            model: (pytorch model)
            train_dataset: set of training data
            (x0, y0): original image
        """

        model = self.model
        y0 = y0[0]
        query_count = 0
        ls_total = 0
        
        if (model.predict_label(x0) != y0):
            print("Fail to classify the image. No need to attack.")
            return x0, 0, True, 0, None

        #### init: Calculate a good starting point (direction)
        num_directions = 100
        best_theta, g_theta = None, float('inf')
        print("Searching for the initial direction on %d random directions: " % (num_directions))
        timestart = time.time()
        for i in range(num_directions):
            query_count += 1
            theta = np.random.randn(*x0.shape) # gaussian distortion
            # register adv directions
            if model.predict_label(x0+torch.tensor(theta, dtype=torch.float).cuda()) != y0: 
                initial_lbd = LA.norm(theta)
                theta /= initial_lbd # l2 normalize
                lbd, count = self.fine_grained_binary_search(model, x0, y0, theta, initial_lbd, g_theta)
                query_count += count
                if lbd < g_theta:
                    best_theta, g_theta = theta, lbd
                    print("--------> Found distortion %.4f" % g_theta)
        timeend = time.time()
        
        ## fail if cannot find a adv direction within 200 Gaussian
        if g_theta == float('inf'):
            print("Couldn't find valid initial, failed")
            return x0, 0, False, query_count, best_theta
        print("==========> Found best distortion %.4f in %.4f seconds "
              "using %d queries" % (g_theta, timeend-timestart, query_count))
        self.log[0][0], self.log[0][1] = g_theta, query_count

        
        #### Begin Gradient Descent.
        timestart = time.time()
        xg, gg = best_theta, g_theta
        vg = np.zeros_like(xg)
        learning_rate = start_learning_rate
        prev_obj = 100000
        distortions = [gg]
        for i in range(iterations):
            ## gradient estimation at x0 + theta (init)
            if svm == True:
                sign_gradient, grad_queries = self.sign_grad_svm(x0, y0, xg, initial_lbd=gg, h=beta)
            else:
                sign_gradient, grad_queries = self.sign_grad_v1(x0, y0, xg, initial_lbd=gg, h=beta)

            ## Line search of the step size of gradient descent
            ls_count = 0
            min_theta = xg ## next theta
            min_g2 = gg ## current g_theta
            min_vg = vg ## velocity (for momentum only)
            for _ in range(15):
                # update theta by one step sgd
                if momentum > 0:
                    new_vg = momentum*vg - alpha*sign_gradient
                    new_theta = xg + new_vg
                else:
                    new_theta = xg - alpha * sign_gradient
                new_theta /= LA.norm(new_theta)

                new_g2, count = self.fine_grained_binary_search_local(
                    model, x0, y0, new_theta, initial_lbd = min_g2, tol=beta/500)
                ls_count += count
                alpha = alpha * 2 # gradually increasing step size
                if new_g2 < min_g2:
                    min_theta = new_theta
                    min_g2 = new_g2
                    if momentum > 0:
                        min_vg = new_vg
                else:
                    break

            if min_g2 >= gg: ## if the above code failed for the init alpha, we then try to decrease alpha
                for _ in range(15):
                    alpha = alpha * 0.25
                    if momentum > 0:
                        new_vg = momentum*vg - alpha*sign_gradient
                        new_theta = xg + new_vg
                    else:
                        new_theta = xg - alpha * sign_gradient
                    new_theta /= LA.norm(new_theta)
                    new_g2, count = self.fine_grained_binary_search_local(
                        model, x0, y0, new_theta, initial_lbd = min_g2, tol=beta/500)
                    ls_count += count
                    if new_g2 < gg:
                        min_theta = new_theta 
                        min_g2 = new_g2
                        if momentum > 0:
                            min_vg = new_vg
                        break

            if alpha < 1e-4:  ## if the above two blocks of code failed
                alpha = 1.0
                print("Warning: not moving")
                beta = beta*0.1
                if (beta < 1e-8):
                    break
            
            ## if all attemps failed, min_theta, min_g2 will be the current theta (i.e. not moving)
            xg, gg = min_theta, min_g2
            vg = min_vg

            query_count += (grad_queries + ls_count)
            ls_total += ls_count
            distortions.append(gg)

            if query_count > query_limit:
                break

            ## logging
            if (i + 1) % 10 == 0:
                print("Iteration %3d distortion %.4f num_queries %d" % (i+1, gg, query_count))
            self.log[i+1][0], self.log[i+1][1] = gg, query_count
            #if distortion is not None and gg < distortion:
            #    print("Success: required distortion reached")
            #    break

        if distortion is None or gg < distortion:
            target = model.predict_label(x0 + torch.tensor(gg*xg, dtype=torch.float).cuda())
            print("Succeed distortion {:.4f} target"
                  " {:d} queries {:d} LS queries {:d}\n".format(gg, target, query_count, ls_total))
            return x0 + torch.tensor(gg*xg, dtype=torch.float).cuda(), gg, True, query_count, xg

        timeend = time.time()
        print("\nFailed: distortion %.4f" % (gg))
        
        self.log[i+1:,0] = gg
        self.log[i+1:,1] = query_count
        return x0 + torch.tensor(gg*xg, dtype=torch.float).cuda(), gg, False, query_count, xg


    def sign_grad_v1(self, x0, y0, theta, initial_lbd, h=0.001, D=4, target=None):
        """
        Evaluate the sign of gradient by formulat
        sign(g) = 1/Q [ \sum_{q=1}^Q sign( g(theta+h*u_i) - g(theta) )u_i$ ]
        """
        K = self.k # 200 random directions (for estimating the gradient)
        sign_grad = np.zeros(theta.shape)
        queries = 0
        ### USe orthogonal transform
        #dim = np.prod(sign_grad.shape)
        #H = np.random.randn(dim, K)
        #Q, R = qr(H, mode='economic')
        for iii in range(K): # for each u
            # # Code for reduced dimension gradient
            # u = np.random.randn(N_d,N_d)
            # u = u.repeat(D, axis=0).repeat(D, axis=1)
            # u /= LA.norm(u)
            # u = u.reshape([1,1,N,N])
            
            u = np.random.randn(*theta.shape); u /= LA.norm(u)
            new_theta = theta + h*u; new_theta /= LA.norm(new_theta)
            sign = 1
            
            # Targeted case.
            if (target is not None and 
                self.model.predict_label(x0+torch.tensor(initial_lbd*new_theta, dtype=torch.float).cuda()) == target):
                sign = -1

            # Untargeted case
            # preds.append(self.model.predict_label(x0+torch.tensor(initial_lbd*new_theta, dtype=torch.float).cuda()).item())
            if (target is None and
                self.model.predict_label(x0+torch.tensor(initial_lbd*new_theta, dtype=torch.float).cuda()) != y0): # success
                sign = -1

            queries += 1
            sign_grad += u*sign
        
        sign_grad /= K

        # sign_grad_u = sign_grad/LA.norm(sign_grad)
        # new_theta = theta + h*sign_grad_u
        # new_theta /= LA.norm(new_theta)
        # fxph, q1 = self.fine_grained_binary_search_local(self.model, x0, y0, new_theta, initial_lbd=initial_lbd, tol=h/500)
        # delta = (fxph - initial_lbd)/h
        # queries += q1
        # sign_grad *= 0.5*delta
        
        return sign_grad, queries
    
    
    ##########################################################################################
    def sign_grad_v2(self, x0, y0, theta, initial_lbd, h=0.001, K=200):
        """
        Evaluate the sign of gradient by formulat
        sign(g) = 1/Q [ \sum_{q=1}^Q sign( g(theta+h*u_i) - g(theta) )u_i$ ]
        """
        sign_grad = np.zeros(theta.shape)
        queries = 0
        for _ in range(K):
            u = np.random.randn(*theta.shape)
            u /= LA.norm(u)
            
            ss = -1
            new_theta = theta + h*u
            new_theta /= LA.norm(new_theta)
            if self.model.predict_label(x0+torch.tensor(initial_lbd*new_theta, dtype=torch.float).cuda()) == y0:
                ss = 1
            queries += 1
            sign_grad += sign(u)*ss
        sign_grad /= K
        return sign_grad, queries


    def sign_grad_svm(self, x0, y0, theta, initial_lbd, h=0.001, K=100, lr=5.0, target=None):
        """
        Evaluate the sign of gradient by formulat
        sign(g) = 1/Q [ \sum_{q=1}^Q sign( g(theta+h*u_i) - g(theta) )u_i$ ]
        """
        sign_grad = np.zeros(theta.shape)
        queries = 0
        dim = np.prod(theta.shape)
        X = np.zeros((dim, K))
        for iii in range(K):
            u = np.random.randn(*theta.shape)
            u /= LA.norm(u)
            
            sign = 1
            new_theta = theta + h*u
            new_theta /= LA.norm(new_theta)            
            
            # Targeted case.
            if (target is not None and 
                self.model.predict_label(x0+torch.tensor(initial_lbd*new_theta, dtype=torch.float).cuda()) == target):
                sign = -1
                
            # Untargeted case
            if (target is None and
                self.model.predict_label(x0+torch.tensor(initial_lbd*new_theta, dtype=torch.float).cuda()) != y0):
                sign = -1
                
            queries += 1
            X[:,iii] = sign*u.reshape((dim,))
        
        Q = X.transpose().dot(X)
        q = -1*np.ones((K,))
        G = np.diag(-1*np.ones((K,)))
        h = np.zeros((K,))
        ### Use quad_qp solver 
        #alpha = solve_qp(Q, q, G, h)
        ### Use coordinate descent solver written by myself, avoid non-positive definite cases
        alpha = quad_solver(Q, q)
        sign_grad = (X.dot(alpha)).reshape(theta.shape)
        
        return sign_grad, queries


    def fine_grained_binary_search_local(self, model, x0, y0, theta, initial_lbd = 1.0, tol=1e-5):
        nquery = 0
        lbd = initial_lbd
         
        # still inside boundary
        if model.predict_label(x0+torch.tensor(lbd*theta, dtype=torch.float).cuda()) == y0:
            lbd_lo = lbd
            lbd_hi = lbd*1.01
            nquery += 1
            while model.predict_label(x0+torch.tensor(lbd_hi*theta, dtype=torch.float).cuda()) == y0:
                lbd_hi = lbd_hi*1.01
                nquery += 1
                if lbd_hi > 20:
                    return float('inf'), nquery
        else:
            lbd_hi = lbd
            lbd_lo = lbd*0.99
            nquery += 1
            while model.predict_label(x0+torch.tensor(lbd_lo*theta, dtype=torch.float).cuda()) != y0 :
                lbd_lo = lbd_lo*0.99
                nquery += 1

        while (lbd_hi - lbd_lo) > tol:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if model.predict_label(x0 + torch.tensor(lbd_mid*theta, dtype=torch.float).cuda()) != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        return lbd_hi, nquery


    def fine_grained_binary_search(self, model, x0, y0, theta, initial_lbd, current_best):
        nquery = 0
        if initial_lbd > current_best: 
            if model.predict_label(x0+torch.tensor(current_best*theta, dtype=torch.float).cuda()) == y0:
                nquery += 1
                return float('inf'), nquery
            lbd = current_best
        else:
            lbd = initial_lbd

        lbd_hi = lbd
        lbd_lo = 0.0
        
        while (lbd_hi - lbd_lo) > 1e-3: # was 1e-5
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if model.predict_label(x0 + torch.tensor(lbd_mid*theta, dtype=torch.float).cuda()) != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        return lbd_hi, nquery


    def __call__(self, input_xi, label_or_target, iterations, targeted=False, distortion=None, seed=None,
                 svm=False, query_limit=4000, momentum=0.0, stopping=0.0001, args=None): # this line: dummy args to match signopt-lf
        if targeted:
            raise NotImplementedError
            # adv = self.attack_targeted(input_xi, label_or_target, target, distortion=distortion,
            #                            seed=seed, svm=svm, query_limit=query_limit, stopping=stopping)
        else:
            adv = self.attack_untargeted(input_xi, label_or_target, iterations = iterations, distortion=distortion,
                                         svm=svm, query_limit=query_limit, momentum=momentum,
                                         stopping=stopping)
        return adv