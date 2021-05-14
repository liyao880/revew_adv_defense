import torch
import argparse
import numpy as np
import setup.utils_optattack as utils
from tqdm import tqdm
from setup.attacks import OPT_attack, ZOO, OPT_attack_sign_SGD, FGSM, PGD
from setup.utils import loadmodel, loaddata
from setup.torchmodelwrapper import PytorchModel


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="mnist",
                    help='Dataset to be used, [mnist,cifar10]')
parser.add_argument("--model", type=str, default="cnn")
parser.add_argument('--attack', type=str, default='ZOO', choices=['OPT_attack','ZOO','Sign_OPT','FGSM','PGD'],
                    help='Attack to be used')
parser.add_argument('--targeted', action='store_true', default=False,
                    help='Targeted attack.')
parser.add_argument('--random_start', action='store_true', default=False,
                    help='PGD attack with random start.')
parser.add_argument('--mode', type=str, help='Which lp constraint to run bandits [linf|l2]')
parser.add_argument('--exploration', type=float, help='\delta, parameterizes the exploration to be done around the prior') 
parser.add_argument('--verbose', action='store_true', default=False,
                    help='verbose.')
parser.add_argument('--batch_size', type=int, default=1,
                    help='batch_size')
parser.add_argument('--test_batch', type=int, default=300,
                    help='test batch number')
parser.add_argument("--root", type=str, required=True)
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--gpu', type=str, default='auto', help='tag for saving, enter debug mode if debug is in it')
args = vars(parser.parse_args())
args['save'] = '/acc_'+args['attack']
args['save_dist'] = '/dist_'+args['attack']
if args['dataset'] == 'mnist':
    args['init'] = "/m_cnn"
    args['epsilon'] = 0.3
elif args['dataset'] == 'cifar10':
    args['init'] = "/vgg16"
    args['epsilon'] = 0.03
else:
    print('invalid dataset')
print(args)


#### macros
attack_list = {
    "PGD": PGD,
    "FGSM": FGSM,
    "OPT_attack": OPT_attack,
    "ZOO": ZOO,
    "Sign_OPT":OPT_attack_sign_SGD
}

l2_list = ["Sign_OPT","OPT_attack","FGSM","ZOO"]
linf_list = ["PGD"]

if args['attack'] in l2_list:
    norm = 'L2'
elif args['attack'] in linf_list:
    norm = 'Linf'



#### load data
print('==> Loading data..')
_, test_loader = loaddata(args)

print('==> Loading model..')
model = loadmodel(args)
model = model.cuda()
model = model.eval()


## load attack model
# sign opt
amodel = PytorchModel(model, bounds=[0,1], num_classes=10) # just a wrapper
attack = attack_list[args['attack']](amodel)


iterations = [5000,8000,10000]
acc = []
avg_dist = []
for iteration in iterations:
    total_r_count = 0
    total_clean_count = 0
    total_distance = 0
    rays_successes = []
    successes = []
    stop_queries = [] # wrc added to match RayS
    for i, (xi, yi) in enumerate(tqdm(test_loader)):    
        ## data
        
        if i == args['test_batch']: break
        xi, yi = xi.cuda(), yi.cuda()
        
        ## attack
        if args['attack'] == 'ZOO':
            adv = attack(xi, yi, TARGETED=args['targeted'], max_iter=iteration)
            
        elif args['attack'] == 'OPT_attack':
            adv, nqueries, is_success = attack(xi, yi,
                TARGETED=args['targeted'], iterations=iteration, epsilon=args['epsilon'])
            
            if is_success:
                stop_queries.append(nqueries)
        elif args['attack'] == 'Sign_OPT':
            adv = attack(xi, yi, iteratoins=iteration)
        elif args['attack'] == 'FGSM':
            adv = attack(xi, yi, epsilon=args['epsilon'])
        elif args['attack'] == 'PGD':
            adv = attack(xi, yi, epsilon=args['epsilon'])  
        else:
            raise NotImplementedError
            
        if args['targeted'] == False:
            r_count = (torch.max(amodel.predict(adv),1)[1]==yi).nonzero().shape[0]
            clean_count = (torch.max(amodel.predict(xi),1)[1]==yi).nonzero().shape[0]
            total_r_count += r_count
            total_clean_count += clean_count
            total_distance += utils.distance(adv,xi,norm=norm.lower())
            successes.append(r_count)
    print("{}==>query:{}, clean_acc:{:4f}, robust.acc:{:4f}, avg.{}.dist:{:4f}".format(args['attack'], iteration,
                                                                          total_clean_count/args['test_batch']*1.0, 
                                                                          total_r_count/args['test_batch']*1.0, norm,
                                                                          total_distance/args['test_batch']*1.0))
    if args['attack'] == 'OPT_attack':
        num_queries = np.mean(np.array(stop_queries))
    acc.append(total_r_count/args['test_batch']*1.0)
    avg_dist.append(total_distance/args['test_batch']*1.0)

np.save("./pgd_results/"+args['dataset']+args['save'],np.array(acc))
np.save("./pgd_results/"+args['dataset']+args['save_dist'],np.array(avg_dist))
