import torch
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms as transforms
import numpy as np
import pandas as pd
import os
import random
import math
from models import *
import foolbox
import pickle
import argparse

from cw import *

def load_data():
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=train_transform)

    train_set_predictions = []
    for i in range(len(train_set)):
        train_set_predictions.append( (train_set[i][0], train_set[i][1]) )

    data = [None] * len(train_set_predictions)
    target = [None] * len(train_set_predictions)

    for i in range(len(train_set_predictions)):
        data[i] = train_set_predictions[i][0]
        target[i] = train_set_predictions[i][1]
    data = torch.stack(data)
    target = torch.LongTensor(target)

    return(data,target)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="l2 attack")
    parser.add_argument('--threshold', default=None, type=float, help='l2 norm threshold')
    args = parser.parse_args()

    # initialize device and parameters
    device = torch.device('cuda')
    lr = 0.001

    # load and initialize model
    model = WideResNet(depth=10, num_classes=11).to(device)
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    optimizer = optim.Adam(model.parameters(), lr = lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 150], gamma=0.5)
    criterion = nn.CrossEntropyLoss().to(device)

    # load CIFAR10 data
    cifar10_data, cifar10_labels = load_data()

    '''
    test getting output
    '''
    data = cifar10_data
    target = cifar10_labels
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)

    '''
    Carlini/Wagner from foolbox
    '''
    # create numpy version of data
    data_cw = data.cpu().numpy()
    target_cw = target.cpu().numpy()

    # initialize foolbox model and attack
    fb_model = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes = 11)
    fb_criterion = foolbox.criteria.TargetClass(10)
    fb_criterion2 = foolbox.criteria.TargetClassProbability(10, p = 0.95)
    fb_distance = foolbox.distances.MeanSquaredDistance
    cw_l2_attack = foolbox.attacks.CarliniWagnerL2Attack(fb_model, fb_criterion, fb_distance, threshold = args.threshold)
    cw_l2_attack2 = foolbox.attacks.CarliniWagnerL2Attack(fb_model, fb_criterion2, fb_distance, threshold = args.threshold)

    # carry out attack
    n = len(data_cw)
    cw_adv_data = [None] * n
    cw_adv_data2 = [None] * n
    print('carrying out CW attacks')
    for i in range(len(data_cw)):
        print('attack {0} of {1}'.format(i+1, n))
        cw_adv_data[i] = torch.Tensor(cw_l2_attack(data_cw[i], target_cw[i]))
        cw_adv_data2[i] = torch.Tensor(cw_l2_attack2(data_cw[i], target_cw[i]))

    '''
    version 1
    '''
    cw_tensor = torch.stack(cw_adv_data).to(device)
    cw_output = model(cw_tensor)
    cw_labels = torch.max(cw_output, 1)

    # get distances between adversarial images and original images
    dists = [0] * len(data_cw)
    for i in range(len(data_cw)):
        data_cw_flat = data_cw[i].flatten()
        cw_adv_data_flat = cw_tensor[i].cpu().numpy().flatten()
        dists[i] = np.linalg.norm(data_cw_flat - cw_adv_data_flat, 2)

    # save adversarial images, supposed class (should all be 10), and distances between adversarial and original
    adv_output = (cw_tensor.cpu().numpy(), # images
                  cw_labels[1].cpu().numpy(), # adversarial classes
                  dists # distances
                  )
    if args.threshold == None:
        pickle.dump(adv_output, open('l2_adversarial_output_cw_threshold_None.pkl', 'wb'))
    else:
        pickle.dump(adv_output, open('l2_adversarial_output_cw_threshold_%s.pkl' % args.threshold, 'wb'))

    '''
    version 2
    '''
    cw_tensor = torch.stack(cw_adv_data).to(device)
    cw_output = model(cw_tensor)
    cw_labels = torch.max(cw_output, 1)

    # get distances between adversarial images and original images
    dists = [0] * len(data_cw)
    for i in range(len(data_cw)):
        data_cw_flat = data_cw[i].flatten()
        cw_adv_data_flat = cw_tensor[i].cpu().numpy().flatten()
        dists[i] = np.linalg.norm(data_cw_flat - cw_adv_data_flat, 2)

    # save adversarial images, supposed class (should all be 10), and distances between adversarial and original
    adv_output = (cw_tensor.cpu().numpy(), # images
                  cw_labels[1].cpu().numpy(), # adversarial classes
                  dists # distances
                  )
    if args.threshold == None:
        pickle.dump(adv_output2, open('l2_adversarial_output_lbfgs_threshold_None.pkl', 'wb'))
    else:
        pickle.dump(adv_output2, open('l2_adversarial_output_lbfgs_threshold_%s.pkl' % args.threshold, 'wb'))

    print('done')
