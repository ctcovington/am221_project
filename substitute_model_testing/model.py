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

import argparse

from models import *
from misc import progress_bar

# import pickle

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck', 'abstain')

def get_predictions(version):
    prediction_dt = pd.read_csv(os.path.join('..', 'project_data', 'black_box_output_%s.txt' % version), sep = '\t')
    return(prediction_dt['predict'].replace(-1,10))

def main():
    parser = argparse.ArgumentParser(description="cifar-10 with PyTorch")
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epoch', default=200, type=int, help='number of epochs to train for')
    parser.add_argument('--trainBatchSize', default=100, type=int, help='training batch size')
    parser.add_argument('--testBatchSize', default=100, type=int, help='testing batch size')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='whether cuda is in use')
    args = parser.parse_args()

    solver = Solver(args)
    solver.run()

    # sub_model_dir = 'substitute_models'
    # if not os.path.exists(sub_model_dir):
    #     os.makedirs(sub_model_dir)
    # with open(os.path.join(sub_model_dir, 'substitute_model.pkl', 'wb')) as file:
    #     pickle.dump(solver, file, pickle.HIGHEST_PROTOCOL)

# class CustomDataLoader(object):
#     def __init__(self, config):
#         self.dataset = config.dataset
#         self.batch_size = config.batch_size
#
#     def batch_sampling(self):
#         sample_indices = np.random.randint(0, len(self.dataset), size = self.batch_size)
#         index_counts = dict()
#         for index in sample_indices:
#             index_counts[index] = index_counts.get(index, 0) + 1
#

class Solver(object):
    def __init__(self, config):
        self.model = None
        self.lr = config.lr
        self.epochs = config.epoch
        self.train_batch_size = config.trainBatchSize
        self.test_batch_size = config.testBatchSize
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.cuda = config.cuda
        # self.train_loader = None
        self.train_set = None
        # self.test_loader = None
        self.test_set = None

    def load_data(self):
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToTensor()])

        train_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=train_transform)
        # self.train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.train_batch_size, shuffle=True)

        # replace train_set labels with predictions from black box querying
        predictions = get_predictions(version = 'cifar10_test')

        train_set_predictions = []
        for i in range(len(train_set)):
            train_set_predictions.append( (train_set[i][0], predictions[i]) )

        # hold out 10% of training set as test set
        train_set_predictions = random.sample(train_set_predictions, len(train_set_predictions))
        train_set_predictions_train = train_set_predictions[math.floor(len(train_set_predictions) / 10):]
        train_set_predictions_test = train_set_predictions[0:math.floor(len(train_set_predictions) / 10)]

        self.train_set = train_set_predictions_train
        self.test_set = train_set_predictions_test

        # self.train_loader = CustomDataLoader(dataset = train_set_predictions, batch_size = self.test_batch_size)

        # test_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=test_transform)
        # self.test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=self.test_batch_size, shuffle=False)

    def load_model(self):
        if self.cuda:
            self.device = torch.device('cuda')
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        self.model = WideResNet(depth=10, num_classes=11).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[75, 150], gamma=0.5)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train(self):
        print("train:")
        self.model.train()
        train_loss = 0
        train_correct = 0
        total = 0

        '''
        TODO: maybe shuffle data here for each epoch
        '''
        train_set = random.sample(self.train_set, len(self.train_set))
        # for batch_num, (data, target) in enumerate(self.train_loader):
        for batch_num in range(int(len(self.train_set) / self.train_batch_size)):
            # subset to batch
            batch = train_set[batch_num*self.train_batch_size : (batch_num+1)*self.train_batch_size]
            data = [None] * self.train_batch_size
            target = [None] * self.train_batch_size
            for i in range(len(batch)):
                data[i] = batch[i][0]
                target[i] = batch[i][1]
            data = torch.stack(data)
            target = torch.LongTensor(target)

            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
            total += target.size(0)

            # train_correct incremented by one if predicted right
            train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

            progress_bar(batch_num, len(self.train_set)/self.train_batch_size, 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_num + 1), 100. * train_correct / total, train_correct, total))

        return train_loss, train_correct / total

    def test(self):
        print("test:")
        self.model.eval()
        test_loss = 0
        test_correct = 0
        total = 0

        with torch.no_grad():
            test_set = random.sample(self.test_set, len(self.test_set))

            for batch_num in range(int(len(self.test_set) / self.test_batch_size)):
                # subset to batch
                batch = test_set[batch_num*self.test_batch_size : (batch_num+1)*self.test_batch_size]
                data = [None] * self.test_batch_size
                target = [None] * self.test_batch_size
                for i in range(len(batch)):
                    data[i] = batch[i][0]
                    target[i] = batch[i][1]
                data = torch.stack(data)
                target = torch.LongTensor(target)

                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                test_loss += loss.item()
                prediction = torch.max(output, 1)
                total += target.size(0)
                test_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

                progress_bar(batch_num, len(self.test_set)/self.test_batch_size, 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_num + 1), 100. * test_correct / total, test_correct, total))

        return test_loss, test_correct / total

    def save(self):
        model_out_path = "model.pth"
        torch.save(self.model.state_dict(), model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def run(self):
        self.load_data()
        self.load_model()
        accuracy = 0
        for epoch in range(1, self.epochs + 1):
            self.scheduler.step(epoch)
            print("\n===> epoch: %d/200" % epoch)
            train_result = self.train()
            print(train_result)
            test_result = self.test()
            accuracy = max(accuracy, test_result[1])
            if epoch == self.epochs:
                print("===> BEST ACC. PERFORMANCE: %.3f%%" % (accuracy * 100))
                self.save()

if __name__ == '__main__':
    main()
