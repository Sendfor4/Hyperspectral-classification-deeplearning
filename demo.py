# -*- coding: utf-8 -*-

import os
import time
import sys
import spectral
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn,optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import utils
import models

parser = argparse.ArgumentParser(description='settings of this tools')
parser.add_argument('--method', type=str, default='HResNet')
parser.add_argument('--dataset', type=str, default='paviaU')
parser.add_argument('--gt', type=str, default='data/paviaU_raw_gt.npy')
parser.add_argument('--trial_epoch', type=int, default=5)
parser.add_argument('--samples_per_class', type=float, default=0.02)
parser.add_argument('--patch', type=int, default=9)
parser.add_argument('--patience', type=int, default=100)
parser.add_argument('--epoch_alpha', type=int, default=170)
parser.add_argument('--epoch_beta', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--verbose', type=bool, default=True)
parser.add_argument('--output', type=str, default='output')
args = parser.parse_args()
dict_args = vars(args)

# 定义训练设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
else:
    nw = 0

# 设置起始训练时间
trial_begin_time = time.strftime("%Y_%m_%d_%H_%M", time.localtime())

# 导入高光谱图片数据dataset和ground truth数据
dataset_path = 'data/' + args.dataset + '_im.npy'
save_path = args.output + args.dataset + '_' + str(round(args.samples_per_class)) \
            + '_per_class_' + trial_begin_time + '/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

dataset = np.load(dataset_path)
ground_truth = np.load(args.gt)
num_of_label = np.max(ground_truth)
dataset_shape_x, dataset_shape_y, dataset_bands_num = dataset.shape

# 数据归一化以加快收敛
dataset = np.float32(dataset)
dataset = dataset / dataset.max()

output_chart = utils.OutputData(num_of_label, args.trial_turn)

# 网络选择部分
net = []
if args.method == "HResNet":
    print("Using HResNet")
    net = models.HResNet(num_of_bands=dataset_bands_num, num_of_class=num_of_label, patch_size=args.patch)
elif args.method == "FAST3DCNN":
    print("Using FAST3DCNN")
    net = models.FAST3DCNN(num_of_bands=dataset_bands_num, num_of_class=num_of_label, patch_size=args.patch)
elif args.method == "HybridSN":
    print("Using HybridSN")
    net = models.HybridSN(num_of_bands=dataset_bands_num, num_of_class=num_of_label, patch_size=args.patch)
elif args.method == "ResNet-18":
    print("Using ResNet-18")
    net = models.ResNet18(num_of_bands=dataset_bands_num, num_of_class=num_of_label)
elif args.method == "ResNet-34":
    print("Using ResNet-34")
    net = models.ResNet18(num_of_bands=dataset_bands_num, num_of_class=num_of_label)
else:
    print("the network doesn't exist!")

for current_trial_epoch in range(args.trial_epoch):
    # 对数据进行padding,得到的padding_dataset在外面加了patch/2圈数据
    padding_dataset = utils.padding(dataset, args.patch)
    # 制作训练集,以及没有ground truth的训练集
    predict_ground_truth, train_set, trial_label = \
        utils.make_train_set(padding_dataset, ground_truth, args.samples_per_class)


train_num = train_loader.dataset.__len__()
val_num = test_loader.dataset.__len__()

net.to(device=device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)


for epoch in range(epochs):
    # train
    net.train()
    running_loss = 0.0
    train_bar = tqdm(train_loader, file=sys.stdout)

    for batch_idx, (inputs, labels) in enumerate(train_bar):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        train_bar.desc = 'train epoch[{}/{}] loss:{:.3f}'.format(epoch + 1, epochs, loss)
    loss_list.append(running_loss / train_num)

    # validate,after train 5 times
    if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
        acc = 0.0  # accumulate accurate number / epoch
        net.eval()
        with torch.no_grad():
            val_bar = tqdm(test_loader, file=sys.stdout)

            for batch_idx, (val_inputs, val_labels) in enumerate(val_bar):
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)
                outputs = net(val_inputs)
                predict_y = torch.max(outputs, dim=1)[1]  # [1]我们只需要知道最大值所在的位置在哪里，也就是索引
                acc += torch.eq(predict_y, val_labels).sum().item()

        val_accurate = acc / val_num
        val_acc_list.append(val_accurate)
        val_epoch_list.append(epoch)
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        # get best model
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
