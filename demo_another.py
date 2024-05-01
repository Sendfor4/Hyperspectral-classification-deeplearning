# -*- coding: utf-8 -*-
import time
import sys
import spectral
import argparse
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
from utils_another import *
import models
import random

parser = argparse.ArgumentParser(description='settings of this tools')
parser.add_argument('--method', type=str, default='HybridSNHResNetCBAM')
parser.add_argument('--dataset', type=str, default='IP')
parser.add_argument('--components', type=int, default=30)
parser.add_argument('--test_ratio', type=float, default=0.7)
parser.add_argument('--patch_size', type=int, default=25)
parser.add_argument('--class_nums', type=int, default=16)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--verbose', type=bool, default=True)
parser.add_argument('--output', type=str, default='output')
args = parser.parse_args()
dict_args = vars(args)


RANDOM_SEED = 42
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.autograd.set_detect_anomaly(True)


class TrainDataset(Dataset):
    def __init__(self):
        self.len = Xtrain.shape[0]
        self.X_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class TestDataset(Dataset):
    def __init__(self):
        self.len = Xtest.shape[0]
        self.X_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(ytest)

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return self.len


# 定义训练设备
device = get_device()

# 设置起始训练时间
trial_begin_time = time.strftime("%Y_%m_%d_%H_%M", time.localtime())

# 载入数据和标签
data, label = load_data(args.dataset)
print(f'data shape before pre-processing {data.shape}')
print(f'nums of label {label.max()} ')

spectral.save_rgb(f'./_RGB_origin.jpg', data, (30, 20, 10))
spectral.save_rgb(f'./_gt.jpg', label, colors=spectral.spy_colors)

# 数据处理
pca_components = args.components if args.dataset == 'IP' else 15
X, pca = apply_pca(data, num_components=pca_components)
y = label
# print(f' data shape after pca {X.shape}', f' labels shape {y.shape}')

X, y = create_image_cubes(X, y, patch_size=args.patch_size)
# print(f' data shape after create_image_cubes {X.shape}')

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=args.test_ratio, random_state=42, stratify=y)

# 转换数据以适应 pytorch 结构
Xtrain = Xtrain.reshape(-1, args.patch_size, args.patch_size, pca_components)
Xtest = Xtest.reshape(-1, args.patch_size, args.patch_size, pca_components)
Xtrain = Xtrain.transpose(0, 3, 1, 2)
Xtest = Xtest.transpose(0, 3, 1, 2)

TrainDataset = TrainDataset()
TestDataset = TestDataset()

if torch.cuda.is_available():
    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
else:
    nw = 0

train_loader = DataLoader(TrainDataset, batch_size=args.batch_size, shuffle=True, num_workers=nw)
test_loader = DataLoader(TestDataset, batch_size=args.batch_size, shuffle=False, num_workers=nw)

# 网络选择部分
net = []
if args.method == 'HybridSNHResNetCBAM':
    print('Using HybridSNHResNetCBAM')
    net = models.HybridSNHResNetCBAM(num_of_bands=pca_components, num_of_class=args.class_nums, patch_size=args.patch_size)
elif args.method == 'HybridSN':
    print('Using HybridSN')
    net = models.HybridSN(num_of_bands=pca_components, num_of_class=args.class_nums, patch_size=args.patch_size)

net.to(device=device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30], gamma=0.5)

loss_list = []
val_acc_list = []
val_epoch_list = []

best_acc = 0.0
save_path = './{}Net.pth'.format(args.method)
train_steps = len(train_loader)  # 得到每次训练时，迭代的次数

writer = SummaryWriter(log_dir=r'C:\Users\Sendfor\PyWORKSPACE\stu_torch\HSIC\logs\TrainLogs\.')

train_num = train_loader.dataset.__len__()  # 得到训练集的长度
val_num = test_loader.dataset.__len__()  # 得到验证集的长度

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)
for epoch in range(args.epochs):
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
        train_bar.desc = 'train epoch[{}/{}] loss:{:.5f}'.format(epoch + 1, args.epochs, loss)
    scheduler.step()
    writer.add_scalar('loss', running_loss, epoch + 1)
    loss_list.append(running_loss / train_num)

    # validate,after train 2 times
    if (epoch + 1) % 2 == 0 or (epoch + 1) == args.epochs:
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
        print('[epoch %d] train_loss: %.5f  val_accuracy: %.5f' %
              (epoch + 1, running_loss / train_steps, val_accurate))
        writer.add_scalar('train_loss', running_loss / train_steps, epoch + 1)
        writer.add_scalar('val_accuracy', val_accurate, epoch + 1)

        # get best model
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

model = models.HybridSNHResNetCBAM(num_of_bands=pca_components, num_of_class=args.class_nums,
                        patch_size=args.patch_size).to(device=device)
weights_path = save_path
assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
model.load_state_dict(torch.load(weights_path, map_location=device))

classification, confusion, oa, each_acc, aa, kappa = reports(model=model, device=device,
                                                             test_loader=test_loader, y_test=ytest, name=args.dataset)
classification = str(classification)
confusion = str(confusion)
file_name = "classification_report.txt"

with open(file_name, 'w') as x_file:
    x_file.write('\n')
    x_file.write('{} Kappa accuracy (%)'.format(kappa))
    x_file.write('\n')
    x_file.write('{} Overall accuracy (%)'.format(oa))
    x_file.write('\n')
    x_file.write('{} Average accuracy (%)'.format(aa))
    x_file.write('\n')
    x_file.write('\n')
    x_file.write('{}'.format(classification))
    x_file.write('\n')
    x_file.write('{}'.format(confusion))

# 载入原始数据集
X, y = load_data(args.dataset)

height = y.shape[0]
width = y.shape[1]
patch_size = args.patch_size

pca_components = pca_components if args.dataset == 'IP' else 15
X, pca = apply_pca(data, num_components=pca_components)

X = pad_with_zero(X, patch_size//2)

# 逐像素预测图像类别
outputs = np.zeros((height, width))
for i in range(height):
    for j in range(width):
        if int(y[i, j]) == 0:
            continue
        else:
            image_patch = X[i:i+patch_size, j:j+patch_size, :]
            image_patch = image_patch.reshape(-1, image_patch.shape[0], image_patch.shape[1], image_patch.shape[2])
            X_test_image = torch.FloatTensor(image_patch.transpose(0, 3, 1, 2)).to(device)
            prediction = net(X_test_image)
            prediction = np.argmax(prediction.detach().cpu().numpy(), axis=1)
            outputs[i][j] = prediction+1
    if i % 20 == 0:
        print('... ... row ', i, ' handling ... ...')
predict_image = spectral.imshow(classes=outputs.astype(int), figsize=(6, 6))
spectral.save_rgb(f'./{patch_size} x {patch_size} _gt.jpg', outputs, colors=spectral.spy_colors)
