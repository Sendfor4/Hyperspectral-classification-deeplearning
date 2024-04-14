import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


# 1.HResNet,快速好用
class HResNet(nn.Module):
    def __init__(self, num_of_bands, num_of_class, patch_size):
        super(HResNet, self).__init__()
        self.num_of_bands = num_of_bands
        self.num_of_class = num_of_class
        self.conv0 = nn.Conv2d(self.num_of_bands, 64, kernel_size=(3, 3), padding=(0, 0))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu11 = nn.ReLU()
        self.conv11 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1))
        self.relu12 = nn.ReLU()
        self.conv12 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.relu21 = nn.ReLU()
        self.conv21 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1))
        self.relu22 = nn.ReLU()
        self.conv22 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1))
        self.avg_pool = nn.AvgPool2d((patch_size - 2, patch_size - 2))
        # self.dense = nn.Linear(64 * (patch_size - 2) * (patch_size - 2), num_of_class)
        self.dense = nn.Linear(64, num_of_class)

    def forward(self, x):
        x1 = self.conv0(x)
        x0 = x1
        x1 = self.bn1(x1)
        x1 = self.relu11(x1)
        x1 = self.conv11(x1)
        x1 = self.relu12(x1)
        x1 = self.conv12(x1)
        x1 = x0 + x1
        x2 = self.bn2(x1)
        x2 = self.relu21(x2)
        x2 = self.conv21(x2)
        x2 = self.relu22(x2)
        x2 = self.conv22(x2)
        res = x1 + x2
        # res = x0 + x1
        res = self.avg_pool(res)
        res = res.contiguous().view(res.size(0), -1)
        res = self.dense(res)
        return res


# 2. 快速3D卷积
class FAST3DCNN(nn.Module):
    def __init__(self, num_of_bands, num_of_class, patch_size):
        super(FAST3DCNN, self).__init__()
        self.patch_size = patch_size
        self.num_of_bands = num_of_bands
        self.num_of_class = num_of_class

        self.conv1 = nn.Conv3d(1, 8, (7, 3, 3), padding=(0, 0, 0))
        self.conv1_bn = nn.BatchNorm3d(8)

        self.conv2 = nn.Conv3d(8, 16, (5, 3, 3), stride=(1, 1, 1), padding=(0, 0, 0))
        self.conv2_bn = nn.BatchNorm3d(16)
        self.conv3 = nn.Conv3d(16, 32, (3, 3, 3), stride=(1, 1, 1), padding=(0, 0, 0))
        self.conv3_bn = nn.BatchNorm3d(32)

        self.conv4 = nn.Conv3d(32, 64, (3, 3, 3), stride=(1, 1, 1), padding=(0, 0, 0))
        self.conv4_bn = nn.BatchNorm3d(64)

        self.dropout = nn.Dropout(p=0.4)

        self.features_size = self._get_final_flattened_size()

        self.fc1 = nn.Linear(self.features_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, self.num_of_class)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.num_of_bands, self.patch_size, self.patch_size))
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            b, t, c, w, h = x.size()
        return b * t * c * w * h

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = x.view(-1, self.features_size)
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


# 3. 混合光谱网
class HybridSN(nn.Module):
    def __init__(self, num_of_bands, num_of_class, patch_size):
        super().__init__()

        self.patch_size = patch_size
        self.num_of_bands = num_of_bands
        self.num_of_class = num_of_class

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(7, 3, 3)),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(5, 3, 3)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True))

        self.x1_shape = self.get_shape_after_3dconv()
        print(self.x1_shape)

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=self.x1_shape[1] * self.x1_shape[2], out_channels=64, kernel_size=(3, 3)),
            nn.ReLU(inplace=True))

        self.x2_shape = self._get_final_flattened_size()

        self.dense1 = nn.Sequential(
            nn.Linear(self.x2_shape, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4))

        self.dense2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4))

        self.dense3 = nn.Sequential(
            nn.Linear(128, self.num_of_class)
        )

    def get_shape_after_3dconv(self):
        x = torch.zeros((1, 1, self.num_of_bands, self.patch_size, self.patch_size))
        with torch.no_grad():
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
        return x.shape

    def _get_final_flattened_size(self):
        x = torch.zeros((1, self.x1_shape[1] * self.x1_shape[2], self.x1_shape[3], self.x1_shape[4]))
        with torch.no_grad():
            x = self.conv4(x)
        return x.shape[1] * x.shape[2] * x.shape[3]

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4])
        # print(x.shape)
        x = self.conv4(x)
        x = x.contiguous().view(x.shape[0], -1)
        # print(x.shape)
        x = self.dense1(x)
        x = self.dense2(x)
        out = self.dense3(x)
        # print(out.shape)
        return out


# 4.ResNet18和ResNet34，残差神经网络
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=(1, 1)):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_ch)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, blocks_num, num_of_bands, num_of_class):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.num_of_bands = num_of_bands
        self.num_of_class = num_of_class
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.num_of_bands, 64, kernel_size=(3, 3), padding=(0, 0), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, blocks_num[0], stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, blocks_num[1], stride=1)
        self.layer3 = self.make_layer(ResidualBlock, 256, blocks_num[1], stride=1)
        self.layer4 = self.make_layer(ResidualBlock, 512, blocks_num[3], stride=1)
        self.fc = nn.Linear(512*7*7, self.num_of_class)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        out = out.contiguous().view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18(num_of_bands, num_of_class):
    return ResNet(ResidualBlock, [2, 2, 2, 2], num_of_bands, num_of_class)


def ResNet34(num_of_bands, num_of_class):
    return ResNet(ResidualBlock, [3, 4, 6, 3], num_of_bands, num_of_class)


if __name__ == '__main__':
    net = []
    model = 'HResNet'

    if model == 'HResNet':
        net = HResNet(num_of_bands=20, num_of_class=9, patch_size=11)
        writer = SummaryWriter(log_dir=r'C:\Users\Sendfor\PyWORKSPACE\stu_torch\HSIC\logs\HResNet\.',
                               comment='HResNet')
        writer.add_graph(net, input_to_model=torch.zeros((1, 20, 11, 11)))
        writer.close()

    if model == 'FAST3DCNN':
        net = FAST3DCNN(num_of_bands=20, num_of_class=9, patch_size=11)
        writer = SummaryWriter(log_dir=r'C:\Users\Sendfor\PyWORKSPACE\stu_torch\HSIC\logs\FAST3DCNN\.',
                               comment='FAST3DCNN')
        writer.add_graph(net, input_to_model=torch.zeros((1, 20, 11, 11)))
        writer.close()

    if model == 'HybridSN':
        net = HybridSN(num_of_bands=30, num_of_class=16, patch_size=25)
        writer = SummaryWriter(log_dir=r'C:\Users\Sendfor\PyWORKSPACE\stu_torch\HSIC\logs\HybridSN\.',
                               comment='HybridSN')
        writer.add_graph(net, input_to_model=torch.zeros((1, 30, 25, 25)))
        writer.close()



