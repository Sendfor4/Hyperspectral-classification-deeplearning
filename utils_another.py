# -*- coding: utf-8 -*-
"""
大概是主要的预处理方法都在这个部分
"""
import torch
import numpy as np
import os
from scipy.io import loadmat
from torchinfo import summary
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, cohen_kappa_score
from sklearn.model_selection import train_test_split
from operator import truediv


# 判断可训练设备
def get_device():
    if torch.cuda.is_available():
        print("Computation on CUDA GPU device ")
        device = torch.device('cuda:0')
    else:
        print("/!\\ CUDA was requested but is not available! Computation will go on CPU. /!\\")
        device = torch.device('cpu')
    return device


# 加载数据集和标签
def load_data(name):
    data_path = os.path.join(os.getcwd(), 'dataset')
    print(data_path)
    data = []
    labels = []
    if name == 'IP':
        data = loadmat(r'dataset/IndianPines/Indian_pines_corrected.mat')['indian_pines_corrected']
        labels = loadmat(r'dataset/IndianPines/Indian_pines_gt.mat')['indian_pines_gt']
    elif name == 'SA':
        data = loadmat(r'dataset/Salinas/Salinas_corrected.mat')['salinas_corrected']
        labels = loadmat(r'dataset/Salinas/Salinas_gt.mat')['salinas_gt']
    elif name == 'PU':
        data = loadmat(r'dataset/PaviaU/PaviaU.mat')['paviaU']
        labels = loadmat(r'dataset/PaviaU/PaviaU_gt.mat')['paviaU_gt']
    return data, labels


# 数据降维
def apply_pca(dataset, num_components=15):
    new = np.reshape(dataset, (-1, dataset.shape[2]))
    pca = PCA(n_components=num_components, whiten=True)  # PCA 以及 归一
    new = pca.fit_transform(new)  # 拟合数据并转换 (145*145, 30)
    new = np.reshape(new, (dataset.shape[0], dataset.shape[1], num_components))
    return new, pca


# 划分遥感图像数据集
def split_train_test_set(x, y, test_ratio, random_state=345):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_ratio,
                                                        random_state=random_state,
                                                        stratify=y)
    return x_train, x_test, y_train, y_test


# padZero函数，填充边，使用0进行填充
def pad_with_zero(dataset, margin):
    new_dataset = np.zeros((dataset.shape[0] + 2 * margin, dataset.shape[1] + 2 * margin, dataset.shape[2]))
    x_offset = margin
    y_offset = margin
    new_dataset[x_offset:dataset.shape[0] + x_offset, y_offset:dataset.shape[1] + y_offset, :] = dataset
    return new_dataset


def create_image_cubes(X, y, patch_size=5, removeZeroLabels=True):
    margin = int((patch_size - 1) / 2)
    zeroPaddedX = pad_with_zero(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], patch_size, patch_size, X.shape[2]))
    # (21025, 25, 25, 200)
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    # (21025, )
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            # [r - 12 : r + 12 + 1, c -12 : c + 12 + 1]
            # print(patch.shape)
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex = patchIndex + 1

    # 去掉 gt 标签集 groundtruth 中为 0 的数据
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1

    print(f'shape of patches_data{patchesData.shape}')
    print(f'shape of patches_labels {patchesLabels.shape}')
    return patchesData, patchesLabels


def aa_and_each_class_accuracy(confusion_matrix):
    """
        :param confusion_matrix: 混淆矩阵
        :return:
    """

    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def reports(model, device, test_loader, y_test, name):
    model.eval()
    y_pred_test = []
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        predicted = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        y_pred_test.append(predicted)

    y_pred_test = np.concatenate(y_pred_test)

    if name == 'IP':
        target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
            , 'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                        'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                        'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                        'Stone-Steel-Towers']
    elif name == 'SA':
        target_names = ['Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow', 'Fallow_rough_plow',
                        'Fallow_smooth',
                        'Stubble', 'Celery', 'Grapes_untrained', 'Soil_vinyard_develop', 'Corn_senesced_green_weeds',
                        'Lettuce_romaine_4wk', 'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk', 'Lettuce_romaine_7wk',
                        'Vinyard_untrained', 'Vinyard_vertical_trellis']
    elif name == 'PU':
        target_names = ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal sheets', 'Bare Soil', 'Bitumen',
                        'Self-Blocking Bricks', 'Shadows']

    classification = classification_report(y_test, y_pred_test, target_names=target_names)
    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = aa_and_each_class_accuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)

    return classification, confusion, oa * 100, each_acc * 100, aa * 100, kappa * 100

def oversampleWeakClasses(X, y):
    """
    增强弱势的数据,以平衡数据集
    :param X: 数据样本
    :param y:数据标签
    :return:
    """
    uniqueLabels, labelCounts = np.unique(y, return_counts=True)
    maxCount = np.max(labelCounts)
    labelInverseRatios = maxCount / labelCounts
    # repeat for every label and concat
    newX = X[y == uniqueLabels[0], :, :, :].repeat(round(labelInverseRatios[0]), axis=0)
    newY = y[y == uniqueLabels[0]].repeat(round(labelInverseRatios[0]), axis=0)
    for label, labelInverseRatio in zip(uniqueLabels[1:], labelInverseRatios[1:]):
        cX = X[y== label,:,:,:].repeat(round(labelInverseRatio), axis=0)
        cY = y[y == label].repeat(round(labelInverseRatio), axis=0)
        newX = np.concatenate((newX, cX))
        newY = np.concatenate((newY, cY))
    np.random.seed(seed=42)
    rand_perm = np.random.permutation(newY.shape[0])
    newX = newX[rand_perm, :, :, :]
    newY = newY[rand_perm]
    return newX, newY

def AugmentData(X_train):
    """
    数据增强
    :param X_train: 训练数据集
    :return:
    """
    for i in range(int(X_train.shape[0] / 2)):
        patch = X_train[i, :, :, :]
        num = random.randint(0, 2)
        if (num == 0):
            flipped_patch = np.flipud(patch)
        if (num == 1):
            flipped_patch = np.fliplr(patch)
        if (num == 2):
            no = random.randrange(-180, 180, 30)
            flipped_patch = scipy.ndimage.interpolation.rotate(patch, no, axes=(1, 0),
                                                               reshape=False, output=None, order=3, mode='constant',
                                                               cval=0.0, prefilter=False)

    patch2 = flipped_patch
    X_train[i, :, :, :] = patch2

    return X_train
