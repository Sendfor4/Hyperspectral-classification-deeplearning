## data
npy格式的Pavia University数据集、Salinas Valey数据集、Indian Pines数据集。
## dataset
原生的mat格式数据集，包含Pavia University数据集、Salinas Valey数据集、Indian Pines数据集。
## logs
使用tensorboard生成的部分可视化模型。

安装tensorboard后可以在```Terminal```中键入诸如```tensorboard --logdir C:\Users\81635\PyWORKSPACE\HSIC\logs\FAST3DCNN\.```的语句打开。
## Attention.py
CBAM原论文中的通道注意力机制和空间注意力机制，论文地址[CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521)
## models.py
一些分类模型。
## demo.py
还在弄。
## pytorchtools.py
来源：https://github.com/Bjarten/early-stopping-pytorch， 实现了pytorch的早停功能。
## utils.py
暂时封装了一些常见的处理方法


# 一部分我翻阅的外文文献
### 图像分类
- AlexNet [http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
- VGG [https://arxiv.org/abs/1409.1556](https://arxiv.org/abs/1409.1556)
- GoogLeNet, Inceptionv1(Going deeper with convolutions) [https://arxiv.org/abs/1409.4842](https://arxiv.org/abs/1409.4842)
- Inceptionv3(Rethinking the Inception Architecture for Computer Vision) [https://arxiv.org/abs/1512.00567](https://arxiv.org/abs/1512.00567)
- Xception(Deep Learning with Depthwise Separable Convolutions) [https://arxiv.org/abs/1610.02357](https://arxiv.org/abs/1610.02357)
- ResNet [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)
- ResNeXt [https://arxiv.org/abs/1611.05431](https://arxiv.org/abs/1611.05431)
- DenseNet [https://arxiv.org/abs/1608.06993](https://arxiv.org/abs/1608.06993)
- MobileNet(v1) [https://arxiv.org/abs/1704.04861](https://arxiv.org/abs/1704.04861)
- MobileNet(v2) [https://arxiv.org/abs/1801.04381](https://arxiv.org/abs/1801.04381)
- MobileNet(v3) [https://arxiv.org/abs/1905.02244](https://arxiv.org/abs/1905.02244)
- ShuffleNet(v1) [https://arxiv.org/abs/1707.01083](https://arxiv.org/abs/1707.01083)
- ShuffleNet(v2) [https://arxiv.org/abs/1807.11164](https://arxiv.org/abs/1807.11164)
- EfficientNet(v1) [https://arxiv.org/abs/1905.11946](https://arxiv.org/abs/1905.11946)
- EfficientNet(v2) [https://arxiv.org/abs/2104.00298](https://arxiv.org/abs/2104.00298)
- Vision Transformer [https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)

### 自然语言处理
- Attention Is All You Need [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

### 随机深度
- Deep Networks with Stochastic Depth [https://arxiv.org/pdf/1603.09382v1](https://arxiv.org/pdf/1603.09382v1)
### 注意力机制
- Squeeze-and-Excitation Networks [https://arxiv.org/pdf/1709.01507](https://arxiv.org/pdf/1709.01507)
- Selective Kernel Networks [https://arxiv.org/pdf/1903.06586](https://arxiv.org/pdf/1903.06586)
- CBAM: Convolutional Block Attention Module [https://arxiv.org/abs/1807.06521](https://arxiv.org/abs/1807.06521)
  
### 高光谱分类
- Deep Learning for Hyperspectral Image Classification: An Overview [https://arxiv.org/pdf/1910.12861v1](https://arxiv.org/pdf/1910.12861v1)
- HybridSN: Exploring 3D-2D CNN Feature Hierarchy for Hyperspectral Image Classification [https://arxiv.org/pdf/1902.06701](https://arxiv.org/pdf/1902.06701)
- A Fast 3D CNN for Hyperspectral Image Classification [https://arxiv.org/pdf/2004.14152](https://arxiv.org/pdf/2004.14152)
- Going Deeper with Contextual CNN for Hyperspectral Image Classification [https://arxiv.org/pdf/1604.03519](https://arxiv.org/pdf/1604.03519)
- SpectralFormer: Rethinking Hyperspectral Image Classification with Transformers [https://arxiv.org/pdf/2107.02988](https://arxiv.org/pdf/2107.02988)
- Multitask Deep Learning with Spectral Knowledge for Hyperspectral Image Classification [https://arxiv.org/pdf/1905.04535](https://arxiv.org/pdf/1905.04535)
- Deep Convolutional Neural Networks for Hyperspectral Image Classification [Deep Convolutional Neural Networks for Hyperspectral Image Classification](https://www.hindawi.com/journals/js/2015/258619/)
- Wide Contextual Residual Network with Active Learning for Remote Sensing Image Classification [Wide Contextual Residual Network with Active Learning for Remote Sensing Image ](https://www.researchgate.net/publication/328991664_Wide_Contextual_Residual_Network_with_Active_Learning_for_Remote_Sensing_Image_Classification)
-  Spectral–Spatial Residual Network for Hyperspectral Image Classification:A 3-D Deep Learning Framework [Spectral–Spatial Residual Network for Hyperspectral Image Classification:A 3-D Deep Learning Framework](https://www.researchgate.net/publication/320145356_Deep_Residual_Networks_for_Hyperspectral_Image_Classification)
-  A Fast Dense Spectral–Spatial Convolution Network Framework for Hyperspectral Images Classification [A Fast Dense Spectral–Spatial Convolution Network Framework for Hyperspectral Images Classification](https://www.mdpi.com/2072-4292/10/7/1068)
- Double-Branch Multi-Attention Mechanism Network for Hyperspectral Image Classification [Double-Branch Multi-Attention Mechanism Network for Hyperspectral Image Classification](https://www.mdpi.com/2072-4292/11/11/1307)
- Hyperspectral Image Classification Based on Multi-Scale Residual Network with Attention Mechanism [Hyperspectral Image Classification Based on Multi-Scale Residual Network with Attention Mechanism](https://www.mdpi.com/2072-4292/13/3/335)
