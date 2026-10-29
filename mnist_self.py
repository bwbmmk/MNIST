import torch
import torchvision
from tqdm import tqdm
import matplotlib

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 如果网络能在GPU中训练，就使用GPU；否则使用CPU进行训练
print(f'Using device: {device}')

print('-------------------------------数据准备----------------------------------------')


transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(
), torchvision.transforms.Normalize(mean=[0.5], std=[0.5])])
# 如果是from torchvision import transforms，就可以直接写transform = transforms.Compose([...])
# 本质上transforms是从torchvision中导入的一个模块，里面包含了很多图像处理的函数和类
# compose是transforms中的一个类，Compose([...]) 接收一个“操作列表”，并按照列表顺序依次执行
# ToTensor()：把图片转换成 PyTorch 张量,像素值归一化到 0～1,Normalize：继续调整张量的数值
# Normalize它对每个像素执行：output = (input - mean)/std,也就是把数值从[0, 1] → [-1, 1]
# 因为归一化是按通道进行的。MNIST 只有一个灰度通道，所以列表里只有一个数。
# 如果是三通道 RGB 图片，通常需要分别给三个通道设置参数，例如：Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])

path = 'D:/ai/mnist_data/'  # 数据集下载后保存的目录

train_data = torchvision.datasets.MNIST(
    path, train=True, transform=transform, download=True)
test_data = torchvision.datasets.MNIST(
    path, train=False, download=True, transform=transform)
# torchvision中的torchvision.datasets库中提供了MNIST数据集的下载地址
# 因此我们可以直接二调用对应的函数来下载MNIST的训练集和测试集

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=256, shuffle=False)
# Pytorch中提供了一种叫做DataLoader的方法来让我们进行训练该方法自动将数据集打包成为迭代器，能够让我们很方便地进行后续的训练处理
# batch_size：每次训练的样本数，shuffle：是否打乱数据集
# 训练时打乱有助于模型不刻意记住，测试时不打乱方便复现。

print("训练集数量：", len(train_data))
print("测试集数量：", len(test_data))
print("图片形状：", train_data[0][0].shape)
# train_data[0] 得到 (image, label),train_data[0][0] 取出元组中的第 0 个元素，也就是 image.


print('-------------------------------数据准备完成,开始构建模型------------------------------------')


#定义神经网路模型（model）
class Net(torch.nn.Module):
#继承（nn.Module）:在pytorch中，所有神经网络都必须继承nn.Module
#其非常强大，几乎管理所有的层

#初始化init
    def __init__(self):
        super(Net,self).__init__()#按上面的继承nn.Module的init__方法初始化

        self.conv1=torch.nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,stride=1,padding=1)
        #in_channels=1：输入是单通道灰度图。out_channels=16：使用 16 个卷积核，产生 16 张特征图。
        #kernel_size=3：每个卷积核大小是 3×3。stride=1：每次向右、向下移动 1 个像素。padding=1：在四周补一圈 0。
        #padding=1 的作用是让卷积前后图像大小不变。卷积后图像大小计算公式：W_out = (W_in + 2*padding - kernel_size)/stride + 1,也可以自己理解
        self.pool1=torch.nn.MaxPool2d(kernel_size=2, stride=2)
        #步长为2，2*2的最大池化层，作用是在2*2中找到最大值，把图像缩小一半，减少计算量，同时保留主要特征
        #原28*28池化后成为14*14，16个通道
        
        self.conv2=torch.nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.pool2=torch.nn.MaxPool2d(kernel_size=2,stride=2)
        #原14*14池化后成为7*7，32个通道

        self.conv3=torch.nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1)
        #现在是7*7，64个通道

        self.flatten=torch.nn.Flatten()
        #flatten()：把多维张量展平为一维张量，方便后续全连接层处理

        self.fc1=torch.nn.Linear(in_features=7 * 7 * 64, out_features=128)
        #fc，fullyconnected layer全连接层，它的主要作用是将前一层提取的特征整合并映射到输出空间
        #linear(in_features, out_features)表示输入特征数和输出特征数，可以实现线性变换：y = xA^T + b，其中A是权重矩阵，b是偏置向量
        #linear是全连接层的核心操作，它将输入张量与权重矩阵相乘，并加上偏置向量，得到输出张量
        self.fc2=torch.nn.Linear(128,10)
        #10是输出类别数，表示数字0-9的分类任务

        self.relu=torch.nn.ReLU()
        #把激活函数提出来，方便复用

    def forward(self,x):
        #这里就是数据流水线了，让上面的一直传下去，形成一个完整的网络结构
        #卷积，激活，池化，卷积，激活，池化，卷积，激活，全连接，激活，全连接：
        x=self.conv1(x)
        x=self.relu(x)
        x=self.pool1(x)

        x=self.conv2(x)
        x=self.relu(x)
        x=self.pool2(x)

        x=self.conv3(x)
        x=self.relu(x)

        x=self.flatten(x)
        x=self.fc1(x)
        x=self.relu(x)
        x=self.fc2(x)
        return x

#实例化class，并把它放到显卡上
net=Net().to(device) 
#to() 方法用于将张量放入到指定的设备（如CPU或GPU中）
print('模型的结构是：\n',net)
#看一下模型结构

#然后就是写优化器和损失函数：

criterion=torch.nn.CrossEntropyLoss()
#交叉熵损失函数（CrossEntropyLoss）是分类问题中常用的损失函数，它结合了 softmax 和负对数似然损失（Negative Log-Likelihood Loss），适用于多分类任务。
#它的任务是“打分,计算出误差有多大。Loss 越小，说明模型越聪明
#在 PyTorch 里，所有的多分类任务几乎全部默认用交叉熵。

optimizer=torch.optim.Adam(net.parameters(),lr=0.001)
#Adam优化器（Adaptive Moment Estimation）是一种自适应学习率优化算法（初学无脑选），它结合了动量法和RMSProp的优点，能够在训练过程中动态调整每个参数的学习率，从而加快收敛速度并提高模型性能。
#Adam是optim模块中的一个类，它的构造函数接收两个主要参数：net.parameters() 和 lr=0.001。
#net.parameters(),告诉优化器要更新哪些参数，lr=0.001，学习率是一个超参数，它控制着模型在每次参数更新时的步长大小。学习率过大可能导致训练找不到最优解，过小则可能导致收敛速度过慢。
#黄金起点：对于初学者，0.001（1e-3）是 Adam 优化器的万能起步值。

print('-------------------------------模型构建完毕，开始训练模型---------------------------------')