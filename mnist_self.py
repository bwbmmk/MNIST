import torch
import torchvision
from tqdm import tqdm
from torch.utils.data import random_split
from datetime import datetime

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

split_generator = torch.Generator().manual_seed(42)
# torch.Generator 是 PyTorch 中负责生成伪随机数的对象
# 有趣的是42源于道格拉斯·亚当斯的科幻小说《银河系漫游指南》，在书中，42被描述为“生命、宇宙以及一切问题的终极答案”。因此，42在编程和科学界常被用作随机数生成器的种子值，象征着对复杂问题的简单解答。

train_data = torchvision.datasets.MNIST(
    path, train=True, transform=transform, download=True)
test_data = torchvision.datasets.MNIST(
    path, train=False, download=True, transform=transform)
# torchvision中的torchvision.datasets库中提供了MNIST数据集的下载地址
# 因此我们可以直接二调用对应的函数来下载MNIST的训练集和测试集

train_set, val_set = random_split(
    train_data, [54000, 6000], generator=split_generator)
# val是validation的简称

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(
    val_set, batch_size=256, shuffle=False)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=256, shuffle=False)
# Pytorch中提供了一种叫做DataLoader的方法来让我们进行训练该方法自动将数据集打包成为迭代器，能够让我们很方便地进行后续的训练处理
# batch_size：每次训练的样本数，shuffle：是否打乱数据集
# 训练时打乱有助于模型不刻意记住，测试时不打乱方便复现。

print("训练集数量：", len(train_set))
print("验证集数量：", len(val_set))
print("测试集数量：", len(test_data))
print("图片形状：", train_data[0][0].shape)
# train_data[0] 得到 (image, label),train_data[0][0] 取出元组中的第 0 个元素，也就是 image.


print('-------------------------------数据准备完成,开始构建模型------------------------------------')


# 定义神经网路模型（model）
class Net(torch.nn.Module):
    # 继承（nn.Module）:在pytorch中，所有神经网络都必须继承nn.Module
    # 其非常强大，几乎管理所有的层

    # 初始化init
    def __init__(self):
        super(Net, self).__init__()  # 按上面的继承nn.Module的init__方法初始化

        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        # in_channels=1：输入是单通道灰度图。out_channels=16：使用 16 个卷积核，产生 16 张特征图。
        # kernel_size=3：每个卷积核大小是 3×3。stride=1：每次向右、向下移动 1 个像素。padding=1：在四周补一圈 0。
        # padding=1 的作用是让卷积前后图像大小不变。卷积后图像大小计算公式：W_out = (W_in + 2*padding - kernel_size)/stride + 1,也可以自己理解
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # 步长为2，2*2的最大池化层，作用是在2*2中找到最大值，把图像缩小一半，减少计算量，同时保留主要特征
        # 原28*28池化后成为14*14，16个通道

        self.conv2 = torch.nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # 原14*14池化后成为7*7，32个通道

        self.conv3 = torch.nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # 现在是7*7，64个通道

        self.flatten = torch.nn.Flatten()
        # flatten()：把多维张量展平为一维张量，方便后续全连接层处理

        self.fc1 = torch.nn.Linear(in_features=7 * 7 * 64, out_features=128)
        # fc，fullyconnected layer全连接层，它的主要作用是将前一层提取的特征整合并映射到输出空间
        # linear(in_features, out_features)表示输入特征数和输出特征数，可以实现线性变换：y = xA^T + b，其中A是权重矩阵，b是偏置向量
        # linear是全连接层的核心操作，它将输入张量与权重矩阵相乘，并加上偏置向量，得到输出张量
        self.fc2 = torch.nn.Linear(128, 10)
        # 10是输出类别数，表示数字0-9的分类任务

        self.relu = torch.nn.ReLU()
        # 把激活函数提出来，方便复用

    def forward(self, x):
        # 这里就是数据流水线了，让上面的一直传下去，形成一个完整的网络结构
        # 卷积，激活，池化，卷积，激活，池化，卷积，激活，全连接，激活，全连接：
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# 实例化class，并把它放到显卡上
net = Net().to(device)
# to() 方法用于将张量放入到指定的设备（如CPU或GPU中）
print('模型的结构是：\n', net)
# 看一下模型结构

# 然后就是写优化器和损失函数：

criterion = torch.nn.CrossEntropyLoss()
# 交叉熵损失函数（CrossEntropyLoss）是分类问题中常用的损失函数，它结合了 softmax 和负对数似然损失（Negative Log-Likelihood Loss），适用于多分类任务。
# 它的任务是“打分,计算出误差有多大。Loss 越小，说明模型越聪明
# 在 PyTorch 里，所有的多分类任务几乎全部默认用交叉熵。

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
# Adam优化器（Adaptive Moment Estimation）是一种自适应学习率优化算法（初学无脑选），它结合了动量法和RMSProp的优点，能够在训练过程中动态调整每个参数的学习率，从而加快收敛速度并提高模型性能。
# Adam是optim模块中的一个类，它的构造函数接收两个主要参数：net.parameters() 和 lr=0.001。
# net.parameters(),告诉优化器要更新哪些参数，lr=0.001，学习率是一个超参数，它控制着模型在每次参数更新时的步长大小。学习率过大可能导致训练找不到最优解，过小则可能导致收敛速度过慢。
# 黄金起点：对于初学者，0.001（1e-3）是 Adam 优化器的万能起步值。


print('-------------------------------模型构建完毕，开始训练模型---------------------------------')


# 训练的基本流程是训练—>验证->训练->验证，直到模型收敛。

# 这里我们定义一个验证函数，计算验证集的平均损失和准确率
def evaluate(model, loader, criterion, device):
    model.eval()  # 验证模式
    # 这会影响到：Dropout：评估时不再随机丢弃神经元；BatchNorm：评估时使用保存的统计量。
    # 但目前的cnn没有使用Dropout和BatchNorm，但养成习惯
    loss = 0.0  # 样本损失
    correct = 0  # 正确分类的样本数
    total = 0  # 总样本数
    with torch.no_grad():  # 在验证阶段不需要计算梯度，节省内存和计算资源
        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)
            # 每个 batch 的形状是：images  # [B, 1, 28, 28]   labels  # [B]

            outputs = model(images)  # forward()方法，得到预测结果
            # 因为处于评估模式,模型不会更新参数，也不会使用 Dropout,其输出为logits,即未经过softmax的原始输出

            loss += criterion(outputs, labels).item() * images.size(0)  # 累加损失
            # 计算当前批次的损失，并乘以该批次的样本数（images.size(0)），得到该批次的总损失，然后累加到总损失中。
            # 因为最后一个 batch 可能不满 256 张，如果直接total_loss += loss.item()，结果会不准确
            # 后面会再除以总样本数，得到平均损失

            correct += (outputs.argmax(dim=1) == labels).sum().item()
            # outputs.argmax(dim=1) 返回每个样本预测的类别索引，dim=1表示按行取最大值的索引
            # 如[0.1, 0.2, 5.3, ...] → 索引 2，得到形状 [B] 的预测标签。
            # (outputs.argmax(dim=1) == labels) 返回一个布尔张量，表示每个样本的预测是否正确
            # .sum() 统计正确预测的样本数，.item() 将结果转换为 Python 数值类型，方便累加到 correct 中

            total += labels.size(0)  # 累加当前 batch 的真实样本数量。

    return loss / total, correct / total  # 返回平均损失和准确率


# 正式训练
epochs = 10  # 循环轮次
best_val_acc = 0.0  # 初始化最佳验证准确率
# 保存最佳模型的路径,每一次都有一个新命名
best_path = f'best_model_{datetime.now():%m%d_%H%M}.pth'


for epoch in range(epochs):

    net.train()  # 训练模式
    total_loss = 0.0  # 初始化总损失
    train_correct = 0.0

    processbar = tqdm(
        train_loader, desc=f'epoch {epoch+1}/{epochs}', unit='batch', leave=True)
    # tqdm是一个进度条库，desc是进度条前的描述，unit是进度条的单位,true表示训练完成后进度条不消失

    for i, (images, labels) in enumerate(processbar):
        # enumerate()函数用于将一个可遍历的数据对象（如列表、元组或字符串）组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环中。
        # train_loader 每次产出一个批次(images, labels),enumerate(processbar) 每次产生：(批次编号, (images, labels))
        images = images.to(device)
        labels = labels.to(device)
        # .to() 通常返回一个新张量，不是原地修改原变量,所以要重新赋值

        optimizer.zero_grad()  # 每一次循环清空上一次的梯度，防止梯度累加
        outputs = net(images)  # 前向传播，得到预测结果
        # 这里的net(images)就是调用了Net类的forward方法，传入images，得到输出outputs
        loss = criterion(outputs, labels)  # 计算损失
        # 因为 CrossEntropyLoss 内部已经包含 LogSoftmax。
        loss.backward()  # 反向传播，计算梯度
        # 我们这里看作output=intput*w+b,反向传播计算出conv.b conv.w fc.w等等的这些参数
        # 后续就可以通过优化器来更新这些参数，让训练更好
        # 假设简单梯度下降w.new=w.old-lr*dw,其中dw就是loss.backward()计算出来的梯度
        # 这里Adam还会根据梯度的均值和方差来调整每个参数的学习率，从而加快收敛速度并提高模型性能。
        optimizer.step()  # 更新参数

        total_loss += loss.item() * images.size(0)  # 累加损失,同上验证函数
        train_correct += (outputs.argmax(dim=1) ==
                          labels).sum().item()  # 累加正确分类的样本数,同上

        processbar.set_postfix({'loss': loss.item()})  # 更新进度条，显示当前的损失值
        # set_postfix() 用来在 tqdm 进度条右侧显示附加信息,其每次会替换旧内容，不会无限追加.
        # loss本身是一个单元素张量，loss.item()返回一个 Python 数值类型，方便显示。

    # 计算训练集的平均损失和准确率
    train_loss = total_loss / len(train_loader.dataset)
    train_acc = train_correct / len(train_loader.dataset)

    # 在每个 epoch 结束后，进行验证集的评估
    val_loss, val_acc = evaluate(net, val_loader, criterion, device)

    print(f'epoch:{epoch+1}/{epochs}\n,train_loss:{train_loss:.4f},train_acc:{train_acc:.4f}\n,val_loss:{val_loss:.4f},val_acc:{val_acc:.4f}')
    # 显示轮数等

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(net.state_dict(), best_path)
        # torch.save是 PyTorch 提供的一个函数，用于将对象（如模型、张量或字典）保存到文件中,文件格式：建议使用 .pth 或 .pt 作为文件扩展名。
        print(f'目前最佳的模型准确率:{val_acc:.4f},且保存')
    # 这样可以确保测试时使用最佳模型，而非最后一轮模型

print('-------------------------------训练循环结束，接下来是测试模型时间--------------------------------')


net.load_state_dict(torch.load(best_path, map_location=device))
# 加载验证集表现最好的模型参数。
# torch.load用于加载由 torch.save() 保存的对象,再根据 map_location 参数映射到目标设备。
# state_dict是一个包含模型参数的字典
# load_state_dict()是一个用于加载模型参数的关键方法。它允许我们将预训练模型的参数加载到新的模型实例中，从而实现模型的复用和迁移学习。

test_loss, test_acc = evaluate(net, test_loader, criterion, device)
print(f'最终测试结果为:loss={test_loss:.4f},acc={test_acc:.4f}')


print('-----------------------------------------------------结束！-------------------------------------------------')

