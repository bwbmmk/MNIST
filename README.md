# MNIST 手写数字识别（PyTorch 学习记录）

从零手写的 CNN 十分类练习：用卷积神经网络识别 MNIST 手写数字 0–9。
代码里带大量中文注释，主要用来自己复习、看清每一步的数据变化。

> 目前已完成：数据准备 → 模型定义 → 训练 → 验证 → 最佳模型保存 → 测试评估，全流程跑通。

## 结果

| 指标 | 数值 |
| --- | --- |
| 最佳验证准确率 | **99.20%**（第 5 轮） |
| 测试集准确率 | **99.26%**（9926 / 10000） |
| 测试集损失 | 0.0243 |
| 可训练参数 | 426,122 |
| 权重文件大小 | 约 1.7 MB（float32） |
| 单轮训练耗时 | 约 8 秒（844 个 batch） |

单次训练的完整日志：

| Epoch | train_loss | train_acc | val_loss | val_acc | 是否刷新最佳 |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.1742 | 0.9464 | 0.0602 | 0.9817 | ✅ |
| 2 | 0.0473 | 0.9857 | 0.0529 | 0.9818 | ✅ |
| 3 | 0.0337 | 0.9887 | 0.0471 | 0.9857 | ✅ |
| 4 | 0.0260 | 0.9910 | 0.0376 | 0.9900 | ✅ |
| 5 | 0.0211 | 0.9929 | 0.0327 | **0.9920** | ✅ 最终采用 |
| 6 | 0.0161 | 0.9950 | 0.0430 | 0.9880 | ❌ |
| 7 | 0.0148 | 0.9951 | 0.0293 | 0.9893 | ❌ |
| 8 | 0.0129 | 0.9956 | 0.0431 | 0.9882 | ❌ |
| 9 | 0.0095 | 0.9969 | 0.0377 | 0.9915 | ❌ |
| 10 | 0.0098 | 0.9969 | 0.0489 | 0.9893 | ❌ |

可以看到第 6 轮之后 `val_loss` 不降反升、`val_acc` 也没再突破，属于轻微过拟合。
因为最终测试加载的是**验证集最优**的第 5 轮权重，所以这部分退化不会影响结果——这正是保存最佳模型的意义。
每次重新训练数值都会有小幅波动（随机初始化 + `shuffle=True` 的批次顺序不同）。

## 网络结构

```
输入 1×28×28
  ├─ Conv2d(1→16, 3×3, pad=1) + ReLU + MaxPool(2)    → 16×14×14
  ├─ Conv2d(16→32, 3×3, pad=1) + ReLU + MaxPool(2)   → 32×7×7
  ├─ Conv2d(32→64, 3×3, pad=1) + ReLU                → 64×7×7
  ├─ Flatten                                          → 3136
  ├─ Linear(3136→128) + ReLU
  └─ Linear(128→10)                                   → logits
```

- 损失函数：`CrossEntropyLoss`（内部自带 LogSoftmax，所以模型末尾**不加** Softmax）
- 优化器：`Adam(lr=0.001)`
- 输出是 10 个类别的原始分数（logits），预测时用 `argmax(dim=1)` 取类别

## 数据划分

MNIST 官方只提供训练集和测试集，验证集是从训练集里分出来的：

| 数据 | 数量 | 用途 |
| --- | --- | --- |
| 训练集 | 54,000 | 更新参数 |
| 验证集 | 6,000 | 每轮评估、挑选最佳模型 |
| 测试集 | 10,000 | 训练全部结束后只评估一次 |

划分用固定种子，保证每次运行切分一致：

```python
split_generator = torch.Generator().manual_seed(42)
train_set, val_set = random_split(train_data, [54000, 6000], generator=split_generator)
```

- 训练集 `batch_size=64, shuffle=True`
- 验证集 / 测试集 `batch_size=256, shuffle=False`

预处理：`ToTensor()` 把像素从 `0–255` 映射到 `0–1`，`Normalize([0.5], [0.5])` 再映射到 `-1–1`。

## 训练流程

每个 batch 固定五步，顺序不能乱：

```python
optimizer.zero_grad()              # 1. 清空上一批的梯度
outputs = net(images)              # 2. 前向传播
loss = criterion(outputs, labels)  # 3. 计算损失
loss.backward()                    # 4. 反向传播，求梯度
optimizer.step()                   # 5. 用梯度更新参数
```

每个 epoch 结束后：

1. 在验证集上评估，得到 `val_loss` / `val_acc`
2. `val_acc` 刷新历史最佳时，保存 `state_dict`
3. 训练全部结束后，加载最佳权重，在测试集上评估一次

损失按样本数加权累计，避免最后一个不满 batch 的批次权重偏大：

```python
total_loss += loss.item() * images.size(0)
train_loss = total_loss / len(train_loader.dataset)
```

## 环境依赖

```
torch
torchvision
tqdm
```

```bash
pip install torch torchvision tqdm
```

有 NVIDIA 显卡时脚本会自动使用 CUDA，否则回退到 CPU：

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

## 运行

```bash
python mnist_self.py
```

数据集默认下载/缓存到 `D:/ai/mnist_data/`，**换机器运行前请先修改 `mnist_self.py` 里的 `path` 变量**：

```python
path = 'D:/ai/mnist_data/'   # 也可以改成 './data/'
```

模型权重按时间戳命名，保存在**运行时的当前工作目录**：

```python
best_path = f'best_model_{datetime.now():%m%d_%H%M}.pth'
```

用 VS Code 的「运行 Python 文件」时，工作目录是工作区根目录，所以文件会落在项目外层。
想固定在脚本旁边，可以改成：

```python
from pathlib import Path
best_path = Path(__file__).parent / f'best_model_{datetime.now():%m%d_%H%M}.pth'
```

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `mnist_self.py` | 主文件：数据准备、模型定义、训练、验证、测试 |
| `一些知识.md` | 卷积与训练流程的笔记 |
| `踩坑与问答.md` | 从零手写过程中遇到的问题、报错与结论汇总 |
| `image.png` / `image-1.png` | 笔记中的配图 |

## 备注

- `mnist_data.lnk` 是本地数据目录的快捷方式，不纳入版本管理。
- `*.pth` / `*.pt` 已被 `.gitignore` 排除，权重文件不进入仓库。
- 想用保存的模型做推理，需要先有 `Net` 的类定义，再 `load_state_dict`；单独的 `.pth` 文件不包含网络结构。

## 后续计划

- [ ] 把 `Net` 抽到独立的 `model.py`，方便推理脚本导入（当前 import 主文件会重新训练）
- [ ] 增加 `predict.py`，支持单张图片预测
- [ ] 做一个网页手写板 demo（Canvas 写字 → 后端返回识别结果）
- [ ] 尝试 BatchNorm / Dropout / 数据增强，观察过拟合是否缓解
