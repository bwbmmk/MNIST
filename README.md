# MNIST 手写数字识别（PyTorch）

用 PyTorch 从零手写一个卷积神经网络（CNN），完成 MNIST 手写数字 0-9 的十分类任务。
代码里带大量中文注释，适合作为 PyTorch 入门练习。

> **注：项目正在编写中，尚未完工。**
> 目前完成了数据准备与模型定义，训练循环和测试评估代码还在写，训练结果待补充。

## 网络结构

```
输入 1×28×28
  ├─ Conv2d(1→16, 3×3, pad=1) + ReLU + MaxPool(2)
  ├─ Conv2d(16→32, 3×3, pad=1) + ReLU + MaxPool(2)
  ├─ Conv2d(32→64, 3×3, pad=1) + ReLU
  ├─ Flatten (64×7×7)
  ├─ Linear(3136→128) + ReLU
  └─ Linear(128→10)
```

- 损失函数：`CrossEntropyLoss`
- 优化器：`Adam(lr=0.001)`
- 训练 batch size 64，测试 batch size 256

## 环境依赖

```
torch
torchvision
tqdm
matplotlib
```

安装：

```bash
pip install torch torchvision tqdm matplotlib
```

有 NVIDIA 显卡时脚本会自动使用 CUDA，否则回退到 CPU：

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

## 运行

```bash
python mnist_self.py
```

数据集默认下载/缓存到 `D:/ai/mnist_data/`，**换机器运行前请先修改 `mnist_self.py` 里的 `path` 变量**，例如改成相对路径：

```python
path = './data/'
```

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `mnist_self.py` | 主程序：数据准备、模型定义、训练与测试 |
| `一些知识.md` | 卷积相关的小笔记 |
| `image.png` / `image-1.png` | 笔记中的配图 |

## 备注

`mnist_data.lnk` 是本地数据目录的快捷方式，不纳入版本管理。
