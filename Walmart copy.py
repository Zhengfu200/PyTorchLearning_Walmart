# -*- coding: utf-8 -*-
"""
注意：本代码直接是.py 文件的内容, 你可以分段复制到jupyter(.ipynb)中运行，你也可以直接执行本.py 文件
本代码示例具有超级详细的注释, 如果依然有疑问, 优先问AI, 要养成习惯

实现一个简单的线性回归模型，使用 PyTorch 框架
目的是根据沃尔玛门店的周度信息预测其每周销售额
数据来源: Walmart_Sales.csv (kaggle 数据集)
源作者: 依力 EL@zju.edu.cn

数据划分是：
训练集 60% + 验证集 20% + 测试集 20%

三者的作用分别是：

训练集：
真正参与模型训练，参与 loss.backward() 和 optimizer.step()
也就是训练集会影响模型参数的更新

验证集：
不参与模型参数更新
只是在每一轮训练后，用来检查模型在“没参与训练的数据”上的表现
验证集可以帮助我们观察模型是否过拟合，也可以帮助我们选择效果最好的模型

测试集：
训练结束之后，最后才使用
测试集用于最终评估模型效果
测试集不应该参与训练，也不应该参与模型选择

需要第三方库：
- pandas
一个用于数据处理和分析的库

- torch
PyTorch 框架，用于构建和训练深度学习模型

- matplotlib
用于绘图和数据可视化

- seaborn
用于数据可视化

- scikit-learn
一个用于专注于数据挖掘和数据分析的库，提供了许多机器学习算法和工具
"""

import copy
import pandas as pd
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 解决 Windows 控制台 GBK 编码无法显示部分 Unicode 字符（如 R² 中的 ²）的问题
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')


# =========================
# 1. GPU 配置说明
# =========================

# 英伟达 GPU 配置
# 如果你使用的是带 NVIDIA 显卡的电脑，可以使用 cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Apple M系列 GPU 配置
# 如果你使用的是 Apple M1 / M2 / M3 / M4 系列芯片，可以尝试使用 mps
# 不过本案例数据量很小，用 CPU 完全足够
# 为了避免不同电脑环境导致报错，本代码默认使用 CPU
#
# if torch.backends.mps.is_available():
#     device = torch.device("mps")  # 使用 Apple GPU
# else:
#     print("❌ MPS 不可用，回退到 CPU")
#     device = torch.device("cpu")

device = torch.device("cpu")
print(f"当前使用设备：{device}")


# =========================
# 2. 设置支持中文字体
# =========================

# 设置支持中文字体
# macOS 通常可以使用 PingFang HK、Heiti TC、Arial Unicode MS
# Windows 可以尝试 SimHei、Microsoft YaHei
# Linux 可以尝试 WenQuanYi Micro Hei
#
# 这里设置多个字体，matplotlib 会按照顺序寻找可用字体
# 找到第一个可用字体后就会使用它
matplotlib.rcParams['font.sans-serif'] = [
    'PingFang HK',
    'Heiti TC',
    'Arial Unicode MS',
    'SimHei',
    'Microsoft YaHei'
]

# 解决负号显示为方块的问题
# 比如坐标轴上出现 -0.1 时，负号可能显示异常
matplotlib.rcParams['axes.unicode_minus'] = False


# =========================
# 3. 固定随机种子
# =========================

# 固定随机种子的作用：
# 让每次运行代码时，数据划分、模型初始化等随机过程尽可能保持一致
#
# 这样做的好处：
# 1. 方便课堂演示
# 2. 方便调试代码
# 3. 方便比较不同模型或不同参数的效果
torch.manual_seed(42)


# =========================
# 4. 加载数据
# =========================

# 数据文件路径
# 注意：请确保 Walmart_Sales.csv 文件和当前 .py 文件在同一个目录下
file_path = "Walmart_Sales.csv"

# 使用 pandas 读取 csv 文件
df = pd.read_csv(file_path)


# =========================
# 5. 数据预处理
# =========================

# 清洗列名
# 原始数据集中有些列名可能带有多余空格
# strip() 可以去掉字符串两边的空格
df.columns = [col.strip() for col in df.columns]

# 删除无用列
# Store 是门店编号（1~45），虽然是数字，但它代表的是类别，不是大小关系
# 门店编号 45 不是"比门店 1 大 45 倍"，所以不能直接当数值特征输入
# 这里用独热编码（One-Hot）把它转成 45 个 0/1 列
# 这样模型就能知道"这是哪一家店"，不同门店的销售基数差异很大
# pd.get_dummies 会为 Store 的每个取值创建一个新列
# drop_first=True 去掉第一列，避免多重共线性（哑变量陷阱）
df = pd.get_dummies(df, columns=['Store'], drop_first=True)

 # Date 是日期字符串，不能直接输入模型
 # 数据集中的日期格式为 DD-MM-YYYY（日-月-年）
 # 例如 05-02-2010 表示 2010 年 2 月 5 日
 # 这里将其解析为日期，并提取年、月、周作为数值特征
 # 这些时间特征可以帮助模型捕捉销售的季节性和趋势规律
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
df.drop(columns=['Date'], inplace=True)

# 独热编码产生的列是 bool 类型，需要转为数值类型才能用于标准化和训练
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)

# 查看数据基本信息
print("数据前 5 行：")
print(df.head())

print("\n数据形状：")
print(df.shape)

print("\n数据列名：")
print(df.columns)


# =========================
# 6. 数据分布可视化
# =========================

# 遍历每一列，绘制该列的数据分布图
# 这样可以帮助学生直观看到每个特征的取值范围和分布情况
#
# 例如：
# Weekly_Sales 的销售额整体分布是什么样？
# Temperature 的温度分布集中在哪些范围？
 # Fuel_Price 的燃油价格分布如何？
 # Holiday_Flag 中假日和非假日的比例如何？
 # 数据分布可视化已跳过，直接进入训练。
 # 如需查看各列分布，可取消下面这段循环的注释。
 # for col in df.columns:
 #     plt.figure(figsize=(6, 4))
 #     sns.histplot(data=df, x=col, kde=True, bins=30)
 #     plt.title(f"{col} 的数据分布")
 #     plt.xlabel(col)
 #     plt.ylabel("频数")
 #     plt.grid(True)
 #     plt.tight_layout()
 #     plt.show()


# =========================
# 7. 特征和目标变量
# =========================

# 特征 X：
# 用来预测每周销售额的输入数据
#
# 这里包括：
# Holiday_Flag（是否假日）
# Temperature（温度）
# Fuel_Price（燃油价格）
# CPI（消费者物价指数）
# Unemployment（失业率）
# Year（年份）
# Month（月份）
# Week（周数）
# Store_2 ~ Store_45（各门店的独热编码，共 44 列）
X = df.drop(columns=['Weekly_Sales'])

# 目标变量 y：
# 也就是我们希望模型预测的结果
# 这里是 Weekly_Sales，即每周销售额
y = df['Weekly_Sales']

print("\n特征 X 前 5 行：")
print(X.head())

print("\n目标变量 y 前 5 行：")
print(y.head())


# =========================
# 8. 划分训练集、验证集、测试集
# =========================

# 原来代码中只有训练集和测试集：
# X_train, X_test, y_train, y_test = train_test_split(...)

# 现在我们加入验证集，所以需要分两步划分。

# 第一步：
# 先从全部数据中划分出 20% 作为测试集
# 剩下的 80% 暂时命名为 X_temp 和 y_temp
#
# test_size=0.2 表示测试集占 20%
# random_state=42 表示固定随机种子，保证每次运行划分结果一致
X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 第二步：
# 再从剩下的 80% 数据中划分出一部分作为验证集
#
# 这里 test_size=0.25 的意思是：
# 从 X_temp 这 80% 数据中，再拿出 25% 作为验证集
#
# 因为：
# 80% × 25% = 20%
#
# 所以最终比例就是：
# 训练集：60%
# 验证集：20%
# 测试集：20%
X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.25,
    random_state=67
)

print("\n训练集大小：", X_train.shape)
print("验证集大小：", X_val.shape)
print("测试集大小：", X_test.shape)


# =========================
# 9. 数据标准化
# =========================

# 为什么要标准化？
#
# 不同特征的量纲和取值范围可能不同：
# 例如：
# Temperature 大约是 -2 ~ 100（华氏温度）
# Fuel_Price 大约是 2 ~ 4
# CPI 大约是 126 ~ 228
# Unemployment 大约是 3 ~ 15
# Holiday_Flag 只有 0 或 1
#
# 如果不标准化，不同特征的数值尺度差别很大，
# 这可能导致模型训练不稳定，或者收敛速度变慢。
#
# 标准化的作用：
# 让每个特征变成均值约为 0、标准差约为 1 的分布。
#
# 标准化后，每个特征都站在一个比较公平的尺度上，
# 梯度下降时更新参数会更加稳定。

# 创建一个 StandardScaler 实例
# StandardScaler 是 sklearn.preprocessing 模块中的一个工具
# 用于将特征缩放为均值为 0，标准差为 1 的分布
# 也称为 Z-score 标准化
scaler = StandardScaler()

# 非常重要：
# 这里只能对训练集使用 fit_transform
#
# fit 的意思是：
# 计算训练集每一列的均值和标准差
#
# transform 的意思是：
# 使用刚才计算出的均值和标准差，对数据进行标准化
#
# 为什么不能对全部数据先 fit_transform 再划分？
#
# 因为那样会导致测试集、验证集的信息提前泄露给模型。
# 测试集和验证集应该假装是模型从来没有见过的新数据。
#
# 所以正确做法是：
# 训练集：fit_transform
# 验证集：transform
# 测试集：transform
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 同样需要对目标变量 y 进行标准化
#
# Weekly_Sales 的数值在百万级别，
# 如果不标准化，MSE 会非常大（百万的平方 = 万亿），
# 导致梯度爆炸，模型无法收敛。
#
# 标准化后 y 也变成均值约为 0、标准差约为 1 的分布，
# 这样损失值和梯度都在合理范围内，训练才能稳定进行。
#
# 评估时需要用 inverse_transform 还原为原始的销售额数值。
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_val_scaled = y_scaler.transform(y_val.values.reshape(-1, 1))
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))


# =========================
# 10. 转换为 PyTorch 张量
# =========================

# PyTorch 中的张量 Tensor 是多维数组
# 它类似于 NumPy 数组，但可以和 PyTorch 的自动求导、GPU 加速等功能配合使用
#
# torch.tensor() 可以将 NumPy 数组或其他数据结构转换为 PyTorch 张量
#
# dtype=torch.float32 表示使用 32 位浮点数
# 深度学习中通常使用 float32，因为它在精度和计算速度之间比较平衡

# 特征 X 本身是二维数据：
# 形状为：
# 样本数 × 特征数
#
# 例如：
# 训练集可能是：
# 3861 × 52
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

# 标签 y 已经在标准化时转换好了，这里只需转为张量
# 标准化后的 y 已经是二维形状 (样本数, 1)，无需再 reshape
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

# 将张量移动到指定设备
# 本代码默认使用 CPU
X_train_tensor = X_train_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)

X_val_tensor = X_val_tensor.to(device)
y_val_tensor = y_val_tensor.to(device)

X_test_tensor = X_test_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)


# =========================
# 11. 构建数据加载器 DataLoader
# =========================

# TensorDataset 可以把特征张量和标签张量组合成一个数据集
# 这样每次取数据时，就可以同时取出 X 和 y
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

# DataLoader 用于把数据集分成一小批一小批的数据
#
# batch_size 表示每次迭代用多少个样本来计算梯度
#
# batch_size 小：
# 例如 1、8、16
# 优点：更新更频繁，有时泛化较好
# 缺点：训练慢，损失曲线抖动大
#
# batch_size 大：
# 例如 128、256、512
# 优点：训练稳定，计算速度快
# 缺点：可能泛化能力差，占用内存大
#
# batch_size=32：
# 是深度学习中非常常用的经验值
# 通常在稳定性、速度、泛化能力之间比较平衡
#
# shuffle=True 表示每个 epoch 都随机打乱训练数据顺序
#
# 为什么训练集需要 shuffle？
#
# 如果每一轮训练都按完全一样的顺序喂给模型，
# 模型可能会受到样本顺序的影响。
#
# shuffle=True 可以避免模型学到“样本顺序的偏见”，
# 让每个 batch 的样本组合更加随机，从而提升泛化能力。
#
# 类比：
# 假设你是老师，要教 60 个学生考试技巧。
# 如果每次上课都让前 10 个是学霸，最后 10 个是学渣，
# 教学过程可能受到顺序影响。
# 如果每次都随机分组，教学效果可能更平均。
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

# 验证集和测试集通常不需要 DataLoader
# 因为验证和测试只需要前向传播，不需要反向传播
# 而且这个数据集很小，可以一次性输入模型进行评估
#
# 验证集和测试集通常也不需要 shuffle
# 因为评估时我们只关心整体指标，保持顺序更方便分析预测结果


# =========================
# 12. 定义线性回归模型
# =========================

# 注意：
# 这里虽然使用了 PyTorch 框架，
# 但是模型本身仍然是一个线性回归模型。
#
# 因为它只有一个线性层：
# nn.Linear(input_dim, 1)
#
# 它表达的数学公式是：
# y = w1*x1 + w2*x2 + ... + wn*xn + b
#
# 也就是：
# 预测周销售额 =
# w1 * Holiday_Flag
# + w2 * Temperature
# + w3 * Fuel_Price
# + ...
# + b
#
# 它和 sklearn 的 LinearRegression 在模型表达能力上是类似的，
# 区别是：
#
# sklearn.LinearRegression：
# 通常使用数学方法直接求解最优参数
#
# 这里的 PyTorch 版本：
# 使用梯度下降一轮一轮更新参数
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # input_dim 表示输入特征数量
        # 例如本数据集有 52 个特征（8 个数值特征 + 44 个门店独热编码列）
        #
        # 1 表示输出 1 个值，也就是周销售额
        #
        # nn.Linear(input_dim, 1) 表示一个线性层
        # 线性层内部会自动创建权重 W 和偏置 b
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        # forward 表示前向传播
        # 输入 x 后，模型输出预测值
        return self.linear(x)


# 创建一个线性回归模型实例
#
# X.shape[0] 是样本数量
# X.shape[1] 是特征数量
#
# 这里要传入的是 X.shape[1]
# 因为模型需要知道每个样本有多少个输入特征
model = LinearRegressionModel(input_dim=X.shape[1])

# 将模型移动到指定设备
model = model.to(device)

print("\n模型结构：")
print(model)


# =========================
# 13. 定义损失函数和优化器
# =========================

# 定义损失函数 Loss Function
#
# MSELoss 是均方误差损失
#
# MSE 的计算方式是：
# 把每个样本的：
# 预测值 - 真实值
# 计算平方
# 然后求平均
#
# MSE 越小，说明预测值和真实值越接近。
#
# 回归问题中经常使用 MSELoss。
criterion = nn.MSELoss()

# 定义优化器 Optimizer
#
# 优化器的作用：
# 根据反向传播得到的梯度，真正更新模型参数。
#
# SGD 是随机梯度下降。
# 这里由于我们使用的是 batch_size=32，
# 所以实际是“小批量随机梯度下降”。
#
# lr 是 learning rate，学习率。
#
# 学习率可以理解为每次更新参数时走多大一步。
#
# 学习率太大：
# 可能导致训练不稳定，甚至越过最低点。
#
# 学习率太小：
# 训练会很慢，可能很久都收敛不了。
#
# 0.01 是一个常见的初始尝试值。
# 这里我们改用 Adam 优化器。
#
# Adam 通常比普通 SGD 更容易训练稳定、收敛更快。
# 它会为每个参数自适应地调整学习率，
# 因此对学习率的选择不像 SGD 那样敏感，
# 在标准化后的特征上表现通常更好，Loss 也能降得更低。
#
# 由于目标值 y 也做了标准化，Loss 是在标准化尺度上计算的，
# 所以 lr=0.01 是一个比较合适的初始值。
optimizer = torch.optim.Adam(
model.parameters(),
lr=0.01
)


# =========================
# 14. 用于可视化的记录变量
# =========================

# 记录每一轮训练集 Loss
train_losses = []

# 记录每一轮验证集 Loss
val_losses = []

# 用于保存验证集表现最好的模型
#
# 为什么要保存“验证集表现最好的模型”？
#
# 因为训练集 Loss 通常会不断下降，
# 但是验证集 Loss 不一定一直下降。
#
# 当模型训练太久时，可能会出现：
# 训练集 Loss 继续下降，
# 但是验证集 Loss 开始上升。
#
# 这种情况说明模型可能开始过拟合。
#
# 所以我们更关心验证集表现最好的那一轮。
best_val_loss = float("inf")
best_model_state = None
best_epoch = 0


# =========================
# 15. 训练模型
# =========================

# epochs 表示训练轮数
#
# 一个 epoch 表示：
# 模型完整看完一遍训练集。
#
# 例如训练集有 300 个样本，batch_size=32
# 那么一个 epoch 中大约会有 10 个 batch
#
# 500 个 epoch 表示：
# 模型会反复看训练集 500 遍
epochs = 500

# 遍历每一个 epoch
for epoch in range(epochs):

    # -------------------------
    # 15.1 训练阶段
    # -------------------------

    # model.train() 表示模型进入训练模式
    #
    # 对于本案例中的线性层来说，train() 和 eval() 的区别不明显。
    #
    # 但是在更复杂的神经网络中，
    # 例如包含 Dropout 或 BatchNorm 的模型，
    # train() 会启用训练时的行为。
    #
    # 所以养成习惯：
    # 训练前写 model.train()
    model.train()

    # 用于记录当前 epoch 的累计损失
    epoch_loss = 0.0

    # 遍历训练数据
    # 每次从 train_loader 中取出一个 batch
    for batch_X, batch_y in train_loader:

        # 第一步：清空上一轮的梯度信息
        #
        # PyTorch 默认会累加梯度。
        # 如果不清空，上一轮的梯度会和这一轮的梯度叠加，
        # 导致参数更新错误。
        optimizer.zero_grad()

        # 第二步：前向传播
        #
        # 将输入数据 batch_X 喂给模型，
        # 得到模型预测值 y_pred
        y_pred = model(batch_X)

        # 第三步：计算损失
        #
        # 比较预测值 y_pred 和真实值 batch_y 的差距
        loss = criterion(y_pred, batch_y)

        # 第四步：反向传播
        #
        # 根据 loss 计算每个参数的梯度
        #
        # 梯度可以理解为：
        # 每个参数应该往哪个方向调整，才能让 loss 变小
        loss.backward()

        # 第五步：更新参数
        #
        # optimizer 根据刚才计算出来的梯度，
        # 更新模型中的权重和偏置
        optimizer.step()

        # 累加当前 batch 的损失值
        epoch_loss += loss.item()

    # 计算当前 epoch 的平均训练损失
    #
    # 因为一个 epoch 中有多个 batch，
    # 所以这里用所有 batch 的 loss 求平均
    avg_train_loss = epoch_loss / len(train_loader)

    # 记录训练损失，用于后续绘图
    train_losses.append(avg_train_loss)


    # -------------------------
    # 15.2 验证阶段
    # -------------------------

    # 验证集的作用：
    # 检查模型在没有参与训练的数据上的表现
    #
    # 注意：
    # 验证阶段不更新参数。
    #
    # 所以这里不会调用：
    # loss.backward()
    # optimizer.step()

    # model.eval() 表示模型进入评估模式
    #
    # 在评估模式下：
    # Dropout 会关闭
    # BatchNorm 会使用训练好的统计信息
    #
    # 虽然本案例没有 Dropout 和 BatchNorm，
    # 但是仍然建议养成这个习惯。
    model.eval()

    # torch.no_grad() 表示不计算梯度
    #
    # 验证时只需要前向传播和计算 loss，
    # 不需要反向传播。
    #
    # 这样可以：
    # 1. 节省内存
    # 2. 提高计算速度
    # 3. 防止误更新模型参数
    with torch.no_grad():
        # 使用验证集进行预测
        val_pred = model(X_val_tensor)

        # 计算验证集损失
        val_loss = criterion(val_pred, y_val_tensor).item()

    # 记录验证集损失，用于后续绘图
    val_losses.append(val_loss)


    # -------------------------
    # 15.3 保存验证集表现最好的模型
    # -------------------------

    # 这里不是保存训练集 Loss 最低的模型，
    # 而是保存验证集 Loss 最低的模型。
    #
    # 原因：
    # 训练集表现好，不一定代表模型对新数据表现好。
    # 验证集没有参与参数更新，
    # 所以验证集表现更能反映模型的泛化能力。
    if val_loss < best_val_loss:
        best_val_loss = val_loss

        # deepcopy 用于复制当前模型参数
        # 如果不用 deepcopy，后续模型继续训练时，
        # best_model_state 可能也会跟着变化
        best_model_state = copy.deepcopy(model.state_dict())

        # 记录最佳模型出现在哪一轮
        best_epoch = epoch + 1

    # 每训练 20 个 epoch 打印一次损失值
    if (epoch + 1) % 20 == 0:
        print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"训练集 Loss: {avg_train_loss:.4f}, "
            f"验证集 Loss: {val_loss:.4f}"
        )


# =========================
# 16. 加载验证集表现最好的模型
# =========================

# 训练结束后，我们不一定使用最后一轮的模型。
#
# 因为最后一轮模型可能已经过拟合。
#
# 所以这里加载验证集 Loss 最低时保存的模型参数。
model.load_state_dict(best_model_state)

print("\n训练完成！")
print(f"验证集表现最好的模型出现在第 {best_epoch} 轮")
print(f"最佳验证集 Loss: {best_val_loss:.4f}")


# =========================
# 17. 可视化训练损失和验证损失
# =========================

# 这张图是本版本最重要的图之一。
#
# 训练集 Loss：
# 模型在训练数据上的误差。
#
# 验证集 Loss：
# 模型在没有参与训练的数据上的误差。
#
# 观察方式：
#
# 如果训练集 Loss 下降，验证集 Loss 也下降：
# 说明模型在学习规律。
#
# 如果训练集 Loss 一直下降，但验证集 Loss 开始上升：
# 说明模型可能过拟合。
#
# 过拟合可以理解为：
# 模型把训练集记得越来越熟，
# 但是面对没见过的数据，表现反而变差。
plt.figure(figsize=(10, 5))

plt.plot(train_losses, label='训练集 Loss')
plt.plot(val_losses, label='验证集 Loss')

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("训练集与验证集 Loss 变化")
plt.legend()
plt.grid(True)
plt.show()


# =========================
# 18. 定义模型评估函数
# =========================

def evaluate_model(model, X_tensor, y_tensor, dataset_name, y_scaler=None):
    """
    评估模型在某个数据集上的效果。

    参数说明：
    model:
        已训练好的 PyTorch 模型

    X_tensor:
        特征张量

    y_tensor:
        真实标签张量

    dataset_name:
        数据集名称，例如：训练集、验证集、测试集

    y_scaler:
        目标变量的标准化器，用于将预测值还原为原始尺度

    返回：
    predictions:
        模型预测值

    y_true:
        真实值

    mae:
        平均绝对误差

    mse:
        均方误差

    rmse:
        均方根误差

    r2:
        R² 决定系数
    """

    # 进入评估模式
    model.eval()

    # 评估阶段不需要计算梯度
    with torch.no_grad():

        # 模型预测
        predictions = model(X_tensor)

        # 如果张量在 GPU 上，需要先移动到 CPU 才能转换为 numpy
        predictions = predictions.cpu().numpy()
        y_true = y_tensor.cpu().numpy()

        # 如果提供了 y_scaler，将标准化后的预测值和真实值还原为原始尺度
        # 这样评估指标的单位才是真实的销售额，便于理解
        if y_scaler is not None:
            predictions = y_scaler.inverse_transform(predictions)
            y_true = y_scaler.inverse_transform(y_true)

    # MAE：
    # Mean Absolute Error，平均绝对误差
    #
    # 可以理解为：
    # 模型平均预测偏差多少。
    #
    # 例如 MAE=150000，
    # 表示模型预测周销售额平均偏差约 15 万美元。
    mae = mean_absolute_error(y_true, predictions)

    # MSE：
    # Mean Squared Error，均方误差
    #
    # 对较大的错误更加敏感。
    mse = mean_squared_error(y_true, predictions)

    # RMSE：
    # Root Mean Squared Error，均方根误差
    #
    # RMSE 和目标变量单位一致，
    # 也可以理解为模型预测的典型误差。
    rmse = mse ** 0.5

    # R²：
    # 决定系数
    #
    # 越接近 1，说明模型解释能力越强。
    #
    # R²=0.8 可以粗略理解为：
    # 模型能解释目标变量中约 80% 的变化。
    r2 = r2_score(y_true, predictions)

    print(f"\n{dataset_name}评估结果：")
    print(f"MAE  平均绝对误差：{mae:.4f}")
    print(f"MSE  均方误差：{mse:.4f}")
    print(f"RMSE 均方根误差：{rmse:.4f}")
    print(f"R²   决定系数：{r2:.4f}")

    return predictions, y_true, mae, mse, rmse, r2


# =========================
# 19. 分别评估训练集、验证集、测试集
# =========================

# 训练集评估：
# 可以观察模型对训练数据拟合得怎么样
train_predictions, train_true, train_mae, train_mse, train_rmse, train_r2 = evaluate_model(
    model,
    X_train_tensor,
    y_train_tensor,
    "训练集",
    y_scaler
)

# 验证集评估：
# 可以观察模型对训练过程中没参与参数更新的数据表现如何
val_predictions, val_true, val_mae, val_mse, val_rmse, val_r2 = evaluate_model(
    model,
    X_val_tensor,
    y_val_tensor,
    "验证集",
    y_scaler
)

# 测试集评估：
# 最终模型效果应该以测试集为准
test_predictions, test_true, test_mae, test_mse, test_rmse, test_r2 = evaluate_model(
    model,
    X_test_tensor,
    y_test_tensor,
    "测试集",
    y_scaler
)


# =========================
# 20. 绘制预测值 vs 实际值
# =========================

# 这张图用于直观观察模型预测效果。
#
# 横轴是真实周销售额。
# 纵轴是模型预测周销售额。
#
# 图中的红色虚线是理想预测线。
#
# 如果模型预测完全准确，
# 所有点都应该落在这条线上。
#
# 点离这条线越远，
# 说明预测误差越大。
plt.figure(figsize=(8, 6))

plt.scatter(test_true.flatten(), test_predictions.flatten(), alpha=0.6)

# 理想预测线
min_val = min(test_true.min(), test_predictions.min())
max_val = max(test_true.max(), test_predictions.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='理想预测')

plt.xlabel("真实周销售额")
plt.ylabel("预测周销售额")
plt.title("测试集：预测值 vs 真实值")
plt.legend()
plt.grid(True)
plt.show()


# =========================
# 21. 残差分析
# =========================

# 残差 residual 的含义：
#
# 残差 = 真实值 - 预测值
#
# 如果残差接近 0：
# 说明预测比较准确。
#
# 如果残差为正：
# 说明真实值比预测值大，模型预测低了。
#
# 如果残差为负：
# 说明真实值比预测值小，模型预测高了。
#
# 残差分布图可以帮助我们观察：
# 模型的错误是否大致集中在 0 附近。
residuals = test_true - test_predictions

plt.figure(figsize=(8, 5))

sns.histplot(residuals.flatten(), bins=30, kde=True)

plt.title("测试集残差分布：真实值 - 预测值")
plt.xlabel("残差")
plt.ylabel("频数")
plt.grid(True)
plt.show()


# =========================
# 22. 训练集、验证集、测试集指标对比
# =========================

# 将三个数据集的指标整理到一个 DataFrame 中
#
# 这样方便打印查看，也方便后续画图。
metric_df = pd.DataFrame({
    "数据集": ["训练集", "验证集", "测试集"],
    "MAE": [train_mae, val_mae, test_mae],
    "MSE": [train_mse, val_mse, test_mse],
    "RMSE": [train_rmse, val_rmse, test_rmse],
    "R²": [train_r2, val_r2, test_r2]
})

print("\n模型指标对比：")
print(metric_df)


# 绘制 MAE 对比图
#
# MAE 越小越好。
#
# 如果训练集 MAE 很小，
# 但验证集和测试集 MAE 明显变大，
# 可能说明模型过拟合。
plt.figure(figsize=(8, 5))

sns.barplot(data=metric_df, x="数据集", y="MAE")

plt.title("训练集、验证集、测试集 MAE 对比")
plt.xlabel("数据集")
plt.ylabel("MAE 平均绝对误差")
plt.grid(True)
plt.show()


# 绘制 R² 对比图
#
# R² 越接近 1 越好。
#
# 如果训练集 R² 很高，
# 但验证集和测试集 R² 明显降低，
# 也可能说明模型泛化能力不够好。
plt.figure(figsize=(8, 5))

sns.barplot(data=metric_df, x="数据集", y="R²")

plt.title("训练集、验证集、测试集 R² 对比")
plt.xlabel("数据集")
plt.ylabel("R² 决定系数")
plt.grid(True)
plt.show()


# =========================
# 23. 展示模型权重
# =========================

# 模型权重可以帮助我们理解：
# 每个特征对预测结果的影响方向和影响大小。
#
# 注意：
# 因为我们对特征做了标准化，
# 所以这里的权重是“标准化后的特征权重”。
#
# 这意味着：
# 权重不能直接理解为原始单位下的影响。
#
# 例如：
# 不能简单说温度升高 1 度，销售额增加多少。
#
# 更合理的理解是：
# 在标准化后的尺度上，
# 哪些特征对模型预测结果影响更明显。
weights = model.linear.weight.detach().cpu().numpy().flatten()

feature_importance = pd.Series(
weights,
index=X.columns
)

# 门店编号（Store_2 ~ Store_45）是独热编码产生的列，
# 它们代表的是"是哪家店"这种类别信息，而不是真正的影响因子。
# 把它们画进特征权重图意义不大，反而会让真正的连续特征被挤到角落看不清。
# 所以这里只展示非 Store_ 的特征权重。
display_importance = feature_importance[
~feature_importance.index.str.startswith("Store_")
]

display_importance.plot(
kind='barh',
     figsize=(10, 6),
     title="线性模型中的特征权重"
 )

plt.xlabel("权重大小")
plt.grid(True)
plt.show()


# =========================
# 24. 保存模型权重文件
# =========================

# 保存模型参数
#
# state_dict 中保存的是模型的权重和偏置，
# 也就是模型训练后学到的参数。
#
# 保存后，下次可以重新创建同样结构的模型，
# 然后用 load_state_dict() 加载这些参数。
torch.save(model.state_dict(), "linear_regression_model.pt")

print("\n模型参数已保存到 linear_regression_model.pt")
