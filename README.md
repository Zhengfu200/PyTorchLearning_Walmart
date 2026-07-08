# PyTorchLearning_Walmart

这是锐意升学——依力老师的作业项目。

## 项目简介

本项目是一个基于 PyTorch 的机器学习练习项目，主要围绕线性回归模型展开。项目使用 Walmart 周销售数据进行销售额预测练习，同时包含一个基于研究生录取数据的线性回归示例，用于巩固 PyTorch 张量处理、模型训练、数据划分和模型保存等基础知识。

项目内容包括：

- `Walmart_v1.py` / `Walmart copy.py`：基于 `Walmart_Sales.csv` 数据集，使用 PyTorch 线性回归模型预测 Walmart 每周销售额。
- `LR.py` / `LR_jupyter.ipynb`：基于 `Admission_Predict.csv` 数据集，使用 PyTorch 线性回归模型预测研究生录取概率。
- `linear_regression_model.pt`：训练后保存的线性回归模型权重文件。
- `Walmart_Sales.csv` / `Admission_Predict.csv`：项目练习使用的数据集。

## 环境依赖

- Python 3
- pandas
- PyTorch (`torch`)

## 数据划分

项目中采用训练集、验证集和测试集划分方式：

- 训练集：用于模型训练和参数更新。
- 验证集：用于观察模型表现、辅助判断是否过拟合。
- 测试集：用于训练结束后的最终效果评估。

## 致谢

感谢锐意升学——依力老师布置的课程作业和学习指导。
