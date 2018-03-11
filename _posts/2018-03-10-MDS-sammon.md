---
layout: post
title: 多维缩放MDS和Sammon映射
categories: machine-learning
author: CC
tags: machine-learning

---

# 多维缩放MDS和Sammon映射

多维缩放和sammon映射都是比较经典的降维技术，相同点在于期望在将高维数据从原空间映射到低维子空间时，保留数据点之间的距离信息；不同点在于MDS是线性降维，而sammon映射的非线性降维。下面将简单介绍这两种技术。



## 1. 符号说明

假设$D$ 维空间中存在$m$个样本，我们的目标中获得样本在$d'$维空间的表示。

$d_{ij}$：样本${\bf{x}}_i$与${\bf{x}}_j$在$D$维空间的距离（通常为欧式距离）

$d'_{ij}$：样本${\bf{x}}_i$与${\bf{x}}_j$在$d'$维空间的距离（通常为欧式距离）

${\bf{z}}_i$： 样本${\bf{x}}_i$在$d'$维空间的表示，则$d'_{ij}=\vert \vert{\bf{z}}_i-{\bf{z}}_j\vert \vert _2$（欧式距离）



## 2. 多维缩放—— MDS（Mutiple Dimension Scaling）

### 2.1 原理

**假设：**任意两个样本在$d'$维空间中的欧式距离等于原空间中的欧式距离。

令$B =Z^TZ$为降维后样本的内积矩阵，$b_{ij}={\bf{z}}_i^T{\bf{z}}_j$，且$Z$被中心化，即$\sum_i{\bf{z}}_i=0$。易知，矩阵$B$的行之和和列之和均为0，则：

$d^{'2}_{ij}=\vert \vert{\bf{z}}_i-{\bf{z}}_j\vert \vert _2^2$

​      $=\vert \vert{\bf{z}}_i\vert \vert _2^2+\vert \vert{\bf{z}}_j\vert \vert _2^2-2{\bf{z}}_i^T{\bf{z}}_j$

​      $=b_{ii}+b_{jj}-2b_{ij}$ (1)

$\sum_i^m d^{'2}_{ij}=tr(B)+mb_{jj}$ (2)

$\sum_j^m d^{'2}_{ij}=tr(B)+mb_{ii}$ (3)

$\sum_i^m\sum_j^m d^{'2}_{ij}=mtr(B)+mtr(B)=2m tr(B)$ (4)

由(4)得：

$tr(B)=\frac{1}{2m}\sum_i^m\sum_j^m d^{'2}_{ij}$

那么：

$b_{jj}=\frac{1}{m}(\sum_j^m d^{'2}_{ij}-\frac{1}{2m}\sum_i^m\sum_j^m d^{'2}_{ij})$

$b_{ii}=\frac{1}{m}(\sum_i^m d^{'2}_{ij}-\frac{1}{2m}\sum_i^m\sum_j^m d^{'2}_{ij})$

进一步的：

$b_{ij}=\frac{1}{2}(\frac{1}{m}\sum_i^m d^{'2}_{ij}+\frac{1}{m}\sum_j^m d^{'2}_{ij}-\frac{1}{m}\sum_i^m\sum_j^m d^{'2}_{ij}-d^{'2}_{ij})$ (5)

由于我们假定**任意两个样本在$d'$维空间中的欧式距离等于原空间中的欧式距离**，那么，$d^{'2}_{ij}=d^{2}_{ij}$

得到矩阵$B$后，可通过SVD求得${\bf{z}}_i$。

### 2.2 算法流程

**输入：**$D$ 维空间中存在$m$个样本，$d'$

**输出：**：$d'$维空间的$m$个样本表示

（1）计算原始空间中，样本点距离$d_{ij}$

（2）根据公式(5)，求得矩阵$B$

（3）对矩阵$B$进行特征值分解

（4）取$\hat{\Lambda}$为$d'$个特征值所构成的对角矩阵，$\hat{V}$为其对应的特征向量

（5）$Z$ = $\hat{V}\hat{\Lambda}^{1/2}$(维度$m\times d'$)每行是一个样本



> 线性降维 均满足$Z=W^TX$，对$W$的约束不同，形成了不同的算法



## 3. Sammon 映射

**目标：**样本点在原始空间的距离与其在低维空间中的距离应尽可能的接近。

$J = \frac{1}{\sum_{i<j}d_{ij}}\sum_{i<j}\frac{(d_{ij}-d'_{ij})^2}{d_{ij}}$

通过梯度迭代法使目标函数最小化来求得${\bf{z}}_i$。

**从目标函数中可以发现，Sammon映射，降低了大相异度（距离远）在目标函数中主导地位，使得低相异度对目标函数的贡献更大。从而对低相异度有更好的保持。**



> 低相异度表示样本相似，这在实际应用中是一个很重要的特性。
>
> Sammon映射的性能对低维空间初始样本点的选择很敏感，可以用PCA的结果进行初始化。

