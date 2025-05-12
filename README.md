# PyCGH
[![Python](https://img.shields.io/badge/Python-v3.11-14354C.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Nvidia](https://img.shields.io/badge/Nvidia-CUDA_12.x-76B900.svg?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-B5594B.svg)](https://mit-license.org/)

一个用作计算全息基础原理验证的Python程序仓库

借助[CuPy](https://cupy.dev/)库实现GPU上的加速运算

提供了二维和三维全息图的仿真代码以及离线计算的LUT加速算法

## 依赖
[CuPy](https://cupy.dev/)：提供GPU上的Numpy和Scipy功能

    pip install cupy-cuda12x
[Matplotlib](https://matplotlib.org/)：提供可视化数据显示

    pip install matplotlib
[PIL](https://pillow.readthedocs.io/en/latest/)：提供一系列图像处理功能

    pip install pillow
[tqdm](https://tqdm.github.io/): 提供可视化循环进度，用于性能分析

    pip install tqdm

## 索引
### 计算全息的编码方法
- [迂回相位型](#迂回相位型编码方法)
- [干涉模拟](#干涉模拟计算方法)
- [相息图](#相息图)
### 三维计算全息图
- [基本点源法](#点源法)
- [N-LUT算法](#n-lut法)
- [S-LUT算法](#s-lut法)

## 简介
所谓全息图编码，即把复数域上的复振幅分布信息映射到实数域上，并使得该过程可逆，所以我们可以通过重建图像的逆过程求解全息图。

由于平面对平面的衍射过程可简化为一个简单的傅里叶变换形式，故下面在该情形下讨论对全息图的编码方法。
### 迂回相位型编码方法
[代码](Circuitous%20Phase%20Type%20Hologram/main.py)

该方法利用光栅衍射带来的相位差，通过光栅位置编码相位，窗口大小编码振幅，实现了二值化的全息图编码。

目标图像
![](Circuitous%20Phase%20Type%20Hologram/test.bmp)

全息图效果
![](Circuitous%20Phase%20Type%20Hologram/encoded_hologram.png)

重建效果
![](Circuitous%20Phase%20Type%20Hologram/reconstructed_image.png)
### 干涉模拟计算方法
对于一系列光源在光屏上成像的过程，只能记录其振幅信息而丢失了相位信息，通过施加平行光干涉后，可以同时记录振幅和相位信息。

目标图像
![](Computational%20Interference%20Hologram/BIT.jpg)
菲涅尔全息图：在菲涅尔衍射条件下进行干涉的数值模拟

[代码](Computational%20Interference%20Hologram/main.py)

全息图效果
![](Computational%20Interference%20Hologram/encoded_hologram0.png)

重建效果
![](Computational%20Interference%20Hologram/reconstructed_image0.png)

傅里叶全息图：记录光场的频域信息

[代码](Computational%20Interference%20Hologram/fourier.py)

该代码中同时应用的单边带编码方法，有效消除了共轭像

全息图效果
![](Computational%20Interference%20Hologram/encoded_hologram.png)

重建效果
![](Computational%20Interference%20Hologram/reconstructed_image.png)
### 相息图
相息图是假设在整个记录平面内光波振幅为常数的条件下，仅记录波前位相信息的方法。这里以GS算法为例进行验证。

 GS 算法的主要思想就是通过控制全息图平面与成像平面的振幅信息，通过迭代运算使全息图相位分布得到优化。
 
 [代码](kinoform/main.py)
 
 在这个程序中，你可以选择迭代的次数，观察它对全息图质量的影响。这里以10次为例展示效果。

目标图像
![](kinoform/BIT.jpg)

全息图效果
![](kinoform\hologram_image.png)

重建效果
![](kinoform\reconstructed_image.png)

接下来考虑对三维物体的全息图构建。

### 点源法
三维物体可以抽象成一系列离散的点描述。这里通过[CloudCompare](https://www.cloudcompare.org/)软件导出纯文本形式的[点云数据](3D%20test\bun000%20-%20cloud.txt)。

将单个点的贡献对全息图平面线性叠加再进行编码即可得到三维物体对应的全息图，为了方便处理重建图像时选取一与全息图平行的平面进行观察。

全息图效果
![](3D%20test\hologram_image.png)

重建效果
![](3D%20test\reconstructed_image.png)

### N-LUT法
注意到点源产生全息图的相似性，对于z坐标相同的点在全息图上的贡献实际等于在x=0，y=0的点元平移而来。通过适当数量的z轴切片在保留景深的同时大大提高了计算效率。

全息图效果
![](LUT\encoded_hologram0.png)

重建效果
![](LUT\reconstructed_image0.png)

(此处重建算法存在一定问题导致图像被拉伸)
### S-LUT法
为了提高成像质量N-LUT法不得不保存大量切片，这对内存空间的占用提出了极大的挑战。S-LUT算法通过x，y轴的独立性将数据分别存储，实现了LUT的压缩效果。

全息图效果
![](LUT\encoded_hologram.png)

重建效果
![](LUT\reconstructed_image.png)

## 计划
暂无。