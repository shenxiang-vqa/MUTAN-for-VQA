## MUTAN-for-VQAv2

This project is to reproduce the resulting code from the paper MUTAN: Multimodal Tucker Fusion for Visual Question Answering, and I will teach you in detail how to run through this code, including the processing of the dataset.
## 1.背景介绍

​		此项目是基于MUTAN这篇论文而写的代码，其论文地址和论文源码如下。

论文地址：https://arxiv.org/abs/1705.06676

源码地址：https://github.com/Cadene/vqa.pytorch

​		我们需要解决的问题是视觉问答，再具体一点是视觉问答里面的特征融合部分。在过去，还没有那么大的预训练模型出现之前，对于VQA任务最重要的就是做两种不同模态之间的特征融合。我们希望通过模态融合来学习到一个联合特征表示。在过去的那几年，模态融合做的比较好的代表就是：**双线性（Bilinear）模型 or 双线性池化**，但是这种原始的Bilinear再后续的线性层会产生大量的参数。因此后续工作都是对Bilinear做降维处理，其中典型的代表有：**MCBP、MLBP、MFBP、MUTAN**等工作。我们今天想介绍的就是MUTAN这篇工作，原理大家可以自行去论文里面学习，我们这个项目重点是为了教会大家如何跑起来我们的项目。



## 2.run 基线模型

### 2.1数据下载

### 2.2数据处理

### 2.3数据存放位置

### 2.4修改train代码

### 2.5画图

### 2.6评估

## 3.run 注意力模型

### 3.1数据下载

### 3.2数据处理

### 3.3数据存放位置

### 3.4修改train代码

### 3.5画图

### 3.6评估

## 4.可以修改的地方

​				对于注意力模型而言，我觉得视觉嵌入部分换成VIT，文本特征提取部分换成一个纯的transformer的Encoder，然后将注意力那里重复L次，可能得到的结果会提升2-5个点，但需要大量的计算资源。