## MUTAN-for-VQAv2

&emsp;&emsp;​​This project is to reproduce the resulting code from the paper MUTAN: Multimodal Tucker Fusion for Visual Question Answering, and I will teach you in detail how to run through this code, including the processing of the dataset.
## 1.背景介绍

&emsp;&emsp;​​此项目是基于MUTAN这篇论文而写的代码，其论文地址和论文源码如下。

论文地址：https://arxiv.org/abs/1705.06676

源码地址：https://github.com/Cadene/vqa.pytorch

我对多模态特征融合(双线性池化方法)的总结在链接：https://blog.csdn.net/2301_78651472/article/details/136592162

&emsp;&emsp;​​我们需要解决的问题是视觉问答，再具体一点是视觉问答里面的特征融合部分。在过去，还没有那么大的预训练模型出现之前，对于VQA任务最重要的就是做两种不同模态之间的特征融合。我们希望通过模态融合来学习到一个联合特征表示。在过去的那几年，模态融合做的比较好的代表就是：**双线性（Bilinear）模型 or 双线性池化**，但是这种原始的Bilinear再后续的线性层会产生大量的参数。因此后续工作都是对Bilinear做降维处理，其中典型的代表有：**MCBP、MLBP、MFBP、MUTAN**等工作。我们今天想介绍的就是MUTAN这篇工作，原理大家可以自行去论文里面学习，我们这个项目重点是为了教会大家如何跑起来我们的项目。

![image](https://github.com/nuistzimoli/MUTAN-for-VQA/blob/main/Image/mutan.png)



## 2.run

### 2.0预训练模型下载

&emsp;&emsp;​​我们要下载BERT的训练预训练模型，网址为：https://huggingface.co/google-bert/bert-base-uncased/tree/main

&emsp;&emsp;​​将下面文件下载到路径：D:\data\Pretrained_data\BERT下面

![image](https://github.com/nuistzimoli/MUTAN-for-VQA/blob/main/Image/bertpath.png)



### 2.1数据下载

&emsp;&emsp;​​在官网分别下载下面的数据：https://visualqa.org/download.html

1.![image](https://github.com/nuistzimoli/MUTAN-for-VQA/blob/main/Image/dataannotation.png)

2.![image](https://github.com/nuistzimoli/MUTAN-for-VQA/blob/main/Image/dataquestion.png)

3.![image](https://github.com/nuistzimoli/MUTAN-for-VQA/blob/main/Image/dataimage.png)

&emsp;&emsp;​​数据存放地址为：D:/data/VQA_data/VQAv2，压缩包下载到这个地址，然后创建三个文件夹，annotations，images，questions。分别把下载的压缩文件解压到对应的文件夹里面。如下：
![image](https://github.com/nuistzimoli/MUTAN-for-VQA/blob/main/Image/jieyaannotation.png)

![image](https://github.com/nuistzimoli/MUTAN-for-VQA/blob/main/Image/jieyaimage.png)

![image](https://github.com/nuistzimoli/MUTAN-for-VQA/blob/main/Image/jieyaquestion.png)



### 2.2数据处理

&emsp;&emsp;​​我们先来看看数据处理的过程！


&emsp;&emsp;​​数据处理的脚本文件在utils文件夹里面，下面我们按顺序依次运行即可。

&emsp;&emsp;​​首先运行，resize_images.py脚本，它的目的是为了把原始的coco图像都变换为448x448大小，运行结束后会在D:/data/VQA_data/VQAv2这个路径下产生一个resize_images文件夹，该文件夹和images文件夹存放的数据是相同的，只不过图像大小不同而已。

&emsp;&emsp;​​然后运行make_vacabs_for_questions_answers.py脚本，这个脚本是为了获取问题的词汇表以及答案的词汇表。运行结束后在D:/data/VQA_data/VQAv2这个路径下产生两个txt文件，分别为：vocab_questions.txt和vocab_answers.txt。

&emsp;&emsp;​​最后运行build_vqa_inputs.py脚本。该脚本是为了做一个数据集，数据格式是字典类型，每一个字典代表一个样本，示例在上面图片可见。运行后会在D:/data/VQA_data/VQAv2这个路径下产生三个文件，分别为：train.npy , test.npy , valid.npy 。

### 2.3 创建自己的DataSet和dataloader

&emsp;&emsp;​​对于dataloader我们有两个脚本文件，分别为：data_loader.py和data_loader_bert.py。

&emsp;&emsp;​​其中：data_loader.py用的是question_vocab，且做词嵌入用的是torch里面的Embedding，我们并没有为它写模型脚本，如果想尝试用它训练的话，请自行更改模型代码。

&emsp;&emsp;​​对于data_loader_bert.py脚本而言，其实就是把.npy文件里的样本读取进来。里面的实现就是正常的DataSet和DataLoader的方法，值得关注的是，如果你不会创建自己的DataSet，请你去官方文档学习一下，日后还会用得到，我觉得DataSet里面最终要的就是self.data这个东西，因为一般它就是我们的数据对象。

### 2.4修改train代码

&emsp;&emsp;​​对于训练部分，你需要的修改如下：

- 如果训练baseline模型，请注释掉run.py的第三行，把第二行注释去掉即可。
- 如果训练att模型，就正常运行run.py文件即可

### 2.5画图

&emsp;&emsp;​​训练完成后，在MUTAN_Baseline_logs_epoch_20文件夹下会产生很多的txt文件。这些文件是为了方便我们画图（Loss和Acc的变化）使用的。

&emsp;&emsp;​​首先进入到utils/ploter.py脚本下：您可能要修改如下路径，要对应好你自己的路径，不然会报错

![image](https://github.com/nuistzimoli/MUTAN-for-VQA/blob/main/Image/1.png)



### 2.6评估

&emsp;&emsp;​​模型跑完会在MUTAN_Baseline_model_epoch_20文件夹下产生一个best_model.pt文件，我们需要用这个文件做VQA的评估。

&emsp;&emsp;​​1.对验证集评估

- 先在E:\Python_Code\Study\Vqa\FeatureFusion\MUTAN\Evaluate_Vqav2\Results下创建一个名为：v2_OpenEnded_mscoco_val2014_fake_results.json的空文件。

- 运行get_result.py文件即可，运行结束后，v2_OpenEnded_mscoco_val2014_fake_results.json文件就被写入了东西

- 然后我们来到E:\Python_Code\Study\Vqa\FeatureFusion\MUTAN\Evaluate_Vqav2下的evaluate.py文件

- 需要修改的地方可能为：（除了它其余地方不要改）

  ![image](https://github.com/nuistzimoli/MUTAN-for-VQA/blob/main/Image/2.png)

- 然后运行evaluate.py即可，就得到了训练好的模型对验证集的评估。

2.对测试集评估

&emsp;&emsp;​​这里有些不同，因为测试集没有annotations文件，所以我们需要先得到测试集上的结果，然后去官方提交结果评估。

- 首先修改get_result.py文件
- 注释掉32行
- 注释掉35行，取消注释36行
- 注释掉37行，取消注释38行
- 在E:\Python_Code\Study\Vqa\FeatureFusion\MUTAN\Evaluate_Vqav2\Results下创建一个名为：v2_OpenEnded_mscoco_test2015_fake_results.json的空文件。
- 把最后一行phase='valid'改为phase='test'
- 运行get_result.py就得到了test集的结果
- 然后打开官网：https://eval.ai/web/challenges/challenge-page/830/phases
- 先注册账号
- 然后按照下面执行

![image](https://github.com/nuistzimoli/MUTAN-for-VQA/blob/main/Image/3.png)

![image](https://github.com/nuistzimoli/MUTAN-for-VQA/blob/main/Image/4.png)

![image](https://github.com/nuistzimoli/MUTAN-for-VQA/blob/main/Image/5.png)

&emsp;&emsp;​​最后点击submit即可，等待五分钟左右即可看到结果。

### 


## 3.实验结果

### 3.1 Atten模型在训练集和验证集上的表现

![image](https://github.com/nuistzimoli/MUTAN-for-VQA/blob/main/Image/Att_Train2Val_Loss.png)
![image](https://github.com/nuistzimoli/MUTAN-for-VQA/blob/main/Image/Att_Train2Val_Acc.png)

### 3.2 Baseline模型和Atten模型在测试集上的表现

#### 3.2.1 Baseline模型

| 模型 | Test-Dev | Test-std |  
| :--: | :--: | :--: |  
| 源码 | 60.17 | 58.16 |  
| Ours | 51.22 | 51.5 |

#### 3.2.2 AttenMUTAN模型

| 模型 | Test-Dev | Test-std |  
| :--: | :--: | :--: |  
| 源码 | 67.42 | 67.36 |  
| Ours | 52.65 | 52.89 |

#### 3.2.3 精度分析

&emsp;&emsp;可以看到无论是Baseline模型还是Atten模型，我们的整体精度都比源论文给的精度要低很多。我认为可能的原因如下：

(1)：文本特征提取部分可能效果不如Glove好

(2)：在模型参数初始化上，本代码没有做任何调整

(3)：模型超参数和原文没保持一致，特别是初始学习率和学习率更新的步长，然后训练的epoch也不够，因此模型没有收敛


## 4.可以修改的地方

&emsp;&emsp;​​对于注意力模型而言，我觉得视觉嵌入部分换成VIT，文本特征提取部分换成一个纯的transformer的Encoder，然后将注意力那里重复L次，可能得到的结果会提升2-5个点，但需要大量的计算资源。
