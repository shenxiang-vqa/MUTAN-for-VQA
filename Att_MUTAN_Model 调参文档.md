# Att_MUTAN_Model 调参文档

注意，每次尽量只调整1~2个超参数，要按照顺序调整，batch_size和学习率的更新步长以及学习率是有关系的。

## 1.一个epoch的调参

### 1.1 文档1

- **imgEncoder** : ResNet152 as backbone
- **qstEncoder** : Bert as backbone of Embedding ,and LSTM as feature extraction

- **batch_size** : 600
- **learning_rate** : 0.0001
- **step_lr** : 50
- **gamma** : 0.1
- **Loss fuction** ：Cross-Entropy Loss
- **Optimizer** ：Adam
- **no-linear activation fuction** : thanh
- **param init method** : 随机初始化
- **dim_v** : 2048
- **dim_q** : 2400 == embed_size
- **dim_hv** : 620
- **dim_hq** : 310
- **dim_mm** : 510
- **R** : 10 (This is  Number of heads of Mulit-Head attention)
- **dropout** : 0.5 and 0
- **att_nb_glimpses** : 2
- **att_R** : 5
- **Rand** **Seed** ： 本实验暂未设置

| Average Loss | Best  Loss |  Acc1  | Acc2   | Epoch Time | Step=1 Time | Loss变化      |
| :----------: | ---------- | :----: | ------ | :--------: | ----------- | ------------- |
|    3.1155    | 2.3978     | 0.4183 | 0.3860 |  5383.75s  | 8s          | 7.5916-2.3978 |

1个epoch训练约1.5h

### 1.2 文档2

- **imgEncoder** : ResNet152 as backbone
- **qstEncoder** : Bert as backbone of Embedding ,and LSTM as feature extraction

- **batch_size** : 600
- **learning_rate** : 0.0001
- **step_lr** : <!--10-->
- **gamma** : 0.1
- **Loss fuction** ：Cross-Entropy Loss
- **Optimizer** ：Adam
- **no-linear activation fuction** : thanh
- **param init method** : 随机初始化
- **dim_v** : 2048
- **dim_q** : 2400 == embed_size
- **dim_hv** : 620
- **dim_hq** : 310
- **dim_mm** : 510
- **R** : 10 (This is  Number of heads of Mulit-Head attention)
- **dropout** : 0.5 and 0
- **att_nb_glimpses** : 2
- **att_R** : 5
- **Rand** **Seed** ： 本实验暂未设置

- 

| Average Loss | Best  Loss |  Acc1  | Acc2   | Epoch Time | Step=1 Time | Loss变化   |
| :----------: | ---------- | :----: | ------ | :--------: | ----------- | ---------- |
|    3.1155    | 2.3978     | 0.4183 | 0.3860 |  5383.75s  | 8s          | 7.6005-xxx |

1个epoch训练约xxh