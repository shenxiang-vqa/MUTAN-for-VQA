import torch
import torch.nn as nn
from torchsummary import summary
from baseline_models.image_feature import ImageEncoder
from baseline_models.MUTANmodel import QuestionEncoder
import torch.nn.functional as F
from baseline_models.MUTANmodel import MutanModel
import time
from configs import config
from transformers import BertModel,BertTokenizer
from AttenMUTANmodels.My_Att_MUTANmodel import My_Att_MUTAN



# #baseline
# start_time = time.time()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# config = config()
# v = torch.randn(1,3,448,448).to(device)
# bert = BertModel.from_pretrained(r"D:\data\Pretrained_data\BERT")
# tokenizer = BertTokenizer.from_pretrained(r"D:\data\Pretrained_data\BERT")
# text = "This is a dog"
# tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
# print(tokens.size())
# print("文本编码的token向量为：{}".format(tokens))
# qst = tokens.unsqueeze(0).to(device)  # 添加一个维度，使其成为形状为 [1, max_seq_length] 的张量
# model = MutanModel(bert,config).to(device)
# out = model(v,qst)
# end_time = time.time()
# a_sample_time = end_time - start_time
# print("单个样本走一遍模型的正向传播所用时间为：{}s".format(a_sample_time))
# print("模型输出的大小为：{}".format(out.size()))  # 预期大小为[1,1000]
# print("模型的输出为：{}".format(out))

#Att Model
start_time = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = config()
v = torch.randn(1,3,448,448).to(device)
bert = BertModel.from_pretrained(r"D:\data\Pretrained_data\BERT")
tokenizer = BertTokenizer.from_pretrained(r"D:\data\Pretrained_data\BERT")
text = "This is a dog"
tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
print(tokens.size())
print("文本编码的token向量为：{}".format(tokens))
qst = tokens.unsqueeze(0).to(device)  # 添加一个维度，使其成为形状为 [1, max_seq_length] 的张量
model = My_Att_MUTAN(bert,config).to(device)
out = model(v,qst)
end_time = time.time()
a_sample_time = end_time - start_time
print("单个样本走一遍模型的正向传播所用时间为：{}s".format(a_sample_time))
print("模型输出的大小为：{}".format(out.size()))  # 预期大小为[1,1000]
print("模型的输出为：{}".format(out))
