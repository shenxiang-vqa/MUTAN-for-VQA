import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from data_loader_bert import get_loader
from baseline_models.bert_encoder import QuestionEncoder
from configs import config

# #npy文件读取
# test=np.load('D:/data/VQA_data/VQAv2/test448.npy',encoding = "latin1",allow_pickle=True)  #加载文件
# doc = open('1.txt', 'a')  #打开一个存储文件，并依次写入
# print(test, file=doc)  #将打印内容写入文件中

#查看dataloader
# config = config()
# data_loader = get_loader(
#     input_dir=config.input_dir,
#     input_vqa_train='train448.npy',
#     input_vqa_valid='valid448.npy',
#     input_vqa_test='test448.npy',
#     max_qst_length=config.max_qst_length,
#     max_num_ans=config.max_num_ans,
#     batch_size=4,
#     num_workers=config.num_workers)
# for data in data_loader['test']:
#     questions = data['question']
#     images = data['image']
#     question_id = data['qst_id']
#     #answers = data['answer_label']
#     print(questions.size(0))
#     break

class Mymodel(nn.Module):
    def __init__(self):
        super(Mymodel).__init__()
        self.fc = nn.ModuleList([
            nn.Linear(310, 510)
            for i in range(5)])
    def forward(self,x):
        y = self.fc(x)
        return
model = Mymodel()
print(model)