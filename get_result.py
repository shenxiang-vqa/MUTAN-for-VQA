#这个脚本的目的是为了得到（train2014/valid2014/test2015.npy）的答案，根据官网的数据格式存储，
import time
from torch.autograd import Variable
import torch
import json
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
from transformers import BertModel,BertTokenizer
from baseline_models.MUTANmodel import MutanModel

def test_model(model, dataloader, itoa, outputfile, use_gpu=False,phase='train'):
    model.eval()  # Set model to evaluate mode
    example_count = 0
    test_begin = time.time()
    outputs = []

    # Iterate over data.
    for data in dataloader[phase]:
        questions = data['question']
        images = data['image']
        question_id = data['qst_id']
        #if is to get result of test_dataset , please comment out answers
        answers = data['answer_label']

        if use_gpu:
            questions, images, answers = questions.cuda(), images.cuda(), answers.cuda()
            #questions, images = questions.cuda(), images.cuda()
        questions, images, answers = Variable(questions).transpose(0, 1), Variable(images), Variable(answers)
        #questions, images = Variable(questions), Variable(images)
        # zero grad
        ans_scores = model(images, questions)
        _, preds = torch.max(ans_scores, 1)

        outputs.extend([{'question_id': question_id[i].item(), 'answer': itoa.idx2word(preds[i].item())} for i in range(question_id.size(0))])

        if example_count % 100 == 0:
            print('(Example Count: {})'.format(example_count))
        # statistics
        example_count += questions.size(0)

    json.dump(outputs, open(outputfile, 'w'))
    print('(Example Count: {})'.format(example_count))
    test_time = time.time() - test_begin
    print('Test Time: {:.0f}m {:.0f}s'.format(test_time // 60, test_time % 60))

if __name__ == '__main__':
    config = config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert = BertModel.from_pretrained(r"D:\data\Pretrained_data\BERT")
    model_path = r'E:\Python_Code\Study\Vqa\FeatureFusion\MUTAN\MUTAN_Baseline_model_epoch_20\best_model.pt'
    ##如果只保存了模型的参数
    # model = MutanModel(bert,config)
    # #model.load_state_dict(torch.load(model_path))  # 使用单卡训练的模型

    #如果保存了模型的结构
    model = torch.load(model_path)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    data_loader = get_loader(
    input_dir=config.input_dir,
    input_vqa_train='train448.npy',
    input_vqa_valid='valid448.npy',
    input_vqa_test='test448.npy',
    max_qst_length=config.max_qst_length,
    max_num_ans=config.max_num_ans,
    batch_size=config.batch_size,
    num_workers=config.num_workers)
    itoa = data_loader['train'].dataset.ans_vocab
    test_model(model,data_loader,itoa,r'E:\Python_Code\Study\Vqa\FeatureFusion\MUTAN\Evaluate_Vqav2\Results\v2_OpenEnded_mscoco_test2015_fake_results.json',use_gpu=True,phase='valid')