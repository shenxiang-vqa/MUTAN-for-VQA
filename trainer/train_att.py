import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from data_loader_bert import get_loader
from AttenMUTANmodels.My_Att_MUTANmodel import My_Att_MUTAN
from transformers import BertTokenizer, BertModel
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    data_loader = get_loader(
        input_dir=args.input_dir,
        input_vqa_train='train448.npy',
        input_vqa_valid='valid448.npy',
        input_vqa_test='test448.npy',
        max_qst_length=args.max_qst_length,
        max_num_ans=args.max_num_ans,
        batch_size=args.batch_size,
        num_workers=args.num_workers)
    ans_vocab_size = data_loader['train'].dataset.ans_vocab.vocab_size
    ans_unk_idx = data_loader['train'].dataset.ans_vocab.unk2idx
    bert = BertModel.from_pretrained(r"D:\data\Pretrained_data\BERT")
    model = My_Att_MUTAN(bert,args).to(device)
    criterion = nn.CrossEntropyLoss()

    if args.resume_epoch!=0:
        model = torch.load(args.saved_model)
        torch.cuda.empty_cache()

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    early_stop_threshold = 3
    best_loss = 99999
    val_increase_count = 0
    stop_training = False
    prev_loss = 9999

    #Start Training
    for epoch in range(args.resume_epoch,args.num_epochs):
        print("Epoch [{} / {}] Start Training : the following will print the loss perstep taken".format(epoch+1,args.num_epochs))
        start_epoch_time = time.time()  # 记录每个 epoch 的开始时间
        for phase in ["train","valid"]:
            running_loss = 0.0
            running_corr_exp1 = 0
            running_corr_exp2 = 0
            #下面这行代码计算了每个 epoch 中需要执行的总批次数
            batch_step_size = len(data_loader[phase].dataset) / args.batch_size
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()
            for batch_idx, batch_sample in enumerate(data_loader[phase]):
                image = batch_sample["image"].to(device)
                question = batch_sample["question"].to(device)
                label = batch_sample["answer_label"].to(device)
                multi_choice = batch_sample['answer_multi_choice']  # not tensor, list.

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase=="train"):
                    out = model(image,question)
                    _, pred_exp1 = torch.max(out, 1)  # [batch_size]
                    _, pred_exp2 = torch.max(out, 1)  # [batch_size]
                    loss = criterion(out, label.long())
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                pred_exp2[pred_exp2 == ans_unk_idx] = -9999  # 排除对未知答案的影响。
                running_loss += loss.item()
                running_corr_exp1 += torch.stack([(ans == pred_exp1.cpu()) for ans in multi_choice]).any(dim=0).sum()
                running_corr_exp2 += torch.stack([(ans == pred_exp2.cpu()) for ans in multi_choice]).any(dim=0).sum()
                # 每训练10个batch_size，就打印一次当前batch_size的损失 ， 这里走一个步长就是一个batch_size
                if batch_idx % 1 == 0:
                    print('| {} SET | Epoch [{:02d}/{:02d}]   |   Step [{:04d}/{:04d}]   |  Loss: {:.4f}'
                          .format(phase, epoch + 1, args.num_epochs, batch_idx, int(batch_step_size),
                                  loss.item()))
            end_epoch_time = time.time()  # 记录每个 phase 的结束时间
            epoch_loss = running_loss / batch_step_size
            epoch_acc_exp1 = running_corr_exp1.double() / len(data_loader[phase].dataset)  # multiple choice
            epoch_acc_exp2 = running_corr_exp2.double() / len(data_loader[phase].dataset)  # multiple choice
            print('| {} SET | Epoch [{:02d}/{:02d}] | Loss: {:.4f} | Acc(Exp1): {:.4f} | Acc(Exp2): {:.4f} | Time：{:.2f}\n'
                  .format(phase, epoch + 1, args.num_epochs, epoch_loss, epoch_acc_exp1, epoch_acc_exp2,end_epoch_time - start_epoch_time))
            print("-------------------------------------------------------------------------------------------------------------------------------------------------")
            with open(os.path.join(args.log_dir, '{}-{}-log-epoch-{:02}.txt')
                              .format(args.model_name, phase, epoch + 1), 'w') as f:
                f.write(str(epoch + 1) + '\t'
                        + str(epoch_loss) + '\t'
                        + str(epoch_acc_exp1.item()) + '\t'
                        + str(epoch_acc_exp2.item()))
            if phase == 'train':
                scheduler.step()  # 更新学习率
            if phase == 'valid':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    torch.save(model, os.path.join(args.model_dir, 'best_model.pt'))
                if epoch_loss > prev_loss:
                    val_increase_count += 1
                else:
                    val_increase_count = 0
                if val_increase_count >= early_stop_threshold:
                    stop_training = True
                prev_loss = epoch_loss
            # Save the model check points.
        if (epoch + 1) % args.save_step == 0:
            #             pass
            #             torch.save({'epoch': epoch+1, 'state_dict': model.state_dict()},
            torch.save(model,
                       os.path.join(args.model_dir, '{}-epoch-{:02d}.pt'.format(args.model_name, epoch + 1)))

        if stop_training:
            break