import torch
import torch.nn as nn
from torchsummary import summary
from baseline_models.image_feature import ImageEncoder
from baseline_models.bert_encoder import QuestionEncoder
import torch.nn.functional as F
from torch.autograd import Variable
from configs import config
from transformers import BertModel,BertTokenizer

class MutanFusion(nn.Module):

    def __init__(self, bert , config, visual_embedding=True, question_embedding=True):
        super(MutanFusion, self).__init__()
        self.config = config
        self.imgEncoder = ImageEncoder()
        self.qstEncoder = QuestionEncoder(bert , self.config,freeze_bert=True)

        self.visual_embedding = visual_embedding
        self.question_embedding = question_embedding
        # Modules
        if self.visual_embedding:
            self.linear_v = nn.Linear(self.config.dim_v, self.config.dim_hv)
        else:
            print('Warning fusion.py: no visual embedding before fusion')

        if self.question_embedding:
            self.linear_q = nn.Linear(self.config.dim_q, self.config.dim_hq)
        else:
            print('Warning fusion.py: no question embedding before fusion')

        self.list_linear_hv = nn.ModuleList([
            nn.Linear(self.config.dim_hv, self.config.dim_mm)
            for i in range(self.config.R)])

        self.list_linear_hq = nn.ModuleList([
            nn.Linear(self.config.dim_hq, self.config.dim_mm)
            for i in range(self.config.R)])

    def forward(self, input_v, input_q):
        #print(input_v.size())
        #print(input_q.size())
        input_v = self.imgEncoder(input_v)# [batch,2048]
        #print(input_v.size())
        #print("V经过ResNet变为了：{}".format(input_v.size()))
        input_q = self.qstEncoder(input_q)# [batch,2048]
        #print(input_q.size())
        #print("Q经过Embedding变为了：{}".format(input_q.size()))
        batch_size = input_v.size(0)

        if self.visual_embedding:
            x_v = F.dropout(input_v, p=self.config.dropout_v, training=self.training)
            x_v = self.linear_v(x_v) #[batch,dim_v]
            #print("V的映射大小为：{}".format(x_v.size()))
            x_v = getattr(F, self.config.activation_v)(x_v)

        if self.question_embedding:
            x_q = F.dropout(input_q, p=self.config.dropout_q, training=self.training)
            x_q = self.linear_q(x_q) #[batch,dim_q]
            #print("Q的映射大小为：{}".format(x_q.size()))
            x_q = getattr(F, self.config.activation_q)(x_q)

        x_mm = []
        for i in range(self.config.R):

            x_hv = F.dropout(x_v, p=self.config.dropout_hv, training=self.training)
            x_hv = self.list_linear_hv[i](x_hv)
            #print(x_hv.size())

            x_hq = F.dropout(x_q, p=self.config.dropout_hq, training=self.training)
            x_hq = self.list_linear_hq[i](x_hq)
            #print(x_hq.size())

            x_mm.append(torch.mul(x_hq, x_hv))

        x_mm = torch.stack(x_mm, dim=1)
        #print(x_mm.size())
        x_mm = x_mm.sum(1).view(batch_size, self.config.dim_mm)
        #print("xmm的大小为：{}".format(x_mm.size()))

        return x_mm

if __name__ == "__main__":
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
    model = MutanFusion(bert,config).to(device)
    out = model(v,qst)
    print(out)