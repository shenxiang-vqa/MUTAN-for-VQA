import torch
import torch.nn as nn
from torchsummary import summary
from AttenMUTANmodels.image_feature import ImageEncoder
from AttenMUTANmodels.bert_encoder import QuestionEncoder
import torch.nn.functional as F
from torch.autograd import Variable
from configs import config
from transformers import BertModel,BertTokenizer

class MutanFusion(nn.Module):

    def __init__(self, config, visual_embedding=True, question_embedding=True):
        super(MutanFusion, self).__init__()
        self.config = config
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
        #这段代码创建了一个包含多个线性层的列表 list_linear_hq，列表中的每个线性层都具有相同的输入维度 self.config.dim_hq 和输出维度 self.config.dim_mm。列表的长度由 self.config.R 决定。

        #在神经网络中，当需要对同一层结构进行多次应用时，可以使用列表或其他数据结构来存储这些层，这样可以方便地对它们进行迭代处理。在这种情况下，list_linear_hq 存储了多个相同结构的线性层，每个线性层的参数是独立训练的，但它们共享相同的结构和参数数量。
        self.list_linear_hq = nn.ModuleList([
            nn.Linear(self.config.dim_hq, self.config.dim_mm)
            for i in range(self.config.R)])
        self.list_linear1d_hv = nn.ModuleList([
            nn.Linear(self.config.dim_v, self.config.dim_mm)
            for i in range(self.config.R)])
        self.list_linear1d_hq = nn.ModuleList([
            nn.Linear(self.config.dim_q, self.config.dim_mm)
            for i in range(self.config.R)])

    def forward(self, input_v, input_q):
        if input_v.dim() != input_q.dim() and input_v.dim() != 2:
            raise ValueError
        #input_v = self.imgEncoder(input_v)# [batch,2048]
        #print("V经过ResNet变为了：{}".format(input_v.size()))
        #input_q = self.qstEncoder(input_q)# [batch,2048]
        #print("Q经过Embedding变为了：{}".format(input_q.size()))
        batch_size = input_v.size(0) #batch*196 or batch

        if self.visual_embedding and self.question_embedding:
            x_v = F.dropout(input_v, p=self.config.dropout_v, training=self.training)
            #print('在Fusion1D经过Drop后的大小为：{}'.format(x_v.size()))
            x_v = self.linear_v(x_v) #[batch,dim_hv]
            #print("V的映射大小为：{}".format(x_v.size()))
            x_v = getattr(F, self.config.activation_v)(x_v)
            x_q = F.dropout(input_q, p=self.config.dropout_q, training=self.training)
            x_q = self.linear_q(x_q) #[batch,dim_q]
            #print("Q的映射大小为：{}".format(x_q.size()))
            x_q = getattr(F, self.config.activation_q)(x_q)

            x_mm = []  # x_mm是一个注意力头的列表，每个元素为[batch*196,dim_mm]
            for i in range(self.config.R):
                x_hv = F.dropout(x_v, p=self.config.dropout_hv, training=self.training)
                x_hv = self.list_linear_hv[i](x_hv)  # 此时的x_hv大小是[196,310]但是W参数大小是[620,510]，做不了矩阵乘法

                x_hq = F.dropout(x_q, p=self.config.dropout_hq, training=self.training)
                x_hq = self.list_linear_hq[i](x_hq)

                x_mm.append(torch.mul(x_hq, x_hv))
        else:
            x_v = input_v
            x_q = input_q
            x_mm = []  # x_mm是一个注意力头的列表，每个元素为[batch*196,dim_mm]
            for i in range(self.config.R):
                x_hv = F.dropout(x_v, p=self.config.dropout_hv, training=self.training)
                x_hv = self.list_linear1d_hv[i](x_hv)  # 此时的x_hv大小是[196,310]但是W参数大小是[620,510]，做不了矩阵乘法

                x_hq = F.dropout(x_q, p=self.config.dropout_hq, training=self.training)
                x_hq = self.list_linear1d_hq[i](x_hq)

                x_mm.append(torch.mul(x_hq, x_hv))



        x_mm = torch.stack(x_mm, dim=1)
        """
        
        torch.stack(x_mm, dim=1) 表示沿着指定维度 dim 对列表 x_mm 中的张量进行堆叠。在这种情况下，它会把 x_mm 中的张量沿着第二个维度进行堆叠。

        假设 x_mm 是一个具有三个元素的列表，每个元素都是 [196, 4] 的张量。那么 torch.stack(x_mm, dim=1) 将会把这三个张量在第二个维度上堆叠起来，生成一个新的张量，其维度将是 [196, 3, 4]。
        """
        #print(x_mm.size())
        x_mm = x_mm.sum(1).view(batch_size, self.config.dim_mm)
        #print("xmm的大小为：{}".format(x_mm.size()))

        return x_mm


#已更新
class MutanFusion2d(MutanFusion):

    def __init__(self, config, visual_embedding=True, question_embedding=True):
        super(MutanFusion2d, self).__init__(config,
                                            visual_embedding,
                                            question_embedding)
    def forward(self, input_v, input_q):
        if input_v.dim() != input_q.dim() and input_v.dim() != 3:
            raise ValueError
        # #注意下面注释掉的部分，不是训练用的，是下面的if name == main 测试用的
        # input_v = self.imgEncoder(input_v)##[batch,2048,14,14]
        # input_v = input_v.view(input_v.size(0), 2048, -1)  # [batch,2048,196]
        # input_v = torch.transpose(input_v, 1, 2)  # [batch,196,2048]
        # #print("图像嵌入大小为：{}".format(input_v.size()))
        # input_q = self.qstEncoder(input_q)#[batch,2048]
        # input_q = input_q.unsqueeze(1).expand(input_q.size(0), 196,
        #                                         input_q.size(1))  # [batch,196,2400]相当于将一个向量复制为一个张量
        # #print("文本嵌入大小为：{}".format(input_q.size()))
        batch_size = input_v.size(0)
        weight_height = input_v.size(1)
        if not input_v.is_contiguous():#检查张量是否为连续的，如果不是，转换为连续的张量
            input_v = input_v.contiguous()
        if not input_q.is_contiguous():
            input_q = input_q.contiguous()
        x_v = input_v.view(batch_size * weight_height, self.config.dim_v)
        #print('送到Fusion1D的大小为：{}'.format(x_v.size()))
        x_q = input_q.view(batch_size * weight_height, self.config.dim_q)

        x_mm = super().forward(x_v, x_q) #[batch*wh , dim_mm]
        #print("经过1d的Fusion出来变为：{}".format(x_mm.size()))
        x_mm = x_mm.view(batch_size, weight_height, self.config.dim_mm) #[batch,196,dim_mm]
        #print("view变为：{}".format(x_mm.size()))
        return x_mm

# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     config = config()
#     v = torch.randn(1,3,448,448).to(device)
#     bert = BertModel.from_pretrained(r"D:\data\Pretrained_data\BERT")
#     tokenizer = BertTokenizer.from_pretrained(r"D:\data\Pretrained_data\BERT")
#     text = "This is a dog"
#     tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
#     print(tokens.size())
#     print("文本编码的token向量为：{}".format(tokens))
#     qst = tokens.unsqueeze(0).to(device)  # 添加一个维度，使其成为形状为 [1, max_seq_length] 的张量
#     model = MutanFusion2d(bert,config).to(device)
#     out = model(v,qst)
#     print(out)