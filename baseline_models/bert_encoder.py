import torch
import torch.nn as nn
from transformers import BertModel,BertTokenizer
from torchsummary import summary
from configs import config
# use pretrained of bert to do embedding
"""
If you use this script, change the way the tokenizer is used in data_loader.py
"""
class QuestionEncoder(nn.Module):
    def __init__(self,bert,config,freeze_bert=True):
        #bert = BertModel.pretrained("your bert pretrained model path")
        super(QuestionEncoder,self).__init__()
        self.config = config
        self.embedding = bert
        self.embedding_dim = self.embedding.config.hidden_size  # BERT模型的输出大小即为embedding维度
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(self.embedding_dim,self.config.hidden_size,self.config.num_layers,batch_first=True)
        self.fc = nn.Linear(self.config.hidden_size,self.config.embed_size)
        #freeze param of bert 
        if freeze_bert:
            for param in self.embedding.parameters():
                param.requires_grad = False
    def forward(self,question):
        #print("QstEncoder的原始的问题大小为：{}".format(question.size()))
        embed = self.embedding(question)[0]#[batch,max_seq_length,self.embedding_dim=768]
        #print("QstEncoder的原始的bert嵌入大小为：{}".format(embed.size()))
        out,(hn,cn) = self.lstm(embed) # out ==> [batch,max_seq_length,hidden_size] , hn ==> [num_layers,batch,hidden_size], cn ==> [num_layers,batch,hidden_size]
        features = out[:,-1,:] #[batch,hidden_size]
        features = self.fc(features)#[batch,embed_size]
        return features
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert = BertModel.from_pretrained(r"D:\data\Pretrained_data\BERT")
    tokenizer = BertTokenizer.from_pretrained(r"D:\data\Pretrained_data\BERT")
    text = "This is a dog"
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    print(tokens.size())
    print("文本编码的token向量为：{}".format(tokens))
    qst = tokens.unsqueeze(0).to(device)  # 添加一个维度，使其成为形状为 [1, max_seq_length] 的张量
    print(qst.size())
    config = config()
    model = QuestionEncoder(bert,config,freeze_bert=True).to(device)

    out = model(qst)
    print(out)
    print(out.size())
