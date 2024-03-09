import torch
import torch.nn as nn
from torchsummary import summary
from baseline_models.image_feature import ImageEncoder
from baseline_models.bert_encoder import QuestionEncoder
import torch.nn.functional as F
from torch.autograd import Variable
from configs import config
from transformers import BertModel,BertTokenizer
from baseline_models.fusion import MutanFusion

class MutanModel(nn.Module):
    def __init__(self,bert,config):
        super(MutanModel,self).__init__()
        self.config = config
        self.fusion = MutanFusion(bert,self.config)
        self.fc = nn.Linear(self.config.dim_mm,self.config.num_ans)
    def forward(self,input_v,input_q):
        fusion_vector = self.fusion(input_v,input_q) #[batch,dim_mm]
        y = self.fc(fusion_vector)
        # y = nn.functional.softmax(y, dim=1)
        return y
