import torch
import torch.nn as nn
from AttenMUTANmodels.image_feature import ImageEncoder
from AttenMUTANmodels.bert_encoder import QuestionEncoder
from AttenMUTANmodels.fusion import MutanFusion , MutanFusion2d
import torch.nn.functional as F
class My_Att_MUTAN(nn.Module):
    def __init__(self,bert,config):
        super(My_Att_MUTAN,self).__init__()
        #Encoder
        self.config = config
        self.img_enccoder = ImageEncoder()
        self.qst_encoder = QuestionEncoder(bert,self.config,freeze_bert=True)
        self.fusion1D = MutanFusion(self.config,visual_embedding=False,question_embedding=False)
        self.fusion2D = MutanFusion2d(self.config)
        #conv att
        self.conv1 = nn.Conv2d(self.config.dim_mm,
                                    self.config.att_dim_hv, 1, 1)
        self.linear_q_att = nn.Linear(self.config.dim_q,
                                      self.config.att_dim_hq)
        self.conv2 = nn.Conv2d(self.config.att_dim_hv,
                                  self.config.att_nb_glimpses, 1, 1)
        self.linear_v_cat = nn.Linear(self.config.dim_v,int(self.config.dim_v / config.att_nb_glimpses))
        self.fc = nn.Linear(self.config.dim_mm,self.config.num_ans)
    def forward(self,input_v,input_q):
        #input change
        v = self.img_enccoder(input_v)#[batch,2048,14,14]
        h = v.size(2)
        w = v.size(3)
        batch_size = input_v.size(0)
        v = v.view(v.size(0),2048,-1) #[batch,2048,196]
        x_v = torch.transpose(v,1,2) #[batch,196,2048]
        q = self.qst_encoder(input_q)#[batch,2400]
        x_q = q.unsqueeze(1).expand(q.size(0),h*w,self.config.dim_q) #[batch,196,2400]
        #2D fusion
        fusion2D_features = self.fusion2D(x_v,x_q) #[batch,196,510]
        fusion2D_features = torch.transpose(fusion2D_features,1,2).view(fusion2D_features.size(0),self.config.dim_mm,h,w)#[batch,510,14,14]

        #conv softmax to get att_score
        conv1_features = self.conv1(fusion2D_features) #[batch,310,14,14]
        conv1_features = F.dropout(conv1_features,
                          p=self.config.att_dropout_mm,
                          training=self.training)
        conv1_features = getattr(F, self.config.att_activation_v)(conv1_features)#[batch,310,14,14]
        conv2_features = self.conv2(conv1_features)#[batch,2,14,14] 2个注意力视角
        conv2_features = F.dropout(conv2_features,
                                   p=self.config.att_dropout_mm,
                                   training=self.training)
        conv2_features = getattr(F, self.config.att_activation_v)(conv2_features)  # [batch,2,14,14]

        #get attention \hat(v)
        list_att_split = torch.split(conv2_features, 1, dim=1)
        list_att = []  # 完成下面循环后，这里面一共是2个注意力视角（两个元素）的注意力分布。
        for x_att in list_att_split:
            x_att = x_att.contiguous()
            x_att = x_att.view(batch_size, w*h)
            x_att = F.softmax(x_att)
            list_att.append(x_att)
        self.list_att = [x_att.data for x_att in list_att] #转为tensor
        list_v_att = []
        for i, x_att in enumerate(list_att):
            x_att = x_att.view(batch_size,
                               w * h,
                               1)
            x_att = x_att.expand(batch_size,
                                 w * h,
                                 self.config.dim_v)
            x_v_att = torch.mul(x_att, x_v) #[batch,196,2048]
            x_v_att = x_v_att.sum(1) #[batch,1,2048]
            x_v_att = x_v_att.view(batch_size, self.config.dim_v) #[batch,2048] 这个就是注意力出来的一个v向量
            list_v_att.append(x_v_att)
        list_v = []
        for glimpse_id, x_v_att in enumerate(list_v_att):
            x_v = F.dropout(x_v_att,
                            p=self.config.dropout_v,
                            training=self.training)
            x_v = self.linear_v_cat(x_v) #[batch,1024]
            x_v = getattr(F, self.config.activation_v)(x_v)
            list_v.append(x_v)
        x_v = torch.cat(list_v, 1)  # 这个是最终的hat v ， 用于和q再次做fusion [batch,2048]

        #fusion1D and classify
        fusion1D_features = self.fusion1D(x_v,q) #[batch,dim_mm]
        y = self.fc(fusion1D_features) #[batch,1000]
        return y
