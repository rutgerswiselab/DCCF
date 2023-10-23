# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.RecModel import RecModel
from models.DMF import DMF
import os
import numpy as np
from utils import utils

import pdb

class DCCF(DMF):
    @staticmethod
    def parse_model_args(parser, model_name='DCCF'):
        parser.add_argument('--sentence-model', type=str, default='paraphrase-distilroberta-base-v1', help='the name of sentence model')
        parser.add_argument('--sample-num', type=int, default=10, help='the number of sampled items')
        parser.add_argument('--attribute-num', type=int, default=2, help='the number of item features')
        parser.add_argument('--std', type=float, default=0.1, help='std of feature distribution')
        return DMF.parse_model_args(parser, model_name)

    def __init__(self, path, dataset, sentence_model, sample_num, attribute_num, std, label_min, label_max, feature_num, user_num, item_num, u_vector_size, i_vector_size,
                 n_layers, random_seed, model_path):
        self.path = path
        self.dataset = dataset
        self.sentence_model = sentence_model
        self.sample_num = sample_num
        self.attribute_num = attribute_num
        self.std = std
        DMF.__init__(self, label_min=label_min, label_max=label_max, feature_num=feature_num,
                          user_num=user_num, item_num=item_num, u_vector_size=u_vector_size,
                          i_vector_size=i_vector_size, n_layers=n_layers, random_seed=random_seed, model_path=model_path)

    def _load_ui_inter_matrix(self, file_path):
        ui_inter_matrix = np.array([[0] * self.item_num for _ in range(self.user_num)])
        is_first = True
        with open(file_path, 'r') as file:
            for line in file:
                if is_first:
                    is_first = False
                    continue
                items = line.split('\t')
                ui_inter_matrix[int(items[0]) - 1][int(items[1]) - 1] = float(items[2])
        return ui_inter_matrix

    def _init_weights(self):
        # Obtain ui interaction matrix
        # ui_inter_matrix = self._load_ui_inter_matrix('../data/ml100k.all.csv')
        # ui_tensor_matrix = torch.from_numpy(ui_inter_matrix).float()
        # self.uid_embeddings = nn.Embedding.from_pretrained(ui_tensor_matrix)
        # self.iid_embeddings = nn.Embedding.from_pretrained(ui_tensor_matrix.t())
        self.uid_embeddings = nn.Embedding(self.user_num, self.ui_vector_size)
        self.iid_embeddings = nn.Embedding(self.item_num, self.ui_vector_size)
        self.feature_embedding = torch.FloatTensor(np.load(os.path.join(self.path, self.dataset + '_' + self.sentence_model + '.npy'))).to('cuda:' + str(torch.cuda.current_device()))

#         self.cos = nn.CosineSimilarity()
            
        self.mlp = nn.ModuleList([nn.Linear(self.ui_vector_size + self.feature_embedding.shape[1], self.ui_vector_size)])
        # self.u_mlp = nn.ModuleList([nn.Linear(self.item_num, self.ui_vector_size)])
        for layer in range(self.n_layers - 1):
            self.mlp.append(nn.Linear(self.ui_vector_size, self.ui_vector_size))
            
        self.expo_prob = torch.FloatTensor(np.load(os.path.join(self.path, self.dataset + '.ips_expo_prob.npy'))).to('cuda:' + str(torch.cuda.current_device()))

    def predict(self, feed_dict):
        check_list = []
        u_ids = feed_dict['X'][:, 0]
        i_ids = feed_dict['X'][:, 1]
        
#         pdb.set_trace()
        sample_item = torch.randint(self.item_num, size=(u_ids.shape[0], self.sample_num)).to('cuda:' + str(torch.cuda.current_device()))
        
        items = torch.cat((i_ids.view(-1,1), sample_item), 1)
        
        items = items.view(-1,self.sample_num+1,1).expand(items.shape[0], self.sample_num+1, self.attribute_num)
        users = u_ids.view(-1,1,1).expand(items.shape[0], items.shape[1], items.shape[2])
        true_items = i_ids.view(-1,1,1).expand(items.shape[0], items.shape[1], items.shape[2])
        
        uid = users.reshape(-1)
        iid = items.reshape(-1)
        fid = true_items.reshape(-1)

        user_embeddings = self.uid_embeddings(uid)
        item_embeddings = self.iid_embeddings(iid)
        feature_embeddings = self.feature_embedding[fid]
        sample_feature_embeddings = feature_embeddings + torch.cuda.FloatTensor(feature_embeddings.shape).normal_(std=self.std)
        
        mlp_input = torch.cat((item_embeddings, sample_feature_embeddings), 1)

        for layer in self.mlp:
            mlp_input = layer(mlp_input)
            mlp_input = F.relu(mlp_input)
            mlp_input = torch.nn.Dropout(p=feed_dict['dropout'])(mlp_input)

        mlp_out = (user_embeddings * mlp_input).sum(dim=1).reshape(items.shape)
        
        exposure_score = torch.softmax(self.expo_prob[uid,iid].reshape(items.shape), dim=1)
        
        prediction = (exposure_score * mlp_out).sum(dim=1).mean(dim=1).view([-1])
        
        
        # prediction = F.relu(self.cos(u_input, i_input)).view([-1]) * 10
        check_list.append(('prediction', prediction))
        out_dict = {'prediction': prediction,
                    'check': check_list}
        return out_dict
    
    def forward(self, feed_dict):
        """
        除了预测之外，还计算loss
        :param feed_dict: 型输入，是个dict
        :return: 输出，是个dict，prediction是预测值，check是需要检查的中间结果，loss是损失
        """
        out_dict = self.predict(feed_dict)
        if feed_dict['rank'] == 1:
            # 计算topn推荐的loss，batch前一半是正例，后一半是负例
            batch_size = int(feed_dict['Y'].shape[0] / 2)
            pos, neg = out_dict['prediction'][:batch_size], out_dict['prediction'][batch_size:]
            loss = -(pos - neg).sigmoid().log().sum()
#             pos_exp, neg_exp = out_dict['exposure'][:batch_size], out_dict['exposure'][batch_size:]
#             loss += self.exposure_weight * (-(pos_exp - neg_exp).sigmoid().log().sum())
        else:
            # 计算rating/clicking预测的loss，默认使用mse
            loss = torch.nn.MSELoss()(out_dict['prediction'], feed_dict['Y'])
        out_dict['loss'] = loss
        return out_dict
