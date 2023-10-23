# coding=utf-8

import torch
import torch.nn.functional as F
from models.RecModel import RecModel
from utils import utils
import numpy as np
import os
from utils import global_p


class IPSBiasedMF(RecModel):
    def parse_model_args(parser, model_name='IPSBiasedMF'):
        parser.add_argument('--M', type=float, default=0.1, help='minimum propensity to avoid high variance.')
        return RecModel.parse_model_args(parser, model_name)
    
    def __init__(self, path, dataset, M, label_min, label_max, feature_num, user_num, item_num, u_vector_size, i_vector_size,
                 random_seed, model_path):
        self.path = path
        self.dataset = dataset
        RecModel.__init__(self, label_min=label_min, label_max=label_max,
                           feature_num=feature_num, user_num=user_num, item_num=item_num, 
                           u_vector_size=u_vector_size, i_vector_size=i_vector_size,
                           random_seed=random_seed,
                           model_path=model_path)
        self.M = M
        self.propensity = torch.FloatTensor(np.load(os.path.join(path, dataset + global_p.PROPENSITY_SUFFIX))).to('cuda:' + str(torch.cuda.current_device()))
    
    def _init_weights(self):
        self.uid_embeddings = torch.nn.Embedding(self.user_num, self.ui_vector_size)
        self.iid_embeddings = torch.nn.Embedding(self.item_num, self.ui_vector_size)
        self.user_bias = torch.nn.Embedding(self.user_num, 1)
        self.item_bias = torch.nn.Embedding(self.item_num, 1)
        self.global_bias = torch.nn.Parameter(torch.tensor(0.1))
        

    def predict(self, feed_dict):
        check_list = []
        u_ids = feed_dict['X'][:, 0]
        i_ids = feed_dict['X'][:, 1]

        # bias
        u_bias = self.user_bias(u_ids).view([-1])
        i_bias = self.item_bias(i_ids).view([-1])

        cf_u_vectors = self.uid_embeddings(u_ids)
        cf_i_vectors = self.iid_embeddings(i_ids)
        prediction = (cf_u_vectors * cf_i_vectors).sum(dim=1).view([-1])
        prediction = prediction + u_bias + i_bias + self.global_bias
        # prediction = prediction + self.global_bias
        propensity = self.propensity[i_ids]
        propensity = torch.max(propensity, torch.tensor(self.M).to('cuda:' + str(torch.cuda.current_device())))
        prediction /= propensity

        out_dict = {'prediction': prediction,
                    'check': check_list}
        return out_dict
