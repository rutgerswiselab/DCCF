# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.RecModel import RecModel
import numpy as np
from utils import utils


class DMF(RecModel):
    @staticmethod
    def parse_model_args(parser, model_name='DMF'):
        parser.add_argument('--n_layers', type=int, default=1,
                            help="Number of mlp layers.")
        return RecModel.parse_model_args(parser, model_name)

    def __init__(self, label_min, label_max, feature_num, user_num, item_num, u_vector_size, i_vector_size,
                 n_layers, random_seed, model_path):
        self.n_layers = n_layers
        RecModel.__init__(self, label_min=label_min, label_max=label_max, feature_num=feature_num,
                          user_num=user_num, item_num=item_num, u_vector_size=u_vector_size,
                          i_vector_size=i_vector_size, random_seed=random_seed, model_path=model_path)

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

        self.cos = nn.CosineSimilarity()

        self.u_mlp = nn.ModuleList([nn.Linear(self.ui_vector_size, self.ui_vector_size)])
        # self.u_mlp = nn.ModuleList([nn.Linear(self.item_num, self.ui_vector_size)])
        for layer in range(self.n_layers - 1):
            self.u_mlp.append(nn.Linear(self.ui_vector_size, self.ui_vector_size))
        self.i_mlp = nn.ModuleList([nn.Linear(self.ui_vector_size, self.ui_vector_size)])
        # self.i_mlp = nn.ModuleList([nn.Linear(self.user_num, self.ui_vector_size)])
        for layer in range(self.n_layers - 1):
            self.i_mlp.append(nn.Linear(self.ui_vector_size, self.ui_vector_size))

    def predict(self, feed_dict):
        check_list = []
        u_ids = feed_dict['X'][:, 0]
        i_ids = feed_dict['X'][:, 1]

        user_embeddings = self.uid_embeddings(u_ids)
        item_embeddings = self.iid_embeddings(i_ids)
        u_input = user_embeddings

        for layer in self.u_mlp:
            u_input = layer(u_input)
            u_input = F.relu(u_input)
            u_input = torch.nn.Dropout(p=feed_dict['dropout'])(u_input)

        i_input = item_embeddings
        for layer in self.i_mlp:
            i_input = layer(i_input)
            i_input = F.relu(i_input)
            i_input = torch.nn.Dropout(p=feed_dict['dropout'])(i_input)

        # prediction = F.relu(self.cos(u_input, i_input)).view([-1]) * 10
        prediction = self.cos(u_input, i_input).view([-1]) * 10
        check_list.append(('prediction', prediction))
        out_dict = {'prediction': prediction,
                    'check': check_list}
        return out_dict
