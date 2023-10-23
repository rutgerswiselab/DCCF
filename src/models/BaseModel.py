# coding=utf-8

import torch
import logging
from sklearn.metrics import *
import numpy as np
import torch.nn.functional as F
import os
import pandas as pd
from utils.rank_metrics import *

import pdb


class BaseModel(torch.nn.Module):
    """
    基类模型，一般新模型需要重载的函数有
    parse_model_args,
    __init__,
    _init_weights,
    predict,
    forward,
    """

    '''
    DataProcessor的format_data_dict()会用到这四个变量

    通常会把特征全部转换为multi-hot向量
    例:uid(0-2),iid(0-2),u_age(0-2),i_xx(0-1)，
    那么uid=0,iid=1,u_age=1,i_xx=0会转换为100 010 010 10的稀疏表示0,4,7,9
    如果include_id=False，那么multi-hot不会包括uid,iid，即u_age=1,i_xx=0转化为010 10的稀疏表示 1,3
    include_user_features 和 include_item_features同理
    append id 是指是否将 uid,iid append在输入'X'的最前，比如在append_id=True, include_id=False的情况下：
    uid=0,iid=1,u_age=1,i_xx=0会转换为 0,1,1,3
    '''
    append_id = False
    include_id = True
    include_user_features = True
    include_item_features = True
    include_context_features = False

    @staticmethod
    def parse_model_args(parser, model_name='BaseModel'):
        """
        模型命令行参数
        :param parser:
        :param model_name: 模型名称
        :return:
        """
        parser.add_argument('--model_path', type=str,
                            default='../model/%s/%s.pt' % (model_name, model_name),
                            help='Model save path.')
        return parser

    @staticmethod
    def evaluate_method(p, data, metrics):
        """
        计算模型评价指标
        :param p: 预测值，np.array，一般由runner.predict产生
        :param data: data dict，一般由DataProcessor产生
        :param metrics: 评价指标的list，一般是runner.metrics，例如 ['rmse', 'auc']
        :return:
        """
        l = data['Y']
        evaluations = []
#         pdb.set_trace()
        for metric in metrics:
            if metric == 'rmse':
                evaluations.append(np.sqrt(mean_squared_error(l, p)))
            elif metric == 'mae':
                evaluations.append(mean_absolute_error(l, p))
            elif metric == 'auc':
                evaluations.append(roc_auc_score(l, p))
            elif metric == 'f1':
                evaluations.append(f1_score(l, p))
            elif metric == 'accuracy':
                evaluations.append(accuracy_score(l, p))
            elif metric == 'precision':
                evaluations.append(precision_score(l, p))
            elif metric == 'recall':
                evaluations.append(recall_score(l, p))
            else:
                k = int(metric.split('@')[-1])
                df = pd.DataFrame()
                df['uid'] = data['uid']
                df['p'] = p
                df['l'] = l
#                 pdb.set_trace()
                df = df.sort_values(by='p', ascending=False)
                df_group = df.groupby('uid')
                if metric.startswith('ndcg@'):
                    ndcgs = []
                    for uid, group in df_group:
#                         pdb.set_trace()
                        ndcgs.append(ndcg_at_k(group['l'].tolist(), k=k, method=1))
#                     pdb.set_trace()
                    evaluations.append(np.average(ndcgs))
                elif metric.startswith('hit@'):
                    hits = []
                    for uid, group in df_group:
                        hits.append(int(np.sum(group['l'][:k]) > 0))
                    evaluations.append(np.average(hits))
                elif metric.startswith('precision@'):
                    precisions = []
                    for uid, group in df_group:
                        precisions.append(precision_at_k(group['l'].tolist()[:k], k=k))
                    evaluations.append(np.average(precisions))
                elif metric.startswith('recall@'):
                    recalls = []
                    for uid, group in df_group:
                        recalls.append(1.0 * np.sum(group['l'][:k]) / np.sum(group['l']))
                    evaluations.append(np.average(recalls))
                # elif metric.startswith('f1@'):
                #     f1 = []
                #     for uid, group in df_group:
                #         precision = precision_at_k(group['l'].tolist()[:k], k=k)
                #         recall = 1.0 * np.sum(group['l'][:k]) / np.sum(group['l'])
                #         if precision > 0 or recall > 0:
                #             f1.append(2 * precision * recall / (precision + recall))
                #     evaluations.append(np.average(f1))
                elif metric.startswith('f1@'):
                    f1 = []
                    for uid, group in df_group:
                        num_overlap = 1.0 * np.sum(group['l'][:k])
                        f1.append(2 * num_overlap / (k + 1.0 * np.sum(group['l'])))
                    evaluations.append(np.average(f1))

        return evaluations

    @staticmethod
    def init_paras(m):
        """
        模型自定义初始化函数，在main.py中会被调用
        :param m: 参数或含参数的层
        :return:
        """
        if type(m) == torch.nn.Linear:
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                torch.nn.init.normal_(m.bias, mean=0.0, std=0.01)
        elif type(m) == torch.nn.Embedding:
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def __init__(self, label_min, label_max, feature_num, random_seed=2018, model_path='../model/Model/Model.pt'):
        super(BaseModel, self).__init__()
        self.label_min = label_min
        self.label_max = label_max
        self.feature_num = feature_num
        self.random_seed = random_seed
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        self.model_path = model_path

        self._init_weights()
        logging.debug(list(self.parameters()))

        self.total_parameters = self.count_variables()
        logging.info('# of params: %d' % self.total_parameters)

        # optimizer 由runner生成并赋值
        self.optimizer = None

    def _init_weights(self):
        """
        初始化需要的权重（带权重层）
        :return:
        """
        self.x_bn = torch.nn.BatchNorm1d(self.feature_num)
        self.prediction = torch.nn.Linear(self.feature_num, 1)

    def count_variables(self):
        """
        模型所有参数数目
        :return:
        """
        total_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_parameters

    def l2(self):
        """
        模型l2计算，默认是所有参数的平方和
        :return:
        """
        l2 = 0
        for p in self.parameters():
            l2 += (p ** 2).sum()
        return l2

    def predict(self, feed_dict):
        """
        只预测，不计算loss
        :param feed_dict: 模型输入，是个dict
        :return: 输出，是个dict，prediction是预测值，check是需要检查的中间结果
        """
        check_list = []
        x = self.x_bn(feed_dict['X'].float())
        x = torch.nn.Dropout(p=feed_dict['dropout'])(x)
        prediction = F.relu(self.prediction(x)).view([-1])
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
        else:
            # 计算rating/clicking预测的loss，默认使用mse
            loss = torch.nn.MSELoss()(out_dict['prediction'], feed_dict['Y'])
        out_dict['loss'] = loss
        return out_dict

    def lrp(self):
        pass

    def save_model(self, model_path=None):
        """
        保存模型，一般使用默认路径
        :param model_path: 指定模型保存路径
        :return:
        """
        if model_path is None:
            model_path = self.model_path
        dir_path = os.path.dirname(model_path)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        torch.save(self.state_dict(), model_path)
        logging.info('Save model to ' + model_path)

    def load_model(self, model_path=None):
        """
        载入模型，一般使用默认路径
        :param model_path: 指定模型载入路径
        :return:
        """
        if model_path is None:
            model_path = self.model_path
        self.load_state_dict(torch.load(model_path))
        self.eval()
        logging.info('Load model from ' + model_path)
