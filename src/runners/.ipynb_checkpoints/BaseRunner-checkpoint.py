# coding=utf-8

import torch
import logging
from time import time
from utils import utils, global_p
from tqdm import tqdm
import numpy as np
import pandas as pd
import copy
import os

import pdb


class BaseRunner(object):
    @staticmethod
    def parse_runner_args(parser):
        """
        跑模型的命令行参数
        :param parser:
        :return:
        """
        parser.add_argument('--load', type=int, default=0,
                            help='Whether load model and continue to train')
        parser.add_argument('--epoch', type=int, default=100,
                            help='Number of epochs.')
        parser.add_argument('--check_epoch', type=int, default=1,
                            help='Check every epochs.')
        parser.add_argument('--early_stop', type=int, default=1,
                            help='whether to early-stop.')
        parser.add_argument('--lr', type=float, default=0.01,
                            help='Learning rate.')
        parser.add_argument('--batch_size', type=int, default=128,
                            help='Batch size during training.')
        parser.add_argument('--eval_batch_size', type=int, default=128 * 128,
                            help='Batch size during testing.')
        parser.add_argument('--dropout', type=float, default=0.2,
                            help='Dropout probability for each deep layer')
        parser.add_argument('--l2', type=float, default=1e-4,
                            help='Weight of l2_regularize in loss.')
        parser.add_argument('--optimizer', type=str, default='GD',
                            help='optimizer: GD, Adam, Adagrad')
        parser.add_argument('--metric', type=str, default="RMSE",
                            help='metrics: RMSE, MAE, AUC, F1, Accuracy, Precision, Recall')
        parser.add_argument('--skip_eval', type=int, default=0,
                            help='number of epochs without evaluation')
        return parser

    def __init__(self, optimizer='GD', learning_rate=0.01, epoch=100, batch_size=128, eval_batch_size=128 * 128,
                 dropout=0.2, l2=1e-5, metrics='RMSE', check_epoch=10, early_stop=1):
        """
        初始化
        :param optimizer: 优化器名字
        :param learning_rate: 学习率
        :param epoch: 总共跑几轮
        :param batch_size: 训练batch大小
        :param eval_batch_size: 测试batch大小
        :param dropout: dropout比例
        :param l2: l2权重
        :param metrics: 评价指标，逗号分隔
        :param check_epoch: 每几轮输出check一次模型中间的一些tensor
        :param early_stop: 是否自动提前终止训练
        """
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.dropout = dropout
        self.no_dropout = 0.0
        self.l2_weight = l2

        # 把metrics转换为list of str
        self.metrics = metrics.lower().split(',')
        self.check_epoch = check_epoch
        self.early_stop = early_stop
        self.time = None

        # 用来记录训练集、验证集、测试集每一轮的评价指标
        self.train_results, self.valid_results, self.test_results = [], [], []

    def _build_optimizer(self, model):
        """
        创建优化器
        :param model: 模型
        :return: 优化器
        """
        optimizer_name = self.optimizer_name.lower()
        if optimizer_name == 'gd':
            logging.info("Optimizer: GD")
            optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, weight_decay=self.l2_weight)
            # optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        elif optimizer_name == 'adagrad':
            logging.info("Optimizer: Adagrad")
            optimizer = torch.optim.Adagrad(model.parameters(), lr=self.learning_rate, weight_decay=self.l2_weight)
            # optimizer = torch.optim.Adagrad(model.parameters(), lr=self.learning_rate)
        elif optimizer_name == 'adam':
            logging.info("Optimizer: Adam")
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.l2_weight)
            # optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        else:
            logging.error("Unknown Optimizer: " + self.optimizer_name)
            assert self.optimizer_name in ['GD', 'Adagrad', 'Adam']
            optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, weight_decay=self.l2_weight)
            # optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        return optimizer

    def _check_time(self, start=False):
        """
        记录时间用，self.time保存了[起始时间，上一步时间]
        :param start: 是否开始计时
        :return: 上一步到当前位置的时间
        """
        if self.time is None or start:
            self.time = [time()] * 2
            return self.time[0]
        tmp_time = self.time[1]
        self.time[1] = time()
        return self.time[1] - tmp_time

    def batches_add_control(self, batches, train):
        """
        向所有batch添加一些控制信息比如'dropout'
        :param batches: 所有batch的list，由DataProcessor产生
        :param train: 是否是训练阶段
        :return: 所有batch的list
        """
        for batch in batches:
            batch['train'] = train
            batch['dropout'] = self.dropout if train else self.no_dropout
        return batches

    def predict(self, model, data, data_processor):
        """
        预测，不训练
        :param model: 模型
        :param data: 数据dict，由DataProcessor的self.get_*_data()和self.format_data_dict()系列函数产生
        :param data_processor: DataProcessor实例
        :return: prediction 拼接好的 np.array
        """
        batches = data_processor.prepare_batches(data, self.eval_batch_size, train=False)
        batches = self.batches_add_control(batches, train=False)
        
#         pdb.set_trace()
        model.eval()
        predictions = []
        for batch in tqdm(batches, leave=False, ncols=100, mininterval=1, desc='Predict'):
            # gc.collect()
            prediction = model.predict(batch)['prediction']
            predictions.append(prediction.detach().cpu())
        predictions = np.concatenate(predictions)
        sample_ids = np.concatenate([b[global_p.K_SAMPLE_ID] for b in batches])

        reorder_dict = dict(zip(sample_ids, predictions))
        predictions = np.array([reorder_dict[i] for i in data[global_p.K_SAMPLE_ID]])
        return predictions

    def fit(self, model, data, data_processor, epoch=-1):  # fit the results for an input set
        """
        训练
        :param model: 模型
        :param data: 数据dict，由DataProcessor的self.get_*_data()和self.format_data_dict()系列函数产生
        :param data_processor: DataProcessor实例
        :param epoch: 第几轮
        :return: 返回最后一轮的输出，可供self.check函数检查一些中间结果
        """
        if model.optimizer is None:
            model.optimizer = self._build_optimizer(model)
        batches = data_processor.prepare_batches(data, self.batch_size, train=True)
        batches = self.batches_add_control(batches, train=True)
        batch_size = self.batch_size if data_processor.rank == 0 else self.batch_size * 2
        model.train()
        accumulate_size = 0
        for batch in tqdm(batches, leave=False, desc='Epoch %5d' % (epoch + 1), ncols=100, mininterval=1):
            accumulate_size += len(batch['Y'])
            # gc.collect()
            model.optimizer.zero_grad()
#             pdb.set_trace()
            output_dict = model(batch)
            loss = output_dict['loss'] + model.l2() * self.l2_weight
#             pdb.set_trace()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
            torch.nn.utils.clip_grad_value_(model.parameters(), 50)
            if accumulate_size >= batch_size or batch is batches[-1]:
                model.optimizer.step()
                accumulate_size = 0
            # model.optimizer.step()
        model.eval()
        return output_dict

    def eva_termination(self, model):
        """
        检查是否终止训练，基于验证集
        :param model: 模型
        :return: 是否终止训练
        """
        metric = self.metrics[0]
        valid = self.valid_results
        # 如果已经训练超过20轮，且评价指标越小越好，且评价已经连续五轮非减
        if len(valid) > 20 and metric in utils.LOWER_METRIC_LIST and utils.strictly_increasing(valid[-5:]):
            return True
        # 如果已经训练超过20轮，且评价指标越大越好，且评价已经连续五轮非增
        elif len(valid) > 20 and metric not in utils.LOWER_METRIC_LIST and utils.strictly_decreasing(valid[-5:]):
            return True
        # 训练好结果离当前已经20轮以上了
        elif len(valid) - valid.index(utils.best_result(metric, valid)) > 20:
            return True
        return False

    def train(self, model, data_processor, skip_eval=0):
        """
        训练模型
        :param model: 模型
        :param data_processor: DataProcessor实例
        :param skip_eval: number of epochs to skip for evaluations
        :return:
        """

        # 获得训练、验证、测试数据，epoch=-1不shuffle
        train_data = data_processor.get_train_data(epoch=-1)
        validation_data = data_processor.get_validation_data()
        test_data = data_processor.get_test_data()
        self._check_time(start=True)  # 记录初始时间
#         pdb.set_trace()

        # 训练之前的模型效果
        init_train = self.evaluate(model, train_data, data_processor, metrics=['rmse', 'mae']) \
            if train_data is not None else [-1.0] * len(self.metrics)
        init_valid = self.evaluate(model, validation_data, data_processor) \
            if validation_data is not None else [-1.0] * len(self.metrics)
        init_test = self.evaluate(model, test_data, data_processor) \
            if test_data is not None else [-1.0] * len(self.metrics)
        logging.info("Init: \t train= %s validation= %s test= %s [%.1f s] " % (
            utils.format_metric(init_train), utils.format_metric(init_valid), utils.format_metric(init_test),
            self._check_time()) + ','.join(self.metrics))

        try:
            for epoch in range(self.epoch):
                self._check_time()
                # 每一轮需要重新获得训练数据，因为涉及shuffle或者topn推荐时需要重新采样负例
                epoch_train_data = data_processor.get_train_data(epoch=epoch)
#                 pdb.set_trace()
                last_batch = self.fit(model, epoch_train_data, data_processor, epoch=epoch)
                # 检查模型中间结果
                if self.check_epoch > 0 and (epoch == 1 or epoch % self.check_epoch == 0):
                    self.check(model, last_batch)
                training_time = self._check_time()

                if epoch >= skip_eval:
                    # 重新evaluate模型效果
                    train_result = self.evaluate(model, train_data, data_processor, metrics=['rmse', 'mae']) \
                        if train_data is not None else [-1.0] * len(self.metrics)
                    valid_result = self.evaluate(model, validation_data, data_processor, write_rank=False) \
                        if validation_data is not None else [-1.0] * len(self.metrics)
                    test_result = self.evaluate(model, test_data, data_processor, write_rank=False) \
                        if test_data is not None else [-1.0] * len(self.metrics)
                    testing_time = self._check_time()

                    self.train_results.append(train_result)
                    self.valid_results.append(valid_result)
                    self.test_results.append(test_result)

                    # 输出当前模型效果
                    logging.info("Epoch %5d [%.1f s]\t train= %s validation= %s test= %s [%.1f s] "
                                 % (epoch + 1, training_time, utils.format_metric(train_result),
                                    utils.format_metric(valid_result), utils.format_metric(test_result),
                                    testing_time) + ','.join(self.metrics))

                    # 如果当前效果是最优的，保存模型，基于验证集
                    if utils.best_result(self.metrics[0], self.valid_results) == self.valid_results[-1]:
                        model.save_model()
                    # 检查是否终止训练，基于验证集
                    if self.eva_termination(model) and self.early_stop == 1:
                        logging.info("Early stop at %d based on validation result." % (epoch + 1))
                        break
                if epoch < skip_eval:
                    logging.info("Epoch %5d [%.1f s]" % (epoch + 1, training_time))
        except KeyboardInterrupt:
            logging.info("Early stop manually")
            save_here = input("Save here? (1/0) (default 0):")
            if str(save_here).lower().startswith('1'):
                model.save_model()

        # Find the best validation result across iterations
        best_valid_score = utils.best_result(self.metrics[0], self.valid_results)
        best_epoch = self.valid_results.index(best_valid_score)
        logging.info("Best Iter(validation)= %5d\t train= %s valid= %s test= %s [%.1f s] "
                     % (best_epoch + 1,
                        utils.format_metric(self.train_results[best_epoch]),
                        utils.format_metric(self.valid_results[best_epoch]),
                        utils.format_metric(self.test_results[best_epoch]),
                        self.time[1] - self.time[0]) + ','.join(self.metrics))
        best_test_score = utils.best_result(self.metrics[0], self.test_results)
        best_epoch = self.test_results.index(best_test_score)
        logging.info("Best Iter(test)= %5d\t train= %s valid= %s test= %s [%.1f s] "
                     % (best_epoch + 1,
                        utils.format_metric(self.train_results[best_epoch]),
                        utils.format_metric(self.valid_results[best_epoch]),
                        utils.format_metric(self.test_results[best_epoch]),
                        self.time[1] - self.time[0]) + ','.join(self.metrics))
        model.load_model()

    def evaluate(self, model, data, data_processor, metrics=None, write_rank=False):  # evaluate the results for an input set
        """
        evaluate模型效果
        :param model: 模型
        :param data: 数据dict，由DataProcessor的self.get_*_data()和self.format_data_dict()系列函数产生
        :param data_processor: DataProcessor
        :param metrics: list of str
        :param write_rank: if write rank list into a file
        :return: list of float 每个对应一个 metric
        """
        def _write_output(p):
            df = pd.DataFrame()
            df['uid'] = data['uid']
            df['iid'] = data['iid']
            df['score'] = p
            df['label'] = data['Y']
            df = df.sort_values(by='uid')
            path = os.path.join(data_processor.data_loader.path, global_p.RANK_FILE_NAME)
            df.to_csv(path, sep='\t', index=False)

        if metrics is None:
            metrics = self.metrics
        predictions = self.predict(model, data, data_processor)
        if write_rank:
            _write_output(predictions)
#         pdb.set_trace()

        return model.evaluate_method(predictions, data, metrics=metrics)

    def check(self, model, out_dict):
        """
        检查模型中间结果
        :param model: 模型
        :param out_dict: 某一个batch的模型输出结果
        :return:
        """
        # batch = data_processor.get_feed_dict(data, 0, self.batch_size, True)
        # self.batches_add_control([batch], train=False)
        # model.eval()
        # check = model(batch)
        check = out_dict
        logging.info(os.linesep)
        for i, t in enumerate(check['check']):
            d = np.array(t[1].cpu().detach())
            logging.info(os.linesep.join([t[0] + '\t' + str(d.shape), np.array2string(d, threshold=20)]) + os.linesep)

        loss, l2 = check['loss'], model.l2()
        l2 = l2 * self.l2_weight
        logging.info('loss = %.4f, l2 = %.4f' % (loss, l2))
        if not (loss.abs() * 0.005 < l2 < loss.abs() * 0.1):
            logging.warning('l2 inappropriate: loss = %.4f, l2 = %.4f' % (loss, l2))
