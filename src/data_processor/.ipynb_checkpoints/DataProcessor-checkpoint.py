# coding=utf-8
import copy
from utils import utils
import numpy as np
import logging
import pandas as pd
from tqdm import tqdm
import torch
from collections import defaultdict
from utils import global_p

import pdb


class DataProcessor(object):
    data_columns = ['X']  # data dict中存储模型所需特征信息的key（负例data需要append在最后）

    @staticmethod
    def parse_dp_args(parser):
        """
        数据处理生成batch的命令行参数
        :param parser:
        :return:
        """
        parser.add_argument('--test_neg_n', type=int, default=100,
                            help='Negative sample num for each instance in test/validation set.')
        return parser

    def __init__(self, data_loader, model, rank, test_neg_n):
        """
        初始化
        :param data_loader: DataLoader对象
        :param model: Model对象
        :param rank: 1=topn推荐 0=评分或点击预测
        :param test_neg_n: topn推荐时的测试集负例采样比例 正:负=1:test_neg_n
        """
        self.data_loader = data_loader
        self.model = model
        self.rank = rank
        self.train_data, self.validation_data, self.test_data = None, None, None

        self.test_neg_n = test_neg_n

        if self.rank == 1:
            # 生成用户交互的字典，方便采样负例时查询，不要采到正例
            self.train_history_dict = defaultdict(set)
            for uid in data_loader.train_user_his.keys():
                self.train_history_dict[uid] = set(data_loader.train_user_his[uid])
            # print(self.train_history_dict[405])
            self.vt_history_dict = defaultdict(set)
            for uid in data_loader.vt_user_his.keys():
                self.vt_history_dict[uid] = set(data_loader.vt_user_his[uid])
            # print(self.vt_history_dict[405])

        self.vt_batches_buffer = {}

    def get_train_data(self, epoch):
        """
        将dataloader中的训练集Dataframe转换为所需要的字典后返回，每一轮都要shuffle
        该字典会被用来生成batches
        :param epoch: <0则不shuffle
        :return: 字典dict
        """
        if self.train_data is None or epoch < 0:
            logging.info('Prepare Train Data...')
            self.train_data = self.format_data_dict(self.data_loader.train_df)
            self.train_data[global_p.K_SAMPLE_ID] = np.arange(0, len(self.train_data['Y']))
        if epoch >= 0:
            utils.shuffle_in_unison_scary(self.train_data)
#         pdb.set_trace()
        return self.train_data

    def get_validation_data(self):
        """
        将dataloader中的验证集Dataframe转换为所需要的字典后返回
        如果是topn推荐则每个正例对应采样test_neg_n个负例
        该字典会被用来生成batches
        :return: 字典dict
        """
        if self.validation_data is None:
            logging.info('Prepare Validation Data...')
            df = self.data_loader.validation_df
            if self.rank == 1:
                neg_df = self.generate_neg_df(
                    uid_list=df['uid'].tolist(), iid_list=df['iid'].tolist(),
                    df=df, neg_n=self.test_neg_n, train=False)
                df = pd.concat([df, neg_df], ignore_index=True)
            self.validation_data = self.format_data_dict(df)
            self.validation_data[global_p.K_SAMPLE_ID] = np.arange(0, len(self.validation_data['Y']))
        return self.validation_data

    def get_test_data(self):
        """
        将dataloader中的测试集Dataframe转换为所需要的字典后返回
        如果是topn推荐则每个正例对应采样test_neg_n个负例
        该字典会被用来生成batches
        :return: 字典dict
        """
        if self.test_data is None:
            logging.info('Prepare Test Data...')
            df = self.data_loader.test_df
            if self.rank == 1:
                neg_df = self.generate_neg_df(
                    uid_list=df['uid'].tolist(), iid_list=df['iid'].tolist(),
                    df=df, neg_n=self.test_neg_n, train=False)
                df = pd.concat([df, neg_df], ignore_index=True)
            self.test_data = self.format_data_dict(df)
            self.test_data[global_p.K_SAMPLE_ID] = np.arange(0, len(self.test_data['Y']))
#         pdb.set_trace()

        return self.test_data

    def get_train_batches(self, batch_size, epoch):
        """
        生成训练batch
        :param batch_size: batch大小
        :param epoch: 第几轮
        :return: dict的list，每个dict是一个batch
        """
        return self.prepare_batches(self.get_train_data(epoch), batch_size, train=True)

    def get_validation_batches(self, batch_size):
        """
        生成验证batch
        :param batch_size: batch大小
        :return: dict的list，每个dict是一个batch
        """
        return self.prepare_batches(self.get_validation_data(), batch_size, train=False)

    def get_test_batches(self, batch_size):
        """
        生成测试batch
        :param batch_size: batch大小
        :return: dict的list，每个dict是一个batch
        """
        return self.prepare_batches(self.get_test_data(), batch_size, train=False)

    def _get_feed_dict_rt(self, data, batch_start, batch_size, train):
        """
        rating/clicking预测产生一个batch
        :param data: data dict，由self.get_*_data()和self.format_data_dict()系列函数产生
        :param batch_start: batch开始的index
        :param batch_size: batch大小
        :param train: 训练还是测试
        :return: batch的feed dict
        """
        batch_end = min(len(data['X']), batch_start + batch_size)
        real_batch_size = batch_end - batch_start
        feed_dict = {'train': train, 'rank': 0,
                     global_p.K_SAMPLE_ID: data[global_p.K_SAMPLE_ID][batch_start:batch_start + real_batch_size]}
        if 'Y' in data:
            feed_dict['Y'] = utils.numpy_to_torch(data['Y'][batch_start:batch_start + real_batch_size])
        else:
            feed_dict['Y'] = utils.numpy_to_torch(np.zeros(shape=real_batch_size))
        for c in self.data_columns:
            feed_dict[c] = utils.numpy_to_torch(
                data[c][batch_start: batch_start + real_batch_size])
        return feed_dict

    def _get_feed_dict_rk(self, data, batch_start, batch_size, train, neg_data=None):
        """
        topn模型产生一个batch，如果是训练需要对每个正样本采样一个负样本，保证每个batch前一半是正样本，后一半是对应的负样本
        :param data: data dict，由self.get_*_data()和self.format_data_dict()系列函数产生
        :param batch_start: batch开始的index
        :param batch_size: batch大小
        :param train: 训练还是测试
        :param neg_data: 负例的data dict，如果已经有可以传入拿来用
        :return: batch的feed dict
        """
        # 如果是测试货验证，不需要对每个正样本采负样本，负样本已经1:test_neg_n采样好
        if not train:
            feed_dict = self._get_feed_dict_rt(
                data=data, batch_start=batch_start, batch_size=batch_size, train=train)
            feed_dict['rank'] = 1
        else:
            batch_end = min(len(data['X']), batch_start + batch_size)
            real_batch_size = batch_end - batch_start
            neg_columns_dict = {}
            if neg_data is None:
                # 如果还没有负例，则根据uid采样对应的负例
                logging.warning('neg_data is None')
                neg_df = self.generate_neg_df(
                    uid_list=data['uid'][batch_start: batch_start + real_batch_size],
                    iid_list=data['iid'][batch_start: batch_start + real_batch_size],
                    df=self.data_loader.train_df, neg_n=1, train=True)
                neg_data = self.format_data_dict(neg_df)
                for c in self.data_columns:
                    neg_columns_dict[c] = neg_data[c]
            else:
                for c in self.data_columns:
                    neg_columns_dict[c] = neg_data[c][batch_start: batch_start + real_batch_size]
            y = np.concatenate([np.ones(shape=real_batch_size, dtype=np.float32),
                                np.zeros(shape=real_batch_size, dtype=np.float32)])
            sample_id = data[global_p.K_SAMPLE_ID][batch_start:batch_start + real_batch_size]
            neg_sample_id = sample_id + len(self.train_data['Y'])
            total_batch_size = real_batch_size * (1 + 1) if self.rank == 1 and train else real_batch_size
            feed_dict = {
                'train': train, 'rank': 1,
                'Y': utils.numpy_to_torch(y),
                global_p.K_SAMPLE_ID: np.concatenate([sample_id, neg_sample_id]),
                global_p.REAL_BATCH_SIZE: real_batch_size,
                global_p.TOTAL_BATCH_SIZE: total_batch_size
                }
            for c in self.data_columns:
                feed_dict[c] = utils.numpy_to_torch(
                    np.concatenate([data[c][batch_start: batch_start + real_batch_size], neg_columns_dict[c]]))
        return feed_dict

    def _prepare_batches_rt(self, data, batch_size, train):
        """
        rating/clicking预测，将data dict全部转换为batch
        :param data: dict 由self.get_*_data()和self.format_data_dict()系列函数产生
        :param batch_size: batch大小
        :param train: 训练还是测试
        :return: list of batches
        """
        if data is None:
            return None
        num_example = len(data['X'])
        total_batch = int((num_example + batch_size - 1) / batch_size)
        assert num_example > 0
        batches = []
        for batch in tqdm(range(total_batch), leave=False, ncols=100, mininterval=1, desc='Prepare Batches'):
            batches.append(self._get_feed_dict_rt(data, batch * batch_size, batch_size, train))
        return batches

    def _prepare_batches_rk(self, data, batch_size, train):
        """
        topn模型，将data dict全部转换为batch
        :param data: dict 由self.get_*_data()和self.format_data_dict()系列函数产生
        :param batch_size: batch大小
        :param train: 训练还是测试
        :return: list of batches
        """
        if data is None:
            return None
        num_example = len(data['X'])
        total_batch = int((num_example + batch_size - 1) / batch_size)
        assert num_example > 0
        # 如果是训练，则需要对对应的所有正例采一个负例
        neg_data = None
        if train:
            neg_df = self.generate_neg_df(
                uid_list=data['uid'], iid_list=data['iid'],
                df=self.data_loader.train_df, neg_n=1, train=True)
            neg_data = self.format_data_dict(neg_df)
        batches = []
        for batch in tqdm(range(total_batch), leave=False, ncols=100, mininterval=1, desc='Prepare Batches'):
            batches.append(self._get_feed_dict_rk(data, batch * batch_size, batch_size, train, neg_data))
        return batches

    def prepare_batches(self, data, batch_size, train):
        """
        将data dict全部转换为batch
        :param data: dict 由self.get_*_data()和self.format_data_dict()系列函数产生
        :param batch_size: batch大小
        :param train: 训练还是测试
        :return: list of batches
        """
        buffer_key = ''
        if data is self.validation_data:
            buffer_key = 'validation_' + str(batch_size)
        elif data is self.test_data:
            buffer_key = 'test_' + str(batch_size)
        if buffer_key in self.vt_batches_buffer:
            return self.vt_batches_buffer[buffer_key]

        if self.rank == 1:
            batches = self._prepare_batches_rk(data=data, batch_size=batch_size, train=train)
        else:
            batches = self._prepare_batches_rt(data=data, batch_size=batch_size, train=train)

        if buffer_key != '':
            self.vt_batches_buffer[buffer_key] = batches
        return batches

    def get_feed_dict(self, data, batch_start, batch_size, train, neg_data=None):
        """
        :param data: data dict，由self.get_*_data()和self.format_data_dict()系列函数产生
        :param batch_start: batch开始的index
        :param batch_size: batch大小
        :param train: 训练还是测试
        :param neg_data: 负例的data dict，如果已经有可以传入拿来用
        :return: batch的feed dict
        :return:
        """
        if self.rank == 1:
            return self._get_feed_dict_rk(data=data, batch_start=batch_start, batch_size=batch_size, train=train,
                                          neg_data=neg_data)
        return self._get_feed_dict_rt(data=data, batch_start=batch_start, batch_size=batch_size, train=train)

    def format_data_dict(self, df):
        """
        将dataloader的训练、验证、测试Dataframe转换为需要的data dict
        :param df: pandas Dataframe, 在推荐问题中通常包含 'uid','iid','label' 三列
        :return: data dict
        """

        data_loader, model = self.data_loader, self.model
        data = {}
        # 记录uid, iid
        out_columns = []
        if 'uid' in df:
            out_columns.append('uid')
            data['uid'] = df['uid'].values
        if 'iid' in df:
            out_columns.append('iid')
            data['iid'] = df['iid'].values

        # label 记录在 'Y' 中
        if data_loader.label in df.columns:
            data['Y'] = np.array(df[data_loader.label], dtype=np.float32)
        else:
            logging.warning('No Labels In Data: ' + data_loader.label)
            data['Y'] = np.zeros(len(df), dtype=np.float32)

        ui_id = df[out_columns]

        # 根据uid和iid拼接上用户特征和物品特征
        out_df = ui_id
        if data_loader.user_df is not None and model.include_user_features:
            out_columns.extend(data_loader.user_features)
            out_df = pd.merge(out_df, data_loader.user_df, on='uid', how='left')
        if data_loader.item_df is not None and model.include_item_features:
            out_columns.extend(data_loader.item_features)
            out_df = pd.merge(out_df, data_loader.item_df, on='iid', how='left')
        out_df = out_df.fillna(0)

        # 是否包含context feature
        if model.include_context_features:
            context = df[data_loader.context_features]
            out_df = pd.concat([out_df, context], axis=1, ignore_index=True)

        # 如果模型不把uid和iid当成普通特征一同看待，即不和其他特征一起转换为multi-hot向量
        if not model.include_id:
            out_df = out_df.drop(columns=['uid', 'iid'])

        '''
        把特征全部转换为multi-hot向量
        例:uid(0-2),iid(0-2),u_age(0-2),i_xx(0-1)，
        那么uid=0,iid=1,u_age=1,i_xx=0会转换为100 010 010 10的稀疏表示0,4,7,9
        '''
        base = 0
        for feature in out_df.columns:
            out_df[feature] = out_df[feature].apply(lambda x: x + base)
            base += int(data_loader.column_max[feature] + 1)

        # 如果模型需要，uid,iid拼接在x的前两列
        if model.append_id:
            x = pd.concat([ui_id, out_df], axis=1, ignore_index=True)
            data['X'] = x.values.astype(int)
        else:
            data['X'] = out_df.values.astype(int)
        # print(data['X'].shape)
        assert len(data['X']) == len(data['Y'])
        return data

    # def sample_neg_from_df(self, df, neg_n, train):
    #     uid_list, iid_list = [], []
    #     other_columns = [c for c in df.columns if c not in ['uid', 'iid', self.data_loader.label]]
    #     other_columns_list = {}
    #     for c in other_columns:
    #         other_columns_list[c] = []
    #     contain_history = False if 'his' not in other_columns else True
    #
    #     tmp_history_dict = defaultdict(set)
    #     item_num = self.data_loader.item_num
    #     for index, row in df.iterrows():
    #         uid = row['uid']
    #         if train:
    #             inter_iids = self.train_history_dict[uid] | tmp_history_dict[uid]
    #         else:
    #             inter_iids = self.train_history_dict[uid] | self.vt_history_dict[uid] | tmp_history_dict[uid]
    #         remain_iids_num = item_num - len(inter_iids)
    #         remain_iids = None
    #         if 1.0 * remain_iids_num / item_num < 0.2:
    #             # logging.warning('Sampling Negative: uid=%d, %d from %d, items=%d' %
    #             #                 (uid, neg_n, remain_iids_num, item_num))
    #             remain_iids = [i for i in range(1, item_num) if i not in inter_iids]
    #         assert remain_iids_num >= neg_n
    #         if remain_iids is None:
    #             for i in range(neg_n):
    #                 iid = np.random.randint(1, self.data_loader.item_num)
    #                 while iid in inter_iids:
    #                     iid = np.random.randint(1, self.data_loader.item_num)
    #                 uid_list.append(uid)
    #                 iid_list.append(iid)
    #                 tmp_history_dict[uid].add(iid)
    #         else:
    #             iids = np.random.choice(remain_iids, neg_n, replace=False)
    #             uid_list.extend([uid] * neg_n)
    #             iid_list.extend(iids)
    #             tmp_history_dict[uid].update(iids)
    #         for c in other_columns:
    #             other_columns_list[c].extend([row[c]] * neg_n)
    #         if contain_history:
    #             tmp_history_dict[uid] = set()
    #
    #     neg_df = pd.DataFrame(data=list(zip(uid_list, iid_list)), columns=['uid', 'iid'])
    #     neg_df[self.data_loader.label] = 0
    #     for c in other_columns:
    #         neg_df[c] = other_columns_list[c]
    #     # print(neg_df)
    #     # origin_counter, sample_counter = Counter(uids), Counter(uid_list)
    #     # print(origin_counter.most_common(5), sample_counter.most_common(5))
    #     return neg_df

    def generate_neg_df(self, uid_list, iid_list, df, neg_n, train):
        """
        根据uid,iid和训练or验证测试的dataframe产生负样本
        :param uid_list: 对这些user采样
        :param iid_list: 对应的正例item
        :param df: df中可能包含了一些需要复制的信息比如历史记录
        :param neg_n: 负采样数目
        :param train: 训练集or验证集测试集负采样
        :return:
        """
        # for non-loo dataset use, only sample neg_n for one user no matter how many records in the valid/test set
        # filter out redundant uid and iids
        if not train:
            filtered_u_list = []
            filtered_i_list = []
            for i in range(len(uid_list)):
                if uid_list[i] not in filtered_u_list:
                    filtered_u_list.append(uid_list[i])
                    filtered_i_list.append(iid_list[i])
#             pdb.set_trace()
        else:
            filtered_u_list = uid_list
            filtered_i_list = iid_list
        # filtered_u_list = uid_list
        # filtered_i_list = iid_list

        neg_df = self._sample_neg_from_uid_list(
            uids=filtered_u_list, neg_n=neg_n, train=train, other_infos={'iid': filtered_i_list})
        
        neg_df = pd.merge(neg_df, df, on=['uid', 'iid'], how='left')
        neg_df.drop_duplicates(subset=['uid','iid_neg', 'iid'],inplace=True)
        neg_df = neg_df.drop(columns=['iid'])
        neg_df = neg_df.rename(columns={'iid_neg': 'iid'})
        neg_df = neg_df[df.columns]
        neg_df[self.data_loader.label] = 0

        return neg_df

    def _sample_neg_from_uid_list(self, uids, neg_n, train, other_infos=None):
        """
        根据uid的list采样对应的负样本
        :param uids: uid list
        :param neg_n: 每个uid采样几个负例
        :param train: 为训练集采样还是测试集采样
        :param other_infos: 除了uid,iid,label之外可能需要复制的信息，比如交互历史（前n个item），
            在generate_neg_df被用来复制原始正例iid
        :return: 返回DataFrame，还需经过self.format_data_dict()转换为data dict
        """
        if other_infos is None:
            other_infos = {}
        uid_list, iid_list = [], []

        other_info_list = {}
        for info in other_infos:
            other_info_list[info] = []

        # 记录采样过程中采到的iid，避免重复采样
        tmp_history_dict = defaultdict(set)
        item_num = self.data_loader.item_num

        # if not train:
        #     tmp_uids = []
        #     for uid in uids:
        #         if uid not in tmp_uids:
        #             tmp_uids.append(uid)
        #     uids = tmp_uids

        for index, uid in enumerate(uids):
            low_neg_items = False
            if train:
                # 训练集采样避免采训练集中正例和采样过的负例
                inter_iids = self.train_history_dict[uid] | tmp_history_dict[uid]
            else:
                # 测试集采样避免所有的正例和采样过的负例
                inter_iids = self.train_history_dict[uid] | self.vt_history_dict[uid] | tmp_history_dict[uid]
                # inter_iids = tmp_history_dict[uid]

            # 检查所剩可以采样的负例
            remain_iids_num = item_num - len(inter_iids)
            remain_iids = None

            # 如果数目不多则采用np.choice
            if 1.0 * remain_iids_num / item_num < 0.2:
                # logging.warning('Sampling Negative: uid=%d, %d from %d, items=%d' %
                #                 (uid, neg_n, remain_iids_num, item_num))
                remain_iids = [i for i in range(1, item_num) if i not in inter_iids]
            # 所有可采负例数目不够则报错
            assert remain_iids_num >= neg_n

            if remain_iids is None:
                for i in range(neg_n):
                    iid = np.random.randint(self.data_loader.item_num)
                    while iid in inter_iids or iid in tmp_history_dict[uid]:
                        iid = np.random.randint(self.data_loader.item_num)
                    uid_list.append(uid)
                    iid_list.append(iid)
                    tmp_history_dict[uid].add(iid)
            else:
                try:
                    iids = np.random.choice(remain_iids, neg_n, replace=False)
                except:
                    pdb.set_trace()
                uid_list.extend([uid] * neg_n)
                iid_list.extend(iids)
                tmp_history_dict[uid].update(iids)
            # 复制其他信息
            for info in other_infos:
                other_info_list[info].extend([other_infos[info][index]] * neg_n)
            if not train:
                tmp_history_dict = defaultdict(set)
        neg_df = pd.DataFrame(data=list(zip(uid_list, iid_list)), columns=['uid', 'iid_neg'])
        for info in other_infos:
            neg_df[info] = other_info_list[info]
        # print(neg_df)
        # origin_counter, sample_counter = Counter(uids), Counter(uid_list)
        # print(origin_counter.most_common(5), sample_counter.most_common(5))
        return neg_df