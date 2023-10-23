import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import pickle

import pdb


def get_user_item_info(data_df, root, args):
    
    user2id_file = os.path.join(root, '{}_user2id.pickle'.format(args.dataset))
    id2user_file = os.path.join(root, '{}_id2user.pickle'.format(args.dataset))
    item2id_file = os.path.join(root, '{}_item2id.pickle'.format(args.dataset))
    id2item_file = os.path.join(root, '{}_id2item.pickle'.format(args.dataset))
    if os.path.exists(user2id_file):
        print('loading dataset user item information...')
        with open(user2id_file, 'rb') as fp:
            user2id = pickle.load(fp)

        with open(item2id_file, 'rb') as fp:
            item2id = pickle.load(fp)

        with open(id2user_file, 'rb') as fp:
            id2user = pickle.load(fp)

        with open(id2item_file, 'rb') as fp:
            id2item = pickle.load(fp)
            
            
    else:    
        print('generating dataset user item information...')
        users = data_df['userId'].unique()
        items = data_df['itemId'].unique()
        user2id = dict()
        id2user = dict()
        item2id = dict()
        id2item = dict()
        for u in users:
            if u not in user2id:
                user2id[u] = len(user2id)
                id2user[len(id2user)] = u

        for i in items:
            if i not in item2id:
                item2id[i] = len(item2id)
                id2item[len(id2item)] = i

        with open(user2id_file, 'wb') as fp:
            pickle.dump(user2id, fp)

        with open(item2id_file, 'wb') as fp:
            pickle.dump(item2id, fp)

        with open(id2user_file, 'wb') as fp:
            pickle.dump(id2user, fp)

        with open(id2item_file, 'wb') as fp:
            pickle.dump(id2item, fp)
        
    print("users in total: " + str(len(user2id)))
    print("items in total: " + str(len(item2id)))
    print('interactions in total: ' + str(len(data_df)))
    
    return user2id, id2user, item2id, id2item

def uniform_expo_split(data_df, user2id, item2id, ratio):
    user_num = len(data_df['userId'].unique())
    item_num = len(data_df['itemId'].unique())
    user_group_df = data_df.groupby('userId')
    index_list = []
    for i in tqdm(data_df['userId'].unique()):
        group_df = user_group_df.get_group(i)
        total_count = len(group_df.index)
        treat_num = int(total_count * ratio) if total_count * ratio >= 1 else 1
        sampled_index = group_df.sample(n=treat_num, weights=group_df['weight']).index.tolist()
        index_list.extend(sampled_index)
#         pdb.set_trace()
#         count = 0
#         while count < treat_num:
#             for index, row in group_df.iterrows():
#                 if index in index_list:
#                     continue
#                 row = data_df.iloc[index]
#                 if np.random.rand() < prob_list[item2id[row['itemId']]]:
#                     index_list.append(index)
#                     count += 1
#                 if count == treat_num:
#                     break
    split_df = data_df.iloc[index_list].reset_index(drop=True)
    remain_df = data_df[~data_df.index.isin(index_list)].reset_index(drop=True)
    return remain_df, split_df

def convert_pop2prob(popularity_dict, item2id, min_pop, item_num, max_prob):
    prob_list = np.zeros(item_num)
    for item in popularity_dict.keys():
        i = item
        try:
            prob_list[i] =  min_pop / (popularity_dict[item] - min_pop)
        except ZeroDivisionError:
            prob_list[i] = max_prob
        if prob_list[i] > max_prob:
            prob_list[i] = max_prob
    return prob_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model', add_help=False)
    parser.add_argument('--dataset', type=str, default='Electronics_SKEW', help='the name of dataset')
    parser.add_argument('--max-prob', type=float, default=0.9, help='the maximum probability for splitting into test set.')
    parser.add_argument('--test-ratio', type=float, default=0.2, help='the ratio of test data for each user')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='the ratio of validation data for each user')
    parser.add_argument('--imp', type=int, default=1, help='1 for implicit feedback, 0 for >3 as positive')
    args = parser.parse_args()

    root = '../../datasets/{}'.format(args.dataset)
    data_path = os.path.join(root, '{}.csv'.format(args.dataset))
    print("loading all data")
    col_names = ['itemId', 'userId', 'rating', 'time']
    data_df = pd.read_csv(data_path, names=col_names)
    correct_col_names = ['userId', 'itemId', 'rating', 'time']
    data_df = data_df[correct_col_names]
    
    user2id, id2user, item2id, id2item = get_user_item_info(data_df, root, args)
    data_df['userId'] = [user2id[i] for i in data_df['userId'].tolist()]
    data_df['itemId'] = [item2id[i] for i in data_df['itemId'].tolist()]
    
    popularity_dict = data_df['itemId'].value_counts().to_dict()
    min_pop = min(popularity_dict.values())
    prob_list = convert_pop2prob(popularity_dict, item2id, min_pop, len(item2id), args.max_prob)
    weight = [prob_list[item] for item in data_df['itemId'].tolist()]
    data_df['weight'] = weight
    
    if args.imp == 0:
        data_df = data_df[data_df['rating'] > 3].reset_index(drop=True)
    
#     pdb.set_trace()
    print('splitting test set which expose as uniformly as possible each user to each product')
    remain_df, test_df = uniform_expo_split(data_df, user2id, item2id, args.test_ratio)
    print('splitting validation set which expose as uniformly as possible each user to each product')
    train_df, val_df = uniform_expo_split(remain_df, user2id, item2id, args.val_ratio/(1-args.test_ratio))
    
    print('interactions in train: ' + str(len(train_df)))
    print('interactions in validation: ' + str(len(val_df)))
    print('interactions in test: ' + str(len(test_df)))
    
    if args.imp == 1:
        train_df[correct_col_names].to_csv(os.path.join(root, '{}_imp.train.csv'.format(args.dataset)), index=False, header=False)
        val_df[correct_col_names].to_csv(os.path.join(root, '{}_imp.validation.csv'.format(args.dataset)), index=False, header=False)
        test_df[correct_col_names].to_csv(os.path.join(root, '{}_imp.test.csv'.format(args.dataset)), index=False, header=False)
    train_df[correct_col_names].to_csv(os.path.join(root, '{}.train.csv'.format(args.dataset)), index=False, header=False)
    val_df[correct_col_names].to_csv(os.path.join(root, '{}.validation.csv'.format(args.dataset)), index=False, header=False)
    test_df[correct_col_names].to_csv(os.path.join(root, '{}.test.csv'.format(args.dataset)), index=False, header=False)