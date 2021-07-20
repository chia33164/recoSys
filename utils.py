import numpy as np
import pandas as pd
import scipy.sparse as sp
import csv

def get_user_num(df):
    return df.UserId.unique().shape[0]

def get_item_num(df):
    maxi = 0
    for index, row in df.iterrows():
        data = np.array(row['ItemId'].split())
        for ele in data:
            maxi = max(maxi, int(ele))
    return maxi + 1

def create_negative_data(data, item_num, positive_num, rate=1.0):
    data = data.astype(np.int)
    full_ele = np.arange(0, item_num, 1)
    missing_data = np.delete(full_ele, data).astype(np.str)
    negative_data = np.random.choice(missing_data, int(positive_num * rate))
    return " ".join(negative_data)

def train_test_split(df, user_num, item_num, split_rate = 0.89):
    train_data = df.copy()
    train_data_negative = df.copy()
    test_data = df.copy()
    for index, row in df.iterrows():
        data = np.array(row['ItemId'].split())
        num_col = len(data)
        train_data_num = int(num_col * split_rate)
        train_data.loc[index, 'ItemId'] = " ".join(data[0 : train_data_num])
        test_data.loc[index, 'ItemId'] = " ".join(data[train_data_num:num_col])
        train_data_negative.loc[index, 'ItemId'] = create_negative_data(data, item_num, train_data_num)
    return train_data, train_data_negative, test_data

def transform(df, filename):
    mat = {
        'UserId': [],
        'ItemId': [],
        'Rating': []
    }
    for index, row in df.iterrows():
        ItemIds = row['ItemId'].split()
        UserId = [row['UserId']] * len(ItemIds)
        Rating = [1] * len(ItemIds)
        mat['UserId'].extend(UserId)
        mat['ItemId'].extend(ItemIds)
        mat['Rating'].extend(Rating)
    mat = pd.DataFrame(mat, columns = ['UserId', 'ItemId', 'Rating'])
    mat.to_csv(filename, index=0)
    return mat

def interaction_matrix(df, n_users, n_items):
    target = np.zeros((n_users, n_items))
    for index, row in df.iterrows():
        target[int(row[0]), int(row[1])] = int(row[2])
    return target

def get_test():
    with open('dataset/train.csv', 'r', newline='') as csvfile:
        we = np.array(list(csv.reader(csvfile)))

    with open('dataset/test_data.csv', 'r', newline='') as csvfile:
        ans = np.array(list(csv.reader(csvfile)))


    ans_format = np.copy(we)
    for i in range(len(we)):
        ans_format[i] = ''

    for i in range(1, len(ans)):
        idx = int(ans[i][0]) + 1
        ans_format[idx][1] = str(ans_format[idx][1]) + " " + str(ans[i][1])
    return ans_format


def get_all():
    df = pd.read_csv('./dataset/train.csv')
    n_users, n_items = get_user_num(df), get_item_num(df)
    train_positive = interaction_matrix(transform(df, 'train_pd.csv'), n_users, n_items)
    train_positive = sp.coo_matrix(train_positive)
    return train_positive

get_all()

def get_dataset():
    df = pd.read_csv('./dataset/train.csv')
    n_users, n_items = get_user_num(df), get_item_num(df)
    train_positive, train_negative, test = train_test_split(df, n_users, n_items)
    train_positive = interaction_matrix(transform(train_positive, 'train_data.csv'), n_users, n_items)
    test = interaction_matrix(transform(test, 'test_data.csv'), n_users, n_items)
    assert(np.all((train_positive * test) == 0))
    train_positive = sp.coo_matrix(train_positive)
    test = sp.coo_matrix(test)
    return train_positive, test

def get_dataset_from_local():
    df = pd.read_csv('./dataset/train.csv')
    n_users, n_items = get_user_num(df), get_item_num(df)
    df1 = pd.read_csv('./dataset/train_data.csv')
    df2 = pd.read_csv('./dataset/test_data.csv')
    train = interaction_matrix(df1, n_users, n_items)
    test = interaction_matrix(df2, n_users, n_items)
    train = sp.coo_matrix(train)
    test = sp.coo_matrix(test)
    return train, test

