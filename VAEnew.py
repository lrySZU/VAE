import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F


def split_data(data, part1, part2):
    """
    random.seed: If the parameter remains the same, a fixed initial value will be generated
    generate random sequences, use 'iloc' to split train / val / test
    :param data: rating_base
    :param part1: train_percentage
    :param part2: valid_percentage
    :return:
    """
    np.random.seed(0)
    shuffled_indices = np.random.permutation(len(data))  # a shuffled sequence

    train_size = int(len(data) * part1)  # type conversion
    valid_size = int(len(data) * part2)

    train_idx = shuffled_indices[: train_size]
    valid_idx = shuffled_indices[train_size: train_size + valid_size]
    test_idx = shuffled_indices[train_size + valid_size:]

    # iloc: just integer index -> [ )
    # loc: just label index -> [ ]
    train_data = data.iloc[train_idx]
    valid_data = data.iloc[valid_idx]
    test_data = data.iloc[test_idx]

    return train_data, valid_data, test_data


def filter_data(train_data, valid_data, test_data):
    """
    Remove the movies from the validation set and test set that are not present in the training set
    :param train_data:
    :param valid_data:
    :param test_data:
    :return: valid_data and test_data after filtering
    """
    train_movie = train_data['movie_id'].drop_duplicates().tolist()
    valid_movie = valid_data['movie_id'].drop_duplicates().tolist()
    test_movie = test_data['movie_id'].drop_duplicates().tolist()

    valid_movie = list(set(valid_movie).intersection(set(train_movie)))
    test_movie = list(set(test_movie).intersection(set(train_movie)))

    # isin: Simultaneously check if the data is equal to multiple values (list / dict / dataframe)
    valid_data = valid_data[valid_data.movie_id.isin(valid_movie)]
    test_data = test_data[test_data.movie_id.isin(test_movie)]

    return valid_data, test_data


def read_data(root1):
    r_cols = ['user_id', 'movie_id', 'rating', 'time']
    ratings_base = pd.read_csv(root1, sep="::", names=['user_id', 'movie_id', 'rating', 'time'], engine='python')

    ratings_base.drop(['time'], axis=1, inplace=True)  # only when inplace is true do an operation

    ratings_base = ratings_base[(ratings_base['rating'] >= 1)]

    train_data, valid_data, test_data = split_data(ratings_base, 0.6, 0.2)
    valid_data, test_data = filter_data(train_data, valid_data, test_data)

    return train_data, valid_data, test_data


def ranking_matrix(train_data, valid_data, test_data, user_num, item_num):
    """
    :param train_data: pd.DataFrame
    :param valid_data: pd.DataFrame
    :param test_data: pd.DataFrame
    :param user_num: not filtered
    :param item_num: not filtered
    :return:
    """
    train_mat = np.zeros((user_num, item_num))
    valid_mat = np.zeros((user_num, item_num))
    test_mat = np.zeros((user_num, item_num))

    train_user = train_data['user_id'].drop_duplicates().tolist()
    valid_user = valid_data['user_id'].drop_duplicates().tolist()
    test_user = test_data['user_id'].drop_duplicates().tolist()

    # hashset for user -> movies
    train_movie = {}
    for user in train_user:
        train_movie[user] = train_data.loc[train_data['user_id'] == user]['movie_id'].drop_duplicates().tolist()
    valid_movie = {}
    for user in valid_user:
        valid_movie[user] = valid_data.loc[valid_data['user_id'] == user]['movie_id'].drop_duplicates().tolist()
    test_movie = {}
    for user in test_user:
        test_movie[user] = test_data.loc[test_data['user_id'] == user]['movie_id'].drop_duplicates().tolist()

    # matrix
    for user in train_user:
        for movie in train_movie[user]:
            train_mat[user][movie] = 1

    for user in valid_user:
        for movie in valid_movie[user]:
            valid_mat[user][movie] = 1

    for user in test_user:
        for movie in test_movie[user]:
            test_mat[user][movie] = 1

    return train_mat, valid_mat, test_mat


class MyDataset(Dataset):
    def __init__(self, train_mat):
        self.train_mat = train_mat

    def __getitem__(self, user):
        X = torch.tensor(self.train_mat[user], dtype=torch.float)
        user = torch.tensor([user, ], dtype=torch.long)  # integer -> tensor
        return X, user

    def __len__(self):
        return len(self.train_mat)


class VaE(nn.Module):
    def __init__(self, movie_num, hidden_num):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(movie_num, movie_num),
            nn.Dropout(0.1)  # set some zeros
        )
        self.mu = nn.Linear(movie_num, hidden_num)
        self.logger = nn.Linear(movie_num, hidden_num)  # input layer -> hidden layer
        self.decoder = nn.Linear(hidden_num, movie_num)  # hidden layer -> output layer

    def encode(self, x):
        encoder = torch.sigmoid(self.encoder(x))  # Linear encoding + activation function  !not the original mat
        # mu and logger are not weight parameters to be optimized
        mu, logger = self.mu(encoder), self.logger(encoder)
        return mu, logger

    def reset_parameter(self, mu, logger, flag):
        if flag:  # training mode
            std = torch.exp(logger / 2)
            eps = torch.randn_like(std)  # the same shape as std with normal distribution data
            zu = mu + eps * std  # ~ N(mu, eps * std)
            return zu

        else:  # not regularization in testing mode
            return mu

    def decode(self, z):
        out = torch.sigmoid(self.decoder(z))
        return out

    # combines encoder and decoder
    def forward(self, x, flag):
        mu, logger = self.encode(x)
        zu = self.reset_parameter(mu, logger, flag)
        pred_y = self.decode(zu)
        return pred_y, mu, logger


def loss_function(pred_y, x, mu, logger):
    """
    loss = loss for logistic regression (classification for feedback) + kl loss
    kl loss: the expected value of the logarithmic difference between the probabilities of the original distribution
     and the approximate distribution (normal distribution) of the data
    :param pred_y:
    :param x:
    :param mu:
    :param logger:
    :return:
    """
    beta = 0.2  # why I need to set beta as 0.2
    pred_loss = F.binary_cross_entropy(input=pred_y, target=x, reduction='sum')
    kld = - 0.5 * torch.sum(1 + logger - mu ** 2 - logger.exp())
    loss = pred_loss + beta * kld
    return loss


def mytest(train_data, test_data, vae, test_mat, train_mat):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pre_k, rec_k = 0, 0

    test_user = test_data['user_id'].drop_duplicates().tolist()
    train_user = train_data['user_id'].drop_duplicates().tolist()
    train_movie = {}

    for user in train_user:
        train_movie[user] = train_data.loc[train_data['user_id'] == user][
            "movie_id"].drop_duplicates().tolist()

    # use list to slice np array, convert it into tensor
    input_data = torch.tensor(train_mat[test_user], dtype=torch.float).to(device)
    flag = 0  # test mode
    # out is a tensor, need to be converted into ndarray
    out, mu, logger = vae(input_data, flag)
    top_k = 10

    # no grad -> cpu -> numpy
    out = out.detach()
    out = out.cpu().numpy()

    # sorted by lines
    tmp_list = np.argsort(-out, axis=1, kind='quicksort')
    print(f'tmp_list.shape={tmp_list.shape}')

    for i, u in enumerate(test_user):
        hit = 0
        test_list = test_data.loc[test_data['user_id'] == u][
            "movie_id"].drop_duplicates().tolist()

        recommend = []
        for movie in tmp_list[i]:
            if train_mat[u][movie] == 0:
                recommend.append(movie)
            if len(recommend) == top_k:
                break

        for movie in recommend:
            if test_mat[u][movie] == 1:
                hit += 1
        rec_k += hit / len(test_list)
        pre_k += hit / top_k

    recall = rec_k / len(test_user)
    precision = pre_k / len(test_user)
    print('recall:{}, precision:{}'.format(recall, precision))


def train(movie_num, hidden_num, dataloader, train_data, test_data, test_mat, train_mat):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae = VaE(movie_num, hidden_num).to(device)
    lr = 1e-3
    opt = torch.optim.Adam(vae.parameters(), lr=lr)
    iters = 100
    for i in range(1, iters + 1):
        for x, user in dataloader:
            x = x.to(device)
            flag = 1  # training mode
            out, mu, logger = vae.forward(x, flag)
            loss = loss_function(out, x, mu, logger)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if i % 10 == 0:
            print(f'iter = {i}')
            mytest(train_data, test_data, vae, test_mat, train_mat)


if __name__ == '__main__':
    root1 = 'ml-1m/ratings.txt'
    train_data, valid_data, test_data = read_data(root1)
    print('Reading data is complete')

    user_num = 6041
    movie_num = 4000
    hidden_num = 200

    train_mat, valid_mat, test_mat = ranking_matrix(train_data, valid_data, test_data, user_num, movie_num)
    dataset = MyDataset(train_mat)
    # shuffle=True: returns in a different order each time
    dataloader = DataLoader(dataset=dataset, batch_size=100, shuffle=True)
    print('Data loaded')

    train(movie_num, hidden_num, dataloader, train_data, test_data, test_mat, train_mat)
