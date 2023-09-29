from util import  load_pickle
from util import train_valid_test_split , split_feature_label
from model import FM
import torch
import pandas as pd
import numpy as np
import random
import torch


if __name__ == "__main__":


    # seed
    random_seed = 1
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    # read feature table from pickle ->
    data_path = '../../data/df_feature_final.pickle'
    df = load_pickle(data_path)

    # return dataframe
    df_train , df_valid , df_test  = train_valid_test_split(df )


    # return values , numpy array

    df_train_feat , df_train_label = split_feature_label(df_train)
    df_valid_feat, df_valid_label = split_feature_label(df_valid)
    df_test_feat, df_test_label = split_feature_label(df_test)


    # 마지막 feature의 값 + 1 이 field_dims가 됨 .
    # 이거 바꿔야한다 .
    field_dims = 245495

    # hyperparm 후에 input으로 받도록 개선
    learning_rate = 0.01
    reg_lambda = 0.001
    batch_size = 1024
    early_stop_trial = 20
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fm = FM(df_train_feat, df_train_label, df_valid_feat, df_valid_label, field_dims, num_epochs=1, embed_dim=20,
                     learning_rate= learning_rate , reg_lambda= reg_lambda, batch_size=batch_size, early_stop_trial=10, device=device)
    fm.fit()
    print('end')

    # df_size

    # hyper param




    print(len(df_train), len(df_valid) , len(df_test)   )

