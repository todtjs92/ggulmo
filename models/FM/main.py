from util import  load_pickle
from util import train_valid_test_split , split_feature_label
from model import FM
import pickle
import torch
import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
import sys
import datetime
import configparser
from pymongo import MongoClient , UpdateOne


if __name__ == "__main__":

    # 
    config = configparser.ConfigParser()
    config.read('/home/todtjs92/ggulmo_rec/ggulmo/models/FM/config.ini')

    host = config['mongoDB']['host']
    port = config['mongoDB']['port']
    recDB = config['mongoDB']['recDB']
    fmCollection = config['mongoDB']['fmCollection']

    # mongo annotation
    connection_string = f"mongodb://{host}:{port}/"
    print(connection_string)
    client = MongoClient(connection_string)
    database = client[recDB]
    collection = database[fmCollection]


    start_time = datetime.datetime.now()
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

    # cold users 
    with open('../../data/cold_users.pickle','rb') as f:
        cold_users = pickle.load(f)
 
    # return values , numpy array

    df_train_feat , df_train_label = split_feature_label(df_train)
    df_valid_feat, df_valid_label = split_feature_label(df_valid)
    df_test_feat, df_test_label = split_feature_label(df_test)


    # 마지막 feature의 값 + 1 이 field_dims가 됨 .
    # 이거 바꿔야한다 .258610 + 1
    with open('../../data/state_dict.pickle','rb') as f:
        state_dict = pickle.load(f)

    field_dims = state_dict['max_dimension']
    categorical_vars_length = state_dict['categorical_vars_length']
    categorical_vars_length = 3
    print(field_dims)
    print(categorical_vars_length)

    # hyperparm 후에 input으로 받도록 개선
    learning_rate = 0.01
    reg_lambda = 0.001
    batch_size = 512
    early_stop_trial = 10
    num_epochs = 100
    embed_dim = 16

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    fm = FM(df_train_feat, df_train_label, df_valid_feat, df_valid_label, field_dims, num_epochs=num_epochs, embed_dim=embed_dim,categorical_vars_length=categorical_vars_length,
                     learning_rate= learning_rate , reg_lambda= reg_lambda, batch_size=batch_size, early_stop_trial=10, device=device)
    fm.fit()
    print( start_time  - datetime.datetime.now() , "Train end ")

    # 30 days = 2days 
    # make weak 30days weight . 

    weight_2day = torch.tensor(fm.linear.fc.weight.data[-3])
    weight_2day = torch.abs(weight_2day)
    weight_30day = torch.tensor(fm.linear.fc.weight.data[-1])
    weight_30day = torch.abs(weight_30day)

    fm.linear.fc.weight.data[-3] = weight_30day
    fm.linear.fc.weight.data[-1] = weight_2day
    print(fm.linear.fc.weight.data[-3:])
  
    # predict

    fm.eval()

    df_feature = pd.read_pickle('../../data/df_category_onsale_label1.pickle')
    

    ## add feature table join with selling table.

    # href columns
    # item
    df_href = df_feature[['user_id','href','large1','large2','middle1','middle2','click_count2idx','click_count7idx','click_count30idx','click_count2','click_count7','click_count30']]
    df_href = df_href.drop_duplicates('href')


    # user
    df_user = df_feature[['user_id']]
    df_user = df_user.drop_duplicates('user_id')
    
    # model import

    with open('../../data/state_dict.pickle', 'rb') as f:
        state_dict = pickle.load(f)
    field_dims = state_dict['max_dimension']
    user_dims = state_dict['user_dimension']
    item_dims = state_dict['item_dimension']
    categorical_vars_length = state_dict['categorical_vars_length']
    print(user_dims)

######
    # click item dict
    with open('../../data/user_click_items.pickle', 'rb') as f:
        user_click_items = pickle.load(f)

    # user_encodes , item encodes
    with open('../../data/user_encodes.pickle', 'rb') as f:
        user_encodes = pickle.load(f)
    with open('../../data/item_encodes.pickle', 'rb') as f:
        item_encodes = pickle.load(f)

    def decoding(encoder, input):
        result = encoder.inverse_transform(input)
        return result

    with open('../../data/category2_dict.pickle','rb') as f:
        category2_dict = pickle.load(f)
       
    category2_dict_inverse = {v: k for k, v in category2_dict.items()}

    def categoy_decode(x):
        category = category2_dict_inverse[x]
        return category
    
    # category TOP 30 per user
    #  user | category | href
    columns = ['user_id','href','large1','large2','middle1','middle2','click_count2idx','click_count7idx','click_count30idx','click_count2','click_count7','click_count30','score']
    #columns = ['user_id','href','large1','large2','middle1','middle2','salePriceidx','clickcountidx','timediffidx','salePrice','click_count','time_diff','score']
    
    
    result_df = pd.DataFrame(columns = columns)
    cold_df = pd.DataFrame(columns = columns)

    is_cold_df = False
    count = 0
    print("the length is ", len(df_user['user_id'].values))
    for user_id in df_user['user_id'].values:
        
        user_decodes = decoding(user_encodes, [user_id])[0]
        if user_decodes in cold_users :
            print("cold_user  ", user_decodes)
            if is_cold_df == True :
                count+=1
                print(count)
                continue
            
            elif is_cold_df == False :
                print('no cold_df')
                df_href['user_id'] = user_id

                

                df_pred = df_href[['user_id','href','large1','large2','middle1','middle2','click_count2idx','click_count7idx','click_count30idx','click_count2','click_count7','click_count30']]
                pred_data = df_pred.values
                pred_data_loader = DataLoader(range(pred_data.shape[0]), batch_size= 1024, shuffle=False)
                pred_array = np.zeros(pred_data.shape[0])
                for b, batch_idxes in enumerate(pred_data_loader):
                    batch_data = torch.tensor(pred_data[batch_idxes], dtype=torch.float, device=device)

                    with torch.no_grad():
                        pred_array[batch_idxes] = fm.forward(batch_data).cpu().numpy()
                df_pred['score'] = pred_array
                ## cold user는 제거안함 . 

                df_pred['href'] -= user_dims
                item_decodes = decoding(item_encodes, df_pred['href'].values)
                df_pred['user_id'] = user_decodes
                df_pred['href'] = item_decodes

                df_pred['middle1'] -= (user_dims + item_dims)
                # 2 is having no category 
                df_pred = df_pred.loc[df_pred['middle1']!= 2 ]
                df_pred['middle1'] = df_pred['middle1'].map(category2_dict_inverse)



                # cold users don't need click_items
                #df_pred = df_pred.loc[df_pred['href'].isin(click_items) == False]
                cold_df = df_pred.groupby('middle1').head(30).reset_index(drop=True)
                is_cold_df = True
                #user_decodes = decoding(user_encodes , df_pred['user_id'].values)
                #df_pred = df_pred.sort_values(by='middle1')
                result_df = pd.concat([result_df, cold_df ])
                print(count)

                count +=1
                continue


        # add userid
        df_href['user_id'] = user_id
        df_pred = df_href[['user_id','href','large1','large2','middle1','middle2','click_count2idx','click_count7idx','click_count30idx','click_count2','click_count7','click_count30']]

        # df to torch
        pred_data = df_pred.values
        # batch_size = 2048?
        pred_data_loader = DataLoader(range(pred_data.shape[0]), batch_size= 1024, shuffle=False)
        pred_array = np.zeros(pred_data.shape[0])

        for b, batch_idxes in enumerate(pred_data_loader):
            batch_data = torch.tensor(pred_data[batch_idxes], dtype=torch.float, device=device)

            with torch.no_grad():
                pred_array[batch_idxes] = fm.forward(batch_data).cpu().numpy()


        
        # top 30

        df_pred['score'] = pred_array

        # filter by seen data
        
        click_items = user_click_items[user_decodes]
        #click_items = set(click_items)

        df_pred['href'] -= user_dims
        item_decodes = decoding(item_encodes, df_pred['href'].values)
        df_pred['user_id'] = user_decodes
        df_pred['href'] = item_decodes

        df_pred['middle1'] -= (user_dims + item_dims)
        # 2 is having no category 
        df_pred = df_pred.loc[df_pred['middle1']!= 2 ]
        df_pred['middle1'] = df_pred['middle1'].map(category2_dict_inverse)

        df_pred = df_pred.loc[df_pred['href'].isin(click_items) == False]
        df_pred = df_pred.sort_values(by='score',ascending=False)
        df_pred = df_pred.groupby('middle1').head(30).reset_index(drop=True)

        df_pred_select = df_pred[['middle1','href']]

        insert_data = []
        for category, group_df in df_pred_select.groupby('middle1'):
            transformed_data = {}
            transformed_data[category] = {i: v for i, v in enumerate(group_df['href'], 1)}
            insert_data.append(Update_one({'_id':user_decodes},{'$set':transformed_data},upsert=True))

        collection.bulk_write(insert_data)
        
        #user_decodes = decoding(user_encodes , df_pred['user_id'].values)
        #df_pred = df_pred.sort_values(by='middle1')
        #result_df = pd.concat([result_df, df_pred ])
        print(count)
        count +=1

    # insert cold user data
    cold_df = cold_df.sort_values(by='score',ascending=False)
    cold_df = cold_df.groupby('middle1').head(30).reset_index(drop=True)

    cold_df_select = cold_df[['middle1','href']]

    insert_data = []
    for category, group_df in cold_df_select.groupby('middle1'):
        transformed_data = {}
        transformed_data[category] = {i: v for i, v in enumerate(group_df['href'], 1)}
        insert_data.append(Update_one({'_id':'colduser'},{'$set':transformed_data},upsert=True))

    collection.bulk_write(insert_data)


    #print( start_time  - datetime.datetime.now() , "Predict end ")
    #result_df.to_csv('top30_each_category_test.csv',index=False)
    print( start_time  - datetime.datetime.now() , "File write end ")
  