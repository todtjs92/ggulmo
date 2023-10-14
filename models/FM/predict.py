import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from model import FM
import pickle


if __name__ == "__main__":

    # recommend the item more than 1 ( label 1 )
    df_feature = pd.read_pickle('../../data/df_feature_final.pickle')
    df_feature = df_feature.loc[df_feature['label'] == 1]

    # href columns
    # item
    df_href = df_feature[['href','large1','large2','middle1','middle2','salePriceidx','clickcountidx','timediffidx','salePrice','click_count','time_diff']]
    df_href = df_href.drop_duplicates('href')


    # user
    df_user = df_feature[['user_id']]
    df_user = df_user.drop_duplicates('user_id')

    # model import

    with open('../../data/state_dict.pickle', 'rb') as f:
        state_dict = pickle.load(f)
    field_dims = state_dict['max_dimension']
    user_dims = state_dict['user_dimension']
    print(user_dims)

    learning_rate = 0.01
    reg_lambda = 0.001
    batch_size = 512
    early_stop_trial = 10
    num_epochs = 100
    embed_dim = 16

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model =  FM('', '', '', '', field_dims, num_epochs=num_epochs, embed_dim=embed_dim,
                     learning_rate= learning_rate , reg_lambda= reg_lambda, batch_size=batch_size, early_stop_trial=10, device=device)

    model.restore('../../bestmodel/FM_best_model.pt', device )
    model.eval()

    # click item dict
    with open('../../data/user_click_items.pickle', 'rb') as f:
        user_click_items = pickle.load(f)

    # category TOP 20 per user
    #  user | category | href
    columns = ['user_id','href','large1','large2','middle1','middle2','salePriceidx','clickcountidx','timediffidx','salePrice','click_count','time_diff','score']
    top50df = pd.DataFrame(columns = columns)
    count = 0
    for user_id in df_user['user_id'].values:

        # add userid
        df_href['user_id'] = user_id
        df_pred = df_href[['user_id','href','large1','large2','middle1','middle2','salePriceidx','clickcountidx','timediffidx','salePrice','click_count','time_diff']]

        # df to torch
        pred_data = df_pred.values
        # batch_size = 1024?
        pred_data_loader = DataLoader(range(pred_data.shape[0]), batch_size= 1024, shuffle=False)
        pred_array = np.zeros(pred_data.shape[0])

        for b, batch_idxes in enumerate(pred_data_loader):
            batch_data = torch.tensor(pred_data[batch_idxes], dtype=torch.float, device=device)

            with torch.no_grad():
                pred_array[batch_idxes] = model.forward(batch_data).cpu().numpy()


        # top 50

        df_pred['score'] = pred_array

        # filter by seen data
        click_items = user_click_items[user_id]
        print(len(click_items))
        df_pred = df_pred.loc[df_pred['href'].isin(click_items) == False]
        df_pred = df_pred.sort_values(by='score' , ascending=False)

        #df_pred = df_pred.groupby('middle1').head(20).reset_index(drop=True)


        # user_encodes , item encodes
        with open('../../data/user_encodes.pickle','rb') as f:
            user_encodes= pickle.load(f)
        with open('../../data/item_encodes.pickle','rb') as f:
            item_encodes= pickle.load(f)

        def decoding(encoder , input ):
            result = encoder.inverse_transform(input)
            return result

        user_decodes = decoding(user_encodes , df_pred['user_id'].values)
        df_pred['href'] -= user_dims
        item_decodes = decoding(item_encodes , df_pred['href'].values)
        df_pred['user_id'] = user_decodes
        df_pred['href'] = item_decodes
        #df_pred = df_pred.sort_values(by='middle1')
        top50df = pd.concat([top50df, df_pred ])
        print(count)
        count +=1


    top50df.to_csv('top20_category.csv',index=False)
