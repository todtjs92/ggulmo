import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import pickle

if __name__ == "__main__":
    df_meta= pd.read_pickle('../data/df_meta_feature.pickle')
    df_meta = df_meta[['href','salePrice','click_count','category_encoding','time_diff']]

    df_positive = pd.read_pickle('../data/df_positive.pickle')
    df_negative = pd.read_pickle('../data/df_negative_filter.pickle')

    df_positive =  df_positive[['uuid','href']]
    df_negative =  df_negative[['uuid','href']]

    # join with meta
    df_positive_merge = df_positive.merge(df_meta, how ='left' , on = 'href')
    df_negative_merge = df_negative.merge(df_meta ,how = 'left' , on = 'href')

    # null 혹시 모르니 제거
    df_positive_merge = df_positive_merge.loc[df_positive_merge['salePrice'].isna()==False]
    df_negative_merge = df_negative_merge.loc[df_negative_merge['salePrice'].isna()==False]

    df_positive_merge['label'] = 1
    df_negative_merge['label'] = 0

    df = pd.concat([df_positive_merge, df_negative_merge])
    df.index = range(len(df))

    # encoding부분
    # 여기있는 유저 , 여기 있는 상품을 대상으로 추천이 나감.. 당연하게도
    category_valeus = df['category_encoding'].values
    category_valeus = list(category_valeus)
    df_category_valeus = pd.DataFrame(category_valeus)

    # category encode
    df_category = pd.concat([df ,df_category_valeus ],axis = 1)
    del df_category['category_encoding']

    # user encode
    user_encoder = LabelEncoder()
    user_encodes = user_encoder.fit(df_category['uuid'])
    with open('../data/user_encodes.pickle','wb') as f:
        pickle.dump(user_encodes,f)

    df_category['uuid'] = user_encodes.transform(df_category['uuid'])

    # item encode
    item_encoder = LabelEncoder()
    item_encodes = item_encoder.fit(df_category['href'])
    with open('../data/item_encodes.pickle','wb') as f:
        pickle.dump(item_encodes,f)

    df_category['href'] = item_encodes.transform(df_category['href'])

    # minmax scale the features
    scaler = MinMaxScaler()
    scaler_fit = scaler.fit(df_category[['salePrice','click_count','time_diff']])
    scaler_transform = scaler_fit.transform(df_category[['salePrice','click_count','time_diff']])
    df_scaler = pd.DataFrame(scaler_transform)
    df_scaler.columns = ['salePrice', 'click_count' , 'time_diff']
    del df_category['salePrice']
    del df_category['click_count']
    del df_category['time_diff']

    # concat features
    df_category_ = pd.concat([df_category, df_scaler],axis= 1)

    user_total = len(set(df_category['uuid'].values))
    item_total = len(set(df_category['href'].values))

    # for FM modeling
    df_category_['href'] += user_total
    df_category_[0] += (user_total + item_total)
    df_category_[1] += (user_total + item_total)
    df_category_[2] += (user_total + item_total)
    df_category_[3] += (user_total + item_total)

    category_max = max(max(df_category_[2].values) , max(df_category_[3].values) )

    df_category_['salePriceidx'] = category_max + 1
    df_category_['click_countidx'] = category_max + 2
    df_category_['time_diffidx'] = category_max + 3
    df_category_.columns = ['user_id','href','label','large1','large2','middle1','middle2','salePrice','click_count','time_diff','salePriceidx','clickcountidx','timediffidx']
    df_category_ = df_category_[['user_id','href','large1','large2','middle1','middle2','salePriceidx','clickcountidx','timediffidx','salePrice','click_count','time_diff','label']]

    state_dict = {}
    state_dict['max_dimension'] = df_category_['timediffidx'][0] + 1
    state_dict['user_dimension'] = user_total
    state_dict['item_dimension'] = item_total
    with open ('../data/state_dict.pickle','wb') as f:
        pickle.dump(state_dict,f)
    df_category_.to_pickle('../data/df_feature_final.pickle')







    #le = LabelEncoder()
    #user_encodes = le.fit(df_click_feature['user_id'])
