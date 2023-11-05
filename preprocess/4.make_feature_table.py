import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import pickle
import os
if __name__ == "__main__":

    current_file_path = os.path.abspath(__file__)

    preprocess_dir_path = os.path.dirname(current_file_path)
    main_dir_path = os.path.dirname(os.path.dirname(current_file_path))
    data_dir_path = os.path.join(main_dir_path, 'data')

    print(current_file_path)
    print(data_dir_path)

    '''
    모델에 들어가기 위한 최종 feature 테이블 생성 . 
    '''

    df_meta= pd.read_pickle(data_dir_path + '/df_meta_feature.pickle')
    df_meta = df_meta[['href','category_encoding','saleStatus','click_count2','click_count7','click_count30']]

    df_positive = pd.read_pickle(data_dir_path + '/df_positive.pickle')
    df_negative = pd.read_pickle(data_dir_path + '/df_negative_filter.pickle')
    positive_datas = len(df_positive)
    negative_datas = len(df_negative)

    df_positive =  df_positive[['uuid','href']]
    df_negative =  df_negative[['uuid','href']]

    # join with meta
    df_positive_merge = df_positive.merge(df_meta, how ='left' , on = 'href')
    df_negative_merge = df_negative.merge(df_meta ,how = 'left' , on = 'href')

    # null 혹시 모르니 제거
    df_positive_merge = df_positive_merge.loc[df_positive_merge['click_count2'].isna()==False]
    df_negative_merge = df_negative_merge.loc[df_negative_merge['click_count2'].isna()==False]

    df_positive_merge['label'] = 1
    df_negative_merge['label'] = 0

    df = pd.concat([df_positive_merge, df_negative_merge])
    df.index = range(len(df))

    # encoding부분
    # 여기있는 유저 , 여기 있는 상품을 대상으로 추천이 나감..
    category_valeus = df['category_encoding'].values
    category_valeus = list(category_valeus)
    df_category_valeus = pd.DataFrame(category_valeus)

    # category encode
    df_category = pd.concat([df ,df_category_valeus ],axis = 1)
    del df_category['category_encoding']

    # user encode
    user_encoder = LabelEncoder()
    user_encodes = user_encoder.fit(df_category['uuid'])
    with open(data_dir_path + '/user_encodes.pickle','wb') as f:
        pickle.dump(user_encodes,f)

    df_category['uuid'] = user_encodes.transform(df_category['uuid'])

    # item encode
    item_encoder = LabelEncoder()
    item_encodes = item_encoder.fit(df_category['href'])
    with open(data_dir_path + '/item_encodes.pickle','wb') as f:
        pickle.dump(item_encodes,f)

    df_category['href'] = item_encodes.transform(df_category['href'])
    categorical_vars = df_category.columns
    categorical_vars_length = len(categorical_vars)

    # minmax scale the continuous variables.
    continuous_vars = ['click_count2','click_count7','click_count30']
    continuous_vars_length = len(continuous_vars)
    scaler = MinMaxScaler()
    scaler_fit = scaler.fit(df_category[continuous_vars])
    scaler_transform = scaler_fit.transform(df_category[continuous_vars])
    df_scaler = pd.DataFrame(scaler_transform)
    df_scaler.columns = continuous_vars


    del df_category['click_count2']
    del df_category['click_count7']
    del df_category['click_count30']

    # concat features
    df_category_ = pd.concat([df_category, df_scaler],axis= 1)

    user_total = len(set(df_category['uuid'].values))
    item_total = len(set(df_category['href'].values))

    # for FM modeling
    # To do : category 2개 까지 반영 ,이거 넓히는 시나리오 대비 .

    df_category_['href'] += user_total
    df_category_[0] += (user_total + item_total)
    df_category_[1] += (user_total + item_total)
    df_category_[2] += (user_total + item_total)
    df_category_[3] += (user_total + item_total)

    category_max = max(max(df_category_[2].values) , max(df_category_[3].values) )

    df_category_['click_count2idx'] = category_max + 1
    df_category_['click_count7idx'] = category_max + 2
    df_category_['click_count30idx'] = category_max + 3
    df_category_.columns = ['user_id','href','saleStatus','label','large1','large2','middle1','middle2','click_count2','click_count7','click_count30','click_count2idx','click_count7idx','click_count30idx']
    df_category_ = df_category_[['user_id','href','saleStatus','large1','large2','middle1','middle2','click_count2idx','click_count7idx','click_count30idx','click_count2','click_count7','click_count30','label']]
    
    df_category_onsale = df_category_.loc[df_category_['saleStatus']!='판매완료']
    df_category_onsale_label1 = df_category_onsale.loc[df_category_onsale['label']==1]
    del df_category_['saleStatus']
    del df_category_onsale_label1['saleStatus']
    del df_category_onsale_label1['label']
    
    df_category_.to_pickle(data_dir_path + '/df_feature_final.pickle')
    df_category_onsale_label1.to_pickle(data_dir_path + '/df_category_onsale_label1.pickle')

    # continuous var, label 제외한 부분들이 categorical 변수
    categorical_vars = df_category_.columns[:-(2*continuous_vars_length + 1 ) ]
    categorical_vars_length = len(categorical_vars)

    # feature state 저장 .
    state_dict = {}
    state_dict['max_dimension'] = df_category_['click_count30idx'][0] + 1
    state_dict['user_dimension'] = user_total
    state_dict['item_dimension'] = item_total
    state_dict['continuous_vars'] = continuous_vars
    state_dict['continuous_vars_length'] = continuous_vars_length
    state_dict['categorical_vars'] = categorical_vars
    state_dict['categorical_vars_length'] = categorical_vars_length
    state_dict['positive_datas'] = positive_datas
    state_dict['negative_datas'] = negative_datas


    with open (data_dir_path + '/state_dict.pickle','wb') as f:
        pickle.dump(state_dict,f)

