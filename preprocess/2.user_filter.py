import pandas as pd
import os
if __name__ == '__main__':

    current_file_path = os.path.abspath(__file__)

    preprocess_dir_path = os.path.dirname(current_file_path)
    main_dir_path = os.path.dirname(os.path.dirname(current_file_path))
    data_dir_path = os.path.join(main_dir_path, 'data')

    print(current_file_path)
    print(data_dir_path)

    '''
    click 데이터를 바탕으로 postivie data set , 
    impression 데이터를 바탕으로 negative data set 
    을 만듬.  
    '''

    # click data
    df_result_get_item = pd.read_pickle(data_dir_path + '/df_get_item.pickle')

    # impression data
    df_result_get_items = pd.read_pickle(data_dir_path + '/df_get_items.pickle')
    df_result_get_related_items = pd.read_pickle(data_dir_path + '/df_get_related_items.pickle')
    df_result_get_newest_items = pd.read_pickle(data_dir_path + '/df_get_newest_items.pickle')


    # user가 '' 인 것들 제거

    df_result_get_item = df_result_get_item.loc[df_result_get_item['uuid']!= '']
    df_result_get_items = df_result_get_items.loc[df_result_get_items['uuid']!= '']
    df_result_get_related_items = df_result_get_related_items.loc[df_result_get_related_items['uuid']!= '']
    df_result_get_newest_items = df_result_get_newest_items.loc[df_result_get_newest_items['uuid']!='']

    # 중복인 로그는 제거하였음, current_time 이랑 user id 가 같은 경우.
    df_result_get_item = df_result_get_item.drop_duplicates(['uuid','current_time'])
    df_result_get_items = df_result_get_items.drop_duplicates(['uuid','current_time'])
    df_result_get_related_items = df_result_get_related_items.drop_duplicates(['uuid','current_time'])
    df_result_get_newest_items = df_result_get_newest_items.drop_duplicates(['uuid','current_time'])

    # drop outlier
    # 클릭이 1회 이상인 사람 구하기 ,
    df_group = df_result_get_item.groupby(by='uuid').count()
    df_group_moreone_user = df_group.loc[df_group['href']>=2]
    # 기준 체크 해봐야는데 한 1000개 미만인사람만 걸러보자 # 기준 정하기 .
    df_group_moreone_user_limit = df_group_moreone_user.loc[df_group_moreone_user['href']< 1000]

    user_list = df_group_moreone_user_limit.index.values

    # click이든 imp든 적어도 2번이상 클릭한 사람들을 대상으로함

    df_positive = df_result_get_item.loc[df_result_get_item['uuid'].isin(user_list)]
    df_positive.to_pickle(data_dir_path + '/df_positive.pickle')

    df_result_get_items = df_result_get_items.loc[df_result_get_items['uuid'].isin(user_list)]
    df_result_get_related_items = df_result_get_related_items.loc[df_result_get_related_items['uuid'].isin(user_list)]
    df_result_get_newest_items = df_result_get_newest_items.loc[df_result_get_newest_items['uuid'].isin(user_list)]

    # aggregate 3 df to df_negative

    df_negative = pd.concat([df_result_get_items,df_result_get_related_items ,df_result_get_newest_items])
    df_negative = df_negative.to_pickle(data_dir_path + '/df_negative.pickle')


