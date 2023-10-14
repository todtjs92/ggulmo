import pandas as pd

if __name__ == '__main__':

    # click data
    df_result_get_item = pd.read_pickle('../data/df_result_get_item.pickle')

    # impression data
    df_result_get_items = pd.read_pickle('../data/df_result_get_items.pickle')
    df_result_get_related_items = pd.read_pickle('../data/df_result_get_related_items.pickle')
    df_result_get_newest_items = pd.read_pickle('../data/df_result_get_newest_items.pickle')
    # newest 지금은 제외
    # ''제거

    df_result_get_item = df_result_get_item.loc[df_result_get_item['uuid']!= '']
    df_result_get_items = df_result_get_items.loc[df_result_get_items['uuid']!= '']
    df_result_get_related_items = df_result_get_related_items.loc[df_result_get_related_items['uuid']!= '']
    df_result_get_newest_items = df_result_get_newest_items.loc[df_result_get_newest_items['uuid']!='']

    # drop duplicate
    df_result_get_item = df_result_get_item.drop_duplicates(['uuid','current_time'])
    df_result_get_items = df_result_get_items.drop_duplicates(['uuid','current_time'])
    df_result_get_related_items = df_result_get_related_items.drop_duplicates(['uuid','current_time'])
    df_result_get_newest_items = df_result_get_newest_items.drop_duplicates(['uuid','current_time'])

    # drop outlier
    # 클릭이 1회 이상인 사람 구하기 ,
    df_group = df_result_get_item.groupby(by='uuid').count()
    df_group_moreone_user = df_group.loc[df_group['href']>=2]
    # 기준 체크 해봐야는데 한 1000개 미만인사람만 걸러보자
    df_group_moreone_user_limit = df_group_moreone_user.loc[df_group_moreone_user['href']< 1000]

    user_list = df_group_moreone_user_limit.index.values

    # click이든 imp든 적어도 2번이상 클릭한 사람들을 대상으로함

    df_positive = df_result_get_item.loc[df_result_get_item['uuid'].isin(user_list)]
    df_positive.to_pickle('../data/df_positive.pickle')

    df_result_get_items = df_result_get_items.loc[df_result_get_items['uuid'].isin(user_list)]
    df_result_get_related_items = df_result_get_related_items.loc[df_result_get_related_items['uuid'].isin(user_list)]
    df_result_get_newest_items = df_result_get_newest_items.loc[df_result_get_newest_items['uuid'].isin(user_list)]

    # aggregate 3 df to df_negative

    df_negative = pd.concat([df_result_get_items,df_result_get_related_items ,df_result_get_newest_items])
    df_negative = df_negative.to_pickle('../data/df_negative.pickle')


