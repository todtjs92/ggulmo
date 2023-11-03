import pandas as pd
from collections import defaultdict
import random
import pickle

if __name__ == "__main__":

    '''
    negtative data set에서 학습에 들어갈 데이터 선별 하는 부분 . 
    
    '''

    df_negative = pd.read_pickle('../data/df_negative.pickle')
    df_positive = pd.read_pickle('../data/df_positive.pickle')

    user_click_items = defaultdict(set)
    for uuid , href in zip(df_positive['uuid'] , df_positive['href']):
        user_click_items[uuid].add(href)

    # 클릭 3회 이하 유저들 따로 분리해둠 . - interaction 쪽에서 크게 영향안 받어 같은 추천 결과가 나갈거임 .
    cold_users = set()
    for user_id , href_set in user_click_items.items():
        item_length = len(href_set)
        if item_length <= 3 :
            cold_users.add(user_id)
    with open('../data/cold_users.pickle','wb') as f:
        pickle.dump(cold_users, f )

    # 클릭 2회이상인 유저여도 같은 상품 2번이면 나올수 있음 .

    with open('../data/user_click_items.pickle','wb') as f:
        pickle.dump(user_click_items, f)

    user_impressed_items = defaultdict(set)
    for uuid , href_list in zip(df_negative['uuid'] , df_negative['item_id_ret_list']):
        set_ = set(href_list)
        if 'none' in set_:
            set_.remove('none')
        user_impressed_items[uuid].update(set_)

    # 본 id 제외
    for uuid , item_list in user_impressed_items.items():
        if uuid in user_click_items:
            user_impressed_items[uuid] = user_impressed_items[uuid] - user_click_items[uuid]

    # Convert the defaultdict to DataFrame
    df_impress = pd.DataFrame(list(user_impressed_items.items()), columns=['uuid', 'impressed_items'])
    df_impress['impressed_items'] = df_impress['impressed_items'].apply(list)

    df_impress['length'] = df_impress['impressed_items'].map(lambda x: len(x))



    # 0이상인 impression 만 가져옴 , 잘못된 데이터 필터링 .
    #
    df_impress = df_impress.loc[df_impress['length']> 0]

    # make clickcount for resize the negative set . 
    df_positive_group = df_positive.groupby(by='uuid').count()
    df_positive_group= df_positive_group[['func']]
    df_positive_group= df_positive_group.reset_index()
    click_count = dict()
    for uuid, count in zip(df_positive_group['uuid'] , df_positive_group['func']):
        click_count[uuid] = count

    # get random impression data
    def select_random(impressed_items, click_counts_user ):
        try:
            selected_items = random.sample( impressed_items ,click_counts_user )
        except:
            selected_items = impressed_items
        return selected_items
    # iteration for random sampling
    random_items = []
    for uuid, impressed_items in zip(df_impress['uuid'] , df_impress['impressed_items']): 
        click_counts_user = click_count[uuid]
        samples = select_random(impressed_items,click_counts_user)
        random_items.append(samples)
        

    df_impress['href'] = random_items
    df_impress_ = df_impress.explode(column=['href'],ignore_index=True)
    df_impress = df_impress_[['uuid', 'href']]

    df_impress.to_pickle('../data/df_negative_filter.pickle')
