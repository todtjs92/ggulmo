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

    # 클릭 2회이상인 유저여도 같은 상품 2번이면 나올수 있음 .

    with open('../data/user_click_items.pickle','wb') as f:
        pickle.dump(user_click_items, f)

    user_impressed_items = defaultdict(set)
    for uuid , href_list in zip(df_negative['uuid'] , df_negative['item_id_ret_list']):
        set_ = set(href_list)
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
    df_impress = df_impress.loc[df_impress['length']> 0 ]


    # get random impression data
    def select_random(x):
        try:
            selected_items = random.sample(x, 20)
        except:
            selected_items = x
        return selected_items
    df_impress['href'] = df_impress['impressed_items'].map(select_random)
    df_impress_ = df_impress.explode(column=['href'],ignore_index=True)
    df_impress = df_impress_[['uuid', 'href']]

    df_impress.to_pickle('../data/df_negative_filter.pickle')
