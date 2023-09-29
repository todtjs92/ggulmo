import pandas as pd
from collections import defaultdict
import random

df_meta= pd.read_pickle('../data/df_meta_feature.pickle')
df_negative = pd.read_pickle('../data/df_negative.pickle')
df_positive = pd.read_pickle('../data/df_positive.pickle')



user_click_items = defaultdict(set)
for uuid , href in zip(df_positive['uuid'] , df_positive['href']):
    user_click_items[uuid].add(href)


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
df_impress = df_impress.loc[df_impress['length']> 0 ]


def select_random_3(x):
    try:
        selected_items = random.sample(x, 20)
    except:
        selected_items = x
    return selected_items
df_impress['href'] = df_impress['impressed_items'].map(select_random_3)
df_impress_ = df_impress.explode(column=['href'],ignore_index=True)
df_impress = df_impress_[['uuid', 'href']]

df_impress.to_pickle('../data/df_negative_filter.pickle')
