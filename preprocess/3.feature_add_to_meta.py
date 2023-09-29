import pandas as pd
from datetime import datetime, timezone


if __name__ == '__main__':


    df_meta = pd.read_pickle('../data/df_meta_view.pickle')

    print(df_meta.columns)


    # category
    category1_set = set()
    category2_set = set()

    def get_category(category2_nm):
        for i in category2_nm:
            category1, category2 = i.split('-')
            category1_set.add(category1)
            category2_set.add(category2)


    df_meta['category2_nm'].map(lambda x: get_category(x))
    category1_list = list(category1_set)
    category2_list = list(category2_set)

    all_categories = len(category1_list + category2_list)

    category1_dict = {}
    category2_dict = {}

    for category_index in range(len(category1_list)):
        category = category1_list[category_index]
        category1_dict[category] = category_index + 3


    for category_index in range(len(category2_list)):
        category = category2_list[category_index]
        category2_dict[category] = category_index + len(category1_dict) + 3

    print(category1_dict)
    print(category2_dict)


    def get_categorylist(category2_nm):
        # 4개까지 1순위 대,중 2순위 대 , 중 , 0 은 없음임 .

        category_list = [x for x in range(4)]
        count = 0
        for i in category2_nm:
            category1, category2 = i.split('-')
            if category1 in category1_dict:
                category1_idx = category1_dict[category1]
                category_list[count] = category1_idx

            if category2 in category2_dict:
                category2_idx = category2_dict[category2]
                category_list[count + 2] = category2_idx
            count += 1
            if count == 2:
                break
        return category_list


    df_meta_category_encoding = df_meta['category2_nm'].map(lambda x: get_categorylist(x))
    df_meta['category_encoding'] = df_meta_category_encoding


    # time
    df_meta['uploadTime'] = df_meta['uploadTime'].astype(str)
    current_dt = datetime.now()


    def get_timediff(date_str):
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S.%f")
        except:
            try:
                date_obj = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
            except:
                date_obj = datetime.fromisoformat(date_str)
                date_obj = date_obj.replace(tzinfo=None)

        delta = current_dt - date_obj
        if delta.days >= 180:
            return 180
        return delta.days


    df_meta['time_diff'] = df_meta['uploadTime'].map(lambda x: get_timediff(x))



    df_meta.to_pickle('../data/df_meta_feature.pickle')
