import pandas as pd
import pickle
from datetime import datetime, timezone


if __name__ == '__main__':

    current_file_path = os.path.abspath(__file__)

    preprocess_dir_path = os.path.dirname(current_file_path)
    main_dir_path = os.path.dirname(os.path.dirname(current_file_path))
    data_dir_path = os.path.join(main_dir_path, 'data')

    print(current_file_path)
    print(data_dir_path)

    '''
    카테고리 정보 모델에 넣주기 위해 처리해줌 
    +
    시간 정보도 처리해줌 -> 시간은 우선 안넣기로 함. 
    
    '''

    # 조회수 add 한 meta 테이블
    df_meta = pd.read_pickle('../data/df_metaview.pickle')

    # parse and encode category
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

    # 모델에 넣주기 위해 category 가져와서 , index 변환해주는 부분임.
    # 0,1,2,3 = 카테고리 정보 없는 경우를 위한 인덱스 번호임 . 이를 비워주기 위해 3을 더해줌.
    # 두번쨰카테고리의 경우 1번째 카테고리와 중복 피하기 위해 1번째 카테고리 길이만큼 더해줌.

    for category_index in range(len(category1_list)):
        category = category1_list[category_index]
        category1_dict[category] = category_index + 3


    for category_index in range(len(category2_list)):
        category = category2_list[category_index]
        category2_dict[category] = category_index + len(category1_dict) + 3


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

    # 사전 저장함 .
    with open('../data/category1_dict.pickle','wb') as f:
        pickle.dump(category1_dict,f)

    with open('../data/category2_dict.pickle','wb') as f:
        pickle.dump(category2_dict,f)



    # 시간부분은 주석 처리하겠음

    # parse time
    # string으로 안바꾸면 에러나는 부분이 있어서 우선 string으로 다 바꿈.

    # df_meta['uploadTime'] = df_meta['uploadTime'].astype(str)
    # current_dt = datetime.now()
    #
    #
    # def get_timediff(date_str):
    #     try:
    #         date_obj = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S.%f")
    #     except:
    #         try:
    #             date_obj = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    #         except:
    #             date_obj = datetime.fromisoformat(date_str)
    #             date_obj = date_obj.replace(tzinfo=None)
    #
    #     delta = current_dt - date_obj
    #     # 180 반년 으로 두어봤음. 회사에서 쓰던거는 240 ,
    #     # 이거는 데이터 보면서 정해야긴함 .
    #     #
    #     if delta.days >= 180:
    #         return 180
    #     return delta.days
    #
    #
    # df_meta['time_diff'] = df_meta['uploadTime'].map(lambda x: get_timediff(x))
    #


    df_meta.to_pickle('../data/df_meta_feature.pickle')
