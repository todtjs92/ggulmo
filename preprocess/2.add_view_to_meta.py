import pandas as pd


if __name__ == '__main__':

    df_meta = pd.read_pickle('../data/df_result_meta.pickle')


    # 조회수 계산할때는 굳이 필터링한것으로 하진 않겠음.

    df_click = pd.read_pickle('../data/df_result_get_item.pickle')


    # 필요한 column만 가져옴 .
    df_click = df_click[['uuid' , 'current_time' , 'href' ]]

    # uuid = '' 없앰.
    df_click = df_click.loc[df_click['uuid'] != '' ]

    # 중복 없앰
    df_click = df_click.drop_duplicates(['uuid', 'current_time'])

    # href가 item_id 임 ,  그룹 바이
    df_click_count = df_click.groupby(by='href').count()
    df_click_count = df_click_count.reset_index()
    df_click_count = df_click_count[['href', 'uuid']]
    df_click_count.columns = ['href', 'click_count']

    df_meta_click = df_meta.merge(df_click_count, on='href', how='left')
    df_meta_click = df_meta_click.fillna(value=0.0)

    df_meta_click.to_pickle('../data/df_meta_view.pickle')

