import pandas as pd
from datetime import datetime , timedelta

if __name__ == '__main__':

    '''
    1달치 로그를 바탕으로 조회수를 구하는 부분 . 
    '''

    df_meta = pd.read_pickle('../data/df_meta.pickle')


    df_click = pd.read_pickle('../data/df_get_item.pickle')

    # 필요한 column만 가져옴 .
    df_click = df_click[['uuid' , 'current_time' , 'href' ]]

    # uuid = '' 없앰. 잘못들어온 로그임.
    df_click = df_click.loc[df_click['uuid'] != '' ]

    # 중복 없앰
    df_click = df_click.drop_duplicates(['uuid', 'current_time'])

    # 2일, 7일 , 30일 형태로 데이터 만듬 .
    today = datetime.now()
    days30ago = today - timedelta(days=30)
    days7ago = today - timedelta(days=7)
    days2ago = today - timedelta(days=2)

    # 0시 기준으로 하겠음 내림함수
    days30ago = days30ago.replace(hour=0, minute=0, second=0, microsecond=0)
    days7ago  = days7ago.replace(hour=0 ,minute=0 , second= 0 ,microsecond=0)
    days2ago = days2ago.replace(hour=0, minute=0, second=0, microsecond=0)

    #30일
    df_click30 = df_click.loc[df_click['current_time'] >= days30ago]
    df_click30 = df_click30.groupby(by='href').count()
    df_click30 = df_click30.reset_index()
    df_click30 = df_click30[['href', 'uuid']]
    df_click30.columns = ['href', 'click_count30']

    # 7일
    df_click7 = df_click.loc[df_click['current_time'] >= days7ago]
    df_click7 = df_click7.groupby(by='href').count()
    df_click7 = df_click7.reset_index()
    df_click7 = df_click7[['href', 'uuid']]
    df_click7.columns = ['href', 'click_count7']

    # 2일
    df_click2 = df_click.loc[df_click['current_time'] >= days2ago]
    df_click2 = df_click2.groupby(by='href').count()
    df_click2 = df_click2.reset_index()
    df_click2 = df_click2[['href', 'uuid']]
    df_click2.columns = ['href', 'click_count2']

    # href가 item_id 임 ,  그룹 바이
    # 2일 , 3일 , 7일 조인 .
    df_meta_click = df_meta.merge(df_click2, on='href', how='left')
    df_meta_click = df_meta_click.merge(df_click7, on='href', how='left')
    df_meta_click = df_meta_click.merge(df_click30, on='href', how='left')

    df_meta_click = df_meta_click.fillna(value=0.0)

    df_meta_click.to_pickle('../data/df_metaview.pickle')

