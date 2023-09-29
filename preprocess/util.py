def get_item_parser(response):
    '''
    Parser for get_item - 클릭로그
    '''
    # user_info
    func = response['func']
    user_info = response['user_info']
    uuid = user_info['uuid']
    url = user_info['url']
    query = user_info['query_parmeters']
    item_id = user_info['path_parameters']

    # To Do -> Make a rule for cookie
    try:
        cookie = user_info['headers']['cookie']
    except:
        try:
            cookie = user_info['headers']['cookies']
        except:
            cookie = user_info['cookies']

    # headers에 없고 그냥 쿠키에있는 부분도있음.
    # response

    current_time = response['currentTime']

    # ret
    ret = response['ret']
    # href 가 곧 item_id가 됨 .
    href = ret['href']
    # ret의 상품아이디 체크하려고 우선 뽑음
    item_id_ret = ret['id']
    item_id_ret_list = [item_id_ret]
    # 꿀모에 업로드 된시간
    upload_time = ret['uploadTime']

    return func, uuid, url, query, item_id, href,  item_id_ret_list, cookie , current_time, upload_time



def get_items_parser(response):
    '''
    Parser for get_items - 검색
    '''

    # user_info
    func = response['func']
    user_info = response['user_info']
    uuid = user_info['uuid']
    url = user_info['url']

    query = user_info['query_parmeters']
    item_id = user_info['path_parameters']

    try:
        cookie = user_info['headers']['cookie']
    except:
        try:
            cookie = user_info['headers']['cookies']
        except:
            cookie = user_info['cookies']

            # referer
    href = ''

    # response
    current_time = response['currentTime']

    # ret , get_items = ret['docs']안에 아이템이 있음.

    docs = response['ret']['docs']

    # ret의 상품아이디 체크하려고 우선 뽑음 , list안에 클릭한 item_id 다 뽑자
    item_id_ret_list = []

    for doc in docs:
        try:
            item_id_ret = doc['href']
        except:
            item_id_ret = 'none'

        item_id_ret_list.append(item_id_ret)

    return func, uuid, url, query, item_id, item_id_ret_list, current_time, cookie


def get_related_items_parser(response):
    '''
    Parser for get_items
    '''
    func = response['func']

    # user_info
    user_info = response['user_info']
    uuid = user_info['uuid']
    url = user_info['url']

    query = user_info['query_parmeters']
    item_id = user_info['path_parameters']

    try:
        cookie = user_info['headers']['cookie']
    except:
        try:
            cookie = user_info['headers']['cookies']
        except:
            cookie = user_info['cookies']

    # response
    current_time = response['currentTime']

    # ret -> get_related_items = ret이 여러개있음

    rets = response['ret']

    # ret의 상품아이디 체크하려고 우선 뽑음 , list안에 클릭한 item_id 다 뽑자
    item_id_ret_list = []

    for ret in rets:
        try:
            item_id_ret = ret['href']  # _id로 되있음
        except:

            item_id_ret = 'none'

        item_id_ret_list.append(item_id_ret)

    return func, uuid, url, query, item_id, item_id_ret_list, current_time, cookie


def get_newest_items_parser(response):
    '''
    Parser for newest_items
    '''

    func = response['func']
    # user_info
    user_info = response['user_info']
    uuid = user_info['uuid']
    url = user_info['url']

    query = user_info['query_parmeters']
    item_id = user_info['path_parameters']

    try:
        cookie = user_info['headers']['cookie']
    except:
        try:
            cookie = user_info['headers']['cookies']
        except:
            cookie = user_info['cookies']

    # response
    current_time = response['currentTime']

    # ret , get_items = ret['docs']안에 아이템이 있음.

    docs = response['ret']['docs']

    # ret의 상품아이디 체크하려고 우선 뽑음 , list안에 클릭한 item_id 다 뽑자
    item_id_ret_list = []

    for doc in docs:
        try:
            item_id_ret = doc['href']
        except:
            item_id_ret = ''
        item_id_ret_list.append(item_id_ret)

    return func, uuid, url, query, item_id, item_id_ret_list, current_time, cookie

def meta_parser(response):

    _id = response['_id']
    title = response['title']
    category1_nm = response['category1Name']
    category2_nm = response['category2Name']
    community_nm = response['communityName']
    href = response['href']
    # like = doc['like']
    regions = response['regions']
    salePrice = response['salePrice']
    saleStatus = response['saleStatus']
    try:
        tdview = response['tdView']
    except:
        tdview = -1

    uploadTime = response['uploadTime']
    return _id , title , category1_nm , category2_nm , community_nm , href  , regions , salePrice ,saleStatus ,  tdview , uploadTime




