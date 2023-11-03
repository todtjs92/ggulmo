import pymongo
import configparser
from datetime import datetime , timedelta
from util import get_item_parser , get_items_parser , get_newest_items_parser , get_related_items_parser
import pandas as pd
# MongoDB 접속 정보 설정

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')

    username = config['mongoDB']['username']
    password = config['mongoDB']['password']
    host = config['mongoDB']['host']
    port = config['mongoDB']['port']
    logDB = config['mongoDB']['logDB']
    logCollection = config['mongoDB']['logCollection']


    mongo_uri = f"mongodb://{username}:{password}@{host}:{port}"
    client = pymongo.MongoClient(mongo_uri)


    database = client[logDB]
    collection = database[logCollection]


    # Calculate the date for "month"
    start_time = datetime.now() - timedelta(30)
    # Query for documents where the `date` field is greater than "yesterday"
    responses = collection.find({"currentTime": {"$gt": start_time}})
    
    #responses = logCollection.find()

    count = 0

    # list for each logs
    result_get_item = []
    result_get_items = []
    result_get_related_items = []
    result_get_newest_items = []

    error_list = []
    for response in responses:
        func = response['func']

        # 모르는 api임.
        if func not in ['get_item', "get_items", "get_related_items", "get_newest_items"]:
            continue
        try:
            url = response['user_info']['url']
        except:
            print(response)
            continue

        # 클릭 로그
        if func == 'get_item':
            func, uuid, url, query, item_id, href, item_id_ret_list, cookie, current_time, upload_time  = get_item_parser(response)
            result_get_item.append([func, uuid, url, query, item_id, href, item_id_ret_list, cookie, current_time, upload_time])
        # 검색
        elif func == 'get_items':
            func, uuid, url, query, item_id, item_id_ret_list, current_time, cookie = get_items_parser(response)
            result_get_items.append([func, uuid, url, query, item_id, item_id_ret_list, current_time, cookie])

        elif func == "get_related_items":

            func, uuid, url, query, item_id, item_id_ret_list, current_time, cookie = get_related_items_parser(response)
            result_get_related_items.append([func, uuid, url, query, item_id, item_id_ret_list, current_time, cookie])

        elif func == "get_newest_items":

            func, uuid, url, query, item_id, item_id_ret_list, current_time, cookie = get_newest_items_parser(response)
            result_get_newest_items.append([func, uuid, url, query, item_id, item_id_ret_list, current_time, cookie])

        else:
            count += 1
            continue
        count += 1
        if count % 10000 == 0:
            print(count)

    client.close()

    # pandas 테이블 형태를 pickle로 저장할거임 .

    df_result_get_item = pd.DataFrame(result_get_item)
    df_result_get_item.columns = ["func", "uuid", "url", "query", "item_id", "href", "item_id_ret_list", "cookie", "current_time", "upload_time"]
    df_result_get_item.to_pickle('../data/df_get_item.pickle')


    df_result_get_items =  pd.DataFrame(result_get_items)
    df_result_get_items.columns = ["func", "uuid", "url", "query", "item_id", "item_id_ret_list", "current_time", "cookie"]
    df_result_get_items.to_pickle('../data/df_get_items.pickle')


    df_result_get_related_items =  pd.DataFrame(result_get_related_items)
    df_result_get_related_items.columns = ["func", "uuid", "url", "query", "item_id", "item_id_ret_list", "current_time", "cookie"]
    df_result_get_related_items.to_pickle('../data/df_get_related_items.pickle')


    df_result_get_newest_items =  pd.DataFrame(result_get_newest_items)
    df_result_get_newest_items.columns = ["func", "uuid", "url", "query", "item_id", "item_id_ret_list", "current_time", "cookie"]
    df_result_get_newest_items.to_pickle('../data/df_get_newest_items.pickle')