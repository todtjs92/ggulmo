import pymongo
import configparser
from datetime import datetime , timedelta
from util import meta_parser
import pandas as pd
import os

if __name__ == '__main__':

    current_file_path = os.path.abspath(__file__)

    preprocess_dir_path = os.path.dirname(current_file_path)
    main_dir_path = os.path.dirname(os.path.dirname(current_file_path))
    data_dir_path = os.path.join(main_dir_path, 'data')

    print(current_file_path)
    print(data_dir_path)

    config = configparser.ConfigParser()
    config.read(preprocess_dir_path + '/config.ini')

    username = config['mongoDB']['username']
    password = config['mongoDB']['password']
    host = config['mongoDB']['host']
    port = config['mongoDB']['port']
    logDB = config['mongoDB']['logDB']
    itemCollection = config['mongoDB']['itemCollection']

    mongo_uri = f"mongodb://{username}:{password}@{host}:{port}"
    client = pymongo.MongoClient(mongo_uri)

    database = client[logDB]
    collection = database[itemCollection]

    # 30일 이상 애들만 가져옴 . 
    start_time = datetime.now() - timedelta(31)
    items_docs = collection.find()
    #items_docs = collection.find({"uploadTime": {"$gt": start_time}})
    count = 0
    result_meta = []

    for response in items_docs:
        _id , title , category1 , category2 , community_nm , href  , regions , salePrice ,saleStatus , uploadTime = meta_parser(response)
        result_meta.append([_id , title , category1, category2 , community_nm , href  , regions , salePrice ,saleStatus , uploadTime])

        count += 1
        if count % 100000 == 0:
            print(count)


    client.close()


    df_result_meta = pd.DataFrame(result_meta)
    df_result_meta.columns = ["_id" , "title" , "category1" , "category2" , "community_nm" , "href"  , "regions" , "salePrice" ,"saleStatus" , "uploadTime"]
    df_result_meta.to_pickle(data_dir_path + '/df_meta.pickle')




