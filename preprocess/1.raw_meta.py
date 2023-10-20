import pymongo
import configparser
from datetime import datetime , timedelta
from util import meta_parser
import pandas as pd

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')

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

    items_docs = collection.find()

    count = 0
    result_meta = []

    for response in items_docs:
        _id , title , category1_nm , category2_nm , community_nm , href  , regions , salePrice ,saleStatus ,  tdview , uploadTime = meta_parser(response)
        result_meta.append([_id , title , category1_nm , category2_nm , community_nm , href  , regions , salePrice ,saleStatus ,  tdview , uploadTime])

        count += 1
        if count % 10000 == 0:
            print(count)


    client.close()


    df_result_meta = pd.DataFrame(result_meta)
    df_result_meta.columns = ["_id" , "title" , "category1_nm" , "category2_nm" , "community_nm" , "href"  , "regions" , "salePrice" ,"saleStatus" ,  "tdview" , "uploadTime"]
    df_result_meta.to_pickle('../data/df_meta.pickle')




