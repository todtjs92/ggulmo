import pymongo
import configparser
from datetime import datetime , timedelta
import pandas as pd
# MongoDB 접속 정보 설정

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')

    host = config['mongoDB']['host']
    port = config['mongoDB']['port']

    connection_string = f"mongodb://{host}:{port}/"
    print(connection_string)
    client = pymongo.MongoClient(connection_string)
    db_list = client.list_database_names()
    for db_name in db_list:
        print(db_name)