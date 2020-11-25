import sys
from pymongo import MongoClient

from utils.kafka_utils import logging_time

sys.path.append('conf')
from conf.mongo_config import DATABASE_CONFIG


class MongoConnection:
    def __init__(self, collection, running_environment):
        config = DATABASE_CONFIG[running_environment]
        self.host = config['host']
        self.port = config['port']
        self.user = config['user']
        self.password = config['password']
        self.authSource = config['authSource']
        self.dbname = config['dbname']
        self.conn = None

        try:
            client = MongoClient(host=self.host,
                                port=self.port,
                                username=self.user,
                                password=self.password,
                                authSource=self.authSource)

            if collection not in client[self.dbname].list_collection_names():
                raise ConnectionError(f'"{collection}" not in DB')
            else:
                self.conn = client[self.dbname][collection]
        except Exception as e:
            print('Connection error:', e)

    def write_one(self, doc):
        if not self.conn:
            raise ConnectionError("Must connect to DB first")
        else:
            try:
                if self.conn.find_one({'piwikId': doc['piwikId']}):
                    self.conn.update_one({'piwikId': doc['piwikId']}, {'$set': {'recoResult': doc['recoResult']}})
                else:
                    self.conn.insert_one(doc)
            except Exception as e:
                # insert error
                print('Insert error:', e)

    @logging_time
    def write_many(self, docs):
        if not self.conn:
            raise ConnectionError("Must connect first")
        else:
            for doc in docs:
                try:
                    if self.conn.find_one({'piwikId': doc['piwikId']}):
                        self.conn.update_one({'piwikId': doc['piwikId']}, {'$set': {'recoResult': doc['recoResult']}})
                    else:
                        self.conn.insert_one(doc)
                except Exception as e:
                    # insert error
                    print('Insert error:', e)

    def load_one(self, piwikId):
        if not self.conn:
            raise ConnectionError("Must connect first")
        else:
            try:
                result = self.conn.find_one({'piwikId': piwikId})
                return {'piwikId': piwikId, 'recoResult': result['recoResult']}
            except Exception as e:
                # load error
                print('Load error:', e)
                return {'piwikId': piwikId, 'recoResult': None}

    def load_many(self, piwikIds):
        if not self.conn:
            raise ConnectionError("Must connect first")

        else:
            results = []
            for piwikId in piwikIds:
                try:
                    result = self.conn.find_one({'piwikId': piwikId})
                    results.append({'piwikId': piwikId, 'recoResult': result['recoResult']})
                except Exception as e:
                    # load error
                    print('Load error:', e)
                    results.append({'piwikId': piwikId, 'recoResult': None})
            
            return results

    @logging_time
    def load_all(self):
        if not self.conn:
            raise ConnectionError("Must connect first")

        else:
            results = []
            for result in self.conn.find():
                results.append({'piwikId': result['piwikId'], 'recoResult': result['recoResult']})

            return results
