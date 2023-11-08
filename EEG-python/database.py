from threading import Lock
from pymongo import MongoClient

class Database:
    def __init__(self,train,label,names):
        self.train = train
        self.label = label
        self._lock = Lock()
        self.str = names
        self.db_name = 'mongodb+srv://newturno:newturno123@cluster0.nwci3nc.mongodb.net/'

    def locked_update(self, name):
        local_copy = self.train
        local_copy2 = self.str
        local_copy3 = self.label
        
        self.value = local_copy
        self.str =  local_copy2
        self.lable = local_copy3