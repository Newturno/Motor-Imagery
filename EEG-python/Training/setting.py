# MongoDB attributes
from pymongo import MongoClient
def get_database():
    # Provide the mongodb atlas url to connect python to mongodb using pymongo
    CONNECTION_STRING = 'mongodb+srv://newturno:newturno123@cluster0.nwci3nc.mongodb.net/'

    # Create a connection using MongoClient. You can import MongoClient or use pymongo.MongoClient
    client = MongoClient(CONNECTION_STRING)

    # Create the database for our example (we will use the same database throughout the tutorial
    return client['MI']

# This is added so that many files can reuse the function get_database()
if __name__ == "__main__":   
    # Get the database
    dbname = get_database()
    collection_name = dbname["user_data"]
    print(collection_name)
    #json file  
    item_1 = {
    "_id" : "U1IT00001",
    "item_name" : "Blender",
    "max_discount" : "10%",
    "batch_number" : "RR450020FRG",
    "price" : 340,
    "arrayX" : [[3,4,5]],
    "arrayY":[[1]]
    }

    item_2 = {
    "_id" : "U1IT00002",
    "item_name" : "Egg",
    "category" : "food",
    "quantity" : 12,
    "price" : 36,
    "array" : [[3,4,5]],
    "item_description" : "brown country eggs"
    }
    collection_name.insert_many([item_1,item_2])