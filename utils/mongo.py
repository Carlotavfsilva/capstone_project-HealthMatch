from pymongo import MongoClient

def get_health_collection(mongo_uri):
    client = MongoClient(mongo_uri)
    db = client["HealthMatch"]
    return db["Health Dictionary"]